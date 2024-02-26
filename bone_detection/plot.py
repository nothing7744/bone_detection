# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from datasets import build_data
import util.misc as utils
from engine import evaluate, train_one_epoch
from models import build_model
from util.box_ops import *
from models.matcher import build_matcher
import math
import os
import sys
from test import __plot
# from datasets.myDataset import *

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    # parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
     # * Segmentation   
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    #path
    parser.add_argument('--train_root',default='/data/kehao/myDataset/detection_data2.0/train/',type=str)
    parser.add_argument('--val_root', default='/data/kehao/myDataset/detection_data2.0/val/', type=str)
    parser.add_argument('--train_anno',default='./data_json2.0/train.json',type=str)
    parser.add_argument('--val_anno', default='./data_json2.0/val.json', type=str)

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')





    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients(这部分是添加损失函数的系数)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='bone_lessions')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./pth',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser



def plot_results(img, prob, boxes,name):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='red', linewidth=2))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig('result'+name+'.png')
    plt.clf()
    # plt.draw()
    # plt.show()

def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def main(args):
    # utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, criterion, postprocessors = build_model(args)

    state_dict = torch.load("./pth/checkpoint0099.pth")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    model_without_ddp = model

    output_dir = Path(args.output_dir)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    dataset_train = build_data('train',args)
    dataset_val = build_data('val',args)

    data_loader_train = DataLoader(dataset_train, batch_size=1, num_workers=args.num_workers,shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=1, num_workers=args.num_workers,shuffle=True)
    
    print("Start training")
    start_time = time.time()
    fw = open("/home/kehao/mydetection/logs/trlog.txt", 'w')
    #导入输入的图片,然后在输出的图片上加入bounding box得到最后的结果

    # @torch.no_grad()
    for i,batch in enumerate(data_loader_val):
        (samples,targets)=batch
        samples = samples.to(device).float()
        samples=np.array(samples)
        samples=samples.squeeze()
        img=samples[:,:,0]
        img=(img-np.min(img))/float(np.max(img))
        img=(img*255).astype('uint8')
        img_x1=np.zeros((512,512,3),np.uint8)
    #得到image,然后在这个image上增加bounding box即可
        img_x1[:,:,0]=x1
        img_x1[:,:,1]=x1
        img_x1[:,:,2]=x1

        mytargets=list()
        batch_size=samples.shape[0]

        # for i in range(batch_size):
        #     tmp={"boxes":targets["boxes"][i].squeeze(0).to(device),"labels":targets["labels"][i].to(device)}
        # mytargets.append(tmp)
        # outputs = model(samples)
        
        #需要在这里画出标注的区域
        for i in range(batch_size):
            tmp={"boxes":targets["boxes"][i].squeeze(0).to(device),"labels":targets["labels"][i].to(device)}
        mytargets.append(tmp)


        # guard against no boxes via resizing
        
        # bboxes = targets["boxes"].unsqueeze(0)
        # print("=============================")
        # print(bboxes.shape)

        outputs = model(samples)
        loss_dict = criterion(outputs, mytargets)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        matcher=build_matcher(args)
        indices=matcher(outputs_without_aux,mytargets)
        idx= _get_src_permutation_idx(indices)
        #得到两个候选框的bounding box
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'].unsqueeze(0) for t in mytargets], dim=0)
        
        giou=torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),box_cxcywh_to_xyxy(target_boxes)))
        # val_num_correct+=torch.sum(giou>0.2)
        # print("val Epoch: {}\t Acc: {:.6f}".format(epoch,val_num_correct/len(data_loader_val.dataset)))
        # fw.write("val Epoch: {}\t Acc: {:.6f}".format(epoch,val_num_correct/len(data_loader_val.dataset)))
        logits, bboxes = detr(source_image)
        # logits, bbox = detr(source_image)

#通过这两个返回的参数计算loss
# print(logits.shape)
pre_class = logits.softmax(-1)[0, :, :-1].cpu()
# pre_class = pre_class[0, :, :-1].cpu()
# print(pre_class.shape)
bboxes_scaled = rescale_bboxes(bboxes[0,].cpu(), (source_image.shape[3], source_image.shape[2]))
print("hello,world")

score, pre_box = filter_boxes(pre_class, bboxes_scaled)
# print(score.shape)
plot_results(image, score, pre_box)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
