# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
# import
from tensorboardX import SummaryWriter
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import datasets
from datasets import build_data,get_bone_api_from_dataset
import util.misc as utils
from engine import evaluate, train_one_epoch,evaluate_plotRoc
from models import build_model
from util.box_ops import *
from models.matcher import build_matcher
import math
import os
import sys
from test import myplot

# from datasets.myDataset import *

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    # parser.add_argument('--lr', default=5e-5, type=float)
    # parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)

    parser.add_argument('--batch_size', default=32, type=int)
    # parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    # parser.add_argument('--lr_drop', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
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
    # parser.add_argument('--train_root',default='/data/kehao/myDataset/detection_data4.0/train/',type=str)
    # parser.add_argument('--val_root', default='/data/kehao/myDataset/detection_data4.0/val/', type=str)
    # parser.add_argument('--train_anno',default='./data_json4.0/train.json',type=str)
    # parser.add_argument('--val_anno', default='./data_json4.0/val.json', type=str)


    parser.add_argument('--train_root',default='/data/kehao/myDataset/detection_data5.0/train/',type=str)
    parser.add_argument('--val_root', default='/data/kehao/myDataset/detection_data5.0/val/', type=str)
    parser.add_argument('--train_anno',default='./data_json5.0/train.json',type=str)
    parser.add_argument('--val_anno', default='./data_json5.0/val.json', type=str)


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

    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")

    # parser.add_argument('--num_queries', default=100, type=int,
    #                     help="Number of query slots")

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

    parser.add_argument('--output_dir', default='/data/kehao/models/checkpoint/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./logs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
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
    model.to(device)
    model_without_ddp=model

    # print("===========================")
    # print(args.gpu)
    # print(args.distributed)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    output_dir = Path(args.output_dir)
        

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    dataset_train = build_data('train',args)
    dataset_val = build_data('val',args)

    # base_ds=dataset_val
    # print(type(base_ds.dataset))

    # print(type(dataset_val))
    base_ds = get_bone_api_from_dataset(dataset_val)

    # data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True)
    # data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True)


    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train,shuffle=False)
        # sampler_train = DistributedSampler(dataset_train,shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

#args.resume存储的是pth文件的路径
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1


    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,pin_memory=False)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,pin_memory=False)    
    
    print("Start training")
    start_time = time.time()
    # fw = open("./pth/log.txt", 'w')
    # for epoch in range(args.start_epoch, args.epochs):
    for epoch in range(args.start_epoch, args.start_epoch+1):
        # train_stats= train_one_epoch(
        #     model, criterion, data_loader_train, optimizer, device, epoch,
        #     args.clip_max_norm)  
        # lr_scheduler.step()
        # # summaryWriter.add_scalars('loss', {"loss":train_loss}, epoch)
        # if args.output_dir:
        #     checkpoint_paths = [output_dir / 'checkpoint.pth']
        #     if epoch >40 and epoch % 10 == 0:
        #         checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
        #     for checkpoint_path in checkpoint_paths:
        #         utils.save_on_master({
        #             'model': model_without_ddp.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'lr_scheduler': lr_scheduler.state_dict(),
        #             'epoch': epoch,
        #             'args': args,
        #         }, checkpoint_path)
        test_stats, coco_evaluator= evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )
        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Training time {}'.format(total_time_str))

        # summaryWriter = SummaryWriter(log_dir='logs',comment='Linear')
        # ls=list([])
        # for T in range(40,100,5):
        #     T=T*1.0/100
        #     test_stats, coco_evaluator,val_loss,recall,precision= evaluate_plotRoc(
        #     model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,T)
        #     # ls.append((recall,precision))
        #     ls.append((precision,recall))
        # fw.write(str(ls))
        # ls.sort()
        # for m in ls:
        #     (precision,recall)=m
        #     summaryWriter.add_scalars('PRcurve', {"recall":100*recall}, 1000*precision)       
        myplot(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
        # myplot(model, criterion, postprocessors, data_loader_train, base_ds, device, args.output_dir)
        # summaryWriter.add_scalars('loss', {"val_acc":val_num_correct/len(val_loader.dataset)}, epoch)
        # summaryWriter.add_scalars('loss', {"val_loss":val_loss}, epoch)
        
        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              **{f'test_{k}': v for k, v in test_stats.items()},
        #              'epoch': epoch,
        #              'n_parameters': n_parameters}
        # if args.log_dir and utils.is_main_process():
        #         fw.write(json.dumps(log_stats) + "\n")
            # 将dict类型转换为st,json.dump
    # summaryWriter.add_scalars('time', {"time":total_time_str}, epoch)
    # fw.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
