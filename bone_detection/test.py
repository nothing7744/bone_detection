import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import argparse
import random
import time
from pathlib import Path
import os
import torchvision
from torchvision.ops.boxes import batched_nms
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import os
import sys
from typing import Iterable
import datasets
from datasets.coco_eval import CocoEvaluator
import util.misc as utils
import pdb
import util.box_ops


#两种表示方法之间的转换
# COLORS = ['blue','red','blue']
COLORS = [[0,0,255],[255,0,0],[0,0,255]]
CLASSES = [
    'B','bone_lesion', 'N'
]

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


#对于阈值的box才留下来,否则将bbox剔除
def filter_boxes(scores, boxes, confidence=0.4, apply_nms=False, iou=0):
    keep = scores.max(-1).values > confidence
    # keep=scores>confidence
    scores, boxes = scores[keep], boxes[keep]
    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]
    return scores, boxes

# def plot_results(img, prob, boxes,i):
#     fig, ax = plt.subplots(figsize=(15, 10))
#     ax.imshow(img, aspect='equal')
#     plt.figure(figsize=(15, 10))
#     plt.imshow(img)
#     ax=plt.gca()
#     for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
#         cl=p.argmax()
#         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                    fill=False, color=COLORS[cl], linewidth=2))
#         text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
#         ax.text(xmin, ymin, text, fontsize=5,
#                 bbox=dict(facecolor='yellow', alpha=0.5))
#     plt.axis('off')
#     plt.draw()
#     plt.savefig('./image/'+'result'+str(i)+'.png')
#     # plt.pause(0.0005)
#     # plt.savefig("./image/result{}.png".format(i))
#     # cv2.imwrite("./image/result{}.png".format(i),img)
#     # plt.show()
#     plt.clf()
#     plt.close()
# for output bounding box post-processing

def plot_results(img, prob, boxes,label_bboxes_scaled,i):
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        cl=p.argmax()
        if cl!=1:
            continue
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        c1, c2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        cv2.rectangle(img,c1,c2,COLORS[cl],1,cv2.LINE_AA)
        t_size = cv2.getTextSize(text, 0, fontScale=0.3, thickness=1)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1,c2,COLORS[cl],-1, cv2.LINE_AA)
        cv2.putText(img,text,(c1[0],c1[1]-2),0,0.3,[255,255,255], thickness=1, lineType=cv2.LINE_AA)
    for (x1min, y1min, x1max, y1max) in label_bboxes_scaled.tolist():
        cc1, cc2 = (int(x1min), int(y1min)), (int(x1max), int(y1max))
        cv2.rectangle(img,cc1,cc2,[255,255,0],1,cv2.LINE_AA)
    # cv2.imshow("images", img)
    cv2.waitKey()
    image = Image.fromarray(img)
    image.save(os.path.join("./image/result{}.jpg".format(i)))


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b=b*torch.tensor([img_w,img_h,img_w,img_h],dtype=torch.float32)
    return b
    
@torch.no_grad()
def myplot(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Val:'
    iou_types = tuple(k for k in ('bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    ourloss=0
    cnt=0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        mytargets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        samples,_=samples.decompose()
        samples=samples.squeeze(0)
        samples=np.array(samples.cpu())
        img=samples[0,:,:]
        # print("================3==============")
        # print(img[300:350,300:350])
        img=(img-np.min(img))/float(np.max(img))
        # print("================4==============")
        # print(img[300:350,300:350])
        img=(img*255).astype('uint8')
        # print(img.shape)
        w,h=img.shape
        img_x1=np.zeros((w,h,3),np.uint8)
        # img_x1=np.zeros((512,512,3),np.uint8)
    #得到image,然后在这个image上增加bounding box即可
        img_x1[:,:,0]=img
        img_x1[:,:,1]=img
        img_x1[:,:,2]=img
        logits=outputs['pred_logits']
        bboxes=outputs['pred_boxes']
        # print("#########################")
        # print(bboxes)
        label_bboxes=mytargets[0]["boxes"]
        # print(label_bboxes)
        bboxes_scaled = rescale_bboxes(bboxes[0,].cpu(), (w, h))
        # print(bboxes_scaled)
        label_bboxes_scaled = rescale_bboxes(label_bboxes.cpu(), (w, h))
        pre_class = logits.softmax(-1)[0, :, :].cpu()
        score, pre_box = filter_boxes(pre_class, bboxes_scaled)
        
        pre_box=pre_box.squeeze(0)
        plot_results(img_x1, score, pre_box,label_bboxes_scaled,cnt)
        loss_dict = criterion(outputs, mytargets)
        weight_dict = criterion.weight_dict
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        ourloss=ourloss+sum(loss_dict_reduced_scaled.values())
        cnt+=1
        orig_target_sizes = torch.stack([t["orig_size"] for t in mytargets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(mytargets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}