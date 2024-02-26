# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import datasets
from datasets.coco_eval import CocoEvaluator
import util.misc as utils
import pdb
import util.box_ops
from util import box_ops

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    #calculate precision and recall
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        mytargets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, mytargets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Val:'
    iou_types = tuple(k for k in ('bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # num_boxes_r,num_boxes_p,Tp_r,Tp_p,ourloss,cnt,T,object_querry,T_iou=0,0,0,0,0,0,0.5,10,0.3
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        mytargets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, mytargets)
        # print("===============1=======================")
        # print(type(loss_dict))
        # print(loss_dict)
        weight_dict = criterion.weight_dict
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # print("================2======================")
        # print(type(loss_dict_reduced))
        # print(loss_dict_reduced)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        #把两个函数的loss表示出来
        # print("================3======================")
        # print(type(loss_dict_reduced_unscaled))
        # print(loss_dict_reduced_unscaled)
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        # ourloss=ourloss+sum(loss_dict_reduced_scaled.values())
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
    return stats, coco_evaluator


# @torch.no_grad()
def evaluate_plotRoc(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,T):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Val:'
    iou_types = tuple(k for k in ('bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # num_boxes_r,num_boxes_p,Tp_r,Tp_p,ourloss,cnt,T,object_querry,T_iou=0,0,0,0,0,0,0.5,10,0
    num_boxes_r,num_boxes_p,Tp_r,Tp_p,ourloss,cnt,object_querry,T_iou=0.01,0.01,0,0,0,0,10,0.5
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        mytargets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        # Tp_r,Tp_p,num_boxes_r,num_boxes_p=calculatePr(targets, mytargets,outputs,Tp_r,Tp_p,num_boxes_r, num_boxes_p,T,T_iou,device,object_querry)
        Tp_r,Tp_p,num_boxes_r,num_boxes_p=calculateRoc(targets, mytargets,outputs,Tp_r,Tp_p,num_boxes_r, num_boxes_p,T,T_iou,device,object_querry)
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
    recall=Tp_r/num_boxes_r
    precision=Tp_p/num_boxes_p
    print("Averaged stats:", metric_logger)
    print("recall: ", recall)
    print("precision", precision)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, coco_evaluator,ourloss*1.0/cnt,recall,precision


def calculatePr(targets, mytargets,outputs,Tp_r,Tp_p,num_boxes_r, num_boxes_p,T,T_iou,device,object_querry):
    pred_value=outputs['pred_logits']
    softmax=torch.nn.Softmax(dim=2)
    pred_value=softmax(pred_value)
    pred_boxes=outputs['pred_boxes']


    # num_boxes_r =num_boxes_r+int(sum(len(t["boxes"]) for t in targets))
    # print("======================")
    # print(targets)
    # print(targets["bounding_box"])
    # print(t["bounding_box"] for t in targets)

    num_boxes_r =num_boxes_r+int(sum(len(t["boxes"]) for t in targets if t["bounding_box"][0]!=0))
    num_boxes_p=num_boxes_p+int(sum(sum(pred_value[:,:,1]>T)))
    print("num_boxes_r:{:.3f}".format(num_boxes_r))
    print("num_boxes_p:{:.3f}".format(num_boxes_p))
    tgt = torch.as_tensor([len(t["boxes"]) for t in targets], device=device)
    # tgt = torch.as_tensor([len(t["boxes"]) for t in targets if t["bounding box"]!='0'], device=device)
    b=tgt.shape[0]
    for i in range(b):
        st=int(tgt[i])
        for j in range(st):
            for k in range(object_querry):
                if float(mytargets[i]["boxes"][j][0])==0 and float(mytargets[i]["boxes"][j][1])==0:
                    break
                iou,union=box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(mytargets[i]["boxes"][j]).unsqueeze(0),box_ops.box_cxcywh_to_xyxy(pred_boxes[i,k,:]).unsqueeze(0))
                iou_uni=iou*1.0/(union+0.001)
                x=float(pred_value[i,k,1])
                if iou>T_iou and x>T:
                    Tp_r+=1
                    break
                    
    for i in range(b):
        st=int(tgt[i])
        for k in range(object_querry):
            for j in range(st):
                iou,union=box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(mytargets[i]["boxes"][j]).unsqueeze(0),box_ops.box_cxcywh_to_xyxy(pred_boxes[i,k,:]).unsqueeze(0))
                iou_uni=iou*1.0/(union+0.001)
                x=float(pred_value[i,k,1])
                if iou>T_iou and x>T:
                    Tp_p+=1
                    break
    print("Tp_r:{:.3f}".format(Tp_r))
    print("Tp_p:{:.3f}".format(Tp_p))
    return Tp_r,Tp_p,num_boxes_r, num_boxes_p

def calculateRoc(targets, mytargets,outputs,Tp,Fp,num_boxes_p, num_boxes_n,T,T_iou,device,object_querry):
    pred_value=outputs['pred_logits']
    softmax=torch.nn.Softmax(dim=2)
    pred_value=softmax(pred_value)
    pred_boxes=outputs['pred_boxes']
    
    num_boxes_p =num_boxes_p+int(sum(len(t["boxes"]) for t in targets if t["bounding_box"][0]!=0))
    # num_boxes_r =num_boxes_r+int(sum(len(t["boxes"]) for t in targets if t["bounding_box"][0]!=0))
    num_boxes_n=num_boxes_n+int(sum(10-len(t["boxes"]) for t in targets if t["bounding_box"][0]!=0))
    num_boxes_n=num_boxes_n+int(sum(10 for t in targets if t["bounding_box"][0]==0))

    print("num_boxes_p:{:.3f}".format(num_boxes_p))
    print("num_boxes_n:{:.3f}".format(num_boxes_n))
    tgt = torch.as_tensor([len(t["boxes"]) for t in targets], device=device)
    b=tgt.shape[0]
    # st=0
    for i in range(b):
        st=int(tgt[i])
        for j in range(st):
            for k in range(object_querry):
                if float(mytargets[i]["boxes"][j][0])==0 and float(mytargets[i]["boxes"][j][1])==0:
                    break
                iou,union=box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(mytargets[i]["boxes"][j]).unsqueeze(0),box_ops.box_cxcywh_to_xyxy(pred_boxes[i,k,:]).unsqueeze(0))
                # iou_uni=iou*1.0/union
                x=float(pred_value[i,k,1])
                if iou>T_iou and x>T:
                    Tp+=1
                    break

    for i in range(b):
        st=int(tgt[i])
        for k in range(object_querry):
            for j in range(st):
                iou,union=box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(mytargets[i]["boxes"][j]).unsqueeze(0),box_ops.box_cxcywh_to_xyxy(pred_boxes[i,k,:]).unsqueeze(0))
                x=float(pred_value[i,k,1])
                if x>T and iou<=T_iou:
                    Fp+=1
                    break
    print("Tp:{:.3f}".format(Tp))
    print("Fp:{:.3f}".format(Fp))
    return Tp,Fp,num_boxes_p, num_boxes_n