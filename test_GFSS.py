import os
import os.path as osp
import random
import datetime
import time
import cv2
import numpy as np
import logging
import argparse
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from model import BAM

from util import dataset
from util import transform, transform_tri, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, get_logger, get_save_path

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = '7'
val_manual_seed = 123
val_num = 5
setup_seed(val_manual_seed, False)
seed_array = np.random.randint(0,1000,val_num)    # seed->[0,999]


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='BAM')
    parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_vgg.yaml', help='config file') # coco/coco_split0_resnet101.yaml
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def get_model(args):
    model = eval(args.arch).OneModel(args, cls_type='Novel')
    model = model.cuda()

    # Resume
    get_save_path(args)

    if args.weight:
        weight_path = osp.join(args.snapshot_path, args.weight)
        if osp.isfile(weight_path):
            logger.info("=> loading weight '{}'".format(weight_path))
            checkpoint = torch.load(weight_path, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try: 
                model.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
        else:
            logger.info("=> no weight found at '{}'".format(weight_path))
    else:
        logger.info("=> no weight specified")

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    print('Number of Parameters: %d' % (total_number))
    print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model


def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    print(args)

    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.split in [0, 1, 2, 3, 999]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    
    logger.info("=> creating model ...")
    model = get_model(args)
    logger.info(model)

# ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    if args.resized_val:
        val_transform = transform.Compose([
            transform.Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])    
        val_transform_tri = transform_tri.Compose([
            transform_tri.Resize(size=args.val_size),
            transform_tri.ToTensor(),
            transform_tri.Normalize(mean=mean, std=std)])            
    else:
        val_transform = transform.Compose([
            transform.test_Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])           
        val_transform_tri = transform_tri.Compose([
            transform_tri.test_Resize(size=args.val_size),
            transform_tri.ToTensor(),
            transform_tri.Normalize(mean=mean, std=std)])    

    if args.data_set == 'pascal' or args.data_set == 'coco':
        val_data = dataset.GSemData(split=args.split, shot=args.shot, data_root=args.data_root, base_data_root=args.base_data_root, data_list=args.val_list, \
                            transform=val_transform, transform_tri=val_transform_tri, mode='val', ann_type=args.ann_type, \
                            data_set=args.data_set, use_split_coco=args.use_split_coco)                              
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=None)

# ----------------------  VAL  ----------------------
    start_time = time.time()
    FBIoU_array = np.zeros(val_num)
    mIoU_array_n = np.zeros(val_num)
    mIoU_array_b = np.zeros(val_num)
    mIoU_array_t = np.zeros(val_num)
    pIoU_array = np.zeros(val_num)

    for val_id in range(val_num):
        val_seed = seed_array[val_id]
        print('Val: [{}/{}] \t Seed: {}'.format(val_id+1, val_num, val_seed))
        loss_val, FBIoU, mIoU_n, mIoU_b, mIoU_t, pIoU = validate_GFSS(val_loader, model, val_seed) 

        FBIoU_array[val_id], mIoU_array_n[val_id], mIoU_array_b[val_id], mIoU_array_t[val_id], pIoU_array[val_id] = \
        FBIoU, mIoU_n, mIoU_b, mIoU_t, pIoU
    
    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    print('\nTotal running time: {}'.format(total_time))
    print('Seed0: {}'.format(val_manual_seed))
    print('Seed:  {}'.format(seed_array))
    print('mIoU_n:  {}'.format(np.round(mIoU_array_n, 4)))
    print('mIoU_b:  {}'.format(np.round(mIoU_array_b, 4)))
    print('mIoU_t:  {}'.format(np.round(mIoU_array_t, 4)))
    print('FBIoU:   {}'.format(np.round(FBIoU_array, 4)))
    print('pIoU:    {}'.format(np.round(pIoU_array, 4)))
    print('-'*43)
    print('Best_Seed_m: {} \t Best_Seed_F: {} \t Best_Seed_p: {}'.format(seed_array[mIoU_array_t.argmax()], seed_array[FBIoU_array.argmax()], seed_array[pIoU_array.argmax()]))
    print('-'*15 + ' Best ' + '-'*15)
    print('mIoU_n: {:.4f} \t  mIoU_b: {:.4f} \t mIoU_t: {:.4f}'.format(mIoU_array_n.max(), mIoU_array_b.max(), mIoU_array_t.max()))
    print('FBIoU : {:.4f} \t  pIoU:   {:.4f}'.format(FBIoU_array.max(), pIoU_array.max()))
    print('-'*15 + ' Mean ' + '-'*15)
    print('mIoU_n: {:.4f} \t  mIoU_b: {:.4f} \t mIoU_t: {:.4f}'.format(mIoU_array_n.mean(), mIoU_array_b.mean(), mIoU_array_t.mean()))
    print('FBIoU:  {:.4f} \t  pIoU:   {:.4f}'.format(FBIoU_array.mean(), pIoU_array.mean()))



def validate_GFSS(val_loader, model, val_seed):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    if args.data_set == 'pascal':
        test_num = 1000
        split_gap = 5
    elif args.data_set == 'coco':
        test_num = 1000
        split_gap = 20

    class_intersection_meter_t = [0]*split_gap*4
    class_union_meter_t = [0]*split_gap*4
    class_target_meter_t = [0]*split_gap*4

    setup_seed(val_seed, args.seed_deterministic)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model.eval()
    end = time.time()
    val_start = end

    assert test_num % args.batch_size_val == 0
    db_epoch = math.ceil(test_num/(len(val_loader)-args.batch_size_val))
    iter_num = 0
    novel_num = 0

    for e in range(db_epoch):
        for i, (input, target_t, s_input, s_mask, subcls, ori_label_t) in enumerate(val_loader):
            if iter_num * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)
            
            input = input.cuda(non_blocking=True)
            target_t = target_t.cuda(non_blocking=True)
            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)           
            ori_label_t = ori_label_t.cuda(non_blocking=True)

            novel_num +=1 if split_gap*3+1 in target_t.unique() else 0
            start_time = time.time()
            output, meta_out, base_out = model(x=input, s_x=s_input, s_y=s_mask, y_m=None, y_b=None, cat_idx=subcls)
            model_time.update(time.time() - start_time)

            if args.ori_resize:
                longerside = max(ori_label_t.size(1), ori_label_t.size(2))
                backmask = torch.ones(ori_label_t.size(0), longerside, longerside, device='cuda')*255
                backmask[0, :ori_label_t.size(1), :ori_label_t.size(2)] = ori_label_t
                target_t = backmask.clone().long()

            output = F.interpolate(output, size=target_t.size()[1:], mode='bilinear', align_corners=True)
            meta_out = F.interpolate(meta_out, size=target_t.size()[1:], mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=target_t.size()[1:], mode='bilinear', align_corners=True)
            
            base_out = base_out.max(1)[1]
            output = torch.where(output.softmax(1)[:,1]>args.merge_tau,torch.ones_like(base_out),torch.zeros_like(base_out))
            
            if args.merge == 'final':            
                merge_out = output.clone()
                merge_out[torch.where(output==1)] = split_gap*3+1
                uncertain_pix = torch.where(output == 0)
                select_mask = base_out[uncertain_pix] != 0
                select_pix = (uncertain_pix[0][select_mask], uncertain_pix[1][select_mask], uncertain_pix[2][select_mask])
                merge_out[select_pix] = base_out[select_pix]
            elif args.merge == 'base':
                merge_out = base_out.clone()
                uncertain_pix = torch.where(base_out == 0)
                select_mask = output[uncertain_pix] != 0
                select_pix = (uncertain_pix[0][select_mask], uncertain_pix[1][select_mask], uncertain_pix[2][select_mask])
                merge_out[select_pix] = split_gap*3+1

            subcls = subcls[0].cpu().numpy()[0]

            intersection, union, new_target = intersectionAndUnionGPU(merge_out, target_t, split_gap*3+2, args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
            for idx in range(1,len(intersection)-1):
                class_intersection_meter_t[idx-1] += intersection[idx]
                class_union_meter_t[idx-1] += union[idx]
                class_target_meter_t[idx-1] += new_target[idx]
            class_intersection_meter_t[split_gap*3+subcls] += intersection[-1]
            class_union_meter_t[split_gap*3+subcls] += union[-1]
            class_target_meter_t[split_gap*3+subcls] += new_target[-1]

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()

            remain_iter = test_num/args.batch_size_val - iter_num
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

            if ((i + 1) % round((test_num/100)) == 0):
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'Accuracy {accuracy:.4f}.'.format(iter_num* args.batch_size_val, test_num,
                                                            data_time=data_time,
                                                            batch_time=batch_time,
                                                            remain_time=remain_time,
                                                            accuracy=accuracy))
    val_time = time.time()-val_start

    iou_class_0 = intersection_meter.sum[0] / (union_meter.sum[0] + 1e-10)
    iou_class_1 = intersection_meter.sum[1:].sum() / (union_meter.sum[1:].sum() + 1e-10)
    mIoU = (iou_class_0+iou_class_1)/2
    
    class_iou_class = np.zeros(split_gap*4)
    class_miou_n = 0
    class_miou_b = 0
    class_miou_t = 0
    for i in range(len(class_intersection_meter_t)):
        class_iou = class_intersection_meter_t[i]/(class_union_meter_t[i]+ 1e-10)
        class_iou_class[i] = class_iou

    class_miou_n = class_iou_class[-split_gap:].sum() / split_gap
    class_miou_b = class_iou_class[:-split_gap].sum() / (split_gap*3)
    class_miou_t = class_iou_class.sum() / (split_gap*4)

    logger.info('meanIoU---Val result: mIoU_n {:.4f}.'.format(class_miou_n))   # novel
    logger.info('meanIoU---Val result: mIoU_b {:.4f}.'.format(class_miou_b))   # base
    logger.info('meanIoU---Val result: mIoU_t {:.4f}.'.format(class_miou_t))   # total (base&novel)

    logger.info('<<<<<<< Total Results <<<<<<<')
    for i in range(split_gap*4):
        if i < split_gap:
            logger.info('Class_{} Result: iou_n {:.4f}.'.format(i+1, class_iou_class[i+split_gap*3]))   
        else:
            logger.info('Class_{} Result: iou_b {:.4f}.'.format(i+1, class_iou_class[i-split_gap]))   

    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('total time: {:.4f}, avg inference time: {:.4f}, novel_count: {}, count: {}'.format(val_time, model_time.avg, novel_num, test_num))

    return loss_meter.avg, mIoU, class_miou_n, class_miou_b, class_miou_t, iou_class_1


if __name__ == '__main__':
    main()
