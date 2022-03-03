import os
import datetime
import random
import time
import cv2
import numpy as np
import logging
import argparse
import math
from visdom import Visdom
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from tensorboardX import SummaryWriter

from model import BAM

from util import dataset
from util import transform, transform_tri, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, get_logger, get_save_path, \
                                    is_same_model, fix_bn, sum_list, check_makedirs

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = '9'

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='BAM') # 
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_vgg.yaml', help='config file') # coco/coco_split0_resnet50.yaml
    parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')    
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_model(args):

    model = eval(args.arch).OneModel(args, cls_type='Base')
    optimizer = model.get_optim(model, args, LR=args.base_lr)

    if hasattr(model,'freeze_modules'):
        model.freeze_modules(model)

    if args.distributed:
        # Initialize Process Group
        dist.init_process_group(backend='nccl')
        print('args.local_rank: ', args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        model = model.cuda()

    # Resume
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    if args.resume:
        resume_path = osp.join(args.snapshot_path, args.resume)
        if os.path.isfile(resume_path):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try: 
                model.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            if main_process():       
                logger.info("=> no checkpoint found at '{}'".format(resume_path))

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    if main_process():
        print('Number of Parameters: %d' % (total_number))
        print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer

def main_process():
    return not args.distributed or (args.distributed and (args.local_rank == 0))

def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    args.distributed = True if torch.cuda.device_count() > 1 else False
    if main_process():
        print(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    
    if main_process():
        logger.info("=> creating model ...")
    model, optimizer = get_model(args)
    if main_process():
        logger.info(model)
    if main_process() and args.viz:
        writer = SummaryWriter(args.result_path)

# ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    # Train
    train_transform = transform.Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    train_transform_tri = transform_tri.Compose([
        transform_tri.RandScale([args.scale_min, args.scale_max]),
        transform_tri.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform_tri.RandomGaussianBlur(),
        transform_tri.RandomHorizontalFlip(),
        transform_tri.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform_tri.ToTensor(),
        transform_tri.Normalize(mean=mean, std=std)])
    if args.data_set == 'pascal' or args.data_set == 'coco':
        train_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, base_data_root=args.base_data_root, data_list=args.train_list, \
                                    transform=train_transform, transform_tri=train_transform_tri, mode='train', \
                                    data_set=args.data_set, use_split_coco=args.use_split_coco)
    train_sampler = DistributedSampler(train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, \
                                                pin_memory=True, sampler=train_sampler, drop_last=True, \
                                                shuffle=False if args.distributed else True)
    # Val
    if args.evaluate:
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
            val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, base_data_root=args.base_data_root, data_list=args.val_list, \
                                    transform=val_transform, transform_tri=val_transform_tri, mode='val', \
                                    data_set=args.data_set, use_split_coco=args.use_split_coco)                                   
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=False, sampler=None)


# ----------------------  TRAINVAL  ----------------------
    global best_miou, best_FBiou, best_piou, best_epoch, keep_epoch, val_num
    global best_miou_m, best_miou_b, best_FBiou_m
    best_miou = 0.
    best_FBiou = 0.
    best_piou = 0.
    best_epoch = 0
    keep_epoch = 0
    val_num = 0
    best_miou_m = 0.
    best_miou_b = 0.
    best_FBiou_m = 0.

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if keep_epoch == args.stop_interval:
            break
        if args.fix_random_seed_val:
            setup_seed(args.manual_seed + epoch, args.seed_deterministic)

        epoch_log = epoch + 1
        keep_epoch += 1
        if args.distributed:
            train_sampler.set_epoch(epoch)    

        # ----------------------  TRAIN  ----------------------
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, val_loader, model, optimizer, epoch)

        if main_process() and args.viz:
            writer.add_scalar('FBIoU_train', mIoU_train, epoch_log)

        # save model for <resuming>
        if (epoch % args.save_freq == 0) and (epoch > 0) and main_process():
            filename = args.snapshot_path + '/epoch_{}.pth'.format(epoch)
            logger.info('Saving checkpoint to: ' + filename)
            if osp.exists(filename):
                os.remove(filename)            
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

        # -----------------------  VAL  -----------------------
        if args.evaluate and epoch%1==0:
            loss_val, FBIoU, FBIoU_m, mIoU, mIoU_m, mIoU_b, pIoU = validate(val_loader, model)            
            val_num += 1
            if main_process() and args.viz:
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('FBIoU_val', FBIoU, epoch_log)
                writer.add_scalar('mIoU_val', mIoU, epoch_log)
                writer.add_scalar('mIoU_val_m', mIoU_m, epoch_log)
                writer.add_scalar('mIoU_val_b', mIoU_b, epoch_log)
                writer.add_scalar('FBIoU_val_m', FBIoU_m, epoch_log)

        # save model for <testing>
            if mIoU > best_miou:
                best_miou, best_FBiou, best_piou, best_epoch = mIoU, FBIoU, pIoU, epoch
                best_miou_m, best_miou_b, best_FBiou_m = mIoU_m, mIoU_b, FBIoU_m
                keep_epoch = 0
                if args.shot == 1:
                    filename = args.snapshot_path + '/train_epoch_' + str(epoch) + '_{:.4f}'.format(best_miou) + '.pth'
                else:
                    filename = args.snapshot_path + '/train{}_epoch_'.format(args.shot) + str(epoch) + '_{:.4f}'.format(best_miou) + '.pth'
                if main_process():                    
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)   

    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    if main_process():
        print('\nEpoch: {}/{} \t Total running time: {}'.format(epoch_log, args.epochs, total_time))
        print('The number of models validated: {}'.format(val_num))            
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Final Best Result   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(args.arch + '\t Group:{} \t Best_step:{}'.format(args.split, best_epoch))
        print('mIoU:{:.4f} \t mIoU_m:{:.4f} \t mIoU_b:{:.4f}'.format(best_miou, best_miou_m, best_miou_b))
        print('FBIoU:{:.4f} \t FBIoU_m:{:.4f} \t pIoU:{:.4f}'.format(best_FBiou, best_FBiou_m, best_piou))
        print('>'*80)
        print ('%s' % datetime.datetime.now())


def train(train_loader, val_loader, model, optimizer, epoch):
    global best_miou, best_FBiou, best_piou, best_epoch, keep_epoch, val_num
    global best_miou_m, best_miou_b, best_FBiou_m
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter1 = AverageMeter()
    aux_loss_meter2 = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    model.train()
    if args.fix_bn:
        model.apply(fix_bn) # fix batchnorm

    end = time.time()
    val_time = 0.
    max_iter = args.epochs * len(train_loader)
    if main_process():
        print('Warmup: {}'.format(args.warmup))

    for i, (input, target, target_b, s_input, s_mask, subcls) in enumerate(train_loader):

        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1
        
        poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power, index_split=args.index_split, warmup=args.warmup, warmup_step=len(train_loader)//2)

        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target_b = target_b.cuda(non_blocking=True)
        
        output, main_loss, aux_loss1, aux_loss2 = model(s_x=s_input, s_y=s_mask, x=input, y_m=target, y_b=target_b, cat_idx=subcls)

        loss = main_loss + args.aux_weight1*aux_loss1 + args.aux_weight2*aux_loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0) # batch_size

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)  # allAcc
        
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter1.update(aux_loss1.item(), n)
        aux_loss_meter2.update(aux_loss2.item(), n)
        loss_meter.update(loss.item(), n)

        batch_time.update(time.time() - end - val_time)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss1 {aux_loss_meter1.val:.4f} '                        
                        'AuxLoss2 {aux_loss_meter2.val:.4f} '                        
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                        batch_time=batch_time,
                                                        data_time=data_time,
                                                        remain_time=remain_time,
                                                        main_loss_meter=main_loss_meter,
                                                        aux_loss_meter1=aux_loss_meter1,
                                                        aux_loss_meter2=aux_loss_meter2,
                                                        loss_meter=loss_meter,
                                                        accuracy=accuracy))
            if args.viz:
                writer.add_scalar('loss_train', loss_meter.val, current_iter)
                writer.add_scalar('loss_train_main', main_loss_meter.val, current_iter)
                writer.add_scalar('loss_train_aux1', aux_loss_meter1.val, current_iter)
                writer.add_scalar('loss_train_aux2', aux_loss_meter2.val, current_iter)

        # -----------------------  SubEpoch VAL  -----------------------
        if args.evaluate and args.SubEpoch_val and (args.epochs<=100 and epoch%1==0 and epoch>0) and (i==round(len(train_loader)/2)): # <if> max_epoch<=100 <do> half_epoch Val
            loss_val, FBIoU, FBIoU_m, mIoU, mIoU_m, mIoU_b, pIoU = validate(val_loader, model)
            val_num += 1 
        # save model for <testing>
            if mIoU > best_miou:
                best_miou, best_FBiou, best_piou, best_epoch = mIoU, FBIoU, pIoU, (epoch-0.5)
                best_miou_m, best_miou_b, best_FBiou_m = mIoU_m, mIoU_b, FBIoU_m
                keep_epoch = 0
                if args.shot == 1:
                    filename = args.snapshot_path + '/train_epoch_' + str(epoch-0.5) + '_{:.4f}'.format(best_miou) + '.pth'
                else:
                    filename = args.snapshot_path + '/train{}_epoch_'.format(args.shot) + str(epoch-0.5) + '_{:.4f}'.format(best_miou) + '.pth'
                if main_process():
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch-0.5, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

            model.train()
            if args.fix_bn:
                model.apply(fix_bn) # fix batchnorm            


    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))

    return main_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    intersection_meter = AverageMeter()   # final
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    intersection_meter_m = AverageMeter() # meta
    union_meter_m = AverageMeter()
    target_meter_m = AverageMeter()

    if args.data_set == 'pascal':
        test_num = 1000 # 5000
        split_gap = 5
    elif args.data_set == 'coco':
        test_num = 1000 # 20000 
        split_gap = 20

    class_intersection_meter = [0]*split_gap
    class_union_meter = [0]*split_gap  
    class_intersection_meter_m = [0]*split_gap
    class_union_meter_m = [0]*split_gap  
    class_intersection_meter_b = [0]*split_gap*3
    class_union_meter_b = [0]*split_gap*3
    class_target_meter_b = [0]*split_gap*3

    if args.manual_seed is not None and args.fix_random_seed_val:
        setup_seed(args.manual_seed, args.seed_deterministic)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model.eval()
    end = time.time()
    val_start = end

    assert test_num % args.batch_size_val == 0
    db_epoch = math.ceil(test_num/(len(val_loader)-args.batch_size_val))
    iter_num = 0

    for e in range(db_epoch):
        for i, (input, target, target_b, s_input, s_mask, subcls, ori_label, ori_label_b) in enumerate(val_loader):
            if iter_num * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)
            
            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)                 
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target_b = target_b.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)
            ori_label_b = ori_label_b.cuda(non_blocking=True)

            start_time = time.time()
            output, meta_out, base_out = model(s_x=s_input, s_y=s_mask, x=input, y_m=target, y_b=target_b, cat_idx=subcls)
            model_time.update(time.time() - start_time)

            if args.ori_resize:  # 真值转化为方形
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside, device='cuda')*255
                backmask_b = torch.ones(ori_label.size(0), longerside, longerside, device='cuda')*255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                backmask_b[0, :ori_label.size(1), :ori_label.size(2)] = ori_label_b
                target = backmask.clone().long()
                target_b = backmask_b.clone().long()

            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
            meta_out = F.interpolate(meta_out, size=target.size()[1:], mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=target.size()[1:], mode='bilinear', align_corners=True)
            
            loss = criterion(output, target)

            output = output.max(1)[1]
            meta_out = meta_out.max(1)[1]
            base_out = base_out.max(1)[1]

            subcls = subcls[0].cpu().numpy()[0]

            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
            class_intersection_meter[subcls] += intersection[1]
            class_union_meter[subcls] += union[1] 
            
            intersection, union, new_target = intersectionAndUnionGPU(meta_out, target, args.classes, args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter_m.update(intersection), union_meter_m.update(union), target_meter_m.update(new_target)
            class_intersection_meter_m[subcls] += intersection[1]
            class_union_meter_m[subcls] += union[1]

            intersection, union, new_target = intersectionAndUnionGPU(base_out, target_b, split_gap*3+1, args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            for idx in range(1,len(intersection)):
                class_intersection_meter_b[idx-1] += intersection[idx]
                class_union_meter_b[idx-1] += union[idx]
                class_target_meter_b[idx-1] += new_target[idx]

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % round((test_num/100)) == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num* args.batch_size_val, test_num,
                                                            data_time=data_time,
                                                            batch_time=batch_time,
                                                            loss_meter=loss_meter,
                                                            accuracy=accuracy))
    val_time = time.time()-val_start

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    iou_class_m = intersection_meter_m.sum / (union_meter_m.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mIoU_m = np.mean(iou_class_m)
    
    class_iou_class = []
    class_iou_class_m = []
    class_iou_class_b = []
    class_miou = 0
    class_miou_m = 0
    class_miou_b = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i]/(class_union_meter[i]+ 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
        class_iou = class_intersection_meter_m[i]/(class_union_meter_m[i]+ 1e-10)
        class_iou_class_m.append(class_iou)
        class_miou_m += class_iou
    for i in range(len(class_intersection_meter_b)):
        class_iou = class_intersection_meter_b[i]/(class_union_meter_b[i]+ 1e-10)
        class_iou_class_b.append(class_iou)
        class_miou_b += class_iou

    target_b = np.array(class_target_meter_b)

    class_miou = class_miou*1.0 / len(class_intersection_meter)
    class_miou_m = class_miou_m*1.0 / len(class_intersection_meter)
    class_miou_b = class_miou_b*1.0 / (len(class_intersection_meter_b) - len(target_b[target_b==0]))  # filter the results with GT mIoU=0

    if main_process():
        logger.info('meanIoU---Val result: mIoU_f {:.4f}.'.format(class_miou))     # final
        logger.info('meanIoU---Val result: mIoU_m {:.4f}.'.format(class_miou_m))   # meta
        logger.info('meanIoU---Val result: mIoU_b {:.4f}.'.format(class_miou_b))   # base

        logger.info('<<<<<<< Novel Results <<<<<<<')
        for i in range(split_gap):
            logger.info('Class_{} Result: iou_f {:.4f}.'.format(i+1, class_iou_class[i]))         
            logger.info('Class_{} Result: iou_m {:.4f}.'.format(i+1, class_iou_class_m[i]))   
        logger.info('<<<<<<< Base Results <<<<<<<')
        for i in range(split_gap*3):
            if class_target_meter_b[i] == 0:
                logger.info('Class_{} Result: iou_b None.'.format(i+1+split_gap))
            else:
                logger.info('Class_{} Result: iou_b {:.4f}.'.format(i+1+split_gap, class_iou_class_b[i]))

        logger.info('FBIoU---Val result: FBIoU_f {:.4f}.'.format(mIoU))
        logger.info('FBIoU---Val result: FBIoU_m {:.4f}.'.format(mIoU_m))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou_f {:.4f}.'.format(i, iou_class[i]))
            logger.info('Class_{} Result: iou_m {:.4f}.'.format(i, iou_class_m[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

        print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, test_num))

    return loss_meter.avg, mIoU, mIoU_m, class_miou, class_miou_m, class_miou_b, iou_class[1]

if __name__ == '__main__':
    main()
