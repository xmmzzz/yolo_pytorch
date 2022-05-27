#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from net.yolo_tiny import YoloBody
from net.yoloLoss import (YOLOLoss, weights_init, get_lr_scheduler,
                          set_optimizer_lr)
from utils.mydataloader2 import (YoloDataset, yolo_dataset_collate)
from utils.my_fit import fit_one_epoch
from utils.callbacks import LossHistory
from utils.utils import get_classes


if __name__ == "__main__":
    anchors_mask = [[3, 4, 5], [1, 2, 3]]
    class_names, num_classes = get_classes("model_data/my_classes.txt")
    pretrained = False
    phi = 0

    anchors = np.array([[10.,14.],[23.,27.],[37.,58.],[81.,82.],[135.,169.],[344.,319.]])
    input_shape = [480,640]
    Cuda = True
    label_smoothing = 0

    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01

    Init_Epoch = 0
    UnFreeze_Epoch = 50
    Unfreeze_batch_size = 16

    optimizer_type = "adam"
    momentum = 0.937
    weight_decay = 5e-4

    ngpus_per_node = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0

    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    model = YoloBody(anchors_mask, num_classes, pretrained = pretrained, phi = phi)
    if not pretrained:
        weights_init(model)

    yolo_loss  = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)

    #----------------------#
    #   记录Loss
    #----------------------#
    save_period = 10
    save_dir = "./save"
    loss_history = LossHistory(save_dir, model, input_shape=input_shape)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # utils.mydataloader
    # train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, train = True)
    # val_dataset = YoloDataset(val_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, train = False)

    # utils.mydataloader2
    train_dataset = YoloDataset("temp_data/train.txt", "temp_data/obj",
                                input_shape, num_classes, epoch_length=UnFreeze_Epoch, train=True)
    val_dataset = YoloDataset("temp_data/test.txt", "temp_data/obj",
                              input_shape, num_classes, epoch_length=UnFreeze_Epoch, train=True)

    num_train = len(train_dataset)
    num_val = len(val_dataset)

    if True:
        UnFreeze_flag = False
        #-------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3
        lr_limit_min    = 3e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        #---------------------------------------#
        #   构建数据集加载器。
        #---------------------------------------#
        with open(train_annotation_path, encoding='utf-8') as f:
            train_lines = f.readlines()
        with open(val_annotation_path, encoding='utf-8') as f:
            val_lines = f.readlines()
        num_train = len(train_lines)
        num_val = len(val_lines)



        shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size,
                         pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle = shuffle, batch_size = batch_size,
                             pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch
            fit_one_epoch(model_train, model, yolo_loss, loss_history,
                          optimizer, epoch, epoch_step, epoch_step_val,
                          gen, gen_val, UnFreeze_Epoch, Cuda, save_period, save_dir)

        loss_history.writer.close()


