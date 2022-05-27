import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from net.yolo_tiny import YoloBody
from net.yoloLoss import (YOLOLoss, weights_init, get_lr_scheduler,
                          set_optimizer_lr)
from utils.my_fit import fit_one_epoch
from utils.callbacks import LossHistory
from utils.utils import get_classes,get_anchors

if __name__ == "__main__":
    anchors_mask = [[3, 4, 5], [1, 2, 3]]
    classes_path = "model_data/my_classes.txt"
    anchors_path = "model_data/yolo_anchors.txt"

    dataloader_type = 2
    train_path = "temp_data/train.txt"
    val_path = "temp_data/test.txt"
    data_path = "temp_data/obj"

    save_period = 5
    save_dir = "./save"

    Init_Epoch = 0
    End_Epoch = 10
    batch_size = 16
    # -------------------------------#
    #   所使用的注意力机制的类型
    #   phi = 0为不使用注意力机制
    #   phi = 1为SE
    #   phi = 2为CBAM
    #   phi = 3为ECA
    # -------------------------------#
    phi = 0

    pretrained = False

    input_shape = [480, 640]
    Cuda = True

    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01

    optimizer_type = "adam"
    momentum = 0.937
    weight_decay = 5e-4

    # --------------
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = YoloBody(anchors_mask, num_classes, pretrained=pretrained, phi=phi)
    if not pretrained:
        weights_init(model)

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)

    loss_history = LossHistory(save_dir, model, input_shape=input_shape)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    if dataloader_type == 1:
        from utils.mydataloader import (YoloDataset, yolo_dataset_collate)
        # train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, train = True)
        # val_dataset = YoloDataset(val_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, train = False)
    elif dataloader_type == 2:
        from utils.mydataloader2 import (YoloDataset, yolo_dataset_collate)
        train_dataset = YoloDataset(train_path, data_path,
                                    input_shape, num_classes, epoch_length=End_Epoch, train=True)
        val_dataset = YoloDataset(val_path, data_path,
                                  input_shape, num_classes, epoch_length=End_Epoch, train=True)

    num_train = len(train_dataset)
    num_val = len(val_dataset)

    if True:
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
            'adam'  : optim.Adam(pg0, Init_lr, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(Init_lr, Min_lr, End_Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size,
                         pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size,
                             pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

        # loss_history = None
        for epoch in range(Init_Epoch, End_Epoch):
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch
            fit_one_epoch(model_train, model, yolo_loss, loss_history,
                          optimizer, epoch, epoch_step, epoch_step_val,
                          gen, gen_val, End_Epoch, Cuda, save_period, save_dir)
            # break

        loss_history.writer.close()
