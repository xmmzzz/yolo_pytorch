from utils.mydataloader import YoloDataset, yolo_dataset_collate
from torch.utils.data import DataLoader

from net.yolo_tiny import YoloBody
from net.yoloLoss import (YOLOLoss, weights_init)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import get_anchors, get_classes
import torch.backends.cudnn as cudnn


train_annotation_path   = '2007_train.txt'
val_annotation_path     = '2007_val.txt'

with open(train_annotation_path, encoding='utf-8') as f:
    train_lines = f.readlines()
with open(val_annotation_path, encoding='utf-8') as f:
    val_lines = f.readlines()
num_train = len(train_lines)
num_val = len(val_lines)
UnFreeze_Epoch = 300

anchors_mask = [[3, 4, 5], [1, 2, 3]]
class_names, num_classes = get_classes("model_data/voc_classes.txt")
anchors, num_anchors  = get_anchors("model_data/yolo_anchors.txt")
input_shape = [640,480]

train_dataset  = YoloDataset(train_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, train = True)

gen = DataLoader(train_dataset, shuffle = False, batch_size = 1, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)

pretrained = False
phi = 0
Cuda = True
label_smoothing = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_rank = 0
model = YoloBody(anchors_mask, num_classes, pretrained = False, phi = 0)
weights_init(model)
yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)

Init_lr = 1e-2
Min_lr = Init_lr * 0.01
Unfreeze_batch_size = 16
optimizer_type = "adam"
momentum = 0.937
weight_decay = 5e-4
ngpus_per_node = torch.cuda.device_count()
nbs = 64
batch_size = 32
lr_limit_max = 1e-3
lr_limit_min = 3e-4
Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max) #均取后两者的限制中的第一位值
Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
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


model_train = model.train()
if Cuda:
    model_train = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model_train = model_train.cuda()


for iteration, batch in enumerate(gen):
    images, targets = batch[0], batch[1]
    with torch.no_grad():
        if Cuda:
            images = images.cuda()
    optimizer.zero_grad()
    outputs = model_train(images)
    loss_value_all = 0
    for l in range(len(outputs)):
        loss_item = yolo_loss(l, outputs[l], targets)
        loss_value_all += loss_item
        break
    loss_value = loss_value_all
    break


