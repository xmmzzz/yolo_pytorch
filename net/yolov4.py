import torch
import torch.nn as nn

from net.CSPdarknet53_yolov4 import darknet53

class BasicConv(nn.Module):
    def __init__(self, filter_in, filter_out, kernel_size, stride=1):
        super(BasicConv, self).__init__()
        pad = (kernel_size - 1) // 2 if kernel_size else 0
        self.conv = nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size,
                              stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(filter_out)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
# in_filter (1*1,stride = 1) -> filter[0] (3*3, stride = 1) ->
# filter[1] (1*1,stride = 1) -> filter[0]
#---------------------------------------------------#
class threeConv(nn.Module):
    def __init__(self, filters_list, in_filters):
        super(threeConv, self).__init__()
        self.net = nn.Sequential(
            BasicConv(in_filters, filters_list[0], 1),
            BasicConv(filters_list[0], filters_list[1], 3),
            BasicConv(filters_list[1], filters_list[0], 1)
        )

    def forward(self, x):
        return self.net(x)

#---------------------------------------------------#
#   五次卷积块
# in_filter (1*1,stride = 1) -> filter[0] (3*3, stride = 1) ->
# filter[1] (1*1,stride = 1) -> filter[0] (3*3, stride = 1) ->
# filter[1] (1*1,stride = 1) -> filter[0]
#---------------------------------------------------#
class fiveConv(nn.Module):
    def __init__(self, filters_list, in_filters):
        super(fiveConv, self).__init__()
        self.net = nn.Sequential(
            BasicConv(in_filters, filters_list[0], 1),
            BasicConv(filters_list[0], filters_list[1], 3),
            BasicConv(filters_list[1], filters_list[0], 1),
            BasicConv(filters_list[0], filters_list[1], 3),
            BasicConv(filters_list[1], filters_list[0], 1),
        )

    def forward(self, x):
        return self.net(x)

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone   = darknet53(pretrained)

        self.conv1      = threeConv([512,1024],1024)
        self.SPP        = SpatialPyramidPooling()
        self.conv2      = threeConv([512,1024],2048)

        self.upsample1          = Upsample(512,256)
        self.conv_for_P4        = BasicConv(512,256,1)
        self.make_five_conv1    = fiveConv([256, 512],512)

        self.upsample2          = Upsample(256,128)
        self.conv_for_P3        = BasicConv(256,128,1)
        self.make_five_conv2    = fiveConv([128, 256],256)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head3         = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)],128)

        self.down_sample1       = BasicConv(128,256,3,stride=2)
        self.make_five_conv3    = fiveConv([256, 512],512)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head2         = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)],256)

        self.down_sample2       = BasicConv(256,512,3,stride=2)
        self.make_five_conv4    = fiveConv([512, 1024],1024)

        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        self.yolo_head1         = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)],512)


    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)

        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.conv2(P5)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        P5_upsample = self.upsample1(P5)
        # 26,26,512 -> 26,26,256
        P4 = self.conv_for_P4(x1)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P4,P5_upsample],axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv1(P4)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        P4_upsample = self.upsample2(P4)
        # 52,52,256 -> 52,52,128
        P3 = self.conv_for_P3(x2)
        # 52,52,128 + 52,52,128 -> 52,52,256
        P3 = torch.cat([P3,P4_upsample],axis=1)
        # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        P3 = self.make_five_conv2(P3)

        # 52,52,128 -> 26,26,256
        P3_downsample = self.down_sample1(P3)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P3_downsample,P4],axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv3(P4)

        # 26,26,256 -> 13,13,512
        P4_downsample = self.down_sample2(P4)
        # 13,13,512 + 13,13,512 -> 13,13,1024
        P5 = torch.cat([P4_downsample,P5],axis=1)
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.make_five_conv4(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,52,52)
        #---------------------------------------------------#
        out2 = self.yolo_head3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,26,26)
        #---------------------------------------------------#
        out1 = self.yolo_head2(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,13,13)
        #---------------------------------------------------#
        out0 = self.yolo_head1(P5)

        return out0, out1, out2
