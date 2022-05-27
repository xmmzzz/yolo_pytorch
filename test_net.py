import torch
from net.yolov4 import YoloBody
from utils.utils import get_anchors, get_classes

classes_path = "model_data/my_classes.txt"
anchors_path = "model_data/yolo_anchors.txt"

class_names, num_classes = get_classes(classes_path)
anchors, num_anchors = get_anchors(anchors_path)
anchors_mask = [[3, 4, 5], [1, 2, 3], [0, 1, 2]]

net = YoloBody(anchors_mask, num_classes)
x = torch.rand([1,3,480,640])
out = net(x)

for o in out:
    print(o.size())
