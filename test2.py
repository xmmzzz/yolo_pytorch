from myYolo import YOLO
from net.yolo_tiny import YoloBody

import numpy as np
import cv2

input_shape = [480, 640]
class_names = ["boat"]


yolo = YOLO("model_data/my_classes.txt", "model_data/yolo_anchors.txt", [480,640],
            [[3,4,5],[1,2,3]], "save/best_epoch_weights.pth", 0, True, 0.25, 0.3)

image = cv2.imread("temp_data/obj/0055.jpg")
yolo.detect_heatmap(image)
# yolo.convert_to_onnx(False, "onnx_model/yolo_tiny.onnx")
# image = yolo.draw_predict(image)
#
#
#
# cv2.imshow('a', image)
# cv2.waitKey(-1)
# cv2.destroyAllWindows()

