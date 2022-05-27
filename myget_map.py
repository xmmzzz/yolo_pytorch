from utils.utils import get_classes
from utils.utils_map import get_map
from myYolo import YOLO

import os
from tqdm import tqdm
from PIL import Image

import xml.etree.ElementTree as ET

if __name__ == "__main__":
    map_mode = 0

    classes_path = "model_data/my_classes.txt"
    anchor_path = "model_data/yolo_anchors.txt"
    input_shape = [480, 640]
    anchor_mask = [[3,4,5],[1,2,3]]
    save_path = "save/best_epoch_weights.pth"
    phi = 0
    Cuda = True

    # --------------------------------------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x，mAP0.x的意义是什么请同学们百度一下。
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #
    #   当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样本。
    #   因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
    # --------------------------------------------------------------------------------------#
    MINOVERLAP = 0.5
    confidence = 0.001
    nms_iou = 0.5
    # ---------------------------------------------------------------------------------------------------------------#
    #   Recall和Precision不像AP是一个面积的概念，因此在门限值不同时，网络的Recall和Precision值是不同的。
    #
    #   默认情况下，本代码计算的Recall和Precision代表的是当门限值为0.5（此处定义为score_threhold）时所对应的Recall和Precision值。
    #   因为计算mAP需要获得近乎所有的预测框，上面定义的confidence不能随便更改。
    #   这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision值。
    # ---------------------------------------------------------------------------------------------------------------#
    score_threhold = 0.5
    # -------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    # -------------------------------------------------------#
    map_vis = False
    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    VOCdevkit_path = "temp_data/obj"
    # -------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    # -------------------------------------------------------#
    map_out_path = 'map_out'

    image_ids = open(os.path.join("temp_data", "test.txt")).read().strip().split()
    for i in range(len(image_ids)):
        image_ids[i] = image_ids[i].split("/")[-1]

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(classes_path, anchor_path, input_shape,
            anchor_mask, save_path, phi, Cuda, confidence, nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, image_id)
            image = Image.open(image_path)

            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id))
            yolo.get_map_txt(image_id.split('.')[0], image, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            image_id = image_id.split('.')[0]
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                with open(os.path.join(VOCdevkit_path, image_id + ".txt"), "r") as anno_f:
                    records = anno_f.readlines()
                for rec in records:
                    rec = rec.split(' ')[:-1]
                    obj_name = class_names[int(rec[0])]
                    x, y, w, h = (float(rec[1]) * input_shape[1], float(rec[2]) * input_shape[0],
                                float(rec[3]) * input_shape[1], float(rec[4]) * input_shape[0])

                    left = round(x - w / 2)
                    right = round(x + w / 2)
                    top = round(y - h / 2)
                    bottom = round(y + h / 2)

                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold=score_threhold, path=map_out_path)
        print("Get map done.")


