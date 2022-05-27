from torch.utils.data.dataset import Dataset
import numpy as np
from utils.utils import cvtColor, preprocess_input
from PIL import Image
import torch
import os


class YoloDataset(Dataset):
    def __init__(self, dataset_file, data_path, input_shape, num_classes, epoch_length, train):
        super(YoloDataset, self).__init__()
        self.annotation_lines = self._read_datafile(dataset_file, data_path)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.train = train
        self.epoch_now = -1
        self.length = len(self.annotation_lines)

    def _read_datafile(self, dataset_file, data_path):
        with open(dataset_file, encoding='utf-8') as f:
            train_lines = f.readlines()[:-1]
        for i in range(len(train_lines)):
            train_lines[i] = os.path.join(data_path, train_lines[i].split('/')[-1].strip('\n'))
        return train_lines

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # index = index % self.length
        image = self.get_image(self.annotation_lines[index], self.input_shape)
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = self.get_box(self.annotation_lines[index])
        return image, box

    def get_image(self, annotation_line, input_shape):
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = Image.open(annotation_line)
        # image = cvtColor(image)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        # ---------------------------------#
        #   将图像多余的部分加上灰条
        # ---------------------------------#
        image = image.resize((w, h), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)
        return image_data

    def get_box(self, annotation_line):
        with open(annotation_line.split('.')[0] + '.txt', encoding='utf-8') as f:
            boxes = f.readlines()
        outs = []
        for box in boxes:
            box = box.strip(' \n').split(' ')
            out = list(map(float, box[1:]))
            out.append(float(box[0]))
            outs.append(np.array(out))
        return np.array(outs)

# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes
