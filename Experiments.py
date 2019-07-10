#!/usr/bin/env python
# coding: utf-8

# # YOLO Lite Model - PyTorch
# Replication of the Yolo Lite Model in Pytorch.

import os
import scipy.io
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import xml
from PIL import Image
from matplotlib.patches import Rectangle, Circle
from tqdm import tqdm_notebook
import torch
from torch import nn
from torch import optim
import torch.functional as F
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import CocoDetection, VOCDetection


# # Model Architecture
#
# - 2 Anchors
# - 4 Classes


class BoundingBox():
    """
    Creates a bouding box element
    """
    def __init__(self, x, y, width, heigh):
        super().__init__()
        self.x = x
        self.y = y
        self.heigh = heigh
        self.width = width
    def iou(self, box):
        """
        Return the Intersection over union between the boxes.
        """
    @staticmethod
    def _intersection(bb1, b12):
        return


class YoloV3Loss(nn.Module):
    """
    Implements the loss described in the YOLO v3 paper.
    """
    def __init__(self, lambda_coord=5, lambda_noobj=.5,):
        super(YoloLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.MSE = nn.MSELoss()
        self.BCE = nn.BCEWithLogitsLoss()
        self.CE = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        xy_loss = self.lambda_coord * self.MSE(pred[..., 0:2], target[..., 0:2])  # xy loss
        wh_loss = self.lambda_coord * self.MSE(pred[..., 2:4], target[..., 2:4])  # wh yolo loss
        cls_loss = self.lambda_noobj * self.CE(pred[..., 5:], target[..., 5:])    # class conf loss
        obj_loss = self.lambda_noobj * self.BCE(pred[..., 4], target[..., 4])     # obj conf loss

        return xy_loss + wh_loss + cls_loss + obj_loss

# Model
class YoloLayer(nn.Module):
    """
    Network defining class.

    """
    def __init__(self, anchors=((10., 13.), (33., 23.))):
        super().__init__()
        self.anchors = anchors

    def forward(self, x):
        out_xy = torch.sigmoid(x[..., 0:2])  # xy
        # wh for each anchor
        out_wh = torch.stack([torch.exp(x[..., i, 2:4]) * torch.tensor(a) for i, a in enumerate(self.anchors)],
                             dim=3)
        out_obj = torch.sigmoid(x[..., 4:5])  # obj
        out_cls = torch.softmax(x[..., 5:], -1)  # classes
        return torch.cat([out_xy, out_wh, out_obj, out_cls], dim=-1)


class TinyYOLO(nn.Module):
    """
    Network defining class.
    """
    def __init__(self, n_anchors=2):
        super().__init__()
        self.n_anchors = n_anchors
        # Sequence of Convolution + Maxpool Layers
        self.C1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),  # sees 448x448x3 tensor
                                nn.LeakyReLU(),
                                nn.Conv2d(16, 16, 3, padding=1),
                                nn.LeakyReLU(),
                                nn.MaxPool2d(2, 2))
        self.C2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1),  # sees 224x224x16 tensor
                                nn.LeakyReLU(),
                                nn.Conv2d(32, 32, 3, padding=1),
                                nn.LeakyReLU(),
                                nn.MaxPool2d(2, 2))
        self.C3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1),  # sees 112x112x32 tensor
                                nn.LeakyReLU(),
                                nn.Conv2d(64,64, 3, padding=1),  # sees 112x112x32 tensor
                                nn.LeakyReLU(),
                                nn.MaxPool2d(2, 2))
        self.C4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),  # sees 56x56x64 tensor
                                nn.LeakyReLU(),
                                nn.MaxPool2d(2, 2))
        self.C5 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),  # sees 28x28x128 tensor
                                nn.LeakyReLU(),
                                nn.Conv2d(256, 256, 3, padding=1),  # sees 28x28x128 tensor
                                nn.LeakyReLU(),
                                nn.MaxPool2d(2, 2))
        self.C6 = nn.Sequential(nn.Conv2d(256, 256, 1),  # sees 14x14x256 tensor
                                nn.LeakyReLU())
        self.C7 = nn.Sequential(nn.Conv2d(256, 128, 1),
                                nn.LeakyReLU())
        self.C8 = nn.Sequential(nn.Conv2d(128, 9 * n_anchors, 1),
                                nn.LeakyReLU())

        self.network = nn.Sequential(self.C1, self.C2, self.C3, self.C4,
                                      self.C5, self.C7, self.C8)
        self.yolo_layer = YoloLayer()

    def forward(self, x):
        """
        Forward pass in the network.
        """
        # Sequence of Conv2D + Maxpool
        x = self.network(x)
        x = x.permute(0, 2, 3, 1)
        bs, i, j, sout = x.shape
        x = x.view(bs, i, j, self.n_anchors, sout//self.n_anchors)
        out = self.yolo_layer(x)
        return out

# TEST
    # model = TinyYOLO()
    # model


    # print(torch.softmax(torch.tensor([2,3,4,5.]),-1))


class VOCDetectionCustom(Dataset):
    """
    Creates the VOC 2012 Dataset with only certain classes and ready for YOLO format.
    """
    YEAR = 'VOC2012'
    CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse',
               'sheep', 'aeroplane', 'bicycle', 'boat', 'bus',
               'car', 'motorbike', 'train', 'bottle', 'chair',
               'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

    def __init__(self, root_dir, transform=None, target_transform=None,
                 classes='all', image_set='train'):
        """
        Args:
            root_dir: Root directory of the images.
            transform: Tranformation applied to the images.
            target_transform: Tranformation applied to the target dict.
            classes: Classes used in this dataset, a list of classes names.
                    'all': all classes are used.
            image_set: 'train', 'val' or 'trainval'.

        Return:
            (image,taret): Tuple with the image and target.s
        """
        # Attributes
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        if classes == 'all':
            self.classes = self.CLASSES
        else:
            self.classes = classes
            self.classes_id = {cls:i for i, cls in enumerate(classes)}

        # Load images
        self.images = self._get_images_list()

    def _get_images_list(self,):
        """
        List of images present in the classes used.
        """
        main_path = ['VOCdevkit', 'VOC2012', 'ImageSets', 'Main']
        main_dir = os.path.join(self.root_dir, *main_path)
        # For each class
        images = []
        for c in self.classes:
            file_path = os.path.join(main_dir, c + '_' + self.image_set +
                                     '.txt')
            with open(file_path) as f:
                files = f.readlines()
                imgs = [line.split(' ')[0] for line in files if line[-3] != '-']
            images += imgs
        return list(set(images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        if type(idx) == torch.Tensor:
            idx = idx.item()
        # Load images
        img_path = os.path.join(self.root_dir,
                                'VOCdevkit',
                                'VOC2012',
                                'JPEGImages',
                                self.images[idx] + '.jpg')
        target_path = os.path.join(self.root_dir,
                                   'VOCdevkit',
                                   'VOC2012',
                                   'Annotations',
                                   self.images[idx] + '.xml')
        img = Image.open(img_path).convert('RGB')
        # Get data as dict
        xml_doc = xml.etree.ElementTree.parse(target_path)
        root = xml_doc.getroot()
        target = {}
        target['image'] = {'name': root.find('filename').text,
                           'width': root.find('size').find('width').text,
                           'heigth': root.find('size').find('height').text}

        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in self.classes:
                objects.append({'class': name,
                                'class_id': self.classes_id[name],
                                'xmin': int(obj.find('bndbox').find('xmin').text),
                                'ymin': int(obj.find('bndbox').find('ymin').text),
                                'xmax': int(obj.find('bndbox').find('xmax').text),
                                'ymax': int(obj.find('bndbox').find('ymax').text)})
        target['objects'] = objects
        # Transforms
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        # Output
        return (img, target)

# Image Transforms
IMG_SIZE = (448, 448)
img_transform = T.Compose([T.Resize(IMG_SIZE),
                           T.ToTensor()])

# Target transform
GRID = 14
IMG_SIZE = (448, 448)

def target_transform(target, img_size=IMG_SIZE):
    """
    y = [b_x, b_y, b_w, b_h, prob, cls]
    cls = (bicycle, bus, car, motorbike)
    """
    img_w0, img_h0 = int(target['image']['width']), int(target['image']['heigth'])
    img_w, img_h = img_size
    # Ratio
    w_ratio = img_w / img_w0
    h_ratio = img_h / img_h0
    for obj in target['objects']:
        grid_h, grid_w = (img_h // GRID, img_w // GRID)
        # Bounding box data
        xmin, ymin, xmax, ymax = (obj['xmin'], obj['ymin'],
                                  obj['xmax'], obj['ymax'])
        # Center of annotation
        mid_x, mid_y = ((xmin + (xmax - xmin) / 2) * w_ratio,
                        (ymin + (ymax - ymin) / 2) * h_ratio)
        # Volume
        volume = np.zeros((GRID, GRID, 6))
        n_grid_x = int(mid_x // grid_w)
        n_grid_y = int(mid_y // grid_h)
        volume[n_grid_x, n_grid_y, 0] = (mid_x - n_grid_x * grid_w) / grid_w
        volume[n_grid_x, n_grid_y, 1] = (mid_y - n_grid_y * grid_h) / grid_h
        volume[n_grid_x, n_grid_y, 2] = (xmax - xmin) * w_ratio / grid_w
        volume[n_grid_x, n_grid_y, 3] = (ymax - ymin) * h_ratio / grid_h
        volume[n_grid_x, n_grid_y, 4] = 1
        volume[n_grid_x, n_grid_y, 5] = obj['class_id']

    return volume

# Test Data set
cls_test = ['bicycle', 'bus', 'car', 'motorbike']
ds = VOCDetectionCustom('data/pascal_voc/', classes=cls_test, transform=img_transform,
                        target_transform=target_transform)
ds_it = iter(ds)
img, target = next(ds_it)
print(target.shape)
print(img.shape)

def show_image(img, ax=None):
    """
    Show Image in the path variable.
    """
    if isinstance(img, str):
        image = Image.open(img)
    else:
        image = img
    if ax is None:
        f = plt.figure(figsize=(12, 10))
        ax = f.add_subplot(1, 1, 1, xticks=[], yticks=[])
    ax.imshow(image)
    return ax


# Check transforms
def show_tensors_data(img, target):
    img = T.ToPILImage()(img)
    img_data = np.array(img)

    img_h, img_w = img.size
    grid_h, grid_w = (img_h // GRID, img_w // GRID)
    ax = show_image(img)
    colors = {0:'r', 1:'orange',
              2:'g', 3:'k',}
    for i in range(GRID):
        for j in range(GRID):
            rel_x = i * grid_w
            rel_y = j * grid_h
            rect = Rectangle((rel_x, rel_y), grid_w, grid_h, linewidth=1,
                             edgecolor='b', facecolor='none')
            if target[i, j, 4] == 1:
                delta_x, delta_y = target[i, j, 2:4]
                x, y = target[i, j, :2]
                x, y = int(rel_x + x * grid_w), int(rel_y + y * grid_h)
                bound_rect = Rectangle((max(int(x - delta_x / 2 * grid_w), 0),
                                        max(int(y - delta_y/2 * grid_h), 0)),
                                       int(delta_x * grid_w),
                                       int(delta_y * grid_h),
                                       linewidth=3, edgecolor=colors[target[i, j, 5]], facecolor='none')

                mid_circ = Circle((x, y), 5, linewidth=3, edgecolor='b',
                                  facecolor='k')
                ax.add_patch(bound_rect)
                # ax.add_patch(mid_circ)

            ax.add_patch(rect)

    return ax

show_tensors_data(img, target)
plt.show()
