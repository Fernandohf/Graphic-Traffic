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
model = TinyYOLO()
model


print(torch.softmax(torch.tensor([2,3,4,5.]),-1))


inp = torch.ones((1,3, 448, 448))
model(inp)

