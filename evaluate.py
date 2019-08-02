"""
Trainer for the model.
"""
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VOCDetectionCustom, show_tensors_data
from model import TinyYOLO, YoloV3Loss
from train import load_checkpoint, find_best_model


class Evaluation():
    """
    Implements the auxiliary function to evaluate the model.
    """

    def __init__(self, model, test_dl=None):
        self.test_dl = test_dl
        self.model = model
        # check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = lr

    def evaluate(self, image):
        """
        Function to evaluate the model on an image

        Args:
            image: PIL image, Numpy array or Torch Tensor

        """
        self.model = self.model.eval().to(self.device)
        with torch.no_grad():
            # move tensors to GPU if CUDA is available
            img = img.to(self.device)
            # Check cache
            pred = self.model(img)

            # TODO

            # Calculate the batch loss
            loss = self.criterion(pred, target)
            # Update average validation loss
            valid_loss += loss.item() * bs


if __name__ == "__main__":
    cls_test = ['bicycle', 'bus', 'car', 'motorbike']
    dataset = VOCDetectionCustom(classes=cls_test)
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_len,
                                                                test_len])
    train_dl = DataLoader(train_ds,
                          batch_size=32,
                          shuffle=True)

    test_dl = DataLoader(test_ds,
                         batch_size=32,
                         shuffle=True)
    model = TinyYOLO(dataset.ANCHORS, len(dataset.classes))
    criterion = YoloV3Loss()
    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer,
                                  'min')
    best_model = find_best_model()
    model, optimizer, scheduler, loss = load_checkpoint(best_model['path'], model, optimizer, scheduler, 'cuda')
    # model, optimizer, scheduler, loss = load_checkpoint('models/Epoch_249_worse_2019-07-30_00-43-19.model', model, optimizer, scheduler, 'cuda')
    model = model.cuda()
    for i in range(3):
        img, target = next(train_dl.__iter__())
        # img, target = next(test_dl.__iter__())
        show_tensors_data(img[0], target[0])
        plt.show()
        pred = model(img[0:1].cuda())
        show_tensors_data(img[0], pred[0], thresh=.8)
        plt.show()
