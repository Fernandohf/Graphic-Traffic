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
from train import load_checkpoint


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
        # Valid loss
        valid_loss = valid_loss / total_valid
        return valid_loss

    def save_checkpoint(self, epoch, losses, file_name, directory="models"):
        """
        Saves the current model checkpoint

        Args:
            model: Model used.
            optimizer: Optimizer used.
            Scheduler: Scheduler used.
            epoch: Epoch number.
            losses: Dict with the losses.
            file_name: name of the saved file.
            directory: directory to save models.

        """
        # Append current datetime
        file_name += '_{date:%Y-%m-%d_%H-%M-%S}.model'.format(date=datetime.datetime.now())
        directory_name = os.path.join(directory, file_name)
        # Saves the model
        checkpoint = {"model_state_dict": self.model.state_dict(),
                      "optim_state_dict": self.optimizer.state_dict(),
                      "scheduler_state_dict": self.scheduler.state_dict(),
                      "epoch": epoch,
                      "train_loss": losses["train"],
                      "valid_loss": losses["valid"]}
        # Created directory
        torch.save(checkpoint, directory_name)


def load_checkpoint(file_path, model, optimizer, scheduler, location='cpu'):
    """
    Load all info from last model.

    Args:
        file_path: Relatice/full path to file.
        model: model to load weights from
        optimizer: optimizer to load parameters from.
        scheduler: to load from.
        location: Where to load the model.

    Return:
        Dict with all the weight loaded
    """
    # Loads the model
    checkpoint = torch.load(file_path, map_location=location)

    # Load in given objects
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    losses = {}
    losses["train"] = checkpoint["train_loss"]
    losses["valid"] = checkpoint["valid_loss"]

    return {'model': model, 'optimizer': optimizer,
            'scheduler': scheduler, 'losses': losses}

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
    loaded_results = load_checkpoint('models/Epoch_3better_2019-07-26_15-04-33.model', model, optimizer, scheduler, 'cuda')

    model = loaded_results['model'].cuda()
    img, target = next(train_dl.__iter__())
    show_tensors_data(img[0], target[0])
    plt.show()
    pred = model(img[0:1].cuda())
    show_tensors_data(img[0], pred[0], thresh=.7)
    plt.show()
