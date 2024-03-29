"""
Classes to build the model and its loss
"""
import torch
from torch import nn
from dataset import VOCDetectionCustom
# Model


# TODO - Try add depthwise convolution
class ResBlock(nn.Module):
    """
    Residual block
    """
    def __init__(self):
        super().__init__()


class YoloLayer(nn.Module):
    """
    Final Layer of YOLO network, assuming input in the format:
    (BS, GRIDX, GRIDY, N_ANCHORS, 5 + N_CLASSSES)
    where the last dimension is defined as:
    [x, y, w, h, prob, [one_hot_classes]]

    Perform the following operations:
    - Sigmoid on `x` and `y`
    - Exp on w, h and multiply by anchors dimensions.
    - Sigmoid on `prob`

    OBS.: Softmax on `one_hot_classes` is not used.
          Because `CrossEntropyLoss` is used.

    Args:
        anchors: tuples of tuples of list of lists with the dimensions
                 of the anchors being used. Example:
                 ((10., 13.), (33., 23.))
    """

    def __init__(self, anchors):  # anchors yolo
        super().__init__()
        self.anchors = anchors

    def forward(self, x):
        out_xy = torch.sigmoid(x[..., 0:2])  # xy
        # wh for each anchor
        out_wh = torch.stack([torch.exp(x[..., i, 2:4]) * torch.tensor(a).to(x.device)
                              for i, a in enumerate(self.anchors)],
                             dim=-2)
        # out_obj = torch.sigmoid(x[..., 4:5])  # obj
        out_obj = torch.sigmoid(x[..., 4:5])  # obj
        # out_cls = x[..., 5:]  # classes
        # out_cls = torch.softmax(x[..., 5:], -1)  # classes
        out_cls = torch.sigmoid(x[..., 5:])  # classes
        return torch.cat([out_xy, out_wh, out_obj, out_cls], dim=-1)


class TinyYOLO(nn.Module):
    """
    YoloLite inspired class.
    Network Stride: 32

    Args:
        anchors: Anchors being used in the network.
        n_class: Number of possible classes.
    """

    def __init__(self, anchors, n_classes):
        super().__init__()
        self.n_anchors = len(anchors)
        self.n_classes = n_classes
        # Removing Features
        self.conv_1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),  # 448x448
                                    nn.LeakyReLU(),
                                    nn.Conv2d(16, 32, 3, padding=1),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(2, 2))
        self.conv_2 = nn.Sequential(nn.BatchNorm2d(32),  # 224x224
                                    nn.Conv2d(32, 32, 3, padding=1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(32, 64, 3, padding=1),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(2, 2))
        self.conv_3 = nn.Sequential(nn.BatchNorm2d(64),  # 112x112
                                    nn.Conv2d(64, 64, 3, padding=1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(64, 128, 3, padding=1),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(2, 2))
        self.conv_4 = nn.Sequential(nn.BatchNorm2d(128),
                                    nn.Conv2d(128, 128, 3, padding=1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(128, 256, 3, padding=1),  # 56x56
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(2, 2))
        self.conv_5 = nn.Sequential(nn.BatchNorm2d(256),
                                    nn.Conv2d(256, 256, 3, padding=1),  # 28x28
                                    nn.LeakyReLU(),
                                    nn.Conv2d(256, 512, 3, padding=1),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(2, 2))
        self.conv_6 = nn.Sequential(nn.BatchNorm2d(512),
                                    nn.Conv2d(512, 512, 3, padding=1),  # 14x14
                                    nn.LeakyReLU(),
                                    nn.Conv2d(512, 1024, 3, padding=1),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(2, 2))
        # 1D Convolutions
        self.conv_7 = nn.Sequential(nn.BatchNorm2d(1024),
                                    nn.Conv2d(1024, 1280, 1),  # 7x7
                                    nn.LeakyReLU())
        self.conv_8 = nn.Sequential(nn.BatchNorm2d(1280),
                                    nn.Conv2d(1280, 2048, 1),
                                    nn.LeakyReLU())
        self.conv_9 = nn.Sequential(nn.BatchNorm2d(2048),
                                    nn.Conv2d(2048, 2048, 1),
                                    nn.LeakyReLU())
        self.conv_10 = nn.Sequential(nn.BatchNorm2d(2048),
                                     nn.Conv2d(2048,
                                               (5 + n_classes) * self.n_anchors,
                                               1),
                                     nn.ReLU6())
        self.features = nn.Sequential(self.conv_1, self.conv_2, self.conv_3,
                                      self.conv_4, self.conv_5, self.conv_6,
                                      self.conv_7, self.conv_8, self.conv_9,
                                      self.conv_10)
        self.yolo_layer = YoloLayer(anchors)

    def forward(self, x):
        """
        Forward pass in the network.
        """
        # Sequence of Conv2D + Maxpool
        x = self.features(x).float()
        # Reorder dimensions
        x = x.permute(0, 3, 2, 1)  # bs, w, h, c
        batch_size, i, j, _ = x.shape
        x = x.view(batch_size, i, j, self.n_anchors, -1)
        out = self.yolo_layer(x)
        return out


class YoloV3Loss(nn.Module):
    """
    Implements the loss described in the YOLO v3 paper, the
    loss is composed of 4 components:
    - XY Coordinates Loss: Mean squared error between the bounding
                           box centers, multiplied by `lambda_coord`.
    - HW Loss: Mean squared error between the bounding
               height and width, multiplied by `lambda_coord`.
    - CLS Loss: Binary cross entropy loss between possible classes one
                hot encoded.
    """

    def __init__(self, lambda_coord=5, lambda_noobj=.5,):
        super(YoloV3Loss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.MSE = nn.MSELoss()
        self.BCE = nn.BCELoss()

    def forward(self, pred, target):
        obj_mask = target[..., 4] == 1
        noobj_mask = ~obj_mask
        pred = pred.float()
        if torch.sum(obj_mask) == 0:
            raise IndexError(
                "Image has no class. Check the image or the target volume.")
        # XY Loss
        xy_loss = (self.lambda_coord *
                   (self.MSE(pred[obj_mask][..., 0:2],
                             target[obj_mask][..., 0:2].float())))
        # HW Loss
        wh_loss = (self.lambda_coord *
                   (self.MSE(torch.sqrt(pred[obj_mask][..., 2:4]),
                             torch.sqrt(target[obj_mask][..., 2:4].float()))))
        # Object Loss
        conf = pred[..., 4] * batch_iou(pred[..., 0:4], target[..., 0:4])
        obj_loss = (self.MSE(conf[obj_mask],
                             target[obj_mask][..., 4].float()) +
                    self.lambda_noobj * self.MSE(conf[noobj_mask],
                                                 target[noobj_mask][..., 4].float()))
        # Class Loss
        cls_loss = (self.BCE(pred[obj_mask][..., 5:],
                             target[obj_mask][..., 5:].float()))

        return xy_loss + wh_loss + obj_loss + cls_loss


def convert_bbox_xy(bbx):
    """
    Convert a bounding box from x,y w, h representation to x0, y0, x1, y1.

    Args:
        bbx: bounding boxes batch with shape (BATCH, 4).

    Return:
        New bounding boxes with x0, y0, x1, y1 representation.

    """
    shape = bbx.shape
    x0 = (bbx[..., 0] - bbx[..., 2] / 2)
    x1 = (bbx[..., 0] + bbx[..., 2] / 2)
    y0 = (bbx[..., 1] - bbx[..., 3] / 2)
    y1 = (bbx[..., 1] + bbx[..., 3] / 2)

    combined = torch.stack([x0, y0, x1, y1], dim=-2).transpose_(-1, -2)

    return combined.float()


def batch_iou(batch_bb1, batch_bb2, epsilon=1e-16):
    """
    TODO
    """
    # Convert to xy representation
    bb1_xy = convert_bbox_xy(batch_bb1.float())
    bb2_xy = convert_bbox_xy(batch_bb2.float())

    # Calculate the left/right intersection coords
    x0_i = torch.max(bb1_xy[..., 0], bb2_xy[..., 0])
    y0_i = torch.max(bb1_xy[..., 1], bb2_xy[..., 1])
    x1_i = torch.min(bb1_xy[..., 2], bb2_xy[..., 2])
    y1_i = torch.min(bb1_xy[..., 3], bb2_xy[..., 3])

    inter_area = torch.max((x1_i - x0_i) * (y1_i - y0_i), torch.tensor(0.).to(x0_i.device))
    union_area = (torch.abs((bb1_xy[..., 2] - bb1_xy[..., 0]) * (bb1_xy[..., 3] - bb1_xy[..., 1])) +
                  torch.abs((bb2_xy[..., 2] - bb2_xy[..., 0]) * (bb2_xy[..., 3] - bb2_xy[..., 1])) -
                  inter_area)
    return inter_area / (union_area + epsilon)


if __name__ == '__main__':
    criteria = YoloV3Loss()
    ANCHORS = ((4., 6.), (5., 5.))  # wxh
    model = TinyYOLO(ANCHORS, 3)
    p = torch.tensor([[[.5, .2, 1, 2., 1, .9, .1],
                       [3, 3, 2, 2., 1, .5, .5],
                       [3, 3, 2, 2., 0, .5, .5]]])
    t1 = torch.tensor([[[.5, .2, .5, 1.5, 1, 1, 0],
                        [1, 4, 1, 2., 1, .5, .5],
                        [3, 3, 2, 2., 0, .5, .5]]])
    crit = YoloV3Loss()
    out1 = crit(p, t1)
