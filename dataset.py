"""
Classes for the Datasets
"""

import os
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class VOCDetectionCustom(Dataset):
    """
    Creates the VOC 2012 Dataset with certain classes
    and target ready for YOLO format.
    """
    YEAR = 'VOC2012'
    ANCHORS = ((4., 6.), (5., 5.))  # wxh
    IMG_SIZE = (448, 448)
    DEFAULT_PATH = ['VOCdevkit', 'VOC2012', 'ImageSets', 'Main']
    CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
               'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike',
               'train', 'bottle', 'chair', 'diningtable', 'pottedplant',
               'sofa', 'tvmonitor']

    def __init__(self, root_dir="./data/pascal_voc/", i_transform=None,
                 t_transform=None, classes='all', image_set='train'):
        """
        Args:
            root_dir: Root directory of the images.
            i_transform: Tranformation applied to the images.
            t_transform: Tranformation applied to the target dict.
            classes: Classes used in this dataset, a list of classes names.
                     'all': all classes are used.
            image_set: 'train', 'val' or 'trainval'.

        Return:
            (image, target): Tuple with the image and target.
        """
        # Attributes
        self.root_dir = root_dir
        self.image_set = image_set
        self.i_transform = (i_transform if i_transform is not None
                            else T.Compose([T.Resize(self.IMG_SIZE),
                                            T.ToTensor()]))
        self.t_transform = (t_transform if t_transform is not None
                            else self.default_t_transform)
        if classes == 'all':
            self.classes = self.CLASSES
        else:
            self.classes = classes
        self.classes_id = {cls: i for i, cls in enumerate(self.classes)}

        # Load images
        self.images = self._get_images_list()

    def _get_images_list(self,):
        """
        List of images present in the classes used.
        """
        main_dir = os.path.join(self.root_dir, *self.DEFAULT_PATH)
        # For each class
        images = []
        for c in self.classes:
            file_path = os.path.join(main_dir,
                                     c + '_' + self.image_set + '.txt')
            with open(file_path) as f:
                files = f.readlines()
                imgs = [line.split(' ')[0]
                        for line in files if line[-3] != '-']
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
        xml_doc = ElementTree.parse(target_path)
        root = xml_doc.getroot()
        target = {}
        target['image'] = {'name': root.find('filename').text,
                           'width': root.find('size').find('width').text,
                           'heigth': root.find('size').find('height').text}

        obj = []
        for o in root.findall('object'):
            name = o.find('name').text
            if name in self.classes:
                obj.append({'class': name,
                            'class_id': self.classes_id[name],
                            'xmin': int(o.find('bndbox').find('xmin').text),
                            'ymin': int(o.find('bndbox').find('ymin').text),
                            'xmax': int(o.find('bndbox').find('xmax').text),
                            'ymax': int(o.find('bndbox').find('ymax').text)})
        target['objects'] = obj
        # Transforms
        img = self.i_transform(img)
        target = self.t_transform(target)

        # Output
        return (img, target)

    def _get_anchor_index(self, w, h):
        """
        Get the anchor index with highest iou in height and width
        """
        ious = np.zeros(len(self.ANCHORS))
        for i, a in enumerate(self.ANCHORS):
            bb1 = [w, h]
            bb2 = a
            ious[i] = self.wh_iou(bb1, bb2)
        return ious.argmax()

    def wh_iou(self, bb1, bb2):
        w1, h1 = bb1[0], bb1[1]
        w2, h2 = bb2[0], bb2[1]

        # Intersection area
        inter = min(w1, w2) * min(h1, h2)

        # Union Area
        union = w1 * h1 + w2 * h2 - inter

        return inter / union

    def default_t_transform(self, target, stride=32):
        """
        Transform the dictionary into the target volume.

        The volume has shape

            (BS, GRIDX, GRIDY, N_ANCHORS, 5 + N_CLASSSES)

        where the last dimension is defined as:

        [b_x, b_y, b_w, b_h, prob, one_hot_classes]

        Args:
            target: dictionary with the bounding box info.
            img_size: size of the input image.
            target: stride of the network.

        Returns:
            volume: Tranformed target.
        """
        img_w0, img_h0 = (int(target['image']['width']),
                          int(target['image']['heigth']))
        img_w, img_h = self.IMG_SIZE
        # Ratio
        grid = img_w // stride
        n_classes = len(self.classes)
        n_anchors = len(self.ANCHORS)
        w_ratio = img_w / img_w0
        h_ratio = img_h / img_h0

        volume = np.zeros((grid, grid, n_anchors, 5 + n_classes))
        for obj in target['objects']:
            grid_h, grid_w = (img_h // grid, img_w // grid)
            # Bounding box data
            xmin, ymin, xmax, ymax = (obj['xmin'], obj['ymin'],
                                      obj['xmax'], obj['ymax'])
            # Center of annotation
            mid_x, mid_y = ((xmin + (xmax - xmin) / 2) * w_ratio,
                            (ymin + (ymax - ymin) / 2) * h_ratio)
            # Volume
            i_grid_x = int(mid_x // grid_w)
            i_grid_y = int(mid_y // grid_h)
            vol_x = (mid_x - i_grid_x * grid_w) / grid_w
            vol_y = (mid_y - i_grid_y * grid_h) / grid_h
            vol_w = (xmax - xmin) * w_ratio / grid_w
            vol_h = (ymax - ymin) * h_ratio / grid_h
            i_anchor = self._get_anchor_index(vol_w, vol_h)
            vol_cls = [0] * n_classes
            vol_cls[obj['class_id']] = 1
            volume[i_grid_x, i_grid_y, i_anchor, 0] = vol_x
            volume[i_grid_x, i_grid_y, i_anchor, 1] = vol_y
            volume[i_grid_x, i_grid_y, i_anchor, 2] = vol_w
            volume[i_grid_x, i_grid_y, i_anchor, 3] = vol_h
            volume[i_grid_x, i_grid_y, i_anchor, 4] = 1
            volume[i_grid_x, i_grid_y, i_anchor, 5:] = vol_cls

        return torch.from_numpy(volume)


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
def show_tensors_data(img, target, thresh=.5):
    """
    Show the image with the network output or expected target.

    Args:
        img: input tensor image
        target: target volume or network prediction.
        thresh: threshold to filter bounding boxes
    """
    img = T.ToPILImage()(img)
    img_data = np.array(img)
    target = target.detach().cpu().numpy()
    img_h, img_w = img.size
    grid = target.shape[0]
    anchors = target.shape[-2]
    grid_h, grid_w = (img_h // grid, img_w // grid)
    ax = show_image(img)
    colors = {0: 'r', 1: 'orange',
              2: 'g', 3: 'cyan', }
    classes = {0: 'bicycle', 1: 'bus',
               2: 'car', 3: 'motorbike'}
    for i in range(grid):
        for j in range(grid):
            rel_x = i * grid_w
            rel_y = j * grid_h
            rect = Rectangle((rel_x, rel_y), grid_w, grid_h, linewidth=.2,
                             edgecolor='b', facecolor='none')
            for k in range(anchors):
                if target[i, j, k, 4] > thresh:
                    dx, dy = target[i, j, k, 2:4]
                    x, y = target[i, j, k, :2]
                    x, y = int(rel_x + x * grid_w), int(rel_y + y * grid_h)
                    idx = target[i, j, k, 5:].argmax()
                    rec_xy = (max(int(x - dx / 2 * grid_w), 0),
                              max(int(y - dy / 2 * grid_h), 0))
                    # Bounding box
                    bound_rect = Rectangle(rec_xy,
                                           int(dx * grid_w),
                                           int(dy * grid_h),
                                           linewidth=3,
                                           edgecolor=colors[idx],
                                           facecolor='none')
                    # Anchor name

                    s = 'Anchor: ' + str(k) + ', Class: ' + str(classes[idx]) + ', Confidence: ' + str(round(target[i, j, k, 5 + idx], 1))
                    ax.text(*rec_xy, s, bbox=dict(facecolor=colors[idx]))
                    ax.add_patch(bound_rect)
            ax.add_patch(rect)

    return ax

if __name__ == "__main__":
    cls_test = ['bicycle', 'bus', 'car', 'motorbike']
    ds = VOCDetectionCustom(classes=cls_test)
    iter_ds = iter(ds)
    for i in range(10):
        img, data = next(iter_ds)
        show_tensors_data(img, data)
        plt.show()
