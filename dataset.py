"""
Classes for the Datasets
"""

import os
import xml

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class VOCDetectionCustom(Dataset):
    """
    Creates the VOC 2012 Dataset with certain classes
    and target ready for YOLO format.
    """
    YEAR = 'VOC2012'
    ANCHORS = ((10., 13.), (33., 23.))
    CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse',
               'sheep', 'aeroplane', 'bicycle', 'boat', 'bus',
               'car', 'motorbike', 'train', 'bottle', 'chair',
               'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

    def __init__(self, root_dir, i_transform=None, t_transform=None,
                 classes='all', image_set='train'):
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
                            else self.default_i_transform)
        self.t_transform = (t_transform if t_transform is not None
                            else self.default_t_transform)
        if classes == 'all':
            self.classes = self.CLASSES
        else:
            self.classes = classes
            self.classes_id = {cls: i for i, cls in enumerate(classes)}

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
        xml_doc = xml.etree.ElementTree.parse(target_path)
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
                            'xmin': int(obj.find('bndbox').find('xmin').text),
                            'ymin': int(obj.find('bndbox').find('ymin').text),
                            'xmax': int(obj.find('bndbox').find('xmax').text),
                            'ymax': int(obj.find('bndbox').find('ymax').text)})
        target['objects'] = objects
        # Transforms
        img = self.i_transform(img)
        target = self.t_transform(target)

        # Output
        return (img, target)

    def _calc_iou(self,):
        # TODO
        pass

    def default_i_transform(self, img):
        # TODO
        pass

    def default_t_transform(self, target, img_size=IMG_SIZE, grid=GRID):
        """
        y = [b_x, b_y, b_w, b_h, prob, cls]
        cls = (bicycle, bus, car, motorbike)
        """
        img_w0, img_h0 = int(target['image']['width']), int(
            target['image']['heigth'])
        img_w, img_h = img_size
        # Ratio
        w_ratio = img_w / img_w0
        h_ratio = img_h / img_h0

        volume = np.zeros((grid, grid, 6))
        for obj in target['objects']:
            print('aa')
            grid_h, grid_w = (img_h // grid, img_w // grid)
            # Bounding box data
            xmin, ymin, xmax, ymax = (obj['xmin'], obj['ymin'],
                                      obj['xmax'], obj['ymax'])
            # Center of annotation
            mid_x, mid_y = ((xmin + (xmax - xmin) / 2) * w_ratio,
                            (ymin + (ymax - ymin) / 2) * h_ratio)
            # Volume
            n_grid_x = int(mid_x // grid_w)
            n_grid_y = int(mid_y // grid_h)
            volume[n_grid_x, n_grid_y, 0] = (
                mid_x - n_grid_x * grid_w) / grid_w
            volume[n_grid_x, n_grid_y, 1] = (
                mid_y - n_grid_y * grid_h) / grid_h
            volume[n_grid_x, n_grid_y, 2] = (xmax - xmin) * w_ratio / grid_w
            volume[n_grid_x, n_grid_y, 3] = (ymax - ymin) * h_ratio / grid_h
            volume[n_grid_x, n_grid_y, 4] = 1
            volume[n_grid_x, n_grid_y, 5] = obj['class_id']

        return volume


# Target transform
GRID = 14
IMG_SIZE = (448, 448)
