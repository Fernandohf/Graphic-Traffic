import pytest
import torch

from dataset import VOCDetectionCustom


class TestVocDataset():
    """
    Tests for CustomVOCDataset Class
    """

    def test_outputs_1(self,):
        ds = VOCDetectionCustom()
        iter_ds = iter(ds)
        img, target = next(iter_ds)

        assert (img.shape == (3, 448, 448) and
                target.shape == (14, 14, 2, 25))

    def test_outputs_2(self,):
        cls_test = ['bicycle', 'bus', 'car', 'motorbike']
        ds = VOCDetectionCustom(classes=cls_test)
        iter_ds = iter(ds)
        img, target = next(iter_ds)

        assert (img.shape == (3, 448, 448) and
                target.shape == (14, 14, 2, 9))

    def test_outputs_4(self,):
        cls_test = ['bus', 'car']
        ds = VOCDetectionCustom()
        iter_ds = iter(ds)
        img, target = next(iter_ds)
        assert (target[..., 4].max() == 1)

    def test_outputs_5(self,):
        cls_test = ['bus', 'car']
        ds = VOCDetectionCustom()
        iter_ds = iter(ds)
        img, target = next(iter_ds)
        assert (target[..., 4].max() == 1)

    def test_errors_1(self,):
        cls_test = ['xxx', 'bus', 'car', 'motorbike']
        with pytest.raises(FileNotFoundError):
            ds = VOCDetectionCustom(classes=cls_test)

    def test_errors_2(self,):
        with pytest.raises(FileNotFoundError):
            ds = VOCDetectionCustom('/ddata')
