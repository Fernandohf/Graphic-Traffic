import sys

import pytest
import torch

from model import TinyYOLO, YoloLayer


class TestYoloLayer():
    """
    Tests for YoloLayer Class
    """
    size = (1, 1, 1, 2, 9)

    def test_outputs_1(self,):
        yolo = YoloLayer()
        size = self.size
        input_ = torch.ones(size)
        out = yolo(input_)
        assert (out[..., 0].max() < 1).item() and (
            out[..., 0].min().item() > 0)

    def test_outputs_2(self,):
        yolo = YoloLayer()
        size = self.size
        input_ = torch.ones(size)
        out = yolo(input_)
        assert (out[..., 1].max() < 1).item() and (
            out[..., 1].min().item() > 0)

    def test_outputs_3(self,):
        yolo = YoloLayer()
        size = self.size
        input_ = torch.ones(size)
        out = yolo(input_)
        assert (out[..., 4].max() < 1).item() and (
            out[..., 4].min().item() > 0)

    def test_inputs_1(self,):
        yolo = YoloLayer(((12., 23.), (11., 22), (11, 12)))
        size = self.size
        input_ = torch.ones(size)
        with pytest.raises(IndexError):
            out = yolo(input_)
