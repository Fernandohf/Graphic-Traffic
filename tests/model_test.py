import pytest
import torch

from model import TinyYOLO, YoloLayer, YoloV3Loss


class TestYoloLayer():
    """
    Tests for YoloLayer Class
    """
    size = (1, 1, 1, 2, 9)
    anchors = ((10., 13.), (33., 23.))

    def test_outputs_1(self,):
        yolo = YoloLayer(self.anchors)
        size = self.size
        input_ = torch.ones(size)
        out = yolo(input_)
        assert (out[..., 0].max().item() < 1) and (
            out[..., 0].min().item() > 0)

    def test_outputs_2(self,):
        yolo = YoloLayer(self.anchors)
        size = self.size
        input_ = torch.ones(size)
        out = yolo(input_)
        assert (out[..., 1].max().item() < 1) and (
            out[..., 1].min().item() > 0)

    def test_outputs_3(self,):
        yolo = YoloLayer(self.anchors)
        size = self.size
        input_ = torch.ones(size)
        out = yolo(input_)
        assert (out[..., 4].max().item() < 1) and (
            out[..., 4].min().item() > 0)

    def test_inputs_1(self,):
        yolo = YoloLayer(((12., 23.), (11., 22), (11, 12)))
        size = self.size
        input_ = torch.ones(size)
        with pytest.raises(IndexError):
            out = yolo(input_)


class TestTinyYolo():
    """
    Tests for TinyYolo Class
    """

    def test_outputs_1(self,):
        model = TinyYOLO()
        inp_shape = (1, 3, 448, 448)
        inp = torch.ones(inp_shape)
        out = model(inp)
        assert ((inp_shape[0], inp_shape[2] // 32, inp_shape[3] // 32,
                 model.n_anchors, 5 + model.n_classes) == out.shape)

    def test_outputs_2(self,):
        model = TinyYOLO(((10., 13.), (33., 23.), (21,  11.)), 5)
        inp_shape = (1, 3, 448, 448)
        inp = torch.ones(inp_shape)
        out = model(inp)
        assert ((inp_shape[0], inp_shape[2] // 32, inp_shape[3] // 32,
                 model.n_anchors, 5 + model.n_classes) == out.shape)

    def test_outputs_3(self,):
        model = TinyYOLO()
        inp_shape = (1, 3, 448, 448)
        inp = torch.ones(inp_shape)
        out = model(inp)
        assert (out[..., 0].max().item() < 1) and (
            out[..., 0].min().item() > 0)

    def test_outputs_4(self,):
        model = TinyYOLO()
        inp_shape = (1, 3, 448, 448)
        inp = torch.ones(inp_shape)
        out = model(inp)
        assert (out[..., 1].max().item() < 1) and (
            out[..., 1].min().item() > 0)

    def test_outputs_5(self,):
        model = TinyYOLO()
        inp_shape = (1, 3, 448, 448)
        inp = torch.ones(inp_shape)
        out = model(inp)
        assert (out[..., 4].max().item() < 1) and (
            out[..., 4].min().item() > 0)


class TestYoloLoss():
    """
    Tests for YoloLoss Class
    """

    def test_outputs_1(self,):
        p = torch.tensor([[[.5, .2, 1, 2., 1, .9, .1],
                           [.1, .4, 2, 3., 0, .5, .5]]])
        t1 = torch.tensor([[[.5, .2, .5, 1.5, 1, 1, 0],
                            [55, 25, 5, 25., 0, 0, 0]]])
        t2 = torch.tensor([[[.5, .2, .5, 1.5, 1, 1, 0],
                            [.1, .4, 2, 3., 0, 0, 0]]])
        crit = YoloV3Loss()
        out1 = crit(p, t1)
        out2 = crit(p, t2)
        assert (out1 == out2)

    def test_outputs_2(self,):
        p = torch.tensor([[[.5, .2, 1, 2., 1, .9, .1],
                           [.1, .4, 2, 3., 0, .5, .5]]])
        t = torch.tensor([[[.5, .2, .5, 5, 0, 1, 0],
                           [.1, .4, 2, 3., 0, 0, 0]]])
        crit = YoloV3Loss()
        with pytest.raises(IndexError):
            out1 = crit(p, t)

    def test_outputs_3(self,):
        p1 = torch.tensor([[[.5, .2, 1, 2., .8, .9, .1],
                            [55, 25, 5, 25., 0, 0, 0]]])
        p2 = torch.tensor([[[.5, .2, .5, 1.5, .5, 1, 0],
                            [.2, .5, 2.5, 3.5, 0, .5, .5]]])
        t = torch.tensor([[[.5, .2, .5, 1.5, 1, 1, 0],
                           [.1, .4, 2., 3., 1, 0, 0]]])
        crit = YoloV3Loss()
        out1 = crit(p1, t)
        out2 = crit(p2, t)
        assert (out1 > out2)

    def test_outputs_4(self,):
        p1 = torch.tensor([[[.5, .2, .5, 1.5, 1, 1, 0],
                            [.1, .4, 2., 3., 1, 0, 1]]])
        p2 = torch.tensor([[[.5, .2, .5, 1.5, 1, 0, 1],
                            [.1, .4, 2., 3., 1, 1, 0]]])
        t = torch.tensor([[[.5, .2, .5, 1.5, 1, 1, 0],
                           [.1, .4, 2., 3., 1, 0, 1]]])
        crit = YoloV3Loss()
        out1 = crit(p1, t)
        out2 = crit(p2, t)
        assert (out1 < out2)
