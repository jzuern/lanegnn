import sys
from collections import OrderedDict
from functools import partial

import torch.nn as nn
import torch

from ._util import try_index

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class ABN(nn.Sequential):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, activation=nn.ReLU(inplace=True), **kwargs):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        activation : nn.Module
            Module used as an activation function.
        kwargs
            All other arguments are forwarded to the `BatchNorm2d` constructor.
        """
        super(ABN, self).__init__(OrderedDict([
            ("bn", nn.BatchNorm2d(num_features, **kwargs)),
            ("act", activation)
        ]))


class IdentityResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_act=ABN,
                 dropout=None):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps. Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False,
                                    dilation=dilation)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                    dilation=dilation))
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                    groups=groups, dilation=dilation)),
                ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)

        return out



class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 12, 24, 36], kernels=[1, 3, 3, 3]):
        super(ASPP, self).__init__()
        self.conv1 = self.createConv(in_channels, 256, kernels[0], rates[0])
        self.conv2 = self.createConv(in_channels, 256, kernels[1], rates[1])
        self.conv3 = self.createConv(in_channels, 256, kernels[2], rates[2])
        self.conv4 = self.createConv(in_channels, 256, kernels[3], rates[3])

        # Adaptive average pooling does not work when changing input resolutions at test time
        self.downsampling = GlobalAvgPool2d()

        self.conv5 = self.createConv(in_channels, 256, 1, 1)

        self.upsample = nn.UpsamplingBilinear2d(size=(72, 128))

        self.fuse_conv = self.createConv(256*5, 256, 1, 1)
        self.final_conv = self.createConv(256, out_channels, 1, 1)

    def createConv(self, in_channels, out_channels, kernel_size, dil_rate):
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0

        if dil_rate != 1:
            padding = dil_rate

        layers = [("conv1", nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False, dilation=dil_rate)),
                  ("bn1", ABN(out_channels))]
        return nn.Sequential(OrderedDict(layers))

    def forward(self, in_channels):
        x1 = self.conv1(in_channels)
        x2 = self.conv2(in_channels)
        x3 = self.conv3(in_channels)
        x4 = self.conv4(in_channels)

        x5 = self.downsampling(in_channels).view((in_channels.size(0), in_channels.size(1), 1, 1))
        x5 = self.conv5(x5)
        x5 = torch.nn.functional.upsample_bilinear(x5, size=(x4.size(2), x4.size(3)))

        fusion = self.fuse_conv(torch.cat((x1, x2, x3, x4, x5), 1))
        out = self.final_conv(fusion)

        return out

def shared_cat(in_share, in_main):
    num_main = in_main.size(1)
    num_share = in_share.size(1)
    print("Main number: %d Share Number: %d" % (num_main, num_share))
    concat = torch.zeros((in_main.size(0), num_main * num_share + num_main, in_main.size(2), in_main.size(3)))

    for i in range(num_main):
        main_slice = in_main[:, i, :, :].contiguous()
        main_slice = main_slice.view((main_slice.size(0), 1, main_slice.size(1), main_slice.size(2)))
        implant = torch.cat((main_slice, in_share), 1)
        concat[:, i*(num_share+1):(i+1)*(num_share+1), :, :] = implant
    return concat


def fuseModule(channel_in, channel_out, norm_act):
    layers = [("conv1", nn.Conv2d(channel_in, channel_in, 3, stride=1, padding=1, bias=False)),
              ("bn1", norm_act(channel_in)),
              ("conv2", nn.Conv2d(channel_in, channel_in, 3, stride=1, padding=1, bias=False)),
              ("bn2", norm_act(channel_in)),
              ("conv3", nn.Conv2d(channel_in, channel_out, 3, stride=1, padding=1, bias=False)),
              ("up", nn.ConvTranspose2d(channel_out, channel_out, 8, 4, 2, bias=False))
              ]
    return nn.Sequential(OrderedDict(layers))


def convolveUpscale2(channel_in, channel_out, upscale_factor, kernel_size):
    layers = [("conv1", nn.Conv2d(channel_in, channel_out, kernel_size, stride=1, padding=0, bias=False)),
              ("bn1", nn.BatchNorm2d(channel_out))
              ]
    if upscale_factor > 1:
        layers.append(("upsample", nn.ConvTranspose2d(channel_out, channel_out, int(upscale_factor*2), upscale_factor, int(upscale_factor/2), bias=False)))

    return nn.Sequential(OrderedDict(layers))

def convolveDownsample(channel_in, channel_out, size, kernel_size):
    layers = [("conv1", nn.Conv2d(channel_in, channel_out, kernel_size, stride=1, padding=0, bias=False)),
              ("bn1", nn.BatchNorm2d(channel_out)),
              ("downsample", nn.AdaptiveAvgPool2d(size))
              ]

    return nn.Sequential(OrderedDict(layers))


def convolveUpscale(channel_in, channel_out, upscale_factor, kernel_size, norm_act):
    layers = [("conv1", nn.Conv2d(channel_in, channel_out, kernel_size, stride=1, padding=0, bias=False)),
              ("bn1", norm_act(channel_out)),
              ("upsample", nn.ConvTranspose2d(channel_out, channel_out, int(upscale_factor*2), upscale_factor, int(upscale_factor/2), bias=False))
              ]
    return nn.Sequential(OrderedDict(layers))

class ResNeXt(nn.Module):
    def __init__(self,
                 structure,
                 groups=64,
                 norm_act=ABN,
                 input_3x3=False,
                 classes=12,
                 dilation=[1, 1, 2, 4],
                 base_channels=(128, 128, 256)):
        """Pre-activation (identity mapping) ResNeXt model

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the four modules of the network.
        groups : int
            Number of groups in each ResNeXt block
        norm_act : callable
            Function to create normalization / activation Module.
        input_3x3 : bool
            If `True` use three `3x3` convolutions in the input module instead of a single `7x7` one.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
            of the network.
        dilation : list of list of int or list of int or int
            List of dilation factors, or `1` to ignore dilation. For each module, if a single value is given it is
            used for all its blocks, otherwise this expects a value for each block.
        base_channels : list of int
            Channels in the blocks of the first residual module. Each following module will multiply these values by 2.
        """
        super(ResNeXt, self).__init__()
        self.structure = structure

        if len(structure) != 4:
            raise ValueError("Expected a structure with four values")
        if dilation != 1 and len(dilation) != 4:
            raise ValueError("If dilation is not 1 it must contain four values")

        # Initial layers
        if input_3x3:
            layers = [
                ("conv1", nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
                ("bn1", norm_act(64)),
                ("conv2", nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ("bn2", norm_act(64)),
                ("conv3", nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ("pool", nn.MaxPool2d(3, stride=2, padding=1))
            ]
        else:
            layers = [
                ("conv1", nn.Conv2d(3, 64, 7, stride=1, padding=3, bias=False)),
                ("pool", nn.MaxPool2d(3, stride=2, padding=1))
            ]
        self.mod1 = nn.Sequential(OrderedDict(layers))

        # Groups of residual blocks
        in_channels = 64
        channels = base_channels
        repetitions = [1, 1, 1, 1]

        for mod_id, num in enumerate(structure):
            # Create blocks for module

            # Repeat certain modules for double branches
            in_channels_copy = in_channels
            for repeat_ct in range(repetitions[mod_id]):
                blocks = []
                in_channels = in_channels_copy
                if repeat_ct == 1:
                    in_channels += 256
                print(in_channels)
                for block_id in range(num):
                    s, d = self._stride_dilation(mod_id, block_id, dilation)
                    blocks.append((
                        "block%d" % (block_id + 1),
                        IdentityResidualBlock(in_channels, channels, stride=s, norm_act=norm_act, groups=groups,
                                              dilation=d)
                    ))

                    in_channels = channels[-1]

                # Create and add module
                print("Creating layer: mod%d_%d" % (mod_id + 2, repeat_ct + 1))
                self.add_module("mod%d_%d" % (mod_id + 2, repeat_ct + 1), nn.Sequential(OrderedDict(blocks)))

            # Update channels
            channels = [c * 2 for c in channels]
            print(channels)

        # Pooling and predictor
        self.bn_out_1 = norm_act(in_channels)
        self.bn_out_2 = norm_act(in_channels)

        self.aspp = ASPP(in_channels, classes)

        # Upscaling of convolved earlier stages
        self.up_seg_2 = nn.ConvTranspose2d(classes, classes, 4, 2, 1, bias=False)
        self.up_borders_3 = nn.ConvTranspose2d(5, 5, 4, 2, 1, bias=False)
        self.up_inst_2 = nn.ConvTranspose2d(5, 5, 4, 2, 1, bias=False)

        self.fuse_seg = fuseModule(256+classes, classes, norm_act)
        self.fuse_border_2 = fuseModule(256 + 5, 5, norm_act)
        self.fuse_regborder = fuseModule(256 + 5, 5, norm_act)

        # Loss helper
        self.logsoft = nn.LogSoftmax()

    def forward(self, img):

        out = self.mod1(img)  # Stage 1

        out_4 = self.mod2_1(out)  # Stage 2

        out = self.mod3_1(out_4)  # Stage 3

        out = self.mod4_1(out)  # Stage 4

        # Segmentation
        seg = self.mod5_1(out)
        seg = self.bn_out_1(seg)  # Seg Stage 5

        # Segmentation Backend
        seg = self.aspp(seg)

        seg = self.up_seg_2(seg)

        seg = self.fuse_seg(torch.cat((seg, out_4), dim=1))

        return seg

    @staticmethod
    def _stride_dilation(mod_id, block_id, dilation):
        if dilation == 1:
            s = 2 if mod_id > 0 and block_id == 0 else 1
            d = 1
        else:
            if dilation[mod_id] == 1:
                s = 2 if mod_id > 0 and block_id == 0 else 1
                d = 1
            else:
                s = 1
                d = try_index(dilation[mod_id], block_id)
        return s, d


_NETS = {
    "50": {"structure": [3, 4, 6, 3]},
    "101": {"structure": [3, 4, 23, 3]},
    "152": {"structure": [3, 8, 36, 3]},
}

__all__ = []
for name, params in _NETS.items():
    net_name = "net_resnext" + name
    setattr(sys.modules[__name__], net_name, partial(ResNeXt, **params))
    __all__.append(net_name)
