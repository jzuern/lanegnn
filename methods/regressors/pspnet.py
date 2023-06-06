import torch
from torch import nn
from torch.nn import functional as F

import regressors.extractors as extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        # priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_scale=2):
        super().__init__()
        self.upsample_scale = upsample_scale
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = self.upsample_scale * x.size(2), self.upsample_scale * x.size(3)
        # p = F.upsample(input=x, size=(h, w), mode='bilinear')
        p = F.interpolate(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, in_channels, n_classes=14, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.feats = getattr(extractors, backend)(in_channels=in_channels, pretrained=pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
        )


    def forward(self, x):
        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        return self.final(p)


class PSPNetDoubleHead(nn.Module):

    def __init__(self, n_classes=2, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=1024, backend='resnet34',
                 pretrained=False):

        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)

        self.drop_1 = nn.Dropout2d(p=0.3)
        self.drop_2 = nn.Dropout2d(p=0.15)

        self.up_1_nodes = PSPUpsample(1024, 256)
        self.up_2_nodes = PSPUpsample(256, 64)
        self.up_3_nodes = PSPUpsample(64, 64)
        self.final_nodes = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.up_1_centerlines = PSPUpsample(1024, 256)
        self.up_2_centerlines = PSPUpsample(256, 64)
        self.up_3_centerlines = PSPUpsample(64, 64)
        self.final_centerlines = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
        )


    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)

        p_nodes = self.drop_1(p)
        p_nodes = self.up_1_nodes(p_nodes)
        p_nodes = self.drop_2(p_nodes)
        p_nodes = self.up_2_nodes(p_nodes)
        p_nodes = self.drop_2(p_nodes)
        p_nodes = self.up_3_nodes(p_nodes)
        p_nodes = self.drop_2(p_nodes)

        p_centerlines = self.drop_1(p)
        p_centerlines = self.up_1_centerlines(p_centerlines)
        p_centerlines = self.drop_2(p_centerlines)
        p_centerlines = self.up_2_centerlines(p_centerlines)
        p_centerlines = self.drop_2(p_centerlines)
        p_centerlines = self.up_3_centerlines(p_centerlines)
        p_centerlines = self.drop_2(p_centerlines)

        return self.final_centerlines(p_centerlines), self.final_nodes(p_nodes)


class PSPNetQuadrupleHead(nn.Module):

    def __init__(self, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):

        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)

        self.drop_1 = nn.Dropout2d(p=0.3)
        self.drop_2 = nn.Dropout2d(p=0.15)

        self.up_1_direction = PSPUpsample(1024, 256)
        self.up_2_direction = PSPUpsample(256, 64)
        self.up_3_direction = PSPUpsample(64, 64)
        self.final_direction = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),  # Here, we predict [sin(direction), cos(direction)] to allow for MSE loss
        )

        self.up_1_centerline = PSPUpsample(1024, 256)
        self.up_2_centerline = PSPUpsample(256, 64)
        self.up_3_centerline = PSPUpsample(64, 64)
        self.final_centerline = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.up_1_intersection = PSPUpsample(1024, 256)
        self.up_2_intersection = PSPUpsample(256, 64)
        self.up_3_intersection = PSPUpsample(64, 64)
        self.final_intersection = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.up_1_anchors = PSPUpsample(1024, 256)
        self.up_2_anchors = PSPUpsample(256, 64)
        self.up_3_anchors = PSPUpsample(64, 64)
        self.final_anchors = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)

        p_direction = self.drop_1(p)
        p_direction = self.up_1_direction(p_direction)
        p_direction = self.drop_2(p_direction)
        p_direction = self.up_2_direction(p_direction)
        p_direction = self.drop_2(p_direction)
        p_direction = self.up_3_direction(p_direction)
        p_direction = self.drop_2(p_direction)

        p_centerline = self.drop_1(p)
        p_centerline = self.up_1_centerline(p_centerline)
        p_centerline = self.drop_2(p_centerline)
        p_centerline = self.up_2_centerline(p_centerline)
        p_centerline = self.drop_2(p_centerline)
        p_centerline = self.up_3_centerline(p_centerline)
        p_centerline = self.drop_2(p_centerline)

        p_intersection = self.drop_1(p)
        p_intersection = self.up_1_intersection(p_intersection)
        p_intersection = self.drop_2(p_intersection)
        p_intersection = self.up_2_intersection(p_intersection)
        p_intersection = self.drop_2(p_intersection)
        p_intersection = self.up_3_intersection(p_intersection)
        p_intersection = self.drop_2(p_intersection)

        p_anchors = self.drop_1(p)
        p_anchors = self.up_1_anchors(p_anchors)
        p_anchors = self.drop_2(p_anchors)
        p_anchors = self.up_2_anchors(p_anchors)
        p_anchors = self.drop_2(p_anchors)
        p_anchors = self.up_3_anchors(p_anchors)
        p_anchors = self.drop_2(p_anchors)

        return self.final_direction(p_direction), \
               self.final_centerline(p_centerline), \
               self.final_intersection(p_intersection), \
               self.final_anchors(p_anchors)
