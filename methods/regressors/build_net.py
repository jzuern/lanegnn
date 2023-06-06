import os
import torch
from regressors.pspnet import PSPNet, PSPNetDoubleHead
from collections import OrderedDict


models = {
    'squeezenet': lambda n_classes, in_channels: PSPNet(pretrained=False, in_channels=in_channels, n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda n_classes, in_channels: PSPNet(pretrained=False, in_channels=in_channels, n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda n_classes, in_channels: PSPNet(pretrained=False, in_channels=in_channels, n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda n_classes, in_channels: PSPNet(pretrained=False, in_channels=in_channels, n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda n_classes, in_channels: PSPNet(pretrained=False, in_channels=in_channels, n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda n_classes, in_channels: PSPNet(pretrained=False, in_channels=in_channels, n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda n_classes, in_channels: PSPNet(pretrained=False, in_channels=in_channels, n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


def build_network(snapshot, backend, num_channels=3, use_cuda=True, n_classes=14):

    backend = backend.lower()
    net = models[backend](in_channels=num_channels, n_classes=n_classes)

    map_location = torch.device('cpu') if not use_cuda else torch.device('cuda')
    if snapshot is not None:
        #net = torch.nn.DataParallel(net)
        # net.load_state_dict(torch.load(snapshot, map_location=map_location), strict=False) #['state_dict'])
        state_dict = torch.load(snapshot, map_location=map_location)["state_dict"]

        # Rename keys in state dict and remove "module." from keys
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        net.load_state_dict(new_state_dict, strict=True)

    if use_cuda:
        net = net.cuda()

    return net


def build_doublehead(use_cuda=True):

    net = PSPNetDoubleHead(psp_size=512)
    if use_cuda:
        net = net.cuda()

    return net
