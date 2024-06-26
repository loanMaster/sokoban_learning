import torch
from torch import nn
from src.common.encoder import Encoder
from src.common.resnet_block import ResBlock


class EncoderResnetModel(nn.Module):
    def __init__(self, n_class, activation_function = None, scaling_factor = 3, resnets = 10, resnet_channels = 256):
        super().__init__()
        self.activation_function = activation_function
        self.encoder = Encoder(6, scaling_factor, resnet_channels)
        self.backBone = nn.ModuleList(
            [ResBlock(resnet_channels) for i in range(resnets)]
        )
        self.fcn = torch.nn.Linear(resnet_channels, n_class)

    def forward(self, x1, x2):
        _x = torch.concat([x1, x2], dim=1)
        _x = self.encoder(_x)

        for resBlock in self.backBone:
            _x = resBlock(_x)

        _x = torch.nn.MaxPool2d(kernel_size=(_x.size()[2], _x.size()[3]), stride=1, padding=0)(_x)

        _x = _x.squeeze()
        _x = self.fcn(_x)

        if self.activation_function:
            _x = self.activation_function(_x)

        return _x

