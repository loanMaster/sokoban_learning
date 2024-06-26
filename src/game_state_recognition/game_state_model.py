import torch
from torch import nn
from src.common.encoder import Encoder

class GameStateModel(nn.Module):
    """This model determines if a level is finished or not.
    As input, it receives the start image and an image of the current state
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(6, 2, 256)
        self.fcn = torch.nn.Linear(256, 1)

    def forward(self, x1, x2):
        _x = torch.concat([x1, x2], dim=1)
        _x = self.encoder(_x)
        _x = torch.nn.MaxPool2d(kernel_size=(_x.size()[2], _x.size()[3]), stride=1, padding=0)(_x)

        _x = _x.squeeze()
        _x = self.fcn(_x)

        return _x
