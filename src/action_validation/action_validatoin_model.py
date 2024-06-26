import torch
from torch import nn
from src.common.encoder import Encoder

class ActionValidationModel(nn.Module):
    """This model determines if a move was valid/plausible
    It receives as input the start frame, current frame and the next frame.
    If the change from current to next frame is invalid/implausible the model will return 0 otherwise 1
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(6, 2, 256)
        self.fcn = torch.nn.Linear(256, 1)

    def forward(self, current, next_img):
        _x = torch.concat([current, next_img], dim=1)
        _x = self.encoder(_x)
        _x = torch.nn.MaxPool2d(kernel_size=(_x.size()[2], _x.size()[3]), stride=1, padding=0)(_x)
        _x = _x.squeeze()
        _x = self.fcn(_x)
        return _x
