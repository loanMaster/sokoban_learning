import torch
from torch import nn
from torch.nn.functional import leaky_relu

class Encoder(nn.Module):
    """Encodes a single screenshot or a stack of screenshots of Sokoban
    The expected input is of size NUM_CHANNELSx160x160
    Output is of size OUTPUT_CHANNELSx10x10
    """
    def __init__(self, in_channels = 3, scaling_factor = 2, output_channels = 16):
        super().__init__()

        self.e11a = nn.Conv2d(in_channels, 16 * scaling_factor, kernel_size=3, padding='same')
        self.e11b = nn.Conv2d(in_channels, 16 * scaling_factor, kernel_size=5, padding='same')
        self.e12a = nn.Conv2d(16 * scaling_factor * 2, 16 * scaling_factor, kernel_size=3, padding='same')
        self.e12b = nn.Conv2d(16 * scaling_factor * 2, 16 * scaling_factor, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm2d(2 * 16 * scaling_factor)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21a = nn.Conv2d(16 * scaling_factor * 2, 32 * scaling_factor, kernel_size=3, padding='same')
        self.e21b = nn.Conv2d(16 * scaling_factor * 2, 32 * scaling_factor, kernel_size=5, padding='same')
        self.e22a = nn.Conv2d(32 * scaling_factor * 2, 32 * scaling_factor, kernel_size=3, padding='same')
        self.e22b = nn.Conv2d(32 * scaling_factor * 2, 32 * scaling_factor, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm2d(2 * 32 * scaling_factor)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31a = nn.Conv2d(32 * scaling_factor * 2, 64 * scaling_factor, kernel_size=3, padding='same')
        self.e31b = nn.Conv2d(32 * scaling_factor * 2, 64 * scaling_factor, kernel_size=5, padding='same')
        self.e32a = nn.Conv2d(64 * scaling_factor * 2, 64 * scaling_factor, kernel_size=3, padding='same')
        self.e32b = nn.Conv2d(64 * scaling_factor * 2, 64 * scaling_factor, kernel_size=5, padding='same')
        self.bn3 = nn.BatchNorm2d(2 * 64 * scaling_factor)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41a = nn.Conv2d(64 * scaling_factor * 2, 128 * scaling_factor, kernel_size=3, padding='same')
        self.e41b = nn.Conv2d(64 * scaling_factor * 2, 128 * scaling_factor, kernel_size=5, padding='same')
        self.e42a = nn.Conv2d(128 * scaling_factor * 2, 128 * scaling_factor, kernel_size=3, padding='same')
        self.e42b = nn.Conv2d(128 * scaling_factor * 2, 128 * scaling_factor, kernel_size=5, padding='same')
        self.bn4 = nn.BatchNorm2d(2 * 128 * scaling_factor)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(128 * scaling_factor * 2, 256 * scaling_factor, kernel_size=3, padding='same')
        self.e52 = nn.Conv2d(256 * scaling_factor, output_channels, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        xe11a = leaky_relu(self.e11a(x))
        xe11b = leaky_relu(self.e11b(x))
        xe11 = torch.concat([xe11a, xe11b], dim=1)

        xe12a = leaky_relu(self.e12a(xe11))
        xe12b = leaky_relu(self.e12b(xe11))
        xe12 = torch.concat([xe12a, xe12b], dim=1)

        xe12 = self.bn1(xe12)
        xp1 = self.pool1(xe12)

        xe21a = leaky_relu(self.e21a(xp1))
        xe21b = leaky_relu(self.e21b(xp1))
        xe21 = torch.concat([xe21a, xe21b], dim=1)

        xe22a = leaky_relu(self.e22a(xe21))
        xe22b = leaky_relu(self.e22b(xe21))
        xe22 = torch.concat([xe22a, xe22b], dim=1)

        xe22 = self.bn2(xe22)
        xp2 = self.pool2(xe22)

        xe31a = leaky_relu(self.e31a(xp2))
        xe31b = leaky_relu(self.e31b(xp2))
        xe31 = torch.concat([xe31a, xe31b], dim=1)

        xe32a = leaky_relu(self.e32a(xe31))
        xe32b = leaky_relu(self.e32b(xe31))
        xe32 = torch.concat([xe32a, xe32b], dim=1)

        xe32 = self.bn3(xe32)
        xp3 = self.pool3(xe32)

        xe41a = leaky_relu(self.e41a(xp3))
        xe41b = leaky_relu(self.e41b(xp3))
        xe41 = torch.concat([xe41a, xe41b], dim=1)

        xe42a = leaky_relu(self.e42a(xe41))
        xe42b = leaky_relu(self.e42b(xe41))
        xe42 = torch.concat([xe42a, xe42b], dim=1)

        xe42 = self.bn4(xe42)
        xp4 = self.pool4(xe42)

        xe51 = leaky_relu(self.e51(xp4))
        xe52 = leaky_relu(self.e52(xe51))
        return self.bn5(xe52)

