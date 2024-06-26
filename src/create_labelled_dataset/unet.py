import torch
from torch import nn
from torch.nn.functional import leaky_relu

class UNet(nn.Module):
    def __init__(self, n_class, n_actions):
        super().__init__()

        self.e11 = nn.Conv2d(6 + n_actions, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 130

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 65

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 32

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 16

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024 + n_actions, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x, y, action = None):
        # Encoder
        if action is not None:
            action_temp = action.unsqueeze(-1).unsqueeze(-1).expand(action.shape[0], action.shape[1], x.shape[2], x.shape[3])
            x = torch.concat([x, y, action_temp], 1)

        xe11 = leaky_relu(self.e11(x))
        xe12 = leaky_relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = leaky_relu(self.e21(xp1))
        xe22 = leaky_relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = leaky_relu(self.e31(xp2))
        xe32 = leaky_relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = leaky_relu(self.e41(xp3))
        xe42 = leaky_relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = leaky_relu(self.e51(xp4))
        xe52 = leaky_relu(self.e52(xe51))

        # Decoder
        if action is not None:
            action_temp = action.unsqueeze(-1).unsqueeze(-1).expand(action.shape[0], action.shape[1], xe52.shape[2], xe52.shape[3])
            xe52 = torch.concat([xe52, action_temp], 1)

        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = leaky_relu(self.d11(xu11))
        xd12 = leaky_relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = leaky_relu(self.d21(xu22))
        xd22 = leaky_relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = leaky_relu(self.d31(xu33))
        xd32 = leaky_relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = leaky_relu(self.d41(xu44))
        xd42 = leaky_relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out
