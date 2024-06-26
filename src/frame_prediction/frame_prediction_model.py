import torch
from torch import nn
from torch.nn.functional import leaky_relu

class FramePredictionModel(nn.Module):
    def __init__(self, n_class, actions):
        super().__init__()

        scaling_factor = 4
        self.e11 = nn.Conv2d(6 + actions, 16 * scaling_factor, kernel_size=3, padding='same')
        self.e11a = nn.Conv2d(6 + actions, 16 * scaling_factor, kernel_size=5, padding='same')
        self.e11b = nn.Conv2d(6 + actions, 16 * scaling_factor, kernel_size=7, padding='same')
        self.e12 = nn.Conv2d(16 * scaling_factor * 3, 16 * scaling_factor, kernel_size=3, padding='same')
        self.e12a = nn.Conv2d(16 * scaling_factor * 3, 16 * scaling_factor, kernel_size=5, padding='same')
        self.e12b = nn.Conv2d(16 * scaling_factor * 3, 16 * scaling_factor, kernel_size=7, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 284x284x16
        self.e21 = nn.Conv2d(16 * scaling_factor * 3, 32 * scaling_factor, kernel_size=3, padding='same')
        self.e21a = nn.Conv2d(16 * scaling_factor * 3, 32 * scaling_factor, kernel_size=5, padding='same')
        self.e21b = nn.Conv2d(16 * scaling_factor * 3, 32 * scaling_factor, kernel_size=7, padding='same')
        self.e22 = nn.Conv2d(32 * scaling_factor * 3, 32 * scaling_factor, kernel_size=3, padding='same')
        self.e22a = nn.Conv2d(32 * scaling_factor * 3, 32 * scaling_factor, kernel_size=5, padding='same')
        self.e22b = nn.Conv2d(32 * scaling_factor * 3, 32 * scaling_factor, kernel_size=7, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 140x140x32
        self.e31 = nn.Conv2d(32 * scaling_factor * 3, 64 * scaling_factor, kernel_size=3, padding='same')
        self.e31a = nn.Conv2d(32 * scaling_factor * 3, 64 * scaling_factor, kernel_size=5, padding='same')
        self.e31b = nn.Conv2d(32 * scaling_factor * 3, 64 * scaling_factor, kernel_size=7, padding='same')
        self.e32 = nn.Conv2d(64 * scaling_factor * 3, 64 * scaling_factor, kernel_size=3, padding='same')
        self.e32a = nn.Conv2d(64 * scaling_factor * 3, 64 * scaling_factor, kernel_size=5, padding='same')
        self.e32b = nn.Conv2d(64 * scaling_factor * 3, 64 * scaling_factor, kernel_size=7, padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 68x68x64
        self.e41 = nn.Conv2d(64 * scaling_factor * 3, 128 * scaling_factor, kernel_size=3, padding='same')
        self.e41a = nn.Conv2d(64 * scaling_factor * 3, 128 * scaling_factor, kernel_size=5, padding='same')
        self.e41b = nn.Conv2d(64 * scaling_factor * 3, 128 * scaling_factor, kernel_size=7, padding='same')
        self.e42 = nn.Conv2d(128 * scaling_factor * 3, 128 * scaling_factor, kernel_size=3, padding='same')
        self.e42a = nn.Conv2d(128 * scaling_factor * 3, 128 * scaling_factor, kernel_size=5, padding='same')
        self.e42b = nn.Conv2d(128 * scaling_factor * 3, 128 * scaling_factor, kernel_size=7, padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 32x32x128
        self.e51 = nn.Conv2d(128 * scaling_factor * 3, 256 * scaling_factor, kernel_size=3, padding='same')
        self.e52 = nn.Conv2d(256 * scaling_factor, 256 * scaling_factor, kernel_size=3, padding='same')

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256 * scaling_factor, 128 * scaling_factor, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(128 * scaling_factor + 128 * scaling_factor * 3, 128 * scaling_factor, kernel_size=3, padding='same')
        self.d12 = nn.Conv2d(128 * scaling_factor, 128 * scaling_factor, kernel_size=3, padding='same')

        self.upconv2 = nn.ConvTranspose2d(128 * scaling_factor, 64 * scaling_factor, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(64 * scaling_factor + 64 * scaling_factor * 3, 64 * scaling_factor, kernel_size=3, padding='same')
        self.d22 = nn.Conv2d(64 * scaling_factor, 64 * scaling_factor, kernel_size=3, padding='same')

        self.upconv3 = nn.ConvTranspose2d(64 * scaling_factor, 32 * scaling_factor, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(32 * scaling_factor + 32 * scaling_factor * 3, 32 * scaling_factor, kernel_size=3, padding='same')
        self.d32 = nn.Conv2d(32 * scaling_factor, 32 * scaling_factor, kernel_size=3, padding='same')

        self.upconv4 = nn.ConvTranspose2d(32 * scaling_factor, 16 * scaling_factor, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(16 * scaling_factor + 16 * scaling_factor * 3, 16 * scaling_factor, kernel_size=3, padding='same')
        self.d42 = nn.Conv2d(16 * scaling_factor, 16 * scaling_factor, kernel_size=3, padding='same')

        # Output layer
        self.outconv = nn.Conv2d(16 * scaling_factor, n_class, kernel_size=1)

    def forward(self, frame0, frame1, action):
        # Encoder
        action_temp = action.unsqueeze(-1).unsqueeze(-1).expand(action.shape[0], action.shape[1], frame0.shape[2], frame0.shape[3])
        x = torch.concat([frame0, frame1, action_temp], dim=1)

        xe11 = leaky_relu(self.e11(x))
        xe11a = leaky_relu(self.e11a(x))
        xe11b = leaky_relu(self.e11b(x))
        xe11 = torch.concat([xe11, xe11a, xe11b], dim=1)

        xe12 = leaky_relu(self.e12(xe11))
        xe12a = leaky_relu(self.e12a(xe11))
        xe12b = leaky_relu(self.e12b(xe11))
        xe12 = torch.concat([xe12, xe12a, xe12b], dim=1)

        xp1 = self.pool1(xe12)

        xe21 = leaky_relu(self.e21(xp1))
        xe21a = leaky_relu(self.e21a(xp1))
        xe21b = leaky_relu(self.e21b(xp1))
        xe21 = torch.concat([xe21, xe21a, xe21b], dim=1)

        xe22 = leaky_relu(self.e22(xe21))
        xe22a = leaky_relu(self.e22a(xe21))
        xe22b = leaky_relu(self.e22b(xe21))
        xe22 = torch.concat([xe22, xe22a, xe22b], dim=1)

        xp2 = self.pool2(xe22)

        xe31 = leaky_relu(self.e31(xp2))
        xe31a = leaky_relu(self.e31a(xp2))
        xe31b = leaky_relu(self.e31b(xp2))
        xe31 = torch.concat([xe31, xe31a, xe31b], dim=1)

        xe32 = leaky_relu(self.e32(xe31))
        xe32a = leaky_relu(self.e32a(xe31))
        xe32b = leaky_relu(self.e32b(xe31))
        xe32 = torch.concat([xe32, xe32a, xe32b], dim=1)

        xp3 = self.pool3(xe32)

        xe41 = leaky_relu(self.e41(xp3))
        xe41a = leaky_relu(self.e41a(xp3))
        xe41b = leaky_relu(self.e41b(xp3))
        xe41 = torch.concat([xe41, xe41a, xe41b], dim=1)

        xe42 = leaky_relu(self.e42(xe41))
        xe42a = leaky_relu(self.e42a(xe41))
        xe42b = leaky_relu(self.e42b(xe41))
        xe42 = torch.concat([xe42, xe42a, xe42b], dim=1)

        xp4 = self.pool4(xe42)

        xe51 = leaky_relu(self.e51(xp4))
        xe52 = leaky_relu(self.e52(xe51))


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

        out = self.outconv(xd42)

        return out
