import torch.nn as nn
import torchvision
import torch


class Resnet34Actor(nn.Module):
    def __init__(self, latent_dim=256):
        super(Resnet34Actor, self).__init__()
        self.action_bound = 2
        self.conv_in = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.resnet = torchvision.models.resnet34()
        self.linear_out = nn.Linear(1000, latent_dim)

    def forward(self, img):
        z = self.conv_in(img)
        z = self.resnet(z)
        z = self.linear_out(z)
        z = torch.tanh(z)
        z = z * self.action_bound
        return z
