import torch
import torch.nn as nn
import torch.nn.functional as F
from ours.modules import Encoder, Decoder


class DefensiveModel1(nn.Module):
    """Defensive model used for MNIST in MagNet paper
    """

    def __init__(self, in_channels=1):
        super(DefensiveModel1, self).__init__()
        self.conv_11 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_21 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_22 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_31 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_32 = nn.Conv2d(in_channels=3, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        """Forward propagation

        :param X: Mini-batch of shape [-1, 1, H, W]
        :return: Mini-batch of shape [-1, 1, H, W]
        """
        X = torch.tanh(self.conv_11(X))
        X = F.avg_pool2d(X, 2)
        X = torch.tanh(self.conv_21(X))
        X = torch.tanh(self.conv_22(X))
        X = F.interpolate(X, scale_factor=2)
        X = torch.tanh(self.conv_31(X))
        X = torch.tanh(self.conv_32(X))

        return X


class DefensiveModel2(nn.Module):
    """Defensive model used for CIFAR-10 in MagNet paper
    """

    def __init__(self, in_channels=1):
        super(DefensiveModel2, self).__init__()
        self.conv_11 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_21 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_31 = nn.Conv2d(in_channels=3, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        """Forward propagation

        :param X: Mini-batch of shape [-1, 1, H, W]
        :return: Mini-batch of shape [-1, 1, H, W]
        """
        X = torch.sigmoid(self.conv_11(X))
        X = torch.sigmoid(self.conv_21(X))
        X = torch.sigmoid(self.conv_31(X))

        return X


class Magnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(in_channel=1, channel=256, n_res_channel=64, n_res_block=2, stride=2)
        self.decoder = Decoder(in_channel=256, channel=256, out_channel=1, n_res_channel=64, n_res_block=2, stride=2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    model = DefensiveModel1()
    x = torch.randn(4, 1, 64, 64)
    y = model(x)
    print(y.shape)
