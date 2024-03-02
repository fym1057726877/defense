import torch
import torch.nn as nn


class ConvGenerator(nn.Module):
    def __init__(self, latent_dim=256):
        super(ConvGenerator, self).__init__()
        self.net_dim = 64

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=8 * self.net_dim,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(8 * self.net_dim),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8 * self.net_dim, out_channels=4 * self.net_dim,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * self.net_dim),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4 * self.net_dim, out_channels=2 * self.net_dim,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * self.net_dim),
            nn.ReLU(inplace=True),
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * self.net_dim, out_channels=self.net_dim,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.net_dim),
            nn.ReLU(inplace=True),
        )

        self.final_deconv = nn.ConvTranspose2d(in_channels=self.net_dim, out_channels=1,
                                               kernel_size=4, stride=2, padding=1)
        self.out = nn.Tanh()

    def forward(self, z):
        while len(z.shape) < 4:
            z = z.unsqueeze(-1)
        x = self.block1(z)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.final_deconv(x)
        x = self.out(x)
        return x


class ConvDiscriminator(nn.Module):
    def __init__(self):
        super(ConvDiscriminator, self).__init__()
        self.net_dim = 64
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.net_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.net_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.net_dim, out_channels=2 * self.net_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.net_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=2 * self.net_dim, out_channels=4 * self.net_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.net_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=4 * self.net_dim, out_channels=8 * self.net_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.net_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=8 * self.net_dim, out_channels=1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.main(x).view(-1, 1)
        return x


def test():
    g = ConvGenerator()
    d = ConvDiscriminator()
    z = torch.randn((16, 256))
    x = g(z)
    r = d(x)
    print(x.shape, r.shape)


if __name__ == '__main__':
    test()


