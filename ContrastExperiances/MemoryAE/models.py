import torch
from torch import nn, optim
import torch.nn.functional as F
from ours.modules import Encoder, Decoder, MemoryBlock


class MemoryAE(nn.Module):
    def __init__(
            self,
            num_memories=600,
            in_channel=1,
            channel=256,
            n_res_block=2,
            n_res_channel=64,
            feature_channel=4,
    ):
        super(MemoryAE, self).__init__()
        self.num_memories = num_memories
        self.encoder = Encoder(in_channel=in_channel, channel=channel, n_res_channel=n_res_channel,
                               n_res_block=n_res_block, stride=4)
        self.memory = MemoryBlock(num_memories=num_memories, features=feature_channel * 16 * 16)
        self.decoder = Decoder(in_channel=channel, out_channel=in_channel, channel=channel,
                               n_res_channel=n_res_channel, n_res_block=n_res_block, stride=4)
        self.pre_conv = nn.Conv2d(channel, feature_channel, 3, 1, 1)
        self.post_conv = nn.Conv2d(feature_channel, channel, 3, 1, 1)

    def forward(self, x):
        h = self.encoder(x)
        h = self.pre_conv(h)
        h, mem_loss = self.memory(h)
        h = self.post_conv(h)
        out = self.decoder(h)
        return out, mem_loss

    def config_optimizer(self, lr=1e-4):
        return optim.Adam(list(self.encoder.parameters()) +
                          list(self.memory.parameters()) +
                          list(self.decoder.parameters()) +
                          list(self.pre_conv.parameters()) +
                          list(self.post_conv.parameters()),
                          lr=lr, betas=(0.9, 0.999))

    def training_loss(self, x):
        rec_x, mem_loss = self(x)
        rec_loss = F.mse_loss(rec_x, x)
        loss = rec_loss + mem_loss
        return loss

    def EntropyLoss(self, mem_weight):
        entropy_loss = -mem_weight * torch.log(mem_weight + 1e-12)
        entropy_loss = entropy_loss.sum() / mem_weight.shape[0]
        entropy_loss *= self.entropy_loss_coef
        return entropy_loss


def test():
    x = torch.rand(16, 1, 64, 64)
    m = MemoryAE()
    rec, loss = m(x)
    print(rec.shape)

# test()
