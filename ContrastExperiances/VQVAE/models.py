import torch
from torch import nn, optim
from torch.nn import functional as F
from ours.modules import Encoder, Decoder


class VQVAE(nn.Module):
    def __init__(
            self,
            in_channel=1,
            channel=128,
            n_res_block=2,
            n_res_channel=64,
            embed_dim=64,
            n_embed=512,
    ):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channel=in_channel, channel=channel, n_res_channel=n_res_channel,
                               n_res_block=n_res_block, stride=4)
        self.vq = VectorQuantize(n_embed=n_embed, embed_dim=embed_dim)
        self.decoder = Decoder(in_channel=channel, out_channel=in_channel, channel=channel,
                               n_res_channel=n_res_channel, n_res_block=n_res_block, stride=4)

        self.pre_conv = nn.Conv2d(channel, embed_dim, 3, 1, 1)
        self.post_conv = nn.Conv2d(embed_dim, channel, 3, 1, 1)

    def forward(self, x):
        h = self.encoder(x)
        h = self.pre_conv(h)
        h, vq_loss, _, _, _ = self.vq(h)
        h = self.post_conv(h)
        out = self.decoder(h)
        return out, vq_loss

    def training_loss(self, x):
        out, vq_loss = self(x)
        rec_loss = F.mse_loss(out, x)
        loss = rec_loss + vq_loss
        return loss

    def config_optimizer(self, lr=1e-4):
        return optim.Adam(list(self.encoder.parameters()) +
                          list(self.vq.parameters()) +
                          list(self.decoder.parameters()) +
                          list(self.pre_conv.parameters()) +
                          list(self.post_conv.parameters()),
                          lr=lr, betas=(0.9, 0.999))


class VectorQuantize(nn.Module):
    def __init__(self, n_embed, embed_dim, beta=0.25):
        """
           Discretization bottleneck part of the VQ-VAE.

           Inputs:
           - n_embed : number of embeddings
           - embed_dim : dimension of embedding
           - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        """
        super(VectorQuantize, self).__init__()

        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta

        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1. / self.n_embed, 1. / self.n_embed)

    def forward(self, z):
        """
            Inputs the output of the encoder network z and maps it to a discrete
            one-hot vector that is the index of the closest embedding vector e_j

            z (continuous) -> z_q (discrete)
            z.shape = (batch, channel, height, width)
            quantization pipeline:
                1. get encoder output (B,C,H,W)
                2. flatten input to (B*H*W,C)
        """
        z = z.permute(0, 2, 3, 1)

        z_flatten = z.reshape(-1, self.embed_dim).contiguous()
        dist = (
                z_flatten.pow(2).sum(dim=1, keepdim=True)
                - 2 * z_flatten @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(dim=1, keepdim=False)
        )
        embed_index = dist.argmin(dim=1)
        embed_onehot = F.one_hot(embed_index, self.n_embed).type(z_flatten.dtype)
        quantize = torch.matmul(embed_onehot, self.embedding.weight).view(z.shape)

        vq_loss = (quantize.detach() - z).pow(2).mean() + self.beta * (quantize - z.detach()).pow(2).mean()
        quantize = z + (quantize - z).detach()  # trick, 通过常数让编码器和解码器连接，可导

        # perplexity
        e_mean = torch.mean(embed_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        quantize = quantize.permute(0, 3, 1, 2).contiguous()

        return quantize, vq_loss, perplexity, embed_onehot, embed_index


def test():
    model = VQVAE(
        in_channel=1,
        channel=128,
        n_res_block=2,
        n_res_channel=64,
        embed_dim=64,
        n_embed=512
    )
    x = torch.randn((16, 1, 64, 64))
    out, _ = model(x)
    print(out.shape)

if __name__ == "__main__":
    test()
    