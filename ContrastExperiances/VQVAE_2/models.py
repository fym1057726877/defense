import torch
from torch import nn
from torch.nn import functional as F
from ours.modules import Encoder, Decoder, ResBlock


class VQVAE_v2(nn.Module):
    def __init__(
            self,
            in_channel=1,
            channel=128,
            n_res_block=2,
            n_res_channel=32,
            embed_dim=64,
            n_embed=512,
    ):
        super(VQVAE_v2, self).__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.quantize_b = VectorQuantize(n_embed, embed_dim)
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, kernel_size=1, stride=1, padding=0)
        self.dec_b = Decoder(embed_dim + embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=4, )

        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_t = VectorQuantize(n_embed, embed_dim)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, kernel_size=1, stride=1, padding=0)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)

    def forward(self, input):
        z = self.encode(input)
        quant, vq_loss = self.encode_encode(z)
        rec = self.decode(quant)

        return rec, vq_loss

    def encode(self, input):
        return self.enc_b(input)

    def encode_encode(self, z):
        enc_t = self.enc_t(z)

        quant_t = self.quantize_conv_t(enc_t)
        quant_t, vq_loss_t, _, _, _ = self.quantize_t(quant_t)
        dec_t = self.dec_t(quant_t)

        enc_b = torch.cat([dec_t, z], dim=1)
        quant_b = self.quantize_conv_b(enc_b)
        quant_b, vq_loss_b, _, _, _ = self.quantize_b(quant_b)

        vq_loss = vq_loss_t + vq_loss_b

        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], dim=1)

        return quant, vq_loss

    def decode(self, quant):
        return self.dec_b(quant)


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
    x = torch.randn((16, 1, 64, 64))
    e1 = Encoder(in_channel=1, channel=128, n_res_block=2, n_res_channel=32, stride=4)
    quantize_b = VectorQuantize(512, 64)
    e2 = Encoder(in_channel=128, channel=128, n_res_block=2, n_res_channel=32, stride=2)
    quantize_t = VectorQuantize(512, 64)
    dec_b = Decoder(in_channel=64 + 64, out_channel=1, channel=128, n_res_block=2, n_res_channel=32, stride=4)
    dec_t = Decoder(in_channel=64, out_channel=64, channel=128, n_res_block=2, n_res_channel=32, stride=2)
    quantize_conv_b = nn.Conv2d(64 + 128, 64, kernel_size=1, stride=1, padding=0)
    quantize_conv_t = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
    upsample_t = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)

    x1 = e1(x)
    print(x1.shape)
    x2 = e2(x1)
    print(x2.shape)

    q_t = quantize_conv_t(x2)
    print(q_t.shape)
    q_t, loss_t, _, _, _ = quantize_t(q_t)
    dec_t = dec_t(q_t)
    print(q_t.shape, loss_t.shape)
    print(dec_t.shape)

    enc_b = torch.cat([dec_t, x1], dim=1)
    print(enc_b.shape)
    q_b = quantize_conv_b(enc_b)
    print(q_b.shape)
    q_b, loss_b, _, _, _ = quantize_b(q_b)
    print(q_b.shape, loss_b.shape)

    up_t = upsample_t(q_t)
    print(up_t.shape)
    quant = torch.cat([up_t, q_b], dim=1)
    print(quant.shape)
    dec = dec_b(quant)
    print(dec.shape)


if __name__ == '__main__':
    test()
