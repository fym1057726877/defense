import torch
from torch import nn
from LDM.aemodel.distributions import DiagonalGaussianDistribution
from LDM.aemodel.functions import LPIPSWithDiscriminator
from LDM.aemodel.modules import VectorQuantize, Encoder, Decoder, MemoryBlock, MaskMemoryBlock


class AutoencoderKL(nn.Module):
    def __init__(
            self, *, in_ch=1, ch=32, z_ch=4, out_ch=1, embed_dim=4, ch_mult=(1, 2, 4), resolution=64,
            num_res_blocks=2, attn_resolutions=(), dropout=0.0, double_z=True
    ):
        super().__init__()
        assert double_z
        self.encoder = Encoder(in_ch=in_ch, ch=ch, z_channels=z_ch, ch_mult=ch_mult, resolution=resolution,
                               num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout,
                               double_z=double_z)
        self.decoder = Decoder(ch=ch, out_ch=out_ch, z_channels=z_ch, ch_mult=ch_mult, resolution=resolution,
                               num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
        self.loss = LPIPSWithDiscriminator()
        self.quant_conv = nn.Conv2d(2 * z_ch, 2 * embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_ch, kernel_size=1)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, chunks=2, dim=1)
        posterior = DiagonalGaussianDistribution(mean, logvar)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def training_loss(self, inputs):
        reconstructions, posterior = self(inputs)
        aeloss = self.loss(inputs, reconstructions, posterior)
        return aeloss


class MemoryAutoencoder(nn.Module):
    def __init__(
            self, *, in_ch=1, ch=32, z_ch=4, out_ch=1, embed_dim=4, ch_mult=(1, 2, 4), resolution=64, num_res_blocks=2,
            attn_resolutions=(), dropout=0.0, double_z=False, num_memories=200, threshold=None, recloss_type="l1",
            entropy_loss_coef=2e-4, sparse=True
    ):
        super(MemoryAutoencoder, self).__init__()

        assert not double_z

        feature_resolution = int(resolution / 2 ** (len(ch_mult) - 1))
        feature_size = embed_dim * (feature_resolution ** 2)

        print(f"init MemoryBlock with num_memories={num_memories} and feature_size={feature_size}")

        self.encoder = Encoder(in_ch=in_ch, ch=ch, z_channels=z_ch, ch_mult=ch_mult, resolution=resolution,
                               num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout,
                               double_z=double_z)
        self.decoder = Decoder(ch=ch, out_ch=out_ch, z_channels=z_ch, ch_mult=ch_mult, resolution=resolution,
                               num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)

        self.loss = LPIPSLOSS(rec_loss_type=recloss_type)
        self.quant_conv = nn.Conv2d(z_ch, embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_ch, kernel_size=1)
        self.embed_dim = embed_dim

        self.memoryblock = MemoryBlock(num_memories=num_memories, features=feature_size, threshold=threshold,
                                       entropy_loss_coef=entropy_loss_coef, sparse=sparse)

    def encode(self, x):
        return self.encoder(x)

    def encode_encode(self, h):
        z = self.quant_conv(h)
        z_hat, mem_loss = self.memoryblock(z)
        return z_hat, mem_loss

    def decode(self, z_hat):
        z = self.post_quant_conv(z_hat)
        dec = self.decoder(z)
        return dec

    def forward(self, input):
        z = self.encode(input)
        z_hat, mem_loss = self.encode_encode(z)
        rec = self.decode(z_hat)
        return rec, mem_loss

    def training_loss(self, inputs):
        rec, mem_loss = self(inputs)
        aeloss = self.loss(inputs, rec) + mem_loss
        return aeloss


class AutoencoderVQ(nn.Module):
    def __init__(
            self, *, in_ch=1, ch=32, z_ch=4, out_ch=1, embed_dim=4, n_embed=512, quant_embed_dim=64, ch_mult=(1, 2, 4),
            resolution=64, num_res_blocks=2, attn_resolutions=(), dropout=0.0, double_z=False, recloss_type="l1"
    ):
        super(AutoencoderVQ, self).__init__()
        assert not double_z
        self.encoder = Encoder(in_ch=in_ch, ch=ch, z_channels=z_ch, ch_mult=ch_mult, resolution=resolution,
                               num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout,
                               double_z=double_z)
        self.decoder = Decoder(ch=ch, out_ch=out_ch, z_channels=z_ch, ch_mult=ch_mult, resolution=resolution,
                               num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
        self.quantize = VectorQuantize(n_embed, quant_embed_dim)
        self.loss = LPIPSLOSS(rec_loss_type=recloss_type)
        self.quant_conv = nn.Conv2d(z_ch, embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_ch, kernel_size=1)

    def forward(self, input):
        quant, z, vq_loss = self.encode(input)
        rec = self.decode(quant)
        return rec, vq_loss

    def encode(self, input):
        z = self.encoder(input)
        quant = self.quant_conv(z)
        quant, vq_loss, _, _, _ = self.quantize(quant)
        return quant, z, vq_loss

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def training_loss(self, input):
        reconstructions, vq_loss = self(input)
        aeloss = self.loss(input, reconstructions)
        return aeloss + vq_loss


class MaskMemoryAutoencoder(nn.Module):
    def __init__(
            self, *, in_ch=1, ch=32, z_ch=4, out_ch=1, embed_dim=4, ch_mult=(1, 2, 4), resolution=64, num_res_blocks=2,
            attn_resolutions=(), dropout=0.0, double_z=False, num_memories=200, threshold=None, recloss_type="l1",
            entropy_loss_coef=2e-4, sparse=True, num_classes=10,
    ):
        super(MaskMemoryAutoencoder, self).__init__()

        assert not double_z

        feature_resolution = int(resolution / 2 ** (len(ch_mult) - 1))
        feature_size = embed_dim * (feature_resolution ** 2)

        print(f"init MemoryBlock with num_memories={num_memories} and feature_size={feature_size}")

        self.encoder = Encoder(in_ch=in_ch, ch=ch, z_channels=z_ch, ch_mult=ch_mult, resolution=resolution,
                               num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout,
                               double_z=double_z)
        self.decoder = Decoder(ch=ch, out_ch=out_ch, z_channels=z_ch, ch_mult=ch_mult, resolution=resolution,
                               num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)

        self.loss = LPIPSWithDiscriminator(rec_loss_type=recloss_type)
        self.pre_conv = nn.Conv2d(z_ch, embed_dim, kernel_size=1)
        self.post_conv = nn.Conv2d(embed_dim, z_ch, kernel_size=1)
        self.embed_dim = embed_dim

        self.memoryblock = MaskMemoryBlock(num_memories=num_memories, features=feature_size, threshold=threshold,
                                           num_classes=num_classes, entropy_loss_coef=entropy_loss_coef, sparse=sparse)

        self.beta = 1e-4

    def encode(self, x):
        return self.encoder(x)

    def encode_encode(self, h, label):
        z = self.pre_conv(h)
        return self.memoryblock(z, label)

    def decode(self, memory_outdict):
        z_hat_target = memory_outdict["z_hat_target"]
        z_hat_non_target = memory_outdict["z_hat_non_target"]
        z_target = self.post_conv(z_hat_target)
        z_non_target = self.post_conv(z_hat_non_target)
        dec_target = self.decoder(z_target)
        dec_non_target = self.decoder(z_non_target)
        return dec_target, dec_non_target, memory_outdict["target_memloss"], memory_outdict["non_target_memloss"]

    def forward(self, input, label):
        z = self.encode(input)
        memory_outdict = self.encode_encode(z, label)
        return self.decode(memory_outdict)

    def training_loss(self, input, label, optimizer_idx):
        dec_target, dec_non_target, target_memloss, non_target_memloss = self(input, label)
        if optimizer_idx == 0:
            target_loss, log1 = self.loss(input, dec_target, optimizer_idx,
                                          last_layer=self.get_last_layer(), weights=1.)
            non_target_loss, log2 = self.loss(input, dec_non_target, optimizer_idx,
                                              last_layer=self.get_last_layer(), weights=1.)
            target_loss += target_memloss
            non_target_loss += non_target_memloss
            aeloss = target_loss - self.beta * non_target_loss.sigmoid()
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            target_discloss, log1 = self.loss(input, dec_target, optimizer_idx,
                                              last_layer=self.get_last_layer(), weights=1.)
            non_target_discloss, log2 = self.loss(input, dec_non_target, optimizer_idx,
                                                  last_layer=self.get_last_layer(), weights=1.)
            discloss = target_discloss - self.beta * non_target_discloss.sigmoid()
            return discloss

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def config_optimizer(self, lr=2e-4):
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.memoryblock.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.pre_conv.parameters()) +
                                  list(self.post_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc]


def testAutoencoderKL():
    model = AutoencoderKL(
        in_ch=1,
        ch=128,
        z_ch=4,
        out_ch=1,
        embed_dim=4,
        ch_mult=(1, 2, 4),
        resolution=64,
        num_res_blocks=2,
        attn_resolutions=(4, 2, 1),
        dropout=0.0,
        double_z=True
    )
    x = torch.randn((16, 1, 64, 64))
    z = model.encode(x).sample()
    print(z.shape)


def testMemoryAutoencoder():
    model = MaskMemoryAutoencoder(
        in_ch=1,
        ch=128,
        z_ch=4,
        out_ch=1,
        embed_dim=4,
        ch_mult=(1, 2, 4),
        resolution=28,
        num_res_blocks=2,
        attn_resolutions=(),
        dropout=0.0,
        double_z=False,
        num_memories=600,
        num_classes=10,
    )
    label = torch.randint(0, 10, (16,))
    x = torch.randn((16, 1, 28, 28))
    z = model.encode(x)
    print(z.shape)

    out = model(x, label)[0]
    print(out.shape)


def testAutoencoderVQ():
    model = AutoencoderVQ(
        in_ch=1,
        ch=128,
        z_ch=4,
        out_ch=1,
        embed_dim=4,
        ch_mult=(1, 2, 4),
        resolution=64,
        num_res_blocks=2,
        attn_resolutions=(4, 2, 1),
        dropout=0.0,
        double_z=False,
        n_embed=512,
        quant_embed_dim=64
    )
    x = torch.randn((16, 1, 64, 64))
    quant, z, _ = model.encode(x)
    print(quant.shape, z.shape)

    out = model(x)[0]
    print(out.shape)

# testMemoryAutoencoder()
# if __name__ == '__main__':
# testAutoencoderKL()
# testMemoryAutoencoder()
# testAutoencoderVQ()
# summary(model, input_size=(1, 64, 64), batch_size=16, device="cuda")
