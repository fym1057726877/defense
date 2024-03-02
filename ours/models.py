from ours.modules import *
from ours.losses import LPIPSWithDiscriminator


class MemoryGAN_v2(nn.Module):
    def __init__(
            self,
            in_channel=1,
            channel=256,
            n_res_block=2,
            n_res_channel=64,
            num_memories=600,
            memory_threshold=1e-4,
            sparse=True,
            features_channels=4,
            resolution=64
    ):
        super().__init__()

        self.bottom_enc = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        bottom_memory_features = features_channels * (resolution // 4) ** 2
        self.bottom_memory = MemoryBlock(num_memories=num_memories, features=bottom_memory_features,
                                         threshold=memory_threshold, sparse=sparse)
        self.bottom_pre_conv = nn.Conv2d(channel * 2, features_channels, kernel_size=3, stride=1, padding=1)
        self.bottom_post_conv = nn.Conv2d(features_channels, channel * 2, kernel_size=3, stride=1, padding=1)
        self.bottom_dec = Decoder(channel * 2, in_channel, channel, n_res_block, n_res_channel, stride=4)

        self.top_enc = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        top_memory_features = features_channels * (resolution // (4 * 2)) ** 2
        self.top_memory = MemoryBlock(num_memories=num_memories, features=top_memory_features,
                                      threshold=memory_threshold, sparse=sparse)
        self.top_memory_pre_conv = nn.Conv2d(channel, features_channels, kernel_size=3, stride=1, padding=1)
        self.top_memory_post_conv = nn.Conv2d(features_channels, channel, kernel_size=3, stride=1, padding=1)
        self.top_dec = Decoder(channel, channel, channel, n_res_block, n_res_channel, stride=2)
        self.top_upsample = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1)

        self.conv_to_decode = nn.Conv2d(channel * 3, channel * 2, kernel_size=3, stride=1, padding=1)

        self.loss = LPIPSWithDiscriminator(disc_num_layers=3, disc_in_channels=1, disc_weight=1.0, rec_loss_type="l1",
                                           perceptual_weight=1.0, disc_conditional=False, disc_loss="hinge",
                                           lpips_type="resnet18")

    def forward(self, input):
        z = self.encode(input)
        quant, mem_loss = self.encode_encode(z)
        rec = self.decode(quant)
        return rec, mem_loss

    def encode(self, input):
        return self.bottom_enc(input)

    def encode_encode(self, z):
        quant_top = self.top_enc(z)
        quant_top = self.top_memory_pre_conv(quant_top)
        quant_top, mem_loss_top = self.top_memory(quant_top)
        quant_top = self.top_memory_post_conv(quant_top)
        dec_top = self.top_dec(quant_top)
        quant_bottom = torch.cat([dec_top, z], dim=1)
        quant_bottom = self.bottom_pre_conv(quant_bottom)
        quant_bottom, mem_loss_bottom = self.bottom_memory(quant_bottom)
        quant_bottom = self.bottom_post_conv(quant_bottom)
        mem_loss = mem_loss_top + mem_loss_bottom
        upsample_t = self.top_upsample(quant_top)
        quant = torch.cat([upsample_t, quant_bottom], dim=1)
        quant = self.conv_to_decode(quant)
        return quant, mem_loss

    def decode(self, quant):
        return self.bottom_dec(quant)

    def training_loss(self, x, optimizer_idx):
        rec, mem_loss = self(x)
        if optimizer_idx == 0:
            # train encoder+decoder+logvar+memory
            aeloss, log_dict_ae = self.loss(x, rec, optimizer_idx, last_layer=self.get_last_layer(), weights=1.)
            # self.logs("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            aeloss = aeloss + mem_loss
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(x, rec, optimizer_idx, last_layer=self.get_last_layer(), weights=1.)

            # self.logs("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            # self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def get_last_layer(self):
        return self.bottom_dec.blocks[-1].weight

    def config_optimizer(self, lr=2e-4):
        opt_ae = torch.optim.Adam(list(self.bottom_enc.parameters()) +
                                  list(self.bottom_memory.parameters()) +
                                  list(self.bottom_pre_conv.parameters()) +
                                  list(self.bottom_post_conv.parameters()) +
                                  list(self.bottom_dec.parameters()) +
                                  list(self.top_enc.parameters()) +
                                  list(self.top_memory.parameters()) +
                                  list(self.top_memory_pre_conv.parameters()) +
                                  list(self.top_memory_post_conv.parameters()) +
                                  list(self.top_dec.parameters()) +
                                  list(self.top_upsample.parameters()) +
                                  list(self.conv_to_decode.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc]


class MlpMemoryGAN(nn.Module):
    def __init__(
            self,
            in_channel=1,
            channel=256,
            n_res_block=2,
            n_res_channel=64,
            num_memories=600,
            memory_threshold=None,
            sparse=True,
            features_channels=4,
            resolution=64,
            other_dims=1
    ):
        super().__init__()

        self.bottom_enc = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        bottom_memory_features = features_channels * (resolution // 4) ** 2
        self.bottom_memory = MlpMemoryBlock(num_memories=num_memories,
                                            features=bottom_memory_features,
                                            other_dims=other_dims,
                                            threshold=memory_threshold,
                                            sparse=sparse)
        self.bottom_pre_conv = nn.Conv2d(channel * 2, features_channels, kernel_size=3, stride=1, padding=1)
        self.bottom_post_conv = nn.Conv2d(features_channels, channel * 2, kernel_size=3, stride=1, padding=1)
        self.bottom_dec = Decoder(channel * 2, in_channel, channel, n_res_block, n_res_channel, stride=4)

        self.top_enc = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        top_memory_features = features_channels * (resolution // (4 * 2)) ** 2
        self.top_memory = MlpMemoryBlock(num_memories=num_memories,
                                         features=top_memory_features,
                                         threshold=memory_threshold,
                                         other_dims=other_dims,
                                         sparse=sparse)
        self.top_memory_pre_conv = nn.Conv2d(channel, features_channels, kernel_size=3, stride=1, padding=1)
        self.top_memory_post_conv = nn.Conv2d(features_channels, channel, kernel_size=3, stride=1, padding=1)
        self.top_dec = Decoder(channel, channel, channel, n_res_block, n_res_channel, stride=2)
        self.top_upsample = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1)

        self.conv_to_decode = nn.Conv2d(channel * 3, channel * 2, kernel_size=3, stride=1, padding=1)

        self.loss = LPIPSWithDiscriminator(disc_num_layers=3, disc_in_channels=1, disc_weight=1.0, rec_loss_type="l1",
                                           perceptual_weight=1.0, disc_conditional=False, disc_loss="hinge",
                                           lpips_type="resnet18")

    def forward(self, input):
        z = self.encode(input)
        quant, mem_loss = self.encode_encode(z)
        rec = self.decode(quant)
        return rec, mem_loss

    def encode(self, input):
        return self.bottom_enc(input)

    def encode_encode(self, z):
        quant_top = self.top_enc(z)
        quant_top = self.top_memory_pre_conv(quant_top)
        quant_top, mem_loss_top = self.top_memory(quant_top)
        quant_top = self.top_memory_post_conv(quant_top)
        dec_top = self.top_dec(quant_top)
        quant_bottom = torch.cat([dec_top, z], dim=1)
        quant_bottom = self.bottom_pre_conv(quant_bottom)
        quant_bottom, mem_loss_bottom = self.bottom_memory(quant_bottom)
        quant_bottom = self.bottom_post_conv(quant_bottom)
        mem_loss = mem_loss_top + mem_loss_bottom
        upsample_t = self.top_upsample(quant_top)
        quant = torch.cat([upsample_t, quant_bottom], dim=1)
        quant = self.conv_to_decode(quant)
        return quant, mem_loss

    def decode(self, quant):
        return self.bottom_dec(quant)

    def training_loss(self, x, optimizer_idx):
        rec, mem_loss = self(x)
        if optimizer_idx == 0:
            # train encoder+decoder+logvar+memory
            aeloss, log_dict_ae = self.loss(x, rec, optimizer_idx, last_layer=self.get_last_layer(), weights=1.)
            # self.logs("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            aeloss = aeloss + mem_loss
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(x, rec, optimizer_idx, last_layer=self.get_last_layer(), weights=1.)

            # self.logs("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            # self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def get_last_layer(self):
        return self.bottom_dec.blocks[-1].weight

    def config_optimizer(self, lr=2e-4):
        opt_ae = torch.optim.Adam(list(self.bottom_enc.parameters()) +
                                  list(self.bottom_memory.parameters()) +
                                  list(self.bottom_pre_conv.parameters()) +
                                  list(self.bottom_post_conv.parameters()) +
                                  list(self.bottom_dec.parameters()) +
                                  list(self.top_enc.parameters()) +
                                  list(self.top_memory.parameters()) +
                                  list(self.top_memory_pre_conv.parameters()) +
                                  list(self.top_memory_post_conv.parameters()) +
                                  list(self.top_dec.parameters()) +
                                  list(self.top_upsample.parameters()) +
                                  list(self.conv_to_decode.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc]


class MlpMemoryGAN_PV600(nn.Module):
    def __init__(
            self,
            in_channel=1,
            channel=256,
            n_res_block=2,
            n_res_channel=64,
            num_memories=600,
            memory_threshold=None,
            sparse=True,
            features_channels=4,
            resolution=64,
            other_dims=1
    ):
        super().__init__()

        self.bottom_enc = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        bottom_memory_features = features_channels * (resolution // 4) ** 2
        self.bottom_memory = MlpMemoryBlock_PV600(num_memories=num_memories,
                                                  features=bottom_memory_features,
                                                  other_dims=other_dims,
                                                  threshold=memory_threshold,
                                                  sparse=sparse)
        self.bottom_pre_conv = nn.Conv2d(channel * 2, features_channels, kernel_size=3, stride=1, padding=1)
        self.bottom_post_conv = nn.Conv2d(features_channels, channel * 2, kernel_size=3, stride=1, padding=1)
        self.bottom_dec = Decoder(channel * 2, in_channel, channel, n_res_block, n_res_channel, stride=4)

        self.top_enc = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        top_memory_features = features_channels * (resolution // (4 * 2)) ** 2
        self.top_memory = MlpMemoryBlock_PV600(num_memories=num_memories,
                                               features=top_memory_features,
                                               threshold=memory_threshold,
                                               other_dims=other_dims,
                                               sparse=sparse)
        self.top_memory_pre_conv = nn.Conv2d(channel, features_channels, kernel_size=3, stride=1, padding=1)
        self.top_memory_post_conv = nn.Conv2d(features_channels, channel, kernel_size=3, stride=1, padding=1)
        self.top_dec = Decoder(channel, channel, channel, n_res_block, n_res_channel, stride=2)
        self.top_upsample = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1)

        self.conv_to_decode = nn.Conv2d(channel * 3, channel * 2, kernel_size=3, stride=1, padding=1)

        self.loss = LPIPSWithDiscriminator(disc_num_layers=3, disc_in_channels=1, disc_weight=1.0, rec_loss_type="l1",
                                           perceptual_weight=1.0, disc_conditional=False, disc_loss="hinge",
                                           lpips_type="resnet18")

    def forward(self, input):
        z = self.encode(input)
        quant, mem_loss = self.encode_encode(z)
        rec = self.decode(quant)
        return rec, mem_loss

    def encode(self, input):
        return self.bottom_enc(input)

    def encode_encode(self, z):
        quant_top = self.top_enc(z)
        quant_top = self.top_memory_pre_conv(quant_top)
        quant_top, mem_loss_top = self.top_memory(quant_top)
        quant_top = self.top_memory_post_conv(quant_top)
        dec_top = self.top_dec(quant_top)
        quant_bottom = torch.cat([dec_top, z], dim=1)
        quant_bottom = self.bottom_pre_conv(quant_bottom)
        quant_bottom, mem_loss_bottom = self.bottom_memory(quant_bottom)
        quant_bottom = self.bottom_post_conv(quant_bottom)
        mem_loss = mem_loss_top + mem_loss_bottom
        upsample_t = self.top_upsample(quant_top)
        quant = torch.cat([upsample_t, quant_bottom], dim=1)
        quant = self.conv_to_decode(quant)
        return quant, mem_loss

    def decode(self, quant):
        return self.bottom_dec(quant)

    def training_loss(self, x, optimizer_idx):
        rec, mem_loss = self(x)
        if optimizer_idx == 0:
            # train encoder+decoder+logvar+memory
            aeloss, log_dict_ae = self.loss(x, rec, optimizer_idx, last_layer=self.get_last_layer(), weights=1.)
            # self.logs("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            aeloss = aeloss + mem_loss
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(x, rec, optimizer_idx, last_layer=self.get_last_layer(), weights=1.)

            # self.logs("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            # self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def get_last_layer(self):
        return self.bottom_dec.blocks[-1].weight

    def config_optimizer(self, lr=2e-4):
        opt_ae = torch.optim.Adam(list(self.bottom_enc.parameters()) +
                                  list(self.bottom_memory.parameters()) +
                                  list(self.bottom_pre_conv.parameters()) +
                                  list(self.bottom_post_conv.parameters()) +
                                  list(self.bottom_dec.parameters()) +
                                  list(self.top_enc.parameters()) +
                                  list(self.top_memory.parameters()) +
                                  list(self.top_memory_pre_conv.parameters()) +
                                  list(self.top_memory_post_conv.parameters()) +
                                  list(self.top_dec.parameters()) +
                                  list(self.top_upsample.parameters()) +
                                  list(self.conv_to_decode.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc]


def test():
    x = torch.randn((16, 1, 64, 64))
    model = MemoryGAN_v2(
        in_channel=1,
        channel=256,
        n_res_block=2,
        n_res_channel=64,
        num_memories=600,
        memory_threshold=1e-4,
        sparse=True,
        features_channels=4,
        resolution=64
    )
    out, loss = model(x)
    print(out.shape)


if __name__ == '__main__':
    test()
