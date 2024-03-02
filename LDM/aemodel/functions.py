import os
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from collections import namedtuple
from utils import get_project_path
import functools


# ===========================================Loss Functions==============================================
def l1(x, y):
    return torch.abs(x - y)


def l2(x, y):
    return torch.pow((x - y), 2)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
            torch.mean(F.softplus(-logits_real)) +
            torch.mean(F.softplus(logits_fake))
    )
    return d_loss


class LPIPSLOSS(nn.Module):
    def __init__(self, recloss_weight=1.0, perceptual_weight=1.0, rec_loss_type="l1"):
        super(LPIPSLOSS, self).__init__()
        self.recloss_weight = recloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.rec_loss_type = rec_loss_type

    def forward(self, inputs, reconstructions):
        if self.rec_loss_type == "l1":
            rec_loss = self.recloss_weight * l1(inputs.contiguous(), reconstructions.contiguous())
        elif self.rec_loss_type == "l2":
            rec_loss = self.recloss_weight * l2(inputs.contiguous(), reconstructions.contiguous())
        else:
            raise NotImplementedError("Loss type not implemented")

        p_loss = self.perceptual_weight * self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())

        total_loss = rec_loss + p_loss

        total_loss = torch.sum(total_loss) / total_loss.shape[0]

        return total_loss


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_num_layers=3, disc_in_channels=1, rec_loss_type="l1", lpips_type="vgg16",
                 disc_weight=1.0, perceptual_weight=1.0, disc_conditional=False, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert lpips_type in ["vgg16", "resnet18"]
        if lpips_type == "vgg16":
            self.perceptual_loss = LPIPS().eval()
        else:
            self.perceptual_loss = LPIPS_with_Resnet18().eval()
        self.perceptual_weight = perceptual_weight
        if rec_loss_type == "l1":
            self.rec_loss = l1
        if rec_loss_type == "l2":
            self.rec_loss = l2

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 ).apply(weights_init)
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, optimizer_idx, last_layer=None,
                cond=None, weights=1., split="train"):
        rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        weighted_rec_loss = weights * rec_loss
        weighted_rec_loss = torch.sum(weighted_rec_loss) / weighted_rec_loss.shape[0]
        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))

            g_loss = -torch.mean(logits_fake)

            d_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer=last_layer)

            loss = weighted_rec_loss + d_weight * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/g_loss".format(split): g_loss.detach().mean(), }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            d_loss = self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


# =============================================================================================================
"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=False, requires_grad=False)
        self.load_vgg16()

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)

        for param in self.parameters():
            param.requires_grad = False

    def load_vgg16(self, name="vgg16"):
        ckpt_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", f"{name}.pth")
        if os.path.exists(ckpt_path):
            self.net.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cuda")), strict=False)
            print("loaded pretrained vgg16 from {}".format(ckpt_path))

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for i in range(1, len(self.chns)):
            val += res[i]
        return val


class LPIPS_with_Resnet18(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 64, 128, 256, 512]  # vg16 features
        self.net = Resnet18(requires_grad=False)
        self.load_resnet18()

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)

        for param in self.parameters():
            param.requires_grad = False

    def load_resnet18(self):
        ckpt_path = os.path.join(get_project_path("Defense"), "pretrained", "resnet18_downloaded.pth")
        if os.path.exists(ckpt_path):
            self.net.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cuda")), strict=False)
            print("loaded pretrained resnet18 from {}".format(ckpt_path))

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for i in range(1, len(self.chns)):
            val += res[i]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=False):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class Resnet18(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.slice1 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), list(self.model.children())[x])
        self.slice2 = self.model.layer1
        self.slice3 = self.model.layer2
        self.slice4 = self.model.layer3
        self.slice5 = self.model.layer4
        self.N_slices = 5
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = requires_grad

    def forward(self, X):
        h = self.slice1(X)
        f1 = h
        h = self.slice2(h)
        f2 = h
        h = self.slice3(h)
        f3 = h
        h = self.slice4(h)
        f4 = h
        h = self.slice5(h)
        f5 = h
        resnet_outputs = namedtuple("Resnet_outputs", ['f1', 'f2', 'f3', 'f4', 'f5'])
        out = resnet_outputs(f1, f2, f3, f4, f5)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=1, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        norm_layer = nn.BatchNorm2d
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

