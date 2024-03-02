import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from einops import rearrange


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=2,
                                  padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels,
                                       out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                               out_channels,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class LinAttnBlock(nn.Module):
    def __init__(self, dim, heads=1, dim_head=None):
        super().__init__()
        self.heads = heads
        dim_head = dim or dim_head
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.k = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.v = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.proj_out = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


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


class MemoryBlock(nn.Module):
    def __init__(self, num_memories, features, sparse=True, threshold=None, entropy_loss_coef=0.0002):
        super(MemoryBlock, self).__init__()

        self.num_memories = num_memories
        self.features = features

        self.memory = torch.zeros((self.num_memories, self.features))
        nn.init.kaiming_uniform_(self.memory)
        self.memory = nn.Parameter(self.memory)

        self.Cosine_Similiarity = nn.CosineSimilarity(dim=2)
        self.sparse = sparse
        if self.sparse:
            self.threshold = threshold or 1 / self.MEM_DIM
            self.epsilon = 1e-12

        self.entropy_loss_coef = entropy_loss_coef

    def forward(self, x):
        B = x.shape[0]
        z = x.view(B, -1)
        features = z.shape[1]
        assert self.features == features

        ex_mem = self.memory.unsqueeze(0).repeat(B, 1, 1)  # [b, mem_dim, fea]
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)  # [b, mem_dim, fea]

        mem_logit = self.Cosine_Similiarity(ex_z, ex_mem)  # [b, mem_dim]
        mem_weight = mem_logit.softmax(dim=1)  # [b, num_mem]

        # 稀疏寻址
        if self.sparse:
            mem_weight = ((F.relu(mem_weight - self.threshold) * mem_weight)
                          / (torch.abs(mem_weight - self.threshold) + self.epsilon))
            mem_weight = F.normalize(mem_weight, p=1, dim=1)

        z_hat = torch.matmul(mem_weight, self.memory)
        mem_loss = self.EntropyLoss(mem_weight)
        out = z_hat.view(x.shape).contiguous()

        return out, mem_loss

    def EntropyLoss(self, mem_weight):
        entropy_loss = -mem_weight * torch.log(mem_weight + 1e-12)
        entropy_loss = entropy_loss.sum()
        entropy_loss *= self.entropy_loss_coef
        return entropy_loss


class MaskMemoryBlock(nn.Module):
    def __init__(self, num_memories, features, sparse=True, num_classes=10,
                 threshold=None, entropy_loss_coef=0.0002):
        super(MaskMemoryBlock, self).__init__()
        self.num_memories = num_memories
        self.features = features
        self.num_classes = num_classes

        self.memory = torch.zeros((self.num_memories, self.features))
        nn.init.kaiming_uniform_(self.memory)
        self.memory = nn.Parameter(self.memory)

        self.Cosine_Similiarity = nn.CosineSimilarity(dim=2)

        self.sparse = sparse
        if self.sparse:
            self.threshold = threshold or 1 / self.num_memories
            self.epsilon = 1e-12
            self.hardshrink = nn.Hardshrink(lambd=self.threshold)

        self.entropy_loss_coef = entropy_loss_coef

    def forward(self, z, labels):
        batch, c, h, w = z.shape
        z = z.view(batch, -1)
        assert self.features == z.shape[1]

        ex_mem = self.memory.unsqueeze(0).repeat(batch, 1, 1)  # [b, mem_dim, fea]
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)  # [b, mem_dim, fea]

        mem_logit = self.Cosine_Similiarity(ex_z, ex_mem)  # [b, mem_dim]
        mem_weight = mem_logit.softmax(dim=1)  # [b, num_mem]

        # Masking using one hot encoding scheme over memory slots.
        m1, m2 = self.masking(labels)  # Generating Mask
        masked_mem_weight_target = mem_weight * m1  # Masking target class
        masked_mem_weight_non_target = mem_weight * m2  # Masking non-target class

        if self.sparse:
            # Unmask Weight Target Class
            masked_mem_weight_target = self.hardshrink(masked_mem_weight_target)
            masked_mem_weight_target = masked_mem_weight_target / masked_mem_weight_target.norm(p=1, dim=1).unsqueeze(1).expand(batch, self.num_memories)
            # Mask Weight Non-target Class
            masked_mem_weight_non_target = self.hardshrink(masked_mem_weight_non_target)
            masked_mem_weight_non_target = masked_mem_weight_non_target / masked_mem_weight_non_target.norm(p=1, dim=1).unsqueeze(1).expand(batch, self.num_memories)

        z_hat_target = torch.mm(masked_mem_weight_target, self.memory).view(batch, c, h, w).contiguous()
        z_hat_non_target = torch.mm(masked_mem_weight_non_target, self.memory).view(batch, c, h, w).contiguous()
        target_memloss = self.EntropyLoss(masked_mem_weight_target)
        non_target_memloss = self.EntropyLoss(masked_mem_weight_non_target)
        return dict(z_hat_target=z_hat_target, z_hat_non_target=z_hat_non_target,
                    target_memloss=target_memloss, non_target_memloss=non_target_memloss)

    def EntropyLoss(self, mem_weight):
        entropy_loss = -mem_weight * torch.log(mem_weight + 1e-12)
        entropy_loss = entropy_loss.sum()
        entropy_loss *= self.entropy_loss_coef
        return entropy_loss

    def masking(self, label):
        memoryPerClass = self.num_memories // self.num_classes
        batch_size = len(label)

        mask1 = torch.zeros(batch_size, self.num_memories)
        mask2 = torch.ones(batch_size, self.num_memories)
        ones = torch.ones(memoryPerClass)
        zeros = torch.zeros(memoryPerClass)

        for i in range(batch_size):
            lab = torch.arange(memoryPerClass * label[i], memoryPerClass * (label[i] + 1), dtype=torch.long)
            if lab.nelement() == 0:
                print("Label tensor empty in the memory module.")
            else:
                mask1[i, lab] = ones
                mask2[i, lab] = zeros
        return mask1.to(label.device), mask2.to(label.device)


class Encoder(nn.Module):
    def __init__(self, *, in_ch, ch, z_channels, resolution, num_res_blocks, ch_mult=(1, 2, 4, 8),
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, double_z=False, use_linear_attn=False,
                 attn_type="vanilla", ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_ch

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_ch,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, out_ch, ch, z_channels, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 resolution, attn_resolutions, dropout=0.0, resamp_with_conv=True, give_pre_end=False,
                 tanh_out=False, use_linear_attn=False, attn_type="vanilla"):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


def testEncoder():
    x = torch.randn((8, 1, 64, 64))
    e = Encoder(
        in_ch=1,
        ch=128,
        z_channels=4,
        ch_mult=[1, 2],
        resolution=64,
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
    )
    z = e(x)
    print(z.shape)


def testDecoder():
    x = torch.randn((8, 4, 16, 16))
    d = Decoder(
        ch=128,
        out_ch=1,
        z_channels=4,
        ch_mult=[1, 2, 4],
        resolution=64,
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0
    )
    out = d(x)
    print(out.shape)


def testMemoryBlock():
    x = torch.rand((8, 4, 16, 16)).to("cuda")
    m = MemoryBlock(num_memories=100, features=1024).to("cuda")
    out = m(x)[0]
    print(out.shape)


if __name__ == '__main__':
    # testEncoder()
    # testMemoryEncoder()
    # testMemoryDecoder()
    testMemoryBlock()