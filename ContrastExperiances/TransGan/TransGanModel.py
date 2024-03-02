import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, DropPath, trunc_normal_
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = h_sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., act='hs+se', reduction=4,
                 wo_dp_conv=False, dp_first=False):
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        layers = []
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])

        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError('Activation type {} is not implemented'.format(act))

        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, is_local=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.is_local = is_local
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        if is_local:
            self.conv = LocalityFeedForward(dim, dim, 1, mlp_ratio, act='hs+se', reduction=dim // 4)
        else:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)

        if not self.is_local:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            batch_size, num, embed_dim = x.shape
            x = x.transpose(1, 2).view(batch_size, embed_dim, H, W)
            x = self.conv(x).flatten(2).transpose(1, 2)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, pos_embed,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, is_local=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.pos_embed = pos_embed

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 is_local=is_local
                                 )
            for i in range(depth)])

    def forward(self, x):
        for j, blk in enumerate(self.blocks):
            x = blk(x)
            if j == 0:
                if self.pos_embed is not None:
                    x = self.pos_embed(x, self.input_resolution[0], self.input_resolution[1])
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


def bicubic_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic', align_corners=True)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


class matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1 @ x2
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self,
            input_resolution,
            dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
            drop_path=0., norm_layer=nn.LayerNorm,
            is_local=True
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.is_local = is_local
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        if not is_local:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        else:
            self.conv = LocalityFeedForward(dim, dim, 1, mlp_ratio, act='hs+se', reduction=dim // 4)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if not self.is_local:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            batch_size, num, embed_dim = x.shape
            cls_token, x = torch.split(x, [1, num - 1], dim=1)
            x = x.transpose(1, 2).view(batch_size, embed_dim, self.input_resolution[0], self.input_resolution[1])
            x = self.conv(x).flatten(2).transpose(1, 2)
            x = torch.cat([cls_token, x], dim=1)
        return x


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class SwinTransGenerator(nn.Module):
    def __init__(self, embed_dim=256, bottom_width=8, bottom_height=8, window_size=4, depth=None,
                 is_local=True, is_peg=True):
        super(SwinTransGenerator, self).__init__()
        self.bottom_width = bottom_width
        self.bottom_height = bottom_height
        self.is_local = is_local
        self.is_peg = is_peg
        self.embed_dim = embed_dim
        if depth is None:
            depth = [4, 2, 2, 2]
        self.window_size = 8

        self.l1 = nn.Linear(256, (self.bottom_height * self.bottom_width) * self.embed_dim)
        self.layer1 = BasicLayer(
            dim=embed_dim,
            input_resolution=[self.bottom_height, self.bottom_width],
            depth=depth[0], num_heads=4, window_size=window_size,
            pos_embed=PosCNN(embed_dim, embed_dim) if is_peg else None,
            is_local=is_local
        )

        self.layer2 = BasicLayer(
            dim=embed_dim,
            input_resolution=[self.bottom_height * 2, self.bottom_width * 2],
            depth=depth[1], num_heads=4, window_size=window_size,
            pos_embed=PosCNN(embed_dim, embed_dim) if is_peg else None,
            is_local=is_local
        )

        self.layer3 = BasicLayer(
            dim=embed_dim // 4,
            input_resolution=[self.bottom_height * 4, self.bottom_width * 4],
            depth=depth[2], num_heads=4, window_size=window_size,
            pos_embed=PosCNN(embed_dim // 4, embed_dim // 4) if is_peg else None,
            is_local=is_local
        )

        self.layer4 = BasicLayer(
            dim=embed_dim // 16,
            input_resolution=[self.bottom_height * 8, self.bottom_width * 8],
            depth=depth[3], num_heads=4, window_size=window_size,
            pos_embed=PosCNN(embed_dim // 16, embed_dim // 16) if is_peg else None,
            is_local=is_local
        )

        # self.deconv = nn.Sequential(
        #     nn.Conv2d(self.embed_dim // 16, 1, 1, 1, 0)
        # )
        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim // 16, 1, 3, 1, 1)
        )

        self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        if not is_peg:
            self.pos_embed_1 = nn.Parameter(
                torch.zeros(1, self.bottom_height * self.bottom_width, embed_dim)
            )
            self.pos_embed_2 = nn.Parameter(
                torch.zeros(1, (self.bottom_height * 2) * (self.bottom_width * 2), embed_dim)
            )
            self.pos_embed_3 = nn.Parameter(
                torch.zeros(1, (self.bottom_height * 4) * (self.bottom_width * 4), embed_dim // 4)
            )
            self.pos_embed_4 = nn.Parameter(
                torch.zeros(1, (self.bottom_height * 8) * (self.bottom_width * 8), embed_dim // 16)
            )
            trunc_normal_(self.pos_embed_1, std=.02)
            trunc_normal_(self.pos_embed_2, std=.02)
            trunc_normal_(self.pos_embed_3, std=.02)
            trunc_normal_(self.pos_embed_4, std=.02)

    def forward(self, noise):
        x = self.l1(noise)
        x = x.reshape(-1, self.bottom_width * self.bottom_height, self.embed_dim)
        if not self.is_peg:
            x = x + self.pos_embed_1
        H, W = self.bottom_height, self.bottom_width
        x = self.layer1(x)
        x, H, W = bicubic_upsample(x, H, W)
        if not self.is_peg:
            x = x + self.pos_embed_2
        x = self.layer2(x)
        x, H, W = pixel_upsample(x, H, W)
        if not self.is_peg:
            x = x + self.pos_embed_3
        x = self.layer3(x)
        x, H, W = pixel_upsample(x, H, W)
        if not self.is_peg:
            x = x + self.pos_embed_4
        B, _, C = x.size()
        x = self.layer4(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.deconv(x)
        # x = self.sigmoid(x)
        # x = self.tanh(x)
        return x


class SwinTransDiscriminator(nn.Module):
    def __init__(self,
                 img_height=64, img_width=64, patch_size=4, in_channel=1,
                 embed_dim=512, depth: list = None,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 is_local=True, is_peg=True):
        super(SwinTransDiscriminator, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.window_size = patch_size
        self.is_local = is_local
        self.is_peg = is_peg
        if depth is None:
            depth = [4, 2, 2, 2]
        self.PatchEmbed_1 = nn.Conv2d(in_channel, embed_dim // 4, kernel_size=patch_size, stride=patch_size, padding=0)
        self.PatchEmbed_2 = nn.Conv2d(in_channel, embed_dim // 4, kernel_size=patch_size, stride=patch_size, padding=0)
        self.PatchEmbed_3 = nn.Conv2d(in_channel, embed_dim // 2, kernel_size=patch_size, stride=patch_size, padding=0)

        self.initial_height = img_height // patch_size
        self.initial_width = img_width // patch_size

        if not is_peg:
            num_patches_1 = (img_height // patch_size) * (img_width // patch_size)
            num_patches_2 = (img_height // (2 * patch_size)) * (img_width // (2 * patch_size))
            num_patches_3 = (img_height // (4 * patch_size)) * (img_width // (4 * patch_size))
            self.pos_embed_1 = nn.Parameter(torch.zeros(1, num_patches_1, embed_dim // 4))
            self.pos_embed_2 = nn.Parameter(torch.zeros(1, num_patches_2, embed_dim // 2))
            self.pos_embed_3 = nn.Parameter(torch.zeros(1, num_patches_3, embed_dim))
            trunc_normal_(self.pos_embed_1, std=.02)
            trunc_normal_(self.pos_embed_2, std=.02)
            trunc_normal_(self.pos_embed_3, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks_1 = BasicLayer(
            dim=embed_dim // 4,
            input_resolution=[self.initial_height, self.initial_width],
            depth=depth[0], num_heads=4,
            window_size=self.window_size,
            pos_embed=PosCNN(embed_dim // 4, embed_dim // 4) if is_peg else None,
            is_local=is_local
        )

        self.blocks_2 = BasicLayer(
            dim=embed_dim // 2,
            input_resolution=[self.initial_height // 2, self.initial_width // 2],
            depth=depth[1], num_heads=4,
            window_size=self.window_size,
            pos_embed=PosCNN(embed_dim // 2, embed_dim // 2) if is_peg else None,
            is_local=is_local
        )

        self.blocks_3 = BasicLayer(
            dim=embed_dim,
            input_resolution=[self.initial_height // 4, self.initial_width // 4],
            depth=depth[2], num_heads=4,
            window_size=self.window_size,
            pos_embed=PosCNN(embed_dim, embed_dim) if is_peg else None,
            is_local=is_local
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth[3])]
        self.last_block = nn.Sequential(
            Block(
                input_resolution=[self.initial_height // 4, self.initial_width // 4],
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
                is_local=is_local
            )
        )
        self.norm = norm_layer(embed_dim)

        self.out = nn.Linear(embed_dim, 1)

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x_1 = self.PatchEmbed_1(x).flatten(2).permute(0, 2, 1)
        x_2 = self.PatchEmbed_2(nn.AvgPool2d(2)(x)).flatten(2).permute(0, 2, 1)
        x_3 = self.PatchEmbed_3(nn.AvgPool2d(4)(x)).flatten(2).permute(0, 2, 1)

        if not self.is_peg:
            x_1 = x_1 + self.pos_embed_1
        x = self.pos_drop(x_1)
        B, _, C = x.size()
        x = self.blocks_1(x)

        x = x.permute(0, 2, 1).reshape(B, C, self.initial_height, self.initial_width)
        x = nn.AvgPool2d(2)(x)
        _, _, H, W = x.shape
        x = x.flatten(2)
        x = x.permute(0, 2, 1)
        x = torch.cat([x, x_2], dim=-1)
        if not self.is_peg:
            x = x + self.pos_embed_2
        x = self.blocks_2(x)
        _, _, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = nn.AvgPool2d(2)(x)
        _, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = torch.cat([x, x_3], dim=-1)
        if not self.is_peg:
            x = x + self.pos_embed_3
        x = self.blocks_3(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.last_block(x)
        x = self.norm(x)
        x = self.out(x[:, 0])
        return x


def test_dis():
    x = torch.randn((16, 1, 64, 64))
    d = SwinTransDiscriminator()
    out = d(x)
    print(out.shape)


def test_gen():
    x = torch.randn((16, 256))
    g = SwinTransGenerator(embed_dim=256)
    out = g(x)
    print(out.shape)


if __name__ == '__main__':
    test_gen()
    test_dis()
