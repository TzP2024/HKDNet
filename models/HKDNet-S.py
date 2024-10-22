import torch
import torch.nn.functional as F
from torch.optim import Adam
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn
from torchvision import models
from COA_RGBD_SOD.Shunted_Transformer_master.SSA import *
from thop import profile
from COA_RGBD_SOD.al.models.mix_transformer import *
import numpy as np
from timm.models.layers import DropPath, trunc_normal_

def upsample(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)

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


class EncoderAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.2, attn_drop=0.2, drop_path=0.5,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, Group=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.Group = Group
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, Group=self.Group)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1---Important！！！
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

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, d):
        # x: B,C,H,W\
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        b, c, h, w = d.shape
        d = d.flatten(2).transpose(1, 2)
        H, W = self.input_resolution
        B, L, C = x.shape
        b, l, c = d.shape
        assert L == H * W, "input feature has wrong size"

        shortcut_x = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        shortcut_d = d
        d = self.norm1(d)
        d = d.view(b, H, W, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_d = torch.roll(d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_d = d

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        d_windows = window_partition(shifted_d, self.window_size)
        d_windows = d_windows.view(-1, self.window_size * self.window_size, c)
        # W-MSA/SW-MSA

        if self.Group != None:
            attn_windows_x, attn_windows_d, attn_xi, attn_di = self.attn(x_windows, d_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows_x.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

            attn_windows_d = attn_windows_d.view(-1, self.window_size, self.window_size, c)
            shifted_d = window_reverse(attn_windows_d, self.window_size, H, W)

            attn_xi = attn_xi.view(-1, self.window_size, self.window_size, C)
            shifted_xi = window_reverse(attn_xi, self.window_size, H, W)

            attn_di = attn_di.view(-1, self.window_size, self.window_size, c)
            shifted_di = window_reverse(attn_di, self.window_size, h, w)

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                d = torch.roll(shifted_d, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                xi = torch.roll(shifted_xi, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                di = torch.roll(shifted_di, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

            else:
                x = shifted_x
                d = shifted_d
                xi = shifted_xi
                di = shifted_di

            x = x.view(B, H * W, C)
            d = d.view(b, H * W, c)
            xi = xi.view(B, H * W, C)
            di = di.view(b, H * W, c)

            # FFN
            x = shortcut_x + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            d = shortcut_d + self.drop_path(d)
            d = d + self.drop_path(self.mlp(self.norm2(d)))

            xi = shortcut_x + self.drop_path(xi)
            xi = xi + self.drop_path(self.mlp(self.norm2(xi)))

            di = shortcut_d + self.drop_path(di)
            di = di + self.drop_path(self.mlp(self.norm2(di)))

            x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
            d = d.view(b, int(np.sqrt(l)), int(np.sqrt(l)), -1).permute(0, 3, 1, 2).contiguous()

            xi = xi.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
            di = di.view(b, int(np.sqrt(l)), int(np.sqrt(l)), -1).permute(0, 3, 1, 2).contiguous()
            # print('FFN',x.shape)
            return x, d, xi, di

        else:
            attn_windows_x, attn_windows_d = self.attn(x_windows, d_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows_x.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

            attn_windows_d = attn_windows_d.view(-1, self.window_size, self.window_size, c)
            shifted_d = window_reverse(attn_windows_d, self.window_size, H, W)

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                d = torch.roll(shifted_d, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

            else:
                x = shifted_x
                d = shifted_d

            x = x.view(B, H * W, C)
            d = d.view(b, H * W, c)

            # FFN
            x = shortcut_x + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            d = shortcut_d + self.drop_path(d)
            d = d + self.drop_path(self.mlp(self.norm2(d)))

            x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
            d = d.view(b, int(np.sqrt(l)), int(np.sqrt(l)), -1).permute(0, 3, 1, 2).contiguous()
            return x, d

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., Group=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.Group = Group

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 5, bias=qkv_bias)
        self.qkv1 = nn.Linear(dim, dim * 4, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, d, mask=None):

        if self.Group != None:
            B_, N, C = x.shape
            qkv_r = self.qkv(x).reshape(B_, N, 5, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q_r, k_r, v_r, q_rd, q_ri = qkv_r[0], qkv_r[1], qkv_r[2], qkv_r[3], qkv_r[4]  # make torchscript happy (cannot use tensor as tuple)

            b_, n, c = d.shape
            qkv_d = self.qkv(d).reshape(b_, n, 5, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
            q_d, k_d, v_d, q_dr, q_di = qkv_d[0], qkv_d[1], qkv_d[2], qkv_d[3], qkv_d[4]

            q = q_r * self.scale
            attn = (q @ k_r.transpose(-2, -1))

            attn_rd = ((q_dr * self.scale) @ k_r.transpose(-2, -1))

            attn_ri = ((q_ri * self.scale) @ k_r.transpose(-2, -1))
            attn_di =  ((q_di * self.scale) @ k_d.transpose(-2, -1))

            qd = q_d * self.scale
            attn_d = (qd @ k_d.transpose(-2, -1))

            attn_dr = ((q_rd * self.scale) @ k_d.transpose(-2, -1))

            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)

                attn_d = attn_d.view(b_ // nW, nW, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                attn_d = attn_d.view(-1, self.num_heads, n, n)
                attn_d = self.softmax(attn_d)

                attn_rd = attn_rd.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn_rd = attn_rd.view(-1, self.num_heads, N, N)
                attn_rd = self.softmax(attn_rd)

                attn_dr = attn_dr.view(b_ // nW, nW, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                attn_dr = attn_dr.view(-1, self.num_heads, n, n)
                attn_dr = self.softmax(attn_dr)

                attn_ri = attn_ri.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn_ri = attn_ri.view(-1, self.num_heads, N, N)
                attn_ri = self.softmax(attn_ri)

                attn_di = attn_di.view(b_ // nW, nW, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                attn_di = attn_di.view(-1, self.num_heads, n, n)
                attn_di = self.softmax(attn_di)

            else:
                attn = self.softmax(attn)
                attn_d = self.softmax(attn_d)
                attn_rd = self.softmax(attn_rd)
                attn_dr = self.softmax(attn_dr)
                attn_ri = self.softmax(attn_ri)
                attn_di = self.softmax(attn_di)

            attn = self.attn_drop(attn)
            attn_d = self.attn_drop(attn_d)
            attn_rd = self.attn_drop(attn_rd)
            attn_dr = self.attn_drop(attn_dr)
            attn_ri = self.attn_drop(attn_ri)
            attn_di = self.attn_drop(attn_di)

            x = (attn @ v_r).transpose(1, 2).reshape(B_, N, C)
            d = (attn_d @ v_d).transpose(1, 2).reshape(b_, n, c)
            x_d = (attn_rd @ v_r).transpose(1, 2).reshape(B_, N, C)
            d_x = (attn_dr @ v_d).transpose(1, 2).reshape(b_, n, c)
            x_i = (attn_ri @ v_r).transpose(1, 2).reshape(B_, N, C)
            d_i = (attn_di @ v_d).transpose(1, 2).reshape(b_, n, c)
            x = x + x_d
            d = d_x + d
            x = self.proj(x)
            x = self.proj_drop(x)
            d = self.proj(d)
            d = self.proj_drop(d)
            x_i = self.proj(x_i)
            x_i = self.proj_drop(x_i)
            d_i = self.proj(d_i)
            d_i = self.proj_drop(d_i)

            return x, d, x_i, d_i

        else:

            B_, N, C = x.shape
            qkv_r = self.qkv1(x).reshape(B_, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q_r, k_r, v_r, q_rd = qkv_r[0], qkv_r[1], qkv_r[2], qkv_r[3]  # make torchscript happy (cannot use tensor as tuple)

            b_, n, c = d.shape
            qkv_d = self.qkv1(d).reshape(b_, n, 4, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
            q_d, k_d, v_d, q_dr, = qkv_d[0], qkv_d[1], qkv_d[2], qkv_d[3]

            q = q_r * self.scale
            attn = (q @ k_r.transpose(-2, -1))

            attn_rd = ((q_dr * self.scale) @ k_r.transpose(-2, -1))

            qd = q_d * self.scale
            attn_d = (qd @ k_d.transpose(-2, -1))

            attn_dr = ((q_rd * self.scale) @ k_d.transpose(-2, -1))

            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)

                attn_d = attn_d.view(b_ // nW, nW, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                attn_d = attn_d.view(-1, self.num_heads, n, n)
                attn_d = self.softmax(attn_d)

                attn_rd = attn_rd.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn_rd = attn_rd.view(-1, self.num_heads, N, N)
                attn_rd = self.softmax(attn_rd)

                attn_dr = attn_dr.view(b_ // nW, nW, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                attn_dr = attn_dr.view(-1, self.num_heads, n, n)
                attn_dr = self.softmax(attn_dr)


            else:
                attn = self.softmax(attn)
                attn_d = self.softmax(attn_d)
                attn_rd = self.softmax(attn_rd)
                attn_dr = self.softmax(attn_dr)

            attn = self.attn_drop(attn)
            attn_d = self.attn_drop(attn_d)
            attn_rd = self.attn_drop(attn_rd)
            attn_dr = self.attn_drop(attn_dr)

            x = (attn @ v_r).transpose(1, 2).reshape(B_, N, C)
            d = (attn_d @ v_d).transpose(1, 2).reshape(b_, n, c)
            x_d = (attn_rd @ v_r).transpose(1, 2).reshape(B_, N, C)
            d_x = (attn_dr @ v_d).transpose(1, 2).reshape(b_, n, c)
            x = x + x_d
            d = d_x + d
            x = self.proj(x)
            x = self.proj_drop(x)
            d = self.proj(d)
            d = self.proj_drop(d)

            return x, d

class CISA(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, depths, Group):
        super(CISA, self).__init__()
        self.EA = EncoderAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, Group=Group)
        self.depths = depths
        self.Group = Group

    def forward(self, r, d):
        if self.Group !=None:
            for i in range(self.depths):
                r, d, ri, di = self.EA(r, d)
            return r, d, ri, di
        else:
            for i in range(self.depths):
                r, d = self.EA(r, d)
            return r, d


class CDAM(nn.Module):
    def __init__(self, in_channels, input_resolution, num_heads, window_size, depths, Group):
        super(CDAM, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.SpatialAttention = SpatialAttention()
        self.conv = DepthWiseConv(in_channels * 5, in_channels)
        self.EA = CISA(dim=in_channels, input_resolution=to_2tuple(input_resolution), num_heads=num_heads, window_size=window_size,
                       depths=depths, Group=Group)
        self.convr1 = DepthWiseConv(in_channels * 2, in_channels)
        self.convd1 = DepthWiseConv(in_channels * 2, in_channels)

    def forward(self, r, d, rd):
        R, D = self.EA(r, d)
        R = self.convr1(torch.cat([r, R], dim=1)) + r
        D = self.convd1(torch.cat([d, D], dim=1)) + d
        Con = R + D
        Diff = torch.abs(R - D)
        f = (0.5 * r + 0.5 * d) * rd
        Con = Con * self.SpatialAttention(Con)
        Diff = Diff * self.SpatialAttention(Diff)
        f = f * self.SpatialAttention(f)
        F = self.conv(torch.cat([r, d, Con, Diff, f], dim=1))
        F = F * self.sigmoid(F) + F
        return F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.point_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)

        return out


class RGBD_sal(nn.Module):

    def __init__(self):
        super(RGBD_sal, self).__init__()

        self.conv_RGB = shunted_t()
        self.conv_depth = shunted_t()

        self.channels = [64, 128, 256, 512]

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up16 = nn.UpsamplingBilinear2d(scale_factor=16)
        self.up32 = nn.UpsamplingBilinear2d(scale_factor=32)


        self.conv44 = DepthWiseConv(self.channels[3] * 2, self.channels[3])
        self.conv4_3 = DepthWiseConv(self.channels[3], self.channels[2])
        self.conv3_2 = DepthWiseConv(self.channels[2], self.channels[1])
        self.conv2_1 = DepthWiseConv(self.channels[1], self.channels[0])

        self.conv1_44 = DepthWiseConv(self.channels[3] * 2, self.channels[3])
        self.conv1_34 = DepthWiseConv(self.channels[2] + self.channels[3], self.channels[2])
        self.conv1_23 = DepthWiseConv(self.channels[1] + self.channels[2], self.channels[1])
        self.conv1_12 = DepthWiseConv(self.channels[0] + self.channels[1], self.channels[0])
        self.convd1_44 = DepthWiseConv(self.channels[3] * 2, self.channels[3])
        self.convd1_34 = DepthWiseConv(self.channels[2] + self.channels[3], self.channels[2])
        self.convd1_23 = DepthWiseConv(self.channels[1] + self.channels[2], self.channels[1])
        self.convd1_12 = DepthWiseConv(self.channels[0] + self.channels[1], self.channels[0])


        self.conv4 = DepthWiseConv(self.channels[3], 1)
        self.conv3 = DepthWiseConv(self.channels[2], 1)
        self.conv2 = DepthWiseConv(self.channels[1], 1)
        self.conv1 = DepthWiseConv(self.channels[0], 1)

        self.fusion1 = CDAM(self.channels[0], 64, 4, 8, 4, None)
        self.fusion2 = CDAM(self.channels[1], 32, 8, 8, 8, None)
        self.fusion3 = CDAM(self.channels[2], 16, 16, 8, 8, None)
        self.fusion4 = CDAM(self.channels[3], 8, 32, 8, 4, None)

    def forward(self, x, depth):

        e1_rgb, e2_rgb, e3_rgb, e4_rgb = self.conv_RGB(x)
        e1_depth, e2_depth, e3_depth, e4_depth = self.conv_depth(depth)

        e4 = self.conv44(torch.cat([e4_rgb, e4_depth], dim=1))

        e4_r = self.conv1_44(torch.cat([e4_rgb, e4_depth], dim=1))
        e4_d = self.convd1_44(torch.cat([e4_depth, e4_rgb], dim=1))

        e3_r = self.conv1_34(torch.cat([e3_rgb, self.up2(e4_r)], dim=1))
        e3_d = self.convd1_34(torch.cat([e3_depth, self.up2(e4_d)], dim=1))

        e2_r = self.conv1_23(torch.cat([e2_rgb, self.up2(e3_r)], dim=1))
        e2_d = self.convd1_23(torch.cat([e2_depth, self.up2(e3_d)], dim=1))

        e1_r = self.conv1_12(torch.cat([e1_rgb, self.up2(e2_r)], dim=1))
        e1_d = self.convd1_12(torch.cat([e1_depth, self.up2(e2_d)], dim=1))

        f4 = self.fusion4(e4_r, e4_d, e4)
        f3 = self.fusion3(e3_r, e3_d, self.up2(self.conv4_3(f4)))
        f2 = self.fusion2(e2_r, e2_d, self.up2(self.conv3_2(f3)))
        f1 = self.fusion1(e1_r, e1_d, self.up2(self.conv2_1(f2)))

        F4 = self.up32(self.conv4(f4))
        F3 = self.up16(self.conv3(f3))
        F2 = self.up8(self.conv2(f2))
        F1 = self.up4(self.conv1(f1))

        return F1, F2, F3, F4, f1, f2, f3, f4, e4_rgb, e4_depth, e1_r, e2_r, e3_r, e4_r, e1_d, e2_d, e3_d, e4_d

    def load_pre(self, pre_model):
        save_model = torch.load(pre_model)
        model_dict_r = self.conv_RGB.state_dict()
        state_dict_r = {k:v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_r.update(state_dict_r)
        self.conv_RGB.load_state_dict(model_dict_r)
        print(f"RGB Loading pre_model ${pre_model}")

        save_model = torch.load(pre_model)
        model_dict_d = self.conv_RGB.state_dict()
        state_dict_d = {k: v for k, v in save_model.items() if k in model_dict_d.keys()}
        model_dict_d.update(state_dict_d)
        self.conv_depth.load_state_dict(model_dict_d)
        print(f"Depth Loading pre_model ${pre_model}")

if __name__ == "__main__":
    model = RGBD_sal()

    def print_network(model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print("The number of parameters:{}M".format(num_params / 1000000))

    model.train()
    depth = torch.randn(5, 3, 256, 256)
    input = torch.randn(5, 3, 256, 256)
    # model.load_pre('/home/map/Alchemist/COA/COA_RGBD_SOD/al/Pretrain/segformer.b4.512x512.ade.160k.pth')
    flops, params = profile(model, inputs=(input, depth))
    print("the number of Flops {} G ".format(flops / 1e9))
    print("the number of Parameter {}M ".format(params / 1e6)) #1048576

    print_network(model, 'ccc')

    # out = model(input, depth)
    # for i in range(len(out)):
    #     print(out[i])
