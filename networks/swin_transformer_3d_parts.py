import torch
import torch.nn as nn
from torch import einsum
import numpy as np
from einops import rearrange


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor=2):
        super().__init__()
        self.df = downscaling_factor
        self.linear = nn.Linear(in_channels * (downscaling_factor ** 3), out_channels)

    def forward(self, x):
        merged_x = rearrange(x, 'b c (n_d df_d) (n_h df_h) (n_w df_w) -> b n_d n_h n_w (df_d df_h df_w c)',
                             df_d=self.df, df_h=self.df, df_w=self.df)

        linear_x = self.linear(merged_x)

        return linear_x


def create_mask(window_size, displacement, top_bottom, upper_lower, left_right):
    mask = torch.zeros(window_size ** 3, window_size ** 3)

    mask = rearrange(mask, '(z1 y1 x1) (z2 y2 x2) -> z1 y1 x1 z2 y2 x2',
                     z1=window_size, y1=window_size, z2=window_size, y2=window_size)

    if top_bottom:
        mask[:, :, -displacement:, :, :, :-displacement] = float('-inf')
        mask[:, :, :-displacement, :, :, -displacement:] = float('-inf')
    elif upper_lower:
        mask[:, -displacement:, :, :, :-displacement, :] = float('-inf')
        mask[:, :-displacement, :, :, -displacement:, :] = float('-inf')
    elif left_right:
        mask[:, :, -displacement:, :, :, :-displacement] = float('-inf')
        mask[:, :, :-displacement, :, :, -displacement:] = float('-inf')

    mask = rearrange(mask, 'z1 y1 x1 z2 y2 x2 -> (z1 y1 x1) (z2 y2 x2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(
        np.array([[z, y, x] for z in range(window_size) for y in range(window_size) for x in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowMSA(nn.Module):

    def __init__(self, embed_dim, num_heads, window_size, shifted, rel_pos_embed=True):
        super().__init__()

        head_dim = embed_dim // num_heads

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.shifted = shifted
        self.rel_pos_embed = rel_pos_embed

        if self.shifted:
            self.displacement = window_size // 2

            self.top_bottom_mask = nn.Parameter(create_mask(window_size=window_size, displacement=self.displacement,
                                                            top_bottom=True, upper_lower=False, left_right=False),
                                                requires_grad=False)

            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=self.displacement,
                                                             top_bottom=False, upper_lower=True, left_right=False),
                                                 requires_grad=False)

            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=self.displacement,
                                                            top_bottom=False, upper_lower=False, left_right=True),
                                                requires_grad=False)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)

        if self.rel_pos_embed:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embed = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embed = nn.Parameter(torch.randn(window_size ** 3, window_size ** 3))

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        if self.shifted:
            x = torch.roll(x, shifts=(-self.displacement, -self.displacement, -self.displacement), dims=(1, 2, 3))

        b, n_z, n_y, n_x, _, h = *x.shape, self.num_heads

        qkv = self.qkv(x).chunk(3, dim=-1)

        nw_z = n_z // self.window_size
        nw_y = n_y // self.window_size
        nw_x = n_x // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_z w_z) (nw_y w_y) (nw_x w_x) (h d) -> b h (nw_z nw_y nw_x) (w_z w_y w_x) d',
                                h=h, w_z=self.window_size, w_y=self.window_size, w_x=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.rel_pos_embed:
            dots += self.pos_embed[self.relative_indices[:, :, 0],
                                   self.relative_indices[:, :, 1],
                                   self.relative_indices[:, :, 2]]
        else:
            dots += self.pos_embed

        if self.shifted:
            dots = rearrange(dots, 'b h (nw_z nw_y nw_x) i j -> b h nw_x nw_y nw_z i j',
                             nw_z=nw_z, nw_y=nw_y)

            dots[:, :, :, :, -1] += self.top_bottom_mask

            dots = rearrange(dots, 'b h nw_x nw_y nw_z i j -> b h nw_x nw_z nw_y i j')
            dots[:, :, :, :, -1] += self.upper_lower_mask

            dots = rearrange(dots, 'b h nw_x nw_z nw_y i j -> b h nw_y nw_z nw_x i j')
            dots[:, :, :, :, -1] += self.left_right_mask

            dots = rearrange(dots, 'b h nw_y nw_z nw_x i j -> b h (nw_z nw_y nw_x) i j')

        attn = self.softmax(dots)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)

        out = rearrange(out, 'b h (nw_z nw_y nw_x) (w_z w_y w_x) d -> b (nw_z w_z) (nw_y w_y) (nw_x w_x) (h d)',
                        h=h, w_z=self.window_size, w_y=self.window_size, w_x=self.window_size,
                        nw_z=nw_z, nw_y=nw_y, nw_x=nw_x)

        out = self.proj(out)

        if self.shifted:
            out = torch.roll(out, shifts=(self.displacement, self.displacement, self.displacement), dims=(1, 2, 3))

        return out


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class SwinTBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, window_size, shifted, mlp_ratio):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=embed_dim)
        self.msa = WindowMSA(embed_dim=embed_dim, num_heads=num_heads,
                             window_size=window_size, shifted=shifted)
        self.mlp = MLP(embed_dim=embed_dim, mlp_hidden_dim=int(embed_dim * mlp_ratio))

    def forward(self, z_0):
        z_hat_1 = self.msa(self.ln(z_0)) + z_0

        z_1 = self.mlp(self.ln(z_hat_1)) + z_hat_1

        return z_1
