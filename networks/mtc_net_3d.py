import torch
import torch.nn as nn

from utils.configuration import config
from networks.swin_transformer_3d_parts import SwinTBlock


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm=config["norm"]):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if norm == "Batch Normalization":
            self.norm = nn.BatchNorm3d(num_features=out_channels)
        elif norm == "Instance Normalization":
            self.norm = nn.InstanceNorm3d(num_features=out_channels)
        elif norm == "Group Normalization":
            self.norm = nn.GroupNorm(num_groups=config["num_groups"], num_channels=out_channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        norm = self.norm(conv)
        act = self.act(norm)

        return act


class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        conv_1 = self.conv_block_1(x)
        conv_2 = self.conv_block_2(conv_1)

        return conv_2


class DownsamplingBlock(nn.Module):

    def __init__(self, channels, downsampling=config["downsampling"]):
        super().__init__()
        if downsampling == "Max pooling":
            self.dn_smpl = nn.MaxPool3d(kernel_size=2, stride=2)
        elif downsampling == "Strided convolution":
            self.dn_smpl = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.dn_smpl(x)


class UpsamplingBlock(nn.Module):

    def __init__(self, channels, upsampling=config["upsampling"]):
        super().__init__()
        if upsampling == "Nearest neighbor unpooling":
            self.up_smpl = nn.Upsample(scale_factor=2, mode='nearest')
        elif upsampling == "Trilinear upsampling":
            self.up_smpl = nn.Upsample(scale_factor=2, mode='trilinear')
        elif upsampling == "Strided transpose convolution":
            self.up_smpl = nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.up_smpl(x)


class Resize(nn.Module):

    def __init__(self, in_channels, out_channels, enc_lvl, mtc_lvl):
        super().__init__()
        if enc_lvl > mtc_lvl:
            factor = 2 ** (enc_lvl - mtc_lvl)
            self.resize = nn.ConvTranspose3d(in_channels, out_channels,
                                             kernel_size=factor, stride=factor, padding=0)
        elif enc_lvl < mtc_lvl:
            factor = 2 ** (mtc_lvl - enc_lvl)
            self.resize = nn.Conv3d(in_channels, out_channels,
                                    kernel_size=factor + 1, stride=factor, padding=factor // 2)

    def forward(self, x):
        resized_feature = self.resize(x)
        return resized_feature


class MTCBlock(nn.Module):

    def __init__(self, in_channels, embed_dim, num_heads,
                 window_size=config["window_size"], mlp_ratio=config["mlp_ratio"], norm=config["norm"]):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0)

        if norm == "Batch Normalization":
            self.norm = nn.BatchNorm3d(num_features=embed_dim)
        elif norm == "Instance Normalization":
            self.norm = nn.InstanceNorm3d(num_features=embed_dim)
        elif norm == "Group Normalization":
            self.norm = nn.GroupNorm(num_groups=config["num_groups"], num_channels=embed_dim)

        self.relu = nn.ReLU(inplace=True)

        self.regular_swin_t_block = SwinTBlock(embed_dim=embed_dim, num_heads=num_heads,
                                               window_size=window_size, shifted=False, mlp_ratio=mlp_ratio)
        self.shifted_swin_t_block = SwinTBlock(embed_dim=embed_dim, num_heads=num_heads,
                                               window_size=window_size, shifted=True, mlp_ratio=mlp_ratio)

    def forward(self, in_list):
        x = torch.cat(in_list, dim=1)

        conv = self.conv(x)
        norm = self.norm(conv)
        y = self.relu(norm)

        swin_t_blocks_in = y.permute(0, 2, 3, 4, 1)
        regular_swin_t_block = self.regular_swin_t_block(swin_t_blocks_in)
        shifted_swin_t_block = self.shifted_swin_t_block(regular_swin_t_block)
        swin_t_blocks_out = shifted_swin_t_block.permute(0, 4, 1, 2, 3)

        z = swin_t_blocks_out + y

        return z


class MTCNet3D(nn.Module):

    def __init__(self, in_channels, out_channels, num_levels=config["num_levels"],
                 base_channels=config["base_channels"], base_embed_dim=48, base_num_heads=3):
        super().__init__()

        num_channels = base_channels
        num_channels_lst = [num_channels]
        embed_dim = base_embed_dim
        embed_dim_lst = [embed_dim]
        num_heads = base_num_heads
        num_heads_lst = [num_heads]
        for _ in range(2, num_levels + 1):
            num_channels *= 2
            num_channels_lst.append(num_channels)
            embed_dim *= 2
            embed_dim_lst.append(embed_dim)
            num_heads *= 2
            num_heads_lst.append(num_heads)

        enc_lvl = 1
        self.enc_block_1 = CNNBlock(in_channels=in_channels, out_channels=num_channels_lst[enc_lvl - 1])
        self.dn_smpl_1 = DownsamplingBlock(channels=num_channels_lst[enc_lvl - 1])

        enc_lvl = 2
        self.enc_block_2 = CNNBlock(in_channels=num_channels_lst[enc_lvl - 2],
                                    out_channels=num_channels_lst[enc_lvl - 1])
        self.dn_smpl_2 = DownsamplingBlock(channels=num_channels_lst[enc_lvl - 1])

        enc_lvl = 3
        self.enc_block_3 = CNNBlock(in_channels=num_channels_lst[enc_lvl - 2],
                                    out_channels=num_channels_lst[enc_lvl - 1])
        self.dn_smpl_3 = DownsamplingBlock(channels=num_channels_lst[enc_lvl - 1])

        enc_lvl = 4
        self.enc_block_4 = CNNBlock(in_channels=num_channels_lst[enc_lvl - 2],
                                    out_channels=num_channels_lst[enc_lvl - 1])
        self.dn_smpl_4 = DownsamplingBlock(channels=num_channels_lst[enc_lvl - 1])

        bottleneck_lvl = 5
        self.bottleneck_block = CNNBlock(in_channels=num_channels_lst[bottleneck_lvl - 2],
                                         out_channels=num_channels_lst[bottleneck_lvl - 1])

        mtc_lvl = 1
        self.resized_feature_2_1 = Resize(in_channels=num_channels_lst[1], out_channels=num_channels_lst[mtc_lvl - 1],
                                          enc_lvl=2, mtc_lvl=mtc_lvl)
        self.resized_feature_3_1 = Resize(in_channels=num_channels_lst[2], out_channels=num_channels_lst[mtc_lvl - 1],
                                          enc_lvl=3, mtc_lvl=mtc_lvl)
        self.resized_feature_4_1 = Resize(in_channels=num_channels_lst[3], out_channels=num_channels_lst[mtc_lvl - 1],
                                          enc_lvl=4, mtc_lvl=mtc_lvl)
        self.mtc_block_1 = MTCBlock(in_channels=num_channels_lst[mtc_lvl - 1] * (num_levels - 1),
                                    embed_dim=embed_dim_lst[mtc_lvl - 1],
                                    num_heads=num_heads_lst[mtc_lvl - 1])

        mtc_lvl = 2
        self.resized_feature_1_2 = Resize(in_channels=num_channels_lst[0], out_channels=num_channels_lst[mtc_lvl - 1],
                                          enc_lvl=1, mtc_lvl=mtc_lvl)
        self.resized_feature_3_2 = Resize(in_channels=num_channels_lst[2], out_channels=num_channels_lst[mtc_lvl - 1],
                                          enc_lvl=3, mtc_lvl=mtc_lvl)
        self.resized_feature_4_2 = Resize(in_channels=num_channels_lst[3], out_channels=num_channels_lst[mtc_lvl - 1],
                                          enc_lvl=4, mtc_lvl=mtc_lvl)
        self.mtc_block_2 = MTCBlock(in_channels=num_channels_lst[mtc_lvl - 1] * (num_levels - 1),
                                    embed_dim=embed_dim_lst[mtc_lvl - 1],
                                    num_heads=num_heads_lst[mtc_lvl - 1])

        mtc_lvl = 3
        self.resized_feature_1_3 = Resize(in_channels=num_channels_lst[0], out_channels=num_channels_lst[mtc_lvl - 1],
                                          enc_lvl=1, mtc_lvl=mtc_lvl)
        self.resized_feature_2_3 = Resize(in_channels=num_channels_lst[1], out_channels=num_channels_lst[mtc_lvl - 1],
                                          enc_lvl=2, mtc_lvl=mtc_lvl)
        self.resized_feature_4_3 = Resize(in_channels=num_channels_lst[3], out_channels=num_channels_lst[mtc_lvl - 1],
                                          enc_lvl=4, mtc_lvl=mtc_lvl)
        self.mtc_block_3 = MTCBlock(in_channels=num_channels_lst[mtc_lvl - 1] * (num_levels - 1),
                                    embed_dim=embed_dim_lst[mtc_lvl - 1],
                                    num_heads=num_heads_lst[mtc_lvl - 1])

        mtc_lvl = 4
        self.resized_feature_1_4 = Resize(in_channels=num_channels_lst[0], out_channels=num_channels_lst[mtc_lvl - 1],
                                          enc_lvl=1, mtc_lvl=mtc_lvl)
        self.resized_feature_2_4 = Resize(in_channels=num_channels_lst[1], out_channels=num_channels_lst[mtc_lvl - 1],
                                          enc_lvl=2, mtc_lvl=mtc_lvl)
        self.resized_feature_3_4 = Resize(in_channels=num_channels_lst[2], out_channels=num_channels_lst[mtc_lvl - 1],
                                          enc_lvl=3, mtc_lvl=mtc_lvl)
        self.mtc_block_4 = MTCBlock(in_channels=num_channels_lst[mtc_lvl - 1] * (num_levels - 1),
                                    embed_dim=embed_dim_lst[mtc_lvl - 1],
                                    num_heads=num_heads_lst[mtc_lvl - 1])

        dec_lvl = 4
        self.up_smpl_4 = UpsamplingBlock(channels=num_channels_lst[dec_lvl])
        self.dec_block_4 = CNNBlock(in_channels=embed_dim_lst[dec_lvl - 1] + num_channels_lst[dec_lvl],
                                    out_channels=num_channels_lst[dec_lvl - 1])

        dec_lvl = 3
        self.up_smpl_3 = UpsamplingBlock(channels=num_channels_lst[dec_lvl])
        self.dec_block_3 = CNNBlock(in_channels=embed_dim_lst[dec_lvl - 1] + num_channels_lst[dec_lvl],
                                    out_channels=num_channels_lst[dec_lvl - 1])

        dec_lvl = 2
        self.up_smpl_2 = UpsamplingBlock(channels=num_channels_lst[dec_lvl])
        self.dec_block_2 = CNNBlock(in_channels=embed_dim_lst[dec_lvl - 1] + num_channels_lst[dec_lvl],
                                    out_channels=num_channels_lst[dec_lvl - 1])

        dec_lvl = 1
        self.up_smpl_1 = UpsamplingBlock(channels=num_channels_lst[dec_lvl])
        self.dec_block_1 = CNNBlock(in_channels=embed_dim_lst[dec_lvl - 1] + num_channels_lst[dec_lvl],
                                    out_channels=num_channels_lst[dec_lvl - 1])

        self.conv_out = nn.Conv3d(num_channels_lst[dec_lvl - 1], out_channels, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        enc_block_1 = self.enc_block_1(input)
        dn_smpl_1 = self.dn_smpl_1(enc_block_1)

        enc_block_2 = self.enc_block_2(dn_smpl_1)
        dn_smpl_2 = self.dn_smpl_2(enc_block_2)

        enc_block_3 = self.enc_block_3(dn_smpl_2)
        dn_smpl_3 = self.dn_smpl_3(enc_block_3)

        enc_block_4 = self.enc_block_4(dn_smpl_3)
        dn_smpl_4 = self.dn_smpl_4(enc_block_4)

        bottleneck_block = self.bottleneck_block(dn_smpl_4)

        resized_feature_2_1 = self.resized_feature_2_1(enc_block_2)
        resized_feature_3_1 = self.resized_feature_3_1(enc_block_3)
        resized_feature_4_1 = self.resized_feature_4_1(enc_block_4)
        mtc_block_1 = self.mtc_block_1([enc_block_1, resized_feature_2_1, resized_feature_3_1, resized_feature_4_1])

        resized_feature_1_2 = self.resized_feature_1_2(enc_block_1)
        resized_feature_3_2 = self.resized_feature_3_2(enc_block_3)
        resized_feature_4_2 = self.resized_feature_4_2(enc_block_4)
        mtc_block_2 = self.mtc_block_2([resized_feature_1_2, enc_block_2, resized_feature_3_2, resized_feature_4_2])

        resized_feature_1_3 = self.resized_feature_1_3(enc_block_1)
        resized_feature_2_3 = self.resized_feature_2_3(enc_block_2)
        resized_feature_4_3 = self.resized_feature_4_3(enc_block_4)
        mtc_block_3 = self.mtc_block_3([resized_feature_1_3, resized_feature_2_3, enc_block_3, resized_feature_4_3])

        resized_feature_1_4 = self.resized_feature_1_4(enc_block_1)
        resized_feature_2_4 = self.resized_feature_2_4(enc_block_2)
        resized_feature_3_4 = self.resized_feature_3_4(enc_block_3)
        mtc_block_4 = self.mtc_block_4([resized_feature_1_4, resized_feature_2_4, resized_feature_3_4, enc_block_4])

        up_smpl_4 = self.up_smpl_4(bottleneck_block)
        skip_4 = torch.cat([mtc_block_4, up_smpl_4], dim=1)
        dec_block_4 = self.dec_block_4(skip_4)

        up_smpl_3 = self.up_smpl_3(dec_block_4)
        skip_3 = torch.cat([mtc_block_3, up_smpl_3], dim=1)
        dec_block_3 = self.dec_block_3(skip_3)

        up_smpl_2 = self.up_smpl_2(dec_block_3)
        skip_2 = torch.cat([mtc_block_2, up_smpl_2], dim=1)
        dec_block_2 = self.dec_block_2(skip_2)

        up_smpl_1 = self.up_smpl_1(dec_block_2)
        skip_1 = torch.cat([mtc_block_1, up_smpl_1], dim=1)
        dec_block_1 = self.dec_block_1(skip_1)

        conv_out = self.conv_out(dec_block_1)
        output = self.softmax(conv_out)

        return output
