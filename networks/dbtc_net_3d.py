import torch
import torch.nn as nn
from einops.layers.torch import Reduce

from utils.configuration import config
from networks.swin_transformer_3d_parts import PatchMerging, SwinTBlock


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


class SwinTCABlock(nn.Module):

    def __init__(self, embed_dim, num_heads, window_size=config["window_size"], mlp_ratio=config["mlp_ratio"]):
        super().__init__()

        self.regular_swin_t_block = SwinTBlock(embed_dim=embed_dim, num_heads=num_heads,
                                               window_size=window_size, shifted=False, mlp_ratio=mlp_ratio)
        self.shifted_swin_t_block = SwinTBlock(embed_dim=embed_dim, num_heads=num_heads,
                                               window_size=window_size, shifted=True, mlp_ratio=mlp_ratio)

        self.conv_1 = nn.Conv3d(embed_dim, embed_dim // 2, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, swin_t_blocks_in):
        regular_swin_t_block = self.regular_swin_t_block(swin_t_blocks_in)
        shifted_swin_t_block = self.shifted_swin_t_block(regular_swin_t_block)

        swin_t_blocks_out = shifted_swin_t_block.permute(0, 4, 1, 2, 3)

        conv_1 = self.conv_1(swin_t_blocks_out)
        relu = self.relu(conv_1)
        conv_2 = self.conv_2(relu)
        a_c = self.sigmoid(conv_2)
        z = a_c * swin_t_blocks_out

        return z


class TCFCBlock(nn.Module):

    def __init__(self, cnn_enc_channels, swintca_channels, out_channels, norm=config["norm"]):
        super().__init__()
        in_channels = cnn_enc_channels + swintca_channels
        self.conv_cat = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if norm == "Batch Normalization":
            self.norm = nn.BatchNorm3d(num_features=out_channels)
        elif norm == "Instance Normalization":
            self.norm = nn.InstanceNorm3d(num_features=out_channels)
        elif norm == "Group Normalization":
            self.norm = nn.GroupNorm(num_groups=config["num_groups"], num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv_s1 = nn.Conv3d(out_channels, 1, kernel_size=1, stride=1, padding=0)
        self.conv_s2 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_c, x_t):
        x = torch.cat([x_c, x_t], dim=1)

        conv_cat = self.conv_cat(x)
        norm_cat = self.norm(conv_cat)
        y = self.relu(norm_cat)

        conv_s1 = self.conv_s1(y)
        conv_s2 = self.conv_s2(conv_s1)
        a_s = self.sigmoid(conv_s2)

        z = a_s * y

        return z


class MCCBlock(nn.Module):

    def __init__(self, cnn_enc_in_channels, swintca_in_channels, num_heads, out_channels, norm=config["norm"]):
        super().__init__()
        self.cnn_enc_block = CNNBlock(in_channels=cnn_enc_in_channels,
                                      out_channels=out_channels)
        self.swintca_block = SwinTCABlock(embed_dim=swintca_in_channels, num_heads=num_heads)

        in_channels = out_channels + swintca_in_channels
        ms_channels = out_channels // 16
        self.conv_1 = nn.Conv3d(in_channels, ms_channels, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv3d(in_channels, ms_channels, kernel_size=5, stride=1, padding=2)
        self.conv_3 = nn.Conv3d(in_channels, ms_channels, kernel_size=7, stride=1, padding=3)

        self.conv_cat = nn.Conv3d(3 * ms_channels, out_channels, kernel_size=1, stride=1, padding=0)
        if norm == "Batch Normalization":
            self.norm = nn.BatchNorm3d(num_features=out_channels)
        elif norm == "Instance Normalization":
            self.norm = nn.InstanceNorm3d(num_features=out_channels)
        elif norm == "Group Normalization":
            self.norm = nn.GroupNorm(num_groups=config["num_groups"], num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv_s = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_l, x_g):
        cnn_enc_block = self.cnn_enc_block(x_l)
        swintca_block = self.swintca_block(x_g)

        x_1 = torch.cat([cnn_enc_block, swintca_block], dim=1)

        conv_1 = self.conv_1(x_1)
        conv_2 = self.conv_2(x_1)
        conv_3 = self.conv_3(x_1)
        x_2 = torch.cat([conv_1, conv_2, conv_3], dim=1)

        conv_cat = self.conv_cat(x_2)
        norm_cat = self.norm(conv_cat)
        y_m = self.relu(norm_cat)

        avg_pooling_layer = Reduce('b c d h w -> b 1 d h w', 'mean')(y_m)
        conv_s = self.conv_s(avg_pooling_layer)
        a_m = self.sigmoid(conv_s)

        z_out = a_m * y_m

        return z_out


class DBTCNet3D(nn.Module):

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
        self.cnn_enc_block_1 = CNNBlock(in_channels=in_channels, out_channels=num_channels_lst[enc_lvl - 1])
        self.dn_smpl_1 = DownsamplingBlock(channels=num_channels_lst[enc_lvl - 1])
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=embed_dim_lst[enc_lvl - 1])

        enc_lvl = 2
        self.cnn_enc_block_2 = CNNBlock(in_channels=num_channels_lst[enc_lvl - 2],
                                        out_channels=num_channels_lst[enc_lvl - 1])
        self.dn_smpl_2 = DownsamplingBlock(channels=num_channels_lst[enc_lvl - 1])
        self.swintca_block_1 = SwinTCABlock(embed_dim=embed_dim_lst[enc_lvl - 2], num_heads=num_heads_lst[enc_lvl - 2])
        self.patch_merging_1 = PatchMerging(in_channels=embed_dim_lst[enc_lvl - 2],
                                            out_channels=embed_dim_lst[enc_lvl - 1])
        self.tcfc_block_1 = TCFCBlock(cnn_enc_channels=num_channels_lst[enc_lvl - 1],
                                      swintca_channels=embed_dim_lst[enc_lvl - 2],
                                      out_channels=num_channels_lst[enc_lvl - 1])

        enc_lvl = 3
        self.cnn_enc_block_3 = CNNBlock(in_channels=num_channels_lst[enc_lvl - 2],
                                        out_channels=num_channels_lst[enc_lvl - 1])
        self.dn_smpl_3 = DownsamplingBlock(channels=num_channels_lst[enc_lvl - 1])
        self.swintca_block_2 = SwinTCABlock(embed_dim=embed_dim_lst[enc_lvl - 2], num_heads=num_heads_lst[enc_lvl - 2])
        self.patch_merging_2 = PatchMerging(in_channels=embed_dim_lst[enc_lvl - 2],
                                            out_channels=embed_dim_lst[enc_lvl - 1])
        self.tcfc_block_2 = TCFCBlock(cnn_enc_channels=num_channels_lst[enc_lvl - 1],
                                      swintca_channels=embed_dim_lst[enc_lvl - 2],
                                      out_channels=num_channels_lst[enc_lvl - 1])

        enc_lvl = 4
        self.cnn_enc_block_4 = CNNBlock(in_channels=num_channels_lst[enc_lvl - 2],
                                        out_channels=num_channels_lst[enc_lvl - 1])
        self.dn_smpl_4 = DownsamplingBlock(channels=num_channels_lst[enc_lvl - 1])
        self.swintca_block_3 = SwinTCABlock(embed_dim=embed_dim_lst[enc_lvl - 2], num_heads=num_heads_lst[enc_lvl - 2])
        self.patch_merging_3 = PatchMerging(in_channels=embed_dim_lst[enc_lvl - 2],
                                            out_channels=embed_dim_lst[enc_lvl - 1])
        self.tcfc_block_3 = TCFCBlock(cnn_enc_channels=num_channels_lst[enc_lvl - 1],
                                      swintca_channels=embed_dim_lst[enc_lvl - 2],
                                      out_channels=num_channels_lst[enc_lvl - 1])

        bottleneck_lvl = 5
        self.mcc_block = MCCBlock(cnn_enc_in_channels=num_channels_lst[bottleneck_lvl - 2],
                                  swintca_in_channels=embed_dim_lst[bottleneck_lvl - 2],
                                  num_heads=num_heads_lst[bottleneck_lvl - 2],
                                  out_channels=num_channels_lst[bottleneck_lvl - 1])

        dec_lvl = 4
        self.up_smpl_4 = UpsamplingBlock(channels=num_channels_lst[dec_lvl])
        self.dec_block_4 = CNNBlock(in_channels=num_channels_lst[dec_lvl - 1] + num_channels_lst[dec_lvl],
                                    out_channels=num_channels_lst[dec_lvl - 1])

        dec_lvl = 3
        self.up_smpl_3 = UpsamplingBlock(channels=num_channels_lst[dec_lvl])
        self.dec_block_3 = CNNBlock(in_channels=num_channels_lst[dec_lvl - 1] + num_channels_lst[dec_lvl],
                                    out_channels=num_channels_lst[dec_lvl - 1])

        dec_lvl = 2
        self.up_smpl_2 = UpsamplingBlock(channels=num_channels_lst[dec_lvl])
        self.dec_block_2 = CNNBlock(in_channels=num_channels_lst[dec_lvl - 1] + num_channels_lst[dec_lvl],
                                    out_channels=num_channels_lst[dec_lvl - 1])

        dec_lvl = 1
        self.up_smpl_1 = UpsamplingBlock(channels=num_channels_lst[dec_lvl])
        self.dec_block_1 = CNNBlock(in_channels=num_channels_lst[dec_lvl - 1] + num_channels_lst[dec_lvl],
                                    out_channels=num_channels_lst[dec_lvl - 1])

        self.conv_out = nn.Conv3d(num_channels_lst[dec_lvl - 1], out_channels, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        cnn_enc_block_1 = self.cnn_enc_block_1(input)
        dn_smpl_1 = self.dn_smpl_1(cnn_enc_block_1)
        patch_partition = self.patch_partition(input)

        cnn_enc_block_2 = self.cnn_enc_block_2(dn_smpl_1)
        dn_smpl_2 = self.dn_smpl_2(cnn_enc_block_2)
        swintca_block_1 = self.swintca_block_1(patch_partition)
        patch_merging_1 = self.patch_merging_1(swintca_block_1)
        tcfc_block_1 = self.tcfc_block_1(cnn_enc_block_2, swintca_block_1)

        cnn_enc_block_3 = self.cnn_enc_block_3(dn_smpl_2)
        dn_smpl_3 = self.dn_smpl_3(cnn_enc_block_3)
        swintca_block_2 = self.swintca_block_2(patch_merging_1)
        patch_merging_2 = self.patch_merging_2(swintca_block_2)
        tcfc_block_2 = self.tcfc_block_2(cnn_enc_block_3, swintca_block_2)

        cnn_enc_block_4 = self.cnn_enc_block_4(dn_smpl_3)
        dn_smpl_4 = self.dn_smpl_4(cnn_enc_block_4)
        swintca_block_3 = self.swintca_block_3(patch_merging_2)
        patch_merging_3 = self.patch_merging_3(swintca_block_3)
        tcfc_block_3 = self.tcfc_block_3(cnn_enc_block_4, swintca_block_3)

        mcc_block = self.mcc_block(dn_smpl_4, patch_merging_3)

        up_smpl_4 = self.up_smpl_4(mcc_block)
        skip_4 = torch.cat([tcfc_block_3, up_smpl_4], dim=1)
        dec_block_4 = self.dec_block_4(skip_4)

        up_smpl_3 = self.up_smpl_3(dec_block_4)
        skip_3 = torch.cat([tcfc_block_2, up_smpl_3], dim=1)
        dec_block_3 = self.dec_block_3(skip_3)

        up_smpl_2 = self.up_smpl_2(dec_block_3)
        skip_2 = torch.cat([tcfc_block_1, up_smpl_2], dim=1)
        dec_block_2 = self.dec_block_2(skip_2)

        up_smpl_1 = self.up_smpl_1(dec_block_2)
        skip_1 = torch.cat([cnn_enc_block_1, up_smpl_1], dim=1)
        dec_block_1 = self.dec_block_1(skip_1)

        conv_out = self.conv_out(dec_block_1)
        output = self.softmax(conv_out)

        return output
