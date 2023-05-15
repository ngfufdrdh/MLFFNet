
# 这里写会用到的各种模块

import torch.nn as nn
import torch.nn.functional as F
import torch

class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same',
                 bias=False, bn=True, relu=False):
        super(conv, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, activate=False):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.activate = activate
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate == False:
            return x
        else:
            return self.relu(x)


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):     # [b, 32, 22, 22]

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)     # [b, 16, 121],       self.inter_channels = in_channels // 2
        g_x = g_x.permute(0, 2, 1)     # [b, 121, 16]       这里是Q

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)    # [b, 16, 22*22]
        theta_x = theta_x.permute(0, 2, 1)       # [b, 22*22, 16]
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)      # [b, 16, 11*11]
        f = torch.matmul(theta_x, phi_x)      # [b, 22*22, 11*11]
        f_div_C = F.softmax(f, dim=-1)        # [b, 22*22, 11*11]

        y = torch.matmul(f_div_C, g_x)        # [b, 22*22, 16]
        y = y.permute(0, 2, 1).contiguous()   # [b, 16, 22*22]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # [b, 16, 22, 22]
        W_y = self.W(y)
        z = W_y + x   # [b, 32, 22, 22]

        return z

class low_level_feature_process_3_8(nn.Module):

    def __init__(self, channel):
        super(low_level_feature_process_3_8, self).__init__()
        self.low_level_feature_mutli = Local_Feature_Enhance_3_8(in_channel=channel, out_channel=channel)
        self.channel_attention = ChannelAttention(in_planes=channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x_res = self.low_level_feature_mutli(x)
        x_res = x_res * self.channel_attention(x_res)
        x_res = x_res * self.spatial_attention(x_res)
        x = x + x_res
        return x


class Multi_Scale_kernel_3_8(nn.Module):
    def __init__(self, in_channel, out_channel, receptive_size=3):
        super(Multi_Scale_kernel_3_8, self).__init__()
        self.conv0 = conv(in_channel, out_channel, 1)
        self.conv1 = conv(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = conv(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = conv(out_channel, out_channel, 3, dilation=receptive_size)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Local_Feature_Enhance_3_8(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Local_Feature_Enhance_3_8, self).__init__()
        # self.relu = nn.ReLU(True)

        self.branch0 = conv(in_channel, out_channel, 1)
        self.branch1 = Multi_Scale_kernel_3_8(in_channel, out_channel, 3)
        self.branch2 = Multi_Scale_kernel_3_8(in_channel, out_channel, 5)
        self.branch3 = Multi_Scale_kernel_3_8(in_channel, out_channel, 7)

        self.conv_cat = conv(4 * out_channel, out_channel, 3, relu=True)
        # self.conv_res = conv(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        # x = self.relu(x_cat + x)
        x = x_cat + x

        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(AttentionBlock, self).__init__()
        # self.shape = shape
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

        # self.upsample = nn.Upsample(scale_factor=shape, mode='bilinear', align_corners=True)
        self.high_change_channel = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=self.inter_channels, kernel_size=1),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x, level_feature):     # [b, 32, 22, 22]

        # high_level_feature = self.upsample(high_level_feature)   # 上采样到当前对应的尺寸大小
        level_feature = self.high_change_channel(level_feature)        # 调整通道数

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)     # [b, 16, 121],       self.inter_channels = in_channels // 2
        g_x = g_x.permute(0, 2, 1)     # [b, 121, 16]       这里是Q


        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)    # [b, 16, 22*22],  这里将最深层的feature map与x相乘
        theta_x = theta_x.permute(0, 2, 1)       # [b, 22*22, 16]

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)      # [b, 16, 11*11]
        f = torch.matmul(theta_x, phi_x)      # [b, 22*22, 11*11]

        theta_y = theta_x
        level_feature = level_feature.view(batch_size, self.inter_channels, -1)
        g = torch.matmul(theta_y, level_feature)

        f_div_C = F.softmax(f + g, dim=-1)        # [b, 22*22, 11*11]

        y = torch.matmul(f_div_C, g_x)        # [b, 22*22, 16]
        y = y.permute(0, 2, 1).contiguous()   # [b, 16, 22*22]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # [b, 16, 22, 22]
        W_y = self.W(y)
        z = W_y + x   # [b, 32, 22, 22]

        return z



class Level_Cat_3_8(nn.Module):

    def __init__(self, channel):
        super(Level_Cat_3_8, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.x2_cat_attention = combine_attention(channel=channel)
        self.x3_cat_attention = combine_attention(channel=channel)
        self.x4_low_feature_progress = low_level_feature_process_3_8(channel=channel)


        self.conv_upsample_1 = BasicConv2d(in_planes=channel, out_planes=channel, kernel_size=3, padding=1, activate=True)
        self.conv_upsample_2 = BasicConv2d(in_planes=channel, out_planes=channel, kernel_size=3, padding=1, activate=True)
        self.conv_upsample_3 = BasicConv2d(in_planes=channel, out_planes=channel, kernel_size=3, padding=1, activate=True)

        self.conv_upsample_4 = BasicConv2d(in_planes=channel, out_planes=channel, kernel_size=3, padding=1, activate=True)
        self.conv_upsample_5 = BasicConv2d(in_planes=2*channel, out_planes=2*channel, kernel_size=3, padding=1, activate=True)
        self.conv_upsample_6 = BasicConv2d(in_planes=3*channel, out_planes=3*channel, kernel_size=3, padding=1, activate=True)

        self.upsample_attention_4 = AttentionBlock(in_channels=channel)
        self.upsample_attention_5 = AttentionBlock(in_channels=2*channel)
        self.upsample_attention_6 = AttentionBlock(in_channels=3*channel)

        self.conv4_all = BasicConv2d(4 * channel, channel, 3, padding=1)



    def forward(self, x1, x2, x3, x4):
        x1_1 = x1
        x2_cat = self.upsample(self.conv_upsample_1(x1)) * x2
        x2_cat = self.x2_cat_attention(x2_cat)

        x3_cat = self.upsample(self.upsample(self.conv_upsample_2(x1))) * self.upsample(self.conv_upsample_3(x2)) * x3
        x3_cat = self.x3_cat_attention(x3_cat)

        x4 = self.x4_low_feature_progress(x4)

        x2_2 = torch.cat((x2_cat, self.upsample_attention_4(self.upsample(x1_1), x2_cat)), dim=1)
        # x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_cat, self.upsample_attention_5(self.upsample(x2_2), x3_cat)), dim=1)
        # x3_2 = self.conv_concat3(x3_2)

        x4_2 = torch.cat((x4, self.upsample_attention_6(self.upsample(x3_2), x4)), dim=1)
        # x4_2 = self.conv_concat4(x4_2)

        out = self.conv4_all(x4_2)

        return x1_1, x3_2, out


class Level_Cat_pruduct_high_3_level_2_fourloss(nn.Module):

    def __init__(self, channel):
        super(Level_Cat_pruduct_high_3_level_2_fourloss, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv_upsample_1 = BasicConv2d(in_planes=channel, out_planes=channel, kernel_size=1)
        # self.conv_upsample_2 = BasicConv2d(in_planes=channel, out_planes=channel, kernel_size=1)
        # self.conv_upsample_3 = BasicConv2d(in_planes=channel, out_planes=channel, kernel_size=1)

        self.x2_cat_attention = combine_attention(channel=channel)
        self.x3_cat_attention = combine_attention(channel=channel)
        self.x4_cat_attention = combine_attention(channel=channel)

        self.conv_upsample_1 = BasicConv2d(in_planes=channel, out_planes=channel, kernel_size=3, padding=1, activate=True)
        self.conv_upsample_2 = BasicConv2d(in_planes=channel, out_planes=channel, kernel_size=3, padding=1, activate=True)
        self.conv_upsample_3 = BasicConv2d(in_planes=channel, out_planes=channel, kernel_size=3, padding=1, activate=True)

        self.conv_upsample_4 = BasicConv2d(in_planes=channel, out_planes=channel, kernel_size=3, padding=1, activate=True)
        self.conv_upsample_5 = BasicConv2d(in_planes=2*channel, out_planes=2*channel, kernel_size=3, padding=1, activate=True)
        self.conv_upsample_6 = BasicConv2d(in_planes=3*channel, out_planes=3*channel, kernel_size=3, padding=1, activate=True)


        self.conv_concat2 = BasicConv2d(in_planes=2 * channel, out_planes=2 * channel, kernel_size=3, padding=1)
        self.conv_concat3 = BasicConv2d(in_planes=3 * channel, out_planes=3 * channel, kernel_size=3, padding=1)
        self.conv_concat4 = BasicConv2d(in_planes=4 * channel, out_planes=4 * channel, kernel_size=3, padding=1)

        self.conv4_all = BasicConv2d(4 * channel, channel, 3, padding=1)



    def forward(self, x1, x2, x3, x4):
        x1_1 = x1
        x2_cat = self.upsample(self.conv_upsample_1(x1)) * x2
        x2_cat = self.x2_cat_attention(x2_cat)

        x3_cat = self.upsample(self.upsample(self.conv_upsample_2(x1))) * self.upsample(self.conv_upsample_3(x2)) * x3
        x3_cat = self.x3_cat_attention(x3_cat)

        x4 = self.x4_cat_attention(x4)

        x2_2 = torch.cat((x2_cat, self.conv_upsample_4(self.upsample(x1_1))), dim=1)
        # x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_cat, self.conv_upsample_5(self.upsample(x2_2))), dim=1)
        # x3_2 = self.conv_concat3(x3_2)

        x4_2 = torch.cat((x4, self.conv_upsample_6(self.upsample(x3_2))), dim=1)
        # x4_2 = self.conv_concat4(x4_2)

        out = self.conv4_all(x4_2)

        return x1_1, x2_2, x3_2, out


class combine_SE_and_Spatial_attention(nn.Module):
    def __init__(self, channel):
        super(combine_SE_and_Spatial_attention, self).__init__()
        self.SE = SE_Block(in_planes=channel)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x_c = x
        x_s = x
        x_c = x_c * self.SE(x_c)
        x_s = x_s * self.spatial(x_s)
        x = x_s + x_c
        return x

class combine_attention(nn.Module):
    def __init__(self, channel):
        super(combine_attention, self).__init__()
        self.channel_attention = ChannelAttention(in_planes=channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x_c = x
        x_s = x
        x_c = x_c * self.channel_attention(x_c)
        x_s = x_s * self.spatial_attention(x_s)
        x = x_s + x_c
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SE_Block(nn.Module):  # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out



