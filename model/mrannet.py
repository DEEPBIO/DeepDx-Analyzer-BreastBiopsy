import torch
import torch.nn as nn
from torch.nn import functional as F


class NonLocalBlock(nn.Module):
    def __init__(self, in_dim):
        super(NonLocalBlock, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim//4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.size()
        self_query = self.query_conv(x).view(B, -1, W*H).permute(0, 2, 1)
        self_key = self.key_conv(x).view(B, -1, W*H)
        bmm = torch.bmm(self_query, self_key)
        attention = self.softmax(bmm)
        self_value = self.value_conv(x).view(B, -1, W*H)

        output = torch.bmm(self_value, attention.permute(0, 2, 1))
        output = output.view(B, -1, W, H)

        output = self.gamma * output + x

        return output


class AdaptiveNorm2d(nn.Module):
    def __init__(self, in_channel, context_channel, Norm=nn.BatchNorm2d, track_running_stats=False):
        super(AdaptiveNorm2d, self).__init__()

        self.in_channel = in_channel
        self.fc = nn.Linear(context_channel, in_channel*2, bias=False)
        if Norm == nn.BatchNorm2d:
            self.norm = Norm(in_channel, track_running_stats=track_running_stats)
        else:
            self.norm = Norm(in_channel)

    def forward(self, x, context_info):
        B, C, H, W = x.size()
        #print(B, C, H, W)
        w = self.fc(context_info)
        x = self.norm(x)
        #print(x.size())
        #x = x.view(B*C, H*W).permute(1, 0)
        x = x.reshape(B*C, H*W).permute(1, 0)
        w_gamma = w[:, :self.in_channel].contiguous().view(-1)
        w_beta = w[:, self.in_channel:].contiguous().view(-1)
        x = x * w_gamma + w_beta
        x = x.permute(1, 0).contiguous().view(B, C, H, W)

        return x


class MRANNet(nn.Module):
    def __init__(self, context_ch=1280, low_level_conv_ch=32, high_level_conv_ch=128, n_classes=3, Norm=nn.BatchNorm2d, selective=False, track_running_stats=False):
        super(MRANNet, self).__init__()

        self.selective = selective

        self.non_local_block = NonLocalBlock(high_level_conv_ch)
        #self.conv_high = nn.Conv2d(high_level_conv_ch, high_level_conv_ch, 1, bias=False)
        #self.high_adap_norm = AdaptiveNorm2d(high_level_conv_ch, context_ch)

        self.low_conv = nn.Conv2d(low_level_conv_ch, 32, 1, bias=False)
        self.low_adap_norm = AdaptiveNorm2d(32, context_ch, Norm=Norm, track_running_stats=track_running_stats)
        self.low_relu = nn.ReLU()

        self.last_conv1 = nn.Conv2d(high_level_conv_ch+32, 128, 3, 1, 1, bias=False)
        self.last_adap_norm1 = AdaptiveNorm2d(128, context_ch, Norm=Norm, track_running_stats=track_running_stats)
        self.last_relu1 = nn.ReLU()
        self.last_conv2 = nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        self.last_adap_norm2 = AdaptiveNorm2d(64, context_ch, Norm=Norm, track_running_stats=track_running_stats)
        self.last_relu2 = nn.ReLU()
        self.last_conv3 = nn.Conv2d(64, n_classes, 1, 1)
        if self.selective:
            self.last_conv_selection = nn.Conv2d(64, 2, 1, 1)
            self.last_conv_aux = nn.Conv2d(64, n_classes, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, low_level_feature, high_level_feature, context_info):
        high_level_feature = self.non_local_block(high_level_feature)
        high_level_feature = F.upsample_bilinear(high_level_feature, size=low_level_feature.shape[-2:])
        #high_level_feature = self.conv_high(high_level_feature)
        #high_level_feature = self.high_adap_norm(high_level_feature, context_info)
        #high_level_feature = nn.ReLU(high_level_feature)

        low_level_feature = self.low_conv(low_level_feature)
        #print(low_level_feature.shape)
        low_level_feature = self.low_adap_norm(low_level_feature, context_info)
        low_level_feature = self.low_relu(low_level_feature)

        concat_feature = torch.cat([low_level_feature, high_level_feature], dim=1)
        output = self.last_conv1(concat_feature)
        output = self.last_adap_norm1(output, context_info)
        output = self.last_relu1(output)
        output = self.last_conv2(output)
        output = self.last_adap_norm2(output, context_info)
        output = self.last_relu2(output)
        #for training only last_conv3
        output = output.clone().detach()
        real_output = self.last_conv3(output)
        if self.selective:
            selection = self.last_conv_selection(output)
            aux = self.last_conv_aux(output)
            return real_output, selection, aux

        return real_output

class MRANNetLogit(nn.Module):
    def __init__(self, context_ch=1280, low_level_conv_ch=32, high_level_conv_ch=128, n_classes=3, Norm=nn.BatchNorm2d, selective=False, track_running_stats=False):
        super(MRANNetLogit, self).__init__()

        self.selective = selective

        self.non_local_block = NonLocalBlock(high_level_conv_ch)
        #self.conv_high = nn.Conv2d(high_level_conv_ch, high_level_conv_ch, 1, bias=False)
        #self.high_adap_norm = AdaptiveNorm2d(high_level_conv_ch, context_ch)

        self.low_conv = nn.Conv2d(low_level_conv_ch, 32, 1, bias=False)
        self.low_adap_norm = AdaptiveNorm2d(32, context_ch, Norm=Norm, track_running_stats=track_running_stats)
        self.low_relu = nn.ReLU()

        self.last_conv1 = nn.Conv2d(high_level_conv_ch+32, 128, 3, 1, 1, bias=False)
        self.last_adap_norm1 = AdaptiveNorm2d(128, context_ch, Norm=Norm, track_running_stats=track_running_stats)
        self.last_relu1 = nn.ReLU()
        self.last_conv2 = nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        self.last_adap_norm2 = AdaptiveNorm2d(64, context_ch, Norm=Norm, track_running_stats=track_running_stats)

        self.sigmoid = nn.Sigmoid()

    def forward(self, low_level_feature, high_level_feature, context_info):
        high_level_feature = self.non_local_block(high_level_feature)
        high_level_feature = F.upsample_bilinear(high_level_feature, size=low_level_feature.shape[-2:])
        #high_level_feature = self.conv_high(high_level_feature)
        #high_level_feature = self.high_adap_norm(high_level_feature, context_info)
        #high_level_feature = nn.ReLU(high_level_feature)

        low_level_feature = self.low_conv(low_level_feature)
        #print(low_level_feature.shape)
        low_level_feature = self.low_adap_norm(low_level_feature, context_info)
        low_level_feature = self.low_relu(low_level_feature)

        concat_feature = torch.cat([low_level_feature, high_level_feature], dim=1)
        output = self.last_conv1(concat_feature)
        output = self.last_adap_norm1(output, context_info)
        output = self.last_relu1(output)
        output = self.last_conv2(output)
        real_output = self.last_adap_norm2(output, context_info)

        return real_output