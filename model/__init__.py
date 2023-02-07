from .mnasnet import *
from .mnasnet import *
from .mrannet import *
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import argparse

"""Provide a unified API to Ellie.
"""

def load_model(model_name, class_num):
    torch.backends.cudnn.benchmark = True

    args = argparse.Namespace()
    args.model = 'mran_selective'
    args.backbone = 'mnasnet'
    args.version = '1.0'
    args.batch_size = 160
    # args.csv = None
    args.low = False
    args.high = False
    args.selective = False
    args.norm='bn'
    args.track_stat = True
    args.pretrained = True
    if model_name == "mran_selective":
        args.selective = True
        model = MRANNetForward(args, class_num, False, version=args.version)
        args.selective = False
    else:
        raise ("Model Name Error")

    return model

class Group(nn.Module):

    def __init__(self, out_channels):
        super(Group, self).__init__()
        self.groupnorm = nn.GroupNorm(4, out_channels)

    def forward(self, x):
        return self.groupnorm(x)


class MRANNetForward(nn.Module):
    def __init__(self, args, n_classes=3, pretrained=False, version='1.3'):
        super(MRANNetForward, self).__init__()
        self.args = args

        if args.norm == 'gn':
            self.norm = Group
        elif args.norm == 'bn':
            self.norm = nn.BatchNorm2d
        elif args.norm == 'syncbn':
            self.norm = SynchronizedBatchNorm2d

        if version == '1.0':
            self.low_res_mnasnet = mnasnet1_0(pretrained=pretrained, Norm=self.norm,
                                              track_running_stats=args.track_stat)
            if self.args.backbone == 'mnasnet':
                self.high_res_mnasnet = mnasnet1_0(pretrained=pretrained, Norm=self.norm,
                                                   track_running_stats=args.track_stat)
                low_level_conv_ch = 24
                high_level_conv_ch = 96
            self.mrain_net = MRANNet(low_level_conv_ch=low_level_conv_ch, high_level_conv_ch=high_level_conv_ch,
                                     n_classes=n_classes, selective=self.args.selective,
                                     track_running_stats=args.track_stat)
        elif version == '1.3':
            self.low_res_mnasnet = mnasnet1_3(pretrained=pretrained, Norm=self.norm,
                                              track_running_stats=args.track_stat)
            self.high_res_mnasnet = mnasnet1_3(pretrained=pretrained, Norm=self.norm,
                                               track_running_stats=args.track_stat)
            self.mrain_net = MRANNet(low_level_conv_ch=32, high_level_conv_ch=128, n_classes=n_classes, Norm=self.norm,
                                     track_running_stats=args.track_stat)
        # self.low_res_mnasnet = torchvision.models.mnasnet1_3(pretrained=pretrained, progress=True)
        self.low_res_mnasnet.classifier = nn.Sequential(nn.Linear(1280, n_classes), nn.Sigmoid())
        # self.high_res_mnasnet = torchvision.models.mnasnet1_3(pretrained=pretrained, progress=True)
        self.high_res_mnasnet.classifier = nn.Sequential(nn.Linear(1280, n_classes), nn.Sigmoid())

    def forward(self, low_res_input, high_res_input):
        _, context_info, _, _ = self.low_res_mnasnet(low_res_input)
        if self.args.backbone == 'mnasnet':
            _, _, high_level_feature, low_level_feature = self.high_res_mnasnet(high_res_input)
        if self.args.selective:
            final_output, selection, aux = self.mrain_net(low_level_feature, high_level_feature, context_info)
            return final_output, selection, aux
        else:
            final_output = self.mrain_net(low_level_feature, high_level_feature, context_info)
            return final_output