import math
import warnings

import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['MNASNet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3']

_MODEL_URLS = {
    "mnasnet0_5":
        "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
    "mnasnet0_75": None,
    "mnasnet1_0":
        "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
    "mnasnet1_3": None
}

# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997


class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor,
                 Norm=nn.BatchNorm2d, bn_momentum=0.1, track_running_stats=False):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)
        if Norm == nn.BatchNorm2d:
            self.layers = nn.Sequential(
                # Pointwise
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                Norm(mid_ch, momentum=bn_momentum, track_running_stats=track_running_stats),
                nn.ReLU(inplace=True),
                # Depthwise
                nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2,
                          stride=stride, groups=mid_ch, bias=False),
                Norm(mid_ch, momentum=bn_momentum, track_running_stats=track_running_stats),
                nn.ReLU(inplace=True),
                # Linear pointwise. Note that there's no activation.
                nn.Conv2d(mid_ch, out_ch, 1, bias=False),
                Norm(out_ch, momentum=bn_momentum, track_running_stats=track_running_stats))
        else:
            self.layers = nn.Sequential(
                # Pointwise
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                Norm(mid_ch),
                nn.ReLU(inplace=True),
                # Depthwise
                nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2,
                          stride=stride, groups=mid_ch, bias=False),
                Norm(mid_ch),
                nn.ReLU(inplace=True),
                # Linear pointwise. Note that there's no activation.
                nn.Conv2d(mid_ch, out_ch, 1, bias=False),
                Norm(out_ch))

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats,
           Norm, bn_momentum, track_running_stats=False):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor,
                              Norm=Norm, bn_momentum=bn_momentum, track_running_stats=track_running_stats)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            _InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor,
                              Norm=Norm, bn_momentum=bn_momentum, track_running_stats=track_running_stats))
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MNASNet(torch.nn.Module):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1000, 1.0)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    1
    >>> y.nelement()
    1000
    """
    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(self, alpha, num_classes=1000, dropout=0.2, Norm=nn.BatchNorm2d, track_running_stats=False):
        super(MNASNet, self).__init__()
        assert alpha > 0.0
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths(alpha)
        if Norm == nn.BatchNorm2d:
            layers = [
                # First layer: regular conv.
                nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False),
                Norm(depths[0], momentum=_BN_MOMENTUM, track_running_stats=track_running_stats),
                nn.ReLU(inplace=True),
                # Depthwise separable, no skip.
                nn.Conv2d(depths[0], depths[0], 3, padding=1, stride=1,
                          groups=depths[0], bias=False),
                Norm(depths[0], momentum=_BN_MOMENTUM, track_running_stats=track_running_stats),
                nn.ReLU(inplace=True),
                nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False),
                Norm(depths[1], momentum=_BN_MOMENTUM, track_running_stats=track_running_stats),
                # MNASNet blocks: stacks of inverted residuals.
                _stack(depths[1], depths[2], 3, 2, 3, 3, Norm, _BN_MOMENTUM, track_running_stats),
                _stack(depths[2], depths[3], 5, 2, 3, 3, Norm, _BN_MOMENTUM, track_running_stats),
                _stack(depths[3], depths[4], 5, 2, 6, 3, Norm, _BN_MOMENTUM, track_running_stats),
                _stack(depths[4], depths[5], 3, 1, 6, 2, Norm, _BN_MOMENTUM, track_running_stats),
                _stack(depths[5], depths[6], 5, 2, 6, 4, Norm, _BN_MOMENTUM, track_running_stats),
                _stack(depths[6], depths[7], 3, 1, 6, 1, Norm, _BN_MOMENTUM, track_running_stats),
                # Final mapping to classifier input.
                nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False),
                Norm(1280, momentum=_BN_MOMENTUM, track_running_stats=track_running_stats),
                nn.ReLU(inplace=True),
            ]
        else:
            layers = [
                # First layer: regular conv.
                nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False),
                Norm(depths[0]),
                nn.ReLU(inplace=True),
                # Depthwise separable, no skip.
                nn.Conv2d(depths[0], depths[0], 3, padding=1, stride=1,
                          groups=depths[0], bias=False),
                Norm(depths[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False),
                Norm(depths[1]),
                # MNASNet blocks: stacks of inverted residuals.
                _stack(depths[1], depths[2], 3, 2, 3, 3, Norm, _BN_MOMENTUM),
                _stack(depths[2], depths[3], 5, 2, 3, 3, Norm, _BN_MOMENTUM),
                _stack(depths[3], depths[4], 5, 2, 6, 3, Norm, _BN_MOMENTUM),
                _stack(depths[4], depths[5], 3, 1, 6, 2, Norm, _BN_MOMENTUM),
                _stack(depths[5], depths[6], 5, 2, 6, 4, Norm, _BN_MOMENTUM),
                _stack(depths[6], depths[7], 3, 1, 6, 1, Norm, _BN_MOMENTUM),
                # Final mapping to classifier input.
                nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False),
                Norm(1280),
                nn.ReLU(inplace=True),
            ]
        self.layers = nn.Sequential(*layers)
        self.adap_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                        nn.Linear(1280, num_classes))

    def forward(self, x):
        module_dict = dict(self.layers.named_children())
        for key, module in module_dict.items():
            x = module(x)
            if isinstance(module, nn.Sequential) and key == '8':
                one_fourth = x
            elif isinstance(module, nn.Sequential) and key == '11':
                one_sixteenth = x
            else:
                pass
        # Equivalent to global avgpool and removing H and W dimensions.
        avg = self.adap_avg_pool(x)
        avg = avg.view(avg.size(0), -1)

        return self.classifier(avg), avg, one_sixteenth, one_fourth


"""
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get("version", None)
        assert version in [1, 2]

        if version == 1 and not self.alpha == 1.0:
            # In the initial version of the model (v1), stem was fixed-size.
            # All other layer configurations were the same. This will patch
            # the model so that it's identical to v1. Model with alpha 1.0 is
            # unaffected.
            depths = _get_depths(self.alpha)
            v1_stem = [
                nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32,
                          bias=False),
                nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=False),
                nn.BatchNorm2d(16, momentum=_BN_MOMENTUM),
                _stack(16, depths[2], 3, 2, 3, 3, _BN_MOMENTUM),
            ]
            for idx, layer in enumerate(v1_stem):
                self.layers[idx] = layer

            # The model is now identical to v1, and must be saved as such.
            self._version = 1
            warnings.warn(
                "A new version of MNASNet model has been implemented. "
                "Your checkpoint was saved using the previous version. "
                "This checkpoint will load and work as before, but "
                "you may want to upgrade by training a newer model or "
                "transfer learning from an updated ImageNet checkpoint.",
                UserWarning)

        super(MNASNet, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)
"""


def _load_pretrained(model_name, model, progress):
    if model_name not in _MODEL_URLS or _MODEL_URLS[model_name] is None:
        raise ValueError(
            "No checkpoint is available for model type {}".format(model_name))
    checkpoint_url = _MODEL_URLS[model_name]
    checkpoint = load_state_dict_from_url(checkpoint_url, progress=progress)
    model_state_dict = model.state_dict()
    filtered_dict = {}
    for k, v in model_state_dict.items():
        if v.shape == checkpoint[k].shape:
            filtered_dict[k] = checkpoint[k]
    model_state_dict.update(filtered_dict)
    model.load_state_dict(model_state_dict)
    # model.load_state_dict(
    #    load_state_dict_from_url(checkpoint_url, progress=progress))


def mnasnet0_5(pretrained=False, progress=True, **kwargs):
    """MNASNet with depth multiplier of 0.5 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.5, **kwargs)
    if pretrained:
        _load_pretrained("mnasnet0_5", model, progress)
    return model


def mnasnet0_75(pretrained=False, progress=True, **kwargs):
    """MNASNet with depth multiplier of 0.75 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.75, **kwargs)
    if pretrained:
        _load_pretrained("mnasnet0_75", model, progress)
    return model


def mnasnet1_0(pretrained=False, progress=True, **kwargs):
    """MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.0, **kwargs)
    if pretrained:
        _load_pretrained("mnasnet1_0", model, progress)
    return model


def mnasnet1_3(pretrained=False, progress=True, **kwargs):
    """MNASNet with depth multiplier of 1.3 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.3, **kwargs)
    if pretrained:
        _load_pretrained("mnasnet1_3", model, progress)
    return model
