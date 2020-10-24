import re

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from collections import OrderedDict
#from functools import partial

from args import parser
#from NAS_utils.ops import Conv3d_with_CD, Linear_with_CD

args = parser.parse_args()

#Conv2d = Conv2d
Conv3d = nn.Conv3d #partial(Conv3d_with_CD, weight_reg=args.weight_reg, deterministic=True if args.selection_mode else False, training_size=args.training_size)
Linear = nn.Linear #partial(Linear_with_CD, weight_reg=args.weight_reg, deterministic=True if args.selection_mode else False, training_size=args.training_size)
nnConv2d = nn.Conv2d

_TEMPORAL_NASAS_ONLY = args.temporal_nasas_only


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

input_sizes = {}
means = {}
stds = {}

for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]

pretrained_settings = {}

for model_name in __all__:
    pretrained_settings[model_name] = {
        'imagenet': {
            'url': model_urls[model_name],
            'input_space': 'RGB',
            'input_size': input_sizes[model_name],
            'crop_size': input_sizes[model_name][-1] * 256 // 224,
            'input_range': [0, 1],
            'mean': means[model_name],
            'std': stds[model_name]
            #'num_classes': 174
        }
    }


def update_state_dict(state_dict):
    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    """
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    """
    # Inflate to 3d densenet
    pattern = re.compile(
        r'^(.*)((?:conv|bn)(?:[0123]?)\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            v = state_dict[key]
            if 'conv' in key:
                v = torch.unsqueeze(v, dim=2)
                if 'layer' not in key:
                    v = v.repeat([1, 1, 5, 1, 1])
                    v /= 5.0
                    state_dict[key] = v
                elif 'conv1' in key:
                    new_key_btnk = res.group(1) + 'bottleneck.' + res.group(2)
                    state_dict[new_key_btnk] = v
                    new_key_tmpr = res.group(1) + 'temporal.' + res.group(2)
                    state_dict[new_key_tmpr] = v.repeat([1, 1, 3, 1, 1]) / 3.0
                    del state_dict[key]
                elif 'conv2' in key:
                    state_dict[key] = v
                elif 'conv3' in key:
                    state_dict[key] = v
            else:
                if 'bn1' in key:
                    pass
                elif 'bn2' in key:
                    pass
                else:
                    pass
        if 'downsample' in key:
            v = state_dict[key]
            if 'downsample.0' in key:
                v = torch.unsqueeze(v, dim=2)
                state_dict[key] = v
        if 'fc' in key:
            del state_dict[key]
    return state_dict


def load_pretrained(model, num_classes, settings):
    #assert num_classes == settings['num_classes'], \
    #    "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
    state_dict = model_zoo.load_url(settings['url'])
    state_dict = update_state_dict(state_dict)
    mk, uk = model.load_state_dict(state_dict, strict=False)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, temporal_stride=1, enable_fuse=False, modality='temporal'):
        super(Bottleneck, self).__init__()
        self.enable_fuse=enable_fuse
        self.modality = modality

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if args.net_version == 'pure_fused':
            self.bottleneck = nn.Sequential(OrderedDict([
                ('conv1', Conv3d(inplanes, width, kernel_size=1, stride=1, bias=False))
            ]))
            self.temporal = nn.Sequential(OrderedDict([
                ('conv1', Conv3d(inplanes, width, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False))
            ]))
        elif args.net_version == 'pure_spatial':
            self.bottleneck = nn.Sequential(OrderedDict([
                ('conv1', Conv3d(inplanes, width, kernel_size=1, stride=1, bias=False))
            ]))
        elif args.net_version == 'pure_temporal':
            self.temporal = nn.Sequential(OrderedDict([
                ('conv1', Conv3d(inplanes, width, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False))
            ]))
        else:
            assert args.net_version == 'pure_adaptive', 'Unknown network version: {}'.format(args.net_version)
            if self.enable_fuse:
                self.bottleneck = nn.Sequential(OrderedDict([
                    ('conv1', Conv3d(inplanes, width, kernel_size=1, stride=1, bias=False))
                ]))
                self.temporal = nn.Sequential(OrderedDict([
                    ('conv1', Conv3d(inplanes, width, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False))
                ]))
            else:
                if self.modality == 'temporal':
                    self.temporal = nn.Sequential(OrderedDict([
                        ('conv1', Conv3d(inplanes, width, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False))
                    ]))
                else:
                    self.bottleneck = nn.Sequential(OrderedDict([
                        ('conv1', Conv3d(inplanes, width, kernel_size=1, stride=1, bias=False))
                    ]))


        self.bn1 = norm_layer(width)

        if temporal_stride != 1:
            self.temporal_pool = nn.AvgPool3d(kernel_size=(temporal_stride, 1, 1), stride=(temporal_stride, 1, 1))

        self.conv2 = Conv3d(width, width, groups=groups, dilation=dilation, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = Conv3d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.temporal_stride = temporal_stride

    def forward(self, x):
        identity = x

        if self.temporal_stride != 1:
            out = self.temporal_pool(x)
        else:
            out = x
        if args.net_version == 'pure_fused':
            out = self.temporal(out) + self.bottleneck(out)
        elif args.net_version == 'pure_spatial':
            out = self.bottleneck(out)
        elif args.net_version == 'pure_temporal':
            out = self.temporal(out)
        else:
            assert args.net_version == 'pure_adaptive', 'Unknown network version: {}'.format(args.net_version)
            if self.enable_fuse:
                out = self.temporal(out) + self.bottleneck(out)
            else:
                if self.modality == 'temporal':
                    out = self.temporal(out)
                else:
                    out = self.bottleneck(out)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, drop_rate=0.0):
        super(ResNet, self).__init__()
        self.drop_rate = drop_rate

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=(5, 7, 7), stride=(2, 2, 2) if 'kinetics' in args.dataset or 'ucf' in args.dataset else (1, 2, 2), padding=(2, 3, 3),
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0], temporal_stride=1, enable_fuse=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], temporal_stride=1, enable_fuse=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], temporal_stride=1, enable_fuse=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], temporal_stride=1, enable_fuse=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.new_fc = Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, temporal_stride=1, enable_fuse=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1:
            downsample = nn.Sequential(OrderedDict([
                ('avepool', nn.AvgPool3d(kernel_size=(temporal_stride, 3, 3), stride=(temporal_stride, stride, stride), padding=(0, 1, 1))),
                ('0', Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=(1, 1, 1), bias=False)),
                ('1', norm_layer(planes * block.expansion))
            ]))
        elif self.inplanes != planes * block.expansion:
            assert temporal_stride==1, 'temporal stride != 1'
            downsample = nn.Sequential(OrderedDict([
                ('0', Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=(1, 1, 1), bias=False)),
                ('1', norm_layer(planes * block.expansion))
            ]))
        layers = []
        if blocks == 23:
            if 'kinetics' in args.dataset:
                layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                    self.base_width, previous_dilation, norm_layer, temporal_stride=temporal_stride,
                                    enable_fuse=False, modality='spatial'))
                self.inplanes = planes * block.expansion
                for _ in range(1, 5):
                    layers.append(block(self.inplanes, planes, groups=self.groups,
                                        base_width=self.base_width, dilation=self.dilation,
                                        norm_layer=norm_layer, enable_fuse=False, modality='spatial'))
                for _ in range(5, 15):
                    layers.append(block(self.inplanes, planes, groups=self.groups,
                                        base_width=self.base_width, dilation=self.dilation,
                                        norm_layer=norm_layer, enable_fuse=True))
                for _ in range(15, 19):
                    layers.append(block(self.inplanes, planes, groups=self.groups,
                                        base_width=self.base_width, dilation=self.dilation,
                                        norm_layer=norm_layer, enable_fuse=False, modality='spatial'))
                for _ in range(19, blocks):
                    layers.append(block(self.inplanes, planes, groups=self.groups,
                                        base_width=self.base_width, dilation=self.dilation,
                                        norm_layer=norm_layer, enable_fuse=False, modality='temporal'))
            else:
                layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                    self.base_width, previous_dilation, norm_layer, temporal_stride=temporal_stride,
                                    enable_fuse=False))
                self.inplanes = planes * block.expansion
                for _ in range(1, 10):
                    layers.append(block(self.inplanes, planes, groups=self.groups,
                                        base_width=self.base_width, dilation=self.dilation,
                                        norm_layer=norm_layer, enable_fuse=True))
                for _ in range(10, blocks):
                    layers.append(block(self.inplanes, planes, groups=self.groups,
                                        base_width=self.base_width, dilation=self.dilation,
                                        norm_layer=norm_layer, enable_fuse=False))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer, temporal_stride=temporal_stride,
                                enable_fuse=enable_fuse))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, enable_fuse=enable_fuse))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), -1)
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.new_fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings[arch][pretrained]
        model = load_pretrained(model, kwargs['num_classes'], settings)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)



def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet50(pretrained='imagenet', progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (str): pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)



def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)



def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)



def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)



def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)



def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


if __name__ == '__main__':
    resnet50(pretrained='imagenet', num_classes=1000)