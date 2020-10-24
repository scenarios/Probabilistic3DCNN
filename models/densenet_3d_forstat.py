import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import types

import os

from collections import OrderedDict
from functools import partial

from args import parser
from NAS_utils.ops import Conv3d_with_CD, Linear_with_CD

args = parser.parse_args()

#Conv2d = Conv2d
Conv3d = nn.Conv3d#partial(Conv3d_with_CD, weight_reg=args.weight_reg, deterministic=True if args.selection_mode or args.finetune_mode else False, training_size=args.training_size, p_init=args.p_init)
Linear = nn.Linear #partial(Linear_with_CD, weight_reg=args.weight_reg, deterministic=True if args.selection_mode or args.finetune_mode else False, training_size=args.training_size, p_init=args.p_init)
nnConv2d = nn.Conv2d
BatchNorm3d = partial(nn.BatchNorm3d, track_running_stats=not args.freeze_bn)
_TEMPORAL_NASAS_ONLY = args.temporal_nasas_only

__all__ = ['densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
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


def load_pretrained(model, num_classes, settings):
    #assert num_classes == settings['num_classes'], \
    #    "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
    try:
        state_dict = torch.load('/log/checkpoint/Densenet121_2D_ImagenetPretrained/densenet121-a639ec97.pth')
    except:
        state_dict = model_zoo.load_url(settings['url'])
    state_dict = update_state_dict(state_dict)
    mk, uk = model.load_state_dict(state_dict, strict=False)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model


def update_state_dict(state_dict):
    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    # Inflate to 3d densenet
    pattern = re.compile(
        r'^(.*)((?:conv|norm)(?:[012]?)\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            v = state_dict[key]
            if 'conv' in key:
                v = torch.unsqueeze(v, dim=2)
                if 'conv0' in key:
                    v = v.repeat([1, 1, 5, 1, 1])
                    v /= 5.0
                    state_dict[key] = v
                elif 'conv1' in key:
                    new_key_btnk = res.group(1) + 'bottleneck.' + res.group(2)
                    state_dict[new_key_btnk] = v
                    if 'v1' in args.net_version:
                        new_key_tmpr = res.group(1) + 'temporal.' + res.group(2)
                        state_dict[new_key_tmpr] = v.repeat([1, 1, 3, 1, 1]) / 3.0
                    del state_dict[key]
                elif 'conv2' in key:
                    new_key_sptl = res.group(1) + 'spatial.' + res.group(2)
                    state_dict[new_key_sptl] = v
                    del state_dict[key]
                else:
                    if 'transition' in key:
                        new_key_btnk = res.group(1) + 'original.' + res.group(2)
                        state_dict[new_key_btnk] = v
                        if args.net_version in ['v1', 'v1d2', 'vt', 'v1d3']:
                            new_key_tmpr = res.group(1) + 'temporal.' + res.group(2)
                            state_dict[new_key_tmpr] = v.repeat([1, 1, 3, 1, 1]) / 3.0
                        del state_dict[key]
                    else:
                        state_dict[key] = v
            else:
                if 'norm1' in key:
                    new_key_btnk = res.group(1) + 'bottleneck.' + res.group(2)
                    state_dict[new_key_btnk] = v
                    if args.net_version in ['v1d2', 'v1nt', 'v1d3']:
                        new_key_tmpr = res.group(1) + 'temporal.' + res.group(2)
                        state_dict[new_key_tmpr] = v
                    del state_dict[key]
                elif 'norm2' in key:
                    new_key_sptl = res.group(1) + 'spatial.' + res.group(2)
                    state_dict[new_key_sptl] = v
                    del state_dict[key]
                else:
                    if 'transition' in key:
                        new_key_btnk = res.group(1) + 'original.' + res.group(2)
                        state_dict[new_key_btnk] = v
                        if args.net_version in ['v1d2', 'vt', 'v1d3']:
                            new_key_tmpr = res.group(1) + 'temporal.' + res.group(2)
                            state_dict[new_key_tmpr] = v
                        del state_dict[key]
                    else:
                        state_dict[key] = v

        if 'classifier' in key:
            del state_dict[key]
    return state_dict


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, split=1, split_pattern=None):
        super(_DenseLayer, self).__init__()
        self.bottleneck = nn.Sequential(OrderedDict([
            ('norm1', BatchNorm3d(num_input_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv1', Conv3d(num_input_features, bn_size *
                             growth_rate, kernel_size=1, stride=1, bias=False))
        ]))
        self.spatial = nn.Sequential(OrderedDict([
            ('norm2', BatchNorm3d(bn_size * growth_rate)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv2', Conv3d(bn_size * growth_rate, growth_rate,
                             kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False))
        ]))

        if 'v1' in args.net_version:
            self.temporal = nn.Sequential(OrderedDict([
                ('norm1', BatchNorm3d(num_input_features)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv1', Conv3d(num_input_features, bn_size *
                                 growth_rate, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False))
            ]))
        elif 'v2' in args.net_version:
            self.temporal = nn.Sequential(OrderedDict([
                ('norm1', BatchNorm3d(bn_size * growth_rate)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv1', Conv3d(bn_size * growth_rate, bn_size * growth_rate,
                                 kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False))
            ]))
        elif 'v3' in args.net_version:
            self.temporal = nn.Sequential(OrderedDict([
                ('norm1', BatchNorm3d(growth_rate)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv1', Conv3d(growth_rate, growth_rate,
                                 kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False))
            ]))
        else:
            pass
        self.drop_rate = drop_rate

    def forward(self, x):
        if 'v1' in args.net_version:
            new_features = self.temporal.forward(x) + self.bottleneck.forward(x)
            new_features = self.spatial.forward(new_features)
        elif 'v2' in args.net_version:
            new_features = self.bottleneck.forward(x)
            new_features = self.temporal.forward(new_features) + new_features
            new_features = self.spatial.forward(new_features)
        elif 'v3' in args.net_version:
            new_features = self.bottleneck.forward(x)
            new_features = self.spatial.forward(new_features)
            new_features = self.temporal.forward(new_features) + new_features
        else:
            new_features = self.bottleneck.forward(x)
            new_features = self.spatial.forward(new_features)
        #if self.drop_rate > 0:
        #    new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        self._split_pattern = [num_input_features]
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, split=1, split_pattern=None)#split=i+1, split_pattern=self._split_pattern)
            self.add_module('denselayer%d' % (i + 1), layer)
            # DO NOT use += in-place operator here!
            self._split_pattern = self._split_pattern + [growth_rate]


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, split=1, split_pattern=None, temporal_pool_size=1):
        super(_Transition, self).__init__()
        '''
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False, split=split, split_pattern=split_pattern))
        self.add_module('pool', nn.AvgPool3d(kernel_size=(temporal_pool_size, 2, 2), stride=(temporal_pool_size, 2, 2)))
        '''
        self.original = nn.Sequential(OrderedDict([
            ('norm', BatchNorm3d(num_input_features)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        ]))
        self.transition_pool = nn.Sequential(OrderedDict([
            ('pool', nn.AvgPool3d(kernel_size=(temporal_pool_size, 2, 2), stride=(temporal_pool_size, 2, 2)))
        ]))

        if args.net_version in ['v1', 'v1d2', 'vt', 'v1d3']:
            self.temporal = nn.Sequential(OrderedDict([
                ('norm', BatchNorm3d(num_input_features)),
                ('relu', nn.ReLU(inplace=True)),
                ('conv', Conv3d(num_input_features, num_output_features,
                                kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False))
            ]))
        elif args.net_version in ['v2', 'v3', 'v4']:
            self.temporal = nn.Sequential(OrderedDict([
                ('norm', BatchNorm3d(num_output_features)),
                ('relu', nn.ReLU(inplace=True)),
                ('conv', Conv3d(num_output_features, num_output_features,
                                kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False))
            ]))
        else:
            pass

    def forward(self, input):
        if args.net_version in ['v1', 'v1d2', 'vt', 'v1d3']:
            new_features = self.original(input) + self.temporal(input)
        elif args.net_version in ['v2', 'v3', 'v4']:
            new_features = self.original(input)
            new_features = self.temporal(new_features) + new_features
        else:
            new_features = self.original(input)
        new_features = self.transition_pool(new_features)
        return new_features


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()
        self.drop_rate = drop_rate

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(3, num_init_features, kernel_size=(5, 7, 7), stride=(1, 2, 2) if args.net_version=='v1d3' else 2, padding=(2, 3, 3), bias=False)),
            ('norm0', BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2) if args.net_version=='v1d3' or args.random_dense_sample_stride else 2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        downsample_pos = [-1] if args.net_version=='v1d3' else [0]
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=self.drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                    split=1,#split=num_layers+1,
                                    split_pattern=None, #split_pattern=[num_features - num_layers * growth_rate]+[growth_rate]*num_layers,
                                    temporal_pool_size=2 if i in downsample_pos else 1)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', BatchNorm3d(num_features))

        # Linear layer
        self.classifier = Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(features.size(0), -1)
        out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.classifier(out)
        return out


def _load_state_dict(model, model_url):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    state_dict = model_zoo.load_url(model_url)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def modify_densenets(model):
    # Modify attributs
    model.last_linear = model.classifier
    del model.classifier

    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def _densenet121(num_classes, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=num_classes,
                     **kwargs)
    return model

def _densenet169(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls['densenet169'])
    return model

def densenet121(num_classes=1000, pretrained='imagenet', drop_rate=0.0):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = _densenet121(num_classes=num_classes, drop_rate=drop_rate)
    if pretrained is not None:
        settings = pretrained_settings['densenet121'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    return model


def densenet169(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = _densenet169(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet169'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    #model = modify_densenets(model)
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls['densenet201'])
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls['densenet161'])
    return model