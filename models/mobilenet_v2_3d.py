from torch import nn
from torch.functional import F
import torch.utils.model_zoo as model_zoo

from args import parser
args = parser.parse_args()

import torch
import re


__all__ = ['mobilenet_v2']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
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
        if True:
            v = state_dict[key]
            if 'features.0.' in key:
                if 'features.0.0.weight' in key:
                    v = torch.unsqueeze(v, dim=2)
                    v = v.repeat([1, 1, 5, 1, 1])
                    v /= 5.0
                    state_dict[key] = v
                else:
                    pass
            elif 'features.1.' in key:
                if 'conv.0' in key:
                    if 'conv.0.0' in key:
                        v = torch.unsqueeze(v, dim=2)
                    new_key_btnk = key.replace('conv.0', 'depth_wise')
                    state_dict[new_key_btnk] = v
                    del state_dict[key]
                elif 'conv.1' in key:
                    v = torch.unsqueeze(v, dim=2)
                    new_key_btnk = key.replace('conv.1', 'point_wise')
                    state_dict[new_key_btnk] = v
                    del state_dict[key]
                else:
                    assert 'conv.2' in key
                    new_key_btnk = key.replace('conv.2', 'bn')
                    state_dict[new_key_btnk] = v
                    del state_dict[key]
            elif 'features.18.' in key:
                if 'features.18.0.weight' in key:
                    v = torch.unsqueeze(v, dim=2)
                    state_dict[key] = v
                else:
                    pass
            elif 'classifier' in key:
                pass
            else:
                if 'conv.0.' in key:
                    if 'conv.0.0.' in key:
                        v = torch.unsqueeze(v, dim=2)
                        new_key_btnk = key.replace('conv.0', 'bottleneck')
                        state_dict[new_key_btnk] = v
                        new_key_btnk = key.replace('conv.0', 'temporal')
                        state_dict[new_key_btnk] = v.repeat([1, 1, 3, 1, 1]) / 3.0
                    else:
                        new_key_btnk = key.replace('conv.0', 'bottleneck')
                        state_dict[new_key_btnk] = v
                        new_key_btnk = key.replace('conv.0', 'temporal')
                        state_dict[new_key_btnk] = v
                    del state_dict[key]
                elif 'conv.1.' in key:
                    if 'conv.1.0.' in key:
                        v = torch.unsqueeze(v, dim=2)
                    new_key_btnk = key.replace('conv.1', 'depth_wise')
                    state_dict[new_key_btnk] = v
                    del state_dict[key]
                elif 'conv.2.' in key:
                    v = torch.unsqueeze(v, dim=2)
                    new_key_btnk = key.replace('conv.2', 'point_wise')
                    state_dict[new_key_btnk] = v
                    del state_dict[key]
                else:
                    assert 'conv.3.' in key
                    new_key_btnk = key.replace('conv.3', 'bn')
                    state_dict[new_key_btnk] = v
                    del state_dict[key]

    return state_dict


def load_pretrained(model, num_classes, settings):
    #assert num_classes == settings['num_classes'], \
    #    "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
    state_dict = model_zoo.load_url(settings['url'])
    state_dict = update_state_dict(state_dict)
    mk, uk = model.load_state_dict(state_dict, strict=False)
    print('mk: {}'.format(mk))
    print('uk: {}'.format(uk))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=(1, 3, 3), stride=(1, 1, 1), groups=1):

        padding = tuple([(k - 1) // 2 for k in kernel_size])
        super(ConvBNReLU, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, modality=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.modality = modality
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio != 1:
            if args.net_version == 'pure_spatial':
                self.bottleneck = ConvBNReLU(inp, hidden_dim, kernel_size=(1, 1, 1))
            elif args.net_version == 'pure_temporal':
                self.temporal = ConvBNReLU(inp, hidden_dim, kernel_size=(3, 1, 1))
            elif args.net_version == 'pure_fused':
                self.bottleneck = ConvBNReLU(inp, hidden_dim, kernel_size=(1, 1, 1))
                self.temporal = ConvBNReLU(inp, hidden_dim, kernel_size=(3, 1, 1))
            elif args.net_version == 'pure_adaptive':
                assert self.modality is not None
                if self.modality == 'fused':
                    self.bottleneck = ConvBNReLU(inp, hidden_dim, kernel_size=(1, 1, 1))
                    self.temporal = ConvBNReLU(inp, hidden_dim, kernel_size=(3, 1, 1))
                elif self.modality == 'spatial':
                    self.bottleneck = ConvBNReLU(inp, hidden_dim, kernel_size=(1, 1, 1))
                else:
                    assert self.modality == 'temporal'
                    self.temporal = ConvBNReLU(inp, hidden_dim, kernel_size=(3, 1, 1))
            else:
                self.bottleneck = ConvBNReLU(inp, hidden_dim, kernel_size=(1, 1, 1))
            # pw
        self.depth_wise = ConvBNReLU(hidden_dim, hidden_dim, stride=(1, stride, stride), groups=hidden_dim)
        self.point_wise =  nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm3d(oup)

    def forward(self, x):
        if self.expand_ratio != 1:
            if args.net_version == 'pure_spatial':
                new_features = self.bottleneck(x)
            elif args.net_version == 'pure_temporal':
                new_features = self.temporal(x)
            elif args.net_version == 'pure_fused':
                new_features = self.bottleneck(x) + self.temporal(x)
            elif args.net_version == 'pure_adaptive':
                assert self.modality is not None
                if self.modality == 'fused':
                    new_features = self.bottleneck(x) + self.temporal(x)
                elif self.modality == 'spatial':
                    new_features = self.bottleneck(x)
                else:
                    assert self.modality == 'temporal'
                    new_features = self.temporal(x)
            else:
                new_features = self.bottleneck(x)
        else:
            new_features = x
        new_features = self.bn(self.point_wise(self.depth_wise(new_features)))
        if self.use_res_connect:
            return x + new_features
        else:
            return new_features


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, drop_rate=0.0):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            if 'something' in args.dataset:
                inverted_residual_setting = [
                    # t, c, n, s, m
                    [1, 16, 1, 1, 'fused'],
                    [6, 24, 2, 2, 'fused'],
                    [6, 32, 3, 2, 'fused'],
                    [6, 64, 4, 2, 'temporal'],
                    [6, 96, 3, 1, 'temporal'],
                    [6, 160, 3, 2, 'fused'],
                    [6, 320, 1, 1, 'fused'],
                ]
            else:
                inverted_residual_setting = [
                    # t, c, n, s, m
                    [1, 16, 1, 1, 'fused'],
                    [6, 24, 2, 2, 'fused'],
                    [6, 32, 3, 2, 'spatial'],
                    [6, 64, 4, 2, 'fused'],
                    [6, 96, 3, 1, 'spatial'],
                    [6, 160, 3, 2, 'temporal'],
                    [6, 320, 1, 1, 'fused'],
                ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 5:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 5-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, kernel_size=(5, 3, 3))]
        # building inverted residual blocks
        for t, c, n, s, m in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, modality=m))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=(1, 1, 1)))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.new_classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), -1)
        x = self.new_classifier(x)
        return x


def mobilenet_v2(pretrained='imagenet', progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        settings = pretrained_settings['mobilenet_v2'][pretrained]
        model = load_pretrained(model, kwargs['num_classes'], settings)
    return model


if __name__ == '__main__':
    mobilenet_v2(pretrained='imagenet', num_classes=1000)