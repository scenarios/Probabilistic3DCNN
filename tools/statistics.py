import sys
sys.path.append('..')

from thop import profile
from models.densenet_3d_forstat import densenet121
from args import parser

import torch

args = parser.parse_args()
model = densenet121(num_classes=174, pretrained=None, drop_rate=0.5)
input = torch.randn(1, 3, 128, 256, 256)
flops, params = profile(model, inputs=(input, ))

print(flops)
print(params)