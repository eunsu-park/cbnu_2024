# 29_conv_5.py
# Convolution Layer - stride

import torch
import torch.nn as nn

def get_num_params(model):
    """
    모델(레이어)의 파라미터 수를 계산하는 함수
    
    Args:
        model : torch.nn.Module
    Returns:
        num_params : int
    """
    return sum([p.numel() for p in model.parameters()])

inp = torch.randn(128, 1, 16, 16)
conv1 = nn.Conv2d(1, 3, kernel_size=3)
print(conv1)
print(f"Number of parameters in conv1: {get_num_params(conv1)}")
out = conv1(inp)
print(out.size())

inp = torch.randn(128, 1, 16, 16)
conv2 = nn.Conv2d(1, 3, kernel_size=3, stride=2)
print(conv2)
print(f"Number of parameters in conv2: {get_num_params(conv2)}")
out = conv2(inp)
print(out.size())

inp = torch.randn(128, 1, 16, 16)
conv3 = nn.Conv2d(1, 3, kernel_size=3, stride=(2, 1))
print(conv3)
print(f"Number of parameters in conv3: {get_num_params(conv3)}")
out = conv3(inp)
print(out.size())
