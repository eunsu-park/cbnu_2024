# 26_conv_2.py
# Convolution Layer - input image size
# 입력 이미지의 크기가 달라져도 파라미터 수는 변하지 않음, 다만 출력 크기는 달라짐

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

inp = torch.randn(128, 1, 32, 32)

conv1 = nn.Conv2d(1, 1, kernel_size=3)
print(conv1)
print(f"Number of parameters in conv1: {get_num_params(conv1)}")
out = conv1(inp)
print(out.size())
print("")

conv2 = nn.Conv2d(1, 1, kernel_size=2)
print(conv2)
print(f"Number of parameters in conv2: {get_num_params(conv2)}")
out = conv2(inp)
print(out.size())
print("")

conv3 = nn.Conv2d(1, 1, kernel_size=(4, 2))
print(conv3)
print(f"Number of parameters in conv3: {get_num_params(conv3)}")
out = conv3(inp)
print(out.size())
print("")

conv4 = nn.Conv2d(1, 1, kernel_size=1)
print(conv4)
print(f"Number of parameters in conv4: {get_num_params(conv4)}")
out = conv4(inp)
print(out.size())
print("")
