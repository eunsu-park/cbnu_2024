# 35_batchnorm.py
# BatchNorm2d

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

inp = torch.randn(128, 3, 16, 16)
layer = nn.BatchNorm2d(3) # 3개의 채널을 갖는 2D 이미지에 대한 배치 정규화
out = layer(inp)
print(out.size())
print(f"Number of parameters in layer: {get_num_params(layer)}")
print("")

inp = torch.randn(128, 64, 16, 16)
layer = nn.BatchNorm2d(64) # 64개의 채널을 갖는 2D 이미지에 대한 배치 정규화
out = layer(inp)
print(out.size())
print(f"Number of parameters in layer: {get_num_params(layer)}")
print("")
