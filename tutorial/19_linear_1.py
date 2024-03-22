# 19_linear_1.py
# Linear Layer

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

linear1 = nn.Linear(2, 4) # input feature의 개수가 2, output feature의 개수가 4
print(linear1) 
print(f"Number of parameters in linear1: {get_num_params(linear1)}")

linear2 = nn.Linear(4, 3) # input feature의 개수가 4, output feature의 개수가 3
print(linear2)
print(f"Number of parameters in linear2: {get_num_params(linear2)}")

inp = torch.randn(128, 2) ## 128개의 데이터 포인트, 각 데이터 포인트는 2개의 값은 갖는 벡터 
out = linear1(inp)
print(out.size())
out = linear2(out)
print(out.size())

inp = torch.randn(256, 2) ## 256개의 데이터 포인트, 각 데이터 포인트는 2개의 값은 갖는 벡터 
out = linear1(inp)
print(out.size())
out = linear2(out)
print(out.size())
