import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def get_num_params(model):
    """
    모델(레이어)의 파라미터 수를 계산하는 함수
    
    Args:
        model : torch.nn.Module
    Returns:
        num_params : int
    """
    return sum([p.numel() for p in model.parameters()])

drouput = nn.Dropout(p=0.5, inplace=True)
print(drouput)
print(f"Number of parameters in drouput: {get_num_params(drouput)}")

inp = torch.randn(128, 20)
print(inp)
out = drouput(inp)
print(out)
print(out.size())
