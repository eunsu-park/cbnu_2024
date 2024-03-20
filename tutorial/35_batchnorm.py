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

## BatchNorm2d
## 배치 정규화를 수행하는 레이어

inp = torch.randn(128, 3, 16, 16)
layer = nn.BatchNorm2d(3)
out = layer(inp)
print(out.size())
print(f"Number of parameters in layer: {get_num_params(layer)}")

inp = torch.randn(128, 64, 16, 16)
layer = nn.BatchNorm2d(64)
out = layer(inp)
print(out.size())
print(f"Number of parameters in layer: {get_num_params(layer)}")
