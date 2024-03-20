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

## 1x1 커널을 사용한 Conv2d Layer
## 입력 이미지의 채널 수를 변경할 수 있음

inp1 = torch.randn(1, 1, 256, 256)
layer1 = nn.Conv2d(1, 3, kernel_size=1, bias=True)
out1 = layer1(inp1)
print(out1.size())
print(f"Number of parameters in layer1: {get_num_params(layer1)}")

inp2 = torch.randn(1, 3, 256, 256)
layer2 = nn.Conv2d(3, 1, kernel_size=1, bias=True)
out2 = layer2(inp2)
print(out2.size())
print(f"Number of parameters in layer2: {get_num_params(layer2)}")
