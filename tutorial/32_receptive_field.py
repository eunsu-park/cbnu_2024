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

## model1 : 3x3 커널을 사용한 두 개의 Conv2d Layer -> receptive field가 5x5 
## model2 : 5x5 커널을 사용한 한 개의 Conv2d Layer -> receptive field가 5x5

## 두 모델이 같은 크기의 출력을 생성하므로 receptive field가 같음
## model1과 model2의 파라미터 수는 다름

inp = torch.randn(1, 1, 256, 256)

model1 = []
model1 += [nn.Conv2d(1, 1, kernel_size=3, bias=False)]
model1 += [nn.Conv2d(1, 1, kernel_size=3, bias=False)]
model1 = nn.Sequential(*model1)
print(model1)
out1 = model1(inp)
print(out1.size())
print(f"Number of parameters in model1: {get_num_params(model1)}")

model2 = []
model2 += [nn.Conv2d(1, 1, kernel_size=5, bias=False)]
model2 = nn.Sequential(*model2)
print(model2)
out2 = model2(inp)
print(out2.size())
print(f"Number of parameters in model2: {get_num_params(model2)}")
