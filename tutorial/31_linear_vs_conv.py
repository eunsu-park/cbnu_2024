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

## 128x128 이미지에 대한 Linear Layer와 Conv2d Layer의 파라미터 수 비교
## in_channels, out_channels = 1, 3
## kernel_size : 3으로 고정

inp = torch.randn(1, 1, 128, 128)

linear = nn.Linear(128*128, 64*64*3, bias=False)
print(linear)
out_linear = linear(inp.reshape(1, 128*128))
out_linear = out_linear.reshape(1, 3, 64, 64)
print(out_linear.size())
nb_params_linear = sum(p.numel() for p in linear.parameters())
print(nb_params_linear)

conv2d = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1, bias=False)
print(conv2d)
out_conv2d = conv2d(inp)
print(out_conv2d.size())
nb_params_conv2d = sum(p.numel() for p in conv2d.parameters())
print(nb_params_conv2d)

## 출력 feature map의 갯수를 크게 늘려도 Parameter 수가 크게 늘지 않음
conv2d = nn.Conv2d(1, 1024, kernel_size=3, stride=2, padding=1, bias=False)
print(conv2d)
out_conv2d = conv2d(inp)
print(out_conv2d.size())
nb_params_conv2d = sum(p.numel() for p in conv2d.parameters())
print(nb_params_conv2d)
