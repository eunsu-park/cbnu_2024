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

## MaxPool2d
## kernel_size, stride, padding : Conv2d와 동일

inp = torch.randn(128, 1, 16, 16)

layer1 = nn.MaxPool2d(kernel_size=2)
out1 = layer1(inp)
print(out1.size())
print(f"Number of parameters in layer1: {get_num_params(layer1)}")

layer2 = nn.MaxPool2d(kernel_size=2, stride=2)
out2 = layer2(inp)
print(out2.size())
print(f"Number of parameters in layer2: {get_num_params(layer2)}")

layer3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
out3 = layer3(inp)
print(out3.size())
print(f"Number of parameters in layer3: {get_num_params(layer3)}")
