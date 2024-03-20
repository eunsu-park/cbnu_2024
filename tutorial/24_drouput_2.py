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

model = []
model += [nn.Linear(2, 4)]
model += [nn.ReLU()]
model += [nn.Dropout(p=0.5)]
model += [nn.Linear(4, 3)]
model += [nn.Sigmoid()]
model = nn.Sequential(*model)
print(model)
print(f"Number of parameters in model: {get_num_params(model)}")

inp = torch.randn(128, 2)
out = model(inp)
print(out.size())
