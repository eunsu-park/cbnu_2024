# 22_activation_2.py
# Activation Function

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

model = []
model += [nn.Linear(2, 4)]
model += [nn.ReLU()] # 일반적인 신경망에서 activation function을 연산 후에 적용
model += [nn.Linear(4, 3)]
model += [nn.Sigmoid()] # 신경망의 마지막에 sigmoid를 사용하였기 때문에 신경망의 최종 출력이 0과 1 사이의 값으로 나옴
model = nn.Sequential(*model)
print(model)
print(f"Number of parameters in model: {get_num_params(model)}")

inp = torch.randn(128, 2)
out = model(inp)
print(out.size())
