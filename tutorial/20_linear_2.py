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


## nn.Sequential : 여러 레이어를 순차적으로 쌓아서 모델을 만드는 클래스

model = []
model += [nn.Linear(2, 4)]
model += [nn.Linear(4, 3)]
model = nn.Sequential(*model)
print(model)
print(f"Number of parameters in model: {get_num_params(model)}")

inp = torch.randn(128, 2) ## 128개의 데이터 포인트, 각 데이터 포인트는 2개의 값은 갖는 벡터 
out = model(inp)
print(out.size())

inp = torch.randn(256, 2) ## 256개의 데이터 포인트, 각 데이터 포인트는 2개의 값은 갖는 벡터 
out = model(inp)
print(out.size())
