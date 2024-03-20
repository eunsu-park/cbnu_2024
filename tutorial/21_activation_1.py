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


## ReLU : y = max(0, x)
relu = nn.ReLU()
print(relu)
print(f"Number of parameters in relu: {get_num_params(relu)}")

## Sigmoid : y = 1 / (1 + exp(-x))
## 0과 1 사이의 값을 출력하는 함수로, 이진 분류 문제에서 출력층에 주로 사용됨
sigmoid = nn.Sigmoid()
print(sigmoid)
print(f"Number of parameters in sigmoid: {get_num_params(sigmoid)}")

## Tanh : y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
## -1 <= y <= 1
tanh = nn.Tanh()
print(tanh)
print(f"Number of parameters in tanh: {get_num_params(tanh)}")


inp = torch.randn(128, 2)
out_relu = relu(inp)
print(out_relu.size())
out_sigmoid = sigmoid(inp)
print(out_sigmoid.size())
out_tanh = tanh(inp)
print(out_tanh.size())

plt.plot(inp.numpy(), out_relu.numpy(), 'o', label='ReLU', color='r')
plt.plot(inp.numpy(), out_sigmoid.numpy(), 'x', label='Sigmoid', color='g')
plt.plot(inp.numpy(), out_tanh.numpy(), '*', label='Tanh', color='b')
plt.legend()
plt.grid(True)
plt.show()
