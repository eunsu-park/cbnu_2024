# 21_activation_1.py
# Activation functions

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

relu = nn.ReLU() # ReLU : y = max(0, x)
print(relu)
print(f"Number of parameters in relu: {get_num_params(relu)}") # ReLU 레이어는 학습 가능한 파라미터가 없음

sigmoid = nn.Sigmoid() # Sigmoid : y = 1 / (1 + exp(-x)), 0 <= y <= 1, 이진 분류 문제에서 출력층에 주로 사용됨
print(sigmoid)
print(f"Number of parameters in sigmoid: {get_num_params(sigmoid)}") # Sigmoid 레이어는 학습 가능한 파라미터가 없음

## -1 <= y <= 1
tanh = nn.Tanh() # Tanh : y = (exp(x) - exp(-x)) / (exp(x) + exp(-x)), -1 <= y <= 1
print(tanh)
print(f"Number of parameters in tanh: {get_num_params(tanh)}") # Tanh 레이어는 학습 가능한 파라미터가 없음

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
plt.xlim(-2, 2)
plt.ylim(-1, 1)
plt.legend()
plt.grid(True)
plt.show()
