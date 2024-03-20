import numpy as np
import torch

array = np.random.randn(2, 2)
print(array)
print(array.shape)
array = array.reshape(1, -1)
print(array)
print(array.shape)
array = array.reshape(-1, 1)
print(array)
print(array.shape)

tensor = torch.randn(2, 2)
print(tensor)
print(tensor.shape)
tensor = tensor.view(1, -1)
print(tensor)
print(tensor.shape)
tensor = tensor.view(-1, 1)
print(tensor)
print(tensor.shape)

## -1을 사용하면 나머지 차원을 자동으로 계산함을 의미
