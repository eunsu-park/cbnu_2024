import numpy as np
import torch

array = np.random.randn(2, 2)
print(array)
print(array.shape)
array = array.reshape(4, 1)
print(array)
print(array.shape)
array = array.reshape(1, 4)
print(array)
print(array.shape)
array = array.reshape(2, 2)
print(array)
print(array.shape)

tensor = torch.randn(2, 2)
print(tensor)
print(tensor.shape)
tensor = tensor.reshape(4, 1)
print(tensor)
print(tensor.shape)
tensor = tensor.reshape(1, 4)
print(tensor)
print(tensor.shape)
tensor = tensor.reshape(2, 2)
print(tensor)
print(tensor.shape)

## tensor.reshape 대신 tensor.view 사용 가능
