# 05_reshape_1.py
# 데이터의 차원 변경

import numpy as np
import torch

# numpy array
array = np.random.randn(2, 2)
print(array)
print(array.shape)
print("")
array = np.reshape(array, (1, 4))
print(array)
print(array.shape)
print("")
array = array.reshape(1, 4) # np.reshape() 대신 array.reshape() 사용 가능
print(array)
print(array.shape)
print("")
array = np.reshape(array, (2, 2))
print(array)
print(array.shape)
print("")

# torch tensor
tensor = torch.randn(2, 2)
print(tensor)
print(tensor.shape)
print("")
tensor = torch.reshape(tensor, (4, 1))
print(tensor)
print(tensor.shape)
print("")
tensor = tensor.reshape(1, 4) # tensor.reshape() 대신 torch.reshape() 사용 가능
print(tensor)
print(tensor.shape)
print("")
tensor = tensor.view(2, 2) # tensor.reshape() 대신 tensor.view() 사용 가능, torch.view는 없음
print(tensor)
print(tensor.shape)
print("")
