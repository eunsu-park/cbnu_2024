## 06_reshape_2.py
## reshape(view) 사용 시 -1 활용법
## -1을 사용하면 나머지 차원을 자동으로 계산함을 의미

import numpy as np
import torch

# numpy array
array = np.random.randn(2, 2)
print(array)
print(array.shape)
print("")
array = array.reshape(1, -1) # 앞 차원을 1로 고정하고 뒤 차원을 자동으로 계산
print(array)
print(array.shape)
print("")
array = array.reshape(-1, 1) # 뒤 차원을 1로 고정하고 앞 차원을 자동으로 계산
print(array)
print(array.shape)
print("")

# torch tensor
tensor = torch.randn(2, 2)
print(tensor)
print(tensor.shape)
print("")
tensor = tensor.view(1, -1)
print(tensor)
print(tensor.shape)
print("")
tensor = tensor.view(-1, 1)
print(tensor)
print(tensor.shape)
print("")
