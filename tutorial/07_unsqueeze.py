# 07_unsqueeze.py
# 차원 추가 혹은 축소

import numpy as np
import torch

# numpy array
x = np.random.rand(32, 32)
y = np.expand_dims(x, axis=0) # np.expand_dims() : 차원 추가 함수, axis=0은 0번째 차원에 추가
print(y.shape)
print("")
z = np.squeeze(y, axis=0) # np.squeeze() : 차원 제거 함수, axis=0은 0번째 차원 제거, z = y.squeeze(0) 로 사용 가능
print(z.shape)
print("")

# torch tensor
x = torch.randn(32, 32)
y = torch.unsqueeze(x, 0) # torch.unsqueeze() : 차원 추가 함수, 0번째 차원에 추가, y = x.unsqueeze(0) 로 사용 가능
print(y.shape)
print("")
z = torch.squeeze(y, 0) # torch.squeeze() : 차원 제거 함수, 0번째 차원 제거, z = y.squeeze(0) 로 사용 가능
print(z.shape)
print("")
