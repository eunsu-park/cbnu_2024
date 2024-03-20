import numpy as np
import torch

x = np.random.rand(32, 32)
y = np.expand_dims(x, axis=0)
print(y.shape)
z = np.squeeze(y, axis=0)
print(z.shape)

x = torch.randn(32, 32)
y = x.unsqueeze(0)
print(y.shape)
z = y.squeeze(0)
print(z.shape)

## np.expand_dims() : 차원 추가 함수
## np.squeeze() : 차원 제거 함수
## tensor.unsqueeze() : 차원 추가 함수
## tensor.squeeze() : 차원 제거 함수
