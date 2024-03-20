import numpy as np
import torch

x = np.random.rand(32, 32, 3)
print(x.shape)
y = np.transpose(x, (2, 0, 1))
print(y.shape)

x = torch.randn(32, 32, 3)
print(x.shape)
y = x.permute(2, 0, 1)
print(y.shape)

## np.transpose() : 차원 순서 변경 함수
## tensor.permute() : 차원 순서 변경 함수
