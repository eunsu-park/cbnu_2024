import numpy as np
import torch

x = np.random.rand(1, 32, 32)
y = np.random.rand(1, 32, 32)
z = np.concatenate([x, y], axis=0)
print(z.shape)
w = np.concatenate([x, y], axis=1)
print(w.shape)

x = torch.randn(1, 32, 32)
y = torch.randn(1, 32, 32)
z = torch.cat([x, y], dim=0)
print(z.shape)
w = torch.cat([x, y], dim=1)
print(w.shape)

## np.concatenate() : 차원 합치기 함수
## torch.cat() : 차원 합치기 함수, torch.concat()과 같은 기능을 함
