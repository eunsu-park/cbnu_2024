# 09_cat.py
# 차원 합치기

import numpy as np
import torch

# numpy array
x = np.random.rand(1, 32, 32)
y = np.random.rand(1, 32, 32)
z = np.concatenate([x, y], axis=0) # np.concatenate() : 차원 합치기 함수, axis=0은 0번째 차원을 기준으로 합침
print(z.shape)
print("")
w = np.concatenate([x, y], axis=1) # axis=1은 1번째 차원을 기준으로 합침
print(w.shape)
print("")

# torch tensor
x = torch.randn(1, 32, 32)
y = torch.randn(1, 32, 32)
z = torch.cat([x, y], dim=0) # torch.cat() : 차원 합치기 함수, dim=0은 0번째 차원을 기준으로 합침
print(z.shape)
print("")
w = torch.cat([x, y], dim=1) # dim=1은 1번째 차원을 기준으로 합침
print(w.shape)
print("")
