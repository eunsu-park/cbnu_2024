import numpy as np
import torch

x = np.arange(4)
print(x)
print(x.shape)
y = x.copy().reshape(2, 2)
print(y)
print(y.shape)
x[0] = 9
print(y)

x = torch.arange(4)
print(x)
print(x.shape)
y = x.clone().reshape(2, 2)
print(y)
print(y.shape)
x[0] = 9
print(y)

## copy() : 데이터를 복사하여 새로운 메모리에 저장
## clone() : 데이터를 복사하여 새로운 메모리에 저장
## x의 값이 변해도 y의 값은 변하지 않음
