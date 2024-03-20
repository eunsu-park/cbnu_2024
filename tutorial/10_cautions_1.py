import numpy as np
import torch

x = np.arange(4)
print(x)
print(x.shape)
y = x.reshape(2, 2)
print(y)
print(y.shape)
x[0] = 9
print(y)

x = torch.arange(4)
print(x)
print(x.shape)
y = x.reshape(2, 2)
print(y)
print(y.shape)
x[0] = 9
print(y)

## y와 x는 같은 메모리를 참조하여
## x의 값이 변하면 y의 값도 변함
