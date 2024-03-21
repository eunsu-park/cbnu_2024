# 10_cautions_1.py
# 차원 변경 시 주의사항

import numpy as np
import torch

# numpy array
x = np.arange(4)
print(x)
print("")
y = x.reshape(2, 2)
print(y)
print("")
x[0] = 9 ## x의 값을 변경
print(y) ## y와 x는 같은 메모리를 참조하기 때문에 x의 값이 변하면 y의 값도 변함

# torch tensor
x = torch.arange(4)
print(x)
print("")
y = x.reshape(2, 2)
print(y)
print("")
x[0] = 9
print(y)
print("")
