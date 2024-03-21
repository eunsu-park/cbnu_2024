# 11_cautions_2.py
# 차원 변경 시 주의사항 - 데이터 복사

import numpy as np
import torch

# numpy array
x = np.arange(4)
print(x)
print("")
y = x.copy().reshape(2, 2) # x.copy() : 데이터를 복사하여 새로운 메모리에 저장
print(y)
print("")
x[0] = 9 ## x의 값을 변경
print(y) ## y와 x는 다른 메모리를 참조하기 때문에 x의 값이 변해도 y의 값은 변하지 않음

# torch tensor
x = torch.arange(4)
print(x)
print("")
y = x.clone().reshape(2, 2) # x.clone() : 데이터를 복사하여 새로운 메모리에 저장
print(y)
print("")
x[0] = 9
print(y)
print("")
