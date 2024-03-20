import numpy as np
import torch

array = np.random.randint(0, 10, (3, 3))
print(array)
print(array.dtype)
array = array.astype(np.int32)
print(array)
print(array.dtype)
array = array.astype(np.float64)
print(array)
print(array.dtype)

tensor = torch.randint(10, (3, 3))
print(tensor)
print(tensor.dtype)
tensor = tensor.to(torch.int)
print(tensor)
tensor = tensor.to(torch.double)
print(tensor)

## np.random.randint() : 정수 난수 생성 함수
## torch.randint() : 정수 난수 생성 함수
## np.astype() : 데이터 타입 변경 함수
## tensor.to() : 데이터 타입 변경 함수
