# 12_change_dtype.py
# 데이터 타입(자료형) 변경

import numpy as np
import torch

# numpy array
array = np.random.randint(0, 10, (3, 3)) # np.random.randint() : 정수 난수 생성 함수
print(array)
print(array.dtype)
print("")
array = array.astype(np.int32) # array.astype() : 데이터 타입 변경 함, array = np.int32(array)로 사용 가능
print(array)
print(array.dtype)
print("")
array = array.astype(np.float64)
print(array)
print(array.dtype)
print("")

# torch tensor
tensor = torch.randint(10, (3, 3)) # torch.randint() : 정수 난수 생성 함수
print(tensor)
print(tensor.dtype)
print("")
tensor = tensor.to(torch.int) # tensor.to() : 데이터 타입 변경 함수
print(tensor)
print(tensor.dtype)
print("")
tensor = tensor.to(torch.double)
print(tensor)
print(tensor.dtype)
print("")
