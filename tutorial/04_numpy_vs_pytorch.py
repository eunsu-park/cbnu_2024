# 04_numpy_vs_pytorch.py
# numpy array와 pytorch tensor의 차이점

import numpy as np
import torch

# int64, rank-1
array = np.array([1, 2, 3])
tensor = torch.tensor([1, 2, 3]) # torch.tensor() : torch tensor 생성 함수
print(array.shape, array.dtype)
print(tensor.shape, tensor.dtype)
print("")

# float64, rank-1
array = np.array([1., 2., 3.])
tensor = torch.tensor([1., 2., 3.])
print(array.shape, array.dtype)
print(tensor.shape, tensor.dtype)
print("")

# int64, rank-2
array = np.array([[1, 2, 3], [4, 5, 6]])
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(array.shape, array.dtype)
print(tensor.shape, tensor.dtype)
print("")

# float64, rank-2
array = np.array([1, 2, 3], dtype=np.int64)
tensor = torch.LongTensor([1, 2, 3]) # torch.LongTensor() : torch long tensor 생성 함수
print(array.shape, array.dtype)
print(tensor.shape, tensor.dtype)
print("")

# float32, rank-1
array = np.array([1, 2, 3], dtype=np.float32)
tensor = torch.FloatTensor([1, 2, 3]) # torch.FloatTensor() : torch float tensor 생성 함수
print(array.shape, array.dtype)
print(tensor.shape, tensor.dtype)
print("")
