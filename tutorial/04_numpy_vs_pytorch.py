import numpy as np
import torch

array = np.array([1, 2, 3])
print(array)
print(array.shape)
print(array.dtype)

tensor = torch.tensor([1, 2, 3])
print(tensor)
print(tensor.shape)
print(tensor.dtype)

## torch.Tensor() : torch tensor 생성 함수

array = np.array([1., 2., 3.])
print(array)
print(array.shape)
print(array.dtype)

tensor = torch.tensor([1., 2., 3.])
print(tensor)
print(tensor.shape)
print(tensor.dtype)

array = np.array([[1, 2, 3],
                  [4, 5, 6]])
print(array)
print(array.shape)
print(array.dtype)

tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
print(tensor)
print(tensor.shape)
print(tensor.dtype)

array = np.array([1, 2, 3], dtype=np.int64)
print(array)
print(array.shape)
print(array.dtype)

tensor = torch.LongTensor([1, 2, 3])
print(tensor)
print(tensor.shape)
print(tensor.dtype)

array = np.array([1, 2, 3], dtype=np.float32)
print(array)
print(array.shape)
print(array.dtype)

tensor = torch.FloatTensor([1, 2, 3])
print(tensor)
print(tensor.shape)
print(tensor.dtype)
