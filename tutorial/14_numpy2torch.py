# 14_numpy2torch.py
# numpy array to torch tensor

import numpy as np
import torch

array = np.array([1, 2, 3])
print(array)
print(array.dtype)
print("")

tensor = torch.tensor(array) # torch.tensor()
print(tensor)
print(tensor.dtype)
print("")

tensor = torch.Tensor(array) # torch.Tensor(), 위와 다르게 float tensor로 변환
print(tensor)
print(tensor.dtype)
print("")

tensor = torch.as_tensor(array) # torch.as_tensor()
print(tensor)
print(tensor.dtype)
print("")

tensor = torch.from_numpy(array) # torch.from_numpy()
print(tensor)
print(tensor.dtype)
print("")
