import numpy as np
import torch

array = np.array([1, 2, 3])
tensor = torch.tensor(array)
print(tensor)
tensor = torch.Tensor(array)
print(tensor)
tensor = torch.as_tensor(array)
print(tensor)
tensor = torch.from_numpy(array)
print(tensor)
