# 15_torch2numpy.py
# torch tensor to numpy array
# NVidia GPU가 없는 경우 중간에 에러가 발생함

import numpy as np
import torch

tensor = torch.FloatTensor([1, 2, 3])
print(tensor)
print(tensor.dtype)
print("")

array = tensor.detach().numpy() ## tensor.detach() : tensor에서 연산 기록 제거 함수
print(array)
print(array.dtype)
print("")

tensor = torch.cuda.FloatTensor([1, 2, 3])
print(tensor)
print(tensor.dtype)
print(tensor.device)
print("")
array = tensor.detach().cpu().numpy() ## tensor.detach() : tensor에서 연산 기록 제거 함수
print(array)
print(array.dtype)
print("")
