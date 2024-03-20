import numpy as np
import torch

tensor = torch.FloatTensor([1, 2, 3])
array = tensor.detach().numpy()
print(array)

tensor = torch.cuda.FloatTensor([1, 2, 3])
array = tensor.detach().cpu().numpy()
print(array)

## tensor.detach() : tensor에서 연산 기록 제거 함수
