# 08_permute.py
# 차원 순서 변경

import numpy as np
import torch

# numpy array
x = np.random.rand(32, 32, 3)
print(x.shape)
print("")
y = np.transpose(x, (2, 0, 1)) # np.transpose() : 차원 순서 변경 함수, y = x.transpose(2, 0, 1)로 사용 가능
# (2, 0, 1)은 0번째 차원을 2번째로, 1번째 차원을 0번째로, 2번째 차원을 1번째로 변경함을 의미함
print(y.shape)
print("")

# torch tensor
x = torch.randn(32, 32, 3)
print(x.shape)
print("")
y = torch.permute(x, (2, 0, 1)) # torch.permute() : 차원 순서 변경 함수, y = x.permute(2, 0, 1)로 사용 가능
print(y.shape)
print("")
#주의사항 : torch.transpose는 2개의 차원만 변경 가능, torch.permute는 모든 차원 변경 가능
