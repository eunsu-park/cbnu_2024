import numpy as np
import torch

## torch.cuda.is_available() : GPU 사용 가능 여부 확인 함수

tensor = torch.Tensor([1, 2, 3])
print(tensor.device)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

cpu = torch.tensor([1, 2, 3])
print(cpu)

gpu = torch.cuda.FloatTensor([1, 2, 3])
print(gpu)

tensor = torch.tensor([1, 2, 3], device=device)
print(tensor)

cpu = torch.FloatTensor([1, 2, 3])
print(cpu)

gpu = cpu.cuda()
print(gpu)

gpu2cpu = gpu.cpu()
print(gpu2cpu)

cpu2gpu = cpu.to("cuda")
print(cpu2gpu)
