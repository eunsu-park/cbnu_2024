# 13_device.py
# CPU와 GPU 사용하기
# NVidia GPU가 없는 경우 중간에서 에러가 발생

import torch

tensor = torch.tensor([1, 2, 3])
print(tensor.device) # tensor.device : tensor의 device 확인 함수

device = "cuda" if torch.cuda.is_available() else "cpu" # torch.cuda.is_available() : GPU 사용 가능 여부 확인 함수
print(device)

cpu = torch.tensor([1, 2, 3])
print(cpu)

gpu = torch.cuda.FloatTensor([1, 2, 3]) # torch.cuda.FloatTensor() : torch float tensor를 GPU에 올리는 함수
print(gpu)

tensor = torch.tensor([1, 2, 3], device=device) # device 옵션을 사용하여 tensor를 GPU에 올릴 수 있음
print(tensor)

cpu = torch.FloatTensor([1, 2, 3])
print(cpu)

gpu = cpu.cuda() # tensor.cuda() : tensor를 GPU에 올리는 함수
print(gpu)

gpu2cpu = gpu.cpu() # tensor.cpu() : tensor를 CPU에 내리는 함수
print(gpu2cpu)

cpu2gpu = cpu.to("cuda") # tensor.to() : tensor를 다른 device로 옮기는 함수
print(cpu2gpu)
