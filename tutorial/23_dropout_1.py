# 23_dropout_1.py
# Dropout Layer

import torch
import torch.nn as nn

def get_num_params(model):
    """
    모델(레이어)의 파라미터 수를 계산하는 함수
    
    Args:
        model : torch.nn.Module
    Returns:
        num_params : int
    """
    return sum([p.numel() for p in model.parameters()])

drouput = nn.Dropout(p=0.5, inplace=True) # p : 드롭아웃 확률, inplace : True이면 드롭아웃을 적용한 결과를 입력으로 바로 반환
print(drouput)
print(f"Number of parameters in drouput: {get_num_params(drouput)}") # 드롭아웃 레이어는 학습 가능한 파라미터가 없음

inp = torch.randn(32, 16)
print("Before dropout")
print(inp)
print("")
out = drouput(inp)
print("After dropout")
print(inp) # inplace=True이기 때문에 드롭아웃을 적용한 결과를 입력으로 바로 반환
print("")
