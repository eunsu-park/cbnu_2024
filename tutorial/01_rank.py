# 01_rank.py
# 데이터의 차원 확인

import numpy as np

print("Rank-0 Tensor")
x = np.array(1) # np.array() : numpy array 생성 함수
print(f"Rank: {x.ndim}") # x.ndim : 차원 수
print(f"Shape: {x.shape}") # x.shape : 차원의 크기
print(x)
print("")

print("Rank-1 Tensor")
x = np.array([1,2,3,4,5])
print(f"Rank: {x.ndim}")
print(f"Shape: {x.shape}")
print(x[0]) ## 0번째 원소
print("")

print("Rank-2 Tensor")
x = np.array([[11,12,13,14,15],
              [21,22,23,24,25],
              [31,32,33,34,35]])
print(f"Rank: {x.ndim}")
print(f"Shape: {x.shape}")
print(x[0]) ## 0번째 행
print(x[0, -1]) ## 0번째 행의 마지막 열
print(x[0][-1]) ## 0번째 행의 마지막 열, 위와 동일한 결과
print("")

print("Rank-3 Tensor")
x = np.array([[[101,102,103,104,105],
               [111,112,113,114,115],
               [121,122,123,124,125]],
              [[201,202,203,204,205],
               [211,212,213,214,215],
               [221,222,223,224,225]]])
print(f"Rank: {x.ndim}")
print(f"Shape: {x.shape}")
print(x[0]) ## 0번째 행렬
print(x[0][1]) ## 0번째 행렬의 1번째 행
print(x[0][1][2]) ## 0번째 행렬의 1번째 행의 2번째 열
print("")
