# 02_batch.py
# 데이터를 배치 단위로 불러오는 방법

import numpy as np

# (3, 28, 28) 크기의 데이터가 10000개 있다고 가정
data = np.random.randn(10000, 3, 28, 28) # np.random.randn() : 정규분포를 따르는 난수 생성 함수

# 배치를 완전 수동으로 불러오는 방법
batch_1 = data[:128] # batch_1은 0~127번째 데이터를 불러옴 (128개)
print(batch_1.shape)
batch_2 = data[128:256] # batch_2는 128~255번째 데이터를 불러옴 (128개)
print(batch_2.shape)

# 배치를 반복문으로 불러오는 방법 (n번째 배치만 불러옴)
batch_idx = 0 # batch_idx : 불러올 배치의 인덱스, n번째 배치를 불러옴
batch_size = 128 # batch_size : 배치의 크기
start = batch_idx * batch_size # start : 배치의 시작 인덱스
end = (batch_idx + 1) * batch_size # end : 배치의 끝 인덱스
batch_n = data[start:end] # batch_n : 배치 데이터
print(batch_n.shape)

# 배치를 반복문으로 불러오는 방법 (for문을 이용해 모든 배치를 불러옴) + epoch 개념도 추가
epochs = 5 # epochs : 학습 횟수
batch_size = 128 # batch_size : 배치의 크기
shuffle = True # shuffle : 데이터를 섞을지 여부
nb_batch = data.shape[0] // batch_size # nb_batch : 배치의 수, 편의상 나머지는 제외

for epoch in range(epochs):
    for batch_idx in range(nb_batch): 
        batch_start = batch_idx*batch_size
        batch_end = (batch_idx+1)*batch_size
        batch = data[batch_start:batch_end]
        print(epoch, batch_idx, batch.shape)
    if shuffle is True : ## 한 epoch 에서 모든 배치를 불러온 후 shuffle이 True일 경우 데이터를 섞음
        np.random.shuffle(data) ## np.random.shuffle() : 데이터를 섞는 함수
