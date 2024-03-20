import numpy as np

# np.random.randn() : 정규분포를 따르는 난수 생성 함수
# 10000개의 데이터 생성, 각 데이터는 (3, 28, 28) 크기를 가지고 있음
data = np.random.randn(10000, 3, 28, 28)


batch_1 = data[:128] # batch_1은 0~127번째 데이터를 불러옴 (128개)
print(batch_1.shape)
batch_2 = data[128:256] # batch_2는 128~255번째 데이터를 불러옴 (128개)
print(batch_2.shape)


batch_idx = 0 # batch_idx : 불러올 배치의 인덱스, n번째 배치를 불러옴
batch_size = 128 # batch_size : 배치의 크기
start = batch_idx * batch_size # start : 배치의 시작 인덱스
end = (batch_idx + 1) * batch_size # end : 배치의 끝 인덱스
batch_n = data[start:end] # batch_n : 배치 데이터
print(batch_n.shape)


epochs = 10 # epochs : 학습 횟수
batch_size = 128 # batch_size : 배치의 크기
shuffle = True # shuffle : 데이터를 섞을지 여부
nb_batch = data.shape[0] // batch_size # nb_batch : 배치의 수, 편의상 나머지는 제외

for epoch in range(epochs):
    for batch_idx in range(nb_batch):
        batch_start = batch_idx*batch_size
        batch_end = (batch_idx+1)*batch_size
        batch = data[batch_start:batch_end]
        print(epoch, batch_idx, batch.shape)
    if shuffle is True :
        np.random.shuffle(data)
