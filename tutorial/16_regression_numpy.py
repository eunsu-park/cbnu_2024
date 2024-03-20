import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

## 랜덤 데이터셋 생성
## y_true = w_true * x + b_true 이며 w_true=0.9, b_true=0.3라고 가정
## x는 0부터 100까지 200개의 데이터 포인트로 구성
## y_target는 w_true * x + b_true에 노이즈를 추가한 값
## 모델 훈련에는 x, y_target의 페어를 사용

w_true, b_true = 0.9, 0.3

x = np.linspace(0, 100, 200)
x = np.expand_dims(x, axis=1)
y_true = w_true * x + b_true
noise = np.random.normal(0, 5, size=y_true.shape)
y_target = y_true + noise
x = x.astype(np.float32)
y_target = y_target.astype(np.float32)
print(x.shape, y_target.shape)
print(x.dtype, y_target.dtype)

plt.plot(x, y_target, 'o')
plt.plot(x, y_true, 'r-')
plt.show()

## mse : 평균 제곱 오차(Mean Squared Error, MSE)

def mse(y_pred, y_target):
    return np.mean(np.square(y_pred - y_target))

## NumPy를 이용한 선형 회귀
## w_pred, b_pred는 0.0으로 초기화
## learning_rate = 0.0002로 설정

w_pred, b_pred = 0.0, 0.0
y_pred = w_pred * x + b_pred
loss = mse(y_pred, y_target)
learning_rate = 0.0002
print(f"Initial w: {w_pred:5.3f}, b: {b_pred:5.3f}, loss: {loss:5.3f}")

for epoch in range(10000):
    y_pred = w_pred * x + b_pred
    w_pred -= 2 * learning_rate * ((y_pred - y_target) * x).mean()
    b_pred -= 2 * learning_rate * (y_pred - y_target).mean()
    
    if epoch % 1000 == 0 :
        y_pred = w_pred * x + b_pred
        loss = np.square(y_pred - y_target).mean()
        print(f"Epoch: {epoch}, w: {w_pred:5.3f}, b: {b_pred:5.3f}, loss: {loss:5.3f}")

y_pred = w_pred * x + b_pred
loss = np.square(y_pred - y_target).mean()
print(f"Final w: {w_pred:5.3f}, b: {b_pred:5.3f}, loss: {loss:5.3f}")
