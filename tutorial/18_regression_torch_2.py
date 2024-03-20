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


## torch.nn을 이용한 선형 회귀
## torch.nn.Linear를 사용하여 모델 정의
## torch.nn.MSELoss를 사용하여 손실 함수 정의
## torch.optim.SGD를 사용하여 optimizer 정의

model = torch.nn.Linear(1, 1, bias=True)
loss_function = torch.nn.MSELoss()
learning_rate = 0.0002
optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate)
print(model)
print(optimizer)
print(optimizer)

x = torch.from_numpy(x)
y_target = torch.from_numpy(y_target)
print(x.shape, y_target.shape)
print(x.dtype, y_target.dtype)

y_pred = model(x)
loss = loss_function(y_pred, y_target)
w_pred = model.weight.data
b_pred = model.bias.data
print(f"Initial w: {w_pred.item():5.3f}, b: {b_pred.item():5.3f}, loss: {loss.item():5.3f}")

for epoch in range(10000):
    y_pred = model(x)
    loss = loss_function(y_pred, y_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0 :
        y_pred = model(x)
        loss = loss_function(y_pred, y_target)
        w_pred = model.weight.data
        b_pred = model.bias.data
        loss = torch.mean(torch.square(y_pred - y_target))
        print(f"Epoch: {epoch}, w: {w_pred.item():5.3f}, b: {b_pred.item():5.3f}, loss: {loss.item():5.3f}")

y_pred = model(x)
loss = loss_function(y_pred, y_target)
w_pred = model.weight.data
b_pred = model.bias.data
print(f"Final w: {w_pred.item():5.3f}, b: {b_pred.item():5.3f}, loss: {loss.item():5.3f}")
