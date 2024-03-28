## 내장 라이브러리 import
import os
import time

## 외부 라이브러리 import
import torch
import numpy as np

## 제작한 모듈 import
from options import TrainOptions
from networks import define_discriminator, define_generator
from pipeline import define_dataset
from utils import fix_seed, get_num_params

## options 설정
opt = TrainOptions().parse()

## seed 고정
fix_seed(opt.seed)

## device 설정
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)

## discriminator, generator 정의
discriminator = define_discriminator(opt).to(device)
generator = define_generator(opt).to(device)
print(discriminator)
print(generator)

## optimizer 정의
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
print(optimizer_D, optimizer_G)

## criterion 정의
criterion = torch.nn.BCELoss().to(device)
print(criterion)
l1_criterion = torch.nn.L1Loss().to(device)
print(l1_criterion)

## dataset, dataloader 정의
dataset, dataloader = define_dataset(opt)
print(len(dataset), len(dataloader))

## 학습 결과 저장 디렉토리 생성
save_dir = os.path.join(opt.save_root, opt.name)
save_dir_model = os.path.join(save_dir, "model")
save_dir_image = os.path.join(save_dir, "image")
os.makedirs(save_dir_model, exist_ok=True)
os.makedirs(save_dir_image, exist_ok=True)

## 학습 시작
discriminator.train()
generator.train()
iters = 0
epochs = 0
losses_D = []
losses_G = []
losses_L = []
t0 = time.time()

while epochs < opt.num_epochs:

    for idx, (inp, tar) in enumerate(dataloader):
        inp = inp.to(device)
        tar = tar.to(device)
        gen = generator(inp)

        ## Discriminator 학습
        optimizer_D.zero_grad()
        pred_real = discriminator(tar)
        pred_fake = discriminator(gen.detach())
        loss_D_real = criterion(pred_real, torch.ones_like(pred_real))
        loss_D_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()
        losses_D.append(loss_D.item())

        ## Generator 학습
        optimizer_G.zero_grad()
        pred_fake = discriminator(gen)
        loss_G = criterion(pred_fake, torch.ones_like(pred_fake))
        loss_L = l1_criterion(gen, tar)
        loss = loss_G + opt.lamb * loss_L
        loss.backward()
        optimizer_G.step()
        losses_G.append(loss_G.item())
        losses_L.append(loss_L.item())

        iters += 1

        if iters % 100 == 0:
            print(f"Epoch [{epochs}/{opt.num_epochs}], Step [{iters}], Loss_D: {np.mean(losses_D):.4f}, Loss_G: {np.mean(losses_G):.4f}, Loss_L: {np.mean(losses_L):.4f}, Time: {time.time()-t0:.4f}")
            losses_D = []
            losses_G = []
            losses_L = []
            t0 = time.time()
