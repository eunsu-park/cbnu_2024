## 내장 라이브러리 import
import os
import time

## 외부 라이브러리 import
import torch
import numpy as np

## 제작한 모듈 import
from options import TrainOptions
from networks import define_network, define_criterion, define_optimizer
from pipeline import define_dataset
from utils import fix_seed, get_num_params

## options 설정
opt = TrainOptions().parse()

## seed 고정
fix_seed(opt.seed)

## device 설정
if torch.cuda.is_available() :
    if opt.gpu_id != -1 :
        device = torch.device(f"cuda:{opt.gpu_id}")
    else :
        device = torch.device("cpu")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else :
    device = torch.device("cpu")
print(device)

## network, criterion, optimizer 정의
network = define_network(opt)
criterion = define_criterion(opt).to(device)
optimizer = define_optimizer(network, opt)
print(network)
print(f"Number of parameters : {get_num_params(network)}")
print(criterion)
print(optimizer)

## dataset, dataloader 정의
dataset, dataloader = define_dataset(opt)
print(len(dataset), len(dataloader))

## 학습 결과 저장 디렉토리 생성
save_dir = os.path.join(opt.save_root, opt.name)
os.makedirs(opt.save_dir, exist_ok=True)




