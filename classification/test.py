## 내장 라이브러리 import
import os
import time

## 외부 라이브러리 import
import torch
import numpy as np

## 제작한 모듈 import
from options import TestOptions
from networks import define_network
from pipeline import define_dataset

opt = TestOptions().parse()  # get test options

## device 설정
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)

save_dir = os.path.join(opt.save_root, opt.name)
state = torch.load(f"{save_dir}/model_{opt.epoch_test:04d}.pt")
print(state)

network = define_network(opt, state_dict=state["network"]).to(device)
print(network)

dataset, dataloader = define_dataset(opt)
print(dataset, dataloader)
print(len(dataset), len(dataloader))