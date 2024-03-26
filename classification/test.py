## 내장 라이브러리 import
import os
import time

## 외부 라이브러리 import
import torch
import numpy as np

## 제작한 모듈 import
from options import TestOptions
from networks import define_network, define_criterion
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
state = torch.load(f"{save_dir}/model_{opt.epoch_test:04d}.pt", map_location=device)
print(state)

network = define_network(opt, state_dict=state["network"]).to(device)
print(network)

criterion = define_criterion(opt).to(device)

dataset, dataloader = define_dataset(opt)
print(dataset, dataloader)
print(len(dataset), len(dataloader))

losses = []
correct = 0

labels = []
predictions = []

network.eval()
with torch.no_grad():
    for idx, (image, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)
        output = network(image)
        loss = criterion(output, label)
        losses.append(loss.item())

        label = label.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        labels.append(label)
        predictions.append(output)

print(f"Average Loss: {np.mean(losses):.4f}")

labels = np.concatenate(labels, 0)
predictions = np.concatenate(predictions, 0)
print(labels.shape)
print(predictions.shape)

labels_class = np.argmax(labels, 1)
predictions_class = np.argmax(predictions, 1)
print(labels_class.shape)
print(predictions_class.shape)

correct = np.sum(labels_class == predictions_class)
print(f"Accuracy: {correct / len(dataset):.4f}")
