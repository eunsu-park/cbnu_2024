


from options import TrainOptions

opt = TrainOptions().parse()

from torch.utils.data import DataLoader
from pipeline import CustomDataset

dataset = CustomDataset(opt.data_root, opt.image_size, opt.is_train)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

print(len(dataset), len(dataloader))

for i, (image, label) in enumerate(dataloader):
    print(image.shape, label.shape)
    if i == 10:
        break

