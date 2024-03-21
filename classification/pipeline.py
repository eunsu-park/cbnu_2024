import os, glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from imageio import imread
from skimage.transform import resize
import numpy as np


class LoadData:
    def __call__(self, filepath):
        image = imread(filepath)[:,:,0]
        flare_class = filepath.split(os.sep)[-2]
        return image, flare_class


class ResizeData:
    def __init__(self, image_size=224):
        if isinstance(image_size, int) :
            image_size = (image_size, image_size)
        self.image_size = image_size

    def __call__(self, image):
        image = resize(image, self.image_size, order=1, mode="constant", preserve_range=True)
        if len(image.shape) == 2 :
            image = np.expand_dims(image, axis=0)
        return image


class NormalizeData:
    def __call__(self, image) :
        return image / 255.0


class MakeLabel:
    def __call__(self, flare_class):
        if flare_class in ['C', 'M', 'X'] :
            label = [0., 1.]
        elif ['B', 'A', 'Non'] :
            label = [1., 0.]
        else :
            raise ValueError(f"Invalid flare class : {flare_class}")
        return label


class ToTensor:
    def __call__(self, data):
        return torch.tensor(data, dtype=torch.float32)    


class CustomDataset(Dataset):
    def __init__(self, data_root, image_size, is_train=True):
        if is_train is True :
            pattern = f"{data_root}/train/*/*/*.png"
        else :
            pattern = f"{data_root}/test/*/*/*.png"
        self.list_data = glob(pattern)
        self.nb_data = len(self.list_data)
        self.transform_image = Compose([NormalizeData(),
                                        ResizeData(image_size),
                                        ToTensor()])
        self.transform_label = Compose([ToTensor()])
        
    def __len__(self):
        return self.nb_data
    
    def __getitem__(self, idx):
        image, flare_class = LoadData()(self.list_data[idx])
        label = MakeLabel()(flare_class)

        image = self.transform_image(image)
        label = self.transform_label(label)

        return image, label


def define_dataset(opt):
    dataset = CustomDataset(opt.data_root, opt.image_size, opt.is_train)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                            shuffle=opt.is_train, num_workers=opt.num_workers)
    return dataset, dataloader
