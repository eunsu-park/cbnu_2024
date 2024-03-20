from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self):
        super(CustomDataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass