from torch.utils.data import Dataset

class FashionDataset(Dataset):
    def __init__(self, X, y, mean=0.2860, std=0.3530):
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = (self.X[i] - self.mean) / self.std
        y = self.y[i]
        return x, y