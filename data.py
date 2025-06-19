import h5py
from torch.utils.data import Dataset
import torch

class NYUDataset(Dataset):
    def __init__(self, mat_file):
        with h5py.File(mat_file, 'r') as f:
            self.images = f['images']
            self.depths = f['depths']
            assert self.images.shape[0] == self.depths.shape[0]

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.images[idx]),
            torch.tensor(self.depths[idx])
        )

ds = NYUDataset("nyu_depth_v2_labeled.mat")

print(len(ds.images))
