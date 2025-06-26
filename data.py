import os
from typing import Callable, Optional
import h5py
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2 as t

from augmentation import augmentation2D

class NYUDataset(VisionDataset):
    def __init__(self, root, transforms: Optional[Callable] = None):
        super().__init__(None, transforms)
        with h5py.File(mat_file, 'r') as f:
            images = f['images']
            assert isinstance(images, h5py.Dataset)
            self.images = images
            depths = f['depths']
            assert isinstance(depths, h5py.Dataset)
            self.depths = depths
            assert type(self.images) == h5py.Dataset
            assert type(self.depths) == h5py.Dataset

            assert self.images.shape[0] == self.depths.shape[0]

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.images[idx]),
            torch.tensor(self.depths[idx])
        )

    
