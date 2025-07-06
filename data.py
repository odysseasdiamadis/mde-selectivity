import os
from typing import Callable, Optional
from PIL import Image
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2 as t
from torchvision.transforms.functional import to_tensor
from augmentation import augmentation2D, custom_tensor_augmentation


class Depth2Tensor(t.Transform):
    def __init__(self, scale):
        super().__init__()

    def forward(self, *img):
        rgb, depth = img
        return rgb, depth * self.scale



class NYUDataset(VisionDataset):
    def __init__(self,
                 root,
                 transforms: Optional[Callable] = None,
                 test=False,
                 dtype=torch.float32
                ):
        super().__init__(root, transforms)
        self.dtype = dtype
        self.rgb_transform = t.Compose([
            t.Resize((192,256)),
            t.ToImage(),
            t.ToDtype(dtype, scale=True)
        ])
        self.depth_transform = t.Compose([
            t.Resize((48,64)),
            t.ToImage(),
            t.ToDtype(dtype, scale=(not self.test))
        ])
        self.test = test
        if test:
            csv_path = os.path.join(root, 'nyu2_test.csv')
        else:
            csv_path = os.path.join(root, 'nyu2_train.csv')

        self.samples = []

        with open(csv_path) as f:
            for line in f:
                rgb, depth = tuple(line.strip().split(','))
                rgb_fixed = os.path.sep.join(rgb.split(os.path.sep)[1:])
                depth_fixed = os.path.sep.join(depth.split(os.path.sep)[1:])

                self.samples.append((os.path.join(root, rgb_fixed), os.path.join(root, depth_fixed)))
        
        for sample in self.samples:
            assert os.path.isfile(sample[0]), f"{sample[0]} does not exist"
            assert os.path.isfile(sample[1]), f"{sample[1]} does not exist"


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]
        rgb = Image.open(rgb_path)
        depth = Image.open(depth_path)

        rgb = self.rgb_transform(rgb)
        depth = self.depth_transform(depth)

        if not self.test:
            rgb, depth = custom_tensor_augmentation(rgb, depth, dtype=self.dtype)
            depth = depth*1000 # cm
        
        return rgb, depth
