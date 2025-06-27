import os
from typing import Callable, Optional
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2 as t

from augmentation import augmentation2D

class NYUDataset(VisionDataset):
    def __init__(self, root, transforms: Optional[Callable] = None, test=False):
        super().__init__(root, transforms)
        
        self.resize_rgb = t.Resize((192,256))
        self.resize_depth = t.Resize((48,64))
        if test:
            csv_path = os.path.join(root, 'nyu2_test.csv')
        else:
            csv_path = os.path.join(root, 'nyu2_train.csv')
        print(root)

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

        if self.transforms:
            rgb, depth = self.transforms(rgb, depth)
        
        rgb = self.resize_rgb(rgb)
        depth = self.resize_depth(depth)
        return rgb, depth
