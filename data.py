import os
from typing import Callable, Optional
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2 as t
from torchvision.transforms.functional import to_tensor
import torchvision.transforms.functional
from augmentation import augmentation2D, custom_tensor_augmentation


class Depth2Tensor(t.Transform):
    def __init__(self):
        super().__init__()

    def forward(self, *depth):
        depth = torchvision.transforms.functional.to_tensor(depth[0])
        if depth.max() > 1:  # test set, 10m in mm (10_000)
            depth = depth / 1000
        else:  # train set, regular [0,255] that gets normalized to [0,1]
            depth = depth * 10
        return depth


class NYUDataset(VisionDataset):
    def __init__(
        self,
        root,
        transforms: Optional[Callable] = None,
        test=False,
        dtype=torch.float32,
    ):
        super().__init__(root, transforms)
        self.test = test
        self.dtype = dtype
        self.rgb_transform = t.Compose(
            [t.Resize((192, 256)), t.ToImage(), t.ToDtype(dtype)]
        )
        self.depth_transform = t.Compose([t.Resize((48, 64)), Depth2Tensor()])
        if test:
            csv_path = os.path.join(root, "nyu2_test.csv")
        else:
            csv_path = os.path.join(root, "nyu2_train.csv")

        self.samples = []

        with open(csv_path) as f:
            for line in f:
                rgb, depth = tuple(line.strip().split(","))
                rgb_fixed = os.path.sep.join(rgb.split(os.path.sep)[1:])
                depth_fixed = os.path.sep.join(depth.split(os.path.sep)[1:])

                self.samples.append(
                    (os.path.join(root, rgb_fixed), os.path.join(root, depth_fixed))
                )

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

        return rgb, depth
