import os
from typing import Callable, Optional
import h5py
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2 as t

from augmentation import augmentation2D

@DeprecationWarning
class NYUDataset(VisionDataset):
    def __init__(self, mat_file):
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


class KittiDataset(VisionDataset):
    def __init__(
            self,
            root_raw: str,
            root_annotated: str,
            split_files_folder: str,
            train: bool = True,
            transforms: Optional[Callable] = None,
            ):
        super().__init__(root_raw, transforms=transforms)
        
        self.root_raw = root_raw
        self.split_file_train = os.path.join(split_files_folder, "kitti_eigen_train_files_with_gt.txt")
        self.split_file_test = os.path.join(split_files_folder, "kitti_eigen_test_files_with_gt.txt")
        self.train = train
        self.paths: list[tuple[str, str]] = []
        self.img_resize = t.Resize((192, 636))
        self.depth_resize = t.Resize(size=(48, 160))

        if train:
            self.split_file = self.split_file_train
            self.root_annotated = os.path.join(root_annotated, "train")
        else:
            self.split_file = self.split_file_test
            self.root_annotated = os.path.join(root_annotated, "val")

        self.root_annotated = os.path.join(root_annotated)
        with open(self.split_file) as f:
            for line in f:
                raw_file, target_file, _ = tuple(line.split())
                if not target_file == 'None':
                    raw_path = os.path.join(self.root_raw, raw_file)
                    target_path = os.path.join(self.root_annotated, target_file)
                    assert os.path.isfile(raw_path)
                    assert os.path.isfile(target_path)
                    self.paths.append((raw_path, target_path))
        
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        raw, depth = self.paths[idx]
        raw_img = Image.open(raw)
        depth_img = Image.open(depth)

        # raw_img, depth_img = augmentation2D(img_np, depth=depth_np)
        # depth_img = self.img_resize(depth_img)
        raw_img = self.img_resize(raw_img)
        depth_img = self.depth_resize(depth_img)
        if self.transforms:
            raw_img, depth_img = self.transforms(raw_img, depth_img)
        
        
        return raw_img, depth_img
    
