import random
import numpy as np
from itertools import product
import torchvision.transforms.functional as F

import torch
from globals import augmentation_parameters


def pixel_shift(depth_img, shift):
    depth_img = depth_img + shift
    return depth_img


def random_crop(x, y, crop_size=(192, 256)):
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    h, w, _ = x.shape
    rangew = (w - crop_size[0]) // 2 if w > crop_size[0] else 0
    rangeh = (h - crop_size[1]) // 2 if h > crop_size[1] else 0
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    cropped_x = x[offseth:offseth + crop_size[0], offsetw:offsetw + crop_size[1], :]
    cropped_y = y[offseth:offseth + crop_size[0], offsetw:offsetw + crop_size[1], :]
    cropped_y = cropped_y[:, :, ~np.all(cropped_y == 0, axis=(0, 1))]
    if cropped_y.shape[-1] == 0:
        return x, y
    else:
        return cropped_x, cropped_y


def augmentation2D(img, depth, print_info_aug=False):
    # Random flipping
    if random.uniform(0, 1) <= augmentation_parameters['flip']:
        img = (img[::1, :, :]).copy()
        depth = (depth[::1, :, :]).copy()
        if print_info_aug:
            print('--> Random flipped')
    # Random mirroring
    if random.uniform(0, 1) <= augmentation_parameters['mirror']:
        img = (img[::-1, :]).copy()
        depth = (depth[::-1, :]).copy()
        if print_info_aug:
            print('--> Random mirrored')   
    # Channel swap
    if random.uniform(0, 1) <= augmentation_parameters['c_swap']:
        indices = list(product([0, 1, 2], repeat=3))
        policy_idx = random.randint(0, len(indices) - 1)
        img = img[list(indices[policy_idx])]
        if print_info_aug:
            print('--> Channel swapped')
    # Random crop
    if random.random() <= augmentation_parameters['random_crop']:
        img, depth = random_crop(img, depth)
        if print_info_aug:
            print('--> Random cropped')
    # Shifting Strategy
    if random.uniform(0, 1) <= augmentation_parameters['shifting_strategy']:
        # gamma augmentation
        gamma: float = random.uniform(0.9, 1.1)
        img = img ** gamma
        brightness = random.uniform(0.9, 1.1)
        img = img * brightness
        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((img.shape[0], img.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        img *= color_image
        img = np.clip(img, 0, 255)  # Originally with 0 and 1
        # depth aumgnetation
        random_shift = random.randint(-10, 10)
        depth = pixel_shift(depth, shift=random_shift)
        if print_info_aug:
            print('--> Depth Shifted of {} cm/dm and Image randomly augmented'.format(random_shift))

    return img, depth


def custom_tensor_augmentation(img, depth):
    # img and depth: torch tensors of shape [C, H, W] and [1, H, W]
    
    # Random horizontal flip
    if torch.rand(1) < 0.5:
        img = torch.flip(img, dims=[2])     # Flip width (W)
        depth = torch.flip(depth, dims=[2])

    # Random crop
    crop_h, crop_w = 192, 256
    _, H, W = img.shape
    top = torch.randint(0, H - crop_h + 1, (1,)).item()
    left = torch.randint(0, W - crop_w + 1, (1,)).item()
    img = img[:, top:top+crop_h, left:left+crop_w]
    depth = depth[:, top:top+crop_h, left:left+crop_w]

    # Brightness and gamma
    gamma = torch.empty(1).uniform_(0.9, 1.1).item()
    brightness = torch.empty(1).uniform_(0.9, 1.1).item()
    img = img ** gamma
    img = img * brightness
    img = torch.clamp(img, 0, 1)

    # Random channel permutation
    if torch.rand(1) < 0.5:
        perm = torch.randperm(3)
        img = img[perm]

    # Random depth shift
    if torch.rand(1) < 0.5:
        shift = torch.randint(-10, 10, (1,)).item()
        depth = depth.to(torch.int32) + shift
        depth = torch.clamp(depth, min=0).to(torch.float32)

    return img, depth

class CShift(object):
    """Color shift (unchanged)."""
    def __init__(self, brightness_range=(0.9,1.1), 
                       gamma_range=(0.9,1.1),
                       color_scale_range=(0.9,1.1)):
        self.brightness_range = brightness_range
        self.gamma_range = gamma_range
        self.color_scale_range = color_scale_range

    def __call__(self, img, depth):
        # 1) random brightness
        b = random.uniform(*self.brightness_range)
        img = F.adjust_brightness(img, b)
        # 2) random gamma
        g = random.uniform(*self.gamma_range)
        img = F.adjust_gamma(img, gamma=g)
        # 3) global color scale
        η = random.uniform(*self.color_scale_range)
        img =  img * η
        img = torch.clamp(img, 0, 1)
        return img, depth

class DShift(object):
    """Depth shift for normalized depth map in [0,1] (1 == 80 m)."""
    def __init__(self, shift_meters_range=(-1.0, 1.0), max_depth_m=80.0):
        """
        shift_meters_range: tuple (min, max) in meters
        max_depth_m: the value of normalized depth 1.0 in meters
        """
        self.shift_meters_range = shift_meters_range
        self.max_depth_m = max_depth_m

    def __call__(self, img, depth):
        # depth is a torch.Tensor with values in [0,1]:
        # draw a shift s in meters, convert to normalized units:
        s_m = random.uniform(*self.shift_meters_range)
        s_norm = s_m / self.max_depth_m
        depth = depth + s_norm
        depth = torch.clamp(depth, 0.0, 1.0)
        return img, depth