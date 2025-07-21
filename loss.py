"""
MIT License

Copyright (c) 2022 lorenzopapa5

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import tqdm


class Sobel(nn.Module):
    def __init__(self, device="cuda"):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).to(device=device).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k).to(device=device)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out


class balanced_loss_function(nn.Module):

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [
                exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(
        self,
        img1,
        img2,
        val_range,
        window_size=11,
        window=None,
        size_average=True,
        full=False,
    ):
        L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(
                device=img1.device, dtype=self.dtype
            )

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs

        return ret

    def __init__(self, device, dtype):
        super(balanced_loss_function, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=0)
        self.get_gradient = Sobel().to(device)
        self.device = device
        self.dtype = dtype
        self.lambda_1 = 0.5
        self.lambda_2 = 100
        self.lambda_3 = 100

    def forward(self, output, depth):
        with torch.no_grad():
            ones = (
                torch.ones(depth.size(0), 1, depth.size(2), depth.size(3))
                .float()
                .to(self.device)
            )

        depth_grad = self.get_gradient(depth)
        output_grad = self.get_gradient(output)

        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        loss_depth = torch.abs(output - depth).mean()
        loss_dx = torch.abs(output_grad_dx - depth_grad_dx).mean()
        loss_dy = torch.abs(output_grad_dy - depth_grad_dy).mean()
        loss_normal = (
            self.lambda_2 * torch.abs(1 - self.cos(output_normal, depth_normal)).mean()
        )

        loss_ssim = (1 - self.ssim(output, depth, val_range=1000.0)) * self.lambda_3  # type: ignore

        loss_grad = (loss_dx + loss_dy) / self.lambda_1

        return loss_depth, loss_grad, loss_normal, loss_ssim


class L_assign(nn.Module):
    def __init__(
        self,
        response_compute: "ResponseCompute",
        lambda_: float,
        device: torch.device,
        N_b: int,  # Number of depth bins (from paper)
        assign_formula=None,
    ):
        super().__init__()
        self.lambda_ = lambda_
        self.device = device
        self.response_compute: ResponseCompute = response_compute
        self.N_b = N_b  # Store N_b as per paper
        self.assign_formula = assign_formula or "original"

    def forward(
        self,
        batch,
        fmaps,
    ) -> torch.Tensor:
        self.channel_counts = [fmap.shape[1] for fmap in fmaps]
        R = self.response_compute(batch, fmaps)
        L, Kmax, D = R.shape

        ks = torch.arange(Kmax, device=R.device).unsqueeze(0)  # [1, Kmax]
        channel_counts_t = torch.tensor(self.channel_counts, device=R.device)  # [L]

        # CORRECTED: Use N_b as per equation (6) in paper
        # d_k = floor(k / K_l * N_b)
        N_b_tensor = torch.tensor(self.N_b, device=R.device)
        d_k = (ks * N_b_tensor.unsqueeze(0)) // channel_counts_t.unsqueeze(1)  # [L, Kmax]
        
        # Ensure valid bin indices (should not exceed D-1)
        d_k = torch.clamp(d_k.long(), 0, D - 1)

        # Extract responses for assigned bins
        l_indices = torch.arange(L, device=R.device).unsqueeze(1)  # [L, 1]
        k_indices = ks.expand(L, Kmax)  # [L, Kmax]
        R_dk = R[l_indices, k_indices, d_k]  # [L, Kmax] - Response in assigned bin

        # Calculate average response in all OTHER bins (excluding assigned bin)
        # This requires more careful computation
        R_minus_dk = torch.zeros_like(R_dk)
        
        for l in range(L):
            for k in range(self.channel_counts[l]):
                assigned_bin = d_k[l, k].item()
                # Sum all responses except the assigned bin
                other_responses = torch.cat([R[l, k, :assigned_bin], R[l, k, assigned_bin+1:]])
                if len(other_responses) > 0:
                    R_minus_dk[l, k] = other_responses.mean()
                else:
                    R_minus_dk[l, k] = 0.0

        # Calculate selectivity scores as per equation (5)
        abs_R_dk = R_dk.abs()
        abs_R_minus_dk = R_minus_dk.abs()
        s_k = (abs_R_dk - abs_R_minus_dk) / (abs_R_dk + abs_R_minus_dk + 1e-6)

        # Only consider valid units (k < K_l for each layer)
        valid_mask = ks < channel_counts_t.unsqueeze(1)  # [L, Kmax]

        # Calculate final loss as per equation (5)
        total_loss = 0.0
        for l in range(L):
            K_l = self.channel_counts[l]
            layer_selectivity = (s_k[l, :K_l] * valid_mask[l, :K_l]).sum()
            total_loss += layer_selectivity / K_l

        L_assign = -self.lambda_ * (total_loss / L)  # Average over layers

        return L_assign


class ResponseCompute(nn.Module):
    def __init__(self, model: nn.Module, device: torch.device, n_of_bins: int):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.D = n_of_bins

    def forward(self, batch, fmap_list) -> torch.Tensor:
        imgs, depths = batch
        imgs = imgs.to(self.device)
        depths = depths.to(self.device).squeeze(1)  # → (B, H, W)
        self.channel_counts = [fmap.shape[1] for fmap in fmap_list]
        self.K = max(self.channel_counts)  # max # of channels

        # Create depth bin edges
        edges = torch.linspace(0, 1000, steps=self.D + 1, device=self.device)  # (D+1,)

        R = torch.zeros(len(fmap_list), self.K, self.D, device=self.device)

        for layer_idx, fmap in enumerate(fmap_list):
            # Interpolate feature map to depth resolution
            A = F.interpolate(
                fmap, size=depths.shape[-2:], mode="bilinear", align_corners=False
            )  # (B, C, H, W)

            C = A.shape[1]
            B, _, H, W = A.shape
            
            # CORRECTED: Ensure consistent flattening order
            flat_depths = depths.reshape(-1)  # (B*H*W,)
            flat_A = A.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            
            # Bin the depths
            bin_idx = torch.bucketize(flat_depths, edges, right=True) - 1
            bin_idx = torch.clamp(bin_idx, 0, self.D - 1)

            # Calculate average response for each channel in each depth bin
            for c in range(C):
                for d in range(self.D):
                    mask = (bin_idx == d)
                    if mask.sum() > 0:
                        R[layer_idx, c, d] = flat_A[mask, c].mean()
                    else:
                        R[layer_idx, c, d] = 0.0

        return R