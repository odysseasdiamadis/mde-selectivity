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
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
        L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(device=img1.device, dtype=self.dtype)

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
            ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().to(self.device)

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
        loss_normal = self.lambda_2 * torch.abs(1 - self.cos(output_normal, depth_normal)).mean()

        loss_ssim = (1 - self.ssim(output, depth, val_range=1000.0)) * self.lambda_3 # type: ignore

        loss_grad = (loss_dx + loss_dy) / self.lambda_1

        return loss_depth, loss_grad, loss_normal, loss_ssim


class L_assign(nn.Module):
    def __init__(self,channel_counts, response_compute: "ResponseCompute", lambda_: float, device: torch.device):
        super().__init__()
        self.channel_counts = channel_counts
        self.lambda_ = lambda_
        self.device = device
        self.response_compute = response_compute

    def forward(self,
                #  R: torch.Tensor,
                model,
                batch,
                device="cuda") -> torch.Tensor:
        """
        R: Tensor of shape [L, K_max, D] containing average responses R_{l,k,d}
        channel_counts: list of length L, where channel_counts[l] = K_l (#units in layer l)
        λ: weighting hyperparameter
        
        Returns: scalar loss L_assign
        """
        R = self.response_compute(model, batch)
        L, Kmax, D = R.shape
        
        # Precompute tensors for vectorization
        ks = torch.arange(Kmax, device=R.device).unsqueeze(0)  # [1, Kmax]
        channel_counts_t = torch.tensor(self.channel_counts, device=R.device)  # [L]
        
        # 1. Compute bin assignments (Eq.6) for all layers/units
        n_b = torch.minimum(channel_counts_t, torch.tensor(D, device=R.device))  # [L]
        d_k = (ks[:, :Kmax] * n_b.unsqueeze(1)) // channel_counts_t.unsqueeze(1)  # [L, Kmax]
        d_k = torch.clamp(d_k.long(), 0, D-1)  # Ensure valid bin indices
        
        # 2. Gather responses at assigned bins
        l_indices = torch.arange(L, device=R.device).unsqueeze(1)  # [L, 1]
        k_indices = ks.expand(L, Kmax)  # [L, Kmax]
        R_dk = R[l_indices, k_indices, d_k]  # [L, Kmax]
        
        # 3. Compute average of other bins
        R_sum = R.sum(dim=2)  # [L, Kmax]
        R_minus = (R_sum - R_dk) / (D - 1)  # [L, Kmax]
        
        # 4. Compute selectivity scores (Eq.5)
        abs_R_dk = R_dk.abs()
        abs_R_minus = R_minus.abs()
        s_k = (abs_R_dk - abs_R_minus) / (abs_R_dk + abs_R_minus + 1e-6)  # [L, Kmax]
        
        # 5. Create mask for valid units
        valid_mask = ks < channel_counts_t.unsqueeze(1)  # [L, Kmax]
        
        # 6. Compute loss (Eq.5 scaled)
        total_s_k = (s_k * valid_mask).sum()  # Only sum valid units
        total_units = channel_counts_t.sum()
        L_assign = -self.lambda_ * (total_s_k / total_units)
        
        return L_assign


class ResponseCompute(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config,
        n_of_bins: int = 10
    ):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.D = n_of_bins
        self.layers = config['training']['layers']

        # 1) Gather all Conv2d layers in order, record out_channels
        self.conv_modules  = []
        self.channel_counts = []
        all_channels = []
        for name, m in self.model.named_modules():
            if m.__class__.__name__ in self.layers:
                self.conv_modules.append(m)
                self.channel_counts.append(m.out_channels)

        self.L = len(self.conv_modules)         # # of layers
        self.K = max(self.channel_counts)       # max # of channels

        # 2) Allocate accumulators
        #    total_response[l, k, d]: sum of activations for layer l, unit k, bin d
        #    mask_count[d]: total # of pixels falling in bin d (shared)
        self.mask_count = torch.zeros(self.D, device=device)

    def forward(self, model: nn.Module, batch: tuple[torch.Tensor,torch.Tensor]):
        imgs, depths = batch
        imgs = imgs.to(self.device)
        depths = depths.to(self.device).squeeze(1)   # → (B, H, W)
        
        # 1) register hooks that just stash the up‐sampled fmap per layer
        fmap_list: list[torch.Tensor] = []
        hooks = []
        def make_hook():
            def hook(m, _in, out):
                # out: (B, C, h', w')
                fmap = F.interpolate(out, size=depths.shape[-2:], 
                                    mode='bilinear', align_corners=False)
                fmap_list.append(fmap)  # list of (B, C, H, W)
            return hook

        for conv in self.conv_modules:
            hooks.append(conv.register_forward_hook(make_hook()))
        
        _ = model(imgs)
        
        for h in hooks:
            h.remove()

        # 2) compute global min/max and bin edges
        d_min, d_max = depths.min(), depths.max()
        edges = torch.linspace(d_min, d_max, steps=self.D + 1,
                            device=self.device)  # (D+1,)

        # 3) bucketize every pixel once:  idx in [0 .. D-1]
        #    depths: (B,H,W) → flat idx: (B*H*W,)
        flat_depths = depths.reshape(-1)
        bin_idx = torch.bucketize(flat_depths, edges, right=True) - 1
        # clamp to [0,D-1]
        bin_idx = bin_idx.clamp(0, self.D-1)  # → (B*H*W,)

        # 4) for each layer, do one scatter‐reduce over the pixel dimension
        R = torch.zeros(len(fmap_list), self.K, self.D, device=self.device)
        counts = torch.zeros(self.D, device=self.device)
        
        # We'll also build a global count vector in one go:
        counts = torch.bincount(bin_idx, minlength=self.D)  # → (D,)

        for layer_idx, fmap in enumerate(fmap_list):
            # fmap: (B, C, H, W) → (C, B*H*W)
            C = fmap.shape[1]
            flat_f = fmap.reshape(fmap.shape[0], C, -1).permute(1, 0, 2)
            flat_f = flat_f.reshape(C, -1)  # (C, B*H*W)

            # scatter‐sum across the pixels into depth‐bins:
            #   out: (C, D)
            summed = torch.zeros(C, self.D, device=self.device)
            # dim=1 because we scatter along the pixel axis
            summed.scatter_reduce_(1, 
                                bin_idx.unsqueeze(0).expand(C, -1), 
                                flat_f, 
                                reduce="sum", 
                                include_self=False)

            # store (and optionally trim/pad if C < K)
            R[layer_idx, :C, :] = summed

        # 5) divide out the counts (avoid div by zero)
        denom = counts.clamp(min=1e-6).view(1, 1, self.D)
        R = R / denom

        return R
