import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from math import exp

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
    def __init__(self, channel_counts: list[int], lambda_: float, device: torch.device):
        self.channel_counts = channel_counts
        self.lambda_ = lambda_
        self.device = device

    def forward(self,
                 R: torch.Tensor,
                device="cuda") -> torch.Tensor:
        """
        R: Tensor of shape [L, K_max, D] containing average responses R_{l,k,d}
        channel_counts: list of length L, where channel_counts[l] = K_l (#units in layer l)
        λ: weighting hyperparameter
        
        Returns: scalar loss L_assign
        """
        L, Kmax, D = R.shape
        total = torch.tensor(0.0, device=device)
        for l in range(L):
            K_l = self.channel_counts[l]
            # Slice out only the actual units in this layer:
            R_l = R[l, :K_l, :]             # shape [K_l, D]
            
            # Assigned bin for each unit k:
            # according to Eq. 6: d_k = floor( k * (K_l / D) )
            # (here k runs 0..K_l-1)
            ks = torch.arange(K_l, device=R.device)
            d_k = (ks * K_l // D).long()    # shape [K_l]
            
            # Responses at assigned bins:
            R_dk     = R_l[ks, d_k]         # shape [K_l]
            
            # Average of all *other* bins:
            sum_all  = R_l.sum(dim=1)       # shape [K_l]
            sum_oth  = sum_all - R_dk       # shape [K_l]
            R_minus  = sum_oth / (D - 1)    # shape [K_l]
            
            # Now selectivity for each unit:
            #   s_k = (|R_dk| - |R_minus|) / (|R_dk| + |R_minus| + ε)
            eps = 1e-6
            num =  (R_dk.abs() - R_minus.abs())
            den =  (R_dk.abs() + R_minus.abs()).clamp(min=eps)
            s_k = num / den                 # shape [K_l]
            
            total += s_k.sum()              # sum over units in layer l

        # Average over all layers & scale
        L_assign = - self.lambda_ * (total / sum(self.channel_counts))
        return L_assign

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ResponseCompute:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        n_of_bins: int = 10
    ):
        self.model      = model.to(device)
        self.loader     = dataloader
        self.device     = device
        self.D          = n_of_bins

        # 1) Gather all Conv2d layers in order, record out_channels
        self.conv_modules  = []
        self.channel_counts = []
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                self.conv_modules.append(m)
                self.channel_counts.append(m.out_channels)

        self.L = len(self.conv_modules)         # # of layers
        self.K = max(self.channel_counts)       # max # of channels

        # 2) Allocate accumulators
        #    total_response[l, k, d]: sum of activations for layer l, unit k, bin d
        #    mask_count[d]: total # of pixels falling in bin d (shared)
        self.total_response = torch.zeros(self.L, self.K, self.D, device=device)
        self.mask_count = torch.zeros(self.D, device=device)

    def compute_response(self, model, dataloader):
        hooks = []
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output: (B, C, h_out, w_out)
                fmap = output
                B, C, _, _ = fmap.shape

                # Upsample to current depth resolution
                fmap_up = F.interpolate(
                    fmap,
                    size=self.current_depth.shape[-2:],  # (H, W)
                    mode='bilinear',
                    align_corners=False
                )  # → (B, C, H, W)

                # Depth bin edges for this batch
                dmap = self.current_depth            # (B, H, W)
                min_d, max_d = dmap.min(), dmap.max()
                edges = torch.linspace(min_d, max_d, steps=self.D+1, device=self.device)

                for d in range(self.D):
                    # Build mask for bin d
                    mask = (dmap > edges[d]) & (dmap <= edges[d+1])  # (B, H, W)
                    if not mask.any():
                        continue

                    # Mask & sum activations
                    masked = fmap_up * mask.unsqueeze(1)   # broadcast → (B, C, H, W)
                    # Sum over batch & spatial dims → (C,)
                    bin_sum = masked.sum(dim=(0,2,3))
                    # Count how many pixels fell in bin d across the batch
                    pix_cnt = mask.sum()                   # scalar

                    # Accumulate
                    self.total_response[layer_idx, :C, d] += bin_sum
                    self.mask_count[d] += pix_cnt
            return hook_fn

        # 3) Register hooks on each Conv2d
        for idx, module in enumerate(self.conv_modules):
            hooks.append(module.register_forward_hook(make_hook(idx)))

        # 4) Loop over DataLoader
        model.eval()
        with torch.no_grad():
            for imgs, depths in tqdm.tqdm(dataloader):
                # imgs: (B, 3, H, W); depths: (B, 1, H, W) or (B, H, W)
                imgs   = imgs.to(self.device)
                depths = depths.to(self.device)
                # Normalize depth shape to (B, H, W)
                if depths.dim() == 4:
                    depths = depths.squeeze(1)
                self.current_depth = depths

                _ = model(imgs)

        # 5) Cleanup hooks
        for h in hooks:
            h.remove()
        

        # 6) Finalize R by dividing sums by pixel counts
        denom = self.mask_count.clamp(min=1e-6).view(1, 1, self.D)
        R = self.total_response / denom  # → (L, K, D)

        # 7) Set model in train mode again
        model.train()
        
        return R
