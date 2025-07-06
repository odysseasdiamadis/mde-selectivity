import logging
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2 as t
from tqdm import tqdm

from architecture import build_METER_model
from data import NYUDataset, RescaleDepth
from loss import balanced_loss_function
from metrics import REL, RMSE, delta
from torch.profiler import profile, record_function, ProfilerActivity


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def compute_dataset_response(
    model: torch.nn.Module,
    conv_modules: list[torch.nn.Conv2d],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_of_bins: int,
) -> tuple[torch.Tensor, list]:
    """
    Compute R over the *entire* dataset for evaluation.

    Args:
      model         : your full depth model
      conv_modules  : list of Conv2d layers (in the same order you use for L_assign)
      dataset       : a torch Dataset returning (img, depth) pairs
      device        : cuda or cpu
      n_of_bins     : number of depth bins D

    Returns:
      R : Tensor of shape [L, K_max, D] with the dataset‐wide average responses
    """
    D = n_of_bins
    L = len(conv_modules)
    K_list = [m.out_channels for m in conv_modules]
    Kmax = max(K_list)

    # accumulators
    sum_response = torch.zeros(L, Kmax, D, device=device)
    count_bins = torch.zeros(D, device=device)

    model.eval()
    with torch.no_grad():
        for imgs, depths in loader:
            # move & shape
            imgs = imgs.to(device)
            depths = depths.to(device).squeeze(1)  # (B, H, W)
            B, H, W = depths.shape

            # compute bin edges for this batch
            dmin, dmax = depths.min(), depths.max()
            edges = torch.linspace(dmin, dmax, steps=D + 1, device=device)

            # flatten depths & compute bin idx
            flat_d = depths.view(-1)
            bidx = torch.bucketize(flat_d, edges, right=True) - 1
            bidx = bidx.clamp(0, D - 1)  # (B*H*W,)

            # accumulate pixel counts
            count_bins += torch.bincount(bidx, minlength=D).to(device)

            out, fmap_list = model(imgs)

            # now for each layer, scatter‐sum
            for idx, fmap in fmap_list:
                # fmap: (B, C, H, W) → flatten pixels
                C = fmap.shape[1]
                flat_f = fmap.permute(1, 0, 2, 3).reshape(C, -1)  # (C, B*H*W)

                # scatter into bins
                summed = torch.zeros(C, D, device=device)
                # bidx expands for each channel
                summed.scatter_reduce_(
                    dim=1,
                    index=bidx.unsqueeze(0).expand(C, -1),
                    src=flat_f,
                    reduce="sum",
                    include_self=False,
                )

                sum_response[idx, :C, :] += summed

    # normalize by count_bins → R
    denom = count_bins.clamp(min=1e-6).view(1, 1, D)
    R = sum_response / denom

    return R, fmap_list


def compute_selectivity(
    R: torch.Tensor, channel_counts: list[int], eps: float = 1e-6
) -> torch.Tensor:
    L, Kmax, D = R.shape
    device = R.device

    # 1) Raw-argmax along bins (d*) per layer/unit
    #    shape [L, Kmax]
    d_star = torch.argmax(R.abs(), dim=2)

    # 2) Gather R_{l,k,d*}
    l_idx = torch.arange(L, device=device).unsqueeze(1).expand(L, Kmax)
    k_idx = torch.arange(Kmax, device=device).unsqueeze(0).expand(L, Kmax)
    R_max = R.abs()[l_idx, k_idx, d_star]  # |R_{l,k,d*}| shape [L,Kmax]

    # 3) Sum of abs over all bins: sum_d |R_{l,k,d}|
    R_sum_abs = R.abs().sum(dim=2)  # shape [L,Kmax]

    # 4) Compute average of the “other” bins:
    #    (sum_abs - R_max) / (D-1)
    R_rest_avg = (R_sum_abs - R_max) / (D - 1 + eps)

    # 5) Selectivity index s = (R_max - R_rest_avg) / (R_max + R_rest_avg + eps)
    S = (R_max - R_rest_avg) / (R_max + R_rest_avg + eps)

    # 6) Zero out invalid units (k >= K_l)
    #    Build mask [L,Kmax] where True for k < K_l
    kc = torch.tensor(channel_counts, device=device)
    valid = k_idx < kc.unsqueeze(1)  # shape [L,Kmax]
    S = S * valid.float()

    return S  # shape [L, Kmax]


def evaluate(model: nn.Module, dataloader, criterion, device):
    model.eval()
    total_samples = 0

    metrics = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")

        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)

            torch.cuda.synchronize()

            tic = time.time()
            outputs, fmaps = model(images)
            torch.cuda.synchronize()
            toc = time.time() - tic
            loss = criterion(outputs, targets)

            metrics.append(
                {
                    "RMSE": RMSE(outputs, targets).item(),
                    "REL": REL(outputs, targets).item(),
                    "delta": delta(outputs, targets).item(),
                    "loss_depth": loss[0].item(),
                    "loss_grad": loss[1].item(),
                    "loss_normal": loss[2].item(),
                    "loss_ssim": loss[3].item(),
                    "inference_time (ms)": toc * 1000,
                }
            )

            batch_size = images.size(0)
            total_samples += batch_size

    return metrics


def main():
    dtype = torch.float32
    config_path = "config.yaml"
    if len(sys.argv) < 3:
        print("USAGE: ", sys.argv[0], " pconfig.yaml] [checkpoint path]")

    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(config_path)

    if config["device"]["cuda"] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['device_id']}")
        logging.info(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # rescale_0_80 = transforms.Lambda(lambda (raw, depth): (raw, depth * 80))
    dataset = NYUDataset(
        root=config["data"]["root"],
        test=True,
    )

    dataloader = DataLoader(dataset, 1)

    model = build_METER_model(device, arch_type=config["model"]["variant"])
    model = nn.DataParallel(model)
    model = model.to(device, dtype=dtype)

    load_checkpoint(model, checkpoint_path)
    metrics = evaluate(
        model,
        dataloader,
        criterion=balanced_loss_function(device, dtype=dtype).to(device, dtype),
        device=device,
    )
    conv_layers = [
        m
        for name, m in model.named_modules()
        if m.__class__.__name__ in config["training"]["layers"]
    ]
    counts = [m.out_channels for m in conv_layers]


    R, fmap = compute_dataset_response(
        model, conv_layers, dataloader, device, config["training"]["n_of_bins"]
    )

    S = compute_selectivity(R, counts, eps=1e-6)

    df = pd.DataFrame(metrics).mean()
    for i in range(S.shape[0]):
        for j in range(S.shape[0]):
            df[f'S_{i}_{j}'] = S[i,j].item()
    df.to_csv("./metrics.csv", float_format="%.6f")
    print(df.to_string(float_format="{:,.6f}".format))


if __name__ == "__main__":
    main()
