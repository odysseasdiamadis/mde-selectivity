import logging
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from architecture import build_METER_model
from data import NYUDataset
from loss import balanced_loss_function
from metrics import REL, RMSE, delta


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


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_dataset_response_batch_resistant(
    model, counts, loader, device, n_of_bins, depth_min=0.0, depth_max=1000.0
):
    D = n_of_bins
    L = len(counts)
    K_l = counts
    Kmax = max(K_l)

    # fixed, global edges
    edges = torch.linspace(depth_min, depth_max, D + 1, device=device)

    sum_response = torch.zeros(L, Kmax, D, device=device)
    count_bins = torch.zeros(D, device=device)

    model.eval()
    with torch.no_grad():
        for imgs, depths in loader:
            imgs = imgs.to(device)
            depths = depths.to(device).squeeze(1)  # (B,H,W)
            B, H, W = depths.shape

            # bucketize with **fixed** edges
            bidx = torch.bucketize(depths.reshape(-1), edges, right=True) - 1
            bidx = bidx.clamp(0, D - 1)
            count_bins += torch.bincount(bidx, minlength=D).to(device)

            # forward + get your fmap_list (in same order as conv_modules)
            _, fmap_list = model(imgs)

            for i, fmap in enumerate(fmap_list):
                C = fmap.shape[1]
                # upsample to (H,W)
                fmap_up = F.interpolate(
                    fmap, size=(H, W), mode="bilinear", align_corners=False
                )
                flat_f = fmap_up.permute(1, 0, 2, 3).reshape(C, -1)
                summed = torch.zeros(C, D, device=device)
                summed.scatter_reduce_(
                    1,
                    bidx.unsqueeze(0).expand(C, -1),
                    flat_f,
                    reduce="sum",
                    include_self=False,
                )
                sum_response[i, :C, :] += summed

    # normalize
    R = sum_response / count_bins.clamp(min=1e-6).view(1, 1, D)
    return R


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

    R_sum_abs = R.abs().sum(dim=2)  # shape [L,Kmax]
    R_rest_avg = (R_sum_abs - R_max) / (D - 1 + eps)

    S = (R_max - R_rest_avg) / (R_max + R_rest_avg + eps)

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

    return metrics, fmaps  # type: ignore


def build_eval_df(exp_name, dataloader, model, device, dtype, include_config=True):
    config = load_config(f"configs/{exp_name}.yaml")
    checkpoint_path = os.path.join(
        "experiments", exp_name, "checkpoints", "best_model.pth"
    )
    load_checkpoint(model, checkpoint_path, device=device)
    metrics, fmaps = evaluate(
        model,
        dataloader,
        criterion=balanced_loss_function(device, dtype=dtype).to(device, dtype),
        device=device,
    )
    df = pd.DataFrame(metrics).mean().to_dict()
    if include_config:
        train_conf = config["training"]
        df["lambda"] = train_conf.get("lambda", "")
        df["n_of_bins"] = train_conf.get("n_of_bins", "")
        df["fmaps_decoder"] = train_conf.get("fmaps_decoder", "")
    df["exp_name"] = exp_name
    return df


def main():
    dtype = torch.float32
    config_path = "config.yaml"
    if len(sys.argv) < 3:
        print("USAGE: ", sys.argv[0], " --exp-list exp1,exp2...")

    exp_list = sys.argv[2].split(",")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(config_path)

    if config["device"]["cuda"] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['device_id']}")
        logging.info(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    test_ds = NYUDataset(
        root=config["data"]["root"],
        test=True,
    )

    test_dl = DataLoader(test_ds, 128)

    model = build_METER_model(
        device,
        arch_type=config["model"]["variant"],
        fmap_decoder=config["training"].get("fmaps_decoder"),
    )
    model = nn.DataParallel(model)
    model = model.to(device, dtype=dtype)

    evals = {exp: build_eval_df(exp, test_dl, model, device, dtype) for exp in exp_list}

    df = pd.DataFrame.from_dict(evals)

    df.to_csv(f"./metrics_experiments.csv", float_format='%.2f')


if __name__ == "__main__":
    main()
