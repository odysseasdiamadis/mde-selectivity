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


def evaluate(model: nn.Module, dataloader, criterion, device):
    model.eval()
    total_samples = 0

    metrics = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")

        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            _ = model(images) # warm-up step because of image lazy loading
            
            torch.cuda.synchronize()

            tic = time.time()
            outputs = model(images)
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
                    "inference_time (ms)": toc*1000,
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
    model = model.to(device, dtype=dtype)

    load_checkpoint(model, checkpoint_path)
    metrics = evaluate(
        model,
        dataloader,
        criterion=balanced_loss_function(device, dtype=dtype).to(device, dtype),
        device=device,
    )
    df = pd.DataFrame(metrics).mean()
    df.to_csv("./metrics.csv", float_format='%.6f', index=False)
    print(df.to_string(float_format='{:,.6f}'.format))


if __name__ == "__main__":
    main()
