import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import logging
from tqdm import tqdm
from torchvision.transforms import v2 as t
from torchvision.datasets import VisionDataset

from architecture import build_METER_model
from augmentation import CShift, DShift, augmentation2D
from data import KittyDataset
from loss import balanced_loss_function
from metrics import REL, RMSE, delta



def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model



def evaluate(model: nn.Module, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_samples = 0

    metrics = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)

            metrics.append({
                "RMSE": RMSE(outputs, targets).item(),
                "REL": REL(outputs, targets).item(),
                "delta": delta(outputs, targets).item(),
                "loss_depth": loss[0].item(),
                "loss_grad": loss[1].item(),
                "loss_normal": loss[2].item(),
                "loss_ssim": loss[3].item(),
            })
            
            batch_size = images.size(0)
            total_samples += batch_size

    return metrics

    

def main():
    config_path = "config.yaml"
    if len(sys.argv) < 3:
        print("USAGE: ", sys.argv[0], " pconfig.yaml] [checkpoint path]")
    
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = load_config(config_path)
    
    if config['device']['cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['device_id']}")
        logging.info(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")

    dataset = KittyDataset(
        root_raw=config['data']['root_raw'],
        root_annotated=config['data']['root_annotated'],
        split_files_folder=config['data']['split_files_folder'],
        train=True,
        transforms=t.Compose([
            t.ToImage(),
            t.ToDtype(torch.float32, scale=True),  # scale to [0, 1]
            t.RandomHorizontalFlip(p=0.5),
            t.RandomVerticalFlip(p=0.5),
            CShift(),
            DShift()
            ])
    )
    dataloader = DataLoader(dataset, 128)

    model = build_METER_model(device, arch_type=config['model']['variant'])
    model = model.to(device)
    
    load_checkpoint(model, checkpoint_path)
    metrics = evaluate(model, dataloader, criterion=balanced_loss_function(device), device=device)
    df = pd.DataFrame(metrics).mean()
    df.to_csv("./metrics")
    print(df)

if __name__ == "__main__":
    main()