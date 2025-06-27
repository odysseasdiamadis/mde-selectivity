import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import yaml
import os
import logging
from tqdm import tqdm
from torchvision.transforms import v2 as t
import csv

from architecture import build_METER_model
from data import KittiDataset
from loss import balanced_loss_function
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler


def on_trace_ready_handler(trace):
    print("Trace file generated:", trace)
    return tensorboard_trace_handler('./tb_logs')(trace)


def profile_epoch(model, dataloader, criterion, optimizer, device, config):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc="Training")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True) as prof:

        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with record_function("model_forward"):
                outputs = model(images)

            with record_function("loss"):
                loss_tuple = criterion(outputs, targets)
                loss = loss_tuple[0]

            with record_function("backward"):
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # torch.cuda.synchronize()  # 🔁 Force all CUDA ops to complete each batch
            prof.step()

    # Print profiler results *after* loop
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    

    return total_loss / num_batches



def train(config) -> None:
    """Main training function."""
    # Setup device
    if config['device']['cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['device_id']}")
        logging.info(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    
    # Get model
    model = build_METER_model(device, arch_type=config['model']['variant'])
    model = model.to(device)
    logging.info(f"Model: {config['model']['variant']} variant")
    
    # Get optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Get loss function
    criterion = balanced_loss_function(device)
        
    # Placeholder for actual dataloaders
    dataset = KittiDataset(
        root_raw=config['data']['root_raw'],
        root_annotated=config['data']['root_annotated'],
        split_files_folder=config['data']['split_files_folder'],
        train=True,
        transforms=t.Compose([
            t.Resize((192, 636)),
            t.RandomRotation(degrees=10), # type: ignore
            t.ToImage(),
            t.ToDtype(torch.float32, scale=True)
        ])
    )

    train_ds = Subset(dataset, range(500))
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], num_workers=config['data']['num_workers'])
    train_loss = profile_epoch(model, train_loader, criterion, optimizer, device, config)
        

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    config_path = "config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = load_config(config_path)
    train(config)


if __name__ == "__main__":
    main()