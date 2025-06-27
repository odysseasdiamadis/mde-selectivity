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
from torchvision import transforms

from architecture import build_METER_model
from augmentation import CShift, DShift, augmentation2D
from data import NYUDataset
from loss import balanced_loss_function



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


def get_optimizer(model, config):
    """Get optimizer based on configuration."""
    optimizer_name = config['optimizer']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        momentum = config['training'].get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")



def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
    
    logging.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = torch.tensor([0.0], device=device)
    num_batches = len(dataloader)
    log_step = config["training"].get("log_step_loss")

    step_losses = []

    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, targets)

        if log_step:
            step_losses.append({
                "depth": loss[0].item(),
                "grad": loss[1].item(),
                "normal": loss[2].item(),
                "ssim": loss[3].item(),
            })
        
        loss = torch.stack(loss, dim=0).sum()

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Log batch loss
        if config['logging']['log_interval'] > 0 and batch_idx % config['logging']['log_interval'] == 0:
            logging.info(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches, step_losses


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = torch.tensor(0.0)
    num_batches = len(dataloader)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss = torch.stack(loss, dim=0).sum()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def train(config, ckpt_path=None) -> None:
    dtype = torch.bfloat16
    # Setup logging
    setup_logging(config['logging']['log_dir'])
    logging.info("Starting training...")
    logging.info(f"Configuration: {config}")
    
    # Setup device
    if config['device']['cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['device_id']}")
        logging.info(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    
    # Get model
    model = build_METER_model(device, arch_type=config['model']['variant'])


    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20)
    saved_epoch, saved_loss = 0, 0

    if ckpt_path:
        saved_epoch, saved_loss = load_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, checkpoint_path=ckpt_path)
    
    # Get loss function
    criterion = balanced_loss_function(device, dtype=dtype)
    
    model = model.to(device=device, dtype=dtype)
    criterion = criterion.to(device="cuda", dtype=dtype)
    
    # Placeholder for actual dataloaders
    dataset = NYUDataset(
        root=config['data']['root'],
        test=False,
        transforms=t.Compose([
            t.ToImage(),
            t.ToDtype(dtype, scale=True),  # scale to [0, 1]
            t.RandomHorizontalFlip(p=0.5),
            t.RandomVerticalFlip(p=0.5),
            CShift(),
            DShift()
            ])
    )

    pin_memory = (device == "cuda")
    train_ds, val_ds = torch.utils.data.random_split(dataset, config['data']["train_val_split"])
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], num_workers=config['data']['num_workers'], pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], num_workers=config['data']['num_workers'])
    
    # Training loop
    best_val_loss = float('inf')
    losses = []
    
    for epoch in range(saved_epoch,config['training']['num_epochs']):
        logging.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, epoch_losses = train_epoch(model, train_loader, criterion, optimizer, device, config)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch results
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch+1} - Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, LR: {current_lr:.6f}")
        losses = losses + epoch_losses
        # Save checkpoint
        is_best: bool = val_loss.item() < best_val_loss
        if is_best:
            best_val_loss: float = val_loss.item()
        
        if (epoch + 1) % config['logging']['save_interval'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, val_loss,
                config['logging']['checkpoint_dir'], is_best
            )
    
    logging.info("Training completed!")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    pd.DataFrame(losses).to_csv(os.path.join(config['logging']['log_dir'], './epoch_losses.csv'))



def main():
    config_path = "config.yaml"
    ckpt_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if len(sys.argv) > 2:
        ckpt_path = sys.argv[2]
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = load_config(config_path)
    train(config, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()