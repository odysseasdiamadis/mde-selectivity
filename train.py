import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import logging
from tqdm import tqdm
import numpy as np

from architecture import MobileViT_S, MobileViT_XS, MobileViT_XXS
from loss import TotalVariationLoss


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


def get_model(config):
    """Get model based on configuration."""
    variant = config['model']['variant']
    models = {
        'xxs': MobileViT_XXS,
        'xs': MobileViT_XS,
        's': MobileViT_S
    }
    
    if variant not in models:
        raise ValueError(f"Unknown model variant: {variant}")
    
    return models[variant]()


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


def get_scheduler(optimizer, config):
    """Get learning rate scheduler based on configuration."""
    scheduler_name = config['scheduler']
    
    if scheduler_name == "StepLR":
        step_size = config['step_size']
        gamma = config['gamma']
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    elif scheduler_name == "CosineAnnealingLR":
        T_max = config['training']['num_epochs']
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def get_loss_function(config):
    """Get loss function based on configuration."""
    loss_name = config['loss']
    
    if loss_name == "MSELoss":
        return nn.MSELoss()
    elif loss_name == "L1Loss":
        return nn.L1Loss()
    elif loss_name == "SmoothL1Loss":
        return nn.SmoothL1Loss()
    elif loss_name == "TotalVariationLoss":
        return TotalVariationLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


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
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
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
    total_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Log batch loss
        if batch_idx % config['logging']['log_interval'] == 0:
            logging.info(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def train(config):
    """Main training function."""
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
    model = get_model(config)
    model = model.to(device)
    logging.info(f"Model: {config['model']['variant']} variant")
    
    # Get optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Get loss function
    criterion = get_loss_function(config)
    
    # TODO: Initialize dataloaders - this requires implementing dataset classes
    # For now, we'll create dummy dataloaders as placeholders
    logging.warning("Using dummy dataloaders - implement actual dataset loading")
    
    # Placeholder for actual dataloaders
    train_loader = None
    val_loader = None
    
    if train_loader is None or val_loader is None:
        logging.error("Dataloaders not implemented. Please implement dataset loading.")
        return
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        logging.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Log epoch results
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if (epoch + 1) % config['logging']['save_interval'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, val_loss,
                config['logging']['checkpoint_dir'], is_best
            )
    
    logging.info("Training completed!")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")


def main():
    """Main function."""
    config_path = "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = load_config(config_path)
    train(config)


if __name__ == "__main__":
    main()