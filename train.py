import sys
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import logging
from tqdm import tqdm
from torchvision.transforms import v2 as t

from architecture import build_METER_model
from data import NYUDataset
from loss import L_assign, ResponseCompute, balanced_loss_function

torch.manual_seed(42)

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = torch.inf

    def __call__(self, val_loss):
        score = -val_loss  # Since we want to minimize loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        else:
            self.best_score = score
            self.counter = 0
        return False


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


def get_scheduler(config, optim: torch.optim.Optimizer):
    opt_conf = config["training"].get("optimizer")
    if opt_conf is None:
        return torch.optim.lr_scheduler.StepLR(optim, 20)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", threshold=0.1, patience=5
    )


def save_checkpoint(
    model, optimizer, scheduler, epoch, loss, checkpoint_dir, is_best=False
):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth")
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(checkpoint, best_path)

    logging.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]


def train_epoch(
    model,
    dataloader,
    l_assign,
    criterion,
    optimizer,
    device,
    config,
):
    """Train for one epoch."""
    model.train()
    use_selectivity = config["training"]["selectivity"]

    epoch_total_loss = torch.tensor([0.0], device=device)
    epoch_total_selectivity_loss = torch.tensor([0.0], device=device)
    epoch_total_depth_loss = torch.tensor([0.0], device=device)
    num_batches = len(dataloader)
    log_step = config["training"].get("log_step_loss")

    step_losses = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs, fmaps = model(images)
        loss_base = criterion(outputs, targets)

        if log_step:
            step_losses.append(
                {
                    "depth": loss_base[0].item(),
                    "grad": loss_base[1].item(),
                    "normal": loss_base[2].item(),
                    "ssim": loss_base[3].item(),
                }
            )

        loss_base = torch.stack(tensors=loss_base, dim=0).sum()
        if use_selectivity:
            loss_assign = l_assign(model, (images, targets), fmaps)
        else:
            loss_assign = torch.tensor(0.0, device=device)

        loss = loss_base + loss_assign
        loss.backward()
        optimizer.step()

        epoch_total_loss += loss.item()
        epoch_total_selectivity_loss += loss_assign.item()
        epoch_total_depth_loss += loss_base.item()
        # Update progress bar
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Log batch loss
        if (
            config["logging"]["log_interval"] > 0
            and batch_idx % config["logging"]["log_interval"] == 0
        ):
            logging.info(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

    return (
        epoch_total_loss / num_batches,
        step_losses,
        epoch_total_depth_loss / num_batches,
        epoch_total_selectivity_loss / num_batches,
    )


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

            outputs, fmaps = model(images)
            loss = criterion(outputs, targets)
            loss = torch.stack(loss, dim=0).sum()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


def train(config_path, ckpt_file=None) -> None:
    config = load_config(config_path)
    dtype = torch.float32
    # Setup logging
    exp_name = config["logging"]["experiment_name"]
    ckpt_dir = os.path.join("experiments", exp_name, "checkpoints")
    log_dir = os.path.join("experiments", exp_name, "logs")

    setup_logging(log_dir)
    logging.info(f"Configuration: {config}")

    # Setup device
    if config["device"]["cuda"] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['device_id']}")
        logging.info(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # Get model
    model = build_METER_model(
        device,
        arch_type=config["model"]["variant"],
        fmap_decoder=config["training"].get("fmaps_decoder"),
    )

    model = torch.nn.DataParallel(model, device_ids=[config["device"]["device_id"]])
    model = model.to(device)
    use_selectivity = config["training"]["selectivity"]
    if use_selectivity:
        resp_compute = ResponseCompute(
            model, device=device, n_of_bins=config["training"]["n_of_bins"]
        )
        l_assign = L_assign(
            resp_compute,
            config["training"]["lambda"],
            device,
            assign_formula=config["training"].get("formula"),
        )
    else:
        l_assign = None  # placeholder for LSPs

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay") or 0.01,
    )
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20)
    scheduler = get_scheduler(config, optimizer)

    criterion = balanced_loss_function(device, dtype=dtype)
    model = model.to(device=device, dtype=dtype)
    criterion = criterion.to(device=device, dtype=dtype)
    saved_epoch, saved_loss = 0, 0

    if ckpt_file:
        saved_epoch, saved_loss = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=ckpt_file,
            device=device,
        )

    dataset = NYUDataset(
        root=config["data"]["root"],
        test=False,
    )

    pin_memory = device.type == "cuda"
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, config["data"]["train_val_split"]
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # Training loop
    best_val_loss = float("inf")
    losses = []

    early_stop = EarlyStopping(20, delta=0.1)
    for epoch in range(saved_epoch, config["training"]["num_epochs"]):
        logging.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")

        # Train
        train_loss, epoch_losses, loss_depth, loss_selectivity = train_epoch(
            model,
            train_loader,
            l_assign=l_assign,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config,
        )

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)

        if early_stop(val_loss):
            logging.info(f"Triggered early stopping at epoch {epoch+1}")
            break

        if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            scheduler.step()
        elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss, epoch)

        # Log epoch results
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(
            f"Epoch {epoch+1} - Train Loss: {train_loss.item():.4f} ({loss_depth.item():.4f} + {loss_selectivity.item():.4f}), Val Loss: {val_loss.item():.4f}, LR: {current_lr:.6f}"
        )
        losses = losses + epoch_losses

        is_best: bool = val_loss.item() < best_val_loss
        if is_best:
            best_val_loss: float = val_loss.item()

        if (epoch + 1) % config["logging"]["save_interval"] == 0 or is_best:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                val_loss,
                ckpt_dir,
                is_best,
            )

    logging.info("Training completed!")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    pd.DataFrame(losses).to_csv(os.path.join(log_dir, "./epoch_losses.csv"))


def main():
    config_path = "config.yaml"
    ckpt_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if len(sys.argv) > 2:
        ckpt_path = sys.argv[2]

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    train(config_path, ckpt_file=ckpt_path)


if __name__ == "__main__":
    main()
