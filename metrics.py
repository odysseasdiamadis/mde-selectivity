from torch import Tensor
import torch

MAX = 1000.0
_eps = 1e-4

def RMSE(pred: Tensor, target: Tensor) -> Tensor:
    mask = (target > 0) & (target < MAX)
    diff2  = (pred[mask] - target[mask])**2
    mse    = diff2.view(pred.shape).mean(dim=(-1, -2))
    return torch.sqrt(input=mse).mean()

def REL(pred: Tensor, target: Tensor) -> Tensor:
    mask = (target > 0) & (target < MAX)  # Avoid divide by zero
    rel = torch.abs(target - pred)[mask] / (target[mask]+_eps)
    return torch.mean(rel)

def delta(pred: Tensor, target: Tensor, threshold: float = 1.25) -> Tensor:
    mask = (target > 0) & (target < MAX)
    pred = pred[mask]
    target = target[mask]
    ratio = torch.max(pred / (target+_eps), target / (pred+_eps))
    return torch.mean((ratio < threshold).float())