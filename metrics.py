from torch import Tensor
import torch

_eps = 1e-4

def RMSE(pred: Tensor, target: Tensor) -> Tensor:
    pred   = pred[..., 23:189, 22:615]
    target = target[..., 23:189, 22:615]
    diff2  = (pred - target)**2
    mse    = diff2.mean(dim=(1,2))
    return torch.mean(torch.sqrt(mse))

def REL(pred: Tensor, target: Tensor, eps: float = _eps) -> Tensor:
    pred   = pred[..., 23:189, 22:615]
    target = target[..., 23:189, 22:615]
    valid  = target > eps
    rel_map = torch.abs(pred[valid] - target[valid]) / (target[valid] + eps)
    return rel_map.mean()

def delta(pred: Tensor, target: Tensor, threshold: float = 1.25, eps: float = _eps) -> Tensor:
    pred   = pred[..., 23:189, 22:615]
    target = target[..., 23:189, 22:615]
    valid  = target > eps
    ratio  = torch.max(pred[valid]/(target[valid]+eps),
                       target[valid]/(pred[valid]+eps))
    return (ratio < threshold).float().mean()
