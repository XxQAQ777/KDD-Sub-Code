import torch
import numpy as np


def masked_wasserstein(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked Wasserstein Distance (1-Wasserstein distance / Earth Mover's Distance).

    For deterministic forecasts, the 1-Wasserstein distance equals MAE.
    This implementation computes the average absolute difference between predictions and labels.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked Wasserstein distance
    """

    # fix very small values of labels, which should be 0. Otherwise, nan detector will fail.
    labels = torch.where(labels < 1e-4, torch.zeros_like(labels), labels)

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # For deterministic forecasts (point predictions),
    # 1-Wasserstein distance = MAE
    wasserstein = torch.abs(preds - labels)
    wasserstein = wasserstein * mask
    wasserstein = torch.where(torch.isnan(wasserstein), torch.zeros_like(wasserstein), wasserstein)

    return torch.mean(wasserstein)
