import torch
import numpy as np


def masked_crps(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked Continuous Ranked Probability Score.

    CRPS measures the difference between the predicted and observed cumulative distribution functions.
    For deterministic forecasts, CRPS reduces to MAE.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked CRPS
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

    # For deterministic forecasts, CRPS = MAE
    # CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
    # For deterministic forecast, the second term is 0, so CRPS = |pred - y|
    crps = torch.abs(preds - labels)
    crps = crps * mask
    crps = torch.where(torch.isnan(crps), torch.zeros_like(crps), crps)

    return torch.mean(crps)
