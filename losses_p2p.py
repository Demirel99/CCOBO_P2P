#losses.py
"""
Loss functions for model training.
"""
import torch
import torch.nn as nn

# We will use a standard regression loss like MSE or SmoothL1
# For simplicity, let's use MSELoss here. SmoothL1Loss is often preferred for robustness.

def coordinate_regression_loss(predicted_coords, target_coords):
    """
    Computes Mean Squared Error loss for coordinate regression.

    Args:
        predicted_coords (torch.Tensor): Predicted coordinates (B, 2).
        target_coords (torch.Tensor): Target coordinates (B, 2).

    Returns:
        torch.Tensor: Scalar MSE loss.
    """
    if predicted_coords.shape != target_coords.shape:
        raise ValueError(f"Shape mismatch: pred_coords {predicted_coords.shape}, target_coords {target_coords.shape}")
    
    loss_fn = nn.MSELoss() # Could also use nn.SmoothL1Loss()
    loss = loss_fn(predicted_coords, target_coords)
    return loss

# # Example of KL Divergence (kept for reference, but not used in this hybrid)
# def kl_divergence_loss(predicted_psf, target_psf, epsilon=1e-7):
#     pred_clamped = torch.clamp(predicted_psf, min=epsilon)
#     tgt_clamped = torch.clamp(target_psf, min=epsilon)
#     kl_div = torch.where(target_psf > epsilon,
#                          target_psf * (torch.log(tgt_clamped) - torch.log(pred_clamped)),
#                          torch.zeros_like(target_psf))
#     kl_loss = kl_div.sum(dim=(2, 3)).mean()
#     return kl_loss