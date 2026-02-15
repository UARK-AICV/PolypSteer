"""
Training losses for medical image diffusion models.

Contains the color distribution loss that encourages generated images to match
the per-channel color statistics of ground truth medical images.
"""

import torch
import torch.nn.functional as F


def color_distribution_loss(gen_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 loss on per-channel color statistics (mean and std) between
    generated and target images.

    This loss encourages the generated images to match the overall color
    distribution of the target medical images, which is important for
    maintaining realistic tissue coloring in endoscopy images.

    Args:
        gen_images: Generated images tensor of shape (B, C, H, W).
        target_images: Target images tensor of shape (B, C, H, W).

    Returns:
        Scalar loss tensor.
    """
    batch_size = gen_images.size(0)

    total_loss = 0.0

    for i in range(batch_size):
        gen_image = gen_images[i]  # (C, H, W)
        target_image = target_images[i]  # (C, H, W)

        gen_mean = gen_image.mean(dim=[1, 2])  # (C,)
        gen_std = gen_image.std(dim=[1, 2])  # (C,)
        target_mean = target_image.mean(dim=[1, 2])  # (C,)
        target_std = target_image.std(dim=[1, 2])  # (C,)

        loss_mean = F.mse_loss(gen_mean, target_mean)
        loss_std = F.mse_loss(gen_std, target_std)

        total_loss += loss_mean + loss_std

    return total_loss / batch_size
