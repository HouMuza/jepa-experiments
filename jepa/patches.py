"""Functions for turning small images into non-overlapping patches."""

import torch


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert images shaped (batch, channels, height, width) into patch vectors."""
    if images.ndim != 4:
        raise ValueError("images must have shape (batch, channels, height, width)")

    batch, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("image height and width must be divisible by patch_size")

    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    return patches.reshape(batch, -1, channels * patch_size * patch_size)

