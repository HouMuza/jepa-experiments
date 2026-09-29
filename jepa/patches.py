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


def unpatchify(
    patches: torch.Tensor, *, patch_size: int, image_size: int, channels: int = 3
) -> torch.Tensor:
    """Reassemble patch vectors into images shaped (batch, channels, height, width)."""
    if patches.ndim != 3:
        raise ValueError("patches must have shape (batch, number_of_patches, patch_values)")

    grid_size = image_size // patch_size
    expected_patches = grid_size * grid_size
    expected_values = channels * patch_size * patch_size
    if patches.shape[1:] != (expected_patches, expected_values):
        raise ValueError(
            f"expected patch shape (*, {expected_patches}, {expected_values}), "
            f"received {tuple(patches.shape)}"
        )

    batch = patches.shape[0]
    images = patches.reshape(batch, grid_size, grid_size, channels, patch_size, patch_size)
    images = images.permute(0, 3, 1, 4, 2, 5)
    return images.reshape(batch, channels, image_size, image_size)
