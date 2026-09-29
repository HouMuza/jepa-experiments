"""Mask construction and visualization helpers."""

import torch


def centered_block_mask(grid_size: int, block_size: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return a flat boolean mask for a square block at the centre of a patch grid."""
    if block_size <= 0 or block_size > grid_size:
        raise ValueError("block_size must be between 1 and grid_size")

    start = (grid_size - block_size) // 2
    mask = torch.zeros(grid_size, grid_size, dtype=torch.bool, device=device)
    mask[start : start + block_size, start : start + block_size] = True
    return mask.flatten()


def random_block_mask(
    grid_size: int,
    block_size: int,
    *,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a flat boolean mask for a square block at a random valid position."""
    if block_size <= 0 or block_size > grid_size:
        raise ValueError("block_size must be between 1 and grid_size")

    positions = grid_size - block_size + 1
    row = int(torch.randint(positions, (), generator=generator))
    column = int(torch.randint(positions, (), generator=generator))
    mask = torch.zeros(grid_size, grid_size, dtype=torch.bool, device=device)
    mask[row : row + block_size, column : column + block_size] = True
    return mask.flatten()


def paint_masked_patches(
    images: torch.Tensor, mask: torch.Tensor, *, patch_size: int, value: float = 0.5
) -> torch.Tensor:
    """Replace masked image patches with a constant value for display."""
    masked = images.clone()
    grid_size = images.shape[-1] // patch_size
    mask_grid = mask.reshape(grid_size, grid_size)
    for row, column in mask_grid.nonzero(as_tuple=False):
        y = int(row) * patch_size
        x = int(column) * patch_size
        masked[:, :, y : y + patch_size, x : x + patch_size] = value
    return masked

