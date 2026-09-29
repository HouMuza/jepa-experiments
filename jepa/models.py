"""Small models used across the JEPA experiments."""

import torch
from torch import nn

from jepa.patches import patchify


class PatchAutoencoder(nn.Module):
    """A small transformer that reconstructs RGB values for masked image patches."""

    def __init__(
        self,
        *,
        image_size: int = 32,
        patch_size: int = 4,
        embedding_dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        channels: int = 3,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_values = channels * patch_size * patch_size
        self.number_of_patches = (image_size // patch_size) ** 2

        self.patch_projection = nn.Linear(self.patch_values, embedding_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.number_of_patches, embedding_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, norm=nn.LayerNorm(embedding_dim))
        self.pixel_head = nn.Linear(embedding_dim, self.patch_values)

        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = patchify(images, self.patch_size)
        tokens = self.patch_projection(patches)

        if mask.ndim != 1 or mask.shape[0] != self.number_of_patches:
            raise ValueError(f"mask must have shape ({self.number_of_patches},)")

        tokens = torch.where(mask.reshape(1, -1, 1), self.mask_token, tokens)
        encoded = self.encoder(tokens + self.position_embedding)
        predicted_patches = self.pixel_head(encoded)
        return predicted_patches, patches


def masked_pixel_loss(
    predicted_patches: torch.Tensor, target_patches: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over hidden patches only."""
    return nn.functional.mse_loss(predicted_patches[:, mask], target_patches[:, mask])

