"""Small models used across the JEPA experiments."""

import copy
import torch
from torch import nn
from torch.nn import functional as F

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

    def encode(self, images: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        patches = patchify(images, self.patch_size)
        tokens = self.patch_projection(patches)

        if mask is not None:
            if mask.ndim != 1 or mask.shape[0] != self.number_of_patches:
                raise ValueError(f"mask must have shape ({self.number_of_patches},)")
            tokens = torch.where(mask.reshape(1, -1, 1), self.mask_token, tokens)

        return self.encoder(tokens + self.position_embedding)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = patchify(images, self.patch_size)
        encoded = self.encode(images, mask)
        predicted_patches = self.pixel_head(encoded)
        return predicted_patches, patches


def masked_pixel_loss(
    predicted_patches: torch.Tensor, target_patches: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over hidden patches only."""
    return nn.functional.mse_loss(predicted_patches[:, mask], target_patches[:, mask])


class PatchEncoder(nn.Module):
    """Turn image patches into contextualized embedding vectors."""

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
        self.patch_size = patch_size
        self.number_of_patches = (image_size // patch_size) ** 2
        patch_values = channels * patch_size * patch_size
        self.patch_projection = nn.Linear(patch_values, embedding_dim)
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
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth, norm=nn.LayerNorm(embedding_dim)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, images: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        patches = patchify(images, self.patch_size)
        tokens = self.patch_projection(patches)
        if mask is not None:
            tokens = torch.where(mask.reshape(1, -1, 1), self.mask_token, tokens)
        return self.transformer(tokens + self.position_embedding)


class PatchPredictor(nn.Module):
    """Predict one target-encoder vector for every patch position."""

    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        number_of_patches: int = 64,
    ) -> None:
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, number_of_patches, embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth, norm=nn.LayerNorm(embedding_dim)
        )
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        return self.output_projection(self.transformer(context_tokens + self.position_embedding))


class TinyJEPA(nn.Module):
    """Online encoder, EMA target encoder, and latent predictor."""

    def __init__(self, *, embedding_dim: int = 128, ema_decay: float = 0.99) -> None:
        super().__init__()
        self.context_encoder = PatchEncoder(embedding_dim=embedding_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.predictor = PatchPredictor(embedding_dim=embedding_dim)
        self.ema_decay = ema_decay

    def train(self, mode: bool = True) -> "TinyJEPA":
        super().train(mode)
        self.target_encoder.eval()
        return self

    def predict(
        self, images: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            target = self.target_encoder(images)
        context = self.context_encoder(images, mask)
        predicted = self.predictor(context)
        return predicted, target

    @torch.no_grad()
    def update_target(self) -> None:
        for online_parameter, target_parameter in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_parameter.lerp_(online_parameter, 1.0 - self.ema_decay)


def masked_embedding_loss(
    predicted_tokens: torch.Tensor, target_tokens: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Normalized MSE between predicted and target embeddings at hidden positions."""
    predicted = F.normalize(predicted_tokens[:, mask], dim=-1)
    target = F.normalize(target_tokens[:, mask], dim=-1)
    return F.mse_loss(predicted, target)
