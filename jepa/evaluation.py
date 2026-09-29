"""Representation measurements used by comparison experiments."""

import torch
from torch.nn import functional as F


def diversity_metrics(tokens: torch.Tensor) -> dict[str, float]:
    """Measure variation across normalized patch tokens shaped (images, patches, dimensions)."""
    tokens = F.normalize(tokens.detach().cpu(), dim=-1)
    flat = tokens.flatten(0, 1)
    image_vectors = F.normalize(tokens.mean(dim=1), dim=-1)
    image_centres = F.normalize(tokens.mean(dim=1, keepdim=True), dim=-1)
    centred = flat - flat.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centred)
    weights = singular_values / singular_values.sum().clamp_min(1e-12)

    return {
        "feature_std": flat.std(dim=0).mean().item(),
        "within_image_similarity": F.cosine_similarity(tokens, image_centres, dim=-1).mean().item(),
        "across_image_similarity": F.cosine_similarity(
            image_vectors, torch.roll(image_vectors, shifts=1, dims=0), dim=-1
        ).mean().item(),
        "effective_rank": torch.exp(-(weights * torch.log(weights + 1e-12)).sum()).item(),
    }


def nearest_neighbour_label_agreement(
    vectors: torch.Tensor, labels: torch.Tensor
) -> float:
    """Return the fraction whose nearest other vector has the same label."""
    vectors = F.normalize(vectors.detach().cpu(), dim=-1)
    similarities = vectors @ vectors.T
    similarities.fill_diagonal_(-torch.inf)
    neighbours = similarities.argmax(dim=1)
    return (labels.cpu()[neighbours] == labels.cpu()).float().mean().item()

