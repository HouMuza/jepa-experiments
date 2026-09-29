"""Visualize the path from one CIFAR-10 image to patch embeddings."""

from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torch import nn

from jepa.data import CIFAR10_CLASSES, load_cifar10
from jepa.device import best_device
from jepa.patches import patchify

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PATCH_SIZE = 4
EMBEDDING_DIM = 128
EXAMPLE_INDEX = 0


def draw(image: torch.Tensor, label: int, embeddings: torch.Tensor, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(image.permute(1, 2, 0).numpy())
    axes[0].set_title(f"CIFAR-10: {CIFAR10_CLASSES[label]}\n32 x 32 x 3 = 3,072 values")
    axes[0].set_xticks(np.arange(-0.5, image.shape[2], PATCH_SIZE), minor=True)
    axes[0].set_yticks(np.arange(-0.5, image.shape[1], PATCH_SIZE), minor=True)
    axes[0].grid(which="minor", color="white", linewidth=0.7)
    axes[0].tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    patch_row, patch_column = 4, 4
    axes[0].add_patch(
        Rectangle(
            (patch_column * PATCH_SIZE - 0.5, patch_row * PATCH_SIZE - 0.5),
            PATCH_SIZE,
            PATCH_SIZE,
            fill=False,
            edgecolor="#f3b33d",
            linewidth=3,
        )
    )
    selected_patch = image[
        :,
        patch_row * PATCH_SIZE : (patch_row + 1) * PATCH_SIZE,
        patch_column * PATCH_SIZE : (patch_column + 1) * PATCH_SIZE,
    ].permute(1, 2, 0)
    axes[1].imshow(selected_patch.numpy(), interpolation="nearest")
    axes[1].set_title("One enlarged 4 x 4 patch\n48 values = 4 x 4 x 3")
    axes[1].axis("off")

    heatmap = axes[2].imshow(embeddings.numpy(), aspect="auto", cmap="coolwarm")
    axes[2].set_title("64 patch embeddings\n128 numbers per patch")
    axes[2].set_xlabel("Embedding dimension")
    axes[2].set_ylabel("Patch number")
    figure.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)

    figure.suptitle("Experiment 01: pixels → patches → vectors", fontsize=15)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    torch.manual_seed(7)
    device = best_device()
    dataset = load_cifar10(train=True)
    image, label = dataset[EXAMPLE_INDEX]
    patches = patchify(image.unsqueeze(0), PATCH_SIZE)

    projection = nn.Linear(patches.shape[-1], EMBEDDING_DIM).to(device)
    with torch.no_grad():
        embeddings = projection(patches.to(device)).squeeze(0).cpu()

    output = Path("artifacts/01-image-embeddings.png")
    draw(image, label, embeddings, output)

    print(f"Device:              {device}")
    print(f"Example:             {CIFAR10_CLASSES[label]}")
    print(f"Image shape:         {tuple(image.shape)}")
    print(f"Patch tensor shape:  {tuple(patches.shape)}")
    print(f"Embedding shape:     {tuple(embeddings.shape)}")
    print(f"Saved figure:        {output}")


if __name__ == "__main__":
    main()
