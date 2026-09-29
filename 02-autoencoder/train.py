"""Train the masked pixel-reconstruction baseline."""

import argparse
import json
from pathlib import Path

import matplotlib
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from jepa.data import load_cifar10
from jepa.device import best_device
from jepa.masking import centered_block_mask, paint_masked_patches
from jepa.models import PatchAutoencoder, masked_pixel_loss
from jepa.patches import unpatchify

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def save_loss_curve(losses: list[float], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(range(1, len(losses) + 1), losses, marker="o", color="#7c3aed")
    axis.set(xlabel="Epoch", ylabel="Masked pixel MSE", title="Experiment 02 training loss")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_reconstructions(
    originals: torch.Tensor,
    predictions: torch.Tensor,
    mask: torch.Tensor,
    output: Path,
) -> None:
    visible = paint_masked_patches(originals, mask.cpu(), patch_size=4)
    predicted_images = unpatchify(predictions.cpu(), patch_size=4, image_size=32).clamp(0, 1)
    completed = originals.clone()
    grid_mask = mask.cpu().reshape(8, 8)
    for row, column in grid_mask.nonzero(as_tuple=False):
        y, x = int(row) * 4, int(column) * 4
        completed[:, :, y : y + 4, x : x + 4] = predicted_images[:, :, y : y + 4, x : x + 4]

    figure, axes = plt.subplots(3, 6, figsize=(12, 6.5))
    for column in range(6):
        for row, images in enumerate((originals, visible, completed)):
            axes[row, column].imshow(images[column].permute(1, 2, 0).clamp(0, 1))
            axes[row, column].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=11)
    axes[1, 0].set_ylabel("Visible input", fontsize=11)
    axes[2, 0].set_ylabel("Reconstruction", fontsize=11)
    figure.suptitle("Masked pixel reconstruction after training")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)
    device = best_device()
    dataset = load_cifar10(train=True)
    sample_count = min(args.samples, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(sample_count)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = PatchAutoencoder().to(device)
    mask = centered_block_mask(8, 4, device=device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
    losses: list[float] = []

    print(f"Device: {device} | Training images: {sample_count:,} | Hidden patches: {mask.sum().item()}")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        examples_seen = 0
        for images, _ in loader:
            images = images.to(device)
            predicted, target = model(images, mask)
            loss = masked_pixel_loss(predicted, target, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.shape[0]
            examples_seen += images.shape[0]

        epoch_loss = total_loss / examples_seen
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1:02d}/{args.epochs}: {epoch_loss:.6f}")

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": vars(args), "losses": losses},
        artifacts / "02-autoencoder.pt",
    )
    (artifacts / "02-autoencoder-history.json").write_text(
        json.dumps({"config": vars(args), "losses": losses}, indent=2) + "\n"
    )
    save_loss_curve(losses, artifacts / "02-autoencoder-loss.png")

    model.eval()
    originals, _ = next(iter(loader))
    with torch.no_grad():
        predictions, _ = model(originals[:6].to(device), mask)
    save_reconstructions(
        originals[:6], predictions, mask, artifacts / "02-autoencoder-reconstructions.png"
    )
    print("Saved checkpoint and figures in artifacts/")


if __name__ == "__main__":
    main()

