"""Download CIFAR-10, inspect its shape, and save a labelled sample grid."""

from pathlib import Path

import matplotlib
import torch

from jepa.data import CIFAR10_CLASSES, load_cifar10

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_sample_grid(images: torch.Tensor, labels: list[int], output: Path) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(11, 5))
    for axis, image, label in zip(axes.flat, images, labels, strict=True):
        axis.imshow(image.permute(1, 2, 0))
        axis.set_title(CIFAR10_CLASSES[label])
        axis.axis("off")

    figure.suptitle("Ten examples from the CIFAR-10 training set", fontsize=15)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    train_data = load_cifar10(train=True, download=True)
    test_data = load_cifar10(train=False, download=True)

    examples = [train_data[index] for index in range(10)]
    images = torch.stack([image for image, _ in examples])
    labels = [label for _, label in examples]

    expected_shape = (3, 32, 32)
    if tuple(images[0].shape) != expected_shape:
        raise RuntimeError(f"Expected image shape {expected_shape}, received {tuple(images[0].shape)}")
    if len(train_data) != 50_000 or len(test_data) != 10_000:
        raise RuntimeError("CIFAR-10 split sizes do not match the published dataset")

    output = Path("artifacts/00-cifar10-samples.png")
    save_sample_grid(images, labels, output)

    print(f"Training images: {len(train_data):,}")
    print(f"Test images:     {len(test_data):,}")
    print(f"Image shape:     {tuple(images[0].shape)}")
    print(f"Pixel range:     {images.min().item():.3f} to {images.max().item():.3f}")
    print(f"Saved samples:   {output}")


if __name__ == "__main__":
    main()

