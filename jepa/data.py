"""Dataset helpers shared by the image experiments."""

from pathlib import Path

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


CIFAR10_CLASSES = (
    "aeroplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_cifar10(root: str | Path = "data", *, train: bool, download: bool = False) -> CIFAR10:
    """Load one standard CIFAR-10 split as tensors with values between 0 and 1."""
    return CIFAR10(root=Path(root), train=train, transform=ToTensor(), download=download)

