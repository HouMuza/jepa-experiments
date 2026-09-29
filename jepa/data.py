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

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_cifar10(
    root: str | Path | None = None, *, train: bool, download: bool = False
) -> CIFAR10:
    """Load one standard CIFAR-10 split as tensors with values between 0 and 1."""
    data_root = PROJECT_ROOT / "data" if root is None else Path(root)
    return CIFAR10(root=data_root, train=train, transform=ToTensor(), download=download)
