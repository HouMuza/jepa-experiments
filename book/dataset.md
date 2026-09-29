# Dataset Record — CIFAR-10

## Why we chose it

CIFAR-10 gives us enough real images to train a small model while keeping every experiment practical
on an M5 MacBook with 16 GB of memory. Every image is already 32 by 32 pixels, which matches the
architecture used throughout the first experiments.

The labels will not be shown to JEPA during representation learning. We keep them so that a later
linear-probe experiment can measure whether the learned embeddings contain useful information about
object identity.

## Contents

| Split | Images | Purpose |
| --- | ---: | --- |
| Training | 50,000 | Learn model parameters |
| Test | 10,000 | Evaluate on unseen images |

Each image has shape `3 × 32 × 32`: three RGB channels, 32 rows, and 32 columns. The ten labels are
aeroplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Source and integrity

The dataset was created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton at the University of
Toronto. The canonical dataset page is:

<https://www.cs.toronto.edu/~kriz/cifar.html>

The local archive was downloaded from the Zenodo mirror because the canonical host was transferring
too slowly. We accepted it only after its MD5 checksum matched the checksum published for the
canonical archive:

```text
c58f30108f718f92721af3b95e74349a
```

## Local storage

The downloaded archive and extracted batches live in `data/`. That directory is excluded from Git
because datasets should be acquired by the script rather than committed to the repository.

```text
data/
├── cifar-10-python.tar.gz
└── cifar-10-batches-py/
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── test_batch
    └── batches.meta
```

Run the acquisition and inspection step with:

```bash
python 01-image-embeddings/download_cifar10.py
```

The script verifies the split sizes and image shape, then saves a sample grid to
`artifacts/01-cifar10-samples.png`.

