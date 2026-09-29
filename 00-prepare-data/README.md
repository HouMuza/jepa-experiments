# 00 — Prepare the data

## Question

What data will every image experiment use, and how do we know it was acquired correctly?

## What we do

1. Download the standard CIFAR-10 training and test splits.
2. Verify that they contain 50,000 and 10,000 images.
3. Confirm that every image has shape `3 × 32 × 32`.
4. Save a labelled sample grid so we can inspect what the model will see.

This preparation happens once. Experiments 01–06 load the resulting files from `data/` without
having to download or validate them again.

## Run it

After installing the project dependencies, run this from the project root:

```bash
python 00-prepare-data/prepare_cifar10.py
```

The downloaded files are stored in `data/`, which is deliberately excluded from Git. The sample
grid is saved to `artifacts/00-cifar10-samples.png`.

