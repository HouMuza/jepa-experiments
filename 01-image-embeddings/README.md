# 01 — Image embeddings

## Question

How does a 32 x 32 RGB image become a sequence of 128-number vectors?

## What we do

1. Create one 32 x 32 image. It contains `32 x 32 x 3 = 3,072` channel values.
2. Divide it into 4 x 4 patches. This produces `8 x 8 = 64` patches.
3. Flatten each patch. One patch contains `4 x 4 x 3 = 48` values.
4. Pass every patch through the same linear projection to produce a 128-number vector.

The projection is deliberately untrained in this first demonstration. It lets us inspect the
mechanics before training decides what the vector should represent.

## Run it

From the project root:

```bash
python 01-image-embeddings/inspect_embeddings.py
```

Then open `artifacts/01-image-embeddings.png`.

## Download CIFAR-10

After installing the project dependencies, download and inspect the real dataset with:

```bash
python 01-image-embeddings/download_cifar10.py
```

This downloads the standard 50,000-image training split and 10,000-image test split into `data/`.
The files in that folder are deliberately excluded from Git. The script also saves ten labelled
examples to `artifacts/01-cifar10-samples.png`.

## What to notice

- RGB still means three values per pixel.
- The 128 embedding dimensions are learned features, not colour channels.
- There is one embedding for each patch, so the output shape is `64 x 128`.
- At this stage the vectors have no useful meaning because the projection has not been trained.
