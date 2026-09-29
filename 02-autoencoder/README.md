# 02 — Masked pixel reconstruction

**Question:** What does a model learn when it must reconstruct missing pixels?

We hide the central 16 × 16 region of each CIFAR-10 image. A small transformer sees the visible
patches and learns to predict the 768 hidden RGB values. This gives us the pixel-prediction baseline
for the later JEPA comparison.

## The path through the model

```text
32 × 32 RGB image
        ↓ divide into 4 × 4 patches
64 patches, each containing 48 values
        ↓ replace 16 central patches with a mask token
small transformer
        ↓
48 predicted RGB values for every patch
        ↓ compare only the 16 hidden patches
masked pixel loss
```

## Run the lesson

```bash
jupyter lab 02-autoencoder/experiment.ipynb
```

Choose the `Python (JEPA Experiments)` kernel and run the cells from top to bottom.

## Run training without the notebook

```bash
python 02-autoencoder/train.py --epochs 10 --samples 10000
```

The script saves the model, loss history, loss curve, and reconstruction examples in `artifacts/`.
