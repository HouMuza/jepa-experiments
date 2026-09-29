# 03 — Tiny image JEPA

**Question:** Can a model predict the embedding of a hidden image region without reconstructing its pixels?

This notebook reuses Experiment 02’s CIFAR-10 subset and central patch mask. It builds an online context encoder, a slowly updated EMA target encoder, and a predictor. The loss compares predicted and target embeddings only at the hidden patch positions.

The experiment records the embedding prediction loss and cosine similarity. These measurements are not numerically comparable to Experiment 02’s pixel MSE; they describe different targets.

## Run the lesson

```bash
jupyter lab 03-tiny-jepa/experiment.ipynb
```

Choose the `Python (JEPA Experiments)` kernel and run the cells from top to bottom.

## First run

The first ten-epoch run reached its best training fit at epoch 3:

| Measurement | Before training | Best epoch | Epoch 10 |
| --- | ---: | ---: | ---: |
| Hidden-patch loss | 0.014584 | 0.000090 | 0.000355 |
| Cosine similarity | 0.0666 | 0.9942 | 0.9773 |

The predictor matched the target vectors closely, but the target heatmap looked very similar across
hidden patches. The notebook now includes diversity diagnostics because high predictor–target
agreement can also occur when a representation collapses toward the same vector for many inputs.
