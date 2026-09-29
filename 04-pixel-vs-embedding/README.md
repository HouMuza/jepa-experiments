# 04 — Pixels versus embeddings

**Question:** What changes when the training target is an embedding instead of pixels?

We train the masked pixel model and tiny JEPA under the same controlled conditions. Raw loss values
cannot be compared because they have different units, so we compare relative learning progress,
held-out prediction, representation diversity, and nearest-neighbour label agreement.

```bash
jupyter lab 04-pixel-vs-embedding/experiment.ipynb
```

Choose the `Python (JEPA Experiments)` kernel. This notebook trains both models, so it takes longer
than the earlier lessons.
