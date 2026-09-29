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

## First run

Both models predicted their held-out targets. JEPA separated different images more strongly, while
the pixel representation had higher effective rank and slightly higher nearest-neighbour label
agreement (`18.3%` versus `16.5%`). This run did not establish a clear winner; see the notebook and
Chapter 4 for the full interpretation.
