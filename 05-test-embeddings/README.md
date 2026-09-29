# 05 — Asking what an embedding knows

**Question:** How easily can a simple classifier read class information from each frozen representation?

Experiment 04 used nearest neighbours as a first semantic check. Here we use a linear probe: a
single learned linear layer placed on top of a frozen encoder. The encoder cannot change to solve
the classification task, so the probe measures information already present in its representation.

```bash
jupyter lab 05-test-embeddings/experiment.ipynb
```

Choose the `Python (JEPA Experiments)` kernel and run the cells in order. The notebook retrains the
two self-supervised models from Experiment 04, extracts frozen image vectors, and trains five probes
for each representation. CIFAR-10 labels are used only by the probes.

This notebook is intentionally standalone. It does not depend on model objects left in memory by a
previous notebook.

## First run

The pixel-model probe reached **35.3%** held-out accuracy, compared with **31.6%** for JEPA and
**31.3%** for raw pixels. Our current JEPA beat chance but did not produce a clear improvement over
the raw input. See the notebook and Chapter 5 for the interpretation.

The follow-up notebook, `collapse_diagnostics.ipynb`, tests random block masking and a variance
regularizer as a possible response.
