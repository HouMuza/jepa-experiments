# 01 — Image embeddings

## Question

How does a 32 x 32 RGB image become a sequence of 128-number vectors?

## What we do

1. Load one prepared CIFAR-10 image. It contains `32 x 32 x 3 = 3,072` channel values.
2. Divide it into 4 x 4 patches. This produces `8 x 8 = 64` patches.
3. Flatten each patch. One patch contains `4 x 4 x 3 = 48` values.
4. Pass every patch through the same linear projection to produce a 128-number vector.

The projection is deliberately untrained in this first demonstration. It lets us inspect the
mechanics before training decides what the vector should represent. The image label is displayed
for us, but it is never given to the projection.

## Run it

From the project root:

```bash
python 01-image-embeddings/inspect_embeddings.py
```

Then open `artifacts/01-image-embeddings.png`.

For the step-by-step lesson, open:

```bash
jupyter lab 01-image-embeddings/experiment.ipynb
```

Run the first code cell before the imports below it. That cell locates the project root and makes
the shared `jepa` package available to the active notebook kernel.

The notebook uses the `Python (JEPA Experiments)` kernel, which points to the Python environment
where this project and its dependencies are installed. If Jupyter asks you to choose a kernel,
select that one.

## What to notice

- RGB still means three values per pixel.
- The 128 embedding dimensions are learned features, not colour channels.
- There is one embedding for each patch, so the output shape is `64 x 128`.
- At this stage the vectors have no useful meaning because the projection has not been trained.
