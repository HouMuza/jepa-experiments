# JEPA Experiments

This project is a sequence of small experiments for understanding JEPA from first principles.
We begin with pixels and embeddings, compare pixel reconstruction with latent prediction, and
then add time and actions.

The experiments are designed for a 2026 Apple M5 MacBook with 16 GB of unified memory. Training
scripts will use Apple Metal (`mps`) when it is available and fall back to the CPU otherwise.

## The question

Can a model learn useful structure about the world by predicting representations of things it
cannot see, rather than recreating every missing pixel?

## Experiments

| Folder | Question |
| --- | --- |
| `01-image-embeddings` | How does an image become patches and vectors? |
| `02-autoencoder` | What does a model learn when it must reconstruct pixels? |
| `03-tiny-jepa` | Can it predict the embedding of a hidden region? |
| `04-pixel-vs-embedding` | How do pixel prediction and embedding prediction differ? |
| `05-test-embeddings` | What useful information is present in the embeddings? |
| `06-invariance` | What changes can the representation learn to ignore? |
| `07-temporal-jepa` | Can it predict the representation of the next moment? |
| `08-action-jepa` | Can it predict how an action changes the world? |

Each experiment has its own README. We will build them in order and record what we observe before
moving on.

## The book

The experiments and the explanation are developed together. The `book/` folder contains the
long-form account: the question behind each experiment, the intuition, diagrams, code walkthroughs,
results, mistakes, and conclusions. Start with the [book introduction](book/README.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Download and inspect CIFAR-10:

```bash
python 01-image-embeddings/download_cifar10.py
```

Start with:

```bash
python 01-image-embeddings/inspect_embeddings.py
```

The first script creates a small synthetic RGB image, divides it into 4 x 4 patches, projects each
48-number patch into a 128-number vector, and saves a visual explanation in `artifacts/`.

## Project layout

```text
JEPA Experiments/
├── jepa/                       shared code used by several experiments
├── artifacts/                  generated figures and results
├── book/                       long-form chapters and experiment records
├── 01-image-embeddings/
├── 02-autoencoder/
├── 03-tiny-jepa/
├── 04-pixel-vs-embedding/
├── 05-test-embeddings/
├── 06-invariance/
├── 07-temporal-jepa/
└── 08-action-jepa/
```
