# 5. Asking What an Embedding Knows

The previous experiment found a weak sign of class structure. Images with the same label were
nearest neighbours more often than chance, but that test depended on the exact geometry of the
embedding space. We now ask a more direct question:

> Can a simple classifier recover an image's class from a frozen representation?

![A linear probe reads a frozen representation](images/07-linear-probe.svg)

*Figure 5.1 — The encoder produces one vector per image. Its weights stay frozen. Only the small
linear classifier learns from CIFAR-10 labels.*

## What a linear probe is

We first train an encoder without labels. We then stop the encoder from learning and use it to turn
each image into one vector. A linear layer receives that vector and produces ten class scores:

```text
image → frozen encoder → 128-number vector → linear layer → 10 class scores
```

The linear layer can draw flat boundaries through the representation space. If it classifies images
well, the encoder has already arranged useful class information so that a simple boundary can read
it. If it performs poorly, class information may be absent, tangled together, or require a more
complex reader.

The probe does not prove that an embedding understands an object. It gives us one controlled and
repeatable measurement of how accessible the class information is.

## The controlled comparison

We retrain the pixel model and JEPA with the same settings used in Experiment 04. After this
self-supervised stage, both encoders are frozen. We then:

1. give each encoder complete, unmasked images;
2. average its 64 patch vectors into one 128-number image vector;
3. standardize each feature using training-set statistics;
4. train a single linear layer from 128 inputs to ten CIFAR-10 classes;
5. evaluate it on held-out test images.

The labels never update either encoder. Gradients stop at the linear layer.

## Why repeat the probe

A probe starts with random weights and sees randomly ordered batches. One run can therefore be
slightly fortunate or unfortunate. We train five probes for each frozen representation and report
the mean accuracy and its standard deviation.

This repetition measures variation in the probe stage. It does not measure every source of
uncertainty because the self-supervised models are still trained once. Repeating the complete
pretraining process would be a stronger and more expensive experiment.

## Three reference points

The result is easier to interpret beside three references:

- **Chance:** CIFAR-10 has ten balanced classes, so chance accuracy is about 10 percent.
- **Raw pixels:** a linear classifier sees the flattened RGB values directly.
- **Learned representations:** linear classifiers see frozen pixel-model or JEPA vectors.

The raw-pixel probe tells us whether either encoder offers a useful arrangement beyond a direct
linear reading of the input.

## Running the experiment

```bash
jupyter lab 05-test-embeddings/experiment.ipynb
```

Select `Python (JEPA Experiments)` and run the notebook in order. Record the measured accuracies in
the final notebook section before drawing a conclusion.

## What remains uncertain

This experiment uses one pretraining seed, a limited training subset, mean pooling, and one fixed
probe recipe. Its conclusion applies to these particular representations and settings. Later work
can repeat pretraining, tune the probe on a validation set, or test which individual visual
properties are encoded.
