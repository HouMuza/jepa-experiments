# Glossary

## Channel

One measurement recorded at every pixel. An RGB image has red, green, and blue channels.

## Pixel

One location in an image. In an RGB image, one pixel contains three channel values.

## Patch

A small rectangular region of an image. A 4 by 4 RGB patch contains 48 channel values.

## Encoder

A neural network that converts input data into an internal representation.

## Embedding

A vector of numbers produced by an encoder. Training determines what information is useful for the
vector to preserve.

## Latent representation

The model's internal description of an input. In our experiments, this usually means the embedding
produced by an encoder.

## Predictor

The part of a JEPA that uses visible context embeddings to predict target embeddings.

## Context

The part of an input that the JEPA is allowed to see.

## Target

The hidden region whose representation the JEPA must predict.

## Exponential moving average

A slow update that makes the target encoder follow the context encoder. It gives the predictor a
more stable target during training.

## Linear probe

A small linear classifier trained on frozen embeddings. It tests how easily particular information
can be extracted from a representation.

## Invariance

The property of keeping a similar representation when an unimportant part of the input changes.

