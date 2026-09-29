# 4. Pixels Versus Embeddings

We now place the two prediction targets side by side.

Both models receive the same CIFAR-10 images. Both lose the same central region. Both use
four-block encoders with 128-number patch tokens. The difference is what counts as the correct
answer.

## The question

What changes when a model predicts a hidden representation instead of hidden pixels?

![Pixel targets compared with embedding targets](images/06-pixels-versus-embeddings.svg)

*Figure 4.1 — The input and hidden region are shared. The prediction targets and meanings of the
losses are different.*

## What we hold fixed

| Choice | Both models use |
| --- | ---: |
| Training images | 10,000 |
| Held-out test images | 1,000 |
| Image size | 32 × 32 |
| Patch size | 4 × 4 |
| Hidden region | central 16 patches |
| Encoder token dimension | 128 |
| Encoder blocks | 4 |
| Attention heads | 4 |
| Batch size | 64 |
| Training epochs | 5 |
| Learning rate | 0.0003 |

JEPA also needs a two-block predictor. Its complete trainable model is therefore larger. The
notebook prints both parameter counts so this limitation remains visible.

## Why the raw losses cannot be compared

The pixel model measures squared error between RGB values. JEPA measures squared error between
normalized vectors. A value of `0.03` in one objective does not mean the same thing as `0.03` in the
other.

We instead divide each epoch loss by that model’s loss before training:

```text
relative loss = current loss ÷ initial loss
```

This tells us how much of each model’s starting error remains. It does not pretend that the two
targets have acquired the same units.

## Four comparisons

### 1. Relative learning progress

How rapidly does each model reduce its own error under the same training budget?

### 2. Held-out prediction

Does each model also predict its target on 1,000 images that were not used for training? Each result
is interpreted within its own objective.

### 3. Representation diversity

We give both encoders complete test images and measure:

- feature standard deviation;
- similarity among patches inside an image;
- similarity across different images;
- effective representation rank out of 128.

A collapsed representation has little variation, very high similarity across unrelated inputs, and
low effective rank.

### 4. Nearest-neighbour label agreement

We average the 64 patch vectors into one vector per image. For each test image, we find its closest
other image and ask whether their CIFAR-10 labels match.

The models never see labels during training. Labels appear only in this evaluation. With ten
balanced categories, chance agreement is roughly 10 percent. Higher agreement suggests that nearby
representations share some semantic structure.

This is an early diagnostic. Experiment 05 will use a trained linear probe for a more direct test.

## Running the experiment

```bash
jupyter lab 04-pixel-vs-embedding/experiment.ipynb
```

Select `Python (JEPA Experiments)` and run the notebook in order. It trains both models, so it will
take longer than the earlier notebooks.

## How we will write the conclusion

We will not choose a winner from one convenient number. The conclusion must distinguish:

1. prediction fit;
2. generalization to held-out images;
3. resistance to representation collapse;
4. semantic information in the embedding neighbourhoods.

The executed results will be added here after the run.

