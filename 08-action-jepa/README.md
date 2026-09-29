# 08 — Action-conditioned JEPA

**Question:** Can a model predict how an action changes the state of a world?

In a tiny simulated world, the predictor will receive the present embedding and an action, then
predict the next embedding: `(z_t, a_t) -> z_(t+1)`.

