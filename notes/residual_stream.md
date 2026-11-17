# Background knowledge

> **Disclaimer:** Claude is generating / helping me get answers.

## Residue

> *"The part that is left once the main part has gone or been taken away."*

## Residual (stream)

A **residual** (the **residual stream**) is the vector a transformer keeps for each token position, which every layer reads from and adds to.

- **Start:** the token's embedding, a vector of size `d_model` (e.g. 4096).
- **Each layer** (attention, then MLP) reads that vector, computes an update, and adds it back:

  $$
  h_{l+1} = h_l + \mathrm{Attn}(h_l) + \mathrm{MLP}(\dots)
  $$

  The name comes from residual (skip) connections: a layer only writes a *change* (the "residual") on top of what is already there.
- **End:** after the last layer, a final norm and the unembedding (LM head) turn the vector into next-token logits.

So $h_l$, "the residual at layer $l$", is a snapshot of that running vector partway through the network: everything the model has computed about that token so far. Lenses like the **logit lens** and **J-Lens** take this mid-layer snapshot and try to read off which tokens it is already "thinking" of.

### Potential for confusion

$h_l$ **is** the residual: it's the whole running vector. $\mathrm{Attn}(\cdot)$ and $\mathrm{MLP}(\cdot)$ are each layer's *update* (its "write" to the stream).

My earlier wording was confusing, because "residual" means two different things:

- **In the original ResNet paper:** in $x + F(x)$, the "residual" is $F(x)$, the change a layer learns. That's where the name comes from.
- **In interpretability (and the J-Lens paper):** "the residual" or "residual stream" means $h_l$, the sum of the embedding and every update so far:

  $$
  h_l = \text{embedding} + \sum_{k<l} \left(\mathrm{attn}_k + \mathrm{mlp}_k\right)
  $$

When this repo or the J-Lens paper talks about the residual at layer $l$, it means $h_l$, the snapshot the lens reads.
