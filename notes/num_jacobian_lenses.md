# Background knowledge

> **Disclaimer:** Claude is generating / helping me get answers.

## Num of Jacobians / J-Lenses

### Question

> How many Jacobian lenses are there for one sentence? One per layer and per token?

### Answer

There are two different things here, so the count depends on which one you mean.

#### The true Jacobian: one per layer **and** per token

As you said. For a sentence with $T$ tokens and a model with $L$ layers, there are roughly $L \times T$ exact Jacobians. Each describes how the final layer at that token reacts to nudges at layer $l$ at that same token, in that specific context.

Because of attention, a nudge at one token can also change later tokens. I'm not sure whether the fitted lens includes those cross-token effects or only the same-token one; the fitting code in `jacobian-lens` would settle it.

#### The fitted J-Lens: one per layer, shared by every token and every sentence

That's the averaging step. All those <span style="color: red;">**per-token Jacobians from the fitting prompts**</span> are **pooled** into a single `d_model × d_model` matrix per layer. At readout time you use that same matrix whether the token is "elephant" in this sentence or "Paris" in another. You never compute a Jacobian for the sentence you're reading.

### In this repo

From `docs/README_lenses.md`:

- **J-Lens (`j.pt`):** one matrix per fitted layer, and only some layers are fitted. That's 7 matrices for Qwen3.5-9B (layers 4, 8, …, 28) and 7 for Qwen3.6-27B.
- **J++ (`jpp.pt`):** 8 matrices ("experts") per layer, so 56 in total. A router looks at each token's residual and decides how to combine the 8. That puts it in between: still a fixed set fitted once, but the token picks which part applies to it.

### Summary

For one sentence, you read every token at layer $l$ through:

| Lens   | Matrices used per token at layer $l$ |
| ------ | ------------------------------------ |
| J-Lens | the same 1 matrix                    |
| J++    | a per-token mix of 8                 |
