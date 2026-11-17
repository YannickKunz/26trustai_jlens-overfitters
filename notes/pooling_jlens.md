# Background knowledge

> **Disclaimer:** Claude is generating / helping me get answers.

## Pooling: one J-Lens for every token

### Previous statement

> That's the averaging step. All those per-token Jacobians from the fitting prompts are pooled into a single `d_model × d_model` matrix per layer. At readout time you use that same matrix whether the token is "elephant" in this sentence or "Paris" in another. You never compute a Jacobian for the sentence you're reading.

### Question

> So this essentially allows me to see what happens to the final layer residual stream, if I change an arbitrary residual stream in the direction "size" at a given layer. Whether I change "size" for elefant, or for Paris, or for nostalgia, etc.?

### Answer

Yes, with one important qualifier: it shows you the **typical** effect, not the **actual** effect for that specific token.

$J_l \cdot \text{size\_direction}$ gives one answer, no matter whether the token is elephant, Paris or nostalgia. That answer is the average of how "size" spread forward across all the fitting tokens. The real model can behave differently in each case:

- For **elephant**, a nudge to size might really spill into "weight."
- For **Paris**, it might spill into "population" or "importance."
- For **nostalgia**, "size" may hardly mean anything, and the nudge may do almost nothing.

The fitted lens blurs these together. It's accurate for a given token only to the extent that the token behaves like the average. That also makes it a linear, small-change approximation of an average.

### Two consequences

1. **The fitting data shapes the answer.** If the fitting prompts were mostly encyclopedic WikiText, the average reflects how "size" behaves in that kind of text. Whether it still holds for French text, math (GSM8K), or unusual words like "nostalgia" is the generalization question your repo is testing with the en/fr/wikitext runs and prompt-size sweeps.
2. **You can check it per token.** For a specific token, compute the true Jacobian with one backward pass through the model. Then compare $J_{\text{true}} \cdot \text{size}$ with $J_{\text{fitted}} \cdot \text{size}$. The gap tells you how much the averaging loses for that case.

### Practical note

When you actually read with the lens, you don't apply $J$ to a single direction like "size." You apply it to the whole residual, $J_l \cdot h_l$. Because $J$ is linear, that equals translating every direction in $h_l$ forward at once ("size," "animal," "grey," …) and adding the results. Your "size" thought experiment is just one slice of that.
