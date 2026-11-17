# Background knowledge

> **Disclaimer:** Claude is generating / helping me get answers.

## Jacobian-Lens: what is one entry?

### Question

> I'm trying to understand this, is this conceptually correct: assume at layer $l$ the residual stream describes the token/word "elefant". If I change an entry in the residual stream that hypothetically describes the **height** of the elefant, then an entry describing the **weight** of the elefant may change in the final layer. This change would be one entry in the Jacobian-Lens?

### Answer

That's basically right. One entry of the Jacobian is:

$$
J_l[i, j] = \frac{\partial\, h_{\text{final}}[i]}{\partial\, h_l[j]}
$$

In words: if I nudge entry $j$ (your "height") at layer $l$ a tiny bit, how much does entry $i$ (your "weight") at the final layer move? So your example describes one entry exactly.

Four things make the picture more accurate:

1. **It's a rate, not a yes/no.** The entry is a number like "weight moves 0.3 units per unit of height." It's a linear approximation, so it only holds for small nudges.
2. **"Height" probably isn't a single entry.** Models store concepts as *directions*, meaning combinations of many entries, and many concepts share the same entries. That's fine for the Jacobian: it's a matrix, so it maps any direction at layer $l$ to a direction at the final layer. Your example becomes "the height direction at $l$ gets mapped partly onto the weight direction at the end."
3. **Height mostly stays height.** Because of the skip connections, whatever is in $h_l$ is carried forward unchanged to the end. So the biggest effect of nudging "height" is usually that "height" at the final layer moves too. Cross-effects like height → weight come on top of that. A Jacobian that was just the identity matrix would give you the plain **logit lens**.
4. **The fitted lens is an average.** The true Jacobian for "elephant" in this sentence is different from the one for "Paris" in another sentence. The J-Lens averages Jacobians over many tokens and prompts. So the height → weight entry in the stored matrix is the *typical* effect, not the elephant-specific one. That averaging is the part that needs data, and the part that could overfit to the fitting data.

### Why this helps the lens

<span style="color: red;">.**$J_l\, h_l$ answers: *"what will this layer-$l$ content look like once it reaches the final layer?"* The unembedding only understands final-layer vectors, so that translated vector is what gets turned into words.</span> It's a linear approximation of what the final-layer vector will look like.**
