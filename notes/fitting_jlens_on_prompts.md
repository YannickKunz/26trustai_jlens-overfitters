# Background knowledge

> **Disclaimer:** Claude is generating / helping me get answers.

## Fitting the J-Lens on prompts

### Question

> How is the Jacobian fitted on the prompts? (training)

### Answer

The short answer is that **it isn't trained**. There's no loss function, no gradient descent and no optimizer. The Jacobian is *measured* on each prompt with backpropagation, and those measurements are *averaged*. That's all "fitting" means here.

I read the actual code in your local `jacobian-lens` checkout (`jlens/fitting.py`). Here's what it does.

#### For each prompt (`jacobian_for_prompt`)

1. **Truncate and run forward.** Cut the prompt to 128 tokens and run the model forward once, recording the residual at each source layer and at the final layer.
2. **Pick which tokens count.** Tokens 0–15 are skipped because early positions act as "attention sinks" and have unusual residuals. The last token is skipped too.
3. **Build the matrix one row at a time.** Row $i$ of $J_l$ answers: *"if I want final-layer entry $i$ to go up, how should each entry of $h_l$ change?"* That's exactly what one backward pass gives you. The code puts a 1 on final-layer dimension $i$ (at all valid tokens at once), backpropagates to layer $l$, and reads off the gradient. That gradient is row $i$.
4. **Repeat for every output dimension.** That takes `d_model` backward passes, e.g. 4096. It's batched 8 at a time, so about 512 passes per prompt. This is why fitting is expensive.
5. **Average over tokens.** Each token gives its own version of the row, and these are averaged over the valid positions. The result is one `d_model × d_model` matrix for this prompt.

#### Across prompts (`fit`)

6. **Average again.** The per-prompt matrices are added up and divided by the number of prompts. The result is $J_l$, which is what gets saved as `j.pt`.

The log even prints how much the running mean is still changing. Once the average settles, that change shrinks roughly like $1/n$. That's a direct way to see whether 64 prompts is "enough."

### A correction to the earlier answer

The earlier answer (see [num_jacobian_lenses.md](num_jacobian_lenses.md)) said it wasn't sure whether the lens includes effects on later tokens. **It does.** The 1 is placed at every valid target position at once, so the gradient arriving at token $p$ includes how $h_l[p]$ affects the final layer at $p$ *and* at every later token (through attention). The docstring says this is the paper's choice. It also says a strict same-token version gives a slightly different $J_l$, and both work as a lens.

### Summary

The full recipe: **backprop** to measure how layer $l$ feeds into the final layer, **averaged over tokens**, then **averaged over prompts**. It depends on the data only because it's an average over the prompts you chose. That's the opening for overfitting your project is testing.
