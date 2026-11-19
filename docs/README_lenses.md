# Trained lenses: Qwen3.6-27B and Qwen3.5-9B

Download (Google Drive, view only): https://drive.google.com/drive/folders/1tplAOFWpquRA9qhkYCRSIDD6WsLmG741?usp=sharing

Fitted on UBELIX with the J++ paper's pipeline (`jpp_cli.py`, jpp_lens commit `d7efc0de`), in its locked environment (torch 2.11.0, transformers 5.14.1).

## Folders

`<model>/<run>/[n<prompts>/]<lens>.pt`

| Lens | How it is fitted |
|---|---|
| `j.pt` | J-Lens: standard gradients, one pooled Jacobian per layer |
| `r.pt` | R-Lens: LRP backward pass, one pooled Jacobian per layer |
| `jpp.pt` | J++ Lens: same LRP fit as R, 8 experts per layer combined with learned weights |
| `jpp_wen.pt`, `jpp_wfr.pt` | The same J++ experts, with expert weights refitted on the bilingual benchmark: English items (`wen`) or French items (`wfr`). Only for the 1,000-prompt 9B lenses and the 64-prompt 27B lenses. "Matched weights" in the paper: `jpp_wen` for WikiText and English Wikipedia, `jpp_wfr` for French Wikipedia |

R and J++ come from the same fit, so they are not independent.

| Run | Prompts |
|---|---|
| `wikitext-64` | First 64 WikiText-103 prompts: the paper's recipe |
| `en-wiki-64`, `fr-wiki-64` | First 64 English or French Wikipedia (`20231101`) articles |
| `wikitext-32-half-a`, `-half-b` | WikiText prompts 0-31 and 32-63: a first noise floor |
| `<dataset>-a/n<size>` | Prompts 0-999 of that dataset; size n uses the first n |
| `<dataset>-b/n<size>` | Prompts 1000-1999 |

`wikitext`, `en` and `fr` are WikiText-103, English Wikipedia and French Wikipedia. A prompt is a record of at least 600 characters, in dataset order, cut to 128 tokens.

## Fitting recipe

- Source positions: tokens 16 to 127.
- Target: the final block.
- Router: k-means with 8 clusters on 1,000 prompts of the same pool.
- J++ expert weights: the labelled English evaluation items of `fit:0` (80 on 27B, 75 on 9B), for every dataset.

## Layers

Only these layers are fitted, not every layer:

- Qwen3.6-27B: 8, 16, 24, 32, 40, 48, 56
- Qwen3.5-9B: 4, 8, 12, 16, 20, 24, 28

## Format

Each file is a `torch.save` dictionary with `config`, `source_layers` and `parameters["jacobians"]` (`{layer: fp32 [d_model, d_model]}`), loadable with `torch.load(path, weights_only=True)`.

With jpp_lens on the path, `workspace_lens.utils.load_lens_file(path, hf_model_name=...)` returns a lens object.

Readout at layer l for a residual h: `logits = unembed(h @ J[l].T)`, where `unembed` is the final norm followed by the LM head. The paper ranks J++ with Readout Filtering, which leaves out tokens with no letter or digit.

`*_stamp.json` records each stage's arguments, the git commit and the prompt counts. `jpp_expert_weights.json` holds the learned J++ weights.
