# J-lens / R-lens / J++ lens: released-lens reproduction (Qwen3.5-9B)

Descriptive evaluation of three already-released lenses (J, R, J++) plus a logit-lens baseline
on 5 readout tasks, metric recall@10. Not a certified held-out replication of the paper tables
(see "Limits"). No lens is fitted here.

## What this folder adds
| Path | What it is |
|---|---|
| `scripts/reproduce_3070ti.py` | The evaluator. Downloads the released J, R and J++ lenses, runs Qwen3.5-9B (4-bit NF4, 8 GB GPU), ranks the hidden intermediate concept at every layer, writes the JSON. Also runs one exploratory arithmetic 8<->9 swap. The docstring says which code comes from which repo. |
| `scripts/summarize_reproduction.py` | Turns the JSON into the markdown tables. |
| `results/corrected_3070ti.json` | Raw output: per-item, per-layer ranks, provenance (24 MB). |
| `results/summarize_reproduction.md` | Result tables, denominators, comparison limits, pinned revisions. |
| `results/results_summary.ipynb` | Same numbers as the deck, with charts, gains, and the comparison to the J++ paper (Qwen3.5-9B macro, Qwen3.6-27B per task). Needs only pandas + matplotlib. |

## Setup (not included)
`scripts/` imports the upstream J++ repo from `jpp_lens/` next to it:
```bash
git clone https://github.com/safety-research/jpp_lens.git jpp_lens
git -C jpp_lens checkout d7efc0de372e20589e899188c0cfe8434acc63ec   # commit used for the results
```
Python >= 3.12 and a CUDA GPU. Install the upstream pinned deps, then the two extras the script needs:
```bash
cd jpp_lens && uv sync --extra cu128 && cd ..      # torch (CUDA 12.8), transformers, pandas, ...
uv pip install bitsandbytes accelerate               # 4-bit loading
```
Needs roughly 7.2 GiB of free VRAM (peak 7.19 GiB on our run), the ~18 GB Qwen3.5-9B download, and the three lens files from Hugging Face.

## Run
```bash
python scripts/reproduce_3070ti.py --evals multihop multilingual typo association poetry \
  --arithmetic --out results/corrected_3070ti.json
```

## Protocol (summary)
- Score the **intermediate** concept (e.g. *Brazil*), not the final answer; keep every genuine single-token surface, never a multi-token concept's first subtoken; dropped concepts are recorded.
- Original prompts verbatim; readout at the final prompt token (last newline for poetry).
- Item-weighted min-over-layers pass@1/@10 over all 31 layers, and over seven fixed depths (4, 8, 12, 16, 20, 24, 28); the seven-layer result is also reported with this model's own greedy-correctness filter.
- J and R use the full vocabulary. J++ uses Readout Filtering (upstream's rule). `J+readout_only` is a separate ablation, never called J++.
- Immutable HF revisions of model and lenses are recorded in the JSON. Don't pass `--max-items` for a benchmark run (smoke tests only).

## Using another model
Changing `--model` is not enough. The script refuses anything other than `Qwen/Qwen3.5-9B` and
`Qwen/Qwen3.6-27B`, because it needs a released J, R and J++ lens file for the exact model
(hard-coded paths in `main()`), a CUDA GPU, and a model that `jlens.from_hf` supports.
A new model needs those three lens files first (or lenses you fit yourself).

## Limits
- Quantized 9B, and the released lenses have different fitting budgets: not a table replication.
- Released J++ mixer training membership is unknown: scores are not certified held-out.
- No significance tests; small denominators (e.g. multihop has 84 scoreable items).
- The arithmetic intervention is one problem with a supplied CoT: inconclusive.

## Papers
- J-lens: Gurnee et al. 2026, arXiv 2607.15495
- J++ lens: Ayonrinde & Lindsey 2026
