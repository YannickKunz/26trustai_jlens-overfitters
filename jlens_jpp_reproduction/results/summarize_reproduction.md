# Corrected released-lens evaluation

- Model: `Qwen/Qwen3.5-9B`; precision: `NF4-double/fp16`.
- GPU: NVIDIA GeForce RTX 3070 Ti; peak allocated CUDA memory: 7.19 GiB.
- Started: 2026-10-19T15:48:59.260389+00:00; finished: 2026-10-19T16:02:55.668632+00:00.
- 496 raw prompts; layers [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]; paper-style layers [4, 8, 12, 16, 20, 24, 28].
- Readout filter excluded 6133 token IDs.
- **Descriptive evaluation of released lenses, not certified held-out paper-table replication.**

## Denominators

| Task | Raw prompts | Scoreable | Correct and scoreable | Dropped concepts |
|---|---:|---:|---:|---:|
| multihop | 93 | 84 | 32 | 9 |
| multilingual | 107 | 107 | 72 | 2 |
| typo | 96 | 96 | 96 | 0 |
| association | 102 | 102 | 102 | 0 |
| poetry | 98 | 98 | 74 | 0 |

Association and typo have no correctness target; their scoreable items are retained.

## Recall@10 — all_source_layers_unfiltered

| Task | Logit | J | R | J + readout only | J++ |
|---|---:|---:|---:|---:|---:|
| multihop | 47.62% | 60.12% | 54.56% | 62.30% | 73.02% |
| multilingual | 37.77% | 48.29% | 50.70% | 55.30% | 72.82% |
| typo | 58.33% | 59.38% | 76.04% | 69.79% | 97.92% |
| association | 0.00% | 3.92% | 3.92% | 9.80% | 32.35% |
| poetry | 1.02% | 3.06% | 2.04% | 3.06% | 11.22% |
| macro | 28.95% | 34.95% | 37.45% | 40.05% | 57.47% |

## Recall@10 — seven_layers_unfiltered

| Task | Logit | J | R | J + readout only | J++ |
|---|---:|---:|---:|---:|---:|
| multihop | 38.69% | 47.02% | 45.24% | 52.98% | 61.71% |
| multilingual | 30.45% | 42.68% | 46.73% | 50.86% | 65.03% |
| typo | 43.75% | 45.83% | 60.42% | 53.12% | 93.75% |
| association | 0.00% | 2.94% | 3.92% | 7.84% | 28.43% |
| poetry | 0.00% | 2.04% | 1.02% | 3.06% | 7.14% |
| macro | 22.58% | 28.10% | 31.47% | 33.57% | 51.21% |

## Recall@10 — seven_layers_correctness_filtered

| Task | Logit | J | R | J + readout only | J++ |
|---|---:|---:|---:|---:|---:|
| multihop | 73.44% | 70.31% | 75.00% | 71.88% | 79.69% |
| multilingual | 31.02% | 43.98% | 46.18% | 51.97% | 67.13% |
| typo | 43.75% | 45.83% | 60.42% | 53.12% | 93.75% |
| association | 0.00% | 2.94% | 3.92% | 7.84% | 28.43% |
| poetry | 0.00% | 2.70% | 1.35% | 4.05% | 9.46% |
| macro | 29.64% | 33.15% | 37.37% | 37.77% | 55.69% |

## Exploratory arithmetic intervention

Supplied (not generated) CoT. Single-layer, final-position J++ 8↔9 coordinate swaps; not a paper causal benchmark.

| Arm | Layer | Baseline P(8) | Swapped P(8) | Baseline P(9) | Swapped P(9) |
|---|---:|---:|---:|---:|---:|
| Silent | 4 | 0.000003 | 0.000003 | 0.000005 | 0.000006 |
| Silent | 8 | 0.000003 | 0.000003 | 0.000005 | 0.000006 |
| Silent | 12 | 0.000003 | 0.000003 | 0.000005 | 0.000006 |
| Silent | 16 | 0.000003 | 0.000003 | 0.000005 | 0.000005 |
| Silent | 20 | 0.000003 | 0.000003 | 0.000005 | 0.000005 |
| Silent | 24 | 0.000003 | 0.000003 | 0.000005 | 0.000005 |
| Silent | 28 | 0.000003 | 0.000007 | 0.000005 | 0.000003 |
| Supplied-CoT | 4 | 0.000043 | 0.000043 | 0.000000 | 0.000000 |
| Supplied-CoT | 8 | 0.000043 | 0.000044 | 0.000000 | 0.000000 |
| Supplied-CoT | 12 | 0.000043 | 0.000043 | 0.000000 | 0.000000 |
| Supplied-CoT | 16 | 0.000043 | 0.000043 | 0.000000 | 0.000000 |
| Supplied-CoT | 20 | 0.000043 | 0.000043 | 0.000000 | 0.000000 |
| Supplied-CoT | 24 | 0.000043 | 0.000042 | 0.000000 | 0.000001 |
| Supplied-CoT | 28 | 0.000043 | 0.000021 | 0.000000 | 0.000013 |
| Given | 4 | 0.000072 | 0.000073 | 0.000018 | 0.000018 |
| Given | 8 | 0.000072 | 0.000073 | 0.000018 | 0.000019 |
| Given | 12 | 0.000072 | 0.000072 | 0.000018 | 0.000019 |
| Given | 16 | 0.000072 | 0.000071 | 0.000018 | 0.000018 |
| Given | 20 | 0.000072 | 0.000069 | 0.000018 | 0.000019 |
| Given | 24 | 0.000072 | 0.000050 | 0.000018 | 0.000031 |
| Given | 28 | 0.000072 | 0.000057 | 0.000018 | 0.000025 |

Maximum identity-control probability drift: 0.00000000.
Raw JSON retains ranks, token IDs and coordinates. One arithmetic problem is not enough to establish a causal effect.

## Comparison limits

- Not a paper-table replication: quantization/model/baseline fitting budgets differ.
- Released J++ mixer fitting membership unavailable; scores are NOT certified held-out.
- Original J filename says Base; checkpoint has no embedded model identity to verify.
- Original all-layer convention and J++ seven-layer convention both reported.
- R-Lens is a separately fitted released baseline, with different corpus/budget/transport target. Inspect `lens_configs` before comparing it to a matched-budget paper row.
- No significance test or evidence of generalization follows from these descriptive point estimates.
- The old `results_3070ti.json` scores answer-token first subtokens and is not comparable to these bridge-concept scores.

## Provenance

J++ code commit: `d7efc0de372e20589e899188c0cfe8434acc63ec`.
Model revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
Runner SHA-256: `eb07fdd3a2a28ecc9d2e77f97b785f390f1d12829247b5608b1e76a780d4278e`.

- J: `neuronpedia/jacobian-lens` @ `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a`, `qwen3.5-9b-pt/jlens/Salesforce-wikitext/Qwen3.5-9B-Base_jacobian_lens.pt`.
- R: `camilablank/workspace-lenses` @ `d740106d1e0f95456dc8718fba2895e9c8ffd6ef`, `qwen3.5-9b/r-lens/lens.pt`.
- J++: `koayon/jpp-lenses` @ `237dab6ea0e611d964bf407f13ba1655f1a6ae43`, `qwen3.5-9b/lens.pt`.

Package versions, dataset hashes, fitting configurations and per-item/per-layer ranks are embedded in the JSON.
