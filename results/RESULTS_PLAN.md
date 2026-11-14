# Lens generalisation: results and how to read them

What was run, what came out, how to read it, and what is in the results zip.
All runs finished on 2026-11-05. The lenses are on Google Drive under `Trained Lenses/`.

## 1. Questions

From `docs/research_question.md`:

1. **RQ1:** How can we measure whether a lens fitted on one corpus is close to one fitted on another?
2. **RQ2:** Does the J++ Lens generalise across language (English/French) and domain (general text vs mathematics)? How does it compare to the J-Lens?
3. **H1:** The J++ Lens is less generalisable than the J-Lens.
4. **Side question:** How many fitting prompts does a lens need before it converges?

Only the language half of RQ2 is covered here. Mathematics is not fitted yet.

## 2. What was run

### Lenses

| Lens | How it is fitted |
|---|---|
| J | Standard gradients, one pooled Jacobian per layer. Unsupervised baseline. |
| R | LRP backward pass, one pooled Jacobian per layer. Isolates the LRP contribution. |
| J++ | The same LRP fit as R, 8 experts per layer combined with learned signed weights. |
| Logit | No fitting. It must not change across datasets, which makes it a check. |

R and J++ come from the same fit, so they are not independent.

### Recipe, identical for every fit

Upstream `jpp_cli.py` at commit `d7efc0de`, in the paper's locked environment (`jlens_paper.sif`: torch 2.11.0, transformers 5.14.1).

- **Prompts:** the first records of at least 600 characters, in dataset order, cut to 128 tokens.
- **Positions:** tokens 16 to 127 are sources.
- **Target:** the final block.
- **Router:** k-means with 8 clusters, on 1,000 prompts of the same pool.
- **J++ weights:** fitted on the labelled English `fit:0` items (80 on 27B, 75 on 9B), for every dataset.
- **Scoring:** recall@10 on the held-out items (`held-out:0`), never overlapping the 80 items above.

### Runs

| Model | Data | Prompts | Pools | Layers |
|---|---|---|---|---|
| Qwen3.6-27B | WikiText, English and French Wikipedia | 64 | One | 8, 16, ..., 56 |
| Qwen3.6-27B | WikiText, split in two | 32 + 32 | - | 8, 16, ..., 56 |
| Qwen3.5-9B | WikiText (paper recipe) | 64 | One | 4, 8, ..., 28 |
| **Qwen3.5-9B** | **WikiText, English and French Wikipedia** | **100, 200, 500, 1000** | **A and B** | **4, 8, ..., 28** |

The bold row is the main experiment, with 72 lenses. Pool A uses prompts 0-999 and pool B prompts 1000-1999 of each dataset, each with its own router. Sizes are nested within a pool: 100 is the first tenth of 1000.

The 9B scores use our own correctness grading, because the shipped file covers 27B only. That leaves 384 scoreable items: 75 tune J++ and 309 are scored.

### Comparisons

`qwen3.5-9b/scores/compare.csv`: 195 pairs, 7 layers each, 4 sample sets.

| Kind | Pair | What it measures |
|---|---|---|
| `noise-<dataset>` | Pool A vs B, same dataset | Noise floor: two fits on disjoint data |
| `language-a`, `-b` | English vs French Wikipedia | Language effect, measured twice |
| `corpus-a`, `-b` | WikiText vs English Wikipedia | Corpus effect, measured twice |
| `mixed-a` | WikiText vs French Wikipedia | Both effects together |
| `converge-<pool>` | Size n vs 1000, same pool | Convergence (nested, not independent) |
| `method-<pool>` | J vs J++, R vs J++, J vs R | Cross-method grid |

KL sample sets:

| Name | Size | Content |
|---|---|---|
| `eval` | 384 | Labelled evaluation readouts, mostly English |
| `wikitext`, `en`, `fr` | 300 each | Unseen prompts of each dataset, last of 128 tokens |

## 3. How to read the metrics

- **Recall@10:** how often the lens finds the hidden intermediate concept in its top 10 at any of the seven layers. Higher is better. The items are English, so this measures usefulness on these tasks, not general quality.
- **Cosine:** similarity of two Jacobians as flattened vectors. 1 means the same direction.
- **Relative Frobenius:** `||A - B|| / mean(||A||, ||B||)`. 0 means identical. Unlike cosine, it also reacts to scale.
- **Symmetric KL:** the mean of KL(A‖B) and KL(B‖A) between the two lenses' token distributions on real text. Lower means more alike.
- **KL to the model:** KL from the model's own next-token distribution to the lens's readout. Lower means the lens predicts more like the model. This metric does not depend on J++'s tuning items.

**Main rule:** a distance only counts when it exceeds the noise floor at the same layer, size and sample set. `noise_test.csv` applies this conservatively: an effect pair counts at a layer only if its cosine is below, and its KL above, the least similar of the relevant noise pairs. For language that is `noise-en` and `noise-fr`.

## 4. Results

### Reproduction of the paper

| Lens | 27B ours | 27B paper | 9B ours | 9B paper |
|---|---:|---:|---:|---:|
| J++ | 54.51 | 55.2 | 53.48 | 55.0 |
| R | 39.10 | 37.7 | 37.08 | - |
| J | 30.83 | 35.7 | 33.96 | - |

Paper protocol: 64 WikiText prompts, J and R unfiltered. J++ and R reproduce. J on 27B is 4.9 points low, just outside its 95% interval, for a reason we have not found.

The released J++ lenses score about 4 points above their published values on our held-out items (59.68 on 27B, 58.85 on 9B), possibly because they were fitted on a different split.

### Recall@10, 9B, 1000 prompts, Readout Filtering on every lens

| Lens | WikiText | English Wikipedia | French Wikipedia | English to French |
|---|---:|---:|---:|---:|
| J++ | 56.5 | 57.9 | 48.2 | -9.7 (-17%) |
| R | 43.2 | 37.7 | 30.6 | -7.1 (-19%) |
| J | 38.0 | 37.7 | 30.5 | -7.2 (-19%) |
| Logit | 32.8 | 32.8 | 32.8 | 0 |

Means of pools A and B. The two pools differ by at most 1.6 points. Per-run 95% intervals are about ±4 points (`recall_ci.csv`).

### Distances, 9B, 1000 prompts, `eval` samples

| Lens | Noise cosine | Corpus cosine | Language cosine | Noise KL | Corpus KL | Language KL |
|---|---:|---:|---:|---:|---:|---:|
| J | 0.995 | 0.895 | 0.834 | 0.03 | 1.98 | 1.91 |
| R | 0.998 | 0.939 | 0.890 | 0.01 | 1.13 | 0.97 |
| J++ | 0.795 | 0.682 | 0.707 | 3.12 | 4.78 | 3.54 |

On the English and French text sets, language KL is above corpus KL for J and R: for R it is 2.77 vs 1.01 on English text.

Layers above the noise floor, out of 14 (7 layers × 2 pairs):

| Lens | Language | Corpus |
|---|---:|---:|
| J | 14 at every size | 14 at every size |
| R | 14 at every size | 14 at every size |
| J++ | 6-10 | 4-11 |

### Convergence

- **Recall:** flat from 100 to 1000 prompts for every lens and dataset: within about 1 point for J and R, and within 3 points for J++, with no upward trend.
- **J and R matrices:** cosine to the 1000-prompt lens is ≥ 0.983 at 100 prompts and ≥ 0.998 at 500. The noise floor tightens from 0.95-0.996 at 100 prompts to 0.994-0.998 at 1000.
- **J++ matrices:** cosine to the 1000-prompt lens stays at 0.88-0.98, and the J++ noise floor does not tighten with more prompts (0.69-0.89).

### Other

- **Cross-method cosine (1000 prompts):** J–R 0.88-0.91, R–J++ 0.68-0.76, J–J++ 0.58-0.67.
- **KL to the model (1000 prompts, mean over layers):** R 5.6-6.7, J 7.8-9.1, J++ 7.5-9.4.

### 27B, 64 prompts (one pool)

Recall@10, paper protocol:

| Lens | WikiText | English Wikipedia | French Wikipedia |
|---|---:|---:|---:|
| J++ | 54.51 | 54.39 | 44.30 |
| R | 39.10 | 34.99 | 21.26 |
| J | 30.83 | 28.98 | 19.34 |

English vs French Wikipedia cosine: J 0.68, R 0.82, J++ 0.54, against a 32-prompt noise floor of 0.80, 0.94 and 0.80.

### 27B with Readout Filtering on J and R (64 prompts)

| Lens | Paper | Ours, unfiltered | Ours, filtered |
|---|---:|---:|---:|
| Logit | 31.1 | 31.6 | 34.2 |
| J | 35.7 | 30.8 | 39.0 |
| R | 37.7 | 39.1 | 44.3 |

Logit and R match the paper unfiltered, as on 9B, so the paper does not filter J and R. Filtering lifts our J's typo score from 42.9 to 63.6 (paper 60.4): our 27B J ranks non-letter tokens highly on typo items. That is where its gap comes from, but not why.

### French prompts with French or English targets (pilot, 9B, 1000 prompts)

The `multilingual` items with French prompts all have English targets ("French", "color", "night", "black"). Scoring the same 11 items again with French targets ("français", "couleur", "nuit", "noir"), recall@10 over all target words, mean of pools A and B:

| Lens | Fit | English targets | French targets |
|---|---|---:|---:|
| J | English Wikipedia | 56.8 | 29.5 |
| J | French Wikipedia | 30.7 | 45.5 |
| R | English Wikipedia | 59.1 | 29.5 |
| R | French Wikipedia | 40.9 | 44.3 |
| J++ | English Wikipedia | 78.4 | 30.7 |
| J++ | French Wikipedia | 61.4 | 34.1 |

With French targets the French-fitted J and R win, as in-domain lenses should. The "FR text" column of the November 10 slides measured English targets, not French text. The answer word alone is near ceiling for every lens (80-100%), so the difference comes from the other target words. Only 11 items: this is a pilot, not a result. J++ is English-tuned in both rows. Ranks: `qwen3.5-9b/fr_pilot/ranks.csv`; script: `scripts/fr_target_pilot.py`.

## 5. What the results say

1. **RQ1, metrics:** all three distance measures separate the effects from noise for J and R. KL on real text is the easiest to interpret. Cosine and relative Frobenius agree on the ordering.
2. **Language has a real cost:** fitting on French Wikipedia lowers recall on these English tasks by about 7 points for J and R, and 10 points for J++. French-fitted J and R fall below the Logit lens. French-fitted J++ is still above every English-fitted J and R.
3. **H1 is not supported by recall:** J++ loses slightly more points, but slightly less in relative terms (17% vs 19%), and it stays far ahead.
4. **J++ is unstable as a matrix:** two J++ lenses fitted on disjoint prompts from the same corpus differ almost as much as English vs French lenses. More prompts do not fix this. Its recall is stable, though, so many different J++ matrices work equally well. A likely cause is the expert weights, fitted on only 75 items. For J++, distances across corpora therefore cannot be separated from refit noise; recall is the more reliable comparison.
5. **Corpus also matters:** WikiText vs English Wikipedia moves the J and R matrices clearly above noise, but less than language does. In recall it only costs R (-5.5 points).
6. **Few prompts are enough:** 100 prompts already give the 1000-prompt recall, and J and R matrices are essentially converged by 500. The paper's 64 prompts look sufficient for recall.
7. **Model size:** 27B shows the same direction and a larger French drop, but with one pool and 64 prompts only.

## 6. Limits to state with the results

- **English items only:** every recall@10 is on English (plus some multilingual) items, so FR-text, EN-maths and FR-maths columns of matrix P are missing. KL on French text is the only French-side measure.
- **J++ weights are English-tuned for every dataset:** French J++ is French-fitted but English-tuned.
- **Two pools per setting:** they show the spread, not strong statistics.
- **Bootstrap intervals** cover item sampling only, not fit-to-fit variation.
- **Prompts follow dataset order,** so topics can cluster.
- **Seven layers only:** the lenses cover the scored layers, not every layer.
- **Maths, code and `probe-swap` were not run.**
- **The 27B J baseline is 4.9 points below the paper,** unexplained.

## 7. Fit vs evaluation matrix

| Fit \ Eval | EN text | FR text | EN maths | FR maths |
|---|---|---|---|---|
| EN text | Done | No eval set | No eval set | No eval set |
| FR text | Done | No eval set | No eval set | No eval set |
| EN maths | Not fitted | - | - | - |
| FR maths | Not fitted | - | - | - |

New columns need labelled items, not refits: scoring the saved lenses on new items takes minutes. Maths needs a scoring design, because Qwen splits multi-digit numbers into digits and single-token candidates are required.

## 8. Results zip

`jlens_results.zip`, on UBELIX at `/storage/scratch/users/yk20q078/`:

| Path | Content |
|---|---|
| `RESULTS_PLAN.md` | This file |
| `LENSES.md` | What is on Google Drive and how to load it |
| `analysis_9b/` | `summary_tables.md` and tidy CSVs: `recall.csv`, `recall_ci.csv`, `distances.csv`, `distance_summary.csv`, `noise_test.csv` |
| `reports/` | 27B reproduction reports |
| `qwen3.5-9b/runs/<run>/n<size>/` | Per-item ranks (`*.csv`), recall tables (`*_pass_at_k.csv`), stamps, J++ weights |
| `qwen3.5-9b/scores/` | `compare.csv` (all distances) and the pair list |
| `qwen3.5-9b/` | Correctness file, released J++ scores, router stamps |
| `qwen3.5-9b/prompts/` | Every prompt pool (A, B and probe sets) |
| `qwen3.6-27b/` | Paper refit, datasets, noise test, English vs French, first evaluator's JSON |
| `scripts/` | Every script, Apptainer definition and job script |

Lens weights are not in the zip; they are on Drive. Shard checkpoints (about 500 GB) stay on UBELIX scratch.

`scripts/analyze_results.py --root <9b folder> --out <folder>` regenerates `analysis_9b/` from the raw scores.

## 9. Prompt for a presentation

Unzip the results, open Claude Code in that folder, and paste:

```
You are making slides for a 15-minute presentation to our TrustAI seminar group on whether
workspace lenses (J-Lens, R-Lens, J++ Lens) generalise across fitting corpora.

Sources, in this folder:
- RESULTS_PLAN.md: the questions, setup, results, interpretation and limits. Follow its
  interpretation; do not add claims it does not make.
- analysis_9b/: tidy CSVs and summary_tables.md. Take every number from these files, never
  from memory, and round as the tables do.
- reports/PAPER_REPRODUCTION_27B.md: the reproduction details.

Build:
1. A figures/ folder of PNG figures made with matplotlib, colour-blind safe, readable at slide
   size, one lens per colour used consistently (J, R, J++, Logit):
   a. Recall@10 by fitting dataset and lens at 1000 prompts, with the 95% intervals from
      recall_ci.csv (both pools shown).
   b. Recall@10 against prompts (100, 200, 500, 1000; log x-axis), one panel per lens,
      one line per dataset.
   c. Cosine by layer at 1000 prompts for noise, corpus and language pairs, one panel per lens.
   d. The same as c for symmetric KL on the en and fr sample sets.
   e. Noise floor (noise pairs' cosine) against prompts, one line per lens.
   f. Cross-method cosine grid (J, R, J++) as a small heatmap.
2. slides.pptx made with python-pptx, about 12 slides:
   title; question and hypothesis H1; the three lenses in one sentence each; setup (models,
   datasets, pools A/B, sizes, recipe); reproduction table; recall by dataset; language vs
   corpus distances with the noise-floor rule; the J++ instability finding; convergence;
   answer to H1; limits; next steps (maths, French evaluation items, 27B sweep).
   One message per slide as its title, at most 4 short bullets, speaker notes with the
   numbers and caveats for that slide.
3. Check every number on the slides against the CSVs and list any you could not find.

Install matplotlib, pandas and python-pptx if missing. Keep the code in make_slides.py so the
deck can be rebuilt.
```
