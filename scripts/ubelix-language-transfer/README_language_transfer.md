# Language-transfer evaluation: what to run on UBELIX

Everything here reuses the lenses already fitted by `slurm_fit_lenses.sh`. No Jacobian is
refitted. The only new fitting is the J++ expert weights (minutes per run).

## Install

Copy these files into `scripts/ubelix/` of the repo:

| File | Role | Needs GPU |
|---|---|---|
| `parallel_items.json` | French side of the bilingual benchmark (data) | |
| `build_language_evals.py` | Builds the benchmark and its fitting/held-out split | no |
| `grade_language_items.py` | Model correctness per prompt language | yes |
| `score_language_matrix.py` | Scores every lens under prompt language x target language | yes |
| `analyze_language_transfer.py` | Tables with bootstrap intervals | no |
| `slurm_language_transfer.sh` | Runs the five stages | |
| `make_random_pools.py` | Optional: random pools (requires refitting, see the end) | no |

## Run

From the repo root, same environment as the sweep (same container, `ROOT`, `MODEL`, `LAYERS`):

```
sbatch scripts/ubelix/slurm_language_transfer.sh
```

That runs `prepare grade weights score analyze` in order. Every stage skips finished work, so
a preempted job can be resubmitted as is. To run a subset:
`STAGE="score analyze" sbatch scripts/ubelix/slurm_language_transfer.sh`.

Send back: `$ROOT/lang/analysis/summary.md`, `$ROOT/lang/ranks_dropped.csv`,
`$ROOT/lang/correctness_fr.csv`, and the job log.

## What each stage does and why

**prepare.** Builds a benchmark where a domain is a pair (prompt language, target language).
- *Parallel items* (38 multihop, 39 association): the same concept with an English prompt and
  a French prompt, and the hidden intermediate as an English word and as a French word
  (Brazil / Brésil, grief / deuil). Each item is scored under four conditions.
- *French multilingual items* (the 17 upstream French items): French prompt only, scored with
  the original English intermediates and with French ones. This reproduces the old "FR text"
  column and adds the missing French-target column.
- Items were kept only if the two words differ and are not ambiguous across languages
  (dropped for that reason: Mars/mars, gold/or, east/est, names such as Einstein). Rhyme,
  first-letter, typo and poetry items are tied to English spelling and were not translated.
- The split is per concept (35% fitting), so an item is on the same side in both languages.
  Items the existing `jpp.pt` lenses were weighted on are forced into the fitting side, so the
  existing lenses are scored out-of-sample.

**grade.** Greedy correctness in each prompt language. The analysis keeps a parallel item only
if the model is right in both languages.

**weights.** For every run, two new J++ lenses from the same expert Jacobians:
`jpp_wen.pt` (weights fitted on English prompts with English targets) and `jpp_wfr.pt` (same
concepts, French prompts with French targets). The merged experts were deleted by the sweep,
so they are merged again from the shard checkpoints (`runs/<run>/rlens/`), then removed.
This is the fix for the label leak: until now every J++ lens, including the French ones, had
its weights tuned on English labels.

**score.** One CSV with the rank of every intermediate, per layer, for J, R, the three J++
variants and the Logit Lens, on all six runs. Readout Filtering as in the sweep.

**analyze.** `summary.md` holds, for the parallel items, for the French multilingual items,
and for each intermediate position of the latter (language name, category, operand, answer):
1. Recall@10 per lens, fitting corpus and condition, with 95% intervals.
2. Paired differences: language (EN fit - FR fit), corpus control (WikiText fit - EN fit),
   and fitting noise (pool A - pool B). A language effect counts only if its interval
   excludes 0 and it is clearly larger than the corpus and noise rows.
3. Target-language specificity: does each lens read its own language's words better?
4. The draft's G on true diagonals (prompts and targets in the lens's language), and the H1
   comparison G(lens) - G(J) with an interval. `jpp_matched` (weights in the language of the
   lens's own corpus) is the J++ row to use for H1; `jpp_wen` and `jpp` are ablations.

## How to read the result

- **Specificity clearly positive, G positive on both diagonals:** lenses are tied to the
  vocabulary they were fitted on. The old "English fit wins on French" cell was English
  targets, nothing more.
- **English fit still ahead with French prompts and French targets:** the stronger claim,
  an English-aligned workspace or noisier French Jacobians. Check the Logit Lens rows first:
  they show which language the model itself surfaces, with no fitted lens involved.
- **Language-name position carries the French-multilingual gap:** confirms that the single
  repeated word "French" drove the old number.
- **`jpp` or `jpp_wen` has a smaller gap than `jpp_matched`:** the apparent robustness of
  J++ came from English supervision.

## Check before trusting the numbers

1. **Proofread `parallel_items.json`.** The French prompts were machine-drafted. A prompt
   must not contain its own intermediate, and multihop prompts must end exactly where the
   French answer starts. After edits: delete `$ROOT/lang` and `runs/*/n1000/jpp_w*.pt`, rerun.
2. **Single-token French words.** A word that is several tokens for the Qwen tokenizer cannot
   be scored and is dropped (upstream rule). `ranks_dropped.csv` lists them and the top of
   `summary.md` gives the item counts left after each filter. I could not load the tokenizer
   here, so this rate is unknown. If many are lost, replace those words with shorter synonyms.
3. **French correctness.** If the model fails many French multihop prompts
   (`correctness_fr.csv`), the phrasing is the first suspect.
4. **Weight-fit size.** `runs/<run>/n1000/jpp_wfr_stamp.json` lists `fitting_item_names` and
   `train_macro_pass`. About 26 items are available against roughly 100 upstream, so these
   weights are noisier than the original ones. `jpp_wen` has the same handicap, which keeps
   the English/French comparison fair.

## What was tested, and what was not

Tested here on CPU: the builder against the pinned upstream data; the grader and the scorer
end to end on upstream's tiny test model; the analysis on synthetic scores with a planted
effect, which it recovered; the upstream loader on the weight-fitting folders.
Not tested: anything on the real model or tokenizer, the `fit-weights` and `merge-experts`
calls, and the Slurm script beyond a syntax check.

## Two things found while reading the code

- `compute_P.py` labels every non-multilingual item "EN", but the association task contains
  nine non-English prompts (`fr-ecole`, `de-konig`, `es-dia`, ...). They were counted in the
  English column. The new analysis replaces that script.
- Readout Filtering is not English-centric: it removes only tokens with no letter or digit,
  so it treats both languages alike. No change needed.

## Optional: random pools

`make_random_pools.py` samples pools A, B and a probe set at random instead of taking the
first articles in dataset order. Using it means refitting routers, shards and lenses for
every corpus, the full GPU cost of the sweep, under a new `ROOT`. It tightens the noise
floor argument but is not needed for the language question. Example:

```
python scripts/ubelix/make_random_pools.py --dataset wikimedia/wikipedia --config 20231101.fr \
    --name fr --out-dir $ROOT_V2/pools
```
