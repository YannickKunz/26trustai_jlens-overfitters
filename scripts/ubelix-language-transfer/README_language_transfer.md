# Language-transfer evaluation: what to run on UBELIX

Everything here reuses the lenses already fitted by `slurm_fit_lenses.sh`. No Jacobian is
refitted. The only new fitting is the J++ expert weights (minutes per run).

## Files

All in `scripts/ubelix-language-transfer/`:

| File | Role | Needs GPU |
|---|---|---|
| `parallel_items.json` | French side of the bilingual benchmark (data) | |
| `build_language_evals.py` | Builds the benchmark and its fitting/held-out split | no |
| `grade_language_items.py` | Model correctness per prompt language | yes |
| `score_language_matrix.py` | Scores every lens under prompt language x target language | yes |
| `analyze_language_transfer.py` | Tables with bootstrap intervals | no |
| `slurm_language_transfer.sh` | Runs the five stages | |
| `validate_parallel_items.py` | Checks `parallel_items.json` against upstream and the tokenizer | no (tokenizer only) |
| `check_french_tokens.py` | Which French words are single tokens (scoreable) | no (tokenizer only) |
| `diagnose_correctness.py` | Prints the model's next tokens on every multihop prompt, to see why it fails | yes |
| `make_random_pools.py` | Optional: random pools (requires refitting, see the end) | no |

## Run

From the repo root, same environment as the sweep (same container, `ROOT`, `MODEL`, `LAYERS`):

```
sbatch --account=gratis scripts/ubelix-language-transfer/slurm_language_transfer.sh
```

That runs `prepare grade weights score analyze` in order. Every stage skips finished work, so
a preempted job can be resubmitted as is. To run a subset:
`STAGE="score analyze" sbatch scripts/ubelix-language-transfer/slurm_language_transfer.sh`.

**Before a rerun with edited items**, delete the previous outputs *and the residual caches*:
`rm -rf $ROOT/lang $ROOT/runs/*/n1000/jpp_w* $ROOT/runs/*/n1000/residuals_weights_*.pt`.
The cache is keyed on item names, not on prompts: a changed item list makes `fit-weights`
fail, and changed prompts under the same names would silently reuse old activations.

Send back: `$ROOT/lang/analysis/summary.md`, `$ROOT/lang/ranks_dropped.csv`,
`$ROOT/lang/correctness_fr.csv`, and the job log.

## What each stage does and why

**prepare.** Builds a benchmark where a domain is a pair (prompt language, target language).
- *Parallel items* (53 multihop, of which 22 custom items with their own English side, and
  39 association): the same concept with an English prompt and
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

- *Prompt format.* A multihop prompt that ends with a space before a word answer loses that
  space, in both languages (`--keep-trailing-space` turns this off). With the space, Qwen3.5-9B
  tends to start a number ("...where Mount Fuji rises is " -> "1000 km") although it knows the
  answer; without it, it answers " Tokyo". Prompts with a number answer keep the space
  (otherwise the next token is a lone space), and prompts ending in a quote are unchanged.
  The main sweep and the standard benchmark keep the upstream prompts.

**grade.** Greedy correctness in each prompt language. The analysis keeps a parallel item only
if the model is right in both languages. The job grades with `--ignore-case`: " North" counts
for "north". The upstream, case-sensitive verdict is kept in a second column.

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

1. **Proofread `parallel_items.json`** and run `validate_parallel_items.py` (0 errors needed).
   The French prompts were machine-drafted. A prompt must not contain its own intermediate,
   and multihop prompts must end exactly where the French answer starts. Never replace a
   multi-token French word by a different concept. After edits: delete the outputs and the
   caches (see Run), then rerun.
2. **Single-token French words.** A word that is several tokens for the Qwen tokenizer cannot
   be scored and is dropped (upstream rule). `ranks_dropped.csv` lists them and the top of
   `summary.md` gives the item counts left after each filter. Eight association concepts are
   lost this way (deuil, mensonge, pauvreté, soulagement, honte, horreur, échecs, aube) and
   two multilingual words (marcher, neveu); no single-token true synonym exists.
3. **Correctness.** If the model fails many multihop prompts (`correctness_*.csv`), run
   `diagnose_correctness.py` to see what it answers instead. Remaining failures are answers
   in Chinese (北京 for Beijing) and genuine errors (Vienna for Berlin).
4. **Weight-fit size.** `runs/<run>/n1000/jpp_wfr_stamp.json` lists `fitting_item_names` and
   `train_macro_pass`. About 25 items are available against roughly 100 upstream, so these
   weights are noisier than the original ones. `jpp_wen` has the same handicap, which keeps
   the English/French comparison fair.

## Runs so far (Qwen3.5-9B, UBELIX)

| Run | Job | Items file | Scored items after filters | What was wrong |
|---|---|---|---|---|
| 1 | 22199941 | original (77 parallel) | 29 parallel + 11 multilingual | 17 French words not single tokens; few multihop items |
| 2 | 22218714 | revised (92 parallel) | 31 parallel + 11 multilingual | trailing-space prompts and case-sensitive grading failed most multihop items |
| 3 | 22235951 | revised, new prompt format, case-insensitive grading | 40 parallel (20 association, 20 multihop) + 11 multilingual | (current results) |

Run 3 is the one the paper uses. Outputs are not committed: the job writes them to `$ROOT/lang/`, and `make_gap_slide.py` and `make_p_matrices.py` draw the slides and the full P matrices from `$ROOT/lang/analysis/`.

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
