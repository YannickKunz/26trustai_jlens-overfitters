#!/usr/bin/env python3
"""Per-layer tables for H2 (is generalisation weakest in early layers?).

Three views, each per lens and source layer:
1. Parameters and distribution, from the sweep's noise_test.csv (analyze_results.py): the
   floor-normalised gaps G_S = (1 - cos) / (1 - cos_floor) and KL / KL_floor for the language
   (EN vs FR Wikipedia) and corpus (WikiText vs EN Wikipedia) pairs, 1000 prompts, eval residuals.
   G_S = 1: the change moves the lens as much as resampling the corpus.
2. Behaviour on the bilingual benchmark (score_language_matrix.py's ranks.csv): recall@k per layer
   under each condition, same item filters as analyze_language_transfer.py, mean of pools A and B.
3. Behaviour on the standard benchmark (optional, --sweep-runs): recall@k per layer per fitting
   corpus, Readout Filtering, macro over the five tasks, mean of pools.
Writes layers_*.csv and layers.md to --out. Needs pandas and numpy only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LENS_NAMES = {"logit": "Logit", "j": "J", "r": "R", "jpp": "J++", "jpp_wen": "J++ (EN weights)",
              "jpp_wfr": "J++ (FR weights)"}
MACRO_EVALS = ["multihop", "multilingual", "typo", "association", "poetry"]


def markdown(frame, digits=2):
    frame = frame.reset_index()
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:.{digits}f}"
        return str(value)
    header = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def parameter_gaps(noise_test, size):
    frame = pd.read_csv(noise_test)
    frame = frame[(frame["size"] == size) & (frame["samples"] == "eval") & frame["family"].isin(["language", "corpus"])]
    return (frame.groupby(["family", "lens", "layer"])
            .agg(distance=("cosine", lambda c: float((1 - c).mean())), floor_distance=("noise_cosine", lambda c: float((1 - c).mean())),
                 G_S=("excess_distance", "mean"), KL_ratio=("excess_kl", "mean"),
                 above_noise=("above_noise", "sum"), pairs=("above_noise", "size")).reset_index())


def bilingual_layers(ranks_path, lang, k):
    """Recall@k per layer, held-out parallel items the model answers in both languages, concepts that are
    different single-token words in both languages (the filters of analyze_language_transfer.py)."""
    ranks = pd.read_csv(ranks_path)
    records = {r["name"]: r for r in json.loads(Path(lang, "items.json").read_text())}
    ranks = ranks[(ranks["group"] == "parallel") & (ranks["split"] == "held-out")]

    def correct(language):
        frame = pd.read_csv(Path(lang, f"correctness_{language}.csv"))
        value = frame.iloc[:, 2]
        return set(frame["item"][value.isna() | (value.astype(str) == "True")])
    both = correct("en") & correct("fr")
    ranks = ranks[ranks["item"].isin(both)]
    same = [records[i]["en"]["intermediates"][p].casefold() == records[i]["fr"]["intermediates"][p].casefold()
            for i, p in zip(ranks["item"], ranks["position"])]
    ranks = ranks[~np.array(same)]
    languages = ranks.groupby(["item", "position"])["target_lang"].nunique()
    keep = set(languages[languages == 2].index)
    ranks = ranks[[(i, p) in keep for i, p in zip(ranks["item"], ranks["position"])]]

    ranks = ranks.assign(hit=(ranks["rank"] <= k).astype(float))
    per_item = ranks.groupby(["lens", "fit", "pool", "prompt_lang", "target_lang", "layer", "eval", "item"])["hit"].mean()
    per_task = per_item.groupby(["lens", "fit", "pool", "prompt_lang", "target_lang", "layer", "eval"]).mean()
    macro = per_task.groupby(["lens", "fit", "pool", "prompt_lang", "target_lang", "layer"]).mean()
    out = macro.groupby(["lens", "fit", "prompt_lang", "target_lang", "layer"]).mean().reset_index()
    out["recall"] = out.pop("hit") * 100
    return out, ranks["item"].nunique()


def sweep_layers(runs, k, size):
    rows = []
    for run in ("wikitext-a", "wikitext-b", "en-a", "en-b", "fr-a", "fr-b"):
        for stem in ("jpp_scores", "baselines_filtered"):
            path = Path(runs, run, f"n{size}", f"{stem}.csv")
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            frame["hit"] = (frame["rank"] <= k).astype(float)
            per_item = frame.groupby(["lens", "layer", "eval", "item"])["hit"].mean()
            per_task = per_item.groupby(["lens", "layer", "eval"]).mean().reset_index()
            per_task = per_task[per_task["eval"].isin(MACRO_EVALS)]
            macro = per_task.groupby(["lens", "layer"])["hit"].mean().reset_index()
            fit, pool = run.rsplit("-", 1)
            rows.append(macro.assign(fit=fit, pool=pool))
    frame = pd.concat(rows, ignore_index=True)
    # The Logit Lens is identical in every run; keep one copy
    frame.loc[frame["lens"] == "logit", "fit"] = "none"
    out = frame.groupby(["lens", "fit", "layer"])["hit"].mean().reset_index()
    out["recall"] = out.pop("hit") * 100
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-test", required=True, help="noise_test.csv from analyze_results.py")
    parser.add_argument("--ranks", help="ranks.csv from score_language_matrix.py")
    parser.add_argument("--lang", help="Folder with items.json and correctness_{en,fr}.csv")
    parser.add_argument("--sweep-runs", help="The sweep's runs/ folder (for the standard benchmark per layer)")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sections = ["# Per-layer analysis (H2)", ""]

    gaps = parameter_gaps(args.noise_test, args.size)
    gaps.to_csv(out / "layers_parameters.csv", index=False)
    raw = gaps.pivot_table(index=["family", "lens"], columns="layer", values="distance")
    floor = gaps[gaps["family"] == "language"].pivot_table(index="lens", columns="layer", values="floor_distance")
    floor.index = pd.MultiIndex.from_tuples([("noise floor (lowest of the pair)", lens) for lens in floor.index], names=["family", "lens"])
    sections += ["## Raw cosine distance 1 - cos, before normalising by the floor", "",
                 "Early layers move most in absolute terms, for any change of corpus; late layers move little, "
                 "but far more for a change of language than of corpus.", "",
                 markdown(pd.concat([raw, floor]), 3), ""]
    sections += [f"## Floor-normalised gap G_S = (1 - cos) / (1 - cos_floor), {args.size} prompts, eval residuals", "",
                 "Mean over the two pairs (pool A, pool B). 1 = as different as two fits of the same corpus.", "",
                 markdown(gaps.pivot_table(index=["family", "lens"], columns="layer", values="G_S"), 1), "",
                 "## Same with KL: KL / KL_floor", "",
                 markdown(gaps.pivot_table(index=["family", "lens"], columns="layer", values="KL_ratio"), 1), "",
                 "## Pairs above the noise floor on both cosine and KL (of 2)", "",
                 markdown(gaps.pivot_table(index=["family", "lens"], columns="layer", values="above_noise"), 0), ""]

    if args.ranks and args.lang:
        bilingual, n_items = bilingual_layers(args.ranks, args.lang, args.k)
        bilingual.to_csv(out / "layers_bilingual.csv", index=False)
        bilingual["condition"] = bilingual["prompt_lang"].str.upper() + " prompt, " + bilingual["target_lang"].str.upper() + " targets"
        for condition, part in bilingual.groupby("condition"):
            sections += [f"## Bilingual benchmark, recall@{args.k} per layer: {condition} ({n_items} items)", "",
                         markdown(part.pivot_table(index=["lens", "fit"], columns="layer", values="recall"), 1), ""]

    if args.sweep_runs:
        sweep = sweep_layers(args.sweep_runs, args.k, args.size)
        sweep.to_csv(out / "layers_standard.csv", index=False)
        sections += [f"## Standard benchmark (English targets), recall@{args.k} per layer, {args.size} prompts, filtered", "",
                     markdown(sweep.pivot_table(index=["lens", "fit"], columns="layer", values="recall"), 1), ""]

    (out / "layers.md").write_text("\n".join(sections))
    print("\n".join(sections))


if __name__ == "__main__":
    main()
