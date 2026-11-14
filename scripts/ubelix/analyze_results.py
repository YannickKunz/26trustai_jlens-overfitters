#!/usr/bin/env python3
"""Collect the 9B sweep's scores and lens distances into tidy CSVs and summary tables.

Reads $ROOT/runs/<run>/n<size>/*_pass_at_k.csv and rank CSVs, and $ROOT/scores/compare.csv.
Writes to --out: recall.csv, recall_ci.csv, distances.csv, distance_summary.csv,
noise_test.csv and summary_tables.md. Needs pandas and numpy only.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MACRO_EVALS = ["multihop", "multilingual", "typo", "association", "poetry"]
SIZES = [100, 200, 500, 1000]
DATASETS = {"wikitext": "WikiText", "en": "English Wikipedia", "fr": "French Wikipedia"}
# Readout Filtering on every lens: the project's main protocol
SCORE_FILES = {"filtered": ["jpp_scores", "baselines_filtered"], "unfiltered": ["jpp_scores", "baselines"]}
# Noise pairs whose datasets make up each effect
NOISE_FOR = {"language": ["noise-en", "noise-fr"], "corpus": ["noise-wikitext", "noise-en"],
             "mixed": ["noise-wikitext", "noise-fr"]}


def run_dirs(root):
    for path in sorted(Path(root, "runs").glob("*/n[0-9]*")):
        run, size = path.parent.name, int(path.name[1:])
        yield run, size, path


def load_recall(root):
    rows = []
    for run, size, path in run_dirs(root):
        for scoring, stems in SCORE_FILES.items():
            for stem in stems:
                table = path / f"{stem}_pass_at_k.csv"
                if table.is_file():
                    frame = pd.read_csv(table)
                    rows.append(frame.assign(run=run, size=size, scoring=scoring))
    frame = pd.concat(rows, ignore_index=True)
    frame["dataset"] = frame["run"].str.replace(r"-(a|b|64)$", "", regex=True)
    frame["pool"] = frame["run"].str.extract(r"-(a|b|64)$")[0]
    return frame[frame["weighting"] == "item"]


def per_item_hits(ranks, k=10):
    """[eval, lens, item] -> fraction of intermediates with min-over-layers rank <= k."""
    mins = ranks.groupby(["eval", "lens", "item", "intermediate"])["rank"].min().reset_index()
    mins["hit"] = mins["rank"] <= k
    return mins.groupby(["eval", "lens", "item"])["hit"].mean()


def bootstrap_macro(hits, n_boot, rng):
    """Macro recall and its 95% interval, resampling items within each eval."""
    groups = [hits.xs(e, level="eval").to_numpy() for e in MACRO_EVALS
              if e in hits.index.get_level_values("eval")]
    if len(groups) < len(MACRO_EVALS):
        return np.nan, np.nan, np.nan
    point = np.mean([g.mean() for g in groups])
    draws = np.mean([g[rng.integers(0, len(g), (n_boot, len(g)))].mean(axis=1) for g in groups], axis=0)
    return point, np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def recall_intervals(root, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for run, size, path in run_dirs(root):
        for scoring, stems in SCORE_FILES.items():
            for stem in stems:
                csv = path / f"{stem}.csv"
                if not csv.is_file():
                    continue
                hits = per_item_hits(pd.read_csv(csv))
                for lens in hits.index.get_level_values("lens").unique():
                    point, low, high = bootstrap_macro(hits.xs(lens, level="lens"), n_boot, rng)
                    rows.append({"run": run, "size": size, "scoring": scoring, "lens": lens,
                                 "recall10": point, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def load_distances(root):
    frame = pd.read_csv(Path(root, "scores", "compare.csv"))
    parts = frame["pair"].str.extract(r"^(?P<kind>.+?)_n(?P<size>\d+)_(?P<lens>.+)$")
    frame = pd.concat([frame, parts], axis=1)
    frame["size"] = frame["size"].astype(int)
    frame["family"] = frame["kind"].str.replace(r"-(a|b|en|fr|wikitext|en-a|en-b|fr-a|fr-b|wikitext-a|wikitext-b)$",
                                                "", regex=True)
    frame["kl_sym"] = (frame["kl_a_b"] + frame["kl_b_a"]) / 2
    return frame


def noise_test(distances):
    """Per effect pair, layer and sample set: is it further apart than the relevant noise pairs?
    Conservative: compared against the least similar of the relevant noise pairs."""
    keys = ["size", "lens", "layer", "samples"]
    noise = distances[distances["family"] == "noise"]
    rows = []
    for family, noise_kinds in NOISE_FOR.items():
        floor = (noise[noise["kind"].isin(noise_kinds)].groupby(keys)
                 .agg(noise_cosine=("cosine", "min"), noise_kl=("kl_sym", "max"),
                      noise_frob=("frobenius_rel", "max")).reset_index())
        effect = distances[distances["family"] == family].merge(floor, on=keys)
        effect["above_noise"] = (effect["cosine"] < effect["noise_cosine"]) & (effect["kl_sym"] > effect["noise_kl"])
        effect["excess_distance"] = (1 - effect["cosine"]) / (1 - effect["noise_cosine"])
        effect["excess_kl"] = effect["kl_sym"] / effect["noise_kl"]
        rows.append(effect)
    return pd.concat(rows, ignore_index=True)


def markdown(frame, floatfmt=".1f"):
    """A GitHub Markdown table, without the optional tabulate dependency."""
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    frame = frame.reset_index()
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else format(value, floatfmt)
        return str(value)
    header = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="The 9B sweep root, e.g. .../jlens_paper/9b")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    recall = load_recall(args.root)
    recall.to_csv(out / "recall.csv", index=False)
    intervals = recall_intervals(args.root, args.n_boot)
    intervals.to_csv(out / "recall_ci.csv", index=False)
    distances = load_distances(args.root)
    distances.to_csv(out / "distances.csv", index=False)
    summary = (distances.groupby(["family", "kind", "lens", "size", "samples"])
               [["cosine", "frobenius_rel", "kl_sym", "kl_model_a", "kl_model_b"]].mean().reset_index())
    summary.to_csv(out / "distance_summary.csv", index=False)
    tested = noise_test(distances)
    tested.to_csv(out / "noise_test.csv", index=False)

    macro = recall[(recall["eval"] == "macro") & (recall["k"] == 10)].copy()
    macro["pass_at_k"] *= 100
    sections = ["# Summary tables (Qwen3.5-9B)", "",
                "Recall@10 in %, macro over five tasks, item-weighted. Distances are means over the seven layers.", ""]

    sweep = macro[(macro["pool"].isin(["a", "b"])) & (macro["scoring"] == "filtered")]
    table = sweep.pivot_table(index=["dataset", "lens"], columns="size", values="pass_at_k", aggfunc="mean")
    sections += ["## Recall@10 by prompts, Readout Filtering on every lens (mean of pools A and B)", "", markdown(table), ""]
    spread = sweep[sweep["size"] == 1000].pivot_table(index=["dataset", "lens"], columns="pool", values="pass_at_k")
    sections += ["## Recall@10 at 1000 prompts, pool A vs pool B (filtered)", "", markdown(spread), ""]
    ci = intervals[(intervals["size"] == 1000) & (intervals["scoring"] == "filtered")].copy()
    ci[["recall10", "ci_low", "ci_high"]] *= 100
    sections += ["## Recall@10 at 1000 prompts with 95% bootstrap intervals over items (filtered)", "",
                 markdown(ci.drop(columns=["size", "scoring"]).set_index(["run", "lens"]).sort_index()), ""]
    unfiltered = macro[(macro["pool"].isin(["a", "b"])) & (macro["scoring"] == "unfiltered") & (macro["size"] == 1000)]
    sections += ["## Recall@10 at 1000 prompts, paper protocol (J, R and Logit unfiltered; mean of pools)", "",
                 markdown(unfiltered.pivot_table(index="lens", columns="dataset", values="pass_at_k")), ""]

    eval_only = summary[summary["samples"] == "eval"]
    main_kinds = eval_only[eval_only["family"].isin(["noise", "corpus", "language", "mixed"]) & (eval_only["size"] == 1000)]
    sections += ["## Distances at 1000 prompts, eval samples (each kind averaged over its pairs)", "",
                 markdown(main_kinds.groupby(["family", "lens"])[["cosine", "frobenius_rel", "kl_sym"]].mean(), ".3f"), ""]
    by_samples = summary[summary["family"].isin(["noise", "corpus", "language"]) & (summary["size"] == 1000)]
    sections += ["## Symmetric KL at 1000 prompts by sample set", "",
                 markdown(by_samples.pivot_table(index=["family", "lens"], columns="samples", values="kl_sym"), ".2f"), ""]
    floor = summary[(summary["family"] == "noise") & (summary["samples"] == "eval")]
    sections += ["## Noise floor (A vs B, same dataset) by prompts: cosine", "",
                 markdown(floor.pivot_table(index=["kind", "lens"], columns="size", values="cosine"), ".3f"), ""]
    above = (tested.groupby(["family", "lens", "size", "samples"])["above_noise"].sum().unstack("samples"))
    sections += ["## Layers (of 7) where an effect is above the noise floor on both cosine and KL", "",
                 "Each effect has two pairs (A and B), so the maximum count is 14 for language and corpus, 7 for mixed.", "",
                 markdown(above, ".0f"), ""]
    converge = summary[(summary["family"] == "converge") & (summary["samples"] == "eval")]
    sections += ["## Convergence: cosine to the 1000-prompt lens of the same pool", "",
                 markdown(converge.pivot_table(index=["kind", "lens"], columns="size", values="cosine"), ".3f"), ""]
    method = summary[(summary["family"] == "method") & (summary["samples"] == "eval") & (summary["size"] == 1000)]
    sections += ["## Cross-method distances at 1000 prompts (eval samples)", "",
                 markdown(method.pivot_table(index="kind", columns="lens", values="cosine"), ".3f"), ""]
    model_kl = distances[distances["family"] == "noise"].melt(
        id_vars=["kind", "lens", "size", "samples", "layer"], value_vars=["kl_model_a", "kl_model_b"])
    sections += ["## KL from the model's own output to each lens (noise pairs' lenses, 1000 prompts)", "",
                 markdown(model_kl[model_kl["size"] == 1000].pivot_table(index="lens", columns="samples", values="value"), ".2f"), ""]
    (out / "summary_tables.md").write_text("\n".join(sections))
    print("\n".join(sections))


if __name__ == "__main__":
    main()
