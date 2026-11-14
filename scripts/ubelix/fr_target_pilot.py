#!/usr/bin/env python3
"""Pilot: are English-fitted lenses better on French prompts only because the targets are English?

Scores the 11 held-out French `multilingual` items twice, with the original English intermediates
and with French translations, for J, R and J++ lenses fitted on English and French Wikipedia
(Qwen3.5-9B, 1000 prompts, pools A and B), with Readout Filtering. Writes per-item ranks.
Run from jpp_lens/ with src/ on PYTHONPATH.
"""
import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from lens_evals.readout_evals.readout_eval_items import load_readout_eval_items, non_semantic_token_ids
from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file, vocab_size_of

# The same concepts in French, in the order of the original intermediates
FRENCH = {
    "fr-color-night": ["français", "couleur", "nuit", "noir"],
    "fr-double-three": ["français", "double", "trois", "six"],
    "fr-month-after-jan": ["français", "mois", "janvier", "février"],
    "fr-month-after-may": ["français", "mois", "mai", "juin"],
    "fr-month-before-mar": ["français", "mois", "mars", "février"],
    "fr-number-after-four": ["français", "nombre", "quatre", "cinq"],
    "fr-place-near-far": ["français", "lieu", "près", "loin"],
    "fr-place-up-down": ["français", "lieu", "haut", "bas"],
    "fr-size-big-small": ["français", "taille", "grand", "petit"],
    "fr-time-old-young": ["français", "temps", "vieux", "jeune"],
    "french-number-seven": ["français", "nombre", "sept", "huit"],
}
MODEL = "Qwen/Qwen3.5-9B"
LAYERS = [4, 8, 12, 16, 20, 24, 28]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="The 9B sweep's runs/ folder")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    all_items = load_readout_eval_items()
    originals = [item for item in all_items if item.eval_slug == "multilingual" and item.name in FRENCH]
    items = [dataclasses.replace(item, name=f"{item.name}|en") for item in originals]
    items += [dataclasses.replace(item, name=f"{item.name}|fr", intermediates=tuple(FRENCH[item.name]))
              for item in originals]

    model = get_hf_model(MODEL, attn_implementation="sdpa")
    # Readout Filtering as in the sweep, sparing the candidates of every item scored here
    excluded = non_semantic_token_ids(model.tokenizer, vocab_size=vocab_size_of(model), items=all_items + items)
    frames = []
    for fit in ("en", "fr"):
        for pool in ("a", "b"):
            lenses = {lens: load_lens_file(f"{args.runs}/{fit}-{pool}/n1000/{lens}.pt", hf_model_name=MODEL)
                      for lens in ("j", "r", "jpp")}
            runner = ReadoutEvalRunner(model, lenses, layers=LAYERS, excluded_token_ids=excluded)
            frames.append(runner.run(items).assign(fit=fit, pool=pool))
            print(f"scored {fit}-{pool}", flush=True)
    ranks = pd.concat(frames, ignore_index=True)
    ranks[["item", "targets"]] = ranks["item"].str.split("|", expand=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ranks.to_csv(args.out, index=False)

    best = ranks.groupby(["lens", "fit", "pool", "targets", "item", "intermediate"])["rank"].min().reset_index()
    best["hit"] = best["rank"] <= 10
    # The answer word is the last intermediate of each item
    answers = {name: (orig.intermediates[-1], FRENCH[name][-1]) for orig in originals for name in [orig.name]}
    best["is_answer"] = [intermediate in answers[item] for item, intermediate in zip(best["item"], best["intermediate"])]
    for label, subset in (("all intermediates", best), ("answer word only", best[best["is_answer"]]),
                          ("answer word, without six", best[best["is_answer"] & (best["item"] != "fr-double-three")])):
        table = (subset.groupby(["lens", "fit", "targets", "pool", "item"])["hit"].mean()
                 .groupby(["lens", "fit", "targets"]).mean().unstack("targets") * 100)
        print(f"\nRecall@10 (%), {label}, mean of pools A and B:\n{table.round(1).to_string()}")


if __name__ == "__main__":
    main()
