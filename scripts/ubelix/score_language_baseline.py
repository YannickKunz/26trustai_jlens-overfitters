#!/usr/bin/env python3
"""Baseline ranks: how high does each target word rank at prompts it has nothing to do with?

A French token may rank low under every lens simply because it is rarer, whatever the prompt.
For every prompt of the bilingual benchmark and each target language, this scores all the
benchmark's target words of that language EXCEPT the prompt's own, with the same lenses and
Readout Filtering as score_language_matrix.py. The baseline rank of a word under a lens is its
rank averaged over these unrelated prompts; the lift of an item is baseline minus its own rank
(in log2). Writes one row per (run, lens, prompt, target language, word) with the best rank
over layers, since the full per-layer table would be several million rows.
Run from jpp_lens/ with src/ on PYTHONPATH.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "scripts")
import jpp_cli  # noqa: E402  (for the Logit Lens constructor)
from lens_evals.readout_evals.readout_eval_items import (  # noqa: E402
    ReadoutEvalItem, load_readout_eval_items, non_semantic_token_ids)
from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import load_lens_file, vocab_size_of  # noqa: E402

LENSES = ["j", "r", "jpp", "jpp_wen", "jpp_wfr"]


def control_items(records):
    """One item per (prompt, target language) whose intermediates are every other item's
    target words in that language. A prompt's own words are left out in both languages'
    spellings, so a shared word (six, double) is never its own control."""
    words = {language: sorted({word for record in records for word in record[language]["intermediates"]})
             for language in ("en", "fr")}
    items = []
    for record in records:
        own = {word.casefold() for language in ("en", "fr") for word in record[language]["intermediates"]}
        for prompt_language in ("en", "fr"):
            if record[prompt_language]["prompt"] is None:
                continue
            for target_language in ("en", "fr"):
                controls = tuple(word for word in words[target_language] if word.casefold() not in own)
                items.append(ReadoutEvalItem(
                    eval_slug=record["eval"], name=f"{record['name']}|{prompt_language}|{target_language}",
                    prompt=record[prompt_language]["prompt"], target=record[prompt_language]["target"],
                    intermediates=controls))
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated")
    parser.add_argument("--runs", required=True, help="The sweep's runs/ folder")
    parser.add_argument("--run-names", default="wikitext-a,wikitext-b,en-a,en-b,fr-a,fr-b")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--lang", required=True, help="Folder with items.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    layers = [int(layer) for layer in args.layers.split(",")]

    records = json.loads(Path(args.lang, "items.json").read_text())
    by_name = {record["name"]: record for record in records}
    items = control_items(records)
    model = get_hf_model(args.model, attn_implementation="sdpa")
    excluded = non_semantic_token_ids(model.tokenizer, vocab_size=vocab_size_of(model),
                                      items=load_readout_eval_items() + items)
    frames = []

    def score(lenses, fit, pool):
        runner = ReadoutEvalRunner(model, lenses, layers=layers, excluded_token_ids=excluded)
        ranks = runner.run(items)
        # Best rank over layers, as in the main analysis
        best = ranks.groupby(["eval", "lens", "item", "intermediate"], sort=False)["rank"].min().reset_index()
        frames.append(best.assign(fit=fit, pool=pool))
        print(f"scored {fit}-{pool}: {', '.join(lenses)}", flush=True)

    score(jpp_cli.load_lenses(["logit"], model, hf_model_name=args.model, layers=layers), "none", "a")
    for run in args.run_names.split(","):
        folder = Path(args.runs, run, f"n{args.size}")
        lenses = {lens: load_lens_file(str(folder / f"{lens}.pt"), hf_model_name=args.model)
                  for lens in LENSES if (folder / f"{lens}.pt").is_file()}
        missing = [lens for lens in LENSES if lens not in lenses]
        if missing:
            print(f"WARNING: {folder}: no {', '.join(missing)}", flush=True)
        if lenses:
            fit, pool = run.rsplit("-", 1)
            score(lenses, fit, pool)

    ranks = pd.concat(frames, ignore_index=True).rename(columns={"rank": "best_rank", "intermediate": "word"})
    ranks[["item", "prompt_lang", "target_lang"]] = ranks["item"].str.split("|", expand=True)
    ranks["group"] = ranks["item"].map(lambda name: by_name[name]["group"])
    ranks["split"] = ranks["item"].map(lambda name: by_name[name]["split"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ranks.to_csv(args.out, index=False)
    print(f"{len(ranks)} rows to {args.out}; {ranks['word'].nunique()} control words, "
          f"{ranks['item'].nunique()} prompts' items")


if __name__ == "__main__":
    main()
