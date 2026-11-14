#!/usr/bin/env python3
"""Score every lens on the bilingual benchmark, with English and with French targets.

Each item is read out under up to four conditions, prompt language x target language. The
prompt fixes the residual; the target language only changes which words are looked for.
Lenses: J, R, the existing J++ (jpp), and the J++ lenses whose expert weights were fitted on
English labels (jpp_wen) or French labels (jpp_wfr), for every run in --run-names, plus the
Logit Lens once. Readout Filtering as in the sweep. Writes one row per
(run, lens, item, condition, intermediate, layer) to --out, and the intermediates that have
no single-token form to <out stem>_dropped.csv.
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


def conditions(records):
    """One ReadoutEvalItem per (item, prompt language, target language), named item|prompt|target."""
    items = []
    for record in records:
        for prompt_language in ("en", "fr"):
            if record[prompt_language]["prompt"] is None:
                continue
            for target_language in ("en", "fr"):
                items.append(ReadoutEvalItem(
                    eval_slug=record["eval"], name=f"{record['name']}|{prompt_language}|{target_language}",
                    prompt=record[prompt_language]["prompt"], target=record[prompt_language]["target"],
                    intermediates=tuple(record[target_language]["intermediates"])))
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated")
    parser.add_argument("--runs", required=True, help="The sweep's runs/ folder")
    parser.add_argument("--run-names", default="wikitext-a,wikitext-b,en-a,en-b,fr-a,fr-b")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--lang", required=True, help="Folder written by build_language_evals.py")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    layers = [int(layer) for layer in args.layers.split(",")]

    records = json.loads(Path(args.lang, "items.json").read_text())
    by_name = {record["name"]: record for record in records}
    items = conditions(records)
    model = get_hf_model(args.model, attn_implementation="sdpa")
    # Readout Filtering as in the sweep, sparing the candidates of every item scored here
    excluded = non_semantic_token_ids(model.tokenizer, vocab_size=vocab_size_of(model),
                                      items=load_readout_eval_items() + items)

    frames, dropped = [], []

    def score(lenses, fit, pool):
        runner = ReadoutEvalRunner(model, lenses, layers=layers, excluded_token_ids=excluded)
        frames.append(runner.run(items).assign(fit=fit, pool=pool))
        dropped.extend(runner.dropped_intermediates)
        print(f"scored {fit}-{pool}: {', '.join(lenses)}", flush=True)

    score(jpp_cli.load_lenses(["logit"], model, hf_model_name=args.model, layers=layers), "none", "a")
    for run in args.run_names.split(","):
        folder = Path(args.runs, run, f"n{args.size}")
        lenses = {}
        for lens in LENSES:
            if (folder / f"{lens}.pt").is_file():
                lenses[lens] = load_lens_file(str(folder / f"{lens}.pt"), hf_model_name=args.model)
            else:
                print(f"WARNING: {folder}/{lens}.pt is missing, skipped", flush=True)
        if lenses:
            fit, pool = run.rsplit("-", 1)
            score(lenses, fit, pool)

    ranks = pd.concat(frames, ignore_index=True)
    ranks[["item", "prompt_lang", "target_lang"]] = ranks["item"].str.split("|", expand=True)
    # position: the intermediate's index in the item, the same concept in both target languages
    ranks["position"] = [by_name[item][target]["intermediates"].index(word)
                         for item, target, word in zip(ranks["item"], ranks["target_lang"], ranks["intermediate"])]
    ranks["group"] = ranks["item"].map(lambda name: by_name[name]["group"])
    ranks["split"] = ranks["item"].map(lambda name: by_name[name]["split"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ranks.to_csv(args.out, index=False)
    lost = pd.DataFrame(sorted(set(dropped)), columns=["eval", "item", "intermediate"])
    lost.to_csv(args.out.replace(".csv", "_dropped.csv"), index=False)
    print(f"\n{len(ranks)} rows to {args.out}; {len(lost)} intermediates have no single-token form:")
    print(lost.to_string(index=False))


if __name__ == "__main__":
    main()
