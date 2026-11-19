#!/usr/bin/env python3
"""Grade the model's greedy answer on the bilingual benchmark, once per prompt language.

Writes <lang>/correctness_en.csv and <lang>/correctness_fr.csv in the upstream format
(eval, item, <model>): True/False for items with a target, empty for association items,
which pose no question and are always kept.
--ignore-case also accepts a next token that matches the target up to capitalisation
(" North" for "north", " Yen" for "yen"); the upstream rule is case-sensitive. The upstream
verdict is kept in a second column, <model>_case_sensitive, for comparison. Run from jpp_lens/ with src/ on PYTHONPATH.
"""
import argparse
import json
from pathlib import Path

import torch as t

from lens_evals.eval_utils import next_token_matches_target
from lens_evals.readout_evals.readout_eval_items import ReadoutEvalItem
from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner
from workspace_lens import get_hf_model
from workspace_lens.utils import final_position_model_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated; any valid layer list")
    parser.add_argument("--lang", required=True, help="Folder written by build_language_evals.py")
    parser.add_argument("--ignore-case", action="store_true", help="Grade up to capitalisation")
    args = parser.parse_args()

    records = json.loads(Path(args.lang, "items.json").read_text())
    model = get_hf_model(args.model, attn_implementation="sdpa")
    runner = ReadoutEvalRunner(model, {}, layers=[int(layer) for layer in args.layers.split(",")])
    for language in ("en", "fr"):
        items = [ReadoutEvalItem(record["eval"], record["name"], record[language]["prompt"],
                                 record[language]["target"], tuple(record[language]["intermediates"]))
                 for record in records if record[language]["prompt"] is not None]
        frame = runner.grade_model_correctness(items, hf_model_name=args.model)
        frame[f"{args.model}_case_sensitive"] = frame[args.model]
        if args.ignore_case:
            for row, item in enumerate(items):
                if item.correctness_target() is None or bool(frame.at[row, args.model]):
                    continue
                with t.no_grad():
                    logits = final_position_model_logits(model, item.prompt, runner.max_seq_len)
                next_token = model.tokenizer.decode([int(logits.argmax())])
                frame.at[row, args.model] = next_token_matches_target(next_token.casefold(),
                                                                      item.correctness_target().casefold())
        frame.to_csv(Path(args.lang, f"correctness_{language}.csv"), index=False)
        print(f"\n{language} prompts:\n{frame.groupby('eval')[args.model].agg(['size', 'count', 'sum']).to_string()}")


if __name__ == "__main__":
    main()
