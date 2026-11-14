#!/usr/bin/env python3
"""Grade the model's greedy answer on the bilingual benchmark, once per prompt language.

Writes <lang>/correctness_en.csv and <lang>/correctness_fr.csv in the upstream format
(eval, item, <model>): True/False for items with a target, empty for association items,
which pose no question and are always kept. Run from jpp_lens/ with src/ on PYTHONPATH.
"""
import argparse
import json
from pathlib import Path

from lens_evals.readout_evals.readout_eval_items import ReadoutEvalItem
from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner
from workspace_lens import get_hf_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated; any valid layer list")
    parser.add_argument("--lang", required=True, help="Folder written by build_language_evals.py")
    args = parser.parse_args()

    records = json.loads(Path(args.lang, "items.json").read_text())
    model = get_hf_model(args.model, attn_implementation="sdpa")
    runner = ReadoutEvalRunner(model, {}, layers=[int(layer) for layer in args.layers.split(",")])
    for language in ("en", "fr"):
        items = [ReadoutEvalItem(record["eval"], record["name"], record[language]["prompt"],
                                 record[language]["target"], tuple(record[language]["intermediates"]))
                 for record in records if record[language]["prompt"] is not None]
        frame = runner.grade_model_correctness(items, hf_model_name=args.model)
        frame.to_csv(Path(args.lang, f"correctness_{language}.csv"), index=False)
        print(f"\n{language} prompts:\n{frame.groupby('eval')[args.model].agg(['size', 'count', 'sum']).to_string()}")


if __name__ == "__main__":
    main()
