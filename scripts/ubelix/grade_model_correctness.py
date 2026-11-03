#!/usr/bin/env python3
"""Write a model's greedy-correctness CSV, for models the shipped CSV has no column for.

Run from jlens_jpp_reproduction/jpp_lens/ with src/ on PYTHONPATH.
"""
import argparse
from pathlib import Path

from lens_evals.readout_evals.readout_eval_items import load_readout_eval_items
from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner
from workspace_lens import get_hf_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated; any valid layer list")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = get_hf_model(args.model, attn_implementation="sdpa")
    runner = ReadoutEvalRunner(model, {}, layers=[int(l) for l in args.layers.split(",")])
    frame = runner.grade_model_correctness(load_readout_eval_items(), hf_model_name=args.model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.groupby("eval")[args.model].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    main()
