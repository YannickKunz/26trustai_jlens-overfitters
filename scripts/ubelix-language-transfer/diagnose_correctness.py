#!/usr/bin/env python3
"""Why does the model fail multihop items? Print its next tokens for each prompt and variants.

For every multihop item of items.json (both languages) and every prompt variant:
  as is          the prompt exactly as graded (most end with a trailing space or a quote)
  no space       the trailing space removed, so the answer can start with " Tokyo"
  no quote       a trailing double quote removed (upstream English items end with ' "')
it prints the graded verdict (same rule as upstream), the top-5 next tokens and a greedy
continuation of 6 tokens. Writes a CSV with one row per item, language and variant.
Run from jpp_lens/ with src/ on PYTHONPATH.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch as t

from lens_evals.eval_utils import next_token_matches_target
from workspace_lens import get_hf_model
from workspace_lens.utils import final_position_model_logits


def variants(prompt):
    out = {"as is": prompt}
    if prompt.endswith(" "):
        out["no space"] = prompt.rstrip(" ")
    if prompt.rstrip().endswith('"'):
        out["no quote"] = prompt.rstrip()[:-1].rstrip(" ")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--lang", required=True, help="Folder with items.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model = get_hf_model(args.model, attn_implementation="sdpa")
    decode = lambda ids: model.tokenizer.decode(ids)  # noqa: E731
    records = [r for r in json.loads(Path(args.lang, "items.json").read_text()) if r["eval"] == "multihop"]
    rows = []
    with t.no_grad():
        for record in records:
            for language in ("en", "fr"):
                side = record[language]
                if side["prompt"] is None or side["target"] is None:
                    continue
                for variant, prompt in variants(side["prompt"]).items():
                    logits = final_position_model_logits(model, prompt, 512)
                    top = logits.topk(5).indices.tolist()
                    tokens = [decode([i]) for i in top]
                    text, generated = prompt, []
                    for _ in range(6):
                        next_id = int(final_position_model_logits(model, text, 512).argmax())
                        generated.append(next_id)
                        text = prompt + decode(generated)
                    rows.append({"item": record["name"], "split": record["split"], "source": record.get("source"),
                                 "lang": language, "variant": variant, "target": side["target"],
                                 "correct": next_token_matches_target(tokens[0], side["target"]),
                                 "top5": " | ".join(repr(x) for x in tokens), "continuation": decode(generated)})
            print("done", record["name"], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)
    print(frame.groupby(["lang", "variant"])["correct"].agg(["size", "sum"]).to_string())


if __name__ == "__main__":
    main()
