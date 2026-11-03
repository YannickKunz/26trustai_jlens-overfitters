#!/usr/bin/env python3
"""Save a fitting-prompt pool as JSON, with the paper's WikiText rule: the first records
of at least --min-chars characters, in the dataset's fixed order."""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--n-prompts", type=int, default=1000)
    parser.add_argument("--min-chars", type=int, default=600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    prompts = []
    for record in load_dataset(args.dataset, args.config, split=args.split, streaming=True):
        if len(record[args.text_field].strip()) >= args.min_chars:
            prompts.append(record[args.text_field])
            if len(prompts) == args.n_prompts:
                break
    if len(prompts) < args.n_prompts:
        raise ValueError(f"Only {len(prompts)} prompts of at least {args.min_chars} chars")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prompts, ensure_ascii=False))
    print(f"Saved {len(prompts)} prompts to {out}")


if __name__ == "__main__":
    main()
