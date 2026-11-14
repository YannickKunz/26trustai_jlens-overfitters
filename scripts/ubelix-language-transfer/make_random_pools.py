#!/usr/bin/env python3
"""Save disjoint random fitting pools A and B, and a probe set, from one dataset.

Replaces the "first records in dataset order" rule of make_prompt_pool.py: the records of at
least --min-chars characters among the first --scan records of a seeded shuffled stream are
sampled uniformly (reservoir sampling), then split into <name>-a.json, <name>-b.json and
<name>-probe.json. The shuffle reorders the dataset's files and mixes records within a
buffer, so the sample is spread over the corpus instead of its first articles.
Lenses fitted on the old pools are not comparable with these: refit under a new ROOT.
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--n-prompts", type=int, default=1000, help="Per pool")
    parser.add_argument("--n-probe", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=600)
    parser.add_argument("--scan", type=int, default=200_000, help="Records read from the shuffled stream")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", required=True, help="Pool prefix: wikitext, en or fr")
    args = parser.parse_args()

    wanted = 2 * args.n_prompts + args.n_probe
    rng = random.Random(args.seed)
    stream = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    stream = stream.shuffle(seed=args.seed, buffer_size=10_000)
    reservoir, eligible = [], 0
    for scanned, record in enumerate(stream):
        if scanned == args.scan:
            break
        text = record[args.text_field]
        if len(text.strip()) < args.min_chars:
            continue
        eligible += 1
        if len(reservoir) < wanted:
            reservoir.append(text)
        else:
            slot = rng.randrange(eligible)
            if slot < wanted:
                reservoir[slot] = text
    if len(reservoir) < wanted:
        raise ValueError(f"Only {len(reservoir)} records of at least {args.min_chars} chars; {wanted} needed")
    rng.shuffle(reservoir)
    if len(set(reservoir)) != len(reservoir):
        raise ValueError("Duplicate records in the sample: the pools would not be disjoint")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts = {"a": reservoir[:args.n_prompts], "b": reservoir[args.n_prompts:2 * args.n_prompts],
             "probe": reservoir[2 * args.n_prompts:]}
    for suffix, prompts in parts.items():
        path = out / f"{args.name}-{suffix}.json"
        path.write_text(json.dumps(prompts, ensure_ascii=False))
        print(f"Saved {len(prompts)} prompts to {path}")
    print(f"Sampled from {eligible} eligible records")


if __name__ == "__main__":
    main()
