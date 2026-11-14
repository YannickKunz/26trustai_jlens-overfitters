#!/usr/bin/env python3
"""List the French intermediates of parallel_items.json that the scorer would drop.

A word can be scored only if one of its surfaces (with or without a leading space, any case
variant upstream allows) is a single token for the model's tokenizer. Run it after every edit
of parallel_items.json, before submitting the GPU job. CPU only, a few seconds.

    python check_french_tokens.py parallel_items.json [--model Qwen/Qwen3.5-9B] [--words mot1 mot2]

With --words, checks candidate replacement words instead of the file.
Run from jpp_lens/ with src/ on PYTHONPATH (inside the container, as for the other scripts).
"""
import argparse
import json

from transformers import AutoTokenizer

from lens_evals.readout_evals.readout_eval_items import candidate_token_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("items", nargs="?", help="parallel_items.json")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--words", nargs="*", help="Check these words instead of the file")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def scoreable(word):
        return bool(candidate_token_ids(tokenizer, word, expand_order_ops_synonyms=False))

    if args.words:
        for word in args.words:
            print(f"{word:20s} {'ok' if scoreable(word) else 'SPLITS: not scoreable'}")
        return

    data = json.load(open(args.items))
    rows = [(row["name"], word) for row in data["parallel"] for word in row["fr_intermediates"]]
    rows += [(name, word) for name, words in data["french_multilingual"].items() for word in words]
    lost = [(name, word) for name, word in rows if not scoreable(word)]
    print(f"{len(lost)} of {len(rows)} French intermediates are not single tokens:")
    for name, word in lost:
        print(f"  {name:30s} {word}")


if __name__ == "__main__":
    main()
