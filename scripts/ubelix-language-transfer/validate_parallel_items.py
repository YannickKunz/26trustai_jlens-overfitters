#!/usr/bin/env python3
"""Check parallel_items.json against the upstream English items and the tokenizer.

A custom item (no upstream counterpart) carries its own English side: en_prompt, en_target,
en_intermediates. Errors (the job fails or the item is wrong): unknown item name without en_prompt, wrong number of intermediates,
duplicate names, missing fields, fr_upstream_item not found.
Warnings (the item is scored but weakened or dropped): an intermediate appears in its own prompt,
a French word identical to the English one, a word that is not a single token, a multihop prompt
that does not end where the answer starts.
Prints every item side by side (English | French) with --show, for a human check of the meaning.

Run from jpp_lens/ with src/ on PYTHONPATH (inside the container):
    python validate_parallel_items.py parallel_items.json [--show]
"""
import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

from lens_evals.readout_evals.readout_eval_items import candidate_token_ids

DATA = Path("data/jlens/evaluations")


def load_upstream():
    upstream = {}
    for path in DATA.glob("lens-eval-*.json"):
        slug = path.stem.removeprefix("lens-eval-")
        for item in json.loads(path.read_text())["items"]:
            upstream[(slug, item["name"])] = item
    return upstream


def mentions(prompt, word):
    """Whole-word, case-insensitive occurrence of word in prompt."""
    return re.search(rf"(?<!\w){re.escape(word.casefold())}(?!\w)", prompt.casefold()) is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("items")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--show", action="store_true", help="Print every item side by side")
    args = parser.parse_args()
    upstream = load_upstream()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = json.loads(Path(args.items).read_text())
    errors, warnings, rows = [], [], []

    def single(word):
        return bool(candidate_token_ids(tokenizer, word, expand_order_ops_synonyms=False))

    names = [row.get("name") for row in data.get("parallel", [])] + list(data.get("french_multilingual", {}))
    for name in {n for n in names if names.count(n) > 1}:
        errors.append(f"{name}: name used more than once")

    for row in data.get("parallel", []):
        name, slug = row.get("name"), row.get("eval")
        missing = [k for k in ("eval", "name", "fr_target", "fr_intermediates") if k not in row]
        if missing:
            errors.append(f"{name}: missing fields {missing}")
            continue
        if "en_prompt" in row:
            if (slug, name) in upstream:
                errors.append(f"{name}: custom item (en_prompt) with the name of an upstream item")
                continue
            if not isinstance(row.get("en_intermediates"), list):
                errors.append(f"{name}: custom item without an en_intermediates list")
                continue
            english = {"prompt": row["en_prompt"], "target": row.get("en_target"), "intermediates": row["en_intermediates"]}
            for en in english["intermediates"]:
                if not single(en):
                    warnings.append(f"{name}: English '{en}' is not a single token (concept dropped)")
                if mentions(english["prompt"], en):
                    warnings.append(f"{name}: the English prompt contains its own intermediate '{en}'")
            if slug == "multihop" and not english["prompt"].endswith(" "):
                warnings.append(f"{name}: English multihop prompt does not end with a space")
        else:
            english = upstream.get((slug, name))
        if english is None:
            errors.append(f"{name}: no upstream {slug} item with this name, and no en_prompt (invented or renamed?)")
            continue
        prompt = row.get("fr_prompt")
        if prompt is None:
            source = upstream.get((slug, row.get("fr_upstream_item")))
            if source is None:
                errors.append(f"{name}: fr_prompt is null and fr_upstream_item {row.get('fr_upstream_item')!r} not found")
                continue
            prompt = source["prompt"]
        en_words, fr_words = english["intermediates"], row["fr_intermediates"]
        if len(en_words) != len(fr_words):
            errors.append(f"{name}: {len(en_words)} English intermediates but {len(fr_words)} French")
            continue
        for en, fr in zip(en_words, fr_words):
            if en.casefold() == fr.casefold():
                warnings.append(f"{name}: '{fr}' is the same word in both languages (concept not scored)")
            if not single(fr):
                warnings.append(f"{name}: '{fr}' is not a single token (concept dropped)")
            if mentions(prompt, fr):
                warnings.append(f"{name}: the French prompt contains its own intermediate '{fr}'")
        if slug == "multihop" and not (prompt.endswith(" ") or prompt.endswith("'") or prompt.endswith("’")):
            warnings.append(f"{name}: multihop prompt does not end with a space or an elision: {prompt[-25:]!r}")
        rows.append((slug, name + (" [custom]" if "en_prompt" in row else ""), english["prompt"], english.get("target"), en_words, prompt, row["fr_target"], fr_words))

    for name, fr_words in data.get("french_multilingual", {}).items():
        english = upstream.get(("multilingual", name))
        if english is None:
            errors.append(f"{name}: no upstream multilingual item with this name (invented or renamed?)")
            continue
        if len(english["intermediates"]) != len(fr_words):
            errors.append(f"{name}: {len(english['intermediates'])} English intermediates but {len(fr_words)} French")
            continue
        for en, fr in zip(english["intermediates"], fr_words):
            if not single(fr):
                warnings.append(f"{name}: '{fr}' is not a single token (concept dropped)")
            if en.casefold() == fr.casefold():
                warnings.append(f"{name}: '{fr}' is the same word in both languages (concept not scored)")
        rows.append(("multilingual", name, None, None, english["intermediates"], english["prompt"],
                     english.get("target"), fr_words))

    n_par = len(data.get("parallel", []))
    print(f"{n_par} parallel items, {len(data.get('french_multilingual', {}))} French multilingual items")
    print(f"\n{len(errors)} ERRORS")
    for line in errors:
        print("  ERROR  ", line)
    print(f"\n{len(warnings)} warnings")
    for line in warnings:
        print("  warning", line)
    if args.show:
        print("\nSide by side (check the meaning):")
        for slug, name, en_prompt, en_target, en_words, fr_prompt, fr_target, fr_words in rows:
            print(f"\n[{slug}] {name}")
            if en_prompt is not None:
                print(f"  EN: {en_prompt!r} -> {en_target!r}   via {en_words}")
            print(f"  FR: {fr_prompt!r} -> {fr_target!r}   via {fr_words}")


if __name__ == "__main__":
    main()
