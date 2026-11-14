#!/usr/bin/env python3
"""Build the bilingual (English/French) benchmark and its fitting/held-out split.

Joins the French side in parallel_items.json with the upstream English items and writes,
under --out:
  items.json     every item with both languages, its group and its split
  weights_en/    lens-eval-<slug>.json files: fitting split, English prompts, English targets
  weights_fr/    the same concepts, French prompts, French targets
The two weights_* folders are what `jpp_cli.py fit-weights --eval-data-dir ... --items all`
reads, so J++ expert weights can be fitted on labels in the lens's own language.

The split is per concept, so an item is on the same side in every language. Items named in
--upstream-stamps (the fitting_item_names of the existing jpp.pt lenses) are forced into the
fitting split, which keeps those lenses out-of-sample on everything scored here.
CPU only, standard library only. Run from jlens_jpp_reproduction/jpp_lens/.
"""
import argparse
import json
import random
from pathlib import Path

SLUGS = ["multihop", "multilingual", "order-ops", "poetry", "association", "typo"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--french", required=True, help="parallel_items.json")
    parser.add_argument("--upstream-data", default="data/jlens/evaluations")
    parser.add_argument("--upstream-stamps", nargs="*", default=[], help="jpp_stamp.json files of the existing J++ lenses")
    parser.add_argument("--fit-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    upstream = {}
    for slug in SLUGS:
        for item in json.loads(Path(args.upstream_data, f"lens-eval-{slug}.json").read_text())["items"]:
            upstream[(slug, item["name"])] = item
    french = json.loads(Path(args.french).read_text())

    forced = set()
    for stamp in args.upstream_stamps:
        forced |= set(json.loads(Path(stamp).read_text())["fitting_item_names"])
    if not forced:
        print("WARNING: no upstream stamps given; the existing jpp.pt lenses may be scored in-sample")

    items = []
    for row in french["parallel"]:
        english = upstream[(row["eval"], row["name"])]
        prompt = row["fr_prompt"] or upstream[(row["eval"], row["fr_upstream_item"])]["prompt"]
        if len(english["intermediates"]) != len(row["fr_intermediates"]):
            raise ValueError(f"{row['name']}: the two languages need the same number of intermediates")
        aliases = {row["name"], row.get("fr_upstream_item")}
        items.append({
            "eval": row["eval"], "name": row["name"], "group": "parallel",
            "forced_fit": bool(aliases & forced),
            "en": {"prompt": english["prompt"], "target": english.get("target"), "intermediates": english["intermediates"]},
            "fr": {"prompt": prompt, "target": row["fr_target"], "intermediates": row["fr_intermediates"]},
        })
    for name, fr_intermediates in french["french_multilingual"].items():
        english = upstream[("multilingual", name)]
        items.append({
            "eval": "multilingual", "name": name, "group": "fr-multilingual", "forced_fit": name in forced,
            # French prompt only: there is no English-prompt version of these items
            "en": {"prompt": None, "target": None, "intermediates": english["intermediates"]},
            "fr": {"prompt": english["prompt"], "target": english["target"], "intermediates": fr_intermediates},
        })
    names = [item["name"] for item in items]
    if len(set(names)) != len(names):
        raise ValueError("item names must be unique across evals (fit-weights selects by name)")
    for item in items:
        for language in ("en", "fr"):
            words = item[language]["intermediates"]
            if len(set(words)) != len(words):
                raise ValueError(f"{item['name']}: repeated {language} intermediate")

    # Forced items first, then a seeded draw per eval up to the fitting fraction
    rng = random.Random(args.seed)
    for item in items:
        item["split"] = "fit" if item.pop("forced_fit") else "held-out"
    for slug in sorted({item["eval"] for item in items if item["group"] == "parallel"}):
        pool = [item for item in items if item["group"] == "parallel" and item["eval"] == slug]
        wanted = max(1, int(args.fit_fraction * len(pool)))
        free = [item for item in pool if item["split"] == "held-out"]
        rng.shuffle(free)
        for item in free[:max(0, wanted - sum(i["split"] == "fit" for i in pool))]:
            item["split"] = "fit"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
    for language in ("en", "fr"):
        folder = out / f"weights_{language}"
        folder.mkdir(exist_ok=True)
        for slug in SLUGS:
            # Every slug file must exist for the upstream loader; unused ones are empty
            rows = [{"name": item["name"], "prompt": item[language]["prompt"], "target": item[language]["target"],
                     "intermediates": item[language]["intermediates"]}
                    for item in items if item["eval"] == slug and item["group"] == "parallel" and item["split"] == "fit"]
            for row in rows:
                if row["target"] is None:
                    del row["target"]
            (folder / f"lens-eval-{slug}.json").write_text(json.dumps({"items": rows}, ensure_ascii=False, indent=1))

    for group in ("parallel", "fr-multilingual"):
        for slug in sorted({item["eval"] for item in items if item["group"] == group}):
            subset = [item for item in items if item["group"] == group and item["eval"] == slug]
            fit = sum(item["split"] == "fit" for item in subset)
            print(f"{group:16s} {slug:13s} {len(subset):3d} items: {fit} fit, {len(subset) - fit} held-out")


if __name__ == "__main__":
    main()
