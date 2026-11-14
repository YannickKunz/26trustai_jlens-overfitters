#!/usr/bin/env python3
"""Language-transfer tables from score_language_matrix.py's ranks, with bootstrap intervals.

A cell is recall@k of lenses fitted on one corpus, read under one condition
(prompt language, target language). An item's score is the share of its intermediates whose
best rank over layers is <= k; a cell is the item mean per task, then the mean over tasks,
then the mean over pools A and B. Every interval is a 95% percentile interval from
resampling items within each task; all lenses and conditions share each resample, so
differences between cells are paired.

Only held-out items the model answers correctly are used, and only intermediates that are
single-token and different words in the two languages, so the four conditions of an item
score the same concepts. Writes cells.csv, contrasts.csv, gaps.csv and summary.md to --out.

Recall@k saturates when a lens never ranks a target in the top k (English-fitted lenses on French
targets score 0, so G = 1 for every lens). The log-rank gap does not saturate: for a domain d (prompts
and targets in language d), it is the mean log2 best rank of the targets under the other-language
lens minus that under the d-language lens, in bits (+1: the targets are ranked twice as low).
It is written to logrank.csv and to the end of summary.md, with the H1 contrast against J.
Needs pandas and numpy only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = [("en", "en"), ("en", "fr"), ("fr", "en"), ("fr", "fr")]
FITS = ["wikitext", "en", "fr"]
LENSES = ["logit", "j", "r", "jpp", "jpp_wen", "jpp_wfr", "jpp_matched"]
# jpp_matched: J++ with expert weights fitted on labels in the language of its own corpus
MATCHED = {"wikitext": "jpp_wen", "en": "jpp_wen", "fr": "jpp_wfr"}
POSITIONS = ["language name", "category", "operand", "answer"]
OTHER = {"en": "fr", "fr": "en"}


def load_units(ranks_path, lang, k, value="hit"):
    """{unit: frame [task, item] x [lens, fit, pool, prompt_lang, target_lang]} of item scores,
    and the lines describing what each filter removed."""
    ranks = pd.read_csv(ranks_path)
    records = {record["name"]: record for record in json.loads(Path(lang, "items.json").read_text())}
    notes = []
    keys = ["lens", "fit", "pool", "group", "split", "eval", "item", "prompt_lang", "target_lang", "position"]
    best = ranks.groupby(keys)["rank"].min().reset_index()
    # value: "hit" (recall@k) or "logrank" (log2 of the best rank, lower is better)
    best["hit"] = (best["rank"] <= k).astype(float) if value == "hit" else np.log2(best["rank"].astype(float))
    notes.append(f"scored items: {best['item'].nunique()}")

    best = best[best["split"] == "held-out"]
    notes.append(f"held-out: {best['item'].nunique()}")

    def correct_items(language):
        frame = pd.read_csv(Path(lang, f"correctness_{language}.csv"))
        value = frame.iloc[:, 2]
        return set(frame["item"][value.isna() | (value.astype(str) == "True")])
    correct = {language: correct_items(language) for language in ("en", "fr")}
    needed = {"parallel": ("en", "fr"), "fr-multilingual": ("fr",)}
    best = best[[all(item in correct[language] for language in needed[group])
                 for item, group in zip(best["item"], best["group"])]]
    notes.append(f"answered correctly in every prompt language used: {best['item'].nunique()}")

    def same_word(item, position):
        english, french = (records[item][language]["intermediates"][position] for language in ("en", "fr"))
        return english.casefold() == french.casefold()
    best = best[[not same_word(item, position) for item, position in zip(best["item"], best["position"])]]
    # Keep a concept only where both of its words were scoreable (single-token)
    languages = best.groupby(["item", "position"])["target_lang"].nunique()
    both = set(languages[languages == 2].index)
    best = best[[(item, position) in both for item, position in zip(best["item"], best["position"])]]
    notes.append(f"with a concept that is a different, single-token word in both languages: {best['item'].nunique()}")

    columns = ["lens", "fit", "pool", "prompt_lang", "target_lang"]
    def wide(frame):
        return frame.groupby(["eval", "item"] + columns)["hit"].mean().unstack(columns)
    units = {group: wide(best[best["group"] == group]) for group in ("parallel", "fr-multilingual")}
    multilingual = best[best["group"] == "fr-multilingual"]
    for position, label in enumerate(POSITIONS):
        units[f"fr-multilingual, {label}"] = wide(multilingual[multilingual["position"] == position])
    units = {unit: table.dropna(axis=1, how="all") for unit, table in units.items() if len(table)}
    for unit, table in units.items():
        incomplete = table.columns[table.isna().any()]
        if len(incomplete):
            raise ValueError(f"{unit}: lenses were scored on different items: {list(incomplete)[:3]}")
        notes.append(f"{unit}: " + ", ".join(f"{len(part)} {task}" for task, part in table.groupby(level="eval")))
    return units, notes


class Unit:
    """Point estimates and bootstrap draws of every cell of one unit."""

    def __init__(self, table, n_boot, rng, scale=100):
        self.columns = table.columns
        tasks = [part.to_numpy() for _, part in table.groupby(level="eval")]
        self.point = np.mean([task.mean(axis=0) for task in tasks], axis=0) * scale
        self.draws = np.mean([task[rng.integers(0, len(task), (n_boot, len(task)))].mean(axis=1)
                              for task in tasks], axis=0) * scale
        self.n_items = len(table)

    def cell(self, lens, fit, condition, pool=None):
        """(point, draws) of one cell, averaged over the pools scored; None when absent."""
        if lens == "jpp_matched":
            lens = MATCHED.get(fit)
        if lens == "logit":
            fit = "none"
        picked = [i for i, (l, f, p, prompt, target) in enumerate(self.columns)
                  if (l, f, prompt, target) == (lens, fit, *condition) and pool in (None, p)]
        if not picked:
            return None
        return self.point[picked].mean(), self.draws[:, picked].mean(axis=1)


def combine(function, *cells):
    """Apply ``function`` to the points and to the draws of several cells; None if any is absent."""
    if any(cell is None for cell in cells):
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        return function(*[cell[0] for cell in cells]), function(*[cell[1] for cell in cells])


def row(value):
    """point, interval and the share of resamples above zero, for a (point, draws) pair."""
    point, draws = value
    draws = draws[np.isfinite(draws)]
    if not len(draws) or not np.isfinite(point):
        return {"value": np.nan, "ci_low": np.nan, "ci_high": np.nan, "share_above_0": np.nan}
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"value": point, "ci_low": low, "ci_high": high, "share_above_0": float((draws > 0).mean())}


def text(entry, digits=1):
    if not np.isfinite(entry["value"]):
        return ""
    return f"{entry['value']:.{digits}f} [{entry['ci_low']:.{digits}f}, {entry['ci_high']:.{digits}f}]"


def markdown(frame):
    frame = frame.reset_index()
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in values) + " |"
              for values in frame.itertuples(index=False)]
    return "\n".join(lines)


def gap(unit, lens, domain):
    """The draft's G for one evaluation domain (prompts and targets in ``domain``):
    (in-language lens - other-language lens) / in-language lens."""
    condition = (domain, domain)
    return combine(lambda own, other: (own - other) / own,
                   unit.cell(lens, domain, condition), unit.cell(lens, OTHER[domain], condition))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", required=True)
    parser.add_argument("--lang", required=True, help="Folder with items.json and correctness_{en,fr}.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tables, notes = load_units(args.ranks, args.lang, args.k)
    units = {name: Unit(table, args.n_boot, rng) for name, table in tables.items()}
    label = lambda condition: f"{condition[0].upper()} prompt, {condition[1].upper()} targets"  # noqa: E731
    cells, contrasts = [], []

    for name, unit in units.items():
        for lens in LENSES:
            for fit in ["none"] if lens == "logit" else FITS:
                for condition in CONDITIONS:
                    value = unit.cell(lens, fit, condition)
                    if value is not None:
                        cells.append({"unit": name, "lens": lens, "fit": fit, "condition": label(condition), **row(value)})
            if lens == "logit":
                continue
            for condition in CONDITIONS:
                def add(contrast, value):
                    if value is not None:
                        contrasts.append({"unit": name, "lens": lens, "contrast": contrast,
                                          "condition": label(condition), **row(value)})
                difference = lambda a, b: a - b  # noqa: E731
                add("language: EN fit - FR fit", combine(difference, unit.cell(lens, "en", condition), unit.cell(lens, "fr", condition)))
                add("corpus: WikiText fit - EN fit", combine(difference, unit.cell(lens, "wikitext", condition), unit.cell(lens, "en", condition)))
                for fit in ("en", "fr"):
                    add(f"noise: {fit.upper()} fit, pool A - pool B",
                        combine(difference, unit.cell(lens, fit, condition, "a"), unit.cell(lens, fit, condition, "b")))
            # Does each lens read its own language's words better? (difference of differences)
            for prompt in ("en", "fr"):
                on_english = combine(lambda a, b: a - b, unit.cell(lens, "en", (prompt, "en")), unit.cell(lens, "fr", (prompt, "en")))
                on_french = combine(lambda a, b: a - b, unit.cell(lens, "en", (prompt, "fr")), unit.cell(lens, "fr", (prompt, "fr")))
                value = combine(lambda a, b: a - b, on_english, on_french)
                if value is not None:
                    contrasts.append({"unit": name, "lens": lens, "contrast": "target-language specificity",
                                      "condition": f"{prompt.upper()} prompt", **row(value)})

    # The draft's transfer gap and the H1 comparison, on the parallel items
    gaps = []
    if "parallel" in units:
        unit = units["parallel"]
        mean_gap = {}
        for lens in LENSES[1:]:
            per_domain = {domain: gap(unit, lens, domain) for domain in ("en", "fr")}
            mean_gap[lens] = combine(lambda a, b: (a + b) / 2, per_domain["en"], per_domain["fr"])
            for domain, value in [*per_domain.items(), ("mean", mean_gap[lens])]:
                if value is not None:
                    gaps.append({"lens": lens, "quantity": f"G, {domain.upper()} domain" if domain != "mean" else "G, mean", **row(value)})
        for lens in LENSES[2:]:
            value = combine(lambda a, b: a - b, mean_gap[lens], mean_gap["j"])
            if value is not None:
                gaps.append({"lens": lens, "quantity": "G mean - G mean of J (H1)", **row(value)})
    # The old matrix's French column, on the new held-out items: French prompts, English targets
    if "fr-multilingual" in units:
        for lens in LENSES[1:]:
            unit = units["fr-multilingual"]
            value = combine(lambda own, other: (own - other) / own,
                            unit.cell(lens, "fr", ("fr", "en")), unit.cell(lens, "en", ("fr", "en")))
            if value is not None:
                gaps.append({"lens": lens, "quantity": "old G: FR prompt, EN targets (fr-multilingual)", **row(value)})

    cells, contrasts, gaps = pd.DataFrame(cells), pd.DataFrame(contrasts), pd.DataFrame(gaps)
    cells.to_csv(out / "cells.csv", index=False)
    contrasts.to_csv(out / "contrasts.csv", index=False)
    gaps.to_csv(out / "gaps.csv", index=False)

    sections = [f"# Language transfer (recall@{args.k} in %, 95% bootstrap intervals over items)", "",
                "Items kept at each filter:", *[f"- {note}" for note in notes], "",
                "`jpp`: the existing J++ lens. `jpp_wen` / `jpp_wfr`: expert weights fitted on English / French labels. "
                "`jpp_matched`: weights in the language of the lens's own corpus.", ""]
    for name in units:
        part = cells[cells["unit"] == name].assign(cell=lambda f: [text(r) for r in f.to_dict("records")])
        sections += [f"## {name}: recall by fitting corpus and condition ({units[name].n_items} items)", "",
                     markdown(part.pivot(index=["lens", "fit"], columns="condition", values="cell").reindex(
                         [(l, f) for l in LENSES for f in ["none"] + FITS], axis=0).dropna(how="all")), ""]
        part = contrasts[contrasts["unit"] == name].assign(cell=lambda f: [text(r) for r in f.to_dict("records")])
        specific = part["contrast"] == "target-language specificity"
        if len(part[~specific]):
            sections += [f"## {name}: differences in points (an interval excluding 0 is a detectable effect)", "",
                         markdown(part[~specific].pivot(index=["contrast", "lens"], columns="condition", values="cell")), ""]
        if len(part[specific]):
            sections += [f"## {name}: target-language specificity", "",
                         "(EN fit - FR fit on English targets) - (EN fit - FR fit on French targets), in points. "
                         "Positive: each lens reads the words of its own fitting language better.", "",
                         markdown(part[specific].pivot(index="lens", columns="condition", values="cell")), ""]
    if len(gaps):
        gaps = gaps.assign(estimate=[text(r, 2) for r in gaps.to_dict("records")])
        sections += ["## Transfer gap G = (in-language lens - other-language lens) / in-language lens", "",
                     "Domain = prompts and targets in that language, parallel items. The H1 rows are G(lens) - G(J): "
                     "positive means the lens transfers worse than J.", "",
                     markdown(gaps.set_index(["quantity", "lens"])[["estimate", "share_above_0"]].round(3)), ""]
    # Log-rank gap: the same items and filters, log2 best rank instead of hit (seeded separately)
    log_tables, _ = load_units(args.ranks, args.lang, args.k, value="logrank")
    log_rng = np.random.default_rng(args.seed + 1)
    log_units = {name: Unit(table, args.n_boot, log_rng, scale=1) for name, table in log_tables.items()}
    logrank = []
    if "parallel" in log_units:
        unit = log_units["parallel"]
        mean_gap = {}
        for lens in LENSES[1:]:
            per_domain = {}
            for domain in ("en", "fr"):
                condition = (domain, domain)
                per_domain[domain] = combine(lambda own, other: other - own,
                                             unit.cell(lens, domain, condition), unit.cell(lens, OTHER[domain], condition))
            mean_gap[lens] = combine(lambda a, b: (a + b) / 2, per_domain["en"], per_domain["fr"])
            for domain, value in [*per_domain.items(), ("mean", mean_gap[lens])]:
                if value is not None:
                    name = f"log-rank gap, {domain.upper()} domain" if domain != "mean" else "log-rank gap, mean"
                    logrank.append({"lens": lens, "quantity": name, **row(value)})
        for lens in LENSES[2:]:
            value = combine(lambda a, b: a - b, mean_gap[lens], mean_gap["j"])
            if value is not None:
                logrank.append({"lens": lens, "quantity": "log-rank gap - that of J (H1)", **row(value)})
        # Mean log2 best rank per cell, for reference
        for lens in LENSES:
            for fit in ["none"] if lens == "logit" else FITS:
                for condition in CONDITIONS:
                    value = unit.cell(lens, fit, condition)
                    if value is not None:
                        logrank.append({"lens": lens, "quantity": f"mean log2 rank, {fit} fit, {label(condition)}", **row(value)})
    logrank = pd.DataFrame(logrank)
    logrank.to_csv(out / "logrank.csv", index=False)
    if len(logrank):
        main_rows = logrank[~logrank["quantity"].str.startswith("mean log2 rank")]
        main_rows = main_rows.assign(estimate=[text(r, 2) for r in main_rows.to_dict("records")])
        sections += ["## Log-rank transfer gap (parallel items, bits)", "",
                     "Per domain d (prompts and targets in d): mean log2 best rank of the targets under the "
                     "other-language lens minus under the d-language lens. +1 means the targets are ranked twice as "
                     "low. Unlike recall@k it does not saturate. The H1 rows are gap(lens) - gap(J): positive means "
                     "the lens transfers worse than J.", "",
                     markdown(main_rows.set_index(["quantity", "lens"])[["estimate", "share_above_0"]].round(3)), ""]
    (out / "summary.md").write_text("\n".join(sections))
    print("\n".join(sections))


if __name__ == "__main__":
    main()
