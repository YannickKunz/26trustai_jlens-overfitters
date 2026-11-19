#!/usr/bin/env python3
"""Full performance matrices P[c, d]: recall@10 of the lens fitted on corpus c (WikiText, English
Wikipedia, French Wikipedia) under condition d = (prompt language, target language), parallel
bilingual items, for J, R and J++ (matched-language weights), with the Logit Lens as a fitting-free
reference row.

  make_p_matrices.py OUT_STEM --row "LABEL=RESULTS_DIR" [--row ...] [--title T] [--note N]

Writes OUT_STEM.png (one row of three matrices per --row) and OUT_STEM.md (the same numbers with
their 95% intervals). RESULTS_DIR holds cells.csv of analyze_language_transfer.py.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

CONDITIONS = ["EN prompt, EN targets", "EN prompt, FR targets", "FR prompt, EN targets", "FR prompt, FR targets"]
HEADERS = ["EN prompt\nEN targets", "EN prompt\nFR targets", "FR prompt\nEN targets", "FR prompt\nFR targets"]
FITS = [("wikitext", "WikiText"), ("en", "EN Wiki"), ("fr", "FR Wiki"), ("none", "Logit Lens")]
LENSES = [("j", "J", "#555555"), ("r", "R", "#d39b00"), ("jpp_matched", "J++", "#1a73b8")]
IN_DOMAIN = {("wikitext", 0), ("en", 0), ("fr", 3)}
INK, MUTED = "#1d1d1f", "#5f6368"
CMAP = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#2a78d6", "#174a8b"])


def lookup(cells, lens, fit, condition):
    lens = "logit" if fit == "none" else lens
    row = cells[(cells.lens == lens) & (cells.fit == fit) & (cells.condition == condition)]
    return None if row.empty else row.iloc[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_stem")
    parser.add_argument("--row", action="append", required=True, help="LABEL=RESULTS_DIR")
    parser.add_argument("--title", default="Performance matrices P (recall@10, %)")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    rows = [r.split("=", 1) for r in args.row]

    md = [f"# {args.title}", "", "Rows: fitting corpus; columns: prompt language x target language. Parallel "
          "bilingual items (same concepts in both languages), held-out, answered correctly. 95% intervals over "
          "items. In-domain cells in bold. The Logit Lens needs no fitting and is the same for every corpus.", ""]
    fig = plt.figure(figsize=(20, 5.2 * len(rows) + 1.6), dpi=130)
    top = 1 - 1.1 / fig.get_figheight()
    fig.text(0.02, top + 0.4 / fig.get_figheight(), args.title, fontsize=30, color=INK, va="center")
    height = (top - 0.8 / fig.get_figheight()) / len(rows)
    for n, (label, folder) in enumerate(rows):
        cells = pd.read_csv(Path(folder, "cells.csv"))
        cells = cells[cells.unit == "parallel"]
        md += [f"## {label.replace(chr(10), ', ')}", ""]
        bottom = top - (n + 1) * height
        for k, (lens, name, color) in enumerate(LENSES):
            md += [f"### {name}", "", "| fit | " + " | ".join(CONDITIONS) + " |", "|---" * 5 + "|"]
            ax = fig.add_axes([0.07 + k * 0.315, bottom + 0.03 * 5.2 / fig.get_figheight(), 0.29, height * 0.88])
            ax.set_xlim(-1.05, 4); ax.set_ylim(4.05, -0.95); ax.axis("off")
            if n == 0:
                ax.text(2, -0.78, name, ha="center", fontsize=24, weight="bold", color=color)
            for j, header in enumerate(HEADERS):
                ax.text(j + 0.5, -0.12, header, ha="center", va="bottom", fontsize=11.5, color=INK, linespacing=1.2)
            for i, (fit, fit_label) in enumerate(FITS):
                ax.text(-0.08, i + 0.5, fit_label, ha="right", va="center", fontsize=13,
                        color=MUTED if fit == "none" else INK, style="italic" if fit == "none" else "normal")
                line = []
                for j, condition in enumerate(CONDITIONS):
                    cell = lookup(cells, lens, fit, condition)
                    if cell is None:
                        line.append("")
                        continue
                    value = cell.value
                    bold = (fit, j) in IN_DOMAIN
                    ax.add_patch(Rectangle((j + 0.03, i + 0.04), 0.94, 0.92, facecolor=CMAP(value / 90),
                                           edgecolor=INK if bold else "white", lw=2.2 if bold else 2))
                    txt = "white" if value > 45 else INK
                    ax.text(j + 0.5, i + 0.42, f"{value:.1f}", ha="center", va="center", fontsize=15,
                            weight="bold" if bold else "normal", color=txt)
                    ax.text(j + 0.5, i + 0.75, f"[{cell.ci_low:.0f}, {cell.ci_high:.0f}]", ha="center",
                            va="center", fontsize=9.5, color=txt)
                    text = f"{value:.1f} [{cell.ci_low:.1f}, {cell.ci_high:.1f}]"
                    line.append(f"**{text}**" if bold else text)
                md.append(f"| {fit_label} | " + " | ".join(line) + " |")
            md.append("")
        if len(rows) > 1:
            fig.text(0.012, bottom + height * 0.45, label, fontsize=16, weight="bold", color=INK, rotation=90,
                     ha="left", va="center", linespacing=1.3)
    fig.text(0.5, 0.25 / fig.get_figheight(), "Black border: in-domain cell (prompts and targets in the lens's fitting "
             "language; English for WikiText). " + args.note, ha="center", va="center", fontsize=12.5, color=MUTED)
    fig.savefig(args.out_stem + ".png", facecolor="white")
    Path(args.out_stem + ".md").write_text("\n".join(md) + "\n")
    print("wrote", args.out_stem + ".png", args.out_stem + ".md")


if __name__ == "__main__":
    main()
