#!/usr/bin/env python3
"""Slides 'Transfer gap on text': P (recall@10) and G on the bilingual benchmark, prompts and target
words in the same language, J / R / J++ (matched-language weights).

  make_gap_slide.py OUT.png --row "LABEL=RESULTS_DIR" [--row ...] [--title T] [--note N] [--bullets FILE]

One row of three matrices per --row (RESULTS_DIR holds cells.csv and gaps.csv of
analyze_language_transfer.py); LABEL is printed left of the row when there are several rows.
--bullets: a text file, one bullet per line, printed under the matrices.
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

DOM = {"en": "EN prompt, EN targets", "fr": "FR prompt, FR targets"}
INK, MUTED, GREY = "#1d1d1f", "#5f6368", "#efefef"
CMAP = LinearSegmentedColormap.from_list("blue", ["#e3eefb", "#2a78d6", "#174a8b"])
TITLES = {"j": ("J", "#555555"), "r": ("R", "#d39b00"), "jpp_matched": ("J++", "#1a73b8")}


def load(folder):
    cells = pd.read_csv(Path(folder, "cells.csv"))
    return cells[cells.unit == "parallel"], pd.read_csv(Path(folder, "gaps.csv"))


def draw_row(fig, folder, bottom, height, label, show_titles):
    cells, gaps = load(folder)
    n_items = None
    for k, lens in enumerate(["j", "r", "jpp_matched"]):
        ax = fig.add_axes([0.07 + k * 0.315, bottom, 0.26, height])
        ax.set_xlim(-0.55, 2); ax.set_ylim(2, -0.62 if show_titles else -0.25); ax.axis("off")
        if show_titles:
            name, color = TITLES[lens]
            ax.text(1, -0.4, name, ha="center", fontsize=26, weight="bold", color=color)
        for j, dom in enumerate(["en", "fr"]):
            ax.text(j + 0.5, -0.08, f"{dom.upper()} text", ha="center", fontsize=16, color=INK)
        for i, fit in enumerate(["en", "fr"]):
            ax.text(-0.08, i + 0.5, f"{fit.upper()} fit", ha="right", va="center", fontsize=16, color=INK)
            for j, dom in enumerate(["en", "fr"]):
                p = cells[(cells.lens == lens) & (cells.fit == fit) & (cells.condition == DOM[dom])].iloc[0].value
                if fit == dom:
                    face, txt, head, ci = GREY, INK, "in-domain", None
                else:
                    g = gaps[(gaps.lens == lens) & (gaps.quantity == f"G, {dom.upper()} domain")].iloc[0]
                    face = CMAP(min(g.value, 1)); txt = "white" if g.value > 0.55 else INK
                    head, ci = f"G = {g.value:+.2f}", f"[{g.ci_low:.2f}, {g.ci_high:.2f}]"
                ax.add_patch(Rectangle((j + 0.02, i + 0.02), 0.96, 0.96, facecolor=face, edgecolor="white", lw=3))
                ax.text(j + 0.5, i + 0.34, head, ha="center", va="center", fontsize=17, weight="bold", color=txt)
                ax.text(j + 0.5, i + 0.6, f"P = {p:.1f}", ha="center", va="center", fontsize=16, color=txt)
                if ci:
                    ax.text(j + 0.5, i + 0.82, ci, ha="center", va="center", fontsize=11.5, color=txt)
    if label:
        fig.text(0.012, bottom + height * 0.42, label, fontsize=17, weight="bold", color=INK, rotation=90,
                 ha="left", va="center", linespacing=1.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--row", action="append", required=True, help="LABEL=RESULTS_DIR")
    parser.add_argument("--title", default="Results - Transfer gap on text (corrected)")
    parser.add_argument("--note", default="P = recall@10 (%) on the held-out bilingual items, mean of the pools, "
                        "95% CI over items;  J++ with expert weights fitted in its own language")
    parser.add_argument("--bullets", help="Text file, one bullet per line")
    args = parser.parse_args()
    rows = [r.split("=", 1) for r in args.row]
    bullets = [b.strip() for b in Path(args.bullets).read_text().splitlines() if b.strip()] if args.bullets else []

    fig = plt.figure(figsize=(20, 10.5), dpi=130)
    fig.text(0.03, 0.945, args.title, fontsize=38, color=INK, va="center")
    # vertical budget: matrices from 0.89 down to the caption, then the bullets
    lines = sum(len(textwrap.wrap(b, 125)) for b in bullets)
    bullets_h = 0.031 * lines + 0.016 * len(bullets)
    caption_y = 0.05 + bullets_h + 0.05
    top, gap = 0.89, 0.015
    height = (top - caption_y - 0.03 - gap * (len(rows) - 1)) / len(rows)
    for n, (label, folder) in enumerate(rows):
        bottom = top - (n + 1) * height - n * gap
        draw_row(fig, folder, bottom, height, label if len(rows) > 1 else "", show_titles=(n == 0))
    fig.text(0.5, caption_y, "columns: evaluation domain d (prompts AND target words in that language), rows: fitting "
             "corpus c;  G(c,d) = (P[d,d] - P[c,d]) / P[d,d]\n" + args.note, ha="center", va="center",
             fontsize=13, color=MUTED, linespacing=1.5)
    y = caption_y - 0.045
    for b in bullets:
        wrapped = textwrap.wrap(b, 125)
        fig.text(0.045, y, "-", fontsize=18, color=MUTED, va="top")
        fig.text(0.07, y, "\n".join(wrapped), fontsize=18, color="#444444", va="top", linespacing=1.35)
        y -= 0.031 * len(wrapped) + 0.016
    fig.savefig(args.out, facecolor="white")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
