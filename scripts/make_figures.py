#!/usr/bin/env python3
"""Paper figures (PDF and PNG) from the analysis CSVs. Needs pandas, numpy and matplotlib only.

  fig1_conditions   recall@10 under the four language conditions, EN-fit vs FR-fit lenses
  fig2_layers       cosine distance per layer: noise floor, corpus change, language change
  fig3_convergence  cosine to the 1000-prompt lens, and noise-floor cosine, vs prompts
  fig4_h1           log-rank transfer gap per lens and domain (H1)
  fig5_layer_recall recall@10 per layer on the standard benchmark, per fitting corpus
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots 1-3 of the reference palette (validated, light surface); text in ink tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
LENSES = {"j": "J-Lens", "r": "R-Lens", "jpp": "J++ Lens"}
WIDTH = 5.5  # inches, the ICLR text width

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})


def save(fig, out, name):
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"{name}.{suffix}")
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def fig_conditions(cells, out):
    cells = cells[cells["unit"] == "parallel"]
    conditions = ["EN prompt, EN targets", "EN prompt, FR targets", "FR prompt, EN targets", "FR prompt, FR targets"]
    ticks = ["EN\nEN", "EN\nFR", "FR\nEN", "FR\nFR"]
    panels = [("j", "J-Lens"), ("r", "R-Lens"), ("jpp_matched", "J++ Lens (matched weights)")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.2), sharey=True)
    x = np.arange(len(conditions))
    for ax, (lens, title) in zip(axes, panels):
        for offset, fit, color, label in ((-0.19, "en", BLUE, "fitted on English Wikipedia"),
                                          (0.19, "fr", ORANGE, "fitted on French Wikipedia")):
            part = cells[(cells["lens"] == lens) & (cells["fit"] == fit)].set_index("condition").reindex(conditions)
            values = part["value"].to_numpy()
            ax.bar(x + offset, values, width=0.36, color=color, label=label, zorder=2, linewidth=0)
            ax.errorbar(x + offset, values, yerr=[values - part["ci_low"], part["ci_high"] - values],
                        fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.5, zorder=3)
            for xi, v in zip(x + offset, values):
                if v == 0:
                    ax.text(xi, 1, "0", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(x, ticks)
        ax.set_xlabel("prompt (top), targets (bottom)", fontsize=7)
        ax.set_title(title, color=INK)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("recall@10 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1))
    save(fig, out, "fig1_conditions")


def fig_layers(distances, out):
    d = distances[(distances["size"] == 1000) & (distances["samples"] == "eval")
                  & distances["family"].isin(["noise", "corpus", "language"])].copy()
    d["distance"] = 1 - d["cosine"]
    table = d.groupby(["family", "lens", "layer"])["distance"].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.1), sharey=True)
    series = (("noise", MUTED, "noise floor (pool A vs B)", "--"), ("corpus", BLUE, "corpus (WikiText vs EN Wiki)", "-"),
              ("language", ORANGE, "language (EN vs FR Wiki)", "-"))
    for ax, (lens, title) in zip(axes, LENSES.items()):
        for family, color, label, style in series:
            part = table[(table["lens"] == lens) & (table["family"] == family)]
            ax.plot(part["layer"], part["distance"], color=color, linestyle=style, label=label,
                    marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.6)
        ax.set_yscale("log")
        ax.set_title(title, color=INK)
        ax.set_xticks([4, 8, 12, 16, 20, 24, 28])
        ax.set_xlabel("source layer")
        ax.grid(True, axis="y", which="major")
    axes[0].set_ylabel("1 − cosine (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    save(fig, out, "fig2_layers")


def fig_convergence(summary, out):
    s = summary[summary["samples"] == "eval"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.1))
    for lens, color in zip(LENSES, (BLUE, ORANGE, AQUA)):
        conv = s[(s["family"] == "converge") & (s["lens"] == lens)].groupby("size")["cosine"].mean()
        conv.loc[1000] = 1.0
        floor = s[(s["family"] == "noise") & (s["lens"] == lens)].groupby("size")["cosine"].mean()
        for ax, data in ((axes[0], conv.sort_index()), (axes[1], floor)):
            ax.plot(data.index, data.values, color=color, marker="o", markersize=3.5, markeredgecolor="white",
                    markeredgewidth=0.6, label=LENSES[lens])
    axes[0].set_title("Similarity to the 1,000-prompt lens", color=INK)
    axes[1].set_title("Noise floor: pool A vs pool B", color=INK)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([100, 200, 500, 1000], ["100", "200", "500", "1,000"])
        ax.minorticks_off()
        ax.set_xlabel("fitting prompts")
        ax.set_ylabel("cosine")
        ax.set_ylim(0.65, 1.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, out, "fig3_convergence")


def fig_h1(logrank, out):
    rows = [("j", "J-Lens"), ("r", "R-Lens"), ("jpp", "J++ (original\nEnglish weights)"),
            ("jpp_matched", "J++ (matched-\nlanguage weights)")]
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.2))
    y = np.arange(len(rows))[::-1]
    for offset, domain, color, label in ((0.13, "EN", BLUE, "English domain"), (-0.13, "FR", ORANGE, "French domain")):
        part = logrank.set_index(["lens", "quantity"])
        values = [part.loc[(lens, f"log-rank gap, {domain} domain")] for lens, _ in rows]
        point = np.array([v["value"] for v in values])
        low = np.array([v["ci_low"] for v in values])
        high = np.array([v["ci_high"] for v in values])
        ax.errorbar(point, y + offset, xerr=[point - low, high - point], fmt="o", color=color, markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.6, elinewidth=1.2, capsize=0, label=label)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, [label for _, label in rows])
    ax.set_xlabel("log-rank transfer gap (bits; higher = transfers worse)")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, out, "fig4_h1")


def fig_layer_recall(layers, out):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.1), sharey=True)
    for ax, (lens, title) in zip(axes, LENSES.items()):
        for fit, color, label in (("wikitext", AQUA, "WikiText"), ("en", BLUE, "English Wikipedia"),
                                  ("fr", ORANGE, "French Wikipedia")):
            part = layers[(layers["lens"] == lens) & (layers["fit"] == fit)]
            ax.plot(part["layer"], part["recall"], color=color, label=label, marker="o", markersize=3.5,
                    markeredgecolor="white", markeredgewidth=0.6)
        logit = layers[layers["lens"] == "logit"]
        ax.plot(logit["layer"], logit["recall"], color=MUTED, linestyle="--", linewidth=1.2, label="Logit Lens")
        ax.set_title(title, color=INK)
        ax.set_xticks([4, 8, 12, 16, 20, 24, 28])
        ax.set_xlabel("source layer")
    axes[0].set_ylabel("recall@10 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    save(fig, out, "fig5_layer_recall")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, help="results/analysis_9b (distances.csv, distance_summary.csv)")
    parser.add_argument("--lang", required=True, help="results/language_transfer (cells.csv, logrank.csv)")
    parser.add_argument("--layers", required=True, help="Folder with layers_standard.csv (analyze_layers.py)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_conditions(pd.read_csv(Path(args.lang, "cells.csv")), out)
    fig_layers(pd.read_csv(Path(args.sweep, "distances.csv")), out)
    fig_convergence(pd.read_csv(Path(args.sweep, "distance_summary.csv")), out)
    fig_h1(pd.read_csv(Path(args.lang, "logrank.csv")), out)
    fig_layer_recall(pd.read_csv(Path(args.layers, "layers_standard.csv")), out)


if __name__ == "__main__":
    main()
