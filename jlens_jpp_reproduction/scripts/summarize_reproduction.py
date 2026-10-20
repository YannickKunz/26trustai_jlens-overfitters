#!/usr/bin/env python3
"""Produce a Markdown report from a completed, auditable reproduction JSON."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def render(data):
    if data["status"] != "complete":
        raise ValueError("Refusing to report an incomplete run as final")
    meta = data["metadata"]
    lines = ["# Corrected released-lens evaluation", "",
             f"- Model: `{meta['model']}`; precision: `{meta['quantization']}`.",
             f"- GPU: {meta['gpu']}; peak allocated CUDA memory: {data['peak_cuda_allocated_bytes']/2**30:.2f} GiB.",
             f"- Started: {meta['started_at']}; finished: {data['finished_at']}.",
             f"- {len(data['items'])} raw prompts; layers {data['layers']}; paper-style layers {data['paper_layers']}.",
             f"- Readout filter excluded {data['masked_tokens']} token IDs.",
             "- **Descriptive evaluation of released lenses, not certified held-out paper-table replication.**", ""]
    lines += ["## Denominators", "", "| Task | Raw prompts | Scoreable | Correct and scoreable | Dropped concepts |",
              "|---|---:|---:|---:|---:|"]
    groups = defaultdict(list)
    for item in data["items"]:
        groups[item["eval"]].append(item)
    for task, items in groups.items():
        scoreable = [item for item in items if item["candidates"]]
        kept = [item for item in scoreable if item["correct"] is not False]
        lines.append(f"| {task} | {len(items)} | {len(scoreable)} | {len(kept)} | {sum(len(i['dropped_intermediates']) for i in items)} |")
    lines += ["", "Association and typo have no correctness target; their scoreable items are retained.", ""]
    names = ["logit", "J", "R", "J+readout_only", "J++"]
    for protocol, values in data["summaries"].items():
        table = {(r["eval"], r["lens"]): r["pass_at_k"] for r in values if r["k"] == 10}
        lines += [f"## Recall@10 — {protocol}", "", "| Task | Logit | J | R | J + readout only | J++ |",
                  "|---|---:|---:|---:|---:|---:|"]
        for task in [*groups, "macro"]:
            cells = [table.get((task, name)) for name in names]
            lines.append("| " + task + " | " + " | ".join("—" if v is None else f"{100*v:.2f}%" for v in cells) + " |")
        lines.append("")
    if "arithmetic" in data:
        lines += ["## Exploratory arithmetic intervention", "",
                  "Supplied (not generated) CoT. Single-layer, final-position J++ 8↔9 coordinate swaps; not a paper causal benchmark.", "",
                  "| Arm | Layer | Baseline P(8) | Swapped P(8) | Baseline P(9) | Swapped P(9) |",
                  "|---|---:|---:|---:|---:|---:|"]
        control_diffs = []
        for arm, values in data["arithmetic"]["arms"].items():
            for layer, edit in values["layers"].items():
                base = values["baseline"]
                cells = [base["8"]["probability"], edit["swap"]["8"]["probability"],
                         base["9"]["probability"], edit["swap"]["9"]["probability"]]
                lines.append(f"| {arm} | {layer} | " + " | ".join(f"{v:.6f}" for v in cells) + " |")
                control_diffs += [abs(edit["identity"][d]["probability"] - base[d]["probability"]) for d in ("5", "8", "9")]
        lines += ["", f"Maximum identity-control probability drift: {max(control_diffs):.8f}.",
                  "Raw JSON retains ranks, token IDs and coordinates. One arithmetic problem is not enough to establish a causal effect.", ""]
    lines += ["## Comparison limits", ""] + [f"- {s}" for s in meta["limitations"]]
    lines += ["- R-Lens is a separately fitted released baseline, with different corpus/budget/transport target. Inspect `lens_configs` before comparing it to a matched-budget paper row.",
              "- No significance test or evidence of generalization follows from these descriptive point estimates.",
              "- The old `results_3070ti.json` scores answer-token first subtokens and is not comparable to these bridge-concept scores.",
              "", "## Provenance", "", f"J++ code commit: `{meta['jpp_git_commit']}`.",
              f"Model revision: `{meta['model_revision']}`.",
              f"Runner SHA-256: `{meta['script_sha256']}`.", ""]
    for name, source in meta["lenses"].items():
        lines.append(f"- {name}: `{source['repo']}` @ `{source['revision']}`, `{source['file']}`.")
    lines += ["", "Package versions, dataset hashes, fitting configurations and per-item/per-layer ranks are embedded in the JSON."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(render(json.loads(args.input.read_text())))
