#!/usr/bin/env python3
"""Compare lens pairs per layer, on the Jacobians and on their token distributions.

Per (pair, layer, sample set): cosine and Frobenius distance of the two Jacobians, KL between
the two lenses' readouts, and each lens's KL from the model's own output distribution.
Sample sets are the readout residuals of the labelled eval items ("eval") and the last token
of each prompt in every --probe JSON list.

Run from jlens_jpp_reproduction/jpp_lens/ with src/ on PYTHONPATH. Pairs are name:lens_a.pt:lens_b.pt, on the command
line or one per line in --pairs-file; a pair with a missing lens file is skipped with a warning.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch as t

from lens_evals.readout_evals.readout_eval_items import load_readout_eval_items
from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner, ReadoutResiduals
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file, record_activations


def matrix_cosine(a, b):
    """Cosine between two matrices flattened to vectors, in fp64."""
    a, b = a.double().flatten(), b.double().flatten()
    return float(t.dot(a, b) / (a.norm() * b.norm()))


def frobenius(a, b):
    """``||a - b||_F`` and the same divided by the mean of the two norms, in fp64."""
    a, b = a.double(), b.double()
    distance = float((a - b).norm())
    return distance, distance / float((a.norm() + b.norm()) / 2)


def mean_kl(logp_a, logp_b):
    """Mean over rows of KL(a || b), for log-probabilities [P, V]."""
    return float((logp_a.exp() * (logp_a - logp_b)).sum(-1).mean())


@t.no_grad()
def log_probs(model, lens, residuals, layer, batch_size=32):
    """Lens token log-probabilities over the full vocabulary, [P, V] on the GPU."""
    return t.cat([
        model.unembed(lens.transport(batch, layer)).float().log_softmax(-1)
        for batch in residuals.split(batch_size)
    ])


@t.no_grad()
def model_log_probs(model, final_residuals, batch_size=32):
    """The model's own next-token log-probabilities from its final-block residuals."""
    return t.cat([
        model.unembed(batch).float().log_softmax(-1) for batch in final_residuals.split(batch_size)
    ])


@t.no_grad()
def probe_residuals(model, prompts, layers, max_seq_len=128):
    """{layer: [P, d_model]} fp32 residuals at the last token of each truncated prompt,
    for ``layers`` and the final block."""
    final = model.n_layers - 1
    wanted = sorted(set(layers) | {final})
    recorded = [record_activations(model, prompt, max_seq_len, wanted)[1] for prompt in prompts]
    return {layer: t.stack([acts[layer][-1].float() for acts in recorded]) for layer in wanted}


def eval_final_residuals(model, readout, model_name):
    """Final-block residuals at the eval items' readout positions, in the cache's item order."""
    by_key = {(item.eval_slug, item.name): item for item in load_readout_eval_items()}
    items = [by_key[key] for key in readout.item_keys]
    final = model.n_layers - 1
    runner = ReadoutEvalRunner(model, {}, layers=[final], max_seq_len=readout.max_seq_len)
    recorded = runner.record_residuals(items, hf_model_name=model_name)
    if recorded.item_keys != readout.item_keys:
        raise ValueError("final-block recording kept different items than the residual cache")
    return recorded.residuals_L_dict_PN[final]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs="*", default=[], help="name:lens_a.pt:lens_b.pt")
    parser.add_argument("--pairs-file", default=None, help="One name:lens_a.pt:lens_b.pt per line")
    parser.add_argument("--probe", nargs="*", default=[], help="name:prompts.json")
    parser.add_argument("--residuals", required=True, help="ReadoutResiduals cache from fit-weights")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    readout = ReadoutResiduals.load(args.residuals)
    model = get_hf_model(args.model, attn_implementation="sdpa")
    device = model.input_device
    final = model.n_layers - 1
    sample_sets = {"eval": dict(readout.residuals_L_dict_PN)}
    sample_sets["eval"][final] = eval_final_residuals(model, readout, args.model)
    for spec in args.probe:
        name, path = spec.split(":")
        sample_sets[name] = probe_residuals(model, json.loads(Path(path).read_text()), readout.layers)
    model_logp = {name: model_log_probs(model, residuals[final].to(device))
                  for name, residuals in sample_sets.items()}
    specs = list(args.pair)
    if args.pairs_file:
        specs += Path(args.pairs_file).read_text().split()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        name, path_a, path_b = spec.split(":")
        if not (Path(path_a).is_file() and Path(path_b).is_file()):
            print(f"WARNING: skipping {name}: missing lens file", flush=True)
            continue
        lens_a = load_lens_file(path_a, hf_model_name=args.model)
        lens_b = load_lens_file(path_b, hf_model_name=args.model)
        for layer in readout.layers:
            jacobian_a = lens_a.jacobians_L_dict_FN[layer].to(device)
            jacobian_b = lens_b.jacobians_L_dict_FN[layer].to(device)
            cosine = matrix_cosine(jacobian_a, jacobian_b)
            distance, relative = frobenius(jacobian_a, jacobian_b)
            for samples, residuals_L_dict_PN in sample_sets.items():
                residuals = residuals_L_dict_PN[layer].to(device)
                logp_a = log_probs(model, lens_a, residuals, layer)
                logp_b = log_probs(model, lens_b, residuals, layer)
                rows.append({
                    "pair": name, "layer": layer, "samples": samples, "n_samples": len(residuals),
                    "cosine": cosine, "frobenius": distance, "frobenius_rel": relative,
                    "kl_a_b": mean_kl(logp_a, logp_b),
                    "kl_b_a": mean_kl(logp_b, logp_a),
                    # The model's own output as the reference distribution
                    "kl_model_a": mean_kl(model_logp[samples], logp_a),
                    "kl_model_b": mean_kl(model_logp[samples], logp_b),
                })
        print(rows[-1], flush=True)
        # Checkpoint: a preempted comparison keeps what it finished
        pd.DataFrame(rows).to_csv(args.out, index=False)
    summary = pd.DataFrame(rows).groupby(["pair", "samples"])[
        ["cosine", "frobenius_rel", "kl_a_b", "kl_b_a", "kl_model_a", "kl_model_b"]].mean()
    print(summary.to_string())


if __name__ == "__main__":
    main()
