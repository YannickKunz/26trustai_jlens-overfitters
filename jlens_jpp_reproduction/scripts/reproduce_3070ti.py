#!/usr/bin/env python3
"""Low-memory released-lens evaluation; never substitutes fake/aliased weights.

Uses the J++ repository's token candidates, readout positions, correctness rule,
and item-weighted scoring. Released J++ mixer training membership is unknown:
these are descriptive evaluations, NOT a new held-out replication of its table.

This script does NOT fit any lens. It downloads three already-fitted lenses and
scores them on the readout evals:
  J    neuronpedia/jacobian-lens (HF)   fitted by the J-lens authors
  R    camilablank/workspace-lenses (HF)
  J++  koayon/jpp-lenses (HF)

Where the code comes from:
  jpp_lens/  (github.com/safety-research/jpp_lens, cloned beside scripts/; not in this folder)
    jlens.*             J-lens package (same API as the jacobian-lens repo): from_hf model wrapper
    lens_evals.*        eval items, candidate_token_ids, readout_position, min-rank, pass@k scoring,
                        non_semantic_token_ids (the Readout Filtering used by J++)
    workspace_lens.*    record_activations, Clamp / InterventionHooks (activation patching)
  Written for this reproduction (everything else in this file): MappedLens, download_lens,
  the evaluate() loop and its J / J+readout_only / J++ variants, the protocol summaries,
  arithmetic_diagnostic() and main().
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jpp_lens" / "src"))
import torch
import pandas as pd
import transformers
from huggingface_hub import hf_hub_download, model_info
# --- imported from jpp_lens (see module docstring) ---
import jlens  # jpp_lens/src/jlens
from lens_evals.eval_utils import candidate_token_ids, next_token_matches_target
from lens_evals.readout_evals.readout_eval_items import (
    load_readout_eval_items, non_semantic_token_ids, READOUT_EVAL_SPECS,
)
from lens_evals.readout_evals.readout_evals import readout_position, min_rank_over_candidates
from lens_evals.readout_evals.readout_eval_scoring import pass_at_k_over_layers
from workspace_lens.utils import record_activations
from workspace_lens.interventions import Clamp, InterventionHooks


# Thin wrapper around jpp_lens' non_semantic_token_ids: builds the keep-mask for Readout Filtering.
def build_v_sem_mask(tokenizer, vocab_size=None, *, items):
    """Exact upstream readout filter, including its candidate-sparing convention.

    Unlike the old implementation, this retains stopwords and word-like special
    tokens, as the official code does. Padding and Unicode punctuation are handled.
    """
    size = vocab_size if vocab_size is not None else len(tokenizer)
    keep = torch.ones(size, dtype=torch.bool)
    keep[non_semantic_token_ids(tokenizer, vocab_size=size, items=items)] = False
    return keep


# Written here: loads a released lens .pt memory-mapped on CPU (the 8 GB GPU cannot hold all
# [d_model, d_model] matrices). Replaces jlens.JacobianLens so one layer is moved at a time.
class MappedLens:
    """Keep checkpoint matrices memory-mapped on CPU; transport one layer at a time."""
    def __init__(self, path, model_name, width, n_layers):
        self.payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        config = self.payload.get("config", {})
        embedded_model = config.get("hf_model_name") or self.payload.get("provenance", {}).get("model_id")
        if embedded_model and embedded_model != model_name:
            raise ValueError(f"Lens model {embedded_model} != requested {model_name}")
        self.matrices = self.payload.get("J", self.payload.get("parameters", {}).get("jacobians"))
        if not self.matrices:
            raise ValueError(f"No Jacobians in {path}")
        self.layers = sorted(self.matrices)
        if any(not 0 <= l < n_layers or tuple(m.shape) != (width, width)
               for l, m in self.matrices.items()):
            raise ValueError("Incompatible lens layer/shape")

    def transport(self, residual, layer):
        # CPU fp32 transport avoids storing even one 67MB Jacobian on the GPU.
        return residual.float().cpu() @ self.matrices[layer].float().T


# Written here: pin an HF repo to an immutable commit SHA, then download the lens file.
def download_lens(repo, filename, revision):
    sha = model_info(repo, revision=revision).sha
    path = hf_hub_download(repo, filename=filename, revision=sha)
    return path, {"repo": repo, "file": filename, "revision": sha}


# Written here: glue around jpp_lens' pass_at_k_over_layers (recall@1 / @10, item-weighted).
def summaries(rows):
    if not rows:
        return []
    return json.loads(pass_at_k_over_layers(pd.DataFrame(rows), [1, 10]).to_json(orient="records"))


# Written here: the three reporting protocols (all layers; seven paper layers; seven layers
# restricted to items the model answers correctly).
def summarize_protocols(rows, audits, paper_layers):
    correct = {(a["eval"], a["item"]) for a in audits if a["correct"] is not False}
    return {
        "all_source_layers_unfiltered": summaries(rows),
        "seven_layers_unfiltered": summaries([r for r in rows if r["layer"] in paper_layers]),
        "seven_layers_correctness_filtered": summaries([
            r for r in rows if r["layer"] in paper_layers and (r["eval"], r["item"]) in correct
        ]),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


# Written here, built from jpp_lens primitives. One forward pass per item; for every layer and
# lens: transport the residual (J/R/J++), unembed, rank the intermediate-concept tokens.
# Variants: J (full vocab), J+readout_only (ablation), J++ (always readout-filtered).
@torch.no_grad()
def evaluate(model, lenses, items, layers, paper_layers, out, metadata):
    tokenizer = model.tokenizer
    vocab = model._lm_head.weight.shape[0]
    keep = build_v_sem_mask(tokenizer, vocab, items=items)
    rows, audits = [], []
    result = {"status": "running", "metadata": metadata, "layers": layers,
              "paper_layers": paper_layers, "masked_tokens": int((~keep).sum()),
              "items": audits, "ranks": rows}
    for idx, item in enumerate(items):
        candidates = {word: candidate_token_ids(tokenizer, word) for word in item.intermediates}
        dropped = [word for word, ids in candidates.items() if not ids]
        candidates = {word: ids for word, ids in candidates.items() if ids}
        # Fail rather than silently changing the benchmark by truncating it.
        if len(tokenizer.encode(item.prompt)) > 512:
            raise ValueError(f"Prompt exceeds 512 tokens: {item.name}")
        ids, acts = record_activations(model, item.prompt, 512, sorted(set(layers) | {model.n_layers - 1}))
        pos = readout_position(ids, tokenizer, READOUT_EVAL_SPECS[item.eval_slug].readout_rule)
        next_logits = model.unembed(acts[model.n_layers - 1][[-1]].float())[0].float().cpu()
        next_token = tokenizer.decode([int(next_logits.argmax())])
        target = item.correctness_target()
        audits.append({"eval": item.eval_slug, "item": item.name, "target": item.target,
                       "intermediates": list(item.intermediates), "candidates": candidates,
                       "dropped_intermediates": dropped, "readout_position": pos,
                       "greedy_token": next_token,
                       "correct": next_token_matches_target(next_token, target) if target else None})
        for layer in layers:
            residual = acts[layer][[pos]].float()
            for name, lens in {"logit": None, **lenses}.items():
                scores = model.unembed(residual if lens is None else lens.transport(residual, layer))[0].float().cpu()
                if not torch.isfinite(scores).all():
                    raise ValueError(f"Nonfinite readout: {item.name}/{name}/{layer}")
                variants = [(name, scores)]
                if name == "J":
                    variants.append(("J+readout_only", scores.masked_fill(~keep, -torch.inf)))
                if name == "J++":
                    variants = [(name, scores.masked_fill(~keep, -torch.inf))]
                for label, logits in variants:
                    for word, token_ids in candidates.items():
                        if not torch.isfinite(logits[token_ids]).any():
                            raise ValueError(f"All candidates masked: {word}")
                        rows.append({"eval": item.eval_slug, "item": item.name, "lens": label,
                                     "intermediate": word, "layer": layer,
                                     "rank": min_rank_over_candidates(logits, token_ids),
                                     "n_candidates": len(token_ids)})
        del acts
        print(f"Scored {idx + 1}/{len(items)} {item.eval_slug}/{item.name}", flush=True)
        write_json(out, result)
    result["summaries"] = summarize_protocols(rows, audits, paper_layers)
    if rows:
        frame = pd.DataFrame(rows)
        per_item = frame.assign(recall1=frame["rank"] <= 1, recall10=frame["rank"] <= 10,
                                mrr=1 / frame["rank"]).groupby(
            ["eval", "lens", "layer", "item"])[["recall1", "recall10", "mrr"]].mean()
        result["per_layer"] = json.loads(per_item.groupby(["eval", "lens", "layer"]).mean()
                                         .reset_index().to_json(orient="records"))
    result["status"] = "readouts_complete"
    write_json(out, result)
    return result


# Written here (exploratory, not a paper benchmark): swap the lens coordinates of "8" and "9"
# at the last position of a z=5, x=z+3 prompt and compare the output; identity clamp = control.
@torch.no_grad()
def arithmetic_diagnostic(model, lens, layers):
    """Exploratory supplied-CoT + actual single-layer 8↔9 coordinate swaps.

    This is not a paper benchmark and the CoT is supplied, not model-generated.
    Include identity-clamp controls and record actual model probabilities/ranks.
    """
    prompts = {
        "Silent": "Given z = 5, and x = z + 3, what is x? Answer with a single digit: x =",
        "Supplied-CoT": "Given z = 5, and x = z + 3, what is x? I substitute z and get x = 8. Answer: x =",
        "Given": "Given z = 5, and x = 8, what is x? Answer with a single digit: x =",
    }
    ids = {s: candidate_token_ids(model.tokenizer, s) for s in ("5", "8", "9")}
    if any(not v for v in ids.values()):
        raise ValueError("Arithmetic concepts need single-token candidates")
    result = {"kind": "exploratory_supplied_CoT_single_layer_swap", "arms": {}}
    for arm, prompt in prompts.items():
        _, acts = record_activations(model, prompt, 512, sorted(set(layers) | {model.n_layers - 1}))
        def output_metrics(final):
            scores = model.unembed(final.float())[0].float().cpu()
            probs = scores.softmax(-1)
            return {s: {"rank": min_rank_over_candidates(scores, candidates),
                        "probability": float(probs[candidates].sum())} for s, candidates in ids.items()}
        baseline = output_metrics(acts[model.n_layers - 1][[-1]])
        arm_result = {"prompt": prompt, "baseline": baseline, "layers": {}}
        for layer in layers:
            readout = model.unembed(lens.transport(acts[layer][[-1]], layer))[0].float().cpu()
            # Commit to one genuine single-token surface per intervention atom.
            tokens = [ids[s][0] for s in ("8", "9")]
            atoms = (model._lm_head.weight[tokens].float().cpu() @ lens.matrices[layer].float()).T
            atoms = torch.nn.functional.normalize(atoms, dim=0)
            if abs(float(torch.dot(atoms[:, 0], atoms[:, 1]))) > .999:
                raise ValueError("Degenerate swap basis")
            coords = torch.linalg.pinv(atoms) @ acts[layer][-1].float().cpu()
            layer_result = {"lens_ranks": {s: min_rank_over_candidates(readout, v) for s, v in ids.items()},
                            "atom_token_ids": tokens, "coordinates": coords.tolist()}
            for mode, targets in (("identity", coords), ("swap", coords.flip(0))):
                clamp = Clamp({layer: atoms}, {layer: targets}, token_positions=[-1])
                with InterventionHooks(model, [clamp]):
                    _, edited = record_activations(model, prompt, 512, [model.n_layers - 1])
                layer_result[mode] = output_metrics(edited[model.n_layers - 1][[-1]])
                del edited
            arm_result["layers"][layer] = layer_result
        result["arms"][arm] = arm_result
    return result


# Written here: CLI, lens/model download with pinned revisions, provenance metadata, run.
def main(default_model="Qwen/Qwen3.5-9B", default_4bit=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--load-4bit", action=argparse.BooleanOptionalAction, default=default_4bit)
    parser.add_argument("--max-items", type=int, default=None, help="Smoke-test limit per task; default all")
    parser.add_argument("--evals", nargs="+", default=["multihop"], choices=list(READOUT_EVAL_SPECS))
    parser.add_argument("--jpp-repo", default="koayon/jpp-lenses")
    parser.add_argument("--jpp-revision", default="main")
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument("--lens-file", default=None)
    parser.add_argument("--revision", default="qwen-n1000")
    parser.add_argument("--r-repo", default="camilablank/workspace-lenses")
    parser.add_argument("--r-revision", default="main")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--out", default="results/corrected_3070ti.json")
    parser.add_argument("--arithmetic", action="store_true")
    args = parser.parse_args()
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    if "order-ops" in args.evals:
        parser.error("order-ops synonym expansion is not part of this five-task runner")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing an accidental CPU model run")
    if args.model not in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.6-27B"):
        parser.error("Provide a model with verified matching released lens assets (9B or 27B)")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO)
    slug = args.model.split("/")[-1].lower()
    lens_file = args.lens_file or (
        "qwen3.5-9b-pt/jlens/Salesforce-wikitext/Qwen3.5-9B-Base_jacobian_lens.pt"
        if slug == "qwen3.5-9b" else "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens.pt")
    sources = {"J": (args.lens_repo, lens_file, args.revision),
               "R": (args.r_repo, f"{slug}/r-lens/lens.pt", args.r_revision),
               "J++": (args.jpp_repo, f"{slug}/lens.pt", args.jpp_revision)}
    paths, provenance = {}, {}
    for name, source in sources.items():
        print(f"Resolving {name}: {source}", flush=True)
        paths[name], provenance[name] = download_lens(*source)
    model_sha = model_info(args.model, revision=args.model_revision).sha
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, revision=model_sha)
    quant = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True) if args.load_4bit else None
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(args.model, revision=model_sha,
        quantization_config=quant, device_map={"": 0}, dtype=torch.float16 if args.load_4bit else torch.bfloat16,
        attn_implementation="sdpa")
    model = jlens.from_hf(hf_model, tokenizer)
    lenses = {name: MappedLens(path, args.model, model.d_model, model.n_layers) for name, path in paths.items()}
    layers = sorted(set.intersection(*(set(lens.layers) for lens in lenses.values())))
    paper_layers = [round(model.n_layers * k / 8) for k in range(1, 8)]
    if not set(paper_layers) <= set(layers):
        raise ValueError("Released lenses do not cover the seven paper layers")
    items = []
    data_dir = ROOT / "jpp_lens/data/jlens/evaluations"
    for slug in args.evals:
        items.extend(load_readout_eval_items(str(data_dir), [slug])[:args.max_items])
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(), "model": args.model, "model_revision": model_sha,
        "quantization": "NF4-double/fp16" if args.load_4bit else "bf16", "lenses": provenance,
        "lens_configs": {k: v.payload.get("config", v.payload.get("provenance", {})) for k, v in lenses.items()},
        "jpp_git_commit": subprocess.check_output(["git", "-C", str(ROOT / "jpp_lens"), "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_sha256": {s: hashlib.sha256((data_dir / f"lens-eval-{s}.json").read_bytes()).hexdigest() for s in args.evals},
        "torch": torch.__version__, "transformers": transformers.__version__,
        "packages": {p: version(p) for p in ("torch", "transformers", "bitsandbytes", "accelerate", "huggingface_hub", "pandas", "numpy") if p != "bitsandbytes" or args.load_4bit},
        "lens_n_prompts": {k: v.payload.get("n_prompts", v.payload.get("config", {}).get("num_prompts_trained_on")) for k, v in lenses.items()},
        "gpu": torch.cuda.get_device_name(), "max_items_per_task": args.max_items,
        "limitations": ["Not a paper-table replication: quantization/model/baseline fitting budgets differ.",
            "Released J++ mixer fitting membership unavailable; scores are NOT certified held-out.",
            "Original J filename says Base; checkpoint has no embedded model identity to verify.",
            "Original all-layer convention and J++ seven-layer convention both reported."]}
    result = evaluate(model, lenses, items, layers, paper_layers, args.out, metadata)
    if args.arithmetic:
        result["arithmetic"] = arithmetic_diagnostic(model, lenses["J++"], paper_layers)
    result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    result["status"] = "complete"
    write_json(args.out, result)
    print(json.dumps(result["summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
