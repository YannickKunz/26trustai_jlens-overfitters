#!/bin/bash
#SBATCH --job-name=fit_lenses
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=88G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_sweep_%x_%A_%a.log
#SBATCH --error=slurm_sweep_%x_%A_%a.err

# Fit J, R and J++ lenses with the J++ paper's pipeline (jpp_cli.py), score them and compare them.
# Submit from the repo root. One-time setup:
#   git clone https://github.com/safety-research/jpp_lens.git jlens_jpp_reproduction/jpp_lens
#   git -C jlens_jpp_reproduction/jpp_lens checkout d7efc0de372e20589e899188c0cfe8434acc63ec
#   apptainer build /storage/scratch/users/$USER/jlens_paper.sif scripts/ubelix/Apptainer_jpp.def
# Pools are $ROOT/pools/<pool>.json, from scripts/ubelix/make_prompt_pool.py.
# The compare stage expects pools wikitext-*, en-* and fr-*, with A/B halves and a *-probe set.
# Defaults are Qwen3.5-9B; set MODEL, LAYERS, PUBLISHED and ROOT for another model.
# Runs on preemptable H100s (gpu-invest). For the non-preemptable slot:
#   sbatch --partition=gpu --qos=job_gratis --no-requeue ...
#   STAGE=grade                    the model's correctness CSV
#   STAGE=validate                 the released J++ lens on the held-out items (RELEASED_JPP)
#   STAGE=router POOL=p            the pool's router
#   STAGE=shard  POOL=p RUN=r NUM_PROMPTS=n NUM_SHARDS=s   with --array=0-(2s-1)
#   STAGE=score  RUN=r NUM_PROMPTS=n NUM_SHARDS=s SIZES="a b"   merge, fit weights, evaluate
#   STAGE=gate   RUN=r SIZE=n      fail unless J++ recall@10 is within 5 points of PUBLISHED
#   STAGE=rescore                  J, R and Logit again with Readout Filtering, for every scored size
#   STAGE=compare                  cosine, Frobenius and KL over every run and size (OUT to rename)
# Array task t fits shard t % NUM_SHARDS of mode t / NUM_SHARDS (0 rlens, 1 none).
# Sizes are prefixes of the pool. shard and score skip finished work, so a requeue is safe.
set -euo pipefail

STAGE="${STAGE:?set STAGE}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
LAYERS="${LAYERS:-4,8,12,16,20,24,28}"
PUBLISHED="${PUBLISHED:-55.0}"
RELEASED_JPP="${RELEASED_JPP:-qwen3.5-9b/lens.pt}"   # file in koayon/jpp-lenses, for STAGE=validate
ROWS="${ROWS:-16}"
REPO="$(pwd)"
SCRATCH_DIR="/storage/scratch/users/$USER"
SIF="${SIF:-$SCRATCH_DIR/jlens_paper.sif}"
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
ROOT="${ROOT:-$SCRATCH_DIR/jlens_paper/9b}"
JPP="jlens_jpp_reproduction/jpp_lens"
HELPERS="/work/scripts/ubelix"
CSV="$ROOT/correctness.csv"
RESIDUALS="$ROOT/residuals.pt"
mkdir -p "$ROOT/routers" "$ROOT/runs" "$ROOT/scores"

echo "Node: $(hostname)  Date: $(date)  Stage: $STAGE  Run: ${RUN:-none}  Task: ${SLURM_ARRAY_TASK_ID:-none}  Restarts: ${SLURM_RESTART_COUNT:-0}"

run() {
    apptainer exec --nv \
        --bind "$REPO:/work" \
        --bind "$SCRATCH_DIR:$SCRATCH_DIR" \
        --env HF_HOME="$HF_HOME" \
        --env PYTHONPATH="/work/$JPP/src" \
        --env PROMPTS_JSON="$ROOT/pools/${POOL:-none}.json" \
        --pwd "/work/$JPP" \
        "$SIF" python "$@"
}
cli() { run scripts/jpp_cli.py "$@"; }

# Finished shard dirs for shards 0..$2-1 of the fit root $1, in order; the stamp is written on success
shard_dirs() {
    for shard in $(seq 0 $(($2 - 1))); do
        find "$1" -type f -path "*/shard${shard}of${NUM_SHARDS}/fit_shard_stamp.json" -printf '%h\n' | head -1
    done
}

case "$STAGE" in
grade)
    [ -f "$CSV" ] || run "$HELPERS/grade_model_correctness.py" --model "$MODEL" --layers "$LAYERS" --out "$CSV"
    ;;
validate)
    LENS="$ROOT/released/jpp.pt"
    mkdir -p "$ROOT/released"
    ln -sf "$(ls "$HF_HOME"/hub/models--koayon--jpp-lenses/snapshots/*/$RELEASED_JPP)" "$LENS"
    cli evaluate --hf-model-name "$MODEL" --layers "$LAYERS" --correctness-csv "$CSV" \
        --lens "$LENS" --out "$ROOT/released/jpp_scores.csv"
    ;;
router)
    [ -f "$ROOT/routers/$POOL.pt" ] || run "$HELPERS/jpp_cli_with_prompts.py" fit-router \
        --hf-model-name "$MODEL" --layers "$LAYERS" --out "$ROOT/routers/$POOL.pt"
    ;;
shard)
    task="${SLURM_ARRAY_TASK_ID:?submit with --array}"
    modes=(rlens none)
    mode="${modes[$((task / NUM_SHARDS))]}"
    shard=$((task % NUM_SHARDS))
    fit_root="$ROOT/runs/$RUN/$mode"
    if [ -n "$(find "$fit_root" -type f -path "*/shard${shard}of${NUM_SHARDS}/fit_shard_stamp.json" 2>/dev/null | head -1)" ]; then
        echo "Skipping finished $mode shard $shard"
    else
        cli fit-shard --hf-model-name "$MODEL" --layers "$LAYERS" --router-path "$ROOT/routers/$POOL.pt" \
            --prompts-json "$ROOT/pools/$POOL.json" --num-prompts "$NUM_PROMPTS" --lrp-mode "$mode" \
            --shard-idx "$shard" --num-shards "$NUM_SHARDS" --jacobian-rows-per-pass "$ROWS" \
            --checkpoint-every-n-prompts "$NUM_PROMPTS" --artifacts-base-dir "$fit_root"
    fi
    ;;
score)
    for size in $SIZES; do
        out="$ROOT/runs/$RUN/n$size"
        [ -f "$out/baselines_pass_at_k.csv" ] && { echo "Skipping scored size $size"; continue; }
        mkdir -p "$out"
        shards=$((size * NUM_SHARDS / NUM_PROMPTS))
        cli merge-experts --checkpoint-dirs $(shard_dirs "$ROOT/runs/$RUN/rlens" "$shards") \
            --out "$out/experts_rlens.pt" --pooled-lens-out "$out/r.pt"
        cli merge-experts --checkpoint-dirs $(shard_dirs "$ROOT/runs/$RUN/none" "$shards") \
            --out "$out/experts_none.pt" --pooled-lens-out "$out/j.pt"
        # The labelled fit:0 items are English here too, so every lens is weighted alike
        cli fit-weights --hf-model-name "$MODEL" --layers "$LAYERS" --correctness-csv "$CSV" \
            --experts "$out/experts_rlens.pt" --cache-path "$RESIDUALS" --out "$out/jpp.pt"
        cli evaluate --hf-model-name "$MODEL" --layers "$LAYERS" --correctness-csv "$CSV" \
            --lens "$out/jpp.pt" --out "$out/jpp_scores.csv"
        cli evaluate --hf-model-name "$MODEL" --layers "$LAYERS" --correctness-csv "$CSV" \
            --lens "$out/j.pt" "$out/r.pt" logit --no-readout-filter --out "$out/baselines.csv"
        rm -f "$out/experts_rlens.pt" "$out/experts_none.pt"
    done
    ;;
rescore)
    for out in "$ROOT"/runs/*/n*; do
        [ -f "$out/j.pt" ] && [ -f "$out/r.pt" ] || continue
        [ -f "$out/baselines_filtered_pass_at_k.csv" ] && { echo "Skipping rescored $out"; continue; }
        cli evaluate --hf-model-name "$MODEL" --layers "$LAYERS" --correctness-csv "$CSV" \
            --lens "$out/j.pt" "$out/r.pt" logit --out "$out/baselines_filtered.csv"
    done
    ;;
gate)
    table="$ROOT/runs/$RUN/n$SIZE/jpp_scores_pass_at_k.csv"
    recall=$(awk -F, '$1 == "macro" && $2 == "jpp" && $3 == 10 && $4 == "item" {print 100 * $5}' "$table")
    echo "J++ recall@10: $recall (published $PUBLISHED)"
    awk -v r="$recall" -v p="$PUBLISHED" 'BEGIN {d = r - p; exit !(r != "" && d <= 5 && d >= -5)}'
    ;;
compare)
    out="${OUT:-$ROOT/scores/compare.csv}"
    pairs="${out%.csv}_pairs.txt"
    : > "$pairs"
    for size in 100 200 500 1000; do
        for lens in j r jpp; do
            add() { echo "$1_n${size}_$lens:$ROOT/runs/$2/n$size/$lens.pt:$ROOT/runs/$3/n$size/$lens.pt" >> "$pairs"; }
            add noise-wikitext wikitext-a wikitext-b
            add noise-en en-a en-b
            add noise-fr fr-a fr-b
            add language-a en-a fr-a
            add language-b en-b fr-b
            add corpus-a wikitext-a en-a
            add corpus-b wikitext-b en-b
            add mixed-a wikitext-a fr-a
            for pool in wikitext-a wikitext-b en-a en-b fr-a fr-b; do
                [ "$lens" = jpp ] && continue
                echo "method-${pool}_n${size}_$lens-jpp:$ROOT/runs/$pool/n$size/$lens.pt:$ROOT/runs/$pool/n$size/jpp.pt" >> "$pairs"
            done
            [ "$lens" = j ] && for pool in wikitext-a wikitext-b en-a en-b fr-a fr-b; do
                echo "method-${pool}_n${size}_j-r:$ROOT/runs/$pool/n$size/j.pt:$ROOT/runs/$pool/n$size/r.pt" >> "$pairs"
            done
            if [ "$size" != 1000 ]; then
                for pool in wikitext-a en-a fr-a; do
                    echo "converge-${pool}_n${size}_$lens:$ROOT/runs/$pool/n$size/$lens.pt:$ROOT/runs/$pool/n1000/$lens.pt" >> "$pairs"
                done
            fi
        done
    done
    run "$HELPERS/compare_lenses.py" --model "$MODEL" --residuals "$RESIDUALS" --pairs-file "$pairs" \
        --out "$out" --probe \
        "wikitext:$ROOT/pools/wikitext-probe.json" "en:$ROOT/pools/en-probe.json" "fr:$ROOT/pools/fr-probe.json"
    ;;
*)
    echo "Unknown STAGE=$STAGE"; exit 1
    ;;
esac

echo "Stage $STAGE finished at $(date)"
