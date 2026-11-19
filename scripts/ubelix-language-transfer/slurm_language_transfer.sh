#!/bin/bash
#SBATCH --job-name=lang_transfer
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=88G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_lang_%x_%A.log
#SBATCH --error=slurm_lang_%x_%A.err

# Language-transfer evaluation on the lenses slurm_fit_lenses.sh already fitted. No lens is refitted;
# only the J++ expert weights are fitted again, on English and on French labels.
# Submit from the repo root, with the helper files in scripts/ubelix-language-transfer/:
#   sbatch scripts/ubelix-language-transfer/slurm_language_transfer.sh                 all stages, in order
#   STAGE="score analyze" sbatch scripts/ubelix-language-transfer/slurm_language_transfer.sh
# Stages (each skips finished work, so a requeue or a resubmission is safe):
#   prepare   the bilingual benchmark and its fitting/held-out split   -> $ROOT/lang/items.json, weights_{en,fr}/
#   grade     the model's own answers in each prompt language          -> $ROOT/lang/correctness_{en,fr}.csv
#   weights   J++ expert weights on English and on French labels       -> runs/<run>/n$SIZE/jpp_wen.pt, jpp_wfr.pt
#   score     every lens under prompt language x target language       -> $ROOT/lang/ranks.csv
#   analyze   tables with bootstrap intervals                          -> $ROOT/lang/analysis/summary.md
# To rebuild after editing parallel_items.json, delete $ROOT/lang, runs/*/n$SIZE/jpp_w* and
# runs/*/n$SIZE/residuals_weights_*.pt first (the residual cache is keyed on item names, not prompts).
set -euo pipefail

STAGE="${STAGE:-prepare grade weights score analyze}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
LAYERS="${LAYERS:-4,8,12,16,20,24,28}"
RUNS="${RUNS:-wikitext-a wikitext-b en-a en-b fr-a fr-b}"
SIZE="${SIZE:-1000}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"   # the size the shards were fitted with
REPO="$(pwd)"
SCRATCH_DIR="/storage/scratch/users/$USER"
SIF="${SIF:-$SCRATCH_DIR/jlens_paper.sif}"
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
ROOT="${ROOT:-$SCRATCH_DIR/jlens_paper/9b}"
JPP="${JPP:-jlens_jpp_reproduction/jpp_lens}"
HELPERS="${HELPERS:-/work/scripts/ubelix-language-transfer}"
LANG_DIR="$ROOT/lang"

echo "Node: $(hostname)  Date: $(date)  Stages: $STAGE  Restarts: ${SLURM_RESTART_COUNT:-0}"

run() {
    apptainer exec --nv \
        --bind "$REPO:/work" \
        --bind "$SCRATCH_DIR:$SCRATCH_DIR" \
        --env HF_HOME="$HF_HOME" \
        --env PYTHONPATH="/work/$JPP/src" \
        --pwd "/work/$JPP" \
        "$SIF" python "$@"
}
cli() { run scripts/jpp_cli.py "$@"; }

# Finished shard dirs for shards 0..$2-1 of the fit root $1, in order (as in slurm_fit_lenses.sh)
shard_dirs() {
    for shard in $(seq 0 $(($2 - 1))); do
        find "$1/" -type f -path "*/shard${shard}of${NUM_SHARDS}/fit_shard_stamp.json" -printf '%h\n' | head -1
    done
}

for stage in $STAGE; do
case "$stage" in
prepare)
    if [ -f "$LANG_DIR/items.json" ]; then
        echo "Skipping prepare: $LANG_DIR/items.json exists"
    else
        run "$HELPERS/build_language_evals.py" --french "$HELPERS/parallel_items.json" \
            --upstream-stamps $(ls "$ROOT"/runs/*/n"$SIZE"/jpp_stamp.json) --out "$LANG_DIR"
    fi
    ;;
grade)
    [ -f "$LANG_DIR/correctness_fr.csv" ] || run "$HELPERS/grade_language_items.py" \
        --model "$MODEL" --layers "$LAYERS" --lang "$LANG_DIR" --ignore-case
    ;;
weights)
    for name in $RUNS; do
        out="$ROOT/runs/$name/n$SIZE"
        [ -f "$out/jpp_wen.pt" ] && [ -f "$out/jpp_wfr.pt" ] && { echo "Skipping weighted $name"; continue; }
        experts="$out/experts_rlens.pt"
        merged_here=0
        if [ ! -f "$experts" ]; then
            merged_here=1
            # slurm_fit_lenses.sh deletes the merged experts after scoring: merge the shards again
            NUM_SHARDS="$(find "$ROOT/runs/$name/rlens/" -type d -name 'shard0of*' | head -1 | sed 's/.*shard0of//')"
            [ -n "$NUM_SHARDS" ] || { echo "No rlens shards under $ROOT/runs/$name"; exit 1; }
            cli merge-experts --checkpoint-dirs $(shard_dirs "$ROOT/runs/$name/rlens" $((SIZE * NUM_SHARDS / NUM_PROMPTS))) \
                --out "$experts"
        fi
        for language in en fr; do
            [ -f "$out/jpp_w$language.pt" ] && continue
            # --items all: every item of the folder, which holds the fitting split only
            cli fit-weights --hf-model-name "$MODEL" --layers "$LAYERS" --experts "$experts" \
                --eval-data-dir "$LANG_DIR/weights_$language" --correctness-csv "$LANG_DIR/correctness_$language.csv" \
                --items all --cache-path "$out/residuals_weights_$language.pt" \
                --checkpoint-name "jpp_w$language" --out "$out/jpp_w$language.pt"
        done
        if [ "$merged_here" = 1 ]; then rm -f "$experts"; fi
    done
    ;;
score)
    [ -f "$LANG_DIR/ranks.csv" ] || run "$HELPERS/score_language_matrix.py" --model "$MODEL" --layers "$LAYERS" \
        --runs "$ROOT/runs" --run-names "$(echo $RUNS | tr ' ' ',')" --size "$SIZE" \
        --lang "$LANG_DIR" --out "$LANG_DIR/ranks.csv"
    ;;
analyze)
    run "$HELPERS/analyze_language_transfer.py" --ranks "$LANG_DIR/ranks.csv" --lang "$LANG_DIR" \
        --out "$LANG_DIR/analysis"
    # recall@100 does not saturate on French targets the way recall@10 does
    run "$HELPERS/analyze_language_transfer.py" --ranks "$LANG_DIR/ranks.csv" --lang "$LANG_DIR" \
        --out "$LANG_DIR/analysis_k100" --k 100
    ;;
*)
    echo "Unknown stage $stage"; exit 1
    ;;
esac
echo "Stage $stage finished at $(date)"
done
