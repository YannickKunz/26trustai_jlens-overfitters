#!/bin/bash
#SBATCH --job-name=lang_extras
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

# The GPU jobs that remain after slurm_language_transfer.sh. Nothing is fitted.
# Submit from the repo root, with the same MODEL, LAYERS and ROOT as the sweep:
#   sbatch scripts/ubelix/slurm_language_extras.sh                 all stages, in order
#   STAGE="baseline" sbatch scripts/ubelix/slurm_language_extras.sh
# Stages (each skips finished work, so a requeue or a resubmission is safe):
#   baseline  ranks of every target word at unrelated prompts     -> $ROOT/lang/baseline_ranks.csv
#   compare   cosine, Frobenius and KL for the two new J++ lenses -> $ROOT/scores/compare_weights.csv
#   package   every CSV the local analysis needs, in one archive  -> $ROOT/language_transfer_results.tar.gz
# baseline must be rerun (delete its CSV) whenever items.json changes.
set -euo pipefail

STAGE="${STAGE:-baseline compare package}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
LAYERS="${LAYERS:-4,8,12,16,20,24,28}"
RUNS="${RUNS:-wikitext-a wikitext-b en-a en-b fr-a fr-b}"
SIZE="${SIZE:-1000}"
REPO="$(pwd)"
SCRATCH_DIR="/storage/scratch/users/$USER"
SIF="${SIF:-$SCRATCH_DIR/jlens_paper.sif}"
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
ROOT="${ROOT:-$SCRATCH_DIR/jlens_paper/9b}"
JPP="jlens_jpp_reproduction/jpp_lens"
HELPERS="/work/scripts/ubelix"
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

for stage in $STAGE; do
case "$stage" in
baseline)
    [ -f "$LANG_DIR/baseline_ranks.csv" ] || run "$HELPERS/score_language_baseline.py" --model "$MODEL" \
        --layers "$LAYERS" --runs "$ROOT/runs" --run-names "$(echo $RUNS | tr ' ' ',')" --size "$SIZE" \
        --lang "$LANG_DIR" --out "$LANG_DIR/baseline_ranks.csv"
    ;;
compare)
    out="$ROOT/scores/compare_weights.csv"
    pairs="${out%.csv}_pairs.txt"
    if [ -f "$out" ]; then
        echo "Skipping compare: $out exists"
    else
        : > "$pairs"
        lens_path() { echo "$ROOT/runs/$1/n$SIZE/$2.pt"; }
        # name, run and lens of each side. Names keep the sweep's <kind>_n<size>_<lens> pattern.
        add() { echo "$1_n${SIZE}_$2:$(lens_path "$3" "$4"):$(lens_path "$5" "$6")" >> "$pairs"; }
        for lens in jpp_wen jpp_wfr; do
            # The sweep's families, for each weight language
            add noise-wikitext "$lens" wikitext-a "$lens" wikitext-b "$lens"
            add noise-en "$lens" en-a "$lens" en-b "$lens"
            add noise-fr "$lens" fr-a "$lens" fr-b "$lens"
            add language-a "$lens" en-a "$lens" fr-a "$lens"
            add language-b "$lens" en-b "$lens" fr-b "$lens"
            add corpus-a "$lens" wikitext-a "$lens" en-a "$lens"
            add corpus-b "$lens" wikitext-b "$lens" en-b "$lens"
        done
        # Matched weights: English corpus with English labels against French corpus with French labels
        add language-a jpp_matched en-a jpp_wen fr-a jpp_wfr
        add language-b jpp_matched en-b jpp_wen fr-b jpp_wfr
        for name in $RUNS; do
            # Same expert Jacobians, only the label language of the weights differs
            add "weights-$name" jpp_wen-jpp_wfr "$name" jpp_wen "$name" jpp_wfr
            # Same expert Jacobians, the sweep's original weights against the English-label ones
            add "weights-$name" jpp-jpp_wen "$name" jpp "$name" jpp_wen
        done
        run "$HELPERS/compare_lenses.py" --model "$MODEL" --residuals "$ROOT/residuals.pt" --pairs-file "$pairs" \
            --out "$out" --probe \
            "wikitext:$ROOT/pools/wikitext-probe.json" "en:$ROOT/pools/en-probe.json" "fr:$ROOT/pools/fr-probe.json"
    fi
    ;;
package)
    archive="$ROOT/language_transfer_results.tar.gz"
    files=""
    for path in lang/items.json lang/correctness_en.csv lang/correctness_fr.csv lang/ranks.csv \
                lang/ranks_dropped.csv lang/baseline_ranks.csv lang/analysis \
                scores/compare.csv scores/compare_weights.csv; do
        if [ -e "$ROOT/$path" ]; then files="$files $path"; else echo "WARNING: $ROOT/$path is missing"; fi
    done
    tar -czf "$archive" -C "$ROOT" $files
    echo "Wrote $archive ($(du -h "$archive" | cut -f1)):"
    tar -tzf "$archive"
    ;;
*)
    echo "Unknown stage $stage"; exit 1
    ;;
esac
echo "Stage $stage finished at $(date)"
done
