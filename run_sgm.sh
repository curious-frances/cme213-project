#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#
# Compare 4-path GPU SGM against local SAD on Middlebury pairs (with ground
# truth). Two stages:
#   1. P1/P2 penalty sweep on tsukuba to pick good smoothness penalties.
#   2. Full SAD-vs-SGM comparison on tsukuba / sawtooth / venus at the chosen
#      penalties (each run also prints SAD basic/smem/tiled + LR-check).
#
# Read the "GPU SGM (4-path)" blocks against "GPU tiled" / "GPU tiled + LR-check"
# for accuracy (MAE, bad-px, coverage) and runtime.

set -e
make main_gpu

RESULTS="results"
mkdir -p "$RESULTS"
LOG="$RESULTS/sgm_compare.log"
: > "$LOG"

echo "SGM vs SAD comparison  $(date)" | tee -a "$LOG"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tee -a "$LOG"
echo | tee -a "$LOG"

TSU_L="data/middlebury/tsukuba/left.pgm"
TSU_R="data/middlebury/tsukuba/right.pgm"
TSU_GT="data/middlebury/tsukuba/gt_disp.pgm"

# ----------------------------------------------------------------------
# 1. P1/P2 sweep on tsukuba (--no-cpu, SGM only relevance shown by SGM block)
# ----------------------------------------------------------------------
echo "######## P1/P2 SWEEP (tsukuba, max_disp=16) ########" | tee -a "$LOG"
for p1 in 20 50 80; do
  for p2 in 400 800 1600; do
    echo "---- P1=$p1 P2=$p2 ----" | tee -a "$LOG"
    ./main_gpu --left "$TSU_L" --right "$TSU_R" --gt "$TSU_GT" --gt-scale 16 \
               --max-disp 16 --radius 2 --repeats 10 --no-cpu \
               --sgm --p1 $p1 --p2 $p2 2>&1 \
        | grep -A4 "GPU SGM" | tee -a "$LOG"
    echo | tee -a "$LOG"
  done
done

# ----------------------------------------------------------------------
# 2. Full comparison at chosen penalties (P1=200, P2=500).
# ----------------------------------------------------------------------
compare_scene() {
    local scene="$1" md="$2" scale="$3"
    local left="data/middlebury/$scene/left.pgm"
    local right="data/middlebury/$scene/right.pgm"
    local gt="data/middlebury/$scene/gt_disp.pgm"
    [ -f "$left" ] || { echo "SKIP $scene"; return; }
    echo "######## FULL COMPARE: $scene (max_disp=$md) ########" | tee -a "$LOG"
    ./main_gpu --left "$left" --right "$right" --gt "$gt" --gt-scale "$scale" \
               --max-disp "$md" --radius 2 --repeats 10 --no-cpu \
               --lr-check --sgm --p1 200 --p2 500 \
               --save-images --csv "$RESULTS/sgm_compare.csv" | tee -a "$LOG"
    for ext in pgm ppm; do
        [ -f "disp_tiled.$ext" ]     && mv "disp_tiled.$ext"     "$RESULTS/${scene}_tiled.$ext"
        [ -f "disp_tiled_lrc.$ext" ] && mv "disp_tiled_lrc.$ext" "$RESULTS/${scene}_tiled_lrc.$ext"
        [ -f "disp_sgm.$ext" ]       && mv "disp_sgm.$ext"       "$RESULTS/${scene}_sgm.$ext"
        [ -f "disp_gt.$ext" ]        && mv "disp_gt.$ext"        "$RESULTS/${scene}_gt.$ext"
    done
    echo | tee -a "$LOG"
}

compare_scene tsukuba  16 16
compare_scene sawtooth 20 8
compare_scene venus    20 8

echo "=== Done. Full log: $LOG ===" | tee -a "$LOG"
