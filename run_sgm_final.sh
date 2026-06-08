#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
set -e
RESULTS="results"; mkdir -p "$RESULTS"
LOG="$RESULTS/sgm_compare.log"; : > "$LOG"
CSV="$RESULTS/sgm_compare.csv"; rm -f "$CSV"
echo "SGM vs SAD final comparison (P1=200 P2=500)  $(date)" | tee -a "$LOG"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tee -a "$LOG"
compare_scene() {
    local scene="$1" md="$2" scale="$3"
    local left="data/middlebury/$scene/left.pgm" right="data/middlebury/$scene/right.pgm" gt="data/middlebury/$scene/gt_disp.pgm"
    [ -f "$left" ] || { echo "SKIP $scene"; return; }
    echo "######## $scene (max_disp=$md) ########" | tee -a "$LOG"
    ./main_gpu --left "$left" --right "$right" --gt "$gt" --gt-scale "$scale" \
               --max-disp "$md" --radius 2 --repeats 10 --no-cpu \
               --lr-check --sgm --p1 200 --p2 500 \
               --save-images --csv "$CSV" | tee -a "$LOG"
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
echo "=== Done ===" | tee -a "$LOG"
