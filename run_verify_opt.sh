#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#
# Verify the SAD optimizations added for the final report:
#   (A) pinned host memory + async H2D/D2H copies (GpuBuffers)
#   (B) sub-pixel parabolic disparity refinement (kernel_tiled)
#   (C) left-right consistency check (sad_stereo_gpu_tiled_lrc, --lr-check)
#
# Two things to check in the output:
#   1. SYNTHETIC: MAE must stay ~0 for every kernel. Sub-pixel refinement is
#      exact/symmetric at an integer minimum, so the constant true disparity
#      must be recovered to ~0 error, and the LR-check must keep ~100% coverage.
#   2. REAL (Middlebury, has ground truth): the LR-check should LOWER the
#      bad-pixel rate (rejecting occlusions/mismatches) at the cost of some
#      coverage. Compare "GPU tiled" vs "GPU tiled + LR-check" blocks.

set -e

make main_gpu

RESULTS="results"
mkdir -p "$RESULTS"
LOG="$RESULTS/verify_opt.log"
: > "$LOG"

echo "Verify SAD optimizations  $(date)"  | tee -a "$LOG"
echo "Node: $(hostname)"                   | tee -a "$LOG"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tee -a "$LOG"
echo                                       | tee -a "$LOG"

# ----------------------------------------------------------------------
# 1. Synthetic correctness — MAE must be ~0 (sub-pixel preserves exactness),
#    LR-check coverage must be ~100% (all matches mutually consistent).
# ----------------------------------------------------------------------
echo "======== SYNTHETIC (expect MAE ~0, coverage ~100%) ========" | tee -a "$LOG"
./main_gpu --height 480 --width 640 --disp 24 --max-disp 64 --radius 2 \
           --repeats 10 --no-cpu --lr-check | tee -a "$LOG"

# ----------------------------------------------------------------------
# 2. Real images with ground truth — LR-check should reduce bad-pixel rate.
# ----------------------------------------------------------------------
# Middlebury 2001 GT gray-to-disparity scale: tsukuba=16, sawtooth/venus=8.
run_scene() {
    local scene="$1" md="$2" scale="$3"
    local left="data/middlebury/$scene/left.pgm"
    local right="data/middlebury/$scene/right.pgm"
    local gt="data/middlebury/$scene/gt_disp.pgm"
    if [ ! -f "$left" ]; then
        echo "SKIP $scene — images not found (run download_stereo_data.sh)" | tee -a "$LOG"
        return
    fi
    echo "======== REAL: $scene (max_disp=$md, gt-scale=$scale) ========" | tee -a "$LOG"
    ./main_gpu --left "$left" --right "$right" --max-disp "$md" --radius 2 \
               --gt "$gt" --gt-scale "$scale" \
               --repeats 10 --no-cpu --lr-check --save-images \
               --csv "$RESULTS/verify_opt.csv" | tee -a "$LOG"
    for ext in pgm ppm; do
        [ -f "disp_tiled.$ext" ]     && mv "disp_tiled.$ext"     "$RESULTS/${scene}_tiled.$ext"
        [ -f "disp_tiled_lrc.$ext" ] && mv "disp_tiled_lrc.$ext" "$RESULTS/${scene}_tiled_lrc.$ext"
    done
    echo | tee -a "$LOG"
}

run_scene tsukuba  16 16
run_scene sawtooth 20 8
run_scene venus    20 8

echo "=== Done. Full log: $LOG ===" | tee -a "$LOG"
