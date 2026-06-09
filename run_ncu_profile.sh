#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#
# Nsight Compute kernel profiling. Produces .ncu-rep report files that you open
# in the Nsight Compute GUI (File -> Open, or `ncu-ui results/sad_kernels.ncu-rep`)
# to inspect roofline, memory throughput, occupancy, and warp-stall reasons.
#
# Same pattern as homework 7: `ncu -o <report> -f --set full --kernel-name ...`.
# We profile a small 480x640 image with --repeats 1 so the full metric replay
# stays within the cluster's time budget.

set -e
make main_gpu >/dev/null

# Nsight Compute needs a writable lock dir (see hw7_q1_profile.sh).
export TMPDIR=/tmp/nsight-compute-lock-$USER
mkdir -p "$TMPDIR"

NCU=ncu
command -v ncu >/dev/null 2>&1 || \
  NCU=/home/cme213/software/nvidia-hpc-sdk/2024_24.1/Linux_x86_64/24.1/profilers/Nsight_Compute/ncu

RESULTS=results
mkdir -p "$RESULTS"
ARGS="--height 480 --width 640 --max-disp 64 --radius 2 --repeats 1 --no-cpu"

echo "Nsight Compute profiling  $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# --- SAD kernels: basic vs shared-memory vs register-tiled ---------------
# -c 6 captures the warm-up + timed launch of each of the three kernels.
echo "Profiling SAD kernels -> $RESULTS/sad_kernels.ncu-rep"
$NCU -o "$RESULTS/sad_kernels" -f --set full -c 6 \
     --kernel-name "regex:kernel_basic|kernel_smem|kernel_tiled" \
     ./main_gpu $ARGS

# --- SGM kernels: cost-volume build + path aggregation -------------------
echo "Profiling SGM kernels -> $RESULTS/sgm_kernels.ncu-rep"
$NCU -o "$RESULTS/sgm_kernels" -f --set full -c 8 \
     --kernel-name "regex:sgm_cost_sad_kernel|sgm_aggregate" \
     ./main_gpu $ARGS --sgm --sgm-paths 4

echo
echo "Done. Open in the Nsight Compute GUI:"
echo "  ncu-ui $RESULTS/sad_kernels.ncu-rep"
echo "  ncu-ui $RESULTS/sgm_kernels.ncu-rep"
echo "Or dump a text summary on the cluster:"
echo "  ncu --import $RESULTS/sad_kernels.ncu-rep --page details | less"
