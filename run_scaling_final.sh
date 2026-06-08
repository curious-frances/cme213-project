#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:15:00
set -e
make main_mpi >/dev/null
RESULTS="results"; mkdir -p "$RESULTS"
STRONG="$RESULTS/strong_scaling.csv"; WEAK="$RESULTS/weak_scaling.csv"
rm -f "$STRONG" "$WEAK"
echo "Scaling study (Scatterv+halo)  $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -4
echo

echo "######## STRONG SCALING ########"
for dims in "960 1280" "1920 2560"; do
  set -- $dims; H=$1; W=$2
  for np in 1 2 4; do
    echo "--- strong ${H}x${W} np=$np ---"
    mpirun -np $np ./main_mpi --height $H --width $W --max-disp 64 --radius 2 \
           --repeats 20 --no-verify --csv "$STRONG"
  done
done

echo "######## WEAK SCALING (fixed 480 rows/rank, width 1280) ########"
for np in 1 2 4; do
  H=$((480*np))
  echo "--- weak ${H}x1280 np=$np (480 rows/rank) ---"
  mpirun -np $np ./main_mpi --height $H --width 1280 --max-disp 64 --radius 2 \
         --repeats 20 --no-verify --csv "$WEAK"
done

echo "######## MULTI-RANK CORRECTNESS (synthetic, verify on) ########"
for np in 1 2 4; do
  echo "--- verify 960x1280 np=$np ---"
  mpirun -np $np ./main_mpi --height 960 --width 1280 --max-disp 64 --radius 2 \
         --repeats 3 2>&1 | grep -A3 "Accuracy" || true
done

echo "=== STRONG ==="; cat "$STRONG"
echo "=== WEAK ===";   cat "$WEAK"
echo "Done $(date)"
