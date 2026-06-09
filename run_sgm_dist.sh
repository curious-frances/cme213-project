#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:15:00
set -e
make main_mpi main_gpu >/dev/null
RESULTS="results"; mkdir -p "$RESULTS"
echo "Distributed SGM test  $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader|head -4; echo

echo "######## CORRECTNESS: synthetic 960x1280 SGM, verify across ranks ########"
for np in 1 2 4; do
  echo "--- np=$np ---"
  mpirun -np $np ./main_mpi --height 960 --width 1280 --max-disp 64 --radius 2 --sgm \
    2>&1 | grep -A3 "Accuracy" || true
done

echo "######## EXACTNESS: distributed SGM np=4 must equal np=1 (real tsukuba) ########"
TL=data/middlebury/tsukuba/left.pgm; TR=data/middlebury/tsukuba/right.pgm
mpirun -np 1 ./main_mpi --left $TL --right $TR --max-disp 16 --radius 2 --sgm \
   --save-images --output-prefix "$RESULTS/tsukuba_sgm_np1" >/dev/null 2>&1
mpirun -np 4 ./main_mpi --left $TL --right $TR --max-disp 16 --radius 2 --sgm \
   --save-images --output-prefix "$RESULTS/tsukuba_sgm_np4" >/dev/null 2>&1
if cmp -s "$RESULTS/tsukuba_sgm_np1.pgm" "$RESULTS/tsukuba_sgm_np4.pgm"; then
  echo "  PASS: np1 and np4 disparity maps are byte-identical (frontier exchange exact)"
else
  echo "  DIFFER: np1 vs np4"; cmp "$RESULTS/tsukuba_sgm_np1.pgm" "$RESULTS/tsukuba_sgm_np4.pgm" | head
fi

echo "######## STRONG SCALING: distributed SGM ########"
SGMCSV="$RESULTS/sgm_scaling.csv"; rm -f "$SGMCSV"
for dims in "960 1280" "1920 2560"; do
  set -- $dims; H=$1; W=$2
  for np in 1 2 4; do
    echo "--- SGM ${H}x${W} np=$np ---"
    mpirun -np $np ./main_mpi --height $H --width $W --max-disp 64 --radius 2 --sgm \
      --no-verify --csv "$SGMCSV"
  done
done
echo "=== SGM scaling CSV ==="; cat "$SGMCSV"
echo "Done $(date)"
