#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:05:00
#SBATCH --job-name=verify-scatter

# Verify Scatterv+halo implementation:
#   1. Correctness: MAE=0, bad-px=0% on synthetic images (all rank counts)
#   2. Timing: scatter+halo should be cheaper than the old bcast
#   3. Real images: make sure pipeline still works end-to-end

set -e
make main_mpi

echo "=== Correctness: synthetic 480x640, 1/2/4 ranks ==="
for NP in 1 2 4; do
    echo "--- $NP rank(s) ---"
    mpirun -np $NP ./main_mpi \
        --height 480 --width 640 --disp 24 --max-disp 64 --radius 2 --repeats 3
done

echo
echo "=== Correctness: synthetic 960x1280, 4 ranks ==="
mpirun -np 4 ./main_mpi \
    --height 960 --width 1280 --disp 24 --max-disp 64 --radius 2 --repeats 3

echo
echo "=== Timing: 960x1280, 1/2/4 ranks (scatter vs old bcast) ==="
for NP in 1 2 4; do
    echo "--- $NP rank(s) ---"
    mpirun -np $NP ./main_mpi \
        --height 960 --width 1280 --disp 24 --max-disp 64 --radius 2 --repeats 10 --no-verify
done

echo
echo "=== Real images: motorcycle 1482x1000, 4 ranks ==="
mpirun -np 4 ./main_mpi \
    --left  data/middlebury/motorcycle/left.pgm \
    --right data/middlebury/motorcycle/right.pgm \
    --max-disp 64 --radius 2 --repeats 5

echo
echo "Done at $(date)"
