#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00

echo "Starting MPI scaling study at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo

make main_mpi

CSV=scaling.csv
rm -f $CSV

echo "=== Strong scaling: 480x640, max_disp=64, radius=2 ==="
for NP in 1 2 4; do
    echo "--- nranks=$NP ---"
    mpirun -np $NP ./main_mpi \
        --height 480 --width 640 --disp 24 --max-disp 64 --radius 2 --repeats 10 \
        --csv $CSV
done

echo
echo "=== Larger image: 960x1280 ==="
for NP in 1 2 4; do
    echo "--- nranks=$NP ---"
    mpirun -np $NP ./main_mpi \
        --height 960 --width 1280 --disp 24 --max-disp 64 --radius 2 --repeats 10 \
        --csv $CSV
done

echo
echo "Scaling CSV written to $CSV"
echo "Done at $(date)"
