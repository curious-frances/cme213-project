#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:20:00
#
# Run the stereo SAD pipeline on real Middlebury image pairs (small + large)
# and on synthetic images at large sizes for scaling comparison.
# Prerequisites: run sbatch download_stereo_data.sh first.

set -e

make main_gpu main_mpi

RESULTS="results"
mkdir -p "$RESULTS"

# -----------------------------------------------------------------------
# Helper: run single-GPU benchmark on a scene and save color disparity
# -----------------------------------------------------------------------
run_gpu_scene() {
    local scene="$1"
    local max_disp="$2"
    local radius="${3:-2}"
    local left="data/middlebury/$scene/left.pgm"
    local right="data/middlebury/$scene/right.pgm"

    if [ ! -f "$left" ]; then
        echo "SKIP $scene — images not found (run download_stereo_data.sh)"
        return
    fi

    echo "=== GPU: $scene (max_disp=$max_disp, radius=$radius) ==="
    ./main_gpu \
        --left "$left" --right "$right" \
        --max-disp "$max_disp" --radius "$radius" \
        --repeats 10 --no-cpu \
        --save-images \
        --csv "$RESULTS/real_images_gpu.csv"
    # save-images writes to disp_tiled.pgm/.ppm in cwd; rename by scene
    for ext in pgm ppm; do
        [ -f "disp_tiled.$ext" ] && mv "disp_tiled.$ext" "$RESULTS/${scene}_tiled.$ext"
        [ -f "disp_gt.$ext"    ] && mv "disp_gt.$ext"    "$RESULTS/${scene}_gt.$ext"
    done
    echo "  → $RESULTS/${scene}_tiled.pgm/.ppm"
}

# -----------------------------------------------------------------------
# Helper: run MPI pipeline on a scene
# -----------------------------------------------------------------------
run_mpi_scene() {
    local scene="$1"
    local max_disp="$2"
    local nranks="${3:-4}"
    local radius="${4:-2}"
    local left="data/middlebury/$scene/left.pgm"
    local right="data/middlebury/$scene/right.pgm"

    if [ ! -f "$left" ]; then
        echo "SKIP $scene MPI — images not found"
        return
    fi

    echo "=== MPI ($nranks ranks): $scene (max_disp=$max_disp) ==="
    mpirun -np "$nranks" ./main_mpi \
        --left "$left" --right "$right" \
        --max-disp "$max_disp" --radius "$radius" \
        --repeats 5 \
        --save-images \
        --output-prefix "$RESULTS/${scene}_mpi_np${nranks}" \
        --csv "$RESULTS/real_images_mpi.csv"
    echo "  → $RESULTS/${scene}_mpi_np${nranks}.pgm/.ppm"
}

# -----------------------------------------------------------------------
# Helper: run single-GPU on a synthetic image at given size
# -----------------------------------------------------------------------
run_gpu_synthetic() {
    local label="$1"
    local height="$2"
    local width="$3"
    local max_disp="${4:-64}"
    local radius="${5:-2}"

    echo "=== GPU synthetic: ${height}x${width} (max_disp=$max_disp) ==="
    ./main_gpu \
        --height "$height" --width "$width" \
        --disp 24 --max-disp "$max_disp" --radius "$radius" \
        --repeats 10 --no-cpu \
        --csv "$RESULTS/synthetic_gpu.csv"
    echo
}

run_mpi_synthetic() {
    local height="$1"
    local width="$2"
    local nranks="${3:-4}"
    local max_disp="${4:-64}"

    echo "=== MPI ($nranks ranks) synthetic: ${height}x${width} (max_disp=$max_disp) ==="
    mpirun -np "$nranks" ./main_mpi \
        --height "$height" --width "$width" \
        --disp 24 --max-disp "$max_disp" --radius 2 \
        --repeats 5 \
        --csv "$RESULTS/synthetic_mpi.csv"
    echo
}

# -----------------------------------------------------------------------
# Run scenes
# -----------------------------------------------------------------------
echo "Starting real-image stereo evaluation at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -4
echo

echo "======== Small real images (Middlebury 2001) ========"
run_gpu_scene  tsukuba  16
run_gpu_scene  sawtooth 20
run_gpu_scene  venus    20

run_mpi_scene  tsukuba  16 4
run_mpi_scene  venus    20 4

echo "======== Large real image (Middlebury 2014, ~1482x1000) ========"
run_gpu_scene  motorcycle 64
run_mpi_scene  motorcycle 64 1
run_mpi_scene  motorcycle 64 2
run_mpi_scene  motorcycle 64 4

echo "======== Synthetic large images ========"
run_gpu_synthetic "960x1280"   960  1280 64
run_gpu_synthetic "1920x2560" 1920 2560 64

for nranks in 1 2 4; do
    run_mpi_synthetic 960  1280 $nranks 64
done
for nranks in 1 2 4; do
    run_mpi_synthetic 1920 2560 $nranks 64
done

echo
echo "=== All results in $RESULTS/ ==="
ls -lh "$RESULTS/"
echo "Done at $(date)"
