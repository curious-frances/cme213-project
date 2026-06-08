#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --job-name=dl-stereo
#
# Download Middlebury 2001 stereo pairs and prepare them as grayscale PGM files.
# The raw images are PPM (color); this script converts them with convert_png_to_pgm.py.
#
# Submit with:  sbatch download_stereo_data.sh
# Or run directly on a node with internet access: bash download_stereo_data.sh
#
# After running, use:
#   ./main_gpu --left data/middlebury/tsukuba/left.pgm \
#              --right data/middlebury/tsukuba/right.pgm \
#              --max-disp 16 --radius 2 --save-images --no-cpu
#
# For KITTI (PNG, requires registration at www.cvlibs.net/datasets/kitti):
#   python3 convert_png_to_pgm.py image_0/000000.png image_1/000000.png \
#           data/kitti/left.pgm data/kitti/right.pgm

set -e

BASE="https://vision.middlebury.edu/stereo/data/scenes2001/data"
DATADIR="data/middlebury"

mkdir -p "$DATADIR"

fetch() {
    local url="$1" out="$2"
    if [ -f "$out" ] && [ -s "$out" ]; then
        echo "  already have $out"
    else
        echo "  downloading $url"
        wget -q --show-progress -O "$out" "$url" || {
            echo "  ERROR fetching $url"; rm -f "$out"; return 1
        }
    fi
}

# -----------------------------------------------------------------
# Tsukuba (384x288): PPM multi-view; col3 = left, col4 = right
# GT disparity: truedisp.row3.col3.pgm
# Suggested max_disp=16
# -----------------------------------------------------------------
prepare_tsukuba() {
    local dir="$DATADIR/tsukuba"
    mkdir -p "$dir"
    fetch "$BASE/tsukuba/scene1.row3.col3.ppm" "$dir/imL.ppm"
    fetch "$BASE/tsukuba/scene1.row3.col4.ppm" "$dir/imR.ppm"
    fetch "$BASE/tsukuba/truedisp.row3.col3.pgm" "$dir/gt_disp.pgm" || true
    python3 convert_png_to_pgm.py "$dir/imL.ppm" "$dir/imR.ppm" \
            "$dir/left.pgm" "$dir/right.pgm"
    echo "  tsukuba ready: $dir/left.pgm  $dir/right.pgm"
}

# -----------------------------------------------------------------
# Sawtooth (434x380): im2 = left, im6 = right
# GT: disp2.pgm (disparity at im2)
# Suggested max_disp=20
# -----------------------------------------------------------------
prepare_sawtooth() {
    local dir="$DATADIR/sawtooth"
    mkdir -p "$dir"
    fetch "$BASE/sawtooth/im2.ppm" "$dir/imL.ppm"
    fetch "$BASE/sawtooth/im6.ppm" "$dir/imR.ppm"
    fetch "$BASE/sawtooth/disp2.pgm" "$dir/gt_disp.pgm" || true
    python3 convert_png_to_pgm.py "$dir/imL.ppm" "$dir/imR.ppm" \
            "$dir/left.pgm" "$dir/right.pgm"
    echo "  sawtooth ready: $dir/left.pgm  $dir/right.pgm"
}

# -----------------------------------------------------------------
# Venus (434x383): im2 = left, im6 = right
# GT: disp2.pgm
# Suggested max_disp=20
# -----------------------------------------------------------------
prepare_venus() {
    local dir="$DATADIR/venus"
    mkdir -p "$dir"
    fetch "$BASE/venus/im2.ppm" "$dir/imL.ppm"
    fetch "$BASE/venus/im6.ppm" "$dir/imR.ppm"
    fetch "$BASE/venus/disp2.pgm" "$dir/gt_disp.pgm" || true
    python3 convert_png_to_pgm.py "$dir/imL.ppm" "$dir/imR.ppm" \
            "$dir/left.pgm" "$dir/right.pgm"
    echo "  venus ready: $dir/left.pgm  $dir/right.pgm"
}

echo "=== Middlebury 2001 stereo data download ==="
echo "Target: $DATADIR/"
echo

prepare_tsukuba
echo
prepare_sawtooth
echo
prepare_venus

echo
echo "=== Summary ==="
for scene in tsukuba sawtooth venus; do
    f="$DATADIR/$scene/left.pgm"
    [ -f "$f" ] && { dims=$(head -2 "$f" | tr '\n' ' '); echo "  $scene: $dims"; } || echo "  $scene: MISSING"
done

echo
echo "=== Ready to run ==="
echo "Single-GPU:"
echo "  ./main_gpu --left $DATADIR/tsukuba/left.pgm --right $DATADIR/tsukuba/right.pgm \\"
echo "             --max-disp 16 --radius 2 --no-cpu --save-images"
echo
echo "MPI (4 ranks):"
echo "  mpirun -np 4 ./main_mpi \\"
echo "    --left $DATADIR/tsukuba/left.pgm --right $DATADIR/tsukuba/right.pgm \\"
echo "    --max-disp 16 --radius 2 --repeats 5 --save-images --output-prefix results/tsukuba"
