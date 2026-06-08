#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
set -e
make main_gpu >/dev/null
L=data/middlebury/tsukuba/left.pgm; R=data/middlebury/tsukuba/right.pgm; G=data/middlebury/tsukuba/gt_disp.pgm
echo "SGM P1/P2 sweep (tsukuba, max_disp=16)  $(date)"
printf "%-6s %-6s %-10s %-10s\n" P1 P2 MAE bad%
for p1 in 20 50 80 120; do for p2 in 400 800 1200 2000; do
  out=$(./main_gpu --left $L --right $R --gt $G --gt-scale 16 --max-disp 16 --radius 2 \
        --repeats 5 --no-cpu --sgm --p1 $p1 --p2 $p2 2>&1)
  mae=$(echo "$out" | awk '/GPU SGM/{f=1} f&&/MAE/{print $3; exit}')
  bad=$(echo "$out" | awk '/GPU SGM/{f=1} f&&/Bad-px/{print $4; exit}')
  printf "%-6s %-6s %-10s %-10s\n" $p1 $p2 $mae $bad
done; done
