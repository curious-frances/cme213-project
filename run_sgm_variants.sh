#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
set -e
make main_gpu >/dev/null
RESULTS="results"; mkdir -p "$RESULTS"; CSV="$RESULTS/sgm_variants.csv"; rm -f "$CSV"
echo "SGM variant study (paths x cost)  $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader|head -1; echo
run() {
  local scene=$1 md=$2 scale=$3
  local L=data/middlebury/$scene/left.pgm R=data/middlebury/$scene/right.pgm G=data/middlebury/$scene/gt_disp.pgm
  echo "######## $scene (D=$md) ########"
  for paths in 4 8; do
    for cost in sad census; do
      flag=""; [ "$cost" = census ] && flag="--sgm-census"
      out=$(./main_gpu --left $L --right $R --gt $G --gt-scale $scale --max-disp $md --radius 2 \
            --repeats 10 --no-cpu --sgm --sgm-paths $paths $flag --csv "$CSV" 2>&1)
      mean=$(echo "$out" | awk '/GPU SGM/{f=1} f&&/Mean/{print $3; exit}')
      mae=$(echo "$out"  | awk '/GPU SGM/{f=1} f&&/MAE/{print $3; exit}')
      bad=$(echo "$out"  | awk '/GPU SGM/{f=1} f&&/Bad-px/{print $4; exit}')
      printf "  %d-path %-7s : %7s ms   MAE %-7s  bad %s%%\n" $paths $cost $mean $mae $bad
    done
  done
  echo
}
run tsukuba 16 16
run sawtooth 20 8
run venus 20 8
echo "Done $(date)"
