#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
set -e
make main_gpu >/dev/null
echo "Census radiometric-robustness test  $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader|head -1; echo
# Run tiled SAD, SGM(SAD), SGM(census) with and without a right-image gain/bias.
probe() {
  local scene=$1 md=$2 scale=$3 gain=$4 bias=$5
  local L=data/middlebury/$scene/left.pgm R=data/middlebury/$scene/right.pgm G=data/middlebury/$scene/gt_disp.pgm
  echo "==== $scene  gain=$gain bias=$bias ===="
  out=$(./main_gpu --left $L --right $R --gt $G --gt-scale $scale --max-disp $md --radius 2 \
        --repeats 5 --no-cpu --right-gain $gain --right-bias $bias \
        --sgm --sgm-paths 4 2>&1)
  # SAD tiled block
  echo "$out" | awk '/GPU tiled\]/{f=1} f&&/Bad-px/{print "  SAD-tiled     bad =", $4"%"; f=0}'
  # SGM SAD block
  echo "$out" | awk '/GPU SGM.*SAD/{f=1} f&&/Bad-px/{print "  SGM(SAD)      bad =", $4"%"; f=0}'
  # now census
  out2=$(./main_gpu --left $L --right $R --gt $G --gt-scale $scale --max-disp $md --radius 2 \
        --repeats 5 --no-cpu --right-gain $gain --right-bias $bias \
        --sgm --sgm-paths 4 --sgm-census 2>&1)
  echo "$out2" | awk '/GPU SGM.*Census/{f=1} f&&/Bad-px/{print "  SGM(Census)   bad =", $4"%"; f=0}'
  echo
}
for scene_md_scale in "tsukuba 16 16" "venus 20 8"; do
  set -- $scene_md_scale
  probe $1 $2 $3 1.0 0     # baseline (matched)
  probe $1 $2 $3 1.4 15    # right camera brighter (gain+bias)
done
echo "Done $(date)"
