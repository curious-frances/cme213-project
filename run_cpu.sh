#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1

### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

echo "Starting at `date`"
echo
make main_cpu main_gpu

echo
echo Output from main_cpu
echo ----------------
./main_cpu --height 480 --width 640 --disp 24 --max-disp 64 --radius 2 --repeats 5 --save-images

echo
echo Output from main_gpu
echo ----------------
./main_gpu --height 480 --width 640 --disp 24 --max-disp 64 --radius 2 --repeats 5 --save-images --csv benchmark.csv
