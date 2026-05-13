#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1

### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

echo "Starting at `date`"
echo
make main_cpu

echo
echo Output from main_cpu
echo ----------------
./main_cpu --height 480 --width 640 --disp 24 --max-disp 64 --radius 2 --repeats 5