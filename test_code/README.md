# Stereo SAD — CPU Baseline (CME 213 Final Project, Milestone 3)

## Overview

This directory contains the **single-GPU CPU reference implementation** for the
stereo Sum-of-Absolute-Differences (SAD) disparity estimator described in the
Milestone 2 design document.  It plays the same role as `gemm_test.cpp` and
`gtest` in the homework: an authoritative correctness reference and timing
baseline that the future CUDA kernel must match.

---

## File Structure

```
perception_common.h         Shared types (Image, DisparityMap, StereoParams),
                            macros, and tolerance constants — mirrors common.h
perception_cpu.h            Public API: data generation, SAD estimator, timing,
                            accuracy metrics, pretty-print helpers
perception_cpu.cpp          Full CPU implementation of everything in .h
main_cpu.cpp                CLI driver — mirrors main_q1.cu / main_q2.cu style
perception_cpu_test.cpp     GoogleTest suite — mirrors gemm_test.cpp style
Makefile                    Build system — mirrors the homework Makefile exactly
README.md                   This file
```

---

## Dependencies

| Requirement | Notes |
|---|---|
| GCC ≥ 9 or Clang ≥ 10 | C++17 required (`-std=c++17`) |
| GoogleTest source | Place at `./googletest-main` (same path as homework) |
| No OpenCV | All image data is generated synthetically |
| No CUDA | Pure CPU code; CUDA comes in Milestone 4 |

Clone GoogleTest if you don't have it:
```bash
git clone https://github.com/google/googletest.git googletest-main
```

---

## How to Build

```bash
# Build just the driver
make

# Build and run the GoogleTest suite
make test_cpu

# Build driver + run with default args (480×640, disp=24, max_disp=64, r=2)
make run

# Fast iteration during development
make run_small

# Remove all build artifacts
make clean
```

---

## How to Run

```
./main_cpu [options]

  --height   H    image height in pixels        (default: 480)
  --width    W    image width in pixels          (default: 640)
  --disp     D    ground-truth disparity (shift) (default: 24)
  --max-disp MD   disparity search range [0,MD)  (default: 64)
  --radius   R    SAD patch half-size            (default: 2)
  --repeats  N    number of timing repetitions   (default: 5)
```

### Example runs

```bash
# Default 480×640 benchmark
./main_cpu

# Larger image
./main_cpu --height 720 --width 1280 --disp 32 --max-disp 128 --radius 3 --repeats 10

# Tiny smoke test
./main_cpu --height 60 --width 80 --disp 4 --max-disp 16 --radius 1 --repeats 1
```

---

## Expected Output

```
====================================================
  Stereo SAD — CPU baseline
====================================================
  Image size   : 480 x 640
  Patch radius : 2  (patch = 5x5)
  Max disparity: 64
  True disparity (GT): 24
  Repeats      : 5
----------------------------------------------------
  [CPU SAD]
    Mean  :   412.371 ms  ±3.812 ms
    Min   :   407.119 ms
    FPS   :     2.425
  Accuracy (valid pixels only):
    MAE         : 0.0000 px
    Bad-px rate : 0.0000 %  (|err| > 1 px)
----------------------------------------------------
  Compute (approx): 14.155 GOps/frame
  Throughput      : 34.324 GOps/s  (CPU, single-threaded)
====================================================
```

**What the numbers mean:**

| Field | Meaning |
|---|---|
| Mean / Min | Wall-clock time per frame averaged / best over `--repeats` runs |
| FPS | Frames per second = 1000 / mean_ms |
| MAE | Mean absolute disparity error on pixels the GT marks valid |
| Bad-px rate | Fraction of valid pixels with \|estimated − GT\| > `DISP_TOL` (1 px) |
| Compute | Approximate ADD+ABS operations per frame |
| Throughput | GOps/s — compare against GPU peak to gauge headroom |

On a synthetic pair with a uniform shift, MAE and bad-pixel rate should both be
**exactly 0** because the texture is rich enough that SAD always finds the correct
disparity.

---

## Algorithm Details

### Synthetic stereo pair

The left image is generated with a deterministic multi-frequency sinusoidal
texture (two frequencies horizontally, two vertically).  This ensures every
local patch is unique, so SAD has a single, unambiguous minimum at the true
disparity.  The right image is the left image shifted left by `true_disp`
pixels; columns with no correspondence are filled with mid-grey (128).

### SAD disparity search

For each interior pixel `(r, c)` in the left image:

```
best_d = argmin_{d in [0, max_disp)} SAD(left[r,c], right[r, c-d], radius)

SAD(p, q, r) = sum_{dr=-r..r} sum_{dc=-r..r} |left[p_r+dr, p_c+dc]
                                               - right[q_r+dr, q_c+dc]|
```

Border pixels (within `radius` of any edge, or where the right patch would
go out of bounds) receive `BORDER_FILL` (0) and are excluded from accuracy
metrics via the ground-truth mask (GT = −1).

### Coordinate convention

```
Left pixel  (r, c)  ←→  Right pixel  (r, c − d)
```

This is the standard left-to-right convention: the right camera sees the scene
shifted left, so we search negative column offsets in the right image.

---

## Mapping to the Future CUDA Implementation

The CPU baseline is deliberately written so that porting to CUDA requires
minimal structural changes:

| CPU concept | CUDA equivalent |
|---|---|
| `Image` / `DisparityMap` | Row-major device buffers (`cudaMalloc`) |
| `sad_stereo_cpu(left, right, out, …)` | One CUDA kernel per implementation variant |
| `patch_sad(…)` inner loop | Per-thread computation; candidates for shared-memory tiling |
| `StereoParams` struct | Passed as kernel arguments or constant memory |
| `time_sad_stereo_cpu(…)` | `cudaEventRecord` / `cudaEventElapsedTime` |
| `compute_mae` / `compute_bad_pixel_rate` | Run on CPU after `cudaMemcpy` back to host |
| `generate_left_image` / `generate_right_image` | Reused unchanged on the host side |

The recommended CUDA optimisation sequence (mirrors the homework progression
from Q1 → Q2 → Q3):

1. **Naive kernel** — one thread per output pixel, global memory only
   (analogous to `basicGEMMRowMajorThreads`).
2. **Shared-memory tiling** — load image rows into shared memory to reduce
   redundant global reads across the patch loop
   (analogous to `sharedMemoryGEMM`).
3. **Hierarchical tiling + register caching** — load larger tiles, each thread
   computes multiple output pixels
   (analogous to `tiledGEMM`).

---

## Running the Test Suite

```bash
make test_cpu
```

All tests should pass with output similar to:

```
[==========] Running 14 tests from 3 test suites.
[----------] 4 tests from ImageGeneration
[ RUN      ] ImageGeneration.LeftImageIsInRange
[       OK ] ImageGeneration.LeftImageIsInRange (2 ms)
...
[==========] 14 tests from 3 test suites ran.
[  PASSED  ] 14 tests.
```

Run a specific suite:
```bash
./test_cpu --gtest_filter=StereoSAD.*
```

---

## Submitting

Include all `.h`, `.cpp`, and `Makefile` files in your Milestone 3 submission.
The progress report (≈2 pages) should reference the timing numbers printed by
`main_cpu` as the CPU baseline against which CUDA speedup is measured.
