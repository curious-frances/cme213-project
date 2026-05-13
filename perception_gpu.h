#ifndef PERCEPTION_GPU_H_
#define PERCEPTION_GPU_H_

// ---------------------------------------------------------------------------
// perception_gpu.h
//
// Three CUDA implementations of stereo SAD disparity estimation:
//   1. Basic     — one thread per output pixel, all global memory
//   2. Smem      — block loads image tiles into shared memory
//   3. Tiled     — precomputes column-wise SAD sums to reduce redundant work
//
// All kernels use the same coordinate convention as sad_stereo_cpu:
//   Left patch centred at (r, c); right patch centred at (r, c - d).
// ---------------------------------------------------------------------------

#include "perception_common.h"

// Host wrappers — each allocates GPU memory, launches the kernel,
// copies results back, and returns timing in milliseconds.
// `repeats` warm + timed runs; only the last result is written to disp_out.

float sad_stereo_gpu_basic(const Image&  left,
                           const Image&  right,
                           DisparityMap& disp_out,
                           int           max_disp,
                           int           radius,
                           int           repeats);

float sad_stereo_gpu_smem(const Image&  left,
                          const Image&  right,
                          DisparityMap& disp_out,
                          int           max_disp,
                          int           radius,
                          int           repeats);

float sad_stereo_gpu_tiled(const Image&  left,
                           const Image&  right,
                           DisparityMap& disp_out,
                           int           max_disp,
                           int           radius,
                           int           repeats);

#endif  // PERCEPTION_GPU_H_
