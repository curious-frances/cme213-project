// ---------------------------------------------------------------------------
// perception_gpu.cu
//
// Three CUDA stereo SAD implementations.
// ---------------------------------------------------------------------------

#include "perception_gpu.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d — %s\n",                         \
              __FILE__, __LINE__, cudaGetErrorString(_e));                \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// ============================================================================
// 1. BASIC KERNEL — global memory only, one thread per output pixel
// ============================================================================

__global__ void kernel_basic(const uint8_t* __restrict__ left,
                             const uint8_t* __restrict__ right,
                             int*           disp_out,
                             int            H, int W,
                             int            max_disp, int radius) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= H || c >= W) return;

  bool at_border = (r < radius) || (r >= H - radius) ||
                   (c < radius) || (c >= W - radius);
  bool right_oob = (c - (max_disp - 1) - radius < 0);

  if (at_border || right_oob) {
    disp_out[r * W + c] = 0;
    return;
  }

  unsigned int best_sad = 0xFFFFFFFFu;
  int          best_d   = 0;

  for (int d = 0; d < max_disp; ++d) {
    if (c - d - radius < 0) break;

    unsigned int sad = 0;
    for (int dr = -radius; dr <= radius; ++dr) {
      for (int dc = -radius; dc <= radius; ++dc) {
        int lv = left [(r + dr) * W + (c + dc)];
        int rv = right[(r + dr) * W + (c + dc - d)];
        sad += (unsigned int)abs(lv - rv);
      }
    }
    if (sad < best_sad) { best_sad = sad; best_d = d; }
  }
  disp_out[r * W + c] = best_d;
}

// ============================================================================
// 2. SHARED MEMORY KERNEL
//
// Each block loads a tile of the LEFT image (BH+2R) x (BW+2R) and
// a wider tile of the RIGHT image (BH+2R) x (BW+2R+max_disp) into smem,
// covering all right patches for every disparity a thread in this block needs.
// ============================================================================

// Block dimensions for smem kernel
#define SMEM_BW 16
#define SMEM_BH 16

// Shared memory arrays — max_disp ≤ 128, radius ≤ 8 assumed for sizing.
// Left tile:  (SMEM_BH+2*8) x (SMEM_BW+2*8)  = 32 x 32 = 1 KB
// Right tile: (SMEM_BH+16) x (SMEM_BW+16+128) = 32 x 160 = 5 KB
// Total ≈ 6 KB — well within 48 KB per SM.

__global__ void kernel_smem(const uint8_t* __restrict__ left,
                            const uint8_t* __restrict__ right,
                            int*           disp_out,
                            int            H, int W,
                            int            max_disp, int radius) {
  // Tile origin in global image coordinates (top-left of the block's valid region)
  int block_r0 = blockIdx.y * SMEM_BH;
  int block_c0 = blockIdx.x * SMEM_BW;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Shared memory: left tile and right tile
  // Right tile is wider: needs extra max_disp columns on the left (for d offsets)
  // and 2*radius halo on each side.
  const int LT_H  = SMEM_BH + 2 * 8;   // over-allocate for radius up to 8
  const int LT_W  = SMEM_BW + 2 * 8;
  const int RT_H  = SMEM_BH + 2 * 8;
  const int RT_W  = SMEM_BW + 2 * 8 + 128;  // +128 for max_disp up to 128

  __shared__ uint8_t s_left [LT_H][LT_W];
  __shared__ uint8_t s_right[RT_H][RT_W];

  // Load left tile: (SMEM_BH + 2*radius) x (SMEM_BW + 2*radius)
  // The halo region starts at (block_r0 - radius, block_c0 - radius)
  int left_halo_r0 = block_r0 - radius;
  int left_halo_c0 = block_c0 - radius;
  int left_tile_h  = SMEM_BH + 2 * radius;
  int left_tile_w  = SMEM_BW + 2 * radius;

  for (int idx = ty * SMEM_BW + tx; idx < left_tile_h * left_tile_w;
       idx += SMEM_BH * SMEM_BW) {
    int lr = idx / left_tile_w;
    int lc = idx % left_tile_w;
    int gr = left_halo_r0 + lr;
    int gc = left_halo_c0 + lc;
    s_left[lr][lc] = (gr >= 0 && gr < H && gc >= 0 && gc < W)
                     ? left[gr * W + gc] : 0;
  }

  // Load right tile: same rows, but shifted left by max_disp for disparity coverage.
  // Right patch at disparity d is centred at (r, c - d).
  // With c in [block_c0, block_c0+SMEM_BW) and d in [0, max_disp),
  // the right columns span [block_c0 - (max_disp-1) - radius, block_c0 + SMEM_BW + radius).
  int right_halo_r0 = block_r0 - radius;
  int right_halo_c0 = block_c0 - (max_disp - 1) - radius;
  int right_tile_h  = SMEM_BH + 2 * radius;
  int right_tile_w  = SMEM_BW + 2 * radius + max_disp;

  for (int idx = ty * SMEM_BW + tx; idx < right_tile_h * right_tile_w;
       idx += SMEM_BH * SMEM_BW) {
    int rr = idx / right_tile_w;
    int rc = idx % right_tile_w;
    int gr = right_halo_r0 + rr;
    int gc = right_halo_c0 + rc;
    s_right[rr][rc] = (gr >= 0 && gr < H && gc >= 0 && gc < W)
                      ? right[gr * W + gc] : 0;
  }

  __syncthreads();

  int r = block_r0 + ty;
  int c = block_c0 + tx;
  if (r >= H || c >= W) return;

  bool at_border = (r < radius) || (r >= H - radius) ||
                   (c < radius) || (c >= W - radius);
  bool right_oob = (c - (max_disp - 1) - radius < 0);

  if (at_border || right_oob) {
    disp_out[r * W + c] = 0;
    return;
  }

  // Local coordinates within the tiles
  // Left tile: s_left[ty + radius][tx + radius] is the centre pixel
  int l_tr = ty + radius;  // thread row in left tile
  int l_tc = tx + radius;  // thread col in left tile

  // Right tile: for disparity d, right centre is at column offset
  //   (c - d) - right_halo_c0 = (block_c0 + tx - d) - (block_c0 - (max_disp-1) - radius)
  //                            = tx + (max_disp - 1) + radius - d
  int r_tr_base = ty + radius;
  int r_tc_base = tx + (max_disp - 1) + radius;  // d=0 offset in right tile

  unsigned int best_sad = 0xFFFFFFFFu;
  int          best_d   = 0;

  for (int d = 0; d < max_disp; ++d) {
    int r_tc = r_tc_base - d;
    if (r_tc - radius < 0) break;

    unsigned int sad = 0;
    for (int dr = -radius; dr <= radius; ++dr) {
      for (int dc = -radius; dc <= radius; ++dc) {
        int lv = s_left [l_tr + dr][l_tc + dc];
        int rv = s_right[r_tr_base + dr][r_tc  + dc];
        sad += (unsigned int)abs(lv - rv);
      }
    }
    if (sad < best_sad) { best_sad = sad; best_d = d; }
  }
  disp_out[r * W + c] = best_d;
}

// ============================================================================
// 3. TILED KERNEL — compile-time unrolled disparity loop to force registers
//
// Root cause of prior spilling: the disparity loop bound was a runtime value,
// so the compiler used variable-indexed array accesses → local memory (DRAM).
//
// Fix: declare the accumulator array with a compile-time size TILED_MAX_DISP
// and use #pragma unroll on the disparity loop.  After unrolling, each sad[d]
// has a compile-time-known index, so the compiler assigns it its own register
// — no array indexing, no local memory.  With 64 entries × 4 bytes = 256 B
// per thread and 128 threads/block, the register budget is comfortably met.
//
// The left-image pixel for each patch position is loaded once into a register
// and reused across all TILED_MAX_DISP disparity accumulators, giving a
// TILED_MAX_DISP-fold reduction in left-image loads vs the basic kernel.
// Right-image accesses are a consecutive descending sequence per patch element,
// which maps to a single coalesced transaction per warp per disparity.
//
// Requires: max_disp <= TILED_MAX_DISP (checked at launch).
// ============================================================================

#define TILED_BW       16
#define TILED_BH        8
#define TILED_MAX_DISP 64   // compile-time; #pragma unroll uses this bound

__global__ void kernel_tiled(const uint8_t* __restrict__ left,
                             const uint8_t* __restrict__ right,
                             int*           disp_out,
                             int            H, int W,
                             int            max_disp, int radius) {
  int c = blockIdx.x * TILED_BW + threadIdx.x;
  int r = blockIdx.y * TILED_BH + threadIdx.y;
  if (r >= H || c >= W) return;

  bool at_border = (r < radius) || (r >= H - radius) ||
                   (c < radius) || (c >= W - radius);
  bool right_oob = (c - (max_disp - 1) - radius < 0);

  if (at_border || right_oob) {
    disp_out[r * W + c] = 0;
    return;
  }

  // Each sad[d] gets its own register after unrolling — no local memory.
  unsigned int sad[TILED_MAX_DISP];
  #pragma unroll
  for (int d = 0; d < TILED_MAX_DISP; ++d) sad[d] = 0;

  // Traverse patch once; for each position load left pixel into a register
  // and reuse it across all TILED_MAX_DISP disparity comparisons.
  for (int dr = -radius; dr <= radius; ++dr) {
    int gr = r + dr;
    for (int dc = -radius; dc <= radius; ++dc) {
      int lv = left[gr * W + c + dc];   // one load, reused TILED_MAX_DISP×
      #pragma unroll
      for (int d = 0; d < TILED_MAX_DISP; ++d) {
        int rv = right[gr * W + c + dc - d];
        sad[d] += (unsigned int)abs(lv - rv);
      }
    }
  }

  // Select best disparity within the runtime-valid range.
  unsigned int best_sad = 0xFFFFFFFFu;
  int          best_d   = 0;
  #pragma unroll
  for (int d = 0; d < TILED_MAX_DISP; ++d) {
    if (d < max_disp && sad[d] < best_sad) {
      best_sad = sad[d];
      best_d   = d;
    }
  }
  disp_out[r * W + c] = best_d;
}

// ============================================================================
// Host-side timing helper
// ============================================================================

static float launch_and_time(void (*launcher)(const uint8_t*, const uint8_t*,
                                               int*, int, int, int, int,
                                               dim3, dim3),
                             const uint8_t* d_left, const uint8_t* d_right,
                             int* d_disp, int H, int W, int max_disp, int radius,
                             dim3 grid, dim3 block, int repeats) {
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));

  // Warm-up
  launcher(d_left, d_right, d_disp, H, W, max_disp, radius, grid, block);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(t0));
  for (int i = 0; i < repeats; ++i)
    launcher(d_left, d_right, d_disp, H, W, max_disp, radius, grid, block);
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
  return ms / repeats;
}

// Trampoline functions so we can pass kernel pointers through the helper.
static void launch_basic(const uint8_t* l, const uint8_t* r, int* d,
                         int H, int W, int md, int rad, dim3 grid, dim3 blk) {
  kernel_basic<<<grid, blk>>>(l, r, d, H, W, md, rad);
}
static void launch_smem(const uint8_t* l, const uint8_t* r, int* d,
                        int H, int W, int md, int rad, dim3 grid, dim3 blk) {
  kernel_smem<<<grid, blk>>>(l, r, d, H, W, md, rad);
}
static void launch_tiled(const uint8_t* l, const uint8_t* r, int* d,
                         int H, int W, int md, int rad, dim3 grid, dim3 blk) {
  kernel_tiled<<<grid, blk>>>(l, r, d, H, W, md, rad);
}

// ============================================================================
// Shared host-side allocation + copy boilerplate
// ============================================================================

struct GpuBuffers {
  uint8_t* d_left  = nullptr;
  uint8_t* d_right = nullptr;
  int*     d_disp  = nullptr;
  int      H, W;

  GpuBuffers(const Image& left, const Image& right) : H(left.height), W(left.width) {
    size_t img_bytes  = (size_t)H * W * sizeof(uint8_t);
    size_t disp_bytes = (size_t)H * W * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_left,  img_bytes));
    CUDA_CHECK(cudaMalloc(&d_right, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_disp,  disp_bytes));
    CUDA_CHECK(cudaMemcpy(d_left,  left.data,  img_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right, right.data, img_bytes,  cudaMemcpyHostToDevice));
  }

  void copy_disp_to(DisparityMap& dst) const {
    CUDA_CHECK(cudaMemcpy(dst.data, d_disp,
                          (size_t)H * W * sizeof(int), cudaMemcpyDeviceToHost));
  }

  ~GpuBuffers() {
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_disp);
  }
};

// ============================================================================
// Public host wrappers
// ============================================================================

float sad_stereo_gpu_basic(const Image& left, const Image& right,
                           DisparityMap& disp_out,
                           int max_disp, int radius, int repeats) {
  GpuBuffers buf(left, right);
  dim3 block(16, 16);
  dim3 grid((left.width  + block.x - 1) / block.x,
            (left.height + block.y - 1) / block.y);

  float ms = launch_and_time(launch_basic, buf.d_left, buf.d_right, buf.d_disp,
                              left.height, left.width, max_disp, radius,
                              grid, block, repeats);
  buf.copy_disp_to(disp_out);
  return ms;
}

float sad_stereo_gpu_smem(const Image& left, const Image& right,
                          DisparityMap& disp_out,
                          int max_disp, int radius, int repeats) {
  GpuBuffers buf(left, right);
  dim3 block(SMEM_BW, SMEM_BH);
  dim3 grid((left.width  + block.x - 1) / block.x,
            (left.height + block.y - 1) / block.y);

  float ms = launch_and_time(launch_smem, buf.d_left, buf.d_right, buf.d_disp,
                              left.height, left.width, max_disp, radius,
                              grid, block, repeats);
  buf.copy_disp_to(disp_out);
  return ms;
}

float sad_stereo_gpu_tiled(const Image& left, const Image& right,
                           DisparityMap& disp_out,
                           int max_disp, int radius, int repeats) {
  CHECK(max_disp <= TILED_MAX_DISP, "sad_stereo_gpu_tiled: max_disp exceeds TILED_MAX_DISP");
  GpuBuffers buf(left, right);
  dim3 block(TILED_BW, TILED_BH);
  dim3 grid((left.width  + block.x - 1) / block.x,
            (left.height + block.y - 1) / block.y);

  float ms = launch_and_time(launch_tiled, buf.d_left, buf.d_right, buf.d_disp,
                              left.height, left.width, max_disp, radius,
                              grid, block, repeats);
  buf.copy_disp_to(disp_out);
  return ms;
}
