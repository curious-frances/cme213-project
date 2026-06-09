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

__global__ void kernel_basic(const uint8_t* __restrict__ left,
                             const uint8_t* __restrict__ right,
                             disp_t* disp_out,
                             int H, int W, int max_disp, int radius) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= H || c >= W) return;

  bool at_border = (r < radius) || (r >= H - radius) ||
                   (c < radius) || (c >= W - radius);
  bool right_oob = (c - (max_disp - 1) - radius < 0);

  if (at_border || right_oob) { disp_out[r * W + c] = 0; return; }

  unsigned int best_sad = 0xFFFFFFFFu;
  int          best_d   = 0;

  for (int d = 0; d < max_disp; ++d) {
    if (c - d - radius < 0) break;
    unsigned int sad = 0;
    for (int dr = -radius; dr <= radius; ++dr)
      for (int dc = -radius; dc <= radius; ++dc) {
        int lv = left [(r + dr) * W + (c + dc)];
        int rv = right[(r + dr) * W + (c + dc - d)];
        sad += (unsigned int)abs(lv - rv);
      }
    if (sad < best_sad) { best_sad = sad; best_d = d; }
  }
  disp_out[r * W + c] = best_d;
}

#define SMEM_BW 16
#define SMEM_BH 16

__global__ void kernel_smem(const uint8_t* __restrict__ left,
                            const uint8_t* __restrict__ right,
                            disp_t* disp_out,
                            int H, int W, int max_disp, int radius) {
  int block_r0 = blockIdx.y * SMEM_BH;
  int block_c0 = blockIdx.x * SMEM_BW;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  const int LT_H = SMEM_BH + 2 * 8;
  const int LT_W = SMEM_BW + 2 * 8;
  const int RT_H = SMEM_BH + 2 * 8;
  const int RT_W = SMEM_BW + 2 * 8 + 128;

  __shared__ uint8_t s_left [LT_H][LT_W];
  __shared__ uint8_t s_right[RT_H][RT_W];

  int left_halo_r0 = block_r0 - radius;
  int left_halo_c0 = block_c0 - radius;
  int left_tile_h  = SMEM_BH + 2 * radius;
  int left_tile_w  = SMEM_BW + 2 * radius;

  for (int idx = ty * SMEM_BW + tx; idx < left_tile_h * left_tile_w;
       idx += SMEM_BH * SMEM_BW) {
    int lr = idx / left_tile_w, lc = idx % left_tile_w;
    int gr = left_halo_r0 + lr,  gc = left_halo_c0 + lc;
    s_left[lr][lc] = (gr >= 0 && gr < H && gc >= 0 && gc < W)
                     ? left[gr * W + gc] : 0;
  }

  int right_halo_r0 = block_r0 - radius;
  int right_halo_c0 = block_c0 - (max_disp - 1) - radius;
  int right_tile_h  = SMEM_BH + 2 * radius;
  int right_tile_w  = SMEM_BW + 2 * radius + max_disp;

  for (int idx = ty * SMEM_BW + tx; idx < right_tile_h * right_tile_w;
       idx += SMEM_BH * SMEM_BW) {
    int rr = idx / right_tile_w, rc = idx % right_tile_w;
    int gr = right_halo_r0 + rr,  gc = right_halo_c0 + rc;
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
  if (at_border || right_oob) { disp_out[r * W + c] = 0; return; }

  int l_tr      = ty + radius;
  int l_tc      = tx + radius;
  int r_tr_base = ty + radius;
  int r_tc_base = tx + (max_disp - 1) + radius;

  unsigned int best_sad = 0xFFFFFFFFu;
  int          best_d   = 0;

  for (int d = 0; d < max_disp; ++d) {
    int r_tc = r_tc_base - d;
    if (r_tc - radius < 0) break;
    unsigned int sad = 0;
    for (int dr = -radius; dr <= radius; ++dr)
      for (int dc = -radius; dc <= radius; ++dc) {
        int lv = s_left [l_tr + dr][l_tc + dc];
        int rv = s_right[r_tr_base + dr][r_tc + dc];
        sad += (unsigned int)abs(lv - rv);
      }
    if (sad < best_sad) { best_sad = sad; best_d = d; }
  }
  disp_out[r * W + c] = best_d;
}

#define TILED_BW       16
#define TILED_BH        8
#define TILED_MAX_DISP 64

__global__ void kernel_tiled(const uint8_t* __restrict__ left,
                             const uint8_t* __restrict__ right,
                             disp_t* disp_out,
                             int H, int W, int max_disp, int radius) {
  int c = blockIdx.x * TILED_BW + threadIdx.x;
  int r = blockIdx.y * TILED_BH + threadIdx.y;
  if (r >= H || c >= W) return;

  bool at_border = (r < radius) || (r >= H - radius) ||
                   (c < radius) || (c >= W - radius);
  bool right_oob = (c - (max_disp - 1) - radius < 0);
  if (at_border || right_oob) { disp_out[r * W + c] = 0; return; }

  unsigned int sad[TILED_MAX_DISP];
  #pragma unroll
  for (int d = 0; d < TILED_MAX_DISP; ++d) sad[d] = 0;

  for (int dr = -radius; dr <= radius; ++dr) {
    int gr = r + dr;
    for (int dc = -radius; dc <= radius; ++dc) {
      int lv = left[gr * W + c + dc];
      #pragma unroll
      for (int d = 0; d < TILED_MAX_DISP; ++d) {
        int rv = right[gr * W + c + dc - d];
        sad[d] += (unsigned int)abs(lv - rv);
      }
    }
  }

  unsigned int best_sad = 0xFFFFFFFFu;
  int          best_d   = 0;
  #pragma unroll
  for (int d = 0; d < TILED_MAX_DISP; ++d) {
    if (d < max_disp && sad[d] < best_sad) { best_sad = sad[d]; best_d = d; }
  }

  // parabolic sub-pixel refinement
  disp_t result = (disp_t)best_d;
  if (best_d > 0 && best_d < max_disp - 1) {
    float c0 = (float)sad[best_d - 1];
    float c1 = (float)sad[best_d];
    float c2 = (float)sad[best_d + 1];
    float denom = c0 - 2.0f * c1 + c2;
    if (denom > 0.0f) {
      float delta = 0.5f * (c0 - c2) / denom;   // in (-0.5, 0.5)
      result = (disp_t)best_d + delta;
    }
  }
  disp_out[r * W + c] = result;
}

// right-reference SAD for left-right consistency check
__global__ void kernel_tiled_right(const uint8_t* __restrict__ left,
                                   const uint8_t* __restrict__ right,
                                   disp_t* disp_out,
                                   int H, int W, int max_disp, int radius) {
  int c = blockIdx.x * TILED_BW + threadIdx.x;
  int r = blockIdx.y * TILED_BH + threadIdx.y;
  if (r >= H || c >= W) return;

  bool at_border = (r < radius) || (r >= H - radius) ||
                   (c < radius) || (c >= W - radius);
  bool left_oob = (c + (max_disp - 1) + radius >= W);
  if (at_border || left_oob) { disp_out[r * W + c] = 0; return; }

  unsigned int sad[TILED_MAX_DISP];
  #pragma unroll
  for (int d = 0; d < TILED_MAX_DISP; ++d) sad[d] = 0;

  for (int dr = -radius; dr <= radius; ++dr) {
    int gr = r + dr;
    for (int dc = -radius; dc <= radius; ++dc) {
      int rv = right[gr * W + c + dc];
      #pragma unroll
      for (int d = 0; d < TILED_MAX_DISP; ++d) {
        int lv = left[gr * W + c + dc + d];
        sad[d] += (unsigned int)abs(rv - lv);
      }
    }
  }

  unsigned int best_sad = 0xFFFFFFFFu;
  int          best_d   = 0;
  #pragma unroll
  for (int d = 0; d < TILED_MAX_DISP; ++d) {
    if (d < max_disp && sad[d] < best_sad) { best_sad = sad[d]; best_d = d; }
  }
  disp_out[r * W + c] = (disp_t)best_d;
}

// LR consistency check: reject pixels where left/right disparities disagree by more than LRC_TOL
#define LRC_TOL 1.0f
__global__ void kernel_lr_check(disp_t* dispL, const disp_t* __restrict__ dispR,
                                int H, int W) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= H || c >= W) return;

  disp_t dL = dispL[r * W + c];
  int    xr = c - (int)lrintf(dL);
  if (xr < 0 || xr >= W) { dispL[r * W + c] = (disp_t)(-1); return; }
  disp_t dR = dispR[r * W + xr];
  if (fabsf(dL - dR) > LRC_TOL) dispL[r * W + c] = (disp_t)(-1);
}

static float launch_and_time(void (*launcher)(const uint8_t*, const uint8_t*,
                                               disp_t*, int, int, int, int,
                                               dim3, dim3),
                             const uint8_t* d_left, const uint8_t* d_right,
                             disp_t* d_disp, int H, int W, int max_disp, int radius,
                             dim3 grid, dim3 block, int repeats) {
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));

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

static void launch_basic(const uint8_t* l, const uint8_t* r, disp_t* d,
                         int H, int W, int md, int rad, dim3 grid, dim3 blk) {
  kernel_basic<<<grid, blk>>>(l, r, d, H, W, md, rad);
}
static void launch_smem(const uint8_t* l, const uint8_t* r, disp_t* d,
                        int H, int W, int md, int rad, dim3 grid, dim3 blk) {
  kernel_smem<<<grid, blk>>>(l, r, d, H, W, md, rad);
}
static void launch_tiled(const uint8_t* l, const uint8_t* r, disp_t* d,
                         int H, int W, int md, int rad, dim3 grid, dim3 blk) {
  kernel_tiled<<<grid, blk>>>(l, r, d, H, W, md, rad);
}

// host-device buffer manager with page-locked async H2D/D2H copies
struct GpuBuffers {
  uint8_t*    d_left  = nullptr;
  uint8_t*    d_right = nullptr;
  disp_t*     d_disp  = nullptr;
  int         H, W;
  cudaStream_t stream = nullptr;
  const uint8_t* h_left;
  const uint8_t* h_right;

  GpuBuffers(const Image& left, const Image& right) : H(left.height), W(left.width) {
    size_t img_bytes  = (size_t)H * W * sizeof(uint8_t);
    size_t disp_bytes = (size_t)H * W * sizeof(disp_t);

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_left,  img_bytes));
    CUDA_CHECK(cudaMalloc(&d_right, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_disp,  disp_bytes));

    h_left  = left.data;
    h_right = right.data;
    CUDA_CHECK(cudaHostRegister((void*)h_left,  img_bytes, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister((void*)h_right, img_bytes, cudaHostRegisterDefault));

    CUDA_CHECK(cudaMemcpyAsync(d_left,  h_left,  img_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_right, h_right, img_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaHostUnregister((void*)h_left));
    CUDA_CHECK(cudaHostUnregister((void*)h_right));
  }

  void copy_disp_to(DisparityMap& dst) const {
    size_t disp_bytes = (size_t)H * W * sizeof(disp_t);
    CUDA_CHECK(cudaHostRegister(dst.data, disp_bytes, cudaHostRegisterDefault));
    CUDA_CHECK(cudaMemcpyAsync(dst.data, d_disp, disp_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaHostUnregister(dst.data));
  }

  ~GpuBuffers() {
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_disp);
    if (stream) cudaStreamDestroy(stream);
  }
};

float sad_stereo_gpu_basic(const Image& left, const Image& right,
                           DisparityMap& disp_out,
                           int max_disp, int radius, int repeats) {
  GpuBuffers buf(left, right);
  dim3 block(16, 16);
  dim3 grid((left.width  + block.x - 1) / block.x,
            (left.height + block.y - 1) / block.y);
  float ms = launch_and_time(launch_basic, buf.d_left, buf.d_right, buf.d_disp,
                              left.height, left.width, max_disp, radius, grid, block, repeats);
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
                              left.height, left.width, max_disp, radius, grid, block, repeats);
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
                              left.height, left.width, max_disp, radius, grid, block, repeats);
  buf.copy_disp_to(disp_out);
  return ms;
}

// tiled SAD + left-right consistency check
float sad_stereo_gpu_tiled_lrc(const Image& left, const Image& right,
                               DisparityMap& disp_out,
                               int max_disp, int radius, int repeats) {
  CHECK(max_disp <= TILED_MAX_DISP, "sad_stereo_gpu_tiled_lrc: max_disp exceeds TILED_MAX_DISP");
  GpuBuffers buf(left, right);
  int H = left.height, W = left.width;

  disp_t* d_dispR = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dispR, (size_t)H * W * sizeof(disp_t)));

  dim3 block(TILED_BW, TILED_BH);
  dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

  auto run_once = [&]() {
    kernel_tiled      <<<grid, block>>>(buf.d_left, buf.d_right, buf.d_disp, H, W, max_disp, radius);
    kernel_tiled_right<<<grid, block>>>(buf.d_left, buf.d_right, d_dispR,    H, W, max_disp, radius);
    kernel_lr_check   <<<grid, block>>>(buf.d_disp, d_dispR, H, W);
  };

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  run_once();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(t0));
  for (int i = 0; i < repeats; ++i) run_once();
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));

  buf.copy_disp_to(disp_out);
  cudaFree(d_dispR);
  return ms / repeats;
}
