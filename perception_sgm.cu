#include "perception_sgm.h"

#include <cuda_runtime.h>
#include <climits>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d — %s\n",                         \
              __FILE__, __LINE__, cudaGetErrorString(_e));                \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// Sentinel cost for invalid (out-of-bounds) disparities. Large enough never to
// win the argmin, small enough that 4-path accumulation stays well within int.
#define SGM_INF_COST 30000

// ---------------------------------------------------------------------------
// Data term: windowed SAD cost volume C[(r*W+c)*D + d].
// Same matching cost as the local SAD baseline so the comparison isolates the
// effect of path aggregation. Stored uint16 (a 5x5 8-bit window SAD <= 6375).
// ---------------------------------------------------------------------------
__global__ void sgm_cost_kernel(const uint8_t* __restrict__ left,
                                const uint8_t* __restrict__ right,
                                unsigned short* __restrict__ C,
                                int H, int W, int D, int radius) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= H || c >= W) return;

  bool at_border = (r < radius) || (r >= H - radius) ||
                   (c < radius) || (c >= W - radius);
  unsigned short* Cp = C + (size_t)(r * W + c) * D;

  for (int d = 0; d < D; ++d) {
    if (at_border || (c - d - radius < 0)) { Cp[d] = SGM_INF_COST; continue; }
    unsigned int sad = 0;
    for (int dr = -radius; dr <= radius; ++dr)
      for (int dc = -radius; dc <= radius; ++dc) {
        int lv = left [(r + dr) * W + (c + dc)];
        int rv = right[(r + dr) * W + (c + dc - d)];
        sad += (unsigned int)abs(lv - rv);
      }
    Cp[d] = (unsigned short)min(sad, (unsigned int)SGM_INF_COST);
  }
}

// ---------------------------------------------------------------------------
// Path aggregation. One block per scanline; blockDim.x = next_pow2(D) so the
// min-over-disparity reduction is a clean tree. Each block sweeps its scanline
// sequentially (the SGM recurrence is a serial dependency along the path), while
// the H (or W) independent scanlines run in parallel across blocks.
//
//   L_r(p,d) = C(p,d) + min( L_r(p-r,d),
//                            L_r(p-r,d-1)+P1, L_r(p-r,d+1)+P1,
//                            min_k L_r(p-r,k)+P2 ) - min_k L_r(p-r,k)
//
// The accumulator S is summed over all four path directions (separate launches).
// ---------------------------------------------------------------------------
__global__ void sgm_aggregate(const unsigned short* __restrict__ C,
                              int* __restrict__ S,
                              int H, int W, int D,
                              int dir_r, int dir_c,
                              int P1, int P2) {
  extern __shared__ int sh[];
  int* s_Lprev = sh;                 // [blockDim.x]
  int* s_red   = sh + blockDim.x;    // [blockDim.x]

  int  d       = threadIdx.x;
  bool valid_d = (d < D);

  int r, c, steps;
  if (dir_c != 0) {                  // horizontal path: one block per row
    r = blockIdx.x;
    c = (dir_c > 0) ? 0 : (W - 1);
    steps = W;
  } else {                           // vertical path: one block per column
    c = blockIdx.x;
    r = (dir_r > 0) ? 0 : (H - 1);
    steps = H;
  }

  // First pixel of the scanline: aggregated cost equals the data term.
  int Lcur;
  if (valid_d) {
    int idx = (r * W + c) * D + d;
    Lcur = C[idx];
    S[idx] += Lcur;
  } else {
    Lcur = SGM_INF_COST;
  }
  s_Lprev[d] = Lcur;
  __syncthreads();

  for (int step = 1; step < steps; ++step) {
    // min_k L_r(p-r, k) via tree reduction over the previous pixel's costs.
    s_red[d] = s_Lprev[d];
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
      if (d < s) s_red[d] = min(s_red[d], s_red[d + s]);
      __syncthreads();
    }
    int minPrev = s_red[0];

    // Read the three neighbour terms before any thread overwrites s_Lprev.
    int lprev_d  = s_Lprev[d];
    int lprev_m1 = (d > 0)                ? s_Lprev[d - 1] : SGM_INF_COST;
    int lprev_p1 = (d < blockDim.x - 1)   ? s_Lprev[d + 1] : SGM_INF_COST;
    __syncthreads();

    r += dir_r;
    c += dir_c;
    if (valid_d) {
      int idx  = (r * W + c) * D + d;
      int cost = C[idx];
      int best = min(min(lprev_d, lprev_m1 + P1),
                     min(lprev_p1 + P1, minPrev + P2));
      Lcur = cost + best - minPrev;
      S[idx] += Lcur;
    } else {
      Lcur = SGM_INF_COST;
    }
    s_Lprev[d] = Lcur;
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Winner-take-all over the summed path costs + parabolic sub-pixel refinement.
// ---------------------------------------------------------------------------
__global__ void sgm_argmin(const int* __restrict__ S, disp_t* __restrict__ disp,
                           int H, int W, int D, int radius, int max_disp) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= H || c >= W) return;

  bool at_border = (r < radius) || (r >= H - radius) ||
                   (c < radius) || (c >= W - radius);
  bool right_oob = (c - (max_disp - 1) - radius < 0);
  if (at_border || right_oob) { disp[r * W + c] = 0; return; }

  const int* Sp = S + (size_t)(r * W + c) * D;
  int best = INT_MAX, best_d = 0;
  for (int d = 0; d < D; ++d) {
    if (Sp[d] < best) { best = Sp[d]; best_d = d; }
  }

  disp_t result = (disp_t)best_d;
  if (best_d > 0 && best_d < D - 1) {
    float c0 = (float)Sp[best_d - 1];
    float c1 = (float)Sp[best_d];
    float c2 = (float)Sp[best_d + 1];
    float denom = c0 - 2.0f * c1 + c2;
    if (denom > 0.0f) result = (disp_t)best_d + 0.5f * (c0 - c2) / denom;
  }
  disp[r * W + c] = result;
}

static int next_pow2(int x) {
  int p = 1;
  while (p < x) p <<= 1;
  return p;
}

float sgm_stereo_gpu(const Image& left, const Image& right,
                     DisparityMap& disp_out,
                     int max_disp, int radius, int p1, int p2, int repeats) {
  int    H = left.height, W = left.width, D = max_disp;
  int    threads = next_pow2(D);
  size_t npix    = (size_t)H * W;

  uint8_t*        d_left  = nullptr;
  uint8_t*        d_right = nullptr;
  unsigned short* d_C     = nullptr;
  int*            d_S     = nullptr;
  disp_t*         d_disp  = nullptr;

  CUDA_CHECK(cudaMalloc(&d_left,  npix * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_right, npix * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_C,     npix * D * sizeof(unsigned short)));
  CUDA_CHECK(cudaMalloc(&d_S,     npix * D * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_disp,  npix * sizeof(disp_t)));

  CUDA_CHECK(cudaMemcpy(d_left,  left.data,  npix, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_right, right.data, npix, cudaMemcpyHostToDevice));

  dim3 cblk(16, 16);
  dim3 cgrd((W + cblk.x - 1) / cblk.x, (H + cblk.y - 1) / cblk.y);
  dim3 ablk(threads);
  dim3 argblk(16, 16);
  dim3 arggrd((W + argblk.x - 1) / argblk.x, (H + argblk.y - 1) / argblk.y);
  size_t shmem = 2 * (size_t)threads * sizeof(int);

  auto run_once = [&]() {
    sgm_cost_kernel<<<cgrd, cblk>>>(d_left, d_right, d_C, H, W, D, radius);
    CUDA_CHECK(cudaMemset(d_S, 0, npix * D * sizeof(int)));
    sgm_aggregate<<<H, ablk, shmem>>>(d_C, d_S, H, W, D,  0, +1, p1, p2);  // L->R
    sgm_aggregate<<<H, ablk, shmem>>>(d_C, d_S, H, W, D,  0, -1, p1, p2);  // R->L
    sgm_aggregate<<<W, ablk, shmem>>>(d_C, d_S, H, W, D, +1,  0, p1, p2);  // T->B
    sgm_aggregate<<<W, ablk, shmem>>>(d_C, d_S, H, W, D, -1,  0, p1, p2);  // B->T
    sgm_argmin<<<arggrd, argblk>>>(d_S, d_disp, H, W, D, radius, max_disp);
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

  CUDA_CHECK(cudaMemcpy(disp_out.data, d_disp, npix * sizeof(disp_t),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_left);
  cudaFree(d_right);
  cudaFree(d_C);
  cudaFree(d_S);
  cudaFree(d_disp);
  return ms / repeats;
}
