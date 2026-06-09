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

// sentinel cost for invalid disparities
#define SGM_INF_COST 30000

typedef unsigned short cost_t;   // per-(pixel,disparity) data term

// SAD cost volume: C[(r*W+c)*D+d]
__global__ void sgm_cost_sad_kernel(const uint8_t* __restrict__ left,
                                    const uint8_t* __restrict__ right,
                                    cost_t* __restrict__ C,
                                    int H, int W, int D, int radius) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= H || c >= W) return;

  bool at_border = (r < radius) || (r >= H - radius) ||
                   (c < radius) || (c >= W - radius);
  cost_t* Cp = C + (size_t)(r * W + c) * D;

  for (int d = 0; d < D; ++d) {
    if (at_border || (c - d - radius < 0)) { Cp[d] = SGM_INF_COST; continue; }
    unsigned int sad = 0;
    for (int dr = -radius; dr <= radius; ++dr)
      for (int dc = -radius; dc <= radius; ++dc) {
        int lv = left [(r + dr) * W + (c + dc)];
        int rv = right[(r + dr) * W + (c + dc - d)];
        sad += (unsigned int)abs(lv - rv);
      }
    Cp[d] = (cost_t)min(sad, (unsigned int)SGM_INF_COST);
  }
}

// census transform + Hamming distance cost volume
__global__ void sgm_census_kernel(const uint8_t* __restrict__ img,
                                  unsigned long long* __restrict__ cen,
                                  int H, int W, int radius) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= H || c >= W) return;

  bool at_border = (r < radius) || (r >= H - radius) ||
                   (c < radius) || (c >= W - radius);
  if (at_border) { cen[r * W + c] = 0ULL; return; }

  unsigned long long bits = 0ULL;
  int center = img[r * W + c];
  for (int dr = -radius; dr <= radius; ++dr)
    for (int dc = -radius; dc <= radius; ++dc) {
      if (dr == 0 && dc == 0) continue;
      bits = (bits << 1) | (img[(r + dr) * W + (c + dc)] < center ? 1ULL : 0ULL);
    }
  cen[r * W + c] = bits;
}

__global__ void sgm_cost_census_kernel(const unsigned long long* __restrict__ cenL,
                                       const unsigned long long* __restrict__ cenR,
                                       cost_t* __restrict__ C,
                                       int H, int W, int D, int radius) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= H || c >= W) return;

  bool at_border = (r < radius) || (r >= H - radius) ||
                   (c < radius) || (c >= W - radius);
  cost_t* Cp = C + (size_t)(r * W + c) * D;
  unsigned long long left_bits = cenL[r * W + c];

  for (int d = 0; d < D; ++d) {
    if (at_border || (c - d - radius < 0)) { Cp[d] = SGM_INF_COST; continue; }
    Cp[d] = (cost_t)__popcll(left_bits ^ cenR[r * W + (c - d)]);
  }
}

// scanline start (r0, c0) and length for path direction (dir_r, dir_c)
__device__ void sgm_line_start(int b, int dir_r, int dir_c, int H, int W,
                               int& r0, int& c0, int& steps) {
  int r_entry = (dir_r > 0) ? 0 : (H - 1);
  int c_entry = (dir_c > 0) ? 0 : (W - 1);

  if (dir_r == 0) {                 // horizontal: block index is the row
    r0 = b;        c0 = c_entry;
  } else if (dir_c == 0) {          // vertical: block index is the column
    r0 = r_entry;  c0 = b;
  } else {                          // diagonal: entry row edge then entry col edge
    if (b < W) { r0 = r_entry; c0 = b; }            // top/bottom edge
    else {
      int k = b - W;                                 // left/right edge, skip corner
      r0 = (k < r_entry) ? k : (k + 1);
      c0 = c_entry;
    }
  }
  int t_r = (dir_r > 0) ? (H - 1 - r0) : (dir_r < 0 ? r0 : INT_MAX);
  int t_c = (dir_c > 0) ? (W - 1 - c0) : (dir_c < 0 ? c0 : INT_MAX);
  steps = min(t_r, t_c) + 1;
}

static int sgm_num_lines(int dir_r, int dir_c, int H, int W) {
  if (dir_r == 0) return H;
  if (dir_c == 0) return W;
  return W + H - 1;
}

// SGM path aggregation along (dir_r, dir_c): one block per scanline, one thread per disparity
__global__ void sgm_aggregate(const cost_t* __restrict__ C, int* __restrict__ S,
                              int H, int W, int D,
                              int dir_r, int dir_c, int P1, int P2) {
  extern __shared__ int sh[];
  int* s_Lprev = sh;
  int* s_red   = sh + blockDim.x;

  int  d       = threadIdx.x;
  bool valid_d = (d < D);

  int r, c, steps;
  sgm_line_start(blockIdx.x, dir_r, dir_c, H, W, r, c, steps);

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
    s_red[d] = s_Lprev[d];
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
      if (d < s) s_red[d] = min(s_red[d], s_red[d + s]);
      __syncthreads();
    }
    int minPrev  = s_red[0];
    int lprev_d  = s_Lprev[d];
    int lprev_m1 = (d > 0)              ? s_Lprev[d - 1] : SGM_INF_COST;
    int lprev_p1 = (d < blockDim.x - 1) ? s_Lprev[d + 1] : SGM_INF_COST;
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

// Winner-take-all over the summed path costs + parabolic sub-pixel refinement.
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

static int next_pow2(int x) { int p = 1; while (p < x) p <<= 1; return p; }

// Eight path directions; the first four are the axis-aligned 4-path set.
struct SgmDir { int dr, dc; };
static const SgmDir kDirs[8] = {
  { 0, 1}, { 0,-1}, { 1, 0}, {-1, 0},          // L->R, R->L, T->B, B->T
  { 1, 1}, { 1,-1}, {-1, 1}, {-1,-1}           // four diagonals
};

float sgm_stereo_gpu(const Image& left, const Image& right,
                     DisparityMap& disp_out,
                     int max_disp, int radius, int p1, int p2,
                     int repeats, int num_paths, int use_census) {
  int    H = left.height, W = left.width, D = max_disp;
  int    threads = next_pow2(D);
  size_t npix    = (size_t)H * W;

  uint8_t*            d_left  = nullptr;
  uint8_t*            d_right = nullptr;
  cost_t*             d_C     = nullptr;
  int*                d_S     = nullptr;
  disp_t*             d_disp  = nullptr;
  unsigned long long* d_cenL  = nullptr;
  unsigned long long* d_cenR  = nullptr;

  CUDA_CHECK(cudaMalloc(&d_left,  npix * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_right, npix * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_C,     npix * D * sizeof(cost_t)));
  CUDA_CHECK(cudaMalloc(&d_S,     npix * D * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_disp,  npix * sizeof(disp_t)));
  if (use_census) {
    CUDA_CHECK(cudaMalloc(&d_cenL, npix * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_cenR, npix * sizeof(unsigned long long)));
  }

  CUDA_CHECK(cudaMemcpy(d_left,  left.data,  npix, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_right, right.data, npix, cudaMemcpyHostToDevice));

  dim3 cblk(16, 16);
  dim3 cgrd((W + cblk.x - 1) / cblk.x, (H + cblk.y - 1) / cblk.y);
  dim3 ablk(threads);
  dim3 argblk(16, 16);
  dim3 arggrd((W + argblk.x - 1) / argblk.x, (H + argblk.y - 1) / argblk.y);
  size_t shmem = 2 * (size_t)threads * sizeof(int);

  auto run_once = [&]() {
    if (use_census) {
      sgm_census_kernel<<<cgrd, cblk>>>(d_left,  d_cenL, H, W, radius);
      sgm_census_kernel<<<cgrd, cblk>>>(d_right, d_cenR, H, W, radius);
      sgm_cost_census_kernel<<<cgrd, cblk>>>(d_cenL, d_cenR, d_C, H, W, D, radius);
    } else {
      sgm_cost_sad_kernel<<<cgrd, cblk>>>(d_left, d_right, d_C, H, W, D, radius);
    }
    CUDA_CHECK(cudaMemset(d_S, 0, npix * D * sizeof(int)));
    for (int pth = 0; pth < num_paths; ++pth) {
      int dr = kDirs[pth].dr, dc = kDirs[pth].dc;
      int nlines = sgm_num_lines(dr, dc, H, W);
      sgm_aggregate<<<nlines, ablk, shmem>>>(d_C, d_S, H, W, D, dr, dc, p1, p2);
    }
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

  cudaFree(d_left);  cudaFree(d_right); cudaFree(d_C);
  cudaFree(d_S);     cudaFree(d_disp);
  if (use_census) { cudaFree(d_cenL); cudaFree(d_cenR); }
  return ms / repeats;
}

// distributed SGM: horizontal paths local, vertical paths pass W*D frontier between ranks

// vertical slab aggregation; frontier_in/out are W*D ints (null at global border)
__global__ void sgm_vert_slab(const cost_t* __restrict__ C, int* __restrict__ S,
                              int W, int D, int dir, int first_row, int n,
                              const int* __restrict__ frontier_in,
                              int* __restrict__ frontier_out,
                              int P1, int P2) {
  extern __shared__ int sh[];
  int* s_Lprev = sh;
  int* s_red   = sh + blockDim.x;
  int  c       = blockIdx.x;
  int  d       = threadIdx.x;
  bool valid_d = (d < D);
  int  Lcur    = SGM_INF_COST;

  int start_step;
  if (frontier_in != nullptr) {                 // continue from neighbour's row
    s_Lprev[d] = valid_d ? frontier_in[c * D + d] : SGM_INF_COST;
    start_step = 0;
  } else {                                       // global border: L = C
    int idx = (first_row * W + c) * D + d;
    Lcur = valid_d ? (int)C[idx] : SGM_INF_COST;
    if (valid_d) S[idx] += Lcur;
    s_Lprev[d] = Lcur;
    start_step = 1;
  }
  __syncthreads();

  for (int step = start_step; step < n; ++step) {
    int row = first_row + dir * step;
    s_red[d] = s_Lprev[d];
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
      if (d < s) s_red[d] = min(s_red[d], s_red[d + s]);
      __syncthreads();
    }
    int minPrev  = s_red[0];
    int lprev_d  = s_Lprev[d];
    int lprev_m1 = (d > 0)              ? s_Lprev[d - 1] : SGM_INF_COST;
    int lprev_p1 = (d < blockDim.x - 1) ? s_Lprev[d + 1] : SGM_INF_COST;
    __syncthreads();
    int idx = (row * W + c) * D + d;
    if (valid_d) {
      int cost = C[idx];
      int best = min(min(lprev_d, lprev_m1 + P1), min(lprev_p1 + P1, minPrev + P2));
      Lcur = cost + best - minPrev;
      S[idx] += Lcur;
    } else {
      Lcur = SGM_INF_COST;
    }
    s_Lprev[d] = Lcur;
    __syncthreads();
  }
  if (valid_d) frontier_out[c * D + d] = s_Lprev[d];
}

struct SgmDistCtx {
  int slabH, W, D, threads, radius, p1, p2, halo_top, local_rows, use_census;
  uint8_t*            d_l;    uint8_t*            d_r;
  cost_t*             d_C;    int*                d_S;    disp_t* d_disp;
  int*                d_fin;  int*                d_fout;
  unsigned long long* d_cenL; unsigned long long* d_cenR;
};

void* sgm_dist_begin(const Image& slab_l, const Image& slab_r,
                     int max_disp, int radius, int p1, int p2, int use_census,
                     int halo_top, int local_rows) {
  SgmDistCtx* x = new SgmDistCtx();
  int H = slab_l.height, W = slab_l.width, D = max_disp;
  x->slabH = H; x->W = W; x->D = D; x->threads = next_pow2(D); x->radius = radius;
  x->p1 = p1; x->p2 = p2; x->halo_top = halo_top; x->local_rows = local_rows;
  x->use_census = use_census;
  size_t npix = (size_t)H * W;

  CUDA_CHECK(cudaMalloc(&x->d_l,    npix));
  CUDA_CHECK(cudaMalloc(&x->d_r,    npix));
  CUDA_CHECK(cudaMalloc(&x->d_C,    npix * D * sizeof(cost_t)));
  CUDA_CHECK(cudaMalloc(&x->d_S,    npix * D * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&x->d_disp, npix * sizeof(disp_t)));
  CUDA_CHECK(cudaMalloc(&x->d_fin,  (size_t)W * D * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&x->d_fout, (size_t)W * D * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(x->d_l, slab_l.data, npix, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(x->d_r, slab_r.data, npix, cudaMemcpyHostToDevice));

  dim3 cblk(16, 16), cgrd((W + 15) / 16, (H + 15) / 16);
  if (use_census) {
    CUDA_CHECK(cudaMalloc(&x->d_cenL, npix * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&x->d_cenR, npix * sizeof(unsigned long long)));
    sgm_census_kernel<<<cgrd, cblk>>>(x->d_l, x->d_cenL, H, W, radius);
    sgm_census_kernel<<<cgrd, cblk>>>(x->d_r, x->d_cenR, H, W, radius);
    sgm_cost_census_kernel<<<cgrd, cblk>>>(x->d_cenL, x->d_cenR, x->d_C, H, W, D, radius);
  } else {
    x->d_cenL = nullptr; x->d_cenR = nullptr;
    sgm_cost_sad_kernel<<<cgrd, cblk>>>(x->d_l, x->d_r, x->d_C, H, W, D, radius);
  }
  CUDA_CHECK(cudaMemset(x->d_S, 0, npix * D * sizeof(int)));

  // horizontal paths, local to each rank
  size_t shmem = 2 * (size_t)x->threads * sizeof(int);
  sgm_aggregate<<<H, x->threads, shmem>>>(x->d_C, x->d_S, H, W, D, 0,  1, p1, p2);
  sgm_aggregate<<<H, x->threads, shmem>>>(x->d_C, x->d_S, H, W, D, 0, -1, p1, p2);
  return x;
}

// vertical sweep: dir=+1 top->bottom, dir=-1 bottom->top
void sgm_dist_vert(void* ctx, int dir, const int* fin_host, int* fout_host) {
  SgmDistCtx* x = (SgmDistCtx*)ctx;
  int W = x->W, D = x->D;
  const int* d_fin_ptr = nullptr;
  if (fin_host) {
    CUDA_CHECK(cudaMemcpy(x->d_fin, fin_host, (size_t)W * D * sizeof(int),
                          cudaMemcpyHostToDevice));
    d_fin_ptr = x->d_fin;
  }
  int first_row = (dir > 0) ? x->halo_top
                            : (x->halo_top + x->local_rows - 1);
  size_t shmem = 2 * (size_t)x->threads * sizeof(int);
  sgm_vert_slab<<<W, x->threads, shmem>>>(x->d_C, x->d_S, W, D, dir,
                                          first_row, x->local_rows,
                                          d_fin_ptr, x->d_fout, x->p1, x->p2);
  CUDA_CHECK(cudaMemcpy(fout_host, x->d_fout, (size_t)W * D * sizeof(int),
                        cudaMemcpyDeviceToHost));
}

void sgm_dist_finish(void* ctx, DisparityMap& slab_disp) {
  SgmDistCtx* x = (SgmDistCtx*)ctx;
  int H = x->slabH, W = x->W, D = x->D;
  dim3 ab(16, 16), ag((W + 15) / 16, (H + 15) / 16);
  sgm_argmin<<<ag, ab>>>(x->d_S, x->d_disp, H, W, D, x->radius, x->D);
  CUDA_CHECK(cudaMemcpy(slab_disp.data, x->d_disp, (size_t)H * W * sizeof(disp_t),
                        cudaMemcpyDeviceToHost));
  cudaFree(x->d_l); cudaFree(x->d_r); cudaFree(x->d_C); cudaFree(x->d_S);
  cudaFree(x->d_disp); cudaFree(x->d_fin); cudaFree(x->d_fout);
  if (x->d_cenL) cudaFree(x->d_cenL);
  if (x->d_cenR) cudaFree(x->d_cenR);
  delete x;
}
