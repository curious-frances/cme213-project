#ifndef PERCEPTION_SGM_H_
#define PERCEPTION_SGM_H_

#include "perception_common.h"

// GPU Semi-Global Matching with a configurable number of paths and data term.
//
// Aggregates a matching cost along `num_paths` scanline directions (4 = the
// axis-aligned L->R, R->L, T->B, B->T; 8 = also the four diagonals) with the
// standard SGM smoothness penalties P1 (small disparity change) and P2
// (disparity discontinuity), then takes the argmin of the summed path costs
// with parabolic sub-pixel refinement.
//
// The data term is selected by `use_census`:
//   0 -> windowed SAD  (same cost as the local SAD baseline; apples-to-apples)
//   1 -> Census transform + Hamming distance (robust to gain/exposure changes)
//
// Returns mean kernel time in milliseconds over `repeats` runs.
float sgm_stereo_gpu(const Image&  left,
                     const Image&  right,
                     DisparityMap& disp_out,
                     int           max_disp,
                     int           radius,
                     int           p1,
                     int           p2,
                     int           repeats,
                     int           num_paths,
                     int           use_census);

// ---------------------------------------------------------------------------
// Distributed (MPI) 4-path SGM: slab-level building blocks. The MPI driver
// owns the frontier MPI_Send/Recv between adjacent ranks; these calls do the
// per-rank GPU work. See perception_sgm.cu for the decomposition rationale.
//
//   ctx = sgm_dist_begin(...)              // build cost volume + horizontal paths
//   sgm_dist_vert(ctx, +1, fin, fout)      // T->B vertical (frontier in/out)
//   sgm_dist_vert(ctx, -1, fin, fout)      // B->T vertical
//   sgm_dist_finish(ctx, slab_disp)        // argmin + sub-pixel, frees ctx
//
// Frontier buffers are W*D ints. Pass fin=nullptr at the global image border.
// ---------------------------------------------------------------------------
void* sgm_dist_begin(const Image& slab_l, const Image& slab_r,
                     int max_disp, int radius, int p1, int p2, int use_census,
                     int halo_top, int local_rows);
void  sgm_dist_vert(void* ctx, int dir, const int* fin_host, int* fout_host);
void  sgm_dist_finish(void* ctx, DisparityMap& slab_disp);

#endif
