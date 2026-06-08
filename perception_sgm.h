#ifndef PERCEPTION_SGM_H_
#define PERCEPTION_SGM_H_

#include "perception_common.h"

// 4-path GPU Semi-Global Matching.
//
// Uses the SAME windowed-SAD matching cost as the local SAD baseline as its
// data term, then aggregates costs along four scanline directions (L->R, R->L,
// T->B, B->T) with the standard SGM smoothness penalties P1 (small disparity
// change) and P2 (disparity discontinuity). Disparity is the argmin of the
// summed path costs, with the same parabolic sub-pixel refinement as the tiled
// SAD kernel. Returns mean kernel time in milliseconds over `repeats` runs.
float sgm_stereo_gpu(const Image&  left,
                     const Image&  right,
                     DisparityMap& disp_out,
                     int           max_disp,
                     int           radius,
                     int           p1,
                     int           p2,
                     int           repeats);

#endif
