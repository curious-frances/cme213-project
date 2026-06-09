#ifndef PERCEPTION_GPU_H_
#define PERCEPTION_GPU_H_

#include "perception_common.h"

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

// Tiled SAD followed by a left-right consistency check that rejects
// occluded / mismatched pixels (sets them to a negative sentinel).
float sad_stereo_gpu_tiled_lrc(const Image&  left,
                               const Image&  right,
                               DisparityMap& disp_out,
                               int           max_disp,
                               int           radius,
                               int           repeats);

#endif
