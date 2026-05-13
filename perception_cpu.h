#ifndef PERCEPTION_CPU_H_
#define PERCEPTION_CPU_H_

// ---------------------------------------------------------------------------
// perception_cpu.h
//
// CPU baseline for stereo SAD (Sum of Absolute Differences) disparity
// estimation.  All data-generation helpers are declared here so the future
// CUDA host-side code can include this header and reuse them without change.
// ---------------------------------------------------------------------------

#include "perception_common.h"

// ===========================================================================
// Synthetic stereo pair generation
// ===========================================================================

// generate_left_image
//   Fills `img` with a deterministic grayscale texture.
//   The pattern is a combination of horizontal and vertical sinusoids so that
//   every local patch is unique — a necessary condition for SAD to converge to
//   the correct disparity.
//
//   The texture formula (mimicking the fill() helper in gemm_test.cpp):
//     pixel(r, c) = 127 + 60*sin(2π*r/period_r) * cos(2π*c/period_c)
//                       + 40*sin(2π*r/period_r2)
//                       + 30*cos(2π*c/period_c2)
//   clamped to [0, 255].
void generate_left_image(Image& img);

// generate_right_image
//   Generates the right image as a horizontal shift of the left image by
//   `true_disp` pixels.  Pixels in the first `true_disp` columns of the right
//   image (for which there is no correspondence) are filled with 128 (mid-grey).
//
//   Right(r, c) = Left(r, c + true_disp)    for c + true_disp < width
//   Right(r, c) = 128                        otherwise
void generate_right_image(const Image& left, Image& right, int true_disp);

// generate_ground_truth
//   Fills `gt` with `true_disp` for every pixel the SAD algorithm will actually
//   compute (i.e. same validity mask as sad_stereo_cpu), and -1 elsewhere.
//   This ensures error is exactly 0 on a perfect synthetic pair.
void generate_ground_truth(DisparityMap& gt,
                           int           true_disp,
                           int           radius,
                           int           max_disp);

// ===========================================================================
// CPU stereo disparity estimation
// ===========================================================================

// sad_stereo_cpu
//   For every valid interior pixel (r, c) in the left image, search
//   disparities d in [0, max_disp) and return the d that minimises the
//   Sum of Absolute Differences (SAD) over a (2*radius+1)×(2*radius+1) patch.
//
//   Coordinate convention (matches CUDA thread indexing we will use):
//     Left patch centred at  (r, c)
//     Right patch centred at (r, c - d)   ← standard left-to-right convention
//
//   Border policy:
//     Pixels within `radius` of any edge, or for which the best-match right
//     patch would fall outside the image, receive BORDER_FILL (0).
//
//   This function is the single source of truth for the algorithm.  The CUDA
//   kernel must produce results within DISP_TOL of these values.
void sad_stereo_cpu(const Image&   left,
                    const Image&   right,
                    DisparityMap&  disp_out,
                    int            max_disp,
                    int            radius);

// ===========================================================================
// Accuracy evaluation
// ===========================================================================

// compute_mae
//   Mean Absolute Error between estimated and ground-truth disparity maps,
//   considering only pixels where gt.at(r,c) >= 0 (valid pixels).
//   Returns the MAE in pixels.
double compute_mae(const DisparityMap& estimated,
                   const DisparityMap& gt);

// compute_bad_pixel_rate
//   Fraction of valid pixels whose |estimated - gt| > threshold.
double compute_bad_pixel_rate(const DisparityMap& estimated,
                              const DisparityMap& gt,
                              int                 threshold = DISP_TOL);

// ===========================================================================
// Timing helper
// ===========================================================================

// TimerResult bundles the statistics collected over `repeats` runs.
struct TimerResult {
  double mean_ms  = 0.0;   // mean wall-clock time in milliseconds
  double std_ms   = 0.0;   // sample standard deviation
  double min_ms   = 0.0;   // fastest run
  double fps      = 0.0;   // frames per second (1000 / mean_ms)
};

// time_sad_stereo_cpu
//   Runs sad_stereo_cpu `repeats` times and collects timing statistics.
//   The disparity map from the *last* run is written to `disp_out`.
TimerResult time_sad_stereo_cpu(const Image&  left,
                                const Image&  right,
                                DisparityMap& disp_out,
                                int           max_disp,
                                int           radius,
                                int           repeats);

// ===========================================================================
// Pretty-printing helpers (used by both main_cpu and the test harness)
// ===========================================================================

void print_stereo_header(const StereoParams& p);
void print_timing_result(const TimerResult& t, const char* label);
void print_accuracy(const DisparityMap& estimated, const DisparityMap& gt);

// ===========================================================================
// Image I/O helpers
// ===========================================================================
void save_pgm(const Image& img, const std::string& path);
void save_disparity_pgm(const DisparityMap& disp, const std::string& path, int max_disp);

// ===========================================================================
// CSV benchmark output
// ===========================================================================
// Appends one row to `csv_path` (creates file + header if it doesn't exist).
void append_benchmark_csv(const std::string& csv_path,
                          const std::string& impl,
                          const StereoParams& p,
                          const TimerResult& t,
                          double gops_per_frame);

#endif  // PERCEPTION_CPU_H_
