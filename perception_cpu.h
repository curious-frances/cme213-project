#ifndef PERCEPTION_CPU_H_
#define PERCEPTION_CPU_H_

#include "perception_common.h"

void generate_left_image(Image& img);
void generate_right_image(const Image& left, Image& right, int true_disp);
void generate_ground_truth(DisparityMap& gt, int true_disp, int radius, int max_disp);

void sad_stereo_cpu(const Image&  left,
                    const Image&  right,
                    DisparityMap& disp_out,
                    int           max_disp,
                    int           radius);

double compute_mae(const DisparityMap& estimated, const DisparityMap& gt);
double compute_bad_pixel_rate(const DisparityMap& estimated,
                              const DisparityMap& gt,
                              int                 threshold = DISP_TOL);

struct TimerResult {
  double mean_ms = 0.0;
  double std_ms  = 0.0;
  double min_ms  = 0.0;
  double fps     = 0.0;
};

TimerResult time_sad_stereo_cpu(const Image&  left,
                                const Image&  right,
                                DisparityMap& disp_out,
                                int           max_disp,
                                int           radius,
                                int           repeats);

void print_stereo_header(const StereoParams& p);
void print_timing_result(const TimerResult& t, const char* label);
void print_accuracy(const DisparityMap& estimated, const DisparityMap& gt);

void save_pgm(const Image& img, const std::string& path);
void save_disparity_pgm(const DisparityMap& disp, const std::string& path, int max_disp);

void append_benchmark_csv(const std::string&  csv_path,
                          const std::string&  impl,
                          const StereoParams& p,
                          const TimerResult&  t,
                          double              gops_per_frame);

#endif
