// ---------------------------------------------------------------------------
// perception_cpu.cpp
//
// CPU implementation of the stereo SAD disparity estimator.
// All functions declared in perception_cpu.h are defined here.
// ---------------------------------------------------------------------------

#include "perception_cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

// ===========================================================================
// Synthetic stereo pair generation
// ===========================================================================

void generate_left_image(Image& img) {
  // Deterministic multi-frequency sinusoidal texture.
  // Using the same style as gemm_test.cpp's fill() — purely arithmetic,
  // no external data, fully reproducible.
  const double pi       = std::acos(-1.0);
  const double pr       = 64.0;   // row period (pixels)
  const double pc       = 80.0;   // col period
  const double pr2      = 32.0;   // second row frequency
  const double pc2      = 40.0;   // second col frequency

  for (int r = 0; r < img.height; ++r) {
    for (int c = 0; c < img.width; ++c) {
      double v = 127.0
               + 60.0 * std::sin(2.0 * pi * r / pr)  * std::cos(2.0 * pi * c / pc)
               + 40.0 * std::sin(2.0 * pi * r / pr2)
               + 30.0 * std::cos(2.0 * pi * c / pc2);
      // Clamp to [0, 255]
      int iv = static_cast<int>(std::round(v));
      img.at(r, c) = static_cast<pixel_t>(std::max(0, std::min(255, iv)));
    }
  }
}

void generate_right_image(const Image& left, Image& right, int true_disp) {
  CHECK(left.height == right.height && left.width == right.width,
        "generate_right_image: image size mismatch");
  CHECK(true_disp >= 0 && true_disp < left.width,
        "generate_right_image: true_disp out of range");

  for (int r = 0; r < left.height; ++r) {
    for (int c = 0; c < left.width; ++c) {
      // Right(r, c) = Left(r, c + true_disp)
      // i.e. the right camera sees the scene shifted left by true_disp pixels.
      int src_c = c + true_disp;
      if (src_c < left.width) {
        right.at(r, c) = left.at(r, src_c);
      } else {
        right.at(r, c) = 128;  // no correspondence — fill with mid-grey
      }
    }
  }
}

void generate_ground_truth(DisparityMap& gt,
                           int           true_disp,
                           int           radius,
                           int           max_disp) {
  const int H = gt.height;
  const int W = gt.width;

  for (int r = 0; r < H; ++r) {
    for (int c = 0; c < W; ++c) {
      // Mark as invalid if:
      //  - within border of radius pixels
      //  - no valid right-image patch for this disparity (c - true_disp < radius)
      //  - true_disp >= max_disp (would not be found in search range)
      bool border  = (r < radius) || (r >= H - radius) ||
                     (c < radius) || (c >= W - radius);
      // Match exactly what sad_stereo_cpu skips: needs right patch valid for
      // ALL d in [0, max_disp), so c - (max_disp-1) - radius >= 0.
      bool right_oob    = (c - (max_disp - 1) - radius < 0);
      bool out_of_range = (true_disp >= max_disp);

      if (border || right_oob || out_of_range) {
        gt.at(r, c) = -1;
      } else {
        gt.at(r, c) = true_disp;
      }
    }
  }
}

// ===========================================================================
// CPU stereo disparity estimation — the reference implementation
// ===========================================================================

// Inner SAD: sum |left_patch - right_patch| for a (2r+1)×(2r+1) window.
// Left patch centred at (lr, lc); right patch centred at (lr, lc - d).
static inline score_t patch_sad(const Image& left,
                                 const Image& right,
                                 int          lr,
                                 int          lc,
                                 int          d,
                                 int          radius) {
  score_t sad = 0;
  for (int dr = -radius; dr <= radius; ++dr) {
    for (int dc = -radius; dc <= radius; ++dc) {
      int l_val = static_cast<int>(left.at(lr + dr, lc + dc));
      int r_val = static_cast<int>(right.at(lr + dr, lc + dc - d));
      sad += static_cast<score_t>(std::abs(l_val - r_val));
    }
  }
  return sad;
}

void sad_stereo_cpu(const Image&  left,
                    const Image&  right,
                    DisparityMap& disp_out,
                    int           max_disp,
                    int           radius) {
  CHECK(left.height == right.height && left.width == right.width,
        "sad_stereo_cpu: image size mismatch");
  CHECK(disp_out.height == left.height && disp_out.width == left.width,
        "sad_stereo_cpu: output size mismatch");

  const int H = left.height;
  const int W = left.width;

  for (int r = 0; r < H; ++r) {
    for (int c = 0; c < W; ++c) {
      // ------------------------------------------------------------------ //
      // Border check: need full patch in both images for all d in [0, max_disp)
      // ------------------------------------------------------------------ //
      bool at_border = (r < radius) || (r >= H - radius) ||
                       (c < radius) || (c >= W - radius);

      // For disparity d, the right patch runs from (c - d - radius) to
      // (c - d + radius).  We need c - (max_disp-1) - radius >= 0.
      bool right_oob = (c - (max_disp - 1) - radius < 0);

      if (at_border || right_oob) {
        disp_out.at(r, c) = BORDER_FILL;
        continue;
      }

      // ------------------------------------------------------------------ //
      // Search over disparities
      // ------------------------------------------------------------------ //
      score_t best_sad = std::numeric_limits<score_t>::max();
      disp_t  best_d   = 0;

      for (int d = 0; d < max_disp; ++d) {
        // Right patch centred at (r, c - d): left boundary is c - d - radius
        if (c - d - radius < 0) break;  // no more valid disparities

        score_t s = patch_sad(left, right, r, c, d, radius);
        if (s < best_sad) {
          best_sad = s;
          best_d   = d;
        }
      }

      disp_out.at(r, c) = best_d;
    }
  }
}

// ===========================================================================
// Accuracy evaluation
// ===========================================================================

double compute_mae(const DisparityMap& estimated, const DisparityMap& gt) {
  CHECK(estimated.height == gt.height && estimated.width == gt.width,
        "compute_mae: size mismatch");

  double sum   = 0.0;
  long   count = 0;
  for (int i = 0; i < gt.size(); ++i) {
    if (gt.data[i] >= 0) {
      sum += std::abs(static_cast<double>(estimated.data[i]) - gt.data[i]);
      ++count;
    }
  }
  return (count > 0) ? sum / count : 0.0;
}

double compute_bad_pixel_rate(const DisparityMap& estimated,
                              const DisparityMap& gt,
                              int                 threshold) {
  CHECK(estimated.height == gt.height && estimated.width == gt.width,
        "compute_bad_pixel_rate: size mismatch");

  long bad   = 0;
  long count = 0;
  for (int i = 0; i < gt.size(); ++i) {
    if (gt.data[i] >= 0) {
      if (std::abs(estimated.data[i] - gt.data[i]) > threshold) ++bad;
      ++count;
    }
  }
  return (count > 0) ? static_cast<double>(bad) / count : 0.0;
}

// ===========================================================================
// Timing
// ===========================================================================

TimerResult time_sad_stereo_cpu(const Image&  left,
                                const Image&  right,
                                DisparityMap& disp_out,
                                int           max_disp,
                                int           radius,
                                int           repeats) {
  CHECK(repeats > 0, "time_sad_stereo_cpu: repeats must be > 0");

  using Clock = std::chrono::high_resolution_clock;
  std::vector<double> times_ms(repeats);

  for (int i = 0; i < repeats; ++i) {
    auto t0 = Clock::now();
    sad_stereo_cpu(left, right, disp_out, max_disp, radius);
    auto t1 = Clock::now();
    times_ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  double mean = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / repeats;

  double var = 0.0;
  for (double t : times_ms) var += (t - mean) * (t - mean);
  var /= repeats;

  double mn = *std::min_element(times_ms.begin(), times_ms.end());

  TimerResult res;
  res.mean_ms = mean;
  res.std_ms  = std::sqrt(var);
  res.min_ms  = mn;
  res.fps     = (mean > 0.0) ? 1000.0 / mean : 0.0;
  return res;
}

// ===========================================================================
// Pretty-printing helpers
// ===========================================================================

void print_stereo_header(const StereoParams& p) {
  std::cout << "\n";
  std::cout << "====================================================\n";
  std::cout << "  Stereo SAD — CPU baseline\n";
  std::cout << "====================================================\n";
  std::cout << std::left;
  std::cout << "  Image size   : " << p.height << " x " << p.width << "\n";
  std::cout << "  Patch radius : " << p.radius
            << "  (patch = " << (2*p.radius+1) << "x" << (2*p.radius+1) << ")\n";
  std::cout << "  Max disparity: " << p.max_disp << "\n";
  std::cout << "  True disparity (GT): " << p.true_disp << "\n";
  std::cout << "  Repeats      : " << p.repeats << "\n";
  std::cout << "----------------------------------------------------\n";
}

void print_timing_result(const TimerResult& t, const char* label) {
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  [" << label << "]\n";
  std::cout << "    Mean  : " << std::setw(9) << t.mean_ms << " ms"
            << "  ±" << t.std_ms << " ms\n";
  std::cout << "    Min   : " << std::setw(9) << t.min_ms  << " ms\n";
  std::cout << "    FPS   : " << std::setw(9) << t.fps     << "\n";
}

void print_accuracy(const DisparityMap& estimated, const DisparityMap& gt) {
  double mae  = compute_mae(estimated, gt);
  double bad  = compute_bad_pixel_rate(estimated, gt, DISP_TOL) * 100.0;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "  Accuracy (valid pixels only):\n";
  std::cout << "    MAE         : " << mae  << " px\n";
  std::cout << "    Bad-px rate : " << bad  << " %"
            << "  (|err| > " << DISP_TOL << " px)\n";
  std::cout << "----------------------------------------------------\n";
}

// ===========================================================================
// Image I/O
// ===========================================================================

void save_pgm(const Image& img, const std::string& path) {
  std::ofstream f(path, std::ios::binary);
  CHECK(f.is_open(), ("save_pgm: cannot open " + path).c_str());
  f << "P5\n" << img.width << " " << img.height << "\n255\n";
  f.write(reinterpret_cast<const char*>(img.data), img.size());
}

void save_disparity_pgm(const DisparityMap& disp, const std::string& path, int max_disp) {
  std::ofstream f(path, std::ios::binary);
  CHECK(f.is_open(), ("save_disparity_pgm: cannot open " + path).c_str());
  f << "P5\n" << disp.width << " " << disp.height << "\n255\n";
  // Scale disparity [0, max_disp) → [0, 255] for visualization
  std::vector<uint8_t> buf(disp.size());
  for (int i = 0; i < disp.size(); ++i) {
    int d = disp.data[i];
    if (d < 0) d = 0;
    buf[i] = static_cast<uint8_t>(std::min(255, d * 255 / std::max(1, max_disp - 1)));
  }
  f.write(reinterpret_cast<const char*>(buf.data()), buf.size());
}

// ===========================================================================
// CSV benchmark output
// ===========================================================================

void append_benchmark_csv(const std::string& csv_path,
                          const std::string& impl,
                          const StereoParams& p,
                          const TimerResult& t,
                          double gops_per_frame) {
  bool write_header = false;
  {
    std::ifstream probe(csv_path);
    write_header = !probe.good();
  }
  std::ofstream f(csv_path, std::ios::app);
  CHECK(f.is_open(), ("append_benchmark_csv: cannot open " + csv_path).c_str());
  if (write_header) {
    f << "impl,height,width,max_disp,radius,mean_ms,min_ms,fps,gops_per_frame,gops_per_s\n";
  }
  double gops_per_s = (t.mean_ms > 0) ? gops_per_frame / (t.mean_ms / 1e3) : 0.0;
  f << std::fixed << std::setprecision(4)
    << impl << ","
    << p.height << "," << p.width << "," << p.max_disp << "," << p.radius << ","
    << t.mean_ms << "," << t.min_ms << "," << t.fps << ","
    << gops_per_frame << "," << gops_per_s << "\n";
}
