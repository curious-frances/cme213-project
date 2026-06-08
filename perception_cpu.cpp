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

void generate_left_image(Image& img) {
  const double pi  = std::acos(-1.0);
  const double pr  = 64.0;
  const double pc  = 80.0;
  const double pr2 = 32.0;
  const double pc2 = 40.0;

  for (int r = 0; r < img.height; ++r) {
    for (int c = 0; c < img.width; ++c) {
      double v = 127.0
               + 60.0 * std::sin(2.0 * pi * r / pr) * std::cos(2.0 * pi * c / pc)
               + 40.0 * std::sin(2.0 * pi * r / pr2)
               + 30.0 * std::cos(2.0 * pi * c / pc2);
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
      int src_c = c + true_disp;
      right.at(r, c) = (src_c < left.width) ? left.at(r, src_c) : 128;
    }
  }
}

void generate_ground_truth(DisparityMap& gt, int true_disp, int radius, int max_disp) {
  const int H = gt.height;
  const int W = gt.width;

  for (int r = 0; r < H; ++r) {
    for (int c = 0; c < W; ++c) {
      bool border       = (r < radius) || (r >= H - radius) ||
                          (c < radius) || (c >= W - radius);
      bool right_oob    = (c - (max_disp - 1) - radius < 0);
      bool out_of_range = (true_disp >= max_disp);

      gt.at(r, c) = (border || right_oob || out_of_range) ? -1 : true_disp;
    }
  }
}

static inline score_t patch_sad(const Image& left, const Image& right,
                                 int lr, int lc, int d, int radius) {
  score_t sad = 0;
  for (int dr = -radius; dr <= radius; ++dr)
    for (int dc = -radius; dc <= radius; ++dc) {
      int l_val = static_cast<int>(left.at(lr + dr, lc + dc));
      int r_val = static_cast<int>(right.at(lr + dr, lc + dc - d));
      sad += static_cast<score_t>(std::abs(l_val - r_val));
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
      bool at_border = (r < radius) || (r >= H - radius) ||
                       (c < radius) || (c >= W - radius);
      bool right_oob = (c - (max_disp - 1) - radius < 0);

      if (at_border || right_oob) {
        disp_out.at(r, c) = BORDER_FILL;
        continue;
      }

      score_t best_sad = std::numeric_limits<score_t>::max();
      disp_t  best_d   = 0;

      for (int d = 0; d < max_disp; ++d) {
        if (c - d - radius < 0) break;
        score_t s = patch_sad(left, right, r, c, d, radius);
        if (s < best_sad) { best_sad = s; best_d = d; }
      }

      disp_out.at(r, c) = best_d;
    }
  }
}

// Accuracy is measured only over pixels that are valid in BOTH the ground
// truth (gt.data[i] >= 0) and the estimate (estimated.data[i] >= 0). Pixels
// that the left-right consistency check rejected (set to a negative sentinel)
// are excluded so the MAE/bad-rate reflect the surviving, trustworthy matches;
// the lost coverage is reported separately by compute_density().
double compute_mae(const DisparityMap& estimated, const DisparityMap& gt) {
  CHECK(estimated.height == gt.height && estimated.width == gt.width,
        "compute_mae: size mismatch");
  double sum   = 0.0;
  long   count = 0;
  for (int i = 0; i < gt.size(); ++i) {
    if (gt.data[i] >= 0 && estimated.data[i] >= 0) {
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
    if (gt.data[i] >= 0 && estimated.data[i] >= 0) {
      if (std::abs(estimated.data[i] - gt.data[i]) > threshold) ++bad;
      ++count;
    }
  }
  return (count > 0) ? static_cast<double>(bad) / count : 0.0;
}

// Fraction of valid-GT pixels for which the estimate is also valid (i.e. not
// rejected by the consistency check). 1.0 means full coverage.
double compute_density(const DisparityMap& estimated, const DisparityMap& gt) {
  CHECK(estimated.height == gt.height && estimated.width == gt.width,
        "compute_density: size mismatch");
  long valid = 0;
  long count = 0;
  for (int i = 0; i < gt.size(); ++i) {
    if (gt.data[i] >= 0) {
      if (estimated.data[i] >= 0) ++valid;
      ++count;
    }
  }
  return (count > 0) ? static_cast<double>(valid) / count : 0.0;
}

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
    auto t0    = Clock::now();
    sad_stereo_cpu(left, right, disp_out, max_disp, radius);
    auto t1    = Clock::now();
    times_ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  double mean = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / repeats;
  double var  = 0.0;
  for (double t : times_ms) var += (t - mean) * (t - mean);
  var /= repeats;

  TimerResult res;
  res.mean_ms = mean;
  res.std_ms  = std::sqrt(var);
  res.min_ms  = *std::min_element(times_ms.begin(), times_ms.end());
  res.fps     = (mean > 0.0) ? 1000.0 / mean : 0.0;
  return res;
}

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
  double mae     = compute_mae(estimated, gt);
  double bad     = compute_bad_pixel_rate(estimated, gt, DISP_TOL) * 100.0;
  double density = compute_density(estimated, gt) * 100.0;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "  Accuracy (valid pixels only):\n";
  std::cout << "    MAE         : " << mae << " px\n";
  std::cout << "    Bad-px rate : " << bad << " %"
            << "  (|err| > " << DISP_TOL << " px)\n";
  std::cout << "    Coverage    : " << density << " %"
            << "  (estimated / valid-GT pixels)\n";
  std::cout << "----------------------------------------------------\n";
}

Image load_pgm(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  CHECK(f.is_open(), ("load_pgm: cannot open " + path).c_str());

  std::string magic;
  f >> magic;
  CHECK(magic == "P5", ("load_pgm: not a binary (P5) PGM: " + path).c_str());

  while (f.peek() == '\n' || f.peek() == '\r' || f.peek() == ' ') f.get();
  while (f.peek() == '#') { std::string line; std::getline(f, line); }

  int w, h, maxval;
  f >> w >> h >> maxval;
  CHECK(maxval == 255, ("load_pgm: only 8-bit PGM (maxval 255) supported: " + path).c_str());
  f.get();

  Image img(h, w);
  f.read(reinterpret_cast<char*>(img.data), img.size());
  CHECK(f.good(), ("load_pgm: read error in " + path).c_str());
  return img;
}

// Load a Middlebury-style ground-truth disparity map stored as an 8-bit PGM.
// Gray values are scaled disparities (true_disp = gray / scale); a gray value
// of 0 marks an unknown/occluded pixel and is stored as the negative sentinel
// so the accuracy metrics ignore it.
DisparityMap load_disparity_pgm(const std::string& path, double scale) {
  std::ifstream f(path, std::ios::binary);
  CHECK(f.is_open(), ("load_disparity_pgm: cannot open " + path).c_str());

  std::string magic;
  f >> magic;
  CHECK(magic == "P5", ("load_disparity_pgm: not a binary (P5) PGM: " + path).c_str());

  while (f.peek() == '\n' || f.peek() == '\r' || f.peek() == ' ') f.get();
  while (f.peek() == '#') { std::string line; std::getline(f, line); }

  int w, h, maxval;
  f >> w >> h >> maxval;
  CHECK(maxval == 255, ("load_disparity_pgm: only 8-bit PGM supported: " + path).c_str());
  f.get();

  std::vector<uint8_t> raw(static_cast<size_t>(w) * h);
  f.read(reinterpret_cast<char*>(raw.data()), raw.size());
  CHECK(f.good(), ("load_disparity_pgm: read error in " + path).c_str());

  DisparityMap gt(h, w);
  for (int i = 0; i < gt.size(); ++i) {
    gt.data[i] = (raw[i] == 0) ? static_cast<disp_t>(-1)
                               : static_cast<disp_t>(raw[i] / scale);
  }
  return gt;
}

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
  std::vector<uint8_t> buf(disp.size());
  for (int i = 0; i < disp.size(); ++i) {
    float d = std::max(0.0f, static_cast<float>(disp.data[i]));
    int   v = static_cast<int>(d * 255.0f / std::max(1, max_disp - 1) + 0.5f);
    buf[i]  = static_cast<uint8_t>(std::min(255, v));
  }
  f.write(reinterpret_cast<const char*>(buf.data()), buf.size());
}

void save_disparity_ppm(const DisparityMap& disp, const std::string& path, int max_disp) {
  std::ofstream f(path, std::ios::binary);
  CHECK(f.is_open(), ("save_disparity_ppm: cannot open " + path).c_str());
  f << "P6\n" << disp.width << " " << disp.height << "\n255\n";

  // Jet colormap: blue(0) -> cyan -> green -> yellow -> red(1)
  std::vector<uint8_t> buf(disp.size() * 3);
  for (int i = 0; i < disp.size(); ++i) {
    float t = std::max(0.0f, static_cast<float>(disp.data[i])) /
              static_cast<float>(std::max(1, max_disp - 1));
    t = std::min(1.0f, t);

    float r, g, b;
    if      (t < 0.125f) { r = 0.0f; g = 0.0f; b = 0.5f + 4.0f * t; }
    else if (t < 0.375f) { r = 0.0f; g = 4.0f * (t - 0.125f); b = 1.0f; }
    else if (t < 0.625f) { r = 4.0f * (t - 0.375f); g = 1.0f; b = 1.0f - 4.0f * (t - 0.375f); }
    else if (t < 0.875f) { r = 1.0f; g = 1.0f - 4.0f * (t - 0.625f); b = 0.0f; }
    else                  { r = 1.0f - 4.0f * (t - 0.875f); g = 0.0f; b = 0.0f; }

    buf[3 * i + 0] = static_cast<uint8_t>(std::min(255.0f, r * 255.0f));
    buf[3 * i + 1] = static_cast<uint8_t>(std::min(255.0f, g * 255.0f));
    buf[3 * i + 2] = static_cast<uint8_t>(std::min(255.0f, b * 255.0f));
  }
  f.write(reinterpret_cast<const char*>(buf.data()), buf.size());
}

void append_benchmark_csv(const std::string&  csv_path,
                          const std::string&  impl,
                          const StereoParams& p,
                          const TimerResult&  t,
                          double              gops_per_frame) {
  bool write_header = false;
  { std::ifstream probe(csv_path); write_header = !probe.good(); }

  std::ofstream f(csv_path, std::ios::app);
  CHECK(f.is_open(), ("append_benchmark_csv: cannot open " + csv_path).c_str());

  if (write_header)
    f << "impl,height,width,max_disp,radius,mean_ms,min_ms,fps,gops_per_frame,gops_per_s\n";

  double gops_per_s = (t.mean_ms > 0) ? gops_per_frame / (t.mean_ms / 1e3) : 0.0;
  f << std::fixed << std::setprecision(4)
    << impl << ","
    << p.height << "," << p.width << "," << p.max_disp << "," << p.radius << ","
    << t.mean_ms << "," << t.min_ms << "," << t.fps << ","
    << gops_per_frame << "," << gops_per_s << "\n";
}
