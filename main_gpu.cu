// ---------------------------------------------------------------------------
// main_gpu.cu — Benchmarks all three GPU SAD kernels + CPU baseline.
// ---------------------------------------------------------------------------

#include <cstring>
#include <iostream>
#include <string>

#include "perception_common.h"
#include "perception_cpu.h"
#include "perception_gpu.h"

struct RunOptions {
  StereoParams p;
  bool        save_images  = false;
  bool        run_cpu      = true;
  std::string csv_path     = "";
};

static void parse_args(int argc, char** argv, RunOptions& opt) {
  StereoParams& p = opt.p;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if ((key == "--height"   || key == "-H") && i+1<argc) { p.height    = std::stoi(argv[++i]); continue; }
    if ((key == "--width"    || key == "-W") && i+1<argc) { p.width     = std::stoi(argv[++i]); continue; }
    if ((key == "--disp"     || key == "-d") && i+1<argc) { p.true_disp = std::stoi(argv[++i]); continue; }
    if ((key == "--max-disp" || key == "-m") && i+1<argc) { p.max_disp  = std::stoi(argv[++i]); continue; }
    if ((key == "--radius"   || key == "-r") && i+1<argc) { p.radius    = std::stoi(argv[++i]); continue; }
    if ((key == "--repeats"  || key == "-n") && i+1<argc) { p.repeats   = std::stoi(argv[++i]); continue; }
    if ((key == "--csv")                     && i+1<argc) { opt.csv_path = argv[++i];            continue; }
    if (key == "--save-images")  { opt.save_images = true;  continue; }
    if (key == "--no-cpu")       { opt.run_cpu     = false; continue; }
    if (key == "--help" || key == "-h") {
      std::cout << "Usage: ./main_gpu [options]\n"
                << "  --height   H   image height            (default 480)\n"
                << "  --width    W   image width             (default 640)\n"
                << "  --disp     D   true disparity (GT)     (default 24)\n"
                << "  --max-disp MD  disparity search range  (default 64)\n"
                << "  --radius   R   patch half-size         (default 2)\n"
                << "  --repeats  N   timing repetitions      (default 5)\n"
                << "  --save-images  write left/right/disp PGMs\n"
                << "  --no-cpu       skip CPU baseline\n"
                << "  --csv PATH     append benchmark rows to CSV\n";
      std::exit(0);
    }
    std::cerr << "Unknown argument: " << key << "\n"; std::exit(1);
  }
  CHECK(p.height > 0 && p.width > 0, "Image dimensions must be positive");
  CHECK(p.radius > 0,  "Radius must be > 0");
  CHECK(p.max_disp > 0, "max_disp must be > 0");
  CHECK(p.true_disp >= 0 && p.true_disp < p.max_disp, "true_disp out of range");
  CHECK(p.repeats > 0, "repeats must be > 0");
  CHECK(p.width > 2 * p.radius + p.max_disp, "Image too narrow");
  CHECK(p.height > 2 * p.radius, "Image too short");
  CHECK(p.max_disp <= 128, "max_disp must be <= 128 for GPU kernels");
  CHECK(p.radius   <=   8, "radius must be <= 8 for smem kernel");
}

// Compute GOps/frame for the given params (2 ops per SAD element: abs + add)
static double compute_gops(const StereoParams& p) {
  long patch   = (long)(2 * p.radius + 1) * (2 * p.radius + 1);
  long valid_w = p.width  - 2 * p.radius - p.max_disp;
  long valid_h = p.height - 2 * p.radius;
  if (valid_w < 0) valid_w = 0;
  return static_cast<double>(valid_h * valid_w * p.max_disp * patch * 2L) / 1e9;
}

static void print_gpu_result(const char* label, float mean_ms,
                             double gops_frame) {
  double fps       = (mean_ms > 0) ? 1000.0 / mean_ms : 0.0;
  double throughput = (mean_ms > 0) ? gops_frame / (mean_ms / 1e3) : 0.0;
  std::cout << "  [" << label << "]\n";
  std::cout << "    Mean  : " << mean_ms   << " ms\n";
  std::cout << "    FPS   : " << fps       << "\n";
  std::cout << "    Tput  : " << throughput << " GOps/s\n";
}

static TimerResult gpu_ms_to_timer(float mean_ms) {
  TimerResult t;
  t.mean_ms = mean_ms;
  t.min_ms  = mean_ms;
  t.std_ms  = 0.0;
  t.fps     = (mean_ms > 0) ? 1000.0 / mean_ms : 0.0;
  return t;
}

int main(int argc, char** argv) {
  RunOptions opt;
  parse_args(argc, argv, opt);
  const StereoParams& p = opt.p;

  std::cout << "\n====================================================\n";
  std::cout << "  Stereo SAD — GPU benchmark\n";
  std::cout << "====================================================\n";
  std::cout << "  Image size   : " << p.height << " x " << p.width << "\n";
  std::cout << "  Patch radius : " << p.radius
            << "  (patch = " << (2*p.radius+1) << "x" << (2*p.radius+1) << ")\n";
  std::cout << "  Max disparity: " << p.max_disp << "\n";
  std::cout << "  True disparity (GT): " << p.true_disp << "\n";
  std::cout << "  Repeats      : " << p.repeats << "\n";
  std::cout << "----------------------------------------------------\n";

  // Generate images
  Image left(p.height, p.width);
  Image right(p.height, p.width);
  generate_left_image(left);
  generate_right_image(left, right, p.true_disp);

  DisparityMap gt(p.height, p.width);
  generate_ground_truth(gt, p.true_disp, p.radius, p.max_disp);

  if (opt.save_images) {
    save_pgm(left,  "left.pgm");
    save_pgm(right, "right.pgm");
    std::cout << "  Saved: left.pgm, right.pgm\n";
  }

  double gops = compute_gops(p);
  std::cout << "  Compute (approx): " << gops << " GOps/frame\n\n";

  DisparityMap disp_out(p.height, p.width);

  // ---- CPU baseline ----
  if (opt.run_cpu) {
    TimerResult t_cpu = time_sad_stereo_cpu(left, right, disp_out,
                                            p.max_disp, p.radius, p.repeats);
    print_timing_result(t_cpu, "CPU");
    print_accuracy(disp_out, gt);
    if (!opt.csv_path.empty())
      append_benchmark_csv(opt.csv_path, "CPU", p, t_cpu, gops);
  }

  // ---- GPU basic ----
  {
    float ms = sad_stereo_gpu_basic(left, right, disp_out,
                                    p.max_disp, p.radius, p.repeats);
    print_gpu_result("GPU basic", ms, gops);
    print_accuracy(disp_out, gt);
    if (opt.save_images) {
      save_disparity_pgm(disp_out, "disp_basic.pgm", p.max_disp);
      std::cout << "  Saved: disp_basic.pgm\n";
    }
    if (!opt.csv_path.empty())
      append_benchmark_csv(opt.csv_path, "GPU_basic", p, gpu_ms_to_timer(ms), gops);
  }

  // ---- GPU shared memory ----
  {
    float ms = sad_stereo_gpu_smem(left, right, disp_out,
                                   p.max_disp, p.radius, p.repeats);
    print_gpu_result("GPU smem", ms, gops);
    print_accuracy(disp_out, gt);
    if (opt.save_images) {
      save_disparity_pgm(disp_out, "disp_smem.pgm", p.max_disp);
      std::cout << "  Saved: disp_smem.pgm\n";
    }
    if (!opt.csv_path.empty())
      append_benchmark_csv(opt.csv_path, "GPU_smem", p, gpu_ms_to_timer(ms), gops);
  }

  // ---- GPU tiled ----
  {
    float ms = sad_stereo_gpu_tiled(left, right, disp_out,
                                    p.max_disp, p.radius, p.repeats);
    print_gpu_result("GPU tiled", ms, gops);
    print_accuracy(disp_out, gt);
    if (opt.save_images) {
      save_disparity_pgm(disp_out, "disp_tiled.pgm", p.max_disp);
      save_disparity_pgm(gt,       "disp_gt.pgm",    p.max_disp);
      std::cout << "  Saved: disp_tiled.pgm, disp_gt.pgm\n";
    }
    if (!opt.csv_path.empty())
      append_benchmark_csv(opt.csv_path, "GPU_tiled", p, gpu_ms_to_timer(ms), gops);
  }

  std::cout << "====================================================\n\n";
  if (!opt.csv_path.empty())
    std::cout << "  Benchmark data written to: " << opt.csv_path << "\n";

  return 0;
}
