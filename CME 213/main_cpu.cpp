// ---------------------------------------------------------------------------
// main_cpu.cpp
//
// Driver for the CPU stereo SAD baseline.
// Mirrors the structure of main_q1.cu / main_q2.cu in the CME 213 homework:
//   – parse command-line args
//   – generate synthetic stereo pair
//   – run and time the function
//   – print a structured report
//
// Usage:
//   ./main_cpu [--height H] [--width W] [--disp D] [--max-disp MD]
//              [--radius R] [--repeats N]
// ---------------------------------------------------------------------------

#include <cstring>
#include <iostream>
#include <string>

#include "perception_common.h"
#include "perception_cpu.h"

// ---------------------------------------------------------------------------
// CLI parsing — keeps the code readable without pulling in a third-party lib
// ---------------------------------------------------------------------------
static void parse_args(int argc, char** argv, StereoParams& p) {
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if ((key == "--height"   || key == "-H") && i + 1 < argc) { p.height    = std::stoi(argv[++i]); continue; }
    if ((key == "--width"    || key == "-W") && i + 1 < argc) { p.width     = std::stoi(argv[++i]); continue; }
    if ((key == "--disp"     || key == "-d") && i + 1 < argc) { p.true_disp = std::stoi(argv[++i]); continue; }
    if ((key == "--max-disp" || key == "-m") && i + 1 < argc) { p.max_disp  = std::stoi(argv[++i]); continue; }
    if ((key == "--radius"   || key == "-r") && i + 1 < argc) { p.radius    = std::stoi(argv[++i]); continue; }
    if ((key == "--repeats"  || key == "-n") && i + 1 < argc) { p.repeats   = std::stoi(argv[++i]); continue; }
    if (key == "--help" || key == "-h") {
      std::cout << "Usage: ./main_cpu [options]\n"
                << "  --height   H   image height          (default 480)\n"
                << "  --width    W   image width           (default 640)\n"
                << "  --disp     D   true disparity (GT)   (default 24)\n"
                << "  --max-disp MD  disparity search range (default 64)\n"
                << "  --radius   R   patch half-size        (default 2)\n"
                << "  --repeats  N   timing repetitions     (default 5)\n";
      std::exit(0);
    }
    std::cerr << "Unknown argument: " << key << "\n";
    std::exit(1);
  }

  // Sanity checks — same style as CHECK() macros in gemm_test.cpp
  CHECK(p.height > 0 && p.width > 0, "Image dimensions must be positive");
  CHECK(p.radius > 0,                  "Radius must be > 0");
  CHECK(p.max_disp > 0,               "max_disp must be > 0");
  CHECK(p.true_disp >= 0 && p.true_disp < p.max_disp,
        "true_disp must be in [0, max_disp)");
  CHECK(p.repeats > 0, "repeats must be > 0");
  CHECK(p.width  > 2 * p.radius + p.max_disp,
        "Image too narrow for given radius and max_disp");
  CHECK(p.height > 2 * p.radius,
        "Image too short for given radius");
}

// ---------------------------------------------------------------------------
// run_case — analogous to run_case<LA,LB,LC> in gemm_test.cpp:
//   allocates data, runs timing, prints report.
// ---------------------------------------------------------------------------
static void run_case(const StereoParams& p) {
  print_stereo_header(p);

  // ------------------------------------------------------------------
  // 1. Allocate images and disparity maps
  // ------------------------------------------------------------------
  Image left(p.height, p.width);
  Image right(p.height, p.width);
  DisparityMap disp_out(p.height, p.width);
  DisparityMap gt(p.height, p.width);

  // ------------------------------------------------------------------
  // 2. Generate synthetic stereo pair and ground truth
  // ------------------------------------------------------------------
  generate_left_image(left);
  generate_right_image(left, right, p.true_disp);
  generate_ground_truth(gt, p.true_disp, p.radius, p.max_disp);

  // ------------------------------------------------------------------
  // 3. Warm-up run (not counted in timing — mirrors DWarmup() pattern)
  // ------------------------------------------------------------------
  sad_stereo_cpu(left, right, disp_out, p.max_disp, p.radius);

  // ------------------------------------------------------------------
  // 4. Timed runs
  // ------------------------------------------------------------------
  TimerResult t = time_sad_stereo_cpu(left, right, disp_out,
                                      p.max_disp, p.radius, p.repeats);
  print_timing_result(t, "CPU SAD");

  // ------------------------------------------------------------------
  // 5. Accuracy report
  // ------------------------------------------------------------------
  print_accuracy(disp_out, gt);

  // ------------------------------------------------------------------
  // 6. Theoretical compute cost (for context, used in M3 report)
  //    Each valid pixel: max_disp * (2r+1)^2 ADD + ABS operations
  // ------------------------------------------------------------------
  long patch   = static_cast<long>((2 * p.radius + 1)) * (2 * p.radius + 1);
  long valid_w = p.width  - 2 * p.radius - p.max_disp;
  long valid_h = p.height - 2 * p.radius;
  if (valid_w < 0) valid_w = 0;
  long ops     = valid_h * valid_w * p.max_disp * patch * 2L; // 2: sub + abs
  double gops  = static_cast<double>(ops) / 1e9;
  double tput  = (t.mean_ms > 0) ? gops / (t.mean_ms / 1e3) : 0.0;
  std::cout << "  Compute (approx): " << gops  << " GOps/frame\n";
  std::cout << "  Throughput      : " << tput  << " GOps/s  (CPU, single-threaded)\n";
  std::cout << "====================================================\n\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  StereoParams p;           // starts with default values from perception_common.h
  parse_args(argc, argv, p);
  run_case(p);
  return 0;
}
