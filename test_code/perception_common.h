#ifndef PERCEPTION_COMMON_H_
#define PERCEPTION_COMMON_H_

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

// ---------------------------------------------------------------------------
// Pixel type (uint8, 0–255 grayscale)
// ---------------------------------------------------------------------------
using pixel_t  = uint8_t;
using disp_t   = int;      // disparity in pixels
using score_t  = uint32_t; // SAD accumulator

// ---------------------------------------------------------------------------
// Tolerances for correctness tests
// ---------------------------------------------------------------------------
#define DISP_TOL  1      // acceptable absolute disparity error (pixels)
#define BORDER_FILL 0    // value written to border pixels we cannot compute

// ---------------------------------------------------------------------------
// Lightweight error-checking macro (CPU-only; mirrors CUDA_CHECK style)
// ---------------------------------------------------------------------------
#define CHECK(cond, msg)                                           \
  do {                                                             \
    if (!(cond)) {                                                 \
      std::cerr << "CHECK failed: " << (msg) << "\n"              \
                << "  at " << __FILE__ << ":" << __LINE__ << "\n";\
      std::exit(1);                                                \
    }                                                              \
  } while (0)

// ---------------------------------------------------------------------------
// Simple 2-D image stored in row-major order (matches the CUDA layout we will
// use later: image[row * width + col]).
// ---------------------------------------------------------------------------
struct Image {
  int        height{0};
  int        width{0};
  pixel_t*   data{nullptr};   // owned, heap-allocated

  Image() = default;
  Image(int h, int w) : height(h), width(w) {
    data = new pixel_t[h * w]();
  }
  ~Image() { delete[] data; }

  // Non-copyable, movable
  Image(const Image&)            = delete;
  Image& operator=(const Image&) = delete;
  Image(Image&& o) noexcept
      : height(o.height), width(o.width), data(o.data) {
    o.data = nullptr;
  }

  pixel_t& at(int r, int c)       { return data[r * width + c]; }
  pixel_t  at(int r, int c) const { return data[r * width + c]; }
  int      size()            const { return height * width; }
};

// ---------------------------------------------------------------------------
// Disparity map: same layout as Image but stores signed disparity values.
// ---------------------------------------------------------------------------
struct DisparityMap {
  int     height{0};
  int     width{0};
  disp_t* data{nullptr};

  DisparityMap() = default;
  DisparityMap(int h, int w) : height(h), width(w) {
    data = new disp_t[h * w]();
  }
  ~DisparityMap() { delete[] data; }

  DisparityMap(const DisparityMap&)            = delete;
  DisparityMap& operator=(const DisparityMap&) = delete;
  DisparityMap(DisparityMap&& o) noexcept
      : height(o.height), width(o.width), data(o.data) {
    o.data = nullptr;
  }

  disp_t& at(int r, int c)       { return data[r * width + c]; }
  disp_t  at(int r, int c) const { return data[r * width + c]; }
  int     size()            const { return height * width; }
};

// ---------------------------------------------------------------------------
// Parameters that fully describe one stereo SAD run.
// Keeping them in one struct makes it easy to pass the same config to the
// CPU baseline and the future CUDA kernel.
// ---------------------------------------------------------------------------
struct StereoParams {
  int height     = 480;
  int width      = 640;
  int max_disp   = 64;   // search d in [0, max_disp)
  int radius     = 2;    // patch half-size → patch is (2r+1)×(2r+1)
  int true_disp  = 24;   // ground-truth shift used when generating the pair
  int repeats    = 5;    // timing repetitions
};

#endif  // PERCEPTION_COMMON_H_
