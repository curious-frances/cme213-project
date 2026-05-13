#ifndef PERCEPTION_COMMON_H_
#define PERCEPTION_COMMON_H_

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

using pixel_t  = uint8_t;
using disp_t   = int;
using score_t  = uint32_t;

#define DISP_TOL    1
#define BORDER_FILL 0

#define CHECK(cond, msg)                                           \
  do {                                                             \
    if (!(cond)) {                                                 \
      std::cerr << "CHECK failed: " << (msg) << "\n"              \
                << "  at " << __FILE__ << ":" << __LINE__ << "\n";\
      std::exit(1);                                                \
    }                                                              \
  } while (0)

struct Image {
  int      height{0};
  int      width{0};
  pixel_t* data{nullptr};

  Image() = default;
  Image(int h, int w) : height(h), width(w) {
    data = new pixel_t[h * w]();
  }
  ~Image() { delete[] data; }

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

struct StereoParams {
  int height    = 480;
  int width     = 640;
  int max_disp  = 64;
  int radius    = 2;
  int true_disp = 24;
  int repeats   = 5;
};

#endif
