// ---------------------------------------------------------------------------
// perception_cpu_test.cpp
//
// GoogleTest suite for the CPU stereo SAD baseline.
// Structure mirrors gemm_test.cpp from the CME 213 homework:
//   – helper utilities declared at the top
//   – individual TEST() cases for each logical property
//   – an explicit accuracy threshold matching DISP_TOL from perception_common.h
//
// Build:
//   make test_cpu
// Run:
//   ./test_cpu
//   ./test_cpu --gtest_filter=StereoSAD.*
// ---------------------------------------------------------------------------

#include "gtest/gtest.h"
#include "perception_common.h"
#include "perception_cpu.h"

#include <cmath>
#include <numeric>

// ===========================================================================
// Test parameters (kept small so tests run quickly)
// ===========================================================================
static constexpr int T_HEIGHT   = 120;
static constexpr int T_WIDTH    = 160;
static constexpr int T_RADIUS   = 2;
static constexpr int T_MAX_DISP = 32;
static constexpr int T_DISP     = 8;   // ground-truth disparity

// ===========================================================================
// Helper: build a standard synthetic stereo pair for tests
// (analogous to fill<Layout> in gemm_test.cpp)
// ===========================================================================
static void make_stereo_pair(Image& left, Image& right,
                             DisparityMap& gt, int true_disp = T_DISP) {
  generate_left_image(left);
  generate_right_image(left, right, true_disp);
  generate_ground_truth(gt, true_disp, T_RADIUS, T_MAX_DISP);
}

// ===========================================================================
// ── Suite: ImageGeneration ──────────────────────────────────────────────── //
// ===========================================================================

TEST(ImageGeneration, LeftImageIsInRange) {
  Image left(T_HEIGHT, T_WIDTH);
  generate_left_image(left);

  for (int i = 0; i < left.size(); ++i) {
    EXPECT_GE(left.data[i], 0)
        << "Pixel " << i << " below 0";
    EXPECT_LE(left.data[i], 255)
        << "Pixel " << i << " above 255";
  }
}

TEST(ImageGeneration, LeftImageIsDeterministic) {
  // Two calls with the same size must produce bit-identical results
  Image a(T_HEIGHT, T_WIDTH);
  Image b(T_HEIGHT, T_WIDTH);
  generate_left_image(a);
  generate_left_image(b);

  for (int i = 0; i < a.size(); ++i) {
    ASSERT_EQ(a.data[i], b.data[i]) << "Mismatch at pixel " << i;
  }
}

TEST(ImageGeneration, RightImageIsShiftedLeft) {
  Image left(T_HEIGHT, T_WIDTH);
  Image right(T_HEIGHT, T_WIDTH);
  generate_left_image(left);
  generate_right_image(left, right, T_DISP);

  // Right(r, c) == Left(r, c + T_DISP) for c + T_DISP < width
  for (int r = 0; r < T_HEIGHT; ++r) {
    for (int c = 0; c + T_DISP < T_WIDTH; ++c) {
      ASSERT_EQ(right.at(r, c), left.at(r, c + T_DISP))
          << "Shift mismatch at (" << r << "," << c << ")";
    }
  }
}

TEST(ImageGeneration, GroundTruthValid) {
  DisparityMap gt(T_HEIGHT, T_WIDTH);
  generate_ground_truth(gt, T_DISP, T_RADIUS, T_MAX_DISP);

  // Valid pixels must have exactly T_DISP
  int valid_count = 0;
  for (int r = 0; r < T_HEIGHT; ++r) {
    for (int c = 0; c < T_WIDTH; ++c) {
      int g = gt.at(r, c);
      if (g >= 0) {
        EXPECT_EQ(g, T_DISP)
            << "GT pixel (" << r << "," << c << ") = " << g
            << " expected " << T_DISP;
        ++valid_count;
      }
    }
  }
  // There should be a meaningful number of valid pixels
  EXPECT_GT(valid_count, 0) << "No valid GT pixels found";
}

// ===========================================================================
// ── Suite: StereoSAD ────────────────────────────────────────────────────── //
// ===========================================================================

// Correctness: on a clean synthetic pair the MAE should be < 1 pixel
TEST(StereoSAD, AccuracyOnSyntheticPair) {
  Image left(T_HEIGHT, T_WIDTH);
  Image right(T_HEIGHT, T_WIDTH);
  DisparityMap gt(T_HEIGHT, T_WIDTH);
  DisparityMap out(T_HEIGHT, T_WIDTH);
  make_stereo_pair(left, right, gt);

  sad_stereo_cpu(left, right, out, T_MAX_DISP, T_RADIUS);

  double mae = compute_mae(out, gt);
  EXPECT_LT(mae, 1.0)
      << "MAE " << mae << " px exceeds 1 px on synthetic pair";
}

// Bad-pixel rate should be low on a perfect shift
TEST(StereoSAD, BadPixelRateLow) {
  Image left(T_HEIGHT, T_WIDTH);
  Image right(T_HEIGHT, T_WIDTH);
  DisparityMap gt(T_HEIGHT, T_WIDTH);
  DisparityMap out(T_HEIGHT, T_WIDTH);
  make_stereo_pair(left, right, gt);

  sad_stereo_cpu(left, right, out, T_MAX_DISP, T_RADIUS);

  double bad = compute_bad_pixel_rate(out, gt, DISP_TOL);
  EXPECT_LT(bad, 0.05)
      << "Bad-pixel rate " << bad * 100.0
      << "% exceeds 5% on synthetic pair";
}

// Disparity map dimensions must match the input
TEST(StereoSAD, OutputDimensionsMatch) {
  Image left(T_HEIGHT, T_WIDTH);
  Image right(T_HEIGHT, T_WIDTH);
  DisparityMap out(T_HEIGHT, T_WIDTH);
  generate_left_image(left);
  generate_right_image(left, right, T_DISP);

  sad_stereo_cpu(left, right, out, T_MAX_DISP, T_RADIUS);

  EXPECT_EQ(out.height, T_HEIGHT);
  EXPECT_EQ(out.width,  T_WIDTH);
}

// Border pixels must be BORDER_FILL (0), not some garbage value
TEST(StereoSAD, BorderPixelsAreZero) {
  Image left(T_HEIGHT, T_WIDTH);
  Image right(T_HEIGHT, T_WIDTH);
  DisparityMap out(T_HEIGHT, T_WIDTH);
  generate_left_image(left);
  generate_right_image(left, right, T_DISP);

  sad_stereo_cpu(left, right, out, T_MAX_DISP, T_RADIUS);

  for (int c = 0; c < T_WIDTH; ++c) {
    // Top border rows
    for (int r = 0; r < T_RADIUS; ++r)
      EXPECT_EQ(out.at(r, c), BORDER_FILL)
          << "Border pixel (" << r << "," << c << ") != BORDER_FILL";
    // Bottom border rows
    for (int r = T_HEIGHT - T_RADIUS; r < T_HEIGHT; ++r)
      EXPECT_EQ(out.at(r, c), BORDER_FILL)
          << "Border pixel (" << r << "," << c << ") != BORDER_FILL";
  }
}

// All estimated disparities must lie in [0, max_disp)
TEST(StereoSAD, EstimatedDisparityInRange) {
  Image left(T_HEIGHT, T_WIDTH);
  Image right(T_HEIGHT, T_WIDTH);
  DisparityMap out(T_HEIGHT, T_WIDTH);
  generate_left_image(left);
  generate_right_image(left, right, T_DISP);

  sad_stereo_cpu(left, right, out, T_MAX_DISP, T_RADIUS);

  for (int i = 0; i < out.size(); ++i) {
    EXPECT_GE(out.data[i], 0)
        << "Disparity at pixel " << i << " is negative";
    EXPECT_LT(out.data[i], T_MAX_DISP)
        << "Disparity at pixel " << i << " >= max_disp";
  }
}

// Vary patch radius: larger patch should not increase error
TEST(StereoSAD, LargerPatchDoesNotIncreaseMae) {
  Image left(T_HEIGHT, T_WIDTH);
  Image right(T_HEIGHT, T_WIDTH);
  DisparityMap gt2(T_HEIGHT, T_WIDTH);
  DisparityMap gt4(T_HEIGHT, T_WIDTH);
  DisparityMap out2(T_HEIGHT, T_WIDTH);
  DisparityMap out4(T_HEIGHT, T_WIDTH);

  generate_left_image(left);
  generate_right_image(left, right, T_DISP);

  generate_ground_truth(gt2, T_DISP, 2, T_MAX_DISP);
  generate_ground_truth(gt4, T_DISP, 4, T_MAX_DISP);

  sad_stereo_cpu(left, right, out2, T_MAX_DISP, 2);
  sad_stereo_cpu(left, right, out4, T_MAX_DISP, 4);

  double mae2 = compute_mae(out2, gt2);
  double mae4 = compute_mae(out4, gt4);

  // Both should be small; this is a sanity check, not a strict bound
  EXPECT_LT(mae2, 1.0) << "r=2 MAE too high";
  EXPECT_LT(mae4, 1.0) << "r=4 MAE too high";
}

// Determinism: two calls must produce identical output
TEST(StereoSAD, IsDeterministic) {
  Image left(T_HEIGHT, T_WIDTH);
  Image right(T_HEIGHT, T_WIDTH);
  DisparityMap outA(T_HEIGHT, T_WIDTH);
  DisparityMap outB(T_HEIGHT, T_WIDTH);

  generate_left_image(left);
  generate_right_image(left, right, T_DISP);

  sad_stereo_cpu(left, right, outA, T_MAX_DISP, T_RADIUS);
  sad_stereo_cpu(left, right, outB, T_MAX_DISP, T_RADIUS);

  for (int i = 0; i < outA.size(); ++i) {
    ASSERT_EQ(outA.data[i], outB.data[i])
        << "Non-deterministic output at pixel " << i;
  }
}

// ===========================================================================
// ── Suite: AccuracyMetrics ──────────────────────────────────────────────── //
// ===========================================================================

TEST(AccuracyMetrics, MAEZeroWhenPerfect) {
  DisparityMap est(4, 4);
  DisparityMap gt(4, 4);
  for (int i = 0; i < 16; ++i) {
    est.data[i] = 5;
    gt.data[i]  = 5;
  }
  EXPECT_DOUBLE_EQ(compute_mae(est, gt), 0.0);
}

TEST(AccuracyMetrics, MAECorrectWithKnownError) {
  DisparityMap est(1, 4);
  DisparityMap gt(1, 4);
  // est  = [0, 1, 2, 3]
  // gt   = [0, 0, 0, 0]   → errors = 0, 1, 2, 3   → MAE = 1.5
  for (int i = 0; i < 4; ++i) { est.data[i] = i; gt.data[i] = 0; }
  EXPECT_NEAR(compute_mae(est, gt), 1.5, 1e-9);
}

TEST(AccuracyMetrics, InvalidGTPixelsIgnored) {
  DisparityMap est(1, 4);
  DisparityMap gt(1, 4);
  // Only pixel 2 is valid
  gt.data[0] = -1; gt.data[1] = -1; gt.data[2] = 5; gt.data[3] = -1;
  est.data[2] = 5;
  EXPECT_DOUBLE_EQ(compute_mae(est, gt), 0.0);
}

TEST(AccuracyMetrics, BadPixelRateCorrect) {
  DisparityMap est(1, 4);
  DisparityMap gt(1, 4);
  // errors: 0, 0, 2, 2 with threshold 1 → 2/4 = 0.5
  gt.data[0] = 3; gt.data[1] = 3; gt.data[2] = 3; gt.data[3] = 3;
  est.data[0]= 3; est.data[1]= 3; est.data[2]= 5; est.data[3]= 5;
  EXPECT_NEAR(compute_bad_pixel_rate(est, gt, 1), 0.5, 1e-9);
}

// ===========================================================================
// ── GoogleTest main ─────────────────────────────────────────────────────── //
// ===========================================================================
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
