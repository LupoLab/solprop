// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The Runge-Kutta march's strict unit (solprop SEP 45). Its norms and steps are held to
// solprop's by solprop's tests; these check what those cannot see: that the unit rounds
// every product and sum separately, and the order of a stage's position and the norm.

#include "solprop_native/rk_march.hpp"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace {
namespace rk = solprop_native::rk;

volatile double factor_a = 1.0 + 0x1p-30;
volatile double factor_b = 1.0 - 0x1p-30;
volatile double addend = -1.0;

// This file is compiled with the build's contraction (solprop #135's control).
[[gnu::noinline]] double contracting(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}
} // namespace

TEST(RungeKuttaMarch, TheArithmeticRoundsEveryOperation) {
  EXPECT_EQ(rk::strict_probe(factor_a, factor_b, addend), 0.0);
  // A stage's position: the product rounded before the sum, as Python's.
  const auto at = rk::positions(addend, factor_b, {factor_a});
  EXPECT_EQ(at[0], 0.0);
  EXPECT_EQ(at[1], addend + factor_b);
}

TEST(RungeKuttaMarch, TheArithmeticStaysStrictWhereTheBuildFuses) {
#if defined(__OPTIMIZE__) && defined(__FP_FAST_FMA) && !defined(__CUDACC__) &&         \
    !defined(__HIPCC__)
  ASSERT_EQ(contracting(factor_a, factor_b, addend), -0x1p-60)
      << "the control did not fuse, so the witness proves nothing";
  EXPECT_EQ(rk::strict_probe(factor_a, factor_b, addend), 0.0);
#else
  const double value = contracting(factor_a, factor_b, addend);
  EXPECT_TRUE(value == 0.0 || value == -0x1p-60);
  GTEST_SKIP() << "an unoptimised build, or a target without fused multiply-add, "
                  "cannot show a contraction";
#endif
}

TEST(RungeKuttaMarch, TheNormIsTheWeakScaledNorm) {
  // Two slots: |y|^2 = 3 + 1, |y_new|^2 = 9 + 7, |e|^2 = 1 + 3.
  const std::vector<double> partials = {3.0, 1.0, 9.0, 7.0, 1.0, 3.0};
  EXPECT_EQ(rk::score(partials.data(), 2, 0.5, 0.0), 2.0 / (0.5 * 4.0));
  EXPECT_EQ(rk::score(partials.data(), 2, 0.5, 1.0), 2.0 / (0.5 * 4.0 + 1.0));
}
