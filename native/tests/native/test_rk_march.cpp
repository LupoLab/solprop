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

namespace {
rk::Policy controller() {
  rk::Policy policy;
  policy.safety = 0.9;
  policy.beta1 = 0.7 / 5.0;
  policy.beta2 = 0.4 / 5.0;
  policy.min_factor = 0.2;
  policy.max_factor = 5.0;
  policy.lag = 1;
  policy.power = 5.0;
  policy.exponent = 1.0 / 5.0;
  policy.start_growth = 100.0;
  policy.dz_max = 1.0;
  return policy;
}
} // namespace

TEST(RungeKuttaMarch, TheControllerTakesPIControllersBranches) {
  const auto policy = controller();
  const double inf = INFINITY;
  // A non-finite norm is rejected, shrinks by the smallest factor and stays as it is.
  EXPECT_FALSE(rk::accepts(policy, inf));
  EXPECT_FALSE(rk::accepts(policy, NAN));
  EXPECT_EQ(rk::rejection_factor(policy, inf), policy.min_factor);
  EXPECT_EQ(rk::equivalent(policy, inf, 1.0, 2.0), inf);
  EXPECT_EQ(rk::equivalent(policy, 0.5, 1.0, 2.0), 0.5 * std::pow(2.0, 5.0));
  // A zero norm: the start-up rule grows by its cap; a rejection keeps the step.
  EXPECT_EQ(rk::start_factor(policy, 0.0), policy.start_growth);
  EXPECT_EQ(rk::rejection_factor(policy, 0.0), 1.0);
  // The lag's unit factors until the history holds lag + 1 norms.
  EXPECT_EQ(rk::propose(policy, {0.5}), 1.0);
  const double expected =
      (0.9 * std::pow(1.0 / 0.25, policy.beta1)) * std::pow(1.0 / 0.5, policy.beta2);
  EXPECT_EQ(rk::propose(policy, {0.5, 0.25, 0.75}), expected);
  // A zero current norm grows by the largest factor; the cap bounds a proposal.
  EXPECT_EQ(rk::propose(policy, {0.5, 0.0, 0.75}), policy.max_factor);
  EXPECT_EQ(rk::limited(policy, 0.75, 2.0), policy.dz_max);
}
