// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The boundary watch's unit (solprop SEP 45 phase 1). Its predictions are held to
// solprop's monitor on drawn readings by solprop's tests; these check what those cannot
// see: that the unit rounds every product and sum separately, that a policy is refused
// by the field at fault, and the split limit's sequential order.

#include "solprop_native/watch.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {
namespace watch = solprop_native::watch;

volatile double factor_a = 1.0 + 0x1p-30;
volatile double factor_b = 1.0 - 0x1p-30;
volatile double addend = -1.0;

// This file is compiled with the build's contraction (solprop #135's control).
[[gnu::noinline]] double contracting(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}

watch::Policy policy() {
  watch::Policy policy;
  policy.activity = 1e-8;
  policy.warn = 1e-3;
  policy.stop = 1e-2;
  policy.quiet_share = 1e-12;
  policy.quiet_checks = 2;
  policy.cadence = 8;
  policy.interval = 1e-3;
  return policy;
}
} // namespace

TEST(BoundaryWatch, ThePredictionsRoundEveryOperation) {
  EXPECT_EQ(watch::strict_probe(factor_a, factor_b, addend), 0.0);
}

TEST(BoundaryWatch, ThePredictionsStayStrictWhereTheBuildFuses) {
#if defined(__OPTIMIZE__) && defined(__FP_FAST_FMA) && !defined(__CUDACC__) &&         \
    !defined(__HIPCC__)
  ASSERT_EQ(contracting(factor_a, factor_b, addend), -0x1p-60)
      << "the control did not fuse, so the witness proves nothing";
  EXPECT_EQ(watch::strict_probe(factor_a, factor_b, addend), 0.0);
#else
  const double value = contracting(factor_a, factor_b, addend);
  EXPECT_TRUE(value == 0.0 || value == -0x1p-60);
  GTEST_SKIP() << "an unoptimised build, or a target without fused multiply-add, "
                  "cannot show a contraction";
#endif
}

TEST(BoundaryWatch, APolicyIsRefusedByTheFieldAtFault) {
  auto bad = policy();
  bad.stop = 1e-4;
  try {
    watch::validate(bad);
    FAIL() << "a stop threshold below the warning was accepted";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("threshold"), std::string::npos);
  }
  bad = policy();
  bad.crossing = true;
  EXPECT_THROW(watch::validate(bad), std::invalid_argument);
  EXPECT_NO_THROW(watch::validate(policy()));
}

TEST(BoundaryWatch, TheSplitLimitSumsInOrder) {
  // The near-tie of solprop's SEP 45 section 4.8: the sequential total selects bucket
  // 0.
  auto rules = policy();
  rules.crossing = true;
  rules.share = 1e-6;
  rules.bucket_limits.resize(watch::buckets);
  for (std::size_t index = 0; index < watch::buckets; ++index)
    rules.bucket_limits[index] = static_cast<double>(index + 1);
  std::vector<double> occupancy(watch::buckets, 0x1p-54);
  occupancy[0] = 0x1.0c6f8ba2f9813p-20;
  occupancy[1] = 1.0;
  EXPECT_EQ(watch::split_limit(rules, occupancy.data()), 1.0);
  std::vector<double> empty(watch::buckets, 0.0);
  EXPECT_TRUE(std::isinf(watch::split_limit(rules, empty.data())));
}

TEST(BoundaryWatch, ABusyCheckKeepsEveryStepDue) {
  const auto rules = policy();
  watch::State state;
  EXPECT_TRUE(watch::due(rules, state, 0.0)); // before the first check
  std::vector<double> reading(watch::reading_head, 0.0);
  reading[0] = 1.0;
  reading[1] = 4e-6; // a guard amplitude of 2e-3: busy, and above the warning
  EXPECT_EQ(watch::check(rules, state, reading.data(), 1e-4), watch::Verdict::warn);
  EXPECT_TRUE(state.busy && state.warned);
  EXPECT_TRUE(watch::due(rules, state, 1e-4));
  reading[1] = 1e-20;
  EXPECT_EQ(watch::check(rules, state, reading.data(), 2e-4), watch::Verdict::none);
  EXPECT_FALSE(state.busy);
  EXPECT_FALSE(watch::due(rules, state, 2e-4));
  for (int step = 0; step < 6; ++step)
    watch::skip(state);
  EXPECT_FALSE(watch::due(rules, state, 2e-4));
  watch::skip(state);
  EXPECT_TRUE(watch::due(rules, state, 2e-4)); // the cadence
  reading[1] = 0.04;                           // an amplitude of 0.2, above the stop
  EXPECT_EQ(watch::check(rules, state, reading.data(), 3e-4), watch::Verdict::stop);
}
