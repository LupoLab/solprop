// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/core.hpp"
#include "solprop_native/envelope_plasma.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace {

namespace native = solprop_native;
using native::Complex;
using RuleView = Kokkos::View<double**, native::Layout, native::MemorySpace>;

struct PolynomialRate {
  KOKKOS_INLINE_FUNCTION double operator()(const double field) const {
    return field * field * field * field;
  }
};

std::array<RuleView, 2> polynomial_rules(const bool inconsistent = false) {
  RuleView nodes("envelope_test.nodes", 4, 512),
      weights("envelope_test.weights", 4, 512);
  auto host_nodes = Kokkos::create_mirror_view(nodes);
  auto host_weights = Kokkos::create_mirror_view(weights);
  for (std::size_t level = 0; level < 4; ++level) {
    const std::size_t order = std::size_t{64} << level;
    for (std::size_t index = 0; index < order; ++index) {
      // This finite Fourier polynomial is integrated exactly by quadrant midpoint
      // sums. These test rules isolate the traversal from production GL generation.
      host_nodes(level, index) =
          2.0 * (static_cast<double>(index) + 0.5) / static_cast<double>(order) - 1.0;
      host_weights(level, index) =
          (inconsistent && level % 2 != 0 ? 4.0 : 2.0) / static_cast<double>(order);
    }
  }
  Kokkos::deep_copy(nodes, host_nodes);
  Kokkos::deep_copy(weights, host_weights);
  return {nodes, weights};
}

void polynomial_moments_and_directional_work() {
  const auto rules = polynomial_rules();
  const auto& nodes = rules[0];
  const auto& weights = rules[1];
  const std::array<double, 5> ratios{0.0, 1e-12, 0.35, 1.0 - 1e-12, 1.0};
  RuleView input("envelope_test.ratio", ratios.size(), 1);
  auto host_input = Kokkos::create_mirror_view(input);
  for (std::size_t index = 0; index < ratios.size(); ++index)
    host_input(index, 0) = ratios[index];
  Kokkos::deep_copy(input, host_input);
  RuleView output("envelope_test.output", ratios.size(), 8);
  Kokkos::parallel_for(
      "envelope_test.polynomial",
      Kokkos::RangePolicy<native::ExecutionSpace>(native::ExecutionSpace{}, 0,
                                                  ratios.size()),
      KOKKOS_LAMBDA(const std::size_t index) {
        const double ratio = input(index, 0);
        const auto axes = native::envelope_axes({2.0, 0.0}, {0.0, 2.0 * ratio});
        const auto moments = native::envelope_moments(
            axes.major, axes.ratio, 0.0, 100.0, PolynomialRate{}, nodes, weights);
        const auto loss =
            native::envelope_loss(axes, moments.cosine, moments.sine, 1.0);
        output(index, 0) = axes.major;
        output(index, 1) = axes.ratio;
        output(index, 2) = moments.cosine;
        output(index, 3) = moments.sine;
        output(index, 4) = moments.failed ? 1.0 : 0.0;
        output(index, 5) = loss[0].real();
        output(index, 6) = loss[1].imag();
        output(index, 7) = loss[0].imag() + loss[1].real();
      });
  const auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  // Two ordered sums of at most 128 terms, plus bounded elementary arithmetic.
  constexpr double tolerance = 1024 * std::numeric_limits<double>::epsilon();
  for (std::size_t index = 0; index < ratios.size(); ++index) {
    const double ratio = ratios[index], square = ratio * ratio;
    const double mc = 2.0 * (3.0 + square), ms = 2.0 * (1.0 + 3.0 * square);
    EXPECT_NEAR(actual(index, 0), 2.0, tolerance);
    EXPECT_NEAR(actual(index, 1), ratio, tolerance);
    EXPECT_NEAR(actual(index, 2), mc, tolerance * mc);
    EXPECT_NEAR(actual(index, 3), ms, tolerance * ms);
    EXPECT_EQ(actual(index, 4), 0.0);
    EXPECT_NEAR(actual(index, 5), mc, tolerance * mc);
    EXPECT_NEAR(actual(index, 6), ratio * ms, tolerance * ratio * ms);
    EXPECT_NEAR(actual(index, 7), 0.0, tolerance);
    EXPECT_NEAR(actual(index, 5) + ratio * actual(index, 6), mc + square * ms,
                tolerance * (mc + square * ms));
  }
}

void phase_and_real_rotation() {
  constexpr double phase = 0.37, angle = 0.61;
  const Complex common(std::cos(phase), std::sin(phase));
  const Complex first = common * Complex(2.0 * std::cos(angle), -0.7 * std::sin(angle));
  const Complex second = common * Complex(2.0 * std::sin(angle), 0.7 * std::cos(angle));
  Kokkos::View<Complex*, native::MemorySpace> result("envelope_test.covariance", 2);
  Kokkos::parallel_for(
      "envelope_test.covariance",
      Kokkos::RangePolicy<native::ExecutionSpace>(native::ExecutionSpace{}, 0, 1),
      KOKKOS_LAMBDA(const int) {
        const auto axes = native::envelope_axes(first, second);
        const auto loss = native::envelope_loss(axes, 6.245, 2.735, 1.0);
        result(0) = loss[0];
        result(1) = loss[1];
      });
  const auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
  const Complex expected_x =
      common * Complex(6.245 * std::cos(angle), -0.95725 * std::sin(angle));
  const Complex expected_y =
      common * Complex(6.245 * std::sin(angle), 0.95725 * std::cos(angle));
  // The two rotations and scaled Gram decomposition contain fewer than 100 flops.
  constexpr double tolerance = 100 * std::numeric_limits<double>::epsilon();
  EXPECT_LE(Kokkos::abs(actual(0) - expected_x), tolerance * Kokkos::abs(expected_x));
  EXPECT_LE(Kokkos::abs(actual(1) - expected_y), tolerance * Kokkos::abs(expected_y));
}

void zero_subnormal_and_unresolved_cases() {
  const auto rules = polynomial_rules(true);
  const auto& nodes = rules[0];
  const auto& weights = rules[1];
  RuleView result("envelope_test.limits", 1, 5);
  Kokkos::parallel_for(
      "envelope_test.limits",
      Kokkos::RangePolicy<native::ExecutionSpace>(native::ExecutionSpace{}, 0, 1),
      KOKKOS_LAMBDA(const int) {
        const auto zero = native::envelope_axes({}, {});
        const auto tiny = native::envelope_axes({1e-310, 1e-310}, {});
        const auto loss = native::envelope_loss(tiny, 0.0, 0.0, 1.0);
        const auto below = native::envelope_moments(0.0, 0.0, 1e-8, 100.0,
                                                    PolynomialRate{}, nodes, weights);
        const auto unresolved = native::envelope_moments(
            2.0, 0.35, 0.0, 100.0, PolynomialRate{}, nodes, weights);
        result(0, 0) = zero.major;
        result(0, 1) = tiny.major;
        result(0, 2) = Kokkos::abs(loss[0]) + Kokkos::abs(loss[1]);
        result(0, 3) = below.cosine + below.sine + (below.failed ? 1.0 : 0.0);
        result(0, 4) = unresolved.failed ? 1.0 : 0.0;
      });
  const auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
  EXPECT_EQ(actual(0, 0), 0.0);
  EXPECT_TRUE(std::isfinite(actual(0, 1)));
  EXPECT_GT(actual(0, 1), 0.0);
  EXPECT_EQ(actual(0, 2), 0.0);
  EXPECT_EQ(actual(0, 3), 0.0);
  EXPECT_EQ(actual(0, 4), 1.0);
}

// CUDA extended lambdas require a publicly accessible enclosing function.
TEST(EnvelopePlasma, PolynomialMomentsAndDirectionalWorkOnExecutionSpace) {
  polynomial_moments_and_directional_work();
}

TEST(EnvelopePlasma, PhaseAndRealRotationPreserveBothLossComponents) {
  phase_and_real_rotation();
}

TEST(EnvelopePlasma, ZeroSubnormalAndUnresolvedCasesStayExplicit) {
  zero_subnormal_and_unresolved_cases();
}

} // namespace
