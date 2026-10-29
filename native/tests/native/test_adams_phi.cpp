// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/adams_phi.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

// The exponential Adams march's phi-functions on a host block's lanes
// (include/solprop_native/adams_phi.hpp) against the element-by-element `adams_phi`:
// every lane's values are the scalar form's bits, whatever its neighbours, on each of
// the three branches, at the branch and halving thresholds, and under a carrier frame.
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
namespace {
namespace native = solprop_native;
using native::Complex;

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double quiet_nan = std::numeric_limits<double>::quiet_NaN();

// The same bits, or NaN in both (a NaN's payload is not part of the contract).
bool same(const double a, const double b) {
  if (std::isnan(a) && std::isnan(b))
    return true;
  std::uint64_t x = 0;
  std::uint64_t y = 0;
  std::memcpy(&x, &a, sizeof x);
  std::memcpy(&y, &b, sizeof y);
  return x == y;
}

// The flow e^x as the pass forms it (any value serves: both forms take it as given).
Complex flow_of(const Complex x) {
  if (!std::isfinite(x.real()) || !std::isfinite(x.imag()))
    return {std::exp(x.real()), 0.0};
  return Kokkos::exp(x);
}

// One block of `values` through the lanes and each element through `adams_phi`.
void expect_block(const std::vector<Complex>& values, const std::size_t order,
                  const native::ExponentialWeights& weights) {
  ASSERT_LE(values.size(), native::adams_lane_width);
  native::AdamsLanes lanes;
  lanes.count = values.size();
  for (std::size_t b = 0; b < values.size(); ++b) {
    lanes.xr[b] = values[b].real();
    lanes.xi[b] = values[b].imag();
    const Complex flow = flow_of(values[b]);
    lanes.fr[b] = flow.real();
    lanes.fi[b] = flow.imag();
  }
  native::adams_lanes_classify(lanes, weights);
  // e^y as the scalar form's `Kokkos::exp(y)` gives it.
  for (std::size_t b = 0; b < values.size(); ++b) {
    const Complex e = Kokkos::exp(Complex(lanes.yr[b], lanes.yi[b]));
    lanes.er[b] = e.real();
    lanes.ei[b] = e.imag();
  }
  if (lanes.any_large)
    native::adams_lanes_large_exponential(lanes, weights);
  native::adams_lanes_phi(lanes, order);
  for (std::size_t b = 0; b < values.size(); ++b) {
    const Complex x = values[b];
    const Complex argument = weights.shifted ? x - Complex(0.0, weights.shift) : x;
    Complex exp_x(lanes.fr[b], lanes.fi[b]);
    if (weights.shifted)
      exp_x =
          native::strict_mul(exp_x, Complex(weights.unshift_re, weights.unshift_im));
    Complex phi[native::adams_max_samples + 1];
    native::adams_phi(argument, Kokkos::abs(argument), exp_x, order, phi);
    for (std::size_t k = 0; k <= order; ++k) {
      EXPECT_TRUE(same(lanes.pr[k][b], phi[k].real()) &&
                  same(lanes.pi[k][b], phi[k].imag()))
          << "x = " << x.real() << " + " << x.imag() << "i, order " << order << ", phi_"
          << k << ": lanes " << lanes.pr[k][b] << " + " << lanes.pi[k][b]
          << "i, scalar " << phi[k].real() << " + " << phi[k].imag() << "i";
    }
  }
}

// Arguments at and one ulp either side of |x| = t, on the real and imaginary axes and
// on the diagonal, where the squared magnitude and `hypot` round apart.
void add_threshold(std::vector<Complex>& out, const double t) {
  for (const double r : {std::nextafter(t, 0.0), t, std::nextafter(t, infinity)}) {
    out.emplace_back(r, 0.0);
    out.emplace_back(0.0, -r);
    const double d = r / std::sqrt(2.0);
    for (const double e : {std::nextafter(d, 0.0), d, std::nextafter(d, infinity)})
      out.emplace_back(-e, e);
  }
}

std::vector<Complex> edge_values() {
  std::vector<Complex> values = {
      {0.0, 0.0},       {-0.0, 0.0},      {0.0, -0.0},          {1e-300, 0.0},
      {0.0, 5e-324},    {1e-160, 1e-160}, {1e-20, -3e-21},      {-1e-8, 2e-8},
      {0.3, -0.4},      {-2.5, 1.5},      {3.0000001, 0.0},     {0.0, 1e5},
      {-40.0, 300.0},   {1e200, 1e200},   {infinity, 0.0},      {0.0, -infinity},
      {quiet_nan, 1.0}, {1.0, quiet_nan}, {infinity, quiet_nan}};
  for (const double t : {0.5, 1.0, 2.0, 3.0})
    add_threshold(values, t);
  return values;
}

native::ExponentialWeights unshifted() { return {}; }

native::ExponentialWeights shifted(const double sigma) {
  native::ExponentialWeights weights;
  weights.shifted = true;
  weights.shift = sigma;
  const Complex unshift = Kokkos::exp(Complex(0.0, -sigma));
  weights.unshift_re = unshift.real();
  weights.unshift_im = unshift.imag();
  return weights;
}

} // namespace

TEST(AdamsPhiLanes, EdgeArgumentsAreTheScalarFormsBits) {
  const auto values = edge_values();
  for (std::size_t order = 1; order <= native::adams_max_samples; ++order)
    for (std::size_t start = 0; start < values.size();
         start += native::adams_lane_width) {
      const std::size_t end = start + native::adams_lane_width < values.size()
                                  ? start + native::adams_lane_width
                                  : values.size();
      const std::vector<Complex> block(
          values.begin() + static_cast<std::ptrdiff_t>(start),
          values.begin() + static_cast<std::ptrdiff_t>(end));
      expect_block(block, order, unshifted());
      expect_block(block, order, shifted(0.75));
    }
}

TEST(AdamsPhiLanes, EachLaneIsItsOwnWhateverItsNeighbours) {
  // Mixed blocks of every length: small, series, near-threshold and large arguments
  // side by side, so that lanes stop their series, square and branch apart.
  std::mt19937_64 generator(20261029);
  std::uniform_real_distribution<double> phase(-3.2, 3.2);
  std::uniform_real_distribution<double> exponent(-12.0, 1.2);
  for (std::size_t length = 1; length <= native::adams_lane_width; ++length)
    for (std::size_t order = 1; order <= native::adams_max_samples; ++order) {
      std::vector<Complex> block(length);
      for (auto& value : block) {
        const double magnitude = std::pow(10.0, exponent(generator));
        const double angle = phase(generator);
        value = Complex(magnitude * std::cos(angle), magnitude * std::sin(angle));
      }
      expect_block(block, order, unshifted());
      expect_block(block, order, shifted(-0.3));
    }
}

TEST(AdamsPhiLanes, ThresholdsDecideAsHypotDoes) {
  // Every argument of a dense sweep across |x| = 1/2, 1, 2 and 3 at many angles.
  std::vector<Complex> values;
  for (const double t : {0.5, 1.0, 2.0, 3.0})
    for (int step = -40; step <= 40; ++step)
      for (const double angle : {0.0, 0.3, 0.7853981633974483, 1.2, 2.5}) {
        const double r = t * (1.0 + step * 0x1p-52);
        values.emplace_back(r * std::cos(angle), r * std::sin(angle));
      }
  for (std::size_t start = 0; start < values.size();
       start += native::adams_lane_width) {
    const std::size_t end = start + native::adams_lane_width < values.size()
                                ? start + native::adams_lane_width
                                : values.size();
    const std::vector<Complex> block(values.begin() +
                                         static_cast<std::ptrdiff_t>(start),
                                     values.begin() + static_cast<std::ptrdiff_t>(end));
    expect_block(block, 5, unshifted());
  }
}
#endif
