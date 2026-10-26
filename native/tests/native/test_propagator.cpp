// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/propagator.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <cfloat>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <vector>

// solprop SEP 36 section 4.3 and 7.2: the binary32 propagator from a binary64
// argument, against a binary64 reference whose own relative error (a few binary64
// ulp, the library's exp, cos and sin with exact reduction) is far below binary32's.
namespace {
namespace native = solprop_native;
constexpr double u32 = 0x1p-24;
// The stated accuracy: at most 10 u32 relative.
constexpr double stated = 10.0 * u32;

std::complex<double> reference(const double x, const double theta) {
  return std::exp(x) * std::complex<double>(std::cos(theta), std::sin(theta));
}

double relative_error(const Kokkos::complex<float> value, const double x,
                      const double theta) {
  const auto expected = reference(x, theta);
  const std::complex<double> got(static_cast<double>(value.real()),
                                 static_cast<double>(value.imag()));
  return std::abs(got - expected) / std::abs(expected);
}

std::vector<double> moduli() {
  // Up to binary32's damping limit ln(FLT_MAX), both signs, near multiples of ln 2 and
  // just past them, and SEP 36's x = -80.000003, where rounding x alone would cost 50
  // u32.
  const double limit = std::log(static_cast<double>(FLT_MAX));
  std::vector<double> values{0.0,          -80.000003, 80.000003, limit,
                             -limit + 1.0, 1e-30,      -1e-30};
  for (int m = -126; m <= 127; m += 7) {
    const double base = m * std::log(2.0);
    for (const double offset : {0.0, 1e-12, -1e-12, 0.3465, -0.3466})
      if (std::abs(base + offset) <= limit && base + offset > -limit + 1.0)
        values.push_back(base + offset);
  }
  return values;
}

std::vector<double> phases() {
  // Up to and beyond 2^30 rad, near multiples of 2 pi and just past them.
  std::vector<double> values{0.0,    1.0,     -1.0,         3.14159, 1e4,   -1e4,
                             0x1p30, -0x1p30, 0x1p30 + 1.0, 1e14,    -1e14, 1e15};
  const double two_pi = 2.0 * std::acos(-1.0);
  for (const double q : {1.0, 7.0, 1e3, 12345.0, 1e6, 1.6e8}) {
    const double base = q * two_pi;
    for (const double offset : {0.0, 1e-9, -1e-9, 1e-15 * base, 3.0})
      values.push_back(base + offset);
  }
  return values;
}
} // namespace

TEST(Propagator, BinaryThirtyTwoIsWithinTenUnitsOfTheReference) {
  double largest = 0.0;
  for (const double x : moduli())
    for (const double theta : phases()) {
      const auto value =
          native::propagator_exp<float>(Kokkos::complex<double>(x, theta));
      const auto pair =
          native::propagator_pair<float>(Kokkos::complex<double>(x, theta));
      const double error = relative_error(value, x, theta);
      largest = std::max(largest, error);
      EXPECT_LE(error, stated) << "x = " << x << ", theta = " << theta;
      EXPECT_EQ(pair.forward, value) << "x = " << x << ", theta = " << theta;
      // The inverse propagator, exp(-x - i theta).
      EXPECT_LE(relative_error(pair.inverse, -x, -theta), stated)
          << "x = " << x << ", theta = " << theta;
    }
  RecordProperty("largest_relative_error_in_u32", std::to_string(largest / u32));
}

TEST(Propagator, RoundingTheArgumentAloneWouldNotMeetTheBound) {
  // The reason for the reduction: exp of the rounded x is 50 u32 off at x = -80.000003.
  const double x = -80.000003;
  const double naive = std::exp(static_cast<double>(static_cast<float>(x)));
  EXPECT_GT(std::abs(naive - std::exp(x)) / std::exp(x), 40.0 * u32);
  EXPECT_LE(relative_error(
                native::propagator_exp<float>(Kokkos::complex<double>(x, 0.0)), x, 0.0),
            stated);
}

TEST(Propagator, RangeIsDecidedByThePowerOfTwo) {
  const double limit = std::log(static_cast<double>(FLT_MAX));
  EXPECT_TRUE(std::isinf(native::propagator_exp<float>({limit + 1.0, 0.0}).real()));
  EXPECT_TRUE(std::isinf(native::propagator_exp<float>({2000.0, 0.0}).real()));
  EXPECT_EQ(native::propagator_exp<float>({-2000.0, 0.0}).real(), 0.0F);
  // A subnormal result keeps its leading bits.
  const auto small = native::propagator_exp<float>({-95.0, 0.0}).real();
  EXPECT_GT(small, 0.0F);
  EXPECT_LT(small, FLT_MIN);
  EXPECT_LE(std::abs(static_cast<double>(small) - std::exp(-95.0)),
            std::exp(-95.0) * 0x1p-14);
  EXPECT_TRUE(std::isnan(
      native::propagator_exp<float>({std::numeric_limits<double>::quiet_NaN(), 0.0})
          .real()));
}

TEST(Propagator, BinarySixtyFourIsTheLibraryExpression) {
  for (const double x : {-3.0, 0.0, 2.5})
    for (const double theta : {0.0, 1.0, 1e6}) {
      const auto value = native::propagator_exp<double>({x, theta});
      EXPECT_EQ(value.real(), std::exp(x) * std::cos(theta));
      EXPECT_EQ(value.imag(), std::exp(x) * std::sin(theta));
      const auto pair = native::propagator_pair<double>({x, theta});
      EXPECT_EQ(pair.forward, value);
      EXPECT_EQ(pair.inverse.real(), std::exp(-x) * std::cos(theta));
      EXPECT_EQ(pair.inverse.imag(), -(std::exp(-x) * std::sin(theta)));
    }
}

TEST(Propagator, BinaryThirtyTwoIsTheBinary64ProductRoundedOncePerComponent) {
  // The oracle's `exp(dPhi).astype(complex64)`: one rounding per component. Rounding
  // the modulus and the phasor apart before a binary32 product doubled the coherent
  // drift of a constant step.
  for (const double x : moduli())
    for (const double theta : phases()) {
      const auto value = native::propagator_exp<float>({x, theta});
      const auto wide = native::propagator_exp<double>({x, theta});
      EXPECT_EQ(value.real(), static_cast<float>(wide.real()));
      EXPECT_EQ(value.imag(), static_cast<float>(wide.imag()));
      const auto pair = native::propagator_pair<float>({x, theta});
      const auto wide_pair = native::propagator_pair<double>({x, theta});
      EXPECT_EQ(pair.inverse.real(), static_cast<float>(wide_pair.inverse.real()));
      EXPECT_EQ(pair.inverse.imag(), static_cast<float>(wide_pair.inverse.imag()));
    }
}

TEST(Propagator, ModulusDriftOverManyConstantStepsIsOfOrderStepsTimesUnit) {
  // M constant unit-modulus steps multiplied in binary32: the modulus drifts by about
  // c M u32 (SEP 33 section 4.1, contribution 3), here with c = 2 for the product's
  // and the phasor's rounding.
  constexpr int steps = 10000;
  for (const double theta : {0.7, 123.4, 1e5}) {
    const auto step = native::propagator_exp<float>({0.0, theta});
    Kokkos::complex<float> value(1.0F, 0.0F);
    for (int index = 0; index < steps; ++index)
      value = value * step;
    const double modulus = std::hypot(static_cast<double>(value.real()),
                                      static_cast<double>(value.imag()));
    EXPECT_LE(std::abs(modulus - 1.0), 2.0 * steps * u32) << "theta = " << theta;
  }
}

// Named, not anonymous: an extended device lambda needs an enclosing function with
// external linkage under nvcc.
namespace solprop_native_test {
// The mean of |p|^2 - 1 over `count` phases evenly spread over (-scale pi, scale pi],
// for the binary32 propagator formed in the default execution space (a device on a
// device build), in units of u32.
double device_modulus_bias(const int count, const double scale) {
  double total = 0.0;
  const double pi = std::acos(-1.0);
  Kokkos::parallel_reduce(
      "solprop_native.test.propagator_bias",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, count),
      KOKKOS_LAMBDA(const int index, double& update) {
        const double theta = scale * (-pi + 2.0 * pi * (index + 0.5) / count);
        const auto value = native::propagator_exp<float>({0.0, theta});
        const auto real = static_cast<double>(value.real());
        const auto imag = static_cast<double>(value.imag());
        update += real * real + imag * imag - 1.0;
      },
      total);
  return total / count / u32;
}
} // namespace solprop_native_test

TEST(Propagator, TheBinaryThirtyTwoModulusIsUnbiasedWhereItRuns) {
  // A propagator multiplies the state once per step, so a mean bias of its squared
  // modulus is an energy drift linear in the step count, which Kerr phase turns into
  // a coherent field error (CUDA's sinf and cosf averaged +0.30 u32 on an A40). Rounded
  // binary64 values average zero: over 2^22 phases the mean's spread is about
  // 2^-11 u32, so 0.02 u32 is many standard deviations from both.
  for (const double scale : {1.0, 1e4})
    EXPECT_LE(std::abs(solprop_native_test::device_modulus_bias(1 << 22, scale)), 0.02)
        << "phases up to " << scale << " pi";
}
