// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "bounds.hpp"
#include "solprop_native/vmath.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

// The host vector-maths layer (include/solprop_native/vmath.hpp): every block length,
// the special values, and agreement with the system libm. Its accuracy against a
// high-precision reference over the kernels' argument ranges is
// tests/test_vector_math.py.
namespace {
namespace vmath = solprop_native::vmath;
using solprop_native::test::kUnitRoundoff;

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double quiet_nan = std::numeric_limits<double>::quiet_NaN();

// Lengths around every multiple of the vector width up to three, and a stage block.
std::vector<std::size_t> lengths() {
  std::vector<std::size_t> result;
  for (std::size_t n = 0; n <= 3 * vmath::width + 1; ++n)
    result.push_back(n);
  for (const std::size_t n : {63U, 64U, 65U, 127U})
    result.push_back(n);
  return result;
}

std::vector<double> uniform(const std::size_t n, const double lower, const double upper,
                            const unsigned seed) {
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<double> draw(lower, upper);
  std::vector<double> values(n);
  for (double& value : values)
    value = draw(generator);
  return values;
}

// Two results each within one ulp of the same exact value are within two ulps of
// each other: 4 u of the larger magnitude, or two subnormal spacings.
void expect_within_two_ulps(const double value, const double reference) {
  const double scale = std::fmax(std::fabs(value), std::fabs(reference));
  const double bound = std::fmax(4.0 * kUnitRoundoff * scale,
                                 2.0 * std::numeric_limits<double>::denorm_min());
  EXPECT_LE(std::fabs(value - reference), bound) << value << " vs " << reference;
}

} // namespace

TEST(VectorMath, EveryElementOfABlockIsItsOwnValue) {
  // A block's last partial vector is padded, not computed another way: each element's
  // result is the one a single-element call gives, at every length and position.
  for (const std::size_t n : lengths()) {
    const auto x = uniform(n, -30.0, 5.0, 11U + static_cast<unsigned>(n));
    const auto angle = uniform(n, -400.0, 400.0, 29U + static_cast<unsigned>(n));
    std::vector<double> e(n), l(n), s(n), c(n), re(n), im(n);
    std::vector<double> positive(n);
    for (std::size_t i = 0; i < n; ++i)
      positive[i] = std::exp(x[i]);
    vmath::exp(x.data(), e.data(), n);
    vmath::log(positive.data(), l.data(), n);
    vmath::sincos(angle.data(), s.data(), c.data(), n);
    vmath::exp_complex(x.data(), angle.data(), re.data(), im.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
      double one_e = 0.0, one_l = 0.0, one_s = 0.0, one_c = 0.0, one_re = 0.0,
             one_im = 0.0;
      vmath::exp(&x[i], &one_e, 1);
      vmath::log(&positive[i], &one_l, 1);
      vmath::sincos(&angle[i], &one_s, &one_c, 1);
      vmath::exp_complex(&x[i], &angle[i], &one_re, &one_im, 1);
      EXPECT_EQ(e[i], one_e);
      EXPECT_EQ(l[i], one_l);
      EXPECT_EQ(s[i], one_s);
      EXPECT_EQ(c[i], one_c);
      EXPECT_EQ(re[i], one_re);
      EXPECT_EQ(im[i], one_im);
    }
  }
}

TEST(VectorMath, InPlaceGivesTheSameValues) {
  const std::size_t n = 2 * vmath::width + 1;
  const auto x = uniform(n, -10.0, 10.0, 3U);
  std::vector<double> out(n);
  vmath::exp(x.data(), out.data(), n);
  auto in_place = x;
  vmath::exp(in_place.data(), in_place.data(), n);
  EXPECT_EQ(in_place, out);
  std::vector<double> s(n), c(n);
  vmath::sincos(x.data(), s.data(), c.data(), n);
  auto sine = x;
  vmath::sincos(sine.data(), sine.data(), c.data(), n);
  EXPECT_EQ(sine, s);
}

TEST(VectorMath, AgreesWithTheSystemLibm) {
  // Each side is within one ulp of the exact value (SLEEF's u10 functions; glibc's
  // and Apple's libm), over the ranges the kernels form and beyond: the propagators'
  // and moments' exponents, the plasma's ionisation-rate exponents and logarithms of
  // field magnitudes, and phases up to the largest a step's sum reaches.
  const std::size_t n = 4099;
  const auto exponent = uniform(n, -745.0, 709.0, 5U);
  const auto small = uniform(n, -40.0, 1.0, 7U);
  const auto magnitude_exponent = uniform(n, -300.0, 300.0, 13U);
  const auto phase = uniform(n, -1.0e4, 1.0e4, 17U);
  const auto huge_phase = uniform(n, -1.0e15, 1.0e15, 19U);
  std::vector<double> out(n), s(n), c(n), re(n), im(n), magnitude(n);
  for (const auto* values : {&exponent, &small}) {
    vmath::exp(values->data(), out.data(), n);
    for (std::size_t i = 0; i < n; ++i)
      expect_within_two_ulps(out[i], std::exp((*values)[i]));
  }
  for (std::size_t i = 0; i < n; ++i)
    magnitude[i] = std::pow(10.0, magnitude_exponent[i]);
  vmath::log(magnitude.data(), out.data(), n);
  for (std::size_t i = 0; i < n; ++i)
    expect_within_two_ulps(out[i], std::log(magnitude[i]));
  for (const auto* values : {&phase, &huge_phase}) {
    vmath::sincos(values->data(), s.data(), c.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
      // Near a zero of sin or cos one ulp of the result is far below one ulp of the
      // partner value, so each is compared at its own magnitude.
      expect_within_two_ulps(s[i], std::sin((*values)[i]));
      expect_within_two_ulps(c[i], std::cos((*values)[i]));
    }
  }
  vmath::exp_complex(small.data(), phase.data(), re.data(), im.data(), n);
  for (std::size_t i = 0; i < n; ++i) {
    // Each side: the exponential and the cosine (sine) within one ulp and the product
    // rounded once, within 5 u |e^a| of the exact component; the two within 10 u.
    const double m = std::exp(small[i]);
    const double bound = 10.0 * kUnitRoundoff * m;
    EXPECT_LE(std::fabs(re[i] - m * std::cos(phase[i])), bound);
    EXPECT_LE(std::fabs(im[i] - m * std::sin(phase[i])), bound);
  }
}

TEST(VectorMath, SpecialValuesFollowC99) {
  const std::vector<double> x = {0.0,       -0.0,  infinity, -infinity,
                                 quiet_nan, 710.0, -746.0,   -740.0};
  const std::size_t n = x.size();
  std::vector<double> e(n);
  vmath::exp(x.data(), e.data(), n);
  EXPECT_EQ(e[0], 1.0);
  EXPECT_EQ(e[1], 1.0);
  EXPECT_EQ(e[2], infinity);
  EXPECT_EQ(e[3], 0.0);
  EXPECT_TRUE(std::isnan(e[4]));
  EXPECT_EQ(e[5], infinity);
  EXPECT_EQ(e[6], 0.0);
  // A subnormal result, within one subnormal spacing of the system libm's.
  EXPECT_GT(e[7], 0.0);
  EXPECT_LE(std::fabs(e[7] - std::exp(-740.0)),
            2.0 * std::numeric_limits<double>::denorm_min());

  const std::vector<double> positive = {1.0,
                                        0.0,
                                        -0.0,
                                        infinity,
                                        -1.0,
                                        quiet_nan,
                                        std::numeric_limits<double>::denorm_min()};
  std::vector<double> l(positive.size());
  vmath::log(positive.data(), l.data(), positive.size());
  EXPECT_EQ(l[0], 0.0);
  EXPECT_EQ(l[1], -infinity);
  EXPECT_EQ(l[2], -infinity);
  EXPECT_EQ(l[3], infinity);
  EXPECT_TRUE(std::isnan(l[4]));
  EXPECT_TRUE(std::isnan(l[5]));
  expect_within_two_ulps(l[6], std::log(std::numeric_limits<double>::denorm_min()));

  const std::vector<double> angle = {0.0, -0.0, infinity, -infinity, quiet_nan};
  std::vector<double> s(angle.size()), c(angle.size());
  vmath::sincos(angle.data(), s.data(), c.data(), angle.size());
  EXPECT_EQ(s[0], 0.0);
  EXPECT_FALSE(std::signbit(s[0]));
  EXPECT_EQ(c[0], 1.0);
  EXPECT_EQ(s[1], 0.0);
  EXPECT_TRUE(std::signbit(s[1]));
  EXPECT_EQ(c[1], 1.0);
  for (std::size_t i = 2; i < angle.size(); ++i) {
    EXPECT_TRUE(std::isnan(s[i]));
    EXPECT_TRUE(std::isnan(c[i]));
  }

  // A lossless exponent (a signed-zero real part) has magnitude one exactly, and a
  // NaN in either part makes both parts NaN.
  const std::vector<double> a = {0.0, -0.0, quiet_nan, 0.0};
  const std::vector<double> b = {0.5, -0.5, 0.5, quiet_nan};
  std::vector<double> re(a.size()), im(a.size());
  vmath::exp_complex(a.data(), b.data(), re.data(), im.data(), a.size());
  for (std::size_t i = 0; i < 2; ++i) {
    double sine = 0.0, cosine = 0.0;
    vmath::sincos(&b[i], &sine, &cosine, 1);
    EXPECT_EQ(re[i], cosine);
    EXPECT_EQ(im[i], sine);
  }
  for (std::size_t i = 2; i < a.size(); ++i) {
    EXPECT_TRUE(std::isnan(re[i]));
    EXPECT_TRUE(std::isnan(im[i]));
  }
}

TEST(VectorMath, RouteMatchesTheBuild) {
#if defined(SOLPROP_NATIVE_VECTOR_MATH_SLEEF)
  EXPECT_TRUE(vmath::vectorised);
  EXPECT_GE(vmath::width, 2U);
#else
  EXPECT_FALSE(vmath::vectorised);
  EXPECT_EQ(vmath::width, 1U);
#endif
}
