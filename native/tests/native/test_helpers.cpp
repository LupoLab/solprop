// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/chebyshev.hpp"
#include "solprop_native/lagrange_basis.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>

namespace {
namespace chebyshev = solprop_native::chebyshev;
namespace lagrange = solprop_native::lagrange;
using chebyshev::Complex;

constexpr double eps = std::numeric_limits<double>::epsilon();

TEST(Chebyshev, ColumnIsNumpysChebval) {
  // 1 + 2 T1 + 3 T2 at x = 0.5: 1 + 1 + 3 (2 x^2 - 1) = 0.5, each step exact in binary.
  const std::array<double, 3> c{1.0, 2.0, 3.0};
  EXPECT_EQ(chebyshev::clenshaw_column(c.data(), 1, 3, 0.5), 0.5);
  EXPECT_EQ(chebyshev::clenshaw_column(c.data(), 1, 2, 0.5), 2.0);
  EXPECT_EQ(chebyshev::clenshaw_column(c.data(), 1, 1, 0.5), 1.0);
  // T_8(cos t) = cos 8 t: eight Clenshaw steps on values of order one, each rounded
  // once, against the library cosine; 64 eps is a loose budget for both.
  std::array<double, 9> t8{};
  t8[8] = 1.0;
  const double angle = 0.3;
  EXPECT_NEAR(chebyshev::clenshaw_column(t8.data(), 1, 9, std::cos(angle)),
              std::cos(8.0 * angle), 64.0 * eps);
}

TEST(Chebyshev, ComplexColumnsMatchTheirParts) {
  // A complex column's parts are the real recursion of each part: the product with a
  // real x has a zero imaginary factor, so no rounding mixes the parts.
  const std::array<Complex, 4> c{Complex(1.0, -0.5), Complex(0.25, 2.0),
                                 Complex(-3.0, 0.125), Complex(0.75, -1.5)};
  std::array<double, 4> real{};
  std::array<double, 4> imag{};
  for (std::size_t i = 0; i < c.size(); ++i) {
    real[i] = c[i].real();
    imag[i] = c[i].imag();
  }
  for (const double x : {-1.0, -0.3, 0.0, 0.7, 1.0}) {
    const Complex value = chebyshev::clenshaw_column(c.data(), 1, 4, x);
    EXPECT_EQ(value.real(), chebyshev::clenshaw_column(real.data(), 1, 4, x));
    EXPECT_EQ(value.imag(), chebyshev::clenshaw_column(imag.data(), 1, 4, x));
  }
}

TEST(Chebyshev, IntervalsFollowSearchsortedRight) {
  const std::array<double, 4> edges{0.0, 1.0, 2.0, 4.0};
  EXPECT_EQ(chebyshev::interval_index(-1e-12, edges.data(), 4), 0U);
  EXPECT_EQ(chebyshev::interval_index(0.0, edges.data(), 4), 0U);
  EXPECT_EQ(chebyshev::interval_index(1.0, edges.data(), 4), 1U);
  EXPECT_EQ(chebyshev::interval_index(1.5, edges.data(), 4), 1U);
  EXPECT_EQ(chebyshev::interval_index(2.0, edges.data(), 4), 2U);
  EXPECT_EQ(chebyshev::interval_index(4.0, edges.data(), 4), 2U);
  EXPECT_EQ(chebyshev::interval_index(4.0 + 1e-12, edges.data(), 4), 2U);
  EXPECT_EQ(chebyshev::interval_index(std::nan(""), edges.data(), 4), 2U);
}

TEST(Chebyshev, PointsEvaluateEachIntervalAndRefuseOutside) {
  // Two intervals of [0, 2], each the linear function z itself: c = (mid, half).
  const std::array<double, 3> edges{0.0, 1.0, 2.0};
  const std::array<std::complex<double>, 4> c{std::complex<double>(0.5, -0.5), 0.5,
                                              std::complex<double>(1.5, -1.5), 0.5};
  const std::array<double, 4> z{0.0, 0.25, 1.0, 2.0};
  std::array<std::complex<double>, 4> out{};
  EXPECT_EQ(chebyshev::clenshaw_points(z.data(), 4, edges.data(), 3, 0.0, 2.0, c.data(),
                                       2, 1, out.data()),
            -1);
  for (std::size_t p = 0; p < z.size(); ++p) {
    EXPECT_EQ(out[p].real(), z[p]);
    EXPECT_EQ(out[p].imag(), -0.5 - std::floor(std::min(z[p], 1.5)));
  }
  const std::array<double, 3> outside{1.0, 2.5, 0.5};
  std::array<std::complex<double>, 3> partial{};
  EXPECT_EQ(chebyshev::clenshaw_points(outside.data(), 3, edges.data(), 3, 0.0, 2.0,
                                       c.data(), 2, 1, partial.data()),
            1);
  EXPECT_EQ(partial[0].real(), 1.0);
  EXPECT_EQ(partial[2].real(), 0.0); // not written
  const std::array<double, 1> missing{std::nan("")};
  std::array<std::complex<double>, 1> nan_out{};
  EXPECT_EQ(chebyshev::clenshaw_points(missing.data(), 1, edges.data(), 3, 0.0, 2.0,
                                       c.data(), 2, 1, nan_out.data()),
            -1);
  EXPECT_TRUE(std::isnan(nan_out[0].real()));
}

TEST(LagrangeBasis, AdamsWeightsOnEqualSteps) {
  // Adams-Bashforth 4 and Adams-Moulton 4, times 24. Each weight is a handful of
  // operations on small integers and quarters; 64 eps on values near 50 is generous.
  const std::array<double, 4> bashforth{0.0, -1.0, -2.0, -3.0};
  const std::array<double, 4> moulton{1.0, 0.0, -1.0, -2.0};
  std::array<double, 16> rows{};
  std::array<double, 4> weights{};
  rows.fill(7.0);
  lagrange::basis_rows(bashforth.data(), 4, 1.0, true, rows.data(), weights.data());
  const std::array<double, 4> expected_ab{55.0, -59.0, 37.0, -9.0};
  for (std::size_t j = 0; j < 4; ++j)
    EXPECT_NEAR(24.0 * weights[j], expected_ab[j], 64.0 * 60.0 * eps);
  for (const double value : rows)
    EXPECT_EQ(value, 0.0);
  lagrange::basis_rows(moulton.data(), 4, 1.0, true, rows.data(), weights.data());
  const std::array<double, 4> expected_am{9.0, 19.0, -5.0, 1.0};
  for (std::size_t j = 0; j < 4; ++j)
    EXPECT_NEAR(24.0 * weights[j], expected_am[j], 64.0 * 20.0 * eps);
}

TEST(LagrangeBasis, CoefficientsOfTheLinearBasis) {
  // On the nodes 1 and 0 the basis is theta and 1 - theta, exactly.
  const std::array<double, 2> nodes{1.0, 0.0};
  std::array<double, 4> rows{};
  std::array<double, 2> weights{-3.0, -3.0};
  lagrange::basis_rows(nodes.data(), 2, 1.0, false, rows.data(), weights.data());
  EXPECT_EQ(rows[0], 0.0);
  EXPECT_EQ(rows[1], 1.0);
  EXPECT_EQ(rows[2], 1.0);
  EXPECT_EQ(rows[3], -1.0);
  EXPECT_EQ(weights[0], -3.0); // not written
  // One node: the constant one, integrated to `upper`.
  const std::array<double, 1> single{-0.5};
  std::array<double, 1> constant{};
  std::array<double, 1> integral{};
  lagrange::basis_rows(single.data(), 1, 0.25, true, constant.data(), integral.data());
  EXPECT_EQ(integral[0], 0.25);
}

} // namespace
