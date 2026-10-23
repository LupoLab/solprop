// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/chebyshev.hpp"
#include "solprop_native/core.hpp"
#include "solprop_native/z_table.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
namespace native = solprop_native;
namespace chebyshev = solprop_native::chebyshev;
namespace z_table = solprop_native::z_table;
using native::Complex;

constexpr double eps = std::numeric_limits<double>::epsilon();

// A table of `n_values` columns on the breakpoints `edges`: on interval `k`, value `e`
// is `a_ke + b_ke T1 + c_ke T2` with small integers, so a value is exact in binary at
// the unit positions the tests choose.
z_table::Table quadratic_table(const std::vector<double>& edges,
                               const std::size_t n_values) {
  z_table::Table table;
  const std::size_t intervals = edges.size() - 1;
  table.coefficients = Kokkos::View<Complex***, native::Layout, native::MemorySpace>(
      "test.z_table", intervals, 3, n_values);
  auto host = Kokkos::create_mirror_view(table.coefficients);
  for (std::size_t k = 0; k < intervals; ++k)
    for (std::size_t e = 0; e < n_values; ++e) {
      const auto shift = static_cast<double>(k + e);
      host(k, 0, e) = Complex(1.0 + shift, -shift);
      host(k, 1, e) = Complex(2.0, 0.5 * shift);
      host(k, 2, e) = Complex(-1.0, 0.25);
    }
  Kokkos::deep_copy(table.coefficients, host);
  table.edges = edges;
  table.lower = edges.front();
  table.upper = edges.back();
  return table;
}

TEST(ZTable, LocateTakesSolpropsIntervalAndEdgeSide) {
  auto table = quadratic_table({0.0, 1.0, 3.0}, 1);
  const auto points = z_table::locate(table, {4, 2, 7, 0}, {0.0, 1.0, 2.0, 3.0}, false);
  ASSERT_EQ(points.count, 4U);
  // An interior breakpoint belongs to the interval starting there, the span's end to
  // the last interval; x is (2 z - lo - hi) / (hi - lo).
  EXPECT_EQ(points.interval[0], 0U);
  EXPECT_EQ(points.x[0], -1.0);
  EXPECT_EQ(points.interval[1], 1U);
  EXPECT_EQ(points.x[1], -1.0);
  EXPECT_EQ(points.interval[2], 1U);
  EXPECT_EQ(points.x[2], 0.0);
  EXPECT_EQ(points.interval[3], 1U);
  EXPECT_EQ(points.x[3], 1.0);
  EXPECT_EQ(points.slot[0], 4U);
  EXPECT_EQ(points.slot[3], 0U);
  // A rate read `before` an interior breakpoint takes the interval ending there.
  const auto before = z_table::locate(table, {0, 1, 2}, {0.0, 1.0, 2.0}, true);
  EXPECT_EQ(before.interval[0], 0U);
  EXPECT_EQ(before.interval[1], 0U);
  EXPECT_EQ(before.x[1], 1.0);
  EXPECT_EQ(before.interval[2], 1U);
}

TEST(ZTable, LocateRefusesWhatItCannotEvaluate) {
  auto table = quadratic_table({0.0, 1.0}, 1);
  table.lower = -1e-10;
  table.upper = 1.0 + 1e-10;
  EXPECT_NO_THROW(static_cast<void>(z_table::locate(table, {0}, {-1e-11}, false)));
  EXPECT_THROW(static_cast<void>(z_table::locate(table, {0}, {1.1}, false)),
               std::out_of_range);
  EXPECT_THROW(static_cast<void>(z_table::locate(table, {0}, {std::nan("")}, false)),
               std::out_of_range);
  EXPECT_THROW(static_cast<void>(z_table::locate(table, {0, 1}, {0.5}, false)),
               std::invalid_argument);
  const std::vector<std::size_t> slots(z_table::max_points + 1, 0);
  const std::vector<double> positions(z_table::max_points + 1, 0.5);
  EXPECT_THROW(static_cast<void>(z_table::locate(table, slots, positions, false)),
               std::invalid_argument);
  const z_table::Table empty;
  EXPECT_THROW(static_cast<void>(z_table::locate(empty, {0}, {0.5}, false)),
               std::logic_error);
}

TEST(ZTable, EvaluateWritesEachPositionIntoItsSlot) {
  constexpr std::size_t n_values = 37;
  const auto table = quadratic_table({0.0, 1.0, 3.0}, n_values);
  constexpr std::size_t slots = 4;
  Kokkos::View<Complex**, native::Layout, native::MemorySpace> out("test.out", slots,
                                                                   n_values);
  Kokkos::deep_copy(out, Complex(7.0, 7.0));
  // Slots 3 and 1 at x = 0.5 on interval 0 and x = -0.5 on interval 1; slot 0 and 2
  // are not named and keep their values.
  const std::vector<double> positions{0.75, 1.5};
  const auto points = z_table::locate(table, {3, 1}, positions, false);
  z_table::evaluate("test.evaluate", native::ExecutionSpace{}, table, points,
                    out.data());
  Kokkos::fence();
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  const auto coefficients =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, table.coefficients);
  for (std::size_t e = 0; e < n_values; ++e) {
    EXPECT_EQ(host(0, e), Complex(7.0, 7.0));
    EXPECT_EQ(host(2, e), Complex(7.0, 7.0));
    for (const auto [slot, k, x] : {std::array<double, 3>{3.0, 0.0, 0.5},
                                    std::array<double, 3>{1.0, 1.0, -0.5}}) {
      const auto interval = static_cast<std::size_t>(k);
      // The kernel is the header's recursion per value: the same operations.
      const Complex* column = &coefficients(interval, 0, e);
      EXPECT_EQ(host(static_cast<std::size_t>(slot), e),
                chebyshev::clenshaw_column(column, n_values, 3, x));
      // And it is the quadratic: a + b x + c (2 x^2 - 1), exact at these x.
      const Complex expected = coefficients(interval, 0, e) +
                               coefficients(interval, 1, e) * x +
                               coefficients(interval, 2, e) * (2.0 * x * x - 1.0);
      const Complex got = host(static_cast<std::size_t>(slot), e);
      EXPECT_NEAR(got.real(), expected.real(), 8.0 * eps * std::abs(expected.real()));
      EXPECT_NEAR(got.imag(), expected.imag(), 8.0 * eps * std::abs(expected.imag()));
    }
  }
}

TEST(ZTable, AConstantTableIsItsCoefficient) {
  // One coefficient: the value is c0 + 0 x, the coefficient itself.
  z_table::Table table;
  table.coefficients = Kokkos::View<Complex***, native::Layout, native::MemorySpace>(
      "test.constant", 1, 1, 2);
  auto host = Kokkos::create_mirror_view(table.coefficients);
  host(0, 0, 0) = Complex(-2.5, 0.125);
  host(0, 0, 1) = Complex(3.0, -0.0);
  Kokkos::deep_copy(table.coefficients, host);
  table.edges = {0.0, 2.0};
  table.upper = 2.0;
  Kokkos::View<Complex**, native::Layout, native::MemorySpace> out("test.out", 1, 2);
  z_table::evaluate("test.constant", native::ExecutionSpace{}, table,
                    z_table::locate(table, {0}, {1.3}, false), out.data());
  Kokkos::fence();
  const auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  EXPECT_EQ(values(0, 0), Complex(-2.5, 0.125));
  EXPECT_EQ(values(0, 1).real(), 3.0);
}

TEST(ZTable, BlockedEvaluationIsThePerValueRecursionExactly) {
  // Each degree's branch (1, 2 and more terms), on full blocks and a tail: a value
  // count that is not a multiple of the block, through the kernel, against
  // `clenshaw_column` per value. The same operations in the same order, so the
  // values are equal, signed zeros and all.
  constexpr std::size_t n_values = 3 * z_table::block_values + 5;
  for (const std::size_t terms : {1U, 2U, 3U, 9U, 33U}) {
    z_table::Table table;
    table.coefficients = Kokkos::View<Complex***, native::Layout, native::MemorySpace>(
        "test.blocked", 2, terms, n_values);
    auto host = Kokkos::create_mirror_view(table.coefficients);
    std::uint64_t state = 0x9E3779B97F4A7C15ULL + terms;
    const auto draw = [&state] {
      // A 53-bit uniform in [-1, 1), deterministic.
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      return static_cast<double>(state >> 11U) * 0x1.0p-52 - 1.0;
    };
    for (std::size_t k = 0; k < 2; ++k)
      for (std::size_t i = 0; i < terms; ++i)
        for (std::size_t e = 0; e < n_values; ++e)
          host(k, i, e) = Complex(draw(), e % 7 == 0 ? -0.0 : draw());
    Kokkos::deep_copy(table.coefficients, host);
    table.edges = {0.0, 0.37, 1.0};
    table.upper = 1.0;
    const std::vector<double> positions{0.0, 0.2, 0.37, 0.81, 1.0};
    const std::vector<std::size_t> slots{4, 0, 2, 1, 3};
    const auto points = z_table::locate(table, slots, positions, false);
    Kokkos::View<Complex**, native::Layout, native::MemorySpace> out("test.out", 5,
                                                                     n_values);
    z_table::evaluate("test.blocked", native::ExecutionSpace{}, table, points,
                      out.data());
    Kokkos::fence();
    const auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
    for (std::size_t p = 0; p < positions.size(); ++p)
      for (std::size_t e = 0; e < n_values; ++e) {
        const Complex want = chebyshev::clenshaw_column(&host(points.interval[p], 0, e),
                                                        n_values, terms, points.x[p]);
        const Complex got = values(points.slot[p], e);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(got.real()),
                  std::bit_cast<std::uint64_t>(want.real()))
            << terms << " terms, position " << p << ", value " << e;
        EXPECT_EQ(std::bit_cast<std::uint64_t>(got.imag()),
                  std::bit_cast<std::uint64_t>(want.imag()))
            << terms << " terms, position " << p << ", value " << e;
      }
  }
}

} // namespace
