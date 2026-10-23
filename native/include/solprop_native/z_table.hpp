// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/chebyshev.hpp"
#include "solprop_native/core.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * The general z table, resident in the execution space (solprop SEP 29 section 6.6,
 * G1).
 *
 * solprop falls back to a piecewise Chebyshev polynomial of the state's own size for
 * Phi (`LinearOperator.phi_coefficients`) or the response prefactor
 * (`CompiledResponses.prefactor`) when neither is a weighted sum of fixed arrays. The
 * coefficients are uploaded once, laid out `(n_intervals, n_terms, n_values)` with
 * the state's values `(column, bin)` fastest, and a native call evaluates the table at
 * a trial's positions into the term slots the stage kernels already read
 * (`phi_terms`, `prefactor_terms`): the adapter then weighs the slots by +-1, so that
 * `Phi(z1) - Phi(z0)` is one rounded subtraction of two evaluated values, as solprop's
 * reference forms it. Each value is solprop's Clenshaw recursion
 * (`chebyshev::clenshaw_column`) under strict arithmetic, operation for operation, at
 * the interval and unit position solprop's `PiecewiseChebyshev` chooses; a rate table
 * (`LinearOperator.rate`, the derivative of Phi's) may read an interior breakpoint
 * from the interval ending there, as solprop's `phi_rate(..., before=True)` does.
 */
namespace solprop_native::z_table {

/** The most positions one evaluation takes. */
inline constexpr std::size_t max_points = 16;

/** Where each position of one evaluation reads its table, and the slot it fills. */
struct Points {
  std::size_t count{};
  std::size_t slot[max_points]{};
  std::size_t interval[max_points]{};
  double x[max_points]{};
};

/** One table's coefficients in the execution space and its breakpoints on the host. */
struct Table {
  Kokkos::View<Complex***, Layout, MemorySpace> coefficients;
  std::vector<double> edges;
  /** The span widened by solprop's edge slack: a position outside is refused. */
  double lower{};
  double upper{};

  [[nodiscard]] bool present() const noexcept { return coefficients.extent(0) != 0; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return coefficients.size() * sizeof(Complex) + edges.size() * sizeof(double);
  }
};

/**
 * Locate `positions` in `table` for the slots `slots`: each position's interval
 * (solprop's `interval_containing`, or with `before` the interval ending at an
 * interior breakpoint it sits on) and its unit position. Refuses a position outside
 * the widened span, or more than `max_points` of them.
 */
inline Points locate(const Table& table, const std::vector<std::size_t>& slots,
                     const std::vector<double>& positions, const bool before) {
  if (slots.size() != positions.size())
    throw std::invalid_argument("z table: one slot per position");
  if (positions.size() > max_points)
    throw std::invalid_argument("z table: at most " + std::to_string(max_points) +
                                " positions per evaluation, got " +
                                std::to_string(positions.size()));
  if (table.edges.size() < 2)
    throw std::logic_error("z table: no table has been uploaded");
  Points points;
  points.count = positions.size();
  for (std::size_t p = 0; p < positions.size(); ++p) {
    const double z = positions[p];
    if (!(z >= table.lower && z <= table.upper))
      throw std::out_of_range("z = " + std::to_string(z) +
                              " m is outside the z table's span [" +
                              std::to_string(table.edges.front()) + ", " +
                              std::to_string(table.edges.back()) + "] m");
    std::size_t k =
        chebyshev::interval_index(z, table.edges.data(), table.edges.size());
    if (before && k > 0 && z <= table.edges[k])
      --k;
    points.slot[p] = slots[p];
    points.interval[p] = k;
    points.x[p] = chebyshev::unit_position(z, table.edges[k], table.edges[k + 1]);
  }
  return points;
}

/**
 * The values a host work item of `evaluate` takes at a time; one on a device, where
 * each thread takes one value. Sixteen holds a block's four running real arrays in
 * registers on the x86-64 and Arm baselines and keeps the block loop vectorisable.
 */
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
inline constexpr std::size_t block_values = 1;
#else
inline constexpr std::size_t block_values = 16;
#endif

/**
 * Write `count` consecutive values of one interval's series at `x`, at most
 * `block_values`: `c[i * stride + b]` is value `b`'s coefficient of degree `i`, and
 * value `b` goes to `out[b]`.
 *
 * Each value is `chebyshev::clenshaw_column`'s recursion operation for operation --
 * the same products with `x` (and with the zero imaginary part of `x + 0i`), the same
 * sums, in the same order, each rounded once -- so every value equals the per-value
 * form exactly. The recursion is spelled here on real and imaginary arrays across the
 * block, with the degree outside and the block inside: the block's values are
 * independent, so that loop vectorises and overlaps the recursion's dependency chain,
 * which a per-value recursion cannot. The whole block is one strict function
 * (fp.hpp): one call per block where GCC does not inline a strict function into a
 * contracting kernel, rather than one call per operation of a per-value recursion of
 * strict helpers.
 */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION void clenshaw_block(const Complex* c, const std::size_t stride,
                                           const std::size_t n_terms, const double x,
                                           const std::size_t count, Complex* out) {
  SOLPROP_NATIVE_STRICT_BLOCK
  if (n_terms == 1) {
    // `c_0 + (0 x + 0i)`.
    const double zero = 0.0 * x;
    for (std::size_t b = 0; b < count; ++b)
      out[b] = Complex(c[b].real() + zero, c[b].imag() + 0.0);
    return;
  }
  if (n_terms == 2) {
    // `c_0 + c_1 (x + 0i)`, the complex product with every term rounded.
    for (std::size_t b = 0; b < count; ++b) {
      const double real = c[stride + b].real() * x - c[stride + b].imag() * 0.0;
      const double imag = c[stride + b].real() * 0.0 + c[stride + b].imag() * x;
      out[b] = Complex(c[b].real() + real, c[b].imag() + imag);
    }
    return;
  }
  double c0_real[block_values];
  double c0_imag[block_values];
  double c1_real[block_values];
  double c1_imag[block_values];
  const double x2 = 2.0 * x;
  const Complex* second = c + (n_terms - 2) * stride;
  const Complex* last = c + (n_terms - 1) * stride;
  for (std::size_t b = 0; b < count; ++b) {
    c0_real[b] = second[b].real();
    c0_imag[b] = second[b].imag();
    c1_real[b] = last[b].real();
    c1_imag[b] = last[b].imag();
  }
  for (std::size_t i = 3; i <= n_terms; ++i) {
    const Complex* row = c + (n_terms - i) * stride;
    for (std::size_t b = 0; b < count; ++b) {
      // `held = c0; c0 = c_{n-i} - c1; c1 = held + c1 (2x + 0i)`.
      const double held_real = c0_real[b];
      const double held_imag = c0_imag[b];
      c0_real[b] = row[b].real() - c1_real[b];
      c0_imag[b] = row[b].imag() - c1_imag[b];
      const double real = c1_real[b] * x2 - c1_imag[b] * 0.0;
      const double imag = c1_real[b] * 0.0 + c1_imag[b] * x2;
      c1_real[b] = held_real + real;
      c1_imag[b] = held_imag + imag;
    }
  }
  for (std::size_t b = 0; b < count; ++b) {
    // `c0 + c1 (x + 0i)`.
    const double real = c1_real[b] * x - c1_imag[b] * 0.0;
    const double imag = c1_real[b] * 0.0 + c1_imag[b] * x;
    out[b] = Complex(c0_real[b] + real, c0_imag[b] + imag);
  }
}

/**
 * Write the table's value at each located position into `out[slot * n_values + e]`,
 * for every value `e`: every value solprop's strict Clenshaw recursion at the
 * position's interval and unit position. A work item takes `block_values`
 * consecutive values (the last one the remainder) at every position.
 * `out` holds at least `(max slot + 1) * n_values` values in the table's memory space.
 */
template <class Space>
void evaluate(const char* label, const Space& space, const Table& table,
              const Points& points, Complex* out) {
  const auto coefficients = table.coefficients;
  const std::size_t n_terms = coefficients.extent(1);
  const std::size_t n_values = coefficients.extent(2);
  const std::size_t blocks = (n_values + block_values - 1) / block_values;
  const Complex* data = coefficients.data();
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<Space>(space, 0, blocks),
      KOKKOS_LAMBDA(const std::size_t block) {
        const std::size_t first = block * block_values;
        const std::size_t count =
            n_values - first < block_values ? n_values - first : block_values;
        for (std::size_t p = 0; p < points.count; ++p)
          clenshaw_block(data + points.interval[p] * n_terms * n_values + first,
                         n_values, n_terms, points.x[p], count,
                         out + points.slot[p] * n_values + first);
      });
}

} // namespace solprop_native::z_table
