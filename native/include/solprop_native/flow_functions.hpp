// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/core.hpp"

#include <cstddef>

// The matrix functions of a coupled block's bin off its flow's eigenbasis route, for
// the exponential Adams formulation (solprop SEP 47 phase 4, section 6.3's coupled-flow
// row).
//
// solprop's reference (`backends/multistep.py`, `coupled_moments`) reads the flow and
// the phi-functions of `X = shift I + h T`, `T` the bin's upper-triangular Schur
// factor, off the first block row of the exponential of the augmented matrix
//
//     A = [[X, I, 0, ..., 0], [0, 0, I, ..., 0], ..., [0, ..., 0, I], [0, ..., 0]],
//
// `count + 1` blocks of `n` channels a side, whose first block row is `exp(X)`,
// `phi_1(X)`, ..., `phi_count(X)` (Higham, *Functions of Matrices*, section 10.7.4;
// Al-Mohy and Higham 2011): the reference by SciPy's Pade `expm`, this engine by the
// scaling, sixteen-term Taylor series and squaring of solprop's reference flow action
// (`numpy_ref/backend.py`, `_triangular_exponential`), the native
// `squared_series_action`.
//
// Every block row of `A` but the first is a scaled shift, so every power of `A / 2^s`,
// and so every partial sum and square below, keeps the first block row `R_0, ...,
// R_count` and, on every later block row `i`, the scalar `p_(j - i)` times the identity
// at block column `j`. The passes form those alone: each sum is the full augmented
// product's over the entries that are not exact zeros, in its inner order, so the
// result is the full algorithm's but for the sign of a zero, and the full algorithm's
// rounding bound, with `n (count + 1)` channels, holds (`tests/flow_parity.py`,
// `augmented_formation_bound`). The triangular factor keeps every product upper
// triangular, whose lower entries are exact zeros and are neither formed nor read: the
// factor's own lower entries, zero in a Schur form, are not read either.
namespace solprop_native::flow_functions {

/** solprop's reference `_TAYLOR_ORDER`: the series degree. */
inline constexpr int taylor_order = 16;
/** solprop's reference `_TAYLOR_SCALE`: the row-sum norm the scaled matrix stays below.
 */
inline constexpr double taylor_scale = 0.5;
/** solprop's reference `_MAX_FLOW_SQUARINGS`: the most squarings a bin may need. */
inline constexpr int max_squarings = 30;

/**
 * The squarings the series of the augmented matrix of `shift I + length T` with `count`
 * phi-blocks takes: from its row-sum norm, `max_r sum_c |X_rc|` plus one for the
 * identity block beside it (when `count > 0`), as solprop's reference counts them for
 * the flow's action. A norm that is not finite, or would need more than
 * `max_squarings`, returns `max_squarings + 1`, which the caller refuses.
 */
template <class Triangular>
KOKKOS_INLINE_FUNCTION int
augmented_squarings(const Complex shift, const double length,
                    const Triangular& triangular, const std::size_t fallback,
                    const std::size_t modes, const std::size_t count) {
  double largest = count > 0 ? 1.0 : 0.0;
  for (std::size_t row = 0; row < modes; ++row) {
    double total = count > 0 ? 1.0 : 0.0;
    for (std::size_t column = row; column < modes; ++column) {
      const Complex entry = row == column
                                ? shift + length * triangular(fallback, row, column)
                                : length * triangular(fallback, row, column);
      total += Kokkos::abs(entry);
    }
    // A NaN is kept, and refused below.
    if (!(total <= largest))
      largest = total;
  }
  // 2^30 times the scale: past it the refusal, and the cast below stays in range.
  if (!(largest <= taylor_scale * 0x1p30))
    return max_squarings + 1;
  if (largest <= taylor_scale)
    return 0;
  return static_cast<int>(Kokkos::ceil(Kokkos::log2(largest / taylor_scale)));
}

/** The blocks `augmented_row` holds per bin for `count` phi-blocks: the first block
    row, its scratch and the scaled block `X / 2^s`. */
KOKKOS_INLINE_FUNCTION constexpr std::size_t augmented_blocks(const std::size_t count) {
  return 2 * (count + 1) + 1;
}

/**
 * The first block row of `exp(A)`, `A` the augmented matrix of `X = shift I + length T`
 * with `count` phi-blocks: `rows(f, j, :, :)` holds `phi_j(X)` for `j <= count`
 * (`phi_0` the exponential), upper triangular, on return; the next `count + 2` blocks
 * of `rows(f, ...)` are scratch (`augmented_blocks`). `T` is `triangular(fallback, :,
 * :)`, `f` the bin's row of `rows`, and `squarings` the count `augmented_squarings`
 * returned (at most `max_squarings`). Sequential and in fixed order, so the result does
 * not depend on the thread count.
 */
template <class Triangular, class Rows>
KOKKOS_INLINE_FUNCTION void
augmented_row(const Complex shift, const double length, const Triangular& triangular,
              const std::size_t fallback, const std::size_t modes,
              const std::size_t count, const int squarings, const Rows& rows,
              const std::size_t f) {
  double divisor = 1.0;
  for (int index = 0; index < squarings; ++index)
    divisor *= 2.0;
  // The scaled identity block's entry, exact: a power of two.
  const double inverse = 1.0 / divisor;
  const auto scratch = count + 1;
  const auto scaled = 2 * (count + 1);
  for (std::size_t row = 0; row < modes; ++row)
    for (std::size_t column = 0; column < modes; ++column) {
      const Complex entry = row == column
                                ? shift + length * triangular(fallback, row, column)
                                : length * triangular(fallback, row, column);
      rows(f, scaled, row, column) = column >= row ? entry / divisor : Complex{};
    }
  // The scalars p_d of every later block row, d < count, from the identity's.
  double p[adams_max_samples + 1]{};
  p[0] = 1.0;
  for (std::size_t j = 0; j <= count; ++j)
    for (std::size_t row = 0; row < modes; ++row)
      for (std::size_t column = 0; column < modes; ++column)
        rows(f, j, row, column) =
            j == 0 && row == column ? Complex(1.0, 0.0) : Complex{};
  // Horner's steps, P <- I + (A / 2^s) P / term: the first block row reads the scaled
  // block and the next block row's scalars, a later block row the scalars alone. Each
  // new block is formed from the old ones into the scratch, then copied.
  for (int term = taylor_order; term > 0; --term) {
    const double by = static_cast<double>(term);
    for (std::size_t j = 0; j <= count; ++j) {
      const double previous = j >= 1 ? p[j - 1] : 0.0;
      for (std::size_t row = 0; row < modes; ++row)
        for (std::size_t column = row; column < modes; ++column) {
          Complex total{};
          for (std::size_t inner = row; inner <= column; ++inner)
            total += rows(f, scaled, row, inner) * rows(f, j, inner, column);
          if (j >= 1 && row == column)
            total += Complex(inverse * previous, 0.0);
          const Complex diagonal =
              j == 0 && row == column ? Complex(1.0, 0.0) : Complex{};
          rows(f, scratch + j, row, column) = diagonal + total / by;
        }
    }
    for (std::size_t j = 0; j <= count; ++j)
      for (std::size_t row = 0; row < modes; ++row)
        for (std::size_t column = row; column < modes; ++column)
          rows(f, j, row, column) = rows(f, scratch + j, row, column);
    for (std::size_t d = count; d-- > 1;)
      p[d] = (inverse * p[d - 1]) / by;
  }
  // The squarings, P <- P^2: block j of the first row is R_0 R_j + sum_m R_m p_(j - m),
  // the later rows' scalars the Cauchy product of their own.
  for (int index = 0; index < squarings; ++index) {
    for (std::size_t j = 0; j <= count; ++j)
      for (std::size_t row = 0; row < modes; ++row)
        for (std::size_t column = row; column < modes; ++column) {
          Complex total{};
          for (std::size_t inner = row; inner <= column; ++inner)
            total += rows(f, 0, row, inner) * rows(f, j, inner, column);
          for (std::size_t m = 1; m <= j; ++m)
            total += rows(f, m, row, column) * Complex(p[j - m], 0.0);
          rows(f, scratch + j, row, column) = total;
        }
    for (std::size_t j = 0; j <= count; ++j)
      for (std::size_t row = 0; row < modes; ++row)
        for (std::size_t column = row; column < modes; ++column)
          rows(f, j, row, column) = rows(f, scratch + j, row, column);
    for (std::size_t d = count; d-- > 1;) {
      double total = 0.0;
      for (std::size_t a = 0; a <= d; ++a)
        total += p[a] * p[d - a];
      p[d] = total;
    }
  }
}

} // namespace solprop_native::flow_functions
