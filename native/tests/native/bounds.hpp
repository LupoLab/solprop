// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Rounding bounds for the native tests (solprop SEP 25): two computations of the same
// discrete arithmetic in another order, contraction or thread layout are compared by
// a bound fixed from their operation counts before they run, not by their bits.
// The names mirror solprop's `tests/bounds.py`; `u` is the unit roundoff 2^-53 and
// epsilon `2 u`.

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace solprop_native::test {

inline constexpr double kUnitRoundoff = std::numeric_limits<double>::epsilon() / 2.0;

// Higham's `gamma_k = k u / (1 - k u)`: a sum of fixed inputs evaluated by any tree
// of depth `k` is within `gamma_k sum |x_i|` of the exact sum, a real dot product of
// length `n` within `gamma_n sum |a_i b_i|`, contracted or not.
inline double higham_gamma(const std::size_t depth) {
  const double scaled = static_cast<double>(depth) * kUnitRoundoff;
  if (scaled >= 1.0)
    throw std::invalid_argument("gamma's depth is beyond its range");
  return scaled / (1.0 - scaled);
}

// The equivalence ladder's `k eps / (1 - k eps)`, twice Higham's gamma.
inline double gamma(const std::size_t operations) {
  return higham_gamma(2 * operations);
}

} // namespace solprop_native::test
