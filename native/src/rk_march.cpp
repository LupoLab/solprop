// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/rk_march.hpp"

#include <cmath>

// As in adams_march.cpp (solprop #135): the optimize attribute this pragma gives every
// function below keeps GCC from inlining it into a contracting caller under link-time
// optimisation. Clang carries the file's setting on each instruction.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace solprop_native::rk {

std::vector<double> positions(const double z, const double dz,
                              const std::vector<double>& nodes) {
  std::vector<double> out;
  out.reserve(nodes.size() + 1);
  for (const double node : nodes) {
    const double offset = node * dz;
    out.push_back(z + offset);
  }
  out.push_back(z + dz);
  return out;
}

double score(const double* partials, const std::size_t slots, const double rtol,
             const double atol) {
  double sums[3] = {0.0, 0.0, 0.0};
  for (std::size_t row = 0; row < 3; ++row)
    for (std::size_t slot = 0; slot < slots; ++slot)
      sums[row] += partials[row * slots + slot];
  const double old = std::sqrt(sums[0]);
  const double advanced = std::sqrt(sums[1]);
  const double residual = std::sqrt(sums[2]);
  // `rtol * max(norm_y, norm_y_new) + atol`, Python's `max`.
  const double larger = advanced > old ? advanced : old;
  const double scaled = rtol * larger;
  return residual / (scaled + atol);
}

double strict_probe(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}

} // namespace solprop_native::rk
