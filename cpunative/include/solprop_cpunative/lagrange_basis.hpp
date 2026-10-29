// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_cpunative/fp.hpp"

#include <cmath>
#include <complex>

#include <cstddef>

/*
 * The Lagrange basis of a variable-step Adams formula, as solprop forms it.
 *
 * solprop's `solprop.stepper.multistep._basis_rows`: on distinct normalised
 * nodes `theta_m`, each basis numerator `prod_{m != j} (theta - theta_m)` is
 * built from its own roots, ascending, as `numpy.polynomial.polyfromroots`
 * builds it, then either divided by its denominator `prod_{m != j} (theta_j -
 * theta_m)` (the monomial coefficients) or integrated from 0 to `upper` by
 * Horner's rule and divided (the integrated weights). These are the multistep
 * increments and defects, small differences of large products, so every
 * operation is rounded on its own and in solprop's order (fp.hpp, solprop SEP
 * 25 section 4.2).
 */
namespace solprop_cpunative::lagrange {

/** The most nodes a basis takes: the scratch is on the stack. */
inline constexpr std::size_t max_nodes = 32;

/**
 * Write the Lagrange basis on `n` nodes: its coefficients, or its integrals.
 *
 * `rows` is `(n, n)` and C-contiguous: row `j` holds the ascending monomial
 * coefficients of the `j`-th basis polynomial, or zeros when integrating.
 * `weights` is
 * `(n,)`: the integrals from 0 to `upper` when integrating, and not written
 * otherwise. `n` is at most `max_nodes`; the nodes are not checked for being
 * distinct and finite.
 */
SOLPROP_CPUNATIVE_STRICT_FUNCTION
inline void basis_rows(const double* points, const std::size_t n, const double upper,
                       const bool integrate, double* rows, double* weights) {
  SOLPROP_CPUNATIVE_STRICT_BLOCK
  double coefficients[max_nodes];
  double product[max_nodes];
  if (integrate) {
    for (std::size_t index = 0; index < n * n; ++index)
      rows[index] = 0.0;
  }
  for (std::size_t index = 0; index < n; ++index) {
    const double node = points[index];
    std::size_t length = 1;
    coefficients[0] = 1.0;
    double denominator = 1.0;
    for (std::size_t other = 0; other < n; ++other) {
      if (other == index)
        continue;
      const double root = points[other];
      product[0] = 0.0;
      for (std::size_t power = 0; power < length; ++power)
        product[power + 1] = coefficients[power];
      for (std::size_t power = 0; power < length; ++power) {
        const double term = root * coefficients[power];
        product[power] = product[power] - term;
      }
      ++length;
      for (std::size_t power = 0; power < length; ++power)
        coefficients[power] = product[power];
      denominator = denominator * (node - root);
    }
    if (integrate) {
      double integral = 0.0;
      for (std::size_t power = length; power-- > 0;) {
        const double term = coefficients[power] / static_cast<double>(power + 1);
        integral = (integral + term) * upper;
      }
      weights[index] = integral / denominator;
    } else {
      for (std::size_t power = 0; power < length; ++power)
        rows[index * n + power] = coefficients[power] / denominator;
    }
  }
}

} // namespace solprop_cpunative::lagrange
