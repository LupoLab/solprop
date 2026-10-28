// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/fp.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>

/*
 * The Lagrange basis of a variable-step Adams formula, as solprop forms it.
 *
 * solprop's `solprop.stepper.multistep._basis_rows`: on distinct normalised nodes
 * `theta_m`, each basis numerator `prod_{m != j} (theta - theta_m)` is built from its
 * own roots, ascending, as `numpy.polynomial.polyfromroots` builds it, then either
 * divided by its denominator `prod_{m != j} (theta_j - theta_m)` (the monomial
 * coefficients) or integrated from 0 to `upper` by Horner's rule and divided (the
 * integrated weights). These are the multistep increments and defects, small
 * differences of large products, so every operation is rounded on its own and in
 * solprop's order (fp.hpp, solprop SEP 25 section 4.2).
 */
namespace solprop_native::lagrange {

/** The most nodes a basis takes: the scratch is on the stack. */
inline constexpr std::size_t max_nodes = 32;

/**
 * Write the Lagrange basis on `n` nodes: its coefficients, or its integrals.
 *
 * `rows` is `(n, n)` and C-contiguous: row `j` holds the ascending monomial
 * coefficients of the `j`-th basis polynomial, or zeros when integrating. `weights` is
 * `(n,)`: the integrals from 0 to `upper` when integrating, and not written otherwise.
 * `n` is at most `max_nodes`; the nodes are not checked for being distinct and finite.
 */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION void basis_rows(const double* points, const std::size_t n,
                                       const double upper, const bool integrate,
                                       double* rows, double* weights) {
  SOLPROP_NATIVE_STRICT_BLOCK
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

/**
 * Write an exponential Adams trial's coefficient tables in one call.
 *
 * solprop's `solprop.backends.multistep.trial_tables` over `basis_rows`, every value
 * formed by the same operations in the same order, so the tables are those of its
 * composition over this backend's basis, bit for bit. On the history's normalised
 * nodes `nodes`, newest first, for a step of order `order` scoring the orders
 * `orders` (`n_orders` of them, the step's own first) and `width = 1 + max(orders)`:
 *
 * - `predictor` `(order, order)`: the Lagrange coefficients on the `order` newest
 *   nodes;
 * - `corrector` `(width, width)`: those on `(1, nodes[:order])`, zero past them;
 * - `factored` `(n_orders, 2 width)`: each order `k`'s defect as the divided-difference
 *   coefficients on `(1, nodes[:k])`, then its node polynomial's ascending monomial
 *   coefficients, each expanded one linear factor at a time, newest node first;
 *
 * and, when `compensated`, the transported compensation's:
 *
 * - `spans` `(width - 1,)`: `1 - theta_t`;
 * - `carried` `(order,)` and `endpoint` `(order + 1,)`: the integrated weights on the
 *   `order` newest nodes and on `(1, nodes[:order])`;
 * - `defects` `(n_orders, width)`: each order's corrector less its predictor.
 *
 * The compensation's arrays are not written otherwise. `width` is at most
 * `max_nodes`; nothing is checked.
 */
SOLPROP_NATIVE_STRICT_FUNCTION
inline void trial_tables(const double* nodes, const std::size_t order,
                         const std::size_t* orders, const std::size_t n_orders,
                         const std::size_t width, const bool compensated,
                         double* predictor, double* corrector, double* factored,
                         double* spans, double* carried, double* endpoint,
                         double* defects) {
  SOLPROP_NATIVE_STRICT_BLOCK
  double through[max_nodes];
  double scratch[max_nodes * max_nodes];
  double weights[max_nodes];
  through[0] = 1.0;
  for (std::size_t t = 0; t + 1 < width; ++t)
    through[t + 1] = nodes[t];
  basis_rows(nodes, order, 1.0, false, predictor, weights);
  basis_rows(through, order + 1, 1.0, false, scratch, weights);
  for (std::size_t row = 0; row < width; ++row)
    for (std::size_t column = 0; column < width; ++column)
      corrector[row * width + column] =
          row <= order && column <= order ? scratch[row * (order + 1) + column] : 0.0;
  for (std::size_t index = 0; index < n_orders; ++index) {
    const std::size_t scored = orders[index];
    double* const row = factored + index * 2 * width;
    for (std::size_t t = 0; t < 2 * width; ++t)
      row[t] = 0.0;
    for (std::size_t t = 0; t <= scored; ++t) {
      double denominator = 1.0;
      for (std::size_t s = 0; s <= scored; ++s) {
        if (s != t)
          denominator = denominator * (through[t] - through[s]);
      }
      row[t] = 1.0 / denominator;
    }
    double coefficients[max_nodes];
    double product[max_nodes];
    std::size_t length = 1;
    coefficients[0] = 1.0;
    for (std::size_t root = 0; root < scored; ++root) {
      product[0] = 0.0;
      for (std::size_t power = 0; power < length; ++power)
        product[power + 1] = coefficients[power];
      for (std::size_t power = 0; power < length; ++power) {
        const double term = nodes[root] * coefficients[power];
        product[power] = product[power] - term;
      }
      ++length;
      for (std::size_t power = 0; power < length; ++power)
        coefficients[power] = product[power];
    }
    for (std::size_t power = 0; power < length; ++power)
      row[width + power] = coefficients[power];
  }
  if (!compensated)
    return;
  for (std::size_t t = 0; t + 1 < width; ++t)
    spans[t] = 1.0 - nodes[t];
  basis_rows(nodes, order, 1.0, true, scratch, carried);
  basis_rows(through, order + 1, 1.0, true, scratch, endpoint);
  for (std::size_t index = 0; index < n_orders; ++index) {
    const std::size_t scored = orders[index];
    double* const row = defects + index * width;
    basis_rows(through, scored + 1, 1.0, true, scratch, row);
    for (std::size_t t = scored + 1; t < width; ++t)
      row[t] = 0.0;
    basis_rows(nodes, scored, 1.0, true, scratch, weights);
    for (std::size_t t = 0; t < scored; ++t)
      row[t + 1] = row[t + 1] - weights[t];
  }
}

} // namespace solprop_native::lagrange
