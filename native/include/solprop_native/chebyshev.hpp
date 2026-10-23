// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/fp.hpp"

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include <complex>
#include <cstddef>

/*
 * Piecewise Chebyshev evaluation by Clenshaw recursion, as solprop forms it.
 *
 * solprop's tables (`solprop.translate.chebyshev.PiecewiseChebyshev`) hold, on interval
 * `k` of the breakpoints `edges`, the coefficients `c[k, i, j]` of value `j` in the
 * Chebyshev basis, lowest degree first, laid out `(n_intervals, n_terms, n_values)` and
 * C-contiguous. A position `z` takes the interval `searchsorted(edges, z, "right") - 1`
 * clamped to the intervals, so an interior breakpoint belongs to the interval starting
 * there and the span's right end to the last one; it is mapped onto [-1, 1] as
 * `x = (2 z - lo - hi) / (hi - lo)` and every value is NumPy's `chebval` recursion at
 * `x`. The functions here perform the operations of solprop's `_clenshaw_points`, in
 * its order, under strict arithmetic (fp.hpp): no contraction and no reassociation, and
 * complex values multiplied by a real `x` as NumPy multiplies a complex by a real
 * promoted to complex, `(a + i b)(x + i 0)`. Contraction would change the integrated
 * weights whose difference is a stage's phase increment (solprop SEP 25 section 4.2).
 *
 * The functions are callable from host and device code; the host bindings are in
 * src/helpers.cpp.
 */
namespace solprop_native::chebyshev {

using Complex = Kokkos::complex<double>;

/** How a stored coefficient or value is read and written in its working type. */
template <class Stored> struct Coefficient;

template <> struct Coefficient<double> {
  using Value = double;
  static KOKKOS_INLINE_FUNCTION double load(const double& stored) { return stored; }
  static KOKKOS_INLINE_FUNCTION void store(double& stored, const double value) {
    stored = value;
  }
};

template <> struct Coefficient<Complex> {
  using Value = Complex;
  static KOKKOS_INLINE_FUNCTION Complex load(const Complex& stored) { return stored; }
  static KOKKOS_INLINE_FUNCTION void store(Complex& stored, const Complex value) {
    stored = value;
  }
};

/**
 * A NumPy complex128 element, read through the array access the standard guarantees
 * for `std::complex` ([complex.numbers]): its real and imaginary parts as two doubles.
 */
template <> struct Coefficient<std::complex<double>> {
  using Value = Complex;
  static KOKKOS_INLINE_FUNCTION Complex load(const std::complex<double>& stored) {
    const auto& parts = reinterpret_cast<const double (&)[2]>(stored);
    return {parts[0], parts[1]};
  }
  static KOKKOS_INLINE_FUNCTION void store(std::complex<double>& stored,
                                           const Complex value) {
    auto& parts = reinterpret_cast<double (&)[2]>(stored);
    parts[0] = value.real();
    parts[1] = value.imag();
  }
};

/** `a + b`, rounded once. */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION double add(const double a, const double b) {
  SOLPROP_NATIVE_STRICT_BLOCK
  return a + b;
}

/** `a - b`, rounded once. */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION double sub(const double a, const double b) {
  SOLPROP_NATIVE_STRICT_BLOCK
  return a - b;
}

/** `a b`, rounded once. */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION double mul(const double a, const double b) {
  SOLPROP_NATIVE_STRICT_BLOCK
  return a * b;
}

/** `a + b`, each part rounded once. */
KOKKOS_INLINE_FUNCTION Complex add(const Complex a, const Complex b) {
  return strict_add(a, b);
}

/** `a - b`, each part rounded once. */
KOKKOS_INLINE_FUNCTION Complex sub(const Complex a, const Complex b) {
  return strict_sub(a, b);
}

/** `a b` by the complex product, every product and sum rounded. */
KOKKOS_INLINE_FUNCTION Complex mul(const Complex a, const Complex b) {
  return strict_mul(a, b);
}

/** Map `z` in `[lo, hi]` onto [-1, 1] as `(2 z - lo - hi) / (hi - lo)`. */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION double unit_position(const double z, const double lo,
                                            const double hi) {
  SOLPROP_NATIVE_STRICT_BLOCK
  return (2.0 * z - lo - hi) / (hi - lo);
}

/**
 * Return the interval of `z`: `searchsorted(edges, z, side="right") - 1`, clamped to
 * `[0, n_edges - 2]`. A NaN sorts last, as in NumPy, and takes the last interval.
 * `n_edges` is at least two.
 */
KOKKOS_INLINE_FUNCTION std::size_t interval_index(const double z, const double* edges,
                                                  const std::size_t n_edges) {
  const std::size_t last = n_edges - 2;
  if (Kokkos::isnan(z))
    return last;
  std::size_t low = 0;
  std::size_t high = n_edges;
  while (low < high) {
    const std::size_t middle = (low + high) / 2;
    if (edges[middle] <= z)
      low = middle + 1;
    else
      high = middle;
  }
  const std::size_t index = low == 0 ? 0 : low - 1;
  return index < last ? index : last;
}

/**
 * Return one column's Chebyshev series at `x`: NumPy's `chebval` recursion.
 *
 * `c[i * stride]` is the coefficient of degree `i`, for `i < n_terms`; `n_terms` is at
 * least one. The working type is the coefficients' (double, or Kokkos complex).
 */
template <class Stored>
KOKKOS_INLINE_FUNCTION typename Coefficient<Stored>::Value
clenshaw_column(const Stored* c, const std::size_t stride, const std::size_t n_terms,
                const double x) {
  using Value = typename Coefficient<Stored>::Value;
  using Load = Coefficient<Stored>;
  if (n_terms == 1)
    return add(Load::load(c[0]), Value(mul(0.0, x)));
  if (n_terms == 2)
    return add(Load::load(c[0]), mul(Load::load(c[stride]), Value(x)));
  const Value x2(mul(2.0, x));
  Value c0 = Load::load(c[(n_terms - 2) * stride]);
  Value c1 = Load::load(c[(n_terms - 1) * stride]);
  for (std::size_t i = 3; i <= n_terms; ++i) {
    const Value held = c0;
    c0 = sub(Load::load(c[(n_terms - i) * stride]), c1);
    c1 = add(held, mul(c1, x2));
  }
  return add(c0, mul(c1, Value(x)));
}

/**
 * Write a piecewise series at each of several positions: solprop's `_clenshaw_points`.
 *
 * `c` is `(n_intervals, n_terms, n_values)` with `n_intervals = n_edges - 1`, `out` is
 * `(n_points, n_values)`, both C-contiguous. A position outside `[lower, upper]` (the
 * span widened by its slack) stops the pass: its index is returned and the rows from
 * it on are not written. Otherwise the result is -1. A NaN position is not refused;
 * it takes the last interval and gives NaN values.
 */
template <class Stored>
KOKKOS_INLINE_FUNCTION std::ptrdiff_t
clenshaw_points(const double* z, const std::size_t n_points, const double* edges,
                const std::size_t n_edges, const double lower, const double upper,
                const Stored* c, const std::size_t n_terms, const std::size_t n_values,
                Stored* out) {
  for (std::size_t p = 0; p < n_points; ++p) {
    const double position = z[p];
    if (position < lower || position > upper)
      return static_cast<std::ptrdiff_t>(p);
    const std::size_t k = interval_index(position, edges, n_edges);
    const double x = unit_position(position, edges[k], edges[k + 1]);
    const Stored* interval = c + k * n_terms * n_values;
    for (std::size_t j = 0; j < n_values; ++j)
      Coefficient<Stored>::store(out[p * n_values + j],
                                 clenshaw_column(interval + j, n_values, n_terms, x));
  }
  return -1;
}

} // namespace solprop_native::chebyshev
