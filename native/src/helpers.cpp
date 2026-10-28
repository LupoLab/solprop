// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "helpers.hpp"

#include "solprop_native/chebyshev.hpp"
#include "solprop_native/lagrange_basis.hpp"
#include "solprop_native/vmath.hpp"

#include <nanobind/ndarray.h>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

/*
 * Host bindings of the helpers solprop's shared evaluations take from the backend
 * that runs them (solprop SEP 29 section 6.6, R2): the stage scalars' Chebyshev rows
 * and the Adams basis. Each is a sub-microsecond call on a few hundred numbers, so the
 * GIL is held, nothing is allocated, and every array is the caller's: its dtype, rank,
 * C-contiguity and shape are checked exactly and nothing is converted. The host
 * vector-maths layer's batch functions (vmath.hpp) are bound the same way, for its
 * accuracy tests against a high-precision reference.
 */
namespace solprop_native::bindings {
namespace {

namespace nb = nanobind;
using namespace nb::literals;

template <class T, std::size_t Rank>
using Input = nb::ndarray<const T, nb::ndim<Rank>, nb::c_contig, nb::device::cpu>;
template <class T, std::size_t Rank>
using Output = nb::ndarray<T, nb::ndim<Rank>, nb::c_contig, nb::device::cpu>;

std::string shape_text(const std::size_t rows, const std::size_t columns) {
  return "(" + std::to_string(rows) + ", " + std::to_string(columns) + ")";
}

template <class Stored>
std::ptrdiff_t clenshaw_points(const Input<double, 1>& z, const Input<double, 1>& edges,
                               const double lower, const double upper,
                               const Input<Stored, 3>& c,
                               const Output<Stored, 2>& out) {
  const std::size_t n_edges = edges.shape(0);
  if (n_edges < 2)
    throw std::invalid_argument(
        "clenshaw_points: edges needs at least two breakpoints");
  if (c.shape(0) != n_edges - 1 || c.shape(1) < 1)
    throw std::invalid_argument("clenshaw_points: coefficients of shape " +
                                shape_text(c.shape(0), c.shape(1)) +
                                "[...] do not match " + std::to_string(n_edges) +
                                " breakpoints with at least one term");
  const std::size_t n_points = z.shape(0);
  const std::size_t n_values = c.shape(2);
  if (out.shape(0) != n_points || out.shape(1) != n_values)
    throw std::invalid_argument("clenshaw_points: out has shape " +
                                shape_text(out.shape(0), out.shape(1)) + ", expected " +
                                shape_text(n_points, n_values));
  return chebyshev::clenshaw_points(z.data(), n_points, edges.data(), n_edges, lower,
                                    upper, c.data(), c.shape(1), n_values, out.data());
}

void basis_rows(const Input<double, 1>& points, const double upper,
                const bool integrate, const Output<double, 2>& rows,
                const Output<double, 1>& weights) {
  const std::size_t n = points.shape(0);
  if (n > lagrange::max_nodes)
    throw std::invalid_argument("basis_rows: " + std::to_string(n) +
                                " nodes exceed the " +
                                std::to_string(lagrange::max_nodes) + " supported");
  if (rows.shape(0) != n || rows.shape(1) != n || weights.shape(0) != n)
    throw std::invalid_argument(
        "basis_rows: rows " + shape_text(rows.shape(0), rows.shape(1)) +
        " and weights (" + std::to_string(weights.shape(0)) +
        ",) must be (n, n) and (n,) for n = " + std::to_string(n) + " nodes");
  lagrange::basis_rows(points.data(), n, upper, integrate, rows.data(), weights.data());
}

void require_length(const char* name, const std::size_t expected,
                    const std::size_t actual) {
  if (actual != expected)
    throw std::invalid_argument(std::string(name) + ": the arrays have lengths " +
                                std::to_string(expected) + " and " +
                                std::to_string(actual));
}

void vector_exp(const Input<double, 1>& x, const Output<double, 1>& out) {
  require_length("vector_exp", x.shape(0), out.shape(0));
  vmath::exp(x.data(), out.data(), x.shape(0));
}

void vector_log(const Input<double, 1>& x, const Output<double, 1>& out) {
  require_length("vector_log", x.shape(0), out.shape(0));
  vmath::log(x.data(), out.data(), x.shape(0));
}

void vector_sincos(const Input<double, 1>& x, const Output<double, 1>& sine,
                   const Output<double, 1>& cosine) {
  require_length("vector_sincos", x.shape(0), sine.shape(0));
  require_length("vector_sincos", x.shape(0), cosine.shape(0));
  vmath::sincos(x.data(), sine.data(), cosine.data(), x.shape(0));
}

void vector_exp_complex(const Input<double, 1>& a, const Input<double, 1>& b,
                        const Output<double, 1>& real, const Output<double, 1>& imag) {
  require_length("vector_exp_complex", a.shape(0), b.shape(0));
  require_length("vector_exp_complex", a.shape(0), real.shape(0));
  require_length("vector_exp_complex", a.shape(0), imag.shape(0));
  vmath::exp_complex(a.data(), b.data(), real.data(), imag.data(), a.shape(0));
}

} // namespace

void bind_helpers(nb::module_& module_) {
  module_.def("clenshaw_points", &clenshaw_points<double>, "z"_a.noconvert(),
              "edges"_a.noconvert(), "lower"_a, "upper"_a, "c"_a.noconvert(),
              "out"_a.noconvert());
  module_.def("clenshaw_points", &clenshaw_points<std::complex<double>>,
              "z"_a.noconvert(), "edges"_a.noconvert(), "lower"_a, "upper"_a,
              "c"_a.noconvert(), "out"_a.noconvert());
  module_.def("basis_rows", &basis_rows, "points"_a.noconvert(), "upper"_a,
              "integrate"_a, "rows"_a.noconvert(), "weights"_a.noconvert());
  module_.def("vector_exp", &vector_exp, "x"_a.noconvert(), "out"_a.noconvert());
  module_.def("vector_log", &vector_log, "x"_a.noconvert(), "out"_a.noconvert());
  module_.def("vector_sincos", &vector_sincos, "x"_a.noconvert(), "sine"_a.noconvert(),
              "cosine"_a.noconvert());
  module_.def("vector_exp_complex", &vector_exp_complex, "a"_a.noconvert(),
              "b"_a.noconvert(), "real"_a.noconvert(), "imag"_a.noconvert());
}

} // namespace solprop_native::bindings
