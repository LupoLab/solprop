// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "helpers.hpp"

#include "solprop_cpunative/chebyshev.hpp"
#include "solprop_cpunative/lagrange_basis.hpp"

#include <nanobind/ndarray.h>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

/*
 * Host bindings of the helpers solprop's shared evaluations take from the
 * backend that runs them (solprop SEP 29 section 6.6, R2): the stage scalars'
 * Chebyshev rows and the Adams basis. Each is a sub-microsecond call on a few
 * hundred numbers, so the GIL is held, nothing is allocated, and every array is
 * the caller's: its dtype, rank, C-contiguity and shape are checked exactly and
 * nothing is converted.
 */
namespace solprop_cpunative::bindings {
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
}

} // namespace solprop_cpunative::bindings
