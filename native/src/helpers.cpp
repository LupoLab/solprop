// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "helpers.hpp"

#include "solprop_native/chebyshev.hpp"
#include "solprop_native/lagrange_basis.hpp"

#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * Host bindings of the helpers solprop's shared evaluations take from the backend
 * that runs them (solprop SEP 29 section 6.6, R2): the stage scalars' Chebyshev rows,
 * the Adams basis and an exponential Adams trial's coefficient tables, formed from
 * that basis in one call. Each is a call of a microsecond or less on a few hundred
 * numbers, so the GIL is held, nothing is allocated, and every array is the caller's:
 * its dtype, rank, C-contiguity and shape are checked exactly and nothing is
 * converted.
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

void trial_tables(const Input<double, 1>& nodes, const std::size_t order,
                  const std::vector<std::size_t>& orders, const bool compensated,
                  const Output<double, 2>& predictor,
                  const Output<double, 2>& corrector, const Output<double, 2>& factored,
                  const Output<double, 1>& spans, const Output<double, 1>& carried,
                  const Output<double, 1>& endpoint, const Output<double, 2>& defects) {
  const std::size_t n = nodes.shape(0);
  const std::size_t n_orders = orders.size();
  const std::size_t highest =
      orders.empty() ? 0 : *std::max_element(orders.begin(), orders.end());
  const bool scored = std::all_of(orders.begin(), orders.end(),
                                  [](const std::size_t value) { return value > 0; });
  if (n_orders == 0 || n_orders > 3 || !scored || order == 0 || order > highest ||
      highest > n || highest + 1 > lagrange::max_nodes)
    throw std::invalid_argument("trial_tables: order " + std::to_string(order) +
                                " and " + std::to_string(n_orders) +
                                " scored orders up to " + std::to_string(highest) +
                                " do not fit " + std::to_string(n) + " nodes");
  const std::size_t width = highest + 1;
  const std::size_t carried_size = compensated ? order : 0;
  if (predictor.shape(0) != order || predictor.shape(1) != order ||
      corrector.shape(0) != width || corrector.shape(1) != width ||
      factored.shape(0) != n_orders || factored.shape(1) != 2 * width ||
      spans.shape(0) != (compensated ? width - 1 : 0) ||
      carried.shape(0) != carried_size ||
      endpoint.shape(0) != (compensated ? order + 1 : 0) ||
      defects.shape(0) != (compensated ? n_orders : 0) ||
      defects.shape(1) != (compensated ? width : 0))
    throw std::invalid_argument(
        "trial_tables: the tables must be (order, order), (width, width) and "
        "(n_orders, 2 width), and the compensation's (width - 1,), (order,), "
        "(order + 1,) and (n_orders, width) when compensated, else empty; order " +
        std::to_string(order) + ", width " + std::to_string(width));
  lagrange::trial_tables(nodes.data(), order, orders.data(), n_orders, width,
                         compensated, predictor.data(), corrector.data(),
                         factored.data(), spans.data(), carried.data(), endpoint.data(),
                         defects.data());
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
  module_.def("trial_tables", &trial_tables, "nodes"_a.noconvert(), "order"_a,
              "orders"_a, "compensated"_a, "predictor"_a.noconvert(),
              "corrector"_a.noconvert(), "factored"_a.noconvert(),
              "spans"_a.noconvert(), "carried"_a.noconvert(), "endpoint"_a.noconvert(),
              "defects"_a.noconvert());
}

} // namespace solprop_native::bindings
