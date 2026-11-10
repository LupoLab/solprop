// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/dense_output.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "solprop_native/engines.hpp"
#include "solprop_native/lagrange_basis.hpp"

// As in session.cpp (solprop #135): the weights an observation reads round as
// solprop's Python forms them, so this file takes no contraction (CMake) and the
// optimize attribute this pragma gives every function below keeps GCC from inlining
// one into a contracting caller under link-time optimisation.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace solprop_native::dense {

double fraction(const double position, const double z, const double dz) {
  const double offset = position - z;
  return offset / dz;
}

AdamsWeights fused_weights(const double* nodes, const std::size_t count,
                           const double theta, const double dz) {
  if (count == 0 || count > adams_max_samples || count > lagrange::max_nodes)
    throw std::invalid_argument("an Adams interpolant has too many nodes");
  // `dz * integrated_weights(nodes, theta)`: the native basis, then NumPy's product.
  AdamsWeights table;
  std::vector<double> scratch(count * count, 0.0);
  std::vector<double> integrated(count, 0.0);
  lagrange::basis_rows(nodes, count, theta, true, scratch.data(), integrated.data());
  for (std::size_t t = 0; t < count; ++t) {
    const double weight = dz * integrated[t];
    table.predictor[t] = weight;
    table.corrector[t] = weight;
  }
  return table;
}

ExponentialWeights exponential_weights(const double* nodes, const std::size_t count,
                                       const double xi, const bool compensated) {
  if (count == 0 || count > adams_max_samples || count > lagrange::max_nodes)
    throw std::invalid_argument("an Adams interpolant has too many nodes");
  ExponentialWeights weights;
  // `lagrange_coefficients(nodes)`, row-major as the binding lays it out.
  std::vector<double> rows(count * count, 0.0);
  std::vector<double> unused(count, 0.0);
  lagrange::basis_rows(nodes, count, 1.0, false, rows.data(), unused.data());
  for (std::size_t j = 0; j < count; ++j)
    for (std::size_t k = 0; k < count; ++k)
      weights.corrector[j][k] = rows[j * count + k];
  if (compensated) {
    // `integrated_weights(nodes, xi)` over the step's own samples.
    std::vector<double> scratch(count * count, 0.0);
    std::vector<double> integrated(count, 0.0);
    lagrange::basis_rows(nodes, count, xi, true, scratch.data(), integrated.data());
    for (std::size_t t = 0; t < count; ++t) {
      weights.dense_weight[t] = integrated[t];
      weights.dense_theta[t] = nodes[t];
    }
  }
  return weights;
}

double strict_probe(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}

} // namespace solprop_native::dense

namespace solprop_native::engines {

AdamsDense::AdamsDense(GeneralState& state, const stage::Tables& tables,
                       stage::Anchor& anchor, const bool fused, const bool compensated)
    : state_(state), tables_(tables), anchor_(anchor), fused_(fused),
      compensated_(compensated) {}

void AdamsDense::stepped(const march::Result& result) {
  // `march_stepped`: a call that accepted a step sets the interpolant, its nodes
  // `[1.0, *step_nodes[:order]]`, and the step's carrier rotations; a call that
  // accepted none keeps the step before.
  if (!result.stepped)
    return;
  if (result.step_order + 1 > nodes_.size())
    throw std::logic_error("an accepted Adams step has more nodes than a history");
  held_ = true;
  z_ = result.step_z;
  dz_ = result.step_dz;
  count_ = result.step_order + 1;
  nodes_[0] = 1.0;
  for (std::size_t t = 0; t < result.step_order; ++t)
    nodes_[t + 1] = result.step_nodes[t];
  carrier_ = result.step_carrier;
}

bool AdamsDense::crosses(const double position, const double target,
                         int /*ulps*/) const {
  return position > target;
}

void AdamsDense::observe(const double position) {
  if (!held_)
    throw std::logic_error("no accepted Adams step is held for a dense observation");
  // `_NativeAdamsMarch.march_to`: the accepted state is kept aside, then observed.
  state_.adams_hold();
  if (!fused_) {
    const double xi = dense::fraction(position, z_, dz_);
    auto weights = dense::exponential_weights(nodes_.data(), count_, xi, compensated_);
    // The accepted step's carrier frame, its history rows after the endpoint.
    carrier_frame(weights, carrier_, count_ - 1);
    state_.adams_exponential_observe(xi, dz_, weights);
    return;
  }
  const double theta = dense::fraction(position, z_, dz_);
  const auto weights = dense::fused_weights(nodes_.data(), count_, theta, dz_);
  // The trial's arrays from the step's anchor to the plane (`arrays(z, position)`).
  const auto scalars =
      stage::trial(tables_, stage::row(tables_, anchor_, z_, position));
  state_.adams_observe(weights, false, scalars);
}

void AdamsDense::restore() { state_.adams_restore(); }

RungeKuttaDense::RungeKuttaDense(GeneralState& state, const stage::Tables& tables,
                                 stage::Anchor& anchor,
                                 const std::vector<double>& weights)
    : state_(state), tables_(tables), anchor_(anchor) {
  state_.rk_dense_allocate(weights);
}

RungeKuttaDense::~RungeKuttaDense() { state_.rk_dense_release(); }

void RungeKuttaDense::hold(const double z, const double dz) {
  // `_dense_record`, before the step's state is adopted.
  state_.rk_dense_hold(dz);
  held_ = true;
  z_ = z;
  dz_ = dz;
}

void RungeKuttaDense::stepped(const march::Result& /*result*/) {}

bool RungeKuttaDense::crosses(const double position, const double target,
                              const int ulps) const {
  return !march::reached(target, position, ulps);
}

void RungeKuttaDense::observe(const double position) {
  if (!held_)
    throw std::logic_error(
        "no accepted Runge-Kutta step is held for a dense observation");
  // `_observe_inside`: the extension at the plane's fraction, carried there from the
  // step's anchor by `U(position, z)`.
  const double theta = dense::fraction(position, z_, dz_);
  const auto carry = stage::trial(tables_, stage::row(tables_, anchor_, z_, position));
  state_.rk_dense_observe(theta, carry);
}

void RungeKuttaDense::restore() { state_.rk_dense_restore(); }

} // namespace solprop_native::engines
