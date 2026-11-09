// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/engines.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace solprop_native::engines {

void carrier_frame(ExponentialWeights& weights, const std::vector<double>& values,
                   const std::size_t rows) {
  if (values.empty())
    return;
  if (rows > adams_max_samples || values.size() != 1 + 2 * rows)
    throw std::invalid_argument(
        "the carrier frame's rotations have inconsistent extents");
  weights.shifted = true;
  weights.shift = values[0];
  weights.unshift_re = std::cos(values[0]);
  weights.unshift_im = -std::sin(values[0]);
  for (std::size_t t = 0; t < rows; ++t) {
    weights.factor_re[t] = values[1 + 2 * t];
    weights.factor_im[t] = values[2 + 2 * t];
  }
}

ResidentEngine::ResidentEngine(GeneralState& state, const stage::Tables& tables,
                               stage::Anchor& anchor, const bool fused,
                               const bool compensated, const bool interpolate,
                               std::optional<MarchFault>& fault)
    : state_(state), tables_(tables), anchor_(anchor), fused_(fused),
      compensated_(compensated), interpolate_(interpolate), fault_(fault) {}

void ResidentEngine::start(const double z) {
  inject(march::Stage::startup);
  const auto row = stage::row(tables_, anchor_, z, z);
  state_.adams_start(0, stage::trial(tables_, row));
  if (compensated_)
    // The startup sample's rate is read on the far side of a kink.
    state_.adams_compensation_start(0, stage::rate(tables_, z, false));
}

void ResidentEngine::trial(const double z, const double dz,
                           const march::Stencil& stencil, double* norms) {
  inject(march::Stage::trial);
  std::size_t highest = 0;
  for (std::size_t index = 0; index < stencil.n_orders; ++index)
    highest = std::max(highest, stencil.orders[index]);
  const std::size_t slots = std::min(std::max(highest, stencil.order), stencil.held);
  if (fused_) {
    // The interaction formulation: the history transported to the anchor, the
    // source evaluated at the step's end (solprop's `_ResidentAdams.fused_trial`).
    const auto weights = stage::fused_weights(stencil.nodes.data(), stencil.held,
                                              stencil.rows.data(), slots, stencil.order,
                                              stencil.orders.data(), stencil.n_orders);
    const auto row = stage::row(tables_, anchor_, z, z + dz);
    const auto found =
        state_.adams_fused_trial(dz, weights, {}, stage::trial(tables_, row));
    if (found.size() < stencil.n_orders)
      throw std::logic_error("the trial returned fewer norms than it scored");
    std::copy(found.begin(),
              found.begin() + static_cast<std::ptrdiff_t>(stencil.n_orders), norms);
    return;
  }
  auto weights = stage::weights(stencil.nodes.data(), stencil.held, stencil.rows.data(),
                                slots, stencil.order, stencil.orders.data(),
                                stencil.n_orders, compensated_);
  const double end = z + dz;
  if (tables_.chord != stage::Chord::none) {
    // The carrier frame (solprop's `_ResidentAdams._carrier`): the chord's shift and
    // each history row's rotation exp(i shift (1 - theta_t)).
    const double shift = stage::chord(tables_, z, end);
    const std::size_t rows = weights.width - 1;
    carrier_.assign(1 + 2 * rows, 0.0);
    carrier_[0] = shift;
    for (std::size_t t = 0; t < rows; ++t) {
      const double span = 1.0 - stencil.nodes[t];
      carrier_[1 + 2 * t] = std::cos(shift * span);
      carrier_[2 + 2 * t] = std::sin(shift * span);
    }
    carrier_frame(weights, carrier_, rows);
  }
  const auto row = stage::row(tables_, anchor_, z, end);
  const auto step = stage::trial(tables_, row);
  const auto sample = stage::sample(tables_, anchor_, step, end);
  const auto rate = compensated_ ? stage::rate(tables_, end) : std::vector<double>{};
  const auto found =
      state_.adams_exponential_trial(dz, weights, {}, step, sample, rate);
  if (found.size() < stencil.n_orders)
    throw std::logic_error("the trial returned fewer norms than it scored");
  std::copy(found.begin(),
            found.begin() + static_cast<std::ptrdiff_t>(stencil.n_orders), norms);
}

void ResidentEngine::accept(const double dz, const std::size_t free,
                            const march::Stencil& stencil,
                            const std::vector<march::Entry>& kept) {
  inject(march::Stage::accept);
  AdamsWeights dense;
  for (std::size_t t = 0; t < stencil.order; ++t)
    dense.slot[t] = stencil.rows[t];
  dense.width = interpolate_ ? stencil.order + 1 : 0;
  if (fused_) {
    // The kept rows are transported to the step's end, oldest first.
    AdamsWeights keep;
    for (std::size_t t = 0; t < kept.size(); ++t)
      keep.slot[t] = kept[t].row;
    keep.width = kept.size();
    state_.adams_fused_accept(dz, keep, free, dense);
  } else {
    state_.adams_exponential_accept(dz, free, dense);
  }
  // The step's quadrature score, which solprop's accepted-step hook reads.
  state_.promote_fsal();
}

double ResidentEngine::chord_phase(const double z1, const double z2) {
  // One batch member per native state: its phase is the batch's largest.
  return std::fabs(stage::chord(tables_, z1, z2));
}

std::vector<double> ResidentEngine::step_carrier(const std::size_t order) {
  if (carrier_.empty())
    return {};
  return {carrier_.begin(),
          carrier_.begin() + static_cast<std::ptrdiff_t>(1 + 2 * order)};
}

std::vector<double> ResidentEngine::reading() { return state_.monitor_reading(); }

void ResidentEngine::inject(const march::Stage stage) {
  if (!fault_ || fault_->stage != stage)
    return;
  if (fault_->after > 0) {
    --fault_->after;
    return;
  }
  const auto category = fault_->category;
  fault_.reset();
  if (category == march::Category::convergence)
    throw EnvelopeConvergenceError("an injected native failure");
  if (category == march::Category::value)
    throw std::invalid_argument("an injected native failure");
  throw std::runtime_error("an injected native failure");
}

ResidentRungeKutta::ResidentRungeKutta(GeneralState& state, Trial trial,
                                       const double rtol, const double atol,
                                       const std::vector<double>& weights)
    : state_(state), trial_(std::move(trial)), rtol_(rtol), atol_(atol),
      weights_(weights) {}

double ResidentRungeKutta::trial(const double z, const double dz) {
  const auto partials = trial_(z, dz);
  if (!weights_.empty())
    return weighted_norm(state_, partials, rtol_, atol_, weights_);
  return rk::score(partials.data(), partials.size() / 3, rtol_, atol_);
}

double ResidentRungeKutta::weighted_norm(GeneralState& state,
                                         const std::vector<double>& partials,
                                         const double rtol, const double atol,
                                         const std::vector<double>& weights) {
  const auto scale =
      rk::weighted_scale(partials.data(), partials.size() / 3, rtol, atol, weights);
  return rk::weighted_score(state.weighted_error(scale));
}

void ResidentRungeKutta::accept(const double dz) {
  state_.promote_fsal();
  state_.finish(dz);
}

std::vector<double> ResidentRungeKutta::reading() { return state_.monitor_reading(); }

GeneralResident::GeneralResident(GeneralState& state, const std::size_t elements,
                                 const std::size_t n_omega)
    : state_(state), elements_(elements), n_omega_(n_omega) {}

const double* GeneralResident::host_field() {
  // `Complex` is two doubles, real then imaginary (static-asserted by the bindings).
  return reinterpret_cast<const double*>(state_.host_field());
}

void GeneralResident::field_to_host(double* out) {
  state_.field_to_host(std::span<Complex>(reinterpret_cast<Complex*>(out), elements_));
}

void GeneralResident::count_to_host(const std::size_t bytes) {
  state_.count_to_host(bytes);
}

std::vector<double> GeneralResident::statistics(const session::StatisticsRecipe& recipe,
                                                const double z) {
  using Kind = session::StatisticsRecipe::Kind;
  if (recipe.kind == Kind::spatial)
    return state_.spatial_statistics(recipe.flags, recipe.intensity_factor, z);
  if (recipe.kind != Kind::temporal)
    throw std::logic_error("no statistics were prepared");
  // The admitted statistics read neither the plasma's density, preionisation and
  // taper nor the on-axis synthesis, which the session refuses by name.
  return state_.statistics(recipe.flags, recipe.windows, {}, 0.0, 1.0, 0.0,
                           recipe.intensity_factor, {}, recipe.power_weight);
}

ScriptedEngine::ScriptedEngine(std::vector<double> norms, const double carrier_rate,
                               std::vector<ScriptedEvent>& log)
    : norms_(std::move(norms)), carrier_rate_(carrier_rate), log_(log) {}

void ScriptedEngine::start(const double z) {
  ScriptedEvent event;
  event.start = true;
  event.z = z;
  log_.push_back(std::move(event));
}

void ScriptedEngine::trial(const double z, const double dz,
                           const march::Stencil& stencil, double* norms) {
  if ((next_ + 1) * 3 > norms_.size())
    throw std::runtime_error("the script ran out of norms");
  ScriptedEvent event;
  event.z = z;
  event.dz = dz;
  event.nodes.assign(stencil.nodes.begin(),
                     stencil.nodes.begin() + static_cast<std::ptrdiff_t>(stencil.held));
  event.slots.assign(stencil.rows.begin(),
                     stencil.rows.begin() + static_cast<std::ptrdiff_t>(stencil.held));
  event.order = stencil.order;
  event.orders.assign(stencil.orders.begin(),
                      stencil.orders.begin() +
                          static_cast<std::ptrdiff_t>(stencil.n_orders));
  log_.push_back(std::move(event));
  for (std::size_t index = 0; index < stencil.n_orders; ++index)
    norms[index] = norms_[next_ * 3 + index];
  ++next_;
}

void ScriptedEngine::accept(double /*dz*/, const std::size_t free,
                            const march::Stencil& /*stencil*/,
                            const std::vector<march::Entry>& /*kept*/) {
  log_.back().free = free;
}

double ScriptedEngine::chord_phase(const double z1, const double z2) {
  // A constant chord turning by `carrier_rate` per metre, as solprop's harness
  // forms it: the imaginary part of (1j rate)(z2 - z1).
  return std::fabs(carrier_rate_ * (z2 - z1));
}

ScriptedRungeKutta::ScriptedRungeKutta(const std::vector<double>& norms,
                                       std::size_t& next, const double h0,
                                       const double power,
                                       std::vector<ScriptedTrial>& log)
    : norms_(norms), next_(next), h0_(h0), power_(power), log_(log) {}

double ScriptedRungeKutta::trial(const double z, const double dz) {
  if (next_ >= norms_.size())
    throw std::out_of_range("the script ran out of norms");
  log_.push_back(ScriptedTrial{z, dz});
  const double noise = norms_[next_++];
  return h0_ > 0.0 ? noise * std::pow(dz / h0_, power_) : noise;
}

void ScriptedRungeKutta::accept(double /*dz*/) {}

std::vector<double> ScriptedRungeKutta::reading() {
  throw std::logic_error("a scripted march takes no boundary readings");
}

} // namespace solprop_native::engines
