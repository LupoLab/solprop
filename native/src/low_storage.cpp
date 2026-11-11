// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/low_storage.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "solprop_native/adams_march.hpp"

// As in adams_march.cpp (solprop #135): the step's positions and the table keys round
// as solprop's Python does, so this file takes no contraction (CMake) and the optimize
// attribute this pragma gives every function below keeps GCC from inlining one into a
// contracting caller under link-time optimisation.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace solprop_native::rk::low_storage {

namespace {

/** Python's `max(a, b)`. */
double py_max(const double a, const double b) { return b > a ? b : a; }

/** Append one stage's scalars, all but the increment, as `_trial_arrays` stacks a
    bundle's: the order of `stage::rows`. */
void append(TrialScalars& out, const stage::Row& one) {
  out.phi_at.insert(out.phi_at.end(), one.phi_at.begin(), one.phi_at.end());
  out.prefactor.insert(out.prefactor.end(), one.prefactor.begin(), one.prefactor.end());
  out.density.push_back(one.density);
  out.taper.push_back(one.taper);
  out.preionisation.push_back(
      one.plasma_preionisation.size() == 1 ? one.plasma_preionisation[0] : 0.0);
  out.raman_rates.insert(out.raman_rates.end(), one.raman_rates.begin(),
                         one.raman_rates.end());
  out.kerr_density.push_back(one.kerr_density);
  out.plasma_densities.insert(out.plasma_densities.end(), one.plasma_densities.begin(),
                              one.plasma_densities.end());
  out.plasma_preionisation.insert(out.plasma_preionisation.end(),
                                  one.plasma_preionisation.begin(),
                                  one.plasma_preionisation.end());
  out.raman_densities.insert(out.raman_densities.end(), one.raman_densities.begin(),
                             one.raman_densities.end());
  out.dispersive_densities.insert(out.dispersive_densities.end(),
                                  one.dispersive_densities.begin(),
                                  one.dispersive_densities.end());
  out.transform_weights.insert(out.transform_weights.end(),
                               one.transform_weights.begin(),
                               one.transform_weights.end());
}

/** Write an increment's weights into row `row` of `phi_delta`, `n_phi` wide. */
void put(std::vector<double>& phi_delta, const std::size_t row, const std::size_t n_phi,
         const std::vector<double>& weights) {
  if (weights.size() != n_phi)
    throw std::logic_error("an increment's weights are not the stage rows' width");
  std::copy(weights.begin(), weights.end(),
            phi_delta.begin() + static_cast<std::ptrdiff_t>(row * n_phi));
}

} // namespace

std::vector<double> table_steps(const std::vector<double>& steps,
                                const std::vector<double>& edges,
                                const int landing_ulps) {
  const std::size_t n = steps.size();
  if (edges.size() != n + 1)
    throw std::invalid_argument("a schedule has one edge more than its steps");
  std::vector<double> keys(n);
  std::size_t start = 0;
  while (start < n) {
    std::size_t end = start + 1;
    while (end < n) {
      // `LANDING_ULPS * max(spacing(reach), spacing(steps[start]))`.
      const double reach = py_max(std::fabs(edges[start]), std::fabs(edges[end + 1]));
      const double spacing = py_max(march::ulp(reach), march::ulp(steps[start]));
      const double slack = static_cast<double>(landing_ulps) * spacing;
      if (std::fabs(steps[end] - steps[start]) > slack)
        break;
      ++end;
    }
    // A run of one keys on its own step, unrounded by the edges' difference.
    const double key = end - start == 1 ? steps[start]
                                        : (edges[end] - edges[start]) /
                                              static_cast<double>(end - start);
    std::fill(keys.begin() + static_cast<std::ptrdiff_t>(start),
              keys.begin() + static_cast<std::ptrdiff_t>(end), key);
    start = end;
  }
  return keys;
}

double schedule_key(const std::vector<double>& keys, const std::vector<double>& edges,
                    const double z, const double dz) {
  if (keys.empty())
    throw std::invalid_argument("a schedule's table keys are one a step");
  // `searchsorted(edges, z + 0.5 * dz) - 1`, clamped to the steps.
  const double half = 0.5 * dz;
  const double midpoint = z + half;
  const auto found = std::lower_bound(edges.begin(), edges.end(), midpoint);
  const auto index = static_cast<std::ptrdiff_t>(found - edges.begin()) - 1;
  const auto last = static_cast<std::ptrdiff_t>(keys.size()) - 1;
  return keys[static_cast<std::size_t>(
      std::min(std::max(index, std::ptrdiff_t{0}), last))];
}

TrialScalars scalars(const stage::Tables& tables, stage::Anchor& anchor,
                     const Config& config, const double z, const double dz,
                     const bool rebuild, const double key) {
  const std::size_t stages = tables.n_stages;
  if (stages < 3)
    throw std::invalid_argument("the low-storage march reads three scalar rows");
  if (rebuild && config.flows != 1)
    throw std::logic_error("only a constant operator's table is rebuilt");
  const double half_step = 0.5 * dz;
  const double midpoint = z + half_step;
  const double end = z + dz;
  // A z table's positions, loaded together as `_weight_rows` loads a step's.
  if (tables.phi == stage::Terms::table || tables.prefactor == stage::Terms::table)
    static_cast<void>(tables.z_slots->load(*tables.z_state, {z, midpoint, end}));
  // Each evaluation reads its position as its own anchor; the end's row pads the
  // stages the march does not read.
  const stage::Row start = stage::row(tables, anchor, z, z);
  const stage::Row middle = stage::row(tables, anchor, midpoint, midpoint);
  const stage::Row last = stage::row(tables, anchor, end, end);
  TrialScalars out;
  append(out, start);
  append(out, middle);
  for (std::size_t stage = 2; stage < stages; ++stage)
    append(out, last);
  const std::size_t n_phi = start.phi_at.size();
  out.phi_delta.assign((stages + 1) * n_phi, 0.0);
  if (config.separable) {
    // A constant operator's increment is the distance itself.
    out.phi_delta[0] = half_step;
    out.phi_delta[n_phi] = half_step;
    if (rebuild)
      out.phi_delta[2 * n_phi] = 0.5 * key;
    return out;
  }
  if (config.flows != 1) {
    // A modal flow formed for the step reads its interval.
    put(out.phi_delta, 0, n_phi, stage::row(tables, anchor, z, midpoint).phi_delta);
    put(out.phi_delta, 1, n_phi, stage::row(tables, anchor, midpoint, end).phi_delta);
  }
  if (rebuild) {
    // From the operator's origin, not fl(z + h/2) - z (solprop's engines).
    const double half = config.origin + 0.5 * key;
    put(out.phi_delta, 2, n_phi,
        stage::row(tables, anchor, config.origin, half).phi_delta);
  }
  return out;
}

double strict_probe(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}

} // namespace solprop_native::rk::low_storage

namespace solprop_native::engines {

ResidentLowStorage::ResidentLowStorage(GeneralState& state, const stage::Tables& tables,
                                       stage::Anchor& anchor,
                                       const rk::low_storage::Config config,
                                       const march::Schedule& schedule,
                                       const int landing_ulps,
                                       std::optional<MarchFault>& fault)
    : state_(state), tables_(tables), anchor_(anchor), config_(config),
      keys_(rk::low_storage::table_steps(schedule.steps, schedule.edges, landing_ulps)),
      edges_(schedule.edges), fault_(fault),
      table_step_(std::numeric_limits<double>::quiet_NaN()) {
  if (config_.flows > 2)
    throw std::invalid_argument("the low-storage march holds at most two flows");
  if (keys_.empty())
    throw std::invalid_argument("the low-storage march replays a schedule with steps");
}

double ResidentLowStorage::trial(const double z, const double dz) {
  // In flight from here to the finish: the state is overwritten in place
  // (`StepDriver._in_flight`, set before the engine's step).
  in_flight_ = true;
  double key = std::numeric_limits<double>::quiet_NaN();
  bool rebuild = false;
  if (config_.flows == 1) {
    key = rk::low_storage::schedule_key(keys_, edges_, z, dz);
    rebuild = !(key == table_step_);
  }
  const auto scalars =
      rk::low_storage::scalars(tables_, anchor_, config_, z, dz, rebuild, key);
  const double squared = state_.low_storage_step(dz, scalars, rebuild);
  if (rebuild)
    table_step_ = key;
  // An injected failure after the passes, the state already overwritten.
  inject(march::Stage::trial);
  // The history's norm is 0.0, "not estimated"; a non-finite norm is a divergence,
  // which the replay stops at with the step uncommitted.
  return std::isfinite(squared) ? 0.0 : squared;
}

void ResidentLowStorage::accept(const double dz) {
  inject(march::Stage::accept);
  // The backend's finish, then the accepted-step hook (`_record`'s
  // `_on_accepted_step`); the step commits only once both are done.
  state_.finish(dz);
  state_.promote_fsal();
  in_flight_ = false;
}

std::vector<double> ResidentLowStorage::reading() { return state_.monitor_reading(); }

void ResidentLowStorage::inject(const march::Stage stage) {
  // `ResidentEngine::inject`: the lifecycle tests' failure, after `after` passes.
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

} // namespace solprop_native::engines
