// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/adams_march.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

#include "solprop_native/core.hpp"

/*
 * The exponential Adams march's decisions and loop (solprop SEP 39 phase 2).
 *
 * Each function performs the operations of the solprop function it is named after
 * (`solprop.backends.adams_core`, `stepper.schedule.landing`, `stepper.controller
 * .limit_step`, `backends.driver.reached`, and `AdamsMarch.march_to` with its attempt,
 * decision, rejection and landing), in the same order, including how a comparison
 * treats a NaN: Python's `min(a, b)` is `b` only if `b < a`, its `max` over a sequence
 * keeps the first element unless a later one is strictly greater, and a comparison with
 * a NaN is false. This translation unit is compiled without contraction and without
 * fast-math on every build (CMakeLists.txt), so every product and sum rounds as
 * Python's do; `pow` is the C library's, as Python's float power is.
 */
namespace solprop_native::march {
namespace {

/** Python's `min(a, b)`. */
double py_min(const double a, const double b) { return b < a ? b : a; }

/** Python's `max(a, b)`. */
double py_max(const double a, const double b) { return b > a ? b : a; }

/** `bisect.bisect_left` over an increasing sequence. */
std::size_t bisect_left(const std::vector<double>& values, const double x) {
  return static_cast<std::size_t>(std::lower_bound(values.begin(), values.end(), x) -
                                  values.begin());
}

/** `bisect.bisect_right` over an increasing sequence. */
std::size_t bisect_right(const std::vector<double>& values, const double x) {
  return static_cast<std::size_t>(std::upper_bound(values.begin(), values.end(), x) -
                                  values.begin());
}

void refuse(const char* field, const double value) {
  throw std::invalid_argument(std::string("the Adams march's policy has ") + field +
                              " = " + std::to_string(value));
}

bool increasing(const std::vector<double>& values) {
  for (std::size_t index = 1; index < values.size(); ++index)
    if (!(values[index] > values[index - 1]))
      return false;
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return std::isfinite(value); });
}

} // namespace

double ulp(const double x) {
  const double magnitude = std::fabs(x);
  if (!std::isfinite(magnitude))
    return std::numeric_limits<double>::quiet_NaN();
  return std::nextafter(magnitude, std::numeric_limits<double>::infinity()) - magnitude;
}

double strict_probe(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}

void validate(const Policy& policy) {
  if (policy.order < 1 || policy.order > 6)
    refuse("order", static_cast<double>(policy.order));
  if (policy.real_intervals.size() < policy.order)
    refuse("real_intervals", static_cast<double>(policy.real_intervals.size()));
  if (!(policy.safety > 0.0 && policy.safety <= 1.0))
    refuse("safety", policy.safety);
  if (!(policy.max_growth > 1.0))
    refuse("max_growth", policy.max_growth);
  if (!(policy.min_factor > 0.0 && policy.min_factor < 1.0))
    refuse("min_factor", policy.min_factor);
  if (!(policy.rejection_growth > 0.0 && policy.rejection_growth <= 1.0))
    refuse("rejection_growth", policy.rejection_growth);
  if (!(policy.hysteresis >= 1.0))
    refuse("hysteresis", policy.hysteresis);
  if (policy.restart_after < 1)
    refuse("restart_after", static_cast<double>(policy.restart_after));
  if (!(policy.collar_stability > 0.0))
    refuse("collar_stability", policy.collar_stability);
  if (!(policy.collar_rate >= 0.0))
    refuse("collar_rate", policy.collar_rate);
  if (!(policy.carrier_cap > 0.0))
    refuse("carrier_cap", policy.carrier_cap);
  if (policy.has_budget && policy.max_calls < 1)
    refuse("max_calls", static_cast<double>(policy.max_calls));
  if (!(policy.dz_min >= 0.0 && policy.dz_min <= policy.dz_max) ||
      !std::isfinite(policy.dz_max))
    refuse("dz_min", policy.dz_min);
  if (!(policy.stretch >= 1.0) || !std::isfinite(policy.stretch))
    refuse("stretch", policy.stretch);
  if (!std::isfinite(policy.z_end))
    refuse("z_end", policy.z_end);
  if (policy.breakpoints.size() < 2 || !increasing(policy.breakpoints))
    refuse("breakpoints", static_cast<double>(policy.breakpoints.size()));
  if (!increasing(policy.save_planes))
    refuse("save_planes", static_cast<double>(policy.save_planes.size()));
  if (!increasing(policy.hard_kinks))
    refuse("hard_kinks", static_cast<double>(policy.hard_kinks.size()));
  if (policy.landing_ulps < 1)
    refuse("landing_ulps", policy.landing_ulps);
  if (policy.has_quiet_interval && !(policy.quiet_interval > 0.0))
    refuse("quiet_interval", policy.quiet_interval);
}

double factor(const Policy& policy, const double norm, const std::size_t order,
              const double growth) {
  if (norm <= 0.0)
    return growth;
  const double raw =
      policy.safety * std::pow(norm, -1.0 / static_cast<double>(order + 1));
  return py_min(growth, py_max(policy.min_factor, raw));
}

Stencil stencil(const Policy& policy, const double z, const double dz,
                const std::vector<Entry>& history, const std::size_t stored_order,
                const bool last_rejected) {
  Stencil out;
  const std::size_t held = history.size();
  out.held = held;
  for (std::size_t index = 0; index < held; ++index) {
    const Entry& entry = history[held - 1 - index];
    out.nodes[index] = (entry.z - z) / dz;
    out.rows[index] = entry.row;
  }
  const std::size_t order = std::min(stored_order, held);
  out.order = order;
  const bool rising = !last_rejected && held > order;
  out.orders[0] = order;
  out.n_orders = 1;
  if (policy.variable_order) {
    // `order - 1` then `order + 1`, each where the history can form it, rising only
    // after an attempt that was not rejected.
    if (order >= 2 && order - 1 <= policy.order && order - 1 <= held)
      out.orders[out.n_orders++] = order - 1;
    if (order + 1 <= policy.order && order + 1 <= held && rising)
      out.orders[out.n_orders++] = order + 1;
  }
  return out;
}

double stable_factor(const Policy& policy, const std::size_t order, const double dz,
                     const bool quiet) {
  if (policy.collar_rate <= 0.0 || quiet)
    return std::numeric_limits<double>::infinity();
  const double interval = policy.collar_stability * policy.real_intervals[order - 1];
  return interval / (policy.collar_rate * dz);
}

double order_weight(const Policy& policy, const double norm, const std::size_t order,
                    const double dz, const bool quiet) {
  const double rate = norm <= 0.0
                          ? std::numeric_limits<double>::infinity()
                          : std::pow(norm, -1.0 / static_cast<double>(order + 1));
  const double allowed = policy.safety * rate;
  return py_min(allowed, stable_factor(policy, order, dz, quiet));
}

OrderChoice choose_order(const Policy& policy, const Stencil& stencil,
                         const double* norms, const double dz, const bool quiet) {
  const std::size_t current = stencil.order;
  if (!policy.variable_order)
    return {policy.order, current, norms[0]};
  std::array<double, max_orders> weights{};
  for (std::size_t index = 0; index < stencil.n_orders; ++index)
    weights[index] =
        order_weight(policy, norms[index], stencil.orders[index], dz, quiet);
  // `max(scores, key=weight)`: the first unless a later one is strictly greater.
  std::size_t best = 0;
  for (std::size_t index = 1; index < stencil.n_orders; ++index)
    if (weights[index] > weights[best])
      best = index;
  if (stencil.orders[best] != current && weights[best] > policy.hysteresis * weights[0])
    return {stencil.orders[best], stencil.orders[best], norms[best]};
  return {current, current, norms[0]};
}

Rejection rejection(const Policy& policy, const double norm, const std::size_t order,
                    const std::size_t rejections, const std::size_t held) {
  const std::size_t count = rejections + 1;
  const bool restart = count >= policy.restart_after && held > 1;
  const double scale = !std::isfinite(norm)
                           ? policy.min_factor
                           : factor(policy, norm, order, policy.rejection_growth);
  return {scale, restart ? 0 : count, restart};
}

double carrier_capped(const Policy& policy, const double z, const double proposal,
                      const double phase_over) {
  // `phase_over` is the chord's phase over [z, min(z + proposal, z_end)].
  if (!policy.carrier)
    return proposal;
  const double end = py_min(z + proposal, policy.z_end);
  if (phase_over <= policy.carrier_cap)
    return proposal;
  return (end - z) * (policy.carrier_cap / phase_over);
}

double quiet_capped(const Policy& policy, const double proposal, const bool quiet) {
  if (!quiet || !policy.has_quiet_interval)
    return proposal;
  return py_min(proposal, policy.quiet_interval);
}

double collar_reentry(const Policy& policy, const double proposal,
                      const std::size_t order, const bool was_quiet, const bool quiet) {
  const bool returning = was_quiet && !quiet;
  if (!returning || policy.collar_rate <= 0.0)
    return proposal;
  const double interval = policy.collar_stability * policy.real_intervals[order - 1];
  return py_min(proposal, interval / policy.collar_rate);
}

Landing landing(const Policy& policy, const double z, const double dz,
                const double z_end) {
  if (!(dz > 0.0) || !std::isfinite(dz))
    throw std::invalid_argument("dz must be finite and positive, got " +
                                std::to_string(dz));
  const auto& edges = policy.breakpoints;
  if (!(edges.front() <= z && z <= edges.back()))
    throw std::invalid_argument("z = " + std::to_string(z) +
                                " m lies outside the operator table");
  const double tolerance =
      policy.landing_ulps * ulp(py_max(std::fabs(z), std::fabs(z_end)));
  if (z >= z_end - tolerance)
    throw std::invalid_argument("z = " + std::to_string(z) +
                                " m is already at or past z_end");
  const std::size_t found = bisect_right(edges, z);
  std::size_t index = found == 0 ? 0 : found - 1;
  index = std::min(index, edges.size() - 2);
  const double width = edges[index + 1] - edges[index];
  // The caps, then the landings: the shortest wins, among equals the target of
  // highest priority (`_bound_order`).
  struct Bound {
    double dz;
    Target target;
    double at;
  };
  const auto better = [](const Bound& candidate, const Bound& held) {
    if (candidate.dz < held.dz)
      return true;
    if (held.dz < candidate.dz)
      return false;
    return static_cast<int>(candidate.target) > static_cast<int>(held.target);
  };
  Bound cap{dz, Target::none, 0.0};
  for (const Bound& other : {Bound{policy.dz_max, Target::max_step, 0.0},
                             Bound{width, Target::interval_width, 0.0}})
    if (better(other, cap))
      cap = other;
  bool any = false;
  Bound land{};
  const auto consider = [&](const Target target, const double position) {
    const Bound candidate{position - z, target, position};
    if (!any || better(candidate, land)) {
      land = candidate;
      any = true;
    }
  };
  const double from = z + tolerance;
  if (const auto k = bisect_left(policy.hard_kinks, from); k < policy.hard_kinks.size())
    consider(Target::hard_kink, policy.hard_kinks[k]);
  if (const auto p = bisect_left(policy.save_planes, from);
      p < policy.save_planes.size())
    consider(Target::save_plane, policy.save_planes[p]);
  if (!(z_end < from))
    consider(Target::end, z_end);
  if (land.dz <= cap.dz)
    return {land.dz, land.target, true, land.at};
  if (land.dz <= py_min(policy.stretch * cap.dz, policy.dz_max))
    return {land.dz, land.target, true, land.at};
  return {cap.dz, cap.target, false, 0.0};
}

bool next_landing(const Policy& policy, const double z, const double target,
                  double& ahead) {
  const double tolerance =
      policy.landing_ulps * ulp(py_max(std::fabs(z), std::fabs(target)));
  double nearest = target;
  for (const auto* positions : {&policy.save_planes, &policy.hard_kinks}) {
    const std::size_t index = bisect_right(*positions, z + tolerance);
    if (index < positions->size())
      nearest = py_min(nearest, (*positions)[index]);
  }
  if (nearest > z + tolerance) {
    ahead = nearest;
    return true;
  }
  return false;
}

Approach approach(const Policy& policy, const double z, const Landing& stop,
                  const double target) {
  if (!stop.landed && policy.equal_approach) {
    double ahead = 0.0;
    if (next_landing(policy, z, target, ahead) && stop.dz < ahead - z &&
        ahead - z < 2.0 * stop.dz) {
      const double half = 0.5 * (ahead - z);
      return {half, z + half, false};
    }
  }
  const double end = stop.landed ? stop.z_target : z + stop.dz;
  const double tolerance = policy.landing_ulps * ulp(py_max(std::fabs(end), 1.0));
  const bool interior = end < policy.z_end - tolerance;
  bool on_kink = false;
  if (interior)
    for (const double kink : policy.hard_kinks)
      if (std::fabs(kink - end) <= tolerance) {
        on_kink = true;
        break;
      }
  return {stop.dz, end, stop.target == Target::hard_kink || on_kink};
}

double proposal(const Policy& policy, const double dz, const double factor) {
  return py_min(dz * factor, policy.dz_max);
}

bool replay_kink(const Policy& policy, const double end, const double slack) {
  return std::any_of(policy.hard_kinks.begin(), policy.hard_kinks.end(),
                     [&](const double kink) { return std::fabs(kink - end) <= slack; });
}

bool reached(const double position, const double target, const int ulps) {
  const double tolerance = ulps * ulp(py_max(std::fabs(position), std::fabs(target)));
  return position >= target - tolerance;
}

namespace {

/** Record a native failure's stage, its solprop error class and its message. */
void fail(Result& result, const Stage stage, const std::exception& error) {
  result.reason = Reason::error;
  result.stage = stage;
  result.message = error.what();
  if (dynamic_cast<const EnvelopeConvergenceError*>(&error) != nullptr)
    result.category = Category::convergence;
  else if (dynamic_cast<const std::invalid_argument*>(&error) != nullptr)
    result.category = Category::value;
  else
    result.category = Category::backend;
}

} // namespace

Result run(const Policy& policy, State& state, Engine& engine, const Bounds& bounds,
           const Schedule& schedule) {
  using Clock = std::chrono::steady_clock;
  const auto started = Clock::now();
  Result result;
  const double target = bounds.target;
  // Dense observation steps across the save planes (SEP 13 section 6.4); the march
  // lands on the run's end and its kinks.
  const double z_end = policy.interpolate ? policy.z_end : target;
  std::uint64_t accepted = 0;
  std::array<double, max_orders> norms{};
  const auto elapsed = [&]() {
    return std::chrono::duration<double>(Clock::now() - started).count();
  };
  const auto record = [&](const double end, const double dz, const double norm,
                          const bool kept, const std::size_t order) {
    result.rows.insert(
        result.rows.end(),
        {end, dz, std::isfinite(norm) ? norm : std::numeric_limits<double>::infinity(),
         kept ? 1.0 : 0.0, static_cast<double>(order), elapsed()});
  };
  // A replay runs while the schedule has steps and its target is not within the
  // slack (`AdamsMarch._replay_to`); an adaptive march until the target is reached.
  const std::size_t n_steps = schedule.steps.size();
  const auto running = [&]() {
    if (bounds.replay)
      return state.cursor < n_steps && state.z < target - bounds.slack;
    return !reached(state.z, target, policy.landing_ulps);
  };
  if (bounds.replay && schedule.edges.size() != n_steps + 1)
    throw std::invalid_argument("a replayed schedule has one more edge than steps");
  while (running()) {
    // The budget, before any call is made (`_afford`).
    const std::uint64_t cost = (state.history.empty() ? 1 : 0) + 1;
    if (policy.has_budget && state.ledger.total() + cost > policy.max_calls) {
      result.reason = Reason::budget;
      result.cost = cost;
      return result;
    }
    if (state.history.empty()) {
      // A call that fails has still been charged (R4).
      ++state.ledger.startup;
      try {
        engine.start(state.z);
      } catch (const std::exception& error) {
        fail(result, Stage::startup, error);
        return result;
      }
      state.history = {Entry{state.z, Identity::accepted, 0}};
      state.order = policy.variable_order ? 1 : policy.order;
    }
    Approach step;
    if (bounds.replay) {
      // The schedule's step, ending on its edge; a kink within the slack restarts.
      step.dz = schedule.steps[state.cursor];
      step.end = schedule.edges[state.cursor + 1];
      step.kink = replay_kink(policy, step.end, bounds.slack);
    } else {
      // The caps: the carrier frame's, the quiet collar's, the collar's re-entry.
      double proposed = state.proposal;
      if (policy.carrier) {
        const double end = py_min(state.z + proposed, policy.z_end);
        proposed =
            carrier_capped(policy, state.z, proposed, engine.chord_phase(state.z, end));
      }
      proposed = quiet_capped(policy, proposed, bounds.quiet);
      proposed = collar_reentry(policy, proposed, state.order, state.collar_was_quiet,
                                bounds.quiet);
      state.collar_was_quiet = bounds.quiet;
      try {
        const Landing stop = landing(policy, state.z, proposed, z_end);
        step = approach(policy, state.z, stop, target);
      } catch (const std::exception& error) {
        fail(result, Stage::trial, error);
        return result;
      }
    }
    const Stencil trial = stencil(policy, state.z, step.dz, state.history, state.order,
                                  state.last_rejected);
    ++state.ledger.predictor;
    try {
      engine.trial(state.z, step.dz, trial, norms.data());
    } catch (const std::exception& error) {
      fail(result, Stage::trial, error);
      return result;
    }
    const double norm = norms[0];
    // A replayed step is accepted whatever its finite norm.
    if (!(norm <= 1.0) && !(bounds.replay && std::isfinite(norm))) {
      // A rejection leaves the accepted history; three in a row restart it.
      ++state.ledger.rejected;
      ++state.ledger.rejected_steps;
      state.last_rejected = true;
      const Rejection outcome =
          rejection(policy, norm, trial.order, state.rejections, state.history.size());
      state.rejections = outcome.rejections;
      if (outcome.restart) {
        state.history.clear();
        ++state.ledger.failure_restarts;
      }
      const double next = proposal(policy, step.dz, outcome.factor);
      if (next < policy.dz_min) {
        // `limit_step` raises inside the rejection, before the attempt is recorded.
        result.reason = Reason::floor;
        result.floor_proposal = next;
        return result;
      }
      state.proposal = next;
      if (bounds.replay) {
        // A replay cannot reject: the field has diverged (`replay_step`), and the
        // attempt is not recorded.
        result.reason = Reason::diverged;
        result.replay_dz = step.dz;
        result.replay_end = step.end;
        result.replay_norm = norm;
        return result;
      }
      record(step.end, step.dz, norm, false, trial.order);
    } else {
      // The new sample takes the lowest row the kept entries do not use.
      const std::size_t keep = std::min(policy.order, state.history.size());
      std::vector<Entry> kept(state.history.end() - static_cast<std::ptrdiff_t>(keep),
                              state.history.end());
      std::size_t free = 0;
      for (;; ++free)
        if (std::none_of(kept.begin(), kept.end(),
                         [free](const Entry& entry) { return entry.row == free; }))
          break;
      try {
        engine.accept(step.dz, free, trial, kept);
      } catch (const std::exception& error) {
        fail(result, Stage::accept, error);
        return result;
      }
      kept.push_back(Entry{state.z + step.dz, Identity::predicted, free});
      state.history = std::move(kept);
      ++state.ledger.accepted_steps;
      state.rejections = 0;
      result.orders.push_back(trial.order);
      const OrderChoice choice =
          choose_order(policy, trial, norms.data(), step.dz, bounds.quiet);
      state.order = choice.stored;
      state.last_rejected = false;
      // The position, and a kink's restart, move in the same commit (SEP 39 4.4).
      result.stepped = true;
      result.step_z = state.z;
      result.step_dz = step.dz;
      result.step_order = trial.order;
      result.step_nodes = trial.nodes;
      state.z = step.end;
      if (step.kink) {
        state.history.clear();
        ++state.ledger.restarts;
      }
      const double next =
          proposal(policy, step.dz,
                   factor(policy, choice.norm, choice.control, policy.max_growth));
      record(step.end, step.dz, norm, true, trial.order);
      if (next < policy.dz_min) {
        result.reason = Reason::floor;
        result.floor_proposal = next;
        return result;
      }
      state.proposal = next;
      ++accepted;
      if (bounds.replay)
        ++state.cursor;
      if (!running()) {
        result.reason = Reason::reached;
        return result;
      }
      if (bounds.host_step > 0 && accepted == bounds.host_step) {
        result.reason = Reason::host_step;
        return result;
      }
      if (bounds.monitor && (bounds.first_check ||
                             state.z - bounds.last_check >= bounds.check_interval)) {
        result.reason = Reason::monitor;
        return result;
      }
    }
    if (result.rows.size() / row_columns >= bounds.max_attempts ||
        elapsed() >= bounds.max_seconds) {
      result.reason = Reason::bound;
      return result;
    }
  }
  result.reason = Reason::reached;
  return result;
}

} // namespace solprop_native::march
