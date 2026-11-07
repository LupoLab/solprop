// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/rk_march.hpp"

#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>

#include "solprop_native/core.hpp"

// As in adams_march.cpp (solprop #135): the optimize attribute this pragma gives every
// function below keeps GCC from inlining it into a contracting caller under link-time
// optimisation. Clang carries the file's setting on each instruction.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace solprop_native::rk {

std::vector<double> positions(const double z, const double dz,
                              const std::vector<double>& nodes) {
  std::vector<double> out;
  out.reserve(nodes.size() + 1);
  for (const double node : nodes) {
    const double offset = node * dz;
    out.push_back(z + offset);
  }
  out.push_back(z + dz);
  return out;
}

double score(const double* partials, const std::size_t slots, const double rtol,
             const double atol) {
  double sums[3] = {0.0, 0.0, 0.0};
  for (std::size_t row = 0; row < 3; ++row)
    for (std::size_t slot = 0; slot < slots; ++slot)
      sums[row] += partials[row * slots + slot];
  const double old = std::sqrt(sums[0]);
  const double advanced = std::sqrt(sums[1]);
  const double residual = std::sqrt(sums[2]);
  // `rtol * max(norm_y, norm_y_new) + atol`, Python's `max`.
  const double larger = advanced > old ? advanced : old;
  const double scaled = rtol * larger;
  return residual / (scaled + atol);
}

double strict_probe(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}

namespace {

/** Python's `min(a, b)`. */
double py_min(const double a, const double b) { return b < a ? b : a; }

/** Python's `max(a, b)`. */
double py_max(const double a, const double b) { return b > a ? b : a; }

/** Record a native failure's stage, its solprop error class and its message. */
void fail(march::Result& result, const march::Stage stage,
          const std::exception& error) {
  result.reason = march::Reason::error;
  result.stage = stage;
  result.message = error.what();
  if (dynamic_cast<const EnvelopeConvergenceError*>(&error) != nullptr)
    result.category = march::Category::convergence;
  else if (dynamic_cast<const BackendRefusal*>(&error) != nullptr)
    result.category = march::Category::refused;
  else if (dynamic_cast<const std::invalid_argument*>(&error) != nullptr)
    result.category = march::Category::value;
  else
    result.category = march::Category::backend;
}

} // namespace

double equivalent(const Policy& policy, const double norm, const double dz_taken,
                  const double dz_proposed) {
  // `PIController.equivalent_norm`: a non-finite norm as it is, else the norm the
  // proposal would have scored, `norm * (dz_proposed / dz_taken) ** (q + 1)`.
  if (!std::isfinite(norm))
    return norm;
  return norm * std::pow(dz_proposed / dz_taken, policy.power);
}

bool accepts(const Policy& policy, const double norm) {
  if (!std::isfinite(norm))
    return false;
  return policy.reject_free || norm <= 1.0;
}

double propose(const Policy& policy, const std::deque<double>& norms) {
  // `PIController.propose`, with the lag's unit factors until the history fills.
  const auto length = static_cast<std::ptrdiff_t>(norms.size());
  const std::ptrdiff_t index = length - 1 - static_cast<std::ptrdiff_t>(policy.lag);
  if (index < 0)
    return 1.0;
  const double current = norms[static_cast<std::size_t>(index)];
  if (!std::isfinite(current))
    return policy.min_factor;
  if (current <= 0.0)
    return policy.max_factor;
  double previous = index >= 1 ? norms[static_cast<std::size_t>(index - 1)] : 1.0;
  if (!std::isfinite(previous) || previous <= 0.0)
    previous = 1.0;
  const double first = policy.safety * std::pow(1.0 / current, policy.beta1);
  const double factor = first * std::pow(1.0 / previous, policy.beta2);
  return py_min(py_max(factor, policy.min_factor), policy.max_factor);
}

double rejection_factor(const Policy& policy, const double norm) {
  if (!std::isfinite(norm))
    return policy.min_factor;
  if (norm <= 0.0)
    return 1.0;
  const double factor = policy.safety * std::pow(1.0 / norm, policy.exponent);
  return py_min(py_max(factor, policy.min_factor), 1.0);
}

double start_factor(const Policy& policy, const double norm) {
  if (!std::isfinite(norm) || norm <= 0.0)
    return norm == 0.0 ? policy.start_growth : policy.min_factor;
  const double factor = policy.safety * std::pow(1.0 / norm, policy.exponent);
  return py_min(py_max(factor, policy.min_factor), policy.start_growth);
}

double limited(const Policy& policy, const double dz, const double factor) {
  return py_min(dz * factor, policy.dz_max);
}

Result run(const Policy& policy, State& state, Engine& engine,
           const march::Bounds& bounds) {
  using Clock = std::chrono::steady_clock;
  const auto started = Clock::now();
  Result out;
  march::Result& result = out.march;
  const double target = bounds.target;
  const int ulps = policy.landings.landing_ulps;
  std::uint64_t accepted = 0;
  const auto elapsed = [&]() {
    return std::chrono::duration<double>(Clock::now() - started).count();
  };
  const auto record = [&](const double end, const double dz, const double norm,
                          const bool kept) {
    result.rows.insert(
        result.rows.end(),
        {end, dz, std::isfinite(norm) ? norm : std::numeric_limits<double>::infinity(),
         kept ? 1.0 : 0.0, 0.0, elapsed()});
  };
  const auto floor = [&](const FloorSite site, const double proposed) {
    result.reason = march::Reason::floor;
    result.floor_proposal = proposed;
    out.floor_site = site;
  };
  const auto remember = [&](const double norm) {
    state.norms.push_back(norm);
    while (state.norms.size() > policy.lag + 2)
      state.norms.pop_front();
  };
  if (bounds.watch)
    watch::validate(bounds.watch_policy);
  // Interpolated planes are stepped across: the march lands on the run's end and its
  // kinks (`StepDriver._march_to`), and returns the step that crosses the target.
  const bool dense = policy.landings.interpolate;
  const double z_end = dense ? policy.landings.z_end : target;
  while (!march::reached(state.z, target, ulps)) {
    // The landing candidate: the proposal capped by the quiet collar's interval, the
    // transverse edge's ceiling and the split collar's cap, in that order; the
    // proposal itself is unchanged.
    const double proposal = state.proposal;
    double candidate = march::quiet_capped(policy.landings, proposal, bounds.quiet);
    // The transverse edge's ceiling (`TransverseEdge.ceiling`): infinite without one.
    candidate = py_min(candidate, bounds.read_spacing);
    candidate = py_min(candidate, bounds.watch ? state.watch.cap : bounds.step_cap);
    march::Landing stop;
    double norm = 0.0;
    try {
      stop = march::landing(policy.landings, state.z, candidate, z_end);
      norm = engine.trial(state.z, stop.dz);
    } catch (const std::exception& error) {
      fail(result, march::Stage::trial, error);
      return out;
    }
    const double equivalent_norm = equivalent(policy, norm, stop.dz, proposal);
    const bool kept = accepts(policy, norm);
    const double end = stop.landed ? stop.z_target : state.z + stop.dz;
    record(end, stop.dz, norm, kept);
    if (!kept) {
      const double next =
          limited(policy, proposal, rejection_factor(policy, equivalent_norm));
      if (next < policy.dz_min) {
        floor(FloorSite::rejection, next);
        return out;
      }
      state.proposal = next;
    } else {
      if (dense && !march::reached(target, end, ulps)) {
        // solprop records the step's extension from the state before it, then
        // finishes and commits it with the reference's own code.
        result.reason = march::Reason::dense;
        out.pending_equivalent = equivalent_norm;
        return out;
      }
      try {
        engine.accept(stop.dz);
      } catch (const std::exception& error) {
        fail(result, march::Stage::accept, error);
        return out;
      }
      state.z = end;
      remember(equivalent_norm);
      double held = proposal;
      if (policy.auto_init && state.norms.size() == 1) {
        // The start-up rule: the deadbeat step, and the norm it would have scored.
        const double factor = start_factor(policy, equivalent_norm);
        const double next = limited(policy, held, factor);
        if (next < policy.dz_min) {
          floor(FloorSite::start_up, next);
          return out;
        }
        state.norms.back() = factor >= policy.start_growth
                                 ? 0.0
                                 : equivalent(policy, equivalent_norm, held, next);
        held = next;
        state.proposal = next;
      }
      const double next = limited(policy, held, propose(policy, state.norms));
      if (next < policy.dz_min) {
        floor(FloorSite::proposal, next);
        return out;
      }
      state.proposal = next;
      ++accepted;
      if (bounds.watch) {
        // The step's boundary check, as the Adams march takes it (SEP 45 4.8).
        const auto& rules = bounds.watch_policy;
        watch::Verdict verdict = watch::Verdict::none;
        if (!watch::due(rules, state.watch, state.z)) {
          watch::skip(state.watch);
        } else {
          std::vector<double> reading;
          try {
            reading = engine.reading();
            if (reading.size() < watch::reading_head ||
                (rules.crossing &&
                 reading.size() < watch::reading_head + watch::buckets))
              throw std::invalid_argument(
                  "a boundary reading is too short for its policy");
          } catch (const std::exception& error) {
            fail(result, march::Stage::accept, error);
            return out;
          }
          result.reading_size = reading.size();
          result.readings.insert(result.readings.end(), reading.begin(), reading.end());
          result.reading_rows.push_back(result.rows.size() / march::row_columns - 1);
          verdict = watch::check(rules, state.watch, reading.data(), state.z);
        }
        if (verdict != watch::Verdict::none) {
          result.reason = march::Reason::watch;
          result.verdict = verdict;
          return out;
        }
      }
      if (march::reached(state.z, target, ulps)) {
        result.reason = march::Reason::reached;
        return out;
      }
      // The transverse edge is read on the host before a step would carry past its
      // next read (`TransverseEdge.due`, solprop SEP 40 section 5.3).
      const bool read_due = bounds.has_read_at &&
                            state.z + state.proposal >=
                                bounds.read_at - 1e-12 * py_max(std::abs(state.z), 1.0);
      if ((bounds.host_step > 0 && accepted == bounds.host_step) || read_due) {
        result.reason = march::Reason::host_step;
        return out;
      }
      // Beside a transverse edge solprop takes the monitor's checks itself, by its
      // distance test (`BoundaryMonitor.due`).
      if (!bounds.watch && bounds.monitor &&
          (bounds.first_check ||
           state.z - bounds.last_check >= bounds.check_interval)) {
        result.reason = march::Reason::monitor;
        return out;
      }
    }
    if (result.rows.size() / march::row_columns >= bounds.max_attempts ||
        elapsed() >= bounds.max_seconds) {
      result.reason = march::Reason::bound;
      return out;
    }
    if (bounds.watch && result.reading_rows.size() >= bounds.watch_capacity) {
      result.reason = march::Reason::capacity;
      return out;
    }
  }
  result.reason = march::Reason::reached;
  return out;
}

} // namespace solprop_native::rk
