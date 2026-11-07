// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/watch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

// As in adams_march.cpp (solprop #135): GCC does not take a file's `-ffp-contract=off`
// as an inlining barrier under link-time optimisation, and the optimize attribute this
// pragma gives every function below is one. Clang carries the file's setting on each
// instruction.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

/*
 * The boundary monitor's predictions (solprop SEP 45 section 4.8). Each function
 * performs the operations of the solprop function it is named after, in the same order,
 * including how a comparison treats a NaN: Python's `max(a, b, ...)` keeps the first
 * argument unless a later one compares strictly greater, NumPy's `maximum(x, 0.0)`
 * keeps a NaN, and a comparison with a NaN is false. This translation unit is compiled
 * without contraction and without fast-math on every build (CMakeLists.txt), so every
 * product, quotient and sum rounds as solprop's do; `sqrt` is correctly rounded, as
 * Python's `math.sqrt` is.
 */
namespace solprop_native::watch {
namespace {

/** Python's `max(a, b)`. */
double py_max(const double a, const double b) { return b > a ? b : a; }

/** NumPy's `maximum(x, 0.0)`: the larger, a NaN kept. */
double clip_low(const double x) { return std::isnan(x) || x >= 0.0 ? x : 0.0; }

[[noreturn]] void refuse(const std::string& field) {
  throw std::invalid_argument("the boundary watch's policy has an invalid " + field);
}

/** `_settle`: the cadence and the quiet flag from a check's shares. */
void settle(const Policy& policy, State& state, const bool light, const double* shares,
            const double raw_collar) {
  bool calm = true;
  if (!light) {
    state.busy = false;
  } else {
    state.busy = py_max(py_max(shares[0], shares[1]), shares[2]) > policy.activity;
    calm = raw_collar < policy.quiet_share;
  }
  state.quiet_run = calm ? state.quiet_run + 1 : 0;
  state.quiet = policy.quiet_collar && state.quiet_run >= policy.quiet_checks;
}

} // namespace

void validate(const Policy& policy) {
  if (policy.cadence < 1)
    refuse("cadence");
  if (!(policy.interval > 0.0))
    refuse("interval");
  if (!(policy.warn > 0.0) || !(policy.stop >= policy.warn))
    refuse("warning or stop threshold");
  if (!(policy.activity >= 0.0) || !(policy.launch_floor >= 0.0))
    refuse("activity threshold or launch floor");
  if (policy.crossing && policy.bucket_limits.size() != buckets)
    refuse("bucket limits");
}

bool due(const Policy& policy, const State& state, const double z) {
  if (state.busy || state.since + 1 >= policy.cadence)
    return true;
  return !state.has_last || z - state.last_z >= policy.interval;
}

void skip(State& state) { ++state.since; }

double split_limit(const Policy& policy, const double* occupancy) {
  // The cumulative sum in order, its last element the total (solprop SEP 45 Q7).
  double cumulative[buckets];
  double running = 0.0;
  for (std::size_t index = 0; index < buckets; ++index) {
    running += occupancy[index];
    cumulative[index] = running;
  }
  const double total = cumulative[buckets - 1];
  if (!(total > 0.0))
    return std::numeric_limits<double>::infinity();
  const double target = policy.share * total;
  // `searchsorted(cumulative, target, side="right")`.
  const auto index = static_cast<std::size_t>(
      std::upper_bound(cumulative, cumulative + buckets, target) - cumulative);
  return policy.bucket_limits[std::min(index, buckets - 1)];
}

std::uint64_t steps_to_check(const Policy& policy, const State& state) {
  if (state.busy || !state.has_last)
    return 1;
  return policy.cadence > state.since
             ? std::max<std::uint64_t>(policy.cadence - state.since, 1)
             : 1;
}

Verdict check(const Policy& policy, State& state, const double* reading,
              const double z) {
  // `_read_cap`, before anything is judged.
  if (policy.crossing)
    state.cap = split_limit(policy, reading + reading_head);
  // `_found`: under the split, what the step's split found.
  double total = reading[0];
  double regions[4] = {reading[1], reading[2], reading[3], reading[4]};
  if (policy.split) {
    const double* found = reading + 6;
    if (!(found[0] > 0.0 || !(reading[0] > 0.0)))
      return Verdict::error;
    total = found[0];
    regions[0] = found[1];
    regions[1] = found[2];
    regions[2] = found[3];
  }
  // `_judged`.
  if (total <= 0.0 || !std::isfinite(total)) {
    total = 0.0;
    regions[0] = regions[1] = regions[2] = regions[3] = 0.0;
  }
  const bool was_quiet = state.quiet;
  state.since = 0;
  state.has_last = true;
  state.last_z = z;
  if (!(total > policy.launch_floor)) {
    // Nothing was launched, or nothing is left: no light to judge.
    settle(policy, state, false, nullptr, 0.0);
    return state.quiet != was_quiet ? Verdict::quiet : Verdict::none;
  }
  double shares[4];
  for (std::size_t index = 0; index < 4; ++index) {
    const double above = regions[index] - policy.floor_regions[index];
    shares[index] = (index < 3 ? clip_low(above) : above) / total;
  }
  settle(policy, state, true, shares, regions[2] / total);
  // `_judge`: the amplitude, compared with the amplitudes.
  const double worst = std::sqrt(py_max(py_max(shares[0], shares[1]), 0.0));
  if (worst > policy.stop)
    return Verdict::stop;
  if (worst > policy.warn && !state.warned) {
    state.warned = true;
    return Verdict::warn;
  }
  return state.quiet != was_quiet ? Verdict::quiet : Verdict::none;
}

double strict_probe(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}

} // namespace solprop_native::watch
