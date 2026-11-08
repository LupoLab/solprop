// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

/*
 * The boundary monitor's checks inside a compiled march (solprop SEP 45 section 4.8).
 *
 * A march asks, after each committed step, whether solprop's boundary monitor would
 * check it (`due`), and if so takes the state's reading and predicts what the monitor
 * will make of it (`check`): the split collar's new crossing cap, the cadence, the
 * quiet collar's verdict, and whether the check warns for the first time or stops the
 * run. solprop replays every reading through its own monitor when it drains the call,
 * so the monitor's record and its arithmetic stay solprop's; these functions only
 * decide where the march must hand back. Each performs the comparisons and arithmetic
 * of the solprop method it is named after (`BoundaryMonitor.due`, `.skip`, `.step` with
 * `_read_cap`, `_found`, `_judged`, `_above`, `_settle` and `_judge`, and
 * `SplitCrossing.limit`), in the same order, with Python's comparison semantics. Every
 * threshold arrives precomputed in the policy, so no constant is held here.
 */
namespace solprop_native::watch {

/** A reading's layout: the state's energy, its four regions, the split's removal since
    the last reading, the four energies the split found, then the occupancy buckets. */
inline constexpr std::size_t reading_head = 10;
inline constexpr std::size_t buckets = 128;

/** Every number a prediction reads, formed by solprop from its monitor
    (`BoundaryMonitor.watch_policy`). */
struct Policy {
  /** The split scheme: a check judges what the step's split found. */
  bool split = false;
  /** A crossing limit: each reading carries the occupancy and gives a new cap. */
  bool crossing = false;
  /** The floor's regions times their margins, the guards' and collar's three and the
      rate's one (`_floor_regions * [FLOOR_MARGIN] * 3 + [1]`). */
  double floor_regions[4] = {0.0, 0.0, 0.0, 0.0};
  /** No light is judged below this energy (`_FLOOR * _launch`). */
  double launch_floor = 0.0;
  /** A share above this keeps every step checked (`(ACTIVE * budget) ** 2`). */
  double activity = 0.0;
  /** The amplitude above which a check warns once (`budget`) and stops
      (`stop_factor * budget`). */
  double warn = 0.0;
  double stop = 0.0;
  /** The collar's share below which a check is calm (`QUIET`), the calm checks in a
      row before it is quiet (`QUIET_CHECKS`), and whether the run opted in. */
  double quiet_share = 0.0;
  std::uint64_t quiet_checks = 0;
  bool quiet_collar = false;
  /** Committed steps between checks while quiet (`CADENCE`) and the longest stretch
      between them, m (`interval`). */
  std::uint64_t cadence = 0;
  double interval = 0.0;
  /** The crossing limit's share set aside and each bucket's step limit, m. */
  double share = 0.0;
  std::vector<double> bucket_limits;
};

/** What a prediction reads and changes: solprop's monitor state at the call. */
struct State {
  bool has_last = false;       ///< whether a check has been made (`_last is not None`)
  double last_z = 0.0;         ///< the last check's position, m (`_last[0]`)
  std::uint64_t since = 0;     ///< committed steps since it (`_since_check`)
  bool busy = false;           ///< `_busy`
  std::uint64_t quiet_run = 0; ///< `_quiet_run`
  bool quiet = false;          ///< `_quiet`
  bool warned = false;         ///< `_warned`
  double cap = std::numeric_limits<double>::infinity(); ///< `_cap`, m
};

/** What a check predicts solprop's monitor will do, the first that applies. */
enum class Verdict : int {
  none = 0,   ///< nothing solprop must act on before the next step
  stop = 1,   ///< `_judge` raises `BoundaryError`
  warn = 2,   ///< `_judge` warns for the first time
  quiet = 3,  ///< the quiet collar's verdict changes
  refuse = 4, ///< a replayed schedule passes the crossing cap before the next check
  error = 5,  ///< `_found` raises: the split found nothing where the state holds light
};

/** Validate a policy; throw `std::invalid_argument` naming the field at fault. */
void validate(const Policy& policy);
/** `BoundaryMonitor.due`: whether the step just committed at `z` is checked. */
[[nodiscard]] bool due(const Policy& policy, const State& state, double z);
/** `BoundaryMonitor.skip`: count a committed step that is not checked. */
void skip(State& state);
/** `SplitCrossing.limit` over 128 bucket occupancies, m. */
[[nodiscard]] double split_limit(const Policy& policy, const double* occupancy);
/** `BoundaryMonitor.steps_to_check` after a check. */
[[nodiscard]] std::uint64_t steps_to_check(const Policy& policy, const State& state);
/** Predict `BoundaryMonitor.step` on `reading` at `z`, updating `state` as it does. The
    split cap is read first, as `_read_cap` does, whatever the verdict. */
[[nodiscard]] Verdict check(const Policy& policy, State& state, const double* reading,
                            double z);
/** The strictness test's witness, as this translation unit rounds `a b + c`; never
    inlined, as `march::strict_probe`. */
[[nodiscard, gnu::noinline]] double strict_probe(double a, double b, double c);

} // namespace solprop_native::watch
