// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

/*
 * The boundary monitor's checks inside a compiled march (solprop SEP 45 section 4.8)
 * and the monitor itself, owned by a native session (SEP 47 section 6.4).
 *
 * A march asks, after each committed step, whether solprop's boundary monitor would
 * check it (`due`), and if so takes the state's reading and decides what the monitor
 * makes of it (`check`): the split collar's new crossing cap, the cadence, the quiet
 * collar's verdict, and whether the check warns for the first time or stops the run.
 * Under solprop's driver these are predictions: solprop replays every reading through
 * its own monitor, and they only decide where the march must hand back. Under a native
 * session the march checks with `Monitor`, which owns the monitor's whole state and
 * record and decides by these same functions. Each performs the comparisons and
 * arithmetic of the solprop method it is named after (`BoundaryMonitor.due`, `.skip`,
 * `.step` with `_read_cap`, `_found`, `_judged`, `_above`, `_settle` and `_judge`, and
 * `SplitCrossing.limit`), in the same order, with Python's comparison semantics. Every
 * threshold arrives precomputed, so no constant is held here.
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
  divide = 6, ///< the owning monitor's check divides by a launch that held no energy,
              ///< where solprop's monitor raises `ZeroDivisionError`
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
/** The longest of a replayed schedule's steps from `cursor` over the next `ahead`
    (at least one), as `_refuse_scheduled_crossing` takes `np.max` of them. */
[[nodiscard]] double longest_ahead(const std::vector<double>& steps, std::size_t cursor,
                                   std::uint64_t ahead);
/** `_refuse_scheduled_crossing`: the longest step of a replay with a crossing limit
    before the next check, `ahead` steps at most, when it passes `cap`; zero when the
    schedule is held to the cap, or there is no replay, limit or step left. The one
    predicate of the marches' checks and a session's. */
[[nodiscard]] double refused_crossing(bool crossing, bool replay,
                                      const std::vector<double>& steps,
                                      std::size_t cursor, std::uint64_t ahead,
                                      double cap);

/*
 * The owning monitor (solprop SEP 47 sections 4.3, 5.5 and 6.4, and the maintainer's
 * decision 4 at its acceptance): `BoundaryMonitor`'s whole state and record, held by
 * the native session that owns the run. It makes every decision -- the launch
 * judgement, each check's cadence, quiet collar, warning and stop, the split collar's
 * cap -- by the functions above, so a prediction and the owner share one
 * implementation of each, and it keeps every record: the cumulative removed, raw and
 * floor energies, the guard and collar maxima, the steps checked and skipped, the plane
 * rows, the finish verdict and the record a resume restores. solprop formats and emits
 * what it reports and computes nothing of it.
 *
 * Scalar decisions and the record's scalar arithmetic are strict and in solprop's
 * order, so on identical readings they are solprop's exactly. Two reductions belong to
 * the reading instead (SEP 47 section 4.3) and are compared under SEP 25: the floor's
 * share of a split step's removal, `sum_t (1 - exp(-alpha(t) h))`, summed here in order
 * where NumPy sums pairwise; and the input floor's energy and regions, which the
 * caller reads from the noise draw and hands to `launch`.
 */

/** Every number the owning monitor reads, formed by solprop from its `BoundaryMonitor`
    and module constants. */
struct Config {
  /** The predictions' policy (`BoundaryMonitor.watch_policy`); its floor regions and
      launch floor are formed here at the launch, from `floor_margin` and
      `floor_share`. */
  Policy policy;
  /** `_FLOOR`: no light is judged below this share of the launch energy. */
  double floor_share = 0.0;
  /** `FLOOR_MARGIN`: the floor's expected energy in a region is allowed this many
      times before light is counted there. */
  double floor_margin = 0.0;
  /** The check's budget, which the finish's removal is judged against. */
  double budget = 0.0;
  /** The collar's power rate on the coarse axis, 1/m (`weights[3]`); its length is the
      axis's. */
  std::vector<double> rate;
  /** Whether the route's stages may leave a quiet collar out (`_collar_can_rest`). */
  bool can_rest = false;
};

/** An earlier execution's record as a resume reads it (`BoundaryMonitor.resume`):
    each number present in it, and the lists. solprop reads the record's types; the
    reconstruction -- which fields are restored, their defaults, the fallbacks of a
    record from before a field was kept -- is made here. */
struct Resumption {
  /** Whether there is a record at all. */
  bool record = false;
  std::optional<double> launch_energy;
  std::optional<double> launch_guard_low;
  std::optional<double> launch_guard_high;
  std::optional<double> launch_collar;
  std::optional<double> guard_low_max;
  std::optional<double> guard_high_max;
  std::optional<double> guard_low_at;
  std::optional<double> guard_high_at;
  std::optional<double> collar_max;
  std::optional<double> raw_removed_fraction;
  std::optional<double> removed_signed;
  std::optional<double> removed_fraction;
  std::optional<double> floor_removed_fraction;
  std::optional<double> quiet_checks;
  /** `bool(record.get("collar_quiet", False))`. */
  bool collar_quiet = false;
  std::optional<double> raw_guard_max;
  std::optional<double> steps;
  std::optional<double> skipped;
  std::optional<double> collar_left_out_checks;
  /** The record's `unwatched` list, when it is one. */
  std::optional<std::vector<std::array<double, 2>>> unwatched;
  /** The record's `resumed_at` list, when it is one. */
  std::optional<std::vector<double>> resumed_at;
  /** The record's plane rows, when they have the five columns (NaN for `None`). */
  bool has_planes = false;
  std::array<std::vector<double>, 5> planes;
};

/** One saved plane's rows: `z`, `collar_occupancy`, `guard_low`, `guard_high`,
    `removed_fraction`, in that order. */
inline constexpr std::size_t plane_columns = 5;

/** The monitor's record (`BoundaryMonitor.summary` less the check's thresholds, which
    solprop holds): every number, NaN where solprop records `None`. */
struct Record {
  double launch_energy = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> resumed_at;
  std::array<double, 3> launch_occupancy{};
  std::array<double, 2> guard_max{};
  std::array<double, 2> guard_at{};
  double collar_max = 0.0;
  double removed_fraction = 0.0;
  double removed_signed = 0.0;
  double floor_removed_fraction = 0.0;
  std::uint64_t quiet_checks = 0;
  bool collar_quiet = false;
  std::vector<std::array<double, 2>> unwatched;
  std::uint64_t steps = 0;
  std::uint64_t skipped = 0;
  double noise_floor = 0.0;
  double raw_guard_max = 0.0;
  double raw_removed_fraction = 0.0;
  std::uint64_t collar_left_out_checks = 0;
  std::array<std::vector<double>, plane_columns> planes;
};

/** What a check or the launch judged: its verdict and what solprop's message names. */
struct Judgement {
  Verdict verdict = Verdict::none;
  /** The amplitude of the worse guard occupancy, `sqrt(max(low, high, 0))`. */
  double worst = 0.0;
  /** Where, m; the launch's position for the launch. */
  double z = 0.0;
  /** Whether it is the launch's judgement. */
  bool launch = false;
};

/** The end of a run (`BoundaryMonitor.finish`): the removed fraction, clipped at zero,
    the guards' peaks, and whether the removal warns. */
struct Finish {
  double removed = 0.0;
  std::array<double, 2> guard_max{};
  bool warns = false;
};

/** `BoundaryMonitor`, owned natively. Not thread-safe: the session's lock serialises
    it. */
class Monitor {
public:
  /** Validate `config`; throw `std::invalid_argument` naming the field at fault. */
  explicit Monitor(Config config);

  /** `resume`: continue an earlier execution's record, before `launch`. `plane` is
      whether the run resumes at a saved plane; `unwatched` the stretches, m, run
      since without a monitor. */
  void resume(const Resumption& record, bool plane,
              const std::vector<std::array<double, 2>>& unwatched);
  /** `launch`: record the launched state's reading at `z` and judge it, unless
      resumed. `floor` and `floor_regions` (four, or null for none) are the input
      noise's energy and its expected regions, read from the draw (a reading). */
  Judgement launch(const double* reading, std::size_t size, double z, double floor,
                   const double* floor_regions);
  /** `due`. */
  [[nodiscard]] bool due(double z) const;
  /** `skip`. */
  void skip(double z);
  /** `step`: account the committed state's reading at `z`; `plane` for a saved
      plane's check of a step `skip` already counted. */
  Judgement step(const double* reading, std::size_t size, double z, bool plane = false);
  /** `plane`: the row of the saved plane just reached. */
  void plane(double z);
  /** `note_collar`: whether the stages leave the collar's term out from here on. */
  void note_collar(bool left_out) noexcept { left_out_ = left_out; }
  /** `finish`. */
  [[nodiscard]] Finish finish() const;
  /** `summary`, its numbers. */
  [[nodiscard]] Record record() const;

  /** `stale`: whether the last committed step went unchecked. */
  [[nodiscard]] bool stale() const noexcept { return stale_; }
  /** `collar_quiet`. */
  [[nodiscard]] bool quiet() const noexcept { return state_.quiet; }
  /** `step_cap`, m. */
  [[nodiscard]] double cap() const noexcept { return state_.cap; }
  /** `steps_to_check`. */
  [[nodiscard]] std::uint64_t steps_to_check() const;
  /** The last check's or the launch's judgement. */
  [[nodiscard]] const Judgement& last_judgement() const noexcept { return last_; }
  /** `removed`, signed. */
  [[nodiscard]] double removed() const;
  /** The decision state, as `watch_state` reports it. */
  [[nodiscard]] const State& state() const noexcept { return state_; }
  [[nodiscard]] const Config& config() const noexcept { return config_; }
  /** The fewest doubles a reading holds under this configuration. */
  [[nodiscard]] std::size_t reading_size() const noexcept;
  /** The interval a resting collar caps the steps at (`interval`), m. */
  [[nodiscard]] double interval() const noexcept { return config_.policy.interval; }
  /** Whether a stage may leave the collar out now: quiet, and the route can rest it. */
  [[nodiscard]] bool rests() const noexcept { return state_.quiet && config_.can_rest; }

private:
  void commit(double z);
  void read_cap(const double* reading, std::size_t size);
  void settle_from(bool light, const double* shares, double raw_collar);
  [[nodiscard]] Judgement judge(const double* shares, double z, bool launch);
  void restore_quiet();
  void plane_row(double z);

  Config config_;
  State state_;
  double launch_ = std::numeric_limits<double>::quiet_NaN();
  double floor_ = 0.0;
  std::array<double, 4> floor_regions_{};
  double raw_guard_max_ = 0.0;
  double raw_removed_ = 0.0;
  bool has_raw_last_ = false;
  double raw_last_z_ = 0.0;
  double raw_last_rate_ = 0.0;
  double floor_removed_ = 0.0;
  std::vector<std::array<double, 2>> unwatched_;
  bool has_restored_quiet_ = false;
  std::uint64_t restored_quiet_run_ = 0;
  bool restored_quiet_ = false;
  std::array<double, 2> guard_max_{};
  std::array<double, 2> guard_at_{std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN()};
  double collar_max_ = 0.0;
  std::array<double, 3> launch_occupancy_{std::numeric_limits<double>::quiet_NaN(),
                                          std::numeric_limits<double>::quiet_NaN(),
                                          std::numeric_limits<double>::quiet_NaN()};
  double collar_now_ = 0.0;
  std::array<double, 2> since_{};
  bool read_since_plane_ = false;
  std::array<std::vector<double>, plane_columns> planes_;
  std::uint64_t steps_ = 0;
  std::uint64_t skipped_ = 0;
  bool left_out_ = false;
  std::uint64_t left_out_checks_ = 0;
  bool stale_ = false;
  double committed_ = std::numeric_limits<double>::quiet_NaN();
  double floor_taken_ = 0.0;
  bool resuming_ = false;
  bool resume_plane_ = false;
  std::vector<double> resumed_at_;
  Judgement last_;
};

/** A march's check of the step just committed at `z` under an owning monitor (SEP 47
    section 6.4): skip it, or take `read()`'s reading and check it, as `_monitor_step`
    and `_check_at` do; a replay whose longest step before the next check passes the
    cap this check read is refused (`_refuse_scheduled_crossing`) where the check
    neither stopped nor warned. A warning's refusal is the caller's, after the warning
    is serviced. */
template <class Reader>
[[nodiscard]] Verdict owned_check(Monitor& monitor, Reader&& read, const double z,
                                  const bool replay, const std::vector<double>& steps,
                                  const std::size_t cursor) {
  if (!monitor.due(z)) {
    monitor.skip(z);
    return Verdict::none;
  }
  const std::vector<double> reading = read();
  const bool rested = monitor.rests();
  Verdict verdict = monitor.step(reading.data(), reading.size(), z).verdict;
  // The stages change only where the collar's rest does.
  if (verdict == Verdict::none && monitor.rests() != rested)
    verdict = Verdict::quiet;
  if ((verdict == Verdict::none || verdict == Verdict::quiet) &&
      refused_crossing(monitor.config().policy.crossing, replay, steps, cursor,
                       monitor.steps_to_check(), monitor.cap()) > 0.0)
    verdict = Verdict::refuse;
  return verdict;
}

/** The strictness test's witness, as this translation unit rounds `a b + c`; never
    inlined, as `march::strict_probe`. */
[[nodiscard, gnu::noinline]] double strict_probe(double a, double b, double c);

} // namespace solprop_native::watch
