// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/watch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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

constexpr double nan = std::numeric_limits<double>::quiet_NaN();

/** Python's `max(a, b)`. */
double py_max(const double a, const double b) { return b > a ? b : a; }

/** Python's `min(a, b)`. */
double py_min(const double a, const double b) { return b < a ? b : a; }

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

/** `_found` then `_judged`: the energy and regions a committed step is judged on, or,
    without `found` (the launch), the reading's own. False where `_found` raises: the
    split found nothing where the state holds light. */
bool judged(const Policy& policy, const double* reading, const bool found_by_split,
            double& total, double* regions) {
  total = reading[0];
  for (std::size_t index = 0; index < 4; ++index)
    regions[index] = reading[1 + index];
  if (found_by_split && policy.split) {
    const double* found = reading + 6;
    if (!(found[0] > 0.0 || !(reading[0] > 0.0)))
      return false;
    total = found[0];
    regions[0] = found[1];
    regions[1] = found[2];
    regions[2] = found[3];
  }
  if (total <= 0.0 || !std::isfinite(total)) {
    total = 0.0;
    regions[0] = regions[1] = regions[2] = regions[3] = 0.0;
  }
  return true;
}

/** `_above(regions) / total`: each region's share above the floor's. */
void shares_of(const Policy& policy, const double total, const double* regions,
               double* shares) {
  for (std::size_t index = 0; index < 4; ++index) {
    const double over = regions[index] - policy.floor_regions[index];
    shares[index] = (index < 3 ? clip_low(over) : over) / total;
  }
}

/** `_judge`: the amplitude, compared with the amplitudes; a warning once. */
Verdict judge_shares(const Policy& policy, State& state, const double* shares,
                     double& worst) {
  worst = std::sqrt(py_max(py_max(shares[0], shares[1]), 0.0));
  if (worst > policy.stop)
    return Verdict::stop;
  if (worst > policy.warn && !state.warned) {
    state.warned = true;
    return Verdict::warn;
  }
  return Verdict::none;
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
  double total = 0.0;
  double regions[4];
  if (!judged(policy, reading, true, total, regions))
    return Verdict::error;
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
  shares_of(policy, total, regions, shares);
  settle(policy, state, true, shares, regions[2] / total);
  double worst = 0.0;
  const Verdict verdict = judge_shares(policy, state, shares, worst);
  if (verdict != Verdict::none)
    return verdict;
  return state.quiet != was_quiet ? Verdict::quiet : Verdict::none;
}

double refused_crossing(const bool crossing, const bool replay,
                        const std::vector<double>& steps, const std::size_t cursor,
                        const std::uint64_t ahead, const double cap) {
  if (!crossing || !replay || cursor >= steps.size())
    return 0.0;
  const double longest = longest_ahead(steps, cursor, ahead);
  return longest > cap ? longest : 0.0;
}

double longest_ahead(const std::vector<double>& steps, const std::size_t cursor,
                     const std::uint64_t ahead) {
  const std::size_t n_steps = steps.size();
  const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(
      std::max<std::uint64_t>(ahead, 1), n_steps - std::min(cursor, n_steps)));
  double longest = steps[cursor];
  for (std::size_t index = 1; index < count; ++index)
    longest = std::max(longest, steps[cursor + index]);
  return longest;
}

// ------------------------------------------------------------------ the owning monitor

Monitor::Monitor(Config config) : config_(std::move(config)) {
  auto& policy = config_.policy;
  // The floor regions and the launch floor are formed at the launch.
  for (double& value : policy.floor_regions)
    value = 0.0;
  policy.launch_floor = 0.0;
  validate(policy);
  if (!(config_.floor_share >= 0.0) || !(config_.floor_margin >= 0.0))
    refuse("floor share or margin");
  if (!(config_.budget > 0.0))
    refuse("budget");
  if (config_.rate.empty())
    refuse("collar rate: the coarse axis has no samples");
  // `_busy` starts true: every step is checked until the launch says otherwise.
  state_.busy = true;
}

std::size_t Monitor::reading_size() const noexcept {
  return config_.policy.crossing ? reading_head + buckets : reading_head;
}

void Monitor::read_cap(const double* reading, const std::size_t size) {
  if (size < reading_size())
    throw std::invalid_argument("a boundary reading is too short for its policy");
  // `_read_cap`: the split collar's crossing limit from the state's occupancy.
  if (config_.policy.crossing)
    state_.cap = split_limit(config_.policy, reading + reading_head);
}

void Monitor::commit(const double z) {
  // `_commit`: a position behind the last committed one adds nothing.
  const double step = z - committed_;
  if (!(step > 0.0))
    return;
  committed_ = z;
  if (config_.policy.split && floor_ > 0.0) {
    // The floor's share of the split's removal: a reading's sum, in order.
    double taken = 0.0;
    for (const double rate : config_.rate)
      taken += -std::expm1(-rate * step);
    floor_taken_ += floor_ / static_cast<double>(config_.rate.size()) * taken;
  }
}

void Monitor::settle_from(const bool light, const double* shares,
                          const double raw_collar) {
  settle(config_.policy, state_, light, shares, raw_collar);
}

Judgement Monitor::judge(const double* shares, const double z, const bool launch) {
  Judgement judgement;
  judgement.z = z;
  judgement.launch = launch;
  judgement.verdict = judge_shares(config_.policy, state_, shares, judgement.worst);
  last_ = judgement;
  return judgement;
}

void Monitor::restore_quiet() {
  // `_restore_quiet`: the resumed state's count already holds its plane's check.
  if (!has_restored_quiet_)
    return;
  state_.quiet_run = restored_quiet_run_;
  state_.quiet = config_.policy.quiet_collar && restored_quiet_;
  has_restored_quiet_ = false;
}

void Monitor::resume(const Resumption& record, const bool plane,
                     const std::vector<std::array<double, 2>>& unwatched) {
  resuming_ = true;
  resume_plane_ = plane;
  if (record.record && record.unwatched)
    unwatched_ = *record.unwatched;
  for (const auto& stretch : unwatched)
    if (stretch[1] > stretch[0])
      unwatched_.push_back(stretch);
  if (!record.record)
    return;
  const double launch = record.launch_energy.value_or(nan);
  if (!(launch > 0.0))
    return;
  launch_ = launch;
  config_.policy.launch_floor = config_.floor_share * launch_;
  launch_occupancy_ = {record.launch_guard_low.value_or(nan),
                       record.launch_guard_high.value_or(nan),
                       record.launch_collar.value_or(nan)};
  guard_max_ = {record.guard_low_max.value_or(0.0),
                record.guard_high_max.value_or(0.0)};
  guard_at_ = {record.guard_low_at.value_or(nan), record.guard_high_at.value_or(nan)};
  collar_max_ = record.collar_max.value_or(0.0);
  raw_removed_ = record.raw_removed_fraction.value_or(0.0);
  // The floor's expected removal, uncapped; a record from before it was kept gives it
  // back from the signed or the reported removal.
  double judged_removal = record.removed_signed.value_or(nan);
  if (!std::isfinite(judged_removal))
    judged_removal = record.removed_fraction.value_or(0.0);
  floor_removed_ = record.floor_removed_fraction ? *record.floor_removed_fraction
                                                 : raw_removed_ - judged_removal;
  const double quiet_checks = record.quiet_checks.value_or(nan);
  const auto count = [](const double value) {
    // Python's `int(float)`, which refuses what is not a whole count.
    if (!std::isfinite(value) || std::trunc(value) < 0.0)
      throw std::invalid_argument("a resumed boundary record holds an invalid count");
    return static_cast<std::uint64_t>(std::trunc(value));
  };
  if (std::isfinite(quiet_checks)) {
    has_restored_quiet_ = true;
    restored_quiet_run_ = count(quiet_checks);
    restored_quiet_ = record.collar_quiet;
  }
  raw_guard_max_ = record.raw_guard_max.value_or(0.0);
  steps_ = count(record.steps.value_or(0.0));
  skipped_ = count(record.skipped.value_or(0.0));
  left_out_checks_ = count(record.collar_left_out_checks.value_or(0.0));
  if (record.resumed_at)
    resumed_at_ = *record.resumed_at;
  if (record.has_planes)
    planes_ = record.planes;
}

Judgement Monitor::launch(const double* reading, const std::size_t size, const double z,
                          const double floor, const double* floor_regions) {
  auto& policy = config_.policy;
  if (floor_regions != nullptr) {
    // The input noise's floor, read from its draw.
    floor_ = floor;
    for (std::size_t index = 0; index < 4; ++index) {
      floor_regions_[index] = floor_regions[index];
      // `_floor_regions * [FLOOR_MARGIN] * 3 + [1]`, as `_above` forms it.
      policy.floor_regions[index] =
          floor_regions_[index] * (index < 3 ? config_.floor_margin : 1.0);
    }
  }
  committed_ = z;
  read_since_plane_ = true;
  read_cap(reading, size);
  Judgement judgement;
  judgement.z = z;
  judgement.launch = true;
  last_ = judgement;
  // The launch is judged on its own reading, not on what a split found (`_judged`).
  double total = 0.0;
  double regions[4];
  judged(policy, reading, false, total, regions);
  const bool resuming = resuming_;
  resuming_ = false;
  if (resuming)
    resumed_at_.push_back(z);
  if (!(resuming && launch_ > 0.0)) {
    // A resume restored its original launch energy, unless its record had none.
    launch_ = total;
    policy.launch_floor = config_.floor_share * launch_;
  }
  if (total <= 0.0) {
    state_.has_last = true;
    state_.last_z = z;
    has_raw_last_ = true;
    raw_last_z_ = z;
    raw_last_rate_ = 0.0;
    settle_from(false, nullptr, 0.0);
    restore_quiet();
    return judgement;
  }
  double shares[4];
  shares_of(policy, total, regions, shares);
  settle_from(true, shares, regions[2] / total);
  restore_quiet();
  state_.has_last = true;
  state_.last_z = z;
  has_raw_last_ = true;
  raw_last_z_ = z;
  raw_last_rate_ = regions[3] / launch_;
  collar_now_ = shares[2];
  since_ = {shares[0], shares[1]};
  if (resuming) {
    if (resume_plane_)
      plane_row(z);
    return judgement;
  }
  launch_occupancy_ = {shares[0], shares[1], shares[2]};
  plane_row(z);
  return judge(shares, z, true);
}

bool Monitor::due(const double z) const {
  return watch::due(config_.policy, state_, z);
}

void Monitor::skip(const double z) {
  commit(z);
  ++skipped_;
  watch::skip(state_);
  stale_ = true;
}

std::uint64_t Monitor::steps_to_check() const {
  return watch::steps_to_check(config_.policy, state_);
}

Judgement Monitor::step(const double* reading, const std::size_t size, const double z,
                        const bool plane) {
  const auto& policy = config_.policy;
  const bool rechecked = plane && stale_;
  commit(z);
  read_since_plane_ = true;
  read_cap(reading, size);
  Judgement judgement;
  judgement.z = z;
  last_ = judgement;
  double total = 0.0;
  double regions[4];
  if (!judged(policy, reading, true, total, regions)) {
    judgement.verdict = Verdict::error;
    last_ = judgement;
    return judgement;
  }
  if (!rechecked)
    ++steps_;
  left_out_checks_ += left_out_ ? 1 : 0;
  state_.since = 0;
  stale_ = false;
  if (policy.split && launch_ > 0.0) {
    // What the split took at every accepted step since the previous reading, checked
    // or not, and however little of the light is left.
    raw_removed_ += reading[5] / launch_;
    floor_removed_ += floor_taken_ / launch_;
  }
  floor_taken_ = 0.0;
  if (!(total > policy.launch_floor)) {
    // Nothing was launched, or nothing is left: no light to judge.
    state_.has_last = true;
    state_.last_z = z;
    has_raw_last_ = true;
    raw_last_z_ = z;
    raw_last_rate_ = 0.0;
    collar_now_ = 0.0;
    settle_from(false, nullptr, 0.0);
    return judgement;
  }
  if (launch_ == 0.0) {
    // Light after a launch that held none: solprop's monitor divides by the launch
    // energy here and raises ZeroDivisionError.
    judgement.verdict = Verdict::divide;
    last_ = judgement;
    return judgement;
  }
  // The whole field's record, floor included, kept beside the judged one; a resting
  // collar absorbed nothing since the last check, and a split's removal was read above.
  const double raw_rate = regions[3] / launch_;
  const double previous_z = has_raw_last_ ? raw_last_z_ : z;
  const double previous_rate = has_raw_last_ ? raw_last_rate_ : raw_rate;
  if (!(policy.split || left_out_)) {
    raw_removed_ += 0.5 * (z - previous_z) * (previous_rate + raw_rate);
    floor_removed_ += (z - previous_z) * floor_regions_[3] / launch_;
  }
  has_raw_last_ = true;
  raw_last_z_ = z;
  raw_last_rate_ = raw_rate;
  raw_guard_max_ = py_max(raw_guard_max_, py_max(regions[0], regions[1]) / total);
  double shares[4];
  shares_of(policy, total, regions, shares);
  settle_from(true, shares, regions[2] / total);
  state_.has_last = true;
  state_.last_z = z;
  collar_max_ = py_max(collar_max_, shares[2]);
  collar_now_ = shares[2];
  for (std::size_t edge = 0; edge < 2; ++edge) {
    since_[edge] = py_max(since_[edge], shares[edge]);
    if (shares[edge] > guard_max_[edge]) {
      guard_max_[edge] = shares[edge];
      guard_at_[edge] = z;
    }
  }
  return judge(shares, z, false);
}

double Monitor::removed() const {
  const double floor = launch_ > 0.0 ? floor_ / launch_ : 0.0;
  return raw_removed_ - py_min(floor_removed_, floor);
}

void Monitor::plane_row(const double z) {
  const bool read = read_since_plane_;
  read_since_plane_ = false;
  auto& rows = planes_;
  if (!rows[0].empty() && rows[0].back() == z) {
    // A held row (a resume landing on a recorded plane): the next row's guard maxima
    // still start here.
    since_ = {0.0, 0.0};
    return;
  }
  rows[0].push_back(z);
  // A plane no check read since the previous one has no occupancies of its own.
  rows[1].push_back(read ? collar_now_ : nan);
  rows[2].push_back(read ? since_[0] : nan);
  rows[3].push_back(read ? since_[1] : nan);
  rows[4].push_back(py_max(removed(), 0.0));
  since_ = {0.0, 0.0};
}

void Monitor::plane(const double z) { plane_row(z); }

Finish Monitor::finish() const {
  Finish out;
  out.removed = py_max(removed(), 0.0);
  out.guard_max = guard_max_;
  out.warns = out.removed > 0.0 && std::sqrt(out.removed) > config_.budget;
  return out;
}

Record Monitor::record() const {
  Record out;
  out.launch_energy = launch_;
  out.resumed_at = resumed_at_;
  out.launch_occupancy = launch_occupancy_;
  out.guard_max = guard_max_;
  out.guard_at = guard_at_;
  out.collar_max = collar_max_;
  out.removed_signed = removed();
  out.removed_fraction = py_max(out.removed_signed, 0.0);
  out.floor_removed_fraction = floor_removed_;
  out.quiet_checks = state_.quiet_run;
  out.collar_quiet = state_.quiet;
  out.unwatched = unwatched_;
  out.steps = steps_;
  out.skipped = skipped_;
  out.noise_floor = launch_ > 0.0 ? floor_ / launch_ : 0.0;
  out.raw_guard_max = raw_guard_max_;
  out.raw_removed_fraction = raw_removed_;
  out.collar_left_out_checks = left_out_checks_;
  out.planes = planes_;
  return out;
}

double strict_probe(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}

} // namespace solprop_native::watch
