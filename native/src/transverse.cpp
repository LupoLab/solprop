// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/transverse.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

// As in watch.cpp (solprop #135): the edge's decisions round as solprop's Python does,
// so this file takes no contraction (CMake) and the optimize attribute this pragma
// gives every function below keeps GCC from inlining one into a contracting caller
// under link-time optimisation.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace solprop_native::transverse {

namespace {

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

/** Python's `max(a, b)`: the first unless the second is greater. */
double py_max(const double a, const double b) { return b > a ? b : a; }

/** Python's `min(a, b)`: the first unless the second is less. */
double py_min(const double a, const double b) { return b < a ? b : a; }

/** `np.maximum(a, b)` on one element: NaN in either is NaN. */
double np_maximum(const double a, const double b) {
  return (a >= b || std::isnan(a)) ? a : b;
}

/** Whether a status leaves the verdict to a warning (`in ("clear", "coarse")`). */
bool unjudged(const Status status) {
  return status == Status::clear || status == Status::coarse;
}

Notice notice_of(const Notice::Kind kind, const double value) {
  Notice notice;
  notice.kind = kind;
  notice.value = value;
  return notice;
}

} // namespace

double strict_probe(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}

Edge::Edge(Config config) : config_(std::move(config)) {
  n_faces_ = config_.faces.size();
  if (n_faces_ != 1 && n_faces_ != 2 && n_faces_ != 4)
    throw std::invalid_argument("a transverse edge has one, two or four faces");
  for (const auto& name : config_.faces)
    if (name.empty())
      throw std::invalid_argument("a transverse edge's face has a name");
  const std::size_t n = config_.columns.size();
  if (n == 0)
    throw std::invalid_argument("a transverse edge reads at least one column");
  if (config_.kappa.size() != n)
    throw std::invalid_argument("a transverse edge's wavenumbers are its columns'");
  if (config_.kappa_order.size() != n)
    throw std::invalid_argument("a transverse edge's order sorts its columns");
  std::vector<bool> seen(n, false);
  for (const std::size_t index : config_.kappa_order) {
    if (index >= n || seen[index])
      throw std::invalid_argument(
          "a transverse edge's order is a permutation of its columns");
    seen[index] = true;
  }
  if (config_.design ? config_.ramp_rate.size() != n : !config_.ramp_rate.empty())
    throw std::invalid_argument(
        "a transverse edge's ramp rate is on its columns, with a boundary alone");
  if (!config_.design && (config_.collar || config_.collar_rate))
    throw std::invalid_argument("a transverse edge's collar needs its boundary");
  if (!(config_.reads > 0.0) || !(config_.backstop_reads > 0.0) ||
      !(config_.noise_modes > 0.0))
    throw std::invalid_argument("a transverse edge's counts are positive");
  baseline_.assign(n_faces_, 0.0);
  peak_.assign(n_faces_, 0.0);
  peak_raw_.assign(n_faces_, 0.0);
}

std::vector<double> Edge::weighted(const std::vector<double>& columns) const {
  // `_column_power`: `self._columns * columns`, one rounding each.
  const std::size_t n = config_.columns.size();
  if (columns.size() != n)
    throw std::invalid_argument("a transverse reading's columns are the edge's");
  std::vector<double> out(n);
  for (std::size_t index = 0; index < n; ++index)
    out[index] = config_.columns[index] * columns[index];
  return out;
}

double Edge::collar_rate(const Reading& reading) const {
  // `_collar_rate`: the requested collar's removal rate, a share of the launch per m.
  if (!config_.design || !config_.collar_rate)
    return 0.0;
  if (!reading.totals)
    throw std::invalid_argument("a collar's transverse reading carries its totals");
  return (*reading.totals)[1] / launch_;
}

double Edge::reach(const std::vector<double>& columns) const {
  // `_reach`: the tail of the spectrum above each wavenumber, accumulated in order
  // from the top as `np.cumsum` accumulates it, and the highest wavenumber whose tail
  // holds more than the warning's share of the launch.
  const auto& order = config_.kappa_order;
  const std::size_t n = order.size();
  double kappa_q = 0.0;
  double tail = 0.0;
  for (std::size_t back = 0; back < n; ++back) {
    const std::size_t position = n - 1 - back;
    const std::size_t column = order[position];
    double value = columns[column];
    if (has_noise_)
      value = np_maximum(value - noise_columns_[column], 0.0);
    tail = back == 0 ? value : tail + value;
    if (tail / launch_ > config_.warn) {
      kappa_q = config_.kappa[column];
      break;
    }
  }
  const double k = config_.k_min;
  const double k_z = std::sqrt(py_max(k * k - kappa_q * kappa_q, 0.0));
  return kappa_q / py_max(k_z, config_.k_floor * k);
}

void Edge::respace(const std::vector<double>& columns, const bool launch) {
  // `_respace`: `min(B, 4 g / v_q)`, the backstop `B` doubling while the reach holds.
  const double found = reach(columns);
  const bool grown = found > reach_reference_ * (1.0 + config_.reach_growth);
  if (launch || grown || found == 0.0) {
    backstop_ = config_.length / config_.reads;
    reach_reference_ = found;
  } else {
    backstop_ = py_min(2.0 * backstop_, config_.length / config_.backstop_reads);
  }
  double spacing = backstop_;
  if (found > 0.0)
    spacing = py_min(spacing, 4.0 * config_.guard / found);
  spacing_now_ = spacing;
}

std::array<double, 2> Edge::strip_shares(const std::vector<double>& strip) const {
  // `_strip_shares`: the larger line's inward share and its windowed share.
  if (strip.size() % 2 != 0)
    throw std::invalid_argument("a collar's strip is an inward and a windowed energy "
                                "per line");
  double best = 0.0;
  double windowed = 0.0;
  for (std::size_t line = 0; line < strip.size() / 2; ++line) {
    const double share = strip[2 * line] / launch_;
    if (share > best) {
      best = share;
      windowed = strip[2 * line + 1] / launch_;
    }
  }
  return {best, windowed};
}

LaunchSums Edge::launch_sums(const std::vector<double>& columns) const {
  // The launch's energy in the safety ramp's band, a reading: in order here, where
  // NumPy sums pairwise.
  if (!config_.design)
    throw std::logic_error("a transverse edge without a boundary has no ramp's band");
  LaunchSums sums;
  const auto power = weighted(columns);
  for (std::size_t index = 0; index < power.size(); ++index) {
    sums.total += power[index];
    if (config_.ramp_rate[index] > 0.0)
      sums.ramp += power[index];
  }
  return sums;
}

ReadSums Edge::read_sums(const double gap) const {
  // The ramp's removal over the interval from each column's power at its start, a
  // reading: the C library's `expm1`, summed in order.
  if (!config_.design || !has_last_columns_)
    throw std::logic_error("a transverse edge's ramp reads the columns it last read");
  ReadSums sums;
  const std::size_t n = config_.columns.size();
  for (std::size_t index = 0; index < n; ++index) {
    const double exponent = -config_.ramp_rate[index] * gap;
    const double taken = -std::expm1(exponent);
    sums.raw += last_columns_[index] * taken;
    if (has_noise_)
      sums.noise += noise_columns_[index] * taken;
  }
  return sums;
}

void Edge::resume(const Resumption& record) {
  if (!record.record)
    return;
  const auto fits = [this](const std::optional<std::vector<double>>& values) {
    return values.has_value() && values->size() == n_faces_;
  };
  const double launch = record.launch_energy.value_or(nan);
  if (!(launch > 0.0) || !fits(record.edge_baseline))
    return;
  resumed_ = true;
  launch_ = launch;
  baseline_ = *record.edge_baseline;
  if (fits(record.edge_max))
    peak_ = *record.edge_max;
  if (fits(record.edge_max_raw))
    peak_raw_ = *record.edge_max_raw;
  allowance_ = record.noise_allowance.value_or(0.0);
  ramp_raw_ = record.raw_ramp_removed.value_or(0.0);
  ramp_noise_ = record.noise_ramp_removed.value_or(0.0);
  noise_share_in_ramp_ = record.noise_share_in_ramp.value_or(0.0);
  declared_share_ = record.declared_share_in_ramp.value_or(0.0);
  spatial_removed_ = record.spatial_removed.value_or(0.0);
  current_ = record.current.value_or(0.0);
  current_baseline_ = record.current_baseline.value_or(0.0);
  strip_baseline_ = record.strip_baseline.value_or(0.0);
  strip_floor_ = record.strip_floor.value_or(0.0);
  strip_peak_ = record.strip_max.value_or(0.0);
  returned_peak_ = record.returned_max.value_or(0.0);
  reads_ = record.reads;
  largest_gap_ = record.largest_gap.value_or(0.0);
  warned_at_ = record.warned_at.value_or(nan);
  warned_ = std::isfinite(warned_at_);
  if (record.status) {
    const std::string& status = *record.status;
    if (status == "clear")
      status_ = Status::clear;
    else if (status == "warned")
      status_ = Status::warned;
    else if (status == "insensitive")
      status_ = Status::insensitive;
    else if (status == "coarse")
      status_ = Status::coarse;
  }
}

Notice Edge::launch(const Reading& reading, const double z, const LaunchSums* sums,
                    const Noise* noise) {
  stage_ = Stage::none;
  if (!reading.faces)
    throw std::invalid_argument("a transverse launch reads the faces and the columns");
  if (resumed_) {
    // The launch was judged by the first execution; its baselines and totals are
    // restored, and the reads continue from the resumed state.
    last_z_ = z;
    if (noise != nullptr) {
      noise_columns_ = weighted(noise->columns);
      has_noise_ = true;
    }
    last_columns_ = weighted(reading.columns);
    has_last_columns_ = true;
    respace(last_columns_, true);
    next_ = config_.reads_edge ? z + spacing_now_ : infinity;
    last_rate_ = collar_rate(reading);
    last_rate_z_ = z;
    has_last_rate_ = true;
    if (reading.current && launch_ > 0.0) {
      current_last_z_ = z;
      current_last_ = *reading.current / launch_;
      has_current_last_ = true;
    }
    if (!config_.reads_edge)
      status_ = Status::unwatched;
    return {};
  }
  if (!reading.clip || !reading.totals)
    throw std::invalid_argument("a transverse launch reads every node");
  launch_ = (*reading.totals)[0];
  last_z_ = z;
  if (!(launch_ > 0.0)) {
    next_ = infinity;
    return {};
  }
  held_ = reading;
  held_z_ = z;
  if (noise != nullptr)
    held_noise_ = *noise;
  else
    held_noise_.reset();
  // About the launch, not the backend: judged whether or not the edge is read.
  if (config_.design) {
    const LaunchSums band =
        sums != nullptr
            ? *sums
            : launch_sums(noise != nullptr && noise->clean ? *noise->clean
                                                           : reading.columns);
    const Notice notice = judge_ramp_band(band);
    if (notice.kind != Notice::Kind::none) {
      stage_ = notice.stop ? Stage::none : Stage::launch_unread;
      return notice;
    }
  }
  stage_ = Stage::launch_unread;
  return run_launch();
}

Notice Edge::judge_ramp_band(const LaunchSums& sums) {
  // `_judge_ramp_band`: the noise-free launch's share in the safety ramp's band.
  const double share = sums.total > 0.0 ? sums.ramp / sums.total : 0.0;
  if (config_.declared) {
    // Declared above discard_above: not wanted, so its removal is not judged; solprop
    // logs the share from the record.
    declared_share_ = share;
    return {};
  }
  if (share > config_.stop) {
    Notice notice = notice_of(Notice::Kind::ramp_band, share);
    notice.stop = true;
    return notice;
  }
  if (share > config_.warn)
    return notice_of(Notice::Kind::ramp_band, share);
  return {};
}

Notice Edge::run_launch() {
  // The launch's stages after its ramp band, in solprop's order; a warning keeps the
  // stage it arose in, and `proceed` resumes from the next.
  switch (stage_) {
  case Stage::launch_unread:
    if (!config_.reads_edge) {
      stage_ = Stage::none;
      status_ = Status::unwatched;
      reason_ = Reason::unwatched;
      return notice_of(Notice::Kind::unwatched, 0.0);
    }
    face_ = 0;
    stage_ = Stage::launch_clip;
    [[fallthrough]];
  case Stage::launch_clip:
    while (face_ < n_faces_) {
      const std::size_t face = face_++;
      baseline_[face] = (*held_.faces)[face] / launch_;
      const double clipped = (*held_.clip)[face] / launch_;
      if (clipped > config_.warn) {
        Notice notice = notice_of(Notice::Kind::clip, clipped);
        notice.face = face;
        return notice;
      }
    }
    stage_ = Stage::launch_noise;
    [[fallthrough]];
  case Stage::launch_noise:
    stage_ = Stage::launch_spacing;
    if (held_noise_) {
      const double eta = held_noise_->energy / launch_;
      allowance_ = config_.noise_factor * eta * config_.noise_share /
                   std::sqrt(config_.noise_modes);
      noise_columns_ = weighted(held_noise_->columns);
      has_noise_ = true;
      if (config_.design)
        noise_share_in_ramp_ = held_noise_->ramp / launch_;
      if (allowance_ > config_.warn) {
        status_ = Status::insensitive;
        reason_ = Reason::insensitive;
        return notice_of(Notice::Kind::insensitive, allowance_);
      }
    }
    [[fallthrough]];
  case Stage::launch_spacing: {
    stage_ = Stage::launch_rest;
    last_columns_ = weighted(held_.columns);
    has_last_columns_ = true;
    respace(last_columns_, true);
    const double spacing = spacing_now_;
    next_ = held_z_ + spacing;
    if (config_.longest_step && *config_.longest_step > spacing) {
      if (status_ == Status::clear)
        status_ = Status::coarse;
      largest_gap_ = *config_.longest_step;
      Notice notice = notice_of(Notice::Kind::coarse, *config_.longest_step);
      notice.spacing = spacing;
      return notice;
    }
  }
    [[fallthrough]];
  case Stage::launch_rest:
    stage_ = Stage::none;
    last_rate_ = collar_rate(held_);
    last_rate_z_ = held_z_;
    has_last_rate_ = true;
    if (every_step())
      launch_return();
    held_ = Reading{};
    held_noise_.reset();
    return {};
  default:
    stage_ = Stage::none;
    return {};
  }
}

void Edge::launch_return() {
  // `_launch_return`: a collar's launch baselines and floors.
  if (!held_.strip || !held_.totals)
    throw std::invalid_argument("a collar's launch reads its strip and its totals");
  strip_baseline_ = strip_shares(*held_.strip)[0];
  if (held_noise_)
    strip_floor_ = 3.0 * strip_shares(held_noise_->strip)[0];
  current_baseline_ = (*held_.totals)[2] / launch_;
  has_current_last_ = false;
  current_ = 0.0;
}

Notice Edge::read(const Reading& reading, const double z, const ReadSums* sums) {
  stage_ = Stage::none;
  if (!wants_read(z))
    return {};
  if (!reading.faces)
    throw std::invalid_argument("a transverse read reads the faces and the columns");
  if (!has_last_rate_)
    throw std::logic_error("a transverse edge is read before its launch set its rate");
  const double gap = z - last_z_;
  ++reads_;
  largest_gap_ = py_max(largest_gap_, gap);
  // The ramp's removal over the interval, from each column's power at its start.
  if (config_.design && has_last_columns_) {
    const ReadSums ramp = sums != nullptr ? *sums : read_sums(gap);
    ramp_raw_ = ramp_raw_ + ramp.raw / launch_;
    if (has_noise_)
      ramp_noise_ = ramp_noise_ + ramp.noise / launch_;
  }
  const double rate = collar_rate(reading);
  spatial_removed_ = spatial_removed_ + 0.5 * (z - last_rate_z_) * (last_rate_ + rate);
  last_rate_z_ = z;
  last_rate_ = rate;
  last_z_ = z;
  last_columns_ = weighted(reading.columns);
  has_last_columns_ = true;
  respace(last_columns_, false);
  next_ = z + spacing_now_;
  Notice notice;
  if (config_.design && config_.collar) {
    // With a collar, outward flow is intended: recorded, never stopped.
    for (std::size_t face = 0; face < n_faces_; ++face) {
      const double share = (*reading.faces)[face] / launch_;
      peak_raw_[face] = py_max(peak_raw_[face], share);
    }
    notice = judge_return(reading, z);
  } else {
    notice = judge_faces(reading, z);
  }
  if (notice.kind != Notice::Kind::none) {
    if (!notice.stop)
      stage_ = Stage::read_ramp;
    return notice;
  }
  return judge_ramp_removal();
}

Notice Edge::judge_faces(const Reading& reading, const double z) {
  // Each face's share above its launch value and the noise's allowance; the worst is
  // `max` of `(judged, name)` tuples, ties broken on the name.
  std::array<double, 4> judged{};
  for (std::size_t face = 0; face < n_faces_; ++face) {
    const double share = (*reading.faces)[face] / launch_;
    peak_raw_[face] = py_max(peak_raw_[face], share);
    judged[face] = share - baseline_[face] - allowance_;
    peak_[face] = py_max(peak_[face], judged[face]);
  }
  std::size_t worst = 0;
  for (std::size_t face = 1; face < n_faces_; ++face) {
    // `(a, m) > (b, n)`: the shares, unless they are equal, then the names.
    const bool greater = judged[face] == judged[worst]
                             ? config_.faces[face] > config_.faces[worst]
                             : judged[face] > judged[worst];
    if (greater)
      worst = face;
  }
  if (judged[worst] > config_.stop) {
    status_ = Status::stopped;
    Notice notice = notice_of(Notice::Kind::face, judged[worst]);
    notice.stop = true;
    notice.face = worst;
    notice.z = z;
    notice.warned_at = warned_at_;
    return notice;
  }
  for (std::size_t face = 0; face < n_faces_; ++face) {
    if (judged[face] > config_.warn && !warned_) {
      warned_ = true;
      warned_at_ = z;
      if (unjudged(status_))
        status_ = Status::warned;
      Notice notice = notice_of(Notice::Kind::face, judged[face]);
      notice.face = face;
      notice.z = z;
      return notice;
    }
  }
  return {};
}

Notice Edge::judge_return(const Reading& reading, const double z) {
  // `_judge_return`: what a collar returns, the strip and the current.
  if (!reading.strip)
    throw std::invalid_argument("a collar's transverse read carries its strip");
  const auto shares = strip_shares(*reading.strip);
  const double strip = shares[0];
  const double windowed = shares[1];
  strip_peak_ = py_max(strip_peak_, strip);
  const double judged_strip =
      py_max(strip - strip_baseline_ - strip_floor_ - 1e-6 * windowed, 0.0);
  const double judged_current = py_max(current_ - current_baseline_, 0.0);
  const double returned = py_max(judged_strip, judged_current);
  returned_peak_ = py_max(returned_peak_, returned);
  if (returned > config_.stop) {
    status_ = Status::stopped;
    Notice notice = notice_of(Notice::Kind::returned, returned);
    notice.stop = true;
    notice.z = z;
    return notice;
  }
  if (returned > config_.warn && !warned_) {
    warned_ = true;
    warned_at_ = z;
    status_ = Status::warned;
    Notice notice = notice_of(Notice::Kind::returned, returned);
    notice.z = z;
    return notice;
  }
  return {};
}

Notice Edge::judge_ramp_removal() {
  // `_judge_ramp_removal`: what the safety ramp has removed, at the edge's levels.
  const double removed = ramp_removed();
  if (removed > config_.stop) {
    status_ = Status::stopped;
    Notice notice = notice_of(Notice::Kind::ramp_removal, removed);
    notice.stop = true;
    return notice;
  }
  if (removed > config_.warn && !ramp_warned_) {
    ramp_warned_ = true;
    if (unjudged(status_))
      status_ = Status::warned;
    return notice_of(Notice::Kind::ramp_removal, removed);
  }
  return {};
}

Notice Edge::proceed() {
  switch (stage_) {
  case Stage::none:
    return {};
  case Stage::read_ramp:
    stage_ = Stage::none;
    return judge_ramp_removal();
  default:
    return run_launch();
  }
}

void Edge::step(const double current, const double z) {
  // `step`: a collar's inward current, integrated by the trapezoid rule.
  if (!config_.design || !config_.collar || !(launch_ > 0.0) ||
      status_ == Status::unwatched)
    return;
  const double share = current / launch_;
  if (has_current_last_)
    current_ = current_ + 0.5 * (z - current_last_z_) * (current_last_ + share);
  current_last_z_ = z;
  current_last_ = share;
  has_current_last_ = true;
}

double Edge::ceiling() const noexcept {
  return std::isfinite(next_) ? spacing_now_ : infinity;
}

bool Edge::due(const double z, const double proposal) const {
  return z + proposal >= next_ - 1e-12 * py_max(std::fabs(z), 1.0);
}

bool Edge::wants_read(const double z) const {
  return launch_ > 0.0 && status_ != Status::unwatched && z > last_z_;
}

double Edge::ramp_removed() const {
  return ramp_raw_ - py_min(ramp_noise_, noise_share_in_ramp_) - declared_share_;
}

Record Edge::record() const {
  Record out;
  out.launch_energy = launch_;
  out.noise_ramp_removed = ramp_noise_;
  out.noise_share_in_ramp = noise_share_in_ramp_;
  out.declared_share_in_ramp = declared_share_;
  out.status = status_;
  out.reason = reason_;
  out.edge_baseline = baseline_;
  out.edge_max = peak_;
  out.edge_max_raw = peak_raw_;
  out.noise_allowance = allowance_;
  out.ramp_removed = ramp_removed();
  out.raw_ramp_removed = ramp_raw_;
  out.spatial_removed = spatial_removed_;
  out.reads = reads_;
  out.returned_max = returned_peak_;
  out.strip_max = strip_peak_;
  out.strip_baseline = strip_baseline_;
  out.strip_floor = strip_floor_;
  out.current = current_;
  out.current_baseline = current_baseline_;
  out.largest_gap = largest_gap_;
  out.warned_at = warned_at_;
  out.resumed = resumed_;
  return out;
}

Finish Edge::finish() const {
  Finish out;
  out.status = status_;
  // `max(self._peak, default=0.0)`: the first, unless a later one is greater.
  if (!peak_.empty()) {
    out.largest = peak_[0];
    for (std::size_t face = 1; face < peak_.size(); ++face)
      if (peak_[face] > out.largest)
        out.largest = peak_[face];
  }
  out.ramp_removed = ramp_removed();
  out.reads = reads_;
  return out;
}

} // namespace solprop_native::transverse
