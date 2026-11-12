// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/session.hpp"

#include "solprop_native/core.hpp"
#include "solprop_native/refuse.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

// As in adams_march.cpp (solprop #135): the scheduler's comparisons and the export's
// recipe round as solprop's Python does, so this file takes no contraction (CMake) and
// the optimize attribute this pragma gives every function below keeps GCC from
// inlining one into a contracting caller under link-time optimisation.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace solprop_native::session {

namespace {

/** Python's `max(a, b)`. */
double py_max(const double a, const double b) { return b > a ? b : a; }

/** Whether a march call's return is an event on which the driver has work: its last
    accepted row is then the event step (`_NativeAdamsMarch._call`). A halt is the
    driver's halt latch, which returns at an accepted endpoint as a host step does. */
bool event_reason(const march::Reason reason) {
  return reason == march::Reason::reached || reason == march::Reason::host_step ||
         reason == march::Reason::monitor || reason == march::Reason::watch ||
         reason == march::Reason::halted;
}

/** A failure's record from a march call's error return. */
Failure failure_of(const march::Result& result, const std::uint64_t cursor) {
  Failure failure;
  failure.reason = result.reason;
  failure.category = result.category;
  failure.stage = result.stage;
  failure.message = result.message;
  failure.floor_proposal = result.floor_proposal;
  failure.cost = result.cost;
  failure.replay_dz = result.replay_dz;
  failure.replay_end = result.replay_end;
  failure.replay_norm = result.replay_norm;
  failure.cursor = cursor;
  return failure;
}

/** A failure outside a march call: an exception the session caught, its solprop
    error class as the marches map it (`rk_march.cpp`, `adams_march.cpp`). */
Failure failure_of(const std::exception& error) {
  Failure failure;
  failure.reason = march::Reason::error;
  failure.message = error.what();
  if (dynamic_cast<const EnvelopeConvergenceError*>(&error) != nullptr)
    failure.category = march::Category::convergence;
  else if (dynamic_cast<const BackendRefusal*>(&error) != nullptr)
    failure.category = march::Category::refused;
  else if (dynamic_cast<const std::invalid_argument*>(&error) != nullptr)
    failure.category = march::Category::value;
  else
    failure.category = march::Category::backend;
  return failure;
}

} // namespace

bool plane_reached(const double position, const double target) {
  // `abs(x - y) <= atol + rtol * abs(y)`, with `atol = 0.0` and `rtol = 1e-12`.
  const double difference = std::fabs(position - target);
  const double scaled = 1e-12 * std::fabs(target);
  const double allowed = 0.0 + scaled;
  return difference <= allowed;
}

double replay_tolerance(const std::size_t steps, const double length,
                        const double position, const int landing_ulps) {
  // `max(LANDING_ULPS, schedule.n_steps) * spacing(max(total_length, abs(position)))`.
  const std::size_t ulps = std::max(static_cast<std::size_t>(landing_ulps), steps);
  return static_cast<double>(ulps) * march::ulp(py_max(length, std::fabs(position)));
}

std::uint64_t steps_to_next(const std::uint64_t period, const std::uint64_t count) {
  return period - count % period;
}

namespace {

// One row of `n_omega` bins of `apply_recipe`, its two choices fixed so that the loop
// carries no branch and no division: `(-1)^k` alternates along the row.
template <bool Scale, bool Optics>
void recipe_row(const double* state, const std::size_t n_omega, const double* weight,
                double* out) {
  for (std::size_t bin = 0; bin < n_omega; ++bin) {
    double re = state[2 * bin];
    double im = state[2 * bin + 1];
    if constexpr (Scale) {
      const double root = std::sqrt(weight[bin]);
      const double real_part = re * root - im * 0.0;
      const double imaginary_part = re * 0.0 + im * root;
      re = real_part;
      im = imaginary_part;
    }
    if constexpr (Optics) {
      const double conjugate = -im;
      const double sign = bin % 2 == 0 ? 1.0 : -1.0;
      const double real_part = sign * re - 0.0 * conjugate;
      const double imaginary_part = sign * conjugate + 0.0 * re;
      re = real_part;
      im = imaginary_part;
    }
    out[2 * bin] = re;
    out[2 * bin + 1] = im;
  }
}

template <bool Scale, bool Optics>
void recipe_rows(const double* state, const std::size_t n, const std::size_t n_omega,
                 const double* weight, double* out) {
  for (std::size_t start = 0; start < n; start += n_omega)
    recipe_row<Scale, Optics>(state + 2 * start, n_omega,
                              Scale ? weight + start : nullptr, out + 2 * start);
}

} // namespace

void apply_recipe(const double* state, const std::size_t n, const std::size_t n_omega,
                  const double* weight, const bool scale, const bool optics,
                  double* out) {
  // `np.multiply(state, np.sqrt(weight))` multiplies by the complex `sqrt(w) + 0j`, and
  // `np.multiply(shift, np.conj(out))` by the complex `(-1)^k + 0j`: each product is
  // NumPy's complex one, `(a c - b d, a d + b c)`, its zero terms kept so that signed
  // zeros and non-finite values come out as NumPy's. The field is whole rows of
  // `n_omega` bins.
  if (scale && optics)
    recipe_rows<true, true>(state, n, n_omega, weight, out);
  else if (scale)
    recipe_rows<true, false>(state, n, n_omega, weight, out);
  else if (optics)
    recipe_rows<false, true>(state, n, n_omega, weight, out);
  else
    recipe_rows<false, false>(state, n, n_omega, weight, out);
}

double strict_probe(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}

Session::Session(std::unique_ptr<Resident> resident, Method method, Plan plan,
                 const double z)
    : resident_(std::move(resident)), method_(std::move(method)),
      plan_(std::move(plan)), clock_(Clock::now()) {
  if (!resident_)
    throw std::invalid_argument("a session needs the state it leases");
  elements_ = resident_->elements();
  adams_ = std::holds_alternative<AdamsMethod>(method_);
  if (plan_.monitor) {
    monitor_ = std::make_unique<watch::Monitor>(plan_.monitor->config);
    if (plan_.monitor->edge)
      edge_ = std::make_unique<transverse::Edge>(*plan_.monitor->edge);
    // A resting collar caps the steps at the monitor's interval, as the march's policy
    // takes it from solprop's monitor (`quiet_interval`).
    std::visit(
        [this](auto& marching) {
          march::Policy* landings = nullptr;
          if constexpr (std::is_same_v<std::decay_t<decltype(marching)>, AdamsMethod>)
            landings = &marching.policy;
          else
            landings = &marching.policy.landings;
          landings->has_quiet_interval = true;
          landings->quiet_interval = monitor_->interval();
        },
        method_);
  }
  if (adams_) {
    auto& adams = std::get<AdamsMethod>(method_);
    if (!adams.engine)
      throw std::invalid_argument("a session's Adams march needs its engine");
    march::validate(adams.policy);
    landing_ulps_ = adams.policy.landing_ulps;
    if (adams.state.z != z)
      throw std::invalid_argument("the Adams march's state is not at the start");
    if (adams.policy.interpolate != (adams.dense != nullptr))
      throw std::invalid_argument(
          "a session interpolates its planes with the march's dense output");
    dense_ = adams.dense.get();
  } else {
    auto& runge_kutta = std::get<RungeKuttaMethod>(method_);
    if (!runge_kutta.engine)
      throw std::invalid_argument("a session's Runge-Kutta march needs its engine");
    landing_ulps_ = runge_kutta.policy.landings.landing_ulps;
    if (runge_kutta.state.z != z)
      throw std::invalid_argument("the Runge-Kutta march's state is not at the start");
    if (runge_kutta.policy.landings.interpolate != (runge_kutta.dense != nullptr))
      throw std::invalid_argument(
          "a session interpolates its planes with the march's dense output");
    dense_ = runge_kutta.dense.get();
  }
  if (!std::ranges::is_sorted(plan_.planes))
    throw std::invalid_argument("a session's save planes increase");
  if (plan_.next_plane > plan_.planes.size())
    throw std::invalid_argument("a session's first plane is past its planes");
  if (plan_.period > 0 && plan_.row_statistics.empty())
    throw std::invalid_argument("periodic rows need their statistics");
  if (plan_.call_attempts < 1 || !(plan_.call_seconds > 0.0))
    throw std::invalid_argument("a march call needs at least one attempt and time");
  if (plan_.attempt_capacity < 1 || plan_.row_capacity < 1)
    throw std::invalid_argument("a session's chunk buffers hold at least one entry");
  if (plan_.replay && plan_.replay_steps == 0)
    throw std::invalid_argument("a replayed schedule has steps");
  if (plan_.replay && dense_ != nullptr)
    throw std::invalid_argument("a replayed schedule lands on its planes");
  next_plane_ = plan_.next_plane;
  launch_pending_ = plan_.launch_pending;
  history_.reserve(attempt_columns * plan_.attempt_capacity);
  if (plan_.period > 0) {
    row_z_.reserve(plan_.row_capacity);
    row_step_.reserve(plan_.row_capacity);
  }
  published_z_.store(z, std::memory_order_relaxed);
  published_plane_.store(next_plane_, std::memory_order_relaxed);
}

void Session::require_open() const {
  if (closed())
    throw LifecycleError("the session is closed");
}

double Session::position() const noexcept {
  if (const auto* adams = std::get_if<AdamsMethod>(&method_))
    return adams->state.z;
  return std::get_if<RungeKuttaMethod>(&method_)->state.z;
}

double Session::here() const noexcept {
  // The observed position: a dense plane's while the state holds it, the committed
  // endpoint's otherwise (`DriverSession._z`).
  return observed_ ? observed_z_ : position();
}

std::uint64_t Session::cursor() const {
  return std::visit(
      [](const auto& method) {
        return static_cast<std::uint64_t>(method.state.cursor);
      },
      method_);
}

bool Session::snapshot_latched() const noexcept {
  return (latch_.load(std::memory_order_acquire) & 1U) != 0;
}

bool Session::poisoned() const noexcept {
  // An in-place step abandoned partway (solprop SEP 47 section 5.5): no state stands.
  const auto* runge_kutta = std::get_if<RungeKuttaMethod>(&method_);
  return runge_kutta != nullptr && runge_kutta->engine &&
         runge_kutta->engine->in_flight();
}

std::uint64_t Session::host_step() const {
  // The accepted steps to the driver's next work, the least of them: a periodic row
  // and a step plane (each march's `_bounds`). A latched snapshot is the halt flag,
  // which the march reads at every accepted endpoint.
  std::uint64_t steps = 0;
  const auto take = [&steps](const std::uint64_t value) {
    steps = steps == 0 ? value : std::min(steps, value);
  };
  if (plan_.period > 0)
    take(steps_to_next(plan_.period, accepted_steps_));
  if (plan_.step_planes > 0)
    take(steps_to_next(plan_.step_planes, natural_count_));
  // A requested collar's current is taken at every committed step, in the session.
  if (edge_ && edge_->every_step())
    take(1);
  return steps;
}

bool Session::at_endpoint() const {
  // An accepted endpoint this session reached, with no attempt made since: the
  // starting position holds no history to snapshot, and a rejection's trial is
  // past. A dense observation holds the state between endpoints: a latched snapshot
  // waits for the next endpoint, every plane its step crossed observed first (SEP 47
  // section 5.5).
  return stepped_ && last_accepted_ && !observed_;
}

std::size_t Session::chunk_attempts() const noexcept {
  return history_.size() / attempt_columns;
}

bool Session::chunk_full() const noexcept {
  // Room for one more call: its attempts are bounded by what is left, and it takes
  // at most one row, at its event step.
  return chunk_attempts() >= plan_.attempt_capacity ||
         (plan_.period > 0 && row_z_.size() >= plan_.row_capacity);
}

march::Result Session::call(const double target, const double slack,
                            const std::size_t attempts, const double seconds) {
  march::Bounds bounds;
  bounds.target = target;
  bounds.host_step = host_step();
  // The session's own monitor takes each committed step's check in the march, its cap
  // the split collar's. The transverse edge's spacing caps every adaptive step, and the
  // march returns to the session at the committed position from which the next step
  // would carry past the next read (`TransverseEdge.ceiling`, `.due`); infinite
  // without an edge.
  bounds.read_spacing = std::numeric_limits<double>::infinity();
  if (edge_) {
    bounds.read_at = edge_->next_read();
    bounds.has_read_at = std::isfinite(bounds.read_at);
    bounds.read_spacing = edge_->ceiling();
  }
  if (monitor_) {
    bounds.watch = true;
    bounds.owner = monitor_.get();
    bounds.quiet = collar_quiet_;
  }
  bounds.max_attempts = attempts;
  bounds.max_seconds = seconds;
  bounds.replay = plan_.replay;
  bounds.slack = slack;
  bounds.stop = &stop_;
  bounds.halt = &halt_;
  // A Runge-Kutta step across an interpolated plane holds its extension and is
  // finished in the march, which returns at its endpoint as the Adams march does.
  bounds.extend = dense_ != nullptr && !adams_;
  march_calls_.fetch_add(1, std::memory_order_relaxed);
  if (adams_) {
    auto& adams = std::get<AdamsMethod>(method_);
    return march::run(adams.policy, adams.state, *adams.engine, bounds, adams.schedule);
  }
  auto& runge_kutta = std::get<RungeKuttaMethod>(method_);
  return rk::run(runge_kutta.policy, runge_kutta.state, *runge_kutta.engine, bounds,
                 runge_kutta.schedule)
      .march;
}

void Session::publish() noexcept {
  published_z_.store(here(), std::memory_order_relaxed);
  published_plane_.store(next_plane_, std::memory_order_relaxed);
  published_attempts_.store(attempts_, std::memory_order_relaxed);
  published_rejected_.store(rejected_, std::memory_order_relaxed);
  published_generation_.store(generation_, std::memory_order_relaxed);
}

void Session::absorb(const march::Result& result, const Clock::time_point started) {
  // `_record_attempt`: each row with the wall time since the previous observation.
  const double offset = std::chrono::duration<double>(started - clock_).count();
  const std::size_t n_rows = result.rows.size() / march::row_columns;
  for (std::size_t row = 0; row < n_rows; ++row) {
    const double* values = result.rows.data() + row * march::row_columns;
    const bool accepted = values[march::accepted_column] != 0.0;
    history_.insert(history_.end(),
                    {values[march::end_column], values[march::step_column],
                     values[march::norm_column], accepted ? 1.0 : 0.0,
                     offset + values[march::wall_column]});
    ++attempts_;
    if (!accepted)
      ++rejected_;
    last_accepted_ = accepted;
    if (accepted) {
      stepped_ = true;
      ++generation_;
    }
  }
  // `_synchronise`: the call ledger's orders, one per accepted step.
  if (auto* adams = std::get_if<AdamsMethod>(&method_))
    for (const std::size_t order : result.orders)
      adams->orders.push_back(static_cast<std::uint8_t>(order));
  publish();
}

void Session::take_row() {
  // `_record_step_stats`: the row of the step just committed, its attempt index the
  // one `_record` has just appended, counted in the chunk: `-1` when a warning barrier
  // delivered that attempt in the chunk before (a row the barrier's continuation
  // takes), so that the assembled history indexes it exactly once.
  const double z = position();
  const auto values = resident_->statistics(plan_.row_statistics, z);
  if (row_width_ == 0)
    row_width_ = values.size();
  else if (values.size() != row_width_)
    throw std::logic_error("a periodic row changed its width");
  if (row_values_.capacity() == 0)
    row_values_.reserve(plan_.row_capacity * row_width_);
  row_z_.push_back(z);
  row_step_.push_back(static_cast<std::int64_t>(chunk_attempts()) - 1);
  row_values_.insert(row_values_.end(), values.begin(), values.end());
  rows_taken_.fetch_add(1, std::memory_order_relaxed);
  published_width_.store(row_width_, std::memory_order_relaxed);
}

void Session::drain(Batch& batch) {
  // Exactly-once delivery: the chunk moves into the batch, and the buffers start
  // again at their bounded capacity.
  batch.attempts = std::move(history_);
  // Each chunk's wall times start at the previous event (`join_attempts` offsets
  // them by the chunks before).
  clock_ = Clock::now();
  history_ = {};
  history_.reserve(attempt_columns * plan_.attempt_capacity);
  batch.row_width = row_width_;
  batch.row_z = std::move(row_z_);
  batch.row_step = std::move(row_step_);
  batch.row_values = std::move(row_values_);
  row_z_ = {};
  row_step_ = {};
  row_values_ = {};
  if (plan_.period > 0) {
    row_z_.reserve(plan_.row_capacity);
    row_step_.reserve(plan_.row_capacity);
    row_values_.reserve(plan_.row_capacity * row_width_);
  }
}

Event Session::observe(Batch& batch, const Observation kind, const std::int64_t plane) {
  Event event;
  event.kind = EventKind::observation;
  event.observation = kind;
  event.z = here();
  event.plane_index = plane;
  event.ticket = true;
  event.generation = generation_;
  if (!plan_.plane_statistics.empty()) {
    batch.plane_stats = resident_->statistics(plan_.plane_statistics, event.z);
    event.has_stats = true;
  }
  drain(batch);
  return event;
}

void Session::progress(Batch& batch) {
  Event event;
  event.kind = EventKind::progress;
  event.z = here();
  drain(batch);
  batch.event = std::move(event);
}

void Session::snapshot(Batch& batch) {
  // The committed endpoint, with the chunk since the previous event; the caller's
  // running checkpoint includes every chunk since the previous plane (SEP 47 5.5).
  latch_.fetch_and(~std::uint64_t{1}, std::memory_order_acq_rel);
  halt_.store(false, std::memory_order_release);
  Event event;
  event.kind = EventKind::snapshot;
  event.z = here();
  event.ticket = true;
  event.generation = generation_;
  drain(batch);
  batch.event = std::move(event);
}

void Session::terminal(Batch& batch, const Outcome outcome, Failure failure) {
  Event event;
  event.kind = EventKind::terminal;
  event.z = here();
  event.outcome = outcome;
  event.failure = std::move(failure);
  // A stop or a failure names the committed state, eligible, unless a step left in
  // flight poisoned it (`DriverSession._terminal`).
  event.ticket = outcome != Outcome::completed && !poisoned();
  event.generation = generation_;
  drain(batch);
  batch.event = std::move(event);
}

Batch Session::issue(Batch batch, const Outcome outcome) {
  batch.sequence = ++sequence_;
  pending_ = true;
  pending_generation_ = generation_;
  if (outcome != Outcome::none)
    outcome_.store(outcome, std::memory_order_release);
  state_.store(State::pending, std::memory_order_release);
  batches_.fetch_add(1, std::memory_order_relaxed);
  publish();
  return batch;
}

const march::Schedule& Session::schedule() const noexcept {
  if (const auto* adams = std::get_if<AdamsMethod>(&method_))
    return adams->schedule;
  return std::get_if<RungeKuttaMethod>(&method_)->schedule;
}

void Session::sync_collar() {
  // `_set_collar_quiet`: the stages rest the collar where the monitor finds it quiet
  // and the route can rest it; the monitor is told what the stages do.
  const bool rest = monitor_->rests();
  monitor_->note_collar(rest);
  if (rest != collar_quiet_) {
    collar_quiet_ = rest;
    resident_->set_collar_quiet(rest);
  }
}

void Session::warning(Batch& batch, const Notice& notice) {
  // A service barrier at the committed endpoint: its field can be exported for a
  // checkpoint while the caller emits the warning, which its filters may raise.
  Event event;
  event.kind = EventKind::warning;
  event.z = here();
  event.ticket = true;
  event.generation = generation_;
  event.notice = notice;
  drain(batch);
  batch.event = std::move(event);
}

void Session::monitor_stop(Batch& batch, const Notice& notice) {
  Failure failure;
  failure.reason = march::Reason::error;
  failure.category = notice.kind == Notice::Kind::guard ||
                             notice.kind == Notice::Kind::crossing ||
                             notice.kind == Notice::Kind::transverse
                         ? march::Category::refused
                         : march::Category::backend;
  failure.notice = notice;
  failure.cursor = cursor();
  terminal(batch, Outcome::failed, std::move(failure));
}

bool Session::judged(Batch& batch, const watch::Judgement& judgement) {
  // A check's verdict: a stop or an unreadable split ends the run here; a first
  // warning is a barrier, the caller's continuation set beforehand.
  Notice notice;
  notice.z = judgement.z;
  notice.launch = judgement.launch;
  notice.value = judgement.worst;
  switch (judgement.verdict) {
  case watch::Verdict::stop:
    notice.kind = Notice::Kind::guard;
    monitor_stop(batch, notice);
    return true;
  case watch::Verdict::error:
    notice.kind = Notice::Kind::reading;
    monitor_stop(batch, notice);
    return true;
  case watch::Verdict::divide:
    notice.kind = Notice::Kind::division;
    monitor_stop(batch, notice);
    return true;
  case watch::Verdict::warn:
    notice.kind = Notice::Kind::guard;
    warning(batch, notice);
    return true;
  default:
    return false;
  }
}

bool Session::refused_crossing(Batch& batch) {
  // `_refuse_scheduled_crossing`: a replay's longest step before the next check
  // against the cap the check just read.
  const double longest = watch::refused_crossing(
      monitor_->config().policy.crossing, plan_.replay, schedule().steps,
      static_cast<std::size_t>(cursor()), monitor_->steps_to_check(), monitor_->cap());
  if (!(longest > 0.0))
    return false;
  Notice notice;
  notice.kind = Notice::Kind::crossing;
  notice.z = here();
  notice.cap = monitor_->cap();
  notice.step = longest;
  monitor_stop(batch, notice);
  return true;
}

bool Session::attach(Batch& batch) {
  // `DriverSession._attach_monitor` after the launch plane's commit: restore a resumed
  // record, then record and judge the launch (`attach_monitor`).
  attached_ = true;
  const auto& plan = plan_.monitor.value();
  if (plan.resume) {
    // `BoundaryMonitor.resume` continues the edge's record first.
    if (edge_)
      edge_->resume(plan.edge_record);
    monitor_->resume(plan.record, plan.plane, plan.unwatched);
  }
  const std::vector<double> reading = resident_->boundary_reading();
  // The input noise's floor, read from its draw on the host once, or none.
  const watch::Judgement judgement =
      monitor_->launch(reading.data(), reading.size(), position(), plan.floor_energy,
                       plan.floor ? plan.floor_regions.data() : nullptr);
  if (judgement.verdict == watch::Verdict::warn)
    continuation_ = Continuation::attach;
  if (judged(batch, judgement))
    return true;
  sync_collar();
  // Then the transverse edge's launch (`attach_monitor`).
  return refused_crossing(batch) || (edge_ && edge_launch(batch));
}

bool Session::endpoint(Batch& batch, const double target, const bool edge) {
  // The event step after its monitor check: the transverse edge's read where due and
  // its collar's current, its statistics row, then the natural stop
  // (`_take_step_stats`, `_natural_stop`).
  if (edge && edge_ && edge_endpoint(batch))
    return true;
  if (plan_.period > 0) {
    ++accepted_steps_;
    if (accepted_steps_ % plan_.period == 0)
      take_row();
  }
  ++natural_count_;
  const bool step_plane =
      plan_.step_planes > 0 && natural_count_ % plan_.step_planes == 0;
  const double z = position();
  if (step_plane) {
    if (!march::reached(z, target, landing_ulps_))
      return observe_plane(batch, Observation::step,
                           static_cast<std::int64_t>(next_plane_), false);
    // A step that crossed an interpolated target is observed after it, once the
    // planes it crossed are (`_natural_stop`).
    if (dense_ != nullptr)
      natural_due_ = !march::reached(target, z, landing_ulps_);
  } else if (!march::reached(z, target, landing_ulps_) && snapshot_latched()) {
    // `_halted_short`: the latch halts the march at this endpoint.
    snapshot(batch);
    return true;
  }
  return false;
}

bool Session::observe_plane(Batch& batch, const Observation kind,
                            const std::int64_t plane, const bool next) {
  // `_monitor_plane` before the observation: a plane's state is checked even when its
  // last step was not (SEP 47 section 5.3, the requested plane).
  continued_kind_ = kind;
  continued_plane_ = plane;
  continued_next_ = next;
  if (monitor_ && monitor_->stale()) {
    const std::vector<double> reading = resident_->boundary_reading();
    const watch::Judgement judgement =
        monitor_->step(reading.data(), reading.size(), here(), true);
    sync_collar();
    if (judgement.verdict == watch::Verdict::warn)
      continuation_ = Continuation::plane;
    if (judged(batch, judgement))
      return true;
    if (refused_crossing(batch))
      return true;
  }
  // The transverse edge reads every plane it has not read (`_read_transverse`).
  if (edge_ && edge_plane(batch))
    return true;
  return plane_rest(batch);
}

bool Session::plane_rest(Batch& batch) {
  // The plane's monitor row, then its statistics and the observation.
  if (monitor_)
    monitor_->plane(here());
  if (continued_next_)
    ++next_plane_;
  batch.event = observe(batch, continued_kind_, continued_plane_);
  return true;
}

bool Session::observe_inside(Batch& batch, const double target) {
  // `_observe_inside` and the Adams march's observation: the state becomes the
  // crossing step's interpolated field at the plane, the committed state held aside,
  // and the plane is observed there (SEP 47 section 5.3).
  dense_->observe(target);
  observed_ = true;
  observed_z_ = target;
  if (auto* adams = std::get_if<AdamsMethod>(&method_))
    ++adams->state.ledger.dense_observations;
  publish();
  return observe_plane(batch, Observation::plane,
                       static_cast<std::int64_t>(next_plane_), true);
}

Batch Session::advance(const std::size_t work_limit, const double deadline) {
  const std::scoped_lock lock(mutex_);
  require_open();
  if (state_.load() == State::pending)
    throw LifecycleError(
        "acknowledge the pending batch before advancing the session again");
  if (state_.load() == State::finished)
    throw LifecycleError("the session has ended and cannot advance");
  const auto started = Clock::now();
  const auto elapsed = [&started] {
    return std::chrono::duration<double>(Clock::now() - started).count();
  };
  const std::size_t budget = std::max<std::size_t>(work_limit, 1);
  const std::uint64_t before = attempts_;
  Batch batch;
  const auto made = [&] { return static_cast<std::size_t>(attempts_ - before); };
  // Issue a filled batch, terminal or not.
  const auto ready = [&] {
    const Outcome outcome =
        batch.event.kind == EventKind::terminal ? batch.event.outcome : Outcome::none;
    return issue(std::move(batch), outcome);
  };
  try {
    // `DriverSession.advance`'s order: the owed launch plane, a latched snapshot at
    // an endpoint, completion, then the march to the next plane.
    if (launch_pending_) {
      launch_pending_ = false;
      batch.event = observe(batch, Observation::launch, -1);
      return issue(std::move(batch));
    }
    // A warning the caller allowed: the rest of its point's work, once (SEP 47 5.3).
    const Continuation continued = continuation_;
    continuation_ = Continuation::none;
    switch (continued) {
    case Continuation::attach:
      sync_collar();
      if (refused_crossing(batch) || (edge_ && edge_launch(batch)))
        return ready();
      break;
    case Continuation::edge_launch:
      if (edge_proceed(batch, Continuation::edge_launch))
        return ready();
      break;
    case Continuation::endpoint:
      // The step's check warned: its crossing, then the edge, its row and natural stop.
      if (refused_crossing(batch) || endpoint(batch, plan_.planes[next_plane_]))
        return ready();
      break;
    case Continuation::edge_endpoint:
      // The edge warned at the step: the rest of its read, then the row and stop.
      if (edge_proceed(batch, Continuation::edge_endpoint) ||
          endpoint(batch, plan_.planes[next_plane_], false))
        return ready();
      break;
    case Continuation::plane:
      if (refused_crossing(batch) || (edge_ && edge_plane(batch)))
        return ready();
      plane_rest(batch);
      return ready();
    case Continuation::edge_plane:
      if (edge_proceed(batch, Continuation::edge_plane))
        return ready();
      plane_rest(batch);
      return ready();
    case Continuation::finish:
      terminal(batch, Outcome::completed, Failure{});
      return ready();
    case Continuation::none:
      break;
    }
    if (continued != Continuation::endpoint &&
        continued != Continuation::edge_endpoint) {
      if (monitor_ && !attached_ && attach(batch))
        return ready();
      if (snapshot_latched() && at_endpoint()) {
        snapshot(batch);
        return issue(std::move(batch));
      }
      if (next_plane_ >= plan_.planes.size()) {
        // The final check before completion (`BoundaryMonitor.finish`).
        if (monitor_) {
          const watch::Finish finish = monitor_->finish();
          if (finish.warns) {
            Notice notice;
            notice.kind = Notice::Kind::removal;
            notice.z = here();
            notice.value = finish.removed;
            continuation_ = Continuation::finish;
            warning(batch, notice);
            return ready();
          }
        }
        terminal(batch, Outcome::completed, Failure{});
        return issue(std::move(batch), Outcome::completed);
      }
    }
    const double target = plan_.planes[next_plane_];
    const bool restored = observed_;
    if (restored) {
      // The march is ahead of the last interpolated plane: resume it there.
      dense_->restore();
      observed_ = false;
      publish();
    }
    if (natural_due_ && !march::reached(position(), target, landing_ulps_)) {
      // The step that crossed the last plane ends before this one, and is observed
      // first (`_natural_stop_due`).
      natural_due_ = false;
      observe_plane(batch, Observation::step, static_cast<std::int64_t>(next_plane_),
                    false);
      return ready();
    }
    if (restored && snapshot_latched() && at_endpoint() &&
        !march::reached(position(), target, landing_ulps_)) {
      // A snapshot latched while a plane was held is taken at the endpoint ahead of
      // it, every plane its step crossed observed (`halt_ahead`): with planes closer
      // than the steps, a later endpoint would only be reached past another plane.
      snapshot(batch);
      return issue(std::move(batch));
    }
    const double slack = plan_.replay
                             ? replay_tolerance(plan_.replay_steps, plan_.replay_length,
                                                target, landing_ulps_)
                             : 0.0;
    const std::size_t n_steps = schedule().steps.size();
    const auto running = [&] {
      // `replay_to`'s and `march_to`'s loops: to the edge within the slack, or
      // reached.
      const double z = position();
      if (plan_.replay)
        return cursor() < n_steps && z < target - slack;
      return !march::reached(z, target, landing_ulps_);
    };
    while (running()) {
      if (stop_.load(std::memory_order_acquire)) {
        terminal(batch, Outcome::stopped, Failure{});
        return issue(std::move(batch), Outcome::stopped);
      }
      // A bounded yield, between attempts and after at least one, or a full chunk
      // buffer: a transport no-op carrying the chunk.
      if (chunk_full() || (made() > 0 && (made() >= budget || elapsed() >= deadline))) {
        progress(batch);
        return issue(std::move(batch));
      }
      const std::size_t attempts =
          std::min({plan_.call_attempts, budget - std::min(budget, made()),
                    plan_.attempt_capacity - chunk_attempts()});
      const double seconds = std::min(plan_.call_seconds, deadline - elapsed());
      const auto call_started = Clock::now();
      const march::Result result =
          call(target, slack, std::max<std::size_t>(attempts, 1),
               seconds > 0.0 ? seconds : 0.0);
      absorb(result, call_started);
      if (dense_ != nullptr)
        dense_->stepped(result);
      const march::Reason reason = result.reason;
      const bool event = event_reason(reason);
      // Every accepted step but an event's last is accounted where it falls: rows
      // and natural stops count it, each once (`_account_step`).
      const std::size_t n_rows = result.rows.size() / march::row_columns;
      for (std::size_t row = 0; row < n_rows; ++row) {
        const bool accepted =
            result.rows[row * march::row_columns + march::accepted_column] != 0.0;
        const bool final =
            row + 1 == n_rows && (event || reason == march::Reason::floor);
        if (accepted && !final) {
          if (plan_.period > 0)
            ++accepted_steps_;
          ++natural_count_;
        }
      }
      if (monitor_) {
        // The monitor checked each committed step in the march: the stages follow its
        // quiet verdict, and its stop or first warning acts at the step it checked,
        // before that step's statistics row (SEP 47 section 5.3, the committed
        // endpoint).
        sync_collar();
        if (reason == march::Reason::watch) {
          if (result.verdict == watch::Verdict::refuse) {
            if (refused_crossing(batch))
              return ready();
            throw std::logic_error("a native march refused a crossing the session "
                                   "does not find");
          }
          // The check's own judgement, with the amplitude its message names.
          const watch::Judgement& judgement = monitor_->last_judgement();
          if (judgement.verdict == watch::Verdict::warn)
            continuation_ = Continuation::endpoint;
          if (judged(batch, judgement))
            return ready();
        }
      }
      if (event && n_rows > 0 && last_accepted_ && endpoint(batch, target))
        return ready();
      switch (reason) {
      case march::Reason::stopped:
        terminal(batch, Outcome::stopped, Failure{});
        return issue(std::move(batch), Outcome::stopped);
      case march::Reason::floor:
      case march::Reason::budget:
      case march::Reason::error:
      case march::Reason::diverged:
        terminal(batch, Outcome::failed, failure_of(result, cursor()));
        return issue(std::move(batch), Outcome::failed);
      case march::Reason::capacity:
      case march::Reason::dense:
      case march::Reason::monitor:
        throw std::logic_error("a native march returned for work this session refused");
      default:
        break;
      }
    }
    const double z = position();
    if (dense_ != nullptr && dense_->crosses(z, target, landing_ulps_)) {
      observe_inside(batch, target);
      return ready();
    }
    if (plan_.replay && z < target - slack) {
      // `replay_to`: the schedule ran out before the target.
      Failure failure;
      failure.reason = march::Reason::error;
      failure.category = march::Category::refused;
      failure.schedule_exhausted = true;
      failure.target = target;
      failure.cursor = cursor();
      terminal(batch, Outcome::failed, std::move(failure));
      return issue(std::move(batch), Outcome::failed);
    }
    if (plan_.replay || plane_reached(z, target)) {
      // A step owed as a step plane that ends on the target is observed by its plane,
      // once.
      natural_due_ = false;
      observe_plane(batch, Observation::plane, static_cast<std::int64_t>(next_plane_),
                    true);
      return ready();
    }
    observe_plane(batch,
                  plan_.step_planes > 0 ? Observation::step : Observation::natural,
                  static_cast<std::int64_t>(next_plane_), false);
    return ready();
  } catch (const std::exception& error) {
    // A failure outside the march's own error returns (a statistics evaluation, a
    // refused return): the committed state stands, eligible.
    Batch failed;
    terminal(failed, Outcome::failed, failure_of(error));
    return issue(std::move(failed), Outcome::failed);
  }
}

void Session::continue_monitor(const watch::Resumption& record, const bool plane,
                               const std::vector<std::array<double, 2>>& unwatched,
                               const transverse::Resumption& edge_record) {
  const std::scoped_lock lock(mutex_);
  require_open();
  if (attached_)
    throw LifecycleError("a boundary monitor is continued before it watches the run");
  if (!plan_.monitor || plan_.monitor->resume)
    return;
  plan_.monitor->resume = true;
  plan_.monitor->record = record;
  plan_.monitor->plane = plane;
  plan_.monitor->unwatched = unwatched;
  plan_.monitor->edge_record = edge_record;
}

void Session::read(const std::uint64_t sequence, const std::uint64_t generation,
                   const std::function<void(const Method&, Resident&)>& reader) {
  const std::scoped_lock lock(mutex_);
  require_open();
  if (sequence != 0 &&
      (!pending_ || sequence != sequence_ || generation != pending_generation_))
    throw LifecycleError("the ticket has expired: its batch was acknowledged, or it "
                         "belongs to another session");
  reader(method_, *resident_);
}

std::optional<Batch> Session::fail_outside() {
  const std::scoped_lock lock(mutex_);
  if (closed() || pending_ || state_.load() == State::finished)
    return std::nullopt;
  // The committed state is the last commit's, eligible. The chunk buffers are drained
  // into this batch, so an attempt an aborted advance left there is delivered once,
  // with the failure's checkpoint, never dropped or delivered twice.
  Failure failure;
  failure.reason = march::Reason::error;
  failure.category = march::Category::backend;
  failure.message = "the run failed outside its session";
  failure.cursor = cursor();
  Batch batch;
  terminal(batch, Outcome::failed, std::move(failure));
  return issue(std::move(batch), Outcome::failed);
}

std::optional<watch::Record> Session::monitor_record() const {
  const std::scoped_lock lock(mutex_);
  if (!monitor_)
    return std::nullopt;
  return monitor_->record();
}

std::optional<double> Session::monitor_cap() const {
  const std::scoped_lock lock(mutex_);
  if (!monitor_)
    return std::nullopt;
  return monitor_->cap();
}

double Session::quadrature_peak() {
  const std::scoped_lock lock(mutex_);
  if (closed())
    return std::numeric_limits<double>::quiet_NaN();
  return resident_->quadrature_peak();
}

void Session::acknowledge(const std::uint64_t sequence) {
  const std::scoped_lock lock(mutex_);
  require_open();
  if (!pending_ || sequence != sequence_)
    refuse<LifecycleError>("receipt ", sequence,
                           " is not the pending batch's: a foreign, stale or repeated "
                           "acknowledgment");
  pending_ = false;
  state_.store(outcome_.load() != Outcome::none ? State::finished : State::ready,
               std::memory_order_release);
}

void Session::export_field(const std::uint64_t sequence, const std::uint64_t generation,
                           const bool power, const bool optics, const double* weight,
                           double* out) {
  const std::scoped_lock lock(mutex_);
  if (closed() || !pending_ || sequence != sequence_ ||
      generation != pending_generation_)
    throw LifecycleError("the export ticket has expired: its batch was acknowledged, "
                         "or it belongs to another session");
  if (poisoned())
    throw LifecycleError("the state was left in flight by an abandoned step and "
                         "cannot be exported");
  const std::size_t n = resident_->elements();
  bool scale = false;
  if (power) {
    if (weight == nullptr)
      throw std::invalid_argument(
          "a true-power export needs the route's weight at its position");
    bool valid = true;
    for (std::size_t index = 0; index < n; ++index) {
      const double value = weight[index];
      valid = valid && std::isfinite(value) && value > 0.0;
      scale = scale || value != 1.0;
    }
    if (!valid)
      throw std::domain_error("power weights must be positive and finite");
  }
  const double* state = resident_->host_field();
  if (state == nullptr) {
    // A device state crosses once, into the destination itself, and the recipe then
    // converts it there in place: no state-sized host staging buffer per export.
    resident_->field_to_host(out);
    state = out;
  } else {
    resident_->count_to_host(2 * n * sizeof(double));
  }
  apply_recipe(state, n, resident_->n_omega(), weight, scale, optics, out);
  exports_.fetch_add(1, std::memory_order_relaxed);
  export_bytes_.fetch_add(2 * n * sizeof(double), std::memory_order_relaxed);
}

SnapshotReceipt Session::request_snapshot() {
  if (closed())
    throw LifecycleError("the session is closed");
  // The latch holds the request count times two, plus one while a request is
  // outstanding: one compare-and-swap either joins the outstanding request or makes
  // a new one, with no lock.
  std::uint64_t value = latch_.load(std::memory_order_acquire);
  while (true) {
    if ((value & 1U) != 0)
      return {.request = value >> 1U, .coalesced = true};
    const std::uint64_t next = (((value >> 1U) + 1) << 1U) | 1U;
    if (latch_.compare_exchange_weak(value, next, std::memory_order_acq_rel)) {
      halt_.store(true, std::memory_order_release);
      return {.request = next >> 1U, .coalesced = false};
    }
  }
}

Status Session::status() const noexcept {
  Status out;
  out.z = published_z_.load(std::memory_order_relaxed);
  out.plane_index = published_plane_.load(std::memory_order_relaxed);
  out.attempts = published_attempts_.load(std::memory_order_relaxed);
  out.rejected = published_rejected_.load(std::memory_order_relaxed);
  out.state = state_.load(std::memory_order_acquire);
  out.outcome = outcome_.load(std::memory_order_acquire);
  out.generation = published_generation_.load(std::memory_order_relaxed);
  out.march_calls = march_calls_.load(std::memory_order_relaxed);
  out.rows_taken = rows_taken_.load(std::memory_order_relaxed);
  out.batches = batches_.load(std::memory_order_relaxed);
  out.exports = exports_.load(std::memory_order_relaxed);
  out.export_bytes = export_bytes_.load(std::memory_order_relaxed);
  const std::size_t width = published_width_.load(std::memory_order_relaxed);
  out.buffer_bytes =
      plan_.attempt_capacity * attempt_columns * sizeof(double) +
      (plan_.period > 0
           ? plan_.row_capacity * ((1 + width) * sizeof(double) + sizeof(std::uint64_t))
           : 0);
  return out;
}

std::size_t Session::elements() const {
  if (closed())
    throw LifecycleError("the session is closed");
  return elements_;
}

bool Session::close() noexcept {
  std::unique_lock lock(mutex_, std::defer_lock);
  try {
    lock.lock();
  } catch (const std::exception&) {
    // The lock failed: nothing is released, and the caller keeps its lease.
    return false;
  }
  if (closed())
    return true;
  state_.store(State::closed, std::memory_order_release);
  closed_.store(true, std::memory_order_release);
  pending_ = false;
  // The march's engine refers to the leased state: both go, the engine first.
  dense_ = nullptr;
  if (auto* adams = std::get_if<AdamsMethod>(&method_)) {
    adams->engine.reset();
    adams->dense.reset();
  }
  if (auto* runge_kutta = std::get_if<RungeKuttaMethod>(&method_)) {
    runge_kutta->engine.reset();
    runge_kutta->dense.reset();
  }
  resident_.reset();
  // Move-assigning an empty vector releases the storage and cannot throw; assigning
  // `{}` goes through the initializer-list assignment, which may.
  history_ = std::vector<double>();
  row_z_ = std::vector<double>();
  row_step_ = std::vector<std::int64_t>();
  row_values_ = std::vector<double>();
  return true;
}

} // namespace solprop_native::session
