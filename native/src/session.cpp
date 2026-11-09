// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/session.hpp"

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
    accepted row is then the event step (`_NativeAdamsMarch._call`). */
bool event_reason(const march::Reason reason) {
  return reason == march::Reason::reached || reason == march::Reason::host_step ||
         reason == march::Reason::monitor || reason == march::Reason::watch;
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

/** A failure outside a march call: an exception the session caught. */
Failure failure_of(const std::exception& error) {
  Failure failure;
  failure.reason = march::Reason::error;
  failure.message = error.what();
  failure.category = dynamic_cast<const std::invalid_argument*>(&error) != nullptr
                         ? march::Category::value
                         : march::Category::backend;
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

void apply_recipe(const double* state, const std::size_t n, const std::size_t n_omega,
                  const double* weight, const bool scale, const bool optics,
                  double* out) {
  // `np.multiply(state, np.sqrt(weight))` multiplies by the complex `sqrt(w) + 0j`, and
  // `np.multiply(shift, np.conj(out))` by the complex `(-1)^k + 0j`: each product is
  // NumPy's complex one, `(a c - b d, a d + b c)`, its zero terms kept so that signed
  // zeros and non-finite values come out as NumPy's.
  for (std::size_t index = 0; index < n; ++index) {
    double re = state[2 * index];
    double im = state[2 * index + 1];
    if (scale) {
      const double root = std::sqrt(weight[index]);
      const double real_part = re * root - im * 0.0;
      const double imaginary_part = re * 0.0 + im * root;
      re = real_part;
      im = imaginary_part;
    }
    if (optics) {
      const double conjugate = -im;
      const double sign = (index % n_omega) % 2 == 0 ? 1.0 : -1.0;
      const double real_part = sign * re - 0.0 * conjugate;
      const double imaginary_part = sign * conjugate + 0.0 * re;
      re = real_part;
      im = imaginary_part;
    }
    out[2 * index] = re;
    out[2 * index + 1] = im;
  }
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
  adams_ = std::holds_alternative<AdamsMethod>(method_);
  if (adams_) {
    auto& adams = std::get<AdamsMethod>(method_);
    if (!adams.engine)
      throw std::invalid_argument("a session's Adams march needs its engine");
    march::validate(adams.policy);
    landing_ulps_ = adams.policy.landing_ulps;
    if (adams.state.z != z)
      throw std::invalid_argument("the Adams march's state is not at the start");
  } else {
    auto& runge_kutta = std::get<RungeKuttaMethod>(method_);
    if (!runge_kutta.engine)
      throw std::invalid_argument("a session's Runge-Kutta march needs its engine");
    landing_ulps_ = runge_kutta.policy.landings.landing_ulps;
    if (runge_kutta.state.z != z)
      throw std::invalid_argument("the Runge-Kutta march's state is not at the start");
  }
  if (!std::is_sorted(plan_.planes.begin(), plan_.planes.end()))
    throw std::invalid_argument("a session's save planes increase");
  if (plan_.next_plane > plan_.planes.size())
    throw std::invalid_argument("a session's first plane is past its planes");
  if (plan_.period > 0 && plan_.row_statistics.empty())
    throw std::invalid_argument("periodic rows need their statistics");
  if (plan_.call_attempts < 1 || !(plan_.call_seconds > 0.0))
    throw std::invalid_argument("a march call needs at least one attempt and time");
  if (plan_.replay && plan_.replay_steps == 0)
    throw std::invalid_argument("a replayed schedule has steps");
  next_plane_ = plan_.next_plane;
  launch_pending_ = plan_.launch_pending;
  history_.reserve(attempt_columns * plan_.call_attempts);
  if (plan_.period > 0) {
    row_z_.reserve(plan_.row_capacity);
    row_step_.reserve(plan_.row_capacity);
  }
}

void Session::require_open() const {
  if (state_ == State::closed)
    throw LifecycleError("the session is closed");
}

double Session::position() const {
  return std::visit([](const auto& method) { return method.state.z; }, method_);
}

std::uint64_t Session::cursor() const {
  return std::visit(
      [](const auto& method) {
        return static_cast<std::uint64_t>(method.state.cursor);
      },
      method_);
}

std::uint64_t Session::host_step() const {
  // The accepted steps to the driver's next work, the least of them: a periodic row,
  // a step plane, and every step while a snapshot is latched (each march's `_bounds`).
  std::uint64_t steps = 0;
  const auto take = [&steps](const std::uint64_t value) {
    steps = steps == 0 ? value : std::min(steps, value);
  };
  if (plan_.period > 0)
    take(steps_to_next(plan_.period, accepted_steps_));
  if (plan_.step_planes > 0)
    take(steps_to_next(plan_.step_planes, natural_count_));
  if (snapshot_latched_.load(std::memory_order_acquire))
    take(1);
  return steps;
}

bool Session::at_endpoint() const {
  // An accepted endpoint this session reached, with no attempt made since: the
  // starting position holds no history to snapshot, and a rejection's trial is past.
  return stepped_ && last_accepted_;
}

Session::Call Session::call(const double target, const double slack,
                            const std::size_t attempts, const double seconds) {
  march::Bounds bounds;
  bounds.target = target;
  bounds.host_step = host_step();
  // No monitor, transverse edge or split collar (refused at preparation): the caps
  // are infinite and nothing is read.
  bounds.read_spacing = std::numeric_limits<double>::infinity();
  bounds.max_attempts = attempts;
  bounds.max_seconds = seconds;
  bounds.replay = plan_.replay;
  bounds.slack = slack;
  bounds.stop = &stop_;
  ++march_calls_;
  Call made;
  if (adams_) {
    auto& adams = std::get<AdamsMethod>(method_);
    made.result =
        march::run(adams.policy, adams.state, *adams.engine, bounds, adams.schedule);
  } else {
    auto& runge_kutta = std::get<RungeKuttaMethod>(method_);
    made.rk = true;
    made.result = rk::run(runge_kutta.policy, runge_kutta.state, *runge_kutta.engine,
                          bounds, runge_kutta.schedule)
                      .march;
  }
  return made;
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
}

void Session::take_row() {
  // `_record_step_stats`: the row of the step just committed, its attempt index the one
  // `_record` has just appended, counted since the previous observation.
  const double z = position();
  const auto values = resident_->statistics(plan_.row_statistics, z);
  if (row_z_.empty())
    row_width_ = values.size();
  else if (values.size() != row_width_)
    throw std::logic_error("a periodic row changed its width");
  row_z_.push_back(z);
  row_step_.push_back(history_.size() / attempt_columns - 1);
  row_values_.insert(row_values_.end(), values.begin(), values.end());
  ++rows_taken_;
}

void Session::consume() {
  // An observation drains the attempts and rows since the previous one.
  history_.clear();
  progress_mark_ = 0;
  row_z_.clear();
  row_step_.clear();
  row_values_.clear();
  clock_ = Clock::now();
}

void Session::fill(Batch& batch, Event& event, const bool rows) {
  // The attempts since the previous observation, and with `rows` its periodic rows.
  batch.attempts = history_;
  event.attempts_begin = 0;
  event.attempts_end = history_.size() / attempt_columns;
  if (rows) {
    batch.row_z = row_z_;
    batch.row_step = row_step_;
    batch.row_values = row_values_;
    batch.row_width = row_width_;
    event.has_rows = true;
    event.rows_begin = 0;
    event.rows_end = row_z_.size();
  }
}

Event Session::observe(Batch& batch, const Observation kind, const std::int64_t plane) {
  Event event;
  event.kind = EventKind::observation;
  event.observation = kind;
  event.z = position();
  event.plane_index = plane;
  event.ticket = true;
  event.generation = generation_;
  if (!plan_.plane_statistics.empty()) {
    batch.plane_stats = resident_->statistics(plan_.plane_statistics, event.z);
    event.has_stats = true;
  }
  fill(batch, event, true);
  consume();
  return event;
}

void Session::progress(Batch& batch) {
  // `_progress`: the attempts since the previous event, the next observation's still.
  Event event;
  event.kind = EventKind::progress;
  event.z = position();
  const std::size_t total = history_.size() / attempt_columns;
  if (batch.attempts.empty()) {
    // A batch of progress alone carries the attempts from the mark.
    batch.attempts.assign(history_.begin() + static_cast<std::ptrdiff_t>(
                                                 progress_mark_ * attempt_columns),
                          history_.end());
    event.attempts_begin = 0;
    event.attempts_end = total - progress_mark_;
  } else {
    event.attempts_begin = progress_mark_;
    event.attempts_end = total;
  }
  progress_mark_ = total;
  batch.events.push_back(event);
}

void Session::snapshot(Batch& batch, const bool progressed) {
  // `DriverSession._snapshot`: the attempts and rows since the previous observation,
  // included without being consumed (SEP 47 section 5.5).
  snapshot_latched_.store(false, std::memory_order_release);
  Event event;
  event.kind = EventKind::snapshot;
  event.z = position();
  event.ticket = true;
  event.generation = generation_;
  fill(batch, event, true);
  if (progressed && progress_mark_ < history_.size() / attempt_columns) {
    Event moved;
    moved.kind = EventKind::progress;
    moved.z = event.z;
    moved.attempts_begin = progress_mark_;
    moved.attempts_end = history_.size() / attempt_columns;
    batch.events.push_back(moved);
  }
  progress_mark_ = history_.size() / attempt_columns;
  batch.events.push_back(event);
}

void Session::terminal(Batch& batch, const Outcome outcome, Failure failure) {
  Event event;
  event.kind = EventKind::terminal;
  event.z = position();
  event.outcome = outcome;
  event.failure = std::move(failure);
  if (outcome != Outcome::completed) {
    // A stop or a failure: the committed state, eligible, with every attempt and row
    // since the previous observation (`DriverSession._terminal`).
    event.ticket = true;
    event.generation = generation_;
    fill(batch, event, true);
  }
  batch.events.push_back(event);
}

Batch Session::issue(Batch batch, const Outcome outcome) {
  batch.sequence = ++sequence_;
  pending_ = true;
  pending_generation_ = generation_;
  state_ = State::pending;
  if (outcome != Outcome::none)
    outcome_ = outcome;
  ++batches_;
  return batch;
}

Batch Session::advance(const std::size_t work_limit, const double deadline) {
  const std::scoped_lock lock(mutex_);
  if (state_ == State::pending)
    throw LifecycleError(
        "acknowledge the pending batch before advancing the session again");
  if (state_ == State::finished)
    throw LifecycleError("the session has ended and cannot advance");
  require_open();
  const auto started = Clock::now();
  const auto elapsed = [&started]() {
    return std::chrono::duration<double>(Clock::now() - started).count();
  };
  const std::size_t budget = std::max<std::size_t>(work_limit, 1);
  const std::uint64_t before = attempts_;
  Batch batch;
  const auto made = [&]() { return static_cast<std::size_t>(attempts_ - before); };
  const auto finish = [&](Batch& done, const Outcome outcome = Outcome::none) {
    done.new_attempts = made();
    return issue(std::move(done), outcome);
  };
  try {
    // `DriverSession.advance`'s order: the owed launch plane, a latched snapshot at
    // an endpoint, completion, then the march to the next plane.
    if (launch_pending_) {
      launch_pending_ = false;
      batch.events.push_back(observe(batch, Observation::launch, -1));
      return finish(batch);
    }
    if (snapshot_latched_.load(std::memory_order_acquire) && at_endpoint()) {
      snapshot(batch, false);
      return finish(batch);
    }
    if (next_plane_ >= plan_.planes.size()) {
      terminal(batch, Outcome::completed, Failure{});
      return finish(batch, Outcome::completed);
    }
    const double target = plan_.planes[next_plane_];
    const double slack = plan_.replay
                             ? replay_tolerance(plan_.replay_steps, plan_.replay_length,
                                                target, landing_ulps_)
                             : 0.0;
    const std::size_t n_steps =
        adams_ ? std::get<AdamsMethod>(method_).schedule.steps.size()
               : std::get<RungeKuttaMethod>(method_).schedule.steps.size();
    const auto running = [&]() {
      // `replay_to`'s and `march_to`'s loops: to the edge within the slack, or reached.
      const double z = position();
      if (plan_.replay)
        return cursor() < n_steps && z < target - slack;
      return !march::reached(z, target, landing_ulps_);
    };
    while (running()) {
      if (stop_.load(std::memory_order_acquire)) {
        terminal(batch, Outcome::stopped, Failure{});
        return finish(batch, Outcome::stopped);
      }
      // A bounded yield, between attempts and after at least one: a transport no-op.
      if (made() > 0 && (made() >= budget || elapsed() >= deadline)) {
        progress(batch);
        return finish(batch);
      }
      const std::size_t attempts = std::min(plan_.call_attempts, budget - made());
      const double seconds = std::min(plan_.call_seconds, deadline - elapsed());
      const auto call_started = Clock::now();
      const Call done = call(target, slack, attempts, seconds > 0.0 ? seconds : 0.0);
      const march::Result& result = done.result;
      absorb(result, call_started);
      const march::Reason reason = result.reason;
      const bool event = event_reason(reason);
      // Every accepted step but an event's last is accounted where it falls: rows and
      // natural stops count it, each once (`_account_step`).
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
      if (event && n_rows > 0 && last_accepted_) {
        // The event step: its statistics row, then the natural stop
        // (`_take_step_stats`, `_natural_stop`).
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
          if (!march::reached(z, target, landing_ulps_)) {
            batch.events.push_back(observe(batch, Observation::step,
                                           static_cast<std::int64_t>(next_plane_)));
            return finish(batch);
          }
        } else if (!march::reached(z, target, landing_ulps_) &&
                   snapshot_latched_.load(std::memory_order_acquire)) {
          // `_halted_short`: the latch halts the march at this endpoint.
          snapshot(batch, true);
          return finish(batch);
        }
      }
      switch (reason) {
      case march::Reason::stopped:
        terminal(batch, Outcome::stopped, Failure{});
        return finish(batch, Outcome::stopped);
      case march::Reason::floor:
      case march::Reason::budget:
      case march::Reason::error:
      case march::Reason::diverged:
        terminal(batch, Outcome::failed, failure_of(result, cursor()));
        return finish(batch, Outcome::failed);
      case march::Reason::watch:
      case march::Reason::capacity:
      case march::Reason::dense:
      case march::Reason::monitor:
        throw std::logic_error("a native march returned for work this session refused");
      default:
        break;
      }
    }
    const double z = position();
    if (plan_.replay && z < target - slack) {
      // `replay_to`: the schedule ran out before the target.
      Failure failure;
      failure.reason = march::Reason::error;
      failure.category = march::Category::refused;
      failure.schedule_exhausted = true;
      failure.target = target;
      failure.cursor = cursor();
      terminal(batch, Outcome::failed, std::move(failure));
      return finish(batch, Outcome::failed);
    }
    if (plan_.replay || plane_reached(z, target)) {
      const std::size_t plane = next_plane_++;
      batch.events.push_back(
          observe(batch, Observation::plane, static_cast<std::int64_t>(plane)));
      return finish(batch);
    }
    batch.events.push_back(
        observe(batch, plan_.step_planes > 0 ? Observation::step : Observation::natural,
                static_cast<std::int64_t>(next_plane_)));
    return finish(batch);
  } catch (const std::exception& error) {
    // A failure outside the march's own error returns (a statistics evaluation, a
    // refused return): the committed state stands, eligible.
    Batch failed;
    terminal(failed, Outcome::failed, failure_of(error));
    return finish(failed, Outcome::failed);
  }
}

void Session::acknowledge(const std::uint64_t sequence) {
  const std::scoped_lock lock(mutex_);
  require_open();
  if (!pending_ || sequence != sequence_)
    throw LifecycleError("receipt " + std::to_string(sequence) +
                         " is not the pending batch's: a foreign, stale or repeated "
                         "acknowledgment");
  pending_ = false;
  state_ = outcome_ != Outcome::none ? State::finished : State::ready;
}

void Session::export_field(const std::uint64_t sequence, const std::uint64_t generation,
                           const bool power, const bool optics, const double* weight,
                           double* out) {
  const std::scoped_lock lock(mutex_);
  if (state_ == State::closed || !pending_ || sequence != sequence_ ||
      generation != pending_generation_)
    throw LifecycleError("the export ticket has expired: its batch was acknowledged, "
                         "or it belongs to another session");
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
  std::vector<double> staged;
  if (state == nullptr) {
    // A device state crosses to the host first; the recipe follows in place.
    staged.resize(2 * n);
    resident_->field_to_host(staged.data());
    state = staged.data();
  } else {
    resident_->count_to_host(2 * n * sizeof(double));
  }
  apply_recipe(state, n, resident_->n_omega(), weight, scale, optics, out);
  ++exports_;
  export_bytes_ += 2 * n * sizeof(double);
}

SnapshotReceipt Session::request_snapshot() {
  const std::scoped_lock lock(snapshot_mutex_);
  if (closed())
    throw LifecycleError("the session is closed");
  if (snapshot_latched_.load(std::memory_order_acquire))
    return {snapshot_requests_, true};
  ++snapshot_requests_;
  snapshot_latched_.store(true, std::memory_order_release);
  return {snapshot_requests_, false};
}

Status Session::status() const {
  const std::scoped_lock lock(mutex_);
  Status out;
  out.z = position();
  out.plane_index = next_plane_;
  out.attempts = attempts_;
  out.rejected = rejected_;
  out.state = state_;
  out.outcome = outcome_;
  out.generation = generation_;
  out.march_calls = march_calls_;
  out.rows_taken = rows_taken_;
  out.batches = batches_;
  out.exports = exports_;
  out.export_bytes = export_bytes_;
  out.buffer_bytes =
      (history_.capacity() + row_z_.capacity() + row_values_.capacity()) *
          sizeof(double) +
      row_step_.capacity() * sizeof(std::uint64_t);
  return out;
}

std::size_t Session::elements() const {
  if (!resident_)
    throw LifecycleError("the session is closed");
  return resident_->elements();
}

void Session::close() noexcept {
  const std::scoped_lock lock(mutex_);
  if (state_ == State::closed)
    return;
  state_ = State::closed;
  closed_.store(true, std::memory_order_release);
  pending_ = false;
  // The march's engine refers to the leased state: both go, the engine first.
  std::visit([](auto& method) { method.engine.reset(); }, method_);
  resident_.reset();
  history_ = {};
  row_z_ = {};
  row_step_ = {};
  row_values_ = {};
}

} // namespace solprop_native::session
