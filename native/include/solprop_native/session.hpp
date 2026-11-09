// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "solprop_native/adams_march.hpp"
#include "solprop_native/rk_march.hpp"

/*
 * The native execution session (solprop SEP 47 sections 5 and 6.1): the one mutable
 * owner of a trajectory. It holds the march's policy, state and history, the plane
 * schedule, the periodic statistics cadence with its resident rows, the stop flag, the
 * snapshot latch, the committed-state generation and the pending batch of events, and
 * it runs the marches (`march::run`, `rk::run`) between Python's service points. Python
 * prepares it once and then only advances it, services what it reports and
 * acknowledges each batch; it never rebuilds a bound, replays an attempt or takes a
 * statistics row.
 *
 * The scheduler's decisions -- which plane is next, when a step plane, a statistics
 * row, a bounded yield or a snapshot falls due, and the reached and replay tolerances
 * -- are solprop's `DriverSession`, `StepDriver` and `solprop.backends.driver`
 * functions, transcribed. Like the marches they are compiled strict (no contraction, no
 * fast-math) in their own translation unit, so they round as solprop's Python does;
 * that unit also holds the export, whose recipe is `solprop.backends.export`'s
 * `apply_recipe` operation for operation.
 *
 * The field arithmetic and the statistics stay where the state lives, behind
 * `Resident`. Python-free and Kokkos-free: host code only.
 */
namespace solprop_native::session {

/** A session used outside its lifecycle: a stale or foreign receipt, an advance across
    a pending batch, use after close or after a terminal outcome, an expired ticket. */
class LifecycleError : public std::logic_error {
public:
  using std::logic_error::logic_error;
};

/** Where a session is in its lifecycle (`solprop.backends.session.SessionState`). */
enum class State : int { ready = 0, pending = 1, finished = 2, closed = 3 };

/** The closed set of event kinds (SEP 47 section 5.3). */
enum class EventKind : int {
  observation = 0,
  progress = 1,
  snapshot = 2,
  terminal = 3
};

/** Why an observation was made (`solprop.backends.events.ObservationKind`). */
enum class Observation : int { launch = 0, plane = 1, step = 2, natural = 3 };

/** How a session ended (`solprop.backends.events.Outcome`). */
enum class Outcome : int { none = 0, completed = 1, stopped = 2, failed = 3 };

/** A statistics evaluation the resident state makes (`GeneralState::statistics` or
    `spatial_statistics`), with every argument fixed at preparation. */
struct StatisticsRecipe {
  enum class Kind : int { none = 0, temporal = 1, spatial = 2 };
  Kind kind = Kind::none;
  std::uint64_t flags = 0;
  /** The energy windows' lower and upper edges, pairs. */
  std::vector<double> windows;
  /** The route's modal power weight, constant over the run; empty for a spatial run. */
  std::vector<double> power_weight;
  double intensity_factor = 0.0;
  [[nodiscard]] bool empty() const noexcept { return kind == Kind::none || flags == 0; }
};

/** What a session reads from and writes through the state it leases: the field and
    the statistics, where they live. The resident engines' implementation is
    `engines::GeneralResident`. */
class Resident {
public:
  Resident() = default;
  Resident(const Resident&) = delete;
  Resident& operator=(const Resident&) = delete;
  Resident(Resident&&) = delete;
  Resident& operator=(Resident&&) = delete;
  virtual ~Resident() = default;
  /** The field's complex element count, `columns * n_omega`. */
  [[nodiscard]] virtual std::size_t elements() const = 0;
  /** The field's last axis, the frequency bins, along which `(-1)^k` alternates. */
  [[nodiscard]] virtual std::size_t n_omega() const = 0;
  /** The committed field as host memory, interleaved real and imaginary parts, or
      null where it is not host-accessible. */
  virtual const double* host_field() = 0;
  /** Copy the committed field into `out`, `2 elements()` doubles. */
  virtual void field_to_host(double* out) = 0;
  /** Count bytes a host reader of `host_field` took out of the state. */
  virtual void count_to_host(std::size_t bytes) = 0;
  /** Evaluate `recipe` on the committed state at `z`. */
  virtual std::vector<double> statistics(const StatisticsRecipe& recipe, double z) = 0;
};

/** The Adams march a session runs: its policy, state, replayed schedule and engine. */
struct AdamsMethod {
  march::Policy policy;
  march::State state;
  march::Schedule schedule;
  std::unique_ptr<march::Engine> engine;
};

/** The Runge-Kutta march a session runs. */
struct RungeKuttaMethod {
  rk::Policy policy;
  rk::State state;
  march::Schedule schedule;
  std::unique_ptr<rk::Engine> engine;
};

using Method = std::variant<AdamsMethod, RungeKuttaMethod>;

/** The observation request and the run's fixed itinerary, as Python resolved them. */
struct Plan {
  /** The save planes, m, increasing. */
  std::vector<double> planes;
  /** The first save plane strictly beyond the start. */
  std::size_t next_plane = 0;
  /** Whether the plane at the start is owed (the launch plane of a fresh run). */
  bool launch_pending = true;
  /** Observe the end of every `step_planes`-th accepted step; zero for none. */
  std::uint64_t step_planes = 0;
  /** The plane statistics, and the periodic rows' subset with its period (zero for
      no rows). */
  StatisticsRecipe plane_statistics;
  StatisticsRecipe row_statistics;
  std::uint64_t period = 0;
  /** Rows the resident buffer holds before it first grows. */
  std::size_t row_capacity = 64;
  /** A replayed schedule: its step count and total length, for `replay_tolerance`. */
  bool replay = false;
  std::size_t replay_steps = 0;
  double replay_length = 0.0;
  /** Attempts one march call makes at most (SEP 39 Q3), and its wall time, s. */
  std::size_t call_attempts = 256;
  double call_seconds = 0.1;
};

/** The failure a terminal event reports: the march's return and its diagnostics, which
    Python formats as solprop's exception. */
struct Failure {
  march::Reason reason = march::Reason::reached;
  march::Category category = march::Category::none;
  march::Stage stage = march::Stage::none;
  std::string message;
  double floor_proposal = 0.0;
  std::uint64_t cost = 0;
  double replay_dz = 0.0;
  double replay_end = 0.0;
  double replay_norm = 0.0;
  /** The replay cursor when it failed. */
  std::uint64_t cursor = 0;
  /** A replayed schedule ran out short of its target (`replay_to`'s error). */
  bool schedule_exhausted = false;
  double target = 0.0;
};

/** One event of a batch. Ranges index the batch's arrays. */
struct Event {
  EventKind kind = EventKind::progress;
  Observation observation = Observation::launch;
  double z = 0.0;
  std::int64_t plane_index = -1;
  /** Attempt rows `[attempts_begin, attempts_end)` of `Batch::attempts`. */
  std::size_t attempts_begin = 0;
  std::size_t attempts_end = 0;
  /** Periodic rows `[rows_begin, rows_end)`, and whether the event carries rows. */
  std::size_t rows_begin = 0;
  std::size_t rows_end = 0;
  bool has_rows = false;
  /** Whether the event carries plane statistics, `Batch::plane_stats`. */
  bool has_stats = false;
  /** Whether the event's field can be exported while the batch is pending, and the
      committed-state generation it names. */
  bool ticket = false;
  std::uint64_t generation = 0;
  Outcome outcome = Outcome::none;
  Failure failure;
};

/** Columns of one attempt row: its end, step, scaled norm, acceptance and the wall
    time since the previous observation, s. */
constexpr std::size_t attempt_columns = 5;

/** What one `advance` reports: its events in committed order and the arrays they
    index. The attempt rows are those since the previous observation (from the last
    progress mark for a batch of progress alone); the periodic rows are those since the
    previous observation. */
struct Batch {
  std::uint64_t sequence = 0;
  std::vector<Event> events;
  std::vector<double> attempts;
  /** The attempts this advance made: the last of `attempts`. */
  std::size_t new_attempts = 0;
  std::vector<double> row_z;
  std::vector<std::uint64_t> row_step;
  std::vector<double> row_values;
  std::size_t row_width = 0;
  std::vector<double> plane_stats;
};

/** A compact read-only view (`solprop.backends.session.SessionStatus`), with the
    structural counters of SEP 47 section 7.3. */
struct Status {
  double z = 0.0;
  std::size_t plane_index = 0;
  std::uint64_t attempts = 0;
  std::uint64_t rejected = 0;
  State state = State::ready;
  Outcome outcome = Outcome::none;
  std::uint64_t generation = 0;
  /** March calls made, statistics rows taken natively, batches issued, field exports
      and their bytes. */
  std::uint64_t march_calls = 0;
  std::uint64_t rows_taken = 0;
  std::uint64_t batches = 0;
  std::uint64_t exports = 0;
  std::uint64_t export_bytes = 0;
  /** Bytes the session's own buffers hold: attempt history, rows and plane stats. */
  std::uint64_t buffer_bytes = 0;
};

/** A snapshot request's receipt (`solprop.backends.session.SnapshotReceipt`). */
struct SnapshotReceipt {
  std::uint64_t request = 0;
  bool coalesced = false;
};

/** One trajectory's native execution. */
class Session {
public:
  Session(std::unique_ptr<Resident> resident, Method method, Plan plan, double z);
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) = delete;
  Session& operator=(Session&&) = delete;
  ~Session() = default;

  /** Run until a barrier, `work_limit` attempts (at least one) or `deadline` seconds,
      checked between attempts; report it as a pending batch. */
  Batch advance(std::size_t work_limit, double deadline);
  /** Release the pending batch `sequence` and allow advancing across its barrier. */
  void acknowledge(std::uint64_t sequence);
  /** Fill `out`, `2 elements()` doubles, with the pending batch's field at
      `generation`, converted by the recipe (`apply_recipe`): the true-power weight
      `weight` (`elements()` doubles) when `power`, then `(-1)^k conj` when `optics`. */
  void export_field(std::uint64_t sequence, std::uint64_t generation, bool power,
                    bool optics, const double* weight, double* out);
  /** Ask for a stop at the next safe boundary; any thread, no lock. */
  void request_stop() noexcept { stop_.store(true, std::memory_order_release); }
  /** Latch a snapshot for the next eligible accepted endpoint. */
  SnapshotReceipt request_snapshot();
  [[nodiscard]] Status status() const;
  [[nodiscard]] std::size_t elements() const;
  /** Release the lease and every buffer; idempotent. */
  void close() noexcept;
  [[nodiscard]] bool closed() const noexcept {
    return closed_.load(std::memory_order_acquire);
  }

private:
  using Clock = std::chrono::steady_clock;
  struct Call {
    march::Result result;
    bool rk = false;
  };

  void require_open() const;
  Call call(double target, double slack, std::size_t attempts, double seconds);
  void absorb(const march::Result& result, Clock::time_point started);
  [[nodiscard]] double position() const;
  [[nodiscard]] std::uint64_t cursor() const;
  [[nodiscard]] std::uint64_t host_step() const;
  [[nodiscard]] bool at_endpoint() const;
  void take_row();
  void consume();
  Event observe(Batch& batch, Observation kind, std::int64_t plane);
  void progress(Batch& batch);
  void snapshot(Batch& batch, bool progressed);
  void terminal(Batch& batch, Outcome outcome, Failure failure);
  void fill(Batch& batch, Event& event, bool rows);
  Batch issue(Batch batch, Outcome outcome = Outcome::none);

  std::unique_ptr<Resident> resident_;
  Method method_;
  Plan plan_;
  bool adams_ = false;
  int landing_ulps_ = 0;

  // Lifecycle.
  State state_ = State::ready;
  Outcome outcome_ = Outcome::none;
  std::uint64_t sequence_ = 0;
  bool pending_ = false;
  std::uint64_t pending_generation_ = 0;
  mutable std::mutex mutex_;

  // Controls any thread may set, and whether the session was closed, which they read.
  std::atomic<bool> stop_{false};
  std::atomic<bool> closed_{false};
  std::atomic<bool> snapshot_latched_{false};
  std::uint64_t snapshot_requests_ = 0;
  std::mutex snapshot_mutex_;

  // The schedule's cursors and the committed state's counters.
  std::size_t next_plane_ = 0;
  bool launch_pending_ = true;
  std::uint64_t accepted_steps_ = 0;
  std::uint64_t natural_count_ = 0;
  std::uint64_t generation_ = 0;
  bool stepped_ = false;
  bool last_accepted_ = false;

  // Accounting since the start and the buffers since the previous observation.
  std::uint64_t attempts_ = 0;
  std::uint64_t rejected_ = 0;
  std::vector<double> history_;
  std::size_t progress_mark_ = 0;
  Clock::time_point clock_;
  std::vector<double> row_z_;
  std::vector<std::uint64_t> row_step_;
  std::vector<double> row_values_;
  std::size_t row_width_ = 0;

  // Structural counters.
  std::uint64_t march_calls_ = 0;
  std::uint64_t rows_taken_ = 0;
  std::uint64_t batches_ = 0;
  std::uint64_t exports_ = 0;
  std::uint64_t export_bytes_ = 0;
};

// The scheduler's decisions, strict (session.cpp).

/** `numpy.isclose(position, target, rtol=1e-12, atol=0.0)`: whether a march that
    returned at `position` reached the plane it was sent to (`DriverSession.advance`).
 */
[[nodiscard]] bool plane_reached(double position, double target);
/** `solprop.backends.driver.replay_tolerance`: a replayed edge's slack at `position`,
    m, for a schedule of `steps` steps and length `length`. */
[[nodiscard]] double replay_tolerance(std::size_t steps, double length, double position,
                                      int landing_ulps);
/** Accepted steps to the next of `period`-th, counting from `count` taken. */
[[nodiscard]] std::uint64_t steps_to_next(std::uint64_t period, std::uint64_t count);
/** `solprop.backends.export.apply_recipe` over `n` elements of `n_omega` bins: `out`
    from `state`, both interleaved, the weight's square root when `scale`, then
    `(-1)^k conj` when `optics`. */
void apply_recipe(const double* state, std::size_t n, std::size_t n_omega,
                  const double* weight, bool scale, bool optics, double* out);
/** `a b + c` as this translation unit rounds it (solprop #135's witness); never
    inlined, as `march::strict_probe`. */
[[nodiscard, gnu::noinline]] double strict_probe(double a, double b, double c);

} // namespace solprop_native::session
