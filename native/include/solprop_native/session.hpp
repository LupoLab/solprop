// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "solprop_native/adams_march.hpp"
#include "solprop_native/rk_march.hpp"
#include "solprop_native/statistics_plan.hpp"
#include "solprop_native/transverse.hpp"
#include "solprop_native/watch.hpp"

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
 * The boundary monitor is the session's own (SEP 47 section 6.4): `watch::Monitor`
 * holds every decision and record, the marches take its checks at each committed
 * step, and the session takes the launch, plane and final checks, in the order of
 * section 5.3. A warning is a service barrier that remembers where it was: once the
 * caller allows it, the next advance does the rest of that point's work, once; a stop
 * is a terminal failure at the committed endpoint. A free-space run's transverse edge
 * is the session's too (`transverse::Edge`): launched after the temporal launch, read
 * after each committed step's temporal check where it falls due and at each plane, its
 * collar's current taken at every committed step from the resident state. A point that
 * raises several warnings serves each as its own barrier, in order.
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
  terminal = 3,
  warning = 4
};

/** Why an observation was made (`solprop.backends.events.ObservationKind`). */
enum class Observation : int { launch = 0, plane = 1, step = 2, natural = 3 };

/** How a session ended (`solprop.backends.events.Outcome`). */
enum class Outcome : int { none = 0, completed = 1, stopped = 2, failed = 3 };

/** A statistics evaluation the resident state makes, with every argument fixed at
    preparation (`statistics::Plan`, solprop SEP 47 section 6.4); none without a
    plan. */
struct StatisticsRecipe {
  std::shared_ptr<const statistics::Plan> plan;
  [[nodiscard]] bool empty() const noexcept { return !plan || plan->empty(); }
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
  /** The committed state's boundary reading (`monitor_reading`, `watch::reading_head`
      doubles and the occupancy buckets), for the monitor's launch and plane checks. */
  virtual std::vector<double> boundary_reading() {
    throw std::logic_error("this state takes no boundary readings");
  }
  /** Leave the in-stage temporal collar out of the stages, or put it back, dropping
      what was evaluated under the other setting (`_collar_changed`). */
  virtual void set_collar_quiet(bool /*quiet*/) {
    throw std::logic_error("this state cannot rest its collar");
  }
  /** The committed state's transverse edge reading (`state.edge_reading`, SEP 44): the
      faces' sums and the columns when `faces`, with the clip and totals of every node
      when `whole`, the totals when `totals`, a collar's strip when `strip` and its
      current when `current`. */
  virtual transverse::Reading edge_reading(bool /*faces*/, bool /*whole*/,
                                           bool /*totals*/, bool /*strip*/,
                                           bool /*current*/) {
    throw std::logic_error("this state takes no transverse edge readings");
  }
};

/** The boundary monitor a session owns (SEP 47 section 6.4): its configuration and,
    for a resumed run, the record it continues. */
struct MonitorPlan {
  watch::Config config;
  /** Whether the run resumes a monitored execution (`BoundaryMonitor.resume`). */
  bool resume = false;
  watch::Resumption record;
  /** Whether it resumes at a saved plane, and the stretches run since unwatched, m. */
  bool plane = false;
  std::vector<std::array<double, 2>> unwatched;
  /** A free-space run's transverse edge, and the `transverse` record it continues. */
  std::optional<transverse::Config> edge;
  transverse::Resumption edge_record;
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
  /** The bounded chunk buffers (SEP 47 section 5.3): the most attempts and periodic
      rows the session holds between two events. A full buffer yields a progress event
      carrying the chunk; it never grows. */
  std::size_t attempt_capacity = 4096;
  std::size_t row_capacity = 1024;
  /** A replayed schedule: its step count and total length, for `replay_tolerance`. */
  bool replay = false;
  std::size_t replay_steps = 0;
  double replay_length = 0.0;
  /** Attempts one march call makes at most (SEP 39 Q3), and its wall time, s. */
  std::size_t call_attempts = 256;
  double call_seconds = 0.1;
  /** The run's boundary monitor, or none. */
  std::optional<MonitorPlan> monitor;
};

/** A boundary-monitor verdict a session reports (SEP 47 section 6.4): the warning a
    service barrier delivers, or the stop a terminal failure carries. solprop formats
    its message from these numbers alone. */
struct Notice {
  enum class Kind : int {
    none = 0,
    guard = 1,      ///< a guard occupancy (`_judge`), at the launch or at a check
    removal = 2,    ///< the collar's removal at the end of a run (`finish`)
    crossing = 3,   ///< a replay's step passes the split collar's cap
    reading = 4,    ///< the split found nothing where the state holds light (`_found`)
    division = 5,   ///< light after a launch that held none, where solprop's monitor
                    ///< divides by the launch energy (`ZeroDivisionError`)
    transverse = 6, ///< the transverse edge's verdict, `edge`
  };
  Kind kind = Kind::none;
  /** For a guard: whether it is the launch's judgement. */
  bool launch = false;
  /** Where, m. */
  double z = 0.0;
  /** A guard's amplitude (`worst`), or the removed fraction. */
  double value = 0.0;
  /** A crossing: the cap and the longest step before the next check, m. */
  double cap = 0.0;
  double step = 0.0;
  /** The transverse edge's verdict. */
  transverse::Notice edge;
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
  /** A boundary monitor's stop, when it is one. */
  Notice notice;
};

/** One event of a batch: a batch holds exactly one, carrying the chunk. */
struct Event {
  EventKind kind = EventKind::progress;
  Observation observation = Observation::launch;
  double z = 0.0;
  std::int64_t plane_index = -1;
  /** Whether the event carries plane statistics, `Batch::plane_stats`. */
  bool has_stats = false;
  /** Whether the event's field can be exported while the batch is pending, and the
      committed-state generation it names. */
  bool ticket = false;
  std::uint64_t generation = 0;
  Outcome outcome = Outcome::none;
  Failure failure;
  /** A warning's verdict. */
  Notice notice;
};

/** Columns of one attempt row: its end, step, scaled norm, acceptance and the wall
    time since the previous event, s. */
constexpr std::size_t attempt_columns = 5;

/** What one `advance` reports (SEP 47 section 5.3, exactly-once chunked delivery): one
    event, and the chunk of attempts and periodic rows made since the previous event,
    moved out of the session's buffers and delivered once. A row's `step` indexes the
    chunk's attempts, counted from the chunk's first; it is negative for a row whose
    attempt an earlier chunk delivered: a row taken after a warning barrier, which
    delivered its step's attempt, is `-1`, the barrier chunk's last. */
struct Batch {
  std::uint64_t sequence = 0;
  Event event;
  std::vector<double> attempts;
  std::vector<double> row_z;
  std::vector<std::int64_t> row_step;
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
  /** Bytes the session's chunk buffers reserve: attempts and rows. */
  std::uint64_t buffer_bytes = 0;
};

/** A snapshot request's receipt (`solprop.backends.session.SnapshotReceipt`). */
struct SnapshotReceipt {
  std::uint64_t request = 0;
  bool coalesced = false;
};

/** One trajectory's native execution.

    Thread safety: `advance`, `acknowledge`, `export_field` and `close` serialise on
    the session's lock (a binding releases its interpreter lock before taking it);
    `status`, `request_stop` and `request_snapshot` take no lock and may be called from
    any thread at any time, an advance included. */
class Session {
public:
  Session(std::unique_ptr<Resident> resident, Method method, Plan plan, double z);
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) = delete;
  Session& operator=(Session&&) = delete;
  ~Session() = default;

  /** Run until a barrier, `work_limit` attempts (at least one), `deadline` seconds or
      a full chunk buffer, checked between attempts; report it as a pending batch. */
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
  /** Latch a snapshot for the next eligible accepted endpoint; any thread, no lock. A
      request while one is outstanding joins it. */
  SnapshotReceipt request_snapshot();
  /** The session's position and counters; any thread, no lock. */
  [[nodiscard]] Status status() const noexcept;
  [[nodiscard]] std::size_t elements() const;
  /** The boundary monitor's record (`BoundaryMonitor.summary`'s numbers), or none
      without a monitor; takes the session's lock. */
  [[nodiscard]] std::optional<watch::Record> monitor_record() const;
  /** The transverse edge's record (`TransverseEdge.summary`'s numbers) and its finish,
      or none without an edge; takes the session's lock. */
  [[nodiscard]] std::optional<std::pair<transverse::Record, transverse::Finish>>
  edge_record() const;
  /** Continue a resumed run's monitor record before the monitor watches the run
      (`DriverSession.continue_monitor`): a continuation the plan already holds is
      kept, and a session without a monitor has nothing to continue. Takes the
      session's lock; refused once the monitor watches. */
  void continue_monitor(const watch::Resumption& record, bool plane,
                        const std::vector<std::array<double, 2>>& unwatched,
                        const transverse::Resumption& edge_record = {});
  /** Release the engine, the leased state and every buffer; idempotent. Returns
      whether the session is closed: false only when its lock could not be taken, and
      then nothing was released. Waits for an advance in hand. */
  [[nodiscard]] bool close() noexcept;
  [[nodiscard]] bool closed() const noexcept {
    return closed_.load(std::memory_order_acquire);
  }

private:
  using Clock = std::chrono::steady_clock;

  void require_open() const;
  march::Result call(double target, double slack, std::size_t attempts, double seconds);
  void absorb(const march::Result& result, Clock::time_point started);
  [[nodiscard]] double position() const noexcept;
  [[nodiscard]] std::uint64_t cursor() const;
  [[nodiscard]] std::uint64_t host_step() const;
  [[nodiscard]] bool at_endpoint() const;
  [[nodiscard]] bool snapshot_latched() const noexcept;
  /** Whether the march left a step in flight in the state, which is then poisoned. */
  [[nodiscard]] bool poisoned() const noexcept;
  [[nodiscard]] std::size_t chunk_attempts() const noexcept;
  [[nodiscard]] bool chunk_full() const noexcept;
  void take_row();
  void drain(Batch& batch);
  Event observe(Batch& batch, Observation kind, std::int64_t plane);
  void progress(Batch& batch);
  void snapshot(Batch& batch);
  void terminal(Batch& batch, Outcome outcome, Failure failure);
  Batch issue(Batch batch, Outcome outcome = Outcome::none);
  void publish() noexcept;
  // The boundary monitor's actions (SEP 47 sections 5.3 and 6.4): each fills `batch`
  // and returns true where its point ends the advance -- a warning barrier, a stop or,
  // for `observe_plane`, the observation.
  void sync_collar();
  void warning(Batch& batch, const Notice& notice);
  void monitor_stop(Batch& batch, const Notice& notice);
  bool judged(Batch& batch, const watch::Judgement& judgement);
  bool refused_crossing(Batch& batch);
  bool attach(Batch& batch);
  bool endpoint(Batch& batch, double target, bool edge = true);
  bool observe_plane(Batch& batch, Observation kind, std::int64_t plane, bool next);
  bool plane_rest(Batch& batch);
  [[nodiscard]] const march::Schedule& schedule() const noexcept;

  std::unique_ptr<Resident> resident_;
  Method method_;
  Plan plan_;
  bool adams_ = false;
  int landing_ulps_ = 0;
  std::size_t elements_ = 0;

  // Lifecycle, under the lock; `state_` and `outcome_` also published for `status`.
  std::atomic<State> state_{State::ready};
  std::atomic<Outcome> outcome_{Outcome::none};
  std::uint64_t sequence_ = 0;
  bool pending_ = false;
  std::uint64_t pending_generation_ = 0;
  mutable std::mutex mutex_;

  // Controls any thread may set, and whether the session was closed, which they read.
  std::atomic<bool> stop_{false};
  std::atomic<bool> closed_{false};
  /** The snapshot latch: request count times two, plus one while one is latched. */
  std::atomic<std::uint64_t> latch_{0};
  /** The latch as the marches read it at each accepted endpoint (`Bounds::halt`). */
  std::atomic<bool> halt_{false};

  // The schedule's cursors and the committed state's counters.
  std::size_t next_plane_ = 0;
  bool launch_pending_ = true;
  std::uint64_t accepted_steps_ = 0;
  std::uint64_t natural_count_ = 0;
  std::uint64_t generation_ = 0;
  bool stepped_ = false;
  bool last_accepted_ = false;

  // The chunk since the previous event, bounded by the plan's capacities.
  std::vector<double> history_;
  Clock::time_point clock_;
  std::vector<double> row_z_;
  std::vector<std::int64_t> row_step_;
  std::vector<double> row_values_;
  std::size_t row_width_ = 0;

  // Accounting and structural counters, published for `status` (relaxed atomics; the
  // lock orders the writer).
  std::uint64_t attempts_ = 0;
  std::uint64_t rejected_ = 0;
  std::atomic<double> published_z_{0.0};
  std::atomic<std::size_t> published_plane_{0};
  std::atomic<std::uint64_t> published_attempts_{0};
  std::atomic<std::uint64_t> published_rejected_{0};
  std::atomic<std::uint64_t> published_generation_{0};
  std::atomic<std::uint64_t> march_calls_{0};
  std::atomic<std::uint64_t> rows_taken_{0};
  std::atomic<std::uint64_t> batches_{0};
  std::atomic<std::uint64_t> exports_{0};
  std::atomic<std::uint64_t> export_bytes_{0};
  std::atomic<std::size_t> published_width_{0};

  // The boundary monitor, whether it was attached after the launch plane, whether the
  // stages rest the collar, and where a warning barrier left the point it arose at.
  std::unique_ptr<watch::Monitor> monitor_;
  std::unique_ptr<transverse::Edge> edge_;
  bool attached_ = false;
  bool collar_quiet_ = false;
  enum class Continuation : int {
    none = 0,
    attach = 1,
    endpoint = 2,
    plane = 3,
    finish = 4,
    // The transverse edge warned at the launch, an endpoint or a plane.
    edge_launch = 5,
    edge_endpoint = 6,
    edge_plane = 7
  };
  Continuation continuation_ = Continuation::none;
  Observation continued_kind_ = Observation::plane;
  std::int64_t continued_plane_ = -1;
  bool continued_next_ = false;

  // The transverse edge's actions (session_edge.cpp), in the same manner: a warning
  // sets `continued` as the continuation, and `edge_proceed` serves the rest of its
  // point up to the next.
  [[nodiscard]] double next_step() const;
  transverse::Reading edge_reading(bool faces, bool whole, bool current);
  bool edge_judged(Batch& batch, const transverse::Notice& notice,
                   Continuation continued);
  bool edge_launch(Batch& batch);
  bool edge_endpoint(Batch& batch);
  bool edge_plane(Batch& batch);
  bool edge_proceed(Batch& batch, Continuation continued);
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
