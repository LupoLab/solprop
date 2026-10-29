// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/*
 * The exponential Adams march executed natively (solprop SEP 39 phase 2).
 *
 * solprop's `AdamsMarch` and its controller core (`solprop.backends.adams_core`) are
 * the reference; SEP 39 section 4 and the kernel contract's section 11 specify the
 * algorithm. The march here performs it for a run of attempts, returning to solprop
 * only at a host event: a reached or crossed target, an accepted step on which solprop
 * has work (a statistics row, a natural stop, a monitor check), the call's bound, the
 * budget, the floor, or a native error. Every number it reads comes in the `Policy`
 * solprop forms (`solprop.stepper.multistep.MarchPolicy`); it holds no default and no
 * constant of its own (native/AGENTS.md).
 *
 * Field arithmetic is the engine's (`Engine`): the resident engine's trial, accept and
 * startup over the native state, or, in tests, a script of norms. The decisions and
 * the landing rule are compiled strict (no contraction, no fast-math) in their own
 * translation unit, so that they round as solprop's Python does; the only elementary
 * function they call is the C library's `pow`.
 *
 * Host code only.
 */
namespace solprop_native::march {

/** The most history entries a march holds: the highest order and two more. */
constexpr std::size_t max_history = 8;
/** The most orders a trial scores: its own and its two neighbours. */
constexpr std::size_t max_orders = 3;

/** Every number the algorithm reads: solprop's `MarchPolicy`, field for field. */
struct Policy {
  std::size_t order = 0;
  bool variable_order = false;
  double safety = 0.0;
  double max_growth = 0.0;
  double min_factor = 0.0;
  double rejection_growth = 0.0;
  double hysteresis = 0.0;
  std::size_t restart_after = 0;
  double collar_stability = 0.0;
  std::vector<double> real_intervals;
  double collar_rate = 0.0;
  bool carrier = false;
  double carrier_cap = 0.0;
  bool has_budget = false;
  std::uint64_t max_calls = 0;
  bool equal_approach = false;
  bool interval_cap = false;
  double dz_min = 0.0;
  double dz_max = 0.0;
  double stretch = 0.0;
  double z_end = 0.0;
  std::vector<double> save_planes;
  std::vector<double> hard_kinks;
  std::vector<double> breakpoints;
  int landing_ulps = 0;
  bool interpolate = false;
  bool has_quiet_interval = false;
  double quiet_interval = 0.0;
};

/** Refuse a policy the march could not execute, naming the field. */
void validate(const Policy& policy);

/** What a history entry's sample is: an accepted state's or a predicted one's. */
enum class Identity : std::uint8_t { accepted = 0, predicted = 1 };

/** One history entry: its position, what it is, and its engine row. */
struct Entry {
  double z = 0.0;
  Identity identity = Identity::accepted;
  std::size_t row = 0;
};

/** The call ledger's counters (solprop's `CallLedger`, less its order list). */
struct Ledger {
  std::uint64_t startup = 0;
  std::uint64_t predictor = 0;
  std::uint64_t rejected = 0;
  std::uint64_t accepted_steps = 0;
  std::uint64_t rejected_steps = 0;
  std::uint64_t restarts = 0;
  std::uint64_t failure_restarts = 0;
  std::uint64_t dense_observations = 0;
  [[nodiscard]] std::uint64_t total() const { return startup + predictor; }
};

/** Everything the algorithm changes (SEP 39 section 4.2). */
struct State {
  double z = 0.0;
  double proposal = 0.0;
  std::size_t order = 1;
  std::vector<Entry> history;
  std::size_t rejections = 0;
  bool last_rejected = false;
  bool collar_was_quiet = false;
  Ledger ledger;
};

/** A trial's history, newest first, and the orders it scores. */
struct Stencil {
  std::array<double, max_history> nodes{};
  std::array<std::size_t, max_history> rows{};
  std::size_t held = 0;
  std::size_t order = 0;
  std::array<std::size_t, max_orders> orders{};
  std::size_t n_orders = 0;
};

/** What bound a landing: solprop's `LandingTarget`, in its priority order. */
enum class Target : int {
  none = 0,
  max_step = 1,
  interval_width = 2,
  hard_kink = 3,
  save_plane = 4,
  end = 5,
};

/** The step a landing allows (solprop's `Landing`). */
struct Landing {
  double dz = 0.0;
  Target target = Target::none;
  bool landed = false;
  double z_target = 0.0;
};

/** The step to take, where it ends and whether that is a hard kink. */
struct Approach {
  double dz = 0.0;
  double end = 0.0;
  bool kink = false;
};

/** The order to store, and the order and norm that set the next step. */
struct OrderChoice {
  std::size_t stored = 0;
  std::size_t control = 0;
  double norm = 0.0;
};

/** What a rejected trial leaves. */
struct Rejection {
  double factor = 0.0;
  std::size_t rejections = 0;
  bool restart = false;
};

// The decisions, solprop.backends.adams_core's functions (adams_control.cpp).
[[nodiscard]] double factor(const Policy& policy, double norm, std::size_t order,
                            double growth);
[[nodiscard]] Stencil stencil(const Policy& policy, double z, double dz,
                              const std::vector<Entry>& history,
                              std::size_t stored_order, bool last_rejected);
[[nodiscard]] double stable_factor(const Policy& policy, std::size_t order, double dz,
                                   bool quiet);
[[nodiscard]] double order_weight(const Policy& policy, double norm, std::size_t order,
                                  double dz, bool quiet);
[[nodiscard]] OrderChoice choose_order(const Policy& policy, const Stencil& stencil,
                                       const double* norms, double dz, bool quiet);
[[nodiscard]] Rejection rejection(const Policy& policy, double norm, std::size_t order,
                                  std::size_t rejections, std::size_t held);
[[nodiscard]] double carrier_capped(const Policy& policy, double z, double proposal,
                                    double phase_over);
[[nodiscard]] double quiet_capped(const Policy& policy, double proposal, bool quiet);
[[nodiscard]] double collar_reentry(const Policy& policy, double proposal,
                                    std::size_t order, bool was_quiet, bool quiet);
/** solprop's `stepper.schedule.landing`, the run's end `z_end` for this march. */
[[nodiscard]] Landing landing(const Policy& policy, double z, double dz, double z_end);
/** The next forced landing strictly beyond `z` (`adams_core.next_landing`). */
[[nodiscard]] bool next_landing(const Policy& policy, double z, double target,
                                double& ahead);
[[nodiscard]] Approach approach(const Policy& policy, double z, const Landing& stop,
                                double target);
/** `limit_step`'s proposal, `min(dz factor, dz_max)`; the caller tests the floor. */
[[nodiscard]] double proposal(const Policy& policy, double dz, double factor);
/** `solprop.backends.driver.reached`. */
[[nodiscard]] bool reached(double position, double target, int ulps);
/** The spacing of binary64 at a non-negative `x`: `math.ulp`, `numpy.spacing`. */
[[nodiscard]] double ulp(double x);

/** The field arithmetic a march hands its trials to. */
class Engine {
public:
  Engine() = default;
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) = delete;
  Engine& operator=(Engine&&) = delete;
  virtual ~Engine() = default;
  /** Evaluate the history's first sample at the accepted state at `z` into row 0. */
  virtual void start(double z) = 0;
  /** Form a trial from `z` over `dz`; write its scored orders' norms into `norms`. */
  virtual void trial(double z, double dz, const Stencil& stencil, double* norms) = 0;
  /** Adopt the held trial; the new sample takes row `free`. */
  virtual void accept(double dz, std::size_t free, const Stencil& stencil) = 0;
  /** The carrier chord's largest phase over `[z1, z2]`, rad. */
  virtual double chord_phase(double z1, double z2) = 0;
};

/** Why a call returned to solprop (SEP 39 section 4.8). */
enum class Reason : int {
  reached = 0,   ///< the target is reached, or crossed with planes interpolated
  host_step = 1, ///< an accepted step on which solprop has work
  monitor = 2,   ///< the monitor's distance or first check falls due
  bound = 3,     ///< the call's attempts or wall time are spent
  budget = 4,    ///< the next attempt would exceed the call budget
  floor = 5,     ///< a proposal fell below the floor
  error = 6,     ///< a native failure; the steps committed before it stand
};

/** Where a native failure happened. */
enum class Stage : int { none = 0, startup = 1, trial = 2, accept = 3 };

/** What class of solprop error a failure maps to. */
enum class Category : int { none = 0, convergence = 1, backend = 2, value = 3 };

/** The host's bounds for one call (SEP 39 section 4.8). */
struct Bounds {
  double target = 0.0;
  bool quiet = false;
  /** Accepted steps after which solprop has work; zero for none. */
  std::uint64_t host_step = 0;
  /** The monitor's distance test: whether it applies, its interval and last check. */
  bool monitor = false;
  bool first_check = false;
  double check_interval = 0.0;
  double last_check = 0.0;
  /** The call's own bound: its attempts and its wall time, s. */
  std::size_t max_attempts = 256;
  double max_seconds = 0.1;
};

/** Columns of one attempt's row. */
enum Column : std::size_t {
  end_column = 0,
  step_column = 1,
  norm_column = 2,
  accepted_column = 3,
  order_column = 4,
  wall_column = 5,
  row_columns = 6,
};

/** What a call returns, besides the state. */
struct Result {
  Reason reason = Reason::reached;
  /** One row per attempt, `row_columns` doubles each. */
  std::vector<double> rows;
  std::vector<std::size_t> orders; ///< each accepted step's order, in order
  std::uint64_t cost = 0;          ///< the refused attempt's calls (budget)
  double floor_proposal = 0.0;     ///< the proposal below the floor (floor)
  Stage stage = Stage::none;       ///< where the failure happened (error)
  Category category = Category::none;
  std::string message;
  /** The last accepted step: its anchor, step, order and nodes, for an observation. */
  bool stepped = false;
  double step_z = 0.0;
  double step_dz = 0.0;
  std::size_t step_order = 0;
  std::array<double, max_history> step_nodes{};
};

/** Advance `state` under `policy` by the algorithm of SEP 39 section 4, to an event. */
Result run(const Policy& policy, State& state, Engine& engine, const Bounds& bounds);

} // namespace solprop_native::march
