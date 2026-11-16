// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The native workspace's records (solprop SEP 52 section 5.3): the set-once data of a
// problem's `Setup` and the records one run starts from, with what the marches and the
// session read from them (`workspace.hpp` builds and runs the workspace).
//
// The binding converts Python's records into the structs here (each with its field
// table, `fields.hpp`), releases the interpreter lock and forwards; the logic is here,
// where the native tests reach it.
//
// Types: the set-once `Setup` and its parts (`ZTable`, `ZSlotsConfig`, `FlowGuard`,
// `RungeKuttaConfig`, `AdamsConfig`, `Occupancy`); a run's state-side tables
// (`MonitorBinding`, `EdgeBinding`); the run records mirroring
// `solprop.backends.records` (`RunRecordsRecord` and its parts, `HistoryRow`); and
// `RunStart`, what a run starts from. Functions: the `to_native` conversions,
// `fault_from` (a test's injected failure) and `validate` (the stage tables), defined
// in src/problem/workspace.cpp.

#include "solprop_native/general/general_state.hpp"
#include "solprop_native/march/adams_march.hpp"
#include "solprop_native/march/engines.hpp"
#include "solprop_native/march/low_storage.hpp"
#include "solprop_native/march/rk_march.hpp"
#include "solprop_native/march/stage_tables.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/fields.hpp"
#include "solprop_native/session/readings.hpp"
#include "solprop_native/session/statistics_plan.hpp"
#include "solprop_native/session/transverse.hpp"
#include "solprop_native/session/watch.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace solprop_native {

// ------------------------------------------------------------------ the set-once data

/** One general z table made resident at construction (solprop SEP 29 G1): its
    breakpoints, admitted span and coefficients `(n_edges - 1, n_terms, n_values)`. */
struct ZTable {
  TableKind kind = TableKind::phi;
  std::vector<double> edges;
  double lower = 0.0;
  double upper = 0.0;
  std::vector<Complex> coefficients;
  std::size_t n_terms = 0;
  std::size_t n_values = 0;
};

[[nodiscard]] constexpr auto describe(const ZTable* /*unused*/) {
  using E = fields::Extent;
  return fields::table<ZTable>(
      "ZTable", fields::value("kind", &ZTable::kind),
      fields::array("edges", &ZTable::edges, E::of("n_edges")),
      fields::value("lower", &ZTable::lower), fields::value("upper", &ZTable::upper),
      fields::array("coefficients", &ZTable::coefficients, E::of("n_edges", -1),
                    E::of("n_terms"), E::of("n_values")),
      fields::count("n_terms", &ZTable::n_terms),
      fields::count("n_values", &ZTable::n_values));
}

/** The z tables' position slots (`stage::ZSlots::configure`). */
struct ZSlotsConfig {
  bool phi = false;
  bool prefactor = false;
  std::size_t slots = 0;
  bool rate = false;
};

[[nodiscard]] constexpr auto describe(const ZSlotsConfig* /*unused*/) {
  using Z = ZSlotsConfig;
  return fields::table<Z>("ZSlots", fields::value("phi", &Z::phi),
                          fields::value("prefactor", &Z::prefactor),
                          fields::value("slots", &Z::slots),
                          fields::value("rate", &Z::rate));
}

/** A coupled flow's certificate and the guard's constants (solprop SEP 45 phase 4):
    each section's row-sum norm, each section's conditioning bound per active bin
    `(n_sections, n_bins)`, the Taylor scale, the most squarings and the rounding
    allowance, as solprop's `_flow_squarings` and `_check_flow_conditioning` read
    them. */
struct FlowGuard {
  std::vector<double> row_sum;
  std::vector<double> condition_bound;
  double taylor_scale = 0.0;
  int max_squarings = 0;
  double allowance = 0.0;
};

[[nodiscard]] constexpr auto describe(const FlowGuard* /*unused*/) {
  using F = FlowGuard;
  using E = fields::Extent;
  return fields::table<F>("FlowGuard",
                          fields::array("row_sum", &F::row_sum, E::of("n_sections")),
                          fields::array("condition_bound", &F::condition_bound,
                                        E::of("n_sections"), E::of("n_bins")),
                          fields::value("taylor_scale", &F::taylor_scale),
                          fields::value("max_squarings", &F::max_squarings),
                          fields::value("allowance", &F::allowance));
}

/** The Runge-Kutta trial's tableau nodes and error control (solprop SEP 45): the
    nodes, the tolerances, a weighted controller's weights (empty for none) and the
    tableau's dense-output weights (empty without dense output). */
struct RungeKuttaConfig {
  std::vector<double> nodes;
  double rtol = 0.0;
  double atol = 0.0;
  std::vector<double> weights;
  std::vector<double> dense_weights;
};

[[nodiscard]] constexpr auto describe(const RungeKuttaConfig* /*unused*/) {
  using R = RungeKuttaConfig;
  using E = fields::Extent;
  return fields::table<R>(
      "RungeKuttaConfig", fields::array("nodes", &R::nodes, E::of("n_stages")),
      fields::value("rtol", &R::rtol), fields::value("atol", &R::atol),
      fields::array("weights", &R::weights, E::any()),
      fields::array("dense_weights", &R::dense_weights, E::any()));
}

/** The resident Adams history (solprop SEP 16, SEP 39): its rows, whether trials carry
    the transported compensation, whether the steps are kept for interpolation, the
    fused (interaction) formulation, and the controller's scale. */
struct AdamsConfig {
  std::size_t slots = 0;
  bool compensated = false;
  bool interpolate = false;
  bool fused = false;
  double rtol = 0.0;
  double atol = 0.0;
  std::vector<double> weights;
};

[[nodiscard]] constexpr auto describe(const AdamsConfig* /*unused*/) {
  using A = AdamsConfig;
  using E = fields::Extent;
  return fields::table<A>(
      "AdamsConfig", fields::value("slots", &A::slots),
      fields::value("compensated", &A::compensated),
      fields::value("interpolate", &A::interpolate), fields::value("fused", &A::fused),
      fields::value("rtol", &A::rtol), fields::value("atol", &A::atol),
      fields::array("weights", &A::weights, E::any()));
}

/** The boundary monitor's weights where the state lives (solprop SEP 21), a run's: each
    run's monitor is its own (`RunRequest.for_run`). `split` says the monitor reads the
    split collar's crossing, whose walk-off buckets are the problem's (`Occupancy`). */
struct MonitorBinding {
  std::vector<double> weights;
  std::vector<double> column_weights;
  bool split = false;
};

[[nodiscard]] constexpr auto describe(const MonitorBinding* /*unused*/) {
  using M = MonitorBinding;
  using E = fields::Extent;
  return fields::table<M>("MonitorBinding",
                          fields::array("weights", &M::weights, E::any()),
                          fields::array("column_weights", &M::column_weights, E::any()),
                          fields::value("split", &M::split));
}

/** The split collar's walk-off bucket of each bin (solprop #440), the problem's: set
    once, and bound to the state by the first run whose monitor reads it. */
struct Occupancy {
  std::vector<unsigned char> buckets;
  std::size_t count = 0;
};

[[nodiscard]] constexpr auto describe(const Occupancy* /*unused*/) {
  using O = Occupancy;
  using E = fields::Extent;
  return fields::table<O>(
      "Occupancy",
      fields::array("buckets", &O::buckets, fields::Dtype::uint8, E::any()),
      fields::value("count", &O::count));
}

/** The transverse edge's tables where the state lives (solprop SEP 44), a run's: the
    faces', the clip's and the totals' weights, a reduction's synthesis (empty for
   none), the first row, and a requested collar's lines. */
struct EdgeBinding {
  std::vector<double> faces;
  std::vector<double> clip;
  std::vector<double> totals;
  std::vector<double> synthesis;
  std::size_t first_row = 0;
  std::optional<EdgeLines> lines;
};

[[nodiscard]] constexpr auto describe(const EdgeBinding* /*unused*/) {
  using B = EdgeBinding;
  using E = fields::Extent;
  return fields::table<B>("EdgeBinding", fields::array("faces", &B::faces, E::any()),
                          fields::array("clip", &B::clip, E::any()),
                          fields::array("totals", &B::totals, E::any()),
                          fields::array("synthesis", &B::synthesis, E::any()),
                          fields::value("first_row", &B::first_row),
                          fields::value("lines", &B::lines));
}

/** Everything fixed for a workspace beside the problem's groups (SEP 52 section 5.3).
    Each part is absent where the problem has none; `crossing_workers` is the threads a
    crossing's copy may use (zero keeps the state's default). */
struct Setup {
  std::optional<stage::Tables> stage_tables;
  std::optional<ZSlotsConfig> z_slots;
  std::vector<ZTable> z_tables;
  std::optional<FlowGuard> flow_guard;
  std::optional<RungeKuttaConfig> runge_kutta;
  std::optional<AdamsConfig> adams;
  std::optional<rk::low_storage::Config> low_storage;
  std::optional<Occupancy> occupancy;
  std::size_t crossing_workers = 0;
};

[[nodiscard]] constexpr auto describe(const Setup* /*unused*/) {
  using S = Setup;
  return fields::table<S>(
      "Setup", fields::value("stage_tables", &S::stage_tables),
      fields::value("z_slots", &S::z_slots), fields::value("z_tables", &S::z_tables),
      fields::value("flow_guard", &S::flow_guard),
      fields::value("runge_kutta", &S::runge_kutta), fields::value("adams", &S::adams),
      fields::value("low_storage", &S::low_storage),
      fields::value("occupancy", &S::occupancy),
      fields::value("crossing_workers", &S::crossing_workers));
}

// -------------------------------------------------------------------- the run records
// The structs below mirror `solprop.backends.records`, field for field, so that one
// table holds each to its Python dataclass; `to_native` forms what the marches and the
// session read from them.

/** `solprop.stepper.controller.PIController`. */
struct ControllerRecord {
  double rtol = 0.0;
  double atol = 0.0;
  double safety = 0.0;
  double max_factor = 0.0;
  double min_factor = 0.0;
  double beta1 = 0.0;
  double beta2 = 0.0;
  std::int64_t lag = 0;
  bool reject_free = false;
  std::optional<std::vector<double>> weights;
  std::int64_t estimator_order = 0;
  std::string region;
};

[[nodiscard]] constexpr auto describe(const ControllerRecord* /*unused*/) {
  using C = ControllerRecord;
  using E = fields::Extent;
  return fields::table<C>(
      "PIController", fields::value("rtol", &C::rtol), fields::value("atol", &C::atol),
      fields::value("safety", &C::safety), fields::value("max_factor", &C::max_factor),
      fields::value("min_factor", &C::min_factor), fields::value("beta1", &C::beta1),
      fields::value("beta2", &C::beta2), fields::value("lag", &C::lag),
      fields::value("reject_free", &C::reject_free),
      fields::array("weights", &C::weights, E::of("n_omega")),
      fields::value("estimator_order", &C::estimator_order),
      fields::value("region", &C::region));
}

/** `solprop.backends.records.RungeKuttaPolicy`. */
struct RungeKuttaPolicyRecord {
  ControllerRecord controller;
  double dz_min = 0.0;
  double dz_max = 0.0;
  bool auto_init = false;
  double stretch = 0.0;
  double z_end = 0.0;
  std::vector<double> save_planes;
  bool interpolate = false;
  std::vector<double> hard_kinks;
  std::vector<double> breakpoints;
  int landing_ulps = 0;
  double start_growth = 0.0;
  std::optional<double> quiet_interval;
};

[[nodiscard]] constexpr auto describe(const RungeKuttaPolicyRecord* /*unused*/) {
  using R = RungeKuttaPolicyRecord;
  return fields::table<R>(
      "RungeKuttaPolicy", fields::value("controller", &R::controller),
      fields::value("dz_min", &R::dz_min), fields::value("dz_max", &R::dz_max),
      fields::value("auto_init", &R::auto_init), fields::value("stretch", &R::stretch),
      fields::value("z_end", &R::z_end), fields::value("save_planes", &R::save_planes),
      fields::value("interpolate", &R::interpolate),
      fields::value("hard_kinks", &R::hard_kinks),
      fields::value("breakpoints", &R::breakpoints),
      fields::value("landing_ulps", &R::landing_ulps),
      fields::value("start_growth", &R::start_growth),
      fields::value("quiet_interval", &R::quiet_interval));
}

/** `solprop.stepper.schedule.FixedSchedule`: its steps and start; the edges are
    accumulated in step order (`FixedSchedule.edges`). */
struct ScheduleRecord {
  std::vector<double> steps;
  double z_start = 0.0;
};

[[nodiscard]] constexpr auto describe(const ScheduleRecord* /*unused*/) {
  using E = fields::Extent;
  return fields::table<ScheduleRecord>(
      "FixedSchedule", fields::array("steps", &ScheduleRecord::steps, E::of("n_steps")),
      fields::value("z_start", &ScheduleRecord::z_start));
}

/** `solprop.backends.records.PlanePlan`. */
struct PlanePlanRecord {
  double z = 0.0;
  std::vector<double> planes;
  std::size_t next_plane = 0;
  bool launch_pending = true;
  std::uint64_t step_planes = 0;
  bool dense = false;
  std::uint64_t period = 0;
  std::size_t replay_steps = 0;
  double replay_length = 0.0;
  std::size_t attempt_capacity = 0;
  std::size_t row_capacity = 0;
  std::size_t call_attempts = 0;
  double call_seconds = 0.0;
};

[[nodiscard]] constexpr auto describe(const PlanePlanRecord* /*unused*/) {
  using P = PlanePlanRecord;
  using E = fields::Extent;
  return fields::table<P>("PlanePlan", fields::value("z", &P::z),
                          fields::array("planes", &P::planes, E::of("n_z")),
                          fields::value("next_plane", &P::next_plane),
                          fields::value("launch_pending", &P::launch_pending),
                          fields::value("step_planes", &P::step_planes),
                          fields::value("dense", &P::dense),
                          fields::value("period", &P::period),
                          fields::value("replay_steps", &P::replay_steps),
                          fields::value("replay_length", &P::replay_length),
                          fields::value("attempt_capacity", &P::attempt_capacity),
                          fields::value("row_capacity", &P::row_capacity),
                          fields::value("call_attempts", &P::call_attempts),
                          fields::value("call_seconds", &P::call_seconds));
}

/** `solprop.backends.records.Resumption`: the numbers a resume reads by name. */
struct ResumptionRecord {
  bool record = false;
  bool collar_quiet = false;
  std::map<std::string, double> numbers;
  std::optional<std::vector<std::vector<double>>> unwatched;
  std::optional<std::vector<double>> resumed_at;
  std::optional<std::vector<std::vector<double>>> planes;
};

[[nodiscard]] constexpr auto describe(const ResumptionRecord* /*unused*/) {
  using R = ResumptionRecord;
  return fields::table<R>(
      "Resumption", fields::value("record", &R::record),
      fields::value("collar_quiet", &R::collar_quiet),
      fields::value("numbers", &R::numbers), fields::value("unwatched", &R::unwatched),
      fields::value("resumed_at", &R::resumed_at), fields::value("planes", &R::planes));
}

/** `solprop.backends.records.EdgeResumption`. */
struct EdgeResumptionRecord {
  bool record = false;
  std::map<std::string, double> numbers;
  std::optional<std::vector<double>> edge_baseline;
  std::optional<std::vector<double>> edge_max;
  std::optional<std::vector<double>> edge_max_raw;
  std::optional<std::int64_t> reads;
  std::optional<std::string> transverse_status;
};

[[nodiscard]] constexpr auto describe(const EdgeResumptionRecord* /*unused*/) {
  using R = EdgeResumptionRecord;
  return fields::table<R>("EdgeResumption", fields::value("record", &R::record),
                          fields::value("numbers", &R::numbers),
                          fields::value("edge_baseline", &R::edge_baseline),
                          fields::value("edge_max", &R::edge_max),
                          fields::value("edge_max_raw", &R::edge_max_raw),
                          fields::value("reads", &R::reads),
                          fields::value("transverse_status", &R::transverse_status));
}

/** `solprop.backends.records.MonitorContinuation`. */
struct ContinuationRecord {
  ResumptionRecord record;
  EdgeResumptionRecord edge;
  bool plane = false;
  std::vector<std::array<double, 2>> unwatched;
};

[[nodiscard]] constexpr auto describe(const ContinuationRecord* /*unused*/) {
  using C = ContinuationRecord;
  return fields::table<C>("MonitorContinuation", fields::value("record", &C::record),
                          fields::value("edge", &C::edge),
                          fields::value("plane", &C::plane),
                          fields::value("unwatched", &C::unwatched));
}

/** `solprop.backends.records.MonitorRecords`. */
struct MonitorRecordsRecord {
  watch::Config config;
  std::optional<transverse::Config> edge;
  std::optional<ContinuationRecord> continuation;
};

[[nodiscard]] constexpr auto describe(const MonitorRecordsRecord* /*unused*/) {
  using M = MonitorRecordsRecord;
  return fields::table<M>("MonitorRecords", fields::value("config", &M::config),
                          fields::value("edge", &M::edge),
                          fields::value("continuation", &M::continuation));
}

/** `solprop.backends.records.RunRecords`: the method's policy is the Adams march's or
    the Runge-Kutta march's, told apart by their fields. */
struct RunRecordsRecord {
  std::variant<march::Policy, RungeKuttaPolicyRecord> policy;
  std::optional<rk::low_storage::Config> low_storage;
  std::optional<ScheduleRecord> schedule;
  PlanePlanRecord planes;
  std::optional<MonitorRecordsRecord> monitor;
};

[[nodiscard]] constexpr auto describe(const RunRecordsRecord* /*unused*/) {
  using R = RunRecordsRecord;
  return fields::table<R>("RunRecords", fields::value("policy", &R::policy),
                          fields::value("low_storage", &R::low_storage),
                          fields::value("schedule", &R::schedule),
                          fields::value("planes", &R::planes),
                          fields::value("monitor", &R::monitor));
}

/** One persistent row of a resumed Adams history, `(n_slots, rows, n_omega)`. */
struct HistoryRow {
  std::string name;
  std::vector<Complex> values;
  std::size_t n_slots = 0;
  std::size_t n_rows = 0;
  std::size_t n_omega = 0;
};

[[nodiscard]] constexpr auto describe(const HistoryRow* /*unused*/) {
  using H = HistoryRow;
  using E = fields::Extent;
  return fields::table<H>("HistoryRow", fields::value("name", &H::name),
                          fields::array("values", &H::values, E::of("n_slots"),
                                        E::of("n_rows"), E::of("n_omega")),
                          fields::count("n_slots", &H::n_slots),
                          fields::count("n_rows", &H::n_rows),
                          fields::count("n_omega", &H::n_omega));
}

/** What one run starts from (`Workspace::start`): its records, the initial field and
    noise as the state takes them (`set_initial`), borrowed from the caller for the
   call, the march's state at the start (the Adams march's or the Runge-Kutta march's,
   none for a fresh state of its own), a resumed history's rows, the statistics plans,
   the monitor's launch readings of the input noise (its floor: energy, then four
   regions' shares; and the edge's), and the run's monitor weights and edge tables where
   the state lives. */
struct RunStart {
  RunRecordsRecord records;
  std::span<const Complex> field;
  std::span<const Complex> noise;
  std::optional<march::State> adams;
  std::optional<rk::State> runge_kutta;
  std::vector<HistoryRow> rows;
  std::shared_ptr<const statistics::Plan> plane_statistics;
  std::shared_ptr<const statistics::Plan> row_statistics;
  std::optional<std::array<double, 5>> floor;
  std::optional<transverse::Noise> edge_noise;
  std::optional<MonitorBinding> monitor;
  std::optional<EdgeBinding> edge;
};

[[nodiscard]] constexpr auto describe(const RunStart* /*unused*/) {
  using R = RunStart;
  using E = fields::Extent;
  return fields::table<R>(
      "RunStart", fields::value("records", &R::records),
      fields::array("field", &R::field, E::any()),
      fields::array("noise", &R::noise, E::any()), fields::value("adams", &R::adams),
      fields::value("runge_kutta", &R::runge_kutta), fields::value("rows", &R::rows),
      fields::value("plane_statistics", &R::plane_statistics),
      fields::value("row_statistics", &R::row_statistics),
      fields::value("floor", &R::floor), fields::value("edge_noise", &R::edge_noise),
      fields::value("monitor", &R::monitor), fields::value("edge", &R::edge));
}

/** The Runge-Kutta march's policy from its record (kernel contract section 12): the
    controller's gains, factors, lag and acceptance rule flattened in, and the
    equivalent norm's power `estimator_order + 1` and its reciprocal. */
[[nodiscard]] rk::Policy to_native(const RungeKuttaPolicyRecord& record);
/** A fixed schedule's steps and edges, accumulated in step order. */
[[nodiscard]] march::Schedule to_native(const ScheduleRecord& record);
/** A monitor record's resumption, each number by name; refuses an unknown name. */
[[nodiscard]] watch::Resumption to_native(const ResumptionRecord& record);
/** An edge record's resumption, likewise. */
[[nodiscard]] transverse::Resumption to_native(const EdgeResumptionRecord& record);

/** A test's injected failure by name: the stage (`startup`, `trial`, `trial_source`
    or `accept`) and the category (`convergence`, `value`, `backend`, `sigint` or
    `sigterm`). */
[[nodiscard]] engines::MarchFault
fault_from(const std::string& stage, std::size_t after, const std::string& category);

/** The stage tables' consistency (the checks `adams_stage_tables` makes). */
void validate(const stage::Tables& tables);

} // namespace solprop_native
