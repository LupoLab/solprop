// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/general/general_state.hpp"
#include "solprop_native/march/adams_march.hpp"
#include "solprop_native/march/engines.hpp"
#include "solprop_native/march/low_storage.hpp"
#include "solprop_native/march/rk_march.hpp"
#include "solprop_native/march/stage_tables.hpp"
#include "solprop_native/problem/problem.hpp"
#include "solprop_native/problem/run_records.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/session/session.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

/*
 * The native workspace (solprop SEP 52 section 5.3): one problem's state, built once
 * and reused across the runs of a scan, with every set-once table, guard and geometry
 * applied from the problem's `Setup`; the lease between it and the one session or
 * probe that uses it; the coupled flow's guard; the marches' construction from a run's
 * records; and the timing probe of section 5.5.
 */
namespace solprop_native {

// ------------------------------------------------------------------------ the
// workspace

/** Which march a workspace runs, from its setup: the Adams march with its history,
    the low-storage replay with its settings, or else the Runge-Kutta march. */
enum class MethodKind : int { adams = 0, runge_kutta = 1, low_storage = 2 };

class Workspace;

/** A session over a leased workspace (SEP 47): it keeps the lease until it closes, and
    reads what only the workspace knows (the history rows, the z table) for the
    session's snapshot and noise phase under the session's lock. */
class Run {
public:
  Run(Workspace& owner, std::unique_ptr<session::Session> session);
  ~Run();
  Run(const Run&) = delete;
  Run& operator=(const Run&) = delete;
  Run(Run&&) = delete;
  Run& operator=(Run&&) = delete;

  [[nodiscard]] session::Session& session() { return *session_; }

  /** A method-history snapshot at the pending ticket: the march's state, the accepted
      field and each persistent history row with its extents; none for a Runge-Kutta
      march or before any history (`HostAdamsMarch.snapshot`). */
  struct Snapshot {
    march::State state;
    std::vector<std::complex<double>> field;
    std::vector<std::string> names;
    std::vector<std::array<std::size_t, 3>> extents;
    std::vector<std::vector<std::complex<double>>> rows;
  };
  [[nodiscard]] std::optional<Snapshot> snapshot(std::uint64_t sequence,
                                                 std::uint64_t generation);
  /** The Adams march's state and the orders of its accepted steps from `since`, read
      between advances; none for a Runge-Kutta march. */
  [[nodiscard]] std::optional<std::pair<march::State, std::vector<std::size_t>>>
  march_state(std::size_t since);
  /** A z table's `Phi` at `z`: the noise phase at the ticket's plane. */
  [[nodiscard]] std::vector<Complex> noise_phase(std::uint64_t sequence,
                                                 std::uint64_t generation, double z);
  /** Close the session and release the lease; false (nothing released) when the
      session's lock could not be taken. */
  bool close() noexcept;
  [[nodiscard]] bool closed() const noexcept { return owner_ == nullptr; }

private:
  Workspace* owner_;
  std::unique_ptr<session::Session> session_;
};

/** The timing probe (SEP 52 section 5.5): trials of the workspace's march from a loaded
    start, committing nothing. A Runge-Kutta trial keeps the loaded state, and its end
    state reads back as an owning copy; the low-storage march's trial advances the
    discarded registers in place, its field is refused, and the workspace closes with
    the probe. */
class Probe {
public:
  Probe(Workspace& owner, std::vector<Complex> field, std::vector<Complex> noise,
        double z);
  ~Probe();
  Probe(const Probe&) = delete;
  Probe& operator=(const Probe&) = delete;
  Probe(Probe&&) = delete;
  Probe& operator=(Probe&&) = delete;

  /** One trial from `z` over `dz`: the Runge-Kutta march's scaled error norm, or the
      low-storage step's (0, or non-finite on a divergence). */
  double trial(double z, double dz);
  /** The last Runge-Kutta trial's end state, as a kept step would be, the loaded
      state loaded again after it; refused for the low-storage march. */
  [[nodiscard]] std::vector<Complex> field();
  /** Trials taken. */
  [[nodiscard]] std::size_t trials() const noexcept { return trials_; }
  /** The source evaluations the trials made: the low-storage step's four passes each;
      the state's own count of a Runge-Kutta trial's stages. */
  [[nodiscard]] std::size_t evaluations() const;
  [[nodiscard]] bool low_storage() const noexcept { return low_storage_; }
  [[nodiscard]] double z() const noexcept { return z_; }
  void close() noexcept;
  [[nodiscard]] bool closed() const noexcept { return owner_ == nullptr; }

private:
  Workspace* owner_;
  std::vector<Complex> field_;
  std::vector<Complex> noise_;
  double z_;
  bool low_storage_;
  std::optional<double> length_;
  std::size_t trials_ = 0;
  std::size_t evaluations_at_load_ = 0;
  std::unique_ptr<rk::Engine> engine_;
};

/** One problem's state and every set-once table, guard and geometry (SEP 52 section
    5.3). It holds no march state: each run's comes in its `RunStart`. */
class Workspace {
public:
  explicit Workspace(Problem problem);
  ~Workspace();
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  Workspace(Workspace&&) = delete;
  Workspace& operator=(Workspace&&) = delete;

  // -- the setup, applied at construction (each part is public for the native tests)
  void set_stage_tables(stage::Tables tables);
  void configure_z_slots(const ZSlotsConfig& slots);
  void upload_z_table(const ZTable& table);
  void configure_flow_guard(FlowGuard guard);
  void configure_runge_kutta(RungeKuttaConfig config);
  void configure_adams(const AdamsConfig& config);
  void configure_low_storage(rk::low_storage::Config config);
  void set_occupancy(std::optional<Occupancy> occupancy);
  /** Fail the resident engine's `stage` after `after` more passes (a test hook). */
  void inject_fault(engines::MarchFault fault);
  /** Bind a host source (solprop SEP 49), or unbind with null; CPU builds only. */
  void bind_host_source(std::shared_ptr<HostSource> source);

  // -- the lease (SEP 47 section 5.2): one session or probe at a time
  void lease(const void* holder);
  void release(const void* holder) noexcept;
  void require_unleased() const;
  [[nodiscard]] bool leased() const noexcept { return holder_ != nullptr; }

  /** The state, open and not leased: every mutation and every live read. */
  GeneralState& state();
  /** The state's fixed facts, which a lease leaves readable. */
  [[nodiscard]] GeneralState& built() const;
  /** The state `holder` leases, for its own reads under its lock. */
  [[nodiscard]] GeneralState& leased_state(const void* holder) const;
  [[nodiscard]] bool closed() const noexcept;
  /** Close the session or probe that leases the workspace first, then release the
      state; a session that cannot close keeps it. */
  void close() noexcept;

  // -- the trial and the coupled flow's guard
  /** A Runge-Kutta trial whose stage rows the state forms from the stage tables at
      `z + c_i dz` and the step's end (solprop SEP 45 phase 2): the partials. */
  std::vector<double> staged_trial(double z, double dz, bool prepare_only);
  /** `_flow_trial`'s guards, transcribed (solprop SEP 45 phase 4): a set of actions is
      held for one (section, step); a length whose exponent needs too many squarings
      is refused before forming, and a measured conditioning above the certificate's
      after, each as a `BackendRefusal`. */
  void guard_flow(std::size_t section, double dz, const std::vector<double>& nodes);
  /** Forget the held actions' key (a test forming its own). */
  void forget_flow_key() noexcept { flow_key_.reset(); }

  // -- the marches
  [[nodiscard]] MethodKind kind() const noexcept;
  /** One call of the Adams march with the policy and schedule a run's records gave
      (the decision traces' entry point). */
  march::Result run_adams(march::State& state, const march::Bounds& bounds);
  /** One call of the Runge-Kutta march, likewise. */
  rk::Result run_rk(rk::State& state, const march::Bounds& bounds);
  /** Take a run's march policy and schedule from its records without starting it (the
      decision traces' configuration). */
  void configure_march(const RunRecordsRecord& records);

  // -- runs and probes
  /** Load a start as a fresh workspace's (SEP 22 section 4.7): the field and noise,
      the collar back in the stages, no recycled stage and no monitor bound. */
  void load(std::span<const Complex> field, std::span<const Complex> noise);
  /** Start one run from `start` (SEP 52 section 5.3), leasing the workspace to it. */
  std::unique_ptr<Run> start(RunStart start);
  /** Load `field` and `noise` at `z` and return a probe of the configured march. */
  std::unique_ptr<Probe> probe(std::vector<Complex> field, std::vector<Complex> noise,
                               double z);
  /** A probe from a run's start (SEP 52 section 5.5): its field, noise and position,
      and, for the low-storage replay, the policy's landings and the schedule its table
      keys come from. Its march state, history and monitor are not read. */
  std::unique_ptr<Probe> probe(const RunStart& start);

  // -- what the old seam's accessors read
  [[nodiscard]] const std::optional<stage::Tables>& stage_tables() const noexcept {
    return stage_tables_;
  }
  [[nodiscard]] stage::ZSlots& z_slots() noexcept { return z_slots_; }
  [[nodiscard]] std::size_t elements() const noexcept { return elements_; }
  [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
  [[nodiscard]] std::size_t stages() const noexcept { return stages_; }
  /** A trial's stage rows from the tables (`stage::row`, `trial`, `sample`). */
  stage::Trial stage_trial(double z0, double z1);
  [[nodiscard]] std::vector<double> stage_rate(double z) const;
  /** The z table's `Phi` at `z`, through the one slot cache, for `holder`. */
  std::vector<Complex> table_phi(const void* holder, double z);
  /** The configured march's engines over the state, for an engine-level test: the
      Runge-Kutta (or low-storage) engine and the Adams engine. With `dense`, a
      Runge-Kutta engine's extensions go to a dense output it holds there, as a dense
      run's do. */
  std::unique_ptr<rk::Engine>
  trial_engine(std::unique_ptr<session::Dense>* dense = nullptr);
  /** Whether the configured Runge-Kutta trial has the tableau's dense weights. */
  [[nodiscard]] bool extends() const noexcept {
    return !low_storage_ && runge_kutta_ && !runge_kutta_->dense_weights.empty();
  }
  /** The low-storage replay's engine over the state, its type kept for a test that
      reads its squared norms. */
  std::unique_ptr<engines::ResidentLowStorage> low_storage_engine();
  std::unique_ptr<march::Engine> adams_engine();

private:
  friend class Run;
  friend class Probe;
  void apply(const Setup& setup);
  void require_built() const;
  void require_open() const;
  void bind_run_monitor(const MonitorBinding& monitor,
                        const std::optional<EdgeBinding>& edge);
  /** The march a session runs: from the run's policy and schedule and the stage tables,
      its engine over the state, its dense output when `dense`, and its state at the
      start. */
  session::Method method(MethodKind kind, std::variant<march::State, rk::State> start,
                         bool dense);
  std::unique_ptr<Run> launch(session::Method method, session::Plan plan, double z);

  std::size_t columns_ = 0;
  std::size_t elements_ = 0;
  std::size_t stages_ = 0;
  std::shared_ptr<GeneralState> state_;
  std::optional<stage::Tables> stage_tables_;
  stage::Anchor stage_anchor_;
  stage::ZSlots z_slots_;
  std::optional<FlowGuard> flow_guard_;
  std::optional<std::pair<std::size_t, double>> flow_key_;
  std::optional<RungeKuttaConfig> runge_kutta_;
  std::optional<AdamsConfig> adams_;
  std::optional<rk::low_storage::Config> low_storage_;
  std::optional<Occupancy> occupancy_;
  bool occupancy_bound_ = false;
  std::optional<march::Policy> march_policy_;
  march::Schedule march_schedule_;
  bool march_fused_ = false;
  bool march_compensated_ = false;
  bool march_interpolate_ = false;
  std::optional<rk::Policy> rk_policy_;
  march::Schedule rk_schedule_;
  std::optional<engines::MarchFault> fault_;
  const void* holder_ = nullptr;
  Run* run_ = nullptr;
  Probe* probe_ = nullptr;
};

} // namespace solprop_native
