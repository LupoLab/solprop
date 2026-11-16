// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/workspace.hpp"

#include "solprop_native/dense_output.hpp"
#include "solprop_native/refuse.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace solprop_native {

namespace {

/** Python's `format(value, f".{precision}g")`, which follows C's `%g`. */
std::string py_g(const double value, const int precision) {
  std::array<char, 64> buffer{};
  std::snprintf(buffer.data(), buffer.size(), "%.*g", precision, value);
  return buffer.data();
}

/** Set the optional `name` names in `numbers`; refuse a name `known` lacks. */
template <class Slots>
void numbers_into(const std::map<std::string, double>& numbers, const Slots& known,
                  const char* record) {
  for (const auto& [name, value] : numbers) {
    const auto found = std::ranges::find_if(
        known, [&name](const auto& slot) { return name == slot.first; });
    if (found == known.end())
      refuse(record, " holds an unknown number '", name, "'");
    *found->second = value;
  }
}

} // namespace

// ----------------------------------------------------------------- the run records

rk::Policy to_native(const RungeKuttaPolicyRecord& record) {
  rk::Policy policy;
  auto& landings = policy.landings;
  landings.dz_min = record.dz_min;
  landings.dz_max = record.dz_max;
  landings.stretch = record.stretch;
  landings.z_end = record.z_end;
  landings.save_planes = record.save_planes;
  landings.hard_kinks = record.hard_kinks;
  landings.breakpoints = record.breakpoints;
  landings.landing_ulps = record.landing_ulps;
  landings.interpolate = record.interpolate;
  landings.has_quiet_interval = record.quiet_interval.has_value();
  landings.quiet_interval = record.quiet_interval.value_or(0.0);
  const auto& controller = record.controller;
  if (controller.estimator_order < 0 || controller.lag < 0)
    refuse("a Runge-Kutta controller's lag and estimator order are not negative");
  policy.safety = controller.safety;
  policy.beta1 = controller.beta1;
  policy.beta2 = controller.beta2;
  policy.min_factor = controller.min_factor;
  policy.max_factor = controller.max_factor;
  policy.lag = static_cast<std::size_t>(controller.lag);
  policy.reject_free = controller.reject_free;
  // `_records.rk_policy`: the equivalent norm's power and its reciprocal, as solprop
  // forms them from the estimator's order.
  const auto power = static_cast<double>(controller.estimator_order + 1);
  policy.power = power;
  policy.exponent = 1.0 / power;
  policy.start_growth = record.start_growth;
  policy.auto_init = record.auto_init;
  policy.dz_min = record.dz_min;
  policy.dz_max = record.dz_max;
  if (policy.landings.breakpoints.size() < 2 || !(policy.dz_max >= policy.dz_min))
    refuse("the Runge-Kutta march's landings are invalid");
  return policy;
}

march::Schedule to_native(const ScheduleRecord& record) {
  // `FixedSchedule.edges`: z_start + concatenate(([0.0], cumsum(steps))), the sum
  // accumulated in step order.
  march::Schedule schedule;
  schedule.steps = record.steps;
  schedule.edges.reserve(record.steps.size() + 1);
  double sum = 0.0;
  schedule.edges.push_back(record.z_start + sum);
  for (const double step : record.steps) {
    sum += step;
    schedule.edges.push_back(record.z_start + sum);
  }
  return schedule;
}

watch::Resumption to_native(const ResumptionRecord& record) {
  watch::Resumption out;
  out.record = record.record;
  out.collar_quiet = record.collar_quiet;
  const std::array<std::pair<const char*, std::optional<double>*>, 18> known{{
      {"launch_energy", &out.launch_energy},
      {"launch_guard_low", &out.launch_guard_low},
      {"launch_guard_high", &out.launch_guard_high},
      {"launch_collar", &out.launch_collar},
      {"guard_low_max", &out.guard_low_max},
      {"guard_high_max", &out.guard_high_max},
      {"guard_low_at", &out.guard_low_at},
      {"guard_high_at", &out.guard_high_at},
      {"collar_max", &out.collar_max},
      {"raw_removed_fraction", &out.raw_removed_fraction},
      {"removed_signed", &out.removed_signed},
      {"removed_fraction", &out.removed_fraction},
      {"floor_removed_fraction", &out.floor_removed_fraction},
      {"quiet_checks", &out.quiet_checks},
      {"raw_guard_max", &out.raw_guard_max},
      {"steps", &out.steps},
      {"skipped", &out.skipped},
      {"collar_left_out_checks", &out.collar_left_out_checks},
  }};
  numbers_into(record.numbers, known, "a resumed boundary record");
  if (record.unwatched) {
    std::vector<std::array<double, 2>> stretches;
    for (const auto& pair : *record.unwatched) {
      if (pair.size() != 2)
        refuse("an unwatched stretch is a start and an end");
      stretches.push_back({pair[0], pair[1]});
    }
    out.unwatched = std::move(stretches);
  }
  out.resumed_at = record.resumed_at;
  if (record.planes) {
    if (record.planes->size() != watch::plane_columns)
      refuse("a monitor record's plane rows have five columns");
    out.has_planes = true;
    for (std::size_t column = 0; column < watch::plane_columns; ++column)
      out.planes[column] = (*record.planes)[column];
  }
  return out;
}

transverse::Resumption to_native(const EdgeResumptionRecord& record) {
  transverse::Resumption out;
  out.record = record.record;
  // `_records.edge_resumption`: a record that is none carries nothing else.
  if (!record.record)
    return out;
  const std::array<std::pair<const char*, std::optional<double>*>, 15> known{{
      {"launch_energy", &out.launch_energy},
      {"noise_allowance", &out.noise_allowance},
      {"raw_ramp_removed", &out.raw_ramp_removed},
      {"noise_ramp_removed", &out.noise_ramp_removed},
      {"noise_share_in_ramp", &out.noise_share_in_ramp},
      {"declared_share_in_ramp", &out.declared_share_in_ramp},
      {"spatial_removed", &out.spatial_removed},
      {"current", &out.current},
      {"current_baseline", &out.current_baseline},
      {"strip_baseline", &out.strip_baseline},
      {"strip_floor", &out.strip_floor},
      {"strip_max", &out.strip_max},
      {"returned_max", &out.returned_max},
      {"largest_gap", &out.largest_gap},
      {"warned_at", &out.warned_at},
  }};
  numbers_into(record.numbers, known, "a resumed transverse record");
  out.edge_baseline = record.edge_baseline;
  out.edge_max = record.edge_max;
  out.edge_max_raw = record.edge_max_raw;
  out.reads = record.reads.value_or(0);
  out.status = record.transverse_status;
  return out;
}

engines::MarchFault fault_from(const std::string& stage, const std::size_t after,
                               const std::string& category) {
  engines::MarchFault fault;
  // `trial_source` fails inside the trial's source evaluation, its prediction in the
  // state.
  fault.in_source = stage == "trial_source";
  if (stage == "startup")
    fault.stage = march::Stage::startup;
  else if (stage == "trial" || fault.in_source)
    fault.stage = march::Stage::trial;
  else if (stage == "accept")
    fault.stage = march::Stage::accept;
  else
    throw std::invalid_argument(
        "a march fault's stage is startup, trial, trial_source or accept");
  if (category == "convergence")
    fault.category = march::Category::convergence;
  else if (category == "value")
    fault.category = march::Category::value;
  else if (category == "backend")
    fault.category = march::Category::backend;
  else if (category == "sigint")
    fault.signal = SIGINT;
  else if (category == "sigterm")
    fault.signal = SIGTERM;
  else
    throw std::invalid_argument("a march fault's category is convergence, value, "
                                "backend, sigint or sigterm");
  fault.after = after;
  return fault;
}

void validate(const stage::Tables& tables) {
  using stage::Terms;
  const bool expansion = tables.phi == Terms::expansion;
  const auto known_terms = [](const Terms terms) {
    return terms == Terms::constant || terms == Terms::expansion ||
           terms == Terms::table;
  };
  const bool known = known_terms(tables.phi) && known_terms(tables.prefactor);
  if (static_cast<int>(tables.chord) < static_cast<int>(stage::Chord::none) ||
      static_cast<int>(tables.chord) > static_cast<int>(stage::Chord::table))
    refuse("the stage tables are inconsistent: chord kind ",
           static_cast<int>(tables.chord), " is unknown");
  if (!known || tables.n_stages == 0 || !tables.density.present() ||
      (expansion && (!tables.phi_weights.present() || !tables.phi_rate.present() ||
                     tables.phi_rate.n_values != tables.phi_weights.n_values)) ||
      (tables.prefactor == Terms::expansion && !tables.prefactor_weights.present()) ||
      (tables.kerr == stage::KerrDensity::table && !tables.kerr_density.present()) ||
      tables.plasma_densities.size() != tables.plasma_preionisation.size() ||
      (tables.chord == stage::Chord::expansion &&
       (!expansion || tables.chord_terms.size() != tables.phi_weights.n_values)) ||
      (tables.chord == stage::Chord::table &&
       (!tables.chord_table.present() || tables.chord_table.n_values != 1)) ||
      (tables.flow.present() && tables.flow.edges.size() < 2) ||
      (tables.transform.present() && tables.transform.edges.size() < 2))
    refuse("the stage tables are inconsistent");
  stage::require_scalar_tables(tables);
}

// ------------------------------------------------------------------------ the run

Run::Run(Workspace& owner, std::unique_ptr<session::Session> session)
    : owner_(&owner), session_(std::move(session)) {
  owner.lease(this);
  owner.run_ = this;
}

Run::~Run() { static_cast<void>(close()); }

bool Run::close() noexcept {
  if (owner_ == nullptr)
    return true;
  if (!session_->close())
    return false;
  owner_->release(this);
  owner_->run_ = nullptr;
  owner_ = nullptr;
  return true;
}

std::optional<Run::Snapshot> Run::snapshot(const std::uint64_t sequence,
                                           const std::uint64_t generation) {
  std::optional<Snapshot> out;
  session_->read(
      sequence, generation,
      [&](const session::Method& method, session::Resident& resident) {
        const auto* adams = std::get_if<session::AdamsMethod>(&method);
        if (adams == nullptr || adams->state.history.empty())
          return;
        Snapshot snapshot;
        snapshot.state = adams->state;
        snapshot.field.resize(resident.elements());
        resident.field_to_host(reinterpret_cast<double*>(snapshot.field.data()));
        if (owner_ == nullptr)
          throw session::LifecycleError("the run is closed");
        const auto& general = owner_->leased_state(this);
        snapshot.names = general.adams_row_names();
        for (const auto& name : snapshot.names) {
          const auto extent = general.adams_row_extents(name);
          std::vector<std::complex<double>> row(extent[0] * extent[1] * extent[2]);
          general.adams_rows_to_host(name, row.data(), row.size());
          snapshot.extents.push_back(extent);
          snapshot.rows.push_back(std::move(row));
        }
        out = std::move(snapshot);
      });
  return out;
}

std::optional<std::pair<march::State, std::vector<std::size_t>>>
Run::march_state(const std::size_t since) {
  std::optional<std::pair<march::State, std::vector<std::size_t>>> out;
  session_->read(0, 0,
                 [&](const session::Method& method, session::Resident& /*resident*/) {
                   const auto* adams = std::get_if<session::AdamsMethod>(&method);
                   if (adams == nullptr)
                     return;
                   const std::size_t first = std::min(since, adams->orders.size());
                   std::vector<std::size_t> orders(
                       adams->orders.begin() + static_cast<std::ptrdiff_t>(first),
                       adams->orders.end());
                   out.emplace(adams->state, std::move(orders));
                 });
  return out;
}

std::vector<Complex> Run::noise_phase(const std::uint64_t sequence,
                                      const std::uint64_t generation, const double z) {
  std::vector<Complex> values;
  session_->read(
      sequence, generation,
      [&](const session::Method& /*method*/, session::Resident& /*resident*/) {
        if (owner_ == nullptr)
          throw session::LifecycleError("the run is closed");
        values = owner_->table_phi(this, z);
      });
  return values;
}

// ---------------------------------------------------------------------- the probe

Probe::Probe(Workspace& owner, std::vector<Complex> field, std::vector<Complex> noise,
             const double z)
    : owner_(&owner), field_(std::move(field)), noise_(std::move(noise)), z_(z),
      low_storage_(owner.low_storage_.has_value()) {
  owner.load(field_, noise_);
  engine_ = owner.trial_engine();
  evaluations_at_load_ = owner.built().rk_source_evaluations();
  owner.lease(this);
  owner.probe_ = this;
}

Probe::~Probe() { close(); }

std::size_t Probe::evaluations() const {
  if (low_storage_)
    return trials_ * rk::low_storage::evaluations;
  if (owner_ == nullptr)
    return 0;
  return owner_->built().rk_source_evaluations() - evaluations_at_load_;
}

double Probe::trial(const double z, const double dz) {
  if (owner_ == nullptr)
    throw std::logic_error("the probe is closed");
  if (!(dz > 0.0) || !std::isfinite(z))
    refuse("a probe's trial takes a finite position and a positive step");
  const double norm = engine_->trial(z, dz);
  ++trials_;
  if (!low_storage_)
    length_ = dz;
  return norm;
}

std::vector<Complex> Probe::field() {
  if (owner_ == nullptr)
    throw std::logic_error("the probe is closed");
  if (low_storage_)
    refuse<BackendRefusal>("the low-storage march accepts its step in place and has "
                           "no trial field to read back");
  if (!length_)
    refuse<BackendRefusal>("take a trial before reading its field");
  // The trial lives in the state, which only finishing the step there reads back: the
  // loaded state is kept first and loaded again after, so that nothing is committed.
  auto& state = owner_->built();
  std::vector<Complex> kept(owner_->elements());
  state.field_to_host(kept);
  state.finish(*length_);
  std::vector<Complex> out(owner_->elements());
  state.field_to_host(out);
  state.unbind_monitor();
  state.set_initial(kept, noise_);
  state.set_absorber_quiet(false);
  state.clear_fsal();
  length_.reset();
  return out;
}

void Probe::close() noexcept {
  if (owner_ == nullptr)
    return;
  engine_.reset();
  Workspace* owner = owner_;
  owner->release(this);
  owner->probe_ = nullptr;
  owner_ = nullptr;
  // The low-storage trial overwrote the registers: nothing is left to run from.
  if (low_storage_)
    owner->close();
}

// ------------------------------------------------------------------ the workspace

Workspace::Workspace(Problem problem)
    : columns_(problem.n_columns()),
      elements_(problem.n_columns() * problem.dims.n_omega),
      stages_(problem.stepper.n_stages) {
  problem.validate();
  const auto setup = std::move(problem.setup);
  state_ = std::make_shared<GeneralState>(std::move(problem));
  if (setup)
    apply(*setup);
}

Workspace::~Workspace() { close(); }

void Workspace::apply(const Setup& setup) {
  if (setup.z_slots)
    configure_z_slots(*setup.z_slots);
  for (const auto& table : setup.z_tables)
    upload_z_table(table);
  if (setup.stage_tables)
    set_stage_tables(*setup.stage_tables);
  if (setup.flow_guard)
    configure_flow_guard(*setup.flow_guard);
  if (setup.runge_kutta)
    configure_runge_kutta(*setup.runge_kutta);
  if (setup.adams)
    configure_adams(*setup.adams);
  if (setup.low_storage)
    configure_low_storage(*setup.low_storage);
  set_occupancy(setup.occupancy);
  if (setup.crossing_workers != 0)
    state().set_crossing_workers(setup.crossing_workers);
}

void Workspace::require_built() const {
  if (closed())
    throw std::logic_error("native state is closed");
}

void Workspace::require_unleased() const {
  if (holder_ != nullptr)
    throw std::logic_error("native state is leased by an execution session");
}

void Workspace::require_open() const {
  require_built();
  require_unleased();
}

GeneralState& Workspace::state() {
  require_open();
  return *state_;
}

GeneralState& Workspace::built() const {
  require_built();
  return *state_;
}

GeneralState& Workspace::leased_state(const void* holder) const {
  if (holder_ != holder || holder == nullptr)
    throw std::logic_error("native state is not leased by this session");
  return built();
}

bool Workspace::closed() const noexcept { return !state_ || state_->closed(); }

void Workspace::lease(const void* holder) {
  require_open();
  holder_ = holder;
}

void Workspace::release(const void* holder) noexcept {
  if (holder_ == holder)
    holder_ = nullptr;
}

void Workspace::close() noexcept {
  if (run_ != nullptr && !run_->close())
    return; // a session that did not close still uses the state: keep it, leased
  if (probe_ != nullptr)
    probe_->close();
  if (state_) {
    state_->close();
    state_.reset();
  }
}

void Workspace::set_stage_tables(stage::Tables tables) {
  require_open();
  validate(tables);
  if (tables.n_stages != stages_)
    throw std::invalid_argument("the stage tables name another tableau's stages");
  // A z table's rows are its slots: the tables read the state's one cache.
  const bool phi_table = tables.phi == stage::Terms::table;
  const bool prefactor_table = tables.prefactor == stage::Terms::table;
  if (phi_table != z_slots_.phi || prefactor_table != z_slots_.prefactor)
    throw std::invalid_argument("the stage tables and the z table's slots disagree");
  tables.z_slots = nullptr;
  tables.z_state = nullptr;
  if (phi_table || prefactor_table) {
    tables.z_slots = &z_slots_;
    tables.z_state = state_.get();
  }
  stage_tables_ = std::move(tables);
  stage_anchor_ = {};
}

void Workspace::configure_z_slots(const ZSlotsConfig& slots) {
  require_open();
  z_slots_.configure(slots.phi, slots.prefactor, slots.slots, slots.rate);
}

void Workspace::upload_z_table(const ZTable& table) {
  auto& state = this->state();
  if (table.edges.size() < 2)
    refuse("a z table needs at least two breakpoints");
  state.table_upload(
      table.kind, table.edges, table.lower, table.upper,
      reinterpret_cast<const std::complex<double>*>(table.coefficients.data()),
      table.edges.size() - 1, table.n_terms, table.n_values);
}

void Workspace::configure_flow_guard(FlowGuard guard) {
  require_open();
  const std::size_t sections = guard.row_sum.size();
  if (sections == 0 || guard.condition_bound.size() % sections != 0)
    throw std::invalid_argument("the flow guard's certificate is one row a section");
  flow_guard_ = std::move(guard);
  flow_key_.reset();
}

void Workspace::configure_runge_kutta(RungeKuttaConfig config) {
  require_open();
  if (config.nodes.size() != stages_)
    throw std::invalid_argument("the march's nodes are the tableau's stages");
  runge_kutta_ = std::move(config);
}

void Workspace::configure_adams(const AdamsConfig& config) {
  auto& state = this->state();
  state.adams_allocate(config.slots, config.compensated, config.interpolate);
  state.adams_scale(config.rtol, config.atol, config.weights);
  adams_ = config;
  march_fused_ = config.fused;
  march_compensated_ = config.compensated;
  march_interpolate_ = config.interpolate;
}

void Workspace::configure_low_storage(const rk::low_storage::Config config) {
  require_open();
  if (config.flows > 2)
    throw std::invalid_argument("the low-storage march holds at most two flows");
  low_storage_ = config;
}

void Workspace::set_occupancy(std::optional<Occupancy> occupancy) {
  require_open();
  if (occupancy && (occupancy->buckets.empty() || occupancy->count == 0))
    refuse("a split collar's occupancy needs its buckets and their count");
  occupancy_ = std::move(occupancy);
  occupancy_bound_ = false;
}

void Workspace::inject_fault(const engines::MarchFault fault) {
  require_open();
  fault_ = fault;
}

void Workspace::bind_host_source(std::shared_ptr<HostSource> source) {
  require_unleased();
  if (!source) {
    if (state_)
      state_->bind_host_source(nullptr);
    return;
  }
  state().bind_host_source(std::move(source));
}

std::vector<double> Workspace::staged_trial(const double z, const double dz,
                                            const bool prepare_only) {
  if (!stage_tables_)
    throw std::logic_error("a staged Runge-Kutta trial needs the stage tables");
  if (!runge_kutta_)
    throw std::logic_error("a staged Runge-Kutta trial needs the tableau's nodes");
  const auto& nodes = runge_kutta_->nodes;
  if (nodes.size() != stages_)
    throw std::invalid_argument("the trial's nodes are the tableau's stages");
  const auto scalars =
      stage::rows(*stage_tables_, stage_anchor_, z, rk::positions(z, dz, nodes));
  if (stage_tables_->flow.present())
    guard_flow(scalars.flow_section, dz, nodes);
  auto& general = built();
  if (prepare_only) {
    general.prepare_trial(dz, scalars);
    return {};
  }
  return general.trial_step(dz, scalars);
}

void Workspace::guard_flow(const std::size_t section, const double dz,
                           const std::vector<double>& nodes) {
  if (!flow_guard_)
    throw std::logic_error("a staged trial beside a coupled flow needs its guard");
  if (flow_key_ && flow_key_->first == section && flow_key_->second == dz)
    return;
  // Forgotten first: a refusal below leaves no set of actions behind a key.
  flow_key_.reset();
  const auto& guard = *flow_guard_;
  if (section >= guard.row_sum.size())
    throw std::out_of_range("the trial's flow section is past the guard's");
  std::vector<double> fractions(nodes);
  fractions.push_back(1.0);
  for (const double fraction : fractions) {
    if (fraction == 0.0)
      continue;
    const double largest = fraction * dz * guard.row_sum[section];
    int squarings = 0;
    if (largest > guard.taylor_scale)
      squarings = static_cast<int>(std::ceil(std::log2(largest / guard.taylor_scale)));
    if (squarings > guard.max_squarings)
      refuse<BackendRefusal>(
          "native cannot take a coupled flow action whose exponent has row-sum norm ",
          py_g(largest, 4), ": it would need ", squarings, " squarings, above the ",
          guard.max_squarings, " this backend admits. Cap dz_max");
  }
  const auto measured = built().form_flow_actions(section, dz, fractions);
  const std::size_t width = guard.condition_bound.size() / guard.row_sum.size();
  if (!measured.empty()) {
    if (measured.size() != width)
      throw std::logic_error("the measured conditioning is one value an active bin");
    const double* declared = guard.condition_bound.data() + section * width;
    // NumPy's argmax: the first largest, or the first NaN.
    std::size_t worst = 0;
    double largest = measured[0] / (declared[0] * guard.allowance);
    for (std::size_t bin = 1; bin < width && !std::isnan(largest); ++bin) {
      const double excess = measured[bin] / (declared[bin] * guard.allowance);
      if (std::isnan(excess) || excess > largest) {
        largest = excess;
        worst = bin;
      }
    }
    if (largest > 1.0)
      refuse<BackendRefusal>(
          "native measured a coupled flow conditioning of ", py_g(measured[worst], 12),
          " over a step of ", py_g(dz, 4), " m, above the ", py_g(declared[worst], 12),
          " the certificate declares at active bin ", worst, " of section ", section,
          " -- exceeding it by a factor ", py_g(largest, 12),
          " of the rounding allowance: the inverse action carries the nonlinear "
          "source back by the measured factor. Lower dz_max, which is what the "
          "bound was declared over");
  }
  flow_key_ = std::make_pair(section, dz);
}

MethodKind Workspace::kind() const noexcept {
  if (adams_)
    return MethodKind::adams;
  if (low_storage_)
    return MethodKind::low_storage;
  return MethodKind::runge_kutta;
}

std::unique_ptr<rk::Engine> Workspace::trial_engine() {
  auto& general = built();
  if (low_storage_)
    return low_storage_engine();
  if (!runge_kutta_)
    throw std::logic_error("the native Runge-Kutta trial is not configured");
  return std::make_unique<engines::ResidentRungeKutta>(
      general,
      [this](const double z, const double dz) { return staged_trial(z, dz, false); },
      runge_kutta_->rtol, runge_kutta_->atol, runge_kutta_->weights);
}

std::unique_ptr<engines::ResidentLowStorage> Workspace::low_storage_engine() {
  auto& general = built();
  if (!low_storage_ || !stage_tables_ || !rk_policy_)
    throw std::logic_error("the native low-storage replay is not configured");
  return std::make_unique<engines::ResidentLowStorage>(
      general, *stage_tables_, stage_anchor_, *low_storage_, rk_schedule_,
      rk_policy_->landings.landing_ulps, fault_);
}

std::unique_ptr<march::Engine> Workspace::adams_engine() {
  auto& general = built();
  if (!stage_tables_)
    throw std::logic_error("the native Adams engine needs the stage tables");
  return std::make_unique<engines::ResidentEngine>(
      general, *stage_tables_, stage_anchor_, march_fused_, march_compensated_,
      march_interpolate_, fault_);
}

session::Method Workspace::method(const MethodKind kind,
                                  std::variant<march::State, rk::State> start,
                                  const bool dense) {
  require_unleased();
  auto& general = state();
  if (kind == MethodKind::adams) {
    if (!march_policy_ || !stage_tables_)
      throw std::logic_error("the native Adams march is not configured");
    session::AdamsMethod method;
    method.policy = *march_policy_;
    method.state = std::get<march::State>(std::move(start));
    method.schedule = march_schedule_;
    method.engine = std::make_unique<engines::ResidentEngine>(
        general, *stage_tables_, stage_anchor_, march_fused_, march_compensated_,
        march_interpolate_, fault_);
    if (dense) {
      if (!march_interpolate_)
        throw std::invalid_argument(
            "a session's dense output needs an Adams engine that keeps its steps");
      method.dense = std::make_unique<engines::AdamsDense>(
          general, *stage_tables_, stage_anchor_, march_fused_, march_compensated_);
    }
    return method;
  }
  if (!rk_policy_)
    throw std::logic_error("the native Runge-Kutta march is not configured");
  session::RungeKuttaMethod method;
  method.policy = *rk_policy_;
  method.state = std::get<rk::State>(std::move(start));
  method.schedule = rk_schedule_;
  if (kind == MethodKind::low_storage) {
    // solprop SEP 47 section 6.3: the replay is the Runge-Kutta march's, each step one
    // in-place low-storage step whose scalars the state forms itself.
    if (!stage_tables_ || !low_storage_)
      throw std::logic_error("the native low-storage replay is not configured");
    method.engine = std::make_unique<engines::ResidentLowStorage>(
        general, *stage_tables_, stage_anchor_, *low_storage_, rk_schedule_,
        rk_policy_->landings.landing_ulps, fault_);
    return method;
  }
  if (!runge_kutta_)
    throw std::logic_error("the native Runge-Kutta march is not configured");
  auto engine = std::make_unique<engines::ResidentRungeKutta>(
      general,
      [this](const double z, const double dz) { return staged_trial(z, dz, false); },
      runge_kutta_->rtol, runge_kutta_->atol, runge_kutta_->weights);
  if (dense) {
    if (!stage_tables_)
      throw std::logic_error("a session's dense output needs the stage tables");
    auto holder = std::make_unique<engines::RungeKuttaDense>(
        general, *stage_tables_, stage_anchor_, runge_kutta_->dense_weights);
    // The engine holds each crossing step's extension through it; both live in the
    // method, the engine released first.
    auto* held = holder.get();
    engine->set_extension(
        [held](const double z, const double dz) { held->hold(z, dz); });
    method.dense = std::move(holder);
  }
  method.engine = std::move(engine);
  return method;
}

march::Result Workspace::run_adams(march::State& state, const march::Bounds& bounds) {
  require_unleased();
  if (!march_policy_ || !stage_tables_)
    throw std::logic_error("the native Adams march is not configured");
  engines::ResidentEngine engine(this->state(), *stage_tables_, stage_anchor_,
                                 march_fused_, march_compensated_, march_interpolate_,
                                 fault_);
  return march::run(*march_policy_, state, engine, bounds, march_schedule_);
}

rk::Result Workspace::run_rk(rk::State& state, const march::Bounds& bounds) {
  require_unleased();
  if (!rk_policy_)
    throw std::logic_error("the native Runge-Kutta march is not configured");
  if (!runge_kutta_)
    throw std::logic_error("the native Runge-Kutta trial is not configured");
  engines::ResidentRungeKutta engine(
      this->state(),
      [this](const double z, const double dz) { return staged_trial(z, dz, false); },
      runge_kutta_->rtol, runge_kutta_->atol, runge_kutta_->weights);
  return rk::run(*rk_policy_, state, engine, bounds, rk_schedule_);
}

void Workspace::load(const std::span<const Complex> field,
                     const std::span<const Complex> noise) {
  auto& state = this->state();
  if (field.size() != elements_)
    refuse("a start's field has ", field.size(), " elements, expected ", elements_);
  state.unbind_monitor();
  state.set_initial(field, noise);
  // A collar an earlier run rested goes back into the stages (SEP 21).
  state.set_absorber_quiet(false);
  state.clear_fsal();
}

void Workspace::bind_run_monitor(const MonitorBinding& monitor,
                                 const std::optional<EdgeBinding>& edge) {
  auto& state = this->state();
  state.bind_monitor(monitor.weights, monitor.column_weights);
  // The problem's table, shared by a runner's points: it crosses once.
  if (monitor.split && occupancy_ && !occupancy_bound_) {
    state.bind_occupancy(occupancy_->buckets, occupancy_->count);
    occupancy_bound_ = true;
  }
  if (edge) {
    state.bind_edge(edge->faces, edge->clip, edge->totals, edge->synthesis,
                    edge->first_row);
    if (edge->lines)
      state.bind_edge_lines(*edge->lines);
  }
}

void Workspace::configure_march(const RunRecordsRecord& records) {
  require_open();
  if (!stage_tables_)
    throw std::logic_error("the native marches need the stage tables");
  auto schedule = records.schedule ? to_native(*records.schedule) : march::Schedule{};
  if (kind() == MethodKind::adams) {
    const auto* policy = std::get_if<march::Policy>(&records.policy);
    if (policy == nullptr)
      refuse("an Adams workspace's run needs the Adams march's policy");
    march::validate(*policy);
    march_policy_ = *policy;
    march_schedule_ = std::move(schedule);
  } else {
    const auto* policy = std::get_if<RungeKuttaPolicyRecord>(&records.policy);
    if (policy == nullptr)
      refuse("a Runge-Kutta workspace's run needs the Runge-Kutta march's policy");
    rk_policy_ = to_native(*policy);
    rk_schedule_ = std::move(schedule);
  }
}

std::unique_ptr<Run> Workspace::start(RunStart start) {
  require_open();
  const auto method_kind = kind();
  auto& records = start.records;
  std::variant<march::State, rk::State> initial;
  if (method_kind == MethodKind::adams) {
    if (!start.adams)
      refuse("an Adams run's start needs the march's state");
    initial = std::move(*start.adams);
  } else {
    if (!start.runge_kutta)
      refuse("a Runge-Kutta run's start needs the march's state");
    initial = std::move(*start.runge_kutta);
  }
  if (records.monitor && !start.monitor)
    refuse("a run with a boundary monitor needs its weights");
  configure_march(records);
  // The load first: a history's rows are read from a loaded state. They are every one
  // checked before any is copied, so that a refused history leaves the rows as they
  // were and starts no run.
  load(start.field, start.noise);
  if (!start.rows.empty()) {
    if (method_kind != MethodKind::adams)
      refuse("only the Adams march resumes a history's rows");
    const auto names = state().adams_row_names();
    if (names.size() != start.rows.size())
      refuse("a resumed history holds ", start.rows.size(), " rows, and the engine ",
             names.size());
    for (const auto& row : start.rows) {
      if (std::ranges::find(names, row.name) == names.end())
        refuse("the engine keeps no Adams history row '", row.name, "'");
      const auto extents = state().adams_row_extents(row.name);
      if (row.n_slots != extents[0] || row.n_rows != extents[1] ||
          row.n_omega != extents[2])
        refuse("the Adams history row '", row.name, "' must have shape (", extents[0],
               ", ", extents[1], ", ", extents[2], ")");
    }
  }
  for (const auto& row : start.rows)
    state().adams_rows_from_host(row.name, row.values);
  session::Plan plan;
  const auto& planes = records.planes;
  plan.planes = planes.planes;
  plan.next_plane = planes.next_plane;
  plan.launch_pending = planes.launch_pending;
  plan.step_planes = planes.step_planes;
  plan.plane_statistics.plan = std::move(start.plane_statistics);
  plan.row_statistics.plan = std::move(start.row_statistics);
  plan.period = planes.period;
  plan.attempt_capacity = planes.attempt_capacity;
  plan.row_capacity = planes.row_capacity;
  plan.replay = planes.replay_steps > 0;
  plan.replay_steps = planes.replay_steps;
  plan.replay_length = planes.replay_length;
  plan.call_attempts = planes.call_attempts;
  plan.call_seconds = planes.call_seconds;
  if (records.monitor) {
    const auto& monitor = *records.monitor;
    watch::validate(monitor.config.policy);
    session::MonitorPlan held;
    held.config = monitor.config;
    held.resume = monitor.continuation.has_value();
    if (monitor.continuation) {
      held.record = to_native(monitor.continuation->record);
      held.plane = monitor.continuation->plane;
      held.unwatched = monitor.continuation->unwatched;
      held.edge_record = to_native(monitor.continuation->edge);
    }
    if (start.floor) {
      held.floor = true;
      held.floor_energy = (*start.floor)[0];
      std::copy(start.floor->begin() + 1, start.floor->end(),
                held.floor_regions.begin());
    }
    if (monitor.edge) {
      held.edge = *monitor.edge;
      held.edge_noise = start.edge_noise;
    }
    plan.monitor = std::move(held);
    bind_run_monitor(*start.monitor, start.edge);
  }
  auto method = this->method(method_kind, std::move(initial), planes.dense);
  return launch(std::move(method), std::move(plan), planes.z);
}

std::unique_ptr<Run> Workspace::launch(session::Method method, session::Plan plan,
                                       const double z) {
  auto& general = state();
  auto session =
      std::make_unique<session::Session>(std::make_unique<engines::GeneralResident>(
                                             general, elements_, elements_ / columns_),
                                         std::move(method), std::move(plan), z);
  return std::make_unique<Run>(*this, std::move(session));
}

std::unique_ptr<Probe> Workspace::probe(std::vector<Complex> field,
                                        std::vector<Complex> noise, const double z) {
  require_open();
  return std::make_unique<Probe>(*this, std::move(field), std::move(noise), z);
}

std::unique_ptr<Probe> Workspace::probe(RunStart start) {
  require_open();
  // The low-storage replay's landings and table keys come from the run's records; an
  // Adams workspace's probe takes Runge-Kutta trials, which read none of them.
  if (kind() != MethodKind::adams)
    configure_march(start.records);
  return probe(std::move(start.field), std::move(start.noise), start.records.planes.z);
}

stage::Trial Workspace::stage_trial(const double z0, const double z1) {
  require_open();
  if (!stage_tables_)
    throw std::logic_error("no stage tables were uploaded");
  const auto row = stage::row(*stage_tables_, stage_anchor_, z0, z1);
  auto step = stage::trial(*stage_tables_, row);
  auto sample = stage::sample(*stage_tables_, stage_anchor_, step, z1);
  return {.step = std::move(step), .sample = std::move(sample)};
}

std::vector<double> Workspace::stage_rate(const double z) const {
  require_open();
  if (!stage_tables_)
    throw std::logic_error("no stage tables were uploaded");
  return stage::rate(*stage_tables_, z);
}

std::vector<Complex> Workspace::table_phi(const void* holder, const double z) {
  auto& state = leased_state(holder);
  if (!z_slots_.phi)
    throw std::logic_error("native state holds no z table of Phi");
  const std::size_t slot = z_slots_.load(state, {z}).front();
  return state.table_terms_to_host(TableKind::phi, slot);
}

} // namespace solprop_native
