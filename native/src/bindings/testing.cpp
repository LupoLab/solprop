// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The private `_testing` module (solprop SEP 52 section 6.5): every entry point that
// only solprop's own tests reach, shipped in the wheel and promising nothing
// (`bind_testing`), and native's engines through solprop's Python protocols for the
// engine-level parity tests (`bind_engines`).
//
// Types: `TestState` (a workspace's state for the entry points that read or step it,
// holding no lease), `TestEngines` (the configured engines, leasing the workspace),
// `ScriptedMarch` and `ScriptedRkMarch` (the decision-trace tests' scripted marches).
// Functions: the trial-scalar and Adams-table readers, `round_trip` (a record through
// its table and back) and `bind_testing`'s one table of entry points.

#include "arrays.hpp"
#include "bind.hpp"
#include "march.hpp"
#include "module.hpp"
#include "records.hpp"
#include "solprop_native/general/general_state.hpp"
#include "solprop_native/march/adams_march.hpp"
#include "solprop_native/march/dense.hpp"
#include "solprop_native/march/engines.hpp"
#include "solprop_native/march/lagrange_basis.hpp"
#include "solprop_native/march/low_storage.hpp"
#include "solprop_native/march/rk_march.hpp"
#include "solprop_native/march/stage_tables.hpp"
#include "solprop_native/problem/problem.hpp"
#include "solprop_native/response/flow_functions.hpp"
#include "solprop_native/runtime/cpu_execution.hpp"
#include "solprop_native/runtime/crossing.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/fields.hpp"
#include "solprop_native/runtime/host_source.hpp"
#include "solprop_native/runtime/refuse.hpp"
#include "solprop_native/session/readings.hpp"
#include "solprop_native/session/session.hpp"
#include "solprop_native/session/statistics_plan.hpp"
#include "solprop_native/session/transverse.hpp"
#include "solprop_native/session/watch.hpp"
#include "solprop_native/transforms/fftw_wisdom.hpp"
#include "solprop_native/transforms/reference_timing.hpp"
#include "solprop_native/workspace/run_records.hpp"
#include "solprop_native/workspace/workspace.hpp"

#include <Kokkos_Core.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace solprop_native::bindings {

using namespace nb::literals;

namespace {

native::TrialScalars trial_from(const RealInput phi_delta, const RealInput phi_at,
                                const RealInput prefactor, const RealInput density,
                                const RealInput taper, const RealInput preionisation,
                                const RealInput raman_rates) {
  return {.phi_delta = real_vector(phi_delta),
          .phi_at = real_vector(phi_at),
          .prefactor = real_vector(prefactor),
          .density = real_vector(density),
          .taper = real_vector(taper),
          .preionisation = real_vector(preionisation),
          .raman_rates = real_vector(raman_rates),
          .kerr_density = {},
          .plasma_densities = {},
          .plasma_preionisation = {},
          .raman_densities = {},
          .flow_section = 0,
          .flow_weights = {},
          .transform_weights = {},
          .dispersive_densities = {}};
}

// A trial's scalars with its material rows, as the Adams entry points take them.
native::TrialScalars adams_scalars(const RealInput phi_delta, const RealInput phi_at,
                                   const RealInput prefactor, const RealInput density,
                                   const RealInput taper, const RealInput preionisation,
                                   const RealInput raman_rates,
                                   const nb::dict material) {
  auto scalars = trial_from(phi_delta, phi_at, prefactor, density, taper, preionisation,
                            raman_rates);
  scalars.kerr_density = real_vector(exact_array<RealInput>(material["kerr_density"]));
  scalars.plasma_densities =
      real_vector(exact_array<RealInput>(material["plasma_densities"]));
  scalars.plasma_preionisation =
      real_vector(exact_array<RealInput>(material["plasma_preionisation"]));
  scalars.raman_densities =
      real_vector(exact_array<RealInput>(material["raman_densities"]));
  if (material.contains("transform_weights"))
    scalars.transform_weights =
        real_vector(exact_array<RealInput>(material["transform_weights"]));
  if (material.contains("dispersive_densities"))
    scalars.dispersive_densities =
        real_vector(exact_array<RealInput>(material["dispersive_densities"]));
  // Beside a coupled flow the trial's section and a frozen-residual flow's weights
  // (solprop SEP 16 phase 6).
  if (material.contains("flow_section")) {
    scalars.flow_section = nb::cast<std::size_t>(material["flow_section"]);
    scalars.flow_weights =
        real_vector(exact_array<RealInput>(material["flow_weights"]));
  }
  return scalars;
}

// One exponential trial's tables formed natively from its nodes (solprop SEP 39
// phase 1): `stage::weights` of the arrays solprop passes.
native::ExponentialWeights
exponential_weights_at(const RealInput nodes, const IndexInput slots,
                       const std::size_t order, const std::vector<std::size_t>& orders,
                       const bool compensated) {
  const auto slot_values = index_vector(slots);
  return native::stage::weights(nodes.data(), element_count(nodes), slot_values.data(),
                                slot_values.size(), order, orders.data(), orders.size(),
                                compensated);
}

// One exponential trial's tables (solprop SEP 16 phase 3b): `predictor` is
// (order, order), `corrector` (width, width) and `rows` (n_rows, 2 width), row-major.
native::ExponentialWeights exponential_weights(const IndexInput slots,
                                               const RealInput predictor,
                                               const RealInput corrector,
                                               const RealInput rows) {
  native::ExponentialWeights weights;
  const auto slot_values = index_vector(slots);
  const auto predictor_values = real_vector(predictor);
  const auto corrector_values = real_vector(corrector);
  const auto row_values = real_vector(rows);
  const auto order = static_cast<std::size_t>(
      std::llround(std::sqrt(static_cast<double>(predictor_values.size()))));
  const auto width = static_cast<std::size_t>(
      std::llround(std::sqrt(static_cast<double>(corrector_values.size()))));
  if (order * order != predictor_values.size() ||
      width * width != corrector_values.size() || width == 0 ||
      width > native::adams_max_samples || order >= width ||
      slot_values.size() + 1 < width || row_values.size() % (2 * width) != 0 ||
      row_values.size() / (2 * width) > 3 || row_values.empty())
    throw std::invalid_argument("Adams coefficient tables have inconsistent extents");
  for (std::size_t t = 0; t < slot_values.size() && t < native::adams_max_samples; ++t)
    weights.slot[t] = slot_values[t];
  for (std::size_t j = 0; j < order; ++j)
    for (std::size_t k = 0; k < order; ++k)
      weights.predictor[j][k] = predictor_values[j * order + k];
  for (std::size_t j = 0; j < width; ++j)
    for (std::size_t k = 0; k < width; ++k)
      weights.corrector[j][k] = corrector_values[j * width + k];
  weights.order = order;
  weights.width = width;
  weights.count = width;
  weights.n_rows = row_values.size() / (2 * width);
  for (std::size_t row = 0; row < weights.n_rows; ++row)
    for (std::size_t t = 0; t < 2 * width; ++t)
      weights.rows[row][t] = row_values[row * 2 * width + t];
  return weights;
}

void carrier_frame(native::ExponentialWeights& weights, const RealInput carrier,
                   const std::size_t rows) {
  native::engines::carrier_frame(weights, real_vector(carrier), rows);
}

// History rows alone, as a transport or a dense copy names them.
native::AdamsWeights adams_rows(const IndexInput slots) {
  native::AdamsWeights weights;
  const auto values = index_vector(slots);
  if (values.size() > native::adams_max_samples)
    throw std::invalid_argument("too many Adams history rows");
  for (std::size_t t = 0; t < values.size(); ++t)
    weights.slot[t] = values[t];
  weights.width = values.size();
  return weights;
}

// The scripted march solprop's decision-trace tests drive.
class ScriptedMarch {
public:
  ScriptedMarch(const nb::dict policy, const RealInput norms, const double carrier_rate,
                const RealInput steps, const RealInput edges)
      : policy_(march_policy_from(policy)), norms_(real_vector(norms)),
        carrier_rate_(carrier_rate),
        schedule_{.steps = real_vector(steps), .edges = real_vector(edges)} {
    if (norms_.size() % 3 != 0)
      throw std::invalid_argument("a script of norms is (attempts, 3)");
  }
  nb::dict run(const nb::dict state_values, const nb::dict bounds_values) {
    auto state = march_state_from(state_values);
    const auto bounds = march_bounds_from(bounds_values);
    std::vector<double> remaining(
        norms_.begin() + static_cast<std::ptrdiff_t>(used_ * 3), norms_.end());
    native::engines::ScriptedEngine engine(std::move(remaining), carrier_rate_, log_);
    const auto result = march::run(policy_, state, engine, bounds, schedule_);
    used_ += engine.used();
    return march_result_to(result, state);
  }
  // What the march asked, one dict an entry: a startup's position, or a trial's
  // position, step, nodes, rows and orders with the row its accept freed.
  [[nodiscard]] nb::list log() const {
    nb::list out;
    for (const auto& entry : log_) {
      nb::dict event;
      if (entry.start) {
        event["start"] = entry.z;
        event["stored"] = true;
      } else {
        event["z"] = entry.z;
        event["dz"] = entry.dz;
        event["nodes"] = entry.nodes;
        event["slots"] = entry.slots;
        event["order"] = entry.order;
        event["orders"] = entry.orders;
        if (entry.free)
          event["free"] = *entry.free;
      }
      out.append(event);
    }
    return out;
  }

private:
  march::Policy policy_;
  std::vector<double> norms_;
  double carrier_rate_;
  march::Schedule schedule_;
  std::vector<native::engines::ScriptedEvent> log_;
  std::size_t used_ = 0;
};

class ScriptedRkMarch {
public:
  ScriptedRkMarch(const nb::object& policy, const RealInput norms, const double h0,
                  const RealInput steps, const RealInput edges)
      : policy_(native::to_native(
            native::records::read<native::RungeKuttaPolicyRecord>(policy, "policy"))),
        norms_(real_vector(norms)), h0_(h0),
        schedule_{.steps = real_vector(steps), .edges = real_vector(edges)} {}
  nb::dict run(const nb::dict state_values, const nb::dict bounds_values) {
    auto state = native::records::read<native::rk::State>(state_values, "state");
    const auto bounds = march_bounds_from(bounds_values);
    native::engines::ScriptedRungeKutta engine(norms_, next_, h0_, policy_.power, log_);
    const auto result = native::rk::run(policy_, state, engine, bounds, schedule_);
    return rk_result_to(result, state);
  }
  // Each trial's position and step, one dict a trial.
  [[nodiscard]] nb::list log() const {
    nb::list out;
    for (const auto& entry : log_) {
      nb::dict event;
      event["z"] = entry.z;
      event["dz"] = entry.dz;
      out.append(event);
    }
    return out;
  }
  [[nodiscard]] std::size_t used() const { return next_; }

private:
  native::rk::Policy policy_;
  std::vector<double> norms_;
  double h0_;
  march::Schedule schedule_;
  std::size_t next_ = 0;
  std::vector<native::engines::ScriptedTrial> log_;
};

/** Read `value` as the record named `name` and write it back (the round-trip tests). */
nb::dict round_trip(const std::string& name, const nb::handle value) {
  const auto trip = [&]<class R>() -> nb::dict {
    return records::write_record(records::read<R>(value, name));
  };
#define SOLPROP_NATIVE_TRIP(Type)                                                      \
  if (name == native::fields::table_of<Type>().name)                                   \
    return trip.template operator()<Type>();
  SOLPROP_NATIVE_TRIP(native::RunRecordsRecord)
  SOLPROP_NATIVE_TRIP(native::RunStart)
  SOLPROP_NATIVE_TRIP(native::march::Policy)
  SOLPROP_NATIVE_TRIP(native::RungeKuttaPolicyRecord)
  SOLPROP_NATIVE_TRIP(native::ControllerRecord)
  SOLPROP_NATIVE_TRIP(native::rk::low_storage::Config)
  SOLPROP_NATIVE_TRIP(native::ScheduleRecord)
  SOLPROP_NATIVE_TRIP(native::PlanePlanRecord)
  SOLPROP_NATIVE_TRIP(native::watch::Config)
  SOLPROP_NATIVE_TRIP(native::watch::Policy)
  SOLPROP_NATIVE_TRIP(native::transverse::Config)
  SOLPROP_NATIVE_TRIP(native::ResumptionRecord)
  SOLPROP_NATIVE_TRIP(native::EdgeResumptionRecord)
  SOLPROP_NATIVE_TRIP(native::ContinuationRecord)
  SOLPROP_NATIVE_TRIP(native::MonitorRecordsRecord)
  SOLPROP_NATIVE_TRIP(native::march::State)
  SOLPROP_NATIVE_TRIP(native::rk::State)
  SOLPROP_NATIVE_TRIP(native::stage::Table)
  SOLPROP_NATIVE_TRIP(native::stage::Tables)
  SOLPROP_NATIVE_TRIP(native::ZTable)
  SOLPROP_NATIVE_TRIP(native::FlowGuard)
  SOLPROP_NATIVE_TRIP(native::MonitorBinding)
  SOLPROP_NATIVE_TRIP(native::EdgeBinding)
  SOLPROP_NATIVE_TRIP(native::Problem::Raman)
  SOLPROP_NATIVE_TRIP(native::Problem::Plasma)
  SOLPROP_NATIVE_TRIP(native::Setup)
#undef SOLPROP_NATIVE_TRIP
  native::refuse("no record is named '", name, "'");
}

// A test's view of a workspace's state, for the `_testing` entry points that read or
// step it directly (solprop SEP 52 section 6.5): it owns nothing and holds no lease, so
// each call refuses while a session or probe holds the workspace.
class TestState {
public:
  explicit TestState(BoundWorkspace& owner)
      : owner_(&owner), workspace_(&owner.workspace()) {}

  [[nodiscard]] native::GeneralState& open_state() const { return workspace_->state(); }
  [[nodiscard]] std::size_t elements() const noexcept { return workspace_->elements(); }
  [[nodiscard]] std::size_t columns() const noexcept { return workspace_->columns(); }
  [[nodiscard]] std::size_t stages() const noexcept { return workspace_->stages(); }
  [[nodiscard]] std::size_t raman_terms() const noexcept {
    return owner_->raman_terms();
  }
  [[nodiscard]] bool raman() const noexcept { return owner_->raman(); }
  [[nodiscard]] std::vector<double> trial_step(const double dz,
                                               const native::TrialScalars& scalars,
                                               const bool prepare_only) const {
    auto& state = open_state();
    if (prepare_only) {
      state.prepare_trial(dz, scalars);
      return {};
    }
    return state.trial_step(dz, scalars);
  }
  void adopt(const std::vector<native::Complex>& field, const double dz) const {
    open_state().adopt(field, dz);
  }
  void stage_to_host(const std::size_t stage,
                     const std::span<native::Complex> destination) const {
    open_state().stage_to_host(stage, destination);
  }
  [[nodiscard]] std::vector<native::Complex>
  projected_to_host(const bool coarse) const {
    return open_state().projected_to_host(coarse);
  }
  [[nodiscard]] std::vector<double>
  weighted_error(const std::vector<double>& scale) const {
    return open_state().weighted_error(scale);
  }
  [[nodiscard]] std::vector<double>
  form_flow_actions(const std::size_t section, const double dz,
                    const std::vector<double>& fractions) const {
    auto& state = open_state();
    workspace_->forget_flow_key();
    return state.form_flow_actions(section, dz, fractions);
  }
  [[nodiscard]] std::vector<native::Complex>
  flow_actions_to_host(const bool inverse) const {
    return open_state().flow_actions_to_host(inverse);
  }
  [[nodiscard]] std::vector<native::Complex> adams_block_to_host() const {
    return open_state().adams_block_to_host();
  }
  [[nodiscard]] std::vector<native::Complex> adams_exponent_to_host() const {
    return open_state().adams_exponent_to_host();
  }
  [[nodiscard]] native::stage::Trial stage_trial(const double z0,
                                                 const double z1) const {
    return workspace_->stage_trial(z0, z1);
  }
  [[nodiscard]] std::vector<double> stage_rate(const double z) const {
    return workspace_->stage_rate(z);
  }
  [[nodiscard]] bool has_stage_tables() const {
    return workspace_->stage_tables().has_value();
  }
  void table_rate_at(const double z, const bool before) const {
    auto& slots = workspace_->z_slots();
    if (!slots.has_rate)
      throw std::logic_error("the z table reads no rate");
    slots.rate(open_state(), z, before);
  }
  void promote_fsal() const { open_state().promote_fsal(); }
  [[nodiscard]] nb::dict march_run(const nb::dict state_values,
                                   const nb::dict bounds_values) const {
    workspace_->require_unleased();
    auto state = march_state_from(state_values);
    const auto bounds = march_bounds_from(bounds_values);
    march::Result result;
    {
      const nb::gil_scoped_release release;
      result = workspace_->run_adams(state, bounds);
    }
    return march_result_to(result, state);
  }
  [[nodiscard]] nb::dict rk_march_run(const nb::dict state_values,
                                      const nb::dict bounds_values) const {
    workspace_->require_unleased();
    auto state = native::records::read<native::rk::State>(state_values, "state");
    const auto bounds = march_bounds_from(bounds_values);
    native::rk::Result result;
    {
      const nb::gil_scoped_release release;
      result = workspace_->run_rk(state, bounds);
    }
    return rk_result_to(result, state);
  }
  [[nodiscard]] std::vector<double>
  statistics(const std::uint64_t flags, const std::vector<double>& windows,
             const std::vector<double>& on_axis, const double density,
             const double taper, const double preionisation,
             const double intensity_factor,
             const std::vector<double>& channel_preionisation,
             const std::vector<double>& power_weight) const {
    return open_state().statistics(flags, windows, on_axis, density, taper,
                                   preionisation, intensity_factor,
                                   channel_preionisation, power_weight);
  }
  [[nodiscard]] std::vector<double> spatial_statistics(const std::uint64_t flags,
                                                       const double intensity_factor,
                                                       const double z) const {
    return open_state().spatial_statistics(flags, intensity_factor, z);
  }
  [[nodiscard]] std::vector<double>
  planned_statistics(const std::shared_ptr<const native::statistics::Plan>& plan,
                     const double z) const {
    return native::statistics::evaluate(open_state(), *plan, z);
  }
  [[nodiscard]] std::vector<double>
  plane_fluence_trace(const double center_x, const double center_y, const double dx,
                      const double dy, const double extent, const double spacing,
                      const std::size_t n_radii, const std::vector<double>& cosines,
                      const std::vector<double>& sines) const {
    return open_state().plane_fluence_trace(center_x, center_y, dx, dy, extent, spacing,
                                            n_radii, cosines, sines);
  }
  [[nodiscard]] std::vector<native::Complex>
  axis_spectrum(const std::vector<double>& matrix) const {
    return open_state().axis_spectrum(matrix);
  }
  [[nodiscard]] std::vector<double> monitor_reading() const {
    return open_state().monitor_reading();
  }
  [[nodiscard]] double quadrature_error() const {
    workspace_->require_unleased();
    return workspace_->built().quadrature_error();
  }

private:
  BoundWorkspace* owner_;
  native::Workspace* workspace_;
};

// solprop SEP 52 section 6.5: native's engines through solprop's Python protocols
// (`solprop.backends.engines`), for the engine-level parity tests, with a read-back of
// the field and of a stage register. The object leases the workspace while it lives,
// as a session does.
class TestEngines {
public:
  explicit TestEngines(BoundWorkspace& owner) : owner_(&owner.workspace()) {
    auto& workspace = *owner_;
    workspace.require_unleased();
    // Each engine the workspace's configuration makes; one it cannot make is absent.
    try {
      runge_kutta_ = workspace.trial_engine(workspace.extends() ? &dense_ : nullptr);
    } catch (const std::logic_error&) {
      runge_kutta_.reset();
      dense_.reset();
    }
    try {
      adams_ = workspace.adams_engine();
    } catch (const std::logic_error&) {
      adams_.reset();
    }
    workspace.lease(this);
  }
  ~TestEngines() { close(); }
  TestEngines(const TestEngines&) = delete;
  TestEngines& operator=(const TestEngines&) = delete;
  TestEngines(TestEngines&&) = delete;
  TestEngines& operator=(TestEngines&&) = delete;

  void close() noexcept {
    if (owner_ == nullptr)
      return;
    runge_kutta_.reset();
    dense_.reset();
    adams_.reset();
    owner_->release(this);
    owner_ = nullptr;
  }
  [[nodiscard]] native::GeneralState& state() const {
    if (owner_ == nullptr)
      throw std::logic_error("the engines are closed");
    return owner_->leased_state(this);
  }
  [[nodiscard]] native::rk::Engine& runge_kutta() const {
    static_cast<void>(state());
    if (!runge_kutta_)
      throw std::logic_error("this workspace has no Runge-Kutta engine");
    return *runge_kutta_;
  }
  [[nodiscard]] native::march::Engine& adams() const {
    static_cast<void>(state());
    if (!adams_)
      throw std::logic_error("this workspace has no Adams engine");
    return *adams_;
  }
  [[nodiscard]] ComplexOutput field() const {
    const std::size_t count = owner_ == nullptr ? 0 : owner_->elements();
    auto storage = native::crossing::host_buffer(2 * count);
    state().field_to_host({reinterpret_cast<native::Complex*>(storage.get()), count});
    return complex_buffer_output(std::move(storage),
                                 {owner_->columns(), count / owner_->columns()});
  }
  [[nodiscard]] ComplexOutput stage(const std::size_t index) const {
    const std::size_t count = owner_ == nullptr ? 0 : owner_->elements();
    auto storage = native::crossing::host_buffer(2 * count);
    state().stage_to_host(index,
                          {reinterpret_cast<native::Complex*>(storage.get()), count});
    return complex_buffer_output(std::move(storage),
                                 {owner_->columns(), count / owner_->columns()});
  }
  void load(const ComplexInput field, const ComplexInput noise) const {
    auto& general = state();
    const nb::gil_scoped_release release;
    general.set_initial(complex_span(field), complex_span(noise));
  }

private:
  native::Workspace* owner_;
  /** Where a dense configuration's Runge-Kutta engine holds its extensions. */
  std::unique_ptr<native::session::Dense> dense_;
  std::unique_ptr<native::rk::Engine> runge_kutta_;
  std::unique_ptr<native::march::Engine> adams_;
};

/** solprop's protocol records, built where a test reads them. */
nb::object engines_module() { return nb::module_::import_("solprop.backends.engines"); }

/** An accepted step's `AcceptRecord` (SEP 52 section 5.2): the split collar's sums its
    finish measured, from the masked field before the split, where a monitor reads a
    split collar; the stiffness is not formed natively (NaN). */
nb::object accept_record(native::GeneralState& state) {
  auto sums = state.split_sums();
  if (!sums)
    return engines_module().attr("AcceptRecord")();
  const std::vector<double> found(sums->begin() + 1, sums->end());
  return engines_module().attr("AcceptRecord")("removed"_a = (*sums)[0],
                                               "found"_a = vector_output(found));
}

nb::object boundary_reading(const std::vector<double>& values) {
  if (values.size() < native::watch::reading_head)
    throw std::logic_error("a boundary reading is too short");
  const std::vector<double> regions(values.begin() + 1, values.begin() + 5);
  nb::object occupancy = nb::none();
  if (values.size() > native::watch::reading_head)
    occupancy = nb::cast(vector_output(std::vector<double>(
        values.begin() + native::watch::reading_head, values.end())));
  return engines_module().attr("BoundaryReading")(values[0], vector_output(regions),
                                                  occupancy);
}

/** A Python `adams_core.Stencil` as the march's. */
native::march::Stencil stencil_from(const nb::handle value) {
  native::march::Stencil stencil;
  const auto nodes = nb::cast<std::vector<double>>(value.attr("nodes"));
  const auto rows =
      nb::cast<std::vector<std::size_t>>(value.attr("rows").attr("tolist")());
  const auto orders = nb::cast<std::vector<std::size_t>>(value.attr("orders"));
  if (nodes.size() != rows.size() || nodes.size() > native::march::max_history ||
      orders.size() > native::march::max_orders)
    throw std::invalid_argument("a stencil's nodes and rows are one per held entry");
  std::ranges::copy(nodes, stencil.nodes.begin());
  std::ranges::copy(rows, stencil.rows.begin());
  stencil.held = nodes.size();
  stencil.order = nb::cast<std::size_t>(value.attr("order"));
  std::ranges::copy(orders, stencil.orders.begin());
  stencil.n_orders = orders.size();
  return stencil;
}

} // namespace

void bind_engines(nb::module_& testing) {
  nb::class_<TestEngines>(testing, "Engines")
      .def("field", &TestEngines::field)
      .def("stage", &TestEngines::stage, "index"_a)
      .def("load", &TestEngines::load, "field"_a.noconvert(), "noise"_a.noconvert())
      .def("close", &TestEngines::close)
      // The Runge-Kutta protocol (`RungeKuttaEngine`), each prefixed `rk_` here and
      // dispatched by the Python wrapper in the tests.
      .def("rk_trial",
           [](TestEngines& self, const double z, const double dz) {
             auto& engine = self.runge_kutta();
             const nb::gil_scoped_release release;
             return engine.trial(z, dz);
           })
      .def(
          "rk_accept",
          [](TestEngines& self, const double dz) -> nb::object {
            auto& engine = self.runge_kutta();
            {
              const nb::gil_scoped_release release;
              engine.accept(dz);
            }
            return accept_record(self.state());
          },
          nb::sig("def rk_accept(self, dz: float) -> "
                  "solprop.backends.engines.AcceptRecord"))
      .def("rk_extend", [](TestEngines& self, const double z,
                           const double dz) { self.runge_kutta().extend(z, dz); })
      .def(
          "rk_reading",
          [](TestEngines& self) {
            return boundary_reading(self.runge_kutta().reading());
          },
          nb::sig("def rk_reading(self) -> solprop.backends.engines.BoundaryReading"))
      .def_prop_ro("rk_in_flight",
                   [](TestEngines& self) { return self.runge_kutta().in_flight(); })
      // The Adams protocol (`AdamsEngine`).
      .def("adams_start",
           [](TestEngines& self, const double z) {
             auto& engine = self.adams();
             const nb::gil_scoped_release release;
             engine.start(z);
           })
      .def("adams_trial",
           [](TestEngines& self, const double z, const double dz,
              const nb::handle held) {
             auto& engine = self.adams();
             const auto stencil = stencil_from(held);
             std::vector<double> norms(stencil.n_orders);
             {
               const nb::gil_scoped_release release;
               engine.trial(z, dz, stencil, norms.data());
             }
             return vector_output(std::move(norms));
           })
      .def(
          "adams_accept",
          [](TestEngines& self, const double dz, const std::size_t free,
             const nb::handle held, const nb::handle kept) {
            auto& engine = self.adams();
            const auto stencil = stencil_from(held);
            std::vector<native::march::Entry> entries;
            for (const auto entry : nb::cast<nb::sequence>(kept))
              entries.push_back(
                  native::records::read<native::march::Entry>(entry, "kept"));
            {
              const nb::gil_scoped_release release;
              engine.accept(dz, free, stencil, entries);
            }
            return accept_record(self.state());
          },
          nb::sig("def adams_accept(self, dz: float, free: int, held: object, kept: "
                  "collections.abc.Sequence[solprop.backends.engines.HistoryEntry]) "
                  "-> solprop.backends.engines.AcceptRecord"))
      .def("adams_chord_phase",
           [](TestEngines& self, const double z1, const double z2) {
             return self.adams().chord_phase(z1, z2);
           })
      .def("adams_step_carrier",
           [](TestEngines& self, const std::size_t order) {
             return vector_output(self.adams().step_carrier(order));
           })
      .def("adams_begin_attempt",
           [](TestEngines& self, const std::uint64_t cost) {
             self.adams().begin_attempt(cost);
           })
      .def("adams_end_attempt",
           [](TestEngines& self, const int outcome) {
             if (outcome < 0 || outcome > 2)
               throw std::invalid_argument("an attempt's outcome is 0, 1 or 2");
             self.adams().end_attempt(static_cast<native::HostOutcome>(outcome));
           })
      .def(
          "adams_reading",
          [](TestEngines& self) { return boundary_reading(self.adams().reading()); },
          nb::sig(
              "def adams_reading(self) -> solprop.backends.engines.BoundaryReading"));
  testing.def(
      "engines",
      [](BoundWorkspace& workspace) {
        return std::make_unique<TestEngines>(workspace);
      },
      "workspace"_a, nb::keep_alive<0, 1>());
  testing.def(
      "inject_fault",
      [](BoundWorkspace& workspace, const std::string& stage, const std::size_t after,
         const std::string& category) {
        workspace.workspace().inject_fault(native::fault_from(stage, after, category));
      },
      "workspace"_a, "stage"_a, "after"_a, "category"_a);
  testing.def(
      "evaluate",
      [](BoundWorkspace& workspace, const BoundStatisticsPlan& plan, const double z) {
        auto& state = workspace.workspace().state();
        const auto& held = plan.plan();
        std::vector<double> values;
        {
          const nb::gil_scoped_release release;
          values = native::statistics::evaluate(state, *held, z);
        }
        return real_output(std::move(values), 1);
      },
      "workspace"_a, "plan"_a, "z"_a);
  testing.def("round_trip", &round_trip, "name"_a, "value"_a,
              nb::sig("def round_trip(name: str, value: object) -> dict[str, "
                      "typing.Any]"));
}

// The private `_testing` module (solprop SEP 52 section 6.5): every entry point that
// only solprop's own tests reach, built into the same extension and shipped in the
// wheel. It promises nothing. A `State` method that moved here takes the state as its
// first argument.
// NOLINTNEXTLINE(readability-function-size): the test module's one table of bindings
void bind_testing(nb::module_& testing) {
  // solprop SEP 45 phase 1: the boundary watch's predictions, for solprop's tests to
  // hold against its monitor on drawn readings.
  testing.def(
      "watch_check",
      [](const nb::dict policy, const nb::dict state, const RealInput reading,
         const double z) {
        const auto rules = watch_policy_from(policy);
        auto held = watch_state_from(state);
        const auto values = real_vector(reading);
        if (values.size() < native::watch::reading_head ||
            (rules.crossing &&
             values.size() < native::watch::reading_head + native::watch::buckets))
          throw std::invalid_argument("a boundary reading is too short for its policy");
        const auto verdict = native::watch::check(rules, held, values.data(), z);
        nb::dict out;
        out["verdict"] = static_cast<int>(verdict);
        out["state"] = watch_state_to(held);
        return out;
      },
      "policy"_a, "state"_a, "reading"_a, "z"_a);
  // The coupled flow's series constants compiled into the kernels no `FlowGuard`
  // reaches, which a test holds to solprop's (`solprop.backends.exponential`).
  testing.attr("flow_taylor_order") = native::flow_functions::taylor_order;
  testing.attr("flow_taylor_scale") = native::flow_functions::taylor_scale;
  testing.attr("flow_max_squarings") = native::flow_functions::max_squarings;
  // solprop SEP 47: the native half-maximum width the statistics plan finishes with.
  testing.def(
      "half_width",
      [](const RealInput t, const RealInput power, const bool outer) {
        if (t.ndim() != 1 || power.ndim() != 1 || t.shape(0) != power.shape(0))
          throw std::invalid_argument("a width's axis and trace are one-dimensional "
                                      "and of one length");
        return native::statistics::half_width(t.data(), power.data(), t.shape(0),
                                              outer);
      },
      "t"_a.noconvert(), "power"_a.noconvert(), "outer"_a);
  // solprop #135: the strict files' probes as the built library holds them, beside a
  // control compiled with this file's contraction. Where the control fuses, a probe
  // that also fused has lost its file's rule, through inlining or the build's flags.
  testing.def(
      "strict_witness",
      [](const double a, const double b, const double c) {
        const double product = a * b;
        const double control = product + c;
        return nb::make_tuple(control, native::march::strict_probe(a, b, c),
                              native::watch::strict_probe(a, b, c),
                              native::rk::strict_probe(a, b, c),
                              native::session::strict_probe(a, b, c),
                              native::statistics::strict_probe(a, b, c),
                              native::rk::low_storage::strict_probe(a, b, c),
                              native::transverse::strict_probe(a, b, c),
                              native::dense::strict_probe(a, b, c),
                              native::statistics::widths::strict_probe(a, b, c));
      },
      "a"_a, "b"_a, "c"_a);
  // solprop SEP 47: the fluence widths' native finishing on given profiles, for the
  // tests that hold it to solprop's own (`solprop.spatial.radial_fluence_widths`,
  // `_line_fluence_widths`).
  testing.def(
      "radial_fluence_widths",
      [](const RealInput radius, const RealInput fluence, const RealInput weights,
         const double axis_fluence, const double e2_fraction) {
        if (radius.ndim() != 1 || fluence.ndim() != 1 || weights.ndim() != 1 ||
            fluence.shape(0) != radius.shape(0) || weights.shape(0) != radius.shape(0))
          throw std::invalid_argument("a radial profile's samples are one-dimensional "
                                      "and of one length");
        const auto widths = native::statistics::widths::radial(
            radius.data(), fluence.data(), weights.data(), radius.shape(0),
            axis_fluence, e2_fraction);
        return nb::make_tuple(widths[0], widths[1], widths[2]);
      },
      "radius"_a.noconvert(), "fluence"_a.noconvert(), "weights"_a.noconvert(),
      "axis_fluence"_a, "e2_fraction"_a);
  testing.def(
      "fluence_block_widths",
      [](const BoundStatisticsPlan& plan, const RealInput block) {
        if (block.ndim() != 1)
          throw std::invalid_argument("a fluence block is one-dimensional");
        const auto widths =
            native::statistics::widths::of_block(*plan.plan(), real_vector(block));
        return nb::make_tuple(widths[0], widths[1], widths[2]);
      },
      "plan"_a, "block"_a.noconvert());
  testing.def(
      "line_fluence_widths",
      [](const RealInput x, const RealInput samples, const double e2_fraction) {
        if (x.ndim() != 1 || samples.ndim() != 1 || samples.shape(0) != x.shape(0) ||
            x.shape(0) < 2)
          throw std::invalid_argument("a line's samples are one-dimensional and of "
                                      "one length, at least two");
        const auto widths = native::statistics::widths::line(
            real_vector(x), real_vector(samples), e2_fraction);
        return nb::make_tuple(widths[0], widths[1], widths[2]);
      },
      "x"_a.noconvert(), "samples"_a.noconvert(), "e2_fraction"_a);
  testing.def(
      "watch_due",
      [](const nb::dict policy, const nb::dict state, const double z) {
        return native::watch::due(watch_policy_from(policy), watch_state_from(state),
                                  z);
      },
      "policy"_a, "state"_a, "z"_a);
  testing.def(
      "watch_split_limit",
      [](const nb::dict policy, const RealInput occupancy) {
        const auto rules = watch_policy_from(policy);
        const auto values = real_vector(occupancy);
        if (values.size() != native::watch::buckets)
          throw std::invalid_argument("an occupancy has 128 buckets");
        return native::watch::split_limit(rules, values.data());
      },
      "policy"_a, "occupancy"_a);
  // solprop SEP 28 section 4.2's reference parts, for its benchmarks only.
  testing.def(
      "reference_fft3_pairs",
      [](const ComplexInput& values, const std::size_t warm, const std::size_t pairs) {
        if (values.ndim() != 3)
          throw std::invalid_argument("the reference volume must be three-dimensional");
        return native::reference_fft3_pairs(values.data(), values.shape(0),
                                            values.shape(1), values.shape(2), warm,
                                            pairs);
      },
      "values"_a, "warm"_a, "pairs"_a);
  // The calling thread's OpenMP default as a native call sees it (solprop issue #130).
  testing.def("scoped_openmp_default", &native::scoped_openmp_default);
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  // solprop SEP 46: forget FFTW's process-wide wisdom, for a test of the planning
  // transaction.
  testing.def("forget_fftw_wisdom", [] { native::forget_fftw_wisdom(); });
#else
  testing.def("forget_fftw_wisdom", [] {});
#endif
  testing.def("scoped_block_time", &native::scoped_block_time);
  testing.def(
      "import_fftw_wisdom",
      [](const std::string& text) -> bool {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        static_cast<void>(text);
        return false;
#else
        return native::import_fftw_wisdom(text);
#endif
      },
      "text"_a);
  nb::class_<ScriptedMarch>(testing, "ScriptedMarch")
      .def(nb::init<nb::dict, RealInput, double, RealInput, RealInput>(), "policy"_a,
           "norms"_a.noconvert(), "carrier_rate"_a, "steps"_a.noconvert(),
           "edges"_a.noconvert())
      .def("run", &ScriptedMarch::run, "state"_a, "bounds"_a)
      .def("log", &ScriptedMarch::log);
  nb::class_<ScriptedRkMarch>(testing, "ScriptedRkMarch")
      .def(nb::init<nb::object, RealInput, double, RealInput, RealInput>(), "policy"_a,
           "norms"_a.noconvert(), "h0"_a, "steps"_a.noconvert(), "edges"_a.noconvert())
      .def("run", &ScriptedRkMarch::run, "state"_a, "bounds"_a)
      .def("log", &ScriptedRkMarch::log)
      .def("used", &ScriptedRkMarch::used);
  // solprop SEP 47 section 6.4: the owning boundary monitor alone, for solprop's tests
  // to feed recorded readings through it and its own monitor alike.
  nb::class_<native::watch::Monitor>(testing, "Monitor")
      .def(
          "__init__",
          [](native::watch::Monitor* self, const nb::handle config) {
            new (self) native::watch::Monitor(watch_config_from(config));
          },
          "config"_a,
          nb::sig("def __init__(self, config: solprop.backends.records.MonitorConfig) "
                  "-> None"))
      .def(
          "resume",
          [](native::watch::Monitor& self, const nb::handle record, const bool plane,
             const nb::handle unwatched) {
            const auto held =
                native::records::read<native::ResumptionRecord>(record, "record");
            const auto stretches =
                native::records::read<std::vector<std::array<double, 2>>>(unwatched,
                                                                          "unwatched");
            self.resume(native::to_native(held), plane, stretches);
          },
          "record"_a, "plane"_a, "unwatched"_a,
          nb::sig(
              "def resume(self, record: solprop.backends.records.Resumption, plane: "
              "bool, unwatched: collections.abc.Sequence[tuple[float, float]]) -> "
              "None"))
      .def(
          "launch",
          [](native::watch::Monitor& self, const RealInput reading, const double z,
             const double floor, const std::optional<RealInput>& floor_regions) {
            const auto values = real_vector(reading);
            std::vector<double> regions;
            if (floor_regions) {
              regions = real_vector(*floor_regions);
              if (regions.size() != 4)
                throw std::invalid_argument("the floor's regions are four");
            }
            return watch_judgement_to(
                self.launch(values.data(), values.size(), z, floor,
                            floor_regions ? regions.data() : nullptr));
          },
          "reading"_a, "z"_a, "floor"_a = 0.0, "floor_regions"_a.none() = nb::none())
      .def("due", &native::watch::Monitor::due, "z"_a)
      .def("skip", &native::watch::Monitor::skip, "z"_a)
      .def(
          "step",
          [](native::watch::Monitor& self, const RealInput reading, const double z,
             const bool plane) {
            const auto values = real_vector(reading);
            return watch_judgement_to(
                self.step(values.data(), values.size(), z, plane));
          },
          "reading"_a, "z"_a, "plane"_a = false)
      .def("plane", &native::watch::Monitor::plane, "z"_a)
      .def("note_collar", &native::watch::Monitor::note_collar, "left_out"_a)
      .def("finish",
           [](const native::watch::Monitor& self) {
             const auto finish = self.finish();
             nb::dict out;
             out["removed"] = finish.removed;
             out["guard_low_max"] = finish.guard_max[0];
             out["guard_high_max"] = finish.guard_max[1];
             out["warns"] = finish.warns;
             return out;
           })
      .def("record",
           [](const native::watch::Monitor& self) {
             return watch_record_to(self.record());
           })
      .def("watch_state",
           [](const native::watch::Monitor& self) {
             return watch_state_to(self.state());
           })
      .def_prop_ro("stale", &native::watch::Monitor::stale)
      .def_prop_ro("collar_quiet", &native::watch::Monitor::quiet)
      .def_prop_ro("step_cap", &native::watch::Monitor::cap)
      .def_prop_ro("rests", &native::watch::Monitor::rests)
      .def("steps_to_check", &native::watch::Monitor::steps_to_check);
  testing.def(
      "trial_step",
      [](BoundWorkspace& owner, const double dz, const RealInput phi_delta,
         const RealInput phi_at, const RealInput prefactor, const RealInput density,
         const RealInput taper, const RealInput preionisation,
         const RealInput raman_rates, const nb::object material,
         const bool prepare_only) {
        TestState self(owner);
        if (self.raman() &&
            (raman_rates.ndim() != 2 || raman_rates.shape(0) != self.stages() ||
             raman_rates.shape(1) != self.raman_terms()))
          throw std::invalid_argument(
              "raman_rates must have shape (n_stages, n_terms)");
        auto scalars = trial_from(phi_delta, phi_at, prefactor, density, taper,
                                  preionisation, raman_rates);
        if (!material.is_none()) {
          const auto values = nb::cast<nb::dict>(material);
          scalars.kerr_density =
              real_vector(exact_array<RealInput>(values["kerr_density"]));
          scalars.plasma_densities =
              real_vector(exact_array<RealInput>(values["plasma_densities"]));
          scalars.plasma_preionisation =
              real_vector(exact_array<RealInput>(values["plasma_preionisation"]));
          scalars.raman_densities =
              real_vector(exact_array<RealInput>(values["raman_densities"]));
          if (values.contains("flow_section")) {
            scalars.flow_section = nb::cast<std::size_t>(values["flow_section"]);
            scalars.flow_weights =
                real_vector(exact_array<RealInput>(values["flow_weights"]));
          }
          if (values.contains("transform_weights"))
            scalars.transform_weights =
                real_vector(exact_array<RealInput>(values["transform_weights"]));
          if (values.contains("dispersive_densities"))
            scalars.dispersive_densities =
                real_vector(exact_array<RealInput>(values["dispersive_densities"]));
        }
        std::vector<double> partials;
        {
          const nb::gil_scoped_release release;
          partials = self.trial_step(dz, scalars, prepare_only);
        }
        return real_output(std::move(partials), 3);
      },
      "workspace"_a, "dz"_a, "phi_delta"_a.noconvert(), "phi_at"_a.noconvert(),
      "prefactor"_a.noconvert(), "density"_a.noconvert(), "taper"_a.noconvert(),
      "preionisation"_a.noconvert(), "raman_rates"_a.noconvert(),
      "material"_a = nb::none(), "prepare_only"_a = false);
  testing.def(
      "adopt",
      [](BoundWorkspace& owner, const ComplexInput field, const double dz) {
        TestState self(owner);
        const auto held = complex_vector(field);
        const nb::gil_scoped_release release;
        self.adopt(held, dz);
      },
      "workspace"_a, "field"_a.noconvert(), "dz"_a);
  testing.def(
      "stage_to_host",
      [](BoundWorkspace& owner, const std::size_t stage) {
        TestState self(owner);
        const auto count = self.elements();
        auto storage = native::crossing::host_buffer(2 * count);
        {
          const nb::gil_scoped_release release;
          self.stage_to_host(
              stage, {reinterpret_cast<native::Complex*>(storage.get()), count});
        }
        return complex_buffer_output(std::move(storage),
                                     {self.columns(), count / self.columns()});
      },
      "workspace"_a, "stage"_a);
  testing.def(
      "projected_to_host",
      [](BoundWorkspace& owner, const bool coarse) {
        TestState self(owner);
        std::vector<native::Complex> values;
        {
          const nb::gil_scoped_release release;
          values = self.projected_to_host(coarse);
        }
        return complex_output(values, {self.columns(), values.size() / self.columns()});
      },
      "workspace"_a, "coarse"_a = false);
  testing.def(
      "weighted_error",
      [](BoundWorkspace& owner, const RealInput scale) {
        TestState self(owner);
        std::vector<double> partials;
        const auto held_scale = real_vector(scale);
        {
          const nb::gil_scoped_release release;
          partials = self.weighted_error(held_scale);
        }
        return real_output(std::move(partials), 1);
      },
      "workspace"_a, "scale"_a.noconvert());
  testing.def(
      "form_flow_actions",
      [](BoundWorkspace& owner, const std::size_t section, const double dz,
         const RealInput fractions) {
        TestState self(owner);
        const auto held_fractions = real_vector(fractions);
        std::vector<double> measured;
        {
          const nb::gil_scoped_release release;
          measured = self.form_flow_actions(section, dz, held_fractions);
        }
        return real_output(std::move(measured), 1);
      },
      "workspace"_a, "section"_a, "dz"_a, "fractions"_a.noconvert());
  testing.def(
      "flow_actions_to_host",
      [](BoundWorkspace& owner, const bool inverse) {
        TestState self(owner);
        std::vector<native::Complex> values;
        {
          const nb::gil_scoped_release release;
          values = self.flow_actions_to_host(inverse);
        }
        return complex_output(values, {values.size()});
      },
      "workspace"_a, "inverse"_a);
  // solprop SEP 47 phase 4: the last exponential Adams trial's coupled block and
  // exponent, observations for tests.
  testing.def(
      "adams_block_to_host",
      [](BoundWorkspace& owner) {
        TestState self(owner);
        std::vector<native::Complex> values;
        {
          const nb::gil_scoped_release release;
          values = self.adams_block_to_host();
        }
        return complex_output(values, {values.size()});
      },
      "workspace"_a);
  testing.def(
      "adams_exponent_to_host",
      [](BoundWorkspace& owner) {
        TestState self(owner);
        std::vector<native::Complex> values;
        {
          const nb::gil_scoped_release release;
          values = self.adams_exponent_to_host();
        }
        return complex_output(values, {values.size()});
      },
      "workspace"_a);
  testing.def(
      "adams_exponential_trial",
      [](BoundWorkspace& owner, const double dz, const IndexInput slots,
         const RealInput predictor, const RealInput corrector, const RealInput rows,
         const RealInput divisor, const RealInput phi_delta, const RealInput phi_at,
         const RealInput prefactor, const RealInput density, const RealInput taper,
         const RealInput preionisation, const RealInput raman_rates,
         const nb::dict material, const RealInput sample_phi_delta,
         const RealInput sample_phi_at, const RealInput sample_prefactor,
         const RealInput sample_density, const RealInput sample_taper,
         const RealInput sample_preionisation, const RealInput sample_raman_rates,
         const nb::dict sample_material, const RealInput spans,
         const RealInput comp_predictor, const RealInput comp_corrector,
         const RealInput comp_rows, const RealInput rate, const RealInput carrier) {
        TestState self(owner);
        auto weights = exponential_weights(slots, predictor, corrector, rows);
        carrier_frame(weights, carrier, weights.width - 1);
        const auto rate_weights = real_vector(rate);
        // A varying operator's compensation tables (solprop's `compensation_tables`,
        // numpy_ref/adams_tables.py); empty for a constant one.
        weights.compensated = !rate_weights.empty();
        if (weights.compensated) {
          const auto span_values = real_vector(spans);
          const auto predictor_values = real_vector(comp_predictor);
          const auto corrector_values = real_vector(comp_corrector);
          const auto row_values = real_vector(comp_rows);
          if (span_values.size() + 1 != weights.width ||
              predictor_values.size() != weights.order ||
              corrector_values.size() != weights.order + 1 ||
              row_values.size() != weights.n_rows * weights.width)
            throw std::invalid_argument(
                "Adams compensation tables have inconsistent extents");
          for (std::size_t t = 0; t < span_values.size(); ++t)
            weights.span[t] = span_values[t];
          for (std::size_t t = 0; t < predictor_values.size(); ++t)
            weights.comp_predictor[t] = predictor_values[t];
          for (std::size_t t = 0; t < corrector_values.size(); ++t)
            weights.comp_corrector[t] = corrector_values[t];
          for (std::size_t row = 0; row < weights.n_rows; ++row)
            for (std::size_t t = 0; t < weights.width; ++t)
              weights.comp_rows[row][t] = row_values[row * weights.width + t];
        }
        const auto scale = real_vector(divisor);
        const auto step = adams_scalars(phi_delta, phi_at, prefactor, density, taper,
                                        preionisation, raman_rates, material);
        const auto sample = adams_scalars(
            sample_phi_delta, sample_phi_at, sample_prefactor, sample_density,
            sample_taper, sample_preionisation, sample_raman_rates, sample_material);
        std::vector<double> norms;
        {
          const nb::gil_scoped_release release;
          norms = self.open_state().adams_exponential_trial(dz, weights, scale, step,
                                                            sample, rate_weights);
        }
        return real_output(std::move(norms), 1);
      },
      "workspace"_a, "dz"_a, "slots"_a.noconvert(), "predictor"_a.noconvert(),
      "corrector"_a.noconvert(), "rows"_a.noconvert(), "divisor"_a.noconvert(),
      "phi_delta"_a.noconvert(), "phi_at"_a.noconvert(), "prefactor"_a.noconvert(),
      "density"_a.noconvert(), "taper"_a.noconvert(), "preionisation"_a.noconvert(),
      "raman_rates"_a.noconvert(), "material"_a, "sample_phi_delta"_a.noconvert(),
      "sample_phi_at"_a.noconvert(), "sample_prefactor"_a.noconvert(),
      "sample_density"_a.noconvert(), "sample_taper"_a.noconvert(),
      "sample_preionisation"_a.noconvert(), "sample_raman_rates"_a.noconvert(),
      "sample_material"_a, "spans"_a.noconvert(), "comp_predictor"_a.noconvert(),
      "comp_corrector"_a.noconvert(), "comp_rows"_a.noconvert(), "rate"_a.noconvert(),
      "carrier"_a.noconvert());
  testing.def(
      "adams_exponential_trial_at",
      [](BoundWorkspace& owner, const double z, const double dz, const RealInput nodes,
         const IndexInput slots, const std::size_t order,
         const std::vector<std::size_t>& orders, const bool compensated,
         const RealInput carrier) {
        TestState self(owner);
        auto weights = exponential_weights_at(nodes, slots, order, orders, compensated);
        carrier_frame(weights, carrier, weights.width - 1);
        const double end = z + dz;
        auto trial = self.stage_trial(z, end);
        const auto rate = compensated ? self.stage_rate(end) : std::vector<double>{};
        std::vector<double> norms;
        {
          const nb::gil_scoped_release release;
          norms = self.open_state().adams_exponential_trial(dz, weights, {}, trial.step,
                                                            trial.sample, rate);
        }
        return real_output(std::move(norms), 1);
      },
      "workspace"_a, "z"_a, "dz"_a, "nodes"_a.noconvert(), "slots"_a.noconvert(),
      "order"_a, "orders"_a, "compensated"_a, "carrier"_a.noconvert());
  testing.def(
      "adams_exponential_accept",
      [](BoundWorkspace& owner, const double dz, const std::size_t free,
         const IndexInput dense, const bool interpolate) {
        TestState self(owner);
        auto slopes = adams_rows(dense);
        slopes.width = interpolate ? slopes.width + 1 : 0;
        const nb::gil_scoped_release release;
        self.open_state().adams_exponential_accept(dz, free, slopes);
      },
      "workspace"_a, "dz"_a, "free"_a, "dense"_a.noconvert(), "interpolate"_a);
  testing.def(
      "adams_start",
      [](BoundWorkspace& owner, const std::size_t slot, const RealInput phi_delta,
         const RealInput phi_at, const RealInput prefactor, const RealInput density,
         const RealInput taper, const RealInput preionisation,
         const RealInput raman_rates, const nb::dict material) {
        TestState self(owner);
        const auto scalars = adams_scalars(phi_delta, phi_at, prefactor, density, taper,
                                           preionisation, raman_rates, material);
        const nb::gil_scoped_release release;
        self.open_state().adams_start(slot, scalars);
      },
      "workspace"_a, "slot"_a, "phi_delta"_a.noconvert(), "phi_at"_a.noconvert(),
      "prefactor"_a.noconvert(), "density"_a.noconvert(), "taper"_a.noconvert(),
      "preionisation"_a.noconvert(), "raman_rates"_a.noconvert(), "material"_a);
  testing.def(
      "adams_compensation_start",
      [](BoundWorkspace& owner, const std::size_t slot, const RealInput rate) {
        TestState self(owner);
        const auto weights = real_vector(rate);
        const nb::gil_scoped_release release;
        self.open_state().adams_compensation_start(slot, weights);
      },
      "workspace"_a, "slot"_a, "rate"_a.noconvert());
  testing.def(
      "adams_hold",
      [](BoundWorkspace& owner) {
        TestState self(owner);
        self.open_state().adams_hold();
      },
      "workspace"_a);
  testing.def(
      "adams_restore",
      [](BoundWorkspace& owner) {
        TestState self(owner);
        self.open_state().adams_restore();
      },
      "workspace"_a);
  testing.def(
      "adams_rows_to_host",
      [](BoundWorkspace& owner, const std::string& name) {
        TestState self(owner);
        auto& state = self.open_state();
        const auto extents = state.adams_row_extents(name);
        // The array's own storage is written once, as the host engine's copy is.
        auto storage =
            std::make_unique<ComplexStorage>(extents[0] * extents[1] * extents[2]);
        {
          const nb::gil_scoped_release release;
          state.adams_rows_to_host(name, storage->data(), storage->size());
        }
        auto* pointer = storage.get();
        nb::capsule keeper(pointer, [](void* value) noexcept {
          const std::unique_ptr<ComplexStorage> held(
              static_cast<ComplexStorage*>(value));
        });
        static_cast<void>(storage.release()); // the capsule owns it now
        const std::array<std::size_t, 3> shape{extents[0], extents[1], extents[2]};
        return ComplexOutput(pointer->data(), shape.size(), shape.data(), keeper);
      },
      "workspace"_a, "name"_a);
  testing.def(
      "adams_stage_scalars",
      [](BoundWorkspace& owner, const double z0, const double z1) {
        TestState self(owner);
        const auto trial = self.stage_trial(z0, z1);
        const auto& step = trial.step;
        // The first stage row's values; the test of the native evaluation reads
        // them against solprop's bundle.
        nb::dict out;
        out["phi_delta"] = step.phi_delta;
        out["phi_at"] = step.phi_at;
        out["prefactor"] = step.prefactor;
        out["density"] = step.density;
        out["taper"] = step.taper;
        out["preionisation"] = step.preionisation;
        out["raman_rates"] = step.raman_rates;
        out["kerr_density"] = step.kerr_density;
        out["plasma_densities"] = step.plasma_densities;
        out["plasma_preionisation"] = step.plasma_preionisation;
        out["raman_densities"] = step.raman_densities;
        out["dispersive_densities"] = step.dispersive_densities;
        out["flow_section"] = step.flow_section;
        out["flow_weights"] = step.flow_weights;
        out["transform_weights"] = step.transform_weights;
        return out;
      },
      "workspace"_a, "z0"_a, "z1"_a);
  testing.def(
      "adams_stage_rate",
      [](BoundWorkspace& owner, const double z) {
        TestState self(owner);
        return self.stage_rate(z);
      },
      "workspace"_a, "z"_a);
  testing.def(
      "has_stage_tables",
      [](BoundWorkspace& owner) {
        TestState self(owner);
        return self.has_stage_tables();
      },
      "workspace"_a);
  testing.def(
      "table_rate_at",
      [](BoundWorkspace& owner, const double z, const bool before) {
        TestState self(owner);
        const nb::gil_scoped_release release;
        self.table_rate_at(z, before);
      },
      "workspace"_a, "z"_a, "before"_a);
  testing.def(
      "phase_sum_to_host",
      [](BoundWorkspace& owner, const RealInput weights) {
        TestState self(owner);
        const auto held = real_vector(weights);
        auto& state = self.open_state();
        std::vector<native::Complex> values;
        {
          const nb::gil_scoped_release release;
          values = state.phase_sum_to_host(held);
        }
        return complex_output(values, {self.columns(), values.size() / self.columns()});
      },
      "workspace"_a, "weights"_a.noconvert());
  testing.def(
      "promote_fsal",
      [](BoundWorkspace& owner) {
        TestState self(owner);
        self.promote_fsal();
      },
      "workspace"_a);
  testing.def(
      "adams_march_run",
      [](BoundWorkspace& owner, const nb::dict march, const nb::dict bounds) {
        TestState self(owner);
        return self.march_run(march, bounds);
      },
      "workspace"_a, "march"_a, "bounds"_a);
  testing.def(
      "rk_march_run",
      [](BoundWorkspace& owner, const nb::dict march, const nb::dict bounds) {
        TestState self(owner);
        return self.rk_march_run(march, bounds);
      },
      "workspace"_a, "march"_a, "bounds"_a);
  // A Runge-Kutta trial's scaled error norm from its partials (solprop SEP 45 phase 2).
  testing.def(
      "rk_score",
      [](const RealInput partials, const double rtol, const double atol) {
        if (partials.ndim() != 2 || partials.shape(0) != 3)
          throw std::invalid_argument("a trial's partials have shape (3, slots)");
        return native::rk::score(partials.data(), partials.shape(1), rtol, atol);
      },
      "partials"_a, "rtol"_a, "atol"_a);
  // The decision traces' configuration: a run's march policy and schedule, taken from
  // its records without starting it.
  testing.def(
      "configure_march",
      [](BoundWorkspace& owner, const nb::handle records) {
        owner.workspace().configure_march(
            native::records::read<native::RunRecordsRecord>(records, "records"));
      },
      "workspace"_a, "records"_a,
      nb::sig("def configure_march(workspace: solprop_native._core.Workspace, "
              "records: solprop.backends.records.RunRecords) -> None"));
  // Load a start into the state as a run would, for the tests that step it directly.
  testing.def(
      "load",
      [](BoundWorkspace& owner, const ComplexInput field, const ComplexInput noise) {
        auto& workspace = owner.workspace();
        const nb::gil_scoped_release release;
        workspace.load(complex_span(field), complex_span(noise));
      },
      "workspace"_a, "field"_a.noconvert(), "noise"_a.noconvert());
  testing.def(
      "field",
      [](BoundWorkspace& owner) {
        TestState self(owner);
        auto& state = self.open_state();
        const auto count = self.elements();
        auto storage = native::crossing::host_buffer(2 * count);
        {
          const nb::gil_scoped_release release;
          state.field_to_host(
              {reinterpret_cast<native::Complex*>(storage.get()), count});
        }
        return complex_buffer_output(std::move(storage),
                                     {self.columns(), count / self.columns()});
      },
      "workspace"_a);
  testing.def(
      "finish",
      [](BoundWorkspace& owner, const double dz) {
        auto& state = owner.workspace().state();
        const nb::gil_scoped_release release;
        state.finish(dz);
      },
      "workspace"_a, "dz"_a);
  testing.def(
      "clear_fsal",
      [](BoundWorkspace& owner) { owner.workspace().state().clear_fsal(); },
      "workspace"_a);
  // A Runge-Kutta trial on the stage tables at the setup's nodes, as the march takes
  // it (solprop SEP 45 phase 2): its partials `(3, slots)`, its stages kept in the
  // state.
  testing.def(
      "rk_trial",
      [](BoundWorkspace& owner, const double z, const double dz) {
        auto& workspace = owner.workspace();
        workspace.require_unleased();
        std::vector<double> partials;
        {
          const nb::gil_scoped_release release;
          partials = workspace.staged_trial(z, dz, false);
        }
        return real_output(std::move(partials), 3);
      },
      "workspace"_a, "z"_a, "dz"_a);
  // A weighted controller's norm of a trial's partials, as the march scores it.
  testing.def(
      "weighted_score",
      [](BoundWorkspace& owner, const RealInput partials, const double rtol,
         const double atol, const RealInput weights) {
        if (partials.ndim() != 2 || partials.shape(0) != 3)
          throw std::invalid_argument("a trial's partials have shape (3, slots)");
        const auto values = real_vector(partials);
        const auto scale = real_vector(weights);
        return native::engines::ResidentRungeKutta::weighted_norm(
            owner.workspace().state(), values, rtol, atol, scale);
      },
      "workspace"_a, "partials"_a.noconvert(), "rtol"_a, "atol"_a,
      "weights"_a.noconvert());
  // A z table's value at `z` through the one slot cache (solprop SEP 29 G1).
  testing.def(
      "table_values",
      [](BoundWorkspace& owner, const std::string& kind, const double z) {
        auto& workspace = owner.workspace();
        auto& state = workspace.state();
        auto& slots = workspace.z_slots();
        if (!slots.present())
          throw std::logic_error("no z table's slots were configured");
        const auto known =
            native::names(static_cast<const native::TableKind*>(nullptr));
        // NOLINTNEXTLINE(readability-qualified-auto): an iterator, a class on MSVC
        const auto found = std::ranges::find_if(
            known, [&kind](const auto& entry) { return kind == entry.first; });
        if (found == known.end())
          throw std::invalid_argument("a z table is phi, prefactor, rate or flux");
        std::vector<native::Complex> values;
        {
          const nb::gil_scoped_release release;
          const std::size_t slot = slots.load(state, {z}).front();
          values = state.table_terms_to_host(found->second, slot);
        }
        return complex_output(
            values, {workspace.columns(), values.size() / workspace.columns()});
      },
      "workspace"_a, "kind"_a, "z"_a);
  // The slot each position's z-table value is held in, evaluating only those no slot
  // holds (solprop SEP 29 G1; at most the slots a call).
  testing.def(
      "table_load",
      [](BoundWorkspace& owner, const std::vector<double>& positions) {
        auto& workspace = owner.workspace();
        auto& state = workspace.state();
        auto& slots = workspace.z_slots();
        if (!slots.present())
          throw std::logic_error("no z table's slots were configured");
        const nb::gil_scoped_release release;
        return slots.load(state, positions);
      },
      "workspace"_a, "positions"_a);
  // Write a history row `(n_slots, rows, n_omega)` straight into the resident Adams
  // history, as a resumed run's start does.
  testing.def(
      "adams_rows_from_host",
      [](BoundWorkspace& owner, const std::string& name, const ComplexInput rows) {
        auto& state = owner.workspace().state();
        const auto extents = state.adams_row_extents(name);
        if (rows.ndim() != 3 || rows.shape(0) != extents[0] ||
            rows.shape(1) != extents[1] || rows.shape(2) != extents[2])
          native::refuse("the Adams history row '", name, "' must have shape (",
                         extents[0], ", ", extents[1], ", ", extents[2], ")");
        const auto values = complex_vector(rows);
        const nb::gil_scoped_release release;
        state.adams_rows_from_host(name, values);
      },
      "workspace"_a, "name"_a, "rows"_a.noconvert());
  // The far-field statistics' banded pass on the state (solprop SEP 23 section 6).
  testing.def(
      "aperture_statistics",
      [](BoundWorkspace& owner, const std::uint64_t flags, const RealInput apertures,
         const int exponent, const RealInput regions, const double intensity_factor) {
        if (apertures.ndim() != 2 || regions.ndim() != 2 || regions.shape(1) != 4)
          throw std::invalid_argument(
              "aperture tables must be (n_apertures, 11 + n_omega) and regions "
              "(n_regions, 4)");
        const auto held_apertures = real_vector(apertures);
        const auto held_regions = real_vector(regions);
        auto& state = owner.workspace().state();
        std::vector<double> values;
        {
          const nb::gil_scoped_release release;
          values = state.aperture_statistics(flags, held_apertures, exponent,
                                             held_regions, intensity_factor);
        }
        return real_output(std::move(values), 1);
      },
      "workspace"_a, "flags"_a, "apertures"_a.noconvert(), "exponent"_a,
      "regions"_a.noconvert(), "intensity_factor"_a);
  // March the low-storage replay's engine over consecutive positions, each step
  // finished as the state's finish alone, and return each step's squared norm over the
  // norm mask from the passes' own reduction (the forced-thread folds' reading). The
  // run's records must be configured (`configure_march`).
  testing.def(
      "low_storage_norms",
      [](BoundWorkspace& owner, const std::vector<double>& positions) {
        auto& workspace = owner.workspace();
        workspace.require_unleased();
        auto engine = workspace.low_storage_engine();
        auto& state = workspace.state();
        std::vector<double> norms;
        {
          const nb::gil_scoped_release release;
          for (std::size_t index = 0; index + 1 < positions.size(); ++index) {
            const double dz = positions[index + 1] - positions[index];
            static_cast<void>(engine->trial(positions[index], dz));
            norms.push_back(engine->squared());
            state.finish(dz);
          }
        }
        return real_output(norms.empty() ? std::vector<double>{} : std::move(norms), 1);
      },
      "workspace"_a, "positions"_a);
  // The largest accepted step's quadrature score since the last read, drained.
  testing.def(
      "quadrature_peak",
      [](BoundWorkspace& owner) { return owner.workspace().state().quadrature_peak(); },
      "workspace"_a);
  testing.def(
      "table_evaluations",
      [](BoundWorkspace& owner) { return owner.workspace().z_slots().evaluations; },
      "workspace"_a);
  testing.def(
      "prepare_raman",
      [](BoundWorkspace& owner, const RealInput rates, const RealInput densities) {
        if (rates.ndim() != 1 || rates.shape(0) != owner.raman_terms())
          throw std::invalid_argument("Raman decay rates must have shape (n_terms,)");
        const auto held_rates = real_vector(rates);
        const auto held_densities = real_vector(densities);
        auto& state = owner.workspace().state();
        const nb::gil_scoped_release release;
        state.prepare_raman(held_rates, held_densities);
      },
      "workspace"_a, "rates"_a.noconvert(), "densities"_a.noconvert());
  // Bind a run's monitor weights, the problem's occupancy and the edge's tables as a
  // run's start binds them, for the tests that read the state directly.
  testing.def(
      "bind_monitor",
      [](BoundWorkspace& owner, const nb::handle monitor, const nb::handle edge,
         const nb::handle occupancy) {
        const auto weights =
            native::records::read<native::MonitorBinding>(monitor, "monitor");
        const auto tables =
            native::records::read<std::optional<native::EdgeBinding>>(edge, "edge");
        const auto buckets = native::records::read<std::optional<native::Occupancy>>(
            occupancy, "occupancy");
        auto& state = owner.workspace().state();
        // Holding the interpreter lock: a strip's plan goes through FFTW's planner.
        state.bind_monitor(weights.weights, weights.column_weights);
        if (weights.split && buckets)
          state.bind_occupancy(buckets->buckets, buckets->count);
        if (tables) {
          state.bind_edge(tables->faces, tables->clip, tables->totals,
                          tables->synthesis, tables->first_row);
          if (tables->lines)
            state.bind_edge_lines(*tables->lines);
        }
      },
      "workspace"_a, "monitor"_a, "edge"_a.none(), "occupancy"_a.none(),
      nb::sig("def bind_monitor(workspace: solprop_native._core.Workspace, monitor: "
              "collections.abc.Mapping[str, object], edge: "
              "collections.abc.Mapping[str, object] | None, occupancy: "
              "collections.abc.Mapping[str, object] | None) -> None"));
  testing.def(
      "edge_reading",
      [](BoundWorkspace& owner, const bool faces, const bool whole, const bool totals,
         const bool strip, const bool current) {
        auto& state = owner.workspace().state();
        native::EdgeNumbers numbers;
        {
          const nb::gil_scoped_release release;
          numbers = state.edge_reading(faces, whole, totals, strip, current);
        }
        return nb::make_tuple(real_output(std::move(numbers.sums), 1),
                              real_output(std::move(numbers.columns), 1),
                              real_output(std::move(numbers.strip), 1),
                              numbers.current);
      },
      "workspace"_a, "faces"_a, "whole"_a, "totals"_a, "strip"_a, "current"_a);
  testing.def(
      "statistics",
      [](BoundWorkspace& owner, const std::uint64_t flags, const RealInput windows,
         const RealInput on_axis, const double density, const double taper,
         const double preionisation, const double intensity_factor,
         const nb::object channel_preionisation, const nb::object power_weight) {
        TestState self(owner);
        const auto held_weight =
            power_weight.is_none() ? std::vector<double>{}
                                   : real_vector(exact_array<RealInput>(power_weight));
        const auto held_windows = real_vector(windows);
        const auto held_on_axis = real_vector(on_axis);
        const auto initial =
            channel_preionisation.is_none()
                ? std::vector<double>{}
                : real_vector(exact_array<RealInput>(channel_preionisation));
        std::vector<double> values;
        {
          const nb::gil_scoped_release release;
          values =
              self.statistics(flags, held_windows, held_on_axis, density, taper,
                              preionisation, intensity_factor, initial, held_weight);
        }
        return real_output(std::move(values), 1);
      },
      "workspace"_a, "flags"_a, "windows"_a.noconvert(), "on_axis"_a.noconvert(),
      "density"_a, "taper"_a, "preionisation"_a, "intensity_factor"_a,
      "channel_preionisation"_a = nb::none(), "power_weight"_a = nb::none());
  testing.def(
      "spatial_statistics",
      [](BoundWorkspace& owner, const std::uint64_t flags,
         const double intensity_factor, const double z) {
        TestState self(owner);
        std::vector<double> values;
        {
          const nb::gil_scoped_release release;
          values = self.spatial_statistics(flags, intensity_factor, z);
        }
        return real_output(std::move(values), 1);
      },
      "workspace"_a, "flags"_a, "intensity_factor"_a, "z"_a);
  testing.def(
      "planned_statistics",
      [](BoundWorkspace& owner, const BoundStatisticsPlan& plan, const double z) {
        TestState self(owner);
        const auto& held = plan.plan();
        std::vector<double> values;
        {
          const nb::gil_scoped_release release;
          values = self.planned_statistics(held, z);
        }
        return real_output(std::move(values), 1);
      },
      "workspace"_a, "plan"_a, "z"_a);
  testing.def(
      "plane_fluence_trace",
      [](BoundWorkspace& owner, const double center_x, const double center_y,
         const double dx, const double dy, const double extent, const double spacing,
         const std::size_t n_radii, const RealInput cosines, const RealInput sines) {
        TestState self(owner);
        if (cosines.ndim() != 1 || sines.ndim() != 1)
          throw std::invalid_argument("fluence rings must be one-dimensional");
        const auto held_cosines = real_vector(cosines);
        const auto held_sines = real_vector(sines);
        std::vector<double> values;
        {
          const nb::gil_scoped_release release;
          values = self.plane_fluence_trace(center_x, center_y, dx, dy, extent, spacing,
                                            n_radii, held_cosines, held_sines);
        }
        return real_output(std::move(values), 1);
      },
      "workspace"_a, "center_x"_a, "center_y"_a, "dx"_a, "dy"_a, "extent"_a,
      "spacing"_a, "n_radii"_a, "cosines"_a.noconvert(), "sines"_a.noconvert());
  testing.def(
      "axis_spectrum",
      [](BoundWorkspace& owner, const RealInput matrix) {
        TestState self(owner);
        if (matrix.ndim() != 2)
          throw std::invalid_argument("an on-axis matrix must be two-dimensional");
        const auto held = real_vector(matrix);
        std::vector<native::Complex> values;
        {
          const nb::gil_scoped_release release;
          values = self.axis_spectrum(held);
        }
        return complex_output(values, {values.size()});
      },
      "workspace"_a, "matrix"_a.noconvert());
  testing.def(
      "monitor_reading",
      [](BoundWorkspace& owner) {
        TestState self(owner);
        std::vector<double> values;
        {
          const nb::gil_scoped_release release;
          values = self.monitor_reading();
        }
        return real_output(std::move(values), 1);
      },
      "workspace"_a);
  testing.def(
      "quadrature_error",
      [](BoundWorkspace& owner) {
        TestState self(owner);
        return self.quadrature_error();
      },
      "workspace"_a);
}

} // namespace solprop_native::bindings
