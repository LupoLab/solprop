// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The extension module `_core` (`NB_MODULE`): its one table of bindings and the classes
// only it binds. It registers the exception types, the device-memory probe, the FFT
// plan timing and wisdom export, the runtime's configuration and build record, the
// session with its events, status and batches, the statistics plan, the probe and the
// workspace; calls the other binding files' entry points in the order bind.hpp gives;
// exports every field table (`_core.schemas()`); and registers the runtime's shutdown
// at interpreter exit.
//
// Types: `BoundSession` (solprop SEP 47's session over a leased workspace),
// `BatchRecord`, `BoundProbe` (the timing probe, SEP 52 section 5.5) and
// `PythonHostSource` (a Python object as the march's host source, SEP 49); the
// `ExportRecipe` record an export takes; and the members of `BoundWorkspace` and
// `BoundStatisticsPlan` (module.hpp) that make or read them.

#include "module.hpp"

#include "arrays.hpp"
#include "bind.hpp"
#include "helpers.hpp"
#include "march.hpp"
#include "records.hpp"
#include "solprop_native/problem/problem.hpp"
#include "solprop_native/runtime/crossing.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/fields.hpp"
#include "solprop_native/runtime/host_source.hpp"
#include "solprop_native/runtime/refuse.hpp"
#include "solprop_native/session/session.hpp"
#include "solprop_native/session/statistics_plan.hpp"
#include "solprop_native/session/transverse.hpp"
#include "solprop_native/session/watch.hpp"
#include "solprop_native/transforms/fftw_wisdom.hpp"
#include "solprop_native/transforms/reference_timing.hpp"
#include "solprop_native/workspace/run_records.hpp"
#include "solprop_native/workspace/workspace.hpp"
#include "transverse.hpp"

#include <Kokkos_Core.hpp>

// `available_device_memory` calls the vendor runtime directly. Kokkos pulls
// these headers in transitively today; declaring them here is what makes the
// probe's own dependency explicit rather than a property of a Kokkos version.
// The guards resolve only after Kokkos_Core.hpp has defined them.
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

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
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace solprop_native::bindings {

using namespace nb::literals;

namespace {

// solprop SEP 49 section 6.2: a host source whose three entry points are a Python
// object's `begin(cost)`, `evaluate(field, z, purpose, out)` and `end(outcome)`. Each
// call takes the interpreter lock the march released; a Python exception is caught
// here, under that lock, and reported to the march as a `HostSourceError`, the
// exception itself kept by the Python object for solprop's session to raise. The
// object is borrowed: its state owns it, so no reference moves without the lock.
class PythonHostSource final : public native::HostSource {
public:
  explicit PythonHostSource(const nb::handle owner) : owner_(owner) {}
  void begin(const std::uint64_t cost) override {
    call([&] { owner_.attr("begin")(cost); });
  }
  void evaluate(const std::span<const std::complex<double>> field, const double z,
                const native::HostPurpose purpose,
                const std::span<std::complex<double>> source) override {
    call([&] {
      // Views of the engine's two host rows, valid for the call alone: the field
      // read-only, the source written in place. Each has an owner, a capsule that
      // frees nothing, because nanobind copies an ownerless array it hands to
      // Python, and a write into that copy would never reach the engine.
      const std::array<std::size_t, 1> in_shape{field.size()};
      const std::array<std::size_t, 1> out_shape{source.size()};
      const nb::capsule borrowed_in(field.data(), [](void* /*unused*/) noexcept {});
      const nb::capsule borrowed_out(source.data(), [](void* /*unused*/) noexcept {});
      const HostField in(field.data(), 1, in_shape.data(), borrowed_in);
      const ComplexOutput out(source.data(), 1, out_shape.data(), borrowed_out);
      owner_.attr("evaluate")(
          in, z, purpose == native::HostPurpose::startup ? "startup" : "predictor",
          out);
    });
  }
  void end(const native::HostOutcome outcome) override {
    call([&] { owner_.attr("end")(static_cast<int>(outcome)); });
  }

private:
  template <class Call> void call(const Call& body) {
    const nb::gil_scoped_acquire acquire;
    try {
      body();
    } catch (const nb::python_error& error) {
      throw native::HostSourceError(error.what());
    }
  }
  nb::handle owner_;
};

// solprop SEP 47: a batch's chunk as owning NumPy arrays, each made once from the
// vector the session moved into the batch (no copy), its capsule owning the storage.
template <class T>
nb::ndarray<nb::numpy, T, nb::c_contig, nb::device::cpu>
moved_array(std::vector<T>&& values, const std::size_t rows,
            const std::size_t columns) {
  auto storage = std::make_unique<std::vector<T>>(std::move(values));
  if (storage->size() != rows * columns)
    throw std::logic_error("a session batch's array has an inconsistent shape");
  auto* pointer = storage.get();
  nb::capsule owner(pointer, [](void* value) noexcept {
    const std::unique_ptr<std::vector<T>> held(static_cast<std::vector<T>*>(value));
  });
  static_cast<void>(storage.release()); // the capsule owns it now
  const std::array<std::size_t, 2> shape{rows, columns};
  return {pointer->data(), shape.size(), shape.data(), owner};
}

// The Python record of one batch: its event and its chunk's arrays, built once when the
// batch is handed over, with the interpreter lock held.
struct BatchRecord {
  explicit BatchRecord(native::session::Batch&& batch)
      : sequence(batch.sequence), event(std::move(batch.event)) {
    const std::size_t n_attempts =
        batch.attempts.size() / native::session::attempt_columns;
    const std::size_t n_rows = batch.row_z.size();
    attempts = nb::cast(moved_array(std::move(batch.attempts), n_attempts,
                                    native::session::attempt_columns));
    row_z = nb::cast(moved_array(std::move(batch.row_z), n_rows, 1));
    row_step = nb::cast(moved_array(std::move(batch.row_step), n_rows, 1));
    row_values = nb::cast(moved_array(std::move(batch.row_values), n_rows,
                                      n_rows == 0 ? 0 : batch.row_width));
    const std::size_t n_stats = batch.plane_stats.size();
    plane_stats = nb::cast(moved_array(std::move(batch.plane_stats), 1, n_stats));
  }
  std::uint64_t sequence;
  native::session::Event event;
  nb::object attempts;
  nb::object row_z;
  nb::object row_step;
  nb::object row_values;
  nb::object plane_stats;
};

/** A session's snapshot as Python reads it: the march's state, the field and each
    history row, every array owning its copy; None without one. */
nb::object snapshot_to(std::optional<native::Run::Snapshot> snapshot) {
  if (!snapshot)
    return nb::none();
  nb::dict held;
  for (std::size_t index = 0; index < snapshot->names.size(); ++index) {
    const auto& extent = snapshot->extents[index];
    held[snapshot->names[index].c_str()] = owned_complex(
        std::make_unique<ComplexStorage>(std::move(snapshot->rows[index])),
        {extent[0], extent[1], extent[2]});
  }
  const std::size_t n = snapshot->field.size();
  return nb::make_tuple(
      march_state_to(snapshot->state),
      owned_complex(std::make_unique<ComplexStorage>(std::move(snapshot->field)), {n}),
      held);
}

// A pending batch's ticket, `(sequence, generation)` (SEP 52 section 5.3).
using Ticket = std::pair<std::uint64_t, std::uint64_t>;

} // namespace

} // namespace solprop_native::bindings

namespace solprop_native {

namespace {

// An export's conversion (`solprop.backends.export.ExportRecipe`) with the route's
// power weight at the ticket's position, none unless `power`.
struct ExportRecipe {
  bool power = false;
  bool optics = false;
  std::optional<std::vector<double>> weight;
};

[[nodiscard]] constexpr auto describe(const ExportRecipe* /*unused*/) {
  using R = ExportRecipe;
  return fields::table<R>("ExportRecipe", fields::value("power", &R::power),
                          fields::value("optics", &R::optics),
                          fields::array("weight", &R::weight, fields::Extent::any()));
}

} // namespace

} // namespace solprop_native

namespace solprop_native::bindings {

using native::ExportRecipe;

// solprop SEP 47: a native execution session over a leased workspace (`native::Run`).
// The binding keeps the workspace's Python object alive while the session lives; the
// workspace closes the session first when it closes itself.
class BoundSession {
public:
  BoundSession(std::unique_ptr<native::Run> run, const std::size_t columns)
      : run_(std::move(run)), columns_(columns) {}
  ~BoundSession() { close(); }
  BoundSession(const BoundSession&) = delete;
  BoundSession& operator=(const BoundSession&) = delete;
  BoundSession(BoundSession&&) = delete;
  BoundSession& operator=(BoundSession&&) = delete;

  [[nodiscard]] native::session::Session& session() const { return run_->session(); }

  // The advance runs without the interpreter lock; its batch becomes a record with it.
  BatchRecord advance(const std::size_t work_limit, const double deadline) {
    native::session::Batch batch;
    {
      const nb::gil_scoped_release release;
      batch = session().advance(work_limit, deadline);
    }
    return BatchRecord(std::move(batch));
  }
  // Takes the session's lock: bound with the interpreter lock released.
  void acknowledge(const std::uint64_t sequence) { session().acknowledge(sequence); }
  // Fill `out` with the pending ticket's field, converted by `recipe` natively.
  void
  export_field(const Ticket ticket, const nb::handle recipe,
               nb::ndarray<std::complex<double>, nb::c_contig, nb::device::cpu> out) {
    const auto held = native::records::read<ExportRecipe>(recipe, "recipe");
    const std::size_t n = session().elements();
    if (out.size() != n)
      native::refuse("an export destination has ", out.size(), " elements, expected ",
                     n);
    if (held.weight && held.weight->size() != n)
      native::refuse("an export's power weight has ", held.weight->size(),
                     " elements, expected ", n);
    const double* values = held.weight ? held.weight->data() : nullptr;
    auto* destination = reinterpret_cast<double*>(out.data());
    const nb::gil_scoped_release release;
    session().export_field(ticket.first, ticket.second, held.power, held.optics, values,
                           destination);
  }
  void request_stop() noexcept { session().request_stop(); }
  nb::tuple request_snapshot() {
    const auto receipt = session().request_snapshot();
    return nb::make_tuple(receipt.request, receipt.coalesced);
  }
  [[nodiscard]] native::session::Status status() const noexcept {
    return session().status();
  }
  // Continue a resumed run's monitor record, given as
  // `solprop.backends.records.MonitorContinuation`; takes the session's lock.
  void continue_monitor(const nb::handle continuation) {
    const auto held =
        native::records::read<native::ContinuationRecord>(continuation, "continuation");
    const auto record = native::to_native(held.record);
    const auto edge = native::to_native(held.edge);
    const nb::gil_scoped_release release;
    session().continue_monitor(record, held.plane, held.unwatched, edge);
  }
  // SEP 52: every record the session keeps, in one, each read under the session's
  // lock: the monitor's and the edge's (None without them), the split collar's cap
  // (None without one) and the quadrature peak, drained since the last read.
  nb::dict records(const std::optional<std::vector<std::string>>& names) {
    // Only the records asked for: the quadrature peak drains the state's score with a
    // fence, and the monitor's record is a dictionary of its every number, so a reader
    // that wants the cap alone (at every batch) takes neither.
    const auto wanted = [&names](const char* name) {
      return !names || std::ranges::find(*names, name) != names->end();
    };
    std::optional<native::watch::Record> monitor;
    std::optional<std::pair<native::transverse::Record, native::transverse::Finish>>
        edge;
    std::optional<double> cap;
    std::optional<double> peak;
    {
      const nb::gil_scoped_release release;
      if (wanted("monitor"))
        monitor = session().monitor_record();
      if (wanted("edge"))
        edge = session().edge_record();
      if (wanted("cap"))
        cap = session().monitor_cap();
      if (wanted("quadrature_peak"))
        peak = session().quadrature_peak();
    }
    nb::dict out;
    if (wanted("monitor"))
      out["monitor"] = monitor ? nb::object(watch_record_to(*monitor)) : nb::none();
    if (wanted("edge"))
      out["edge"] = edge ? nb::object(native::bindings::transverse_record_to(
                               edge->first, edge->second))
                         : nb::none();
    if (wanted("cap"))
      out["cap"] = cap ? nb::object(nb::float_(*cap)) : nb::none();
    if (peak)
      out["quadrature_peak"] = *peak;
    return out;
  }
  // solprop SEP 47 section 5.5: the Adams march's state with the orders of its
  // accepted steps from `since` (the call ledger), or None for a Runge-Kutta march.
  nb::object march_state(const std::size_t since) {
    std::optional<std::pair<march::State, std::vector<std::size_t>>> state;
    {
      const nb::gil_scoped_release release;
      state = run_->march_state(since);
    }
    if (!state)
      return nb::none();
    nb::dict out = march_state_to(state->first);
    out["orders"] = state->second;
    return out;
  }
  // The Adams march's history at the pending ticket (SEP 47 section 5.4), each part
  // copied once; None for a Runge-Kutta march or before any history.
  nb::object snapshot(const Ticket ticket) {
    std::optional<native::Run::Snapshot> held;
    {
      const nb::gil_scoped_release release;
      held = run_->snapshot(ticket.first, ticket.second);
    }
    return snapshot_to(std::move(held));
  }
  // A z table's `Phi` at `z`, the noise phase at the pending ticket's plane.
  ComplexOutput noise_phase(const Ticket ticket, const double z) {
    std::vector<native::Complex> values;
    {
      const nb::gil_scoped_release release;
      values = run_->noise_phase(ticket.first, ticket.second, z);
    }
    return complex_output(values, {columns_, values.size() / columns_});
  }
  // Issue a failed terminal batch for a failure outside the session; takes its lock.
  std::optional<BatchRecord> fail_outside() {
    std::optional<native::session::Batch> batch;
    {
      const nb::gil_scoped_release release;
      batch = session().fail_outside();
    }
    if (!batch)
      return std::nullopt;
    return BatchRecord(std::move(*batch));
  }
  // Takes the session's lock (waiting for an advance in hand): bound with the
  // interpreter lock released. The lease is released only once the session closed.
  void close() noexcept { static_cast<void>(run_->close()); }

private:
  std::unique_ptr<native::Run> run_;
  std::size_t columns_;
};

namespace {

// solprop SEP 47 section 6.4: a statistics evaluation lowered once from the problem,
// the one native implementation of each statistic (statistics_plan.hpp).
template <class Value>
native::statistics::Table<Value> table_from(const nb::handle value) {
  native::statistics::Table<Value> table;
  if (value.is_none())
    return table;
  const auto values = nb::cast<nb::dict>(value);
  table.edges = real_vector(exact_array<RealInput>(values["edges"]));
  if constexpr (std::is_same_v<Value, double>) {
    table.coefficients = real_vector(exact_array<RealInput>(values["coefficients"]));
  } else {
    const auto coefficients = exact_array<ComplexInput>(values["coefficients"]);
    table.coefficients.assign(coefficients.data(),
                              coefficients.data() + element_count(coefficients));
  }
  table.n_terms = nb::cast<std::size_t>(values["n_terms"]);
  table.n_values = nb::cast<std::size_t>(values["n_values"]);
  table.lower = nb::cast<double>(values["lower"]);
  table.upper = nb::cast<double>(values["upper"]);
  return table;
}

std::vector<native::statistics::Table<double>> tables_from(const nb::handle value) {
  std::vector<native::statistics::Table<double>> tables;
  for (const auto item : nb::cast<nb::list>(value))
    tables.push_back(table_from<double>(item));
  return tables;
}

} // namespace

BoundStatisticsPlan::BoundStatisticsPlan(const nb::dict values) {
  auto plan = std::make_shared<native::statistics::Plan>();
  namespace statistics = native::statistics;
  try {
    const auto route = nb::cast<int>(values["route"]);
    if (route < 0 || route > 2)
      throw std::invalid_argument("a statistics plan's route is 0, 1 or 2");
    plan->route = static_cast<statistics::Route>(route);
    for (const auto item : nb::cast<nb::list>(values["outputs"])) {
      const auto id = nb::cast<int>(item);
      if (id < 0 || id >= statistics::stat_count)
        throw std::invalid_argument("a statistics plan names an unknown statistic");
      plan->outputs.push_back(static_cast<statistics::Stat>(id));
    }
    plan->n_modes = nb::cast<std::size_t>(values["n_modes"]);
    plan->n_pol = nb::cast<std::size_t>(values["n_pol"]);
    plan->n_t = nb::cast<std::size_t>(values["n_t"]);
    plan->n_omega = nb::cast<std::size_t>(values["n_omega"]);
    plan->n_nodes = nb::cast<std::size_t>(values["n_nodes"]);
    plan->n_channels = nb::cast<std::size_t>(values["n_channels"]);
    plan->n_species = nb::cast<std::size_t>(values["n_species"]);
    plan->windows = real_vector(exact_array<RealInput>(values["windows"]));
    plan->intensity_factor = nb::cast<double>(values["intensity_factor"]);
    plan->t = real_vector(exact_array<RealInput>(values["t"]));
    const auto weight = nb::cast<int>(values["weight_kind"]);
    if (weight < 0 || weight > 3)
      throw std::invalid_argument("a statistics plan's weight kind is 0 to 3");
    plan->weight_kind = static_cast<statistics::Weight>(weight);
    plan->weight = real_vector(exact_array<RealInput>(values["weight"]));
    plan->weight_table = table_from<double>(values["weight_table"]);
    plan->on_axis = real_vector(exact_array<RealInput>(values["on_axis"]));
    plan->taper = table_from<double>(values["taper"]);
    plan->pad_scale = nb::cast<double>(values["pad_scale"]);
    plan->axis_node = nb::cast<std::int64_t>(values["axis_node"]);
    plan->plasma_density = table_from<double>(values["plasma_density"]);
    plan->plasma_preionisation = table_from<double>(values["plasma_preionisation"]);
    plan->channel_preionisation = tables_from(values["channel_preionisation"]);
    plan->envelope_plasma = nb::cast<bool>(values["envelope_plasma"]);
    plan->species_density = tables_from(values["species_density"]);
    for (const auto item : nb::cast<nb::list>(values["channel_species"]))
      plan->channel_species.push_back(nb::cast<std::size_t>(item));
    plan->potentials = real_vector(exact_array<RealInput>(values["potentials"]));
    const auto transverse = nb::cast<int>(values["transverse"]);
    if (transverse < 0 || transverse > 3)
      throw std::invalid_argument("a statistics plan's transverse kind is 0 to 3");
    plan->transverse = static_cast<statistics::Transverse>(transverse);
    plan->node_weights = real_vector(exact_array<RealInput>(values["node_weights"]));
    plan->aeff = table_from<double>(values["aeff"]);
    const auto profiles = nb::cast<nb::list>(values["profiles"]);
    if (profiles.size() != plan->profiles.size())
      throw std::invalid_argument("a statistics plan has three profile slots");
    for (std::size_t index = 0; index < plan->profiles.size(); ++index)
      plan->profiles[index] = table_from<double>(profiles[index]);
    plan->far_field_axis = nb::cast<bool>(values["far_field_axis"]);
    const auto apertures = exact_array<RealInput>(values["apertures"]);
    if (apertures.ndim() != 2)
      throw std::invalid_argument("a statistics plan's apertures are two-dimensional");
    plan->n_apertures = apertures.shape(0);
    plan->apertures = real_vector(apertures);
    plan->exponent = nb::cast<int>(values["exponent"]);
    const auto regions = exact_array<RealInput>(values["regions"]);
    if (regions.ndim() != 2)
      throw std::invalid_argument("a statistics plan's regions are two-dimensional");
    plan->n_regions = regions.shape(0);
    plan->regions = real_vector(regions);
    plan->n_absorbed = nb::cast<std::size_t>(values["n_absorbed"]);
    plan->root = table_from<std::complex<double>>(values["root"]);
    plan->spectral_prefactor = nb::cast<double>(values["spectral_prefactor"]);
    const auto widths = nb::cast<int>(values["widths"]);
    if (widths < 0 || widths > 4)
      throw std::invalid_argument("a statistics plan's width rule is 0 to 4");
    plan->widths = static_cast<statistics::Widths>(widths);
    plan->width_x = real_vector(exact_array<RealInput>(values["width_x"]));
    plan->width_y = real_vector(exact_array<RealInput>(values["width_y"]));
    plan->width_weights = real_vector(exact_array<RealInput>(values["width_weights"]));
    plan->width_n_x = nb::cast<std::size_t>(values["width_n_x"]);
    plan->width_n_y = nb::cast<std::size_t>(values["width_n_y"]);
    plan->width_terms = nb::cast<std::size_t>(values["width_terms"]);
    plan->ring_cosines = real_vector(exact_array<RealInput>(values["ring_cosines"]));
    plan->ring_sines = real_vector(exact_array<RealInput>(values["ring_sines"]));
    plan->e2_fraction = nb::cast<double>(values["e2_fraction"]);
    plan->zdw = tables_from(values["zdw"]);
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("a statistics plan has the wrong types");
  } catch (const nb::python_error&) {
    throw std::invalid_argument("a statistics plan lacks a field");
  }
  plan->validate();
  plan_ = std::move(plan);
}

namespace records = native::records;

// The timing probe (SEP 52 section 5.5), over the workspace that leases it.
class BoundProbe {
public:
  BoundProbe(std::unique_ptr<native::Probe> probe, const std::size_t columns)
      : probe_(std::move(probe)), columns_(columns) {}
  double trial(const double z, const double dz) {
    const nb::gil_scoped_release release;
    return probe_->trial(z, dz);
  }
  ComplexOutput field() {
    std::vector<native::Complex> values;
    {
      const nb::gil_scoped_release release;
      values = probe_->field();
    }
    return complex_output(values, {columns_, values.size() / columns_});
  }
  [[nodiscard]] std::size_t trials() const noexcept { return probe_->trials(); }
  [[nodiscard]] std::size_t evaluations() const { return probe_->evaluations(); }
  [[nodiscard]] bool low_storage() const noexcept { return probe_->low_storage(); }
  [[nodiscard]] double z() const noexcept { return probe_->z(); }
  [[nodiscard]] bool closed() const noexcept { return probe_->closed(); }
  void close() noexcept { probe_->close(); }

private:
  std::unique_ptr<native::Probe> probe_;
  std::size_t columns_;
};

// The workspace's members that make a session, a probe or a host source (module.hpp).
std::unique_ptr<BoundSession> BoundWorkspace::start(const nb::handle run,
                                                    const nb::object& host) {
  auto held = records::read<native::RunStart>(run, "run");
  bind_host(host);
  std::unique_ptr<native::Run> started;
  {
    const nb::gil_scoped_release release;
    started = workspace_->start(std::move(held));
  }
  return std::make_unique<BoundSession>(std::move(started), workspace_->columns());
}

std::unique_ptr<BoundProbe> BoundWorkspace::probe(const nb::handle run,
                                                  const nb::object& host) {
  const auto held = records::read<native::RunStart>(run, "start");
  bind_host(host);
  std::unique_ptr<native::Probe> probe;
  {
    const nb::gil_scoped_release release;
    probe = workspace_->probe(held);
  }
  return std::make_unique<BoundProbe>(std::move(probe), workspace_->columns());
}

void BoundWorkspace::bind_host(const nb::object& host) {
  workspace_->require_unleased();
  if (host.is_none()) {
    workspace_->bind_host_source(nullptr);
    host_owner_ = nb::object();
    return;
  }
  workspace_->bind_host_source(std::make_shared<PythonHostSource>(host));
  host_owner_ = host;
}

namespace {

/** Every table `schemas()` exports, by name. */
nb::dict schemas() {
  nb::dict tables;
  records::collect<native::RunRecordsRecord>(tables);
  records::collect<native::RunStart>(tables);
  records::collect<native::march::State>(tables);
  records::collect<native::rk::State>(tables);
  records::collect<native::Setup>(tables);
  records::collect<native::Problem::Dims>(tables);
  records::collect<native::Problem::Grid>(tables);
  records::collect<native::Problem::Spatial>(tables);
  records::collect<native::Problem::Flux>(tables);
  records::collect<native::Problem::Stepper>(tables);
  records::collect<native::Problem::Linear>(tables);
  records::collect<native::Problem::Absorber>(tables);
  records::collect<native::Problem::Transform>(tables);
  records::collect<native::Problem::Kerr>(tables);
  records::collect<native::Problem::Plasma>(tables);
  records::collect<native::Problem::Raman>(tables);
  records::collect<native::Problem::General>(tables);
  records::collect<native::Problem::Flow>(tables);
  records::collect<native::Problem::Dispersive>(tables);
  records::collect<RuntimeRecord>(tables);
  return tables;
}

} // namespace

} // namespace solprop_native::bindings

// The module's table names the binding files' classes and entry points unqualified.
using namespace solprop_native::bindings;

// NOLINTNEXTLINE(readability-function-size): the module's one table of bindings, kept whole
NB_MODULE(_core, module_) {
  // Each registration makes the Python exception type; the handle need not outlive it.
  const nb::exception<native::EnvelopeConvergenceError> envelope_convergence_error(
      module_, "EnvelopeConvergenceError");
  const nb::exception<native::BackendRefusal> backend_refusal(module_,
                                                              "BackendRefusal");
  module_.doc() = "Private nanobind seam for solprop-native";
  module_.def(
      "available_device_memory",
      []() -> nb::object {
#ifdef KOKKOS_ENABLE_CUDA
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
          throw std::runtime_error("cannot query CUDA free memory");
        return nb::int_(free_bytes);
#elif defined(KOKKOS_ENABLE_HIP)
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess)
          throw std::runtime_error("cannot query HIP free memory");
        return nb::int_(free_bytes);
#else
        return nb::none();
#endif
      },
      nb::sig("def available_device_memory() -> int | None"));
  // solprop issue #595: a captured plan's execution alone, against estimate's.
  module_.def(
      "time_fft_plan",
      [](const std::string& key, const std::string& wisdom, const std::string& effort) {
        nb::dict result;
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
        // With the GIL held, as every planning: it serialises FFTW's planner.
        const auto timed = native::time_fft_plan(key, wisdom, effort);
        result["plan"] = timed.plan;
        result["estimate"] = timed.estimate;
#else
        static_cast<void>(key);
        static_cast<void>(wisdom);
        static_cast<void>(effort);
#endif
        return result;
      },
      nb::arg("key"), nb::arg("wisdom"), nb::arg("effort"));
  module_.def("reference_stream_copies", &native::reference_stream_copies, "elements"_a,
              "warm"_a, "repeats"_a);
  module_.def("build_record", &build_record);
  // FFTW's wisdom as text, for the plan store the adapter owns
  // (solprop_native._wisdom). None where there is none to keep: a device build,
  // which plans with the device FFT, or a provider without FFTW's wisdom.
  module_.def(
      "export_fftw_wisdom",
      []() -> nb::object {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        return nb::none();
#else
        std::string text = native::export_fftw_wisdom();
        return text.empty() ? nb::none() : nb::cast(std::move(text));
#endif
      },
      nb::sig("def export_fftw_wisdom() -> str | None"));
  bind_problem(module_);
  // solprop SEP 47: the native execution session and its compact event records.
  const nb::exception<native::session::LifecycleError> lifecycle_error(
      module_, "SessionLifecycleError", PyExc_RuntimeError);
  nb::class_<native::session::Event>(module_, "SessionEvent")
      .def_prop_ro("kind",
                   [](const native::session::Event& event) {
                     return static_cast<int>(event.kind);
                   })
      .def_prop_ro("observation",
                   [](const native::session::Event& event) {
                     return static_cast<int>(event.observation);
                   })
      .def_ro("z", &native::session::Event::z)
      .def_ro("plane_index", &native::session::Event::plane_index)
      .def_ro("has_stats", &native::session::Event::has_stats)
      .def_ro("ticket", &native::session::Event::ticket)
      .def_ro("generation", &native::session::Event::generation)
      .def_prop_ro("outcome",
                   [](const native::session::Event& event) {
                     return static_cast<int>(event.outcome);
                   })
      .def_prop_ro("reason",
                   [](const native::session::Event& event) {
                     return static_cast<int>(event.failure.reason);
                   })
      .def_prop_ro("category",
                   [](const native::session::Event& event) {
                     return static_cast<int>(event.failure.category);
                   })
      .def_prop_ro(
          "message",
          [](const native::session::Event& event) { return event.failure.message; })
      .def_prop_ro("floor_proposal",
                   [](const native::session::Event& event) {
                     return event.failure.floor_proposal;
                   })
      .def_prop_ro(
          "cost",
          [](const native::session::Event& event) { return event.failure.cost; })
      .def_prop_ro(
          "replay_dz",
          [](const native::session::Event& event) { return event.failure.replay_dz; })
      .def_prop_ro(
          "replay_end",
          [](const native::session::Event& event) { return event.failure.replay_end; })
      .def_prop_ro(
          "replay_norm",
          [](const native::session::Event& event) { return event.failure.replay_norm; })
      .def_prop_ro(
          "cursor",
          [](const native::session::Event& event) { return event.failure.cursor; })
      .def_prop_ro("schedule_exhausted",
                   [](const native::session::Event& event) {
                     return event.failure.schedule_exhausted;
                   })
      .def_prop_ro(
          "target",
          [](const native::session::Event& event) { return event.failure.target; })
      // A boundary monitor's warning, or the stop a failure carries (SEP 47 6.4).
      .def_prop_ro("notice", [](const native::session::Event& event) {
        const auto& notice = event.kind == native::session::EventKind::warning
                                 ? event.notice
                                 : event.failure.notice;
        nb::dict out;
        out["kind"] = static_cast<int>(notice.kind);
        out["launch"] = notice.launch;
        out["z"] = notice.z;
        out["value"] = notice.value;
        out["cap"] = notice.cap;
        out["step"] = notice.step;
        out["edge"] = native::bindings::transverse_notice_to(notice.edge);
        return out;
      });
  nb::class_<BoundStatisticsPlan>(module_, "StatisticsPlan")
      .def(nb::init<const nb::dict>(), "values"_a)
      .def_prop_ro("packed_size", &BoundStatisticsPlan::packed_size);
  nb::class_<BatchRecord>(module_, "SessionBatch")
      .def_ro("sequence", &BatchRecord::sequence)
      .def_ro("event", &BatchRecord::event)
      .def_ro("attempts", &BatchRecord::attempts,
              nb::for_getter(
                  nb::sig("def attempts(self) -> numpy.typing.NDArray[numpy.float64]")))
      .def_ro("row_z", &BatchRecord::row_z,
              nb::for_getter(
                  nb::sig("def row_z(self) -> numpy.typing.NDArray[numpy.float64]")))
      .def_ro("row_step", &BatchRecord::row_step,
              nb::for_getter(
                  nb::sig("def row_step(self) -> numpy.typing.NDArray[numpy.int64]")))
      .def_ro("row_values", &BatchRecord::row_values,
              nb::for_getter(nb::sig(
                  "def row_values(self) -> numpy.typing.NDArray[numpy.float64]")))
      .def_ro("plane_stats", &BatchRecord::plane_stats,
              nb::for_getter(nb::sig(
                  "def plane_stats(self) -> numpy.typing.NDArray[numpy.float64]")));
  nb::class_<native::session::Status>(module_, "SessionStatus")
      .def_ro("z", &native::session::Status::z)
      .def_ro("plane_index", &native::session::Status::plane_index)
      .def_ro("attempts", &native::session::Status::attempts)
      .def_ro("rejected", &native::session::Status::rejected)
      .def_prop_ro("state",
                   [](const native::session::Status& status) {
                     return static_cast<int>(status.state);
                   })
      .def_prop_ro("outcome",
                   [](const native::session::Status& status) {
                     return static_cast<int>(status.outcome);
                   })
      .def_ro("generation", &native::session::Status::generation)
      .def_ro("march_calls", &native::session::Status::march_calls)
      .def_ro("rows_taken", &native::session::Status::rows_taken)
      .def_ro("batches", &native::session::Status::batches)
      .def_ro("exports", &native::session::Status::exports)
      .def_ro("export_bytes", &native::session::Status::export_bytes)
      .def_ro("buffer_bytes", &native::session::Status::buffer_bytes);
  nb::class_<BoundSession>(module_, "Session")
      .def("advance", &BoundSession::advance, "work_limit"_a, "deadline"_a)
      .def("acknowledge", &BoundSession::acknowledge, "sequence"_a,
           nb::call_guard<nb::gil_scoped_release>())
      .def("request_stop", &BoundSession::request_stop,
           nb::call_guard<nb::gil_scoped_release>())
      .def("request_snapshot", &BoundSession::request_snapshot)
      .def("status", &BoundSession::status)
      .def("march_state", &BoundSession::march_state, "since"_a,
           nb::sig("def march_state(self, since: int) -> dict[str, typing.Any] | None"))
      .def("fail_outside", &BoundSession::fail_outside)
      // solprop SEP 52 section 5.3: a ticket is `(sequence, generation)`; a recipe is
      // `{"power", "optics", "weight"}`.
      .def("export", &BoundSession::export_field, "ticket"_a, "recipe"_a, "out"_a,
           nb::sig("def export(self, ticket: tuple[int, int], recipe: "
                   "collections.abc.Mapping[str, object], out: "
                   "numpy.typing.NDArray[numpy.complex128]) -> None"))
      .def("snapshot", &BoundSession::snapshot, "ticket"_a,
           nb::sig("def snapshot(self, ticket: tuple[int, int]) -> tuple[dict[str, "
                   "typing.Any], numpy.typing.NDArray[numpy.complex128], dict[str, "
                   "numpy.typing.NDArray[numpy.complex128]]] | None"))
      .def("noise_phase", &BoundSession::noise_phase, "ticket"_a, "z"_a)
      .def("records", &BoundSession::records, "names"_a.none() = nb::none(),
           nb::sig("def records(self, names: collections.abc.Sequence[str] | None = "
                   "None) -> dict[str, typing.Any]"))
      .def("continue_monitor", &BoundSession::continue_monitor, "continuation"_a,
           nb::sig("def continue_monitor(self, continuation: "
                   "solprop.backends.records.MonitorContinuation) -> None"))
      .def("close", &BoundSession::close, nb::call_guard<nb::gil_scoped_release>());
  // solprop SEP 52 section 5.3: the new seam -- one workspace a problem, one run record
  // a start, the probe, and the tables every record is read by.
  nb::class_<BoundProbe>(module_, "Probe")
      .def("trial", &BoundProbe::trial, "z"_a, "dz"_a)
      .def("field", &BoundProbe::field)
      .def_prop_ro("trials", &BoundProbe::trials)
      .def_prop_ro("evaluations", &BoundProbe::evaluations)
      .def_prop_ro("low_storage", &BoundProbe::low_storage)
      .def_prop_ro("z", &BoundProbe::z)
      .def_prop_ro("closed", &BoundProbe::closed)
      .def("close", &BoundProbe::close, nb::call_guard<nb::gil_scoped_release>());
  nb::class_<BoundWorkspace>(module_, "Workspace")
      .def(nb::init<const native::Problem&>(), "problem"_a)
      .def("start", &BoundWorkspace::start, "run"_a,
           "host_source"_a.none() = nb::none(), nb::keep_alive<0, 1>(),
           nb::sig("def start(self, run: collections.abc.Mapping[str, object], "
                   "host_source: object | None = None) -> Session"))
      .def("probe", &BoundWorkspace::probe, "start"_a,
           "host_source"_a.none() = nb::none(), nb::keep_alive<0, 1>(),
           nb::sig("def probe(self, start: collections.abc.Mapping[str, object], "
                   "host_source: object | None = None) -> Probe"))
      .def("diagnostics", &BoundWorkspace::diagnostics)
      .def("counters", &BoundWorkspace::counters)
      .def_prop_ro("closed", &BoundWorkspace::closed)
      .def("close", &BoundWorkspace::close_from_python);
  module_.def("configure_runtime", &configure_runtime, "record"_a,
              nb::sig("def configure_runtime(record: collections.abc.Mapping[str, "
                      "object]) -> dict[str, str]"));
  module_.def("schemas", &schemas);
  auto testing =
      module_.def_submodule("_testing", "Private entry points for solprop's own tests");
  native::bindings::bind_helpers(module_, testing);
  native::bindings::bind_transverse(testing);
  bind_testing(testing);
  bind_engines(testing);
  if (Py_AtExit(&shutdown_after_interpreter) != 0)
    throw std::runtime_error(
        "cannot register the Kokkos runtime's shutdown at interpreter exit");
}
