// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The native marches' records from and to solprop's dictionaries (march.hpp): the
// Adams march's policy, state, bounds and result (solprop SEP 39 phase 2), the
// Runge-Kutta march's state and result, and the boundary watch's policy, state,
// configuration, record and judgement (solprop SEP 45 phase 1, SEP 47 section 6.4).
// The policies, the march states and the watch's state and configuration go through
// their field tables (`records.hpp`) and the march's validation; the bounds and the
// results, and the watch's record and judgement, which have no field table, are
// converted key by key.

#include "march.hpp"

#include "arrays.hpp"
#include "records.hpp"
#include "solprop_native/march/adams_march.hpp"
#include "solprop_native/march/rk_march.hpp"
#include "solprop_native/session/watch.hpp"
#include "solprop_native/workspace/run_records.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace solprop_native::bindings {

// ------------------------------------------------------------- the native Adams march
// solprop SEP 39 phase 2: the march's policy, state, bounds and result as solprop's
// adapter passes and reads them (the decision-trace tests' scripted march is
// testing.cpp's).

namespace march = native::march;

march::Policy march_policy_from(const nb::dict values) {
  auto policy = native::records::read<march::Policy>(values, "policy");
  march::validate(policy);
  return policy;
}

// solprop SEP 45 phase 1: the boundary watch's policy and state, from and to solprop's
// `BoundaryMonitor.watch_policy` and `.watch_state`.
native::watch::Policy watch_policy_from(const nb::dict values) {
  auto policy = native::records::read<native::watch::Policy>(values, "policy");
  native::watch::validate(policy);
  return policy;
}

native::watch::State watch_state_from(const nb::dict values) {
  return native::records::read<native::watch::State>(values, "watch");
}

nb::dict watch_state_to(const native::watch::State& state) {
  return native::records::write_record(state);
}

// solprop SEP 47 section 6.4: the owning boundary monitor's configuration, the record
// a resume continues and the record it keeps, from and to solprop's dictionaries.
native::watch::Config watch_config_from(const nb::handle values) {
  auto config = native::records::read<native::watch::Config>(values, "config");
  native::watch::validate(config.policy);
  return config;
}

nb::dict watch_record_to(const native::watch::Record& record) {
  nb::dict out;
  out["launch_energy"] = record.launch_energy;
  out["resumed_at"] = record.resumed_at;
  out["launch_guard_low"] = record.launch_occupancy[0];
  out["launch_guard_high"] = record.launch_occupancy[1];
  out["launch_collar"] = record.launch_occupancy[2];
  out["guard_low_max"] = record.guard_max[0];
  out["guard_high_max"] = record.guard_max[1];
  out["guard_low_at"] = record.guard_at[0];
  out["guard_high_at"] = record.guard_at[1];
  out["collar_max"] = record.collar_max;
  out["removed_fraction"] = record.removed_fraction;
  out["removed_signed"] = record.removed_signed;
  out["floor_removed_fraction"] = record.floor_removed_fraction;
  out["quiet_checks"] = record.quiet_checks;
  out["collar_quiet"] = record.collar_quiet;
  nb::list unwatched;
  for (const auto& stretch : record.unwatched)
    unwatched.append(nb::make_tuple(stretch[0], stretch[1]));
  out["unwatched"] = unwatched;
  out["steps"] = record.steps;
  out["skipped"] = record.skipped;
  out["noise_floor"] = record.noise_floor;
  out["raw_guard_max"] = record.raw_guard_max;
  out["raw_removed_fraction"] = record.raw_removed_fraction;
  out["collar_left_out_checks"] = record.collar_left_out_checks;
  static constexpr std::array<const char*, native::watch::plane_columns> names{
      "z", "collar_occupancy", "guard_low", "guard_high", "removed_fraction"};
  nb::dict planes;
  for (std::size_t column = 0; column < native::watch::plane_columns; ++column)
    planes[names[column]] = record.planes[column];
  out["planes"] = planes;
  return out;
}

nb::dict watch_judgement_to(const native::watch::Judgement& judgement) {
  nb::dict out;
  out["verdict"] = static_cast<int>(judgement.verdict);
  out["worst"] = judgement.worst;
  out["z"] = judgement.z;
  out["launch"] = judgement.launch;
  return out;
}

nb::dict rk_result_to(const native::rk::Result& out, const native::rk::State& state) {
  const auto& result = out.march;
  nb::dict values;
  values["reason"] = static_cast<int>(result.reason);
  const std::size_t n_rows = result.rows.size() / march::row_columns;
  std::vector<double> rows = result.rows;
  values["rows"] = n_rows == 0 ? nb::cast(nb::list())
                               : nb::cast(real_output(std::move(rows), n_rows));
  values["floor_proposal"] = result.floor_proposal;
  values["floor_site"] = static_cast<int>(out.floor_site);
  values["pending_equivalent"] = out.pending_equivalent;
  values["replay_dz"] = result.replay_dz;
  values["replay_end"] = result.replay_end;
  values["replay_norm"] = result.replay_norm;
  values["stage"] = static_cast<int>(result.stage);
  values["category"] = static_cast<int>(result.category);
  values["message"] = result.message;
  const std::size_t n_readings = result.reading_rows.size();
  std::vector<double> readings = result.readings;
  values["readings"] = n_readings == 0
                           ? nb::cast(nb::list())
                           : nb::cast(real_output(std::move(readings), n_readings));
  values["reading_rows"] = result.reading_rows;
  values["verdict"] = static_cast<int>(result.verdict);
  values["state"] = native::records::write_record(state);
  return values;
}

march::State march_state_from(const nb::dict values) {
  auto state = native::records::read<march::State>(values, "state");
  march::validate(state);
  return state;
}

march::Bounds march_bounds_from(const nb::dict values) {
  march::Bounds bounds;
  try {
    bounds.target = nb::cast<double>(values["target"]);
    bounds.quiet = nb::cast<bool>(values["quiet"]);
    bounds.host_step = nb::cast<std::uint64_t>(values["host_step"]);
    bounds.monitor = nb::cast<bool>(values["monitor"]);
    bounds.first_check = nb::cast<bool>(values["first_check"]);
    bounds.check_interval = nb::cast<double>(values["check_interval"]);
    bounds.last_check = nb::cast<double>(values["last_check"]);
    bounds.has_read_at = !values["read_at"].is_none();
    bounds.read_at = bounds.has_read_at ? nb::cast<double>(values["read_at"]) : 0.0;
    // The Adams march reads the spacing only with a read (`read_capped`); the
    // Runge-Kutta march takes it as the edge's ceiling, infinite without an edge.
    bounds.read_spacing = nb::cast<double>(values["read_spacing"]);
    bounds.step_cap = nb::cast<double>(values["step_cap"]);
    bounds.max_attempts = nb::cast<std::size_t>(values["max_attempts"]);
    bounds.max_seconds = nb::cast<double>(values["max_seconds"]);
    bounds.replay = nb::cast<bool>(values["replay"]);
    bounds.slack = nb::cast<double>(values["slack"]);
    // The march's own boundary checks (solprop SEP 45), when the host asks for them.
    if (values.contains("watch") && !values["watch"].is_none()) {
      bounds.watch = true;
      bounds.watch_policy = watch_policy_from(nb::cast<nb::dict>(values["watch"]));
      bounds.watch_capacity = nb::cast<std::size_t>(values["watch_capacity"]);
    }
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("the Adams march's bounds have the wrong types");
  } catch (const nb::python_error&) {
    throw std::invalid_argument("the Adams march's bounds lack a field");
  }
  if (bounds.max_attempts < 1)
    throw std::invalid_argument("a march call needs at least one attempt");
  return bounds;
}

nb::dict march_result_to(const march::Result& result, const march::State& state) {
  nb::dict out;
  out["reason"] = static_cast<int>(result.reason);
  const std::size_t n_rows = result.rows.size() / march::row_columns;
  std::vector<double> rows = result.rows;
  out["rows"] = n_rows == 0 ? nb::cast(nb::list())
                            : nb::cast(real_output(std::move(rows), n_rows));
  out["orders"] = result.orders;
  out["cost"] = result.cost;
  out["floor_proposal"] = result.floor_proposal;
  out["stage"] = static_cast<int>(result.stage);
  out["category"] = static_cast<int>(result.category);
  out["message"] = result.message;
  out["replay_dz"] = result.replay_dz;
  out["replay_end"] = result.replay_end;
  out["replay_norm"] = result.replay_norm;
  out["stepped"] = result.stepped;
  out["step_z"] = result.step_z;
  out["step_dz"] = result.step_dz;
  out["step_order"] = result.step_order;
  out["step_nodes"] = std::vector<double>(
      result.step_nodes.begin(),
      result.step_nodes.begin() + static_cast<std::ptrdiff_t>(result.step_order));
  out["step_carrier"] = result.step_carrier;
  const std::size_t n_readings = result.reading_rows.size();
  std::vector<double> readings = result.readings;
  out["readings"] = n_readings == 0
                        ? nb::cast(nb::list())
                        : nb::cast(real_output(std::move(readings), n_readings));
  out["reading_rows"] = result.reading_rows;
  out["verdict"] = static_cast<int>(result.verdict);
  out["state"] = native::records::write_record(state);
  return out;
}

} // namespace solprop_native::bindings
