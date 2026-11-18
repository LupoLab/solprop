// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The run records' conversions of run_records.hpp, host code a run's setup calls:
// `to_native` for the Runge-Kutta policy, the schedule and the two resumptions (each
// resumption's numbers read by name, an unknown name refused), `fault_from` (a test's
// injected failure by name) and the stage tables' `validate`.

#include "solprop_native/workspace/run_records.hpp"

#include "solprop_native/runtime/refuse.hpp"

#include <algorithm>
#include <array>
#include <csignal>
#include <cstddef>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace solprop_native {

namespace {

/** Set the optional `name` names in `numbers`; refuse a name `known` lacks. */
template <class Slots>
void numbers_into(const std::map<std::string, double>& numbers, const Slots& known,
                  const char* record) {
  for (const auto& entry : numbers) {
    // Named variables, not a structured binding: Clang 18 does not capture one in a
    // lambda of a unit built with OpenMP.
    const std::string& name = entry.first;
    const double value = entry.second;
    const auto found = std::ranges::find_if(
        known, [&name](const auto& slot) { return name == slot.first; });
    if (found == known.end())
      refuse(record, " holds an unknown number '", name, "'");
    *found->second = value;
  }
}

} // namespace

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
  // The equivalent norm's power and its reciprocal, as solprop's controller forms them
  // from the estimator's order (`solprop.stepper.controller`).
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
  // As `solprop.backends.records.edge_resumption` reads it: a record that is none
  // carries nothing else.
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

} // namespace solprop_native
