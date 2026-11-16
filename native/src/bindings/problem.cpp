// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The `Problem` class and its field groups (`bind_problem`): each group a nested class
// bound from its field table (`bind_group`), each part of the set-once `Setup` a
// property read through its table (`bind_setup`), and the CPU execution record and FFT
// planning (solprop SEP 46) read from and written to the dictionaries solprop_native's
// `_execution` and `_wisdom` write. `fft_planning_dict` and `cpu_execution_dict` report
// them in a workspace's diagnostics.

#include "solprop_native/problem/problem.hpp"

#include "arrays.hpp"
#include "bind.hpp"
#include "records.hpp"
#include "solprop_native/problem/run_records.hpp"
#include "solprop_native/runtime/cpu_execution.hpp"
#include "solprop_native/runtime/fields.hpp"
#include "solprop_native/runtime/refuse.hpp"
#include "solprop_native/transforms/fftw_wisdom.hpp"

#include <Kokkos_Core.hpp>
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
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace solprop_native::bindings {

using namespace nb::literals;

namespace {

// solprop SEP 46: a workspace's CPU execution record, as solprop_native's
// `_execution.ExecutionRecord.native()` writes it: each operation's threshold and full
// size, then the response bounds, widths and rules, every key required.
constexpr std::array<const char*, native::cpu_operation_count> cpu_operation_names{
    "stage", "response", "dense", "fft", "raman", "copy", "flow", "modal", "adams"};
constexpr std::array<const char*, 8> cpu_execution_scalars{
    "response_full_samples",    "response_full_columns",
    "medium_threads",           "compact_threads",
    "compact_max_fine_samples", "stable_kernel_team",
    "fft_thread_elements",      "fft_long_row"};

native::CpuExecution cpu_execution_from(const nb::dict& values) {
  native::CpuExecution result;
  const std::size_t expected =
      2 * native::cpu_operation_count + cpu_execution_scalars.size();
  if (values.size() != expected)
    native::refuse("a CPU execution record has ", expected, " fields, got ",
                   values.size());
  const auto field = [&values](const std::string& name) {
    if (!values.contains(name.c_str()))
      throw std::invalid_argument("a CPU execution record misses a field: " + name);
    return values[name.c_str()];
  };
  try {
    for (std::size_t index = 0; index < native::cpu_operation_count; ++index) {
      const std::string name = cpu_operation_names[index];
      result.grain[index] = {.threshold =
                                 nb::cast<std::size_t>(field(name + "_threshold")),
                             .full = nb::cast<std::size_t>(field(name + "_full"))};
    }
    result.response_full_samples =
        nb::cast<std::size_t>(field("response_full_samples"));
    result.response_full_columns =
        nb::cast<std::size_t>(field("response_full_columns"));
    result.medium_threads = nb::cast<int>(field("medium_threads"));
    result.compact_threads = nb::cast<int>(field("compact_threads"));
    result.compact_max_fine_samples =
        nb::cast<std::size_t>(field("compact_max_fine_samples"));
    result.stable_kernel_team = nb::cast<bool>(field("stable_kernel_team"));
    result.fft_thread_elements = nb::cast<std::size_t>(field("fft_thread_elements"));
    result.fft_long_row = nb::cast<std::size_t>(field("fft_long_row"));
  } catch (const nb::cast_error&) {
    throw std::invalid_argument(
        "a CPU execution record holds nonnegative integers and one bool");
  }
  native::require_cpu_execution(result);
  return result;
}

// solprop SEP 46 phase 3: a workspace's FFT planning, as solprop_native's backend
// writes it: its mode, each group's effort, the profile's wisdom by plan key, a
// capture's effort and time limit, and the plans runs measured before with whether to
// keep new ones (ADR-076); absent, the plan policy (`legacy`).
[[maybe_unused]] native::FftPlanning fft_planning_from(const nb::dict& values) {
  native::FftPlanning result;
  try {
    const auto mode = nb::cast<std::string>(values["mode"]);
    if (mode == "legacy")
      result.mode = native::FftPlanning::Mode::legacy;
    else if (mode == "profile")
      result.mode = native::FftPlanning::Mode::profile;
    else if (mode == "capture")
      result.mode = native::FftPlanning::Mode::capture;
    else
      throw std::invalid_argument("an FFT planning mode is legacy, profile or capture, "
                                  "not '" +
                                  mode + "'");
    if (values.contains("efforts")) {
      const auto efforts = nb::cast<nb::dict>(values["efforts"]);
      for (auto [name, effort] : efforts) {
        const auto group = nb::cast<std::string>(name);
        auto text = nb::cast<std::string>(effort);
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
        static_cast<void>(native::fftw_effort_flag(text));
#endif
        if (group == "rows")
          result.rows = std::move(text);
        else if (group == "threaded_rows")
          result.threaded_rows = std::move(text);
        else if (group == "planes")
          result.planes = std::move(text);
        else
          throw std::invalid_argument("an FFT planning group is rows, threaded_rows "
                                      "or planes, not '" +
                                      group + "'");
      }
    }
    if (values.contains("wisdom"))
      for (auto [key, text] : nb::cast<nb::dict>(values["wisdom"]))
        result.wisdom[nb::cast<std::string>(key)] = nb::cast<std::string>(text);
    if (values.contains("capture_effort")) {
      result.capture_effort = nb::cast<std::string>(values["capture_effort"]);
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
      static_cast<void>(native::fftw_effort_flag(result.capture_effort));
#endif
    }
    if (values.contains("time_limit"))
      result.time_limit = nb::cast<double>(values["time_limit"]);
    if (values.contains("stored"))
      for (auto [key, text] : nb::cast<nb::dict>(values["stored"]))
        result.stored[nb::cast<std::string>(key)] = nb::cast<std::string>(text);
    if (values.contains("keep"))
      result.keep = nb::cast<bool>(values["keep"]);
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("an FFT planning record has the wrong types");
  }
  return result;
}

// The planning a workspace is asked for, as `fft_planning_from` reads it back.
nb::dict fft_request_dict(const native::FftPlanning& planning) {
  static constexpr std::array<const char*, 3> modes{"legacy", "profile", "capture"};
  nb::dict efforts;
  efforts["rows"] = planning.rows;
  efforts["threaded_rows"] = planning.threaded_rows;
  efforts["planes"] = planning.planes;
  nb::dict result;
  result["mode"] = modes[static_cast<std::size_t>(planning.mode)];
  result["efforts"] = efforts;
  result["wisdom"] = planning.wisdom;
  result["capture_effort"] = planning.capture_effort;
  result["time_limit"] = planning.time_limit;
  result["stored"] = planning.stored;
  result["keep"] = planning.keep;
  return result;
}

/** The problem's set-once data, copied on write (`Problem::setup`). */
native::Setup& setup_of(native::Problem& problem) {
  auto copy = problem.setup ? std::make_shared<native::Setup>(*problem.setup)
                            : std::make_shared<native::Setup>();
  auto& held = *copy;
  problem.setup = std::move(copy);
  return held;
}

/** Bind one part of the problem's setup as a property that reads a record (a
    dataclass or a mapping) through its table, and writes it back as a dict. */
template <class Value>
void bind_setup(nb::class_<native::Problem>& problem, const char* name,
                Value native::Setup::* member, const char* python_type) {
  const std::string getter = native::concat("def ", name, "(self) -> ", python_type);
  const std::string setter =
      native::concat("def ", name, "(self, arg: ", python_type, ", /) -> None");
  problem.def_prop_rw(
      name,
      [member](const native::Problem& self) -> nb::object {
        if (!self.setup)
          return native::records::write_value(Value{});
        return native::records::write_value((*self.setup).*member);
      },
      [member, name](native::Problem& self, const nb::handle value) {
        auto held =
            native::records::read<Value>(value, native::concat("problem.", name));
        setup_of(self).*member = std::move(held);
      },
      nb::for_getter(nb::sig(getter.c_str())), nb::for_setter(nb::sig(setter.c_str())),
      nb::for_setter(nb::arg("value").none()));
}

/** One group of the problem's fields, bound as a nested class from its field table
    (solprop SEP 52 section 5.3): scalars read and written as they are, arrays as exact
    arrays (`exact_vector`) of the table's rank, or of any rank where it gives none. A
    group whose layouts its counts give keeps each array's shape for
    `Problem::validate`, which holds the relations across groups. */
template <class Group>
void bind_group(const nb::handle scope, const char* name, const std::string& path) {
  nb::class_<Group> bound(scope, name);
  bound.def(nb::init<>());
  native::fields::each(native::fields::table_of<Group>(), [&](const auto& field) {
    using Field = std::remove_cvref_t<decltype(field)>;
    constexpr int kind = native::records::FieldKind<Field>::kind;
    if constexpr (kind == 2) {
      using Vector =
          std::remove_cvref_t<decltype(std::declval<Group&>().*field.member)>;
      // NOLINTNEXTLINE(readability-redundant-typename): nvcc's front end needs it
      using Element = typename Vector::value_type;
      constexpr std::size_t rank = std::tuple_size_v<decltype(field.extents)>;
      const auto member = field.member;
      const char* label = field.name;
      // The setter's stub names the one dtype it takes.
      const std::string setter =
          native::concat("def ", label, "(self, arg: numpy.typing.NDArray[",
                         ExactArray<Element>::scalar, "], /) -> None");
      bound.def_prop_rw(
          label, [member](const Group& self) { return array_of(self.*member); },
          [member, label, where = path + "." + label](Group& self,
                                                      const nb::handle value) {
            if constexpr (requires { self.shapes; }) {
              std::vector<std::size_t> shape;
              self.*member = exact_vector<Element>(value, where, rank, &shape);
              self.shapes[label] = std::move(shape);
            } else {
              static_cast<void>(label);
              self.*member = exact_vector<Element>(value, where, rank);
            }
          },
          nb::for_getter(nb::rv_policy::automatic),
          nb::for_setter(nb::sig(setter.c_str())));
    } else if constexpr (kind == 0) {
      bound.def_rw(field.name, field.member);
    }
  });
}

} // namespace

nb::dict fft_planning_dict(const native::FftPlanning& planning) {
  nb::dict result;
  nb::dict groups;
  for (const auto& [key, group] : planning.groups)
    groups[key.c_str()] = group;
  nb::dict captured;
  for (const auto& [key, text] : planning.captured)
    captured[key.c_str()] = text;
  nb::list missing;
  for (const auto& key : planning.missing)
    missing.append(key);
  nb::dict measured;
  for (const auto& [key, text] : planning.measured)
    measured[key.c_str()] = text;
  nb::list reused;
  for (const auto& key : planning.reused)
    reused.append(key);
  result["groups"] = groups;
  result["captured"] = captured;
  result["missing"] = missing;
  result["measured"] = measured;
  result["reused"] = reused;
  return result;
}

nb::dict cpu_execution_dict(const native::CpuExecution& execution) {
  nb::dict result;
  for (std::size_t index = 0; index < native::cpu_operation_count; ++index) {
    const std::string name = cpu_operation_names[index];
    result[(name + "_threshold").c_str()] = execution.grain[index].threshold;
    result[(name + "_full").c_str()] = execution.grain[index].full;
  }
  result["response_full_samples"] = execution.response_full_samples;
  result["response_full_columns"] = execution.response_full_columns;
  result["medium_threads"] = execution.medium_threads;
  result["compact_threads"] = execution.compact_threads;
  result["compact_max_fine_samples"] = execution.compact_max_fine_samples;
  result["stable_kernel_team"] = execution.stable_kernel_team;
  result["fft_thread_elements"] = execution.fft_thread_elements;
  result["fft_long_row"] = execution.fft_long_row;
  return result;
}

void bind_problem(nb::module_& module_) {
  using P = native::Problem;
  nb::enum_<native::FieldFamily>(module_, "FieldFamily")
      .value("field_real", native::FieldFamily::field_real)
      .value("field_analytic", native::FieldFamily::field_analytic)
      .value("envelope", native::FieldFamily::envelope);
  nb::class_<P> problem(module_, "Problem");
  bind_group<P::Dims>(problem, "Dims", "dims");
  bind_group<P::Grid>(problem, "Grid", "grid");
  bind_group<P::Spatial>(problem, "Spatial", "spatial");
  bind_group<P::Flux>(problem, "Flux", "flux");
  bind_group<P::Stepper>(problem, "Stepper", "stepper");
  bind_group<P::Linear>(problem, "Linear", "linear");
  bind_group<P::Absorber>(problem, "Absorber", "absorber");
  bind_group<P::Transform>(problem, "Transform", "transform");
  bind_group<P::Kerr>(problem, "Kerr", "kerr");
  bind_group<P::Plasma>(problem, "Plasma", "plasma");
  bind_group<P::Raman>(problem, "Raman", "raman");
  bind_group<P::General>(problem, "General", "general");
  bind_group<P::Flow>(problem, "Flow", "flow");
  bind_group<P::Dispersive>(problem, "Dispersive", "dispersive");
  // solprop SEP 52 section 5.3: the set-once data a workspace applies, each part a
  // record read through its table.
  bind_setup(problem, "stage_tables", &native::Setup::stage_tables,
             "collections.abc.Mapping[str, object] | None");
  bind_setup(problem, "z_slots", &native::Setup::z_slots,
             "collections.abc.Mapping[str, object] | None");
  bind_setup(problem, "z_tables", &native::Setup::z_tables,
             "collections.abc.Sequence[collections.abc.Mapping[str, object]]");
  bind_setup(problem, "flow_guard", &native::Setup::flow_guard,
             "collections.abc.Mapping[str, object] | None");
  bind_setup(problem, "runge_kutta", &native::Setup::runge_kutta,
             "collections.abc.Mapping[str, object] | None");
  bind_setup(problem, "adams", &native::Setup::adams,
             "collections.abc.Mapping[str, object] | None");
  bind_setup(problem, "low_storage", &native::Setup::low_storage,
             "solprop.backends.records.LowStorageSettings | None");
  bind_setup(problem, "occupancy", &native::Setup::occupancy,
             "collections.abc.Mapping[str, object] | None");
  bind_setup(problem, "crossing_workers", &native::Setup::crossing_workers, "int");
  problem.def(nb::init<>())
      .def_rw("dims", &P::dims)
      .def_rw("grid", &P::grid)
      .def_rw("spatial", &P::spatial)
      .def_rw("flux", &P::flux)
      .def_rw("stepper", &P::stepper)
      .def_rw("linear", &P::linear)
      .def_rw("absorber", &P::absorber)
      .def_rw("transform", &P::transform)
      .def_rw("kerr", &P::kerr)
      .def_rw("plasma", &P::plasma)
      .def_rw("raman", &P::raman)
      .def_rw("general", &P::general)
      .def_rw("flow", &P::flow)
      .def_rw("dispersive", &P::dispersive)
      // solprop SEP 46: the CPU execution record and the FFT planning cross as the
      // records solprop_native's `_execution` and `_wisdom` write.
      .def_prop_rw(
          "cpu_execution",
          [](const P& self) { return cpu_execution_dict(self.cpu_execution); },
          [](P& self, const nb::handle values) {
            if (!nb::isinstance<nb::dict>(values))
              throw std::invalid_argument(
                  "a native problem's cpu_execution must be a dict");
            self.cpu_execution = cpu_execution_from(nb::borrow<nb::dict>(values));
          },
          nb::for_getter(nb::sig("def cpu_execution(self) -> dict[str, int | bool]")),
          nb::for_setter(nb::sig(
              "def cpu_execution(self, arg: dict[str, int | bool], /) -> None")),
          nb::for_setter(nb::arg("values").none()))
      .def_prop_rw(
          "fft_planning",
          [](const P& self) { return fft_request_dict(self.fft_planning); },
          [](P& self, const nb::handle values) {
            if (!nb::isinstance<nb::dict>(values))
              throw std::invalid_argument(
                  "a native problem's fft_planning must be a dict");
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
            self.fft_planning = fft_planning_from(nb::borrow<nb::dict>(values));
#else
            // A device build plans its transforms with the vendor's FFT alone.
            static_cast<void>(self);
            static_cast<void>(values);
#endif
          },
          nb::for_getter(nb::sig("def fft_planning(self) -> dict[str, object]")),
          nb::for_setter(
              nb::sig("def fft_planning(self, arg: dict[str, object], /) -> None")),
          nb::for_setter(nb::arg("values").none()))
      .def("validate", &P::validate);
}

} // namespace solprop_native::bindings
