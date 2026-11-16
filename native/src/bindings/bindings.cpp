// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "bindings_helpers.hpp"
#include "bindings_transverse.hpp"
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
#include "solprop_native/problem/run_records.hpp"
#include "solprop_native/problem/workspace.hpp"
#include "solprop_native/runtime/cpu_execution.hpp"
#include "solprop_native/runtime/crossing.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/fields.hpp"
#include "solprop_native/runtime/host_source.hpp"
#include "solprop_native/runtime/refuse.hpp"
#include "solprop_native/runtime/vmath.hpp"
#include "solprop_native/session/readings.hpp"
#include "solprop_native/session/session.hpp"
#include "solprop_native/session/watch.hpp"
#include "solprop_native/transforms/fftw_wisdom.hpp"
#include "solprop_native/transforms/reference_timing.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>

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

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <array>
#include <complex>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace solprop_native {
// The digest of what this extension is compiled from (cmake/SourceDigest.cmake).
const char* source_digest() noexcept;
} // namespace solprop_native

namespace {

namespace native = solprop_native;
using ComplexInput =
    nb::ndarray<const std::complex<double>, nb::c_contig, nb::device::cpu>;
using RealInput = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using BoolInput = nb::ndarray<const bool, nb::c_contig, nb::device::cpu>;
using ByteInput = nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu>;
using IndexInput = nb::ndarray<const std::uint64_t, nb::c_contig, nb::device::cpu>;
using ComplexOutput =
    nb::ndarray<nb::numpy, std::complex<double>, nb::c_contig, nb::device::cpu>;
using RealOutput = nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu>;
// A read-only NumPy view of the engine's memory (solprop SEP 49's host source field).
using HostField =
    nb::ndarray<nb::numpy, const std::complex<double>, nb::c_contig, nb::device::cpu>;
using ComplexStorage = std::vector<std::complex<double>>;
using RealStorage = std::vector<double>;

// A native object the runtime closes before Kokkos is finalised: a state or a
// workspace, each of which closes the session that leases it first.
class Owned {
public:
  Owned() = default;
  Owned(const Owned&) = delete;
  Owned& operator=(const Owned&) = delete;
  Owned(Owned&&) = delete;
  Owned& operator=(Owned&&) = delete;
  virtual ~Owned() = default;
  virtual void close() noexcept = 0;
};

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

template <class Array> Array exact_array(const nb::handle value) {
  return nb::cast<Array>(value, false);
}

struct RuntimeState {
  std::mutex mutex;
  bool owns_runtime = false;
  int requested_threads = 0;
  // The BLAS thread policy the package applied at start-up, if any.
  std::string blas_policy;
  std::vector<Owned*> states;
};

RuntimeState runtime_state;

void register_owned(Owned* owned) {
  const std::scoped_lock lock(runtime_state.mutex);
  runtime_state.states.push_back(owned);
}

void forget_owned(Owned* owned) {
  const std::scoped_lock lock(runtime_state.mutex);
  const auto found = std::ranges::find(runtime_state.states, owned);
  if (found != runtime_state.states.end())
    runtime_state.states.erase(found);
}

template <class T>
std::size_t element_count(const nb::ndarray<T, nb::c_contig, nb::device::cpu>& values) {
  std::size_t count = 1;
  for (std::size_t axis = 0; axis < values.ndim(); ++axis) {
    count *= values.shape(axis);
  }
  return count;
}

// NumPy's complex128 and `native::Complex` are both two doubles, real part first.
static_assert(sizeof(native::Complex) == sizeof(std::complex<double>));
static_assert(__STDCPP_DEFAULT_NEW_ALIGNMENT__ >= alignof(native::Complex));

/** A complex array's own memory as native values, read in place. */
std::span<const native::Complex> complex_span(const ComplexInput values) {
  return {reinterpret_cast<const native::Complex*>(values.data()),
          element_count(values)};
}

std::vector<native::Complex> complex_vector(const ComplexInput values) {
  const auto count = element_count(values);
  std::vector<native::Complex> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    result[index] =
        native::Complex(values.data()[index].real(), values.data()[index].imag());
  }
  return result;
}

std::vector<double> real_vector(const RealInput values) {
  const auto count = element_count(values);
  return {values.data(), values.data() + count};
}

std::vector<unsigned char> bool_vector(const BoolInput values) {
  const auto count = element_count(values);
  std::vector<unsigned char> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    result[index] = values.data()[index] ? 1U : 0U;
  }
  return result;
}

std::vector<std::size_t> index_vector(const IndexInput values) {
  const auto count = element_count(values);
  std::vector<std::size_t> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    result[index] = static_cast<std::size_t>(values.data()[index]);
  }
  return result;
}

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

// ------------------------------------------------------------- the problem's fields
// A vector field of a bound struct crosses as an exact NumPy array: float64,
// complex128, uint64 or bool, C-contiguous on the CPU and of any rank, copied in, and
// out as a one-dimensional copy. Anything else is refused, naming the field.

template <class Element> struct ExactArray;
template <> struct ExactArray<double> {
  using Input = RealInput;
  using Output = nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig>;
  using Stored = double;
  static constexpr const char* dtype = "float64";
  static constexpr const char* scalar = "numpy.float64";
};
template <> struct ExactArray<native::Complex> {
  using Input = ComplexInput;
  using Output =
      nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>, nb::c_contig>;
  using Stored = std::complex<double>;
  static constexpr const char* dtype = "complex128";
  static constexpr const char* scalar = "numpy.complex128";
};
template <> struct ExactArray<std::size_t> {
  using Input = IndexInput;
  using Output = nb::ndarray<nb::numpy, std::uint64_t, nb::ndim<1>, nb::c_contig>;
  using Stored = std::uint64_t;
  static constexpr const char* dtype = "uint64";
  static constexpr const char* scalar = "numpy.uint64";
};
template <> struct ExactArray<unsigned char> {
  using Input = BoolInput;
  using Output = nb::ndarray<nb::numpy, bool, nb::ndim<1>, nb::c_contig>;
  using Stored = bool;
  static constexpr const char* dtype = "bool";
  static constexpr const char* scalar = "numpy.bool_";
};

template <class Element>
std::vector<Element> exact_vector(const nb::handle value, const std::string& field,
                                  const std::size_t rank,
                                  std::vector<std::size_t>* shape = nullptr) {
  typename ExactArray<Element>::Input array;
  if (!nb::try_cast(value, array, false) || (rank != 0 && array.ndim() != rank))
    native::refuse(field, " must be a C-contiguous ", ExactArray<Element>::dtype,
                   " array", rank == 0 ? "" : native::concat(" of rank ", rank));
  if (shape != nullptr) {
    shape->resize(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis)
      (*shape)[axis] = array.shape(axis);
  }
  if constexpr (std::is_same_v<Element, double>)
    return real_vector(array);
  else if constexpr (std::is_same_v<Element, native::Complex>)
    return complex_vector(array);
  else if constexpr (std::is_same_v<Element, std::size_t>)
    return index_vector(array);
  else
    return bool_vector(array);
}

template <class Element>
// NOLINTNEXTLINE(readability-redundant-typename): nvcc's front end needs it
typename ExactArray<Element>::Output array_of(const std::vector<Element>& values) {
  // NOLINTNEXTLINE(readability-redundant-typename): nvcc's front end needs it
  using Stored = typename ExactArray<Element>::Stored;
  auto storage = std::make_unique<Stored[]>(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    if constexpr (std::is_same_v<Element, native::Complex>)
      storage[index] = {values[index].real(), values[index].imag()};
    else
      storage[index] = static_cast<Stored>(values[index]);
  }
  nb::capsule owner(storage.get(), [](void* held) noexcept {
    const std::unique_ptr<Stored[]> owned(static_cast<Stored*>(held));
  });
  const std::array<std::size_t, 1> shape{values.size()};
  return {storage.release(), 1, shape.data(), owner};
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

ComplexOutput complex_output(const std::vector<native::Complex>& values,
                             std::vector<std::size_t> shape) {
  auto storage = std::make_unique<ComplexStorage>(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    (*storage)[index] =
        std::complex<double>(values[index].real(), values[index].imag());
  }
  auto* pointer = storage.get();
  nb::capsule owner(pointer, [](void* value) noexcept {
    const std::unique_ptr<ComplexStorage> held(static_cast<ComplexStorage*>(value));
  });
  static_cast<void>(storage.release()); // the capsule owns it now
  return {pointer->data(), shape.size(), shape.data(), owner};
}

/** A complex array owning `storage`. */
ComplexOutput owned_complex(std::unique_ptr<ComplexStorage> storage,
                            std::vector<std::size_t> shape) {
  auto* pointer = storage.get();
  nb::capsule owner(pointer, [](void* value) noexcept {
    const std::unique_ptr<ComplexStorage> held(static_cast<ComplexStorage*>(value));
  });
  static_cast<void>(storage.release()); // the capsule owns it now
  return {pointer->data(), shape.size(), shape.data(), owner};
}

/** A complex array owning `storage`, uninitialised doubles a native copy filled. */
ComplexOutput complex_buffer_output(native::crossing::HostBuffer storage,
                                    std::vector<std::size_t> shape) {
  auto* pointer = storage.release();
  nb::capsule owner(pointer, [](void* value) noexcept {
    const native::crossing::HostBuffer held(static_cast<double*>(value));
  });
  return {reinterpret_cast<std::complex<double>*>(pointer), shape.size(), shape.data(),
          owner};
}

/** `values` as a one-dimensional array that owns them. */
RealOutput vector_output(std::vector<double> values) {
  auto storage = std::make_unique<RealStorage>(std::move(values));
  auto* pointer = storage.get();
  nb::capsule owner(pointer, [](void* value) noexcept {
    const std::unique_ptr<RealStorage> held(static_cast<RealStorage*>(value));
  });
  static_cast<void>(storage.release()); // the capsule owns it now
  const std::array<std::size_t, 1> shape{pointer->size()};
  return {pointer->data(), shape.size(), shape.data(), owner};
}

RealOutput real_output(std::vector<double> values, const std::size_t rows) {
  if (rows == 0 || values.size() % rows != 0)
    throw std::invalid_argument("native output has an invalid packed shape");
  auto storage = std::make_unique<RealStorage>(std::move(values));
  auto* pointer = storage.get();
  nb::capsule owner(pointer, [](void* value) noexcept {
    const std::unique_ptr<RealStorage> held(static_cast<RealStorage*>(value));
  });
  static_cast<void>(storage.release()); // the capsule owns it now
  const std::array<std::size_t, 2> shape{rows, pointer->size() / rows};
  return {pointer->data(), shape.size(), shape.data(), owner};
}

void initialize_kokkos(const int threads, const int kernels, const int fft,
                       const int raman_fft, const int raman_tail_fft,
                       const int band_tiles) {
  if (threads < 1)
    throw std::invalid_argument("thread count must be positive");
  const std::scoped_lock lock(runtime_state.mutex);
  if (!Kokkos::is_initialized()) {
    Kokkos::InitializationSettings settings;
    settings.set_num_threads(threads);
    Kokkos::initialize(settings);
    runtime_state.owns_runtime = true;
    runtime_state.requested_threads = threads;
  } else if (runtime_state.requested_threads != 0 &&
             runtime_state.requested_threads != threads) {
    throw std::runtime_error(
        "Kokkos was already initialized with another thread count");
  }
  native::configure_thread_policy({.kernels = kernels,
                                   .fft = fft,
                                   .raman_fft = raman_fft,
                                   .raman_tail_fft = raman_tail_fft,
                                   .band_tiles = band_tiles});
}

// ------------------------------------------------------------- the native Adams march
// solprop SEP 39 phase 2: the march's policy, state, bounds and result as solprop's
// adapter passes and reads them, its resident engine over the general state, and a
// scripted engine for the decision-trace tests.

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

native::rk::State rk_state_from(const nb::dict values) {
  native::rk::State state;
  try {
    state.z = nb::cast<double>(values["z"]);
    state.proposal = nb::cast<double>(values["proposal"]);
    for (const auto value : nb::cast<nb::list>(values["norms"]))
      state.norms.push_back(nb::cast<double>(value));
    if (values.contains("watch"))
      state.watch = watch_state_from(nb::cast<nb::dict>(values["watch"]));
    if (values.contains("cursor"))
      state.cursor = nb::cast<std::uint64_t>(values["cursor"]);
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("the Runge-Kutta march's state has the wrong types");
  }
  return state;
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
  nb::dict held;
  held["z"] = state.z;
  held["proposal"] = state.proposal;
  held["norms"] = std::vector<double>(state.norms.begin(), state.norms.end());
  held["watch"] = watch_state_to(state.watch);
  held["cursor"] = state.cursor;
  values["state"] = held;
  return values;
}

march::State march_state_from(const nb::dict values) {
  march::State state;
  try {
    state.z = nb::cast<double>(values["z"]);
    state.proposal = nb::cast<double>(values["proposal"]);
    state.order = nb::cast<std::size_t>(values["order"]);
    state.rejections = nb::cast<std::size_t>(values["rejections"]);
    state.last_rejected = nb::cast<bool>(values["last_rejected"]);
    state.collar_was_quiet = nb::cast<bool>(values["collar_was_quiet"]);
    state.cursor = nb::cast<std::size_t>(values["cursor"]);
    for (const auto entry : nb::cast<nb::list>(values["history"]))
      state.history.push_back(
          native::records::read<march::Entry>(entry, "state.history"));
    const auto ledger = nb::cast<nb::dict>(values["ledger"]);
    state.ledger.startup = nb::cast<std::uint64_t>(ledger["startup"]);
    state.ledger.predictor = nb::cast<std::uint64_t>(ledger["predictor"]);
    state.ledger.rejected = nb::cast<std::uint64_t>(ledger["rejected"]);
    state.ledger.accepted_steps = nb::cast<std::uint64_t>(ledger["accepted_steps"]);
    state.ledger.rejected_steps = nb::cast<std::uint64_t>(ledger["rejected_steps"]);
    state.ledger.restarts = nb::cast<std::uint64_t>(ledger["restarts"]);
    state.ledger.failure_restarts = nb::cast<std::uint64_t>(ledger["failure_restarts"]);
    state.ledger.dense_observations =
        nb::cast<std::uint64_t>(ledger["dense_observations"]);
    if (values.contains("watch"))
      state.watch = watch_state_from(nb::cast<nb::dict>(values["watch"]));
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("the Adams march's state has the wrong types");
  }
  if (state.history.size() > march::max_history)
    throw std::invalid_argument("the Adams march's history is too long");
  return state;
}

nb::dict march_state_to(const march::State& state) {
  nb::dict out;
  out["z"] = state.z;
  out["proposal"] = state.proposal;
  out["order"] = state.order;
  out["rejections"] = state.rejections;
  out["last_rejected"] = state.last_rejected;
  out["collar_was_quiet"] = state.collar_was_quiet;
  out["cursor"] = state.cursor;
  nb::list history;
  for (const auto& entry : state.history)
    history.append(nb::make_tuple(entry.z, native::records::write_value(entry.identity),
                                  entry.row));
  out["history"] = history;
  nb::dict ledger;
  ledger["startup"] = state.ledger.startup;
  ledger["predictor"] = state.ledger.predictor;
  ledger["rejected"] = state.ledger.rejected;
  ledger["accepted_steps"] = state.ledger.accepted_steps;
  ledger["rejected_steps"] = state.ledger.rejected_steps;
  ledger["restarts"] = state.ledger.restarts;
  ledger["failure_restarts"] = state.ledger.failure_restarts;
  ledger["dense_observations"] = state.ledger.dense_observations;
  out["ledger"] = ledger;
  out["watch"] = watch_state_to(state.watch);
  return out;
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
  out["state"] = march_state_to(state);
  return out;
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
    auto state = rk_state_from(state_values);
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

namespace solprop_native {

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

} // namespace solprop_native

namespace {

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

  native::session::Session& session() const { return run_->session(); }

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

class BoundStatisticsPlan {
public:
  explicit BoundStatisticsPlan(const nb::dict values) {
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
        throw std::invalid_argument(
            "a statistics plan's apertures are two-dimensional");
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
      plan->width_weights =
          real_vector(exact_array<RealInput>(values["width_weights"]));
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
  [[nodiscard]] const std::shared_ptr<const native::statistics::Plan>& plan() const {
    return plan_;
  }
  [[nodiscard]] std::size_t packed_size() const { return plan_->packed_size(); }

private:
  std::shared_ptr<const native::statistics::Plan> plan_;
};

} // namespace

// The statistics plans a run's start names cross as the bound `StatisticsPlan`.
template <>
struct solprop_native::records::Extern<
    std::shared_ptr<const solprop_native::statistics::Plan>> {
  static constexpr const char* kind = "StatisticsPlan";
  static void read(const nb::handle value,
                   std::shared_ptr<const solprop_native::statistics::Plan>* out,
                   const std::string& path) {
    if (value.is_none()) {
      if (out != nullptr)
        out->reset();
      return;
    }
    const BoundStatisticsPlan* plan = nullptr;
    if (!nb::try_cast<const BoundStatisticsPlan*>(value, plan) || plan == nullptr)
      solprop_native::records::type_refusal(path, "must be a StatisticsPlan or None");
    if (out != nullptr)
      *out = plan->plan();
  }
  static nb::object
  write(const std::shared_ptr<const solprop_native::statistics::Plan>& plan) {
    return plan ? nb::cast(plan->packed_size()) : nb::none();
  }
};

namespace {

namespace records = native::records;

// The runtime's one record (solprop SEP 52 section 5.3): the thread pools, the BLAS
// fitting and the spin, applied once at import.
struct RuntimeRecord {
  int threads = 0;
  int kernels = 0;
  int fft = 0;
  int raman_fft = 0;
  int raman_tail_fft = 0;
  int band_tiles = -1;
  int spin_milliseconds = 0;
  std::optional<int> blas_threads;
  bool host_blas = true;
};

[[nodiscard]] constexpr auto describe(const RuntimeRecord* /*unused*/) {
  using R = RuntimeRecord;
  return native::fields::table<R>(
      "RuntimeRecord", native::fields::value("threads", &R::threads),
      native::fields::value("kernels", &R::kernels),
      native::fields::value("fft", &R::fft),
      native::fields::value("raman_fft", &R::raman_fft),
      native::fields::value("raman_tail_fft", &R::raman_tail_fft),
      native::fields::value("band_tiles", &R::band_tiles),
      native::fields::value("spin_milliseconds", &R::spin_milliseconds),
      native::fields::value("blas_threads", &R::blas_threads),
      native::fields::value("host_blas", &R::host_blas));
}

// Defined with the library lookups below.
std::string fit_blas_threads(int threads);
void register_host_blas();

nb::dict configure_runtime(const nb::handle value) {
  const auto record = records::read<RuntimeRecord>(value, "runtime");
  initialize_kokkos(record.threads, record.kernels, record.fft, record.raman_fft,
                    record.raman_tail_fft, record.band_tiles);
  native::configure_native_spin(record.spin_milliseconds);
  nb::dict out;
  out["blas_policy"] =
      record.blas_threads ? fit_blas_threads(*record.blas_threads) : std::string();
  if (record.host_blas)
    register_host_blas();
  return out;
}

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

// solprop SEP 52 section 5.3: one problem's workspace, its set-once data applied when
// it is built, reused across the runs of a scan.
class BoundWorkspace final : public Owned {
public:
  explicit BoundWorkspace(native::Problem problem)
      : raman_terms_(problem.raman.n_terms), raman_(problem.raman.enabled),
        workspace_(std::make_unique<native::Workspace>(std::move(problem))) {
    register_owned(this);
  }
  ~BoundWorkspace() override {
    close();
    forget_owned(this);
  }
  BoundWorkspace(const BoundWorkspace&) = delete;
  BoundWorkspace& operator=(const BoundWorkspace&) = delete;
  BoundWorkspace(BoundWorkspace&&) = delete;
  BoundWorkspace& operator=(BoundWorkspace&&) = delete;

  [[nodiscard]] native::Workspace& workspace() const { return *workspace_; }

  std::unique_ptr<BoundSession> start(const nb::handle run, const nb::object& host) {
    auto held = records::read<native::RunStart>(run, "run");
    bind_host(host);
    std::unique_ptr<native::Run> started;
    {
      const nb::gil_scoped_release release;
      started = workspace_->start(std::move(held));
    }
    return std::make_unique<BoundSession>(std::move(started), workspace_->columns());
  }
  std::unique_ptr<BoundProbe> probe(const nb::handle run, const nb::object& host) {
    const auto held = records::read<native::RunStart>(run, "start");
    bind_host(host);
    std::unique_ptr<native::Probe> probe;
    {
      const nb::gil_scoped_release release;
      probe = workspace_->probe(held);
    }
    return std::make_unique<BoundProbe>(std::move(probe), workspace_->columns());
  }
  // Everything `NativeBackend.metadata` assembles from the state's facts.
  [[nodiscard]] nb::dict diagnostics() const {
    const auto& state = workspace_->built();
    nb::dict out;
    out["memory_ledger"] = state.memory_ledger();
    out["execution_sizing"] = state.execution_sizing();
    out["cpu_execution"] = cpu_execution_dict(state.cpu_execution());
    out["fft_planning"] = fft_planning_dict(state.fft_planning());
    out["planning_seconds"] = state.planning_seconds();
    out["raman_planning_seconds"] = state.raman_planning_seconds();
    out["raman_fft_provider"] = state.raman_fft_provider();
    out["raman_fft_tasks"] = state.raman_fft_tasks();
    out["raman_kernel_transforms"] = state.raman_kernel_transforms();
    out["fsal_ready"] = state.fsal_ready();
    out["rk_source_evaluations"] = state.rk_source_evaluations();
    out["axis_scans"] = state.axis_scans();
    out["synthesis_provider"] = state.synthesis_provider();
    out["projection_provider"] = state.projection_provider();
    out["gemm_shape_policy"] = state.gemm_shape_policy();
    out["response_team_size"] = state.response_team_size();
    out["dispersive_path"] = state.dispersive_path();
    const auto crossed = state.crossed_bytes();
    out["crossed_bytes"] = nb::make_tuple(crossed.to_host, crossed.from_host);
    out["table_evaluations"] = workspace_->z_slots().evaluations;
    static constexpr std::array<const char*, 3> kinds{"adams", "runge_kutta",
                                                      "low_storage"};
    const auto kind = workspace_->kind();
    out["method"] = kinds[static_cast<std::size_t>(kind)];
    out["history_rows"] = kind == native::MethodKind::adams
                              ? state.adams_row_names()
                              : std::vector<std::string>{};
    return out;
  }
  // The live counters alone, cheap enough to read at every run's start and close and
  // every plane's metadata: the on-axis row scans, the Runge-Kutta stage evaluations,
  // the z-table positions, the Raman kernel's transforms and the recycled stage.
  [[nodiscard]] nb::dict counters() const {
    const auto& state = workspace_->built();
    nb::dict out;
    out["axis_scans"] = state.axis_scans();
    out["rk_source_evaluations"] = state.rk_source_evaluations();
    out["table_evaluations"] = workspace_->z_slots().evaluations;
    out["raman_kernel_transforms"] = state.raman_kernel_transforms();
    out["fsal_ready"] = state.fsal_ready();
    return out;
  }
  [[nodiscard]] bool closed() const noexcept { return workspace_->closed(); }
  // Closes a session or probe that leases the workspace first. Python-free: the runtime
  // calls it at shutdown, after the interpreter.
  void close() noexcept override { workspace_->close(); }
  // `close` from Python: without the interpreter lock (a session's close waits for an
  // advance in hand), then the host source dropped with it once the state is gone,
  // since the state's reference to it is invisible to the garbage collector.
  void close_from_python() noexcept {
    {
      const nb::gil_scoped_release release;
      close();
    }
    if (workspace_->closed())
      host_owner_ = nb::object();
  }
  [[nodiscard]] std::size_t raman_terms() const noexcept { return raman_terms_; }
  [[nodiscard]] bool raman() const noexcept { return raman_; }

private:
  // Bind `host` as the next run's host source (solprop SEP 49), or unbind with None.
  void bind_host(const nb::object& host) {
    workspace_->require_unleased();
    if (host.is_none()) {
      workspace_->bind_host_source(nullptr);
      host_owner_ = nb::object();
      return;
    }
    workspace_->bind_host_source(std::make_shared<PythonHostSource>(host));
    host_owner_ = host;
  }

  std::size_t raman_terms_{};
  bool raman_{};
  std::unique_ptr<native::Workspace> workspace_;
  nb::object host_owner_;
};

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

void shutdown_kokkos() {
  const std::scoped_lock lock(runtime_state.mutex);
  for (auto* state : runtime_state.states)
    state->close();
  if (runtime_state.owns_runtime && Kokkos::is_initialized() &&
      !Kokkos::is_finalized()) {
    Kokkos::finalize();
    runtime_state.owns_runtime = false;
  }
}

// Kokkos::finalize runs Kokkos-FFT's hook, fftw_cleanup_threads, which destroys
// FFTW's process-wide planner. Any other user of the same FFTW library must have
// destroyed its plans by then, so the owned
// runtime is finalised after the interpreter has released every Python object,
// rather than from a Python exit handler, which runs before. Nothing here calls
// Python.
void shutdown_after_interpreter() noexcept { shutdown_kokkos(); }

std::string decode_record_text(std::string value) {
  const std::string marker = "__SOLPROP_SPACE__";
  std::size_t position = 0;
  while ((position = value.find(marker, position)) != std::string::npos) {
    value.replace(position, marker.size(), " ");
    ++position;
  }
  return value;
}

#ifdef _WIN32
// Windows looks a symbol up in one module only, so the search walks the modules this
// extension imports (its FFT, BLAS and threading libraries, under whatever names the
// wheel repair gave them), as dlsym on the extension's own handle does on POSIX.
HMODULE native_module() {
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(&decode_record_text), &module) == 0)
    throw std::runtime_error("cannot locate the native module for library provenance");
  return module;
}

std::pair<void*, HMODULE> find_symbol(const char* symbol) {
  const HMODULE self = native_module();
  // A library linked statically into the extension (the wheel's OpenBLAS) is found
  // through the extension's own exports.
  if (const auto address = GetProcAddress(self, symbol))
    return {reinterpret_cast<void*>(address), self};
  const auto* base = reinterpret_cast<const unsigned char*>(self);
  const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
  const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
  const auto& directory =
      nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
  if (directory.VirtualAddress == 0)
    return {nullptr, nullptr};
  for (const auto* entry = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(
           base + directory.VirtualAddress);
       entry->Name != 0; ++entry) {
    const HMODULE dependency =
        GetModuleHandleA(reinterpret_cast<const char*>(base + entry->Name));
    if (dependency == nullptr)
      continue;
    if (const auto address = GetProcAddress(dependency, symbol))
      return {reinterpret_cast<void*>(address), dependency};
  }
  return {nullptr, nullptr};
}

std::string loaded_library(const char* symbol) {
  const auto [address, module] = find_symbol(symbol);
  if (address == nullptr)
    return {};
  std::wstring path(MAX_PATH, L'\0');
  const auto length =
      GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
  if (length == 0)
    return {};
  path.resize(length);
  const int bytes =
      WideCharToMultiByte(CP_UTF8, 0, path.data(), static_cast<int>(path.size()),
                          nullptr, 0, nullptr, nullptr);
  std::string result(static_cast<std::size_t>(bytes), '\0');
  WideCharToMultiByte(CP_UTF8, 0, path.data(), static_cast<int>(path.size()),
                      result.data(), bytes, nullptr, nullptr);
  return result;
}

void* library_symbol(const char* symbol) { return find_symbol(symbol).first; }
#else
struct LibraryCloser {
  // Keep glibc's function attributes out of unique_ptr's template arguments.
  void operator()(void* handle) const noexcept { static_cast<void>(dlclose(handle)); }
};

auto native_library_handle() {
  Dl_info module{};
  if (dladdr(reinterpret_cast<const void*>(&decode_record_text), &module) == 0)
    throw std::runtime_error("cannot locate the native module for library provenance");
  std::unique_ptr<void, LibraryCloser> handle(
      dlopen(module.dli_fname, RTLD_NOW | RTLD_NOLOAD));
  if (!handle)
    throw std::runtime_error("cannot inspect the native module's loaded libraries");
  return handle;
}

std::string loaded_library(const char* symbol) {
  const auto handle = native_library_handle();
  const auto* address = dlsym(handle.get(), symbol);
  Dl_info provider{};
  return address != nullptr && dladdr(address, &provider) != 0
             ? std::string(provider.dli_fname)
             : std::string{};
}

void* library_symbol(const char* symbol) {
  const auto handle = native_library_handle();
  return dlsym(handle.get(), symbol);
}
#endif

// A distribution build's OpenBLAS (scipy-openblas32) prefixes every symbol.
std::string blas_symbol(const char* name) {
  return std::string(SOLPROP_NATIVE_BLAS_SYMBOL_PREFIX) + name;
}

template <class... Arguments>
std::optional<int> library_integer(const char* symbol, Arguments... arguments) {
  void* const address = library_symbol(symbol);
  if (address == nullptr)
    return std::nullopt;
  return reinterpret_cast<int (*)(Arguments...)>(address)(arguments...);
}

// Fit a host OpenBLAS to the Kokkos pool. A pthreads OpenBLAS keeps its own thread
// pool beside Kokkos's and oversubscribes the cores, so it is held to one thread; an
// OpenMP OpenBLAS shares the runtime but sizes itself from OMP_NUM_THREADS (every
// logical CPU when unset), so it is capped at the Kokkos thread count. Returns the
// policy applied, or an empty string when nothing was changed.
std::string fit_blas_threads(const int threads) {
  const std::scoped_lock lock(runtime_state.mutex);
  if (std::string(SOLPROP_NATIVE_BLAS_BACKEND) != "OpenBLAS" || threads < 1)
    return {};
  const auto parallel = library_integer(blas_symbol("openblas_get_parallel").c_str());
  void* const address = library_symbol(blas_symbol("openblas_set_num_threads").c_str());
  if (address == nullptr || (parallel != 1 && parallel != 2))
    return {};
  reinterpret_cast<void (*)(int)>(address)(parallel == 1 ? 1 : threads);
  runtime_state.blas_policy =
      parallel == 1 ? "one thread beside the Kokkos pool" : "the Kokkos thread count";
  return runtime_state.blas_policy;
}

// Let a compact problem's scope hold an OpenMP OpenBLAS at one thread (`CpuTeamScope`),
// whatever width the environment or `fit_blas_threads` gave it. A pthreads or serial
// OpenBLAS keeps no team beside Kokkos's to resize.
void register_host_blas() {
  if (std::string(SOLPROP_NATIVE_BLAS_BACKEND) != "OpenBLAS")
    return;
  const auto parallel = library_integer(blas_symbol("openblas_get_parallel").c_str());
  native::register_openmp_blas(parallel == 2);
}

nb::dict build_record() {
  const std::scoped_lock lock(runtime_state.mutex);
  nb::dict result;
  result["backend_commit"] = SOLPROP_NATIVE_BACKEND_COMMIT;
  result["solprop_built_against"] = SOLPROP_NATIVE_SOLPROP_BUILT_AGAINST;
  result["source_digest"] = native::source_digest();
  result["kokkos_version"] = SOLPROP_NATIVE_KOKKOS_VERSION;
  result["kokkos_fft_version"] = SOLPROP_NATIVE_KOKKOSFFT_VERSION;
  result["kokkos_kernels_version"] = SOLPROP_NATIVE_KOKKOSKERNELS_VERSION;
  result["nanobind_version"] = SOLPROP_NATIVE_NANOBIND_VERSION;
  result["gemm_provider"] = "HYBRID";
  result["tpl_blas_enabled"] = (SOLPROP_NATIVE_TPL_BLAS_ENABLED != 0);
  result["blas_backend"] = SOLPROP_NATIVE_BLAS_BACKEND;
  result["blas_thread_policy"] = decode_record_text(SOLPROP_NATIVE_BLAS_THREAD_POLICY);
  result["blas_selection"] = decode_record_text(SOLPROP_NATIVE_BLAS_SELECTION);
  result["build_profile"] = SOLPROP_NATIVE_BUILD_PROFILE;
  result["cpu_baseline"] = SOLPROP_NATIVE_CPU_BASELINE;
  // The host kernels' transcendental passes (solprop ADR-059): SLEEF's u10 functions
  // in the variant the build targets, or the system libm.
  result["vector_math"] = solprop_native::vmath::vectorised
                              ? std::string("SLEEF ") + SOLPROP_NATIVE_SLEEF_VERSION +
                                    " u10, " + solprop_native::vmath::route
                              : std::string(solprop_native::vmath::route);
  std::optional<int> blas_threads;
  std::string blas_threading = "provider-managed";
  if (std::string(SOLPROP_NATIVE_BLAS_BACKEND) == "OpenBLAS") {
    blas_threads = library_integer(blas_symbol("openblas_get_num_threads").c_str());
    const auto parallel = library_integer(blas_symbol("openblas_get_parallel").c_str());
    blas_threading = parallel == 2   ? "OpenMP"
                     : parallel == 1 ? "pthreads"
                     : parallel == 0 ? "serial"
                                     : "unknown";
    if (!runtime_state.blas_policy.empty())
      result["blas_thread_policy"] = runtime_state.blas_policy;
    else if (parallel == 2)
      result["blas_thread_policy"] = "OMP_NUM_THREADS at process start";
  } else if (std::string(SOLPROP_NATIVE_BLAS_BACKEND) == "MKL") {
    // MKL_DOMAIN_BLAS is 1 in the installed/public LP64 service interface.
    blas_threads = library_integer("MKL_Domain_Get_Max_Threads", 1);
    blas_threading = "MKL BLAS domain";
  }
  result["blas_threads"] = blas_threads;
  result["blas_threading"] = blas_threading;
  result["execution_space"] = native::ExecutionSpace::name();
#ifdef KOKKOS_ENABLE_CUDA
  result["fft_backend"] = "cuFFT";
  result["fft_library"] = loaded_library("cufftExecZ2Z");
  result["blas_library"] = loaded_library("cublasDgemm_v2");
  result["device_backend"] = SOLPROP_NATIVE_DEVICE_BACKEND;
  result["device_arch"] = SOLPROP_NATIVE_DEVICE_ARCH;
  result["device_count"] = Kokkos::num_devices();
#elif defined(KOKKOS_ENABLE_HIP)
  result["fft_backend"] = "rocFFT";
  result["fft_library"] = loaded_library("rocfft_execute");
  result["blas_library"] = loaded_library("rocblas_dgemm");
  result["device_backend"] = SOLPROP_NATIVE_DEVICE_BACKEND;
  result["device_arch"] = SOLPROP_NATIVE_DEVICE_ARCH;
  result["device_count"] = Kokkos::num_devices();
#elif defined(KOKKOS_ENABLE_OPENMP)
  result["fft_backend"] = SOLPROP_NATIVE_HOST_FFT_BACKEND;
  result["fft_library"] = loaded_library("fftw_execute");
  result["blas_library"] = loaded_library(blas_symbol("dgemm_").c_str());
  result["device_backend"] = SOLPROP_NATIVE_DEVICE_BACKEND;
  result["device_arch"] = SOLPROP_NATIVE_DEVICE_ARCH;
  result["device_count"] = 0;
#else
  result["fft_backend"] = SOLPROP_NATIVE_HOST_FFT_BACKEND;
  result["fft_library"] = loaded_library("fftw_execute");
  result["blas_library"] = loaded_library(blas_symbol("dgemm_").c_str());
  result["device_backend"] = SOLPROP_NATIVE_DEVICE_BACKEND;
  result["device_arch"] = SOLPROP_NATIVE_DEVICE_ARCH;
  result["device_count"] = 0;
#endif
  result["raman_fft_option"] = SOLPROP_NATIVE_RAMAN_FFT_PROVIDER;
#ifdef SOLPROP_NATIVE_RAMAN_ACCELERATE
  result["raman_fft_library"] = loaded_library("vDSP_fftm_zropD");
#else
  result["raman_fft_library"] = result["fft_library"];
#endif
  result["openmp_library"] = loaded_library("omp_get_num_threads");
  result["device_team_size"] = SOLPROP_NATIVE_DEVICE_TEAM_SIZE_RECORD;
  result["device_reduction_size"] = SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE_RECORD;
#ifdef KOKKOS_ENABLE_OPENMP
  result["openmp_enabled"] = true;
#else
  result["openmp_enabled"] = false;
#endif
  result["host_execution_space"] = Kokkos::DefaultHostExecutionSpace::name();
  const auto threads = native::thread_policy();
  result["thread_count"] = threads.kernels;
  result["runtime_threads"] = Kokkos::DefaultHostExecutionSpace().concurrency();
  result["requested_threads"] = runtime_state.requested_threads;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  result["fftw_threads"] = 0;
  result["raman_fft_threads"] = 0;
  result["raman_tail_fft_threads"] = 0;
#else
  result["fftw_threads"] = threads.fft;
  result["raman_fft_threads"] = threads.raman_fft;
  result["raman_tail_fft_threads"] = threads.raman_tail_fft;
#endif
  result["cpu_size_policy_enabled"] = native::cpu_size_policy_enabled;
  result["explicit_thread_pools"] = threads.explicit_pools;
  result["band_column"] = native::band_tiled(threads) ? "tiled" : "team";
  result["band_tile_threads"] = native::band_tile_threads(threads);
  result["cxx_standard"] = 20;
  result["compiler"] = __VERSION__;
  result["compiler_flags"] = decode_record_text(SOLPROP_NATIVE_COMPILER_FLAGS);
  result["build_type"] = SOLPROP_NATIVE_BUILD_TYPE;
  result["coverage_instrumented"] = (SOLPROP_NATIVE_COVERAGE_INSTRUMENTED != 0);
  result["coverage_source_hash"] = SOLPROP_NATIVE_COVERAGE_SOURCE_HASH;
  result["fft_normalization"] = "none (explicit inverse scaling)";
#ifdef KOKKOS_ENABLE_CUDA
  result["fft_planner"] = "Kokkos-FFT persistent Plan / cuFFT";
#elif defined(KOKKOS_ENABLE_HIP)
  result["fft_planner"] = "Kokkos-FFT persistent Plan / rocFFT";
#else
  result["fft_planner"] = std::string("Kokkos-FFT persistent Plan / ") +
                          SOLPROP_NATIVE_HOST_FFT_PLANNER +
                          "; wisdom seeded before planning: rows measured serial, "
                          "estimate threaded; transverse planes measured; "
                          "SOLPROP_FFTW_PLAN overrides";
#endif
  result["layout"] = "LayoutRight";
  result["native_arch_enabled"] = (SOLPROP_NATIVE_HOST_ARCH_ENABLED != 0);
  result["aggressive_vectorization_enabled"] =
      (SOLPROP_NATIVE_AGGRESSIVE_VECTORIZATION_ENABLED != 0);
  result["ipo_enabled"] = (SOLPROP_NATIVE_IPO_ENABLED != 0);
  const std::string cpu_baseline = SOLPROP_NATIVE_CPU_BASELINE;
  const std::string architecture_policy =
      SOLPROP_NATIVE_HOST_ARCH_ENABLED ? "native architecture"
      : cpu_baseline == "none"         ? "generic architecture"
                                       : cpu_baseline + " baseline";
  const std::string ipo_policy =
      SOLPROP_NATIVE_IPO_ENABLED ? "checked IPO" : "IPO disabled";
  const std::string vectorization_policy =
      SOLPROP_NATIVE_AGGRESSIVE_VECTORIZATION_ENABLED
          ? "aggressive Kokkos loop vectorization"
          : "standard Kokkos loop vectorization";
  // solprop SEP 25: contraction on (protected blocks strict), or the device compiler's.
  const std::string contraction = std::string(SOLPROP_NATIVE_FP_CONTRACT_RECORD);
  const std::string contraction_policy =
      contraction.empty() ? "compiler-default contraction"
                          : "contraction " + contraction + ", protected blocks strict";
  result["optimization_policy"] = std::string("CMake ") + SOLPROP_NATIVE_BUILD_TYPE +
                                  "; no fast-math; " + contraction_policy + "; " +
                                  architecture_policy + "; " + ipo_policy + "; " +
                                  vectorization_policy;
#ifdef __FAST_MATH__
  result["fast_math_enabled"] = true;
#else
  result["fast_math_enabled"] = false;
#endif
#ifdef __apple_build_version__
  result["compiler_family"] = "AppleClang";
#elif defined(__clang__)
  result["compiler_family"] = "Clang";
#elif defined(__GNUC__)
  result["compiler_family"] = "GNU";
#else
  result["compiler_family"] = "unknown";
#endif
  result["live_native_states"] = runtime_state.states.size();
  return result;
}

// solprop SEP 52 section 6.5: native's engines through solprop's Python protocols
// (`solprop.backends.engines`), for the engine-level parity tests, with a read-back of
// the field and of a stage register. The object leases the workspace while it lives,
// as a session does.
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
  std::vector<double> trial_step(const double dz, const native::TrialScalars& scalars,
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
  std::vector<double> form_flow_actions(const std::size_t section, const double dz,
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
    auto state = rk_state_from(state_values);
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
  native::GeneralState& state() const {
    if (owner_ == nullptr)
      throw std::logic_error("the engines are closed");
    return owner_->leased_state(this);
  }
  native::rk::Engine& runge_kutta() const {
    state();
    if (!runge_kutta_)
      throw std::logic_error("this workspace has no Runge-Kutta engine");
    return *runge_kutta_;
  }
  native::march::Engine& adams() const {
    state();
    if (!adams_)
      throw std::logic_error("this workspace has no Adams engine");
    return *adams_;
  }
  ComplexOutput field() const {
    const std::size_t count = owner_ == nullptr ? 0 : owner_->elements();
    auto storage = native::crossing::host_buffer(2 * count);
    state().field_to_host({reinterpret_cast<native::Complex*>(storage.get()), count});
    return complex_buffer_output(std::move(storage),
                                 {owner_->columns(), count / owner_->columns()});
  }
  ComplexOutput stage(const std::size_t index) const {
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
  std::copy(nodes.begin(), nodes.end(), stencil.nodes.begin());
  std::copy(rows.begin(), rows.end(), stencil.rows.begin());
  stencil.held = nodes.size();
  stencil.order = nb::cast<std::size_t>(value.attr("order"));
  std::copy(orders.begin(), orders.end(), stencil.orders.begin());
  stencil.n_orders = orders.size();
  return stencil;
}

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
        // A varying operator's compensation tables (solprop's
        // backends.multistep.compensation_tables); empty for a constant one.
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

} // namespace

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
