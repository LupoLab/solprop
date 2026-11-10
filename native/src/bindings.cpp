// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "bindings_transverse.hpp"
#include "helpers.hpp"
#include "solprop_native/adams_march.hpp"
#include "solprop_native/core.hpp"
#include "solprop_native/crossing.hpp"
#include "solprop_native/dense_output.hpp"
#include "solprop_native/engines.hpp"
#include "solprop_native/fft_plan.hpp"
#include "solprop_native/lagrange_basis.hpp"
#include "solprop_native/low_storage.hpp"
#include "solprop_native/rk_march.hpp"
#include "solprop_native/session.hpp"
#include "solprop_native/stage_tables.hpp"
#include "solprop_native/vmath.hpp"
#include "solprop_native/watch.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>

// `available_device_memory` calls the vendor runtime directly. Kokkos pulls
// these headers in transitively today; declaring them here is what makes the
// probe's own dependency explicit rather than a property of a Kokkos version.
// The guards resolve only after Kokkos_Core.hpp has defined them.
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
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
using ComplexStorage = std::vector<std::complex<double>>;
using RealStorage = std::vector<double>;

class BoundState;
class BoundBatch;
class BoundSession;

template <class Array> Array exact_array(const nb::handle value) {
  return nb::cast<Array>(value, false);
}

struct RuntimeState {
  std::mutex mutex;
  bool owns_runtime = false;
  int requested_threads = 0;
  // The BLAS thread policy the package applied at start-up, if any.
  std::string blas_policy;
  std::vector<BoundState*> states;
  std::vector<BoundBatch*> batches;
};

RuntimeState runtime_state;

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

native::FieldFamily field_family(const std::string& name) {
  if (name == "field_real")
    return native::FieldFamily::field_real;
  if (name == "field_analytic")
    return native::FieldFamily::field_analytic;
  if (name == "envelope")
    return native::FieldFamily::envelope;
  throw std::invalid_argument("unknown field family " + name);
}

native::TableKind table_kind(const std::string& name) {
  if (name == "phi")
    return native::TableKind::phi;
  if (name == "prefactor")
    return native::TableKind::prefactor;
  if (name == "rate")
    return native::TableKind::rate;
  if (name == "flux")
    return native::TableKind::flux;
  throw std::invalid_argument("unknown z table '" + name +
                              "'; expected 'phi', 'prefactor', 'rate' or 'flux'");
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
    throw std::invalid_argument("a CPU execution record has " +
                                std::to_string(expected) + " fields, got " +
                                std::to_string(values.size()));
  const auto field = [&values](const std::string& name) {
    if (!values.contains(name.c_str()))
      throw std::invalid_argument("a CPU execution record misses a field: " + name);
    return values[name.c_str()];
  };
  try {
    for (std::size_t index = 0; index < native::cpu_operation_count; ++index) {
      const std::string name = cpu_operation_names[index];
      result.grain[index] = {nb::cast<std::size_t>(field(name + "_threshold")),
                             nb::cast<std::size_t>(field(name + "_full"))};
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

native::Problem problem_from(const nb::dict& values) {
  native::Problem result;
  if (!values.contains("cpu_execution"))
    throw std::invalid_argument("a native problem needs its CPU execution record "
                                "(cpu_execution, solprop SEP 46)");
  try {
    result.cpu_execution =
        cpu_execution_from(nb::cast<nb::dict>(values["cpu_execution"]));
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("a native problem's cpu_execution must be a dict");
  }
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  if (values.contains("fft_planning")) {
    try {
      result.fft_planning =
          fft_planning_from(nb::cast<nb::dict>(values["fft_planning"]));
    } catch (const nb::cast_error&) {
      throw std::invalid_argument("a native problem's fft_planning must be a dict");
    }
  }
#endif
  try {
    if (values.contains("n_x")) {
      result.n_x = nb::cast<std::size_t>(values["n_x"]);
      result.n_y = nb::cast<std::size_t>(values["n_y"]);
      if (values.contains("n_p"))
        result.n_p = nb::cast<std::size_t>(values["n_p"]);
      if (values.contains("spatial_far_field_slots"))
        result.spatial_far_field_slots =
            nb::cast<std::size_t>(values["spatial_far_field_slots"]);
    }
    result.n_omega = nb::cast<std::size_t>(values["n_omega"]);
    result.n_t = nb::cast<std::size_t>(values["n_t"]);
    result.n_to = nb::cast<std::size_t>(values["n_to"]);
    result.n_stages = nb::cast<std::size_t>(values["n_stages"]);
    if (values.contains("n_modes")) {
      result.n_modes = nb::cast<std::size_t>(values["n_modes"]);
      result.n_pol_state = nb::cast<std::size_t>(values["n_pol_state"]);
      result.n_pol_transform = nb::cast<std::size_t>(values["n_pol_transform"]);
      result.n_nodes = nb::cast<std::size_t>(values["n_nodes"]);
      result.node_chunk = nb::cast<std::size_t>(values["node_chunk"]);
    }
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("problem dimensions must be integers");
  }
  try {
    result.family = field_family(nb::cast<std::string>(values["family"]));
    result.nonlinear = nb::cast<bool>(values["nonlinear"]);
    result.modified_noise = nb::cast<bool>(values["modified_noise"]);
    result.in_stage_absorber = nb::cast<bool>(values["in_stage_absorber"]);
    result.split_step_absorber = nb::cast<bool>(values["split_step_absorber"]);
    result.first_same_as_last = nb::cast<bool>(values["first_same_as_last"]);
    if (values.contains("low_storage")) {
      result.low_storage = nb::cast<bool>(values["low_storage"]);
      result.low_storage_stage_register =
          nb::cast<bool>(values["low_storage_stage_register"]);
      result.low_storage_flows = nb::cast<std::size_t>(values["low_storage_flows"]);
      result.low_storage_tiled_input =
          nb::cast<bool>(values["low_storage_tiled_input"]);
    }
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("problem family and flags have wrong types");
  }
  try {
    result.crop_index = index_vector(exact_array<IndexInput>(values["crop_index"]));
    result.active_mask = bool_vector(exact_array<BoolInput>(values["active_mask"]));
    result.norm_mask = bool_vector(exact_array<BoolInput>(values["norm_mask"]));
    result.omega_window = real_vector(exact_array<RealInput>(values["omega_window"]));
    if (values.contains("omega"))
      result.omega = real_vector(exact_array<RealInput>(values["omega"]));
    if (values.contains("power_weight"))
      result.power_weight = real_vector(exact_array<RealInput>(values["power_weight"]));
    result.time_window = real_vector(exact_array<RealInput>(values["time_window"]));
    result.time_window_derivative =
        real_vector(exact_array<RealInput>(values["time_window_derivative"]));
    result.absorber_fine = real_vector(exact_array<RealInput>(values["absorber_fine"]));
    result.absorber_coarse =
        real_vector(exact_array<RealInput>(values["absorber_coarse"]));
    if (result.separable_geometry()) {
      result.spatial_evanescent_rate =
          nb::cast<double>(values["spatial_evanescent_rate"]);
      result.spatial_k_squared =
          complex_vector(exact_array<ComplexInput>(values["spatial_k_squared"]));
      result.spatial_k_squared_low =
          complex_vector(exact_array<ComplexInput>(values["spatial_k_squared_low"]));
      result.spatial_numerator =
          complex_vector(exact_array<ComplexInput>(values["spatial_numerator"]));
      result.spatial_clamped =
          complex_vector(exact_array<ComplexInput>(values["spatial_clamped"]));
      result.spatial_kperp_sq =
          real_vector(exact_array<RealInput>(values["spatial_kperp_sq"]));
      result.spatial_kperp_sq_low =
          real_vector(exact_array<RealInput>(values["spatial_kperp_sq_low"]));
      result.spatial_frame =
          real_vector(exact_array<RealInput>(values["spatial_frame"]));
      result.spatial_spectral_rate =
          real_vector(exact_array<RealInput>(values["spatial_spectral_rate"]));
      result.spatial_user_rate =
          real_vector(exact_array<RealInput>(values["spatial_user_rate"]));
      result.spatial_absorber_rate =
          real_vector(exact_array<RealInput>(values["spatial_absorber_rate"]));
      result.spatial_kperp_rate =
          real_vector(exact_array<RealInput>(values["spatial_kperp_rate"]));
      if (result.slab_geometry())
        result.spatial_kappa_sq =
            complex_vector(exact_array<ComplexInput>(values["spatial_kappa_sq"]));
      if (result.free_geometry()) {
        result.spatial_k_x = real_vector(exact_array<RealInput>(values["spatial_k_x"]));
        result.spatial_k_y = real_vector(exact_array<RealInput>(values["spatial_k_y"]));
      }
    }
    if (values.contains("spatial_absorber_matrix"))
      result.spatial_absorber_matrix =
          real_vector(exact_array<RealInput>(values["spatial_absorber_matrix"]));
    if (values.contains("spatial_spectral_weights")) {
      result.spatial_spectral_weights =
          real_vector(exact_array<RealInput>(values["spatial_spectral_weights"]));
      result.spatial_on_axis =
          complex_vector(exact_array<ComplexInput>(values["spatial_on_axis"]));
    }
    if (values.contains("flux_k_squared")) {
      result.flux_k_squared =
          complex_vector(exact_array<ComplexInput>(values["flux_k_squared"]));
      result.flux_k_squared_low =
          complex_vector(exact_array<ComplexInput>(values["flux_k_squared_low"]));
      result.flux_kperp_sq =
          real_vector(exact_array<RealInput>(values["flux_kperp_sq"]));
      result.flux_kperp_sq_low =
          real_vector(exact_array<RealInput>(values["flux_kperp_sq_low"]));
      result.flux_table = nb::cast<bool>(values["flux_table"]);
    }
    if (values.contains("flux_closure"))
      result.flux_closure = nb::cast<bool>(values["flux_closure"]);
    result.carrier = complex_vector(exact_array<ComplexInput>(values["carrier"]));
    if (values.contains("synthesis_matrix")) {
      result.synthesis_matrix =
          real_vector(exact_array<RealInput>(values["synthesis_matrix"]));
      result.projection_matrix =
          real_vector(exact_array<RealInput>(values["projection_matrix"]));
      result.projection_coarse_matrix =
          real_vector(exact_array<RealInput>(values["projection_coarse_matrix"]));
    }
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("problem masks and transform arrays have wrong dtypes");
  }
  ComplexInput phi;
  ComplexInput prefactor;
  try {
    phi = exact_array<ComplexInput>(values["phi_terms"]);
    prefactor = exact_array<ComplexInput>(values["prefactor_terms"]);
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("fixed terms must be complex128 arrays");
  }
  if ((phi.ndim() != 2 && phi.ndim() != 3) ||
      (prefactor.ndim() != 2 && prefactor.ndim() != 3)) {
    throw std::invalid_argument("fixed terms must be rank-two or rank-three arrays");
  }
  result.n_phi_terms = phi.shape(0);
  result.n_prefactor_terms = prefactor.shape(0);
  result.phi_terms = complex_vector(phi);
  result.prefactor_terms = complex_vector(prefactor);
  // solprop SEP 29 G1: a z table's terms are slots its evaluation fills; the leading
  // extent of an empty terms array names how many.
  try {
    if (values.contains("phi_table"))
      result.phi_table = nb::cast<bool>(values["phi_table"]);
    if (values.contains("prefactor_table"))
      result.prefactor_table = nb::cast<bool>(values["prefactor_table"]);
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("z table flags must be booleans");
  }
  try {
    result.tableau_matrix =
        real_vector(exact_array<RealInput>(values["tableau_matrix"]));
    result.tableau_nodes = real_vector(exact_array<RealInput>(values["tableau_nodes"]));
    result.solution_weights =
        real_vector(exact_array<RealInput>(values["solution_weights"]));
    result.error_weights = real_vector(exact_array<RealInput>(values["error_weights"]));
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("tableau values must be float64 arrays");
  }
  try {
    result.synthesis = nb::cast<double>(values["synthesis"]);
    result.projection = nb::cast<double>(values["projection"]);
    result.kerr_coefficient = nb::cast<double>(values["kerr_coefficient"]);
    result.self_factor = nb::cast<double>(values["self_factor"]);
    result.conjugate_factor = nb::cast<double>(values["conjugate_factor"]);
    result.harmonic_factor = nb::cast<double>(values["harmonic_factor"]);
    result.take_real = nb::cast<bool>(values["take_real"]);
    if (values.contains("plasma")) {
      result.plasma = nb::cast<bool>(values["plasma"]);
      result.raman = nb::cast<bool>(values["raman"]);
      result.raman_thg = nb::cast<bool>(values["raman_thg"]);
      result.stash_plasma = nb::cast<bool>(values["stash_plasma"]);
      result.weighted_controller = nb::cast<bool>(values["weighted_controller"]);
      result.dt_fine = nb::cast<double>(values["dt_fine"]);
      result.dt = nb::cast<double>(values["dt"]);
      result.raman_fraction = nb::cast<double>(values["raman_fraction"]);
      result.raman_epsilon0 = nb::cast<double>(values["raman_epsilon0"]);
      const auto raman_undamped = exact_array<RealInput>(values["raman_undamped"]);
      if (result.raman && (raman_undamped.ndim() != 2 || raman_undamped.shape(0) == 0 ||
                           result.n_to > std::numeric_limits<std::size_t>::max() / 2 ||
                           raman_undamped.shape(1) != 2 * result.n_to))
        throw std::invalid_argument(
            "raman_undamped must have shape (n_terms, 2 * n_to)");
      result.n_raman_terms = result.raman ? raman_undamped.shape(0) : 0;
      if (result.raman) {
        result.raman_undamped = real_vector(raman_undamped);
        result.raman_tensor_capable = nb::cast<bool>(values["raman_tensor_capable"]);
        const auto weights = exact_array<RealInput>(values["raman_tensor_weights"]);
        if (weights.ndim() != 2 || weights.shape(0) != result.n_raman_terms ||
            weights.shape(1) != 2)
          throw std::invalid_argument(
              "raman_tensor_weights must have shape (n_terms, 2)");
        result.raman_tensor_weights = real_vector(weights);
      }
      result.charge_mass_ratio = nb::cast<double>(values["charge_mass_ratio"]);
      result.ionisation_potential = nb::cast<double>(values["ionisation_potential"]);
      result.log_field_min = nb::cast<double>(values["log_field_min"]);
      result.d_log_field = nb::cast<double>(values["d_log_field"]);
      result.field_min = nb::cast<double>(values["field_min"]);
      result.field_max = nb::cast<double>(values["field_max"]);
      result.rate_max = nb::cast<double>(values["rate_max"]);
      const auto coefficients = exact_array<RealInput>(values["rate_coefficients"]);
      if (coefficients.ndim() != 2 || coefficients.shape(0) != 4)
        throw std::invalid_argument(
            "rate_coefficients must have shape (4, n_intervals)");
      result.rate_intervals = coefficients.shape(1);
      result.rate_coefficients = real_vector(coefficients);
      // solprop SEP 18: the running-integral rule. Translation designed and
      // certified the taps; the core checks their extent, finiteness and the
      // trapezoid's one identity tap.
      if (values.contains("plasma_band")) {
        result.plasma_band = nb::cast<bool>(values["plasma_band"]);
        const auto taps = exact_array<RealInput>(values["plasma_taps"]);
        if (taps.ndim() != 1)
          throw std::invalid_argument("plasma_taps must be one-dimensional");
        result.plasma_taps = real_vector(taps);
      }
      if (values.contains("n_channels")) {
        result.n_channels = nb::cast<std::size_t>(values["n_channels"]);
        result.channel_parameters =
            real_vector(exact_array<RealInput>(values["channel_parameters"]));
        result.channel_coefficients =
            real_vector(exact_array<RealInput>(values["channel_coefficients"]));
        result.n_raman_components = nb::cast<std::size_t>(values["n_raman_components"]);
        result.raman_row_component =
            real_vector(exact_array<RealInput>(values["raman_row_component"]));
        result.raman_component_fraction =
            real_vector(exact_array<RealInput>(values["raman_component_fraction"]));
      }
      if (values.contains("envelope_plasma")) {
        result.envelope_plasma = nb::cast<bool>(values["envelope_plasma"]);
        if (result.envelope_plasma) {
          const auto nodes = exact_array<RealInput>(values["envelope_rule_nodes"]);
          const auto weights = exact_array<RealInput>(values["envelope_rule_weights"]);
          for (const auto& rule : {nodes, weights})
            if (rule.ndim() != 3 || rule.shape(0) != result.n_channels ||
                rule.shape(1) != 4 || rule.shape(2) != 512)
              throw std::invalid_argument(
                  "envelope rules require (n_channels, 4, 512)");
          const auto drude = exact_array<RealInput>(values["envelope_drude_factor"]);
          const auto loss = exact_array<ComplexInput>(values["envelope_loss_factor"]);
          if (drude.ndim() != 1 || drude.shape(0) != result.n_omega ||
              loss.ndim() != 1 || loss.shape(0) != result.n_omega)
            throw std::invalid_argument("envelope current factors require (n_omega,)");
          result.envelope_rule_nodes = real_vector(nodes);
          result.envelope_rule_weights = real_vector(weights);
          result.envelope_drude_factor = real_vector(drude);
          result.envelope_loss_factor = complex_vector(loss);
        }
      }
    }
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("response scalars have wrong types");
  }
  if (values.contains("dispersive")) {
    try {
      result.dispersive = nb::cast<bool>(values["dispersive"]);
      if (result.dispersive) {
        result.dispersive_symmetric = nb::cast<bool>(values["dispersive_symmetric"]);
        result.dispersive_form = nb::cast<int>(values["dispersive_form"]);
        result.dispersive_inputs =
            real_vector(exact_array<RealInput>(values["dispersive_inputs"]));
        result.dispersive_term_multipliers =
            real_vector(exact_array<RealInput>(values["dispersive_term_multipliers"]));
        result.dispersive_term_rows =
            index_vector(exact_array<IndexInput>(values["dispersive_term_rows"]));
        result.dispersive_term_channels =
            index_vector(exact_array<IndexInput>(values["dispersive_term_channels"]));
        result.dispersive_coefficients =
            real_vector(exact_array<RealInput>(values["dispersive_coefficients"]));
        result.dispersive_pair_constants =
            real_vector(exact_array<RealInput>(values["dispersive_pair_constants"]));
        result.dispersive_group_offsets =
            index_vector(exact_array<IndexInput>(values["dispersive_group_offsets"]));
        result.dispersive_group_members =
            index_vector(exact_array<IndexInput>(values["dispersive_group_members"]));
        result.dispersive_quartics =
            real_vector(exact_array<RealInput>(values["dispersive_quartics"]));
        result.dispersive_input_scale =
            real_vector(exact_array<RealInput>(values["dispersive_input_scale"]));
        result.dispersive_output_scale =
            real_vector(exact_array<RealInput>(values["dispersive_output_scale"]));
        // The general modal route's group keys and node regions; absent elsewhere.
        if (values.contains("dispersive_group_keys"))
          result.dispersive_group_keys =
              index_vector(exact_array<IndexInput>(values["dispersive_group_keys"]));
        if (values.contains("dispersive_node_region"))
          result.dispersive_node_region =
              index_vector(exact_array<IndexInput>(values["dispersive_node_region"]));
      }
    } catch (const nb::cast_error&) {
      throw std::invalid_argument("dispersive response data have wrong types");
    }
  }
  if (values.contains("general_modal")) {
    try {
      result.general_modal = nb::cast<bool>(values["general_modal"]);
      if (result.general_modal) {
        result.general_real_form = nb::cast<bool>(values["general_real_form"]);
        result.general_map_intervals =
            nb::cast<std::size_t>(values["general_map_intervals"]);
        result.general_map_n_deg = nb::cast<std::size_t>(values["general_map_n_deg"]);
        result.general_components = nb::cast<std::size_t>(values["general_components"]);
        result.general_nodes = nb::cast<std::size_t>(values["general_nodes"]);
        result.general_node_chunk = nb::cast<std::size_t>(values["general_node_chunk"]);
        result.general_responses = nb::cast<std::size_t>(values["general_responses"]);
        result.general_terms = nb::cast<std::size_t>(values["general_terms"]);
        result.general_pad_scale = nb::cast<double>(values["general_pad_scale"]);
        result.general_epsilon0 = nb::cast<double>(values["general_epsilon0"]);
        result.general_synthesis =
            complex_vector(exact_array<ComplexInput>(values["general_synthesis"]));
        result.general_projection =
            complex_vector(exact_array<ComplexInput>(values["general_projection"]));
        result.general_tensor =
            real_vector(exact_array<RealInput>(values["general_tensor"]));
        result.general_frequency_factor = complex_vector(
            exact_array<ComplexInput>(values["general_frequency_factor"]));
        result.general_delayed_tensor =
            real_vector(exact_array<RealInput>(values["general_delayed_tensor"]));
        result.general_delayed_spectrum = complex_vector(
            exact_array<ComplexInput>(values["general_delayed_spectrum"]));
        result.general_node_response =
            index_vector(exact_array<IndexInput>(values["general_node_response"]));
      }
    } catch (const nb::cast_error&) {
      throw std::invalid_argument("general modal data have wrong types");
    }
  }
  if (values.contains("coupled_flow")) {
    try {
      result.coupled_flow = nb::cast<bool>(values["coupled_flow"]);
      if (result.coupled_flow) {
        result.flow_active = nb::cast<std::size_t>(values["flow_active"]);
        result.flow_sections = nb::cast<std::size_t>(values["flow_sections"]);
        result.flow_terms = nb::cast<std::size_t>(values["flow_terms"]);
        result.flow_slot = index_vector(exact_array<IndexInput>(values["flow_slot"]));
        result.flow_frozen =
            complex_vector(exact_array<ComplexInput>(values["flow_frozen"]));
        result.flow_residual_terms =
            complex_vector(exact_array<ComplexInput>(values["flow_residual_terms"]));
        if (values.contains("flow_left")) {
          result.flow_left =
              complex_vector(exact_array<ComplexInput>(values["flow_left"]));
          result.flow_right =
              complex_vector(exact_array<ComplexInput>(values["flow_right"]));
          result.flow_eigenvalues =
              complex_vector(exact_array<ComplexInput>(values["flow_eigenvalues"]));
          result.flow_route =
              index_vector(exact_array<IndexInput>(values["flow_route"]));
          result.flow_fallback_vectors = complex_vector(
              exact_array<ComplexInput>(values["flow_fallback_vectors"]));
          result.flow_fallback_triangular = complex_vector(
              exact_array<ComplexInput>(values["flow_fallback_triangular"]));
        }
      }
    } catch (const nb::cast_error&) {
      throw std::invalid_argument("coupled flow data have wrong types");
    }
  }
  return result;
}

native::TrialScalars trial_from(const RealInput phi_delta, const RealInput phi_at,
                                const RealInput prefactor, const RealInput density,
                                const RealInput taper, const RealInput preionisation,
                                const RealInput raman_rates) {
  return {real_vector(phi_delta),
          real_vector(phi_at),
          real_vector(prefactor),
          real_vector(density),
          real_vector(taper),
          real_vector(preionisation),
          real_vector(raman_rates),
          {},
          {},
          {},
          {},
          0,
          {},
          {},
          {}};
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

// One of solprop's piecewise Chebyshev tables of real values, as solprop's adapter
// packs it: (edges, lower, upper, coefficients) with the coefficients
// (n_intervals, n_terms, n_values); `None` for an absent table (solprop SEP 39).
native::stage::Table stage_table_from(const nb::handle value) {
  native::stage::Table table;
  if (value.is_none())
    return table;
  using Coefficients =
      nb::ndarray<const double, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
  try {
    const auto parts = nb::cast<nb::tuple>(value);
    if (parts.size() != 4)
      throw std::invalid_argument(
          "a stage table is (edges, lower, upper, coefficients)");
    table.edges = real_vector(exact_array<RealInput>(parts[0]));
    table.lower = nb::cast<double>(parts[1]);
    table.upper = nb::cast<double>(parts[2]);
    const auto coefficients = exact_array<Coefficients>(parts[3]);
    if (table.edges.size() < 2 || coefficients.shape(0) + 1 != table.edges.size() ||
        coefficients.shape(1) < 1 || coefficients.shape(2) < 1)
      throw std::invalid_argument(
          "a stage table's coefficients do not match its breakpoints");
    table.n_terms = coefficients.shape(1);
    table.n_values = coefficients.shape(2);
    table.coefficients.assign(coefficients.data(),
                              coefficients.data() + coefficients.size());
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("a stage table has the wrong types");
  }
  return table;
}

std::vector<native::stage::Table> stage_table_list(const nb::handle values) {
  std::vector<native::stage::Table> tables;
  for (const auto value : nb::cast<nb::list>(values))
    tables.push_back(stage_table_from(value));
  return tables;
}

// Every table an exponential Adams trial's stage scalars read (solprop SEP 39 phase 1):
// solprop's adapter packs them once per run from the problem's own leaves.
native::stage::Tables stage_tables_from(const nb::dict values) {
  native::stage::Tables tables;
  try {
    tables.n_stages = nb::cast<std::size_t>(values["n_stages"]);
    tables.phi = static_cast<native::stage::Terms>(nb::cast<int>(values["phi"]));
    tables.phi_weights = stage_table_from(values["phi_weights"]);
    tables.phi_rate = stage_table_from(values["phi_rate"]);
    tables.prefactor =
        static_cast<native::stage::Terms>(nb::cast<int>(values["prefactor"]));
    tables.prefactor_weights = stage_table_from(values["prefactor_weights"]);
    tables.prefactor_low = nb::cast<double>(values["prefactor_low"]);
    tables.prefactor_high = nb::cast<double>(values["prefactor_high"]);
    tables.density = stage_table_from(values["density"]);
    tables.taper = stage_table_from(values["taper"]);
    tables.raman_rates = stage_table_list(values["raman_rates"]);
    tables.kerr =
        static_cast<native::stage::KerrDensity>(nb::cast<int>(values["kerr"]));
    tables.kerr_density = stage_table_from(values["kerr_density"]);
    tables.plasma_densities = stage_table_list(values["plasma_densities"]);
    tables.plasma_preionisation = stage_table_list(values["plasma_preionisation"]);
    tables.raman_densities = stage_table_list(values["raman_densities"]);
    tables.dispersive_densities = stage_table_list(values["dispersive_densities"]);
    const int chord = nb::cast<int>(values["chord"]);
    if (chord < static_cast<int>(native::stage::Chord::none) ||
        chord > static_cast<int>(native::stage::Chord::table))
      throw std::invalid_argument("the stage tables are inconsistent: chord kind " +
                                  std::to_string(chord) + " is unknown");
    tables.chord = static_cast<native::stage::Chord>(chord);
    tables.chord_rate = nb::cast<double>(values["chord_rate"]);
    tables.chord_terms = nb::cast<std::vector<double>>(values["chord_terms"]);
    if (values.contains("chord_table"))
      tables.chord_table = stage_table_from(values["chord_table"]);
    tables.flow.edges = nb::cast<std::vector<double>>(values["flow_edges"]);
    tables.flow.degrees = nb::cast<std::size_t>(values["flow_degrees"]);
    tables.flow.slack = nb::cast<double>(values["flow_slack"]);
    tables.flow_residual = nb::cast<bool>(values["flow_residual"]);
    tables.transform.edges = nb::cast<std::vector<double>>(values["transform_edges"]);
    tables.transform.degrees = nb::cast<std::size_t>(values["transform_degrees"]);
    tables.transform.slack = nb::cast<double>(values["transform_slack"]);
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("the stage tables have the wrong types");
  }
  const bool expansion = tables.phi == native::stage::Terms::expansion;
  const auto known_terms = [](const native::stage::Terms terms) {
    return terms == native::stage::Terms::constant ||
           terms == native::stage::Terms::expansion ||
           terms == native::stage::Terms::table;
  };
  const bool known = known_terms(tables.phi) && known_terms(tables.prefactor);
  if (!known || tables.n_stages == 0 || !tables.density.present() ||
      (expansion && (!tables.phi_weights.present() || !tables.phi_rate.present() ||
                     tables.phi_rate.n_values != tables.phi_weights.n_values)) ||
      (tables.prefactor == native::stage::Terms::expansion &&
       !tables.prefactor_weights.present()) ||
      (tables.kerr == native::stage::KerrDensity::table &&
       !tables.kerr_density.present()) ||
      tables.plasma_densities.size() != tables.plasma_preionisation.size() ||
      (tables.chord == native::stage::Chord::expansion &&
       (!expansion || tables.chord_terms.size() != tables.phi_weights.n_values)) ||
      (tables.chord == native::stage::Chord::table &&
       (!tables.chord_table.present() || tables.chord_table.n_values != 1)) ||
      (tables.flow.present() && tables.flow.edges.size() < 2) ||
      (tables.transform.present() && tables.transform.edges.size() < 2))
    throw std::invalid_argument("the stage tables are inconsistent");
  return tables;
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

// One Adams trial's coefficient tables (solprop SEP 16 phase 3b): the history rows,
// newest first, and the weights over them; `rows` is (n_rows, width), at most three.
native::AdamsWeights adams_weights(const IndexInput slots, const RealInput predictor,
                                   const RealInput corrector, const RealInput rows,
                                   const std::size_t order) {
  native::AdamsWeights weights;
  const auto slot_values = index_vector(slots);
  const auto predictor_values = real_vector(predictor);
  const auto corrector_values = real_vector(corrector);
  const auto row_values = real_vector(rows);
  const std::size_t width = corrector_values.size();
  if (width > native::adams_max_samples || slot_values.size() + 1 < width ||
      predictor_values.size() != order || order > slot_values.size() ||
      (width > 0 && row_values.size() % width != 0) ||
      (width > 0 && row_values.size() / width > 3))
    throw std::invalid_argument("Adams coefficient tables have inconsistent extents");
  for (std::size_t t = 0; t < slot_values.size() && t < native::adams_max_samples; ++t)
    weights.slot[t] = slot_values[t];
  for (std::size_t t = 0; t < order; ++t)
    weights.predictor[t] = predictor_values[t];
  for (std::size_t t = 0; t < width; ++t)
    weights.corrector[t] = corrector_values[t];
  weights.order = order;
  weights.width = width;
  weights.n_rows = width > 0 ? row_values.size() / width : 0;
  for (std::size_t row = 0; row < weights.n_rows; ++row)
    for (std::size_t t = 0; t < width; ++t)
      weights.rows[row][t] = row_values[row * width + t];
  return weights;
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
  storage.release();
  return ComplexOutput(pointer->data(), shape.size(), shape.data(), owner);
}

/** A complex array owning `storage`. */
ComplexOutput owned_complex(std::unique_ptr<ComplexStorage> storage,
                            std::vector<std::size_t> shape) {
  auto* pointer = storage.get();
  nb::capsule owner(pointer, [](void* value) noexcept {
    const std::unique_ptr<ComplexStorage> held(static_cast<ComplexStorage*>(value));
  });
  storage.release();
  return ComplexOutput(pointer->data(), shape.size(), shape.data(), owner);
}

/** A complex array owning `storage`, uninitialised doubles a native copy filled. */
ComplexOutput complex_buffer_output(native::crossing::HostBuffer storage,
                                    std::vector<std::size_t> shape) {
  auto* pointer = storage.release();
  nb::capsule owner(pointer, [](void* value) noexcept {
    const native::crossing::HostBuffer held(static_cast<double*>(value));
  });
  return ComplexOutput(reinterpret_cast<std::complex<double>*>(pointer), shape.size(),
                       shape.data(), owner);
}

RealOutput real_output(std::vector<double> values, const std::size_t rows) {
  if (rows == 0 || values.size() % rows != 0)
    throw std::invalid_argument("native output has an invalid packed shape");
  auto storage = std::make_unique<RealStorage>(std::move(values));
  auto* pointer = storage.get();
  nb::capsule owner(pointer, [](void* value) noexcept {
    const std::unique_ptr<RealStorage> held(static_cast<RealStorage*>(value));
  });
  storage.release();
  const std::array<std::size_t, 2> shape{rows, pointer->size() / rows};
  return RealOutput(pointer->data(), shape.size(), shape.data(), owner);
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
  native::configure_thread_policy(
      {kernels, fft, raman_fft, raman_tail_fft, band_tiles});
}

// ------------------------------------------------------------- the native Adams march
// solprop SEP 39 phase 2: the march's policy, state, bounds and result as solprop's
// adapter passes and reads them, its resident engine over the general state, and a
// scripted engine for the decision-trace tests.

namespace march = native::march;

march::Policy march_policy_from(const nb::dict values) {
  march::Policy policy;
  try {
    policy.order = nb::cast<std::size_t>(values["order"]);
    policy.variable_order = nb::cast<bool>(values["variable_order"]);
    policy.safety = nb::cast<double>(values["safety"]);
    policy.max_growth = nb::cast<double>(values["max_growth"]);
    policy.min_factor = nb::cast<double>(values["min_factor"]);
    policy.rejection_growth = nb::cast<double>(values["rejection_growth"]);
    policy.hysteresis = nb::cast<double>(values["hysteresis"]);
    policy.restart_after = nb::cast<std::size_t>(values["restart_after"]);
    policy.collar_stability = nb::cast<double>(values["collar_stability"]);
    policy.real_intervals = nb::cast<std::vector<double>>(values["real_intervals"]);
    policy.collar_rate = nb::cast<double>(values["collar_rate"]);
    policy.carrier = nb::cast<bool>(values["carrier"]);
    policy.carrier_cap = nb::cast<double>(values["carrier_cap"]);
    policy.has_budget = !values["max_calls"].is_none();
    policy.max_calls =
        policy.has_budget ? nb::cast<std::uint64_t>(values["max_calls"]) : 0;
    policy.equal_approach = nb::cast<bool>(values["equal_approach"]);
    policy.interval_cap = nb::cast<bool>(values["interval_cap"]);
    policy.dz_min = nb::cast<double>(values["dz_min"]);
    policy.dz_max = nb::cast<double>(values["dz_max"]);
    policy.stretch = nb::cast<double>(values["stretch"]);
    policy.z_end = nb::cast<double>(values["z_end"]);
    policy.save_planes = nb::cast<std::vector<double>>(values["save_planes"]);
    policy.hard_kinks = nb::cast<std::vector<double>>(values["hard_kinks"]);
    policy.breakpoints = nb::cast<std::vector<double>>(values["breakpoints"]);
    policy.landing_ulps = nb::cast<int>(values["landing_ulps"]);
    policy.interpolate = nb::cast<bool>(values["interpolate"]);
    policy.has_quiet_interval = !values["quiet_interval"].is_none();
    policy.quiet_interval =
        policy.has_quiet_interval ? nb::cast<double>(values["quiet_interval"]) : 0.0;
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("the Adams march's policy has the wrong types");
  } catch (const nb::python_error&) {
    throw std::invalid_argument("the Adams march's policy lacks a field");
  }
  march::validate(policy);
  return policy;
}

// solprop SEP 45 phase 1: the boundary watch's policy and state, from and to solprop's
// `BoundaryMonitor.watch_policy` and `.watch_state`.
native::watch::Policy watch_policy_from(const nb::dict values) {
  native::watch::Policy policy;
  try {
    policy.split = nb::cast<bool>(values["split"]);
    policy.crossing = nb::cast<bool>(values["crossing"]);
    const auto regions = nb::cast<std::vector<double>>(values["floor_regions"]);
    if (regions.size() != 4)
      throw std::invalid_argument("the boundary watch's floor regions are four");
    for (std::size_t index = 0; index < 4; ++index)
      policy.floor_regions[index] = regions[index];
    policy.launch_floor = nb::cast<double>(values["launch_floor"]);
    policy.activity = nb::cast<double>(values["activity"]);
    policy.warn = nb::cast<double>(values["warn"]);
    policy.stop = nb::cast<double>(values["stop"]);
    policy.quiet_share = nb::cast<double>(values["quiet_share"]);
    policy.quiet_checks = nb::cast<std::uint64_t>(values["quiet_checks"]);
    policy.quiet_collar = nb::cast<bool>(values["quiet_collar"]);
    policy.cadence = nb::cast<std::uint64_t>(values["cadence"]);
    policy.interval = nb::cast<double>(values["interval"]);
    policy.share = nb::cast<double>(values["share"]);
    policy.bucket_limits = nb::cast<std::vector<double>>(values["bucket_limits"]);
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("the boundary watch's policy has the wrong types");
  } catch (const nb::python_error&) {
    throw std::invalid_argument("the boundary watch's policy lacks a field");
  }
  native::watch::validate(policy);
  return policy;
}

native::watch::State watch_state_from(const nb::dict values) {
  native::watch::State state;
  try {
    state.has_last = nb::cast<bool>(values["has_last"]);
    state.last_z = nb::cast<double>(values["last_z"]);
    state.since = nb::cast<std::uint64_t>(values["since"]);
    state.busy = nb::cast<bool>(values["busy"]);
    state.quiet_run = nb::cast<std::uint64_t>(values["quiet_run"]);
    state.quiet = nb::cast<bool>(values["quiet"]);
    state.warned = nb::cast<bool>(values["warned"]);
    state.cap = nb::cast<double>(values["cap"]);
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("the boundary watch's state has the wrong types");
  } catch (const nb::python_error&) {
    throw std::invalid_argument("the boundary watch's state lacks a field");
  }
  return state;
}

nb::dict watch_state_to(const native::watch::State& state) {
  nb::dict out;
  out["has_last"] = state.has_last;
  out["last_z"] = state.last_z;
  out["since"] = state.since;
  out["busy"] = state.busy;
  out["quiet_run"] = state.quiet_run;
  out["quiet"] = state.quiet;
  out["warned"] = state.warned;
  out["cap"] = state.cap;
  return out;
}

// solprop SEP 47 section 6.4: the owning boundary monitor's configuration, the record
// a resume continues and the record it keeps, from and to solprop's dictionaries.
native::watch::Config watch_config_from(const nb::dict values) {
  native::watch::Config config;
  config.policy = watch_policy_from(nb::cast<nb::dict>(values["policy"]));
  try {
    config.floor_share = nb::cast<double>(values["floor_share"]);
    config.floor_margin = nb::cast<double>(values["floor_margin"]);
    config.budget = nb::cast<double>(values["budget"]);
    config.rate = real_vector(exact_array<RealInput>(values["rate"]));
    config.can_rest = nb::cast<bool>(values["can_rest"]);
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("the boundary monitor's configuration has the wrong "
                                "types");
  } catch (const nb::python_error&) {
    throw std::invalid_argument("the boundary monitor's configuration lacks a field");
  }
  return config;
}

std::vector<std::array<double, 2>> watch_stretches_from(const nb::handle values) {
  std::vector<std::array<double, 2>> stretches;
  for (const auto& pair : nb::cast<std::vector<std::vector<double>>>(values)) {
    if (pair.size() != 2)
      throw std::invalid_argument("an unwatched stretch is a start and an end");
    stretches.push_back({pair[0], pair[1]});
  }
  return stretches;
}

native::watch::Resumption watch_resumption_from(const nb::dict values) {
  native::watch::Resumption record;
  const auto number = [&values](const char* name) -> std::optional<double> {
    if (!values.contains(name))
      return std::nullopt;
    return nb::cast<double>(values[name]);
  };
  try {
    record.record = nb::cast<bool>(values["record"]);
    record.launch_energy = number("launch_energy");
    record.launch_guard_low = number("launch_guard_low");
    record.launch_guard_high = number("launch_guard_high");
    record.launch_collar = number("launch_collar");
    record.guard_low_max = number("guard_low_max");
    record.guard_high_max = number("guard_high_max");
    record.guard_low_at = number("guard_low_at");
    record.guard_high_at = number("guard_high_at");
    record.collar_max = number("collar_max");
    record.raw_removed_fraction = number("raw_removed_fraction");
    record.removed_signed = number("removed_signed");
    record.removed_fraction = number("removed_fraction");
    record.floor_removed_fraction = number("floor_removed_fraction");
    record.quiet_checks = number("quiet_checks");
    record.collar_quiet = nb::cast<bool>(values["collar_quiet"]);
    record.raw_guard_max = number("raw_guard_max");
    record.steps = number("steps");
    record.skipped = number("skipped");
    record.collar_left_out_checks = number("collar_left_out_checks");
    if (values.contains("unwatched"))
      record.unwatched = watch_stretches_from(values["unwatched"]);
    if (values.contains("resumed_at"))
      record.resumed_at = nb::cast<std::vector<double>>(values["resumed_at"]);
    if (values.contains("planes")) {
      const auto rows = nb::cast<std::vector<std::vector<double>>>(values["planes"]);
      if (rows.size() != native::watch::plane_columns)
        throw std::invalid_argument("a monitor record's plane rows have five columns");
      record.has_planes = true;
      for (std::size_t column = 0; column < native::watch::plane_columns; ++column)
        record.planes[column] = rows[column];
    }
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("a resumed boundary record has the wrong types");
  } catch (const nb::python_error&) {
    throw std::invalid_argument("a resumed boundary record lacks a field");
  }
  return record;
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

native::session::MonitorPlan monitor_plan_from(const nb::dict values) {
  native::session::MonitorPlan plan;
  plan.config = watch_config_from(nb::cast<nb::dict>(values["config"]));
  try {
    plan.resume = nb::cast<bool>(values["resume"]);
    if (plan.resume) {
      plan.record = watch_resumption_from(nb::cast<nb::dict>(values["record"]));
      plan.plane = nb::cast<bool>(values["plane"]);
      plan.unwatched = watch_stretches_from(values["unwatched"]);
      if (values.contains("edge_record"))
        plan.edge_record = native::bindings::transverse_resumption_from(
            nb::cast<nb::dict>(values["edge_record"]));
    }
    // The input noise's floor: its energy, then its four regions' shares.
    if (values.contains("floor") && !values["floor"].is_none()) {
      const auto floor = nb::cast<std::vector<double>>(values["floor"]);
      if (floor.size() != 5)
        throw std::invalid_argument("a monitor's floor is its energy and four regions");
      plan.floor = true;
      plan.floor_energy = floor[0];
      std::copy(floor.begin() + 1, floor.end(), plan.floor_regions.begin());
    }
    // A free-space run's transverse edge (SEP 40 section 5.3), owned alike.
    if (values.contains("edge") && !values["edge"].is_none())
      plan.edge =
          native::bindings::transverse_config_from(nb::cast<nb::dict>(values["edge"]));
    if (values.contains("edge_noise") && !values["edge_noise"].is_none())
      plan.edge_noise = native::bindings::transverse_noise_from(
          nb::cast<nb::dict>(values["edge_noise"]));
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("a session's boundary monitor has the wrong types");
  } catch (const nb::python_error&) {
    throw std::invalid_argument("a session's boundary monitor lacks a field");
  }
  return plan;
}

/** Python's `format(value, f".{precision}g")`, which follows C's `%g`. */
std::string py_g(const double value, const int precision) {
  std::array<char, 64> buffer{};
  std::snprintf(buffer.data(), buffer.size(), "%.*g", precision, value);
  return buffer.data();
}

// solprop SEP 45 phase 3: the Runge-Kutta march's policy and state, from and to
// solprop's `rk_march_policy` and `_NativeRungeKuttaMarch._state`.
native::rk::Policy rk_policy_from(const nb::dict values) {
  native::rk::Policy policy;
  try {
    auto& landings = policy.landings;
    landings.dz_min = nb::cast<double>(values["dz_min"]);
    landings.dz_max = nb::cast<double>(values["dz_max"]);
    landings.stretch = nb::cast<double>(values["stretch"]);
    landings.z_end = nb::cast<double>(values["z_end"]);
    landings.save_planes = nb::cast<std::vector<double>>(values["save_planes"]);
    landings.hard_kinks = nb::cast<std::vector<double>>(values["hard_kinks"]);
    landings.breakpoints = nb::cast<std::vector<double>>(values["breakpoints"]);
    landings.landing_ulps = nb::cast<int>(values["landing_ulps"]);
    landings.interpolate = nb::cast<bool>(values["interpolate"]);
    landings.has_quiet_interval = !values["quiet_interval"].is_none();
    landings.quiet_interval =
        landings.has_quiet_interval ? nb::cast<double>(values["quiet_interval"]) : 0.0;
    policy.safety = nb::cast<double>(values["safety"]);
    policy.beta1 = nb::cast<double>(values["beta1"]);
    policy.beta2 = nb::cast<double>(values["beta2"]);
    policy.min_factor = nb::cast<double>(values["min_factor"]);
    policy.max_factor = nb::cast<double>(values["max_factor"]);
    policy.lag = nb::cast<std::size_t>(values["lag"]);
    policy.reject_free = nb::cast<bool>(values["reject_free"]);
    policy.power = nb::cast<double>(values["power"]);
    policy.exponent = nb::cast<double>(values["exponent"]);
    policy.start_growth = nb::cast<double>(values["start_growth"]);
    policy.auto_init = nb::cast<bool>(values["auto_init"]);
    policy.dz_min = landings.dz_min;
    policy.dz_max = landings.dz_max;
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("the Runge-Kutta march's policy has the wrong types");
  } catch (const nb::python_error&) {
    throw std::invalid_argument("the Runge-Kutta march's policy lacks a field");
  }
  if (policy.landings.breakpoints.size() < 2 || !(policy.dz_max >= policy.dz_min))
    throw std::invalid_argument("the Runge-Kutta march's landings are invalid");
  return policy;
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
    for (const auto entry : nb::cast<nb::list>(values["history"])) {
      const auto parts = nb::cast<nb::tuple>(entry);
      state.history.push_back({nb::cast<double>(parts[0]),
                               nb::cast<bool>(parts[1]) ? march::Identity::predicted
                                                        : march::Identity::accepted,
                               nb::cast<std::size_t>(parts[2])});
    }
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
    history.append(nb::make_tuple(entry.z, entry.identity == march::Identity::predicted,
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
        carrier_rate_(carrier_rate), schedule_{real_vector(steps), real_vector(edges)} {
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
  nb::list log() const {
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
  ScriptedRkMarch(const nb::dict policy, const RealInput norms, const double h0,
                  const RealInput steps, const RealInput edges)
      : policy_(rk_policy_from(policy)), norms_(real_vector(norms)), h0_(h0),
        schedule_{real_vector(steps), real_vector(edges)} {}
  nb::dict run(const nb::dict state_values, const nb::dict bounds_values) {
    auto state = rk_state_from(state_values);
    const auto bounds = march_bounds_from(bounds_values);
    native::engines::ScriptedRungeKutta engine(norms_, next_, h0_, policy_.power, log_);
    const auto result = native::rk::run(policy_, state, engine, bounds, schedule_);
    return rk_result_to(result, state);
  }
  // Each trial's position and step, one dict a trial.
  nb::list log() const {
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

class BoundState {
public:
  explicit BoundState(const nb::dict& problem) : BoundState(problem_from(problem)) {}
  explicit BoundState(native::Problem problem)
      : columns_(problem.n_columns()), elements_(problem.n_columns() * problem.n_omega),
        stages_(problem.n_stages), raman_terms_(problem.n_raman_terms),
        raman_(problem.raman) {
    general_state_ = std::make_shared<native::GeneralState>(std::move(problem));
    const std::scoped_lock lock(runtime_state.mutex);
    runtime_state.states.push_back(this);
  }
  ~BoundState() {
    close();
    const std::scoped_lock lock(runtime_state.mutex);
    const auto found =
        std::find(runtime_state.states.begin(), runtime_state.states.end(), this);
    if (found != runtime_state.states.end())
      runtime_state.states.erase(found);
  }
  BoundState(const BoundState&) = delete;
  BoundState& operator=(const BoundState&) = delete;

  void set_initial(const std::span<const native::Complex> field,
                   const std::span<const native::Complex> noise) {
    require_unleased();
    require_open();
    general_state_->set_initial(field, noise);
  }
  // solprop SEP 45 phase 2: a Runge-Kutta trial whose stage rows the state forms from
  // the uploaded tables, at the positions `z + c_i dz` and the step's end.
  std::vector<double> rk_trial(const double z, const double dz,
                               const std::vector<double>& nodes,
                               const bool prepare_only) {
    require_open();
    return staged_trial(z, dz, nodes, prepare_only);
  }
  // The trial itself, which a session's engine calls on the state it leases.
  std::vector<double> staged_trial(const double z, const double dz,
                                   const std::vector<double>& nodes,
                                   const bool prepare_only) {
    if (!stage_tables_)
      throw std::logic_error("a staged Runge-Kutta trial needs the stage tables");
    if (nodes.size() != stages_)
      throw std::invalid_argument("the trial's nodes are the tableau's stages");
    const auto scalars = native::stage::rows(*stage_tables_, stage_anchor_, z,
                                             native::rk::positions(z, dz, nodes));
    if (stage_tables_->flow.present())
      guarded_flow_actions(scalars.flow_section, dz, nodes);
    auto& general = built_state();
    if (prepare_only) {
      general.prepare_trial(dz, scalars);
      return {};
    }
    return general.trial_step(dz, scalars);
  }
  // solprop SEP 45 phase 4: `_flow_trial`'s guards, transcribed. A set of actions is
  // held for one (section, step); a length whose exponent needs too many squarings is
  // refused before forming, and a measured conditioning above the section's
  // certificate after, each as solprop's `_flow_squarings` and
  // `_check_flow_conditioning` refuse them, with their constants handed over.
  void flow_guard_configure(std::vector<double> row_sum,
                            std::vector<double> condition_bound,
                            const double taylor_scale, const int max_squarings,
                            const double allowance) {
    open_state();
    const std::size_t sections = row_sum.size();
    if (sections == 0 || condition_bound.size() % sections != 0)
      throw std::invalid_argument("the flow guard's certificate is one row a section");
    flow_guard_ = FlowGuard{std::move(row_sum), std::move(condition_bound),
                            taylor_scale, max_squarings, allowance};
    flow_key_.reset();
  }
  void guarded_flow_actions(const std::size_t section, const double dz,
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
        squarings =
            static_cast<int>(std::ceil(std::log2(largest / guard.taylor_scale)));
      if (squarings > guard.max_squarings)
        throw native::BackendRefusal(
            "native cannot take a coupled flow action whose exponent has row-sum "
            "norm " +
            py_g(largest, 4) + ": it would need " + std::to_string(squarings) +
            " squarings, above the " + std::to_string(guard.max_squarings) +
            " this backend admits. Cap dz_max");
    }
    const auto measured = general_state_->form_flow_actions(section, dz, fractions);
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
        throw native::BackendRefusal(
            "native measured a coupled flow conditioning of " +
            py_g(measured[worst], 12) + " over a step of " + py_g(dz, 4) +
            " m, above the " + py_g(declared[worst], 12) +
            " the certificate declares at active bin " + std::to_string(worst) +
            " of section " + std::to_string(section) + " -- exceeding it by a factor " +
            py_g(largest, 12) +
            " of the rounding allowance: the inverse action carries the nonlinear "
            "source back by the measured factor. Lower dz_max, which is what the "
            "bound was declared over");
    }
    flow_key_ = std::make_pair(section, dz);
  }
  // solprop SEP 45 phase 3: the native Runge-Kutta march over the staged trial.
  void rk_march_configure(native::rk::Policy policy, std::vector<double> nodes,
                          const double rtol, const double atol,
                          std::vector<double> steps, std::vector<double> edges) {
    open_state();
    if (!stage_tables_)
      throw std::logic_error("the native Runge-Kutta march needs the stage tables");
    if (nodes.size() != stages_)
      throw std::invalid_argument("the march's nodes are the tableau's stages");
    rk_policy_ = std::move(policy);
    rk_nodes_ = std::move(nodes);
    rk_rtol_ = rtol;
    rk_atol_ = atol;
    rk_schedule_ = {std::move(steps), std::move(edges)};
  }
  nb::dict rk_march_run(const nb::dict state_values, const nb::dict bounds_values) {
    require_unleased();
    if (!rk_policy_)
      throw std::logic_error("the native Runge-Kutta march is not configured");
    auto state = rk_state_from(state_values);
    const auto bounds = march_bounds_from(bounds_values);
    auto& general = open_state();
    native::engines::ResidentRungeKutta engine(
        general,
        [this](const double z, const double dz) {
          return rk_trial(z, dz, rk_nodes_, false);
        },
        rk_rtol_, rk_atol_, rk_weights_);
    native::rk::Result result;
    {
      const nb::gil_scoped_release release;
      result = native::rk::run(*rk_policy_, state, engine, bounds, rk_schedule_);
    }
    return rk_result_to(result, state);
  }
  std::vector<double> trial_step(const double dz, const native::TrialScalars& scalars,
                                 const bool prepare_only = false) {
    require_open();
    if (prepare_only) {
      general_state_->prepare_trial(dz, scalars);
      return {};
    }
    return general_state_->trial_step(dz, scalars);
  }
  void set_flow_actions(const std::vector<native::Complex>& forward,
                        const std::vector<native::Complex>& inverse,
                        const std::vector<unsigned char>& identity) {
    require_open();
    flow_key_.reset();
    general_state_->set_flow_actions(forward, inverse, identity);
  }
  std::vector<native::Complex> flow_actions_to_host(const bool inverse) {
    require_open();
    return general_state_->flow_actions_to_host(inverse);
  }
  std::vector<native::Complex> adams_block_to_host() {
    require_open();
    return general_state_->adams_block_to_host();
  }
  std::vector<native::Complex> adams_exponent_to_host() {
    require_open();
    return general_state_->adams_exponent_to_host();
  }
  std::vector<double> form_flow_actions(const std::size_t section, const double dz,
                                        const std::vector<double>& fractions) {
    require_open();
    flow_key_.reset();
    return general_state_->form_flow_actions(section, dz, fractions);
  }
  std::vector<double> weighted_error(const std::vector<double>& scale) {
    require_open();
    return general_state_->weighted_error(scale);
  }
  // solprop SEP 47 section 6.3: the low-storage replay's flows and operator origin,
  // which a session's low-storage engine reads beside the Runge-Kutta policy.
  void low_storage_configure(const std::size_t flows, const bool separable,
                             const double origin) {
    open_state();
    if (flows > 2)
      throw std::invalid_argument("the low-storage march holds at most two flows");
    low_storage_ = native::rk::low_storage::Config{flows, separable, origin};
  }
  // solprop SEP 45 phase 4: a weighted controller's weights, and its norm from a
  // trial's partials, for both Runge-Kutta loops.
  void rk_weights_configure(std::vector<double> weights) {
    open_state();
    rk_weights_ = std::move(weights);
  }
  double rk_weighted_score(const double* partials, const std::size_t slots,
                           const double rtol, const double atol) {
    if (rk_weights_.empty())
      throw std::logic_error("no weighted controller's weights were configured");
    const std::vector<double> values(partials, partials + 3 * slots);
    return native::engines::ResidentRungeKutta::weighted_norm(open_state(), values,
                                                              rtol, atol, rk_weights_);
  }
  void prepare_raman(const std::vector<double>& rates,
                     const std::vector<double>& densities) {
    require_open();
    general_state_->prepare_raman(rates, densities);
  }
  std::vector<double> statistics(const std::uint64_t flags,
                                 const std::vector<double>& windows,
                                 const std::vector<double>& on_axis,
                                 const double density, const double taper,
                                 const double preionisation,
                                 const double intensity_factor,
                                 const std::vector<double>& channel_preionisation,
                                 const std::vector<double>& power_weight) {
    require_open();
    return general_state_->statistics(flags, windows, on_axis, density, taper,
                                      preionisation, intensity_factor,
                                      channel_preionisation, power_weight);
  }
  std::vector<double> spatial_statistics(const std::uint64_t flags,
                                         const double intensity_factor,
                                         const double z) {
    require_open();
    return general_state_->spatial_statistics(flags, intensity_factor, z);
  }
  std::vector<double> plane_fluence_trace(const double center_x, const double center_y,
                                          const double dx, const double dy,
                                          const double extent, const double spacing,
                                          const std::size_t n_radii,
                                          const std::vector<double>& cosines,
                                          const std::vector<double>& sines) {
    require_open();
    return general_state_->plane_fluence_trace(center_x, center_y, dx, dy, extent,
                                               spacing, n_radii, cosines, sines);
  }
  std::vector<native::Complex> axis_spectrum(const std::vector<double>& matrix) {
    require_open();
    return general_state_->axis_spectrum(matrix);
  }
  std::vector<double> aperture_statistics(const std::uint64_t flags,
                                          const std::vector<double>& apertures,
                                          const int exponent,
                                          const std::vector<double>& regions,
                                          const double intensity_factor) {
    require_open();
    return general_state_->aperture_statistics(flags, apertures, exponent, regions,
                                               intensity_factor);
  }
  std::vector<double>
  planned_statistics(const std::shared_ptr<const native::statistics::Plan>& plan,
                     const double z) {
    require_open();
    return native::statistics::evaluate(*general_state_, *plan, z);
  }
  void bind_edge(const std::vector<double>& faces, const std::vector<double>& clip,
                 const std::vector<double>& totals,
                 const std::vector<double>& synthesis, const std::size_t first_row) {
    open_state().bind_edge(faces, clip, totals, synthesis, first_row);
  }
  native::CrossedBytes crossed_bytes() { return built_state().crossed_bytes(); }
  void bind_edge_lines(const native::EdgeLines& lines) {
    open_state().bind_edge_lines(lines);
  }
  native::EdgeNumbers edge_reading(const bool faces, const bool whole,
                                   const bool totals, const bool strip,
                                   const bool current) {
    return open_state().edge_reading(faces, whole, totals, strip, current);
  }
  // Not `require_open`: this stays a no-op on a closed state, whose pointer is null.
  void set_crossing_workers(const std::size_t workers) noexcept {
    if (general_state_)
      general_state_->set_crossing_workers(workers);
  }
  void field_to_host(const std::span<native::Complex> destination) {
    require_open();
    general_state_->field_to_host(destination);
  }
  /** The state's element count, `columns * n_omega`. */
  std::size_t elements() const noexcept { return elements_; }
  std::vector<native::Complex> adams_source(const std::vector<native::Complex>& input,
                                            const native::TrialScalars& scalars) {
    require_open();
    return general_state_->adams_source(input, scalars);
  }
  void adopt(const std::vector<native::Complex>& field, const double dz) {
    require_unleased();
    require_open();
    general_state_->adopt(field, dz);
  }
  native::GeneralState& open_state() {
    require_open();
    return *general_state_;
  }
  // The stage tables an exponential Adams trial reads natively (solprop SEP 39).
  void set_stage_tables(native::stage::Tables tables) {
    open_state();
    if (tables.n_stages != stages_)
      throw std::invalid_argument("the stage tables name another tableau's stages");
    // A z table's rows are its slots: the tables read the state's one cache.
    const bool phi_table = tables.phi == native::stage::Terms::table;
    const bool prefactor_table = tables.prefactor == native::stage::Terms::table;
    if (phi_table != z_slots_.phi || prefactor_table != z_slots_.prefactor)
      throw std::invalid_argument("the stage tables and the z table's slots disagree");
    if (phi_table || prefactor_table) {
      tables.z_slots = &z_slots_;
      tables.z_state = &open_state();
    }
    stage_tables_ = std::move(tables);
    stage_anchor_ = {};
  }
  // solprop SEP 39 phase 4: the z table's slots, one cache for every consumer.
  void table_slots(const bool phi, const bool prefactor, const std::size_t slots,
                   const bool rate) {
    open_state();
    z_slots_.configure(phi, prefactor, slots, rate);
  }
  std::vector<std::size_t> table_load(const std::vector<double>& positions) {
    if (!z_slots_.present())
      throw std::logic_error("no z table's slots were configured");
    return z_slots_.load(open_state(), positions);
  }
  void table_rate_at(const double z, const bool before) {
    if (!z_slots_.has_rate)
      throw std::logic_error("the z table reads no rate");
    z_slots_.rate(open_state(), z, before);
  }
  [[nodiscard]] std::uint64_t table_evaluations() const { return z_slots_.evaluations; }
  [[nodiscard]] bool has_stage_tables() const { return stage_tables_.has_value(); }
  native::stage::Trial stage_trial(const double z0, const double z1) {
    require_open();
    if (!stage_tables_)
      throw std::logic_error("no stage tables were uploaded");
    const auto row = native::stage::row(*stage_tables_, stage_anchor_, z0, z1);
    auto step = native::stage::trial(*stage_tables_, row);
    auto sample = native::stage::sample(*stage_tables_, stage_anchor_, step, z1);
    return {std::move(step), std::move(sample)};
  }
  std::vector<double> stage_rate(const double z) const {
    require_open();
    if (!stage_tables_)
      throw std::logic_error("no stage tables were uploaded");
    return native::stage::rate(*stage_tables_, z);
  }
  // solprop SEP 39 phase 2: the native march over the resident engine.
  void march_configure(march::Policy policy, const bool fused, const bool compensated,
                       const bool interpolate, march::Schedule schedule) {
    open_state();
    if (!stage_tables_)
      throw std::logic_error("the native Adams march needs the stage tables");
    if (!schedule.steps.empty() && schedule.edges.size() != schedule.steps.size() + 1)
      throw std::invalid_argument("a replayed schedule has one more edge than steps");
    march_policy_ = std::move(policy);
    march_schedule_ = std::move(schedule);
    march_fused_ = fused;
    march_compensated_ = compensated;
    march_interpolate_ = interpolate;
  }
  nb::dict march_run(const nb::dict state_values, const nb::dict bounds_values) {
    require_unleased();
    if (!march_policy_)
      throw std::logic_error("the native Adams march is not configured");
    auto state = march_state_from(state_values);
    const auto bounds = march_bounds_from(bounds_values);
    auto& general = open_state();
    native::engines::ResidentEngine engine(general, *stage_tables_, stage_anchor_,
                                           march_fused_, march_compensated_,
                                           march_interpolate_, march_fault_);
    march::Result result;
    {
      const nb::gil_scoped_release release;
      result = march::run(*march_policy_, state, engine, bounds, march_schedule_);
    }
    return march_result_to(result, state);
  }
  void march_fault(const std::string& stage, const std::size_t after,
                   const std::string& category) {
    require_open();
    native::engines::MarchFault fault;
    if (stage == "startup")
      fault.stage = march::Stage::startup;
    else if (stage == "trial")
      fault.stage = march::Stage::trial;
    else if (stage == "accept")
      fault.stage = march::Stage::accept;
    else
      throw std::invalid_argument("a march fault's stage is startup, trial or accept");
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
    march_fault_ = fault;
  }
  void stage_to_host(const std::size_t stage,
                     const std::span<native::Complex> destination) {
    require_open();
    general_state_->stage_to_host(stage, destination);
  }
  std::vector<native::Complex> projected_to_host(const bool coarse) {
    require_open();
    return general_state_->projected_to_host(coarse);
  }
  // Not `require_open`: a closed state, whose pointer is null, reads as having none.
  std::map<std::string, bool> dispersive_path() const {
    return general_state_ ? general_state_->dispersive_path()
                          : std::map<std::string, bool>{};
  }
  void finish(const double dz) {
    require_unleased();
    require_open();
    general_state_->finish(dz);
  }
  void promote_fsal() {
    require_open();
    general_state_->promote_fsal();
  }
  void clear_fsal() {
    require_open();
    general_state_->clear_fsal();
  }
  // The engine leaves a quiet collar out where the reference does (solprop SEP 21).
  void set_absorber_quiet(const bool quiet) {
    require_open();
    general_state_->set_absorber_quiet(quiet);
  }
  void bind_monitor(const std::vector<double>& weights,
                    const std::vector<double>& column_weights) {
    require_open();
    general_state_->bind_monitor(weights, column_weights);
  }
  void unbind_monitor() {
    require_open();
    general_state_->unbind_monitor();
  }
  void bind_occupancy(const std::vector<unsigned char>& buckets,
                      const std::size_t count) {
    require_open();
    general_state_->bind_occupancy(buckets, count);
  }
  std::vector<double> monitor_reading() {
    require_open();
    return general_state_->monitor_reading();
  }
  std::map<std::string, std::size_t> memory_ledger() {
    require_built();
    return general_state_->memory_ledger();
  }
  native::CpuSizing execution_sizing() const {
    require_built();
    return general_state_->execution_sizing();
  }
  nb::dict cpu_execution() const {
    require_built();
    return cpu_execution_dict(general_state_->cpu_execution());
  }
  nb::dict fft_planning() const {
    require_built();
    return fft_planning_dict(general_state_->fft_planning());
  }
  double planning_seconds() const {
    require_built();
    return general_state_->planning_seconds();
  }
  double raman_planning_seconds() const {
    require_built();
    return general_state_->raman_planning_seconds();
  }
  std::string raman_fft_provider() const {
    require_built();
    return general_state_->raman_fft_provider();
  }
  std::map<std::string, int> raman_fft_tasks() const {
    require_built();
    return general_state_->raman_fft_tasks();
  }
  std::size_t raman_kernel_transforms() const {
    require_built();
    return general_state_->raman_kernel_transforms();
  }
  bool fsal_ready() const {
    require_built();
    return general_state_->fsal_ready();
  }
  std::size_t rk_source_evaluations() const {
    require_built();
    return general_state_->rk_source_evaluations();
  }
  double quadrature_error() const {
    require_open();
    return general_state_->quadrature_error();
  }
  double quadrature_peak() {
    require_open();
    return general_state_->quadrature_peak();
  }
  std::string synthesis_provider() const {
    require_built();
    return general_state_->synthesis_provider();
  }
  std::string projection_provider() const {
    require_built();
    return general_state_->projection_provider();
  }
  std::string gemm_shape_policy() const {
    require_built();
    return general_state_->gemm_shape_policy();
  }
  std::size_t response_team_size() const {
    require_built();
    return general_state_->response_team_size();
  }
  std::shared_ptr<native::GeneralState> batch_member() {
    require_open();
    return general_state_;
  }
  std::size_t columns() const noexcept { return columns_; }
  std::size_t stages() const noexcept { return stages_; }
  std::size_t raman_terms() const noexcept { return raman_terms_; }
  bool raman() const noexcept { return raman_; }
  bool closed() const noexcept { return !general_state_ || general_state_->closed(); }
  // Closes a session that leases this state first (below, after its class).
  void close() noexcept;

  // solprop SEP 47: the march a native execution session runs, from the policy, the
  // schedule and the stage tables this state was configured with, its engine over the
  // state and its tables; `state` is the march's state at the session's start.
  // `dense`, a session's dense output (solprop SEP 47 section 5.3), is none or
  // `{"weights": float64 array}`: a Runge-Kutta tableau's dense-output weights, empty
  // for the Adams march.
  native::session::Method session_method(const std::string& kind,
                                         const nb::dict state_values,
                                         const nb::handle dense) {
    require_unleased();
    auto& general = open_state();
    const bool interpolated = !dense.is_none();
    if (kind == "adams") {
      if (!march_policy_ || !stage_tables_)
        throw std::logic_error("the native Adams march is not configured");
      native::session::AdamsMethod method;
      method.policy = *march_policy_;
      method.state = march_state_from(state_values);
      method.schedule = march_schedule_;
      method.engine = std::make_unique<native::engines::ResidentEngine>(
          general, *stage_tables_, stage_anchor_, march_fused_, march_compensated_,
          march_interpolate_, march_fault_);
      if (interpolated) {
        if (!march_interpolate_)
          throw std::invalid_argument(
              "a session's dense output needs an Adams engine that keeps its steps");
        method.dense = std::make_unique<native::engines::AdamsDense>(
            general, *stage_tables_, stage_anchor_, march_fused_, march_compensated_);
      }
      return method;
    }
    if (kind == "runge_kutta") {
      if (!rk_policy_)
        throw std::logic_error("the native Runge-Kutta march is not configured");
      native::session::RungeKuttaMethod method;
      method.policy = *rk_policy_;
      method.state = rk_state_from(state_values);
      method.schedule = rk_schedule_;
      auto engine = std::make_unique<native::engines::ResidentRungeKutta>(
          general,
          [this](const double z, const double dz) {
            return staged_trial(z, dz, rk_nodes_, false);
          },
          rk_rtol_, rk_atol_, rk_weights_);
      if (interpolated) {
        if (!stage_tables_)
          throw std::logic_error("a session's dense output needs the stage tables");
        auto holder = std::make_unique<native::engines::RungeKuttaDense>(
            general, *stage_tables_, stage_anchor_,
            real_vector(exact_array<RealInput>(nb::cast<nb::dict>(dense)["weights"])));
        // The engine holds each crossing step's extension through it; both live in
        // the method, the engine released first.
        auto* held = holder.get();
        engine->set_extension(
            [held](const double z, const double dz) { held->hold(z, dz); });
        method.dense = std::move(holder);
      }
      method.engine = std::move(engine);
      return method;
    }
    if (kind == "low_storage") {
      // solprop SEP 47 section 6.3: the replay is the Runge-Kutta march's, each step
      // one in-place low-storage step whose scalars the state forms itself.
      if (!rk_policy_ || !stage_tables_ || !low_storage_)
        throw std::logic_error("the native low-storage replay is not configured");
      native::session::RungeKuttaMethod method;
      method.policy = *rk_policy_;
      method.state = rk_state_from(state_values);
      method.schedule = rk_schedule_;
      method.engine = std::make_unique<native::engines::ResidentLowStorage>(
          general, *stage_tables_, stage_anchor_, *low_storage_, rk_schedule_,
          rk_policy_->landings.landing_ulps, march_fault_);
      return method;
    }
    throw std::invalid_argument(
        "a session's march is \"adams\", \"runge_kutta\" or \"low_storage\"");
  }
  // The state's field and statistics, as a session reads them.
  std::unique_ptr<native::session::Resident> session_resident() {
    return std::make_unique<native::engines::GeneralResident>(open_state(), elements_,
                                                              elements_ / columns_);
  }
  // The state `session` leases, for that session's own reads under its lock.
  native::GeneralState& leased_state(const BoundSession* session) const {
    if (session_ != session)
      throw std::logic_error("native state is not leased by this session");
    return built_state();
  }
  // A z table's `Phi` at `z` (`_ZTables.values`), for the session that leases the
  // state, under its lock: the noise phase at a plane, through the one slot cache.
  std::vector<native::Complex> leased_table_phi(const BoundSession* session,
                                                const double z) {
    auto& state = leased_state(session);
    if (!z_slots_.phi)
      throw std::logic_error("native state holds no z table of Phi");
    const std::size_t slot = z_slots_.load(state, {z}).front();
    return state.table_terms_to_host(native::TableKind::phi, slot);
  }
  // A session leases the state exclusively: the host's own march and loads refuse.
  void lease(BoundSession* session) {
    require_unleased();
    session_ = session;
  }
  void release(const BoundSession* session) noexcept {
    if (session_ == session)
      session_ = nullptr;
  }

private:
  // Every method that mutates the state or reads its live field or workspace: open,
  // and not leased by an execution session (SEP 47 section 5.2), whose engines alone
  // use the state while it holds it.
  void require_open() const {
    require_built();
    require_unleased();
  }
  // A reader of the state's fixed facts (its plans, execution record, ledger and
  // providers), which a lease leaves readable.
  void require_built() const {
    if (closed())
      throw std::logic_error("native state is closed");
  }
  native::GeneralState& built_state() const {
    require_built();
    return *general_state_;
  }
  void require_unleased() const {
    if (session_ != nullptr)
      throw std::logic_error("native state is leased by an execution session");
  }
  BoundSession* session_ = nullptr;
  std::size_t columns_{};
  std::size_t elements_{};
  std::size_t stages_{};
  std::size_t raman_terms_{};
  bool raman_{};
  std::shared_ptr<native::GeneralState> general_state_;
  std::optional<native::stage::Tables> stage_tables_;
  native::stage::Anchor stage_anchor_;
  native::stage::ZSlots z_slots_;
  std::optional<march::Policy> march_policy_;
  struct FlowGuard {
    std::vector<double> row_sum;
    std::vector<double> condition_bound;
    double taylor_scale;
    int max_squarings;
    double allowance;
  };
  std::optional<FlowGuard> flow_guard_;
  std::optional<std::pair<std::size_t, double>> flow_key_;
  std::optional<native::rk::Policy> rk_policy_;
  std::vector<double> rk_nodes_;
  double rk_rtol_ = 0.0;
  double rk_atol_ = 0.0;
  march::Schedule rk_schedule_;
  std::vector<double> rk_weights_;
  std::optional<native::rk::low_storage::Config> low_storage_;
  march::Schedule march_schedule_;
  bool march_fused_ = false;
  bool march_compensated_ = false;
  bool march_interpolate_ = false;
  std::optional<native::engines::MarchFault> march_fault_;
};

class BoundBatch {
public:
  explicit BoundBatch(const nb::list& members) : count_(members.size()) {
    std::vector<std::shared_ptr<native::GeneralState>> held;
    held.reserve(count_);
    for (const auto& member : members)
      held.push_back(nb::cast<BoundState&>(member).batch_member());
    batch_ = std::make_unique<native::GeneralBatch>(std::move(held));
    const std::scoped_lock lock(runtime_state.mutex);
    runtime_state.batches.push_back(this);
  }
  ~BoundBatch() {
    close();
    const std::scoped_lock lock(runtime_state.mutex);
    const auto found =
        std::find(runtime_state.batches.begin(), runtime_state.batches.end(), this);
    if (found != runtime_state.batches.end())
      runtime_state.batches.erase(found);
  }
  BoundBatch(const BoundBatch&) = delete;
  BoundBatch& operator=(const BoundBatch&) = delete;
  std::vector<double> accept(const double dz) {
    if (!batch_)
      throw std::logic_error("native batch is closed");
    return batch_->accept(dz);
  }
  std::map<std::string, std::size_t> memory_ledger() const {
    return batch_ ? batch_->memory_ledger() : std::map<std::string, std::size_t>{};
  }
  native::CpuSizing execution_sizing() const {
    return batch_ ? batch_->execution_sizing() : native::CpuSizing{};
  }
  std::size_t count() const noexcept { return count_; }
  void close() noexcept {
    if (batch_) {
      batch_->close();
      batch_.reset();
    }
  }

private:
  std::size_t count_{};
  std::unique_ptr<native::GeneralBatch> batch_;
};

// solprop SEP 47: a native execution session over a leased state. It keeps that state
// alive (the binding's keep-alive) and releases the lease when it closes; the state
// closes the session first when it closes itself.
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
  storage.release();
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

class BoundSession {
public:
  BoundSession(BoundState& owner, std::unique_ptr<native::session::Session> session)
      : owner_(&owner), session_(std::move(session)) {
    owner.lease(this);
  }
  ~BoundSession() { close(); }
  BoundSession(const BoundSession&) = delete;
  BoundSession& operator=(const BoundSession&) = delete;

  // The advance runs without the interpreter lock; its batch becomes a record with it.
  BatchRecord advance(const std::size_t work_limit, const double deadline) {
    native::session::Batch batch;
    {
      const nb::gil_scoped_release release;
      batch = session_->advance(work_limit, deadline);
    }
    return BatchRecord(std::move(batch));
  }
  // Takes the session's lock: bound with the interpreter lock released.
  void acknowledge(const std::uint64_t sequence) { session_->acknowledge(sequence); }
  void
  export_field(const std::uint64_t sequence, const std::uint64_t generation,
               const bool power, const bool optics,
               const std::optional<RealInput>& weight,
               nb::ndarray<std::complex<double>, nb::c_contig, nb::device::cpu> out) {
    const std::size_t n = session_->elements();
    if (out.size() != n)
      throw std::invalid_argument("an export destination has " +
                                  std::to_string(out.size()) + " elements, expected " +
                                  std::to_string(n));
    if (weight && weight->size() != n)
      throw std::invalid_argument("an export's power weight has " +
                                  std::to_string(weight->size()) +
                                  " elements, expected " + std::to_string(n));
    const double* values = weight ? weight->data() : nullptr;
    auto* destination = reinterpret_cast<double*>(out.data());
    const nb::gil_scoped_release release;
    session_->export_field(sequence, generation, power, optics, values, destination);
  }
  void request_stop() noexcept { session_->request_stop(); }
  nb::tuple request_snapshot() {
    const auto receipt = session_->request_snapshot();
    return nb::make_tuple(receipt.request, receipt.coalesced);
  }
  native::session::Status status() const noexcept { return session_->status(); }
  std::size_t elements() const { return session_->elements(); }
  // Continue a resumed run's monitor record; takes the session's lock.
  void continue_monitor(const nb::dict record, const bool plane,
                        const nb::handle unwatched, const nb::dict edge_record) {
    auto resumption = watch_resumption_from(record);
    auto stretches = watch_stretches_from(unwatched);
    auto edge = native::bindings::transverse_resumption_from(edge_record);
    const nb::gil_scoped_release release;
    session_->continue_monitor(resumption, plane, stretches, edge);
  }
  // The transverse edge's record and finish, or None; takes the session's lock.
  nb::object edge_record() {
    std::optional<std::pair<native::transverse::Record, native::transverse::Finish>>
        record;
    {
      const nb::gil_scoped_release release;
      record = session_->edge_record();
    }
    if (!record)
      return nb::none();
    return native::bindings::transverse_record_to(record->first, record->second);
  }
  // The boundary monitor's record, or None; takes the session's lock.
  nb::object monitor_record() {
    std::optional<native::watch::Record> record;
    {
      const nb::gil_scoped_release release;
      record = session_->monitor_record();
    }
    if (!record)
      return nb::none();
    return watch_record_to(*record);
  }
  // solprop SEP 47 section 5.5: the Adams march's state with the orders of its
  // accepted steps from `since` (the call ledger), or None for a Runge-Kutta march;
  // read under the session's lock, between advances.
  nb::object march_state(const std::size_t since) {
    std::optional<march::State> state;
    std::vector<std::size_t> orders;
    {
      const nb::gil_scoped_release release;
      session_->read(
          0, 0,
          [&](const native::session::Method& method,
              native::session::Resident& /*resident*/) {
            const auto* adams = std::get_if<native::session::AdamsMethod>(&method);
            if (adams == nullptr)
              return;
            state = adams->state;
            const std::size_t first = std::min(since, adams->orders.size());
            orders.assign(adams->orders.begin() + static_cast<std::ptrdiff_t>(first),
                          adams->orders.end());
          });
    }
    if (!state)
      return nb::none();
    nb::dict out = march_state_to(*state);
    out["orders"] = orders;
    return out;
  }
  // The Adams march's history at the pending batch's committed state, its ticket
  // `(sequence, generation)` (SEP 47 section 5.4): the march's state, the accepted
  // field as the state holds it and each persistent history row, each copied once,
  // losslessly (`AdamsMarch.snapshot`); None for a Runge-Kutta march or before any
  // history.
  nb::object march_snapshot(const std::uint64_t sequence,
                            const std::uint64_t generation) {
    std::optional<march::State> state;
    std::unique_ptr<ComplexStorage> field;
    std::vector<std::string> names;
    std::vector<std::array<std::size_t, 3>> extents;
    std::vector<std::unique_ptr<ComplexStorage>> rows;
    {
      const nb::gil_scoped_release release;
      session_->read(
          sequence, generation,
          [&](const native::session::Method& method,
              native::session::Resident& resident) {
            const auto* adams = std::get_if<native::session::AdamsMethod>(&method);
            if (adams == nullptr || adams->state.history.empty())
              return;
            state = adams->state;
            field = std::make_unique<ComplexStorage>(resident.elements());
            resident.field_to_host(reinterpret_cast<double*>(field->data()));
            const auto& general = owner_->leased_state(this);
            names = general.adams_row_names();
            for (const auto& name : names) {
              const auto extent = general.adams_row_extents(name);
              auto storage =
                  std::make_unique<ComplexStorage>(extent[0] * extent[1] * extent[2]);
              general.adams_rows_to_host(name, storage->data(), storage->size());
              extents.push_back(extent);
              rows.push_back(std::move(storage));
            }
          });
    }
    if (!state)
      return nb::none();
    const std::size_t n = field->size();
    nb::dict held;
    for (std::size_t index = 0; index < names.size(); ++index)
      held[names[index].c_str()] =
          owned_complex(std::move(rows[index]),
                        {extents[index][0], extents[index][1], extents[index][2]});
    return nb::make_tuple(march_state_to(*state), owned_complex(std::move(field), {n}),
                          held);
  }
  // A z table's `Phi` at `z`, the noise phase at the pending ticket
  // `(sequence, generation)`'s plane (`_ZTables.values`); takes the session's lock.
  ComplexOutput table_phi(const std::uint64_t sequence, const std::uint64_t generation,
                          const double z) {
    std::vector<native::Complex> values;
    {
      const nb::gil_scoped_release release;
      session_->read(sequence, generation,
                     [&](const native::session::Method& /*method*/,
                         native::session::Resident& /*resident*/) {
                       values = owner_->leased_table_phi(this, z);
                     });
    }
    const std::size_t columns = owner_->columns();
    return complex_output(values, {columns, values.size() / columns});
  }
  // Issue a failed terminal batch for a failure outside the session; takes its lock.
  std::optional<BatchRecord> fail_outside() {
    std::optional<native::session::Batch> batch;
    {
      const nb::gil_scoped_release release;
      batch = session_->fail_outside();
    }
    if (!batch)
      return std::nullopt;
    return BatchRecord(std::move(*batch));
  }
  // Takes the session's lock (waiting for an advance in hand): bound with the
  // interpreter lock released. The lease is released only once the session closed.
  void close() noexcept {
    BoundState* owner = owner_;
    if (close_native() && owner != nullptr)
      owner->release(this);
  }
  // The owner's own close: the owner forgets the session only once it closed, so a
  // session that could not take its lock keeps the state it uses alive.
  bool close_native() noexcept {
    if (!session_->close())
      return false;
    owner_ = nullptr;
    return true;
  }

private:
  BoundState* owner_;
  std::unique_ptr<native::session::Session> session_;
};

void BoundState::close() noexcept {
  if (session_ != nullptr) {
    // A session that did not close still uses the state: keep it, leased.
    if (!session_->close_native())
      return;
    session_ = nullptr;
  }
  if (general_state_) {
    general_state_->close();
    general_state_.reset();
  }
}

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
  [[nodiscard]] const std::shared_ptr<const native::statistics::Plan>& plan() const {
    return plan_;
  }
  [[nodiscard]] std::size_t packed_size() const { return plan_->packed_size(); }

private:
  std::shared_ptr<const native::statistics::Plan> plan_;
};

native::session::StatisticsRecipe statistics_recipe_from(const nb::handle value) {
  native::session::StatisticsRecipe recipe;
  if (value.is_none())
    return recipe;
  try {
    recipe.plan = nb::cast<const BoundStatisticsPlan&>(value).plan();
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("a session's statistics recipe is a StatisticsPlan");
  }
  return recipe;
}

native::session::Plan session_plan_from(const nb::dict values) {
  native::session::Plan plan;
  try {
    plan.planes = real_vector(exact_array<RealInput>(values["planes"]));
    plan.next_plane = nb::cast<std::size_t>(values["next_plane"]);
    plan.launch_pending = nb::cast<bool>(values["launch_pending"]);
    plan.step_planes = nb::cast<std::uint64_t>(values["step_planes"]);
    plan.plane_statistics = statistics_recipe_from(values["plane_statistics"]);
    plan.row_statistics = statistics_recipe_from(values["row_statistics"]);
    plan.period = nb::cast<std::uint64_t>(values["period"]);
    plan.attempt_capacity = nb::cast<std::size_t>(values["attempt_capacity"]);
    plan.row_capacity = nb::cast<std::size_t>(values["row_capacity"]);
    plan.replay = nb::cast<bool>(values["replay"]);
    plan.replay_steps = nb::cast<std::size_t>(values["replay_steps"]);
    plan.replay_length = nb::cast<double>(values["replay_length"]);
    plan.call_attempts = nb::cast<std::size_t>(values["call_attempts"]);
    plan.call_seconds = nb::cast<double>(values["call_seconds"]);
    if (values.contains("monitor") && !values["monitor"].is_none())
      plan.monitor = monitor_plan_from(nb::cast<nb::dict>(values["monitor"]));
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("a session's plan has the wrong types");
  } catch (const nb::python_error&) {
    throw std::invalid_argument("a session's plan lacks a field");
  }
  return plan;
}

std::unique_ptr<BoundSession> start_session(BoundState& state, const std::string& kind,
                                            const nb::dict march_state,
                                            const nb::dict plan, const double z) {
  const nb::object dense =
      plan.contains("dense") ? nb::object(plan["dense"]) : nb::object(nb::none());
  auto method = state.session_method(kind, march_state, dense);
  auto session = std::make_unique<native::session::Session>(
      state.session_resident(), std::move(method), session_plan_from(plan), z);
  return std::make_unique<BoundSession>(state, std::move(session));
}

void shutdown_kokkos() {
  const std::scoped_lock lock(runtime_state.mutex);
  for (auto* batch : runtime_state.batches)
    batch->close();
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

#if defined(_WIN32)
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
  result["gemm_provider"] = SOLPROP_NATIVE_GEMM_PROVIDER;
  result["tpl_blas_enabled"] = static_cast<bool>(SOLPROP_NATIVE_TPL_BLAS_ENABLED);
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
#if defined(KOKKOS_ENABLE_CUDA)
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
#if defined(SOLPROP_NATIVE_RAMAN_ACCELERATE)
  result["raman_fft_library"] = loaded_library("vDSP_fftm_zropD");
#else
  result["raman_fft_library"] = result["fft_library"];
#endif
  result["openmp_library"] = loaded_library("omp_get_num_threads");
  result["device_team_size"] = SOLPROP_NATIVE_DEVICE_TEAM_SIZE_RECORD;
  result["device_reduction_size"] = SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE_RECORD;
#if defined(KOKKOS_ENABLE_OPENMP)
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
  result["coverage_instrumented"] =
      static_cast<bool>(SOLPROP_NATIVE_COVERAGE_INSTRUMENTED);
  result["coverage_source_hash"] = SOLPROP_NATIVE_COVERAGE_SOURCE_HASH;
  result["precision"] = "complex128";
  result["fft_normalization"] = "none (explicit inverse scaling)";
#if defined(KOKKOS_ENABLE_CUDA)
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
  result["native_arch_enabled"] = static_cast<bool>(SOLPROP_NATIVE_HOST_ARCH_ENABLED);
  result["aggressive_vectorization_enabled"] =
      static_cast<bool>(SOLPROP_NATIVE_AGGRESSIVE_VECTORIZATION_ENABLED);
  result["ipo_enabled"] = static_cast<bool>(SOLPROP_NATIVE_IPO_ENABLED);
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
#if defined(__FAST_MATH__)
  result["fast_math_enabled"] = true;
#else
  result["fast_math_enabled"] = false;
#endif
#if defined(__apple_build_version__)
  result["compiler_family"] = "AppleClang";
#elif defined(__INTEL_LLVM_COMPILER)
  result["compiler_family"] = "IntelLLVM";
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

} // namespace

NB_MODULE(_core, module_) {
  nb::exception<native::EnvelopeConvergenceError>(module_, "EnvelopeConvergenceError");
  nb::exception<native::BackendRefusal>(module_, "BackendRefusal");
  module_.doc() = "Private nanobind seam for solprop-native";
  module_.def("initialize_runtime", &initialize_kokkos, "threads"_a, "kernels"_a = 0,
              "fft"_a = 0, "raman_fft"_a = 0, "raman_tail_fft"_a = 0,
              "band_tiles"_a = -1);
  module_.def("fit_blas_threads", &fit_blas_threads, "threads"_a);
  module_.def("register_host_blas", &register_host_blas);
  // solprop SEP 45 phase 1: the boundary watch's predictions, for solprop's tests to
  // hold against its monitor on drawn readings.
  module_.def(
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
  module_.def(
      "rk_score",
      [](const RealInput partials, const double rtol, const double atol) {
        if (partials.ndim() != 2 || partials.shape(0) != 3)
          throw std::invalid_argument("a trial's partials have shape (3, slots)");
        return native::rk::score(partials.data(), partials.shape(1), rtol, atol);
      },
      "partials"_a, "rtol"_a, "atol"_a);
  // solprop SEP 47: the native half-maximum width the statistics plan finishes with.
  module_.def(
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
  module_.def(
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
  module_.def(
      "radial_fluence_widths",
      [](const RealInput radius, const RealInput fluence, const RealInput weights,
         const double axis_fluence, const double e2_fraction) {
        if (radius.ndim() != 1 || fluence.ndim() != 1 || weights.ndim() != 1 ||
            fluence.shape(0) != radius.shape(0) || weights.shape(0) != radius.shape(0))
          throw std::invalid_argument("a radial profile's samples are one-dimensional "
                                      "and of one length");
        const auto widths = native::statistics::widths::radial(
            radius.data(), fluence.data(), weights.data(), radius.shape(0), axis_fluence,
            e2_fraction);
        return nb::make_tuple(widths[0], widths[1], widths[2]);
      },
      "radius"_a.noconvert(), "fluence"_a.noconvert(), "weights"_a.noconvert(),
      "axis_fluence"_a, "e2_fraction"_a);
  module_.def(
      "fluence_block_widths",
      [](const BoundStatisticsPlan& plan, const RealInput block) {
        if (block.ndim() != 1)
          throw std::invalid_argument("a fluence block is one-dimensional");
        const auto widths =
            native::statistics::widths::of_block(*plan.plan(), real_vector(block));
        return nb::make_tuple(widths[0], widths[1], widths[2]);
      },
      "plan"_a, "block"_a.noconvert());
  module_.def(
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
  module_.def(
      "watch_due",
      [](const nb::dict policy, const nb::dict state, const double z) {
        return native::watch::due(watch_policy_from(policy), watch_state_from(state),
                                  z);
      },
      "policy"_a, "state"_a, "z"_a);
  module_.def(
      "watch_split_limit",
      [](const nb::dict policy, const RealInput occupancy) {
        const auto rules = watch_policy_from(policy);
        const auto values = real_vector(occupancy);
        if (values.size() != native::watch::buckets)
          throw std::invalid_argument("an occupancy has 128 buckets");
        return native::watch::split_limit(rules, values.data());
      },
      "policy"_a, "occupancy"_a);
  module_.def("available_device_memory", []() -> nb::object {
#if defined(KOKKOS_ENABLE_CUDA)
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
  });
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
  // solprop SEP 28 section 4.2's reference parts, for its benchmarks only.
  module_.def(
      "reference_fft3_pairs",
      [](const ComplexInput& values, const std::size_t warm, const std::size_t pairs) {
        if (values.ndim() != 3)
          throw std::invalid_argument("the reference volume must be three-dimensional");
        return native::reference_fft3_pairs(values.data(), values.shape(0),
                                            values.shape(1), values.shape(2), warm,
                                            pairs);
      },
      "values"_a, "warm"_a, "pairs"_a);
  module_.def("reference_stream_copies", &native::reference_stream_copies, "elements"_a,
              "warm"_a, "repeats"_a);
  // The calling thread's OpenMP default as a native call sees it (solprop issue #130).
  module_.def("scoped_openmp_default", &native::scoped_openmp_default);
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  // solprop SEP 46: forget FFTW's process-wide wisdom, for a test of the planning
  // transaction.
  module_.def("forget_fftw_wisdom", [] {
    fftw_init_threads();
    fftw_forget_wisdom();
  });
#else
  module_.def("forget_fftw_wisdom", [] {});
#endif
  module_.def("configure_native_spin", &native::configure_native_spin,
              "milliseconds"_a);
  module_.def("scoped_block_time", &native::scoped_block_time);
  module_.def("shutdown_runtime", &shutdown_kokkos);
  module_.def("build_record", &build_record);
  // FFTW's wisdom as text, for the plan store the adapter owns
  // (solprop_native._wisdom). None where there is none to keep: a device build,
  // which plans with the device FFT, or a provider without FFTW's wisdom.
  module_.def("export_fftw_wisdom", []() -> nb::object {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return nb::none();
#else
    std::string text = native::export_fftw_wisdom();
    return text.empty() ? nb::none() : nb::cast(std::move(text));
#endif
  });
  module_.def(
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
  native::bindings::bind_helpers(module_);
  native::bindings::bind_transverse(module_);
  nb::class_<ScriptedMarch>(module_, "ScriptedMarch")
      .def(nb::init<nb::dict, RealInput, double, RealInput, RealInput>(), "policy"_a,
           "norms"_a.noconvert(), "carrier_rate"_a, "steps"_a.noconvert(),
           "edges"_a.noconvert())
      .def("run", &ScriptedMarch::run, "state"_a, "bounds"_a)
      .def("log", &ScriptedMarch::log);
  nb::class_<ScriptedRkMarch>(module_, "ScriptedRkMarch")
      .def(nb::init<nb::dict, RealInput, double, RealInput, RealInput>(), "policy"_a,
           "norms"_a.noconvert(), "h0"_a, "steps"_a.noconvert(), "edges"_a.noconvert())
      .def("run", &ScriptedRkMarch::run, "state"_a, "bounds"_a)
      .def("log", &ScriptedRkMarch::log)
      .def("used", &ScriptedRkMarch::used);
  nb::class_<BoundState>(module_, "State")
      .def(nb::init<const nb::dict&>())
      .def(
          "set_initial",
          [](BoundState& self, const ComplexInput field, const ComplexInput noise) {
            // Read in place: the arguments keep the arrays alive for the call.
            const nb::gil_scoped_release release;
            self.set_initial(complex_span(field), complex_span(noise));
          },
          "field"_a.noconvert(), "noise"_a.noconvert())
      .def(
          "rk_trial",
          [](BoundState& self, const double z, const double dz,
             const std::vector<double>& nodes, const bool prepare_only) {
            std::vector<double> partials;
            {
              const nb::gil_scoped_release release;
              partials = self.rk_trial(z, dz, nodes, prepare_only);
            }
            return prepare_only ? nb::object(nb::none())
                                : nb::cast(real_output(std::move(partials), 3));
          },
          "z"_a, "dz"_a, "nodes"_a, "prepare_only"_a = false)
      .def(
          "trial_step",
          [](BoundState& self, const double dz, const RealInput phi_delta,
             const RealInput phi_at, const RealInput prefactor, const RealInput density,
             const RealInput taper, const RealInput preionisation,
             const RealInput raman_rates, const nb::object material,
             const bool prepare_only) {
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
          "dz"_a, "phi_delta"_a.noconvert(), "phi_at"_a.noconvert(),
          "prefactor"_a.noconvert(), "density"_a.noconvert(), "taper"_a.noconvert(),
          "preionisation"_a.noconvert(), "raman_rates"_a.noconvert(),
          "material"_a = nb::none(), "prepare_only"_a = false)
      .def(
          "adams_source",
          [](BoundState& self, const ComplexInput input, const RealInput phi_delta,
             const RealInput phi_at, const RealInput prefactor, const RealInput density,
             const RealInput taper, const RealInput preionisation,
             const RealInput raman_rates, const nb::dict material) {
            // solprop's Adams source (SEP 14 phase 2): the trial's scalar layout, every
            // stage's row the evaluation position's, of which stage zero is read.
            auto scalars = trial_from(phi_delta, phi_at, prefactor, density, taper,
                                      preionisation, raman_rates);
            scalars.kerr_density =
                real_vector(exact_array<RealInput>(material["kerr_density"]));
            scalars.plasma_densities =
                real_vector(exact_array<RealInput>(material["plasma_densities"]));
            scalars.plasma_preionisation =
                real_vector(exact_array<RealInput>(material["plasma_preionisation"]));
            scalars.raman_densities =
                real_vector(exact_array<RealInput>(material["raman_densities"]));
            // A moving guide's maps are combined at the evaluation position, as a
            // trial's stages are.
            if (material.contains("transform_weights"))
              scalars.transform_weights =
                  real_vector(exact_array<RealInput>(material["transform_weights"]));
            if (material.contains("dispersive_densities"))
              scalars.dispersive_densities =
                  real_vector(exact_array<RealInput>(material["dispersive_densities"]));
            const auto held = complex_vector(input);
            std::vector<native::Complex> values;
            {
              const nb::gil_scoped_release release;
              values = self.adams_source(held, scalars);
            }
            return complex_output(values,
                                  {self.columns(), values.size() / self.columns()});
          },
          "input"_a.noconvert(), "phi_delta"_a.noconvert(), "phi_at"_a.noconvert(),
          "prefactor"_a.noconvert(), "density"_a.noconvert(), "taper"_a.noconvert(),
          "preionisation"_a.noconvert(), "raman_rates"_a.noconvert(), "material"_a)
      .def(
          "adopt",
          [](BoundState& self, const ComplexInput field, const double dz) {
            const auto held = complex_vector(field);
            const nb::gil_scoped_release release;
            self.adopt(held, dz);
          },
          "field"_a.noconvert(), "dz"_a)
      // solprop SEP 24 phase 3: the low-storage march's resident step.
      .def(
          "low_storage_step",
          [](BoundState& self, const double dz, const RealInput phi_delta,
             const RealInput phi_at, const RealInput prefactor, const RealInput density,
             const RealInput taper, const RealInput preionisation,
             const RealInput raman_rates, const nb::dict material,
             const bool rebuild_table) {
            if (self.raman() &&
                (raman_rates.ndim() != 2 || raman_rates.shape(0) != self.stages() ||
                 raman_rates.shape(1) != self.raman_terms()))
              throw std::invalid_argument(
                  "raman_rates must have shape (n_stages, n_terms)");
            const auto scalars =
                adams_scalars(phi_delta, phi_at, prefactor, density, taper,
                              preionisation, raman_rates, material);
            auto& state = self.open_state();
            const nb::gil_scoped_release release;
            return state.low_storage_step(dz, scalars, rebuild_table);
          },
          "dz"_a, "phi_delta"_a.noconvert(), "phi_at"_a.noconvert(),
          "prefactor"_a.noconvert(), "density"_a.noconvert(), "taper"_a.noconvert(),
          "preionisation"_a.noconvert(), "raman_rates"_a.noconvert(), "material"_a,
          "rebuild_table"_a)
      // solprop SEP 16 phase 3b: the Adams history resident in the engine.
      .def(
          "adams_allocate",
          [](BoundState& self, const std::size_t slots, const bool compensated,
             const bool interpolate) {
            self.open_state().adams_allocate(slots, compensated, interpolate);
          },
          "slots"_a, "compensated"_a = false, "interpolate"_a = true)
      .def(
          "adams_compensation_start",
          [](BoundState& self, const std::size_t slot, const RealInput rate) {
            const auto weights = real_vector(rate);
            const nb::gil_scoped_release release;
            self.open_state().adams_compensation_start(slot, weights);
          },
          "slot"_a, "rate"_a.noconvert())
      .def(
          "adams_start",
          [](BoundState& self, const std::size_t slot, const RealInput phi_delta,
             const RealInput phi_at, const RealInput prefactor, const RealInput density,
             const RealInput taper, const RealInput preionisation,
             const RealInput raman_rates, const nb::dict material) {
            const auto scalars =
                adams_scalars(phi_delta, phi_at, prefactor, density, taper,
                              preionisation, raman_rates, material);
            const nb::gil_scoped_release release;
            self.open_state().adams_start(slot, scalars);
          },
          "slot"_a, "phi_delta"_a.noconvert(), "phi_at"_a.noconvert(),
          "prefactor"_a.noconvert(), "density"_a.noconvert(), "taper"_a.noconvert(),
          "preionisation"_a.noconvert(), "raman_rates"_a.noconvert(), "material"_a)
      .def("adams_magnitude",
           [](BoundState& self) {
             const nb::gil_scoped_release release;
             return self.open_state().adams_magnitude();
           })
      .def(
          "adams_scale",
          [](BoundState& self, const double rtol, const double atol,
             const RealInput weights) {
            const auto values = real_vector(weights);
            const nb::gil_scoped_release release;
            self.open_state().adams_scale(rtol, atol, values);
          },
          "rtol"_a, "atol"_a, "weights"_a.noconvert())
      .def(
          "adams_fused_trial",
          [](BoundState& self, const double dz, const IndexInput slots,
             const RealInput predictor, const RealInput corrector, const RealInput rows,
             const RealInput divisor, const RealInput phi_delta, const RealInput phi_at,
             const RealInput prefactor, const RealInput density, const RealInput taper,
             const RealInput preionisation, const RealInput raman_rates,
             const nb::dict material) {
            const auto weights =
                adams_weights(slots, predictor, corrector, rows, predictor.size());
            const auto scale = real_vector(divisor);
            const auto scalars =
                adams_scalars(phi_delta, phi_at, prefactor, density, taper,
                              preionisation, raman_rates, material);
            std::vector<double> norms;
            {
              const nb::gil_scoped_release release;
              norms = self.open_state().adams_fused_trial(dz, weights, scale, scalars);
            }
            return real_output(std::move(norms), 1);
          },
          "dz"_a, "slots"_a.noconvert(), "predictor"_a.noconvert(),
          "corrector"_a.noconvert(), "rows"_a.noconvert(), "divisor"_a.noconvert(),
          "phi_delta"_a.noconvert(), "phi_at"_a.noconvert(), "prefactor"_a.noconvert(),
          "density"_a.noconvert(), "taper"_a.noconvert(), "preionisation"_a.noconvert(),
          "raman_rates"_a.noconvert(), "material"_a)
      .def(
          "adams_fused_accept",
          [](BoundState& self, const double dz, const IndexInput keep,
             const std::size_t free, const IndexInput dense, const bool interpolate) {
            const auto kept = adams_rows(keep);
            auto slopes = adams_rows(dense);
            // The interpolant's slopes are the new source and these rows.
            slopes.width = interpolate ? slopes.width + 1 : 0;
            const nb::gil_scoped_release release;
            self.open_state().adams_fused_accept(dz, kept, free, slopes);
          },
          "dz"_a, "keep"_a.noconvert(), "free"_a, "dense"_a.noconvert(),
          "interpolate"_a)
      .def("adams_hold", [](BoundState& self) { self.open_state().adams_hold(); })
      .def("adams_restore", [](BoundState& self) { self.open_state().adams_restore(); })
      // solprop SEP 29 G2: the resident history's persistent rows across a resume.
      .def("adams_row_names",
           [](BoundState& self) { return self.open_state().adams_row_names(); })
      .def(
          "adams_rows_to_host",
          [](BoundState& self, const std::string& name) {
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
            nb::capsule owner(pointer, [](void* value) noexcept {
              const std::unique_ptr<ComplexStorage> held(
                  static_cast<ComplexStorage*>(value));
            });
            storage.release();
            const std::array<std::size_t, 3> shape{extents[0], extents[1], extents[2]};
            return ComplexOutput(pointer->data(), shape.size(), shape.data(), owner);
          },
          "name"_a)
      .def(
          "adams_rows_from_host",
          [](BoundState& self, const std::string& name, const ComplexInput rows) {
            auto& state = self.open_state();
            // The extents are checked against the row's view before any copy.
            const auto extents = state.adams_row_extents(name);
            if (rows.ndim() != 3 || rows.shape(0) != extents[0] ||
                rows.shape(1) != extents[1] || rows.shape(2) != extents[2])
              throw std::invalid_argument(
                  "the Adams history row '" + name + "' must have shape (" +
                  std::to_string(extents[0]) + ", " + std::to_string(extents[1]) +
                  ", " + std::to_string(extents[2]) + ")");
            const auto values = complex_vector(rows);
            const nb::gil_scoped_release release;
            state.adams_rows_from_host(name, values);
          },
          "name"_a, "rows"_a.noconvert())
      // solprop SEP 29 G1: the general z tables, resident and evaluated natively.
      .def(
          "table_upload",
          [](BoundState& self, const std::string& kind, const RealInput edges,
             const double lower, const double upper,
             const nb::ndarray<const std::complex<double>, nb::ndim<3>, nb::c_contig,
                               nb::device::cpu>
                 coefficients) {
            if (edges.ndim() != 1)
              throw std::invalid_argument(
                  "z table breakpoints must be one-dimensional");
            auto& state = self.open_state();
            const nb::gil_scoped_release release;
            state.table_upload(table_kind(kind), real_vector(edges), lower, upper,
                               coefficients.data(), coefficients.shape(0),
                               coefficients.shape(1), coefficients.shape(2));
          },
          "kind"_a, "edges"_a.noconvert(), "lower"_a, "upper"_a,
          "coefficients"_a.noconvert())
      .def(
          "table_evaluate",
          [](BoundState& self, const IndexInput slots, const RealInput positions) {
            if (slots.ndim() != 1 || positions.ndim() != 1)
              throw std::invalid_argument(
                  "z table slots and positions must be one-dimensional");
            const auto held_slots = index_vector(slots);
            const auto held_positions = real_vector(positions);
            auto& state = self.open_state();
            const nb::gil_scoped_release release;
            state.table_evaluate(held_slots, held_positions);
          },
          "slots"_a.noconvert(), "positions"_a.noconvert())
      .def(
          "table_rate",
          [](BoundState& self, const std::size_t slot, const double z,
             const bool before) {
            auto& state = self.open_state();
            const nb::gil_scoped_release release;
            state.table_rate(slot, z, before);
          },
          "slot"_a, "z"_a, "before"_a)
      .def(
          "table_terms_to_host",
          [](BoundState& self, const std::string& kind, const std::size_t slot) {
            auto& state = self.open_state();
            std::vector<native::Complex> values;
            {
              const nb::gil_scoped_release release;
              values = state.table_terms_to_host(table_kind(kind), slot);
            }
            return complex_output(values,
                                  {self.columns(), values.size() / self.columns()});
          },
          "kind"_a, "slot"_a)
      .def(
          "phase_sum_to_host",
          [](BoundState& self, const RealInput weights) {
            const auto held = real_vector(weights);
            auto& state = self.open_state();
            std::vector<native::Complex> values;
            {
              const nb::gil_scoped_release release;
              values = state.phase_sum_to_host(held);
            }
            return complex_output(values,
                                  {self.columns(), values.size() / self.columns()});
          },
          "weights"_a.noconvert())
      .def(
          "adams_observe",
          [](BoundState& self, const RealInput weights, const bool hermite,
             const RealInput phi_delta, const RealInput phi_at,
             const RealInput prefactor, const RealInput density, const RealInput taper,
             const RealInput preionisation, const RealInput raman_rates,
             const nb::dict material) {
            native::AdamsWeights table;
            const auto values = real_vector(weights);
            if (values.size() > native::adams_max_samples)
              throw std::invalid_argument("too many Adams interpolation weights");
            for (std::size_t t = 0; t < values.size(); ++t) {
              table.predictor[t] = values[t];
              table.corrector[t] = values[t];
            }
            const auto scalars =
                adams_scalars(phi_delta, phi_at, prefactor, density, taper,
                              preionisation, raman_rates, material);
            const nb::gil_scoped_release release;
            self.open_state().adams_observe(table, hermite, scalars);
          },
          "weights"_a.noconvert(), "hermite"_a, "phi_delta"_a.noconvert(),
          "phi_at"_a.noconvert(), "prefactor"_a.noconvert(), "density"_a.noconvert(),
          "taper"_a.noconvert(), "preionisation"_a.noconvert(),
          "raman_rates"_a.noconvert(), "material"_a)
      .def(
          "adams_exponential_trial",
          [](BoundState& self, const double dz, const IndexInput slots,
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
            const auto step =
                adams_scalars(phi_delta, phi_at, prefactor, density, taper,
                              preionisation, raman_rates, material);
            const auto sample =
                adams_scalars(sample_phi_delta, sample_phi_at, sample_prefactor,
                              sample_density, sample_taper, sample_preionisation,
                              sample_raman_rates, sample_material);
            std::vector<double> norms;
            {
              const nb::gil_scoped_release release;
              norms = self.open_state().adams_exponential_trial(
                  dz, weights, scale, step, sample, rate_weights);
            }
            return real_output(std::move(norms), 1);
          },
          "dz"_a, "slots"_a.noconvert(), "predictor"_a.noconvert(),
          "corrector"_a.noconvert(), "rows"_a.noconvert(), "divisor"_a.noconvert(),
          "phi_delta"_a.noconvert(), "phi_at"_a.noconvert(), "prefactor"_a.noconvert(),
          "density"_a.noconvert(), "taper"_a.noconvert(), "preionisation"_a.noconvert(),
          "raman_rates"_a.noconvert(), "material"_a, "sample_phi_delta"_a.noconvert(),
          "sample_phi_at"_a.noconvert(), "sample_prefactor"_a.noconvert(),
          "sample_density"_a.noconvert(), "sample_taper"_a.noconvert(),
          "sample_preionisation"_a.noconvert(), "sample_raman_rates"_a.noconvert(),
          "sample_material"_a, "spans"_a.noconvert(), "comp_predictor"_a.noconvert(),
          "comp_corrector"_a.noconvert(), "comp_rows"_a.noconvert(),
          "rate"_a.noconvert(), "carrier"_a.noconvert())
      // solprop SEP 39 phase 1: the trial's tables, stage scalars and rate formed
      // natively from the nodes and the tables uploaded once per run.
      .def(
          "adams_stage_tables",
          [](BoundState& self, const nb::dict tables) {
            self.set_stage_tables(stage_tables_from(tables));
          },
          "tables"_a)
      .def("has_stage_tables", &BoundState::has_stage_tables)
      .def("table_slots", &BoundState::table_slots, "phi"_a, "prefactor"_a, "slots"_a,
           "rate"_a)
      .def(
          "table_load",
          [](BoundState& self, const std::vector<double>& positions) {
            const nb::gil_scoped_release release;
            return self.table_load(positions);
          },
          "positions"_a)
      .def(
          "table_rate_at",
          [](BoundState& self, const double z, const bool before) {
            const nb::gil_scoped_release release;
            self.table_rate_at(z, before);
          },
          "z"_a, "before"_a)
      .def("table_evaluations", &BoundState::table_evaluations)
      // solprop SEP 39 phase 2: the native Adams march.
      .def(
          "adams_march_configure",
          [](BoundState& self, const nb::dict policy, const bool fused,
             const bool compensated, const bool interpolate, const RealInput steps,
             const RealInput edges) {
            self.march_configure(march_policy_from(policy), fused, compensated,
                                 interpolate, {real_vector(steps), real_vector(edges)});
          },
          "policy"_a, "fused"_a, "compensated"_a, "interpolate"_a,
          "steps"_a.noconvert(), "edges"_a.noconvert())
      .def("adams_march_run", &BoundState::march_run, "state"_a, "bounds"_a)
      .def(
          "rk_march_configure",
          [](BoundState& self, const nb::dict policy, const std::vector<double>& nodes,
             const double rtol, const double atol, const RealInput steps,
             const RealInput edges) {
            self.rk_march_configure(rk_policy_from(policy), nodes, rtol, atol,
                                    real_vector(steps), real_vector(edges));
          },
          "policy"_a, "nodes"_a, "rtol"_a, "atol"_a, "steps"_a.noconvert(),
          "edges"_a.noconvert())
      .def("rk_march_run", &BoundState::rk_march_run, "state"_a, "bounds"_a)
      .def("low_storage_configure", &BoundState::low_storage_configure, "flows"_a,
           "separable"_a, "origin"_a)
      .def("start_session", &start_session, "kind"_a, "state"_a, "plan"_a, "z"_a,
           nb::keep_alive<0, 1>())
      .def("rk_weights_configure", &BoundState::rk_weights_configure, "weights"_a)
      .def(
          "rk_weighted_score",
          [](BoundState& self, const RealInput partials, const double rtol,
             const double atol) {
            if (partials.ndim() != 2 || partials.shape(0) != 3)
              throw std::invalid_argument("a trial's partials have shape (3, slots)");
            return self.rk_weighted_score(partials.data(), partials.shape(1), rtol,
                                          atol);
          },
          "partials"_a.noconvert(), "rtol"_a, "atol"_a)
      .def("flow_guard_configure", &BoundState::flow_guard_configure, "row_sum"_a,
           "condition_bound"_a, "taylor_scale"_a, "max_squarings"_a, "allowance"_a)
      // A test hook: fail the resident engine's `stage` after `after` more calls.
      .def("adams_march_fault", &BoundState::march_fault, "stage"_a, "after"_a,
           "category"_a)
      .def(
          "adams_exponential_trial_at",
          [](BoundState& self, const double z, const double dz, const RealInput nodes,
             const IndexInput slots, const std::size_t order,
             const std::vector<std::size_t>& orders, const bool compensated,
             const RealInput divisor, const RealInput carrier) {
            auto weights =
                exponential_weights_at(nodes, slots, order, orders, compensated);
            carrier_frame(weights, carrier, weights.width - 1);
            const auto scale = real_vector(divisor);
            const double end = z + dz;
            auto trial = self.stage_trial(z, end);
            const auto rate =
                compensated ? self.stage_rate(end) : std::vector<double>{};
            std::vector<double> norms;
            {
              const nb::gil_scoped_release release;
              norms = self.open_state().adams_exponential_trial(
                  dz, weights, scale, trial.step, trial.sample, rate);
            }
            return real_output(std::move(norms), 1);
          },
          "z"_a, "dz"_a, "nodes"_a.noconvert(), "slots"_a.noconvert(), "order"_a,
          "orders"_a, "compensated"_a, "divisor"_a.noconvert(), "carrier"_a.noconvert())
      .def(
          "adams_fused_trial_at",
          [](BoundState& self, const double z, const double dz, const RealInput nodes,
             const IndexInput slots, const std::size_t order,
             const std::vector<std::size_t>& orders, const RealInput divisor) {
            const auto slot_values = index_vector(slots);
            const auto weights = native::stage::fused_weights(
                nodes.data(), element_count(nodes), slot_values.data(),
                slot_values.size(), order, orders.data(), orders.size());
            const auto scale = real_vector(divisor);
            const auto trial = self.stage_trial(z, z + dz);
            std::vector<double> norms;
            {
              const nb::gil_scoped_release release;
              norms =
                  self.open_state().adams_fused_trial(dz, weights, scale, trial.step);
            }
            return real_output(std::move(norms), 1);
          },
          "z"_a, "dz"_a, "nodes"_a.noconvert(), "slots"_a.noconvert(), "order"_a,
          "orders"_a, "divisor"_a.noconvert())
      .def(
          "adams_stage_scalars",
          [](BoundState& self, const double z0, const double z1) {
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
          "z0"_a, "z1"_a)
      .def(
          "adams_stage_rate",
          [](BoundState& self, const double z) { return self.stage_rate(z); }, "z"_a)
      .def(
          "adams_exponential_accept",
          [](BoundState& self, const double dz, const std::size_t free,
             const IndexInput dense, const bool interpolate) {
            auto slopes = adams_rows(dense);
            slopes.width = interpolate ? slopes.width + 1 : 0;
            const nb::gil_scoped_release release;
            self.open_state().adams_exponential_accept(dz, free, slopes);
          },
          "dz"_a, "free"_a, "dense"_a.noconvert(), "interpolate"_a)
      .def(
          "adams_exponential_observe",
          [](BoundState& self, const double xi, const double dz,
             const RealInput coefficients, const RealInput dense_weight,
             const RealInput dense_theta, const RealInput carrier) {
            native::ExponentialWeights weights;
            const auto weight_values = real_vector(dense_weight);
            const auto theta_values = real_vector(dense_theta);
            if (weight_values.size() != theta_values.size() ||
                weight_values.size() > native::adams_max_samples + 1)
              throw std::invalid_argument(
                  "Adams compensation weights are inconsistent");
            for (std::size_t t = 0; t < weight_values.size(); ++t) {
              weights.dense_weight[t] = weight_values[t];
              weights.dense_theta[t] = theta_values[t];
            }
            const auto values = real_vector(coefficients);
            const auto count = static_cast<std::size_t>(
                std::llround(std::sqrt(static_cast<double>(values.size()))));
            if (count * count != values.size() || count > native::adams_max_samples)
              throw std::invalid_argument(
                  "Adams interpolant coefficients are not square");
            for (std::size_t j = 0; j < count; ++j)
              for (std::size_t k = 0; k < count; ++k)
                weights.corrector[j][k] = values[j * count + k];
            // The accepted step's carrier frame, its history rows after the endpoint.
            carrier_frame(weights, carrier, count == 0 ? 0 : count - 1);
            const nb::gil_scoped_release release;
            self.open_state().adams_exponential_observe(xi, dz, weights);
          },
          "xi"_a, "dz"_a, "coefficients"_a.noconvert(), "dense_weight"_a.noconvert(),
          "dense_theta"_a.noconvert(), "carrier"_a.noconvert())
      .def(
          "set_flow_actions",
          [](BoundState& self, const ComplexInput forward, const ComplexInput inverse,
             const BoolInput identity) {
            const auto held_forward = complex_vector(forward);
            const auto held_inverse = complex_vector(inverse);
            const auto held_identity = bool_vector(identity);
            const nb::gil_scoped_release release;
            self.set_flow_actions(held_forward, held_inverse, held_identity);
          },
          "forward"_a.noconvert(), "inverse"_a.noconvert(), "identity"_a.noconvert())
      .def(
          "form_flow_actions",
          [](BoundState& self, const std::size_t section, const double dz,
             const RealInput fractions) {
            const auto held_fractions = real_vector(fractions);
            std::vector<double> measured;
            {
              const nb::gil_scoped_release release;
              measured = self.form_flow_actions(section, dz, held_fractions);
            }
            return real_output(std::move(measured), 1);
          },
          "section"_a, "dz"_a, "fractions"_a.noconvert())
      // solprop SEP 47 phase 4: the last exponential Adams trial's coupled block and
      // exponent, observations for tests.
      .def("adams_block_to_host",
           [](BoundState& self) {
             std::vector<native::Complex> values;
             {
               const nb::gil_scoped_release release;
               values = self.adams_block_to_host();
             }
             return complex_output(values, {values.size()});
           })
      .def("adams_exponent_to_host",
           [](BoundState& self) {
             std::vector<native::Complex> values;
             {
               const nb::gil_scoped_release release;
               values = self.adams_exponent_to_host();
             }
             return complex_output(values, {values.size()});
           })
      .def(
          "flow_actions_to_host",
          [](BoundState& self, const bool inverse) {
            std::vector<native::Complex> values;
            {
              const nb::gil_scoped_release release;
              values = self.flow_actions_to_host(inverse);
            }
            return complex_output(values, {values.size()});
          },
          "inverse"_a)
      .def(
          "prepare_raman",
          [](BoundState& self, const RealInput rates, const nb::object densities) {
            if (rates.ndim() != 1 || rates.shape(0) != self.raman_terms())
              throw std::invalid_argument(
                  "Raman decay rates must have shape (n_terms,)");
            const auto held_rates = real_vector(rates);
            const auto held_densities =
                densities.is_none() ? std::vector<double>{}
                                    : real_vector(exact_array<RealInput>(densities));
            const nb::gil_scoped_release release;
            self.prepare_raman(held_rates, held_densities);
          },
          "rates"_a.noconvert(), "densities"_a = nb::none())
      .def(
          "weighted_error",
          [](BoundState& self, const RealInput scale) {
            std::vector<double> partials;
            const auto held_scale = real_vector(scale);
            {
              const nb::gil_scoped_release release;
              partials = self.weighted_error(held_scale);
            }
            return real_output(std::move(partials), 1);
          },
          "scale"_a.noconvert())
      .def(
          "spatial_statistics",
          [](BoundState& self, const std::uint64_t flags, const double intensity_factor,
             const double z) {
            std::vector<double> values;
            {
              const nb::gil_scoped_release release;
              values = self.spatial_statistics(flags, intensity_factor, z);
            }
            return real_output(std::move(values), 1);
          },
          "flags"_a, "intensity_factor"_a, "z"_a)
      .def(
          "plane_fluence_trace",
          [](BoundState& self, const double center_x, const double center_y,
             const double dx, const double dy, const double extent,
             const double spacing, const std::size_t n_radii, const RealInput cosines,
             const RealInput sines) {
            if (cosines.ndim() != 1 || sines.ndim() != 1)
              throw std::invalid_argument("fluence rings must be one-dimensional");
            const auto held_cosines = real_vector(cosines);
            const auto held_sines = real_vector(sines);
            std::vector<double> values;
            {
              const nb::gil_scoped_release release;
              values =
                  self.plane_fluence_trace(center_x, center_y, dx, dy, extent, spacing,
                                           n_radii, held_cosines, held_sines);
            }
            return real_output(std::move(values), 1);
          },
          "center_x"_a, "center_y"_a, "dx"_a, "dy"_a, "extent"_a, "spacing"_a,
          "n_radii"_a, "cosines"_a.noconvert(), "sines"_a.noconvert())
      .def(
          "axis_spectrum",
          [](BoundState& self, const RealInput matrix) {
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
          "matrix"_a.noconvert())
      .def(
          "aperture_statistics",
          [](BoundState& self, const std::uint64_t flags, const RealInput apertures,
             const int exponent, const RealInput regions,
             const double intensity_factor) {
            if (apertures.ndim() != 2 || regions.ndim() != 2 || regions.shape(1) != 4)
              throw std::invalid_argument(
                  "aperture tables must be (n_apertures, 11 + n_omega) and regions "
                  "(n_regions, 4)");
            const auto held_apertures = real_vector(apertures);
            const auto held_regions = real_vector(regions);
            std::vector<double> values;
            {
              const nb::gil_scoped_release release;
              values = self.aperture_statistics(flags, held_apertures, exponent,
                                                held_regions, intensity_factor);
            }
            return real_output(std::move(values), 1);
          },
          "flags"_a, "apertures"_a.noconvert(), "exponent"_a, "regions"_a.noconvert(),
          "intensity_factor"_a)
      .def(
          "planned_statistics",
          [](BoundState& self, const BoundStatisticsPlan& plan, const double z) {
            const auto& held = plan.plan();
            std::vector<double> values;
            {
              const nb::gil_scoped_release release;
              values = self.planned_statistics(held, z);
            }
            return real_output(std::move(values), 1);
          },
          "plan"_a, "z"_a)
      .def(
          "statistics",
          [](BoundState& self, const std::uint64_t flags, const RealInput windows,
             const RealInput on_axis, const double density, const double taper,
             const double preionisation, const double intensity_factor,
             const nb::object channel_preionisation, const nb::object power_weight) {
            const auto held_weight =
                power_weight.is_none()
                    ? std::vector<double>{}
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
              values = self.statistics(flags, held_windows, held_on_axis, density,
                                       taper, preionisation, intensity_factor, initial,
                                       held_weight);
            }
            return real_output(std::move(values), 1);
          },
          "flags"_a, "windows"_a.noconvert(), "on_axis"_a.noconvert(), "density"_a,
          "taper"_a, "preionisation"_a, "intensity_factor"_a,
          "channel_preionisation"_a = nb::none(), "power_weight"_a = nb::none())
      .def(
          "finish",
          [](BoundState& self, const double dz) {
            const nb::gil_scoped_release release;
            self.finish(dz);
          },
          "dz"_a)
      .def("promote_fsal", [](BoundState& self) { self.promote_fsal(); })
      .def("clear_fsal", [](BoundState& self) { self.clear_fsal(); })
      .def(
          "set_absorber_quiet",
          [](BoundState& self, const bool quiet) { self.set_absorber_quiet(quiet); },
          "quiet"_a)
      // solprop SEP 21: the boundary monitor's weights, and its five numbers and the
      // split's removal read where the state lives.
      .def(
          "bind_monitor",
          [](BoundState& self, const RealInput weights,
             const RealInput column_weights) {
            const auto held_weights = real_vector(weights);
            const auto held_columns = real_vector(column_weights);
            const nb::gil_scoped_release release;
            self.bind_monitor(held_weights, held_columns);
          },
          "weights"_a.noconvert(), "column_weights"_a.noconvert())
      .def("unbind_monitor", [](BoundState& self) { self.unbind_monitor(); })
      // solprop #440: the split collar's walk-off bucket of each bin, whose energy
      // per bucket a reading then returns beside the five.
      .def(
          "bind_occupancy",
          [](BoundState& self, const ByteInput buckets, const std::size_t count) {
            const auto count_in = element_count(buckets);
            const std::vector<unsigned char> held(buckets.data(),
                                                  buckets.data() + count_in);
            const nb::gil_scoped_release release;
            self.bind_occupancy(held, count);
          },
          "buckets"_a.noconvert(), "count"_a)
      .def("monitor_reading",
           [](BoundState& self) {
             std::vector<double> values;
             {
               const nb::gil_scoped_release release;
               values = self.monitor_reading();
             }
             return real_output(std::move(values), 1);
           })
      // solprop SEP 44: the bytes through the state's crossings, each way.
      .def("crossed_bytes",
           [](BoundState& self) {
             const auto crossed = self.crossed_bytes();
             return nb::make_tuple(crossed.to_host, crossed.from_host);
           })
      // solprop SEP 44: the transverse edge's reading, where the state lives.
      .def(
          "bind_edge",
          [](BoundState& self, const RealInput faces, const RealInput clip,
             const RealInput totals, const RealInput synthesis,
             const std::size_t first_row) {
            const auto held_faces = real_vector(faces);
            const auto held_clip = real_vector(clip);
            const auto held_totals = real_vector(totals);
            const auto held_synthesis = real_vector(synthesis);
            const nb::gil_scoped_release release;
            self.bind_edge(held_faces, held_clip, held_totals, held_synthesis,
                           first_row);
          },
          "faces"_a.noconvert(), "clip"_a.noconvert(), "totals"_a.noconvert(),
          "synthesis"_a.noconvert(), "first_row"_a)
      .def(
          "edge_reading",
          [](BoundState& self, const bool faces, const bool whole, const bool totals,
             const bool strip, const bool current) {
            native::EdgeNumbers numbers;
            {
              const nb::gil_scoped_release release;
              numbers = self.edge_reading(faces, whole, totals, strip, current);
            }
            return nb::make_tuple(real_output(std::move(numbers.sums), 1),
                                  real_output(std::move(numbers.columns), 1),
                                  real_output(std::move(numbers.strip), 1),
                                  numbers.current);
          },
          "faces"_a, "whole"_a, "totals"_a, "strip"_a, "current"_a)
      .def(
          "bind_edge_lines",
          [](BoundState& self, const IndexInput axes, const IndexInput counts,
             const IndexInput positions, const RealInput windows,
             const IndexInput inward, const RealInput factors, const RealInput measures,
             const IndexInput onsets, const RealInput steps, const double k_min,
             const RealInput scale, const std::size_t batch_bytes) {
            native::EdgeLines lines;
            lines.axes = index_vector(axes);
            lines.counts = index_vector(counts);
            lines.positions = index_vector(positions);
            lines.windows = real_vector(windows);
            lines.inward = index_vector(inward);
            lines.factors = real_vector(factors);
            lines.measures = real_vector(measures);
            lines.onsets = index_vector(onsets);
            lines.steps = real_vector(steps);
            lines.k_min = k_min;
            lines.scale = real_vector(scale);
            lines.batch_bytes = batch_bytes;
            // Holding the GIL: a strip's plan goes through FFTW's planner, which the
            // GIL serialises against another thread's backend being built.
            self.bind_edge_lines(lines);
          },
          "axes"_a.noconvert(), "counts"_a.noconvert(), "positions"_a.noconvert(),
          "windows"_a.noconvert(), "inward"_a.noconvert(), "factors"_a.noconvert(),
          "measures"_a.noconvert(), "onsets"_a.noconvert(), "steps"_a.noconvert(),
          "k_min"_a, "scale"_a.noconvert(), "batch_bytes"_a)
      .def(
          "set_crossing_workers",
          [](BoundState& self, const std::size_t workers) {
            self.set_crossing_workers(workers);
          },
          "workers"_a)
      .def("field_to_host",
           [](BoundState& self) {
             // The state crosses once, straight into the returned array's memory.
             const auto count = self.elements();
             auto storage = native::crossing::host_buffer(2 * count);
             {
               const nb::gil_scoped_release release;
               self.field_to_host(
                   {reinterpret_cast<native::Complex*>(storage.get()), count});
             }
             return complex_buffer_output(std::move(storage),
                                          {self.columns(), count / self.columns()});
           })
      .def(
          "stage_to_host",
          [](BoundState& self, const std::size_t stage) {
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
          "stage"_a)
      .def(
          "projected_to_host",
          [](BoundState& self, const bool coarse) {
            std::vector<native::Complex> values;
            {
              const nb::gil_scoped_release release;
              values = self.projected_to_host(coarse);
            }
            return complex_output(values,
                                  {self.columns(), values.size() / self.columns()});
          },
          "coarse"_a = false)
      .def_prop_ro("dispersive_path",
                   [](BoundState& self) { return self.dispersive_path(); })
      .def("memory_ledger",
           [](BoundState& self) {
             nb::dict result;
             for (const auto& [name, bytes] : self.memory_ledger()) {
               result[name.c_str()] = bytes;
             }
             return result;
           })
      .def_prop_ro("planning_seconds",
                   [](BoundState& self) { return self.planning_seconds(); })
      .def_prop_ro("raman_fft_provider",
                   [](BoundState& self) { return self.raman_fft_provider(); })
      .def_prop_ro("execution_sizing", &BoundState::execution_sizing)
      .def_prop_ro("cpu_execution", &BoundState::cpu_execution)
      .def_prop_ro("fft_planning", &BoundState::fft_planning)
      .def_prop_ro("raman_fft_tasks",
                   [](BoundState& self) { return self.raman_fft_tasks(); })
      .def_prop_ro("raman_planning_seconds",
                   [](BoundState& self) { return self.raman_planning_seconds(); })
      .def_prop_ro("raman_kernel_transforms",
                   [](BoundState& self) { return self.raman_kernel_transforms(); })
      .def_prop_ro("fsal_ready", [](BoundState& self) { return self.fsal_ready(); })
      .def_prop_ro("rk_source_evaluations",
                   [](BoundState& self) { return self.rk_source_evaluations(); })
      .def_prop_ro("quadrature_error",
                   [](BoundState& self) { return self.quadrature_error(); })
      .def("quadrature_peak", [](BoundState& self) { return self.quadrature_peak(); })
      .def_prop_ro("synthesis_provider",
                   [](BoundState& self) { return self.synthesis_provider(); })
      .def_prop_ro("projection_provider",
                   [](BoundState& self) { return self.projection_provider(); })
      .def_prop_ro("gemm_shape_policy",
                   [](BoundState& self) { return self.gemm_shape_policy(); })
      .def_prop_ro("response_team_size",
                   [](BoundState& self) { return self.response_team_size(); })
      .def_prop_ro("closed", &BoundState::closed)
      .def("close", &BoundState::close, nb::call_guard<nb::gil_scoped_release>());
  // solprop SEP 47: the native execution session and its compact event records.
  nb::exception<native::session::LifecycleError>(module_, "SessionLifecycleError",
                                                 PyExc_RuntimeError);
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
      .def_ro("attempts", &BatchRecord::attempts)
      .def_ro("row_z", &BatchRecord::row_z)
      .def_ro("row_step", &BatchRecord::row_step)
      .def_ro("row_values", &BatchRecord::row_values)
      .def_ro("plane_stats", &BatchRecord::plane_stats);
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
      .def("export", &BoundSession::export_field, "sequence"_a, "generation"_a,
           "power"_a, "optics"_a, "weight"_a.none(), "out"_a)
      .def("request_stop", &BoundSession::request_stop,
           nb::call_guard<nb::gil_scoped_release>())
      .def("request_snapshot", &BoundSession::request_snapshot)
      .def("status", &BoundSession::status)
      .def_prop_ro("elements", &BoundSession::elements)
      .def("monitor_record", &BoundSession::monitor_record)
      .def("continue_monitor", &BoundSession::continue_monitor, "record"_a, "plane"_a,
           "unwatched"_a, "edge_record"_a)
      .def("edge_record", &BoundSession::edge_record)
      .def("march_state", &BoundSession::march_state, "since"_a)
      .def("march_snapshot", &BoundSession::march_snapshot, "sequence"_a,
           "generation"_a)
      .def("table_phi", &BoundSession::table_phi, "sequence"_a, "generation"_a, "z"_a)
      .def("fail_outside", &BoundSession::fail_outside)
      .def("close", &BoundSession::close, nb::call_guard<nb::gil_scoped_release>());
  // solprop SEP 47 section 6.4: the owning boundary monitor alone, for solprop's tests
  // to feed recorded readings through it and its own monitor alike.
  nb::class_<native::watch::Monitor>(module_, "Monitor")
      .def(
          "__init__",
          [](native::watch::Monitor* self, const nb::dict config) {
            new (self) native::watch::Monitor(watch_config_from(config));
          },
          "config"_a)
      .def(
          "resume",
          [](native::watch::Monitor& self, const nb::dict record, const bool plane,
             const nb::handle unwatched) {
            self.resume(watch_resumption_from(record), plane,
                        watch_stretches_from(unwatched));
          },
          "record"_a, "plane"_a, "unwatched"_a)
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
  nb::class_<BoundBatch>(module_, "BatchState")
      .def(nb::init<const nb::list&>())
      .def(
          "accept",
          [](BoundBatch& self, const double dz) {
            std::vector<double> partials;
            {
              const nb::gil_scoped_release release;
              partials = self.accept(dz);
            }
            return real_output(std::move(partials), 3 * self.count());
          },
          "dz"_a)
      .def("memory_ledger",
           [](const BoundBatch& self) {
             nb::dict result;
             for (const auto& [name, bytes] : self.memory_ledger())
               result[name.c_str()] = bytes;
             return result;
           })
      .def_prop_ro("execution_sizing", &BoundBatch::execution_sizing)
      .def("close", &BoundBatch::close);
  if (Py_AtExit(&shutdown_after_interpreter) != 0)
    throw std::runtime_error(
        "cannot register the Kokkos runtime's shutdown at interpreter exit");
}
