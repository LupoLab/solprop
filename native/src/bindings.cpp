// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "helpers.hpp"
#include "solprop_native/core.hpp"
#include "solprop_native/vmath.hpp"

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
#include <nanobind/stl/vector.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

namespace native = solprop_native;
using ComplexInput =
    nb::ndarray<const std::complex<double>, nb::c_contig, nb::device::cpu>;
using RealInput = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using BoolInput = nb::ndarray<const bool, nb::c_contig, nb::device::cpu>;
using IndexInput = nb::ndarray<const std::uint64_t, nb::c_contig, nb::device::cpu>;
using ComplexOutput =
    nb::ndarray<nb::numpy, std::complex<double>, nb::c_contig, nb::device::cpu>;
using RealOutput = nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu>;
using ComplexStorage = std::vector<std::complex<double>>;
using RealStorage = std::vector<double>;

class BoundState;
class BoundBatch;

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

native::Problem problem_from(const nb::dict& values) {
  native::Problem result;
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
    result.cubic_real = nb::cast<bool>(values["cubic_real"]);
    result.take_real = nb::cast<bool>(values["take_real"]);
    if (values.contains("plasma")) {
      result.plasma = nb::cast<bool>(values["plasma"]);
      result.raman = nb::cast<bool>(values["raman"]);
      result.raman_thg = nb::cast<bool>(values["raman_thg"]);
      result.stash_plasma = nb::cast<bool>(values["stash_plasma"]);
      result.weighted_controller = nb::cast<bool>(values["weighted_controller"]);
      result.phase_b_engine = nb::cast<bool>(values["phase_b_engine"]);
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

// A scalar carrier frame's shift and rotations (solprop SEP 35), packed by solprop's
// adapter as (Im sigma, Re f_0, Im f_0, ..., Re f_{n-1}, Im f_{n-1}) over the `rows`
// history rows a formula reads; empty for none.
void carrier_frame(native::ExponentialWeights& weights, const RealInput carrier,
                   const std::size_t rows) {
  const auto values = real_vector(carrier);
  if (values.empty())
    return;
  if (rows > native::adams_max_samples || values.size() != 1 + 2 * rows)
    throw std::invalid_argument(
        "the carrier frame's rotations have inconsistent extents");
  weights.shifted = true;
  weights.shift = values[0];
  weights.unshift_re = std::cos(values[0]);
  weights.unshift_im = -std::sin(values[0]);
  for (std::size_t t = 0; t < rows; ++t) {
    weights.factor_re[t] = values[1 + 2 * t];
    weights.factor_im[t] = values[2 + 2 * t];
  }
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
                       const std::size_t threaded_fft_min_samples,
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
      {kernels, fft, raman_fft, raman_tail_fft, threaded_fft_min_samples, band_tiles});
}

class BoundState {
public:
  explicit BoundState(const nb::dict& problem) : BoundState(problem_from(problem)) {}
  explicit BoundState(native::Problem problem)
      : columns_(problem.n_columns()), stages_(problem.n_stages),
        raman_terms_(problem.n_raman_terms), raman_(problem.raman) {
    if (problem.needs_general_engine())
      general_state_ = std::make_shared<native::GeneralState>(std::move(problem));
    else
      state_ = std::make_unique<native::State>(std::move(problem));
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

  void set_initial(const std::vector<native::Complex>& field,
                   const std::vector<native::Complex>& noise) {
    require_open();
    if (general_state_)
      general_state_->set_initial(field, noise);
    else
      state_->set_initial(field, noise);
  }
  std::vector<double> trial_step(const double dz, const native::TrialScalars& scalars,
                                 const bool prepare_only = false) {
    require_open();
    if (general_state_) {
      if (prepare_only) {
        general_state_->prepare_trial(dz, scalars);
        return {};
      }
      return general_state_->trial_step(dz, scalars);
    }
    if (prepare_only)
      throw std::logic_error("batch preparation requires the general native state");
    const auto values = state_->trial_step(dz, scalars);
    return {values.begin(), values.end()};
  }
  void set_flow_actions(const std::vector<native::Complex>& forward,
                        const std::vector<native::Complex>& inverse,
                        const std::vector<unsigned char>& identity) {
    require_open();
    if (!general_state_)
      throw std::logic_error("a coupled flow requires the general native state");
    general_state_->set_flow_actions(forward, inverse, identity);
  }
  std::vector<native::Complex> flow_actions_to_host(const bool inverse) {
    require_open();
    if (!general_state_)
      throw std::logic_error("a coupled flow requires the general native state");
    return general_state_->flow_actions_to_host(inverse);
  }
  std::vector<double> form_flow_actions(const std::size_t section, const double dz,
                                        const std::vector<double>& fractions) {
    require_open();
    if (!general_state_)
      throw std::logic_error("a coupled flow requires the general native state");
    return general_state_->form_flow_actions(section, dz, fractions);
  }
  std::vector<double> weighted_error(const std::vector<double>& scale) {
    require_open();
    if (!general_state_)
      throw std::logic_error("weighted error requires the general native state");
    return general_state_->weighted_error(scale);
  }
  void prepare_raman(const std::vector<double>& rates,
                     const std::vector<double>& densities) {
    require_open();
    if (!general_state_)
      throw std::logic_error("Raman preparation requires the general native state");
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
    if (!general_state_)
      throw std::logic_error("statistics require the general native state");
    return general_state_->statistics(flags, windows, on_axis, density, taper,
                                      preionisation, intensity_factor,
                                      channel_preionisation, power_weight);
  }
  std::vector<double> spatial_statistics(const std::uint64_t flags,
                                         const double intensity_factor,
                                         const double z) {
    require_open();
    if (!general_state_)
      throw std::logic_error("spatial statistics require the general native state");
    return general_state_->spatial_statistics(flags, intensity_factor, z);
  }
  std::vector<double> aperture_statistics(const std::uint64_t flags,
                                          const std::vector<double>& apertures,
                                          const int profile_code, const int exponent,
                                          const std::vector<double>& regions,
                                          const double intensity_factor) {
    require_open();
    if (!general_state_)
      throw std::logic_error("far-field statistics require the general native state");
    return general_state_->aperture_statistics(flags, apertures, profile_code, exponent,
                                               regions, intensity_factor);
  }
  std::vector<native::Complex> field_to_host() {
    require_open();
    return general_state_ ? general_state_->field_to_host() : state_->field_to_host();
  }
  std::vector<native::Complex> adams_source(const std::vector<native::Complex>& input,
                                            const native::TrialScalars& scalars) {
    require_open();
    if (!general_state_)
      throw std::logic_error("the Adams source requires the general native state");
    return general_state_->adams_source(input, scalars);
  }
  void adopt(const std::vector<native::Complex>& field, const double dz) {
    require_open();
    if (!general_state_)
      throw std::logic_error("the Adams adoption requires the general native state");
    general_state_->adopt(field, dz);
  }
  native::GeneralState& low_storage_state() {
    require_open();
    if (!general_state_)
      throw std::logic_error("the low-storage march requires the general native state");
    return *general_state_;
  }
  native::GeneralState& general_state() {
    require_open();
    if (!general_state_)
      throw std::logic_error("a z table requires the general native state");
    return *general_state_;
  }
  native::GeneralState& adams_state() {
    require_open();
    if (!general_state_)
      throw std::logic_error(
          "the resident Adams history requires the general native state");
    return *general_state_;
  }
  std::vector<native::Complex> stage_to_host(const std::size_t stage) {
    require_open();
    if (!general_state_)
      throw std::logic_error("stage observation requires the general native state");
    return general_state_->stage_to_host(stage);
  }
  std::vector<native::Complex> projected_to_host(const bool coarse) {
    require_open();
    if (!general_state_)
      throw std::logic_error(
          "projection observation requires the general native state");
    return general_state_->projected_to_host(coarse);
  }
  std::map<std::string, bool> dispersive_path() const {
    return general_state_ ? general_state_->dispersive_path()
                          : std::map<std::string, bool>{};
  }
  void finish(const double dz) {
    require_open();
    if (general_state_)
      general_state_->finish(dz);
    else
      state_->finish(dz);
  }
  void promote_fsal() {
    require_open();
    if (general_state_)
      general_state_->promote_fsal();
    else
      state_->promote_fsal();
  }
  void clear_fsal() {
    require_open();
    if (general_state_)
      general_state_->clear_fsal();
    else
      state_->clear_fsal();
  }
  // The engines leave a quiet collar out where the reference does (solprop SEP 21).
  void set_absorber_quiet(const bool quiet) {
    require_open();
    if (general_state_)
      general_state_->set_absorber_quiet(quiet);
    else
      state_->set_absorber_quiet(quiet);
  }
  void bind_monitor(const std::vector<double>& weights,
                    const std::vector<double>& column_weights) {
    require_open();
    if (general_state_)
      general_state_->bind_monitor(weights, column_weights);
    else
      state_->bind_monitor(weights, column_weights);
  }
  void unbind_monitor() {
    require_open();
    if (general_state_)
      general_state_->unbind_monitor();
    else
      state_->unbind_monitor();
  }
  std::vector<double> monitor_reading() {
    require_open();
    return general_state_ ? general_state_->monitor_reading()
                          : state_->monitor_reading();
  }
  std::map<std::string, std::size_t> memory_ledger() {
    require_open();
    return general_state_ ? general_state_->memory_ledger() : state_->memory_ledger();
  }
  native::CpuSizing execution_sizing() const {
    require_open();
    return general_state_ ? general_state_->execution_sizing()
                          : state_->execution_sizing();
  }
  double planning_seconds() const {
    require_open();
    return general_state_ ? general_state_->planning_seconds()
                          : state_->planning_seconds();
  }
  double raman_planning_seconds() const {
    require_open();
    return general_state_ ? general_state_->raman_planning_seconds() : 0.0;
  }
  std::string raman_fft_provider() const {
    require_open();
    return general_state_ ? general_state_->raman_fft_provider() : "none";
  }
  std::map<std::string, int> raman_fft_tasks() const {
    require_open();
    return general_state_ ? general_state_->raman_fft_tasks()
                          : std::map<std::string, int>{};
  }
  std::size_t raman_kernel_transforms() const {
    require_open();
    return general_state_ ? general_state_->raman_kernel_transforms() : 0;
  }
  bool fsal_ready() const {
    require_open();
    return general_state_ ? general_state_->fsal_ready() : state_->fsal_ready();
  }
  std::size_t rk_source_evaluations() const {
    require_open();
    return general_state_ ? general_state_->rk_source_evaluations()
                          : state_->rk_source_evaluations();
  }
  double quadrature_error() const {
    require_open();
    return general_state_ ? general_state_->quadrature_error()
                          : Kokkos::Experimental::quiet_NaN_v<double>;
  }
  double quadrature_peak() {
    require_open();
    return general_state_ ? general_state_->quadrature_peak()
                          : Kokkos::Experimental::quiet_NaN_v<double>;
  }
  std::string synthesis_provider() const {
    require_open();
    return general_state_ ? general_state_->synthesis_provider() : "custom";
  }
  std::string projection_provider() const {
    require_open();
    return general_state_ ? general_state_->projection_provider() : "custom";
  }
  std::string gemm_shape_policy() const {
    require_open();
    return general_state_ ? general_state_->gemm_shape_policy()
                          : "mode-averaged 1 x 1 transforms use custom scaling";
  }
  std::size_t response_team_size() const {
    require_open();
    return general_state_ ? general_state_->response_team_size()
                          : SOLPROP_NATIVE_DEVICE_TEAM_SIZE_RECORD;
  }
  std::shared_ptr<native::GeneralState> batch_member() {
    require_open();
    if (!general_state_)
      throw std::logic_error("batch execution requires the general native state");
    return general_state_;
  }
  std::size_t columns() const noexcept { return columns_; }
  std::size_t stages() const noexcept { return stages_; }
  std::size_t raman_terms() const noexcept { return raman_terms_; }
  bool raman() const noexcept { return raman_; }
  bool closed() const noexcept {
    return (!state_ || state_->closed()) &&
           (!general_state_ || general_state_->closed());
  }
  void close() noexcept {
    if (state_) {
      state_->close();
      state_.reset();
    }
    if (general_state_) {
      general_state_->close();
      general_state_.reset();
    }
  }

private:
  void require_open() const {
    if (closed())
      throw std::logic_error("native state is closed");
  }
  std::size_t columns_{};
  std::size_t stages_{};
  std::size_t raman_terms_{};
  bool raman_{};
  std::unique_ptr<native::State> state_;
  std::shared_ptr<native::GeneralState> general_state_;
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
// FFTW's process-wide planner. Any other user of the same FFTW library (pyFFTW in
// solprop's Numba backend) must have destroyed its plans by then, so the owned
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

nb::dict build_record() {
  const std::scoped_lock lock(runtime_state.mutex);
  nb::dict result;
  result["backend_commit"] = SOLPROP_NATIVE_BACKEND_COMMIT;
  result["solprop_built_against"] = SOLPROP_NATIVE_SOLPROP_BUILT_AGAINST;
  result["schema_version"] = 38;
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
  result["threaded_fft_min_samples"] = threads.threaded_fft_min_samples;
  result["cpu_size_policy_enabled"] = native::cpu_size_policy_enabled;
  result["explicit_thread_pools"] = threads.explicit_pools;
  result["stage_thread_min_elements"] = native::stage_thread_min_elements;
  result["copy_thread_min_elements"] = native::copy_thread_min_elements;
  result["flow_thread_min_elements"] = native::flow_thread_min_elements;
  result["modal_thread_min_elements"] = native::modal_thread_min_elements;
  result["response_thread_min_elements"] = native::response_thread_min_elements;
  result["dense_thread_min_elements"] = native::dense_thread_min_elements;
  result["raman_thread_min_elements"] = native::raman_thread_min_elements;
  result["stage_full_thread_elements"] = native::stage_full_thread_elements;
  result["response_full_thread_elements"] = native::response_full_thread_elements;
  result["response_full_thread_samples"] = native::response_full_thread_samples;
  result["response_full_thread_columns"] = native::response_full_thread_columns;
  result["dense_full_thread_elements"] = native::dense_full_thread_elements;
  result["fft_full_thread_elements"] = native::fft_full_thread_elements;
  result["raman_full_thread_elements"] = native::raman_full_thread_elements;
  result["cpu_medium_threads"] = native::cpu_medium_threads;
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
  module_.doc() = "Private nanobind seam for solprop-native";
  module_.def("initialize_runtime", &initialize_kokkos, "threads"_a, "kernels"_a = 0,
              "fft"_a = 0, "raman_fft"_a = 0, "raman_tail_fft"_a = 0,
              "threaded_fft_min_samples"_a = 0, "band_tiles"_a = -1);
  module_.def("fit_blas_threads", &fit_blas_threads, "threads"_a);
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
  module_.def("shutdown_runtime", &shutdown_kokkos);
  module_.def("build_record", &build_record);
  native::bindings::bind_helpers(module_);
  nb::class_<BoundState>(module_, "State")
      .def(nb::init<const nb::dict&>())
      .def(
          "set_initial",
          [](BoundState& self, const ComplexInput field, const ComplexInput noise) {
            const auto held_field = complex_vector(field);
            const auto held_noise = complex_vector(noise);
            const nb::gil_scoped_release release;
            self.set_initial(held_field, held_noise);
          },
          "field"_a.noconvert(), "noise"_a.noconvert())
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
            auto& state = self.low_storage_state();
            const nb::gil_scoped_release release;
            return state.low_storage_step(dz, scalars, rebuild_table);
          },
          "dz"_a, "phi_delta"_a.noconvert(), "phi_at"_a.noconvert(),
          "prefactor"_a.noconvert(), "density"_a.noconvert(), "taper"_a.noconvert(),
          "preionisation"_a.noconvert(), "raman_rates"_a.noconvert(), "material"_a,
          "rebuild_table"_a)
      .def("hold_plane",
           [](BoundState& self) {
             auto& state = self.low_storage_state();
             const nb::gil_scoped_release release;
             state.hold_plane();
           })
      .def("held_to_host",
           [](BoundState& self) {
             auto& state = self.low_storage_state();
             std::vector<native::Complex> values;
             {
               const nb::gil_scoped_release release;
               values = state.held_to_host();
             }
             return complex_output(values,
                                   {self.columns(), values.size() / self.columns()});
           })
      // solprop SEP 16 phase 3b: the Adams history resident in the engine.
      .def(
          "adams_allocate",
          [](BoundState& self, const std::size_t slots, const bool compensated,
             const bool interpolate) {
            self.adams_state().adams_allocate(slots, compensated, interpolate);
          },
          "slots"_a, "compensated"_a = false, "interpolate"_a = true)
      .def(
          "adams_compensation_start",
          [](BoundState& self, const std::size_t slot, const RealInput rate) {
            const auto weights = real_vector(rate);
            const nb::gil_scoped_release release;
            self.adams_state().adams_compensation_start(slot, weights);
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
            self.adams_state().adams_start(slot, scalars);
          },
          "slot"_a, "phi_delta"_a.noconvert(), "phi_at"_a.noconvert(),
          "prefactor"_a.noconvert(), "density"_a.noconvert(), "taper"_a.noconvert(),
          "preionisation"_a.noconvert(), "raman_rates"_a.noconvert(), "material"_a)
      .def("adams_magnitude",
           [](BoundState& self) {
             const nb::gil_scoped_release release;
             return self.adams_state().adams_magnitude();
           })
      .def(
          "adams_scale",
          [](BoundState& self, const double rtol, const double atol,
             const RealInput weights) {
            const auto values = real_vector(weights);
            const nb::gil_scoped_release release;
            self.adams_state().adams_scale(rtol, atol, values);
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
              norms = self.adams_state().adams_fused_trial(dz, weights, scale, scalars);
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
            self.adams_state().adams_fused_accept(dz, kept, free, slopes);
          },
          "dz"_a, "keep"_a.noconvert(), "free"_a, "dense"_a.noconvert(),
          "interpolate"_a)
      .def("adams_hold", [](BoundState& self) { self.adams_state().adams_hold(); })
      .def("adams_restore",
           [](BoundState& self) { self.adams_state().adams_restore(); })
      // solprop SEP 29 G2: the resident history's persistent rows across a resume.
      .def("adams_row_names",
           [](BoundState& self) { return self.adams_state().adams_row_names(); })
      .def(
          "adams_rows_to_host",
          [](BoundState& self, const std::string& name) {
            auto& state = self.adams_state();
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
            auto& state = self.adams_state();
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
            auto& state = self.general_state();
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
            auto& state = self.general_state();
            const nb::gil_scoped_release release;
            state.table_evaluate(held_slots, held_positions);
          },
          "slots"_a.noconvert(), "positions"_a.noconvert())
      .def(
          "table_rate",
          [](BoundState& self, const std::size_t slot, const double z,
             const bool before) {
            auto& state = self.general_state();
            const nb::gil_scoped_release release;
            state.table_rate(slot, z, before);
          },
          "slot"_a, "z"_a, "before"_a)
      .def(
          "table_terms_to_host",
          [](BoundState& self, const std::string& kind, const std::size_t slot) {
            auto& state = self.general_state();
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
            auto& state = self.general_state();
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
            self.adams_state().adams_observe(table, hermite, scalars);
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
              norms = self.adams_state().adams_exponential_trial(
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
      .def(
          "adams_exponential_accept",
          [](BoundState& self, const double dz, const std::size_t free,
             const IndexInput dense, const bool interpolate) {
            auto slopes = adams_rows(dense);
            slopes.width = interpolate ? slopes.width + 1 : 0;
            const nb::gil_scoped_release release;
            self.adams_state().adams_exponential_accept(dz, free, slopes);
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
            self.adams_state().adams_exponential_observe(xi, dz, weights);
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
          "aperture_statistics",
          [](BoundState& self, const std::uint64_t flags, const RealInput apertures,
             const int profile_code, const int exponent, const RealInput regions,
             const double intensity_factor) {
            if (apertures.ndim() != 2 || regions.ndim() != 2 || regions.shape(1) != 4)
              throw std::invalid_argument(
                  "aperture tables must be (n_apertures, 6 + n_omega) and regions "
                  "(n_regions, 4)");
            const auto held_apertures = real_vector(apertures);
            const auto held_regions = real_vector(regions);
            std::vector<double> values;
            {
              const nb::gil_scoped_release release;
              values =
                  self.aperture_statistics(flags, held_apertures, profile_code,
                                           exponent, held_regions, intensity_factor);
            }
            return real_output(std::move(values), 1);
          },
          "flags"_a, "apertures"_a.noconvert(), "profile_code"_a, "exponent"_a,
          "regions"_a.noconvert(), "intensity_factor"_a)
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
      .def("monitor_reading",
           [](BoundState& self) {
             std::vector<double> values;
             {
               const nb::gil_scoped_release release;
               values = self.monitor_reading();
             }
             return real_output(std::move(values), 1);
           })
      .def("field_to_host",
           [](BoundState& self) {
             std::vector<native::Complex> values;
             {
               const nb::gil_scoped_release release;
               values = self.field_to_host();
             }
             return complex_output(values,
                                   {self.columns(), values.size() / self.columns()});
           })
      .def(
          "stage_to_host",
          [](BoundState& self, const std::size_t stage) {
            std::vector<native::Complex> values;
            {
              const nb::gil_scoped_release release;
              values = self.stage_to_host(stage);
            }
            return complex_output(values,
                                  {self.columns(), values.size() / self.columns()});
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
      .def("close", &BoundState::close);
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
