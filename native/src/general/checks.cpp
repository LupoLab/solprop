// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// What the general workspace refuses before it allocates, and a trial's scalars.

#include "workspace.hpp"

namespace solprop_native {

void require_general_problem(const Problem& problem) {
  require_general_modal(problem);
  require_coupled_flow(problem);
  require_dispersive(problem);
  if (problem.dims.n_omega == 0 || problem.dims.n_t == 0 || problem.dims.n_to == 0 ||
      problem.dims.n_modes == 0 || problem.dims.n_pol_state == 0 ||
      problem.dims.n_nodes == 0 || problem.dims.n_pol_transform == 0)
    throw std::invalid_argument("all transform and FFT extents must be positive");
  const auto effective_node_pol =
      checked_product(problem.dims.n_pol_transform, problem.dims.n_pol_state,
                      "effective node polarisation");
  if (effective_node_pol > 2)
    throw std::invalid_argument("at most two effective node components are supported");
  if (problem.dims.node_chunk == 0 || problem.dims.node_chunk > problem.dims.n_nodes)
    throw std::invalid_argument("node_chunk must lie in [1, n_nodes]");
  if (problem.stepper.n_stages == 0)
    throw std::invalid_argument("the tableau must contain a stage");
  const auto columns =
      checked_product(problem.dims.n_modes, problem.dims.n_pol_state, "field columns");
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr auto policy_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (problem.dims.n_to > policy_limit || problem.dims.n_t > policy_limit ||
      problem.dims.n_omega > policy_limit || columns > policy_limit ||
      problem.dims.node_chunk > policy_limit)
    throw std::length_error("accelerator policy extent exceeds signed 32-bit range");
#endif
  const auto rows = checked_product(problem.dims.n_nodes, problem.dims.n_pol_transform,
                                    "transform rows");
  const auto field = checked_product(columns, problem.dims.n_omega, "field");
  const auto modal_time = checked_product(columns, problem.dims.n_to, "modal time");
  const auto chunk_rows =
      checked_product(problem.dims.node_chunk, effective_node_pol, "node chunk rows");
  const auto node_time = checked_product(chunk_rows, problem.dims.n_to, "node time");
  // The hybrid planar kind carries its x half alone, (n_x, n_p) and its transpose.
  const auto transform =
      problem.free_geometry() ? 0
      : problem.slab_geometry()
          ? checked_product(problem.spatial.n_x, problem.spatial.n_p,
                            "slab transform matrix")
          : checked_product(rows, problem.dims.n_modes, "transform matrix");
  const auto stage_field =
      checked_product(problem.stepper.n_stages, field, "stage field");
  require_allocation(field, sizeof(Complex), "field bytes");
  require_allocation(modal_time, sizeof(Complex), "modal-time bytes");
  require_allocation(node_time, sizeof(Complex), "node-time bytes");
  require_allocation(transform, sizeof(double), "transform-matrix bytes");
  if (!problem.transform.projection_coarse_matrix.empty() &&
      problem.dims.family != FieldFamily::envelope) {
    const auto stacked_rows =
        checked_product(2, problem.dims.n_modes, "stacked projection rows");
    require_allocation(
        checked_product(stacked_rows, problem.dims.n_to, "stacked projection"),
        sizeof(double), "stacked-projection bytes");
  }
  require_allocation(stage_field, sizeof(Complex), "stage-field bytes");
  if (problem.plasma.enabled && !problem.plasma.envelope) {
    require_allocation(
        checked_product(chunk_rows, problem.dims.n_omega, "held nodal spectrum"),
        sizeof(Complex), "held nodal spectrum bytes");
    require_allocation(checked_product(2, modal_time, "held stacked projection"),
                       sizeof(double), "held stacked projection bytes");
    require_allocation(checked_product(problem.dims.n_modes,
                                       checked_product(problem.dims.node_chunk,
                                                       problem.dims.n_pol_transform,
                                                       "held projection rows"),
                                       "held projection"),
                       sizeof(double), "held projection bytes");
  }
  require_extent(problem.grid.crop_index.size(), problem.dims.n_omega, "crop_index");
  require_extent(problem.grid.active_mask.size(), problem.dims.n_omega, "active_mask");
  require_extent(problem.grid.norm_mask.size(), problem.dims.n_omega, "norm_mask");
  require_extent(problem.grid.omega_window.size(), problem.dims.n_omega,
                 "omega_window");
  if (!problem.grid.omega.empty())
    require_extent(problem.grid.omega.size(), problem.dims.n_omega, "omega");
  if (!problem.spatial.spectral_weights.empty()) {
    require_extent(problem.spatial.spectral_weights.size(), problem.dims.n_modes,
                   "spatial_spectral_weights");
    require_extent(problem.spatial.on_axis.size(), problem.dims.n_modes,
                   "spatial_on_axis");
    if ((!problem.slab_geometry() && problem.dims.n_nodes != problem.dims.n_modes) ||
        problem.dims.n_pol_transform != 1)
      throw std::invalid_argument(
          "radial statistics require a scalar square transform");
    for (const double weight : problem.spatial.spectral_weights)
      if (!std::isfinite(weight) || weight <= 0.0)
        throw std::invalid_argument(
            "spatial spectral weights must be finite and positive");
  } else if (!problem.spatial.on_axis.empty()) {
    throw std::invalid_argument("spatial on-axis weights require the spectral measure");
  }
  if (problem.slab_geometry()) {
    // solprop's hybrid planar kind: columns (p, k_y), nodes (x, y), a scalar state,
    // whole y rows per node chunk and no embedded coarse rule (solprop's reference
    // scores no quadrature alarm on this kind).
    if (problem.spatial.n_x < 1 || problem.spatial.n_y < 1 ||
        checked_product(problem.spatial.n_p, problem.spatial.n_y, "planar columns") !=
            problem.dims.n_modes ||
        checked_product(problem.spatial.n_x, problem.spatial.n_y, "planar nodes") !=
            problem.dims.n_nodes ||
        problem.dims.n_pol_transform != 1 || problem.dims.n_pol_state != 1 ||
        problem.dims.node_chunk % problem.spatial.n_x != 0 ||
        problem.linear.n_phi_terms != 1 || problem.linear.n_prefactor_terms != 1 ||
        !problem.transform.projection_coarse_matrix.empty())
      throw std::invalid_argument(
          "the hybrid planar kind requires matching axes, a scalar state, whole y "
          "rows per node chunk and scalar separable data");
    require_extent(
        problem.spatial.kappa_sq.size(),
        checked_product(problem.spatial.n_p, problem.dims.n_omega, "planar wall terms"),
        "spatial_kappa_sq");
    for (const auto value : problem.spatial.kappa_sq)
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument("spatial_kappa_sq must be finite");
  } else if (!problem.spatial.kappa_sq.empty()) {
    throw std::invalid_argument("spatial_kappa_sq belongs to the hybrid planar kind");
  }
  // n_y == 1 is solprop's line uniform in y (grid.LineGrid): one y row carrying
  // k_y = 0, transformed by the same two-axis plan with a unit extent.
  if (problem.free_geometry() &&
      (problem.spatial.n_x < 2 || problem.spatial.n_y < 1 ||
       checked_product(problem.spatial.n_x, problem.spatial.n_y, "Cartesian shape") !=
           problem.dims.n_modes ||
       problem.dims.n_nodes != problem.dims.n_modes ||
       problem.dims.n_pol_transform != 1 || problem.linear.n_phi_terms != 1 ||
       problem.linear.n_prefactor_terms != 1))
    throw std::invalid_argument(
        "Cartesian geometry requires matching axes and scalar separable data");
  if (problem.separable_geometry()) {
    if (!std::isfinite(problem.spatial.evanescent_rate) ||
        problem.spatial.evanescent_rate <= 0)
      throw std::invalid_argument(
          "Cartesian evanescent rate must be finite and positive");
    require_extent(problem.spatial.k_squared.size(), problem.dims.n_omega,
                   "spatial_k_squared");
    require_extent(problem.spatial.k_squared_low.size(), problem.dims.n_omega,
                   "spatial_k_squared_low");
    require_extent(problem.spatial.numerator.size(), problem.dims.n_omega,
                   "spatial_numerator");
    require_extent(problem.spatial.clamped.size(), problem.dims.n_omega,
                   "spatial_clamped");
    require_extent(problem.spatial.frame.size(), problem.dims.n_omega, "spatial_frame");
    require_extent(problem.spatial.spectral_rate.size(), problem.dims.n_omega,
                   "spatial_spectral_rate");
    require_extent(problem.spatial.user_rate.size(), problem.dims.n_omega,
                   "spatial_user_rate");
    require_extent(problem.spatial.kperp_sq.size(), problem.dims.n_modes,
                   "spatial_kperp_sq");
    require_extent(problem.spatial.kperp_sq_low.size(), problem.dims.n_modes,
                   "spatial_kperp_sq_low");
    require_extent(problem.spatial.absorber_rate.size(), problem.dims.n_modes,
                   "spatial_absorber_rate");
    require_extent(problem.spatial.kperp_rate.size(), problem.dims.n_modes,
                   "spatial_kperp_rate");
    for (const auto value : problem.spatial.k_squared)
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument("spatial_k_squared must be finite");
    for (const auto value : problem.spatial.k_squared_low)
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument("spatial_k_squared_low must be finite");
    for (const auto value : problem.spatial.numerator)
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument("spatial_numerator must be finite");
    for (const auto value : problem.spatial.clamped)
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument("spatial_clamped must be finite");
    for (const auto value : problem.spatial.frame)
      if (!std::isfinite(value))
        throw std::invalid_argument("spatial_frame is invalid");
    for (const auto value : problem.spatial.spectral_rate)
      if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("spatial_spectral_rate is invalid");
    for (std::size_t bin = 0; bin < problem.dims.n_omega; ++bin)
      if (const double value = problem.spatial.user_rate[bin];
          !std::isfinite(value) || value < 0 ||
          value > problem.spatial.spectral_rate[bin])
        throw std::invalid_argument("spatial_user_rate is invalid");
    for (const auto value : problem.spatial.kperp_sq)
      if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("spatial_kperp_sq is invalid");
    for (const auto value : problem.spatial.kperp_sq_low)
      if (!std::isfinite(value))
        throw std::invalid_argument("spatial_kperp_sq_low must be finite");
    for (const auto value : problem.spatial.absorber_rate)
      if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("spatial_absorber_rate is invalid");
    for (const auto value : problem.spatial.kperp_rate)
      if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("spatial_kperp_rate is invalid");
    if (problem.free_geometry()) {
      // solprop SEP 23: the stored transverse axes the far-field statistics read.
      require_extent(problem.spatial.k_x.size(), problem.spatial.n_x, "spatial_k_x");
      require_extent(problem.spatial.k_y.size(), problem.spatial.n_y, "spatial_k_y");
      for (const auto value : problem.spatial.k_x)
        if (!std::isfinite(value))
          throw std::invalid_argument("spatial_k_x must be finite");
      for (const auto value : problem.spatial.k_y)
        if (!std::isfinite(value))
          throw std::invalid_argument("spatial_k_y must be finite");
    }
  } else if (problem.spatial.n_y != 0 || problem.spatial.n_p != 0) {
    throw std::invalid_argument("Cartesian axes must be supplied together");
  }
  if (!problem.free_geometry() &&
      (!problem.spatial.k_x.empty() || !problem.spatial.k_y.empty() ||
       problem.spatial.far_field_slots != 0))
    throw std::invalid_argument("the far-field statistics belong to free space");
  require_extent(problem.grid.time_window.size(), problem.dims.n_to, "time_window");
  if (problem.plasma.enabled && !problem.plasma.envelope) {
    require_extent(problem.grid.time_window_derivative.size(), problem.dims.n_to,
                   "time_window_derivative");
    require_extent(problem.grid.omega.size(), problem.dims.n_omega,
                   "held current omega");
    for (const auto value : problem.grid.time_window_derivative)
      if (!std::isfinite(value))
        throw std::invalid_argument("time_window_derivative is not finite");
    for (std::size_t index = 0; index < problem.dims.n_omega; ++index)
      if (problem.grid.active_mask[index] != 0 &&
          (!(problem.grid.omega[index] > 0.0) ||
           !std::isfinite(problem.grid.omega[index])))
        throw std::invalid_argument(
            "held plasma current requires positive active omega");
  }
  require_extent(problem.grid.carrier.size(), problem.dims.n_to, "carrier");
  require_extent(problem.transform.synthesis_matrix.size(), transform,
                 "synthesis_matrix");
  require_extent(problem.transform.projection_matrix.size(), transform,
                 "projection_matrix");
  if (!problem.transform.projection_coarse_matrix.empty())
    require_extent(problem.transform.projection_coarse_matrix.size(), transform,
                   "projection_coarse_matrix");
  // A z table's terms are slots its evaluation fills, so nothing crosses for them.
  if ((problem.linear.phi_table || problem.linear.prefactor_table) &&
      problem.separable_geometry())
    throw std::invalid_argument("a separable operator takes no z table");
  require_extent(problem.linear.phi_terms.size(),
                 problem.separable_geometry() || problem.linear.phi_table
                     ? 0
                     : checked_product(problem.linear.n_phi_terms, field, "phi_terms"),
                 "phi_terms");
  require_extent(
      problem.linear.prefactor_terms.size(),
      problem.separable_geometry() || problem.linear.prefactor_table
          ? 0
          : checked_product(problem.linear.n_prefactor_terms, field, "prefactor_terms"),
      "prefactor_terms");
  require_extent(
      problem.stepper.tableau_matrix.size(),
      checked_product(problem.stepper.n_stages, problem.stepper.n_stages, "tableau"),
      "tableau_matrix");
  require_extent(problem.stepper.tableau_nodes.size(), problem.stepper.n_stages,
                 "tableau_nodes");
  require_extent(problem.stepper.solution_weights.size(), problem.stepper.n_stages,
                 "solution_weights");
  require_extent(problem.stepper.error_weights.size(), problem.stepper.n_stages,
                 "error_weights");
  if (problem.absorber.in_stage)
    require_extent(problem.absorber.fine.size(), problem.dims.n_to, "absorber_fine");
  if (!problem.spatial.absorber_matrix.empty()) {
    require_extent(
        problem.spatial.absorber_matrix.size(),
        checked_product(problem.dims.n_modes, problem.dims.n_modes, "spatial absorber"),
        "spatial_absorber_matrix");
    if (!problem.absorber.in_stage || problem.dims.n_pol_transform != 1 ||
        problem.dims.n_nodes != problem.dims.n_modes)
      throw std::invalid_argument("spatial absorber requires scalar radial geometry");
  }
  if (problem.absorber.split_step)
    require_extent(problem.absorber.coarse.size(), problem.dims.n_t, "absorber_coarse");
  const std::size_t fine_size = problem.dims.family == FieldFamily::field_real
                                    ? problem.dims.n_to / 2 + 1
                                    : problem.dims.n_to;
  if (problem.plasma.envelope && problem.dims.family != FieldFamily::envelope)
    throw std::invalid_argument("envelope plasma requires an envelope field family");
  std::vector<bool> seen(fine_size, false);
  for (const auto index : problem.grid.crop_index) {
    if (index >= fine_size)
      throw std::invalid_argument("crop_index lies outside the fine spectrum");
    if (seen[index])
      throw std::invalid_argument("crop_index contains a duplicate fine-spectrum bin");
    seen[index] = true;
  }
  if (problem.plasma.enabled && problem.plasma.n_channels == 0)
    throw std::invalid_argument("plasma requires at least one channel");
  if (problem.plasma.n_channels > 1 || problem.plasma.envelope) {
    if (!problem.plasma.enabled)
      throw std::invalid_argument("channels require plasma");
    require_extent(problem.plasma.channel_parameters.size(),
                   checked_product(problem.plasma.n_channels, 9, "channel parameters"),
                   "channel_parameters");
    if (!std::ranges::all_of(problem.plasma.channel_parameters,
                             [](double value) { return std::isfinite(value); }) ||
        !std::ranges::all_of(problem.plasma.channel_coefficients,
                             [](double value) { return std::isfinite(value); }))
      throw std::invalid_argument("channel parameters and coefficients must be finite");
    std::size_t expected_offset = 0;
    // Each interval holds four coefficients.
    const std::size_t total_intervals = problem.plasma.channel_coefficients.size() / 4;
    for (std::size_t channel = 0; channel < problem.plasma.n_channels; ++channel) {
      const double* row = problem.plasma.channel_parameters.data() + 9 * channel;
      if (!(row[1] > 0.0 && row[2] > 0.0 && row[3] > row[2] && row[4] >= 0.0 &&
            row[5] >= 1.0 && row[5] == std::floor(row[5]) &&
            row[6] == static_cast<double>(expected_offset) && row[7] > 0.0 &&
            row[8] > 0.0) ||
          row[5] > static_cast<double>(total_intervals))
        throw std::invalid_argument(
            "channel constants and coefficient ranges must be positive and consistent");
      expected_offset = checked_sum(
          expected_offset,
          checked_product(4, static_cast<std::size_t>(row[5]), "channel table"),
          "channel table offsets");
    }
    require_extent(problem.plasma.channel_coefficients.size(), expected_offset,
                   "channel_coefficients");
  }
  if (problem.raman.enabled && problem.raman.n_components == 0)
    throw std::invalid_argument("Raman requires at least one component");
  if (problem.raman.n_components > 1) {
    if (!problem.raman.enabled)
      throw std::invalid_argument("Raman components require Raman");
    require_extent(problem.raman.row_component.size(), problem.raman.n_terms,
                   "raman_row_component");
    require_extent(problem.raman.component_fraction.size(), problem.raman.n_components,
                   "raman_component_fraction");
    for (double value : problem.raman.row_component)
      if (!std::isfinite(value) || value < 0.0 ||
          value >= static_cast<double>(problem.raman.n_components) ||
          value != std::floor(value))
        throw std::invalid_argument("Raman row component index is invalid");
    for (double value : problem.raman.component_fraction)
      if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(
            "Raman component fractions must be finite and positive");
  }
  if (problem.plasma.enabled) {
    if (!(problem.grid.dt_fine > 0.0) || !(problem.plasma.d_log_field > 0.0) ||
        !(problem.plasma.field_min > 0.0) ||
        !(problem.plasma.field_max > problem.plasma.field_min) ||
        !(problem.plasma.charge_mass_ratio > 0.0) ||
        !(problem.plasma.ionisation_potential > 0.0) ||
        problem.plasma.rate_intervals == 0)
      throw std::invalid_argument("plasma constants must be positive and consistent");
    require_extent(
        problem.plasma.rate_coefficients.size(),
        checked_product(4, problem.plasma.rate_intervals, "rate coefficients"),
        "rate_coefficients");
  }
  // solprop SEP 18's running-integral rule. Translation certified the taps; only their
  // extent, finiteness, the trapezoid's identity tap and the route are checked here.
  if (problem.plasma.taps.empty() || problem.plasma.taps.size() > plasma_max_taps ||
      !std::ranges::all_of(problem.plasma.taps,
                           [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("plasma_taps must hold one to nine finite taps");
  if (!problem.plasma.band &&
      (problem.plasma.taps.size() != 1 || problem.plasma.taps[0] != 1.0))
    throw std::invalid_argument("the trapezoid rule is the one identity tap");
  if (problem.plasma.gated &&
      (!problem.plasma.enabled || problem.plasma.envelope || problem.general.enabled))
    throw std::invalid_argument(
        "the gated plasma benchmark needs a field plasma column off the general route");
  if (problem.plasma.band) {
    if (!problem.plasma.enabled || problem.plasma.envelope ||
        problem.dims.family == FieldFamily::envelope)
      throw std::invalid_argument(
          "the band integration rule needs a field-grid plasma column");
    if (problem.dims.n_to <= problem.plasma.half_width())
      throw std::invalid_argument(
          "the band integration rule's filter needs more than K samples");
  }
  if (problem.plasma.envelope) {
    if (!problem.plasma.enabled || problem.dims.family != FieldFamily::envelope ||
        problem.kerr.harmonic_factor != 0.0 || problem.raman.thg ||
        problem.plasma.stash)
      throw std::invalid_argument("envelope plasma requires separate envelope currents "
                                  "without THG or field stashing");
    const auto rule_size = checked_product(problem.plasma.n_channels,
                                           std::size_t{4} * 512, "envelope rules");
    require_extent(problem.plasma.envelope_rule_nodes.size(), rule_size,
                   "envelope_rule_nodes");
    require_extent(problem.plasma.envelope_rule_weights.size(), rule_size,
                   "envelope_rule_weights");
    for (std::size_t channel = 0; channel < problem.plasma.n_channels; ++channel)
      for (std::size_t level = 0; level < 4; ++level) {
        const auto order = std::size_t{64} << level;
        double sum = 0.0;
        double previous = -1.0;
        for (std::size_t index = 0; index < 512; ++index) {
          const auto offset = (channel * 4 + level) * 512 + index;
          const auto node = problem.plasma.envelope_rule_nodes[offset];
          const auto weight = problem.plasma.envelope_rule_weights[offset];
          if (!std::isfinite(node) || !std::isfinite(weight) ||
              (index < order ? !(node > previous && node < 1.0 && weight > 0.0)
                             : node != 0.0 || weight != 0.0))
            throw std::invalid_argument(
                "envelope quadrature nodes or weights are invalid");
          if (index < order) {
            previous = node;
            sum += weight;
          }
        }
        if (std::abs(sum - 2.0) >
            8 * static_cast<double>(order) * std::numeric_limits<double>::epsilon())
          throw std::invalid_argument(
              "envelope quadrature weights must integrate a constant");
      }
    require_extent(problem.plasma.envelope_drude_factor.size(), problem.dims.n_omega,
                   "envelope_drude_factor");
    require_extent(problem.plasma.envelope_loss_factor.size(), problem.dims.n_omega,
                   "envelope_loss_factor");
    for (std::size_t index = 0; index < problem.dims.n_omega; ++index) {
      const auto drude = problem.plasma.envelope_drude_factor[index];
      const auto loss = problem.plasma.envelope_loss_factor[index];
      const bool active =
          problem.grid.active_mask[index] != 0 && problem.grid.omega[index] > 0.0;
      if (!std::isfinite(drude) || !std::isfinite(loss.real()) ||
          !std::isfinite(loss.imag()) || loss.real() != 0.0 ||
          (active ? !(drude < 0.0 && loss.imag() < 0.0)
                  : drude != 0.0 || loss.imag() != 0.0))
        throw std::invalid_argument("envelope current factors require negative active "
                                    "factors and zero inactive bins");
    }
  }
  if (problem.raman.enabled) {
    if (effective_node_pol != 1 && !problem.raman.tensor_capable)
      throw std::invalid_argument(
          "vector Raman requires tensor-capable terms; found scalar_only");
    if (problem.raman.n_terms == 0)
      throw std::invalid_argument("a Raman problem must contain at least one term");
    require_extent(problem.raman.tensor_weights.size(),
                   checked_product(problem.raman.n_terms, 2, "Raman tensor weights"),
                   "raman_tensor_weights");
    for (std::size_t term = 0; term < problem.raman.n_terms; ++term) {
      const double trace = problem.raman.tensor_weights[2 * term];
      const double traceless = problem.raman.tensor_weights[2 * term + 1];
      if (!((trace == 1.0 && traceless == 0.0) || (trace == 0.25 && traceless == 0.75)))
        throw std::invalid_argument(
            "Raman tensor weights must be isotropic or rank two");
    }
    const auto doubled = checked_product(2, problem.dims.n_to, "Raman doubled grid");
    const auto raman_rows = checked_product(problem.dims.node_chunk,
                                            problem.raman_drives(), "Raman drive rows");
    require_allocation(checked_product(raman_rows, doubled, "Raman drive"),
                       sizeof(double), "Raman drive bytes");
    require_extent(problem.raman.undamped.size(),
                   checked_product(problem.raman.n_terms, doubled, "raman_undamped"),
                   "raman_undamped");
    if (!(problem.raman.fraction > 0.0) || !std::isfinite(problem.raman.fraction) ||
        !(problem.grid.dt_fine > 0.0) || !std::isfinite(problem.grid.dt_fine) ||
        !(problem.raman.epsilon0 > 0.0) || !std::isfinite(problem.raman.epsilon0))
      throw std::invalid_argument(
          "Raman fraction, dt_fine and epsilon0 must be positive and finite");
    if (!std::ranges::all_of(problem.raman.undamped,
                             [](const double value) { return std::isfinite(value); }))
      throw std::invalid_argument("raman_undamped must contain only finite values");
  } else if (problem.raman.thg || problem.raman.n_terms != 0 ||
             !problem.raman.undamped.empty() || !problem.raman.tensor_weights.empty() ||
             problem.raman.tensor_capable) {
    throw std::invalid_argument("Raman data were supplied while Raman is disabled");
  }
}

void GeneralWorkspace::require_trial_scalars(const TrialScalars& scalars) const {
  const auto checked_values = [](const std::vector<double>& values, const char* name,
                                 bool nonnegative, bool fraction) {
    for (double value : values)
      if (!std::isfinite(value) || (nonnegative && value < 0.0) ||
          (fraction && value > 1.0))
        throw std::invalid_argument(std::string(name) + " contains an invalid value");
  };
  if (!scalars.kerr_density.empty())
    require_extent(scalars.kerr_density.size(), problem_.stepper.n_stages,
                   "kerr_density");
  checked_values(scalars.kerr_density, "kerr_density", false, false);
  if (problem_.plasma.enabled &&
      (problem_.plasma.n_channels > 1 || !scalars.plasma_densities.empty() ||
       !scalars.plasma_preionisation.empty())) {
    const auto count = checked_product(
        problem_.stepper.n_stages, problem_.plasma.n_channels, "stage channel values");
    require_extent(scalars.plasma_densities.size(), count, "plasma_densities");
    require_extent(scalars.plasma_preionisation.size(), count, "plasma_preionisation");
    checked_values(scalars.plasma_densities, "plasma_densities", true, false);
    checked_values(scalars.plasma_preionisation, "plasma_preionisation", true, true);
  }
  if (problem_.raman.enabled &&
      (problem_.raman.n_components > 1 || !scalars.raman_densities.empty())) {
    require_extent(scalars.raman_densities.size(),
                   checked_product(problem_.stepper.n_stages,
                                   problem_.raman.n_components, "stage Raman values"),
                   "raman_densities");
    checked_values(scalars.raman_densities, "raman_densities", true, false);
  }
  if (const auto* weights = dispersive_weights())
    weights->require_densities(scalars.dispersive_densities);
  else if (!scalars.dispersive_densities.empty())
    throw std::invalid_argument("dispersive_densities weigh SEP 6's dispersive "
                                "response, and this problem has none");
  if (!general_ && !scalars.transform_weights.empty())
    throw std::invalid_argument("transform_weights weight a general modal guide's "
                                "moving maps, and this problem has none");
  require_extent(scalars.phi_delta.size(),
                 (problem_.stepper.n_stages + 1) * problem_.linear.n_phi_terms,
                 "phi_delta");
  require_extent(scalars.phi_at.size(),
                 problem_.stepper.n_stages * problem_.linear.n_phi_terms, "phi_at");
  require_extent(scalars.prefactor.size(),
                 problem_.stepper.n_stages * problem_.linear.n_prefactor_terms,
                 "prefactor");
  require_extent(scalars.density.size(), problem_.stepper.n_stages, "density");
  require_extent(scalars.taper.size(), problem_.stepper.n_stages, "taper");
  if (problem_.plasma.enabled)
    require_extent(scalars.preionisation.size(), problem_.stepper.n_stages,
                   "preionisation");
  if (problem_.raman.enabled) {
    require_extent(scalars.raman_rates.size(),
                   problem_.stepper.n_stages * problem_.raman.n_terms, "raman_rates");
    if (!std::ranges::all_of(scalars.raman_rates, [](const double value) {
          return std::isfinite(value) && value >= 0.0;
        }))
      throw std::invalid_argument("Raman decay rates must be finite and non-negative");
  }
}

} // namespace solprop_native
