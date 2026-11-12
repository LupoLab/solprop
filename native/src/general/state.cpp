// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// GeneralState over its workspace: host transfers, closing and the forwarders.

#include "workspace.hpp"

namespace solprop_native {

std::vector<Complex> GeneralWorkspace::field_to_host() const {
  std::vector<Complex> result(problem_.n_columns() * problem_.dims.n_omega);
  field_to_host(result);
  return result;
}

// One copy from the memory space into the caller's buffer: no staging mirror and no
// element loop, so a large state crosses once (a device's one transfer).
void GeneralWorkspace::cross_to_host(const ComplexView2& source,
                                     const std::span<Complex> destination) const {
  crossed_.to_host += destination.size_bytes();
  if (crossing::takes_ring(destination.size_bytes())) {
    crossing_ring_.to_host(destination.data(), source.data(), destination.size_bytes());
    return;
  }
  const Kokkos::View<Complex**, Layout, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host(destination.data(), source.extent(0), source.extent(1));
  // On the stage's instance: a compact problem's team, which the default instance's
  // whole-runtime copy would end and restart.
  graph::deep_copy(execution_space_, host, source);
  execution_space_.fence("solprop_native crossing");
}

void GeneralWorkspace::field_to_host(const std::span<Complex> destination) const {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  require_extent(destination.size(), problem_.n_columns() * problem_.dims.n_omega,
                 "field destination");
  execution_space_.fence("solprop_native field_to_host");
  cross_to_host(current_, destination);
}

std::vector<Complex> GeneralWorkspace::stage_to_host(const std::size_t stage) const {
  std::vector<Complex> result(problem_.n_columns() * problem_.dims.n_omega);
  stage_to_host(stage, result);
  return result;
}

void GeneralWorkspace::stage_to_host(const std::size_t stage,
                                     const std::span<Complex> destination) const {
  require_open();
  if (!stages_ready_)
    throw std::logic_error("no trial stages have been evaluated");
  if (stage >= problem_.stepper.n_stages)
    throw std::out_of_range("stage index is outside the tableau");
  require_extent(destination.size(), problem_.n_columns() * problem_.dims.n_omega,
                 "stage destination");
  execution_space_.fence("solprop_native stage_to_host");
  const auto selected = Kokkos::subview(stages_, stage, Kokkos::ALL, Kokkos::ALL);
  crossed_.to_host += destination.size_bytes();
  if (crossing::takes_ring(destination.size_bytes())) {
    crossing_ring_.to_host(destination.data(), selected.data(),
                           destination.size_bytes());
    return;
  }
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, selected);
  for (std::size_t column = 0; column < problem_.n_columns(); ++column)
    for (std::size_t index = 0; index < problem_.dims.n_omega; ++index)
      destination[column * problem_.dims.n_omega + index] = host(column, index);
}

std::vector<Complex> GeneralWorkspace::projected_to_host(const bool coarse) {
  require_open();
  if (!stages_ready_)
    throw std::logic_error("no trial stages have been evaluated");
  if (coarse && projected_coarse_spectrum_.extent(0) == 0)
    throw std::logic_error("this quadrature has no embedded rule");
  if (coarse && dispersive_)
    dispersive_->finish_coarse();
  if (coarse)
    finish_quadrature();
  execution_space_.fence("solprop_native projected_to_host");
  const auto host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, coarse ? projected_coarse_spectrum_ : projected_spectrum_);
  return {host.data(), host.data() + host.size()};
}

std::map<std::string, bool> GeneralWorkspace::dispersive_path() const {
  if (separable_dispersive_)
    return {{"alone", separable_dispersive_->alone()},
            {"in_place", separable_dispersive_->in_place()},
            {"borrowed", separable_dispersive_->borrowed()}};
  // The general route: alone when no regional tensor or delayed response runs.
  if (general_ && general_->dispersive() != nullptr)
    return {{"alone", !general_->regional()}, {"in_place", false}, {"borrowed", false}};
  if (!dispersive_)
    return {};
  return {{"alone", dispersive_->alone()},
          {"in_place", dispersive_->in_place()},
          {"borrowed", dispersive_->borrowed()}};
}

void GeneralWorkspace::close() noexcept {
  if (closed_)
    return;
  execution_space_.fence();
  if (general_)
    general_->close();
  general_.reset();
  if (flow_)
    flow_->close();
  flow_.reset();
  if (dispersive_)
    dispersive_->close();
  dispersive_.reset();
  if (far_field_)
    far_field_->close();
  far_field_.reset();
  if (separable_dispersive_)
    separable_dispersive_->close();
  separable_dispersive_.reset();
  for (auto& stream : held_streams_) {
    stream.forward.reset();
    stream.coarse_forward.reset();
    stream = {};
  }
  held_pack_forward_.reset();
  held_pack_ = {};
  held_pack_spectrum_ = {};
  forward_pack_plan_.reset();
  forward_pack_tail_plan_.reset();
  forward_pack_fine_plan_.reset();
  forward_pack_fine_tail_plan_.reset();
  forward_pack_coarse_plan_.reset();
  forward_pack_fine_ = {};
  forward_pack_fine_tail_ = {};
  forward_pack_coarse_ = {};
  forward_pack_fine_spectrum_ = {};
  forward_pack_fine_tail_spectrum_ = {};
  forward_pack_coarse_spectrum_ = {};
  forward_pack_ = {};
  forward_pack_tail_ = {};
  forward_pack_spectrum_ = {};
  forward_pack_tail_spectrum_ = {};
  held_current_ = {};
  gated_prefix_ = {};
  gated_filtered_ = {};
  held_projection_ = {};
  held_nodal_spectrum_ = {};
  held_projected_spectrum_ = {};
  held_derivative_ = {};
  held_inverse_ = {};
  for (auto& stream : envelope_streams_)
    stream = {};
  envelope_rule_nodes_ = {};
  envelope_rule_weights_ = {};
  envelope_moments_ = {};
  envelope_rates_ = {};
  envelope_exposure_ = {};
  envelope_drude_factor_ = {};
  envelope_loss_factor_ = {};
  envelope_failures_ = {};
  envelope_status_ = {};
  envelope_host_status_ = {};
  radial_blas_input_ = {};
  radial_blas_output_ = {};
  transverse_forward_.reset();
  transverse_inverse_.reset();
  volume_forward_.reset();
  volume_inverse_.reset();
  volume_scratch_ = {};
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  transform_work_ = {};
#endif
  slab_forward_.reset();
  slab_inverse_.reset();
  transverse_ = {};
  spatial_ = {};
  flux_ = {};
  flux_table_ = {};
  flux_transverse_ = {};
  statistics_magnitude_ = {};
  temporal_tail_inverse_real_.clear();
  temporal_tail_inverse_complex_.clear();
  temporal_tail_forward_real_.clear();
  temporal_tail_forward_complex_.clear();
  coarse_tail_inverse_real_.clear();
  coarse_tail_inverse_complex_.clear();
  coarse_tail_forward_real_.clear();
  coarse_tail_forward_complex_.clear();
  inverse_real_plan_.reset();
  inverse_complex_plan_.reset();
  forward_real_plan_.reset();
  forward_complex_plan_.reset();
  forward_coarse_real_plan_.reset();
  forward_coarse_complex_plan_.reset();
  statistics_inverse_plan_.reset();
  absorber_real_plan_.reset();
  absorber_complex_plan_.reset();
  coarse_inverse_real_plan_.reset();
  split_inverse_plan_.reset();
  split_forward_plan_.reset();
  split_tail_inverse_plan_.reset();
  split_tail_forward_plan_.reset();
  coarse_inverse_complex_plan_.reset();
  coarse_forward_real_plan_.reset();
  coarse_forward_complex_plan_.reset();
  raman_drive_forward_plan_.reset();
  raman_convolution_inverse_plan_.reset();
  raman_tail_forward_plan_.reset();
  raman_tail_inverse_plan_.reset();
  raman_tail_drive_ = {};
  raman_tail_spectrum_ = {};
  raman_tail_convolved_ = {};
  retained_inverse_plan_.reset();
  retained_tail_inverse_plan_.reset();
  retained_modal_ = {};
  retained_node_ = {};
  raman_product_forward_plan_.reset();
  raman_product_inverse_plan_.reset();
  raman_product_drive_ = {};
  raman_product_spectrum_ = {};
  raman_kernel_forward_plan_.reset();
  current_ = {};
  register_accumulator_ = {};
  register_stage_ = {};
  low_storage_flows_ = {};
  advanced_ = {};
  stages_ = {};
  solution_ = {};
  error_ = {};
  residual_ = {};
  recycled_ = {};
  noise_ = {};
  whole_propagator_ = {};
  stage_phase_ = {};
  stage_inverse_ = {};
  adams_store_ = {};
  adams_slopes_ = {};
  adams_moments_ = {};
  adams_source_ = {};
  adams_propagator_ = {};
  adams_exponent_ = {};
  adams_saved_ = {};
  adams_held_ = {};
  adams_dense_start_ = {};
  adams_dense_end_ = {};
  adams_dense_exponent_ = {};
  adams_divisor_ = {};
  adams_norms_ = {};
  adams_source_graph_.reset();
  adams_states_ = {};
  adams_rates_ = {};
  adams_carried_ = {};
  adams_dense_terms_ = {};
  adams_lab_ = {};
  adams_endpoint_ = {};
  adams_endpoint_rate_ = {};
  adams_rate_weights_ = {};
  adams_block_ = {};
  adams_channels_ = {};
  adams_fallback_ = {};
  adams_applied_ = {};
  adams_end_applied_ = {};
  adams_end_weights_ = {};
  fine_spectrum_ = {};
  projected_spectrum_ = {};
  projected_coarse_spectrum_ = {};
  absorber_spectrum_ = {};
  statistics_spectrum_ = {};
  statistics_time_ = {};
  statistics_weighted_time_ = {};
  statistics_power_weight_ = {};
  modal_complex_ = {};
  node_complex_ = {};
  polarisation_complex_ = {};
  projected_complex_ = {};
  projected_coarse_complex_ = {};
  absorber_complex_ = {};
  coarse_spectrum_ = {};
  split_factor_ = {};
  coarse_time_complex_ = {};
  scaled_synthesis_complex_ = {};
  scaled_projection_complex_ = {};
  scaled_projection_coarse_complex_ = {};
  scaled_projection_stacked_complex_ = {};
  projected_stacked_complex_ = {};
  chunk_synthesis_complex_ = {};
  tail_synthesis_complex_ = {};
  chunk_projection_complex_ = {};
  tail_projection_complex_ = {};
  chunk_projection_stacked_complex_ = {};
  tail_projection_stacked_complex_ = {};
  node_raman_spectrum_ = {};
  raman_kernel_spectrum_ = {};
  modal_real_ = {};
  node_real_ = {};
  polarisation_real_ = {};
  projected_real_ = {};
  projected_coarse_real_ = {};
  projected_stacked_real_ = {};
  absorber_real_ = {};
  coarse_time_real_ = {};
  node_raman_drive_ = {};
  node_raman_convolved_ = {};
  raman_undamped_ = {};
  raman_tensor_weights_ = {};
  raman_kernel_time_ = {};
  stage_raman_rates_ = {};
  synthesis_ = {};
  projection_ = {};
  projection_coarse_ = {};
  scaled_synthesis_ = {};
  scaled_projection_ = {};
  scaled_projection_coarse_ = {};
  scaled_projection_stacked_ = {};
  chunk_synthesis_ = {};
  tail_synthesis_ = {};
  chunk_projection_ = {};
  tail_projection_ = {};
  chunk_projection_stacked_ = {};
  tail_projection_stacked_ = {};
  partials_ = {};
  pinned_partials_ = {};
  quadrature_partials_ = {};
  weighted_scale_ = {};
  weighted_partials_ = {};
  stash_candidate_ = {};
  stash_current_ = {};
  statistics_power_ = {};
  plasma_statistics_scratch_ = {};
  plasma_rate_ = {};
  plasma_exposure_ = {};
  plasma_fraction_ = {};
  plasma_current_integrand_ = {};
  plasma_free_current_ = {};
  plasma_outer_current_ = {};
  band_tile_totals_ = {};
  plasma_tile_totals_ = {};
  spatial_spectral_weights_ = {};
  spatial_on_axis_ = {};
  spatial_statistics_partials_ = {};
  statistics_total_power_ = {};
  statistics_on_axis_ = {};
  statistics_packed_ = {};
  statistics_axis_row_ = {};
  monitor_bound_ = false;
  monitor_weights_ = {};
  monitor_column_weights_ = {};
  monitor_share_ = {};
  monitor_half_ = {};
  monitor_removed_ = {};
  monitor_split_sum_ = {};
  monitor_found_ = {};
  occupancy_buckets_ = {};
  occupancy_count_ = 0;
  crossing_ring_.release();
  edge_batches_.clear();
  edge_strip_ = {};
  edge_lines_.clear();
  edge_scale_ = {};
  edge_faces_ = {};
  edge_clip_ = {};
  edge_totals_ = {};
  edge_synthesis_ = {};
  edge_columns_ = {};
  tableau_ = {};
  omega_window_ = {};
  omega_ = {};
  time_window_ = {};
  absorber_fine_ = {};
  absorber_coarse_ = {};
  spatial_absorber_real_ = {};
  spatial_absorber_complex_ = {};
  stage_phi_delta_ = {};
  stage_scalars_ = {};
  stage_rate_weights_ = {};
  stage_staging_.clear();
  stage_phi_at_ = {};
  stage_prefactor_ = {};
  stage_density_ = {};
  stage_taper_ = {};
  stage_preionisation_ = {};
  rate_coefficients_ = {};
  channel_parameters_ = {};
  channel_coefficients_ = {};
  channel_scratch_ = {};
  plasma_taps_ = {};
  plasma_diagnostics_ = {};
  stage_channel_density_ = {};
  stage_channel_preionisation_ = {};
  stage_kerr_density_ = {};
  stage_plasma_density_ = {};
  stage_raman_density_ = {};
  stage_raman_densities_ = {};
  raman_row_component_ = {};
  raman_component_fraction_ = {};
  cached_raman_densities_.clear();
  carrier_ = {};
  active_mask_ = {};
  norm_mask_ = {};
  crop_index_ = {};
  fine_gaps_ = {};
  separable_gaps_ = {};
  // FFT plans must release their execution instances before runtime shutdown.
  raman_tail_execution_space_ = ExecutionSpace();
  raman_execution_space_ = ExecutionSpace();
  raman_kernel_execution_space_ = ExecutionSpace();
  raman_product_execution_space_ = ExecutionSpace();
  raman_product_space_ = ExecutionSpace();
  temporal_execution_space_ = ExecutionSpace();
  flow_execution_space_ = ExecutionSpace();
  absorber_execution_space_ = ExecutionSpace();
  adams_execution_space_ = ExecutionSpace();
  adams_copy_space_ = ExecutionSpace();
  raman_multiply_space_ = ExecutionSpace();
  raman_tail_multiply_space_ = ExecutionSpace();
  raman_rebuild_space_ = ExecutionSpace();
  raman_scale_space_ = ExecutionSpace();
  response_execution_space_ = ExecutionSpace();
  response_tail_execution_space_ = ExecutionSpace();
  transform_execution_space_ = ExecutionSpace();
  transform_tail_execution_space_ = ExecutionSpace();
  matrix_copy_space_ = ExecutionSpace();
  tail_matrix_copy_space_ = ExecutionSpace();
  stacked_matrix_copy_space_ = ExecutionSpace();
  tail_stacked_matrix_copy_space_ = ExecutionSpace();
  node_spectral_space_ = ExecutionSpace();
  kernel_pool_.clear();
  fft_pool_.clear();
  raman_pool_.clear();
  raman_tail_pool_.clear();
  fft_execution_space_ = ExecutionSpace();
  execution_space_ = ExecutionSpace();
  closed_ = true;
  initialized_ = false;
  table_ready_ = false;
  has_recycled_ = false;
  pending_fsal_ = false;
  coarse_pending_ = false;
  stash_is_current_ = false;
  raman_kernel_ready_ = false;
}

template <class Function, class... Arguments>
decltype(auto) GeneralState::with_team(Function function,
                                       Arguments&&... arguments) const {
  const CpuTeamScope team(workspace_->cpu_team());
  // NOLINTNEXTLINE(readability-avoid-return-with-void-value): returns what the member returns, void included
  return std::invoke(function, *workspace_, std::forward<Arguments>(arguments)...);
}

GeneralState::GeneralState(Problem problem)
    : workspace_(std::make_unique<GeneralWorkspace>(std::move(problem))) {}
GeneralState::~GeneralState() { close(); }
void GeneralState::set_initial(const std::span<const Complex> field,
                               const std::span<const Complex> noise) {
  with_team(&GeneralWorkspace::set_initial, field, noise);
}
std::vector<double> GeneralState::trial_step(const double dz,
                                             const TrialScalars& scalars) {
  return with_team(&GeneralWorkspace::trial_step, dz, scalars);
}
void GeneralState::prepare_trial(const double dz, const TrialScalars& scalars) {
  with_team(&GeneralWorkspace::prepare_trial, dz, scalars);
}
std::vector<Complex> GeneralState::adams_source(const std::vector<Complex>& input,
                                                const TrialScalars& scalars) {
  return with_team(&GeneralWorkspace::adams_source, input, scalars);
}
void GeneralState::adopt(const std::vector<Complex>& field, const double dz) {
  with_team(&GeneralWorkspace::adopt, field, dz);
}
double GeneralState::low_storage_step(const double dz, const TrialScalars& scalars,
                                      const bool rebuild_table) {
  return with_team(&GeneralWorkspace::low_storage_step, dz, scalars, rebuild_table);
}
void GeneralState::adams_allocate(const std::size_t slots, const bool compensated,
                                  const bool interpolate) {
  with_team(&GeneralWorkspace::adams_allocate, slots, compensated, interpolate);
}
void GeneralState::adams_compensation_start(const std::size_t slot,
                                            const std::vector<double>& rate) {
  with_team(&GeneralWorkspace::adams_compensation_start, slot, rate);
}
void GeneralState::adams_start(const std::size_t slot, const TrialScalars& scalars) {
  with_team(&GeneralWorkspace::adams_start, slot, scalars);
}
double GeneralState::adams_magnitude() {
  return with_team(&GeneralWorkspace::adams_magnitude);
}
void GeneralState::adams_scale(const double rtol, const double atol,
                               const std::vector<double>& weights) {
  with_team(&GeneralWorkspace::adams_scale, rtol, atol, weights);
}
std::vector<double> GeneralState::adams_fused_trial(const double dz,
                                                    const AdamsWeights& weights,
                                                    const std::vector<double>& divisor,
                                                    const TrialScalars& scalars) {
  return with_team(&GeneralWorkspace::adams_fused_trial, dz, weights, divisor, scalars);
}
void GeneralState::adams_fused_accept(const double dz, const AdamsWeights& keep,
                                      const std::size_t free,
                                      const AdamsWeights& dense) {
  with_team(&GeneralWorkspace::adams_fused_accept, dz, keep, free, dense);
}
void GeneralState::adams_hold() { with_team(&GeneralWorkspace::adams_hold); }
void GeneralState::adams_restore() { with_team(&GeneralWorkspace::adams_restore); }
void GeneralState::table_upload(const TableKind kind, std::vector<double> edges,
                                const double lower, const double upper,
                                const std::complex<double>* coefficients,
                                const std::size_t n_intervals,
                                const std::size_t n_terms, const std::size_t n_values) {
  with_team(&GeneralWorkspace::table_upload, kind, std::move(edges), lower, upper,
            coefficients, n_intervals, n_terms, n_values);
}
void GeneralState::table_evaluate(const std::vector<std::size_t>& slots,
                                  const std::vector<double>& positions) {
  with_team(&GeneralWorkspace::table_evaluate, slots, positions);
}
void GeneralState::table_rate(const std::size_t slot, const double z,
                              const bool before) {
  with_team(&GeneralWorkspace::table_rate, slot, z, before);
}
std::vector<Complex> GeneralState::table_terms_to_host(const TableKind kind,
                                                       const std::size_t slot) const {
  return with_team(&GeneralWorkspace::table_terms_to_host, kind, slot);
}
std::vector<Complex>
GeneralState::phase_sum_to_host(const std::vector<double>& weights) {
  return with_team(&GeneralWorkspace::phase_sum_to_host, weights);
}
std::vector<std::string> GeneralState::adams_row_names() const {
  return with_team(&GeneralWorkspace::adams_row_names);
}
std::array<std::size_t, 3>
GeneralState::adams_row_extents(const std::string& name) const {
  return with_team(&GeneralWorkspace::adams_row_extents, name);
}
std::vector<Complex> GeneralState::adams_rows_to_host(const std::string& name) const {
  return with_team(
      [&](GeneralWorkspace& workspace) { return workspace.adams_rows_to_host(name); });
}
void GeneralState::adams_rows_to_host(const std::string& name,
                                      std::complex<double>* destination,
                                      const std::size_t size) const {
  with_team([&](GeneralWorkspace& workspace) {
    workspace.adams_rows_to_host(name, destination, size);
  });
}
void GeneralState::adams_rows_from_host(const std::string& name,
                                        const std::vector<Complex>& values) {
  with_team(&GeneralWorkspace::adams_rows_from_host, name, values);
}
void GeneralState::adams_observe(const AdamsWeights& weights,
                                 const TrialScalars& scalars) {
  with_team(&GeneralWorkspace::adams_observe, weights, scalars);
}
std::vector<double> GeneralState::adams_exponential_trial(
    const double dz, const ExponentialWeights& weights,
    const std::vector<double>& divisor, const TrialScalars& step,
    const TrialScalars& sample, const std::vector<double>& rate) {
  return with_team(&GeneralWorkspace::adams_exponential_trial, dz, weights, divisor,
                   step, sample, rate);
}
void GeneralState::adams_exponential_accept(const double dz, const std::size_t free,
                                            const AdamsWeights& dense) {
  with_team(&GeneralWorkspace::adams_exponential_accept, dz, free, dense);
}
void GeneralState::rk_dense_allocate(const std::vector<double>& weights) {
  with_team(&GeneralWorkspace::rk_dense_allocate, weights);
}
void GeneralState::rk_dense_hold(const double dz) {
  with_team(&GeneralWorkspace::rk_dense_hold, dz);
}
void GeneralState::rk_dense_observe(const double theta, const TrialScalars& carry) {
  with_team(&GeneralWorkspace::rk_dense_observe, theta, carry);
}
void GeneralState::rk_dense_restore() {
  with_team(&GeneralWorkspace::rk_dense_restore);
}
void GeneralState::rk_dense_release() noexcept {
  if (workspace_)
    workspace_->rk_dense_release();
}
void GeneralState::adams_exponential_observe(const double xi, const double dz,
                                             const ExponentialWeights& weights) {
  with_team(&GeneralWorkspace::adams_exponential_observe, xi, dz, weights);
}
std::vector<double> GeneralState::weighted_error(const std::vector<double>& scale) {
  return with_team(&GeneralWorkspace::weighted_error, scale);
}
void GeneralState::set_flow_actions(const std::vector<Complex>& forward,
                                    const std::vector<Complex>& inverse,
                                    const std::vector<unsigned char>& identity) {
  with_team(&GeneralWorkspace::set_flow_actions, forward, inverse, identity);
}
std::vector<Complex> GeneralState::flow_actions_to_host(const bool inverse) const {
  return with_team(&GeneralWorkspace::flow_actions_to_host, inverse);
}
std::vector<Complex> GeneralState::adams_block_to_host() const {
  return with_team(&GeneralWorkspace::adams_block_to_host);
}
std::vector<Complex> GeneralState::adams_exponent_to_host() const {
  return with_team(&GeneralWorkspace::adams_exponent_to_host);
}
std::vector<double>
GeneralState::form_flow_actions(const std::size_t section, const double dz,
                                const std::vector<double>& fractions) {
  return with_team(&GeneralWorkspace::form_flow_actions, section, dz, fractions);
}
void GeneralState::prepare_raman(const std::vector<double>& rates,
                                 const std::vector<double>& densities) {
  with_team(&GeneralWorkspace::prepare_raman, rates, densities);
}
std::vector<double>
GeneralState::statistics(const std::uint64_t flags, const std::vector<double>& windows,
                         const std::vector<double>& on_axis, const double density,
                         const double taper, const double preionisation,
                         const double intensity_factor,
                         const std::vector<double>& channel_preionisation,
                         const std::vector<double>& power_weight) {
  return with_team(&GeneralWorkspace::statistics, flags, windows, on_axis, density,
                   taper, preionisation, intensity_factor, channel_preionisation,
                   power_weight);
}
std::vector<double>
GeneralState::whitened_statistics(const std::vector<Complex>& root,
                                  const std::uint64_t flags,
                                  const std::vector<double>& windows,
                                  const double intensity_factor, const bool spectral) {
  return with_team(&GeneralWorkspace::whitened_statistics, root, flags, windows,
                   intensity_factor, spectral);
}
std::vector<double> GeneralState::spatial_statistics(const std::uint64_t flags,
                                                     const double intensity_factor,
                                                     const double z) {
  return with_team(&GeneralWorkspace::spatial_statistics, flags, intensity_factor, z);
}
std::vector<double> GeneralState::plane_fluence_trace(
    const double center_x, const double center_y, const double dx, const double dy,
    const double extent, const double spacing, const std::size_t n_radii,
    const std::vector<double>& cosines, const std::vector<double>& sines) {
  return with_team(&GeneralWorkspace::plane_fluence_trace, center_x, center_y, dx, dy,
                   extent, spacing, n_radii, cosines, sines);
}
std::vector<Complex> GeneralState::axis_spectrum(const std::vector<double>& matrix) {
  return with_team(&GeneralWorkspace::axis_spectrum, matrix);
}
std::vector<double>
GeneralState::axis_populations(const std::vector<double>& matrix,
                               const std::vector<double>& channel_preionisation) {
  return with_team(&GeneralWorkspace::axis_populations, matrix, channel_preionisation);
}
std::vector<double> GeneralState::aperture_statistics(
    const std::uint64_t flags, const std::vector<double>& apertures, const int exponent,
    const std::vector<double>& regions, const double intensity_factor) {
  return with_team(&GeneralWorkspace::aperture_statistics, flags, apertures, exponent,
                   regions, intensity_factor);
}
void GeneralState::finish(const double dz) { with_team(&GeneralWorkspace::finish, dz); }
void GeneralState::bind_monitor(const std::vector<double>& weights,
                                const std::vector<double>& column_weights) {
  with_team(&GeneralWorkspace::bind_monitor, weights, column_weights);
}
void GeneralState::unbind_monitor() noexcept {
  if (workspace_)
    workspace_->unbind_monitor();
}
std::vector<double> GeneralState::monitor_reading() {
  return with_team(&GeneralWorkspace::monitor_reading);
}
void GeneralState::bind_edge(const std::vector<double>& faces,
                             const std::vector<double>& clip,
                             const std::vector<double>& totals,
                             const std::vector<double>& synthesis,
                             const std::size_t first_row) {
  with_team(&GeneralWorkspace::bind_edge, faces, clip, totals, synthesis, first_row);
}
CrossedBytes GeneralState::crossed_bytes() const noexcept {
  return workspace_->crossed_bytes();
}
void GeneralState::bind_edge_lines(const EdgeLines& lines) {
  with_team(&GeneralWorkspace::bind_edge_lines, lines);
}
EdgeNumbers GeneralState::edge_reading(const bool faces, const bool whole,
                                       const bool totals, const bool strip,
                                       const bool current) {
  return with_team(&GeneralWorkspace::edge_reading, faces, whole, totals, strip,
                   current);
}
void GeneralState::bind_occupancy(const std::vector<unsigned char>& buckets,
                                  const std::size_t count) {
  with_team(&GeneralWorkspace::bind_occupancy, buckets, count);
}
void GeneralState::promote_fsal() { with_team(&GeneralWorkspace::promote_fsal); }
void GeneralState::clear_fsal() { with_team(&GeneralWorkspace::clear_fsal); }
void GeneralState::set_absorber_quiet(const bool quiet) {
  with_team(&GeneralWorkspace::set_absorber_quiet, quiet);
}
double GeneralState::quadrature_error() const {
  return workspace_ ? workspace_->quadrature_error()
                    : Kokkos::Experimental::quiet_NaN_v<double>;
}
double GeneralState::quadrature_peak() {
  return workspace_ ? workspace_->quadrature_peak()
                    : Kokkos::Experimental::quiet_NaN_v<double>;
}
std::string GeneralState::synthesis_provider() const {
  return workspace_ ? workspace_->synthesis_provider() : "closed";
}
std::string GeneralState::projection_provider() const {
  return workspace_ ? workspace_->projection_provider() : "closed";
}
std::string GeneralState::gemm_shape_policy() const {
  const CpuTeamScope team(workspace_->cpu_team());
  return std::string(GeneralWorkspace::gemm_shape_policy());
}
std::size_t GeneralState::response_team_size() const noexcept {
  return workspace_ ? workspace_->response_team_size() : 0;
}
std::vector<Complex> GeneralState::field_to_host() const {
  return with_team(
      [&](GeneralWorkspace& workspace) { return workspace.field_to_host(); });
}
void GeneralState::set_crossing_workers(const std::size_t workers) noexcept {
  if (workspace_)
    workspace_->set_crossing_workers(workers);
}
void GeneralState::field_to_host(const std::span<Complex> destination) const {
  with_team([&](GeneralWorkspace& workspace) { workspace.field_to_host(destination); });
}
const Complex* GeneralState::host_field() const {
  return with_team(&GeneralWorkspace::host_field);
}
void GeneralState::count_to_host(const std::size_t bytes) const noexcept {
  workspace_->count_to_host(bytes);
}
void GeneralState::stage_to_host(const std::size_t stage,
                                 const std::span<Complex> destination) const {
  with_team([&](GeneralWorkspace& workspace) {
    workspace.stage_to_host(stage, destination);
  });
}
std::vector<Complex> GeneralState::stage_to_host(const std::size_t stage) const {
  return with_team(
      [&](GeneralWorkspace& workspace) { return workspace.stage_to_host(stage); });
}
std::vector<Complex> GeneralState::projected_to_host(const bool coarse) {
  return with_team(&GeneralWorkspace::projected_to_host, coarse);
}
std::map<std::string, bool> GeneralState::dispersive_path() const {
  return with_team(&GeneralWorkspace::dispersive_path);
}
std::map<std::string, std::size_t> GeneralState::memory_ledger() const {
  return with_team(&GeneralWorkspace::memory_ledger);
}
CpuSizing GeneralState::execution_sizing() const {
  return with_team(&GeneralWorkspace::execution_sizing);
}
const CpuExecution& GeneralState::cpu_execution() const {
  return workspace_->cpu_execution();
}
const FftPlanning& GeneralState::fft_planning() const {
  return workspace_->problem_.fft_planning;
}
double GeneralState::planning_seconds() const noexcept {
  return workspace_ ? workspace_->planning_seconds() : 0.0;
}
double GeneralState::raman_planning_seconds() const noexcept {
  return workspace_ ? workspace_->raman_planning_seconds() : 0.0;
}
std::string GeneralState::raman_fft_provider() const {
  return workspace_ ? workspace_->raman_fft_provider() : "none";
}
std::map<std::string, int> GeneralState::raman_fft_tasks() const {
  return workspace_ ? workspace_->raman_fft_tasks() : std::map<std::string, int>{};
}
std::size_t GeneralState::raman_kernel_transforms() const noexcept {
  return workspace_ ? workspace_->raman_kernel_transforms() : 0;
}
bool GeneralState::fsal_ready() const noexcept {
  return workspace_ && workspace_->fsal_ready();
}
std::size_t GeneralState::rk_source_evaluations() const noexcept {
  return workspace_ ? workspace_->rk_source_evaluations() : 0;
}
std::size_t GeneralState::axis_scans() const noexcept {
  return workspace_ ? workspace_->axis_scans() : 0;
}
bool GeneralState::closed() const noexcept {
  return !workspace_ || workspace_->closed();
}
void GeneralState::close() noexcept {
  if (workspace_)
    workspace_->close();
}

} // namespace solprop_native
