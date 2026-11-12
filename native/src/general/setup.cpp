// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The general workspace's construction: its allocations, uploads, FFT plans and
// memory ledger.

#include "workspace.hpp"

namespace solprop_native {

GeneralWorkspace::GeneralWorkspace(Problem problem) : problem_(std::move(problem)) {
  require_general_problem(problem_);
  require_cpu_execution(problem_.cpu_execution);
  identity_crop_ = problem_.grid.crop_index.size() == problem_.dims.n_omega;
  for (std::size_t index = 0; identity_crop_ && index < problem_.dims.n_omega; ++index)
    identity_crop_ = problem_.grid.crop_index[index] == index;
  one_mode_products_ =
      one_mode_products_enabled && problem_.dims.family == FieldFamily::field_real &&
      problem_.dims.n_modes == 1 && problem_.dims.n_pol_state == 1 &&
      problem_.dims.n_pol_transform == 1 && problem_.dims.n_nodes > 1 &&
      problem_.dims.node_chunk == problem_.dims.n_nodes;
  compact_ = compact_problem(problem_);
  if (compact_)
    for (auto* pool : {&kernel_pool_, &fft_pool_, &raman_pool_, &raman_tail_pool_})
      pool->compact();
  execution_space_ = kernel_pool_.get(CpuOperation::stage,
                                      problem_.n_columns() * problem_.dims.n_omega);
  const auto nodes = problem_.dims.node_chunk;
  const auto response_elements =
      nodes * problem_.effective_node_pol() * problem_.dims.n_to;
  response_execution_space_ =
      kernel_pool_.get(CpuOperation::response, response_elements, problem_.dims.n_to);
  // On one thread the vendor GEMM's serial kernels are as fast.
  one_mode_products_ =
      one_mode_products_ && response_execution_space_.concurrency() > 1;
  const auto tail = problem_.dims.n_nodes % nodes;
  response_tail_execution_space_ = kernel_pool_.get(
      CpuOperation::response, tail * problem_.effective_node_pol() * problem_.dims.n_to,
      problem_.dims.n_to);
  const auto dense_elements = response_elements * problem_.dims.n_modes;
  const auto tail_dense_elements =
      tail * problem_.effective_node_pol() * problem_.dims.n_to * problem_.dims.n_modes;
  transform_execution_space_ = kernel_pool_.get(CpuOperation::dense, dense_elements);
  transform_tail_execution_space_ =
      kernel_pool_.get(CpuOperation::dense, tail_dense_elements);
  temporal_execution_space_ =
      kernel_pool_.get(CpuOperation::copy, problem_.n_columns() * problem_.dims.n_to);
  // Packing copies only matrix entries, and splitting copies only output samples;
  // neither repeats the dense transform's inner product.
  const auto matrix_elements =
      nodes * problem_.dims.n_pol_transform * problem_.dims.n_modes;
  const auto tail_matrix_elements =
      tail * problem_.dims.n_pol_transform * problem_.dims.n_modes;
  matrix_copy_space_ = kernel_pool_.get(CpuOperation::copy, matrix_elements);
  tail_matrix_copy_space_ = kernel_pool_.get(CpuOperation::copy, tail_matrix_elements);
  stacked_matrix_copy_space_ =
      kernel_pool_.get(CpuOperation::copy, 2 * matrix_elements);
  tail_stacked_matrix_copy_space_ =
      kernel_pool_.get(CpuOperation::copy, 2 * tail_matrix_elements);
  node_spectral_space_ =
      kernel_pool_.get(CpuOperation::copy,
                       nodes * problem_.effective_node_pol() * problem_.dims.n_omega);
  flow_execution_space_ = kernel_pool_.get(
      CpuOperation::flow, problem_.flow.active * problem_.dims.n_modes *
                              problem_.dims.n_modes * problem_.dims.n_modes);
  if (!problem_.spatial.absorber_matrix.empty())
    absorber_execution_space_ = kernel_pool_.get(
        CpuOperation::dense,
        problem_.n_columns() * problem_.dims.n_to * problem_.dims.n_modes);
  // solprop SEP 46: the workspace's own FFT planning, wisdom isolated from the rest
  // of the process in a calibration's modes.
  const FftPlanningScope planning(problem_.fft_planning);
  allocate();
  upload();
  const auto heap_before = heap_bytes();
  const auto device_before = device_free_bytes();
  const auto started = std::chrono::steady_clock::now();
  plan();
  execution_space_.fence("solprop_native planned workspace accounting");
  const auto heap_after = heap_bytes();
  const auto device_after = device_free_bytes();
  plan_heap_growth_ = heap_after > heap_before ? heap_after - heap_before : 0;
  plan_device_growth_ = device_before > device_after ? device_before - device_after : 0;
  planning_seconds_ =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  execution_space_.fence("solprop_native construction");
}

void GeneralWorkspace::allocate() {
  if (problem_.stepper.low_storage
          ? problem_.stepper.low_storage_flows > 2 || problem_.flow.enabled ||
                (!problem_.stepper.low_storage_stage_register &&
                 !problem_.free_geometry())
          : problem_.stepper.low_storage_flows != 0 ||
                !problem_.stepper.low_storage_stage_register)
    throw std::invalid_argument(
        "the low-storage march holds 0 to 2 flows beside no coupled flow, and elides "
        "its stage register on free space alone");
  // The exchange fold of an in-stage absorber re-forms its input from the stage
  // register, and a separable fold applies one flow volume, a constant operator's
  // table (solprop SEP 24 section 4.3): neither has a meaning to fall back on.
  if (problem_.stepper.low_storage && problem_.absorber.in_stage &&
      !problem_.stepper.low_storage_stage_register)
    throw std::invalid_argument("the low-storage march cannot elide its stage register "
                                "beside an in-stage absorber");
  if (problem_.stepper.low_storage && problem_.separable_geometry() &&
      problem_.stepper.low_storage_flows > 1)
    throw std::invalid_argument(
        "the low-storage march holds at most one flow volume on "
        "a separable geometry");
  const auto columns = problem_.n_columns();
  const auto n = problem_.dims.n_omega;
  const auto chunk_rows = problem_.dims.node_chunk * problem_.effective_node_pol();
  const auto transform_rows =
      problem_.separable_geometry()
          ? 0
          : problem_.dims.n_nodes * problem_.dims.n_pol_transform;
  // A planar chunk's temporal rows are its y rows' columns, not its nodes.
  const auto temporal_columns =
      problem_.slab_geometry() ? transverse_units_per_chunk() * problem_.spatial.n_p
                               : (problem_.free_geometry() ? chunk_rows : columns);
  separable_aliased_ = problem_.free_geometry() && !problem_.slab_geometry() &&
                       !problem_.dispersive.enabled && temporal_columns == chunk_rows;
  const auto statistic_nodes =
      problem_.separable_geometry() ? problem_.dims.node_chunk : problem_.dims.n_modes;
  // Spatial statistics reconstruct one chunk of physical nodes at a time.
  const auto statistics_rows = problem_.slab_geometry() ? chunk_rows : temporal_columns;
  if (problem_.separable_geometry()) {
    // Keeping transform axes trailing avoids Kokkos-FFT's per-execution
    // transpose allocations. The canonical state retains omega-contiguous columns.
    transverse_ = problem_.slab_geometry()
                      ? ComplexView3("solprop_native.slab.transverse", n,
                                     problem_.spatial.n_p, problem_.spatial.n_y)
                      : ComplexView3("solprop_native.free.transverse",
                                     problem_.dims.n_pol_state * n,
                                     problem_.spatial.n_y, problem_.spatial.n_x);
    spatial_.square = ComplexView("solprop_native.free.k_squared", n);
    spatial_.square_low = ComplexView("solprop_native.free.k_squared_low", n);
    spatial_.numerator = ComplexView("solprop_native.free.numerator", n);
    spatial_.clamped = ComplexView("solprop_native.free.clamped", n);
    spatial_.transverse =
        RealView("solprop_native.free.kperp_sq", problem_.dims.n_modes);
    spatial_.transverse_low =
        RealView("solprop_native.free.kperp_sq_low", problem_.dims.n_modes);
    spatial_.frame = RealView("solprop_native.free.frame", n);
    spatial_.spectral_rate = RealView("solprop_native.free.spectral_rate", n);
    spatial_.user_rate = RealView("solprop_native.free.user_rate", n);
    spatial_.absorber = RealView("solprop_native.free.absorber", problem_.dims.n_modes);
    spatial_.transverse_rate =
        RealView("solprop_native.free.kperp_rate", problem_.dims.n_modes);
    spatial_.evanescent = problem_.spatial.evanescent_rate;
    if (problem_.slab_geometry()) {
      spatial_.kappa =
          ComplexView2("solprop_native.slab.kappa_sq", problem_.spatial.n_p, n);
      spatial_.channels = problem_.spatial.n_p;
    }
    flux_ = spatial_;
    if (problem_.flux.closure)
      flux_transverse_ =
          ComplexView3("solprop_native.flux.closure_transverse", transverse_.extent(0),
                       transverse_.extent(1), transverse_.extent(2));
  } else if (!problem_.flux.k_squared.empty()) {
    flux_.square = ComplexView("solprop_native.flux.k_squared", n);
    flux_.square_low = ComplexView("solprop_native.flux.k_squared_low", n);
    flux_.transverse = RealView("solprop_native.flux.kperp_sq", problem_.dims.n_modes);
    flux_.transverse_low =
        RealView("solprop_native.flux.kperp_sq_low", problem_.dims.n_modes);
  }
  current_ = ComplexView2("solprop_native.state.current", columns, n);
  if (problem_.stepper.low_storage) {
    // solprop SEP 24 section 6.7: the march holds its registers beside the state and
    // none of the Runge-Kutta views; the stage's input is formed in the route's own
    // scratch (`transverse_`, or the fine spectrum of the modal routes).
    register_accumulator_ =
        ComplexView2("solprop_native.low_storage.accumulator", columns, n);
    if (problem_.stepper.low_storage_stage_register)
      register_stage_ = ComplexView2("solprop_native.low_storage.stage", columns, n);
    low_storage_flows_ = ComplexView3("solprop_native.low_storage.flows",
                                      problem_.stepper.low_storage_flows, columns, n);
  } else {
    advanced_ = ComplexView2("solprop_native.state.advanced", columns, n);
    stages_ = ComplexView3("solprop_native.step.stages", problem_.stepper.n_stages,
                           columns, n);
    solution_ = ComplexView2("solprop_native.step.solution", columns, n);
    error_ = ComplexView2("solprop_native.step.error", columns, n);
    residual_ = ComplexView2("solprop_native.step.residual", columns, n);
    recycled_ = ComplexView2("solprop_native.step.recycled", columns, n);
    whole_propagator_ =
        ComplexView2("solprop_native.step.whole_propagator", columns, n);
    if (!problem_.separable_geometry())
      stage_inverse_ = ComplexView2("solprop_native.step.stage_inverse", columns, n);
  }
  if (problem_.stepper.modified_noise)
    noise_ = ComplexView2("solprop_native.noise", columns, n);
  if (!problem_.separable_geometry() && problem_.flow.enabled)
    stage_phase_ = ComplexView2("solprop_native.step.stage_phase", columns, n);
  partials_ =
      RealView2("solprop_native.step.error_partials", 3,
                checked_product(columns, reduction_slots_, "norm partial slots"));
  if (!problem_.transform.projection_coarse_matrix.empty())
    quadrature_partials_ =
        RealView2("solprop_native.transform.quadrature_partials", 2, columns);
  quadrature_scores_ = RealView("solprop_native.transform.quadrature_scores", 2);
  Kokkos::deep_copy(quadrature_scores_, Kokkos::Experimental::quiet_NaN_v<double>);
  if (!problem_.stepper.low_storage)
    pinned_partials_ =
        Kokkos::create_mirror_view(Kokkos::SharedHostPinnedSpace{}, partials_);
  if (problem_.stepper.weighted_controller) {
    weighted_scale_ = RealView("solprop_native.step.weighted_scale", n);
    weighted_partials_ =
        RealView("solprop_native.step.weighted_partials", columns * reduction_slots_);
  }
  if (!problem_.spatial.spectral_weights.empty()) {
    spatial_spectral_weights_ =
        RealView("solprop_native.spatial.spectral_weights", problem_.dims.n_modes);
    spatial_on_axis_ =
        ComplexView("solprop_native.spatial.on_axis", problem_.dims.n_modes);
    // One row per column for the energy and per node for the peaks and fluence,
    // which differ on the hybrid planar kind alone.
    spatial_statistics_partials_ =
        RealView2("solprop_native.spatial.statistics_partials",
                  std::max(problem_.dims.n_modes, problem_.dims.n_nodes), 3);
  }
  statistics_spectrum_ = ComplexView2("solprop_native.statistics.spectrum",
                                      statistics_rows, problem_.dims.n_t);
  statistics_time_ = ComplexView2("solprop_native.statistics.time", statistics_rows,
                                  problem_.dims.n_t);
  statistics_power_weight_ =
      RealView2("solprop_native.statistics.power_weight",
                problem_.spatial.spectral_weights.empty() ? columns : 0, n);
  graph::deep_copy(execution_space_, statistics_power_weight_, 1.0);
  // A spatial run's rows hold the magnetic closure in time (solprop SEP 34) when a
  // local flux statistic is asked for.
  statistics_weighted_time_ =
      ComplexView2("solprop_native.statistics.weighted_time",
                   problem_.spatial.spectral_weights.empty() ? temporal_columns
                   : problem_.flux.closure                   ? statistics_rows
                                                             : 0,
                   problem_.dims.n_t);
  statistics_power_ =
      RealView2("solprop_native.statistics.power", statistic_nodes, problem_.dims.n_t);
  if (problem_.flux.closure && !problem_.separable_geometry())
    statistics_magnitude_ = RealView2("solprop_native.statistics.flux_magnitude",
                                      statistic_nodes, problem_.dims.n_t);
  statistics_total_power_ =
      RealView("solprop_native.statistics.total_power", problem_.dims.n_t);
  statistics_on_axis_ =
      RealView("solprop_native.statistics.on_axis", 2 * problem_.dims.n_modes);
  const auto base_statistics = checked_sum(
      checked_sum(problem_.dims.n_t,
                  checked_product(2, problem_.n_columns(), "statistics columns"),
                  "statistics packed output"),
      checked_product(3, problem_.dims.n_modes, "statistics modes"),
      "statistics packed output");
  statistics_packed_ =
      RealView("solprop_native.statistics.packed",
               checked_sum(base_statistics, 5, "statistics packed output"));
  crop_index_ = IndexView("solprop_native.transform.crop_index", n);
  active_mask_ = ByteView("solprop_native.mask.active", n);
  norm_mask_ = ByteView("solprop_native.mask.norm", n);
  omega_window_ = RealView("solprop_native.window.omega", n);
  omega_ = RealView("solprop_native.statistics.omega", problem_.grid.omega.size());
  time_window_ = RealView("solprop_native.window.time", problem_.dims.n_to);
  carrier_ = ComplexView("solprop_native.kerr.carrier", problem_.dims.n_to);
  phi_terms_ = ComplexView3(
      "solprop_native.linear.phi_terms",
      problem_.separable_geometry() ? 0 : problem_.linear.n_phi_terms, columns, n);
  prefactor_terms_ = ComplexView3(
      "solprop_native.response.prefactor_terms",
      problem_.separable_geometry() ? 0 : problem_.linear.n_prefactor_terms, columns,
      n);
  tableau_ = RealView("solprop_native.step.tableau",
                      problem_.stepper.n_stages * problem_.stepper.n_stages);
  {
    // One device buffer for every per-stage scalar view, so that a trial's scalars
    // cross in one copy instead of one per view: on a device each small copy from
    // pageable memory is a transfer of its own.
    const auto stages = problem_.stepper.n_stages;
    const auto channels = problem_.plasma.enabled ? problem_.plasma.n_channels : 0;
    const auto raman_terms = problem_.raman.enabled ? problem_.raman.n_terms : 0;
    const auto raman_components =
        problem_.raman.enabled && problem_.raman.n_components > 1
            ? problem_.raman.n_components
            : 0;
    const std::array<std::size_t, 14> sizes{(stages + 1) * problem_.linear.n_phi_terms,
                                            stages * problem_.linear.n_phi_terms,
                                            stages * problem_.linear.n_prefactor_terms,
                                            stages,
                                            stages,
                                            stages,
                                            stages,
                                            stages * channels,
                                            stages * channels,
                                            stages,
                                            stages,
                                            stages * raman_terms,
                                            stages * raman_components,
                                            problem_.linear.n_phi_terms};
    std::size_t total = 0;
    for (std::size_t region = 0; region < sizes.size(); ++region) {
      stage_offsets_[region] = total;
      total += sizes[region];
    }
    stage_offsets_[sizes.size()] = total;
    stage_slot_size_ = total;
    stage_scalars_ = RealView("solprop_native.step.scalars", 2 * total);
    bind_stage_scalars(0);
    // No trial's tapers yet: every scaling runs until one is bound.
    trial_taper_.assign(problem_.stepper.n_stages,
                        std::numeric_limits<double>::quiet_NaN());
    // A pinned buffer's copy is asynchronous, so its source must not be rewritten
    // until the copy has run: `upload_trial` fences before the ring comes round again.
    // A host build's mirror is the buffer itself.
    stage_staging_.clear();
    for (std::size_t copy = 0; copy < stage_staging_ring; ++copy)
      stage_staging_.push_back(
          Kokkos::create_mirror_view(Kokkos::SharedHostPinnedSpace{}, stage_scalars_));
  }
  if (problem_.raman.enabled) {
    const auto doubled = 2 * problem_.dims.n_to;
    node_raman_drive_ =
        RealView2("solprop_native.raman.node_drive",
                  problem_.dims.node_chunk * problem_.raman_drives(), doubled);
    node_raman_spectrum_ = ComplexView2(
        "solprop_native.raman.node_spectrum",
        problem_.dims.node_chunk * problem_.raman_drives(), problem_.dims.n_to + 1);
    node_raman_convolved_ =
        RealView2("solprop_native.raman.node_convolved",
                  problem_.dims.node_chunk * problem_.raman_drives(), doubled);
    raman_undamped_ =
        RealView2("solprop_native.raman.undamped", problem_.raman.n_terms, doubled);
    raman_kernel_time_ = RealView2("solprop_native.raman.kernel_time",
                                   problem_.raman_kernels(), doubled);
    raman_kernel_spectrum_ =
        ComplexView2("solprop_native.raman.kernel_spectrum", problem_.raman_kernels(),
                     problem_.dims.n_to + 1);
    raman_tensor_weights_ =
        RealView2("solprop_native.raman.tensor_weights", problem_.raman.n_terms, 2);
    cached_raman_rates_.resize(problem_.raman.n_terms);
    if (problem_.raman.n_components > 1) {
      raman_row_component_ =
          RealView("solprop_native.raman.row_component", problem_.raman.n_terms);
      raman_component_fraction_ = RealView("solprop_native.raman.component_fraction",
                                           problem_.raman.n_components);
      cached_raman_densities_.resize(problem_.raman.n_components);
    }
  }
  if (use_packed_radial_blas()) {
    radial_blas_input_ = RealView2("solprop_native.radial.blas_input",
                                   problem_.dims.n_nodes, 2 * problem_.dims.n_to);
    radial_blas_output_ = RealView2("solprop_native.radial.blas_output",
                                    problem_.dims.n_nodes, 2 * problem_.dims.n_to);
  }
  // The hybrid planar kind's slab rule in x, (n_x, n_p) and (n_p, n_x).
  const auto synthesis_rows =
      problem_.slab_geometry() ? problem_.spatial.n_x : transform_rows;
  const auto synthesis_columns =
      problem_.slab_geometry() ? problem_.spatial.n_p : problem_.dims.n_modes;
  synthesis_ = RealView2("solprop_native.transform.synthesis", synthesis_rows,
                         synthesis_columns);
  projection_ = RealView2("solprop_native.transform.projection", synthesis_columns,
                          synthesis_rows);
  scaled_synthesis_ = RealView2("solprop_native.transform.scaled_synthesis",
                                transform_rows, problem_.dims.n_modes);
  scaled_projection_ = RealView2("solprop_native.transform.scaled_projection",
                                 problem_.dims.n_modes, transform_rows);
  if (problem_.dims.family == FieldFamily::envelope && use_blas_for_synthesis()) {
    scaled_synthesis_complex_ =
        ComplexView2("solprop_native.transform.scaled_synthesis_complex",
                     transform_rows, problem_.dims.n_modes);
    scaled_projection_complex_ =
        ComplexView2("solprop_native.transform.scaled_projection_complex",
                     problem_.dims.n_modes, transform_rows);
  }
  if (!problem_.transform.projection_coarse_matrix.empty()) {
    projection_coarse_ = RealView2("solprop_native.transform.projection_coarse",
                                   problem_.dims.n_modes, transform_rows);
    scaled_projection_coarse_ =
        RealView2("solprop_native.transform.scaled_projection_coarse",
                  problem_.dims.n_modes, transform_rows);
    if (problem_.dims.family == FieldFamily::envelope && use_blas_for_projection()) {
      scaled_projection_coarse_complex_ =
          ComplexView2("solprop_native.transform.scaled_projection_coarse_complex",
                       problem_.dims.n_modes, transform_rows);
      scaled_projection_stacked_complex_ =
          ComplexView2("solprop_native.transform.scaled_projection_stacked_complex",
                       2 * problem_.dims.n_modes, transform_rows);
      projected_stacked_complex_ =
          ComplexView2("solprop_native.response.projected_stacked_complex", 2 * columns,
                       problem_.dims.n_to);
    } else if (problem_.dims.family != FieldFamily::envelope) {
      scaled_projection_stacked_ =
          RealView2("solprop_native.transform.scaled_projection_stacked",
                    2 * problem_.dims.n_modes, transform_rows);
      projected_stacked_real_ =
          RealView2("solprop_native.response.projected_stacked_real",
                    2 * temporal_columns, problem_.dims.n_to);
    }
  }
  const auto chunk_transform_rows =
      problem_.dims.node_chunk * problem_.dims.n_pol_transform;
  const auto tail_nodes = problem_.dims.n_nodes % problem_.dims.node_chunk;
  const auto tail_transform_rows = tail_nodes * problem_.dims.n_pol_transform;
  if (use_chunked_blas_for_synthesis()) {
    if (problem_.dims.family == FieldFamily::envelope) {
      chunk_synthesis_complex_ =
          ComplexView2("solprop_native.transform.chunk_synthesis_complex",
                       chunk_transform_rows, problem_.dims.n_modes);
      if (tail_transform_rows != 0)
        tail_synthesis_complex_ =
            ComplexView2("solprop_native.transform.tail_synthesis_complex",
                         tail_transform_rows, problem_.dims.n_modes);
    } else {
      chunk_synthesis_ = RealView2("solprop_native.transform.chunk_synthesis",
                                   chunk_transform_rows, problem_.dims.n_modes);
      if (tail_transform_rows != 0)
        tail_synthesis_ = RealView2("solprop_native.transform.tail_synthesis",
                                    tail_transform_rows, problem_.dims.n_modes);
    }
  }
  if (use_chunked_blas_for_projection()) {
    const bool stacked = !problem_.transform.projection_coarse_matrix.empty();
    const auto projection_rows =
        stacked ? 2 * problem_.dims.n_modes : problem_.dims.n_modes;
    if (problem_.dims.family == FieldFamily::envelope) {
      if (stacked) {
        chunk_projection_stacked_complex_ =
            ComplexView2("solprop_native.transform.chunk_projection_stacked_complex",
                         projection_rows, chunk_transform_rows);
        if (tail_transform_rows != 0)
          tail_projection_stacked_complex_ =
              ComplexView2("solprop_native.transform.tail_projection_stacked_complex",
                           projection_rows, tail_transform_rows);
      } else {
        chunk_projection_complex_ =
            ComplexView2("solprop_native.transform.chunk_projection_complex",
                         projection_rows, chunk_transform_rows);
        if (tail_transform_rows != 0)
          tail_projection_complex_ =
              ComplexView2("solprop_native.transform.tail_projection_complex",
                           projection_rows, tail_transform_rows);
      }
    } else if (stacked) {
      chunk_projection_stacked_ =
          RealView2("solprop_native.transform.chunk_projection_stacked",
                    projection_rows, chunk_transform_rows);
      if (tail_transform_rows != 0)
        tail_projection_stacked_ =
            RealView2("solprop_native.transform.tail_projection_stacked",
                      projection_rows, tail_transform_rows);
    } else {
      chunk_projection_ = RealView2("solprop_native.transform.chunk_projection",
                                    projection_rows, chunk_transform_rows);
      if (tail_transform_rows != 0)
        tail_projection_ = RealView2("solprop_native.transform.tail_projection",
                                     projection_rows, tail_transform_rows);
    }
  }
  fine_spectrum_ = ComplexView2("solprop_native.fft.fine_spectrum", temporal_columns,
                                fine_spectrum_size());
  if (!problem_.separable_geometry() && problem_.dims.n_omega != fine_spectrum_size())
    fine_gaps_ = IndexView("solprop_native.fft.fine_gaps", 3 * fine_gap_runs().size());
  if (problem_.separable_geometry() && problem_.dims.n_omega != fine_spectrum_size()) {
    std::vector<std::size_t> bins;
    for (const auto& [bin, length] : fine_gap_runs())
      for (std::size_t offset = 0; offset < length; ++offset)
        bins.push_back(bin + offset);
    separable_gaps_ = IndexView("solprop_native.free.fine_gaps", bins.size());
    copy_vector(execution_space_, separable_gaps_, bins);
  }
  // A device's packed forward rows (`packs_forward`): the absorber's, the projection's,
  // the coarse projection's and the held pack's, in that order, carved from one buffer
  // and one spectrum.
  const bool pack = packs_forward();
  const std::size_t absorber_rows =
      pack && problem_.absorber.in_stage ? temporal_columns : 0;
  const std::size_t coarse_rows =
      pack && !problem_.transform.projection_coarse_matrix.empty() ? temporal_columns
                                                                   : 0;
  const std::size_t pack_rows =
      absorber_rows + temporal_columns + coarse_rows +
      (held_plasma() ? 2 * (temporal_columns + coarse_rows) : 0);
  const auto pack_slice = [&](const std::size_t first, const std::size_t count) {
    return std::pair<std::size_t, std::size_t>(first, first + count);
  };
  constexpr bool host_pack_layout = build::host;
  // A host's pack: [absorber | fine | held fine x2 | coarse | held coarse x2]; a
  // device's: [absorber | fine | coarse | (held fine, held coarse) x2].
  const std::size_t held_fine_rows = held_plasma() ? 2 * temporal_columns : 0;
  const std::size_t fine_end = absorber_rows + temporal_columns + held_fine_rows;
  const std::size_t coarse_first =
      host_pack_layout ? fine_end : absorber_rows + temporal_columns;
  forward_pack_absorber_rows_ = absorber_rows;
  if (pack) {
    forward_pack_ =
        RealView2("solprop_native.fft.forward_pack", pack_rows, problem_.dims.n_to);
    forward_pack_spectrum_ = ComplexView2("solprop_native.fft.forward_pack_spectrum",
                                          pack_rows, projected_spectrum_size());
    if (absorber_rows != 0 && pack_rows - absorber_rows > temporal_columns) {
      forward_pack_tail_ = Kokkos::subview(
          forward_pack_, pack_slice(absorber_rows, pack_rows - absorber_rows),
          Kokkos::ALL);
      forward_pack_tail_spectrum_ = Kokkos::subview(
          forward_pack_spectrum_, pack_slice(absorber_rows, pack_rows - absorber_rows),
          Kokkos::ALL);
    }
    if (host_pack_layout && coarse_rows != 0 && deferred_quadrature_eligible()) {
      forward_pack_fine_ =
          Kokkos::subview(forward_pack_, pack_slice(0, fine_end), Kokkos::ALL);
      forward_pack_fine_spectrum_ =
          Kokkos::subview(forward_pack_spectrum_, pack_slice(0, fine_end), Kokkos::ALL);
      if (absorber_rows != 0 && fine_end - absorber_rows > temporal_columns) {
        forward_pack_fine_tail_ = Kokkos::subview(
            forward_pack_, pack_slice(absorber_rows, fine_end - absorber_rows),
            Kokkos::ALL);
        forward_pack_fine_tail_spectrum_ = Kokkos::subview(
            forward_pack_spectrum_, pack_slice(absorber_rows, fine_end - absorber_rows),
            Kokkos::ALL);
      }
      if (held_plasma()) {
        forward_pack_coarse_ = Kokkos::subview(
            forward_pack_, pack_slice(fine_end, pack_rows - fine_end), Kokkos::ALL);
        forward_pack_coarse_spectrum_ =
            Kokkos::subview(forward_pack_spectrum_,
                            pack_slice(fine_end, pack_rows - fine_end), Kokkos::ALL);
      }
    }
  }
  projected_spectrum_ =
      pack ? ComplexView2(Kokkos::subview(forward_pack_spectrum_,
                                          pack_slice(absorber_rows, temporal_columns),
                                          Kokkos::ALL))
           : ComplexView2("solprop_native.fft.projected_spectrum", temporal_columns,
                          projected_spectrum_size());
  if (!problem_.transform.projection_coarse_matrix.empty())
    projected_coarse_spectrum_ =
        coarse_rows != 0
            ? ComplexView2(Kokkos::subview(forward_pack_spectrum_,
                                           pack_slice(coarse_first, coarse_rows),
                                           Kokkos::ALL))
            : ComplexView2("solprop_native.fft.projected_coarse_spectrum", columns,
                           projected_spectrum_size());
  absorber_spectrum_ =
      absorber_rows != 0
          ? ComplexView2(Kokkos::subview(forward_pack_spectrum_,
                                         pack_slice(0, absorber_rows), Kokkos::ALL))
          : ComplexView2("solprop_native.fft.absorber_spectrum", temporal_columns,
                         projected_spectrum_size());
  if (problem_.dims.family == FieldFamily::field_real) {
    modal_real_ = RealView2("solprop_native.fft.modal_time_real", temporal_columns,
                            problem_.dims.n_to);
    node_real_ = separable_aliased_ ? modal_real_
                                    : RealView2("solprop_native.response.node_real",
                                                chunk_rows, problem_.dims.n_to);
  } else {
    modal_complex_ = ComplexView2("solprop_native.fft.modal_time_complex",
                                  temporal_columns, problem_.dims.n_to);
    node_complex_ = separable_aliased_
                        ? modal_complex_
                        : ComplexView2("solprop_native.response.node_complex",
                                       chunk_rows, problem_.dims.n_to);
    retained_synthesis_ = retained_synthesis();
    if (retained_synthesis_) {
      retained_modal_ = ComplexView2("solprop_native.transform.retained_modal",
                                     temporal_columns, problem_.dims.n_omega);
      retained_node_ = ComplexView2("solprop_native.transform.retained_node",
                                    chunk_rows, problem_.dims.n_omega);
    }
  }
  if (problem_.dims.family == FieldFamily::envelope) {
    projected_complex_ = ComplexView2("solprop_native.response.projected_complex",
                                      temporal_columns, problem_.dims.n_to);
    polarisation_complex_ =
        separable_aliased_
            ? projected_complex_
            : ComplexView2("solprop_native.response.polarisation_complex", chunk_rows,
                           problem_.dims.n_to);
    if (!problem_.transform.projection_coarse_matrix.empty())
      projected_coarse_complex_ =
          ComplexView2("solprop_native.response.projected_coarse_complex",
                       temporal_columns, problem_.dims.n_to);
    absorber_complex_ = ComplexView2("solprop_native.absorber.fine_complex",
                                     temporal_columns, problem_.dims.n_to);
  } else {
    projected_real_ =
        pack ? RealView2(Kokkos::subview(forward_pack_,
                                         pack_slice(absorber_rows, temporal_columns),
                                         Kokkos::ALL))
             : RealView2("solprop_native.response.projected_real", temporal_columns,
                         problem_.dims.n_to);
    polarisation_real_ = separable_aliased_
                             ? projected_real_
                             : RealView2("solprop_native.response.polarisation_real",
                                         chunk_rows, problem_.dims.n_to);
    if (!problem_.transform.projection_coarse_matrix.empty())
      projected_coarse_real_ =
          coarse_rows != 0
              ? RealView2(Kokkos::subview(
                    forward_pack_, pack_slice(coarse_first, coarse_rows), Kokkos::ALL))
              : RealView2("solprop_native.response.projected_coarse_real",
                          temporal_columns, problem_.dims.n_to);
    absorber_real_ =
        absorber_rows != 0
            ? RealView2(Kokkos::subview(forward_pack_, pack_slice(0, absorber_rows),
                                        Kokkos::ALL))
            : RealView2("solprop_native.absorber.fine_real", temporal_columns,
                        problem_.dims.n_to);
  }
  if (field_plasma())
    held_current_ =
        RealView2("solprop_native.held.current", chunk_rows, problem_.dims.n_to);
  if (gated_plasma()) {
    gated_prefix_ =
        RealView2("solprop_native.gated.prefix", chunk_rows, problem_.dims.n_to);
    if (problem_.plasma.band)
      gated_filtered_ =
          RealView2("solprop_native.gated.filtered", chunk_rows, problem_.dims.n_to);
  }
  if (held_plasma()) {
    held_derivative_ = RealView("solprop_native.held.derivative", problem_.dims.n_to);
    held_inverse_ = RealView("solprop_native.held.inverse", problem_.dims.n_omega);
    const bool radial = held_radial();
    const auto rows = radial ? chunk_rows : temporal_columns;
    const bool coarse = !problem_.transform.projection_coarse_matrix.empty();
    // On a device the two currents' fine and coarse rows are one buffer, a stream's
    // fine rows then its coarse ones, so that one transform takes all four: on an AMD
    // Instinct MI325X each transform of a single-mode run was four library kernels of a
    // few microseconds, launched one by one. On the radial route the three streams'
    // fine rows are one buffer, which a node chunk's one transform takes.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    const bool packed = !radial;
    const bool radial_packed = radial;
#else
    const bool packed = false;
    const bool radial_packed = false;
#endif
    // A host's pack holds the streams' rows itself, each fine block then each coarse.
    const bool host_held = host_pack_layout && pack && !radial;
    if (packed && pack) {
      // The stage's forward pack ends with the held pack's rows.
      const auto held_rows =
          pack_slice(absorber_rows + rows + coarse_rows, 2 * (rows + coarse_rows));
      held_pack_ = Kokkos::subview(forward_pack_, held_rows, Kokkos::ALL);
      held_pack_spectrum_ =
          Kokkos::subview(forward_pack_spectrum_, held_rows, Kokkos::ALL);
    } else if (packed) {
      const std::size_t held_rows = 2 * (coarse ? 2 * rows : rows);
      held_pack_ = RealView2("solprop_native.held.pack", held_rows, problem_.dims.n_to);
      held_pack_spectrum_ = ComplexView2("solprop_native.held.pack_spectrum", held_rows,
                                         problem_.dims.n_to / 2 + 1);
    } else if (radial_packed) {
      held_pack_ = RealView2("solprop_native.held.pack", 3 * rows, problem_.dims.n_to);
      held_pack_spectrum_ = ComplexView2("solprop_native.held.pack_spectrum", 3 * rows,
                                         problem_.dims.n_to / 2 + 1);
    }
    for (std::size_t index = 0; index < (radial ? 3U : 2U); ++index) {
      auto& stream = held_streams_[index];
      const auto label =
          std::string("solprop_native.held.stream") + std::to_string(index) + ".";
      if (index < 2)
        stream.node = RealView2(label + "node", chunk_rows, problem_.dims.n_to);
      if (host_held && index < 2) {
        const auto fine = pack_slice(absorber_rows + rows + index * rows, rows);
        stream.projected = Kokkos::subview(forward_pack_, fine, Kokkos::ALL);
        stream.spectrum = Kokkos::subview(forward_pack_spectrum_, fine, Kokkos::ALL);
        if (coarse) {
          const auto held_coarse = pack_slice(fine_end + rows + index * rows, rows);
          stream.coarse = Kokkos::subview(forward_pack_, held_coarse, Kokkos::ALL);
          stream.coarse_spectrum =
              Kokkos::subview(forward_pack_spectrum_, held_coarse, Kokkos::ALL);
          stream.stacked = RealView2(label + "stacked", 2 * rows, problem_.dims.n_to);
        }
        continue;
      }
      if (packed) {
        // Each of the two streams holds half of the pack.
        const std::size_t first = index * (held_pack_.extent(0) / 2);
        const auto rows_from = [&](const std::size_t start, const std::size_t count) {
          return std::pair<std::size_t, std::size_t>(start, start + count);
        };
        stream.projected =
            Kokkos::subview(held_pack_, rows_from(first, rows), Kokkos::ALL);
        stream.spectrum =
            Kokkos::subview(held_pack_spectrum_, rows_from(first, rows), Kokkos::ALL);
        if (coarse) {
          stream.coarse =
              Kokkos::subview(held_pack_, rows_from(first + rows, rows), Kokkos::ALL);
          stream.coarse_spectrum = Kokkos::subview(
              held_pack_spectrum_, rows_from(first + rows, rows), Kokkos::ALL);
          stream.stacked = RealView2(label + "stacked", 2 * rows, problem_.dims.n_to);
        }
        continue;
      }
      if (radial_packed) {
        // Each of the three streams holds a third of the pack.
        const auto third = pack_slice(index * rows, rows);
        stream.projected = Kokkos::subview(held_pack_, third, Kokkos::ALL);
        stream.spectrum = Kokkos::subview(held_pack_spectrum_, third, Kokkos::ALL);
      } else {
        stream.projected = RealView2(label + "projected", rows, problem_.dims.n_to);
        stream.spectrum =
            ComplexView2(label + "spectrum", rows, problem_.dims.n_to / 2 + 1);
      }
      if (coarse) {
        stream.coarse = RealView2(label + "coarse", rows, problem_.dims.n_to);
        stream.coarse_spectrum =
            ComplexView2(label + "coarse_spectrum", rows, problem_.dims.n_to / 2 + 1);
        stream.stacked = RealView2(label + "stacked", 2 * rows, problem_.dims.n_to);
      }
    }
    if (radial) {
      held_nodal_spectrum_ = ComplexView2("solprop_native.held.nodal_spectrum",
                                          chunk_rows, problem_.dims.n_omega);
      held_projected_spectrum_ =
          ComplexView2("solprop_native.held.projected_spectrum", problem_.n_columns(),
                       problem_.dims.n_omega);
      held_projection_ =
          RealView2("solprop_native.held.projection", problem_.dims.n_modes,
                    problem_.dims.node_chunk * problem_.dims.n_pol_transform);
    }
  }
  absorber_fine_ =
      RealView("solprop_native.absorber.rate_fine", problem_.absorber.fine.size());
  absorber_coarse_ =
      RealView("solprop_native.absorber.rate_coarse", problem_.absorber.coarse.size());
  if (!problem_.spatial.absorber_matrix.empty()) {
    if (problem_.dims.family == FieldFamily::envelope) {
      spatial_absorber_complex_ =
          ComplexView2("solprop_native.absorber.spatial_complex", problem_.dims.n_modes,
                       problem_.dims.n_modes);
      std::vector<Complex> matrix;
      matrix.reserve(problem_.spatial.absorber_matrix.size());
      for (const auto value : problem_.spatial.absorber_matrix)
        matrix.emplace_back(value, 0.0);
      copy_vector(execution_space_, spatial_absorber_complex_, matrix);
    } else {
      spatial_absorber_real_ = RealView2("solprop_native.absorber.spatial_real",
                                         problem_.dims.n_modes, problem_.dims.n_modes);
      copy_vector(execution_space_, spatial_absorber_real_,
                  problem_.spatial.absorber_matrix);
    }
  }
  if (problem_.plasma.envelope) {
    envelope_rule_nodes_ = RealView3("solprop_native.envelope.rule_nodes",
                                     problem_.plasma.n_channels, 4, 512);
    envelope_rule_weights_ = RealView3("solprop_native.envelope.rule_weights",
                                       problem_.plasma.n_channels, 4, 512);
    envelope_moments_ = RealView3("solprop_native.envelope.moments", 2,
                                  problem_.dims.node_chunk, problem_.dims.n_to);
    envelope_rates_ = RealView2("solprop_native.envelope.rates",
                                problem_.dims.node_chunk, problem_.dims.n_to);
    envelope_exposure_ = RealView2("solprop_native.envelope.exposure",
                                   problem_.dims.node_chunk, problem_.dims.n_to);
    envelope_failures_ = IndexView("solprop_native.envelope.failures",
                                   problem_.dims.node_chunk * problem_.dims.n_to);
    envelope_status_ = IndexView("solprop_native.envelope.status", 1);
    envelope_host_status_ = Kokkos::View<std::size_t*, Layout, Kokkos::HostSpace>(
        "solprop_native.envelope.host_status", 1);
    envelope_drude_factor_ =
        RealView("solprop_native.envelope.drude_factor", problem_.dims.n_omega);
    envelope_loss_factor_ =
        ComplexView("solprop_native.envelope.loss_factor", problem_.dims.n_omega);
    for (std::size_t index = 0; index < envelope_streams_.size(); ++index) {
      auto& stream = envelope_streams_[index];
      const auto label =
          std::string("solprop_native.envelope.") + (index == 0 ? "drude." : "loss.");
      stream.node = ComplexView2(label + "node", chunk_rows, problem_.dims.n_to);
      stream.projected =
          ComplexView2(label + "projected", temporal_columns, problem_.dims.n_to);
      stream.spectrum =
          ComplexView2(label + "spectrum", temporal_columns, problem_.dims.n_to);
      if (!problem_.transform.projection_coarse_matrix.empty()) {
        stream.coarse =
            ComplexView2(label + "coarse", temporal_columns, problem_.dims.n_to);
        stream.coarse_spectrum = ComplexView2(label + "coarse_spectrum",
                                              temporal_columns, problem_.dims.n_to);
      }
      if (!projected_stacked_complex_.empty())
        stream.stacked =
            ComplexView2(label + "stacked", 2 * temporal_columns, problem_.dims.n_to);
    }
  }
  if (problem_.plasma.enabled) {
    if (problem_.plasma.n_channels > 1 || problem_.plasma.envelope) {
      channel_parameters_ = RealView2("solprop_native.plasma.channel_parameters",
                                      problem_.plasma.n_channels, 9);
      channel_coefficients_ = RealView("solprop_native.plasma.channel_coefficients",
                                       problem_.plasma.channel_coefficients.size());
      if (!problem_.plasma.envelope)
        channel_scratch_ = decltype(channel_scratch_)(
            "solprop_native.plasma.channel_scratch", problem_.plasma.n_channels,
            problem_.dims.node_chunk, 4);
    }
    rate_coefficients_ = RealView("solprop_native.plasma.rate_coefficients",
                                  problem_.plasma.rate_coefficients.size());
    plasma_statistics_scratch_ =
        RealView2("solprop_native.plasma.statistics_scratch", problem_.dims.n_nodes,
                  1 + 2 * problem_.plasma.n_channels);
    if (!problem_.plasma.envelope) {
      // solprop SEP 18: the rule's taps and the run diagnostics' six parts per node.
      plasma_taps_ =
          RealView("solprop_native.plasma.taps", problem_.plasma.taps.size());
      plasma_diagnostics_ = RealView2("solprop_native.plasma.diagnostics",
                                      problem_.dims.n_nodes, plasma_diagnostic_parts);
    }
    if (!problem_.plasma.envelope) {
      // The band rule's team column and its statistics (`team_band_plasma_column`).
      // A device keeps the trapezoid device column's rows, one per node of a chunk,
      // and the band column reuses them. A host team owns one row of each per slot,
      // and at most one team per thread runs at once (solprop SEP 25 section
      // 6.7.5): the rate, exposure and filtered rows for every field plasma (the
      // trapezoid's run diagnostics come from the band statistics), and under the
      // band rule the guard, a mixture's weight and a row per component. The
      // tiled column (solprop SEP 25 section 6.7.6) keeps the device's rows on every
      // build, the weight rows holding the free-current local prefixes, and the tiles'
      // totals by row.
      const bool tiled = band_tiled_ && problem_.plasma.band;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
      band_slots_ = problem_.dims.node_chunk;
      const auto guard_rows = chunk_rows;
      const auto weight_rows = chunk_rows;
      const auto component_rows = chunk_rows;
#else
      if (tiled) {
        band_slots_ = problem_.dims.node_chunk;
      } else {
        band_slots_ = std::min(problem_.dims.node_chunk,
                               static_cast<std::size_t>(std::max(
                                   1, response_execution_space_.concurrency())));
      }
      const bool band = problem_.plasma.band;
      const auto guard_rows = tiled ? chunk_rows : band ? band_slots_ : 0;
      const auto weight_rows = tiled                                    ? chunk_rows
                               : band && problem_.plasma.n_channels > 1 ? band_slots_
                                                                        : 0;
      const auto component_rows =
          band ? checked_product(band_slots_, problem_.effective_node_pol(),
                                 "band component rows")
               : 0;
#endif
      if (tiled) {
        // The widest tile whose two scratch rows the device's team scratch holds.
        using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
        const auto scratch_limit =
            static_cast<std::size_t>(Policy::scratch_size_max(0));
        std::size_t largest = band_tile_max;
        while (largest > band_tile_min &&
               band_tile_bytes<ExecutionSpace>(largest) > scratch_limit)
          largest /= 2;
        if (band_tile_bytes<ExecutionSpace>(largest) > scratch_limit)
          throw std::length_error("the band tile exceeds the team scratch");
        band_tile_ = band_tile_samples(problem_.dims.n_to, largest);
        band_tile_count_ = band_tile_count(problem_.dims.n_to, band_tile_);
        require_allocation(checked_product(chunk_rows, band_tile_count_, "band tiles"),
                           sizeof(double), "band-tile bytes");
        band_tile_totals_ = RealView2("solprop_native.plasma.band_tile_totals",
                                      chunk_rows, band_tile_count_);
      }
      const auto band_rows = checked_sum(
          checked_sum(checked_product(3, band_slots_, "band rows"), guard_rows,
                      "band rows"),
          checked_sum(weight_rows, component_rows, "band rows"), "band rows");
      require_allocation(checked_product(band_rows, problem_.dims.n_to, "band rows"),
                         sizeof(double), "band-row bytes");
      plasma_rate_ =
          RealView2("solprop_native.plasma.rate", band_slots_, problem_.dims.n_to);
      plasma_exposure_ =
          RealView2("solprop_native.plasma.exposure", band_slots_, problem_.dims.n_to);
      plasma_fraction_ =
          RealView2("solprop_native.plasma.fraction", band_slots_, problem_.dims.n_to);
      if (component_rows != 0) {
        plasma_current_integrand_ = RealView2("solprop_native.plasma.current_integrand",
                                              component_rows, problem_.dims.n_to);
      }
      if (guard_rows != 0)
        plasma_free_current_ = RealView2("solprop_native.plasma.free_current",
                                         guard_rows, problem_.dims.n_to);
      if (weight_rows != 0)
        plasma_outer_current_ = RealView2("solprop_native.plasma.outer_current",
                                          weight_rows, problem_.dims.n_to);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
      if (!problem_.plasma.band) {
        plasma_tiles_ =
            (problem_.dims.n_to + plasma_tile_samples - 1) / plasma_tile_samples;
        plasma_tile_totals_ =
            RealView2("solprop_native.plasma.tile_totals", chunk_rows, plasma_tiles_);
      }
#endif
    }
    if (problem_.plasma.stash) {
      stash_candidate_ =
          RealView2("solprop_native.plasma.stash_candidate", problem_.dims.n_nodes,
                    1 + 2 * problem_.plasma.n_channels);
      stash_current_ =
          RealView2("solprop_native.plasma.stash_current", problem_.dims.n_nodes,
                    1 + 2 * problem_.plasma.n_channels);
    }
  }
  if (problem_.absorber.split_step) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    split_factor_ = RealView("solprop_native.absorber.split_factor", problem_.dims.n_t);
#endif
    if (problem_.free_geometry() && problem_.dims.family == FieldFamily::envelope) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
      // In place on the state's rows, a block holds no buffer of its own, only its
      // plans' work area (9 MiB on cuFFT for every row at 768 x 768 x 600), so a device
      // takes every row at once whatever the node budget: 14 kernels and 10 transforms
      // a step there, against 119 and 220 at the node chunk (solprop #361). A CPU keeps
      // the chunk, where FFTW's threaded plan of a whole state was much the slower.
      split_rows_ = problem_.n_columns();
#else
      split_rows_ = temporal_columns;
#endif
    } else {
      coarse_spectrum_ =
          ComplexView2("solprop_native.absorber.coarse_spectrum", temporal_columns, n);
      if (problem_.dims.family == FieldFamily::envelope)
        coarse_time_complex_ =
            ComplexView2("solprop_native.absorber.coarse_time_complex",
                         temporal_columns, problem_.dims.n_t);
      else
        coarse_time_real_ = RealView2("solprop_native.absorber.coarse_time_real",
                                      temporal_columns, problem_.dims.n_t);
    }
  }
  if (problem_.general.enabled) {
    // solprop SEP 47: a general modal basis's statistics whiten the columns by the
    // metric's root (`whitened_statistics`), into scratch held from the start.
    statistics_whitened_ = ComplexView2("solprop_native.statistics.whitened",
                                        problem_.n_columns(), problem_.dims.n_omega);
    statistics_root_ = ComplexView(
        "solprop_native.statistics.metric_root",
        checked_product(problem_.dims.n_omega,
                        checked_product(problem_.dims.n_modes, problem_.dims.n_modes,
                                        "metric root"),
                        "metric root"));
    statistics_whitened_sums_ = RealView("solprop_native.statistics.whitened_sums",
                                         problem_.n_columns() + problem_.dims.n_omega);
    general_ = std::make_unique<GeneralModalSource>(problem_);
    // The source owns execution-space copies; the host maps are the largest arrays a
    // general guide carries and nothing reads them again.
    std::vector<Complex>().swap(problem_.general.synthesis);
    std::vector<Complex>().swap(problem_.general.projection);
    std::vector<double>().swap(problem_.general.tensor);
    std::vector<Complex>().swap(problem_.general.frequency_factor);
    std::vector<double>().swap(problem_.general.delayed_tensor);
    std::vector<Complex>().swap(problem_.general.delayed_spectrum);
    std::vector<std::size_t>().swap(problem_.general.node_response);
  }
  if (problem_.free_geometry() && !problem_.spatial.spectral_weights.empty())
    far_field_ = std::make_unique<FarFieldPass>(problem_, execution_space_);
  if (problem_.dispersive.enabled && problem_.separable_geometry()) {
    // The planar kind's slab rule in x; free space has none.
    const bool slab = problem_.slab_geometry();
    separable_dispersive_ = std::make_unique<SeparableDispersive>(
        problem_, kernel_pool_,
        SeparableDispersive::StageViews{.fine_spectrum = fine_spectrum_,
                                        .projected_spectrum = projected_spectrum_,
                                        .node_real = node_real_,
                                        .polarisation_real = polarisation_real_,
                                        .node_complex = node_complex_,
                                        .polarisation_complex = polarisation_complex_,
                                        .synthesis = slab ? synthesis_ : RealView2{},
                                        .projection = slab ? projection_ : RealView2{},
                                        .time_window = time_window_,
                                        .carrier = carrier_});
  } else if (problem_.dispersive.enabled && !problem_.general.enabled) {
    dispersive_ = std::make_unique<DispersiveResponse>(
        problem_, kernel_pool_,
        DispersiveResponse::StageViews{
            .fine_spectrum = fine_spectrum_,
            .projected_spectrum = projected_spectrum_,
            .projected_coarse_spectrum = projected_coarse_spectrum_,
            .node_real = node_real_,
            .polarisation_real = polarisation_real_,
            .node_complex = node_complex_,
            .polarisation_complex = polarisation_complex_,
            .scaled_synthesis = scaled_synthesis_,
            .scaled_projection = scaled_projection_,
            .scaled_projection_coarse = scaled_projection_coarse_,
            .time_window = time_window_,
            .carrier = carrier_});
  }
  if (problem_.dispersive.enabled) {
    // The response holds execution-space copies of its layout (the general route's
    // source built its own above).
    std::vector<double>().swap(problem_.dispersive.inputs);
    std::vector<double>().swap(problem_.dispersive.term_multipliers);
    std::vector<double>().swap(problem_.dispersive.quartics);
  }
  if (problem_.flow.enabled) {
    flow_ = std::make_unique<CoupledFlowAction>(problem_, flow_execution_space_);
    std::vector<Complex>().swap(problem_.flow.frozen);
    std::vector<Complex>().swap(problem_.flow.residual_terms);
    std::vector<Complex>().swap(problem_.flow.left);
    std::vector<Complex>().swap(problem_.flow.right);
    std::vector<Complex>().swap(problem_.flow.eigenvalues);
    std::vector<std::size_t>().swap(problem_.flow.route);
    std::vector<Complex>().swap(problem_.flow.fallback_vectors);
    std::vector<Complex>().swap(problem_.flow.fallback_triangular);
  }
}

void GeneralWorkspace::upload() {
  if (problem_.raman.n_components > 1) {
    copy_vector(execution_space_, raman_row_component_, problem_.raman.row_component);
    copy_vector(execution_space_, raman_component_fraction_,
                problem_.raman.component_fraction);
  }
  if (problem_.separable_geometry()) {
    copy_vector(execution_space_, spatial_.square, problem_.spatial.k_squared);
    copy_vector(execution_space_, spatial_.square_low, problem_.spatial.k_squared_low);
    copy_vector(execution_space_, spatial_.numerator, problem_.spatial.numerator);
    copy_vector(execution_space_, spatial_.clamped, problem_.spatial.clamped);
    copy_vector(execution_space_, spatial_.transverse, problem_.spatial.kperp_sq);
    copy_vector(execution_space_, spatial_.transverse_low,
                problem_.spatial.kperp_sq_low);
    copy_vector(execution_space_, spatial_.frame, problem_.spatial.frame);
    copy_vector(execution_space_, spatial_.spectral_rate,
                problem_.spatial.spectral_rate);
    copy_vector(execution_space_, spatial_.user_rate, problem_.spatial.user_rate);
    copy_vector(execution_space_, spatial_.absorber, problem_.spatial.absorber_rate);
    copy_vector(execution_space_, spatial_.transverse_rate,
                problem_.spatial.kperp_rate);
    if (problem_.slab_geometry())
      copy_vector(execution_space_, spatial_.kappa, problem_.spatial.kappa_sq);
  } else if (!problem_.flux.k_squared.empty()) {
    copy_vector(execution_space_, flux_.square, problem_.flux.k_squared);
    copy_vector(execution_space_, flux_.square_low, problem_.flux.k_squared_low);
    copy_vector(execution_space_, flux_.transverse, problem_.flux.kperp_sq);
    copy_vector(execution_space_, flux_.transverse_low, problem_.flux.kperp_sq_low);
  }
  if (problem_.plasma.n_channels > 1 || problem_.plasma.envelope) {
    copy_vector(execution_space_, channel_parameters_,
                problem_.plasma.channel_parameters);
    copy_vector(execution_space_, channel_coefficients_,
                problem_.plasma.channel_coefficients);
  }
  copy_vector(execution_space_, crop_index_, problem_.grid.crop_index);
  if (!fine_gaps_.empty()) {
    std::vector<std::size_t> runs;
    std::size_t position = 0;
    for (const auto& [bin, length] : fine_gap_runs()) {
      runs.insert(runs.end(), {position, bin, length});
      position += length;
    }
    copy_vector(execution_space_, fine_gaps_, runs);
  }
  if (problem_.plasma.envelope) {
    copy_vector(execution_space_, envelope_rule_nodes_,
                problem_.plasma.envelope_rule_nodes);
    copy_vector(execution_space_, envelope_rule_weights_,
                problem_.plasma.envelope_rule_weights);
    copy_vector(execution_space_, envelope_drude_factor_,
                problem_.plasma.envelope_drude_factor);
    copy_vector(execution_space_, envelope_loss_factor_,
                problem_.plasma.envelope_loss_factor);
  }
  copy_vector(execution_space_, active_mask_, problem_.grid.active_mask);
  copy_vector(execution_space_, norm_mask_, problem_.grid.norm_mask);
  copy_vector(execution_space_, omega_window_, problem_.grid.omega_window);
  copy_vector(execution_space_, omega_, problem_.grid.omega);
  copy_vector(execution_space_, time_window_, problem_.grid.time_window);
  if (held_plasma()) {
    copy_vector(execution_space_, held_derivative_,
                problem_.grid.time_window_derivative);
    std::vector<double> inverse(problem_.dims.n_omega, 0.0);
    for (std::size_t index = 0; index < problem_.dims.n_omega; ++index)
      if (problem_.grid.active_mask[index] != 0)
        inverse[index] = 1.0 / problem_.grid.omega[index];
    copy_vector(execution_space_, held_inverse_, inverse);
  }
  copy_vector(execution_space_, carrier_, problem_.grid.carrier);
  // A z table's slots start at zero (Kokkos initialises a view) until evaluated.
  lossless_phase_ = false;
  if (!problem_.linear.phi_table) {
    copy_vector(execution_space_, phi_terms_, problem_.linear.phi_terms);
    lossless_phase_ =
        std::ranges::all_of(problem_.linear.phi_terms,
                            [](const Complex& term) { return term.real() == 0.0; });
  }
  if (!problem_.linear.prefactor_table)
    copy_vector(execution_space_, prefactor_terms_, problem_.linear.prefactor_terms);
  copy_vector(execution_space_, tableau_, problem_.stepper.tableau_matrix);
  copy_vector(execution_space_, spatial_spectral_weights_,
              problem_.spatial.spectral_weights);
  copy_vector(execution_space_, spatial_on_axis_, problem_.spatial.on_axis);
  copy_vector(execution_space_, synthesis_, problem_.transform.synthesis_matrix);
  copy_vector(execution_space_, projection_, problem_.transform.projection_matrix);
  if (!problem_.transform.projection_coarse_matrix.empty())
    copy_vector(execution_space_, projection_coarse_,
                problem_.transform.projection_coarse_matrix);
  copy_vector(execution_space_, absorber_fine_, problem_.absorber.fine);
  spatial_only_absorber_ = std::ranges::all_of(
      problem_.absorber.fine, [](const double rate) { return rate == 0.0; });
  copy_vector(execution_space_, absorber_coarse_, problem_.absorber.coarse);
  if (problem_.plasma.enabled)
    copy_vector(execution_space_, rate_coefficients_,
                problem_.plasma.rate_coefficients);
  if (!plasma_taps_.empty())
    copy_vector(execution_space_, plasma_taps_, problem_.plasma.taps);
  if (problem_.raman.enabled) {
    copy_vector(execution_space_, raman_undamped_, problem_.raman.undamped);
    copy_vector(execution_space_, raman_tensor_weights_, problem_.raman.tensor_weights);
    graph::deep_copy(execution_space_, node_raman_drive_, 0.0);
  }
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
// Measured plans for free space's batched transforms (SEP 28): the transverse pair in
// place on the scratch and, on the complex fields, the chunk's temporal pair, each at
// the plan policy's effort for its shape and width (`fftw_seed_effort`). The scratch
// buffers are cleared after planning overwrote them.
void GeneralWorkspace::seed_free_wisdom() {
  const auto planes = static_cast<int>(transverse_.extent(0));
  const auto n_y = static_cast<int>(transverse_.extent(1));
  const auto n_x = static_cast<int>(transverse_.extent(2));
  for (const int sign : {FFTW_FORWARD, FFTW_BACKWARD})
    seed_fftw_wisdom(fft_space_for_rows(transverse_), transverse_.data(),
                     transverse_.data(), {n_y, n_x}, planes, n_y * n_x, n_y * n_x,
                     sign);
  graph::deep_copy(execution_space_, transverse_, Complex{});
  // The chunk's temporal pair is the modal routes' (`separable_to_time`,
  // `separable_to_frequency`), on every field family: the real family's pair and the
  // analytic family's forward transform were left at FFTW_ESTIMATE (solprop #560).
  // A fine spectrum wider than n_to is not a plain row pair; the helper leaves it.
  seed_modal_wisdom();
}

// Measured plans for the modal routes' temporal pair, the two transforms of every
// source evaluation: Kokkos-FFT plans them at FFTW_ESTIMATE, which picks markedly
// slower codelets for long rows (a 16384-sample envelope GNLSE ran its pair at about
// twice the measured cost). The plan policy (`fftw_seed_effort`) decides the effort.
void GeneralWorkspace::seed_modal_wisdom() {
  if (problem_.dims.family == FieldFamily::field_real) {
    seed_row_fft_wisdom(fft_space_for_rows(modal_real_), fine_spectrum_, modal_real_,
                        KokkosFFT::Direction::backward);
    graph::deep_copy(execution_space_, modal_real_, 0.0);
  } else {
    seed_row_fft_wisdom(fft_space_for_rows(modal_complex_), fine_spectrum_,
                        modal_complex_, KokkosFFT::Direction::backward);
    graph::deep_copy(execution_space_, modal_complex_, Complex{});
  }
  if (problem_.dims.family == FieldFamily::envelope) {
    seed_row_fft_wisdom(fft_space_for_rows(projected_complex_), projected_complex_,
                        projected_spectrum_, KokkosFFT::Direction::forward);
    graph::deep_copy(execution_space_, projected_complex_, Complex{});
  } else {
    seed_row_fft_wisdom(fft_space_for_rows(projected_real_), projected_real_,
                        projected_spectrum_, KokkosFFT::Direction::forward);
    graph::deep_copy(execution_space_, projected_real_, 0.0);
  }
  graph::deep_copy(execution_space_, fine_spectrum_, Complex{});
  graph::deep_copy(execution_space_, projected_spectrum_, Complex{});
}

// The plasma streams' forward rows, fine and coarse. Most share a shape with the
// projected pair seeded above, whose wisdom FFTW then finds at once; the radial held
// streams' chunk rows do not.
void GeneralWorkspace::seed_stream_wisdom() {
  // The forward pack's plans (`packs_forward`); seeding overwrites their rows.
  for (const auto& [rows, spectrum] :
       {std::pair{forward_pack_, forward_pack_spectrum_},
        std::pair{forward_pack_tail_, forward_pack_tail_spectrum_},
        std::pair{forward_pack_fine_, forward_pack_fine_spectrum_},
        std::pair{forward_pack_fine_tail_, forward_pack_fine_tail_spectrum_},
        std::pair{forward_pack_coarse_, forward_pack_coarse_spectrum_}}) {
    if (rows.empty())
      continue;
    seed_row_fft_wisdom(fft_space_for_rows(rows), rows, spectrum,
                        KokkosFFT::Direction::forward);
  }
  if (!forward_pack_.empty()) {
    graph::deep_copy(execution_space_, forward_pack_, 0.0);
    graph::deep_copy(execution_space_, forward_pack_spectrum_, Complex{});
  }
  if (held_plasma())
    for (auto& stream : held_streams_) {
      if (stream.projected.empty())
        continue;
      seed_row_fft_wisdom(fft_space_for_rows(stream.projected), stream.projected,
                          stream.spectrum, KokkosFFT::Direction::forward);
      graph::deep_copy(execution_space_, stream.projected, 0.0);
      graph::deep_copy(execution_space_, stream.spectrum, Complex{});
      if (stream.coarse.empty())
        continue;
      seed_row_fft_wisdom(fft_space_for_rows(stream.coarse), stream.coarse,
                          stream.coarse_spectrum, KokkosFFT::Direction::forward);
      graph::deep_copy(execution_space_, stream.coarse, 0.0);
      graph::deep_copy(execution_space_, stream.coarse_spectrum, Complex{});
    }
  if (problem_.plasma.envelope)
    for (auto& stream : envelope_streams_) {
      seed_row_fft_wisdom(fft_space_for_rows(stream.projected), stream.projected,
                          stream.spectrum, KokkosFFT::Direction::forward);
      graph::deep_copy(execution_space_, stream.projected, Complex{});
      graph::deep_copy(execution_space_, stream.spectrum, Complex{});
      if (stream.coarse.empty())
        continue;
      seed_row_fft_wisdom(fft_space_for_rows(stream.coarse), stream.coarse,
                          stream.coarse_spectrum, KokkosFFT::Direction::forward);
      graph::deep_copy(execution_space_, stream.coarse, Complex{});
      graph::deep_copy(execution_space_, stream.coarse_spectrum, Complex{});
    }
}
#endif

void GeneralWorkspace::plan() {
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  // Seed every measured shape before the first plan exists: a Kokkos-FFT plan made
  // before its wisdom stays at FFTW_ESTIMATE for the run.
  if (problem_.free_geometry())
    seed_free_wisdom();
  else if (!problem_.separable_geometry())
    seed_modal_wisdom();
  seed_stream_wisdom();
#endif
  fft_execution_space_ = modal_complex_.size() > modal_real_.size()
                             ? fft_space_for_rows(modal_complex_)
                             : fft_space_for_rows(modal_real_);
  if (problem_.raman.enabled) {
    raman_execution_space_ = raman_pool_.get(
        CpuOperation::raman, node_raman_drive_.size(), row_length(node_raman_drive_));
    const auto raman_tail = problem_.dims.n_nodes % problem_.dims.node_chunk;
    raman_tail_execution_space_ = raman_tail_pool_.get(
        CpuOperation::raman,
        raman_tail * problem_.raman_drives() * 2 * problem_.dims.n_to,
        2 * problem_.dims.n_to);
    raman_kernel_execution_space_ = raman_pool_.get(
        CpuOperation::raman, raman_kernel_time_.size(), row_length(raman_kernel_time_));
    raman_multiply_space_ =
        kernel_pool_.get(CpuOperation::stage, node_raman_spectrum_.size());
    raman_tail_multiply_space_ =
        kernel_pool_.get(CpuOperation::stage, raman_tail * problem_.raman_drives() *
                                                  (problem_.dims.n_to + 1));
    raman_rebuild_space_ = kernel_pool_.get(
        CpuOperation::response, raman_kernel_time_.size() * problem_.raman.n_terms);
    raman_scale_space_ =
        kernel_pool_.get(CpuOperation::stage, raman_kernel_spectrum_.size());
    raman_products_ = raman_modal_products();
    if (raman_products_ != 0) {
      raman_product_execution_space_ = raman_tail_pool_.get(
          CpuOperation::raman, raman_products_ * 2 * problem_.dims.n_to,
          2 * problem_.dims.n_to);
      raman_product_space_ = kernel_pool_.get(CpuOperation::stage,
                                              raman_products_ * 2 * problem_.dims.n_to);
    }
  }
  if (general_)
    general_->plan();
  if (dispersive_)
    dispersive_->plan(fft_pool_);
  if (separable_dispersive_)
    separable_dispersive_->plan(fft_pool_);
  if (problem_.slab_geometry()) {
    // The y transforms are many short ones; below the threaded-plan threshold they
    // run on one FFTW thread, as every other small plan (`fft_space_for`).
    constexpr int y_axis = -1;
    const auto& space = fft_space_for_rows(transverse_);
    slab_forward_ = std::make_unique<SlabPlan>(space, transverse_, transverse_,
                                               KokkosFFT::Direction::forward, y_axis);
    slab_inverse_ = std::make_unique<SlabPlan>(space, transverse_, transverse_,
                                               KokkosFFT::Direction::backward, y_axis);
  } else if (problem_.free_geometry()) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    constexpr std::size_t axes = 2;
#else
    const KokkosFFT::axis_type<2> axes{-2, -1};
#endif
    transverse_forward_ = std::make_unique<TransversePlan>(
        fft_space_for_rows(transverse_), transverse_, transverse_,
        KokkosFFT::Direction::forward, axes);
    transverse_inverse_ = std::make_unique<TransversePlan>(
        fft_space_for_rows(transverse_), transverse_, transverse_,
        KokkosFFT::Direction::backward, axes);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    bool identity_crop = problem_.grid.crop_index.size() == problem_.dims.n_omega;
    for (std::size_t index = 0; identity_crop && index < problem_.dims.n_omega; ++index)
      identity_crop = problem_.grid.crop_index[index] == index;
    volume_kerr_ = identity_crop && problem_.dims.family == FieldFamily::envelope &&
                   problem_.dims.n_pol_state == 1 &&
                   problem_.effective_node_pol() == 1 &&
                   problem_.dims.n_to == problem_.dims.n_omega &&
                   problem_.dims.n_t == problem_.dims.n_to && separable_aliased_ &&
                   problem_.kerr.nonlinear && !problem_.plasma.enabled &&
                   !problem_.raman.enabled && !problem_.plasma.envelope &&
                   !problem_.dispersive.enabled && separable_dispersive_ == nullptr &&
                   problem_.kerr.harmonic_factor == 0.0 && separable_gaps_.empty();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    // At 768 x 768 x 600 on an NVIDIA H200, cuFFT's `{N,N,T}` takes 14.7 ms a transform
    // against 17.6 ms for `{T,N,N}` (solprop #365). An in-stage absorber evaluates its
    // term from the same input in the scratch's order, so it keeps that order.
    volume_order_ = volume_kerr_ && !problem_.absorber.in_stage;
#endif
    if (volume_order_)
      volume_scratch_ = ComplexView3(transverse_.data(), problem_.spatial.n_y,
                                     problem_.spatial.n_x, problem_.dims.n_omega);
    if (volume_kerr_) {
      constexpr std::size_t volume_axes = 3;
      const auto& volume = volume_order_ ? volume_scratch_ : transverse_;
      volume_forward_ = std::make_unique<VolumePlan>(
          execution_space_, volume, volume, KokkosFFT::Direction::forward, volume_axes);
      volume_inverse_ =
          std::make_unique<VolumePlan>(execution_space_, volume, volume,
                                       KokkosFFT::Direction::backward, volume_axes);
    }
    std::size_t work = KokkosFFT::compute_required_workspace_size<Complex>(
        *transverse_forward_, *transverse_inverse_);
    if (volume_kerr_)
      work = std::max(work, KokkosFFT::compute_required_workspace_size<Complex>(
                                *volume_forward_, *volume_inverse_));
    if (work != 0) {
      transform_work_ =
          ComplexView(Kokkos::view_alloc("solprop_native.free.transform_work",
                                         Kokkos::WithoutInitializing),
                      work);
      transverse_forward_->set_work_area(transform_work_);
      transverse_inverse_->set_work_area(transform_work_);
      if (volume_kerr_) {
        volume_forward_->set_work_area(transform_work_);
        volume_inverse_->set_work_area(transform_work_);
      }
    }
#endif
  }
  constexpr int axis = -1;
  if (held_plasma()) {
    for (auto& stream : held_streams_) {
      if (stream.projected.empty())
        continue;
      stream.forward = std::make_unique<ForwardRealPlan>(
          fft_space_for_rows(stream.projected), stream.projected, stream.spectrum,
          KokkosFFT::Direction::forward, axis);
      if (!stream.coarse.empty())
        stream.coarse_forward = std::make_unique<ForwardRealPlan>(
            fft_space_for_rows(stream.coarse), stream.coarse, stream.coarse_spectrum,
            KokkosFFT::Direction::forward, axis);
    }
    if (!held_pack_.empty() && forward_pack_.empty())
      held_pack_forward_ = std::make_unique<ForwardRealPlan>(
          fft_space_for_rows(held_pack_), held_pack_, held_pack_spectrum_,
          KokkosFFT::Direction::forward, axis);
  }
  if (problem_.plasma.envelope)
    for (auto& stream : envelope_streams_) {
      stream.forward = std::make_unique<ComplexPlan>(
          fft_space_for_rows(stream.projected), stream.projected, stream.spectrum,
          KokkosFFT::Direction::forward, axis);
      if (!stream.coarse.empty())
        stream.coarse_forward = std::make_unique<ComplexPlan>(
            fft_space_for_rows(stream.coarse), stream.coarse, stream.coarse_spectrum,
            KokkosFFT::Direction::forward, axis);
    }
  statistics_inverse_plan_ = std::make_unique<ComplexPlan>(
      fft_space_for_rows(statistics_time_), statistics_spectrum_, statistics_time_,
      KokkosFFT::Direction::backward, axis);
  if (problem_.dims.family == FieldFamily::field_real) {
    inverse_real_plan_ = std::make_unique<InverseRealPlan>(
        fft_space_for_rows(modal_real_), fine_spectrum_, modal_real_,
        KokkosFFT::Direction::backward, axis);
  } else {
    inverse_complex_plan_ = std::make_unique<ComplexPlan>(
        fft_space_for_rows(modal_complex_), fine_spectrum_, modal_complex_,
        KokkosFFT::Direction::backward, axis);
  }
  if (retained_synthesis_) {
    const auto plan_rows = [&](const std::size_t count) {
      const auto rows = std::make_pair(std::size_t{0}, count);
      const ComplexView2 padded = Kokkos::subview(modal_complex_, rows, Kokkos::ALL());
      const ComplexView2 node = Kokkos::subview(node_complex_, rows, Kokkos::ALL());
      const auto& space = fft_space_for_rows(node);
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
      seed_row_fft_wisdom(space, padded, node, KokkosFFT::Direction::backward);
#endif
      auto plan = std::make_unique<ComplexPlan>(space, padded, node,
                                                KokkosFFT::Direction::backward, axis);
      // The padded rows' unretained bins stay zero from here on
      // (`synthesise_retained`), whatever planning wrote.
      graph::deep_copy(execution_space_, padded, Complex{});
      graph::deep_copy(execution_space_, node, Complex{});
      return plan;
    };
    retained_inverse_plan_ = plan_rows(problem_.dims.node_chunk);
    if (const auto tail = problem_.dims.n_nodes % problem_.dims.node_chunk; tail != 0)
      retained_tail_inverse_plan_ = plan_rows(tail);
  }
  if (problem_.dims.family == FieldFamily::envelope) {
    forward_complex_plan_ = std::make_unique<ComplexPlan>(
        fft_space_for_rows(projected_complex_), projected_complex_, projected_spectrum_,
        KokkosFFT::Direction::forward, axis);
    absorber_complex_plan_ = std::make_unique<ComplexPlan>(
        fft_space_for_rows(absorber_complex_), absorber_complex_, absorber_spectrum_,
        KokkosFFT::Direction::forward, axis);
    if (!problem_.transform.projection_coarse_matrix.empty())
      forward_coarse_complex_plan_ = std::make_unique<ComplexPlan>(
          fft_space_for_rows(projected_coarse_complex_), projected_coarse_complex_,
          projected_coarse_spectrum_, KokkosFFT::Direction::forward, axis);
  } else {
    forward_real_plan_ = std::make_unique<ForwardRealPlan>(
        fft_space_for_rows(projected_real_), projected_real_, projected_spectrum_,
        KokkosFFT::Direction::forward, axis);
    absorber_real_plan_ = std::make_unique<ForwardRealPlan>(
        fft_space_for_rows(absorber_real_), absorber_real_, absorber_spectrum_,
        KokkosFFT::Direction::forward, axis);
    if (!forward_pack_.empty())
      forward_pack_plan_ = std::make_unique<ForwardRealPlan>(
          fft_space_for_rows(forward_pack_), forward_pack_, forward_pack_spectrum_,
          KokkosFFT::Direction::forward, axis);
    if (!forward_pack_tail_.empty())
      forward_pack_tail_plan_ = std::make_unique<ForwardRealPlan>(
          fft_space_for_rows(forward_pack_tail_), forward_pack_tail_,
          forward_pack_tail_spectrum_, KokkosFFT::Direction::forward, axis);
    if (!forward_pack_fine_.empty())
      forward_pack_fine_plan_ = std::make_unique<ForwardRealPlan>(
          fft_space_for_rows(forward_pack_fine_), forward_pack_fine_,
          forward_pack_fine_spectrum_, KokkosFFT::Direction::forward, axis);
    if (!forward_pack_fine_tail_.empty())
      forward_pack_fine_tail_plan_ = std::make_unique<ForwardRealPlan>(
          fft_space_for_rows(forward_pack_fine_tail_), forward_pack_fine_tail_,
          forward_pack_fine_tail_spectrum_, KokkosFFT::Direction::forward, axis);
    if (!forward_pack_coarse_.empty())
      forward_pack_coarse_plan_ = std::make_unique<ForwardRealPlan>(
          fft_space_for_rows(forward_pack_coarse_), forward_pack_coarse_,
          forward_pack_coarse_spectrum_, KokkosFFT::Direction::forward, axis);
    if (!problem_.transform.projection_coarse_matrix.empty())
      forward_coarse_real_plan_ = std::make_unique<ForwardRealPlan>(
          fft_space_for_rows(projected_coarse_real_), projected_coarse_real_,
          projected_coarse_spectrum_, KokkosFFT::Direction::forward, axis);
  }
  if constexpr (cpu_size_policy_enabled) {
    if (problem_.separable_geometry()) {
      const auto full_units = transverse_units_per_chunk();
      const auto tail_units = transverse_units() % full_units;
      const auto rows_per_unit =
          problem_.slab_geometry() ? problem_.spatial.n_p : problem_.dims.n_pol_state;
      temporal_tail_rows_ = tail_units * rows_per_unit;
      if (temporal_tail_rows_ != 0) {
        const auto& space =
            fft_space_for(temporal_tail_rows_ * problem_.dims.n_to, problem_.dims.n_to);
        const auto rows = fine_spectrum_.extent(0);
        if (problem_.dims.family == FieldFamily::field_real)
          temporal_tail_inverse_real_.create(space, fine_spectrum_, modal_real_, rows,
                                             temporal_tail_rows_,
                                             KokkosFFT::Direction::backward);
        else
          temporal_tail_inverse_complex_.create(space, fine_spectrum_, modal_complex_,
                                                rows, temporal_tail_rows_,
                                                KokkosFFT::Direction::backward);
        if (problem_.dims.family == FieldFamily::envelope)
          temporal_tail_forward_complex_.create(
              space, projected_complex_, projected_spectrum_, rows, temporal_tail_rows_,
              KokkosFFT::Direction::forward);
        else
          temporal_tail_forward_real_.create(
              space, projected_real_, projected_spectrum_, rows, temporal_tail_rows_,
              KokkosFFT::Direction::forward);
        // The tail plans' measured planning overwrote their rows.
        graph::deep_copy(execution_space_, fine_spectrum_, Complex{});
        graph::deep_copy(execution_space_, projected_spectrum_, Complex{});
        if (problem_.dims.family == FieldFamily::envelope)
          graph::deep_copy(execution_space_, projected_complex_, Complex{});
        else
          graph::deep_copy(execution_space_, projected_real_, 0.0);
        if (problem_.dims.family == FieldFamily::field_real)
          graph::deep_copy(execution_space_, modal_real_, 0.0);
        else
          graph::deep_copy(execution_space_, modal_complex_, Complex{});
      }
    }
  }
  if (problem_.raman.enabled) {
    const auto started = std::chrono::steady_clock::now();
    raman_drive_forward_plan_ = std::make_unique<RamanForwardPlan>(
        raman_execution_space_, node_raman_drive_, node_raman_spectrum_,
        KokkosFFT::Direction::forward, axis);
    raman_convolution_inverse_plan_ = std::make_unique<RamanInversePlan>(
        raman_execution_space_, node_raman_spectrum_, node_raman_convolved_,
        KokkosFFT::Direction::backward, axis);
    const auto tail = problem_.dims.n_nodes % problem_.dims.node_chunk;
    if (tail != 0) {
      // Reuse the leading rows of the full chunk buffers. A separate persistent
      // plan avoids transforming stale rows in the final partial node chunk.
      const auto rows = std::make_pair(std::size_t{0}, tail * problem_.raman_drives());
      raman_tail_drive_ = Kokkos::subview(node_raman_drive_, rows, Kokkos::ALL());
      raman_tail_spectrum_ = Kokkos::subview(node_raman_spectrum_, rows, Kokkos::ALL());
      raman_tail_convolved_ =
          Kokkos::subview(node_raman_convolved_, rows, Kokkos::ALL());
      raman_tail_forward_plan_ = std::make_unique<RamanForwardPlan>(
          raman_tail_execution_space_, raman_tail_drive_, raman_tail_spectrum_,
          KokkosFFT::Direction::forward, axis);
      raman_tail_inverse_plan_ = std::make_unique<RamanInversePlan>(
          raman_tail_execution_space_, raman_tail_spectrum_, raman_tail_convolved_,
          KokkosFFT::Direction::backward, axis);
    }
    if (raman_products_ != 0) {
      const auto rows = std::make_pair(std::size_t{0}, raman_products_);
      raman_product_drive_ = Kokkos::subview(node_raman_drive_, rows, Kokkos::ALL());
      raman_product_spectrum_ =
          Kokkos::subview(node_raman_spectrum_, rows, Kokkos::ALL());
      raman_product_forward_plan_ = std::make_unique<RamanForwardPlan>(
          raman_product_execution_space_, raman_product_drive_, raman_product_spectrum_,
          KokkosFFT::Direction::forward, axis);
      raman_product_inverse_plan_ = std::make_unique<RamanInversePlan>(
          raman_product_execution_space_, raman_product_spectrum_, raman_product_drive_,
          KokkosFFT::Direction::backward, axis);
    }
    raman_kernel_forward_plan_ = std::make_unique<RamanForwardPlan>(
        raman_kernel_execution_space_, raman_kernel_time_, raman_kernel_spectrum_,
        KokkosFFT::Direction::forward, axis);
    raman_planning_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
  }
  if (!problem_.absorber.split_step)
    return;
  if (split_rows_ != 0) {
    // In place on the state's rows; a later execution may name another row block or the
    // other of the state's two views, of the same extents.
    const auto& state = problem_.stepper.low_storage ? current_ : advanced_;
    const auto plan_rows = [&](const std::size_t count,
                               const KokkosFFT::Direction direction) {
      const ComplexView2 rows =
          Kokkos::subview(state, std::make_pair(std::size_t{0}, count), Kokkos::ALL());
      const auto& space = fft_space_for_rows(rows);
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
      // Seeding may overwrite the rows; nothing is loaded yet.
      seed_row_fft_wisdom(space, rows, rows, direction);
      graph::deep_copy(execution_space_, rows, Complex{});
#endif
      return std::make_unique<ComplexPlan>(space, rows, rows, direction, axis);
    };
    split_inverse_plan_ = plan_rows(split_rows_, KokkosFFT::Direction::backward);
    split_forward_plan_ = plan_rows(split_rows_, KokkosFFT::Direction::forward);
    if (const auto tail = problem_.n_columns() % split_rows_; tail != 0) {
      split_tail_inverse_plan_ = plan_rows(tail, KokkosFFT::Direction::backward);
      split_tail_forward_plan_ = plan_rows(tail, KokkosFFT::Direction::forward);
    }
    return;
  }
  if (problem_.dims.family == FieldFamily::envelope) {
    coarse_inverse_complex_plan_ = std::make_unique<ComplexPlan>(
        fft_space_for_rows(coarse_time_complex_), coarse_spectrum_,
        coarse_time_complex_, KokkosFFT::Direction::backward, axis);
    coarse_forward_complex_plan_ = std::make_unique<ComplexPlan>(
        fft_space_for_rows(coarse_time_complex_), coarse_time_complex_,
        coarse_spectrum_, KokkosFFT::Direction::forward, axis);
  } else {
    coarse_inverse_real_plan_ = std::make_unique<InverseRealPlan>(
        fft_space_for_rows(coarse_time_real_), coarse_spectrum_, coarse_time_real_,
        KokkosFFT::Direction::backward, axis);
    coarse_forward_real_plan_ = std::make_unique<ForwardRealPlan>(
        fft_space_for_rows(coarse_time_real_), coarse_time_real_, coarse_spectrum_,
        KokkosFFT::Direction::forward, axis);
  }
  if (temporal_tail_rows_ != 0) {
    const auto& space =
        fft_space_for(temporal_tail_rows_ * problem_.dims.n_t, problem_.dims.n_t);
    const auto rows = coarse_spectrum_.extent(0);
    if (problem_.dims.family == FieldFamily::envelope) {
      coarse_tail_inverse_complex_.create(space, coarse_spectrum_, coarse_time_complex_,
                                          rows, temporal_tail_rows_,
                                          KokkosFFT::Direction::backward);
      coarse_tail_forward_complex_.create(space, coarse_time_complex_, coarse_spectrum_,
                                          rows, temporal_tail_rows_,
                                          KokkosFFT::Direction::forward);
    } else {
      coarse_tail_inverse_real_.create(space, coarse_spectrum_, coarse_time_real_, rows,
                                       temporal_tail_rows_,
                                       KokkosFFT::Direction::backward);
      coarse_tail_forward_real_.create(space, coarse_time_real_, coarse_spectrum_, rows,
                                       temporal_tail_rows_,
                                       KokkosFFT::Direction::forward);
    }
    // The tail plans' measured planning overwrote their rows.
    graph::deep_copy(execution_space_, coarse_spectrum_, Complex{});
    if (problem_.dims.family == FieldFamily::envelope)
      graph::deep_copy(execution_space_, coarse_time_complex_, Complex{});
    else
      graph::deep_copy(execution_space_, coarse_time_real_, 0.0);
  }
}

std::map<std::string, std::size_t> GeneralWorkspace::memory_ledger() const {
  auto ledger =
      general_ ? general_->memory_ledger() : GeneralModalSource::empty_ledger();
  ledger.merge(flow_ ? flow_->memory_ledger() : CoupledFlowAction::empty_ledger());
  ledger.merge(dispersive_             ? dispersive_->memory_ledger()
               : separable_dispersive_ ? separable_dispersive_->memory_ledger()
               : general_ && general_->dispersive() != nullptr
                   ? general_->dispersive()->memory_ledger()
                   : DispersiveResponse::empty_ledger());
  ledger.merge(base_memory_ledger());
  return ledger;
}

std::map<std::string, std::size_t> GeneralWorkspace::base_memory_ledger() const {
  std::size_t edge_plan_heap = 0;
  std::size_t edge_plan_device = 0;
  for (const auto& [shape, batch] : edge_batches_) {
    edge_plan_heap += batch.heap_growth;
    edge_plan_device += batch.device_growth;
  }
  const auto complex_bytes = sizeof(Complex);
  const auto real_bytes = sizeof(double);
  const auto byte_bytes = sizeof(unsigned char);
  const auto index_bytes = sizeof(std::size_t);
  const auto field = problem_.n_columns() * problem_.dims.n_omega * complex_bytes;
  const auto modal =
      modal_real_.size() * real_bytes + modal_complex_.size() * complex_bytes;
  // Aliased views (free space, SEP 28) own no storage of their own.
  const auto node = separable_aliased_ ? 0
                                       : node_real_.size() * real_bytes +
                                             node_complex_.size() * complex_bytes;
  const auto polarisation = separable_aliased_
                                ? 0
                                : polarisation_real_.size() * real_bytes +
                                      polarisation_complex_.size() * complex_bytes;
  const auto projected =
      projected_real_.size() * real_bytes + projected_complex_.size() * complex_bytes;
  return {
      {"held_current_host_derivative",
       problem_.grid.time_window_derivative.capacity() * real_bytes},
      {"held_current", held_current_.size() * real_bytes},
      {"gated_current_prefix",
       (gated_prefix_.size() + gated_filtered_.size()) * real_bytes},
      {"held_current_gj", held_streams_[0].bytes()},
      {"held_current_edge", held_streams_[1].bytes()},
      {"held_current_nonplasma", held_streams_[2].bytes()},
      {"held_current_projection", held_projection_.size() * real_bytes},
      {"held_current_radial_spectra",
       (held_nodal_spectrum_.size() + held_projected_spectrum_.size()) * complex_bytes},
      {"held_current_tables",
       (held_derivative_.size() + held_inverse_.size()) * real_bytes},
      {"envelope_drude_stream", envelope_streams_[0].bytes()},
      {"envelope_loss_stream", envelope_streams_[1].bytes()},
      {"envelope_cycles",
       (envelope_rule_nodes_.size() + envelope_rule_weights_.size() +
        envelope_moments_.size() + envelope_rates_.size() + envelope_exposure_.size()) *
           real_bytes},
      {"envelope_factors", envelope_drude_factor_.size() * real_bytes +
                               envelope_loss_factor_.size() * complex_bytes},
      {"envelope_status", (envelope_failures_.size() + envelope_status_.size() +
                           envelope_host_status_.size()) *
                              index_bytes},
      {"radial_blas_buffers",
       (radial_blas_input_.size() + radial_blas_output_.size()) * real_bytes},
      {"state", field},
      {"advanced", advanced_.size() * complex_bytes},
      {"stages", stages_.size() * complex_bytes},
      {"accumulator_solution", solution_.size() * complex_bytes},
      {"accumulator_error", error_.size() * complex_bytes},
      {"weighted_residual", residual_.size() * complex_bytes},
      {"recycled_stage", recycled_.size() * complex_bytes},
      // A native session's Runge-Kutta dense output (solprop SEP 47 section 5.3).
      {"rk_dense",
       (rk_dense_coefficients_.size() + rk_dense_held_.size()) * complex_bytes +
           rk_dense_stiff_.size() * byte_bytes},
      {"noise", problem_.stepper.modified_noise ? field : 0},
      // The Runge-Kutta march's whole-step propagator, or the low-storage march's
      // table or step pair (solprop's `propagators` row).
      {"propagators",
       (whole_propagator_.size() + low_storage_flows_.size()) * complex_bytes},
      {"register_accumulator", register_accumulator_.size() * complex_bytes},
      {"register_stage", register_stage_.size() * complex_bytes},
      {"stage_phase", stage_phase_.size() * complex_bytes},
      {"stage_inverse", stage_inverse_.size() * complex_bytes},
      {"fine_spectrum", fine_spectrum_.size() * complex_bytes},
      {"retained_synthesis",
       (retained_modal_.size() + retained_node_.size()) * complex_bytes},
      {"fine_gaps", (fine_gaps_.size() + separable_gaps_.size()) * index_bytes},
      {"modal_time", modal},
      {"node_field", node},
      {"node_polarisation", polarisation},
      {"projected", projected},
      {"projected_coarse",
       problem_.transform.projection_coarse_matrix.empty() ? 0 : projected},
      {"projected_spectrum", projected_spectrum_.size() * complex_bytes},
      {"projected_coarse_spectrum", projected_coarse_spectrum_.size() * complex_bytes},
      {"absorber_spectrum", absorber_spectrum_.size() * complex_bytes},
      {"absorber_time",
       absorber_complex_.size() * complex_bytes + absorber_real_.size() * real_bytes},
      {"split_absorber_spectrum", coarse_spectrum_.size() * complex_bytes},
      {"split_absorber_time", coarse_time_complex_.size() * complex_bytes +
                                  coarse_time_real_.size() * real_bytes},
      {"transform_matrices",
       (synthesis_.size() + projection_.size() + projection_coarse_.size()) *
           real_bytes},
      // The native problem owns host copies as well as the execution-space views.
      // Count retained vector capacity, including construction-only coefficients.
      {"host_problem_arrays",
       (problem_.plasma.envelope_loss_factor.capacity() +
        problem_.grid.carrier.capacity() + problem_.general.synthesis.capacity() +
        problem_.general.projection.capacity() +
        problem_.general.frequency_factor.capacity() +
        problem_.general.delayed_spectrum.capacity() +
        problem_.linear.phi_terms.capacity() +
        problem_.linear.prefactor_terms.capacity() +
        problem_.spatial.on_axis.capacity() + problem_.spatial.k_squared.capacity() +
        problem_.spatial.k_squared_low.capacity() +
        problem_.spatial.numerator.capacity() + problem_.spatial.clamped.capacity() +
        problem_.spatial.kappa_sq.capacity()) *
               complex_bytes +
           (problem_.plasma.envelope_rule_nodes.capacity() +
            problem_.plasma.envelope_rule_weights.capacity() +
            problem_.plasma.envelope_drude_factor.capacity() +
            problem_.grid.omega_window.capacity() + problem_.grid.omega.capacity() +
            problem_.grid.power_weight.capacity() +
            problem_.grid.time_window.capacity() + problem_.absorber.fine.capacity() +
            problem_.absorber.coarse.capacity() +
            problem_.spatial.absorber_matrix.capacity() +
            problem_.spatial.spectral_weights.capacity() +
            problem_.spatial.kperp_sq.capacity() +
            problem_.spatial.kperp_sq_low.capacity() +
            problem_.spatial.frame.capacity() +
            problem_.spatial.spectral_rate.capacity() +
            problem_.spatial.user_rate.capacity() +
            problem_.spatial.absorber_rate.capacity() +
            problem_.spatial.kperp_rate.capacity() + problem_.spatial.k_x.capacity() +
            problem_.spatial.k_y.capacity() +
            problem_.transform.synthesis_matrix.capacity() +
            problem_.transform.projection_matrix.capacity() +
            problem_.transform.projection_coarse_matrix.capacity() +
            problem_.stepper.tableau_matrix.capacity() +
            problem_.stepper.tableau_nodes.capacity() +
            problem_.stepper.solution_weights.capacity() +
            problem_.stepper.error_weights.capacity() +
            problem_.raman.undamped.capacity() +
            problem_.raman.tensor_weights.capacity() +
            problem_.plasma.rate_coefficients.capacity() +
            problem_.plasma.channel_parameters.capacity() +
            problem_.plasma.channel_coefficients.capacity() +
            problem_.raman.row_component.capacity() +
            problem_.raman.component_fraction.capacity() +
            problem_.general.tensor.capacity() +
            problem_.general.delayed_tensor.capacity()) *
               real_bytes +
           problem_.general.node_response.capacity() * index_bytes +
           problem_.grid.crop_index.capacity() * index_bytes +
           (problem_.grid.active_mask.capacity() + problem_.grid.norm_mask.capacity()) *
               byte_bytes},
      {"scaled_transform_matrices",
       (scaled_synthesis_.size() + scaled_projection_.size() +
        scaled_projection_coarse_.size() + scaled_projection_stacked_.size()) *
               real_bytes +
           (scaled_synthesis_complex_.size() + scaled_projection_complex_.size() +
            scaled_projection_coarse_complex_.size() +
            scaled_projection_stacked_complex_.size()) *
               complex_bytes},
      {"stacked_projection", projected_stacked_real_.size() * real_bytes +
                                 projected_stacked_complex_.size() * complex_bytes},
      {"chunk_transform_buffers",
       (chunk_synthesis_.size() + tail_synthesis_.size() + chunk_projection_.size() +
        tail_projection_.size() + chunk_projection_stacked_.size() +
        tail_projection_stacked_.size()) *
               real_bytes +
           (chunk_synthesis_complex_.size() + tail_synthesis_complex_.size() +
            chunk_projection_complex_.size() + tail_projection_complex_.size() +
            chunk_projection_stacked_complex_.size() +
            tail_projection_stacked_complex_.size()) *
               complex_bytes},
      {"transverse_scratch", transverse_.size() * complex_bytes},
      {"spatial_coefficients",
       (spatial_.square.size() + spatial_.square_low.size() +
        spatial_.numerator.size() + spatial_.clamped.size() + spatial_.kappa.size()) *
               complex_bytes +
           (spatial_.transverse.size() + spatial_.transverse_low.size() +
            spatial_.frame.size() + spatial_.spectral_rate.size() +
            spatial_.user_rate.size() + spatial_.absorber.size() +
            spatial_.transverse_rate.size()) *
               real_bytes},
      {"adams_history",
       (adams_store_.size() + adams_slopes_.size() + adams_moments_.size() +
        adams_source_.size() + adams_propagator_.size() + adams_exponent_.size() +
        adams_saved_.size() + adams_held_.size() + adams_dense_start_.size() +
        adams_dense_end_.size() + adams_dense_exponent_.size() + adams_states_.size() +
        adams_rates_.size() + adams_carried_.size() + adams_dense_terms_.size() +
        adams_lab_.size() + adams_endpoint_.size() + adams_endpoint_rate_.size() +
        adams_block_.size() + adams_channels_.size() + adams_fallback_.size() +
        adams_applied_.size() + adams_end_applied_.size()) *
               complex_bytes +
           (adams_divisor_.size() + adams_rate_weights_.size() +
            adams_end_weights_.size() + adams_scale_weights_.size() +
            (adams_magnitude_sum_.data() != nullptr ? 1 : 0)) *
               real_bytes},
      {"fixed_phi_terms", phi_terms_.size() * complex_bytes},
      {"fixed_prefactor_terms", prefactor_terms_.size() * complex_bytes},
      {"z_tables", phi_table_.bytes() + prefactor_table_.bytes() + rate_table_.bytes()},
      {"statistics_metric", statistics_whitened_.size() * complex_bytes +
                                statistics_root_.size() * complex_bytes +
                                statistics_whitened_sums_.size() * real_bytes},
      // solprop SEP 34: a radial run's plane-measure squares and its material table
      // (a separable run's alias its operator's), and the magnetic closure's
      // transverse volume of a separable run's local flux statistics.
      {"flux_coefficients",
       problem_.separable_geometry()
           ? 0
           : (flux_.square.size() + flux_.square_low.size()) * complex_bytes +
                 (flux_.transverse.size() + flux_.transverse_low.size()) * real_bytes},
      {"flux_table", flux_table_.bytes()},
      {"flux_closure_transverse", flux_transverse_.size() * complex_bytes},
      {"tableau", tableau_.size() * real_bytes},
      // The per-stage scalars are held twice, in the packed buffer's two slots, and
      // a device's pinned host ring holds copies of the whole buffer.
      {"stage_scalars",
       2 *
           (stage_phi_delta_.size() + stage_phi_at_.size() + stage_prefactor_.size() +
            stage_density_.size() + stage_taper_.size() + stage_preionisation_.size() +
            stage_rate_weights_.size()) *
           real_bytes},
      {"stage_scalar_staging",
       stage_staging_.empty() || stage_staging_.front().data() == stage_scalars_.data()
           ? 0
           : stage_staging_.size() * stage_scalars_.size() * real_bytes},
      // An Adams trial's norms, read back in pinned host memory on a device, and a
      // Runge-Kutta trial's mirror of its partials (the view itself on a host build).
      {"trial_norms",
       (adams_norms_.size() +
        (pinned_partials_.data() == partials_.data() ? 0 : pinned_partials_.size())) *
           real_bytes},
      {"crop_index", crop_index_.size() * index_bytes},
      {"masks", (active_mask_.size() + norm_mask_.size()) * byte_bytes},
      {"windows", (omega_window_.size() + time_window_.size()) * real_bytes},
      {"spatial_statistics_geometry", spatial_spectral_weights_.size() * real_bytes +
                                          spatial_on_axis_.size() * complex_bytes},
      {"spatial_statistics_partials", spatial_statistics_partials_.size() * real_bytes},
      {"far_field_partials", far_field_ ? far_field_->partial_bytes() : 0},
      {"far_field_tables", far_field_ ? far_field_->table_bytes() : 0},
      {"statistics_frequency_axis", omega_.size() * real_bytes},
      {"carrier", carrier_.size() * complex_bytes},
      {"node_raman_drive", node_raman_drive_.size() * real_bytes},
      {"node_raman_spectrum", node_raman_spectrum_.size() * complex_bytes},
      {"node_raman_convolved", node_raman_convolved_.size() * real_bytes},
      {"raman_undamped", raman_undamped_.size() * real_bytes},
      {"raman_tensor_weights", raman_tensor_weights_.size() * real_bytes},
      {"raman_kernel_time", raman_kernel_time_.size() * real_bytes},
      {"raman_kernel_spectrum", raman_kernel_spectrum_.size() * complex_bytes},
      {"stage_raman_rates", 2 * stage_raman_rates_.size() * real_bytes},
      {"absorber_rates",
       (absorber_fine_.size() + absorber_coarse_.size() + split_factor_.size()) *
           real_bytes},
      {"spatial_absorber_matrix", spatial_absorber_real_.size() * real_bytes +
                                      spatial_absorber_complex_.size() * complex_bytes},
      {"plasma_rate_table", rate_coefficients_.size() * real_bytes},
      {"channel_parameters", channel_parameters_.size() * real_bytes},
      {"channel_rate_tables", channel_coefficients_.size() * real_bytes},
      {"channel_scratch", channel_scratch_.size() * real_bytes},
      {"material_stage_values",
       2 *
           (stage_kerr_density_.size() + stage_plasma_density_.size() +
            stage_raman_density_.size() + stage_channel_density_.size() +
            stage_channel_preionisation_.size()) *
           real_bytes},
      {"raman_component_values",
       (2 * stage_raman_densities_.size() + raman_row_component_.size() +
        raman_component_fraction_.size() + cached_raman_densities_.capacity()) *
           real_bytes},
      {"plasma_stash", (stash_candidate_.size() + stash_current_.size()) * real_bytes},
      {"plasma_statistics_scratch", plasma_statistics_scratch_.size() * real_bytes},
      {"plasma_integration_diagnostics",
       (plasma_taps_.size() + plasma_diagnostics_.size()) * real_bytes},
      {"plasma_device_scan",
       (plasma_rate_.size() + plasma_exposure_.size() + plasma_fraction_.size() +
        plasma_current_integrand_.size() + plasma_free_current_.size() +
        plasma_outer_current_.size() + band_tile_totals_.size() +
        plasma_tile_totals_.size()) *
           real_bytes},
      {"error_partials", 3 * problem_.n_columns() * reduction_slots_ * real_bytes},
      {"quadrature_partials", problem_.transform.projection_coarse_matrix.empty()
                                  ? 0
                                  : 2 * problem_.n_columns() * real_bytes},
      {"weighted_control",
       (weighted_scale_.size() + weighted_partials_.size()) * real_bytes},
      {"statistics_spectrum", statistics_spectrum_.size() * complex_bytes},
      {"statistics_time", statistics_time_.size() * complex_bytes},
      {"statistics_weighted_time", statistics_weighted_time_.size() * complex_bytes},
      {"statistics_power_weight", statistics_power_weight_.size() * sizeof(double)},
      {"statistics_power", statistics_power_.size() * real_bytes},
      // The boundary monitor's weights and accumulators, once a monitor is bound.
      {"monitor_weights",
       (monitor_weights_.size() + monitor_column_weights_.size() +
        monitor_share_.size() + monitor_half_.size() + monitor_removed_.size()) *
               real_bytes +
           occupancy_buckets_.size() +
           (monitor_split_sum_.size() + monitor_found_.size()) * sizeof(SplitSums)},
      {"statistics_flux_magnitude", statistics_magnitude_.size() * real_bytes},
      {"host_crossing_ring", crossing_ring_.bytes()},
      {"transverse_edge", edge_bytes()},
      {"statistics_outputs", (statistics_total_power_.size() +
                              statistics_on_axis_.size() + statistics_packed_.size()) *
                                     real_bytes +
                                 statistics_axis_row_.size() * complex_bytes},
      {"fft_plan_heap_growth", plan_heap_growth_ + edge_plan_heap},
      {"fft_plan_device_growth", plan_device_growth_ + edge_plan_device},
  };
}

} // namespace solprop_native
