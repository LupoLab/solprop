// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The general workspace's hot set: the trial step, its stages and responses, the
// Adams and low-storage marches and the split-step absorber, in one unit so that the
// compiler sees every call between them (a device build has no link-time
// optimisation). The unit is this file and the fragments it includes, in this order,
// each a contiguous run of definitions that is never compiled alone:
//
//   trial.cpp              the transform providers' policy, the trial's scalars and
//                          entry (set_initial, prepare_trial, trial_step, adopt)
//   trial_low_storage.inc  the low-storage RK4IP march (SEP 24)
//   trial_adams.inc        the Adams history and trials, and the RK dense output
//   trial_separable.inc    the separable geometries' stage path
//   trial_stage.inc        the stage's orchestration and the synthesis
//   trial_response.inc     response dispatch, band plasma, Raman, envelope plasma
//   trial_finish.inc       transforms, projection, absorber, finishes, quadrature, mask
//   trial_split_step.inc   the split-step absorber

#include "general_workspace.hpp"

namespace solprop_native {

// The hybrid policy's shape guard. A host build sends every eligible transform to the
// vendor GEMM: measured on Apple's Accelerate with one thread, it beat the custom
// kernel at every shape tried -- a radial field's transforms 1.1 to 1.5 times faster
// at 2 to 6 nodes and 1.6 to 35 times at 16 to 256 nodes and 256 to 2048 samples, a
// one-mode field capillary's 2.4 to 3.1 times -- and lost at none. A device build
// takes the vendor GEMM from eight modes (with eight nodes and 1024 samples): below
// that the product is a few rows deep, which the custom kernels take in one pass over
// the samples. On an NVIDIA A40 a four-mode capillary's stacked projection (8 rows,
// 36 nodes, 10240 samples) took 71 us in cuBLAS and its synthesis 27 us; at 256
// radial modes the vendor GEMM runs near the FP64 peak.
bool GeneralWorkspace::hybrid_shape_eligible() const noexcept {
  return build::host || (problem_.dims.n_modes >= 8 && problem_.dims.n_nodes >= 8 &&
                         problem_.dims.n_to >= 1024);
}

bool GeneralWorkspace::use_packed_radial_blas() const noexcept {
  if constexpr (build::device)
    return false;
  // Two packed work arrays add two complex-equivalent time columns to the
  // shared conservative twenty-column inventory. Decide before allocation.
  constexpr std::size_t max_samples =
      std::size_t{64} * 1024 * 1024 / (22 * sizeof(Complex));
  return !problem_.separable_geometry() && !problem_.spatial.spectral_weights.empty() &&
         problem_.dims.n_modes == problem_.dims.n_nodes &&
         problem_.dims.node_chunk == problem_.dims.n_nodes &&
         problem_.dims.n_pol_state == 1 && problem_.dims.n_pol_transform == 1 &&
         problem_.dims.family == FieldFamily::envelope &&
         problem_.transform.projection_coarse_matrix.empty() &&
         problem_.dims.n_nodes >= 256 && problem_.dims.n_to >= 256 &&
         problem_.dims.n_nodes <= max_samples / problem_.dims.n_to;
}

// The retained-spectrum radial synthesis, on a host and a device. A real static
// synthesis commutes with the temporal transform: synthesising the coarse bins the
// stage entry fills and padding each node's spectrum gives the nodes' fields, so the
// product covers n_omega complex columns rather than the n_to complex samples of the
// padded, oversampled analytic signal. On a square map the node transforms replace the
// modal ones one for one; the held radial plasma's projection already contracts after
// cropping (`held_radial_projection`). An in-stage absorber or a Raman drive reads the
// modal time field, and a dispersive response its own synthesis, so they keep the
// time-domain product; skinny capillary maps would multiply the transforms.
bool GeneralWorkspace::retained_synthesis() const noexcept {
  return problem_.dims.family == FieldFamily::field_analytic &&
         !problem_.spatial.spectral_weights.empty() && !problem_.separable_geometry() &&
         !problem_.general.enabled && !problem_.dispersive.enabled &&
         !problem_.absorber.in_stage && !problem_.raman.enabled &&
         problem_.dims.n_modes == problem_.dims.n_nodes && problem_.dims.n_nodes > 1 &&
         problem_.dims.n_pol_state == 1 && problem_.dims.n_pol_transform == 1 &&
         problem_.dims.n_omega < problem_.dims.n_to;
}

void GeneralWorkspace::synthesise_retained(const std::size_t first_node,
                                           const std::size_t node_count) {
  Kokkos::Profiling::pushRegion("solprop_native.transform.synthesis.retained");
  const auto crop = crop_index_;
  const auto n = problem_.dims.n_omega;
  // The coarse modal spectra once per evaluation, at its first chunk (the stage core
  // and the statistics' synthesis both start at node zero). Rows run in parallel at
  // the modal copies' width (`temporal_execution_space_`), each a contiguous pass.
  if (first_node == 0) {
    const auto fine = fine_spectrum_;
    const auto modal = retained_modal_;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    // A device takes a thread per element.
    launch::parallel_for(
        "solprop_native.transform.retained_gather",
        Range2(execution_space_, {0, 0}, {problem_.dims.n_modes, n}),
        KOKKOS_LAMBDA(const std::size_t mode, const std::size_t index) {
          modal(mode, index) = fine(mode, crop(index));
        });
#else
    launch::parallel_for(
        "solprop_native.transform.retained_gather",
        Kokkos::RangePolicy<ExecutionSpace>(temporal_execution_space_, 0,
                                            problem_.dims.n_modes),
        KOKKOS_LAMBDA(const std::size_t mode) {
          for (std::size_t index = 0; index < n; ++index)
            modal(mode, index) = fine(mode, crop(index));
        });
#endif
  }
  using Interleaved = Kokkos::View<double**, Layout, MemorySpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  static_assert(sizeof(Complex) == 2 * sizeof(double));
  const Interleaved synthesis(scaled_synthesis_.data() +
                                  first_node * problem_.dims.n_modes,
                              node_count, problem_.dims.n_modes);
  const Interleaved input(reinterpret_cast<double*>(retained_modal_.data()),
                          problem_.dims.n_modes, 2 * n);
  const Interleaved output(reinterpret_cast<double*>(retained_node_.data()), node_count,
                           2 * n);
  engine_gemm(transform_space_for(node_count), synthesis, input, output, 0.0);
  // Each node's padded spectrum in the modal time buffer's leading rows, which are
  // free as no modal transform runs on this path. Only the retained bins are
  // written: the others were zeroed when the plans were made, and an out-of-place
  // complex transform leaves its input as it was.
  const auto node = retained_node_;
  const auto padded = modal_complex_;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // A device zeroes the chunk's padded rows first, a fill at its bandwidth, rather
  // than relying on the modal buffer's other bins staying zero, and pads a thread per
  // element.
  graph::deep_copy(execution_space_,
                   Kokkos::subview(padded, std::make_pair(std::size_t{0}, node_count),
                                   Kokkos::ALL()),
                   Complex{});
  launch::parallel_for(
      "solprop_native.transform.retained_pad",
      Range2(execution_space_, {0, 0}, {node_count, n}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
        padded(row, crop(index)) = node(row, index);
      });
#else
  launch::parallel_for(
      "solprop_native.transform.retained_pad",
      Kokkos::RangePolicy<ExecutionSpace>(temporal_execution_space_, 0, node_count),
      KOKKOS_LAMBDA(const std::size_t row) {
        for (std::size_t index = 0; index < n; ++index)
          padded(row, crop(index)) = node(row, index);
      });
#endif
  const bool tail = node_count != problem_.dims.node_chunk;
  const auto rows = std::make_pair(std::size_t{0}, node_count);
  graph::execute(tail ? *retained_tail_inverse_plan_ : *retained_inverse_plan_,
                 Kokkos::subview(modal_complex_, rows, Kokkos::ALL()),
                 Kokkos::subview(node_complex_, rows, Kokkos::ALL()),
                 KokkosFFT::Normalization::none);
  Kokkos::Profiling::popRegion();
}

void GeneralWorkspace::packed_radial_gemm(const RealView2& matrix,
                                          const ComplexView2& source,
                                          const ComplexView2& destination) {
  const auto input = radial_blas_input_;
  const auto output = radial_blas_output_;
  const auto rows = problem_.dims.n_nodes;
  const auto times = problem_.dims.n_to;
  Kokkos::Profiling::pushRegion("solprop_native.radial.packed_real_gemm");
  launch::parallel_for(
      "solprop_native.radial.pack",
      Range2(temporal_execution_space_, {0, 0}, {rows, times}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
        input(row, 2 * time) = source(row, time).real();
        input(row, 2 * time + 1) = source(row, time).imag();
      });
  engine_gemm(transform_execution_space_, matrix, input, output, 0.0);
  launch::parallel_for(
      "solprop_native.radial.unpack",
      Range2(temporal_execution_space_, {0, 0}, {rows, times}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
        destination(row, time) =
            Complex(output(row, 2 * time), output(row, 2 * time + 1));
      });
  Kokkos::Profiling::popRegion();
}

bool GeneralWorkspace::use_blas_for_synthesis() const noexcept {
  if (use_packed_radial_blas())
    return false;
  // The hybrid planar kind contracts its small x rule per y row natively.
  if (problem_.separable_geometry())
    return false;
  if (problem_.dims.n_pol_state != 1)
    return false;
#if !defined(SOLPROP_NATIVE_TPL_BLAS_OPENBLAS) &&                                      \
    !defined(SOLPROP_NATIVE_TPL_BLAS_CUBLAS) &&                                        \
    !defined(SOLPROP_NATIVE_TPL_BLAS_ROCBLAS)
  // Without a vendor BLAS an envelope keeps the custom kernels.
  if (problem_.dims.family == FieldFamily::envelope)
    return false;
#endif
  if (problem_.dims.family != FieldFamily::envelope &&
      problem_.dims.family != FieldFamily::field_real)
    return false;
#if defined(SOLPROP_NATIVE_TPL_BLAS_CUBLAS) || defined(SOLPROP_NATIVE_TPL_BLAS_ROCBLAS)
  // Spatial envelope matrices can be large even with a short temporal grid.
  if (!problem_.spatial.spectral_weights.empty() &&
      problem_.dims.family == FieldFamily::envelope &&
      problem_.transform.projection_coarse_matrix.empty() &&
      problem_.dims.n_modes == problem_.dims.n_nodes && problem_.dims.n_nodes >= 256 &&
      problem_.dims.n_to >= 256)
    return true;
#endif
  return hybrid_shape_eligible();
}

// An analytic field's or an envelope's synthesis is a real matrix times complex
// samples: viewed as real numbers, its samples are a real matrix twice as wide, so the
// transform is one real GEMM through the selected BLAS, with no copy and no complex
// BLAS. An envelope's projection is the same (`use_interleaved_projection`).
//
// A device takes it at the shapes where it already takes BLAS for the projection
// (`hybrid_shape_eligible`): there the custom synthesis kernel cost 4.6 times the
// projection's GEMM at 512 radial nodes (an AMD CDNA 3 device, 2026-10-06).
bool GeneralWorkspace::use_interleaved_synthesis() const noexcept {
  if (problem_.dims.family == FieldFamily::field_real ||
      problem_.separable_geometry() || problem_.dims.n_pol_state != 1 ||
      problem_.dims.n_pol_transform != 1)
    return false;
  return hybrid_shape_eligible();
}

bool GeneralWorkspace::use_interleaved_projection() const noexcept {
  if (problem_.dims.family != FieldFamily::envelope || problem_.separable_geometry() ||
      problem_.dims.n_pol_state != 1 || problem_.dims.n_pol_transform != 1)
    return false;
  return hybrid_shape_eligible();
}

bool GeneralWorkspace::use_blas_for_projection() const noexcept {
  if (use_packed_radial_blas())
    return false;
  // The hybrid planar kind contracts its small x rule per y row natively.
  if (problem_.separable_geometry())
    return false;
  if (problem_.dims.n_pol_state != 1)
    return false;
#if !defined(SOLPROP_NATIVE_TPL_BLAS_OPENBLAS) &&                                      \
    !defined(SOLPROP_NATIVE_TPL_BLAS_CUBLAS) &&                                        \
    !defined(SOLPROP_NATIVE_TPL_BLAS_ROCBLAS)
  // Without a vendor BLAS an envelope keeps the custom kernels.
  if (problem_.dims.family == FieldFamily::envelope)
    return false;
#endif
#if defined(SOLPROP_NATIVE_TPL_BLAS_CUBLAS) || defined(SOLPROP_NATIVE_TPL_BLAS_ROCBLAS)
  // Spatial envelope matrices can be large even with a short temporal grid.
  if (!problem_.spatial.spectral_weights.empty() &&
      problem_.dims.family == FieldFamily::envelope &&
      problem_.transform.projection_coarse_matrix.empty() &&
      problem_.dims.n_modes == problem_.dims.n_nodes && problem_.dims.n_nodes >= 256 &&
      problem_.dims.n_to >= 256)
    return true;
#endif
  return hybrid_shape_eligible();
}

bool GeneralWorkspace::use_chunked_blas_for_synthesis() const noexcept {
  return problem_.dims.node_chunk < problem_.dims.n_nodes && use_blas_for_synthesis();
}

// One node, one mode and one scalar component: the synthesis and the projection are
// each one product, which the response column forms itself (solprop SEP 25 section
// 6.7), in place of the stage's taper scaling, two one-element GEMMs and the zero
// fill of the projection. Each folded product is the GEMM's own one rounding,
// `(s * taper) * x` and `(p / taper) * y`.
// The host's columns only: a device keeps its own column kernels.
bool GeneralWorkspace::folds_single_node() const noexcept {
  if constexpr (build::device)
    return false;
  return !general_ && !dispersive_ && !held_radial() &&
         !problem_.separable_geometry() && problem_.dims.n_modes == 1 &&
         problem_.dims.n_nodes == 1 && problem_.dims.node_chunk == 1 &&
         problem_.dims.n_pol_state == 1 && problem_.dims.n_pol_transform == 1 &&
         problem_.transform.projection_coarse_matrix.empty() &&
         !problem_.raman.enabled && !problem_.plasma.envelope &&
         radial_blas_input_.empty();
}

bool GeneralWorkspace::use_chunked_blas_for_projection() const noexcept {
  return problem_.dims.node_chunk < problem_.dims.n_nodes && use_blas_for_projection();
}

std::string GeneralWorkspace::synthesis_provider() const {
  const auto provider = selected_blas_provider_name();
  if (folds_single_node())
    return "folded";
  if (use_packed_radial_blas())
    return std::string(provider) + "-packed-real";
  if (problem_.free_geometry())
    return "kokkos-fft";
  if (retained_synthesis_)
    return std::string(provider) + "-retained-interleaved-real";
  if (use_interleaved_synthesis())
    return std::string(provider) + "-interleaved-real";
  return std::string(use_blas_for_synthesis() ? provider : std::string_view("custom"));
}

std::string GeneralWorkspace::projection_provider() const {
  const auto provider = selected_blas_provider_name();
  if (folds_single_node())
    return "folded";
  if (use_packed_radial_blas())
    return std::string(provider) + "-packed-real";
  if (problem_.free_geometry())
    return "kokkos-fft";
  if (use_interleaved_projection())
    return std::string(provider) + "-interleaved-real";
  return std::string(use_blas_for_projection() ? provider : std::string_view("custom"));
}

std::string_view GeneralWorkspace::gemm_shape_policy() {
  return "host radial scalar envelope uses packed real BLAS for whole transforms "
         "with n_nodes/n_to >= 256 and a 22-column inventory within 64 MiB; "
         "device radial scalar envelope uses vendor complex BLAS with "
         "n_nodes/n_to >= 256; otherwise hybrid uses KokkosBlas for every whole "
         "transform on the host, and on a device for those with n_modes >= 8, "
         "n_nodes >= 8 and n_to >= 1024: real ones directly, an analytic field's "
         "synthesis as one real GEMM over its samples' interleaved real view, and its "
         "projection; host radial analytic synthesis also uses unallocated row views "
         "for partial chunks; an envelope's synthesis and projection the same way; "
         "OpenBLAS, "
         "cuBLAS and rocBLAS builds also use "
         "preconverted complex GEMM for eligible envelope transforms; eligible chunked "
         "transforms are packed into persistent buffers for vendor BLAS; vector state "
         "uses custom; a host build without a vendor BLAS runs KokkosBlas's native "
         "arithmetic through a tiled host kernel, bit for bit; a compute capability "
         "8.6 build forms the real GEMMs with a device kernel in place of cuBLAS";
}

std::size_t GeneralWorkspace::response_team_size() const noexcept {
  constexpr auto configured = static_cast<std::size_t>(SOLPROP_NATIVE_DEVICE_TEAM_SIZE);
#ifdef KOKKOS_ENABLE_CUDA
  // Measured on a Hopper GPU, the crossover is sharp: 4,096-sample fibre problems
  // favour teams of 256, while 8,192 samples and more favour 512. Keep an explicitly
  // configured smaller team intact.
  if (problem_.dims.n_to < 8192)
    return std::min(configured, std::size_t{256});
#endif
  return configured;
}

// Size from the plan's real-side batch samples, using the shared complete-window
// thresholds in cpu_execution.hpp. Explicit FFT-pool requests bypass automatic sizing.
// FFTW's factorisation can change with the width; comparisons use solprop's SEP 25
// transform/propagation rounding budgets rather than bitwise agreement.
const ExecutionSpace& GeneralWorkspace::fft_space_for(const std::size_t samples,
                                                      const std::size_t length) {
  // NOLINTNEXTLINE(readability-suspicious-call-argument): the names only coincide
  return fft_pool_.get(CpuOperation::fft, samples, length);
}

void GeneralWorkspace::set_initial(const std::span<const Complex> field,
                                   const std::span<const Complex> noise) {
  require_extent(field.size(), problem_.n_columns() * problem_.dims.n_omega,
                 "initial field");
  if (!noise.empty())
    require_extent(noise.size(), field.size(), "noise");
  upload_span(execution_space_, crossing_ring_, current_, field);
  crossed_.from_host += field.size_bytes();
  if (noise.empty()) {
    if (problem_.stepper.modified_noise)
      graph::deep_copy(execution_space_, noise_, Complex{});
    has_noise_ = false;
  } else {
    if (!problem_.stepper.modified_noise)
      throw std::invalid_argument(
          "a native noise field is valid only for modified noise");
    upload_span(execution_space_, crossing_ring_, noise_, noise);
    crossed_.from_host += noise.size_bytes();
    has_noise_ = true;
  }
  mask(current_);
  if (!stash_current_.empty())
    graph::deep_copy(execution_space_, stash_current_, 0.0);
  execution_space_.fence("solprop_native set_initial");
  initialized_ = true;
  rk_source_evaluations_ = 0;
  table_ready_ = false;
  stages_ready_ = false;
  stash_is_current_ = false;
  quadrature_error_ = Kokkos::Experimental::quiet_NaN_v<double>;
  quadrature_stale_ = false;
  graph::deep_copy(execution_space_, quadrature_scores_,
                   Kokkos::Experimental::quiet_NaN_v<double>);
  clear_fsal();
}

// Point the per-stage scalar views at a slot of the packed buffer.
void GeneralWorkspace::bind_stage_scalars(const std::size_t slot) {
  const auto stages = problem_.stepper.n_stages;
  const auto channels = problem_.plasma.enabled ? problem_.plasma.n_channels : 0;
  double* const base = stage_scalars_.data() + slot * stage_slot_size_;
  const auto region = [&](const std::size_t index) {
    return base + stage_offsets_[index];
  };
  const auto extent = [&](const std::size_t index) {
    return stage_offsets_[index + 1] - stage_offsets_[index];
  };
  stage_phi_delta_ = RealView(region(0), extent(0));
  stage_phi_at_ = RealView(region(1), extent(1));
  stage_prefactor_ = RealView(region(2), extent(2));
  stage_density_ = RealView(region(3), stages);
  stage_kerr_density_ = RealView(region(4), stages);
  stage_plasma_density_ = RealView(region(5), stages);
  stage_raman_density_ = RealView(region(6), stages);
  stage_channel_density_ = RealView2(region(7), stages, channels);
  stage_channel_preionisation_ = RealView2(region(8), stages, channels);
  stage_taper_ = RealView(region(9), stages);
  stage_preionisation_ = RealView(region(10), stages);
  if (problem_.raman.enabled) {
    stage_raman_rates_ = RealView2(region(11), stages, problem_.raman.n_terms);
    if (problem_.raman.n_components > 1)
      stage_raman_densities_ =
          RealView2(region(12), stages, problem_.raman.n_components);
  }
  // A compensated trial's rate weights are its step's, in slot 0 whichever slot the
  // stage views read.
  stage_rate_weights_ =
      RealView(stage_scalars_.data() + stage_offsets_[13], extent(13));
}

// The host's record of a trial's scalars beside the views: which stages' phase
// weights repeat the stage before's, and the dispersive response's densities.
void GeneralWorkspace::note_trial_scalars(const TrialScalars& scalars) {
  // Rows of equal weights give bitwise equal phases (a zero weight's term is skipped).
  const auto n_phi = problem_.linear.n_phi_terms;
  const auto same_row = [&](const std::size_t row) {
    return std::equal(
        scalars.phi_delta.begin() + static_cast<std::ptrdiff_t>(row * n_phi),
        scalars.phi_delta.begin() + static_cast<std::ptrdiff_t>((row + 1) * n_phi),
        scalars.phi_delta.begin() + static_cast<std::ptrdiff_t>((row - 1) * n_phi));
  };
  trial_taper_.assign(scalars.taper.begin(), scalars.taper.end());
  phase_repeats_.assign(problem_.stepper.n_stages, false);
  for (std::size_t stage = 1; stage < problem_.stepper.n_stages; ++stage)
    phase_repeats_[stage] = same_row(stage);
  whole_repeats_ = problem_.stepper.n_stages > 0 && same_row(problem_.stepper.n_stages);
  entered_stage_ = std::numeric_limits<std::size_t>::max();
  if (auto* weights = dispersive_weights())
    weights->set_densities(scalars.dispersive_densities);
}

// Fill the next staging buffer with the trial's scalars, at each view's place in
// slot 0 -- and an Adams trial's sample's in slot 1, and a compensated trial's rate
// weights -- then copy it across once (a host build's mirror is the buffer itself)
// and point the views at slot 0. `use_uploaded_sample` turns them to slot 1.
void GeneralWorkspace::upload_trial(const TrialScalars& scalars,
                                    const TrialScalars* sample,
                                    const std::vector<double>* rate) {
  auto staging = stage_staging_[stage_staging_next_];
  const bool staged = staging.data() != stage_scalars_.data();
  // The ring comes round: every earlier copy from it has run once the space drains.
  if (staged && stage_staging_next_ == 0)
    execution_space_.fence("solprop_native stage staging reuse");
  const auto& previous =
      stage_staging_[(stage_staging_next_ + stage_staging_.size() - 1) %
                     stage_staging_.size()];
  stage_staging_next_ = (stage_staging_next_ + 1) % stage_staging_.size();
  if (staged)
    // A device's staging buffer starts from the last values uploaded, so that a view
    // this trial does not set keeps them, as the separate copies did.
    std::copy(previous.data(), previous.data() + staging.size(), staging.data());
  const auto fill_slot = [&](const std::size_t slot, const TrialScalars& values) {
    double* const base = staging.data() + slot * stage_slot_size_;
    const auto place = [&](const std::size_t region,
                           const std::vector<double>& source) {
      if (source.size() != stage_offsets_[region + 1] - stage_offsets_[region])
        throw std::invalid_argument("fixed problem array has the wrong extent");
      std::ranges::copy(source, base + stage_offsets_[region]);
    };
    const auto fill = [&](const std::size_t region, const double value) {
      std::fill(base + stage_offsets_[region], base + stage_offsets_[region + 1],
                value);
    };
    place(0, values.phi_delta);
    place(1, values.phi_at);
    place(2, values.prefactor);
    place(3, values.density);
    place(4, values.kerr_density.empty() ? values.density : values.kerr_density);
    if (problem_.plasma.enabled) {
      if (values.plasma_densities.empty()) {
        place(5, values.density);
        place(7, values.density);
        place(8, values.preionisation);
      } else {
        place(7, values.plasma_densities);
        place(8, values.plasma_preionisation);
        // The sole channel already has stage-major contiguous scalar vectors.
        // Multiple channels consume their own matrices and need no extracted copy.
        if (problem_.plasma.n_channels == 1) {
          place(5, values.plasma_densities);
          place(10, values.plasma_preionisation);
        }
      }
    }
    if (problem_.raman.enabled) {
      if (values.raman_densities.empty())
        place(6, values.density);
      else if (problem_.raman.n_components == 1)
        place(6, values.raman_densities);
      else
        fill(6, 1.0);
      place(11, values.raman_rates);
      if (problem_.raman.n_components > 1)
        place(12, values.raman_densities);
    }
    place(9, values.taper);
    if (problem_.plasma.enabled && values.plasma_preionisation.empty())
      place(10, values.preionisation);
  };
  fill_slot(0, scalars);
  if (sample != nullptr)
    fill_slot(1, *sample);
  if (rate != nullptr) {
    require_extent(rate->size(), problem_.linear.n_phi_terms, "Adams rate weights");
    std::ranges::copy(*rate, staging.data() + stage_offsets_[13]);
  }
  graph::deep_copy(execution_space_, stage_scalars_, staging);
  bind_stage_scalars(0);
  note_trial_scalars(scalars);
}

// An Adams trial's sample scalars, uploaded with its step's: the views turn to slot 1.
void GeneralWorkspace::use_uploaded_sample(const TrialScalars& sample) {
  bind_stage_scalars(1);
  note_trial_scalars(sample);
}

void GeneralWorkspace::prepare_raman(const std::vector<double>& rates,
                                     const std::vector<double>& densities) {
  if (!problem_.raman.enabled)
    throw std::logic_error("this state has no Raman workspace");
  require_extent(rates.size(), problem_.raman.n_terms, "Raman decay rates");
  if (!std::ranges::all_of(rates, [](const double value) {
        return std::isfinite(value) && value >= 0.0;
      }))
    throw std::invalid_argument("Raman decay rates must be finite and non-negative");
  auto host = Kokkos::create_mirror_view(stage_raman_rates_);
  for (std::size_t term = 0; term < problem_.raman.n_terms; ++term)
    host(0, term) = rates[term];
  graph::deep_copy(execution_space_, stage_raman_rates_, host);
  if (problem_.raman.n_components > 1) {
    require_extent(densities.size(), problem_.raman.n_components,
                   "Raman amplitude densities");
    if (!std::ranges::all_of(densities, [](double value) {
          return std::isfinite(value) && value >= 0.0;
        }))
      throw std::invalid_argument(
          "Raman amplitude densities must be finite and non-negative");
    auto density_host = Kokkos::create_mirror_view(stage_raman_densities_);
    for (std::size_t component = 0; component < densities.size(); ++component)
      density_host(0, component) = densities[component];
    graph::deep_copy(execution_space_, stage_raman_densities_, density_host);
  }
  prepare_raman_stage(0, rates.data(), densities.data());
  execution_space_.fence("solprop_native Raman diagnostic boundary");
}

void GeneralWorkspace::prepare_raman_stage(const std::size_t stage, const double* rates,
                                           const double* densities) {
  bool unchanged = raman_kernel_ready_;
  for (std::size_t term = 0; term < problem_.raman.n_terms; ++term)
    unchanged = unchanged && cached_raman_rates_[term] == rates[term];
  if (problem_.raman.n_components > 1)
    for (std::size_t component = 0; component < problem_.raman.n_components;
         ++component)
      unchanged =
          unchanged && cached_raman_densities_[component] == densities[component];
  if (unchanged)
    return;
  for (std::size_t term = 0; term < problem_.raman.n_terms; ++term)
    cached_raman_rates_[term] = rates[term];
  if (problem_.raman.n_components > 1)
    for (std::size_t component = 0; component < problem_.raman.n_components;
         ++component)
      cached_raman_densities_[component] = densities[component];

  Kokkos::Profiling::pushRegion("solprop_native.raman.kernel_rebuild");
  const auto undamped = raman_undamped_;
  const auto weights = raman_tensor_weights_;
  const auto kernel = raman_kernel_time_;
  const auto stage_rates = stage_raman_rates_;
  const auto amplitudes = stage_raman_densities_;
  const auto row_component = raman_row_component_;
  const auto fractions = raman_component_fraction_;
  const bool multiple = problem_.raman.n_components > 1;
  const bool tensor = problem_.effective_node_pol() == 2;
  const auto terms = problem_.raman.n_terms;
  const auto doubled = 2 * problem_.dims.n_to;
  const double fraction = problem_.raman.fraction;
  const double dt = problem_.grid.dt_fine;
  launch::parallel_for(
      "solprop_native.raman.kernel_rebuild",
      Range2(raman_rebuild_space_, {0, 0}, {problem_.raman_kernels(), doubled}),
      KOKKOS_LAMBDA(const std::size_t channel, const std::size_t time) {
        double total = 0.0;
        const double delay = static_cast<double>(time) * dt;
        if (multiple) {
          for (std::size_t component = 0; component < amplitudes.extent(1);
               ++component) {
            double response = 0.0;
            for (std::size_t term = 0; term < terms; ++term) {
              if (static_cast<std::size_t>(row_component(term)) == component) {
                const double damped = undamped(term, time) *
                                      Kokkos::exp(-stage_rates(stage, term) * delay);
                response += tensor ? weights(term, channel) * damped : damped;
              }
            }
            total += amplitudes(stage, component) * fractions(component) * response;
          }
          kernel(channel, time) = total;
        } else {
          for (std::size_t term = 0; term < terms; ++term) {
            const double damped =
                undamped(term, time) * Kokkos::exp(-stage_rates(stage, term) * delay);
            total += tensor ? weights(term, channel) * damped : damped;
          }
          kernel(channel, time) = fraction * total;
        }
      });
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("solprop_native.raman.kernel_fft");
  execute_raman(*raman_kernel_forward_plan_, raman_kernel_time_, raman_kernel_spectrum_,
                KokkosFFT::Normalization::none);
  const auto spectrum = raman_kernel_spectrum_;
  launch::parallel_for(
      "solprop_native.raman.kernel_dt_scale",
      Range2(raman_scale_space_, {0, 0},
             {problem_.raman_kernels(), problem_.dims.n_to + 1}),
      KOKKOS_LAMBDA(const std::size_t channel, const std::size_t index) {
        spectrum(channel, index) *= dt;
      });
  Kokkos::Profiling::popRegion();

  raman_kernel_ready_ = true;
  ++raman_kernel_transforms_;
}

void GeneralWorkspace::prepare_trial(const double dz, const TrialScalars& scalars) {
  if (problem_.stepper.low_storage)
    refuse<std::logic_error>("trial_step: a low-storage state of ", current_.extent(0),
                             " x ", current_.extent(1),
                             " holds no Runge-Kutta views; its steps are "
                             "low_storage_step's");
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  require_trial_scalars(scalars);
  coarse_pending_ = false;
  upload_trial(scalars);
  if (flow_)
    flow_->set_trial(scalars.flow_section, scalars.flow_weights);
  if (general_)
    general_->set_weights(scalars.transform_weights);
  std::size_t first = 0;
  if (has_recycled_) {
    reuse_fsal();
    first = 1;
  }
  // A stage's finish runs in the next stage's entry kernel (solprop SEP 25 section
  // 6.7) unless a coupled flow carries the stage back first, or the stage is the last.
  finish_pending_ = false;
  const bool deferrable = !flow_ && !problem_.separable_geometry();
  defer_quadrature_ = deferred_quadrature_eligible();
  for (std::size_t stage = first; stage < problem_.stepper.n_stages; ++stage) {
    try {
      if (problem_.raman.enabled)
        prepare_raman_stage(
            stage, scalars.raman_rates.data() + stage * problem_.raman.n_terms,
            problem_.raman.n_components > 1
                ? scalars.raman_densities.data() + stage * problem_.raman.n_components
                : nullptr);
      defer_finish_ = deferrable && stage + 1 < problem_.stepper.n_stages;
      project_coarse_ = !defer_quadrature_ || stage + 1 == problem_.stepper.n_stages;
      evaluate_stage(stage, dz);
    } catch (...) {
      defer_finish_ = false;
      finish_pending_ = false;
      defer_quadrature_ = false;
      project_coarse_ = true;
      coarse_pending_ = false;
      throw;
    }
  }
  defer_finish_ = false;
  defer_quadrature_ = false;
  project_coarse_ = true;
  if (finish_pending_)
    throw std::logic_error("a deferred stage finish was never run");
  stages_ready_ = true;
}

std::vector<double> GeneralWorkspace::trial_step(const double dz,
                                                 const TrialScalars& scalars) {
  prepare_trial(dz, scalars);
  if (flow_)
    flow_->carry_step(current_, solution_, error_);
  const auto current = current_;
  const auto carried = flow_ ? flow_->carried_state() : current_;
  const auto solution = flow_ ? flow_->carried_solution() : solution_;
  const auto error = flow_ ? flow_->carried_error() : error_;
  const auto whole = whole_propagator_;
  const auto advanced = advanced_;
  const auto residual = residual_;
  const auto in_norm = norm_mask_;
  const auto partials = partials_;
  const auto count = problem_.n_columns();
  const auto n = problem_.dims.n_omega;
  const auto slots = reduction_slots_;
  accept_update(
      execution_space_, count, n, dz,
      KOKKOS_LAMBDA(const std::size_t column) {
        const auto slot = column * slots;
        return UpdateColumn{.current = current.data() + column * current.stride(0),
                            .solution = solution.data() + column * solution.stride(0),
                            .error = error.data() + column * error.stride(0),
                            .whole = whole.data() + column * whole.stride(0),
                            .advanced = advanced.data() + column * advanced.stride(0),
                            .residual = residual.data() + column * residual.stride(0),
                            .in_norm = in_norm.data(),
                            .old_partial = partials.data() + slot,
                            .new_partial = partials.data() + partials.stride(0) + slot,
                            .error_partial =
                                partials.data() + 2 * partials.stride(0) + slot,
                            .carried = carried.data() + column * carried.stride(0)};
      },
      "solprop_native.step.accept_error_partials", slots);
  return collect_trial();
}

// Adopt an Adams step's end state, given on the host before its boundary treatments:
// the shared finish masks it, or applies the split-step absorber, and makes it current.
void GeneralWorkspace::adopt(const std::vector<Complex>& field, const double dz) {
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  require_extent(field.size(), problem_.n_columns() * problem_.dims.n_omega,
                 "Adams state");
  copy_vector(execution_space_, advanced_, field);
  finish(dz);
  // The stash a response pass left is the predicted field's, not this one's.
  stash_is_current_ = false;
}

// clang-format off: the fragments keep the unit's order.
#include "trial_low_storage.inc"
#include "trial_adams.inc"
#include "trial_separable.inc"
#include "trial_stage.inc"
#include "trial_response.inc"
#include "trial_finish.inc"
#include "trial_split_step.inc"
// clang-format on

} // namespace solprop_native
