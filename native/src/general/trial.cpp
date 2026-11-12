// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The general workspace's hot set: the trial step, its stages and responses, the
// Adams and low-storage marches and the split-step absorber, in one unit so that the
// compiler sees every call between them.

#include "workspace.hpp"

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
  return !general_ && !dispersive_ && !held_radial() && !gated_plasma() &&
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
// thresholds in core.hpp. Explicit FFT-pool requests bypass automatic sizing.
// FFTW's factorisation can change with the width; comparisons use solprop's SEP 25
// transform/propagation rounding budgets rather than bitwise agreement.
const ExecutionSpace& GeneralWorkspace::fft_space_for(const std::size_t samples,
                                                      const std::size_t length) {
  // NOLINTNEXTLINE(readability-suspicious-call-argument): the names only coincide
  return fft_pool_.get(CpuOperation::fft, samples, length);
}

void GeneralWorkspace::set_initial(const std::span<const Complex> field,
                                   const std::span<const Complex> noise) {
  require_open();
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
  require_open();
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
  require_open();
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

// solprop's low-call Adams march (SEP 13, SEP 14 phase 2): one source evaluation at a
// field given in the step's anchor frame, `U(anchor, position) N(U(position, anchor)
// v)`. It is stage zero of a trial whose stage-zero scalars are the evaluation
// position's: the combination with no earlier stage carries the input to the position
// and the finish brings the source back, so the shared stage pipeline, its responses
// and its in-stage absorber are the march's too. The accepted state is held in
// `advanced_`, which only the Runge-Kutta accept writes, while the input takes its
// place.
std::vector<Complex> GeneralWorkspace::adams_source(const std::vector<Complex>& input,
                                                    const TrialScalars& scalars) {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  // Every route's stage source serves the Adams march (solprop SEP 16 phase 3): the
  // separable stage takes its flow from the stage's own Phi increment, as the modal one
  // does. Beside a coupled flow the sample is the material source alone (solprop SEP
  // 16 phase 5): the exponential formulation carries the residual in its compensation
  // and the flow in its moments, so the stage runs without the flow's pipeline.
  require_extent(input.size(), problem_.n_columns() * problem_.dims.n_omega,
                 "Adams input");
  // The flow is set aside for the evaluation and put back whatever happens.
  auto held_flow = std::exchange(flow_, nullptr);
  // An aggregate, so it declares no copy or move; it is never copied or moved.
  // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions): a local scope guard
  struct RestoreFlow {
    std::unique_ptr<CoupledFlowAction>& slot;
    std::unique_ptr<CoupledFlowAction>& held;
    // NOLINTNEXTLINE(modernize-use-equals-default): restores the slot (see FlowAside).
    ~RestoreFlow() { slot = std::move(held); }
  } restore{.slot = flow_, .held = held_flow};
  require_trial_scalars(scalars);
  upload_trial(scalars);
  if (general_)
    general_->set_weights(scalars.transform_weights);
  graph::deep_copy(execution_space_, advanced_, current_);
  copy_vector(execution_space_, current_, input);
  if (problem_.raman.enabled)
    prepare_raman_stage(0, scalars.raman_rates.data(),
                        problem_.raman.n_components > 1 ? scalars.raman_densities.data()
                                                        : nullptr);
  evaluate_stage_source(0, 0.0);
  graph::deep_copy(execution_space_, current_, advanced_);
  stages_ready_ = true;
  // No Runge-Kutta stage of this state exists to recycle.
  pending_fsal_ = false;
  has_recycled_ = false;
  return stage_to_host(0);
}

// Adopt an Adams step's end state, given on the host before its boundary treatments:
// the shared finish masks it, or applies the split-step absorber, and makes it current.
void GeneralWorkspace::adopt(const std::vector<Complex>& field, const double dz) {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  require_extent(field.size(), problem_.n_columns() * problem_.dims.n_omega,
                 "Adams state");
  copy_vector(execution_space_, advanced_, field);
  finish(dz);
  // The stash a response pass left is the predicted field's, not this one's.
  stash_is_current_ = false;
}

// --- solprop SEP 24 phase 3: the low-storage RK4IP march -----------------------------
//
// Classical RK4 in the interaction picture anchored at the step's midpoint (solprop's
// backends/low_storage.py, SEP 24 section 4.3). The state is the register A; B
// accumulates the bracket of the accepted step and C holds the next pass's lab-frame
// input, unless it is elided (solprop's `low_storage_stage_register`): then each of the
// first three finishes writes that input straight into the transverse scratch. A step
// is four passes, each forming its input in the route's scratch, running the stage core
// the Runge-Kutta march shares (`separable_core`, `modal_core`; SEP 24 Q5) and folding
// the source into the registers per element:
//
//   pass 1, F1 at z_n:  A <- P A; B <- A + h/6 P S; C <- A + h/2 P S
//   pass 2, F2 at z_m:  B <- B + h/3 S; C <- A + h/2 S
//   pass 3, F3 at z_m:  B <- B + h/3 S; C <- Q (A + h S)
//   pass 4, F4 at end:  A <- Q B + h/6 S
//
// Every pass reads the source at its own position, so nothing carries it back to an
// anchor, and only forward flows are applied. The step is accepted in place; the
// caller's finish masks it or applies the split-step collar.

namespace {

constexpr int fold_complete = 0;
constexpr int fold_exchange = 1;
constexpr int fold_second = 2;
constexpr int term_none = 0;
constexpr int term_material = 1;
constexpr int term_absorber = 2;

// The scalar row each pass reads: the step's start, its midpoint twice, its end.
constexpr std::size_t low_storage_rows[4] = {0, 1, 1, 2};

// One evaluation's fold into the registers at one element: solprop's
// `bodies.low_storage_fold`, the finish column of SEP 24 section 4.3. A pass that
// evaluates twice -- the material response, then the in-stage absorber's term, from the
// same input -- folds the first at the exchange (pass 3's C left unflowed) and the
// second after it, whose sum is the pass's.
KOKKOS_INLINE_FUNCTION void
low_storage_fold(const int stage, const int mode, const double step,
                 const Complex source, Complex& state, Complex& accumulator,
                 Complex& stage_register, const Complex first, const Complex second) {
  if (stage == 0) {
    const Complex slope = first * source;
    if (mode == fold_second) {
      accumulator = accumulator + (step / 6.0) * slope;
      stage_register = stage_register + (0.5 * step) * slope;
      return;
    }
    const Complex moved = first * state;
    state = moved;
    accumulator = moved + (step / 6.0) * slope;
    stage_register = moved + (0.5 * step) * slope;
    return;
  }
  if (stage == 1) {
    accumulator = accumulator + (step / 3.0) * source;
    stage_register =
        (mode == fold_second ? stage_register : state) + (0.5 * step) * source;
    return;
  }
  if (stage == 2) {
    accumulator = accumulator + (step / 3.0) * source;
    if (mode == fold_complete)
      stage_register = second * (state + step * source);
    else if (mode == fold_exchange)
      stage_register = state + step * source;
    else
      stage_register = second * (stage_register + step * source);
    return;
  }
  if (mode == fold_second) {
    state = state + (step / 6.0) * source;
    return;
  }
  state = second * accumulator + (step / 6.0) * source;
}

// A register traversal over the state's elements. The last fold of a step also reduces
// the accepted state's squared norm per column into `partials(0, column)`, one team per
// column as the Runge-Kutta accept reduces its own (its sums associate as the team
// splits them, solprop SEP 25).
template <bool Crossing = false, class Element>
void register_traversal(const char* label, const ExecutionSpace& execution,
                        const std::size_t columns, const std::size_t n,
                        const RealView2& partials, const bool reduce,
                        const Element& element) {
  const auto traversal_for = [](const char* name, const ExecutionSpace& space,
                                const std::size_t rows, const std::size_t width,
                                const auto& body) {
    if constexpr (Crossing)
      crossing_for(name, space, rows, width, body);
    else
      cartesian_for(name, space, rows, width, body);
  };
  if (!reduce) {
    traversal_for(
        label, execution, columns, n,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          static_cast<void>(element(column, index));
        });
    return;
  }
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  if constexpr (Crossing) {
    // Blocks of adjacent columns, each column's sum in ascending bin order.
    constexpr std::size_t block = crossing_tile_columns;
    constexpr std::size_t bins = crossing_tile_bins;
    const std::size_t blocks = (columns + block - 1) / block;
    launch::parallel_for(
        label, Kokkos::RangePolicy<ExecutionSpace>(execution, 0, blocks),
        KOKKOS_LAMBDA(const std::size_t which) {
          const std::size_t first = which * block;
          const std::size_t count = columns - first < block ? columns - first : block;
          double totals[block] = {};
          for (std::size_t start = 0; start < n; start += bins) {
            const std::size_t stop = n - start < bins ? n : start + bins;
            for (std::size_t index = start; index < stop; ++index)
              for (std::size_t offset = 0; offset < count; ++offset)
                totals[offset] += element(first + offset, index);
          }
          for (std::size_t offset = 0; offset < count; ++offset)
            partials(0, first + offset) = totals[offset];
        });
    return;
  }
#endif
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  using Member = Policy::member_type;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const Policy policy(execution, static_cast<int>(columns),
                      SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
  launch::parallel_for(
      label, policy, KOKKOS_LAMBDA(const Member& team) {
        const auto column = static_cast<std::size_t>(team.league_rank());
        double total = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n)),
            [&](const int index, double& update) {
              update += element(column, static_cast<std::size_t>(index));
            },
            total);
        Kokkos::single(Kokkos::PerTeam(team), [&]() { partials(0, column) = total; });
      });
#else
  const Policy policy(execution, static_cast<int>(columns),
                      host_reduction_team_size(execution, columns, n));
  launch::parallel_for(
      label, policy, KOKKOS_LAMBDA(const Member& team) {
        const auto column = static_cast<std::size_t>(team.league_rank());
        double total = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, n),
            [&](const std::size_t index, double& update) {
              update += element(column, index);
            },
            total);
        Kokkos::single(Kokkos::PerTeam(team), [&] { partials(0, column) = total; });
      });
#endif
}

} // namespace

double GeneralWorkspace::low_storage_step(const double dz, const TrialScalars& scalars,
                                          const bool rebuild_table) {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (!problem_.stepper.low_storage)
    throw std::logic_error("this native state was not built for the low-storage march");
  if (!(dz > 0.0) || !std::isfinite(dz))
    throw std::invalid_argument("the low-storage step must be positive and finite");
  if (problem_.stepper.n_stages < 3)
    throw std::invalid_argument("the low-storage march reads three scalar rows");
  if (rebuild_table && problem_.stepper.low_storage_flows != 1)
    throw std::logic_error("only a constant operator's table is rebuilt");
  if (problem_.stepper.low_storage_flows == 1 && !rebuild_table && !table_ready_)
    throw std::logic_error("the low-storage table has not been built since the initial "
                           "field was loaded");
  require_trial_scalars(scalars);
  upload_trial(scalars);
  if (general_)
    general_->set_weights(scalars.transform_weights);
  low_storage_flows_formed(rebuild_table);
  if (rebuild_table)
    table_ready_ = true;
  for (std::size_t pass = 0; pass < 4; ++pass) {
    const auto row = low_storage_rows[pass];
    if (problem_.raman.enabled)
      prepare_raman_stage(
          row, scalars.raman_rates.data() + row * problem_.raman.n_terms,
          problem_.raman.n_components > 1
              ? scalars.raman_densities.data() + row * problem_.raman.n_components
              : nullptr);
    if (problem_.separable_geometry())
      low_storage_separable_pass(pass, dz);
    else
      low_storage_modal_pass(pass, dz);
  }
  // No Runge-Kutta stage exists to observe or recycle.
  stages_ready_ = false;
  stash_is_current_ = false;
  execution_space_.fence("solprop_native low-storage step");
  double total = 0.0;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // One number crosses, not the whole of the partials' table.
  const auto partials = partials_;
  launch::parallel_reduce(
      "solprop_native.low_storage.norm_total",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_columns()),
      KOKKOS_LAMBDA(const std::size_t column, double& sum) {
        sum += partials(0, column);
      },
      Kokkos::Sum<double>(total));
#else
  for (std::size_t column = 0; column < problem_.n_columns(); ++column)
    total += partials_(0, column);
#endif
  return total;
}

// The step's held flows: a constant operator's table `exp(h_t D / 2)`, built from the
// table's interval (scalar row 2 of `phi_delta`) when the caller's key changed, or a
// varying operator's `P` and `Q` from rows 0 and 1, every step. Recomputed flows are
// formed in the finishes that apply them and hold nothing.
void GeneralWorkspace::low_storage_flows_formed(const bool rebuild_table) {
  const auto flows = low_storage_flows_;
  const auto count = problem_.stepper.low_storage_flows;
  if (count == 0 || (count == 1 && !rebuild_table))
    return;
  const auto n = problem_.dims.n_omega;
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  // With vector maths a host task forms a column's exponents a block of bins at a
  // time and their exponentials in one pass (solprop ADR-059); a separable route's
  // task, a block of adjacent columns at one bin, its table's run in the scratch's
  // layout (`RegisterIndex`).
  if constexpr (vmath::vectorised) {
    if (problem_.separable_geometry()) {
      const auto spatial = spatial_;
      const auto delta = stage_phi_delta_;
      const auto active = active_mask_;
      const auto index = register_index();
      const auto columns = problem_.n_columns();
      const std::size_t blocks = (columns + stage_block - 1) / stage_block;
      launch::parallel_for(
          "solprop_native.low_storage.separable_table",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n * blocks),
          KOKKOS_LAMBDA(const std::size_t item) {
            const std::size_t omega = item / blocks;
            const std::size_t first = (item % blocks) * stage_block;
            const std::size_t width =
                columns - first < stage_block ? columns - first : stage_block;
            const bool in_band = active(omega) != 0U;
            double real[stage_block];
            double imag[stage_block];
            for (std::size_t offset = 0; offset < width; ++offset) {
              const Complex x =
                  delta(2) *
                  spatial.value(index.mode(first + offset), omega, in_band).linear;
              real[offset] = x.real();
              imag[offset] = x.imag();
            }
            vmath::exp_complex(real, imag, real, imag, width);
            for (std::size_t offset = 0; offset < width; ++offset)
              flows.data()[index(first + offset, omega)] =
                  Complex(real[offset], imag[offset]);
          });
      return;
    }
    const auto terms = stage_terms(0);
    const auto n_phi = problem_.linear.n_phi_terms;
    const std::size_t blocks = (n + stage_block - 1) / stage_block;
    launch::parallel_for(
        "solprop_native.low_storage.modal_flows",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0,
                                            problem_.n_columns() * blocks),
        KOKKOS_LAMBDA(const std::size_t item) {
          const std::size_t column = item / blocks;
          const std::size_t first = (item % blocks) * stage_block;
          const std::size_t bins = n - first < stage_block ? n - first : stage_block;
          double real[stage_block];
          double imag[stage_block];
          const auto form = [&](const std::size_t flow, const std::size_t offset) {
            terms.phase_block(offset, column, first, bins, real, imag);
            vmath::exp_complex(real, imag, real, imag, bins);
            for (std::size_t bin = 0; bin < bins; ++bin)
              flows(flow, column, first + bin) = Complex(real[bin], imag[bin]);
          };
          if (count == 1) {
            form(0, 2 * n_phi);
          } else {
            form(0, 0);
            form(1, n_phi);
          }
        });
    return;
  }
#endif
  if (problem_.separable_geometry()) {
    const auto spatial = spatial_;
    const auto delta = stage_phi_delta_;
    const auto active = active_mask_;
    const auto n_pol = problem_.dims.n_pol_state;
    const auto index = register_index();
    cartesian_for(
        "solprop_native.low_storage.separable_table", execution_space_,
        problem_.n_columns(), n,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
          const auto values = spatial.value(column / n_pol, omega, active(omega) != 0U);
          flows.data()[index(column, omega)] = propagator_exp(delta(2) * values.linear);
        });
    return;
  }
  const auto terms = stage_terms(0);
  const auto n_phi = problem_.linear.n_phi_terms;
  cartesian_for(
      "solprop_native.low_storage.modal_flows", execution_space_, problem_.n_columns(),
      n, KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        if (count == 1) {
          flows(0, column, index) =
              propagator_exp(terms.phase_sum(2 * n_phi, column, index));
        } else {
          flows(0, column, index) = propagator_exp(terms.phase_sum(0, column, index));
          flows(1, column, index) =
              propagator_exp(terms.phase_sum(n_phi, column, index));
        }
      });
}

// One pass on a separable geometry (free space and the hybrid planar kind). Its input
// is the register A (pass 1) or C, with the noise draw at the pass's position, in the
// transverse scratch -- already written by the previous finish when C is elided. Under
// an in-stage absorber the pass evaluates twice from the same input: the material
// source, then the absorber's term, the exchange between them taking the material
// output out of the scratch and re-forming the input (SEP 24 section 4.3).
void GeneralWorkspace::low_storage_separable_pass(const std::size_t pass,
                                                  const double dz) {
  const bool material = problem_.kerr.nonlinear || problem_.plasma.enabled ||
                        problem_.raman.enabled || separable_dispersive_ != nullptr;
  const bool absorber = problem_.absorber.in_stage;
  const bool elided = !problem_.stepper.low_storage_stage_register;
  const auto row = low_storage_rows[pass];
  if ((material || absorber) && (pass == 0 || !elided))
    low_storage_separable_input(pass);
  if (material && absorber) {
    separable_core(row, true);
    low_storage_separable_fold(pass, fold_exchange, term_material, dz);
    separable_core(row, false);
    low_storage_separable_fold(pass, fold_second, term_absorber, dz);
  } else if (material || absorber) {
    separable_core(row, material);
    low_storage_separable_fold(pass, fold_complete,
                               material ? term_material : term_absorber, dz);
  } else {
    low_storage_separable_fold(pass, fold_complete, term_none, dz);
  }
}

namespace {
// Each column's masked squared norm of `state` into `partials(0, column)`, in a team
// per column: a device's final fold traverses like the others and reduces here, over
// one volume, instead of reducing inside the fold's crossing (SEP 28).
void column_norms(const char* label, const ExecutionSpace& execution,
                  const ComplexView2& state, const ByteView& in_norm,
                  const RealView2& partials) {
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  using Member = Policy::member_type;
  const auto n = state.extent(1);
  launch::parallel_for(
      label, Policy(execution, static_cast<int>(state.extent(0)), 128),
      KOKKOS_LAMBDA(const Member& team) {
        const auto column = static_cast<std::size_t>(team.league_rank());
        double total = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n)),
            [&](const int index, double& update) {
              const auto omega = static_cast<std::size_t>(index);
              if (in_norm(omega) != 0U)
                update += squared_abs(state(column, omega));
            },
            total);
        Kokkos::single(Kokkos::PerTeam(team), [&] { partials(0, column) = total; });
      });
}
} // namespace

void GeneralWorkspace::low_storage_separable_input(const std::size_t pass) {
  const auto state = current_;
  const auto stage_register = register_stage_;
  const auto index = stage_index();
  const auto spatial = spatial_;
  const auto at = stage_phi_at_;
  const auto active = active_mask_;
  const auto noise = noise_;
  const auto scratch = transverse_;
  const bool add_noise = problem_.stepper.modified_noise && has_noise_;
  const double scale = inverse_transverse_count();
  const auto row = low_storage_rows[pass];
  const auto input = KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
    const auto mode = index.mode(column);
    Complex value =
        pass == 0 ? state(column, omega) : stage_register.data()[index(column, omega)];
    if (add_noise) {
      const auto values = spatial.value(mode, omega, active(omega) != 0U);
      const double angle = at(row) * values.linear.imag();
      value += noise(column, omega) * Complex(Kokkos::cos(angle), Kokkos::sin(angle));
    }
    return value * scale;
  };
  // In the volume route's order the scratch is the state's, so the input streams.
  if (problem_.stepper.low_storage_tiled_input && !volume_order_) {
    // Read along the state's bins and written along the scratch's rows, a tile at a
    // time.
    tiled_crossing(
        "solprop_native.low_storage.separable_input", execution_space_,
        problem_.n_columns(), problem_.dims.n_omega, false, input,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega,
                      const Complex value) {
          scratch.data()[index.cell(column, omega)] = value;
        });
    return;
  }
  crossing_for(
      "solprop_native.low_storage.separable_input", execution_space_,
      problem_.n_columns(), problem_.dims.n_omega,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        scratch.data()[index.cell(column, omega)] = input(column, omega);
      });
}

void GeneralWorkspace::low_storage_separable_fold(const std::size_t pass,
                                                  const int mode, const int term,
                                                  const double dz) {
  const auto state = current_;
  const auto accumulator = register_accumulator_;
  const auto stage_register = register_stage_;
  const auto flows = low_storage_flows_;
  const bool tabulated = problem_.stepper.low_storage_flows != 0;
  const auto scratch = transverse_;
  const auto spatial = spatial_;
  const auto window = omega_window_;
  const auto active = active_mask_;
  const auto in_norm = norm_mask_;
  const auto noise = noise_;
  const auto delta = stage_phi_delta_;
  const auto at = stage_phi_at_;
  // A device gathers and scatters the scratch through its own tiles.
  [[maybe_unused]] const auto n_pol = problem_.dims.n_pol_state;
  [[maybe_unused]] const auto layout = transverse_layout();
  const auto index = stage_index();
  const bool add_noise = problem_.stepper.modified_noise && has_noise_;
  const double scale = inverse_transverse_count();
  const int stage = static_cast<int>(pass);
  const bool exchange = mode == fold_exchange;
  const bool second = mode == fold_second;
  // Elided, each of the first three passes writes the next pass's input in place of C.
  const bool forward = !problem_.stepper.low_storage_stage_register && pass < 3;
  const bool final = pass == 3 && !exchange;
  // What this fold reads and writes, invariant over the traversal: each moves only the
  // registers its row of the flow table names.
  const bool flows_here =
      stage == 0 || (stage == 2 && !exchange) || (stage == 3 && !second);
  const bool reads_register = (second && stage < 3) || (exchange && stage > 0);
  const bool reads_accumulator = stage > 0 || second;
  const bool writes_state = (stage == 0 && !second) || stage == 3;
  const bool writes_register = stage < 3 && !forward;
  const auto own = low_storage_rows[pass];
  const auto next = low_storage_rows[pass < 3 ? pass + 1 : 3];
  const bool linear_here = add_noise || (flows_here && !tabulated);
  const std::size_t flow_row = stage == 0 ? 0 : 1;
  const double step = dz;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // A device reduces the accepted state's norms in a pass of their own.
  const bool fused_norm = false;
#else
  const bool fused_norm = final;
#endif
  // The element, given its cell of the transverse scratch and its state (`held`).
  const auto body = KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega,
                                  Complex& cell, Complex& held) {
    const auto column_mode = index.mode(column);
    const bool in_band = active(omega) != 0U;
    // Tabulated flows without noise, or a pass with no flow, read only the prefactor of
    // the element.
    SpatialElement values{};
    if (linear_here)
      values = spatial.value(column_mode, omega, in_band);
    else if (term == term_material)
      values.prefactor = spatial.prefactor(column_mode, omega, in_band);
    Complex source{};
    if (term == term_material)
      source = values.prefactor * window(omega) * cell;
    else if (term == term_absorber && in_band)
      source = cell;
    const std::size_t slot = index(column, omega);
    Complex flow(1.0, 0.0);
    if (flows_here)
      flow = tabulated ? flows.data()[slot]
                       : propagator_exp(delta(flow_row) * values.linear);
    Complex a = held;
    Complex b = reads_accumulator ? accumulator.data()[slot] : Complex{};
    Complex c = reads_register ? stage_register.data()[slot] : Complex{};
    if (exchange) {
      Complex value = stage == 0 ? a : c;
      if (add_noise) {
        const double angle = at(own) * values.linear.imag();
        value += noise(column, omega) * Complex(Kokkos::cos(angle), Kokkos::sin(angle));
      }
      cell = value * scale;
    }
    low_storage_fold(stage, mode, step, source, a, b, c, flow, flow);
    if (forward) {
      Complex value = c;
      if (add_noise) {
        const double angle = at(next) * values.linear.imag();
        value += noise(column, omega) * Complex(Kokkos::cos(angle), Kokkos::sin(angle));
      }
      cell = value * scale;
    }
    if (writes_state)
      held = a;
    if (stage < 3)
      accumulator.data()[slot] = b;
    if (writes_register)
      stage_register.data()[slot] = c;
    return final && in_norm(omega) != 0U ? squared_abs(a) : 0.0;
  };
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // In the volume route's order, the scratch's cell sits beside the registers' and the
  // state's, so the fold streams all three.
  if (volume_order_)
    cartesian_for(
        "solprop_native.low_storage.separable_fold", execution_space_,
        problem_.n_columns(), problem_.dims.n_omega,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
          static_cast<void>(body(column, omega,
                                 scratch.data()[index.cell(column, omega)],
                                 state(column, omega)));
        });
  else
    tiled_update(
        "solprop_native.low_storage.separable_fold", execution_space_,
        problem_.n_columns(), problem_.dims.n_omega, exchange || forward,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
          return layout.cell(scratch, omega * n_pol + column % n_pol, column / n_pol);
        },
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega,
                      Complex& cell) {
          static_cast<void>(body(column, omega, cell, state(column, omega)));
        },
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega,
                      const Complex value) {
          layout.cell(scratch, omega * n_pol + column % n_pol, column / n_pol) = value;
        });
#else
  // The host's registers sit at the scratch cell's own position (`RegisterIndex`), so a
  // block of adjacent columns reads them, and the scratch, in contiguous runs at every
  // bin; only the state is column-major. A work item takes a block of columns over
  // every bin, a tile of bins at a time: it loads the tile's state a column (a
  // contiguous run) at a time, folds the tile in the scratch's order, and stores the
  // state back. The final fold sums each column's norm in ascending bin order (as
  // `register_traversal`).
  constexpr std::size_t block = crossing_tile_columns;
  constexpr std::size_t bins = crossing_tile_bins;
  const std::size_t columns = problem_.n_columns();
  const std::size_t n = problem_.dims.n_omega;
  const std::size_t blocks = (columns + block - 1) / block;
  const auto partials = partials_;
  launch::parallel_for(
      "solprop_native.low_storage.separable_fold",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, blocks),
      KOKKOS_LAMBDA(const std::size_t which) {
        const std::size_t first = which * block;
        const std::size_t count = columns - first < block ? columns - first : block;
        double totals[block] = {};
        Complex tile[bins][block];
        for (std::size_t start = 0; start < n; start += bins) {
          const std::size_t width = n - start < bins ? n - start : bins;
          for (std::size_t offset = 0; offset < count; ++offset) {
            const Complex* row = &state(first + offset, start);
            for (std::size_t bin = 0; bin < width; ++bin)
              tile[bin][offset] = row[bin];
          }
          for (std::size_t bin = 0; bin < width; ++bin)
            for (std::size_t offset = 0; offset < count; ++offset) {
              const std::size_t column = first + offset;
              const std::size_t omega = start + bin;
              totals[offset] +=
                  body(column, omega, scratch.data()[index(column, omega)],
                       tile[bin][offset]);
            }
          if (writes_state)
            for (std::size_t offset = 0; offset < count; ++offset) {
              Complex* row = &state(first + offset, start);
              for (std::size_t bin = 0; bin < width; ++bin)
                row[bin] = tile[bin][offset];
            }
        }
        if (fused_norm)
          for (std::size_t offset = 0; offset < count; ++offset)
            partials(0, first + offset) = totals[offset];
      });
#endif
  if (final && !fused_norm)
    column_norms("solprop_native.low_storage.separable_norms", execution_space_, state,
                 in_norm, partials_);
}

// One pass on a modal route (mode sets, radial, general modal): its input, the
// register A (pass 1) or C with the noise draw at the pass's position, padded into the
// fine spectrum (and the general route's scaled channels); the shared core; and one
// fold of the source -- prefactor, window and the in-stage absorber's term at the
// pass's position, with no flow back to an anchor.
void GeneralWorkspace::low_storage_modal_pass(const std::size_t pass, const double dz) {
  const auto row = low_storage_rows[pass];
  const bool responds = general_
                            ? general_->responds() || problem_.absorber.in_stage
                            : problem_.kerr.nonlinear || problem_.plasma.enabled ||
                                  problem_.raman.enabled ||
                                  problem_.absorber.in_stage || dispersive_ != nullptr;
  if (responds)
    low_storage_modal_input(pass);
  const bool evaluated = modal_core(row);
  const auto terms = stage_terms(row);
  const auto state = current_;
  const auto accumulator = register_accumulator_;
  const auto stage_register = register_stage_;
  const auto flows = low_storage_flows_;
  const auto count = problem_.stepper.low_storage_flows;
  const auto in_norm = norm_mask_;
  const auto n_phi = problem_.linear.n_phi_terms;
  const auto n_prefactor = problem_.linear.n_prefactor_terms;
  const int stage = static_cast<int>(pass);
  const bool final = pass == 3;
  const bool flows_here = stage != 1;
  const std::size_t flow_row = stage == 0 ? 0 : 1;
  const double step = dz;
  register_traversal(
      "solprop_native.low_storage.modal_fold", execution_space_, problem_.n_columns(),
      problem_.dims.n_omega, partials_, final,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        Complex source{};
        if (evaluated) {
          // Zero weights skipped, as in `StageTerms::finish`.
          Complex prefactor{};
          for (std::size_t term = 0; term < n_prefactor; ++term) {
            const double weight = terms.prefactor_weights(row * n_prefactor + term);
            if (weight != 0.0)
              prefactor += weight * terms.prefactor_terms(term, column, index);
          }
          source = prefactor * terms.window(index) *
                   (terms.general ? terms.general_source(column, index)
                                  : terms.projected(column, terms.crop(index)));
          if (terms.in_stage_absorber && terms.active(index) != 0U)
            source += terms.absorber(column, terms.crop(index));
        }
        Complex flow(1.0, 0.0);
        if (flows_here)
          flow = count == 1 ? flows(0, column, index)
                 : count == 2
                     ? flows(flow_row, column, index)
                     : propagator_exp(terms.phase_sum(flow_row * n_phi, column, index));
        Complex a = state(column, index);
        Complex b = stage > 0 ? accumulator(column, index) : Complex{};
        Complex c{};
        low_storage_fold(stage, fold_complete, step, source, a, b, c, flow, flow);
        if (stage == 0 || stage == 3)
          state(column, index) = a;
        if (stage < 3) {
          accumulator(column, index) = b;
          stage_register(column, index) = c;
        }
        return final && in_norm(index) != 0U ? squared_abs(a) : 0.0;
      });
}

void GeneralWorkspace::low_storage_modal_input(const std::size_t pass) {
  const auto source = pass == 0 ? current_ : register_stage_;
  const auto row = low_storage_rows[pass];
  const auto terms = stage_terms(row);
  const auto n = problem_.dims.n_omega;
  const auto gaps = fine_spectrum_size() - n;
  cartesian_for(
      "solprop_native.low_storage.modal_input", execution_space_, problem_.n_columns(),
      n, KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        Complex value = source(column, index);
        if (terms.add_noise) {
          const Complex phase_at = terms.noise_phase(row, column, index);
          value += terms.noise(column, index) *
                   Complex(Kokkos::cos(phase_at.imag()), Kokkos::sin(phase_at.imag()));
        }
        const double analytic_scale =
            terms.analytic && index > 0 && 2 * terms.crop(index) < terms.n_to ? 2.0
                                                                              : 1.0;
        terms.fine(column, terms.crop(index)) =
            analytic_scale * terms.inverse_n_to * value;
        if (terms.general)
          terms.general_scaled(column, index) =
              value *
              (terms.general_pad * general_doubling(terms.crop(index), terms.n_to));
        for (std::size_t gap = index; gap < gaps; gap += n)
          terms.zero_gaps(column, gap, gap + 1);
      });
}

// --- solprop SEP 16 phase 3b: the Adams history resident in the engine ---------------
//
// solprop's HostAdamsMarch keeps the controller and hands every state-sized operation
// of its fused path to an engine (solprop ADR-043). These are that engine's passes: the
// history's rows, the trial's predictor, source, corrected field and defect norms, the
// adoption and transport of an accepted step and dense observation all stay in device
// memory; only the trial's scalars, its coefficient tables (by value, in AdamsWeights)
// and the norms cross. Each pass is the host engine's arithmetic (solprop's
// backends/multistep.py, HostHistory and AdamsKernels), checked against it in parity.

void GeneralWorkspace::adams_allocate(const std::size_t slots, const bool compensated,
                                      const bool interpolate) {
  require_open();
  if (slots == 0 || slots > adams_max_samples)
    throw std::invalid_argument("the Adams history needs 1 to 8 rows");
  if (compensated && problem_.separable_geometry())
    throw std::logic_error("a separable operator needs no compensation");
  const auto columns = problem_.n_columns();
  const auto n = problem_.dims.n_omega;
  if (compensated && adams_states_.extent(0) != slots) {
    adams_states_ = ComplexView3("solprop_native.adams.states", slots, columns, n);
    adams_rates_ = ComplexView3("solprop_native.adams.rates", slots, columns, n);
    adams_carried_ = ComplexView3("solprop_native.adams.carried", slots, columns, n);
    adams_lab_ = ComplexView2("solprop_native.adams.predicted", columns, n);
    adams_endpoint_ = ComplexView2("solprop_native.adams.endpoint", columns, n);
    adams_endpoint_rate_ =
        ComplexView2("solprop_native.adams.endpoint_rate", columns, n);
    adams_rate_weights_ = Kokkos::View<double*, Layout, MemorySpace>(
        "solprop_native.adams.rate_weights", problem_.linear.n_phi_terms);
  }
  // Dense observations are opt-in. Keeping their slopes and saved fields on a run
  // that lands on its planes costs (slots + 4) whole fields without reading them.
  if (interpolate) {
    if (adams_slopes_.extent(0) != slots) {
      adams_slopes_ =
          ComplexView3("solprop_native.adams.dense_slopes", slots, columns, n);
      adams_held_ = ComplexView2("solprop_native.adams.held_state", columns, n);
      adams_dense_start_ = ComplexView2("solprop_native.adams.dense_start", columns, n);
      adams_dense_end_ = ComplexView2("solprop_native.adams.dense_end", columns, n);
      adams_dense_exponent_ =
          ComplexView2("solprop_native.adams.dense_exponent", columns, n);
    }
    if (compensated && adams_dense_terms_.extent(0) != slots)
      adams_dense_terms_ =
          ComplexView3("solprop_native.adams.dense_terms", slots, columns, n);
  } else {
    adams_slopes_ = {};
    adams_held_ = {};
    adams_dense_start_ = {};
    adams_dense_end_ = {};
    adams_dense_exponent_ = {};
    adams_dense_terms_ = {};
    adams_dense_count_ = 0;
  }
  // Beside a coupled flow the trial's block functions, and a frozen-residual flow's
  // weights per sample, live beside the history (solprop SEP 16 phase 6).
  if (flow_) {
    if (!flow_->forms_actions())
      throw std::logic_error("the resident Adams engine needs the flow's eigenbasis "
                             "and route");
    // A bin off the route takes the augmented exponential: a trial reads one
    // section, so the scratch is one row per active bin, whatever the sections. A
    // device's carry also keeps a bin's channel vectors there, in blocks 3 and 4
    // (solprop SEP 47 phase 4), which `augmented_blocks` holds for any history.
    const auto rows = flow_->n_fallback() == 0 ? 0 : flow_->n_active();
    const auto blocks = flow_functions::augmented_blocks(slots);
    if (rows == 0)
      adams_fallback_ = {};
    else if (adams_fallback_.extent(0) != rows || adams_fallback_.extent(1) != blocks)
      adams_fallback_ = Kokkos::View<Complex****, Layout, MemorySpace>(
          "solprop_native.adams.fallback_functions", rows, blocks,
          problem_.dims.n_modes, problem_.dims.n_modes);
    if (columns != problem_.dims.n_modes)
      throw std::logic_error("the resident Adams engine steps a coupled block of one "
                             "batch row");
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
    // A wider block's host passes hold its channel vectors in a row per bin; a
    // device's passes take a thread per (bin, channel) and hold none, at any width
    // (solprop SEP 47 phase 4).
    const std::size_t width = problem_.dims.n_modes > adams_max_channels
                                  ? (adams_max_samples + 1) * problem_.dims.n_modes
                                  : 0;
    if (adams_channels_.extent(1) != width)
      adams_channels_ = width == 0
                            ? ComplexView2{}
                            : ComplexView2("solprop_native.adams.channels", n, width);
#endif
    if (adams_block_.extent(0) != slots + 1)
      adams_block_ = Kokkos::View<Complex****, Layout, MemorySpace>(
          "solprop_native.adams.block", slots + 1, flow_->n_active(),
          problem_.dims.n_modes, problem_.dims.n_modes);
    if (compensated && flow_->n_terms() != 0 && adams_applied_.extent(0) != slots) {
      adams_applied_ = ComplexView3("solprop_native.adams.applied", slots, columns, n);
      adams_end_applied_ = ComplexView2("solprop_native.adams.end_applied", columns, n);
      adams_end_weights_ = Kokkos::View<double*, Layout, MemorySpace>(
          "solprop_native.adams.end_weights", flow_->n_terms());
    }
  }
  // The norm's scale per bin, written each trial; allocated here so that the history's
  // memory is held, and counted, from construction (solprop's adams_memory).
  if (adams_divisor_.extent(0) != n)
    adams_divisor_ =
        Kokkos::View<double*, Layout, MemorySpace>("solprop_native.adams.divisor", n);
  if (adams_norms_.extent(0) == 0)
    adams_norms_ = Kokkos::View<double*, Kokkos::SharedHostPinnedSpace>(
        "solprop_native.adams.norms", adams_max_norms);
  // The history's views may move: a graph that names them is dropped.
  adams_source_graph_.reset();
  // A trial's elementwise passes cost about ten stage entries an element, rising with
  // the moments formed, so they take a width of their own; the accepted step's rows
  // are copies.
  adams_execution_space_ = kernel_pool_.get(CpuOperation::adams, columns * n * slots);
  adams_copy_space_ = kernel_pool_.get(CpuOperation::copy, columns * n);
  if (adams_store_.extent(0) == slots)
    return;
  adams_store_ = ComplexView3("solprop_native.adams.store", slots, columns, n);
  adams_moments_ = ComplexView3("solprop_native.adams.moments", slots, columns, n);
  adams_source_ = ComplexView2("solprop_native.adams.source", columns, n);
  adams_propagator_ = ComplexView2("solprop_native.adams.propagator", columns, n);
  adams_exponent_ = ComplexView2("solprop_native.adams.exponent", columns, n);
  adams_saved_ = ComplexView2("solprop_native.adams.saved_state", columns, n);
}

void GeneralWorkspace::require_adams() const {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (adams_store_.extent(0) == 0)
    throw std::logic_error("the resident Adams history is not allocated");
}

namespace {

// The whole step's phase increment `sum_t dPhi_t Phi_t` at one element, in term order:
// strict (solprop SEP 25 section 4.2).
template <class Terms, class Weights>
SOLPROP_NATIVE_STRICT_FUNCTION KOKKOS_INLINE_FUNCTION Complex adams_phase_sum(
    const Terms& phi_terms, const Weights& phi_delta, const std::size_t offset,
    const std::size_t n_phi, const std::size_t column, const std::size_t index) {
  SOLPROP_NATIVE_STRICT_BLOCK
  double real = 0.0;
  double imag = 0.0;
  for (std::size_t term = 0; term < n_phi; ++term) {
    const double weight = phi_delta(offset + term);
    // Exact, as in `StageTerms::phase_sum`.
    if (weight == 0.0)
      continue;
    const Complex fixed = phi_terms(term, column, index);
    real = real + weight * fixed.real();
    imag = imag + weight * fixed.imag();
  }
  return {real, imag};
}

} // namespace

// The whole step's exponent from the uploaded weights, and its exponential: on a
// separable route the step's length times each element's operator, as the stage
// kernel forms the whole propagator; otherwise `sum_t dPhi_t Phi_t` in term order.
void GeneralWorkspace::adams_phase() {
  if (problem_.separable_geometry()) {
    const auto spatial = spatial_;
    const auto active = active_mask_;
    const auto delta = stage_phi_delta_;
    const auto exponent = adams_exponent_;
    const auto propagator = adams_propagator_;
    const auto n_pol = problem_.dims.n_pol_state;
    const auto n_stages = problem_.stepper.n_stages;
    launch::parallel_for(
        "solprop_native.adams.separable_phase",
        Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          const auto values = spatial.value(column / n_pol, index, active(index) != 0U);
          const Complex x = delta(n_stages) * values.linear;
          exponent(column, index) = x;
          propagator(column, index) = exponential(x);
        });
    return;
  }
  const auto phi_terms = phi_terms_;
  const auto phi_delta = stage_phi_delta_;
  const auto exponent = adams_exponent_;
  const auto propagator = adams_propagator_;
  const auto n_phi = problem_.linear.n_phi_terms;
  const auto offset = problem_.stepper.n_stages * n_phi;
  launch::parallel_for(
      "solprop_native.adams.phase",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      // The step's phase increment: strict (solprop SEP 25 section 4.2).
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index)
          SOLPROP_NATIVE_STRICT_FUNCTION {
            SOLPROP_NATIVE_STRICT_BLOCK
            const Complex x =
                adams_phase_sum(phi_terms, phi_delta, offset, n_phi, column, index);
            exponent(column, index) = x;
            propagator(column, index) = exponential(x);
          });
}

// The stage-zero source at an anchor-frame field held in current_, as adams_source
// evaluates it, into stages_(0).
void GeneralWorkspace::adams_evaluate_source(const TrialScalars& scalars) {
  if (general_)
    general_->set_weights(scalars.transform_weights);
  if (problem_.raman.enabled)
    prepare_raman_stage(0, scalars.raman_rates.data(),
                        problem_.raman.n_components > 1 ? scalars.raman_densities.data()
                                                        : nullptr);
  evaluate_stage_source(0, 0.0, false);
  stages_ready_ = true;
  pending_fsal_ = false;
  has_recycled_ = false;
}

// The same, with current_ restored from adams_saved_ afterwards.
void GeneralWorkspace::adams_evaluate(const TrialScalars& scalars) {
  adams_evaluate_source(scalars);
  graph::deep_copy(execution_space_, current_, adams_saved_);
}

void GeneralWorkspace::adams_start(const std::size_t slot,
                                   const TrialScalars& scalars) {
  require_adams();
  if (slot >= adams_store_.extent(0))
    throw std::out_of_range("Adams history row");
  require_trial_scalars(scalars);
  upload_trial(scalars);
  graph::deep_copy(execution_space_, adams_saved_, current_);
  {
    const FlowAside aside(flow_);
    adams_evaluate(scalars);
  }
  // A frozen-residual flow's generator at the sample applied to its state, on the far
  // side of a kink that restarted the history (solprop SEP 16 phase 6).
  if (adams_applied_.extent(0) != 0) {
    upload_weights(execution_space_, adams_end_weights_, scalars.flow_weights);
    apply_generator(execution_space_, problem_.dims.n_omega, problem_.dims.n_modes,
                    flow_->block_slots(), flow_->residual_terms(), adams_end_weights_,
                    current_,
                    Kokkos::subview(adams_applied_, slot, Kokkos::ALL, Kokkos::ALL));
  }
  const auto stages = stages_;
  const auto store = adams_store_;
  launch::parallel_for(
      "solprop_native.adams.start",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        store(slot, column, index) = stages(0, column, index);
      });
}

void GeneralWorkspace::adams_scale(const double rtol, const double atol,
                                   const std::vector<double>& weights) {
  require_open();
  if (!weights.empty())
    require_extent(weights.size(), problem_.dims.n_omega, "Adams scale weights");
  adams_rtol_ = rtol;
  adams_atol_ = atol;
  adams_scale_weights_ = RealView("solprop_native.adams.scale_weights", weights.size());
  if (!weights.empty())
    copy_vector(execution_space_, adams_scale_weights_, weights);
  adams_magnitude_sum_ =
      Kokkos::View<double, Layout, MemorySpace>("solprop_native.adams.magnitude_sum");
  adams_device_scale_ = true;
}

// A trial's divisor: the host's when one is given, else formed on the device from the
// accepted state, as solprop's PIController.scale forms it from the same magnitude.
void GeneralWorkspace::adams_upload_divisor(const std::vector<double>& divisor) {
  if (adams_divisor_.extent(0) != problem_.dims.n_omega)
    adams_divisor_ = Kokkos::View<double*, Layout, MemorySpace>(
        "solprop_native.adams.divisor", problem_.dims.n_omega);
  if (!divisor.empty() || !adams_device_scale_) {
    require_extent(divisor.size(), problem_.dims.n_omega, "Adams norm divisor");
    copy_vector(execution_space_, adams_divisor_, divisor);
    return;
  }
  const auto current = current_;
  const auto in_norm = norm_mask_;
  const auto n = problem_.dims.n_omega;
  // A threaded reduction in whatever order measures fastest (solprop SEP 25).
  launch::parallel_reduce(
      "solprop_native.adams.magnitude",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0,
                                          problem_.n_columns() * n),
      KOKKOS_LAMBDA(const std::size_t item, double& update) {
        const auto index = item % n;
        if (in_norm(index) != 0)
          update += squared_abs(current(item / n, index));
      },
      adams_magnitude_sum_);
  const auto sum = adams_magnitude_sum_;
  const auto weights = adams_scale_weights_;
  const bool weighted = weights.extent(0) != 0;
  const double rtol = adams_rtol_;
  const double atol = adams_atol_;
  const auto scale = adams_divisor_;
  launch::parallel_for(
      "solprop_native.adams.divisor",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n),
      KOKKOS_LAMBDA(const std::size_t index) {
        const double magnitude = Kokkos::sqrt(sum());
        scale(index) = weighted ? rtol * weights(index) * magnitude + atol
                                : rtol * magnitude + atol;
      });
}

double GeneralWorkspace::adams_magnitude() {
  require_adams();
  const auto current = current_;
  const auto in_norm = norm_mask_;
  const auto n = problem_.dims.n_omega;
  double total = 0.0;
  launch::parallel_reduce(
      "solprop_native.adams.magnitude",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0,
                                          problem_.n_columns() * n),
      KOKKOS_LAMBDA(const std::size_t item, double& update) {
        const auto index = item % n;
        if (in_norm(index) != 0)
          update += squared_abs(current(item / n, index));
      },
      total);
  return Kokkos::sqrt(total);
}

namespace {

/** The squared scaled norms of up to three defects, reduced in one traversal. */
struct AdamsDefectNorms {
  using value_type = double[];
  using size_type = std::size_t;
  size_type value_count;
  Kokkos::View<Complex***, Layout, MemorySpace> store;
  Kokkos::View<Complex**, Layout, MemorySpace> source;
  Kokkos::View<Complex**, Layout, MemorySpace> propagator;
  Kokkos::View<unsigned char*, Layout, MemorySpace> in_norm;
  Kokkos::View<double*, Layout, MemorySpace> divisor;
  AdamsWeights weights;
  double dz;
  std::size_t n;

  // A defect is a divided difference of the samples: strict (solprop SEP 25 4.2).
  SOLPROP_NATIVE_STRICT_FUNCTION
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t item,
                                         value_type update) const {
    SOLPROP_NATIVE_STRICT_BLOCK
    const auto column = item / n;
    const auto index = item % n;
    if (in_norm(index) == 0)
      return;
    const Complex u = propagator(column, index);
    const double scale = divisor(index);
    for (size_type row = 0; row < value_count; ++row) {
      Complex defect = strict_scale(weights.rows[row][0], source(column, index));
      for (std::size_t t = 1; t < weights.width; ++t)
        defect = strict_accumulate(defect, weights.rows[row][t],
                                   store(weights.slot[t - 1], column, index));
      const Complex scaled =
          strict_divide(strict_mul(u, strict_scale(dz, defect)), scale);
      update[row] += squared_abs(scaled);
    }
  }
  KOKKOS_INLINE_FUNCTION void join(value_type destination,
                                   const value_type source_values) const {
    for (size_type row = 0; row < value_count; ++row)
      destination[row] += source_values[row];
  }
  KOKKOS_INLINE_FUNCTION void init(value_type values) const {
    for (size_type row = 0; row < value_count; ++row)
      values[row] = 0.0;
  }
};

} // namespace

std::vector<double>
GeneralWorkspace::adams_fused_trial(const double dz, const AdamsWeights& weights,
                                    const std::vector<double>& divisor,
                                    const TrialScalars& scalars) {
  require_adams();
  if (flow_)
    throw std::logic_error("the Adams source is evaluated without a coupled flow");
  if (weights.n_rows == 0 || weights.n_rows > 3 || weights.width == 0 ||
      weights.width > adams_max_samples || weights.order >= weights.width)
    throw std::invalid_argument("Adams coefficient tables are out of range");
  for (std::size_t t = 0; t + 1 < weights.width; ++t)
    if (weights.slot[t] >= adams_store_.extent(0))
      throw std::out_of_range("Adams history row");
  require_trial_scalars(scalars);
  upload_trial(scalars);
  adams_upload_divisor(divisor);
  adams_phase();
  const auto columns = problem_.n_columns();
  const auto n = problem_.dims.n_omega;
  const auto policy = Range2(execution_space_, {0, 0}, {columns, n});
  const auto store = adams_store_;
  const auto current = current_;
  const auto predicted = adams_saved_;
  // The predictor in the anchor's frame, a_n + dz sum_t w_t g_t, is the field the
  // stage-zero source evaluates; the accepted state is swapped aside meanwhile.
  launch::parallel_for(
      "solprop_native.adams.predict", policy,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        Complex total{};
        for (std::size_t t = 0; t < weights.order; ++t)
          total += weights.predictor[t] * store(weights.slot[t], column, index);
        predicted(column, index) = current(column, index) + dz * total;
      });
  std::swap(current_, adams_saved_);
  adams_evaluate_source(scalars);
  std::swap(current_, adams_saved_);
  const auto stages = stages_;
  const auto source = adams_source_;
  const auto propagator = adams_propagator_;
  const auto advanced = advanced_;
  launch::parallel_for(
      "solprop_native.adams.correct", policy,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        const Complex g = stages(0, column, index);
        source(column, index) = g;
        Complex total = weights.corrector[0] * g;
        for (std::size_t t = 1; t < weights.width; ++t)
          total += weights.corrector[t] * store(weights.slot[t - 1], column, index);
        advanced(column, index) =
            propagator(column, index) * (current(column, index) + dz * total);
      });
  const AdamsDefectNorms reducer{.value_count = weights.n_rows,
                                 .store = store,
                                 .source = source,
                                 .propagator = propagator,
                                 .in_norm = norm_mask_,
                                 .divisor = adams_divisor_,
                                 .weights = weights,
                                 .dz = dz,
                                 .n = n};
  return adams_norms(
      "solprop_native.adams.defect_norms",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, columns * n), reducer,
      weights.n_rows);
}

void GeneralWorkspace::adams_fused_accept(const double dz, const AdamsWeights& keep,
                                          const std::size_t free,
                                          const AdamsWeights& dense) {
  require_adams();
  if (!adams_trial_ready_)
    throw std::logic_error("no Adams trial is held");
  if (free >= adams_store_.extent(0))
    throw std::out_of_range("Adams history row");
  if (dense.width > 0 && adams_slopes_.extent(0) == 0)
    throw std::logic_error("Adams dense observation storage is not allocated");
  adams_trial_ready_ = false;
  const auto columns = problem_.n_columns();
  const auto n = problem_.dims.n_omega;
  const auto policy = Range2(execution_space_, {0, 0}, {columns, n});
  const auto store = adams_store_;
  const auto source = adams_source_;
  const auto propagator = adams_propagator_;
  const auto slopes = adams_slopes_;
  // The step's own interpolant, in its anchor's frame, before the transport moves the
  // rows: the slopes (source, newest, ...) and the state it starts from.
  adams_dense_count_ = dense.width;
  if (dense.width > 0) {
    graph::deep_copy(execution_space_, adams_dense_start_, current_);
    graph::deep_copy(execution_space_, adams_dense_exponent_, adams_exponent_);
    launch::parallel_for(
        "solprop_native.adams.dense_slopes", policy,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          slopes(0, column, index) = source(column, index);
          for (std::size_t t = 1; t < dense.width; ++t)
            slopes(t, column, index) = store(dense.slot[t - 1], column, index);
        });
  }
  finish(dz);
  stash_is_current_ = false;
  if (dense.width > 0) {
    const auto current = current_;
    const auto end = adams_dense_end_;
    launch::parallel_for(
        "solprop_native.adams.dense_end", policy,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          end(column, index) = current(column, index) / propagator(column, index);
        });
  }
  // The history re-anchored at the step's end, and the new derivative in row `free`.
  launch::parallel_for(
      "solprop_native.adams.transport", policy,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        const Complex u = propagator(column, index);
        for (std::size_t t = 0; t < keep.width; ++t)
          store(keep.slot[t], column, index) = store(keep.slot[t], column, index) * u;
        store(free, column, index) = u * source(column, index);
      });
  adams_accepted_boundary();
}

void GeneralWorkspace::adams_hold() {
  require_adams();
  if (adams_held_.empty())
    throw std::logic_error("Adams dense observation storage is not allocated");
  graph::deep_copy(execution_space_, adams_held_, current_);
}

void GeneralWorkspace::adams_restore() {
  require_adams();
  if (adams_held_.empty())
    throw std::logic_error("Adams dense observation storage is not allocated");
  graph::deep_copy(execution_space_, current_, adams_held_);
  stash_is_current_ = false;
  execution_space_.fence("solprop_native Adams restore boundary");
}

// --- solprop SEP 29 G2: the resident history's rows across a resume ----------------
//
// The persistent rows are the samples (store), a compensated history's sampled states
// and rates, and a frozen-residual flow's applied generators G(z_j) u_j: everything a
// later trial reads that an earlier step wrote. The rest of the Adams views are a
// trial's scratch, rebuilt by the next trial, and the step's bookkeeping lives in
// solprop's march. The copies are lossless: the view's own LayoutRight order, element
// by element, with no arithmetic.

ComplexView3 GeneralWorkspace::adams_row_view(const std::string& name) const {
  require_adams();
  ComplexView3 view;
  if (name == "store")
    view = adams_store_;
  else if (name == "states")
    view = adams_states_;
  else if (name == "rates")
    view = adams_rates_;
  else if (name == "applied")
    view = adams_applied_;
  else
    throw std::invalid_argument("unknown Adams history row '" + name + "'");
  if (view.extent(0) == 0)
    throw std::logic_error("the Adams history row '" + name + "' is not allocated");
  return view;
}

std::vector<std::string> GeneralWorkspace::adams_row_names() const {
  require_adams();
  std::vector<std::string> names{"store"};
  if (adams_states_.extent(0) != 0)
    names.insert(names.end(), {"states", "rates"});
  if (adams_applied_.extent(0) != 0)
    names.emplace_back("applied");
  return names;
}

std::array<std::size_t, 3>
GeneralWorkspace::adams_row_extents(const std::string& name) const {
  const auto view = adams_row_view(name);
  return {view.extent(0), view.extent(1), view.extent(2)};
}

std::vector<Complex>
GeneralWorkspace::adams_rows_to_host(const std::string& name) const {
  std::vector<std::complex<double>> values(adams_row_view(name).size());
  adams_rows_to_host(name, values.data(), values.size());
  return {values.begin(), values.end()};
}

// One copy, straight into the caller's buffer (a NumPy array's storage at the seam):
// no intermediate host vector, so a kept save costs what the host engine's copy does.
void GeneralWorkspace::adams_rows_to_host(const std::string& name,
                                          std::complex<double>* destination,
                                          const std::size_t size) const {
  const auto view = adams_row_view(name);
  if (size != view.size())
    throw std::invalid_argument("the Adams history row '" + name +
                                "' has the wrong extent");
  execution_space_.fence("solprop_native Adams rows_to_host");
  crossed_.to_host += size * sizeof(Complex);
  if (crossing::takes_ring(size * sizeof(Complex))) {
    crossing_ring_.to_host(destination, view.data(), size * sizeof(Complex));
    return;
  }
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  const auto* source = host.data();
  for (std::size_t index = 0; index < size; ++index)
    destination[index] = {source[index].real(), source[index].imag()};
}

void GeneralWorkspace::adams_rows_from_host(const std::string& name,
                                            const std::vector<Complex>& values) {
  const auto view = adams_row_view(name);
  if (values.size() != view.size())
    throw std::invalid_argument("the Adams history row '" + name +
                                "' has the wrong extent");
  copy_vector(execution_space_, view, values);
  // A trial in flight and the last step's interpolant belong to the history replaced.
  adams_trial_ready_ = false;
  adams_dense_count_ = 0;
  execution_space_.fence("solprop_native Adams rows_from_host");
}

// solprop SEP 29 G1: the general z tables. A table is uploaded once, after the state
// is constructed and before its first trial, which is the construction's allocation.
void GeneralWorkspace::table_upload(const TableKind kind, std::vector<double> edges,
                                    const double lower, const double upper,
                                    const std::complex<double>* coefficients,
                                    const std::size_t n_intervals,
                                    const std::size_t n_terms,
                                    const std::size_t n_values) {
  require_open();
  const bool prefactor = kind == TableKind::prefactor;
  const bool flux = kind == TableKind::flux;
  if (flux
          ? !problem_.flux.table
          : (prefactor ? !problem_.linear.prefactor_table : !problem_.linear.phi_table))
    throw std::logic_error(std::string("this problem's ") +
                           (flux        ? "plane measure"
                            : prefactor ? "prefactor"
                                        : "Phi") +
                           " is not a z table");
  if (n_intervals == 0 || n_terms == 0 || edges.size() != n_intervals + 1)
    throw std::invalid_argument("a z table needs one more breakpoint than intervals "
                                "and at least one coefficient");
  if (flux && n_values != problem_.dims.n_omega)
    refuse("the plane measure's material table holds n_omega = ", problem_.dims.n_omega,
           " values");
  if (!flux && n_values != problem_.n_columns() * problem_.dims.n_omega)
    refuse("a z table's values must be the state's (", problem_.n_columns(), " x ",
           problem_.dims.n_omega, ")");
  for (std::size_t index = 0; index < edges.size(); ++index)
    if (!std::isfinite(edges[index]) ||
        (index > 0 && !(edges[index] > edges[index - 1])))
      throw std::invalid_argument("a z table's breakpoints must be finite and "
                                  "strictly increasing");
  if (!(lower <= edges.front() && upper >= edges.back()))
    throw std::invalid_argument("a z table's admitted span must contain its edges");
  auto& table = kind == TableKind::phi         ? phi_table_
                : kind == TableKind::prefactor ? prefactor_table_
                : flux                         ? flux_table_
                                               : rate_table_;
  table.coefficients = Kokkos::View<Complex***, Layout, MemorySpace>(
      kind == TableKind::phi         ? "solprop_native.linear.phi_table"
      : kind == TableKind::prefactor ? "solprop_native.response.prefactor_table"
      : flux                         ? "solprop_native.flux.material_table"
                                     : "solprop_native.linear.rate_table",
      n_intervals, n_terms, n_values);
  auto mirror = Kokkos::create_mirror_view(table.coefficients);
  auto* data = mirror.data();
  const std::size_t count = n_intervals * n_terms * n_values;
  for (std::size_t index = 0; index < count; ++index)
    data[index] =
        chebyshev::Coefficient<std::complex<double>>::load(coefficients[index]);
  graph::deep_copy(execution_space_, table.coefficients, mirror);
  table.edges = std::move(edges);
  table.lower = lower;
  table.upper = upper;
  execution_space_.fence("solprop_native z table upload");
}

// The tables' values at a trial's positions into their term slots: Phi's and the
// prefactor's, each a pass over its coefficients for every position at once.
void GeneralWorkspace::table_evaluate(const std::vector<std::size_t>& slots,
                                      const std::vector<double>& positions) {
  require_open();
  if (!phi_table_.present() && !prefactor_table_.present())
    throw std::logic_error("no z table has been uploaded");
  for (const auto slot : slots) {
    if ((phi_table_.present() && slot >= problem_.linear.n_phi_terms) ||
        (prefactor_table_.present() && slot >= problem_.linear.n_prefactor_terms))
      refuse<std::out_of_range>("z table slot ", slot,
                                " is not a term of this problem");
  }
  if (phi_table_.present())
    z_table::evaluate("solprop_native.linear.phi_table_values", execution_space_,
                      phi_table_, z_table::locate(phi_table_, slots, positions, false),
                      phi_terms_.data());
  if (prefactor_table_.present())
    z_table::evaluate("solprop_native.response.prefactor_table_values",
                      execution_space_, prefactor_table_,
                      z_table::locate(prefactor_table_, slots, positions, false),
                      prefactor_terms_.data());
}

// solprop SEP 34: the plane measure's material square at `z`, a radial varying
// medium's, into the flux coefficients the statistics read; nothing for any other.
void GeneralWorkspace::flux_at(const double z) {
  if (!flux_table_.present())
    return;
  z_table::evaluate("solprop_native.flux.material_values", execution_space_,
                    flux_table_, z_table::locate(flux_table_, {0}, {z}, false),
                    flux_.square.data());
}

// D(z) = dPhi/dz from the rate table into one Phi term slot, read on the side of an
// interior breakpoint `before` asks for (solprop's `phi_rate`).
void GeneralWorkspace::table_rate(const std::size_t slot, const double z,
                                  const bool before) {
  require_open();
  if (!rate_table_.present())
    throw std::logic_error("no rate table has been uploaded");
  if (slot >= problem_.linear.n_phi_terms)
    refuse<std::out_of_range>("z table slot ", slot, " is not a term of this problem");
  z_table::evaluate("solprop_native.linear.rate_table_values", execution_space_,
                    rate_table_, z_table::locate(rate_table_, {slot}, {z}, before),
                    phi_terms_.data());
}

std::vector<Complex>
GeneralWorkspace::table_terms_to_host(const TableKind kind,
                                      const std::size_t slot) const {
  require_open();
  const auto& terms = kind == TableKind::prefactor ? prefactor_terms_ : phi_terms_;
  if (slot >= terms.extent(0))
    refuse<std::out_of_range>("term slot ", slot, " is not a term of this problem");
  execution_space_.fence("solprop_native z table terms_to_host");
  const auto host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, Kokkos::subview(terms, slot, Kokkos::ALL, Kokkos::ALL));
  std::vector<Complex> values(host.size());
  for (std::size_t column = 0; column < host.extent(0); ++column)
    for (std::size_t index = 0; index < host.extent(1); ++index)
      values[column * host.extent(1) + index] = host(column, index);
  return values;
}

// The stage kernels' own phase sum (`StageTerms::phase_sum`) over the Phi terms for one
// row of weights, into a fresh host array: a diagnostic, outside any trial.
std::vector<Complex>
GeneralWorkspace::phase_sum_to_host(const std::vector<double>& weights) {
  require_open();
  if (problem_.separable_geometry())
    throw std::logic_error("a separable operator has no Phi terms");
  require_extent(weights.size(), problem_.linear.n_phi_terms, "phase weights");
  RealView row("solprop_native.diagnostic.phase_weights", weights.size());
  copy_vector(execution_space_, row, weights);
  ComplexView2 phase("solprop_native.diagnostic.phase", problem_.n_columns(),
                     problem_.dims.n_omega);
  auto terms = stage_terms(0);
  terms.phi_delta = row;
  launch::parallel_for(
      "solprop_native.diagnostic.phase_sum",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        phase(column, index) = terms.phase_sum(0, column, index);
      });
  execution_space_.fence("solprop_native phase_sum_to_host");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, phase);
  return {host.data(), host.data() + host.size()};
}

// The field at a fraction of the last accepted step: the step's interpolant in its
// anchor's frame -- the Adams corrector's (`weights.predictor` holds dz-scaled
// integrated weights over the slopes) -- with a bin the step's flow changes by more
// than an e-fold taking the anchor's value, carried to the observation by the uploaded
// trial's flow and masked into current_.
void GeneralWorkspace::adams_observe(const AdamsWeights& weights,
                                     const TrialScalars& scalars) {
  require_adams();
  if (adams_dense_count_ == 0)
    throw std::logic_error("no Adams step is held for dense observation");
  require_trial_scalars(scalars);
  upload_trial(scalars);
  adams_phase();
  const auto columns = problem_.n_columns();
  const auto n = problem_.dims.n_omega;
  const auto count = adams_dense_count_;
  const auto start = adams_dense_start_;
  const auto exponent = adams_dense_exponent_;
  const auto slopes = adams_slopes_;
  const auto carry = adams_propagator_;
  const auto current = current_;
  launch::parallel_for(
      "solprop_native.adams.observe", Range2(execution_space_, {0, 0}, {columns, n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        Complex total{};
        for (std::size_t t = 0; t < count; ++t)
          total += weights.predictor[t] * slopes(t, column, index);
        Complex value = start(column, index) + total;
        if (Kokkos::abs(exponent(column, index).real()) > 1.0)
          value = start(column, index);
        current(column, index) = carry(column, index) * value;
      });
  mask(current_);
  stash_is_current_ = false;
  execution_space_.fence("solprop_native Adams observation boundary");
}

// --- solprop SEP 16 phase 3b: the exponential formulation, resident
// -------------------
//
// The exponential formulation's passes (solprop ADR-041, ADR-043): untransported
// samples integrated exactly against the step's flow exp(x), x = Phi(z + h) - Phi(z).
// The moments M_k = k! phi_{k+1}(x) are formed once a trial per element by the
// reference's algorithm (solprop backends/exponential.py, phi_functions), then every
// formula of the trial is
// applied in the monomial basis of its interpolant; the defects use the factored form
// (a divided difference of the samples times one moment combination).

namespace {

/** k! for k < 2 adams_max_samples, the moments' scale. */
KOKKOS_INLINE_FUNCTION double adams_factorial(const std::size_t k) {
  double value = 1.0;
  for (std::size_t i = 2; i <= k; ++i)
    value *= static_cast<double>(i);
  return value;
}

/**
 * The phi-functions a trial's moments take: at x, or under a carrier frame at
 * x - sigma with that argument's own exponential (solprop SEP 35 section 4.1). The
 * flow stays exp(x), which the caller holds.
 */
KOKKOS_INLINE_FUNCTION Complex adams_moment_argument(const ExponentialWeights& weights,
                                                     const Complex x) {
  return weights.shifted ? x - Complex(0.0, weights.shift) : x;
}

/** `adams_phi` at `adams_moment_argument`, with the flow `exp_x`; `Fixed`, when not
    zero, is `order` known at compile time (`adams_phi_fixed`). */
template <std::size_t Fixed = 0>
KOKKOS_INLINE_FUNCTION void adams_moment_phi(const ExponentialWeights& weights,
                                             const Complex x, const Complex exp_x,
                                             const std::size_t order, Complex* out) {
  const Complex argument = adams_moment_argument(weights, x);
  const double size = Kokkos::abs(argument);
  if (!weights.shifted) {
    adams_phi_fixed<Fixed>(argument, size, exp_x, order, out);
    return;
  }
  // exp(x - sigma) = exp(x) exp(-sigma): one product with the trial's exp(-sigma)
  // rather than a second exponential per element (a few ulp, inside the moments'
  // share of the rounding budget).
  adams_phi_fixed<Fixed>(
      argument, size,
      strict_mul(exp_x, Complex(weights.unshift_re, weights.unshift_im)), order, out);
}

/**
 * History row `t` of a trial (`weights.slot[t]`) at one element, rotated by a carrier
 * frame's factor exp(sigma (1 - theta_t)) as it is read (solprop SEP 35 section 6.2):
 * the store keeps the untransported samples.
 */
template <class Store>
KOKKOS_INLINE_FUNCTION Complex adams_history(const Store& store,
                                             const ExponentialWeights& weights,
                                             const std::size_t t,
                                             const std::size_t column,
                                             const std::size_t index) {
  const Complex value = store(weights.slot[t], column, index);
  if (!weights.shifted)
    return value;
  return strict_mul(Complex(weights.factor_re[t], weights.factor_im[t]), value);
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
/**
 * A device element's moments, written to `moments`, and its prediction
 * `exp(x) u + dz sum_k M_k sum_j c_jk F_j`, as the elementwise trial's predict pass
 * forms them, with `formed` phi-functions. `Fixed`, when not zero, is `formed` known at
 * compile time: the loops over the phi-functions unroll and the thread holds them in
 * registers, where an indexed array lives in private (scratch) memory, which on an
 * AMD Instinct MI325X made this the costliest kernel of a small trial.
 */
template <std::size_t Fixed, class Moments, class Store>
KOKKOS_INLINE_FUNCTION Complex adams_device_prediction(
    const ExponentialWeights& weights, const std::size_t formed, const Complex x,
    const Complex propagate, const Complex accepted, const double dz,
    const Moments& moments, const Store& store, const std::size_t column,
    const std::size_t index) {
  constexpr std::size_t bound = Fixed != 0 ? Fixed : 2 * adams_max_samples;
  Complex phi[bound + 1];
  adams_moment_phi<Fixed>(weights, x, propagate, formed, phi);
  for (std::size_t k = 0; k < bound; ++k) {
    if (k >= weights.count)
      break;
    moments(k, column, index) = adams_factorial(k) * phi[k + 1];
  }
  Complex total{};
  for (std::size_t k = 0; k < bound; ++k) {
    if (k >= weights.order)
      break;
    Complex combined{};
    for (std::size_t t = 0; t < weights.order; ++t)
      combined +=
          weights.predictor[t][k] * adams_history(store, weights, t, column, index);
    const Complex moment = adams_factorial(k) * phi[k + 1];
    total += moment * combined;
  }
  return propagate * accepted + dz * total;
}

/** `adams_device_prediction` at the compile-time bound equal to `formed`, or at a
    run-time one past the table's eight samples. */
template <class Moments, class Store>
KOKKOS_INLINE_FUNCTION Complex
adams_device_predict(const ExponentialWeights& weights, const std::size_t formed,
                     const Complex x, const Complex propagate, const Complex accepted,
                     const double dz, const Moments& moments, const Store& store,
                     const std::size_t column, const std::size_t index) {
  static_assert(adams_max_samples == 8, "one case per sample count");
  const auto predict = [&]<std::size_t Formed> {
    return adams_device_prediction<Formed>(weights, formed, x, propagate, accepted, dz,
                                           moments, store, column, index);
  };
  return dispatch_width<1, adams_max_samples>(
      formed, predict, [&] { return predict.template operator()<0>(); });
}
#endif

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
/** The channel vectors a host task of a coupled pass holds: up to `adams_max_samples +
    1` rows of a bin's channels. */
inline constexpr std::size_t adams_channel_stack =
    (adams_max_samples + 1) * adams_max_channels;

/**
 * Where a host task holds bin `index`'s channel vectors, rows of `modes` entries:
 * `stack` up to `adams_max_channels` channels, beyond it the bin's row of `rows`
 * (`adams_channels_`, solprop SEP 47 phase 4). Every pass reads only what it wrote.
 */
// A bin's row of `adams_channels_` is contiguous: the view's last extent is its
// fastest.
static_assert(std::is_same_v<Layout, Kokkos::LayoutRight>,
              "adams_channel_rows reads a bin's row as contiguous memory");

inline Complex*
adams_channel_rows(Complex* stack,
                   const Kokkos::View<Complex**, Layout, MemorySpace>& rows,
                   const std::size_t index) {
  return rows.extent(1) == 0 ? stack : &rows(index, 0);
}
#endif

// The defects' norms beside a coupled flow (solprop SEP 16 phase 6): per bin, the
// samples' channel vectors; off the block each channel's defect is the diagonal one,
// on it each moment combination is a matrix applied to the divided difference.
struct CoupledDefectNorms {
  using value_type = double[];
  using size_type = std::size_t;
  size_type value_count;
  Kokkos::View<Complex***, Layout, MemorySpace> store;
  Kokkos::View<Complex***, Layout, MemorySpace> moments;
  Kokkos::View<Complex**, Layout, MemorySpace> source;
  Kokkos::View<Complex***, Layout, MemorySpace> carried;
  Kokkos::View<Complex**, Layout, MemorySpace> endpoint;
  Kokkos::View<Complex****, Layout, MemorySpace> block;
  Kokkos::View<std::size_t*, Layout, MemorySpace> slots;
  Kokkos::View<unsigned char*, Layout, MemorySpace> in_norm;
  Kokkos::View<double*, Layout, MemorySpace> divisor;
  ExponentialWeights weights;
  double dz;
  std::size_t modes;
  // A host's channel vectors beyond `adams_max_channels` (`adams_channels_`).
  Kokkos::View<Complex**, Layout, MemorySpace> channels;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  SOLPROP_NATIVE_STRICT_FUNCTION
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t item,
                                         value_type update) const {
    SOLPROP_NATIVE_STRICT_BLOCK
    // An item is a (bin, channel) pair: a thread holds no channel vectors.
    const auto index = item / modes;
    const auto i = item % modes;
    if (in_norm(index) == 0)
      return;
    const double scale = divisor(index);
    const auto one = slots(index);
    // Channel c's divided difference for one defect.
    const auto difference = [&](const size_type row, const std::size_t c) {
      Complex total = strict_scale(weights.rows[row][0], source(c, index));
      for (std::size_t t = 1; t < weights.width; ++t)
        total = strict_accumulate(total, weights.rows[row][t],
                                  adams_history(store, weights, t - 1, c, index));
      return total;
    };
    for (size_type row = 0; row < value_count; ++row) {
      Complex defect{};
      if (one == 0) {
        Complex weight{};
        for (std::size_t k = 0; k < weights.count; ++k)
          weight = strict_accumulate(weight, weights.rows[row][weights.width + k],
                                     moments(k, i, index));
        defect = strict_scale(dz, strict_mul(difference(row, i), weight));
      } else {
        const auto bin = one - 1;
        Complex total{};
        for (std::size_t j = 0; j < modes; ++j) {
          Complex weight{};
          for (std::size_t k = 0; k < weights.count; ++k)
            weight = strict_accumulate(weight, weights.rows[row][weights.width + k],
                                       block(k + 1, bin, i, j));
          total = strict_add(total, strict_mul(weight, difference(row, j)));
        }
        defect = strict_scale(dz, total);
      }
      if (weights.compensated) {
        Complex compensation =
            strict_scale(weights.comp_rows[row][0], endpoint(i, index));
        for (std::size_t t = 0; t + 1 < weights.width; ++t)
          compensation = strict_accumulate(compensation, weights.comp_rows[row][t + 1],
                                           carried(t, i, index));
        defect = strict_accumulate(defect, dz, compensation);
      }
      update[row] += squared_abs(strict_divide(defect, scale));
    }
  }
#else
  // The defects and their compensation terms: strict (solprop SEP 25 section 4.2).
  SOLPROP_NATIVE_STRICT_FUNCTION
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t index,
                                         value_type update) const {
    SOLPROP_NATIVE_STRICT_BLOCK
    if (in_norm(index) == 0)
      return;
    const double scale = divisor(index);
    const auto one = slots(index);
    // The samples' channel vectors, row t at `values + t modes`, and the divided
    // difference after them.
    Complex stack[adams_channel_stack];
    Complex* const values = adams_channel_rows(stack, channels, index);
    Complex* const difference = values + adams_max_samples * modes;
    for (std::size_t i = 0; i < modes; ++i) {
      values[i] = source(i, index);
      for (std::size_t t = 1; t < weights.width; ++t)
        values[t * modes + i] = adams_history(store, weights, t - 1, i, index);
    }
    for (size_type row = 0; row < value_count; ++row) {
      for (std::size_t i = 0; i < modes; ++i) {
        Complex total = strict_scale(weights.rows[row][0], values[i]);
        for (std::size_t t = 1; t < weights.width; ++t)
          total = strict_accumulate(total, weights.rows[row][t], values[t * modes + i]);
        difference[i] = total;
      }
      for (std::size_t i = 0; i < modes; ++i) {
        Complex defect{};
        if (one == 0) {
          Complex weight{};
          for (std::size_t k = 0; k < weights.count; ++k)
            weight = strict_accumulate(weight, weights.rows[row][weights.width + k],
                                       moments(k, i, index));
          defect = strict_scale(dz, strict_mul(difference[i], weight));
        } else {
          const auto bin = one - 1;
          Complex total{};
          for (std::size_t j = 0; j < modes; ++j) {
            Complex weight{};
            for (std::size_t k = 0; k < weights.count; ++k)
              weight = strict_accumulate(weight, weights.rows[row][weights.width + k],
                                         block(k + 1, bin, i, j));
            total = strict_add(total, strict_mul(weight, difference[j]));
          }
          defect = strict_scale(dz, total);
        }
        if (weights.compensated) {
          Complex compensation =
              strict_scale(weights.comp_rows[row][0], endpoint(i, index));
          for (std::size_t t = 0; t + 1 < weights.width; ++t)
            compensation = strict_accumulate(
                compensation, weights.comp_rows[row][t + 1], carried(t, i, index));
          defect = strict_accumulate(defect, dz, compensation);
        }
        update[row] += squared_abs(strict_divide(defect, scale));
      }
    }
  }
#endif
  KOKKOS_INLINE_FUNCTION void join(value_type destination,
                                   const value_type source_values) const {
    for (size_type row = 0; row < value_count; ++row)
      destination[row] += source_values[row];
  }
  KOKKOS_INLINE_FUNCTION void init(value_type values) const {
    for (size_type row = 0; row < value_count; ++row)
      values[row] = 0.0;
  }
};

} // namespace

// D(z) = sum_t w_t Phi_t from the operator's fixed terms (solprop's
// translate.linop.rate_weights), into `out`.
void GeneralWorkspace::adams_rate(const std::vector<double>& rate,
                                  const ComplexView2& out) {
  require_extent(rate.size(), problem_.linear.n_phi_terms, "Adams rate weights");
  if (adams_rate_weights_.extent(0) != rate.size())
    adams_rate_weights_ = Kokkos::View<double*, Layout, MemorySpace>(
        "solprop_native.adams.rate_weights", rate.size());
  copy_vector(execution_space_, adams_rate_weights_, rate);
  const auto weights = adams_rate_weights_;
  const auto phi_terms = phi_terms_;
  const auto n_phi = problem_.linear.n_phi_terms;
  launch::parallel_for(
      "solprop_native.adams.rate",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        Complex value{};
        for (std::size_t term = 0; term < n_phi; ++term)
          value += weights(term) * phi_terms(term, column, index);
        out(column, index) = value;
      });
}

// The startup sample's state and rate, beside the source adams_start stored.
void GeneralWorkspace::adams_compensation_start(const std::size_t slot,
                                                const std::vector<double>& rate) {
  require_adams();
  if (adams_states_.extent(0) == 0)
    throw std::logic_error("the Adams history carries no compensation");
  if (slot >= adams_states_.extent(0))
    throw std::out_of_range("Adams history row");
  const auto rates = Kokkos::subview(adams_rates_, slot, Kokkos::ALL, Kokkos::ALL);
  adams_rate(rate, adams_endpoint_rate_);
  graph::deep_copy(execution_space_, rates, adams_endpoint_rate_);
  graph::deep_copy(execution_space_,
                   Kokkos::subview(adams_states_, slot, Kokkos::ALL, Kokkos::ALL),
                   current_);
}

// The exponential Adams march's compensation at one bin, each a single strict function
// (solprop SEP 25 section 4.2) called once per bin: GCC compiles a strict function
// with its own options and does not inline it into a contracting kernel, but inlines
// the strict helpers it calls, which share those options. One call per bin, rather
// than one per helper, keeps the protected arithmetic strict on every compiler at the
// cost of a call (and Clang, which inlines every level, is unchanged).

// The predictor's carried samples `e^{s_t x} (R_t - x / dz) u_t`, written to
// `carried`, and the prediction `value + dz sum_t c_t carried_t`. A span of one (the
// newest sample, at the step's start) takes the step's flow `flow = e^x` as it is:
// `strict_scale(1, x)` is x, so its exponential is the same bits. A host block with
// vector maths forms the same values on its lanes
// (`adams_lanes_predicted_compensation`).
// The strict compensation helpers keep their linkage: changing it could change how GCC
// inlines strict code (fp.hpp), which needs its own A/B.
// NOLINTBEGIN(misc-use-internal-linkage): strict helpers, linkage unchanged
template <class Rates, class States, class Carried>
SOLPROP_NATIVE_STRICT_FUNCTION KOKKOS_INLINE_FUNCTION Complex
adams_predicted_compensation(const ExponentialWeights& weights, const Complex x,
                             const Complex flow, const double inverse_dz,
                             const double dz, const Complex value, const Rates& rates,
                             const States& states, const Carried& carried,
                             const std::size_t column, const std::size_t index) {
  SOLPROP_NATIVE_STRICT_BLOCK
  for (std::size_t t = 0; t + 1 < weights.width; ++t) {
    const auto slot = weights.slot[t];
    Complex term = compensation_residual(rates(slot, column, index), x, inverse_dz,
                                         states(slot, column, index));
    const double span = weights.span[t];
    if (span == 1.0)
      term = strict_mul(flow, term);
    else if (span != 0.0)
      term = strict_mul(exponential(strict_scale(span, x)), term);
    carried(t, column, index) = term;
  }
  Complex compensation{};
  for (std::size_t t = 0; t < weights.order; ++t)
    compensation = strict_accumulate(compensation, weights.comp_predictor[t],
                                     carried(t, column, index));
  return strict_accumulate(value, dz, compensation);
}

// The corrector's endpoint rate `sum_t w_t Phi_t` and compensation `(R - x / dz) u`,
// written to `endpoint_rate` and `endpoint`, and the correction
// `value + dz (c_0 endpoint + sum_t c_{t+1} carried_t)`.
template <class Weights, class Terms, class Points, class Carried>
SOLPROP_NATIVE_STRICT_FUNCTION KOKKOS_INLINE_FUNCTION Complex
adams_corrected_compensation(const ExponentialWeights& weights, const std::size_t n_phi,
                             const Weights& rate_weights, const Terms& phi_terms,
                             const Complex x, const double inverse_dz,
                             const Complex lab, const double dz, const Complex value,
                             const Points& endpoint_rate, const Points& endpoint,
                             const Carried& carried, const std::size_t column,
                             const std::size_t index) {
  SOLPROP_NATIVE_STRICT_BLOCK
  Complex rate_here{};
  for (std::size_t term = 0; term < n_phi; ++term)
    rate_here = strict_accumulate(rate_here, rate_weights(term),
                                  phi_terms(term, column, index));
  endpoint_rate(column, index) = rate_here;
  const Complex compensated = compensation_residual(rate_here, x, inverse_dz, lab);
  endpoint(column, index) = compensated;
  Complex compensation = strict_scale(weights.comp_corrector[0], compensated);
  for (std::size_t t = 0; t < weights.order; ++t)
    compensation = strict_accumulate(compensation, weights.comp_corrector[t + 1],
                                     carried(t, column, index));
  return strict_accumulate(value, dz, compensation);
}

// Dense observation's compensation `value + sum_t dz w_t e^{(xi - theta_t) x} term_t`.
template <class Terms>
SOLPROP_NATIVE_STRICT_FUNCTION KOKKOS_INLINE_FUNCTION Complex
adams_observed_compensation(const ExponentialWeights& weights, const std::size_t count,
                            const double xi, const double dz, const Complex chord,
                            const Complex value, const Terms& terms,
                            const std::size_t column, const std::size_t index) {
  SOLPROP_NATIVE_STRICT_BLOCK
  Complex total = value;
  for (std::size_t t = 0; t < count; ++t)
    total = strict_add(
        total,
        strict_mul(
            strict_scale(dz * weights.dense_weight[t],
                         exponential(strict_scale(xi - weights.dense_theta[t], chord))),
            terms(t, column, index)));
  return total;
}

// NOLINTEND(misc-use-internal-linkage)
namespace {

// The elements a host task of the elementwise trial's passes takes in turn. Each pass
// runs its loops over the block, so that what one loop writes the next reads from
// cache, and a strict loop is one call a block on a compiler that keeps strict
// functions out of line; a device takes a thread an element.
constexpr std::size_t adams_block = build::device ? 1 : adams_lane_width;

// The host blocks run as lanes where the vector maths forms their exponentials in
// passes (`vmath::vectorised`); elsewhere (devices, the system libm on Arm) each
// element forms its own, and the lanes are not compiled.
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) &&                     \
    defined(SOLPROP_NATIVE_VECTOR_MATH_SLEEF)
#define SOLPROP_NATIVE_ADAMS_LANES 1
#endif

// The step's exponent and flow on the elements [first, last) of a column. The phase
// increment: strict (solprop SEP 25 section 4.2). A host block with vector maths
// forms both on its lanes instead (`adams_lanes_predict`).
template <class Terms, class Weights, class Points>
SOLPROP_NATIVE_STRICT_FUNCTION KOKKOS_INLINE_FUNCTION void
adams_phase_block(const Terms& phi_terms, const Weights& phi_delta,
                  const std::size_t offset, const std::size_t n_phi,
                  const std::size_t column, const std::size_t first,
                  const std::size_t last, const Points& exponent, const Points& flow) {
  SOLPROP_NATIVE_STRICT_BLOCK
  for (std::size_t index = first; index < last; ++index) {
    const Complex x =
        adams_phase_sum(phi_terms, phi_delta, offset, n_phi, column, index);
    exponent(column, index) = x;
    flow(column, index) = exponential(x);
  }
}

#ifdef SOLPROP_NATIVE_ADAMS_LANES

// The history rows a pass reads, rotated by a carrier frame's factors
// (`adams_history`). Strict.
template <class Store>
SOLPROP_NATIVE_STRICT_FUNCTION inline void
adams_lanes_history(AdamsLanes& lanes, const ExponentialWeights& weights,
                    const Store& store, const std::size_t rows,
                    const std::size_t column, const std::size_t first) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const std::size_t count = lanes.count;
  for (std::size_t t = 0; t < rows; ++t) {
    const Complex* row = &store(weights.slot[t], column, first);
    if (!weights.shifted) {
      for (std::size_t b = 0; b < count; ++b) {
        lanes.hr[t][b] = row[b].real();
        lanes.hi[t][b] = row[b].imag();
      }
      continue;
    }
    const double factor_r = weights.factor_re[t];
    const double factor_i = weights.factor_im[t];
    for (std::size_t b = 0; b < count; ++b) {
      const double rr = factor_r * row[b].real();
      const double ii = factor_i * row[b].imag();
      const double ri = factor_r * row[b].imag();
      const double ir = factor_i * row[b].real();
      lanes.hr[t][b] = rr - ii;
      lanes.hi[t][b] = ri + ir;
    }
  }
}

// The spans' arguments s_t x of a varying operator's carried samples, into `sr`/`si`
// (strict_scale), for the exponentials the caller forms over them.
SOLPROP_NATIVE_STRICT_FUNCTION
inline void adams_lanes_span(AdamsLanes& lanes, const double span,
                             const std::size_t t) {
  SOLPROP_NATIVE_STRICT_BLOCK
  for (std::size_t b = 0; b < lanes.count; ++b) {
    lanes.sr[t][b] = span * lanes.xr[b];
    lanes.si[t][b] = span * lanes.xi[b];
  }
}

// The predictor's carried samples `e^{s_t x} (R_t - x / dz) u_t` into `cr`/`ci`, and
// the prediction in `vr`/`vi` plus `dz sum_t c_t carried_t`, as
// `adams_predicted_compensation`. Strict.
template <class Rates, class States>
SOLPROP_NATIVE_STRICT_FUNCTION inline void
adams_lanes_predicted_compensation(AdamsLanes& lanes, const ExponentialWeights& weights,
                                   const double inverse_dz, const double dz,
                                   const Rates& rates, const States& states,
                                   const std::size_t column, const std::size_t first) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const std::size_t count = lanes.count;
  for (std::size_t t = 0; t + 1 < weights.width; ++t) {
    const auto slot = weights.slot[t];
    const Complex* rate = &rates(slot, column, first);
    const Complex* state = &states(slot, column, first);
    const double span = weights.span[t];
    for (std::size_t b = 0; b < count; ++b) {
      const double scaled_r = inverse_dz * lanes.xr[b];
      const double scaled_i = inverse_dz * lanes.xi[b];
      const double difference_r = rate[b].real() - scaled_r;
      const double difference_i = rate[b].imag() - scaled_i;
      const double rr = difference_r * state[b].real();
      const double ii = difference_i * state[b].imag();
      const double ri = difference_r * state[b].imag();
      const double ir = difference_i * state[b].real();
      lanes.cr[t][b] = rr - ii;
      lanes.ci[t][b] = ri + ir;
    }
    if (span == 0.0)
      continue;
    const double* factor_r = span == 1.0 ? lanes.fr : lanes.sr[t];
    const double* factor_i = span == 1.0 ? lanes.fi : lanes.si[t];
    for (std::size_t b = 0; b < count; ++b) {
      const double rr = factor_r[b] * lanes.cr[t][b];
      const double ii = factor_i[b] * lanes.ci[t][b];
      const double ri = factor_r[b] * lanes.ci[t][b];
      const double ir = factor_i[b] * lanes.cr[t][b];
      lanes.cr[t][b] = rr - ii;
      lanes.ci[t][b] = ri + ir;
    }
  }
  double total_r[AdamsLanes::B];
  double total_i[AdamsLanes::B];
  for (std::size_t b = 0; b < count; ++b) {
    total_r[b] = 0.0;
    total_i[b] = 0.0;
  }
  for (std::size_t t = 0; t < weights.order; ++t) {
    const double weight = weights.comp_predictor[t];
    for (std::size_t b = 0; b < count; ++b) {
      const double real = weight * lanes.cr[t][b];
      const double imag = weight * lanes.ci[t][b];
      total_r[b] = total_r[b] + real;
      total_i[b] = total_i[b] + imag;
    }
  }
  for (std::size_t b = 0; b < count; ++b) {
    const double real = dz * total_r[b];
    const double imag = dz * total_i[b];
    lanes.vr[b] = lanes.vr[b] + real;
    lanes.vi[b] = lanes.vi[b] + imag;
  }
}

// A varying operator's endpoint rate `sum_t w_t Phi_t` into `dr`/`di` and
// `endpoint_rate`, as `adams_corrected_compensation` forms it. Strict.
template <class Weights, class Terms, class Points>
SOLPROP_NATIVE_STRICT_FUNCTION inline void
adams_lanes_rate(AdamsLanes& lanes, const std::size_t n_phi,
                 const Weights& rate_weights, const Terms& phi_terms,
                 const Points& endpoint_rate, const std::size_t column,
                 const std::size_t first) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const std::size_t count = lanes.count;
  for (std::size_t b = 0; b < count; ++b) {
    lanes.dr[b] = 0.0;
    lanes.di[b] = 0.0;
  }
  for (std::size_t term = 0; term < n_phi; ++term) {
    const double weight = rate_weights(term);
    const Complex* fixed = &phi_terms(term, column, first);
    for (std::size_t b = 0; b < count; ++b) {
      const double real = weight * fixed[b].real();
      const double imag = weight * fixed[b].imag();
      lanes.dr[b] = lanes.dr[b] + real;
      lanes.di[b] = lanes.di[b] + imag;
    }
  }
  Complex* rate_out = &endpoint_rate(column, first);
  for (std::size_t b = 0; b < count; ++b)
    rate_out[b] = Complex(lanes.dr[b], lanes.di[b]);
}

// The corrector's compensation `(R - x / dz) u` at the endpoint rate `dr`/`di` into
// `nr`/`ni`, and the correction in `vr`/`vi` plus `dz (c_0 endpoint + sum_t c_{t+1}
// carried_t)`, as `adams_corrected_compensation`; `ur`/`ui` hold the lab-frame field.
// Strict.
SOLPROP_NATIVE_STRICT_FUNCTION
inline void adams_lanes_corrected_compensation(AdamsLanes& lanes,
                                               const ExponentialWeights& weights,
                                               const double inverse_dz,
                                               const double dz) {
  SOLPROP_NATIVE_STRICT_BLOCK
  constexpr std::size_t B = AdamsLanes::B;
  const std::size_t count = lanes.count;
  double total_r[B];
  double total_i[B];
  const double weight0 = weights.comp_corrector[0];
  for (std::size_t b = 0; b < count; ++b) {
    const double scaled_r = inverse_dz * lanes.xr[b];
    const double scaled_i = inverse_dz * lanes.xi[b];
    const double difference_r = lanes.dr[b] - scaled_r;
    const double difference_i = lanes.di[b] - scaled_i;
    const double rr = difference_r * lanes.ur[b];
    const double ii = difference_i * lanes.ui[b];
    const double ri = difference_r * lanes.ui[b];
    const double ir = difference_i * lanes.ur[b];
    lanes.nr[b] = rr - ii;
    lanes.ni[b] = ri + ir;
    total_r[b] = weight0 * lanes.nr[b];
    total_i[b] = weight0 * lanes.ni[b];
  }
  for (std::size_t t = 0; t < weights.order; ++t) {
    const double weight = weights.comp_corrector[t + 1];
    for (std::size_t b = 0; b < count; ++b) {
      const double real = weight * lanes.cr[t][b];
      const double imag = weight * lanes.ci[t][b];
      total_r[b] = total_r[b] + real;
      total_i[b] = total_i[b] + imag;
    }
  }
  for (std::size_t b = 0; b < count; ++b) {
    const double real = dz * total_r[b];
    const double imag = dz * total_i[b];
    lanes.vr[b] = lanes.vr[b] + real;
    lanes.vi[b] = lanes.vi[b] + imag;
  }
}

// The squared scaled norms of up to three factored defects of every lane into
// `defect`, as `adams_defect_block` forms them: `wr`/`wi` hold the new sample,
// `hr`/`hi` the history, `mr`/`mi` the moments, `nr`/`ni` the endpoint's and `cr`/`ci`
// the carried compensation, `scale` the divisors. Strict.
SOLPROP_NATIVE_STRICT_FUNCTION
inline void adams_lanes_defects(AdamsLanes& lanes, const ExponentialWeights& weights,
                                const std::size_t scored, const double dz) {
  SOLPROP_NATIVE_STRICT_BLOCK
  constexpr std::size_t B = AdamsLanes::B;
  const std::size_t count = lanes.count;
  for (std::size_t row = 0; row < scored; ++row) {
    double difference_r[B];
    double difference_i[B];
    double weight_r[B];
    double weight_i[B];
    const double weight0 = weights.rows[row][0];
    for (std::size_t b = 0; b < count; ++b) {
      difference_r[b] = weight0 * lanes.wr[b];
      difference_i[b] = weight0 * lanes.wi[b];
      weight_r[b] = 0.0;
      weight_i[b] = 0.0;
    }
    for (std::size_t t = 1; t < weights.width; ++t) {
      const double weight = weights.rows[row][t];
      for (std::size_t b = 0; b < count; ++b) {
        const double real = weight * lanes.hr[t - 1][b];
        const double imag = weight * lanes.hi[t - 1][b];
        difference_r[b] = difference_r[b] + real;
        difference_i[b] = difference_i[b] + imag;
      }
    }
    for (std::size_t k = 0; k < weights.count; ++k) {
      const double weight = weights.rows[row][weights.width + k];
      for (std::size_t b = 0; b < count; ++b) {
        const double real = weight * lanes.mr[k][b];
        const double imag = weight * lanes.mi[k][b];
        weight_r[b] = weight_r[b] + real;
        weight_i[b] = weight_i[b] + imag;
      }
    }
    double defect_r[B];
    double defect_i[B];
    for (std::size_t b = 0; b < count; ++b) {
      const double rr = difference_r[b] * weight_r[b];
      const double ii = difference_i[b] * weight_i[b];
      const double ri = difference_r[b] * weight_i[b];
      const double ir = difference_i[b] * weight_r[b];
      const double product_r = rr - ii;
      const double product_i = ri + ir;
      defect_r[b] = dz * product_r;
      defect_i[b] = dz * product_i;
    }
    if (weights.compensated) {
      double total_r[B];
      double total_i[B];
      const double first = weights.comp_rows[row][0];
      for (std::size_t b = 0; b < count; ++b) {
        total_r[b] = first * lanes.nr[b];
        total_i[b] = first * lanes.ni[b];
      }
      for (std::size_t t = 0; t + 1 < weights.width; ++t) {
        const double weight = weights.comp_rows[row][t + 1];
        for (std::size_t b = 0; b < count; ++b) {
          const double real = weight * lanes.cr[t][b];
          const double imag = weight * lanes.ci[t][b];
          total_r[b] = total_r[b] + real;
          total_i[b] = total_i[b] + imag;
        }
      }
      for (std::size_t b = 0; b < count; ++b) {
        const double real = dz * total_r[b];
        const double imag = dz * total_i[b];
        defect_r[b] = defect_r[b] + real;
        defect_i[b] = defect_i[b] + imag;
      }
    }
    for (std::size_t b = 0; b < count; ++b) {
      const double real = defect_r[b] / lanes.scale[b];
      const double imag = defect_i[b] / lanes.scale[b];
      const double rr = real * real;
      const double ii = imag * imag;
      lanes.defect[row][b] = rr + ii;
    }
  }
}
#endif

#ifdef SOLPROP_NATIVE_ADAMS_LANES
// The whole step's phase increment `sum_t dPhi_t Phi_t` of every lane into `xr`/`xi`,
// as `adams_phase_sum` forms it, and with `Rate` a varying operator's endpoint rate
// `sum_t w_t Phi_t` into `dr`/`di`, as `adams_corrected_compensation` forms it, in the
// same pass over the terms. Strict.
template <bool Rate, class Terms, class Weights, class RateWeights>
SOLPROP_NATIVE_STRICT_FUNCTION inline void
adams_lanes_phase(AdamsLanes& lanes, const Terms& phi_terms, const Weights& phi_delta,
                  const RateWeights& rate_weights, const std::size_t offset,
                  const std::size_t n_phi, const std::size_t column,
                  const std::size_t first) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const std::size_t count = lanes.count;
  for (std::size_t b = 0; b < count; ++b) {
    lanes.xr[b] = 0.0;
    lanes.xi[b] = 0.0;
    lanes.dr[b] = 0.0;
    lanes.di[b] = 0.0;
  }
  for (std::size_t term = 0; term < n_phi; ++term) {
    const double weight = phi_delta(offset + term);
    const Complex* fixed = &phi_terms(term, column, first);
    // A zero weight is skipped, exactly, as in `StageTerms::phase_sum`.
    if constexpr (Rate) {
      const double rate = rate_weights(term);
      if (weight == 0.0)
        for (std::size_t b = 0; b < count; ++b) {
          lanes.dr[b] = lanes.dr[b] + rate * fixed[b].real();
          lanes.di[b] = lanes.di[b] + rate * fixed[b].imag();
        }
      else
        for (std::size_t b = 0; b < count; ++b) {
          lanes.xr[b] = lanes.xr[b] + weight * fixed[b].real();
          lanes.xi[b] = lanes.xi[b] + weight * fixed[b].imag();
          lanes.dr[b] = lanes.dr[b] + rate * fixed[b].real();
          lanes.di[b] = lanes.di[b] + rate * fixed[b].imag();
        }
    } else {
      if (weight == 0.0)
        continue;
      for (std::size_t b = 0; b < count; ++b) {
        lanes.xr[b] = lanes.xr[b] + weight * fixed[b].real();
        lanes.xi[b] = lanes.xi[b] + weight * fixed[b].imag();
      }
    }
  }
}
#endif

#ifdef SOLPROP_NATIVE_ADAMS_LANES
/**
 * The predict pass on a host block's lanes, the elements [first, last) of a column
 * whose exponents are formed: the flow, the moments, the predictor's carried samples
 * and the prediction, as the element-by-element form writes them. Returns the block's
 * share of the accepted state's squared magnitude, summed in element order.
 */
template <bool Phase, class Terms, class Weights, class RateWeights, class Points,
          class Cube, class Mask>
inline double adams_lanes_predict(
    const Terms& phi_terms, const Weights& phi_delta, const RateWeights& rate_weights,
    const Points& endpoint_rate, const std::size_t offset, const std::size_t n_phi,
    const ExponentialWeights& weights, const std::size_t formed, const double dz,
    const double inverse_dz, const Points& exponent, const Points& flow,
    const Points& current, const Points& predicted, const Points& lab,
    const Cube& moments, const Cube& store, const Cube& carried, const Cube& states,
    const Cube& rates, const Mask& in_norm, const std::size_t column,
    const std::size_t first, const std::size_t last) {
  constexpr std::size_t B = AdamsLanes::B;
  AdamsLanes lanes;
  const std::size_t n = last - first;
  lanes.count = n;
  if constexpr (Phase) {
    // The corrector's endpoint rate reads the same terms: formed here, in the same
    // pass, and read back by the correct pass.
    if (weights.compensated) {
      adams_lanes_phase<true>(lanes, phi_terms, phi_delta, rate_weights, offset, n_phi,
                              column, first);
      Complex* rate = &endpoint_rate(column, first);
      for (std::size_t b = 0; b < n; ++b)
        rate[b] = Complex(lanes.dr[b], lanes.di[b]);
    } else {
      adams_lanes_phase<false>(lanes, phi_terms, phi_delta, rate_weights, offset, n_phi,
                               column, first);
    }
    Complex* x = &exponent(column, first);
    for (std::size_t b = 0; b < n; ++b)
      x[b] = Complex(lanes.xr[b], lanes.xi[b]);
  } else {
    const Complex* x = &exponent(column, first);
    for (std::size_t b = 0; b < n; ++b) {
      lanes.xr[b] = x[b].real();
      lanes.xi[b] = x[b].imag();
    }
  }
  // Formed before the vector maths writes them: clang-tidy's analyzer does not follow
  // its stores (a few stores a block).
  for (std::size_t b = 0; b < n; ++b) {
    lanes.fr[b] = 0.0;
    lanes.fi[b] = 0.0;
  }
  vmath::exp_complex(lanes.xr, lanes.xi, lanes.fr, lanes.fi, n);
  Complex* flow_out = &flow(column, first);
  for (std::size_t b = 0; b < n; ++b)
    flow_out[b] = Complex(lanes.fr[b], lanes.fi[b]);
  adams_lanes_classify(lanes, weights);
  vmath::exp_complex(lanes.yr, lanes.yi, lanes.er, lanes.ei, n);
  if (lanes.any_large)
    adams_lanes_large_exponential(lanes, weights);
  adams_lanes_phi(lanes, formed);
  if (weights.compensated)
    for (std::size_t t = 0; t + 1 < weights.width; ++t) {
      const double span = weights.span[t];
      if (span == 1.0 || span == 0.0)
        continue;
      adams_lanes_span(lanes, span, t);
      vmath::exp_complex(lanes.sr[t], lanes.si[t], lanes.sr[t], lanes.si[t], n);
    }
  // The moments M_k = k! phi_{k+1}.
  const std::size_t moment_rows = formed;
  for (std::size_t k = 0; k < moment_rows; ++k) {
    const double factorial = adams_factorial(k);
    for (std::size_t b = 0; b < n; ++b) {
      lanes.mr[k][b] = factorial * lanes.pr[k + 1][b];
      lanes.mi[k][b] = factorial * lanes.pi[k + 1][b];
    }
  }
  for (std::size_t k = 0; k < weights.count; ++k) {
    Complex* out = &moments(k, column, first);
    for (std::size_t b = 0; b < n; ++b)
      out[b] = Complex(lanes.mr[k][b], lanes.mi[k][b]);
  }
  adams_lanes_history(lanes, weights, store, weights.order, column, first);
  // The prediction exp(x) u + dz sum_k M_k sum_j c_jk F_j.
  const Complex* accepted = &current(column, first);
  double total_r[B];
  double total_i[B];
  for (std::size_t b = 0; b < n; ++b) {
    lanes.ur[b] = accepted[b].real();
    lanes.ui[b] = accepted[b].imag();
    total_r[b] = 0.0;
    total_i[b] = 0.0;
  }
  for (std::size_t k = 0; k < weights.order; ++k) {
    double combined_r[B];
    double combined_i[B];
    for (std::size_t b = 0; b < n; ++b) {
      combined_r[b] = 0.0;
      combined_i[b] = 0.0;
    }
    for (std::size_t t = 0; t < weights.order; ++t) {
      const double weight = weights.predictor[t][k];
      for (std::size_t b = 0; b < n; ++b) {
        combined_r[b] += weight * lanes.hr[t][b];
        combined_i[b] += weight * lanes.hi[t][b];
      }
    }
    for (std::size_t b = 0; b < n; ++b) {
      total_r[b] += lanes.mr[k][b] * combined_r[b] - lanes.mi[k][b] * combined_i[b];
      total_i[b] += lanes.mr[k][b] * combined_i[b] + lanes.mi[k][b] * combined_r[b];
    }
  }
  for (std::size_t b = 0; b < n; ++b) {
    lanes.vr[b] =
        lanes.fr[b] * lanes.ur[b] - lanes.fi[b] * lanes.ui[b] + dz * total_r[b];
    lanes.vi[b] =
        lanes.fr[b] * lanes.ui[b] + lanes.fi[b] * lanes.ur[b] + dz * total_i[b];
  }
  double squares = 0.0;
  for (std::size_t b = 0; b < n; ++b)
    if (in_norm(first + b) != 0)
      squares += lanes.ur[b] * lanes.ur[b] + lanes.ui[b] * lanes.ui[b];
  if (weights.compensated) {
    adams_lanes_predicted_compensation(lanes, weights, inverse_dz, dz, rates, states,
                                       column, first);
    for (std::size_t t = 0; t + 1 < weights.width; ++t) {
      Complex* out = &carried(t, column, first);
      for (std::size_t b = 0; b < n; ++b)
        out[b] = Complex(lanes.cr[t][b], lanes.ci[t][b]);
    }
    Complex* lab_out = &lab(column, first);
    for (std::size_t b = 0; b < n; ++b)
      lab_out[b] = Complex(lanes.vr[b], lanes.vi[b]);
  }
  Complex* out = &predicted(column, first);
  for (std::size_t b = 0; b < n; ++b)
    out[b] = Complex(lanes.vr[b], lanes.vi[b]);
  return squares;
}
#endif

// The squared scaled norms of up to three factored defects on the in-band elements of
// [first, last) of a column, added to `update`: a divided difference of the samples
// times one moment combination, and a varying operator's compensation term, over the
// divisor `scale[index - first]`. The defects and their compensation: strict (solprop
// SEP 25 section 4.2).
template <class Store, class Moments, class Points, class Carried, class Mask>
SOLPROP_NATIVE_STRICT_FUNCTION KOKKOS_INLINE_FUNCTION void
adams_defect_block(const ExponentialWeights& weights, const std::size_t rows,
                   const Store& store, const Moments& moments, const Points& source,
                   const Carried& carried, const Points& endpoint, const Mask& in_norm,
                   const double dz, const double* scale, const std::size_t column,
                   const std::size_t first, const std::size_t last, double* update) {
  SOLPROP_NATIVE_STRICT_BLOCK
  // The block's sums, added to the reduction's once. A trial scores at most three
  // defects (adams_exponential_trial refuses more); the bound says so here too.
  constexpr std::size_t max_rows = 3;
  const std::size_t scored = rows < max_rows ? rows : max_rows;
  double sums[max_rows] = {0.0, 0.0, 0.0};
  for (std::size_t index = first; index < last; ++index) {
    if (in_norm(index) == 0)
      continue;
    for (std::size_t row = 0; row < scored; ++row) {
      Complex difference = strict_scale(weights.rows[row][0], source(column, index));
      for (std::size_t t = 1; t < weights.width; ++t)
        difference =
            strict_accumulate(difference, weights.rows[row][t],
                              adams_history(store, weights, t - 1, column, index));
      Complex weight{};
      for (std::size_t k = 0; k < weights.count; ++k)
        weight = strict_accumulate(weight, weights.rows[row][weights.width + k],
                                   moments(k, column, index));
      Complex defect = strict_scale(dz, strict_mul(difference, weight));
      if (weights.compensated) {
        Complex compensation =
            strict_scale(weights.comp_rows[row][0], endpoint(column, index));
        for (std::size_t t = 0; t + 1 < weights.width; ++t)
          compensation = strict_accumulate(compensation, weights.comp_rows[row][t + 1],
                                           carried(t, column, index));
        defect = strict_accumulate(defect, dz, compensation);
      }
      sums[row] += squared_abs(strict_divide(defect, scale[index - first]));
    }
  }
  for (std::size_t row = 0; row < scored; ++row)
    update[row] += sums[row];
}

/**
 * The elementwise trial's last pass (solprop SEP 16 phase 3b), a block of a column a
 * task: the new sample and the corrected field `exp(x) u + dz sum_k M_k sum_j c_jk
 * F_j` with a varying operator's compensation, then the squared scaled norms of up to
 * three factored defects, reduced in the same traversal. solprop's own controller's
 * divisor is formed here from the accepted state's magnitude, as its
 * PIController.scale forms it; any other controller's is read from `divisor`.
 */
struct ExponentialCorrection {
  using value_type = double[];
  using size_type = std::size_t;
  size_type value_count;
  Kokkos::View<Complex***, Layout, MemorySpace> store, moments, stages, carried;
  Kokkos::View<Complex***, Layout, MemorySpace> phi_terms;
  Kokkos::View<Complex**, Layout, MemorySpace> source, flow, current, advanced,
      exponent;
  Kokkos::View<Complex**, Layout, MemorySpace> lab, endpoint, endpoint_rate;
  Kokkos::View<double*, Layout, MemorySpace> rate_weights;
  Kokkos::View<unsigned char*, Layout, MemorySpace> in_norm;
  Kokkos::View<double*, Layout, MemorySpace> divisor, scale_weights;
  Kokkos::View<double, Layout, MemorySpace> magnitude_sum;
  ExponentialWeights weights;
  double dz, inverse_dz, rtol, atol;
  std::size_t n, blocks, n_phi;
  bool device_scale;
  // The host lanes' predict pass formed the endpoint rate (off a separable route).
  bool rate_formed;

#ifdef SOLPROP_NATIVE_ADAMS_LANES
  // The pass on a host block's lanes (`AdamsLanes`), as the element-by-element form
  // below writes it; the norms' sums are taken in element order.
  void lanes_block(const std::size_t column, const std::size_t first,
                   const std::size_t last, const double magnitude,
                   value_type update) const {
    constexpr std::size_t B = AdamsLanes::B;
    AdamsLanes lanes;
    const std::size_t count = last - first;
    lanes.count = count;
    const Complex* sample = &stages(0, column, first);
    Complex* sample_out = &source(column, first);
    for (std::size_t b = 0; b < count; ++b) {
      lanes.wr[b] = sample[b].real();
      lanes.wi[b] = sample[b].imag();
      sample_out[b] = sample[b];
    }
    adams_lanes_history(lanes, weights, store, weights.width - 1, column, first);
    for (std::size_t k = 0; k < weights.count; ++k) {
      const Complex* moment = &moments(k, column, first);
      for (std::size_t b = 0; b < count; ++b) {
        lanes.mr[k][b] = moment[b].real();
        lanes.mi[k][b] = moment[b].imag();
      }
    }
    double total_r[B];
    double total_i[B];
    for (std::size_t b = 0; b < count; ++b) {
      total_r[b] = 0.0;
      total_i[b] = 0.0;
    }
    for (std::size_t k = 0; k < weights.count; ++k) {
      double combined_r[B];
      double combined_i[B];
      const double weight0 = weights.corrector[0][k];
      for (std::size_t b = 0; b < count; ++b) {
        combined_r[b] = weight0 * lanes.wr[b];
        combined_i[b] = weight0 * lanes.wi[b];
      }
      for (std::size_t t = 1; t < weights.width; ++t) {
        const double weight = weights.corrector[t][k];
        for (std::size_t b = 0; b < count; ++b) {
          combined_r[b] += weight * lanes.hr[t - 1][b];
          combined_i[b] += weight * lanes.hi[t - 1][b];
        }
      }
      for (std::size_t b = 0; b < count; ++b) {
        total_r[b] += lanes.mr[k][b] * combined_r[b] - lanes.mi[k][b] * combined_i[b];
        total_i[b] += lanes.mr[k][b] * combined_i[b] + lanes.mi[k][b] * combined_r[b];
      }
    }
    const Complex* propagate = &flow(column, first);
    const Complex* accepted = &current(column, first);
    for (std::size_t b = 0; b < count; ++b) {
      const Complex value =
          propagate[b] * accepted[b] + dz * Complex(total_r[b], total_i[b]);
      lanes.vr[b] = value.real();
      lanes.vi[b] = value.imag();
    }
    if (weights.compensated) {
      const Complex* x = &exponent(column, first);
      const Complex* field = &lab(column, first);
      for (std::size_t b = 0; b < count; ++b) {
        lanes.xr[b] = x[b].real();
        lanes.xi[b] = x[b].imag();
        lanes.ur[b] = field[b].real();
        lanes.ui[b] = field[b].imag();
      }
      for (std::size_t t = 0; t + 1 < weights.width; ++t) {
        const Complex* row = &carried(t, column, first);
        for (std::size_t b = 0; b < count; ++b) {
          lanes.cr[t][b] = row[b].real();
          lanes.ci[t][b] = row[b].imag();
        }
      }
      if (rate_formed) {
        const Complex* rate = &endpoint_rate(column, first);
        for (std::size_t b = 0; b < count; ++b) {
          lanes.dr[b] = rate[b].real();
          lanes.di[b] = rate[b].imag();
        }
      } else {
        adams_lanes_rate(lanes, n_phi, rate_weights, phi_terms, endpoint_rate, column,
                         first);
      }
      adams_lanes_corrected_compensation(lanes, weights, inverse_dz, dz);
      Complex* out = &endpoint(column, first);
      for (std::size_t b = 0; b < count; ++b)
        out[b] = Complex(lanes.nr[b], lanes.ni[b]);
    }
    Complex* out = &advanced(column, first);
    for (std::size_t b = 0; b < count; ++b)
      out[b] = Complex(lanes.vr[b], lanes.vi[b]);
    if (device_scale) {
      if (scale_weights.extent(0) != 0)
        for (std::size_t b = 0; b < count; ++b)
          lanes.scale[b] = rtol * scale_weights(first + b) * magnitude + atol;
      else
        for (std::size_t b = 0; b < count; ++b)
          lanes.scale[b] = rtol * magnitude + atol;
    } else {
      for (std::size_t b = 0; b < count; ++b)
        lanes.scale[b] = divisor(first + b);
    }
    constexpr std::size_t max_rows = 3;
    const std::size_t scored = value_count < max_rows ? value_count : max_rows;
    adams_lanes_defects(lanes, weights, scored, dz);
    double sums[max_rows] = {0.0, 0.0, 0.0};
    for (std::size_t b = 0; b < count; ++b) {
      if (in_norm(first + b) == 0)
        continue;
      for (std::size_t row = 0; row < scored; ++row)
        sums[row] += lanes.defect[row][b];
    }
    for (std::size_t row = 0; row < scored; ++row)
      update[row] += sums[row];
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // One device element's pass, `Width` the trial's samples known at compile time: as
  // the loops below and `adams_defect_block` form it, in the same operations and
  // order, but with the history rows and moments read once into registers. Through
  // the views, which the pass also writes, every use reloads them; on an AMD Instinct
  // MI325X this pass is latency-bound on a single-mode trial's few thousand elements.
  template <std::size_t Width>
  KOKKOS_INLINE_FUNCTION void
  device_element(const std::size_t column, const std::size_t index,
                 const double magnitude, value_type update) const {
    const Complex g = stages(0, column, index);
    source(column, index) = g;
    Complex history[Width];
    for (std::size_t t = 0; t + 1 < Width; ++t)
      history[t] = adams_history(store, weights, t, column, index);
    Complex moment[Width];
    for (std::size_t k = 0; k < Width; ++k)
      if (k < weights.count)
        moment[k] = moments(k, column, index);
    Complex total{};
    for (std::size_t k = 0; k < Width; ++k) {
      if (k >= weights.count)
        break;
      Complex combined = weights.corrector[0][k] * g;
      for (std::size_t t = 1; t < Width; ++t)
        combined += weights.corrector[t][k] * history[t - 1];
      total += moment[k] * combined;
    }
    Complex value = flow(column, index) * current(column, index) + dz * total;
    if (weights.compensated)
      value = adams_corrected_compensation(weights, n_phi, rate_weights, phi_terms,
                                           exponent(column, index), inverse_dz,
                                           lab(column, index), dz, value, endpoint_rate,
                                           endpoint, carried, column, index);
    advanced(column, index) = value;
    if (in_norm(index) == 0)
      return;
    const double scale = device_scale
                             ? (scale_weights.extent(0) != 0
                                    ? rtol * scale_weights(index) * magnitude + atol
                                    : rtol * magnitude + atol)
                             : divisor(index);
    constexpr std::size_t max_rows = 3;
    const std::size_t scored = value_count < max_rows ? value_count : max_rows;
    for (std::size_t row = 0; row < max_rows; ++row) {
      if (row >= scored)
        break;
      Complex difference = strict_scale(weights.rows[row][0], g);
      for (std::size_t t = 1; t < Width; ++t)
        difference =
            strict_accumulate(difference, weights.rows[row][t], history[t - 1]);
      Complex weight{};
      for (std::size_t k = 0; k < Width; ++k) {
        if (k >= weights.count)
          break;
        weight = strict_accumulate(weight, weights.rows[row][Width + k], moment[k]);
      }
      Complex defect = strict_scale(dz, strict_mul(difference, weight));
      if (weights.compensated) {
        Complex compensation =
            strict_scale(weights.comp_rows[row][0], endpoint(column, index));
        for (std::size_t t = 0; t + 1 < Width; ++t)
          compensation = strict_accumulate(compensation, weights.comp_rows[row][t + 1],
                                           carried(t, column, index));
        defect = strict_accumulate(defect, dz, compensation);
      }
      update[row] += squared_abs(strict_divide(defect, scale));
    }
  }
#endif

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t task,
                                         value_type update) const {
    const auto column = task / blocks;
    const auto first = (task % blocks) * adams_block;
    const auto last = first + adams_block < n ? first + adams_block : n;
    const double magnitude = device_scale ? Kokkos::sqrt(magnitude_sum()) : 0.0;
#ifdef SOLPROP_NATIVE_ADAMS_LANES
    if constexpr (vmath::vectorised) {
      lanes_block(column, first, last, magnitude, update);
      return;
    }
#endif
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    static_assert(adams_block == 1, "a device thread takes one element");
    if (first < last) {
      const auto element = [&]<std::size_t Width> {
        device_element<Width>(column, first, magnitude, update);
        return true;
      };
      if (dispatch_width<1, adams_max_samples>(weights.width, element,
                                               [] { return false; }))
        return;
    }
#endif
    double scale[adams_block]{};
    for (std::size_t index = first; index < last; ++index) {
      const Complex g = stages(0, column, index);
      source(column, index) = g;
      Complex total{};
      for (std::size_t k = 0; k < weights.count; ++k) {
        Complex combined = weights.corrector[0][k] * g;
        for (std::size_t t = 1; t < weights.width; ++t)
          combined += weights.corrector[t][k] *
                      adams_history(store, weights, t - 1, column, index);
        total += moments(k, column, index) * combined;
      }
      Complex value = flow(column, index) * current(column, index) + dz * total;
      // The new sample's compensation at the predicted field, untransported, with the
      // rate read on the side of the step's end the step came from.
      if (weights.compensated)
        value = adams_corrected_compensation(
            weights, n_phi, rate_weights, phi_terms, exponent(column, index),
            inverse_dz, lab(column, index), dz, value, endpoint_rate, endpoint, carried,
            column, index);
      advanced(column, index) = value;
      if (in_norm(index) == 0)
        continue;
      if (device_scale)
        scale[index - first] = scale_weights.extent(0) != 0
                                   ? rtol * scale_weights(index) * magnitude + atol
                                   : rtol * magnitude + atol;
      else
        scale[index - first] = divisor(index);
    }
    adams_defect_block(weights, value_count, store, moments, source, carried, endpoint,
                       in_norm, dz, scale, column, first, last, update);
  }
  KOKKOS_INLINE_FUNCTION void join(value_type destination,
                                   const value_type source_values) const {
    for (size_type row = 0; row < value_count; ++row)
      destination[row] += source_values[row];
  }
  KOKKOS_INLINE_FUNCTION void init(value_type values) const {
    for (size_type row = 0; row < value_count; ++row)
      values[row] = 0.0;
  }
};

} // namespace

// The exponential trial without a coupled flow, every formula elementwise, in three
// passes on the trial's own width (solprop SEP 16 phase 3b). The first forms the
// step's exponent, flow and moments, a varying operator's carried samples and the
// prediction, and reduces the accepted state's magnitude for the norm's divisor; the
// second is the source at the prediction; the third is ExponentialCorrection. Each
// threaded pass costs a fork and a join, which on a host is most of what one of these
// passes costs on a few thousand elements. The prediction is written beside the
// accepted state and the two exchange places around the source, so the accepted
// state is never copied.
std::vector<double> GeneralWorkspace::adams_elementwise_trial(
    const double dz, const ExponentialWeights& weights,
    const std::vector<double>& divisor, const TrialScalars& step,
    const TrialScalars& sample, const std::vector<double>& rate) {
  const auto columns = problem_.n_columns();
  const auto n = problem_.dims.n_omega;
  const bool compensated = weights.compensated;
  // solprop's own controller's scale is formed from the magnitude below; any other
  // controller's divisor crosses.
  const bool device_scale = divisor.empty() && adams_device_scale_;
  if (!device_scale)
    adams_upload_divisor(divisor);
  // The corrector forms a compensated trial's endpoint rate and compensation; only its
  // weights cross, with the step's and the sample's scalars, in the trial's one copy.
  upload_trial(step, &sample, compensated ? &rate : nullptr);
  const std::size_t blocks = (n + adams_block - 1) / adams_block;
#ifdef KOKKOS_ENABLE_CUDA
  // A device thread takes one element of a long FP64 chain, and Kokkos's CUDA range
  // reduction gives each block 256 of them: a four-mode capillary's 10496 elements
  // were 41 blocks on an NVIDIA A40's 84 multiprocessors. Blocks of 128 take them
  // all.
  using Tasks = Kokkos::RangePolicy<ExecutionSpace, Kokkos::LaunchBounds<128, 1>>;
#else
  using Tasks = Kokkos::RangePolicy<ExecutionSpace>;
#endif
  const auto tasks = Tasks(adams_execution_space_, 0, columns * blocks);
  const bool separable = problem_.separable_geometry();
  const auto spatial = spatial_;
  const auto active = active_mask_;
  const auto n_pol = problem_.dims.n_pol_state;
  const auto n_stages = problem_.stepper.n_stages;
  const auto phi_terms = phi_terms_;
  const auto phi_delta = stage_phi_delta_;
  const auto n_phi = problem_.linear.n_phi_terms;
  const auto offset = n_stages * n_phi;
  const auto exponent = adams_exponent_;
  const auto flow = adams_propagator_;
  const auto moments = adams_moments_;
  const auto store = adams_store_;
  const auto current = current_;
  const auto predicted = adams_saved_;
  const auto lab = adams_lab_;
  const auto carried = adams_carried_;
  const auto states = adams_states_;
  const auto rates = adams_rates_;
  const auto in_norm = norm_mask_;
  const std::size_t count = weights.count;
  const std::size_t formed = count > weights.order ? count : weights.order;
  const double inverse_dz = 1.0 / dz;
#ifdef SOLPROP_NATIVE_ADAMS_LANES
  // A host block's predict pass forms the corrector's endpoint rate beside the phase,
  // from the weights the trial uploaded with its scalars.
  const auto endpoint_rates = stage_rate_weights_;
  const auto endpoint_rate = adams_endpoint_rate_;
#endif
  // The step's exponent, on a separable route the step's length times each element's
  // operator as the stage kernel forms the whole propagator, and its flow; its
  // moments (kept for the corrector and the norms); a varying operator's carried
  // samples; and the prediction, exp(x) u + dz sum_k M_k sum_j c_jk F_j and the
  // compensation's predictor, the field the lab-frame source is evaluated at.
  const auto predict = KOKKOS_LAMBDA(const std::size_t task, double& magnitude) {
    const auto column = task / blocks;
    const auto first = (task % blocks) * adams_block;
    const auto last = first + adams_block < n ? first + adams_block : n;
    // With vector maths a host block forms its exponentials in passes and the rest of
    // the pass on its lanes (`adams_lanes_predict`); otherwise each element forms its
    // own.
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
    constexpr bool passes = vmath::vectorised;
#else
    // Read only by the host's lanes (Clang's -Wunused-variable on HIP builds).
    [[maybe_unused]] constexpr bool passes = false;
#endif
    if (separable)
      for (std::size_t index = first; index < last; ++index) {
        const Complex x =
            phi_delta(n_stages) *
            spatial.value(column / n_pol, index, active(index) != 0U).linear;
        exponent(column, index) = x;
      // A device lambda may not first capture a view inside `if constexpr` (nvcc):
      // devices never form passes, so the guard is the host's alone.
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
        if constexpr (!passes)
#endif
          flow(column, index) = exponential(x);
      }
#ifdef SOLPROP_NATIVE_ADAMS_LANES
    if constexpr (passes) {
      magnitude += separable
                       ? adams_lanes_predict<false>(
                             phi_terms, phi_delta, endpoint_rates, endpoint_rate,
                             offset, n_phi, weights, formed, dz, inverse_dz, exponent,
                             flow, current, predicted, lab, moments, store, carried,
                             states, rates, in_norm, column, first, last)
                       : adams_lanes_predict<true>(
                             phi_terms, phi_delta, endpoint_rates, endpoint_rate,
                             offset, n_phi, weights, formed, dz, inverse_dz, exponent,
                             flow, current, predicted, lab, moments, store, carried,
                             states, rates, in_norm, column, first, last);
      return;
    }
#endif
    if (!separable)
      adams_phase_block(phi_terms, phi_delta, offset, n_phi, column, first, last,
                        exponent, flow);
    double squares = 0.0;
    for (std::size_t index = first; index < last; ++index) {
      const Complex x = exponent(column, index);
      const Complex propagate = flow(column, index);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
      const Complex accepted = current(column, index);
      if (in_norm(index) != 0)
        squares += squared_abs(accepted);
      Complex value = adams_device_predict(weights, formed, x, propagate, accepted, dz,
                                           moments, store, column, index);
#else
      Complex phi[2 * adams_max_samples + 1];
      adams_moment_phi(weights, x, propagate, formed, phi);
      for (std::size_t k = 0; k < count; ++k)
        moments(k, column, index) = adams_factorial(k) * phi[k + 1];
      const Complex accepted = current(column, index);
      if (in_norm(index) != 0)
        squares += squared_abs(accepted);
      Complex total{};
      for (std::size_t k = 0; k < weights.order; ++k) {
        Complex combined{};
        for (std::size_t t = 0; t < weights.order; ++t)
          combined +=
              weights.predictor[t][k] * adams_history(store, weights, t, column, index);
        const Complex moment = adams_factorial(k) * phi[k + 1];
        total += moment * combined;
      }
      Complex value = propagate * accepted + dz * total;
#endif
      if (compensated) {
        value =
            adams_predicted_compensation(weights, x, propagate, inverse_dz, dz, value,
                                         rates, states, carried, column, index);
        lab(column, index) = value;
      }
      predicted(column, index) = value;
    }
    magnitude += squares;
  };
  if (device_scale) {
    // A threaded reduction in whatever order measures fastest (solprop SEP 25).
    launch::parallel_reduce("solprop_native.adams.exponential_predict", tasks, predict,
                            adams_magnitude_sum_);
  } else {
    double unused = 0.0;
    launch::parallel_reduce("solprop_native.adams.exponential_predict", tasks, predict,
                            unused);
  }
  std::swap(current_, adams_saved_);
  use_uploaded_sample(sample);
  // On a device the source's launches replay as one graph while they repeat.
  adams_source_graph_.run(execution_space_, [&] { adams_evaluate_source(sample); });
  std::swap(current_, adams_saved_);
  const ExponentialCorrection reducer{.value_count = weights.n_rows,
                                      .store = store,
                                      .moments = moments,
                                      .stages = stages_,
                                      .carried = carried,
                                      .phi_terms = phi_terms,
                                      .source = adams_source_,
                                      .flow = flow,
                                      .current = current_,
                                      .advanced = advanced_,
                                      .exponent = exponent,
                                      .lab = lab,
                                      .endpoint = adams_endpoint_,
                                      .endpoint_rate = adams_endpoint_rate_,
                                      .rate_weights = stage_rate_weights_,
                                      .in_norm = in_norm,
                                      .divisor = adams_divisor_,
                                      .scale_weights = adams_scale_weights_,
                                      .magnitude_sum = adams_magnitude_sum_,
                                      .weights = weights,
                                      .dz = dz,
                                      .inverse_dz = inverse_dz,
                                      .rtol = adams_rtol_,
                                      .atol = adams_atol_,
                                      .n = n,
                                      .blocks = blocks,
                                      .n_phi = n_phi,
                                      .device_scale = device_scale,
                                      .rate_formed = !separable};
  auto norms = adams_norms("solprop_native.adams.exponential_correct", tasks, reducer,
                           weights.n_rows);
  adams_trial_compensated_ = compensated;
  adams_trial_coupled_ = false;
  return norms;
}

// Refuse a trial whose bins off the flow's eigenbasis route would need more squarings
// than solprop's reference admits for a flow action (solprop SEP 47 phase 4): the
// augmented exponential of each bin's shifted block for the moments, and the block's
// own `span dz T` for each carried sample. Before any pass of the trial moves the
// state, so a refusal leaves the accepted step as it was.
void GeneralWorkspace::adams_fallback_squarings(const std::size_t section,
                                                const double dz,
                                                const ExponentialWeights& weights) {
  const auto slots = flow_->block_slots();
  const auto route = flow_->route();
  const auto triangular = flow_->fallback_triangular();
  const auto exponent = adams_exponent_;
  const auto modes = problem_.dims.n_modes;
  const std::size_t count = weights.count;
  const auto policy = Kokkos::RangePolicy<ExecutionSpace>(flow_execution_space_, 0,
                                                          problem_.dims.n_omega);
  // The moments' augmented exponent, and the carries' `span dz T`, apart, so that a
  // refusal names which.
  int moments = 0;
  int carries = 0;
  launch::parallel_reduce(
      "solprop_native.adams.fallback_squarings", policy,
      KOKKOS_LAMBDA(const std::size_t index, int& most, int& carried) {
        const auto one = slots(index);
        if (one == 0)
          return;
        const auto path = route(section, one - 1);
        if (path == 0)
          return;
        const Complex argument = adams_moment_argument(weights, exponent(0, index));
        const int squarings = flow_functions::augmented_squarings(
            argument, dz, triangular, path - 1, modes, count);
        most = squarings > most ? squarings : most;
        if (weights.compensated)
          for (std::size_t t = 0; t + 1 < weights.width; ++t) {
            const int carry = flow_functions::augmented_squarings(
                Complex{}, weights.span[t] * dz, triangular, path - 1, modes, 0);
            carried = carry > carried ? carry : carried;
          }
      },
      Kokkos::Max<int>(moments), Kokkos::Max<int>(carries));
  const std::string bin =
      "native cannot take the flow and moments of a coupled flow's bin off its "
      "eigenbasis route";
  if (moments == flow_functions::non_finite || carries == flow_functions::non_finite)
    throw BackendRefusal(bin + ": the step's exponent there is not finite");
  if (moments <= flow_functions::max_squarings &&
      carries <= flow_functions::max_squarings)
    return;
  std::array<char, 32> length{};
  std::snprintf(length.data(), length.size(), "%.4g", dz);
  const std::string limit =
      concat(" would need more than ", flow_functions::max_squarings,
             " squarings, the most this backend admits. Cap dz_max");
  if (moments > flow_functions::max_squarings)
    refuse<BackendRefusal>(bin, " over a step of ", length.data(), " m: its exponent",
                           limit);
  refuse<BackendRefusal>(bin, " over a step of ", length.data(),
                         " m: a history sample's carry to the step's end", limit);
}

std::vector<double> GeneralWorkspace::adams_exponential_trial(
    const double dz, const ExponentialWeights& weights,
    const std::vector<double>& divisor, const TrialScalars& step,
    const TrialScalars& sample, const std::vector<double>& rate) {
  require_adams();
  if (weights.compensated && adams_states_.extent(0) == 0)
    throw std::logic_error("the Adams history carries no compensation");
  // Beside a coupled flow the step's flow and moments are matrices on the block's bins
  // (solprop SEP 16 phases 5 and 6); the sample is the material source alone.
  const bool coupled = static_cast<bool>(flow_);
  if (coupled && adams_block_.extent(0) == 0)
    throw std::logic_error("the Adams history holds no coupled block");
  if (weights.n_rows == 0 || weights.n_rows > 3 || weights.width == 0 ||
      weights.width > adams_store_.extent(0) || weights.count > weights.width ||
      weights.order >= weights.width)
    throw std::invalid_argument("Adams coefficient tables are out of range");
  for (std::size_t t = 0; t + 1 < weights.width; ++t)
    if (weights.slot[t] >= adams_store_.extent(0))
      throw std::out_of_range("Adams history row");
  require_trial_scalars(step);
  require_trial_scalars(sample);
  if (!coupled)
    return adams_elementwise_trial(dz, weights, divisor, step, sample, rate);
  adams_upload_divisor(divisor);
  // The step's exponent and flow, and its moments, once for every formula; the
  // sample's scalars cross in the same copy.
  upload_trial(step, &sample);
  adams_phase();
  const std::size_t section = step.flow_section;
  const bool residual = adams_applied_.extent(0) != 0;
  if (residual)
    upload_weights(execution_space_, adams_end_weights_, step.flow_weights);
  const auto columns = problem_.n_columns();
  const auto n = problem_.dims.n_omega;
  const auto policy = Range2(execution_space_, {0, 0}, {columns, n});
  const auto bins = Kokkos::RangePolicy<ExecutionSpace>(flow_execution_space_, 0, n);
  const auto exponent = adams_exponent_;
  const auto flow = adams_propagator_;
  const auto moments = adams_moments_;
  const std::size_t count = weights.count;
  const auto slots = flow_->block_slots();
  // The elementwise moments, off the block's bins: on them the block's matrices below
  // take their place.
  launch::parallel_for(
      "solprop_native.adams.moments", policy,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        if (slots(index) != 0)
          return;
        Complex phi[2 * adams_max_samples + 1];
        adams_moment_phi(weights, exponent(column, index), flow(column, index), count,
                         phi);
        for (std::size_t k = 0; k < count; ++k)
          moments(k, column, index) = adams_factorial(k) * phi[k + 1];
      });
  // The block's flow and moments, W f(x + h lambda) Y from the section's eigenbasis on
  // its route; a bin off it is formed from its Schur factors by the augmented
  // exponential (`flow_functions`), on a host and a device alike.
  const auto block = adams_block_;
  // A host's channel vectors of a block wider than `adams_max_channels`.
  [[maybe_unused]] const auto channels = adams_channels_;
  const auto modes = problem_.dims.n_modes;
  // The bins off the eigenbasis route (solprop SEP 47 phase 4).
  const auto route = flow_->route();
  const auto fallback_vectors = flow_->fallback_vectors();
  const auto fallback_triangular = flow_->fallback_triangular();
  const auto fallback = adams_fallback_;
  const bool falls_back = flow_->falls_back(section);
  if (flow_->falls_back(section))
    adams_fallback_squarings(section, dz, weights);
  const auto left = flow_->left();
  const auto right = flow_->right();
  const auto eigenvalues = flow_->eigenvalues();
  const auto terms =
      residual ? flow_->residual_terms() : CoupledFlowAction::ComplexView4{};
  const auto frozen = residual ? flow_->frozen() : CoupledFlowAction::ComplexView4{};
  // A device takes a thread per (bin, channel) in the coupled passes below: its
  // lanes outnumber the bins and a channel vector does not fit a thread's
  // registers. A host keeps a task per bin, which forms a bin's shared terms once.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  launch::parallel_for(
      "solprop_native.adams.block_functions",
      Range2(flow_execution_space_, {0, 0}, {n, modes * modes}),
      KOKKOS_LAMBDA(const std::size_t index, const std::size_t pair) {
        const auto one = slots(index);
        if (one == 0)
          return;
        const auto bin = one - 1;
        // A bin off the route is the next pass's.
        if (falls_back && route(section, bin) != 0)
          return;
        const auto i = pair / modes;
        const auto j = pair % modes;
        const Complex shift = exponent(0, index);
        // One entry (i, j) of every function's matrix, a thread each: the sum over
        // the eigenvalues in their order, each function formed where it is needed.
        Complex total[adams_max_samples + 1];
        for (std::size_t k = 0; k <= count; ++k)
          total[k] = Complex{};
        Complex phi[2 * adams_max_samples + 1];
        for (std::size_t m = 0; m < modes; ++m) {
          const Complex x = shift + dz * eigenvalues(section, bin, m);
          const Complex exp_x = exponential(x);
          // The flow is exp(h B); the moments are the shifted block's (solprop SEP
          // 35 section 4.5): every eigenvalue moves by -sigma.
          adams_moment_phi(weights, x, exp_x, count, phi);
          const Complex l = left(section, bin, i, m);
          const Complex r = right(section, bin, m, j);
          total[0] += l * exp_x * r;
          for (std::size_t k = 0; k < count; ++k)
            total[k + 1] += l * (adams_factorial(k) * phi[k + 1]) * r;
        }
        for (std::size_t k = 0; k <= count; ++k)
          block(k, bin, i, j) = total[k];
      });
  if (falls_back)
    // Off the eigenbasis route, a thread per bin: the host task's passes, in its order
    // (the first block row of the augmented exponential of the shifted block, carried
    // back by Z), its matrices in the bin's scratch row rather than on a stack.
    launch::parallel_for(
        "solprop_native.adams.fallback_block_functions", bins,
        KOKKOS_LAMBDA(const std::size_t index) {
          const auto one = slots(index);
          if (one == 0)
            return;
          const auto bin = one - 1;
          const auto path = route(section, bin);
          if (path == 0)
            return;
          const auto f = path - 1;
          const Complex argument = adams_moment_argument(weights, exponent(0, index));
          flow_functions::augmented_row(
              argument, dz, fallback_triangular, f, modes, count,
              flow_functions::augmented_squarings(argument, dz, fallback_triangular, f,
                                                  modes, count),
              fallback, bin);
          const Complex factor = weights.shifted
                                     ? exponential(Complex(0.0, weights.shift))
                                     : Complex(1.0, 0.0);
          const auto product = count + 1;
          for (std::size_t k = 0; k <= count; ++k) {
            for (std::size_t i = 0; i < modes; ++i)
              for (std::size_t j = 0; j < modes; ++j) {
                Complex total{};
                for (std::size_t m = 0; m <= j; ++m)
                  total += fallback_vectors(f, i, m) * fallback(bin, k, m, j);
                fallback(bin, product, i, j) = total;
              }
            const Complex scale =
                k == 0 ? factor : Complex(adams_factorial(k - 1), 0.0);
            for (std::size_t i = 0; i < modes; ++i)
              for (std::size_t j = 0; j < modes; ++j) {
                Complex total{};
                for (std::size_t m = 0; m < modes; ++m)
                  total += fallback(bin, product, i, m) *
                           Kokkos::conj(fallback_vectors(f, j, m));
                block(k, bin, i, j) = scale * total;
              }
          }
        });
#else
  static_cast<void>(falls_back);
  launch::parallel_for(
      "solprop_native.adams.block_functions", bins,
      KOKKOS_LAMBDA(const std::size_t index) {
        const auto one = slots(index);
        if (one == 0)
          return;
        const auto bin = one - 1;
        const auto path = route(section, bin);
        if (path != 0) {
          // Off the eigenbasis route: the first block row of the augmented
          // exponential of the shifted block, carried back by Z, the moments
          // scaled by k! and the flow by the carrier's exp(sigma) (solprop's
          // `coupled_moments`).
          const auto f = path - 1;
          const Complex argument = adams_moment_argument(weights, exponent(0, index));
          flow_functions::augmented_row(
              argument, dz, fallback_triangular, f, modes, count,
              flow_functions::augmented_squarings(argument, dz, fallback_triangular, f,
                                                  modes, count),
              fallback, bin);
          const Complex factor = weights.shifted
                                     ? exponential(Complex(0.0, weights.shift))
                                     : Complex(1.0, 0.0);
          const auto product = count + 1;
          for (std::size_t k = 0; k <= count; ++k) {
            for (std::size_t i = 0; i < modes; ++i)
              for (std::size_t j = 0; j < modes; ++j) {
                Complex total{};
                for (std::size_t m = 0; m <= j; ++m)
                  total += fallback_vectors(f, i, m) * fallback(bin, k, m, j);
                fallback(bin, product, i, j) = total;
              }
            const Complex scale =
                k == 0 ? factor : Complex(adams_factorial(k - 1), 0.0);
            for (std::size_t i = 0; i < modes; ++i)
              for (std::size_t j = 0; j < modes; ++j) {
                Complex total{};
                for (std::size_t m = 0; m < modes; ++m)
                  total += fallback(bin, product, i, m) *
                           Kokkos::conj(fallback_vectors(f, j, m));
                block(k, bin, i, j) = scale * total;
              }
          }
          return;
        }
        const Complex shift = exponent(0, index);
        // Eigenvalue m's functions at `values + m (adams_max_samples + 1)`.
        constexpr std::size_t stride = adams_max_samples + 1;
        Complex stack[adams_channel_stack];
        Complex* const values = adams_channel_rows(stack, channels, index);
        Complex phi[2 * adams_max_samples + 1];
        for (std::size_t m = 0; m < modes; ++m) {
          const Complex x = shift + dz * eigenvalues(section, bin, m);
          const Complex exp_x = exponential(x);
          values[m * stride] = exp_x;
          adams_moment_phi(weights, x, exp_x, count, phi);
          for (std::size_t k = 0; k < count; ++k)
            values[m * stride + k + 1] = adams_factorial(k) * phi[k + 1];
        }
        for (std::size_t k = 0; k <= count; ++k)
          for (std::size_t i = 0; i < modes; ++i)
            for (std::size_t j = 0; j < modes; ++j) {
              Complex total{};
              for (std::size_t m = 0; m < modes; ++m)
                total += left(section, bin, i, m) * values[m * stride + k] *
                         right(section, bin, m, j);
              block(k, bin, i, j) = total;
            }
      });
#endif
  // A varying operator's samples carried to the step's end in the chord's flow:
  // exp((1 - theta_j) x) (D(z_j) - x / dz) u_j, the difference of rates formed first.
  // Beside a coupled flow a frozen-residual flow's R_s(z_j) u_j joins each, against
  // the step's own section, and the block's bins are carried by its matrix flow.
  const auto carried = adams_carried_;
  const auto states = adams_states_;
  const auto rates = adams_rates_;
  const auto applied = adams_applied_;
  const bool compensated = weights.compensated;
  const double inverse_dz = 1.0 / dz;
  if (compensated)
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    launch::parallel_for(
        "solprop_native.adams.coupled_carry",
        Range2(execution_space_, {0, 0}, {n, modes}),
        KOKKOS_LAMBDA(const std::size_t index, const std::size_t i) {
          const auto one = slots(index);
          // A bin off the route is the next pass's.
          if (falls_back && one != 0 && route(section, one - 1) != 0)
            return;
          // A sample's term on channel c, formed where it is needed: a thread holds no
          // channel vectors.
          const auto term = [&](const std::size_t slot, const std::size_t c) {
            Complex value =
                compensation_residual(rates(slot, c, index), exponent(c, index),
                                      inverse_dz, states(slot, c, index));
            if (residual && one != 0) {
              const auto bin = one - 1;
              // R_s(z_j) u_j: the sample's own generator applied once, less the
              // step's section's frozen block.
              Complex extra = applied(slot, c, index);
              for (std::size_t j = 0; j < modes; ++j)
                extra = strict_sub(extra, strict_mul(frozen(section, bin, c, j),
                                                     states(slot, j, index)));
              value = strict_add(value, extra);
            }
            return value;
          };
          for (std::size_t t = 0; t + 1 < weights.width; ++t) {
            const auto slot = weights.slot[t];
            const double span = weights.span[t];
            Complex result;
            if (span == 0.0) {
              result = term(slot, i);
            } else if (one == 0) {
              result = exponential(span * exponent(i, index)) * term(slot, i);
            } else {
              const auto bin = one - 1;
              const Complex shift = exponent(0, index);
              Complex total{};
              for (std::size_t m = 0; m < modes; ++m) {
                Complex value{};
                for (std::size_t j = 0; j < modes; ++j)
                  value += right(section, bin, m, j) * term(slot, j);
                total +=
                    left(section, bin, i, m) *
                    (exponential(span * (shift + dz * eigenvalues(section, bin, m))) *
                     value);
              }
              result = total;
            }
            carried(t, i, index) = result;
          }
        });
  if (compensated && falls_back)
    // Off the eigenbasis route, a thread per bin on the block functions' instance,
    // whose scratch row it reuses: the host task's carry in its order, the bin's
    // channel vectors in blocks 3 (the term) and 4 (its rotation) of the row.
    launch::parallel_for(
        "solprop_native.adams.fallback_carry", bins,
        KOKKOS_LAMBDA(const std::size_t index) {
          const auto one = slots(index);
          if (one == 0)
            return;
          const auto bin = one - 1;
          const auto path = route(section, bin);
          if (path == 0)
            return;
          const auto f = path - 1;
          constexpr std::size_t term = 3;
          constexpr std::size_t rotated = 4;
          for (std::size_t t = 0; t + 1 < weights.width; ++t) {
            const auto slot = weights.slot[t];
            for (std::size_t i = 0; i < modes; ++i)
              fallback(bin, term, 0, i) =
                  compensation_residual(rates(slot, i, index), exponent(i, index),
                                        inverse_dz, states(slot, i, index));
            if (residual)
              for (std::size_t i = 0; i < modes; ++i) {
                // R_s(z_j) u_j: the sample's own generator applied once, less the
                // step's section's frozen block.
                Complex value = applied(slot, i, index);
                for (std::size_t j = 0; j < modes; ++j)
                  value = strict_sub(value, strict_mul(frozen(section, bin, i, j),
                                                       states(slot, j, index)));
                fallback(bin, term, 0, i) =
                    strict_add(fallback(bin, term, 0, i), value);
              }
            const double span = weights.span[t];
            if (span != 0.0) {
              // The reference's flow matrix Z exp(span x) Z^H with x = shift I + dz T,
              // the shift's exponential apart: exp(span shift) Z exp(span dz T) Z^H.
              const double length = span * dz;
              flow_functions::augmented_row(
                  Complex{}, length, fallback_triangular, f, modes, 0,
                  flow_functions::augmented_squarings(Complex{}, length,
                                                      fallback_triangular, f, modes, 0),
                  fallback, bin);
              const Complex factor = exponential(span * exponent(0, index));
              for (std::size_t i = 0; i < modes; ++i)
                for (std::size_t j = 0; j < modes; ++j) {
                  Complex total{};
                  for (std::size_t m = 0; m <= j; ++m)
                    total += fallback_vectors(f, i, m) * fallback(bin, 0, m, j);
                  fallback(bin, 1, i, j) = total;
                }
              for (std::size_t i = 0; i < modes; ++i)
                for (std::size_t j = 0; j < modes; ++j) {
                  Complex total{};
                  for (std::size_t m = 0; m < modes; ++m)
                    total += fallback(bin, 1, i, m) *
                             Kokkos::conj(fallback_vectors(f, j, m));
                  fallback(bin, 2, i, j) = factor * total;
                }
              for (std::size_t i = 0; i < modes; ++i) {
                Complex value{};
                for (std::size_t j = 0; j < modes; ++j)
                  value += fallback(bin, 2, i, j) * fallback(bin, term, 0, j);
                fallback(bin, rotated, 0, i) = value;
              }
              for (std::size_t i = 0; i < modes; ++i)
                fallback(bin, term, 0, i) = fallback(bin, rotated, 0, i);
            }
            for (std::size_t i = 0; i < modes; ++i)
              carried(t, i, index) = fallback(bin, term, 0, i);
          }
        });
#else
    launch::parallel_for(
        "solprop_native.adams.coupled_carry", bins,
        KOKKOS_LAMBDA(const std::size_t index) {
          const auto one = slots(index);
          Complex stack[adams_channel_stack];
          Complex* const term = adams_channel_rows(stack, channels, index);
          Complex* const rotated = term + modes;
          for (std::size_t t = 0; t + 1 < weights.width; ++t) {
            const auto slot = weights.slot[t];
            for (std::size_t i = 0; i < modes; ++i)
              term[i] = compensation_residual(rates(slot, i, index), exponent(i, index),
                                              inverse_dz, states(slot, i, index));
            if (residual && one != 0) {
              const auto bin = one - 1;
              for (std::size_t i = 0; i < modes; ++i) {
                // R_s(z_j) u_j: the sample's own generator applied once, less the
                // step's section's frozen block.
                Complex value = applied(slot, i, index);
                for (std::size_t j = 0; j < modes; ++j)
                  value = strict_sub(value, strict_mul(frozen(section, bin, i, j),
                                                       states(slot, j, index)));
                term[i] = strict_add(term[i], value);
              }
            }
            const double span = weights.span[t];
            if (span != 0.0) {
              if (one == 0) {
                for (std::size_t i = 0; i < modes; ++i)
                  term[i] = exponential(span * exponent(i, index)) * term[i];
              } else if (const auto path = route(section, one - 1); path != 0) {
                // Off the eigenbasis route, the reference's flow matrix
                // Z exp(span x) Z^H with x = shift I + dz T, the shift's exponential
                // apart: exp(span shift) Z exp(span dz T) Z^H, then applied.
                const auto f = path - 1;
                const auto bin = one - 1;
                const double length = span * dz;
                flow_functions::augmented_row(
                    Complex{}, length, fallback_triangular, f, modes, 0,
                    flow_functions::augmented_squarings(
                        Complex{}, length, fallback_triangular, f, modes, 0),
                    fallback, bin);
                const Complex factor = exponential(span * exponent(0, index));
                for (std::size_t i = 0; i < modes; ++i)
                  for (std::size_t j = 0; j < modes; ++j) {
                    Complex total{};
                    for (std::size_t m = 0; m <= j; ++m)
                      total += fallback_vectors(f, i, m) * fallback(bin, 0, m, j);
                    fallback(bin, 1, i, j) = total;
                  }
                for (std::size_t i = 0; i < modes; ++i)
                  for (std::size_t j = 0; j < modes; ++j) {
                    Complex total{};
                    for (std::size_t m = 0; m < modes; ++m)
                      total += fallback(bin, 1, i, m) *
                               Kokkos::conj(fallback_vectors(f, j, m));
                    fallback(bin, 2, i, j) = factor * total;
                  }
                for (std::size_t i = 0; i < modes; ++i) {
                  Complex value{};
                  for (std::size_t j = 0; j < modes; ++j)
                    value += fallback(bin, 2, i, j) * term[j];
                  rotated[i] = value;
                }
                for (std::size_t i = 0; i < modes; ++i)
                  term[i] = rotated[i];
              } else {
                const auto bin = one - 1;
                const Complex shift = exponent(0, index);
                for (std::size_t m = 0; m < modes; ++m) {
                  Complex value{};
                  for (std::size_t j = 0; j < modes; ++j)
                    value += right(section, bin, m, j) * term[j];
                  rotated[m] =
                      exponential(span * (shift + dz * eigenvalues(section, bin, m))) *
                      value;
                }
                for (std::size_t i = 0; i < modes; ++i) {
                  Complex value{};
                  for (std::size_t m = 0; m < modes; ++m)
                    value += left(section, bin, i, m) * rotated[m];
                  term[i] = value;
                }
              }
            }
            for (std::size_t i = 0; i < modes; ++i)
              carried(t, i, index) = term[i];
          }
        });
#endif
  // The predicted lab-frame field, exp(x) u + dz sum_k M_k sum_j c_jk F_j and the
  // compensation's predictor, is the field the lab-frame source is evaluated at; the
  // accepted state is kept aside.
  const auto store = adams_store_;
  const auto current = current_;
  const auto saved = adams_saved_;
  const auto lab = adams_lab_;
  graph::deep_copy(execution_space_, saved, current);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  launch::parallel_for(
      "solprop_native.adams.coupled_predict",
      Range2(execution_space_, {0, 0}, {n, modes}),
      KOKKOS_LAMBDA(const std::size_t index, const std::size_t i) {
        const auto one = slots(index);
        // Channel c's combinations sum_t w_tk F_t, its samples read once.
        const auto combine = [&](const std::size_t c, Complex* out) {
          Complex samples[adams_max_samples];
          for (std::size_t t = 0; t < weights.order; ++t)
            samples[t] = adams_history(store, weights, t, c, index);
          for (std::size_t k = 0; k < weights.order; ++k) {
            Complex total{};
            for (std::size_t t = 0; t < weights.order; ++t)
              total += weights.predictor[t][k] * samples[t];
            out[k] = total;
          }
        };
        Complex combined[adams_max_samples];
        Complex value{};
        if (one == 0) {
          combine(i, combined);
          Complex total{};
          for (std::size_t k = 0; k < weights.order; ++k)
            total += moments(k, i, index) * combined[k];
          value = flow(i, index) * saved(i, index) + dz * total;
        } else {
          const auto bin = one - 1;
          Complex linear{};
          Complex total{};
          for (std::size_t j = 0; j < modes; ++j) {
            linear += block(0, bin, i, j) * saved(j, index);
            combine(j, combined);
            for (std::size_t k = 0; k < weights.order; ++k)
              total += block(k + 1, bin, i, j) * combined[k];
          }
          value = linear + dz * total;
        }
        if (compensated) {
          Complex compensation{};
          for (std::size_t t = 0; t < weights.order; ++t)
            compensation = strict_accumulate(compensation, weights.comp_predictor[t],
                                             carried(t, i, index));
          value = strict_accumulate(value, dz, compensation);
          lab(i, index) = value;
        }
        current(i, index) = value;
      });
#else
  launch::parallel_for(
      "solprop_native.adams.coupled_predict", bins,
      KOKKOS_LAMBDA(const std::size_t index) {
        const auto one = slots(index);
        // Combination k of channel i at `combined[k modes + i]`.
        Complex stack[adams_channel_stack];
        Complex* const combined = adams_channel_rows(stack, channels, index);
        for (std::size_t k = 0; k < weights.order; ++k)
          for (std::size_t i = 0; i < modes; ++i) {
            Complex total{};
            for (std::size_t t = 0; t < weights.order; ++t)
              total +=
                  weights.predictor[t][k] * adams_history(store, weights, t, i, index);
            combined[k * modes + i] = total;
          }
        for (std::size_t i = 0; i < modes; ++i) {
          Complex value{};
          if (one == 0) {
            Complex total{};
            for (std::size_t k = 0; k < weights.order; ++k)
              total += moments(k, i, index) * combined[k * modes + i];
            value = flow(i, index) * saved(i, index) + dz * total;
          } else {
            const auto bin = one - 1;
            Complex linear{};
            Complex total{};
            for (std::size_t j = 0; j < modes; ++j) {
              linear += block(0, bin, i, j) * saved(j, index);
              for (std::size_t k = 0; k < weights.order; ++k)
                total += block(k + 1, bin, i, j) * combined[k * modes + j];
            }
            value = linear + dz * total;
          }
          if (compensated) {
            Complex compensation{};
            for (std::size_t t = 0; t < weights.order; ++t)
              compensation = strict_accumulate(compensation, weights.comp_predictor[t],
                                               carried(t, i, index));
            value = strict_accumulate(value, dz, compensation);
            lab(i, index) = value;
          }
          current(i, index) = value;
        }
      });
#endif
  use_uploaded_sample(sample);
  {
    const FlowAside aside(flow_);
    adams_evaluate(sample);
  }
  // The new sample's compensation at the predicted field, untransported, with the
  // rate read on the side of the step's end the step came from; beside a coupled flow
  // a frozen-residual flow's residual at the step's end joins it.
  const auto endpoint = adams_endpoint_;
  const auto end_applied = adams_end_applied_;
  if (compensated) {
    adams_rate(rate, adams_endpoint_rate_);
    const auto endpoint_rate = adams_endpoint_rate_;
    launch::parallel_for(
        "solprop_native.adams.endpoint", policy,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          endpoint(column, index) = compensation_residual(
              endpoint_rate(column, index), exponent(column, index), inverse_dz,
              lab(column, index));
        });
    if (residual) {
      // The generator at the step's end applied to the predicted field: the new
      // sample's G(z_n + h) u, kept for it on acceptance.
      apply_generator(execution_space_, n, modes, slots, terms, adams_end_weights_, lab,
                      end_applied);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
      launch::parallel_for(
          "solprop_native.adams.endpoint_residual",
          Range2(execution_space_, {0, 0}, {n, modes}),
          KOKKOS_LAMBDA(const std::size_t index, const std::size_t i) {
            const auto one = slots(index);
            if (one == 0)
              return;
            const auto bin = one - 1;
            Complex value = end_applied(i, index);
            for (std::size_t j = 0; j < modes; ++j)
              value = strict_sub(value,
                                 strict_mul(frozen(section, bin, i, j), lab(j, index)));
            endpoint(i, index) = strict_add(endpoint(i, index), value);
          });
#else
      launch::parallel_for(
          "solprop_native.adams.endpoint_residual", bins,
          KOKKOS_LAMBDA(const std::size_t index) {
            const auto one = slots(index);
            if (one == 0)
              return;
            const auto bin = one - 1;
            for (std::size_t i = 0; i < modes; ++i) {
              Complex value = end_applied(i, index);
              for (std::size_t j = 0; j < modes; ++j)
                value = strict_sub(
                    value, strict_mul(frozen(section, bin, i, j), lab(j, index)));
              endpoint(i, index) = strict_add(endpoint(i, index), value);
            }
          });
#endif
    }
  }
  const auto stages = stages_;
  const auto source = adams_source_;
  const auto advanced = advanced_;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  launch::parallel_for(
      "solprop_native.adams.coupled_correct",
      Range2(execution_space_, {0, 0}, {n, modes}),
      KOKKOS_LAMBDA(const std::size_t index, const std::size_t i) {
        const auto one = slots(index);
        source(i, index) = stages(0, i, index);
        // Channel c's combinations sum_t w_tk F_t with the new sample first.
        const auto combine = [&](const std::size_t c, Complex* out) {
          const Complex g = stages(0, c, index);
          Complex samples[adams_max_samples];
          for (std::size_t t = 1; t < weights.width; ++t)
            samples[t] = adams_history(store, weights, t - 1, c, index);
          for (std::size_t k = 0; k < count; ++k) {
            Complex total = weights.corrector[0][k] * g;
            for (std::size_t t = 1; t < weights.width; ++t)
              total += weights.corrector[t][k] * samples[t];
            out[k] = total;
          }
        };
        Complex combined[adams_max_samples];
        Complex value{};
        if (one == 0) {
          combine(i, combined);
          Complex total{};
          for (std::size_t k = 0; k < count; ++k)
            total += moments(k, i, index) * combined[k];
          value = flow(i, index) * current(i, index) + dz * total;
        } else {
          const auto bin = one - 1;
          Complex linear{};
          Complex total{};
          for (std::size_t j = 0; j < modes; ++j) {
            linear += block(0, bin, i, j) * current(j, index);
            combine(j, combined);
            for (std::size_t k = 0; k < count; ++k)
              total += block(k + 1, bin, i, j) * combined[k];
          }
          value = linear + dz * total;
        }
        if (compensated) {
          Complex compensation =
              strict_scale(weights.comp_corrector[0], endpoint(i, index));
          for (std::size_t t = 0; t < weights.order; ++t)
            compensation = strict_accumulate(
                compensation, weights.comp_corrector[t + 1], carried(t, i, index));
          value = strict_accumulate(value, dz, compensation);
        }
        advanced(i, index) = value;
      });
#else
  launch::parallel_for(
      "solprop_native.adams.coupled_correct", bins,
      KOKKOS_LAMBDA(const std::size_t index) {
        const auto one = slots(index);
        Complex stack[adams_channel_stack];
        Complex* const combined = adams_channel_rows(stack, channels, index);
        for (std::size_t i = 0; i < modes; ++i) {
          const Complex g = stages(0, i, index);
          source(i, index) = g;
          for (std::size_t k = 0; k < count; ++k) {
            Complex total = weights.corrector[0][k] * g;
            for (std::size_t t = 1; t < weights.width; ++t)
              total += weights.corrector[t][k] *
                       adams_history(store, weights, t - 1, i, index);
            combined[k * modes + i] = total;
          }
        }
        for (std::size_t i = 0; i < modes; ++i) {
          Complex value{};
          if (one == 0) {
            Complex total{};
            for (std::size_t k = 0; k < count; ++k)
              total += moments(k, i, index) * combined[k * modes + i];
            value = flow(i, index) * current(i, index) + dz * total;
          } else {
            const auto bin = one - 1;
            Complex linear{};
            Complex total{};
            for (std::size_t j = 0; j < modes; ++j) {
              linear += block(0, bin, i, j) * current(j, index);
              for (std::size_t k = 0; k < count; ++k)
                total += block(k + 1, bin, i, j) * combined[k * modes + j];
            }
            value = linear + dz * total;
          }
          if (compensated) {
            Complex compensation =
                strict_scale(weights.comp_corrector[0], endpoint(i, index));
            for (std::size_t t = 0; t < weights.order; ++t)
              compensation = strict_accumulate(
                  compensation, weights.comp_corrector[t + 1], carried(t, i, index));
            value = strict_accumulate(value, dz, compensation);
          }
          advanced(i, index) = value;
        }
      });
#endif
  const CoupledDefectNorms reducer{.value_count = weights.n_rows,
                                   .store = store,
                                   .moments = moments,
                                   .source = source,
                                   .carried = carried,
                                   .endpoint = endpoint,
                                   .block = block,
                                   .slots = slots,
                                   .in_norm = norm_mask_,
                                   .divisor = adams_divisor_,
                                   .weights = weights,
                                   .dz = dz,
                                   .modes = modes,
                                   .channels = adams_channels_};
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // A device item is a (bin, channel) pair, a host item a bin.
  const auto items = n * modes;
#else
  const auto items = n;
#endif
  auto norms =
      adams_norms("solprop_native.adams.coupled_norms",
                  Kokkos::RangePolicy<ExecutionSpace>(flow_execution_space_, 0, items),
                  reducer, weights.n_rows);
  adams_trial_compensated_ = compensated;
  adams_trial_coupled_ = residual && compensated;
  return norms;
}

void GeneralWorkspace::adams_exponential_accept(const double dz, const std::size_t free,
                                                const AdamsWeights& dense) {
  require_adams();
  if (!adams_trial_ready_)
    throw std::logic_error("no Adams trial is held");
  if (free >= adams_store_.extent(0))
    throw std::out_of_range("Adams history row");
  if (dense.width > 0 && adams_slopes_.extent(0) == 0)
    throw std::logic_error("Adams dense observation storage is not allocated");
  adams_trial_ready_ = false;
  const auto columns = problem_.n_columns();
  const auto n = problem_.dims.n_omega;
  const auto policy = Range2(execution_space_, {0, 0}, {columns, n});
  const auto store = adams_store_;
  const auto source = adams_source_;
  const auto slopes = adams_slopes_;
  adams_dense_count_ = dense.width;
  if (dense.width > 0) {
    // The step's samples and the state it starts from, before the new sample takes a
    // row: the accepted corrector's interpolant for dense observation.
    graph::deep_copy(execution_space_, adams_dense_start_, current_);
    graph::deep_copy(execution_space_, adams_dense_exponent_, adams_exponent_);
    launch::parallel_for(
        "solprop_native.adams.exponential_dense", policy,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          slopes(0, column, index) = source(column, index);
          for (std::size_t t = 1; t < dense.width; ++t)
            slopes(t, column, index) = store(dense.slot[t - 1], column, index);
        });
    if (adams_trial_compensated_) {
      // The interpolant's compensation terms, untransported: the endpoint's and
      // (D(z_j) - x / dz) u_j of each history sample it reads.
      const auto terms = adams_dense_terms_;
      const auto endpoint = adams_endpoint_;
      const auto exponent = adams_exponent_;
      const auto states = adams_states_;
      const auto rates = adams_rates_;
      const double inverse_dz = 1.0 / dz;
      launch::parallel_for(
          "solprop_native.adams.dense_terms", policy,
          // Compensation terms only: strict as one function (solprop SEP 25 4.2).
          KOKKOS_LAMBDA(const std::size_t column, const std::size_t index)
              SOLPROP_NATIVE_STRICT_FUNCTION {
                SOLPROP_NATIVE_STRICT_BLOCK
                terms(0, column, index) = endpoint(column, index);
                for (std::size_t t = 1; t < dense.width; ++t) {
                  const auto slot = dense.slot[t - 1];
                  terms(t, column, index) = compensation_residual(
                      rates(slot, column, index), exponent(column, index), inverse_dz,
                      states(slot, column, index));
                }
              });
    }
  }
  adams_dense_compensated_ = dense.width > 0 && adams_trial_compensated_;
  // The accepted field takes `finish`'s boundary treatment: a split-step absorber's
  // half steps, or the band's mask in the pass below. Untransported samples: nothing
  // moves; the new sample takes row `free`, and for a varying operator its state and
  // rate the rows beside it.
  const bool masked = !problem_.absorber.split_step;
  if (!masked)
    split_step(dz, advanced_);
  const bool compensated = adams_trial_compensated_;
  const auto active = active_mask_;
  const auto advanced = advanced_;
  const auto states = adams_states_;
  const auto rates = adams_rates_;
  const auto lab = adams_lab_;
  const auto endpoint_rate = adams_endpoint_rate_;
  const std::size_t blocks = (n + adams_block - 1) / adams_block;
  launch::parallel_for(
      "solprop_native.adams.exponential_accept",
      Kokkos::RangePolicy<ExecutionSpace>(adams_copy_space_, 0, columns * blocks),
      KOKKOS_LAMBDA(const std::size_t task) {
        const auto column = task / blocks;
        const auto first = (task % blocks) * adams_block;
        const auto last = first + adams_block < n ? first + adams_block : n;
        for (std::size_t index = first; index < last; ++index) {
          if (masked && active(index) == 0U)
            advanced(column, index) = Complex{};
          store(free, column, index) = source(column, index);
          if (compensated) {
            states(free, column, index) = lab(column, index);
            rates(free, column, index) = endpoint_rate(column, index);
          }
        }
      });
  std::swap(current_, advanced_);
  // The last evaluation was at the prediction, so no response stash is the state's.
  stash_is_current_ = false;
  // A frozen-residual flow's weights at the new sample, the trial's end.
  if (adams_trial_coupled_)
    graph::deep_copy(execution_space_,
                     Kokkos::subview(adams_applied_, free, Kokkos::ALL, Kokkos::ALL),
                     adams_end_applied_);
  adams_accepted_boundary();
}

// A host's execution instances are partitions of its thread pool, and the next
// trial's passes may run on another one than this step's, so the accepted step ends
// with a fence there. A device's instances are its one stream, which already orders
// the next trial's work after this step's: there the host goes on to the next trial's
// tables while the device finishes the step (on an AMD Instinct MI325X a trial's
// kernels take a few microseconds each, so the wait was a visible part of a trial).
void GeneralWorkspace::adams_accepted_boundary() const {
  if constexpr (build::host)
    execution_space_.fence("solprop_native Adams accepted-step boundary");
}

// The field at the fraction `xi` of the last accepted step: the accepted corrector's
// interpolant integrated exactly against the flow, exp(xi x) u_n + dz sum_k
// xi^(k+1) M_k(xi x) sum_j c_jk F_j (`weights.corrector`), masked into current_.
void GeneralWorkspace::adams_exponential_observe(const double xi, const double dz,
                                                 const ExponentialWeights& weights) {
  require_adams();
  if (adams_dense_count_ == 0)
    throw std::logic_error("no Adams step is held for dense observation");
  const auto columns = problem_.n_columns();
  const auto n = problem_.dims.n_omega;
  const std::size_t count = adams_dense_count_;
  const auto start = adams_dense_start_;
  const auto exponent = adams_dense_exponent_;
  const auto slopes = adams_slopes_;
  const auto current = current_;
  const auto terms = adams_dense_terms_;
  const bool compensated = adams_dense_compensated_;
  launch::parallel_for(
      "solprop_native.adams.exponential_observe",
      Range2(execution_space_, {0, 0}, {columns, n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        const Complex x = xi * exponent(column, index);
        const Complex e = exponential(x);
        Complex phi[2 * adams_max_samples + 1];
        if (weights.shifted) {
          // The accepted step's own sigma (solprop SEP 35 section 4.1): the moments at
          // xi (x - sigma), its rotated samples, and exp(-sigma (1 - xi)) on the sum.
          const Complex residual =
              xi * (exponent(column, index) - Complex(0.0, weights.shift));
          adams_phi(residual, Kokkos::abs(residual), exponential(residual), count, phi);
        } else {
          adams_phi(x, Kokkos::abs(x), e, count, phi);
        }
        Complex total{};
        double power = xi;
        for (std::size_t k = 0; k < count; ++k) {
          Complex combined{};
          for (std::size_t t = 0; t < count; ++t) {
            Complex slope = slopes(t, column, index);
            if (weights.shifted && t > 0)
              slope = strict_mul(
                  Complex(weights.factor_re[t - 1], weights.factor_im[t - 1]), slope);
            combined += weights.corrector[t][k] * slope;
          }
          total += (power * (adams_factorial(k) * phi[k + 1])) * combined;
          power *= xi;
        }
        if (weights.shifted)
          total = exponential(Complex(0.0, -weights.shift * (1.0 - xi))) * total;
        Complex value = e * start(column, index) + dz * total;
        if (compensated) {
          // e^{(xi - theta_j) x} carries each term; for the endpoint it is an inverse
          // flow, so a bin the step damps by more than an e-fold takes the anchor's.
          const Complex chord = exponent(column, index);
          value = adams_observed_compensation(weights, count, xi, dz, chord, value,
                                              terms, column, index);
          if (chord.real() < -1.0)
            value = e * start(column, index);
        }
        current(column, index) = value;
      });
  mask(current_);
  stash_is_current_ = false;
  execution_space_.fence("solprop_native Adams observation boundary");
}

// --- solprop SEP 47: a Runge-Kutta march's dense output in a native session ---------
//
// DOPRI5's continuous extension (Hairer, Norsett and Wanner, II.6) in the step's
// interaction picture, as solprop's `extension_coefficients`, `extension_value` and
// `HostDriver._observe_inside` form it: with `v0` the state at the step's anchor, `k_i`
// its stages in the anchor's frame and `v1 - v0 = h sum_i b_i k_i`, the coefficients
// `v0`, `v1 - v0`, `h k1 - (v1 - v0)`, `(v1 - v0) - h k7 - (h k1 - (v1 - v0))` and
// `h sum_i d_i k_i`, evaluated at the fraction theta as `r1 + theta (r2 + (1 - theta)
// (r3 + theta (r4 + (1 - theta) r5)))`. A bin the step's flow changes by more than an
// e-fold takes the anchor's value: no polynomial in theta follows its exponentially
// growing interaction-picture derivative. The value is carried from the anchor to the
// plane by `U(position, anchor)` and masked. Every product and sum is rounded on its
// own, in solprop's order.

namespace {

/** The most stages a dense extension reads. */
constexpr std::size_t rk_dense_max_stages = 16;

/** A tableau's solution and dense-output weights, captured by a kernel. */
struct RungeKuttaDenseWeights {
  double solution[rk_dense_max_stages]{};
  double dense[rk_dense_max_stages]{};
  std::size_t stages{};
};

/** The step's phase increment at one element from the uploaded weights' last row: on a
    separable route the step's length times the element's operator, otherwise `sum_t
    dPhi_t Phi_t` in term order (`adams_phase`'s exponent). */
struct StepExponent {
  bool separable{};
  SpatialCoefficients spatial;
  ByteView active;
  RealView delta;
  ComplexView3 phi_terms;
  std::size_t n_pol{};
  std::size_t n_stages{};
  std::size_t n_phi{};
  KOKKOS_INLINE_FUNCTION Complex operator()(const std::size_t column,
                                            const std::size_t index) const {
    if (separable) {
      const auto values = spatial.value(column / n_pol, index, active(index) != 0U);
      return delta(n_stages) * values.linear;
    }
    return adams_phase_sum(phi_terms, delta, n_stages * n_phi, n_phi, column, index);
  }
};

} // namespace

void GeneralWorkspace::rk_dense_allocate(const std::vector<double>& weights) {
  require_open();
  if (problem_.stepper.low_storage || stages_.extent(0) == 0)
    throw std::logic_error("Runge-Kutta dense output needs the Runge-Kutta stages");
  if (weights.size() != problem_.stepper.n_stages ||
      weights.size() > rk_dense_max_stages)
    throw std::invalid_argument("the dense-output weights are one per stage");
  const auto columns = problem_.n_columns();
  const auto n = problem_.dims.n_omega;
  rk_dense_weights_ = weights;
  if (rk_dense_coefficients_.extent(0) == 0) {
    rk_dense_coefficients_ =
        ComplexView3("solprop_native.rk.dense_coefficients", 5, columns, n);
    rk_dense_stiff_ = Kokkos::View<unsigned char**, Layout, MemorySpace>(
        "solprop_native.rk.dense_stiff", columns, n);
    rk_dense_held_ = ComplexView2("solprop_native.rk.dense_held", columns, n);
  }
  rk_dense_ready_ = false;
}

void GeneralWorkspace::rk_dense_hold(const double dz) {
  require_open();
  if (rk_dense_coefficients_.extent(0) == 0)
    throw std::logic_error("Runge-Kutta dense output storage is not allocated");
  if (!stages_ready_)
    throw std::logic_error("no Runge-Kutta trial is held for a dense extension");
  RungeKuttaDenseWeights weights;
  weights.stages = problem_.stepper.n_stages;
  for (std::size_t stage = 0; stage < weights.stages; ++stage) {
    weights.solution[stage] = problem_.stepper.solution_weights[stage];
    weights.dense[stage] = rk_dense_weights_[stage];
  }
  // The trial's own scalars are bound: their last row is the whole step's.
  const StepExponent step{.separable = problem_.separable_geometry(),
                          .spatial = spatial_,
                          .active = active_mask_,
                          .delta = stage_phi_delta_,
                          .phi_terms = phi_terms_,
                          .n_pol = problem_.dims.n_pol_state,
                          .n_stages = problem_.stepper.n_stages,
                          .n_phi = problem_.linear.n_phi_terms};
  const auto current = current_;
  const auto stages = stages_;
  const auto coefficients = rk_dense_coefficients_;
  const auto stiff = rk_dense_stiff_;
  launch::parallel_for(
      "solprop_native.rk.dense_hold",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index)
          SOLPROP_NATIVE_STRICT_FUNCTION {
            SOLPROP_NATIVE_STRICT_BLOCK
            Complex span{};
            Complex dense{};
            for (std::size_t stage = 0; stage < weights.stages; ++stage) {
              const Complex slope = stages(stage, column, index);
              span = strict_accumulate(span, weights.solution[stage], slope);
              dense = strict_accumulate(dense, weights.dense[stage], slope);
            }
            span = strict_scale(dz, span);
            const Complex first =
                strict_sub(strict_scale(dz, stages(0, column, index)), span);
            const Complex last = strict_sub(
                strict_sub(span,
                           strict_scale(dz, stages(weights.stages - 1, column, index))),
                first);
            coefficients(0, column, index) = current(column, index);
            coefficients(1, column, index) = span;
            coefficients(2, column, index) = first;
            coefficients(3, column, index) = last;
            coefficients(4, column, index) = strict_scale(dz, dense);
            const Complex x = step(column, index);
            stiff(column, index) = Kokkos::abs(x.real()) > 1.0 ? 1U : 0U;
          });
  rk_dense_ready_ = true;
}

void GeneralWorkspace::rk_dense_observe(const double theta, const TrialScalars& carry) {
  require_open();
  if (!rk_dense_ready_)
    throw std::logic_error("no Runge-Kutta step is held for a dense observation");
  require_trial_scalars(carry);
  graph::deep_copy(execution_space_, rk_dense_held_, current_);
  upload_trial(carry);
  const StepExponent flow{.separable = problem_.separable_geometry(),
                          .spatial = spatial_,
                          .active = active_mask_,
                          .delta = stage_phi_delta_,
                          .phi_terms = phi_terms_,
                          .n_pol = problem_.dims.n_pol_state,
                          .n_stages = problem_.stepper.n_stages,
                          .n_phi = problem_.linear.n_phi_terms};
  const double rest = 1.0 - theta;
  const auto coefficients = rk_dense_coefficients_;
  const auto stiff = rk_dense_stiff_;
  const auto current = current_;
  launch::parallel_for(
      "solprop_native.rk.dense_observe",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column,
                    const std::size_t index) SOLPROP_NATIVE_STRICT_FUNCTION {
        SOLPROP_NATIVE_STRICT_BLOCK
        const Complex start = coefficients(0, column, index);
        Complex value = start;
        if (stiff(column, index) == 0U) {
          Complex inner = strict_scale(rest, coefficients(4, column, index));
          inner =
              strict_scale(theta, strict_add(coefficients(3, column, index), inner));
          inner = strict_scale(rest, strict_add(coefficients(2, column, index), inner));
          inner =
              strict_scale(theta, strict_add(coefficients(1, column, index), inner));
          value = strict_add(start, inner);
        }
        current(column, index) = strict_mul(exponential(flow(column, index)), value);
      });
  mask(current_);
  stash_is_current_ = false;
  // A plane inside a step already taken has no quadrature score of its own, as on
  // the driver's path, which reloads the observed field.
  quadrature_error_ = Kokkos::Experimental::quiet_NaN_v<double>;
  quadrature_stale_ = false;
  execution_space_.fence("solprop_native Runge-Kutta observation boundary");
}

void GeneralWorkspace::rk_dense_restore() {
  require_open();
  if (rk_dense_held_.extent(0) == 0)
    throw std::logic_error("Runge-Kutta dense output storage is not allocated");
  graph::deep_copy(execution_space_, current_, rk_dense_held_);
  // As the driver's march reloads the committed state (`_adams_restore`): the next
  // step evaluates its first stage afresh rather than reuse the crossing step's last,
  // so that the session's run is the driver's whatever FFT plans round each.
  clear_fsal();
  execution_space_.fence("solprop_native Runge-Kutta restore boundary");
}

void GeneralWorkspace::rk_dense_release() noexcept {
  rk_dense_coefficients_ = {};
  rk_dense_stiff_ = {};
  rk_dense_held_ = {};
  // A moved-in empty vector releases the storage and cannot throw (`= {}` may).
  rk_dense_weights_ = std::vector<double>();
  rk_dense_ready_ = false;
}

std::vector<double> GeneralWorkspace::collect_trial() {
  // A coupled flow's last stage is carried back by the step's own inverse action, so
  // it is not the next step's first stage; solprop's reference takes no FSAL there.
  pending_fsal_ =
      problem_.stepper.first_same_as_last && !problem_.absorber.split_step && !flow_;
  graph::deep_copy(execution_space_, pinned_partials_, partials_);
  execution_space_.fence("solprop_native trial boundary");
  const auto& host = pinned_partials_;
  // Every column's slots, in order: the host's reduction is the caller's sum of a row.
  const auto slots = host.extent(1);
  std::vector<double> result(3 * slots);
  for (std::size_t row = 0; row < 3; ++row)
    for (std::size_t slot = 0; slot < slots; ++slot)
      result[row * slots + slot] = host(row, slot);
  return result;
}

void GeneralWorkspace::reuse_fsal() {
  const auto stages = stages_;
  const auto recycled = recycled_;
  const auto solution = solution_;
  const auto error = error_;
  const auto whole = whole_propagator_;
  const double solution_weight = problem_.stepper.solution_weights[0];
  const double error_weight = problem_.stepper.error_weights[0];
  const auto n = problem_.dims.n_omega;
  launch::parallel_for(
      "solprop_native.stage.reuse_fsal",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        const Complex value = recycled(column, index);
        stages(0, column, index) = value;
        solution(column, index) = solution_weight * value;
        error(column, index) = error_weight * value;
      });
}

TransverseChunk
GeneralWorkspace::transverse_chunk(const std::size_t first_unit) const noexcept {
  const auto units =
      std::min(transverse_units_per_chunk(), transverse_units() - first_unit);
  if (problem_.slab_geometry())
    return {.first_row = first_unit * problem_.spatial.n_p,
            .rows = units * problem_.spatial.n_p,
            .first_node = first_unit * problem_.spatial.n_x,
            .nodes = units * problem_.spatial.n_x};
  return {
      .first_row = first_unit, .rows = units, .first_node = first_unit, .nodes = units};
}

void GeneralWorkspace::transverse_forward() {
  if (slab_forward_)
    graph::execute(*slab_forward_, transverse_, transverse_,
                   KokkosFFT::Normalization::none);
  else
    graph::execute(*transverse_forward_, transverse_, transverse_,
                   KokkosFFT::Normalization::none);
}

void GeneralWorkspace::transverse_inverse() { transverse_inverse(transverse_); }

// The in-place inverse transverse plan on another volume of the same extents: the
// plan's own scratch or the magnetic closure's (solprop SEP 34).
void GeneralWorkspace::transverse_inverse(const ComplexView3& scratch) {
  if (slab_inverse_)
    graph::execute(*slab_inverse_, scratch, scratch, KokkosFFT::Normalization::none);
  else
    graph::execute(*transverse_inverse_, scratch, scratch,
                   KokkosFFT::Normalization::none);
}

void GeneralWorkspace::separable_stage_input(const std::size_t stage, const double dz) {
  const auto current = current_;
  const auto stages = stages_;
  const auto tableau = tableau_;
  const auto spatial = spatial_;
  const auto delta = stage_phi_delta_;
  const auto at = stage_phi_at_;
  const auto active = active_mask_;
  const auto noise = noise_;
  const auto scratch = transverse_;
  const auto whole = whole_propagator_;
  const auto index = stage_index();
  const auto n_stages = problem_.stepper.n_stages;
  const bool add_noise = problem_.stepper.modified_noise && has_noise_;
  const double inverse_area_count = inverse_transverse_count();
  // The stage input at one element, given its spatial value and its propagator.
  const auto input = KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega,
                                   const Complex linear, const Complex propagator) {
    Complex accumulated{};
    for (std::size_t previous = 0; previous < stage; ++previous)
      accumulated +=
          tableau(stage * n_stages + previous) * stages(previous, column, omega);
    Complex value = propagator * (current(column, omega) + dz * accumulated);
    if (add_noise) {
      const double angle = at(stage) * linear.imag();
      value += noise(column, omega) * Complex(Kokkos::cos(angle), Kokkos::sin(angle));
    }
    scratch.data()[index.cell(column, omega)] = value * inverse_area_count;
  };
  // One element's propagators: devices, and a host whose vector maths is the system
  // libm, where the per-element form is the faster (solprop ADR-059).
  const auto element =
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
    const auto values = spatial.value(index.mode(column), omega, active(omega) != 0U);
    input(column, omega, values.linear, propagator_exp(delta(stage) * values.linear));
    if (stage + 1 == n_stages)
      whole(column, omega) = propagator_exp(delta(n_stages) * values.linear);
  };
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  if constexpr (vmath::vectorised) {
    // Each tile's propagators in one pass of the vector maths, then the elements in
    // `crossing_for`'s order: the propagators' cosines and sines took a fifth of a
    // one-thread `line` run one element at a time (solprop issue #338).
    const bool last = stage + 1 == n_stages;
    crossing_tiles(
        "solprop_native.free.stage_input", execution_space_, problem_.n_columns(),
        problem_.dims.n_omega, [=](const CrossingTile& tile) {
          PropagatorTile step;
          PropagatorTile finish;
          Complex linear[PropagatorTile::capacity];
          tile.visit<true>([&](const std::size_t column, const std::size_t omega) {
            const auto k = tile.at(column, omega);
            linear[k] =
                spatial.value(index.mode(column), omega, active(omega) != 0U).linear;
            const Complex argument = delta(stage) * linear[k];
            step.real[k] = argument.real();
            step.imag[k] = argument.imag();
            if (last) {
              const Complex whole_argument = delta(n_stages) * linear[k];
              finish.real[k] = whole_argument.real();
              finish.imag[k] = whole_argument.imag();
            }
          });
          step.form(tile.size());
          if (last)
            finish.form(tile.size());
          tile.visit<false>([&](const std::size_t column, const std::size_t omega) {
            const auto k = tile.at(column, omega);
            input(column, omega, linear[k], step[k]);
            if (last)
              whole(column, omega) = finish[k];
          });
        });
    return;
  }
#endif
  crossing_for("solprop_native.free.stage_input", execution_space_,
               problem_.n_columns(), problem_.dims.n_omega, element);
}

void GeneralWorkspace::separable_load_current() {
  const auto current = current_;
  const auto scratch = transverse_;
  const auto index = register_index();
  const double scale = inverse_transverse_count();
  crossing_for(
      "solprop_native.free.current_transverse_input", execution_space_,
      problem_.n_columns(), problem_.dims.n_omega,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        scratch.data()[index.cell(column, omega)] = scale * current(column, omega);
      });
  transverse_inverse();
}

void GeneralWorkspace::separable_pack_chunk(const TransverseChunk& chunk,
                                            const bool material) {
  const auto fine = fine_spectrum_;
  // Every crop bin of the chunk's rows is written below; only the bins outside the
  // crop need zeros, each chunk, since the inverse transform may destroy its input.
  // Rows past a short last chunk keep earlier finite values nothing reads back.
  if (!separable_gaps_.empty()) {
    const auto gaps = separable_gaps_;
    cartesian_for(
        "solprop_native.free.zero_fine_gaps", execution_space_,
        chunk.rows * problem_.dims.n_pol_state, gaps.extent(0),
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t gap) {
          fine(row, gaps(gap)) = Complex{};
        });
  }
  const auto scratch = transverse_;
  const auto crop = crop_index_;
  const auto n_pol = problem_.dims.n_pol_state;
  [[maybe_unused]] const auto layout = transverse_layout();
  const auto first = chunk.first_row;
  const auto n = problem_.dims.n_omega;
  const auto n_to = problem_.dims.n_to;
  const bool analytic = problem_.dims.family == FieldFamily::field_analytic;
  // Free space's material field carries the pad scale here; the hybrid planar kind's
  // slab synthesis carries it (solprop's `Transform.synthesis`).
  const double scale = 1.0 / static_cast<double>(material && !problem_.slab_geometry()
                                                     ? problem_.dims.n_t
                                                     : problem_.dims.n_to);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  tiled_crossing(
      "solprop_native.free.pack_temporal_chunk", execution_space_, chunk.rows * n_pol,
      n, true,
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        return layout.cell(scratch, omega * n_pol + row % n_pol, first + row / n_pol);
      },
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega,
                    const Complex value) {
        const double factor =
            analytic && omega > 0 && 2 * crop(omega) < n_to ? 2.0 : 1.0;
        fine(row, crop(omega)) = factor * scale * value;
      });
#else
  // The chunk's row `row` is the state's column `first_column + row`.
  const auto index = register_index();
  const auto first_column = first * n_pol;
  crossing_for<false, false, true>(
      "solprop_native.free.pack_temporal_chunk", execution_space_, chunk.rows * n_pol,
      n, KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        // The analytic signal doubles the bins strictly between zero and the fine
        // grid's Nyquist bin, the same rule as the modal pad and the reference.
        const double factor =
            analytic && omega > 0 && 2 * crop(omega) < n_to ? 2.0 : 1.0;
        fine(row, crop(omega)) =
            factor * scale * scratch.data()[index.cell(first_column + row, omega)];
      });
#endif
}

// The chunk's physical nodes from its temporal rows: free space's are the rows
// themselves; the hybrid planar kind's are the slab synthesis of each y row.
void GeneralWorkspace::separable_nodes(const TransverseChunk& chunk) {
  const bool real_field = problem_.dims.family == FieldFamily::field_real;
  const auto n_to = problem_.dims.n_to;
  const auto n_pol = problem_.dims.n_pol_state;
  if (problem_.slab_geometry()) {
    const auto units = chunk.nodes / problem_.spatial.n_x;
    if (real_field)
      slab_contract("solprop_native.slab.synthesis", execution_space_, synthesis_,
                    modal_real_, node_real_, units, n_to);
    else
      slab_contract("solprop_native.slab.synthesis", execution_space_, synthesis_,
                    modal_complex_, node_complex_, units, n_to);
    return;
  }
  if (separable_aliased_)
    return;
  const auto time = modal_real_;
  const auto nodes = node_real_;
  const auto complex_time = modal_complex_;
  const auto complex_nodes = node_complex_;
  cartesian_for(
      "solprop_native.free.physical_columns", execution_space_, chunk.rows * n_pol,
      n_to, KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
        if (real_field)
          nodes(row, index) = time(row, index);
        else
          complex_nodes(row, index) = complex_time(row, index);
      });
}

// The hybrid planar kind's slab projection of a chunk's node source onto its y rows'
// columns: the polarisation, or one envelope-plasma current stream.
void GeneralWorkspace::slab_project(const TransverseChunk& chunk,
                                    const EnvelopeStream* stream) {
  const auto units = chunk.nodes / problem_.spatial.n_x;
  const auto n_to = problem_.dims.n_to;
  if (stream != nullptr)
    slab_contract("solprop_native.slab.projection", execution_space_, projection_,
                  stream->node, stream->projected, units, n_to);
  else if (problem_.dims.family == FieldFamily::envelope)
    slab_contract("solprop_native.slab.projection", execution_space_, projection_,
                  polarisation_complex_, projected_complex_, units, n_to);
  else
    slab_contract("solprop_native.slab.projection", execution_space_, projection_,
                  polarisation_real_, projected_real_, units, n_to);
}

void GeneralWorkspace::separable_finish_transform(const std::size_t stage,
                                                  const bool material) {
  const auto scratch = transverse_;
  const auto spatial = spatial_;
  const auto stages = stages_;
  const auto window = omega_window_;
  const auto active = active_mask_;
  const auto delta = stage_phi_delta_;
  const auto index = stage_index();
  // The finish at one element, given its prefactor and its inverse propagator.
  const auto finish = KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega,
                                    const Complex prefactor, const Complex inverse) {
    Complex value = scratch.data()[index.cell(column, omega)];
    value *=
        material ? prefactor * window(omega) : Complex(active(omega) != 0U ? 1.0 : 0.0);
    value *= inverse;
    if (material)
      stages(stage, column, omega) = value;
    else
      stages(stage, column, omega) += value;
  };
  // One element, on devices and without host vector maths (as the stage input).
  const auto element =
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
    const auto values = spatial.value(index.mode(column), omega, active(omega) != 0U);
    finish(column, omega, values.prefactor,
           propagator_exp(-delta(stage) * values.linear));
  };
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  if constexpr (vmath::vectorised) {
    // As the stage input: each tile's inverse propagators in one pass (issue #338).
    crossing_tiles(
        "solprop_native.free.finish_transform", execution_space_, problem_.n_columns(),
        problem_.dims.n_omega, [=](const CrossingTile& tile) {
          PropagatorTile inverse;
          Complex prefactor[PropagatorTile::capacity];
          tile.visit<true>([&](const std::size_t column, const std::size_t omega) {
            const auto k = tile.at(column, omega);
            const auto values =
                spatial.value(index.mode(column), omega, active(omega) != 0U);
            prefactor[k] = values.prefactor;
            const Complex argument = -delta(stage) * values.linear;
            inverse.real[k] = argument.real();
            inverse.imag[k] = argument.imag();
          });
          inverse.form(tile.size());
          tile.visit<true>([&](const std::size_t column, const std::size_t omega) {
            const auto k = tile.at(column, omega);
            finish(column, omega, prefactor[k], inverse[k]);
          });
        });
    return;
  }
#endif
  crossing_for<false, false, true>("solprop_native.free.finish_transform",
                                   execution_space_, problem_.n_columns(),
                                   problem_.dims.n_omega, element);
}

// Free space (solprop `NumpyRefBackend._nonlinear_free`) and the hybrid planar kind
// (`_nonlinear_slab`): the stage input to the transverse spectrum, the inverse
// transverse transform, then chunk by chunk the temporal rows to time, the material
// response at the physical nodes (for the planar kind, synthesised from and projected
// back onto the rows in time) or the in-stage absorber on the rows, and back; the
// forward transverse transform and each column's prefactor and propagator.
void GeneralWorkspace::evaluate_separable_stage(const std::size_t stage,
                                                const double dz,
                                                const bool accumulate) {
  auto* const dispersive = separable_dispersive_.get();
  const bool nonlinear = problem_.kerr.nonlinear || problem_.plasma.enabled ||
                         problem_.raman.enabled || dispersive != nullptr;
  const auto stages = stages_;
  const auto solution = solution_;
  const auto error = error_;
  const auto n = problem_.dims.n_omega;
  // A material finish assigns the stage row; only an absorber-only or a linear stage
  // accumulates into (or leaves) a cleared row.
  if (!nonlinear)
    cartesian_for(
        "solprop_native.free.clear_stage", execution_space_, problem_.n_columns(), n,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
          stages(stage, column, omega) = Complex{};
        });
  separable_stage_input(stage, dz);
  for (const bool material : {true, false}) {
    if ((material && !nonlinear) || (!material && !problem_.absorber.in_stage))
      continue;
    if (!material)
      separable_stage_input(stage, dz);
    separable_core(stage, material);
    separable_finish_transform(stage, material);
  }
  if (!accumulate)
    return;
  const double weight = problem_.stepper.solution_weights[stage];
  const double error_weight = problem_.stepper.error_weights[stage];
  cartesian_for(
      "solprop_native.free.accumulate_stage", execution_space_, problem_.n_columns(), n,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        const Complex value = stages(stage, column, omega);
        if (stage == 0) {
          solution(column, omega) = weight * value;
          error(column, omega) = error_weight * value;
        } else {
          solution(column, omega) += weight * value;
          error(column, omega) += error_weight * value;
        }
      });
}

void GeneralWorkspace::separable_to_time(const TransverseChunk& chunk) {
  const bool tail = temporal_tail_rows_ != 0 &&
                    chunk.rows * problem_.dims.n_pol_state == temporal_tail_rows_;
  if (problem_.dims.family == FieldFamily::field_real) {
    if (tail)
      temporal_tail_inverse_real_.execute();
    else
      graph::execute(*inverse_real_plan_, fine_spectrum_, modal_real_,
                     KokkosFFT::Normalization::none);
  } else {
    if (tail)
      temporal_tail_inverse_complex_.execute();
    else
      graph::execute(*inverse_complex_plan_, fine_spectrum_, modal_complex_,
                     KokkosFFT::Normalization::none);
  }
}

void GeneralWorkspace::separable_to_frequency(const TransverseChunk& chunk) {
  const bool tail = temporal_tail_rows_ != 0 &&
                    chunk.rows * problem_.dims.n_pol_state == temporal_tail_rows_;
  if (problem_.dims.family == FieldFamily::envelope) {
    if (tail)
      temporal_tail_forward_complex_.execute();
    else
      graph::execute(*forward_complex_plan_, projected_complex_, projected_spectrum_,
                     KokkosFFT::Normalization::none);
  } else {
    if (tail)
      temporal_tail_forward_real_.execute();
    else
      graph::execute(*forward_real_plan_, projected_real_, projected_spectrum_,
                     KokkosFFT::Normalization::none);
  }
}

// `volume_kerr_`'s stage core: the scratch's (omega, y, x) volume to time and the
// plane at once, the Kerr polarisation in place sample by sample -- the device
// column's arithmetic (`device_response`), the pad scale 1/n_t that the chunk's pack
// would apply taken with the field -- and back. The rank-3 transform is the chunks'
// temporal transforms and the plane's together, the same DFT to rounding, without the
// pack's and the crop's transposes.
void GeneralWorkspace::separable_volume_core(const std::size_t stage) {
  Kokkos::Profiling::pushRegion("solprop_native.free.volume_core");
  const auto scratch = volume_order_ ? volume_scratch_ : transverse_;
  graph::execute(*volume_inverse_, scratch, scratch, KokkosFFT::Normalization::none);
  const auto window = time_window_;
  const auto kerr_density = stage_kerr_density_;
  const double kerr_coefficient = problem_.kerr.coefficient;
  const double self_factor = problem_.kerr.self_factor;
  const double conjugate_factor = problem_.kerr.conjugate_factor;
  const double scale = 1.0 / static_cast<double>(problem_.dims.n_t);
  const auto plane = transverse_.extent(1) * transverse_.extent(2);
  const bool time_fastest = volume_order_;
  const auto n_t = problem_.dims.n_t;
  launch::parallel_for(
      "solprop_native.free.volume_kerr",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, scratch.size()),
      KOKKOS_LAMBDA(const std::size_t item) {
        const std::size_t time = time_fastest ? item % n_t : item / plane;
        Complex& cell = scratch.data()[item];
        const Complex value = scale * cell;
        const double modulus = squared_abs(value);
        const Complex square = value * value;
        const double factor = kerr_coefficient * kerr_density(stage);
        const double weighted_real = conjugate_factor * square.real();
        const double weighted_imag = conjugate_factor * square.imag();
        const double weighted_self = self_factor * modulus;
        const Complex result(
            factor * (weighted_self * value.real() +
                      (weighted_real * value.real() + weighted_imag * value.imag())),
            factor * (weighted_self * value.imag() +
                      (weighted_imag * value.real() - weighted_real * value.imag())));
        cell = result * window(time);
      });
  graph::execute(*volume_forward_, scratch, scratch, KokkosFFT::Normalization::none);
  Kokkos::Profiling::popRegion();
}

// The in-stage collar's term alone (solprop SEP 40 section 4.7): with no temporal
// rate beside it, the rate depends on the column alone and commutes with the temporal
// transforms, so the term is the rate times the scratch's spectrum, in the transverse
// physical space, and the pass takes only the transverse pair. The temporal pair
// would return the crop's bins to rounding; the bins it would drop lie outside the
// active band, which the finish masks.
void GeneralWorkspace::separable_spatial_core() {
  Kokkos::Profiling::pushRegion("solprop_native.free.spatial_core");
  transverse_inverse();
  const auto index = register_index();
  const auto spatial = spatial_;
  const auto scratch = transverse_;
  const auto n_pol = problem_.dims.n_pol_state;
  cartesian_for(
      "solprop_native.free.absorber_spectrum", execution_space_, problem_.n_columns(),
      problem_.dims.n_omega,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        scratch.data()[index.cell(column, omega)] *=
            -0.5 * spatial.absorber(column / n_pol);
      });
  transverse_forward();
  Kokkos::Profiling::popRegion();
}

// The separable stage's core (solprop SEP 24 Q5): from its input in the transverse
// scratch, the material response (or, `material` false, the in-stage absorber's term)
// back into the scratch, in frequency, chunk by chunk through the temporal transforms.
// Shared by the Runge-Kutta stage and the low-storage march's passes, which differ in
// how they form the input and fold the output.
void GeneralWorkspace::separable_core(const std::size_t stage, const bool material) {
  if (material && volume_kerr_) {
    separable_volume_core(stage);
    return;
  }
  if (!material && spatial_only_absorber_) {
    separable_spatial_core();
    return;
  }
  // SEP 6's dispersive response runs chunk by chunk beside the material's response
  // column, or alone in its place (`SeparableDispersive`).
  auto* const dispersive = separable_dispersive_.get();
  const bool alone = dispersive != nullptr && dispersive->alone();
  const auto n_pol = problem_.dims.n_pol_state;
  // The host's scratch positions without a division (`RegisterIndex`).
  [[maybe_unused]] const auto index = register_index();
  [[maybe_unused]] const auto layout = transverse_layout();
  const auto n = problem_.dims.n_omega;
  const auto n_to = problem_.dims.n_to;
  const auto real_time = modal_real_;
  const auto complex_time = modal_complex_;
  const auto polarisation_real = polarisation_real_;
  const auto polarisation_complex = polarisation_complex_;
  const auto projected_real = projected_real_;
  const auto projected_complex = projected_complex_;
  const auto temporal_rate = absorber_fine_;
  const auto spatial = spatial_;
  const auto spectrum = projected_spectrum_;
  const auto crop = crop_index_;
  const auto scratch = transverse_;
  const bool real_field = problem_.dims.family == FieldFamily::field_real;
  const bool envelope = problem_.dims.family == FieldFamily::envelope;
  const bool slab = problem_.slab_geometry();
  // Profiling regions name the core's stages for a Kokkos Tools timer (SEP 28 phase
  // 0); with no tool loaded each is one test of a flag. The kernels inside carry
  // their own labels.
  Kokkos::Profiling::pushRegion("solprop_native.free.core");
  if (material && dispersive != nullptr)
    dispersive->begin(stage);
  Kokkos::Profiling::pushRegion("solprop_native.free.transverse_inverse");
  transverse_inverse();
  Kokkos::Profiling::popRegion();
  for (std::size_t unit = 0; unit < transverse_units();
       unit += transverse_units_per_chunk()) {
    const auto chunk = transverse_chunk(unit);
    const auto first = chunk.first_row;
    Kokkos::Profiling::pushRegion("solprop_native.free.pack");
    separable_pack_chunk(chunk, material);
    Kokkos::Profiling::popRegion();
    // The response filters the packed spectrum first: the inverse transform destroys
    // it. In place, that transform gives the filtered field.
    if (material && dispersive != nullptr)
      dispersive->prepare(chunk.nodes);
    Kokkos::Profiling::pushRegion("solprop_native.free.to_time");
    separable_to_time(chunk);
    Kokkos::Profiling::popRegion();
    // The projected rows are overwritten below by the polarisation (or are the
    // polarisation, aliased) or by the absorber's term; only a planar projection or a
    // response that projects nothing of the stage's needs them zeroed first.
    const bool overwrites =
        !slab &&
        (!material || !alone || (dispersive != nullptr && dispersive->borrowed()));
    Kokkos::Profiling::pushRegion("solprop_native.free.clear_projected");
    if (!overwrites) {
      if (envelope)
        graph::deep_copy(execution_space_, projected_complex_, Complex{});
      else
        graph::deep_copy(execution_space_, projected_real_, 0.0);
    }
    Kokkos::Profiling::popRegion();
    if (material) {
      separable_nodes(chunk);
      Kokkos::Profiling::pushRegion("solprop_native.free.response");
      if (!alone)
        response(stage, chunk.first_node, chunk.nodes);
      if (problem_.plasma.envelope)
        envelope_response(stage, chunk.first_node, chunk.nodes, &chunk);
      if (problem_.raman.enabled)
        convolve_raman(stage, chunk.nodes);
      if (dispersive != nullptr)
        dispersive->respond(chunk.nodes);
      if (held_plasma())
        held_response(stage, chunk.first_node, chunk.nodes, &chunk);
      if (gated_plasma())
        gated_response(chunk.nodes);
      Kokkos::Profiling::popRegion();
      // Alone with products of its own, nothing of the stage's is projected.
      const bool stage_projects = !alone || dispersive->borrowed();
      if (stage_projects && slab)
        slab_project(chunk, nullptr);
      else if (stage_projects && !separable_aliased_)
        cartesian_for(
            "solprop_native.free.polarisation_columns", execution_space_,
            chunk.rows * n_pol, n_to,
            KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
              if (envelope)
                projected_complex(row, time) = polarisation_complex(row, time);
              else
                projected_real(row, time) = polarisation_real(row, time);
            });
    } else {
      // The collar's rate depends on y alone on the planar kind, so it acts on the
      // rows, whose time signals are the slab amplitudes' (solprop's reference).
      cartesian_for(
          "solprop_native.free.absorber_columns", execution_space_, chunk.rows * n_pol,
          n_to, KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
            const double rate =
                -0.5 * (spatial.absorber(first + row / n_pol) + temporal_rate(time));
            if (envelope)
              projected_complex(row, time) = rate * complex_time(row, time);
            else
              projected_real(row, time) =
                  rate *
                  (real_field ? real_time(row, time) : complex_time(row, time).real());
          });
    }
    const bool responds = material && dispersive != nullptr;
    if (responds)
      dispersive->finish();
    Kokkos::Profiling::pushRegion("solprop_native.free.to_frequency");
    if (!responds || !dispersive->replaces_transform()) {
      separable_to_frequency(chunk);
    }
    Kokkos::Profiling::popRegion();
    // The response's spectrum joins the chunk's, before the envelope currents'.
    if (responds)
      dispersive->combine();
    if (material && problem_.plasma.envelope)
      envelope_transform();
    if (material && held_plasma())
      held_transform();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    tiled_crossing(
        "solprop_native.free.crop_temporal_chunk", execution_space_, chunk.rows * n_pol,
        n, false,
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
          return spectrum(row, crop(omega));
        },
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega,
                      const Complex value) {
          layout.cell(scratch, omega * n_pol + row % n_pol, first + row / n_pol) =
              value;
        });
#else
    const auto first_column = first * n_pol;
    crossing_for<true>(
        "solprop_native.free.crop_temporal_chunk", execution_space_, chunk.rows * n_pol,
        n, KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
          stream_store(scratch.data()[index.cell(first_column + row, omega)],
                       spectrum(row, crop(omega)));
        });
#endif
  }
  Kokkos::Profiling::pushRegion("solprop_native.free.transverse_forward");
  transverse_forward();
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

// The fine bins outside the crop as runs `(first bin, length)` in bin order.
std::vector<std::pair<std::size_t, std::size_t>>
GeneralWorkspace::fine_gap_runs() const {
  std::vector<bool> filled(fine_spectrum_size(), false);
  for (const auto index : problem_.grid.crop_index)
    filled[index] = true;
  std::vector<std::pair<std::size_t, std::size_t>> runs;
  for (std::size_t index = 0; index < filled.size(); ++index) {
    if (filled[index])
      continue;
    if (!runs.empty() && runs.back().first + runs.back().second == index)
      ++runs.back().second;
    else
      runs.emplace_back(index, 1);
  }
  return runs;
}

StageTerms GeneralWorkspace::stage_terms(const std::size_t stage) const {
  StageTerms terms;
  terms.current = current_;
  terms.fine = fine_spectrum_;
  terms.whole = whole_propagator_;
  terms.solution = solution_;
  terms.error = error_;
  terms.stage_phase = stage_phase_;
  terms.stage_inverse = stage_inverse_;
  terms.noise = noise_;
  terms.flow = static_cast<bool>(flow_);
  terms.carried = terms.flow && !flow_->identity(stage);
  terms.keep_lab = terms.flow && flow_->residual();
  if (terms.flow) {
    terms.flow_input = flow_->input();
    terms.flow_drive = flow_->drive();
  }
  if (terms.keep_lab)
    terms.flow_lab = flow_->lab();
  terms.general = static_cast<bool>(general_);
  if (terms.general) {
    terms.general_scaled = general_->scaled();
    terms.general_source = general_->source();
  }
  terms.projected = projected_spectrum_;
  terms.absorber = absorber_spectrum_;
  terms.stages = stages_;
  terms.phi_terms = phi_terms_;
  terms.prefactor_terms = prefactor_terms_;
  terms.tableau = tableau_;
  terms.phi_delta = stage_phi_delta_;
  terms.phi_at = stage_phi_at_;
  terms.prefactor_weights = stage_prefactor_;
  terms.window = omega_window_;
  terms.crop = crop_index_;
  terms.gaps = fine_gaps_;
  terms.gap_ranges = fine_gaps_.extent(0) / 3;
  terms.active = active_mask_;
  terms.n = problem_.dims.n_omega;
  terms.n_phi = problem_.linear.n_phi_terms;
  terms.n_stages = problem_.stepper.n_stages;
  terms.n_prefactor = problem_.linear.n_prefactor_terms;
  terms.n_to = problem_.dims.n_to;
  terms.inverse_n_to = 1.0 / static_cast<double>(problem_.dims.n_to);
  terms.general_pad = problem_.general.pad_scale;
  terms.add_noise = problem_.stepper.modified_noise && has_noise_;
  terms.analytic = problem_.dims.family == FieldFamily::field_analytic;
  const bool general_active =
      terms.general && (general_->responds() || problem_.absorber.in_stage);
  terms.zero_response =
      terms.general ? !general_active
                    : !problem_.kerr.nonlinear && !problem_.plasma.enabled &&
                          !problem_.raman.enabled && !absorbing() && !dispersive_;
  terms.in_stage_absorber = absorbing();
  terms.identity_crop = identity_crop_;
  return terms;
}

// The stage's entry kernel: the deferred finish of the stage before it (when a trial
// left one pending), the stage's phase increment and the whole step's at the last
// stage, the combination, and the zero pad of the fine spectrum, in one traversal of
// the bins. The finish and the combination are each the arithmetic of their own
// kernels, so fusing them moves no rounding. A lossless problem's kernel makes no
// `exp` calls, and a lossless stage repeating the stage before's phase (the last two
// stages of a FSAL tableau, and the whole step's at the last) reuses its pair
// (`StageTerms::combine`).
void GeneralWorkspace::stage_entry(const std::size_t stage, const double dz) {
  const bool repeated = lossless_phase_ && stage > 0 && entered_stage_ == stage - 1 &&
                        phase_repeats_[stage];
  entered_stage_ = stage;
  if (repeated)
    stage_entry_with<EntryPhase::repeated>(stage, dz);
  else if (lossless_phase_)
    stage_entry_with<EntryPhase::lossless>(stage, dz);
  else
    stage_entry_with<EntryPhase::general>(stage, dz);
}

template <EntryPhase Mode>
void GeneralWorkspace::stage_entry_with(const std::size_t stage, const double dz) {
  constexpr bool lossless = Mode != EntryPhase::general;
  const bool finish_previous = finish_pending_;
  finish_pending_ = false;
  if (finish_previous && (stage == 0 || pending_stage_ + 1 != stage))
    throw std::logic_error("a deferred stage finish lost its order");
  const std::size_t previous = finish_previous ? stage - 1 : 0;
  const double previous_solution = problem_.stepper.solution_weights[previous];
  const double previous_error = problem_.stepper.error_weights[previous];
  const double solution_weight = problem_.stepper.solution_weights[stage];
  const double error_weight = problem_.stepper.error_weights[stage];
  const auto terms = stage_terms(stage);
  const auto n = problem_.dims.n_omega;
  const auto n_phi = problem_.linear.n_phi_terms;
  const auto n_stages = problem_.stepper.n_stages;
  const auto gaps = problem_.separable_geometry() ? 0 : fine_spectrum_size() - n;
  const bool last = stage + 1 == n_stages;
  // The whole step's propagator is the last stage's when their phases are equal.
  const bool reuse_whole = last && lossless && whole_repeats_;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // On a device, one thread per bin: the host layout's blocks of 64 bins, each a work
  // item holding two 64-element arrays, gave a GPU a few dozen threads walking every
  // term, which made a varying capillary no faster on a GPU than on a CPU.
  launch::parallel_for(
      "solprop_native.stage.entry",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        if (finish_previous)
          terms.finish(previous, previous_solution, previous_error, column, index);
        const Complex phase = terms.phase_sum(stage * n_phi, column, index);
        const Complex propagator = terms.template combine<Mode>(
            stage, dz, solution_weight, error_weight, column, index, phase,
            terms.template entry_factors<Mode>(stage, column, index, phase));
        if (reuse_whole) {
          terms.whole(column, index) = propagator;
        } else if (last) {
          // At a last stage whose node is the step's end (DOPRI5's) the phase sums are
          // equal on a lossy problem too, and the stage's propagator is the step's.
          const Complex whole = terms.phase_sum(n_stages * n_phi, column, index);
          terms.whole(column, index) =
              whole == phase ? propagator : entry_exponential<lossless>(whole);
        }
        for (std::size_t gap = index; gap < gaps; gap += n)
          terms.zero_gaps(column, gap, gap + 1);
      });
#else
  const std::size_t blocks = (n + stage_block - 1) / stage_block;
  const bool block_forms = terms.blocks_vectorise();
  launch::parallel_for(
      "solprop_native.stage.entry",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0,
                                          problem_.n_columns() * blocks),
      KOKKOS_LAMBDA(const std::size_t item) {
        const std::size_t column = item / blocks;
        const std::size_t block = item % blocks;
        const std::size_t first = block * stage_block;
        const std::size_t count = n - first < stage_block ? n - first : stage_block;
        if (finish_previous && block_forms)
          terms.finish_block(previous, previous_solution, previous_error, column, first,
                             count);
        else if (finish_previous)
          for (std::size_t bin = 0; bin < count; ++bin)
            terms.finish(previous, previous_solution, previous_error, column,
                         first + bin);
        double real[stage_block];
        double imag[stage_block];
        terms.phase_block(stage * n_phi, column, first, count, real, imag);
        // With vector maths the block's transcendentals are passes of the layer; with
        // the system libm each bin keeps its own calls, which measured faster there
        // (the passes' buffers cost 1 to 3 % on an AMD EPYC 7763 with glibc's).
        if constexpr (vmath::vectorised) {
          EntryBlock factors{};
          terms.template entry_factors_block<Mode>(stage, column, first, count, real,
                                                   imag, factors);
          if (block_forms)
            terms.template combine_block<Mode>(stage, dz, column, first, count, factors,
                                               reuse_whole);
          else
            for (std::size_t bin = 0; bin < count; ++bin) {
              const Complex propagator = terms.template combine<Mode>(
                  stage, dz, solution_weight, error_weight, column, first + bin,
                  Complex(real[bin], imag[bin]), factors.at(bin));
              if (reuse_whole)
                terms.whole(column, first + bin) = propagator;
            }
          if (last && !reuse_whole) {
            // The whole step's propagator, `entry_exponential` over the block.
            terms.phase_block(n_stages * n_phi, column, first, count, real, imag);
            // `e^{+-0} (cos b, sin b)` is `(cos b, sin b)` to the bit.
            if (lossless || StageTerms::lossless_block(real, count))
              vmath::sincos(imag, factors.sine, factors.cosine, count);
            else
              vmath::exp_complex(real, imag, factors.cosine, factors.sine, count);
            for (std::size_t bin = 0; bin < count; ++bin)
              terms.whole(column, first + bin) =
                  Complex(factors.cosine[bin], factors.sine[bin]);
          }
        } else {
          for (std::size_t bin = 0; bin < count; ++bin) {
            const Complex phase(real[bin], imag[bin]);
            const Complex propagator = terms.template combine<Mode>(
                stage, dz, solution_weight, error_weight, column, first + bin, phase,
                terms.template entry_factors<Mode>(stage, column, first + bin, phase));
            if (reuse_whole)
              terms.whole(column, first + bin) = propagator;
          }
          if (last && !reuse_whole) {
            terms.phase_block(n_stages * n_phi, column, first, count, real, imag);
            for (std::size_t bin = 0; bin < count; ++bin)
              terms.whole(column, first + bin) =
                  entry_exponential<lossless>(Complex(real[bin], imag[bin]));
          }
        }
        terms.zero_gaps(column, block * gaps / blocks, (block + 1) * gaps / blocks);
      });
#endif
}

// A stage's finish, or (inside a trial, when the next stage's entry follows with
// nothing between) its deferral to that entry kernel.
void GeneralWorkspace::finish_or_defer(const std::size_t stage) {
  if (defer_finish_) {
    finish_pending_ = true;
    pending_stage_ = stage;
    return;
  }
  finish_stage(stage);
}

void GeneralWorkspace::evaluate_stage(const std::size_t stage, const double dz) {
  ++rk_source_evaluations_;
  if (!flow_) {
    evaluate_stage_source(stage, dz);
    return;
  }
  // solprop `NumpyRefBackend._trial_step` on a coupled flow: the stage input is
  // carried forward by the stage's action, the source (with a frozen-residual flow's
  // explicit residual) is evaluated into the flow's drive, and carried back.
  if (!flow_->identity(stage))
    flow_->carry_input(stage, dz, current_, stages_, tableau_);
  evaluate_stage_source(stage, dz);
  flow_->add_residual(stage, stage_phase_);
  flow_->carry_back(stage, problem_.stepper.solution_weights[stage],
                    problem_.stepper.error_weights[stage], stages_, solution_, error_);
}

void GeneralWorkspace::evaluate_stage_source(const std::size_t stage, const double dz,
                                             const bool accumulate) {
  if (!defer_quadrature_)
    coarse_pending_ = false;
  if (problem_.separable_geometry()) {
    evaluate_separable_stage(stage, dz, accumulate);
    return;
  }
  stage_entry(stage, dz);
  if (modal_core(stage))
    finish_or_defer(stage);
}

// The modal routes' stage core (solprop SEP 24 Q5): from the input in the fine spectrum
// (and the general route's scaled channels) to the source in the projected spectra,
// the in-stage absorber's beside it. Shared by the Runge-Kutta stage and the
// low-storage march's passes. Returns whether a source was evaluated: a stage without
// a response has none to finish.
bool GeneralWorkspace::modal_core(const std::size_t stage) {
  if (general_) {
    const bool general_active = general_->responds() || problem_.absorber.in_stage;
    if (general_active)
      general_core(stage);
    return general_active;
  }
  if (!problem_.kerr.nonlinear && !problem_.plasma.enabled && !problem_.raman.enabled &&
      !absorbing() && !dispersive_)
    return false;
  // A device's forward rows go in one transform after the absorber's are formed.
  const bool packed = !forward_pack_.empty();
  // SEP 6's dispersive response filters the padded spectrum first: the inverse
  // transform destroys it. In place, that transform gives the filtered field.
  if (dispersive_)
    dispersive_->prepare(stage);
  {
    const Kokkos::Profiling::ScopedRegion region("solprop_native.fft.inverse");
    if (problem_.dims.family == FieldFamily::field_real)
      graph::execute(*inverse_real_plan_, fine_spectrum_, modal_real_,
                     KokkosFFT::Normalization::none);
    else if (!retained_synthesis_)
      graph::execute(*inverse_complex_plan_, fine_spectrum_, modal_complex_,
                     KokkosFFT::Normalization::none);
  }
  // A single node's synthesis and projection are folded into its response column.
  const bool folded = folds_single_node();
  if (!folded)
    scale_transform(stage, trial_taper_[stage]);
  // Alone, the response is the stage's only source: no response column runs, and
  // unless its product is the stage's polarisation nothing is projected or
  // transformed but its own.
  const bool alone = dispersive_ && dispersive_->alone();
  const bool stage_projects = !alone || dispersive_->borrowed();
  const bool stage_synthesises =
      !folded && (!dispersive_ || !dispersive_->synthesises_nodes());
  // A projection writes every sample of its targets at the first chunk (a GEMM with
  // a zero beta, the custom kernel by assignment, the folded column by its own
  // product), so only a stage that does not project needs them zeroed.
  if (!stage_projects) {
    if (problem_.dims.family == FieldFamily::envelope) {
      graph::deep_copy(execution_space_, projected_complex_, Complex{});
      if (!projected_coarse_complex_.empty())
        graph::deep_copy(execution_space_, projected_coarse_complex_, Complex{});
    } else {
      graph::deep_copy(execution_space_, projected_real_, 0.0);
      if (!projected_coarse_real_.empty())
        graph::deep_copy(execution_space_, projected_coarse_real_, 0.0);
    }
  }
  if (dispersive_)
    dispersive_->begin();
  if (raman_products_ != 0)
    convolve_raman_products();
  bool first_chunk = true;
  std::size_t chunk = 0;
  for (std::size_t first_node = 0; first_node < problem_.dims.n_nodes;
       first_node += problem_.dims.node_chunk) {
    const auto node_count =
        std::min(problem_.dims.node_chunk, problem_.dims.n_nodes - first_node);
    if (stage_synthesises)
      synthesise(first_node, node_count);
    if (!alone)
      response(stage, first_node, node_count);
    if (raman_products_ != 0)
      finish_raman_products(stage, first_node, node_count);
    else if (problem_.raman.enabled)
      convolve_raman(stage, node_count);
    if (dispersive_)
      dispersive_->respond(chunk, first_node, node_count);
    if (held_plasma()) {
      const Kokkos::Profiling::ScopedRegion region("solprop_native.held.response");
      held_response(stage, first_node, node_count);
    }
    if (gated_plasma())
      gated_response(node_count);
    if (stage_projects && !folded && !held_radial()) {
      const Kokkos::Profiling::ScopedRegion region("solprop_native.projection");
      project(first_node, node_count, first_chunk);
    }
    if (problem_.plasma.envelope)
      envelope_response(stage, first_node, node_count);
    first_chunk = false;
    ++chunk;
  }
  if (dispersive_)
    dispersive_->finish();
  if (!held_radial() && (!dispersive_ || !dispersive_->replaces_transform())) {
    const Kokkos::Profiling::ScopedRegion region("solprop_native.fft.forward");
    if (problem_.dims.family == FieldFamily::envelope) {
      graph::execute(*forward_complex_plan_, projected_complex_, projected_spectrum_,
                     KokkosFFT::Normalization::none);
      if (forward_coarse_complex_plan_ && !defer_quadrature_)
        graph::execute(*forward_coarse_complex_plan_, projected_coarse_complex_,
                       projected_coarse_spectrum_, KokkosFFT::Normalization::none);
    } else if (!packed) {
      graph::execute(*forward_real_plan_, projected_real_, projected_spectrum_,
                     KokkosFFT::Normalization::none);
      if (forward_coarse_real_plan_ && !defer_quadrature_)
        graph::execute(*forward_coarse_real_plan_, projected_coarse_real_,
                       projected_coarse_spectrum_, KokkosFFT::Normalization::none);
    }
  }
  // The response's spectrum joins the stage's, before the envelope currents'.
  if (dispersive_)
    dispersive_->combine();
  if (absorbing()) {
    form_absorber();
    const Kokkos::Profiling::ScopedRegion region("solprop_native.fft.absorber");
    if (problem_.dims.family == FieldFamily::envelope)
      graph::execute(*absorber_complex_plan_, absorber_complex_, absorber_spectrum_,
                     KokkosFFT::Normalization::none);
    else if (!packed)
      graph::execute(*absorber_real_plan_, absorber_real_, absorber_spectrum_,
                     KokkosFFT::Normalization::none);
  }
  if (packed)
    transform_forward_pack();
  if (problem_.plasma.envelope)
    envelope_transform();
  if (held_plasma() && !held_radial())
    held_transform(false, packed);
  if (defer_quadrature_ && project_coarse_)
    coarse_pending_ = true;
  return true;
}

// The general modal route's stage core: the shared lab-frame input, the in-stage
// absorber on the modal time field as on every route, and the general source in place
// of the legacy node pipeline; the finish applies the shared normalisation and the
// return to the interaction picture (solprop `NumpyRefBackend._nonlinear_general`).
void GeneralWorkspace::general_core(const std::size_t stage) {
  if (problem_.absorber.in_stage) {
    graph::execute(*inverse_real_plan_, fine_spectrum_, modal_real_,
                   KokkosFFT::Normalization::none);
    form_absorber();
    graph::execute(*absorber_real_plan_, absorber_real_, absorber_spectrum_,
                   KokkosFFT::Normalization::none);
  }
  // A guide whose every region is linear has an identically zero source, which
  // solprop's reference short-circuits in the same way.
  if (general_->responds()) {
    general_->combine(stage);
    if (general_->dispersive() != nullptr)
      general_->dispersive()->begin(stage);
    general_->evaluate();
  } else {
    graph::deep_copy(execution_space_, general_->source(), Complex{});
  }
}

// The transforms scaled by stage `stage`'s taper, whose value `taper` the host holds.
// They depend on that value alone: a stage whose taper equals the last one scaled (a
// guide without a taper has one throughout) keeps them, the same bits.
void GeneralWorkspace::scale_transform(const std::size_t stage,
                                       const double taper_value) {
  if (scaled_taper_ready_ && taper_value == scaled_taper_)
    return;
  scaled_taper_ready_ = true;
  scaled_taper_ = taper_value;
  const auto synthesis = synthesis_;
  const auto projection = projection_;
  const auto projection_coarse = projection_coarse_;
  const auto scaled_synthesis = scaled_synthesis_;
  const auto scaled_projection = scaled_projection_;
  const auto scaled_projection_coarse = scaled_projection_coarse_;
  const auto scaled_projection_stacked = scaled_projection_stacked_;
  const auto scaled_synthesis_complex = scaled_synthesis_complex_;
  const auto scaled_projection_complex = scaled_projection_complex_;
  const auto scaled_projection_coarse_complex = scaled_projection_coarse_complex_;
  const auto scaled_projection_stacked_complex = scaled_projection_stacked_complex_;
  const auto taper = stage_taper_;
  const auto rows = problem_.dims.n_nodes * problem_.dims.n_pol_transform;
  const auto modes = problem_.dims.n_modes;
  launch::parallel_for(
      "solprop_native.transform.scale",
      Range2(execution_space_, {0, 0}, {rows, problem_.dims.n_modes}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t mode) {
        const double scale = taper(stage);
        scaled_synthesis(row, mode) = synthesis(row, mode) * scale;
        scaled_projection(mode, row) = projection(mode, row) / scale;
        if (!scaled_synthesis_complex.empty()) {
          scaled_synthesis_complex(row, mode) =
              Complex(synthesis(row, mode) * scale, 0.0);
          scaled_projection_complex(mode, row) =
              Complex(projection(mode, row) / scale, 0.0);
        }
        if (!scaled_projection_coarse.empty())
          scaled_projection_coarse(mode, row) = projection_coarse(mode, row) / scale;
        if (!scaled_projection_coarse_complex.empty())
          scaled_projection_coarse_complex(mode, row) =
              Complex(projection_coarse(mode, row) / scale, 0.0);
        if (!scaled_projection_stacked.empty()) {
          scaled_projection_stacked(mode, row) = projection(mode, row) / scale;
          scaled_projection_stacked(modes + mode, row) =
              projection_coarse(mode, row) / scale;
        }
        if (!scaled_projection_stacked_complex.empty()) {
          scaled_projection_stacked_complex(mode, row) =
              Complex(projection(mode, row) / scale, 0.0);
          scaled_projection_stacked_complex(modes + mode, row) =
              Complex(projection_coarse(mode, row) / scale, 0.0);
        }
      });
}

void GeneralWorkspace::synthesise(const std::size_t first_node,
                                  const std::size_t node_count) {
  if (retained_synthesis_) {
    synthesise_retained(first_node, node_count);
    return;
  }
  const bool whole_transform = first_node == 0 && node_count == problem_.dims.n_nodes &&
                               problem_.dims.n_pol_state == 1;
  if (whole_transform && !radial_blas_input_.empty()) {
    packed_radial_gemm(scaled_synthesis_, modal_complex_, node_complex_);
    return;
  }
  // A radial analytic field can exceed the node budget even at 256 nodes. Its
  // row chunks are contiguous in LayoutRight, including the short final chunk:
  // alias those rows directly instead of falling through to scalar synthesis. A
  // device takes the same GEMM wherever its shape guard admits the interleaved
  // synthesis, which `synthesis_provider` reports.
  const bool radial_analytic_chunk =
      problem_.dims.family == FieldFamily::field_analytic &&
      !problem_.spatial.spectral_weights.empty();
  if ((whole_transform || radial_analytic_chunk) && use_interleaved_synthesis()) {
    Kokkos::Profiling::pushRegion(
        "solprop_native.transform.synthesis.kokkosblas_interleaved");
    using Interleaved = Kokkos::View<double**, Layout, MemorySpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // Complex samples are pairs of doubles, real part first: the same rows read as
    // twice as many real columns.
    static_assert(sizeof(Complex) == 2 * sizeof(double));
    const Interleaved modal(reinterpret_cast<double*>(modal_complex_.data()),
                            problem_.dims.n_modes, 2 * problem_.dims.n_to);
    const Interleaved synthesis(scaled_synthesis_.data() +
                                    first_node * problem_.dims.n_modes,
                                node_count, problem_.dims.n_modes);
    const Interleaved node(reinterpret_cast<double*>(node_complex_.data()), node_count,
                           2 * problem_.dims.n_to);
    engine_gemm(transform_space_for(node_count), synthesis, modal, node, 0.0);
    Kokkos::Profiling::popRegion();
    return;
  }
  if (whole_transform && use_blas_for_synthesis()) {
    Kokkos::Profiling::pushRegion("solprop_native.transform.synthesis.kokkosblas");
    if (problem_.dims.family == FieldFamily::envelope)
      selected_gemm(transform_space_for(node_count), scaled_synthesis_complex_,
                    modal_complex_, node_complex_, Complex{}, true);
    else if (one_mode_products_)
      one_mode_synthesis(response_space_for(node_count), scaled_synthesis_, modal_real_,
                         node_real_);
    else
      selected_gemm(transform_space_for(node_count), scaled_synthesis_, modal_real_,
                    node_real_, 0.0, true);
    Kokkos::Profiling::popRegion();
    return;
  }
  if (use_chunked_blas_for_synthesis()) {
    Kokkos::Profiling::pushRegion(
        "solprop_native.transform.synthesis.kokkosblas_chunked");
    const auto rows = node_count * problem_.dims.n_pol_transform;
    const auto first_row = first_node * problem_.dims.n_pol_transform;
    const bool tail = node_count != problem_.dims.node_chunk;
    if (problem_.dims.family == FieldFamily::envelope) {
      const auto packed = tail ? tail_synthesis_complex_ : chunk_synthesis_complex_;
      const auto source = scaled_synthesis_complex_;
      launch::parallel_for(
          "solprop_native.transform.pack_synthesis_complex",
          Range2(matrix_copy_space_for(node_count), {0, 0},
                 {rows, problem_.dims.n_modes}),
          KOKKOS_LAMBDA(const std::size_t row, const std::size_t mode) {
            packed(row, mode) = source(first_row + row, mode);
          });
      const auto output = Kokkos::subview(
          node_complex_, std::make_pair(std::size_t{0}, rows), Kokkos::ALL());
      selected_gemm(transform_space_for(node_count), packed, modal_complex_, output,
                    Complex{}, true);
    } else {
      const auto packed = tail ? tail_synthesis_ : chunk_synthesis_;
      const auto source = scaled_synthesis_;
      launch::parallel_for(
          "solprop_native.transform.pack_synthesis",
          Range2(matrix_copy_space_for(node_count), {0, 0},
                 {rows, problem_.dims.n_modes}),
          KOKKOS_LAMBDA(const std::size_t row, const std::size_t mode) {
            packed(row, mode) = source(first_row + row, mode);
          });
      const auto output = Kokkos::subview(
          node_real_, std::make_pair(std::size_t{0}, rows), Kokkos::ALL());
      selected_gemm(transform_space_for(node_count), packed, modal_real_, output, 0.0,
                    true);
    }
    Kokkos::Profiling::popRegion();
    return;
  }
  Kokkos::Profiling::pushRegion("solprop_native.transform.synthesis.custom");
  const auto synthesis = scaled_synthesis_;
  const auto modal_real = modal_real_;
  const auto modal_complex = modal_complex_;
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto n_modes = problem_.dims.n_modes;
  const auto n_pol_state = problem_.dims.n_pol_state;
  const auto n_pol_transform = problem_.dims.n_pol_transform;
  const auto n_to = problem_.dims.n_to;
  const bool real_family = problem_.dims.family == FieldFamily::field_real;
  const auto effective_rows = node_count * problem_.effective_node_pol();
  launch::parallel_for(
      "solprop_native.transform.synthesis.custom",
      Range2(transform_space_for(node_count), {0, 0}, {effective_rows, n_to}),
      KOKKOS_LAMBDA(const std::size_t local_row, const std::size_t time) {
        const std::size_t state_pol = local_row % n_pol_state;
        const std::size_t transform_local = local_row / n_pol_state;
        const std::size_t transform_row =
            first_node * n_pol_transform + transform_local;
        if (real_family) {
          double value = 0.0;
          for (std::size_t mode = 0; mode < n_modes; ++mode)
            value += synthesis(transform_row, mode) *
                     modal_real(mode * n_pol_state + state_pol, time);
          node_real(local_row, time) = value;
        } else {
          Complex value{};
          for (std::size_t mode = 0; mode < n_modes; ++mode)
            value += synthesis(transform_row, mode) *
                     modal_complex(mode * n_pol_state + state_pol, time);
          node_complex(local_row, time) = value;
        }
      });
  Kokkos::Profiling::popRegion();
}

template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
          bool Stash>
void GeneralWorkspace::response_harmonic(const std::size_t stage,
                                         const std::size_t first_node,
                                         const std::size_t node_count) {
  if constexpr (Nonlinear && Family != FieldFamily::field_real) {
    if (problem_.kerr.harmonic_factor != 0.0) {
      response_raman<Family, Components, Plasma, Nonlinear, Stash, true>(
          stage, first_node, node_count);
      return;
    }
  }
  response_raman<Family, Components, Plasma, Nonlinear, Stash, false>(stage, first_node,
                                                                      node_count);
}

template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
          bool Stash, bool Harmonic>
void GeneralWorkspace::response_raman(const std::size_t stage,
                                      const std::size_t first_node,
                                      const std::size_t node_count) {
  if (problem_.raman.enabled) {
    if (problem_.raman.thg) {
      response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, true, true,
                    false>(stage, first_node, node_count);
    } else {
      response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, true, false,
                    false>(stage, first_node, node_count);
    }
    return;
  }
  response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, false, false,
                false>(stage, first_node, node_count);
}

template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear>
void GeneralWorkspace::response_stash(const std::size_t stage,
                                      const std::size_t first_node,
                                      const std::size_t node_count) {
  if constexpr (Plasma) {
    // Clang's host loop is measurably faster when the maximum-rate accumulator is
    // present. Keep that loop shape for every plasma problem, while the view and its
    // final-stage stores remain conditional on requested periodic statistics.
    response_harmonic<Family, Components, Plasma, Nonlinear, true>(stage, first_node,
                                                                   node_count);
    return;
  }
  response_harmonic<Family, Components, Plasma, Nonlinear, false>(stage, first_node,
                                                                  node_count);
}

template <FieldFamily Family, std::size_t Components, bool Plasma>
void GeneralWorkspace::response_nonlinear(const std::size_t stage,
                                          const std::size_t first_node,
                                          const std::size_t node_count) {
  if (problem_.kerr.nonlinear) {
    response_stash<Family, Components, Plasma, true>(stage, first_node, node_count);
  } else {
    response_stash<Family, Components, Plasma, false>(stage, first_node, node_count);
  }
}

template <FieldFamily Family, std::size_t Components>
void GeneralWorkspace::response_plasma(const std::size_t stage,
                                       const std::size_t first_node,
                                       const std::size_t node_count) {
  if (problem_.plasma.enabled && !problem_.plasma.envelope) {
    if constexpr (Family != FieldFamily::envelope) {
      // SEP 27: the band column writes the physical current separately. The
      // ordinary traversal forms and collars only the Kerr/Raman polarisation.
      if (problem_.plasma.band) {
        if (band_tiled_) {
          if (problem_.plasma.n_channels > 1)
            tiled_band_plasma_column<Family, Components, true>(stage, first_node,
                                                               node_count);
          else
            tiled_band_plasma_column<Family, Components, false>(stage, first_node,
                                                                node_count);
        } else if (problem_.plasma.n_channels > 1) {
          team_band_plasma_column<Family, Components, true>(stage, first_node,
                                                            node_count);
        } else {
          team_band_plasma_column<Family, Components, false>(stage, first_node,
                                                             node_count);
        }
        response_nonlinear<Family, Components, false>(stage, first_node, node_count);
        return;
      }
    }
    response_nonlinear<Family, Components, true>(stage, first_node, node_count);
  } else {
    response_nonlinear<Family, Components, false>(stage, first_node, node_count);
  }
}

void GeneralWorkspace::response(const std::size_t stage, const std::size_t first_node,
                                const std::size_t node_count) {
  const auto components = problem_.effective_node_pol();
  switch (problem_.dims.family) {
  case FieldFamily::field_real:
    if (components == 1)
      response_plasma<FieldFamily::field_real, 1>(stage, first_node, node_count);
    else
      response_plasma<FieldFamily::field_real, 2>(stage, first_node, node_count);
    return;
  case FieldFamily::field_analytic:
    if (components == 1)
      response_plasma<FieldFamily::field_analytic, 1>(stage, first_node, node_count);
    else
      response_plasma<FieldFamily::field_analytic, 2>(stage, first_node, node_count);
    return;
  case FieldFamily::envelope:
    if (components == 1)
      response_plasma<FieldFamily::envelope, 1>(stage, first_node, node_count);
    else
      response_plasma<FieldFamily::envelope, 2>(stage, first_node, node_count);
  }
}

template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
          bool Stash, bool Harmonic, bool Raman, bool RamanThg, bool Multiple,
          bool Folded>
void GeneralWorkspace::response_impl(const std::size_t stage,
                                     const std::size_t first_node,
                                     const std::size_t node_count) {
  // A single node's transforms fold into the column (`folds_single_node`), compiled
  // apart so that the other columns' arithmetic and layout are untouched.
  if constexpr (Components == 1 && !Raman && !Folded) {
    if (folds_single_node()) {
      response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, Raman,
                    RamanThg, Multiple, true>(stage, first_node, node_count);
      return;
    }
  }
  if constexpr (!Multiple) {
    if (problem_.plasma.n_channels > 1 && Plasma) {
      response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, Raman,
                    RamanThg, true>(stage, first_node, node_count);
      return;
    }
  }
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto out_real = polarisation_real_;
  const auto held_current = held_current_;
  const auto out_complex = polarisation_complex_;
  const auto raman_drive = node_raman_drive_;
  const auto raman_drives = problem_.raman_drives();
  // The modal products replace the nodes' drives (`raman_modal_products`).
  const bool node_drives = raman_products_ == 0;
  const auto window = time_window_;
  const auto carrier = carrier_;
  const auto rate_coefficients = rate_coefficients_;
  const auto stash = stash_candidate_;
  const auto density = stage_plasma_density_;
  const auto kerr_density = stage_kerr_density_;
  const auto channels = problem_.plasma.n_channels;
  const auto channel_parameters = channel_parameters_;
  const auto channel_coefficients = channel_coefficients_;
  const auto channel_density = stage_channel_density_;
  const auto channel_preionisation = stage_channel_preionisation_;
  const auto scratch = channel_scratch_;
  const auto preionisation = stage_preionisation_;
  const auto n_to = problem_.dims.n_to;
  const double kerr_coefficient = problem_.kerr.coefficient;
  const double self_factor = problem_.kerr.self_factor;
  const double conjugate_factor = problem_.kerr.conjugate_factor;
  const double harmonic_factor = problem_.kerr.harmonic_factor;
  const double charge_mass_ratio = problem_.plasma.charge_mass_ratio;
  const double ionisation_potential = problem_.plasma.ionisation_potential;
  const double dt = problem_.grid.dt_fine;
  const double log_field_min = problem_.plasma.log_field_min;
  const double d_log_field = problem_.plasma.d_log_field;
  const double field_min = problem_.plasma.field_min;
  const double field_max = problem_.plasma.field_max;
  const double rate_max = problem_.plasma.rate_max;
  const auto rate_intervals = problem_.plasma.rate_intervals;
  const auto n_stages = problem_.stepper.n_stages;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // The device's form (`device_response`); a mixture of channels takes the generic
  // column below.
  if constexpr (!Multiple) {
    device_response<Family, Components, Plasma, Nonlinear, Stash, Harmonic, Raman,
                    RamanThg>(stage, first_node, node_count);
    return;
  }
#endif
  // A folded single node reads the modal field and writes the projected one, each
  // through its one-element transform (`folds_single_node`); otherwise both scales
  // are one, which changes no bit.
  constexpr bool folded = Folded;
  const auto input_real = folded ? modal_real_ : node_real_;
  const auto input_complex = folded ? modal_complex_ : node_complex_;
  const auto target_real = folded ? projected_real_ : polarisation_real_;
  const auto target_complex = folded ? projected_complex_ : polarisation_complex_;
  const auto synthesis = synthesis_;
  const auto projection = projection_;
  const auto taper = stage_taper_;
  launch::parallel_for(
      "solprop_native.response.column",
      Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0,
                                          node_count),
      KOKKOS_LAMBDA(const std::size_t node) {
        // CUDA extended-lambda restriction 12 requires the first capture outside
        // if-constexpr blocks. Reference binding keeps every compile-time branch.
        capture_anchor(channels, scratch, channel_preionisation, node_real,
                       node_complex, channel_coefficients, channel_parameters, dt,
                       channel_density, rate_coefficients, log_field_min, d_log_field,
                       field_min, field_max, rate_max, rate_intervals,
                       ionisation_potential, charge_mass_ratio, density,
                       conjugate_factor, self_factor, harmonic_factor, carrier,
                       raman_drive, raman_drives, node_drives, out_real, held_current,
                       window, out_complex, stash, n_stages, first_node);
        const std::size_t base = node * Components;
        const double input_scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
        const double output_scale = folded ? projection(0, 0) / taper(stage) : 1.0;
        double exposure = 0.0;
        double last_rate = 0.0;
        double largest_rate = 0.0;
        double fraction = Plasma && !Multiple ? preionisation(stage) : 0.0;
        double free_first = 0.0;
        double free_second = 0.0;
        double last_first = 0.0;
        double last_second = 0.0;
        if constexpr (Multiple) {
          for (std::size_t channel = 0; channel < channels; ++channel) {
            scratch(channel, node, 0) = 0.0;
            scratch(channel, node, 1) = 0.0;
            scratch(channel, node, 2) = channel_preionisation(stage, channel);
            scratch(channel, node, 3) = 0.0;
          }
        }
        // Cache contiguous pointwise addresses to avoid repeated View stride loads.
        // Plasma retains its measured recurrence-loop layout; applying
        // the same change there regresses HE plasma by 8-9% (kokkos-reopt report).
        auto* const real_input = input_real.data();
        auto* const complex_input = input_complex.data();
        auto* const real_target = target_real.data();
        auto* const complex_target = target_complex.data();
        auto* const window_data = window.data();
        const double pointwise_factor =
            Plasma ? 0.0 : kerr_coefficient * kerr_density(stage);
        for (std::size_t time = 0; time < n_to; ++time) {
          Complex first_value{};
          Complex second_value{};
          if constexpr (Family == FieldFamily::field_real) {
            first_value =
                Complex(input_scale * (Plasma ? input_real(base, time)
                                              : real_input[base * n_to + time]),
                        0.0);
            if constexpr (Components == 2)
              second_value =
                  Complex(input_scale * (Plasma ? input_real(base + 1, time)
                                                : real_input[(base + 1) * n_to + time]),
                          0.0);
          } else {
            first_value = input_scale * (Plasma ? input_complex(base, time)
                                                : complex_input[base * n_to + time]);
            if constexpr (Components == 2)
              second_value =
                  input_scale * (Plasma ? input_complex(base + 1, time)
                                        : complex_input[(base + 1) * n_to + time]);
          }
          const double modulus = squared_abs(first_value) +
                                 (Components == 2 ? squared_abs(second_value) : 0.0);
          const Complex square =
              first_value * first_value +
              (Components == 2 ? second_value * second_value : Complex{});
          if constexpr (Plasma) {
            // IntelLLVM's precise model retains an unused hypot outside this branch.
            const double magnitude =
                Components == 2 ? Kokkos::hypot(first_value.real(), second_value.real())
                                : Kokkos::abs(first_value.real());
            double guard = 0.0;
            double weight = 0.0;
            if constexpr (Multiple) {
              double work = 0.0;
              for (std::size_t channel = 0; channel < channels; ++channel) {
                const double rate = ionisation_rate(
                    magnitude, channel_coefficients, channel_parameters(channel, 0),
                    channel_parameters(channel, 1), channel_parameters(channel, 2),
                    channel_parameters(channel, 3), channel_parameters(channel, 4),
                    static_cast<std::size_t>(channel_parameters(channel, 5)),
                    static_cast<std::size_t>(channel_parameters(channel, 6)));
                double channel_exposure = scratch(channel, node, 0);
                const double previous_exposure = channel_exposure;
                if (time > 0)
                  channel_exposure += trapezoid(rate, scratch(channel, node, 1), dt);
                double channel_fraction = scratch(channel, node, 2);
                if (time == 0 || channel_exposure != previous_exposure)
                  channel_fraction = ionised_fraction(
                      channel_exposure, channel_preionisation(stage, channel));
                scratch(channel, node, 0) = channel_exposure;
                scratch(channel, node, 1) = rate;
                scratch(channel, node, 2) = channel_fraction;
                if (rate > scratch(channel, node, 3))
                  scratch(channel, node, 3) = rate;
                const double partial = channel_density(stage, channel);
                weight += partial * channel_fraction * channel_parameters(channel, 7);
                work += partial * channel_parameters(channel, 8) * rate *
                        (1.0 - channel_fraction);
              }
              guard = work > 0.0 && magnitude * magnitude > 0.0
                          ? work / (magnitude * magnitude)
                          : 0.0;
            } else {
              const double rate = ionisation_rate(magnitude, rate_coefficients,
                                                  log_field_min, d_log_field, field_min,
                                                  field_max, rate_max, rate_intervals);
              const double previous_exposure = exposure;
              if (time > 0)
                exposure += trapezoid(rate, last_rate, dt);
              last_rate = rate;
              if constexpr (Stash) {
                if (rate > largest_rate)
                  largest_rate = rate;
              }
              // Preserve the first subtraction's rounding even for pre-ionized
              // input; afterwards an unchanged rounded exposure has the same exp.
              if (time == 0 || exposure != previous_exposure)
                fraction = ionised_fraction(exposure, preionisation(stage));
              guard = rate > 0.0 && magnitude * magnitude > 0.0
                          ? ionisation_potential * rate * (1.0 - fraction) /
                                (magnitude * magnitude)
                          : 0.0;
              weight = fraction * charge_mass_ratio;
            }
            double integrand = weight * first_value.real();
            if (time > 0)
              free_first += trapezoid(integrand, last_first, dt);
            double current = free_first + guard * first_value.real();
            last_first = integrand;
            held_current(base, time) = Multiple ? current : density(stage) * current;
            if constexpr (Components == 2) {
              integrand = weight * second_value.real();
              if (time > 0)
                free_second += trapezoid(integrand, last_second, dt);
              current = free_second + guard * second_value.real();
              last_second = integrand;
              held_current(base + 1, time) =
                  Multiple ? current : density(stage) * current;
            }
          }
          const double factor =
              Plasma ? kerr_coefficient * kerr_density(stage) : pointwise_factor;
          Complex first{};
          Complex second{};
          if constexpr (Nonlinear) {
            if constexpr (Family == FieldFamily::field_real) {
              first += Complex(factor * modulus * first_value.real(), 0.0);
              if constexpr (Components == 2)
                second += Complex(factor * modulus * second_value.real(), 0.0);
            } else {
              const auto analytic = [&](const Complex value) {
                const double weighted_real = conjugate_factor * square.real();
                const double weighted_imag = conjugate_factor * square.imag();
                const double weighted_self = self_factor * modulus;
                Complex result(factor * (weighted_self * value.real() +
                                         (weighted_real * value.real() +
                                          weighted_imag * value.imag())),
                               factor * (weighted_self * value.imag() +
                                         (weighted_imag * value.real() -
                                          weighted_real * value.imag())));
                if constexpr (Harmonic)
                  result +=
                      ((factor * harmonic_factor) * carrier(time)) * square * value;
                return result;
              };
              first += analytic(first_value);
              if constexpr (Components == 2)
                second += analytic(second_value);
            }
          }
          if constexpr (Raman) {
            if (node_drives)
              write_raman_drive<Components, RamanThg>(
                  raman_drive, node, time, first_value, second_value, raman_drives);
          }
          if constexpr (Family != FieldFamily::envelope) {
            (Plasma ? target_real(base, time) : real_target[base * n_to + time]) =
                Raman ? first.real()
                      : output_scale * (first.real() *
                                        (Plasma ? window(time) : window_data[time]));
            if constexpr (Components == 2)
              (Plasma ? target_real(base + 1, time)
                      : real_target[(base + 1) * n_to + time]) =
                  Raman ? second.real()
                        : output_scale * (second.real() *
                                          (Plasma ? window(time) : window_data[time]));
          } else {
            (Plasma ? target_complex(base, time) : complex_target[base * n_to + time]) =
                Raman ? first
                      : output_scale *
                            (first * (Plasma ? window(time) : window_data[time]));
            if constexpr (Components == 2)
              (Plasma ? target_complex(base + 1, time)
                      : complex_target[(base + 1) * n_to + time]) =
                  Raman ? second
                        : output_scale *
                              (second * (Plasma ? window(time) : window_data[time]));
          }
        }
        if constexpr (Stash) {
          if (stash.empty() || stage + 1 != n_stages)
            return;
          if constexpr (Multiple) {
            for (std::size_t channel = 0; channel < channels; ++channel) {
              stash(first_node + node, 2 * channel) = scratch(channel, node, 2);
              stash(first_node + node, 2 * channel + 1) = scratch(channel, node, 3);
            }
          } else {
            stash(first_node + node, 0) = fraction;
            stash(first_node + node, 1) = largest_rate;
          }
        }
      });
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
// The device's response with no team tied to a node, so that a one-node fibre fills the
// device:
//
//   - the pointwise terms (Kerr, the harmonic, the Raman drive and the windowed
//     polarisation) as one two-dimensional range over (node, time), in the column's
//     arithmetic sample by sample;
//   - the trapezoid plasma as two-level scans over tiles of `plasma_tile_samples`:
//     each running integral is a kernel of one team per (row, tile) that sums the
//     tile's increments into its total (`plasma_tile_totals_`), then one per (row,
//     tile) that adds the row's earlier totals and scans the tile again from that
//     offset, forming the exposure's fraction or the current from each prefix.
template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
          bool Stash, bool Harmonic, bool Raman, bool RamanThg>
void GeneralWorkspace::device_response(const std::size_t stage,
                                       const std::size_t first_node,
                                       const std::size_t node_count) {
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto out_real = polarisation_real_;
  const auto out_complex = polarisation_complex_;
  const auto raman_drive = node_raman_drive_;
  const auto raman_drives = problem_.raman_drives();
  const auto window = time_window_;
  const auto carrier = carrier_;
  const auto kerr_density = stage_kerr_density_;
  const auto n_to = problem_.dims.n_to;
  const double kerr_coefficient = problem_.kerr.coefficient;
  const double self_factor = problem_.kerr.self_factor;
  const double conjugate_factor = problem_.kerr.conjugate_factor;
  const double harmonic_factor = problem_.kerr.harmonic_factor;
  if constexpr (Plasma)
    device_trapezoid_plasma<Family, Components, Stash>(stage, first_node, node_count);
  launch::parallel_for(
      Plasma ? "solprop_native.response.device_pointwise_after_plasma"
             : "solprop_native.response.device_pointwise",
      Range2(response_space_for(node_count), {0, 0}, {node_count, n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        capture_anchor(self_factor, conjugate_factor, node_real, node_complex, out_real,
                       out_complex, raman_drive, raman_drives, window, carrier,
                       kerr_density, kerr_coefficient, harmonic_factor, stage);
        const std::size_t base = node * Components;
        const auto value_at = [&](const std::size_t row) {
          if constexpr (Family == FieldFamily::field_real)
            return Complex(node_real(row, time), 0.0);
          else
            return Complex(node_complex(row, time));
        };
        const Complex first_value = value_at(base);
        Complex second_value{};
        if constexpr (Components == 2)
          second_value = value_at(base + 1);
        const double modulus = squared_abs(first_value) +
                               (Components == 2 ? squared_abs(second_value) : 0.0);
        const Complex square =
            first_value * first_value +
            (Components == 2 ? second_value * second_value : Complex{});
        const double factor = kerr_coefficient * kerr_density(stage);
        Complex first{};
        Complex second{};
        if constexpr (Nonlinear) {
          if constexpr (Family == FieldFamily::field_real) {
            first += Complex(factor * modulus * first_value.real(), 0.0);
            if constexpr (Components == 2)
              second += Complex(factor * modulus * second_value.real(), 0.0);
          } else {
            const auto analytic = [&](const Complex value) {
              const double weighted_real = conjugate_factor * square.real();
              const double weighted_imag = conjugate_factor * square.imag();
              const double weighted_self = self_factor * modulus;
              Complex result(
                  factor *
                      (weighted_self * value.real() +
                       (weighted_real * value.real() + weighted_imag * value.imag())),
                  factor *
                      (weighted_self * value.imag() +
                       (weighted_imag * value.real() - weighted_real * value.imag())));
              if constexpr (Harmonic)
                result += ((factor * harmonic_factor) * carrier(time)) * square * value;
              return result;
            };
            first += analytic(first_value);
            if constexpr (Components == 2)
              second += analytic(second_value);
          }
        }
        if constexpr (Raman)
          write_raman_drive<Components, RamanThg>(raman_drive, node, time, first_value,
                                                  second_value, raman_drives);
        if constexpr (Family != FieldFamily::envelope) {
          out_real(base, time) = Raman ? first.real() : first.real() * window(time);
          if constexpr (Components == 2)
            out_real(base + 1, time) =
                Raman ? second.real() : second.real() * window(time);
        } else {
          out_complex(base, time) = Raman ? first : first * window(time);
          if constexpr (Components == 2)
            out_complex(base + 1, time) = Raman ? second : second * window(time);
        }
      });
}

// The trapezoid plasma's two-level scans on a device (`device_response`). Rows of the
// rate, exposure and fraction views are the chunk's nodes; rows of the integrand and
// of the tile totals are its component rows.
template <FieldFamily Family, std::size_t Components, bool Stash>
void GeneralWorkspace::device_trapezoid_plasma(const std::size_t stage,
                                               const std::size_t first_node,
                                               const std::size_t node_count) {
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto held_current = held_current_;
  const auto rates = plasma_rate_;
  const auto exposures = plasma_exposure_;
  const auto fractions = plasma_fraction_;
  const auto integrands = plasma_current_integrand_;
  const auto totals = plasma_tile_totals_;
  const auto stash = stash_candidate_;
  const auto rate_coefficients = rate_coefficients_;
  const auto density = stage_plasma_density_;
  const auto preionisation = stage_preionisation_;
  const double charge_mass_ratio = problem_.plasma.charge_mass_ratio;
  const double ionisation_potential = problem_.plasma.ionisation_potential;
  const double dt = problem_.grid.dt_fine;
  const double log_field_min = problem_.plasma.log_field_min;
  const double d_log_field = problem_.plasma.d_log_field;
  const double field_min = problem_.plasma.field_min;
  const double field_max = problem_.plasma.field_max;
  const double rate_max = problem_.plasma.rate_max;
  const auto rate_intervals = problem_.plasma.rate_intervals;
  const auto n_to = problem_.dims.n_to;
  const auto samples = static_cast<int>(n_to);
  const auto width = static_cast<int>(plasma_tile_samples);
  const auto tiles = static_cast<int>(plasma_tiles_);
  const auto last = n_to - 1;
  const bool stashing =
      Stash && !stash.empty() && stage + 1 == problem_.stepper.n_stages;
  const std::size_t rows = node_count * Components;
  const auto team_size = static_cast<int>(response_team_size());
  // The field a row reads: a real row's value, or an analytic or envelope row's real
  // part.
  const auto raw = [=] KOKKOS_FUNCTION(const std::size_t row, const std::size_t time) {
    capture_anchor(node_real, node_complex);
    if constexpr (Family == FieldFamily::field_real)
      return node_real(row, time);
    else
      return node_complex(row, time).real();
  };

  // 1. The rates.
  launch::parallel_for(
      "solprop_native.response.device_plasma_rate",
      Range2(response_space_for(node_count), {0, 0}, {node_count, n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        const std::size_t base = node * Components;
        const double first = raw(base, time);
        double magnitude = 0.0;
        if constexpr (Components == 2)
          magnitude = Kokkos::hypot(first, raw(base + 1, time));
        else
          magnitude = Kokkos::abs(first);
        rates(node, time) =
            ionisation_rate(magnitude, rate_coefficients, log_field_min, d_log_field,
                            field_min, field_max, rate_max, rate_intervals);
      });
  // 2. The exposure: tile totals, then each tile's prefixes from its offset, and the
  // fraction from each.
  const auto exposure_local = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(rates, totals, samples, width, tiles, dt);
    const auto node = static_cast<std::size_t>(team.league_rank() / tiles);
    const int tile = team.league_rank() % tiles;
    const int first = tile * width;
    const int end = Kokkos::min(first + width, samples);
    const double total = team_trapezoid_total(
        team,
        [&](const int sample) { return rates(node, static_cast<std::size_t>(sample)); },
        first, end, dt);
    Kokkos::single(Kokkos::PerTeam(team),
                   [&] { totals(node, static_cast<std::size_t>(tile)) = total; });
  };
  launch::parallel_for("solprop_native.response.device_plasma_exposure_tiles",
                       TeamPolicy(response_space_for(node_count),
                                  static_cast<int>(node_count) * tiles, team_size),
                       exposure_local);
  const auto exposure_finish = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(rates, totals, exposures, fractions, preionisation, samples, width,
                   tiles, dt, stage);
    const auto node = static_cast<std::size_t>(team.league_rank() / tiles);
    const int tile = team.league_rank() % tiles;
    const int first = tile * width;
    const int end = Kokkos::min(first + width, samples);
    const double offset = team_tile_offset(team, totals, node, tile);
    team_trapezoid_finish(
        team,
        [&](const int sample) { return rates(node, static_cast<std::size_t>(sample)); },
        first, end, dt, offset,
        [&](const int sample, const double prefix) {
          const auto time = static_cast<std::size_t>(sample);
          exposures(node, time) = prefix;
          fractions(node, time) = ionised_fraction(prefix, preionisation(stage));
        });
  };
  launch::parallel_for("solprop_native.response.device_plasma_exposure",
                       TeamPolicy(response_space_for(node_count),
                                  static_cast<int>(node_count) * tiles, team_size),
                       exposure_finish);
  // 3. The free-current integrand, per component row.
  launch::parallel_for(
      "solprop_native.response.device_plasma_integrand",
      Range2(response_space_for(node_count), {0, 0}, {rows, n_to}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
        integrands(row, time) =
            fractions(row / Components, time) * charge_mass_ratio * raw(row, time);
      });
  // 4. The free current: tile totals, then each tile's prefixes from its offset and
  // the current from each.
  const auto current_local = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(integrands, totals, samples, width, tiles, dt);
    const auto row = static_cast<std::size_t>(team.league_rank() / tiles);
    const int tile = team.league_rank() % tiles;
    const int first = tile * width;
    const int end = Kokkos::min(first + width, samples);
    const double total = team_trapezoid_total(
        team,
        [&](const int sample) {
          return integrands(row, static_cast<std::size_t>(sample));
        },
        first, end, dt);
    Kokkos::single(Kokkos::PerTeam(team),
                   [&] { totals(row, static_cast<std::size_t>(tile)) = total; });
  };
  launch::parallel_for("solprop_native.response.device_plasma_current_tiles",
                       TeamPolicy(response_space_for(node_count),
                                  static_cast<int>(rows) * tiles, team_size),
                       current_local);
  const auto current_finish = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(integrands, totals, held_current, rates, fractions, density, samples,
                   width, tiles, dt, ionisation_potential, stage);
    const auto row = static_cast<std::size_t>(team.league_rank() / tiles);
    const int tile = team.league_rank() % tiles;
    const int first = tile * width;
    const int end = Kokkos::min(first + width, samples);
    const double offset = team_tile_offset(team, totals, row, tile);
    team_trapezoid_finish(
        team,
        [&](const int sample) {
          return integrands(row, static_cast<std::size_t>(sample));
        },
        first, end, dt, offset,
        [&](const int sample, const double prefix) {
          const auto time = static_cast<std::size_t>(sample);
          const std::size_t node = row / Components;
          const std::size_t base = node * Components;
          const double first_field = raw(base, time);
          const double second_field = Components == 2 ? raw(base + 1, time) : 0.0;
          const double magnitude_squared =
              first_field * first_field + second_field * second_field;
          const double rate = rates(node, time);
          const double guard = rate > 0.0 && magnitude_squared > 0.0
                                   ? ionisation_potential * rate *
                                         (1.0 - fractions(node, time)) /
                                         magnitude_squared
                                   : 0.0;
          held_current(row, time) = density(stage) * (prefix + guard * raw(row, time));
        });
  };
  launch::parallel_for("solprop_native.response.device_plasma_current",
                       TeamPolicy(response_space_for(node_count),
                                  static_cast<int>(rows) * tiles, team_size),
                       current_finish);
  // 5. The stash: the last sample's fraction and the largest rate, per node.
  if (stashing) {
    const auto peak = KOKKOS_LAMBDA(const TeamMember& team) {
      capture_anchor(rates, fractions, stash, samples, last, first_node);
      const auto node = static_cast<std::size_t>(team.league_rank());
      double largest = 0.0;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, samples),
          [&](const int sample, double& update) {
            const double rate = rates(node, static_cast<std::size_t>(sample));
            if (rate > update)
              update = rate;
          },
          Kokkos::Max<double>(largest));
      Kokkos::single(Kokkos::PerTeam(team), [&] {
        stash(first_node + node, 0) = fractions(node, last);
        stash(first_node + node, 1) = largest;
      });
    };
    launch::parallel_for("solprop_native.response.device_plasma_stash",
                         TeamPolicy(response_space_for(node_count),
                                    static_cast<int>(node_count), team_size),
                         peak);
  }
}
#endif

static_assert(band_max_half + 1 == static_cast<int>(plasma_max_taps),
              "the team band filter's tap bound is the problem's");

namespace {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
/** Whether the band column skips each row's quiet samples (`band_support`): host only.
 */
constexpr bool band_windows = false;
#else
constexpr bool band_windows = true;
/**
 * The fewest samples a host thread of a band team takes from a row: below it a thread's
 * share of a pass no longer repays the team's barriers between the passes. Measured on
 * the mode-averaged helium RDW case (n_to = 8192, one node) on a loaded Apple M-series
 * host: teams of two and four threads ran the band column faster than one, six slower;
 * 2048 caps that row at four. To be re-measured on each host class (solprop SEP 25).
 */
constexpr std::size_t band_host_samples_per_thread = 2048;
#endif

/** The first and last indices whose value is non-zero, as a Kokkos MinMax reduction. */
using BandLive = Kokkos::MinMaxScalar<int>;

/** The band window of a row whose non-zero samples `live` found. */
KOKKOS_INLINE_FUNCTION BandWindow band_window_of(const BandLive& live,
                                                 const int samples, const int half) {
  if constexpr (band_windows)
    return band_support(live.min_val, live.max_val, samples, half);
  else
    return {.first = 0, .last = samples - 1};
}

/** Count `index` into `live` when `value` is not exactly zero. */
KOKKOS_INLINE_FUNCTION void band_note(BandLive& live, const int index,
                                      const double value) {
  if (value != 0.0) {
    live.min_val = Kokkos::min(live.min_val, index);
    live.max_val = Kokkos::max(live.max_val, index);
  }
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
/**
 * Count a host block's non-zero values into `live`, as `band_note` of each:
 * `values[k]` is the sample `offset + k`. Only the first and the last non-zero value
 * matter, so the block is searched from either end.
 */
inline void band_note_block(BandLive& live, const double* values,
                            const std::size_t count, const int offset) {
  std::size_t low = 0;
  while (low < count && values[low] == 0.0)
    ++low;
  if (low == count)
    return;
  std::size_t high = count - 1;
  while (values[high] == 0.0)
    --high;
  live.min_val = std::min(live.min_val, offset + static_cast<int>(low));
  live.max_val = std::max(live.max_val, offset + static_cast<int>(high));
}

/**
 * `ionisation_rate` of a host block's `count <= rate_block` magnitudes into `row`,
 * with the table's limits `lower`, `upper` and `highest` as it takes them (a NaN
 * magnitude is live); `within(values, live)` runs `ionisation_rates_within` on the
 * live ones. A mostly live block runs every sample through the layer, a quiet one's
 * magnitude replaced by the lower limit, and then keeps the limits' values: each value
 * is the same function of its own magnitude, so the passes' lanes do not change it,
 * and no sample is gathered. A mostly quiet block gathers its live magnitudes.
 */
template <class Within>
void band_rates_block(const double* magnitudes, double* row, const std::size_t count,
                      const double lower, const double upper, const double highest,
                      const Within& within) {
  double values[rate_block];
  std::size_t live = 0;
  for (std::size_t k = 0; k < count; ++k)
    live += ((magnitudes[k] > upper) | (magnitudes[k] < lower)) ? 0 : 1;
  if (2 * live >= count && lower > 0.0) {
    for (std::size_t k = 0; k < count; ++k) {
      const double magnitude = magnitudes[k];
      values[k] = ((magnitude > upper) | (magnitude < lower)) ? lower : magnitude;
    }
    within(values, count);
    for (std::size_t k = 0; k < count; ++k) {
      const double magnitude = magnitudes[k];
      row[k] = magnitude > upper ? highest : magnitude < lower ? 0.0 : values[k];
    }
    return;
  }
  if (live == 0) {
    // A quiet block takes only the limits.
    for (std::size_t k = 0; k < count; ++k)
      row[k] = magnitudes[k] > upper ? highest : 0.0;
    return;
  }
  std::size_t active[rate_block];
  std::size_t gathered = 0;
  for (std::size_t k = 0; k < count; ++k) {
    const double magnitude = magnitudes[k];
    row[k] = magnitude > upper ? highest : 0.0;
    if (!(magnitude > upper) && !(magnitude < lower)) {
      values[gathered] = magnitude;
      active[gathered] = k;
      ++gathered;
    }
  }
  within(values, gathered);
  for (std::size_t k = 0; k < gathered; ++k)
    row[active[k]] = values[k];
}
#endif
} // namespace

int GeneralWorkspace::band_host_team_size(const std::size_t node_count) const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  static_cast<void>(node_count);
  return 1;
#else
  // solprop SEP 25 section 6.7.5: share the threads between the chunk's nodes, so a
  // chunk of fewer nodes than threads runs each node's passes over chunks of t.
  const auto threads = static_cast<std::size_t>(
      std::max(1, response_space_for(node_count).concurrency()));
  const auto per_node =
      std::max<std::size_t>(1, threads / std::max<std::size_t>(1, node_count));
  const auto by_samples =
      std::max<std::size_t>(1, problem_.dims.n_to / band_host_samples_per_thread);
  return static_cast<int>(std::min(per_node, by_samples));
#endif
}

std::size_t GeneralWorkspace::band_league(const std::size_t node_count) const {
  return build::device ? node_count : std::min(node_count, band_slots_);
}

template <class Functor>
Kokkos::TeamPolicy<ExecutionSpace>
GeneralWorkspace::band_team_policy(const std::size_t node_count,
                                   const Functor& functor) const {
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  constexpr auto policy_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (node_count > policy_limit || problem_.dims.n_to > policy_limit)
    throw std::length_error("band team column extent exceeds signed 32-bit range");
  const auto league = static_cast<int>(band_league(node_count));
  const auto scratch =
      Kokkos::PerTeam(band_staging_bytes<band_staging, ExecutionSpace>());
  // The widest team this kernel can launch with its scratch: a device's registers per
  // thread can hold the band column below the configured width.
  Policy probe(response_space_for(node_count), league, 1);
  probe.set_scratch_size(0, scratch);
  const int widest = probe.team_size_max(functor, Kokkos::ParallelForTag());
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const int requested = static_cast<int>(response_team_size());
#else
  const int requested = band_host_team_size(node_count);
#endif
  Policy policy(response_space_for(node_count), league,
                std::max(1, std::min(requested, widest)));
  policy.set_scratch_size(0, scratch);
  return policy;
}

// solprop SEP 18's band rule as a team column in whole-row passes (solprop SEP 25
// section 6.7.4; include/solprop_native/plasma.hpp): per channel the rates, their
// filter and scan give the exposure; the fraction (floored where it is formed), the
// loss guard and the free-current weight follow pointwise, a mixture adding each
// channel's share in channel order. Per component the free-current integrand is
// filtered and scanned, and the scan's store forms the current from it, the guard and
// the field. SEP 27 uses this current directly, without a third running integral.
//
// The rows, by slot (a team's own; the node on a device): `plasma_rate_` the raw rates,
// `plasma_fraction_` the filtered row, `plasma_exposure_` the exposure,
// `plasma_free_current_` the loss guard (a mixture's work until the last channel),
// `plasma_outer_current_` a mixture's weight, `plasma_current_integrand_` the
// free-current integrand. The separate `held_current_` stores the total current. Each
// filter and scan runs on its row's live window (`band_support`) and writes the rest:
// zero before it, the final prefix after it, and an unchanged exposure's fraction.
template <FieldFamily Family, std::size_t Components, bool Multiple, bool Folded>
void GeneralWorkspace::team_band_plasma_column(const std::size_t stage,
                                               const std::size_t first_node,
                                               const std::size_t node_count) {
  if constexpr (Components == 1 && !Folded) {
    if (folds_single_node()) {
      team_band_plasma_column<Family, Components, Multiple, true>(stage, first_node,
                                                                  node_count);
      return;
    }
  }
  static_assert(Family != FieldFamily::envelope,
                "the band rule integrates a field-grid plasma column");
  using TeamMember = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  // A folded single node reads the modal field through its one-element synthesis
  // (`folds_single_node`).
  constexpr bool folded = Folded;
  const auto node_real = folded ? modal_real_ : node_real_;
  const auto node_complex = folded ? modal_complex_ : node_complex_;
  const auto synthesis = synthesis_;
  const auto taper = stage_taper_;
  const auto out = held_current_;
  const auto rates = plasma_rate_;
  const auto filtered = plasma_fraction_;
  const auto exposures = plasma_exposure_;
  const auto guards = plasma_free_current_;
  const auto weights = plasma_outer_current_;
  const auto integrands = plasma_current_integrand_;
  const auto taps_view = plasma_taps_;
  const auto stash = stash_candidate_;
  const bool stashing = !stash.empty() && stage + 1 == problem_.stepper.n_stages;
  const auto rate_coefficients = rate_coefficients_;
  const auto channel_parameters = channel_parameters_;
  const auto channel_coefficients = channel_coefficients_;
  const auto channel_density = stage_channel_density_;
  const auto channel_preionisation = stage_channel_preionisation_;
  const auto density = stage_plasma_density_;
  const auto preionisation = stage_preionisation_;
  const double charge_mass_ratio = problem_.plasma.charge_mass_ratio;
  const double ionisation_potential = problem_.plasma.ionisation_potential;
  const double dt = problem_.grid.dt_fine;
  const double log_field_min = problem_.plasma.log_field_min;
  const double d_log_field = problem_.plasma.d_log_field;
  const double field_min = problem_.plasma.field_min;
  const double field_max = problem_.plasma.field_max;
  const double rate_max = problem_.plasma.rate_max;
  const auto rate_intervals = problem_.plasma.rate_intervals;
  const std::size_t channels = Multiple ? problem_.plasma.n_channels : 1;
  const auto samples = static_cast<int>(problem_.dims.n_to);
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  const std::size_t sample_count = problem_.dims.n_to;
#endif
  const auto half = static_cast<int>(problem_.plasma.half_width());
  const std::size_t last = problem_.dims.n_to - 1;
  const std::size_t league = band_league(node_count);
  const auto column = KOKKOS_LAMBDA(const TeamMember& team) {
    // CUDA extended-lambda restriction 12 requires the first capture outside
    // if-constexpr blocks.
    capture_anchor(node_real, node_complex, out, rates, filtered, exposures, guards,
                   weights, integrands, taps_view, stash, stashing, rate_coefficients,
                   channel_parameters, channel_coefficients, channel_density,
                   channel_preionisation, density, preionisation, charge_mass_ratio,
                   ionisation_potential, dt, log_field_min, d_log_field, field_min,
                   field_max, rate_max, rate_intervals, channels, samples, half, last,
                   first_node, stage, league, synthesis, taper);
    const auto slot = static_cast<std::size_t>(team.league_rank());
    const std::size_t own = slot * Components;
    double taps[plasma_max_taps] = {};
    for (int k = 0; k <= half; ++k)
      taps[k] = taps_view(static_cast<std::size_t>(k));
    const auto tile = band_tile<band_staging>(team);
    const double input_scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
    for (std::size_t node = slot; node < node_count; node += league) {
      const std::size_t base = node * Components;
      const auto field_at = [&](const std::size_t row, const std::size_t time) {
        if constexpr (Family == FieldFamily::field_real)
          return input_scale * node_real(row, time);
        else
          return input_scale * node_complex(row, time).real();
      };
      const auto magnitude_at = [&](const std::size_t time) {
        const double first = field_at(base, time);
        if constexpr (Components == 2)
          return Kokkos::hypot(first, field_at(base + 1, time));
        else
          return Kokkos::abs(first);
      };
      [[maybe_unused]] const auto rate_of = [&](const double magnitude,
                                                const std::size_t channel) {
        if constexpr (Multiple)
          return ionisation_rate(
              magnitude, channel_coefficients, channel_parameters(channel, 0),
              channel_parameters(channel, 1), channel_parameters(channel, 2),
              channel_parameters(channel, 3), channel_parameters(channel, 4),
              static_cast<std::size_t>(channel_parameters(channel, 5)),
              static_cast<std::size_t>(channel_parameters(channel, 6)));
        else
          return ionisation_rate(magnitude, rate_coefficients, log_field_min,
                                 d_log_field, field_min, field_max, rate_max,
                                 rate_intervals);
      };
      BandLive integrand_live;
      integrand_live.min_val = samples;
      integrand_live.max_val = -1;

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
      // With vector maths a host team thread takes blocks of samples and gathers the
      // live ones' transcendentals into passes of the layer (`ionisation_rates_within`,
      // the decays below), writing the rest as it goes; with the system libm each
      // sample keeps its own calls, as on a device.
      [[maybe_unused]] const auto blocks =
          static_cast<int>((sample_count + rate_block - 1) / rate_block);
      [[maybe_unused]] const auto rates_into =
          [&](const std::size_t first, const std::size_t count,
              const std::size_t channel, BandLive& update) {
            const double lower = Multiple ? channel_parameters(channel, 2) : field_min;
            const double upper = Multiple ? channel_parameters(channel, 3) : field_max;
            const double highest = Multiple ? channel_parameters(channel, 4) : rate_max;
            const auto within = [&](double* values, const std::size_t live) {
              if constexpr (Multiple)
                ionisation_rates_within(
                    values, live, channel_coefficients, channel_parameters(channel, 0),
                    channel_parameters(channel, 1),
                    static_cast<std::size_t>(channel_parameters(channel, 5)),
                    static_cast<std::size_t>(channel_parameters(channel, 6)));
              else
                ionisation_rates_within(values, live, rate_coefficients, log_field_min,
                                        d_log_field, rate_intervals);
            };
            double magnitudes[rate_block];
            for (std::size_t k = 0; k < count; ++k)
              magnitudes[k] = magnitude_at(first + k);
            double* const row = &rates(slot, first);
            band_rates_block(magnitudes, row, count, lower, upper, highest, within);
            band_note_block(update, row, count, static_cast<int>(first));
          };
#endif

      // 1. Per channel: rates -> filter -> scan -> exposure, then the fraction, the
      // guard and the weight pointwise.
      for (std::size_t channel = 0; channel < channels; ++channel) {
        BandLive rate_live;
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
        if constexpr (vmath::vectorised) {
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, blocks),
              [&](const int block, BandLive& update) {
                const auto first = static_cast<std::size_t>(block) * rate_block;
                rates_into(first, std::min(rate_block, sample_count - first), channel,
                           update);
              },
              Kokkos::MinMax<int>(rate_live));
        } else
#endif
        {
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, samples),
              [&](const int sample, BandLive& update) {
                const auto time = static_cast<std::size_t>(sample);
                const double rate = rate_of(magnitude_at(time), channel);
                rates(slot, time) = rate;
                band_note(update, sample, rate);
              },
              Kokkos::MinMax<int>(rate_live));
        }
        team.team_barrier();
        const BandWindow window = band_window_of(rate_live, samples, half);
        if (window.first <= window.last)
          team_band_filter<band_staging>(team, tile, rates, slot, filtered, slot,
                                         samples, taps, half, window.first,
                                         window.last + 1);
        const double total =
            team_trapezoid_window(team, filtered, slot, window, samples, dt,
                                  [&](const int sample, const double prefix) {
                                    exposures(slot, static_cast<std::size_t>(sample)) =
                                        prefix;
                                  });
        const double initial =
            Multiple ? channel_preionisation(stage, channel) : preionisation(stage);
        // Outside [window.first, window.last + 1] the exposure is 0 before and `total`
        // after, so the fraction there is one of two values (band_support).
        const int changing_end = Kokkos::min(window.last + 2, samples);
        const double before = band_fraction(0.0, initial);
        const double after = band_fraction(total, initial);
        [[maybe_unused]] const auto fraction_at = [&](const int sample) {
          if (sample < window.first)
            return before;
          if (sample >= changing_end)
            return after;
          return band_fraction(exposures(slot, static_cast<std::size_t>(sample)),
                               initial);
        };
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
        // A host block's fractions with vector maths, each handed to `apply` with its
        // sample: the changing samples' decays `exp(-max(X, 0))` in one pass, each
        // fraction then `band_fraction`'s strict arithmetic.
        [[maybe_unused]] const auto fractions_into =
            [&](const std::size_t first, const std::size_t count, const auto& apply) {
              double decay[rate_block];
              std::size_t active[rate_block];
              std::size_t live = 0;
              for (std::size_t time = first; time < first + count; ++time) {
                const auto sample = static_cast<int>(time);
                if (sample < window.first) {
                  apply(time, before);
                } else if (sample >= changing_end) {
                  apply(time, after);
                } else {
                  const double exposure = exposures(slot, time);
                  decay[live] = -(exposure < 0.0 ? 0.0 : exposure);
                  active[live] = time;
                  ++live;
                }
              }
              vmath::exp(decay, decay, live);
              for (std::size_t k = 0; k < live; ++k)
                apply(active[k], decayed_fraction(decay[k], initial));
            };
        // A single gas's host block of free-current integrands and loss guards as
        // loops over the block's samples: the fractions as `fractions_into` forms
        // them (the changing samples' decays in one pass, `decayed_fractions` in
        // another), then `band_loss_scale` and the weights with their branch taken
        // as a selection, and the integrand rows' live samples searched afterwards.
        [[maybe_unused]] const auto integrands_into = [&](const std::size_t first,
                                                          const std::size_t count,
                                                          BandLive& update) {
          double fraction[rate_block];
          const auto start = static_cast<int>(first);
          const auto stop = static_cast<int>(first + count);
          for (std::size_t k = 0; k < count; ++k)
            fraction[k] = start + static_cast<int>(k) < window.first ? before : after;
          const int begin = Kokkos::max(window.first, start);
          const int end = Kokkos::min(changing_end, stop);
          if (begin < end) {
            double* const changing = fraction + (begin - start);
            const auto changing_count = static_cast<std::size_t>(end - begin);
            const double* const exposure =
                &exposures(slot, static_cast<std::size_t>(begin));
            for (std::size_t k = 0; k < changing_count; ++k)
              changing[k] = -(exposure[k] < 0.0 ? 0.0 : exposure[k]);
            vmath::exp(changing, changing, changing_count);
            decayed_fractions(changing, changing_count, initial);
          }
          const double* const rate_row = &rates(slot, first);
          double* const guard_row = &guards(slot, first);
          for (std::size_t k = 0; k < count; ++k) {
            const std::size_t time = first + k;
            const double magnitude = magnitude_at(time);
            const double squared = magnitude * magnitude;
            const double rate = rate_row[k];
            const bool loses = rate > 0.0 && squared > 0.0;
            const double scale = ionisation_potential * rate * (1.0 - fraction[k]) /
                                 (loses ? squared : 1.0);
            guard_row[k] = loses ? scale : 0.0;
            const double weight = fraction[k] * charge_mass_ratio;
            for (std::size_t component = 0; component < Components; ++component)
              integrands(own + component, time) =
                  weight * field_at(base + component, time);
          }
          for (std::size_t component = 0; component < Components; ++component)
            band_note_block(update, &integrands(own + component, first), count, start);
        };
#endif
        if constexpr (Multiple) {
          const double partial = channel_density(stage, channel);
          const double charge = channel_parameters(channel, 7);
          const double potential = channel_parameters(channel, 8);
          const auto share_at = [&](const std::size_t time, const double fraction) {
            const double rate = rates(slot, time);
            const double share = partial * fraction * charge;
            const double paid = partial * potential * rate * (1.0 - fraction);
            const double weight = channel == 0 ? 0.0 : weights(slot, time);
            const double work = channel == 0 ? 0.0 : guards(slot, time);
            weights(slot, time) = weight + share;
            guards(slot, time) = work + paid;
          };
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
          if constexpr (vmath::vectorised) {
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, blocks), [&](const int block) {
                  const auto first = static_cast<std::size_t>(block) * rate_block;
                  fractions_into(first, std::min(rate_block, sample_count - first),
                                 share_at);
                });
          } else
#endif
          {
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, samples), [&](const int sample) {
                  share_at(static_cast<std::size_t>(sample), fraction_at(sample));
                });
          }
        } else {
          const auto integrand_at = [&](const std::size_t time, const double fraction,
                                        BandLive& update) {
            const double magnitude = magnitude_at(time);
            const double squared = magnitude * magnitude;
            guards(slot, time) = band_loss_scale(rates(slot, time), fraction, squared,
                                                 ionisation_potential);
            const double weight = fraction * charge_mass_ratio;
            for (std::size_t component = 0; component < Components; ++component) {
              const double value = weight * field_at(base + component, time);
              integrands(own + component, time) = value;
              band_note(update, static_cast<int>(time), value);
            }
          };
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
          if constexpr (vmath::vectorised) {
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, blocks),
                [&](const int block, BandLive& update) {
                  const auto first = static_cast<std::size_t>(block) * rate_block;
                  integrands_into(first, std::min(rate_block, sample_count - first),
                                  update);
                },
                Kokkos::MinMax<int>(integrand_live));
          } else
#endif
          {
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, samples),
                [&](const int sample, BandLive& update) {
                  integrand_at(static_cast<std::size_t>(sample), fraction_at(sample),
                               update);
                },
                Kokkos::MinMax<int>(integrand_live));
          }
        }
        if (stashing) {
          double largest = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, samples),
              [&](const int sample, double& update) {
                const double rate = rates(slot, static_cast<std::size_t>(sample));
                if (rate > update)
                  update = rate;
              },
              Kokkos::Max<double>(largest));
          Kokkos::single(Kokkos::PerTeam(team), [&] {
            stash(first_node + node, 2 * channel) =
                band_fraction(exposures(slot, last), initial);
            stash(first_node + node, 2 * channel + 1) = largest > 0.0 ? largest : 0.0;
          });
        }
        team.team_barrier();
      }
      if constexpr (Multiple) {
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, samples),
            [&](const int sample, BandLive& update) {
              const auto time = static_cast<std::size_t>(sample);
              const double magnitude = magnitude_at(time);
              const double squared = magnitude * magnitude;
              guards(slot, time) =
                  band_loss_scale(1.0, 0.0, squared, guards(slot, time));
              const double weight = weights(slot, time);
              for (std::size_t component = 0; component < Components; ++component) {
                const double value = weight * field_at(base + component, time);
                integrands(own + component, time) = value;
                band_note(update, sample, value);
              }
            },
            Kokkos::MinMax<int>(integrand_live));
      }
      team.team_barrier();

      // 2. Per component: the free current, and from it the current.
      const BandWindow free_window = band_window_of(integrand_live, samples, half);
      // A single gas's density, read once: inside the store below the compiler must
      // reload it after each store to the current, which might alias it.
      const double stage_density = Multiple ? 0.0 : density(stage);
      for (std::size_t component = 0; component < Components; ++component) {
        const std::size_t row = base + component;
        if (free_window.first <= free_window.last)
          team_band_filter<band_staging>(team, tile, integrands, own + component,
                                         filtered, slot, samples, taps, half,
                                         free_window.first, free_window.last + 1);
        team_trapezoid_window(team, filtered, slot, free_window, samples, dt,
                              [&](const int sample, const double prefix) {
                                const auto time = static_cast<std::size_t>(sample);
                                const double drive =
                                    guards(slot, time) * field_at(row, time);
                                out(row, time) = Multiple
                                                     ? prefix + drive
                                                     : stage_density * (prefix + drive);
                              });
      }
    }
  };
  launch::parallel_for("solprop_native.response.team_band_plasma_column",
                       band_team_policy(node_count, column), column);
}

namespace {
/**
 * A band column's node field: `scale` times the real row of a field-family node (or of
 * the folded single node's modal row), and its magnitude over a node's components.
 */
template <FieldFamily Family, std::size_t Components> struct BandField {
  RealView2 real;
  ComplexView2 complex;
  KOKKOS_INLINE_FUNCTION double at(const std::size_t row, const std::size_t time,
                                   const double scale) const {
    if constexpr (Family == FieldFamily::field_real)
      return scale * real(row, time);
    else
      return scale * complex(row, time).real();
  }
  KOKKOS_INLINE_FUNCTION double
  magnitude(const std::size_t base, const std::size_t time, const double scale) const {
    const double first = at(base, time, scale);
    if constexpr (Components == 2)
      return Kokkos::hypot(first, at(base + 1, time, scale));
    else
      return Kokkos::abs(first);
  }
};

/** A band column's ionisation rate: the single gas's table, or a mixture channel's. */
template <bool Multiple> struct BandRate {
  RealView coefficients;
  RealView2 parameters;
  RealView channel_coefficients;
  double log_field_min{}, d_log_field{}, field_min{}, field_max{}, rate_max{};
  std::size_t intervals{};
  KOKKOS_INLINE_FUNCTION double operator()(const double magnitude,
                                           const std::size_t channel) const {
    if constexpr (Multiple)
      return ionisation_rate(magnitude, channel_coefficients, parameters(channel, 0),
                             parameters(channel, 1), parameters(channel, 2),
                             parameters(channel, 3), parameters(channel, 4),
                             static_cast<std::size_t>(parameters(channel, 5)),
                             static_cast<std::size_t>(parameters(channel, 6)));
    else
      return ionisation_rate(magnitude, coefficients, log_field_min, d_log_field,
                             field_min, field_max, rate_max, intervals);
  }
};
} // namespace

template <class Functor>
Kokkos::TeamPolicy<ExecutionSpace>
GeneralWorkspace::tiled_band_policy(const std::size_t rows, const bool scratch,
                                    const Functor& functor) const {
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  constexpr auto policy_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (band_tile_count_ == 0 || rows > policy_limit / band_tile_count_ ||
      problem_.dims.n_to > policy_limit)
    throw std::length_error("tiled band column extent exceeds signed 32-bit range");
  const auto league = static_cast<int>(rows * band_tile_count_);
  const auto bytes = scratch ? band_tile_bytes<ExecutionSpace>(band_tile_) : 0;
  Policy probe(response_execution_space_, league, 1);
  probe.set_scratch_size(0, Kokkos::PerTeam(bytes));
  const int widest = probe.team_size_max(functor, Kokkos::ParallelForTag());
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const auto requested = static_cast<int>(std::min(response_team_size(), band_tile_));
#else
  const int requested = std::max(1, band_tile_threads_);
#endif
  Policy policy(response_execution_space_, league,
                std::max(1, std::min(requested, widest)));
  policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
  return policy;
}

// The band column in tiles (solprop SEP 25 section 6.7.6;
// include/solprop_native/plasma.hpp): each running integral is a local kernel of one
// team per (row, tile), which loads, filters and scans its tile from zero
// (`team_band_tile`), then a finish kernel of one team per (row, tile), which adds the
// row's earlier tile totals (`team_tile_offset`) and forms the pass's product from the
// prefix, in the team column's arithmetic:
//
//   1. per channel: the local kernel forms the rates in its window (keeping its own
//      tile's in `plasma_rate_`) and the exposure's local prefixes
//      (`plasma_exposure_`); the finish forms the fraction, the loss guard
//      (`plasma_free_current_`) and the free-current integrand
//      (`plasma_current_integrand_`), a mixture accumulating its weight
//      (`plasma_outer_current_`) and work channel by channel;
//   2. per component row: the integrand's local prefixes (`plasma_outer_current_`,
//      free after the first pass), then the current (`held_current_`).
//
// Rows are by node within the chunk, the totals by row (`band_tile_totals_`). The
// stash's fraction comes from the last sample's finish and its largest rate from a
// reduction over the rate row. No window is skipped: every sample is formed.
template <FieldFamily Family, std::size_t Components, bool Multiple, bool Folded>
void GeneralWorkspace::tiled_band_plasma_column(const std::size_t stage,
                                                const std::size_t first_node,
                                                const std::size_t node_count) {
  if constexpr (Components == 1 && !Folded) {
    if (folds_single_node()) {
      tiled_band_plasma_column<Family, Components, Multiple, true>(stage, first_node,
                                                                   node_count);
      return;
    }
  }
  static_assert(Family != FieldFamily::envelope,
                "the band rule integrates a field-grid plasma column");
  using TeamMember = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  constexpr bool folded = Folded;
  const BandField<Family, Components> field{folded ? modal_real_ : node_real_,
                                            folded ? modal_complex_ : node_complex_};
  const BandRate<Multiple> rate{rate_coefficients_,
                                channel_parameters_,
                                channel_coefficients_,
                                problem_.plasma.log_field_min,
                                problem_.plasma.d_log_field,
                                problem_.plasma.field_min,
                                problem_.plasma.field_max,
                                problem_.plasma.rate_max,
                                problem_.plasma.rate_intervals};
  const auto synthesis = synthesis_;
  const auto taper = stage_taper_;
  const auto out = held_current_;
  const auto rates = plasma_rate_;
  const auto exposures = plasma_exposure_;
  const auto guards = plasma_free_current_;
  const auto weights = plasma_outer_current_;
  const auto locals = plasma_outer_current_;
  const auto integrands = plasma_current_integrand_;
  const auto totals = band_tile_totals_;
  const auto taps_view = plasma_taps_;
  const auto stash = stash_candidate_;
  const bool stashing = !stash.empty() && stage + 1 == problem_.stepper.n_stages;
  const auto channel_parameters = channel_parameters_;
  const auto channel_density = stage_channel_density_;
  const auto channel_preionisation = stage_channel_preionisation_;
  const auto density = stage_plasma_density_;
  const auto preionisation = stage_preionisation_;
  const double charge_mass_ratio = problem_.plasma.charge_mass_ratio;
  const double ionisation_potential = problem_.plasma.ionisation_potential;
  const double dt = problem_.grid.dt_fine;
  const std::size_t channels = Multiple ? problem_.plasma.n_channels : 1;
  const auto samples = static_cast<int>(problem_.dims.n_to);
  const auto half = static_cast<int>(problem_.plasma.half_width());
  const auto width = static_cast<int>(band_tile_);
  const auto tiles = static_cast<int>(band_tile_count_);
  const int last = samples - 1;
  const std::size_t rows = node_count * Components;

  // 1. Per channel: the exposure, then the fraction, the guard and the integrand.
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto local = KOKKOS_LAMBDA(const TeamMember& team) {
      capture_anchor(field, rate, rates, exposures, totals, taps_view, synthesis, taper,
                     stage, channel, samples, half, width, tiles, dt);
      const auto node = static_cast<std::size_t>(team.league_rank() / tiles);
      const int index = team.league_rank() % tiles;
      const int first = index * width;
      const int end = Kokkos::min(first + width, samples);
      double taps[plasma_max_taps] = {};
      for (int k = 0; k <= half; ++k)
        taps[k] = taps_view(static_cast<std::size_t>(k));
      const double scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
      const std::size_t base = node * Components;
      const double total = team_band_tile(
          team,
          [&](const int sample) {
            const auto time = static_cast<std::size_t>(sample);
            const double value = rate(field.magnitude(base, time, scale), channel);
            if (sample >= first && sample < end)
              rates(node, time) = value;
            return value;
          },
          first, end, samples, taps, half, dt,
          [&](const int sample, const double prefix) {
            exposures(node, static_cast<std::size_t>(sample)) = prefix;
          });
      Kokkos::single(Kokkos::PerTeam(team),
                     [&] { totals(node, static_cast<std::size_t>(index)) = total; });
    };
    launch::parallel_for("solprop_native.response.tiled_band_exposure",
                         tiled_band_policy(node_count, true, local), local);
    const auto finish = KOKKOS_LAMBDA(const TeamMember& team) {
      capture_anchor(field, rates, exposures, guards, weights, integrands, totals,
                     stash, stashing, synthesis, taper, channel_parameters,
                     channel_density, channel_preionisation, preionisation,
                     charge_mass_ratio, ionisation_potential, channels, channel,
                     samples, width, tiles, last, first_node, stage);
      const auto node = static_cast<std::size_t>(team.league_rank() / tiles);
      const int index = team.league_rank() % tiles;
      const int first = index * width;
      const int end = Kokkos::min(first + width, samples);
      const double offset = team_tile_offset(team, totals, node, index);
      const double scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
      const std::size_t base = node * Components;
      const double initial =
          Multiple ? channel_preionisation(stage, channel) : preionisation(stage);
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, first, end), [&](const int sample) {
            const auto time = static_cast<std::size_t>(sample);
            const double fraction =
                band_fraction(offset + exposures(node, time), initial);
            const double value = rates(node, time);
            if constexpr (Multiple) {
              const double partial = channel_density(stage, channel);
              const double charge = channel_parameters(channel, 7);
              const double potential = channel_parameters(channel, 8);
              const double share = partial * fraction * charge;
              const double paid = partial * potential * value * (1.0 - fraction);
              const double weight = (channel == 0 ? 0.0 : weights(node, time)) + share;
              const double work = (channel == 0 ? 0.0 : guards(node, time)) + paid;
              if (channel + 1 == channels) {
                const double magnitude = field.magnitude(base, time, scale);
                guards(node, time) =
                    band_loss_scale(1.0, 0.0, magnitude * magnitude, work);
                for (std::size_t component = 0; component < Components; ++component)
                  integrands(base + component, time) =
                      weight * field.at(base + component, time, scale);
              } else {
                weights(node, time) = weight;
                guards(node, time) = work;
              }
            } else {
              const double magnitude = field.magnitude(base, time, scale);
              guards(node, time) = band_loss_scale(
                  value, fraction, magnitude * magnitude, ionisation_potential);
              const double weight = fraction * charge_mass_ratio;
              for (std::size_t component = 0; component < Components; ++component)
                integrands(base + component, time) =
                    weight * field.at(base + component, time, scale);
            }
            if (stashing && sample == last)
              stash(first_node + node, 2 * channel) = fraction;
          });
    };
    launch::parallel_for("solprop_native.response.tiled_band_fraction",
                         tiled_band_policy(node_count, false, finish), finish);
    if (stashing) {
      const auto peak = KOKKOS_LAMBDA(const TeamMember& team) {
        capture_anchor(rates, stash, channel, samples, first_node);
        const auto node = static_cast<std::size_t>(team.league_rank());
        double largest = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, samples),
            [&](const int sample, double& update) {
              const double value = rates(node, static_cast<std::size_t>(sample));
              if (value > update)
                update = value;
            },
            Kokkos::Max<double>(largest));
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          stash(first_node + node, 2 * channel + 1) = largest > 0.0 ? largest : 0.0;
        });
      };
      launch::parallel_for("solprop_native.response.tiled_band_peak_rate",
                           Kokkos::TeamPolicy<ExecutionSpace>(
                               response_space_for(node_count),
                               static_cast<int>(node_count), Kokkos::AUTO()),
                           peak);
    }
  }

  // 2. Per component row: the free current, followed by the pointwise loss current.
  tiled_band_local(integrands, rows, "solprop_native.response.tiled_band_free_current");
  const auto current = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(field, guards, locals, out, density, totals, synthesis, taper, stage,
                   samples, width, tiles);
    const auto row = static_cast<std::size_t>(team.league_rank() / tiles);
    const int index = team.league_rank() % tiles;
    const int first = index * width;
    const int end = Kokkos::min(first + width, samples);
    const double offset = team_tile_offset(team, totals, row, index);
    const double scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
    const std::size_t node = row / Components;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, first, end), [&](const int sample) {
          const auto time = static_cast<std::size_t>(sample);
          const double prefix = offset + locals(row, time);
          const double drive = guards(node, time) * field.at(row, time, scale);
          out(row, time) =
              Multiple ? prefix + drive : density(stage) * (prefix + drive);
        });
  };
  launch::parallel_for("solprop_native.response.tiled_band_current",
                       tiled_band_policy(rows, false, current), current);
}

// The tiled column's local kernel over component rows: `source`'s band-filtered tile
// prefixes from zero into the free-current rows (`plasma_outer_current_`),
// and each tile's total into `band_tile_totals_`.
void GeneralWorkspace::tiled_band_local(const RealView2& source, const std::size_t rows,
                                        const char* label) {
  using TeamMember = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  const auto locals = plasma_outer_current_;
  const auto totals = band_tile_totals_;
  const auto taps_view = plasma_taps_;
  const double dt = problem_.grid.dt_fine;
  const auto samples = static_cast<int>(problem_.dims.n_to);
  const auto half = static_cast<int>(problem_.plasma.half_width());
  const auto width = static_cast<int>(band_tile_);
  const auto tiles = static_cast<int>(band_tile_count_);
  const auto local = KOKKOS_LAMBDA(const TeamMember& team) {
    const auto row = static_cast<std::size_t>(team.league_rank() / tiles);
    const int index = team.league_rank() % tiles;
    const int first = index * width;
    const int end = Kokkos::min(first + width, samples);
    double taps[plasma_max_taps] = {};
    for (int k = 0; k <= half; ++k)
      taps[k] = taps_view(static_cast<std::size_t>(k));
    const double total = team_band_tile(
        team,
        [&](const int sample) { return source(row, static_cast<std::size_t>(sample)); },
        first, end, samples, taps, half, dt,
        [&](const int sample, const double prefix) {
          locals(row, static_cast<std::size_t>(sample)) = prefix;
        });
    Kokkos::single(Kokkos::PerTeam(team),
                   [&] { totals(row, static_cast<std::size_t>(index)) = total; });
  };
  launch::parallel_for(label, tiled_band_policy(rows, true, local), local);
}

// The band rule's statistics scan: per channel the rates, their band filter and
// trapezoid scan, then team reductions over the rows. The floored final exposure and
// the largest rate always; with `Diagnose` the six parts of SEP 18's run diagnostics,
// the largest over the channels: the exposure's largest dip and rise (maxima of the
// scan's increments), the largest |paid - freed| and freed (the paid work a sum of the
// trapezoid's increments of the raw rate's owed work), and the largest |f E| in the
// first 2K samples and anywhere. Maxima combine exactly; the sums associate as the
// team's scan does. The rows are the band column's (`plasma_rate_`,
// `plasma_fraction_` filtered and then owed work, `plasma_exposure_` exposure and then
// fraction), by slot as the column walks the nodes. Not a step's hot path: it runs
// whole rows, without the column's live windows.
template <bool Diagnose>
void GeneralWorkspace::team_band_statistics(const std::size_t first_node,
                                            const std::size_t node_count,
                                            const RealView2& output,
                                            const double preionisation) {
  using TeamMember = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const bool real_family = problem_.dims.family == FieldFamily::field_real;
  const auto rates = plasma_rate_;
  const auto filtered = plasma_fraction_;
  const auto exposures = plasma_exposure_;
  const auto taps_view = plasma_taps_;
  const auto diagnostics = plasma_diagnostics_;
  const auto coefficients = rate_coefficients_;
  const auto parameters = channel_parameters_;
  const auto channel_coefficients = channel_coefficients_;
  const auto channel_preionisation = stage_channel_preionisation_;
  const auto channels = problem_.plasma.n_channels;
  const bool multiple = channels > 1;
  const auto n_pol = problem_.effective_node_pol();
  const auto samples = static_cast<int>(problem_.dims.n_to);
  const auto intervals = problem_.plasma.rate_intervals;
  const auto dt = problem_.grid.dt_fine;
  const auto log_field_min = problem_.plasma.log_field_min;
  const auto d_log_field = problem_.plasma.d_log_field;
  const auto field_min = problem_.plasma.field_min;
  const auto field_max = problem_.plasma.field_max;
  const auto rate_max = problem_.plasma.rate_max;
  const auto half = static_cast<int>(problem_.plasma.half_width());
  const int reach = Kokkos::min(2 * half, samples);
  const std::size_t last = problem_.dims.n_to - 1;
  const std::size_t league = band_league(node_count);
  const auto scan = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(node_real, node_complex, real_family, rates, filtered, exposures,
                   taps_view, diagnostics, coefficients, parameters,
                   channel_coefficients, channel_preionisation, channels, multiple,
                   n_pol, samples, intervals, dt, log_field_min, d_log_field, field_min,
                   field_max, rate_max, half, reach, last, first_node, output,
                   preionisation, league);
    const auto slot = static_cast<std::size_t>(team.league_rank());
    double taps[plasma_max_taps] = {};
    for (int k = 0; k <= half; ++k)
      taps[k] = taps_view(static_cast<std::size_t>(k));
    const auto tile = band_tile<band_staging>(team);
    for (std::size_t node = slot; node < node_count; node += league) {
      const auto base = node * n_pol;
      const auto field_at = [&](const std::size_t row, const std::size_t time) {
        return real_family ? node_real(row, time) : node_complex(row, time).real();
      };
      double parts[plasma_diagnostic_parts] = {};
      for (std::size_t channel = 0; channel < channels; ++channel) {
        const double initial =
            multiple ? channel_preionisation(0, channel) : preionisation;
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
        // With vector maths a host thread forms blocks of rates as the column does
        // (`band_rates_block`).
        if constexpr (vmath::vectorised) {
          const double lower = multiple ? parameters(channel, 2) : field_min;
          const double upper = multiple ? parameters(channel, 3) : field_max;
          const double highest = multiple ? parameters(channel, 4) : rate_max;
          const auto within = [&](double* values, const std::size_t live) {
            if (multiple)
              ionisation_rates_within(values, live, channel_coefficients,
                                      parameters(channel, 0), parameters(channel, 1),
                                      static_cast<std::size_t>(parameters(channel, 5)),
                                      static_cast<std::size_t>(parameters(channel, 6)));
            else
              ionisation_rates_within(values, live, coefficients, log_field_min,
                                      d_log_field, intervals);
          };
          const auto sample_count = static_cast<std::size_t>(samples);
          const auto blocks =
              static_cast<int>((sample_count + rate_block - 1) / rate_block);
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, blocks), [&](const int block) {
                const auto start = static_cast<std::size_t>(block) * rate_block;
                const auto count = std::min(rate_block, sample_count - start);
                double magnitudes[rate_block];
                for (std::size_t k = 0; k < count; ++k) {
                  const double first = field_at(base, start + k);
                  magnitudes[k] =
                      n_pol == 2 ? Kokkos::hypot(first, field_at(base + 1, start + k))
                                 : Kokkos::abs(first);
                }
                band_rates_block(magnitudes, &rates(slot, start), count, lower, upper,
                                 highest, within);
              });
        } else {
#endif
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, samples), [&](const int sample) {
                const auto time = static_cast<std::size_t>(sample);
                const double first = field_at(base, time);
                const double magnitude =
                    n_pol == 2 ? Kokkos::hypot(first, field_at(base + 1, time))
                               : Kokkos::abs(first);
                rates(slot, time) =
                    multiple
                        ? ionisation_rate(
                              magnitude, channel_coefficients, parameters(channel, 0),
                              parameters(channel, 1), parameters(channel, 2),
                              parameters(channel, 3), parameters(channel, 4),
                              static_cast<std::size_t>(parameters(channel, 5)),
                              static_cast<std::size_t>(parameters(channel, 6)))
                        : ionisation_rate(magnitude, coefficients, log_field_min,
                                          d_log_field, field_min, field_max, rate_max,
                                          intervals);
              });
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
        }
#endif
        team.team_barrier();
        team_band_filter<band_staging>(team, tile, rates, slot, filtered, slot, samples,
                                       taps, half);
        team_trapezoid_prefix(team, filtered, slot, samples, dt,
                              [&](const int sample, const double prefix) {
                                exposures(slot, static_cast<std::size_t>(sample)) =
                                    prefix;
                              });
        double largest = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, samples),
            [&](const int sample, double& update) {
              const double rate = rates(slot, static_cast<std::size_t>(sample));
              if (rate > update)
                update = rate;
            },
            Kokkos::Max<double>(largest));
        const double exposure = exposures(slot, last);
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          output(first_node + node, 2 * channel) = exposure > 0.0 ? exposure : 0.0;
          output(first_node + node, 2 * channel + 1) = largest > 0.0 ? largest : 0.0;
        });
        if constexpr (Diagnose) {
          // The exposure's increments, as the scan formed them.
          const auto increment = [&](const int sample) {
            const auto time = static_cast<std::size_t>(sample);
            const double current = filtered(slot, time);
            const double previous = filtered(slot, time - 1);
            return dt * (current + previous) / 2.0;
          };
          double dip = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, 1, samples),
              [&](const int sample, double& update) {
                const double fall = -increment(sample);
                if (fall > update)
                  update = fall;
              },
              Kokkos::Max<double>(dip));
          double rise = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, 1, samples),
              [&](const int sample, double& update) {
                const double step = increment(sample);
                if (step > update)
                  update = step;
              },
              Kokkos::Max<double>(rise));
          team.team_barrier();
          // The filtered row becomes the owed work, the exposure row the fraction.
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
          // With vector maths a host thread takes blocks: the decays in one pass,
          // then `decayed_fractions`, as the column forms them.
          if constexpr (vmath::vectorised) {
            const auto sample_count = static_cast<std::size_t>(samples);
            const auto blocks =
                static_cast<int>((sample_count + rate_block - 1) / rate_block);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, blocks), [&](const int block) {
                  const auto start = static_cast<std::size_t>(block) * rate_block;
                  const auto count = std::min(rate_block, sample_count - start);
                  double* const exposed = &exposures(slot, start);
                  double* const owed = &filtered(slot, start);
                  const double* const rate = &rates(slot, start);
                  double fraction[rate_block];
                  for (std::size_t k = 0; k < count; ++k)
                    fraction[k] = -(exposed[k] < 0.0 ? 0.0 : exposed[k]);
                  vmath::exp(fraction, fraction, count);
                  decayed_fractions(fraction, count, initial);
                  for (std::size_t k = 0; k < count; ++k) {
                    owed[k] = rate[k] * (1.0 - fraction[k]);
                    exposed[k] = fraction[k];
                  }
                });
          } else {
#endif
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, samples), [&](const int sample) {
                  const auto time = static_cast<std::size_t>(sample);
                  const double fraction = band_fraction(exposures(slot, time), initial);
                  filtered(slot, time) = rates(slot, time) * (1.0 - fraction);
                  exposures(slot, time) = fraction;
                });
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
          }
#endif
          team.team_barrier();
          double paid = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, 1, samples),
              [&](const int sample, double& sum) { sum += increment(sample); }, paid);
          const auto level = [&](const int sample, double& update) {
            const auto time = static_cast<std::size_t>(sample);
            const double fraction = exposures(slot, time);
            for (std::size_t component = 0; component < n_pol; ++component) {
              const double value =
                  Kokkos::abs(fraction * field_at(base + component, time));
              if (value > update)
                update = value;
            }
          };
          double early = 0.0;
          Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, reach), level,
                                  Kokkos::Max<double>(early));
          double peak = 0.0;
          Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, samples), level,
                                  Kokkos::Max<double>(peak));
          const double freed = exposures(slot, last) - exposures(slot, 0);
          const double values[plasma_diagnostic_parts] = {
              dip > 0.0 ? dip : 0.0,     rise > 0.0 ? rise : 0.0,
              Kokkos::abs(paid - freed), freed,
              early > 0.0 ? early : 0.0, peak > 0.0 ? peak : 0.0};
          for (std::size_t part = 0; part < plasma_diagnostic_parts; ++part)
            parts[part] = Kokkos::max(parts[part], values[part]);
        }
        team.team_barrier();
      }
      if constexpr (Diagnose)
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          for (std::size_t part = 0; part < plasma_diagnostic_parts; ++part)
            diagnostics(first_node + node, part) = parts[part];
        });
      team.team_barrier();
    }
  };
  launch::parallel_for("solprop_native.plasma.team_band_statistics",
                       band_team_policy(node_count, scan), scan);
}

void GeneralWorkspace::convolve_raman(const std::size_t stage,
                                      const std::size_t node_count) {
  const bool tail = node_count != problem_.dims.node_chunk;
  Kokkos::Profiling::pushRegion("solprop_native.raman.drive_fft");
  execute_raman(tail ? *raman_tail_forward_plan_ : *raman_drive_forward_plan_,
                tail ? raman_tail_drive_ : node_raman_drive_,
                tail ? raman_tail_spectrum_ : node_raman_spectrum_,
                KokkosFFT::Normalization::none);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("solprop_native.raman.spectral_multiply");
  const auto node_spectrum = node_raman_spectrum_;
  const auto kernel_spectrum = raman_kernel_spectrum_;
  const auto drives = problem_.raman_drives();
  const double inverse_doubled = 1.0 / static_cast<double>(2 * problem_.dims.n_to);
  launch::parallel_for(
      "solprop_native.raman.spectral_multiply",
      Range2(tail ? raman_tail_multiply_space_ : raman_multiply_space_, {0, 0},
             {node_count * drives, problem_.dims.n_to + 1}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t index) {
        node_spectrum(node, index) =
            (node_spectrum(node, index) *
             kernel_spectrum(node % drives == 0 ? 0 : 1, index)) *
            inverse_doubled;
      });
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("solprop_native.raman.inverse_fft");
  execute_raman(tail ? *raman_tail_inverse_plan_ : *raman_convolution_inverse_plan_,
                tail ? raman_tail_spectrum_ : node_raman_spectrum_,
                tail ? raman_tail_convolved_ : node_raman_convolved_,
                KokkosFFT::Normalization::none);
  Kokkos::Profiling::popRegion();
  finish_raman(stage, node_count);
}

void GeneralWorkspace::finish_raman(const std::size_t stage,
                                    const std::size_t node_count) {
  Kokkos::Profiling::pushRegion("solprop_native.raman.finish");
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto out_real = polarisation_real_;
  const auto out_complex = polarisation_complex_;
  const auto convolved = node_raman_convolved_;
  const auto window = time_window_;
  const auto density = stage_raman_density_;
  const auto family = problem_.dims.family;
  const auto components = problem_.effective_node_pol();
  const auto drives = problem_.raman_drives();
  const double epsilon0 = problem_.raman.epsilon0;
  launch::parallel_for(
      "solprop_native.raman.finish",
      Range2(response_space_for(node_count), {0, 0}, {node_count, problem_.dims.n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        if (components == 2) {
          const auto base = 2 * node;
          const auto row = drives * node;
          const double u = convolved(row, time);
          const double v = drives == 3 ? convolved(row + 1, time) : 0.0;
          const double w = drives == 3 ? convolved(row + 2, time) : 0.0;
          const double factor = epsilon0 * density(stage);
          if (family == FieldFamily::envelope) {
            const Complex x = node_complex(base, time);
            const Complex y = node_complex(base + 1, time);
            out_complex(base, time) =
                (out_complex(base, time) + factor * ((u + v) * x + w * y)) *
                window(time);
            out_complex(base + 1, time) =
                (out_complex(base + 1, time) + factor * (w * x + (u - v) * y)) *
                window(time);
          } else {
            const double x = family == FieldFamily::field_real
                                 ? node_real(base, time)
                                 : node_complex(base, time).real();
            const double y = family == FieldFamily::field_real
                                 ? node_real(base + 1, time)
                                 : node_complex(base + 1, time).real();
            out_real(base, time) =
                (out_real(base, time) + factor * ((u + v) * x + w * y)) * window(time);
            out_real(base + 1, time) =
                (out_real(base + 1, time) + factor * (w * x + (u - v) * y)) *
                window(time);
          }
          return;
        }
        const double factor = epsilon0 * density(stage) * convolved(node, time);
        if (family == FieldFamily::envelope) {
          out_complex(node, time) =
              (out_complex(node, time) + factor * node_complex(node, time)) *
              window(time);
        } else {
          const double field = family == FieldFamily::field_real
                                   ? node_real(node, time)
                                   : node_complex(node, time).real();
          out_real(node, time) = (out_real(node, time) + factor * field) * window(time);
        }
      });
  Kokkos::Profiling::popRegion();
}

// The modal-product convolution (after the CPU experiment's Raman port). With a real
// static synthesis S, a scalar node field is x_j = sum_m S_jm a_m, so the Raman drive
// x_j^2 (or the cycle average |x_j|^2 / 2) is sum_{m <= n} c_mn S_jm S_jn P_mn with
// P_mn = Re(a_m) Re(a_n) (or (Re(a_m conj a_n)) / 2) and c_mn = 2 off the diagonal.
// The causal convolution is linear: convolving each of the M (M + 1) / 2 products once
// and reconstructing each node's convolved drive from them replaces one drive per node.
// It is chosen where it transforms fewer rows than the nodes and its rows fit the node
// drive's own scratch; the stage-dependent kernel, its support and the taper (inside
// S) are those of the node path. The reconstruction sums the products in a different
// order from squaring the synthesised field: the two agree to rounding of the drive's
// scale, sum_m |S_jm a_m| squared. Host builds only: a device keeps the node drives.
std::size_t GeneralWorkspace::raman_modal_products() const noexcept {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  return 0;
#else
  if (!problem_.raman.enabled || problem_.general.enabled ||
      problem_.dispersive.enabled || problem_.separable_geometry() ||
      problem_.effective_node_pol() != 1 || problem_.dims.n_pol_state != 1 ||
      problem_.dims.n_pol_transform != 1 || problem_.raman_drives() != 1 ||
      problem_.dims.n_nodes < 2)
    return 0;
  const auto products = problem_.dims.n_modes * (problem_.dims.n_modes + 1) / 2;
  return products < problem_.dims.n_nodes && products <= problem_.dims.node_chunk
             ? products
             : 0;
#endif
}

void GeneralWorkspace::convolve_raman_products() {
  Kokkos::Profiling::pushRegion("solprop_native.raman.products");
  const auto drive = raman_product_drive_;
  const auto modal_real = modal_real_;
  const auto modal_complex = modal_complex_;
  const auto modes = problem_.dims.n_modes;
  const auto n_to = problem_.dims.n_to;
  const bool real_family = problem_.dims.family == FieldFamily::field_real;
  const bool squared = problem_.raman.thg;
  launch::parallel_for(
      "solprop_native.raman.products",
      Range2(raman_product_space_, {0, 0}, {raman_products_, 2 * n_to}),
      KOKKOS_LAMBDA(const std::size_t product, const std::size_t time) {
        if (time >= n_to) {
          drive(product, time) = 0.0;
          return;
        }
        // The unordered pair (first, second), first <= second, in row-major order.
        std::size_t first = 0;
        std::size_t second = product;
        while (second >= modes - first) {
          second -= modes - first;
          ++first;
        }
        second += first;
        const double multiplicity = first == second ? 1.0 : 2.0;
        double value;
        if (real_family) {
          value = modal_real(first, time) * modal_real(second, time);
          if (!squared)
            value *= 0.5;
        } else {
          const Complex a = modal_complex(first, time);
          const Complex b = modal_complex(second, time);
          value = squared ? a.real() * b.real()
                          : 0.5 * (a.real() * b.real() + a.imag() * b.imag());
        }
        drive(product, time) = multiplicity * value;
      });
  execute_raman(*raman_product_forward_plan_, raman_product_drive_,
                raman_product_spectrum_, KokkosFFT::Normalization::none);
  const auto spectrum = raman_product_spectrum_;
  const auto kernel_spectrum = raman_kernel_spectrum_;
  const double inverse_doubled = 1.0 / static_cast<double>(2 * n_to);
  launch::parallel_for(
      "solprop_native.raman.products_multiply",
      Range2(raman_product_space_, {0, 0}, {raman_products_, n_to + 1}),
      KOKKOS_LAMBDA(const std::size_t product, const std::size_t index) {
        spectrum(product, index) =
            (spectrum(product, index) * kernel_spectrum(0, index)) * inverse_doubled;
      });
  execute_raman(*raman_product_inverse_plan_, raman_product_spectrum_,
                raman_product_drive_, KokkosFFT::Normalization::none);
  Kokkos::Profiling::popRegion();
}

// `finish_raman`'s scalar form with each node's convolved drive reconstructed from
// the convolved modal products.
void GeneralWorkspace::finish_raman_products(const std::size_t stage,
                                             const std::size_t first_node,
                                             const std::size_t node_count) {
  Kokkos::Profiling::pushRegion("solprop_native.raman.finish_products");
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto out_real = polarisation_real_;
  const auto out_complex = polarisation_complex_;
  const auto convolved = raman_product_drive_;
  const auto synthesis = scaled_synthesis_;
  const auto window = time_window_;
  const auto density = stage_raman_density_;
  const auto family = problem_.dims.family;
  const auto modes = problem_.dims.n_modes;
  const double epsilon0 = problem_.raman.epsilon0;
  const auto n_to = problem_.dims.n_to;
  // Blocks of samples per node, the products accumulated in their order over each
  // block as contiguous rows: one pass over the block per product, which vectorises,
  // where a per-sample loop over the products strides between rows.
  constexpr std::size_t block = 512;
  const std::size_t blocks = (n_to + block - 1) / block;
  const double* const products = convolved.data();
  const std::size_t stride = convolved.stride(0);
  launch::parallel_for(
      "solprop_native.raman.finish_products",
      Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0,
                                          node_count * blocks),
      KOKKOS_LAMBDA(const std::size_t item) {
        const std::size_t node = item / blocks;
        const std::size_t begin = (item % blocks) * block;
        const std::size_t count = n_to - begin < block ? n_to - begin : block;
        const std::size_t row = first_node + node;
        double drive[block];
        for (std::size_t offset = 0; offset < count; ++offset)
          drive[offset] = 0.0;
        std::size_t product = 0;
        for (std::size_t first = 0; first < modes; ++first) {
          const double weight = synthesis(row, first);
          for (std::size_t second = first; second < modes; ++second, ++product) {
            const double pair = weight * synthesis(row, second);
            const double* const values = products + product * stride + begin;
            for (std::size_t offset = 0; offset < count; ++offset)
              drive[offset] += pair * values[offset];
          }
        }
        const double scale = epsilon0 * density(stage);
        for (std::size_t offset = 0; offset < count; ++offset) {
          const std::size_t time = begin + offset;
          const double factor = scale * drive[offset];
          if (family == FieldFamily::envelope) {
            out_complex(node, time) =
                (out_complex(node, time) + factor * node_complex(node, time)) *
                window(time);
          } else {
            const double field = family == FieldFamily::field_real
                                     ? node_real(node, time)
                                     : node_complex(node, time).real();
            out_real(node, time) =
                (out_real(node, time) + factor * field) * window(time);
          }
        }
      });
  Kokkos::Profiling::popRegion();
}

// One channel's cycle-averaged rates at each node and their running integral, from
// the polarisation ellipse of `field_rows`' `n_pol` rows per node: the node buffer
// with its components, or an envelope's on-axis row (`axis_populations`).
void GeneralWorkspace::envelope_cycle(const std::size_t channel,
                                      const std::size_t node_count,
                                      const ComplexView2& field_rows,
                                      const std::size_t n_pol) {
  // The kernels capture the view by value, never the caller's reference, which a
  // device cannot read.
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const auto field = field_rows;
  const auto parameters = channel_parameters_;
  const auto coefficients = channel_coefficients_;
  const auto nodes =
      Kokkos::subview(envelope_rule_nodes_, channel, Kokkos::ALL(), Kokkos::ALL());
  const auto weights =
      Kokkos::subview(envelope_rule_weights_, channel, Kokkos::ALL(), Kokkos::ALL());
  const auto moments = envelope_moments_;
  const auto rates = envelope_rates_;
  const auto failures = envelope_failures_;
  const auto n_to = problem_.dims.n_to;
  launch::parallel_for(
      "solprop_native.envelope.cycle_moments",
      Range2(response_space_for(node_count), {0, 0}, {node_count, n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        const auto axes =
            envelope_axes(field(node * n_pol, time),
                          n_pol == 2 ? field(node * n_pol + 1, time) : Complex{});
        const auto rate_at = [&](const double value) {
          return ionisation_rate(value, coefficients, parameters(channel, 0),
                                 parameters(channel, 1), parameters(channel, 2),
                                 parameters(channel, 3), parameters(channel, 4),
                                 static_cast<std::size_t>(parameters(channel, 5)),
                                 static_cast<std::size_t>(parameters(channel, 6)));
        };
        const auto pair =
            envelope_moments(axes.major, axes.ratio, parameters(channel, 2),
                             parameters(channel, 3), rate_at, nodes, weights);
        moments(0, node, time) = pair.cosine;
        moments(1, node, time) = pair.sine;
        rates(node, time) = pair.cosine + axes.ratio * axes.ratio * pair.sine;
        failures(node * n_to + time) = pair.failed ? 1 : 0;
      });
  const auto status = envelope_status_;
  std::size_t no_failure = 0;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // A serial device scan dominated the short combined Raman/plasma profiles.
  // Integer minimum preserves the first failing sample exactly;
  // its identity denotes success without an additional normalization launch.
  no_failure = std::numeric_limits<std::size_t>::max();
  using StatusPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using StatusMember = StatusPolicy::member_type;
  const auto samples = node_count * n_to;
  // One team uses only launch-time shared memory; a global reduction could grow
  // the execution-space scratch allocator inside the first trial.
  launch::parallel_for(
      "solprop_native.envelope.angular_status",
      StatusPolicy(response_space_for(node_count), 1,
                   SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE),
      KOKKOS_LAMBDA(const StatusMember& team) {
        std::size_t first_failure = Kokkos::reduction_identity<std::size_t>::min();
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, samples),
            [&](const std::size_t sample, std::size_t& minimum) {
              if (failures(sample) != 0)
                minimum = Kokkos::min(minimum, sample + 1);
            },
            Kokkos::Min<std::size_t>(first_failure));
        Kokkos::single(Kokkos::PerTeam(team), [&]() { status(0) = first_failure; });
      });
#else
  launch::parallel_for(
      "solprop_native.envelope.angular_status",
      Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0, 1),
      KOKKOS_LAMBDA(const std::size_t) {
        status(0) = 0;
        for (std::size_t sample = 0; sample < node_count * n_to; ++sample)
          if (failures(sample) != 0) {
            status(0) = sample + 1;
            break;
          }
      });
#endif
  // This planned scalar boundary rejects failed refinement before a current FFT.
  graph::deep_copy(response_space_for(node_count), envelope_host_status_,
                   envelope_status_);
  response_space_for(node_count).fence("solprop_native envelope refinement boundary");
  if (envelope_host_status_(0) != no_failure)
    refuse<EnvelopeConvergenceError>("envelope plasma channel ", channel,
                                     " exceeded the angular refinement budget");
  const auto rows = std::make_pair(std::size_t{0}, node_count);
  portable_team_trapezoid_scan<true>(
      response_space_for(node_count),
      Kokkos::subview(envelope_rates_, rows, Kokkos::ALL()),
      Kokkos::subview(envelope_exposure_, rows, Kokkos::ALL()), problem_.grid.dt_fine);
}

// One channel's exposure and peak cycle-averaged rate at each node of `field`
// (`envelope_cycle`), into the rows of `statistics` from `first_node`.
void GeneralWorkspace::envelope_statistics(const std::size_t channel,
                                           const std::size_t first_node,
                                           const std::size_t node_count,
                                           const RealView2& statistics,
                                           const ComplexView2& field,
                                           const std::size_t n_pol) {
  envelope_cycle(channel, node_count, field, n_pol);
  const auto rates = envelope_rates_;
  const auto exposure = envelope_exposure_;
  const auto n_to = problem_.dims.n_to;
  launch::parallel_for(
      "solprop_native.envelope.population_statistics",
      Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0,
                                          node_count),
      KOKKOS_LAMBDA(const std::size_t node) {
        double peak = 0.0;
        for (std::size_t time = 0; time < n_to; ++time)
          peak = Kokkos::max(peak, rates(node, time));
        statistics(first_node + node, 2 * channel) = exposure(node, n_to - 1);
        statistics(first_node + node, 2 * channel + 1) = peak;
      });
}

void GeneralWorkspace::envelope_population(const std::size_t channel,
                                           const std::size_t stage,
                                           const std::size_t node_count) {
  const auto n_pol = problem_.effective_node_pol();
  envelope_cycle(channel, node_count, node_complex_, n_pol);
  const auto exposure = envelope_exposure_;
  const auto n_to = problem_.dims.n_to;
  const auto parameters = channel_parameters_;
  const auto densities = stage_channel_density_;
  const auto initials = stage_channel_preionisation_;
  const auto moments = envelope_moments_;
  const auto field = node_complex_;
  const auto window = time_window_;
  const auto drive = envelope_streams_[0].node;
  const auto loss = envelope_streams_[1].node;
  launch::parallel_for(
      "solprop_native.envelope.currents",
      Range2(response_space_for(node_count), {0, 0}, {node_count, n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        const double initial = initials(stage, channel);
        const double fraction = envelope_fraction(exposure(node, time), initial);
        const double survival = (1.0 - initial) * Kokkos::exp(-exposure(node, time));
        const double density = densities(stage, channel);
        const auto axes =
            envelope_axes(field(node * n_pol, time),
                          n_pol == 2 ? field(node * n_pol + 1, time) : Complex{});
        const auto current =
            envelope_loss(axes, moments(0, node, time), moments(1, node, time),
                          parameters(channel, 8) * density * survival);
        const double drude = parameters(channel, 7) * density * fraction;
        for (std::size_t pol = 0; pol < n_pol; ++pol) {
          const auto row = node * n_pol + pol;
          const auto drive_value = window(time) * drude * field(row, time);
          const auto loss_value = window(time) * current[pol];
          if (channel == 0) {
            drive(row, time) = drive_value;
            loss(row, time) = loss_value;
          } else {
            drive(row, time) += drive_value;
            loss(row, time) += loss_value;
          }
        }
      });
}

void GeneralWorkspace::envelope_response(const std::size_t stage,
                                         const std::size_t first_node,
                                         const std::size_t node_count,
                                         const TransverseChunk* chunk) {
  for (std::size_t channel = 0; channel < problem_.plasma.n_channels; ++channel)
    envelope_population(channel, stage, node_count);
  for (auto& stream : envelope_streams_) {
    if (problem_.slab_geometry()) {
      // As the polarisation: the chunk's y rows, zero beyond a final partial chunk.
      graph::deep_copy(response_space_for(node_count), stream.projected, Complex{});
      slab_project(*chunk, &stream);
    } else if (problem_.free_geometry()) {
      // The persistent maximum-chunk plan transforms zero padding in a final
      // partial chunk; only its actual rows are ever copied to the spatial FFT.
      graph::deep_copy(response_space_for(node_count), stream.projected, Complex{});
      const auto rows =
          std::make_pair(std::size_t{0}, node_count * problem_.effective_node_pol());
      graph::deep_copy(response_space_for(node_count),
                       Kokkos::subview(stream.projected, rows, Kokkos::ALL()),
                       Kokkos::subview(stream.node, rows, Kokkos::ALL()));
    } else {
      project(first_node, node_count, first_node == 0, &stream);
    }
  }
}

// SEP 27 forms two independently collared current products. Every maximum-chunk
// row is written, including a partial chunk's padding, before a persistent FFT plan
// can read it. Modal projection precedes the FFT; radial projection follows it.
void GeneralWorkspace::held_response(const std::size_t stage,
                                     const std::size_t first_node,
                                     const std::size_t node_count,
                                     const TransverseChunk* chunk) {
  const auto current = held_current_;
  const auto window = time_window_;
  const auto derivative = held_derivative_;
  const auto product = held_streams_[0].node;
  const auto edge = held_streams_[1].node;
  const auto live = node_count * problem_.effective_node_pol();
  const auto n_to = problem_.dims.n_to;
  cartesian_for(
      "solprop_native.held.products", response_space_for(node_count), current.extent(0),
      n_to, KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
        const double value = row < live ? current(row, time) : 0.0;
        product(row, time) = window(time) * value;
        edge(row, time) = derivative(time) * value;
      });
  const bool folded = folds_single_node();
  const bool radial = held_radial();
  for (std::size_t index = 0; index < 2; ++index) {
    auto& stream = held_streams_[index];
    if (folded) {
      const auto source = stream.node;
      const auto target = stream.projected;
      const auto projection = projection_;
      const auto taper = stage_taper_;
      cartesian_for(
          "solprop_native.held.folded_projection", response_space_for(node_count), 1,
          n_to, KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
            target(row, time) = projection(0, 0) / taper(stage) * source(row, time);
          });
    } else if (problem_.slab_geometry()) {
      graph::deep_copy(response_space_for(node_count), stream.projected, 0.0);
      slab_contract("solprop_native.held.slab_projection",
                    response_space_for(node_count), projection_, stream.node,
                    stream.projected, chunk->nodes / problem_.spatial.n_x, n_to);
    } else if (problem_.free_geometry() || radial) {
      graph::deep_copy(response_space_for(node_count), stream.projected, stream.node);
    } else {
      project(first_node, node_count, first_node == 0, nullptr, &stream);
    }
  }
  if (radial) {
    const auto source = polarisation_real_;
    const auto target = held_streams_[2].projected;
    cartesian_for(
        "solprop_native.held.nonplasma", response_space_for(node_count),
        target.extent(0), n_to,
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
          target(row, time) = row < live ? source(row, time) : 0.0;
        });
    held_radial_projection(first_node, node_count);
  }
}

// solprop's private gated benchmark (`NumpyRefBackend._column_sources`): the chunk's
// live current rows are integrated by the problem's running-integral rule, the band
// rule's filter then the trapezoid (solprop SEP 18), and w(t) times each prefix joins
// the local polarisation, which is then projected as the held current's streams would
// have been. The scans are the device column's portable primitives; no hot column
// changes, and a run without the benchmark never reaches this.
void GeneralWorkspace::gated_response(const std::size_t node_count) {
  const Kokkos::Profiling::ScopedRegion region("solprop_native.gated.response");
  const auto live = std::pair<std::size_t, std::size_t>(
      0, node_count * problem_.effective_node_pol());
  const auto current = Kokkos::subview(held_current_, live, Kokkos::ALL);
  const auto prefix = Kokkos::subview(gated_prefix_, live, Kokkos::ALL);
  const auto& space = response_space_for(node_count);
  if (problem_.plasma.band)
    portable_team_band_scan(space, current,
                            Kokkos::subview(gated_filtered_, live, Kokkos::ALL), prefix,
                            plasma_taps_, problem_.grid.dt_fine);
  else
    portable_team_trapezoid_scan(space, current, prefix, problem_.grid.dt_fine);
  const auto window = time_window_;
  const auto polarisation = polarisation_real_;
  cartesian_for(
      "solprop_native.gated.polarisation", space, live.second, problem_.dims.n_to,
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
        polarisation(row, time) += window(time) * prefix(row, time);
      });
}

// A stage's forward rows in one transform: all of them when the absorber's are due,
// else those after the absorber's (or the projection's alone). A host deferring its
// quadrature transforms the fine block only; `finish_quadrature` takes the coarse.
void GeneralWorkspace::transform_forward_pack() {
  const Kokkos::Profiling::ScopedRegion region("solprop_native.fft.forward_pack");
  const bool fine = defer_quadrature_ && forward_pack_fine_plan_;
  if (absorbing() || forward_pack_absorber_rows_ == 0) {
    if (fine)
      graph::execute(*forward_pack_fine_plan_, forward_pack_fine_,
                     forward_pack_fine_spectrum_, KokkosFFT::Normalization::none);
    else
      graph::execute(*forward_pack_plan_, forward_pack_, forward_pack_spectrum_,
                     KokkosFFT::Normalization::none);
  } else if (fine ? static_cast<bool>(forward_pack_fine_tail_plan_)
                  : static_cast<bool>(forward_pack_tail_plan_)) {
    if (fine)
      graph::execute(*forward_pack_fine_tail_plan_, forward_pack_fine_tail_,
                     forward_pack_fine_tail_spectrum_, KokkosFFT::Normalization::none);
    else
      graph::execute(*forward_pack_tail_plan_, forward_pack_tail_,
                     forward_pack_tail_spectrum_, KokkosFFT::Normalization::none);
  } else {
    graph::execute(*forward_real_plan_, projected_real_, projected_spectrum_,
                   KokkosFFT::Normalization::none);
  }
}

void GeneralWorkspace::held_transform(const bool coarse_only, const bool transformed) {
  const Kokkos::Profiling::ScopedRegion region("solprop_native.fft.held_streams");
  // The packed rows are every transform this call takes when it takes the fine ones
  // and, where there are coarse rows, those too. The stage's forward pack, when it
  // took them, has transformed the streams' rows already.
  const bool every_row =
      !coarse_only && (!held_streams_[0].coarse_forward || !defer_quadrature_);
  if (held_pack_forward_ && every_row) {
    graph::execute(*held_pack_forward_, held_pack_, held_pack_spectrum_,
                   KokkosFFT::Normalization::none);
  } else if (!transformed) {
    for (std::size_t index = 0; index < 2; ++index) {
      auto& stream = held_streams_[index];
      if (!coarse_only)
        graph::execute(*stream.forward, stream.projected, stream.spectrum,
                       KokkosFFT::Normalization::none);
      if (stream.coarse_forward && (coarse_only || !defer_quadrature_))
        graph::execute(*stream.coarse_forward, stream.coarse, stream.coarse_spectrum,
                       KokkosFFT::Normalization::none);
    }
  }
  const auto product = held_streams_[0].spectrum;
  const auto edge = held_streams_[1].spectrum;
  const auto coarse_product = held_streams_[0].coarse_spectrum;
  const auto coarse_edge = held_streams_[1].coarse_spectrum;
  const auto projected = projected_spectrum_;
  const auto coarse = projected_coarse_spectrum_;
  const auto inverse = held_inverse_;
  const auto active = active_mask_;
  const auto crop = crop_index_;
  const bool include_coarse = !coarse.empty() && (coarse_only || !defer_quadrature_);
  cartesian_for(
      "solprop_native.held.spectral_assembly", execution_space_, projected.extent(0),
      problem_.dims.n_omega,
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        const auto bin = crop(omega);
        if (active(omega)) {
          const double value = inverse(omega);
          const Complex factor(0.0, -value);
          if (!coarse_only)
            projected(row, bin) +=
                factor * product(row, bin) + value * value * edge(row, bin);
          if (include_coarse)
            coarse(row, bin) += factor * coarse_product(row, bin) +
                                value * value * coarse_edge(row, bin);
        } else {
          if (!coarse_only)
            projected(row, bin) = Complex{};
          if (include_coarse)
            coarse(row, bin) = Complex{};
        }
      });
}

void GeneralWorkspace::held_radial_projection(const std::size_t first_node,
                                              const std::size_t node_count) {
  if (held_pack_forward_)
    graph::execute(*held_pack_forward_, held_pack_, held_pack_spectrum_,
                   KokkosFFT::Normalization::none);
  else
    for (auto& stream : held_streams_)
      graph::execute(*stream.forward, stream.projected, stream.spectrum,
                     KokkosFFT::Normalization::none);
  const auto product = held_streams_[0].spectrum;
  const auto edge = held_streams_[1].spectrum;
  const auto polarisation = held_streams_[2].spectrum;
  const auto nodal = held_nodal_spectrum_;
  const auto inverse = held_inverse_;
  const auto active = active_mask_;
  const auto crop = crop_index_;
  const auto n = problem_.dims.n_omega;
  const auto live = node_count * problem_.effective_node_pol();
  cartesian_for(
      "solprop_native.held.radial_assembly", node_spectral_space_, nodal.extent(0), n,
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        Complex value{};
        if (row < live && active(omega)) {
          const auto bin = crop(omega);
          const double inv = inverse(omega);
          value = polarisation(row, bin) + Complex(0.0, -inv) * product(row, bin) +
                  inv * inv * edge(row, bin);
        }
        nodal(row, omega) = value;
      });
  const auto packed = held_projection_;
  const auto projection = scaled_projection_;
  const auto first_row = first_node * problem_.dims.n_pol_transform;
  const auto live_rows = node_count * problem_.dims.n_pol_transform;
  cartesian_for(
      "solprop_native.held.radial_projection_matrix", matrix_copy_space_,
      packed.extent(0), packed.extent(1),
      KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
        packed(mode, row) = row < live_rows ? projection(mode, first_row + row) : 0.0;
      });
  // Interleaved real and imaginary parts share the scalar radial projection. This
  // is one GEMM of the combined cropped source, independent of state components.
  using Interleaved = Kokkos::View<double**, Layout, MemorySpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  static_assert(sizeof(Complex) == 2 * sizeof(double));
  const auto columns = 2 * problem_.dims.n_pol_state * n;
  const Interleaved input(reinterpret_cast<double*>(nodal.data()), packed.extent(1),
                          columns);
  const Interleaved output(reinterpret_cast<double*>(held_projected_spectrum_.data()),
                           problem_.dims.n_modes, columns);
  selected_gemm(transform_space_for(node_count), packed, input, output, 0.0, true);
  if (first_node == 0)
    graph::deep_copy(temporal_execution_space_, projected_spectrum_, Complex{});
  const auto source = held_projected_spectrum_;
  const auto target = projected_spectrum_;
  cartesian_for(
      "solprop_native.held.radial_accumulate", execution_space_, target.extent(0), n,
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        if (active(omega))
          target(row, crop(omega)) += source(row, omega);
      });
}

void GeneralWorkspace::envelope_transform() {
  for (auto& stream : envelope_streams_) {
    graph::execute(*stream.forward, stream.projected, stream.spectrum,
                   KokkosFFT::Normalization::none);
    if (stream.coarse_forward)
      graph::execute(*stream.coarse_forward, stream.coarse, stream.coarse_spectrum,
                     KokkosFFT::Normalization::none);
  }
  const auto drive = envelope_streams_[0].spectrum;
  const auto loss = envelope_streams_[1].spectrum;
  const auto coarse_drive = envelope_streams_[0].coarse_spectrum;
  const auto coarse_loss = envelope_streams_[1].coarse_spectrum;
  const auto projected = projected_spectrum_;
  const auto coarse = projected_coarse_spectrum_;
  const auto drude_factor = envelope_drude_factor_;
  const auto loss_factor = envelope_loss_factor_;
  const auto crop = crop_index_;
  launch::parallel_for(
      "solprop_native.envelope.spectral_assembly",
      Range2(execution_space_, {0, 0}, {projected.extent(0), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        const auto index = crop(omega);
        projected(column, index) += drude_factor(omega) * drive(column, index) +
                                    loss_factor(omega) * loss(column, index);
        if (!coarse.empty())
          coarse(column, index) += drude_factor(omega) * coarse_drive(column, index) +
                                   loss_factor(omega) * coarse_loss(column, index);
      });
}

void GeneralWorkspace::project(const std::size_t first_node,
                               const std::size_t node_count, const bool first_chunk,
                               const EnvelopeStream* stream,
                               const FieldCurrentStream* field_stream) {
  const auto real_source =
      field_stream == nullptr ? polarisation_real_ : field_stream->node;
  const auto real_target =
      field_stream == nullptr ? projected_real_ : field_stream->projected;
  const auto real_coarse_target =
      field_stream == nullptr ? projected_coarse_real_ : field_stream->coarse;
  const auto real_stacked_target =
      field_stream == nullptr ? projected_stacked_real_ : field_stream->stacked;
  const auto node_source = stream == nullptr ? polarisation_complex_ : stream->node;
  const auto projected_target =
      stream == nullptr ? projected_complex_ : stream->projected;
  const auto coarse_target =
      stream == nullptr ? projected_coarse_complex_ : stream->coarse;
  const auto stacked_target =
      stream == nullptr ? projected_stacked_complex_ : stream->stacked;
  const bool whole_transform = first_node == 0 && node_count == problem_.dims.n_nodes &&
                               problem_.dims.n_pol_state == 1;
  if (whole_transform && !radial_blas_input_.empty()) {
    packed_radial_gemm(scaled_projection_, node_source, projected_target);
    return;
  }
  if (whole_transform && use_interleaved_projection()) {
    Kokkos::Profiling::pushRegion(
        "solprop_native.transform.projection.kokkosblas_interleaved");
    using Interleaved = Kokkos::View<double**, Layout, MemorySpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    static_assert(sizeof(Complex) == 2 * sizeof(double));
    const Interleaved nodes(reinterpret_cast<double*>(node_source.data()),
                            problem_.dims.n_nodes, 2 * problem_.dims.n_to);
    const Interleaved fine(reinterpret_cast<double*>(projected_target.data()),
                           problem_.dims.n_modes, 2 * problem_.dims.n_to);
    engine_gemm(transform_space_for(node_count), scaled_projection_, nodes, fine, 0.0);
    if (project_coarse_ && !problem_.transform.projection_coarse_matrix.empty()) {
      const Interleaved coarse(reinterpret_cast<double*>(coarse_target.data()),
                               problem_.dims.n_modes, 2 * problem_.dims.n_to);
      engine_gemm(transform_space_for(node_count), scaled_projection_coarse_, nodes,
                  coarse, 0.0);
    }
    Kokkos::Profiling::popRegion();
    return;
  }
  // The stacked projections' splits copy on the stage's instance: on the temporal
  // copy's serial instance one took about 90 us at 16 threads on the four-mode
  // capillary, beside full-width kernels (docs/dev/benchmarks.md, "Stacked projection
  // splits").
  if (whole_transform && use_blas_for_projection()) {
    Kokkos::Profiling::pushRegion("solprop_native.transform.projection.kokkosblas");
    if (problem_.dims.family == FieldFamily::envelope && project_coarse_ &&
        !scaled_projection_stacked_complex_.empty()) {
      selected_gemm(transform_space_for(node_count), scaled_projection_stacked_complex_,
                    node_source, stacked_target, Complex{}, true);
      const auto& stacked = stacked_target;
      const auto& fine = projected_target;
      const auto& coarse = coarse_target;
      const auto columns = problem_.n_columns();
      launch::parallel_for(
          "solprop_native.transform.projection.split_stacked_complex",
          Range2(execution_space_, {0, 0}, {columns, problem_.dims.n_to}),
          KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
            fine(column, time) = stacked(column, time);
            coarse(column, time) = stacked(columns + column, time);
          });
    } else if (problem_.dims.family == FieldFamily::envelope) {
      selected_gemm(transform_space_for(node_count), scaled_projection_complex_,
                    node_source, projected_target, Complex{}, true);
    } else if (one_mode_products_) {
      const bool stacked = project_coarse_ && !scaled_projection_stacked_.empty();
      one_mode_projection(response_space_for(node_count),
                          stacked ? scaled_projection_stacked_ : scaled_projection_,
                          real_source, real_target, real_coarse_target);
    } else if (project_coarse_ && !scaled_projection_stacked_.empty()) {
      selected_gemm(transform_space_for(node_count), scaled_projection_stacked_,
                    real_source, real_stacked_target, 0.0, true);
      const auto& stacked = real_stacked_target;
      const auto& fine = real_target;
      const auto& coarse = real_coarse_target;
      const auto columns = problem_.n_columns();
      launch::parallel_for(
          "solprop_native.transform.projection.split_stacked",
          Range2(execution_space_, {0, 0}, {columns, problem_.dims.n_to}),
          KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
            fine(column, time) = stacked(column, time);
            coarse(column, time) = stacked(columns + column, time);
          });
    } else {
      selected_gemm(transform_space_for(node_count), scaled_projection_, real_source,
                    real_target, 0.0, true);
    }
    Kokkos::Profiling::popRegion();
    return;
  }
  if (use_chunked_blas_for_projection()) {
    Kokkos::Profiling::pushRegion(
        "solprop_native.transform.projection.kokkosblas_chunked");
    const auto rows = node_count * problem_.dims.n_pol_transform;
    const auto first_row = first_node * problem_.dims.n_pol_transform;
    const bool tail = node_count != problem_.dims.node_chunk;
    const bool last_chunk = first_node + node_count == problem_.dims.n_nodes;
    const bool stacked =
        project_coarse_ && !problem_.transform.projection_coarse_matrix.empty();
    if (problem_.dims.family == FieldFamily::envelope) {
      const auto polarisation = Kokkos::subview(
          node_source, std::make_pair(std::size_t{0}, rows), Kokkos::ALL());
      if (stacked) {
        const auto packed =
            tail ? tail_projection_stacked_complex_ : chunk_projection_stacked_complex_;
        const auto source = scaled_projection_stacked_complex_;
        launch::parallel_for(
            "solprop_native.transform.pack_projection_stacked_complex",
            Range2(matrix_copy_space_for(node_count, true), {0, 0},
                   {2 * problem_.dims.n_modes, rows}),
            KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
              packed(mode, row) = source(mode, first_row + row);
            });
        selected_gemm(transform_space_for(node_count), packed, polarisation,
                      stacked_target, first_chunk ? Complex{} : Complex{1.0, 0.0},
                      true);
        if (last_chunk) {
          const auto& projected = stacked_target;
          const auto& fine = projected_target;
          const auto& coarse = coarse_target;
          const auto columns = problem_.n_columns();
          launch::parallel_for(
              "solprop_native.transform.projection.split_stacked_complex",
              Range2(execution_space_, {0, 0}, {columns, problem_.dims.n_to}),
              KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
                fine(column, time) = projected(column, time);
                coarse(column, time) = projected(columns + column, time);
              });
        }
      } else {
        // The first half of the existing stacked scratch is a contiguous fine
        // matrix. Alias it when this trial stage does not need the coarse rows.
        const auto scratch =
            tail ? tail_projection_stacked_complex_ : chunk_projection_stacked_complex_;
        const ComplexView2 packed =
            scratch.empty()
                ? (tail ? tail_projection_complex_ : chunk_projection_complex_)
                : ComplexView2(Kokkos::subview(
                      scratch, std::pair{std::size_t{0}, problem_.dims.n_modes},
                      Kokkos::ALL()));
        const auto source = scaled_projection_complex_;
        launch::parallel_for(
            "solprop_native.transform.pack_projection_complex",
            Range2(matrix_copy_space_for(node_count), {0, 0},
                   {problem_.dims.n_modes, rows}),
            KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
              packed(mode, row) = source(mode, first_row + row);
            });
        selected_gemm(transform_space_for(node_count), packed, polarisation,
                      projected_target, first_chunk ? Complex{} : Complex{1.0, 0.0},
                      true);
      }
    } else {
      const auto polarisation = Kokkos::subview(
          real_source, std::make_pair(std::size_t{0}, rows), Kokkos::ALL());
      if (stacked) {
        const auto packed = tail ? tail_projection_stacked_ : chunk_projection_stacked_;
        const auto source = scaled_projection_stacked_;
        launch::parallel_for(
            "solprop_native.transform.pack_projection_stacked",
            Range2(matrix_copy_space_for(node_count, true), {0, 0},
                   {2 * problem_.dims.n_modes, rows}),
            KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
              packed(mode, row) = source(mode, first_row + row);
            });
        selected_gemm(transform_space_for(node_count), packed, polarisation,
                      real_stacked_target, first_chunk ? 0.0 : 1.0, true);
        if (last_chunk) {
          const auto& projected = real_stacked_target;
          const auto& fine = real_target;
          const auto& coarse = real_coarse_target;
          const auto columns = problem_.n_columns();
          launch::parallel_for(
              "solprop_native.transform.projection.split_stacked",
              Range2(execution_space_, {0, 0}, {columns, problem_.dims.n_to}),
              KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
                fine(column, time) = projected(column, time);
                coarse(column, time) = projected(columns + column, time);
              });
        }
      } else {
        const auto scratch =
            tail ? tail_projection_stacked_ : chunk_projection_stacked_;
        const RealView2 packed =
            scratch.empty()
                ? (tail ? tail_projection_ : chunk_projection_)
                : RealView2(Kokkos::subview(
                      scratch, std::pair{std::size_t{0}, problem_.dims.n_modes},
                      Kokkos::ALL()));
        const auto source = scaled_projection_;
        launch::parallel_for(
            "solprop_native.transform.pack_projection",
            Range2(matrix_copy_space_for(node_count), {0, 0},
                   {problem_.dims.n_modes, rows}),
            KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
              packed(mode, row) = source(mode, first_row + row);
            });
        selected_gemm(transform_space_for(node_count), packed, polarisation,
                      real_target, first_chunk ? 0.0 : 1.0, true);
      }
    }
    Kokkos::Profiling::popRegion();
    return;
  }
  Kokkos::Profiling::pushRegion("solprop_native.transform.projection.custom");
  const auto projection = scaled_projection_;
  const auto projection_coarse = scaled_projection_coarse_;
  const auto& polar_real = real_source;
  const auto& polar_complex = node_source;
  const auto& projected_real = real_target;
  const auto& projected_complex = projected_target;
  const auto& projected_coarse_real = real_coarse_target;
  const auto& projected_coarse_complex = coarse_target;
  const auto n_pol_state = problem_.dims.n_pol_state;
  const auto n_pol_transform = problem_.dims.n_pol_transform;
  const auto n_to = problem_.dims.n_to;
  const bool envelope = problem_.dims.family == FieldFamily::envelope;
  const bool coarse = project_coarse_ && !projection_coarse.empty();
  launch::parallel_for(
      "solprop_native.transform.projection.custom",
      Range2(transform_space_for(node_count), {0, 0}, {problem_.n_columns(), n_to}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
        const std::size_t mode = column / n_pol_state;
        const std::size_t state_pol = column % n_pol_state;
        if (envelope) {
          Complex fine{};
          Complex rough{};
          for (std::size_t node = 0; node < node_count; ++node) {
            for (std::size_t transform_pol = 0; transform_pol < n_pol_transform;
                 ++transform_pol) {
              const std::size_t global_row =
                  (first_node + node) * n_pol_transform + transform_pol;
              const std::size_t local_row =
                  (node * n_pol_transform + transform_pol) * n_pol_state + state_pol;
              const Complex value = polar_complex(local_row, time);
              fine += projection(mode, global_row) * value;
              if (coarse)
                rough += projection_coarse(mode, global_row) * value;
            }
          }
          if (first_chunk) {
            projected_complex(column, time) = fine;
            if (coarse)
              projected_coarse_complex(column, time) = rough;
          } else {
            projected_complex(column, time) += fine;
            if (coarse)
              projected_coarse_complex(column, time) += rough;
          }
        } else {
          double fine = 0.0;
          double rough = 0.0;
          for (std::size_t node = 0; node < node_count; ++node) {
            for (std::size_t transform_pol = 0; transform_pol < n_pol_transform;
                 ++transform_pol) {
              const std::size_t global_row =
                  (first_node + node) * n_pol_transform + transform_pol;
              const std::size_t local_row =
                  (node * n_pol_transform + transform_pol) * n_pol_state + state_pol;
              const double value = polar_real(local_row, time);
              fine += projection(mode, global_row) * value;
              if (coarse)
                rough += projection_coarse(mode, global_row) * value;
            }
          }
          if (first_chunk) {
            projected_real(column, time) = fine;
            if (coarse)
              projected_coarse_real(column, time) = rough;
          } else {
            projected_real(column, time) += fine;
            if (coarse)
              projected_coarse_real(column, time) += rough;
          }
        }
      });
  Kokkos::Profiling::popRegion();
}

void GeneralWorkspace::form_absorber() {
  const bool spatial = !problem_.spatial.absorber_matrix.empty();
  const auto modes = problem_.dims.n_modes;
  const auto polarisations = problem_.dims.n_pol_state;
  // Flatten (polarisation,time) together: the same scalar radial matrix acts
  // independently on each component. The views alias existing FFT work buffers.
  if (spatial && problem_.dims.family == FieldFamily::envelope) {
    using Matrix = Kokkos::View<Complex**, Layout, MemorySpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const Matrix input(modal_complex_.data(), modes,
                       polarisations * problem_.dims.n_to);
    const Matrix output(absorber_complex_.data(), modes,
                        polarisations * problem_.dims.n_to);
    selected_gemm(absorber_execution_space_, spatial_absorber_complex_, input, output,
                  Complex{}, true);
  } else if (spatial && problem_.dims.family == FieldFamily::field_real) {
    using Matrix = Kokkos::View<double**, Layout, MemorySpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const Matrix input(modal_real_.data(), modes, polarisations * problem_.dims.n_to);
    const Matrix output(absorber_real_.data(), modes,
                        polarisations * problem_.dims.n_to);
    selected_gemm(absorber_execution_space_, spatial_absorber_real_, input, output, 0.0,
                  true);
  }
  const auto rate = absorber_fine_;
  const auto spatial_matrix = spatial_absorber_real_;
  const auto modal_real = modal_real_;
  const auto modal_complex = modal_complex_;
  const auto absorber_real = absorber_real_;
  const auto absorber_complex = absorber_complex_;
  const auto family = problem_.dims.family;
  launch::parallel_for(
      "solprop_native.absorber.nonlinear_term",
      Range2(temporal_execution_space_, {0, 0},
             {problem_.n_columns(), problem_.dims.n_to}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
        if (family == FieldFamily::envelope) {
          absorber_complex(column, time) =
              (spatial ? absorber_complex(column, time) : Complex{}) -
              0.5 * rate(time) * modal_complex(column, time);
        } else if (family == FieldFamily::field_real) {
          absorber_real(column, time) = (spatial ? absorber_real(column, time) : 0.0) -
                                        0.5 * rate(time) * modal_real(column, time);
        } else {
          double transverse = 0.0;
          if (spatial)
            for (std::size_t source = 0; source < modes; ++source)
              transverse +=
                  spatial_matrix(column / polarisations, source) *
                  modal_complex(source * polarisations + column % polarisations, time)
                      .real();
          absorber_real(column, time) =
              transverse - 0.5 * rate(time) * modal_complex(column, time).real();
        }
      });
}

void GeneralWorkspace::finish_stage(const std::size_t stage) {
  const auto terms = stage_terms(stage);
  const double solution_weight = problem_.stepper.solution_weights[stage];
  const double error_weight = problem_.stepper.error_weights[stage];
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  // The host's blocks of bins, as the stage's entry kernel takes them.
  if (terms.blocks_vectorise()) {
    const auto n = problem_.dims.n_omega;
    const std::size_t blocks = (n + stage_block - 1) / stage_block;
    launch::parallel_for(
        "solprop_native.stage.finish_accumulate",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0,
                                            problem_.n_columns() * blocks),
        KOKKOS_LAMBDA(const std::size_t item) {
          const std::size_t first = (item % blocks) * stage_block;
          terms.finish_block(stage, solution_weight, error_weight, item / blocks, first,
                             n - first < stage_block ? n - first : stage_block);
        });
    return;
  }
#endif
  launch::parallel_for(
      "solprop_native.stage.finish_accumulate",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        terms.finish(stage, solution_weight, error_weight, column, index);
      });
}

void GeneralWorkspace::set_flow_actions(const std::vector<Complex>& forward,
                                        const std::vector<Complex>& inverse,
                                        const std::vector<unsigned char>& identity) {
  require_open();
  if (!flow_)
    throw std::logic_error("this state carries no coupled flow");
  flow_->set_actions(forward, inverse, identity);
}

std::vector<Complex> GeneralWorkspace::adams_block_to_host() const {
  require_adams();
  if (adams_block_.extent(0) == 0)
    throw std::logic_error("the Adams history holds no coupled block");
  std::vector<Complex> values(adams_block_.size());
  const auto mirror =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, adams_block_);
  std::copy(mirror.data(), mirror.data() + mirror.size(), values.begin());
  return values;
}

std::vector<Complex> GeneralWorkspace::adams_exponent_to_host() const {
  require_adams();
  std::vector<Complex> values(adams_exponent_.size());
  const auto mirror =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, adams_exponent_);
  std::copy(mirror.data(), mirror.data() + mirror.size(), values.begin());
  return values;
}

std::vector<Complex> GeneralWorkspace::flow_actions_to_host(const bool inverse) const {
  require_open();
  if (!flow_)
    throw std::logic_error("this state carries no coupled flow");
  return flow_->actions_to_host(inverse);
}

std::vector<double>
GeneralWorkspace::form_flow_actions(const std::size_t section, const double dz,
                                    const std::vector<double>& fractions) {
  require_open();
  if (!flow_)
    throw std::logic_error("this state carries no coupled flow");
  return flow_->form_actions(section, dz, fractions);
}

std::vector<double> GeneralWorkspace::weighted_error(const std::vector<double>& scale) {
  require_open();
  if (!problem_.stepper.weighted_controller)
    throw std::logic_error("this state has no weighted controller workspace");
  require_extent(scale.size(), problem_.dims.n_omega, "weighted controller scale");
  copy_vector(execution_space_, weighted_scale_, scale);
  const auto residual = residual_;
  const auto in_norm = norm_mask_;
  const auto native_scale = weighted_scale_;
  const auto partials = weighted_partials_;
  const auto n = problem_.dims.n_omega;
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const int team_size = SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE;
#else
  // A host team per column, sharing the threads when the columns are fewer.
  const int team_size =
      host_reduction_team_size(execution_space_, problem_.n_columns(), n);
#endif
  // On a device each column is `reduction_slots_` teams over its own bins; the
  // caller sums the slots.
  const auto slots = reduction_slots_;
  const TeamPolicy policy(execution_space_,
                          static_cast<int>(problem_.n_columns() * slots), team_size);
  launch::parallel_for(
      "solprop_native.step.weighted_error_partials", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto league = static_cast<std::size_t>(team.league_rank());
        const auto column = league / slots;
        std::size_t first = 0;
        std::size_t end = 0;
        slot_range(n, slots, league % slots, first, end);
        double sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(first),
                                    static_cast<int>(end)),
            [&](const int index_value, double& update) {
              const auto index = static_cast<std::size_t>(index_value);
              if (in_norm(index) == 0U)
                return;
              const Complex value = residual(column, index) / native_scale(index);
              update += squared_abs(value);
            },
            sum);
        Kokkos::single(Kokkos::PerTeam(team), [&] { partials(league) = sum; });
      });
  execution_space_.fence("solprop_native weighted-error boundary");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, partials);
  std::vector<double> result(problem_.n_columns() * slots);
  for (std::size_t slot = 0; slot < result.size(); ++slot)
    result[slot] = host(slot);
  return result;
}

void GeneralWorkspace::scan_plasma_statistics(const std::size_t first_node,
                                              const std::size_t node_count,
                                              const RealView2& output,
                                              const bool diagnose,
                                              const double preionisation) {
  if (problem_.plasma.envelope) {
    if (diagnose)
      throw std::invalid_argument("the plasma integration diagnostics read the "
                                  "field-grid plasma column, and this problem "
                                  "carries none");
    for (std::size_t channel = 0; channel < problem_.plasma.n_channels; ++channel)
      envelope_statistics(channel, first_node, node_count, output, node_complex_,
                          problem_.effective_node_pol());
    return;
  }
  // solprop SEP 18: the band rule's statistics, and every problem's run diagnostics,
  // come from the band scan; the trapezoid's own scan below is unchanged.
  if (problem_.plasma.band || diagnose) {
    if (diagnose)
      team_band_statistics<true>(first_node, node_count, output, preionisation);
    else
      team_band_statistics<false>(first_node, node_count, output, preionisation);
    return;
  }
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const bool real_family = problem_.dims.family == FieldFamily::field_real;
  const auto coefficients = rate_coefficients_;
  const auto n_pol = problem_.effective_node_pol();
  const auto n_to = problem_.dims.n_to;
  const auto intervals = problem_.plasma.rate_intervals;
  const auto dt = problem_.grid.dt_fine;
  const auto log_field_min = problem_.plasma.log_field_min;
  const auto d_log_field = problem_.plasma.d_log_field;
  const auto field_min = problem_.plasma.field_min;
  const auto field_max = problem_.plasma.field_max;
  const auto rate_max = problem_.plasma.rate_max;
  if (problem_.plasma.n_channels > 1) {
    const auto parameters = channel_parameters_;
    const auto channel_coefficients = channel_coefficients_;
    const auto channels = problem_.plasma.n_channels;
    launch::parallel_for(
        "solprop_native.plasma.channel_statistics_scan",
        Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0,
                                            node_count),
        KOKKOS_LAMBDA(const std::size_t node) {
          for (std::size_t channel = 0; channel < channels; ++channel) {
            double exposure = 0.0;
            double previous_rate = 0.0;
            double largest_rate = 0.0;
            for (std::size_t time = 0; time < n_to; ++time) {
              const double first =
                  (real_family ? node_real(node * n_pol, time)
                               : node_complex(node * n_pol, time).real());
              const double second =
                  n_pol == 2
                      ? (real_family ? node_real(node * n_pol + 1, time)
                                     : node_complex(node * n_pol + 1, time).real())
                      : 0.0;
              const double magnitude =
                  n_pol == 2 ? Kokkos::hypot(first, second) : Kokkos::abs(first);
              const double rate = ionisation_rate(
                  magnitude, channel_coefficients, parameters(channel, 0),
                  parameters(channel, 1), parameters(channel, 2),
                  parameters(channel, 3), parameters(channel, 4),
                  static_cast<std::size_t>(parameters(channel, 5)),
                  static_cast<std::size_t>(parameters(channel, 6)));
              if (time > 0)
                exposure += trapezoid(rate, previous_rate, dt);
              previous_rate = rate;
              if (rate > largest_rate)
                largest_rate = rate;
            }
            output(first_node + node, 2 * channel) = exposure;
            output(first_node + node, 2 * channel + 1) = largest_rate;
          }
        });
    return;
  }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const auto rates = plasma_rate_;
  const auto exposures = plasma_exposure_;
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const TeamPolicy policy(response_space_for(node_count), static_cast<int>(node_count),
                          static_cast<int>(response_team_size()));
  launch::parallel_for(
      "solprop_native.plasma.device_statistics_scan", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto node = static_cast<std::size_t>(team.league_rank());
        const auto base = node * n_pol;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, static_cast<int>(n_to)),
            [&](const int time_value) {
              const auto time = static_cast<std::size_t>(time_value);
              const double first = (real_family ? node_real(base, time)
                                                : node_complex(base, time).real());
              const double second =
                  n_pol == 2 ? (real_family ? node_real(base + 1, time)
                                            : node_complex(base + 1, time).real())
                             : 0.0;
              const double magnitude =
                  n_pol == 2 ? Kokkos::hypot(first, second) : Kokkos::abs(first);
              rates(node, time) =
                  ionisation_rate(magnitude, coefficients, log_field_min, d_log_field,
                                  field_min, field_max, rate_max, intervals);
            });
        team.team_barrier();
        Kokkos::parallel_scan(
            Kokkos::TeamThreadRange(team, static_cast<int>(n_to)),
            [&](const int time_value, double& update, const bool final) {
              const auto time = static_cast<std::size_t>(time_value);
              if (time != 0)
                update += trapezoid(rates(node, time), rates(node, time - 1), dt);
              if (final)
                exposures(node, time) = update;
            });
        double largest_rate = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n_to)),
            [&](const int time_value, double& largest) {
              const auto time = static_cast<std::size_t>(time_value);
              if (rates(node, time) > largest)
                largest = rates(node, time);
            },
            Kokkos::Max<double>(largest_rate));
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          output(first_node + node, 0) = exposures(node, n_to - 1);
          output(first_node + node, 1) = largest_rate;
        });
      });
#else
  launch::parallel_for(
      "solprop_native.plasma.statistics_scan",
      Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0,
                                          node_count),
      KOKKOS_LAMBDA(const std::size_t node) {
        const auto base = node * n_pol;
        double exposure = 0.0;
        double previous_rate = 0.0;
        double largest_rate = 0.0;
        for (std::size_t time = 0; time < n_to; ++time) {
          const double first =
              (real_family ? node_real(base, time) : node_complex(base, time).real());
          const double second =
              n_pol == 2 ? (real_family ? node_real(base + 1, time)
                                        : node_complex(base + 1, time).real())
                         : 0.0;
          const double magnitude =
              n_pol == 2 ? Kokkos::hypot(first, second) : Kokkos::abs(first);
          const double rate =
              ionisation_rate(magnitude, coefficients, log_field_min, d_log_field,
                              field_min, field_max, rate_max, intervals);
          if (time > 0)
            exposure += trapezoid(rate, previous_rate, dt);
          previous_rate = rate;
          if (rate > largest_rate)
            largest_rate = rate;
        }
        output(first_node + node, 0) = exposure;
        output(first_node + node, 1) = largest_rate;
      });
#endif
}

void GeneralWorkspace::finish(const double dz) {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (problem_.stepper.low_storage) {
    // The march accepted the step in place (solprop SEP 24 section 6.2): the
    // boundary treatments act on the state itself, and the stash a response pass
    // left is the last evaluation's, at a predicted field, not this one's.
    if (problem_.absorber.split_step)
      split_step(dz, current_);
    else
      mask(current_);
    stash_is_current_ = false;
    execution_space_.fence("solprop_native accepted-step boundary");
    return;
  }
  if (problem_.absorber.split_step)
    split_step(dz, advanced_);
  else
    mask(advanced_);
  std::swap(current_, advanced_);
  if (!stash_current_.empty())
    graph::deep_copy(execution_space_, stash_current_, stash_candidate_);
  stash_is_current_ = !stash_current_.empty();
  execution_space_.fence("solprop_native accepted-step boundary");
}

void GeneralWorkspace::finish_quadrature() {
  const Kokkos::Profiling::ScopedRegion region("solprop_native.quadrature.finish");
  if (!coarse_pending_)
    return;
  if (problem_.dims.family == FieldFamily::envelope)
    graph::execute(*forward_coarse_complex_plan_, projected_coarse_complex_,
                   projected_coarse_spectrum_, KokkosFFT::Normalization::none);
  else if (forward_pack_coarse_plan_)
    graph::execute(*forward_pack_coarse_plan_, forward_pack_coarse_,
                   forward_pack_coarse_spectrum_, KokkosFFT::Normalization::none);
  else
    graph::execute(*forward_coarse_real_plan_, projected_coarse_real_,
                   projected_coarse_spectrum_, KokkosFFT::Normalization::none);
  if (held_plasma())
    held_transform(true, static_cast<bool>(forward_pack_coarse_plan_));
  coarse_pending_ = false;
}

void GeneralWorkspace::score_quadrature() {
  quadrature_error_ = Kokkos::Experimental::quiet_NaN_v<double>;
  quadrature_stale_ = false;
  if (quadrature_partials_.empty() ||
      (!problem_.kerr.nonlinear && !problem_.plasma.enabled && !dispersive_))
    return;
  finish_quadrature();
  // SEP 6's dispersive products join the embedded rule's spectrum once, when scored.
  if (dispersive_)
    dispersive_->finish_coarse();

  const auto fine = projected_spectrum_;
  const auto coarse = projected_coarse_spectrum_;
  const auto crop = crop_index_;
  const auto window = omega_window_;
  const auto prefactor_terms = prefactor_terms_;
  const auto prefactor_weights = stage_prefactor_;
  const auto partials = quadrature_partials_;
  const auto last = problem_.stepper.n_stages - 1;
  const auto n = problem_.dims.n_omega;
  const auto n_prefactor = problem_.linear.n_prefactor_terms;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const TeamPolicy policy(execution_space_, static_cast<int>(problem_.n_columns()),
                          SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
  launch::parallel_for(
      "solprop_native.transform.device_quadrature_partials", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto column = static_cast<std::size_t>(team.league_rank());
        double difference_sum = 0.0;
        double fine_sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n)),
            [&](const int index_value, double& update) {
              const auto index = static_cast<std::size_t>(index_value);
              Complex prefactor{};
              for (std::size_t term = 0; term < n_prefactor; ++term)
                prefactor += prefactor_weights(last * n_prefactor + term) *
                             prefactor_terms(term, column, index);
              const Complex weight = prefactor * window(index);
              const Complex fine_value = weight * fine(column, crop(index));
              const Complex coarse_value = weight * coarse(column, crop(index));
              update += squared_abs(coarse_value - fine_value);
            },
            difference_sum);
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n)),
            [&](const int index_value, double& update) {
              const auto index = static_cast<std::size_t>(index_value);
              Complex prefactor{};
              for (std::size_t term = 0; term < n_prefactor; ++term)
                prefactor += prefactor_weights(last * n_prefactor + term) *
                             prefactor_terms(term, column, index);
              const Complex weight = prefactor * window(index);
              update += squared_abs(weight * fine(column, crop(index)));
            },
            fine_sum);
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          partials(0, column) = difference_sum;
          partials(1, column) = fine_sum;
        });
      });
#else
  launch::parallel_for(
      "solprop_native.transform.quadrature_partials",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_columns()),
      KOKKOS_LAMBDA(const std::size_t column) {
        double difference_sum = 0.0;
        double fine_sum = 0.0;
        for (std::size_t index = 0; index < n; ++index) {
          Complex prefactor{};
          for (std::size_t term = 0; term < n_prefactor; ++term)
            prefactor += prefactor_weights(last * n_prefactor + term) *
                         prefactor_terms(term, column, index);
          const Complex weight = prefactor * window(index);
          const Complex fine_value = weight * fine(column, crop(index));
          const Complex coarse_value = weight * coarse(column, crop(index));
          fine_sum += squared_abs(fine_value);
          difference_sum += squared_abs(coarse_value - fine_value);
        }
        partials(0, column) = difference_sum;
        partials(1, column) = fine_sum;
      });
#endif
  // The columns' sums in their order, on the device: the last score and the peak
  // since the backend last read it.
  const auto scores = quadrature_scores_;
  const auto columns = problem_.n_columns();
  launch::parallel_for(
      "solprop_native.transform.quadrature_score",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
      KOKKOS_LAMBDA(const std::size_t) {
        double difference_sum = 0.0;
        double fine_sum = 0.0;
        for (std::size_t column = 0; column < columns; ++column) {
          difference_sum += partials(0, column);
          fine_sum += partials(1, column);
        }
        double value = Kokkos::Experimental::quiet_NaN_v<double>;
        if (fine_sum != 0.0)
          value = Kokkos::sqrt(difference_sum / fine_sum);
        scores(0) = value;
        if (fine_sum != 0.0 && !(scores(1) >= value))
          scores(1) = value;
      });
  quadrature_stale_ = true;
}

double GeneralWorkspace::quadrature_error() const {
  if (quadrature_stale_) {
    const auto last = Kokkos::subview(quadrature_scores_, 0);
    Kokkos::deep_copy(execution_space_, quadrature_error_, last);
    execution_space_.fence("solprop_native quadrature score");
    quadrature_stale_ = false;
  }
  return quadrature_error_;
}

double GeneralWorkspace::quadrature_peak() {
  if (quadrature_scores_.extent(0) == 0)
    return Kokkos::Experimental::quiet_NaN_v<double>;
  double peak = 0.0;
  const auto stored = Kokkos::subview(quadrature_scores_, 1);
  graph::deep_copy(execution_space_, peak, stored);
  execution_space_.fence("solprop_native quadrature peak");
  graph::deep_copy(execution_space_, stored, Kokkos::Experimental::quiet_NaN_v<double>);
  return peak;
}

void GeneralWorkspace::promote_fsal() {
  require_open();
  score_quadrature();
  if (!pending_fsal_)
    return;
  const auto stages = stages_;
  const auto whole = whole_propagator_;
  const auto recycled = recycled_;
  const auto last = problem_.stepper.n_stages - 1;
  launch::parallel_for(
      "solprop_native.stage.promote_fsal",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        recycled(column, index) = whole(column, index) * stages(last, column, index);
      });
  has_recycled_ = true;
  pending_fsal_ = false;
}

void GeneralWorkspace::clear_fsal() {
  has_recycled_ = false;
  pending_fsal_ = false;
  coarse_pending_ = false;
  if (!stash_current_.empty())
    graph::deep_copy(execution_space_, stash_current_, 0.0);
  stash_is_current_ = false;
}

void GeneralWorkspace::mask(const ComplexView2& values) {
  const auto active = active_mask_;
  launch::parallel_for(
      "solprop_native.mask.accepted",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        if (active(index) == 0U)
          values(column, index) = Complex{};
      });
}

namespace {
// The absorber split's factor `exp(-rate h / 2)` on `rows` rows of `time`. The factor
// depends on the sample alone: a device forms it once a step for every sample, and a
// host task forms it once for a tile of samples, through the vector-maths layer
// (solprop ADR-059), and applies it to a block of rows, the same expression to the bit
// on the libm route. A device reads each sample's factor from `factor`, formed once a
// step by `form_split_factor` with the same expression: one binary64 exponential per
// sample rather than per element, which on a device with a reduced binary64 rate is
// most of the pass.
template <class TimeView>
void split_apply(const char* label, const ExecutionSpace& execution,
                 const TimeView& time, const RealView& rate, const RealView& factors,
                 const double dz, const std::size_t rows, const std::size_t samples,
                 const double scale = 1.0) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  static_cast<void>(rate);
  static_cast<void>(dz);
  launch::parallel_for(
      label, Range2(execution, {0, 0}, {rows, samples}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
        time(row, index) *= scale * factors(index);
      });
#else
  static_cast<void>(factors);
  constexpr std::size_t tile = 256;
  constexpr std::size_t row_block = 64;
  const std::size_t tiles = (samples + tile - 1) / tile;
  const std::size_t blocks = (rows + row_block - 1) / row_block;
  launch::parallel_for(
      label, Kokkos::RangePolicy<ExecutionSpace>(execution, 0, tiles * blocks),
      KOKKOS_LAMBDA(const std::size_t task) {
        const std::size_t first = (task % tiles) * tile;
        const std::size_t count = samples - first < tile ? samples - first : tile;
        const std::size_t row_first = (task / tiles) * row_block;
        const std::size_t row_last =
            rows - row_first < row_block ? rows : row_first + row_block;
        double factor[tile];
        for (std::size_t sample = 0; sample < count; ++sample)
          factor[sample] = -0.5 * rate(first + sample) * dz;
        vmath::exp(factor, factor, count);
        for (std::size_t sample = 0; sample < count; ++sample)
          factor[sample] *= scale;
        for (std::size_t row = row_first; row < row_last; ++row)
          for (std::size_t sample = 0; sample < count; ++sample)
            time(row, first + sample) *= factor[sample];
      });
#endif
}
} // namespace

void GeneralWorkspace::split_columns(const double dz, std::size_t active_rows,
                                     const double parseval,
                                     const std::size_t first_column) {
  const bool tail = temporal_tail_rows_ != 0 && active_rows == temporal_tail_rows_;
  if (active_rows == 0)
    active_rows = coarse_spectrum_.extent(0);
  if (problem_.dims.family == FieldFamily::envelope) {
    if (tail)
      coarse_tail_inverse_complex_.execute();
    else
      graph::execute(*coarse_inverse_complex_plan_, coarse_spectrum_,
                     coarse_time_complex_, KokkosFFT::Normalization::none);
    if (monitoring_split())
      add_split_removal(dz, active_rows, parseval, first_column, coarse_time_complex_,
                        1.0);
    split_apply("solprop_native.absorber.split_apply_complex", execution_space_,
                coarse_time_complex_, absorber_coarse_, split_factor_, dz, active_rows,
                problem_.dims.n_t);
    if (tail)
      coarse_tail_forward_complex_.execute();
    else
      graph::execute(*coarse_forward_complex_plan_, coarse_time_complex_,
                     coarse_spectrum_, KokkosFFT::Normalization::none);
  } else {
    if (tail)
      coarse_tail_inverse_real_.execute();
    else
      graph::execute(*coarse_inverse_real_plan_, coarse_spectrum_, coarse_time_real_,
                     KokkosFFT::Normalization::none);
    if (monitoring_split())
      add_split_removal(dz, active_rows, parseval, first_column, {}, 1.0);
    split_apply("solprop_native.absorber.split_apply_real", execution_space_,
                coarse_time_real_, absorber_coarse_, split_factor_, dz, active_rows,
                problem_.dims.n_t);
    if (tail)
      coarse_tail_forward_real_.execute();
    else
      graph::execute(*coarse_forward_real_plan_, coarse_time_real_, coarse_spectrum_,
                     KokkosFFT::Normalization::none);
  }
}

// Free space's complex rows: the split's factor depends on time alone, so it acts alike
// on every column of any transverse basis, and the temporal pair runs in place on the
// state's own rows, a block of them at a time. Nothing is copied to a coarse buffer and
// back, the transform's normalisation goes with the factor, and the band's mask is the
// one pass after the blocks. Equal to the reference's split in real space up to the
// transforms' rounding, at any block size.
void GeneralWorkspace::split_state_rows(const double dz, const ComplexView2& state) {
  const double inverse_time = 1.0 / static_cast<double>(problem_.dims.n_t);
  const auto columns = problem_.n_columns();
  for (std::size_t first = 0; first < columns; first += split_rows_) {
    const auto rows = std::min(split_rows_, columns - first);
    const ComplexView2 block =
        Kokkos::subview(state, std::make_pair(first, first + rows), Kokkos::ALL());
    const bool tail = rows != split_rows_;
    graph::execute(tail ? *split_tail_inverse_plan_ : *split_inverse_plan_, block,
                   block, KokkosFFT::Normalization::none);
    if (monitoring_split())
      add_split_removal(dz, rows, 0.0, first, block, inverse_time * inverse_time);
    split_apply("solprop_native.absorber.split_apply_complex", execution_space_, block,
                absorber_coarse_, split_factor_, dz, rows, problem_.dims.n_t,
                inverse_time);
    graph::execute(tail ? *split_tail_forward_plan_ : *split_forward_plan_, block,
                   block, KokkosFFT::Normalization::none);
  }
  mask(state);
}

// What one split takes of the power it finds in the coarse time rows, `sum P (1 -
// exp(-alpha h))` with the boundary monitor's column weights, added to the run's
// accumulator in the memory space (solprop SEP 21, ADR-057): the monitor's reading
// drains it at a check, so every accepted step is counted, checked or not. The same
// pass adds to the step's `sum P'` and the guards' and the collar's sums of `P'`, the
// power found halfway through the split's factor, `P exp(-alpha h / 2)`, which the
// monitor judges the occupancies on: the symmetric splitting's state at the step's
// end. A separable route's rows are real-space rows, measured by `parseval`; zero
// means the rows are the stored columns from `first_column` on, weighted by their
// quadrature. `complex_time` holds a complex route's rows, whose power `power_scale`
// brings to the state's measure: the state's own rows carry no transform normalisation.
void GeneralWorkspace::add_split_removal(const double dz, const std::size_t active_rows,
                                         const double parseval,
                                         const std::size_t first_column,
                                         const ComplexView2& complex_time,
                                         const double power_scale) {
  const auto rate = absorber_coarse_;
  const auto share = monitor_share_;
  const auto half = monitor_half_;
  const auto weights = monitor_column_weights_;
  const bool stored = parseval == 0.0;
  // The share the split takes, and half its power factor, depend on time alone:
  // formed once, not per row.
  launch::parallel_for(
      "solprop_native.monitor.split_share",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.dims.n_t),
      KOKKOS_LAMBDA(const std::size_t index) {
        share(index) = -Kokkos::expm1(-rate(index) * dz);
        half(index) = Kokkos::exp(-0.5 * rate(index) * dz);
      });
  const auto guards = monitor_weights_;
  const auto samples =
      Range2(execution_space_, {0, 0}, {active_rows, problem_.dims.n_t});
  if (problem_.dims.family == FieldFamily::envelope) {
    launch::parallel_reduce(
        "solprop_native.monitor.split_removal_complex", samples,
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t index,
                      SplitSums& update) {
          const double power = (stored ? weights(first_column + row) : parseval) *
                               (power_scale * squared_abs(complex_time(row, index)));
          update.values[0] += power * share(index);
          const double found = half(index) * power;
          update.values[1] += found;
          update.values[2] += guards(0, index) * found;
          update.values[3] += guards(1, index) * found;
          update.values[4] += guards(2, index) * found;
        },
        Kokkos::Sum<SplitSums, MemorySpace>(monitor_split_sum_));
  } else {
    const auto time = coarse_time_real_;
    launch::parallel_reduce(
        "solprop_native.monitor.split_removal_real", samples,
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t index,
                      SplitSums& update) {
          const double value = time(row, index);
          const double power =
              (stored ? weights(first_column + row) : parseval) * value * value;
          update.values[0] += power * share(index);
          const double found = half(index) * power;
          update.values[1] += found;
          update.values[2] += guards(0, index) * found;
          update.values[3] += guards(1, index) * found;
          update.values[4] += guards(2, index) * found;
        },
        Kokkos::Sum<SplitSums, MemorySpace>(monitor_split_sum_));
  }
  const auto removed = monitor_removed_;
  const auto split = monitor_split_sum_;
  const auto found = monitor_found_;
  launch::parallel_for(
      "solprop_native.monitor.split_removal_add",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
      KOKKOS_LAMBDA(const std::size_t) {
        removed() += split().values[0];
        found() += split();
      });
}

// The split-step temporal absorber on a separable geometry: the rate is the same at
// every node, so it acts on the temporal rows (free space's nodes, the planar kind's
// y rows of slab amplitudes) between the two transverse transforms.
void GeneralWorkspace::separable_split_step(const double dz,
                                            const ComplexView2& advanced) {
  if (split_rows_ != 0) {
    split_state_rows(dz, advanced);
    return;
  }
  if (problem_.free_geometry()) {
    // The split's factor depends on time alone, so it acts alike on every column of
    // any transverse basis (solprop's reference measures its removal on the stored
    // columns for this reason): free space splits its stored transverse-spectrum
    // columns directly, chunk by chunk of the temporal rows, as a modal route splits
    // its modes, with no transverse transform pair. Equal to the reference's split in
    // real space up to the transforms' rounding.
    const auto coarse = coarse_spectrum_;
    const auto active = active_mask_;
    const auto n_pol = problem_.dims.n_pol_state;
    const auto n = problem_.dims.n_omega;
    const double inverse_time = 1.0 / static_cast<double>(problem_.dims.n_t);
    for (std::size_t unit = 0; unit < transverse_units();
         unit += transverse_units_per_chunk()) {
      const auto chunk = transverse_chunk(unit);
      const auto first = chunk.first_row * n_pol;
      const auto rows = chunk.rows * n_pol;
      launch::parallel_for(
          "solprop_native.free.split_columns_input",
          Range2(execution_space_, {0, 0}, {rows, n}),
          KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
            coarse(row, omega) = inverse_time * advanced(first + row, omega);
          });
      split_columns(dz, rows, 0.0, first);
      launch::parallel_for(
          "solprop_native.free.split_columns_output",
          Range2(execution_space_, {0, 0}, {rows, n}),
          KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
            advanced(first + row, omega) =
                active(omega) != 0U ? coarse(row, omega) : Complex{};
          });
    }
    return;
  }
  const auto scratch = transverse_;
  const auto coarse = coarse_spectrum_;
  const auto active = active_mask_;
  const auto n_pol = problem_.dims.n_pol_state;
  const auto layout = transverse_layout();
  const auto n = problem_.dims.n_omega;
  const double inverse_nodes = inverse_transverse_count();
  const double inverse_time = 1.0 / static_cast<double>(problem_.dims.n_t);
  launch::parallel_for(
      "solprop_native.free.split_input",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        const auto mode = row / n_pol;
        layout.cell(scratch, omega * n_pol + row % n_pol, mode) =
            inverse_nodes * advanced(row, omega);
      });
  transverse_inverse();
  for (std::size_t unit = 0; unit < transverse_units();
       unit += transverse_units_per_chunk()) {
    const auto chunk = transverse_chunk(unit);
    const auto first = chunk.first_row;
    const auto count = chunk.rows;
    graph::deep_copy(execution_space_, coarse, Complex{});
    launch::parallel_for(
        "solprop_native.free.split_temporal_input",
        Range2(execution_space_, {0, 0}, {count * n_pol, n}),
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
          const auto mode = first + row / n_pol;
          coarse(row, omega) =
              inverse_time * layout.cell(scratch, omega * n_pol + row % n_pol, mode);
        });
    // The rows are real-space rows here: by Parseval their power is the stored
    // columns' over the transverse count, which the monitor's measure is in.
    split_columns(dz, count * n_pol, 1.0 / inverse_nodes);
    launch::parallel_for(
        "solprop_native.free.split_temporal_output",
        Range2(execution_space_, {0, 0}, {count * n_pol, n}),
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
          const auto mode = first + row / n_pol;
          layout.cell(scratch, omega * n_pol + row % n_pol, mode) = coarse(row, omega);
        });
  }
  transverse_forward();
  launch::parallel_for(
      "solprop_native.free.split_output",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        const auto mode = row / n_pol;
        advanced(row, omega) =
            active(omega) != 0U
                ? layout.cell(scratch, omega * n_pol + row % n_pol, mode)
                : Complex{};
      });
}

void GeneralWorkspace::form_split_factor(const double dz) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const auto rate = absorber_coarse_;
  const auto factor = split_factor_;
  launch::parallel_for(
      "solprop_native.absorber.split_factor",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.dims.n_t),
      KOKKOS_LAMBDA(const std::size_t index) {
        factor(index) = Kokkos::exp(-0.5 * rate(index) * dz);
      });
#else
  static_cast<void>(dz);
#endif
}

void GeneralWorkspace::split_step(const double dz, const ComplexView2& advanced) {
  form_split_factor(dz);
  if (monitoring_split()) {
    // The chunks below add what this step's split finds; the last step's is dropped.
    const auto found = monitor_found_;
    launch::parallel_for(
        "solprop_native.monitor.split_found_clear",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) { found() = SplitSums{}; });
  }
  if (problem_.separable_geometry()) {
    separable_split_step(dz, advanced);
    return;
  }
  const auto coarse = coarse_spectrum_;
  const double inverse_n = 1.0 / static_cast<double>(problem_.dims.n_t);
  launch::parallel_for(
      "solprop_native.absorber.split_pad",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        coarse(column, index) = inverse_n * advanced(column, index);
      });
  split_columns(dz);
  const auto active = active_mask_;
  launch::parallel_for(
      "solprop_native.absorber.split_commit",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        advanced(column, index) =
            active(index) != 0U ? coarse(column, index) : Complex{};
      });
}

} // namespace solprop_native
