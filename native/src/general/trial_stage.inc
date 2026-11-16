// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The stage's orchestration: the fine gap runs, the stage terms and entry, the
// modal, host-source and general cores, the transform scaling and the synthesis.
//
// Part of trial.cpp's translation unit (included there, in order; never
// compiled alone).

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
  // A host source is the stage's whole source (solprop SEP 49 section 6.2).
  terms.hosted = static_cast<bool>(host_source_);
  terms.host_spectrum = host_spectrum_;
  terms.host_input = host_input_;
  if (terms.hosted)
    terms.zero_response = false;
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
  if (host_source_) {
    host_core();
    return true;
  }
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

void GeneralWorkspace::bind_host_source(std::shared_ptr<HostSource> source) {
  if (!source) {
    host_source_.reset();
    host_input_ = ComplexView2();
    host_spectrum_ = ComplexView2();
    return;
  }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  throw BackendRefusal("a host source runs on the native CPU build only");
#else
  if (general_ || problem_.separable_geometry() || problem_.n_columns() != 1 ||
      problem_.dims.family != FieldFamily::field_real)
    throw BackendRefusal("a host source takes one real-field column of a modal route");
  const auto n = problem_.dims.n_omega;
  host_input_ = ComplexView2("solprop_native.host_source.field", 1, n);
  host_spectrum_ = ComplexView2("solprop_native.host_source.spectrum", 1, n);
  host_source_ = std::move(source);
#endif
}

// solprop SEP 49: the stage core of a host source. The stage's entry kernel left the
// lab-frame field, unscaled and with any noise added, in the host source's field row
// (`StageTerms::combine`); the in-stage absorber is formed from the modal time field
// as on every route; and the host evaluates the source from the thread that called the
// march, outside every parallel region, into the row the finish reads. Both rows are
// host memory on the CPU builds that admit a host source. Host builds only.
void GeneralWorkspace::host_core() {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  throw BackendRefusal("a host source runs on the native CPU build only");
#else
  if (absorbing()) {
    graph::execute(*inverse_real_plan_, fine_spectrum_, modal_real_,
                   KokkosFFT::Normalization::none);
    form_absorber();
    graph::execute(*absorber_real_plan_, absorber_real_, absorber_spectrum_,
                   KokkosFFT::Normalization::none);
  }
  // The entry kernel's row and the absorber's work, on every instance, are done.
  Kokkos::fence("solprop_native.host_source.entry");
  const auto n = problem_.dims.n_omega;
  // Kokkos::complex<double> and std::complex<double> share one layout, two doubles.
  static_assert(sizeof(Complex) == sizeof(std::complex<double>));
  const std::span<const std::complex<double>> field(
      reinterpret_cast<const std::complex<double>*>(host_input_.data()), n);
  const std::span<std::complex<double>> source(
      reinterpret_cast<std::complex<double>*>(host_spectrum_.data()), n);
  host_source_->evaluate(field, host_z_, host_purpose_, source);
#endif
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
