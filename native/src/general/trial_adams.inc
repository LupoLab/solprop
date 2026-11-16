// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The Adams march's engine passes (solprop SEP 16): the resident history (allocation,
// phase, start, scale, divisor, defect norms, fused trial and accept, hold and
// restore, row transfers, z-tables, observation), the exponential and elementwise
// trials with their accept and observation, then the Runge-Kutta dense output
// (solprop SEP 47), which shares the strict adams_phase_sum, and collect_trial and
// reuse_fsal, which follow it in the unit's order.
//
// Part of trial.cpp's translation unit (included there, in order; never
// compiled alone).

// --- solprop SEP 16 phase 3b: the Adams history resident in the engine ---------------
//
// solprop's HostAdamsMarch keeps the controller and hands every state-sized operation
// of its fused path to an engine (solprop ADR-043). These are that engine's passes: the
// history's rows, the trial's predictor, source, corrected field and defect norms, the
// adoption and transport of an accepted step and dense observation all stay in device
// memory; only the trial's scalars, its coefficient tables (by value, in AdamsWeights)
// and the norms cross. Each pass is the host engine's arithmetic (solprop's
// backends/numpy_ref/multistep.py, HostHistory and AdamsKernels), checked against it
// in parity.

void GeneralWorkspace::adams_allocate(const std::size_t slots, const bool compensated,
                                      const bool interpolate) {
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

// The Adams march's source at an anchor-frame field held in current_, into stages_(0):
// stage zero of a trial whose stage-zero scalars are the evaluation position's, so the
// shared stage pipeline, its responses and its in-stage absorber are the march's too.
void GeneralWorkspace::adams_evaluate_source(const TrialScalars& scalars) {
  if (source_fault_)
    std::exchange(source_fault_, {})();
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

// The same, with current_ restored from adams_saved_ afterwards, a failure's too: the
// accepted state stands where the source failed (solprop issue #129).
void GeneralWorkspace::adams_evaluate(const TrialScalars& scalars) {
  try {
    adams_evaluate_source(scalars);
  } catch (...) {
    graph::deep_copy(execution_space_, current_, adams_saved_);
    throw;
  }
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
  {
    const SwappedAside aside(current_, adams_saved_);
    adams_evaluate_source(scalars);
  }
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
  // A fact of the allocation, read before any field is loaded (the workspace's
  // diagnostics).
  if (adams_store_.extent(0) == 0)
    throw std::logic_error("the resident Adams history is not allocated");
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
  // A switch, not dispatch_width's template lambda: device-only code nvcc has not
  // been shown to compile with one (its build of this file once failed internally).
  switch (formed) {
  case 1:
    return adams_device_prediction<1>(weights, formed, x, propagate, accepted, dz,
                                      moments, store, column, index);
  case 2:
    return adams_device_prediction<2>(weights, formed, x, propagate, accepted, dz,
                                      moments, store, column, index);
  case 3:
    return adams_device_prediction<3>(weights, formed, x, propagate, accepted, dz,
                                      moments, store, column, index);
  case 4:
    return adams_device_prediction<4>(weights, formed, x, propagate, accepted, dz,
                                      moments, store, column, index);
  case 5:
    return adams_device_prediction<5>(weights, formed, x, propagate, accepted, dz,
                                      moments, store, column, index);
  case 6:
    return adams_device_prediction<6>(weights, formed, x, propagate, accepted, dz,
                                      moments, store, column, index);
  case 7:
    return adams_device_prediction<7>(weights, formed, x, propagate, accepted, dz,
                                      moments, store, column, index);
  case 8:
    return adams_device_prediction<8>(weights, formed, x, propagate, accepted, dz,
                                      moments, store, column, index);
  default:
    return adams_device_prediction<0>(weights, formed, x, propagate, accepted, dz,
                                      moments, store, column, index);
  }
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
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
constexpr std::size_t adams_block = 1;
#else
constexpr std::size_t adams_block = adams_lane_width;
#endif

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
      // A switch, as in `adams_device_predict`.
      switch (weights.width) {
      case 1:
        return device_element<1>(column, first, magnitude, update);
      case 2:
        return device_element<2>(column, first, magnitude, update);
      case 3:
        return device_element<3>(column, first, magnitude, update);
      case 4:
        return device_element<4>(column, first, magnitude, update);
      case 5:
        return device_element<5>(column, first, magnitude, update);
      case 6:
        return device_element<6>(column, first, magnitude, update);
      case 7:
        return device_element<7>(column, first, magnitude, update);
      case 8:
        return device_element<8>(column, first, magnitude, update);
      default:
        break;
      }
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
  {
    const SwappedAside aside(current_, adams_saved_);
    use_uploaded_sample(sample);
    // On a device the source's launches replay as one graph while they repeat.
    adams_source_graph_.run(execution_space_, [&] { adams_evaluate_source(sample); });
  }
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

// NOLINTNEXTLINE(readability-function-size): hot path, split pending an A/B
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
// `RungeKuttaDense.observe` form it: with `v0` the state at the step's anchor, `k_i`
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
