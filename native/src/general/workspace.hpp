// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The general workspace (GeneralState's implementation), private to the units of
// src/general/: checks.cpp, setup.cpp, trial.cpp (the hot set, kept in one unit so
// that its inlining is the compiler's whole view), statistics.cpp and state.cpp.
#ifdef SOLPROP_NATIVE_STRICT_TU
#error "general/workspace.hpp contracts: a strict translation unit must not include it"
#endif

#include "kernels.hpp"

namespace solprop_native {

// Refuse a problem the workspace cannot run, before it allocates (checks.cpp).
void require_general_problem(const Problem& problem);

// What a Cartesian plane's widths read of its fluence (`plane_fluence_moments`).
inline constexpr std::size_t plane_moment_slots = 8;

class GeneralWorkspace {
public:
  explicit GeneralWorkspace(Problem problem);
  // Every view, plan and execution instance is a member, released by destruction once
  // the queued work has finished (state.cpp).
  ~GeneralWorkspace();
  GeneralWorkspace(const GeneralWorkspace&) = delete;
  GeneralWorkspace& operator=(const GeneralWorkspace&) = delete;
  GeneralWorkspace(GeneralWorkspace&&) = delete;
  GeneralWorkspace& operator=(GeneralWorkspace&&) = delete;

  void set_initial(std::span<const Complex> field, std::span<const Complex> noise);
  std::vector<double> trial_step(double dz, const TrialScalars& scalars);
  void prepare_trial(double dz, const TrialScalars& scalars);
  std::vector<Complex> adams_source(const std::vector<Complex>& input,
                                    const TrialScalars& scalars);
  void adopt(const std::vector<Complex>& field, double dz);
  double low_storage_step(double dz, const TrialScalars& scalars, bool rebuild_table);
  void low_storage_flows_formed(bool rebuild_table);
  void low_storage_separable_pass(std::size_t pass, double dz);
  void low_storage_separable_input(std::size_t pass);
  void low_storage_separable_fold(std::size_t pass, int mode, int term, double dz);
  void low_storage_modal_pass(std::size_t pass, double dz);
  void low_storage_modal_input(std::size_t pass);
  void adams_allocate(std::size_t slots, bool compensated, bool interpolate);
  void adams_compensation_start(std::size_t slot, const std::vector<double>& rate);
  void adams_start(std::size_t slot, const TrialScalars& scalars);
  double adams_magnitude();
  // The controller's norm scale on the device: rtol * w_i * |u| + atol over the
  // in-band magnitude, formed each trial without the magnitude crossing to the host.
  void adams_scale(double rtol, double atol, const std::vector<double>& weights);
  void adams_upload_divisor(const std::vector<double>& divisor);
  std::vector<double> adams_fused_trial(double dz, const AdamsWeights& weights,
                                        const std::vector<double>& divisor,
                                        const TrialScalars& scalars);
  void adams_fused_accept(double dz, const AdamsWeights& keep, std::size_t free,
                          const AdamsWeights& dense);
  void adams_hold();
  void adams_restore();
  void adams_observe(const AdamsWeights& weights, const TrialScalars& scalars);
  std::vector<double>
  adams_exponential_trial(double dz, const ExponentialWeights& weights,
                          const std::vector<double>& divisor, const TrialScalars& step,
                          const TrialScalars& sample, const std::vector<double>& rate);
  void adams_exponential_accept(double dz, std::size_t free, const AdamsWeights& dense);
  // The end of an accepted Adams step (a host fence; nothing on a device).
  void adams_accepted_boundary() const;
  void adams_exponential_observe(double xi, double dz,
                                 const ExponentialWeights& weights);
  // solprop SEP 47: a Runge-Kutta march's dense output in a native session.
  void rk_dense_allocate(const std::vector<double>& weights);
  void rk_dense_hold(double dz);
  void rk_dense_observe(double theta, const TrialScalars& carry);
  void rk_dense_restore();
  void rk_dense_release() noexcept;
  std::vector<std::string> adams_row_names() const;
  std::array<std::size_t, 3> adams_row_extents(const std::string& name) const;
  std::vector<Complex> adams_rows_to_host(const std::string& name) const;
  void adams_rows_to_host(const std::string& name, std::complex<double>* destination,
                          std::size_t size) const;
  void adams_rows_from_host(const std::string& name,
                            const std::vector<Complex>& values);
  std::vector<double> collect_trial();
  void table_upload(TableKind kind, std::vector<double> edges, double lower,
                    double upper, const std::complex<double>* coefficients,
                    std::size_t n_intervals, std::size_t n_terms, std::size_t n_values);
  void table_evaluate(const std::vector<std::size_t>& slots,
                      const std::vector<double>& positions);
  void table_rate(std::size_t slot, double z, bool before);
  std::vector<Complex> table_terms_to_host(TableKind kind, std::size_t slot) const;
  std::vector<Complex> phase_sum_to_host(const std::vector<double>& weights);
  std::vector<double> weighted_error(const std::vector<double>& scale);
  void set_flow_actions(const std::vector<Complex>& forward,
                        const std::vector<Complex>& inverse,
                        const std::vector<unsigned char>& identity);
  std::vector<double> form_flow_actions(std::size_t section, double dz,
                                        const std::vector<double>& fractions);
  std::vector<Complex> flow_actions_to_host(bool inverse) const;
  std::vector<Complex> adams_block_to_host() const;
  std::vector<Complex> adams_exponent_to_host() const;
  void prepare_raman(const std::vector<double>& rates,
                     const std::vector<double>& densities);
  std::vector<double> spatial_statistics(std::uint64_t flags, double intensity_factor,
                                         double z);
  std::vector<double> plane_fluence_trace(double center_x, double center_y, double dx,
                                          double dy, double extent, double spacing,
                                          std::size_t n_radii,
                                          const std::vector<double>& cosines,
                                          const std::vector<double>& sines);
  std::vector<Complex> axis_spectrum(const std::vector<double>& matrix);
  std::vector<double>
  axis_populations(const std::vector<double>& matrix,
                   const std::vector<double>& channel_preionisation);
  std::vector<double> aperture_statistics(std::uint64_t flags,
                                          const std::vector<double>& apertures,
                                          int exponent,
                                          const std::vector<double>& regions,
                                          double intensity_factor);
  std::vector<double> statistics(std::uint64_t flags,
                                 const std::vector<double>& windows,
                                 const std::vector<double>& on_axis, double density,
                                 double taper, double preionisation,
                                 double intensity_factor,
                                 const std::vector<double>& channel_preionisation,
                                 const std::vector<double>& power_weight);
  std::vector<double> whitened_statistics(const std::vector<Complex>& root,
                                          std::uint64_t flags,
                                          const std::vector<double>& windows,
                                          double intensity_factor, bool spectral);
  void finish(double dz);
  void promote_fsal();
  void clear_fsal();
  // The boundary monitor found the temporal collar quiet (solprop SEP 21).
  void set_absorber_quiet(const bool quiet) noexcept { absorber_quiet_ = quiet; }
  void bind_monitor(const std::vector<double>& weights,
                    const std::vector<double>& column_weights);
  void unbind_monitor() noexcept { monitor_bound_ = false; }
  void bind_occupancy(const std::vector<unsigned char>& buckets, std::size_t count);
  std::vector<double> monitor_reading();
  void bind_edge(const std::vector<double>& faces, const std::vector<double>& clip,
                 const std::vector<double>& totals,
                 const std::vector<double>& synthesis, std::size_t first_row);
  void bind_edge_lines(const EdgeLines& lines);
  EdgeNumbers edge_reading(bool faces, bool whole, bool totals, bool strip,
                           bool want_current);
  // What reading the edge holds in the state's memory space (solprop SEP 44).
  [[nodiscard]] std::size_t edge_bytes() const noexcept {
    std::size_t total =
        (edge_faces_.size() + edge_clip_.size() + edge_totals_.size() +
         edge_synthesis_.size() + edge_columns_.size() + edge_scale_.size()) *
        sizeof(double);
    for (const auto& line : edge_lines_)
      total += line.positions.size() * sizeof(std::size_t) +
               line.window.size() * sizeof(double);
    return total + edge_strip_.size() * sizeof(Complex);
  }
  // The bytes through this state's crossings since it was built, each way (solprop SEP
  // 44).
  [[nodiscard]] CrossedBytes crossed_bytes() const noexcept { return crossed_; }
  std::vector<Complex> field_to_host() const;
  void field_to_host(std::span<Complex> destination) const;
  const Complex* host_field() const {
    if (!initialized_)
      throw std::logic_error("no initial field has been loaded");
    if constexpr (!static_cast<bool>(
                      Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                                 MemorySpace>::accessible))
      return nullptr;
    execution_space_.fence("solprop_native host_field");
    return current_.data();
  }
  void count_to_host(const std::size_t bytes) const noexcept {
    crossed_.to_host += bytes;
  }
  void cross_to_host(const ComplexView2& source, std::span<Complex> destination) const;
  void set_crossing_workers(const std::size_t workers) noexcept {
    crossing_ring_.set_workers(workers);
  }
  std::vector<Complex> stage_to_host(std::size_t stage) const;
  void stage_to_host(std::size_t stage, std::span<Complex> destination) const;
  std::vector<Complex> projected_to_host(bool coarse);
  std::map<std::string, bool> dispersive_path() const;
  std::map<std::string, std::size_t> memory_ledger() const;
  /** The team a compact problem's FFTW loops take (`CpuTeamScope`): its pools'
      widest instance, which every threaded kernel and transform of it uses. Zero for
      any other problem: the runtime's team. */
  [[nodiscard]] int cpu_team() const noexcept {
    return compact_ ? std::max({kernel_pool_.widest(), fft_pool_.widest(),
                                raman_pool_.widest(), raman_tail_pool_.widest()})
                    : 0;
  }
  const CpuExecution& cpu_execution() const noexcept {
    return kernel_pool_.execution();
  }
  CpuSizing execution_sizing() const {
    auto result = kernel_pool_.sizing();
    result.merge(fft_pool_.sizing());
    result.merge(raman_pool_.sizing());
    result.merge(raman_tail_pool_.sizing());
    if (general_)
      result.merge(general_->execution_sizing());
    return result;
  }
  std::map<std::string, std::size_t> base_memory_ledger() const;
  double planning_seconds() const noexcept { return planning_seconds_; }
  double raman_planning_seconds() const noexcept { return raman_planning_seconds_; }
  std::string raman_fft_provider() const {
    return raman_drive_forward_plan_ ? raman_drive_forward_plan_->provider() : "none";
  }
  std::map<std::string, int> raman_fft_tasks() const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    const int full = 0;
    const int tail = 0;
    const int kernel = 0;
    const int products = 0;
#else
    const int full = raman_execution_space_.concurrency();
    const int tail = raman_tail_execution_space_.concurrency();
    const int kernel = raman_kernel_execution_space_.concurrency();
    const int products = raman_product_execution_space_.concurrency();
#endif
    // `products` is nonzero when the modal products replace the node drives.
    return {
        {"full",
         raman_drive_forward_plan_ ? raman_drive_forward_plan_->tasks(full) : 0},
        {"tail", raman_tail_forward_plan_ ? raman_tail_forward_plan_->tasks(tail) : 0},
        {"kernel",
         raman_kernel_forward_plan_ ? raman_kernel_forward_plan_->tasks(kernel) : 0},
        {"products", raman_product_forward_plan_
                         ? raman_product_forward_plan_->tasks(products)
                         : 0}};
  }
  std::size_t raman_kernel_transforms() const noexcept {
    return raman_kernel_transforms_;
  }
  // The last accepted step's score, read from the device when asked for.
  double quadrature_error() const;
  // The largest score since the last call, then reset (NaN when none was scored).
  double quadrature_peak();
  std::string synthesis_provider() const;
  std::string projection_provider() const;
  static std::string_view gemm_shape_policy();
  std::size_t response_team_size() const noexcept;
  bool fsal_ready() const noexcept { return has_recycled_; }
  std::size_t rk_source_evaluations() const noexcept { return rk_source_evaluations_; }
  std::size_t axis_scans() const noexcept { return axis_scans_; }

  // NVCC extended device lambdas require their enclosing member to be public.
  // GeneralWorkspace is private to src/general/, so this does not widen the API.
  std::size_t fine_spectrum_size() const {
    return problem_.dims.family == FieldFamily::field_real ? problem_.dims.n_to / 2 + 1
                                                           : problem_.dims.n_to;
  }
  std::size_t projected_spectrum_size() const {
    return problem_.dims.family == FieldFamily::envelope ? problem_.dims.n_to
                                                         : problem_.dims.n_to / 2 + 1;
  }
  void require_trial_scalars(const TrialScalars& scalars) const;
  void require_adams() const;
  void adams_phase();
  void adams_evaluate_source(const TrialScalars& scalars);
  void adams_evaluate(const TrialScalars& scalars);
  std::vector<double>
  adams_elementwise_trial(double dz, const ExponentialWeights& weights,
                          const std::vector<double>& divisor, const TrialScalars& step,
                          const TrialScalars& sample, const std::vector<double>& rate);
  void adams_rate(const std::vector<double>& rate, const ComplexView2& out);
  void adams_fallback_squarings(std::size_t section, double dz,
                                const ExponentialWeights& weights);
  // A trial's `rows` norms: its reduction over `policy` into the pinned buffer, then
  // the trial's one fence, on the reduction's own instance. A reduction into memory the
  // device writes returns before it completes.
  template <class Policy, class Reducer>
  std::vector<double> adams_norms(const char* label, const Policy& policy,
                                  const Reducer& reducer, const std::size_t rows) {
    const Kokkos::View<double*, Kokkos::SharedHostPinnedSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        result(adams_norms_.data(), rows);
    launch::parallel_reduce(label, policy, reducer, result);
    policy.space().fence("solprop_native Adams trial boundary");
    std::vector<double> norms(rows);
    for (std::size_t row = 0; row < rows; ++row)
      norms[row] = std::sqrt(result(row));
    adams_trial_ready_ = true;
    return norms;
  }
  void allocate();
  void upload();
  void upload_trial(const TrialScalars& scalars, const TrialScalars* sample = nullptr,
                    const std::vector<double>* rate = nullptr);
  void use_uploaded_sample(const TrialScalars& sample);
  void bind_stage_scalars(std::size_t slot);
  void note_trial_scalars(const TrialScalars& scalars);
  void plan();
  void reuse_fsal();
  void evaluate_stage(std::size_t stage, double dz);
  // `accumulate` false: an Adams trial reads the stage row alone.
  void evaluate_stage_source(std::size_t stage, double dz, bool accumulate = true);
  bool modal_core(std::size_t stage);
  void general_core(std::size_t stage);
  StageTerms stage_terms(std::size_t stage) const;
  std::vector<std::pair<std::size_t, std::size_t>> fine_gap_runs() const;
  void stage_entry(std::size_t stage, double dz);
  template <EntryPhase Mode> void stage_entry_with(std::size_t stage, double dz);
  void finish_or_defer(std::size_t stage);
  // Free space and the hybrid planar kind (`Problem::separable_geometry`).
  TransverseLayout transverse_layout() const noexcept {
    const std::size_t plane = transverse_.extent(1) * transverse_.extent(2);
    return problem_.slab_geometry() ? TransverseLayout{.inner = problem_.spatial.n_p,
                                                       .channel_fastest = true,
                                                       .plane = plane}
                                    : TransverseLayout{.inner = problem_.spatial.n_x,
                                                       .channel_fastest = false,
                                                       .plane = plane};
  }
  RegisterIndex register_index() const noexcept {
    constexpr bool plane = build::host;
    return {.layout = transverse_layout(),
            .n_pol = problem_.dims.n_pol_state,
            .n = problem_.dims.n_omega,
            .plane = plane,
            .scalar = problem_.dims.n_pol_state == 1};
  }
  // A stage's scratch positions: the volume route's own order on a device, or the
  // scratch's (`register_index`). Every other pass over the scratch forms its own input
  // in `(omega, y, x)`, since the scratch is dead between them.
  RegisterIndex stage_index() const noexcept {
    auto index = register_index();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    index.volume = volume_order_;
#endif
    return index;
  }
  // The unscaled transverse DFT's sample count: the plane's, or the slab's y axis.
  double inverse_transverse_count() const noexcept {
    return 1.0 / static_cast<double>(problem_.slab_geometry() ? problem_.spatial.n_y
                                                              : problem_.dims.n_modes);
  }
  std::size_t transverse_units() const noexcept {
    return problem_.slab_geometry() ? problem_.spatial.n_y : problem_.dims.n_nodes;
  }
  std::size_t transverse_units_per_chunk() const noexcept {
    return problem_.slab_geometry() ? problem_.dims.node_chunk / problem_.spatial.n_x
                                    : problem_.dims.node_chunk;
  }
  TransverseChunk transverse_chunk(std::size_t first_unit) const noexcept;
  void transverse_forward();
  void transverse_inverse();
  void transverse_inverse(const ComplexView3& scratch);
  void flux_at(double z);
  void require_flux_closure();
  void fill_statistics_closure();
  void evaluate_separable_stage(std::size_t stage, double dz, bool accumulate);
  void seed_free_wisdom();
  void seed_modal_wisdom();
  void seed_stream_wisdom();
  void separable_core(std::size_t stage, bool material);
  void separable_spatial_core();
  void separable_volume_core(std::size_t stage);
  void separable_stage_input(std::size_t stage, double dz);
  void separable_load_current();
  void separable_pack_chunk(const TransverseChunk& chunk, bool material);
  void separable_nodes(const TransverseChunk& chunk);
  void slab_project(const TransverseChunk& chunk, const EnvelopeStream* stream);
  void separable_finish_transform(std::size_t stage, bool material);
  void separable_time_statistics(std::size_t axis_peak, std::size_t peak, bool fluence,
                                 double intensity_factor);
  void separable_fluence(double intensity_factor);
  FarFieldPass::Inputs far_field_inputs() const;
  void plane_fluence_moments(double (&moments)[plane_moment_slots]);
  void scale_transform(std::size_t stage, double taper);
  void synthesise(std::size_t first_node, std::size_t node_count);
  void response(std::size_t stage, std::size_t first_node, std::size_t node_count);
  void prepare_raman_stage(std::size_t stage, const double* rates,
                           const double* densities = nullptr);
  void convolve_raman(std::size_t stage, std::size_t node_count);
  void finish_raman(std::size_t stage, std::size_t node_count);
  // A scalar state's Raman drive is a quadratic form of its modal fields, so the
  // causal convolution is applied once per unordered modal product and each node's
  // convolved drive is reconstructed from them (`raman_modal_products`).
  [[nodiscard]] std::size_t raman_modal_products() const noexcept;
  void convolve_raman_products();
  void finish_raman_products(std::size_t stage, std::size_t first_node,
                             std::size_t node_count);
  template <FieldFamily Family, std::size_t Components>
  void response_plasma(std::size_t stage, std::size_t first_node,
                       std::size_t node_count);
  template <FieldFamily Family, std::size_t Components, bool Plasma>
  void response_nonlinear(std::size_t stage, std::size_t first_node,
                          std::size_t node_count);
  template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear>
  void response_stash(std::size_t stage, std::size_t first_node,
                      std::size_t node_count);
  template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
            bool Stash>
  void response_harmonic(std::size_t stage, std::size_t first_node,
                         std::size_t node_count);
  template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
            bool Stash, bool Harmonic>
  void response_raman(std::size_t stage, std::size_t first_node,
                      std::size_t node_count);
  template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
            bool Stash, bool Harmonic, bool Raman, bool RamanThg, bool Multiple = false,
            bool Folded = false>
  void response_impl(std::size_t stage, std::size_t first_node, std::size_t node_count);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // The device's response (`device_response`) and its trapezoid plasma's two-level
  // scans.
  template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
            bool Stash, bool Harmonic, bool Raman, bool RamanThg>
  void device_response(std::size_t stage, std::size_t first_node,
                       std::size_t node_count);
  template <FieldFamily Family, std::size_t Components, bool Stash>
  void device_trapezoid_plasma(std::size_t stage, std::size_t first_node,
                               std::size_t node_count);
#endif
  // solprop SEP 18's band rule as a team column (include/solprop_native/plasma.hpp):
  // filter, then scan, for the population and free current, in whole-row passes.
  // It writes the current separately from the collared non-plasma response.
  template <FieldFamily Family, std::size_t Components, bool Multiple,
            bool Folded = false>
  void team_band_plasma_column(std::size_t stage, std::size_t first_node,
                               std::size_t node_count);
  // The same column in tiles (solprop SEP 25 section 6.7.6, the device form): each
  // running integral is a local kernel, one team per (row, tile) filtering and scanning
  // its tile from zero, and a finish kernel adding the earlier tiles' totals and
  // forming the pass's product.
  template <FieldFamily Family, std::size_t Components, bool Multiple,
            bool Folded = false>
  void tiled_band_plasma_column(std::size_t stage, std::size_t first_node,
                                std::size_t node_count);
  // The tiled column's teams: `rows` x tiles of them, as wide as the build's response
  // team on a device (or `band_tile_threads_` on the host) and the kernel allows, with
  // the tile's scratch when `scratch`.
  template <class Functor>
  [[nodiscard]] Kokkos::TeamPolicy<ExecutionSpace>
  tiled_band_policy(std::size_t rows, bool scratch, const Functor& functor) const;
  void tiled_band_local(const RealView2& source, std::size_t rows, const char* label);
  // The band rule's first stage alone at every node of a chunk: the floored final
  // exposure and largest rate per channel, and with `Diagnose` the six per-node parts
  // of the run diagnostics; team scans with team reductions.
  template <bool Diagnose>
  void team_band_statistics(std::size_t first_node, std::size_t node_count,
                            const RealView2& output, double preionisation);
  // The band column's teams: on a device one per node, as wide as the build's team and
  // the kernel's resources allow; on the host the threads shared between the nodes.
  // Each team walks the nodes `slot, slot + league, ...` with its slot's rows.
  template <class Functor>
  [[nodiscard]] Kokkos::TeamPolicy<ExecutionSpace>
  band_team_policy(std::size_t node_count, const Functor& functor) const;
  [[nodiscard]] std::size_t band_league(std::size_t node_count) const;
  [[nodiscard]] int band_host_team_size(std::size_t node_count) const;
  void project(std::size_t first_node, std::size_t node_count, bool first_chunk,
               const EnvelopeStream* stream = nullptr,
               const FieldCurrentStream* field_stream = nullptr);
  bool field_plasma() const noexcept {
    return problem_.plasma.enabled && !problem_.plasma.envelope;
  }
  bool held_plasma() const noexcept { return field_plasma() && !problem_.plasma.gated; }
  // solprop's private gated benchmark: the current's running integral, gated by the
  // time window, joins the local polarisation before it is projected.
  bool gated_plasma() const noexcept { return field_plasma() && problem_.plasma.gated; }
  bool held_radial() const noexcept {
    return held_plasma() && !problem_.separable_geometry() &&
           !problem_.spatial.spectral_weights.empty();
  }
  // Whether a device takes a modal stage's forward transforms -- the projection's
  // (fine and coarse), the in-stage absorber's and the held currents', each a real row
  // of n_to samples -- as one batched transform at the end of the stage
  // (`transform_forward_pack`). On an AMD Instinct MI325X each transform of a
  // single-mode run was four rocFFT kernels of a few microseconds, launched one by one;
  // on an NVIDIA A40 a transform of a four-mode capillary's rows took about 96
  // microseconds on 16 of its 84 multiprocessors, whatever the rows. Batched, the
  // stage's rows share them. A device never defers the coarse quadrature
  // (`deferred_quadrature_eligible`), so the coarse rows are projected at every stage.
  // A host packs them too, unless the problem is compact or SOLPROP_NATIVE_FORWARD_PACK
  // is 0 (solprop issue #584): one transform of every row, threaded by its shape, in
  // place of one a transform over only the modes' rows, which left most of the
  // threads idle: on 18 cores of an Intel Xeon Gold 6240 the pack ran rdw 0.90,
  // argon_blueshift 0.88 and ctc 0.92 of their windows. A host keeps the coarse rows
  // after the fine ones so that a deferred quadrature's stages transform the fine block
  // alone.
  bool packs_forward() const noexcept {
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
    // A compact problem keeps its separate transforms: its one row a transform runs
    // serially below the FFT start, and the pack's few rows on the compact team ran
    // averaged helium 1.35 times slower on 18 cores of an Intel Xeon Gold 6240 and
    // 1.11 on 16 of an AMD EPYC 7763 (solprop issue #584).
    if (!host_forward_pack() || compact_)
      return false;
#endif
    return problem_.dims.family == FieldFamily::field_real &&
           !problem_.separable_geometry() && !problem_.general.enabled &&
           !problem_.dispersive.enabled && !problem_.plasma.envelope &&
           !held_radial() &&
           (problem_.absorber.in_stage || held_plasma() ||
            !problem_.transform.projection_coarse_matrix.empty());
  }
  static bool host_forward_pack() noexcept {
    static const bool enabled = [] {
      const char* text = std::getenv("SOLPROP_NATIVE_FORWARD_PACK");
      return text == nullptr || std::string_view(text) != "0";
    }();
    return enabled;
  }
  void transform_forward_pack();
  void held_response(std::size_t stage, std::size_t first_node, std::size_t node_count,
                     const TransverseChunk* chunk = nullptr);
  void gated_response(std::size_t node_count);
  void held_transform(bool coarse_only = false, bool transformed = false);
  void finish_quadrature();
  bool deferred_quadrature_eligible() const noexcept {
    if constexpr (build::device)
      return false;
    // The validated seven-stage FSAL tableau identifies the ordinary DOPRI5
    // route without another seam field. Other consumers retain eager spectra.
    return problem_.stepper.first_same_as_last && problem_.stepper.n_stages == 7 &&
           !problem_.stepper.low_storage && !problem_.separable_geometry() &&
           !general_ && !flow_ && !dispersive_ && !problem_.plasma.envelope &&
           !held_radial() && !problem_.transform.projection_coarse_matrix.empty();
  }
  void held_radial_projection(std::size_t first_node, std::size_t node_count);
  void envelope_cycle(std::size_t channel, std::size_t node_count,
                      const ComplexView2& field_rows, std::size_t n_pol);
  void envelope_statistics(std::size_t channel, std::size_t first_node,
                           std::size_t node_count, const RealView2& statistics,
                           const ComplexView2& field, std::size_t n_pol);
  void envelope_population(std::size_t channel, std::size_t stage,
                           std::size_t node_count);
  void envelope_response(std::size_t stage, std::size_t first_node,
                         std::size_t node_count,
                         const TransverseChunk* chunk = nullptr);
  void envelope_transform();
  void finish_stage(std::size_t stage);
  void score_quadrature();
  void fill_statistics_time(bool windowed, double lower, double upper,
                            bool weighted = false);
  void fill_statistics_pair(bool windowed, double lower, double upper);
  void form_statistics_power();
  void form_statistics_total_power();
  void statistics_modal_time();
  void scan_plasma_statistics(std::size_t first_node, std::size_t node_count,
                              const RealView2& output, bool diagnose = false,
                              double preionisation = 0.0);
  void form_absorber();
  void split_step(double dz, const ComplexView2& advanced);
  void form_split_factor(double dz);
  void split_columns(double dz, std::size_t active_rows = 0, double parseval = 0.0,
                     std::size_t first_column = 0);
  void split_state_rows(double dz, const ComplexView2& state);
  void add_split_removal(double dz, std::size_t active_rows, double parseval,
                         std::size_t first_column, const ComplexView2& complex_time,
                         double power_scale);
  void separable_to_time(const TransverseChunk& chunk);
  void separable_to_frequency(const TransverseChunk& chunk);
  void separable_split_step(double dz, const ComplexView2& advanced);
  void mask(const ComplexView2& values);
  bool use_packed_radial_blas() const noexcept;
  // A square radial analytic map synthesises the retained spectrum, before padding,
  // and transforms each node (`retained_synthesis`).
  bool retained_synthesis() const noexcept;
  void synthesise_retained(std::size_t first_node, std::size_t node_count);
  void packed_radial_gemm(const RealView2& matrix, const ComplexView2& source,
                          const ComplexView2& destination);
  bool use_blas_for_synthesis() const noexcept;
  bool use_interleaved_synthesis() const noexcept;
  bool use_interleaved_projection() const noexcept;
  bool hybrid_shape_eligible() const noexcept;
  bool use_blas_for_projection() const noexcept;
  bool use_chunked_blas_for_synthesis() const noexcept;
  bool folds_single_node() const noexcept;
  bool use_chunked_blas_for_projection() const noexcept;
  [[nodiscard]] const ExecutionSpace& fft_space_for(std::size_t samples,
                                                    std::size_t length);
  /** A batch's FFT instance, sized by its rows: extent 0 counts the rows, the
      remaining extents make one row (a transverse plane counts as one row). */
  template <class View>
  [[nodiscard]] const ExecutionSpace& fft_space_for_rows(const View& rows) {
    return fft_space_for(rows.size(), row_length(rows));
  }

  Problem problem_;
  // A compact problem's pools (`compact_problem`) and the team its FFTW loops take.
  bool compact_{};
  // One real mode at several nodes in one chunk: synthesis and projection through
  // the one-mode kernels on the response team (gemm.hpp).
  bool one_mode_products_{};
  CpuExecutionPool kernel_pool_{CpuPool::kernels, problem_.cpu_execution};
  CpuExecutionPool fft_pool_{CpuPool::fft, problem_.cpu_execution};
  CpuExecutionPool raman_pool_{CpuPool::raman_fft, problem_.cpu_execution};
  CpuExecutionPool raman_tail_pool_{CpuPool::raman_tail_fft, problem_.cpu_execution};
  mutable ExecutionSpace execution_space_;
  ExecutionSpace response_execution_space_, response_tail_execution_space_;
  ExecutionSpace transform_execution_space_, transform_tail_execution_space_;
  ExecutionSpace matrix_copy_space_, tail_matrix_copy_space_;
  ExecutionSpace stacked_matrix_copy_space_, tail_stacked_matrix_copy_space_;
  ExecutionSpace node_spectral_space_;
  const ExecutionSpace& matrix_copy_space_for(std::size_t nodes,
                                              bool stacked = false) const {
    if (stacked)
      return nodes == problem_.dims.node_chunk ? stacked_matrix_copy_space_
                                               : tail_stacked_matrix_copy_space_;
    return nodes == problem_.dims.node_chunk ? matrix_copy_space_
                                             : tail_matrix_copy_space_;
  }
  ExecutionSpace fft_execution_space_;
  ExecutionSpace temporal_execution_space_, flow_execution_space_,
      absorber_execution_space_;
  // The exponential Adams trial's passes and its accepted step's rows, sized by
  // adams_allocate when the history's rows are known.
  ExecutionSpace adams_execution_space_, adams_copy_space_;
  ExecutionSpace raman_multiply_space_, raman_tail_multiply_space_,
      raman_rebuild_space_, raman_scale_space_;
  ExecutionSpace raman_execution_space_, raman_tail_execution_space_,
      raman_kernel_execution_space_;
  ExecutionSpace raman_product_execution_space_, raman_product_space_;
  const ExecutionSpace& response_space_for(std::size_t nodes) const {
    return nodes == problem_.dims.node_chunk ? response_execution_space_
                                             : response_tail_execution_space_;
  }
  const ExecutionSpace& transform_space_for(std::size_t nodes) const {
    return nodes == problem_.dims.node_chunk ? transform_execution_space_
                                             : transform_tail_execution_space_;
  }
  bool initialized_{};
  // A constant operator's low-storage table is built (set on a rebuild, cleared by
  // `set_initial` and `close`): a step that reuses a table never built refuses.
  bool table_ready_{};
  bool stages_ready_{};
  bool has_noise_{};
  bool has_recycled_{};
  std::size_t rk_source_evaluations_{};
  // On-axis row scans (`axis_populations`), for the session's record (SEP 19).
  std::size_t axis_scans_{};
  bool pending_fsal_{};
  // Only the final trial source supplies the accepted quadrature score. Its
  // coarse time buffers survive until scoring, or until a new source replaces
  // them. The stored accepted score and peak are independent of these flags.
  bool defer_quadrature_{};
  bool project_coarse_{true};
  bool coarse_pending_{};
  // Set by a trial for a stage whose finish may run in the next stage's entry kernel;
  // `finish_pending_` names the stage's finish still to run there.
  bool defer_finish_{};
  bool finish_pending_{};
  // `problem_.grid.crop_index` is the identity (`StageTerms::identity_crop`).
  bool identity_crop_{};
  // The bound trial's tapers on the host, and the one the scaled transforms hold
  // (`scale_transform`).
  std::vector<double> trial_taper_;
  double scaled_taper_{};
  bool scaled_taper_ready_{};
  std::size_t pending_stage_{};
  bool stash_is_current_{};
  std::size_t plan_heap_growth_{}, plan_device_growth_{};
  double planning_seconds_{};
  double raman_planning_seconds_{};
  // Every fixed Phi term is purely imaginary: the stage phases have no real part.
  bool lossless_phase_ = false;
  // Per stage of the current trial, whether its phase weights equal the stage
  // before's, and whether the whole step's equal the last stage's (`upload_trial`);
  // the last stage whose entry formed the pair in this trial.
  std::vector<bool> phase_repeats_;
  bool whole_repeats_ = false;
  std::size_t entered_stage_ = std::numeric_limits<std::size_t>::max();
  std::size_t raman_kernel_transforms_{};
  bool raman_kernel_ready_{};
  std::vector<double> cached_raman_rates_, cached_raman_densities_;
  mutable double quadrature_error_{Kokkos::Experimental::quiet_NaN_v<double>};
  bool adams_device_scale_ = false;
  double adams_rtol_ = 0.0, adams_atol_ = 0.0;
  RealView adams_scale_weights_;
  Kokkos::View<double, Layout, MemorySpace> adams_magnitude_sum_;
  // The device's (last, peak) scores: an accepted step scores the quadrature on the
  // device and nothing crosses until the score is read.
  RealView quadrature_scores_;
  mutable bool quadrature_stale_ = false;
  SpatialCoefficients spatial_;
  // solprop SEP 34: the plane measure's coefficients. A separable run's are its own
  // operator's views (aliased, nothing more held); a radial run's are the entrance
  // squares, the material square replaced by the material z table's value at the
  // statistic's plane when the medium varies (`flux_table_`).
  SpatialCoefficients flux_;
  z_table::Table flux_table_;
  // The magnetic closure's transverse volume on a separable run, beside
  // `transverse_`, allocated when a local flux statistic is asked for.
  ComplexView3 flux_transverse_;
  ComplexView3 transverse_;
  std::unique_ptr<TransversePlan> transverse_forward_, transverse_inverse_;
  // On a device, a free-space envelope whose material response is the Kerr term alone,
  // on a time grid the spectrum fills (n_t = n_to = n_omega, the crop the identity),
  // with one polarisation: the stage core transforms the whole scratch volume in
  // (omega, y, x) at once and forms the response there, in place of the chunks'
  // transposes and temporal transforms (`separable_volume_core`).
  bool volume_kerr_ = false;
  // The volume route's stages hold the scratch as `(y, x, omega)`, `volume_scratch_`,
  // on a device (solprop #365): its rank-3 transform then keeps frequency contiguous
  // (cuFFT's `{N,N,T}`, rocFFT's `{T,N,N}`), and its input and fold no longer cross
  // layouts.
  bool volume_order_ = false;
  ComplexView3 volume_scratch_;
  std::unique_ptr<VolumePlan> volume_forward_, volume_inverse_;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  ComplexView transform_work_;
#endif
  std::unique_ptr<SlabPlan> slab_forward_, slab_inverse_;
  ComplexView2 current_, advanced_, solution_, error_, residual_, recycled_, noise_;
  // solprop SEP 24 phase 3: the low-storage march's registers B and C (the state is
  // A) and its flows.
  ComplexView2 register_accumulator_, register_stage_;
  ComplexView3 low_storage_flows_;
  ComplexView2 whole_propagator_, fine_spectrum_, projected_spectrum_;
  // The stage's `sum_t dPhi_t Phi_t`, formed by the combination for a coupled flow's
  // residual, and its inverse propagator `exp(-sum)`, which the finish applies.
  ComplexView2 stage_phase_, stage_inverse_;
  ComplexView2 projected_coarse_spectrum_, absorber_spectrum_, modal_complex_;
  ComplexView2 statistics_spectrum_, statistics_time_, statistics_weighted_time_;
  RealView2 statistics_power_weight_;
  bool statistics_unit_weight_ = true;
  ComplexView2 node_complex_, polarisation_complex_, projected_complex_;
  ComplexView2 projected_coarse_complex_, absorber_complex_, coarse_spectrum_;
  ComplexView2 coarse_time_complex_, scaled_synthesis_complex_;
  ComplexView2 scaled_projection_complex_, scaled_projection_coarse_complex_;
  ComplexView2 scaled_projection_stacked_complex_, projected_stacked_complex_;
  ComplexView2 chunk_synthesis_complex_, tail_synthesis_complex_;
  ComplexView2 chunk_projection_complex_, tail_projection_complex_;
  ComplexView2 chunk_projection_stacked_complex_, tail_projection_stacked_complex_;
  ComplexView2 node_raman_spectrum_, raman_kernel_spectrum_;
  ComplexView2 raman_tail_spectrum_;
  ComplexView2 spatial_absorber_complex_;
  ComplexView3 stages_, phi_terms_, prefactor_terms_;
  // solprop SEP 29 G1: the general z tables' coefficients, uploaded once; their values
  // at a trial's positions are written into the term slots above.
  z_table::Table phi_table_, prefactor_table_, rate_table_;
  RealView2 radial_blas_input_, radial_blas_output_;
  // The retained synthesis's coarse modal and node spectra.
  ComplexView2 retained_modal_, retained_node_;
  bool retained_synthesis_ = false;
  RealView2 modal_real_, node_real_, polarisation_real_, projected_real_;
  RealView2 projected_coarse_real_, projected_stacked_real_, absorber_real_;
  RealView2 coarse_time_real_;
  RealView2 node_raman_drive_, node_raman_convolved_, raman_undamped_;
  RealView2 raman_tail_drive_, raman_tail_convolved_;
  // The modal products' rows, the leading rows of the node drive and its spectrum:
  // the drive rows hold each product, then its convolution.
  RealView2 raman_product_drive_;
  ComplexView2 raman_product_spectrum_;
  std::size_t raman_products_ = 0;
  RealView2 raman_kernel_time_, stage_raman_rates_, stage_raman_densities_;
  RealView2 raman_tensor_weights_;
  RealView raman_row_component_, raman_component_fraction_;
  RealView2 synthesis_, projection_, projection_coarse_;
  RealView2 spatial_absorber_real_;
  RealView2 scaled_synthesis_, scaled_projection_, scaled_projection_coarse_;
  RealView2 scaled_projection_stacked_;
  RealView2 chunk_synthesis_, tail_synthesis_;
  RealView2 chunk_projection_, tail_projection_;
  RealView2 chunk_projection_stacked_, tail_projection_stacked_;
  RealView2 partials_, quadrature_partials_, stash_candidate_, stash_current_;
  RealView2 statistics_power_, plasma_statistics_scratch_;
  // solprop SEP 34: each node's `|E| |G|` beside its signed flux, which bounds a radial
  // fluence's rounding; allocated with the closure's buffers.
  RealView2 statistics_magnitude_;
  // `spatial_statistics_partials_` holds a plane's scaled fluence for its trace.
  bool plane_fluence_held_ = false;
  RealView2 plasma_rate_, plasma_exposure_, plasma_fraction_;
  RealView2 plasma_current_integrand_, plasma_free_current_;
  // The device trapezoid plasma's tile totals, by component row, and its tile count
  // (`device_trapezoid_plasma`).
  RealView2 plasma_tile_totals_;
  std::size_t plasma_tiles_{};
  RealView2 plasma_outer_current_;
  RealView weighted_scale_, weighted_partials_;
  RealView spatial_spectral_weights_;
  ComplexView spatial_on_axis_;
  RealView2 spatial_statistics_partials_;
  RealView statistics_total_power_;
  RealView statistics_on_axis_, statistics_packed_;
  // An envelope's on-axis row on a one-component node scan, its x and y components
  // (`axis_populations`), allocated by the first statistics call that needs it.
  ComplexView2 statistics_axis_row_;
  // solprop SEP 47: a general modal basis's whitened columns, the metric's square
  // root at one position and their sums, allocated by the first statistics call that
  // needs them (outside any trial step).
  ComplexView2 statistics_whitened_;
  ComplexView statistics_root_;
  RealView statistics_whitened_sums_;
  RealView tableau_, omega_, omega_window_, time_window_, absorber_fine_,
      absorber_coarse_;
  RealView stage_phi_delta_, stage_phi_at_, stage_prefactor_;
  RealView stage_density_, stage_taper_, stage_preionisation_, rate_coefficients_;
  RealView stage_kerr_density_, stage_plasma_density_, stage_raman_density_;
  // Every per-stage scalar view above, a Raman run's stage rates and amplitude
  // densities and a compensated Adams trial's rate weights live in one buffer of two
  // slots, uploaded in one copy a trial from the next of a ring of pinned host buffers
  // (`upload_trial`, solprop SEP 39 section 6.7): an Adams trial's step scalars in
  // slot 0 and its sample's in slot 1. The views point into the slot in use.
  RealView stage_scalars_;
  using StagingView = decltype(Kokkos::create_mirror_view(
      Kokkos::SharedHostPinnedSpace{}, std::declval<RealView>()));
  std::vector<StagingView> stage_staging_;
  std::size_t stage_staging_next_ = 0;
  std::size_t stage_slot_size_ = 0;
  std::array<std::size_t, 15> stage_offsets_{};
  RealView stage_rate_weights_;
  RealView2 channel_parameters_, stage_channel_density_, stage_channel_preionisation_;
  RealView channel_coefficients_;
  Kokkos::View<double***, Layout, MemorySpace> channel_scratch_;
  // solprop SEP 18 on a CPU build: the running-integral rule's taps, one row of ring
  // scratch per node of a chunk (the band column's, or the statistics scan's alone
  // under the trapezoid) and the run diagnostics' six parts per node.
  RealView plasma_taps_;
  RealView2 plasma_diagnostics_;
  // The band column's scratch rows per kind: one per concurrently running team.
  std::size_t band_slots_{};
  // Whether the band rule runs the tiled column (`tiled_band_plasma_column`), its
  // tile's samples, the tiles of a row, a host tile team's threads, and the tiles'
  // totals by row.
  bool band_tiled_{band_tiled(thread_policy())};
  // Each column's norm reductions on a device are this many teams (`reduction_slots`):
  // the norm partials hold `columns * reduction_slots_` slots, reduced on the host.
  std::size_t reduction_slots_{
      reduction_slots(problem_.n_columns(), problem_.dims.n_omega)};
  std::size_t band_tile_{};
  std::size_t band_tile_count_{};
  int band_tile_threads_{band_tile_threads(thread_policy())};
  RealView2 band_tile_totals_;
  ComplexView carrier_;
  std::array<EnvelopeStream, 2> envelope_streams_;
  std::array<FieldCurrentStream, 3> held_streams_;
  // A device's two held currents, fine and coarse rows of each, in one buffer and its
  // spectrum, which one plan transforms (`held_transform`); empty on a host and on the
  // radial route.
  RealView2 held_pack_;
  ComplexView2 held_pack_spectrum_;
  std::unique_ptr<ForwardRealPlan> held_pack_forward_;
  // A device's forward rows of a stage, one buffer and its spectrum (`packs_forward`):
  // the in-stage absorber's rows, then the projection's, then the held pack's, and the
  // plans of all of them and of the rows after the absorber's (a quiet collar's stage).
  RealView2 forward_pack_, forward_pack_tail_;
  ComplexView2 forward_pack_spectrum_, forward_pack_tail_spectrum_;
  std::unique_ptr<ForwardRealPlan> forward_pack_plan_, forward_pack_tail_plan_;
  std::size_t forward_pack_absorber_rows_ = 0;
  // A host's deferred quadrature (`deferred_quadrature_eligible`): the pack's fine
  // block (the absorber's, the projection's and the held currents' fine rows) with and
  // without the absorber's, for the stages, and its coarse block for the finish.
  RealView2 forward_pack_fine_, forward_pack_fine_tail_, forward_pack_coarse_;
  ComplexView2 forward_pack_fine_spectrum_, forward_pack_fine_tail_spectrum_,
      forward_pack_coarse_spectrum_;
  std::unique_ptr<ForwardRealPlan> forward_pack_fine_plan_,
      forward_pack_fine_tail_plan_, forward_pack_coarse_plan_;
  RealView2 held_current_, held_projection_;
  // The gated benchmark's running integral of each current row, and the band rule's
  // filtered rows (`gated_response`).
  RealView2 gated_prefix_, gated_filtered_;
  ComplexView2 held_nodal_spectrum_, held_projected_spectrum_;
  RealView held_derivative_, held_inverse_;
  RealView3 envelope_rule_nodes_, envelope_rule_weights_, envelope_moments_;
  RealView2 envelope_rates_, envelope_exposure_;
  IndexView envelope_failures_, envelope_status_;
  Kokkos::View<std::size_t*, Layout, Kokkos::HostSpace> envelope_host_status_;
  RealView envelope_drude_factor_;
  ComplexView envelope_loss_factor_;
  ByteView active_mask_, norm_mask_;
  // solprop SEP 16 phase 3b: the resident Adams history and its trial's scratch.
  ComplexView3 adams_store_, adams_slopes_, adams_moments_;
  ComplexView2 adams_source_, adams_propagator_, adams_exponent_, adams_saved_;
  ComplexView2 adams_held_, adams_dense_start_, adams_dense_end_, adams_dense_exponent_;
  Kokkos::View<double*, Layout, MemorySpace> adams_divisor_;
  // A Runge-Kutta trial's norm partials cross through this mirror, pinned on a device
  // (the view itself on a host build), so that no trial takes a fresh host buffer. The
  // copy is asynchronous: it is read after the trial's fence.
  decltype(Kokkos::create_mirror_view(Kokkos::SharedHostPinnedSpace{},
                                      std::declval<RealView2>())) pinned_partials_;
  // A trial's squared norms, which its reduction writes and the host reads: in pinned
  // host memory on a device, which the reduction's last block writes directly, so that
  // no copy from the device follows it before the trial's one fence.
  Kokkos::View<double*, Kokkos::SharedHostPinnedSpace> adams_norms_;
  // An elementwise trial's source evaluation, replayed as a device graph.
  graph::Region adams_source_graph_;
  // A varying operator's compensation: each sample's state and rate, the trial's
  // carried rows, its predicted field, endpoint term and rate, and dense observation's
  // terms.
  ComplexView3 adams_states_, adams_rates_, adams_carried_, adams_dense_terms_;
  ComplexView2 adams_lab_, adams_endpoint_, adams_endpoint_rate_;
  Kokkos::View<double*, Layout, MemorySpace> adams_rate_weights_;
  // A coupled flow's exponential trial (solprop SEP 16 phase 6): the step's flow and
  // moments on the block's bins, `(count + 1, n_active, n_modes, n_modes)` with the
  // flow first; a frozen-residual flow's weights at each sample and at the trial's end.
  Kokkos::View<Complex****, Layout, MemorySpace> adams_block_;
  // A host's channel vectors per bin, `(n_omega, (adams_max_samples + 1) n_modes)`,
  // beside a block wider than `adams_max_channels` (solprop SEP 47 phase 4); empty
  // otherwise, when a host task holds them on its stack.
  ComplexView2 adams_channels_;
  // Each build's functions of the bins off the flow's eigenbasis route, `(n_active,
  // augmented_blocks(slots), n_modes, n_modes)` by active bin: the augmented
  // exponential's first block row and its scratch for the trial's section (solprop
  // SEP 47 phase 4); empty without a bin off the route.
  Kokkos::View<Complex****, Layout, MemorySpace> adams_fallback_;
  // A frozen-residual flow's generator at each sample applied to its state, `G(z_j)
  // u_j`, formed once per sample; the trial's at its end, and the weights it is formed
  // from. A trial forms `R_s(z_j) u_j = G(z_j) u_j - G_s u_j` against its own section.
  ComplexView3 adams_applied_;
  ComplexView2 adams_end_applied_;
  Kokkos::View<double*, Layout, MemorySpace> adams_end_weights_;
  std::size_t adams_dense_count_{};
  // solprop SEP 47: a Runge-Kutta step's continuous extension for dense output, its
  // five coefficients `(5, columns, n_omega)`; the bins its flow changes by more than
  // an e-fold; the committed state held across an observation; the tableau's
  // dense-output weights; and whether a step is held.
  ComplexView3 rk_dense_coefficients_;
  Kokkos::View<unsigned char**, Layout, MemorySpace> rk_dense_stiff_;
  ComplexView2 rk_dense_held_;
  std::vector<double> rk_dense_weights_;
  bool rk_dense_ready_{};
  bool adams_trial_ready_{};
  bool adams_trial_compensated_{};
  bool adams_trial_coupled_{};
  bool adams_dense_compensated_{};
  // The persistent row view a resume exports or installs, by name (SEP 29 G2).
  ComplexView3 adams_row_view(const std::string& name) const;
  IndexView crop_index_;
  // The fine bins outside the crop, zeroed by the stage's entry kernel.
  IndexView fine_gaps_;
  // The separable core's fine bins outside the crop, zeroed per chunk instead of the
  // whole fine spectrum (SEP 28); empty when the crop fills every fine bin.
  IndexView separable_gaps_;
  // Free space's node field and polarisation alias the chunk's time rows and their
  // projected rows (SEP 28): the copies between them are identities.
  bool separable_aliased_{};
  // The in-stage collar with no temporal rate beside it: its rate depends on the
  // column alone, so its pass needs no temporal pair (solprop SEP 40 section 4.7).
  bool spatial_only_absorber_{};
  std::size_t temporal_tail_rows_{};
  TailRowPlans<ExecutionSpace, ComplexView2, RealView2> temporal_tail_inverse_real_,
      coarse_tail_inverse_real_;
  TailRowPlans<ExecutionSpace, RealView2, ComplexView2> temporal_tail_forward_real_,
      coarse_tail_forward_real_;
  TailRowPlans<ExecutionSpace, ComplexView2, ComplexView2>
      temporal_tail_inverse_complex_, temporal_tail_forward_complex_,
      coarse_tail_inverse_complex_, coarse_tail_forward_complex_;
  std::unique_ptr<InverseRealPlan> inverse_real_plan_;
  std::unique_ptr<ComplexPlan> inverse_complex_plan_;
  std::unique_ptr<ForwardRealPlan> forward_real_plan_;
  std::unique_ptr<ComplexPlan> forward_complex_plan_;
  std::unique_ptr<ForwardRealPlan> forward_coarse_real_plan_;
  std::unique_ptr<ComplexPlan> forward_coarse_complex_plan_;
  std::unique_ptr<ComplexPlan> statistics_inverse_plan_;
  std::unique_ptr<ForwardRealPlan> absorber_real_plan_;
  std::unique_ptr<ComplexPlan> absorber_complex_plan_;
  std::unique_ptr<InverseRealPlan> coarse_inverse_real_plan_;
  // A free-space envelope's split collar transforms the state's own rows in place, this
  // many at a time, and holds no coarse buffers (`split_state_rows`).
  std::size_t split_rows_{};
  std::unique_ptr<ComplexPlan> split_inverse_plan_, split_forward_plan_;
  std::unique_ptr<ComplexPlan> split_tail_inverse_plan_, split_tail_forward_plan_;
  std::unique_ptr<ComplexPlan> coarse_inverse_complex_plan_;
  std::unique_ptr<ForwardRealPlan> coarse_forward_real_plan_;
  std::unique_ptr<ComplexPlan> coarse_forward_complex_plan_;
  std::unique_ptr<RamanForwardPlan> raman_drive_forward_plan_;
  std::unique_ptr<RamanInversePlan> raman_convolution_inverse_plan_;
  std::unique_ptr<RamanForwardPlan> raman_tail_forward_plan_;
  std::unique_ptr<RamanInversePlan> raman_tail_inverse_plan_;
  std::unique_ptr<RamanForwardPlan> raman_product_forward_plan_;
  std::unique_ptr<RamanInversePlan> raman_product_inverse_plan_;
  std::unique_ptr<RamanForwardPlan> raman_kernel_forward_plan_;
  // The retained synthesis's node transforms, full and tail chunks: from the leading
  // rows of the modal time buffer, which then holds node spectra, to the node field.
  std::unique_ptr<ComplexPlan> retained_inverse_plan_, retained_tail_inverse_plan_;
  // solprop's general modal route; null on every other route.
  std::unique_ptr<GeneralModalSource> general_;
  bool absorber_quiet_ = false;
  // solprop's boundary monitor (SEP 21, ADR-057), bound for a run that watches its
  // window: its four time weights `(4, n_t)` (low guard, high guard, the collar's
  // 1 - W, the collar's rate), each state column's quadrature weight, and what the
  // split-step collar took since the last reading beside one split's sums, the
  // energies of what the last step's split found and the share it takes at each
  // time. Allocated at the first bind, a few rows, and kept.
  bool monitor_bound_ = false;
  RealView2 monitor_weights_;
  RealView monitor_column_weights_;
  RealView monitor_share_, monitor_half_;
  // A device's split factor per coarse sample, formed once an accepted step.
  RealView split_factor_;
  Kokkos::View<double, Layout, MemorySpace> monitor_removed_;
  Kokkos::View<SplitSums, MemorySpace> monitor_split_sum_, monitor_found_;
  // The split collar's walk-off bucket of each (mode, frequency) bin and the state's
  // energy per bucket, which a reading returns for the crossing limit (solprop
  // #440); empty unbound.
  ByteView occupancy_buckets_;
  std::size_t occupancy_count_ = 0;
  mutable CrossedBytes crossed_;
  // What the transverse edge's reading sums with, once a monitor is bound (SEP 44).
  RealView2 edge_faces_, edge_clip_, edge_totals_, edge_synthesis_;
  // A radial edge's node fields at the rows a read uses, real and imaginary parts as
  // columns, when the state has one polarisation (`edge_reading`); empty otherwise.
  RealView2 edge_nodes_;
  RealView edge_columns_;
  std::size_t edge_first_row_ = 0;
  // A requested collar's lines, and one padded batch per distinct shape, kept across a
  // runner's points. Native chooses none of what's in them.
  struct EdgeLine {
    std::size_t axis{}, nodes{}, rows{}, batch{}, first{}, last{};
    double factor{}, measure{}, step{};
    std::size_t onset{EdgeLines::no_onset}, before{}, after{};
    IndexView positions;
    RealView window;
  };
  struct EdgeBatch {
    ComplexView2 buffer;
    std::unique_ptr<StripPlan> plan;
    std::size_t heap_growth{}, device_growth{};
  };
  std::vector<EdgeLine> edge_lines_;
  std::map<std::pair<std::size_t, std::size_t>, EdgeBatch> edge_batches_;
  ComplexView edge_strip_;
  RealView edge_scale_;
  double edge_k_min_{};
  // A large crossing's pinned chunks, once one has been made (`crossing.hpp`).
  mutable crossing::Ring crossing_ring_;
  [[nodiscard]] bool monitoring_split() const noexcept {
    return monitor_bound_ && problem_.absorber.split_step;
  }
  // Whether the stage adds the in-stage absorber's term. A quiet temporal collar is
  // left out on the modal routes, as solprop's reference leaves it out, unless a
  // real-space collar shares the term or the route is the general modal one or a
  // coupled flow, which keep it (solprop SEP 21).
  [[nodiscard]] bool absorbing() const noexcept {
    return problem_.absorber.in_stage &&
           !(absorber_quiet_ && !general_ && !problem_.flow.enabled &&
             problem_.spatial.absorber_matrix.empty());
  }
  // solprop's linear.coupled_flow; null otherwise.
  std::unique_ptr<CoupledFlowAction> flow_;
  // solprop SEP 6's dispersive electronic response on the fused node path, and on the
  // Cartesian free-space, line and planar routes; null otherwise. The general modal
  // route's is its source's.
  std::unique_ptr<DispersiveResponse> dispersive_;
  // solprop SEP 23's far-field pass, on free space with spatial statistics.
  std::unique_ptr<FarFieldPass> far_field_;
  std::unique_ptr<SeparableDispersive> separable_dispersive_;
  // The response's layout and stage weights on whichever route runs it, or null.
  DispersiveWeights* dispersive_weights() const noexcept {
    if (dispersive_)
      return &dispersive_->weights();
    if (separable_dispersive_)
      return &separable_dispersive_->weights();
    if (general_ && general_->dispersive() != nullptr)
      return &general_->dispersive()->weights();
    return nullptr;
  }
};

} // namespace solprop_native
