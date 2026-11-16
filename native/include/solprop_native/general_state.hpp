// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/cpu_execution.hpp"
#include "solprop_native/execution.hpp"
#include "solprop_native/problem.hpp"
#include "solprop_native/readings.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace solprop_native {

// host_source.hpp's types, which the state only passes on.
class HostSource;
enum class HostPurpose : int;

/** The most samples an Adams formula of solprop's shipped orders combines. */
inline constexpr std::size_t adams_max_samples = 8;

/** The most defect norms an Adams trial scores: its order's and its two neighbours'. */
inline constexpr std::size_t adams_max_norms = 3;

/**
 * The most channels of a coupled block whose channel vectors a host task of the
 * resident Adams engine holds on its stack (solprop SEP 16 phase 6). A wider block's
 * host passes hold them in a scratch row per bin, allocated with the history (solprop
 * SEP 47 phase 4). A device build's passes take a thread per (bin, channel) and
 * hold no channel vectors, so it steps any width.
 */
inline constexpr std::size_t adams_max_channels = 16;

/**
 * One Adams trial's coefficient tables, passed by value to the resident engine's
 * kernels (solprop SEP 16 phase 3b). `slot` names history rows newest first;
 * `predictor` weights the first `order` of them; `corrector` and each of `n_rows`
 * defect rows weight `(source, slot[0], ...)` over `width` samples.
 */
struct AdamsWeights {
  std::size_t slot[adams_max_samples]{};
  double predictor[adams_max_samples]{};
  double corrector[adams_max_samples]{};
  double rows[adams_max_norms][adams_max_samples]{};
  std::size_t order{};
  std::size_t width{};
  std::size_t n_rows{};
};

/**
 * One exponential Adams trial's coefficient tables (solprop SEP 16 phase 3b): the
 * predictor's Lagrange coefficients `predictor[sample][monomial]` over the `order`
 * newest rows, the corrector's over `(source, slot[0], ...)` and, per scored defect,
 * the divided-difference coefficients over the same `width` samples followed by the
 * node polynomial's `count` monomial coefficients.
 */
struct ExponentialWeights {
  std::size_t slot[adams_max_samples]{};
  double predictor[adams_max_samples][adams_max_samples]{};
  double corrector[adams_max_samples][adams_max_samples]{};
  double rows[adams_max_norms][2 * adams_max_samples]{};
  std::size_t order{};
  std::size_t width{};
  std::size_t count{};
  std::size_t n_rows{};
  // A varying operator's transported compensation (solprop SEP 16 section 4.2): the
  // carried samples' spans 1 - theta_j, and the predictor's, the corrector's and each
  // defect's weights over (endpoint, carried...), still to be multiplied by the step.
  bool compensated{};
  double span[adams_max_samples]{};
  double comp_predictor[adams_max_samples]{};
  double comp_corrector[adams_max_samples + 1]{};
  double comp_rows[3][adams_max_samples + 1]{};
  // Dense observation's weights over (endpoint, history...) at the fraction, and the
  // samples' normalised positions.
  double dense_weight[adams_max_samples + 1]{};
  double dense_theta[adams_max_samples + 1]{};
  // A scalar carrier frame (solprop SEP 35): sigma = i shift. The moments are those
  // of x - sigma, and history row t (the t-th of `slot`) is read rotated by
  // factor[t] = exp(sigma (1 - theta_t)); the new source's factor is one.
  bool shifted{};
  double shift{};
  // exp(-sigma), formed once a trial.
  double unshift_re{1.0};
  double unshift_im{};
  double factor_re[adams_max_samples]{};
  double factor_im[adams_max_samples]{};
};

/** Small host bundle for all stages and the whole-step propagator. */
struct TrialScalars {
  std::vector<double> phi_delta;
  std::vector<double> phi_at;
  std::vector<double> prefactor;
  std::vector<double> density;
  std::vector<double> taper;
  std::vector<double> preionisation;
  std::vector<double> raman_rates;
  std::vector<double> kerr_density, plasma_densities, plasma_preionisation,
      raman_densities;
  // A coupled flow's step section and its stages' residual weights (n_stages, terms).
  std::size_t flow_section{};
  std::vector<double> flow_weights;
  // A moving guide's map weights (n_stages, map terms), zero outside the section each
  // stage lies in (solprop kernel schema 14); empty for maps that do not move.
  std::vector<double> transform_weights;
  // Each stage's channel densities of SEP 6's dispersive response (n_stages,
  // channels), m^-3; empty without it.
  std::vector<double> dispersive_densities;
};

/** Which general z table a call names (solprop SEP 29 G1). */
enum class TableKind { phi, prefactor, rate, flux };

/** The z tables by name (`TableKind`). */
[[nodiscard]] constexpr auto names(const TableKind* /*unused*/) {
  return std::array{
      std::pair{"phi", TableKind::phi}, std::pair{"prefactor", TableKind::prefactor},
      std::pair{"rate", TableKind::rate}, std::pair{"flux", TableKind::flux}};
}

class GeneralWorkspace;

/** Multicolumn general state with batched FFTs and chunked node transforms. */
class GeneralState {
public:
  explicit GeneralState(Problem problem);
  ~GeneralState();
  GeneralState(const GeneralState&) = delete;
  GeneralState& operator=(const GeneralState&) = delete;
  GeneralState(GeneralState&&) = delete;
  GeneralState& operator=(GeneralState&&) = delete;

  /** Load the field and noise from host memory, one copy into the memory space. */
  void set_initial(std::span<const Complex> field, std::span<const Complex> noise);
  [[nodiscard]] std::vector<double> trial_step(double dz, const TrialScalars& scalars);
  void prepare_trial(double dz, const TrialScalars& scalars);
  /** Adopt an Adams step's end state through the shared finish. */
  void adopt(const std::vector<Complex>& field, double dz);
  /**
   * One step of solprop's low-storage RK4IP march (SEP 24 phase 3): four register
   * passes accepted in place, returning the raw accepted state's squared norm over the
   * norm mask. `scalars` holds one row per position -- 0 the step's start, 1 its
   * midpoint, 2 its end, any later rows ignored -- and its `phi_delta` rows 0 and 1
   * the flows' intervals `P` and `Q`, row 2 the table's, read only when
   * `rebuild_table` builds a constant operator's table.
   */
  [[nodiscard]] double low_storage_step(double dz, const TrialScalars& scalars,
                                        bool rebuild_table);
  /** The resident Adams history (solprop SEP 16 phase 3b, solprop ADR-043). */
  void adams_allocate(std::size_t slots, bool compensated, bool interpolate = true);
  void adams_compensation_start(std::size_t slot, const std::vector<double>& rate);
  void adams_start(std::size_t slot, const TrialScalars& scalars);
  void adams_scale(double rtol, double atol, const std::vector<double>& weights);
  [[nodiscard]] std::vector<double>
  adams_fused_trial(double dz, const AdamsWeights& weights,
                    const std::vector<double>& divisor, const TrialScalars& scalars);
  void adams_fused_accept(double dz, const AdamsWeights& keep, std::size_t free,
                          const AdamsWeights& dense);
  void adams_hold();
  void adams_restore();
  void adams_observe(const AdamsWeights& weights, const TrialScalars& scalars);
  [[nodiscard]] std::vector<double>
  adams_exponential_trial(double dz, const ExponentialWeights& weights,
                          const std::vector<double>& divisor, const TrialScalars& step,
                          const TrialScalars& sample, const std::vector<double>& rate);
  void adams_exponential_accept(double dz, std::size_t free, const AdamsWeights& dense);
  void adams_exponential_observe(double xi, double dz,
                                 const ExponentialWeights& weights);
  /**
   * A Runge-Kutta march's dense output in a native session (solprop SEP 47 section
   * 5.3), DOPRI5's continuous extension (Hairer, Norsett and Wanner II.6) as solprop's
   * `extension_coefficients` and `_observe_inside` form it. `rk_dense_allocate` keeps
   * the tableau's dense-output weights, one per stage, and the views the extension
   * needs; `rk_dense_hold` contracts the accepted, unfinished trial over `dz` -- the
   * state before it and its stages -- into the extension's five coefficients and marks
   * the bins the step's flow changes by more than an e-fold;
   * `rk_dense_observe` holds the committed state aside and makes the state the
   * extension at the fraction `theta`, carried from the anchor by the flow of `carry`
   * (a trial's arrays from the anchor to the plane) and masked; `rk_dense_restore`
   * makes the held state current again; `rk_dense_release` frees the views.
   */
  void rk_dense_allocate(const std::vector<double>& weights);
  void rk_dense_hold(double dz);
  void rk_dense_observe(double theta, const TrialScalars& carry);
  void rk_dense_restore();
  void rk_dense_release() noexcept;
  /**
   * The resident Adams history's persistent rows, for a resume (solprop SEP 29 G2):
   * `"store"`, and `"states"`, `"rates"` and `"applied"` where allocated, each
   * `(slots, columns, n_omega)`. Scratch views are not rows: a trial rebuilds them.
   */
  [[nodiscard]] std::vector<std::string> adams_row_names() const;
  /** A named row view's extents; refuses an unknown or unallocated name. */
  [[nodiscard]] std::array<std::size_t, 3>
  adams_row_extents(const std::string& name) const;
  /** A lossless host copy of a named row view, in its LayoutRight order. */
  [[nodiscard]] std::vector<Complex> adams_rows_to_host(const std::string& name) const;
  /** The same copy written straight into `destination`, of `size` elements. */
  void adams_rows_to_host(const std::string& name, std::complex<double>* destination,
                          std::size_t size) const;
  /** Replace a named row view with a host copy of its extents; forget any trial. */
  void adams_rows_from_host(const std::string& name,
                            const std::vector<Complex>& values);
  /**
   * The general z tables (solprop SEP 29 G1). `table_upload` makes a table resident:
   * Phi's, the prefactor's, or Phi's rate (the derivative series, for the compensated
   * Adams march); `table_evaluate` writes Phi's and the prefactor's values at each
   * position into the given term slots, and `table_rate` the rate's value into a Phi
   * term slot. `table_terms_to_host` copies one term slot to the host, and
   * `phase_sum_to_host` evaluates the stage kernels' phase sum over the Phi terms for
   * one row of weights; both are diagnostics.
   */
  void table_upload(TableKind kind, std::vector<double> edges, double lower,
                    double upper, const std::complex<double>* coefficients,
                    std::size_t n_intervals, std::size_t n_terms, std::size_t n_values);
  void table_evaluate(const std::vector<std::size_t>& slots,
                      const std::vector<double>& positions);
  void table_rate(std::size_t slot, double z, bool before);
  [[nodiscard]] std::vector<Complex> table_terms_to_host(TableKind kind,
                                                         std::size_t slot) const;
  [[nodiscard]] std::vector<Complex>
  phase_sum_to_host(const std::vector<double>& weights);
  [[nodiscard]] std::vector<double> weighted_error(const std::vector<double>& scale);
  /** Load a coupled flow's actions; see `CoupledFlowAction::set_actions`. */
  void set_flow_actions(const std::vector<Complex>& forward,
                        const std::vector<Complex>& inverse,
                        const std::vector<unsigned char>& identity);
  /** A coupled flow's held actions; see `CoupledFlowAction::actions_to_host`. */
  [[nodiscard]] std::vector<Complex> flow_actions_to_host(bool inverse) const;
  /** The last exponential Adams trial's coupled block functions, row-major
      `(slots + 1, n_active, n_modes, n_modes)`, the flow first: an observation for
      tests (solprop SEP 47 phase 4). */
  [[nodiscard]] std::vector<Complex> adams_block_to_host() const;
  /** The last Adams trial's exponent `Delta Phi`, row-major `(n_columns, n_omega)`:
      an observation for tests. */
  [[nodiscard]] std::vector<Complex> adams_exponent_to_host() const;
  /** Form a coupled flow's actions natively; see `CoupledFlowAction::form_actions`. */
  [[nodiscard]] std::vector<double>
  form_flow_actions(std::size_t section, double dz,
                    const std::vector<double>& fractions);
  void prepare_raman(const std::vector<double>& rates,
                     const std::vector<double>& densities = {});
  /**
   * The spatial statistics of the accepted state at the plane `z`, measured by the
   * medium's axial flux (solprop SEP 34): flags 1 energy, 2 on-axis peak intensity, 4
   * peak intensity, 8 on-axis spectrum, 16 the widths' fluence block (the signed
   * fluence at each node, the axis fluence, which a separable route leaves NaN since
   * only a radial width reads it, then the bound on the largest magnitude), 32 a
   * Cartesian plane's fluence moments, which hold its scaled profile for
   * `plane_fluence_trace` (the lowest value, its node, the bound on the largest
   * magnitude, the largest value, the count of values that aren't finite, then the
   * scaled sum and its x and y index moments), packed in that order.
   */
  [[nodiscard]] std::vector<double>
  spatial_statistics(std::uint64_t flags, double intensity_factor, double z);
  /**
   * The second moment and the azimuthal average of the plane's scaled fluence that a
   * `spatial_statistics` call with flag 32 left, on rings solprop's own rule fixes,
   * with the centre in metres from the axes' first samples. Returns the second-moment
   * sum, then one average per ring. The profile itself stays in the execution space.
   */
  [[nodiscard]] std::vector<double>
  plane_fluence_trace(double center_x, double center_y, double dx, double dy,
                      double extent, double spacing, std::size_t n_radii,
                      const std::vector<double>& cosines,
                      const std::vector<double>& sines);
  /**
   * The modal state's spectrum at one physical point: `matrix`, one or two rows of
   * `n_modes` columns, contracted with the modes, as `(n_rows * n_pol, n_omega)`
   * samples. What an on-axis scan needs of the state.
   */
  [[nodiscard]] std::vector<Complex> axis_spectrum(const std::vector<double>& matrix);
  /**
   * solprop SEP 19's on-axis plasma populations of the accepted state: `matrix`'s row
   * (as `axis_spectrum` takes it, the pad scale and the taper in it) on the fine time
   * grid, scanned by the node scan with each channel's preionisation. Returns each
   * channel's final ionised fraction and peak rate, `(n_channels, 2)`.
   */
  [[nodiscard]] std::vector<double>
  axis_populations(const std::vector<double>& matrix,
                   const std::vector<double>& channel_preionisation);
  /**
   * solprop SEP 23's far-field statistics of the accepted state on a Cartesian or line
   * grid, one banded pass; see `FarFieldPass::run` (`far_field.hpp`) for the tables,
   * the flags (from 32) and the packed order.
   */
  [[nodiscard]] std::vector<double>
  aperture_statistics(std::uint64_t flags, const std::vector<double>& apertures,
                      int exponent, const std::vector<double>& regions,
                      double intensity_factor);
  [[nodiscard]] std::vector<double>
  statistics(std::uint64_t flags, const std::vector<double>& windows,
             const std::vector<double>& on_axis, double density, double taper,
             double preionisation, double intensity_factor,
             const std::vector<double>& channel_preionisation = {},
             const std::vector<double>& power_weight = {});
  /**
   * solprop SEP 47: a general modal basis's statistics, functionals of the physical
   * flux `a.H W a` (`solprop.stats._whitened`). The state's columns are whitened by
   * `root`, `W^(1/2)` at each frequency, `(n_omega, n_modes, n_modes)` row-major
   * (empty for the identity), into a statistics scratch, and the temporal `flags`
   * (omega0, energy, peak power total and the width's trace) are evaluated on
   * `c = W^(1/2) a`. Returns their packed values, then each column's `sum |c|^2`,
   * then, with `spectral`, each bin's `sum |c|^2` over the columns. The state is
   * unchanged.
   */
  [[nodiscard]] std::vector<double>
  whitened_statistics(const std::vector<Complex>& root, std::uint64_t flags,
                      const std::vector<double>& windows, double intensity_factor,
                      bool spectral);
  void finish(double dz);
  void promote_fsal();
  void clear_fsal();
  void set_absorber_quiet(bool quiet);
  /**
   * Bind solprop's boundary monitor (SEP 21): its four time weights, row-major
   * `(4, n_t)`, and its columns' quadrature by mode (empty for unit weights). From
   * here on each split-step finish adds what it takes to an accumulator, which
   * `monitor_reading` drains, and keeps the energies of the power it found.
   */
  void bind_monitor(const std::vector<double>& weights,
                    const std::vector<double>& column_weights);
  void unbind_monitor() noexcept;
  /**
   * The bound monitor's reading of the accepted state, over every column: `sum_t P`,
   * the four weighted sums `sum_t w_k P`, what the split collar took since the last
   * reading, and `sum_t P` and the two guards' and the collar's sums of the power the
   * last step's split found (zeros before the first such step).
   */
  [[nodiscard]] std::vector<double> monitor_reading();
  /**
   * The last accepted step's split collar sums, read without draining anything (SEP
   * 52 section 5.2's `AcceptRecord`): what its split took, then `sum_t P` and the two
   * guards' and the collar's sums of the power it found, from the masked field before
   * the split; none without a bound monitor and a split collar.
   */
  [[nodiscard]] std::optional<std::array<double, 5>> split_sums();
  /**
   * Bind the split collar's walk-off bucket of each frequency bin, by mode
   * `(n_modes, n_omega)`, `count` buckets (solprop #440): from here on
   * `monitor_reading` also returns the state's energy per bucket, the fastest first.
   */
  void bind_occupancy(const std::vector<unsigned char>& buckets, std::size_t count);
  [[nodiscard]] double quadrature_error() const;
  [[nodiscard]] double quadrature_peak();
  [[nodiscard]] std::string synthesis_provider() const;
  [[nodiscard]] std::string projection_provider() const;
  [[nodiscard]] std::string gemm_shape_policy() const;
  [[nodiscard]] std::size_t response_team_size() const noexcept;
  [[nodiscard]] std::vector<Complex> field_to_host() const;
  /** The field into `destination`, of the field's size: one copy from the space. */
  void field_to_host(std::span<Complex> destination) const;
  /** The committed field where the host can read it in place, `columns * n_omega`
      elements in row-major order, after the state's work is fenced; null where the
      memory space is not host-accessible. A reader that passes it through the host
      adds its bytes with `count_to_host` (solprop SEP 47 section 5.4). */
  [[nodiscard]] const Complex* host_field() const;
  /** Count `bytes` read out of the state by a host reader of `host_field`. */
  void count_to_host(std::size_t bytes) const noexcept;
  /** What the transverse edge's reading sums with (solprop SEP 44). */
  void bind_edge(const std::vector<double>& faces, const std::vector<double>& clip,
                 const std::vector<double>& totals,
                 const std::vector<double>& synthesis, std::size_t first_row);
  /** The bytes through this state's crossings since it was built, each way. */
  [[nodiscard]] CrossedBytes crossed_bytes() const noexcept;
  /** The edge's region sums and each mode's energy, where the state lives. */
  /** A requested collar's lines, whose strip and current a reading then forms. */
  void bind_edge_lines(const EdgeLines& lines);
  EdgeNumbers edge_reading(bool faces, bool whole, bool totals, bool strip,
                           bool current);
  /** The threads a large crossing takes (`crossing.hpp`), from solprop's budget. */
  void set_crossing_workers(std::size_t workers) noexcept;
  [[nodiscard]] std::vector<Complex> stage_to_host(std::size_t stage) const;
  void stage_to_host(std::size_t stage, std::span<Complex> destination) const;
  /**
   * The last evaluation's projected polarisation spectrum on the fine rule, or on the
   * embedded one (finishing SEP 6's dispersive products there first), a diagnostic.
   */
  [[nodiscard]] std::vector<Complex> projected_to_host(bool coarse);
  /** How SEP 6's dispersive response runs: empty without it. */
  [[nodiscard]] std::map<std::string, bool> dispersive_path() const;
  [[nodiscard]] std::map<std::string, std::size_t> memory_ledger() const;
  [[nodiscard]] CpuSizing execution_sizing() const;
  /** The CPU execution record the workspace's pools choose widths by. */
  [[nodiscard]] const CpuExecution& cpu_execution() const;
  /** The workspace's FFT planning, with what its planning found. */
  [[nodiscard]] const FftPlanning& fft_planning() const;
  [[nodiscard]] double planning_seconds() const noexcept;
  [[nodiscard]] double raman_planning_seconds() const noexcept;
  [[nodiscard]] std::string raman_fft_provider() const;
  [[nodiscard]] std::map<std::string, int> raman_fft_tasks() const;
  [[nodiscard]] std::size_t raman_kernel_transforms() const noexcept;
  [[nodiscard]] bool fsal_ready() const noexcept;
  [[nodiscard]] std::size_t rk_source_evaluations() const noexcept;
  /** On-axis row scans made so far (`axis_populations`), cumulative. */
  [[nodiscard]] std::size_t axis_scans() const noexcept;
  [[nodiscard]] bool closed() const noexcept;
  void close() noexcept;
  /**
   * A host source (solprop SEP 49 section 6.2): from here on each stage source is
   * `source`'s evaluation of the lab-frame field, the state's own response passes not
   * run, and the stage's finish its collar, absorber and propagator alone. Its two
   * host rows are allocated here, once, and listed in the memory ledger. Refused on a
   * device build and for anything but one real column off the general and separable
   * routes; null unbinds.
   */
  void bind_host_source(std::shared_ptr<HostSource> source);
  /** The bound host source, or null. */
  [[nodiscard]] HostSource* host_source() const noexcept;
  /** The next host evaluation's position, m, and purpose, which the engine sets. */
  void host_evaluation(double z, HostPurpose purpose) noexcept;
  /** A test hook (solprop issue #129): `fault` runs, once, as the next Adams source
      evaluation starts, a trial's with its prediction in the state; empty disarms. */
  void arm_source_fault(std::function<void()> fault) noexcept;

private:
  /** `function(workspace, arguments...)` inside the workspace's CPU team scope
      (`CpuTeamScope`); defined and used in src/general/state.cpp alone, so that no unit
      outside src/general/ sees the workspace class. */
  template <class Function, class... Arguments>
  decltype(auto) with_team(Function function, Arguments&&... arguments) const;

  std::unique_ptr<GeneralWorkspace> workspace_;
};

} // namespace solprop_native
