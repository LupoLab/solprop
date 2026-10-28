// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace solprop_native {

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;
using Layout = Kokkos::LayoutRight;
/** A two-dimensional range whose last index runs fastest, as `Layout` stores it.

    Kokkos's CUDA and HIP default runs the first index fastest, which puts adjacent
    threads a whole row apart in a `LayoutRight` view; a host's default is already
    this order. */
using Range2 = Kokkos::MDRangePolicy<
    ExecutionSpace, Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
using Complex = Kokkos::complex<double>;

/** The boundary monitor's five sums of a state (solprop SEP 21), one reduction value:
    the energy, then the low guard's, the high guard's, the collar's and the rate's. */
struct MonitorSums {
  Kokkos::Array<double, 5> values{};
  KOKKOS_INLINE_FUNCTION MonitorSums& operator+=(const MonitorSums& other) {
    for (std::size_t slot = 0; slot < 5; ++slot)
      values[slot] += other.values[slot];
    return *this;
  }
};

using ComplexView = Kokkos::View<Complex*, Layout, MemorySpace>;
using ConstComplexView = Kokkos::View<const Complex*, Layout, MemorySpace>;
using RealView = Kokkos::View<double*, Layout, MemorySpace>;
using ConstRealView = Kokkos::View<const double*, Layout, MemorySpace>;

/** Signal exhaustion of the validated angular rule before source assembly. */
class EnvelopeConvergenceError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/** Below this many real-side samples per execution a modal or spatial FFTW plan runs
    on one thread (`ThreadPolicy::threaded_fft_min_samples` unless set at start). The
    crossover is platform-dependent. On Linux at 32768 the four-mode RDW's 32000-sample
    coarse batch (n_t 4000) ran serially, its transforms 2.5-3.2 times slower on an
    18-core Cascade Lake socket than the 32768-sample batch's four-thread ones; 16384
    also gains on Zen 3. On an Apple arm64 laptop the serial transforms are the faster
    ones at that size, so macOS keeps 32768 until its own crossover is swept.
   `__APPLE__` is the only platform switch here and for the stage boundary below:
   Windows and linux-aarch64 take the x86 Linux values, which were not measured there
   (solprop's roadmap, "Thread thresholds on Windows and linux-aarch64"). */
#if defined(__APPLE__)
inline constexpr std::size_t default_threaded_fft_min_samples = 32768;
#else
inline constexpr std::size_t default_threaded_fft_min_samples = 16384;
#endif

/** Process-start CPU policy; execution instances never exceed the runtime budget. */
struct ThreadPolicy {
  int kernels{}, fft{}, raman_fft{}, raman_tail_fft{};
  // Zero takes `default_threaded_fft_min_samples`.
  std::size_t threaded_fft_min_samples{};
  // The field plasma column's band form (solprop SEP 25 section 6.7.6): negative takes
  // the build's (tiled on a device, the team column on the host), zero the team
  // column, and N > 0 the tiled column with host teams of N threads.
  int band_tiles{-1};
  // Bit per pool: keep explicit requests distinct from automatic capacities.
  unsigned explicit_pools{};
  bool operator==(const ThreadPolicy&) const = default;
};
void configure_thread_policy(ThreadPolicy requested);

/** Whether the field plasma column runs its tiled form under `policy`. */
[[nodiscard]] inline bool band_tiled(const ThreadPolicy& policy) noexcept {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  return policy.band_tiles != 0;
#else
  return policy.band_tiles > 0;
#endif
}

/** A tiled band column's host team width under `policy`; zero on a device, whose
    teams are its response teams. */
[[nodiscard]] inline int band_tile_threads(const ThreadPolicy& policy) noexcept {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  static_cast<void>(policy);
  return 0;
#else
  return policy.band_tiles > 0 ? policy.band_tiles : 0;
#endif
}
[[nodiscard]] ThreadPolicy thread_policy();
[[nodiscard]] ExecutionSpace execution_instance(int threads);

/** Independent CPU operation classes; device dispatch is unchanged. */
enum class CpuOperation {
  stage,
  response,
  dense,
  fft,
  raman,
  copy,
  flow,
  modal,
  adams
};
/** Process-start pool supplying an operation's capacity and explicit override. */
enum class CpuPool { kernels, fft, raman_fft, raman_tail_fft };

// Shared CPU policy; solprop docs/dev/benchmarks.md, section
// "Kokkos CPU operation-size threading", records the complete windows and sweeps.
// Work is elements for stages/copies, active columns * fine samples for responses,
// rows * inner * samples for dense products, active bins * modes^3 for matrix
// functions, elements * history rows for the exponential Adams trial's passes, and
// actual real-side batch samples for FFT/Raman plans (including tails).
// An 8196-element RDW stage needs the full pool in the server calibration; putting its
// reductions on a separate width loses that gain. In contrast, an 8192-element temporal
// copy must stay serial: threading it changes averaged helium from 0.74 to 1.27 s in
// that calibration. Native products stay serial below the conservative one-million-work
// crossover: 2x2x8192 favours serial on both hosts, whereas 33x4x8192 favours the full
// pool. Intermediate native-product widths did not establish a complete-window benefit.
// General-modal synthesis/projection have dedicated team kernels, with work
// active nodes * components * spectral samples * modes. Keep their measured
// full-width threshold separate: serialising these kernels regresses complete
// windows on both hosts, despite the generic small-GEMM crossover above.
// Responses retain four threads for many short columns (33x4096), while long
// 65536-sample columns enable the full pool without threading small copy kernels.
// Wider chunks also need the full pool: switching between a four-thread response
// and a full-width FFT regressed the Cartesian filament on the server. The column
// boundary retains the measured 33-column regime and includes 44-column tails.
// This is a construction-time shape rule, shared by both hosts and all responses.
// The stage boundary is platform-dependent. At 8192, just under the calibrated
// 8196-element RDW stage, the same case on a 2-3-5-smooth grid (n_t 4000, 8004
// elements) ran its stage entry serially on Linux: 708 us against 59 us at full width
// on an 18-core Cascade Lake socket, 1.8 times the whole evaluation of n_t 4096. At
// 4096 (with the FFT boundary at 16384) n_t 4000 costs no more than 4096 there and on
// Zen 3; at 1024 a 1351-element SSFS stage regressed. On an Apple arm64 laptop the
// serial stage is the faster one at 8004 elements (and 4096's threaded one is already
// dearer than 4000's serial), so macOS keeps 8192 until its own crossover is swept.
// Windows and linux-aarch64 take the Linux value, unmeasured there.
#if defined(__APPLE__)
inline constexpr std::size_t stage_thread_min_elements = 8192;
#else
inline constexpr std::size_t stage_thread_min_elements = 4096;
#endif
inline constexpr std::size_t copy_thread_min_elements = 65536;
// The 38691-work scalar-wall flow needs full width: its serial block functions
// took 2.39 s versus 0.13 s with the full pool in the server window profile.
inline constexpr std::size_t flow_thread_min_elements = 32768;
inline constexpr std::size_t modal_thread_min_elements = 65536;
inline constexpr std::size_t response_thread_min_elements = 32768;
inline constexpr std::size_t dense_thread_min_elements = 1048576;
inline constexpr std::size_t raman_thread_min_elements = 65536;
// ADAMS_THRESHOLDS_PLACEHOLDER
inline constexpr std::size_t adams_thread_min_elements = 4096;
inline constexpr std::size_t adams_full_thread_elements = 16384;
inline constexpr std::size_t stage_full_thread_elements = stage_thread_min_elements;
inline constexpr std::size_t response_full_thread_elements = 196608;
inline constexpr std::size_t response_full_thread_samples = 65536;
inline constexpr std::size_t response_full_thread_columns = 40;
inline constexpr std::size_t dense_full_thread_elements = 1048576;
inline constexpr std::size_t fft_full_thread_elements = 131072;
inline constexpr std::size_t raman_full_thread_elements = 262144;
inline constexpr int cpu_medium_threads = 4;
/** Choose a width without changing the runtime or a vendor library's pool. */
[[nodiscard]] int operation_threads(const ThreadPolicy& policy, CpuPool pool,
                                    CpuOperation operation, std::size_t elements,
                                    std::size_t fine_samples = 0);

using CpuSizing = std::map<std::string, std::map<std::string, std::size_t>>;
#if defined(KOKKOS_ENABLE_OPENMP) && !defined(KOKKOS_ENABLE_CUDA) &&                   \
    !defined(KOKKOS_ENABLE_HIP)
inline constexpr bool cpu_size_policy_enabled = true;
#else
inline constexpr bool cpu_size_policy_enabled = false;
#endif

/** Own execution instances constructed only while a workspace and its plans form. */
class CpuExecutionPool {
public:
  explicit CpuExecutionPool(CpuPool pool) : pool_(pool) {}
  const ExecutionSpace& get(CpuOperation operation, std::size_t elements,
                            std::size_t fine_samples = 0);
  void clear() { instances_.clear(); }
  [[nodiscard]] CpuSizing sizing() const { return sizing_; }

private:
  CpuPool pool_;
  std::map<int, ExecutionSpace> instances_;
  CpuSizing sizing_;
};

/** solprop SEP 28 section 4.2's per-evaluation reference, independent of any
    workspace: the seconds of each timed forward-then-backward pair of an in-place,
    unnormalised rank-3 complex transform of `values` (row-major
    `(planes, n_y, n_x)`, copied in once) through Kokkos-FFT on the FFT execution
    space, after planning and `warm` untimed pairs. */
[[nodiscard]] std::vector<double>
reference_fft3_pairs(const std::complex<double>* values, std::size_t planes,
                     std::size_t n_y, std::size_t n_x, std::size_t warm,
                     std::size_t pairs);

/** The same reference's streaming volume-pass: the seconds of each timed scaled copy
    `b = s a` between two complex volumes of `elements` first touched in parallel on
    the kernels' execution space, after `warm` untimed copies. */
[[nodiscard]] std::vector<double>
reference_stream_copies(std::size_t elements, std::size_t warm, std::size_t repeats);

enum class FieldFamily { field_real, field_analytic, envelope };

enum StatisticsFlag : std::uint64_t {
  statistics_omega0 = 1ULL << 0U,
  statistics_energy = 1ULL << 1U,
  statistics_energy_window = 1ULL << 2U,
  statistics_peak_power = 1ULL << 3U,
  statistics_peak_power_window = 1ULL << 4U,
  statistics_peak_power_total = 1ULL << 5U,
  statistics_fwhm = 1ULL << 6U,
  statistics_electron_density = 1ULL << 7U,
  statistics_peak_ionisation_rate = 1ULL << 8U,
  statistics_peak_intensity = 1ULL << 9U,
  statistics_quadrature_error = 1ULL << 10U,
  statistics_populations = 1ULL << 11U,
  // solprop SEP 18's run diagnostics: the six maxima over every node of the field
  // plasma column's integration (exposure dip and rise, |paid - freed| and freed, the
  // free-current integrand in the first 2K samples and anywhere).
  statistics_plasma_integration = 1ULL << 12U,
};

/** The widest band rule solprop designs (SEP 18 section 5): K <= 8, so 9 taps. */
inline constexpr std::size_t plasma_max_taps = 9;

/** The six parts of SEP 18's run diagnostics, in solprop's order. */
inline constexpr std::size_t plasma_diagnostic_parts = 6;

/** Fixed, Python-free problem data uploaded once by a native state. */
struct Problem {
  std::size_t n_x{}, n_y{};
  std::vector<Complex> spatial_k_squared, spatial_k_squared_low, spatial_numerator,
      spatial_clamped;
  std::vector<double> spatial_kperp_sq, spatial_kperp_sq_low, spatial_frame,
      spatial_spectral_rate;
  std::vector<double> spatial_absorber_rate;
  // solprop kernel schema 31 (SEP 22 phase 2): a transverse collar's power rate per
  // column, added to the field attenuation on active bins; zeros leave the operator.
  std::vector<double> spatial_kperp_rate;
  double spatial_evanescent_rate{};
  // solprop SEP 23 (free space only): the stored transverse axes k_x (n_x) and k_y
  // (n_y) in rad/m, FFT order, column ix + n_x iy, which the far-field statistics read;
  // and the slots of the problem's own far-field request, which size the band partials
  // at construction (zero for none). Empty and zero elsewhere.
  std::vector<double> spatial_k_x, spatial_k_y;
  std::size_t spatial_far_field_slots{};
  // solprop's hybrid planar kind (kernel schema 15): n_p slab
  // channels in x on the real (n_x, n_p) synthesis and (n_p, n_x) projection, times a
  // raw DFT in y. Columns are (p, k_y) with the channel fastest, n_modes = n_p * n_y,
  // and nodes (x, y) with x fastest, n_nodes = n_x * n_y; node_chunk holds whole y
  // rows. Row-major (n_p, n_omega) wall terms kappa_p^2 enter the separable operator,
  // and spatial_absorber_rate is then per column. Zero n_p and empty kappa elsewhere.
  std::size_t n_p{};
  std::vector<Complex> spatial_kappa_sq;
  // solprop kernel schema 35 (SEP 34): a non-separable spatial run's plane measure,
  // the entrance's material squares (n_omega) and transverse squares (n_modes) whose
  // root is solprop's `longitudinal_root`; `flux_table` when the material square
  // varies along z and is uploaded as a `TableKind::flux` table of n_omega values. A
  // separable run's measure is its own operator's (these stay empty). `flux_closure`
  // allocates the magnetic closure's buffers at construction, for a request with a
  // local flux statistic.
  std::vector<Complex> flux_k_squared, flux_k_squared_low;
  std::vector<double> flux_kperp_sq, flux_kperp_sq_low;
  bool flux_table{};
  bool flux_closure{};
  /** Free space or the hybrid planar kind: the separable operator and a y FFT. */
  [[nodiscard]] bool separable_geometry() const noexcept { return n_x != 0; }
  [[nodiscard]] bool slab_geometry() const noexcept { return n_p != 0; }
  [[nodiscard]] bool free_geometry() const noexcept { return n_x != 0 && n_p == 0; }
  std::size_t n_omega{};
  std::size_t n_t{};
  std::size_t n_to{};
  std::size_t n_modes{1};
  std::size_t n_pol_state{1};
  std::size_t n_pol_transform{1};
  std::size_t n_nodes{1};
  std::size_t node_chunk{1};
  std::size_t n_stages{};
  // solprop SEP 24 phase 3 (kernel schema 32): the low-storage RK4IP march. With it no
  // Runge-Kutta view is allocated (stages, accumulators, accept buffer, residual,
  // recycled stage, whole-step propagator, stage inverse); the accumulator register
  // is, the stage register unless it is elided (solprop's
  // `low_storage_stage_register`), and `low_storage_flows` flow volumes (0
  // recomputed, 1 a constant operator's table, 2 a varying operator's step pair).
  bool low_storage{};
  bool low_storage_stage_register{true};
  std::size_t low_storage_flows{};
  std::size_t n_phi_terms{};
  std::size_t n_prefactor_terms{};
  // solprop SEP 29 G1: Phi or the prefactor is the general z table. Its terms are then
  // `n_phi_terms` (or `n_prefactor_terms`) slots that `table_evaluate` fills with the
  // table's values at a trial's positions, and `phi_terms` (`prefactor_terms`) is
  // empty.
  bool phi_table{};
  bool prefactor_table{};
  FieldFamily family{FieldFamily::field_real};
  bool nonlinear{};
  bool modified_noise{};
  bool in_stage_absorber{};
  bool split_step_absorber{};
  bool first_same_as_last{};
  std::vector<std::size_t> crop_index;
  std::vector<unsigned char> active_mask;
  std::vector<unsigned char> norm_mask;
  std::vector<double> omega_window;
  std::vector<double> omega;
  std::vector<double> power_weight;
  std::vector<double> time_window;
  std::vector<double> time_window_derivative;
  std::vector<double> absorber_fine;
  std::vector<double> absorber_coarse;
  std::vector<double> spatial_absorber_matrix;
  std::vector<double> spatial_spectral_weights;
  std::vector<Complex> spatial_on_axis;
  std::vector<Complex> carrier;
  std::vector<Complex> phi_terms;
  std::vector<Complex> prefactor_terms;
  std::vector<double> synthesis_matrix;
  std::vector<double> projection_matrix;
  std::vector<double> projection_coarse_matrix;
  std::vector<double> tableau_matrix;
  std::vector<double> tableau_nodes;
  std::vector<double> solution_weights;
  std::vector<double> error_weights;
  double synthesis{};
  double projection{};
  double kerr_coefficient{};
  double self_factor{};
  double conjugate_factor{};
  double harmonic_factor{};
  bool cubic_real{};
  bool take_real{};
  bool plasma{};
  bool envelope_plasma{};
  // Channel-major 4 x 512 split angular rules and coarse-axis current factors.
  std::vector<double> envelope_rule_nodes, envelope_rule_weights, envelope_drude_factor;
  std::vector<Complex> envelope_loss_factor;
  bool raman{};
  bool raman_thg{};
  bool raman_tensor_capable{};
  bool stash_plasma{};
  bool weighted_controller{};
  bool phase_b_engine{};
  double dt_fine{};
  double dt{};
  std::size_t n_raman_terms{};
  double raman_fraction{};
  double raman_epsilon0{};
  std::vector<double> raman_undamped;
  std::vector<double> raman_tensor_weights;
  double charge_mass_ratio{};
  double ionisation_potential{};
  double log_field_min{};
  double d_log_field{};
  double field_min{};
  double field_max{};
  double rate_max{};
  std::size_t rate_intervals{};
  std::vector<double> rate_coefficients;
  std::size_t n_channels{1};
  // solprop SEP 18 (kernel schema 25): the field plasma column's running-integral
  // rule, one per problem and shared by every channel. Each running integral is the
  // cumulative trapezoid of the integrand filtered by the symmetric taps
  // (c_0, ..., c_K), centre first, with the window's ends closed by point reflection,
  // and the exposure floored at zero where the fraction is formed. The trapezoid rule
  // is `plasma_band == false` with the one identity tap, and runs the version-24
  // column unchanged.
  bool plasma_band{};
  std::vector<double> plasma_taps{1.0};
  [[nodiscard]] std::size_t plasma_half_width() const noexcept {
    return plasma_taps.empty() ? 0 : plasma_taps.size() - 1;
  }
  // Channel-major rows: log_min, d_log, field_min, field_max, rate_max,
  // interval count, coefficient offset, charge/mass ratio, ionisation potential.
  std::vector<double> channel_parameters, channel_coefficients;
  std::size_t n_raman_components{1};
  std::vector<double> raman_row_component, raman_component_fraction;
  // solprop's general modal route. Row-major: synthesis (map terms,
  // nodes, components, modes, n_omega); projection (map terms, modes, nodes,
  // components, n_omega); tensor (nodes, C, C, C, C); delayed tensor (responses,
  // terms, C, C, C, C); delayed spectrum (responses, terms, n_to + 1); node response
  // (nodes), the owning response plus one and zero for none. Empty otherwise. The
  // map terms are general_map_intervals x general_map_n_deg Chebyshev coefficients
  // along z, one for a constant guide; a moving guide's are combined per stage over
  // TrialScalars::transform_weights.
  bool general_modal{};
  bool general_real_form{};
  std::size_t general_map_intervals{1}, general_map_n_deg{1};
  std::size_t general_components{}, general_nodes{}, general_node_chunk{};
  std::size_t general_responses{}, general_terms{};
  double general_pad_scale{}, general_epsilon0{};
  std::vector<Complex> general_synthesis, general_projection, general_frequency_factor,
      general_delayed_spectrum;
  std::vector<double> general_tensor, general_delayed_tensor;
  std::vector<std::size_t> general_node_response;
  // solprop's linear.coupled_flow. flow_slot (n_omega)
  // numbers the bins a section's generator is non-zero at 1, 2, ... in increasing
  // frequency, zero elsewhere. With an explicit residual (flow_terms > 0), row-major
  // frozen blocks (sections, active, modes, modes) and residual terms (terms, active,
  // modes, modes); both empty otherwise. The actions are loaded per step.
  bool coupled_flow{};
  std::size_t flow_active{}, flow_sections{}, flow_terms{};
  std::vector<std::size_t> flow_slot;
  std::vector<Complex> flow_frozen, flow_residual_terms;
  // The actions' native formation, all present or all
  // empty: each section's eigenbasis of its Schur factor, row-major left `W` and
  // right `Y` (sections, active, modes, modes) and eigenvalues (sections, active,
  // modes); the route (sections, active), zero where a bin takes its eigenbasis and
  // 1, 2, ... numbering in row-major order the bins formed by scaling and squaring,
  // whose Schur factors `Z` and `T` are the rows of the fallback arrays (fallback,
  // modes, modes).
  std::vector<Complex> flow_left, flow_right, flow_eigenvalues;
  std::vector<std::size_t> flow_route;
  std::vector<Complex> flow_fallback_vectors, flow_fallback_triangular;

  // solprop SEP 6's frequency-dependent electronic response
  // (`responses.dispersive_kerr`, kernel schema 24) on the fused node path, laid out by
  // the adapter as solprop's `dispersive_terms_layout` lays it out (`dispersive.hpp`).
  // The form is 0 (a field grid's full cubic), 1 (the number-conserving sector) or 2
  // (an envelope's full cubic). Row-major on the leaf's spectral layout of `bins` (n_to
  // / 2 + 1 on a field grid, n_to on an envelope): the input multipliers (fields,
  // bins), each output term's multiplier (terms, bins) with its product row and factor,
  // the static coefficients and the polynomial's pair constants (channels, 2) per
  // channel, the groups as offsets (groups + 1) into members (channels), each symmetric
  // mixture's channels' `g^4` (channels, bins) and the input and output scales (bins).
  // On the general modal route (SEP 6 Phase 3b) each group's key is its node region
  // (groups), the polynomial forms one cube per distinct key, and every general node's
  // region (general_nodes) says where a channel acts; both are empty elsewhere, where
  // every channel acts at every node. Empty otherwise.
  bool dispersive{};
  bool dispersive_symmetric{};
  int dispersive_form{};
  std::vector<double> dispersive_inputs, dispersive_term_multipliers,
      dispersive_coefficients, dispersive_pair_constants, dispersive_quartics,
      dispersive_input_scale, dispersive_output_scale;
  std::vector<std::size_t> dispersive_term_rows, dispersive_term_channels,
      dispersive_group_offsets, dispersive_group_members, dispersive_group_keys,
      dispersive_node_region;

  [[nodiscard]] std::size_t n_columns() const noexcept { return n_modes * n_pol_state; }
  [[nodiscard]] std::size_t effective_node_pol() const noexcept {
    return n_pol_transform * n_pol_state;
  }
  [[nodiscard]] bool needs_general_engine() const noexcept {
    return phase_b_engine || general_modal || coupled_flow || n_columns() != 1 ||
           n_nodes != 1 || n_pol_transform != 1 || plasma || raman ||
           weighted_controller || dispersive || !projection_coarse_matrix.empty();
  }
  [[nodiscard]] std::size_t raman_drives() const noexcept {
    if (effective_node_pol() == 2)
      for (std::size_t index = 1; index < raman_tensor_weights.size(); index += 2)
        if (raman_tensor_weights[index] != 0.0)
          return 3;
    return 1;
  }
  [[nodiscard]] std::size_t raman_kernels() const noexcept {
    return raman_drives() == 3 ? 2 : 1;
  }
};

/** The most samples an Adams formula of solprop's shipped orders combines. */
inline constexpr std::size_t adams_max_samples = 8;

/**
 * The most channels a coupled block the resident Adams engine steps may have (solprop
 * SEP 16 phase 6): its per-bin passes hold a block's channel vectors in registers. A
 * larger block takes solprop's host engine.
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
  double rows[3][adams_max_samples]{};
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
  double rows[3][2 * adams_max_samples]{};
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

/** Execution-space views, FFT plans, and scratch owned by one State. */
class Workspace;
class GeneralWorkspace;
class GeneralBatchWorkspace;

/** Persistent single-column execution-space state, plans, and scratch storage. */
class State {
public:
  explicit State(Problem problem);
  ~State();
  State(const State&) = delete;
  State& operator=(const State&) = delete;
  State(State&&) = delete;
  State& operator=(State&&) = delete;

  void set_initial(const std::vector<Complex>& field,
                   const std::vector<Complex>& noise);
  [[nodiscard]] std::array<double, 3> trial_step(double dz,
                                                 const TrialScalars& scalars);
  void finish(double dz);
  void promote_fsal();
  void clear_fsal();
  void set_absorber_quiet(bool quiet);
  /**
   * Bind solprop's boundary monitor (SEP 21): its four time weights, row-major
   * `(4, n_t)`, and its columns' quadrature by mode (empty for unit weights). From
   * here on each split-step finish adds what it takes to an accumulator, which
   * `monitor_reading` drains.
   */
  void bind_monitor(const std::vector<double>& weights,
                    const std::vector<double>& column_weights);
  void unbind_monitor() noexcept;
  /**
   * The bound monitor's reading of the accepted state: `sum_t P`, the four weighted
   * sums `sum_t w_k P`, and what the split collar took since the last reading.
   */
  [[nodiscard]] std::vector<double> monitor_reading();
  [[nodiscard]] std::vector<Complex> field_to_host() const;
  [[nodiscard]] std::map<std::string, std::size_t> memory_ledger() const;
  [[nodiscard]] CpuSizing execution_sizing() const;
  [[nodiscard]] double planning_seconds() const noexcept;
  [[nodiscard]] bool fsal_ready() const noexcept;
  [[nodiscard]] std::size_t rk_source_evaluations() const noexcept;
  [[nodiscard]] bool closed() const noexcept;
  void close() noexcept;

private:
  std::unique_ptr<Workspace> workspace_;
};

/** Multicolumn general state with batched FFTs and chunked node transforms. */
class GeneralState {
public:
  explicit GeneralState(Problem problem);
  ~GeneralState();
  GeneralState(const GeneralState&) = delete;
  GeneralState& operator=(const GeneralState&) = delete;
  GeneralState(GeneralState&&) = delete;
  GeneralState& operator=(GeneralState&&) = delete;

  void set_initial(const std::vector<Complex>& field,
                   const std::vector<Complex>& noise);
  [[nodiscard]] std::vector<double> trial_step(double dz, const TrialScalars& scalars);
  void prepare_trial(double dz, const TrialScalars& scalars);
  /** solprop's Adams source at an anchor-frame field (SEP 14 phase 2). */
  [[nodiscard]] std::vector<Complex> adams_source(const std::vector<Complex>& input,
                                                  const TrialScalars& scalars);
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
  /** Hold the accepted state at a field-free plane, in the memory space (SEP 22). */
  void hold_plane();
  /** The held plane's state on the host: the one copy a checkpoint takes. */
  [[nodiscard]] std::vector<Complex> held_to_host() const;
  /** The resident Adams history (solprop SEP 16 phase 3b, solprop ADR-043). */
  void adams_allocate(std::size_t slots, bool compensated, bool interpolate = true);
  void adams_compensation_start(std::size_t slot, const std::vector<double>& rate);
  void adams_start(std::size_t slot, const TrialScalars& scalars);
  [[nodiscard]] double adams_magnitude();
  void adams_scale(double rtol, double atol, const std::vector<double>& weights);
  [[nodiscard]] std::vector<double>
  adams_fused_trial(double dz, const AdamsWeights& weights,
                    const std::vector<double>& divisor, const TrialScalars& scalars);
  void adams_fused_accept(double dz, const AdamsWeights& keep, std::size_t free,
                          const AdamsWeights& dense);
  void adams_hold();
  void adams_restore();
  void adams_observe(const AdamsWeights& weights, bool hermite,
                     const TrialScalars& scalars);
  [[nodiscard]] std::vector<double>
  adams_exponential_trial(double dz, const ExponentialWeights& weights,
                          const std::vector<double>& divisor, const TrialScalars& step,
                          const TrialScalars& sample, const std::vector<double>& rate);
  void adams_exponential_accept(double dz, std::size_t free, const AdamsWeights& dense);
  void adams_exponential_observe(double xi, double dz,
                                 const ExponentialWeights& weights);
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
   * fluence at each node, the axis fluence, then each node's magnitude
   * `dt sum |E| |G|`), packed in that order.
   */
  [[nodiscard]] std::vector<double>
  spatial_statistics(std::uint64_t flags, double intensity_factor, double z);
  /**
   * solprop SEP 23's far-field statistics of the accepted state on a Cartesian or line
   * grid, one banded pass; see `FarFieldPass::run` (`far_field.hpp`) for the tables,
   * the flags (from 32) and the packed order.
   */
  [[nodiscard]] std::vector<double>
  aperture_statistics(std::uint64_t flags, const std::vector<double>& apertures,
                      int profile_code, int exponent,
                      const std::vector<double>& regions, double intensity_factor);
  [[nodiscard]] std::vector<double>
  statistics(std::uint64_t flags, const std::vector<double>& windows,
             const std::vector<double>& on_axis, double density, double taper,
             double preionisation, double intensity_factor,
             const std::vector<double>& channel_preionisation = {},
             const std::vector<double>& power_weight = {});
  void finish(double dz);
  void promote_fsal();
  void clear_fsal();
  void set_absorber_quiet(bool quiet);
  /** As `State::bind_monitor`; the columns' weights are by mode. */
  void bind_monitor(const std::vector<double>& weights,
                    const std::vector<double>& column_weights);
  void unbind_monitor() noexcept;
  /** As `State::monitor_reading`, over every column of the state. */
  [[nodiscard]] std::vector<double> monitor_reading();
  [[nodiscard]] double quadrature_error() const;
  [[nodiscard]] double quadrature_peak();
  [[nodiscard]] std::string synthesis_provider() const;
  [[nodiscard]] std::string projection_provider() const;
  [[nodiscard]] std::string gemm_shape_policy() const;
  [[nodiscard]] std::size_t response_team_size() const noexcept;
  [[nodiscard]] std::vector<Complex> field_to_host() const;
  [[nodiscard]] std::vector<Complex> stage_to_host(std::size_t stage) const;
  /**
   * The last evaluation's projected polarisation spectrum on the fine rule, or on the
   * embedded one (finishing SEP 6's dispersive products there first), a diagnostic.
   */
  [[nodiscard]] std::vector<Complex> projected_to_host(bool coarse);
  /** How SEP 6's dispersive response runs: empty without it. */
  [[nodiscard]] std::map<std::string, bool> dispersive_path() const;
  [[nodiscard]] std::map<std::string, std::size_t> memory_ledger() const;
  [[nodiscard]] CpuSizing execution_sizing() const;
  [[nodiscard]] double planning_seconds() const noexcept;
  [[nodiscard]] double raman_planning_seconds() const noexcept;
  [[nodiscard]] std::string raman_fft_provider() const;
  [[nodiscard]] std::map<std::string, int> raman_fft_tasks() const;
  [[nodiscard]] std::size_t raman_kernel_transforms() const noexcept;
  [[nodiscard]] bool fsal_ready() const noexcept;
  [[nodiscard]] std::size_t rk_source_evaluations() const noexcept;
  [[nodiscard]] bool closed() const noexcept;
  void close() noexcept;

private:
  friend class GeneralBatchWorkspace;
  std::unique_ptr<GeneralWorkspace> workspace_;
};

/** Shared update dispatch over independently owned native parameter workspaces. */
class GeneralBatch {
public:
  explicit GeneralBatch(std::vector<std::shared_ptr<GeneralState>> members);
  ~GeneralBatch();
  GeneralBatch(const GeneralBatch&) = delete;
  GeneralBatch& operator=(const GeneralBatch&) = delete;
  GeneralBatch(GeneralBatch&&) = delete;
  GeneralBatch& operator=(GeneralBatch&&) = delete;
  [[nodiscard]] std::vector<double> accept(double dz);
  [[nodiscard]] std::map<std::string, std::size_t> memory_ledger() const;
  [[nodiscard]] CpuSizing execution_sizing() const;
  void close() noexcept;

private:
  std::unique_ptr<GeneralBatchWorkspace> workspace_;
};

void fft(const ExecutionSpace& execution_space, ComplexView input, ComplexView output);
void multiply_phase(const ExecutionSpace& execution_space, ConstComplexView values,
                    ConstRealView phase, ComplexView output);

} // namespace solprop_native

template <> struct Kokkos::reduction_identity<solprop_native::MonitorSums> {
  KOKKOS_FUNCTION static solprop_native::MonitorSums sum() { return {}; }
};
