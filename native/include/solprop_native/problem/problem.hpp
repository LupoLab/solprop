// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The Python-free problem a native state is built from (`Problem`): fixed data grouped
// as solprop's kernel contract groups it, filled field by field through each group's
// field table, and checked by `validate` before a state allocates anything.
//
// Types: `Problem` and its groups (`Dims`, `Grid`, `Spatial`, `Flux`, `Stepper`,
// `Linear`, `Absorber`, `Transform`, `Kerr`, `Plasma`, `Raman`, `General`, `Flow`,
// `Dispersive`), with its CPU execution record, FFT planning and shared set-once
// `Setup` (workspace/run_records.hpp); `FieldFamily`; the statistics and
// spatial-statistics flags; the plasma column's bounds (`plasma_max_taps`,
// `plasma_diagnostic_parts`). Functions: each group's `describe` (its field table,
// solprop SEP 52 section 5.3) and `compact_problem` (whether a problem runs on a
// compact CPU team).

#include "solprop_native/runtime/cpu_execution.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/fields.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace solprop_native {

enum class FieldFamily { field_real, field_analytic, envelope };

/** The field families by name, for the field tables (`fields.hpp`). */
[[nodiscard]] constexpr auto names(const FieldFamily* /*unused*/) {
  return std::array{std::pair{"field_real", FieldFamily::field_real},
                    std::pair{"field_analytic", FieldFamily::field_analytic},
                    std::pair{"envelope", FieldFamily::envelope}};
}

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

// `GeneralState::spatial_statistics`' scalar flags.
inline constexpr std::uint64_t spatial_energy = 1;
inline constexpr std::uint64_t spatial_axis = 2;
inline constexpr std::uint64_t spatial_peak = 4;
inline constexpr std::uint64_t spatial_spectrum = 8;
/** The widths' fluence block: the signed fluence at each node, the axis fluence and
    the largest magnitude. */
inline constexpr std::uint64_t spatial_fluence = 16;
/** A Cartesian plane's fluence moments, which hold its scaled profile for the trace. */
inline constexpr std::uint64_t spatial_moments = 32;
/** Every spatial flag. */
inline constexpr std::uint64_t spatial_flags = spatial_energy | spatial_axis |
                                               spatial_peak | spatial_spectrum |
                                               spatial_fluence | spatial_moments;
// What a Cartesian plane's widths read of its fluence (`plane_fluence_moments`).
inline constexpr std::size_t plane_moment_slots = 8;

/** The widest band rule solprop designs (SEP 18 section 5): K <= 8, so 9 taps. */
inline constexpr std::size_t plasma_max_taps = 9;

/** The six parts of SEP 18's run diagnostics, in solprop's order. */
inline constexpr std::size_t plasma_diagnostic_parts = 6;

struct Setup;

/** Fixed, Python-free problem data uploaded once by a native state, grouped as
   solprop's kernel contract groups it. The adapter fills it field by field (`lower` in
    solprop_native's backend); `validate` checks presence and extents before a state
    allocates anything. */
struct Problem {
  /** The state's extents and field family. */
  struct Dims {
    std::size_t n_omega{};
    std::size_t n_t{};
    std::size_t n_to{};
    std::size_t n_modes{1};
    std::size_t n_pol_state{1};
    std::size_t n_pol_transform{1};
    std::size_t n_nodes{1};
    std::size_t node_chunk{1};
    FieldFamily family{FieldFamily::field_real};
  };
  /** The frequency and time grids, their windows and masks. */
  struct Grid {
    std::vector<std::size_t> crop_index;
    std::vector<unsigned char> active_mask;
    std::vector<unsigned char> norm_mask;
    std::vector<double> omega_window;
    std::vector<double> omega;
    std::vector<double> power_weight;
    std::vector<double> time_window;
    std::vector<double> time_window_derivative;
    std::vector<Complex> carrier;
    double dt_fine{};
    double dt{};
  };
  /** The separable geometries' operator (free space and the hybrid planar kind) and a
      radial run's absorber and on-axis rows. */
  struct Spatial {
    std::size_t n_x{}, n_y{};
    std::vector<Complex> k_squared, k_squared_low, numerator, clamped;
    std::vector<double> kperp_sq, kperp_sq_low, frame, spectral_rate;
    // solprop kernel schema 45 (#231): the user's share of the spectral rate, read only
    // by `absorbed_power`; the operator takes the total.
    std::vector<double> user_rate;
    std::vector<double> absorber_rate;
    // solprop kernel schema 31 (SEP 22 phase 2): a transverse collar's power rate per
    // column, added to the field attenuation on active bins; zeros leave the operator.
    std::vector<double> kperp_rate;
    double evanescent_rate{};
    // solprop SEP 23 (free space only): the stored transverse axes k_x (n_x) and k_y
    // (n_y) in rad/m, FFT order, column ix + n_x iy, which the far-field statistics
    // read; and the slots of the problem's own far-field request, which size the band
    // partials at construction (zero for none). Empty and zero elsewhere.
    std::vector<double> k_x, k_y;
    std::size_t far_field_slots{};
    // solprop's hybrid planar kind (kernel schema 15): n_p slab channels in x on the
    // real (n_x, n_p) synthesis and (n_p, n_x) projection, times a raw DFT in y.
    // Columns are (p, k_y) with the channel fastest, n_modes = n_p * n_y, and nodes
    // (x, y) with x fastest, n_nodes = n_x * n_y; node_chunk holds whole y rows.
    // Row-major (n_p, n_omega) wall terms kappa_p^2 enter the separable operator, and
    // absorber_rate is then per column. Zero n_p and empty kappa elsewhere.
    std::size_t n_p{};
    std::vector<Complex> kappa_sq;
    // A radial run's spatial absorber and its on-axis statistics' rows.
    std::vector<double> absorber_matrix;
    std::vector<double> spectral_weights;
    std::vector<Complex> on_axis;
  };
  /** solprop kernel schema 35 (SEP 34): a non-separable spatial run's plane measure,
      the entrance's material squares (n_omega) and transverse squares (n_modes) whose
      root is solprop's `longitudinal_root`; `table` when the material square varies
      along z and is uploaded as a `TableKind::flux` table of n_omega values. A
      separable run's measure is its own operator's (these stay empty). `closure`
      allocates the magnetic closure's buffers at construction, for a request with a
      local flux statistic. */
  struct Flux {
    std::vector<Complex> k_squared, k_squared_low;
    std::vector<double> kperp_sq, kperp_sq_low;
    bool table{};
    bool closure{};
  };
  /** The Runge-Kutta tableau and the march's own choices. */
  struct Stepper {
    std::size_t n_stages{};
    std::vector<double> tableau_matrix;
    std::vector<double> tableau_nodes;
    std::vector<double> solution_weights;
    std::vector<double> error_weights;
    bool first_same_as_last{};
    bool weighted_controller{};
    bool modified_noise{};
    // solprop SEP 24 phase 3 (kernel schema 32): the low-storage RK4IP march. With it
    // no Runge-Kutta view is allocated (stages, accumulators, accept buffer, residual,
    // recycled stage, whole-step propagator, stage inverse); the accumulator register
    // is, the stage register unless it is elided (solprop's
    // `low_storage_stage_register`), and `low_storage_flows` flow volumes (0
    // recomputed, 1 a constant operator's table, 2 a varying operator's step pair).
    bool low_storage{};
    bool low_storage_stage_register{true};
    std::size_t low_storage_flows{};
    // Whether a separable route's first input crosses to the transverse scratch
    // through tiles of team scratch, as a device's does.
    bool low_storage_tiled_input{};
  };
  /** The linear operator's fixed terms, row-major (terms, columns, n_omega). solprop
     SEP 29 G1: Phi or the prefactor is the general z table when `phi_table`
      (`prefactor_table`); its terms are then `n_phi_terms` (`n_prefactor_terms`) slots
      that `table_evaluate` fills with the table's values at a trial's positions, and
      `phi_terms` (`prefactor_terms`) is empty. */
  /** The shapes the bindings received for the arrays whose layout counts beside them
      give, by field name: `validate` holds each to its counts (a vector keeps its
      elements alone). Empty for a problem built in C++. */
  using Shapes = std::map<std::string, std::vector<std::size_t>>;
  struct Linear {
    std::size_t n_phi_terms{};
    std::size_t n_prefactor_terms{};
    bool phi_table{};
    bool prefactor_table{};
    std::vector<Complex> phi_terms;
    std::vector<Complex> prefactor_terms;
    Shapes shapes;
  };
  /** The temporal absorber, applied in each stage or split around the step. */
  struct Absorber {
    bool in_stage{};
    bool split_step{};
    std::vector<double> fine;
    std::vector<double> coarse;
  };
  /** The modal synthesis and projection onto the nodes. */
  struct Transform {
    std::vector<double> synthesis_matrix;
    std::vector<double> projection_matrix;
    std::vector<double> projection_coarse_matrix;
    double synthesis{};
    double projection{};
  };
  /** The instantaneous Kerr response. */
  struct Kerr {
    bool nonlinear{};
    double coefficient{};
    double self_factor{};
    double conjugate_factor{};
    double harmonic_factor{};
    bool take_real{};
  };
  /** The field and envelope plasma columns and their rate tables. */
  struct Plasma {
    bool enabled{};
    bool envelope{};
    // solprop's private, test-only gated plasma benchmark: the field plasma's current
    // enters as w(t) times its running integral, added to the local polarisation,
    // instead of the held current's two collared spectra. Never set by a user run.
    bool gated{};
    bool stash{};
    // Channel-major 4 x 512 split angular rules and coarse-axis current factors.
    std::vector<double> envelope_rule_nodes, envelope_rule_weights,
        envelope_drude_factor;
    std::vector<Complex> envelope_loss_factor;
    double charge_mass_ratio{};
    double ionisation_potential{};
    double log_field_min{};
    double d_log_field{};
    double field_min{};
    double field_max{};
    double rate_max{};
    std::size_t rate_intervals{};
    // Row-major (4, rate_intervals).
    std::vector<double> rate_coefficients;
    std::size_t n_channels{1};
    // solprop SEP 18 (kernel schema 25): the field plasma column's running-integral
    // rule, one per problem and shared by every channel. Each running integral is the
    // cumulative trapezoid of the integrand filtered by the symmetric taps
    // (c_0, ..., c_K), centre first, with the window's ends closed by point
    // reflection, and the exposure floored at zero where the fraction is formed. The
    // trapezoid rule is `band == false` with the one identity tap, and runs the
    // version-24 column unchanged.
    bool band{};
    std::vector<double> taps{1.0};
    [[nodiscard]] std::size_t half_width() const noexcept {
      return taps.empty() ? 0 : taps.size() - 1;
    }
    // Channel-major rows: log_min, d_log, field_min, field_max, rate_max,
    // interval count, coefficient offset, charge/mass ratio, ionisation potential.
    std::vector<double> channel_parameters, channel_coefficients;
    Shapes shapes;
  };
  /** The delayed Raman response. */
  struct Raman {
    bool enabled{};
    bool thg{};
    bool tensor_capable{};
    std::size_t n_terms{};
    double fraction{};
    double epsilon0{};
    // Row-major (n_terms, 2 n_to) and (n_terms, 2).
    std::vector<double> undamped;
    std::vector<double> tensor_weights;
    std::size_t n_components{1};
    std::vector<double> row_component, component_fraction;
    Shapes shapes;
  };
  /** solprop's general modal route. Row-major: synthesis (map terms, nodes,
      components, modes, n_omega); projection (map terms, modes, nodes, components,
      n_omega); tensor (nodes, C, C, C, C); delayed tensor (responses, terms, C, C, C,
      C); delayed spectrum (responses, terms, n_to + 1); node response (nodes), the
      owning response plus one and zero for none. Empty otherwise. The map terms are
      map_intervals x map_n_deg Chebyshev coefficients along z, one for a constant
      guide; a moving guide's are combined per stage over
      TrialScalars::transform_weights. */
  struct General {
    bool enabled{};
    bool real_form{};
    std::size_t map_intervals{1}, map_n_deg{1};
    std::size_t components{}, nodes{}, node_chunk{};
    std::size_t responses{}, terms{};
    double pad_scale{}, epsilon0{};
    std::vector<Complex> synthesis, projection, frequency_factor, delayed_spectrum;
    std::vector<double> tensor, delayed_tensor;
    std::vector<std::size_t> node_response;
  };
  /** solprop's linear.coupled_flow. `slot` (n_omega) numbers the bins a section's
      generator is non-zero at 1, 2, ... in increasing frequency, zero elsewhere. With
      an explicit residual (terms > 0), row-major frozen blocks (sections, active,
      modes, modes) and residual terms (terms, active, modes, modes); both empty
      otherwise. The actions are loaded per step.

      The actions' native formation, all present or all empty: each section's
      eigenbasis of its Schur factor, row-major left `W` and right `Y` (sections,
      active, modes, modes) and eigenvalues (sections, active, modes); the route
      (sections, active), zero where a bin takes its eigenbasis and 1, 2, ...
      numbering in row-major order the bins formed by scaling and squaring, whose Schur
      factors `Z` and `T` are the rows of the fallback arrays (fallback, modes,
      modes). */
  struct Flow {
    bool enabled{};
    std::size_t active{}, sections{}, terms{};
    std::vector<std::size_t> slot;
    std::vector<Complex> frozen, residual_terms;
    std::vector<Complex> left, right, eigenvalues;
    std::vector<std::size_t> route;
    std::vector<Complex> fallback_vectors, fallback_triangular;
  };
  /** solprop SEP 6's frequency-dependent electronic response
      (`responses.dispersive_kerr`, kernel schema 24) on the fused node path, laid out
      by the adapter as solprop's `dispersive_terms_layout` lays it out
      (`dispersive.hpp`). The form is 0 (a field grid's full cubic), 1 (the
      number-conserving sector) or 2 (an envelope's full cubic). Row-major on the
      leaf's spectral layout of `bins` (n_to / 2 + 1 on a field grid, n_to on an
      envelope): the input multipliers (fields, bins), each output term's multiplier
      (terms, bins) with its product row and factor, the static coefficients and the
      polynomial's pair constants (channels, 2) per channel, the groups as offsets
      (groups + 1) into members (channels), each symmetric mixture's channels' `g^4`
      (channels, bins) and the input and output scales (bins). On the general modal
      route (SEP 6 Phase 3b) each group's key is its node region (groups), the
      polynomial forms one cube per distinct key, and every general node's region
      (general nodes) says where a channel acts; both are empty elsewhere, where every
      channel acts at every node. Empty otherwise. */
  struct Dispersive {
    bool enabled{};
    bool symmetric{};
    int form{};
    std::vector<double> inputs, term_multipliers, coefficients, pair_constants,
        quartics, input_scale, output_scale;
    std::vector<std::size_t> term_rows, term_channels, group_offsets, group_members,
        group_keys, node_region;
  };

  Dims dims;
  Grid grid;
  Spatial spatial;
  Flux flux;
  Stepper stepper;
  Linear linear;
  Absorber absorber;
  Transform transform;
  Kerr kerr;
  Plasma plasma;
  Raman raman;
  General general;
  Flow flow;
  Dispersive dispersive;
  // How the workspace's CPU operations thread (solprop SEP 46): set by the caller,
  // refused when unset (`require_cpu_execution`).
  CpuExecution cpu_execution{};
  // How the workspace plans its host transforms (solprop SEP 46 phase 3).
  FftPlanning fft_planning{};
  /** The set-once data a workspace applies once it has built its state (solprop SEP
      52 section 5.3): the stage tables, the z slots and tables, the flow guard, the
      controllers, the low-storage settings and the split collar's occupancy
      (`workspace/run_records.hpp`). Shared, so that a copy of the problem copies none
      of it. */
  std::shared_ptr<const Setup> setup;

  /** Refuse, naming the field, data whose presence or extents disagree with the
      problem's own flags and extents: the checks that need no execution space. */
  void validate() const;

  /** Free space or the hybrid planar kind: the separable operator and a y FFT. */
  [[nodiscard]] bool separable_geometry() const noexcept { return spatial.n_x != 0; }
  [[nodiscard]] bool slab_geometry() const noexcept { return spatial.n_p != 0; }
  [[nodiscard]] bool free_geometry() const noexcept {
    return spatial.n_x != 0 && spatial.n_p == 0;
  }
  [[nodiscard]] std::size_t n_columns() const noexcept {
    return dims.n_modes * dims.n_pol_state;
  }
  [[nodiscard]] std::size_t effective_node_pol() const noexcept {
    return dims.n_pol_transform * dims.n_pol_state;
  }
  [[nodiscard]] std::size_t raman_drives() const noexcept {
    if (effective_node_pol() == 2)
      for (std::size_t index = 1; index < raman.tensor_weights.size(); index += 2)
        if (raman.tensor_weights[index] != 0.0)
          return 3;
    return 1;
  }
  [[nodiscard]] std::size_t raman_kernels() const noexcept {
    return raman_drives() == 3 ? 2 : 1;
  }
};

// The problem's groups as the seam fills them, one field table a group (solprop SEP 52
// section 5.3). The relations a table states within its group are its schema's; the
// extents that a count in another group gives are `Problem::validate`'s.
[[nodiscard]] constexpr auto describe(const Problem::Dims* /*unused*/) {
  using G = Problem::Dims;
  return fields::table<G>(
      "Problem.Dims", fields::value("n_omega", &G::n_omega),
      fields::value("n_t", &G::n_t), fields::value("n_to", &G::n_to),
      fields::value("n_modes", &G::n_modes),
      fields::value("n_pol_state", &G::n_pol_state),
      fields::value("n_pol_transform", &G::n_pol_transform),
      fields::value("n_nodes", &G::n_nodes),
      fields::value("node_chunk", &G::node_chunk), fields::value("family", &G::family));
}

[[nodiscard]] constexpr auto describe(const Problem::Grid* /*unused*/) {
  using G = Problem::Grid;
  return fields::table<G>(
      "Problem.Grid", fields::array("crop_index", &G::crop_index),
      fields::array("active_mask", &G::active_mask),
      fields::array("norm_mask", &G::norm_mask),
      fields::array("omega_window", &G::omega_window),
      fields::array("omega", &G::omega),
      fields::array("power_weight", &G::power_weight),
      fields::array("time_window", &G::time_window),
      fields::array("time_window_derivative", &G::time_window_derivative),
      fields::array("carrier", &G::carrier), fields::value("dt_fine", &G::dt_fine),
      fields::value("dt", &G::dt));
}

[[nodiscard]] constexpr auto describe(const Problem::Spatial* /*unused*/) {
  using G = Problem::Spatial;
  return fields::table<G>(
      "Problem.Spatial", fields::value("n_x", &G::n_x), fields::value("n_y", &G::n_y),
      fields::array("k_squared", &G::k_squared),
      fields::array("k_squared_low", &G::k_squared_low),
      fields::array("numerator", &G::numerator), fields::array("clamped", &G::clamped),
      fields::array("kperp_sq", &G::kperp_sq),
      fields::array("kperp_sq_low", &G::kperp_sq_low),
      fields::array("frame", &G::frame),
      fields::array("spectral_rate", &G::spectral_rate),
      fields::array("user_rate", &G::user_rate),
      fields::array("absorber_rate", &G::absorber_rate),
      fields::array("kperp_rate", &G::kperp_rate),
      fields::value("evanescent_rate", &G::evanescent_rate),
      fields::array("k_x", &G::k_x), fields::array("k_y", &G::k_y),
      fields::value("far_field_slots", &G::far_field_slots),
      fields::value("n_p", &G::n_p), fields::array("kappa_sq", &G::kappa_sq),
      fields::array("absorber_matrix", &G::absorber_matrix),
      fields::array("spectral_weights", &G::spectral_weights),
      fields::array("on_axis", &G::on_axis));
}

[[nodiscard]] constexpr auto describe(const Problem::Flux* /*unused*/) {
  using G = Problem::Flux;
  return fields::table<G>("Problem.Flux", fields::array("k_squared", &G::k_squared),
                          fields::array("k_squared_low", &G::k_squared_low),
                          fields::array("kperp_sq", &G::kperp_sq),
                          fields::array("kperp_sq_low", &G::kperp_sq_low),
                          fields::value("table", &G::table),
                          fields::value("closure", &G::closure));
}

[[nodiscard]] constexpr auto describe(const Problem::Stepper* /*unused*/) {
  using G = Problem::Stepper;
  return fields::table<G>(
      "Problem.Stepper", fields::value("n_stages", &G::n_stages),
      fields::array("tableau_matrix", &G::tableau_matrix),
      fields::array("tableau_nodes", &G::tableau_nodes),
      fields::array("solution_weights", &G::solution_weights),
      fields::array("error_weights", &G::error_weights),
      fields::value("first_same_as_last", &G::first_same_as_last),
      fields::value("weighted_controller", &G::weighted_controller),
      fields::value("modified_noise", &G::modified_noise),
      fields::value("low_storage", &G::low_storage),
      fields::value("low_storage_stage_register", &G::low_storage_stage_register),
      fields::value("low_storage_flows", &G::low_storage_flows),
      fields::value("low_storage_tiled_input", &G::low_storage_tiled_input));
}

[[nodiscard]] constexpr auto describe(const Problem::Linear* /*unused*/) {
  using G = Problem::Linear;
  return fields::table<G>("Problem.Linear",
                          fields::value("n_phi_terms", &G::n_phi_terms),
                          fields::value("n_prefactor_terms", &G::n_prefactor_terms),
                          fields::value("phi_table", &G::phi_table),
                          fields::value("prefactor_table", &G::prefactor_table),
                          fields::array("phi_terms", &G::phi_terms),
                          fields::array("prefactor_terms", &G::prefactor_terms));
}

[[nodiscard]] constexpr auto describe(const Problem::Absorber* /*unused*/) {
  using G = Problem::Absorber;
  return fields::table<G>("Problem.Absorber", fields::value("in_stage", &G::in_stage),
                          fields::value("split_step", &G::split_step),
                          fields::array("fine", &G::fine),
                          fields::array("coarse", &G::coarse));
}

[[nodiscard]] constexpr auto describe(const Problem::Transform* /*unused*/) {
  using G = Problem::Transform;
  return fields::table<G>(
      "Problem.Transform", fields::array("synthesis_matrix", &G::synthesis_matrix),
      fields::array("projection_matrix", &G::projection_matrix),
      fields::array("projection_coarse_matrix", &G::projection_coarse_matrix),
      fields::value("synthesis", &G::synthesis),
      fields::value("projection", &G::projection));
}

[[nodiscard]] constexpr auto describe(const Problem::Kerr* /*unused*/) {
  using G = Problem::Kerr;
  return fields::table<G>("Problem.Kerr", fields::value("nonlinear", &G::nonlinear),
                          fields::value("coefficient", &G::coefficient),
                          fields::value("self_factor", &G::self_factor),
                          fields::value("conjugate_factor", &G::conjugate_factor),
                          fields::value("harmonic_factor", &G::harmonic_factor),
                          fields::value("take_real", &G::take_real));
}

[[nodiscard]] constexpr auto describe(const Problem::Plasma* /*unused*/) {
  using G = Problem::Plasma;
  using E = fields::Extent;
  return fields::table<G>(
      "Problem.Plasma", fields::value("enabled", &G::enabled),
      fields::value("envelope", &G::envelope), fields::value("gated", &G::gated),
      fields::value("stash", &G::stash),
      fields::array("envelope_rule_nodes", &G::envelope_rule_nodes, E::of("n_channels"),
                    E::fixed(4), E::fixed(512)),
      fields::array("envelope_rule_weights", &G::envelope_rule_weights,
                    E::of("n_channels"), E::fixed(4), E::fixed(512)),
      fields::array("envelope_drude_factor", &G::envelope_drude_factor, E::any()),
      fields::array("envelope_loss_factor", &G::envelope_loss_factor, E::any()),
      fields::value("charge_mass_ratio", &G::charge_mass_ratio),
      fields::value("ionisation_potential", &G::ionisation_potential),
      fields::value("log_field_min", &G::log_field_min),
      fields::value("d_log_field", &G::d_log_field),
      fields::value("field_min", &G::field_min),
      fields::value("field_max", &G::field_max),
      fields::value("rate_max", &G::rate_max),
      fields::value("rate_intervals", &G::rate_intervals),
      fields::array("rate_coefficients", &G::rate_coefficients, E::fixed(4),
                    E::of("rate_intervals")),
      fields::value("n_channels", &G::n_channels), fields::value("band", &G::band),
      fields::array("taps", &G::taps, E::any()),
      fields::array("channel_parameters", &G::channel_parameters),
      fields::array("channel_coefficients", &G::channel_coefficients));
}

[[nodiscard]] constexpr auto describe(const Problem::Raman* /*unused*/) {
  using G = Problem::Raman;
  using E = fields::Extent;
  return fields::table<G>(
      "Problem.Raman", fields::value("enabled", &G::enabled),
      fields::value("thg", &G::thg),
      fields::value("tensor_capable", &G::tensor_capable),
      fields::value("n_terms", &G::n_terms), fields::value("fraction", &G::fraction),
      fields::value("epsilon0", &G::epsilon0),
      fields::array("undamped", &G::undamped, E::of("n_terms"), E::any()),
      fields::array("tensor_weights", &G::tensor_weights, E::of("n_terms"),
                    E::fixed(2)),
      fields::value("n_components", &G::n_components),
      fields::array("row_component", &G::row_component),
      fields::array("component_fraction", &G::component_fraction));
}

[[nodiscard]] constexpr auto describe(const Problem::General* /*unused*/) {
  using G = Problem::General;
  return fields::table<G>(
      "Problem.General", fields::value("enabled", &G::enabled),
      fields::value("real_form", &G::real_form),
      fields::value("map_intervals", &G::map_intervals),
      fields::value("map_n_deg", &G::map_n_deg),
      fields::value("components", &G::components), fields::value("nodes", &G::nodes),
      fields::value("node_chunk", &G::node_chunk),
      fields::value("responses", &G::responses), fields::value("terms", &G::terms),
      fields::value("pad_scale", &G::pad_scale),
      fields::value("epsilon0", &G::epsilon0),
      fields::array("synthesis", &G::synthesis),
      fields::array("projection", &G::projection),
      fields::array("frequency_factor", &G::frequency_factor),
      fields::array("delayed_spectrum", &G::delayed_spectrum),
      fields::array("tensor", &G::tensor),
      fields::array("delayed_tensor", &G::delayed_tensor),
      fields::array("node_response", &G::node_response));
}

[[nodiscard]] constexpr auto describe(const Problem::Flow* /*unused*/) {
  using G = Problem::Flow;
  return fields::table<G>(
      "Problem.Flow", fields::value("enabled", &G::enabled),
      fields::value("active", &G::active), fields::value("sections", &G::sections),
      fields::value("terms", &G::terms), fields::array("slot", &G::slot),
      fields::array("frozen", &G::frozen),
      fields::array("residual_terms", &G::residual_terms),
      fields::array("left", &G::left), fields::array("right", &G::right),
      fields::array("eigenvalues", &G::eigenvalues), fields::array("route", &G::route),
      fields::array("fallback_vectors", &G::fallback_vectors),
      fields::array("fallback_triangular", &G::fallback_triangular));
}

[[nodiscard]] constexpr auto describe(const Problem::Dispersive* /*unused*/) {
  using G = Problem::Dispersive;
  return fields::table<G>("Problem.Dispersive", fields::value("enabled", &G::enabled),
                          fields::value("symmetric", &G::symmetric),
                          fields::value("form", &G::form),
                          fields::array("inputs", &G::inputs),
                          fields::array("term_multipliers", &G::term_multipliers),
                          fields::array("coefficients", &G::coefficients),
                          fields::array("pair_constants", &G::pair_constants),
                          fields::array("quartics", &G::quartics),
                          fields::array("input_scale", &G::input_scale),
                          fields::array("output_scale", &G::output_scale),
                          fields::array("term_rows", &G::term_rows),
                          fields::array("term_channels", &G::term_channels),
                          fields::array("group_offsets", &G::group_offsets),
                          fields::array("group_members", &G::group_members),
                          fields::array("group_keys", &G::group_keys),
                          fields::array("node_region", &G::node_region));
}

/** Whether `problem` runs on a compact CPU team (`CpuExecution::compact_threads`): one
    column on one node, fewer than its record's `compact_max_fine_samples` fine
    samples, and neither the general modal route, a coupled flow nor a separable
    geometry. */
[[nodiscard]] bool compact_problem(const Problem& problem) noexcept;

} // namespace solprop_native
