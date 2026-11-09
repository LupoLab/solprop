// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

/*
 * A statistics evaluation lowered once, at preparation (solprop SEP 47 section 6.4):
 * which statistics, in which order, and every z-dependent input as the representation
 * the propagation itself carries -- solprop's piecewise Chebyshev tables of z
 * (`solprop.translate.chebyshev.PiecewiseChebyshev`) for the plasma density, the
 * preionisation of each channel, the species densities, the taper, the effective
 * area and the modal power weight, and the basis metric's square root formed from
 * the metric's own values on each interval. Nothing is fitted. `evaluate` reads the
 * resident state through the native statistics passes and finishes every statistic
 * natively: the half-maximum widths, the species reduction and the plasma
 * integration ratios are formed here, with no Python finishing. Python names the
 * packed values and assembles the schema.
 *
 * The one native implementation of each statistic: a native session takes its plane
 * statistics and its periodic rows through it, and `NativeBackend.stats` its plane
 * statistics, so coincident requests give the same bits.
 *
 * Kokkos-free: the native passes are behind `GeneralState`, declared elsewhere.
 */
namespace solprop_native {

class GeneralState;

namespace statistics {

/** The route a plan reads: a modal or mode-averaged state, a spatial state, or a
    general modal basis, whose statistics are functionals of `a.H W a`. */
enum class Route : int { modal = 0, spatial = 1, general = 2 };

/** Each statistic a plan can emit, by its solprop catalogue name, and the per-node
    channel populations (`node_populations`, final fraction and peak rate at each
    node and channel), which a host axis statistic reuses (SEP 19). The values are the
    Python adapter's (`solprop_native._statistics.STATISTIC_IDS`). */
enum class Stat : int {
  omega0 = 0,
  energy = 1,
  energy_window = 2,
  peak_power = 3,
  peak_power_window = 4,
  peak_power_total = 5,
  fwhm_t = 6,
  fwhm_t_outer = 7,
  electron_density = 8,
  peak_ionisation_rate = 9,
  peak_intensity = 10,
  quadrature_error = 11,
  species_ionisation = 12,
  species_electron_density = 13,
  species_peak_ionisation_rate = 14,
  species_density = 15,
  species_ionisation_energy = 16,
  plasma_undershoot = 17,
  plasma_work_defect = 18,
  plasma_boundary_level = 19,
  physical_energy = 20,
  spectral_flux = 21,
  analytic_flux = 22,
  on_axis_intensity = 23,
  on_axis_spectrum = 24,
  absorbed_power = 25,
  beam_spectrum = 26,
  region_spectrum = 27,
  aperture_spectrum = 28,
  aperture_on_axis_spectrum = 29,
  node_populations = 30,
};

/** The number of `Stat` values. */
inline constexpr int stat_count = 31;

/** A piecewise Chebyshev table of z: on interval `k` of `edges`, `coefficients[(k *
    n_terms + i) * n_values + j]` is value `j`'s coefficient of degree `i`. A position
    outside `[lower, upper]` (the span widened by solprop's edge slack) is refused. */
template <class Value> struct Table {
  std::vector<double> edges;
  std::vector<Value> coefficients;
  std::size_t n_terms = 0;
  std::size_t n_values = 0;
  double lower = 0.0;
  double upper = 0.0;
  [[nodiscard]] bool present() const noexcept { return edges.size() >= 2; }
  /** Check the extents; throws `std::invalid_argument` naming `name`. */
  void validate(const char* name) const;
  /** The values at `z`, solprop's `PiecewiseChebyshev.__call__` bit for bit. */
  [[nodiscard]] std::vector<Value> at(double z) const;
  /** The one value of a scalar table at `z`. */
  [[nodiscard]] Value scalar(double z) const;
};

using RealTable = Table<double>;
using ComplexTable = Table<std::complex<double>>;

/** How the modal power weight varies with z: absent (unit or spatial), constant, a
    weighted sum of fixed arrays (`ParameterExpansion`) or a full table. */
enum class Weight : int { none = 0, constant = 1, expansion = 2, table = 3 };

/** The physical transverse weights of the species inventory
    (`solprop.backends.populations.population_statistics`): the spatial quadrature,
    the effective area at z, or the node weights over the taper's square. */
enum class Transverse : int { none = 0, spatial = 1, aeff = 2, nodes = 3 };

/** One evaluation's fixed arguments, with its outputs in order. */
struct Plan {
  Route route = Route::modal;
  std::vector<Stat> outputs;
  // The state's extents: modes, polarisations, time samples, frequency bins, nodes,
  // channels and species.
  std::size_t n_modes = 0;
  std::size_t n_pol = 0;
  std::size_t n_t = 0;
  std::size_t n_omega = 0;
  std::size_t n_nodes = 0;
  std::size_t n_channels = 0;
  std::size_t n_species = 0;
  /** The energy windows' edges, pairs, rad/s. */
  std::vector<double> windows;
  /** `epsilon_0 c / 2`, W/(V/m)^2. */
  double intensity_factor = 0.0;
  /** The time axis the widths read, s. */
  std::vector<double> t;
  // The modal power weight: the constant row, or an expansion's fixed terms
  // `(n_terms, n_values)` with its scalar weights' table, or the full table.
  Weight weight_kind = Weight::none;
  std::vector<double> weight;
  RealTable weight_table;
  // The on-axis synthesis at the reference position, rows times modes, which the
  // taper rescales (`StepDriver._on_axis_synthesis`), and the taper's table.
  std::vector<double> on_axis;
  RealTable taper;
  // A single population's density and preionisation, and every channel's
  // preionisation.
  RealTable plasma_density;
  RealTable plasma_preionisation;
  std::vector<RealTable> channel_preionisation;
  bool envelope_plasma = false;
  // The species reduction: each species' density, each channel's species and
  // ionisation potential, J, and the transverse weights.
  std::vector<RealTable> species_density;
  std::vector<std::size_t> channel_species;
  std::vector<double> potentials;
  Transverse transverse = Transverse::none;
  std::vector<double> node_weights;
  RealTable aeff;
  // A spatial route: whether `on_axis_spectrum` comes from the far-field pass (a
  // Cartesian or line grid), and the far-field tables, `(n_apertures,
  // far_field_aperture_constants + n_omega)`
  // and `(n_regions, 4)`, with their exponent; `n_absorbed` channels of absorbed
  // power.
  bool far_field_axis = false;
  std::vector<double> apertures;
  std::size_t n_apertures = 0;
  int exponent = 2;
  std::vector<double> regions;
  std::size_t n_regions = 0;
  std::size_t n_absorbed = 0;
  // A general modal basis: the square root of its flux metric, `(n_omega, n_modes,
  // n_modes)` values a table of z, absent for the identity; the spectral density's
  // prefactor, s^2.
  ComplexTable root;
  double spectral_prefactor = 0.0;

  [[nodiscard]] bool empty() const noexcept { return outputs.empty(); }
  /** Check every extent before anything is evaluated. */
  void validate() const;
  /** The number of values `stat` packs. */
  [[nodiscard]] std::size_t width(Stat stat) const;
  /** The number of values an evaluation returns. */
  [[nodiscard]] std::size_t packed_size() const;
};

/** Evaluate `plan` on `state`'s committed field at `z`: each output in order,
    finished. */
[[nodiscard]] std::vector<double> evaluate(GeneralState& state, const Plan& plan,
                                           double z);

/** solprop's half-maximum width of a power trace (`_half_width` of the native
    adapter, `solprop.stats._half_max_range`): the outer or the central crossings,
    NaN when undefined. Strict, as solprop's NumPy rounds it. */
[[nodiscard]] double half_width(const double* t, const double* power, std::size_t n,
                                bool outer);

/** `a b + c` as session_statistics.cpp rounds it (solprop #135's witness); never
    inlined, as `session::strict_probe`. */
[[nodiscard, gnu::noinline]] double strict_probe(double a, double b, double c);

} // namespace statistics
} // namespace solprop_native
