// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/statistics_plan.hpp"

#include "solprop_native/chebyshev.hpp"
#include "solprop_native/core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// The finishing arithmetic here -- the half-maximum crossings, the species reduction
// and its running sums, the integration ratios, the tables' Clenshaw recursion --
// rounds as solprop's NumPy does, so this file takes no contraction (CMake), and the
// optimize attribute keeps GCC from inlining it into a contracting caller.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace solprop_native::statistics {

namespace {

constexpr double nan = std::numeric_limits<double>::quiet_NaN();

// `GeneralState::statistics`' flags (core.hpp `StatisticsFlag`).
constexpr std::uint64_t temporal_omega0 = statistics_omega0;
constexpr std::uint64_t temporal_energy = statistics_energy;
constexpr std::uint64_t temporal_energy_window = statistics_energy_window;
constexpr std::uint64_t temporal_peak = statistics_peak_power;
constexpr std::uint64_t temporal_peak_window = statistics_peak_power_window;
constexpr std::uint64_t temporal_peak_total = statistics_peak_power_total;
constexpr std::uint64_t temporal_fwhm = statistics_fwhm;
constexpr std::uint64_t temporal_density = statistics_electron_density;
constexpr std::uint64_t temporal_rate = statistics_peak_ionisation_rate;
constexpr std::uint64_t temporal_intensity = statistics_peak_intensity;
constexpr std::uint64_t temporal_quadrature = statistics_quadrature_error;
constexpr std::uint64_t temporal_populations = statistics_populations;
constexpr std::uint64_t temporal_integration = statistics_plasma_integration;
/** The parts of the plasma integration diagnostics, packed last. */
constexpr std::size_t integration_parts = 6;

// `GeneralState::spatial_statistics`' scalar flags.
constexpr std::uint64_t spatial_energy = 1;
constexpr std::uint64_t spatial_axis = 2;
constexpr std::uint64_t spatial_peak = 4;
constexpr std::uint64_t spatial_spectrum = 8;

// `GeneralState::aperture_statistics`' flags (far_field.hpp), packed in this order.
constexpr std::uint64_t far_beam = 1ULL << 5U;
constexpr std::uint64_t far_absorbed = 1ULL << 7U;
constexpr std::uint64_t far_region = 1ULL << 8U;
constexpr std::uint64_t far_aperture = 1ULL << 9U;
constexpr std::uint64_t far_aperture_axis = 1ULL << 10U;
constexpr std::uint64_t far_axis = 1ULL << 11U;

/** Which statistics a plan emits. */
class Wanted {
public:
  explicit Wanted(const std::vector<Stat>& outputs) {
    for (const Stat stat : outputs)
      set_[static_cast<std::size_t>(stat)] = true;
  }
  bool operator()(const Stat stat) const {
    return set_[static_cast<std::size_t>(stat)];
  }
  bool any(std::initializer_list<Stat> stats) const {
    return std::any_of(stats.begin(), stats.end(),
                       [this](const Stat stat) { return (*this)(stat); });
  }

private:
  std::array<bool, stat_count> set_{};
};

/** Reads a native pass's packed values in their order. */
class Cursor {
public:
  explicit Cursor(const std::vector<double>& values) : values_(values) {}
  std::vector<double> take(const std::size_t count) {
    if (offset_ + count > values_.size())
      throw std::logic_error("a native statistics pass packed fewer values than asked");
    std::vector<double> part(values_.begin() + static_cast<std::ptrdiff_t>(offset_),
                             values_.begin() +
                                 static_cast<std::ptrdiff_t>(offset_ + count));
    offset_ += count;
    return part;
  }
  void finish() const {
    if (offset_ != values_.size())
      throw std::logic_error("a native statistics pass packed more values than asked");
  }

private:
  const std::vector<double>& values_;
  std::size_t offset_ = 0;
};

using Results = std::array<std::vector<double>, stat_count>;

void put(Results& results, const Stat stat, std::vector<double> values) {
  results[static_cast<std::size_t>(stat)] = std::move(values);
}

/** The modal power weight at `z`: the constant row, the expansion's
    `numpy.tensordot(weights(z), terms, axes=1)` or the table's value. */
std::vector<double> power_weight(const Plan& plan, const double z) {
  switch (plan.weight_kind) {
  case Weight::constant:
    return plan.weight;
  case Weight::expansion: {
    const auto scalars = plan.weight_table.at(z);
    const std::size_t n_values = plan.weight.size() / scalars.size();
    std::vector<double> value(n_values, 0.0);
    for (std::size_t term = 0; term < scalars.size(); ++term) {
      const double scalar = scalars[term];
      const double* row = plan.weight.data() + term * n_values;
      for (std::size_t index = 0; index < n_values; ++index)
        value[index] += scalar * row[index];
    }
    return value;
  }
  case Weight::table:
    return plan.weight_table.at(z);
  case Weight::none:
    break;
  }
  return {};
}

/**
 * solprop's species reduction (`population_statistics`, `species_statistics`): the
 * channels' final fractions and peak rates at every node mapped onto the mixture's
 * species, reduced at the node of largest total electron density, with the inventory
 * over the physical transverse weights. Each sum runs in NumPy's order.
 */
void species(const Plan& plan, const double z, const double taper,
             const std::vector<double>& populations, const Wanted& wanted,
             Results& results) {
  const std::size_t n_species = plan.n_species;
  const std::size_t n_nodes = plan.n_nodes;
  const std::size_t n_channels = plan.n_channels;
  std::vector<double> densities(n_species);
  for (std::size_t s = 0; s < n_species; ++s)
    densities[s] = plan.species_density[s].scalar(z);
  std::vector<double> fractions(n_species * n_nodes, 0.0);
  std::vector<double> peaks(n_species, 0.0);
  std::vector<double> potentials(n_species, 0.0);
  for (std::size_t c = 0; c < n_channels; ++c) {
    const std::size_t s = plan.channel_species[c];
    double peak = populations[2 * c + 1];
    for (std::size_t node = 0; node < n_nodes; ++node) {
      fractions[s * n_nodes + node] = populations[2 * (node * n_channels + c)];
      const double rate = populations[2 * (node * n_channels + c) + 1];
      // `numpy.max`, which a NaN propagates.
      if (rate > peak || std::isnan(rate))
        peak = std::isnan(peak) ? peak : rate;
    }
    peaks[s] = peak;
    potentials[s] = plan.potentials[c];
  }
  std::vector<double> weights;
  switch (plan.transverse) {
  case Transverse::spatial:
    weights = plan.node_weights;
    break;
  case Transverse::aeff:
    weights = {plan.aeff.scalar(z)};
    break;
  case Transverse::nodes: {
    const double square = taper * taper;
    weights.resize(plan.node_weights.size());
    for (std::size_t node = 0; node < weights.size(); ++node)
      weights[node] = plan.node_weights[node] / square;
    break;
  }
  case Transverse::none:
    throw std::invalid_argument(
        "species statistics require physical transverse node weights");
  }
  if (weights.size() != n_nodes)
    throw std::invalid_argument("species statistics require one weight per node");
  const auto valid = [](const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(), [](const double value) {
      return std::isfinite(value) && value >= 0.0;
    });
  };
  if (fractions.empty() || !valid(densities) || !valid(fractions) || !valid(peaks) ||
      !valid(potentials) || !valid(weights) ||
      std::any_of(fractions.begin(), fractions.end(),
                  [](const double value) { return value > 1.0; }))
    throw std::invalid_argument("species statistics require finite non-negative "
                                "populations and fractions in [0, 1]");
  std::vector<double> electrons(n_species * n_nodes);
  for (std::size_t s = 0; s < n_species; ++s)
    for (std::size_t node = 0; node < n_nodes; ++node)
      electrons[s * n_nodes + node] = densities[s] * fractions[s * n_nodes + node];
  std::vector<double> total(n_nodes, 0.0);
  for (std::size_t s = 0; s < n_species; ++s)
    for (std::size_t node = 0; node < n_nodes; ++node)
      total[node] += electrons[s * n_nodes + node];
  // `numpy.argmax`: the first of the largest.
  std::size_t densest = 0;
  for (std::size_t node = 1; node < n_nodes; ++node)
    if (total[node] > total[densest])
      densest = node;
  if (wanted(Stat::species_ionisation))
    put(results, Stat::species_ionisation, fractions);
  if (wanted(Stat::species_electron_density)) {
    std::vector<double> column(n_species);
    for (std::size_t s = 0; s < n_species; ++s)
      column[s] = electrons[s * n_nodes + densest];
    put(results, Stat::species_electron_density, std::move(column));
  }
  if (wanted(Stat::species_peak_ionisation_rate))
    put(results, Stat::species_peak_ionisation_rate, peaks);
  if (wanted(Stat::species_density))
    put(results, Stat::species_density, densities);
  if (wanted(Stat::species_ionisation_energy)) {
    // `numpy.add.accumulate(weights * row * potential)[-1]`.
    std::vector<double> energy(n_species);
    for (std::size_t s = 0; s < n_species; ++s) {
      double sum = 0.0;
      for (std::size_t node = 0; node < n_nodes; ++node) {
        const double product = weights[node] * electrons[s * n_nodes + node];
        const double term = product * potentials[s];
        sum = node == 0 ? term : sum + term;
      }
      energy[s] = sum;
    }
    put(results, Stat::species_ionisation_energy, std::move(energy));
  }
  if (wanted(Stat::electron_density))
    put(results, Stat::electron_density, {total[densest]});
  if (wanted(Stat::peak_ionisation_rate)) {
    double rate = peaks[0];
    for (std::size_t s = 1; s < n_species; ++s)
      rate = peaks[s] > rate ? peaks[s] : rate;
    if (plan.envelope_plasma && n_channels == 1) {
      // The single population's node selection, on both grids.
      std::size_t node_of = 0;
      for (std::size_t node = 1; node < n_nodes; ++node)
        if (populations[2 * (node * n_channels)] >
            populations[2 * (node_of * n_channels)])
          node_of = node;
      rate = populations[2 * (node_of * n_channels) + 1];
    }
    put(results, Stat::peak_ionisation_rate, {rate});
  }
}

/** The state's temporal and response statistics on a modal or spatial route. */
void temporal(GeneralState& state, const Plan& plan, const double z, const double taper,
              const Wanted& wanted, Results& results) {
  const bool modal = plan.route == Route::modal;
  const bool population_route = plan.n_channels > 1 || plan.envelope_plasma;
  const bool reduce =
      wanted.any({Stat::species_ionisation, Stat::species_electron_density,
                  Stat::species_peak_ionisation_rate, Stat::species_density,
                  Stat::species_ionisation_energy}) ||
      (population_route &&
       wanted.any({Stat::electron_density, Stat::peak_ionisation_rate}));
  std::uint64_t flags = 0;
  if (modal) {
    if (wanted(Stat::omega0))
      flags |= temporal_omega0;
    if (wanted(Stat::energy))
      flags |= temporal_energy;
    if (wanted(Stat::energy_window))
      flags |= temporal_energy_window;
    if (wanted(Stat::peak_power))
      flags |= temporal_peak;
    if (wanted(Stat::peak_power_window))
      flags |= temporal_peak_window;
    if (wanted(Stat::peak_power_total))
      flags |= temporal_peak_total;
    if (wanted.any({Stat::fwhm_t, Stat::fwhm_t_outer}))
      flags |= temporal_fwhm;
    if (wanted(Stat::peak_intensity))
      flags |= temporal_intensity;
  }
  if (wanted(Stat::quadrature_error))
    flags |= temporal_quadrature;
  if (wanted.any({Stat::plasma_undershoot, Stat::plasma_work_defect,
                  Stat::plasma_boundary_level}))
    flags |= temporal_integration;
  if (!population_route) {
    if (wanted(Stat::electron_density))
      flags |= temporal_density;
    if (wanted(Stat::peak_ionisation_rate))
      flags |= temporal_rate;
  }
  const bool populations = reduce && plan.n_channels > 0;
  if (populations)
    flags |= temporal_populations;
  if (flags == 0 && !reduce)
    return;
  std::vector<double> packed;
  if (flags != 0) {
    constexpr std::uint64_t weighted =
        temporal_omega0 | temporal_energy | temporal_energy_window | temporal_peak |
        temporal_peak_window | temporal_peak_total | temporal_fwhm | temporal_intensity;
    const auto weight =
        (flags & weighted) != 0U ? power_weight(plan, z) : std::vector<double>{};
    std::vector<double> on_axis;
    if ((flags & temporal_intensity) != 0U) {
      // `StepDriver._on_axis_synthesis`: the matrix at the reference position times
      // the taper at z.
      on_axis = plan.on_axis;
      if (plan.taper.present())
        for (double& value : on_axis)
          value *= taper;
    }
    const double density =
        plan.plasma_density.present() ? plan.plasma_density.scalar(z) : 0.0;
    const double preionisation =
        plan.plasma_preionisation.present() ? plan.plasma_preionisation.scalar(z) : 0.0;
    std::vector<double> channels(plan.channel_preionisation.size());
    for (std::size_t c = 0; c < channels.size(); ++c)
      channels[c] = plan.channel_preionisation[c].scalar(z);
    packed = state.statistics(flags, plan.windows, on_axis, density, taper,
                              preionisation, plan.intensity_factor, channels, weight);
  }
  Cursor cursor(packed);
  const std::size_t columns = plan.n_modes * plan.n_pol;
  const std::size_t n_windows = plan.windows.size() / 2;
  if ((flags & temporal_omega0) != 0U)
    put(results, Stat::omega0, cursor.take(columns));
  if ((flags & temporal_energy) != 0U)
    put(results, Stat::energy, cursor.take(columns));
  if ((flags & temporal_energy_window) != 0U)
    put(results, Stat::energy_window, cursor.take(n_windows * columns));
  if ((flags & temporal_peak) != 0U)
    put(results, Stat::peak_power, cursor.take(plan.n_modes));
  if ((flags & temporal_peak_window) != 0U)
    put(results, Stat::peak_power_window, cursor.take(n_windows * plan.n_modes));
  if ((flags & temporal_peak_total) != 0U)
    put(results, Stat::peak_power_total, cursor.take(1));
  if ((flags & temporal_density) != 0U)
    put(results, Stat::electron_density, cursor.take(1));
  if ((flags & temporal_rate) != 0U)
    put(results, Stat::peak_ionisation_rate, cursor.take(1));
  if ((flags & temporal_intensity) != 0U)
    put(results, Stat::peak_intensity, cursor.take(1));
  if ((flags & temporal_quadrature) != 0U)
    put(results, Stat::quadrature_error, cursor.take(1));
  if ((flags & temporal_fwhm) != 0U) {
    const auto power = cursor.take(plan.n_t);
    if (wanted(Stat::fwhm_t))
      put(results, Stat::fwhm_t,
          {half_width(plan.t.data(), power.data(), plan.n_t, false)});
    if (wanted(Stat::fwhm_t_outer))
      put(results, Stat::fwhm_t_outer,
          {half_width(plan.t.data(), power.data(), plan.n_t, true)});
  }
  std::vector<double> rows;
  if (populations)
    rows = cursor.take(2 * plan.n_nodes * plan.n_channels);
  if ((flags & temporal_integration) != 0U) {
    // `plasma_integration_statistics`: each ratio zero where its denominator is.
    const auto parts = cursor.take(integration_parts);
    const auto ratio = [](const double top, const double bottom) {
      return bottom > 0.0 ? top / bottom : 0.0;
    };
    put(results, Stat::plasma_undershoot, {ratio(parts[0], parts[1])});
    put(results, Stat::plasma_work_defect, {ratio(parts[2], parts[3])});
    put(results, Stat::plasma_boundary_level, {ratio(parts[4], parts[5])});
  }
  cursor.finish();
  if (reduce)
    species(plan, z, taper, rows, wanted, results);
}

/** A spatial route's statistics of the medium's axial flux and its far field. */
void spatial(GeneralState& state, const Plan& plan, const double z,
             const Wanted& wanted, Results& results) {
  std::uint64_t flags = 0;
  if (wanted(Stat::energy))
    flags |= spatial_energy;
  if (wanted(Stat::on_axis_intensity))
    flags |= spatial_axis;
  if (wanted(Stat::peak_intensity))
    flags |= spatial_peak;
  if (wanted(Stat::on_axis_spectrum) && !plan.far_field_axis)
    flags |= spatial_spectrum;
  if (flags != 0U) {
    const auto packed = state.spatial_statistics(flags, plan.intensity_factor, z);
    Cursor cursor(packed);
    if ((flags & spatial_energy) != 0U)
      put(results, Stat::energy, cursor.take(1));
    if ((flags & spatial_axis) != 0U)
      put(results, Stat::on_axis_intensity, cursor.take(1));
    if ((flags & spatial_peak) != 0U)
      put(results, Stat::peak_intensity, cursor.take(1));
    if ((flags & spatial_spectrum) != 0U)
      put(results, Stat::on_axis_spectrum, cursor.take(plan.n_omega));
    cursor.finish();
  }
  std::uint64_t far = 0;
  if (wanted(Stat::beam_spectrum))
    far |= far_beam;
  if (wanted(Stat::absorbed_power))
    far |= far_absorbed;
  if (wanted(Stat::region_spectrum))
    far |= far_region;
  if (wanted(Stat::aperture_spectrum))
    far |= far_aperture;
  if (wanted(Stat::aperture_on_axis_spectrum))
    far |= far_aperture_axis;
  if (wanted(Stat::on_axis_spectrum) && plan.far_field_axis)
    far |= far_axis;
  if (far == 0U)
    return;
  const auto packed = state.aperture_statistics(far, plan.apertures, plan.exponent,
                                                plan.regions, plan.intensity_factor);
  Cursor cursor(packed);
  const std::size_t n = plan.n_omega;
  if ((far & far_beam) != 0U)
    put(results, Stat::beam_spectrum, cursor.take(n));
  if ((far & far_absorbed) != 0U)
    put(results, Stat::absorbed_power, cursor.take(plan.n_absorbed));
  if ((far & far_region) != 0U)
    put(results, Stat::region_spectrum, cursor.take(plan.n_regions * n));
  if ((far & far_aperture) != 0U)
    put(results, Stat::aperture_spectrum, cursor.take(plan.n_apertures * n));
  if ((far & far_aperture_axis) != 0U)
    put(results, Stat::aperture_on_axis_spectrum, cursor.take(plan.n_apertures * n));
  if ((far & far_axis) != 0U)
    put(results, Stat::on_axis_spectrum, cursor.take(n));
  cursor.finish();
}

/** A general modal basis's statistics: the temporal ones of the whitened columns
    `c = W^(1/2) a` (`solprop.stats._whitened`). */
void general(GeneralState& state, const Plan& plan, const double z,
             const Wanted& wanted, Results& results) {
  std::uint64_t flags = 0;
  if (wanted(Stat::omega0))
    flags |= temporal_omega0;
  if (wanted(Stat::physical_energy))
    flags |= temporal_energy;
  if (wanted(Stat::analytic_flux))
    flags |= temporal_peak_total;
  if (wanted.any({Stat::fwhm_t, Stat::fwhm_t_outer}))
    flags |= temporal_fwhm;
  const bool spectral = wanted(Stat::spectral_flux);
  if (flags == 0U && !spectral)
    return;
  std::vector<Complex> root;
  if (plan.root.present()) {
    const auto values = plan.root.at(z);
    root.reserve(values.size());
    for (const auto& value : values)
      root.emplace_back(value.real(), value.imag());
  }
  const auto packed =
      state.whitened_statistics(root, flags, {}, plan.intensity_factor, spectral);
  Cursor cursor(packed);
  const std::size_t columns = plan.n_modes * plan.n_pol;
  std::vector<double> centroids;
  if ((flags & temporal_omega0) != 0U)
    centroids = cursor.take(columns);
  if ((flags & temporal_energy) != 0U) {
    // `numpy.sum(energy(grid, c))`.
    const auto energies = cursor.take(columns);
    double sum = 0.0;
    for (const double value : energies)
      sum += value;
    put(results, Stat::physical_energy, {sum});
  }
  if ((flags & temporal_peak_total) != 0U)
    put(results, Stat::analytic_flux, cursor.take(1));
  if ((flags & temporal_fwhm) != 0U) {
    const auto power = cursor.take(plan.n_t);
    if (wanted(Stat::fwhm_t))
      put(results, Stat::fwhm_t,
          {half_width(plan.t.data(), power.data(), plan.n_t, false)});
    if (wanted(Stat::fwhm_t_outer))
      put(results, Stat::fwhm_t_outer,
          {half_width(plan.t.data(), power.data(), plan.n_t, true)});
  }
  const auto sums = cursor.take(columns);
  if ((flags & temporal_omega0) != 0U) {
    // `omega0`'s floor: a column whose density sums to at most one machine epsilon
    // of the largest column's has no centroid.
    double largest = 0.0;
    for (const double value : sums)
      largest = value > largest ? value : largest;
    const double floor = std::numeric_limits<double>::epsilon() * largest;
    for (std::size_t column = 0; column < columns; ++column)
      if (!(sums[column] > floor))
        centroids[column] = nan;
    put(results, Stat::omega0, std::move(centroids));
  }
  if (spectral) {
    auto bins = cursor.take(plan.n_omega);
    for (double& value : bins)
      value = plan.spectral_prefactor * value;
    put(results, Stat::spectral_flux, std::move(bins));
  }
  cursor.finish();
}

void check(const bool condition, const std::string& message) {
  if (!condition)
    throw std::invalid_argument("a statistics plan " + message);
}

} // namespace

template <class Value> void Table<Value>::validate(const char* name) const {
  if (!present()) {
    check(coefficients.empty(), std::string("has coefficients without edges: ") + name);
    return;
  }
  check(n_terms >= 1 && n_values >= 1, std::string("table has no terms: ") + name);
  check(coefficients.size() == (edges.size() - 1) * n_terms * n_values,
        std::string("table's coefficients have the wrong extent: ") + name);
  check(std::is_sorted(edges.begin(), edges.end()),
        std::string("table's edges do not increase: ") + name);
  check(lower <= edges.front() && upper >= edges.back(),
        std::string("table's span is not inside its slack: ") + name);
}

template <class Value> std::vector<Value> Table<Value>::at(const double z) const {
  if (!present())
    throw std::logic_error("a statistics plan's table is absent");
  std::vector<Value> out(n_values);
  const auto refused =
      chebyshev::clenshaw_points(&z, 1, edges.data(), edges.size(), lower, upper,
                                 coefficients.data(), n_terms, n_values, out.data());
  if (refused >= 0)
    throw std::out_of_range(
        "z = " + std::to_string(z) + " m is outside a statistics table's span [" +
        std::to_string(edges.front()) + ", " + std::to_string(edges.back()) + "] m");
  return out;
}

template <class Value> Value Table<Value>::scalar(const double z) const {
  if (n_values != 1)
    throw std::logic_error("a statistics plan's scalar table holds several values");
  return at(z)[0];
}

template struct Table<double>;
template struct Table<std::complex<double>>;

void Plan::validate() const {
  for (const Stat stat : outputs)
    check(static_cast<int>(stat) >= 0 && static_cast<int>(stat) < stat_count,
          "names an unknown statistic");
  check(windows.size() % 2 == 0, "has unpaired window edges");
  const std::size_t values = n_modes * n_pol * n_omega;
  weight_table.validate("power weight");
  switch (weight_kind) {
  case Weight::none:
    break;
  case Weight::constant:
    check(weight.size() == values, "power weight has the wrong extent");
    break;
  case Weight::expansion:
    check(weight_table.present() && weight.size() == weight_table.n_values * values,
          "power weight expansion has the wrong extent");
    break;
  case Weight::table:
    check(weight_table.present() && weight_table.n_values == values,
          "power weight table has the wrong extent");
    break;
  }
  check(on_axis.empty() || (n_modes > 0 && on_axis.size() % n_modes == 0 &&
                            on_axis.size() / n_modes <= 2),
        "on-axis synthesis is not one or two rows of the modes");
  for (const auto* table : {&taper, &plasma_density, &plasma_preionisation, &aeff})
    table->validate("scalar");
  for (const auto& table : channel_preionisation)
    table.validate("channel preionisation");
  for (const auto& table : species_density)
    table.validate("species density");
  for (const auto& table : profiles)
    table.validate("profile");
  check(channel_preionisation.empty() || channel_preionisation.size() == n_channels,
        "has one preionisation a channel");
  check(species_density.size() == n_species, "has one density a species");
  check(channel_species.size() == n_channels && potentials.size() == n_channels,
        "maps each channel to a species with its potential");
  for (const std::size_t s : channel_species)
    check(s < n_species, "maps a channel past the species");
  check(transverse != Transverse::aeff || aeff.present(), "lacks the effective area");
  check((transverse != Transverse::spatial && transverse != Transverse::nodes) ||
            node_weights.size() == n_nodes,
        "has one transverse weight a node");
  check(t.empty() || t.size() == n_t, "time axis has the wrong extent");
  root.validate("metric root");
  check(!root.present() || root.n_values == n_omega * n_modes * n_modes,
        "metric root has the wrong extent");
  check(apertures.size() == n_apertures * (11 + n_omega),
        "apertures have the wrong extent");
  check(regions.size() == n_regions * 4, "regions have the wrong extent");
  const Wanted wanted(outputs);
  if (wanted.any({Stat::fwhm_t, Stat::fwhm_t_outer}))
    check(t.size() == n_t, "lacks the time axis its widths read");
  for (const auto [stat, index] :
       {std::pair{Stat::density, 0}, std::pair{Stat::pressure, 1},
        std::pair{Stat::core_radius, 2}})
    if (wanted(stat))
      check(profiles[static_cast<std::size_t>(index)].present() &&
                profiles[static_cast<std::size_t>(index)].n_values == 1,
            "lacks a profile statistic's table");
}

std::size_t Plan::width(const Stat stat) const {
  const std::size_t columns = n_modes * n_pol;
  const std::size_t n_windows = windows.size() / 2;
  switch (stat) {
  case Stat::omega0:
    return columns;
  case Stat::energy:
    return route == Route::spatial ? 1 : columns;
  case Stat::energy_window:
    return n_windows * columns;
  case Stat::peak_power:
    return n_modes;
  case Stat::peak_power_window:
    return n_windows * n_modes;
  case Stat::species_ionisation:
    return n_species * n_nodes;
  case Stat::species_electron_density:
  case Stat::species_peak_ionisation_rate:
  case Stat::species_density:
  case Stat::species_ionisation_energy:
    return n_species;
  case Stat::spectral_flux:
  case Stat::on_axis_spectrum:
  case Stat::beam_spectrum:
    return n_omega;
  case Stat::absorbed_power:
    return n_absorbed;
  case Stat::region_spectrum:
    return n_regions * n_omega;
  case Stat::aperture_spectrum:
  case Stat::aperture_on_axis_spectrum:
    return n_apertures * n_omega;
  default:
    return 1;
  }
}

std::size_t Plan::packed_size() const {
  std::size_t size = 0;
  for (const Stat stat : outputs)
    size += width(stat);
  return size;
}

std::vector<double> evaluate(GeneralState& state, const Plan& plan, const double z) {
  const Wanted wanted(plan.outputs);
  Results results;
  const double taper = plan.taper.present() ? plan.taper.scalar(z) : 1.0;
  if (plan.route == Route::general) {
    general(state, plan, z, wanted, results);
  } else {
    temporal(state, plan, z, taper, wanted, results);
    if (plan.route == Route::spatial)
      spatial(state, plan, z, wanted, results);
  }
  for (const auto [stat, index] :
       {std::pair{Stat::density, 0}, std::pair{Stat::pressure, 1},
        std::pair{Stat::core_radius, 2}})
    if (wanted(stat))
      put(results, stat, {plan.profiles[static_cast<std::size_t>(index)].scalar(z)});
  std::vector<double> packed;
  packed.reserve(plan.packed_size());
  for (const Stat stat : plan.outputs) {
    const auto& values = results[static_cast<std::size_t>(stat)];
    if (values.size() != plan.width(stat))
      throw std::logic_error("statistic " + std::to_string(static_cast<int>(stat)) +
                             " is not evaluated on this route");
    packed.insert(packed.end(), values.begin(), values.end());
  }
  return packed;
}

double half_width(const double* t, const double* power, const std::size_t n,
                  const bool outer) {
  if (n == 0)
    return nan;
  // `numpy.max` and `numpy.argmax`; a NaN anywhere leaves no width.
  std::size_t apex = 0;
  for (std::size_t index = 0; index < n; ++index) {
    if (std::isnan(power[index]))
      return nan;
    if (power[index] > power[apex])
      apex = index;
  }
  const double peak = power[apex];
  if (!(peak > 0.0))
    return nan;
  const double level = 0.5 * peak;
  std::size_t low_pair[2]{};
  std::size_t high_pair[2]{};
  if (outer) {
    // The first and the last sample above half maximum anywhere on the axis.
    std::size_t first = n;
    std::size_t last = n;
    for (std::size_t index = 0; index < n; ++index)
      if (power[index] > level) {
        if (first == n)
          first = index;
        last = index;
      }
    if (first == n || first == 0 || last + 1 >= n)
      return nan;
    low_pair[0] = first - 1;
    low_pair[1] = first;
    high_pair[0] = last + 1;
    high_pair[1] = last;
  } else {
    // Walk outwards from the peak to the first sample below half maximum.
    std::size_t below_left = n;
    for (std::size_t index = 0; index < apex; ++index)
      if (power[index] < level)
        below_left = index;
    std::size_t below_right = n;
    for (std::size_t index = apex; index < n; ++index)
      if (power[index] < level) {
        below_right = index;
        break;
      }
    if (below_left == n || below_right == n)
      return nan;
    low_pair[0] = below_left;
    low_pair[1] = below_left + 1;
    high_pair[0] = below_right;
    high_pair[1] = below_right - 1;
  }
  const auto crossing = [&](const std::size_t low, const std::size_t high) {
    // `x1 + (level - y1)(x2 - x1)/(y2 - y1)`, in Python's order.
    const double span = power[high] - power[low];
    const double rise = level - power[low];
    const double run = t[high] - t[low];
    const double product = rise * run;
    const double step = product / span;
    return t[low] + step;
  };
  const double start = crossing(low_pair[0], low_pair[1]);
  const double end = crossing(high_pair[0], high_pair[1]);
  return end - start;
}

} // namespace solprop_native::statistics
