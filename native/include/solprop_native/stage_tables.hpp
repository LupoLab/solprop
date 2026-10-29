// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/chebyshev.hpp"
#include "solprop_native/core.hpp"
#include "solprop_native/lagrange_basis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
 * The stage scalars of an exponential Adams trial, formed natively from the tables
 * solprop uploads once per run (solprop SEP 39 phase 1).
 *
 * solprop forms a trial's stage scalars (`solprop.translate.stage.scalars`) by reading
 * its piecewise Chebyshev tables at the stage's position: the weights of `Phi` and of
 * the prefactor, the density, the taper scale, the Raman decay rates and densities, the
 * Kerr density, and the plasma channels' densities and pre-ionised fractions. Each read
 * is one call of the native Clenshaw helper (`chebyshev::clenshaw_points`). These
 * functions perform the same reads, by the same helper, with the same operations in
 * the same order -- the increment of an expansion's `Phi` is its weights at the stage
 * less its weights at the anchor, element by element -- so the scalars they form are
 * solprop's to the bit. A varying operator's rate is its derivative table read on the
 * closing side of a kink, as `solprop.translate.linop._rate_value` reads it.
 *
 * Host code only; nothing here is a device function.
 */
namespace solprop_native::stage {

/** One of solprop's piecewise Chebyshev tables of real values. */
struct Table {
  std::vector<double> edges;
  /** The span widened by its slack: a position outside it is refused. */
  double lower = 0.0;
  double upper = 0.0;
  /** `(n_intervals, n_terms, n_values)`, C-contiguous. */
  std::vector<double> coefficients;
  std::size_t n_terms = 0;
  std::size_t n_values = 0;

  [[nodiscard]] bool present() const { return n_values > 0; }

  /** Write the table's values at `z`, refusing a position outside its span. */
  void at(const double z, double* out) const {
    const auto bad =
        chebyshev::clenshaw_points(&z, 1, edges.data(), edges.size(), lower, upper,
                                   coefficients.data(), n_terms, n_values, out);
    if (bad >= 0)
      throw std::invalid_argument("z = " + std::to_string(z) +
                                  " is outside a stage table's span");
  }

  /** The table's single value at `z`. */
  [[nodiscard]] double value(const double z) const {
    double out = 0.0;
    at(z, &out);
    return out;
  }

  /**
   * Write the values at `z` read as `_rate_value` reads them: the chosen interval as
   * a one-interval table at the same map onto [-1, 1], from the interval ending at an
   * interior edge when `before`.
   */
  void closing(const double z, double* out, const bool before = true) const {
    if (z < lower || z > upper)
      throw std::invalid_argument("z = " + std::to_string(z) +
                                  " is outside a stage table's span");
    std::size_t index = chebyshev::interval_index(z, edges.data(), edges.size());
    if (before && index > 0 && z <= edges[index])
      --index;
    const double ends[2] = {edges[index], edges[index + 1]};
    constexpr double infinite = std::numeric_limits<double>::infinity();
    chebyshev::clenshaw_points(&z, 1, ends, 2, -infinite, infinite,
                               coefficients.data() + index * n_terms * n_values,
                               n_terms, n_values, out);
  }
};

/** How `Phi` and the prefactor are represented: a constant or an expansion. */
enum class Terms : int { constant = 1, expansion = 2 };

/** How the Kerr density is read: none, the density itself, or a table of its own. */
enum class KerrDensity : int { none = 0, density = 1, table = 2 };

/** Every table a trial's stage scalars read, uploaded once per run. */
struct Tables {
  std::size_t n_stages = 0;
  Terms phi = Terms::constant;
  /** An expansion's integrated weight tables, and their derivative (the rate). */
  Table phi_weights;
  Table phi_rate;
  Terms prefactor = Terms::constant;
  Table prefactor_weights;
  /** A constant prefactor's span, which a stage's position must lie in. */
  double prefactor_low = 0.0;
  double prefactor_high = 0.0;
  Table density;
  /** Absent for a guide that does not taper: the scale is one. */
  Table taper;
  /** Every Raman term's decay-rate tables, term by term, in order. */
  std::vector<Table> raman_rates;
  KerrDensity kerr = KerrDensity::none;
  Table kerr_density;
  std::vector<Table> plasma_densities;
  std::vector<Table> plasma_preionisation;
  /** One per Raman term; an absent table reads the density. */
  std::vector<Table> raman_densities;
};

/** The anchor's `Phi` weights, kept for every stage of one step. */
struct Anchor {
  double z = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> phi;
};

/** One stage's scalars at `z1` from the anchor `z0`: one row of a trial's arrays. */
struct Row {
  std::vector<double> phi_delta;
  std::vector<double> phi_at;
  std::vector<double> prefactor;
  double density = 0.0;
  double taper = 1.0;
  std::vector<double> raman_rates;
  double kerr_density = 0.0;
  std::vector<double> plasma_densities;
  std::vector<double> plasma_preionisation;
  std::vector<double> raman_densities;
};

/** Return one stage's scalars at `z1`, from the anchor `z0` (`scalars(problem, z0,
 * z1)`). */
inline Row row(const Tables& tables, Anchor& anchor, const double z0, const double z1) {
  Row out;
  if (tables.phi == Terms::constant) {
    // Phi(z) = D z: the increment's weight is the distance and the position's the
    // position, as `_phi_weights` forms them.
    out.phi_delta = {chebyshev::sub(z1, z0)};
    out.phi_at = {z1};
  } else {
    const auto& weights = tables.phi_weights;
    out.phi_at.resize(weights.n_values);
    weights.at(z1, out.phi_at.data());
    if (!(anchor.z == z0)) {
      anchor.phi.resize(weights.n_values);
      weights.at(z0, anchor.phi.data());
      anchor.z = z0;
    }
    out.phi_delta.resize(weights.n_values);
    for (std::size_t j = 0; j < weights.n_values; ++j)
      out.phi_delta[j] = chebyshev::sub(out.phi_at[j], anchor.phi[j]);
  }
  if (tables.prefactor == Terms::constant) {
    if (!(tables.prefactor_low <= z1 && z1 <= tables.prefactor_high))
      throw std::invalid_argument("z = " + std::to_string(z1) +
                                  " lies outside the prefactor's table");
    out.prefactor = {1.0};
  } else {
    out.prefactor.resize(tables.prefactor_weights.n_values);
    tables.prefactor_weights.at(z1, out.prefactor.data());
  }
  out.density = tables.density.value(z1);
  out.taper = tables.taper.present() ? tables.taper.value(z1) : 1.0;
  out.raman_rates.reserve(tables.raman_rates.size());
  for (const auto& table : tables.raman_rates)
    out.raman_rates.push_back(table.value(z1));
  out.kerr_density = tables.kerr == KerrDensity::none ? 0.0
                     : tables.kerr == KerrDensity::density
                         ? out.density
                         : tables.kerr_density.value(z1);
  for (const auto& table : tables.plasma_densities)
    out.plasma_densities.push_back(table.value(z1));
  for (const auto& table : tables.plasma_preionisation)
    out.plasma_preionisation.push_back(table.value(z1));
  for (const auto& table : tables.raman_densities)
    out.raman_densities.push_back(table.present() ? table.value(z1) : out.density);
  return out;
}

/** A trial's step arrays, and its lab-frame sample's: the same, increment zeroed. */
struct Trial {
  TrialScalars step;
  TrialScalars sample;
};

/** Append `values` to `into` `count` times. */
inline void repeat(std::vector<double>& into, const std::vector<double>& values,
                   const std::size_t count) {
  into.reserve(into.size() + count * values.size());
  for (std::size_t copy = 0; copy < count; ++copy)
    into.insert(into.end(), values.begin(), values.end());
}

/**
 * Return a trial's arrays with every stage row at one position: the native backend's
 * `_repeated_arrays` of the bundle, value for value. The increment has one row more
 * than the stages, the step's own; the sole plasma channel's fraction is the
 * one-channel path's, zero otherwise.
 */
inline TrialScalars trial(const Tables& tables, const Row& stage) {
  const std::size_t stages = tables.n_stages;
  TrialScalars out;
  repeat(out.phi_delta, stage.phi_delta, stages + 1);
  repeat(out.phi_at, stage.phi_at, stages);
  repeat(out.prefactor, stage.prefactor, stages);
  out.density.assign(stages, stage.density);
  out.taper.assign(stages, stage.taper);
  out.preionisation.assign(stages, stage.plasma_preionisation.size() == 1
                                       ? stage.plasma_preionisation[0]
                                       : 0.0);
  repeat(out.raman_rates, stage.raman_rates, stages);
  out.kerr_density.assign(stages, stage.kerr_density);
  repeat(out.plasma_densities, stage.plasma_densities, stages);
  repeat(out.plasma_preionisation, stage.plasma_preionisation, stages);
  repeat(out.raman_densities, stage.raman_densities, stages);
  return out;
}

/**
 * Return `D(z)`'s weights over the native `Phi` terms (`rate_weights`), read on the
 * closing side of an edge when `before`: a trial's endpoint; a startup sample reads
 * the far side.
 */
inline std::vector<double> rate(const Tables& tables, const double z,
                                const bool before = true) {
  if (tables.phi == Terms::constant)
    return {1.0};
  std::vector<double> out(tables.phi_rate.n_values);
  tables.phi_rate.closing(z, out.data(), before);
  return out;
}

/**
 * Return an exponential trial's tables formed from its nodes
 * (`lagrange::trial_tables`): the operations of solprop's `trial_tables` in their
 * order, laid out as the native trial takes them, with the compensation's when
 * `compensated`. `nodes` holds the history's normalised positions newest first, `slots`
 * their rows.
 */
inline ExponentialWeights weights(const double* nodes, const std::size_t n,
                                  const std::size_t* slots, const std::size_t n_slots,
                                  const std::size_t order, const std::size_t* orders,
                                  const std::size_t n_orders, const bool compensated) {
  constexpr std::size_t most = lagrange::max_nodes;
  std::size_t highest = 0;
  for (std::size_t index = 0; index < n_orders; ++index)
    highest = std::max(highest, orders[index]);
  if (n_orders == 0 || n_orders > 3 || order == 0 || order > highest || highest > n ||
      highest + 1 > most || highest + 1 > adams_max_samples ||
      n_slots + 1 < highest + 1 ||
      std::any_of(orders, orders + n_orders,
                  [](const std::size_t value) { return value == 0; }))
    throw std::invalid_argument("Adams coefficient tables have inconsistent extents");
  const std::size_t width = highest + 1;
  std::array<double, most * most> predictor{};
  std::array<double, most * most> corrector{};
  std::array<double, 3 * 2 * most> factored{};
  std::array<double, most> spans{};
  std::array<double, most> carried{};
  std::array<double, most + 1> endpoint{};
  std::array<double, 3 * most> defects{};
  lagrange::trial_tables(nodes, order, orders, n_orders, width, compensated,
                         predictor.data(), corrector.data(), factored.data(),
                         spans.data(), carried.data(), endpoint.data(), defects.data());
  ExponentialWeights out;
  for (std::size_t t = 0; t < n_slots && t < adams_max_samples; ++t)
    out.slot[t] = slots[t];
  for (std::size_t j = 0; j < order; ++j)
    for (std::size_t k = 0; k < order; ++k)
      out.predictor[j][k] = predictor[j * order + k];
  for (std::size_t j = 0; j < width; ++j)
    for (std::size_t k = 0; k < width; ++k)
      out.corrector[j][k] = corrector[j * width + k];
  out.order = order;
  out.width = width;
  out.count = width;
  out.n_rows = n_orders;
  for (std::size_t row = 0; row < n_orders; ++row)
    for (std::size_t t = 0; t < 2 * width; ++t)
      out.rows[row][t] = factored[row * 2 * width + t];
  out.compensated = compensated;
  if (compensated) {
    for (std::size_t t = 0; t + 1 < width; ++t)
      out.span[t] = spans[t];
    for (std::size_t t = 0; t < order; ++t)
      out.comp_predictor[t] = carried[t];
    for (std::size_t t = 0; t <= order; ++t)
      out.comp_corrector[t] = endpoint[t];
    for (std::size_t row = 0; row < n_orders; ++row)
      for (std::size_t t = 0; t < width; ++t)
        out.comp_rows[row][t] = defects[row * width + t];
  }
  return out;
}

} // namespace solprop_native::stage
