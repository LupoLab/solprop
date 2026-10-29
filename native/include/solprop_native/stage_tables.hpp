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

/** How `Phi` and the prefactor are represented: a constant, an expansion, or a z
    table's slots (`ZSlots`). A separable operator's are a constant's: the distance and
    the position, and a unit prefactor, with no span. */
enum class Terms : int { constant = 1, expansion = 2, table = 3 };

/** How the Kerr density is read: none, the density itself, or a table of its own. */
enum class KerrDensity : int { none = 0, density = 1, table = 2 };

struct ZSlots;

/**
 * Write NumPy's `chebvander` row at `x`, `degrees` values: its forward recursion
 * `T_i = T_{i-1} 2x - T_{i-2}`, each operation rounded on its own.
 */
SOLPROP_NATIVE_STRICT_FUNCTION
inline void chebvander_row(const double x, const std::size_t degrees, double* row) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const double zero = x * 0.0;
  row[0] = zero + 1.0;
  if (degrees > 1) {
    const double x2 = 2.0 * x;
    row[1] = x;
    for (std::size_t i = 2; i < degrees; ++i) {
      const double product = row[i - 1] * x2;
      row[i] = product - row[i - 2];
    }
  }
}

/**
 * A Chebyshev basis in sections along z: a coupled flow's residual weights
 * (`translate.coupled.flow_weights`) and a moving guide's map weights
 * (`translate.modal_path.transform_weights`). Each position's weights are its
 * section's `chebvander` row at the section's mapped coordinate, zero elsewhere.
 */
struct Sections {
  std::vector<double> edges;
  std::size_t degrees = 0;
  /** How far outside an edge a position still belongs to the section, m. */
  double slack = 0.0;

  [[nodiscard]] bool present() const { return degrees > 0; }
  [[nodiscard]] std::size_t terms() const { return (edges.size() - 1) * degrees; }

  /** `flow_section`/`transform_section`: the section starting at or below the
      position widened by the slack, clamped to the ends. */
  [[nodiscard]] std::size_t section(const double z) const { return clamped(z + slack); }

  /** The section a stage reads at `z` from its step's `own` (`flow_weights`): its own
      while `z` lies in it widened by the slack, else the one closing at `z`. */
  [[nodiscard]] std::size_t reading(const double z, const std::size_t own) const {
    if (edges[own] - slack <= z && z <= edges[own + 1] + slack)
      return own;
    return clamped(z - slack);
  }

  /** The weights at `z` in section `index`: its `chebvander` row at the section's
      mapped coordinate, as `flow_weights` maps it (no product to contract). */
  [[nodiscard]] std::vector<double> weights(const double z,
                                            const std::size_t index) const {
    std::vector<double> out(terms(), 0.0);
    const double lo = edges[index];
    const double hi = edges[index + 1];
    const double doubled = 2.0 * z;
    const double x = ((doubled - lo) - hi) / (hi - lo);
    chebvander_row(x, degrees, out.data() + index * degrees);
    return out;
  }

private:
  [[nodiscard]] std::size_t clamped(const double key) const {
    // `numpy.searchsorted(edges, key, side="right") - 1`, clamped to the sections.
    const auto above = std::upper_bound(edges.begin(), edges.end(), key);
    const auto raw = static_cast<std::ptrdiff_t>(above - edges.begin()) - 1;
    const auto last = static_cast<std::ptrdiff_t>(edges.size()) - 2;
    return static_cast<std::size_t>(std::min(std::max(raw, std::ptrdiff_t{0}), last));
  }
};

/** How the carrier frame's chord is represented (SEP 35 section 6.1). */
enum class Chord : int { none = 0, constant = 1, expansion = 2 };

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
  /** SEP 6's dispersive channels' number densities, one per channel. */
  std::vector<Table> dispersive_densities;
  /** A z table's slots and the state they are evaluated in, not owned. */
  ZSlots* z_slots = nullptr;
  GeneralState* z_state = nullptr;
  /** The carrier frame's chord (solprop's `CarrierChord`), the imaginary part of the
      reference element's operator: none, a constant's rate, or an expansion's terms
      over `phi_weights`. */
  Chord chord = Chord::none;
  double chord_rate = 0.0;
  std::vector<double> chord_terms;
  /** A coupled flow's sections (the step's section, and a frozen-residual flow's
      weights), and a moving guide's maps. */
  Sections flow;
  bool flow_residual = false;
  Sections transform;
};

/**
 * A z table's position slots (solprop's `_ZTables`, SEP 29 G1), least recently used
 * first: the positions a trial reads are evaluated into the native state's term slots
 * only where no slot holds them, so a step's end, the next step's anchor, is evaluated
 * once. Both Adams marches and every other consumer share this one cache, and its
 * memo of the rate slot's position.
 */
struct ZSlots {
  bool phi = false;
  bool prefactor = false;
  std::size_t slots = 0;
  /** A compensated march reads `D(z)` into one more `Phi` term. */
  bool has_rate = false;
  std::uint64_t evaluations = 0;

  void configure(const bool phi_table, const bool prefactor_table,
                 const std::size_t count, const bool rate) {
    phi = phi_table;
    prefactor = prefactor_table;
    slots = count;
    has_rate = rate;
    held.assign(count, std::numeric_limits<double>::quiet_NaN());
    order.resize(count);
    for (std::size_t slot = 0; slot < count; ++slot)
      order[slot] = slot;
    evaluations = 0;
    rate_known = false;
  }

  [[nodiscard]] bool present() const { return phi || prefactor; }
  /** The native `Phi` term count: the slots, and the rate's when it is read. */
  [[nodiscard]] std::size_t n_phi() const { return slots + (has_rate ? 1 : 0); }
  [[nodiscard]] std::size_t rate_slot() const { return slots; }

  /**
   * Return each of `positions`' slots, evaluating in one call those no slot holds:
   * `_ZTables.load`, the slots touched in the same order.
   */
  std::vector<std::size_t> load(GeneralState& state,
                                const std::vector<double>& positions) {
    std::vector<double> wanted;
    for (const double z : positions)
      if (std::find(wanted.begin(), wanted.end(), z) == wanted.end())
        wanted.push_back(z);
    if (wanted.size() > slots)
      throw std::invalid_argument("a native call reads the z table at " +
                                  std::to_string(wanted.size()) +
                                  " positions, more than the " + std::to_string(slots) +
                                  " slots this problem holds");
    std::vector<std::size_t> slot_of(wanted.size(), slots);
    std::vector<bool> taken(slots, false);
    for (std::size_t index = 0; index < wanted.size(); ++index) {
      const auto at = std::find(held.begin(), held.end(), wanted[index]);
      if (at != held.end()) {
        slot_of[index] = static_cast<std::size_t>(at - held.begin());
        taken[slot_of[index]] = true;
        touch(slot_of[index]);
      }
    }
    std::vector<std::size_t> evaluate_slots;
    std::vector<double> evaluate_positions;
    for (std::size_t index = 0; index < wanted.size(); ++index) {
      if (slot_of[index] != slots)
        continue;
      std::size_t slot = slots;
      for (const std::size_t candidate : order)
        if (!taken[candidate]) {
          slot = candidate;
          break;
        }
      held[slot] = wanted[index];
      slot_of[index] = slot;
      taken[slot] = true;
      touch(slot);
      evaluate_slots.push_back(slot);
      evaluate_positions.push_back(wanted[index]);
    }
    if (!evaluate_slots.empty()) {
      state.table_evaluate(evaluate_slots, evaluate_positions);
      evaluations += evaluate_slots.size();
    }
    std::vector<std::size_t> out;
    out.reserve(positions.size());
    for (const double z : positions)
      out.push_back(slot_of[static_cast<std::size_t>(
          std::find(wanted.begin(), wanted.end(), z) - wanted.begin())]);
    return out;
  }

  /** Evaluate `D(z)` into the rate slot unless it holds it already (`_ZTables.rate`).
   */
  void rate(GeneralState& state, const double z, const bool before) {
    if (rate_known && rate_z == z && rate_before == before)
      return;
    state.table_rate(rate_slot(), z, before);
    rate_known = true;
    rate_z = z;
    rate_before = before;
  }

private:
  void touch(const std::size_t slot) {
    order.erase(std::find(order.begin(), order.end(), slot));
    order.push_back(slot);
  }

  std::vector<double> held;
  /** Least recently used first. */
  std::vector<std::size_t> order;
  bool rate_known = false;
  double rate_z = 0.0;
  bool rate_before = false;
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
  std::vector<double> dispersive_densities;
  std::size_t flow_section = 0;
  std::vector<double> flow_weights;
  std::vector<double> transform_weights;
};

/** Return one stage's scalars at `z1`, from the anchor `z0` (`scalars(problem, z0,
 * z1)`). */
inline Row row(const Tables& tables, Anchor& anchor, const double z0, const double z1) {
  Row out;
  // A z table's positions, loaded together as `_weight_rows` loads them: the anchor
  // for `Phi`, then the stage.
  std::size_t from_slot = 0;
  std::size_t at_slot = 0;
  if (tables.phi == Terms::table || tables.prefactor == Terms::table) {
    std::vector<double> positions;
    if (tables.phi == Terms::table)
      positions.push_back(z0);
    positions.push_back(z1);
    const auto slots = tables.z_slots->load(*tables.z_state, positions);
    from_slot = slots.front();
    at_slot = slots.back();
  }
  if (tables.phi == Terms::table) {
    // `phi_row` and `at_row`: the increment `e_1 - e_0`, the value `e_1`.
    const std::size_t terms = tables.z_slots->n_phi();
    out.phi_delta.assign(terms, 0.0);
    out.phi_delta[at_slot] += 1.0;
    out.phi_delta[from_slot] -= 1.0;
    out.phi_at.assign(terms, 0.0);
    out.phi_at[at_slot] = 1.0;
  } else if (tables.phi == Terms::constant) {
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
  if (tables.prefactor == Terms::table) {
    out.prefactor.assign(tables.z_slots->slots, 0.0);
    out.prefactor[at_slot] = 1.0;
  } else if (tables.prefactor == Terms::constant) {
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
  for (const auto& table : tables.dispersive_densities)
    out.dispersive_densities.push_back(table.value(z1));
  if (tables.flow.present()) {
    // The step's section is its anchor's; a frozen residual reads it at the stage.
    out.flow_section = tables.flow.section(z0);
    if (tables.flow_residual)
      out.flow_weights =
          tables.flow.weights(z1, tables.flow.reading(z1, out.flow_section));
  }
  if (tables.transform.present())
    out.transform_weights = tables.transform.weights(z1, tables.transform.section(z1));
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
  repeat(out.dispersive_densities, stage.dispersive_densities, stages);
  out.flow_section = stage.flow_section;
  repeat(out.flow_weights, stage.flow_weights, stages);
  repeat(out.transform_weights, stage.transform_weights, stages);
  return out;
}

/**
 * Return a trial's lab-frame sample's arrays: the step's with the increment of `Phi`
 * zeroed (`_resident_sample`), or, beside a coupled flow, whose section follows the
 * anchor, a stage anchored at the step's end.
 */
inline TrialScalars sample(const Tables& tables, Anchor& anchor,
                           const TrialScalars& step, const double end) {
  if (tables.flow.present())
    return trial(tables, row(tables, anchor, end, end));
  auto out = step;
  std::fill(out.phi_delta.begin(), out.phi_delta.end(), 0.0);
  return out;
}

/**
 * Return the carrier frame's shift over `[z1, z2]`: `Im (Phi(z2) - Phi(z1))` at the
 * reference element (`CarrierChord.exponent`). A constant's is its rate times the
 * distance, as solprop forms it; an expansion's is the weights' difference against the
 * terms, summed in order, where solprop's `tensordot` sums in its BLAS's order: the two
 * agree to rounding (SEP 39 section 4.9).
 */
inline double chord(const Tables& tables, const double z1, const double z2) {
  if (tables.chord == Chord::constant)
    return tables.chord_rate * (z2 - z1);
  if (tables.chord == Chord::none)
    return 0.0;
  const auto& weights = tables.phi_weights;
  std::vector<double> upper(weights.n_values);
  std::vector<double> lower(weights.n_values);
  weights.at(z2, upper.data());
  weights.at(z1, lower.data());
  double sum = 0.0;
  for (std::size_t k = 0; k < weights.n_values; ++k) {
    const double delta = upper[k] - lower[k];
    const double term = delta * tables.chord_terms[k];
    sum = sum + term;
  }
  return sum;
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
  if (tables.phi == Terms::table) {
    // `_ZTables.rate`: `D(z)` evaluated into its slot, one at that slot.
    tables.z_slots->rate(*tables.z_state, z, before);
    std::vector<double> out(tables.z_slots->n_phi(), 0.0);
    out[tables.z_slots->rate_slot()] = 1.0;
    return out;
  }
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

/**
 * Return an interaction-formulation (fused) trial's tables formed from its nodes:
 * solprop's `backends.multistep.fused_tables` over `lagrange::basis_rows`, the
 * operations in their order. The predictor's integrated weights on the `order` newest
 * nodes; the step's own corrector's on `(1, nodes[:order])`, zero past them; and per
 * scored order `k` its corrector's weights on `(1, nodes[:k])` less, from the second
 * sample on, its predictor's on `nodes[:k]`. `nodes` holds the history's normalised
 * positions newest first, `slots` their rows.
 */
inline AdamsWeights fused_weights(const double* nodes, const std::size_t n,
                                  const std::size_t* slots, const std::size_t n_slots,
                                  const std::size_t order, const std::size_t* orders,
                                  const std::size_t n_orders) {
  // `basis_rows` is strict; the defects' subtractions cannot contract.
  std::size_t highest = 0;
  for (std::size_t index = 0; index < n_orders; ++index)
    highest = std::max(highest, orders[index]);
  if (n_orders == 0 || n_orders > 3 || order == 0 || order > highest || highest > n ||
      highest + 1 > lagrange::max_nodes || highest + 1 > adams_max_samples ||
      n_slots + 1 < highest + 1 || orders[0] != order ||
      std::any_of(orders, orders + n_orders,
                  [](const std::size_t value) { return value == 0; }))
    throw std::invalid_argument("Adams coefficient tables have inconsistent extents");
  const std::size_t width = highest + 1;
  std::array<double, lagrange::max_nodes> through{};
  std::array<double, lagrange::max_nodes * lagrange::max_nodes> scratch{};
  std::array<double, lagrange::max_nodes> basis{};
  std::array<double, lagrange::max_nodes> integrated{};
  through[0] = 1.0;
  for (std::size_t t = 0; t + 1 < width; ++t)
    through[t + 1] = nodes[t];
  AdamsWeights out;
  for (std::size_t t = 0; t < n_slots && t < adams_max_samples; ++t)
    out.slot[t] = slots[t];
  lagrange::basis_rows(nodes, order, 1.0, true, scratch.data(), basis.data());
  for (std::size_t t = 0; t < order; ++t)
    out.predictor[t] = basis[t];
  for (std::size_t index = 0; index < n_orders; ++index) {
    const std::size_t scored = orders[index];
    lagrange::basis_rows(through.data(), scored + 1, 1.0, true, scratch.data(),
                         integrated.data());
    double* const row = out.rows[index];
    for (std::size_t t = 0; t <= scored; ++t)
      row[t] = integrated[t];
    if (index == 0)
      for (std::size_t t = 0; t <= scored; ++t)
        out.corrector[t] = integrated[t];
    lagrange::basis_rows(nodes, scored, 1.0, true, scratch.data(), basis.data());
    for (std::size_t t = 0; t < scored; ++t)
      row[t + 1] = row[t + 1] - basis[t];
  }
  out.order = order;
  out.width = width;
  out.n_rows = n_orders;
  return out;
}

} // namespace solprop_native::stage
