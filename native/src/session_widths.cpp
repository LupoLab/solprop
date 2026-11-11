// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// solprop SEP 47 section 6.4: a spatial route's fluence widths, finished natively from
// the reductions the native passes leave, as solprop's host finishing reads them
// (`solprop.backends.spatial.signed_fluence_widths`, `plane_fluence_widths`;
// `solprop.spatial.radial_fluence_widths`, `cartesian_fluence_widths`,
// `_line_fluence_widths`, `_fluence_rings`, `_ring_fluence_widths`). Each NumPy sum is
// taken in NumPy's pairwise order, so on a radial, line or slab profile the widths are
// the host's bit for bit; a plane's extent is `std::hypot`, which may differ from
// Python's `math.hypot` in its last bit. The arithmetic rounds as NumPy's does, so this
// file takes no contraction (CMake), and the optimize attribute keeps GCC from inlining
// it into a contracting caller.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

#include "solprop_native/core.hpp"
#include "solprop_native/statistics_plan.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace solprop_native::statistics::widths {

namespace {

constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double eps = std::numeric_limits<double>::epsilon();

/** NumPy's unrolled block: eight partial sums, then the rest in order. */
constexpr std::size_t pairwise_block = 128;

std::array<double, 3> undefined() { return {nan, nan, nan}; }

/** Python's `max(a, b)`: `b` only where it exceeds `a`. */
double python_max(const double a, const double b) { return b > a ? b : a; }

/** Python's `min(a, b)`: `b` only where it is below `a`. */
double python_min(const double a, const double b) { return b < a ? b : a; }

/** Python's `f"{value:.3e}"`. */
std::string scientific(const double value) {
  if (std::isnan(value))
    return "nan";
  if (std::isinf(value))
    return value > 0.0 ? "inf" : "-inf";
  char text[32];
  std::snprintf(text, sizeof text, "%.3e", value);
  return text;
}

/** `solprop.backends.spatial._negative_fluence`. */
std::invalid_argument negative_fluence(const double value, const std::size_t node) {
  return std::invalid_argument(
      "beam_radius and fwhm_r need a nonnegative fluence profile, and the signed "
      "axial flux is " +
      scientific(value) + " J/m^2 at node " + std::to_string(node) +
      ", beyond its rounding budget; the width is undefined for it");
}

/** `fluence_budget`'s one value: eight rounding units a term of the largest
    magnitude, `8 * n_terms * eps * largest` in Python's order. */
double budget(const std::size_t n_terms, const double largest) {
  return static_cast<double>(8 * n_terms) * eps * largest;
}

/** `numpy.interp(point, x, samples)` on an increasing axis, its compiled rule. */
double interp(const double point, const std::vector<double>& x,
              const std::vector<double>& samples) {
  const std::size_t n = x.size();
  if (std::isnan(point))
    return point;
  if (point < x.front())
    return samples.front();
  if (point > x.back())
    return samples.back();
  // The last `j` with `x[j] <= point`.
  const auto above = std::ranges::upper_bound(x, point);
  const auto j = static_cast<std::size_t>(above - x.begin()) - 1;
  if (j == n - 1 || x[j] == point)
    return samples[j];
  const double slope = (samples[j + 1] - samples[j]) / (x[j + 1] - x[j]);
  double value = slope * (point - x[j]) + samples[j];
  if (std::isnan(value)) {
    value = slope * (point - x[j + 1]) + samples[j + 1];
    if (std::isnan(value) && samples[j] == samples[j + 1])
      value = samples[j];
  }
  return value;
}

/** `cartesian_fluence_widths`' check that an axis is uniformly increasing. */
void require_uniform(const std::vector<double>& axis) {
  const double spacing =
      (axis.back() - axis.front()) / static_cast<double>(axis.size() - 1);
  double largest = 0.0;
  for (const double value : axis)
    largest = python_max(largest, std::abs(value));
  const double tolerance = 32 * eps * std::max(largest, spacing);
  bool uniform = spacing > 0.0;
  for (std::size_t index = 0; uniform && index + 1 < axis.size(); ++index)
    uniform = !(std::abs((axis[index + 1] - axis[index]) - spacing) > tolerance);
  if (!uniform)
    throw std::invalid_argument("Cartesian fluence axes must be uniformly increasing");
}

/** `cartesian_fluence_widths` on a line uniform in x (a line grid, or a slab's
    profile in y): the validation, the scale and `_line_fluence_widths`. */
std::array<double, 3> line_widths(const std::vector<double>& x,
                                  const std::vector<double>& values,
                                  const double e2_fraction) {
  const auto finite = [](const double value) { return std::isfinite(value); };
  if (x.size() < 2 || values.size() != x.size() || !std::ranges::all_of(x, finite) ||
      !std::ranges::all_of(values, finite) ||
      std::ranges::any_of(values, [](const double v) { return v < 0.0; }))
    throw std::invalid_argument("Cartesian fluence widths require finite axes "
                                "and matching nonnegative samples");
  require_uniform(x);
  double maximum = values.front();
  for (const double value : values)
    maximum = std::max(maximum, value);
  if (maximum == 0.0)
    return undefined();
  std::vector<double> samples(values.size());
  for (std::size_t index = 0; index < values.size(); ++index)
    samples[index] = values[index] / maximum;
  return line(x, samples, e2_fraction);
}

/** `_ring_fluence_widths`: the 1/e^2 radius and the FWHM of a trace on rings whose
    first is the axis. */
std::array<double, 2> ring_widths(const std::vector<double>& radii,
                                  const std::vector<double>& averaged,
                                  const double e2_fraction) {
  const std::size_t n = radii.size() - 1;
  const std::vector<double> ones(n, 1.0);
  const auto widths = radial(radii.data() + 1, averaged.data() + 1, ones.data(), n,
                             averaged[0], e2_fraction);
  return {widths[1], widths[2]};
}

/** The rings' radii, `spacing * arange(floor(limit / spacing) + 1)`. */
std::vector<double> ring_radii(const double spacing, const double limit) {
  const auto count = static_cast<std::size_t>(std::floor(limit / spacing)) + 1;
  std::vector<double> radii(count);
  for (std::size_t ring = 0; ring < count; ++ring)
    radii[ring] = spacing * static_cast<double>(ring);
  return radii;
}

} // namespace

double pairwise_sum(const double* values, const std::size_t n) {
  if (n < 8) {
    double sum = 0.0;
    for (std::size_t index = 0; index < n; ++index)
      sum += values[index];
    return sum;
  }
  if (n <= pairwise_block) {
    double partial[8];
    for (std::size_t lane = 0; lane < 8; ++lane)
      partial[lane] = values[lane];
    std::size_t index = 8;
    for (; index < n - (n % 8); index += 8)
      for (std::size_t lane = 0; lane < 8; ++lane)
        partial[lane] += values[index + lane];
    double sum = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
                 ((partial[4] + partial[5]) + (partial[6] + partial[7]));
    for (; index < n; ++index)
      sum += values[index];
    return sum;
  }
  std::size_t half = n / 2;
  half -= half % 8;
  return pairwise_sum(values, half) + pairwise_sum(values + half, n - half);
}

std::array<double, 3> radial(const double* radius, const double* fluence,
                             const double* weights, const std::size_t n,
                             const double axis_fluence, const double e2_fraction) {
  bool valid = n >= 2 && std::isfinite(axis_fluence) && axis_fluence >= 0.0;
  for (std::size_t index = 0; valid && index < n; ++index)
    valid = std::isfinite(radius[index]) && std::isfinite(fluence[index]) &&
            std::isfinite(weights[index]) && radius[index] > 0.0 &&
            (index == 0 || radius[index] - radius[index - 1] > 0.0) &&
            weights[index] > 0.0 && !(fluence[index] < 0.0);
  if (!valid)
    throw std::invalid_argument(
        "radial fluence widths require matching finite samples, increasing "
        "positive radii, positive weights and nonnegative fluence");
  std::vector<double> energy_terms(n);
  std::vector<double> moment_terms(n);
  for (std::size_t index = 0; index < n; ++index) {
    const double weighted = weights[index] * fluence[index];
    const double square = radius[index] * radius[index];
    energy_terms[index] = weighted;
    moment_terms[index] = weighted * square;
  }
  const double energy = pairwise_sum(energy_terms.data(), n);
  const double moment = pairwise_sum(moment_terms.data(), n);
  // Nonnegative finite terms: an overflow anywhere leaves a sum infinite, where NumPy
  // raises under `errstate(over="raise")`.
  if (!std::isfinite(energy) || !std::isfinite(moment))
    throw std::invalid_argument("radial fluence moments are not representable");
  if (energy <= 0.0)
    return undefined();
  const double second = std::sqrt(2.0) * std::sqrt(moment / energy);
  double largest = fluence[0];
  for (std::size_t index = 1; index < n; ++index)
    largest = std::max(largest, fluence[index]);
  const double peak = python_max(axis_fluence, largest);
  // The first outward crossing of a fraction of the peak, on the axis value then the
  // samples; undefined where the axis lies below the level or the trace never crosses.
  const auto crossing = [&](const double fraction) {
    const double level = fraction * peak;
    if (axis_fluence < level)
      return nan;
    double left = axis_fluence;
    double left_position = 0.0;
    for (std::size_t index = 0; index < n; ++index) {
      const double right = fluence[index];
      if (right < level) {
        const double share = (left - level) / (left - right);
        return left_position + share * (radius[index] - left_position);
      }
      left = right;
      left_position = radius[index];
    }
    return nan;
  };
  return {second, crossing(e2_fraction), 2 * crossing(0.5)};
}

std::array<double, 3> line(const std::vector<double>& x,
                           const std::vector<double>& samples,
                           const double e2_fraction) {
  const std::size_t n = x.size();
  const double total = pairwise_sum(samples.data(), n);
  std::vector<double> terms(n);
  for (std::size_t index = 0; index < n; ++index)
    terms[index] = samples[index] * x[index];
  const double center = pairwise_sum(terms.data(), n) / total;
  std::vector<double> offset(n);
  double extent = 0.0;
  for (std::size_t index = 0; index < n; ++index) {
    offset[index] = x[index] - center;
    extent =
        index == 0 ? std::abs(offset[0]) : std::max(extent, std::abs(offset[index]));
  }
  // The moment's terms take their own vector, as `radial`'s do: refilling the
  // centroid's buffer left the arm64 builds summing the centroid's terms again.
  std::vector<double> moment_terms(n);
  for (std::size_t index = 0; index < n; ++index) {
    const double ratio = offset[index] / extent;
    moment_terms[index] = samples[index] * (ratio * ratio);
  }
  const double second =
      2 * extent * std::sqrt(pairwise_sum(moment_terms.data(), n) / total);
  if (!std::isfinite(second))
    throw std::invalid_argument("line fluence moments are not representable");
  const double spacing = (x[1] - x[0]) / 2;
  const double limit = python_min(center - x[0], x[n - 1] - center);
  if (limit < 2 * spacing)
    return {second, nan, nan};
  // The plane's azimuthal average has two directions on a line: the linear
  // interpolant at the same half-pixel radii on both sides.
  const auto radii = ring_radii(spacing, limit);
  std::vector<double> averaged(radii.size());
  for (std::size_t ring = 0; ring < radii.size(); ++ring)
    averaged[ring] = 0.5 * (interp(center - radii[ring], x, samples) +
                            interp(center + radii[ring], x, samples));
  const auto [radius, width] = ring_widths(radii, averaged, e2_fraction);
  return {second, radius, width};
}

std::array<double, 3> of_block(const Plan& plan, const std::vector<double>& block) {
  const std::size_t n_nodes = plan.n_nodes;
  if (block.size() != n_nodes + 2)
    throw std::logic_error("a fluence block holds the nodes, the axis and the size");
  const double axis_fluence = block[n_nodes];
  const double limit = budget(plan.width_terms, block[n_nodes + 1]);
  // `numpy.any(fluence < -budget)`, then `numpy.argmin(fluence + budget)`: the first
  // NaN, or the first of the lowest.
  bool refused = false;
  for (std::size_t node = 0; node < n_nodes; ++node)
    refused = refused || block[node] < -limit;
  if (refused) {
    std::size_t worst = 0;
    for (std::size_t node = 0; node < n_nodes; ++node) {
      const double value = block[node] + limit;
      if (std::isnan(value)) {
        worst = node;
        break;
      }
      if (value < block[worst] + limit)
        worst = node;
    }
    throw negative_fluence(block[worst], worst);
  }
  // `numpy.maximum(fluence, 0.0)`, which a NaN keeps.
  std::vector<double> fluence(block.begin(),
                              block.begin() + static_cast<std::ptrdiff_t>(n_nodes));
  for (double& value : fluence)
    value = std::isnan(value) ? value : std::max(value, 0.0);
  switch (plan.widths) {
  case Widths::radial:
    return radial(plan.width_x.data(), fluence.data(), plan.width_weights.data(),
                  n_nodes, axis_fluence, plan.e2_fraction);
  case Widths::slab: {
    // `_slab_widths`: the weighted fluence summed over x on each row of y.
    const std::size_t n_x = plan.width_n_x;
    std::vector<double> weighted(n_nodes);
    for (std::size_t node = 0; node < n_nodes; ++node)
      weighted[node] = plan.width_weights[node] * fluence[node];
    std::vector<double> across(plan.width_n_y);
    for (std::size_t row = 0; row < across.size(); ++row)
      across[row] = pairwise_sum(weighted.data() + row * n_x, n_x);
    return line_widths(plan.width_x, across, plan.e2_fraction);
  }
  case Widths::line:
    return line_widths(plan.width_x, fluence, plan.e2_fraction);
  case Widths::plane:
  case Widths::none:
    break;
  }
  throw std::logic_error("a fluence block's widths need a radial, slab or line rule");
}

std::array<double, 3> of_plane(GeneralState& state, const Plan& plan,
                               const std::vector<double>& moments) {
  if (moments.size() != 8)
    throw std::logic_error("a plane's fluence moments are eight values");
  const double worst = moments[0];
  const double node = moments[1];
  const double size = moments[2];
  const double maximum = moments[3];
  const double nonfinite = moments[4];
  const double total = moments[5];
  const double moment_x = moments[6];
  const double moment_y = moments[7];
  if (nonfinite != 0.0)
    throw std::invalid_argument("Cartesian fluence widths require finite axes "
                                "and matching nonnegative samples");
  if (worst < -budget(plan.width_terms, size))
    throw negative_fluence(worst, static_cast<std::size_t>(node));
  if (maximum == 0.0)
    return undefined();
  const auto& x = plan.width_x;
  const auto& y = plan.width_y;
  const double dx = x[1] - x[0];
  const double dy = y[1] - y[0];
  const double center_x = x.front() + dx * (moment_x / total);
  const double center_y = y.front() + dy * (moment_y / total);
  const double extent =
      std::hypot(python_max(center_x - x.front(), x.back() - center_x),
                 python_max(center_y - y.front(), y.back() - center_y));
  // `_fluence_rings`: half the finest spacing in radius, out to the largest circle
  // about the centroid inside the axes; none within two half-spacings of an edge.
  const double limit =
      python_min(python_min(python_min(center_x - x.front(), x.back() - center_x),
                            center_y - y.front()),
                 y.back() - center_y);
  const double spacing = python_min(dx, dy) / 2;
  const bool rings = !(limit < 2 * spacing);
  const std::vector<double> none;
  const auto radii = rings ? ring_radii(spacing, limit) : std::vector<double>{};
  const auto values = state.plane_fluence_trace(
      center_x - x.front(), center_y - y.front(), dx, dy, extent,
      rings ? radii[1] : 0.0, radii.size(), rings ? plan.ring_cosines : none,
      rings ? plan.ring_sines : none);
  const double second = extent * std::sqrt(2 * values[0] / total);
  if (!std::isfinite(second))
    throw std::invalid_argument("Cartesian fluence moments are not representable");
  if (!rings)
    return {second, nan, nan};
  const std::vector<double> averaged(values.begin() + 1, values.end());
  const auto [radius, width] = ring_widths(radii, averaged, plan.e2_fraction);
  return {second, radius, width};
}

double strict_probe(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}

} // namespace solprop_native::statistics::widths
