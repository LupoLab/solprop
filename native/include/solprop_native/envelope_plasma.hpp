// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Kokkos_Array.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include <cstddef>
#include <numbers>

namespace solprop_native {

/** Scaled principal vectors of a physical Cartesian V/m envelope. */
struct EnvelopeAxes {
  double major{}, ratio{}, cosine{1.0}, sine{};
  double ux{}, uy{}, vx{}, vy{}, scale{}, major_square{1.0};
};

/** Resolve the ellipse with the operation order of solprop's shared body. */
KOKKOS_INLINE_FUNCTION EnvelopeAxes envelope_axes(
    const Kokkos::complex<double> first, const Kokkos::complex<double> second) {
  const double scale =
      Kokkos::max(Kokkos::max(Kokkos::abs(first.real()), Kokkos::abs(first.imag())),
                  Kokkos::max(Kokkos::abs(second.real()), Kokkos::abs(second.imag())));
  if (scale == 0.0)
    return {};
  const double xr = first.real() / scale, xi = first.imag() / scale;
  const double yr = second.real() / scale, yi = second.imag() / scale;
  const double xx = xr * xr + yr * yr, yy = xi * xi + yi * yi;
  const double xy = xr * xi + yr * yi;
  // A scaled norm avoids an underflow observed in the device hypot.
  const double larger = Kokkos::max(Kokkos::abs(xx - yy), Kokkos::abs(2 * xy));
  const double smaller = Kokkos::min(Kokkos::abs(xx - yy), Kokkos::abs(2 * xy));
  const double norm_ratio = larger == 0.0 ? 0.0 : smaller / larger;
  const double norm = larger * Kokkos::sqrt(1.0 + norm_ratio * norm_ratio);
  const double major_square = 0.5 * (xx + yy + norm);
  const double determinant = xr * yi - xi * yr;
  const double ratio = Kokkos::min(1.0, Kokkos::abs(determinant) / major_square);
  const double angle = 0.5 * Kokkos::atan2(2 * xy, xx - yy);
  const double cosine = Kokkos::cos(angle), sine = Kokkos::sin(angle);
  const double ux = xr * cosine + xi * sine, uy = yr * cosine + yi * sine;
  // The determinant retains the small axis near a degenerate linear ellipse.
  const double signed_ratio = determinant / major_square;
  return {scale * Kokkos::sqrt(major_square),
          ratio,
          cosine,
          sine,
          ux,
          uy,
          -signed_ratio * uy,
          signed_ratio * ux,
          scale,
          major_square};
}

/** Positive directional cycle moments in s^-1 and a refinement failure flag. */
struct EnvelopeMoments {
  double cosine{}, sine{};
  bool failed{};
};

/**
 * Refine the accepted split 64/128/256/512 quadrant rules.
 *
 * Rate and quadrature are compiled numeric inputs; no atomic constants live here.
 * The relative 2e-6 moment budget and 1e-30 s^-1 comparison floor are the frozen
 * schema-9 cycle representation, independently validated during translation.
 */
template <class Rate, class RuleView>
KOKKOS_INLINE_FUNCTION EnvelopeMoments
envelope_moments(const double major, const double ratio, const double field_min,
                 const double field_max, const Rate& rate_at, const RuleView& nodes,
                 const RuleView& weights) {
  if (major < field_min)
    return {};
  if (ratio == 1.0) {
    const double rate = 0.5 * rate_at(major);
    return {rate, rate, false};
  }
  if (!Kokkos::isfinite(major) || !Kokkos::isfinite(ratio))
    return {0.0, 0.0, true};
  constexpr double half_pi = std::numbers::pi_v<double> / 2.0;
  const double high = field_max / major, low = field_min / major;
  const double high_edge =
      high >= 1.0     ? 0.0
      : high <= ratio ? half_pi
                      : Kokkos::acos(Kokkos::sqrt((high * high - ratio * ratio) /
                                                  (1 - ratio * ratio)));
  const double low_edge = low >= 1.0 ? 0.0
                          : low <= ratio
                              ? half_pi
                              : Kokkos::acos(Kokkos::sqrt((low * low - ratio * ratio) /
                                                          (1 - ratio * ratio)));
  double previous_c = 0.0, previous_s = 0.0;
  for (std::size_t level = 0; level < 4; ++level) {
    const std::size_t order = std::size_t{64} << level;
    double mc = 0.0, ms = 0.0;
    for (std::size_t panel = 0; panel < 3; ++panel) {
      const double lower = panel == 0 ? 0.0 : panel == 1 ? high_edge : low_edge;
      const double upper = panel == 0 ? high_edge : panel == 1 ? low_edge : half_pi;
      const double width = upper - lower;
      if (width == 0.0)
        continue;
      for (std::size_t index = 0; index < order; ++index) {
        const double theta = lower + (nodes(level, index) + 1) * (0.5 * width);
        const double cosine = Kokkos::cos(theta), sine = Kokkos::sin(theta);
        const double c2 = cosine * cosine, s2 = sine * sine;
        const double denominator = c2 + ratio * ratio * s2;
        const double rate = rate_at(major * Kokkos::sqrt(denominator)) / denominator;
        const double weight =
            (width / std::numbers::pi_v<double>)*weights(level, index);
        mc += weight * rate * c2;
        ms += weight * rate * s2;
      }
    }
    if (level > 0 && Kokkos::isfinite(mc) && Kokkos::isfinite(ms) &&
        Kokkos::abs(mc - previous_c) <= 2e-6 * Kokkos::max(mc, 1e-30) &&
        Kokkos::abs(ms - previous_s) <= 2e-6 * Kokkos::max(ms, 1e-30))
      return {mc, ms, false};
    previous_c = mc;
    previous_s = ms;
  }
  return {previous_c, previous_s, true};
}

/**
 * The cycle moments on a binary32 quadrature (solprop SEP 36 section 6.5).
 *
 * The rule's angles, their cosines and sines and the field at each node are binary32;
 * each node's rate is the binary64 table lookup rounded once (section 4.5), and the
 * level sums accumulate in binary64, refined to the same budget as `envelope_moments`
 * (2e-6 relative, floor 1e-30 s^-1). SEP 33's device lessons hold: the cut angles,
 * formed in binary64, keep their meaning -- before the high cut the field is above
 * the table and the rate is its ceiling, past the low cut it is below and the panel
 * is skipped -- and between them the field is held inside the table's range, so that
 * binary32's rounding of an angle cannot put a node on the wrong side of a cutoff.
 */
template <class Rate, class RuleView>
KOKKOS_INLINE_FUNCTION EnvelopeMoments envelope_moments_binary32(
    const double major, const double ratio, const double field_min,
    const double field_max, const double rate_max, const Rate& rate_at,
    const RuleView& nodes, const RuleView& weights) {
  if (major < field_min)
    return {};
  if (ratio == 1.0) {
    const double rate = 0.5 * rate_at(major);
    return {rate, rate, false};
  }
  if (!Kokkos::isfinite(major) || !Kokkos::isfinite(ratio))
    return {0.0, 0.0, true};
  constexpr double half_pi = std::numbers::pi_v<double> / 2.0;
  const double high = field_max / major, low = field_min / major;
  const double high_edge =
      high >= 1.0     ? 0.0
      : high <= ratio ? half_pi
                      : Kokkos::acos(Kokkos::sqrt((high * high - ratio * ratio) /
                                                  (1 - ratio * ratio)));
  const double low_edge = low >= 1.0 ? 0.0
                          : low <= ratio
                              ? half_pi
                              : Kokkos::acos(Kokkos::sqrt((low * low - ratio * ratio) /
                                                          (1 - ratio * ratio)));
  const auto major_32 = static_cast<float>(major);
  const auto ratio_32 = static_cast<float>(ratio);
  const auto ceiling = static_cast<float>(rate_max);
  double previous_c = 0.0, previous_s = 0.0;
  for (std::size_t level = 0; level < 4; ++level) {
    const std::size_t order = std::size_t{64} << level;
    double mc = 0.0, ms = 0.0;
    for (std::size_t panel = 0; panel < 2; ++panel) {
      const double lower = panel == 0 ? 0.0 : high_edge;
      const double upper = panel == 0 ? high_edge : low_edge;
      const double width = upper - lower;
      if (!(width > 0.0))
        continue;
      const auto lower_32 = static_cast<float>(lower);
      const auto half_width = static_cast<float>(0.5 * width);
      const auto panel_weight = static_cast<float>(width / std::numbers::pi_v<double>);
      for (std::size_t index = 0; index < order; ++index) {
        const float theta =
            lower_32 + (static_cast<float>(nodes(level, index)) + 1.0F) * half_width;
        const float cosine = Kokkos::cos(theta), sine = Kokkos::sin(theta);
        const float c2 = cosine * cosine, s2 = sine * sine;
        const float denominator = c2 + ratio_32 * ratio_32 * s2;
        float rate = ceiling;
        if (panel == 1) {
          const double field =
              Kokkos::clamp(static_cast<double>(major_32 * Kokkos::sqrt(denominator)),
                            field_min, field_max);
          rate = static_cast<float>(rate_at(field));
        }
        const float term = panel_weight * static_cast<float>(weights(level, index)) *
                           (rate / denominator);
        mc += static_cast<double>(term * c2);
        ms += static_cast<double>(term * s2);
      }
    }
    if (level > 0 && Kokkos::isfinite(mc) && Kokkos::isfinite(ms) &&
        Kokkos::abs(mc - previous_c) <= 2e-6 * Kokkos::max(mc, 1e-30) &&
        Kokkos::abs(ms - previous_s) <= 2e-6 * Kokkos::max(ms, 1e-30))
      return {mc, ms, false};
    previous_c = mc;
    previous_s = ms;
  }
  return {previous_c, previous_s, true};
}

/** Reconstruct the two loss-current components in A/m^2 from work in J/m^3. */
KOKKOS_INLINE_FUNCTION Kokkos::Array<Kokkos::complex<double>, 2>
envelope_loss(const EnvelopeAxes& axes, const double mc, const double ms,
              const double work) {
  if (work == 0.0 || (mc == 0.0 && ms == 0.0))
    return {};
  const Kokkos::complex<double> major_phase(axes.cosine, axes.sine);
  const Kokkos::complex<double> minor_phase(-axes.sine, axes.cosine);
  // Dividing moments first avoids an overflowing standalone inverse tiny scale.
  const double major_moment = mc / axes.scale, minor_moment = ms / axes.scale;
  const double factor = 2 * work / axes.major_square;
  return {factor * (major_moment * axes.ux * major_phase +
                    minor_moment * axes.vx * minor_phase),
          factor * (major_moment * axes.uy * major_phase +
                    minor_moment * axes.vy * minor_phase)};
}

} // namespace solprop_native
