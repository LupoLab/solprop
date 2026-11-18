// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The plasma's pointwise maths: the trapezoid increment, the ionised fractions
// (decayed_fraction and its host block form, ionised_fraction, envelope_fraction and
// band_fraction, strict where they round a small response), band_loss_scale, and the
// rate table's cubic (ionisation_log_rate, ionisation_rate, and the host's vectorised
// ionisation_rates_within over rate_block samples). compensation_residual, a multistep
// compensation's strict residual, sits among them, in kernels.hpp's order. A part of
// kernels.hpp, included through it.
#ifdef SOLPROP_NATIVE_STRICT_TU
#error "a strict translation unit must not include general/ionisation.hpp"
#endif

#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/fp.hpp"
#include "solprop_native/runtime/vmath.hpp"

#include <cstddef>

namespace solprop_native::general {

KOKKOS_INLINE_FUNCTION double trapezoid(const double current, const double previous,
                                        const double dt) {
  return dt * (current + previous) / 2.0;
}

// The fraction ionised from an exposure's decay `exp(-X)`, `1 - (1 - p) exp(-X)`, as
// solprop's `numpy_ref.passes.plasma.plasma_scan` rounds it. A small response: strict
// arithmetic (solprop SEP 25 section 4.2, `fp.hpp`), so the product and the difference
// round separately.
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION double decayed_fraction(const double decay,
                                               const double preionisation) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const double survival = (1.0 - preionisation) * decay;
  return 1.0 - survival;
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
// `decayed_fraction` of each of a host block's `count` decays, in place: one strict
// loop that vectorises, each product and difference rounded as there.
SOLPROP_NATIVE_STRICT_FUNCTION
inline void decayed_fractions(double* values, const std::size_t count,
                              const double preionisation) {
  SOLPROP_NATIVE_STRICT_BLOCK
  for (std::size_t k = 0; k < count; ++k) {
    const double survival = (1.0 - preionisation) * values[k];
    values[k] = 1.0 - survival;
  }
}
#endif

// The fraction ionised from an exposure, `decayed_fraction` of its `exp(-X)`.
KOKKOS_INLINE_FUNCTION double ionised_fraction(const double exposure,
                                               const double preionisation) {
  return decayed_fraction(Kokkos::exp(-exposure), preionisation);
}

// The envelope column's fraction, `p + (1 - p) (-expm1(-X))`, strict as above.
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION double envelope_fraction(const double exposure,
                                                const double preionisation) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const double ionised = (1.0 - preionisation) * (-Kokkos::expm1(-exposure));
  return preionisation + ionised;
}

// The band rule's fraction: the exposure floored at zero where the fraction is formed
// (SEP 18 section 4.5), `max(X, 0)` as NumPy's `maximum` takes it, so a NaN exposure
// stays NaN rather than becoming the preionisation.
KOKKOS_INLINE_FUNCTION double band_fraction(const double exposure,
                                            const double preionisation) {
  return ionised_fraction(exposure < 0.0 ? 0.0 : exposure, preionisation);
}

// A multistep compensation's residual term `(D - x / dz) u`: the difference of two
// nearly equal rates, strict (solprop SEP 25 section 4.2), each complex operation
// spelt with `fp.hpp`'s helpers.
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION Complex compensation_residual(const Complex rate,
                                                     const Complex exponent,
                                                     const double inverse_dz,
                                                     const Complex state) {
  SOLPROP_NATIVE_STRICT_BLOCK
  return strict_mul(strict_sub(rate, strict_scale(inverse_dz, exponent)), state);
}

// The loss current's coefficient of the field, or zero, as solprop's
// `numpy_ref.passes.plasma.plasma_total_current` forms it.
KOKKOS_INLINE_FUNCTION double band_loss_scale(const double rate, const double fraction,
                                              const double squared,
                                              const double ionisation_potential) {
  if (rate > 0.0 && squared > 0.0)
    return ionisation_potential * rate * (1.0 - fraction) / squared;
  return 0.0;
}

// The rate table's cubic in `log(rate)` at a field's `log |E|` within the table's
// range: the interval's local coordinate and its Horner form (`ionisation_rate`).
template <class View>
KOKKOS_INLINE_FUNCTION double
ionisation_log_rate(const double log_field, const View& coefficients,
                    const double log_field_min, const double d_log_field,
                    const std::size_t intervals, const std::size_t offset = 0) {
  long raw_index = static_cast<long>((log_field - log_field_min) / d_log_field);
  if (raw_index < 0)
    raw_index = 0;
  // NOLINTNEXTLINE(modernize-use-integer-sign-comparison): device code; std::cmp_* is host-only
  if (raw_index >= static_cast<long>(intervals))
    raw_index = static_cast<long>(intervals - 1);
  const auto index = static_cast<std::size_t>(raw_index);
  const double local =
      log_field - (log_field_min + static_cast<double>(index) * d_log_field);
  return ((coefficients(offset + index) * local +
           coefficients(offset + intervals + index)) *
              local +
          coefficients(offset + 2 * intervals + index)) *
             local +
         coefficients(offset + 3 * intervals + index);
}

// The ionisation rate at a field magnitude: the table's limits outside its range, and
// within it the exponential of its cubic in `log |E|`.
template <class View>
KOKKOS_INLINE_FUNCTION double
ionisation_rate(const double magnitude, const View& coefficients,
                const double log_field_min, const double d_log_field,
                const double field_min, const double field_max, const double rate_max,
                const std::size_t intervals, const std::size_t offset = 0) {
  if (magnitude > field_max)
    return rate_max;
  if (magnitude < field_min)
    return 0.0;
  return Kokkos::exp(ionisation_log_rate(Kokkos::log(magnitude), coefficients,
                                         log_field_min, d_log_field, intervals,
                                         offset));
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
// The samples a host pass of the plasma column takes at a time.
constexpr std::size_t rate_block = 256;

// `ionisation_rate`'s values at `live <= rate_block` magnitudes within the table's
// range, in place: one `log` and one `exp` pass of the vector-maths layer (solprop
// ADR-059) around the cubic. A host block gathers the magnitudes it passes, so the
// samples outside the range (most of a pulse's wings) cost what they cost before.
template <class View>
void ionisation_rates_within(double* values, const std::size_t live,
                             const View& coefficients, const double log_field_min,
                             const double d_log_field, const std::size_t intervals,
                             const std::size_t offset = 0) {
  // The cubic writes a local block, which no table gather can alias, so the loop
  // vectorises with the table's gathers wherever this is compiled out of line.
  double log_rates[rate_block];
  vmath::log(values, values, live);
  for (std::size_t k = 0; k < live; ++k)
    log_rates[k] = ionisation_log_rate(values[k], coefficients, log_field_min,
                                       d_log_field, intervals, offset);
  vmath::exp(log_rates, values, live);
}
#endif

} // namespace solprop_native::general
