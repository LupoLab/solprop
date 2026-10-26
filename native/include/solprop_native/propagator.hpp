// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include <type_traits>

/*
 * The propagator `exp(x + i theta)` in a policy's propagator precision (solprop SEP 36
 * section 4.3).
 *
 * The argument is binary64 under every policy. In binary64 the propagator is the
 * library's `exp(x) (cos theta, sin theta)`, as every route has formed it. In binary32
 * a binary64-accurate argument is rounded once only after both of its parts are
 * range-reduced in binary64, so that rounding the argument itself costs nothing:
 *
 * - the modulus as `exp(x) = 2^m exp(r)`, with `m = round(x / ln 2)` and `r = x - m
 *   ln 2` formed with a two-part `ln 2` (Cody-Waite), so `|r| <= ln(2) / 2`; `expf(r)`
 *   is then scaled by `2^m` exactly, and overflow and underflow are decided by `m`;
 * - the phase as `theta = 2 pi q + rho`, `rho` formed with a three-part `2 pi` whose
 *   leading parts are short enough that `q` times each is exact for `|theta| <= 2^30`,
 *   which leaves an absolute error in `rho` below `1e-14`, then `sinf` and `cosf` of
 *   `rho`. Above `2^30` the binary64 library's `sin` and `cos`, exact in their own
 *   reduction, are rounded once instead.
 *
 * The binary32 functions are the libraries' `expf`, `sinf` and `cosf` (at most 2 ulp on
 * CUDA and HIP, 1 ulp on the CPU), never fast intrinsics. With the two products of the
 * modulus and phasor the propagator's relative error is at most `10 u32`
 * (tests/native/test_propagator.cpp checks it at the ends of both ranges).
 */

namespace solprop_native {

namespace propagator_detail {
// `ln 2` in two parts, the first with 29 significant bits: `m` times it is exact for
// every `|m| <= 2^24`, far beyond the `|m| <= 1478` that `|x| <= 1024` reaches.
inline constexpr double ln2_hi = 0x1.62e42ffp-1;
inline constexpr double ln2_lo = -0x1.718432a1b0e26p-35;
inline constexpr double inverse_ln2 = 0x1.71547652b82fep+0;
// `2 pi` in three parts, the first two with 25 significant bits: `q` times each is
// exact for `|q| <= 2^28`, which covers `|theta| <= 2^30`. The three sum to `2 pi`
// within 5e-32 relative, so `q` times their error stays below 1e-22 rad.
inline constexpr double two_pi_1 = 0x1.921fb6p+2;
inline constexpr double two_pi_2 = -0x1.777a5cp-23;
inline constexpr double two_pi_3 = -0x1.ee59d9cceba40p-48;
inline constexpr double inverse_two_pi = 0x1.45f306dc9c883p-3;
inline constexpr double reduced_phase_limit = 0x1p30;

/** `exp(x)` in binary32 from a binary64 `x`, reduced by powers of two first. */
KOKKOS_INLINE_FUNCTION float modulus(const double x) {
  if (!(x == x))
    return static_cast<float>(x);
  // Far beyond binary32's range the result is decided without the reduction, whose
  // integer would overflow.
  if (x > 1024.0)
    return Kokkos::Experimental::infinity_v<float>;
  if (x < -1024.0)
    return 0.0F;
  const double m = Kokkos::nearbyint(x * inverse_ln2);
  const double r = (x - m * ln2_hi) - m * ln2_lo;
  return Kokkos::ldexp(Kokkos::exp(static_cast<float>(r)), static_cast<int>(m));
}

/** `(cos theta, sin theta)` in binary32 from a binary64 `theta`. */
KOKKOS_INLINE_FUNCTION Kokkos::complex<float> phasor(const double theta) {
  if (!(Kokkos::abs(theta) <= reduced_phase_limit))
    return {static_cast<float>(Kokkos::cos(theta)),
            static_cast<float>(Kokkos::sin(theta))};
  const double q = Kokkos::nearbyint(theta * inverse_two_pi);
  const double rho = ((theta - q * two_pi_1) - q * two_pi_2) - q * two_pi_3;
  const auto reduced = static_cast<float>(rho);
  return {Kokkos::cos(reduced), Kokkos::sin(reduced)};
}
} // namespace propagator_detail

/** `exp(argument)` in the precision `Real`, from a binary64 argument. */
template <class Real>
KOKKOS_INLINE_FUNCTION Kokkos::complex<Real>
propagator_exp(const Kokkos::complex<double> argument) {
  static_assert(std::is_same_v<Real, double> || std::is_same_v<Real, float>);
  if constexpr (std::is_same_v<Real, double>) {
    const double magnitude = Kokkos::exp(argument.real());
    return {magnitude * Kokkos::cos(argument.imag()),
            magnitude * Kokkos::sin(argument.imag())};
  } else {
    const float magnitude = propagator_detail::modulus(argument.real());
    const Kokkos::complex<float> unit = propagator_detail::phasor(argument.imag());
    return {magnitude * unit.real(), magnitude * unit.imag()};
  }
}

/**
 * `exp(argument)` and `exp(-argument)` in the precision `Real`, sharing the phase's
 * cosine and sine: the inverse is `e^{-x} (cos theta, -sin theta)` with its own
 * modulus, never a reciprocal of the forward value. In binary64 these are the stage
 * combination's former expressions to the bit.
 */
template <class Real> struct PropagatorPair {
  Kokkos::complex<Real> forward, inverse;
};

template <class Real>
KOKKOS_INLINE_FUNCTION PropagatorPair<Real>
propagator_pair(const Kokkos::complex<double> argument) {
  static_assert(std::is_same_v<Real, double> || std::is_same_v<Real, float>);
  if constexpr (std::is_same_v<Real, double>) {
    const double cosine = Kokkos::cos(argument.imag());
    const double sine = Kokkos::sin(argument.imag());
    const double magnitude = Kokkos::exp(argument.real());
    const double inverse_magnitude = Kokkos::exp(-argument.real());
    return {{magnitude * cosine, magnitude * sine},
            {inverse_magnitude * cosine, -(inverse_magnitude * sine)}};
  } else {
    const Kokkos::complex<float> unit = propagator_detail::phasor(argument.imag());
    const float magnitude = propagator_detail::modulus(argument.real());
    const float inverse_magnitude = propagator_detail::modulus(-argument.real());
    return {{magnitude * unit.real(), magnitude * unit.imag()},
            {inverse_magnitude * unit.real(), -(inverse_magnitude * unit.imag())}};
  }
}

} // namespace solprop_native
