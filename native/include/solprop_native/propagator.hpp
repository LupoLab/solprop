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
 * The argument is binary64 under every policy, and so is the evaluation: the library's
 * `exp(x) (cos theta, sin theta)` in binary64, as every route has formed it, and in
 * binary32 the modulus and the phasor's two parts each rounded once from it. Their
 * errors are then round-to-nearest's, unbiased, and the propagator's relative error is
 * at most `3 u32` with the product's rounding.
 *
 * Rounding unbiased matters more than the bound: a unit-modulus propagator multiplies
 * the state once per step, so any mean bias of its squared modulus becomes an energy
 * drift linear in the step count, which Kerr phase turns into a field error growing
 * faster than linearly. CUDA's binary32 `sinf` and `cosf` are within 2 ulp but not
 * unbiased: on an A40, `cosf(r)^2 + sinf(r)^2 - 1` averages `+0.30 u32` over the phases
 * (the CPU libraries' and the rounded binary64 functions' average 0.00), which cost an
 * ionising argon capillary 25 times its CPU error in the ionised fraction after 2500
 * steps. The binary64 functions cost a few binary64 operations per bin and stage, far
 * below a step's transforms even on a GPU with a reduced binary64 rate.
 */

namespace solprop_native {

namespace propagator_detail {
/** `exp(x)` in binary32: the binary64 library's value, rounded once. */
KOKKOS_INLINE_FUNCTION float modulus(const double x) {
  return static_cast<float>(Kokkos::exp(x));
}

/** `(cos theta, sin theta)` in binary32: the binary64 library's, each rounded once. */
KOKKOS_INLINE_FUNCTION Kokkos::complex<float> phasor(const double theta) {
  return {static_cast<float>(Kokkos::cos(theta)),
          static_cast<float>(Kokkos::sin(theta))};
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
