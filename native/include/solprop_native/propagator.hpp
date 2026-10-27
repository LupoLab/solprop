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
 * `exp(x) (cos theta, sin theta)` in binary64, as every route has formed it. In
 * binary32 each component of that binary64 value is rounded once, as the oracle rounds
 * `exp(dPhi)` (`solprop.backends.numpy_ref.single`): within `u32` per component, and
 * unbiased.
 *
 * Rounding once matters more than the bound. A step multiplies each bin by its
 * propagator, and on a constant operator by the same rounded value every step, so the
 * propagator's own error adds coherently, step after step (SEP 33's `c M u32`), and a
 * mean bias of its squared modulus is an energy drift; Kerr phase and Raman gain turn
 * both into field errors growing faster than the step count. Two forms that meet a
 * `10 u32` bound did worse than one rounding: binary32 `sinf` and `cosf` on CUDA
 * (`cos^2 + sin^2 - 1` averages `+0.30 u32` on an A40), and rounding the modulus and
 * the phasor separately before their binary32 product (two roundings per component:
 * on a hydrogen capillary at a constant step the field's error grew by `0.5 u32` per
 * step against the oracle's `0.3 u32`; with one rounding the two agree).
 */

namespace solprop_native {

namespace propagator_detail {
/**
 * `cos theta` and `sin theta` in binary64. On a device they are one `sincos` call,
 * which shares the argument's reduction between the two (solprop SEP 36: on a GPU of
 * reduced binary64 rate the stage propagators are about half of the stage entry); the
 * host keeps the two library calls.
 */
KOKKOS_INLINE_FUNCTION void cosine_sine(const double theta, double& cosine,
                                        double& sine) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  ::sincos(theta, &sine, &cosine);
#else
  cosine = Kokkos::cos(theta);
  sine = Kokkos::sin(theta);
#endif
}
} // namespace propagator_detail

/** `exp(argument)` in the precision `Real`, from a binary64 argument. */
template <class Real>
KOKKOS_INLINE_FUNCTION Kokkos::complex<Real>
propagator_exp(const Kokkos::complex<double> argument) {
  static_assert(std::is_same_v<Real, double> || std::is_same_v<Real, float>);
  if constexpr (std::is_same_v<Real, double>) {
    const double magnitude = Kokkos::exp(argument.real());
    double cosine = 0.0;
    double sine = 0.0;
    propagator_detail::cosine_sine(argument.imag(), cosine, sine);
    return {magnitude * cosine, magnitude * sine};
  } else {
    const auto wide = propagator_exp<double>(argument);
    return {static_cast<float>(wide.real()), static_cast<float>(wide.imag())};
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
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // A stage at the step's anchor has a zero phase: its propagators are exactly the
    // library's values there, `(1, 0)` and `(1, -0)`, without the four evaluations.
    if (argument.real() == 0.0 && argument.imag() == 0.0)
      return {{1.0, 0.0}, {1.0, -0.0}};
#endif
    double cosine = 0.0;
    double sine = 0.0;
    propagator_detail::cosine_sine(argument.imag(), cosine, sine);
    const double magnitude = Kokkos::exp(argument.real());
    const double inverse_magnitude = Kokkos::exp(-argument.real());
    return {{magnitude * cosine, magnitude * sine},
            {inverse_magnitude * cosine, -(inverse_magnitude * sine)}};
  } else {
    const auto wide = propagator_pair<double>(argument);
    return {{static_cast<float>(wide.forward.real()),
             static_cast<float>(wide.forward.imag())},
            {static_cast<float>(wide.inverse.real()),
             static_cast<float>(wide.inverse.imag())}};
  }
}

} // namespace solprop_native
