// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The propagator exponentials: exponential, cosine_sine (one `sincos` on a device),
// flow_magnitude (`exp(+-0)` exactly one on a lossless bin, out of line on a host),
// propagator_exp and entry_exponential; and squared_abs. A part of kernels.hpp,
// included through it.
#ifdef SOLPROP_NATIVE_STRICT_TU
#error "a strict translation unit must not include general/exponentials.hpp"
#endif

#include "solprop_native/runtime/execution.hpp"

namespace solprop_native::general {

KOKKOS_INLINE_FUNCTION Complex exponential(const Complex value) {
  const double magnitude = Kokkos::exp(value.real());
  return {magnitude * Kokkos::cos(value.imag()), magnitude * Kokkos::sin(value.imag())};
}

// `cos theta` and `sin theta`. On a device they are one `sincos` call, which shares
// the argument's reduction between the two (on a GPU of reduced binary64 rate the
// stage propagators are about half of the stage entry); the host keeps the two
// library calls, so its values are unchanged.
KOKKOS_INLINE_FUNCTION void cosine_sine(const double theta, double& cosine,
                                        double& sine) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  ::sincos(theta, &sine, &cosine);
#else
  cosine = Kokkos::cos(theta);
  sine = Kokkos::sin(theta);
#endif
}

// A flow's magnitude: on a lossless bin `exp(+-0)`, exactly one, taken without a
// binary64 `exp` per element. Out of line on a host, whose call to its library was out
// of line already: inlined there, this branch took the tabulated fold's whole kernel
// past the compiler's inlining limit, 40 % slower (solprop's docs/dev/benchmarks.md).
#if defined(__GNUC__) && !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
[[gnu::noinline]]
#endif
KOKKOS_INLINE_FUNCTION double flow_magnitude(const double real) {
  return real == 0.0 ? 1.0 : Kokkos::exp(real);
}

// A stage or step propagator `exp(argument)`: `exponential` with the cosine and sine
// from `cosine_sine`, so it is `exponential` to the bit on the host.
KOKKOS_INLINE_FUNCTION Complex propagator_exp(const Complex argument) {
  const double magnitude = flow_magnitude(argument.real());
  double cosine = 0.0;
  double sine = 0.0;
  cosine_sine(argument.imag(), cosine, sine);
  return {magnitude * cosine, magnitude * sine};
}

// `exponential`, or on a lossless problem (a signed-zero real part) the same value
// with its magnitude `exp(+-0)`, exactly one, taken without the call. The cosine and
// sine come from `cosine_sine`, so on the host it is `exponential` to the bit.
template <bool Lossless>
KOKKOS_INLINE_FUNCTION Complex entry_exponential(const Complex value) {
  if constexpr (Lossless) {
    double cosine = 0.0;
    double sine = 0.0;
    cosine_sine(value.imag(), cosine, sine);
    return {cosine, sine};
  } else {
    return propagator_exp(value);
  }
}

KOKKOS_INLINE_FUNCTION double squared_abs(const Complex value) {
  return value.real() * value.real() + value.imag() * value.imag();
}

} // namespace solprop_native::general
