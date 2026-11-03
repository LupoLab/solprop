// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Kokkos_Core.hpp>

/*
 * Floating-point contraction policy (solprop SEP 25, sections 4.2 and 5).
 *
 * CPU builds compile with contraction on across statements (`-ffp-contract=fast` for
 * GCC, `-ffp-contract=fast-honor-pragmas` for Clang and icpx; CMakeLists.txt), so the
 * compiler may fuse any multiply and add into one rounding. A protected block keeps
 * the arithmetic as written, each product and sum rounded on its own: the stable phase
 * increments, the small-argument phi-functions, the ionised fraction (every column's
 * `1 - (1 - p) e^{-X}` and the envelope's `p + (1 - p) (-expm1(-X))`, in the kernels
 * and in the statistics), and the multistep defects and their compensation (the
 * residual terms `(D - x / dz) u` and a coupled flow's `R_s u`, the compensation sums,
 * their addition to the predicted, corrected and observed solutions, and the endpoint
 * rate). Contraction there is not inaccurate as such; these are the formulas whose
 * small results the policy keeps under strict arithmetic rather than requalifying.
 * The finite-value guards are comparisons, which contraction cannot change; what keeps
 * them meaningful is the absence of `nnan`/`ninf` and fast-math from every build, not
 * a protected block.
 *
 * A protected function or lambda carries `SOLPROP_NATIVE_STRICT_FUNCTION` (before a
 * function's declaration, after a lambda's parameter list) and starts its body with
 * `SOLPROP_NATIVE_STRICT_BLOCK`:
 *
 * - Clang and icpx: `#pragma clang fp contract(off)` at the start of the body. The
 *   setting is carried by each instruction, so it survives inlining.
 * - GCC ignores `#pragma STDC FP_CONTRACT` and Clang's pragma, so the function is
 *   compiled with `optimize("fp-contract=off")`. GCC then does not inline it into a
 *   caller compiled with other options: a protected helper is a call there.
 * - Device compilers (CUDA, HIP) keep their own defaults, as their builds do: the
 *   blocks are empty there, nvcc's host pass included, and device accuracy is held to
 *   the measured device envelopes (solprop's ADR-049, amended). Round-to-nearest
 *   intrinsics with a device fusion test are solprop issue #358.
 *
 * Both settings are lexical: Clang marks each instruction with the setting of the code
 * it was written in, so `Kokkos::complex`'s operators, written in Kokkos' headers, stay
 * contractable even when inlined into a protected block. A protected block therefore
 * spells its complex arithmetic with the strict helpers below (`strict_mul` and the
 * rest), each itself a protected function, and never with `Kokkos::complex`'s
 * operators. Library functions (`exp`, `hypot`) keep their own error bounds.
 *
 * tests/native/test_fp.cpp checks per build that a protected block, and a strict helper
 * inlined into one, rounds a product and a sum separately, and that an ordinary one
 * contracts where the build optimises for a target with fused multiply-add.
 */
#if defined(__CUDACC__) || defined(__HIPCC__)
#define SOLPROP_NATIVE_STRICT_FUNCTION
#define SOLPROP_NATIVE_STRICT_BLOCK
#elif defined(__clang__)
#define SOLPROP_NATIVE_STRICT_FUNCTION
#define SOLPROP_NATIVE_STRICT_BLOCK _Pragma("clang fp contract(off)")
#elif defined(__GNUC__)
#define SOLPROP_NATIVE_STRICT_FUNCTION __attribute__((optimize("fp-contract=off")))
#define SOLPROP_NATIVE_STRICT_BLOCK
#else
#error "solprop-native: no strict floating-point block for this compiler"
#endif

namespace solprop_native {

/** `a + b`, each part rounded once. */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION Kokkos::complex<double>
strict_add(const Kokkos::complex<double> a, const Kokkos::complex<double> b) {
  SOLPROP_NATIVE_STRICT_BLOCK
  return {a.real() + b.real(), a.imag() + b.imag()};
}

/** `a + b` for a real `a`, as `Kokkos::complex`'s operator forms it. */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION Kokkos::complex<double>
strict_add(const double a, const Kokkos::complex<double> b) {
  SOLPROP_NATIVE_STRICT_BLOCK
  return {a + b.real(), b.imag()};
}

/** `a - b`, each part rounded once. */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION Kokkos::complex<double>
strict_sub(const Kokkos::complex<double> a, const Kokkos::complex<double> b) {
  SOLPROP_NATIVE_STRICT_BLOCK
  return {a.real() - b.real(), a.imag() - b.imag()};
}

/** `w a` for a real `w`: two rounded products. */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION Kokkos::complex<double>
strict_scale(const double w, const Kokkos::complex<double> a) {
  SOLPROP_NATIVE_STRICT_BLOCK
  return {w * a.real(), w * a.imag()};
}

/** `a b`, `Kokkos::complex`'s formula with every product and sum rounded. */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION Kokkos::complex<double>
strict_mul(const Kokkos::complex<double> a, const Kokkos::complex<double> b) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const double rr = a.real() * b.real();
  const double ii = a.imag() * b.imag();
  const double ri = a.real() * b.imag();
  const double ir = a.imag() * b.real();
  return {rr - ii, ri + ir};
}

/** `|a|^2`, each square rounded before the sum: a magnitude test without `hypot`. */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION double strict_norm(const Kokkos::complex<double> a) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const double rr = a.real() * a.real();
  const double ii = a.imag() * a.imag();
  return rr + ii;
}

/** `sum + w a` for a real `w`, the product rounded before the sum. */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION Kokkos::complex<double>
strict_accumulate(const Kokkos::complex<double> sum, const double w,
                  const Kokkos::complex<double> a) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const double real = w * a.real();
  const double imag = w * a.imag();
  return {sum.real() + real, sum.imag() + imag};
}

/** `a / w` for a real `w`. */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION Kokkos::complex<double>
strict_divide(const Kokkos::complex<double> a, const double w) {
  SOLPROP_NATIVE_STRICT_BLOCK
  return {a.real() / w, a.imag() / w};
}

/** `a / b` by `Kokkos::complex`'s scaled algorithm, every operation rounded. */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION Kokkos::complex<double>
strict_divide(const Kokkos::complex<double> a, const Kokkos::complex<double> b) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const double s = Kokkos::fabs(b.real()) + Kokkos::fabs(b.imag());
  if (s == 0.0)
    return {a.real() / s, a.imag() / s};
  const Kokkos::complex<double> numerator(a.real() / s, a.imag() / s);
  const Kokkos::complex<double> conjugate(b.real() / s, -b.imag() / s);
  const double rr = conjugate.real() * conjugate.real();
  const double ii = conjugate.imag() * conjugate.imag();
  return strict_divide(strict_mul(numerator, conjugate), rr + ii);
}

} // namespace solprop_native
