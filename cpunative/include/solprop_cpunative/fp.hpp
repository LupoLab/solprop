// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cmath>
#include <complex>

/*
 * Floating-point contraction policy (solprop SEP 25, sections 4.2 and 5).
 *
 * CPU builds compile with contraction on across statements
 * (`-ffp-contract=fast` for GCC, `-ffp-contract=fast-honor-pragmas` for Clang
 * and icpx; CMakeLists.txt), so the compiler may fuse any multiply and add into
 * one rounding. A protected block keeps the arithmetic as written, each product
 * and sum rounded on its own: the stable phase increments, the small-argument
 * phi-functions, the ionised fraction (every column's `1 - (1 - p) e^{-X}` and
 * the envelope's `p + (1 - p) (-expm1(-X))`, in the kernels and in the
 * statistics), and the multistep defects and their compensation (the residual
 * terms `(D - x / dz) u` and a coupled flow's `R_s u`, the compensation sums,
 * their addition to the predicted, corrected and observed solutions, and the
 * endpoint rate). Contraction there is not inaccurate as such; these are the
 * formulas whose small results the policy keeps under strict arithmetic rather
 * than requalifying. The finite-value guards are comparisons, which contraction
 * cannot change; what keeps them meaningful is the absence of `nnan`/`ninf` and
 * fast-math from every build, not a protected block.
 *
 * A protected function or lambda carries `SOLPROP_CPUNATIVE_STRICT_FUNCTION`
 * (before a function's declaration, after a lambda's parameter list) and starts
 * its body with `SOLPROP_CPUNATIVE_STRICT_BLOCK`:
 *
 * - Clang and icpx: `#pragma clang fp contract(off)` at the start of the body.
 * The setting is carried by each instruction, so it survives inlining.
 * - GCC ignores `#pragma STDC FP_CONTRACT` and Clang's pragma, so the function
 * is compiled with `optimize("fp-contract=off")`. GCC then does not inline it
 * into a caller compiled with other options: a protected helper is a call
 * there.
 *
 * Both settings are lexical: Clang marks each instruction with the setting of
 * the code it was written in, so `std::complex`'s operators, written in
 * standard-library headers, stay contractable even when inlined into a
 * protected block. A protected block therefore spells its complex arithmetic
 * with the strict helpers below (`strict_mul` and the rest), each itself a
 * protected function, and never with `std::complex`'s operators. Library
 * functions (`exp`, `hypot`) keep their own error bounds.
 *
 */
#if defined(__clang__)
#define SOLPROP_CPUNATIVE_STRICT_FUNCTION
#define SOLPROP_CPUNATIVE_STRICT_BLOCK _Pragma("clang fp contract(off)")
#elif defined(__GNUC__)
#define SOLPROP_CPUNATIVE_STRICT_FUNCTION __attribute__((optimize("fp-contract=off")))
#define SOLPROP_CPUNATIVE_STRICT_BLOCK
#else
#error "solprop-cpunative: no strict floating-point block for this compiler"
#endif

namespace solprop_cpunative {

/** `a + b`, each part rounded once. */
SOLPROP_CPUNATIVE_STRICT_FUNCTION
inline std::complex<double> strict_add(const std::complex<double> a,
                                       const std::complex<double> b) {
  SOLPROP_CPUNATIVE_STRICT_BLOCK
  return {a.real() + b.real(), a.imag() + b.imag()};
}

/** `a + b` for a real `a`, as `std::complex`'s operator forms it. */
SOLPROP_CPUNATIVE_STRICT_FUNCTION
inline std::complex<double> strict_add(const double a, const std::complex<double> b) {
  SOLPROP_CPUNATIVE_STRICT_BLOCK
  return {a + b.real(), b.imag()};
}

/** `a - b`, each part rounded once. */
SOLPROP_CPUNATIVE_STRICT_FUNCTION
inline std::complex<double> strict_sub(const std::complex<double> a,
                                       const std::complex<double> b) {
  SOLPROP_CPUNATIVE_STRICT_BLOCK
  return {a.real() - b.real(), a.imag() - b.imag()};
}

/** `w a` for a real `w`: two rounded products. */
SOLPROP_CPUNATIVE_STRICT_FUNCTION
inline std::complex<double> strict_scale(const double w, const std::complex<double> a) {
  SOLPROP_CPUNATIVE_STRICT_BLOCK
  return {w * a.real(), w * a.imag()};
}

/** `a b`, `std::complex`'s formula with every product and sum rounded. */
SOLPROP_CPUNATIVE_STRICT_FUNCTION
inline std::complex<double> strict_mul(const std::complex<double> a,
                                       const std::complex<double> b) {
  SOLPROP_CPUNATIVE_STRICT_BLOCK
  const double rr = a.real() * b.real();
  const double ii = a.imag() * b.imag();
  const double ri = a.real() * b.imag();
  const double ir = a.imag() * b.real();
  return {rr - ii, ri + ir};
}

/** `sum + w a` for a real `w`, the product rounded before the sum. */
SOLPROP_CPUNATIVE_STRICT_FUNCTION
inline std::complex<double> strict_accumulate(const std::complex<double> sum,
                                              const double w,
                                              const std::complex<double> a) {
  SOLPROP_CPUNATIVE_STRICT_BLOCK
  const double real = w * a.real();
  const double imag = w * a.imag();
  return {sum.real() + real, sum.imag() + imag};
}

/** `a / w` for a real `w`. */
SOLPROP_CPUNATIVE_STRICT_FUNCTION
inline std::complex<double> strict_divide(const std::complex<double> a,
                                          const double w) {
  SOLPROP_CPUNATIVE_STRICT_BLOCK
  return {a.real() / w, a.imag() / w};
}

/** `a / b` by `std::complex`'s scaled algorithm, every operation rounded. */
SOLPROP_CPUNATIVE_STRICT_FUNCTION
inline std::complex<double> strict_divide(const std::complex<double> a,
                                          const std::complex<double> b) {
  SOLPROP_CPUNATIVE_STRICT_BLOCK
  const double s = std::fabs(b.real()) + std::fabs(b.imag());
  if (s == 0.0)
    return {a.real() / s, a.imag() / s};
  const std::complex<double> numerator(a.real() / s, a.imag() / s);
  const std::complex<double> conjugate(b.real() / s, -b.imag() / s);
  const double rr = conjugate.real() * conjugate.real();
  const double ii = conjugate.imag() * conjugate.imag();
  return strict_divide(strict_mul(numerator, conjugate), rr + ii);
}

} // namespace solprop_cpunative
