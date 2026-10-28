// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cmath>
#include <cstddef>

#if defined(SOLPROP_NATIVE_VECTOR_MATH_SLEEF)
#include <cstring>

#include <sleef.h>
#endif

/*
 * Host vector maths for the transcendental passes of the CPU kernels (solprop
 * ADR-059; cmake/VectorMath.cmake chooses the route).
 *
 * Batch functions over contiguous arrays: `exp`, `log`, `sincos` and the complex
 * exponential `e^a (cos b, sin b)`. A kernel forms its arguments for a block of
 * elements into a buffer, calls one of these over the buffer, and finishes each
 * element from the results, so that the transcendental work is one pass the library
 * vectorises rather than a call inside a branchy per-element body. They are host
 * functions: device kernels keep their own `Kokkos::` functions, and a device build
 * takes the scalar route.
 *
 * - SLEEF (`SOLPROP_NATIVE_VECTOR_MATH_SLEEF`): SLEEF's u10 functions, each within
 *   1.0 ulp of the correctly rounded value over its whole domain (sincos included:
 *   SLEEF reduces large arguments exactly), with C99's results at zeros, infinities
 *   and NaNs. The variant is the code's target: AVX2 with FMA (four lanes) wherever
 *   the target has them, the distribution's x86-64-v3 included, else SSE2 (two);
 *   AdvSIMD (two) on aarch64. AVX-512F (eight) only on request
 *   (`SOLPROP_NATIVE_VECTOR_MATH_AVX512`, for a target with it): faster than AVX2 on
 *   an AMD EPYC 9B45 (Zen 5) and slower on an Intel Xeon of the Ice Lake generation,
 *   whose cores lower their clock under 512-bit work. A block's last partial vector is
 * padded with a harmless argument and goes through the same function, so an element's
 * result does not depend on its place in the block.
 * - Scalar: the system libm's `std::exp`, `std::log`, `std::cos` and `std::sin`, one
 *   call an element, as the kernels made them before.
 *
 * Library functions keep their own error bounds (solprop SEP 25): no contraction or
 * reassociation applies inside them. The complex exponential's two products are
 * ordinary products, each rounded once, as `exponential` forms them. An output may
 * be its own input array (element by element), never an overlapping part of another.
 */
namespace solprop_native::vmath {

// A variant's lanes: SLEEF's vector type, which is the compilers' vector type, so that
// `memcpy` loads and stores it unaligned and `*` forms one rounded product a lane, with
// no intrinsic (written out in each: GCC drops the type's attributes from a template
// argument).
#if defined(SOLPROP_NATIVE_VECTOR_MATH_SLEEF) &&                                       \
    defined(SOLPROP_NATIVE_VECTOR_MATH_AVX512) && defined(__AVX512F__)
struct Lanes {
  using Vector = __m512d;
  static constexpr std::size_t width = 8;
  static Vector load(const double* p) {
    Vector v;
    std::memcpy(&v, p, sizeof(Vector));
    return v;
  }
  static void store(double* p, const Vector v) { std::memcpy(p, &v, sizeof(Vector)); }
  static Vector mul(const Vector a, const Vector b) { return a * b; }
  static constexpr const char* name = "AVX-512F";
  static Vector exp(const Vector v) { return Sleef_expd8_u10avx512f(v); }
  static Vector log(const Vector v) { return Sleef_logd8_u10avx512f(v); }
  static void sincos(const Vector v, Vector& sine, Vector& cosine) {
    const Sleef___m512d_2 pair = Sleef_sincosd8_u10avx512f(v);
    sine = pair.x;
    cosine = pair.y;
  }
};
#elif defined(SOLPROP_NATIVE_VECTOR_MATH_SLEEF) && defined(__AVX2__) && defined(__FMA__)
struct Lanes {
  using Vector = __m256d;
  static constexpr std::size_t width = 4;
  static Vector load(const double* p) {
    Vector v;
    std::memcpy(&v, p, sizeof(Vector));
    return v;
  }
  static void store(double* p, const Vector v) { std::memcpy(p, &v, sizeof(Vector)); }
  static Vector mul(const Vector a, const Vector b) { return a * b; }
  static constexpr const char* name = "AVX2";
  static Vector exp(const Vector v) { return Sleef_expd4_u10avx2(v); }
  static Vector log(const Vector v) { return Sleef_logd4_u10avx2(v); }
  static void sincos(const Vector v, Vector& sine, Vector& cosine) {
    const Sleef___m256d_2 pair = Sleef_sincosd4_u10avx2(v);
    sine = pair.x;
    cosine = pair.y;
  }
};
#elif defined(SOLPROP_NATIVE_VECTOR_MATH_SLEEF) &&                                     \
    (defined(__x86_64__) || defined(_M_X64))
struct Lanes {
  using Vector = __m128d;
  static constexpr std::size_t width = 2;
  static Vector load(const double* p) {
    Vector v;
    std::memcpy(&v, p, sizeof(Vector));
    return v;
  }
  static void store(double* p, const Vector v) { std::memcpy(p, &v, sizeof(Vector)); }
  static Vector mul(const Vector a, const Vector b) { return a * b; }
  static constexpr const char* name = "SSE2";
  static Vector exp(const Vector v) { return Sleef_expd2_u10sse2(v); }
  static Vector log(const Vector v) { return Sleef_logd2_u10sse2(v); }
  static void sincos(const Vector v, Vector& sine, Vector& cosine) {
    const Sleef___m128d_2 pair = Sleef_sincosd2_u10sse2(v);
    sine = pair.x;
    cosine = pair.y;
  }
};
#elif defined(SOLPROP_NATIVE_VECTOR_MATH_SLEEF) &&                                     \
    (defined(__aarch64__) || defined(_M_ARM64))
struct Lanes {
  using Vector = float64x2_t;
  static constexpr std::size_t width = 2;
  static Vector load(const double* p) {
    Vector v;
    std::memcpy(&v, p, sizeof(Vector));
    return v;
  }
  static void store(double* p, const Vector v) { std::memcpy(p, &v, sizeof(Vector)); }
  static Vector mul(const Vector a, const Vector b) { return a * b; }
  static constexpr const char* name = "AdvSIMD";
  static Vector exp(const Vector v) { return Sleef_expd2_u10advsimd(v); }
  static Vector log(const Vector v) { return Sleef_logd2_u10advsimd(v); }
  static void sincos(const Vector v, Vector& sine, Vector& cosine) {
    const Sleef_float64x2_t_2 pair = Sleef_sincosd2_u10advsimd(v);
    sine = pair.x;
    cosine = pair.y;
  }
};
#elif defined(SOLPROP_NATIVE_VECTOR_MATH_SLEEF)
#error "solprop-native: SLEEF vector maths needs an x86-64 or aarch64 target"
#endif

#if defined(SOLPROP_NATIVE_VECTOR_MATH_SLEEF)
inline constexpr bool vectorised = true;
inline constexpr std::size_t width = Lanes::width;
// The variant, for the build record.
inline constexpr const char* route = Lanes::name;

namespace detail {

// A block's last `count < width` elements, through one padded vector. `pad` is an
// argument at which the function is finite (its result is discarded).
template <std::size_t Inputs, std::size_t Outputs, class Apply>
void tail(const double* const (&inputs)[Inputs], double* const (&outputs)[Outputs],
          std::size_t count, const double pad, Apply apply) {
  count = count < Lanes::width ? count : Lanes::width;
  double in[Inputs][Lanes::width];
  double out[Outputs][Lanes::width];
  for (std::size_t input = 0; input < Inputs; ++input)
    for (std::size_t lane = 0; lane < Lanes::width; ++lane)
      in[input][lane] = lane < count ? inputs[input][lane] : pad;
  apply(in, out);
  for (std::size_t output = 0; output < Outputs; ++output)
    for (std::size_t lane = 0; lane < count; ++lane)
      outputs[output][lane] = out[output][lane];
}

} // namespace detail

/** `y_i = exp(x_i)` for `i < n`. */
inline void exp(const double* x, double* y, const std::size_t n) {
  const std::size_t full = n - n % Lanes::width;
  for (std::size_t k = 0; k < full; k += Lanes::width)
    Lanes::store(y + k, Lanes::exp(Lanes::load(x + k)));
  if (full < n)
    detail::tail<1, 1>({x + full}, {y + full}, n - full, 0.0, [](auto& in, auto& out) {
      Lanes::store(out[0], Lanes::exp(Lanes::load(in[0])));
    });
}

/** `y_i = log(x_i)` for `i < n`. */
inline void log(const double* x, double* y, const std::size_t n) {
  const std::size_t full = n - n % Lanes::width;
  for (std::size_t k = 0; k < full; k += Lanes::width)
    Lanes::store(y + k, Lanes::log(Lanes::load(x + k)));
  if (full < n)
    detail::tail<1, 1>({x + full}, {y + full}, n - full, 1.0, [](auto& in, auto& out) {
      Lanes::store(out[0], Lanes::log(Lanes::load(in[0])));
    });
}

/** `sine_i = sin(x_i)` and `cosine_i = cos(x_i)` for `i < n`. */
inline void sincos(const double* x, double* sine, double* cosine, const std::size_t n) {
  const std::size_t full = n - n % Lanes::width;
  for (std::size_t k = 0; k < full; k += Lanes::width) {
    Lanes::Vector s;
    Lanes::Vector c;
    Lanes::sincos(Lanes::load(x + k), s, c);
    Lanes::store(sine + k, s);
    Lanes::store(cosine + k, c);
  }
  if (full < n)
    detail::tail<1, 2>({x + full}, {sine + full, cosine + full}, n - full, 0.0,
                       [](auto& in, auto& out) {
                         Lanes::Vector s;
                         Lanes::Vector c;
                         Lanes::sincos(Lanes::load(in[0]), s, c);
                         Lanes::store(out[0], s);
                         Lanes::store(out[1], c);
                       });
}

/** `(real_i, imag_i) = e^{a_i} (cos b_i, sin b_i)` for `i < n`. */
inline void exp_complex(const double* a, const double* b, double* real, double* imag,
                        const std::size_t n) {
  const auto apply = [](const Lanes::Vector va, const Lanes::Vector vb,
                        Lanes::Vector& vr, Lanes::Vector& vi) {
    const Lanes::Vector magnitude = Lanes::exp(va);
    Lanes::Vector s;
    Lanes::Vector c;
    Lanes::sincos(vb, s, c);
    vr = Lanes::mul(magnitude, c);
    vi = Lanes::mul(magnitude, s);
  };
  const std::size_t full = n - n % Lanes::width;
  for (std::size_t k = 0; k < full; k += Lanes::width) {
    Lanes::Vector vr;
    Lanes::Vector vi;
    apply(Lanes::load(a + k), Lanes::load(b + k), vr, vi);
    Lanes::store(real + k, vr);
    Lanes::store(imag + k, vi);
  }
  if (full < n)
    detail::tail<2, 2>({a + full, b + full}, {real + full, imag + full}, n - full, 0.0,
                       [&apply](auto& in, auto& out) {
                         Lanes::Vector vr;
                         Lanes::Vector vi;
                         apply(Lanes::load(in[0]), Lanes::load(in[1]), vr, vi);
                         Lanes::store(out[0], vr);
                         Lanes::store(out[1], vi);
                       });
}

#else
inline constexpr bool vectorised = false;
inline constexpr std::size_t width = 1;
inline constexpr const char* route = "scalar libm";

/** `y_i = exp(x_i)` for `i < n`. */
inline void exp(const double* x, double* y, const std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    y[i] = std::exp(x[i]);
}

/** `y_i = log(x_i)` for `i < n`. */
inline void log(const double* x, double* y, const std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    y[i] = std::log(x[i]);
}

/** `sine_i = sin(x_i)` and `cosine_i = cos(x_i)` for `i < n`: two library calls. */
inline void sincos(const double* x, double* sine, double* cosine, const std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const double value = x[i];
    cosine[i] = std::cos(value);
    sine[i] = std::sin(value);
  }
}

/** `(real_i, imag_i) = e^{a_i} (cos b_i, sin b_i)` for `i < n`. */
inline void exp_complex(const double* a, const double* b, double* real, double* imag,
                        const std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const double magnitude = std::exp(a[i]);
    const double angle = b[i];
    real[i] = magnitude * std::cos(angle);
    imag[i] = magnitude * std::sin(angle);
  }
}
#endif

} // namespace solprop_native::vmath
