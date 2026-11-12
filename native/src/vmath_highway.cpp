// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The Highway route of the host vector maths (include/solprop_native/vmath.hpp):
// Highway's contrib maths, compiled once for each target this build allows and chosen
// at run time by the CPU (HWY_DYNAMIC_DISPATCH), behind the layer's batch functions.
//
// Highway's functions are not SLEEF's u10 ones, and the wrappers below give them the
// layer's contract where they fall short of it:
// - `Exp` (1 ulp) returns 0 at a NaN and has no overflow to infinity: a NaN keeps its
//   value and an argument above log(DBL_MAX) gives +inf, as C99 says.
// - `Log` (Highway documents 4 ulp): zero gives -inf, a negative or NaN argument NaN,
//   +inf itself, as C99 says.
// - `SinCos` (1 ulp) is valid for |x| <= 39000 (a three-part Cody-Waite reduction): a
//   vector with an argument outside that range, or a NaN or infinity, takes the system
//   libm's `sin` and `cos` for those lanes.
// This file is compiled with -ffp-contract=off, so that the products and sums Highway
// writes out are rounded as it wrote them.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "vmath_highway.cpp"
#include <hwy/contrib/math/math-inl.h>
#include <hwy/foreach_target.h> // IWYU pragma: keep
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace solprop_native::vmath::HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

namespace {

using D = hn::ScalableTag<double>;
using V = hn::Vec<D>;

// Above this, exp overflows: the largest x with exp(x) finite is about 709.7827.
constexpr double exp_overflow = 709.782712893384;
// Highway's documented range for SinCos.
constexpr double sincos_limit = 39000.0;

HWY_INLINE V exp_vector(const D d, const V x) {
  const V y = hn::Exp(d, x);
  const V overflowed =
      hn::IfThenElse(hn::Gt(x, hn::Set(d, exp_overflow)),
                     hn::Set(d, std::numeric_limits<double>::infinity()), y);
  return hn::IfThenElse(hn::IsNaN(x), x, overflowed);
}

HWY_INLINE V log_vector(const D d, const V x) {
  const V y = hn::Log(d, x);
  const V zero = hn::Zero(d);
  const V at_zero = hn::IfThenElse(
      hn::Eq(x, zero), hn::Set(d, -std::numeric_limits<double>::infinity()), y);
  const V at_infinity = hn::IfThenElse(hn::IsInf(x), x, at_zero);
  // A negative argument or a NaN: NaN (the comparison is false at a NaN).
  return hn::IfThenElse(hn::Ge(x, zero), at_infinity,
                        hn::Set(d, std::numeric_limits<double>::quiet_NaN()));
}

// The lanes of `x` that SinCos does not cover: |x| above its range, NaN or infinite.
HWY_INLINE auto sincos_outside(const D d, const V x) {
  return hn::Not(hn::Le(hn::Abs(x), hn::Set(d, sincos_limit)));
}

// `sine` and `cosine` of the `count <= Lanes(d)` lanes of `x`, stored at `s` and `c`
// (either may be where `x` was loaded from); lanes outside SinCos's range take the
// system libm.
HWY_INLINE void sincos_store(const D d, const V x, double* s, double* c,
                             const std::size_t count) {
  V sine;
  V cosine;
  hn::SinCos(d, x, sine, cosine);
  const auto outside = sincos_outside(d, x);
  if (HWY_UNLIKELY(!hn::AllFalse(d, outside))) {
    HWY_ALIGN double in[hn::MaxLanes(d)];
    HWY_ALIGN double out_s[hn::MaxLanes(d)];
    HWY_ALIGN double out_c[hn::MaxLanes(d)];
    hn::Store(x, d, in);
    hn::Store(sine, d, out_s);
    hn::Store(cosine, d, out_c);
    for (std::size_t lane = 0; lane < count; ++lane)
      if (!(std::fabs(in[lane]) <= sincos_limit)) {
        out_s[lane] = std::sin(in[lane]);
        out_c[lane] = std::cos(in[lane]);
      }
    sine = hn::Load(d, out_s);
    cosine = hn::Load(d, out_c);
  }
  hn::StoreN(sine, d, s, count);
  hn::StoreN(cosine, d, c, count);
}

} // namespace

void exp_batch(const double* x, double* y, const std::size_t n) {
  const D d;
  const std::size_t lanes = hn::Lanes(d);
  std::size_t k = 0;
  for (; k + lanes <= n; k += lanes)
    hn::StoreU(exp_vector(d, hn::LoadU(d, x + k)), d, y + k);
  if (k < n) {
    const std::size_t rest = n - k;
    hn::StoreN(exp_vector(d, hn::LoadNOr(hn::Zero(d), d, x + k, rest)), d, y + k, rest);
  }
}

void log_batch(const double* x, double* y, const std::size_t n) {
  const D d;
  const std::size_t lanes = hn::Lanes(d);
  std::size_t k = 0;
  for (; k + lanes <= n; k += lanes)
    hn::StoreU(log_vector(d, hn::LoadU(d, x + k)), d, y + k);
  if (k < n) {
    const std::size_t rest = n - k;
    hn::StoreN(log_vector(d, hn::LoadNOr(hn::Set(d, 1.0), d, x + k, rest)), d, y + k,
               rest);
  }
}

void sincos_batch(const double* x, double* sine, double* cosine, const std::size_t n) {
  const D d;
  const std::size_t lanes = hn::Lanes(d);
  // An output may be its input: each vector is loaded before its results are stored.
  for (std::size_t k = 0; k < n; k += lanes) {
    const std::size_t count = n - k < lanes ? n - k : lanes;
    sincos_store(d, hn::LoadNOr(hn::Zero(d), d, x + k, count), sine + k, cosine + k,
                 count);
  }
}

void exp_complex_batch(const double* a, const double* b, double* real, double* imag,
                       const std::size_t n) {
  const D d;
  const std::size_t lanes = hn::Lanes(d);
  for (std::size_t k = 0; k < n; k += lanes) {
    const std::size_t count = n - k < lanes ? n - k : lanes;
    // Both arguments are loaded before any output is stored (an output may be an
    // input).
    const V magnitude = exp_vector(d, hn::LoadNOr(hn::Zero(d), d, a + k, count));
    const V angle = hn::LoadNOr(hn::Zero(d), d, b + k, count);
    V sine;
    V cosine;
    hn::SinCos(d, angle, sine, cosine);
    if (HWY_UNLIKELY(!hn::AllFalse(d, sincos_outside(d, angle)))) {
      HWY_ALIGN double in[hn::MaxLanes(d)];
      HWY_ALIGN double out_s[hn::MaxLanes(d)];
      HWY_ALIGN double out_c[hn::MaxLanes(d)];
      hn::Store(angle, d, in);
      hn::Store(sine, d, out_s);
      hn::Store(cosine, d, out_c);
      for (std::size_t lane = 0; lane < count; ++lane)
        if (!(std::fabs(in[lane]) <= sincos_limit)) {
          out_s[lane] = std::sin(in[lane]);
          out_c[lane] = std::cos(in[lane]);
        }
      sine = hn::Load(d, out_s);
      cosine = hn::Load(d, out_c);
    }
    // One rounded product a component, as `exponential` forms it.
    hn::StoreN(hn::Mul(magnitude, cosine), d, real + k, count);
    hn::StoreN(hn::Mul(magnitude, sine), d, imag + k, count);
  }
}

const char* target_name() { return hwy::TargetName(HWY_TARGET); }

} // namespace solprop_native::vmath::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
#include <cstdlib>
#include <cstring>

namespace solprop_native::vmath {

HWY_EXPORT(exp_batch);
HWY_EXPORT(log_batch);
HWY_EXPORT(sincos_batch);
HWY_EXPORT(exp_complex_batch);
HWY_EXPORT(target_name);

namespace {

// The targets the dispatch may choose, set once before the first call. By default the
// AVX-512 targets of Intel's Skylake-SP to Ice Lake cores (AVX3, AVX3_DL), which lower
// their clock under 512-bit work (ADR-059 measured SLEEF's AVX-512 functions slower on
// one), are left out, and Zen 4 and later, Sapphire Rapids and AVX10.2 kept. The
// prototype's SOLPROP_NATIVE_VMATH_TARGETS overrides it: `avx2` leaves out every
// AVX-512 target, `all` keeps them all.
bool configure_targets() {
  constexpr std::int64_t all_avx512 =
      HWY_AVX3 | HWY_AVX3_DL | HWY_AVX3_ZEN4 | HWY_AVX3_SPR | HWY_AVX10_2;
  std::int64_t disabled = HWY_AVX3 | HWY_AVX3_DL;
  if (const char* request = std::getenv("SOLPROP_NATIVE_VMATH_TARGETS")) {
    if (std::strcmp(request, "avx2") == 0)
      disabled = all_avx512;
    else if (std::strcmp(request, "all") == 0)
      disabled = 0;
  }
  if (disabled != 0)
    hwy::DisableTargets(disabled);
  return true;
}

void ensure_targets() {
  static const bool configured = configure_targets();
  (void)configured;
}

} // namespace

void exp(const double* x, double* y, const std::size_t n) {
  ensure_targets();
  HWY_DYNAMIC_DISPATCH(exp_batch)(x, y, n);
}

void log(const double* x, double* y, const std::size_t n) {
  ensure_targets();
  HWY_DYNAMIC_DISPATCH(log_batch)(x, y, n);
}

void sincos(const double* x, double* sine, double* cosine, const std::size_t n) {
  ensure_targets();
  HWY_DYNAMIC_DISPATCH(sincos_batch)(x, sine, cosine, n);
}

void exp_complex(const double* a, const double* b, double* real, double* imag,
                 const std::size_t n) {
  ensure_targets();
  HWY_DYNAMIC_DISPATCH(exp_complex_batch)(a, b, real, imag, n);
}

const char* target() {
  ensure_targets();
  return HWY_DYNAMIC_DISPATCH(target_name)();
}

} // namespace solprop_native::vmath
#endif
