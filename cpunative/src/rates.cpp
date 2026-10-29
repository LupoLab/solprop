// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "solprop_cpunative/core.hpp"
#include <cmath>
#ifdef CPUNATIVE_RATE_SLEEF
#include <sleef.h>
#elif defined(CPUNATIVE_RATE_MKL)
#include <mkl_vml.h>
#endif

namespace solprop_cpunative {
namespace {
#if defined(CPUNATIVE_RATE_SLEEF) || defined(CPUNATIVE_RATE_MKL)
void logarithms(double* values, int count) {
#ifdef CPUNATIVE_RATE_MKL
  vmdLn(count, values, values, VML_HA | VML_FTZDAZ_OFF);
#else
  int t = 0;
#if defined(__aarch64__)
  for (; t + 2 <= count; t += 2)
    vst1q_f64(values + t, Sleef_logd2_u10advsimd(vld1q_f64(values + t)));
#else
  for (; t + 4 <= count; t += 4)
    _mm256_storeu_pd(values + t, Sleef_logd4_u10avx2(_mm256_loadu_pd(values + t)));
#endif
  for (; t < count; ++t)
    values[t] = std::log(values[t]);
#endif
}
void exponentials(double* values, int count) {
#ifdef CPUNATIVE_RATE_MKL
  vmdExp(count, values, values, VML_HA | VML_FTZDAZ_OFF);
#else
  int t = 0;
#if defined(__aarch64__)
  for (; t + 2 <= count; t += 2)
    vst1q_f64(values + t, Sleef_expd2_u10advsimd(vld1q_f64(values + t)));
#else
  for (; t + 4 <= count; t += 4)
    _mm256_storeu_pd(values + t, Sleef_expd4_u10avx2(_mm256_loadu_pd(values + t)));
#endif
  for (; t < count; ++t)
    values[t] = std::exp(values[t]);
#endif
}
#endif
} // namespace
double State::ionisation_rate(double field) const {
  const double magnitude = std::abs(field);
  if (magnitude < c_.field_min)
    return 0;
  if (magnitude >= c_.field_max)
    return c_.rate_max;
  const auto columns = c_.rate.size() / 4;
  const double log = std::log(magnitude);
  const auto index = std::min(
      columns - 1,
      std::size_t(std::max(0.0, std::floor((log - c_.log_min) / c_.log_step))));
  const double offset = log - (c_.log_min + double(index) * c_.log_step);
  return std::exp(((c_.rate[index] * offset + c_.rate[columns + index]) * offset +
                   c_.rate[2 * columns + index]) *
                      offset +
                  c_.rate[3 * columns + index]);
}

void State::ionisation_rates(const double* __restrict field,
                             double* __restrict rate) const {
  // Masked vector transcendental functions still compute inactive lanes. The
  // sparse tails need the scalar early exit; dense blocks can amortise SIMD.
  constexpr int block = 256;
  for (int first = 0; first < c_.nf; first += block) {
    const int count = std::min(block, c_.nf - first);
    int live = 0;
#pragma omp simd reduction(+ : live)
    for (int t = 0; t < count; ++t)
      live += std::abs(field[first + t]) >= c_.field_min;
    if (!live) {
      std::fill_n(rate + first, count, 0.);
    } else if (live < count / 2) {
      for (int t = 0; t < count; ++t)
        rate[first + t] = ionisation_rate(field[first + t]);
    } else {
#if defined(CPUNATIVE_RATE_SLEEF) || defined(CPUNATIVE_RATE_MKL)
      // Only the finite positive interpolation interval reaches vector math.
      // Masked tails are restored after the call; no global math mode changes.
      alignas(64) double work[block];
#pragma omp simd
      for (int t = 0; t < count; ++t)
        work[t] = std::clamp(std::abs(field[first + t]), c_.field_min, c_.field_max);
      logarithms(work, count);
      const auto columns = c_.rate.size() / 4;
#pragma omp simd
      for (int t = 0; t < count; ++t) {
        const double log = work[t];
        const auto index = std::min(
            columns - 1,
            std::size_t(std::max(0.0, std::floor((log - c_.log_min) / c_.log_step))));
        const double offset = log - (c_.log_min + double(index) * c_.log_step);
        work[t] = ((c_.rate[index] * offset + c_.rate[columns + index]) * offset +
                   c_.rate[2 * columns + index]) *
                      offset +
                  c_.rate[3 * columns + index];
      }
      exponentials(work, count);
#pragma omp simd
      for (int t = 0; t < count; ++t) {
        const double magnitude = std::abs(field[first + t]);
        rate[first + t] = magnitude < c_.field_min    ? 0.
                          : magnitude >= c_.field_max ? c_.rate_max
                                                      : work[t];
      }
#else
#pragma omp simd
      for (int t = 0; t < count; ++t)
        rate[first + t] = ionisation_rate(field[first + t]);
#endif
    }
  }
}

} // namespace solprop_cpunative
