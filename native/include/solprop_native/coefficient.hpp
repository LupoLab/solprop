// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace solprop_native {

/**
 * A combined coefficient outside binary32's normal range (solprop SEP 33 section 4.3).
 *
 * The bindings raise it as solprop's `TranslationError`, with the single-precision
 * oracle's message, so that native and the oracle refuse a problem alike.
 */
class CoefficientRangeError : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};

/**
 * A scalar coefficient of a binary32 pass, combined in binary64 and rounded once
 * (solprop SEP 33 section 4.3, rule 4).
 *
 * `solprop.translate.precision.combine` and `Coefficient.binary32`, operation for
 * operation, so that native and the oracle round the same value: each factor is split
 * into its mantissa and exponent, the mantissas multiplied and renormalised and the
 * exponents added as integers; the value is formed once and rounded once. Its
 * refusals carry the oracle's messages.
 *
 * @param first The first of the finite binary64 factors.
 * @param last One past the last factor.
 * @param exponent A further power of two.
 * @return The binary32 coefficient.
 * @throws CoefficientRangeError If the value overflows binary64, or its binary32
 * rounding is not finite or is subnormal (and the value is not zero).
 */
inline float combined_binary32(const double* first, const double* last,
                               const int exponent) {
  double mantissa = 1.0;
  long long total = exponent;
  for (; first != last; ++first) {
    const double factor = *first;
    int power = 0;
    const double part = std::frexp(factor, &power);
    mantissa *= part;
    total += power;
    if (mantissa == 0.0)
      return 0.0F;
    mantissa = std::frexp(mantissa, &power);
    total += power;
  }
  const auto clamped = static_cast<int>(
      total > std::numeric_limits<int>::max()   ? std::numeric_limits<int>::max()
      : total < std::numeric_limits<int>::min() ? std::numeric_limits<int>::min()
                                                : total);
  const double value = std::ldexp(mantissa, clamped);
  if (std::isinf(value))
    throw CoefficientRangeError("a combined coefficient 2**" + std::to_string(total) +
                                " overflows binary64");
  const auto narrowed = static_cast<float>(value);
  if (!std::isfinite(narrowed) ||
      (value != 0.0 && std::abs(narrowed) < std::numeric_limits<float>::min())) {
    char text[32];
    std::snprintf(text, sizeof text, "%.3e", value);
    throw CoefficientRangeError(
        std::string("a combined coefficient ") + text +
        " leaves binary32's normal range; the pass applying it must apply it where its "
        "intermediate is largest");
  }
  return narrowed;
}

/** `combined_binary32` of a list of factors. */
inline float combined_binary32(const std::initializer_list<double> factors,
                               const int exponent) {
  return combined_binary32(factors.begin(), factors.end(), exponent);
}

} // namespace solprop_native
