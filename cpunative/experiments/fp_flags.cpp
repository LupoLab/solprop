// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Run under each actual compiler/flag/LTO combination before timing that build.
#include "solprop_cpunative/fp.hpp"
#include <cstdio>
#include <limits>

#ifdef __FAST_MATH__
#error "The experiment must retain finite-value checks and protected arithmetic"
#endif

volatile double a = 1.0 + 0x1p-30;
volatile double b = 1.0 - 0x1p-30;
volatile double c = -1.0;
volatile double small = std::numeric_limits<double>::min();
volatile double half = 0.5;
volatile double not_finite = std::numeric_limits<double>::infinity();

SOLPROP_CPUNATIVE_STRICT_FUNCTION
[[gnu::noinline]] double protected_product(double x, double y, double z) {
  SOLPROP_CPUNATIVE_STRICT_BLOCK
  const double product = x * y;
  return product + z;
}

[[gnu::noinline]] double ordinary_product(double x, double y, double z) {
  const double product = x * y;
  return product + z;
}

int main() {
  using namespace solprop_cpunative;
  const double protected_result = protected_product(a, b, c);
  const double helper = strict_accumulate({c, 0.0}, a, {b, 0.0}).real();
  const double subnormal = small * half;
  const double contraction = ordinary_product(a, b, c);
  const bool valid = protected_result == 0.0 && helper == 0.0 &&
                     subnormal == 0x1p-1023 && !std::isfinite(not_finite) &&
                     (contraction == 0.0 || contraction == -0x1p-60);
  std::printf("{\"passed\":%s,\"protected\":\"%a\",\"helper\":\"%a\","
              "\"subnormal\":\"%a\",\"ordinary\":\"%a\"}\n",
              valid ? "true" : "false", protected_result, helper, subnormal,
              contraction);
  return valid ? 0 : 1;
}
