// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/core.hpp"
#include "solprop_native/fp.hpp"

#include <gtest/gtest.h>

namespace {
namespace native = solprop_native;
using native::Complex;

// a b + c with a = 1 + 2^-30, b = 1 - 2^-30 and c = -1: the exact product 1 - 2^-60
// rounds to 1, so a product and a sum rounded separately give 0, and a fused
// multiply-add gives -2^-60. Volatile reads keep the compiler from folding either.
volatile double factor_a = 1.0 + 0x1p-30;
volatile double factor_b = 1.0 - 0x1p-30;
volatile double addend = -1.0;
constexpr double fused = -0x1p-60;

[[gnu::noinline]] double ordinary(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}

SOLPROP_NATIVE_STRICT_FUNCTION
[[gnu::noinline]] double protected_block(const double a, const double b,
                                         const double c) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const double product = a * b;
  return product + c;
}

// A strict helper inlined into a protected function stays strict.
SOLPROP_NATIVE_STRICT_FUNCTION
[[gnu::noinline]] double protected_helpers(const double a, const double b,
                                           const double c) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const Complex sum = native::strict_accumulate(Complex(c, 0.0), a, Complex(b, 0.0));
  const Complex product = native::strict_mul(Complex(a, 0.0), Complex(b, 0.0));
  return sum.real() + native::strict_add(product, Complex(c, 0.0)).real();
}
} // namespace

TEST(FloatingPointPolicy, ProtectedBlocksRoundEveryOperation) {
  EXPECT_EQ(protected_block(factor_a, factor_b, addend), 0.0);
  EXPECT_EQ(protected_helpers(factor_a, factor_b, addend), 0.0);
}

TEST(FloatingPointPolicy, OrdinaryCodeContracts) {
#if defined(__OPTIMIZE__) && defined(__FP_FAST_FMA) && !defined(__CUDACC__) &&         \
    !defined(__HIPCC__)
  EXPECT_EQ(ordinary(factor_a, factor_b, addend), fused);
#else
  const double value = ordinary(factor_a, factor_b, addend);
  EXPECT_TRUE(value == 0.0 || value == fused);
  GTEST_SKIP() << "an unoptimised build, or a target without fused multiply-add, "
                  "need not contract";
#endif
}
