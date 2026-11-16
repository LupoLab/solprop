// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The general workspace's reduction values: NormSums, the three squared norms an
// accepted step reports, and FluenceMoments, a plane's scaled fluence and its index
// moments, each with its Kokkos::reduction_identity. A part of kernels.hpp, included
// through it: the specialisations name the types `solprop_native::NormSums` and
// `solprop_native::FluenceMoments`, which kernels.hpp's using-directive resolves.
#ifdef SOLPROP_NATIVE_STRICT_TU
#error "a strict translation unit must not include general/reductions.hpp"
#endif

#include "solprop_native/runtime/execution.hpp"

namespace solprop_native::general {
/** The three squared norms an accepted step reports, as one reduction value. */
struct NormSums {
  double old_sum = 0.0;
  double new_sum = 0.0;
  double error_sum = 0.0;
  KOKKOS_INLINE_FUNCTION NormSums& operator+=(const NormSums& other) {
    old_sum += other.old_sum;
    new_sum += other.new_sum;
    error_sum += other.error_sum;
    return *this;
  }
};
} // namespace solprop_native::general

template <> struct Kokkos::reduction_identity<solprop_native::NormSums> {
  KOKKOS_FUNCTION static solprop_native::NormSums sum() { return {}; }
};

namespace solprop_native::general {
/** A plane's scaled fluence, one reduction value: its sum and its index moments. */
struct FluenceMoments {
  double total = 0.0;
  double x = 0.0;
  double y = 0.0;
  KOKKOS_INLINE_FUNCTION FluenceMoments& operator+=(const FluenceMoments& other) {
    total += other.total;
    x += other.x;
    y += other.y;
    return *this;
  }
};
} // namespace solprop_native::general

template <> struct Kokkos::reduction_identity<solprop_native::FluenceMoments> {
  KOKKOS_FUNCTION static solprop_native::FluenceMoments sum() { return {}; }
};
