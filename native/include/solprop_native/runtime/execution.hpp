// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/runtime/fields.hpp"

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace solprop_native {

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;
using Layout = Kokkos::LayoutRight;
/** A two-dimensional range whose last index runs fastest, as `Layout` stores it.

    Kokkos's CUDA and HIP default runs the first index fastest, which puts adjacent
    threads a whole row apart in a `LayoutRight` view; a host's default is already
    this order. */
using Range2Policy = Kokkos::MDRangePolicy<
    ExecutionSpace, Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;

/** Samples in one host tile of a `Range2`. Swept from 64 to 1024 on the four-mode RDW
    at 16 threads (an Apple M5 Max and an AMD EPYC 7763): 64 to 256 were within 2 %
    of each other and 1024 the slowest; 64 balances best, a single 2561-bin column
    making 41 tiles rather than 11. */
inline constexpr std::int64_t host_range2_tile = 64;

/** `Range2Policy` on `space` over `[lower, upper)`.

    On the host a tile holds about `host_range2_tile` samples: part of one row, or as
    many whole rows as fit when a row is shorter. Kokkos's host default tiles two rows
    by the whole of the last extent, which leaves a kernel over `(columns, n)` only
    ceil(columns / 2) tiles to share: one for a single column, two for four modes,
    whatever the thread count. A one-thread instance keeps Kokkos's default: it has
    nothing to share, and the small tiles cost the four-mode RDW about 1 % at one
    thread on an AMD EPYC 7763. Device builds keep Kokkos's default too. A kernel over
    a `Range2` that is elementwise changes no value with the tiling; a reduction over
    one (the boundary monitor's split removal) may add in another order, within its
    rounding (solprop SEP 25). */
template <typename LT, std::size_t LN, typename UT, std::size_t UN>
[[nodiscard]] Range2Policy Range2(const ExecutionSpace& space, const LT (&lower)[LN],
                                  const UT (&upper)[UN]) {
  static_assert(LN == 2 && UN == 2, "Range2 takes two lower and two upper bounds");
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  return Range2Policy(space, lower, upper);
#else
  if (space.concurrency() == 1)
    return Range2Policy(space, lower, upper);
  const auto row = std::clamp<std::int64_t>(static_cast<std::int64_t>(upper[1]) -
                                                static_cast<std::int64_t>(lower[1]),
                                            1, host_range2_tile);
  const std::int64_t tile[2] = {host_range2_tile / row, row};
  return Range2Policy(space, lower, upper, tile);
#endif
}
using Complex = Kokkos::complex<double>;

} // namespace solprop_native

// NumPy's complex128 is two doubles, real part first, as `Complex` is.
template <> struct solprop_native::fields::ElementDtype<solprop_native::Complex> {
  static constexpr Dtype value = Dtype::complex128;
};

namespace solprop_native {

using ComplexView = Kokkos::View<Complex*, Layout, MemorySpace>;
using ConstComplexView = Kokkos::View<const Complex*, Layout, MemorySpace>;
using RealView = Kokkos::View<double*, Layout, MemorySpace>;
using ConstRealView = Kokkos::View<const double*, Layout, MemorySpace>;

/**
 * The vector length of the general source's teams: the adjacent output bins one
 * projection team accumulates, and the lanes a device gives a node's samples.
 *
 * The host runs a team's vector range as one vectorised loop and ignores the
 * length; for the projection it is the block of bins (measured on the general
 * route's source on native CPU before the first public release).
 */
inline constexpr std::size_t general_vector_lanes = 8;

} // namespace solprop_native
