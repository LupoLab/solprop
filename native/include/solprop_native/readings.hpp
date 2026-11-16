// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/fields.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <vector>

namespace solprop_native {

/** The boundary monitor's five sums of a state (solprop SEP 21), one reduction value:
    the energy, then the low guard's, the high guard's, the collar's and the rate's. */
struct MonitorSums {
  Kokkos::Array<double, 5> values{};
  KOKKOS_INLINE_FUNCTION MonitorSums& operator+=(const MonitorSums& other) {
    for (std::size_t slot = 0; slot < 5; ++slot)
      values[slot] += other.values[slot];
    return *this;
  }
};

/** A state's whole-state transfers, in bytes, each way (solprop SEP 44): what lets a
   test hold a field-free run to no transfer at all, whichever path would have made one.
 */
struct CrossedBytes {
  std::size_t to_host{};
  std::size_t from_host{};
};

/** What the transverse edge reads of a state (solprop SEP 44): four numbers per weight
    table, and each mode's energy. */
struct EdgeNumbers {
  std::vector<double> sums;
  std::vector<double> columns;
  /** A requested collar's `(inward, windowed)` per line. */
  std::vector<double> strip;
  /** A requested collar's rectified inward current. */
  double current{};
};

/** A requested collar's lines, as Python describes them (solprop SEP 44): for each, its
    axis (0 `r`, 1 `x`, 2 `y`), its count of strip nodes and those nodes, its window,
   its inward bins `[first, last)`, the factor on its strip's sums and the measure on
   its current, and its onset's node, before and after, with the distance between the
    last two. An onset whose node is `no_onset` has no slope to read. */
struct EdgeLines {
  static constexpr std::size_t no_onset = static_cast<std::size_t>(-1);
  std::vector<std::size_t> axes, counts, positions, inward, onsets;
  std::vector<double> windows, factors, measures, steps, scale;
  double k_min{};
  /** The most one batch of a padded transform holds. */
  std::size_t batch_bytes{};
};

/** What a split-step collar's finish reads of the power it finds (solprop SEP 21,
    ADR-057), one reduction value: the energy it takes, then the energy, the low
    guard's, the high guard's and the collar's, which the monitor judges the
    occupancies on. */
struct SplitSums {
  Kokkos::Array<double, 5> values{};
  KOKKOS_INLINE_FUNCTION SplitSums& operator+=(const SplitSums& other) {
    for (std::size_t slot = 0; slot < 5; ++slot)
      values[slot] += other.values[slot];
    return *this;
  }
};

/** The boundary monitor's occupancy of a state (solprop #440), one reduction value: its
    energy per walk-off bucket. A device's reduction keeps one of these per thread in
    team memory, with no atomics, so its size bounds the buckets (`SPLIT_RATE_BUCKETS`).
 */
inline constexpr std::size_t occupancy_bucket_limit = 128;
// The buckets one reduction holds. A reduction's value has to fit the device's shared
// memory per block, which on HIP's smallest block of 64 threads is `(64 + 2) *
// sizeof(value)`, with the kernel's own static memory beside it within 64 KiB. Kokkos
// checks only the first: a value that fills the block passes its check and the dispatch
// then faults on an AMD Instinct MI325X ("Illegal opcode in command stream"). So a
// reading there takes two passes, leaving half the block free.
#ifdef KOKKOS_ENABLE_HIP
inline constexpr std::size_t occupancy_reduction_slots = 64;
#else
inline constexpr std::size_t occupancy_reduction_slots = occupancy_bucket_limit;
#endif
struct OccupancySums {
  Kokkos::Array<double, occupancy_reduction_slots> values{};
  KOKKOS_INLINE_FUNCTION OccupancySums& operator+=(const OccupancySums& other) {
    for (std::size_t slot = 0; slot < occupancy_reduction_slots; ++slot)
      values[slot] += other.values[slot];
    return *this;
  }
};
#ifdef KOKKOS_ENABLE_HIP
static_assert((64 + 2) * sizeof(OccupancySums) <= 48 * 1024);
#endif

[[nodiscard]] constexpr auto describe(const EdgeLines* /*unused*/) {
  using L = EdgeLines;
  using E = fields::Extent;
  return fields::table<L>(
      "EdgeLines", fields::array("axes", &L::axes, E::of("n_lines")),
      fields::array("counts", &L::counts, E::of("n_lines")),
      fields::array("positions", &L::positions, E::any()),
      fields::array("windows", &L::windows, E::any()),
      fields::array("inward", &L::inward, E::any()),
      fields::array("factors", &L::factors, E::of("n_lines")),
      fields::array("measures", &L::measures, E::of("n_lines")),
      fields::array("onsets", &L::onsets, E::any()),
      fields::array("steps", &L::steps, E::of("n_lines")),
      fields::value("k_min", &L::k_min), fields::array("scale", &L::scale, E::any()),
      fields::value("batch_bytes", &L::batch_bytes));
}

} // namespace solprop_native

template <> struct Kokkos::reduction_identity<solprop_native::MonitorSums> {
  KOKKOS_FUNCTION static solprop_native::MonitorSums sum() { return {}; }
};
template <> struct Kokkos::reduction_identity<solprop_native::OccupancySums> {
  KOKKOS_FUNCTION static solprop_native::OccupancySums sum() { return {}; }
};
template <> struct Kokkos::reduction_identity<solprop_native::SplitSums> {
  KOKKOS_FUNCTION static solprop_native::SplitSums sum() { return {}; }
};
