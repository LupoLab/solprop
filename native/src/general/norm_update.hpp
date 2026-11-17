// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// A step's update with its three norms in one pass: UpdateColumn (a column's borrowed
// pointers), the reduction's split (host_reduction_team_size on the host,
// reduction_slots and slot_range for a device's teams per column) and accept_update.
// plasma_tile_samples, the tile of a device trapezoid plasma's scans, sits first, in
// kernels.hpp's order. A part of kernels.hpp, included through it.
#ifdef SOLPROP_NATIVE_STRICT_TU
#error "a strict translation unit must not include general/norm_update.hpp"
#endif

#include "exponentials.hpp"
#include "reductions.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/launch.hpp"

#include <algorithm>
#include <cstddef>

namespace solprop_native::general {

// The samples of one tile of the device trapezoid plasma's two-level scans (unused on
// a host build).
[[maybe_unused]] inline constexpr std::size_t plasma_tile_samples = 1024;

// Descriptors borrow allocations retained by the owning GeneralWorkspace. They
// contain no ownership or reference counts and are refreshed after state swaps.
struct UpdateColumn {
  const Complex *current{}, *solution{}, *error{}, *whole{};
  Complex *advanced{}, *residual{};
  const unsigned char* in_norm{};
  double *old_partial{}, *new_partial{}, *error_partial{};
  // The state the solution is added to: `current` itself, or a coupled flow's
  // whole-step action of it (solprop `accept_step(U u, ..., U k)`); the old norm is
  // always taken on `current`.
  const Complex* carried{};
};

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
/**
 * The fewest bins a host thread of a reduction team takes: below it the team's
 * synchronisation costs more than the thread's share of the pass.
 */
constexpr std::size_t host_reduction_bins_per_thread = 2048;

/** A host team per column, sharing the threads when the columns are fewer. */
inline int host_reduction_team_size(const ExecutionSpace& execution,
                                    const std::size_t columns, const std::size_t bins) {
  const auto threads = static_cast<std::size_t>(std::max(1, execution.concurrency()));
  const auto per_column =
      std::max<std::size_t>(1, threads / std::max<std::size_t>(1, columns));
  const auto by_bins = std::max<std::size_t>(1, bins / host_reduction_bins_per_thread);
  return static_cast<int>(std::min(per_column, by_bins));
}
#endif

// The teams a column's norm reductions are split over on a device, each writing its
// own partial sums: one team per column fills a GPU only when the columns are many, so
// a one-node fibre's column takes up to `reduction_target_teams` teams of at least
// `reduction_min_bins` bins each. The host keeps one slot per column. The slots depend
// on the state's columns and bins alone, so every run of one problem sums in one
// order.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
constexpr std::size_t reduction_target_teams = 256;
constexpr std::size_t reduction_min_bins = 1024;
#endif
inline std::size_t reduction_slots(const std::size_t columns, const std::size_t n) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const std::size_t wanted =
      (reduction_target_teams + std::max<std::size_t>(columns, 1) - 1) /
      std::max<std::size_t>(columns, 1);
  const std::size_t most =
      std::max<std::size_t>(1, (n + reduction_min_bins - 1) / reduction_min_bins);
  return std::max<std::size_t>(1, std::min(wanted, most));
#else
  static_cast<void>(columns);
  static_cast<void>(n);
  return 1;
#endif
}

// The bins `[first, end)` of slot `slot` of `slots` over `n`.
KOKKOS_INLINE_FUNCTION void slot_range(const std::size_t n, const std::size_t slots,
                                       const std::size_t slot, std::size_t& first,
                                       std::size_t& end) {
  const std::size_t width = (n + slots - 1) / slots;
  first = Kokkos::min(n, slot * width);
  end = Kokkos::min(n, first + width);
}

// The update and its three binary64 norms in one pass. On the host one team per column,
// its threads splitting the bins when the columns are fewer than the threads; on a
// device `slots` teams per column, each over its own bins, writing its partial sums at
// offset `slot` from the column's partial pointers. The sums associate as the teams
// split them (solprop SEP 25), and the host reduces them.
template <class Columns>
void accept_update(const ExecutionSpace& execution, const std::size_t count,
                   const std::size_t n, const double dz, const Columns& columns,
                   const char* label, const std::size_t slots) {
  using ReductionPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using ReductionMember = ReductionPolicy::member_type;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const int team_size = SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE;
#else
  const int team_size = host_reduction_team_size(execution, count, n);
#endif
  const ReductionPolicy reduction_policy(execution, static_cast<int>(count * slots),
                                         team_size);
  launch::parallel_for(
      label, reduction_policy, KOKKOS_LAMBDA(const ReductionMember& team) {
        const auto league = static_cast<std::size_t>(team.league_rank());
        const auto column = league / slots;
        const auto slot = league % slots;
        std::size_t first = 0;
        std::size_t end = 0;
        slot_range(n, slots, slot, first, end);
        const auto values = columns(column);
        NormSums sums;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(first),
                                    static_cast<int>(end)),
            [&](const int index_value, NormSums& update) {
              const auto index = static_cast<std::size_t>(index_value);
              const Complex candidate =
                  values.whole[index] *
                  (values.carried[index] + dz * values.solution[index]);
              values.advanced[index] = candidate;
              const Complex one_residual =
                  values.whole[index] * (dz * values.error[index]);
              values.residual[index] = one_residual;
              if (values.in_norm[index] == 0U)
                return;
              update.old_sum += squared_abs(values.current[index]);
              update.new_sum += squared_abs(candidate);
              update.error_sum += squared_abs(one_residual);
            },
            Kokkos::Sum<NormSums>(sums));
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          values.old_partial[slot] = sums.old_sum;
          values.new_partial[slot] = sums.new_sum;
          values.error_partial[slot] = sums.error_sum;
        });
      });
}

} // namespace solprop_native::general
