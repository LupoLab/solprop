// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/core.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

// solprop SEP 18's open question Q8: whether a team stages each band filter's input row
// in team scratch, tile by tile, or reads the filter's halo directly from the row. Set
// by the CMake option of the same name for a device trial; direct reads are the
// default. The two read the same samples through the same function.
#ifndef SOLPROP_NATIVE_BAND_STAGING
#define SOLPROP_NATIVE_BAND_STAGING 0
#endif

namespace solprop_native {

/*
 * The field plasma column's running integrals under solprop SEP 18's band rule.
 *
 * Under solprop SEP 27 the production column forms two running integrals per
 * node: the exposure int W and the free current int f E. Its total current is
 * assembled through the two collared spectra; it does not integrate polarisation.
 * The band rule replaces each cumulative
 * trapezoid by the trapezoid of the integrand filtered with symmetric taps,
 *
 *   y~_n = c_0 y_n + sum_{k=1..K} c_k (y_{n-k} + y_{n+k}),   K <= 8,
 *
 * summed centre first, each pair added before it is weighted, with the window's ends
 * closed by point reflection (y_{-m} = 2 y_0 - y_m, y_{N-1+m} = 2 y_{N-1} -
 * y_{N-1-m}) and the exposure floored at zero where the fraction is formed. The loss
 * current keeps the raw rate. The trapezoid rule uses the identity tap; its fused
 * column writes the same separate current and non-plasma polarisation streams.
 *
 * A CPU build runs it as a team column (`GeneralWorkspace::team_band_plasma_column`,
 * src/general.cpp): the field is a row over t, and each running integral is a band
 * filter over the row (`team_band_filter`, a parallel stencil with a K-sample halo,
 * reflected at the row's ends) followed by a chunked trapezoid scan of the filtered row
 * (`team_trapezoid_window`). These are solprop SEP 25 section 6.7.4
 * whole-row passes, with the final polarisation integral removed by SEP 27:
 *
 *   1. per channel: rates (parallel over t; the raw rates are kept for the loss current
 *      and the work diagnostic) -> filter -> scan -> exposure; the floor and the
 *      fraction pointwise, and with them the loss guard and the free-current weight (a
 *      mixture adds each channel's share in channel order);
 *   2. per component: filter the free-current integrand -> scan; the current is formed
 *      from the prefix, the guard and the field as the scan stores it.
 *
 * Kerr, Raman drive and their collar follow in a separate pointwise pass. Persistent
 * current streams form gJ and g'J for the held-source spectral assembly.
 *
 * A device runs the same passes as the tiled column below (one team per tile of each
 * row; `SOLPROP_NATIVE_BAND_TILES=0` keeps this team column, where a team owns one
 * node and is as wide as the build's response team, `response_team_size`, unless the
 * kernel's resources need it narrower). On the host
 * (solprop SEP 25 section 6.7.5) the teams split the threads between the chunk's nodes:
 * a chunk of fewer nodes than threads gives each node a team of several threads over
 * chunks of t, and a chunk of many nodes one thread per node, whose scans are then
 * serial (`band_team_policy`). Each team keeps its own scratch rows and walks its share
 * of the nodes, so the rows are sized by the concurrent teams, not by the chunk. A host
 * team also skips the work outside each row's live window (`band_support`): wherever
 * every sample a filtered value reads is exactly zero the filtered value is zero, the
 * prefix keeps its value and an unchanged exposure its fraction, so those samples are
 * written, not computed. A symmetric filter responds K samples before its first
 * non-zero input, and preionisation makes the current live from the first sample, so
 * the windows come from the rows themselves.
 *
 * Each stencil sample is the same function of the same inputs whatever the team size
 * (`band_filter_at`). The scans associate as the team splits the row: Kokkos's host
 * team scan sums each thread's contiguous chunk of m = ceil(n / T) samples, adds the T
 * chunk totals serially, and re-adds each chunk from its offset, so a prefix sum is
 * accumulated through at most D = 2m + T - 3 additions (D = n - 1 for one thread), at
 * most n + T for every T; a device team's tree has its own depth. solprop SEP 25
 * section 4.6 bounds every prefix by gamma_D times the magnitudes entering it, which
 * the parity tests carry. The statistics and the run diagnostics are the same first
 * stage per channel followed by team reductions (maxima, and the paid work's sum) over
 * the same rows (`team_band_statistics`).
 *
 * Q8, the halo read: by default each sample reads its 2K + 1 inputs straight from the
 * row. With SOLPROP_NATIVE_BAND_STAGING the team first copies a tile of
 * `band_staging_tile` samples and its halo into team scratch (level 0), then filters
 * the tile from there. The arithmetic is the same function on the same values; which is
 * faster is the trial's measurement.
 */

/**
 * Form a per-row inclusive trapezoidal prefix with one Kokkos team per row.
 *
 * The device trapezoid column's scan primitive: the response computes rates in
 * parallel, calls this scan, and then applies its current/Kerr pass in parallel over
 * time. The host's trapezoid column keeps one node per work item and a serial scan.
 */
template <bool Envelope = false, class ExecutionSpace, class InputView,
          class OutputView>
void portable_team_trapezoid_scan(const ExecutionSpace& execution_space,
                                  const InputView& values, const OutputView& prefixes,
                                  const double spacing) {
  if (values.extent(0) != prefixes.extent(0) || values.extent(1) != prefixes.extent(1))
    throw std::invalid_argument("plasma team-scan view extents are inconsistent");
  if (!(spacing > 0.0))
    throw std::invalid_argument("plasma team-scan spacing must be positive");

  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const auto rows = values.extent(0);
  const auto samples = values.extent(1);
  if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      samples > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("plasma team-scan extent exceeds Kokkos policy range");
  const auto league_size = static_cast<int>(rows);
  const auto team_samples = static_cast<int>(samples);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const TeamPolicy policy(execution_space, league_size,
                          SOLPROP_NATIVE_DEVICE_TEAM_SIZE);
#else
  // One thread per envelope row fixes prefix association across CPU thread counts.
  const TeamPolicy policy =
      Envelope ? TeamPolicy(execution_space, league_size, 1)
               : TeamPolicy(execution_space, league_size, Kokkos::AUTO());
#endif
  launch::parallel_for(
      "solprop_native.plasma.portable_team_scan", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto row = static_cast<std::size_t>(team.league_rank());
        Kokkos::parallel_scan(
            Kokkos::TeamThreadRange(team, team_samples),
            [&](const int team_sample, double& update, const bool final) {
              const auto sample = static_cast<std::size_t>(team_sample);
              if (sample != 0) {
                // nvcc requires first captures outside constexpr-if branches.
                const double previous = values(row, sample - 1);
                const double current = values(row, sample);
                const double interval = spacing;
                if constexpr (Envelope)
                  update += interval * (0.5 * previous + 0.5 * current);
                else
                  update += interval * (previous + current) / 2.0;
              }
              if (final)
                prefixes(row, sample) = update;
            });
      });
}

/** Whether band filters stage their rows in team scratch by default (Q8). */
inline constexpr bool band_staging = SOLPROP_NATIVE_BAND_STAGING != 0;

/** Samples one staged tile filters; the tile's scratch holds 2K more. */
inline constexpr int band_staging_tile = 1024;

/** The widest band filter's half-width K. */
inline constexpr int band_max_half = static_cast<int>(plasma_max_taps) - 1;

/** Samples one staged tile's scratch holds: the tile and the widest halo. */
inline constexpr int band_staging_extent = band_staging_tile + 2 * band_max_half;

/**
 * Whether host band filters run blocks of samples tap pair by tap pair
 * (`team_band_filter`): x86-64 hosts only. On an Apple M-series host an earlier
 * tap-first traversal slowed the argon window by 6-9 % (docs/dev/benchmarks.md,
 * "Rejected leads"), where on an Intel Xeon Gold 6240 this one takes the CTC column
 * from 0.69 s to 0.54 s of a one-thread window.
 */
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) &&                     \
    (defined(__x86_64__) || defined(_M_X64))
inline constexpr bool band_block_filter = true;
#else
inline constexpr bool band_block_filter = false;
#endif

/** Samples a host thread filters at a time with `band_block_filter`. */
inline constexpr int band_host_filter_block = 512;

/** A staged band filter's tile in a team's scratch. */
template <class ScratchSpace>
using BandTile =
    Kokkos::View<double*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/** Team-scratch bytes the band filters need per team: none for direct reads. */
template <bool Staged, class ExecutionSpace> std::size_t band_staging_bytes() {
  if constexpr (Staged)
    return BandTile<typename ExecutionSpace::scratch_memory_space>::shmem_size(
        static_cast<std::size_t>(band_staging_extent));
  else
    return 0;
}

/**
 * The team's band tile: taken from its level-0 scratch once per kernel when staged, and
 * an empty view otherwise.
 *
 * Every view built on `team.team_scratch(0)` takes fresh scratch, so a kernel that
 * filters several rows builds its tile once and passes it to each `team_band_filter`.
 */
template <bool Staged, class TeamMember>
KOKKOS_INLINE_FUNCTION BandTile<typename TeamMember::scratch_memory_space>
band_tile(const TeamMember& team) {
  using Tile = BandTile<typename TeamMember::scratch_memory_space>;
  if constexpr (Staged)
    return Tile(team.team_scratch(0), static_cast<std::size_t>(band_staging_extent));
  else
    return Tile();
}

/**
 * Filter sample `index` of a row with solprop SEP 18's symmetric taps.
 *
 * `c_0 y_n + sum_k c_k (y_{n-k} + y_{n+k})`: the centre, then each pair added before it
 * is weighted; the compiler may contract the multiply-adds (solprop SEP 25). Past
 * either end a sample is its point reflection, `2 y_0 - y_m` or `2 y_{N-1} -
 * y_{N-1-m}`; `samples > half` keeps every reflected index in the row. `sample(i)`
 * reads y_i. Its rounding is within gamma_{2K+2} of `|c_0 y_n| + sum_k |c_k| (|y_{n-k}|
 * + |y_{n+k}|)` for interior samples, and the reflections add their own two roundings.
 */
template <class Sample>
KOKKOS_INLINE_FUNCTION double band_filter_at(const Sample& sample, const int index,
                                             const int samples, const double* taps,
                                             const int half) {
  const int last = samples - 1;
  if (index >= half && index + half <= last) {
    double total = taps[0] * sample(index);
    for (int k = 1; k <= half; ++k)
      total += taps[k] * (sample(index - k) + sample(index + k));
    return total;
  }
  double total = taps[0] * sample(index);
  for (int k = 1; k <= half; ++k) {
    const int low = index - k;
    const int high = index + k;
    const double before = low >= 0 ? sample(low) : 2.0 * sample(0) - sample(-low);
    const double after =
        high <= last ? sample(high) : 2.0 * sample(last) - sample(2 * last - high);
    total += taps[k] * (before + after);
  }
  return total;
}

/**
 * The samples of a band-filtered row that can differ from zero: `[first, last]`, empty
 * when `first > last`.
 *
 * `live_first` and `live_last` are the row's first and last non-zero samples
 * (`live_first > live_last` for a row of zeros). A filtered sample reads the samples
 * within K of it and, within K of an end, that end's sample and those within K of it;
 * where all of them are zero it is exactly zero. So the window is the live samples
 * widened by K, or reaching an end where the live samples come within K of it.
 */
struct BandWindow {
  int first;
  int last;
};

KOKKOS_INLINE_FUNCTION BandWindow band_support(const int live_first,
                                               const int live_last, const int samples,
                                               const int half) {
  if (live_first > live_last)
    return {samples, -1};
  return {live_first > half ? live_first - half : 0,
          live_last + half < samples - 1 ? live_last + half : samples - 1};
}

/**
 * Write the band-filtered `source(source_row, :)` into `target(target_row, :)`.
 *
 * Called by every thread of one team; a parallel map over the row's samples, each
 * sample's sum in `band_filter_at`'s fixed order, so the result does not depend on the
 * team size. `Staged` copies each tile of `band_staging_tile` samples and its halo into
 * `tile` first (`band_tile`; the policy must provide `band_staging_bytes`). Ends with a
 * team barrier, so the target row is complete on return. `taps` is the caller's
 * thread-local copy of the K + 1 taps. `begin` and `stop` limit the samples written
 * (the whole row by default); the halo reads stay within the row.
 */
template <bool Staged, class TeamMember, class Tile, class Source, class Target>
KOKKOS_INLINE_FUNCTION void
team_band_filter(const TeamMember& team, const Tile& tile, const Source& source,
                 const std::size_t source_row, const Target& target,
                 const std::size_t target_row, const int samples, const double* taps,
                 const int half, const int begin = 0, const int stop = -1) {
  const int finish = stop < 0 ? samples : stop;
  if constexpr (Staged) {
    for (int first = begin; first < finish; first += band_staging_tile) {
      const int end = Kokkos::min(first + band_staging_tile, finish);
      // The tile's halo: K samples either side, within the row. A reflection reads
      // y_0 or y_{N-1} and a sample within K of it, which the halo then holds.
      const int low = Kokkos::max(first - half, 0);
      const int high = Kokkos::min(end + half, samples);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, low, high), [&](const int i) {
        tile(static_cast<std::size_t>(i - low)) =
            source(source_row, static_cast<std::size_t>(i));
      });
      team.team_barrier();
      const auto staged = [&](const int i) {
        return tile(static_cast<std::size_t>(i - low));
      };
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, first, end),
                           [&](const int index) {
                             target(target_row, static_cast<std::size_t>(index)) =
                                 band_filter_at(staged, index, samples, taps, half);
                           });
      team.team_barrier();
    }
  } else {
    static_cast<void>(tile);
    const auto direct = [&](const int i) {
      return source(source_row, static_cast<std::size_t>(i));
    };
    if constexpr (band_block_filter) {
      // On an x86-64 host each thread takes blocks of samples and runs a block's
      // interior samples as loops over the block, one per tap pair, so they vectorise
      // across the samples; each sample's sum keeps `band_filter_at`'s order.
      const double* const row = &source(source_row, 0);
      double* const out = &target(target_row, 0);
      const int interior_end = samples - half;
      const int blocks =
          (finish - begin + band_host_filter_block - 1) / band_host_filter_block;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, blocks), [&](const int block) {
        const int low = begin + block * band_host_filter_block;
        const int high = Kokkos::min(low + band_host_filter_block, finish);
        const int inner_low = Kokkos::min(Kokkos::max(low, half), high);
        const int inner_high = Kokkos::max(Kokkos::min(high, interior_end), inner_low);
        for (int index = low; index < inner_low; ++index)
          out[index] = band_filter_at(direct, index, samples, taps, half);
        for (int index = inner_low; index < inner_high; ++index)
          out[index] = taps[0] * row[index];
        for (int k = 1; k <= half; ++k) {
          const double tap = taps[k];
          for (int index = inner_low; index < inner_high; ++index)
            out[index] += tap * (row[index - k] + row[index + k]);
        }
        for (int index = inner_high; index < high; ++index)
          out[index] = band_filter_at(direct, index, samples, taps, half);
      });
    } else {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, begin, finish),
                           [&](const int index) {
                             target(target_row, static_cast<std::size_t>(index)) =
                                 band_filter_at(direct, index, samples, taps, half);
                           });
    }
    team.team_barrier();
  }
}

/**
 * The inclusive trapezoidal prefix of `values(row, :)`, one team, `store(n, prefix)`.
 *
 * The device column's fixed-tree scan (`portable_team_trapezoid_scan`'s increment,
 * `spacing (y_n + y_{n-1}) / 2`); `store` receives each sample's prefix once, in the
 * scan's final pass, so it may form a pointwise quantity from it. Ends with a team
 * barrier.
 */
template <class TeamMember, class Values, class Store>
KOKKOS_INLINE_FUNCTION void
team_trapezoid_prefix(const TeamMember& team, const Values& values,
                      const std::size_t row, const int samples, const double spacing,
                      const Store& store) {
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  // A host team of one thread forms the same prefixes in one pass, as
  // `team_trapezoid_window` does.
  if (team.team_size() == 1) {
    double update = 0.0;
    for (int sample = 0; sample < samples; ++sample) {
      const auto index = static_cast<std::size_t>(sample);
      if (index != 0)
        update += spacing * (values(row, index) + values(row, index - 1)) / 2.0;
      store(sample, update);
    }
    team.team_barrier();
    return;
  }
#endif
  Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, samples),
                        [&](const int sample, double& update, const bool final) {
                          const auto index = static_cast<std::size_t>(sample);
                          if (index != 0) {
                            const double current = values(row, index);
                            const double previous = values(row, index - 1);
                            update += spacing * (current + previous) / 2.0;
                          }
                          if (final)
                            store(sample, update);
                        });
  team.team_barrier();
}

/**
 * The inclusive trapezoidal prefix of a row that is zero outside `window`, one team.
 *
 * `values(row, n)` is read only for n in `window`, and every other sample is taken as
 * exactly zero (the filtered row of `band_support`, whose other entries may be stale).
 * The prefix is zero before the window and constant from one past its last sample on,
 * so the scan runs over `[first, last + 1]` alone and `store(n, prefix)` receives those
 * constants for the remaining samples without forming them. Returns the final prefix.
 * Ends with a team barrier.
 */
template <class TeamMember, class Values, class Store>
KOKKOS_INLINE_FUNCTION double
team_trapezoid_window(const TeamMember& team, const Values& values,
                      const std::size_t row, const BandWindow window, const int samples,
                      const double spacing, const Store& store) {
  const int first = window.first;
  const int end = Kokkos::min(window.last + 2, samples);
  const auto value = [&](const int index) {
    return index >= first && index <= window.last
               ? values(row, static_cast<std::size_t>(index))
               : 0.0;
  };
  double total = 0.0;
  bool scanned = false;
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  // A host team of one thread forms the same prefixes in one pass: its Kokkos scan
  // runs the increments twice, the second time from the same zero, in the same order.
  if (team.team_size() == 1) {
    double previous = 0.0;
    for (int sample = first; sample < end; ++sample) {
      const double current =
          sample <= window.last ? values(row, static_cast<std::size_t>(sample)) : 0.0;
      if (sample != 0)
        total += spacing * (current + previous) / 2.0;
      previous = current;
      store(sample, total);
    }
    scanned = true;
  }
#endif
  if (!scanned && first < end)
    Kokkos::parallel_scan(
        Kokkos::TeamThreadRange(team, first, end),
        [&](const int sample, double& update, const bool final) {
          if (sample != 0)
            update += spacing * (value(sample) + value(sample - 1)) / 2.0;
          if (final)
            store(sample, update);
        },
        total);
  const int head = Kokkos::min(first, samples);
  const int tail = first < end ? end : samples;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, head),
                       [&](const int sample) { store(sample, 0.0); });
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, tail, samples),
                       [&](const int sample) { store(sample, total); });
  team.team_barrier();
  return total;
}

/*
 * The tiled band column (solprop SEP 25 section 6.7.6), the device form. The team
 * column gives each node one team that walks its whole row, so a chunk of a few dozen
 * nodes occupies a few dozen of a GPU's multiprocessors and each team's scans step
 * through tens of thousands of samples in turn. The tiled form splits every row into
 * tiles of `band_tile_samples` samples and gives each (row, tile) its own team, so the
 * launch is rows x tiles teams wide. A running integral is then two kernels:
 *
 *   local:  each team loads its tile's source samples and their halo into team
 *           scratch (`team_band_tile`; the first pass forms the rates there), filters
 *           them, and scans the tile's trapezoid increments with a Kokkos team scan
 *           from zero: the tile's local prefixes and its total;
 *   finish: each team sums the row's earlier tile totals with a Kokkos team reduction
 *           (`team_tile_offset`), adds that offset to each local prefix and forms the
 *           pass's pointwise product from the prefix.
 *
 * A prefix is thus a sum over a tree (the tile's own scan, the reduction of the
 * earlier totals, and the one addition joining them) in which every addition sums a
 * contiguous range of increments, so the scan bound of solprop SEP 25 section 4.6
 * holds as for the team column: a prefix's tree has n - 1 additions and a depth of at
 * most tile + tiles <= n_to + 1 (tests/plasma_bounds.py, `scan_depth`). Each filtered
 * sample is `band_filter_at` of the same inputs, so only the association differs from
 * the team column.
 */

/** The fewest samples in a band tile. */
inline constexpr std::size_t band_tile_min = 256;
/** The most samples in a band tile: its two scratch rows then take about 16.5 kB,
 * within every execution space's level-0 team scratch (32 kB on the host). */
inline constexpr std::size_t band_tile_max = 1024;
/** The tiles a row is split into when its length allows. */
inline constexpr std::size_t band_tiles_per_row = 32;

/**
 * The tiled band column's tile for rows of `samples`: `samples / 32` rounded down to a
 * power of two, within [`band_tile_min`, `largest`] (`largest` at most `band_tile_max`,
 * smaller when a device's team scratch is). A problem constant, so a row's association
 * does not depend on how its nodes are chunked.
 */
inline std::size_t band_tile_samples(const std::size_t samples,
                                     const std::size_t largest = band_tile_max) {
  std::size_t tile = band_tile_min;
  while (2 * tile <= largest && 2 * tile * band_tiles_per_row <= samples)
    tile *= 2;
  return tile;
}

/** The tiles a row of `samples` is split into. */
inline std::size_t band_tile_count(const std::size_t samples, const std::size_t tile) {
  return (samples + tile - 1) / tile;
}

/** Team-scratch bytes of one tile kernel's team: the source window, `tile + 2K + 1`
 * samples, and the filtered tile, `tile + 1`. */
template <class ExecutionSpace> std::size_t band_tile_bytes(const std::size_t tile) {
  using Scratch = BandTile<typename ExecutionSpace::scratch_memory_space>;
  return Scratch::shmem_size(tile + 2 * static_cast<std::size_t>(band_max_half) + 1) +
         Scratch::shmem_size(tile + 1);
}

/**
 * One tile of a band-filtered running integral: its local prefixes, from zero.
 *
 * Called by every thread of the team that owns samples `[first, end)` of a row of
 * `samples`, with `band_tile_bytes` of level-0 scratch. `load(i)` returns source
 * sample i (the first pass forms a rate there and keeps the tile's own). The team
 * loads the tile and its halo, `[max(first - 1 - K, 0), min(end + K, samples))`, which
 * holds every sample the filter reads for `[max(first - 1, 0), end)`, reflections
 * included (the row has more than K samples); filters those samples
 * (`band_filter_at`); then scans the trapezoid increments `spacing (y_n + y_{n-1}) / 2`
 * of `[first, end)` (none at n = 0) with a Kokkos team scan from zero, `store(n,
 * local)` receiving each inclusive local prefix once. Returns the tile's total and
 * ends with a team barrier.
 */
template <class TeamMember, class Load, class Store>
KOKKOS_INLINE_FUNCTION double
team_band_tile(const TeamMember& team, const Load& load, const int first, const int end,
               const int samples, const double* taps, const int half,
               const double spacing, const Store& store) {
  using Tile = BandTile<typename TeamMember::scratch_memory_space>;
  const int low = Kokkos::max(first - 1 - half, 0);
  const int high = Kokkos::min(end + half, samples);
  const int start = Kokkos::max(first - 1, 0);
  const Tile source(team.team_scratch(0), static_cast<std::size_t>(high - low));
  const Tile filtered(team.team_scratch(0), static_cast<std::size_t>(end - start));
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, low, high), [&](const int i) {
    source(static_cast<std::size_t>(i - low)) = load(i);
  });
  team.team_barrier();
  const auto window = [&](const int i) {
    return source(static_cast<std::size_t>(i - low));
  };
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, start, end), [&](const int i) {
    filtered(static_cast<std::size_t>(i - start)) =
        band_filter_at(window, i, samples, taps, half);
  });
  team.team_barrier();
  double total = 0.0;
  Kokkos::parallel_scan(
      Kokkos::TeamThreadRange(team, first, end),
      [&](const int sample, double& update, const bool final) {
        if (sample != 0) {
          const double current = filtered(static_cast<std::size_t>(sample - start));
          const double previous =
              filtered(static_cast<std::size_t>(sample - 1 - start));
          update += spacing * (current + previous) / 2.0;
        }
        if (final)
          store(sample, update);
      },
      total);
  team.team_barrier();
  return total;
}

/**
 * The binary64 total of a tile's trapezoid increments, `spacing (y_n + y_{n-1}) / 2`
 * for n in `[first, end)` (none at n = 0), by a team reduction every thread receives;
 * `load(n)` returns y_n widened to binary64. The first level of a tiled running
 * integral: `team_trapezoid_finish` is the second.
 */
template <class TeamMember, class Load>
KOKKOS_INLINE_FUNCTION double
team_trapezoid_total(const TeamMember& team, const Load& load, const int first,
                     const int end, const double spacing) {
  double total = 0.0;
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team, first, end),
      [&](const int sample, double& update) {
        if (sample != 0)
          update += spacing * (load(sample) + load(sample - 1)) / 2.0;
      },
      total);
  return total;
}

/**
 * The inclusive trapezoid prefixes of a tile from `offset` (the row's earlier tile
 * totals, `team_tile_offset`), `store(n, offset + local)` once per sample, the local
 * prefix a binary64 team scan of the tile's increments. Ends with a team barrier.
 */
template <class TeamMember, class Load, class Store>
KOKKOS_INLINE_FUNCTION void
team_trapezoid_finish(const TeamMember& team, const Load& load, const int first,
                      const int end, const double spacing, const double offset,
                      const Store& store) {
  Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, first, end),
                        [&](const int sample, double& update, const bool final) {
                          if (sample != 0)
                            update += spacing * (load(sample) + load(sample - 1)) / 2.0;
                          if (final)
                            store(sample, offset + update);
                        });
  team.team_barrier();
}

/**
 * The offset of tile `tile` of row `row`: the sum of the row's earlier tile totals,
 * `totals(row, 0 .. tile - 1)`, by a Kokkos team reduction every thread receives.
 */
template <class TeamMember, class Totals>
KOKKOS_INLINE_FUNCTION double team_tile_offset(const TeamMember& team,
                                               const Totals& totals,
                                               const std::size_t row, const int tile) {
  double offset = 0.0;
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team, tile),
      [&](const int earlier, double& sum) {
        sum += totals(row, static_cast<std::size_t>(earlier));
      },
      offset);
  return offset;
}

/**
 * Band-filter each row of `values` and scan it into `prefixes` as the tiled column
 * does: one team per (row, tile) forms the local prefixes and totals
 * (`team_band_tile`), then one per (row, tile) adds each tile's offset
 * (`team_tile_offset`). The native tests run it to hold the tiled scan to a serial
 * reference over tiles and team sizes. `locals` and `totals` are its scratch, `(rows,
 * samples)` and `(rows, tiles)`. `team_size` 0 is the build's: the device team width,
 * or one host thread.
 */
template <class ExecutionSpace, class InputView, class LocalView, class TotalView,
          class OutputView, class TapsView>
void portable_tiled_band_scan(const ExecutionSpace& execution_space,
                              const InputView& values, const LocalView& locals,
                              const TotalView& totals, const OutputView& prefixes,
                              const TapsView& taps, const double spacing,
                              const std::size_t tile, const int team_size = 0) {
  const auto rows = values.extent(0);
  const auto samples = values.extent(1);
  if (tile == 0 || tile > band_tile_max)
    throw std::invalid_argument("plasma tiled-scan tile must be 1 to band_tile_max");
  const auto tiles = band_tile_count(samples, tile);
  if (locals.extent(0) != rows || locals.extent(1) != samples ||
      prefixes.extent(0) != rows || prefixes.extent(1) != samples ||
      totals.extent(0) != rows || totals.extent(1) != tiles)
    throw std::invalid_argument("plasma tiled-scan view extents are inconsistent");
  if (taps.extent(0) == 0 || taps.extent(0) > plasma_max_taps)
    throw std::invalid_argument("plasma tiled-scan taps must number one to nine");
  if (!(spacing > 0.0))
    throw std::invalid_argument("plasma tiled-scan spacing must be positive");
  const auto half = static_cast<int>(taps.extent(0) - 1);
  constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (samples > limit || rows > limit / std::max<std::size_t>(tiles, 1) ||
      samples <= static_cast<std::size_t>(half))
    throw std::length_error("plasma tiled-scan rows need more than K samples");
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const auto league = static_cast<int>(rows * tiles);
  const auto width = static_cast<int>(tile);
  const auto row_samples = static_cast<int>(samples);
  const auto row_tiles = static_cast<int>(tiles);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const int threads = team_size > 0 ? team_size : SOLPROP_NATIVE_DEVICE_TEAM_SIZE;
#else
  const int threads = team_size > 0 ? team_size : 1;
#endif
  TeamPolicy local_policy(execution_space, league, threads);
  local_policy.set_scratch_size(0,
                                Kokkos::PerTeam(band_tile_bytes<ExecutionSpace>(tile)));
  launch::parallel_for(
      "solprop_native.plasma.portable_tiled_band_local", local_policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto row = static_cast<std::size_t>(team.league_rank() / row_tiles);
        const int index = team.league_rank() % row_tiles;
        const int first = index * width;
        const int end = Kokkos::min(first + width, row_samples);
        double local[plasma_max_taps] = {};
        for (int k = 0; k <= half; ++k)
          local[k] = taps(static_cast<std::size_t>(k));
        const double total = team_band_tile(
            team, [&](const int i) { return values(row, static_cast<std::size_t>(i)); },
            first, end, row_samples, local, half, spacing,
            [&](const int sample, const double prefix) {
              locals(row, static_cast<std::size_t>(sample)) = prefix;
            });
        Kokkos::single(Kokkos::PerTeam(team),
                       [&] { totals(row, static_cast<std::size_t>(index)) = total; });
      });
  const TeamPolicy finish_policy(execution_space, league, threads);
  launch::parallel_for(
      "solprop_native.plasma.portable_tiled_band_finish", finish_policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto row = static_cast<std::size_t>(team.league_rank() / row_tiles);
        const int index = team.league_rank() % row_tiles;
        const int first = index * width;
        const int end = Kokkos::min(first + width, row_samples);
        const double offset = team_tile_offset(team, totals, row, index);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, first, end),
                             [&](const int i) {
                               const auto sample = static_cast<std::size_t>(i);
                               prefixes(row, sample) = offset + locals(row, sample);
                             });
      });
}

/**
 * The trapezoid running integral of each row of `values` into `prefixes` as the device
 * column's two-level scan does: one team per (row, tile) sums the
 * tile's increments into `totals` (`team_trapezoid_total`), then one per (row, tile)
 * scans the tile again from the row's earlier totals (`team_trapezoid_finish`). The
 * native tests hold it to a serial reference over tiles and team sizes. `team_size` 0
 * is the build's: the device team width, or one host thread.
 */
template <class ExecutionSpace, class InputView, class TotalView, class OutputView>
void portable_tiled_trapezoid_scan(const ExecutionSpace& execution_space,
                                   const InputView& values, const TotalView& totals,
                                   const OutputView& prefixes, const double spacing,
                                   const std::size_t tile, const int team_size = 0) {
  const auto rows = values.extent(0);
  const auto samples = values.extent(1);
  if (tile == 0)
    throw std::invalid_argument("plasma tiled trapezoid tile must be positive");
  const auto tiles = (samples + tile - 1) / tile;
  if (prefixes.extent(0) != rows || prefixes.extent(1) != samples ||
      totals.extent(0) != rows || totals.extent(1) != tiles)
    throw std::invalid_argument("plasma tiled trapezoid view extents are inconsistent");
  constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (samples > limit || rows > limit / std::max<std::size_t>(tiles, 1))
    throw std::length_error("plasma tiled trapezoid rows exceed int extents");
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const auto league = static_cast<int>(rows * tiles);
  const auto width = static_cast<int>(tile);
  const auto row_samples = static_cast<int>(samples);
  const auto row_tiles = static_cast<int>(tiles);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const int threads = team_size > 0 ? team_size : SOLPROP_NATIVE_DEVICE_TEAM_SIZE;
#else
  const int threads = team_size > 0 ? team_size : 1;
#endif
  launch::parallel_for(
      "solprop_native.plasma.portable_tiled_trapezoid_local",
      TeamPolicy(execution_space, league, threads),
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto row = static_cast<std::size_t>(team.league_rank() / row_tiles);
        const int index = team.league_rank() % row_tiles;
        const int first = index * width;
        const int end = Kokkos::min(first + width, row_samples);
        const double total = team_trapezoid_total(
            team,
            [&](const int i) {
              return static_cast<double>(values(row, static_cast<std::size_t>(i)));
            },
            first, end, spacing);
        Kokkos::single(Kokkos::PerTeam(team),
                       [&] { totals(row, static_cast<std::size_t>(index)) = total; });
      });
  launch::parallel_for(
      "solprop_native.plasma.portable_tiled_trapezoid_finish",
      TeamPolicy(execution_space, league, threads),
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto row = static_cast<std::size_t>(team.league_rank() / row_tiles);
        const int index = team.league_rank() % row_tiles;
        const int first = index * width;
        const int end = Kokkos::min(first + width, row_samples);
        const double offset = team_tile_offset(team, totals, row, index);
        team_trapezoid_finish(
            team,
            [&](const int i) {
              return static_cast<double>(values(row, static_cast<std::size_t>(i)));
            },
            first, end, spacing, offset,
            [&](const int sample, const double prefix) {
              prefixes(row, static_cast<std::size_t>(sample)) = prefix;
            });
      });
}

/**
 * Band-filter each row of `values` into `filtered` and scan it into `prefixes`.
 *
 * One team per pair of rows, as the device column treats a node's two components: each
 * row `team_band_filter`, then `team_trapezoid_prefix`, in turn, through one tile. The
 * native tests run it on the host to hold the device building blocks to a serial
 * reference, staged and direct, over team sizes. `team_size` 0 is the build's: the
 * device team width, or Kokkos' choice.
 */
template <bool Staged = band_staging, class ExecutionSpace, class InputView,
          class FilteredView, class OutputView, class TapsView>
void portable_team_band_scan(const ExecutionSpace& execution_space,
                             const InputView& values, const FilteredView& filtered,
                             const OutputView& prefixes, const TapsView& taps,
                             const double spacing, const int team_size = 0) {
  const auto rows = values.extent(0);
  const auto samples = values.extent(1);
  if (filtered.extent(0) != rows || filtered.extent(1) != samples ||
      prefixes.extent(0) != rows || prefixes.extent(1) != samples)
    throw std::invalid_argument("plasma band-scan view extents are inconsistent");
  if (taps.extent(0) == 0 || taps.extent(0) > plasma_max_taps)
    throw std::invalid_argument("plasma band-scan taps must number one to nine");
  if (!(spacing > 0.0))
    throw std::invalid_argument("plasma band-scan spacing must be positive");
  const auto half = static_cast<int>(taps.extent(0) - 1);
  if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      samples > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      samples <= static_cast<std::size_t>(half))
    throw std::length_error("plasma band-scan rows need more than K samples");
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const auto team_samples = static_cast<int>(samples);
  const auto pairs = static_cast<int>((rows + 1) / 2);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  TeamPolicy policy(execution_space, pairs,
                    team_size > 0 ? team_size : SOLPROP_NATIVE_DEVICE_TEAM_SIZE);
#else
  TeamPolicy policy = team_size > 0
                          ? TeamPolicy(execution_space, pairs, team_size)
                          : TeamPolicy(execution_space, pairs, Kokkos::AUTO());
#endif
  if constexpr (Staged)
    policy.set_scratch_size(
        0, Kokkos::PerTeam(band_staging_bytes<Staged, ExecutionSpace>()));
  launch::parallel_for(
      "solprop_native.plasma.portable_team_band_scan", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto first_row = 2 * static_cast<std::size_t>(team.league_rank());
        double local[plasma_max_taps] = {};
        for (int k = 0; k <= half; ++k)
          local[k] = taps(static_cast<std::size_t>(k));
        const auto tile = band_tile<Staged>(team);
        for (auto row = first_row; row < first_row + 2 && row < rows; ++row) {
          team_band_filter<Staged>(team, tile, values, row, filtered, row, team_samples,
                                   local, half);
          team_trapezoid_prefix(team, filtered, row, team_samples, spacing,
                                [&](const int sample, const double prefix) {
                                  prefixes(row, static_cast<std::size_t>(sample)) =
                                      prefix;
                                });
        }
      });
}

} // namespace solprop_native
