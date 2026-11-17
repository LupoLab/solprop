// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Tests of the plasma band scans (`response/plasma.hpp`): the portable team and tiled
// trapezoid and band-filter scans against serial references, within rounding bounds
// or exactly on integers, their tile sizes, the live-window skip of exact zeros and
// the filter's dynamic range.

#include "solprop_native/response/plasma.hpp"
#include "solprop_native/runtime/execution.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace {

namespace native = solprop_native;

/** An empty team body: what a team-size query is asked about. A functor rather than a
 * lambda, which CUDA refuses inside a test's (private) body. */
struct EmptyTeamBody {
  KOKKOS_INLINE_FUNCTION void operator()(
      const Kokkos::TeamPolicy<native::ExecutionSpace>::member_type& /*team*/) const {}
};

constexpr double kEpsilon = std::numeric_limits<double>::epsilon();

TEST(PlasmaBandScan, PortablePlasmaTeamScanMatchesTheSerialTrapezoid) {
  using View = Kokkos::View<double**, native::Layout, native::MemorySpace>;
  const native::ExecutionSpace execution_space;
  View rates("solprop_native.test.plasma_rates", 2, 8);
  View prefixes("solprop_native.test.plasma_prefixes", 2, 8);
  auto host_rates = Kokkos::create_mirror_view(rates);
  for (std::size_t row = 0; row < 2; ++row)
    for (std::size_t sample = 0; sample < 8; ++sample)
      host_rates(row, sample) = static_cast<double>((row + 1) * (sample + 2)) / 16.0;
  Kokkos::deep_copy(execution_space, rates, host_rates);
  native::portable_team_trapezoid_scan(execution_space, rates, prefixes, 0.125);
  execution_space.fence("solprop_native plasma team-scan test");
  const auto host_prefixes =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, prefixes);
  for (std::size_t row = 0; row < 2; ++row) {
    double expected = 0.0;
    // Exact: the empty prefix.
    EXPECT_EQ(host_prefixes(row, 0), expected);
    for (std::size_t sample = 1; sample < 8; ++sample) {
      expected += 0.125 * (host_rates(row, sample - 1) + host_rates(row, sample)) / 2.0;
      EXPECT_NEAR(host_prefixes(row, sample), expected, 8.0 * kEpsilon);
    }
  }
}

// solprop SEP 18's band filter written out independently: the row extended by point
// reflection, then the centre and the weighted pairs in the reference's order, every
// product and sum its own statement.
std::vector<double> reference_band_filter(const std::vector<double>& row,
                                          const std::vector<double>& taps) {
  const auto samples = static_cast<std::ptrdiff_t>(row.size());
  const auto half = static_cast<std::ptrdiff_t>(taps.size()) - 1;
  const std::ptrdiff_t last = samples - 1;
  const auto extended = [&](const std::ptrdiff_t index) {
    if (index < 0) {
      const double doubled = 2.0 * row.front();
      return doubled - row[static_cast<std::size_t>(-index)];
    }
    if (index > last) {
      const double doubled = 2.0 * row.back();
      return doubled - row[static_cast<std::size_t>(2 * last - index)];
    }
    return row[static_cast<std::size_t>(index)];
  };
  std::vector<double> filtered(row.size());
  for (std::ptrdiff_t index = 0; index < samples; ++index) {
    double total = taps[0] * row[static_cast<std::size_t>(index)];
    for (std::ptrdiff_t k = 1; k <= half; ++k) {
      const double pair = extended(index - k) + extended(index + k);
      const double term = taps[static_cast<std::size_t>(k)] * pair;
      total = total + term;
    }
    filtered[static_cast<std::size_t>(index)] = total;
  }
  return filtered;
}

// The rounding bound of each filtered sample, `gamma_{2K+3}` times the sum of its
// terms' magnitudes (a reflected sample `2 y_0 - y_m` bounded by `2 |y_0| + |y_m|`),
// whatever order or contraction a compiler gives the products and sums.
std::vector<double> filter_tolerance(const std::vector<double>& row,
                                     const std::vector<double>& taps) {
  const auto samples = static_cast<std::ptrdiff_t>(row.size());
  const auto half = static_cast<std::ptrdiff_t>(taps.size()) - 1;
  const std::ptrdiff_t last = samples - 1;
  const auto size = [&](const std::ptrdiff_t index) {
    if (index < 0)
      return 2.0 * std::abs(row.front()) +
             std::abs(row[static_cast<std::size_t>(-index)]);
    if (index > last)
      return 2.0 * std::abs(row.back()) +
             std::abs(row[static_cast<std::size_t>(2 * last - index)]);
    return std::abs(row[static_cast<std::size_t>(index)]);
  };
  const auto operations = static_cast<double>(2 * half + 3);
  const double gamma = operations * kEpsilon / (1.0 - operations * kEpsilon);
  std::vector<double> tolerance(row.size());
  for (std::ptrdiff_t index = 0; index < samples; ++index) {
    double total = std::abs(taps[0]) * size(index);
    for (std::ptrdiff_t k = 1; k <= half; ++k)
      total += std::abs(taps[static_cast<std::size_t>(k)]) *
               (size(index - k) + size(index + k));
    tolerance[static_cast<std::size_t>(index)] = gamma * total;
  }
  return tolerance;
}

// One band scan of `rows` rows of `samples` samples, staged or direct, on the host:
// one team per pair of rows, as the device column treats a node's two components, each
// row `team_band_filter` then `team_trapezoid_prefix` through one tile. `team_size` 0
// is the build's: the device team width, or Kokkos' choice.
template <bool Staged>
std::pair<std::vector<double>, std::vector<double>>
team_band_scan(const std::vector<double>& values, const std::size_t rows,
               const std::size_t samples, const std::vector<double>& taps,
               const double spacing, const int team_size) {
  using View = Kokkos::View<double**, native::Layout, native::MemorySpace>;
  const native::ExecutionSpace execution_space;
  View input("solprop_native.test.band_input", rows, samples);
  View filtered("solprop_native.test.band_filtered", rows, samples);
  View prefixes("solprop_native.test.band_prefixes", rows, samples);
  Kokkos::View<double*, native::Layout, native::MemorySpace> tap_view(
      "solprop_native.test.band_taps", taps.size());
  auto host_input = Kokkos::create_mirror_view(input);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t sample = 0; sample < samples; ++sample)
      host_input(row, sample) = values[row * samples + sample];
  auto host_taps = Kokkos::create_mirror_view(tap_view);
  for (std::size_t k = 0; k < taps.size(); ++k)
    host_taps(k) = taps[k];
  Kokkos::deep_copy(execution_space, input, host_input);
  Kokkos::deep_copy(execution_space, tap_view, host_taps);
  using TeamPolicy = Kokkos::TeamPolicy<native::ExecutionSpace>;
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
        0,
        Kokkos::PerTeam(native::band_staging_bytes<Staged, native::ExecutionSpace>()));
  const int half = static_cast<int>(taps.size()) - 1;
  const int count = static_cast<int>(samples);
  Kokkos::parallel_for(
      "solprop_native.test.band_team_scan", policy,
      KOKKOS_LAMBDA(const TeamPolicy::member_type& team) {
        const auto first_row = 2 * static_cast<std::size_t>(team.league_rank());
        double local[native::plasma_max_taps] = {};
        for (int k = 0; k <= half; ++k)
          local[k] = tap_view(static_cast<std::size_t>(k));
        const auto tile = native::band_tile<Staged>(team);
        for (auto row = first_row; row < first_row + 2 && row < rows; ++row) {
          native::team_band_filter<Staged>(team, tile, input, row, filtered, row, count,
                                           local, half);
          native::team_trapezoid_prefix(
              team, filtered, row, count, spacing,
              [&](const int sample, const double prefix) {
                prefixes(row, static_cast<std::size_t>(sample)) = prefix;
              });
        }
      });
  execution_space.fence("solprop_native band team-scan test");
  const auto host_filtered =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, filtered);
  const auto host_prefixes =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, prefixes);
  std::vector<double> filtered_values(rows * samples);
  std::vector<double> prefix_values(rows * samples);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t sample = 0; sample < samples; ++sample) {
      filtered_values[row * samples + sample] = host_filtered(row, sample);
      prefix_values[row * samples + sample] = host_prefixes(row, sample);
    }
  return {filtered_values, prefix_values};
}

TEST(PlasmaBandScan, BandTeamScanMatchesTheSerialFilterAndTrapezoid) {
  // A K = 8 rule of unit sum with alternating taps, as the certified designs have.
  std::vector<double> taps(native::plasma_max_taps);
  double wings = 0.0;
  for (std::size_t k = 1; k < taps.size(); ++k) {
    taps[k] = std::pow(-0.3, static_cast<double>(k)) / static_cast<double>(k + 1);
    wings += taps[k];
  }
  taps[0] = 1.0 - 2.0 * wings;
  constexpr double spacing = 0.125;
  // The build's team size and narrower teams: a device's 64 lanes, or on the host
  // (whose own choice is one thread) teams of two and four threads where it has them.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const std::vector<int> team_sizes = {0, 64};
#else
  using TeamPolicy = Kokkos::TeamPolicy<native::ExecutionSpace>;
  const TeamPolicy probe(native::ExecutionSpace{}, 1, 1);
  const int widest = probe.team_size_max(EmptyTeamBody{}, Kokkos::ParallelForTag());
  std::vector<int> team_sizes = {0};
  for (const int size : {2, 4})
    if (size <= widest)
      team_sizes.push_back(size);
#endif
  // A short row whose reflections reach across it, one under a tile, and rows across
  // several staging tiles with a remainder.
  for (const std::size_t samples :
       {std::size_t{9}, std::size_t{40}, std::size_t{1000}, std::size_t{2600}}) {
    for (const std::size_t half : {std::size_t{0}, std::size_t{3}, std::size_t{8}}) {
      if (samples <= half)
        continue;
      // K = 0 is the trapezoid's one identity tap.
      const std::vector<double> local =
          half == 0
              ? std::vector<double>{1.0}
              : std::vector<double>(
                    taps.begin(), taps.begin() + static_cast<std::ptrdiff_t>(half) + 1);
      constexpr std::size_t rows = 3;
      // A rate-like row: exactly zero below a threshold, a smooth pulse above it, and
      // a pre-ionised row that is live at both ends.
      std::vector<double> values(rows * samples);
      for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t sample = 0; sample < samples; ++sample) {
          const double x =
              (static_cast<double>(sample) + 0.5) / static_cast<double>(samples) - 0.5;
          const double pulse =
              std::exp(-40.0 * x * x) * (1.0 + 0.25 * std::cos(90.0 * x));
          values[row * samples + sample] =
              row == 0 ? (pulse > 0.2 ? pulse : 0.0)
                       : (row == 1 ? pulse : 0.5 + pulse * static_cast<double>(row));
        }
      for (const int team_size : team_sizes) {
        const auto direct =
            team_band_scan<false>(values, rows, samples, local, spacing, team_size);
        const auto staged =
            team_band_scan<true>(values, rows, samples, local, spacing, team_size);
        for (std::size_t row = 0; row < rows; ++row) {
          const std::vector<double> input(
              values.begin() + static_cast<std::ptrdiff_t>(row * samples),
              values.begin() + static_cast<std::ptrdiff_t>((row + 1) * samples));
          const auto expected = reference_band_filter(input, local);
          // Each filtered sample is held to the rounding bound of its K + 1 products
          // and 2K + 1 sums, whatever order or contraction a compiler gives them
          // (solprop SEP 25); the staged path, which only reads its halo from
          // elsewhere, to twice that against the direct one.
          const auto tolerance = filter_tolerance(input, local);
          for (std::size_t sample = 0; sample < samples; ++sample) {
            const auto flat = row * samples + sample;
            ASSERT_NEAR(direct.first[flat], expected[sample], tolerance[sample])
                << samples << " " << half << " " << team_size << " " << row << " "
                << sample;
            ASSERT_NEAR(staged.first[flat], direct.first[flat], 2.0 * tolerance[sample])
                << samples << " " << half << " " << team_size << " " << row << " "
                << sample;
          }
          // The scan's tree associates differently from the serial sum. Both stay
          // within gamma_n of the sum of the increments' magnitudes, so they differ
          // by at most twice that (Higham, the recursive and pairwise summation
          // bounds), and the filtered samples' own bound carries into the prefix.
          double serial = 0.0;
          double magnitude = 0.0;
          double carried = 0.0;
          const double gamma = static_cast<double>(samples) * kEpsilon /
                               (1.0 - static_cast<double>(samples) * kEpsilon);
          for (std::size_t sample = 0; sample < samples; ++sample) {
            if (sample != 0) {
              const double increment =
                  spacing * (expected[sample] + expected[sample - 1]) / 2.0;
              serial += increment;
              magnitude += std::abs(increment);
              carried += spacing * (tolerance[sample] + tolerance[sample - 1]) / 2.0;
            }
            const auto flat = row * samples + sample;
            const double bound = 2.0 * gamma * magnitude + carried;
            EXPECT_NEAR(direct.second[flat], serial, bound)
                << samples << " " << half << " " << team_size << " " << row << " "
                << sample;
            // Staged against direct: each within `bound` of the serial sum.
            EXPECT_NEAR(staged.second[flat], direct.second[flat], 2.0 * bound)
                << samples << " " << half << " " << team_size << " " << row << " "
                << sample;
          }
        }
      }
    }
  }
}

// One tiled band scan (`portable_tiled_band_scan`) of `rows` rows of `samples`
// samples in tiles of `tile`, on teams of `team_size` threads.
std::vector<double> tiled_band_scan(const std::vector<double>& values,
                                    const std::size_t rows, const std::size_t samples,
                                    const std::vector<double>& taps,
                                    const double spacing, const std::size_t tile,
                                    const int team_size) {
  using View = Kokkos::View<double**, native::Layout, native::MemorySpace>;
  const native::ExecutionSpace execution_space;
  const auto tiles = native::band_tile_count(samples, tile);
  View input("solprop_native.test.tiled_input", rows, samples);
  View locals("solprop_native.test.tiled_locals", rows, samples);
  View totals("solprop_native.test.tiled_totals", rows, tiles);
  View prefixes("solprop_native.test.tiled_prefixes", rows, samples);
  Kokkos::View<double*, native::Layout, native::MemorySpace> tap_view(
      "solprop_native.test.tiled_taps", taps.size());
  auto host_input = Kokkos::create_mirror_view(input);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t sample = 0; sample < samples; ++sample)
      host_input(row, sample) = values[row * samples + sample];
  auto host_taps = Kokkos::create_mirror_view(tap_view);
  for (std::size_t k = 0; k < taps.size(); ++k)
    host_taps(k) = taps[k];
  Kokkos::deep_copy(execution_space, input, host_input);
  Kokkos::deep_copy(execution_space, tap_view, host_taps);
  // A stale local row would show as NaN.
  Kokkos::deep_copy(execution_space, locals, std::numeric_limits<double>::quiet_NaN());
  native::portable_tiled_band_scan(execution_space, input, locals, totals, prefixes,
                                   tap_view, spacing, tile, team_size);
  execution_space.fence("solprop_native tiled band-scan test");
  const auto host_prefixes =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, prefixes);
  std::vector<double> result(rows * samples);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t sample = 0; sample < samples; ++sample)
      result[row * samples + sample] = host_prefixes(row, sample);
  return result;
}

// The tiled scan's team sizes: the build's and narrower ones, as for the team scan.
std::vector<int> tiled_team_sizes() {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  return {0, 64};
#else
  using TeamPolicy = Kokkos::TeamPolicy<native::ExecutionSpace>;
  const TeamPolicy probe(native::ExecutionSpace{}, 1, 1);
  const int widest = probe.team_size_max(EmptyTeamBody{}, Kokkos::ParallelForTag());
  std::vector<int> sizes = {0};
  for (const int size : {2, 4})
    if (size <= widest)
      sizes.push_back(size);
  return sizes;
#endif
}

TEST(PlasmaBandScan, TiledTrapezoidScanIsExactOnIntegers) {
  // The device trapezoid plasma's two-level scan: integer samples and a spacing of
  // two make every increment and partial sum an exact integer, so any association
  // gives the serial result exactly, over tiles and team sizes.
  using View = Kokkos::View<double**, native::Layout, native::MemorySpace>;
  const native::ExecutionSpace execution_space;
  constexpr double spacing = 2.0;
  for (const std::size_t samples :
       {std::size_t{1}, std::size_t{9}, std::size_t{2600}}) {
    constexpr std::size_t rows = 3;
    View input("solprop_native.test.trapezoid_input", rows, samples);
    View prefixes("solprop_native.test.trapezoid_prefixes", rows, samples);
    auto host_input = Kokkos::create_mirror_view(input);
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t sample = 0; sample < samples; ++sample)
        host_input(row, sample) =
            static_cast<double>((sample * 7 + row * 13) % 23) - 11.0;
    Kokkos::deep_copy(execution_space, input, host_input);
    for (const std::size_t tile :
         {std::size_t{1}, std::size_t{5}, std::size_t{64}, std::size_t{1024}}) {
      View totals("solprop_native.test.trapezoid_totals", rows,
                  (samples + tile - 1) / tile);
      for (const int team_size : tiled_team_sizes()) {
        Kokkos::deep_copy(execution_space, prefixes,
                          std::numeric_limits<double>::quiet_NaN());
        native::portable_tiled_trapezoid_scan(execution_space, input, totals, prefixes,
                                              spacing, tile, team_size);
        execution_space.fence("solprop_native tiled trapezoid test");
        const auto host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, prefixes);
        for (std::size_t row = 0; row < rows; ++row) {
          double serial = 0.0;
          for (std::size_t sample = 0; sample < samples; ++sample) {
            if (sample != 0)
              serial += spacing *
                        (host_input(row, sample) + host_input(row, sample - 1)) / 2.0;
            ASSERT_EQ(host(row, sample), serial)
                << samples << " " << tile << " " << team_size << " " << row << " "
                << sample;
          }
        }
      }
    }
  }
}

TEST(PlasmaBandScan, BandTileSizesFollowTheRowLength) {
  EXPECT_EQ(native::band_tile_samples(64), native::band_tile_min);
  EXPECT_EQ(native::band_tile_samples(8192), std::size_t{256});
  EXPECT_EQ(native::band_tile_samples(16384), std::size_t{512});
  EXPECT_EQ(native::band_tile_samples(32768), native::band_tile_max);
  EXPECT_EQ(native::band_tile_samples(65536), native::band_tile_max);
  EXPECT_EQ(native::band_tile_samples(65536, 512), std::size_t{512});
  EXPECT_EQ(native::band_tile_count(65536, 1024), std::size_t{64});
  EXPECT_EQ(native::band_tile_count(1000, 256), std::size_t{4});
}

TEST(PlasmaBandScan, TiledBandScanIsExactOnIntegers) {
  // Integer taps, integer samples and a spacing of two: every reflection, filtered
  // sample, increment `2 (y_n + y_{n-1}) / 2` and partial sum is an integer far below
  // 2^53, so any association and contraction gives the serial result exactly.
  const std::vector<double> taps = {6.0, -2.0, 1.0, -1.0};
  constexpr double spacing = 2.0;
  for (const std::size_t samples :
       {std::size_t{9}, std::size_t{40}, std::size_t{1000}, std::size_t{2600}}) {
    for (const std::size_t half : {std::size_t{0}, std::size_t{3}}) {
      const std::vector<double> local = half == 0 ? std::vector<double>{1.0} : taps;
      constexpr std::size_t rows = 3;
      std::vector<double> values(rows * samples);
      for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t sample = 0; sample < samples; ++sample)
          values[row * samples + sample] =
              static_cast<double>((sample * 7 + row * 13) % 23) - 11.0;
      for (const std::size_t tile :
           {std::size_t{1}, std::size_t{5}, std::size_t{64}, std::size_t{256}}) {
        for (const int team_size : tiled_team_sizes()) {
          const auto scanned =
              tiled_band_scan(values, rows, samples, local, spacing, tile, team_size);
          for (std::size_t row = 0; row < rows; ++row) {
            const std::vector<double> input(
                values.begin() + static_cast<std::ptrdiff_t>(row * samples),
                values.begin() + static_cast<std::ptrdiff_t>((row + 1) * samples));
            const auto filtered = reference_band_filter(input, local);
            double serial = 0.0;
            for (std::size_t sample = 0; sample < samples; ++sample) {
              if (sample != 0)
                serial += spacing * (filtered[sample] + filtered[sample - 1]) / 2.0;
              ASSERT_EQ(scanned[row * samples + sample], serial)
                  << samples << " " << half << " " << tile << " " << team_size << " "
                  << row << " " << sample;
            }
          }
        }
      }
    }
  }
}

TEST(PlasmaBandScan, TiledBandScanMatchesTheSerialFilterAndTrapezoid) {
  std::vector<double> taps(native::plasma_max_taps);
  double wings = 0.0;
  for (std::size_t k = 1; k < taps.size(); ++k) {
    taps[k] = std::pow(-0.3, static_cast<double>(k)) / static_cast<double>(k + 1);
    wings += taps[k];
  }
  taps[0] = 1.0 - 2.0 * wings;
  constexpr double spacing = 0.125;
  for (const std::size_t samples :
       {std::size_t{9}, std::size_t{40}, std::size_t{1000}, std::size_t{2600}}) {
    for (const std::size_t half : {std::size_t{0}, std::size_t{3}, std::size_t{8}}) {
      if (samples <= half)
        continue;
      const std::vector<double> local =
          half == 0
              ? std::vector<double>{1.0}
              : std::vector<double>(
                    taps.begin(), taps.begin() + static_cast<std::ptrdiff_t>(half) + 1);
      constexpr std::size_t rows = 3;
      std::vector<double> values(rows * samples);
      for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t sample = 0; sample < samples; ++sample) {
          const double x =
              (static_cast<double>(sample) + 0.5) / static_cast<double>(samples) - 0.5;
          const double pulse =
              std::exp(-40.0 * x * x) * (1.0 + 0.25 * std::cos(90.0 * x));
          values[row * samples + sample] =
              row == 0 ? (pulse > 0.2 ? pulse : 0.0)
                       : (row == 1 ? pulse : 0.5 + pulse * static_cast<double>(row));
        }
      for (const std::size_t tile :
           {std::size_t{7}, std::size_t{256}, native::band_tile_max}) {
        for (const int team_size : tiled_team_sizes()) {
          const auto scanned =
              tiled_band_scan(values, rows, samples, local, spacing, tile, team_size);
          for (std::size_t row = 0; row < rows; ++row) {
            const std::vector<double> input(
                values.begin() + static_cast<std::ptrdiff_t>(row * samples),
                values.begin() + static_cast<std::ptrdiff_t>((row + 1) * samples));
            const auto expected = reference_band_filter(input, local);
            const auto tolerance = filter_tolerance(input, local);
            // A prefix's tree (the tile's scan, the earlier totals' reduction and the
            // addition joining them) sums its increments in n - 1 additions, so it and
            // the serial sum are each within gamma_n of the increments' magnitudes and
            // differ by at most twice that, plus the filtered samples' own bound.
            double serial = 0.0;
            double magnitude = 0.0;
            double carried = 0.0;
            const double gamma = static_cast<double>(samples) * kEpsilon /
                                 (1.0 - static_cast<double>(samples) * kEpsilon);
            for (std::size_t sample = 0; sample < samples; ++sample) {
              if (sample != 0) {
                const double increment =
                    spacing * (expected[sample] + expected[sample - 1]) / 2.0;
                serial += increment;
                magnitude += std::abs(increment);
                carried += spacing * (tolerance[sample] + tolerance[sample - 1]) / 2.0;
              }
              ASSERT_NEAR(scanned[row * samples + sample], serial,
                          2.0 * gamma * magnitude + carried)
                  << samples << " " << half << " " << tile << " " << team_size << " "
                  << row << " " << sample;
            }
          }
        }
      }
    }
  }
}

// One row through the host column's two forms on a team of `team_size` threads: the
// whole-row band filter and scan, and the filter and scan limited to the row's live
// window (`band_support`, `team_trapezoid_window`), whose filtered row starts as NaN so
// a read outside the window would show.
struct WindowedScan {
  std::vector<double> filtered;
  std::vector<double> whole;
  std::vector<double> windowed;
  native::BandWindow window;
  double total;
};

WindowedScan windowed_band_scan(const std::vector<double>& row,
                                const std::vector<double>& taps, const double spacing,
                                const int team_size) {
  using View = Kokkos::View<double**, native::Layout, native::MemorySpace>;
  const native::ExecutionSpace execution_space;
  const auto samples = row.size();
  const int count = static_cast<int>(samples);
  const int half = static_cast<int>(taps.size()) - 1;
  int live_first = count;
  int live_last = -1;
  for (int i = 0; i < count; ++i)
    if (row[static_cast<std::size_t>(i)] != 0.0) {
      live_first = std::min(live_first, i);
      live_last = i;
    }
  const auto window = native::band_support(live_first, live_last, count, half);
  View input("solprop_native.test.window_input", 1, samples);
  View filtered("solprop_native.test.window_filtered", 1, samples);
  View partial("solprop_native.test.window_partial", 1, samples);
  View whole("solprop_native.test.window_whole", 1, samples);
  View windowed("solprop_native.test.window_windowed", 1, samples);
  Kokkos::View<double*, native::Layout, native::MemorySpace> totals(
      "solprop_native.test.window_total", 1);
  auto host_input = Kokkos::create_mirror_view(input);
  for (std::size_t i = 0; i < samples; ++i)
    host_input(0, i) = row[i];
  Kokkos::deep_copy(execution_space, input, host_input);
  Kokkos::deep_copy(execution_space, partial, std::numeric_limits<double>::quiet_NaN());
  Kokkos::View<double*, native::Layout, native::MemorySpace> tap_view(
      "solprop_native.test.window_taps", taps.size());
  auto host_taps = Kokkos::create_mirror_view(tap_view);
  for (std::size_t k = 0; k < taps.size(); ++k)
    host_taps(k) = taps[k];
  Kokkos::deep_copy(execution_space, tap_view, host_taps);
  using TeamPolicy = Kokkos::TeamPolicy<native::ExecutionSpace>;
  Kokkos::parallel_for(
      "solprop_native.test.windowed_band_scan",
      TeamPolicy(execution_space, 1, team_size),
      KOKKOS_LAMBDA(const TeamPolicy::member_type& team) {
        double local[native::plasma_max_taps] = {};
        for (int k = 0; k <= half; ++k)
          local[k] = tap_view(static_cast<std::size_t>(k));
        const auto tile = native::band_tile<false>(team);
        native::team_band_filter<false>(team, tile, input, 0, filtered, 0, count, local,
                                        half);
        native::team_trapezoid_prefix(team, filtered, 0, count, spacing,
                                      [&](const int sample, const double prefix) {
                                        whole(0, static_cast<std::size_t>(sample)) =
                                            prefix;
                                      });
        if (window.first <= window.last)
          native::team_band_filter<false>(team, tile, input, 0, partial, 0, count,
                                          local, half, window.first, window.last + 1);
        const double total = native::team_trapezoid_window(
            team, partial, 0, window, count, spacing,
            [&](const int sample, const double prefix) {
              windowed(0, static_cast<std::size_t>(sample)) = prefix;
            });
        Kokkos::single(Kokkos::PerTeam(team), [&] { totals(0) = total; });
      });
  execution_space.fence("solprop_native windowed band-scan test");
  const auto host_filtered =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, filtered);
  const auto host_whole =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, whole);
  const auto host_windowed =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, windowed);
  const auto host_totals =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, totals);
  WindowedScan result{.filtered = std::vector<double>(samples),
                      .whole = std::vector<double>(samples),
                      .windowed = std::vector<double>(samples),
                      .window = window,
                      .total = host_totals(0)};
  for (std::size_t i = 0; i < samples; ++i) {
    result.filtered[i] = host_filtered(0, i);
    result.whole[i] = host_whole(0, i);
    result.windowed[i] = host_windowed(0, i);
  }
  return result;
}

TEST(PlasmaBandScan, BandWindowSkipsOnlyExactZeros) {
  std::vector<double> taps(native::plasma_max_taps);
  double wings = 0.0;
  for (std::size_t k = 1; k < taps.size(); ++k) {
    taps[k] = std::pow(-0.3, static_cast<double>(k)) / static_cast<double>(k + 1);
    wings += taps[k];
  }
  taps[0] = 1.0 - 2.0 * wings;
  constexpr double spacing = 0.125;
  constexpr std::size_t samples = 64;
  const std::size_t half = taps.size() - 1;
  using TeamPolicy = Kokkos::TeamPolicy<native::ExecutionSpace>;
  const TeamPolicy probe(native::ExecutionSpace{}, 1, 1);
  const int widest = probe.team_size_max(EmptyTeamBody{}, Kokkos::ParallelForTag());
  // A zero row, and spikes at the first sample, within K of the start, within K of the
  // end and in the interior: the window reaches an end or stops K samples short of it.
  std::vector<std::vector<double>> rows(5, std::vector<double>(samples, 0.0));
  const std::array<std::size_t, 4> spikes = {0, half - 1, samples - half, samples / 2};
  for (std::size_t i = 0; i < spikes.size(); ++i)
    rows[i + 1][spikes[i]] = 1.0 + 0.25 * static_cast<double>(i);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    for (const int team_size : {1, 2, 4}) {
      if (team_size > widest)
        continue;
      const auto scan = windowed_band_scan(rows[r], taps, spacing, team_size);
      const int first = scan.window.first;
      const int last = scan.window.last;
      for (std::size_t i = 0; i < samples; ++i) {
        const auto index = static_cast<int>(i);
        // Exact, structural: outside the window every sample the filter reads is zero,
        // so the whole-row filter gives exactly zero there; the window's prefix is
        // exactly zero before the window and exactly its total after it.
        if (index < first || index > last) {
          ASSERT_EQ(scan.filtered[i], 0.0) << r << " " << team_size << " " << i;
        }
        if (index < first) {
          ASSERT_EQ(scan.windowed[i], 0.0) << r << " " << team_size << " " << i;
        }
        if (index > last) {
          ASSERT_EQ(scan.windowed[i], scan.total) << r << " " << team_size << " " << i;
        }
        // Exact for one thread: Kokkos's host team scan then adds the increments in
        // sample order in both forms, each filtered sample is the same function of
        // the same inputs, and the increments the window skips are exact zeros that
        // the whole-row form adds to an exact zero or to its total. With more threads
        // the chunks start at different samples and the forms associate differently,
        // so they agree to the scan's rounding bound alone (tests/test_band_stress.py
        // holds the column to the reference there).
        if (team_size == 1) {
          ASSERT_EQ(scan.windowed[i], scan.whole[i]) << r << " " << i;
        }
      }
      if (r == 0) {
        EXPECT_GT(first, last);
        EXPECT_EQ(scan.total, 0.0);
      } else {
        EXPECT_NE(scan.total, 0.0);
      }
    }
  }
}

TEST(PlasmaBandScan, FilteringPreservesShortRowsBlockTailsAndDynamicRange) {
  const std::vector<double> taps{1.1, -0.1, 0.0625, -0.025, 0.0125};
  const int half = static_cast<int>(taps.size()) - 1;
  using TeamPolicy = Kokkos::TeamPolicy<native::ExecutionSpace>;
  const TeamPolicy probe(native::ExecutionSpace{}, 1, 1);
  const int widest = probe.team_size_max(EmptyTeamBody{}, Kokkos::ParallelForTag());
  for (const int samples : {5, 17, 255, 257, 519}) {
    SCOPED_TRACE(samples);
    std::vector<double> row(static_cast<std::size_t>(samples));
    for (int sample = 0; sample < samples; ++sample)
      row[static_cast<std::size_t>(sample)] = sample % 3 == 0   ? 1e100
                                              : sample % 3 == 1 ? -1e100
                                                                : 1e-100;
    for (const int team_size : {1, 2, 4}) {
      if (team_size > widest)
        continue;
      const auto scan = windowed_band_scan(row, taps, 0.125, team_size);
      const auto reflected = [&](const int index) -> long double {
        if (index < 0)
          return 2.0L * row.front() - row[static_cast<std::size_t>(-index)];
        if (index >= samples)
          return 2.0L * row.back() -
                 row[static_cast<std::size_t>(2 * (samples - 1) - index)];
        return row[static_cast<std::size_t>(index)];
      };
      for (int sample = 0; sample < samples; ++sample) {
        long double expected = taps[0] * reflected(sample);
        long double magnitude = std::abs(expected);
        for (int tap = 1; tap <= half; ++tap) {
          const auto before = reflected(sample - tap);
          const auto after = reflected(sample + tap);
          expected += taps[static_cast<std::size_t>(tap)] * (before + after);
          magnitude += std::abs(taps[static_cast<std::size_t>(tap)]) *
                       (std::abs(before) + std::abs(after));
        }
        // Centre + ordered tap pairs + reflections, in both computations. This
        // also covers platforms whose long double has no extra mantissa bits.
        const double bound = (8.0 * half + 16.0) * kEpsilon *
                             std::max(1.0, static_cast<double>(magnitude));
        ASSERT_GT(bound, 0.0);
        const auto actual = scan.filtered[static_cast<std::size_t>(sample)];
        ASSERT_TRUE(std::isfinite(actual));
        EXPECT_NEAR(actual, static_cast<double>(expected), bound);
      }
    }
  }
}

} // namespace
