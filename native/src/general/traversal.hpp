// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The traversals between a separable geometry's registers `(column, omega)` and the
// transverse scratch `(omega, y, x)` (SEP 28): cartesian_for; on the host the tiles of
// crossing_tiles and crossing_for, with the stores that bypass the caches
// (`SOLPROP_NATIVE_STREAMING_STORES`, set in kernels.hpp's include block) and
// PropagatorTile, a tile's propagators in one vector-maths pass; on a device the
// team-scratch tiles of tiled_crossing and tiled_update. SpatialElement, the value
// SpatialCoefficients forms, sits among them. A part of kernels.hpp, included through
// it.
#ifdef SOLPROP_NATIVE_STRICT_TU
#error "a strict translation unit must not include general/traversal.hpp"
#endif

#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/launch.hpp"
#include "solprop_native/runtime/vmath.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace solprop_native::general {

// CPU experiments favour one task per column; accelerator lanes retain their
// two-dimensional launch. Both invoke the same element functor in the same order.
template <class Function>
void cartesian_for(const char* label, const ExecutionSpace& execution,
                   const std::size_t rows, const std::size_t width,
                   const Function& element) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  launch::parallel_for(label, Range2(execution, {0, 0}, {rows, width}), element);
#else
  launch::parallel_for(
      label, Kokkos::RangePolicy<ExecutionSpace>(execution, 0, rows),
      KOKKOS_LAMBDA(const std::size_t row) {
        for (std::size_t inner = 0; inner < width; ++inner)
          // NOLINTNEXTLINE(readability-suspicious-call-argument): the names only coincide
          element(row, inner);
      });
#endif
}

// A traversal between a separable geometry's column-major registers `(column, omega)`
// and the transverse scratch `(omega, y, x)` (SEP 28). On CPU one task per column
// reads the scratch a whole plane apart at every bin; tiles of adjacent columns and
// bins, columns fastest within a tile, read both layouts in contiguous runs. The
// element is the same functor; each (column, bin) is visited once.
// 32 columns by 16 bins: the best of 32x16, 64x32 and 256x8 on one Zen 5 socket
// (SEP 28; Apple silicon is within a few per cent across them).
[[maybe_unused]] inline constexpr std::size_t crossing_tile_columns = 32;
[[maybe_unused]] inline constexpr std::size_t crossing_tile_bins = 16;

// A store that bypasses the caches where the CPU has one (x86 SSE2): a traversal that
// writes a large volume it does not read (the scatter into the transverse scratch)
// otherwise reads every line before writing it. Ordered by `stream_fence`. Device
// builds compile out the host traversals that use these, hence `maybe_unused`.
[[maybe_unused]] KOKKOS_INLINE_FUNCTION void stream_store(Complex& target,
                                                          const Complex value) {
#ifdef SOLPROP_NATIVE_STREAMING_STORES
  _mm_stream_pd(reinterpret_cast<double*>(&target),
                _mm_set_pd(value.imag(), value.real()));
#else
  target = value;
#endif
}

[[maybe_unused]] KOKKOS_INLINE_FUNCTION void stream_fence() {
#ifdef SOLPROP_NATIVE_STREAMING_STORES
  _mm_sfence();
#endif
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
// One host tile of a crossing: its rows `[first_row, last_row)` and bins
// `[first_bin, last_bin)`, at most `crossing_tile_columns` by `crossing_tile_bins`.
struct CrossingTile {
  std::size_t first_row, last_row, first_bin, last_bin;

  [[nodiscard]] std::size_t size() const noexcept {
    return (last_row - first_row) * (last_bin - first_bin);
  }
  // An element's place in a tile buffer, rows outermost.
  [[nodiscard]] std::size_t at(const std::size_t row,
                               const std::size_t bin) const noexcept {
    return (row - first_row) * (last_bin - first_bin) + (bin - first_bin);
  }
  // Every element in the traversal order `crossing_for` names.
  template <bool BinsInner, class Function> void visit(const Function& element) const {
    if constexpr (BinsInner)
      for (std::size_t row = first_row; row < last_row; ++row)
        for (std::size_t inner = first_bin; inner < last_bin; ++inner)
          element(row, inner);
    else
      for (std::size_t inner = first_bin; inner < last_bin; ++inner)
        for (std::size_t row = first_row; row < last_row; ++row)
          element(row, inner);
  }
};

// The host tiles of a crossing, one task each: `crossing_for`'s traversal, for a
// kernel that forms a whole tile's transcendentals in one pass first.
template <bool Streaming = false, class Function>
void crossing_tiles(const char* label, const ExecutionSpace& execution,
                    const std::size_t rows, const std::size_t width,
                    const Function& visit) {
  constexpr std::size_t tile_rows = crossing_tile_columns;
  constexpr std::size_t tile_bins = crossing_tile_bins;
  const std::size_t row_tiles = (rows + tile_rows - 1) / tile_rows;
  const std::size_t bin_tiles = (width + tile_bins - 1) / tile_bins;
  launch::parallel_for(
      label, Kokkos::RangePolicy<ExecutionSpace>(execution, 0, row_tiles * bin_tiles),
      KOKKOS_LAMBDA(const std::size_t index) {
        const std::size_t first_row = (index / bin_tiles) * tile_rows;
        const std::size_t first_bin = (index % bin_tiles) * tile_bins;
        visit(CrossingTile{
            .first_row = first_row,
            .last_row = rows - first_row < tile_rows ? rows : first_row + tile_rows,
            .first_bin = first_bin,
            .last_bin = width - first_bin < tile_bins ? width : first_bin + tile_bins});
        // Each thread orders its own streaming stores before the region ends.
        if constexpr (Streaming)
          stream_fence();
      });
}

// A tile's propagators `exp(argument)`, as `propagator_exp` forms them, in one pass
// of the vector-maths layer (solprop ADR-059, issue #338): the cosines and sines,
// then the magnitudes, `exp(+-0)` exactly one on a lossless element without the
// call. Fill `real` and `imag` for the tile's elements, then `form`.
struct PropagatorTile {
  static constexpr std::size_t capacity = crossing_tile_columns * crossing_tile_bins;
  // The static analyser cannot follow that `form` and `operator[]` read only what the
  // tile's visit wrote, so its build zeroes the arrays; a real build leaves them unset.
#ifdef __clang_analyzer__
  double real[capacity]{}, imag[capacity]{}, cosine[capacity]{}, sine[capacity]{};
#else
  double real[capacity], imag[capacity], cosine[capacity], sine[capacity];
#endif

  void form(const std::size_t count) {
    vmath::sincos(imag, sine, cosine, count);
    bool lossless = true;
    for (std::size_t k = 0; k < count; ++k)
      lossless = lossless && real[k] == 0.0;
    if (lossless)
      return;
#ifdef __clang_analyzer__
    double magnitude[capacity]{};
#else
    double magnitude[capacity];
#endif
    vmath::exp(real, magnitude, count);
    for (std::size_t k = 0; k < count; ++k) {
      const double scale = real[k] == 0.0 ? 1.0 : magnitude[k];
      cosine[k] *= scale;
      sine[k] *= scale;
    }
  }
  Complex operator[](const std::size_t k) const { return {cosine[k], sine[k]}; }
};
#endif

// On a device, `RowFastest` maps adjacent rows (columns of the plane) to adjacent
// threads, so the transverse scratch side of a gather or scatter is coalesced; the
// chunk-local side is the strided one.
// On the host, `BinsInner` visits a tile's rows outermost and its bins innermost: a
// traversal that writes the rows (a pack from the scratch) then stores each row's run
// at once, about 1.7 times faster for the free-space chunk's pack on an Intel Xeon Gold
// 6240 at one thread.
template <bool Streaming = false, bool RowFastest = false, bool BinsInner = false,
          class Function>
void crossing_for(const char* label, const ExecutionSpace& execution,
                  const std::size_t rows, const std::size_t width,
                  const Function& element) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  if constexpr (RowFastest) {
    using Left = Kokkos::MDRangePolicy<
        ExecutionSpace, Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
        Kokkos::IndexType<std::int64_t>>;
    launch::parallel_for(
        label,
        Left(execution, {0, 0},
             {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(width)}),
        KOKKOS_LAMBDA(const std::int64_t row, const std::int64_t inner) {
          element(static_cast<std::size_t>(row), static_cast<std::size_t>(inner));
        });
  } else {
    cartesian_for(label, execution, rows, width, element);
  }
#else
  crossing_tiles<Streaming>(
      label, execution, rows, width,
      [&](const CrossingTile& tile) { tile.visit<BinsInner>(element); });
#endif
}

struct SpatialElement {
  Complex linear, prefactor;
};

// A device crossing through a 32-by-32 tile of team scratch (SEP 28): the tile is read
// with adjacent threads along the side that is contiguous for `read` (rows when
// `read_rows`, else bins) and written with adjacent threads along the other, so both
// the transverse scratch and the chunk's rows are accessed coalesced.
template <class Read, class Write>
void tiled_crossing(const char* label, const ExecutionSpace& execution,
                    const std::size_t rows, const std::size_t width,
                    const bool read_rows, const Read& read, const Write& write) {
  constexpr int tile = 32;
  const int depth = std::min(8, execution.concurrency());
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  using Member = Policy::member_type;
  using Shared =
      Kokkos::View<Complex**, Kokkos::LayoutRight, ExecutionSpace::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const std::size_t row_tiles = (rows + tile - 1) / tile;
  const std::size_t bin_tiles = (width + tile - 1) / tile;
  const auto bytes = Shared::shmem_size(tile, tile + 1);
  launch::parallel_for(
      label,
      Policy(execution, static_cast<int>(row_tiles * bin_tiles), depth, tile)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const Member& team) {
        // The buffer holds (bin, row) within the tile.
        Shared buffer(team.team_scratch(0), tile, tile + 1);
        const auto which = static_cast<std::size_t>(team.league_rank());
        const std::size_t first_row = (which / bin_tiles) * tile;
        const std::size_t first_bin = (which % bin_tiles) * tile;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, tile), [&](const int outer) {
          Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team, tile), [&](const int lane) {
                const int row_local = read_rows ? lane : outer;
                const int bin_local = read_rows ? outer : lane;
                const std::size_t row = first_row + static_cast<std::size_t>(row_local);
                const std::size_t bin = first_bin + static_cast<std::size_t>(bin_local);
                if (row < rows && bin < width)
                  buffer(bin_local, row_local) = read(row, bin);
              });
        });
        team.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, tile), [&](const int outer) {
          Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team, tile), [&](const int lane) {
                const int row_local = read_rows ? outer : lane;
                const int bin_local = read_rows ? lane : outer;
                const std::size_t row = first_row + static_cast<std::size_t>(row_local);
                const std::size_t bin = first_bin + static_cast<std::size_t>(bin_local);
                if (row < rows && bin < width)
                  write(row, bin, buffer(bin_local, row_local));
              });
        });
      });
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
// A device update of registers and the transverse scratch together (SEP 28): a tile of
// the scratch is staged in team scratch with adjacent threads along rows, `body`
// visits each (row, bin) with adjacent threads along bins (the registers' contiguous
// side) and may change its cell, and the tile goes back to the scratch along rows
// when `writes`.
template <class Read, class Body, class Write>
void tiled_update(const char* label, const ExecutionSpace& execution,
                  const std::size_t rows, const std::size_t width, const bool writes,
                  const Read& read, const Body& body, const Write& write) {
  constexpr int tile = 32;
  constexpr int depth = 8;
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  using Member = Policy::member_type;
  using Shared = Kokkos::View<Complex**, Kokkos::LayoutRight,
                              typename ExecutionSpace::scratch_memory_space,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const std::size_t row_tiles = (rows + tile - 1) / tile;
  const std::size_t bin_tiles = (width + tile - 1) / tile;
  const auto bytes = Shared::shmem_size(tile, tile + 1);
  launch::parallel_for(
      label,
      Policy(execution, static_cast<int>(row_tiles * bin_tiles), depth, tile)
          .set_scratch_size(0, Kokkos::PerTeam(bytes)),
      KOKKOS_LAMBDA(const Member& team) {
        Shared buffer(team.team_scratch(0), tile, tile + 1);
        const auto which = static_cast<std::size_t>(team.league_rank());
        const std::size_t first_row = (which / bin_tiles) * tile;
        const std::size_t first_bin = (which % bin_tiles) * tile;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, tile), [&](const int outer) {
          Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team, tile), [&](const int lane) {
                const std::size_t row = first_row + static_cast<std::size_t>(lane);
                const std::size_t bin = first_bin + static_cast<std::size_t>(outer);
                if (row < rows && bin < width)
                  buffer(outer, lane) = read(row, bin);
              });
        });
        team.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, tile), [&](const int outer) {
          Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team, tile), [&](const int lane) {
                const std::size_t row = first_row + static_cast<std::size_t>(outer);
                const std::size_t bin = first_bin + static_cast<std::size_t>(lane);
                if (row < rows && bin < width)
                  body(row, bin, buffer(lane, outer));
              });
        });
        if (!writes)
          return;
        team.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, tile), [&](const int outer) {
          Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team, tile), [&](const int lane) {
                const std::size_t row = first_row + static_cast<std::size_t>(lane);
                const std::size_t bin = first_bin + static_cast<std::size_t>(outer);
                if (row < rows && bin < width)
                  write(row, bin, buffer(outer, lane));
              });
        });
      });
}
#endif

} // namespace solprop_native::general
