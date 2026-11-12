// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The general workspace's helpers: device-callable functions, kernel functors and the
// workspace's view and plan types, shared by its translation units (there is no
// relocatable device code, so a helper a kernel calls is defined here, inline).
// Contracting code: a strict translation unit never includes it.
#ifdef SOLPROP_NATIVE_STRICT_TU
#error "general/kernels.hpp contracts: a strict translation unit must not include it"
#endif

#include "solprop_native/adams_phi.hpp"
#include "solprop_native/build.hpp"
#include "solprop_native/core.hpp"
#include "solprop_native/coupled_flow.hpp"
#include "solprop_native/crossing.hpp"
#include "solprop_native/dispatch.hpp"
#include "solprop_native/dispersive.hpp"
#include "solprop_native/envelope_plasma.hpp"
#include "solprop_native/far_field.hpp"
#include "solprop_native/fft_plan.hpp"
#include "solprop_native/flow_functions.hpp"
#include "solprop_native/fp.hpp"
#include "solprop_native/gemm.hpp"
#include "solprop_native/general_modal.hpp"
#include "solprop_native/plasma.hpp"
#include "solprop_native/raman_fft.hpp"
#include "solprop_native/refuse.hpp"
#include "solprop_native/slab.hpp"
#include "solprop_native/vmath.hpp"
#include "solprop_native/z_table.hpp"

#include <KokkosFFT.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
#include <fftw3.h>
#ifdef __SSE2__
#include <immintrin.h>
#define SOLPROP_NATIVE_STREAMING_STORES 1
#endif
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#ifdef __APPLE__
#include <malloc/malloc.h>
#elif defined(__GLIBC__)
#include <dlfcn.h>
#endif
namespace solprop_native {
namespace general {}
// The helpers are found as an unnamed namespace's would be, under their own name so
// that every unit that includes them sees one definition.
using namespace general;
} // namespace solprop_native

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

namespace solprop_native::general {

// Sets a coupled flow aside for a stage evaluation and puts it back however the scope
// is left: the Adams sample is the material source alone (solprop SEP 16 phase 5).
class FlowAside {
public:
  explicit FlowAside(std::unique_ptr<CoupledFlowAction>& slot)
      : slot_(slot), held_(std::exchange(slot, nullptr)) {}
  // Restores the slot; clang-tidy 18 misreads it as trivial when GCC 13's intrinsic
  // headers fail to parse.
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~FlowAside() { slot_ = std::move(held_); }
  FlowAside(const FlowAside&) = delete;
  FlowAside& operator=(const FlowAside&) = delete;
  FlowAside(FlowAside&&) = delete;
  FlowAside& operator=(FlowAside&&) = delete;

private:
  std::unique_ptr<CoupledFlowAction>& slot_;
  std::unique_ptr<CoupledFlowAction> held_;
};

// One frozen-residual weight row from the host into a View's row.
template <class Row>
void upload_weights(const ExecutionSpace& space, const Row& out,
                    const std::vector<double>& weights) {
  if (weights.size() < out.extent(0))
    throw std::invalid_argument("a coupled flow's weights are shorter than its terms");
  const Kokkos::View<const double*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host(weights.data(), out.extent(0));
  graph::deep_copy(space, out, host);
}

// A frozen-residual flow's generator at one position applied to a field, `sum_t w_t
// G_t u` on the block's bins and zero off them (the terms are zero off the scored
// bins). A zero weight is skipped, which leaves every bit as the full sum has it.
template <class Out>
void apply_generator(const ExecutionSpace& space, const std::size_t n,
                     const std::size_t modes,
                     const Kokkos::View<std::size_t*, Layout, MemorySpace>& slots,
                     const Kokkos::View<Complex****, Layout, MemorySpace>& terms,
                     const Kokkos::View<double*, Layout, MemorySpace>& weights,
                     const Kokkos::View<Complex**, Layout, MemorySpace>& input,
                     const Out& out) {
  const auto n_terms = weights.extent(0);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  launch::parallel_for(
      "solprop_native.adams.generator", Range2(space, {0, 0}, {n, modes}),
      KOKKOS_LAMBDA(const std::size_t index, const std::size_t i) {
        const auto one = slots(index);
        Complex value{};
        if (one != 0) {
          const auto bin = one - 1;
          for (std::size_t entry = 0; entry < n_terms; ++entry) {
            const double weight = weights(entry);
            if (weight == 0.0)
              continue;
            for (std::size_t j = 0; j < modes; ++j)
              value += weight * terms(entry, bin, i, j) * input(j, index);
          }
        }
        out(i, index) = value;
      });
#else
  launch::parallel_for(
      "solprop_native.adams.generator",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
      KOKKOS_LAMBDA(const std::size_t index) {
        const auto one = slots(index);
        for (std::size_t i = 0; i < modes; ++i) {
          Complex value{};
          if (one != 0) {
            const auto bin = one - 1;
            for (std::size_t entry = 0; entry < n_terms; ++entry) {
              const double weight = weights(entry);
              if (weight == 0.0)
                continue;
              for (std::size_t j = 0; j < modes; ++j)
                value += weight * terms(entry, bin, i, j) * input(j, index);
            }
          }
          out(i, index) = value;
        }
      });
#endif
}

template <class... Ts>
KOKKOS_INLINE_FUNCTION void capture_anchor(const Ts&... /*unused*/) {}

// Net process allocator growth includes opaque vendor FFT plan/workspace
// allocations. It is measured separately from Kokkos Views and is not a peak bound.
inline std::size_t heap_bytes() {
#ifdef __APPLE__
  malloc_statistics_t statistics{};
  malloc_zone_statistics(nullptr, &statistics);
  return statistics.size_in_use;
#elif defined(__GLIBC__)
  // mallinfo2 is glibc 2.33 or later; a wheel built against an older glibc (the
  // manylinux_2_28 baseline) finds it at run time when the system provides it. Its
  // layout is ten size_t counters, in glibc's declared order.
  struct Mallinfo2 {
    std::size_t arena, ordblks, smblks, hblks, hblkhd, usmblks, fsmblks, uordblks,
        fordblks, keepcost;
  };
  using Query = Mallinfo2 (*)();
  static const auto query = reinterpret_cast<Query>(dlsym(RTLD_DEFAULT, "mallinfo2"));
  if (query == nullptr)
    return 0;
  const auto statistics = query();
  return statistics.uordblks + statistics.hblkhd;
#else
  // Neither allocator interface exists (musl, Windows): the ledger row then measures
  // nothing, and construction must still succeed.
  return 0;
#endif
}
inline std::size_t device_free_bytes() {
#ifdef KOKKOS_ENABLE_CUDA
  std::size_t available{}, total{};
  if (cudaMemGetInfo(&available, &total) != cudaSuccess)
    throw std::runtime_error("cudaMemGetInfo failed during FFT planning");
  return available;
#elif defined(KOKKOS_ENABLE_HIP)
  std::size_t available{}, total{};
  if (hipMemGetInfo(&available, &total) != hipSuccess)
    throw std::runtime_error("hipMemGetInfo failed during FFT planning");
  return available;
#else
  return 0;
#endif
}

using ComplexView2 = Kokkos::View<Complex**, Layout, MemorySpace>;
using ComplexView3 = Kokkos::View<Complex***, Layout, MemorySpace>;
using RealView2 = Kokkos::View<double**, Layout, MemorySpace>;
using RealView3 = Kokkos::View<double***, Layout, MemorySpace>;

template <int Components, bool RealField>
KOKKOS_INLINE_FUNCTION void
write_raman_drive(const RealView2& drive, const std::size_t node,
                  const std::size_t time, const Complex x, const Complex y,
                  const std::size_t count) {
  const double xx = RealField ? x.real() * x.real()
                              : 0.5 * (x.real() * x.real() + x.imag() * x.imag());
  if constexpr (Components == 1) {
    drive(node, time) = xx;
  } else {
    const double yy = RealField ? y.real() * y.real()
                                : 0.5 * (y.real() * y.real() + y.imag() * y.imag());
    drive(count * node, time) = xx + yy;
    if (count == 3) {
      drive(count * node + 1, time) = xx - yy;
      drive(count * node + 2, time) = RealField
                                          ? 2.0 * x.real() * y.real()
                                          : x.real() * y.real() + x.imag() * y.imag();
    }
  }
}
using ByteView = Kokkos::View<unsigned char*, Layout, MemorySpace>;
using IndexView = Kokkos::View<std::size_t*, Layout, MemorySpace>;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
// A device splits a few long real rows over its multiprocessors (`RealRowPlan`).
using InverseRealPlan = RealRowPlan<ExecutionSpace, ComplexView2, RealView2>;
using ForwardRealPlan = RealRowPlan<ExecutionSpace, RealView2, ComplexView2>;
#else
using InverseRealPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView2, RealView2>;
using ForwardRealPlan = KokkosFFT::Plan<ExecutionSpace, RealView2, ComplexView2>;
#endif
using RamanForwardPlan = RamanFftPlan<ExecutionSpace, RealView2, ComplexView2>;
using RamanInversePlan = RamanFftPlan<ExecutionSpace, ComplexView2, RealView2>;
using ComplexPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView2, ComplexView2>;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
// A device's plans of the transverse scratch run one after another, so they share one
// work area (`transform_work_`) in place of holding one each (solprop #529). A
// Kerr-only free-space envelope takes the stage's time axis with the plane's in one
// rank-3 transform (`volume_kerr_`).
using TransversePlan = KokkosFFT::DynPlan<ExecutionSpace, ComplexView3, ComplexView3>;
using VolumePlan = TransversePlan;
#else
using TransversePlan = KokkosFFT::Plan<ExecutionSpace, ComplexView3, ComplexView3, 2>;
using VolumePlan = KokkosFFT::Plan<ExecutionSpace, ComplexView3, ComplexView3, 3>;
#endif
// The hybrid planar kind transforms y alone, the trailing axis of (bins, n_p, n_y).
using SlabPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView3, ComplexView3, 1>;
// A requested collar's padded transform, along the last axis of a batch (solprop SEP
// 44).
using StripPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView2, ComplexView2, 1>;

// Each current keeps its own chunk accumulator, including stacked embedded
// quadrature. Sharing the latter would mix currents between partial chunks.
struct EnvelopeStream {
  ComplexView2 node, projected, coarse, stacked, spectrum, coarse_spectrum;
  std::unique_ptr<ComplexPlan> forward, coarse_forward;
  [[nodiscard]] std::size_t bytes() const {
    return sizeof(Complex) *
           (node.size() + projected.size() + coarse.size() + stacked.size() +
            spectrum.size() + coarse_spectrum.size());
  }
};

// Real current products own their projections, including an embedded quadrature.
struct FieldCurrentStream {
  RealView2 node, projected, coarse, stacked;
  ComplexView2 spectrum, coarse_spectrum;
  std::unique_ptr<ForwardRealPlan> forward, coarse_forward;
  [[nodiscard]] std::size_t bytes() const {
    return sizeof(double) *
               (node.size() + projected.size() + coarse.size() + stacked.size()) +
           sizeof(Complex) * (spectrum.size() + coarse_spectrum.size());
  }
};

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

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
// Plan, then discard, a batched complex transform of transverse planes at the plan
// policy's effort, leaving its wisdom. It overwrites both arrays; the caller clears
// them. Row transforms are seeded by `seed_row_fft_wisdom` (fft_plan.hpp).
inline void seed_fftw_wisdom(const ExecutionSpace& space, Complex* input,
                             Complex* output, const std::vector<int>& extents,
                             const int howmany, const int input_distance,
                             const int output_distance, const int sign) {
  std::string sizes;
  for (const int extent : extents)
    sizes += (sizes.empty() ? "" : "x") + std::to_string(extent);
  const std::string key = fft_plan_key(
      sign == FFTW_FORWARD ? "planes-forward" : "planes-backward", sizes, howmany,
      input_distance, output_distance, input, output, space.concurrency());
  fftw_init_threads();
  const auto chosen = begin_seed(key, FftShape::planes, space.concurrency());
  if (!chosen)
    return;
  fftw_plan_with_nthreads(space.concurrency());
  auto* in = reinterpret_cast<fftw_complex*>(input);
  auto* out = reinterpret_cast<fftw_complex*>(output);
  auto* const seeded = fftw_plan_many_dft(
      static_cast<int>(extents.size()), extents.data(), howmany, in, nullptr, 1,
      input_distance, out, nullptr, 1, output_distance, sign, *chosen);
  // The policy has always tolerated a plane it cannot seed; a profile records it.
  if (seeded == nullptr && (*chosen & FFTW_WISDOM_ONLY) == 0U &&
      (detail::planning == nullptr ||
       detail::planning->mode == FftPlanning::Mode::legacy)) {
    release_seed(key, false);
    return;
  }
  end_seed(key, *chosen, seeded != nullptr);
  if (seeded != nullptr)
    fftw_destroy_plan(seeded);
}
#endif

// Where a separable geometry's column (or a temporal row, which is the same index)
// lives in the transverse scratch. Free space: (.., n_y, n_x), x fastest. The hybrid
// planar kind: (.., n_p, n_y) with the y transform trailing, the channel fastest in
// the column index p + n_p * iy.
struct TransverseLayout {
  std::size_t inner{};
  bool channel_fastest{};
  // Elements per transverse plane of the row-major scratch `(planes, ., .)`.
  std::size_t plane{};
  // A column's offset within a plane: the column itself on free space (y, x with x
  // fastest), `p * n_y + iy` on the planar kind.
  KOKKOS_INLINE_FUNCTION std::size_t offset(const std::size_t column) const {
    return channel_fastest ? (column % inner) * (plane / inner) + column / inner
                           : column;
  }
  template <class View>
  KOKKOS_INLINE_FUNCTION auto& cell(const View& scratch, const std::size_t plane_index,
                                    const std::size_t column) const {
    return scratch.data()[plane_index * plane + offset(column)];
  }
  KOKKOS_INLINE_FUNCTION std::size_t first(const std::size_t column) const {
    return channel_fastest ? column % inner : column / inner;
  }
  KOKKOS_INLINE_FUNCTION std::size_t second(const std::size_t column) const {
    return channel_fastest ? column / inner : column % inner;
  }
};

// A node's physical spectrum in a transverse volume after the inverse transverse plan:
// on free space the node's own cell; on the hybrid planar kind the slab rule over its y
// row's channels, without the temporal pad scale the planar synthesis carries, since
// the statistics read the coarse field (solprop's `collect_spatial`).
struct PhysicalSpectrum {
  TransverseLayout layout{};
  RealView2 synthesis;
  std::size_t n_x{}, n_p{};
  double crop_scale = 1.0;
  bool slab = false;
  template <class View>
  KOKKOS_INLINE_FUNCTION Complex read(const View& source, const std::size_t plane,
                                      const std::size_t node) const {
    if (!slab)
      return layout.cell(source, plane, node);
    const auto ix = node % n_x;
    const auto rows = (node / n_x) * n_p;
    Complex sum{};
    for (std::size_t channel = 0; channel < n_p; ++channel)
      sum += synthesis(ix, channel) *
             source(plane, layout.first(rows + channel), layout.second(rows + channel));
    return crop_scale * sum;
  }
};

// Where a separable route's low-storage registers (the accumulator, the stage register
// and the flow table) hold the element (column, omega). On the host, at the transverse
// scratch's own position: a fold then reads them in the scratch's order beside it, and
// only the state crosses layouts. On a device, at the state's (column-major).
struct RegisterIndex {
  TransverseLayout layout{};
  std::size_t n_pol{}, n{};
  bool plane{};
  // One polarisation: a column is its mode, without a division (a flag of its own,
  // which the compiler cannot fold back into `column / 1`).
  bool scalar{};
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // A stage of the volume route on a device (`volume_order_`): the scratch is `(y, x,
  // omega)`, the state's and the registers' own order, so nothing crosses.
  bool volume{};
#endif
  KOKKOS_INLINE_FUNCTION std::size_t mode(const std::size_t column) const {
    return scalar ? column : column / n_pol;
  }
  KOKKOS_INLINE_FUNCTION std::size_t pol(const std::size_t column) const {
    return scalar ? 0 : column % n_pol;
  }
  // The transverse scratch's position of the element, on any build.
  KOKKOS_INLINE_FUNCTION std::size_t cell(const std::size_t column,
                                          const std::size_t omega) const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if (volume)
      return column * n + omega;
#endif
    return (omega * n_pol + pol(column)) * layout.plane + layout.offset(mode(column));
  }
  KOKKOS_INLINE_FUNCTION std::size_t operator()(const std::size_t column,
                                                const std::size_t omega) const {
    return plane ? cell(column, omega) : column * n + omega;
  }
};

// One chunk of a separable geometry's response pass: whole transverse units (a node
// on free space, a y row of n_p columns and n_x nodes on the hybrid planar kind), its
// temporal rows (state columns) and its physical nodes.
struct TransverseChunk {
  std::size_t first_row{}, rows{}, first_node{}, nodes{};
};

struct SpatialCoefficients {
  ComplexView square, square_low, numerator, clamped;
  RealView transverse, transverse_low, frame, spectral_rate, user_rate, absorber;
  // A transverse collar's power rate per column (solprop kernel schema 31).
  RealView transverse_rate;
  // The hybrid planar kind's wall terms, (channels, n_omega); empty on free space.
  ComplexView2 kappa;
  std::size_t channels{};
  double evanescent{};
  // The longitudinal square q of a column and bin and its principal root, solprop's
  // `longitudinal_root`: the one evaluation the operator and the plane measure share.
  KOKKOS_INLINE_FUNCTION Complex longitudinal_square(const std::size_t mode,
                                                     const std::size_t omega) const {
    Complex longitudinal =
        (square(omega) - transverse(mode)) + (square_low(omega) - transverse_low(mode));
    // q = k^2 eps_co - k_y^2 - kappa_p^2, the wall term subtracted last as in solprop's
    // separable_values (kernel schema 15), so a free-space column's arithmetic is
    // untouched.
    if (channels != 0)
      longitudinal = longitudinal - kappa(mode % channels, omega);
    return longitudinal;
  }
  // A real q (a lossless medium) takes the real root, the same bits as the complex
  // root of q + 0i at a fraction of its cost (SEP 28): sqrt(q) or i sqrt(-q).
  static KOKKOS_INLINE_FUNCTION Complex root_of(const Complex longitudinal) {
    return longitudinal.imag() == 0.0
               ? (longitudinal.real() >= 0.0
                      ? Complex(Kokkos::sqrt(longitudinal.real()), 0.0)
                      : Complex(0.0, Kokkos::sqrt(-longitudinal.real())))
               : Kokkos::sqrt(longitudinal);
  }
  // The conjugate of the plane's dimensionless forward admittance `c k_z / omega`
  // (solprop SEP 34, `axial_admittance`) on an active propagating bin at positive
  // frequency, zero elsewhere: the factor an internal (conjugate) spectrum takes to
  // its magnetic closure. Its real part weighs energy and spectra.
  KOKKOS_INLINE_FUNCTION Complex closing(const std::size_t mode,
                                         const std::size_t omega, const bool active,
                                         const double frequency) const {
    const Complex longitudinal = longitudinal_square(mode, omega);
    if (!active || !(longitudinal.real() > 0.0) || !(frequency > 0.0))
      return Complex{};
    const Complex root = root_of(longitudinal);
    constexpr double speed_of_light = 299792458.0;
    return Complex(speed_of_light * root.real(), -(speed_of_light * root.imag())) /
           frequency;
  }
  // `value`'s prefactor alone, the same arithmetic: what a fold whose flows are
  // tabulated reads of the element.
  KOKKOS_INLINE_FUNCTION Complex prefactor(const std::size_t mode,
                                           const std::size_t omega,
                                           const bool active) const {
    if (!active)
      return Complex{};
    const Complex longitudinal = longitudinal_square(mode, omega);
    if (!(longitudinal.real() > 0.0))
      return clamped(omega);
    const Complex root = root_of(longitudinal);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return numerator(omega) * (1.0 / root.real());
#else
    return numerator(omega) / root.real();
#endif
  }
  KOKKOS_INLINE_FUNCTION SpatialElement value(const std::size_t mode,
                                              const std::size_t omega,
                                              const bool active) const {
    const Complex longitudinal = longitudinal_square(mode, omega);
    const Complex root = root_of(longitudinal);
    const bool propagating = longitudinal.real() > 0.0;
    const double attenuation =
        propagating ? root.imag() : Kokkos::min(Kokkos::abs(root.imag()), evanescent);
    // The transverse collar's rate joins the attenuation inside the active band, as
    // in solprop's separable_values (SEP 22, Q6); a zero rate changes no bit.
    const Complex linear = active ? Complex(-attenuation - 0.5 * transverse_rate(mode),
                                            -(root.real() - frame(omega)))
                                  : Complex{};
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    // One binary64 reciprocal and two products in place of two divisions, at every
    // element of the stage passes: a device's division is several fused steps (at
    // most an ulp from the quotients).
    const Complex prefactor =
        propagating ? numerator(omega) * (1.0 / root.real()) : clamped(omega);
#else
    const Complex prefactor =
        propagating ? numerator(omega) / root.real() : clamped(omega);
#endif
    return {.linear = linear - 0.5 * spectral_rate(omega),
            .prefactor = active ? prefactor : Complex{}};
  }
};

KOKKOS_INLINE_FUNCTION Complex exponential(const Complex value) {
  const double magnitude = Kokkos::exp(value.real());
  return {magnitude * Kokkos::cos(value.imag()), magnitude * Kokkos::sin(value.imag())};
}

// `cos theta` and `sin theta`. On a device they are one `sincos` call, which shares
// the argument's reduction between the two (on a GPU of reduced binary64 rate the
// stage propagators are about half of the stage entry); the host keeps the two
// library calls, so its values are unchanged.
KOKKOS_INLINE_FUNCTION void cosine_sine(const double theta, double& cosine,
                                        double& sine) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  ::sincos(theta, &sine, &cosine);
#else
  cosine = Kokkos::cos(theta);
  sine = Kokkos::sin(theta);
#endif
}

// A flow's magnitude: on a lossless bin `exp(+-0)`, exactly one, taken without a
// binary64 `exp` per element. Out of line on a host, whose call to its library was out
// of line already: inlined there, this branch took the tabulated fold's whole kernel
// past the compiler's inlining limit, 40 % slower (solprop's docs/dev/benchmarks.md).
#if defined(__GNUC__) && !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
[[gnu::noinline]]
#endif
KOKKOS_INLINE_FUNCTION double flow_magnitude(const double real) {
  return real == 0.0 ? 1.0 : Kokkos::exp(real);
}

// A stage or step propagator `exp(argument)`: `exponential` with the cosine and sine
// from `cosine_sine`, so it is `exponential` to the bit on the host.
KOKKOS_INLINE_FUNCTION Complex propagator_exp(const Complex argument) {
  const double magnitude = flow_magnitude(argument.real());
  double cosine = 0.0;
  double sine = 0.0;
  cosine_sine(argument.imag(), cosine, sine);
  return {magnitude * cosine, magnitude * sine};
}

// `exponential`, or on a lossless problem (a signed-zero real part) the same value
// with its magnitude `exp(+-0)`, exactly one, taken without the call. The cosine and
// sine come from `cosine_sine`, so on the host it is `exponential` to the bit.
template <bool Lossless>
KOKKOS_INLINE_FUNCTION Complex entry_exponential(const Complex value) {
  if constexpr (Lossless) {
    double cosine = 0.0;
    double sine = 0.0;
    cosine_sine(value.imag(), cosine, sine);
    return {cosine, sine};
  } else {
    return propagator_exp(value);
  }
}

KOKKOS_INLINE_FUNCTION double squared_abs(const Complex value) {
  return value.real() * value.real() + value.imag() * value.imag();
}

KOKKOS_INLINE_FUNCTION double trapezoid(const double current, const double previous,
                                        const double dt) {
  return dt * (current + previous) / 2.0;
}

// The fraction ionised from an exposure's decay `exp(-X)`, `1 - (1 - p) exp(-X)`, as
// `bodies.fraction_at` rounds it. A small response: strict arithmetic (solprop SEP 25
// section 4.2, `fp.hpp`), so the product and the difference round separately.
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION double decayed_fraction(const double decay,
                                               const double preionisation) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const double survival = (1.0 - preionisation) * decay;
  return 1.0 - survival;
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
// `decayed_fraction` of each of a host block's `count` decays, in place: one strict
// loop that vectorises, each product and difference rounded as there.
SOLPROP_NATIVE_STRICT_FUNCTION
inline void decayed_fractions(double* values, const std::size_t count,
                              const double preionisation) {
  SOLPROP_NATIVE_STRICT_BLOCK
  for (std::size_t k = 0; k < count; ++k) {
    const double survival = (1.0 - preionisation) * values[k];
    values[k] = 1.0 - survival;
  }
}
#endif

// The fraction ionised from an exposure, `decayed_fraction` of its `exp(-X)`.
KOKKOS_INLINE_FUNCTION double ionised_fraction(const double exposure,
                                               const double preionisation) {
  return decayed_fraction(Kokkos::exp(-exposure), preionisation);
}

// The envelope column's fraction, `p + (1 - p) (-expm1(-X))`, strict as above.
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION double envelope_fraction(const double exposure,
                                                const double preionisation) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const double ionised = (1.0 - preionisation) * (-Kokkos::expm1(-exposure));
  return preionisation + ionised;
}

// The band rule's fraction: the exposure floored at zero where the fraction is formed
// (SEP 18 section 4.5), `max(X, 0)` as NumPy's `maximum` takes it, so a NaN exposure
// stays NaN rather than becoming the preionisation.
KOKKOS_INLINE_FUNCTION double band_fraction(const double exposure,
                                            const double preionisation) {
  return ionised_fraction(exposure < 0.0 ? 0.0 : exposure, preionisation);
}

// A multistep compensation's residual term `(D - x / dz) u`: the difference of two
// nearly equal rates, strict (solprop SEP 25 section 4.2), each complex operation
// spelt with `fp.hpp`'s helpers.
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION Complex compensation_residual(const Complex rate,
                                                     const Complex exponent,
                                                     const double inverse_dz,
                                                     const Complex state) {
  SOLPROP_NATIVE_STRICT_BLOCK
  return strict_mul(strict_sub(rate, strict_scale(inverse_dz, exponent)), state);
}

// `bodies.loss_scale`: the loss current's coefficient of the field, or zero.
KOKKOS_INLINE_FUNCTION double band_loss_scale(const double rate, const double fraction,
                                              const double squared,
                                              const double ionisation_potential) {
  if (rate > 0.0 && squared > 0.0)
    return ionisation_potential * rate * (1.0 - fraction) / squared;
  return 0.0;
}

// The rate table's cubic in `log(rate)` at a field's `log |E|` within the table's
// range: the interval's local coordinate and its Horner form (`ionisation_rate`).
template <class View>
KOKKOS_INLINE_FUNCTION double
ionisation_log_rate(const double log_field, const View& coefficients,
                    const double log_field_min, const double d_log_field,
                    const std::size_t intervals, const std::size_t offset = 0) {
  long raw_index = static_cast<long>((log_field - log_field_min) / d_log_field);
  if (raw_index < 0)
    raw_index = 0;
  // NOLINTNEXTLINE(modernize-use-integer-sign-comparison): device code; std::cmp_* is host-only
  if (raw_index >= static_cast<long>(intervals))
    raw_index = static_cast<long>(intervals - 1);
  const auto index = static_cast<std::size_t>(raw_index);
  const double local =
      log_field - (log_field_min + static_cast<double>(index) * d_log_field);
  return ((coefficients(offset + index) * local +
           coefficients(offset + intervals + index)) *
              local +
          coefficients(offset + 2 * intervals + index)) *
             local +
         coefficients(offset + 3 * intervals + index);
}

// The ionisation rate at a field magnitude: the table's limits outside its range, and
// within it the exponential of its cubic in `log |E|`.
template <class View>
KOKKOS_INLINE_FUNCTION double
ionisation_rate(const double magnitude, const View& coefficients,
                const double log_field_min, const double d_log_field,
                const double field_min, const double field_max, const double rate_max,
                const std::size_t intervals, const std::size_t offset = 0) {
  if (magnitude > field_max)
    return rate_max;
  if (magnitude < field_min)
    return 0.0;
  return Kokkos::exp(ionisation_log_rate(Kokkos::log(magnitude), coefficients,
                                         log_field_min, d_log_field, intervals,
                                         offset));
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
// The samples a host pass of the plasma column takes at a time.
constexpr std::size_t rate_block = 256;

// `ionisation_rate`'s values at `live <= rate_block` magnitudes within the table's
// range, in place: one `log` and one `exp` pass of the vector-maths layer (solprop
// ADR-059) around the cubic. A host block gathers the magnitudes it passes, so the
// samples outside the range (most of a pulse's wings) cost what they cost before.
template <class View>
void ionisation_rates_within(double* values, const std::size_t live,
                             const View& coefficients, const double log_field_min,
                             const double d_log_field, const std::size_t intervals,
                             const std::size_t offset = 0) {
  // The cubic writes a local block, which no table gather can alias, so the loop
  // vectorises with the table's gathers wherever this is compiled out of line.
  double log_rates[rate_block];
  vmath::log(values, values, live);
  for (std::size_t k = 0; k < live; ++k)
    log_rates[k] = ionisation_log_rate(values[k], coefficients, log_field_min,
                                       d_log_field, intervals, offset);
  vmath::exp(log_rates, values, live);
}
#endif

// A host span as an unmanaged view of a state-shaped memory-space view, for one copy
// between them: no staging mirror, no element loop. A span whose data is not aligned
// as `Complex` (a NumPy buffer need not be) is staged in an aligned vector first.
inline void upload_span(const ExecutionSpace& execution_space, crossing::Ring& ring,
                        const ComplexView2& destination,
                        const std::span<const Complex> source) {
  if (destination.size() != source.size())
    throw std::invalid_argument("fixed problem array has the wrong extent");
  if (reinterpret_cast<std::uintptr_t>(source.data()) % alignof(Complex) != 0) {
    const std::vector<Complex> aligned(source.begin(), source.end());
    upload_span(execution_space, ring, destination, aligned);
    execution_space.fence("solprop_native unaligned upload");
    return;
  }
  if (crossing::takes_ring(source.size_bytes())) {
    execution_space.fence("solprop_native upload");
    ring.from_host(destination.data(), source.data(), source.size_bytes());
    return;
  }
  const Kokkos::View<const Complex**, Layout, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host(source.data(), destination.extent(0), destination.extent(1));
  graph::deep_copy(execution_space, destination, host);
}

template <class View, class Value>
void copy_vector(const ExecutionSpace& execution_space, const View& destination,
                 const std::vector<Value>& source) {
  if (destination.size() != source.size())
    throw std::invalid_argument("fixed problem array has the wrong extent");
  auto mirror = Kokkos::create_mirror_view(destination);
  auto* data = mirror.data();
  for (std::size_t index = 0; index < source.size(); ++index)
    data[index] = source[index];
  graph::deep_copy(execution_space, destination, mirror);
}

inline void require_extent(const std::size_t actual, const std::size_t expected,
                           const char* name) {
  if (actual != expected)
    throw std::invalid_argument(std::string(name) + " has the wrong extent");
}

inline std::size_t checked_product(const std::size_t left, const std::size_t right,
                                   const char* name) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(std::string(name) + " extent overflows size_t");
  return left * right;
}

inline std::size_t checked_sum(const std::size_t left, const std::size_t right,
                               const char* name) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::length_error(std::string(name) + " extent overflows size_t");
  return left + right;
}

inline void require_allocation(const std::size_t elements,
                               const std::size_t element_bytes, const char* name) {
  static_cast<void>(checked_product(elements, element_bytes, name));
}

// The samples of one tile of the device trapezoid plasma's two-level scans (unused on
// a host build).
[[maybe_unused]] inline constexpr std::size_t plasma_tile_samples = 1024;

// Descriptors borrow allocations retained by the member GeneralState owners. They
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
// on the member's columns and bins alone, so a batch member and its solo run sum in one
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

// The bins a host work item of the stage's entry kernel takes at a time: every bin's
// phase sum is the same sequence of rounded products and additions in term order as
// a per-bin loop, laid out so that the bins vectorise.
constexpr std::size_t stage_block = 64;
// The pinned staging buffers a state's per-trial scalars cross from (`upload_trial`).
constexpr std::size_t stage_staging_ring = 8;

// The pointwise bodies of a stage over the coarse spectrum: its entry (the phase
// increment, the combination with the earlier stages, the lab frame, the noise and the
// pad) and its finish (prefactor, window, absorber, return to the interaction picture
// and accumulation). The entry kernel runs the finish of the stage before it in the
// same traversal when nothing lies between them (solprop SEP 25 section 6.7): both
// are pointwise over the bins, and the combination reads only its own bin's stages.
// How a stage's entry forms its propagator pair (`StageTerms::combine`).
enum class EntryPhase { general, lossless, repeated };

// The transcendental factors of a stage's combination at one bin: the phase's
// `cos b` and `sin b`, its magnitudes `e^a` and `e^-a` (one on a lossless problem),
// and the noise phase's cosine and sine.
struct EntryFactors {
  double cosine = 0.0, sine = 0.0, magnitude = 1.0, inverse = 1.0;
  double noise_cosine = 0.0, noise_sine = 0.0;
};

// A host work item's factors for its block of bins, formed by the vector-maths layer
// in one pass of each function (`StageTerms::entry_factors_block`); the factors a
// stage does not use stay zero.
struct EntryBlock {
  double cosine[stage_block], sine[stage_block];
  double magnitude[stage_block], inverse[stage_block];
  double noise_cosine[stage_block], noise_sine[stage_block];

  [[nodiscard]] EntryFactors at(const std::size_t bin) const {
    return {.cosine = cosine[bin],
            .sine = sine[bin],
            .magnitude = magnitude[bin],
            .inverse = inverse[bin],
            .noise_cosine = noise_cosine[bin],
            .noise_sine = noise_sine[bin]};
  }
};

struct StageTerms {
  ComplexView2 current, fine, whole, solution, error, stage_phase, stage_inverse, noise;
  ComplexView2 flow_input, flow_lab, flow_drive, general_scaled, general_source;
  ComplexView2 projected, absorber;
  ComplexView3 stages, phi_terms, prefactor_terms;
  RealView tableau, phi_delta, phi_at, prefactor_weights, window;
  IndexView crop, gaps;
  ByteView active;
  std::size_t gap_ranges = 0;
  std::size_t n = 0, n_phi = 0, n_stages = 0, n_prefactor = 0, n_to = 0;
  double inverse_n_to = 0.0, general_pad = 0.0;
  bool add_noise = false, analytic = false, general = false, flow = false;
  bool carried = false, keep_lab = false, zero_response = false;
  bool in_stage_absorber = false;
  // Every coarse bin is the fine bin of its own index (`crop` is the identity).
  bool identity_crop = false;

  // The stage's phase increment `sum_t dPhi_t Phi_t` at one bin: strict (solprop SEP
  // 25 section 4.2), each product and sum rounded in term order. A zero weight's term
  // is skipped: its product with a finite term is a signed zero, which leaves the
  // running sum (started at +0) unchanged, so the skip is exact. A z table's terms
  // are its values at a trial's positions, weighed by +-1 and mostly zero (SEP 29 G1).
  SOLPROP_NATIVE_STRICT_FUNCTION
  KOKKOS_INLINE_FUNCTION Complex phase_sum(const std::size_t offset,
                                           const std::size_t column,
                                           const std::size_t index) const {
    SOLPROP_NATIVE_STRICT_BLOCK
    double real = 0.0;
    double imag = 0.0;
    for (std::size_t term = 0; term < n_phi; ++term) {
      const double weight = phi_delta(offset + term);
      if (weight == 0.0)
        continue;
      const Complex fixed = phi_terms(term, column, index);
      const double scaled_real = weight * fixed.real();
      const double scaled_imag = weight * fixed.imag();
      real = real + scaled_real;
      imag = imag + scaled_imag;
    }
    return {real, imag};
  }

  // The same sums over a block of bins, laid out so that the bins vectorise.
  SOLPROP_NATIVE_STRICT_FUNCTION
  KOKKOS_INLINE_FUNCTION void
  phase_block(const std::size_t offset, const std::size_t column,
              const std::size_t first, const std::size_t count,
              double (&real)[stage_block], double (&imag)[stage_block]) const {
    SOLPROP_NATIVE_STRICT_BLOCK
    for (std::size_t bin = 0; bin < stage_block; ++bin) {
      real[bin] = 0.0;
      imag[bin] = 0.0;
    }
    for (std::size_t term = 0; term < n_phi; ++term) {
      const double weight = phi_delta(offset + term);
      if (weight == 0.0)
        continue;
      for (std::size_t bin = 0; bin < count; ++bin) {
        const Complex fixed = phi_terms(term, column, first + bin);
        const double scaled_real = weight * fixed.real();
        const double scaled_imag = weight * fixed.imag();
        real[bin] = real[bin] + scaled_real;
        imag[bin] = imag[bin] + scaled_imag;
      }
    }
  }

  // The noise's phase: strict (solprop SEP 25 section 4.2).
  SOLPROP_NATIVE_STRICT_FUNCTION
  KOKKOS_INLINE_FUNCTION Complex noise_phase(const std::size_t stage,
                                             const std::size_t column,
                                             const std::size_t index) const {
    SOLPROP_NATIVE_STRICT_BLOCK
    Complex sum{};
    for (std::size_t term = 0; term < n_phi; ++term) {
      const double weight = phi_at(stage * n_phi + term);
      // Exact, as in `phase_sum`.
      if (weight != 0.0)
        sum = strict_accumulate(sum, weight, phi_terms(term, column, index));
    }
    return sum;
  }

  // The factors of stage `stage`'s combination at one bin, one bin at a time (a
  // device's entry; the host forms a block's together, `entry_factors_block`). The
  // propagator `exp(phase)` and its inverse `exp(-phase)` share their cosine and sine,
  // each magnitude is its own `exp`, and a lossless problem's are exactly one, taken
  // without the calls (`combine`).
  template <EntryPhase Mode>
  KOKKOS_INLINE_FUNCTION EntryFactors entry_factors(const std::size_t stage,
                                                    const std::size_t column,
                                                    const std::size_t index,
                                                    const Complex phase) const {
    EntryFactors factors;
    if constexpr (Mode != EntryPhase::repeated)
      cosine_sine(phase.imag(), factors.cosine, factors.sine);
    if constexpr (Mode == EntryPhase::general) {
      factors.magnitude = Kokkos::exp(phase.real());
      factors.inverse = Kokkos::exp(-phase.real());
    }
    if (add_noise) {
      const Complex phase_at = noise_phase(stage, column, index);
      factors.noise_cosine = Kokkos::cos(phase_at.imag());
      factors.noise_sine = Kokkos::sin(phase_at.imag());
    }
    return factors;
  }

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  // Whether the first `count` of a block's phases have real parts that are all zeros.
  static bool lossless_block(const double (&real)[stage_block],
                             const std::size_t count) {
    bool zero = true;
    for (std::size_t bin = 0; bin < count; ++bin)
      zero = zero && real[bin] == 0.0;
    return zero;
  }

  // A host work item's factors for its `count` bins from `first`, each function one
  // pass of the vector-maths layer over the block (solprop ADR-059): the phases' sines
  // and cosines, their magnitudes and inverse magnitudes, and the noise phases'.
  template <EntryPhase Mode>
  [[maybe_unused]] void
  entry_factors_block(const std::size_t stage, const std::size_t column,
                      const std::size_t first, const std::size_t count,
                      const double (&real)[stage_block],
                      const double (&imag)[stage_block], EntryBlock& block) const {
    if constexpr (Mode != EntryPhase::repeated)
      vmath::sincos(imag, block.sine, block.cosine, count);
    if constexpr (Mode == EntryPhase::general) {
      // A block whose phases' real parts are all (signed) zeros -- a lossy problem's
      // loss often lies in a few bands -- has both magnitudes `exp(+-0)`, exactly one,
      // without the calls.
      if (lossless_block(real, count)) {
        for (std::size_t bin = 0; bin < count; ++bin) {
          block.magnitude[bin] = 1.0;
          block.inverse[bin] = 1.0;
        }
      } else {
        double negated[stage_block];
        for (std::size_t bin = 0; bin < count; ++bin)
          negated[bin] = -real[bin];
        vmath::exp(real, block.magnitude, count);
        vmath::exp(negated, block.inverse, count);
      }
    }
    if (add_noise) {
      double angle[stage_block]{};
      for (std::size_t bin = 0; bin < count; ++bin)
        angle[bin] = noise_phase(stage, column, first + bin).imag();
      vmath::sincos(angle, block.noise_sine, block.noise_cosine, count);
    }
  }
#endif

  // The combination of stage `stage` at one bin, from the stage's phase increment and
  // its factors (`entry_factors`). The propagator `exp(phase)` and its inverse
  // `exp(-phase)`, which the finish applies, share their cosine and sine (solprop SEP
  // 25 section 6.7): the angle's parts are even and odd, so `exp(-phase)` is
  // `e^{-a} (cos b, -sin b)` with its own magnitude, never a reciprocal of the forward
  // value. A coupled flow's residual reads the phase. A lossless problem
  // (`EntryPhase::lossless`: every Phi term's real part exactly zero, so the phase's
  // is a signed zero) takes both magnitudes as `exp(+-0)`, exactly one, without the
  // calls. A lossless stage whose phase weights equal the stage before's
  // (`EntryPhase::repeated`) has that stage's inverse still in place, and its
  // propagator is that inverse's conjugate, `(cos b, -(-sin b))`, to the bit.
  // Returns the propagator.
  template <EntryPhase Mode = EntryPhase::general>
  KOKKOS_INLINE_FUNCTION Complex combine(const std::size_t stage, const double dz,
                                         const double solution_weight,
                                         const double error_weight,
                                         const std::size_t column,
                                         const std::size_t index, const Complex phase,
                                         const EntryFactors& factors) const {
    Complex propagator;
    if constexpr (Mode == EntryPhase::repeated) {
      const Complex inverse = stage_inverse(column, index);
      propagator = Complex(inverse.real(), -inverse.imag());
    } else if constexpr (Mode == EntryPhase::lossless) {
      propagator = Complex(factors.cosine, factors.sine);
      stage_inverse(column, index) = Complex(factors.cosine, -factors.sine);
    } else {
      propagator =
          Complex(factors.magnitude * factors.cosine, factors.magnitude * factors.sine);
      stage_inverse(column, index) =
          Complex(factors.inverse * factors.cosine, -(factors.inverse * factors.sine));
    }
    if (flow)
      stage_phase(column, index) = phase;
    Complex value{};
    if (carried) {
      value = propagator * flow_input(column, index);
    } else {
      Complex accumulated{};
      for (std::size_t previous = 0; previous < stage; ++previous)
        accumulated +=
            tableau(stage * n_stages + previous) * stages(previous, column, index);
      value = propagator * (current(column, index) + dz * accumulated);
    }
    if (add_noise)
      value += noise(column, index) * Complex(factors.noise_cosine, factors.noise_sine);
    const double analytic_scale =
        analytic && index > 0 && 2 * crop(index) < n_to ? 2.0 : 1.0;
    fine(column, crop(index)) = analytic_scale * inverse_n_to * value;
    if (keep_lab)
      flow_lab(column, index) = value;
    // The general route synthesises on the coarse grid: the pad scale and the
    // analytic doubling go on the channel spectra, the 1/n_to on its transform.
    if (general)
      general_scaled(column, index) =
          value * (general_pad * general_doubling(crop(index), n_to));
    if (zero_response && flow) {
      flow_drive(column, index) = Complex{};
    } else if (zero_response) {
      stages(stage, column, index) = Complex{};
      if (stage == 0) {
        solution(column, index) = Complex{};
        error(column, index) = Complex{};
      } else {
        solution(column, index) += solution_weight * Complex{};
        error(column, index) += error_weight * Complex{};
      }
    }
    return propagator;
  }

  // The fine bins no coarse bin fills, positions `[begin, end)` of this column's
  // list, zeroed for the inverse transform (which may destroy its input). The list is
  // held as runs of adjacent bins, `gaps(3 r + {0, 1, 2})` the run's first position,
  // first bin and length, so each run zeroes contiguously.
  KOKKOS_INLINE_FUNCTION void zero_gaps(const std::size_t column,
                                        const std::size_t begin,
                                        const std::size_t end) const {
    for (std::size_t run = 0; run < gap_ranges; ++run) {
      const std::size_t position = gaps(3 * run);
      const std::size_t bin = gaps(3 * run + 1);
      const std::size_t length = gaps(3 * run + 2);
      const std::size_t lower = begin > position ? begin : position;
      const std::size_t upper = end < position + length ? end : position + length;
      for (std::size_t gap = lower; gap < upper; ++gap)
        fine(column, bin + (gap - position)) = Complex{};
    }
  }

  // The finish of stage `stage` at one bin, with the inverse propagator its
  // combination formed.
  KOKKOS_INLINE_FUNCTION void finish(const std::size_t stage,
                                     const double solution_weight,
                                     const double error_weight,
                                     const std::size_t column,
                                     const std::size_t index) const {
    // A zero weight is skipped: a z table's terms are its values at a trial's
    // positions, one-hot weighted (SEP 29 G1). The sum's value is unchanged; at most
    // the sign of an exactly zero sum could differ.
    Complex prefactor{};
    for (std::size_t term = 0; term < n_prefactor; ++term) {
      const double weight = prefactor_weights(stage * n_prefactor + term);
      if (weight != 0.0)
        prefactor += weight * prefactor_terms(term, column, index);
    }
    Complex value =
        prefactor * window(index) *
        (general ? general_source(column, index) : projected(column, crop(index)));
    if (in_stage_absorber && active(index) != 0U)
      value += absorber(column, crop(index));
    value *= stage_inverse(column, index);
    if (flow) {
      flow_drive(column, index) = value;
      return;
    }
    stages(stage, column, index) = value;
    if (stage == 0) {
      solution(column, index) = solution_weight * value;
      error(column, index) = error_weight * value;
    } else {
      solution(column, index) += solution_weight * value;
      error(column, index) += error_weight * value;
    }
  }

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  // Whether a host work item takes its finish and combination as the block forms
  // below: a stage that projects a source into the stage rows, off a coupled flow and
  // the general route, whose stores the per-bin forms keep.
  [[nodiscard]] bool blocks_vectorise() const noexcept {
    return !flow && !general && !zero_response;
  }

  // `finish` over the `count` bins of a block from `first`, each bin the same
  // arithmetic in the same order, the block's parts held as separate real and
  // imaginary lanes so that the bins vectorise; only the crop's gathers stay per bin.
  [[maybe_unused]] void finish_block(const std::size_t stage,
                                     const double solution_weight,
                                     const double error_weight,
                                     const std::size_t column, const std::size_t first,
                                     const std::size_t count) const {
    double real[stage_block];
    double imag[stage_block];
    double source_real[stage_block];
    double source_imag[stage_block];
    for (std::size_t bin = 0; bin < count; ++bin) {
      real[bin] = 0.0;
      imag[bin] = 0.0;
    }
    for (std::size_t term = 0; term < n_prefactor; ++term) {
      const double weight = prefactor_weights(stage * n_prefactor + term);
      if (weight == 0.0)
        continue;
      const Complex* values = &prefactor_terms(term, column, first);
      for (std::size_t bin = 0; bin < count; ++bin) {
        real[bin] += weight * values[bin].real();
        imag[bin] += weight * values[bin].imag();
      }
    }
    const std::size_t* bins = &crop(first);
    const auto gather = [&](const Complex* row) {
      if (identity_crop)
        for (std::size_t bin = 0; bin < count; ++bin) {
          source_real[bin] = row[first + bin].real();
          source_imag[bin] = row[first + bin].imag();
        }
      else
        for (std::size_t bin = 0; bin < count; ++bin) {
          source_real[bin] = row[bins[bin]].real();
          source_imag[bin] = row[bins[bin]].imag();
        }
    };
    gather(&projected(column, 0));
    const double* weights = &window(first);
    for (std::size_t bin = 0; bin < count; ++bin) {
      const double scaled_real = real[bin] * weights[bin];
      const double scaled_imag = imag[bin] * weights[bin];
      real[bin] = scaled_real * source_real[bin] - scaled_imag * source_imag[bin];
      imag[bin] = scaled_real * source_imag[bin] + scaled_imag * source_real[bin];
    }
    if (in_stage_absorber) {
      gather(&absorber(column, 0));
      const unsigned char* band = &active(first);
      for (std::size_t bin = 0; bin < count; ++bin) {
        const bool inside = band[bin] != 0U;
        real[bin] = inside ? real[bin] + source_real[bin] : real[bin];
        imag[bin] = inside ? imag[bin] + source_imag[bin] : imag[bin];
      }
    }
    const Complex* inverse = &stage_inverse(column, first);
    Complex* row = &stages(stage, column, first);
    Complex* total = &solution(column, first);
    Complex* estimate = &error(column, first);
    for (std::size_t bin = 0; bin < count; ++bin) {
      const double value_real =
          real[bin] * inverse[bin].real() - imag[bin] * inverse[bin].imag();
      const double value_imag =
          real[bin] * inverse[bin].imag() + imag[bin] * inverse[bin].real();
      row[bin] = Complex(value_real, value_imag);
      if (stage == 0) {
        total[bin] =
            Complex(solution_weight * value_real, solution_weight * value_imag);
        estimate[bin] = Complex(error_weight * value_real, error_weight * value_imag);
      } else {
        total[bin] = Complex(total[bin].real() + solution_weight * value_real,
                             total[bin].imag() + solution_weight * value_imag);
        estimate[bin] = Complex(estimate[bin].real() + error_weight * value_real,
                                estimate[bin].imag() + error_weight * value_imag);
      }
    }
  }

  // `combine` over the `count` bins of a block from `first`, from the block's factors,
  // in the lanes of `finish_block`; with `store_whole`, each propagator is also the
  // whole step's.
  template <EntryPhase Mode>
  [[maybe_unused]] void
  combine_block(const std::size_t stage, const double dz, const std::size_t column,
                const std::size_t first, const std::size_t count,
                const EntryBlock& factors, const bool store_whole) const {
    double real[stage_block];
    double imag[stage_block];
    double sum_real[stage_block];
    double sum_imag[stage_block];
    Complex* inverse = &stage_inverse(column, first);
    for (std::size_t bin = 0; bin < count; ++bin) {
      if constexpr (Mode == EntryPhase::repeated) {
        real[bin] = inverse[bin].real();
        imag[bin] = -inverse[bin].imag();
      } else if constexpr (Mode == EntryPhase::lossless) {
        real[bin] = factors.cosine[bin];
        imag[bin] = factors.sine[bin];
        inverse[bin] = Complex(factors.cosine[bin], -factors.sine[bin]);
      } else {
        real[bin] = factors.magnitude[bin] * factors.cosine[bin];
        imag[bin] = factors.magnitude[bin] * factors.sine[bin];
        inverse[bin] = Complex(factors.inverse[bin] * factors.cosine[bin],
                               -(factors.inverse[bin] * factors.sine[bin]));
      }
    }
    if (store_whole) {
      Complex* propagator = &whole(column, first);
      for (std::size_t bin = 0; bin < count; ++bin)
        propagator[bin] = Complex(real[bin], imag[bin]);
    }
    for (std::size_t bin = 0; bin < count; ++bin) {
      sum_real[bin] = 0.0;
      sum_imag[bin] = 0.0;
    }
    for (std::size_t previous = 0; previous < stage; ++previous) {
      const double weight = tableau(stage * n_stages + previous);
      const Complex* earlier = &stages(previous, column, first);
      for (std::size_t bin = 0; bin < count; ++bin) {
        sum_real[bin] += weight * earlier[bin].real();
        sum_imag[bin] += weight * earlier[bin].imag();
      }
    }
    const Complex* state = &current(column, first);
    for (std::size_t bin = 0; bin < count; ++bin) {
      const double input_real = state[bin].real() + dz * sum_real[bin];
      const double input_imag = state[bin].imag() + dz * sum_imag[bin];
      const double value_real = real[bin] * input_real - imag[bin] * input_imag;
      const double value_imag = real[bin] * input_imag + imag[bin] * input_real;
      real[bin] = value_real;
      imag[bin] = value_imag;
    }
    if (add_noise) {
      const Complex* draw = &noise(column, first);
      for (std::size_t bin = 0; bin < count; ++bin) {
        const double noise_real = draw[bin].real() * factors.noise_cosine[bin] -
                                  draw[bin].imag() * factors.noise_sine[bin];
        const double noise_imag = draw[bin].real() * factors.noise_sine[bin] +
                                  draw[bin].imag() * factors.noise_cosine[bin];
        real[bin] += noise_real;
        imag[bin] += noise_imag;
      }
    }
    Complex* padded = &fine(column, 0);
    const std::size_t* bins = &crop(first);
    if (identity_crop && !analytic) {
      for (std::size_t bin = 0; bin < count; ++bin)
        padded[first + bin] =
            Complex(inverse_n_to * real[bin], inverse_n_to * imag[bin]);
      return;
    }
    for (std::size_t bin = 0; bin < count; ++bin) {
      const double analytic_scale =
          analytic && first + bin > 0 && 2 * bins[bin] < n_to ? 2.0 : 1.0;
      const double scale = analytic_scale * inverse_n_to;
      padded[bins[bin]] = Complex(scale * real[bin], scale * imag[bin]);
    }
  }
#endif
};

} // namespace solprop_native::general
