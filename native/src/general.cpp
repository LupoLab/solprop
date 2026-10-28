// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/core.hpp"
#include "solprop_native/coupled_flow.hpp"
#include "solprop_native/dispersive.hpp"
#include "solprop_native/envelope_plasma.hpp"
#include "solprop_native/far_field.hpp"
#include "solprop_native/fft_plan.hpp"
#include "solprop_native/fp.hpp"
#include "solprop_native/gemm.hpp"
#include "solprop_native/general_modal.hpp"
#include "solprop_native/plasma.hpp"

#include "solprop_native/raman_fft.hpp"
#include "solprop_native/slab.hpp"
#include "solprop_native/z_table.hpp"
#include <KokkosFFT.hpp>
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
#include <fftw3.h>
#if defined(__SSE2__)
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
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#if defined(__APPLE__)
#include <malloc/malloc.h>
#elif defined(__GLIBC__)
#include <dlfcn.h>
#endif

namespace solprop_native::host {
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
} // namespace solprop_native::host

template <> struct Kokkos::reduction_identity<solprop_native::host::NormSums> {
  KOKKOS_FUNCTION static solprop_native::host::NormSums sum() { return {}; }
};

namespace solprop_native {
namespace {

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
  Kokkos::deep_copy(space, out, host);
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
  Kokkos::parallel_for(
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
  Kokkos::parallel_for(
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

template <class... Ts> KOKKOS_INLINE_FUNCTION void capture_anchor(const Ts&...) {}

// Net process allocator growth includes opaque vendor FFT plan/workspace
// allocations. It is measured separately from Kokkos Views and is not a peak bound.
std::size_t heap_bytes() {
#if defined(__APPLE__)
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
std::size_t device_free_bytes() {
#if defined(KOKKOS_ENABLE_CUDA)
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
using InverseRealPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView2, RealView2>;
using ForwardRealPlan = KokkosFFT::Plan<ExecutionSpace, RealView2, ComplexView2>;
using RamanForwardPlan = RamanFftPlan<ExecutionSpace, RealView2, ComplexView2>;
using RamanInversePlan = RamanFftPlan<ExecutionSpace, ComplexView2, RealView2>;
using ComplexPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView2, ComplexView2>;
using TransversePlan = KokkosFFT::Plan<ExecutionSpace, ComplexView3, ComplexView3, 2>;
// The hybrid planar kind transforms y alone, the trailing axis of (bins, n_p, n_y).
using SlabPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView3, ComplexView3, 1>;

// Each current keeps its own chunk accumulator, including stacked embedded
// quadrature. Sharing the latter would mix currents between partial chunks.
struct EnvelopeStream {
  ComplexView2 node, projected, coarse, stacked, spectrum, coarse_spectrum;
  std::unique_ptr<ComplexPlan> forward, coarse_forward;
  std::size_t bytes() const {
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
  std::size_t bytes() const {
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
  Kokkos::parallel_for(label, Range2(execution, {0, 0}, {rows, width}), element);
#else
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecutionSpace>(execution, 0, rows),
      KOKKOS_LAMBDA(const std::size_t row) {
        for (std::size_t inner = 0; inner < width; ++inner)
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
#if defined(SOLPROP_NATIVE_STREAMING_STORES)
  _mm_stream_pd(reinterpret_cast<double*>(&target),
                _mm_set_pd(value.imag(), value.real()));
#else
  target = value;
#endif
}

[[maybe_unused]] KOKKOS_INLINE_FUNCTION void stream_fence() {
#if defined(SOLPROP_NATIVE_STREAMING_STORES)
  _mm_sfence();
#endif
}

// On a device, `RowFastest` maps adjacent rows (columns of the plane) to adjacent
// threads, so the transverse scratch side of a gather or scatter is coalesced; the
// chunk-local side is the strided one.
template <bool Streaming = false, bool RowFastest = false, class Function>
void crossing_for(const char* label, const ExecutionSpace& execution,
                  const std::size_t rows, const std::size_t width,
                  const Function& element) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  if constexpr (RowFastest) {
    using Left = Kokkos::MDRangePolicy<
        ExecutionSpace, Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
        Kokkos::IndexType<std::int64_t>>;
    Kokkos::parallel_for(
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
  constexpr std::size_t tile_rows = crossing_tile_columns;
  constexpr std::size_t tile_bins = crossing_tile_bins;
  const std::size_t row_tiles = (rows + tile_rows - 1) / tile_rows;
  const std::size_t bin_tiles = (width + tile_bins - 1) / tile_bins;
  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<ExecutionSpace>(execution, 0, row_tiles * bin_tiles),
      KOKKOS_LAMBDA(const std::size_t tile) {
        const std::size_t first_row = (tile / bin_tiles) * tile_rows;
        const std::size_t first_bin = (tile % bin_tiles) * tile_bins;
        const std::size_t last_row =
            rows - first_row < tile_rows ? rows : first_row + tile_rows;
        const std::size_t last_bin =
            width - first_bin < tile_bins ? width : first_bin + tile_bins;
        for (std::size_t inner = first_bin; inner < last_bin; ++inner)
          for (std::size_t row = first_row; row < last_row; ++row)
            element(row, inner);
        // Each thread orders its own streaming stores before the region ends.
        if constexpr (Streaming)
          stream_fence();
      });
#endif
}

struct SpatialElement {
  Complex linear, prefactor;
};

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
// A device crossing through a 32-by-32 tile of team scratch (SEP 28): the tile is read
// with adjacent threads along the side that is contiguous for `read` (rows when
// `read_rows`, else bins) and written with adjacent threads along the other, so both
// the transverse scratch and the chunk's rows are accessed coalesced.
template <class Read, class Write>
void tiled_crossing(const char* label, const ExecutionSpace& execution,
                    const std::size_t rows, const std::size_t width,
                    const bool read_rows, const Read& read, const Write& write) {
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
  Kokkos::parallel_for(
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
  Kokkos::parallel_for(
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
  const unsigned effort = fftw_seed_effort(FftShape::planes, space.concurrency());
  if (effort == FFTW_ESTIMATE)
    return;
  fftw_init_threads();
  fftw_plan_with_nthreads(space.concurrency());
  auto* in = reinterpret_cast<fftw_complex*>(input);
  auto* out = reinterpret_cast<fftw_complex*>(output);
  const fftw_plan seeded = fftw_plan_many_dft(
      static_cast<int>(extents.size()), extents.data(), howmany, in, nullptr, 1,
      input_distance, out, nullptr, 1, output_distance, sign, effort);
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

// One chunk of a separable geometry's response pass: whole transverse units (a node
// on free space, a y row of n_p columns and n_x nodes on the hybrid planar kind), its
// temporal rows (state columns) and its physical nodes.
struct TransverseChunk {
  std::size_t first_row{}, rows{}, first_node{}, nodes{};
};

struct SpatialCoefficients {
  ComplexView square, square_low, numerator, clamped;
  RealView transverse, transverse_low, frame, spectral_rate, absorber;
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
    const Complex prefactor =
        propagating ? numerator(omega) / root.real() : clamped(omega);
    return {linear - 0.5 * spectral_rate(omega), active ? prefactor : Complex{}};
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

// A stage or step propagator `exp(argument)`: `exponential` with the cosine and sine
// from `cosine_sine`, so it is `exponential` to the bit on the host.
KOKKOS_INLINE_FUNCTION Complex propagator_exp(const Complex argument) {
  const double magnitude = Kokkos::exp(argument.real());
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

// The fraction ionised from an exposure, `1 - (1 - p) exp(-X)`, as `bodies.fraction_at`
// rounds it. A small response: strict arithmetic (solprop SEP 25 section 4.2,
// `fp.hpp`), so the product and the difference round separately.
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION double ionised_fraction(const double exposure,
                                               const double preionisation) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const double survival = (1.0 - preionisation) * Kokkos::exp(-exposure);
  return 1.0 - survival;
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
  const double log_field = Kokkos::log(magnitude);
  long raw_index = static_cast<long>((log_field - log_field_min) / d_log_field);
  if (raw_index < 0)
    raw_index = 0;
  if (raw_index >= static_cast<long>(intervals))
    raw_index = static_cast<long>(intervals - 1);
  const auto index = static_cast<std::size_t>(raw_index);
  const double local =
      log_field - (log_field_min + static_cast<double>(index) * d_log_field);
  const double cubic = ((coefficients(offset + index) * local +
                         coefficients(offset + intervals + index)) *
                            local +
                        coefficients(offset + 2 * intervals + index)) *
                           local +
                       coefficients(offset + 3 * intervals + index);
  return Kokkos::exp(cubic);
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
  Kokkos::deep_copy(execution_space, destination, mirror);
}

void require_extent(const std::size_t actual, const std::size_t expected,
                    const char* name) {
  if (actual != expected)
    throw std::invalid_argument(std::string(name) + " has the wrong extent");
}

std::size_t checked_product(const std::size_t left, const std::size_t right,
                            const char* name) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(std::string(name) + " extent overflows size_t");
  return left * right;
}

std::size_t checked_sum(const std::size_t left, const std::size_t right,
                        const char* name) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::length_error(std::string(name) + " extent overflows size_t");
  return left + right;
}

void require_allocation(const std::size_t elements, const std::size_t element_bytes,
                        const char* name) {
  static_cast<void>(checked_product(elements, element_bytes, name));
}

void require_general_problem(const Problem& problem) {
  require_general_modal(problem);
  require_coupled_flow(problem);
  require_dispersive(problem);
  if (problem.n_omega == 0 || problem.n_t == 0 || problem.n_to == 0 ||
      problem.n_modes == 0 || problem.n_pol_state == 0 || problem.n_nodes == 0 ||
      problem.n_pol_transform == 0)
    throw std::invalid_argument("all transform and FFT extents must be positive");
  const auto effective_node_pol = checked_product(
      problem.n_pol_transform, problem.n_pol_state, "effective node polarisation");
  if (effective_node_pol > 2)
    throw std::invalid_argument("at most two effective node components are supported");
  if (problem.node_chunk == 0 || problem.node_chunk > problem.n_nodes)
    throw std::invalid_argument("node_chunk must lie in [1, n_nodes]");
  if (problem.n_stages == 0)
    throw std::invalid_argument("the tableau must contain a stage");
  const auto columns =
      checked_product(problem.n_modes, problem.n_pol_state, "field columns");
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr auto policy_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (problem.n_to > policy_limit || problem.n_t > policy_limit ||
      problem.n_omega > policy_limit || columns > policy_limit ||
      problem.node_chunk > policy_limit)
    throw std::length_error("accelerator policy extent exceeds signed 32-bit range");
#endif
  const auto rows =
      checked_product(problem.n_nodes, problem.n_pol_transform, "transform rows");
  const auto field = checked_product(columns, problem.n_omega, "field");
  const auto modal_time = checked_product(columns, problem.n_to, "modal time");
  const auto chunk_rows =
      checked_product(problem.node_chunk, effective_node_pol, "node chunk rows");
  const auto node_time = checked_product(chunk_rows, problem.n_to, "node time");
  // The hybrid planar kind carries its x half alone, (n_x, n_p) and its transpose.
  const auto transform =
      problem.free_geometry() ? 0
      : problem.slab_geometry()
          ? checked_product(problem.n_x, problem.n_p, "slab transform matrix")
          : checked_product(rows, problem.n_modes, "transform matrix");
  const auto stage_field = checked_product(problem.n_stages, field, "stage field");
  require_allocation(field, sizeof(Complex), "field bytes");
  require_allocation(modal_time, sizeof(Complex), "modal-time bytes");
  require_allocation(node_time, sizeof(Complex), "node-time bytes");
  require_allocation(transform, sizeof(double), "transform-matrix bytes");
  if (!problem.projection_coarse_matrix.empty() &&
      problem.family != FieldFamily::envelope) {
    const auto stacked_rows =
        checked_product(2, problem.n_modes, "stacked projection rows");
    require_allocation(
        checked_product(stacked_rows, problem.n_to, "stacked projection"),
        sizeof(double), "stacked-projection bytes");
  }
  require_allocation(stage_field, sizeof(Complex), "stage-field bytes");
  if (problem.plasma && !problem.envelope_plasma) {
    require_allocation(
        checked_product(chunk_rows, problem.n_omega, "held nodal spectrum"),
        sizeof(Complex), "held nodal spectrum bytes");
    require_allocation(checked_product(2, modal_time, "held stacked projection"),
                       sizeof(double), "held stacked projection bytes");
    require_allocation(
        checked_product(problem.n_modes,
                        checked_product(problem.node_chunk, problem.n_pol_transform,
                                        "held projection rows"),
                        "held projection"),
        sizeof(double), "held projection bytes");
  }
  require_extent(problem.crop_index.size(), problem.n_omega, "crop_index");
  require_extent(problem.active_mask.size(), problem.n_omega, "active_mask");
  require_extent(problem.norm_mask.size(), problem.n_omega, "norm_mask");
  require_extent(problem.omega_window.size(), problem.n_omega, "omega_window");
  if (!problem.omega.empty())
    require_extent(problem.omega.size(), problem.n_omega, "omega");
  if (!problem.spatial_spectral_weights.empty()) {
    require_extent(problem.spatial_spectral_weights.size(), problem.n_modes,
                   "spatial_spectral_weights");
    require_extent(problem.spatial_on_axis.size(), problem.n_modes, "spatial_on_axis");
    if ((!problem.slab_geometry() && problem.n_nodes != problem.n_modes) ||
        problem.n_pol_transform != 1)
      throw std::invalid_argument(
          "radial statistics require a scalar square transform");
    for (const double weight : problem.spatial_spectral_weights)
      if (!std::isfinite(weight) || weight <= 0.0)
        throw std::invalid_argument(
            "spatial spectral weights must be finite and positive");
  } else if (!problem.spatial_on_axis.empty()) {
    throw std::invalid_argument("spatial on-axis weights require the spectral measure");
  }
  if (problem.slab_geometry()) {
    // solprop's hybrid planar kind: columns (p, k_y), nodes (x, y), a scalar state,
    // whole y rows per node chunk and no embedded coarse rule (solprop's reference
    // scores no quadrature alarm on this kind).
    if (problem.n_x < 1 || problem.n_y < 1 ||
        checked_product(problem.n_p, problem.n_y, "planar columns") !=
            problem.n_modes ||
        checked_product(problem.n_x, problem.n_y, "planar nodes") != problem.n_nodes ||
        problem.n_pol_transform != 1 || problem.n_pol_state != 1 ||
        problem.node_chunk % problem.n_x != 0 || problem.n_phi_terms != 1 ||
        problem.n_prefactor_terms != 1 || !problem.projection_coarse_matrix.empty())
      throw std::invalid_argument(
          "the hybrid planar kind requires matching axes, a scalar state, whole y "
          "rows per node chunk and scalar separable data");
    require_extent(problem.spatial_kappa_sq.size(),
                   checked_product(problem.n_p, problem.n_omega, "planar wall terms"),
                   "spatial_kappa_sq");
    for (const auto value : problem.spatial_kappa_sq)
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument("spatial_kappa_sq must be finite");
  } else if (!problem.spatial_kappa_sq.empty()) {
    throw std::invalid_argument("spatial_kappa_sq belongs to the hybrid planar kind");
  }
  // n_y == 1 is solprop's line uniform in y (grid.LineGrid): one y row carrying
  // k_y = 0, transformed by the same two-axis plan with a unit extent.
  if (problem.free_geometry() &&
      (problem.n_x < 2 || problem.n_y < 1 ||
       checked_product(problem.n_x, problem.n_y, "Cartesian shape") !=
           problem.n_modes ||
       problem.n_nodes != problem.n_modes || problem.n_pol_transform != 1 ||
       problem.n_phi_terms != 1 || problem.n_prefactor_terms != 1))
    throw std::invalid_argument(
        "Cartesian geometry requires matching axes and scalar separable data");
  if (problem.separable_geometry()) {
    if (!std::isfinite(problem.spatial_evanescent_rate) ||
        problem.spatial_evanescent_rate <= 0)
      throw std::invalid_argument(
          "Cartesian evanescent rate must be finite and positive");
    require_extent(problem.spatial_k_squared.size(), problem.n_omega,
                   "spatial_k_squared");
    require_extent(problem.spatial_k_squared_low.size(), problem.n_omega,
                   "spatial_k_squared_low");
    require_extent(problem.spatial_numerator.size(), problem.n_omega,
                   "spatial_numerator");
    require_extent(problem.spatial_clamped.size(), problem.n_omega, "spatial_clamped");
    require_extent(problem.spatial_frame.size(), problem.n_omega, "spatial_frame");
    require_extent(problem.spatial_spectral_rate.size(), problem.n_omega,
                   "spatial_spectral_rate");
    require_extent(problem.spatial_kperp_sq.size(), problem.n_modes,
                   "spatial_kperp_sq");
    require_extent(problem.spatial_kperp_sq_low.size(), problem.n_modes,
                   "spatial_kperp_sq_low");
    require_extent(problem.spatial_absorber_rate.size(), problem.n_modes,
                   "spatial_absorber_rate");
    require_extent(problem.spatial_kperp_rate.size(), problem.n_modes,
                   "spatial_kperp_rate");
    for (const auto value : problem.spatial_k_squared)
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument("spatial_k_squared must be finite");
    for (const auto value : problem.spatial_k_squared_low)
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument("spatial_k_squared_low must be finite");
    for (const auto value : problem.spatial_numerator)
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument("spatial_numerator must be finite");
    for (const auto value : problem.spatial_clamped)
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument("spatial_clamped must be finite");
    for (const auto value : problem.spatial_frame)
      if (!std::isfinite(value))
        throw std::invalid_argument("spatial_frame is invalid");
    for (const auto value : problem.spatial_spectral_rate)
      if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("spatial_spectral_rate is invalid");
    for (const auto value : problem.spatial_kperp_sq)
      if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("spatial_kperp_sq is invalid");
    for (const auto value : problem.spatial_kperp_sq_low)
      if (!std::isfinite(value))
        throw std::invalid_argument("spatial_kperp_sq_low must be finite");
    for (const auto value : problem.spatial_absorber_rate)
      if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("spatial_absorber_rate is invalid");
    for (const auto value : problem.spatial_kperp_rate)
      if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("spatial_kperp_rate is invalid");
    if (problem.free_geometry()) {
      // solprop SEP 23: the stored transverse axes the far-field statistics read.
      require_extent(problem.spatial_k_x.size(), problem.n_x, "spatial_k_x");
      require_extent(problem.spatial_k_y.size(), problem.n_y, "spatial_k_y");
      for (const auto value : problem.spatial_k_x)
        if (!std::isfinite(value))
          throw std::invalid_argument("spatial_k_x must be finite");
      for (const auto value : problem.spatial_k_y)
        if (!std::isfinite(value))
          throw std::invalid_argument("spatial_k_y must be finite");
    }
  } else if (problem.n_y != 0 || problem.n_p != 0) {
    throw std::invalid_argument("Cartesian axes must be supplied together");
  }
  if (!problem.free_geometry() &&
      (!problem.spatial_k_x.empty() || !problem.spatial_k_y.empty() ||
       problem.spatial_far_field_slots != 0))
    throw std::invalid_argument("the far-field statistics belong to free space");
  require_extent(problem.time_window.size(), problem.n_to, "time_window");
  if (problem.plasma && !problem.envelope_plasma) {
    require_extent(problem.time_window_derivative.size(), problem.n_to,
                   "time_window_derivative");
    require_extent(problem.omega.size(), problem.n_omega, "held current omega");
    for (const auto value : problem.time_window_derivative)
      if (!std::isfinite(value))
        throw std::invalid_argument("time_window_derivative is not finite");
    for (std::size_t index = 0; index < problem.n_omega; ++index)
      if (problem.active_mask[index] &&
          (!(problem.omega[index] > 0.0) || !std::isfinite(problem.omega[index])))
        throw std::invalid_argument(
            "held plasma current requires positive active omega");
  }
  require_extent(problem.carrier.size(), problem.n_to, "carrier");
  require_extent(problem.synthesis_matrix.size(), transform, "synthesis_matrix");
  require_extent(problem.projection_matrix.size(), transform, "projection_matrix");
  if (!problem.projection_coarse_matrix.empty())
    require_extent(problem.projection_coarse_matrix.size(), transform,
                   "projection_coarse_matrix");
  // A z table's terms are slots its evaluation fills, so nothing crosses for them.
  if ((problem.phi_table || problem.prefactor_table) && problem.separable_geometry())
    throw std::invalid_argument("a separable operator takes no z table");
  require_extent(problem.phi_terms.size(),
                 problem.separable_geometry() || problem.phi_table
                     ? 0
                     : checked_product(problem.n_phi_terms, field, "phi_terms"),
                 "phi_terms");
  require_extent(
      problem.prefactor_terms.size(),
      problem.separable_geometry() || problem.prefactor_table
          ? 0
          : checked_product(problem.n_prefactor_terms, field, "prefactor_terms"),
      "prefactor_terms");
  require_extent(problem.tableau_matrix.size(),
                 checked_product(problem.n_stages, problem.n_stages, "tableau"),
                 "tableau_matrix");
  require_extent(problem.tableau_nodes.size(), problem.n_stages, "tableau_nodes");
  require_extent(problem.solution_weights.size(), problem.n_stages, "solution_weights");
  require_extent(problem.error_weights.size(), problem.n_stages, "error_weights");
  if (problem.in_stage_absorber)
    require_extent(problem.absorber_fine.size(), problem.n_to, "absorber_fine");
  if (!problem.spatial_absorber_matrix.empty()) {
    require_extent(
        problem.spatial_absorber_matrix.size(),
        checked_product(problem.n_modes, problem.n_modes, "spatial absorber"),
        "spatial_absorber_matrix");
    if (!problem.in_stage_absorber || problem.n_pol_transform != 1 ||
        problem.n_nodes != problem.n_modes)
      throw std::invalid_argument("spatial absorber requires scalar radial geometry");
  }
  if (problem.split_step_absorber)
    require_extent(problem.absorber_coarse.size(), problem.n_t, "absorber_coarse");
  const std::size_t fine_size =
      problem.family == FieldFamily::field_real ? problem.n_to / 2 + 1 : problem.n_to;
  if (problem.envelope_plasma && problem.family != FieldFamily::envelope)
    throw std::invalid_argument("envelope plasma requires an envelope field family");
  std::vector<bool> seen(fine_size, false);
  for (const auto index : problem.crop_index) {
    if (index >= fine_size)
      throw std::invalid_argument("crop_index lies outside the fine spectrum");
    if (seen[index])
      throw std::invalid_argument("crop_index contains a duplicate fine-spectrum bin");
    seen[index] = true;
  }
  if (problem.plasma && problem.n_channels == 0)
    throw std::invalid_argument("plasma requires at least one channel");
  if (problem.n_channels > 1 || problem.envelope_plasma) {
    if (!problem.plasma)
      throw std::invalid_argument("channels require plasma");
    require_extent(problem.channel_parameters.size(),
                   checked_product(problem.n_channels, 9, "channel parameters"),
                   "channel_parameters");
    if (!std::all_of(problem.channel_parameters.begin(),
                     problem.channel_parameters.end(),
                     [](double value) { return std::isfinite(value); }) ||
        !std::all_of(problem.channel_coefficients.begin(),
                     problem.channel_coefficients.end(),
                     [](double value) { return std::isfinite(value); }))
      throw std::invalid_argument("channel parameters and coefficients must be finite");
    std::size_t expected_offset = 0;
    for (std::size_t channel = 0; channel < problem.n_channels; ++channel) {
      const double* row = problem.channel_parameters.data() + 9 * channel;
      if (!(row[1] > 0.0 && row[2] > 0.0 && row[3] > row[2] && row[4] >= 0.0 &&
            row[5] >= 1.0 && row[5] == std::floor(row[5]) &&
            row[6] == static_cast<double>(expected_offset) && row[7] > 0.0 &&
            row[8] > 0.0) ||
          row[5] > static_cast<double>(problem.channel_coefficients.size() / 4))
        throw std::invalid_argument(
            "channel constants and coefficient ranges must be positive and consistent");
      expected_offset = checked_sum(
          expected_offset,
          checked_product(4, static_cast<std::size_t>(row[5]), "channel table"),
          "channel table offsets");
    }
    require_extent(problem.channel_coefficients.size(), expected_offset,
                   "channel_coefficients");
  }
  if (problem.raman && problem.n_raman_components == 0)
    throw std::invalid_argument("Raman requires at least one component");
  if (problem.n_raman_components > 1) {
    if (!problem.raman)
      throw std::invalid_argument("Raman components require Raman");
    require_extent(problem.raman_row_component.size(), problem.n_raman_terms,
                   "raman_row_component");
    require_extent(problem.raman_component_fraction.size(), problem.n_raman_components,
                   "raman_component_fraction");
    for (double value : problem.raman_row_component)
      if (!std::isfinite(value) || value < 0.0 ||
          value >= static_cast<double>(problem.n_raman_components) ||
          value != std::floor(value))
        throw std::invalid_argument("Raman row component index is invalid");
    for (double value : problem.raman_component_fraction)
      if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(
            "Raman component fractions must be finite and positive");
  }
  if (problem.plasma) {
    if (!(problem.dt_fine > 0.0) || !(problem.d_log_field > 0.0) ||
        !(problem.field_min > 0.0) || !(problem.field_max > problem.field_min) ||
        !(problem.charge_mass_ratio > 0.0) || !(problem.ionisation_potential > 0.0) ||
        problem.rate_intervals == 0)
      throw std::invalid_argument("plasma constants must be positive and consistent");
    require_extent(problem.rate_coefficients.size(),
                   checked_product(4, problem.rate_intervals, "rate coefficients"),
                   "rate_coefficients");
  }
  // solprop SEP 18's running-integral rule. Translation certified the taps; only their
  // extent, finiteness, the trapezoid's identity tap and the route are checked here.
  if (problem.plasma_taps.empty() || problem.plasma_taps.size() > plasma_max_taps ||
      !std::all_of(problem.plasma_taps.begin(), problem.plasma_taps.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("plasma_taps must hold one to nine finite taps");
  if (!problem.plasma_band &&
      (problem.plasma_taps.size() != 1 || problem.plasma_taps[0] != 1.0))
    throw std::invalid_argument("the trapezoid rule is the one identity tap");
  if (problem.plasma_band) {
    if (!problem.plasma || problem.envelope_plasma ||
        problem.family == FieldFamily::envelope)
      throw std::invalid_argument(
          "the band integration rule needs a field-grid plasma column");
    if (problem.n_to <= problem.plasma_half_width())
      throw std::invalid_argument(
          "the band integration rule's filter needs more than K samples");
  }
  if (problem.envelope_plasma) {
    if (!problem.plasma || problem.family != FieldFamily::envelope ||
        problem.harmonic_factor != 0.0 || problem.raman_thg || problem.stash_plasma)
      throw std::invalid_argument("envelope plasma requires separate envelope currents "
                                  "without THG or field stashing");
    const auto rule_size =
        checked_product(problem.n_channels, 4 * 512, "envelope rules");
    require_extent(problem.envelope_rule_nodes.size(), rule_size,
                   "envelope_rule_nodes");
    require_extent(problem.envelope_rule_weights.size(), rule_size,
                   "envelope_rule_weights");
    for (std::size_t channel = 0; channel < problem.n_channels; ++channel)
      for (std::size_t level = 0; level < 4; ++level) {
        const auto order = std::size_t{64} << level;
        double sum = 0.0, previous = -1.0;
        for (std::size_t index = 0; index < 512; ++index) {
          const auto offset = (channel * 4 + level) * 512 + index;
          const auto node = problem.envelope_rule_nodes[offset];
          const auto weight = problem.envelope_rule_weights[offset];
          if (!std::isfinite(node) || !std::isfinite(weight) ||
              (index < order ? !(node > previous && node < 1.0 && weight > 0.0)
                             : node != 0.0 || weight != 0.0))
            throw std::invalid_argument(
                "envelope quadrature nodes or weights are invalid");
          if (index < order) {
            previous = node;
            sum += weight;
          }
        }
        if (std::abs(sum - 2.0) >
            8 * static_cast<double>(order) * std::numeric_limits<double>::epsilon())
          throw std::invalid_argument(
              "envelope quadrature weights must integrate a constant");
      }
    require_extent(problem.envelope_drude_factor.size(), problem.n_omega,
                   "envelope_drude_factor");
    require_extent(problem.envelope_loss_factor.size(), problem.n_omega,
                   "envelope_loss_factor");
    for (std::size_t index = 0; index < problem.n_omega; ++index) {
      const auto drude = problem.envelope_drude_factor[index];
      const auto loss = problem.envelope_loss_factor[index];
      const bool active = problem.active_mask[index] != 0 && problem.omega[index] > 0.0;
      if (!std::isfinite(drude) || !std::isfinite(loss.real()) ||
          !std::isfinite(loss.imag()) || loss.real() != 0.0 ||
          (active ? !(drude < 0.0 && loss.imag() < 0.0)
                  : drude != 0.0 || loss.imag() != 0.0))
        throw std::invalid_argument("envelope current factors require negative active "
                                    "factors and zero inactive bins");
    }
  }
  if (problem.raman) {
    if (effective_node_pol != 1 && !problem.raman_tensor_capable)
      throw std::invalid_argument(
          "vector Raman requires tensor-capable terms; found scalar_only");
    if (problem.n_raman_terms == 0)
      throw std::invalid_argument("a Raman problem must contain at least one term");
    require_extent(problem.raman_tensor_weights.size(),
                   checked_product(problem.n_raman_terms, 2, "Raman tensor weights"),
                   "raman_tensor_weights");
    for (std::size_t term = 0; term < problem.n_raman_terms; ++term) {
      const double trace = problem.raman_tensor_weights[2 * term];
      const double traceless = problem.raman_tensor_weights[2 * term + 1];
      if (!((trace == 1.0 && traceless == 0.0) || (trace == 0.25 && traceless == 0.75)))
        throw std::invalid_argument(
            "Raman tensor weights must be isotropic or rank two");
    }
    const auto doubled = checked_product(2, problem.n_to, "Raman doubled grid");
    const auto raman_rows =
        checked_product(problem.node_chunk, problem.raman_drives(), "Raman drive rows");
    require_allocation(checked_product(raman_rows, doubled, "Raman drive"),
                       sizeof(double), "Raman drive bytes");
    require_extent(problem.raman_undamped.size(),
                   checked_product(problem.n_raman_terms, doubled, "raman_undamped"),
                   "raman_undamped");
    if (!(problem.raman_fraction > 0.0) || !std::isfinite(problem.raman_fraction) ||
        !(problem.dt_fine > 0.0) || !std::isfinite(problem.dt_fine) ||
        !(problem.raman_epsilon0 > 0.0) || !std::isfinite(problem.raman_epsilon0))
      throw std::invalid_argument(
          "Raman fraction, dt_fine and epsilon0 must be positive and finite");
    if (!std::all_of(problem.raman_undamped.begin(), problem.raman_undamped.end(),
                     [](const double value) { return std::isfinite(value); }))
      throw std::invalid_argument("raman_undamped must contain only finite values");
  } else if (problem.raman_thg || problem.n_raman_terms != 0 ||
             !problem.raman_undamped.empty() || !problem.raman_tensor_weights.empty() ||
             problem.raman_tensor_capable) {
    throw std::invalid_argument("Raman data were supplied while Raman is disabled");
  }
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
int host_reduction_team_size(const ExecutionSpace& execution, const std::size_t columns,
                             const std::size_t bins) {
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

// The update and its three binary64 norms. On a device each column is `slots` teams,
// each over its own bins, writing its partial sums at offset `slot` from the column's
// partial pointers; the host reduces them.
template <class Columns>
void accept_update(const ExecutionSpace& execution, const std::size_t count,
                   const std::size_t n, const double dz, const Columns& columns,
                   const char* label, const std::size_t slots) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  using ReductionPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using ReductionMember = ReductionPolicy::member_type;
  const ReductionPolicy reduction_policy(execution, static_cast<int>(count * slots),
                                         SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
  Kokkos::parallel_for(
      label, reduction_policy, KOKKOS_LAMBDA(const ReductionMember& team) {
        const auto league = static_cast<std::size_t>(team.league_rank());
        const auto column = league / slots;
        const auto slot = league % slots;
        std::size_t first = 0;
        std::size_t end = 0;
        slot_range(n, slots, slot, first, end);
        const auto values = columns(column);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, static_cast<int>(first),
                                                     static_cast<int>(end)),
                             [&](const int index_value) {
                               const auto index = static_cast<std::size_t>(index_value);
                               const Complex candidate =
                                   values.whole[index] * (values.carried[index] +
                                                          dz * values.solution[index]);
                               values.advanced[index] = candidate;
                               values.residual[index] =
                                   values.whole[index] * (dz * values.error[index]);
                             });
        team.team_barrier();
        double old_sum = 0.0;
        double new_sum = 0.0;
        double error_sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(first),
                                    static_cast<int>(end)),
            [&](const int index_value, double& update) {
              const auto index = static_cast<std::size_t>(index_value);
              if (values.in_norm[index] != 0U)
                update += squared_abs(values.current[index]);
            },
            old_sum);
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(first),
                                    static_cast<int>(end)),
            [&](const int index_value, double& update) {
              const auto index = static_cast<std::size_t>(index_value);
              if (values.in_norm[index] != 0U)
                update += squared_abs(values.advanced[index]);
            },
            new_sum);
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(first),
                                    static_cast<int>(end)),
            [&](const int index_value, double& update) {
              const auto index = static_cast<std::size_t>(index_value);
              if (values.in_norm[index] != 0U)
                update += squared_abs(values.residual[index]);
            },
            error_sum);
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          values.old_partial[slot] = old_sum;
          values.new_partial[slot] = new_sum;
          values.error_partial[slot] = error_sum;
        });
      });
#else
  // One team per column, its threads splitting the bins when the columns are fewer
  // than the threads, and the update and the three norms in one pass; the sums
  // associate as the team splits them (solprop SEP 25). The host keeps one slot.
  static_cast<void>(slots);
  using ReductionPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using ReductionMember = ReductionPolicy::member_type;
  const ReductionPolicy reduction_policy(execution, static_cast<int>(count),
                                         host_reduction_team_size(execution, count, n));
  Kokkos::parallel_for(
      label, reduction_policy, KOKKOS_LAMBDA(const ReductionMember& team) {
        const auto column = static_cast<std::size_t>(team.league_rank());
        const auto values = columns(column);
        host::NormSums sums;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, n),
            [&](const std::size_t index, host::NormSums& update) {
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
            Kokkos::Sum<host::NormSums>(sums));
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          *values.old_partial = sums.old_sum;
          *values.new_partial = sums.new_sum;
          *values.error_partial = sums.error_sum;
        });
      });
#endif
}

// The bins a host work item of the stage's entry kernel takes at a time: every bin's
// phase sum is the same sequence of rounded products and additions in term order as
// a per-bin loop, laid out so that the bins vectorise.
constexpr std::size_t stage_block = 64;

// The pointwise bodies of a stage over the coarse spectrum: its entry (the phase
// increment, the combination with the earlier stages, the lab frame, the noise and the
// pad) and its finish (prefactor, window, absorber, return to the interaction picture
// and accumulation). The entry kernel runs the finish of the stage before it in the
// same traversal when nothing lies between them (solprop SEP 25 section 6.7): both
// are pointwise over the bins, and the combination reads only its own bin's stages.
// How a stage's entry forms its propagator pair (`StageTerms::combine`).
enum class EntryPhase { general, lossless, repeated };

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

  // The combination of stage `stage` at one bin, from the stage's phase increment.
  // The propagator `exp(phase)` and its inverse `exp(-phase)`, which the finish
  // applies, share their cosine and sine (solprop SEP 25 section 6.7): the angle's
  // parts are even and odd, so `exp(-phase)` is `e^{-a} (cos b, -sin b)` with its own
  // magnitude, never a reciprocal of the forward value, and is the finish's former
  // `exponential(-phase)` to the bit. A coupled flow's residual reads the phase.
  // A lossless problem (`EntryPhase::lossless`: every Phi term's real part exactly
  // zero, so the phase's is a signed zero) takes both magnitudes as `exp(+-0)`,
  // exactly one, without the calls. A lossless stage whose phase weights equal the
  // stage before's (`EntryPhase::repeated`) has that stage's inverse still in place,
  // and its propagator is that inverse's conjugate, `(cos b, -(-sin b))`, to the bit.
  // Returns the propagator.
  template <EntryPhase Mode = EntryPhase::general>
  KOKKOS_INLINE_FUNCTION Complex combine(const std::size_t stage, const double dz,
                                         const double solution_weight,
                                         const double error_weight,
                                         const std::size_t column,
                                         const std::size_t index,
                                         const Complex phase) const {
    const Complex phase_at = add_noise ? noise_phase(stage, column, index) : Complex{};
    Complex propagator;
    if constexpr (Mode == EntryPhase::repeated) {
      const Complex inverse = stage_inverse(column, index);
      propagator = Complex(inverse.real(), -inverse.imag());
    } else {
      constexpr bool lossless = Mode == EntryPhase::lossless;
      double cosine = 0.0;
      double sine = 0.0;
      cosine_sine(phase.imag(), cosine, sine);
      const double magnitude = lossless ? 1.0 : Kokkos::exp(phase.real());
      const double inverse_magnitude = lossless ? 1.0 : Kokkos::exp(-phase.real());
      propagator = Complex(magnitude * cosine, magnitude * sine);
      stage_inverse(column, index) =
          Complex(inverse_magnitude * cosine, -(inverse_magnitude * sine));
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
      value += noise(column, index) *
               Complex(Kokkos::cos(phase_at.imag()), Kokkos::sin(phase_at.imag()));
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
};

} // namespace

class GeneralWorkspace {
public:
  explicit GeneralWorkspace(Problem problem) : problem_(std::move(problem)) {
    require_general_problem(problem_);
    execution_space_ =
        kernel_pool_.get(CpuOperation::stage, problem_.n_columns() * problem_.n_omega);
    const auto nodes = problem_.node_chunk;
    const auto response_elements =
        nodes * problem_.effective_node_pol() * problem_.n_to;
    response_execution_space_ =
        kernel_pool_.get(CpuOperation::response, response_elements, problem_.n_to);
    const auto tail = problem_.n_nodes % nodes;
    response_tail_execution_space_ = kernel_pool_.get(
        CpuOperation::response, tail * problem_.effective_node_pol() * problem_.n_to,
        problem_.n_to);
    const auto dense_elements = response_elements * problem_.n_modes;
    const auto tail_dense_elements =
        tail * problem_.effective_node_pol() * problem_.n_to * problem_.n_modes;
    transform_execution_space_ = kernel_pool_.get(CpuOperation::dense, dense_elements);
    transform_tail_execution_space_ =
        kernel_pool_.get(CpuOperation::dense, tail_dense_elements);
    temporal_execution_space_ =
        kernel_pool_.get(CpuOperation::copy, problem_.n_columns() * problem_.n_to);
    // Packing copies only matrix entries, and splitting copies only output samples;
    // neither repeats the dense transform's inner product.
    const auto matrix_elements = nodes * problem_.n_pol_transform * problem_.n_modes;
    const auto tail_matrix_elements =
        tail * problem_.n_pol_transform * problem_.n_modes;
    matrix_copy_space_ = kernel_pool_.get(CpuOperation::copy, matrix_elements);
    tail_matrix_copy_space_ =
        kernel_pool_.get(CpuOperation::copy, tail_matrix_elements);
    stacked_matrix_copy_space_ =
        kernel_pool_.get(CpuOperation::copy, 2 * matrix_elements);
    tail_stacked_matrix_copy_space_ =
        kernel_pool_.get(CpuOperation::copy, 2 * tail_matrix_elements);
    node_spectral_space_ = kernel_pool_.get(
        CpuOperation::copy, nodes * problem_.effective_node_pol() * problem_.n_omega);
    flow_execution_space_ =
        kernel_pool_.get(CpuOperation::flow, problem_.flow_active * problem_.n_modes *
                                                 problem_.n_modes * problem_.n_modes);
    if (!problem_.spatial_absorber_matrix.empty())
      absorber_execution_space_ = kernel_pool_.get(
          CpuOperation::dense, problem_.n_columns() * problem_.n_to * problem_.n_modes);
    allocate();
    upload();
    const auto heap_before = heap_bytes();
    const auto device_before = device_free_bytes();
    const auto started = std::chrono::steady_clock::now();
    plan();
    execution_space_.fence("solprop_native planned workspace accounting");
    const auto heap_after = heap_bytes();
    const auto device_after = device_free_bytes();
    plan_heap_growth_ = heap_after > heap_before ? heap_after - heap_before : 0;
    plan_device_growth_ =
        device_before > device_after ? device_before - device_after : 0;
    planning_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    execution_space_.fence("solprop_native construction");
  }

  void set_initial(const std::vector<Complex>& field,
                   const std::vector<Complex>& noise);
  std::vector<double> trial_step(double dz, const TrialScalars& scalars);
  void prepare_trial(double dz, const TrialScalars& scalars);
  std::vector<Complex> adams_source(const std::vector<Complex>& input,
                                    const TrialScalars& scalars);
  void adopt(const std::vector<Complex>& field, double dz);
  double low_storage_step(double dz, const TrialScalars& scalars, bool rebuild_table);
  void hold_plane();
  std::vector<Complex> held_to_host() const;
  void low_storage_flows_formed(bool rebuild_table);
  void low_storage_separable_pass(std::size_t pass, double dz);
  void low_storage_separable_input(std::size_t pass);
  void low_storage_separable_fold(std::size_t pass, int mode, int term, double dz);
  void low_storage_modal_pass(std::size_t pass, double dz);
  void low_storage_modal_input(std::size_t pass);
  void adams_allocate(std::size_t slots, bool compensated, bool interpolate);
  void adams_compensation_start(std::size_t slot, const std::vector<double>& rate);
  void adams_start(std::size_t slot, const TrialScalars& scalars);
  double adams_magnitude();
  // The controller's norm scale on the device: rtol * w_i * |u| + atol over the
  // in-band magnitude, formed each trial without the magnitude crossing to the host.
  void adams_scale(double rtol, double atol, const std::vector<double>& weights);
  void adams_upload_divisor(const std::vector<double>& divisor);
  std::vector<double> adams_fused_trial(double dz, const AdamsWeights& weights,
                                        const std::vector<double>& divisor,
                                        const TrialScalars& scalars);
  void adams_fused_accept(double dz, const AdamsWeights& keep, std::size_t free,
                          const AdamsWeights& dense);
  void adams_hold();
  void adams_restore();
  void adams_observe(const AdamsWeights& weights, bool hermite,
                     const TrialScalars& scalars);
  std::vector<double>
  adams_exponential_trial(double dz, const ExponentialWeights& weights,
                          const std::vector<double>& divisor, const TrialScalars& step,
                          const TrialScalars& sample, const std::vector<double>& rate);
  void adams_exponential_accept(double dz, std::size_t free, const AdamsWeights& dense);
  void adams_exponential_observe(double xi, double dz,
                                 const ExponentialWeights& weights);
  std::vector<std::string> adams_row_names() const;
  std::array<std::size_t, 3> adams_row_extents(const std::string& name) const;
  std::vector<Complex> adams_rows_to_host(const std::string& name) const;
  void adams_rows_to_host(const std::string& name, std::complex<double>* destination,
                          std::size_t size) const;
  void adams_rows_from_host(const std::string& name,
                            const std::vector<Complex>& values);
  std::vector<double> collect_trial();
  void table_upload(TableKind kind, std::vector<double> edges, double lower,
                    double upper, const std::complex<double>* coefficients,
                    std::size_t n_intervals, std::size_t n_terms, std::size_t n_values);
  void table_evaluate(const std::vector<std::size_t>& slots,
                      const std::vector<double>& positions);
  void table_rate(std::size_t slot, double z, bool before);
  std::vector<Complex> table_terms_to_host(TableKind kind, std::size_t slot) const;
  std::vector<Complex> phase_sum_to_host(const std::vector<double>& weights);
  std::vector<double> weighted_error(const std::vector<double>& scale);
  void set_flow_actions(const std::vector<Complex>& forward,
                        const std::vector<Complex>& inverse,
                        const std::vector<unsigned char>& identity);
  std::vector<double> form_flow_actions(std::size_t section, double dz,
                                        const std::vector<double>& fractions);
  std::vector<Complex> flow_actions_to_host(bool inverse) const;
  void prepare_raman(const std::vector<double>& rates,
                     const std::vector<double>& densities);
  std::vector<double> spatial_statistics(std::uint64_t flags, double intensity_factor,
                                         double z);
  std::vector<double> aperture_statistics(std::uint64_t flags,
                                          const std::vector<double>& apertures,
                                          int profile_code, int exponent,
                                          const std::vector<double>& regions,
                                          double intensity_factor);
  std::vector<double> statistics(std::uint64_t flags,
                                 const std::vector<double>& windows,
                                 const std::vector<double>& on_axis, double density,
                                 double taper, double preionisation,
                                 double intensity_factor,
                                 const std::vector<double>& channel_preionisation,
                                 const std::vector<double>& power_weight);
  void finish(double dz);
  void promote_fsal();
  void clear_fsal();
  // The boundary monitor found the temporal collar quiet (solprop SEP 21).
  void set_absorber_quiet(const bool quiet) noexcept { absorber_quiet_ = quiet; }
  void bind_monitor(const std::vector<double>& weights,
                    const std::vector<double>& column_weights);
  void unbind_monitor() noexcept { monitor_bound_ = false; }
  std::vector<double> monitor_reading();
  std::vector<Complex> field_to_host() const;
  std::vector<Complex> stage_to_host(std::size_t stage) const;
  std::vector<Complex> projected_to_host(bool coarse);
  std::map<std::string, bool> dispersive_path() const;
  std::map<std::string, std::size_t> memory_ledger() const;
  CpuSizing execution_sizing() const {
    auto result = kernel_pool_.sizing();
    result.merge(fft_pool_.sizing());
    result.merge(raman_pool_.sizing());
    result.merge(raman_tail_pool_.sizing());
    if (general_)
      result.merge(general_->execution_sizing());
    return result;
  }
  std::map<std::string, std::size_t> base_memory_ledger() const;
  double planning_seconds() const noexcept { return planning_seconds_; }
  double raman_planning_seconds() const noexcept { return raman_planning_seconds_; }
  std::string raman_fft_provider() const {
    return raman_drive_forward_plan_ ? raman_drive_forward_plan_->provider() : "none";
  }
  std::map<std::string, int> raman_fft_tasks() const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    const int full = 0, tail = 0, kernel = 0;
#else
    const int full = raman_execution_space_.concurrency(),
              tail = raman_tail_execution_space_.concurrency(),
              kernel = raman_kernel_execution_space_.concurrency();
#endif
    return {
        {"full",
         raman_drive_forward_plan_ ? raman_drive_forward_plan_->tasks(full) : 0},
        {"tail", raman_tail_forward_plan_ ? raman_tail_forward_plan_->tasks(tail) : 0},
        {"kernel",
         raman_kernel_forward_plan_ ? raman_kernel_forward_plan_->tasks(kernel) : 0}};
  }
  std::size_t raman_kernel_transforms() const noexcept {
    return raman_kernel_transforms_;
  }
  // The last accepted step's score, read from the device when asked for.
  double quadrature_error() const;
  // The largest score since the last call, then reset (NaN when none was scored).
  double quadrature_peak();
  std::string synthesis_provider() const;
  std::string projection_provider() const;
  static std::string gemm_shape_policy();
  std::size_t response_team_size() const noexcept;
  bool fsal_ready() const noexcept { return has_recycled_; }
  std::size_t rk_source_evaluations() const noexcept { return rk_source_evaluations_; }
  bool closed() const noexcept { return closed_; }
  void close() noexcept;

  // NVCC extended device lambdas require their enclosing member to be public.
  // GeneralWorkspace is translation-unit-local, so this does not widen the API.
  std::size_t fine_spectrum_size() const {
    return problem_.family == FieldFamily::field_real ? problem_.n_to / 2 + 1
                                                      : problem_.n_to;
  }
  std::size_t projected_spectrum_size() const {
    return problem_.family == FieldFamily::envelope ? problem_.n_to
                                                    : problem_.n_to / 2 + 1;
  }
  void require_open() const {
    if (closed_)
      throw std::logic_error("native state is closed");
  }
  void require_trial_scalars(const TrialScalars& scalars) const;
  void require_adams() const;
  void adams_phase();
  void adams_evaluate(const TrialScalars& scalars);
  void adams_rate(const std::vector<double>& rate, const ComplexView2& out);
  void allocate();
  void upload();
  void upload_trial(const TrialScalars& scalars);
  void plan();
  void reuse_fsal();
  void evaluate_stage(std::size_t stage, double dz);
  void evaluate_stage_source(std::size_t stage, double dz);
  bool modal_core(std::size_t stage);
  void general_core(std::size_t stage);
  StageTerms stage_terms(std::size_t stage) const;
  std::vector<std::pair<std::size_t, std::size_t>> fine_gap_runs() const;
  void stage_entry(std::size_t stage, double dz);
  template <EntryPhase Mode> void stage_entry_with(std::size_t stage, double dz);
  void finish_or_defer(std::size_t stage);
  // Free space and the hybrid planar kind (`Problem::separable_geometry`).
  TransverseLayout transverse_layout() const noexcept {
    const std::size_t plane = transverse_.extent(1) * transverse_.extent(2);
    return problem_.slab_geometry() ? TransverseLayout{problem_.n_p, true, plane}
                                    : TransverseLayout{problem_.n_x, false, plane};
  }
  // The unscaled transverse DFT's sample count: the plane's, or the slab's y axis.
  double inverse_transverse_count() const noexcept {
    return 1.0 / static_cast<double>(problem_.slab_geometry() ? problem_.n_y
                                                              : problem_.n_modes);
  }
  std::size_t transverse_units() const noexcept {
    return problem_.slab_geometry() ? problem_.n_y : problem_.n_nodes;
  }
  std::size_t transverse_units_per_chunk() const noexcept {
    return problem_.slab_geometry() ? problem_.node_chunk / problem_.n_x
                                    : problem_.node_chunk;
  }
  TransverseChunk transverse_chunk(std::size_t first_unit) const noexcept;
  void transverse_forward();
  void transverse_inverse();
  void transverse_inverse(const ComplexView3& scratch);
  void flux_at(double z);
  void require_flux_closure();
  void fill_statistics_closure();
  void evaluate_separable_stage(std::size_t stage, double dz);
  void seed_free_wisdom();
  void seed_modal_wisdom();
  void seed_stream_wisdom();
  void separable_core(std::size_t stage, bool material);
  void separable_stage_input(std::size_t stage, double dz);
  void separable_load_current();
  void separable_pack_chunk(const TransverseChunk& chunk, bool material);
  void separable_nodes(const TransverseChunk& chunk);
  void slab_project(const TransverseChunk& chunk, const EnvelopeStream* stream);
  void separable_finish_transform(std::size_t stage, bool material);
  void separable_time_statistics(std::size_t axis_peak, std::size_t peak,
                                 std::size_t fluence, double intensity_factor);
  void scale_transform(std::size_t stage);
  void synthesise(std::size_t first_node, std::size_t node_count);
  void response(std::size_t stage, std::size_t first_node, std::size_t node_count);
  void prepare_raman_stage(std::size_t stage, const double* rates,
                           const double* densities = nullptr);
  void convolve_raman(std::size_t stage, std::size_t node_count);
  void finish_raman(std::size_t stage, std::size_t node_count);
  template <FieldFamily Family, std::size_t Components>
  void response_plasma(std::size_t stage, std::size_t first_node,
                       std::size_t node_count);
  template <FieldFamily Family, std::size_t Components, bool Plasma>
  void response_nonlinear(std::size_t stage, std::size_t first_node,
                          std::size_t node_count);
  template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear>
  void response_stash(std::size_t stage, std::size_t first_node,
                      std::size_t node_count);
  template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
            bool Stash>
  void response_harmonic(std::size_t stage, std::size_t first_node,
                         std::size_t node_count);
  template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
            bool Stash, bool Harmonic>
  void response_raman(std::size_t stage, std::size_t first_node,
                      std::size_t node_count);
  template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
            bool Stash, bool Harmonic, bool Raman, bool RamanThg, bool Multiple = false,
            bool Folded = false>
  void response_impl(std::size_t stage, std::size_t first_node, std::size_t node_count);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // The device's response (`device_response`) and its trapezoid plasma's two-level
  // scans.
  template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
            bool Stash, bool Harmonic, bool Raman, bool RamanThg>
  void device_response(std::size_t stage, std::size_t first_node,
                       std::size_t node_count);
  template <FieldFamily Family, std::size_t Components, bool Stash>
  void device_trapezoid_plasma(std::size_t stage, std::size_t first_node,
                               std::size_t node_count);
#endif
  // solprop SEP 18's band rule as a team column (include/solprop_native/plasma.hpp):
  // filter, then scan, for the population and free current, in whole-row passes.
  // It writes the current separately from the collared non-plasma response.
  template <FieldFamily Family, std::size_t Components, bool Multiple,
            bool Folded = false>
  void team_band_plasma_column(std::size_t stage, std::size_t first_node,
                               std::size_t node_count);
  // The same column in tiles (solprop SEP 25 section 6.7.6, the device form): each
  // running integral is a local kernel, one team per (row, tile) filtering and scanning
  // its tile from zero, and a finish kernel adding the earlier tiles' totals and
  // forming the pass's product.
  template <FieldFamily Family, std::size_t Components, bool Multiple,
            bool Folded = false>
  void tiled_band_plasma_column(std::size_t stage, std::size_t first_node,
                                std::size_t node_count);
  // The tiled column's teams: `rows` x tiles of them, as wide as the build's response
  // team on a device (or `band_tile_threads_` on the host) and the kernel allows, with
  // the tile's scratch when `scratch`.
  template <class Functor>
  [[nodiscard]] Kokkos::TeamPolicy<ExecutionSpace>
  tiled_band_policy(std::size_t rows, bool scratch, const Functor& functor) const;
  void tiled_band_local(const RealView2& source, std::size_t rows, const char* label);
  // The band rule's first stage alone at every node of a chunk: the floored final
  // exposure and largest rate per channel, and with `Diagnose` the six per-node parts
  // of the run diagnostics; team scans with team reductions.
  template <bool Diagnose>
  void team_band_statistics(std::size_t first_node, std::size_t node_count,
                            const RealView2& output, double preionisation);
  // The band column's teams: on a device one per node, as wide as the build's team and
  // the kernel's resources allow; on the host the threads shared between the nodes.
  // Each team walks the nodes `slot, slot + league, ...` with its slot's rows.
  template <class Functor>
  [[nodiscard]] Kokkos::TeamPolicy<ExecutionSpace>
  band_team_policy(std::size_t node_count, const Functor& functor) const;
  [[nodiscard]] std::size_t band_league(std::size_t node_count) const;
  [[nodiscard]] int band_host_team_size(std::size_t node_count) const;
  void project(std::size_t first_node, std::size_t node_count, bool first_chunk,
               const EnvelopeStream* stream = nullptr,
               const FieldCurrentStream* field_stream = nullptr);
  bool held_plasma() const noexcept {
    return problem_.plasma && !problem_.envelope_plasma;
  }
  bool held_radial() const noexcept {
    return held_plasma() && !problem_.separable_geometry() &&
           !problem_.spatial_spectral_weights.empty();
  }
  void held_response(std::size_t stage, std::size_t first_node, std::size_t node_count,
                     const TransverseChunk* chunk = nullptr);
  void held_transform(bool coarse_only = false);
  void finish_quadrature();
  bool deferred_quadrature_eligible() const noexcept {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return false;
#else
    // The validated seven-stage FSAL tableau identifies the ordinary DOPRI5
    // route without another seam field. Other consumers retain eager spectra.
    return problem_.first_same_as_last && problem_.n_stages == 7 &&
           !problem_.low_storage && !problem_.separable_geometry() && !general_ &&
           !flow_ && !dispersive_ && !problem_.envelope_plasma && !held_radial() &&
           !problem_.projection_coarse_matrix.empty();
#endif
  }
  void held_radial_projection(std::size_t first_node, std::size_t node_count);
  void envelope_cycle(std::size_t channel, std::size_t node_count);
  void envelope_population(std::size_t channel, std::size_t stage,
                           std::size_t first_node, std::size_t node_count,
                           const RealView2& statistics);
  void envelope_response(std::size_t stage, std::size_t first_node,
                         std::size_t node_count,
                         const TransverseChunk* chunk = nullptr);
  void envelope_transform();
  void finish_stage(std::size_t stage);
  void score_quadrature();
  void fill_statistics_time(bool windowed, double lower, double upper,
                            bool weighted = false);
  void fill_statistics_pair(bool windowed, double lower, double upper);
  void form_statistics_power();
  void form_statistics_total_power();
  void scan_plasma_statistics(std::size_t first_node, std::size_t node_count,
                              const RealView2& output, bool diagnose = false,
                              double preionisation = 0.0);
  void form_absorber();
  void split_step(double dz, const ComplexView2& values);
  void split_columns(double dz, std::size_t active_rows = 0, double parseval = 0.0);
  void add_split_removal(double dz, std::size_t active_rows, double parseval);
  void separable_to_time(const TransverseChunk& chunk);
  void separable_to_frequency(const TransverseChunk& chunk);
  void separable_split_step(double dz, const ComplexView2& values);
  void mask(const ComplexView2& values);
  bool use_packed_radial_blas() const noexcept;
  void packed_radial_gemm(const RealView2& matrix, const ComplexView2& source,
                          const ComplexView2& destination);
  bool use_blas_for_synthesis() const noexcept;
  bool use_interleaved_synthesis() const noexcept;
  bool use_interleaved_projection() const noexcept;
  bool hybrid_shape_eligible() const noexcept;
  bool use_blas_for_projection() const noexcept;
  bool use_chunked_blas_for_synthesis() const noexcept;
  bool folds_single_node() const noexcept;
  bool use_chunked_blas_for_projection() const noexcept;
  [[nodiscard]] const ExecutionSpace& fft_space_for(std::size_t samples);

  Problem problem_;
  CpuExecutionPool kernel_pool_{CpuPool::kernels};
  CpuExecutionPool fft_pool_{CpuPool::fft};
  CpuExecutionPool raman_pool_{CpuPool::raman_fft};
  CpuExecutionPool raman_tail_pool_{CpuPool::raman_tail_fft};
  mutable ExecutionSpace execution_space_;
  ExecutionSpace response_execution_space_, response_tail_execution_space_;
  ExecutionSpace transform_execution_space_, transform_tail_execution_space_;
  ExecutionSpace matrix_copy_space_, tail_matrix_copy_space_;
  ExecutionSpace stacked_matrix_copy_space_, tail_stacked_matrix_copy_space_;
  ExecutionSpace node_spectral_space_;
  const ExecutionSpace& matrix_copy_space_for(std::size_t nodes,
                                              bool stacked = false) const {
    if (stacked)
      return nodes == problem_.node_chunk ? stacked_matrix_copy_space_
                                          : tail_stacked_matrix_copy_space_;
    return nodes == problem_.node_chunk ? matrix_copy_space_ : tail_matrix_copy_space_;
  }
  ExecutionSpace fft_execution_space_;
  ExecutionSpace temporal_execution_space_, flow_execution_space_,
      absorber_execution_space_;
  ExecutionSpace raman_multiply_space_, raman_tail_multiply_space_,
      raman_rebuild_space_, raman_scale_space_;
  ExecutionSpace raman_execution_space_, raman_tail_execution_space_,
      raman_kernel_execution_space_;
  const ExecutionSpace& response_space_for(std::size_t nodes) const {
    return nodes == problem_.node_chunk ? response_execution_space_
                                        : response_tail_execution_space_;
  }
  const ExecutionSpace& transform_space_for(std::size_t nodes) const {
    return nodes == problem_.node_chunk ? transform_execution_space_
                                        : transform_tail_execution_space_;
  }
  bool closed_{};
  bool initialized_{};
  // A constant operator's low-storage table is built (set on a rebuild, cleared by
  // `set_initial` and `close`): a step that reuses a table never built refuses.
  bool table_ready_{};
  bool stages_ready_{};
  bool batch_prepared_{};
  bool has_noise_{};
  bool has_recycled_{};
  std::size_t rk_source_evaluations_{};
  bool pending_fsal_{};
  // Only the final trial source supplies the accepted quadrature score. Its
  // coarse time buffers survive until scoring, or until a new source replaces
  // them. The stored accepted score and peak are independent of these flags.
  bool defer_quadrature_{};
  bool project_coarse_{true};
  bool coarse_pending_{};
  // Set by a trial for a stage whose finish may run in the next stage's entry kernel;
  // `finish_pending_` names the stage's finish still to run there.
  bool defer_finish_{};
  bool finish_pending_{};
  std::size_t pending_stage_{};
  bool stash_is_current_{};
  std::size_t plan_heap_growth_{}, plan_device_growth_{};
  double planning_seconds_{};
  double raman_planning_seconds_{};
  // Every fixed Phi term is purely imaginary: the stage phases have no real part.
  bool lossless_phase_ = false;
  // Per stage of the current trial, whether its phase weights equal the stage
  // before's, and whether the whole step's equal the last stage's (`upload_trial`);
  // the last stage whose entry formed the pair in this trial.
  std::vector<bool> phase_repeats_;
  bool whole_repeats_ = false;
  std::size_t entered_stage_ = std::numeric_limits<std::size_t>::max();
  std::size_t raman_kernel_transforms_{};
  bool raman_kernel_ready_{};
  std::vector<double> cached_raman_rates_, cached_raman_densities_;
  mutable double quadrature_error_{Kokkos::Experimental::quiet_NaN_v<double>};
  bool adams_device_scale_ = false;
  double adams_rtol_ = 0.0, adams_atol_ = 0.0;
  RealView adams_scale_weights_;
  Kokkos::View<double, Layout, MemorySpace> adams_magnitude_sum_;
  // The device's (last, peak) scores: an accepted step scores the quadrature on the
  // device and nothing crosses until the score is read.
  RealView quadrature_scores_;
  mutable bool quadrature_stale_ = false;
  SpatialCoefficients spatial_;
  // solprop SEP 34: the plane measure's coefficients. A separable run's are its own
  // operator's views (aliased, nothing more held); a radial run's are the entrance
  // squares, the material square replaced by the material z table's value at the
  // statistic's plane when the medium varies (`flux_table_`).
  SpatialCoefficients flux_;
  z_table::Table flux_table_;
  // The magnetic closure's transverse volume on a separable run, beside
  // `transverse_`, allocated when a local flux statistic is asked for.
  ComplexView3 flux_transverse_;
  ComplexView3 transverse_;
  std::unique_ptr<TransversePlan> transverse_forward_, transverse_inverse_;
  std::unique_ptr<SlabPlan> slab_forward_, slab_inverse_;
  ComplexView2 current_, advanced_, solution_, error_, residual_, recycled_, noise_;
  // solprop SEP 24 phase 3: the low-storage march's registers B and C (the state is
  // A), its flows and the field-free plane it holds (SEP 22, Q15), allocated at the
  // first such plane.
  ComplexView2 register_accumulator_, register_stage_, held_plane_;
  ComplexView3 low_storage_flows_;
  bool held_ready_{};
  ComplexView2 whole_propagator_, fine_spectrum_, projected_spectrum_;
  // The stage's `sum_t dPhi_t Phi_t`, formed by the combination for a coupled flow's
  // residual, and its inverse propagator `exp(-sum)`, which the finish applies.
  ComplexView2 stage_phase_, stage_inverse_;
  ComplexView2 projected_coarse_spectrum_, absorber_spectrum_, modal_complex_;
  ComplexView2 statistics_spectrum_, statistics_time_, statistics_weighted_time_;
  RealView2 statistics_power_weight_;
  bool statistics_unit_weight_ = true;
  ComplexView2 node_complex_, polarisation_complex_, projected_complex_;
  ComplexView2 projected_coarse_complex_, absorber_complex_, coarse_spectrum_;
  ComplexView2 coarse_time_complex_, scaled_synthesis_complex_;
  ComplexView2 scaled_projection_complex_, scaled_projection_coarse_complex_;
  ComplexView2 scaled_projection_stacked_complex_, projected_stacked_complex_;
  ComplexView2 chunk_synthesis_complex_, tail_synthesis_complex_;
  ComplexView2 chunk_projection_complex_, tail_projection_complex_;
  ComplexView2 chunk_projection_stacked_complex_, tail_projection_stacked_complex_;
  ComplexView2 node_raman_spectrum_, raman_kernel_spectrum_;
  ComplexView2 raman_tail_spectrum_;
  ComplexView2 spatial_absorber_complex_;
  ComplexView3 stages_, phi_terms_, prefactor_terms_;
  // solprop SEP 29 G1: the general z tables' coefficients, uploaded once; their values
  // at a trial's positions are written into the term slots above.
  z_table::Table phi_table_, prefactor_table_, rate_table_;
  RealView2 radial_blas_input_, radial_blas_output_;
  RealView2 modal_real_, node_real_, polarisation_real_, projected_real_;
  RealView2 projected_coarse_real_, projected_stacked_real_, absorber_real_;
  RealView2 coarse_time_real_;
  RealView2 node_raman_drive_, node_raman_convolved_, raman_undamped_;
  RealView2 raman_tail_drive_, raman_tail_convolved_;
  RealView2 raman_kernel_time_, stage_raman_rates_, stage_raman_densities_;
  RealView2 raman_tensor_weights_;
  RealView raman_row_component_, raman_component_fraction_;
  RealView2 synthesis_, projection_, projection_coarse_;
  RealView2 spatial_absorber_real_;
  RealView2 scaled_synthesis_, scaled_projection_, scaled_projection_coarse_;
  RealView2 scaled_projection_stacked_;
  RealView2 chunk_synthesis_, tail_synthesis_;
  RealView2 chunk_projection_, tail_projection_;
  RealView2 chunk_projection_stacked_, tail_projection_stacked_;
  RealView2 partials_, quadrature_partials_, stash_candidate_, stash_current_;
  RealView2 statistics_power_, plasma_statistics_scratch_;
  // solprop SEP 34: each node's `|E| |G|` beside its signed flux, which bounds the
  // fluence's rounding; allocated with the closure's buffers.
  RealView2 statistics_magnitude_;
  RealView2 plasma_rate_, plasma_exposure_, plasma_fraction_;
  RealView2 plasma_current_integrand_, plasma_free_current_;
  // The device trapezoid plasma's tile totals, by component row, and its tile count
  // (`device_trapezoid_plasma`).
  RealView2 plasma_tile_totals_;
  std::size_t plasma_tiles_{};
  RealView2 plasma_outer_current_;
  RealView weighted_scale_, weighted_partials_;
  RealView spatial_spectral_weights_;
  ComplexView spatial_on_axis_;
  RealView2 spatial_statistics_partials_;
  RealView statistics_total_power_;
  RealView statistics_on_axis_, statistics_packed_;
  RealView tableau_, omega_, omega_window_, time_window_, absorber_fine_,
      absorber_coarse_;
  RealView stage_phi_delta_, stage_phi_at_, stage_prefactor_;
  RealView stage_density_, stage_taper_, stage_preionisation_, rate_coefficients_;
  RealView stage_kerr_density_, stage_plasma_density_, stage_raman_density_;
  // Every per-stage scalar view above lives in one buffer, uploaded in one copy from
  // the next of a few host staging buffers (`upload_trial`).
  RealView stage_scalars_;
  std::vector<RealView::host_mirror_type> stage_staging_;
  std::size_t stage_staging_next_ = 0;
  RealView2 channel_parameters_, stage_channel_density_, stage_channel_preionisation_;
  RealView channel_coefficients_;
  Kokkos::View<double***, Layout, MemorySpace> channel_scratch_;
  // solprop SEP 18 on a CPU build: the running-integral rule's taps, one row of ring
  // scratch per node of a chunk (the band column's, or the statistics scan's alone
  // under the trapezoid) and the run diagnostics' six parts per node.
  RealView plasma_taps_;
  RealView2 plasma_diagnostics_;
  // The band column's scratch rows per kind: one per concurrently running team.
  std::size_t band_slots_{};
  // Whether the band rule runs the tiled column (`tiled_band_plasma_column`), its
  // tile's samples, the tiles of a row, a host tile team's threads, and the tiles'
  // totals by row.
  bool band_tiled_{band_tiled(thread_policy())};
  // Each column's norm reductions on a device are this many teams (`reduction_slots`):
  // the norm partials hold `columns * reduction_slots_` slots, reduced on the host.
  std::size_t reduction_slots_{reduction_slots(problem_.n_columns(), problem_.n_omega)};
  std::size_t band_tile_{};
  std::size_t band_tile_count_{};
  int band_tile_threads_{band_tile_threads(thread_policy())};
  RealView2 band_tile_totals_;
  ComplexView carrier_;
  std::array<EnvelopeStream, 2> envelope_streams_;
  std::array<FieldCurrentStream, 3> held_streams_;
  RealView2 held_current_, held_projection_;
  ComplexView2 held_nodal_spectrum_, held_projected_spectrum_;
  RealView held_derivative_, held_inverse_;
  RealView3 envelope_rule_nodes_, envelope_rule_weights_, envelope_moments_;
  RealView2 envelope_rates_, envelope_exposure_;
  IndexView envelope_failures_, envelope_status_;
  Kokkos::View<std::size_t*, Layout, Kokkos::HostSpace> envelope_host_status_;
  RealView envelope_drude_factor_;
  ComplexView envelope_loss_factor_;
  ByteView active_mask_, norm_mask_;
  // solprop SEP 16 phase 3b: the resident Adams history and its trial's scratch.
  ComplexView3 adams_store_, adams_slopes_, adams_moments_;
  ComplexView2 adams_source_, adams_propagator_, adams_exponent_, adams_saved_;
  ComplexView2 adams_held_, adams_dense_start_, adams_dense_end_, adams_dense_exponent_;
  Kokkos::View<double*, Layout, MemorySpace> adams_divisor_;
  // A varying operator's compensation: each sample's state and rate, the trial's
  // carried rows, its predicted field, endpoint term and rate, and dense observation's
  // terms.
  ComplexView3 adams_states_, adams_rates_, adams_carried_, adams_dense_terms_;
  ComplexView2 adams_lab_, adams_endpoint_, adams_endpoint_rate_;
  Kokkos::View<double*, Layout, MemorySpace> adams_rate_weights_;
  // A coupled flow's exponential trial (solprop SEP 16 phase 6): the step's flow and
  // moments on the block's bins, `(count + 1, n_active, n_modes, n_modes)` with the
  // flow first; a frozen-residual flow's weights at each sample and at the trial's end.
  Kokkos::View<Complex****, Layout, MemorySpace> adams_block_;
  // A frozen-residual flow's generator at each sample applied to its state, `G(z_j)
  // u_j`, formed once per sample; the trial's at its end, and the weights it is formed
  // from. A trial forms `R_s(z_j) u_j = G(z_j) u_j - G_s u_j` against its own section.
  ComplexView3 adams_applied_;
  ComplexView2 adams_end_applied_;
  Kokkos::View<double*, Layout, MemorySpace> adams_end_weights_;
  std::size_t adams_dense_count_{};
  bool adams_trial_ready_{};
  bool adams_trial_compensated_{};
  bool adams_trial_coupled_{};
  bool adams_dense_compensated_{};
  // The persistent row view a resume exports or installs, by name (SEP 29 G2).
  ComplexView3 adams_row_view(const std::string& name) const;
  IndexView crop_index_;
  // The fine bins outside the crop, zeroed by the stage's entry kernel.
  IndexView fine_gaps_;
  // The separable core's fine bins outside the crop, zeroed per chunk instead of the
  // whole fine spectrum (SEP 28); empty when the crop fills every fine bin.
  IndexView separable_gaps_;
  // Free space's node field and polarisation alias the chunk's time rows and their
  // projected rows (SEP 28): the copies between them are identities.
  bool separable_aliased_{};
  std::size_t temporal_tail_rows_{};
  TailRowPlans<ExecutionSpace, ComplexView2, RealView2> temporal_tail_inverse_real_,
      coarse_tail_inverse_real_;
  TailRowPlans<ExecutionSpace, RealView2, ComplexView2> temporal_tail_forward_real_,
      coarse_tail_forward_real_;
  TailRowPlans<ExecutionSpace, ComplexView2, ComplexView2>
      temporal_tail_inverse_complex_, temporal_tail_forward_complex_,
      coarse_tail_inverse_complex_, coarse_tail_forward_complex_;
  std::unique_ptr<InverseRealPlan> inverse_real_plan_;
  std::unique_ptr<ComplexPlan> inverse_complex_plan_;
  std::unique_ptr<ForwardRealPlan> forward_real_plan_;
  std::unique_ptr<ComplexPlan> forward_complex_plan_;
  std::unique_ptr<ForwardRealPlan> forward_coarse_real_plan_;
  std::unique_ptr<ComplexPlan> forward_coarse_complex_plan_;
  std::unique_ptr<ComplexPlan> statistics_inverse_plan_;
  std::unique_ptr<ForwardRealPlan> absorber_real_plan_;
  std::unique_ptr<ComplexPlan> absorber_complex_plan_;
  std::unique_ptr<InverseRealPlan> coarse_inverse_real_plan_;
  std::unique_ptr<ComplexPlan> coarse_inverse_complex_plan_;
  std::unique_ptr<ForwardRealPlan> coarse_forward_real_plan_;
  std::unique_ptr<ComplexPlan> coarse_forward_complex_plan_;
  std::unique_ptr<RamanForwardPlan> raman_drive_forward_plan_;
  std::unique_ptr<RamanInversePlan> raman_convolution_inverse_plan_;
  std::unique_ptr<RamanForwardPlan> raman_tail_forward_plan_;
  std::unique_ptr<RamanInversePlan> raman_tail_inverse_plan_;
  std::unique_ptr<RamanForwardPlan> raman_kernel_forward_plan_;
  // solprop's general modal route; null on every other route.
  std::unique_ptr<GeneralModalSource> general_;
  bool absorber_quiet_ = false;
  // solprop's boundary monitor (SEP 21, ADR-057), bound for a run that watches its
  // window: its four time weights `(4, n_t)` (low guard, high guard, the collar's
  // 1 - W, the collar's rate), each state column's quadrature weight, and what the
  // split-step collar took since the last reading beside one split's sum and the
  // share it takes at each time. Allocated at the first bind, a few rows, and kept.
  bool monitor_bound_ = false;
  RealView2 monitor_weights_;
  RealView monitor_column_weights_;
  RealView monitor_share_;
  Kokkos::View<double, Layout, MemorySpace> monitor_removed_;
  Kokkos::View<double, Layout, MemorySpace> monitor_split_sum_;
  [[nodiscard]] bool monitoring_split() const noexcept {
    return monitor_bound_ && problem_.split_step_absorber;
  }
  // Whether the stage adds the in-stage absorber's term. A quiet temporal collar is
  // left out on the modal routes, as solprop's reference leaves it out, unless a
  // real-space collar shares the term or the route is the general modal one or a
  // coupled flow, which keep it (solprop SEP 21).
  [[nodiscard]] bool absorbing() const noexcept {
    return problem_.in_stage_absorber &&
           !(absorber_quiet_ && !general_ && !problem_.coupled_flow &&
             problem_.spatial_absorber_matrix.empty());
  }
  // solprop's linear.coupled_flow; null otherwise.
  std::unique_ptr<CoupledFlowAction> flow_;
  // solprop SEP 6's dispersive electronic response on the fused node path, and on the
  // Cartesian free-space, line and planar routes; null otherwise. The general modal
  // route's is its source's.
  std::unique_ptr<DispersiveResponse> dispersive_;
  // solprop SEP 23's far-field pass, on free space with spatial statistics.
  std::unique_ptr<FarFieldPass> far_field_;
  std::unique_ptr<SeparableDispersive> separable_dispersive_;
  // The response's layout and stage weights on whichever route runs it, or null.
  DispersiveWeights* dispersive_weights() const noexcept {
    if (dispersive_)
      return &dispersive_->weights();
    if (separable_dispersive_)
      return &separable_dispersive_->weights();
    if (general_ && general_->dispersive() != nullptr)
      return &general_->dispersive()->weights();
    return nullptr;
  }
};

// The hybrid policy's shape guard. A host build sends every eligible transform to the
// vendor GEMM: measured on Apple's Accelerate with one thread, it beat the custom
// kernel at every shape tried -- a radial field's transforms 1.1 to 1.5 times faster
// at 2 to 6 nodes and 1.6 to 35 times at 16 to 256 nodes and 256 to 2048 samples, a
// one-mode field capillary's 2.4 to 3.1 times -- and lost at none. A device build
// keeps the earlier conservative guard (two modes, eight nodes, 1024 samples) until
// its own crossover is measured.
bool GeneralWorkspace::hybrid_shape_eligible() const noexcept {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  return problem_.n_modes >= 2 && problem_.n_nodes >= 8 && problem_.n_to >= 1024;
#else
  return true;
#endif
}

bool GeneralWorkspace::use_packed_radial_blas() const noexcept {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  return false;
#else
  const auto provider = compiled_gemm_provider();
  // Two packed work arrays add two complex-equivalent time columns to the
  // shared conservative twenty-column inventory. Decide before allocation.
  constexpr std::size_t max_samples = 64 * 1024 * 1024 / (22 * sizeof(Complex));
  return (provider == GemmProvider::hybrid ||
          provider == GemmProvider::kokkosblas_tpl) &&
         !problem_.separable_geometry() && !problem_.spatial_spectral_weights.empty() &&
         problem_.n_modes == problem_.n_nodes &&
         problem_.node_chunk == problem_.n_nodes && problem_.n_pol_state == 1 &&
         problem_.n_pol_transform == 1 && problem_.family == FieldFamily::envelope &&
         problem_.projection_coarse_matrix.empty() && problem_.n_nodes >= 256 &&
         problem_.n_to >= 256 && problem_.n_nodes <= max_samples / problem_.n_to;
#endif
}

void GeneralWorkspace::packed_radial_gemm(const RealView2& matrix,
                                          const ComplexView2& source,
                                          const ComplexView2& destination) {
  const auto input = radial_blas_input_;
  const auto output = radial_blas_output_;
  const auto rows = problem_.n_nodes;
  const auto times = problem_.n_to;
  Kokkos::Profiling::pushRegion("solprop_native.radial.packed_real_gemm");
  Kokkos::parallel_for(
      "solprop_native.radial.pack",
      Range2(temporal_execution_space_, {0, 0}, {rows, times}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
        input(row, 2 * time) = source(row, time).real();
        input(row, 2 * time + 1) = source(row, time).imag();
      });
  engine_gemm(transform_execution_space_, matrix, input, output, 0.0);
  Kokkos::parallel_for(
      "solprop_native.radial.unpack",
      Range2(temporal_execution_space_, {0, 0}, {rows, times}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
        destination(row, time) =
            Complex(output(row, 2 * time), output(row, 2 * time + 1));
      });
  Kokkos::Profiling::popRegion();
}

bool GeneralWorkspace::use_blas_for_synthesis() const noexcept {
  if (use_packed_radial_blas())
    return false;
  // The hybrid planar kind contracts its small x rule per y row natively.
  if (problem_.separable_geometry())
    return false;
  const auto provider = compiled_gemm_provider();
  if (provider == GemmProvider::custom)
    return false;
  if (problem_.n_pol_state != 1)
    return false;
  if (problem_.family == FieldFamily::envelope) {
#if defined(SOLPROP_NATIVE_TPL_BLAS_OPENBLAS) ||                                       \
    defined(SOLPROP_NATIVE_TPL_BLAS_CUBLAS) ||                                         \
    defined(SOLPROP_NATIVE_TPL_BLAS_ROCBLAS)
    if (provider != GemmProvider::kokkosblas_tpl && provider != GemmProvider::hybrid)
      return false;
#else
    return false;
#endif
  } else if (problem_.family != FieldFamily::field_real) {
    return false;
  }
#if defined(SOLPROP_NATIVE_TPL_BLAS_CUBLAS) || defined(SOLPROP_NATIVE_TPL_BLAS_ROCBLAS)
  // Spatial envelope matrices can be large even with a short temporal grid.
  if (!problem_.spatial_spectral_weights.empty() &&
      problem_.family == FieldFamily::envelope &&
      problem_.projection_coarse_matrix.empty() &&
      problem_.n_modes == problem_.n_nodes && problem_.n_nodes >= 256 &&
      problem_.n_to >= 256)
    return true;
#endif
  return provider != GemmProvider::hybrid || hybrid_shape_eligible();
}

// An analytic field's or an envelope's synthesis is a real matrix times complex
// samples: viewed as real numbers, its samples are a real matrix twice as wide, so the
// transform is one real GEMM through the selected BLAS, with no copy and no complex
// BLAS. An envelope's projection is the same (`use_interleaved_projection`).
//
// A device takes it at the shapes where it already takes BLAS for the projection
// (`hybrid_shape_eligible`): there the custom synthesis kernel cost 4.6 times the
// projection's GEMM at 512 radial nodes (an AMD CDNA 3 device, 2026-10-06).
bool GeneralWorkspace::use_interleaved_synthesis() const noexcept {
  if (problem_.family == FieldFamily::field_real || problem_.separable_geometry() ||
      problem_.n_pol_state != 1 || problem_.n_pol_transform != 1)
    return false;
  const auto provider = compiled_gemm_provider();
  if (provider == GemmProvider::custom)
    return false;
  return provider != GemmProvider::hybrid || hybrid_shape_eligible();
}

bool GeneralWorkspace::use_interleaved_projection() const noexcept {
  if (problem_.family != FieldFamily::envelope || problem_.separable_geometry() ||
      problem_.n_pol_state != 1 || problem_.n_pol_transform != 1)
    return false;
  const auto provider = compiled_gemm_provider();
  if (provider == GemmProvider::custom)
    return false;
  return provider != GemmProvider::hybrid || hybrid_shape_eligible();
}

bool GeneralWorkspace::use_blas_for_projection() const noexcept {
  if (use_packed_radial_blas())
    return false;
  // The hybrid planar kind contracts its small x rule per y row natively.
  if (problem_.separable_geometry())
    return false;
  const auto provider = compiled_gemm_provider();
  if (provider == GemmProvider::custom)
    return false;
  if (problem_.n_pol_state != 1)
    return false;
  if (problem_.family == FieldFamily::envelope) {
#if defined(SOLPROP_NATIVE_TPL_BLAS_OPENBLAS) ||                                       \
    defined(SOLPROP_NATIVE_TPL_BLAS_CUBLAS) ||                                         \
    defined(SOLPROP_NATIVE_TPL_BLAS_ROCBLAS)
    if (provider != GemmProvider::kokkosblas_tpl && provider != GemmProvider::hybrid)
      return false;
#else
    return false;
#endif
  }
#if defined(SOLPROP_NATIVE_TPL_BLAS_CUBLAS) || defined(SOLPROP_NATIVE_TPL_BLAS_ROCBLAS)
  // Spatial envelope matrices can be large even with a short temporal grid.
  if (!problem_.spatial_spectral_weights.empty() &&
      problem_.family == FieldFamily::envelope &&
      problem_.projection_coarse_matrix.empty() &&
      problem_.n_modes == problem_.n_nodes && problem_.n_nodes >= 256 &&
      problem_.n_to >= 256)
    return true;
#endif
  return provider != GemmProvider::hybrid || hybrid_shape_eligible();
}

bool GeneralWorkspace::use_chunked_blas_for_synthesis() const noexcept {
  return problem_.node_chunk < problem_.n_nodes && use_blas_for_synthesis();
}

// One node, one mode and one scalar component: the synthesis and the projection are
// each one product, which the response column forms itself (solprop SEP 25 section
// 6.7), as the Kerr-only engine's `kerr_1x1` does, in place of the stage's taper
// scaling, two one-element GEMMs and the zero fill of the projection. Each folded
// product is the GEMM's own one rounding, `(s * taper) * x` and `(p / taper) * y`.
// The host's columns only: a device keeps its own column kernels.
bool GeneralWorkspace::folds_single_node() const noexcept {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  return false;
#else
  return !general_ && !dispersive_ && !held_radial() &&
         !problem_.separable_geometry() && problem_.n_modes == 1 &&
         problem_.n_nodes == 1 && problem_.node_chunk == 1 &&
         problem_.n_pol_state == 1 && problem_.n_pol_transform == 1 &&
         problem_.projection_coarse_matrix.empty() && !problem_.raman &&
         !problem_.envelope_plasma && radial_blas_input_.empty();
#endif
}

bool GeneralWorkspace::use_chunked_blas_for_projection() const noexcept {
  return problem_.node_chunk < problem_.n_nodes && use_blas_for_projection();
}

std::string GeneralWorkspace::synthesis_provider() const {
  const auto provider = selected_blas_provider_name();
  if (folds_single_node())
    return "folded";
  if (use_packed_radial_blas())
    return std::string(provider) + "-packed-real";
  if (problem_.free_geometry())
    return "kokkos-fft";
  if (use_interleaved_synthesis())
    return std::string(provider) + "-interleaved-real";
  return std::string(use_blas_for_synthesis() ? provider : std::string_view("custom"));
}

std::string GeneralWorkspace::projection_provider() const {
  const auto provider = selected_blas_provider_name();
  if (folds_single_node())
    return "folded";
  if (use_packed_radial_blas())
    return std::string(provider) + "-packed-real";
  if (problem_.free_geometry())
    return "kokkos-fft";
  if (use_interleaved_projection())
    return std::string(provider) + "-interleaved-real";
  return std::string(use_blas_for_projection() ? provider : std::string_view("custom"));
}

std::string GeneralWorkspace::gemm_shape_policy() {
  return "host radial scalar envelope uses packed real BLAS for whole transforms "
         "with n_nodes/n_to >= 256 and a 22-column inventory within 64 MiB; "
         "device radial scalar envelope uses vendor complex BLAS with "
         "n_nodes/n_to >= 256; otherwise hybrid uses KokkosBlas for every whole "
         "transform on the host, and on a device for those with n_modes >= 2, "
         "n_nodes >= 8 and n_to >= 1024: real ones directly, an analytic field's "
         "synthesis as one real GEMM over its samples' interleaved real view, and its "
         "projection; host radial analytic synthesis also uses unallocated row views "
         "for partial chunks; an envelope's synthesis and projection the same way; "
         "OpenBLAS, "
         "cuBLAS and rocBLAS builds also use "
         "preconverted complex GEMM for eligible envelope transforms; eligible chunked "
         "transforms are packed into persistent buffers for vendor BLAS; vector state "
         "uses custom; a host build without a vendor BLAS runs KokkosBlas's native "
         "arithmetic through a tiled host kernel, bit for bit";
}

std::size_t GeneralWorkspace::response_team_size() const noexcept {
  constexpr auto configured = static_cast<std::size_t>(SOLPROP_NATIVE_DEVICE_TEAM_SIZE);
#if defined(KOKKOS_ENABLE_CUDA)
  // Measured on a Hopper GPU, the crossover is sharp: 4,096-sample fibre problems
  // favour teams of 256, while 8,192 samples and more favour 512. Keep an explicitly
  // configured smaller team intact.
  if (problem_.n_to < 8192)
    return std::min(configured, std::size_t{256});
#endif
  return configured;
}

class GeneralBatchWorkspace {
public:
  explicit GeneralBatchWorkspace(std::vector<std::shared_ptr<GeneralState>> members)
      : members_(std::move(members)) {
    if (members_.empty())
      throw std::invalid_argument("a native batch needs at least one member");
    for (const auto& state : members_) {
      if (!state || state->closed())
        throw std::invalid_argument("native batch members must be open");
      const auto& one = *state->workspace_;
      if (!one.initialized_)
        throw std::invalid_argument("native batch members must be loaded");
      if (one.flow_)
        throw std::invalid_argument(
            "a native batch member carries a coupled flow, which runs one member at "
            "a time");
      if (columns_ == 0) {
        columns_ = one.problem_.n_columns();
        omega_ = one.problem_.n_omega;
      }
      if (one.problem_.n_columns() != columns_ || one.problem_.n_omega != omega_)
        throw std::invalid_argument("native batch member layouts differ");
    }
    const auto count = checked_product(columns_, members_.size(), "batch columns");
    execution_ = kernel_pool_.get(CpuOperation::stage,
                                  checked_product(count, omega_, "batch elements"));
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::length_error("native batch league exceeds int extent");
    require_allocation(count, sizeof(UpdateColumn), "batch descriptors");
    descriptors_ = Descriptors("solprop_native.batch.update_descriptors", count);
    host_ = Kokkos::create_mirror(descriptors_);
  }

  std::vector<double> accept(const double dz) {
    if (closed_)
      throw std::logic_error("native batch is closed");
    if (!std::isfinite(dz) || dz <= 0.0)
      throw std::invalid_argument("native batch step must be positive and finite");
    for (std::size_t member = 0; member < members_.size(); ++member) {
      auto& one = *members_[member]->workspace_;
      one.require_open();
      if (!one.batch_prepared_)
        throw std::logic_error("every native batch member must prepare this trial");
      for (std::size_t column = 0; column < columns_; ++column) {
        host_(member * columns_ + column) = {
            one.current_.data() + column * one.current_.stride(0),
            one.solution_.data() + column * one.solution_.stride(0),
            one.error_.data() + column * one.error_.stride(0),
            one.whole_propagator_.data() + column * one.whole_propagator_.stride(0),
            one.advanced_.data() + column * one.advanced_.stride(0),
            one.residual_.data() + column * one.residual_.stride(0),
            one.norm_mask_.data(),
            one.partials_.data() + column * one.reduction_slots_,
            one.partials_.data() + one.partials_.stride(0) +
                column * one.reduction_slots_,
            one.partials_.data() + 2 * one.partials_.stride(0) +
                column * one.reduction_slots_,
            one.current_.data() + column * one.current_.stride(0)};
      }
    }
    Kokkos::deep_copy(execution_, descriptors_, host_);
    const auto descriptors = descriptors_;
    // The members' own split: a member sums its norms as its solo run does.
    const auto slots = members_.front()->workspace_->reduction_slots_;
    accept_update(
        execution_, descriptors.extent(0), omega_, dz,
        KOKKOS_LAMBDA(const std::size_t column) { return descriptors(column); },
        "solprop_native.batch.accept_error_partials", slots);
    execution_.fence("solprop_native batch update boundary");
    std::vector<double> result;
    result.reserve(members_.size() * 3 * columns_ * slots);
    for (const auto& member : members_) {
      const auto partials = member->workspace_->collect_trial();
      result.insert(result.end(), partials.begin(), partials.end());
    }
    return result;
  }

  std::map<std::string, std::size_t> memory_ledger() const {
    if (closed_)
      return {};
    return {{"update_descriptors", descriptors_.span() * sizeof(UpdateColumn)},
            {"host_update_descriptors", host_.span() * sizeof(UpdateColumn)}};
  }

  CpuSizing execution_sizing() const { return kernel_pool_.sizing(); }

  void close() noexcept {
    if (closed_)
      return;
    execution_.fence();
    descriptors_ = {};
    host_ = {};
    members_.clear();
    execution_ = ExecutionSpace();
    kernel_pool_.clear();
    closed_ = true;
  }

private:
  using Descriptors = Kokkos::View<UpdateColumn*, Layout, MemorySpace>;
  std::vector<std::shared_ptr<GeneralState>> members_;
  CpuExecutionPool kernel_pool_{CpuPool::kernels};
  ExecutionSpace execution_;
  std::size_t columns_{}, omega_{};
  Descriptors descriptors_;
  Descriptors::host_mirror_type host_;
  bool closed_{};
};

void GeneralWorkspace::allocate() {
  if (problem_.low_storage
          ? problem_.low_storage_flows > 2 || problem_.coupled_flow ||
                (!problem_.low_storage_stage_register && !problem_.free_geometry())
          : problem_.low_storage_flows != 0 || !problem_.low_storage_stage_register)
    throw std::invalid_argument(
        "the low-storage march holds 0 to 2 flows beside no coupled flow, and elides "
        "its stage register on free space alone");
  // The exchange fold of an in-stage absorber re-forms its input from the stage
  // register, and a separable fold applies one flow volume, a constant operator's
  // table (solprop SEP 24 section 4.3): neither has a meaning to fall back on.
  if (problem_.low_storage && problem_.in_stage_absorber &&
      !problem_.low_storage_stage_register)
    throw std::invalid_argument("the low-storage march cannot elide its stage register "
                                "beside an in-stage absorber");
  if (problem_.low_storage && problem_.separable_geometry() &&
      problem_.low_storage_flows > 1)
    throw std::invalid_argument(
        "the low-storage march holds at most one flow volume on "
        "a separable geometry");
  const auto columns = problem_.n_columns();
  const auto n = problem_.n_omega;
  const auto chunk_rows = problem_.node_chunk * problem_.effective_node_pol();
  const auto transform_rows =
      problem_.separable_geometry() ? 0 : problem_.n_nodes * problem_.n_pol_transform;
  // A planar chunk's temporal rows are its y rows' columns, not its nodes.
  const auto temporal_columns = problem_.slab_geometry()
                                    ? transverse_units_per_chunk() * problem_.n_p
                                    : (problem_.free_geometry() ? chunk_rows : columns);
  separable_aliased_ = problem_.free_geometry() && !problem_.slab_geometry() &&
                       !problem_.dispersive && temporal_columns == chunk_rows;
  const auto statistic_nodes =
      problem_.separable_geometry() ? problem_.node_chunk : problem_.n_modes;
  // Spatial statistics reconstruct one chunk of physical nodes at a time.
  const auto statistics_rows = problem_.slab_geometry() ? chunk_rows : temporal_columns;
  if (problem_.separable_geometry()) {
    // Keeping transform axes trailing avoids Kokkos-FFT's per-execution
    // transpose allocations. The canonical state retains omega-contiguous columns.
    transverse_ =
        problem_.slab_geometry()
            ? ComplexView3("solprop_native.slab.transverse", n, problem_.n_p,
                           problem_.n_y)
            : ComplexView3("solprop_native.free.transverse", problem_.n_pol_state * n,
                           problem_.n_y, problem_.n_x);
    spatial_.square = ComplexView("solprop_native.free.k_squared", n);
    spatial_.square_low = ComplexView("solprop_native.free.k_squared_low", n);
    spatial_.numerator = ComplexView("solprop_native.free.numerator", n);
    spatial_.clamped = ComplexView("solprop_native.free.clamped", n);
    spatial_.transverse = RealView("solprop_native.free.kperp_sq", problem_.n_modes);
    spatial_.transverse_low =
        RealView("solprop_native.free.kperp_sq_low", problem_.n_modes);
    spatial_.frame = RealView("solprop_native.free.frame", n);
    spatial_.spectral_rate = RealView("solprop_native.free.spectral_rate", n);
    spatial_.absorber = RealView("solprop_native.free.absorber", problem_.n_modes);
    spatial_.transverse_rate =
        RealView("solprop_native.free.kperp_rate", problem_.n_modes);
    spatial_.evanescent = problem_.spatial_evanescent_rate;
    if (problem_.slab_geometry()) {
      spatial_.kappa = ComplexView2("solprop_native.slab.kappa_sq", problem_.n_p, n);
      spatial_.channels = problem_.n_p;
    }
    flux_ = spatial_;
    if (problem_.flux_closure)
      flux_transverse_ =
          ComplexView3("solprop_native.flux.closure_transverse", transverse_.extent(0),
                       transverse_.extent(1), transverse_.extent(2));
  } else if (!problem_.flux_k_squared.empty()) {
    flux_.square = ComplexView("solprop_native.flux.k_squared", n);
    flux_.square_low = ComplexView("solprop_native.flux.k_squared_low", n);
    flux_.transverse = RealView("solprop_native.flux.kperp_sq", problem_.n_modes);
    flux_.transverse_low =
        RealView("solprop_native.flux.kperp_sq_low", problem_.n_modes);
  }
  current_ = ComplexView2("solprop_native.state.current", columns, n);
  if (problem_.low_storage) {
    // solprop SEP 24 section 6.7: the march holds its registers beside the state and
    // none of the Runge-Kutta views; the stage's input is formed in the route's own
    // scratch (`transverse_`, or the fine spectrum of the modal routes).
    register_accumulator_ =
        ComplexView2("solprop_native.low_storage.accumulator", columns, n);
    if (problem_.low_storage_stage_register)
      register_stage_ = ComplexView2("solprop_native.low_storage.stage", columns, n);
    low_storage_flows_ = ComplexView3("solprop_native.low_storage.flows",
                                      problem_.low_storage_flows, columns, n);
  } else {
    advanced_ = ComplexView2("solprop_native.state.advanced", columns, n);
    stages_ = ComplexView3("solprop_native.step.stages", problem_.n_stages, columns, n);
    solution_ = ComplexView2("solprop_native.step.solution", columns, n);
    error_ = ComplexView2("solprop_native.step.error", columns, n);
    residual_ = ComplexView2("solprop_native.step.residual", columns, n);
    recycled_ = ComplexView2("solprop_native.step.recycled", columns, n);
    whole_propagator_ =
        ComplexView2("solprop_native.step.whole_propagator", columns, n);
    if (!problem_.separable_geometry())
      stage_inverse_ = ComplexView2("solprop_native.step.stage_inverse", columns, n);
  }
  if (problem_.modified_noise)
    noise_ = ComplexView2("solprop_native.noise", columns, n);
  if (!problem_.separable_geometry() && problem_.coupled_flow)
    stage_phase_ = ComplexView2("solprop_native.step.stage_phase", columns, n);
  partials_ =
      RealView2("solprop_native.step.error_partials", 3,
                checked_product(columns, reduction_slots_, "norm partial slots"));
  if (!problem_.projection_coarse_matrix.empty())
    quadrature_partials_ =
        RealView2("solprop_native.transform.quadrature_partials", 2, columns);
  quadrature_scores_ = RealView("solprop_native.transform.quadrature_scores", 2);
  Kokkos::deep_copy(quadrature_scores_, Kokkos::Experimental::quiet_NaN_v<double>);
  if (problem_.weighted_controller) {
    weighted_scale_ = RealView("solprop_native.step.weighted_scale", n);
    weighted_partials_ =
        RealView("solprop_native.step.weighted_partials", columns * reduction_slots_);
  }
  if (!problem_.spatial_spectral_weights.empty()) {
    spatial_spectral_weights_ =
        RealView("solprop_native.spatial.spectral_weights", problem_.n_modes);
    spatial_on_axis_ = ComplexView("solprop_native.spatial.on_axis", problem_.n_modes);
    // One row per column for the energy and per node for the peaks and fluence,
    // which differ on the hybrid planar kind alone.
    spatial_statistics_partials_ =
        RealView2("solprop_native.spatial.statistics_partials",
                  std::max(problem_.n_modes, problem_.n_nodes), 3);
  }
  statistics_spectrum_ =
      ComplexView2("solprop_native.statistics.spectrum", statistics_rows, problem_.n_t);
  statistics_time_ =
      ComplexView2("solprop_native.statistics.time", statistics_rows, problem_.n_t);
  statistics_power_weight_ =
      RealView2("solprop_native.statistics.power_weight",
                problem_.spatial_spectral_weights.empty() ? columns : 0, n);
  Kokkos::deep_copy(execution_space_, statistics_power_weight_, 1.0);
  // A spatial run's rows hold the magnetic closure in time (solprop SEP 34) when a
  // local flux statistic is asked for.
  statistics_weighted_time_ =
      ComplexView2("solprop_native.statistics.weighted_time",
                   problem_.spatial_spectral_weights.empty() ? temporal_columns
                   : problem_.flux_closure                   ? statistics_rows
                                                             : 0,
                   problem_.n_t);
  statistics_power_ =
      RealView2("solprop_native.statistics.power", statistic_nodes, problem_.n_t);
  if (problem_.flux_closure)
    statistics_magnitude_ = RealView2("solprop_native.statistics.flux_magnitude",
                                      statistic_nodes, problem_.n_t);
  statistics_total_power_ =
      RealView("solprop_native.statistics.total_power", problem_.n_t);
  statistics_on_axis_ =
      RealView("solprop_native.statistics.on_axis", 2 * problem_.n_modes);
  const auto base_statistics = checked_sum(
      checked_sum(problem_.n_t,
                  checked_product(2, problem_.n_columns(), "statistics columns"),
                  "statistics packed output"),
      checked_product(3, problem_.n_modes, "statistics modes"),
      "statistics packed output");
  statistics_packed_ =
      RealView("solprop_native.statistics.packed",
               checked_sum(base_statistics, 5, "statistics packed output"));
  crop_index_ = IndexView("solprop_native.transform.crop_index", n);
  active_mask_ = ByteView("solprop_native.mask.active", n);
  norm_mask_ = ByteView("solprop_native.mask.norm", n);
  omega_window_ = RealView("solprop_native.window.omega", n);
  omega_ = RealView("solprop_native.statistics.omega", problem_.omega.size());
  time_window_ = RealView("solprop_native.window.time", problem_.n_to);
  carrier_ = ComplexView("solprop_native.kerr.carrier", problem_.n_to);
  phi_terms_ = ComplexView3("solprop_native.linear.phi_terms",
                            problem_.separable_geometry() ? 0 : problem_.n_phi_terms,
                            columns, n);
  prefactor_terms_ = ComplexView3(
      "solprop_native.response.prefactor_terms",
      problem_.separable_geometry() ? 0 : problem_.n_prefactor_terms, columns, n);
  tableau_ =
      RealView("solprop_native.step.tableau", problem_.n_stages * problem_.n_stages);
  {
    // One device buffer for every per-stage scalar view, so that a trial's scalars
    // cross in one copy instead of one per view: on a device each small copy from
    // pageable memory is a transfer of its own.
    const auto stages = problem_.n_stages;
    const auto channels = problem_.plasma ? problem_.n_channels : 0;
    const std::size_t sizes[] = {(stages + 1) * problem_.n_phi_terms,
                                 stages * problem_.n_phi_terms,
                                 stages * problem_.n_prefactor_terms,
                                 stages,
                                 stages,
                                 stages,
                                 stages,
                                 stages * channels,
                                 stages * channels,
                                 stages,
                                 stages};
    std::size_t total = 0;
    for (const auto size : sizes)
      total += size;
    stage_scalars_ = RealView("solprop_native.step.scalars", total);
    double* next = stage_scalars_.data();
    const auto take = [&next](const std::size_t size) {
      double* const start = next;
      next += size;
      return start;
    };
    stage_phi_delta_ = RealView(take(sizes[0]), sizes[0]);
    stage_phi_at_ = RealView(take(sizes[1]), sizes[1]);
    stage_prefactor_ = RealView(take(sizes[2]), sizes[2]);
    stage_density_ = RealView(take(stages), stages);
    stage_kerr_density_ = RealView(take(stages), stages);
    stage_plasma_density_ = RealView(take(stages), stages);
    stage_raman_density_ = RealView(take(stages), stages);
    stage_channel_density_ = RealView2(take(stages * channels), stages, channels);
    stage_channel_preionisation_ = RealView2(take(stages * channels), stages, channels);
    stage_taper_ = RealView(take(stages), stages);
    stage_preionisation_ = RealView(take(stages), stages);
    // Two uploads a trial (the step's and the sample's) at most between fences; four
    // staging buffers leave each copy its source until it has surely been read.
    stage_staging_.clear();
    for (int copy = 0; copy < 4; ++copy)
      stage_staging_.push_back(Kokkos::create_mirror_view(stage_scalars_));
  }
  if (problem_.raman) {
    const auto doubled = 2 * problem_.n_to;
    node_raman_drive_ =
        RealView2("solprop_native.raman.node_drive",
                  problem_.node_chunk * problem_.raman_drives(), doubled);
    node_raman_spectrum_ =
        ComplexView2("solprop_native.raman.node_spectrum",
                     problem_.node_chunk * problem_.raman_drives(), problem_.n_to + 1);
    node_raman_convolved_ =
        RealView2("solprop_native.raman.node_convolved",
                  problem_.node_chunk * problem_.raman_drives(), doubled);
    raman_undamped_ =
        RealView2("solprop_native.raman.undamped", problem_.n_raman_terms, doubled);
    raman_kernel_time_ = RealView2("solprop_native.raman.kernel_time",
                                   problem_.raman_kernels(), doubled);
    raman_kernel_spectrum_ = ComplexView2("solprop_native.raman.kernel_spectrum",
                                          problem_.raman_kernels(), problem_.n_to + 1);
    stage_raman_rates_ = RealView2("solprop_native.raman.stage_rates",
                                   problem_.n_stages, problem_.n_raman_terms);
    raman_tensor_weights_ =
        RealView2("solprop_native.raman.tensor_weights", problem_.n_raman_terms, 2);
    cached_raman_rates_.resize(problem_.n_raman_terms);
    if (problem_.n_raman_components > 1) {
      stage_raman_densities_ =
          RealView2("solprop_native.raman.stage_densities", problem_.n_stages,
                    problem_.n_raman_components);
      raman_row_component_ =
          RealView("solprop_native.raman.row_component", problem_.n_raman_terms);
      raman_component_fraction_ = RealView("solprop_native.raman.component_fraction",
                                           problem_.n_raman_components);
      cached_raman_densities_.resize(problem_.n_raman_components);
    }
  }
  if (use_packed_radial_blas()) {
    radial_blas_input_ = RealView2("solprop_native.radial.blas_input", problem_.n_nodes,
                                   2 * problem_.n_to);
    radial_blas_output_ = RealView2("solprop_native.radial.blas_output",
                                    problem_.n_nodes, 2 * problem_.n_to);
  }
  // The hybrid planar kind's slab rule in x, (n_x, n_p) and (n_p, n_x).
  const auto synthesis_rows = problem_.slab_geometry() ? problem_.n_x : transform_rows;
  const auto synthesis_columns =
      problem_.slab_geometry() ? problem_.n_p : problem_.n_modes;
  synthesis_ = RealView2("solprop_native.transform.synthesis", synthesis_rows,
                         synthesis_columns);
  projection_ = RealView2("solprop_native.transform.projection", synthesis_columns,
                          synthesis_rows);
  scaled_synthesis_ = RealView2("solprop_native.transform.scaled_synthesis",
                                transform_rows, problem_.n_modes);
  scaled_projection_ = RealView2("solprop_native.transform.scaled_projection",
                                 problem_.n_modes, transform_rows);
  if (problem_.family == FieldFamily::envelope && use_blas_for_synthesis()) {
    scaled_synthesis_complex_ =
        ComplexView2("solprop_native.transform.scaled_synthesis_complex",
                     transform_rows, problem_.n_modes);
    scaled_projection_complex_ =
        ComplexView2("solprop_native.transform.scaled_projection_complex",
                     problem_.n_modes, transform_rows);
  }
  if (!problem_.projection_coarse_matrix.empty()) {
    projection_coarse_ = RealView2("solprop_native.transform.projection_coarse",
                                   problem_.n_modes, transform_rows);
    scaled_projection_coarse_ =
        RealView2("solprop_native.transform.scaled_projection_coarse", problem_.n_modes,
                  transform_rows);
    if (problem_.family == FieldFamily::envelope && use_blas_for_projection()) {
      scaled_projection_coarse_complex_ =
          ComplexView2("solprop_native.transform.scaled_projection_coarse_complex",
                       problem_.n_modes, transform_rows);
      scaled_projection_stacked_complex_ =
          ComplexView2("solprop_native.transform.scaled_projection_stacked_complex",
                       2 * problem_.n_modes, transform_rows);
      projected_stacked_complex_ =
          ComplexView2("solprop_native.response.projected_stacked_complex", 2 * columns,
                       problem_.n_to);
    } else if (problem_.family != FieldFamily::envelope) {
      scaled_projection_stacked_ =
          RealView2("solprop_native.transform.scaled_projection_stacked",
                    2 * problem_.n_modes, transform_rows);
      projected_stacked_real_ =
          RealView2("solprop_native.response.projected_stacked_real",
                    2 * temporal_columns, problem_.n_to);
    }
  }
  const auto chunk_transform_rows = problem_.node_chunk * problem_.n_pol_transform;
  const auto tail_nodes = problem_.n_nodes % problem_.node_chunk;
  const auto tail_transform_rows = tail_nodes * problem_.n_pol_transform;
  if (use_chunked_blas_for_synthesis()) {
    if (problem_.family == FieldFamily::envelope) {
      chunk_synthesis_complex_ =
          ComplexView2("solprop_native.transform.chunk_synthesis_complex",
                       chunk_transform_rows, problem_.n_modes);
      if (tail_transform_rows != 0)
        tail_synthesis_complex_ =
            ComplexView2("solprop_native.transform.tail_synthesis_complex",
                         tail_transform_rows, problem_.n_modes);
    } else {
      chunk_synthesis_ = RealView2("solprop_native.transform.chunk_synthesis",
                                   chunk_transform_rows, problem_.n_modes);
      if (tail_transform_rows != 0)
        tail_synthesis_ = RealView2("solprop_native.transform.tail_synthesis",
                                    tail_transform_rows, problem_.n_modes);
    }
  }
  if (use_chunked_blas_for_projection()) {
    const bool stacked = !problem_.projection_coarse_matrix.empty();
    const auto projection_rows = stacked ? 2 * problem_.n_modes : problem_.n_modes;
    if (problem_.family == FieldFamily::envelope) {
      if (stacked) {
        chunk_projection_stacked_complex_ =
            ComplexView2("solprop_native.transform.chunk_projection_stacked_complex",
                         projection_rows, chunk_transform_rows);
        if (tail_transform_rows != 0)
          tail_projection_stacked_complex_ =
              ComplexView2("solprop_native.transform.tail_projection_stacked_complex",
                           projection_rows, tail_transform_rows);
      } else {
        chunk_projection_complex_ =
            ComplexView2("solprop_native.transform.chunk_projection_complex",
                         projection_rows, chunk_transform_rows);
        if (tail_transform_rows != 0)
          tail_projection_complex_ =
              ComplexView2("solprop_native.transform.tail_projection_complex",
                           projection_rows, tail_transform_rows);
      }
    } else if (stacked) {
      chunk_projection_stacked_ =
          RealView2("solprop_native.transform.chunk_projection_stacked",
                    projection_rows, chunk_transform_rows);
      if (tail_transform_rows != 0)
        tail_projection_stacked_ =
            RealView2("solprop_native.transform.tail_projection_stacked",
                      projection_rows, tail_transform_rows);
    } else {
      chunk_projection_ = RealView2("solprop_native.transform.chunk_projection",
                                    projection_rows, chunk_transform_rows);
      if (tail_transform_rows != 0)
        tail_projection_ = RealView2("solprop_native.transform.tail_projection",
                                     projection_rows, tail_transform_rows);
    }
  }
  fine_spectrum_ = ComplexView2("solprop_native.fft.fine_spectrum", temporal_columns,
                                fine_spectrum_size());
  if (!problem_.separable_geometry() && problem_.n_omega != fine_spectrum_size())
    fine_gaps_ = IndexView("solprop_native.fft.fine_gaps", 3 * fine_gap_runs().size());
  if (problem_.separable_geometry() && problem_.n_omega != fine_spectrum_size()) {
    std::vector<std::size_t> bins;
    for (const auto& [bin, length] : fine_gap_runs())
      for (std::size_t offset = 0; offset < length; ++offset)
        bins.push_back(bin + offset);
    separable_gaps_ = IndexView("solprop_native.free.fine_gaps", bins.size());
    copy_vector(execution_space_, separable_gaps_, bins);
  }
  projected_spectrum_ = ComplexView2("solprop_native.fft.projected_spectrum",
                                     temporal_columns, projected_spectrum_size());
  if (!problem_.projection_coarse_matrix.empty())
    projected_coarse_spectrum_ =
        ComplexView2("solprop_native.fft.projected_coarse_spectrum", columns,
                     projected_spectrum_size());
  absorber_spectrum_ = ComplexView2("solprop_native.fft.absorber_spectrum",
                                    temporal_columns, projected_spectrum_size());
  if (problem_.family == FieldFamily::field_real) {
    modal_real_ = RealView2("solprop_native.fft.modal_time_real", temporal_columns,
                            problem_.n_to);
    node_real_ = separable_aliased_ ? modal_real_
                                    : RealView2("solprop_native.response.node_real",
                                                chunk_rows, problem_.n_to);
  } else {
    modal_complex_ = ComplexView2("solprop_native.fft.modal_time_complex",
                                  temporal_columns, problem_.n_to);
    node_complex_ = separable_aliased_
                        ? modal_complex_
                        : ComplexView2("solprop_native.response.node_complex",
                                       chunk_rows, problem_.n_to);
  }
  if (problem_.family == FieldFamily::envelope) {
    projected_complex_ = ComplexView2("solprop_native.response.projected_complex",
                                      temporal_columns, problem_.n_to);
    polarisation_complex_ =
        separable_aliased_
            ? projected_complex_
            : ComplexView2("solprop_native.response.polarisation_complex", chunk_rows,
                           problem_.n_to);
    if (!problem_.projection_coarse_matrix.empty())
      projected_coarse_complex_ =
          ComplexView2("solprop_native.response.projected_coarse_complex",
                       temporal_columns, problem_.n_to);
    absorber_complex_ = ComplexView2("solprop_native.absorber.fine_complex",
                                     temporal_columns, problem_.n_to);
  } else {
    projected_real_ = RealView2("solprop_native.response.projected_real",
                                temporal_columns, problem_.n_to);
    polarisation_real_ = separable_aliased_
                             ? projected_real_
                             : RealView2("solprop_native.response.polarisation_real",
                                         chunk_rows, problem_.n_to);
    if (!problem_.projection_coarse_matrix.empty())
      projected_coarse_real_ =
          RealView2("solprop_native.response.projected_coarse_real", temporal_columns,
                    problem_.n_to);
    absorber_real_ =
        RealView2("solprop_native.absorber.fine_real", temporal_columns, problem_.n_to);
  }
  if (held_plasma()) {
    held_current_ = RealView2("solprop_native.held.current", chunk_rows, problem_.n_to);
    held_derivative_ = RealView("solprop_native.held.derivative", problem_.n_to);
    held_inverse_ = RealView("solprop_native.held.inverse", problem_.n_omega);
    const bool radial = held_radial();
    const auto rows = radial ? chunk_rows : temporal_columns;
    for (std::size_t index = 0; index < (radial ? 3U : 2U); ++index) {
      auto& stream = held_streams_[index];
      const auto label =
          std::string("solprop_native.held.stream") + std::to_string(index) + ".";
      if (index < 2)
        stream.node = RealView2(label + "node", chunk_rows, problem_.n_to);
      stream.projected = RealView2(label + "projected", rows, problem_.n_to);
      stream.spectrum = ComplexView2(label + "spectrum", rows, problem_.n_to / 2 + 1);
      if (!problem_.projection_coarse_matrix.empty()) {
        stream.coarse = RealView2(label + "coarse", rows, problem_.n_to);
        stream.coarse_spectrum =
            ComplexView2(label + "coarse_spectrum", rows, problem_.n_to / 2 + 1);
        stream.stacked = RealView2(label + "stacked", 2 * rows, problem_.n_to);
      }
    }
    if (radial) {
      held_nodal_spectrum_ = ComplexView2("solprop_native.held.nodal_spectrum",
                                          chunk_rows, problem_.n_omega);
      held_projected_spectrum_ = ComplexView2("solprop_native.held.projected_spectrum",
                                              problem_.n_columns(), problem_.n_omega);
      held_projection_ = RealView2("solprop_native.held.projection", problem_.n_modes,
                                   problem_.node_chunk * problem_.n_pol_transform);
    }
  }
  absorber_fine_ =
      RealView("solprop_native.absorber.rate_fine", problem_.absorber_fine.size());
  absorber_coarse_ =
      RealView("solprop_native.absorber.rate_coarse", problem_.absorber_coarse.size());
  if (!problem_.spatial_absorber_matrix.empty()) {
    if (problem_.family == FieldFamily::envelope) {
      spatial_absorber_complex_ =
          ComplexView2("solprop_native.absorber.spatial_complex", problem_.n_modes,
                       problem_.n_modes);
      std::vector<Complex> matrix;
      matrix.reserve(problem_.spatial_absorber_matrix.size());
      for (const auto value : problem_.spatial_absorber_matrix)
        matrix.emplace_back(value, 0.0);
      copy_vector(execution_space_, spatial_absorber_complex_, matrix);
    } else {
      spatial_absorber_real_ = RealView2("solprop_native.absorber.spatial_real",
                                         problem_.n_modes, problem_.n_modes);
      copy_vector(execution_space_, spatial_absorber_real_,
                  problem_.spatial_absorber_matrix);
    }
  }
  if (problem_.envelope_plasma) {
    envelope_rule_nodes_ =
        RealView3("solprop_native.envelope.rule_nodes", problem_.n_channels, 4, 512);
    envelope_rule_weights_ =
        RealView3("solprop_native.envelope.rule_weights", problem_.n_channels, 4, 512);
    envelope_moments_ = RealView3("solprop_native.envelope.moments", 2,
                                  problem_.node_chunk, problem_.n_to);
    envelope_rates_ =
        RealView2("solprop_native.envelope.rates", problem_.node_chunk, problem_.n_to);
    envelope_exposure_ = RealView2("solprop_native.envelope.exposure",
                                   problem_.node_chunk, problem_.n_to);
    envelope_failures_ = IndexView("solprop_native.envelope.failures",
                                   problem_.node_chunk * problem_.n_to);
    envelope_status_ = IndexView("solprop_native.envelope.status", 1);
    envelope_host_status_ = Kokkos::View<std::size_t*, Layout, Kokkos::HostSpace>(
        "solprop_native.envelope.host_status", 1);
    envelope_drude_factor_ =
        RealView("solprop_native.envelope.drude_factor", problem_.n_omega);
    envelope_loss_factor_ =
        ComplexView("solprop_native.envelope.loss_factor", problem_.n_omega);
    for (std::size_t index = 0; index < envelope_streams_.size(); ++index) {
      auto& stream = envelope_streams_[index];
      const auto label =
          std::string("solprop_native.envelope.") + (index == 0 ? "drude." : "loss.");
      stream.node = ComplexView2(label + "node", chunk_rows, problem_.n_to);
      stream.projected =
          ComplexView2(label + "projected", temporal_columns, problem_.n_to);
      stream.spectrum =
          ComplexView2(label + "spectrum", temporal_columns, problem_.n_to);
      if (!problem_.projection_coarse_matrix.empty()) {
        stream.coarse = ComplexView2(label + "coarse", temporal_columns, problem_.n_to);
        stream.coarse_spectrum =
            ComplexView2(label + "coarse_spectrum", temporal_columns, problem_.n_to);
      }
      if (!projected_stacked_complex_.empty())
        stream.stacked =
            ComplexView2(label + "stacked", 2 * temporal_columns, problem_.n_to);
    }
  }
  if (problem_.plasma) {
    if (problem_.n_channels > 1 || problem_.envelope_plasma) {
      channel_parameters_ =
          RealView2("solprop_native.plasma.channel_parameters", problem_.n_channels, 9);
      channel_coefficients_ = RealView("solprop_native.plasma.channel_coefficients",
                                       problem_.channel_coefficients.size());
      if (!problem_.envelope_plasma)
        channel_scratch_ =
            decltype(channel_scratch_)("solprop_native.plasma.channel_scratch",
                                       problem_.n_channels, problem_.node_chunk, 4);
    }
    rate_coefficients_ = RealView("solprop_native.plasma.rate_coefficients",
                                  problem_.rate_coefficients.size());
    plasma_statistics_scratch_ =
        RealView2("solprop_native.plasma.statistics_scratch", problem_.n_nodes,
                  1 + 2 * problem_.n_channels);
    if (!problem_.envelope_plasma) {
      // solprop SEP 18: the rule's taps and the run diagnostics' six parts per node.
      plasma_taps_ =
          RealView("solprop_native.plasma.taps", problem_.plasma_taps.size());
      plasma_diagnostics_ = RealView2("solprop_native.plasma.diagnostics",
                                      problem_.n_nodes, plasma_diagnostic_parts);
    }
    if (!problem_.envelope_plasma) {
      // The band rule's team column and its statistics (`team_band_plasma_column`).
      // A device keeps the trapezoid device column's rows, one per node of a chunk,
      // and the band column reuses them. A host team owns one row of each per slot,
      // and at most one team per thread runs at once (solprop SEP 25 section
      // 6.7.5): the rate, exposure and filtered rows for every field plasma (the
      // trapezoid's run diagnostics come from the band statistics), and under the
      // band rule the guard, a mixture's weight and a row per component. The
      // tiled column (solprop SEP 25 section 6.7.6) keeps the device's rows on every
      // build, the weight rows holding the free-current local prefixes, and the tiles'
      // totals by row.
      const bool tiled = band_tiled_ && problem_.plasma_band;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
      band_slots_ = problem_.node_chunk;
      const auto guard_rows = chunk_rows;
      const auto weight_rows = chunk_rows;
      const auto component_rows = chunk_rows;
#else
      if (tiled) {
        band_slots_ = problem_.node_chunk;
      } else {
        band_slots_ = std::min(problem_.node_chunk,
                               static_cast<std::size_t>(std::max(
                                   1, response_execution_space_.concurrency())));
      }
      const bool band = problem_.plasma_band;
      const auto guard_rows = tiled ? chunk_rows : band ? band_slots_ : 0;
      const auto weight_rows = tiled                             ? chunk_rows
                               : band && problem_.n_channels > 1 ? band_slots_
                                                                 : 0;
      const auto component_rows =
          band ? checked_product(band_slots_, problem_.effective_node_pol(),
                                 "band component rows")
               : 0;
#endif
      if (tiled) {
        // The widest tile whose two scratch rows the device's team scratch holds.
        using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
        const auto scratch_limit =
            static_cast<std::size_t>(Policy::scratch_size_max(0));
        std::size_t largest = band_tile_max;
        while (largest > band_tile_min &&
               band_tile_bytes<ExecutionSpace>(largest) > scratch_limit)
          largest /= 2;
        if (band_tile_bytes<ExecutionSpace>(largest) > scratch_limit)
          throw std::length_error("the band tile exceeds the team scratch");
        band_tile_ = band_tile_samples(problem_.n_to, largest);
        band_tile_count_ = band_tile_count(problem_.n_to, band_tile_);
        require_allocation(checked_product(chunk_rows, band_tile_count_, "band tiles"),
                           sizeof(double), "band-tile bytes");
        band_tile_totals_ = RealView2("solprop_native.plasma.band_tile_totals",
                                      chunk_rows, band_tile_count_);
      }
      const auto band_rows = checked_sum(
          checked_sum(checked_product(3, band_slots_, "band rows"), guard_rows,
                      "band rows"),
          checked_sum(weight_rows, component_rows, "band rows"), "band rows");
      require_allocation(checked_product(band_rows, problem_.n_to, "band rows"),
                         sizeof(double), "band-row bytes");
      plasma_rate_ =
          RealView2("solprop_native.plasma.rate", band_slots_, problem_.n_to);
      plasma_exposure_ =
          RealView2("solprop_native.plasma.exposure", band_slots_, problem_.n_to);
      plasma_fraction_ =
          RealView2("solprop_native.plasma.fraction", band_slots_, problem_.n_to);
      if (component_rows != 0) {
        plasma_current_integrand_ = RealView2("solprop_native.plasma.current_integrand",
                                              component_rows, problem_.n_to);
      }
      if (guard_rows != 0)
        plasma_free_current_ =
            RealView2("solprop_native.plasma.free_current", guard_rows, problem_.n_to);
      if (weight_rows != 0)
        plasma_outer_current_ = RealView2("solprop_native.plasma.outer_current",
                                          weight_rows, problem_.n_to);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
      if (!problem_.plasma_band) {
        plasma_tiles_ = (problem_.n_to + plasma_tile_samples - 1) / plasma_tile_samples;
        plasma_tile_totals_ =
            RealView2("solprop_native.plasma.tile_totals", chunk_rows, plasma_tiles_);
      }
#endif
    }
    if (problem_.stash_plasma) {
      stash_candidate_ = RealView2("solprop_native.plasma.stash_candidate",
                                   problem_.n_nodes, 1 + 2 * problem_.n_channels);
      stash_current_ = RealView2("solprop_native.plasma.stash_current",
                                 problem_.n_nodes, 1 + 2 * problem_.n_channels);
    }
  }
  if (problem_.split_step_absorber) {
    coarse_spectrum_ =
        ComplexView2("solprop_native.absorber.coarse_spectrum", temporal_columns, n);
    if (problem_.family == FieldFamily::envelope)
      coarse_time_complex_ = ComplexView2("solprop_native.absorber.coarse_time_complex",
                                          temporal_columns, problem_.n_t);
    else
      coarse_time_real_ = RealView2("solprop_native.absorber.coarse_time_real",
                                    temporal_columns, problem_.n_t);
  }
  if (problem_.general_modal) {
    general_ = std::make_unique<GeneralModalSource>(problem_);
    // The source owns execution-space copies; the host maps are the largest arrays a
    // general guide carries and nothing reads them again.
    std::vector<Complex>().swap(problem_.general_synthesis);
    std::vector<Complex>().swap(problem_.general_projection);
    std::vector<double>().swap(problem_.general_tensor);
    std::vector<Complex>().swap(problem_.general_frequency_factor);
    std::vector<double>().swap(problem_.general_delayed_tensor);
    std::vector<Complex>().swap(problem_.general_delayed_spectrum);
    std::vector<std::size_t>().swap(problem_.general_node_response);
  }
  if (problem_.free_geometry() && !problem_.spatial_spectral_weights.empty())
    far_field_ = std::make_unique<FarFieldPass>(problem_, execution_space_);
  if (problem_.dispersive && problem_.separable_geometry()) {
    // The planar kind's slab rule in x; free space has none.
    const bool slab = problem_.slab_geometry();
    separable_dispersive_ = std::make_unique<SeparableDispersive>(
        problem_, kernel_pool_,
        SeparableDispersive::StageViews{
            fine_spectrum_, projected_spectrum_, node_real_, polarisation_real_,
            node_complex_, polarisation_complex_, slab ? synthesis_ : RealView2{},
            slab ? projection_ : RealView2{}, time_window_, carrier_});
  } else if (problem_.dispersive && !problem_.general_modal) {
    dispersive_ = std::make_unique<DispersiveResponse>(
        problem_, kernel_pool_,
        DispersiveResponse::StageViews{
            fine_spectrum_, projected_spectrum_, projected_coarse_spectrum_, node_real_,
            polarisation_real_, node_complex_, polarisation_complex_, scaled_synthesis_,
            scaled_projection_, scaled_projection_coarse_, time_window_, carrier_});
  }
  if (problem_.dispersive) {
    // The response holds execution-space copies of its layout (the general route's
    // source built its own above).
    std::vector<double>().swap(problem_.dispersive_inputs);
    std::vector<double>().swap(problem_.dispersive_term_multipliers);
    std::vector<double>().swap(problem_.dispersive_quartics);
  }
  if (problem_.coupled_flow) {
    flow_ = std::make_unique<CoupledFlowAction>(problem_, flow_execution_space_);
    std::vector<Complex>().swap(problem_.flow_frozen);
    std::vector<Complex>().swap(problem_.flow_residual_terms);
    std::vector<Complex>().swap(problem_.flow_left);
    std::vector<Complex>().swap(problem_.flow_right);
    std::vector<Complex>().swap(problem_.flow_eigenvalues);
    std::vector<std::size_t>().swap(problem_.flow_route);
    std::vector<Complex>().swap(problem_.flow_fallback_vectors);
    std::vector<Complex>().swap(problem_.flow_fallback_triangular);
  }
}

void GeneralWorkspace::upload() {
  if (problem_.n_raman_components > 1) {
    copy_vector(execution_space_, raman_row_component_, problem_.raman_row_component);
    copy_vector(execution_space_, raman_component_fraction_,
                problem_.raman_component_fraction);
  }
  if (problem_.separable_geometry()) {
    copy_vector(execution_space_, spatial_.square, problem_.spatial_k_squared);
    copy_vector(execution_space_, spatial_.square_low, problem_.spatial_k_squared_low);
    copy_vector(execution_space_, spatial_.numerator, problem_.spatial_numerator);
    copy_vector(execution_space_, spatial_.clamped, problem_.spatial_clamped);
    copy_vector(execution_space_, spatial_.transverse, problem_.spatial_kperp_sq);
    copy_vector(execution_space_, spatial_.transverse_low,
                problem_.spatial_kperp_sq_low);
    copy_vector(execution_space_, spatial_.frame, problem_.spatial_frame);
    copy_vector(execution_space_, spatial_.spectral_rate,
                problem_.spatial_spectral_rate);
    copy_vector(execution_space_, spatial_.absorber, problem_.spatial_absorber_rate);
    copy_vector(execution_space_, spatial_.transverse_rate,
                problem_.spatial_kperp_rate);
    if (problem_.slab_geometry())
      copy_vector(execution_space_, spatial_.kappa, problem_.spatial_kappa_sq);
  } else if (!problem_.flux_k_squared.empty()) {
    copy_vector(execution_space_, flux_.square, problem_.flux_k_squared);
    copy_vector(execution_space_, flux_.square_low, problem_.flux_k_squared_low);
    copy_vector(execution_space_, flux_.transverse, problem_.flux_kperp_sq);
    copy_vector(execution_space_, flux_.transverse_low, problem_.flux_kperp_sq_low);
  }
  if (problem_.n_channels > 1 || problem_.envelope_plasma) {
    copy_vector(execution_space_, channel_parameters_, problem_.channel_parameters);
    copy_vector(execution_space_, channel_coefficients_, problem_.channel_coefficients);
  }
  copy_vector(execution_space_, crop_index_, problem_.crop_index);
  if (!fine_gaps_.empty()) {
    std::vector<std::size_t> runs;
    std::size_t position = 0;
    for (const auto& [bin, length] : fine_gap_runs()) {
      runs.insert(runs.end(), {position, bin, length});
      position += length;
    }
    copy_vector(execution_space_, fine_gaps_, runs);
  }
  if (problem_.envelope_plasma) {
    copy_vector(execution_space_, envelope_rule_nodes_, problem_.envelope_rule_nodes);
    copy_vector(execution_space_, envelope_rule_weights_,
                problem_.envelope_rule_weights);
    copy_vector(execution_space_, envelope_drude_factor_,
                problem_.envelope_drude_factor);
    copy_vector(execution_space_, envelope_loss_factor_, problem_.envelope_loss_factor);
  }
  copy_vector(execution_space_, active_mask_, problem_.active_mask);
  copy_vector(execution_space_, norm_mask_, problem_.norm_mask);
  copy_vector(execution_space_, omega_window_, problem_.omega_window);
  copy_vector(execution_space_, omega_, problem_.omega);
  copy_vector(execution_space_, time_window_, problem_.time_window);
  if (held_plasma()) {
    copy_vector(execution_space_, held_derivative_, problem_.time_window_derivative);
    std::vector<double> inverse(problem_.n_omega, 0.0);
    for (std::size_t index = 0; index < problem_.n_omega; ++index)
      if (problem_.active_mask[index])
        inverse[index] = 1.0 / problem_.omega[index];
    copy_vector(execution_space_, held_inverse_, inverse);
  }
  copy_vector(execution_space_, carrier_, problem_.carrier);
  // A z table's slots start at zero (Kokkos initialises a view) until evaluated.
  lossless_phase_ = false;
  if (!problem_.phi_table) {
    copy_vector(execution_space_, phi_terms_, problem_.phi_terms);
    lossless_phase_ =
        std::all_of(problem_.phi_terms.begin(), problem_.phi_terms.end(),
                    [](const Complex& term) { return term.real() == 0.0; });
  }
  if (!problem_.prefactor_table)
    copy_vector(execution_space_, prefactor_terms_, problem_.prefactor_terms);
  copy_vector(execution_space_, tableau_, problem_.tableau_matrix);
  copy_vector(execution_space_, spatial_spectral_weights_,
              problem_.spatial_spectral_weights);
  copy_vector(execution_space_, spatial_on_axis_, problem_.spatial_on_axis);
  copy_vector(execution_space_, synthesis_, problem_.synthesis_matrix);
  copy_vector(execution_space_, projection_, problem_.projection_matrix);
  if (!problem_.projection_coarse_matrix.empty())
    copy_vector(execution_space_, projection_coarse_,
                problem_.projection_coarse_matrix);
  copy_vector(execution_space_, absorber_fine_, problem_.absorber_fine);
  copy_vector(execution_space_, absorber_coarse_, problem_.absorber_coarse);
  if (problem_.plasma)
    copy_vector(execution_space_, rate_coefficients_, problem_.rate_coefficients);
  if (!plasma_taps_.empty())
    copy_vector(execution_space_, plasma_taps_, problem_.plasma_taps);
  if (problem_.raman) {
    copy_vector(execution_space_, raman_undamped_, problem_.raman_undamped);
    copy_vector(execution_space_, raman_tensor_weights_, problem_.raman_tensor_weights);
    Kokkos::deep_copy(execution_space_, node_raman_drive_, 0.0);
  }
}

// Size from the plan's real-side batch samples, using the shared complete-window
// thresholds in core.hpp. Explicit FFT-pool requests bypass automatic sizing.
// FFTW's factorisation can change with the width; comparisons use solprop's SEP 25
// transform/propagation rounding budgets rather than bitwise agreement.
const ExecutionSpace& GeneralWorkspace::fft_space_for(const std::size_t samples) {
  return fft_pool_.get(CpuOperation::fft, samples);
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
// Measured plans for free space's batched transforms (SEP 28): the transverse pair in
// place on the scratch and, on the complex fields, the chunk's temporal pair, each at
// the plan policy's effort for its shape and width (`fftw_seed_effort`). The scratch
// buffers are cleared after planning overwrote them.
void GeneralWorkspace::seed_free_wisdom() {
  const auto planes = static_cast<int>(transverse_.extent(0));
  const auto n_y = static_cast<int>(transverse_.extent(1));
  const auto n_x = static_cast<int>(transverse_.extent(2));
  for (const int sign : {FFTW_FORWARD, FFTW_BACKWARD})
    seed_fftw_wisdom(fft_space_for(transverse_.size()), transverse_.data(),
                     transverse_.data(), {n_y, n_x}, planes, n_y * n_x, n_y * n_x,
                     sign);
  Kokkos::deep_copy(execution_space_, transverse_, Complex{});
  if (problem_.family != FieldFamily::envelope &&
      problem_.family != FieldFamily::field_analytic)
    return;
  // A fine spectrum wider than n_to is not a plain row pair; the helper leaves it.
  seed_row_fft_wisdom(fft_space_for(modal_complex_.size()), fine_spectrum_,
                      modal_complex_, KokkosFFT::Direction::backward);
  Kokkos::deep_copy(execution_space_, fine_spectrum_, Complex{});
  Kokkos::deep_copy(execution_space_, modal_complex_, Complex{});
  if (problem_.family == FieldFamily::envelope) {
    seed_row_fft_wisdom(fft_space_for(projected_complex_.size()), projected_complex_,
                        projected_spectrum_, KokkosFFT::Direction::forward);
    Kokkos::deep_copy(execution_space_, projected_complex_, Complex{});
    Kokkos::deep_copy(execution_space_, projected_spectrum_, Complex{});
  }
}

// Measured plans for the modal routes' temporal pair, the two transforms of every
// source evaluation: Kokkos-FFT plans them at FFTW_ESTIMATE, which picks markedly
// slower codelets for long rows (a 16384-sample envelope GNLSE ran its pair at about
// twice the measured cost). The plan policy (`fftw_seed_effort`) decides the effort.
void GeneralWorkspace::seed_modal_wisdom() {
  if (problem_.family == FieldFamily::field_real) {
    seed_row_fft_wisdom(fft_space_for(modal_real_.size()), fine_spectrum_, modal_real_,
                        KokkosFFT::Direction::backward);
    Kokkos::deep_copy(execution_space_, modal_real_, 0.0);
  } else {
    seed_row_fft_wisdom(fft_space_for(modal_complex_.size()), fine_spectrum_,
                        modal_complex_, KokkosFFT::Direction::backward);
    Kokkos::deep_copy(execution_space_, modal_complex_, Complex{});
  }
  if (problem_.family == FieldFamily::envelope) {
    seed_row_fft_wisdom(fft_space_for(projected_complex_.size()), projected_complex_,
                        projected_spectrum_, KokkosFFT::Direction::forward);
    Kokkos::deep_copy(execution_space_, projected_complex_, Complex{});
  } else {
    seed_row_fft_wisdom(fft_space_for(projected_real_.size()), projected_real_,
                        projected_spectrum_, KokkosFFT::Direction::forward);
    Kokkos::deep_copy(execution_space_, projected_real_, 0.0);
  }
  Kokkos::deep_copy(execution_space_, fine_spectrum_, Complex{});
  Kokkos::deep_copy(execution_space_, projected_spectrum_, Complex{});
}

// The plasma streams' forward rows, fine and coarse. Most share a shape with the
// projected pair seeded above, whose wisdom FFTW then finds at once; the radial held
// streams' chunk rows do not.
void GeneralWorkspace::seed_stream_wisdom() {
  if (held_plasma())
    for (auto& stream : held_streams_) {
      if (stream.projected.empty())
        continue;
      seed_row_fft_wisdom(fft_space_for(stream.projected.size()), stream.projected,
                          stream.spectrum, KokkosFFT::Direction::forward);
      Kokkos::deep_copy(execution_space_, stream.projected, 0.0);
      Kokkos::deep_copy(execution_space_, stream.spectrum, Complex{});
      if (stream.coarse.empty())
        continue;
      seed_row_fft_wisdom(fft_space_for(stream.coarse.size()), stream.coarse,
                          stream.coarse_spectrum, KokkosFFT::Direction::forward);
      Kokkos::deep_copy(execution_space_, stream.coarse, 0.0);
      Kokkos::deep_copy(execution_space_, stream.coarse_spectrum, Complex{});
    }
  if (problem_.envelope_plasma)
    for (auto& stream : envelope_streams_) {
      seed_row_fft_wisdom(fft_space_for(stream.projected.size()), stream.projected,
                          stream.spectrum, KokkosFFT::Direction::forward);
      Kokkos::deep_copy(execution_space_, stream.projected, Complex{});
      Kokkos::deep_copy(execution_space_, stream.spectrum, Complex{});
      if (stream.coarse.empty())
        continue;
      seed_row_fft_wisdom(fft_space_for(stream.coarse.size()), stream.coarse,
                          stream.coarse_spectrum, KokkosFFT::Direction::forward);
      Kokkos::deep_copy(execution_space_, stream.coarse, Complex{});
      Kokkos::deep_copy(execution_space_, stream.coarse_spectrum, Complex{});
    }
}
#endif

void GeneralWorkspace::plan() {
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  // Seed every measured shape before the first plan exists: a Kokkos-FFT plan made
  // before its wisdom stays at FFTW_ESTIMATE for the run.
  if (problem_.free_geometry())
    seed_free_wisdom();
  else if (!problem_.separable_geometry())
    seed_modal_wisdom();
  seed_stream_wisdom();
#endif
  fft_execution_space_ =
      fft_space_for(std::max(modal_complex_.size(), modal_real_.size()));
  if (problem_.raman) {
    raman_execution_space_ =
        raman_pool_.get(CpuOperation::raman, node_raman_drive_.size());
    const auto raman_tail = problem_.n_nodes % problem_.node_chunk;
    raman_tail_execution_space_ = raman_tail_pool_.get(
        CpuOperation::raman, raman_tail * problem_.raman_drives() * 2 * problem_.n_to);
    raman_kernel_execution_space_ =
        raman_pool_.get(CpuOperation::raman, raman_kernel_time_.size());
    raman_multiply_space_ =
        kernel_pool_.get(CpuOperation::stage, node_raman_spectrum_.size());
    raman_tail_multiply_space_ =
        kernel_pool_.get(CpuOperation::stage,
                         raman_tail * problem_.raman_drives() * (problem_.n_to + 1));
    raman_rebuild_space_ = kernel_pool_.get(
        CpuOperation::response, raman_kernel_time_.size() * problem_.n_raman_terms);
    raman_scale_space_ =
        kernel_pool_.get(CpuOperation::stage, raman_kernel_spectrum_.size());
  }
  if (general_)
    general_->plan();
  if (dispersive_)
    dispersive_->plan(fft_pool_);
  if (separable_dispersive_)
    separable_dispersive_->plan(fft_pool_);
  if (problem_.slab_geometry()) {
    // The y transforms are many short ones; below the threaded-plan threshold they
    // run on one FFTW thread, as every other small plan (`fft_space_for`).
    constexpr int y_axis = -1;
    const auto& space = fft_space_for(transverse_.size());
    slab_forward_ = std::make_unique<SlabPlan>(space, transverse_, transverse_,
                                               KokkosFFT::Direction::forward, y_axis);
    slab_inverse_ = std::make_unique<SlabPlan>(space, transverse_, transverse_,
                                               KokkosFFT::Direction::backward, y_axis);
  } else if (problem_.free_geometry()) {
    const KokkosFFT::axis_type<2> axes{-2, -1};
    transverse_forward_ = std::make_unique<TransversePlan>(
        fft_space_for(transverse_.size()), transverse_, transverse_,
        KokkosFFT::Direction::forward, axes);
    transverse_inverse_ = std::make_unique<TransversePlan>(
        fft_space_for(transverse_.size()), transverse_, transverse_,
        KokkosFFT::Direction::backward, axes);
  }
  constexpr int axis = -1;
  if (held_plasma())
    for (auto& stream : held_streams_) {
      if (stream.projected.empty())
        continue;
      stream.forward = std::make_unique<ForwardRealPlan>(
          fft_space_for(stream.projected.size()), stream.projected, stream.spectrum,
          KokkosFFT::Direction::forward, axis);
      if (!stream.coarse.empty())
        stream.coarse_forward = std::make_unique<ForwardRealPlan>(
            fft_space_for(stream.coarse.size()), stream.coarse, stream.coarse_spectrum,
            KokkosFFT::Direction::forward, axis);
    }
  if (problem_.envelope_plasma)
    for (auto& stream : envelope_streams_) {
      stream.forward = std::make_unique<ComplexPlan>(
          fft_space_for(stream.projected.size()), stream.projected, stream.spectrum,
          KokkosFFT::Direction::forward, axis);
      if (!stream.coarse.empty())
        stream.coarse_forward = std::make_unique<ComplexPlan>(
            fft_space_for(stream.coarse.size()), stream.coarse, stream.coarse_spectrum,
            KokkosFFT::Direction::forward, axis);
    }
  statistics_inverse_plan_ = std::make_unique<ComplexPlan>(
      fft_space_for(statistics_time_.size()), statistics_spectrum_, statistics_time_,
      KokkosFFT::Direction::backward, axis);
  if (problem_.family == FieldFamily::field_real) {
    inverse_real_plan_ = std::make_unique<InverseRealPlan>(
        fft_space_for(modal_real_.size()), fine_spectrum_, modal_real_,
        KokkosFFT::Direction::backward, axis);
  } else {
    inverse_complex_plan_ = std::make_unique<ComplexPlan>(
        fft_space_for(modal_complex_.size()), fine_spectrum_, modal_complex_,
        KokkosFFT::Direction::backward, axis);
  }
  if (problem_.family == FieldFamily::envelope) {
    forward_complex_plan_ = std::make_unique<ComplexPlan>(
        fft_space_for(projected_complex_.size()), projected_complex_,
        projected_spectrum_, KokkosFFT::Direction::forward, axis);
    absorber_complex_plan_ = std::make_unique<ComplexPlan>(
        fft_space_for(absorber_complex_.size()), absorber_complex_, absorber_spectrum_,
        KokkosFFT::Direction::forward, axis);
    if (!problem_.projection_coarse_matrix.empty())
      forward_coarse_complex_plan_ = std::make_unique<ComplexPlan>(
          fft_space_for(projected_coarse_complex_.size()), projected_coarse_complex_,
          projected_coarse_spectrum_, KokkosFFT::Direction::forward, axis);
  } else {
    forward_real_plan_ = std::make_unique<ForwardRealPlan>(
        fft_space_for(projected_real_.size()), projected_real_, projected_spectrum_,
        KokkosFFT::Direction::forward, axis);
    absorber_real_plan_ = std::make_unique<ForwardRealPlan>(
        fft_space_for(absorber_real_.size()), absorber_real_, absorber_spectrum_,
        KokkosFFT::Direction::forward, axis);
    if (!problem_.projection_coarse_matrix.empty())
      forward_coarse_real_plan_ = std::make_unique<ForwardRealPlan>(
          fft_space_for(projected_coarse_real_.size()), projected_coarse_real_,
          projected_coarse_spectrum_, KokkosFFT::Direction::forward, axis);
  }
  if constexpr (cpu_size_policy_enabled) {
    if (problem_.separable_geometry()) {
      const auto full_units = transverse_units_per_chunk();
      const auto tail_units = transverse_units() % full_units;
      const auto rows_per_unit =
          problem_.slab_geometry() ? problem_.n_p : problem_.n_pol_state;
      temporal_tail_rows_ = tail_units * rows_per_unit;
      if (temporal_tail_rows_ != 0) {
        const auto& space = fft_space_for(temporal_tail_rows_ * problem_.n_to);
        const auto rows = fine_spectrum_.extent(0);
        if (problem_.family == FieldFamily::field_real)
          temporal_tail_inverse_real_.create(space, fine_spectrum_, modal_real_, rows,
                                             temporal_tail_rows_,
                                             KokkosFFT::Direction::backward);
        else
          temporal_tail_inverse_complex_.create(space, fine_spectrum_, modal_complex_,
                                                rows, temporal_tail_rows_,
                                                KokkosFFT::Direction::backward);
        if (problem_.family == FieldFamily::envelope)
          temporal_tail_forward_complex_.create(
              space, projected_complex_, projected_spectrum_, rows, temporal_tail_rows_,
              KokkosFFT::Direction::forward);
        else
          temporal_tail_forward_real_.create(
              space, projected_real_, projected_spectrum_, rows, temporal_tail_rows_,
              KokkosFFT::Direction::forward);
      }
    }
  }
  if (problem_.raman) {
    const auto started = std::chrono::steady_clock::now();
    raman_drive_forward_plan_ = std::make_unique<RamanForwardPlan>(
        raman_execution_space_, node_raman_drive_, node_raman_spectrum_,
        KokkosFFT::Direction::forward, axis);
    raman_convolution_inverse_plan_ = std::make_unique<RamanInversePlan>(
        raman_execution_space_, node_raman_spectrum_, node_raman_convolved_,
        KokkosFFT::Direction::backward, axis);
    const auto tail = problem_.n_nodes % problem_.node_chunk;
    if (tail != 0) {
      // Reuse the leading rows of the full chunk buffers. A separate persistent
      // plan avoids transforming stale rows in the final partial node chunk.
      const auto rows = std::make_pair(std::size_t{0}, tail * problem_.raman_drives());
      raman_tail_drive_ = Kokkos::subview(node_raman_drive_, rows, Kokkos::ALL());
      raman_tail_spectrum_ = Kokkos::subview(node_raman_spectrum_, rows, Kokkos::ALL());
      raman_tail_convolved_ =
          Kokkos::subview(node_raman_convolved_, rows, Kokkos::ALL());
      raman_tail_forward_plan_ = std::make_unique<RamanForwardPlan>(
          raman_tail_execution_space_, raman_tail_drive_, raman_tail_spectrum_,
          KokkosFFT::Direction::forward, axis);
      raman_tail_inverse_plan_ = std::make_unique<RamanInversePlan>(
          raman_tail_execution_space_, raman_tail_spectrum_, raman_tail_convolved_,
          KokkosFFT::Direction::backward, axis);
    }
    raman_kernel_forward_plan_ = std::make_unique<RamanForwardPlan>(
        raman_kernel_execution_space_, raman_kernel_time_, raman_kernel_spectrum_,
        KokkosFFT::Direction::forward, axis);
    raman_planning_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
  }
  if (!problem_.split_step_absorber)
    return;
  if (problem_.family == FieldFamily::envelope) {
    coarse_inverse_complex_plan_ = std::make_unique<ComplexPlan>(
        fft_space_for(coarse_time_complex_.size()), coarse_spectrum_,
        coarse_time_complex_, KokkosFFT::Direction::backward, axis);
    coarse_forward_complex_plan_ = std::make_unique<ComplexPlan>(
        fft_space_for(coarse_time_complex_.size()), coarse_time_complex_,
        coarse_spectrum_, KokkosFFT::Direction::forward, axis);
  } else {
    coarse_inverse_real_plan_ = std::make_unique<InverseRealPlan>(
        fft_space_for(coarse_time_real_.size()), coarse_spectrum_, coarse_time_real_,
        KokkosFFT::Direction::backward, axis);
    coarse_forward_real_plan_ = std::make_unique<ForwardRealPlan>(
        fft_space_for(coarse_time_real_.size()), coarse_time_real_, coarse_spectrum_,
        KokkosFFT::Direction::forward, axis);
  }
  if (temporal_tail_rows_ != 0) {
    const auto& space = fft_space_for(temporal_tail_rows_ * problem_.n_t);
    const auto rows = coarse_spectrum_.extent(0);
    if (problem_.family == FieldFamily::envelope) {
      coarse_tail_inverse_complex_.create(space, coarse_spectrum_, coarse_time_complex_,
                                          rows, temporal_tail_rows_,
                                          KokkosFFT::Direction::backward);
      coarse_tail_forward_complex_.create(space, coarse_time_complex_, coarse_spectrum_,
                                          rows, temporal_tail_rows_,
                                          KokkosFFT::Direction::forward);
    } else {
      coarse_tail_inverse_real_.create(space, coarse_spectrum_, coarse_time_real_, rows,
                                       temporal_tail_rows_,
                                       KokkosFFT::Direction::backward);
      coarse_tail_forward_real_.create(space, coarse_time_real_, coarse_spectrum_, rows,
                                       temporal_tail_rows_,
                                       KokkosFFT::Direction::forward);
    }
  }
}

void GeneralWorkspace::set_initial(const std::vector<Complex>& field,
                                   const std::vector<Complex>& noise) {
  require_open();
  require_extent(field.size(), problem_.n_columns() * problem_.n_omega,
                 "initial field");
  if (!noise.empty())
    require_extent(noise.size(), field.size(), "noise");
  copy_vector(execution_space_, current_, field);
  if (noise.empty()) {
    if (problem_.modified_noise)
      Kokkos::deep_copy(execution_space_, noise_, Complex{});
    has_noise_ = false;
  } else {
    if (!problem_.modified_noise)
      throw std::invalid_argument(
          "a native noise field is valid only for modified noise");
    copy_vector(execution_space_, noise_, noise);
    has_noise_ = true;
  }
  mask(current_);
  if (!stash_current_.empty())
    Kokkos::deep_copy(execution_space_, stash_current_, 0.0);
  execution_space_.fence("solprop_native set_initial");
  initialized_ = true;
  rk_source_evaluations_ = 0;
  table_ready_ = false;
  stages_ready_ = false;
  stash_is_current_ = false;
  quadrature_error_ = Kokkos::Experimental::quiet_NaN_v<double>;
  quadrature_stale_ = false;
  Kokkos::deep_copy(execution_space_, quadrature_scores_,
                    Kokkos::Experimental::quiet_NaN_v<double>);
  clear_fsal();
}

void GeneralWorkspace::require_trial_scalars(const TrialScalars& scalars) const {
  const auto checked_values = [](const std::vector<double>& values, const char* name,
                                 bool nonnegative, bool fraction) {
    for (double value : values)
      if (!std::isfinite(value) || (nonnegative && value < 0.0) ||
          (fraction && value > 1.0))
        throw std::invalid_argument(std::string(name) + " contains an invalid value");
  };
  if (!scalars.kerr_density.empty())
    require_extent(scalars.kerr_density.size(), problem_.n_stages, "kerr_density");
  checked_values(scalars.kerr_density, "kerr_density", false, false);
  if (problem_.plasma &&
      (problem_.n_channels > 1 || !scalars.plasma_densities.empty() ||
       !scalars.plasma_preionisation.empty())) {
    const auto count =
        checked_product(problem_.n_stages, problem_.n_channels, "stage channel values");
    require_extent(scalars.plasma_densities.size(), count, "plasma_densities");
    require_extent(scalars.plasma_preionisation.size(), count, "plasma_preionisation");
    checked_values(scalars.plasma_densities, "plasma_densities", true, false);
    checked_values(scalars.plasma_preionisation, "plasma_preionisation", true, true);
  }
  if (problem_.raman &&
      (problem_.n_raman_components > 1 || !scalars.raman_densities.empty())) {
    require_extent(scalars.raman_densities.size(),
                   checked_product(problem_.n_stages, problem_.n_raman_components,
                                   "stage Raman values"),
                   "raman_densities");
    checked_values(scalars.raman_densities, "raman_densities", true, false);
  }
  if (const auto* weights = dispersive_weights())
    weights->require_densities(scalars.dispersive_densities);
  else if (!scalars.dispersive_densities.empty())
    throw std::invalid_argument("dispersive_densities weigh SEP 6's dispersive "
                                "response, and this problem has none");
  if (!general_ && !scalars.transform_weights.empty())
    throw std::invalid_argument("transform_weights weight a general modal guide's "
                                "moving maps, and this problem has none");
  require_extent(scalars.phi_delta.size(),
                 (problem_.n_stages + 1) * problem_.n_phi_terms, "phi_delta");
  require_extent(scalars.phi_at.size(), problem_.n_stages * problem_.n_phi_terms,
                 "phi_at");
  require_extent(scalars.prefactor.size(),
                 problem_.n_stages * problem_.n_prefactor_terms, "prefactor");
  require_extent(scalars.density.size(), problem_.n_stages, "density");
  require_extent(scalars.taper.size(), problem_.n_stages, "taper");
  if (problem_.plasma)
    require_extent(scalars.preionisation.size(), problem_.n_stages, "preionisation");
  if (problem_.raman) {
    require_extent(scalars.raman_rates.size(),
                   problem_.n_stages * problem_.n_raman_terms, "raman_rates");
    if (!std::all_of(
            scalars.raman_rates.begin(), scalars.raman_rates.end(),
            [](const double value) { return std::isfinite(value) && value >= 0.0; }))
      throw std::invalid_argument("Raman decay rates must be finite and non-negative");
  }
}

void GeneralWorkspace::upload_trial(const TrialScalars& scalars) {
  // Fill the next staging buffer at each view's place in the packed buffer, then copy
  // it across once (a host build's mirror is the buffer itself).
  auto staging = stage_staging_[stage_staging_next_];
  stage_staging_next_ = (stage_staging_next_ + 1) % stage_staging_.size();
  const auto place = [&](const auto& view, const std::vector<double>& source) {
    if (view.size() != source.size())
      throw std::invalid_argument("fixed problem array has the wrong extent");
    double* const into = staging.data() + (view.data() - stage_scalars_.data());
    std::copy(source.begin(), source.end(), into);
  };
  const auto fill = [&](const auto& view, const double value) {
    double* const into = staging.data() + (view.data() - stage_scalars_.data());
    std::fill(into, into + view.size(), value);
  };
  if (staging.data() != stage_scalars_.data())
    // A device's staging buffer starts from the last values uploaded, so that a view
    // this trial does not set keeps them, as the separate copies did.
    std::copy(stage_staging_[(stage_staging_next_ + 2) % stage_staging_.size()].data(),
              stage_staging_[(stage_staging_next_ + 2) % stage_staging_.size()].data() +
                  staging.size(),
              staging.data());
  place(stage_phi_delta_, scalars.phi_delta);
  // Rows of equal weights give bitwise equal phases (a zero weight's term is skipped).
  const auto n_phi = problem_.n_phi_terms;
  const auto same_row = [&](const std::size_t row) {
    return std::equal(
        scalars.phi_delta.begin() + static_cast<std::ptrdiff_t>(row * n_phi),
        scalars.phi_delta.begin() + static_cast<std::ptrdiff_t>((row + 1) * n_phi),
        scalars.phi_delta.begin() + static_cast<std::ptrdiff_t>((row - 1) * n_phi));
  };
  phase_repeats_.assign(problem_.n_stages, false);
  for (std::size_t stage = 1; stage < problem_.n_stages; ++stage)
    phase_repeats_[stage] = same_row(stage);
  whole_repeats_ = problem_.n_stages > 0 && same_row(problem_.n_stages);
  entered_stage_ = std::numeric_limits<std::size_t>::max();
  place(stage_phi_at_, scalars.phi_at);
  place(stage_prefactor_, scalars.prefactor);
  place(stage_density_, scalars.density);
  place(stage_kerr_density_,
        scalars.kerr_density.empty() ? scalars.density : scalars.kerr_density);
  if (problem_.plasma) {
    if (scalars.plasma_densities.empty()) {
      place(stage_plasma_density_, scalars.density);
      place(stage_channel_density_, scalars.density);
      place(stage_channel_preionisation_, scalars.preionisation);
    } else {
      place(stage_channel_density_, scalars.plasma_densities);
      place(stage_channel_preionisation_, scalars.plasma_preionisation);
      // The sole channel already has stage-major contiguous scalar vectors.
      // Multiple channels consume their own matrices and need no extracted copy.
      if (problem_.n_channels == 1) {
        place(stage_plasma_density_, scalars.plasma_densities);
        place(stage_preionisation_, scalars.plasma_preionisation);
      }
    }
  }
  if (problem_.raman) {
    if (scalars.raman_densities.empty())
      place(stage_raman_density_, scalars.density);
    else if (problem_.n_raman_components == 1)
      place(stage_raman_density_, scalars.raman_densities);
    else
      fill(stage_raman_density_, 1.0);
  }
  place(stage_taper_, scalars.taper);
  if (problem_.plasma && scalars.plasma_preionisation.empty())
    place(stage_preionisation_, scalars.preionisation);
  Kokkos::deep_copy(execution_space_, stage_scalars_, staging);
  if (auto* weights = dispersive_weights())
    weights->set_densities(scalars.dispersive_densities);
  if (problem_.raman) {
    copy_vector(execution_space_, stage_raman_rates_, scalars.raman_rates);
    if (problem_.n_raman_components > 1)
      copy_vector(execution_space_, stage_raman_densities_, scalars.raman_densities);
  }
}

void GeneralWorkspace::prepare_raman(const std::vector<double>& rates,
                                     const std::vector<double>& densities) {
  require_open();
  if (!problem_.raman)
    throw std::logic_error("this state has no Raman workspace");
  require_extent(rates.size(), problem_.n_raman_terms, "Raman decay rates");
  if (!std::all_of(rates.begin(), rates.end(), [](const double value) {
        return std::isfinite(value) && value >= 0.0;
      }))
    throw std::invalid_argument("Raman decay rates must be finite and non-negative");
  auto host = Kokkos::create_mirror_view(stage_raman_rates_);
  for (std::size_t term = 0; term < problem_.n_raman_terms; ++term)
    host(0, term) = rates[term];
  Kokkos::deep_copy(execution_space_, stage_raman_rates_, host);
  if (problem_.n_raman_components > 1) {
    require_extent(densities.size(), problem_.n_raman_components,
                   "Raman amplitude densities");
    if (!std::all_of(densities.begin(), densities.end(),
                     [](double value) { return std::isfinite(value) && value >= 0.0; }))
      throw std::invalid_argument(
          "Raman amplitude densities must be finite and non-negative");
    auto density_host = Kokkos::create_mirror_view(stage_raman_densities_);
    for (std::size_t component = 0; component < densities.size(); ++component)
      density_host(0, component) = densities[component];
    Kokkos::deep_copy(execution_space_, stage_raman_densities_, density_host);
  }
  prepare_raman_stage(0, rates.data(), densities.data());
  execution_space_.fence("solprop_native Raman diagnostic boundary");
}

void GeneralWorkspace::prepare_raman_stage(const std::size_t stage, const double* rates,
                                           const double* densities) {
  bool unchanged = raman_kernel_ready_;
  for (std::size_t term = 0; term < problem_.n_raman_terms; ++term)
    unchanged = unchanged && cached_raman_rates_[term] == rates[term];
  if (problem_.n_raman_components > 1)
    for (std::size_t component = 0; component < problem_.n_raman_components;
         ++component)
      unchanged =
          unchanged && cached_raman_densities_[component] == densities[component];
  if (unchanged)
    return;
  for (std::size_t term = 0; term < problem_.n_raman_terms; ++term)
    cached_raman_rates_[term] = rates[term];
  if (problem_.n_raman_components > 1)
    for (std::size_t component = 0; component < problem_.n_raman_components;
         ++component)
      cached_raman_densities_[component] = densities[component];

  Kokkos::Profiling::pushRegion("solprop_native.raman.kernel_rebuild");
  const auto undamped = raman_undamped_;
  const auto weights = raman_tensor_weights_;
  const auto kernel = raman_kernel_time_;
  const auto stage_rates = stage_raman_rates_;
  const auto amplitudes = stage_raman_densities_;
  const auto row_component = raman_row_component_;
  const auto fractions = raman_component_fraction_;
  const bool multiple = problem_.n_raman_components > 1;
  const bool tensor = problem_.effective_node_pol() == 2;
  const auto terms = problem_.n_raman_terms;
  const auto doubled = 2 * problem_.n_to;
  const double fraction = problem_.raman_fraction;
  const double dt = problem_.dt_fine;
  Kokkos::parallel_for(
      "solprop_native.raman.kernel_rebuild",
      Range2(raman_rebuild_space_, {0, 0}, {problem_.raman_kernels(), doubled}),
      KOKKOS_LAMBDA(const std::size_t channel, const std::size_t time) {
        double total = 0.0;
        const double delay = static_cast<double>(time) * dt;
        if (multiple) {
          for (std::size_t component = 0; component < amplitudes.extent(1);
               ++component) {
            double response = 0.0;
            for (std::size_t term = 0; term < terms; ++term) {
              if (static_cast<std::size_t>(row_component(term)) == component) {
                const double damped = undamped(term, time) *
                                      Kokkos::exp(-stage_rates(stage, term) * delay);
                response += tensor ? weights(term, channel) * damped : damped;
              }
            }
            total += amplitudes(stage, component) * fractions(component) * response;
          }
          kernel(channel, time) = total;
        } else {
          for (std::size_t term = 0; term < terms; ++term) {
            const double damped =
                undamped(term, time) * Kokkos::exp(-stage_rates(stage, term) * delay);
            total += tensor ? weights(term, channel) * damped : damped;
          }
          kernel(channel, time) = fraction * total;
        }
      });
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("solprop_native.raman.kernel_fft");
  execute_raman(*raman_kernel_forward_plan_, raman_kernel_time_, raman_kernel_spectrum_,
                KokkosFFT::Normalization::none);
  const auto spectrum = raman_kernel_spectrum_;
  Kokkos::parallel_for(
      "solprop_native.raman.kernel_dt_scale",
      Range2(raman_scale_space_, {0, 0}, {problem_.raman_kernels(), problem_.n_to + 1}),
      KOKKOS_LAMBDA(const std::size_t channel, const std::size_t index) {
        spectrum(channel, index) *= dt;
      });
  Kokkos::Profiling::popRegion();

  raman_kernel_ready_ = true;
  ++raman_kernel_transforms_;
}

void GeneralWorkspace::prepare_trial(const double dz, const TrialScalars& scalars) {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  require_trial_scalars(scalars);
  coarse_pending_ = false;
  upload_trial(scalars);
  if (flow_)
    flow_->set_trial(scalars.flow_section, scalars.flow_weights);
  if (general_)
    general_->set_weights(scalars.transform_weights);
  std::size_t first = 0;
  if (has_recycled_) {
    reuse_fsal();
    first = 1;
  }
  // A stage's finish runs in the next stage's entry kernel (solprop SEP 25 section
  // 6.7) unless a coupled flow carries the stage back first, or the stage is the last.
  finish_pending_ = false;
  const bool deferrable = !flow_ && !problem_.separable_geometry();
  defer_quadrature_ = deferred_quadrature_eligible();
  for (std::size_t stage = first; stage < problem_.n_stages; ++stage) {
    try {
      if (problem_.raman)
        prepare_raman_stage(
            stage, scalars.raman_rates.data() + stage * problem_.n_raman_terms,
            problem_.n_raman_components > 1
                ? scalars.raman_densities.data() + stage * problem_.n_raman_components
                : nullptr);
      defer_finish_ = deferrable && stage + 1 < problem_.n_stages;
      project_coarse_ = !defer_quadrature_ || stage + 1 == problem_.n_stages;
      evaluate_stage(stage, dz);
    } catch (...) {
      defer_finish_ = false;
      finish_pending_ = false;
      defer_quadrature_ = false;
      project_coarse_ = true;
      coarse_pending_ = false;
      throw;
    }
  }
  defer_finish_ = false;
  defer_quadrature_ = false;
  project_coarse_ = true;
  if (finish_pending_)
    throw std::logic_error("a deferred stage finish was never run");
  stages_ready_ = true;

  batch_prepared_ = true;
}

std::vector<double> GeneralWorkspace::trial_step(const double dz,
                                                 const TrialScalars& scalars) {
  prepare_trial(dz, scalars);
  if (flow_)
    flow_->carry_step(current_, solution_, error_);
  const auto current = current_;
  const auto carried = flow_ ? flow_->carried_state() : current_;
  const auto solution = flow_ ? flow_->carried_solution() : solution_;
  const auto error = flow_ ? flow_->carried_error() : error_;
  const auto whole = whole_propagator_;
  const auto advanced = advanced_;
  const auto residual = residual_;
  const auto in_norm = norm_mask_;
  const auto partials = partials_;
  const auto count = problem_.n_columns();
  const auto n = problem_.n_omega;
  const auto slots = reduction_slots_;
  accept_update(
      execution_space_, count, n, dz,
      KOKKOS_LAMBDA(const std::size_t column) {
        const auto slot = column * slots;
        return UpdateColumn{current.data() + column * current.stride(0),
                            solution.data() + column * solution.stride(0),
                            error.data() + column * error.stride(0),
                            whole.data() + column * whole.stride(0),
                            advanced.data() + column * advanced.stride(0),
                            residual.data() + column * residual.stride(0),
                            in_norm.data(),
                            partials.data() + slot,
                            partials.data() + partials.stride(0) + slot,
                            partials.data() + 2 * partials.stride(0) + slot,
                            carried.data() + column * carried.stride(0)};
      },
      "solprop_native.step.accept_error_partials", slots);
  return collect_trial();
}

// solprop's low-call Adams march (SEP 13, SEP 14 phase 2): one source evaluation at a
// field given in the step's anchor frame, `U(anchor, position) N(U(position, anchor)
// v)`. It is stage zero of a trial whose stage-zero scalars are the evaluation
// position's: the combination with no earlier stage carries the input to the position
// and the finish brings the source back, so the shared stage pipeline, its responses
// and its in-stage absorber are the march's too. The accepted state is held in
// `advanced_`, which only the Runge-Kutta accept writes, while the input takes its
// place.
std::vector<Complex> GeneralWorkspace::adams_source(const std::vector<Complex>& input,
                                                    const TrialScalars& scalars) {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  // Every route's stage source serves the Adams march (solprop SEP 16 phase 3): the
  // separable stage takes its flow from the stage's own Phi increment, as the modal one
  // does. Beside a coupled flow the sample is the material source alone (solprop SEP
  // 16 phase 5): the exponential formulation carries the residual in its compensation
  // and the flow in its moments, so the stage runs without the flow's pipeline.
  require_extent(input.size(), problem_.n_columns() * problem_.n_omega, "Adams input");
  // The flow is set aside for the evaluation and put back whatever happens.
  auto held_flow = std::exchange(flow_, nullptr);
  struct RestoreFlow {
    std::unique_ptr<CoupledFlowAction>& slot;
    std::unique_ptr<CoupledFlowAction>& held;
    // NOLINTNEXTLINE(modernize-use-equals-default): restores the slot (see FlowAside).
    ~RestoreFlow() { slot = std::move(held); }
  } restore{flow_, held_flow};
  require_trial_scalars(scalars);
  upload_trial(scalars);
  if (general_)
    general_->set_weights(scalars.transform_weights);
  Kokkos::deep_copy(execution_space_, advanced_, current_);
  copy_vector(execution_space_, current_, input);
  if (problem_.raman)
    prepare_raman_stage(0, scalars.raman_rates.data(),
                        problem_.n_raman_components > 1 ? scalars.raman_densities.data()
                                                        : nullptr);
  evaluate_stage_source(0, 0.0);
  Kokkos::deep_copy(execution_space_, current_, advanced_);
  stages_ready_ = true;
  // No Runge-Kutta stage of this state exists to recycle.
  pending_fsal_ = false;
  has_recycled_ = false;
  return stage_to_host(0);
}

// Adopt an Adams step's end state, given on the host before its boundary treatments:
// the shared finish masks it, or applies the split-step absorber, and makes it current.
void GeneralWorkspace::adopt(const std::vector<Complex>& field, const double dz) {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  require_extent(field.size(), problem_.n_columns() * problem_.n_omega, "Adams state");
  copy_vector(execution_space_, advanced_, field);
  finish(dz);
  // The stash a response pass left is the predicted field's, not this one's.
  stash_is_current_ = false;
}

// --- solprop SEP 24 phase 3: the low-storage RK4IP march -----------------------------
//
// Classical RK4 in the interaction picture anchored at the step's midpoint (solprop's
// backends/low_storage.py, SEP 24 section 4.3). The state is the register A; B
// accumulates the bracket of the accepted step and C holds the next pass's lab-frame
// input, unless it is elided (solprop's `low_storage_stage_register`): then each of the
// first three finishes writes that input straight into the transverse scratch. A step
// is four passes, each forming its input in the route's scratch, running the stage core
// the Runge-Kutta march shares (`separable_core`, `modal_core`; SEP 24 Q5) and folding
// the source into the registers per element:
//
//   pass 1, F1 at z_n:  A <- P A; B <- A + h/6 P S; C <- A + h/2 P S
//   pass 2, F2 at z_m:  B <- B + h/3 S; C <- A + h/2 S
//   pass 3, F3 at z_m:  B <- B + h/3 S; C <- Q (A + h S)
//   pass 4, F4 at end:  A <- Q B + h/6 S
//
// Every pass reads the source at its own position, so nothing carries it back to an
// anchor, and only forward flows are applied. The step is accepted in place; the
// caller's finish masks it or applies the split-step collar.

namespace {

constexpr int fold_complete = 0;
constexpr int fold_exchange = 1;
constexpr int fold_second = 2;
constexpr int term_none = 0;
constexpr int term_material = 1;
constexpr int term_absorber = 2;

// The scalar row each pass reads: the step's start, its midpoint twice, its end.
constexpr std::size_t low_storage_rows[4] = {0, 1, 1, 2};

// One evaluation's fold into the registers at one element: solprop's
// `bodies.low_storage_fold`, the finish column of SEP 24 section 4.3. A pass that
// evaluates twice -- the material response, then the in-stage absorber's term, from the
// same input -- folds the first at the exchange (pass 3's C left unflowed) and the
// second after it, whose sum is the pass's.
KOKKOS_INLINE_FUNCTION void
low_storage_fold(const int stage, const int mode, const double step,
                 const Complex source, Complex& state, Complex& accumulator,
                 Complex& stage_register, const Complex first, const Complex second) {
  if (stage == 0) {
    const Complex slope = first * source;
    if (mode == fold_second) {
      accumulator = accumulator + (step / 6.0) * slope;
      stage_register = stage_register + (0.5 * step) * slope;
      return;
    }
    const Complex moved = first * state;
    state = moved;
    accumulator = moved + (step / 6.0) * slope;
    stage_register = moved + (0.5 * step) * slope;
    return;
  }
  if (stage == 1) {
    accumulator = accumulator + (step / 3.0) * source;
    stage_register =
        (mode == fold_second ? stage_register : state) + (0.5 * step) * source;
    return;
  }
  if (stage == 2) {
    accumulator = accumulator + (step / 3.0) * source;
    if (mode == fold_complete)
      stage_register = second * (state + step * source);
    else if (mode == fold_exchange)
      stage_register = state + step * source;
    else
      stage_register = second * (stage_register + step * source);
    return;
  }
  if (mode == fold_second) {
    state = state + (step / 6.0) * source;
    return;
  }
  state = second * accumulator + (step / 6.0) * source;
}

// A register traversal over the state's elements. The last fold of a step also reduces
// the accepted state's squared norm per column into `partials(0, column)`, one team per
// column as the Runge-Kutta accept reduces its own (its sums associate as the team
// splits them, solprop SEP 25).
template <bool Crossing = false, class Element>
void register_traversal(const char* label, const ExecutionSpace& execution,
                        const std::size_t columns, const std::size_t n,
                        const RealView2& partials, const bool reduce,
                        const Element& element) {
  const auto traversal_for = [](const char* name, const ExecutionSpace& space,
                                const std::size_t rows, const std::size_t width,
                                const auto& body) {
    if constexpr (Crossing)
      crossing_for(name, space, rows, width, body);
    else
      cartesian_for(name, space, rows, width, body);
  };
  if (!reduce) {
    traversal_for(
        label, execution, columns, n,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          static_cast<void>(element(column, index));
        });
    return;
  }
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  if constexpr (Crossing) {
    // Blocks of adjacent columns, each column's sum in ascending bin order.
    constexpr std::size_t block = crossing_tile_columns;
    constexpr std::size_t bins = crossing_tile_bins;
    const std::size_t blocks = (columns + block - 1) / block;
    Kokkos::parallel_for(
        label, Kokkos::RangePolicy<ExecutionSpace>(execution, 0, blocks),
        KOKKOS_LAMBDA(const std::size_t which) {
          const std::size_t first = which * block;
          const std::size_t count = columns - first < block ? columns - first : block;
          double totals[block] = {};
          for (std::size_t start = 0; start < n; start += bins) {
            const std::size_t stop = n - start < bins ? n : start + bins;
            for (std::size_t index = start; index < stop; ++index)
              for (std::size_t offset = 0; offset < count; ++offset)
                totals[offset] += element(first + offset, index);
          }
          for (std::size_t offset = 0; offset < count; ++offset)
            partials(0, first + offset) = totals[offset];
        });
    return;
  }
#endif
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  using Member = Policy::member_type;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const Policy policy(execution, static_cast<int>(columns),
                      SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
  Kokkos::parallel_for(
      label, policy, KOKKOS_LAMBDA(const Member& team) {
        const auto column = static_cast<std::size_t>(team.league_rank());
        double total = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n)),
            [&](const int index, double& update) {
              update += element(column, static_cast<std::size_t>(index));
            },
            total);
        Kokkos::single(Kokkos::PerTeam(team), [&]() { partials(0, column) = total; });
      });
#else
  const Policy policy(execution, static_cast<int>(columns),
                      host_reduction_team_size(execution, columns, n));
  Kokkos::parallel_for(
      label, policy, KOKKOS_LAMBDA(const Member& team) {
        const auto column = static_cast<std::size_t>(team.league_rank());
        double total = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, n),
            [&](const std::size_t index, double& update) {
              update += element(column, index);
            },
            total);
        Kokkos::single(Kokkos::PerTeam(team), [&] { partials(0, column) = total; });
      });
#endif
}

} // namespace

double GeneralWorkspace::low_storage_step(const double dz, const TrialScalars& scalars,
                                          const bool rebuild_table) {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (!problem_.low_storage)
    throw std::logic_error("this native state was not built for the low-storage march");
  if (!(dz > 0.0) || !std::isfinite(dz))
    throw std::invalid_argument("the low-storage step must be positive and finite");
  if (problem_.n_stages < 3)
    throw std::invalid_argument("the low-storage march reads three scalar rows");
  if (rebuild_table && problem_.low_storage_flows != 1)
    throw std::logic_error("only a constant operator's table is rebuilt");
  if (problem_.low_storage_flows == 1 && !rebuild_table && !table_ready_)
    throw std::logic_error("the low-storage table has not been built since the initial "
                           "field was loaded");
  require_trial_scalars(scalars);
  upload_trial(scalars);
  if (general_)
    general_->set_weights(scalars.transform_weights);
  low_storage_flows_formed(rebuild_table);
  if (rebuild_table)
    table_ready_ = true;
  for (std::size_t pass = 0; pass < 4; ++pass) {
    const auto row = low_storage_rows[pass];
    if (problem_.raman)
      prepare_raman_stage(
          row, scalars.raman_rates.data() + row * problem_.n_raman_terms,
          problem_.n_raman_components > 1
              ? scalars.raman_densities.data() + row * problem_.n_raman_components
              : nullptr);
    if (problem_.separable_geometry())
      low_storage_separable_pass(pass, dz);
    else
      low_storage_modal_pass(pass, dz);
  }
  // No Runge-Kutta stage exists to observe or recycle.
  stages_ready_ = false;
  stash_is_current_ = false;
  execution_space_.fence("solprop_native low-storage step");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, partials_);
  double total = 0.0;
  for (std::size_t column = 0; column < problem_.n_columns(); ++column)
    total += host(0, column);
  return total;
}

// The step's held flows: a constant operator's table `exp(h_t D / 2)`, built from the
// table's interval (scalar row 2 of `phi_delta`) when the caller's key changed, or a
// varying operator's `P` and `Q` from rows 0 and 1, every step. Recomputed flows are
// formed in the finishes that apply them and hold nothing.
void GeneralWorkspace::low_storage_flows_formed(const bool rebuild_table) {
  const auto flows = low_storage_flows_;
  const auto count = problem_.low_storage_flows;
  if (count == 0 || (count == 1 && !rebuild_table))
    return;
  const auto n = problem_.n_omega;
  if (problem_.separable_geometry()) {
    const auto spatial = spatial_;
    const auto delta = stage_phi_delta_;
    const auto active = active_mask_;
    const auto n_pol = problem_.n_pol_state;
    cartesian_for(
        "solprop_native.low_storage.separable_table", execution_space_,
        problem_.n_columns(), n,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
          const auto values = spatial.value(column / n_pol, omega, active(omega) != 0U);
          flows(0, column, omega) = propagator_exp(delta(2) * values.linear);
        });
    return;
  }
  const auto terms = stage_terms(0);
  const auto n_phi = problem_.n_phi_terms;
  cartesian_for(
      "solprop_native.low_storage.modal_flows", execution_space_, problem_.n_columns(),
      n, KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        if (count == 1) {
          flows(0, column, index) =
              propagator_exp(terms.phase_sum(2 * n_phi, column, index));
        } else {
          flows(0, column, index) = propagator_exp(terms.phase_sum(0, column, index));
          flows(1, column, index) =
              propagator_exp(terms.phase_sum(n_phi, column, index));
        }
      });
}

// One pass on a separable geometry (free space and the hybrid planar kind). Its input
// is the register A (pass 1) or C, with the noise draw at the pass's position, in the
// transverse scratch -- already written by the previous finish when C is elided. Under
// an in-stage absorber the pass evaluates twice from the same input: the material
// source, then the absorber's term, the exchange between them taking the material
// output out of the scratch and re-forming the input (SEP 24 section 4.3).
void GeneralWorkspace::low_storage_separable_pass(const std::size_t pass,
                                                  const double dz) {
  const bool material = problem_.nonlinear || problem_.plasma || problem_.raman ||
                        separable_dispersive_ != nullptr;
  const bool absorber = problem_.in_stage_absorber;
  const bool elided = !problem_.low_storage_stage_register;
  const auto row = low_storage_rows[pass];
  if ((material || absorber) && (pass == 0 || !elided))
    low_storage_separable_input(pass);
  if (material && absorber) {
    separable_core(row, true);
    low_storage_separable_fold(pass, fold_exchange, term_material, dz);
    separable_core(row, false);
    low_storage_separable_fold(pass, fold_second, term_absorber, dz);
  } else if (material || absorber) {
    separable_core(row, material);
    low_storage_separable_fold(pass, fold_complete,
                               material ? term_material : term_absorber, dz);
  } else {
    low_storage_separable_fold(pass, fold_complete, term_none, dz);
  }
}

// Each column's masked squared norm of `state` into `partials(0, column)`, in a team
// per column: a device's final fold traverses like the others and reduces here, over
// one volume, instead of reducing inside the fold's crossing (SEP 28).
void column_norms(const char* label, const ExecutionSpace& execution,
                  const ComplexView2& state, const ByteView& in_norm,
                  const RealView2& partials) {
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  using Member = Policy::member_type;
  const auto n = state.extent(1);
  Kokkos::parallel_for(
      label, Policy(execution, static_cast<int>(state.extent(0)), 128),
      KOKKOS_LAMBDA(const Member& team) {
        const auto column = static_cast<std::size_t>(team.league_rank());
        double total = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n)),
            [&](const int index, double& update) {
              const auto omega = static_cast<std::size_t>(index);
              if (in_norm(omega) != 0U)
                update += squared_abs(state(column, omega));
            },
            total);
        Kokkos::single(Kokkos::PerTeam(team), [&] { partials(0, column) = total; });
      });
}

void GeneralWorkspace::low_storage_separable_input(const std::size_t pass) {
  const auto source = pass == 0 ? current_ : register_stage_;
  const auto spatial = spatial_;
  const auto at = stage_phi_at_;
  const auto active = active_mask_;
  const auto noise = noise_;
  const auto scratch = transverse_;
  const auto n_pol = problem_.n_pol_state;
  const auto layout = transverse_layout();
  const bool add_noise = problem_.modified_noise && has_noise_;
  const double scale = inverse_transverse_count();
  const auto row = low_storage_rows[pass];
  crossing_for(
      "solprop_native.low_storage.separable_input", execution_space_,
      problem_.n_columns(), problem_.n_omega,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        const auto mode = column / n_pol;
        Complex value = source(column, omega);
        if (add_noise) {
          const auto values = spatial.value(mode, omega, active(omega) != 0U);
          const double angle = at(row) * values.linear.imag();
          value +=
              noise(column, omega) * Complex(Kokkos::cos(angle), Kokkos::sin(angle));
        }
        layout.cell(scratch, omega * n_pol + column % n_pol, mode) = value * scale;
      });
}

void GeneralWorkspace::low_storage_separable_fold(const std::size_t pass,
                                                  const int mode, const int term,
                                                  const double dz) {
  const auto state = current_;
  const auto accumulator = register_accumulator_;
  const auto stage_register = register_stage_;
  const auto flows = low_storage_flows_;
  const bool tabulated = problem_.low_storage_flows != 0;
  const auto scratch = transverse_;
  const auto spatial = spatial_;
  const auto window = omega_window_;
  const auto active = active_mask_;
  const auto in_norm = norm_mask_;
  const auto noise = noise_;
  const auto delta = stage_phi_delta_;
  const auto at = stage_phi_at_;
  const auto n_pol = problem_.n_pol_state;
  const auto layout = transverse_layout();
  const bool add_noise = problem_.modified_noise && has_noise_;
  const double scale = inverse_transverse_count();
  const int stage = static_cast<int>(pass);
  const bool exchange = mode == fold_exchange;
  const bool second = mode == fold_second;
  // Elided, each of the first three passes writes the next pass's input in place of C.
  const bool forward = !problem_.low_storage_stage_register && pass < 3;
  const bool final = pass == 3 && !exchange;
  // What this fold reads and writes, invariant over the traversal: each moves only the
  // registers its row of the flow table names.
  const bool flows_here =
      stage == 0 || (stage == 2 && !exchange) || (stage == 3 && !second);
  const bool reads_register = (second && stage < 3) || (exchange && stage > 0);
  const bool reads_accumulator = stage > 0 || second;
  const bool writes_state = (stage == 0 && !second) || stage == 3;
  const bool writes_register = stage < 3 && !forward;
  const auto own = low_storage_rows[pass];
  const auto next = low_storage_rows[pass < 3 ? pass + 1 : 3];
  const std::size_t flow_row = stage == 0 ? 0 : 1;
  const double step = dz;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // A device reduces the accepted state's norms in a pass of their own.
  const bool fused_norm = false;
#else
  const bool fused_norm = final;
#endif
  // The element, given its cell of the transverse scratch.
  const auto body =
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega, Complex& cell) {
    const auto column_mode = column / n_pol;
    const bool in_band = active(omega) != 0U;
    const auto values = spatial.value(column_mode, omega, in_band);
    Complex source{};
    if (term == term_material)
      source = values.prefactor * window(omega) * cell;
    else if (term == term_absorber && in_band)
      source = cell;
    Complex flow(1.0, 0.0);
    if (flows_here)
      flow = tabulated ? flows(0, column, omega)
                       : propagator_exp(delta(flow_row) * values.linear);
    Complex a = state(column, omega);
    Complex b = reads_accumulator ? accumulator(column, omega) : Complex{};
    Complex c = reads_register ? stage_register(column, omega) : Complex{};
    if (exchange) {
      Complex value = stage == 0 ? a : c;
      if (add_noise) {
        const double angle = at(own) * values.linear.imag();
        value += noise(column, omega) * Complex(Kokkos::cos(angle), Kokkos::sin(angle));
      }
      cell = value * scale;
    }
    low_storage_fold(stage, mode, step, source, a, b, c, flow, flow);
    if (forward) {
      Complex value = c;
      if (add_noise) {
        const double angle = at(next) * values.linear.imag();
        value += noise(column, omega) * Complex(Kokkos::cos(angle), Kokkos::sin(angle));
      }
      cell = value * scale;
    }
    if (writes_state)
      state(column, omega) = a;
    if (stage < 3)
      accumulator(column, omega) = b;
    if (writes_register)
      stage_register(column, omega) = c;
    return final && in_norm(omega) != 0U ? squared_abs(a) : 0.0;
  };
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  tiled_update(
      "solprop_native.low_storage.separable_fold", execution_space_,
      problem_.n_columns(), problem_.n_omega, exchange || forward,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        return layout.cell(scratch, omega * n_pol + column % n_pol, column / n_pol);
      },
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega, Complex& cell) {
        static_cast<void>(body(column, omega, cell));
      },
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega,
                    const Complex value) {
        layout.cell(scratch, omega * n_pol + column % n_pol, column / n_pol) = value;
      });
#else
  register_traversal<true>(
      "solprop_native.low_storage.separable_fold", execution_space_,
      problem_.n_columns(), problem_.n_omega, partials_, fused_norm,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        return body(
            column, omega,
            layout.cell(scratch, omega * n_pol + column % n_pol, column / n_pol));
      });
#endif
  if (final && !fused_norm)
    column_norms("solprop_native.low_storage.separable_norms", execution_space_, state,
                 in_norm, partials_);
}

// One pass on a modal route (mode sets, radial, general modal): its input, the
// register A (pass 1) or C with the noise draw at the pass's position, padded into the
// fine spectrum (and the general route's scaled channels); the shared core; and one
// fold of the source -- prefactor, window and the in-stage absorber's term at the
// pass's position, with no flow back to an anchor.
void GeneralWorkspace::low_storage_modal_pass(const std::size_t pass, const double dz) {
  const auto row = low_storage_rows[pass];
  const bool responds = general_
                            ? general_->responds() || problem_.in_stage_absorber
                            : problem_.nonlinear || problem_.plasma || problem_.raman ||
                                  problem_.in_stage_absorber || dispersive_ != nullptr;
  if (responds)
    low_storage_modal_input(pass);
  const bool evaluated = modal_core(row);
  const auto terms = stage_terms(row);
  const auto state = current_;
  const auto accumulator = register_accumulator_;
  const auto stage_register = register_stage_;
  const auto flows = low_storage_flows_;
  const auto count = problem_.low_storage_flows;
  const auto in_norm = norm_mask_;
  const auto n_phi = problem_.n_phi_terms;
  const auto n_prefactor = problem_.n_prefactor_terms;
  const int stage = static_cast<int>(pass);
  const bool final = pass == 3;
  const bool flows_here = stage != 1;
  const std::size_t flow_row = stage == 0 ? 0 : 1;
  const double step = dz;
  register_traversal(
      "solprop_native.low_storage.modal_fold", execution_space_, problem_.n_columns(),
      problem_.n_omega, partials_, final,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        Complex source{};
        if (evaluated) {
          // Zero weights skipped, as in `StageTerms::finish`.
          Complex prefactor{};
          for (std::size_t term = 0; term < n_prefactor; ++term) {
            const double weight = terms.prefactor_weights(row * n_prefactor + term);
            if (weight != 0.0)
              prefactor += weight * terms.prefactor_terms(term, column, index);
          }
          source = prefactor * terms.window(index) *
                   (terms.general ? terms.general_source(column, index)
                                  : terms.projected(column, terms.crop(index)));
          if (terms.in_stage_absorber && terms.active(index) != 0U)
            source += terms.absorber(column, terms.crop(index));
        }
        Complex flow(1.0, 0.0);
        if (flows_here)
          flow = count == 1 ? flows(0, column, index)
                 : count == 2
                     ? flows(flow_row, column, index)
                     : propagator_exp(terms.phase_sum(flow_row * n_phi, column, index));
        Complex a = state(column, index);
        Complex b = stage > 0 ? accumulator(column, index) : Complex{};
        Complex c{};
        low_storage_fold(stage, fold_complete, step, source, a, b, c, flow, flow);
        if (stage == 0 || stage == 3)
          state(column, index) = a;
        if (stage < 3) {
          accumulator(column, index) = b;
          stage_register(column, index) = c;
        }
        return final && in_norm(index) != 0U ? squared_abs(a) : 0.0;
      });
}

void GeneralWorkspace::low_storage_modal_input(const std::size_t pass) {
  const auto source = pass == 0 ? current_ : register_stage_;
  const auto row = low_storage_rows[pass];
  const auto terms = stage_terms(row);
  const auto n = problem_.n_omega;
  const auto gaps = fine_spectrum_size() - n;
  cartesian_for(
      "solprop_native.low_storage.modal_input", execution_space_, problem_.n_columns(),
      n, KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        Complex value = source(column, index);
        if (terms.add_noise) {
          const Complex phase_at = terms.noise_phase(row, column, index);
          value += terms.noise(column, index) *
                   Complex(Kokkos::cos(phase_at.imag()), Kokkos::sin(phase_at.imag()));
        }
        const double analytic_scale =
            terms.analytic && index > 0 && 2 * terms.crop(index) < terms.n_to ? 2.0
                                                                              : 1.0;
        terms.fine(column, terms.crop(index)) =
            analytic_scale * terms.inverse_n_to * value;
        if (terms.general)
          terms.general_scaled(column, index) =
              value *
              (terms.general_pad * general_doubling(terms.crop(index), terms.n_to));
        for (std::size_t gap = index; gap < gaps; gap += n)
          terms.zero_gaps(column, gap, gap + 1);
      });
}

// The accepted state at a field-free plane (SEP 22, Q15), held in the memory space: one
// volume, allocated at the first such plane and kept, so that a step abandoned in
// flight still leaves a resumable state without any host transfer at the plane.
void GeneralWorkspace::hold_plane() {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (!held_plane_.is_allocated())
    held_plane_ = ComplexView2("solprop_native.low_storage.held_plane",
                               problem_.n_columns(), problem_.n_omega);
  Kokkos::deep_copy(execution_space_, held_plane_, current_);
  held_ready_ = true;
}

std::vector<Complex> GeneralWorkspace::held_to_host() const {
  require_open();
  if (!held_ready_)
    throw std::logic_error("no field-free plane is held");
  execution_space_.fence("solprop_native held_to_host");
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, held_plane_);
  std::vector<Complex> result(problem_.n_columns() * problem_.n_omega);
  for (std::size_t column = 0; column < problem_.n_columns(); ++column)
    for (std::size_t index = 0; index < problem_.n_omega; ++index)
      result[column * problem_.n_omega + index] = host(column, index);
  return result;
}

// --- solprop SEP 16 phase 3b: the Adams history resident in the engine ---------------
//
// solprop's AdamsMarch keeps the controller and hands every state-sized operation of
// its fused path to an engine (solprop ADR-043). These are that engine's passes: the
// history's rows, the trial's predictor, source, corrected field and defect norms, the
// adoption and transport of an accepted step and dense observation all stay in device
// memory; only the trial's scalars, its coefficient tables (by value, in AdamsWeights)
// and the norms cross. Each pass is the host engine's arithmetic (solprop's
// backends/multistep.py, HostHistory and AdamsKernels), checked against it in parity.

void GeneralWorkspace::adams_allocate(const std::size_t slots, const bool compensated,
                                      const bool interpolate) {
  require_open();
  if (slots == 0 || slots > adams_max_samples)
    throw std::invalid_argument("the Adams history needs 1 to 8 rows");
  if (compensated && problem_.separable_geometry())
    throw std::logic_error("a separable operator needs no compensation");
  const auto columns = problem_.n_columns();
  const auto n = problem_.n_omega;
  if (compensated && adams_states_.extent(0) != slots) {
    adams_states_ = ComplexView3("solprop_native.adams.states", slots, columns, n);
    adams_rates_ = ComplexView3("solprop_native.adams.rates", slots, columns, n);
    adams_carried_ = ComplexView3("solprop_native.adams.carried", slots, columns, n);
    adams_lab_ = ComplexView2("solprop_native.adams.predicted", columns, n);
    adams_endpoint_ = ComplexView2("solprop_native.adams.endpoint", columns, n);
    adams_endpoint_rate_ =
        ComplexView2("solprop_native.adams.endpoint_rate", columns, n);
    adams_rate_weights_ = Kokkos::View<double*, Layout, MemorySpace>(
        "solprop_native.adams.rate_weights", problem_.n_phi_terms);
  }
  // Dense observations are opt-in. Keeping their slopes and saved fields on a run
  // that lands on its planes costs (slots + 4) whole fields without reading them.
  if (interpolate) {
    if (adams_slopes_.extent(0) != slots) {
      adams_slopes_ =
          ComplexView3("solprop_native.adams.dense_slopes", slots, columns, n);
      adams_held_ = ComplexView2("solprop_native.adams.held_state", columns, n);
      adams_dense_start_ = ComplexView2("solprop_native.adams.dense_start", columns, n);
      adams_dense_end_ = ComplexView2("solprop_native.adams.dense_end", columns, n);
      adams_dense_exponent_ =
          ComplexView2("solprop_native.adams.dense_exponent", columns, n);
    }
    if (compensated && adams_dense_terms_.extent(0) != slots)
      adams_dense_terms_ =
          ComplexView3("solprop_native.adams.dense_terms", slots, columns, n);
  } else {
    adams_slopes_ = {};
    adams_held_ = {};
    adams_dense_start_ = {};
    adams_dense_end_ = {};
    adams_dense_exponent_ = {};
    adams_dense_terms_ = {};
    adams_dense_count_ = 0;
  }
  // Beside a coupled flow the trial's block functions, and a frozen-residual flow's
  // weights per sample, live beside the history (solprop SEP 16 phase 6).
  if (flow_) {
    if (!flow_->full_route())
      throw std::logic_error("the resident Adams engine needs every coupled bin on the "
                             "flow's eigenbasis route");
    if (problem_.n_modes > adams_max_channels || columns != problem_.n_modes)
      throw std::logic_error("the resident Adams engine steps a coupled block of at "
                             "most 16 channels and one batch row");
    if (adams_block_.extent(0) != slots + 1)
      adams_block_ = Kokkos::View<Complex****, Layout, MemorySpace>(
          "solprop_native.adams.block", slots + 1, flow_->n_active(), problem_.n_modes,
          problem_.n_modes);
    if (compensated && flow_->n_terms() != 0 && adams_applied_.extent(0) != slots) {
      adams_applied_ = ComplexView3("solprop_native.adams.applied", slots, columns, n);
      adams_end_applied_ = ComplexView2("solprop_native.adams.end_applied", columns, n);
      adams_end_weights_ = Kokkos::View<double*, Layout, MemorySpace>(
          "solprop_native.adams.end_weights", flow_->n_terms());
    }
  }
  // The norm's scale per bin, written each trial; allocated here so that the history's
  // memory is held, and counted, from construction (solprop's adams_memory).
  if (adams_divisor_.extent(0) != n)
    adams_divisor_ =
        Kokkos::View<double*, Layout, MemorySpace>("solprop_native.adams.divisor", n);
  if (adams_store_.extent(0) == slots)
    return;
  adams_store_ = ComplexView3("solprop_native.adams.store", slots, columns, n);
  adams_moments_ = ComplexView3("solprop_native.adams.moments", slots, columns, n);
  adams_source_ = ComplexView2("solprop_native.adams.source", columns, n);
  adams_propagator_ = ComplexView2("solprop_native.adams.propagator", columns, n);
  adams_exponent_ = ComplexView2("solprop_native.adams.exponent", columns, n);
  adams_saved_ = ComplexView2("solprop_native.adams.saved_state", columns, n);
}

void GeneralWorkspace::require_adams() const {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (adams_store_.extent(0) == 0)
    throw std::logic_error("the resident Adams history is not allocated");
}

// The whole step's exponent from the uploaded weights, and its exponential: on a
// separable route the step's length times each element's operator, as the stage
// kernel forms the whole propagator; otherwise `sum_t dPhi_t Phi_t` in term order.
void GeneralWorkspace::adams_phase() {
  if (problem_.separable_geometry()) {
    const auto spatial = spatial_;
    const auto active = active_mask_;
    const auto delta = stage_phi_delta_;
    const auto exponent = adams_exponent_;
    const auto propagator = adams_propagator_;
    const auto n_pol = problem_.n_pol_state;
    const auto n_stages = problem_.n_stages;
    Kokkos::parallel_for(
        "solprop_native.adams.separable_phase",
        Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.n_omega}),
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          const auto values = spatial.value(column / n_pol, index, active(index) != 0U);
          const Complex x = delta(n_stages) * values.linear;
          exponent(column, index) = x;
          propagator(column, index) = exponential(x);
        });
    return;
  }
  const auto phi_terms = phi_terms_;
  const auto phi_delta = stage_phi_delta_;
  const auto exponent = adams_exponent_;
  const auto propagator = adams_propagator_;
  const auto n_phi = problem_.n_phi_terms;
  const auto offset = problem_.n_stages * n_phi;
  Kokkos::parallel_for(
      "solprop_native.adams.phase",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.n_omega}),
      // The step's phase increment: strict (solprop SEP 25 section 4.2).
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index)
          SOLPROP_NATIVE_STRICT_FUNCTION {
            SOLPROP_NATIVE_STRICT_BLOCK
            double real = 0.0;
            double imag = 0.0;
            for (std::size_t term = 0; term < n_phi; ++term) {
              const double weight = phi_delta(offset + term);
              // Exact, as in `StageTerms::phase_sum`.
              if (weight == 0.0)
                continue;
              const Complex fixed = phi_terms(term, column, index);
              real = real + weight * fixed.real();
              imag = imag + weight * fixed.imag();
            }
            exponent(column, index) = Complex(real, imag);
            propagator(column, index) = exponential(Complex(real, imag));
          });
}

// The stage-zero source at an anchor-frame field held in current_, as adams_source
// evaluates it, into stages_(0); current_ is restored from adams_saved_ afterwards.
void GeneralWorkspace::adams_evaluate(const TrialScalars& scalars) {
  if (general_)
    general_->set_weights(scalars.transform_weights);
  if (problem_.raman)
    prepare_raman_stage(0, scalars.raman_rates.data(),
                        problem_.n_raman_components > 1 ? scalars.raman_densities.data()
                                                        : nullptr);
  evaluate_stage_source(0, 0.0);
  Kokkos::deep_copy(execution_space_, current_, adams_saved_);
  stages_ready_ = true;
  pending_fsal_ = false;
  has_recycled_ = false;
}

void GeneralWorkspace::adams_start(const std::size_t slot,
                                   const TrialScalars& scalars) {
  require_adams();
  if (slot >= adams_store_.extent(0))
    throw std::out_of_range("Adams history row");
  require_trial_scalars(scalars);
  upload_trial(scalars);
  Kokkos::deep_copy(execution_space_, adams_saved_, current_);
  {
    const FlowAside aside(flow_);
    adams_evaluate(scalars);
  }
  // A frozen-residual flow's generator at the sample applied to its state, on the far
  // side of a kink that restarted the history (solprop SEP 16 phase 6).
  if (adams_applied_.extent(0) != 0) {
    upload_weights(execution_space_, adams_end_weights_, scalars.flow_weights);
    apply_generator(execution_space_, problem_.n_omega, problem_.n_modes,
                    flow_->block_slots(), flow_->residual_terms(), adams_end_weights_,
                    current_,
                    Kokkos::subview(adams_applied_, slot, Kokkos::ALL, Kokkos::ALL));
  }
  const auto stages = stages_;
  const auto store = adams_store_;
  Kokkos::parallel_for(
      "solprop_native.adams.start",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        store(slot, column, index) = stages(0, column, index);
      });
}

void GeneralWorkspace::adams_scale(const double rtol, const double atol,
                                   const std::vector<double>& weights) {
  require_open();
  if (!weights.empty())
    require_extent(weights.size(), problem_.n_omega, "Adams scale weights");
  adams_rtol_ = rtol;
  adams_atol_ = atol;
  adams_scale_weights_ = RealView("solprop_native.adams.scale_weights", weights.size());
  if (!weights.empty())
    copy_vector(execution_space_, adams_scale_weights_, weights);
  adams_magnitude_sum_ =
      Kokkos::View<double, Layout, MemorySpace>("solprop_native.adams.magnitude_sum");
  adams_device_scale_ = true;
}

// A trial's divisor: the host's when one is given, else formed on the device from the
// accepted state, as solprop's PIController.scale forms it from the same magnitude.
void GeneralWorkspace::adams_upload_divisor(const std::vector<double>& divisor) {
  if (adams_divisor_.extent(0) != problem_.n_omega)
    adams_divisor_ = Kokkos::View<double*, Layout, MemorySpace>(
        "solprop_native.adams.divisor", problem_.n_omega);
  if (!divisor.empty() || !adams_device_scale_) {
    require_extent(divisor.size(), problem_.n_omega, "Adams norm divisor");
    copy_vector(execution_space_, adams_divisor_, divisor);
    return;
  }
  const auto current = current_;
  const auto in_norm = norm_mask_;
  const auto n = problem_.n_omega;
  // A threaded reduction in whatever order measures fastest (solprop SEP 25).
  Kokkos::parallel_reduce(
      "solprop_native.adams.magnitude",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0,
                                          problem_.n_columns() * n),
      KOKKOS_LAMBDA(const std::size_t item, double& update) {
        const auto index = item % n;
        if (in_norm(index) != 0)
          update += squared_abs(current(item / n, index));
      },
      adams_magnitude_sum_);
  const auto sum = adams_magnitude_sum_;
  const auto weights = adams_scale_weights_;
  const bool weighted = weights.extent(0) != 0;
  const double rtol = adams_rtol_;
  const double atol = adams_atol_;
  const auto scale = adams_divisor_;
  Kokkos::parallel_for(
      "solprop_native.adams.divisor",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n),
      KOKKOS_LAMBDA(const std::size_t index) {
        const double magnitude = Kokkos::sqrt(sum());
        scale(index) = weighted ? rtol * weights(index) * magnitude + atol
                                : rtol * magnitude + atol;
      });
}

double GeneralWorkspace::adams_magnitude() {
  require_adams();
  const auto current = current_;
  const auto in_norm = norm_mask_;
  const auto n = problem_.n_omega;
  double total = 0.0;
  Kokkos::parallel_reduce(
      "solprop_native.adams.magnitude",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0,
                                          problem_.n_columns() * n),
      KOKKOS_LAMBDA(const std::size_t item, double& update) {
        const auto index = item % n;
        if (in_norm(index) != 0)
          update += squared_abs(current(item / n, index));
      },
      total);
  return Kokkos::sqrt(total);
}

namespace {

/** The squared scaled norms of up to three defects, reduced in one traversal. */
struct AdamsDefectNorms {
  using value_type = double[];
  using size_type = std::size_t;
  size_type value_count;
  Kokkos::View<Complex***, Layout, MemorySpace> store;
  Kokkos::View<Complex**, Layout, MemorySpace> source;
  Kokkos::View<Complex**, Layout, MemorySpace> propagator;
  Kokkos::View<unsigned char*, Layout, MemorySpace> in_norm;
  Kokkos::View<double*, Layout, MemorySpace> divisor;
  AdamsWeights weights;
  double dz;
  std::size_t n;

  // A defect is a divided difference of the samples: strict (solprop SEP 25 4.2).
  SOLPROP_NATIVE_STRICT_FUNCTION
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t item,
                                         value_type update) const {
    SOLPROP_NATIVE_STRICT_BLOCK
    const auto column = item / n;
    const auto index = item % n;
    if (in_norm(index) == 0)
      return;
    const Complex u = propagator(column, index);
    const double scale = divisor(index);
    for (size_type row = 0; row < value_count; ++row) {
      Complex defect = strict_scale(weights.rows[row][0], source(column, index));
      for (std::size_t t = 1; t < weights.width; ++t)
        defect = strict_accumulate(defect, weights.rows[row][t],
                                   store(weights.slot[t - 1], column, index));
      const Complex scaled =
          strict_divide(strict_mul(u, strict_scale(dz, defect)), scale);
      update[row] += squared_abs(scaled);
    }
  }
  KOKKOS_INLINE_FUNCTION void join(value_type destination,
                                   const value_type source_values) const {
    for (size_type row = 0; row < value_count; ++row)
      destination[row] += source_values[row];
  }
  KOKKOS_INLINE_FUNCTION void init(value_type values) const {
    for (size_type row = 0; row < value_count; ++row)
      values[row] = 0.0;
  }
};

} // namespace

std::vector<double>
GeneralWorkspace::adams_fused_trial(const double dz, const AdamsWeights& weights,
                                    const std::vector<double>& divisor,
                                    const TrialScalars& scalars) {
  require_adams();
  if (flow_)
    throw std::logic_error("the Adams source is evaluated without a coupled flow");
  if (weights.n_rows == 0 || weights.n_rows > 3 || weights.width == 0 ||
      weights.width > adams_max_samples || weights.order >= weights.width)
    throw std::invalid_argument("Adams coefficient tables are out of range");
  for (std::size_t t = 0; t + 1 < weights.width; ++t)
    if (weights.slot[t] >= adams_store_.extent(0))
      throw std::out_of_range("Adams history row");
  require_trial_scalars(scalars);
  upload_trial(scalars);
  adams_upload_divisor(divisor);
  adams_phase();
  const auto columns = problem_.n_columns();
  const auto n = problem_.n_omega;
  const auto policy = Range2(execution_space_, {0, 0}, {columns, n});
  const auto store = adams_store_;
  const auto current = current_;
  const auto saved = adams_saved_;
  // The predictor in the anchor's frame, a_n + dz sum_t w_t g_t, is the field the
  // stage-zero source evaluates; the accepted state is kept aside meanwhile.
  Kokkos::deep_copy(execution_space_, saved, current);
  Kokkos::parallel_for(
      "solprop_native.adams.predict", policy,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        Complex total{};
        for (std::size_t t = 0; t < weights.order; ++t)
          total += weights.predictor[t] * store(weights.slot[t], column, index);
        current(column, index) = saved(column, index) + dz * total;
      });
  adams_evaluate(scalars);
  const auto stages = stages_;
  const auto source = adams_source_;
  const auto propagator = adams_propagator_;
  const auto advanced = advanced_;
  Kokkos::parallel_for(
      "solprop_native.adams.correct", policy,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        const Complex g = stages(0, column, index);
        source(column, index) = g;
        Complex total = weights.corrector[0] * g;
        for (std::size_t t = 1; t < weights.width; ++t)
          total += weights.corrector[t] * store(weights.slot[t - 1], column, index);
        advanced(column, index) =
            propagator(column, index) * (current(column, index) + dz * total);
      });
  std::vector<double> norms(weights.n_rows, 0.0);
  const AdamsDefectNorms reducer{weights.n_rows, store,      source,
                                 propagator,     norm_mask_, adams_divisor_,
                                 weights,        dz,         n};
  const Kokkos::View<double*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      result(norms.data(), weights.n_rows);
  Kokkos::parallel_reduce(
      "solprop_native.adams.defect_norms",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, columns * n), reducer,
      result);
  execution_space_.fence("solprop_native Adams trial boundary");
  for (auto& value : norms)
    value = std::sqrt(value);
  adams_trial_ready_ = true;
  return norms;
}

void GeneralWorkspace::adams_fused_accept(const double dz, const AdamsWeights& keep,
                                          const std::size_t free,
                                          const AdamsWeights& dense) {
  require_adams();
  if (!adams_trial_ready_)
    throw std::logic_error("no Adams trial is held");
  if (free >= adams_store_.extent(0))
    throw std::out_of_range("Adams history row");
  if (dense.width > 0 && adams_slopes_.extent(0) == 0)
    throw std::logic_error("Adams dense observation storage is not allocated");
  adams_trial_ready_ = false;
  const auto columns = problem_.n_columns();
  const auto n = problem_.n_omega;
  const auto policy = Range2(execution_space_, {0, 0}, {columns, n});
  const auto store = adams_store_;
  const auto source = adams_source_;
  const auto propagator = adams_propagator_;
  const auto slopes = adams_slopes_;
  // The step's own interpolant, in its anchor's frame, before the transport moves the
  // rows: the slopes (source, newest, ...) and the state it starts from.
  adams_dense_count_ = dense.width;
  if (dense.width > 0) {
    Kokkos::deep_copy(execution_space_, adams_dense_start_, current_);
    Kokkos::deep_copy(execution_space_, adams_dense_exponent_, adams_exponent_);
    Kokkos::parallel_for(
        "solprop_native.adams.dense_slopes", policy,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          slopes(0, column, index) = source(column, index);
          for (std::size_t t = 1; t < dense.width; ++t)
            slopes(t, column, index) = store(dense.slot[t - 1], column, index);
        });
  }
  finish(dz);
  stash_is_current_ = false;
  if (dense.width > 0) {
    const auto current = current_;
    const auto end = adams_dense_end_;
    Kokkos::parallel_for(
        "solprop_native.adams.dense_end", policy,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          end(column, index) = current(column, index) / propagator(column, index);
        });
  }
  // The history re-anchored at the step's end, and the new derivative in row `free`.
  Kokkos::parallel_for(
      "solprop_native.adams.transport", policy,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        const Complex u = propagator(column, index);
        for (std::size_t t = 0; t < keep.width; ++t)
          store(keep.slot[t], column, index) = store(keep.slot[t], column, index) * u;
        store(free, column, index) = u * source(column, index);
      });
  execution_space_.fence("solprop_native Adams accepted-step boundary");
}

void GeneralWorkspace::adams_hold() {
  require_adams();
  if (adams_held_.empty())
    throw std::logic_error("Adams dense observation storage is not allocated");
  Kokkos::deep_copy(execution_space_, adams_held_, current_);
}

void GeneralWorkspace::adams_restore() {
  require_adams();
  if (adams_held_.empty())
    throw std::logic_error("Adams dense observation storage is not allocated");
  Kokkos::deep_copy(execution_space_, current_, adams_held_);
  stash_is_current_ = false;
  execution_space_.fence("solprop_native Adams restore boundary");
}

// --- solprop SEP 29 G2: the resident history's rows across a resume ----------------
//
// The persistent rows are the samples (store), a compensated history's sampled states
// and rates, and a frozen-residual flow's applied generators G(z_j) u_j: everything a
// later trial reads that an earlier step wrote. The rest of the Adams views are a
// trial's scratch, rebuilt by the next trial, and the step's bookkeeping lives in
// solprop's march. The copies are lossless: the view's own LayoutRight order, element
// by element, with no arithmetic.

ComplexView3 GeneralWorkspace::adams_row_view(const std::string& name) const {
  require_adams();
  ComplexView3 view;
  if (name == "store")
    view = adams_store_;
  else if (name == "states")
    view = adams_states_;
  else if (name == "rates")
    view = adams_rates_;
  else if (name == "applied")
    view = adams_applied_;
  else
    throw std::invalid_argument("unknown Adams history row '" + name + "'");
  if (view.extent(0) == 0)
    throw std::logic_error("the Adams history row '" + name + "' is not allocated");
  return view;
}

std::vector<std::string> GeneralWorkspace::adams_row_names() const {
  require_adams();
  std::vector<std::string> names{"store"};
  if (adams_states_.extent(0) != 0)
    names.insert(names.end(), {"states", "rates"});
  if (adams_applied_.extent(0) != 0)
    names.emplace_back("applied");
  return names;
}

std::array<std::size_t, 3>
GeneralWorkspace::adams_row_extents(const std::string& name) const {
  const auto view = adams_row_view(name);
  return {view.extent(0), view.extent(1), view.extent(2)};
}

std::vector<Complex>
GeneralWorkspace::adams_rows_to_host(const std::string& name) const {
  std::vector<std::complex<double>> values(adams_row_view(name).size());
  adams_rows_to_host(name, values.data(), values.size());
  return {values.begin(), values.end()};
}

// One copy, straight into the caller's buffer (a NumPy array's storage at the seam):
// no intermediate host vector, so a kept save costs what the host engine's copy does.
void GeneralWorkspace::adams_rows_to_host(const std::string& name,
                                          std::complex<double>* destination,
                                          const std::size_t size) const {
  const auto view = adams_row_view(name);
  if (size != view.size())
    throw std::invalid_argument("the Adams history row '" + name +
                                "' has the wrong extent");
  execution_space_.fence("solprop_native Adams rows_to_host");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  const auto* source = host.data();
  for (std::size_t index = 0; index < size; ++index)
    destination[index] = {source[index].real(), source[index].imag()};
}

void GeneralWorkspace::adams_rows_from_host(const std::string& name,
                                            const std::vector<Complex>& values) {
  const auto view = adams_row_view(name);
  if (values.size() != view.size())
    throw std::invalid_argument("the Adams history row '" + name +
                                "' has the wrong extent");
  copy_vector(execution_space_, view, values);
  // A trial in flight and the last step's interpolant belong to the history replaced.
  adams_trial_ready_ = false;
  adams_dense_count_ = 0;
  execution_space_.fence("solprop_native Adams rows_from_host");
}

// solprop SEP 29 G1: the general z tables. A table is uploaded once, after the state
// is constructed and before its first trial, which is the construction's allocation.
void GeneralWorkspace::table_upload(const TableKind kind, std::vector<double> edges,
                                    const double lower, const double upper,
                                    const std::complex<double>* coefficients,
                                    const std::size_t n_intervals,
                                    const std::size_t n_terms,
                                    const std::size_t n_values) {
  require_open();
  const bool prefactor = kind == TableKind::prefactor;
  const bool flux = kind == TableKind::flux;
  if (flux ? !problem_.flux_table
           : (prefactor ? !problem_.prefactor_table : !problem_.phi_table))
    throw std::logic_error(std::string("this problem's ") +
                           (flux        ? "plane measure"
                            : prefactor ? "prefactor"
                                        : "Phi") +
                           " is not a z table");
  if (n_intervals == 0 || n_terms == 0 || edges.size() != n_intervals + 1)
    throw std::invalid_argument("a z table needs one more breakpoint than intervals "
                                "and at least one coefficient");
  if (flux && n_values != problem_.n_omega)
    throw std::invalid_argument("the plane measure's material table holds n_omega = " +
                                std::to_string(problem_.n_omega) + " values");
  if (!flux && n_values != problem_.n_columns() * problem_.n_omega)
    throw std::invalid_argument("a z table's values must be the state's (" +
                                std::to_string(problem_.n_columns()) + " x " +
                                std::to_string(problem_.n_omega) + ")");
  for (std::size_t index = 0; index < edges.size(); ++index)
    if (!std::isfinite(edges[index]) ||
        (index > 0 && !(edges[index] > edges[index - 1])))
      throw std::invalid_argument("a z table's breakpoints must be finite and "
                                  "strictly increasing");
  if (!(lower <= edges.front() && upper >= edges.back()))
    throw std::invalid_argument("a z table's admitted span must contain its edges");
  auto& table = kind == TableKind::phi         ? phi_table_
                : kind == TableKind::prefactor ? prefactor_table_
                : flux                         ? flux_table_
                                               : rate_table_;
  table.coefficients = Kokkos::View<Complex***, Layout, MemorySpace>(
      kind == TableKind::phi         ? "solprop_native.linear.phi_table"
      : kind == TableKind::prefactor ? "solprop_native.response.prefactor_table"
      : flux                         ? "solprop_native.flux.material_table"
                                     : "solprop_native.linear.rate_table",
      n_intervals, n_terms, n_values);
  auto mirror = Kokkos::create_mirror_view(table.coefficients);
  auto* data = mirror.data();
  const std::size_t count = n_intervals * n_terms * n_values;
  for (std::size_t index = 0; index < count; ++index)
    data[index] =
        chebyshev::Coefficient<std::complex<double>>::load(coefficients[index]);
  Kokkos::deep_copy(execution_space_, table.coefficients, mirror);
  table.edges = std::move(edges);
  table.lower = lower;
  table.upper = upper;
  execution_space_.fence("solprop_native z table upload");
}

// The tables' values at a trial's positions into their term slots: Phi's and the
// prefactor's, each a pass over its coefficients for every position at once.
void GeneralWorkspace::table_evaluate(const std::vector<std::size_t>& slots,
                                      const std::vector<double>& positions) {
  require_open();
  if (!phi_table_.present() && !prefactor_table_.present())
    throw std::logic_error("no z table has been uploaded");
  for (const auto slot : slots) {
    if ((phi_table_.present() && slot >= problem_.n_phi_terms) ||
        (prefactor_table_.present() && slot >= problem_.n_prefactor_terms))
      throw std::out_of_range("z table slot " + std::to_string(slot) +
                              " is not a term of this problem");
  }
  if (phi_table_.present())
    z_table::evaluate("solprop_native.linear.phi_table_values", execution_space_,
                      phi_table_, z_table::locate(phi_table_, slots, positions, false),
                      phi_terms_.data());
  if (prefactor_table_.present())
    z_table::evaluate("solprop_native.response.prefactor_table_values",
                      execution_space_, prefactor_table_,
                      z_table::locate(prefactor_table_, slots, positions, false),
                      prefactor_terms_.data());
}

// solprop SEP 34: the plane measure's material square at `z`, a radial varying
// medium's, into the flux coefficients the statistics read; nothing for any other.
void GeneralWorkspace::flux_at(const double z) {
  if (!flux_table_.present())
    return;
  z_table::evaluate("solprop_native.flux.material_values", execution_space_,
                    flux_table_, z_table::locate(flux_table_, {0}, {z}, false),
                    flux_.square.data());
}

// D(z) = dPhi/dz from the rate table into one Phi term slot, read on the side of an
// interior breakpoint `before` asks for (solprop's `phi_rate`).
void GeneralWorkspace::table_rate(const std::size_t slot, const double z,
                                  const bool before) {
  require_open();
  if (!rate_table_.present())
    throw std::logic_error("no rate table has been uploaded");
  if (slot >= problem_.n_phi_terms)
    throw std::out_of_range("z table slot " + std::to_string(slot) +
                            " is not a term of this problem");
  z_table::evaluate("solprop_native.linear.rate_table_values", execution_space_,
                    rate_table_, z_table::locate(rate_table_, {slot}, {z}, before),
                    phi_terms_.data());
}

std::vector<Complex>
GeneralWorkspace::table_terms_to_host(const TableKind kind,
                                      const std::size_t slot) const {
  require_open();
  const auto& terms = kind == TableKind::prefactor ? prefactor_terms_ : phi_terms_;
  if (slot >= terms.extent(0))
    throw std::out_of_range("term slot " + std::to_string(slot) +
                            " is not a term of this problem");
  execution_space_.fence("solprop_native z table terms_to_host");
  const auto host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, Kokkos::subview(terms, slot, Kokkos::ALL, Kokkos::ALL));
  std::vector<Complex> values(host.size());
  for (std::size_t column = 0; column < host.extent(0); ++column)
    for (std::size_t index = 0; index < host.extent(1); ++index)
      values[column * host.extent(1) + index] = host(column, index);
  return values;
}

// The stage kernels' own phase sum (`StageTerms::phase_sum`) over the Phi terms for one
// row of weights, into a fresh host array: a diagnostic, outside any trial.
std::vector<Complex>
GeneralWorkspace::phase_sum_to_host(const std::vector<double>& weights) {
  require_open();
  if (problem_.separable_geometry())
    throw std::logic_error("a separable operator has no Phi terms");
  require_extent(weights.size(), problem_.n_phi_terms, "phase weights");
  RealView row("solprop_native.diagnostic.phase_weights", weights.size());
  copy_vector(execution_space_, row, weights);
  ComplexView2 phase("solprop_native.diagnostic.phase", problem_.n_columns(),
                     problem_.n_omega);
  auto terms = stage_terms(0);
  terms.phi_delta = row;
  Kokkos::parallel_for(
      "solprop_native.diagnostic.phase_sum",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        phase(column, index) = terms.phase_sum(0, column, index);
      });
  execution_space_.fence("solprop_native phase_sum_to_host");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, phase);
  return {host.data(), host.data() + host.size()};
}

// The field at a fraction of the last accepted step: the step's interpolant in its
// anchor's frame -- the Adams corrector's (`weights.predictor` holds dz-scaled
// integrated weights over the slopes) or the cubic Hermite one (`hermite`, the four
// coefficients in `weights.corrector`) -- with a bin the step's flow changes by more
// than an e-fold taking the anchor's value, carried to the observation by the uploaded
// trial's flow and masked into current_.
void GeneralWorkspace::adams_observe(const AdamsWeights& weights, const bool hermite,
                                     const TrialScalars& scalars) {
  require_adams();
  if (adams_dense_count_ == 0)
    throw std::logic_error("no Adams step is held for dense observation");
  require_trial_scalars(scalars);
  upload_trial(scalars);
  adams_phase();
  const auto columns = problem_.n_columns();
  const auto n = problem_.n_omega;
  const auto count = adams_dense_count_;
  const auto start = adams_dense_start_;
  const auto end = adams_dense_end_;
  const auto exponent = adams_dense_exponent_;
  const auto slopes = adams_slopes_;
  const auto carry = adams_propagator_;
  const auto current = current_;
  Kokkos::parallel_for(
      "solprop_native.adams.observe", Range2(execution_space_, {0, 0}, {columns, n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        Complex value;
        if (hermite) {
          value = weights.corrector[0] * start(column, index) +
                  weights.corrector[1] * slopes(1, column, index) +
                  weights.corrector[2] * end(column, index) +
                  weights.corrector[3] * slopes(0, column, index);
        } else {
          Complex total{};
          for (std::size_t t = 0; t < count; ++t)
            total += weights.predictor[t] * slopes(t, column, index);
          value = start(column, index) + total;
        }
        if (Kokkos::abs(exponent(column, index).real()) > 1.0)
          value = start(column, index);
        current(column, index) = carry(column, index) * value;
      });
  mask(current_);
  stash_is_current_ = false;
  execution_space_.fence("solprop_native Adams observation boundary");
}

// --- solprop SEP 16 phase 3b: the exponential formulation, resident
// -------------------
//
// The exponential formulation's passes (solprop ADR-041, ADR-043): untransported
// samples integrated exactly against the step's flow exp(x), x = Phi(z + h) - Phi(z).
// The moments M_k = k! phi_{k+1}(x) are formed once a trial per element by the
// reference's algorithm (solprop backends/exponential.py, phi_functions; the compiled
// form of solprop's numba_cpu/multistep.py, _phi), then every formula of the trial is
// applied in the monomial basis of its interpolant; the defects use the factored form
// (a divided difference of the samples times one moment combination).

namespace {

constexpr double adams_phi_join = 3.0;
constexpr int adams_series_terms = 24;

/**
 * phi_0(x) .. phi_order(x) for one element into `out`, as solprop's compiled _phi.
 *
 * The small-argument phi-functions: strict arithmetic (solprop SEP 25 section 4.2).
 */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION void adams_phi(const Complex x, const Complex exp_x,
                                      const std::size_t order, Complex* out) {
  SOLPROP_NATIVE_STRICT_BLOCK
  // 1/k! for k <= 2 adams_max_samples, exactly representable rounding of the reference.
  constexpr double reciprocal[17] = {1.0,
                                     1.0,
                                     0.5,
                                     1.0 / 6.0,
                                     1.0 / 24.0,
                                     1.0 / 120.0,
                                     1.0 / 720.0,
                                     1.0 / 5040.0,
                                     1.0 / 40320.0,
                                     1.0 / 362880.0,
                                     1.0 / 3628800.0,
                                     1.0 / 39916800.0,
                                     1.0 / 479001600.0,
                                     1.0 / 6227020800.0,
                                     1.0 / 87178291200.0,
                                     1.0 / 1307674368000.0,
                                     1.0 / 20922789888000.0};
  const double magnitude = Kokkos::abs(x);
  if (magnitude == 0.0) {
    for (std::size_t k = 0; k <= order; ++k)
      out[k] = reciprocal[k];
    return;
  }
  if (magnitude > adams_phi_join) {
    const Complex inverse = strict_divide(Complex(1.0), x);
    out[0] = exp_x;
    for (std::size_t k = 1; k <= order; ++k)
      out[k] = strict_mul(strict_sub(out[k - 1], Complex(reciprocal[k - 1])), inverse);
    return;
  }
  int squarings = 0;
  if (magnitude > 0.5)
    squarings = static_cast<int>(Kokkos::ceil(Kokkos::log2(magnitude / 0.5)));
  const Complex y = strict_scale(Kokkos::pow(0.5, squarings), x);
  Complex term = reciprocal[order];
  Complex total = term;
  for (int m = 1; m < adams_series_terms; ++m) {
    term = strict_scale(1.0 / static_cast<double>(static_cast<std::size_t>(m) + order),
                        strict_mul(term, y));
    total = strict_add(total, term);
    if (Kokkos::abs(term) <=
        0.25 * Kokkos::Experimental::epsilon_v<double> * Kokkos::abs(total))
      break;
  }
  out[order] = total;
  for (std::size_t k = order - 1; k > 0; --k)
    out[k] = strict_add(reciprocal[k], strict_mul(y, out[k + 1]));
  out[0] = Kokkos::exp(y);
  Complex old[2 * adams_max_samples + 1];
  for (int pass = 0; pass < squarings; ++pass) {
    for (std::size_t k = 0; k <= order; ++k)
      old[k] = out[k];
    out[0] = strict_mul(old[0], old[0]);
    for (std::size_t k = 1; k <= order; ++k) {
      Complex sum = strict_mul(old[0], old[k]);
      for (std::size_t i = 1; i <= k; ++i)
        sum = strict_accumulate(sum, reciprocal[k - i], old[i]);
      out[k] = strict_scale(Kokkos::pow(0.5, static_cast<int>(k)), sum);
    }
  }
}

} // namespace

namespace {

/** k! for k < 2 adams_max_samples, the moments' scale. */
KOKKOS_INLINE_FUNCTION double adams_factorial(const std::size_t k) {
  double value = 1.0;
  for (std::size_t i = 2; i <= k; ++i)
    value *= static_cast<double>(i);
  return value;
}

/**
 * The phi-functions a trial's moments take: at x, or under a carrier frame at
 * x - sigma with that argument's own exponential (solprop SEP 35 section 4.1). The
 * flow stays exp(x), which the caller holds.
 */
KOKKOS_INLINE_FUNCTION void adams_moment_phi(const ExponentialWeights& weights,
                                             const Complex x, const Complex exp_x,
                                             const std::size_t order, Complex* out) {
  if (!weights.shifted) {
    adams_phi(x, exp_x, order, out);
    return;
  }
  // exp(x - sigma) = exp(x) exp(-sigma): one product with the trial's exp(-sigma)
  // rather than a second exponential per element (a few ulp, inside the moments'
  // share of the rounding budget).
  const Complex residual = x - Complex(0.0, weights.shift);
  adams_phi(residual,
            strict_mul(exp_x, Complex(weights.unshift_re, weights.unshift_im)), order,
            out);
}

/**
 * History row `t` of a trial (`weights.slot[t]`) at one element, rotated by a carrier
 * frame's factor exp(sigma (1 - theta_t)) as it is read (solprop SEP 35 section 6.2):
 * the store keeps the untransported samples.
 */
template <class Store>
KOKKOS_INLINE_FUNCTION Complex adams_history(const Store& store,
                                             const ExponentialWeights& weights,
                                             const std::size_t t,
                                             const std::size_t column,
                                             const std::size_t index) {
  const Complex value = store(weights.slot[t], column, index);
  if (!weights.shifted)
    return value;
  return strict_mul(Complex(weights.factor_re[t], weights.factor_im[t]), value);
}

/** The squared scaled norms of up to three factored exponential defects. */
struct ExponentialDefectNorms {
  using value_type = double[];
  using size_type = std::size_t;
  size_type value_count;
  Kokkos::View<Complex***, Layout, MemorySpace> store;
  Kokkos::View<Complex***, Layout, MemorySpace> moments;
  Kokkos::View<Complex**, Layout, MemorySpace> source;
  Kokkos::View<Complex***, Layout, MemorySpace> carried;
  Kokkos::View<Complex**, Layout, MemorySpace> endpoint;
  Kokkos::View<unsigned char*, Layout, MemorySpace> in_norm;
  Kokkos::View<double*, Layout, MemorySpace> divisor;
  ExponentialWeights weights;
  double dz;
  std::size_t n;

  // The defect and its compensation term: strict (solprop SEP 25 section 4.2).
  SOLPROP_NATIVE_STRICT_FUNCTION
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t item,
                                         value_type update) const {
    SOLPROP_NATIVE_STRICT_BLOCK
    const auto column = item / n;
    const auto index = item % n;
    if (in_norm(index) == 0)
      return;
    const double scale = divisor(index);
    for (size_type row = 0; row < value_count; ++row) {
      // A divided difference of the samples times one moment combination.
      Complex difference = strict_scale(weights.rows[row][0], source(column, index));
      for (std::size_t t = 1; t < weights.width; ++t)
        difference =
            strict_accumulate(difference, weights.rows[row][t],
                              adams_history(store, weights, t - 1, column, index));
      Complex weight{};
      for (std::size_t k = 0; k < weights.count; ++k)
        weight = strict_accumulate(weight, weights.rows[row][weights.width + k],
                                   moments(k, column, index));
      Complex defect = strict_scale(dz, strict_mul(difference, weight));
      if (weights.compensated) {
        Complex compensation =
            strict_scale(weights.comp_rows[row][0], endpoint(column, index));
        for (std::size_t t = 0; t + 1 < weights.width; ++t)
          compensation = strict_accumulate(compensation, weights.comp_rows[row][t + 1],
                                           carried(t, column, index));
        defect = strict_accumulate(defect, dz, compensation);
      }
      const Complex scaled = strict_divide(defect, scale);
      update[row] += squared_abs(scaled);
    }
  }
  KOKKOS_INLINE_FUNCTION void join(value_type destination,
                                   const value_type source_values) const {
    for (size_type row = 0; row < value_count; ++row)
      destination[row] += source_values[row];
  }
  KOKKOS_INLINE_FUNCTION void init(value_type values) const {
    for (size_type row = 0; row < value_count; ++row)
      values[row] = 0.0;
  }
};

// The defects' norms beside a coupled flow (solprop SEP 16 phase 6): per bin, the
// samples' channel vectors; off the block each channel's defect is the diagonal one,
// on it each moment combination is a matrix applied to the divided difference.
struct CoupledDefectNorms {
  using value_type = double[];
  using size_type = std::size_t;
  size_type value_count;
  Kokkos::View<Complex***, Layout, MemorySpace> store;
  Kokkos::View<Complex***, Layout, MemorySpace> moments;
  Kokkos::View<Complex**, Layout, MemorySpace> source;
  Kokkos::View<Complex***, Layout, MemorySpace> carried;
  Kokkos::View<Complex**, Layout, MemorySpace> endpoint;
  Kokkos::View<Complex****, Layout, MemorySpace> block;
  Kokkos::View<std::size_t*, Layout, MemorySpace> slots;
  Kokkos::View<unsigned char*, Layout, MemorySpace> in_norm;
  Kokkos::View<double*, Layout, MemorySpace> divisor;
  ExponentialWeights weights;
  double dz;
  std::size_t modes;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  SOLPROP_NATIVE_STRICT_FUNCTION
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t item,
                                         value_type update) const {
    SOLPROP_NATIVE_STRICT_BLOCK
    // An item is a (bin, channel) pair: a thread holds no channel vectors.
    const auto index = item / modes;
    const auto i = item % modes;
    if (in_norm(index) == 0)
      return;
    const double scale = divisor(index);
    const auto one = slots(index);
    // Channel c's divided difference for one defect.
    const auto difference = [&](const size_type row, const std::size_t c) {
      Complex total = strict_scale(weights.rows[row][0], source(c, index));
      for (std::size_t t = 1; t < weights.width; ++t)
        total = strict_accumulate(total, weights.rows[row][t],
                                  adams_history(store, weights, t - 1, c, index));
      return total;
    };
    for (size_type row = 0; row < value_count; ++row) {
      Complex defect{};
      if (one == 0) {
        Complex weight{};
        for (std::size_t k = 0; k < weights.count; ++k)
          weight = strict_accumulate(weight, weights.rows[row][weights.width + k],
                                     moments(k, i, index));
        defect = strict_scale(dz, strict_mul(difference(row, i), weight));
      } else {
        const auto bin = one - 1;
        Complex total{};
        for (std::size_t j = 0; j < modes; ++j) {
          Complex weight{};
          for (std::size_t k = 0; k < weights.count; ++k)
            weight = strict_accumulate(weight, weights.rows[row][weights.width + k],
                                       block(k + 1, bin, i, j));
          total = strict_add(total, strict_mul(weight, difference(row, j)));
        }
        defect = strict_scale(dz, total);
      }
      if (weights.compensated) {
        Complex compensation =
            strict_scale(weights.comp_rows[row][0], endpoint(i, index));
        for (std::size_t t = 0; t + 1 < weights.width; ++t)
          compensation = strict_accumulate(compensation, weights.comp_rows[row][t + 1],
                                           carried(t, i, index));
        defect = strict_accumulate(defect, dz, compensation);
      }
      update[row] += squared_abs(strict_divide(defect, scale));
    }
  }
#else
  // The defects and their compensation terms: strict (solprop SEP 25 section 4.2).
  SOLPROP_NATIVE_STRICT_FUNCTION
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t index,
                                         value_type update) const {
    SOLPROP_NATIVE_STRICT_BLOCK
    if (in_norm(index) == 0)
      return;
    const double scale = divisor(index);
    const auto one = slots(index);
    Complex values[adams_max_samples][adams_max_channels];
    for (std::size_t i = 0; i < modes; ++i) {
      values[0][i] = source(i, index);
      for (std::size_t t = 1; t < weights.width; ++t)
        values[t][i] = adams_history(store, weights, t - 1, i, index);
    }
    Complex difference[adams_max_channels];
    for (size_type row = 0; row < value_count; ++row) {
      for (std::size_t i = 0; i < modes; ++i) {
        Complex total = strict_scale(weights.rows[row][0], values[0][i]);
        for (std::size_t t = 1; t < weights.width; ++t)
          total = strict_accumulate(total, weights.rows[row][t], values[t][i]);
        difference[i] = total;
      }
      for (std::size_t i = 0; i < modes; ++i) {
        Complex defect{};
        if (one == 0) {
          Complex weight{};
          for (std::size_t k = 0; k < weights.count; ++k)
            weight = strict_accumulate(weight, weights.rows[row][weights.width + k],
                                       moments(k, i, index));
          defect = strict_scale(dz, strict_mul(difference[i], weight));
        } else {
          const auto bin = one - 1;
          Complex total{};
          for (std::size_t j = 0; j < modes; ++j) {
            Complex weight{};
            for (std::size_t k = 0; k < weights.count; ++k)
              weight = strict_accumulate(weight, weights.rows[row][weights.width + k],
                                         block(k + 1, bin, i, j));
            total = strict_add(total, strict_mul(weight, difference[j]));
          }
          defect = strict_scale(dz, total);
        }
        if (weights.compensated) {
          Complex compensation =
              strict_scale(weights.comp_rows[row][0], endpoint(i, index));
          for (std::size_t t = 0; t + 1 < weights.width; ++t)
            compensation = strict_accumulate(
                compensation, weights.comp_rows[row][t + 1], carried(t, i, index));
          defect = strict_accumulate(defect, dz, compensation);
        }
        update[row] += squared_abs(strict_divide(defect, scale));
      }
    }
  }
#endif
  KOKKOS_INLINE_FUNCTION void join(value_type destination,
                                   const value_type source_values) const {
    for (size_type row = 0; row < value_count; ++row)
      destination[row] += source_values[row];
  }
  KOKKOS_INLINE_FUNCTION void init(value_type values) const {
    for (size_type row = 0; row < value_count; ++row)
      values[row] = 0.0;
  }
};

} // namespace

// D(z) = sum_t w_t Phi_t from the operator's fixed terms (solprop's
// translate.linop.rate_weights), into `out`.
void GeneralWorkspace::adams_rate(const std::vector<double>& rate,
                                  const ComplexView2& out) {
  require_extent(rate.size(), problem_.n_phi_terms, "Adams rate weights");
  if (adams_rate_weights_.extent(0) != rate.size())
    adams_rate_weights_ = Kokkos::View<double*, Layout, MemorySpace>(
        "solprop_native.adams.rate_weights", rate.size());
  copy_vector(execution_space_, adams_rate_weights_, rate);
  const auto weights = adams_rate_weights_;
  const auto phi_terms = phi_terms_;
  const auto n_phi = problem_.n_phi_terms;
  Kokkos::parallel_for(
      "solprop_native.adams.rate",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        Complex value{};
        for (std::size_t term = 0; term < n_phi; ++term)
          value += weights(term) * phi_terms(term, column, index);
        out(column, index) = value;
      });
}

// The startup sample's state and rate, beside the source adams_start stored.
void GeneralWorkspace::adams_compensation_start(const std::size_t slot,
                                                const std::vector<double>& rate) {
  require_adams();
  if (adams_states_.extent(0) == 0)
    throw std::logic_error("the Adams history carries no compensation");
  if (slot >= adams_states_.extent(0))
    throw std::out_of_range("Adams history row");
  const auto rates = Kokkos::subview(adams_rates_, slot, Kokkos::ALL, Kokkos::ALL);
  adams_rate(rate, adams_endpoint_rate_);
  Kokkos::deep_copy(execution_space_, rates, adams_endpoint_rate_);
  Kokkos::deep_copy(execution_space_,
                    Kokkos::subview(adams_states_, slot, Kokkos::ALL, Kokkos::ALL),
                    current_);
}

// The exponential Adams march's compensation at one bin, each a single strict function
// (solprop SEP 25 section 4.2) called once per bin: GCC compiles a strict function
// with its own options and does not inline it into a contracting kernel, but inlines
// the strict helpers it calls, which share those options. One call per bin, rather
// than one per helper, keeps the protected arithmetic strict on every compiler at the
// cost of a call (and Clang, which inlines every level, is unchanged).

// The predictor's carried samples `e^{s_t x} (R_t - x / dz) u_t`, written to
// `carried`, and the prediction `value + dz sum_t c_t carried_t`.
template <class Rates, class States, class Carried>
SOLPROP_NATIVE_STRICT_FUNCTION KOKKOS_INLINE_FUNCTION Complex
adams_predicted_compensation(const ExponentialWeights& weights, const Complex x,
                             const double inverse_dz, const double dz,
                             const Complex value, const Rates& rates,
                             const States& states, const Carried& carried,
                             const std::size_t column, const std::size_t index) {
  SOLPROP_NATIVE_STRICT_BLOCK
  for (std::size_t t = 0; t + 1 < weights.width; ++t) {
    const auto slot = weights.slot[t];
    Complex term = compensation_residual(rates(slot, column, index), x, inverse_dz,
                                         states(slot, column, index));
    if (weights.span[t] != 0.0)
      term = strict_mul(exponential(strict_scale(weights.span[t], x)), term);
    carried(t, column, index) = term;
  }
  Complex compensation{};
  for (std::size_t t = 0; t < weights.order; ++t)
    compensation = strict_accumulate(compensation, weights.comp_predictor[t],
                                     carried(t, column, index));
  return strict_accumulate(value, dz, compensation);
}

// The corrector's endpoint rate `sum_t w_t Phi_t` and compensation `(R - x / dz) u`,
// written to `endpoint_rate` and `endpoint`, and the correction
// `value + dz (c_0 endpoint + sum_t c_{t+1} carried_t)`.
template <class Weights, class Terms, class Points, class Carried>
SOLPROP_NATIVE_STRICT_FUNCTION KOKKOS_INLINE_FUNCTION Complex
adams_corrected_compensation(const ExponentialWeights& weights, const std::size_t n_phi,
                             const Weights& rate_weights, const Terms& phi_terms,
                             const Complex x, const double inverse_dz,
                             const Complex lab, const double dz, const Complex value,
                             const Points& endpoint_rate, const Points& endpoint,
                             const Carried& carried, const std::size_t column,
                             const std::size_t index) {
  SOLPROP_NATIVE_STRICT_BLOCK
  Complex rate_here{};
  for (std::size_t term = 0; term < n_phi; ++term)
    rate_here = strict_accumulate(rate_here, rate_weights(term),
                                  phi_terms(term, column, index));
  endpoint_rate(column, index) = rate_here;
  const Complex compensated = compensation_residual(rate_here, x, inverse_dz, lab);
  endpoint(column, index) = compensated;
  Complex compensation = strict_scale(weights.comp_corrector[0], compensated);
  for (std::size_t t = 0; t < weights.order; ++t)
    compensation = strict_accumulate(compensation, weights.comp_corrector[t + 1],
                                     carried(t, column, index));
  return strict_accumulate(value, dz, compensation);
}

// Dense observation's compensation `value + sum_t dz w_t e^{(xi - theta_t) x} term_t`.
template <class Terms>
SOLPROP_NATIVE_STRICT_FUNCTION KOKKOS_INLINE_FUNCTION Complex
adams_observed_compensation(const ExponentialWeights& weights, const std::size_t count,
                            const double xi, const double dz, const Complex chord,
                            const Complex value, const Terms& terms,
                            const std::size_t column, const std::size_t index) {
  SOLPROP_NATIVE_STRICT_BLOCK
  Complex total = value;
  for (std::size_t t = 0; t < count; ++t)
    total = strict_add(
        total,
        strict_mul(
            strict_scale(dz * weights.dense_weight[t],
                         exponential(strict_scale(xi - weights.dense_theta[t], chord))),
            terms(t, column, index)));
  return total;
}

std::vector<double> GeneralWorkspace::adams_exponential_trial(
    const double dz, const ExponentialWeights& weights,
    const std::vector<double>& divisor, const TrialScalars& step,
    const TrialScalars& sample, const std::vector<double>& rate) {
  require_adams();
  if (weights.compensated && adams_states_.extent(0) == 0)
    throw std::logic_error("the Adams history carries no compensation");
  // Beside a coupled flow the step's flow and moments are matrices on the block's bins
  // (solprop SEP 16 phases 5 and 6); the sample is the material source alone.
  const bool coupled = static_cast<bool>(flow_);
  if (coupled && adams_block_.extent(0) == 0)
    throw std::logic_error("the Adams history holds no coupled block");
  if (weights.n_rows == 0 || weights.n_rows > 3 || weights.width == 0 ||
      weights.width > adams_store_.extent(0) || weights.count > weights.width ||
      weights.order >= weights.width)
    throw std::invalid_argument("Adams coefficient tables are out of range");
  for (std::size_t t = 0; t + 1 < weights.width; ++t)
    if (weights.slot[t] >= adams_store_.extent(0))
      throw std::out_of_range("Adams history row");
  require_trial_scalars(step);
  require_trial_scalars(sample);
  adams_upload_divisor(divisor);
  // The step's exponent and flow, and its moments, once for every formula.
  upload_trial(step);
  adams_phase();
  const std::size_t section = coupled ? step.flow_section : 0;
  const bool residual = coupled && adams_applied_.extent(0) != 0;
  if (residual)
    upload_weights(execution_space_, adams_end_weights_, step.flow_weights);
  const auto columns = problem_.n_columns();
  const auto n = problem_.n_omega;
  const auto policy = Range2(execution_space_, {0, 0}, {columns, n});
  const auto bins = Kokkos::RangePolicy<ExecutionSpace>(
      coupled ? flow_execution_space_ : execution_space_, 0, n);
  const auto exponent = adams_exponent_;
  const auto flow = adams_propagator_;
  const auto moments = adams_moments_;
  const std::size_t count = weights.count;
  using BlockSlots = Kokkos::View<std::size_t*, Layout, MemorySpace>;
  const BlockSlots slots = coupled ? flow_->block_slots() : BlockSlots{};
  // The elementwise moments, off the block's bins beside a coupled flow: on them the
  // block's matrices below take their place. Without one the predictor forms them.
  if (coupled)
    Kokkos::parallel_for(
        "solprop_native.adams.moments", policy,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          if (coupled && slots(index) != 0)
            return;
          Complex phi[2 * adams_max_samples + 1];
          adams_moment_phi(weights, exponent(column, index), flow(column, index), count,
                           phi);
          for (std::size_t k = 0; k < count; ++k)
            moments(k, column, index) = adams_factorial(k) * phi[k + 1];
        });
  // The block's flow and moments, W f(x + h lambda) Y from the section's eigenbasis,
  // every bin on its route (adams_allocate refuses a flow with a bin off it).
  const auto block = adams_block_;
  const auto modes = problem_.n_modes;
  const auto left = coupled ? flow_->left() : CoupledFlowAction::ComplexView4{};
  const auto right = coupled ? flow_->right() : CoupledFlowAction::ComplexView4{};
  const auto eigenvalues =
      coupled ? flow_->eigenvalues() : CoupledFlowAction::ComplexView3{};
  const auto terms =
      residual ? flow_->residual_terms() : CoupledFlowAction::ComplexView4{};
  const auto frozen = residual ? flow_->frozen() : CoupledFlowAction::ComplexView4{};
  // A device takes a thread per (bin, channel) in the coupled passes below: its
  // lanes outnumber the bins and a channel vector does not fit a thread's
  // registers. A host keeps a task per bin, which forms a bin's shared terms once.
  if (coupled)
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    Kokkos::parallel_for(
        "solprop_native.adams.block_functions",
        Range2(flow_execution_space_, {0, 0}, {n, modes * modes}),
        KOKKOS_LAMBDA(const std::size_t index, const std::size_t pair) {
          const auto one = slots(index);
          if (one == 0)
            return;
          const auto bin = one - 1;
          const auto i = pair / modes;
          const auto j = pair % modes;
          const Complex shift = exponent(0, index);
          // One entry (i, j) of every function's matrix, a thread each: the sum over
          // the eigenvalues in their order, each function formed where it is needed.
          Complex total[adams_max_samples + 1];
          for (std::size_t k = 0; k <= count; ++k)
            total[k] = Complex{};
          Complex phi[2 * adams_max_samples + 1];
          for (std::size_t m = 0; m < modes; ++m) {
            const Complex x = shift + dz * eigenvalues(section, bin, m);
            const Complex exp_x = exponential(x);
            // The flow is exp(h B); the moments are the shifted block's (solprop SEP
            // 35 section 4.5): every eigenvalue moves by -sigma.
            adams_moment_phi(weights, x, exp_x, count, phi);
            const Complex l = left(section, bin, i, m);
            const Complex r = right(section, bin, m, j);
            total[0] += l * exp_x * r;
            for (std::size_t k = 0; k < count; ++k)
              total[k + 1] += l * (adams_factorial(k) * phi[k + 1]) * r;
          }
          for (std::size_t k = 0; k <= count; ++k)
            block(k, bin, i, j) = total[k];
        });
#else
    Kokkos::parallel_for(
        "solprop_native.adams.block_functions", bins,
        KOKKOS_LAMBDA(const std::size_t index) {
          const auto one = slots(index);
          if (one == 0)
            return;
          const auto bin = one - 1;
          const Complex shift = exponent(0, index);
          Complex values[adams_max_channels][adams_max_samples + 1];
          Complex phi[2 * adams_max_samples + 1];
          for (std::size_t m = 0; m < modes; ++m) {
            const Complex x = shift + dz * eigenvalues(section, bin, m);
            const Complex exp_x = exponential(x);
            values[m][0] = exp_x;
            adams_moment_phi(weights, x, exp_x, count, phi);
            for (std::size_t k = 0; k < count; ++k)
              values[m][k + 1] = adams_factorial(k) * phi[k + 1];
          }
          for (std::size_t k = 0; k <= count; ++k)
            for (std::size_t i = 0; i < modes; ++i)
              for (std::size_t j = 0; j < modes; ++j) {
                Complex total{};
                for (std::size_t m = 0; m < modes; ++m)
                  total += left(section, bin, i, m) * values[m][k] *
                           right(section, bin, m, j);
                block(k, bin, i, j) = total;
              }
        });
#endif
  // A varying operator's samples carried to the step's end in the chord's flow:
  // exp((1 - theta_j) x) (D(z_j) - x / dz) u_j, the difference of rates formed first.
  // Beside a coupled flow a frozen-residual flow's R_s(z_j) u_j joins each, against
  // the step's own section, and the block's bins are carried by its matrix flow.
  const auto carried = adams_carried_;
  const auto states = adams_states_;
  const auto rates = adams_rates_;
  const auto applied = adams_applied_;
  const bool compensated = weights.compensated;
  const double inverse_dz = 1.0 / dz;
  if (compensated && coupled)
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    Kokkos::parallel_for(
        "solprop_native.adams.coupled_carry",
        Range2(execution_space_, {0, 0}, {n, modes}),
        KOKKOS_LAMBDA(const std::size_t index, const std::size_t i) {
          const auto one = slots(index);
          // A sample's term on channel c, formed where it is needed: a thread holds no
          // channel vectors.
          const auto term = [&](const std::size_t slot, const std::size_t c) {
            Complex value =
                compensation_residual(rates(slot, c, index), exponent(c, index),
                                      inverse_dz, states(slot, c, index));
            if (residual && one != 0) {
              const auto bin = one - 1;
              // R_s(z_j) u_j: the sample's own generator applied once, less the
              // step's section's frozen block.
              Complex extra = applied(slot, c, index);
              for (std::size_t j = 0; j < modes; ++j)
                extra = strict_sub(extra, strict_mul(frozen(section, bin, c, j),
                                                     states(slot, j, index)));
              value = strict_add(value, extra);
            }
            return value;
          };
          for (std::size_t t = 0; t + 1 < weights.width; ++t) {
            const auto slot = weights.slot[t];
            const double span = weights.span[t];
            Complex result;
            if (span == 0.0) {
              result = term(slot, i);
            } else if (one == 0) {
              result = exponential(span * exponent(i, index)) * term(slot, i);
            } else {
              const auto bin = one - 1;
              const Complex shift = exponent(0, index);
              Complex total{};
              for (std::size_t m = 0; m < modes; ++m) {
                Complex value{};
                for (std::size_t j = 0; j < modes; ++j)
                  value += right(section, bin, m, j) * term(slot, j);
                total +=
                    left(section, bin, i, m) *
                    (exponential(span * (shift + dz * eigenvalues(section, bin, m))) *
                     value);
              }
              result = total;
            }
            carried(t, i, index) = result;
          }
        });
#else
    Kokkos::parallel_for(
        "solprop_native.adams.coupled_carry", bins,
        KOKKOS_LAMBDA(const std::size_t index) {
          const auto one = slots(index);
          Complex term[adams_max_channels];
          Complex rotated[adams_max_channels];
          for (std::size_t t = 0; t + 1 < weights.width; ++t) {
            const auto slot = weights.slot[t];
            for (std::size_t i = 0; i < modes; ++i)
              term[i] = compensation_residual(rates(slot, i, index), exponent(i, index),
                                              inverse_dz, states(slot, i, index));
            if (residual && one != 0) {
              const auto bin = one - 1;
              for (std::size_t i = 0; i < modes; ++i) {
                // R_s(z_j) u_j: the sample's own generator applied once, less the
                // step's section's frozen block.
                Complex value = applied(slot, i, index);
                for (std::size_t j = 0; j < modes; ++j)
                  value = strict_sub(value, strict_mul(frozen(section, bin, i, j),
                                                       states(slot, j, index)));
                term[i] = strict_add(term[i], value);
              }
            }
            const double span = weights.span[t];
            if (span != 0.0) {
              if (one == 0) {
                for (std::size_t i = 0; i < modes; ++i)
                  term[i] = exponential(span * exponent(i, index)) * term[i];
              } else {
                const auto bin = one - 1;
                const Complex shift = exponent(0, index);
                for (std::size_t m = 0; m < modes; ++m) {
                  Complex value{};
                  for (std::size_t j = 0; j < modes; ++j)
                    value += right(section, bin, m, j) * term[j];
                  rotated[m] =
                      exponential(span * (shift + dz * eigenvalues(section, bin, m))) *
                      value;
                }
                for (std::size_t i = 0; i < modes; ++i) {
                  Complex value{};
                  for (std::size_t m = 0; m < modes; ++m)
                    value += left(section, bin, i, m) * rotated[m];
                  term[i] = value;
                }
              }
            }
            for (std::size_t i = 0; i < modes; ++i)
              carried(t, i, index) = term[i];
          }
        });
#endif
  // The predicted lab-frame field, exp(x) u + dz sum_k M_k sum_j c_jk F_j and the
  // compensation's predictor, is the field the lab-frame source is evaluated at; the
  // accepted state is kept aside.
  const auto store = adams_store_;
  const auto current = current_;
  const auto saved = adams_saved_;
  const auto lab = adams_lab_;
  if (coupled)
    Kokkos::deep_copy(execution_space_, saved, current);
  if (!coupled)
    // One pass: the step's moments (kept for the corrector and the norms), a varying
    // operator's carried samples, the accepted state set aside, and the prediction.
    Kokkos::parallel_for(
        "solprop_native.adams.exponential_predict", policy,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          const Complex x = exponent(column, index);
          const Complex propagate = flow(column, index);
          Complex phi[2 * adams_max_samples + 1];
          const std::size_t formed = count > weights.order ? count : weights.order;
          adams_moment_phi(weights, x, propagate, formed, phi);
          for (std::size_t k = 0; k < count; ++k)
            moments(k, column, index) = adams_factorial(k) * phi[k + 1];
          const Complex accepted = current(column, index);
          saved(column, index) = accepted;
          Complex total{};
          for (std::size_t k = 0; k < weights.order; ++k) {
            Complex combined{};
            for (std::size_t t = 0; t < weights.order; ++t)
              combined += weights.predictor[t][k] *
                          adams_history(store, weights, t, column, index);
            const Complex moment = adams_factorial(k) * phi[k + 1];
            total += moment * combined;
          }
          Complex value = propagate * accepted + dz * total;
          if (compensated) {
            value = adams_predicted_compensation(weights, x, inverse_dz, dz, value,
                                                 rates, states, carried, column, index);
            lab(column, index) = value;
          }
          current(column, index) = value;
        });
  else
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    Kokkos::parallel_for(
        "solprop_native.adams.coupled_predict",
        Range2(execution_space_, {0, 0}, {n, modes}),
        KOKKOS_LAMBDA(const std::size_t index, const std::size_t i) {
          const auto one = slots(index);
          // Channel c's combinations sum_t w_tk F_t, its samples read once.
          const auto combine = [&](const std::size_t c, Complex* out) {
            Complex samples[adams_max_samples];
            for (std::size_t t = 0; t < weights.order; ++t)
              samples[t] = adams_history(store, weights, t, c, index);
            for (std::size_t k = 0; k < weights.order; ++k) {
              Complex total{};
              for (std::size_t t = 0; t < weights.order; ++t)
                total += weights.predictor[t][k] * samples[t];
              out[k] = total;
            }
          };
          Complex combined[adams_max_samples];
          Complex value{};
          if (one == 0) {
            combine(i, combined);
            Complex total{};
            for (std::size_t k = 0; k < weights.order; ++k)
              total += moments(k, i, index) * combined[k];
            value = flow(i, index) * saved(i, index) + dz * total;
          } else {
            const auto bin = one - 1;
            Complex linear{};
            Complex total{};
            for (std::size_t j = 0; j < modes; ++j) {
              linear += block(0, bin, i, j) * saved(j, index);
              combine(j, combined);
              for (std::size_t k = 0; k < weights.order; ++k)
                total += block(k + 1, bin, i, j) * combined[k];
            }
            value = linear + dz * total;
          }
          if (compensated) {
            Complex compensation{};
            for (std::size_t t = 0; t < weights.order; ++t)
              compensation = strict_accumulate(compensation, weights.comp_predictor[t],
                                               carried(t, i, index));
            value = strict_accumulate(value, dz, compensation);
            lab(i, index) = value;
          }
          current(i, index) = value;
        });
#else
    Kokkos::parallel_for(
        "solprop_native.adams.coupled_predict", bins,
        KOKKOS_LAMBDA(const std::size_t index) {
          const auto one = slots(index);
          Complex combined[adams_max_samples][adams_max_channels];
          for (std::size_t k = 0; k < weights.order; ++k)
            for (std::size_t i = 0; i < modes; ++i) {
              Complex total{};
              for (std::size_t t = 0; t < weights.order; ++t)
                total += weights.predictor[t][k] *
                         adams_history(store, weights, t, i, index);
              combined[k][i] = total;
            }
          for (std::size_t i = 0; i < modes; ++i) {
            Complex value{};
            if (one == 0) {
              Complex total{};
              for (std::size_t k = 0; k < weights.order; ++k)
                total += moments(k, i, index) * combined[k][i];
              value = flow(i, index) * saved(i, index) + dz * total;
            } else {
              const auto bin = one - 1;
              Complex linear{};
              Complex total{};
              for (std::size_t j = 0; j < modes; ++j) {
                linear += block(0, bin, i, j) * saved(j, index);
                for (std::size_t k = 0; k < weights.order; ++k)
                  total += block(k + 1, bin, i, j) * combined[k][j];
              }
              value = linear + dz * total;
            }
            if (compensated) {
              Complex compensation{};
              for (std::size_t t = 0; t < weights.order; ++t)
                compensation = strict_accumulate(
                    compensation, weights.comp_predictor[t], carried(t, i, index));
              value = strict_accumulate(value, dz, compensation);
              lab(i, index) = value;
            }
            current(i, index) = value;
          }
        });
#endif
  upload_trial(sample);
  {
    const FlowAside aside(flow_);
    adams_evaluate(sample);
  }
  // The new sample's compensation at the predicted field, untransported, with the
  // rate read on the side of the step's end the step came from; beside a coupled flow
  // a frozen-residual flow's residual at the step's end joins it.
  const auto endpoint = adams_endpoint_;
  const auto end_applied = adams_end_applied_;
  if (compensated && !coupled) {
    // The corrector forms the endpoint's rate and compensation; only the weights cross.
    require_extent(rate.size(), problem_.n_phi_terms, "Adams rate weights");
    if (adams_rate_weights_.extent(0) != rate.size())
      adams_rate_weights_ = Kokkos::View<double*, Layout, MemorySpace>(
          "solprop_native.adams.rate_weights", rate.size());
    copy_vector(execution_space_, adams_rate_weights_, rate);
  }
  if (compensated && coupled) {
    adams_rate(rate, adams_endpoint_rate_);
    const auto endpoint_rate = adams_endpoint_rate_;
    Kokkos::parallel_for(
        "solprop_native.adams.endpoint", policy,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          endpoint(column, index) = compensation_residual(
              endpoint_rate(column, index), exponent(column, index), inverse_dz,
              lab(column, index));
        });
    if (residual) {
      // The generator at the step's end applied to the predicted field: the new
      // sample's G(z_n + h) u, kept for it on acceptance.
      apply_generator(execution_space_, n, modes, slots, terms, adams_end_weights_, lab,
                      end_applied);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
      Kokkos::parallel_for(
          "solprop_native.adams.endpoint_residual",
          Range2(execution_space_, {0, 0}, {n, modes}),
          KOKKOS_LAMBDA(const std::size_t index, const std::size_t i) {
            const auto one = slots(index);
            if (one == 0)
              return;
            const auto bin = one - 1;
            Complex value = end_applied(i, index);
            for (std::size_t j = 0; j < modes; ++j)
              value = strict_sub(value,
                                 strict_mul(frozen(section, bin, i, j), lab(j, index)));
            endpoint(i, index) = strict_add(endpoint(i, index), value);
          });
#else
      Kokkos::parallel_for(
          "solprop_native.adams.endpoint_residual", bins,
          KOKKOS_LAMBDA(const std::size_t index) {
            const auto one = slots(index);
            if (one == 0)
              return;
            const auto bin = one - 1;
            for (std::size_t i = 0; i < modes; ++i) {
              Complex value = end_applied(i, index);
              for (std::size_t j = 0; j < modes; ++j)
                value = strict_sub(
                    value, strict_mul(frozen(section, bin, i, j), lab(j, index)));
              endpoint(i, index) = strict_add(endpoint(i, index), value);
            }
          });
#endif
    }
  }
  const auto stages = stages_;
  const auto source = adams_source_;
  const auto advanced = advanced_;
  const auto endpoint_rate = adams_endpoint_rate_;
  const auto rate_weights = adams_rate_weights_;
  const auto phi_terms = phi_terms_;
  const auto n_phi = problem_.n_phi_terms;
  if (!coupled)
    Kokkos::parallel_for(
        "solprop_native.adams.exponential_correct", policy,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          const Complex g = stages(0, column, index);
          source(column, index) = g;
          Complex total{};
          for (std::size_t k = 0; k < count; ++k) {
            Complex combined = weights.corrector[0][k] * g;
            for (std::size_t t = 1; t < weights.width; ++t)
              combined += weights.corrector[t][k] *
                          adams_history(store, weights, t - 1, column, index);
            total += moments(k, column, index) * combined;
          }
          Complex value = flow(column, index) * current(column, index) + dz * total;
          // The new sample's compensation at the predicted field, untransported, with
          // the rate read on the side of the step's end the step came from.
          if (compensated)
            value = adams_corrected_compensation(
                weights, n_phi, rate_weights, phi_terms, exponent(column, index),
                inverse_dz, lab(column, index), dz, value, endpoint_rate, endpoint,
                carried, column, index);
          advanced(column, index) = value;
        });
  else
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    Kokkos::parallel_for(
        "solprop_native.adams.coupled_correct",
        Range2(execution_space_, {0, 0}, {n, modes}),
        KOKKOS_LAMBDA(const std::size_t index, const std::size_t i) {
          const auto one = slots(index);
          source(i, index) = stages(0, i, index);
          // Channel c's combinations sum_t w_tk F_t with the new sample first.
          const auto combine = [&](const std::size_t c, Complex* out) {
            const Complex g = stages(0, c, index);
            Complex samples[adams_max_samples];
            for (std::size_t t = 1; t < weights.width; ++t)
              samples[t] = adams_history(store, weights, t - 1, c, index);
            for (std::size_t k = 0; k < count; ++k) {
              Complex total = weights.corrector[0][k] * g;
              for (std::size_t t = 1; t < weights.width; ++t)
                total += weights.corrector[t][k] * samples[t];
              out[k] = total;
            }
          };
          Complex combined[adams_max_samples];
          Complex value{};
          if (one == 0) {
            combine(i, combined);
            Complex total{};
            for (std::size_t k = 0; k < count; ++k)
              total += moments(k, i, index) * combined[k];
            value = flow(i, index) * current(i, index) + dz * total;
          } else {
            const auto bin = one - 1;
            Complex linear{};
            Complex total{};
            for (std::size_t j = 0; j < modes; ++j) {
              linear += block(0, bin, i, j) * current(j, index);
              combine(j, combined);
              for (std::size_t k = 0; k < count; ++k)
                total += block(k + 1, bin, i, j) * combined[k];
            }
            value = linear + dz * total;
          }
          if (compensated) {
            Complex compensation =
                strict_scale(weights.comp_corrector[0], endpoint(i, index));
            for (std::size_t t = 0; t < weights.order; ++t)
              compensation = strict_accumulate(
                  compensation, weights.comp_corrector[t + 1], carried(t, i, index));
            value = strict_accumulate(value, dz, compensation);
          }
          advanced(i, index) = value;
        });
#else
    Kokkos::parallel_for(
        "solprop_native.adams.coupled_correct", bins,
        KOKKOS_LAMBDA(const std::size_t index) {
          const auto one = slots(index);
          Complex combined[adams_max_samples][adams_max_channels];
          for (std::size_t i = 0; i < modes; ++i) {
            const Complex g = stages(0, i, index);
            source(i, index) = g;
            for (std::size_t k = 0; k < count; ++k) {
              Complex total = weights.corrector[0][k] * g;
              for (std::size_t t = 1; t < weights.width; ++t)
                total += weights.corrector[t][k] *
                         adams_history(store, weights, t - 1, i, index);
              combined[k][i] = total;
            }
          }
          for (std::size_t i = 0; i < modes; ++i) {
            Complex value{};
            if (one == 0) {
              Complex total{};
              for (std::size_t k = 0; k < count; ++k)
                total += moments(k, i, index) * combined[k][i];
              value = flow(i, index) * current(i, index) + dz * total;
            } else {
              const auto bin = one - 1;
              Complex linear{};
              Complex total{};
              for (std::size_t j = 0; j < modes; ++j) {
                linear += block(0, bin, i, j) * current(j, index);
                for (std::size_t k = 0; k < count; ++k)
                  total += block(k + 1, bin, i, j) * combined[k][j];
              }
              value = linear + dz * total;
            }
            if (compensated) {
              Complex compensation =
                  strict_scale(weights.comp_corrector[0], endpoint(i, index));
              for (std::size_t t = 0; t < weights.order; ++t)
                compensation = strict_accumulate(
                    compensation, weights.comp_corrector[t + 1], carried(t, i, index));
              value = strict_accumulate(value, dz, compensation);
            }
            advanced(i, index) = value;
          }
        });
#endif
  std::vector<double> norms(weights.n_rows, 0.0);
  const Kokkos::View<double*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      result(norms.data(), weights.n_rows);
  if (!coupled) {
    const ExponentialDefectNorms reducer{
        weights.n_rows, store,          moments, source, carried, endpoint,
        norm_mask_,     adams_divisor_, weights, dz,     n};
    Kokkos::parallel_reduce(
        "solprop_native.adams.exponential_norms",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, columns * n), reducer,
        result);
  } else {
    const CoupledDefectNorms reducer{
        weights.n_rows, store,      moments,        source,  carried, endpoint, block,
        slots,          norm_mask_, adams_divisor_, weights, dz,      modes};
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    // A device item is a (bin, channel) pair, a host item a bin.
    const auto items = n * modes;
#else
    const auto items = n;
#endif
    Kokkos::parallel_reduce(
        "solprop_native.adams.coupled_norms",
        Kokkos::RangePolicy<ExecutionSpace>(flow_execution_space_, 0, items), reducer,
        result);
  }
  execution_space_.fence("solprop_native Adams trial boundary");
  for (auto& value : norms)
    value = std::sqrt(value);
  adams_trial_ready_ = true;
  adams_trial_compensated_ = compensated;
  adams_trial_coupled_ = residual && compensated;
  return norms;
}

void GeneralWorkspace::adams_exponential_accept(const double dz, const std::size_t free,
                                                const AdamsWeights& dense) {
  require_adams();
  if (!adams_trial_ready_)
    throw std::logic_error("no Adams trial is held");
  if (free >= adams_store_.extent(0))
    throw std::out_of_range("Adams history row");
  if (dense.width > 0 && adams_slopes_.extent(0) == 0)
    throw std::logic_error("Adams dense observation storage is not allocated");
  adams_trial_ready_ = false;
  const auto columns = problem_.n_columns();
  const auto n = problem_.n_omega;
  const auto policy = Range2(execution_space_, {0, 0}, {columns, n});
  const auto store = adams_store_;
  const auto source = adams_source_;
  const auto slopes = adams_slopes_;
  adams_dense_count_ = dense.width;
  if (dense.width > 0) {
    // The step's samples and the state it starts from, before the new sample takes a
    // row: the accepted corrector's interpolant for dense observation.
    Kokkos::deep_copy(execution_space_, adams_dense_start_, current_);
    Kokkos::deep_copy(execution_space_, adams_dense_exponent_, adams_exponent_);
    Kokkos::parallel_for(
        "solprop_native.adams.exponential_dense", policy,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          slopes(0, column, index) = source(column, index);
          for (std::size_t t = 1; t < dense.width; ++t)
            slopes(t, column, index) = store(dense.slot[t - 1], column, index);
        });
    if (adams_trial_compensated_) {
      // The interpolant's compensation terms, untransported: the endpoint's and
      // (D(z_j) - x / dz) u_j of each history sample it reads.
      const auto terms = adams_dense_terms_;
      const auto endpoint = adams_endpoint_;
      const auto exponent = adams_exponent_;
      const auto states = adams_states_;
      const auto rates = adams_rates_;
      const double inverse_dz = 1.0 / dz;
      Kokkos::parallel_for(
          "solprop_native.adams.dense_terms", policy,
          // Compensation terms only: strict as one function (solprop SEP 25 4.2).
          KOKKOS_LAMBDA(const std::size_t column, const std::size_t index)
              SOLPROP_NATIVE_STRICT_FUNCTION {
                SOLPROP_NATIVE_STRICT_BLOCK
                terms(0, column, index) = endpoint(column, index);
                for (std::size_t t = 1; t < dense.width; ++t) {
                  const auto slot = dense.slot[t - 1];
                  terms(t, column, index) = compensation_residual(
                      rates(slot, column, index), exponent(column, index), inverse_dz,
                      states(slot, column, index));
                }
              });
    }
  }
  adams_dense_compensated_ = dense.width > 0 && adams_trial_compensated_;
  finish(dz);
  stash_is_current_ = false;
  // Untransported samples: nothing moves; the new sample takes row `free`, and for a
  // varying operator its state and rate the rows beside it.
  const bool compensated = adams_trial_compensated_;
  const auto states = adams_states_;
  const auto rates = adams_rates_;
  const auto lab = adams_lab_;
  const auto endpoint_rate = adams_endpoint_rate_;
  Kokkos::parallel_for(
      "solprop_native.adams.exponential_store", policy,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        store(free, column, index) = source(column, index);
        if (compensated) {
          states(free, column, index) = lab(column, index);
          rates(free, column, index) = endpoint_rate(column, index);
        }
      });
  // A frozen-residual flow's weights at the new sample, the trial's end.
  if (adams_trial_coupled_)
    Kokkos::deep_copy(execution_space_,
                      Kokkos::subview(adams_applied_, free, Kokkos::ALL, Kokkos::ALL),
                      adams_end_applied_);
  execution_space_.fence("solprop_native Adams accepted-step boundary");
}

// The field at the fraction `xi` of the last accepted step: the accepted corrector's
// interpolant integrated exactly against the flow, exp(xi x) u_n + dz sum_k
// xi^(k+1) M_k(xi x) sum_j c_jk F_j (`weights.corrector`), masked into current_.
void GeneralWorkspace::adams_exponential_observe(const double xi, const double dz,
                                                 const ExponentialWeights& weights) {
  require_adams();
  if (adams_dense_count_ == 0)
    throw std::logic_error("no Adams step is held for dense observation");
  const auto columns = problem_.n_columns();
  const auto n = problem_.n_omega;
  const std::size_t count = adams_dense_count_;
  const auto start = adams_dense_start_;
  const auto exponent = adams_dense_exponent_;
  const auto slopes = adams_slopes_;
  const auto current = current_;
  const auto terms = adams_dense_terms_;
  const bool compensated = adams_dense_compensated_;
  Kokkos::parallel_for(
      "solprop_native.adams.exponential_observe",
      Range2(execution_space_, {0, 0}, {columns, n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        const Complex x = xi * exponent(column, index);
        const Complex e = exponential(x);
        Complex phi[2 * adams_max_samples + 1];
        if (weights.shifted) {
          // The accepted step's own sigma (solprop SEP 35 section 4.1): the moments at
          // xi (x - sigma), its rotated samples, and exp(-sigma (1 - xi)) on the sum.
          const Complex residual =
              xi * (exponent(column, index) - Complex(0.0, weights.shift));
          adams_phi(residual, exponential(residual), count, phi);
        } else {
          adams_phi(x, e, count, phi);
        }
        Complex total{};
        double power = xi;
        for (std::size_t k = 0; k < count; ++k) {
          Complex combined{};
          for (std::size_t t = 0; t < count; ++t) {
            Complex slope = slopes(t, column, index);
            if (weights.shifted && t > 0)
              slope = strict_mul(
                  Complex(weights.factor_re[t - 1], weights.factor_im[t - 1]), slope);
            combined += weights.corrector[t][k] * slope;
          }
          total += (power * (adams_factorial(k) * phi[k + 1])) * combined;
          power *= xi;
        }
        if (weights.shifted)
          total = exponential(Complex(0.0, -weights.shift * (1.0 - xi))) * total;
        Complex value = e * start(column, index) + dz * total;
        if (compensated) {
          // e^{(xi - theta_j) x} carries each term; for the endpoint it is an inverse
          // flow, so a bin the step damps by more than an e-fold takes the anchor's.
          const Complex chord = exponent(column, index);
          value = adams_observed_compensation(weights, count, xi, dz, chord, value,
                                              terms, column, index);
          if (chord.real() < -1.0)
            value = e * start(column, index);
        }
        current(column, index) = value;
      });
  mask(current_);
  stash_is_current_ = false;
  execution_space_.fence("solprop_native Adams observation boundary");
}

std::vector<double> GeneralWorkspace::collect_trial() {
  batch_prepared_ = false;
  // A coupled flow's last stage is carried back by the step's own inverse action, so
  // it is not the next step's first stage; solprop's reference takes no FSAL there.
  pending_fsal_ =
      problem_.first_same_as_last && !problem_.split_step_absorber && !flow_;
  execution_space_.fence("solprop_native trial boundary");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, partials_);
  // Every column's slots, in order: the host's reduction is the caller's sum of a row.
  const auto slots = host.extent(1);
  std::vector<double> result(3 * slots);
  for (std::size_t row = 0; row < 3; ++row)
    for (std::size_t slot = 0; slot < slots; ++slot)
      result[row * slots + slot] = host(row, slot);
  return result;
}

void GeneralWorkspace::reuse_fsal() {
  const auto stages = stages_;
  const auto recycled = recycled_;
  const auto solution = solution_;
  const auto error = error_;
  const auto whole = whole_propagator_;
  const double solution_weight = problem_.solution_weights[0];
  const double error_weight = problem_.error_weights[0];
  const auto n = problem_.n_omega;
  Kokkos::parallel_for(
      "solprop_native.stage.reuse_fsal",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        const Complex value = recycled(column, index);
        stages(0, column, index) = value;
        solution(column, index) = solution_weight * value;
        error(column, index) = error_weight * value;
      });
}

TransverseChunk
GeneralWorkspace::transverse_chunk(const std::size_t first_unit) const noexcept {
  const auto units =
      std::min(transverse_units_per_chunk(), transverse_units() - first_unit);
  if (problem_.slab_geometry())
    return {first_unit * problem_.n_p, units * problem_.n_p, first_unit * problem_.n_x,
            units * problem_.n_x};
  return {first_unit, units, first_unit, units};
}

void GeneralWorkspace::transverse_forward() {
  if (slab_forward_)
    KokkosFFT::execute(*slab_forward_, transverse_, transverse_,
                       KokkosFFT::Normalization::none);
  else
    KokkosFFT::execute(*transverse_forward_, transverse_, transverse_,
                       KokkosFFT::Normalization::none);
}

void GeneralWorkspace::transverse_inverse() { transverse_inverse(transverse_); }

// The in-place inverse transverse plan on another volume of the same extents: the
// plan's own scratch or the magnetic closure's (solprop SEP 34).
void GeneralWorkspace::transverse_inverse(const ComplexView3& scratch) {
  if (slab_inverse_)
    KokkosFFT::execute(*slab_inverse_, scratch, scratch,
                       KokkosFFT::Normalization::none);
  else
    KokkosFFT::execute(*transverse_inverse_, scratch, scratch,
                       KokkosFFT::Normalization::none);
}

void GeneralWorkspace::separable_stage_input(const std::size_t stage, const double dz) {
  const auto current = current_;
  const auto stages = stages_;
  const auto tableau = tableau_;
  const auto spatial = spatial_;
  const auto delta = stage_phi_delta_;
  const auto at = stage_phi_at_;
  const auto active = active_mask_;
  const auto noise = noise_;
  const auto scratch = transverse_;
  const auto whole = whole_propagator_;
  const auto n_pol = problem_.n_pol_state;
  const auto layout = transverse_layout();
  const auto n_stages = problem_.n_stages;
  const bool add_noise = problem_.modified_noise && has_noise_;
  const double inverse_area_count = inverse_transverse_count();
  crossing_for(
      "solprop_native.free.stage_input", execution_space_, problem_.n_columns(),
      problem_.n_omega,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        const auto mode = column / n_pol;
        const auto values = spatial.value(mode, omega, active(omega) != 0U);
        Complex accumulated{};
        for (std::size_t previous = 0; previous < stage; ++previous)
          accumulated +=
              tableau(stage * n_stages + previous) * stages(previous, column, omega);
        Complex value = propagator_exp(delta(stage) * values.linear) *
                        (current(column, omega) + dz * accumulated);
        if (add_noise) {
          const double angle = at(stage) * values.linear.imag();
          value +=
              noise(column, omega) * Complex(Kokkos::cos(angle), Kokkos::sin(angle));
        }
        layout.cell(scratch, omega * n_pol + column % n_pol, mode) =
            value * inverse_area_count;
        if (stage + 1 == n_stages)
          whole(column, omega) = propagator_exp(delta(n_stages) * values.linear);
      });
}

void GeneralWorkspace::separable_load_current() {
  const auto current = current_;
  const auto scratch = transverse_;
  const auto n_pol = problem_.n_pol_state;
  const auto layout = transverse_layout();
  const double scale = inverse_transverse_count();
  crossing_for(
      "solprop_native.free.current_transverse_input", execution_space_,
      problem_.n_columns(), problem_.n_omega,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        const auto mode = column / n_pol;
        layout.cell(scratch, omega * n_pol + column % n_pol, mode) =
            scale * current(column, omega);
      });
  transverse_inverse();
}

void GeneralWorkspace::separable_pack_chunk(const TransverseChunk& chunk,
                                            const bool material) {
  const auto fine = fine_spectrum_;
  // Every crop bin of the chunk's rows is written below; only the bins outside the
  // crop need zeros, each chunk, since the inverse transform may destroy its input.
  // Rows past a short last chunk keep earlier finite values nothing reads back.
  if (!separable_gaps_.empty()) {
    const auto gaps = separable_gaps_;
    cartesian_for(
        "solprop_native.free.zero_fine_gaps", execution_space_,
        chunk.rows * problem_.n_pol_state, gaps.extent(0),
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t gap) {
          fine(row, gaps(gap)) = Complex{};
        });
  }
  const auto scratch = transverse_;
  const auto crop = crop_index_;
  const auto n_pol = problem_.n_pol_state;
  const auto layout = transverse_layout();
  const auto first = chunk.first_row;
  const auto n = problem_.n_omega;
  const auto n_to = problem_.n_to;
  const bool analytic = problem_.family == FieldFamily::field_analytic;
  // Free space's material field carries the pad scale here; the hybrid planar kind's
  // slab synthesis carries it (solprop's `Transform.synthesis`).
  const double scale =
      1.0 / static_cast<double>(material && !problem_.slab_geometry() ? problem_.n_t
                                                                      : problem_.n_to);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  tiled_crossing(
      "solprop_native.free.pack_temporal_chunk", execution_space_, chunk.rows * n_pol,
      n, true,
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        return layout.cell(scratch, omega * n_pol + row % n_pol, first + row / n_pol);
      },
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega,
                    const Complex value) {
        const double factor =
            analytic && omega > 0 && 2 * crop(omega) < n_to ? 2.0 : 1.0;
        fine(row, crop(omega)) = factor * scale * value;
      });
#else
  crossing_for(
      "solprop_native.free.pack_temporal_chunk", execution_space_, chunk.rows * n_pol,
      n, KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        const auto node = first + row / n_pol;
        // The analytic signal doubles the bins strictly between zero and the fine
        // grid's Nyquist bin, the same rule as the modal pad and the reference.
        const double factor =
            analytic && omega > 0 && 2 * crop(omega) < n_to ? 2.0 : 1.0;
        fine(row, crop(omega)) =
            factor * scale * layout.cell(scratch, omega * n_pol + row % n_pol, node);
      });
#endif
}

// The chunk's physical nodes from its temporal rows: free space's are the rows
// themselves; the hybrid planar kind's are the slab synthesis of each y row.
void GeneralWorkspace::separable_nodes(const TransverseChunk& chunk) {
  const bool real_field = problem_.family == FieldFamily::field_real;
  const auto n_to = problem_.n_to;
  const auto n_pol = problem_.n_pol_state;
  if (problem_.slab_geometry()) {
    const auto units = chunk.nodes / problem_.n_x;
    if (real_field)
      slab_contract("solprop_native.slab.synthesis", execution_space_, synthesis_,
                    modal_real_, node_real_, units, n_to);
    else
      slab_contract("solprop_native.slab.synthesis", execution_space_, synthesis_,
                    modal_complex_, node_complex_, units, n_to);
    return;
  }
  if (separable_aliased_)
    return;
  const auto time = modal_real_;
  const auto nodes = node_real_;
  const auto complex_time = modal_complex_;
  const auto complex_nodes = node_complex_;
  cartesian_for(
      "solprop_native.free.physical_columns", execution_space_, chunk.rows * n_pol,
      n_to, KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
        if (real_field)
          nodes(row, index) = time(row, index);
        else
          complex_nodes(row, index) = complex_time(row, index);
      });
}

// The hybrid planar kind's slab projection of a chunk's node source onto its y rows'
// columns: the polarisation, or one envelope-plasma current stream.
void GeneralWorkspace::slab_project(const TransverseChunk& chunk,
                                    const EnvelopeStream* stream) {
  const auto units = chunk.nodes / problem_.n_x;
  const auto n_to = problem_.n_to;
  if (stream != nullptr)
    slab_contract("solprop_native.slab.projection", execution_space_, projection_,
                  stream->node, stream->projected, units, n_to);
  else if (problem_.family == FieldFamily::envelope)
    slab_contract("solprop_native.slab.projection", execution_space_, projection_,
                  polarisation_complex_, projected_complex_, units, n_to);
  else
    slab_contract("solprop_native.slab.projection", execution_space_, projection_,
                  polarisation_real_, projected_real_, units, n_to);
}

void GeneralWorkspace::separable_finish_transform(const std::size_t stage,
                                                  const bool material) {
  const auto scratch = transverse_;
  const auto spatial = spatial_;
  const auto stages = stages_;
  const auto window = omega_window_;
  const auto active = active_mask_;
  const auto delta = stage_phi_delta_;
  const auto n_pol = problem_.n_pol_state;
  const auto layout = transverse_layout();
  crossing_for(
      "solprop_native.free.finish_transform", execution_space_, problem_.n_columns(),
      problem_.n_omega,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        const auto mode = column / n_pol;
        const auto values = spatial.value(mode, omega, active(omega) != 0U);
        Complex value = layout.cell(scratch, omega * n_pol + column % n_pol, mode);
        value *= material ? values.prefactor * window(omega)
                          : Complex(active(omega) != 0U ? 1.0 : 0.0);
        value *= propagator_exp(-delta(stage) * values.linear);
        if (material)
          stages(stage, column, omega) = value;
        else
          stages(stage, column, omega) += value;
      });
}

// Free space (solprop `NumpyRefBackend._nonlinear_free`) and the hybrid planar kind
// (`_nonlinear_slab`): the stage input to the transverse spectrum, the inverse
// transverse transform, then chunk by chunk the temporal rows to time, the material
// response at the physical nodes (for the planar kind, synthesised from and projected
// back onto the rows in time) or the in-stage absorber on the rows, and back; the
// forward transverse transform and each column's prefactor and propagator.
void GeneralWorkspace::evaluate_separable_stage(const std::size_t stage,
                                                const double dz) {
  auto* const dispersive = separable_dispersive_.get();
  const bool nonlinear =
      problem_.nonlinear || problem_.plasma || problem_.raman || dispersive != nullptr;
  const auto stages = stages_;
  const auto solution = solution_;
  const auto error = error_;
  const auto n = problem_.n_omega;
  // A material finish assigns the stage row; only an absorber-only or a linear stage
  // accumulates into (or leaves) a cleared row.
  if (!nonlinear)
    cartesian_for(
        "solprop_native.free.clear_stage", execution_space_, problem_.n_columns(), n,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
          stages(stage, column, omega) = Complex{};
        });
  separable_stage_input(stage, dz);
  for (const bool material : {true, false}) {
    if ((material && !nonlinear) || (!material && !problem_.in_stage_absorber))
      continue;
    if (!material)
      separable_stage_input(stage, dz);
    separable_core(stage, material);
    separable_finish_transform(stage, material);
  }
  const double weight = problem_.solution_weights[stage];
  const double error_weight = problem_.error_weights[stage];
  cartesian_for(
      "solprop_native.free.accumulate_stage", execution_space_, problem_.n_columns(), n,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        const Complex value = stages(stage, column, omega);
        if (stage == 0) {
          solution(column, omega) = weight * value;
          error(column, omega) = error_weight * value;
        } else {
          solution(column, omega) += weight * value;
          error(column, omega) += error_weight * value;
        }
      });
}

void GeneralWorkspace::separable_to_time(const TransverseChunk& chunk) {
  const bool tail = temporal_tail_rows_ != 0 &&
                    chunk.rows * problem_.n_pol_state == temporal_tail_rows_;
  if (problem_.family == FieldFamily::field_real) {
    if (tail)
      temporal_tail_inverse_real_.execute();
    else
      KokkosFFT::execute(*inverse_real_plan_, fine_spectrum_, modal_real_,
                         KokkosFFT::Normalization::none);
  } else {
    if (tail)
      temporal_tail_inverse_complex_.execute();
    else
      KokkosFFT::execute(*inverse_complex_plan_, fine_spectrum_, modal_complex_,
                         KokkosFFT::Normalization::none);
  }
}

void GeneralWorkspace::separable_to_frequency(const TransverseChunk& chunk) {
  const bool tail = temporal_tail_rows_ != 0 &&
                    chunk.rows * problem_.n_pol_state == temporal_tail_rows_;
  if (problem_.family == FieldFamily::envelope) {
    if (tail)
      temporal_tail_forward_complex_.execute();
    else
      KokkosFFT::execute(*forward_complex_plan_, projected_complex_,
                         projected_spectrum_, KokkosFFT::Normalization::none);
  } else {
    if (tail)
      temporal_tail_forward_real_.execute();
    else
      KokkosFFT::execute(*forward_real_plan_, projected_real_, projected_spectrum_,
                         KokkosFFT::Normalization::none);
  }
}

// The separable stage's core (solprop SEP 24 Q5): from its input in the transverse
// scratch, the material response (or, `material` false, the in-stage absorber's term)
// back into the scratch, in frequency, chunk by chunk through the temporal transforms.
// Shared by the Runge-Kutta stage and the low-storage march's passes, which differ in
// how they form the input and fold the output.
void GeneralWorkspace::separable_core(const std::size_t stage, const bool material) {
  // SEP 6's dispersive response runs chunk by chunk beside the material's response
  // column, or alone in its place (`SeparableDispersive`).
  auto* const dispersive = separable_dispersive_.get();
  const bool alone = dispersive != nullptr && dispersive->alone();
  const auto n_pol = problem_.n_pol_state;
  const auto layout = transverse_layout();
  const auto n = problem_.n_omega;
  const auto n_to = problem_.n_to;
  const auto real_time = modal_real_;
  const auto complex_time = modal_complex_;
  const auto polarisation_real = polarisation_real_;
  const auto polarisation_complex = polarisation_complex_;
  const auto projected_real = projected_real_;
  const auto projected_complex = projected_complex_;
  const auto temporal_rate = absorber_fine_;
  const auto spatial = spatial_;
  const auto spectrum = projected_spectrum_;
  const auto crop = crop_index_;
  const auto scratch = transverse_;
  const bool real_field = problem_.family == FieldFamily::field_real;
  const bool envelope = problem_.family == FieldFamily::envelope;
  const bool slab = problem_.slab_geometry();
  // Profiling regions name the core's stages for a Kokkos Tools timer (SEP 28 phase
  // 0); with no tool loaded each is one test of a flag. The kernels inside carry
  // their own labels.
  Kokkos::Profiling::pushRegion("solprop_native.free.core");
  if (material && dispersive != nullptr)
    dispersive->begin(stage);
  Kokkos::Profiling::pushRegion("solprop_native.free.transverse_inverse");
  transverse_inverse();
  Kokkos::Profiling::popRegion();
  for (std::size_t unit = 0; unit < transverse_units();
       unit += transverse_units_per_chunk()) {
    const auto chunk = transverse_chunk(unit);
    const auto first = chunk.first_row;
    Kokkos::Profiling::pushRegion("solprop_native.free.pack");
    separable_pack_chunk(chunk, material);
    Kokkos::Profiling::popRegion();
    // The response filters the packed spectrum first: the inverse transform destroys
    // it. In place, that transform gives the filtered field.
    if (material && dispersive != nullptr)
      dispersive->prepare(chunk.nodes);
    Kokkos::Profiling::pushRegion("solprop_native.free.to_time");
    separable_to_time(chunk);
    Kokkos::Profiling::popRegion();
    // The projected rows are overwritten below by the polarisation (or are the
    // polarisation, aliased) or by the absorber's term; only a planar projection or a
    // response that projects nothing of the stage's needs them zeroed first.
    const bool overwrites =
        !slab &&
        (!material || !alone || (dispersive != nullptr && dispersive->borrowed()));
    Kokkos::Profiling::pushRegion("solprop_native.free.clear_projected");
    if (!overwrites) {
      if (envelope)
        Kokkos::deep_copy(execution_space_, projected_complex_, Complex{});
      else
        Kokkos::deep_copy(execution_space_, projected_real_, 0.0);
    }
    Kokkos::Profiling::popRegion();
    if (material) {
      separable_nodes(chunk);
      Kokkos::Profiling::pushRegion("solprop_native.free.response");
      if (!alone)
        response(stage, chunk.first_node, chunk.nodes);
      if (problem_.envelope_plasma)
        envelope_response(stage, chunk.first_node, chunk.nodes, &chunk);
      if (problem_.raman)
        convolve_raman(stage, chunk.nodes);
      if (dispersive != nullptr)
        dispersive->respond(chunk.nodes);
      if (held_plasma())
        held_response(stage, chunk.first_node, chunk.nodes, &chunk);
      Kokkos::Profiling::popRegion();
      // Alone with products of its own, nothing of the stage's is projected.
      const bool stage_projects = !alone || dispersive->borrowed();
      if (stage_projects && slab)
        slab_project(chunk, nullptr);
      else if (stage_projects && !separable_aliased_)
        cartesian_for(
            "solprop_native.free.polarisation_columns", execution_space_,
            chunk.rows * n_pol, n_to,
            KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
              if (envelope)
                projected_complex(row, time) = polarisation_complex(row, time);
              else
                projected_real(row, time) = polarisation_real(row, time);
            });
    } else {
      // The collar's rate depends on y alone on the planar kind, so it acts on the
      // rows, whose time signals are the slab amplitudes' (solprop's reference).
      cartesian_for(
          "solprop_native.free.absorber_columns", execution_space_, chunk.rows * n_pol,
          n_to, KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
            const double rate =
                -0.5 * (spatial.absorber(first + row / n_pol) + temporal_rate(time));
            if (envelope)
              projected_complex(row, time) = rate * complex_time(row, time);
            else
              projected_real(row, time) =
                  rate *
                  (real_field ? real_time(row, time) : complex_time(row, time).real());
          });
    }
    const bool responds = material && dispersive != nullptr;
    if (responds)
      dispersive->finish();
    Kokkos::Profiling::pushRegion("solprop_native.free.to_frequency");
    if (!responds || !dispersive->replaces_transform()) {
      separable_to_frequency(chunk);
    }
    Kokkos::Profiling::popRegion();
    // The response's spectrum joins the chunk's, before the envelope currents'.
    if (responds)
      dispersive->combine();
    if (material && problem_.envelope_plasma)
      envelope_transform();
    if (material && held_plasma())
      held_transform();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    tiled_crossing(
        "solprop_native.free.crop_temporal_chunk", execution_space_, chunk.rows * n_pol,
        n, false,
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
          return spectrum(row, crop(omega));
        },
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega,
                      const Complex value) {
          layout.cell(scratch, omega * n_pol + row % n_pol, first + row / n_pol) =
              value;
        });
#else
    crossing_for<true>(
        "solprop_native.free.crop_temporal_chunk", execution_space_, chunk.rows * n_pol,
        n, KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
          const auto node = first + row / n_pol;
          stream_store(layout.cell(scratch, omega * n_pol + row % n_pol, node),
                       spectrum(row, crop(omega)));
        });
#endif
  }
  Kokkos::Profiling::pushRegion("solprop_native.free.transverse_forward");
  transverse_forward();
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

// The fine bins outside the crop as runs `(first bin, length)` in bin order.
std::vector<std::pair<std::size_t, std::size_t>>
GeneralWorkspace::fine_gap_runs() const {
  std::vector<bool> filled(fine_spectrum_size(), false);
  for (const auto index : problem_.crop_index)
    filled[index] = true;
  std::vector<std::pair<std::size_t, std::size_t>> runs;
  for (std::size_t index = 0; index < filled.size(); ++index) {
    if (filled[index])
      continue;
    if (!runs.empty() && runs.back().first + runs.back().second == index)
      ++runs.back().second;
    else
      runs.emplace_back(index, 1);
  }
  return runs;
}

StageTerms GeneralWorkspace::stage_terms(const std::size_t stage) const {
  StageTerms terms;
  terms.current = current_;
  terms.fine = fine_spectrum_;
  terms.whole = whole_propagator_;
  terms.solution = solution_;
  terms.error = error_;
  terms.stage_phase = stage_phase_;
  terms.stage_inverse = stage_inverse_;
  terms.noise = noise_;
  terms.flow = static_cast<bool>(flow_);
  terms.carried = terms.flow && !flow_->identity(stage);
  terms.keep_lab = terms.flow && flow_->residual();
  if (terms.flow) {
    terms.flow_input = flow_->input();
    terms.flow_drive = flow_->drive();
  }
  if (terms.keep_lab)
    terms.flow_lab = flow_->lab();
  terms.general = static_cast<bool>(general_);
  if (terms.general) {
    terms.general_scaled = general_->scaled();
    terms.general_source = general_->source();
  }
  terms.projected = projected_spectrum_;
  terms.absorber = absorber_spectrum_;
  terms.stages = stages_;
  terms.phi_terms = phi_terms_;
  terms.prefactor_terms = prefactor_terms_;
  terms.tableau = tableau_;
  terms.phi_delta = stage_phi_delta_;
  terms.phi_at = stage_phi_at_;
  terms.prefactor_weights = stage_prefactor_;
  terms.window = omega_window_;
  terms.crop = crop_index_;
  terms.gaps = fine_gaps_;
  terms.gap_ranges = fine_gaps_.extent(0) / 3;
  terms.active = active_mask_;
  terms.n = problem_.n_omega;
  terms.n_phi = problem_.n_phi_terms;
  terms.n_stages = problem_.n_stages;
  terms.n_prefactor = problem_.n_prefactor_terms;
  terms.n_to = problem_.n_to;
  terms.inverse_n_to = 1.0 / static_cast<double>(problem_.n_to);
  terms.general_pad = problem_.general_pad_scale;
  terms.add_noise = problem_.modified_noise && has_noise_;
  terms.analytic = problem_.family == FieldFamily::field_analytic;
  const bool general_active =
      terms.general && (general_->responds() || problem_.in_stage_absorber);
  terms.zero_response = terms.general
                            ? !general_active
                            : !problem_.nonlinear && !problem_.plasma &&
                                  !problem_.raman && !absorbing() && !dispersive_;
  terms.in_stage_absorber = absorbing();
  return terms;
}

// The stage's entry kernel: the deferred finish of the stage before it (when a trial
// left one pending), the stage's phase increment and the whole step's at the last
// stage, the combination, and the zero pad of the fine spectrum, in one traversal of
// the bins. The finish and the combination are each the arithmetic of their own
// kernels, so fusing them moves no rounding. A lossless problem's kernel makes no
// `exp` calls, and a lossless stage repeating the stage before's phase (the last two
// stages of a FSAL tableau, and the whole step's at the last) reuses its pair
// (`StageTerms::combine`), as numba_cpu's propagator cache does.
void GeneralWorkspace::stage_entry(const std::size_t stage, const double dz) {
  const bool repeated = lossless_phase_ && stage > 0 && entered_stage_ == stage - 1 &&
                        phase_repeats_[stage];
  entered_stage_ = stage;
  if (repeated)
    stage_entry_with<EntryPhase::repeated>(stage, dz);
  else if (lossless_phase_)
    stage_entry_with<EntryPhase::lossless>(stage, dz);
  else
    stage_entry_with<EntryPhase::general>(stage, dz);
}

template <EntryPhase Mode>
void GeneralWorkspace::stage_entry_with(const std::size_t stage, const double dz) {
  constexpr bool lossless = Mode != EntryPhase::general;
  const bool finish_previous = finish_pending_;
  finish_pending_ = false;
  if (finish_previous && (stage == 0 || pending_stage_ + 1 != stage))
    throw std::logic_error("a deferred stage finish lost its order");
  const std::size_t previous = finish_previous ? stage - 1 : 0;
  const double previous_solution = problem_.solution_weights[previous];
  const double previous_error = problem_.error_weights[previous];
  const double solution_weight = problem_.solution_weights[stage];
  const double error_weight = problem_.error_weights[stage];
  const auto terms = stage_terms(stage);
  const auto n = problem_.n_omega;
  const auto n_phi = problem_.n_phi_terms;
  const auto n_stages = problem_.n_stages;
  const auto gaps = problem_.separable_geometry() ? 0 : fine_spectrum_size() - n;
  const bool last = stage + 1 == n_stages;
  // The whole step's propagator is the last stage's when their phases are equal.
  const bool reuse_whole = last && lossless && whole_repeats_;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // On a device, one thread per bin: the host layout's blocks of 64 bins, each a work
  // item holding two 64-element arrays, gave a GPU a few dozen threads walking every
  // term, which made a varying capillary no faster on a GPU than on a CPU.
  Kokkos::parallel_for(
      "solprop_native.stage.entry",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        if (finish_previous)
          terms.finish(previous, previous_solution, previous_error, column, index);
        const Complex phase = terms.phase_sum(stage * n_phi, column, index);
        const Complex propagator = terms.template combine<Mode>(
            stage, dz, solution_weight, error_weight, column, index, phase);
        if (reuse_whole) {
          terms.whole(column, index) = propagator;
        } else if (last) {
          // At a last stage whose node is the step's end (DOPRI5's) the phase sums are
          // equal on a lossy problem too, and the stage's propagator is the step's.
          const Complex whole = terms.phase_sum(n_stages * n_phi, column, index);
          terms.whole(column, index) =
              whole == phase ? propagator : entry_exponential<lossless>(whole);
        }
        for (std::size_t gap = index; gap < gaps; gap += n)
          terms.zero_gaps(column, gap, gap + 1);
      });
#else
  const std::size_t blocks = (n + stage_block - 1) / stage_block;
  Kokkos::parallel_for(
      "solprop_native.stage.entry",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0,
                                          problem_.n_columns() * blocks),
      KOKKOS_LAMBDA(const std::size_t item) {
        const std::size_t column = item / blocks;
        const std::size_t block = item % blocks;
        const std::size_t first = block * stage_block;
        const std::size_t count = n - first < stage_block ? n - first : stage_block;
        if (finish_previous)
          for (std::size_t bin = 0; bin < count; ++bin)
            terms.finish(previous, previous_solution, previous_error, column,
                         first + bin);
        double real[stage_block];
        double imag[stage_block];
        terms.phase_block(stage * n_phi, column, first, count, real, imag);
        for (std::size_t bin = 0; bin < count; ++bin) {
          const Complex propagator = terms.template combine<Mode>(
              stage, dz, solution_weight, error_weight, column, first + bin,
              Complex(real[bin], imag[bin]));
          if (reuse_whole)
            terms.whole(column, first + bin) = propagator;
        }
        if (last && !reuse_whole) {
          terms.phase_block(n_stages * n_phi, column, first, count, real, imag);
          for (std::size_t bin = 0; bin < count; ++bin)
            terms.whole(column, first + bin) =
                entry_exponential<lossless>(Complex(real[bin], imag[bin]));
        }
        terms.zero_gaps(column, block * gaps / blocks, (block + 1) * gaps / blocks);
      });
#endif
}

// A stage's finish, or (inside a trial, when the next stage's entry follows with
// nothing between) its deferral to that entry kernel.
void GeneralWorkspace::finish_or_defer(const std::size_t stage) {
  if (defer_finish_) {
    finish_pending_ = true;
    pending_stage_ = stage;
    return;
  }
  finish_stage(stage);
}

void GeneralWorkspace::evaluate_stage(const std::size_t stage, const double dz) {
  ++rk_source_evaluations_;
  if (!flow_) {
    evaluate_stage_source(stage, dz);
    return;
  }
  // solprop `NumpyRefBackend._trial_step` on a coupled flow: the stage input is
  // carried forward by the stage's action, the source (with a frozen-residual flow's
  // explicit residual) is evaluated into the flow's drive, and carried back.
  if (!flow_->identity(stage))
    flow_->carry_input(stage, dz, current_, stages_, tableau_);
  evaluate_stage_source(stage, dz);
  flow_->add_residual(stage, stage_phase_);
  flow_->carry_back(stage, problem_.solution_weights[stage],
                    problem_.error_weights[stage], stages_, solution_, error_);
}

void GeneralWorkspace::evaluate_stage_source(const std::size_t stage, const double dz) {
  if (!defer_quadrature_)
    coarse_pending_ = false;
  if (problem_.separable_geometry()) {
    evaluate_separable_stage(stage, dz);
    return;
  }
  stage_entry(stage, dz);
  if (modal_core(stage))
    finish_or_defer(stage);
}

// The modal routes' stage core (solprop SEP 24 Q5): from the input in the fine spectrum
// (and the general route's scaled channels) to the source in the projected spectra,
// the in-stage absorber's beside it. Shared by the Runge-Kutta stage and the
// low-storage march's passes. Returns whether a source was evaluated: a stage without
// a response has none to finish.
bool GeneralWorkspace::modal_core(const std::size_t stage) {
  if (general_) {
    const bool general_active = general_->responds() || problem_.in_stage_absorber;
    if (general_active)
      general_core(stage);
    return general_active;
  }
  if (!problem_.nonlinear && !problem_.plasma && !problem_.raman && !absorbing() &&
      !dispersive_)
    return false;
  // SEP 6's dispersive response filters the padded spectrum first: the inverse
  // transform destroys it. In place, that transform gives the filtered field.
  if (dispersive_)
    dispersive_->prepare(stage);
  if (problem_.family == FieldFamily::field_real)
    KokkosFFT::execute(*inverse_real_plan_, fine_spectrum_, modal_real_,
                       KokkosFFT::Normalization::none);
  else
    KokkosFFT::execute(*inverse_complex_plan_, fine_spectrum_, modal_complex_,
                       KokkosFFT::Normalization::none);
  // A single node's synthesis and projection are folded into its response column.
  const bool folded = folds_single_node();
  if (!folded)
    scale_transform(stage);
  // Alone, the response is the stage's only source: no response column runs, and
  // unless its product is the stage's polarisation nothing is projected or
  // transformed but its own.
  const bool alone = dispersive_ && dispersive_->alone();
  const bool stage_projects = !alone || dispersive_->borrowed();
  const bool stage_synthesises =
      !folded && (!dispersive_ || !dispersive_->synthesises_nodes());
  // A projection writes every sample of its targets at the first chunk (a GEMM with
  // a zero beta, the custom kernel by assignment, the folded column by its own
  // product), so only a stage that does not project needs them zeroed.
  if (!stage_projects) {
    if (problem_.family == FieldFamily::envelope) {
      Kokkos::deep_copy(execution_space_, projected_complex_, Complex{});
      if (!projected_coarse_complex_.empty())
        Kokkos::deep_copy(execution_space_, projected_coarse_complex_, Complex{});
    } else {
      Kokkos::deep_copy(execution_space_, projected_real_, 0.0);
      if (!projected_coarse_real_.empty())
        Kokkos::deep_copy(execution_space_, projected_coarse_real_, 0.0);
    }
  }
  if (dispersive_)
    dispersive_->begin();
  bool first_chunk = true;
  std::size_t chunk = 0;
  for (std::size_t first_node = 0; first_node < problem_.n_nodes;
       first_node += problem_.node_chunk) {
    const auto node_count =
        std::min(problem_.node_chunk, problem_.n_nodes - first_node);
    if (stage_synthesises)
      synthesise(first_node, node_count);
    if (!alone)
      response(stage, first_node, node_count);
    if (problem_.raman)
      convolve_raman(stage, node_count);
    if (dispersive_)
      dispersive_->respond(chunk, first_node, node_count);
    if (held_plasma())
      held_response(stage, first_node, node_count);
    if (stage_projects && !folded && !held_radial())
      project(first_node, node_count, first_chunk);
    if (problem_.envelope_plasma)
      envelope_response(stage, first_node, node_count);
    first_chunk = false;
    ++chunk;
  }
  if (dispersive_)
    dispersive_->finish();
  if (!held_radial() && (!dispersive_ || !dispersive_->replaces_transform())) {
    if (problem_.family == FieldFamily::envelope) {
      KokkosFFT::execute(*forward_complex_plan_, projected_complex_,
                         projected_spectrum_, KokkosFFT::Normalization::none);
      if (forward_coarse_complex_plan_ && !defer_quadrature_)
        KokkosFFT::execute(*forward_coarse_complex_plan_, projected_coarse_complex_,
                           projected_coarse_spectrum_, KokkosFFT::Normalization::none);
    } else {
      KokkosFFT::execute(*forward_real_plan_, projected_real_, projected_spectrum_,
                         KokkosFFT::Normalization::none);
      if (forward_coarse_real_plan_ && !defer_quadrature_)
        KokkosFFT::execute(*forward_coarse_real_plan_, projected_coarse_real_,
                           projected_coarse_spectrum_, KokkosFFT::Normalization::none);
    }
  }
  // The response's spectrum joins the stage's, before the envelope currents'.
  if (dispersive_)
    dispersive_->combine();
  if (absorbing()) {
    form_absorber();
    if (problem_.family == FieldFamily::envelope)
      KokkosFFT::execute(*absorber_complex_plan_, absorber_complex_, absorber_spectrum_,
                         KokkosFFT::Normalization::none);
    else
      KokkosFFT::execute(*absorber_real_plan_, absorber_real_, absorber_spectrum_,
                         KokkosFFT::Normalization::none);
  }
  if (problem_.envelope_plasma)
    envelope_transform();
  if (held_plasma() && !held_radial())
    held_transform();
  if (defer_quadrature_ && project_coarse_)
    coarse_pending_ = true;
  return true;
}

// The general modal route's stage core: the shared lab-frame input, the in-stage
// absorber on the modal time field as on every route, and the general source in place
// of the legacy node pipeline; the finish applies the shared normalisation and the
// return to the interaction picture (solprop `NumpyRefBackend._nonlinear_general`).
void GeneralWorkspace::general_core(const std::size_t stage) {
  if (problem_.in_stage_absorber) {
    KokkosFFT::execute(*inverse_real_plan_, fine_spectrum_, modal_real_,
                       KokkosFFT::Normalization::none);
    form_absorber();
    KokkosFFT::execute(*absorber_real_plan_, absorber_real_, absorber_spectrum_,
                       KokkosFFT::Normalization::none);
  }
  // A guide whose every region is linear has an identically zero source, which
  // solprop's reference short-circuits in the same way.
  if (general_->responds()) {
    general_->combine(stage);
    if (general_->dispersive() != nullptr)
      general_->dispersive()->begin(stage);
    general_->evaluate();
  } else
    Kokkos::deep_copy(execution_space_, general_->source(), Complex{});
}

void GeneralWorkspace::scale_transform(const std::size_t stage) {
  const auto synthesis = synthesis_;
  const auto projection = projection_;
  const auto projection_coarse = projection_coarse_;
  const auto scaled_synthesis = scaled_synthesis_;
  const auto scaled_projection = scaled_projection_;
  const auto scaled_projection_coarse = scaled_projection_coarse_;
  const auto scaled_projection_stacked = scaled_projection_stacked_;
  const auto scaled_synthesis_complex = scaled_synthesis_complex_;
  const auto scaled_projection_complex = scaled_projection_complex_;
  const auto scaled_projection_coarse_complex = scaled_projection_coarse_complex_;
  const auto scaled_projection_stacked_complex = scaled_projection_stacked_complex_;
  const auto taper = stage_taper_;
  const auto rows = problem_.n_nodes * problem_.n_pol_transform;
  const auto modes = problem_.n_modes;
  Kokkos::parallel_for(
      "solprop_native.transform.scale",
      Range2(execution_space_, {0, 0}, {rows, problem_.n_modes}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t mode) {
        const double scale = taper(stage);
        scaled_synthesis(row, mode) = synthesis(row, mode) * scale;
        scaled_projection(mode, row) = projection(mode, row) / scale;
        if (!scaled_synthesis_complex.empty()) {
          scaled_synthesis_complex(row, mode) =
              Complex(synthesis(row, mode) * scale, 0.0);
          scaled_projection_complex(mode, row) =
              Complex(projection(mode, row) / scale, 0.0);
        }
        if (!scaled_projection_coarse.empty())
          scaled_projection_coarse(mode, row) = projection_coarse(mode, row) / scale;
        if (!scaled_projection_coarse_complex.empty())
          scaled_projection_coarse_complex(mode, row) =
              Complex(projection_coarse(mode, row) / scale, 0.0);
        if (!scaled_projection_stacked.empty()) {
          scaled_projection_stacked(mode, row) = projection(mode, row) / scale;
          scaled_projection_stacked(modes + mode, row) =
              projection_coarse(mode, row) / scale;
        }
        if (!scaled_projection_stacked_complex.empty()) {
          scaled_projection_stacked_complex(mode, row) =
              Complex(projection(mode, row) / scale, 0.0);
          scaled_projection_stacked_complex(modes + mode, row) =
              Complex(projection_coarse(mode, row) / scale, 0.0);
        }
      });
}

void GeneralWorkspace::synthesise(const std::size_t first_node,
                                  const std::size_t node_count) {
  const bool whole_transform =
      first_node == 0 && node_count == problem_.n_nodes && problem_.n_pol_state == 1;
  if (whole_transform && !radial_blas_input_.empty()) {
    packed_radial_gemm(scaled_synthesis_, modal_complex_, node_complex_);
    return;
  }
  // A radial analytic field can exceed the node budget even at 256 nodes. Its
  // row chunks are contiguous in LayoutRight, including the short final chunk:
  // alias those rows directly instead of falling through to scalar synthesis.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr bool radial_analytic_chunk = false;
#else
  const bool radial_analytic_chunk = problem_.family == FieldFamily::field_analytic &&
                                     !problem_.spatial_spectral_weights.empty();
#endif
  if ((whole_transform || radial_analytic_chunk) && use_interleaved_synthesis()) {
    Kokkos::Profiling::pushRegion(
        "solprop_native.transform.synthesis.kokkosblas_interleaved");
    using Interleaved = Kokkos::View<double**, Layout, MemorySpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // Complex samples are pairs of doubles, real part first: the same rows read as
    // twice as many real columns.
    static_assert(sizeof(Complex) == 2 * sizeof(double));
    const Interleaved modal(reinterpret_cast<double*>(modal_complex_.data()),
                            problem_.n_modes, 2 * problem_.n_to);
    const Interleaved synthesis(scaled_synthesis_.data() +
                                    first_node * problem_.n_modes,
                                node_count, problem_.n_modes);
    const Interleaved node(reinterpret_cast<double*>(node_complex_.data()), node_count,
                           2 * problem_.n_to);
    engine_gemm(transform_space_for(node_count), synthesis, modal, node, 0.0);
    Kokkos::Profiling::popRegion();
    return;
  }
  if (whole_transform && use_blas_for_synthesis()) {
    Kokkos::Profiling::pushRegion("solprop_native.transform.synthesis.kokkosblas");
    if (problem_.family == FieldFamily::envelope)
      selected_gemm(transform_space_for(node_count), scaled_synthesis_complex_,
                    modal_complex_, node_complex_, Complex{}, true);
    else
      selected_gemm(transform_space_for(node_count), scaled_synthesis_, modal_real_,
                    node_real_, 0.0, true);
    Kokkos::Profiling::popRegion();
    return;
  }
  if (use_chunked_blas_for_synthesis()) {
    Kokkos::Profiling::pushRegion(
        "solprop_native.transform.synthesis.kokkosblas_chunked");
    const auto rows = node_count * problem_.n_pol_transform;
    const auto first_row = first_node * problem_.n_pol_transform;
    const bool tail = node_count != problem_.node_chunk;
    if (problem_.family == FieldFamily::envelope) {
      const auto packed = tail ? tail_synthesis_complex_ : chunk_synthesis_complex_;
      const auto source = scaled_synthesis_complex_;
      Kokkos::parallel_for(
          "solprop_native.transform.pack_synthesis_complex",
          Range2(matrix_copy_space_for(node_count), {0, 0}, {rows, problem_.n_modes}),
          KOKKOS_LAMBDA(const std::size_t row, const std::size_t mode) {
            packed(row, mode) = source(first_row + row, mode);
          });
      const auto output = Kokkos::subview(
          node_complex_, std::make_pair(std::size_t{0}, rows), Kokkos::ALL());
      selected_gemm(transform_space_for(node_count), packed, modal_complex_, output,
                    Complex{}, true);
    } else {
      const auto packed = tail ? tail_synthesis_ : chunk_synthesis_;
      const auto source = scaled_synthesis_;
      Kokkos::parallel_for(
          "solprop_native.transform.pack_synthesis",
          Range2(matrix_copy_space_for(node_count), {0, 0}, {rows, problem_.n_modes}),
          KOKKOS_LAMBDA(const std::size_t row, const std::size_t mode) {
            packed(row, mode) = source(first_row + row, mode);
          });
      const auto output = Kokkos::subview(
          node_real_, std::make_pair(std::size_t{0}, rows), Kokkos::ALL());
      selected_gemm(transform_space_for(node_count), packed, modal_real_, output, 0.0,
                    true);
    }
    Kokkos::Profiling::popRegion();
    return;
  }
  Kokkos::Profiling::pushRegion("solprop_native.transform.synthesis.custom");
  const auto synthesis = scaled_synthesis_;
  const auto modal_real = modal_real_;
  const auto modal_complex = modal_complex_;
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto n_modes = problem_.n_modes;
  const auto n_pol_state = problem_.n_pol_state;
  const auto n_pol_transform = problem_.n_pol_transform;
  const auto n_to = problem_.n_to;
  const bool real_family = problem_.family == FieldFamily::field_real;
  const auto effective_rows = node_count * problem_.effective_node_pol();
  Kokkos::parallel_for(
      "solprop_native.transform.synthesis.custom",
      Range2(transform_space_for(node_count), {0, 0}, {effective_rows, n_to}),
      KOKKOS_LAMBDA(const std::size_t local_row, const std::size_t time) {
        const std::size_t state_pol = local_row % n_pol_state;
        const std::size_t transform_local = local_row / n_pol_state;
        const std::size_t transform_row =
            first_node * n_pol_transform + transform_local;
        if (real_family) {
          double value = 0.0;
          for (std::size_t mode = 0; mode < n_modes; ++mode)
            value += synthesis(transform_row, mode) *
                     modal_real(mode * n_pol_state + state_pol, time);
          node_real(local_row, time) = value;
        } else {
          Complex value{};
          for (std::size_t mode = 0; mode < n_modes; ++mode)
            value += synthesis(transform_row, mode) *
                     modal_complex(mode * n_pol_state + state_pol, time);
          node_complex(local_row, time) = value;
        }
      });
  Kokkos::Profiling::popRegion();
}

template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
          bool Stash>
void GeneralWorkspace::response_harmonic(const std::size_t stage,
                                         const std::size_t first_node,
                                         const std::size_t node_count) {
  if constexpr (Nonlinear && Family != FieldFamily::field_real) {
    if (problem_.harmonic_factor != 0.0) {
      response_raman<Family, Components, Plasma, Nonlinear, Stash, true>(
          stage, first_node, node_count);
      return;
    }
  }
  response_raman<Family, Components, Plasma, Nonlinear, Stash, false>(stage, first_node,
                                                                      node_count);
}

template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
          bool Stash, bool Harmonic>
void GeneralWorkspace::response_raman(const std::size_t stage,
                                      const std::size_t first_node,
                                      const std::size_t node_count) {
  if (problem_.raman) {
    if (problem_.raman_thg) {
      response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, true, true,
                    false>(stage, first_node, node_count);
    } else {
      response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, true, false,
                    false>(stage, first_node, node_count);
    }
    return;
  }
  response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, false, false,
                false>(stage, first_node, node_count);
}

template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear>
void GeneralWorkspace::response_stash(const std::size_t stage,
                                      const std::size_t first_node,
                                      const std::size_t node_count) {
  if constexpr (Plasma) {
    // Clang's host loop is measurably faster when the maximum-rate accumulator is
    // present. Keep that loop shape for every plasma problem, while the view and its
    // final-stage stores remain conditional on requested periodic statistics.
    response_harmonic<Family, Components, Plasma, Nonlinear, true>(stage, first_node,
                                                                   node_count);
    return;
  }
  response_harmonic<Family, Components, Plasma, Nonlinear, false>(stage, first_node,
                                                                  node_count);
}

template <FieldFamily Family, std::size_t Components, bool Plasma>
void GeneralWorkspace::response_nonlinear(const std::size_t stage,
                                          const std::size_t first_node,
                                          const std::size_t node_count) {
  if (problem_.nonlinear) {
    response_stash<Family, Components, Plasma, true>(stage, first_node, node_count);
  } else {
    response_stash<Family, Components, Plasma, false>(stage, first_node, node_count);
  }
}

template <FieldFamily Family, std::size_t Components>
void GeneralWorkspace::response_plasma(const std::size_t stage,
                                       const std::size_t first_node,
                                       const std::size_t node_count) {
  if (problem_.plasma && !problem_.envelope_plasma) {
    if constexpr (Family != FieldFamily::envelope) {
      // SEP 27: the band column writes the physical current separately. The
      // ordinary traversal forms and collars only the Kerr/Raman polarisation.
      if (problem_.plasma_band) {
        if (band_tiled_) {
          if (problem_.n_channels > 1)
            tiled_band_plasma_column<Family, Components, true>(stage, first_node,
                                                               node_count);
          else
            tiled_band_plasma_column<Family, Components, false>(stage, first_node,
                                                                node_count);
        } else if (problem_.n_channels > 1) {
          team_band_plasma_column<Family, Components, true>(stage, first_node,
                                                            node_count);
        } else {
          team_band_plasma_column<Family, Components, false>(stage, first_node,
                                                             node_count);
        }
        response_nonlinear<Family, Components, false>(stage, first_node, node_count);
        return;
      }
    }
    response_nonlinear<Family, Components, true>(stage, first_node, node_count);
  } else {
    response_nonlinear<Family, Components, false>(stage, first_node, node_count);
  }
}

void GeneralWorkspace::response(const std::size_t stage, const std::size_t first_node,
                                const std::size_t node_count) {
  const auto components = problem_.effective_node_pol();
  switch (problem_.family) {
  case FieldFamily::field_real:
    if (components == 1)
      response_plasma<FieldFamily::field_real, 1>(stage, first_node, node_count);
    else
      response_plasma<FieldFamily::field_real, 2>(stage, first_node, node_count);
    return;
  case FieldFamily::field_analytic:
    if (components == 1)
      response_plasma<FieldFamily::field_analytic, 1>(stage, first_node, node_count);
    else
      response_plasma<FieldFamily::field_analytic, 2>(stage, first_node, node_count);
    return;
  case FieldFamily::envelope:
    if (components == 1)
      response_plasma<FieldFamily::envelope, 1>(stage, first_node, node_count);
    else
      response_plasma<FieldFamily::envelope, 2>(stage, first_node, node_count);
  }
}

template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
          bool Stash, bool Harmonic, bool Raman, bool RamanThg, bool Multiple,
          bool Folded>
void GeneralWorkspace::response_impl(const std::size_t stage,
                                     const std::size_t first_node,
                                     const std::size_t node_count) {
  // A single node's transforms fold into the column (`folds_single_node`), compiled
  // apart so that the other columns' arithmetic and layout are untouched.
  if constexpr (Components == 1 && !Raman && !Folded) {
    if (folds_single_node()) {
      response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, Raman,
                    RamanThg, Multiple, true>(stage, first_node, node_count);
      return;
    }
  }
  if constexpr (!Multiple) {
    if (problem_.n_channels > 1 && Plasma) {
      response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, Raman,
                    RamanThg, true>(stage, first_node, node_count);
      return;
    }
  }
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto out_real = polarisation_real_;
  const auto held_current = held_current_;
  const auto out_complex = polarisation_complex_;
  const auto raman_drive = node_raman_drive_;
  const auto raman_drives = problem_.raman_drives();
  const auto window = time_window_;
  const auto carrier = carrier_;
  const auto rate_coefficients = rate_coefficients_;
  const auto stash = stash_candidate_;
  const auto density = stage_plasma_density_;
  const auto kerr_density = stage_kerr_density_;
  const auto channels = problem_.n_channels;
  const auto channel_parameters = channel_parameters_;
  const auto channel_coefficients = channel_coefficients_;
  const auto channel_density = stage_channel_density_;
  const auto channel_preionisation = stage_channel_preionisation_;
  const auto scratch = channel_scratch_;
  const auto preionisation = stage_preionisation_;
  const auto n_to = problem_.n_to;
  const double kerr_coefficient = problem_.kerr_coefficient;
  const double self_factor = problem_.self_factor;
  const double conjugate_factor = problem_.conjugate_factor;
  const double harmonic_factor = problem_.harmonic_factor;
  const double charge_mass_ratio = problem_.charge_mass_ratio;
  const double ionisation_potential = problem_.ionisation_potential;
  const double dt = problem_.dt_fine;
  const double log_field_min = problem_.log_field_min;
  const double d_log_field = problem_.d_log_field;
  const double field_min = problem_.field_min;
  const double field_max = problem_.field_max;
  const double rate_max = problem_.rate_max;
  const auto rate_intervals = problem_.rate_intervals;
  const auto n_stages = problem_.n_stages;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // The device's form (`device_response`); a mixture of channels takes the generic
  // column below.
  if constexpr (!Multiple) {
    device_response<Family, Components, Plasma, Nonlinear, Stash, Harmonic, Raman,
                    RamanThg>(stage, first_node, node_count);
    return;
  }
#endif
  // A folded single node reads the modal field and writes the projected one, each
  // through its one-element transform (`folds_single_node`); otherwise both scales
  // are one, which changes no bit.
  constexpr bool folded = Folded;
  const auto input_real = folded ? modal_real_ : node_real_;
  const auto input_complex = folded ? modal_complex_ : node_complex_;
  const auto target_real = folded ? projected_real_ : polarisation_real_;
  const auto target_complex = folded ? projected_complex_ : polarisation_complex_;
  const auto synthesis = synthesis_;
  const auto projection = projection_;
  const auto taper = stage_taper_;
  Kokkos::parallel_for(
      "solprop_native.response.column",
      Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0,
                                          node_count),
      KOKKOS_LAMBDA(const std::size_t node) {
        // CUDA extended-lambda restriction 12 requires the first capture outside
        // if-constexpr blocks. Reference binding keeps every compile-time branch.
        capture_anchor(channels, scratch, channel_preionisation, node_real,
                       node_complex, channel_coefficients, channel_parameters, dt,
                       channel_density, rate_coefficients, log_field_min, d_log_field,
                       field_min, field_max, rate_max, rate_intervals,
                       ionisation_potential, charge_mass_ratio, density,
                       conjugate_factor, self_factor, harmonic_factor, carrier,
                       raman_drive, raman_drives, out_real, held_current, window,
                       out_complex, stash, n_stages, first_node);
        const std::size_t base = node * Components;
        const double input_scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
        const double output_scale = folded ? projection(0, 0) / taper(stage) : 1.0;
        double exposure = 0.0;
        double last_rate = 0.0;
        double largest_rate = 0.0;
        double fraction = Plasma && !Multiple ? preionisation(stage) : 0.0;
        double free_first = 0.0;
        double free_second = 0.0;
        double last_first = 0.0;
        double last_second = 0.0;
        if constexpr (Multiple) {
          for (std::size_t channel = 0; channel < channels; ++channel) {
            scratch(channel, node, 0) = 0.0;
            scratch(channel, node, 1) = 0.0;
            scratch(channel, node, 2) = channel_preionisation(stage, channel);
            scratch(channel, node, 3) = 0.0;
          }
        }
        // Cache contiguous pointwise addresses to avoid repeated View stride loads.
        // Plasma retains its measured recurrence-loop layout; applying
        // the same change there regresses HE plasma by 8-9% (kokkos-reopt report).
        const auto real_input = input_real.data();
        const auto complex_input = input_complex.data();
        const auto real_target = target_real.data();
        const auto complex_target = target_complex.data();
        const auto window_data = window.data();
        const double pointwise_factor =
            Plasma ? 0.0 : kerr_coefficient * kerr_density(stage);
        for (std::size_t time = 0; time < n_to; ++time) {
          Complex first_value{};
          Complex second_value{};
          if constexpr (Family == FieldFamily::field_real) {
            first_value =
                Complex(input_scale * (Plasma ? input_real(base, time)
                                              : real_input[base * n_to + time]),
                        0.0);
            if constexpr (Components == 2)
              second_value =
                  Complex(input_scale * (Plasma ? input_real(base + 1, time)
                                                : real_input[(base + 1) * n_to + time]),
                          0.0);
          } else {
            first_value = input_scale * (Plasma ? input_complex(base, time)
                                                : complex_input[base * n_to + time]);
            if constexpr (Components == 2)
              second_value =
                  input_scale * (Plasma ? input_complex(base + 1, time)
                                        : complex_input[(base + 1) * n_to + time]);
          }
          const double modulus = squared_abs(first_value) +
                                 (Components == 2 ? squared_abs(second_value) : 0.0);
          const Complex square =
              first_value * first_value +
              (Components == 2 ? second_value * second_value : Complex{});
          if constexpr (Plasma) {
            // IntelLLVM's precise model retains an unused hypot outside this branch.
            const double magnitude =
                Components == 2 ? Kokkos::hypot(first_value.real(), second_value.real())
                                : Kokkos::abs(first_value.real());
            double guard = 0.0;
            double weight = 0.0;
            if constexpr (Multiple) {
              double work = 0.0;
              for (std::size_t channel = 0; channel < channels; ++channel) {
                const double rate = ionisation_rate(
                    magnitude, channel_coefficients, channel_parameters(channel, 0),
                    channel_parameters(channel, 1), channel_parameters(channel, 2),
                    channel_parameters(channel, 3), channel_parameters(channel, 4),
                    static_cast<std::size_t>(channel_parameters(channel, 5)),
                    static_cast<std::size_t>(channel_parameters(channel, 6)));
                double channel_exposure = scratch(channel, node, 0);
                const double previous_exposure = channel_exposure;
                if (time > 0)
                  channel_exposure += trapezoid(rate, scratch(channel, node, 1), dt);
                double channel_fraction = scratch(channel, node, 2);
                if (time == 0 || channel_exposure != previous_exposure)
                  channel_fraction = ionised_fraction(
                      channel_exposure, channel_preionisation(stage, channel));
                scratch(channel, node, 0) = channel_exposure;
                scratch(channel, node, 1) = rate;
                scratch(channel, node, 2) = channel_fraction;
                if (rate > scratch(channel, node, 3))
                  scratch(channel, node, 3) = rate;
                const double partial = channel_density(stage, channel);
                weight += partial * channel_fraction * channel_parameters(channel, 7);
                work += partial * channel_parameters(channel, 8) * rate *
                        (1.0 - channel_fraction);
              }
              guard = work > 0.0 && magnitude * magnitude > 0.0
                          ? work / (magnitude * magnitude)
                          : 0.0;
            } else {
              const double rate = ionisation_rate(magnitude, rate_coefficients,
                                                  log_field_min, d_log_field, field_min,
                                                  field_max, rate_max, rate_intervals);
              const double previous_exposure = exposure;
              if (time > 0)
                exposure += trapezoid(rate, last_rate, dt);
              last_rate = rate;
              if constexpr (Stash) {
                if (rate > largest_rate)
                  largest_rate = rate;
              }
              // Preserve the first subtraction's rounding even for pre-ionized
              // input; afterwards an unchanged rounded exposure has the same exp.
              if (time == 0 || exposure != previous_exposure)
                fraction = ionised_fraction(exposure, preionisation(stage));
              guard = rate > 0.0 && magnitude * magnitude > 0.0
                          ? ionisation_potential * rate * (1.0 - fraction) /
                                (magnitude * magnitude)
                          : 0.0;
              weight = fraction * charge_mass_ratio;
            }
            double integrand = weight * first_value.real();
            if (time > 0)
              free_first += trapezoid(integrand, last_first, dt);
            double current = free_first + guard * first_value.real();
            last_first = integrand;
            held_current(base, time) = Multiple ? current : density(stage) * current;
            if constexpr (Components == 2) {
              integrand = weight * second_value.real();
              if (time > 0)
                free_second += trapezoid(integrand, last_second, dt);
              current = free_second + guard * second_value.real();
              last_second = integrand;
              held_current(base + 1, time) =
                  Multiple ? current : density(stage) * current;
            }
          }
          const double factor =
              Plasma ? kerr_coefficient * kerr_density(stage) : pointwise_factor;
          Complex first{};
          Complex second{};
          if constexpr (Nonlinear) {
            if constexpr (Family == FieldFamily::field_real) {
              first += Complex(factor * modulus * first_value.real(), 0.0);
              if constexpr (Components == 2)
                second += Complex(factor * modulus * second_value.real(), 0.0);
            } else {
              const auto analytic = [&](const Complex value) {
                const double weighted_real = conjugate_factor * square.real();
                const double weighted_imag = conjugate_factor * square.imag();
                const double weighted_self = self_factor * modulus;
                Complex result(factor * (weighted_self * value.real() +
                                         (weighted_real * value.real() +
                                          weighted_imag * value.imag())),
                               factor * (weighted_self * value.imag() +
                                         (weighted_imag * value.real() -
                                          weighted_real * value.imag())));
                if constexpr (Harmonic)
                  result +=
                      ((factor * harmonic_factor) * carrier(time)) * square * value;
                return result;
              };
              first += analytic(first_value);
              if constexpr (Components == 2)
                second += analytic(second_value);
            }
          }
          if constexpr (Raman) {
            write_raman_drive<Components, RamanThg>(
                raman_drive, node, time, first_value, second_value, raman_drives);
          }
          if constexpr (Family != FieldFamily::envelope) {
            (Plasma ? target_real(base, time) : real_target[base * n_to + time]) =
                Raman ? first.real()
                      : output_scale * (first.real() *
                                        (Plasma ? window(time) : window_data[time]));
            if constexpr (Components == 2)
              (Plasma ? target_real(base + 1, time)
                      : real_target[(base + 1) * n_to + time]) =
                  Raman ? second.real()
                        : output_scale * (second.real() *
                                          (Plasma ? window(time) : window_data[time]));
          } else {
            (Plasma ? target_complex(base, time) : complex_target[base * n_to + time]) =
                Raman ? first
                      : output_scale *
                            (first * (Plasma ? window(time) : window_data[time]));
            if constexpr (Components == 2)
              (Plasma ? target_complex(base + 1, time)
                      : complex_target[(base + 1) * n_to + time]) =
                  Raman ? second
                        : output_scale *
                              (second * (Plasma ? window(time) : window_data[time]));
          }
        }
        if constexpr (Stash) {
          if (stash.empty() || stage + 1 != n_stages)
            return;
          if constexpr (Multiple) {
            for (std::size_t channel = 0; channel < channels; ++channel) {
              stash(first_node + node, 2 * channel) = scratch(channel, node, 2);
              stash(first_node + node, 2 * channel + 1) = scratch(channel, node, 3);
            }
          } else {
            stash(first_node + node, 0) = fraction;
            stash(first_node + node, 1) = largest_rate;
          }
        }
      });
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
// The device's response with no team tied to a node, so that a one-node fibre fills the
// device:
//
//   - the pointwise terms (Kerr, the harmonic, the Raman drive and the windowed
//     polarisation) as one two-dimensional range over (node, time), in the column's
//     arithmetic sample by sample;
//   - the trapezoid plasma as two-level scans over tiles of `plasma_tile_samples`:
//     each running integral is a kernel of one team per (row, tile) that sums the
//     tile's increments into its total (`plasma_tile_totals_`), then one per (row,
//     tile) that adds the row's earlier totals and scans the tile again from that
//     offset, forming the exposure's fraction or the current from each prefix.
template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
          bool Stash, bool Harmonic, bool Raman, bool RamanThg>
void GeneralWorkspace::device_response(const std::size_t stage,
                                       const std::size_t first_node,
                                       const std::size_t node_count) {
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto out_real = polarisation_real_;
  const auto out_complex = polarisation_complex_;
  const auto raman_drive = node_raman_drive_;
  const auto raman_drives = problem_.raman_drives();
  const auto window = time_window_;
  const auto carrier = carrier_;
  const auto kerr_density = stage_kerr_density_;
  const auto n_to = problem_.n_to;
  const double kerr_coefficient = problem_.kerr_coefficient;
  const double self_factor = problem_.self_factor;
  const double conjugate_factor = problem_.conjugate_factor;
  const double harmonic_factor = problem_.harmonic_factor;
  if constexpr (Plasma)
    device_trapezoid_plasma<Family, Components, Stash>(stage, first_node, node_count);
  Kokkos::parallel_for(
      Plasma ? "solprop_native.response.device_pointwise_after_plasma"
             : "solprop_native.response.device_pointwise",
      Range2(response_space_for(node_count), {0, 0}, {node_count, n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        capture_anchor(self_factor, conjugate_factor, node_real, node_complex, out_real,
                       out_complex, raman_drive, raman_drives, window, carrier,
                       kerr_density, kerr_coefficient, harmonic_factor, stage);
        const std::size_t base = node * Components;
        const auto value_at = [&](const std::size_t row) {
          if constexpr (Family == FieldFamily::field_real)
            return Complex(node_real(row, time), 0.0);
          else
            return Complex(node_complex(row, time));
        };
        const Complex first_value = value_at(base);
        Complex second_value{};
        if constexpr (Components == 2)
          second_value = value_at(base + 1);
        const double modulus = squared_abs(first_value) +
                               (Components == 2 ? squared_abs(second_value) : 0.0);
        const Complex square =
            first_value * first_value +
            (Components == 2 ? second_value * second_value : Complex{});
        const double factor = kerr_coefficient * kerr_density(stage);
        Complex first{};
        Complex second{};
        if constexpr (Nonlinear) {
          if constexpr (Family == FieldFamily::field_real) {
            first += Complex(factor * modulus * first_value.real(), 0.0);
            if constexpr (Components == 2)
              second += Complex(factor * modulus * second_value.real(), 0.0);
          } else {
            const auto analytic = [&](const Complex value) {
              const double weighted_real = conjugate_factor * square.real();
              const double weighted_imag = conjugate_factor * square.imag();
              const double weighted_self = self_factor * modulus;
              Complex result(
                  factor *
                      (weighted_self * value.real() +
                       (weighted_real * value.real() + weighted_imag * value.imag())),
                  factor *
                      (weighted_self * value.imag() +
                       (weighted_imag * value.real() - weighted_real * value.imag())));
              if constexpr (Harmonic)
                result += ((factor * harmonic_factor) * carrier(time)) * square * value;
              return result;
            };
            first += analytic(first_value);
            if constexpr (Components == 2)
              second += analytic(second_value);
          }
        }
        if constexpr (Raman)
          write_raman_drive<Components, RamanThg>(raman_drive, node, time, first_value,
                                                  second_value, raman_drives);
        if constexpr (Family != FieldFamily::envelope) {
          out_real(base, time) = Raman ? first.real() : first.real() * window(time);
          if constexpr (Components == 2)
            out_real(base + 1, time) =
                Raman ? second.real() : second.real() * window(time);
        } else {
          out_complex(base, time) = Raman ? first : first * window(time);
          if constexpr (Components == 2)
            out_complex(base + 1, time) = Raman ? second : second * window(time);
        }
      });
}

// The trapezoid plasma's two-level scans on a device (`device_response`). Rows of the
// rate, exposure and fraction views are the chunk's nodes; rows of the integrand and
// of the tile totals are its component rows.
template <FieldFamily Family, std::size_t Components, bool Stash>
void GeneralWorkspace::device_trapezoid_plasma(const std::size_t stage,
                                               const std::size_t first_node,
                                               const std::size_t node_count) {
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto held_current = held_current_;
  const auto rates = plasma_rate_;
  const auto exposures = plasma_exposure_;
  const auto fractions = plasma_fraction_;
  const auto integrands = plasma_current_integrand_;
  const auto totals = plasma_tile_totals_;
  const auto stash = stash_candidate_;
  const auto rate_coefficients = rate_coefficients_;
  const auto density = stage_plasma_density_;
  const auto preionisation = stage_preionisation_;
  const double charge_mass_ratio = problem_.charge_mass_ratio;
  const double ionisation_potential = problem_.ionisation_potential;
  const double dt = problem_.dt_fine;
  const double log_field_min = problem_.log_field_min;
  const double d_log_field = problem_.d_log_field;
  const double field_min = problem_.field_min;
  const double field_max = problem_.field_max;
  const double rate_max = problem_.rate_max;
  const auto rate_intervals = problem_.rate_intervals;
  const auto n_to = problem_.n_to;
  const auto samples = static_cast<int>(n_to);
  const auto width = static_cast<int>(plasma_tile_samples);
  const auto tiles = static_cast<int>(plasma_tiles_);
  const auto last = n_to - 1;
  const bool stashing = Stash && !stash.empty() && stage + 1 == problem_.n_stages;
  const std::size_t rows = node_count * Components;
  const auto team_size = static_cast<int>(response_team_size());
  // The field a row reads: a real row's value, or an analytic or envelope row's real
  // part.
  const auto raw = [=] KOKKOS_FUNCTION(const std::size_t row, const std::size_t time) {
    capture_anchor(node_real, node_complex);
    if constexpr (Family == FieldFamily::field_real)
      return node_real(row, time);
    else
      return node_complex(row, time).real();
  };

  // 1. The rates.
  Kokkos::parallel_for(
      "solprop_native.response.device_plasma_rate",
      Range2(response_space_for(node_count), {0, 0}, {node_count, n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        const std::size_t base = node * Components;
        const double first = raw(base, time);
        double magnitude = 0.0;
        if constexpr (Components == 2)
          magnitude = Kokkos::hypot(first, raw(base + 1, time));
        else
          magnitude = Kokkos::abs(first);
        rates(node, time) =
            ionisation_rate(magnitude, rate_coefficients, log_field_min, d_log_field,
                            field_min, field_max, rate_max, rate_intervals);
      });
  // 2. The exposure: tile totals, then each tile's prefixes from its offset, and the
  // fraction from each.
  const auto exposure_local = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(rates, totals, samples, width, tiles, dt);
    const auto node = static_cast<std::size_t>(team.league_rank() / tiles);
    const int tile = team.league_rank() % tiles;
    const int first = tile * width;
    const int end = Kokkos::min(first + width, samples);
    const double total = team_trapezoid_total(
        team,
        [&](const int sample) { return rates(node, static_cast<std::size_t>(sample)); },
        first, end, dt);
    Kokkos::single(Kokkos::PerTeam(team),
                   [&] { totals(node, static_cast<std::size_t>(tile)) = total; });
  };
  Kokkos::parallel_for("solprop_native.response.device_plasma_exposure_tiles",
                       TeamPolicy(response_space_for(node_count),
                                  static_cast<int>(node_count) * tiles, team_size),
                       exposure_local);
  const auto exposure_finish = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(rates, totals, exposures, fractions, preionisation, samples, width,
                   tiles, dt, stage);
    const auto node = static_cast<std::size_t>(team.league_rank() / tiles);
    const int tile = team.league_rank() % tiles;
    const int first = tile * width;
    const int end = Kokkos::min(first + width, samples);
    const double offset = team_tile_offset(team, totals, node, tile);
    team_trapezoid_finish(
        team,
        [&](const int sample) { return rates(node, static_cast<std::size_t>(sample)); },
        first, end, dt, offset,
        [&](const int sample, const double prefix) {
          const auto time = static_cast<std::size_t>(sample);
          exposures(node, time) = prefix;
          fractions(node, time) = ionised_fraction(prefix, preionisation(stage));
        });
  };
  Kokkos::parallel_for("solprop_native.response.device_plasma_exposure",
                       TeamPolicy(response_space_for(node_count),
                                  static_cast<int>(node_count) * tiles, team_size),
                       exposure_finish);
  // 3. The free-current integrand, per component row.
  Kokkos::parallel_for(
      "solprop_native.response.device_plasma_integrand",
      Range2(response_space_for(node_count), {0, 0}, {rows, n_to}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
        integrands(row, time) =
            fractions(row / Components, time) * charge_mass_ratio * raw(row, time);
      });
  // 4. The free current: tile totals, then each tile's prefixes from its offset and
  // the current from each.
  const auto current_local = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(integrands, totals, samples, width, tiles, dt);
    const auto row = static_cast<std::size_t>(team.league_rank() / tiles);
    const int tile = team.league_rank() % tiles;
    const int first = tile * width;
    const int end = Kokkos::min(first + width, samples);
    const double total = team_trapezoid_total(
        team,
        [&](const int sample) {
          return integrands(row, static_cast<std::size_t>(sample));
        },
        first, end, dt);
    Kokkos::single(Kokkos::PerTeam(team),
                   [&] { totals(row, static_cast<std::size_t>(tile)) = total; });
  };
  Kokkos::parallel_for("solprop_native.response.device_plasma_current_tiles",
                       TeamPolicy(response_space_for(node_count),
                                  static_cast<int>(rows) * tiles, team_size),
                       current_local);
  const auto current_finish = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(integrands, totals, held_current, rates, fractions, density, samples,
                   width, tiles, dt, ionisation_potential, stage);
    const auto row = static_cast<std::size_t>(team.league_rank() / tiles);
    const int tile = team.league_rank() % tiles;
    const int first = tile * width;
    const int end = Kokkos::min(first + width, samples);
    const double offset = team_tile_offset(team, totals, row, tile);
    team_trapezoid_finish(
        team,
        [&](const int sample) {
          return integrands(row, static_cast<std::size_t>(sample));
        },
        first, end, dt, offset,
        [&](const int sample, const double prefix) {
          const auto time = static_cast<std::size_t>(sample);
          const std::size_t node = row / Components;
          const std::size_t base = node * Components;
          const double first_field = raw(base, time);
          const double second_field = Components == 2 ? raw(base + 1, time) : 0.0;
          const double magnitude_squared =
              first_field * first_field + second_field * second_field;
          const double rate = rates(node, time);
          const double guard = rate > 0.0 && magnitude_squared > 0.0
                                   ? ionisation_potential * rate *
                                         (1.0 - fractions(node, time)) /
                                         magnitude_squared
                                   : 0.0;
          held_current(row, time) = density(stage) * (prefix + guard * raw(row, time));
        });
  };
  Kokkos::parallel_for("solprop_native.response.device_plasma_current",
                       TeamPolicy(response_space_for(node_count),
                                  static_cast<int>(rows) * tiles, team_size),
                       current_finish);
  // 5. The stash: the last sample's fraction and the largest rate, per node.
  if (stashing) {
    const auto peak = KOKKOS_LAMBDA(const TeamMember& team) {
      capture_anchor(rates, fractions, stash, samples, last, first_node);
      const auto node = static_cast<std::size_t>(team.league_rank());
      double largest = 0.0;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, samples),
          [&](const int sample, double& update) {
            const double rate = rates(node, static_cast<std::size_t>(sample));
            if (rate > update)
              update = rate;
          },
          Kokkos::Max<double>(largest));
      Kokkos::single(Kokkos::PerTeam(team), [&] {
        stash(first_node + node, 0) = fractions(node, last);
        stash(first_node + node, 1) = largest;
      });
    };
    Kokkos::parallel_for("solprop_native.response.device_plasma_stash",
                         TeamPolicy(response_space_for(node_count),
                                    static_cast<int>(node_count), team_size),
                         peak);
  }
}
#endif

static_assert(band_max_half + 1 == static_cast<int>(plasma_max_taps),
              "the team band filter's tap bound is the problem's");

namespace {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
/** Whether the band column skips each row's quiet samples (`band_support`): host only.
 */
constexpr bool band_windows = false;
#else
constexpr bool band_windows = true;
/**
 * The fewest samples a host thread of a band team takes from a row: below it a thread's
 * share of a pass no longer repays the team's barriers between the passes. Measured on
 * the mode-averaged helium RDW case (n_to = 8192, one node) on a loaded Apple M-series
 * host: teams of two and four threads ran the band column faster than one, six slower;
 * 2048 caps that row at four. To be re-measured on each host class (solprop SEP 25).
 */
constexpr std::size_t band_host_samples_per_thread = 2048;
#endif

/** The first and last indices whose value is non-zero, as a Kokkos MinMax reduction. */
using BandLive = Kokkos::MinMaxScalar<int>;

/** The band window of a row whose non-zero samples `live` found. */
KOKKOS_INLINE_FUNCTION BandWindow band_window_of(const BandLive& live,
                                                 const int samples, const int half) {
  if constexpr (band_windows)
    return band_support(live.min_val, live.max_val, samples, half);
  else
    return {0, samples - 1};
}

/** Count `index` into `live` when `value` is not exactly zero. */
KOKKOS_INLINE_FUNCTION void band_note(BandLive& live, const int index,
                                      const double value) {
  if (value != 0.0) {
    live.min_val = Kokkos::min(live.min_val, index);
    live.max_val = Kokkos::max(live.max_val, index);
  }
}
} // namespace

int GeneralWorkspace::band_host_team_size(const std::size_t node_count) const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  static_cast<void>(node_count);
  return 1;
#else
  // solprop SEP 25 section 6.7.5: share the threads between the chunk's nodes, so a
  // chunk of fewer nodes than threads runs each node's passes over chunks of t.
  const auto threads = static_cast<std::size_t>(
      std::max(1, response_space_for(node_count).concurrency()));
  const auto per_node =
      std::max<std::size_t>(1, threads / std::max<std::size_t>(1, node_count));
  const auto by_samples =
      std::max<std::size_t>(1, problem_.n_to / band_host_samples_per_thread);
  return static_cast<int>(std::min(per_node, by_samples));
#endif
}

std::size_t GeneralWorkspace::band_league(const std::size_t node_count) const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  return node_count;
#else
  return std::min(node_count, band_slots_);
#endif
}

template <class Functor>
Kokkos::TeamPolicy<ExecutionSpace>
GeneralWorkspace::band_team_policy(const std::size_t node_count,
                                   const Functor& functor) const {
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  constexpr auto policy_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (node_count > policy_limit || problem_.n_to > policy_limit)
    throw std::length_error("band team column extent exceeds signed 32-bit range");
  const auto league = static_cast<int>(band_league(node_count));
  const auto scratch =
      Kokkos::PerTeam(band_staging_bytes<band_staging, ExecutionSpace>());
  // The widest team this kernel can launch with its scratch: a device's registers per
  // thread can hold the band column below the configured width.
  Policy probe(response_space_for(node_count), league, 1);
  probe.set_scratch_size(0, scratch);
  const int widest = probe.team_size_max(functor, Kokkos::ParallelForTag());
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const int requested = static_cast<int>(response_team_size());
#else
  const int requested = band_host_team_size(node_count);
#endif
  Policy policy(response_space_for(node_count), league,
                std::max(1, std::min(requested, widest)));
  policy.set_scratch_size(0, scratch);
  return policy;
}

// solprop SEP 18's band rule as a team column in whole-row passes (solprop SEP 25
// section 6.7.4; include/solprop_native/plasma.hpp): per channel the rates, their
// filter and scan give the exposure; the fraction (floored where it is formed), the
// loss guard and the free-current weight follow pointwise, a mixture adding each
// channel's share in channel order. Per component the free-current integrand is
// filtered and scanned, and the scan's store forms the current from it, the guard and
// the field. SEP 27 uses this current directly, without a third running integral.
//
// The rows, by slot (a team's own; the node on a device): `plasma_rate_` the raw rates,
// `plasma_fraction_` the filtered row, `plasma_exposure_` the exposure,
// `plasma_free_current_` the loss guard (a mixture's work until the last channel),
// `plasma_outer_current_` a mixture's weight, `plasma_current_integrand_` the
// free-current integrand. The separate `held_current_` stores the total current. Each
// filter and scan runs on its row's live window (`band_support`) and writes the rest:
// zero before it, the final prefix after it, and an unchanged exposure's fraction.
template <FieldFamily Family, std::size_t Components, bool Multiple, bool Folded>
void GeneralWorkspace::team_band_plasma_column(const std::size_t stage,
                                               const std::size_t first_node,
                                               const std::size_t node_count) {
  if constexpr (Components == 1 && !Folded) {
    if (folds_single_node()) {
      team_band_plasma_column<Family, Components, Multiple, true>(stage, first_node,
                                                                  node_count);
      return;
    }
  }
  static_assert(Family != FieldFamily::envelope,
                "the band rule integrates a field-grid plasma column");
  using TeamMember = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  // A folded single node reads the modal field through its one-element synthesis
  // (`folds_single_node`).
  constexpr bool folded = Folded;
  const auto node_real = folded ? modal_real_ : node_real_;
  const auto node_complex = folded ? modal_complex_ : node_complex_;
  const auto synthesis = synthesis_;
  const auto taper = stage_taper_;
  const auto out = held_current_;
  const auto rates = plasma_rate_;
  const auto filtered = plasma_fraction_;
  const auto exposures = plasma_exposure_;
  const auto guards = plasma_free_current_;
  const auto weights = plasma_outer_current_;
  const auto integrands = plasma_current_integrand_;
  const auto taps_view = plasma_taps_;
  const auto stash = stash_candidate_;
  const bool stashing = !stash.empty() && stage + 1 == problem_.n_stages;
  const auto rate_coefficients = rate_coefficients_;
  const auto channel_parameters = channel_parameters_;
  const auto channel_coefficients = channel_coefficients_;
  const auto channel_density = stage_channel_density_;
  const auto channel_preionisation = stage_channel_preionisation_;
  const auto density = stage_plasma_density_;
  const auto preionisation = stage_preionisation_;
  const double charge_mass_ratio = problem_.charge_mass_ratio;
  const double ionisation_potential = problem_.ionisation_potential;
  const double dt = problem_.dt_fine;
  const double log_field_min = problem_.log_field_min;
  const double d_log_field = problem_.d_log_field;
  const double field_min = problem_.field_min;
  const double field_max = problem_.field_max;
  const double rate_max = problem_.rate_max;
  const auto rate_intervals = problem_.rate_intervals;
  const std::size_t channels = Multiple ? problem_.n_channels : 1;
  const auto samples = static_cast<int>(problem_.n_to);
  const auto half = static_cast<int>(problem_.plasma_half_width());
  const std::size_t last = problem_.n_to - 1;
  const std::size_t league = band_league(node_count);
  const auto column = KOKKOS_LAMBDA(const TeamMember& team) {
    // CUDA extended-lambda restriction 12 requires the first capture outside
    // if-constexpr blocks.
    capture_anchor(node_real, node_complex, out, rates, filtered, exposures, guards,
                   weights, integrands, taps_view, stash, stashing, rate_coefficients,
                   channel_parameters, channel_coefficients, channel_density,
                   channel_preionisation, density, preionisation, charge_mass_ratio,
                   ionisation_potential, dt, log_field_min, d_log_field, field_min,
                   field_max, rate_max, rate_intervals, channels, samples, half, last,
                   first_node, stage, league, synthesis, taper);
    const auto slot = static_cast<std::size_t>(team.league_rank());
    const std::size_t own = slot * Components;
    double taps[plasma_max_taps] = {};
    for (int k = 0; k <= half; ++k)
      taps[k] = taps_view(static_cast<std::size_t>(k));
    const auto tile = band_tile<band_staging>(team);
    const double input_scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
    for (std::size_t node = slot; node < node_count; node += league) {
      const std::size_t base = node * Components;
      const auto field_at = [&](const std::size_t row, const std::size_t time) {
        if constexpr (Family == FieldFamily::field_real)
          return input_scale * node_real(row, time);
        else
          return input_scale * node_complex(row, time).real();
      };
      const auto magnitude_at = [&](const std::size_t time) {
        const double first = field_at(base, time);
        if constexpr (Components == 2)
          return Kokkos::hypot(first, field_at(base + 1, time));
        else
          return Kokkos::abs(first);
      };
      const auto rate_of = [&](const double magnitude, const std::size_t channel) {
        if constexpr (Multiple)
          return ionisation_rate(
              magnitude, channel_coefficients, channel_parameters(channel, 0),
              channel_parameters(channel, 1), channel_parameters(channel, 2),
              channel_parameters(channel, 3), channel_parameters(channel, 4),
              static_cast<std::size_t>(channel_parameters(channel, 5)),
              static_cast<std::size_t>(channel_parameters(channel, 6)));
        else
          return ionisation_rate(magnitude, rate_coefficients, log_field_min,
                                 d_log_field, field_min, field_max, rate_max,
                                 rate_intervals);
      };
      BandLive integrand_live;
      integrand_live.min_val = samples;
      integrand_live.max_val = -1;

      // 1. Per channel: rates -> filter -> scan -> exposure, then the fraction, the
      // guard and the weight pointwise.
      for (std::size_t channel = 0; channel < channels; ++channel) {
        BandLive rate_live;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, samples),
            [&](const int sample, BandLive& update) {
              const auto time = static_cast<std::size_t>(sample);
              const double rate = rate_of(magnitude_at(time), channel);
              rates(slot, time) = rate;
              band_note(update, sample, rate);
            },
            Kokkos::MinMax<int>(rate_live));
        team.team_barrier();
        const BandWindow window = band_window_of(rate_live, samples, half);
        if (window.first <= window.last)
          team_band_filter<band_staging>(team, tile, rates, slot, filtered, slot,
                                         samples, taps, half, window.first,
                                         window.last + 1);
        const double total =
            team_trapezoid_window(team, filtered, slot, window, samples, dt,
                                  [&](const int sample, const double prefix) {
                                    exposures(slot, static_cast<std::size_t>(sample)) =
                                        prefix;
                                  });
        const double initial =
            Multiple ? channel_preionisation(stage, channel) : preionisation(stage);
        // Outside [window.first, window.last + 1] the exposure is 0 before and `total`
        // after, so the fraction there is one of two values (band_support).
        const int changing_end = Kokkos::min(window.last + 2, samples);
        const double before = band_fraction(0.0, initial);
        const double after = band_fraction(total, initial);
        const auto fraction_at = [&](const int sample) {
          if (sample < window.first)
            return before;
          if (sample >= changing_end)
            return after;
          return band_fraction(exposures(slot, static_cast<std::size_t>(sample)),
                               initial);
        };
        if constexpr (Multiple) {
          const double partial = channel_density(stage, channel);
          const double charge = channel_parameters(channel, 7);
          const double potential = channel_parameters(channel, 8);
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, samples), [&](const int sample) {
                const auto time = static_cast<std::size_t>(sample);
                const double fraction = fraction_at(sample);
                const double rate = rates(slot, time);
                const double share = partial * fraction * charge;
                const double paid = partial * potential * rate * (1.0 - fraction);
                const double weight = channel == 0 ? 0.0 : weights(slot, time);
                const double work = channel == 0 ? 0.0 : guards(slot, time);
                weights(slot, time) = weight + share;
                guards(slot, time) = work + paid;
              });
        } else {
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, samples),
              [&](const int sample, BandLive& update) {
                const auto time = static_cast<std::size_t>(sample);
                const double fraction = fraction_at(sample);
                const double magnitude = magnitude_at(time);
                const double squared = magnitude * magnitude;
                guards(slot, time) = band_loss_scale(rates(slot, time), fraction,
                                                     squared, ionisation_potential);
                const double weight = fraction * charge_mass_ratio;
                for (std::size_t component = 0; component < Components; ++component) {
                  const double value = weight * field_at(base + component, time);
                  integrands(own + component, time) = value;
                  band_note(update, sample, value);
                }
              },
              Kokkos::MinMax<int>(integrand_live));
        }
        if (stashing) {
          double largest = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, samples),
              [&](const int sample, double& update) {
                const double rate = rates(slot, static_cast<std::size_t>(sample));
                if (rate > update)
                  update = rate;
              },
              Kokkos::Max<double>(largest));
          Kokkos::single(Kokkos::PerTeam(team), [&] {
            stash(first_node + node, 2 * channel) =
                band_fraction(exposures(slot, last), initial);
            stash(first_node + node, 2 * channel + 1) = largest > 0.0 ? largest : 0.0;
          });
        }
        team.team_barrier();
      }
      if constexpr (Multiple) {
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, samples),
            [&](const int sample, BandLive& update) {
              const auto time = static_cast<std::size_t>(sample);
              const double magnitude = magnitude_at(time);
              const double squared = magnitude * magnitude;
              guards(slot, time) =
                  band_loss_scale(1.0, 0.0, squared, guards(slot, time));
              const double weight = weights(slot, time);
              for (std::size_t component = 0; component < Components; ++component) {
                const double value = weight * field_at(base + component, time);
                integrands(own + component, time) = value;
                band_note(update, sample, value);
              }
            },
            Kokkos::MinMax<int>(integrand_live));
      }
      team.team_barrier();

      // 2. Per component: the free current, and from it the current.
      const BandWindow free_window = band_window_of(integrand_live, samples, half);
      for (std::size_t component = 0; component < Components; ++component) {
        const std::size_t row = base + component;
        if (free_window.first <= free_window.last)
          team_band_filter<band_staging>(team, tile, integrands, own + component,
                                         filtered, slot, samples, taps, half,
                                         free_window.first, free_window.last + 1);
        team_trapezoid_window(
            team, filtered, slot, free_window, samples, dt,
            [&](const int sample, const double prefix) {
              const auto time = static_cast<std::size_t>(sample);
              const double drive = guards(slot, time) * field_at(row, time);
              out(row, time) =
                  Multiple ? prefix + drive : density(stage) * (prefix + drive);
            });
      }
    }
  };
  Kokkos::parallel_for("solprop_native.response.team_band_plasma_column",
                       band_team_policy(node_count, column), column);
}

namespace {
/**
 * A band column's node field: `scale` times the real row of a field-family node (or of
 * the folded single node's modal row), and its magnitude over a node's components.
 */
template <FieldFamily Family, std::size_t Components> struct BandField {
  RealView2 real;
  ComplexView2 complex;
  KOKKOS_INLINE_FUNCTION double at(const std::size_t row, const std::size_t time,
                                   const double scale) const {
    if constexpr (Family == FieldFamily::field_real)
      return scale * real(row, time);
    else
      return scale * complex(row, time).real();
  }
  KOKKOS_INLINE_FUNCTION double
  magnitude(const std::size_t base, const std::size_t time, const double scale) const {
    const double first = at(base, time, scale);
    if constexpr (Components == 2)
      return Kokkos::hypot(first, at(base + 1, time, scale));
    else
      return Kokkos::abs(first);
  }
};

/** A band column's ionisation rate: the single gas's table, or a mixture channel's. */
template <bool Multiple> struct BandRate {
  RealView coefficients;
  RealView2 parameters;
  RealView channel_coefficients;
  double log_field_min{}, d_log_field{}, field_min{}, field_max{}, rate_max{};
  std::size_t intervals{};
  KOKKOS_INLINE_FUNCTION double operator()(const double magnitude,
                                           const std::size_t channel) const {
    if constexpr (Multiple)
      return ionisation_rate(magnitude, channel_coefficients, parameters(channel, 0),
                             parameters(channel, 1), parameters(channel, 2),
                             parameters(channel, 3), parameters(channel, 4),
                             static_cast<std::size_t>(parameters(channel, 5)),
                             static_cast<std::size_t>(parameters(channel, 6)));
    else
      return ionisation_rate(magnitude, coefficients, log_field_min, d_log_field,
                             field_min, field_max, rate_max, intervals);
  }
};
} // namespace

template <class Functor>
Kokkos::TeamPolicy<ExecutionSpace>
GeneralWorkspace::tiled_band_policy(const std::size_t rows, const bool scratch,
                                    const Functor& functor) const {
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  constexpr auto policy_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (band_tile_count_ == 0 || rows > policy_limit / band_tile_count_ ||
      problem_.n_to > policy_limit)
    throw std::length_error("tiled band column extent exceeds signed 32-bit range");
  const auto league = static_cast<int>(rows * band_tile_count_);
  const auto bytes = scratch ? band_tile_bytes<ExecutionSpace>(band_tile_) : 0;
  Policy probe(response_execution_space_, league, 1);
  probe.set_scratch_size(0, Kokkos::PerTeam(bytes));
  const int widest = probe.team_size_max(functor, Kokkos::ParallelForTag());
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const auto requested = static_cast<int>(std::min(response_team_size(), band_tile_));
#else
  const int requested = std::max(1, band_tile_threads_);
#endif
  Policy policy(response_execution_space_, league,
                std::max(1, std::min(requested, widest)));
  policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
  return policy;
}

// The band column in tiles (solprop SEP 25 section 6.7.6;
// include/solprop_native/plasma.hpp): each running integral is a local kernel of one
// team per (row, tile), which loads, filters and scans its tile from zero
// (`team_band_tile`), then a finish kernel of one team per (row, tile), which adds the
// row's earlier tile totals (`team_tile_offset`) and forms the pass's product from the
// prefix, in the team column's arithmetic:
//
//   1. per channel: the local kernel forms the rates in its window (keeping its own
//      tile's in `plasma_rate_`) and the exposure's local prefixes
//      (`plasma_exposure_`); the finish forms the fraction, the loss guard
//      (`plasma_free_current_`) and the free-current integrand
//      (`plasma_current_integrand_`), a mixture accumulating its weight
//      (`plasma_outer_current_`) and work channel by channel;
//   2. per component row: the integrand's local prefixes (`plasma_outer_current_`,
//      free after the first pass), then the current (`held_current_`).
//
// Rows are by node within the chunk, the totals by row (`band_tile_totals_`). The
// stash's fraction comes from the last sample's finish and its largest rate from a
// reduction over the rate row. No window is skipped: every sample is formed.
template <FieldFamily Family, std::size_t Components, bool Multiple, bool Folded>
void GeneralWorkspace::tiled_band_plasma_column(const std::size_t stage,
                                                const std::size_t first_node,
                                                const std::size_t node_count) {
  if constexpr (Components == 1 && !Folded) {
    if (folds_single_node()) {
      tiled_band_plasma_column<Family, Components, Multiple, true>(stage, first_node,
                                                                   node_count);
      return;
    }
  }
  static_assert(Family != FieldFamily::envelope,
                "the band rule integrates a field-grid plasma column");
  using TeamMember = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  constexpr bool folded = Folded;
  const BandField<Family, Components> field{folded ? modal_real_ : node_real_,
                                            folded ? modal_complex_ : node_complex_};
  const BandRate<Multiple> rate{
      rate_coefficients_,     channel_parameters_,  channel_coefficients_,
      problem_.log_field_min, problem_.d_log_field, problem_.field_min,
      problem_.field_max,     problem_.rate_max,    problem_.rate_intervals};
  const auto synthesis = synthesis_;
  const auto taper = stage_taper_;
  const auto out = held_current_;
  const auto rates = plasma_rate_;
  const auto exposures = plasma_exposure_;
  const auto guards = plasma_free_current_;
  const auto weights = plasma_outer_current_;
  const auto locals = plasma_outer_current_;
  const auto integrands = plasma_current_integrand_;
  const auto totals = band_tile_totals_;
  const auto taps_view = plasma_taps_;
  const auto stash = stash_candidate_;
  const bool stashing = !stash.empty() && stage + 1 == problem_.n_stages;
  const auto channel_parameters = channel_parameters_;
  const auto channel_density = stage_channel_density_;
  const auto channel_preionisation = stage_channel_preionisation_;
  const auto density = stage_plasma_density_;
  const auto preionisation = stage_preionisation_;
  const double charge_mass_ratio = problem_.charge_mass_ratio;
  const double ionisation_potential = problem_.ionisation_potential;
  const double dt = problem_.dt_fine;
  const std::size_t channels = Multiple ? problem_.n_channels : 1;
  const auto samples = static_cast<int>(problem_.n_to);
  const auto half = static_cast<int>(problem_.plasma_half_width());
  const auto width = static_cast<int>(band_tile_);
  const auto tiles = static_cast<int>(band_tile_count_);
  const int last = samples - 1;
  const std::size_t rows = node_count * Components;

  // 1. Per channel: the exposure, then the fraction, the guard and the integrand.
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto local = KOKKOS_LAMBDA(const TeamMember& team) {
      capture_anchor(field, rate, rates, exposures, totals, taps_view, synthesis, taper,
                     stage, channel, samples, half, width, tiles, dt);
      const auto node = static_cast<std::size_t>(team.league_rank() / tiles);
      const int index = team.league_rank() % tiles;
      const int first = index * width;
      const int end = Kokkos::min(first + width, samples);
      double taps[plasma_max_taps] = {};
      for (int k = 0; k <= half; ++k)
        taps[k] = taps_view(static_cast<std::size_t>(k));
      const double scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
      const std::size_t base = node * Components;
      const double total = team_band_tile(
          team,
          [&](const int sample) {
            const auto time = static_cast<std::size_t>(sample);
            const double value = rate(field.magnitude(base, time, scale), channel);
            if (sample >= first && sample < end)
              rates(node, time) = value;
            return value;
          },
          first, end, samples, taps, half, dt,
          [&](const int sample, const double prefix) {
            exposures(node, static_cast<std::size_t>(sample)) = prefix;
          });
      Kokkos::single(Kokkos::PerTeam(team),
                     [&] { totals(node, static_cast<std::size_t>(index)) = total; });
    };
    Kokkos::parallel_for("solprop_native.response.tiled_band_exposure",
                         tiled_band_policy(node_count, true, local), local);
    const auto finish = KOKKOS_LAMBDA(const TeamMember& team) {
      capture_anchor(field, rates, exposures, guards, weights, integrands, totals,
                     stash, stashing, synthesis, taper, channel_parameters,
                     channel_density, channel_preionisation, preionisation,
                     charge_mass_ratio, ionisation_potential, channels, channel,
                     samples, width, tiles, last, first_node, stage);
      const auto node = static_cast<std::size_t>(team.league_rank() / tiles);
      const int index = team.league_rank() % tiles;
      const int first = index * width;
      const int end = Kokkos::min(first + width, samples);
      const double offset = team_tile_offset(team, totals, node, index);
      const double scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
      const std::size_t base = node * Components;
      const double initial =
          Multiple ? channel_preionisation(stage, channel) : preionisation(stage);
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, first, end), [&](const int sample) {
            const auto time = static_cast<std::size_t>(sample);
            const double fraction =
                band_fraction(offset + exposures(node, time), initial);
            const double value = rates(node, time);
            if constexpr (Multiple) {
              const double partial = channel_density(stage, channel);
              const double charge = channel_parameters(channel, 7);
              const double potential = channel_parameters(channel, 8);
              const double share = partial * fraction * charge;
              const double paid = partial * potential * value * (1.0 - fraction);
              const double weight = (channel == 0 ? 0.0 : weights(node, time)) + share;
              const double work = (channel == 0 ? 0.0 : guards(node, time)) + paid;
              if (channel + 1 == channels) {
                const double magnitude = field.magnitude(base, time, scale);
                guards(node, time) =
                    band_loss_scale(1.0, 0.0, magnitude * magnitude, work);
                for (std::size_t component = 0; component < Components; ++component)
                  integrands(base + component, time) =
                      weight * field.at(base + component, time, scale);
              } else {
                weights(node, time) = weight;
                guards(node, time) = work;
              }
            } else {
              const double magnitude = field.magnitude(base, time, scale);
              guards(node, time) = band_loss_scale(
                  value, fraction, magnitude * magnitude, ionisation_potential);
              const double weight = fraction * charge_mass_ratio;
              for (std::size_t component = 0; component < Components; ++component)
                integrands(base + component, time) =
                    weight * field.at(base + component, time, scale);
            }
            if (stashing && sample == last)
              stash(first_node + node, 2 * channel) = fraction;
          });
    };
    Kokkos::parallel_for("solprop_native.response.tiled_band_fraction",
                         tiled_band_policy(node_count, false, finish), finish);
    if (stashing) {
      const auto peak = KOKKOS_LAMBDA(const TeamMember& team) {
        capture_anchor(rates, stash, channel, samples, first_node);
        const auto node = static_cast<std::size_t>(team.league_rank());
        double largest = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, samples),
            [&](const int sample, double& update) {
              const double value = rates(node, static_cast<std::size_t>(sample));
              if (value > update)
                update = value;
            },
            Kokkos::Max<double>(largest));
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          stash(first_node + node, 2 * channel + 1) = largest > 0.0 ? largest : 0.0;
        });
      };
      Kokkos::parallel_for("solprop_native.response.tiled_band_peak_rate",
                           Kokkos::TeamPolicy<ExecutionSpace>(
                               response_space_for(node_count),
                               static_cast<int>(node_count), Kokkos::AUTO()),
                           peak);
    }
  }

  // 2. Per component row: the free current, followed by the pointwise loss current.
  tiled_band_local(integrands, rows, "solprop_native.response.tiled_band_free_current");
  const auto current = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(field, guards, locals, out, density, totals, synthesis, taper, stage,
                   samples, width, tiles);
    const auto row = static_cast<std::size_t>(team.league_rank() / tiles);
    const int index = team.league_rank() % tiles;
    const int first = index * width;
    const int end = Kokkos::min(first + width, samples);
    const double offset = team_tile_offset(team, totals, row, index);
    const double scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
    const std::size_t node = row / Components;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, first, end), [&](const int sample) {
          const auto time = static_cast<std::size_t>(sample);
          const double prefix = offset + locals(row, time);
          const double drive = guards(node, time) * field.at(row, time, scale);
          out(row, time) =
              Multiple ? prefix + drive : density(stage) * (prefix + drive);
        });
  };
  Kokkos::parallel_for("solprop_native.response.tiled_band_current",
                       tiled_band_policy(rows, false, current), current);
}

// The tiled column's local kernel over component rows: `source`'s band-filtered tile
// prefixes from zero into the free-current rows (`plasma_outer_current_`),
// and each tile's total into `band_tile_totals_`.
void GeneralWorkspace::tiled_band_local(const RealView2& source, const std::size_t rows,
                                        const char* label) {
  using TeamMember = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  const auto locals = plasma_outer_current_;
  const auto totals = band_tile_totals_;
  const auto taps_view = plasma_taps_;
  const double dt = problem_.dt_fine;
  const auto samples = static_cast<int>(problem_.n_to);
  const auto half = static_cast<int>(problem_.plasma_half_width());
  const auto width = static_cast<int>(band_tile_);
  const auto tiles = static_cast<int>(band_tile_count_);
  const auto local = KOKKOS_LAMBDA(const TeamMember& team) {
    const auto row = static_cast<std::size_t>(team.league_rank() / tiles);
    const int index = team.league_rank() % tiles;
    const int first = index * width;
    const int end = Kokkos::min(first + width, samples);
    double taps[plasma_max_taps] = {};
    for (int k = 0; k <= half; ++k)
      taps[k] = taps_view(static_cast<std::size_t>(k));
    const double total = team_band_tile(
        team,
        [&](const int sample) { return source(row, static_cast<std::size_t>(sample)); },
        first, end, samples, taps, half, dt,
        [&](const int sample, const double prefix) {
          locals(row, static_cast<std::size_t>(sample)) = prefix;
        });
    Kokkos::single(Kokkos::PerTeam(team),
                   [&] { totals(row, static_cast<std::size_t>(index)) = total; });
  };
  Kokkos::parallel_for(label, tiled_band_policy(rows, true, local), local);
}

// The band rule's statistics scan: per channel the rates, their band filter and
// trapezoid scan, then team reductions over the rows. The floored final exposure and
// the largest rate always; with `Diagnose` the six parts of SEP 18's run diagnostics,
// the largest over the channels: the exposure's largest dip and rise (maxima of the
// scan's increments), the largest |paid - freed| and freed (the paid work a sum of the
// trapezoid's increments of the raw rate's owed work), and the largest |f E| in the
// first 2K samples and anywhere. Maxima combine exactly; the sums associate as the
// team's scan does. The rows are the band column's (`plasma_rate_`,
// `plasma_fraction_` filtered and then owed work, `plasma_exposure_` exposure and then
// fraction), by slot as the column walks the nodes. Not a step's hot path: it runs
// whole rows, without the column's live windows.
template <bool Diagnose>
void GeneralWorkspace::team_band_statistics(const std::size_t first_node,
                                            const std::size_t node_count,
                                            const RealView2& output,
                                            const double preionisation) {
  using TeamMember = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const bool real_family = problem_.family == FieldFamily::field_real;
  const auto rates = plasma_rate_;
  const auto filtered = plasma_fraction_;
  const auto exposures = plasma_exposure_;
  const auto taps_view = plasma_taps_;
  const auto diagnostics = plasma_diagnostics_;
  const auto coefficients = rate_coefficients_;
  const auto parameters = channel_parameters_;
  const auto channel_coefficients = channel_coefficients_;
  const auto channel_preionisation = stage_channel_preionisation_;
  const auto channels = problem_.n_channels;
  const bool multiple = channels > 1;
  const auto n_pol = problem_.effective_node_pol();
  const auto samples = static_cast<int>(problem_.n_to);
  const auto intervals = problem_.rate_intervals;
  const auto dt = problem_.dt_fine;
  const auto log_field_min = problem_.log_field_min;
  const auto d_log_field = problem_.d_log_field;
  const auto field_min = problem_.field_min;
  const auto field_max = problem_.field_max;
  const auto rate_max = problem_.rate_max;
  const auto half = static_cast<int>(problem_.plasma_half_width());
  const int reach = Kokkos::min(2 * half, samples);
  const std::size_t last = problem_.n_to - 1;
  const std::size_t league = band_league(node_count);
  const auto scan = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(node_real, node_complex, real_family, rates, filtered, exposures,
                   taps_view, diagnostics, coefficients, parameters,
                   channel_coefficients, channel_preionisation, channels, multiple,
                   n_pol, samples, intervals, dt, log_field_min, d_log_field, field_min,
                   field_max, rate_max, half, reach, last, first_node, output,
                   preionisation, league);
    const auto slot = static_cast<std::size_t>(team.league_rank());
    double taps[plasma_max_taps] = {};
    for (int k = 0; k <= half; ++k)
      taps[k] = taps_view(static_cast<std::size_t>(k));
    const auto tile = band_tile<band_staging>(team);
    for (std::size_t node = slot; node < node_count; node += league) {
      const auto base = node * n_pol;
      const auto field_at = [&](const std::size_t row, const std::size_t time) {
        return real_family ? node_real(row, time) : node_complex(row, time).real();
      };
      double parts[plasma_diagnostic_parts] = {};
      for (std::size_t channel = 0; channel < channels; ++channel) {
        const double initial =
            multiple ? channel_preionisation(0, channel) : preionisation;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, samples), [&](const int sample) {
              const auto time = static_cast<std::size_t>(sample);
              const double first = field_at(base, time);
              const double magnitude =
                  n_pol == 2 ? Kokkos::hypot(first, field_at(base + 1, time))
                             : Kokkos::abs(first);
              rates(slot, time) =
                  multiple
                      ? ionisation_rate(
                            magnitude, channel_coefficients, parameters(channel, 0),
                            parameters(channel, 1), parameters(channel, 2),
                            parameters(channel, 3), parameters(channel, 4),
                            static_cast<std::size_t>(parameters(channel, 5)),
                            static_cast<std::size_t>(parameters(channel, 6)))
                      : ionisation_rate(magnitude, coefficients, log_field_min,
                                        d_log_field, field_min, field_max, rate_max,
                                        intervals);
            });
        team.team_barrier();
        team_band_filter<band_staging>(team, tile, rates, slot, filtered, slot, samples,
                                       taps, half);
        team_trapezoid_prefix(team, filtered, slot, samples, dt,
                              [&](const int sample, const double prefix) {
                                exposures(slot, static_cast<std::size_t>(sample)) =
                                    prefix;
                              });
        double largest = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, samples),
            [&](const int sample, double& update) {
              const double rate = rates(slot, static_cast<std::size_t>(sample));
              if (rate > update)
                update = rate;
            },
            Kokkos::Max<double>(largest));
        const double exposure = exposures(slot, last);
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          output(first_node + node, 2 * channel) = exposure > 0.0 ? exposure : 0.0;
          output(first_node + node, 2 * channel + 1) = largest > 0.0 ? largest : 0.0;
        });
        if constexpr (Diagnose) {
          // The exposure's increments, as the scan formed them.
          const auto increment = [&](const int sample) {
            const auto time = static_cast<std::size_t>(sample);
            const double current = filtered(slot, time);
            const double previous = filtered(slot, time - 1);
            return dt * (current + previous) / 2.0;
          };
          double dip = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, 1, samples),
              [&](const int sample, double& update) {
                const double fall = -increment(sample);
                if (fall > update)
                  update = fall;
              },
              Kokkos::Max<double>(dip));
          double rise = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, 1, samples),
              [&](const int sample, double& update) {
                const double step = increment(sample);
                if (step > update)
                  update = step;
              },
              Kokkos::Max<double>(rise));
          team.team_barrier();
          // The filtered row becomes the owed work, the exposure row the fraction.
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, samples), [&](const int sample) {
                const auto time = static_cast<std::size_t>(sample);
                const double fraction = band_fraction(exposures(slot, time), initial);
                filtered(slot, time) = rates(slot, time) * (1.0 - fraction);
                exposures(slot, time) = fraction;
              });
          team.team_barrier();
          double paid = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, 1, samples),
              [&](const int sample, double& sum) { sum += increment(sample); }, paid);
          const auto level = [&](const int sample, double& update) {
            const auto time = static_cast<std::size_t>(sample);
            const double fraction = exposures(slot, time);
            for (std::size_t component = 0; component < n_pol; ++component) {
              const double value =
                  Kokkos::abs(fraction * field_at(base + component, time));
              if (value > update)
                update = value;
            }
          };
          double early = 0.0;
          Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, reach), level,
                                  Kokkos::Max<double>(early));
          double peak = 0.0;
          Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, samples), level,
                                  Kokkos::Max<double>(peak));
          const double freed = exposures(slot, last) - exposures(slot, 0);
          const double values[plasma_diagnostic_parts] = {
              dip > 0.0 ? dip : 0.0,     rise > 0.0 ? rise : 0.0,
              Kokkos::abs(paid - freed), freed,
              early > 0.0 ? early : 0.0, peak > 0.0 ? peak : 0.0};
          for (std::size_t part = 0; part < plasma_diagnostic_parts; ++part)
            parts[part] = Kokkos::max(parts[part], values[part]);
        }
        team.team_barrier();
      }
      if constexpr (Diagnose)
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          for (std::size_t part = 0; part < plasma_diagnostic_parts; ++part)
            diagnostics(first_node + node, part) = parts[part];
        });
      team.team_barrier();
    }
  };
  Kokkos::parallel_for("solprop_native.plasma.team_band_statistics",
                       band_team_policy(node_count, scan), scan);
}

void GeneralWorkspace::convolve_raman(const std::size_t stage,
                                      const std::size_t node_count) {
  const bool tail = node_count != problem_.node_chunk;
  Kokkos::Profiling::pushRegion("solprop_native.raman.drive_fft");
  execute_raman(tail ? *raman_tail_forward_plan_ : *raman_drive_forward_plan_,
                tail ? raman_tail_drive_ : node_raman_drive_,
                tail ? raman_tail_spectrum_ : node_raman_spectrum_,
                KokkosFFT::Normalization::none);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("solprop_native.raman.spectral_multiply");
  const auto node_spectrum = node_raman_spectrum_;
  const auto kernel_spectrum = raman_kernel_spectrum_;
  const auto drives = problem_.raman_drives();
  const double inverse_doubled = 1.0 / static_cast<double>(2 * problem_.n_to);
  Kokkos::parallel_for(
      "solprop_native.raman.spectral_multiply",
      Range2(tail ? raman_tail_multiply_space_ : raman_multiply_space_, {0, 0},
             {node_count * drives, problem_.n_to + 1}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t index) {
        node_spectrum(node, index) =
            (node_spectrum(node, index) *
             kernel_spectrum(node % drives == 0 ? 0 : 1, index)) *
            inverse_doubled;
      });
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("solprop_native.raman.inverse_fft");
  execute_raman(tail ? *raman_tail_inverse_plan_ : *raman_convolution_inverse_plan_,
                tail ? raman_tail_spectrum_ : node_raman_spectrum_,
                tail ? raman_tail_convolved_ : node_raman_convolved_,
                KokkosFFT::Normalization::none);
  Kokkos::Profiling::popRegion();
  finish_raman(stage, node_count);
}

void GeneralWorkspace::finish_raman(const std::size_t stage,
                                    const std::size_t node_count) {
  Kokkos::Profiling::pushRegion("solprop_native.raman.finish");
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto out_real = polarisation_real_;
  const auto out_complex = polarisation_complex_;
  const auto convolved = node_raman_convolved_;
  const auto window = time_window_;
  const auto density = stage_raman_density_;
  const auto family = problem_.family;
  const auto components = problem_.effective_node_pol();
  const auto drives = problem_.raman_drives();
  const double epsilon0 = problem_.raman_epsilon0;
  Kokkos::parallel_for(
      "solprop_native.raman.finish",
      Range2(response_space_for(node_count), {0, 0}, {node_count, problem_.n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        if (components == 2) {
          const auto base = 2 * node;
          const auto row = drives * node;
          const double u = convolved(row, time);
          const double v = drives == 3 ? convolved(row + 1, time) : 0.0;
          const double w = drives == 3 ? convolved(row + 2, time) : 0.0;
          const double factor = epsilon0 * density(stage);
          if (family == FieldFamily::envelope) {
            const Complex x = node_complex(base, time),
                          y = node_complex(base + 1, time);
            out_complex(base, time) =
                (out_complex(base, time) + factor * ((u + v) * x + w * y)) *
                window(time);
            out_complex(base + 1, time) =
                (out_complex(base + 1, time) + factor * (w * x + (u - v) * y)) *
                window(time);
          } else {
            const double x = family == FieldFamily::field_real
                                 ? node_real(base, time)
                                 : node_complex(base, time).real();
            const double y = family == FieldFamily::field_real
                                 ? node_real(base + 1, time)
                                 : node_complex(base + 1, time).real();
            out_real(base, time) =
                (out_real(base, time) + factor * ((u + v) * x + w * y)) * window(time);
            out_real(base + 1, time) =
                (out_real(base + 1, time) + factor * (w * x + (u - v) * y)) *
                window(time);
          }
          return;
        }
        const double factor = epsilon0 * density(stage) * convolved(node, time);
        if (family == FieldFamily::envelope) {
          out_complex(node, time) =
              (out_complex(node, time) + factor * node_complex(node, time)) *
              window(time);
        } else {
          const double field = family == FieldFamily::field_real
                                   ? node_real(node, time)
                                   : node_complex(node, time).real();
          out_real(node, time) = (out_real(node, time) + factor * field) * window(time);
        }
      });
  Kokkos::Profiling::popRegion();
}

void GeneralWorkspace::envelope_cycle(const std::size_t channel,
                                      const std::size_t node_count) {
  const auto field = node_complex_;
  const auto parameters = channel_parameters_;
  const auto coefficients = channel_coefficients_;
  const auto nodes =
      Kokkos::subview(envelope_rule_nodes_, channel, Kokkos::ALL(), Kokkos::ALL());
  const auto weights =
      Kokkos::subview(envelope_rule_weights_, channel, Kokkos::ALL(), Kokkos::ALL());
  const auto moments = envelope_moments_;
  const auto rates = envelope_rates_;
  const auto failures = envelope_failures_;
  const auto n_pol = problem_.effective_node_pol();
  const auto n_to = problem_.n_to;
  Kokkos::parallel_for(
      "solprop_native.envelope.cycle_moments",
      Range2(response_space_for(node_count), {0, 0}, {node_count, n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        const auto axes =
            envelope_axes(field(node * n_pol, time),
                          n_pol == 2 ? field(node * n_pol + 1, time) : Complex{});
        const auto rate_at = [&](const double value) {
          return ionisation_rate(value, coefficients, parameters(channel, 0),
                                 parameters(channel, 1), parameters(channel, 2),
                                 parameters(channel, 3), parameters(channel, 4),
                                 static_cast<std::size_t>(parameters(channel, 5)),
                                 static_cast<std::size_t>(parameters(channel, 6)));
        };
        const auto pair =
            envelope_moments(axes.major, axes.ratio, parameters(channel, 2),
                             parameters(channel, 3), rate_at, nodes, weights);
        moments(0, node, time) = pair.cosine;
        moments(1, node, time) = pair.sine;
        rates(node, time) = pair.cosine + axes.ratio * axes.ratio * pair.sine;
        failures(node * n_to + time) = pair.failed ? 1 : 0;
      });
  const auto status = envelope_status_;
  std::size_t no_failure = 0;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // A serial device scan dominated the short combined Raman/plasma profiles.
  // Integer minimum preserves the first failing sample exactly;
  // its identity denotes success without an additional normalization launch.
  no_failure = std::numeric_limits<std::size_t>::max();
  using StatusPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using StatusMember = StatusPolicy::member_type;
  const auto samples = node_count * n_to;
  // One team uses only launch-time shared memory; a global reduction could grow
  // the execution-space scratch allocator inside the first trial.
  Kokkos::parallel_for(
      "solprop_native.envelope.angular_status",
      StatusPolicy(response_space_for(node_count), 1,
                   SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE),
      KOKKOS_LAMBDA(const StatusMember& team) {
        std::size_t first_failure = Kokkos::reduction_identity<std::size_t>::min();
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, samples),
            [&](const std::size_t sample, std::size_t& minimum) {
              if (failures(sample) != 0)
                minimum = Kokkos::min(minimum, sample + 1);
            },
            Kokkos::Min<std::size_t>(first_failure));
        Kokkos::single(Kokkos::PerTeam(team), [&]() { status(0) = first_failure; });
      });
#else
  Kokkos::parallel_for(
      "solprop_native.envelope.angular_status",
      Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0, 1),
      KOKKOS_LAMBDA(const std::size_t) {
        status(0) = 0;
        for (std::size_t sample = 0; sample < node_count * n_to; ++sample)
          if (failures(sample) != 0) {
            status(0) = sample + 1;
            break;
          }
      });
#endif
  // This planned scalar boundary rejects failed refinement before a current FFT.
  Kokkos::deep_copy(response_space_for(node_count), envelope_host_status_,
                    envelope_status_);
  response_space_for(node_count).fence("solprop_native envelope refinement boundary");
  if (envelope_host_status_(0) != no_failure)
    throw EnvelopeConvergenceError("envelope plasma channel " +
                                   std::to_string(channel) +
                                   " exceeded the angular refinement budget");
  const auto rows = std::make_pair(std::size_t{0}, node_count);
  portable_team_trapezoid_scan<true>(
      response_space_for(node_count),
      Kokkos::subview(envelope_rates_, rows, Kokkos::ALL()),
      Kokkos::subview(envelope_exposure_, rows, Kokkos::ALL()), problem_.dt_fine);
}

void GeneralWorkspace::envelope_population(const std::size_t channel,
                                           const std::size_t stage,
                                           const std::size_t first_node,
                                           const std::size_t node_count,
                                           const RealView2& statistics) {
  envelope_cycle(channel, node_count);
  const auto rates = envelope_rates_;
  const auto exposure = envelope_exposure_;
  const auto n_to = problem_.n_to;
  if (!statistics.empty()) {
    Kokkos::parallel_for(
        "solprop_native.envelope.population_statistics",
        Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0,
                                            node_count),
        KOKKOS_LAMBDA(const std::size_t node) {
          double peak = 0.0;
          for (std::size_t time = 0; time < n_to; ++time)
            peak = Kokkos::max(peak, rates(node, time));
          statistics(first_node + node, 2 * channel) = exposure(node, n_to - 1);
          statistics(first_node + node, 2 * channel + 1) = peak;
        });
    return;
  }
  const auto parameters = channel_parameters_;
  const auto densities = stage_channel_density_;
  const auto initials = stage_channel_preionisation_;
  const auto moments = envelope_moments_;
  const auto field = node_complex_;
  const auto window = time_window_;
  const auto drive = envelope_streams_[0].node;
  const auto loss = envelope_streams_[1].node;
  const auto n_pol = problem_.effective_node_pol();
  Kokkos::parallel_for(
      "solprop_native.envelope.currents",
      Range2(response_space_for(node_count), {0, 0}, {node_count, n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        const double initial = initials(stage, channel);
        const double fraction = envelope_fraction(exposure(node, time), initial);
        const double survival = (1.0 - initial) * Kokkos::exp(-exposure(node, time));
        const double density = densities(stage, channel);
        const auto axes =
            envelope_axes(field(node * n_pol, time),
                          n_pol == 2 ? field(node * n_pol + 1, time) : Complex{});
        const auto current =
            envelope_loss(axes, moments(0, node, time), moments(1, node, time),
                          parameters(channel, 8) * density * survival);
        const double drude = parameters(channel, 7) * density * fraction;
        for (std::size_t pol = 0; pol < n_pol; ++pol) {
          const auto row = node * n_pol + pol;
          const auto drive_value = window(time) * drude * field(row, time);
          const auto loss_value = window(time) * current[pol];
          if (channel == 0) {
            drive(row, time) = drive_value;
            loss(row, time) = loss_value;
          } else {
            drive(row, time) += drive_value;
            loss(row, time) += loss_value;
          }
        }
      });
}

void GeneralWorkspace::envelope_response(const std::size_t stage,
                                         const std::size_t first_node,
                                         const std::size_t node_count,
                                         const TransverseChunk* chunk) {
  for (std::size_t channel = 0; channel < problem_.n_channels; ++channel)
    envelope_population(channel, stage, first_node, node_count, {});
  for (auto& stream : envelope_streams_) {
    if (problem_.slab_geometry()) {
      // As the polarisation: the chunk's y rows, zero beyond a final partial chunk.
      Kokkos::deep_copy(response_space_for(node_count), stream.projected, Complex{});
      slab_project(*chunk, &stream);
    } else if (problem_.free_geometry()) {
      // The persistent maximum-chunk plan transforms zero padding in a final
      // partial chunk; only its actual rows are ever copied to the spatial FFT.
      Kokkos::deep_copy(response_space_for(node_count), stream.projected, Complex{});
      const auto rows =
          std::make_pair(std::size_t{0}, node_count * problem_.effective_node_pol());
      Kokkos::deep_copy(response_space_for(node_count),
                        Kokkos::subview(stream.projected, rows, Kokkos::ALL()),
                        Kokkos::subview(stream.node, rows, Kokkos::ALL()));
    } else {
      project(first_node, node_count, first_node == 0, &stream);
    }
  }
}

// SEP 27 forms two independently collared current products. Every maximum-chunk
// row is written, including a partial chunk's padding, before a persistent FFT plan
// can read it. Modal projection precedes the FFT; radial projection follows it.
void GeneralWorkspace::held_response(const std::size_t stage,
                                     const std::size_t first_node,
                                     const std::size_t node_count,
                                     const TransverseChunk* chunk) {
  const auto current = held_current_;
  const auto window = time_window_;
  const auto derivative = held_derivative_;
  const auto product = held_streams_[0].node;
  const auto edge = held_streams_[1].node;
  const auto live = node_count * problem_.effective_node_pol();
  const auto n_to = problem_.n_to;
  cartesian_for(
      "solprop_native.held.products", response_space_for(node_count), current.extent(0),
      n_to, KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
        const double value = row < live ? current(row, time) : 0.0;
        product(row, time) = window(time) * value;
        edge(row, time) = derivative(time) * value;
      });
  const bool folded = folds_single_node();
  const bool radial = held_radial();
  for (std::size_t index = 0; index < 2; ++index) {
    auto& stream = held_streams_[index];
    if (folded) {
      const auto source = stream.node;
      const auto target = stream.projected;
      const auto projection = projection_;
      const auto taper = stage_taper_;
      cartesian_for(
          "solprop_native.held.folded_projection", response_space_for(node_count), 1,
          n_to, KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
            target(row, time) = projection(0, 0) / taper(stage) * source(row, time);
          });
    } else if (problem_.slab_geometry()) {
      Kokkos::deep_copy(response_space_for(node_count), stream.projected, 0.0);
      slab_contract("solprop_native.held.slab_projection",
                    response_space_for(node_count), projection_, stream.node,
                    stream.projected, chunk->nodes / problem_.n_x, n_to);
    } else if (problem_.free_geometry() || radial) {
      Kokkos::deep_copy(response_space_for(node_count), stream.projected, stream.node);
    } else {
      project(first_node, node_count, first_node == 0, nullptr, &stream);
    }
  }
  if (radial) {
    const auto source = polarisation_real_;
    const auto target = held_streams_[2].projected;
    cartesian_for(
        "solprop_native.held.nonplasma", response_space_for(node_count),
        target.extent(0), n_to,
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
          target(row, time) = row < live ? source(row, time) : 0.0;
        });
    held_radial_projection(first_node, node_count);
  }
}

void GeneralWorkspace::held_transform(const bool coarse_only) {
  for (std::size_t index = 0; index < 2; ++index) {
    auto& stream = held_streams_[index];
    if (!coarse_only)
      KokkosFFT::execute(*stream.forward, stream.projected, stream.spectrum,
                         KokkosFFT::Normalization::none);
    if (stream.coarse_forward && (coarse_only || !defer_quadrature_))
      KokkosFFT::execute(*stream.coarse_forward, stream.coarse, stream.coarse_spectrum,
                         KokkosFFT::Normalization::none);
  }
  const auto product = held_streams_[0].spectrum;
  const auto edge = held_streams_[1].spectrum;
  const auto coarse_product = held_streams_[0].coarse_spectrum;
  const auto coarse_edge = held_streams_[1].coarse_spectrum;
  const auto projected = projected_spectrum_;
  const auto coarse = projected_coarse_spectrum_;
  const auto inverse = held_inverse_;
  const auto active = active_mask_;
  const auto crop = crop_index_;
  const bool include_coarse = !coarse.empty() && (coarse_only || !defer_quadrature_);
  cartesian_for(
      "solprop_native.held.spectral_assembly", execution_space_, projected.extent(0),
      problem_.n_omega, KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        const auto bin = crop(omega);
        if (active(omega)) {
          const double value = inverse(omega);
          const Complex factor(0.0, -value);
          if (!coarse_only)
            projected(row, bin) +=
                factor * product(row, bin) + value * value * edge(row, bin);
          if (include_coarse)
            coarse(row, bin) += factor * coarse_product(row, bin) +
                                value * value * coarse_edge(row, bin);
        } else {
          if (!coarse_only)
            projected(row, bin) = Complex{};
          if (include_coarse)
            coarse(row, bin) = Complex{};
        }
      });
}

void GeneralWorkspace::held_radial_projection(const std::size_t first_node,
                                              const std::size_t node_count) {
  for (auto& stream : held_streams_)
    KokkosFFT::execute(*stream.forward, stream.projected, stream.spectrum,
                       KokkosFFT::Normalization::none);
  const auto product = held_streams_[0].spectrum;
  const auto edge = held_streams_[1].spectrum;
  const auto polarisation = held_streams_[2].spectrum;
  const auto nodal = held_nodal_spectrum_;
  const auto inverse = held_inverse_;
  const auto active = active_mask_;
  const auto crop = crop_index_;
  const auto n = problem_.n_omega;
  const auto live = node_count * problem_.effective_node_pol();
  cartesian_for(
      "solprop_native.held.radial_assembly", node_spectral_space_, nodal.extent(0), n,
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        Complex value{};
        if (row < live && active(omega)) {
          const auto bin = crop(omega);
          const double inv = inverse(omega);
          value = polarisation(row, bin) + Complex(0.0, -inv) * product(row, bin) +
                  inv * inv * edge(row, bin);
        }
        nodal(row, omega) = value;
      });
  const auto packed = held_projection_;
  const auto projection = scaled_projection_;
  const auto first_row = first_node * problem_.n_pol_transform;
  const auto live_rows = node_count * problem_.n_pol_transform;
  cartesian_for(
      "solprop_native.held.radial_projection_matrix", matrix_copy_space_,
      packed.extent(0), packed.extent(1),
      KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
        packed(mode, row) = row < live_rows ? projection(mode, first_row + row) : 0.0;
      });
  // Interleaved real and imaginary parts share the scalar radial projection. This
  // is one GEMM of the combined cropped source, independent of state components.
  using Interleaved = Kokkos::View<double**, Layout, MemorySpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  static_assert(sizeof(Complex) == 2 * sizeof(double));
  const auto columns = 2 * problem_.n_pol_state * n;
  const Interleaved input(reinterpret_cast<double*>(nodal.data()), packed.extent(1),
                          columns);
  const Interleaved output(reinterpret_cast<double*>(held_projected_spectrum_.data()),
                           problem_.n_modes, columns);
  selected_gemm(transform_space_for(node_count), packed, input, output, 0.0, true);
  if (first_node == 0)
    Kokkos::deep_copy(temporal_execution_space_, projected_spectrum_, Complex{});
  const auto source = held_projected_spectrum_;
  const auto target = projected_spectrum_;
  cartesian_for(
      "solprop_native.held.radial_accumulate", execution_space_, target.extent(0), n,
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        if (active(omega))
          target(row, crop(omega)) += source(row, omega);
      });
}

void GeneralWorkspace::envelope_transform() {
  for (auto& stream : envelope_streams_) {
    KokkosFFT::execute(*stream.forward, stream.projected, stream.spectrum,
                       KokkosFFT::Normalization::none);
    if (stream.coarse_forward)
      KokkosFFT::execute(*stream.coarse_forward, stream.coarse, stream.coarse_spectrum,
                         KokkosFFT::Normalization::none);
  }
  const auto drive = envelope_streams_[0].spectrum;
  const auto loss = envelope_streams_[1].spectrum;
  const auto coarse_drive = envelope_streams_[0].coarse_spectrum;
  const auto coarse_loss = envelope_streams_[1].coarse_spectrum;
  const auto projected = projected_spectrum_;
  const auto coarse = projected_coarse_spectrum_;
  const auto drude_factor = envelope_drude_factor_;
  const auto loss_factor = envelope_loss_factor_;
  const auto crop = crop_index_;
  Kokkos::parallel_for(
      "solprop_native.envelope.spectral_assembly",
      Range2(execution_space_, {0, 0}, {projected.extent(0), problem_.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        const auto index = crop(omega);
        projected(column, index) += drude_factor(omega) * drive(column, index) +
                                    loss_factor(omega) * loss(column, index);
        if (!coarse.empty())
          coarse(column, index) += drude_factor(omega) * coarse_drive(column, index) +
                                   loss_factor(omega) * coarse_loss(column, index);
      });
}

void GeneralWorkspace::project(const std::size_t first_node,
                               const std::size_t node_count, const bool first_chunk,
                               const EnvelopeStream* stream,
                               const FieldCurrentStream* field_stream) {
  const auto real_source =
      field_stream == nullptr ? polarisation_real_ : field_stream->node;
  const auto real_target =
      field_stream == nullptr ? projected_real_ : field_stream->projected;
  const auto real_coarse_target =
      field_stream == nullptr ? projected_coarse_real_ : field_stream->coarse;
  const auto real_stacked_target =
      field_stream == nullptr ? projected_stacked_real_ : field_stream->stacked;
  const auto node_source = stream == nullptr ? polarisation_complex_ : stream->node;
  const auto projected_target =
      stream == nullptr ? projected_complex_ : stream->projected;
  const auto coarse_target =
      stream == nullptr ? projected_coarse_complex_ : stream->coarse;
  const auto stacked_target =
      stream == nullptr ? projected_stacked_complex_ : stream->stacked;
  const bool whole_transform =
      first_node == 0 && node_count == problem_.n_nodes && problem_.n_pol_state == 1;
  if (whole_transform && !radial_blas_input_.empty()) {
    packed_radial_gemm(scaled_projection_, node_source, projected_target);
    return;
  }
  if (whole_transform && use_interleaved_projection()) {
    Kokkos::Profiling::pushRegion(
        "solprop_native.transform.projection.kokkosblas_interleaved");
    using Interleaved = Kokkos::View<double**, Layout, MemorySpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    static_assert(sizeof(Complex) == 2 * sizeof(double));
    const Interleaved nodes(reinterpret_cast<double*>(node_source.data()),
                            problem_.n_nodes, 2 * problem_.n_to);
    const Interleaved fine(reinterpret_cast<double*>(projected_target.data()),
                           problem_.n_modes, 2 * problem_.n_to);
    engine_gemm(transform_space_for(node_count), scaled_projection_, nodes, fine, 0.0);
    if (project_coarse_ && !problem_.projection_coarse_matrix.empty()) {
      const Interleaved coarse(reinterpret_cast<double*>(coarse_target.data()),
                               problem_.n_modes, 2 * problem_.n_to);
      engine_gemm(transform_space_for(node_count), scaled_projection_coarse_, nodes,
                  coarse, 0.0);
    }
    Kokkos::Profiling::popRegion();
    return;
  }
  if (whole_transform && use_blas_for_projection()) {
    Kokkos::Profiling::pushRegion("solprop_native.transform.projection.kokkosblas");
    if (problem_.family == FieldFamily::envelope && project_coarse_ &&
        !scaled_projection_stacked_complex_.empty()) {
      selected_gemm(transform_space_for(node_count), scaled_projection_stacked_complex_,
                    node_source, stacked_target, Complex{}, true);
      const auto& stacked = stacked_target;
      const auto& fine = projected_target;
      const auto& coarse = coarse_target;
      const auto columns = problem_.n_columns();
      Kokkos::parallel_for(
          "solprop_native.transform.projection.split_stacked_complex",
          Range2(temporal_execution_space_, {0, 0}, {columns, problem_.n_to}),
          KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
            fine(column, time) = stacked(column, time);
            coarse(column, time) = stacked(columns + column, time);
          });
    } else if (problem_.family == FieldFamily::envelope) {
      selected_gemm(transform_space_for(node_count), scaled_projection_complex_,
                    node_source, projected_target, Complex{}, true);
    } else if (project_coarse_ && !scaled_projection_stacked_.empty()) {
      selected_gemm(transform_space_for(node_count), scaled_projection_stacked_,
                    real_source, real_stacked_target, 0.0, true);
      const auto& stacked = real_stacked_target;
      const auto& fine = real_target;
      const auto& coarse = real_coarse_target;
      const auto columns = problem_.n_columns();
      Kokkos::parallel_for(
          "solprop_native.transform.projection.split_stacked",
          Range2(temporal_execution_space_, {0, 0}, {columns, problem_.n_to}),
          KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
            fine(column, time) = stacked(column, time);
            coarse(column, time) = stacked(columns + column, time);
          });
    } else {
      selected_gemm(transform_space_for(node_count), scaled_projection_, real_source,
                    real_target, 0.0, true);
    }
    Kokkos::Profiling::popRegion();
    return;
  }
  if (use_chunked_blas_for_projection()) {
    Kokkos::Profiling::pushRegion(
        "solprop_native.transform.projection.kokkosblas_chunked");
    const auto rows = node_count * problem_.n_pol_transform;
    const auto first_row = first_node * problem_.n_pol_transform;
    const bool tail = node_count != problem_.node_chunk;
    const bool last_chunk = first_node + node_count == problem_.n_nodes;
    const bool stacked = project_coarse_ && !problem_.projection_coarse_matrix.empty();
    if (problem_.family == FieldFamily::envelope) {
      const auto polarisation = Kokkos::subview(
          node_source, std::make_pair(std::size_t{0}, rows), Kokkos::ALL());
      if (stacked) {
        const auto packed =
            tail ? tail_projection_stacked_complex_ : chunk_projection_stacked_complex_;
        const auto source = scaled_projection_stacked_complex_;
        Kokkos::parallel_for(
            "solprop_native.transform.pack_projection_stacked_complex",
            Range2(matrix_copy_space_for(node_count, true), {0, 0},
                   {2 * problem_.n_modes, rows}),
            KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
              packed(mode, row) = source(mode, first_row + row);
            });
        selected_gemm(transform_space_for(node_count), packed, polarisation,
                      stacked_target, first_chunk ? Complex{} : Complex{1.0, 0.0},
                      true);
        if (last_chunk) {
          const auto& projected = stacked_target;
          const auto& fine = projected_target;
          const auto& coarse = coarse_target;
          const auto columns = problem_.n_columns();
          Kokkos::parallel_for(
              "solprop_native.transform.projection.split_stacked_complex",
              Range2(temporal_execution_space_, {0, 0}, {columns, problem_.n_to}),
              KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
                fine(column, time) = projected(column, time);
                coarse(column, time) = projected(columns + column, time);
              });
        }
      } else {
        // The first half of the existing stacked scratch is a contiguous fine
        // matrix. Alias it when this trial stage does not need the coarse rows.
        const auto scratch =
            tail ? tail_projection_stacked_complex_ : chunk_projection_stacked_complex_;
        const ComplexView2 packed =
            scratch.empty()
                ? (tail ? tail_projection_complex_ : chunk_projection_complex_)
                : ComplexView2(Kokkos::subview(
                      scratch, std::pair{std::size_t{0}, problem_.n_modes},
                      Kokkos::ALL()));
        const auto source = scaled_projection_complex_;
        Kokkos::parallel_for(
            "solprop_native.transform.pack_projection_complex",
            Range2(matrix_copy_space_for(node_count), {0, 0}, {problem_.n_modes, rows}),
            KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
              packed(mode, row) = source(mode, first_row + row);
            });
        selected_gemm(transform_space_for(node_count), packed, polarisation,
                      projected_target, first_chunk ? Complex{} : Complex{1.0, 0.0},
                      true);
      }
    } else {
      const auto polarisation = Kokkos::subview(
          real_source, std::make_pair(std::size_t{0}, rows), Kokkos::ALL());
      if (stacked) {
        const auto packed = tail ? tail_projection_stacked_ : chunk_projection_stacked_;
        const auto source = scaled_projection_stacked_;
        Kokkos::parallel_for(
            "solprop_native.transform.pack_projection_stacked",
            Range2(matrix_copy_space_for(node_count, true), {0, 0},
                   {2 * problem_.n_modes, rows}),
            KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
              packed(mode, row) = source(mode, first_row + row);
            });
        selected_gemm(transform_space_for(node_count), packed, polarisation,
                      real_stacked_target, first_chunk ? 0.0 : 1.0, true);
        if (last_chunk) {
          const auto& projected = real_stacked_target;
          const auto& fine = real_target;
          const auto& coarse = real_coarse_target;
          const auto columns = problem_.n_columns();
          Kokkos::parallel_for(
              "solprop_native.transform.projection.split_stacked",
              Range2(temporal_execution_space_, {0, 0}, {columns, problem_.n_to}),
              KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
                fine(column, time) = projected(column, time);
                coarse(column, time) = projected(columns + column, time);
              });
        }
      } else {
        const auto scratch =
            tail ? tail_projection_stacked_ : chunk_projection_stacked_;
        const RealView2 packed =
            scratch.empty()
                ? (tail ? tail_projection_ : chunk_projection_)
                : RealView2(Kokkos::subview(scratch,
                                            std::pair{std::size_t{0}, problem_.n_modes},
                                            Kokkos::ALL()));
        const auto source = scaled_projection_;
        Kokkos::parallel_for(
            "solprop_native.transform.pack_projection",
            Range2(matrix_copy_space_for(node_count), {0, 0}, {problem_.n_modes, rows}),
            KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
              packed(mode, row) = source(mode, first_row + row);
            });
        selected_gemm(transform_space_for(node_count), packed, polarisation,
                      real_target, first_chunk ? 0.0 : 1.0, true);
      }
    }
    Kokkos::Profiling::popRegion();
    return;
  }
  Kokkos::Profiling::pushRegion("solprop_native.transform.projection.custom");
  const auto projection = scaled_projection_;
  const auto projection_coarse = scaled_projection_coarse_;
  const auto& polar_real = real_source;
  const auto& polar_complex = node_source;
  const auto& projected_real = real_target;
  const auto& projected_complex = projected_target;
  const auto& projected_coarse_real = real_coarse_target;
  const auto& projected_coarse_complex = coarse_target;
  const auto n_pol_state = problem_.n_pol_state;
  const auto n_pol_transform = problem_.n_pol_transform;
  const auto n_to = problem_.n_to;
  const bool envelope = problem_.family == FieldFamily::envelope;
  const bool coarse = project_coarse_ && !projection_coarse.empty();
  Kokkos::parallel_for(
      "solprop_native.transform.projection.custom",
      Range2(transform_space_for(node_count), {0, 0}, {problem_.n_columns(), n_to}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
        const std::size_t mode = column / n_pol_state;
        const std::size_t state_pol = column % n_pol_state;
        if (envelope) {
          Complex fine{};
          Complex rough{};
          for (std::size_t node = 0; node < node_count; ++node) {
            for (std::size_t transform_pol = 0; transform_pol < n_pol_transform;
                 ++transform_pol) {
              const std::size_t global_row =
                  (first_node + node) * n_pol_transform + transform_pol;
              const std::size_t local_row =
                  (node * n_pol_transform + transform_pol) * n_pol_state + state_pol;
              const Complex value = polar_complex(local_row, time);
              fine += projection(mode, global_row) * value;
              if (coarse)
                rough += projection_coarse(mode, global_row) * value;
            }
          }
          if (first_chunk) {
            projected_complex(column, time) = fine;
            if (coarse)
              projected_coarse_complex(column, time) = rough;
          } else {
            projected_complex(column, time) += fine;
            if (coarse)
              projected_coarse_complex(column, time) += rough;
          }
        } else {
          double fine = 0.0;
          double rough = 0.0;
          for (std::size_t node = 0; node < node_count; ++node) {
            for (std::size_t transform_pol = 0; transform_pol < n_pol_transform;
                 ++transform_pol) {
              const std::size_t global_row =
                  (first_node + node) * n_pol_transform + transform_pol;
              const std::size_t local_row =
                  (node * n_pol_transform + transform_pol) * n_pol_state + state_pol;
              const double value = polar_real(local_row, time);
              fine += projection(mode, global_row) * value;
              if (coarse)
                rough += projection_coarse(mode, global_row) * value;
            }
          }
          if (first_chunk) {
            projected_real(column, time) = fine;
            if (coarse)
              projected_coarse_real(column, time) = rough;
          } else {
            projected_real(column, time) += fine;
            if (coarse)
              projected_coarse_real(column, time) += rough;
          }
        }
      });
  Kokkos::Profiling::popRegion();
}

void GeneralWorkspace::form_absorber() {
  const bool spatial = !problem_.spatial_absorber_matrix.empty();
  const auto modes = problem_.n_modes;
  const auto polarisations = problem_.n_pol_state;
  // Flatten (polarisation,time) together: the same scalar radial matrix acts
  // independently on each component. The views alias existing FFT work buffers.
  if (spatial && problem_.family == FieldFamily::envelope) {
    using Matrix = Kokkos::View<Complex**, Layout, MemorySpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const Matrix input(modal_complex_.data(), modes, polarisations * problem_.n_to);
    const Matrix output(absorber_complex_.data(), modes, polarisations * problem_.n_to);
    selected_gemm(absorber_execution_space_, spatial_absorber_complex_, input, output,
                  Complex{}, true);
  } else if (spatial && problem_.family == FieldFamily::field_real) {
    using Matrix = Kokkos::View<double**, Layout, MemorySpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const Matrix input(modal_real_.data(), modes, polarisations * problem_.n_to);
    const Matrix output(absorber_real_.data(), modes, polarisations * problem_.n_to);
    selected_gemm(absorber_execution_space_, spatial_absorber_real_, input, output, 0.0,
                  true);
  }
  const auto rate = absorber_fine_;
  const auto spatial_matrix = spatial_absorber_real_;
  const auto modal_real = modal_real_;
  const auto modal_complex = modal_complex_;
  const auto absorber_real = absorber_real_;
  const auto absorber_complex = absorber_complex_;
  const auto family = problem_.family;
  Kokkos::parallel_for(
      "solprop_native.absorber.nonlinear_term",
      Range2(temporal_execution_space_, {0, 0}, {problem_.n_columns(), problem_.n_to}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
        if (family == FieldFamily::envelope)
          absorber_complex(column, time) =
              (spatial ? absorber_complex(column, time) : Complex{}) -
              0.5 * rate(time) * modal_complex(column, time);
        else if (family == FieldFamily::field_real)
          absorber_real(column, time) = (spatial ? absorber_real(column, time) : 0.0) -
                                        0.5 * rate(time) * modal_real(column, time);
        else {
          double transverse = 0.0;
          if (spatial)
            for (std::size_t source = 0; source < modes; ++source)
              transverse +=
                  spatial_matrix(column / polarisations, source) *
                  modal_complex(source * polarisations + column % polarisations, time)
                      .real();
          absorber_real(column, time) =
              transverse - 0.5 * rate(time) * modal_complex(column, time).real();
        }
      });
}

void GeneralWorkspace::finish_stage(const std::size_t stage) {
  const auto terms = stage_terms(stage);
  const double solution_weight = problem_.solution_weights[stage];
  const double error_weight = problem_.error_weights[stage];
  Kokkos::parallel_for(
      "solprop_native.stage.finish_accumulate",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        terms.finish(stage, solution_weight, error_weight, column, index);
      });
}

void GeneralWorkspace::set_flow_actions(const std::vector<Complex>& forward,
                                        const std::vector<Complex>& inverse,
                                        const std::vector<unsigned char>& identity) {
  require_open();
  if (!flow_)
    throw std::logic_error("this state carries no coupled flow");
  flow_->set_actions(forward, inverse, identity);
}

std::vector<Complex> GeneralWorkspace::flow_actions_to_host(const bool inverse) const {
  require_open();
  if (!flow_)
    throw std::logic_error("this state carries no coupled flow");
  return flow_->actions_to_host(inverse);
}

std::vector<double>
GeneralWorkspace::form_flow_actions(const std::size_t section, const double dz,
                                    const std::vector<double>& fractions) {
  require_open();
  if (!flow_)
    throw std::logic_error("this state carries no coupled flow");
  return flow_->form_actions(section, dz, fractions);
}

std::vector<double> GeneralWorkspace::weighted_error(const std::vector<double>& scale) {
  require_open();
  if (!problem_.weighted_controller)
    throw std::logic_error("this state has no weighted controller workspace");
  require_extent(scale.size(), problem_.n_omega, "weighted controller scale");
  copy_vector(execution_space_, weighted_scale_, scale);
  const auto residual = residual_;
  const auto in_norm = norm_mask_;
  const auto native_scale = weighted_scale_;
  const auto partials = weighted_partials_;
  const auto n = problem_.n_omega;
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const int team_size = SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE;
#else
  // A host team per column, sharing the threads when the columns are fewer.
  const int team_size =
      host_reduction_team_size(execution_space_, problem_.n_columns(), n);
#endif
  // On a device each column is `reduction_slots_` teams over its own bins; the
  // caller sums the slots.
  const auto slots = reduction_slots_;
  const TeamPolicy policy(execution_space_,
                          static_cast<int>(problem_.n_columns() * slots), team_size);
  Kokkos::parallel_for(
      "solprop_native.step.weighted_error_partials", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto league = static_cast<std::size_t>(team.league_rank());
        const auto column = league / slots;
        std::size_t first = 0;
        std::size_t end = 0;
        slot_range(n, slots, league % slots, first, end);
        double sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(first),
                                    static_cast<int>(end)),
            [&](const int index_value, double& update) {
              const auto index = static_cast<std::size_t>(index_value);
              if (in_norm(index) == 0U)
                return;
              const Complex value = residual(column, index) / native_scale(index);
              update += squared_abs(value);
            },
            sum);
        Kokkos::single(Kokkos::PerTeam(team), [&] { partials(league) = sum; });
      });
  execution_space_.fence("solprop_native weighted-error boundary");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, partials);
  std::vector<double> result(problem_.n_columns() * slots);
  for (std::size_t slot = 0; slot < result.size(); ++slot)
    result[slot] = host(slot);
  return result;
}

void GeneralWorkspace::fill_statistics_time(const bool windowed, const double lower,
                                            const double upper, const bool weighted) {
  if (omega_.empty())
    throw std::logic_error("native statistics need the coarse frequency axis");
  Kokkos::deep_copy(execution_space_, statistics_spectrum_, Complex{});
  const auto current = current_;
  const auto spectrum = statistics_spectrum_;
  const auto omega = omega_;
  const auto power_weight = statistics_power_weight_;
  const auto n = problem_.n_omega;
  const auto n_t = problem_.n_t;
  const bool envelope = problem_.family == FieldFamily::envelope;
  Kokkos::parallel_for(
      "solprop_native.statistics.prepare_analytic",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        if (windowed && !(omega(index) >= lower && omega(index) < upper))
          return;
        const double analytic_scale =
            envelope || index == 0 || index + 1 == n ? 1.0 : 2.0;
        spectrum(column, index) =
            analytic_scale / static_cast<double>(n_t) * current(column, index);
        if (weighted)
          spectrum(column, index) *= power_weight(column, index);
      });
  KokkosFFT::execute(*statistics_inverse_plan_, statistics_spectrum_, statistics_time_,
                     KokkosFFT::Normalization::none);
}

void GeneralWorkspace::fill_statistics_pair(const bool windowed, const double lower,
                                            const double upper) {
  if (!statistics_unit_weight_) {
    fill_statistics_time(windowed, lower, upper, true);
    Kokkos::deep_copy(execution_space_, statistics_weighted_time_, statistics_time_);
  }
  fill_statistics_time(windowed, lower, upper);
}

void GeneralWorkspace::form_statistics_power() {
  const auto time_field = statistics_time_;
  const auto weighted_field = statistics_weighted_time_;
  const bool unit_weight = statistics_unit_weight_;
  const auto power = statistics_power_;
  const auto n_pol = problem_.n_pol_state;
  Kokkos::parallel_for(
      "solprop_native.statistics.power",
      Range2(execution_space_, {0, 0}, {problem_.n_modes, problem_.n_t}),
      KOKKOS_LAMBDA(const std::size_t mode, const std::size_t time) {
        double sum = 0.0;
        for (std::size_t pol = 0; pol < n_pol; ++pol)
          sum += unit_weight ? squared_abs(time_field(mode * n_pol + pol, time))
                             : (time_field(mode * n_pol + pol, time) *
                                Kokkos::conj(weighted_field(mode * n_pol + pol, time)))
                                   .real();
        power(mode, time) = sum;
      });
}

void GeneralWorkspace::form_statistics_total_power() {
  const auto power = statistics_power_;
  const auto total = statistics_total_power_;
  const auto modes = problem_.n_modes;
  Kokkos::parallel_for(
      "solprop_native.statistics.total_power",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_t),
      KOKKOS_LAMBDA(const std::size_t time) {
        double sum = 0.0;
        for (std::size_t mode = 0; mode < modes; ++mode)
          sum += power(mode, time);
        total(time) = sum;
      });
}

void GeneralWorkspace::scan_plasma_statistics(const std::size_t first_node,
                                              const std::size_t node_count,
                                              const RealView2& output,
                                              const bool diagnose,
                                              const double preionisation) {
  if (problem_.envelope_plasma) {
    if (diagnose)
      throw std::invalid_argument("the plasma integration diagnostics read the "
                                  "field-grid plasma column, and this problem "
                                  "carries none");
    for (std::size_t channel = 0; channel < problem_.n_channels; ++channel)
      envelope_population(channel, 0, first_node, node_count, output);
    return;
  }
  // solprop SEP 18: the band rule's statistics, and every problem's run diagnostics,
  // come from the band scan; the trapezoid's own scan below is unchanged.
  if (problem_.plasma_band || diagnose) {
    if (diagnose)
      team_band_statistics<true>(first_node, node_count, output, preionisation);
    else
      team_band_statistics<false>(first_node, node_count, output, preionisation);
    return;
  }
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const bool real_family = problem_.family == FieldFamily::field_real;
  const auto coefficients = rate_coefficients_;
  const auto n_pol = problem_.effective_node_pol();
  const auto n_to = problem_.n_to;
  const auto intervals = problem_.rate_intervals;
  const auto dt = problem_.dt_fine;
  const auto log_field_min = problem_.log_field_min;
  const auto d_log_field = problem_.d_log_field;
  const auto field_min = problem_.field_min;
  const auto field_max = problem_.field_max;
  const auto rate_max = problem_.rate_max;
  if (problem_.n_channels > 1) {
    const auto parameters = channel_parameters_;
    const auto channel_coefficients = channel_coefficients_;
    const auto channels = problem_.n_channels;
    Kokkos::parallel_for(
        "solprop_native.plasma.channel_statistics_scan",
        Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0,
                                            node_count),
        KOKKOS_LAMBDA(const std::size_t node) {
          for (std::size_t channel = 0; channel < channels; ++channel) {
            double exposure = 0.0, previous_rate = 0.0, largest_rate = 0.0;
            for (std::size_t time = 0; time < n_to; ++time) {
              const double first =
                  (real_family ? node_real(node * n_pol, time)
                               : node_complex(node * n_pol, time).real());
              const double second =
                  n_pol == 2
                      ? (real_family ? node_real(node * n_pol + 1, time)
                                     : node_complex(node * n_pol + 1, time).real())
                      : 0.0;
              const double magnitude =
                  n_pol == 2 ? Kokkos::hypot(first, second) : Kokkos::abs(first);
              const double rate = ionisation_rate(
                  magnitude, channel_coefficients, parameters(channel, 0),
                  parameters(channel, 1), parameters(channel, 2),
                  parameters(channel, 3), parameters(channel, 4),
                  static_cast<std::size_t>(parameters(channel, 5)),
                  static_cast<std::size_t>(parameters(channel, 6)));
              if (time > 0)
                exposure += trapezoid(rate, previous_rate, dt);
              previous_rate = rate;
              if (rate > largest_rate)
                largest_rate = rate;
            }
            output(first_node + node, 2 * channel) = exposure;
            output(first_node + node, 2 * channel + 1) = largest_rate;
          }
        });
    return;
  }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const auto rates = plasma_rate_;
  const auto exposures = plasma_exposure_;
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const TeamPolicy policy(response_space_for(node_count), static_cast<int>(node_count),
                          static_cast<int>(response_team_size()));
  Kokkos::parallel_for(
      "solprop_native.plasma.device_statistics_scan", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto node = static_cast<std::size_t>(team.league_rank());
        const auto base = node * n_pol;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, static_cast<int>(n_to)),
            [&](const int time_value) {
              const auto time = static_cast<std::size_t>(time_value);
              const double first = (real_family ? node_real(base, time)
                                                : node_complex(base, time).real());
              const double second =
                  n_pol == 2 ? (real_family ? node_real(base + 1, time)
                                            : node_complex(base + 1, time).real())
                             : 0.0;
              const double magnitude =
                  n_pol == 2 ? Kokkos::hypot(first, second) : Kokkos::abs(first);
              rates(node, time) =
                  ionisation_rate(magnitude, coefficients, log_field_min, d_log_field,
                                  field_min, field_max, rate_max, intervals);
            });
        team.team_barrier();
        Kokkos::parallel_scan(
            Kokkos::TeamThreadRange(team, static_cast<int>(n_to)),
            [&](const int time_value, double& update, const bool final) {
              const auto time = static_cast<std::size_t>(time_value);
              if (time != 0)
                update += trapezoid(rates(node, time), rates(node, time - 1), dt);
              if (final)
                exposures(node, time) = update;
            });
        double largest_rate = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n_to)),
            [&](const int time_value, double& largest) {
              const auto time = static_cast<std::size_t>(time_value);
              if (rates(node, time) > largest)
                largest = rates(node, time);
            },
            Kokkos::Max<double>(largest_rate));
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          output(first_node + node, 0) = exposures(node, n_to - 1);
          output(first_node + node, 1) = largest_rate;
        });
      });
#else
  Kokkos::parallel_for(
      "solprop_native.plasma.statistics_scan",
      Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0,
                                          node_count),
      KOKKOS_LAMBDA(const std::size_t node) {
        const auto base = node * n_pol;
        double exposure = 0.0;
        double previous_rate = 0.0;
        double largest_rate = 0.0;
        for (std::size_t time = 0; time < n_to; ++time) {
          const double first =
              (real_family ? node_real(base, time) : node_complex(base, time).real());
          const double second =
              n_pol == 2 ? (real_family ? node_real(base + 1, time)
                                        : node_complex(base + 1, time).real())
                         : 0.0;
          const double magnitude =
              n_pol == 2 ? Kokkos::hypot(first, second) : Kokkos::abs(first);
          const double rate =
              ionisation_rate(magnitude, coefficients, log_field_min, d_log_field,
                              field_min, field_max, rate_max, intervals);
          if (time > 0)
            exposure += trapezoid(rate, previous_rate, dt);
          previous_rate = rate;
          if (rate > largest_rate)
            largest_rate = rate;
        }
        output(first_node + node, 0) = exposure;
        output(first_node + node, 1) = largest_rate;
      });
#endif
}

std::vector<double> GeneralWorkspace::statistics(
    const std::uint64_t flags, const std::vector<double>& windows,
    const std::vector<double>& on_axis, const double density, const double taper,
    const double preionisation, const double intensity_factor,
    const std::vector<double>& channel_preionisation,
    const std::vector<double>& power_weight) {
  require_open();
  const auto& selected_weight =
      power_weight.empty() ? problem_.power_weight : power_weight;
  statistics_unit_weight_ = std::all_of(selected_weight.begin(), selected_weight.end(),
                                        [](double value) { return value == 1.0; });
  if (!selected_weight.empty()) {
    require_extent(selected_weight.size(), statistics_power_weight_.size(),
                   "statistics power weight");
    auto host_weight = Kokkos::create_mirror_view(statistics_power_weight_);
    for (std::size_t column = 0; column < problem_.n_columns(); ++column)
      for (std::size_t index = 0; index < problem_.n_omega; ++index) {
        const double value = selected_weight[column * problem_.n_omega + index];
        if (!(value > 0.0) || !std::isfinite(value))
          throw std::invalid_argument(
              "statistics power weight must be positive and finite");
        host_weight(column, index) = value;
      }
    Kokkos::deep_copy(execution_space_, statistics_power_weight_, host_weight);
  } else {
    Kokkos::deep_copy(execution_space_, statistics_power_weight_, 1.0);
  }
  const auto power_weights = statistics_power_weight_;
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (windows.size() % 2 != 0)
    throw std::invalid_argument("statistics windows must contain lower/upper pairs");
  const auto requested = [flags](const StatisticsFlag flag) {
    return (flags & static_cast<std::uint64_t>(flag)) != 0U;
  };
  // The temporal statistics reconstruct whole columns; on free geometry their
  // buffers hold one node chunk, and the reference refuses these names on any
  // spatial problem, so the adapter never asks. Refuse here as well rather than
  // write past the chunk.
  // The modal power weight has no rows on a spatial problem, radial included, and
  // the energy kernels below read it for every column.
  if ((problem_.separable_geometry() || statistics_power_weight_.extent(0) == 0) &&
      (requested(statistics_omega0) || requested(statistics_energy) ||
       requested(statistics_energy_window) || requested(statistics_peak_power) ||
       requested(statistics_peak_power_window) ||
       requested(statistics_peak_power_total) || requested(statistics_fwhm)))
    throw std::invalid_argument(
        "temporal statistics are not spatial saved-field statistics");
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  using StatisticsTeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using StatisticsTeamMember = StatisticsTeamPolicy::member_type;
#endif
  const auto n_windows = windows.size() / 2;
  constexpr auto absent = std::numeric_limits<std::size_t>::max();
  std::size_t packed_size = 0;
  const auto reserve = [&packed_size, requested](const StatisticsFlag flag,
                                                 const std::size_t count) {
    if (!requested(flag))
      return absent;
    const auto offset = packed_size;
    packed_size = checked_sum(packed_size, count, "statistics packed output");
    return offset;
  };
  const auto omega0_offset = reserve(statistics_omega0, problem_.n_columns());
  const auto energy_offset = reserve(statistics_energy, problem_.n_columns());
  const auto energy_window_offset =
      reserve(statistics_energy_window, checked_product(n_windows, problem_.n_columns(),
                                                        "windowed-energy output"));
  const auto peak_offset = reserve(statistics_peak_power, problem_.n_modes);
  const auto peak_window_offset =
      reserve(statistics_peak_power_window,
              checked_product(n_windows, problem_.n_modes, "windowed-peak output"));
  const auto total_peak_offset = reserve(statistics_peak_power_total, 1);
  const auto density_offset = reserve(statistics_electron_density, 1);
  const auto rate_offset = reserve(statistics_peak_ionisation_rate, 1);
  const auto intensity_offset = reserve(statistics_peak_intensity, 1);
  const auto quadrature_offset = reserve(statistics_quadrature_error, 1);
  const auto fwhm_offset = reserve(statistics_fwhm, problem_.n_t);
  const auto populations_offset =
      reserve(statistics_populations,
              checked_product(2,
                              checked_product(problem_.n_channels, problem_.n_nodes,
                                              "population output"),
                              "population output"));
  // solprop SEP 18's run diagnostics, packed last: six maxima over every node.
  const auto diagnostics_offset =
      reserve(statistics_plasma_integration, plasma_diagnostic_parts);
  const bool diagnose = diagnostics_offset != absent;
  if (populations_offset != absent || (diagnose && problem_.n_channels > 1)) {
    require_extent(channel_preionisation.size(), problem_.n_channels,
                   "statistics channel preionisation");
    for (double value : channel_preionisation)
      if (!std::isfinite(value) || value < 0.0 || value > 1.0)
        throw std::invalid_argument(
            "statistics channel preionisation must be in [0, 1]");
    auto initial = Kokkos::create_mirror_view(stage_channel_preionisation_);
    for (std::size_t channel = 0; channel < problem_.n_channels; ++channel)
      initial(0, channel) = channel_preionisation[channel];
    Kokkos::deep_copy(execution_space_, stage_channel_preionisation_, initial);
  }

  if (statistics_packed_.size() < packed_size) {
    execution_space_.fence("solprop_native statistics output resize");
    statistics_packed_ = RealView("solprop_native.statistics.packed", packed_size);
  }
  const auto packed = statistics_packed_;
  if (packed_size != 0)
    Kokkos::deep_copy(execution_space_, packed,
                      Kokkos::Experimental::quiet_NaN_v<double>);

  if (omega0_offset != absent || energy_offset != absent) {
    if (omega_.empty() || !(problem_.dt > 0.0))
      throw std::logic_error("native statistics need the coarse frequency axis and dt");
    const auto current = current_;
    const auto omega = omega_;
    const auto n = problem_.n_omega;
    const auto n_t = problem_.n_t;
    const auto dt = problem_.dt;
    const bool envelope = problem_.family == FieldFamily::envelope;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    const StatisticsTeamPolicy policy(execution_space_,
                                      static_cast<int>(problem_.n_columns()),
                                      SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
    Kokkos::parallel_for(
        "solprop_native.statistics.device_pack_spectral", policy,
        KOKKOS_LAMBDA(const StatisticsTeamMember& team) {
          const auto column = static_cast<std::size_t>(team.league_rank());
          double density_sum = 0.0;
          double weighted_sum = 0.0;
          double energy_sum = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, static_cast<int>(n)),
              [&](const int index_value, double& update) {
                const auto index = static_cast<std::size_t>(index_value);
                update += (squared_abs(current(column, index)) *
                           power_weights(column, index));
              },
              density_sum);
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, static_cast<int>(n)),
              [&](const int index_value, double& update) {
                const auto index = static_cast<std::size_t>(index_value);
                update += omega(index) * (squared_abs(current(column, index)) *
                                          power_weights(column, index));
              },
              weighted_sum);
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, static_cast<int>(n)),
              [&](const int index_value, double& update) {
                const auto index = static_cast<std::size_t>(index_value);
                const double weight =
                    envelope || index == 0 || index + 1 == n ? 1.0 : 4.0;
                update += weight * (squared_abs(current(column, index)) *
                                    power_weights(column, index));
              },
              energy_sum);
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            if (omega0_offset != absent)
              packed(omega0_offset + column) =
                  density_sum > 0.0 ? weighted_sum / density_sum
                                    : Kokkos::Experimental::quiet_NaN_v<double>;
            if (energy_offset != absent)
              packed(energy_offset + column) =
                  dt / static_cast<double>(n_t) * energy_sum;
          });
        });
#else
    Kokkos::parallel_for(
        "solprop_native.statistics.pack_spectral",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_columns()),
        KOKKOS_LAMBDA(const std::size_t column) {
          double density_sum = 0.0;
          double weighted_sum = 0.0;
          double energy_sum = 0.0;
          for (std::size_t index = 0; index < n; ++index) {
            const double power =
                (squared_abs(current(column, index)) * power_weights(column, index));
            density_sum += power;
            weighted_sum += omega(index) * power;
            const double weight = envelope || index == 0 || index + 1 == n ? 1.0 : 4.0;
            energy_sum += weight * power;
          }
          if (omega0_offset != absent)
            packed(omega0_offset + column) =
                density_sum > 0.0 ? weighted_sum / density_sum
                                  : Kokkos::Experimental::quiet_NaN_v<double>;
          if (energy_offset != absent)
            packed(energy_offset + column) = dt / static_cast<double>(n_t) * energy_sum;
        });
#endif
  }

  if (energy_window_offset != absent) {
    if (omega_.empty() || !(problem_.dt > 0.0))
      throw std::logic_error("native statistics need the coarse frequency axis and dt");
    const auto current = current_;
    const auto omega = omega_;
    const auto n = problem_.n_omega;
    const auto n_t = problem_.n_t;
    const auto dt = problem_.dt;
    const auto columns = problem_.n_columns();
    const bool envelope = problem_.family == FieldFamily::envelope;
    for (std::size_t window = 0; window < n_windows; ++window) {
      const double lower = windows[2 * window];
      const double upper = windows[2 * window + 1];
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
      const StatisticsTeamPolicy policy(execution_space_, static_cast<int>(columns),
                                        SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
      Kokkos::parallel_for(
          "solprop_native.statistics.device_pack_energy_window", policy,
          KOKKOS_LAMBDA(const StatisticsTeamMember& team) {
            const auto column = static_cast<std::size_t>(team.league_rank());
            double sum = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, static_cast<int>(n)),
                [&](const int index_value, double& update) {
                  const auto index = static_cast<std::size_t>(index_value);
                  if (!(omega(index) >= lower && omega(index) < upper))
                    return;
                  const double weight =
                      envelope || index == 0 || index + 1 == n ? 1.0 : 4.0;
                  update += weight * (squared_abs(current(column, index)) *
                                      power_weights(column, index));
                },
                sum);
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              packed(energy_window_offset + window * columns + column) =
                  dt / static_cast<double>(n_t) * sum;
            });
          });
#else
      Kokkos::parallel_for(
          "solprop_native.statistics.pack_energy_window",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, columns),
          KOKKOS_LAMBDA(const std::size_t column) {
            double sum = 0.0;
            for (std::size_t index = 0; index < n; ++index) {
              if (!(omega(index) >= lower && omega(index) < upper))
                continue;
              const double weight =
                  envelope || index == 0 || index + 1 == n ? 1.0 : 4.0;
              sum += weight * (squared_abs(current(column, index)) *
                               power_weights(column, index));
            }
            packed(energy_window_offset + window * columns + column) =
                dt / static_cast<double>(n_t) * sum;
          });
#endif
    }
  }

  const bool needs_base_power = peak_offset != absent || total_peak_offset != absent ||
                                intensity_offset != absent || fwhm_offset != absent;
  if (needs_base_power) {
    fill_statistics_pair(false, 0.0, 0.0);
    form_statistics_power();
  }
  if (peak_offset != absent) {
    const auto power = statistics_power_;
    const auto n_t = problem_.n_t;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    const StatisticsTeamPolicy policy(execution_space_,
                                      static_cast<int>(problem_.n_modes),
                                      SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
    Kokkos::parallel_for(
        "solprop_native.statistics.device_pack_peak_power", policy,
        KOKKOS_LAMBDA(const StatisticsTeamMember& team) {
          const auto mode = static_cast<std::size_t>(team.league_rank());
          double peak = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, static_cast<int>(n_t)),
              [&](const int time_value, double& update) {
                const auto time = static_cast<std::size_t>(time_value);
                if (power(mode, time) > update)
                  update = power(mode, time);
              },
              Kokkos::Max<double>(peak));
          Kokkos::single(Kokkos::PerTeam(team),
                         [&]() { packed(peak_offset + mode) = peak; });
        });
#else
    Kokkos::parallel_for(
        "solprop_native.statistics.pack_peak_power",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_modes),
        KOKKOS_LAMBDA(const std::size_t mode) {
          double peak = 0.0;
          for (std::size_t time = 0; time < n_t; ++time)
            if (power(mode, time) > peak)
              peak = power(mode, time);
          packed(peak_offset + mode) = peak;
        });
#endif
  }
  if (total_peak_offset != absent || fwhm_offset != absent) {
    form_statistics_total_power();
    if (total_peak_offset != absent) {
      const auto total = statistics_total_power_;
      const auto n_t = problem_.n_t;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
      const StatisticsTeamPolicy policy(execution_space_, 1,
                                        SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
      Kokkos::parallel_for(
          "solprop_native.statistics.device_pack_peak_power_total", policy,
          KOKKOS_LAMBDA(const StatisticsTeamMember& team) {
            double peak = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, static_cast<int>(n_t)),
                [&](const int time_value, double& update) {
                  const auto time = static_cast<std::size_t>(time_value);
                  if (total(time) > update)
                    update = total(time);
                },
                Kokkos::Max<double>(peak));
            Kokkos::single(Kokkos::PerTeam(team),
                           [&]() { packed(total_peak_offset) = peak; });
          });
#else
      Kokkos::parallel_for(
          "solprop_native.statistics.pack_peak_power_total",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
          KOKKOS_LAMBDA(const std::size_t) {
            double peak = 0.0;
            for (std::size_t time = 0; time < n_t; ++time)
              if (total(time) > peak)
                peak = total(time);
            packed(total_peak_offset) = peak;
          });
#endif
    }
    if (fwhm_offset != absent) {
      const auto destination = Kokkos::subview(
          packed, std::make_pair(fwhm_offset, fwhm_offset + problem_.n_t));
      Kokkos::deep_copy(execution_space_, destination, statistics_total_power_);
    }
  }
  if (intensity_offset != absent) {
    if (on_axis.empty() || on_axis.size() % problem_.n_modes != 0 ||
        on_axis.size() / problem_.n_modes > 2)
      throw std::invalid_argument(
          "on-axis synthesis must have one or two rows and n_modes columns");
    Kokkos::deep_copy(execution_space_, statistics_on_axis_, 0.0);
    const auto destination = Kokkos::subview(
        statistics_on_axis_, std::make_pair<std::size_t>(0, on_axis.size()));
    copy_vector(execution_space_, destination, on_axis);
    const auto matrix = statistics_on_axis_;
    const auto field = statistics_time_;
    const auto weighted_field = statistics_weighted_time_;
    const bool unit_weight = statistics_unit_weight_;
    const auto rows = on_axis.size() / problem_.n_modes;
    const auto modes = problem_.n_modes;
    const auto n_pol = problem_.n_pol_state;
    const auto n_t = problem_.n_t;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    const StatisticsTeamPolicy policy(execution_space_, 1,
                                      SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
    Kokkos::parallel_for(
        "solprop_native.statistics.device_pack_peak_intensity", policy,
        KOKKOS_LAMBDA(const StatisticsTeamMember& team) {
          double peak = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, static_cast<int>(n_t)),
              [&](const int time_value, double& update) {
                const auto time = static_cast<std::size_t>(time_value);
                double power = 0.0;
                for (std::size_t row = 0; row < rows; ++row) {
                  for (std::size_t pol = 0; pol < n_pol; ++pol) {
                    Complex value{}, weighted_value{};
                    for (std::size_t mode = 0; mode < modes; ++mode) {
                      value +=
                          matrix(row * modes + mode) * field(mode * n_pol + pol, time);
                      if (!unit_weight)
                        weighted_value += matrix(row * modes + mode) *
                                          weighted_field(mode * n_pol + pol, time);
                    }
                    power += unit_weight
                                 ? squared_abs(value)
                                 : (value * Kokkos::conj(weighted_value)).real();
                  }
                }
                if (power > update)
                  update = power;
              },
              Kokkos::Max<double>(peak));
          Kokkos::single(Kokkos::PerTeam(team),
                         [&]() { packed(intensity_offset) = intensity_factor * peak; });
        });
#else
    Kokkos::parallel_for(
        "solprop_native.statistics.pack_peak_intensity",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) {
          double peak = 0.0;
          for (std::size_t time = 0; time < n_t; ++time) {
            double power = 0.0;
            for (std::size_t row = 0; row < rows; ++row) {
              for (std::size_t pol = 0; pol < n_pol; ++pol) {
                Complex value{}, weighted_value{};
                for (std::size_t mode = 0; mode < modes; ++mode) {
                  value += matrix(row * modes + mode) * field(mode * n_pol + pol, time);
                  if (!unit_weight)
                    weighted_value += matrix(row * modes + mode) *
                                      weighted_field(mode * n_pol + pol, time);
                }
                power += unit_weight ? squared_abs(value)
                                     : (value * Kokkos::conj(weighted_value)).real();
              }
            }
            if (power > peak)
              peak = power;
          }
          packed(intensity_offset) = intensity_factor * peak;
        });
#endif
  }

  if (peak_window_offset != absent) {
    for (std::size_t window = 0; window < n_windows; ++window) {
      fill_statistics_pair(true, windows[2 * window], windows[2 * window + 1]);
      form_statistics_power();
      const auto power = statistics_power_;
      const auto n_t = problem_.n_t;
      const auto modes = problem_.n_modes;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
      const StatisticsTeamPolicy policy(execution_space_, static_cast<int>(modes),
                                        SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
      Kokkos::parallel_for(
          "solprop_native.statistics.device_pack_peak_power_window", policy,
          KOKKOS_LAMBDA(const StatisticsTeamMember& team) {
            const auto mode = static_cast<std::size_t>(team.league_rank());
            double peak = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, static_cast<int>(n_t)),
                [&](const int time_value, double& update) {
                  const auto time = static_cast<std::size_t>(time_value);
                  if (power(mode, time) > update)
                    update = power(mode, time);
                },
                Kokkos::Max<double>(peak));
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              packed(peak_window_offset + window * modes + mode) = peak;
            });
          });
#else
      Kokkos::parallel_for(
          "solprop_native.statistics.pack_peak_power_window",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, modes),
          KOKKOS_LAMBDA(const std::size_t mode) {
            double peak = 0.0;
            for (std::size_t time = 0; time < n_t; ++time)
              if (power(mode, time) > peak)
                peak = power(mode, time);
            packed(peak_window_offset + window * modes + mode) = peak;
          });
#endif
    }
  }

  if (density_offset != absent || rate_offset != absent ||
      populations_offset != absent || diagnose) {
    if (!problem_.plasma)
      throw std::logic_error("this problem declares no compiled plasma response");
    if (diagnose && problem_.envelope_plasma)
      throw std::invalid_argument("the plasma integration diagnostics read the "
                                  "field-grid plasma column, and this problem "
                                  "carries none");
    RealView2 source = stash_current_;
    // The diagnostics rescan the plane, and the plasma statistics then read the scan.
    const bool source_is_fraction = stash_is_current_ && !diagnose;
    if (!source_is_fraction) {
      source = plasma_statistics_scratch_;
      if (problem_.separable_geometry()) {
        separable_load_current();
        for (std::size_t unit = 0; unit < transverse_units();
             unit += transverse_units_per_chunk()) {
          const auto chunk = transverse_chunk(unit);
          separable_pack_chunk(chunk, true);
          separable_to_time(chunk);
          separable_nodes(chunk);
          scan_plasma_statistics(chunk.first_node, chunk.nodes, source, diagnose,
                                 preionisation);
        }
      } else {
        Kokkos::deep_copy(execution_space_, fine_spectrum_, Complex{});
        const auto current = current_;
        const auto fine = fine_spectrum_;
        const auto crop = crop_index_;
        const auto inverse_n_to = 1.0 / static_cast<double>(problem_.n_to);
        const auto n_to = problem_.n_to;
        const bool analytic = problem_.family == FieldFamily::field_analytic;
        Kokkos::parallel_for(
            "solprop_native.plasma.statistics_pad",
            Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.n_omega}),
            KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
              const double analytic_scale =
                  analytic && index > 0 && 2 * crop(index) < n_to ? 2.0 : 1.0;
              fine(column, crop(index)) =
                  analytic_scale * inverse_n_to * current(column, index);
            });
        if (problem_.family == FieldFamily::field_real)
          KokkosFFT::execute(*inverse_real_plan_, fine_spectrum_, modal_real_,
                             KokkosFFT::Normalization::none);
        else
          KokkosFFT::execute(*inverse_complex_plan_, fine_spectrum_, modal_complex_,
                             KokkosFFT::Normalization::none);
        Kokkos::deep_copy(execution_space_, stage_taper_, taper);
        scale_transform(problem_.n_stages - 1);
        for (std::size_t first = 0; first < problem_.n_nodes;
             first += problem_.node_chunk) {
          const auto count = std::min(problem_.node_chunk, problem_.n_nodes - first);
          synthesise(first, count);
          scan_plasma_statistics(first, count, source, diagnose, preionisation);
        }
      }
    }
    if (populations_offset != absent) {
      const bool envelope = problem_.envelope_plasma;
      const auto channels = problem_.n_channels;
      const auto initial = stage_channel_preionisation_;
      Kokkos::parallel_for(
          "solprop_native.statistics.pack_populations",
          Range2(execution_space_, {0, 0}, {problem_.n_nodes, channels}),
          KOKKOS_LAMBDA(const std::size_t node, const std::size_t channel) {
            const auto offset = populations_offset + 2 * (node * channels + channel);
            packed(offset) =
                source_is_fraction ? source(node, 2 * channel)
                : envelope
                    ? envelope_fraction(source(node, 2 * channel), initial(0, channel))
                    : ionised_fraction(source(node, 2 * channel), initial(0, channel));
            packed(offset + 1) = source(node, 2 * channel + 1);
          });
    }
    const auto nodes = problem_.n_nodes;
    Kokkos::parallel_for(
        "solprop_native.statistics.pack_plasma",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) {
          std::size_t densest = 0;
          double largest_fraction = -1.0;
          for (std::size_t node = 0; node < nodes; ++node) {
            const double fraction =
                source_is_fraction ? source(node, 0)
                                   : ionised_fraction(source(node, 0), preionisation);
            if (fraction > largest_fraction) {
              densest = node;
              largest_fraction = fraction;
            }
          }
          if (density_offset != absent)
            packed(density_offset) = density * largest_fraction;
          if (rate_offset != absent)
            packed(rate_offset) = source(densest, 1);
        });
    if (diagnose) {
      // Each part is a maximum, so the nodes combine in any order.
      const auto parts = plasma_diagnostics_;
      Kokkos::parallel_for(
          "solprop_native.statistics.pack_plasma_integration",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0,
                                              plasma_diagnostic_parts),
          KOKKOS_LAMBDA(const std::size_t part) {
            double largest = 0.0;
            for (std::size_t node = 0; node < nodes; ++node)
              largest = Kokkos::max(largest, parts(node, part));
            packed(diagnostics_offset + part) = largest;
          });
    }
  }
  if (quadrature_offset != absent) {
    const double value = quadrature_error();
    Kokkos::parallel_for(
        "solprop_native.statistics.pack_quadrature",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) { packed(quadrature_offset) = value; });
  }

  if (packed_size == 0)
    return {};
  const auto active =
      Kokkos::subview(packed, std::make_pair<std::size_t>(0, packed_size));
  auto host = Kokkos::create_mirror_view(active);
  Kokkos::deep_copy(execution_space_, host, active);
  execution_space_.fence("solprop_native statistics packed boundary");
  std::vector<double> result(packed_size);
  for (std::size_t index = 0; index < packed_size; ++index)
    result[index] = host(index);
  return result;
}

// solprop SEP 34: the buffers of the magnetic closure's synthesis, allocated at
// construction when the problem's own request has a local flux statistic and here,
// outside any trial step, when a later request first needs them.
void GeneralWorkspace::require_flux_closure() {
  const auto rows = statistics_time_.extent(0);
  const bool time = statistics_weighted_time_.extent(0) != rows;
  const bool magnitude = statistics_magnitude_.extent(0) != statistics_power_.extent(0);
  const bool volume = problem_.separable_geometry() && flux_transverse_.empty();
  if (!time && !magnitude && !volume)
    return;
  execution_space_.fence("solprop_native flux closure allocation");
  if (time)
    statistics_weighted_time_ =
        ComplexView2("solprop_native.statistics.weighted_time", rows, problem_.n_t);
  if (magnitude)
    statistics_magnitude_ =
        RealView2("solprop_native.statistics.flux_magnitude",
                  statistics_power_.extent(0), statistics_power_.extent(1));
  if (volume)
    flux_transverse_ =
        ComplexView3("solprop_native.flux.closure_transverse", transverse_.extent(0),
                     transverse_.extent(1), transverse_.extent(2));
}

// The magnetic closure of every column in time (a radial run's, solprop SEP 34): the
// analytic spectrum times the conjugate admittance, inverted into the statistics'
// time rows and kept in `statistics_weighted_time_`.
void GeneralWorkspace::fill_statistics_closure() {
  Kokkos::deep_copy(execution_space_, statistics_spectrum_, Complex{});
  const auto current = current_;
  const auto spectrum = statistics_spectrum_;
  const auto omega = omega_;
  const auto active = active_mask_;
  const auto flux = flux_;
  const auto n = problem_.n_omega;
  const auto n_t = problem_.n_t;
  const auto n_pol = problem_.n_pol_state;
  const bool envelope = problem_.family == FieldFamily::envelope;
  Kokkos::parallel_for(
      "solprop_native.statistics.prepare_closure",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        const double analytic_scale =
            envelope || index == 0 || index + 1 == n ? 1.0 : 2.0;
        const Complex closing =
            flux.closing(column / n_pol, index, active(index) != 0U, omega(index));
        spectrum(column, index) = analytic_scale / static_cast<double>(n_t) *
                                  (closing * current(column, index));
      });
  KokkosFFT::execute(*statistics_inverse_plan_, statistics_spectrum_, statistics_time_,
                     KokkosFFT::Normalization::none);
  Kokkos::deep_copy(execution_space_, statistics_weighted_time_, statistics_time_);
}

void GeneralWorkspace::separable_time_statistics(const std::size_t axis_peak,
                                                 const std::size_t peak,
                                                 const std::size_t fluence,
                                                 const double intensity_factor) {
  constexpr auto absent = std::numeric_limits<std::size_t>::max();
  constexpr double lowest = std::numeric_limits<double>::lowest();
  const auto current = current_;
  const auto spectrum = statistics_spectrum_;
  const auto field = statistics_time_;
  const auto closure = statistics_weighted_time_;
  const auto packed = statistics_packed_;
  const auto axis_weights = spatial_on_axis_;
  const auto scratch = transverse_;
  const auto closure_scratch = flux_transverse_;
  const auto power = statistics_power_;
  const auto magnitude = statistics_magnitude_;
  const auto partials = spatial_statistics_partials_;
  const auto synthesis = synthesis_;
  const auto flux = flux_;
  const auto active = active_mask_;
  const auto omega = omega_;
  const auto n_modes = problem_.n_modes;
  const auto n_nodes = problem_.n_nodes;
  const auto n_pol = problem_.n_pol_state;
  const auto layout = transverse_layout();
  const auto n = problem_.n_omega;
  const auto n_t = problem_.n_t;
  const auto dt = problem_.dt;
  const bool envelope = problem_.family == FieldFamily::envelope;
  if (axis_peak != absent || fluence != absent) {
    // The axis's closure first, kept in the closure rows, then its field: each the
    // coherent on-axis sum of the analytic spectrum (solprop SEP 34).
    for (const bool closed : {true, false}) {
      Kokkos::deep_copy(execution_space_, spectrum, Complex{});
      Kokkos::parallel_for(
          "solprop_native.free.statistics_axis_input",
          Range2(execution_space_, {0, 0}, {n_pol, n}),
          KOKKOS_LAMBDA(const std::size_t pol, const std::size_t index) {
            const bool live = active(index) != 0U;
            Complex sum{};
            for (std::size_t mode = 0; mode < n_modes; ++mode) {
              const Complex value = current(mode * n_pol + pol, index);
              sum += axis_weights(mode) *
                     (closed ? flux.closing(mode, index, live, omega(index)) * value
                             : value);
            }
            const double scale =
                (envelope || index == 0 || index + 1 == n ? 1.0 : 2.0) /
                static_cast<double>(n_t);
            spectrum(pol, index) = scale * sum;
          });
      KokkosFFT::execute(*statistics_inverse_plan_, statistics_spectrum_,
                         statistics_time_, KokkosFFT::Normalization::none);
      if (closed)
        Kokkos::deep_copy(
            execution_space_,
            Kokkos::subview(closure, std::make_pair<std::size_t>(0, n_pol),
                            Kokkos::ALL),
            Kokkos::subview(field, std::make_pair<std::size_t>(0, n_pol), Kokkos::ALL));
    }
    Kokkos::parallel_for(
        "solprop_native.free.statistics_axis_reduce",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) {
          double sum = 0.0, maximum = lowest;
          for (std::size_t time = 0; time < n_t; ++time) {
            double value = 0.0;
            for (std::size_t pol = 0; pol < n_pol; ++pol)
              value += (field(pol, time) * Kokkos::conj(closure(pol, time))).real();
            sum += value;
            maximum = Kokkos::max(maximum, value);
          }
          if (axis_peak != absent)
            packed(axis_peak) = intensity_factor * maximum;
          if (fluence != absent)
            packed(fluence + n_nodes) = intensity_factor * dt * sum;
        });
  }
  if (peak == absent && fluence == absent)
    return;
  const auto transverse_count =
      static_cast<double>(problem_.slab_geometry() ? problem_.n_y : n_modes);
  Kokkos::parallel_for(
      "solprop_native.free.statistics_transverse_input",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        const auto mode = column / n_pol;
        const Complex value = current(column, index) / transverse_count;
        const auto plane = index * n_pol + column % n_pol;
        layout.cell(scratch, plane, mode) = value;
        layout.cell(closure_scratch, plane, mode) =
            flux.closing(mode, index, active(index) != 0U, omega(index)) * value;
      });
  transverse_inverse();
  transverse_inverse(flux_transverse_);
  const bool slab = problem_.slab_geometry();
  const auto n_x = problem_.n_x;
  const auto n_p = problem_.n_p;
  // The planar synthesis carries the temporal pad scale; the statistics read the
  // coarse field, so it is divided out (solprop's `collect_spatial`).
  const double crop_scale =
      static_cast<double>(problem_.n_t) / static_cast<double>(problem_.n_to);
  for (std::size_t unit = 0; unit < transverse_units();
       unit += transverse_units_per_chunk()) {
    const auto chunk = transverse_chunk(unit);
    const auto first = chunk.first_node;
    const auto first_row = chunk.first_row;
    const auto count = chunk.nodes;
    // The chunk's closure first, kept in the closure rows, then its field.
    for (const bool closed : {true, false}) {
      const auto source = closed ? closure_scratch : scratch;
      Kokkos::deep_copy(execution_space_, spectrum, Complex{});
      Kokkos::parallel_for(
          "solprop_native.free.statistics_temporal_input",
          Range2(execution_space_, {0, 0}, {count * n_pol, n}),
          KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
            const double scale =
                (envelope || index == 0 || index + 1 == n ? 1.0 : 2.0) /
                static_cast<double>(n_t);
            if (slab) {
              // Node ix of the chunk's y row iy: the slab synthesis of that row.
              const auto ix = row % n_x;
              const auto rows = first_row + (row / n_x) * n_p;
              Complex sum{};
              for (std::size_t channel = 0; channel < n_p; ++channel)
                sum +=
                    synthesis(ix, channel) * source(index, layout.first(rows + channel),
                                                    layout.second(rows + channel));
              spectrum(row, index) = scale * (crop_scale * sum);
            } else {
              const auto node = first + row / n_pol;
              spectrum(row, index) =
                  scale * layout.cell(source, index * n_pol + row % n_pol, node);
            }
          });
      KokkosFFT::execute(*statistics_inverse_plan_, statistics_spectrum_,
                         statistics_time_, KokkosFFT::Normalization::none);
      if (closed)
        Kokkos::deep_copy(
            execution_space_,
            Kokkos::subview(closure, std::make_pair<std::size_t>(0, count * n_pol),
                            Kokkos::ALL),
            Kokkos::subview(field, std::make_pair<std::size_t>(0, count * n_pol),
                            Kokkos::ALL));
    }
    Kokkos::parallel_for(
        "solprop_native.free.statistics_intensity",
        Range2(execution_space_, {0, 0}, {count, n_t}),
        KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
          double value = 0.0, size = 0.0;
          for (std::size_t pol = 0; pol < n_pol; ++pol) {
            const Complex electric = field(node * n_pol + pol, time);
            const Complex magnetic = closure(node * n_pol + pol, time);
            value += (electric * Kokkos::conj(magnetic)).real();
            size += Kokkos::sqrt(squared_abs(electric) * squared_abs(magnetic));
          }
          power(node, time) = intensity_factor * value;
          magnitude(node, time) = intensity_factor * size;
        });
    Kokkos::parallel_for(
        "solprop_native.free.statistics_fluence",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, count),
        KOKKOS_LAMBDA(const std::size_t node) {
          double maximum = lowest, sum = 0.0, size = 0.0;
          for (std::size_t time = 0; time < n_t; ++time) {
            sum += power(node, time);
            size += magnitude(node, time);
            maximum = Kokkos::max(maximum, power(node, time));
          }
          partials(first + node, 1) = maximum;
          if (fluence != absent) {
            packed(fluence + first + node) = dt * sum;
            packed(fluence + n_nodes + 1 + first + node) = dt * size;
          }
        });
  }
  if (peak != absent)
    Kokkos::parallel_for(
        "solprop_native.free.statistics_peak",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) {
          double maximum = lowest;
          for (std::size_t node = 0; node < n_nodes; ++node)
            maximum = Kokkos::max(maximum, partials(node, 1));
          packed(peak) = maximum;
        });
}

std::vector<double> GeneralWorkspace::aperture_statistics(
    const std::uint64_t flags, const std::vector<double>& apertures,
    const int profile_code, const int exponent, const std::vector<double>& regions,
    const double intensity_factor) {
  require_open();
  if (!initialized_ || !far_field_)
    throw std::logic_error(
        "far-field statistics need an initial field on a Cartesian or line grid");
  return far_field_->run(
      flags, apertures, profile_code, exponent, regions, intensity_factor, current_,
      FarFieldPass::Inputs{spatial_spectral_weights_, spatial_on_axis_, spatial_.square,
                           spatial_.square_low, spatial_.transverse,
                           spatial_.transverse_low, spatial_.spectral_rate,
                           spatial_.transverse_rate, omega_, active_mask_,
                           spatial_.evanescent});
}

std::vector<double> GeneralWorkspace::spatial_statistics(const std::uint64_t flags,
                                                         const double intensity_factor,
                                                         const double z) {
  require_open();
  if (!initialized_ || spatial_spectral_weights_.empty())
    throw std::logic_error(
        "spatial statistics need an initial field and spatial geometry");
  if (flux_.square.empty())
    throw std::logic_error("spatial statistics need the run's plane measure");
  if ((flags & ~std::uint64_t{31}) != 0 || !std::isfinite(intensity_factor) ||
      intensity_factor <= 0.0 || !std::isfinite(z))
    throw std::invalid_argument(
        "invalid spatial statistic flags, intensity factor or position");
  constexpr auto absent = std::numeric_limits<std::size_t>::max();
  constexpr double lowest = std::numeric_limits<double>::lowest();
  std::size_t packed_size = 0;
  const auto reserve = [flags, &packed_size](const std::uint64_t bit,
                                             const std::size_t count) {
    if ((flags & bit) == 0)
      return absent;
    const auto offset = packed_size;
    packed_size = checked_sum(packed_size, count, "spatial statistics output");
    return offset;
  };
  const auto energy_offset = reserve(1, 1);
  const auto axis_peak_offset = reserve(2, 1);
  const auto peak_offset = reserve(4, 1);
  const auto spectrum_offset = reserve(8, problem_.n_omega);
  // Widths read the signed fluence at the nodes, the separately reconstructed axis
  // and each node's magnitude `dt sum |E| |G|`, whose largest value fixes the
  // fluence's rounding budget (solprop SEP 34).
  const auto fluence_offset =
      reserve(16, checked_sum(checked_product(2, problem_.n_nodes, "spatial nodes"), 1,
                              "spatial statistics output"));
  if (packed_size == 0)
    return {};
  if (statistics_packed_.size() < packed_size) {
    execution_space_.fence("solprop_native spatial statistics output resize");
    statistics_packed_ = RealView("solprop_native.statistics.packed", packed_size);
  }
  // Every field statistic is the axial flux of the medium at the plane z.
  flux_at(z);
  const bool local = (flags & (2U | 4U | 16U)) != 0;
  if (local)
    require_flux_closure();
  const auto packed = statistics_packed_;
  const auto current = current_;
  const auto spectral_weights = spatial_spectral_weights_;
  const auto axis_weights = spatial_on_axis_;
  const auto partials = spatial_statistics_partials_;
  const auto flux = flux_;
  const auto active = active_mask_;
  const auto omega = omega_;
  const auto n_modes = problem_.n_modes;
  const auto n_nodes = problem_.n_nodes;
  const auto n_pol = problem_.n_pol_state;
  const auto n = problem_.n_omega;
  const auto n_t = problem_.n_t;
  const auto dt = problem_.dt;
  const bool envelope = problem_.family == FieldFamily::envelope;
  if (energy_offset != absent) {
    Kokkos::parallel_for(
        "solprop_native.spatial.energy_partials",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n_modes),
        KOKKOS_LAMBDA(const std::size_t mode) {
          double sum = 0.0;
          for (std::size_t index = 0; index < n; ++index) {
            const double admittance =
                flux.closing(mode, index, active(index) != 0U, omega(index)).real();
            if (admittance == 0.0)
              continue;
            double squared = 0.0;
            for (std::size_t pol = 0; pol < n_pol; ++pol)
              squared += squared_abs(current(mode * n_pol + pol, index));
            const auto weight = envelope || index == 0 || index + 1 == n ? 1.0 : 4.0;
            sum += weight * admittance * squared;
          }
          partials(mode, 0) = intensity_factor * spectral_weights(mode) * dt /
                              static_cast<double>(n_t) * sum;
        });
    Kokkos::parallel_for(
        "solprop_native.spatial.pack_energy",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) {
          double sum = 0.0;
          for (std::size_t mode = 0; mode < n_modes; ++mode)
            sum += partials(mode, 0);
          packed(energy_offset) = sum;
        });
  }
  if (spectrum_offset != absent) {
    // The coherent on-axis amplitude paired at each frequency with its closure.
    Kokkos::parallel_for(
        "solprop_native.spatial.axis_spectrum",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n),
        KOKKOS_LAMBDA(const std::size_t index) {
          const bool live = active(index) != 0U;
          double power = 0.0;
          for (std::size_t pol = 0; pol < n_pol; ++pol) {
            Complex electric{}, magnetic{};
            for (std::size_t mode = 0; mode < n_modes; ++mode) {
              const Complex value =
                  axis_weights(mode) * current(mode * n_pol + pol, index);
              electric += value;
              magnetic += flux.closing(mode, index, live, omega(index)) * value;
            }
            power += (electric * Kokkos::conj(magnetic)).real();
          }
          const auto weight = envelope || index == 0 || index + 1 == n ? 1.0 : 4.0;
          packed(spectrum_offset + index) =
              intensity_factor * dt * dt /
              (2.0 * Kokkos::numbers::pi_v<double>)*weight * power;
        });
  }
  if (problem_.separable_geometry()) {
    separable_time_statistics(axis_peak_offset, peak_offset, fluence_offset,
                              intensity_factor);
  } else if (local) {
    if (n_nodes != n_modes)
      throw std::logic_error("radial statistics need one node per Hankel column");
    fill_statistics_closure();
    fill_statistics_time(false, 0.0, 0.0);
    const auto field = statistics_time_;
    const auto closure = statistics_weighted_time_;
    const auto axis_power = statistics_total_power_;
    if (axis_peak_offset != absent || fluence_offset != absent) {
      Kokkos::parallel_for(
          "solprop_native.spatial.axis_intensity",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n_t),
          KOKKOS_LAMBDA(const std::size_t time) {
            double power = 0.0;
            for (std::size_t pol = 0; pol < n_pol; ++pol) {
              Complex electric{}, magnetic{};
              for (std::size_t mode = 0; mode < n_modes; ++mode) {
                electric += axis_weights(mode) * field(mode * n_pol + pol, time);
                magnetic += axis_weights(mode) * closure(mode * n_pol + pol, time);
              }
              power += (electric * Kokkos::conj(magnetic)).real();
            }
            axis_power(time) = intensity_factor * power;
          });
      Kokkos::parallel_for(
          "solprop_native.spatial.pack_axis",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
          KOKKOS_LAMBDA(const std::size_t) {
            double peak = lowest;
            double sum = 0.0;
            for (std::size_t time = 0; time < n_t; ++time) {
              peak = Kokkos::max(peak, axis_power(time));
              sum += axis_power(time);
            }
            if (axis_peak_offset != absent)
              packed(axis_peak_offset) = peak;
            if (fluence_offset != absent)
              packed(fluence_offset + n_nodes) = dt * sum;
          });
    }
    if (peak_offset != absent || fluence_offset != absent) {
      const auto power = statistics_power_;
      const auto magnitude = statistics_magnitude_;
      const auto synthesis = synthesis_;
      const double crop_scale =
          static_cast<double>(problem_.n_t) / static_cast<double>(problem_.n_to);
      Kokkos::parallel_for(
          "solprop_native.spatial.physical_intensity",
          Range2(execution_space_, {0, 0}, {n_nodes, n_t}),
          KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
            double total = 0.0, size = 0.0;
            for (std::size_t pol = 0; pol < n_pol; ++pol) {
              Complex electric{}, magnetic{};
              for (std::size_t mode = 0; mode < n_modes; ++mode) {
                electric += synthesis(node, mode) * field(mode * n_pol + pol, time);
                magnetic += synthesis(node, mode) * closure(mode * n_pol + pol, time);
              }
              electric *= crop_scale;
              magnetic *= crop_scale;
              total += (electric * Kokkos::conj(magnetic)).real();
              size += Kokkos::sqrt(squared_abs(electric) * squared_abs(magnetic));
            }
            power(node, time) = intensity_factor * total;
            magnitude(node, time) = intensity_factor * size;
          });
      Kokkos::parallel_for(
          "solprop_native.spatial.fluence_partials",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n_nodes),
          KOKKOS_LAMBDA(const std::size_t node) {
            double peak = lowest;
            double sum = 0.0, size = 0.0;
            for (std::size_t time = 0; time < n_t; ++time) {
              peak = Kokkos::max(peak, power(node, time));
              sum += power(node, time);
              size += magnitude(node, time);
            }
            partials(node, 1) = peak;
            if (fluence_offset != absent) {
              packed(fluence_offset + node) = dt * sum;
              packed(fluence_offset + n_nodes + 1 + node) = dt * size;
            }
          });
      if (peak_offset != absent)
        Kokkos::parallel_for(
            "solprop_native.spatial.pack_peak",
            Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
            KOKKOS_LAMBDA(const std::size_t) {
              double peak = lowest;
              for (std::size_t node = 0; node < n_nodes; ++node)
                peak = Kokkos::max(peak, partials(node, 1));
              packed(peak_offset) = peak;
            });
    }
  }
  const auto filled =
      Kokkos::subview(packed, std::make_pair<std::size_t>(0, packed_size));
  auto host = Kokkos::create_mirror_view(filled);
  Kokkos::deep_copy(execution_space_, host, filled);
  execution_space_.fence("solprop_native spatial statistics packed boundary");
  std::vector<double> result(packed_size);
  for (std::size_t index = 0; index < packed_size; ++index)
    result[index] = host(index);
  return result;
}

void GeneralWorkspace::finish(const double dz) {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (problem_.low_storage) {
    // The march accepted the step in place (solprop SEP 24 section 6.2): the
    // boundary treatments act on the state itself, and the stash a response pass
    // left is the last evaluation's, at a predicted field, not this one's.
    if (problem_.split_step_absorber)
      split_step(dz, current_);
    else
      mask(current_);
    stash_is_current_ = false;
    execution_space_.fence("solprop_native accepted-step boundary");
    return;
  }
  if (problem_.split_step_absorber)
    split_step(dz, advanced_);
  else
    mask(advanced_);
  std::swap(current_, advanced_);
  if (!stash_current_.empty())
    Kokkos::deep_copy(execution_space_, stash_current_, stash_candidate_);
  stash_is_current_ = !stash_current_.empty();
  execution_space_.fence("solprop_native accepted-step boundary");
}

void GeneralWorkspace::finish_quadrature() {
  if (!coarse_pending_)
    return;
  if (problem_.family == FieldFamily::envelope)
    KokkosFFT::execute(*forward_coarse_complex_plan_, projected_coarse_complex_,
                       projected_coarse_spectrum_, KokkosFFT::Normalization::none);
  else
    KokkosFFT::execute(*forward_coarse_real_plan_, projected_coarse_real_,
                       projected_coarse_spectrum_, KokkosFFT::Normalization::none);
  if (held_plasma())
    held_transform(true);
  coarse_pending_ = false;
}

void GeneralWorkspace::score_quadrature() {
  quadrature_error_ = Kokkos::Experimental::quiet_NaN_v<double>;
  quadrature_stale_ = false;
  if (quadrature_partials_.empty() ||
      (!problem_.nonlinear && !problem_.plasma && !dispersive_))
    return;
  finish_quadrature();
  // SEP 6's dispersive products join the embedded rule's spectrum once, when scored.
  if (dispersive_)
    dispersive_->finish_coarse();

  const auto fine = projected_spectrum_;
  const auto coarse = projected_coarse_spectrum_;
  const auto crop = crop_index_;
  const auto window = omega_window_;
  const auto prefactor_terms = prefactor_terms_;
  const auto prefactor_weights = stage_prefactor_;
  const auto partials = quadrature_partials_;
  const auto last = problem_.n_stages - 1;
  const auto n = problem_.n_omega;
  const auto n_prefactor = problem_.n_prefactor_terms;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const TeamPolicy policy(execution_space_, static_cast<int>(problem_.n_columns()),
                          SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
  Kokkos::parallel_for(
      "solprop_native.transform.device_quadrature_partials", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto column = static_cast<std::size_t>(team.league_rank());
        double difference_sum = 0.0;
        double fine_sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n)),
            [&](const int index_value, double& update) {
              const auto index = static_cast<std::size_t>(index_value);
              Complex prefactor{};
              for (std::size_t term = 0; term < n_prefactor; ++term)
                prefactor += prefactor_weights(last * n_prefactor + term) *
                             prefactor_terms(term, column, index);
              const Complex weight = prefactor * window(index);
              const Complex fine_value = weight * fine(column, crop(index));
              const Complex coarse_value = weight * coarse(column, crop(index));
              update += squared_abs(coarse_value - fine_value);
            },
            difference_sum);
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n)),
            [&](const int index_value, double& update) {
              const auto index = static_cast<std::size_t>(index_value);
              Complex prefactor{};
              for (std::size_t term = 0; term < n_prefactor; ++term)
                prefactor += prefactor_weights(last * n_prefactor + term) *
                             prefactor_terms(term, column, index);
              const Complex weight = prefactor * window(index);
              update += squared_abs(weight * fine(column, crop(index)));
            },
            fine_sum);
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          partials(0, column) = difference_sum;
          partials(1, column) = fine_sum;
        });
      });
#else
  Kokkos::parallel_for(
      "solprop_native.transform.quadrature_partials",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_columns()),
      KOKKOS_LAMBDA(const std::size_t column) {
        double difference_sum = 0.0;
        double fine_sum = 0.0;
        for (std::size_t index = 0; index < n; ++index) {
          Complex prefactor{};
          for (std::size_t term = 0; term < n_prefactor; ++term)
            prefactor += prefactor_weights(last * n_prefactor + term) *
                         prefactor_terms(term, column, index);
          const Complex weight = prefactor * window(index);
          const Complex fine_value = weight * fine(column, crop(index));
          const Complex coarse_value = weight * coarse(column, crop(index));
          fine_sum += squared_abs(fine_value);
          difference_sum += squared_abs(coarse_value - fine_value);
        }
        partials(0, column) = difference_sum;
        partials(1, column) = fine_sum;
      });
#endif
  // The columns' sums in their order, on the device: the last score and the peak
  // since the backend last read it.
  const auto scores = quadrature_scores_;
  const auto columns = problem_.n_columns();
  Kokkos::parallel_for(
      "solprop_native.transform.quadrature_score",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
      KOKKOS_LAMBDA(const std::size_t) {
        double difference_sum = 0.0;
        double fine_sum = 0.0;
        for (std::size_t column = 0; column < columns; ++column) {
          difference_sum += partials(0, column);
          fine_sum += partials(1, column);
        }
        double value = Kokkos::Experimental::quiet_NaN_v<double>;
        if (fine_sum != 0.0)
          value = Kokkos::sqrt(difference_sum / fine_sum);
        scores(0) = value;
        if (fine_sum != 0.0 && !(scores(1) >= value))
          scores(1) = value;
      });
  quadrature_stale_ = true;
}

double GeneralWorkspace::quadrature_error() const {
  if (quadrature_stale_) {
    const auto last = Kokkos::subview(quadrature_scores_, 0);
    Kokkos::deep_copy(execution_space_, quadrature_error_, last);
    execution_space_.fence("solprop_native quadrature score");
    quadrature_stale_ = false;
  }
  return quadrature_error_;
}

double GeneralWorkspace::quadrature_peak() {
  if (quadrature_scores_.extent(0) == 0)
    return Kokkos::Experimental::quiet_NaN_v<double>;
  double peak = 0.0;
  const auto stored = Kokkos::subview(quadrature_scores_, 1);
  Kokkos::deep_copy(execution_space_, peak, stored);
  execution_space_.fence("solprop_native quadrature peak");
  Kokkos::deep_copy(execution_space_, stored,
                    Kokkos::Experimental::quiet_NaN_v<double>);
  return peak;
}

void GeneralWorkspace::promote_fsal() {
  require_open();
  score_quadrature();
  if (!pending_fsal_)
    return;
  const auto stages = stages_;
  const auto whole = whole_propagator_;
  const auto recycled = recycled_;
  const auto last = problem_.n_stages - 1;
  Kokkos::parallel_for(
      "solprop_native.stage.promote_fsal",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        recycled(column, index) = whole(column, index) * stages(last, column, index);
      });
  has_recycled_ = true;
  pending_fsal_ = false;
}

void GeneralWorkspace::clear_fsal() {
  has_recycled_ = false;
  pending_fsal_ = false;
  coarse_pending_ = false;
  if (!stash_current_.empty())
    Kokkos::deep_copy(execution_space_, stash_current_, 0.0);
  stash_is_current_ = false;
}

void GeneralWorkspace::mask(const ComplexView2& values) {
  const auto active = active_mask_;
  Kokkos::parallel_for(
      "solprop_native.mask.accepted",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        if (active(index) == 0U)
          values(column, index) = Complex{};
      });
}

void GeneralWorkspace::split_columns(const double dz, std::size_t active_rows,
                                     const double parseval) {
  const bool tail = temporal_tail_rows_ != 0 && active_rows == temporal_tail_rows_;
  if (active_rows == 0)
    active_rows = coarse_spectrum_.extent(0);
  if (problem_.family == FieldFamily::envelope) {
    if (tail)
      coarse_tail_inverse_complex_.execute();
    else
      KokkosFFT::execute(*coarse_inverse_complex_plan_, coarse_spectrum_,
                         coarse_time_complex_, KokkosFFT::Normalization::none);
    if (monitoring_split())
      add_split_removal(dz, active_rows, parseval);
    const auto time = coarse_time_complex_;
    const auto rate = absorber_coarse_;
    Kokkos::parallel_for(
        "solprop_native.absorber.split_apply_complex",
        Range2(execution_space_, {0, 0}, {active_rows, problem_.n_t}),
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          time(column, index) *= Kokkos::exp(-0.5 * rate(index) * dz);
        });
    if (tail)
      coarse_tail_forward_complex_.execute();
    else
      KokkosFFT::execute(*coarse_forward_complex_plan_, coarse_time_complex_,
                         coarse_spectrum_, KokkosFFT::Normalization::none);
  } else {
    if (tail)
      coarse_tail_inverse_real_.execute();
    else
      KokkosFFT::execute(*coarse_inverse_real_plan_, coarse_spectrum_,
                         coarse_time_real_, KokkosFFT::Normalization::none);
    if (monitoring_split())
      add_split_removal(dz, active_rows, parseval);
    const auto time = coarse_time_real_;
    const auto rate = absorber_coarse_;
    Kokkos::parallel_for(
        "solprop_native.absorber.split_apply_real",
        Range2(execution_space_, {0, 0}, {active_rows, problem_.n_t}),
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          time(column, index) *= Kokkos::exp(-0.5 * rate(index) * dz);
        });
    if (tail)
      coarse_tail_forward_real_.execute();
    else
      KokkosFFT::execute(*coarse_forward_real_plan_, coarse_time_real_,
                         coarse_spectrum_, KokkosFFT::Normalization::none);
  }
}

// What one split takes of the power it finds in the coarse time rows, `sum P (1 -
// exp(-alpha h))` with the boundary monitor's column weights, added to the run's
// accumulator in the memory space (solprop SEP 21, ADR-057): the monitor's reading
// drains it at a check, so every accepted step is counted, checked or not. A
// separable route's rows are real-space rows, measured by `parseval`; zero means the
// rows are the stored columns themselves, weighted by the columns' quadrature.
void GeneralWorkspace::add_split_removal(const double dz, const std::size_t active_rows,
                                         const double parseval) {
  const auto rate = absorber_coarse_;
  const auto share = monitor_share_;
  const auto weights = monitor_column_weights_;
  const bool stored = parseval == 0.0;
  // The share the split takes depends on time alone: formed once, not per row.
  Kokkos::parallel_for(
      "solprop_native.monitor.split_share",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_t),
      KOKKOS_LAMBDA(const std::size_t index) {
        share(index) = -Kokkos::expm1(-rate(index) * dz);
      });
  const Range2 samples(execution_space_, {0, 0}, {active_rows, problem_.n_t});
  if (problem_.family == FieldFamily::envelope) {
    const auto time = coarse_time_complex_;
    Kokkos::parallel_reduce(
        "solprop_native.monitor.split_removal_complex", samples,
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t index, double& update) {
          update += (stored ? weights(row) : parseval) * squared_abs(time(row, index)) *
                    share(index);
        },
        monitor_split_sum_);
  } else {
    const auto time = coarse_time_real_;
    Kokkos::parallel_reduce(
        "solprop_native.monitor.split_removal_real", samples,
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t index, double& update) {
          const double value = time(row, index);
          update += (stored ? weights(row) : parseval) * value * value * share(index);
        },
        monitor_split_sum_);
  }
  const auto removed = monitor_removed_;
  const auto split = monitor_split_sum_;
  Kokkos::parallel_for(
      "solprop_native.monitor.split_removal_add",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
      KOKKOS_LAMBDA(const std::size_t) { removed() += split(); });
}

// Bind solprop's boundary monitor (SEP 21): its four time weights, row-major `(4,
// n_t)`, and its columns' quadrature by mode (empty for unit weights). The views are
// allocated at the first bind and kept; the split's accumulator starts at zero.
void GeneralWorkspace::bind_monitor(const std::vector<double>& weights,
                                    const std::vector<double>& column_weights) {
  require_open();
  const auto n_t = problem_.n_t;
  const auto columns = problem_.n_columns();
  require_extent(weights.size(), 4 * n_t, "monitor weights");
  if (!column_weights.empty())
    require_extent(column_weights.size(), problem_.n_modes, "monitor column weights");
  if (monitor_weights_.extent(0) == 0) {
    monitor_weights_ = RealView2("solprop_native.monitor.weights", 4, n_t);
    monitor_column_weights_ =
        RealView("solprop_native.monitor.column_weights", columns);
    monitor_share_ = RealView("solprop_native.monitor.split_share", n_t);
    monitor_removed_ =
        Kokkos::View<double, Layout, MemorySpace>("solprop_native.monitor.removed");
    monitor_split_sum_ =
        Kokkos::View<double, Layout, MemorySpace>("solprop_native.monitor.split_sum");
  }
  auto host_weights = Kokkos::create_mirror_view(monitor_weights_);
  for (std::size_t row = 0; row < 4; ++row)
    for (std::size_t index = 0; index < n_t; ++index)
      host_weights(row, index) = weights[row * n_t + index];
  Kokkos::deep_copy(execution_space_, monitor_weights_, host_weights);
  // A state column is `mode * n_pol + pol`.
  std::vector<double> expanded(columns, 1.0);
  if (!column_weights.empty())
    for (std::size_t column = 0; column < columns; ++column)
      expanded[column] = column_weights[column / problem_.n_pol_state];
  copy_vector(execution_space_, monitor_column_weights_, expanded);
  Kokkos::deep_copy(execution_space_, monitor_removed_, 0.0);
  execution_space_.fence("solprop_native monitor bind");
  monitor_bound_ = true;
}

// The boundary monitor's reading of the accepted state (solprop SEP 21, ADR-057):
// `sum_t P`, the four weighted sums `sum_t w_k P` and the split's accumulator, which
// this drains; `P(t)` is the columns' weighted `|E(t)|^2` on the coarse time axis.
// The state is transformed chunk by chunk through the statistics buffers, idle
// between steps, with the analytic signal's spectrum: a real field is its real part,
// exactly the inverse real transform solprop's reference takes.
std::vector<double> GeneralWorkspace::monitor_reading() {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (!monitor_bound_)
    throw std::logic_error("no boundary monitor is bound");
  const auto current = current_;
  const auto spectrum = statistics_spectrum_;
  const auto time = statistics_time_;
  const auto weights = monitor_weights_;
  const auto column_weights = monitor_column_weights_;
  const auto n = problem_.n_omega;
  const auto n_t = problem_.n_t;
  const bool envelope = problem_.family == FieldFamily::envelope;
  // A one-sided spectrum's last bin is the Nyquist bin only on an even axis.
  const bool nyquist = n_t % 2 == 0;
  const auto rows = statistics_spectrum_.extent(0);
  const auto columns = problem_.n_columns();
  MonitorSums sums;
  for (std::size_t first = 0; first < columns; first += rows) {
    const auto count = std::min(rows, columns - first);
    // Every bin of the rows read is written, the empty half of a one-sided spectrum
    // with zeros; a partial chunk's other rows are transformed but never read.
    Kokkos::parallel_for(
        "solprop_native.monitor.spectrum",
        Range2(execution_space_, {0, 0}, {count, n_t}),
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
          if (index >= n) {
            spectrum(row, index) = Complex{};
            return;
          }
          const bool single = envelope || index == 0 || (nyquist && index + 1 == n);
          spectrum(row, index) = (single ? 1.0 : 2.0) / static_cast<double>(n_t) *
                                 current(first + row, index);
        });
    KokkosFFT::execute(*statistics_inverse_plan_, spectrum, time,
                       KokkosFFT::Normalization::none);
    MonitorSums chunk;
    Kokkos::parallel_reduce(
        "solprop_native.monitor.sums",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, count * n_t),
        KOKKOS_LAMBDA(const std::size_t item, MonitorSums& update) {
          const auto row = item / n_t;
          const auto index = item % n_t;
          const Complex value = time(row, index);
          const double power =
              column_weights(first + row) *
              (envelope ? squared_abs(value) : value.real() * value.real());
          update.values[0] += power;
          for (std::size_t region = 0; region < 4; ++region)
            update.values[region + 1] += weights(region, index) * power;
        },
        Kokkos::Sum<MonitorSums>(chunk));
    sums += chunk;
  }
  double removed = 0.0;
  if (problem_.split_step_absorber) {
    execution_space_.fence("solprop_native monitor reading");
    removed =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, monitor_removed_)();
    Kokkos::deep_copy(execution_space_, monitor_removed_, 0.0);
  }
  execution_space_.fence("solprop_native monitor reading");
  return {sums.values[0], sums.values[1], sums.values[2],
          sums.values[3], sums.values[4], removed};
}

// The split-step temporal absorber on a separable geometry: the rate is the same at
// every node, so it acts on the temporal rows (free space's nodes, the planar kind's
// y rows of slab amplitudes) between the two transverse transforms.
void GeneralWorkspace::separable_split_step(const double dz,
                                            const ComplexView2& advanced) {
  const auto scratch = transverse_;
  const auto coarse = coarse_spectrum_;
  const auto active = active_mask_;
  const auto n_pol = problem_.n_pol_state;
  const auto layout = transverse_layout();
  const auto n = problem_.n_omega;
  const double inverse_nodes = inverse_transverse_count();
  const double inverse_time = 1.0 / static_cast<double>(problem_.n_t);
  Kokkos::parallel_for(
      "solprop_native.free.split_input",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        const auto mode = row / n_pol;
        layout.cell(scratch, omega * n_pol + row % n_pol, mode) =
            inverse_nodes * advanced(row, omega);
      });
  transverse_inverse();
  for (std::size_t unit = 0; unit < transverse_units();
       unit += transverse_units_per_chunk()) {
    const auto chunk = transverse_chunk(unit);
    const auto first = chunk.first_row;
    const auto count = chunk.rows;
    Kokkos::deep_copy(execution_space_, coarse, Complex{});
    Kokkos::parallel_for(
        "solprop_native.free.split_temporal_input",
        Range2(execution_space_, {0, 0}, {count * n_pol, n}),
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
          const auto mode = first + row / n_pol;
          coarse(row, omega) =
              inverse_time * layout.cell(scratch, omega * n_pol + row % n_pol, mode);
        });
    // The rows are real-space rows here: by Parseval their power is the stored
    // columns' over the transverse count, which the monitor's measure is in.
    split_columns(dz, count * n_pol, 1.0 / inverse_nodes);
    Kokkos::parallel_for(
        "solprop_native.free.split_temporal_output",
        Range2(execution_space_, {0, 0}, {count * n_pol, n}),
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
          const auto mode = first + row / n_pol;
          layout.cell(scratch, omega * n_pol + row % n_pol, mode) = coarse(row, omega);
        });
  }
  transverse_forward();
  Kokkos::parallel_for(
      "solprop_native.free.split_output",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        const auto mode = row / n_pol;
        advanced(row, omega) =
            active(omega) != 0U
                ? layout.cell(scratch, omega * n_pol + row % n_pol, mode)
                : Complex{};
      });
}

void GeneralWorkspace::split_step(const double dz, const ComplexView2& advanced) {
  if (problem_.separable_geometry()) {
    separable_split_step(dz, advanced);
    return;
  }
  const auto coarse = coarse_spectrum_;
  const double inverse_n = 1.0 / static_cast<double>(problem_.n_t);
  Kokkos::parallel_for(
      "solprop_native.absorber.split_pad",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        coarse(column, index) = inverse_n * advanced(column, index);
      });
  split_columns(dz);
  const auto active = active_mask_;
  Kokkos::parallel_for(
      "solprop_native.absorber.split_commit",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        advanced(column, index) =
            active(index) != 0U ? coarse(column, index) : Complex{};
      });
}

std::vector<Complex> GeneralWorkspace::field_to_host() const {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  execution_space_.fence("solprop_native field_to_host");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, current_);
  std::vector<Complex> result(problem_.n_columns() * problem_.n_omega);
  for (std::size_t column = 0; column < problem_.n_columns(); ++column)
    for (std::size_t index = 0; index < problem_.n_omega; ++index)
      result[column * problem_.n_omega + index] = host(column, index);
  return result;
}

std::vector<Complex> GeneralWorkspace::stage_to_host(const std::size_t stage) const {
  require_open();
  if (!stages_ready_)
    throw std::logic_error("no trial stages have been evaluated");
  if (stage >= problem_.n_stages)
    throw std::out_of_range("stage index is outside the tableau");
  execution_space_.fence("solprop_native stage_to_host");
  const auto selected = Kokkos::subview(stages_, stage, Kokkos::ALL, Kokkos::ALL);
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, selected);
  std::vector<Complex> result(problem_.n_columns() * problem_.n_omega);
  for (std::size_t column = 0; column < problem_.n_columns(); ++column)
    for (std::size_t index = 0; index < problem_.n_omega; ++index)
      result[column * problem_.n_omega + index] = host(column, index);
  return result;
}

std::vector<Complex> GeneralWorkspace::projected_to_host(const bool coarse) {
  require_open();
  if (!stages_ready_)
    throw std::logic_error("no trial stages have been evaluated");
  if (coarse && projected_coarse_spectrum_.extent(0) == 0)
    throw std::logic_error("this quadrature has no embedded rule");
  if (coarse && dispersive_)
    dispersive_->finish_coarse();
  if (coarse)
    finish_quadrature();
  execution_space_.fence("solprop_native projected_to_host");
  const auto host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, coarse ? projected_coarse_spectrum_ : projected_spectrum_);
  return {host.data(), host.data() + host.size()};
}

std::map<std::string, bool> GeneralWorkspace::dispersive_path() const {
  if (separable_dispersive_)
    return {{"alone", separable_dispersive_->alone()},
            {"in_place", separable_dispersive_->in_place()},
            {"borrowed", separable_dispersive_->borrowed()}};
  // The general route: alone when no regional tensor or delayed response runs.
  if (general_ && general_->dispersive() != nullptr)
    return {{"alone", !general_->regional()}, {"in_place", false}, {"borrowed", false}};
  if (!dispersive_)
    return {};
  return {{"alone", dispersive_->alone()},
          {"in_place", dispersive_->in_place()},
          {"borrowed", dispersive_->borrowed()}};
}

std::map<std::string, std::size_t> GeneralWorkspace::memory_ledger() const {
  auto ledger =
      general_ ? general_->memory_ledger() : GeneralModalSource::empty_ledger();
  ledger.merge(flow_ ? flow_->memory_ledger() : CoupledFlowAction::empty_ledger());
  ledger.merge(dispersive_             ? dispersive_->memory_ledger()
               : separable_dispersive_ ? separable_dispersive_->memory_ledger()
               : general_ && general_->dispersive() != nullptr
                   ? general_->dispersive()->memory_ledger()
                   : DispersiveResponse::empty_ledger());
  ledger.merge(base_memory_ledger());
  return ledger;
}

std::map<std::string, std::size_t> GeneralWorkspace::base_memory_ledger() const {
  const auto complex_bytes = sizeof(Complex);
  const auto real_bytes = sizeof(double);
  const auto byte_bytes = sizeof(unsigned char);
  const auto index_bytes = sizeof(std::size_t);
  const auto field = problem_.n_columns() * problem_.n_omega * complex_bytes;
  const auto modal =
      modal_real_.size() * real_bytes + modal_complex_.size() * complex_bytes;
  // Aliased views (free space, SEP 28) own no storage of their own.
  const auto node = separable_aliased_ ? 0
                                       : node_real_.size() * real_bytes +
                                             node_complex_.size() * complex_bytes;
  const auto polarisation = separable_aliased_
                                ? 0
                                : polarisation_real_.size() * real_bytes +
                                      polarisation_complex_.size() * complex_bytes;
  const auto projected =
      projected_real_.size() * real_bytes + projected_complex_.size() * complex_bytes;
  return {
      {"held_current_host_derivative",
       problem_.time_window_derivative.capacity() * real_bytes},
      {"held_current", held_current_.size() * real_bytes},
      {"held_current_gj", held_streams_[0].bytes()},
      {"held_current_edge", held_streams_[1].bytes()},
      {"held_current_nonplasma", held_streams_[2].bytes()},
      {"held_current_projection", held_projection_.size() * real_bytes},
      {"held_current_radial_spectra",
       (held_nodal_spectrum_.size() + held_projected_spectrum_.size()) * complex_bytes},
      {"held_current_tables",
       (held_derivative_.size() + held_inverse_.size()) * real_bytes},
      {"envelope_drude_stream", envelope_streams_[0].bytes()},
      {"envelope_loss_stream", envelope_streams_[1].bytes()},
      {"envelope_cycles",
       (envelope_rule_nodes_.size() + envelope_rule_weights_.size() +
        envelope_moments_.size() + envelope_rates_.size() + envelope_exposure_.size()) *
           real_bytes},
      {"envelope_factors", envelope_drude_factor_.size() * real_bytes +
                               envelope_loss_factor_.size() * complex_bytes},
      {"envelope_status", (envelope_failures_.size() + envelope_status_.size() +
                           envelope_host_status_.size()) *
                              index_bytes},
      {"radial_blas_buffers",
       (radial_blas_input_.size() + radial_blas_output_.size()) * real_bytes},
      {"state", field},
      {"advanced", advanced_.size() * complex_bytes},
      {"stages", stages_.size() * complex_bytes},
      {"accumulator_solution", solution_.size() * complex_bytes},
      {"accumulator_error", error_.size() * complex_bytes},
      {"weighted_residual", residual_.size() * complex_bytes},
      {"recycled_stage", recycled_.size() * complex_bytes},
      {"noise", problem_.modified_noise ? field : 0},
      // The Runge-Kutta march's whole-step propagator, or the low-storage march's
      // table or step pair (solprop's `propagators` row).
      {"propagators",
       (whole_propagator_.size() + low_storage_flows_.size()) * complex_bytes},
      {"register_accumulator", register_accumulator_.size() * complex_bytes},
      {"register_stage", register_stage_.size() * complex_bytes},
      {"held_plane", held_plane_.size() * complex_bytes},
      {"stage_phase", stage_phase_.size() * complex_bytes},
      {"stage_inverse", stage_inverse_.size() * complex_bytes},
      {"fine_spectrum", fine_spectrum_.size() * complex_bytes},
      {"fine_gaps", (fine_gaps_.size() + separable_gaps_.size()) * index_bytes},
      {"modal_time", modal},
      {"node_field", node},
      {"node_polarisation", polarisation},
      {"projected", projected},
      {"projected_coarse", problem_.projection_coarse_matrix.empty() ? 0 : projected},
      {"projected_spectrum", projected_spectrum_.size() * complex_bytes},
      {"projected_coarse_spectrum", projected_coarse_spectrum_.size() * complex_bytes},
      {"absorber_spectrum", absorber_spectrum_.size() * complex_bytes},
      {"absorber_time",
       absorber_complex_.size() * complex_bytes + absorber_real_.size() * real_bytes},
      {"split_absorber_spectrum", coarse_spectrum_.size() * complex_bytes},
      {"split_absorber_time", coarse_time_complex_.size() * complex_bytes +
                                  coarse_time_real_.size() * real_bytes},
      {"transform_matrices",
       (synthesis_.size() + projection_.size() + projection_coarse_.size()) *
           real_bytes},
      // The native problem owns host copies as well as the execution-space views.
      // Count retained vector capacity, including construction-only coefficients.
      {"host_problem_arrays",
       (problem_.envelope_loss_factor.capacity() + problem_.carrier.capacity() +
        problem_.general_synthesis.capacity() + problem_.general_projection.capacity() +
        problem_.general_frequency_factor.capacity() +
        problem_.general_delayed_spectrum.capacity() + problem_.phi_terms.capacity() +
        problem_.prefactor_terms.capacity() + problem_.spatial_on_axis.capacity() +
        problem_.spatial_k_squared.capacity() +
        problem_.spatial_k_squared_low.capacity() +
        problem_.spatial_numerator.capacity() + problem_.spatial_clamped.capacity() +
        problem_.spatial_kappa_sq.capacity()) *
               complex_bytes +
           (problem_.envelope_rule_nodes.capacity() +
            problem_.envelope_rule_weights.capacity() +
            problem_.envelope_drude_factor.capacity() +
            problem_.omega_window.capacity() + problem_.omega.capacity() +
            problem_.power_weight.capacity() + problem_.time_window.capacity() +
            problem_.absorber_fine.capacity() + problem_.absorber_coarse.capacity() +
            problem_.spatial_absorber_matrix.capacity() +
            problem_.spatial_spectral_weights.capacity() +
            problem_.spatial_kperp_sq.capacity() +
            problem_.spatial_kperp_sq_low.capacity() +
            problem_.spatial_frame.capacity() +
            problem_.spatial_spectral_rate.capacity() +
            problem_.spatial_absorber_rate.capacity() +
            problem_.spatial_kperp_rate.capacity() + problem_.spatial_k_x.capacity() +
            problem_.spatial_k_y.capacity() + problem_.synthesis_matrix.capacity() +
            problem_.projection_matrix.capacity() +
            problem_.projection_coarse_matrix.capacity() +
            problem_.tableau_matrix.capacity() + problem_.tableau_nodes.capacity() +
            problem_.solution_weights.capacity() + problem_.error_weights.capacity() +
            problem_.raman_undamped.capacity() +
            problem_.raman_tensor_weights.capacity() +
            problem_.rate_coefficients.capacity() +
            problem_.channel_parameters.capacity() +
            problem_.channel_coefficients.capacity() +
            problem_.raman_row_component.capacity() +
            problem_.raman_component_fraction.capacity() +
            problem_.general_tensor.capacity() +
            problem_.general_delayed_tensor.capacity()) *
               real_bytes +
           problem_.general_node_response.capacity() * index_bytes +
           problem_.crop_index.capacity() * index_bytes +
           (problem_.active_mask.capacity() + problem_.norm_mask.capacity()) *
               byte_bytes},
      {"scaled_transform_matrices",
       (scaled_synthesis_.size() + scaled_projection_.size() +
        scaled_projection_coarse_.size() + scaled_projection_stacked_.size()) *
               real_bytes +
           (scaled_synthesis_complex_.size() + scaled_projection_complex_.size() +
            scaled_projection_coarse_complex_.size() +
            scaled_projection_stacked_complex_.size()) *
               complex_bytes},
      {"stacked_projection", projected_stacked_real_.size() * real_bytes +
                                 projected_stacked_complex_.size() * complex_bytes},
      {"chunk_transform_buffers",
       (chunk_synthesis_.size() + tail_synthesis_.size() + chunk_projection_.size() +
        tail_projection_.size() + chunk_projection_stacked_.size() +
        tail_projection_stacked_.size()) *
               real_bytes +
           (chunk_synthesis_complex_.size() + tail_synthesis_complex_.size() +
            chunk_projection_complex_.size() + tail_projection_complex_.size() +
            chunk_projection_stacked_complex_.size() +
            tail_projection_stacked_complex_.size()) *
               complex_bytes},
      {"transverse_scratch", transverse_.size() * complex_bytes},
      {"spatial_coefficients",
       (spatial_.square.size() + spatial_.square_low.size() +
        spatial_.numerator.size() + spatial_.clamped.size() + spatial_.kappa.size()) *
               complex_bytes +
           (spatial_.transverse.size() + spatial_.transverse_low.size() +
            spatial_.frame.size() + spatial_.spectral_rate.size() +
            spatial_.absorber.size() + spatial_.transverse_rate.size()) *
               real_bytes},
      {"adams_history",
       (adams_store_.size() + adams_slopes_.size() + adams_moments_.size() +
        adams_source_.size() + adams_propagator_.size() + adams_exponent_.size() +
        adams_saved_.size() + adams_held_.size() + adams_dense_start_.size() +
        adams_dense_end_.size() + adams_dense_exponent_.size() + adams_states_.size() +
        adams_rates_.size() + adams_carried_.size() + adams_dense_terms_.size() +
        adams_lab_.size() + adams_endpoint_.size() + adams_endpoint_rate_.size() +
        adams_block_.size() + adams_applied_.size() + adams_end_applied_.size()) *
               complex_bytes +
           (adams_divisor_.size() + adams_rate_weights_.size() +
            adams_end_weights_.size() + adams_scale_weights_.size() +
            (adams_magnitude_sum_.data() != nullptr ? 1 : 0)) *
               real_bytes},
      {"fixed_phi_terms", phi_terms_.size() * complex_bytes},
      {"fixed_prefactor_terms", prefactor_terms_.size() * complex_bytes},
      {"z_tables", phi_table_.bytes() + prefactor_table_.bytes() + rate_table_.bytes()},
      // solprop SEP 34: a radial run's plane-measure squares and its material table
      // (a separable run's alias its operator's), and the magnetic closure's
      // transverse volume of a separable run's local flux statistics.
      {"flux_coefficients",
       problem_.separable_geometry()
           ? 0
           : (flux_.square.size() + flux_.square_low.size()) * complex_bytes +
                 (flux_.transverse.size() + flux_.transverse_low.size()) * real_bytes},
      {"flux_table", flux_table_.bytes()},
      {"flux_closure_transverse", flux_transverse_.size() * complex_bytes},
      {"tableau", tableau_.size() * real_bytes},
      {"stage_scalars",
       (stage_phi_delta_.size() + stage_phi_at_.size() + stage_prefactor_.size() +
        stage_density_.size() + stage_taper_.size() + stage_preionisation_.size()) *
           real_bytes},
      {"crop_index", crop_index_.size() * index_bytes},
      {"masks", (active_mask_.size() + norm_mask_.size()) * byte_bytes},
      {"windows", (omega_window_.size() + time_window_.size()) * real_bytes},
      {"spatial_statistics_geometry", spatial_spectral_weights_.size() * real_bytes +
                                          spatial_on_axis_.size() * complex_bytes},
      {"spatial_statistics_partials", spatial_statistics_partials_.size() * real_bytes},
      {"far_field_partials", far_field_ ? far_field_->partial_bytes() : 0},
      {"far_field_tables", far_field_ ? far_field_->table_bytes() : 0},
      {"statistics_frequency_axis", omega_.size() * real_bytes},
      {"carrier", carrier_.size() * complex_bytes},
      {"node_raman_drive", node_raman_drive_.size() * real_bytes},
      {"node_raman_spectrum", node_raman_spectrum_.size() * complex_bytes},
      {"node_raman_convolved", node_raman_convolved_.size() * real_bytes},
      {"raman_undamped", raman_undamped_.size() * real_bytes},
      {"raman_tensor_weights", raman_tensor_weights_.size() * real_bytes},
      {"raman_kernel_time", raman_kernel_time_.size() * real_bytes},
      {"raman_kernel_spectrum", raman_kernel_spectrum_.size() * complex_bytes},
      {"stage_raman_rates", stage_raman_rates_.size() * real_bytes},
      {"absorber_rates",
       (absorber_fine_.size() + absorber_coarse_.size()) * real_bytes},
      {"spatial_absorber_matrix", spatial_absorber_real_.size() * real_bytes +
                                      spatial_absorber_complex_.size() * complex_bytes},
      {"plasma_rate_table", rate_coefficients_.size() * real_bytes},
      {"channel_parameters", channel_parameters_.size() * real_bytes},
      {"channel_rate_tables", channel_coefficients_.size() * real_bytes},
      {"channel_scratch", channel_scratch_.size() * real_bytes},
      {"material_stage_values",
       (stage_kerr_density_.size() + stage_plasma_density_.size() +
        stage_raman_density_.size() + stage_channel_density_.size() +
        stage_channel_preionisation_.size()) *
           real_bytes},
      {"raman_component_values",
       (stage_raman_densities_.size() + raman_row_component_.size() +
        raman_component_fraction_.size() + cached_raman_densities_.capacity()) *
           real_bytes},
      {"plasma_stash", (stash_candidate_.size() + stash_current_.size()) * real_bytes},
      {"plasma_statistics_scratch", plasma_statistics_scratch_.size() * real_bytes},
      {"plasma_integration_diagnostics",
       (plasma_taps_.size() + plasma_diagnostics_.size()) * real_bytes},
      {"plasma_device_scan",
       (plasma_rate_.size() + plasma_exposure_.size() + plasma_fraction_.size() +
        plasma_current_integrand_.size() + plasma_free_current_.size() +
        plasma_outer_current_.size() + band_tile_totals_.size() +
        plasma_tile_totals_.size()) *
           real_bytes},
      {"error_partials", 3 * problem_.n_columns() * reduction_slots_ * real_bytes},
      {"quadrature_partials", problem_.projection_coarse_matrix.empty()
                                  ? 0
                                  : 2 * problem_.n_columns() * real_bytes},
      {"weighted_control",
       (weighted_scale_.size() + weighted_partials_.size()) * real_bytes},
      {"statistics_spectrum", statistics_spectrum_.size() * complex_bytes},
      {"statistics_time", statistics_time_.size() * complex_bytes},
      {"statistics_weighted_time", statistics_weighted_time_.size() * complex_bytes},
      {"statistics_power_weight", statistics_power_weight_.size() * sizeof(double)},
      {"statistics_power", statistics_power_.size() * real_bytes},
      // The boundary monitor's weights and accumulators, once a monitor is bound.
      {"monitor_weights",
       (monitor_weights_.size() + monitor_column_weights_.size() +
        monitor_share_.size() + monitor_removed_.size() + monitor_split_sum_.size()) *
           real_bytes},
      {"statistics_flux_magnitude", statistics_magnitude_.size() * real_bytes},
      {"statistics_outputs", (statistics_total_power_.size() +
                              statistics_on_axis_.size() + statistics_packed_.size()) *
                                 real_bytes},
      {"fft_plan_heap_growth", plan_heap_growth_},
      {"fft_plan_device_growth", plan_device_growth_},
  };
}

void GeneralWorkspace::close() noexcept {
  if (closed_)
    return;
  execution_space_.fence();
  if (general_)
    general_->close();
  general_.reset();
  if (flow_)
    flow_->close();
  flow_.reset();
  if (dispersive_)
    dispersive_->close();
  dispersive_.reset();
  if (far_field_)
    far_field_->close();
  far_field_.reset();
  if (separable_dispersive_)
    separable_dispersive_->close();
  separable_dispersive_.reset();
  for (auto& stream : held_streams_) {
    stream.forward.reset();
    stream.coarse_forward.reset();
    stream = {};
  }
  held_current_ = {};
  held_projection_ = {};
  held_nodal_spectrum_ = {};
  held_projected_spectrum_ = {};
  held_derivative_ = {};
  held_inverse_ = {};
  for (auto& stream : envelope_streams_)
    stream = {};
  envelope_rule_nodes_ = {};
  envelope_rule_weights_ = {};
  envelope_moments_ = {};
  envelope_rates_ = {};
  envelope_exposure_ = {};
  envelope_drude_factor_ = {};
  envelope_loss_factor_ = {};
  envelope_failures_ = {};
  envelope_status_ = {};
  envelope_host_status_ = {};
  radial_blas_input_ = {};
  radial_blas_output_ = {};
  transverse_forward_.reset();
  transverse_inverse_.reset();
  slab_forward_.reset();
  slab_inverse_.reset();
  transverse_ = {};
  spatial_ = {};
  flux_ = {};
  flux_table_ = {};
  flux_transverse_ = {};
  statistics_magnitude_ = {};
  temporal_tail_inverse_real_.clear();
  temporal_tail_inverse_complex_.clear();
  temporal_tail_forward_real_.clear();
  temporal_tail_forward_complex_.clear();
  coarse_tail_inverse_real_.clear();
  coarse_tail_inverse_complex_.clear();
  coarse_tail_forward_real_.clear();
  coarse_tail_forward_complex_.clear();
  inverse_real_plan_.reset();
  inverse_complex_plan_.reset();
  forward_real_plan_.reset();
  forward_complex_plan_.reset();
  forward_coarse_real_plan_.reset();
  forward_coarse_complex_plan_.reset();
  statistics_inverse_plan_.reset();
  absorber_real_plan_.reset();
  absorber_complex_plan_.reset();
  coarse_inverse_real_plan_.reset();
  coarse_inverse_complex_plan_.reset();
  coarse_forward_real_plan_.reset();
  coarse_forward_complex_plan_.reset();
  raman_drive_forward_plan_.reset();
  raman_convolution_inverse_plan_.reset();
  raman_tail_forward_plan_.reset();
  raman_tail_inverse_plan_.reset();
  raman_tail_drive_ = {};
  raman_tail_spectrum_ = {};
  raman_tail_convolved_ = {};
  raman_kernel_forward_plan_.reset();
  current_ = {};
  register_accumulator_ = {};
  register_stage_ = {};
  held_plane_ = {};
  low_storage_flows_ = {};
  held_ready_ = false;
  advanced_ = {};
  stages_ = {};
  solution_ = {};
  error_ = {};
  residual_ = {};
  recycled_ = {};
  noise_ = {};
  whole_propagator_ = {};
  stage_phase_ = {};
  stage_inverse_ = {};
  adams_store_ = {};
  adams_slopes_ = {};
  adams_moments_ = {};
  adams_source_ = {};
  adams_propagator_ = {};
  adams_exponent_ = {};
  adams_saved_ = {};
  adams_held_ = {};
  adams_dense_start_ = {};
  adams_dense_end_ = {};
  adams_dense_exponent_ = {};
  adams_divisor_ = {};
  adams_states_ = {};
  adams_rates_ = {};
  adams_carried_ = {};
  adams_dense_terms_ = {};
  adams_lab_ = {};
  adams_endpoint_ = {};
  adams_endpoint_rate_ = {};
  adams_rate_weights_ = {};
  adams_block_ = {};
  adams_applied_ = {};
  adams_end_applied_ = {};
  adams_end_weights_ = {};
  fine_spectrum_ = {};
  projected_spectrum_ = {};
  projected_coarse_spectrum_ = {};
  absorber_spectrum_ = {};
  statistics_spectrum_ = {};
  statistics_time_ = {};
  statistics_weighted_time_ = {};
  statistics_power_weight_ = {};
  modal_complex_ = {};
  node_complex_ = {};
  polarisation_complex_ = {};
  projected_complex_ = {};
  projected_coarse_complex_ = {};
  absorber_complex_ = {};
  coarse_spectrum_ = {};
  coarse_time_complex_ = {};
  scaled_synthesis_complex_ = {};
  scaled_projection_complex_ = {};
  scaled_projection_coarse_complex_ = {};
  scaled_projection_stacked_complex_ = {};
  projected_stacked_complex_ = {};
  chunk_synthesis_complex_ = {};
  tail_synthesis_complex_ = {};
  chunk_projection_complex_ = {};
  tail_projection_complex_ = {};
  chunk_projection_stacked_complex_ = {};
  tail_projection_stacked_complex_ = {};
  node_raman_spectrum_ = {};
  raman_kernel_spectrum_ = {};
  modal_real_ = {};
  node_real_ = {};
  polarisation_real_ = {};
  projected_real_ = {};
  projected_coarse_real_ = {};
  projected_stacked_real_ = {};
  absorber_real_ = {};
  coarse_time_real_ = {};
  node_raman_drive_ = {};
  node_raman_convolved_ = {};
  raman_undamped_ = {};
  raman_tensor_weights_ = {};
  raman_kernel_time_ = {};
  stage_raman_rates_ = {};
  synthesis_ = {};
  projection_ = {};
  projection_coarse_ = {};
  scaled_synthesis_ = {};
  scaled_projection_ = {};
  scaled_projection_coarse_ = {};
  scaled_projection_stacked_ = {};
  chunk_synthesis_ = {};
  tail_synthesis_ = {};
  chunk_projection_ = {};
  tail_projection_ = {};
  chunk_projection_stacked_ = {};
  tail_projection_stacked_ = {};
  partials_ = {};
  quadrature_partials_ = {};
  weighted_scale_ = {};
  weighted_partials_ = {};
  stash_candidate_ = {};
  stash_current_ = {};
  statistics_power_ = {};
  plasma_statistics_scratch_ = {};
  plasma_rate_ = {};
  plasma_exposure_ = {};
  plasma_fraction_ = {};
  plasma_current_integrand_ = {};
  plasma_free_current_ = {};
  plasma_outer_current_ = {};
  band_tile_totals_ = {};
  plasma_tile_totals_ = {};
  spatial_spectral_weights_ = {};
  spatial_on_axis_ = {};
  spatial_statistics_partials_ = {};
  statistics_total_power_ = {};
  statistics_on_axis_ = {};
  statistics_packed_ = {};
  monitor_bound_ = false;
  monitor_weights_ = {};
  monitor_column_weights_ = {};
  monitor_share_ = {};
  monitor_removed_ = {};
  monitor_split_sum_ = {};
  tableau_ = {};
  omega_window_ = {};
  omega_ = {};
  time_window_ = {};
  absorber_fine_ = {};
  absorber_coarse_ = {};
  spatial_absorber_real_ = {};
  spatial_absorber_complex_ = {};
  stage_phi_delta_ = {};
  stage_scalars_ = {};
  stage_staging_.clear();
  stage_phi_at_ = {};
  stage_prefactor_ = {};
  stage_density_ = {};
  stage_taper_ = {};
  stage_preionisation_ = {};
  rate_coefficients_ = {};
  channel_parameters_ = {};
  channel_coefficients_ = {};
  channel_scratch_ = {};
  plasma_taps_ = {};
  plasma_diagnostics_ = {};
  stage_channel_density_ = {};
  stage_channel_preionisation_ = {};
  stage_kerr_density_ = {};
  stage_plasma_density_ = {};
  stage_raman_density_ = {};
  stage_raman_densities_ = {};
  raman_row_component_ = {};
  raman_component_fraction_ = {};
  cached_raman_densities_.clear();
  carrier_ = {};
  active_mask_ = {};
  norm_mask_ = {};
  crop_index_ = {};
  fine_gaps_ = {};
  separable_gaps_ = {};
  // FFT plans must release their execution instances before runtime shutdown.
  raman_tail_execution_space_ = ExecutionSpace();
  raman_execution_space_ = ExecutionSpace();
  raman_kernel_execution_space_ = ExecutionSpace();
  temporal_execution_space_ = ExecutionSpace();
  flow_execution_space_ = ExecutionSpace();
  absorber_execution_space_ = ExecutionSpace();
  raman_multiply_space_ = ExecutionSpace();
  raman_tail_multiply_space_ = ExecutionSpace();
  raman_rebuild_space_ = ExecutionSpace();
  raman_scale_space_ = ExecutionSpace();
  response_execution_space_ = ExecutionSpace();
  response_tail_execution_space_ = ExecutionSpace();
  transform_execution_space_ = ExecutionSpace();
  transform_tail_execution_space_ = ExecutionSpace();
  matrix_copy_space_ = ExecutionSpace();
  tail_matrix_copy_space_ = ExecutionSpace();
  stacked_matrix_copy_space_ = ExecutionSpace();
  tail_stacked_matrix_copy_space_ = ExecutionSpace();
  node_spectral_space_ = ExecutionSpace();
  kernel_pool_.clear();
  fft_pool_.clear();
  raman_pool_.clear();
  raman_tail_pool_.clear();
  fft_execution_space_ = ExecutionSpace();
  execution_space_ = ExecutionSpace();
  closed_ = true;
  initialized_ = false;
  table_ready_ = false;
  has_recycled_ = false;
  pending_fsal_ = false;
  coarse_pending_ = false;
  stash_is_current_ = false;
  raman_kernel_ready_ = false;
}

GeneralState::GeneralState(Problem problem)
    : workspace_(std::make_unique<GeneralWorkspace>(std::move(problem))) {}
GeneralState::~GeneralState() { close(); }
void GeneralState::set_initial(const std::vector<Complex>& field,
                               const std::vector<Complex>& noise) {
  workspace_->set_initial(field, noise);
}
std::vector<double> GeneralState::trial_step(const double dz,
                                             const TrialScalars& scalars) {
  return workspace_->trial_step(dz, scalars);
}
void GeneralState::prepare_trial(const double dz, const TrialScalars& scalars) {
  workspace_->prepare_trial(dz, scalars);
}
std::vector<Complex> GeneralState::adams_source(const std::vector<Complex>& input,
                                                const TrialScalars& scalars) {
  return workspace_->adams_source(input, scalars);
}
void GeneralState::adopt(const std::vector<Complex>& field, const double dz) {
  workspace_->adopt(field, dz);
}
double GeneralState::low_storage_step(const double dz, const TrialScalars& scalars,
                                      const bool rebuild_table) {
  return workspace_->low_storage_step(dz, scalars, rebuild_table);
}
void GeneralState::hold_plane() { workspace_->hold_plane(); }
std::vector<Complex> GeneralState::held_to_host() const {
  return workspace_->held_to_host();
}
void GeneralState::adams_allocate(const std::size_t slots, const bool compensated,
                                  const bool interpolate) {
  workspace_->adams_allocate(slots, compensated, interpolate);
}
void GeneralState::adams_compensation_start(const std::size_t slot,
                                            const std::vector<double>& rate) {
  workspace_->adams_compensation_start(slot, rate);
}
void GeneralState::adams_start(const std::size_t slot, const TrialScalars& scalars) {
  workspace_->adams_start(slot, scalars);
}
double GeneralState::adams_magnitude() { return workspace_->adams_magnitude(); }
void GeneralState::adams_scale(const double rtol, const double atol,
                               const std::vector<double>& weights) {
  workspace_->adams_scale(rtol, atol, weights);
}
std::vector<double> GeneralState::adams_fused_trial(const double dz,
                                                    const AdamsWeights& weights,
                                                    const std::vector<double>& divisor,
                                                    const TrialScalars& scalars) {
  return workspace_->adams_fused_trial(dz, weights, divisor, scalars);
}
void GeneralState::adams_fused_accept(const double dz, const AdamsWeights& keep,
                                      const std::size_t free,
                                      const AdamsWeights& dense) {
  workspace_->adams_fused_accept(dz, keep, free, dense);
}
void GeneralState::adams_hold() { workspace_->adams_hold(); }
void GeneralState::adams_restore() { workspace_->adams_restore(); }
void GeneralState::table_upload(const TableKind kind, std::vector<double> edges,
                                const double lower, const double upper,
                                const std::complex<double>* coefficients,
                                const std::size_t n_intervals,
                                const std::size_t n_terms, const std::size_t n_values) {
  workspace_->table_upload(kind, std::move(edges), lower, upper, coefficients,
                           n_intervals, n_terms, n_values);
}
void GeneralState::table_evaluate(const std::vector<std::size_t>& slots,
                                  const std::vector<double>& positions) {
  workspace_->table_evaluate(slots, positions);
}
void GeneralState::table_rate(const std::size_t slot, const double z,
                              const bool before) {
  workspace_->table_rate(slot, z, before);
}
std::vector<Complex> GeneralState::table_terms_to_host(const TableKind kind,
                                                       const std::size_t slot) const {
  return workspace_->table_terms_to_host(kind, slot);
}
std::vector<Complex>
GeneralState::phase_sum_to_host(const std::vector<double>& weights) {
  return workspace_->phase_sum_to_host(weights);
}
std::vector<std::string> GeneralState::adams_row_names() const {
  return workspace_->adams_row_names();
}
std::array<std::size_t, 3>
GeneralState::adams_row_extents(const std::string& name) const {
  return workspace_->adams_row_extents(name);
}
std::vector<Complex> GeneralState::adams_rows_to_host(const std::string& name) const {
  return workspace_->adams_rows_to_host(name);
}
void GeneralState::adams_rows_to_host(const std::string& name,
                                      std::complex<double>* destination,
                                      const std::size_t size) const {
  workspace_->adams_rows_to_host(name, destination, size);
}
void GeneralState::adams_rows_from_host(const std::string& name,
                                        const std::vector<Complex>& values) {
  workspace_->adams_rows_from_host(name, values);
}
void GeneralState::adams_observe(const AdamsWeights& weights, const bool hermite,
                                 const TrialScalars& scalars) {
  workspace_->adams_observe(weights, hermite, scalars);
}
std::vector<double> GeneralState::adams_exponential_trial(
    const double dz, const ExponentialWeights& weights,
    const std::vector<double>& divisor, const TrialScalars& step,
    const TrialScalars& sample, const std::vector<double>& rate) {
  return workspace_->adams_exponential_trial(dz, weights, divisor, step, sample, rate);
}
void GeneralState::adams_exponential_accept(const double dz, const std::size_t free,
                                            const AdamsWeights& dense) {
  workspace_->adams_exponential_accept(dz, free, dense);
}
void GeneralState::adams_exponential_observe(const double xi, const double dz,
                                             const ExponentialWeights& weights) {
  workspace_->adams_exponential_observe(xi, dz, weights);
}
std::vector<double> GeneralState::weighted_error(const std::vector<double>& scale) {
  return workspace_->weighted_error(scale);
}
void GeneralState::set_flow_actions(const std::vector<Complex>& forward,
                                    const std::vector<Complex>& inverse,
                                    const std::vector<unsigned char>& identity) {
  workspace_->set_flow_actions(forward, inverse, identity);
}
std::vector<Complex> GeneralState::flow_actions_to_host(const bool inverse) const {
  return workspace_->flow_actions_to_host(inverse);
}
std::vector<double>
GeneralState::form_flow_actions(const std::size_t section, const double dz,
                                const std::vector<double>& fractions) {
  return workspace_->form_flow_actions(section, dz, fractions);
}
void GeneralState::prepare_raman(const std::vector<double>& rates,
                                 const std::vector<double>& densities) {
  workspace_->prepare_raman(rates, densities);
}
std::vector<double>
GeneralState::statistics(const std::uint64_t flags, const std::vector<double>& windows,
                         const std::vector<double>& on_axis, const double density,
                         const double taper, const double preionisation,
                         const double intensity_factor,
                         const std::vector<double>& channel_preionisation,
                         const std::vector<double>& power_weight) {
  return workspace_->statistics(flags, windows, on_axis, density, taper, preionisation,
                                intensity_factor, channel_preionisation, power_weight);
}
std::vector<double> GeneralState::spatial_statistics(const std::uint64_t flags,
                                                     const double intensity_factor,
                                                     const double z) {
  return workspace_->spatial_statistics(flags, intensity_factor, z);
}
std::vector<double> GeneralState::aperture_statistics(
    const std::uint64_t flags, const std::vector<double>& apertures,
    const int profile_code, const int exponent, const std::vector<double>& regions,
    const double intensity_factor) {
  return workspace_->aperture_statistics(flags, apertures, profile_code, exponent,
                                         regions, intensity_factor);
}
void GeneralState::finish(const double dz) { workspace_->finish(dz); }
void GeneralState::bind_monitor(const std::vector<double>& weights,
                                const std::vector<double>& column_weights) {
  workspace_->bind_monitor(weights, column_weights);
}
void GeneralState::unbind_monitor() noexcept {
  if (workspace_)
    workspace_->unbind_monitor();
}
std::vector<double> GeneralState::monitor_reading() {
  return workspace_->monitor_reading();
}
void GeneralState::promote_fsal() { workspace_->promote_fsal(); }
void GeneralState::clear_fsal() { workspace_->clear_fsal(); }
void GeneralState::set_absorber_quiet(const bool quiet) {
  workspace_->set_absorber_quiet(quiet);
}
double GeneralState::quadrature_error() const {
  return workspace_ ? workspace_->quadrature_error()
                    : Kokkos::Experimental::quiet_NaN_v<double>;
}
double GeneralState::quadrature_peak() {
  return workspace_ ? workspace_->quadrature_peak()
                    : Kokkos::Experimental::quiet_NaN_v<double>;
}
std::string GeneralState::synthesis_provider() const {
  return workspace_ ? workspace_->synthesis_provider() : "closed";
}
std::string GeneralState::projection_provider() const {
  return workspace_ ? workspace_->projection_provider() : "closed";
}
std::string GeneralState::gemm_shape_policy() const {
  return GeneralWorkspace::gemm_shape_policy();
}
std::size_t GeneralState::response_team_size() const noexcept {
  return workspace_ ? workspace_->response_team_size() : 0;
}
std::vector<Complex> GeneralState::field_to_host() const {
  return workspace_->field_to_host();
}
std::vector<Complex> GeneralState::stage_to_host(const std::size_t stage) const {
  return workspace_->stage_to_host(stage);
}
std::vector<Complex> GeneralState::projected_to_host(const bool coarse) {
  return workspace_->projected_to_host(coarse);
}
std::map<std::string, bool> GeneralState::dispersive_path() const {
  return workspace_->dispersive_path();
}
std::map<std::string, std::size_t> GeneralState::memory_ledger() const {
  return workspace_->memory_ledger();
}
CpuSizing GeneralState::execution_sizing() const {
  return workspace_->execution_sizing();
}
double GeneralState::planning_seconds() const noexcept {
  return workspace_ ? workspace_->planning_seconds() : 0.0;
}
double GeneralState::raman_planning_seconds() const noexcept {
  return workspace_ ? workspace_->raman_planning_seconds() : 0.0;
}
std::string GeneralState::raman_fft_provider() const {
  return workspace_ ? workspace_->raman_fft_provider() : "none";
}
std::map<std::string, int> GeneralState::raman_fft_tasks() const {
  return workspace_ ? workspace_->raman_fft_tasks() : std::map<std::string, int>{};
}
std::size_t GeneralState::raman_kernel_transforms() const noexcept {
  return workspace_ ? workspace_->raman_kernel_transforms() : 0;
}
bool GeneralState::fsal_ready() const noexcept {
  return workspace_ && workspace_->fsal_ready();
}
std::size_t GeneralState::rk_source_evaluations() const noexcept {
  return workspace_ ? workspace_->rk_source_evaluations() : 0;
}
bool GeneralState::closed() const noexcept {
  return !workspace_ || workspace_->closed();
}
void GeneralState::close() noexcept {
  if (workspace_)
    workspace_->close();
}

GeneralBatch::GeneralBatch(std::vector<std::shared_ptr<GeneralState>> members)
    : workspace_(std::make_unique<GeneralBatchWorkspace>(std::move(members))) {}
GeneralBatch::~GeneralBatch() { close(); }
std::vector<double> GeneralBatch::accept(const double dz) {
  return workspace_->accept(dz);
}
std::map<std::string, std::size_t> GeneralBatch::memory_ledger() const {
  return workspace_->memory_ledger();
}
CpuSizing GeneralBatch::execution_sizing() const {
  return workspace_->execution_sizing();
}
void GeneralBatch::close() noexcept { workspace_->close(); }

} // namespace solprop_native
