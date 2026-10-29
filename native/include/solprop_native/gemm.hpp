// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/launch.hpp"

#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace solprop_native {

enum class GemmProvider { custom, kokkosblas_native, kokkosblas_tpl, hybrid };

[[nodiscard]] constexpr GemmProvider compiled_gemm_provider() noexcept {
#if defined(SOLPROP_NATIVE_GEMM_PROVIDER_CUSTOM)
  return GemmProvider::custom;
#elif defined(SOLPROP_NATIVE_GEMM_PROVIDER_KOKKOSBLAS_NATIVE)
  return GemmProvider::kokkosblas_native;
#elif defined(SOLPROP_NATIVE_GEMM_PROVIDER_KOKKOSBLAS_TPL)
  return GemmProvider::kokkosblas_tpl;
#else
  return GemmProvider::hybrid;
#endif
}

[[nodiscard]] constexpr std::string_view
gemm_provider_name(const GemmProvider provider) noexcept {
  switch (provider) {
  case GemmProvider::custom:
    return "custom";
  case GemmProvider::kokkosblas_native:
    return "kokkosblas-native";
  case GemmProvider::kokkosblas_tpl:
    return "kokkosblas-tpl";
  case GemmProvider::hybrid:
    return "hybrid";
  }
  return "unknown";
}

// A host build without a vendor BLAS runs KokkosBlas's native real GEMM arithmetic
// through `native_order_gemm`: the same result to
// the bit, several times faster on the engine's skinny-by-wide transforms.
inline constexpr bool native_order_gemm_enabled =
#if defined(KOKKOSKERNELS_ENABLE_TPL_BLAS) ||                                          \
    defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS) ||                                        \
    defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS) || defined(KOKKOS_ENABLE_CUDA) ||        \
    defined(KOKKOS_ENABLE_HIP)
    false;
#else
    true;
#endif

[[nodiscard]] constexpr std::string_view selected_blas_provider_name() noexcept {
#if defined(KOKKOSKERNELS_ENABLE_TPL_BLAS) ||                                          \
    defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS) ||                                        \
    defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)
  return "kokkosblas-tpl";
#else
  return native_order_gemm_enabled ? "kokkosblas-native-order" : "kokkosblas-native";
#endif
}

template <class ExecutionSpace, class AView, class BView, class CView>
void custom_gemm(const ExecutionSpace& execution_space, const AView& a, const BView& b,
                 const CView& c, const typename CView::non_const_value_type beta) {
  using Value = CView::non_const_value_type;
  if (a.extent(1) != b.extent(0) || a.extent(0) != c.extent(0) ||
      b.extent(1) != c.extent(1)) {
    throw std::invalid_argument("GEMM view extents are inconsistent");
  }
  const std::size_t inner = a.extent(1);
  launch::parallel_for(
      "solprop_native.transform.custom_gemm",
      // The last index fastest, as the views store it (a device's default is the
      // first).
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2, Kokkos::Iterate::Right,
                                                         Kokkos::Iterate::Right>>(
          execution_space, {0, 0}, {c.extent(0), c.extent(1)}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t column) {
        Value value{};
        for (std::size_t index = 0; index < inner; ++index) {
          value += a(row, index) * b(index, column);
        }
        // A zero beta does not read C, as BLAS specifies: a stale non-finite C
        // stays out of the product.
        c(row, column) = beta == Value{} ? value : value + beta * c(row, column);
      });
}

template <class ExecutionSpace, class AView, class BView, class CView>
void kokkosblas_gemm(const ExecutionSpace& execution_space, const AView& a,
                     const BView& b, const CView& c,
                     const typename CView::non_const_value_type beta) {
  using Value = CView::non_const_value_type;
  KokkosBlas::gemm(execution_space, "N", "N", Value{1}, a, b, beta, c);
}

// KokkosKernels 5.2.0's native GEMM (`KokkosBlas3_gemm_spec.hpp`, `GEMMImpl`) forms
// every element of C from inner blocks of this depth for double operands: each block's
// products are an FMA chain from zero, the blocks' sums are added in order to a total
// that starts at zero, and C is `alpha * total` (beta zero) or
// `beta * C + alpha * total`. `native_order_gemm` follows that per-element
// arithmetic; a native test holds it to `KokkosBlas::gemm` within the GEMM rounding
// bound, `gamma_{k+1} (|beta c| + sum |a| |b|)` each (solprop SEP 25), not to its bits.
inline constexpr std::size_t kokkosblas_native_inner_block = 16;

// One work item owns a `Rows x Columns` tile of C and accumulates each 16-deep block
// in `Rows x Chunk` register tiles before adding it to the tile's total; the tile
// shapes are measured on the engine's skinny-by-wide transforms (solprop
// benchmarks.md section 35). None of them changes the per-element arithmetic.
template <std::size_t Rows, std::size_t Columns, std::size_t Chunk,
          class ExecutionSpace, class AView, class BView, class CView>
void native_order_gemm_tiles(const ExecutionSpace& execution_space, const AView& a,
                             const BView& b, const CView& c, const double beta) {
  constexpr std::size_t depth = kokkosblas_native_inner_block;
  const std::size_t rows = c.extent(0);
  const std::size_t columns = c.extent(1);
  const std::size_t inner = a.extent(1);
  const std::size_t row_tiles = (rows + Rows - 1) / Rows;
  const std::size_t column_tiles = (columns + Columns - 1) / Columns;
  launch::parallel_for(
      "solprop_native.transform.native_order_gemm",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, row_tiles * column_tiles),
      KOKKOS_LAMBDA(const std::size_t item) {
        const std::size_t first_row = (item / column_tiles) * Rows;
        const std::size_t first_column = (item % column_tiles) * Columns;
        const std::size_t row_count = rows - first_row < Rows ? rows - first_row : Rows;
        const std::size_t column_count =
            columns - first_column < Columns ? columns - first_column : Columns;
        double total[Rows][Columns];
        for (std::size_t row = 0; row < Rows; ++row)
          for (std::size_t column = 0; column < Columns; ++column)
            total[row][column] = 0.0;
        for (std::size_t block = 0; block < inner; block += depth) {
          const std::size_t end = inner - block < depth ? inner : block + depth;
          for (std::size_t offset = 0; offset < column_count; offset += Chunk) {
            const std::size_t first = first_column + offset;
            double partial[Rows][Chunk];
            for (std::size_t row = 0; row < Rows; ++row)
              for (std::size_t column = 0; column < Chunk; ++column)
                partial[row][column] = 0.0;
            if (row_count == Rows && column_count - offset >= Chunk) {
              for (std::size_t index = block; index < end; ++index)
                for (std::size_t row = 0; row < Rows; ++row) {
                  const double left = a(first_row + row, index);
                  for (std::size_t column = 0; column < Chunk; ++column)
                    partial[row][column] += left * b(index, first + column);
                }
            } else {
              const std::size_t width =
                  column_count - offset < Chunk ? column_count - offset : Chunk;
              for (std::size_t index = block; index < end; ++index)
                for (std::size_t row = 0; row < row_count; ++row) {
                  const double left = a(first_row + row, index);
                  for (std::size_t column = 0; column < width; ++column)
                    partial[row][column] += left * b(index, first + column);
                }
            }
            for (std::size_t row = 0; row < Rows; ++row)
              for (std::size_t column = 0; column < Chunk; ++column)
                total[row][offset + column] += partial[row][column];
          }
        }
        for (std::size_t row = 0; row < row_count; ++row) {
          for (std::size_t column = 0; column < column_count; ++column) {
            double& value = c(first_row + row, first_column + column);
            value = beta == 0.0 ? 1.0 * total[row][column]
                                : beta * value + 1.0 * total[row][column];
          }
        }
      });
}

template <class ExecutionSpace, class AView, class BView, class CView>
void native_order_gemm(const ExecutionSpace& execution_space, const AView& a,
                       const BView& b, const CView& c, const double beta) {
  static_assert(std::is_same_v<typename AView::non_const_value_type, double> &&
                    std::is_same_v<typename BView::non_const_value_type, double> &&
                    std::is_same_v<typename CView::non_const_value_type, double>,
                "the KokkosBlas-order kernel reproduces the real GEMM only");
  if (a.extent(1) != b.extent(0) || a.extent(0) != c.extent(0) ||
      b.extent(1) != c.extent(1)) {
    throw std::invalid_argument("GEMM view extents are inconsistent");
  }
  // Eight-row tiles read each row of B once for a stacked projection or a synthesis;
  // four-row tiles keep a short projection off the partial-tile path.
  if (c.extent(0) >= 8)
    native_order_gemm_tiles<8, 128, 4>(execution_space, a, b, c, beta);
  else
    native_order_gemm_tiles<4, 128, 8>(execution_space, a, b, c, beta);
}

// The engine's KokkosBlas call: vendor or device GEMM where the build has one, and
// otherwise KokkosBlas's own native arithmetic through the faster host kernel above.
template <class ExecutionSpace, class AView, class BView, class CView>
void engine_gemm(const ExecutionSpace& execution_space, const AView& a, const BView& b,
                 const CView& c, const typename CView::non_const_value_type beta) {
  if constexpr (native_order_gemm_enabled &&
                std::is_same_v<typename AView::non_const_value_type, double> &&
                std::is_same_v<typename BView::non_const_value_type, double> &&
                std::is_same_v<typename CView::non_const_value_type, double>) {
    native_order_gemm(execution_space, a, b, c, beta);
  } else {
    kokkosblas_gemm(execution_space, a, b, c, beta);
  }
}

template <class ExecutionSpace, class AView, class BView, class CView>
void selected_gemm(const ExecutionSpace& execution_space, const AView& a,
                   const BView& b, const CView& c,
                   const typename CView::non_const_value_type beta,
                   const bool tpl_eligible) {
  const auto provider = compiled_gemm_provider();
  if (provider == GemmProvider::custom ||
      (provider == GemmProvider::hybrid && !tpl_eligible)) {
    custom_gemm(execution_space, a, b, c, beta);
    return;
  }
  engine_gemm(execution_space, a, b, c, beta);
}

} // namespace solprop_native
