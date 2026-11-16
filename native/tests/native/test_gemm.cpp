// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Tests of the host GEMM kernels (`transforms/gemm.hpp`): the native-order kernel and
// the one-mode projection and synthesis against KokkosBlas within the GEMM rounding
// bound, their shape refusals, and the engine's choice of kernel.

#include "bounds.hpp"
#include "solprop_native/runtime/cpu_execution.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/transforms/gemm.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace native = solprop_native;
namespace test = solprop_native::test;

// The host kernel computes KokkosBlas's real GEMM, and both are held to the GEMM
// rounding bound (solprop SEP 25), not to each other's bits: each element of either is
// within `gamma_{k+1} (|beta c| + sum_k |a_ik b_kj|)` of the exact `beta c + a b`
// whatever order or contraction it sums in, so the two differ by at most twice that.
// Rows outside the written subview are a structural contract and stay exact.
// Shapes cover an inner length below, at and past one 16-deep block and several blocks,
// partial row and column tiles, both beta values the engine passes, and row subviews.
TEST(Gemm, NativeOrderKernelIsKokkosBlasWithinTheGemmBound) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // A device build never calls the kernel (`native_order_gemm_enabled` is false).
  GTEST_SKIP() << "the native-order kernel is the host build's";
#endif
  using RealView2 = Kokkos::View<double**, native::Layout, native::MemorySpace>;
  const native::ExecutionSpace execution;
  const std::array<std::array<std::size_t, 3>, 9> shapes{{{33, 3, 2048},
                                                          {3, 33, 2048},
                                                          {6, 33, 257},
                                                          {4, 528, 130},
                                                          {17, 16, 64},
                                                          {5, 17, 63},
                                                          {41, 48, 100},
                                                          {1, 1, 1},
                                                          {9, 2, 5}}};
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  const auto uniform = [&state] {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    return static_cast<double>(state >> 11U) * 0x1.0p-53 - 0.5;
  };
  for (const auto& [rows, inner, columns] : shapes) {
    for (const double beta : {0.0, 1.0}) {
      RealView2 left("left", rows + 2, inner);
      RealView2 right("right", inner, columns);
      RealView2 expected("expected", rows + 2, columns);
      RealView2 actual("actual", rows + 2, columns);
      const auto host_left = Kokkos::create_mirror_view(left);
      const auto host_right = Kokkos::create_mirror_view(right);
      const auto host_start = Kokkos::create_mirror_view(expected);
      for (std::size_t row = 0; row < rows + 2; ++row)
        for (std::size_t index = 0; index < inner; ++index)
          host_left(row, index) = uniform();
      for (std::size_t index = 0; index < inner; ++index)
        for (std::size_t column = 0; column < columns; ++column)
          host_right(index, column) = uniform();
      // On a host build the mirror aliases `expected`, which the GEMM overwrites, so
      // the starting values are kept apart for the bound.
      std::vector<double> start((rows + 2) * columns);
      for (std::size_t row = 0; row < rows + 2; ++row)
        for (std::size_t column = 0; column < columns; ++column) {
          host_start(row, column) = uniform();
          start[row * columns + column] = host_start(row, column);
        }
      Kokkos::deep_copy(left, host_left);
      Kokkos::deep_copy(right, host_right);
      Kokkos::deep_copy(expected, host_start);
      Kokkos::deep_copy(actual, host_start);
      const auto window = std::make_pair(std::size_t{1}, rows + 1);
      const auto a = Kokkos::subview(left, window, Kokkos::ALL());
      native::kokkosblas_gemm(execution, a, right,
                              Kokkos::subview(expected, window, Kokkos::ALL()), beta);
      native::native_order_gemm(execution, a, right,
                                Kokkos::subview(actual, window, Kokkos::ALL()), beta);
      execution.fence("native order GEMM test");
      const auto host_expected =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, expected);
      const auto host_actual =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, actual);
      const double gemm_gamma = test::higham_gamma(inner + 1);
      for (std::size_t row = 0; row < rows + 2; ++row)
        for (std::size_t column = 0; column < columns; ++column) {
          if (row == 0 || row == rows + 1) {
            // Outside the subview: never written, so exactly the start.
            ASSERT_EQ(host_actual(row, column), start[row * columns + column])
                << rows << 'x' << inner << 'x' << columns << " at " << row;
            ASSERT_EQ(host_expected(row, column), start[row * columns + column])
                << rows << 'x' << inner << 'x' << columns << " at " << row;
            continue;
          }
          double magnitude = beta * std::abs(start[row * columns + column]);
          for (std::size_t index = 0; index < inner; ++index)
            magnitude += std::abs(host_left(row, index) * host_right(index, column));
          ASSERT_LE(std::abs(host_actual(row, column) - host_expected(row, column)),
                    2.0 * gemm_gamma * magnitude)
              << rows << 'x' << inner << 'x' << columns << " beta " << beta << " at "
              << row << ',' << column;
        }
    }
  }
  RealView2 wrong("wrong", 2, 3);
  EXPECT_THROW(native::native_order_gemm(execution, wrong, wrong, wrong, 0.0),
               std::invalid_argument);
}

// The one-mode kernels sum each element over the nodes in order from zero, which a
// GEMM may do in any order: both lie within gamma_{k+1} sum |a| |b| of the exact
// product, so they differ by at most twice that. A synthesis element is one product.
TEST(Gemm, OneModeKernelsAreKokkosBlasWithinTheGemmBound) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  GTEST_SKIP() << "the one-mode kernels are the host vendor-BLAS build's";
#endif
  using RealView2 = Kokkos::View<double**, native::Layout, native::MemorySpace>;
  std::uint64_t state = 0x2545f4914f6cdd1dULL;
  const auto uniform = [&state] {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    return static_cast<double>(state >> 11U) * 0x1.0p-53 - 0.5;
  };
  const auto fill = [&uniform](const RealView2& view) {
    const auto host = Kokkos::create_mirror_view(view);
    for (std::size_t row = 0; row < view.extent(0); ++row)
      for (std::size_t column = 0; column < view.extent(1); ++column)
        host(row, column) = uniform();
    Kokkos::deep_copy(view, host);
  };
  const auto host = [](const RealView2& view) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  };
  for (const int threads : {1, 3, native::ExecutionSpace().concurrency()})
    for (const auto& [rows, inner, columns] : std::array<std::array<std::size_t, 3>, 5>{
             {{1, 33, 6400}, {2, 33, 6400}, {2, 33, 257}, {1, 2, 7}, {2, 5, 1}}}) {
      SCOPED_TRACE(std::to_string(threads) + " threads, " + std::to_string(rows) + 'x' +
                   std::to_string(inner) + 'x' + std::to_string(columns));
      const auto execution = native::execution_instance(threads);
      RealView2 weights("weights", rows, inner);
      RealView2 nodes("nodes", inner, columns);
      RealView2 expected("expected", rows, columns);
      RealView2 fine("fine", 1, columns);
      RealView2 coarse("coarse", 1, columns);
      fill(weights);
      fill(nodes);
      native::kokkosblas_gemm(execution, weights, nodes, expected, 0.0);
      native::one_mode_projection(execution, weights, nodes, fine, coarse);
      execution.fence("one-mode projection test");
      const auto a = host(weights);
      const auto b = host(nodes);
      const auto reference = host(expected);
      const auto first = host(fine);
      const auto second = host(coarse);
      const double gamma = test::higham_gamma(inner + 1);
      for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column) {
          double magnitude = 0.0;
          for (std::size_t node = 0; node < inner; ++node)
            magnitude += std::abs(a(row, node) * b(node, column));
          const double actual = row == 0 ? first(0, column) : second(0, column);
          ASSERT_LE(std::abs(actual - reference(row, column)), 2.0 * gamma * magnitude)
              << row << ',' << column;
        }
      // The synthesis: `inner` node rows from one mode's row.
      RealView2 column_weights("column_weights", inner, 1);
      RealView2 mode("mode", 1, columns);
      RealView2 synthesised("synthesised", inner, columns);
      RealView2 direct("direct", inner, columns);
      fill(column_weights);
      fill(mode);
      native::kokkosblas_gemm(execution, column_weights, mode, direct, 0.0);
      native::one_mode_synthesis(execution, column_weights, mode, synthesised);
      execution.fence("one-mode synthesis test");
      const auto w = host(column_weights);
      const auto m = host(mode);
      const auto out = host(synthesised);
      const auto blas = host(direct);
      for (std::size_t node = 0; node < inner; ++node)
        for (std::size_t column = 0; column < columns; ++column) {
          ASSERT_EQ(out(node, column), w(node, 0) * m(0, column));
          ASSERT_LE(std::abs(out(node, column) - blas(node, column)),
                    2.0 * test::higham_gamma(2) * std::abs(out(node, column)));
        }
    }
  const native::ExecutionSpace execution;
  RealView2 three("three", 3, 4);
  RealView2 nodes("nodes", 4, 8);
  RealView2 row("row", 1, 8);
  EXPECT_THROW(native::one_mode_projection(execution, three, nodes, row, row),
               std::invalid_argument);
  EXPECT_THROW(native::one_mode_projection(execution, nodes, three, row, row),
               std::invalid_argument);
  EXPECT_THROW(native::one_mode_synthesis(execution, three, nodes, row),
               std::invalid_argument);
}

TEST(Gemm, EngineSelectsTheNativeOrderKernelWithoutAVendorBlas) {
  EXPECT_EQ(native::native_order_gemm_enabled,
            native::selected_blas_provider_name() == "kokkosblas-native-order");
}

} // namespace
