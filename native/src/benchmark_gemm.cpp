// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/gemm.hpp"

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using Layout = Kokkos::LayoutRight;
using Range2 = Kokkos::MDRangePolicy<
    ExecutionSpace, Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
using MemorySpace = ExecutionSpace::memory_space;
using Complex = Kokkos::complex<double>;
using RealView = Kokkos::View<double**, Layout, MemorySpace>;
using ComplexView = Kokkos::View<Complex**, Layout, MemorySpace>;

struct Shape {
  std::size_t output_rows{};
  std::size_t inner{};
  std::size_t samples{};
};

template <class Value> KOKKOS_INLINE_FUNCTION Value initial_value(const double value) {
  if constexpr (std::is_same_v<Value, Complex>) {
    return Complex(value, -0.5 * value);
  } else {
    return value;
  }
}

template <class View>
void initialise(const ExecutionSpace& execution_space, const View& view,
                const double seed) {
  Kokkos::parallel_for(
      "solprop_native.benchmark.initialise",
      Range2(execution_space, {0, 0}, {view.extent(0), view.extent(1)}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t column) {
        using Value = View::non_const_value_type;
        const double value = seed + 0.01 * static_cast<double>(row) +
                             0.001 * static_cast<double>(column);
        view(row, column) = initial_value<Value>(value);
      });
}

template <class Function>
double measure(const ExecutionSpace& execution_space, const std::size_t repetitions,
               Function&& function) {
  for (std::size_t index = 0; index < 3; ++index)
    function();
  execution_space.fence("solprop_native GEMM benchmark warmup");
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t index = 0; index < repetitions; ++index)
    function();
  execution_space.fence("solprop_native GEMM benchmark boundary");
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
                                                   started)
             .count() /
         static_cast<double>(repetitions);
}

template <class View> double maximum_difference(const View& left, const View& right) {
  double result = 0.0;
  Kokkos::parallel_reduce(
      "solprop_native.benchmark.compare",
      Range2(ExecutionSpace{}, {0, 0}, {left.extent(0), left.extent(1)}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t column, double& update) {
        update =
            Kokkos::max(update, Kokkos::abs(left(row, column) - right(row, column)));
      },
      Kokkos::Max<double>(result));
  return result;
}

template <class AView, class BView, class CView>
void team_gemm(const ExecutionSpace& execution_space, const AView& a, const BView& b,
               const CView& c, const typename CView::non_const_value_type beta,
               const int team_size) {
  using Value = CView::non_const_value_type;
  const std::size_t inner = a.extent(1);
  const std::size_t samples = c.extent(1);
  Kokkos::parallel_for(
      "solprop_native.benchmark.team_gemm",
      Kokkos::TeamPolicy<ExecutionSpace>(execution_space, static_cast<int>(c.extent(0)),
                                         team_size),
      KOKKOS_LAMBDA(
          const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type& team) {
        const std::size_t row = static_cast<std::size_t>(team.league_rank());
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, samples),
                             [&](const std::size_t column) {
                               Value value{};
                               for (std::size_t index = 0; index < inner; ++index)
                                 value += a(row, index) * b(index, column);
                               c(row, column) = value + beta * c(row, column);
                             });
      });
}

double maximum_stacked_difference(const RealView& fine, const RealView& coarse,
                                  const RealView& stacked) {
  double result = 0.0;
  const auto rows = fine.extent(0);
  Kokkos::parallel_reduce(
      "solprop_native.benchmark.compare_stacked",
      Range2(ExecutionSpace{}, {0, 0}, {2 * rows, fine.extent(1)}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t column, double& update) {
        const double expected =
            row < rows ? fine(row, column) : coarse(row - rows, column);
        update = Kokkos::max(update, Kokkos::abs(expected - stacked(row, column)));
      },
      Kokkos::Max<double>(result));
  return result;
}

std::size_t parsed(const char* value, const std::string_view name) {
  const auto result = std::strtoull(value, nullptr, 10);
  if (result == 0) {
    std::cerr << name << " must be a positive integer\n";
    std::exit(2);
  }
  return static_cast<std::size_t>(result);
}

void run(const ExecutionSpace& execution_space, const std::string_view operation,
         const Shape shape, const std::size_t repetitions, const double beta) {
  RealView a("solprop_native.benchmark.a", shape.output_rows, shape.inner);
  RealView b("solprop_native.benchmark.b", shape.inner, shape.samples);
  RealView custom("solprop_native.benchmark.custom", shape.output_rows, shape.samples);
  RealView blas("solprop_native.benchmark.blas", shape.output_rows, shape.samples);
  initialise(execution_space, a, 0.25);
  initialise(execution_space, b, -0.125);
  Kokkos::deep_copy(execution_space, custom, 0.0);
  Kokkos::deep_copy(execution_space, blas, 0.0);
  solprop_native::custom_gemm(execution_space, a, b, custom, beta);
  solprop_native::kokkosblas_gemm(execution_space, a, b, blas, beta);
  execution_space.fence("solprop_native GEMM benchmark reference");
  const double difference = maximum_difference(custom, blas);
  const double custom_us = measure(execution_space, repetitions, [&] {
    solprop_native::custom_gemm(execution_space, a, b, custom, beta);
  });
  const double blas_us = measure(execution_space, repetitions, [&] {
    solprop_native::kokkosblas_gemm(execution_space, a, b, blas, beta);
  });
  std::cout << operation << ",real," << shape.output_rows << ',' << shape.inner << ','
            << shape.samples << ',' << beta << ',' << repetitions << ','
            << std::setprecision(9) << custom_us << ',' << blas_us << ','
            << custom_us / blas_us << ',' << difference << '\n';
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const std::array team_sizes{64, 128, 256, 512};
#else
  const std::array team_sizes{1, 2, 4, 8};
#endif
  for (const int team_size : team_sizes) {
    Kokkos::deep_copy(execution_space, custom, 0.0);
    Kokkos::deep_copy(execution_space, blas, 0.0);
    solprop_native::custom_gemm(execution_space, a, b, custom, beta);
    team_gemm(execution_space, a, b, blas, beta, team_size);
    execution_space.fence("solprop_native team GEMM benchmark reference");
    const double team_difference = maximum_difference(custom, blas);
    const double team_us = measure(execution_space, repetitions, [&] {
      team_gemm(execution_space, a, b, blas, beta, team_size);
    });
    std::cout << operation << "-team" << team_size << ",real," << shape.output_rows
              << ',' << shape.inner << ',' << shape.samples << ',' << beta << ','
              << repetitions << ',' << custom_us << ',' << team_us << ','
              << custom_us / team_us << ',' << team_difference << '\n';
  }

  ComplexView complex_a("solprop_native.benchmark.complex_a", shape.output_rows,
                        shape.inner);
  ComplexView complex_b("solprop_native.benchmark.complex_b", shape.inner,
                        shape.samples);
  ComplexView mixed("solprop_native.benchmark.mixed", shape.output_rows, shape.samples);
  ComplexView complex_blas("solprop_native.benchmark.complex_blas", shape.output_rows,
                           shape.samples);
  initialise(execution_space, complex_b, -0.125);
  Kokkos::parallel_for(
      "solprop_native.benchmark.promote_real_matrix",
      Range2(execution_space, {0, 0}, {a.extent(0), a.extent(1)}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t column) {
        complex_a(row, column) = Complex(a(row, column), 0.0);
      });
  Kokkos::deep_copy(execution_space, mixed, Complex{});
  Kokkos::deep_copy(execution_space, complex_blas, Complex{});
  solprop_native::custom_gemm(execution_space, a, complex_b, mixed, Complex(beta));
  solprop_native::kokkosblas_gemm(execution_space, complex_a, complex_b, complex_blas,
                                  Complex(beta));
  execution_space.fence("solprop_native complex GEMM benchmark reference");
  const double complex_difference = maximum_difference(mixed, complex_blas);
  const double mixed_us = measure(execution_space, repetitions, [&] {
    solprop_native::custom_gemm(execution_space, a, complex_b, mixed, Complex(beta));
  });
  const double complex_blas_us = measure(execution_space, repetitions, [&] {
    solprop_native::kokkosblas_gemm(execution_space, complex_a, complex_b, complex_blas,
                                    Complex(beta));
  });
  std::cout << operation << ",mixed-vs-complex," << shape.output_rows << ','
            << shape.inner << ',' << shape.samples << ',' << beta << ',' << repetitions
            << ',' << mixed_us << ',' << complex_blas_us << ','
            << mixed_us / complex_blas_us << ',' << complex_difference << '\n';
}

void run_transform_pair(const ExecutionSpace& execution_space, const std::size_t nodes,
                        const std::size_t modes, const std::size_t samples,
                        const std::size_t repetitions) {
  run(execution_space, "synthesis-overwrite", {nodes, modes, samples}, repetitions,
      0.0);
  run(execution_space, "projection-overwrite", {modes, nodes, samples}, repetitions,
      0.0);
  run(execution_space, "projection-accumulate", {modes, nodes, samples}, repetitions,
      1.0);
  run(execution_space, "fine-coarse-stacked", {2 * modes, nodes, samples}, repetitions,
      0.0);

  RealView fine_matrix("solprop_native.benchmark.fine_matrix", modes, nodes);
  RealView coarse_matrix("solprop_native.benchmark.coarse_matrix", modes, nodes);
  RealView stacked_matrix("solprop_native.benchmark.stacked_matrix", 2 * modes, nodes);
  RealView response("solprop_native.benchmark.response", nodes, samples);
  RealView fine("solprop_native.benchmark.fine", modes, samples);
  RealView coarse("solprop_native.benchmark.coarse", modes, samples);
  RealView stacked("solprop_native.benchmark.stacked", 2 * modes, samples);
  initialise(execution_space, fine_matrix, 0.25);
  initialise(execution_space, coarse_matrix, -0.5);
  initialise(execution_space, response, -0.125);
  Kokkos::parallel_for(
      "solprop_native.benchmark.stack_projection",
      Range2(execution_space, {0, 0}, {2 * modes, nodes}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t column) {
        stacked_matrix(row, column) =
            row < modes ? fine_matrix(row, column) : coarse_matrix(row - modes, column);
      });
  const auto separate = [&] {
    solprop_native::kokkosblas_gemm(execution_space, fine_matrix, response, fine, 0.0);
    solprop_native::kokkosblas_gemm(execution_space, coarse_matrix, response, coarse,
                                    0.0);
  };
  const auto combined = [&] {
    solprop_native::kokkosblas_gemm(execution_space, stacked_matrix, response, stacked,
                                    0.0);
  };
  separate();
  combined();
  execution_space.fence("solprop_native stacked projection reference");
  const double difference = maximum_stacked_difference(fine, coarse, stacked);
  const double separate_us = measure(execution_space, repetitions, separate);
  const double stacked_us = measure(execution_space, repetitions, combined);
  std::cout << "fine-coarse-dispatch,real," << 2 * modes << ',' << nodes << ','
            << samples << ",0," << repetitions << ',' << separate_us << ','
            << stacked_us << ',' << separate_us / stacked_us << ',' << difference
            << '\n';
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  const ExecutionSpace execution_space;
  std::cout << "operation,kind,rows,inner,samples,beta,repetitions,baseline_us,"
               "candidate_us,speedup,max_abs_diff\n";
  if (argc == 5) {
    run_transform_pair(execution_space, parsed(argv[1], "nodes"),
                       parsed(argv[2], "modes"), parsed(argv[3], "samples"),
                       parsed(argv[4], "repetitions"));
    return 0;
  }
  const std::vector<std::size_t> modes{1, 2, 4, 6, 8};
  const std::vector<std::size_t> rows{1, 8, 21, 33, 41, 64, 96};
  const std::vector<std::size_t> samples{1024, 4096, 8192, 32768, 131072};
  for (const auto mode : modes) {
    for (const auto row : rows) {
      for (const auto count : samples) {
        const std::size_t repetitions =
            std::max<std::size_t>(7, 1048576 / std::max<std::size_t>(count, 1));
        run_transform_pair(execution_space, row, mode, count, repetitions);
      }
    }
  }
}
