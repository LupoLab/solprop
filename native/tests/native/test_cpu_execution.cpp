// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Tests of the CPU execution policy (`runtime/cpu_execution.hpp`): each operation's
// thread width and its boundaries, explicit and compact pools, the OpenMP BLAS hold of
// a compact team, the sizing metadata, and workspaces' independent pools.

#include "execution.hpp"
#include "general_problems.hpp"
#include "solprop_native/general/general_state.hpp"
#include "solprop_native/problem/problem.hpp"
#include "solprop_native/runtime/cpu_execution.hpp"
#include "solprop_native/runtime/execution.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

#ifdef KOKKOS_ENABLE_OPENMP
#include <omp.h>
#endif

namespace {

namespace native = solprop_native;
using native::Complex;
using native::test::general_problem;

TEST(CpuThreadPolicy, AutomaticBoundariesAreMonotoneAndCapped) {
  using native::CpuOperation;
  using native::CpuPool;
  const std::array operations{
      CpuOperation::stage, CpuOperation::response, CpuOperation::dense,
      CpuOperation::fft,   CpuOperation::raman,    CpuOperation::copy,
      CpuOperation::flow,  CpuOperation::modal,    CpuOperation::adams};
  const std::array thresholds{
      native::test::table()[native::CpuOperation::stage].threshold,
      native::test::table()[native::CpuOperation::response].threshold,
      native::test::table()[native::CpuOperation::dense].threshold,
      native::test::table()[native::CpuOperation::fft].threshold,
      native::test::table()[native::CpuOperation::raman].threshold,
      native::test::table()[native::CpuOperation::copy].threshold,
      native::test::table()[native::CpuOperation::flow].threshold,
      native::test::table()[native::CpuOperation::modal].threshold,
      native::test::table()[native::CpuOperation::adams].threshold};
  const std::array full_thresholds{
      native::test::table()[native::CpuOperation::stage].full,
      native::test::table()[native::CpuOperation::response].full,
      native::test::table()[native::CpuOperation::dense].full,
      native::test::table()[native::CpuOperation::fft].full,
      native::test::table()[native::CpuOperation::raman].full,
      native::test::table()[native::CpuOperation::copy].threshold,
      native::test::table()[native::CpuOperation::flow].threshold,
      native::test::table()[native::CpuOperation::modal].threshold,
      native::test::table()[native::CpuOperation::adams].full};
  for (const int cap : {1, 2, 3, 8, 72}) {
    native::ThreadPolicy policy{
        .kernels = cap, .fft = cap, .raman_fft = cap, .raman_tail_fft = cap};
    for (std::size_t index = 0; index < operations.size(); ++index) {
      const auto width = [&](const std::size_t elements) {
        return native::operation_threads(policy, native::test::table(),
                                         CpuPool::kernels, operations[index], elements);
      };
      EXPECT_EQ(width(0), native::cpu_size_policy_enabled ? 1 : cap);
      EXPECT_EQ(width(thresholds[index] - 1),
                native::cpu_size_policy_enabled ? 1 : cap);
      EXPECT_EQ(width(thresholds[index]),
                native::cpu_size_policy_enabled
                    ? (thresholds[index] < full_thresholds[index]
                           ? std::min(cap, native::test::table().medium_threads)
                           : cap)
                    : cap);
      EXPECT_EQ(width(full_thresholds[index]), cap);
      int previous = 1;
      for (std::size_t elements = 1; elements <= 4 * full_thresholds[index];
           elements *= 2) {
        const auto actual = width(elements);
        EXPECT_GE(actual, previous);
        EXPECT_LE(actual, cap);
        previous = actual;
      }
    }
  }
}

TEST(CpuThreadPolicy, ShortCopiesDoNotInheritTheStageWidth) {
  native::ThreadPolicy policy{
      .kernels = 18, .fft = 18, .raman_fft = 18, .raman_tail_fft = 18};
  EXPECT_EQ(native::operation_threads(policy, native::test::table(),
                                      native::CpuPool::kernels,
                                      native::CpuOperation::copy, 8192),
            native::cpu_size_policy_enabled ? 1 : 18);
  EXPECT_EQ(native::operation_threads(policy, native::test::table(),
                                      native::CpuPool::kernels,
                                      native::CpuOperation::stage, 8196),
            18);
  policy.explicit_pools = 1;
  EXPECT_EQ(native::operation_threads(policy, native::test::table(),
                                      native::CpuPool::kernels,
                                      native::CpuOperation::copy, 8192),
            18);
}

TEST(CpuThreadPolicy, SmoothGridSizesKeepThePowerOfTwoWidths) {
  // The four-mode RDW on n_t 4000 (a 2-3-5-smooth size) against 4096: on Linux its
  // stage of 8004 elements and coarse FFT batch of 32000 samples take the widths of
  // 8196 and 32768, rather than falling to one thread just under a power-of-two
  // boundary. macOS keeps the earlier boundaries, where the serial ones are faster.
#ifdef __APPLE__
  GTEST_SKIP() << "macOS keeps the 8192/32768 boundaries";
#endif
  native::ThreadPolicy policy{
      .kernels = 18, .fft = 18, .raman_fft = 18, .raman_tail_fft = 18};
  const auto width = [&](const native::CpuOperation operation, const std::size_t n) {
    return native::operation_threads(policy, native::test::table(),
                                     native::CpuPool::kernels, operation, n);
  };
  using native::CpuOperation;
  EXPECT_EQ(width(CpuOperation::stage, 8004), width(CpuOperation::stage, 8196));
  EXPECT_EQ(width(CpuOperation::stage, 8004), 18);
  EXPECT_EQ(width(CpuOperation::fft, 32000), width(CpuOperation::fft, 32768));
  EXPECT_GT(width(CpuOperation::fft, 32000), 1);
}

TEST(CpuThreadPolicy, CoupledMatrixWindowRetainsFullWidth) {
  native::ThreadPolicy policy{.kernels = 72,
                              .fft = 72,
                              .raman_fft = 72,
                              .raman_tail_fft = 72,
                              .band_tiles = -1};
  // Active bins * modes^3 in the measured three-mode wall, independent of runtime.
  EXPECT_EQ(native::operation_threads(
                policy, native::test::table(), native::CpuPool::kernels,
                native::CpuOperation::flow, std::size_t{1433} * 3 * 3 * 3),
            72);
}

TEST(CpuThreadPolicy, ResponseShapeThresholdsAreMonotoneAndCapped) {
  using native::CpuOperation;
  using native::CpuPool;
  for (const int cap : {1, 3, 18, 72}) {
    native::ThreadPolicy policy{
        .kernels = cap, .fft = cap, .raman_fft = cap, .raman_tail_fft = cap};
    const auto width = [&](const std::size_t rows, const std::size_t samples) {
      return native::operation_threads(policy, native::test::table(), CpuPool::kernels,
                                       CpuOperation::response, rows * samples, samples);
    };
    EXPECT_EQ(width(33, 4096),
              native::cpu_size_policy_enabled ? std::min(cap, 4) : cap);
    const auto columns = native::test::table().response_full_columns;
    EXPECT_EQ(width(columns - 1, 1024),
              native::cpu_size_policy_enabled ? std::min(cap, 4) : cap);
    EXPECT_EQ(width(columns, 1024), cap);
    EXPECT_EQ(width(columns, 1), native::cpu_size_policy_enabled ? 1 : cap);
    EXPECT_EQ(width(44, 2048), cap);
    EXPECT_EQ(width(68, 2048), cap);
    EXPECT_EQ(width(1, native::test::table().response_full_samples - 1),
              native::cpu_size_policy_enabled ? std::min(cap, 4) : cap);
    EXPECT_EQ(width(1, native::test::table().response_full_samples), cap);
    EXPECT_EQ(width(0, native::test::table().response_full_samples),
              native::cpu_size_policy_enabled ? 1 : cap);
    for (std::size_t rows = 1; rows <= 64; rows *= 2) {
      int previous = 1;
      for (std::size_t samples = 1024; samples <= 131072; samples *= 2) {
        const auto actual = width(rows, samples);
        EXPECT_GE(actual, previous);
        EXPECT_LE(actual, cap);
        previous = actual;
      }
    }
    policy.explicit_pools = 1;
    EXPECT_EQ(width(1, 1), cap);
  }
}

TEST(CpuThreadPolicy, ExplicitPoolsBypassSizingIndependently) {
  using native::CpuOperation;
  using native::CpuPool;
  native::ThreadPolicy policy{
      .kernels = 7, .fft = 6, .raman_fft = 5, .raman_tail_fft = 4};
  const std::array pools{CpuPool::kernels, CpuPool::fft, CpuPool::raman_fft,
                         CpuPool::raman_tail_fft};
  for (unsigned requested = 0; requested < pools.size(); ++requested) {
    policy.explicit_pools = 1U << requested;
    for (unsigned other = 0; other < pools.size(); ++other) {
      const auto actual = native::operation_threads(policy, native::test::table(),
                                                    pools[other], CpuOperation::fft, 1);
      EXPECT_EQ(actual, requested == other || !native::cpu_size_policy_enabled
                            ? 7 - static_cast<int>(other)
                            : 1);
    }
  }
  policy.explicit_pools = 0;
  // A record's own FFT start (solprop SEP 46: SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES).
  auto execution = native::test::table();
  execution.grain[static_cast<std::size_t>(CpuOperation::fft)].threshold = 17;
  EXPECT_EQ(
      native::operation_threads(policy, execution, CpuPool::fft, CpuOperation::fft, 16),
      native::cpu_size_policy_enabled ? 1 : 6);
  EXPECT_EQ(
      native::operation_threads(policy, execution, CpuPool::fft, CpuOperation::fft, 17),
      native::cpu_size_policy_enabled
          ? std::min(6, native::test::table().medium_threads)
          : 6);
}

TEST(CpuThreadPolicy, GeneralModalTransformsKeepTheirMeasuredWidth) {
  native::ThreadPolicy policy{
      .kernels = 18, .fft = 18, .raman_fft = 18, .raman_tail_fft = 18};
  for (const std::size_t elements : {98496U, 209304U}) {
    EXPECT_EQ(native::operation_threads(policy, native::test::table(),
                                        native::CpuPool::kernels,
                                        native::CpuOperation::modal, elements),
              18);
    EXPECT_EQ(native::operation_threads(policy, native::test::table(),
                                        native::CpuPool::kernels,
                                        native::CpuOperation::dense, elements),
              native::cpu_size_policy_enabled ? 1 : 18);
  }
}

TEST(CpuThreadPolicy, MetadataNamesEveryOperationClass) {
  native::CpuExecutionPool pool(native::CpuPool::kernels, native::test::table());
  const std::array operations{
      native::CpuOperation::stage, native::CpuOperation::response,
      native::CpuOperation::dense, native::CpuOperation::fft,
      native::CpuOperation::raman, native::CpuOperation::copy,
      native::CpuOperation::flow,  native::CpuOperation::modal,
      native::CpuOperation::adams};
  const std::array names{"stage", "response", "dense", "fft",  "raman",
                         "copy",  "flow",     "modal", "adams"};
  for (std::size_t i = 0; i < operations.size(); ++i) {
    pool.get(operations[i], 1);
    EXPECT_EQ(
        pool.sizing().at(std::string("kernels.") + names[i] + ".1").at("elements"), 1U);
  }
}

TEST(CpuThreadPolicy, ACompactTeamScopeHoldsAnOpenMpBlasAtOneThread) {
#ifndef KOKKOS_ENABLE_OPENMP
  GTEST_SKIP() << "the hold is an OpenMP build's";
#else
  const int before = omp_get_max_threads();
  if (before == 1)
    GTEST_SKIP() << "needs an OpenMP default width above one";
  native::register_openmp_blas(true);
  {
    // A scope without a compact team leaves the default alone.
    const native::CpuTeamScope none(0);
    EXPECT_EQ(omp_get_max_threads(), before);
  }
  {
    const native::CpuTeamScope outer(4);
    EXPECT_EQ(omp_get_max_threads(), 1);
    {
      // Nested scopes hold it once and release it with the outermost.
      const native::CpuTeamScope inner(4);
      EXPECT_EQ(omp_get_max_threads(), 1);
    }
    EXPECT_EQ(omp_get_max_threads(), 1);
  }
  EXPECT_EQ(omp_get_max_threads(), before);
  native::register_openmp_blas(false);
  {
    // Without an OpenMP OpenBLAS there is nothing to hold.
    const native::CpuTeamScope unregistered(4);
    EXPECT_EQ(omp_get_max_threads(), before);
  }
#endif
}

TEST(CpuThreadPolicy, AKernelPoolAtFullWidthKeepsItsTeam) {
  // GNU OpenMP ends a pool's surplus threads when a narrower region follows a wider
  // one, so a kernel pool that holds a full-width instance hands it to medium-width
  // operations too; without one, they keep the medium width. On macOS medium
  // operations keep the medium width beside one too (`stable_kernel_team`).
  const int cap = native::thread_policy().kernels;
  const int promoted = native::test::table().stable_kernel_team
                           ? cap
                           : native::test::table().medium_threads;
  if (!native::cpu_size_policy_enabled || cap <= native::test::table().medium_threads)
    GTEST_SKIP() << "needs per-operation sizing and more than the medium width";
  const auto response = std::size_t{33} * 4096;
  native::CpuExecutionPool alone(native::CpuPool::kernels, native::test::table());
  EXPECT_EQ(alone.get(native::CpuOperation::response, response, 4096).concurrency(),
            native::test::table().medium_threads);
  EXPECT_EQ(alone.get(native::CpuOperation::stage, 1).concurrency(), 1);
  native::CpuExecutionPool full(native::CpuPool::kernels, native::test::table());
  EXPECT_EQ(full.get(native::CpuOperation::stage, 1U << 20).concurrency(), cap);
  EXPECT_EQ(full.get(native::CpuOperation::response, response, 4096).concurrency(),
            promoted);
  EXPECT_EQ(full.get(native::CpuOperation::adams, 16384).concurrency(), promoted);
  EXPECT_EQ(full.get(native::CpuOperation::stage, 1).concurrency(), 1);
  const auto key = "kernels.response." + std::to_string(response) + ".samples.4096";
  EXPECT_EQ(full.sizing().at(key).at("threads"), static_cast<std::size_t>(promoted));
  // The FFT pools keep their medium widths: FFTW's threaded loops run on the
  // runtime's whole team and only distribute fewer iterations.
  native::CpuExecutionPool transforms(native::CpuPool::fft, native::test::table());
  EXPECT_EQ(transforms
                .get(native::CpuOperation::fft,
                     native::test::table()[native::CpuOperation::fft].full)
                .concurrency(),
            cap);
  EXPECT_EQ(transforms.get(native::CpuOperation::fft, 32768).concurrency(),
            native::test::table().medium_threads);
}

TEST(CpuThreadPolicy, TransformBatchesThreadByRowsAndRowLength) {
  // solprop issue #587: from the start, a thread per fft_thread_elements samples, at
  // most one a row, or up to the medium width for rows of fft_long_row samples.
  if (!native::cpu_size_policy_enabled)
    GTEST_SKIP() << "needs automatic per-operation sizing";
  native::CpuExecution execution = native::test::table();
  execution.grain[static_cast<std::size_t>(native::CpuOperation::fft)] = {
      .threshold = 8192, .full = 1U << 22};
  execution.fft_thread_elements = 4096;
  execution.fft_long_row = 16384;
  execution.medium_threads = 4;
  const native::ThreadPolicy policy{
      .kernels = 18, .fft = 18, .raman_fft = 18, .raman_tail_fft = 18};
  const auto threads = [&](const std::size_t rows, const std::size_t length) {
    return native::operation_threads(policy, execution, native::CpuPool::fft,
                                     native::CpuOperation::fft, rows * length, length);
  };
  EXPECT_EQ(threads(1, 4096), 1);      // below the start
  EXPECT_EQ(threads(4, 3072), 3);      // three threads' worth of samples
  EXPECT_EQ(threads(4, 10240), 4);     // a thread a row
  EXPECT_EQ(threads(1, 10240), 1);     // a short row is not split
  EXPECT_EQ(threads(1, 32768), 4);     // a long row, split to the medium width
  EXPECT_EQ(threads(12, 10240), 12);   // one a row
  EXPECT_EQ(threads(128, 1152), 18);   // many short rows take the pool
  EXPECT_EQ(threads(289, 128), 9);     // 36992 samples
  EXPECT_EQ(threads(1, 1U << 22), 18); // the full size is a backstop
}

TEST(CpuThreadPolicy, ACompactPoolKeepsAFewThreads) {
  // A compact problem's kernels share a team of at most cpu_compact_threads, medium
  // widths promoted to it except on macOS; its transforms take that team only where
  // FFTW's loops follow a scope, and run serially otherwise.
  const auto policy = native::thread_policy();
  if (!native::cpu_size_policy_enabled || policy.explicit_pools != 0)
    GTEST_SKIP() << "needs automatic per-operation sizing";
  const int team = std::min(policy.kernels, native::test::table().compact_threads);
  native::CpuExecutionPool kernels(native::CpuPool::kernels, native::test::table());
  kernels.compact();
  EXPECT_EQ(kernels.widest(), 0);
  EXPECT_EQ(kernels.get(native::CpuOperation::stage, 1U << 20).concurrency(), team);
  EXPECT_EQ(kernels.get(native::CpuOperation::response, std::size_t{33} * 4096, 4096)
                .concurrency(),
            native::test::table().stable_kernel_team
                ? team
                : std::min(team, native::test::table().medium_threads));
  EXPECT_EQ(kernels.get(native::CpuOperation::stage, 1).concurrency(), 1);
  EXPECT_EQ(kernels.widest(), team);
  native::CpuExecutionPool transforms(native::CpuPool::fft, native::test::table());
  transforms.compact();
  const int threaded = native::fftw_team_loops() ? team : 1;
  EXPECT_EQ(transforms.get(native::CpuOperation::fft, 1U << 20).concurrency(),
            threaded);
  // A medium transform takes the whole team; a Raman one keeps its own start.
  EXPECT_EQ(transforms.get(native::CpuOperation::fft, 32768).concurrency(), threaded);
  EXPECT_EQ(transforms.get(native::CpuOperation::fft, 1000).concurrency(), 1);
  native::CpuExecutionPool raman(native::CpuPool::raman_fft, native::test::table());
  raman.compact();
  EXPECT_EQ(raman.get(native::CpuOperation::raman, 32768).concurrency(), 1);
  EXPECT_EQ(raman
                .get(native::CpuOperation::raman,
                     native::test::table()[native::CpuOperation::raman].threshold)
                .concurrency(),
            threaded);
  native::CpuExecutionPool full(native::CpuPool::kernels, native::test::table());
  EXPECT_EQ(full.get(native::CpuOperation::stage, 1U << 20).concurrency(),
            policy.kernels);
}

TEST(CpuThreadPolicy, CompactProblemsHaveOneColumnOnOneNode) {
  const auto base = general_problem(native::FieldFamily::field_real);
  EXPECT_TRUE(native::compact_problem(base));
  auto nodes = base;
  nodes.dims.n_nodes = 2;
  EXPECT_FALSE(native::compact_problem(nodes));
  auto modes = base;
  modes.dims.n_modes = 2;
  EXPECT_FALSE(native::compact_problem(modes));
  auto pairs = base;
  pairs.dims.n_pol_transform = 2;
  EXPECT_FALSE(native::compact_problem(pairs));
  auto long_column = base;
  long_column.dims.n_to = native::test::table().compact_max_fine_samples;
  EXPECT_FALSE(native::compact_problem(long_column));
  auto modal = base;
  modal.general.enabled = true;
  EXPECT_FALSE(native::compact_problem(modal));
  auto flow = base;
  flow.flow.enabled = true;
  EXPECT_FALSE(native::compact_problem(flow));
  auto free_space = base;
  free_space.spatial.n_x = 2;
  EXPECT_FALSE(native::compact_problem(free_space));
}

TEST(CpuThreadPolicy, MetadataDistinguishesEqualWorkWithDifferentResponseShapes) {
  native::CpuExecutionPool pool(native::CpuPool::kernels, native::test::table());
  const auto samples = native::test::table().response_full_samples;
  pool.get(native::CpuOperation::response, samples, 4096);
  pool.get(native::CpuOperation::response, samples, samples);
  const auto sizing = pool.sizing();
  for (const auto fine : {std::size_t{4096}, samples}) {
    const auto key = "kernels.response." + std::to_string(samples) + ".samples." +
                     std::to_string(fine);
    EXPECT_EQ(sizing.at(key).at("elements"), samples);
    EXPECT_EQ(sizing.at(key).at("fine_samples"), fine);
    EXPECT_EQ(sizing.at(key).at("columns"), samples / fine);
  }
}

TEST(CpuThreadPolicy, MatrixPackingDoesNotInheritTheDenseWorkWidth) {
  auto problem = general_problem(native::FieldFamily::field_real);
  problem.dims.n_to = 32768;
  problem.dims.n_nodes = 35;
  problem.dims.node_chunk = 33;
  problem.transform.synthesis_matrix.assign(problem.dims.n_nodes, 1.0);
  problem.transform.projection_matrix.assign(problem.dims.n_nodes, 1.0 / 35.0);
  problem.transform.projection_coarse_matrix = problem.transform.projection_matrix;
  problem.grid.time_window.assign(problem.dims.n_to, 1.0);
  problem.grid.time_window_derivative.assign(problem.dims.n_to, 0.0);
  problem.grid.carrier.assign(problem.dims.n_to, Complex(1.0, 0.0));
  native::GeneralState state(problem);
  const auto sizing = state.execution_sizing();
  // Full and tail matrices contain 33 and 2 entries, or twice those counts
  // for stacked fine/coarse projections; the output has 32768 samples.
  for (const std::size_t elements : {2U, 4U, 33U, 66U, 32768U}) {
    const auto& choice = sizing.at("kernels.copy." + std::to_string(elements));
    EXPECT_EQ(choice.at("elements"), elements);
    EXPECT_EQ(choice.at("threads"),
              native::cpu_size_policy_enabled
                  ? 1U
                  : static_cast<unsigned>(native::ExecutionSpace().concurrency()));
  }
  EXPECT_EQ(sizing.at("kernels.dense.1081344").at("threads"),
            static_cast<std::size_t>(native::ExecutionSpace().concurrency()));
}

TEST(CpuThreadPolicy, WorkspacesKeepIndependentChoicesAndReleaseTheirPools) {
  auto problem = general_problem(native::FieldFamily::envelope);
  native::GeneralState small(problem);
  // One long row, past the compact bound: its transforms thread by shape.
  problem.dims.n_to = 65536;
  problem.grid.time_window.assign(problem.dims.n_to, 1.0);
  problem.grid.time_window_derivative.assign(problem.dims.n_to, 0.0);
  problem.grid.carrier.assign(problem.dims.n_to, Complex(1.0, 0.0));
  native::GeneralState large(problem);
  native::GeneralState another_small(general_problem(native::FieldFamily::envelope));
  const auto small_sizing = small.execution_sizing();
  EXPECT_EQ(small_sizing, another_small.execution_sizing());
  const auto key = "fft.fft." + std::to_string(problem.dims.n_to) + ".samples." +
                   std::to_string(problem.dims.n_to);
  EXPECT_EQ(large.execution_sizing().at(key).at("elements"), problem.dims.n_to);
  EXPECT_EQ(large.execution_sizing().at(key).at("threads"),
            static_cast<std::size_t>(native::operation_threads(
                native::thread_policy(), native::test::table(), native::CpuPool::fft,
                native::CpuOperation::fft, problem.dims.n_to, problem.dims.n_to)));
  large.close();
  EXPECT_EQ(small.execution_sizing(), small_sizing);
  small.close();
  another_small.close();
}

} // namespace
