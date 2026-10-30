// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <KokkosFFT.hpp>
#include <memory>

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
#include <cstdlib>
#include <fftw3.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace solprop_native {

// The two kinds of batched transform the plan policy tells apart: one-dimensional rows
// (the temporal, modal, Raman and plasma transforms) and free space's two-dimensional
// transverse planes.
enum class FftShape { rows, planes };

// The one plan policy for the host FFTW plans: the effort at which a transform of this
// shape, run on `threads` threads, is seeded. Kokkos-FFT plans at FFTW_ESTIMATE and
// consumes the wisdom a higher-effort temporary plan of the same transform left, so
// FFTW_ESTIMATE here means "do not seed". An explicit SOLPROP_FFTW_PLAN (shared with
// solprop's Numba plans) applies to every transform at every width. Unset, it is
// `measure`, except for threaded rows, which stay at ESTIMATE: there measured planning
// chose a slower threaded decomposition (solprop's docs/dev/benchmarks.md, "Threaded
// FFT plan selection"). Transverse planes are measured at any width (solprop SEP 28:
// 3.2-3.4 times faster than ESTIMATE).
inline unsigned fftw_seed_effort(const FftShape shape, const int threads) {
  const char* text = std::getenv("SOLPROP_FFTW_PLAN");
  if (text == nullptr)
    return shape == FftShape::rows && threads > 1 ? FFTW_ESTIMATE : FFTW_MEASURE;
  const std::string_view effort = text;
  if (effort == "estimate")
    return FFTW_ESTIMATE;
  return effort == "patient" ? FFTW_PATIENT : FFTW_MEASURE;
}

// Seed only last-axis, contiguous, out-of-place row transforms. Planning may
// destroy BOTH buffers: callers must invoke this before filling working data, and
// before the transform's own Kokkos-FFT plan exists, which would otherwise keep
// ESTIMATE. There is no allocation or planning in execution, and the temporary plan
// owns no field storage. Already measured geometries reuse FFTW's process-local wisdom.
template <class Execution, class Input, class Output>
void seed_row_fft_wisdom(const Execution& execution, const Input& input,
                         const Output& output, const KokkosFFT::Direction direction) {
  const auto effort = fftw_seed_effort(FftShape::rows, execution.concurrency());
  if (effort == FFTW_ESTIMATE)
    return;
  constexpr bool real_input =
      std::is_same_v<typename Input::non_const_value_type, double>;
  constexpr bool real_output =
      std::is_same_v<typename Output::non_const_value_type, double>;
  const int samples =
      static_cast<int>(real_output ? output.extent(1) : input.extent(1));
  // Leave noncanonical geometry to Kokkos-FFT's own validation and planner.
  if (input.extent(0) != output.extent(0) || input.stride(1) != 1 ||
      output.stride(1) != 1 || input.stride(0) != input.extent(1) ||
      output.stride(0) != output.extent(1) ||
      (real_input && output.extent(1) != input.extent(1) / 2 + 1) ||
      (real_output && input.extent(1) != output.extent(1) / 2 + 1) ||
      (!real_input && !real_output && input.extent(1) != output.extent(1)))
    return;
  const int rows = static_cast<int>(input.extent(0));
  const int input_width = static_cast<int>(input.extent(1));
  const int output_width = static_cast<int>(output.extent(1));
  fftw_init_threads();
  fftw_plan_with_nthreads(execution.concurrency());
  fftw_plan seeded = nullptr;
  if constexpr (real_input)
    seeded = fftw_plan_many_dft_r2c(1, &samples, rows, input.data(), &input_width, 1,
                                    input_width,
                                    reinterpret_cast<fftw_complex*>(output.data()),
                                    &output_width, 1, output_width, effort);
  else if constexpr (real_output)
    seeded = fftw_plan_many_dft_c2r(
        1, &samples, rows, reinterpret_cast<fftw_complex*>(input.data()), &input_width,
        1, input_width, output.data(), &output_width, 1, output_width, effort);
  else
    seeded = fftw_plan_many_dft(
        1, &samples, rows, reinterpret_cast<fftw_complex*>(input.data()), &input_width,
        1, input_width, reinterpret_cast<fftw_complex*>(output.data()), &output_width,
        1, output_width,
        direction == KokkosFFT::Direction::forward ? FFTW_FORWARD : FFTW_BACKWARD,
        effort);
  if (seeded == nullptr)
    throw std::runtime_error("FFTW could not plan a contiguous row transform");
  fftw_destroy_plan(seeded);
}

// FFTW's process-wide wisdom as text, for solprop's opt-in SOLPROP_FFTW_WISDOM: the
// Python adapter owns the file, its key and its lock, and calls these with the GIL
// held, so they never race the planner. Empty when the provider implements FFTW's API
// without its wisdom (MKL). Both first register the threaded solvers, as every seed
// does: they are part of the planner's configuration, whose signature heads exported
// wisdom and must match for an import.
inline std::string export_fftw_wisdom() {
  fftw_init_threads();
  // FFTW allocates the text with plain malloc and documents free() for it: fftw_free
  // pairs with fftw_malloc's aligned allocator, which on Windows is another heap.
  const auto release = [](char* pointer) { std::free(pointer); };
  const std::unique_ptr<char, decltype(release)> text(fftw_export_wisdom_to_string(),
                                                      release);
  return text ? std::string(text.get()) : std::string();
}

// Merge `text` into the process's wisdom. FFTW refuses, leaving its wisdom as it was,
// text that is not wisdom or that another FFTW version or configuration wrote (its
// preamble carries the planner's signature). Wisdom only chooses among FFTW's exact
// algorithms, so a plan taken from it computes the same transform.
inline bool import_fftw_wisdom(const std::string& text) {
  fftw_init_threads();
  return fftw_import_wisdom_from_string(text.c_str()) != 0;
}

} // namespace solprop_native
#endif

#include "solprop_native/graph.hpp"

#include <utility>
#include <vector>

namespace solprop_native {

namespace graph {
#if SOLPROP_NATIVE_IMPL_GRAPH
/** `KokkosFFT::execute(plan, input, output, normalization)`, recorded in the calling
    thread's device graph region (`graph::Region`), if any, as the plan, the views it
    transforms and the normalisation. */
template <class Plan, class Input, class Output>
void execute(const Plan& plan, const Input& input, const Output& output,
             const KokkosFFT::Normalization normalization) {
  if (Region::current() == nullptr) {
    KokkosFFT::execute(plan, input, output, normalization);
    return;
  }
  operation<detail::Transform<Plan, Input, Output>>(
      [&plan, input, output, normalization] {
        KokkosFFT::execute(plan, input, output, normalization);
      },
      &plan, input, output, normalization);
}
#else
// Host builds call Kokkos-FFT's own.
using KokkosFFT::execute;
#endif
} // namespace graph

// A stacked product buffer has padding between its active tail blocks. Persist one
// contiguous row plan per block instead of transforming inactive rows or packing
// extra buffers in the stage. All aliases and plans are created with the workspace.
template <class Execution, class Input, class Output> class TailRowPlans {
  using Plan = KokkosFFT::Plan<Execution, Input, Output>;
  struct Block {
    Input input;
    Output output;
    std::unique_ptr<Plan> plan;
  };
  std::vector<Block> blocks_;

public:
  void create(const Execution& execution, const Input& input, const Output& output,
              const std::size_t rows, const std::size_t active,
              const KokkosFFT::Direction direction) {
    for (std::size_t first = 0; first < input.extent(0); first += rows) {
      const auto range = std::make_pair(first, first + active);
      Input in = Kokkos::subview(input, range, Kokkos::ALL());
      Output out = Kokkos::subview(output, range, Kokkos::ALL());
      auto plan = std::make_unique<Plan>(execution, in, out, direction, -1);
      blocks_.push_back({in, out, std::move(plan)});
    }
  }
  void execute() const {
    for (const auto& block : blocks_)
      graph::execute(*block.plan, block.input, block.output,
                     KokkosFFT::Normalization::none);
  }
  void clear() noexcept { blocks_.clear(); }
};

} // namespace solprop_native
