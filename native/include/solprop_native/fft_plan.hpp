// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <KokkosFFT.hpp>

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
#include <cstdlib>
#include <fftw3.h>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace solprop_native {

// Read the explicit effort setting shared with solprop's Numba plans. Kokkos-FFT's
// ESTIMATE plans consume matching wisdom left by a higher-effort temporary plan.
inline unsigned fftw_planner_effort() {
  const char* text = std::getenv("SOLPROP_FFTW_PLAN");
  const std::string_view effort = text == nullptr ? "measure" : text;
  if (effort == "estimate")
    return FFTW_ESTIMATE;
  return effort == "patient" ? FFTW_PATIENT : FFTW_MEASURE;
}

// Seed only last-axis, contiguous, out-of-place row transforms. Planning may
// destroy BOTH buffers: callers must invoke this before filling working data.
// There is no allocation or planning in execution, and the temporary plan owns
// no field storage. Already measured geometries reuse FFTW's process-local wisdom.
template <class Execution, class Input, class Output>
void seed_row_fft_wisdom(const Execution& execution, const Input& input,
                         const Output& output, const KokkosFFT::Direction direction) {
  // Threaded wisdom can select slower plans for the complete modal source.
  // Retain the serial benefit and make threaded measurement opt-in.
  if (execution.concurrency() > 1 && std::getenv("SOLPROP_FFTW_PLAN") == nullptr)
    return;
  const auto effort = fftw_planner_effort();
  if (effort == FFTW_ESTIMATE)
    return;
  using InputValue = typename Input::non_const_value_type;
  using OutputValue = typename Output::non_const_value_type;
  // Binary32 views plan with single-precision FFTW (`fftwf_`), whose wisdom
  // Kokkos-FFT's binary32 plans consume (solprop SEP 36 section 6.3).
  constexpr bool single = std::is_same_v<InputValue, float> ||
                          std::is_same_v<InputValue, Kokkos::complex<float>>;
  constexpr bool real_input =
      std::is_same_v<InputValue, double> || std::is_same_v<InputValue, float>;
  constexpr bool real_output =
      std::is_same_v<OutputValue, double> || std::is_same_v<OutputValue, float>;
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
  if constexpr (single) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(execution.concurrency());
    fftwf_plan seeded = nullptr;
    if constexpr (real_input)
      seeded = fftwf_plan_many_dft_r2c(1, &samples, rows, input.data(), &input_width, 1,
                                       input_width,
                                       reinterpret_cast<fftwf_complex*>(output.data()),
                                       &output_width, 1, output_width, effort);
    else if constexpr (real_output)
      seeded = fftwf_plan_many_dft_c2r(1, &samples, rows,
                                       reinterpret_cast<fftwf_complex*>(input.data()),
                                       &input_width, 1, input_width, output.data(),
                                       &output_width, 1, output_width, effort);
    else
      seeded = fftwf_plan_many_dft(
          1, &samples, rows, reinterpret_cast<fftwf_complex*>(input.data()),
          &input_width, 1, input_width, reinterpret_cast<fftwf_complex*>(output.data()),
          &output_width, 1, output_width,
          direction == KokkosFFT::Direction::forward ? FFTW_FORWARD : FFTW_BACKWARD,
          effort);
    if (seeded == nullptr)
      throw std::runtime_error("FFTW could not plan a contiguous row transform");
    fftwf_destroy_plan(seeded);
  } else {
    fftw_init_threads();
    fftw_plan_with_nthreads(execution.concurrency());
    fftw_plan seeded = nullptr;
    if constexpr (real_input)
      seeded = fftw_plan_many_dft_r2c(1, &samples, rows, input.data(), &input_width, 1,
                                      input_width,
                                      reinterpret_cast<fftw_complex*>(output.data()),
                                      &output_width, 1, output_width, effort);
    else if constexpr (real_output)
      seeded = fftw_plan_many_dft_c2r(1, &samples, rows,
                                      reinterpret_cast<fftw_complex*>(input.data()),
                                      &input_width, 1, input_width, output.data(),
                                      &output_width, 1, output_width, effort);
    else
      seeded = fftw_plan_many_dft(
          1, &samples, rows, reinterpret_cast<fftw_complex*>(input.data()),
          &input_width, 1, input_width, reinterpret_cast<fftw_complex*>(output.data()),
          &output_width, 1, output_width,
          direction == KokkosFFT::Direction::forward ? FFTW_FORWARD : FFTW_BACKWARD,
          effort);
    if (seeded == nullptr)
      throw std::runtime_error("FFTW could not plan a contiguous row transform");
    fftw_destroy_plan(seeded);
  }
}

} // namespace solprop_native
#endif

#include <memory>
#include <utility>
#include <vector>

namespace solprop_native {

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
      KokkosFFT::execute(*block.plan, block.input, block.output,
                         KokkosFFT::Normalization::none);
  }
  void clear() noexcept { blocks_.clear(); }
};

} // namespace solprop_native
