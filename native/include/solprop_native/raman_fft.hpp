// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/fft_plan.hpp"
#include "solprop_native/launch.hpp"

#include <KokkosFFT.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace solprop_native {

#if defined(SOLPROP_NATIVE_RAMAN_ACCELERATE)
// The host adapter owns only vendor setup data. All field storage remains in the
// workspace's Kokkos Views and is included in its existing allocation ledger.
class AccelerateRamanPlan {
public:
  AccelerateRamanPlan(double* source, double* target, std::size_t samples,
                      std::size_t rows, std::size_t source_stride,
                      std::size_t target_stride, bool forward, int tasks);
  ~AccelerateRamanPlan();
  AccelerateRamanPlan(const AccelerateRamanPlan&) = delete;
  AccelerateRamanPlan& operator=(const AccelerateRamanPlan&) = delete;
  void execute_task(std::size_t task) const;
  [[nodiscard]] int tasks() const noexcept;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};
#endif

// Raman alone has a macOS Accelerate provider. Unsupported layouts and lengths,
// and every other platform, retain the pinned Kokkos FFT implementation.
template <class Execution, class Input, class Output> class RamanFftPlan {
public:
  RamanFftPlan(const Execution& execution, const Input& input, const Output& output,
               const KokkosFFT::Direction direction, const int axis)
      : execution_(execution) {
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
    const bool contiguous =
        (axis == -1 || axis == 1) && input.stride(1) == 1 && output.stride(1) == 1;
#if defined(SOLPROP_NATIVE_RAMAN_ACCELERATE)
    constexpr bool forward =
        std::is_same_v<typename Input::non_const_value_type, double>;
    const auto samples = forward ? input.extent(1) : output.extent(1);
    // NEON FFTW wins the long rows, but short threaded plans can still be slower
    // than vDSP. Preserve the previous policy for libraries without NEON codelets;
    // a local build may still resolve such an FFTW even on an Arm CPU.
    const bool neon =
        std::string_view(fftw_version).find("-neon") != std::string_view::npos;
    const bool prefer_fftw =
        (execution.concurrency() == 1 && samples <= 16384) ||
        (neon && (execution.concurrency() == 1 || samples > 16384));
    if (!prefer_fftw && (axis == -1 || axis == 1) && input.stride(1) == 1 &&
        output.stride(1) == 1 && samples >= 8 && std::has_single_bit(samples)) {
      const auto frequencies = forward ? output.extent(1) : input.extent(1);
      if (input.extent(0) != output.extent(0) || frequencies != samples / 2 + 1)
        throw std::invalid_argument("Raman FFT views have inconsistent extents");
      if ((direction == KokkosFFT::Direction::forward) != forward)
        throw std::invalid_argument("Raman FFT direction does not match its views");
      static_assert(sizeof(typename Input::non_const_value_type) ==
                    (forward ? 1 : 2) * sizeof(double));
      static_assert(sizeof(typename Output::non_const_value_type) ==
                    (forward ? 2 : 1) * sizeof(double));
      accelerate_ = std::make_unique<AccelerateRamanPlan>(
          reinterpret_cast<double*>(input.data()),
          reinterpret_cast<double*>(output.data()), samples, input.extent(0),
          input.stride(0) * (forward ? 1 : 2), output.stride(0) * (forward ? 2 : 1),
          forward, execution.concurrency());
      return;
    }
#endif
    // Every FFTW fallback must honour the planner effort, including the long
    // doubled rows used when the Accelerate provider is disabled. The shared
    // helper retains serial-only measured planning unless explicitly overridden.
    if (contiguous)
      seed_row_fft_wisdom(execution, input, output, direction);
#endif
    fallback_ = std::make_unique<Fallback>(execution, input, output, direction, axis);
  }

  void execute(const Input& input, const Output& output,
               const KokkosFFT::Normalization normalization) const {
#if defined(SOLPROP_NATIVE_RAMAN_ACCELERATE)
    if (accelerate_) {
      if (normalization != KokkosFFT::Normalization::none)
        throw std::invalid_argument("Raman vDSP requires explicit normalization");
      // vDSP consumes host buffers synchronously. Make that ordering boundary
      // explicit even if a future host execution space gains asynchronous work.
      execution_.fence("solprop_native Raman vDSP input ready");
      const auto* plan = accelerate_.get();
      const int count = plan->tasks();
      if (count == 1)
        plan->execute_task(0);
      else
        launch::parallel_for(
            "solprop_native.raman.vdsp",
            Kokkos::RangePolicy<Execution>(execution_, 0, count),
            [plan](const std::size_t task) { plan->execute_task(task); });
      return;
    }
#endif
    fft_execute(*fallback_, input, output, normalization);
  }

  [[nodiscard]] const char* provider() const noexcept {
#if defined(SOLPROP_NATIVE_RAMAN_ACCELERATE)
    if (accelerate_)
      return "Accelerate_vDSP";
#endif
    return "KokkosFFT";
  }

  [[nodiscard]] int tasks(const int fallback_threads) const noexcept {
#if defined(SOLPROP_NATIVE_RAMAN_ACCELERATE)
    if (accelerate_)
      return accelerate_->tasks();
#endif
    return fallback_threads;
  }

private:
  using Fallback = KokkosFFT::Plan<Execution, Input, Output>;
  Execution execution_;
  std::unique_ptr<Fallback> fallback_;
#if defined(SOLPROP_NATIVE_RAMAN_ACCELERATE)
  std::unique_ptr<AccelerateRamanPlan> accelerate_;
#endif
};

template <class Execution, class Input, class Output>
void execute_raman(const RamanFftPlan<Execution, Input, Output>& plan,
                   const Input& input, const Output& output,
                   const KokkosFFT::Normalization normalization) {
  plan.execute(input, output, normalization);
}

} // namespace solprop_native
