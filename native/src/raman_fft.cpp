// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/raman_fft.hpp"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace solprop_native {

struct AccelerateRamanPlan::Implementation {
  struct SetupDeleter {
    void operator()(FFTSetupD setup) const noexcept { vDSP_destroy_fftsetupD(setup); }
  };
  using Setup = std::unique_ptr<std::remove_pointer_t<FFTSetupD>, SetupDeleter>;
  double* source;
  double* target;
  std::size_t samples, rows, task_count;
  vDSP_Stride source_stride, target_stride;
  bool forward;
  std::vector<Setup> setups;

  Implementation(double* input, double* output, const std::size_t length,
                 const std::size_t count, const std::size_t input_stride,
                 const std::size_t output_stride, const bool direction, const int tasks)
      : source(input), target(output), samples(length), rows(count),
        task_count(std::min(static_cast<std::size_t>(tasks), count)),
        source_stride(static_cast<vDSP_Stride>(input_stride)),
        target_stride(static_cast<vDSP_Stride>(output_stride)), forward(direction) {
    setups.reserve(task_count);
    for (std::size_t task = 0; task < task_count; ++task) {
      Setup setup(vDSP_create_fftsetupD(
          static_cast<vDSP_Length>(std::countr_zero(samples)), kFFTRadix2));
      if (!setup)
        throw std::runtime_error("Accelerate Raman FFT setup failed");
      setups.push_back(std::move(setup));
    }
  }

  static void execute_range(void* context, const std::size_t task) {
    const auto& plan = *static_cast<Implementation*>(context);
    const auto first = plan.rows * task / plan.task_count;
    const auto last = plan.rows * (task + 1) / plan.task_count;
    auto* input = plan.source + static_cast<vDSP_Stride>(first) * plan.source_stride;
    auto* output = plan.target + static_cast<vDSP_Stride>(first) * plan.target_stride;
    DSPDoubleSplitComplex source_split{input, input + 1};
    DSPDoubleSplitComplex target_split{output, output + 1};
    const auto log_length = static_cast<vDSP_Length>(std::countr_zero(plan.samples));
    if (plan.forward) {
      vDSP_fftm_zropD(plan.setups[task].get(), &source_split, 2, plan.source_stride,
                      &target_split, 2, plan.target_stride, log_length, last - first,
                      FFT_FORWARD);
      for (std::size_t row = 0; row < last - first; ++row) {
        auto* values = output + static_cast<vDSP_Stride>(row) * plan.target_stride;
        // vDSP's real forward transform is twice the raw DFT and packs Nyquist
        // into Im(DC). Restore the FFTW half-spectrum before existing kernels run.
        const double nyquist = 0.5 * values[1];
        for (std::size_t sample = 0; sample < plan.samples; ++sample)
          values[sample] *= 0.5;
        values[1] = 0.0;
        values[plan.samples] = nyquist;
        values[plan.samples + 1] = 0.0;
      }
    } else {
      for (std::size_t row = 0; row < last - first; ++row) {
        const auto* spectrum =
            input + static_cast<vDSP_Stride>(row) * plan.source_stride;
        auto* values = output + static_cast<vDSP_Stride>(row) * plan.target_stride;
        // Use the existing time output for packed inverse scratch. Preserve the
        // input spectrum and leave the convolution's explicit 1/N scale alone.
        std::memcpy(values, spectrum, plan.samples * sizeof(double));
        values[1] = spectrum[plan.samples];
      }
      vDSP_fftm_zripD(plan.setups[task].get(), &target_split, 2, plan.target_stride,
                      log_length, last - first, FFT_INVERSE);
    }
  }
};

AccelerateRamanPlan::AccelerateRamanPlan(double* source, double* target,
                                         const std::size_t samples,
                                         const std::size_t rows,
                                         const std::size_t source_stride,
                                         const std::size_t target_stride,
                                         const bool forward, const int tasks) {
  const auto maximum_stride =
      static_cast<std::size_t>(std::numeric_limits<vDSP_Stride>::max());
  if (source == nullptr || target == nullptr || samples < 8 ||
      !std::has_single_bit(samples) || rows == 0 || tasks < 1 ||
      source_stride > maximum_stride || target_stride > maximum_stride)
    throw std::invalid_argument("Invalid Accelerate Raman plan dimensions");
  implementation_ = std::make_unique<Implementation>(
      source, target, samples, rows, source_stride, target_stride, forward, tasks);
}

AccelerateRamanPlan::~AccelerateRamanPlan() = default;

void AccelerateRamanPlan::execute_task(const std::size_t task) const {
  Implementation::execute_range(implementation_.get(), task);
}

int AccelerateRamanPlan::tasks() const noexcept {
  return static_cast<int>(implementation_->task_count);
}

} // namespace solprop_native
