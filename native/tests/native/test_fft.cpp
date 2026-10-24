// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/core.hpp"
#include "solprop_native/fft_plan.hpp"
#include "solprop_native/raman_fft.hpp"

#include <KokkosFFT.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numbers>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {
namespace native = solprop_native;
using native::Complex;
using ComplexRows = Kokkos::View<Complex**, native::Layout, native::MemorySpace>;
using RealRows = Kokkos::View<double**, native::Layout, native::MemorySpace>;

// The existing native FFT envelope includes direct-DFT reference roundoff.
double fft_bound(const int length, const double scale) {
  return 512.0 * std::numeric_limits<double>::epsilon() * std::ceil(std::log2(length)) *
         std::max(1.0, scale);
}

template <typename View>
auto& element(const View& view, const int axis, const int batch, const int sample) {
  return axis == 1 ? view(batch, sample) : view(sample, batch);
}

template <bool Real> void check_transform(const int axis, const int length) {
  constexpr int batches = 3;
  const int frequencies = Real ? length / 2 + 1 : length;
  using Input = std::conditional_t<Real, RealRows, ComplexRows>;
  const Input input("fft_contract.input", axis == 1 ? batches : length,
                    axis == 1 ? length : batches);
  const Input restored("fft_contract.restored", input.extent(0), input.extent(1));
  const ComplexRows spectrum("fft_contract.spectrum", axis == 1 ? batches : frequencies,
                             axis == 1 ? frequencies : batches);
  const auto host = Kokkos::create_mirror_view(input);
  for (int batch = 0; batch < batches; ++batch) {
    for (int sample = 0; sample < length; ++sample) {
      const double real = std::cos(0.3 * sample) + 0.125 * batch;
      if constexpr (Real)
        element(host, axis, batch, sample) = real;
      else
        element(host, axis, batch, sample) = Complex(real, std::sin(0.7 * sample));
    }
  }
  Kokkos::deep_copy(input, host);
  const native::ExecutionSpace execution;
  KokkosFFT::Plan forward(execution, input, spectrum, KokkosFFT::Direction::forward,
                          axis);
  KokkosFFT::Plan inverse(execution, spectrum, restored, KokkosFFT::Direction::backward,
                          axis);
  KokkosFFT::execute(forward, input, spectrum, KokkosFFT::Normalization::none);
  const auto transformed =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, spectrum);
  for (int batch = 0; batch < batches; ++batch) {
    for (int frequency = 0; frequency < frequencies; ++frequency) {
      Complex expected{};
      for (int sample = 0; sample < length; ++sample) {
        const double phase = -2.0 * std::numbers::pi_v<double> * frequency * sample /
                             static_cast<double>(length);
        expected += element(host, axis, batch, sample) *
                    Complex(std::cos(phase), std::sin(phase));
      }
      const auto actual = element(transformed, axis, batch, frequency);
      const double bound = fft_bound(length, static_cast<double>(length));
      EXPECT_NEAR(actual.real(), expected.real(), bound);
      EXPECT_NEAR(actual.imag(), expected.imag(), bound);
    }
  }
  KokkosFFT::execute(inverse, spectrum, restored, KokkosFFT::Normalization::backward);
  const auto roundtrip =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, restored);
  for (int batch = 0; batch < batches; ++batch)
    for (int sample = 0; sample < length; ++sample)
      EXPECT_LE(Kokkos::abs(Complex(element(roundtrip, axis, batch, sample)) -
                            Complex(element(host, axis, batch, sample))),
                fft_bound(length, 2.0));
}

TEST(FftProvider, RealAndComplexBatchesPreserveSignAndNormalization) {
  for (const int axis : {0, 1}) {
    for (const int length : {15, 32}) {
      SCOPED_TRACE(axis);
      SCOPED_TRACE(length);
      check_transform<true>(axis, length);
      check_transform<false>(axis, length);
    }
  }
}

TEST(FftProvider, StackedTailPlansLeaveEveryBlocksPaddingUntouched) {
  constexpr std::size_t length = 32, rows = 7, active = 3, blocks = 3;
  const RealRows input("fft_contract.stacked_input", blocks * rows, length);
  const ComplexRows output("fft_contract.stacked_output", blocks * rows,
                           length / 2 + 1);
  const RealRows restored("fft_contract.stacked_restored", blocks * rows, length);
  native::TailRowPlans<native::ExecutionSpace, RealRows, ComplexRows> forward;
  native::TailRowPlans<native::ExecutionSpace, ComplexRows, RealRows> inverse;
  const native::ExecutionSpace space;
  forward.create(space, input, output, rows, active, KokkosFFT::Direction::forward);
  inverse.create(space, output, restored, rows, active, KokkosFFT::Direction::backward);
  Kokkos::deep_copy(input, 1.0);
  Kokkos::deep_copy(output, Complex(-7.0, 11.0));
  Kokkos::deep_copy(restored, -9.0);
  for (int repeat = 0; repeat < 2; ++repeat) {
    forward.execute();
    const auto spectrum =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
    for (std::size_t row = 0; row < blocks * rows; ++row)
      for (std::size_t k = 0; k <= length / 2; ++k) {
        const bool live = row % rows < active;
        const auto expected =
            live ? Complex(k == 0 ? length : 0, 0) : Complex(-7.0, 11.0);
        EXPECT_LE(Kokkos::abs(spectrum(row, k) - expected),
                  live ? fft_bound(length, length) : 0.0);
      }
    inverse.execute();
    const auto host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, restored);
    for (std::size_t row = 0; row < blocks * rows; ++row)
      for (std::size_t k = 0; k < length; ++k)
        EXPECT_NEAR(host(row, k), row % rows < active ? length : -9.0,
                    row % rows < active ? fft_bound(length, length) : 0.0);
  }
  forward.clear();
  inverse.clear();
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
TEST(FftProvider, RowWisdomUsesSerialDefaultAndHonorsExplicitEffort) {
  // Providers such as MKL expose the FFTW API without implementing its wisdom.
  using Wisdom = std::unique_ptr<char, decltype(&fftw_free)>;
  const Wisdom supported(fftw_export_wisdom_to_string(), fftw_free);
  if (!supported)
    GTEST_SKIP() << "The linked FFT provider does not export FFTW wisdom";
  using Plan =
      std::unique_ptr<std::remove_pointer_t<fftw_plan>, decltype(&fftw_destroy_plan)>;
  constexpr int rows = 3;
  const int samples = 32;
  const ComplexRows input("fft_contract.wisdom_input", rows, samples);
  const ComplexRows output("fft_contract.wisdom_output", rows, samples);
  const char* requested = std::getenv("SOLPROP_FFTW_PLAN");
  for (const int threads : {1, native::ExecutionSpace().concurrency()}) {
    SCOPED_TRACE(threads);
    const auto execution = native::execution_instance(threads);
    fftw_forget_wisdom();
    native::seed_row_fft_wisdom(execution, input, output,
                                KokkosFFT::Direction::backward);
    fftw_init_threads();
    fftw_plan_with_nthreads(execution.concurrency());
    // WISDOM_ONLY cannot search or create a plan. Its success witnesses that
    // the seeding call really measured this geometry with this thread count.
    const Plan witness(
        fftw_plan_many_dft(
            1, &samples, rows, reinterpret_cast<fftw_complex*>(input.data()), &samples,
            1, samples, reinterpret_cast<fftw_complex*>(output.data()), &samples, 1,
            samples, FFTW_BACKWARD, FFTW_WISDOM_ONLY | FFTW_MEASURE),
        fftw_destroy_plan);
    const bool measured = requested == nullptr
                              ? execution.concurrency() == 1
                              : std::string_view(requested) != "estimate";
    EXPECT_EQ(static_cast<bool>(witness), measured);
  }
}

TEST(RamanFftProvider, LongFallbackRowsHonorPlannerEffort) {
  using Wisdom = std::unique_ptr<char, decltype(&fftw_free)>;
  const Wisdom supported(fftw_export_wisdom_to_string(), fftw_free);
  if (!supported)
    GTEST_SKIP() << "The linked FFT provider does not export FFTW wisdom";
  using Plan =
      std::unique_ptr<std::remove_pointer_t<fftw_plan>, decltype(&fftw_destroy_plan)>;
  constexpr int rows = 1;
  const int samples = 32768;
  const int frequencies = samples / 2 + 1;
  const RealRows input("raman_fft.wisdom_input", rows, samples);
  const ComplexRows output("raman_fft.wisdom_output", rows, frequencies);
  const char* requested = std::getenv("SOLPROP_FFTW_PLAN");
  for (const int threads : {1, native::ExecutionSpace().concurrency()}) {
    SCOPED_TRACE(threads);
    const auto execution = native::execution_instance(threads);
    fftw_forget_wisdom();
    native::RamanFftPlan forward(execution, input, output,
                                 KokkosFFT::Direction::forward, -1);
    if (std::string_view(forward.provider()) != "KokkosFFT")
      GTEST_SKIP() << "This library uses the vDSP provider for long Raman rows";
    fftw_plan_with_nthreads(execution.concurrency());
    // ESTIMATE alone cannot create MEASURE wisdom. This witnesses the Raman
    // wrapper's seeding for a length beyond the former short-row cutoff.
    const Plan witness(fftw_plan_many_dft_r2c(
                           1, &samples, rows, input.data(), &samples, 1, samples,
                           reinterpret_cast<fftw_complex*>(output.data()), &frequencies,
                           1, frequencies, FFTW_WISDOM_ONLY | FFTW_MEASURE),
                       fftw_destroy_plan);
    const bool measured = requested == nullptr
                              ? execution.concurrency() == 1
                              : std::string_view(requested) != "estimate";
    EXPECT_EQ(static_cast<bool>(witness), measured);
  }
}
#endif

TEST(FftProvider, PartialChunkPlanLeavesUnusedRowsUntouched) {
  constexpr int length = 32;
  constexpr int rows = 7;
  constexpr int tail = 3;
  const RealRows allocation("fft_contract.tail_input", rows, length);
  const ComplexRows output("fft_contract.tail_output", rows, length / 2 + 1);
  Kokkos::deep_copy(allocation, 1.0);
  Kokkos::deep_copy(output, Complex(-7.0, 11.0));
  const auto input_tail = Kokkos::subview(allocation, std::pair{0, tail}, Kokkos::ALL);
  const auto output_tail = Kokkos::subview(output, std::pair{0, tail}, Kokkos::ALL);
  const native::ExecutionSpace execution;
  KokkosFFT::Plan forward(execution, input_tail, output_tail,
                          KokkosFFT::Direction::forward, 1);
  KokkosFFT::execute(forward, input_tail, output_tail, KokkosFFT::Normalization::none);
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  for (int row = 0; row < rows; ++row) {
    for (int frequency = 0; frequency <= length / 2; ++frequency) {
      const auto expected =
          row < tail ? Complex(frequency == 0 ? length : 0, 0) : Complex(-7.0, 11.0);
      EXPECT_LE(Kokkos::abs(host(row, frequency) - expected),
                row < tail ? fft_bound(length, length) : 0.0);
    }
  }
}
} // namespace

namespace {
TEST(RamanFftProvider, PreservesRawNormalizationInputAndPartialRows) {
  // Include the long full/tail geometries whose provider choice differs from
  // the short serial rows. Padding remains an exact storage contract.
  constexpr std::array shapes{std::array{15, 7, 3},      std::array{32, 7, 3},
                              std::array{16384, 7, 3},   std::array{32768, 7, 3},
                              std::array{65536, 21, 12}, std::array{131072, 3, 1}};
  for (const int threads : {1, native::ExecutionSpace().concurrency()}) {
    for (const auto& [length, rows, tail] : shapes) {
      SCOPED_TRACE(length);
      SCOPED_TRACE(threads);
      const RealRows input("raman_fft.input", rows, length);
      const RealRows restored("raman_fft.restored", rows, length);
      const ComplexRows output("raman_fft.output", rows, length / 2 + 1);
      const auto input_host = Kokkos::create_mirror(input);
      for (int row = 0; row < rows; ++row)
        for (int sample = 0; sample < length; ++sample)
          input_host(row, sample) =
              1.0 + 0.5 * std::cos(2.0 * std::numbers::pi_v<double> * sample / length) +
              (length == 32 ? (sample % 2 == 0 ? 0.125 : -0.125) : 0.0);
      const auto inputs = Kokkos::subview(input, std::pair{0, tail}, Kokkos::ALL);
      const auto outputs = Kokkos::subview(output, std::pair{0, tail}, Kokkos::ALL);
      const auto inverses = Kokkos::subview(restored, std::pair{0, tail}, Kokkos::ALL);
      const auto execution = native::execution_instance(threads);
      native::RamanFftPlan forward(execution, inputs, outputs,
                                   KokkosFFT::Direction::forward, -1);
      native::RamanFftPlan inverse(execution, outputs, inverses,
                                   KokkosFFT::Direction::backward, -1);
      // MEASURE is destructive; populate all working data only after every plan.
      Kokkos::deep_copy(input, input_host);
      Kokkos::deep_copy(restored, -5.0);
      Kokkos::deep_copy(output, Complex(-7.0, 11.0));
      forward.execute(inputs, outputs, KokkosFFT::Normalization::none);
      // A host mirror View can alias output. FFTW's inverse may consume that
      // spectrum, so the forward-transform witness needs its own allocation.
      const auto spectrum = Kokkos::create_mirror(output);
      Kokkos::deep_copy(spectrum, output);
      inverse.execute(outputs, inverses, KokkosFFT::Normalization::none);
      const auto roundtrip =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, restored);
      for (int row = 0; row < rows; ++row) {
        for (int sample = 0; sample < length; ++sample)
          EXPECT_NEAR(roundtrip(row, sample),
                      row < tail ? length * input_host(row, sample) : -5.0,
                      row < tail ? fft_bound(length, length) : 0.0);
        for (int sample = 0; sample <= length / 2; ++sample) {
          const auto expected =
              row < tail
                  ? Complex(sample == 0                            ? length
                            : sample == 1                          ? 0.25 * length
                            : length == 32 && sample == length / 2 ? 0.125 * length
                                                                   : 0.0,
                            0)
                  : Complex(-7.0, 11.0);
          EXPECT_LE(Kokkos::abs(spectrum(row, sample) - expected),
                    row < tail ? fft_bound(length, length) : 0.0);
        }
      }
#if defined(SOLPROP_NATIVE_RAMAN_ACCELERATE)
      const bool neon =
          std::string_view(fftw_version).find("-neon") != std::string_view::npos;
      const bool accelerated = length != 15 && (neon ? threads > 1 && length <= 16384
                                                     : threads > 1 || length > 16384);
      EXPECT_STREQ(forward.provider(), accelerated ? "Accelerate_vDSP" : "KokkosFFT");
      EXPECT_EQ(forward.tasks(execution.concurrency()),
                accelerated ? std::min(tail, execution.concurrency())
                            : execution.concurrency());
      if (accelerated) {
        const auto after =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
        for (int row = 0; row < rows; ++row)
          for (int sample = 0; sample <= length / 2; ++sample)
            EXPECT_EQ(after(row, sample), spectrum(row, sample));
      }
#else
      EXPECT_STREQ(forward.provider(), "KokkosFFT");
#endif
    }
  }
}
} // namespace
