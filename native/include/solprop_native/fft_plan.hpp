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

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#include "solprop_native/launch.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#endif

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
      [&] { KokkosFFT::execute(plan, input, output, normalization); },
      [&] {
        return [&plan, input, output, normalization] {
          KokkosFFT::execute(plan, input, output, normalization);
        };
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

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
// cuFFT transforms a real row of a few thousand samples in one thread block, so a
// batch of a few rows leaves most of a device's multiprocessors idle. An NVIDIA GA10x
// device (compute capability 8.6) runs FP64 at a sixty-fourth of its FP32 rate, so
// that block is FP64-bound: on an NVIDIA A40 a four-mode capillary's 4 x 10240 inverse
// ran 96 microseconds on 4 of its 84. Such a build splits the rows (`RealRowPlan`).
// An NVIDIA H200 took 12 microseconds for the same transform, faster than the split.
inline constexpr bool split_real_rows_enabled =
#if defined(KOKKOS_ARCH_AMPERE86)
    true;
#else
    false;
#endif

/** The most rows a split real-row plan takes; larger batches fill the device. */
inline constexpr std::size_t split_real_rows_max = 64;

namespace detail {
/** The first factor n1 of a split of `half` = n1 n2 points: n2 is the largest power
    of two dividing `half`, at most 512 (on an NVIDIA A40 the fastest splits of 5120
    points were 10 x 512 and 40 x 128), and n1 at least 2; zero if n2 would be under
    16. */
inline std::size_t split_factor(const std::size_t half) {
  std::size_t second = 1;
  while (second < 512 && half % (2 * second) == 0)
    second *= 2;
  return second >= 16 && half / second >= 2 ? half / second : 0;
}

/** exp(2 pi i sign j / n) for j in [0, count), rounded once from long double. */
inline std::vector<Kokkos::complex<double>>
unit_roots(const std::size_t n, const std::size_t count, const int sign) {
  constexpr long double two_pi = 6.283185307179586476925286766559005768L;
  std::vector<Kokkos::complex<double>> roots(count);
  for (std::size_t j = 0; j < count; ++j) {
    const long double angle =
        two_pi * static_cast<long double>(j % n) / static_cast<long double>(n) * sign;
    roots[j] = Kokkos::complex<double>(static_cast<double>(std::cos(angle)),
                                       static_cast<double>(std::sin(angle)));
  }
  return roots;
}

// The split plan's kernels over its rows of m = n1 n2 complex points: z_j is row
// sample pair (2j, 2j + 1), the four-step transform's input of point j = n2 j1 + j2
// lies at (r n2 + j2, j1) of the first pass and its output of point k = k1 + n1 k2
// at (r n1 + k1, k2) of the second.
template <class Real, class Rows> struct SplitPackRows {
  using Value = typename Rows::non_const_value_type;
  Real real;
  Rows first;
  std::size_t m, n1, n2;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    const std::size_t r = i / m, j = i % m, j2 = j / n1, j1 = j % n1;
    const std::size_t point = n2 * j1 + j2;
    first(r * n2 + j2, j1) = Value(real(r, 2 * point), real(r, 2 * point + 1));
  }
};

template <class Rows, class Table> struct SplitTwiddle {
  Rows first, second;
  Table roots;
  std::size_t m, n1, n2;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    const std::size_t r = i / m, j = i % m, k1 = j / n2, j2 = j % n2;
    second(r * n1 + k1, j2) = first(r * n2 + j2, k1) * roots((j2 * k1) % m);
  }
};

// X_k = (Z_k + conj Z_{m-k}) / 2 - i W^k (Z_k - conj Z_{m-k}) / 2, W = exp(-pi i / m).
template <class Rows, class Table, class Spectrum> struct SplitForwardSpectrum {
  using Value = typename Spectrum::non_const_value_type;
  Rows second;
  Table roots;
  Spectrum spectrum;
  std::size_t m, n1, bins;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    const std::size_t r = i / bins, k = i % bins, a = k % m, b = (m - k) % m;
    const auto za = second(r * n1 + a % n1, a / n1);
    const auto zb = Kokkos::conj(second(r * n1 + b % n1, b / n1));
    const auto even = 0.5 * (za + zb);
    const auto turned = roots(k) * (0.5 * (za - zb));
    spectrum(r, k) = Value(even.real() + turned.imag(), even.imag() - turned.real());
  }
};

// Z_k = (X_k + conj X_{m-k}) + i conj(W^k) (X_k - conj X_{m-k}), the imaginary parts
// of X_0 and X_m dropped as cuFFT's and FFTW's inverses do.
template <class Spectrum, class Table, class Rows> struct SplitInverseRows {
  using Value = typename Rows::non_const_value_type;
  Spectrum spectrum;
  Table roots;
  Rows first;
  std::size_t m, n1, n2;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    const std::size_t r = i / m, j = i % m, j2 = j / n1, j1 = j % n1;
    const std::size_t k = n2 * j1 + j2;
    auto xa = spectrum(r, k);
    auto xb = Kokkos::conj(spectrum(r, m - k));
    if (k == 0) {
      xa = Value(xa.real(), 0.0);
      xb = Value(xb.real(), 0.0);
    }
    const auto even = xa + xb;
    const auto turned = Kokkos::conj(roots(k)) * (xa - xb);
    first(r * n2 + j2, j1) =
        Value(even.real() - turned.imag(), even.imag() + turned.real());
  }
};

template <class Rows, class Real> struct SplitUnpackRows {
  Rows second;
  Real real;
  std::size_t m, n1;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    const std::size_t r = i / m, point = i % m;
    const auto z = second(r * n1 + point % n1, point / n1);
    real(r, 2 * point) = z.real();
    real(r, 2 * point + 1) = z.imag();
  }
};
} // namespace detail

/** A batched transform of real rows -- forward (`Input` real) or inverse -- along
    the last axis, unnormalised, as `KokkosFFT::Plan` gives it.

    A plan of at most `split_real_rows_max` contiguous rows of n = 2 m samples, m =
    n1 n2 (`detail::split_factor`), on a build with `split_real_rows_enabled`,
   transforms them as m-point complex rows (sample pairs as one complex point, the
   standard half-length real transform) by the four-step method: a batched transform of
   n1 points per (row, j2), a twiddle by exp(-+2 pi i j2 k1 / m) and transposition, a
    batched transform of n2 points per (row, k1). Its five kernels and two vendor
    transforms spread a row over the device; the sums are the same DFT in another
    association, so they agree with the vendor's to rounding. Its two work buffers of
    rows x m points and its two root tables are allocated with the plan, so
    the workspace's ledger counts them in its plans' device growth. Any other plan is
    Kokkos-FFT's own. */
template <class Execution, class Input, class Output> class RealRowPlan {
  static constexpr bool forward =
      std::is_same_v<typename Input::non_const_value_type, double>;
  using Real = std::conditional_t<forward, Input, Output>;
  using Spectrum = std::conditional_t<forward, Output, Input>;
  using Complex = typename Spectrum::non_const_value_type;
  using Rows =
      Kokkos::View<Complex**, Kokkos::LayoutRight, typename Spectrum::memory_space>;
  using Buffer = Kokkos::View<Complex*, typename Spectrum::memory_space>;
  using Direct = KokkosFFT::Plan<Execution, Input, Output>;
  using Pass = KokkosFFT::Plan<Execution, Rows, Rows>;

  Execution execution_;
  std::unique_ptr<Direct> direct_;
  std::size_t rows_ = 0, m_ = 0, n1_ = 0, n2_ = 0;
  Buffer first_buffer_, second_buffer_, inner_roots_, outer_roots_;
  Rows first_, first_out_, second_, second_out_;
  std::unique_ptr<Pass> first_pass_, second_pass_;

  static std::size_t split_for(const Real& real, const Spectrum& spectrum,
                               const int axis) {
    const std::size_t samples = real.extent(1);
    if (!split_real_rows_enabled || axis != -1 || real.extent(0) == 0 ||
        real.extent(0) > split_real_rows_max || samples % 2 != 0 ||
        spectrum.extent(0) != real.extent(0) || spectrum.extent(1) != samples / 2 + 1 ||
        real.stride(1) != 1 || spectrum.stride(1) != 1 || real.stride(0) != samples ||
        spectrum.stride(0) != spectrum.extent(1))
      return 0;
    return detail::split_factor(samples / 2);
  }

  template <class Functor>
  void run(const char* label, const std::size_t count, const Functor& functor) const {
    launch::parallel_for(label, Kokkos::RangePolicy<Execution>(execution_, 0, count),
                         functor);
  }

public:
  RealRowPlan(const Execution& execution, const Input& input, const Output& output,
              const KokkosFFT::Direction direction, const int axis)
      : execution_(execution) {
    const Real& real = [&]() -> const Real& {
      if constexpr (forward)
        return input;
      else
        return output;
    }();
    const Spectrum& spectrum = [&]() -> const Spectrum& {
      if constexpr (forward)
        return output;
      else
        return input;
    }();
    n1_ = split_for(real, spectrum, axis);
    if (n1_ == 0) {
      direct_ = std::make_unique<Direct>(execution, input, output, direction, axis);
      return;
    }
    rows_ = real.extent(0);
    m_ = real.extent(1) / 2;
    n2_ = m_ / n1_;
    first_buffer_ = Buffer("solprop_native.fft.split_rows_first", rows_ * m_);
    second_buffer_ = Buffer("solprop_native.fft.split_rows_second", rows_ * m_);
    first_ = Rows(first_buffer_.data(), rows_ * n2_, n1_);
    first_out_ = Rows(second_buffer_.data(), rows_ * n2_, n1_);
    second_ = Rows(first_buffer_.data(), rows_ * n1_, n2_);
    second_out_ = Rows(second_buffer_.data(), rows_ * n1_, n2_);
    const int sign = forward ? -1 : 1;
    const auto copy_roots = [&](Buffer& roots, const char* label, const std::size_t n,
                                const std::size_t count, const int root_sign) {
      const auto host = detail::unit_roots(n, count, root_sign);
      roots = Buffer(label, count);
      const Kokkos::View<const Complex*, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          source(host.data(), count);
      Kokkos::deep_copy(execution, roots, source);
    };
    copy_roots(inner_roots_, "solprop_native.fft.split_rows_inner_roots", m_, m_, sign);
    copy_roots(outer_roots_, "solprop_native.fft.split_rows_outer_roots", 2 * m_,
               m_ + 1, -1);
    execution.fence();
    first_pass_ = std::make_unique<Pass>(execution, first_, first_out_, direction, -1);
    second_pass_ =
        std::make_unique<Pass>(execution, second_, second_out_, direction, -1);
  }

  void execute(const Input& input, const Output& output,
               const KokkosFFT::Normalization normalization) const {
    if (direct_) {
      graph::execute(*direct_, input, output, normalization);
      return;
    }
    const std::size_t real = forward ? input.extent(1) : output.extent(1);
    const std::size_t bins = forward ? output.extent(1) : input.extent(1);
    if (normalization != KokkosFFT::Normalization::none || input.extent(0) != rows_ ||
        output.extent(0) != rows_ || real != 2 * m_ || bins != m_ + 1)
      throw std::invalid_argument("a split real-row plan's views do not match it");
    const std::size_t points = rows_ * m_;
    if constexpr (forward) {
      run("solprop_native.fft.split_rows_pack", points,
          detail::SplitPackRows<Input, Rows>{input, first_, m_, n1_, n2_});
    } else {
      run("solprop_native.fft.split_rows_inverse", points,
          detail::SplitInverseRows<Input, Buffer, Rows>{input, outer_roots_, first_, m_,
                                                        n1_, n2_});
    }
    graph::execute(*first_pass_, first_, first_out_, KokkosFFT::Normalization::none);
    run("solprop_native.fft.split_rows_twiddle", points,
        detail::SplitTwiddle<Rows, Buffer>{first_out_, second_, inner_roots_, m_, n1_,
                                           n2_});
    graph::execute(*second_pass_, second_, second_out_, KokkosFFT::Normalization::none);
    if constexpr (forward) {
      run("solprop_native.fft.split_rows_spectrum", rows_ * (m_ + 1),
          detail::SplitForwardSpectrum<Rows, Buffer, Output>{second_out_, outer_roots_,
                                                             output, m_, n1_, m_ + 1});
    } else {
      run("solprop_native.fft.split_rows_unpack", points,
          detail::SplitUnpackRows<Rows, Output>{second_out_, output, m_, n1_});
    }
  }
};

namespace graph {
/** A real-row plan's transform: its own operations, each recorded in the calling
    thread's device graph region, if any. */
template <class Execution, class Input, class Output>
void execute(const RealRowPlan<Execution, Input, Output>& plan, const Input& input,
             const Output& output, const KokkosFFT::Normalization normalization) {
  plan.execute(input, output, normalization);
}
} // namespace graph
#endif

} // namespace solprop_native
