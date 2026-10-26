// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/precision.hpp"

#include <KokkosFFT.hpp>
#include <Kokkos_Core.hpp>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace solprop_native {

/**
 * One binary32 transform round trip on the execution space through the plans the
 * reduced policies use (Kokkos-FFT on FFTW's `fftwf_`, cuFFT's C2C/R2C, rocFFT's
 * binary32 path), for the transform characterisation of solprop SEP 36 section 6.3.
 *
 * `values` holds `rows x n` binary64 samples, each rounded once to binary32 (real
 * inputs read the real parts); the result is the forward spectrum (`n` bins, or
 * `n / 2 + 1` for a real input) and the backward transform of that spectrum, both
 * unnormalised and widened exactly.
 */
inline std::pair<std::vector<Complex>, std::vector<Complex>>
binary32_round_trip(const std::complex<double>* values, const std::size_t rows,
                    const std::size_t n, const bool real) {
  using Value = Kokkos::complex<float>;
  using ComplexRows = Kokkos::View<Value**, Layout, MemorySpace>;
  using RealRows = Kokkos::View<float**, Layout, MemorySpace>;
  const ExecutionSpace space;
  constexpr int axis = -1;
  const auto bins = real ? n / 2 + 1 : n;
  ComplexRows spectrum("solprop_native.fft_probe.spectrum", rows, bins);
  std::vector<Complex> forward(rows * bins), backward(rows * n);
  if (real) {
    RealRows input("solprop_native.fft_probe.input", rows, n),
        output("solprop_native.fft_probe.output", rows, n);
    auto host = Kokkos::create_mirror_view(input);
    for (std::size_t index = 0; index < rows * n; ++index)
      host.data()[index] = static_cast<float>(values[index].real());
    Kokkos::deep_copy(space, input, host);
    KokkosFFT::Plan<ExecutionSpace, RealRows, ComplexRows> to(
        space, input, spectrum, KokkosFFT::Direction::forward, axis);
    KokkosFFT::execute(to, input, spectrum, KokkosFFT::Normalization::none);
    const auto host_spectrum =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, spectrum);
    for (std::size_t index = 0; index < rows * bins; ++index)
      forward[index] = Complex(host_spectrum.data()[index]);
    KokkosFFT::Plan<ExecutionSpace, ComplexRows, RealRows> back(
        space, spectrum, output, KokkosFFT::Direction::backward, axis);
    KokkosFFT::execute(back, spectrum, output, KokkosFFT::Normalization::none);
    const auto host_output =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
    for (std::size_t index = 0; index < rows * n; ++index)
      backward[index] = Complex(static_cast<double>(host_output.data()[index]), 0.0);
    return {forward, backward};
  }
  ComplexRows input("solprop_native.fft_probe.input", rows, n),
      output("solprop_native.fft_probe.output", rows, n);
  auto host = Kokkos::create_mirror_view(input);
  for (std::size_t index = 0; index < rows * n; ++index)
    host.data()[index] = Value(static_cast<float>(values[index].real()),
                               static_cast<float>(values[index].imag()));
  Kokkos::deep_copy(space, input, host);
  KokkosFFT::Plan<ExecutionSpace, ComplexRows, ComplexRows> to(
      space, input, spectrum, KokkosFFT::Direction::forward, axis);
  KokkosFFT::execute(to, input, spectrum, KokkosFFT::Normalization::none);
  const auto host_spectrum =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, spectrum);
  for (std::size_t index = 0; index < rows * n; ++index)
    forward[index] = Complex(host_spectrum.data()[index]);
  KokkosFFT::Plan<ExecutionSpace, ComplexRows, ComplexRows> back(
      space, spectrum, output, KokkosFFT::Direction::backward, axis);
  KokkosFFT::execute(back, spectrum, output, KokkosFFT::Normalization::none);
  const auto host_output =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  for (std::size_t index = 0; index < rows * n; ++index)
    backward[index] = Complex(host_output.data()[index]);
  return {forward, backward};
}

} // namespace solprop_native
