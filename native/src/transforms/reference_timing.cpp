// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// solprop SEP 28 section 4.2's reference timings (reference_timing.hpp): a rank-3
// Kokkos-FFT forward-then-backward pair on the FFT execution space, and a scaled copy
// between two complex volumes first touched in parallel, each timed per repeat after
// untimed warm-up repeats.

#include "solprop_native/transforms/reference_timing.hpp"

#include "solprop_native/runtime/cpu_execution.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/launch.hpp"

#include <KokkosFFT.hpp>
#include <Kokkos_Core.hpp>

#include <chrono>
#include <complex>
#include <cstddef>
#include <vector>

namespace solprop_native {

std::vector<double> reference_fft3_pairs(const std::complex<double>* values,
                                         const std::size_t planes,
                                         const std::size_t n_y, const std::size_t n_x,
                                         const std::size_t warm,
                                         const std::size_t pairs) {
  using Volume = Kokkos::View<Complex***, Layout, MemorySpace>;
  using Plan3 = KokkosFFT::Plan<ExecutionSpace, Volume, Volume, 3>;
  const ExecutionSpace space = execution_instance(thread_policy().fft);
  Volume volume(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                   "solprop_native.reference.volume"),
                planes, n_y, n_x);
  const Kokkos::View<const Complex***, Layout, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host(reinterpret_cast<const Complex*>(values), planes, n_y, n_x);
  auto mirror = Kokkos::create_mirror_view(Kokkos::HostSpace{}, volume);
  Kokkos::deep_copy(mirror, host);
  Kokkos::deep_copy(volume, mirror);
  const KokkosFFT::axis_type<3> axes{-3, -2, -1};
  const Plan3 forward(space, volume, volume, KokkosFFT::Direction::forward, axes);
  const Plan3 backward(space, volume, volume, KokkosFFT::Direction::backward, axes);
  std::vector<double> seconds;
  seconds.reserve(pairs);
  for (std::size_t index = 0; index < warm + pairs; ++index) {
    space.fence();
    const auto start = std::chrono::steady_clock::now();
    KokkosFFT::execute(forward, volume, volume, KokkosFFT::Normalization::none);
    KokkosFFT::execute(backward, volume, volume, KokkosFFT::Normalization::none);
    space.fence();
    if (index >= warm)
      seconds.push_back(
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
              .count());
  }
  return seconds;
}

std::vector<double> reference_stream_copies(const std::size_t elements,
                                            const std::size_t warm,
                                            const std::size_t repeats) {
  const ExecutionSpace space = execution_instance(thread_policy().kernels);
  const ComplexView source(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                              "solprop_native.reference.source"),
                           elements);
  const ComplexView target(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                              "solprop_native.reference.target"),
                           elements);
  using Policy = Kokkos::RangePolicy<ExecutionSpace>;
  // First touch in parallel, on the threads that copy, so that each page lands where
  // the copy reads or writes it.
  launch::parallel_for(
      "solprop_native.reference.first_touch", Policy(space, 0, elements),
      KOKKOS_LAMBDA(const std::size_t index) {
        source(index) = Complex(static_cast<double>(index % 7U), 1.0);
        target(index) = Complex{};
      });
  std::vector<double> seconds;
  seconds.reserve(repeats);
  const double scale = 0.5;
  for (std::size_t index = 0; index < warm + repeats; ++index) {
    space.fence();
    const auto start = std::chrono::steady_clock::now();
    launch::parallel_for(
        "solprop_native.reference.scaled_copy", Policy(space, 0, elements),
        KOKKOS_LAMBDA(const std::size_t element) {
          target(element) = scale * source(element);
        });
    space.fence();
    if (index >= warm)
      seconds.push_back(
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
              .count());
  }
  return seconds;
}

} // namespace solprop_native
