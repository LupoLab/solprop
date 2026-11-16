// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "memory.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif
#ifdef __APPLE__
#include <malloc/malloc.h>
#elif defined(__GLIBC__)
#include <dlfcn.h>
#endif

namespace solprop_native::general {

std::size_t heap_bytes() {
#ifdef __APPLE__
  malloc_statistics_t statistics{};
  malloc_zone_statistics(nullptr, &statistics);
  return statistics.size_in_use;
#elif defined(__GLIBC__)
  // mallinfo2 is glibc 2.33 or later; a wheel built against an older glibc (the
  // manylinux_2_28 baseline) finds it at run time when the system provides it. Its
  // layout is ten size_t counters, in glibc's declared order.
  struct Mallinfo2 {
    std::size_t arena, ordblks, smblks, hblks, hblkhd, usmblks, fsmblks, uordblks,
        fordblks, keepcost;
  };
  using Query = Mallinfo2 (*)();
  static const auto query = reinterpret_cast<Query>(dlsym(RTLD_DEFAULT, "mallinfo2"));
  if (query == nullptr)
    return 0;
  const auto statistics = query();
  return statistics.uordblks + statistics.hblkhd;
#else
  // Neither allocator interface exists (musl, Windows): the ledger row then measures
  // nothing, and construction must still succeed.
  return 0;
#endif
}
std::size_t device_free_bytes() {
#ifdef KOKKOS_ENABLE_CUDA
  std::size_t available{}, total{};
  if (cudaMemGetInfo(&available, &total) != cudaSuccess)
    throw std::runtime_error("cudaMemGetInfo failed during FFT planning");
  return available;
#elif defined(KOKKOS_ENABLE_HIP)
  std::size_t available{}, total{};
  if (hipMemGetInfo(&available, &total) != hipSuccess)
    throw std::runtime_error("hipMemGetInfo failed during FFT planning");
  return available;
#else
  return 0;
#endif
}

} // namespace solprop_native::general
