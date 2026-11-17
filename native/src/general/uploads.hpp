// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Setup-time copies into memory-space views: upload_span (a host span, through the
// crossing ring when it is large, staged first when it is misaligned) and copy_vector
// (a host vector, through a mirror). A part of kernels.hpp, included through it.
#ifdef SOLPROP_NATIVE_STRICT_TU
#error "a strict translation unit must not include general/uploads.hpp"
#endif

#include "solprop_native/runtime/crossing.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/graph.hpp"
#include "views.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace solprop_native::general {

// A host span as an unmanaged view of a state-shaped memory-space view, for one copy
// between them: no staging mirror, no element loop. A span whose data is not aligned
// as `Complex` (a NumPy buffer need not be) is staged in an aligned vector first.
inline void upload_span(const ExecutionSpace& execution_space, crossing::Ring& ring,
                        const ComplexView2& destination,
                        const std::span<const Complex> source) {
  if (destination.size() != source.size())
    throw std::invalid_argument("fixed problem array has the wrong extent");
  if (reinterpret_cast<std::uintptr_t>(source.data()) % alignof(Complex) != 0) {
    const std::vector<Complex> aligned(source.begin(), source.end());
    upload_span(execution_space, ring, destination, aligned);
    execution_space.fence("solprop_native unaligned upload");
    return;
  }
  if (crossing::takes_ring(source.size_bytes())) {
    execution_space.fence("solprop_native upload");
    ring.from_host(destination.data(), source.data(), source.size_bytes());
    return;
  }
  const Kokkos::View<const Complex**, Layout, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host(source.data(), destination.extent(0), destination.extent(1));
  graph::deep_copy(execution_space, destination, host);
}

template <class View, class Value>
void copy_vector(const ExecutionSpace& execution_space, const View& destination,
                 const std::vector<Value>& source) {
  if (destination.size() != source.size())
    throw std::invalid_argument("fixed problem array has the wrong extent");
  auto mirror = Kokkos::create_mirror_view(destination);
  auto* data = mirror.data();
  for (std::size_t index = 0; index < source.size(); ++index)
    data[index] = source[index];
  graph::deep_copy(execution_space, destination, mirror);
}

} // namespace solprop_native::general
