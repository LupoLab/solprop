// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The general workspace's view and plan types: the view aliases (ComplexView2/3,
// RealView2/3, ByteView, IndexView), the FFT plan aliases (the real row, Raman,
// complex, transverse, volume, slab and strip plans; a device's differ), and the
// plasma current streams that own their views and plans (EnvelopeStream,
// FieldCurrentStream). Two small helpers sit among them, in kernels.hpp's order:
// capture_anchor, which names the views a kernel lambda captures, and
// write_raman_drive, a node's Raman drive rows from its field. A part of kernels.hpp,
// included through it.
#ifdef SOLPROP_NATIVE_STRICT_TU
#error "a strict translation unit must not include general/views.hpp"
#endif

#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/transforms/fft_plan.hpp"
#include "solprop_native/transforms/raman_fft.hpp"

#include <KokkosFFT.hpp>

#include <cstddef>
#include <memory>

namespace solprop_native::general {

template <class... Ts>
KOKKOS_INLINE_FUNCTION void capture_anchor(const Ts&... /*unused*/) {}

using ComplexView2 = Kokkos::View<Complex**, Layout, MemorySpace>;
using ComplexView3 = Kokkos::View<Complex***, Layout, MemorySpace>;
using RealView2 = Kokkos::View<double**, Layout, MemorySpace>;
using RealView3 = Kokkos::View<double***, Layout, MemorySpace>;

template <int Components, bool RealField>
KOKKOS_INLINE_FUNCTION void
write_raman_drive(const RealView2& drive, const std::size_t node,
                  const std::size_t time, const Complex x, const Complex y,
                  const std::size_t count) {
  const double xx = RealField ? x.real() * x.real()
                              : 0.5 * (x.real() * x.real() + x.imag() * x.imag());
  if constexpr (Components == 1) {
    drive(node, time) = xx;
  } else {
    const double yy = RealField ? y.real() * y.real()
                                : 0.5 * (y.real() * y.real() + y.imag() * y.imag());
    drive(count * node, time) = xx + yy;
    if (count == 3) {
      drive(count * node + 1, time) = xx - yy;
      drive(count * node + 2, time) = RealField
                                          ? 2.0 * x.real() * y.real()
                                          : x.real() * y.real() + x.imag() * y.imag();
    }
  }
}
using ByteView = Kokkos::View<unsigned char*, Layout, MemorySpace>;
using IndexView = Kokkos::View<std::size_t*, Layout, MemorySpace>;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
// A device splits a few long real rows over its multiprocessors (`RealRowPlan`).
using InverseRealPlan = RealRowPlan<ExecutionSpace, ComplexView2, RealView2>;
using ForwardRealPlan = RealRowPlan<ExecutionSpace, RealView2, ComplexView2>;
#else
using InverseRealPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView2, RealView2>;
using ForwardRealPlan = KokkosFFT::Plan<ExecutionSpace, RealView2, ComplexView2>;
#endif
using RamanForwardPlan = RamanFftPlan<ExecutionSpace, RealView2, ComplexView2>;
using RamanInversePlan = RamanFftPlan<ExecutionSpace, ComplexView2, RealView2>;
using ComplexPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView2, ComplexView2>;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
// A device's plans of the transverse scratch run one after another, so they share one
// work area (`transform_work_`) in place of holding one each (solprop #529). A
// Kerr-only free-space envelope takes the stage's time axis with the plane's in one
// rank-3 transform (`volume_kerr_`).
using TransversePlan = KokkosFFT::DynPlan<ExecutionSpace, ComplexView3, ComplexView3>;
using VolumePlan = TransversePlan;
#else
using TransversePlan = KokkosFFT::Plan<ExecutionSpace, ComplexView3, ComplexView3, 2>;
using VolumePlan = KokkosFFT::Plan<ExecutionSpace, ComplexView3, ComplexView3, 3>;
#endif
// The hybrid planar kind transforms y alone, the trailing axis of (bins, n_p, n_y).
using SlabPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView3, ComplexView3, 1>;
// A requested collar's padded transform, along the last axis of a batch (solprop SEP
// 44).
using StripPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView2, ComplexView2, 1>;

// Each current keeps its own chunk accumulator, including stacked embedded
// quadrature. Sharing the latter would mix currents between partial chunks.
struct EnvelopeStream {
  ComplexView2 node, projected, coarse, stacked, spectrum, coarse_spectrum;
  std::unique_ptr<ComplexPlan> forward, coarse_forward;
  [[nodiscard]] std::size_t bytes() const {
    return sizeof(Complex) *
           (node.size() + projected.size() + coarse.size() + stacked.size() +
            spectrum.size() + coarse_spectrum.size());
  }
};

// Real current products own their projections, including an embedded quadrature.
struct FieldCurrentStream {
  RealView2 node, projected, coarse, stacked;
  ComplexView2 spectrum, coarse_spectrum;
  std::unique_ptr<ForwardRealPlan> forward, coarse_forward;
  [[nodiscard]] std::size_t bytes() const {
    return sizeof(double) *
               (node.size() + projected.size() + coarse.size() + stacked.size()) +
           sizeof(Complex) * (spectrum.size() + coarse_spectrum.size());
  }
};

} // namespace solprop_native::general
