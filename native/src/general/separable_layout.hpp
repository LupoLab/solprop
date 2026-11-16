// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// A separable geometry's descriptors: TransverseLayout (where a column lives in the
// transverse scratch, on free space or the hybrid planar kind), PhysicalSpectrum (a
// node's physical spectrum after the inverse transverse plan), RegisterIndex (where
// the low-storage registers hold an element), TransverseChunk (one chunk of a
// response pass) and SpatialCoefficients (the longitudinal root, the linear operator
// and the prefactor at a column and bin). A part of kernels.hpp, included through it.
#ifdef SOLPROP_NATIVE_STRICT_TU
#error "a strict translation unit must not include general/separable_layout.hpp"
#endif

#include "solprop_native/runtime/execution.hpp"
#include "traversal.hpp"
#include "views.hpp"

#include <cstddef>

namespace solprop_native::general {

// Where a separable geometry's column (or a temporal row, which is the same index)
// lives in the transverse scratch. Free space: (.., n_y, n_x), x fastest. The hybrid
// planar kind: (.., n_p, n_y) with the y transform trailing, the channel fastest in
// the column index p + n_p * iy.
struct TransverseLayout {
  std::size_t inner{};
  bool channel_fastest{};
  // Elements per transverse plane of the row-major scratch `(planes, ., .)`.
  std::size_t plane{};
  // A column's offset within a plane: the column itself on free space (y, x with x
  // fastest), `p * n_y + iy` on the planar kind.
  KOKKOS_INLINE_FUNCTION std::size_t offset(const std::size_t column) const {
    return channel_fastest ? (column % inner) * (plane / inner) + column / inner
                           : column;
  }
  template <class View>
  KOKKOS_INLINE_FUNCTION auto& cell(const View& scratch, const std::size_t plane_index,
                                    const std::size_t column) const {
    return scratch.data()[plane_index * plane + offset(column)];
  }
  KOKKOS_INLINE_FUNCTION std::size_t first(const std::size_t column) const {
    return channel_fastest ? column % inner : column / inner;
  }
  KOKKOS_INLINE_FUNCTION std::size_t second(const std::size_t column) const {
    return channel_fastest ? column / inner : column % inner;
  }
};

// A node's physical spectrum in a transverse volume after the inverse transverse plan:
// on free space the node's own cell; on the hybrid planar kind the slab rule over its y
// row's channels, without the temporal pad scale the planar synthesis carries, since
// the statistics read the coarse field (solprop's `collect_spatial`).
struct PhysicalSpectrum {
  TransverseLayout layout{};
  RealView2 synthesis;
  std::size_t n_x{}, n_p{};
  double crop_scale = 1.0;
  bool slab = false;
  template <class View>
  KOKKOS_INLINE_FUNCTION Complex read(const View& source, const std::size_t plane,
                                      const std::size_t node) const {
    if (!slab)
      return layout.cell(source, plane, node);
    const auto ix = node % n_x;
    const auto rows = (node / n_x) * n_p;
    Complex sum{};
    for (std::size_t channel = 0; channel < n_p; ++channel)
      sum += synthesis(ix, channel) *
             source(plane, layout.first(rows + channel), layout.second(rows + channel));
    return crop_scale * sum;
  }
};

// Where a separable route's low-storage registers (the accumulator, the stage register
// and the flow table) hold the element (column, omega). On the host, at the transverse
// scratch's own position: a fold then reads them in the scratch's order beside it, and
// only the state crosses layouts. On a device, at the state's (column-major).
struct RegisterIndex {
  TransverseLayout layout{};
  std::size_t n_pol{}, n{};
  bool plane{};
  // One polarisation: a column is its mode, without a division (a flag of its own,
  // which the compiler cannot fold back into `column / 1`).
  bool scalar{};
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // A stage of the volume route on a device (`volume_order_`): the scratch is `(y, x,
  // omega)`, the state's and the registers' own order, so nothing crosses.
  bool volume{};
#endif
  KOKKOS_INLINE_FUNCTION std::size_t mode(const std::size_t column) const {
    return scalar ? column : column / n_pol;
  }
  KOKKOS_INLINE_FUNCTION std::size_t pol(const std::size_t column) const {
    return scalar ? 0 : column % n_pol;
  }
  // The transverse scratch's position of the element, on any build.
  KOKKOS_INLINE_FUNCTION std::size_t cell(const std::size_t column,
                                          const std::size_t omega) const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if (volume)
      return column * n + omega;
#endif
    return (omega * n_pol + pol(column)) * layout.plane + layout.offset(mode(column));
  }
  KOKKOS_INLINE_FUNCTION std::size_t operator()(const std::size_t column,
                                                const std::size_t omega) const {
    return plane ? cell(column, omega) : column * n + omega;
  }
};

// One chunk of a separable geometry's response pass: whole transverse units (a node
// on free space, a y row of n_p columns and n_x nodes on the hybrid planar kind), its
// temporal rows (state columns) and its physical nodes.
struct TransverseChunk {
  std::size_t first_row{}, rows{}, first_node{}, nodes{};
};

struct SpatialCoefficients {
  ComplexView square, square_low, numerator, clamped;
  RealView transverse, transverse_low, frame, spectral_rate, user_rate, absorber;
  // A transverse collar's power rate per column (solprop kernel schema 31).
  RealView transverse_rate;
  // The hybrid planar kind's wall terms, (channels, n_omega); empty on free space.
  ComplexView2 kappa;
  std::size_t channels{};
  double evanescent{};
  // The longitudinal square q of a column and bin and its principal root, solprop's
  // `longitudinal_root`: the one evaluation the operator and the plane measure share.
  KOKKOS_INLINE_FUNCTION Complex longitudinal_square(const std::size_t mode,
                                                     const std::size_t omega) const {
    Complex longitudinal =
        (square(omega) - transverse(mode)) + (square_low(omega) - transverse_low(mode));
    // q = k^2 eps_co - k_y^2 - kappa_p^2, the wall term subtracted last as in solprop's
    // separable_values (kernel schema 15), so a free-space column's arithmetic is
    // untouched.
    if (channels != 0)
      longitudinal = longitudinal - kappa(mode % channels, omega);
    return longitudinal;
  }
  // A real q (a lossless medium) takes the real root, the same bits as the complex
  // root of q + 0i at a fraction of its cost (SEP 28): sqrt(q) or i sqrt(-q).
  static KOKKOS_INLINE_FUNCTION Complex root_of(const Complex longitudinal) {
    return longitudinal.imag() == 0.0
               ? (longitudinal.real() >= 0.0
                      ? Complex(Kokkos::sqrt(longitudinal.real()), 0.0)
                      : Complex(0.0, Kokkos::sqrt(-longitudinal.real())))
               : Kokkos::sqrt(longitudinal);
  }
  // The conjugate of the plane's dimensionless forward admittance `c k_z / omega`
  // (solprop SEP 34, `axial_admittance`) on an active propagating bin at positive
  // frequency, zero elsewhere: the factor an internal (conjugate) spectrum takes to
  // its magnetic closure. Its real part weighs energy and spectra.
  KOKKOS_INLINE_FUNCTION Complex closing(const std::size_t mode,
                                         const std::size_t omega, const bool active,
                                         const double frequency) const {
    const Complex longitudinal = longitudinal_square(mode, omega);
    if (!active || !(longitudinal.real() > 0.0) || !(frequency > 0.0))
      return Complex{};
    const Complex root = root_of(longitudinal);
    constexpr double speed_of_light = 299792458.0;
    return Complex(speed_of_light * root.real(), -(speed_of_light * root.imag())) /
           frequency;
  }
  // `value`'s prefactor alone, the same arithmetic: what a fold whose flows are
  // tabulated reads of the element.
  KOKKOS_INLINE_FUNCTION Complex prefactor(const std::size_t mode,
                                           const std::size_t omega,
                                           const bool active) const {
    if (!active)
      return Complex{};
    const Complex longitudinal = longitudinal_square(mode, omega);
    if (!(longitudinal.real() > 0.0))
      return clamped(omega);
    const Complex root = root_of(longitudinal);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    return numerator(omega) * (1.0 / root.real());
#else
    return numerator(omega) / root.real();
#endif
  }
  KOKKOS_INLINE_FUNCTION SpatialElement value(const std::size_t mode,
                                              const std::size_t omega,
                                              const bool active) const {
    const Complex longitudinal = longitudinal_square(mode, omega);
    const Complex root = root_of(longitudinal);
    const bool propagating = longitudinal.real() > 0.0;
    const double attenuation =
        propagating ? root.imag() : Kokkos::min(Kokkos::abs(root.imag()), evanescent);
    // The transverse collar's rate joins the attenuation inside the active band, as
    // in solprop's separable_values (SEP 22, Q6); a zero rate changes no bit.
    const Complex linear = active ? Complex(-attenuation - 0.5 * transverse_rate(mode),
                                            -(root.real() - frame(omega)))
                                  : Complex{};
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    // One binary64 reciprocal and two products in place of two divisions, at every
    // element of the stage passes: a device's division is several fused steps (at
    // most an ulp from the quotients).
    const Complex prefactor =
        propagating ? numerator(omega) * (1.0 / root.real()) : clamped(omega);
#else
    const Complex prefactor =
        propagating ? numerator(omega) / root.real() : clamped(omega);
#endif
    return {.linear = linear - 0.5 * spectral_rate(omega),
            .prefactor = active ? prefactor : Complex{}};
  }
};

} // namespace solprop_native::general
