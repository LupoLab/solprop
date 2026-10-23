// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/core.hpp"
#include "solprop_native/fp.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

/*
 * The far-field statistics of solprop SEP 23 on a Cartesian or line grid: one banded
 * pass over the stored transverse spectrum that forms the beam and axial-flux spectra,
 * the four absorbed-power channels, the region spectra, each aperture's collected
 * spectrum and on-axis amplitude, and the on-axis amplitude itself (the `W = 1` slot,
 * which `on_axis_spectrum` reads). It mirrors solprop's `numba_cpu` pass: the slot
 * layout of `solprop.backends.spatial.far_field_layout`, the band edges of
 * `far_field_bands` (the same units and budget), the radius of `mask_plane_radius`
 * compiled without contraction, the zeros of `aperture_weight` and the root of
 * `longitudinal_root`. Each band sums its columns in ascending order and the bands are
 * then summed in ascending order, so every value is the `numba_cpu` pass's sum in the
 * same association; agreement between them rests on summation order alone (solprop
 * SEP 25 compares them within a rounding bound, never by bits).
 */

namespace solprop_native {

/** solprop's `FAR_FIELD_BAND_BUDGET`: bytes the band partials may hold. */
inline constexpr std::size_t far_field_band_budget = std::size_t{128} << 20U;

/** solprop's `LINE_BLOCK`: columns per unit of a line's pass; a plane's is a row. */
inline constexpr std::size_t far_field_line_block = 64;

/** The constants per aperture before its scale row: centre x and y (m), half-width
    (m), reciprocal edge width (1/m), edge or order, and distance (m). */
inline constexpr std::size_t far_field_aperture_constants = 6;

/** The absorbed-power channels: spectral, transverse, evanescent and material. */
inline constexpr std::size_t far_field_absorbed_channels = 4;

/** Where `tanh` saturates: correctly rounded, `tanh(t)` is exactly 1 for `t >= 22`. */
inline constexpr double far_field_tanh_saturation = 22.0;

/** The vacuum speed of light, m/s, exact in the SI; the obliquity `c Re k_z / omega`.
 */
inline constexpr double far_field_speed_of_light = 299792458.0;

/** The flags of `aperture_statistics`, from 32: 1 to 16 are `spatial_statistics`'. */
enum FarFieldFlag : std::uint64_t {
  far_field_beam = 1ULL << 5U,
  far_field_poynting = 1ULL << 6U,
  far_field_absorbed = 1ULL << 7U,
  far_field_regions = 1ULL << 8U,
  far_field_collected = 1ULL << 9U,
  far_field_amplitude = 1ULL << 10U,
  far_field_axis = 1ULL << 11U,
};

inline constexpr std::uint64_t far_field_flags =
    far_field_beam | far_field_poynting | far_field_absorbed | far_field_regions |
    far_field_collected | far_field_amplitude | far_field_axis;

/**
 * Where each quantity of the pass lives in its slots, solprop's `FarFieldLayout`.
 *
 * The beam spectrum, the axial flux, the four absorbed channels, the regions, then per
 * aperture its collected spectrum and the real and imaginary parts of its on-axis
 * amplitude per polarisation, then the on-axis amplitude (`2 n_pol` slots); present
 * slots only, an absent quantity -1.
 */
struct FarFieldLayout {
  long beam{-1}, poynting{-1}, absorbed{-1}, regions{-1}, apertures{-1};
  long stride{}, collected{-1}, amplitude{-1}, axis{-1};
  std::size_t n_slots{};
};

/** Lay out the slots of a request, as solprop's `far_field_layout` does. */
[[nodiscard]] FarFieldLayout far_field_layout(std::uint64_t flags,
                                              std::size_t n_apertures,
                                              std::size_t n_regions, std::size_t n_pol);

/** The number of bands of solprop's `far_field_bands`: the units (rows on a plane,
    line blocks on a line) or as many as the budget admits, whichever is fewer, and at
    least one; never the thread count. */
[[nodiscard]] std::size_t far_field_band_count(std::size_t n_x, std::size_t n_y,
                                               std::size_t n_slots,
                                               std::size_t n_omega);

/** The first column of `band` of `n_bands`, and `n_x n_y` for `band == n_bands`. */
KOKKOS_INLINE_FUNCTION std::size_t far_field_band_edge(const std::size_t band,
                                                       const std::size_t n_bands,
                                                       const std::size_t n_x,
                                                       const std::size_t n_y) {
  const std::size_t columns = n_x * n_y;
  const std::size_t unit = n_y > 1 ? n_x : far_field_line_block;
  const std::size_t units = (columns + unit - 1) / unit;
  const std::size_t edge = ((band * units) / n_bands) * unit;
  return edge < columns ? edge : columns;
}

/**
 * How far a column's mask-plane image lies from an aperture's centre (SEP 23 (4)):
 * `sqrt((s k_x + X)^2 + (s k_y + Y)^2)` with the stored axes and the physical centre.
 *
 * Protected: every product and sum rounded on its own and a correctly rounded square
 * root, never `hypot`, so a hard edge decides as solprop's reference decides on every
 * input (an exact hard-mask contract). A device compiler keeps its own contraction
 * default, so a device build spells each operation with a round-to-nearest intrinsic.
 */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION double far_field_radius(const double scale, const double k_x,
                                               const double k_y, const double centre_x,
                                               const double centre_y) {
  SOLPROP_NATIVE_STRICT_BLOCK
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const double across = __dadd_rn(__dmul_rn(scale, k_x), centre_x);
  const double along = __dadd_rn(__dmul_rn(scale, k_y), centre_y);
  return __dsqrt_rn(__dadd_rn(__dmul_rn(across, across), __dmul_rn(along, along)));
#else
  const double across = scale * k_x + centre_x;
  const double along = scale * k_y + centre_y;
  const double sum = across * across + along * along;
  return Kokkos::sqrt(sum);
#endif
}

/**
 * One aperture's amplitude transmission at one column and bin, solprop's
 * `aperture_weight`: exactly zero where the scale is zero (at and below zero
 * frequency) or the image lies at or beyond the distance from the origin; otherwise
 * `hard` (code 0), `tanh` (1), `rcos` (2, exactly one and zero outside its edge, the
 * sine skipped there) and `supergauss` (3), with `t = (radius - half_width) *
 * inverse_width`.
 */
KOKKOS_INLINE_FUNCTION double
far_field_weight(const double radius, const double origin, const double scale,
                 const double half_width, const double inverse_width, const double edge,
                 const double distance, const int code) {
  if (scale <= 0.0 || origin >= distance)
    return 0.0;
  if (code == 0)
    return radius <= half_width ? 1.0 : 0.0;
  if (code == 3)
    return Kokkos::exp(-Kokkos::pow(radius / half_width, edge));
  const double argument = (radius - half_width) * inverse_width;
  if (code == 1) {
    // Beyond |t| = 22, tanh lies within 1.6e-19 of +-1, below half an ulp, so its
    // correctly rounded value is +-1 and the weight exactly 0 or 1; the transcendental
    // is skipped. A routine that rounds otherwise there differs by its own error,
    // which solprop's parity bound already carries.
    if (argument >= far_field_tanh_saturation)
      return 0.0;
    if (argument <= -far_field_tanh_saturation)
      return 1.0;
    return 0.5 * (1.0 - Kokkos::tanh(argument));
  }
  if (argument <= -1.0)
    return 1.0;
  if (argument >= 1.0)
    return 0.0;
  return 0.5 * (1.0 - Kokkos::sin((0.5 * std::numbers::pi) * argument));
}

/** Whether a column lies in a half-open box of physical transverse wavevector: the
    stored axes negated, `[low, high)` on both axes (SEP 23 section 4.7). */
KOKKOS_INLINE_FUNCTION bool far_field_member(const double k_x, const double k_y,
                                             const double x_low, const double x_high,
                                             const double y_low, const double y_high) {
  const double kappa_x = -k_x;
  const double kappa_y = -k_y;
  return x_low <= kappa_x && kappa_x < x_high && y_low <= kappa_y && kappa_y < y_high;
}

/**
 * The far-field pass of one state (SEP 23 section 6 on the native backend).
 *
 * The column axes, the band partials of the problem's own request and the output are
 * allocated at construction; a call that needs more partials or table room than were
 * sized for (a request with other tables) grows them, outside any trial step. Stage
 * one is a team per band over its columns in ascending order, the team's threads over
 * frequency; stage two adds the bands in ascending order and applies the constant
 * factors (`intensity_factor`, the spectral-density prefactor, the energy weights and
 * `|A|^2`), so only the packed statistics cross to the host.
 */
class FarFieldPass {
public:
  using ComplexView2 = Kokkos::View<Complex**, Layout, MemorySpace>;
  using ByteView = Kokkos::View<unsigned char*, Layout, MemorySpace>;

  /** What the pass reads besides the state: the geometry's weights and on-axis
      vector, and the separable operator, the engine's own views. */
  struct Inputs {
    RealView weights;
    ComplexView on_axis;
    ComplexView square, square_low;
    RealView transverse, transverse_low, spectral_rate, transverse_rate, omega;
    ByteView active;
    double evanescent{};
  };

  FarFieldPass(const Problem& problem, const ExecutionSpace& execution);
  ~FarFieldPass() = default;
  FarFieldPass(const FarFieldPass&) = delete;
  FarFieldPass& operator=(const FarFieldPass&) = delete;
  FarFieldPass(FarFieldPass&&) = delete;
  FarFieldPass& operator=(FarFieldPass&&) = delete;

  /**
   * Form the requested statistics of `field`, `(n_x n_y n_pol, n_omega)`.
   *
   * `apertures` is row-major `(n_apertures, 6 + n_omega)`: the six constants of
   * `far_field_aperture_constants` and the scale row `z c / omega` (m^2, zero at and
   * below zero frequency); `regions` row-major `(n_regions, 4)`. Returns, in order and
   * present ones only: the beam and axial-flux spectra (`n_omega` each), the four
   * absorbed powers, the region spectra, the collected spectra and the apertures'
   * on-axis spectra (`n_omega` per row), and the on-axis spectrum (`n_omega`).
   */
  [[nodiscard]] std::vector<double>
  run(std::uint64_t flags, const std::vector<double>& apertures, int profile_code,
      int exponent, const std::vector<double>& regions, double intensity_factor,
      const ComplexView2& field, const Inputs& inputs);

  /** The bytes of the band partials, the ledger's `far_field_partials` row. */
  [[nodiscard]] std::size_t partial_bytes() const noexcept;
  /** The bytes of the column axes, tables and packed output. */
  [[nodiscard]] std::size_t table_bytes() const noexcept;
  void close() noexcept;

  /** Which per-element radii every aperture shares: the distance from the origin
      (one distance and scale row) and the distance from the centre (one centre too). */
  struct Sharing {
    bool centre{}, distance{};
  };

  // NVCC extended lambdas require their enclosing member functions to be public.
  void accumulate(const FarFieldLayout& layout, std::size_t n_bands,
                  std::size_t n_apertures, std::size_t n_regions, int profile_code,
                  int exponent, Sharing sharing, const ComplexView2& field,
                  const Inputs& inputs);
  void finish(const FarFieldLayout& layout, std::size_t n_bands,
              std::size_t n_apertures, std::size_t n_regions, double intensity_factor,
              std::size_t packed_size);

private:
  ExecutionSpace execution_;
  std::size_t n_x_{}, n_y_{}, n_omega_{}, n_pol_{}, n_t_{};
  double dt_{};
  bool envelope_{};
  RealView k_x_, k_y_, partials_, tables_, packed_;
  RealView::host_mirror_type host_tables_, host_packed_;
};

} // namespace solprop_native
