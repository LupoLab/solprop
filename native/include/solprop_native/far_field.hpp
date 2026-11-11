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
 * pass over the stored transverse spectrum that forms the beam spectrum, the five
 * absorbed-power channels, the region spectra, each aperture's collected spectrum and
 * on-axis amplitude, and the on-axis amplitude itself (the `W = 1` slot, which
 * `on_axis_spectrum` reads). Every quantity is measured by the medium's axial flux
 * (solprop SEP 34): a squared modulus is weighted by the real admittance
 * `c Re k_z / omega` of the operator's root, and each amplitude carries its magnetic
 * closure, the same sum with `conj(c k_z / omega)` applied first, formed per element
 * (no column-by-frequency table is held). It takes the
 * slot layout of `solprop.backends.spatial.far_field_layout`, the band edges of
 * `far_field_bands` (the same units and budget), the radius of `mask_plane_radius`
 * compiled without contraction, the outlines of `ApertureOutline`, the zeros of
 * `aperture_weight` and the root of `longitudinal_root`. Each band sums its columns in
 * ascending order and the bands are then summed in ascending order, so every value has
 * one association whatever the thread count (solprop SEP 25 compares it with the
 * reference within a rounding bound, never by bits).
 */

namespace solprop_native {

/** solprop's `FAR_FIELD_BAND_BUDGET`: bytes the band partials may hold. */
inline constexpr std::size_t far_field_band_budget = std::size_t{128} << 20U;

/** solprop's `LINE_BLOCK`: columns per unit of a line's pass; a plane's is a row. */
inline constexpr std::size_t far_field_line_block = 64;

/** The constants per aperture before its scale row, solprop's `ApertureTables`. The
    distance stays last: the rows that share it share the scale row after it. */
enum FarFieldConstant : std::size_t {
  far_field_centre_x = 0,      // m
  far_field_centre_y = 1,      // m
  far_field_half_width = 2,    // m: a circle's or an ellipse's radius, a rectangle's x
  far_field_half_height = 3,   // m: a rectangle's half-width along y
  far_field_stretch_x = 4,     // an ellipse's factor on the offset in x
  far_field_stretch_y = 5,     // an ellipse's factor on the offset in y
  far_field_inverse_width = 6, // 1/m
  far_field_edge = 7,          // m, or a supergauss order
  far_field_profile = 8,       // solprop's `APERTURE_PROFILES` index
  far_field_shape = 9,         // solprop's `APERTURE_SHAPES` index
  far_field_distance = 10,     // m
  far_field_aperture_constants = 11,
};

/** solprop's `APERTURE_PROFILES`, the codes of a row's profile. */
enum FarFieldProfile : int {
  far_field_hard = 0,
  far_field_tanh = 1,
  far_field_rcos = 2,
  far_field_supergauss = 3,
};

/** solprop's `APERTURE_SHAPES`, the codes of a row's shape. */
enum FarFieldShape : int {
  far_field_circle = 0,
  far_field_ellipse = 1,
  far_field_rectangle = 2,
  far_field_half_plane_x_ge = 3,
  far_field_half_plane_x_lt = 4,
  far_field_half_plane_y_ge = 5,
  far_field_half_plane_y_lt = 6,
};

/** The absorbed-power channels: the spectral collar, transverse, evanescent, material
    and the user's spectral rate, solprop's `ABSORBED_CHANNELS`. */
inline constexpr std::size_t far_field_absorbed_channels = 5;

/** Where `tanh` saturates: correctly rounded, `tanh(t)` is exactly 1 for `t >= 22`. */
inline constexpr double far_field_tanh_saturation = 22.0;

/** The vacuum speed of light, m/s, exact in the SI; the admittance `c k_z / omega`. */
inline constexpr double far_field_speed_of_light = 299792458.0;

/** The flags of `aperture_statistics`, from 32: 1 to 16 are `spatial_statistics`'. */
enum FarFieldFlag : std::uint64_t {
  far_field_beam = 1ULL << 5U,
  far_field_absorbed = 1ULL << 7U,
  far_field_regions = 1ULL << 8U,
  far_field_collected = 1ULL << 9U,
  far_field_amplitude = 1ULL << 10U,
  far_field_axis = 1ULL << 11U,
};

inline constexpr std::uint64_t far_field_flags =
    far_field_beam | far_field_absorbed | far_field_regions | far_field_collected |
    far_field_amplitude | far_field_axis;

/**
 * Where each quantity of the pass lives in its slots, solprop's `FarFieldLayout`.
 *
 * The beam spectrum, the five absorbed channels, the regions, then per aperture its
 * collected spectrum and its on-axis amplitude, then the on-axis amplitude itself; an
 * amplitude is the real and imaginary parts per polarisation of the electric sum, then
 * those of its magnetic closure (`4 n_pol` slots). Present slots only, an absent
 * quantity -1.
 */
struct FarFieldLayout {
  long beam{-1}, absorbed{-1}, regions{-1}, apertures{-1};
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
 * The strict block of the far-field helpers on a HIP build, where
 * `SOLPROP_NATIVE_STRICT_BLOCK` is empty. Clang's HIP `__dmul_rn` and `__dadd_rn` are a
 * plain `x * y` and `x + y` in its headers, contractable once inlined, so only a
 * protected block of plain arithmetic rounds each operation on its own. CUDA's
 * intrinsics are real.
 */
#ifdef __HIPCC__
#define SOLPROP_NATIVE_FAR_FIELD_HIP_STRICT _Pragma("clang fp contract(off)")
#else
#define SOLPROP_NATIVE_FAR_FIELD_HIP_STRICT
#endif

/**
 * How far a column's mask-plane image lies from an aperture's centre (SEP 23 (4)):
 * `sqrt((s k_x + X)^2 + (s k_y + Y)^2)` with the stored axes and the physical centre.
 *
 * Protected: every product and sum rounded on its own and a correctly rounded square
 * root, never `hypot`, so a hard edge decides as solprop's reference decides on every
 * input (an exact hard-mask contract). A device compiler keeps its own contraction
 * default, so a CUDA build spells each operation with a round-to-nearest intrinsic and
 * a HIP build, whose intrinsics are no protection, takes the block above. A HIP
 * device's square root is `__builtin_sqrt`, so `llvm.sqrt.f64`, which LLVM defines as a
 * conforming IEEE 754 square root, correctly rounded, unless the call carries `afn`;
 * AMD's own math documentation promises less (docs/dev/accuracy.md section 7).
 */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION double far_field_radius(const double scale, const double k_x,
                                               const double k_y, const double centre_x,
                                               const double centre_y) {
  SOLPROP_NATIVE_STRICT_BLOCK
  SOLPROP_NATIVE_FAR_FIELD_HIP_STRICT
#ifdef __CUDA_ARCH__
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
 * A column's offset from a mask-plane point along one axis, solprop's
 * `mask_plane_offsets`: `s k + X = X - x`, with the stored axis and the physical point.
 *
 * Protected as `far_field_radius` is: a product and a sum, each rounded on its own, so
 * that a half-plane's or a rectangle's hard edge decides as the reference decides.
 */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION double far_field_offset(const double scale, const double k,
                                               const double centre) {
  SOLPROP_NATIVE_STRICT_BLOCK
  SOLPROP_NATIVE_FAR_FIELD_HIP_STRICT
#ifdef __CUDA_ARCH__
  return __dadd_rn(__dmul_rn(scale, k), centre);
#else
  return scale * k + centre;
#endif
}

/**
 * The radius of an ellipse's image in the frame where it is a circle, from the offsets
 * of `far_field_offset` and the two stretches: `sqrt((f_x a)^2 + (f_y b)^2)`.
 *
 * Protected as `far_field_radius` is, so that a hard edge decides as the reference's.
 */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION double far_field_scaled_radius(const double across,
                                                      const double along,
                                                      const double stretch_x,
                                                      const double stretch_y) {
  SOLPROP_NATIVE_STRICT_BLOCK
  SOLPROP_NATIVE_FAR_FIELD_HIP_STRICT
#ifdef __CUDA_ARCH__
  const double scaled_x = __dmul_rn(stretch_x, across);
  const double scaled_y = __dmul_rn(stretch_y, along);
  return __dsqrt_rn(
      __dadd_rn(__dmul_rn(scaled_x, scaled_x), __dmul_rn(scaled_y, scaled_y)));
#else
  const double scaled_x = stretch_x * across;
  const double scaled_y = stretch_y * along;
  const double sum = scaled_x * scaled_x + scaled_y * scaled_y;
  return Kokkos::sqrt(sum);
#endif
}

/**
 * An edge's weight at a signed distance from it, negative inside, solprop's
 * `edge_profile`: `hard` (code 0), `tanh` (1) and `rcos` (2, exactly one and zero
 * outside its edge, the sine skipped there), with `t = distance * inverse_width`. The
 * sign of a difference of two finite doubles is exact, so `hard` on `radius -
 * half_width` decides as `radius <= half_width` does.
 */
KOKKOS_INLINE_FUNCTION double far_field_edge_weight(const double distance,
                                                    const double inverse_width,
                                                    const int code) {
  if (code == far_field_hard)
    return distance <= 0.0 ? 1.0 : 0.0;
  const double argument = distance * inverse_width;
  if (code == far_field_tanh) {
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

/** A profile as a function of the distance from a centre, solprop's `aperture_profile`:
    `supergauss` (code 3) of `radius / half_width`, any other an edge at the half-width.
 */
KOKKOS_INLINE_FUNCTION double
far_field_radial_weight(const double radius, const double half_width,
                        const double inverse_width, const double edge, const int code) {
  if (code == far_field_supergauss)
    return Kokkos::exp(-Kokkos::pow(radius / half_width, edge));
  return far_field_edge_weight(radius - half_width, inverse_width, code);
}

/**
 * One circular aperture's amplitude transmission at one column and bin, solprop's
 * `aperture_weight`: exactly zero where the scale is zero (at and below zero
 * frequency) or the image lies at or beyond the distance from the origin; otherwise the
 * profile of its distance from the centre.
 */
KOKKOS_INLINE_FUNCTION double
far_field_weight(const double radius, const double origin, const double scale,
                 const double half_width, const double inverse_width, const double edge,
                 const double distance, const int code) {
  if (scale <= 0.0 || origin >= distance)
    return 0.0;
  return far_field_radial_weight(radius, half_width, inverse_width, edge, code);
}

/** The constants of one row that are not a circle's, as `far_field_shaped_weight` reads
    them: solprop's `ApertureOutline` less its shape and profile. */
struct FarFieldOutline {
  double half_width{}, half_height{}, stretch_x{}, stretch_y{}, inverse_width{}, edge{};
};

/**
 * The weight of a shape other than a circle from its image's offsets, solprop's
 * `ApertureOutline.weight`. An ellipse is the profile of its scaled radius; a rectangle
 * the product of two straight edges, each a circle's edge of `|offset|`; and a
 * half-plane the edge at its offset, keeping `offset <= 0` (`x >= X`), or one minus
 * that (`x < X`), half-open, so that a half-plane and its complement tile the columns.
 * The zeros of the scale and the origin are the caller's.
 */
KOKKOS_INLINE_FUNCTION double far_field_shaped_weight(const double across,
                                                      const double along,
                                                      const FarFieldOutline& outline,
                                                      const int shape, const int code) {
  if (shape == far_field_ellipse)
    return far_field_radial_weight(
        far_field_scaled_radius(across, along, outline.stretch_x, outline.stretch_y),
        outline.half_width, outline.inverse_width, outline.edge, code);
  if (shape == far_field_rectangle)
    return far_field_radial_weight(Kokkos::abs(across), outline.half_width,
                                   outline.inverse_width, outline.edge, code) *
           far_field_radial_weight(Kokkos::abs(along), outline.half_height,
                                   outline.inverse_width, outline.edge, code);
  const bool bounded_in_x =
      shape == far_field_half_plane_x_ge || shape == far_field_half_plane_x_lt;
  const double kept =
      far_field_edge_weight(bounded_in_x ? across : along, outline.inverse_width, code);
  const bool keeps_above =
      shape == far_field_half_plane_x_ge || shape == far_field_half_plane_y_ge;
  return keeps_above ? kept : 1.0 - kept;
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
 * each amplitude's pairing `Re(A conj(G))` with its closure), so only the packed
 * statistics cross to the host.
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
    RealView transverse, transverse_low, spectral_rate, user_rate, transverse_rate,
        omega;
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
   * `apertures` is row-major `(n_apertures, far_field_aperture_constants + n_omega)`:
   * the constants of `FarFieldConstant` and the scale row `z c / omega` (m^2, zero at
   * and below zero frequency); `regions` row-major `(n_regions, 4)`. Returns, in order
   * and present ones only: the beam spectrum (`n_omega`), the five absorbed powers, the
   * region spectra, the collected spectra and the apertures' on-axis spectra (`n_omega`
   * per row), and the on-axis spectrum (`n_omega`).
   */
  [[nodiscard]] std::vector<double>
  run(std::uint64_t flags, const std::vector<double>& apertures, int exponent,
      const std::vector<double>& regions, double intensity_factor,
      const ComplexView2& field, const Inputs& inputs);

  /**
   * The analytic spectrum of the on-axis amplitude, or of its closure, into the leading
   * `n_pol` rows of `spectrum`, from the partials a `run` with `far_field_axis` alone
   * left: the bins' sums over the columns, which `spatial_statistics` transforms for
   * `on_axis_intensity`.
   */
  void axis_spectrum(bool closed, const ComplexView2& spectrum);

  /** The bytes of the band partials, the ledger's `far_field_partials` row. */
  [[nodiscard]] std::size_t partial_bytes() const noexcept;
  /** The bytes of the column axes, tables and packed output. */
  [[nodiscard]] std::size_t table_bytes() const noexcept;
  void close() noexcept;

  /** What the rows have in common: the distance from the origin is evaluated once per
      element when they share one distance and scale row, and the distance from the
      centre too when they are all circles with one centre. `circles` says no row needs
      the shaped branch. */
  struct Sharing {
    bool centre{}, distance{}, circles{};
  };

  // NVCC extended lambdas require their enclosing member functions to be public.
  void accumulate(const FarFieldLayout& layout, std::size_t n_bands,
                  std::size_t n_apertures, std::size_t n_regions, int exponent,
                  Sharing sharing, const ComplexView2& field, const Inputs& inputs);
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
