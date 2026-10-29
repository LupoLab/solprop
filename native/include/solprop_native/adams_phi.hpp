// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "solprop_native/core.hpp"
#include "solprop_native/fp.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>

/*
 * The exponential Adams march's phi-functions (solprop backends/exponential.py,
 * phi_functions; the compiled form of solprop's numba_cpu/multistep.py, _phi):
 * `adams_phi` for one element, and on the host the same values for a block of
 * elements held as lanes (`AdamsLanes`, `adams_lanes_phi`), which the trial's host
 * passes use. tests/native/test_adams_phi.cpp holds the two to the same bits.
 */
namespace solprop_native {

inline constexpr double adams_phi_join = 3.0;
inline constexpr int adams_series_terms = 24;

/**
 * The halvings of the phi-functions' series branch: the count that brings |x| to at
 * most 1/2, ceil(log2(2 |x|)), and `scale = 2^-count`, each halving exact.
 */
KOKKOS_INLINE_FUNCTION int adams_phi_halvings(const double magnitude, double& scale) {
  int squarings = 0;
  scale = 1.0;
  double reduced = magnitude;
  while (reduced > 0.5) {
    ++squarings;
    scale *= 0.5;
    reduced *= 0.5;
  }
  return squarings;
}

/**
 * phi_0(x) .. phi_order(x) for one element into `out`, as solprop's compiled _phi,
 * given `magnitude = |x|` and `exp_x`. A host block forms the same values on its lanes
 * (`adams_lanes_phi`).
 *
 * The small-argument phi-functions: strict arithmetic (solprop SEP 25 section 4.2).
 */
SOLPROP_NATIVE_STRICT_FUNCTION
KOKKOS_INLINE_FUNCTION void adams_phi(const Complex x, const double magnitude,
                                      const Complex exp_x, const std::size_t order,
                                      Complex* out) {
  SOLPROP_NATIVE_STRICT_BLOCK
  // 1/k! for k <= 2 adams_max_samples, exactly representable rounding of the reference.
  constexpr double reciprocal[17] = {1.0,
                                     1.0,
                                     0.5,
                                     1.0 / 6.0,
                                     1.0 / 24.0,
                                     1.0 / 120.0,
                                     1.0 / 720.0,
                                     1.0 / 5040.0,
                                     1.0 / 40320.0,
                                     1.0 / 362880.0,
                                     1.0 / 3628800.0,
                                     1.0 / 39916800.0,
                                     1.0 / 479001600.0,
                                     1.0 / 6227020800.0,
                                     1.0 / 87178291200.0,
                                     1.0 / 1307674368000.0,
                                     1.0 / 20922789888000.0};
  // 1/n for the series' divisors m + order, the doubles the division rounds to.
  static_assert(adams_series_terms + 2 * adams_max_samples <= 40);
  constexpr double inverse_divisor[40] = {
      0.0,        1.0 / 1.0,  1.0 / 2.0,  1.0 / 3.0,  1.0 / 4.0,  1.0 / 5.0,
      1.0 / 6.0,  1.0 / 7.0,  1.0 / 8.0,  1.0 / 9.0,  1.0 / 10.0, 1.0 / 11.0,
      1.0 / 12.0, 1.0 / 13.0, 1.0 / 14.0, 1.0 / 15.0, 1.0 / 16.0, 1.0 / 17.0,
      1.0 / 18.0, 1.0 / 19.0, 1.0 / 20.0, 1.0 / 21.0, 1.0 / 22.0, 1.0 / 23.0,
      1.0 / 24.0, 1.0 / 25.0, 1.0 / 26.0, 1.0 / 27.0, 1.0 / 28.0, 1.0 / 29.0,
      1.0 / 30.0, 1.0 / 31.0, 1.0 / 32.0, 1.0 / 33.0, 1.0 / 34.0, 1.0 / 35.0,
      1.0 / 36.0, 1.0 / 37.0, 1.0 / 38.0, 1.0 / 39.0};
  if (magnitude == 0.0) {
    for (std::size_t k = 0; k <= order; ++k)
      out[k] = reciprocal[k];
    return;
  }
  if (magnitude > adams_phi_join) {
    const Complex inverse = strict_divide(Complex(1.0), x);
    out[0] = exp_x;
    for (std::size_t k = 1; k <= order; ++k)
      out[k] = strict_mul(strict_sub(out[k - 1], Complex(reciprocal[k - 1])), inverse);
    return;
  }
  double scale = 1.0;
  const int squarings = adams_phi_halvings(magnitude, scale);
  const Complex y = strict_scale(scale, x);
  // Each later term is below |y| / (m + order) of this one, |y| <= 1/2: once this one
  // is under a quarter ulp of the sum, the rest cannot change it. The test compares
  // squared magnitudes, which differ from hypot's only in the test's own rounding, so a
  // term below a quarter ulp may be added once more or once less.
  constexpr double quarter_ulp = 0.25 * Kokkos::Experimental::epsilon_v<double>;
  constexpr double stop = quarter_ulp * quarter_ulp;
  Complex term = reciprocal[order];
  Complex total = term;
  for (int m = 1; m < adams_series_terms; ++m) {
    term = strict_scale(inverse_divisor[static_cast<std::size_t>(m) + order],
                        strict_mul(term, y));
    total = strict_add(total, term);
    if (strict_norm(term) <= stop * strict_norm(total))
      break;
  }
  out[order] = total;
  for (std::size_t k = order - 1; k > 0; --k)
    out[k] = strict_add(reciprocal[k], strict_mul(y, out[k + 1]));
  out[0] = Kokkos::exp(y);
  Complex old[2 * adams_max_samples + 1];
  for (int pass = 0; pass < squarings; ++pass) {
    for (std::size_t k = 0; k <= order; ++k)
      old[k] = out[k];
    out[0] = strict_mul(old[0], old[0]);
    double half = 1.0; // 2^-k, exact
    for (std::size_t k = 1; k <= order; ++k) {
      half *= 0.5;
      Complex sum = strict_mul(old[0], old[k]);
      for (std::size_t i = 1; i <= k; ++i)
        sum = strict_accumulate(sum, reciprocal[k - i], old[i]);
      out[k] = strict_scale(half, sum);
    }
  }
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
/** The elements of a host block of the elementwise trial's passes. */
inline constexpr std::size_t adams_lane_width = 32;

// The elementwise trial's host blocks as lanes (solprop ADR-059). A block's elements
// are held as a structure of arrays, one array a part, and each step of the trial is
// a loop over the block's elements that the compiler vectorises: the phi-functions'
// series, Horner pass and squarings, the predictor and corrector sums, the
// compensation and the defects. Every element takes the operations, in the order and
// with the rounding, that the element-by-element form gives it (`adams_phi`,
// `adams_predicted_compensation`, `adams_corrected_compensation`,
// `adams_defect_block`); where elements branch apart (the phi-functions' three
// branches, their series' stopping point, their squarings) each lane computes every
// branch it may need and keeps its own, so a lane's values do not depend on its
// neighbours'. The strict parts are strict functions over the whole block (solprop
// SEP 25 section 4.2), so that GCC compiles each as one call a block.

// 1/k! and 1/n as `adams_phi` holds them.
inline constexpr double adams_lane_reciprocal[adams_max_samples + 2] = {
    1.0,         1.0,         0.5,          1.0 / 6.0,     1.0 / 24.0,
    1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0, 1.0 / 40320.0, 1.0 / 362880.0};
inline constexpr double adams_lane_inverse_divisor[adams_series_terms +
                                                   adams_max_samples + 1] = {
    0.0,        1.0 / 1.0,  1.0 / 2.0,  1.0 / 3.0,  1.0 / 4.0,  1.0 / 5.0,  1.0 / 6.0,
    1.0 / 7.0,  1.0 / 8.0,  1.0 / 9.0,  1.0 / 10.0, 1.0 / 11.0, 1.0 / 12.0, 1.0 / 13.0,
    1.0 / 14.0, 1.0 / 15.0, 1.0 / 16.0, 1.0 / 17.0, 1.0 / 18.0, 1.0 / 19.0, 1.0 / 20.0,
    1.0 / 21.0, 1.0 / 22.0, 1.0 / 23.0, 1.0 / 24.0, 1.0 / 25.0, 1.0 / 26.0, 1.0 / 27.0,
    1.0 / 28.0, 1.0 / 29.0, 1.0 / 30.0, 1.0 / 31.0, 1.0 / 32.0};

// The phi-functions' branch of a lane, as `adams_phi` chooses it.
inline constexpr unsigned char adams_lane_zero = 0;
inline constexpr unsigned char adams_lane_large = 1;
inline constexpr unsigned char adams_lane_series = 2;

// A block's lanes. Each pass declares one and uses the parts it needs; the parts are
// members of one object so that the compiler knows they do not overlap.
struct AdamsLanes {
  static constexpr std::size_t B = adams_lane_width;
  static constexpr std::size_t rows = adams_max_samples + 1;
  std::size_t count{};
  // The step's exponent x and flow e^x.
  double xr[B], xi[B], fr[B], fi[B];
  // The moments' argument (x, or x - sigma), its series argument 2^-s and that one's
  // exponential, and the large branch's e^argument.
  double ar[B], ai[B], yr[B], yi[B], er[B], ei[B], gr[B], gi[B];
  int squarings[B];
  unsigned char branch[B];
  bool any_large{};
  int most_squarings{};
  // phi_0 .. phi_order, and the squarings' previous values.
  double pr[rows][B], pi[rows][B], qr[rows][B], qi[rows][B];
  // History rows (rotated by a carrier frame) and the moments.
  double hr[adams_max_samples][B], hi[adams_max_samples][B];
  double mr[adams_max_samples][B], mi[adams_max_samples][B];
  // A varying operator's carried samples and spans' exponentials.
  double cr[adams_max_samples][B], ci[adams_max_samples][B];
  double sr[adams_max_samples][B], si[adams_max_samples][B];
  // The value formed (prediction or correction), the accepted state (predict) or the
  // lab-frame field (correct), and the new sample (correct).
  double vr[B], vi[B], ur[B], ui[B], wr[B], wi[B];
  // The endpoint's compensation, and the norms' divisors and squared defects.
  double nr[B], ni[B], scale[B];
  double defect[3][B];
};

// The moments' argument and its branch, halvings and series argument, as
// `adams_moment_argument`, `adams_phi` and `adams_phi_argument` form them, with the
// magnitude `Kokkos::abs` takes.
SOLPROP_NATIVE_STRICT_FUNCTION
inline void adams_lanes_classify(AdamsLanes& lanes, const ExponentialWeights& weights) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const std::size_t count = lanes.count;
  const double shift = weights.shift;
  if (weights.shifted)
    for (std::size_t b = 0; b < count; ++b) {
      lanes.ar[b] = lanes.xr[b] - 0.0;
      lanes.ai[b] = lanes.xi[b] - shift;
    }
  else
    for (std::size_t b = 0; b < count; ++b) {
      lanes.ar[b] = lanes.xr[b];
      lanes.ai[b] = lanes.xi[b];
    }
  // `adams_phi` compares |x| with 3 and its halvings with 1/2, 1 and 2; each test is
  // taken on the squared magnitude where that is at least a part in 2^40 from the
  // threshold's square, which decides it as `hypot` would, and from `hypot` otherwise
  // (and for a non-finite square).
  constexpr double margin = 0x1p-40;
  bool any_large = false;
  bool any_close = false;
  int most = 0;
  double scale[AdamsLanes::B];
  unsigned char close[AdamsLanes::B];
  for (std::size_t b = 0; b < count; ++b) {
    const double rr = lanes.ar[b] * lanes.ar[b];
    const double ii = lanes.ai[b] * lanes.ai[b];
    const double square = rr + ii;
    const bool zero = lanes.ar[b] == 0.0 && lanes.ai[b] == 0.0;
    const bool large = square > 9.0;
    const int squarings =
        (square > 0.25 ? 1 : 0) + (square > 1.0 ? 1 : 0) + (square > 4.0 ? 1 : 0);
    const double d0 = Kokkos::fabs(square - 0.25);
    const double d1 = Kokkos::fabs(square - 1.0);
    const double d2 = Kokkos::fabs(square - 4.0);
    const double d3 = Kokkos::fabs(square - 9.0);
    const bool uncertain = !(square <= Kokkos::Experimental::finite_max_v<double>) ||
                           d0 <= margin * 0.25 || d1 <= margin || d2 <= margin * 4.0 ||
                           d3 <= margin * 9.0;
    close[b] = uncertain ? 1 : 0;
    any_close = any_close || uncertain;
    const unsigned char branch =
        zero ? adams_lane_zero : (large ? adams_lane_large : adams_lane_series);
    const bool series = branch == adams_lane_series;
    lanes.branch[b] = branch;
    lanes.squarings[b] = series ? squarings : 0;
    scale[b] =
        squarings == 0 ? 1.0 : (squarings == 1 ? 0.5 : (squarings == 2 ? 0.25 : 0.125));
  }
  if (any_close)
    for (std::size_t b = 0; b < count; ++b) {
      if (close[b] == 0)
        continue;
      const double magnitude = Kokkos::hypot(lanes.ar[b], lanes.ai[b]);
      unsigned char branch = adams_lane_series;
      int squarings = 0;
      double halved = 1.0;
      if (magnitude == 0.0)
        branch = adams_lane_zero;
      else if (magnitude > adams_phi_join)
        branch = adams_lane_large;
      else
        squarings = adams_phi_halvings(magnitude, halved);
      lanes.branch[b] = branch;
      lanes.squarings[b] = squarings;
      scale[b] = halved;
    }
  for (std::size_t b = 0; b < count; ++b) {
    const bool series = lanes.branch[b] == adams_lane_series;
    any_large = any_large || lanes.branch[b] == adams_lane_large;
    most = lanes.squarings[b] > most ? lanes.squarings[b] : most;
    lanes.yr[b] = series ? scale[b] * lanes.ar[b] : 0.0;
    lanes.yi[b] = series ? scale[b] * lanes.ai[b] : 0.0;
  }
  lanes.any_large = any_large;
  lanes.most_squarings = most;
}

// phi_0 .. phi_order of every lane into `pr`/`pi`, as `adams_phi` forms them; `er`/`ei`
// hold e^y and `gr`/`gi` the large branch's e^argument. Strict (solprop SEP 25
// section 4.2).
SOLPROP_NATIVE_STRICT_FUNCTION
inline void adams_lanes_phi(AdamsLanes& lanes, const std::size_t order) {
  SOLPROP_NATIVE_STRICT_BLOCK
  constexpr std::size_t B = AdamsLanes::B;
  const std::size_t count = lanes.count;
  constexpr double quarter_ulp = 0.25 * Kokkos::Experimental::epsilon_v<double>;
  constexpr double stop = quarter_ulp * quarter_ulp;
  // The series: every lane takes its terms until its own stopping test holds; a lane
  // that has stopped keeps its sum and carries a zero term.
  double term_r[B], term_i[B], sum_r[B], sum_i[B];
  unsigned char live[B];
  const double first = adams_lane_reciprocal[order];
  for (std::size_t b = 0; b < count; ++b) {
    term_r[b] = first;
    term_i[b] = 0.0;
    sum_r[b] = first;
    sum_i[b] = 0.0;
    live[b] = lanes.branch[b] == adams_lane_series ? 1 : 0;
  }
  for (std::size_t m = 1; m < static_cast<std::size_t>(adams_series_terms); ++m) {
    const double inverse = adams_lane_inverse_divisor[m + order];
    unsigned char any = 0;
    for (std::size_t b = 0; b < count; ++b) {
      const double rr = term_r[b] * lanes.yr[b];
      const double ii = term_i[b] * lanes.yi[b];
      const double ri = term_r[b] * lanes.yi[b];
      const double ir = term_i[b] * lanes.yr[b];
      const double product_r = rr - ii;
      const double product_i = ri + ir;
      const double next_r = inverse * product_r;
      const double next_i = inverse * product_i;
      const double total_r = sum_r[b] + next_r;
      const double total_i = sum_i[b] + next_i;
      const double term_rr = next_r * next_r;
      const double term_ii = next_i * next_i;
      const double total_rr = total_r * total_r;
      const double total_ii = total_i * total_i;
      const double term_norm = term_rr + term_ii;
      const double total_norm = total_rr + total_ii;
      const bool on = live[b] != 0;
      sum_r[b] = on ? total_r : sum_r[b];
      sum_i[b] = on ? total_i : sum_i[b];
      const bool more = on && !(term_norm <= stop * total_norm);
      live[b] = more ? 1 : 0;
      term_r[b] = more ? next_r : 0.0;
      term_i[b] = more ? next_i : 0.0;
      any |= live[b];
    }
    if (any == 0)
      break;
  }
  for (std::size_t b = 0; b < count; ++b) {
    lanes.pr[order][b] = sum_r[b];
    lanes.pi[order][b] = sum_i[b];
  }
  // Horner: phi_k = 1/k! + y phi_{k+1}.
  for (std::size_t k = order - 1; k > 0; --k) {
    const double reciprocal = adams_lane_reciprocal[k];
    for (std::size_t b = 0; b < count; ++b) {
      const double rr = lanes.yr[b] * lanes.pr[k + 1][b];
      const double ii = lanes.yi[b] * lanes.pi[k + 1][b];
      const double ri = lanes.yr[b] * lanes.pi[k + 1][b];
      const double ir = lanes.yi[b] * lanes.pr[k + 1][b];
      lanes.pr[k][b] = reciprocal + (rr - ii);
      lanes.pi[k][b] = ri + ir;
    }
  }
  for (std::size_t b = 0; b < count; ++b) {
    lanes.pr[0][b] = lanes.er[b];
    lanes.pi[0][b] = lanes.ei[b];
  }
  // The squarings, each lane its own number.
  for (int pass = 0; pass < lanes.most_squarings; ++pass) {
    for (std::size_t k = 0; k <= order; ++k)
      for (std::size_t b = 0; b < count; ++b) {
        lanes.qr[k][b] = lanes.pr[k][b];
        lanes.qi[k][b] = lanes.pi[k][b];
      }
    for (std::size_t b = 0; b < count; ++b) {
      const double rr = lanes.qr[0][b] * lanes.qr[0][b];
      const double ii = lanes.qi[0][b] * lanes.qi[0][b];
      const double ri = lanes.qr[0][b] * lanes.qi[0][b];
      const double ir = lanes.qi[0][b] * lanes.qr[0][b];
      const bool on = pass < lanes.squarings[b];
      lanes.pr[0][b] = on ? rr - ii : lanes.qr[0][b];
      lanes.pi[0][b] = on ? ri + ir : lanes.qi[0][b];
    }
    double half = 1.0; // 2^-k, exact
    for (std::size_t k = 1; k <= order; ++k) {
      half *= 0.5;
      double total_r[B], total_i[B];
      for (std::size_t b = 0; b < count; ++b) {
        const double rr = lanes.qr[0][b] * lanes.qr[k][b];
        const double ii = lanes.qi[0][b] * lanes.qi[k][b];
        const double ri = lanes.qr[0][b] * lanes.qi[k][b];
        const double ir = lanes.qi[0][b] * lanes.qr[k][b];
        total_r[b] = rr - ii;
        total_i[b] = ri + ir;
      }
      for (std::size_t i = 1; i <= k; ++i) {
        const double weight = adams_lane_reciprocal[k - i];
        for (std::size_t b = 0; b < count; ++b) {
          const double real = weight * lanes.qr[i][b];
          const double imag = weight * lanes.qi[i][b];
          total_r[b] = total_r[b] + real;
          total_i[b] = total_i[b] + imag;
        }
      }
      for (std::size_t b = 0; b < count; ++b) {
        const bool on = pass < lanes.squarings[b];
        lanes.pr[k][b] = on ? half * total_r[b] : lanes.qr[k][b];
        lanes.pi[k][b] = on ? half * total_i[b] : lanes.qi[k][b];
      }
    }
  }
  // Large arguments: phi_k = (phi_{k-1} - 1/(k-1)!) / x, the inverse by
  // `strict_divide`'s scaled algorithm.
  if (lanes.any_large) {
    double inverse_r[B], inverse_i[B], last_r[B], last_i[B];
    for (std::size_t b = 0; b < count; ++b) {
      const double s = Kokkos::fabs(lanes.ar[b]) + Kokkos::fabs(lanes.ai[b]);
      const double numerator_r = 1.0 / s;
      const double numerator_i = 0.0 / s;
      const double conjugate_r = lanes.ar[b] / s;
      const double conjugate_i = -lanes.ai[b] / s;
      const double cc_rr = conjugate_r * conjugate_r;
      const double cc_ii = conjugate_i * conjugate_i;
      const double rr = numerator_r * conjugate_r;
      const double ii = numerator_i * conjugate_i;
      const double ri = numerator_r * conjugate_i;
      const double ir = numerator_i * conjugate_r;
      const double divisor = cc_rr + cc_ii;
      inverse_r[b] = (rr - ii) / divisor;
      inverse_i[b] = (ri + ir) / divisor;
      last_r[b] = lanes.gr[b];
      last_i[b] = lanes.gi[b];
      const bool on = lanes.branch[b] == adams_lane_large;
      lanes.pr[0][b] = on ? last_r[b] : lanes.pr[0][b];
      lanes.pi[0][b] = on ? last_i[b] : lanes.pi[0][b];
    }
    for (std::size_t k = 1; k <= order; ++k) {
      const double reciprocal = adams_lane_reciprocal[k - 1];
      for (std::size_t b = 0; b < count; ++b) {
        const double difference_r = last_r[b] - reciprocal;
        const double difference_i = last_i[b] - 0.0;
        const double rr = difference_r * inverse_r[b];
        const double ii = difference_i * inverse_i[b];
        const double ri = difference_r * inverse_i[b];
        const double ir = difference_i * inverse_r[b];
        last_r[b] = rr - ii;
        last_i[b] = ri + ir;
        const bool on = lanes.branch[b] == adams_lane_large;
        lanes.pr[k][b] = on ? last_r[b] : lanes.pr[k][b];
        lanes.pi[k][b] = on ? last_i[b] : lanes.pi[k][b];
      }
    }
  }
  for (std::size_t k = 0; k <= order; ++k) {
    const double reciprocal = adams_lane_reciprocal[k];
    for (std::size_t b = 0; b < count; ++b) {
      const bool on = lanes.branch[b] == adams_lane_zero;
      lanes.pr[k][b] = on ? reciprocal : lanes.pr[k][b];
      lanes.pi[k][b] = on ? 0.0 : lanes.pi[k][b];
    }
  }
}
// e^argument for the large branch into `gr`/`gi`: the flow `fr`/`fi`, or under a
// carrier frame the flow times exp(-sigma), as `adams_moment_phi` forms it. Strict.
SOLPROP_NATIVE_STRICT_FUNCTION
inline void adams_lanes_large_exponential(AdamsLanes& lanes,
                                          const ExponentialWeights& weights) {
  SOLPROP_NATIVE_STRICT_BLOCK
  const std::size_t count = lanes.count;
  if (!weights.shifted) {
    for (std::size_t b = 0; b < count; ++b) {
      lanes.gr[b] = lanes.fr[b];
      lanes.gi[b] = lanes.fi[b];
    }
    return;
  }
  const double unshift_r = weights.unshift_re;
  const double unshift_i = weights.unshift_im;
  for (std::size_t b = 0; b < count; ++b) {
    const double rr = lanes.fr[b] * unshift_r;
    const double ii = lanes.fi[b] * unshift_i;
    const double ri = lanes.fr[b] * unshift_i;
    const double ir = lanes.fi[b] * unshift_r;
    lanes.gr[b] = rr - ii;
    lanes.gi[b] = ri + ir;
  }
}
#endif

} // namespace solprop_native
