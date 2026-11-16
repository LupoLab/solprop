// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// A stage's pointwise bodies over the coarse spectrum: StageTerms, its entry (the
// phase increment, the combination with the earlier stages, the lab frame, the noise
// and the pad) and its finish (prefactor, window, absorber, accumulation), with the
// host's block forms; EntryPhase, EntryFactors and EntryBlock, the combination's mode
// and transcendental factors; and the host sizes stage_block and stage_staging_ring.
// Its strict helpers (phase_sum, phase_block, noise_phase) carry fp.hpp's macros. A
// part of kernels.hpp, included through it.
#ifdef SOLPROP_NATIVE_STRICT_TU
#error "a strict translation unit must not include general/stage_terms.hpp"
#endif

#include "exponentials.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/fp.hpp"
#include "solprop_native/runtime/vmath.hpp"
#include "views.hpp"

#include <cstddef>

namespace solprop_native::general {

// The bins a host work item of the stage's entry kernel takes at a time: every bin's
// phase sum is the same sequence of rounded products and additions in term order as
// a per-bin loop, laid out so that the bins vectorise.
constexpr std::size_t stage_block = 64;
// The pinned staging buffers a state's per-trial scalars cross from (`upload_trial`).
constexpr std::size_t stage_staging_ring = 8;

// The pointwise bodies of a stage over the coarse spectrum: its entry (the phase
// increment, the combination with the earlier stages, the lab frame, the noise and the
// pad) and its finish (prefactor, window, absorber, return to the interaction picture
// and accumulation). The entry kernel runs the finish of the stage before it in the
// same traversal when nothing lies between them (solprop SEP 25 section 6.7): both
// are pointwise over the bins, and the combination reads only its own bin's stages.
// How a stage's entry forms its propagator pair (`StageTerms::combine`).
enum class EntryPhase { general, lossless, repeated };

// The transcendental factors of a stage's combination at one bin: the phase's
// `cos b` and `sin b`, its magnitudes `e^a` and `e^-a` (one on a lossless problem),
// and the noise phase's cosine and sine.
struct EntryFactors {
  double cosine = 0.0, sine = 0.0, magnitude = 1.0, inverse = 1.0;
  double noise_cosine = 0.0, noise_sine = 0.0;
};

// A host work item's factors for its block of bins, formed by the vector-maths layer
// in one pass of each function (`StageTerms::entry_factors_block`); the factors a
// stage does not use stay zero.
struct EntryBlock {
  double cosine[stage_block], sine[stage_block];
  double magnitude[stage_block], inverse[stage_block];
  double noise_cosine[stage_block], noise_sine[stage_block];

  [[nodiscard]] EntryFactors at(const std::size_t bin) const {
    return {.cosine = cosine[bin],
            .sine = sine[bin],
            .magnitude = magnitude[bin],
            .inverse = inverse[bin],
            .noise_cosine = noise_cosine[bin],
            .noise_sine = noise_sine[bin]};
  }
};

struct StageTerms {
  ComplexView2 current, fine, whole, solution, error, stage_phase, stage_inverse, noise;
  ComplexView2 flow_input, flow_lab, flow_drive, general_scaled, general_source;
  ComplexView2 projected, absorber;
  // A host source's rows (solprop SEP 49), `(1, n_omega)`: the lab-frame field the
  // entry writes for it, and its source, read in place of the projected response, when
  // `hosted`.
  ComplexView2 host_spectrum, host_input;
  ComplexView3 stages, phi_terms, prefactor_terms;
  RealView tableau, phi_delta, phi_at, prefactor_weights, window;
  IndexView crop, gaps;
  ByteView active;
  std::size_t gap_ranges = 0;
  std::size_t n = 0, n_phi = 0, n_stages = 0, n_prefactor = 0, n_to = 0;
  double inverse_n_to = 0.0, general_pad = 0.0;
  bool add_noise = false, analytic = false, general = false, flow = false;
  bool carried = false, keep_lab = false, zero_response = false;
  bool in_stage_absorber = false;
  bool hosted = false;
  // Every coarse bin is the fine bin of its own index (`crop` is the identity).
  bool identity_crop = false;

  // The stage's phase increment `sum_t dPhi_t Phi_t` at one bin: strict (solprop SEP
  // 25 section 4.2), each product and sum rounded in term order. A zero weight's term
  // is skipped: its product with a finite term is a signed zero, which leaves the
  // running sum (started at +0) unchanged, so the skip is exact. A z table's terms
  // are its values at a trial's positions, weighed by +-1 and mostly zero (SEP 29 G1).
  SOLPROP_NATIVE_STRICT_FUNCTION
  KOKKOS_INLINE_FUNCTION Complex phase_sum(const std::size_t offset,
                                           const std::size_t column,
                                           const std::size_t index) const {
    SOLPROP_NATIVE_STRICT_BLOCK
    double real = 0.0;
    double imag = 0.0;
    for (std::size_t term = 0; term < n_phi; ++term) {
      const double weight = phi_delta(offset + term);
      if (weight == 0.0)
        continue;
      const Complex fixed = phi_terms(term, column, index);
      const double scaled_real = weight * fixed.real();
      const double scaled_imag = weight * fixed.imag();
      real = real + scaled_real;
      imag = imag + scaled_imag;
    }
    return {real, imag};
  }

  // The same sums over a block of bins, laid out so that the bins vectorise.
  SOLPROP_NATIVE_STRICT_FUNCTION
  KOKKOS_INLINE_FUNCTION void
  phase_block(const std::size_t offset, const std::size_t column,
              const std::size_t first, const std::size_t count,
              double (&real)[stage_block], double (&imag)[stage_block]) const {
    SOLPROP_NATIVE_STRICT_BLOCK
    for (std::size_t bin = 0; bin < stage_block; ++bin) {
      real[bin] = 0.0;
      imag[bin] = 0.0;
    }
    for (std::size_t term = 0; term < n_phi; ++term) {
      const double weight = phi_delta(offset + term);
      if (weight == 0.0)
        continue;
      for (std::size_t bin = 0; bin < count; ++bin) {
        const Complex fixed = phi_terms(term, column, first + bin);
        const double scaled_real = weight * fixed.real();
        const double scaled_imag = weight * fixed.imag();
        real[bin] = real[bin] + scaled_real;
        imag[bin] = imag[bin] + scaled_imag;
      }
    }
  }

  // The noise's phase: strict (solprop SEP 25 section 4.2).
  SOLPROP_NATIVE_STRICT_FUNCTION
  KOKKOS_INLINE_FUNCTION Complex noise_phase(const std::size_t stage,
                                             const std::size_t column,
                                             const std::size_t index) const {
    SOLPROP_NATIVE_STRICT_BLOCK
    Complex sum{};
    for (std::size_t term = 0; term < n_phi; ++term) {
      const double weight = phi_at(stage * n_phi + term);
      // Exact, as in `phase_sum`.
      if (weight != 0.0)
        sum = strict_accumulate(sum, weight, phi_terms(term, column, index));
    }
    return sum;
  }

  // The factors of stage `stage`'s combination at one bin, one bin at a time (a
  // device's entry; the host forms a block's together, `entry_factors_block`). The
  // propagator `exp(phase)` and its inverse `exp(-phase)` share their cosine and sine,
  // each magnitude is its own `exp`, and a lossless problem's are exactly one, taken
  // without the calls (`combine`).
  template <EntryPhase Mode>
  KOKKOS_INLINE_FUNCTION EntryFactors entry_factors(const std::size_t stage,
                                                    const std::size_t column,
                                                    const std::size_t index,
                                                    const Complex phase) const {
    EntryFactors factors;
    if constexpr (Mode != EntryPhase::repeated)
      cosine_sine(phase.imag(), factors.cosine, factors.sine);
    if constexpr (Mode == EntryPhase::general) {
      factors.magnitude = Kokkos::exp(phase.real());
      factors.inverse = Kokkos::exp(-phase.real());
    }
    if (add_noise) {
      const Complex phase_at = noise_phase(stage, column, index);
      factors.noise_cosine = Kokkos::cos(phase_at.imag());
      factors.noise_sine = Kokkos::sin(phase_at.imag());
    }
    return factors;
  }

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  // Whether the first `count` of a block's phases have real parts that are all zeros.
  static bool lossless_block(const double (&real)[stage_block],
                             const std::size_t count) {
    bool zero = true;
    for (std::size_t bin = 0; bin < count; ++bin)
      zero = zero && real[bin] == 0.0;
    return zero;
  }

  // A host work item's factors for its `count` bins from `first`, each function one
  // pass of the vector-maths layer over the block (solprop ADR-059): the phases' sines
  // and cosines, their magnitudes and inverse magnitudes, and the noise phases'.
  template <EntryPhase Mode>
  [[maybe_unused]] void
  entry_factors_block(const std::size_t stage, const std::size_t column,
                      const std::size_t first, const std::size_t count,
                      const double (&real)[stage_block],
                      const double (&imag)[stage_block], EntryBlock& block) const {
    if constexpr (Mode != EntryPhase::repeated)
      vmath::sincos(imag, block.sine, block.cosine, count);
    if constexpr (Mode == EntryPhase::general) {
      // A block whose phases' real parts are all (signed) zeros -- a lossy problem's
      // loss often lies in a few bands -- has both magnitudes `exp(+-0)`, exactly one,
      // without the calls.
      if (lossless_block(real, count)) {
        for (std::size_t bin = 0; bin < count; ++bin) {
          block.magnitude[bin] = 1.0;
          block.inverse[bin] = 1.0;
        }
      } else {
        double negated[stage_block];
        for (std::size_t bin = 0; bin < count; ++bin)
          negated[bin] = -real[bin];
        vmath::exp(real, block.magnitude, count);
        vmath::exp(negated, block.inverse, count);
      }
    }
    if (add_noise) {
      double angle[stage_block]{};
      for (std::size_t bin = 0; bin < count; ++bin)
        angle[bin] = noise_phase(stage, column, first + bin).imag();
      vmath::sincos(angle, block.noise_sine, block.noise_cosine, count);
    }
  }
#endif

  // The combination of stage `stage` at one bin, from the stage's phase increment and
  // its factors (`entry_factors`). The propagator `exp(phase)` and its inverse
  // `exp(-phase)`, which the finish applies, share their cosine and sine (solprop SEP
  // 25 section 6.7): the angle's parts are even and odd, so `exp(-phase)` is
  // `e^{-a} (cos b, -sin b)` with its own magnitude, never a reciprocal of the forward
  // value. A coupled flow's residual reads the phase. A lossless problem
  // (`EntryPhase::lossless`: every Phi term's real part exactly zero, so the phase's
  // is a signed zero) takes both magnitudes as `exp(+-0)`, exactly one, without the
  // calls. A lossless stage whose phase weights equal the stage before's
  // (`EntryPhase::repeated`) has that stage's inverse still in place, and its
  // propagator is that inverse's conjugate, `(cos b, -(-sin b))`, to the bit.
  // Returns the propagator.
  template <EntryPhase Mode = EntryPhase::general>
  KOKKOS_INLINE_FUNCTION Complex combine(const std::size_t stage, const double dz,
                                         const double solution_weight,
                                         const double error_weight,
                                         const std::size_t column,
                                         const std::size_t index, const Complex phase,
                                         const EntryFactors& factors) const {
    Complex propagator;
    if constexpr (Mode == EntryPhase::repeated) {
      const Complex inverse = stage_inverse(column, index);
      propagator = Complex(inverse.real(), -inverse.imag());
    } else if constexpr (Mode == EntryPhase::lossless) {
      propagator = Complex(factors.cosine, factors.sine);
      stage_inverse(column, index) = Complex(factors.cosine, -factors.sine);
    } else {
      propagator =
          Complex(factors.magnitude * factors.cosine, factors.magnitude * factors.sine);
      stage_inverse(column, index) =
          Complex(factors.inverse * factors.cosine, -(factors.inverse * factors.sine));
    }
    if (flow)
      stage_phase(column, index) = phase;
    Complex value{};
    if (carried) {
      value = propagator * flow_input(column, index);
    } else {
      Complex accumulated{};
      for (std::size_t previous = 0; previous < stage; ++previous)
        accumulated +=
            tableau(stage * n_stages + previous) * stages(previous, column, index);
      value = propagator * (current(column, index) + dz * accumulated);
    }
    if (add_noise)
      value += noise(column, index) * Complex(factors.noise_cosine, factors.noise_sine);
    const double analytic_scale =
        analytic && index > 0 && 2 * crop(index) < n_to ? 2.0 : 1.0;
    fine(column, crop(index)) = analytic_scale * inverse_n_to * value;
    // The host source evaluates the field itself, unscaled (SEP 49 section 6.2).
    if (hosted)
      host_input(column, index) = value;
    if (keep_lab)
      flow_lab(column, index) = value;
    // The general route synthesises on the coarse grid: the pad scale and the
    // analytic doubling go on the channel spectra, the 1/n_to on its transform.
    if (general)
      general_scaled(column, index) =
          value * (general_pad * general_doubling(crop(index), n_to));
    if (zero_response && flow) {
      flow_drive(column, index) = Complex{};
    } else if (zero_response) {
      stages(stage, column, index) = Complex{};
      if (stage == 0) {
        solution(column, index) = Complex{};
        error(column, index) = Complex{};
      } else {
        solution(column, index) += solution_weight * Complex{};
        error(column, index) += error_weight * Complex{};
      }
    }
    return propagator;
  }

  // The fine bins no coarse bin fills, positions `[begin, end)` of this column's
  // list, zeroed for the inverse transform (which may destroy its input). The list is
  // held as runs of adjacent bins, `gaps(3 r + {0, 1, 2})` the run's first position,
  // first bin and length, so each run zeroes contiguously.
  KOKKOS_INLINE_FUNCTION void zero_gaps(const std::size_t column,
                                        const std::size_t begin,
                                        const std::size_t end) const {
    for (std::size_t run = 0; run < gap_ranges; ++run) {
      const std::size_t position = gaps(3 * run);
      const std::size_t bin = gaps(3 * run + 1);
      const std::size_t length = gaps(3 * run + 2);
      const std::size_t lower = begin > position ? begin : position;
      const std::size_t upper = end < position + length ? end : position + length;
      for (std::size_t gap = lower; gap < upper; ++gap)
        fine(column, bin + (gap - position)) = Complex{};
    }
  }

  // The finish of stage `stage` at one bin, with the inverse propagator its
  // combination formed.
  KOKKOS_INLINE_FUNCTION void finish(const std::size_t stage,
                                     const double solution_weight,
                                     const double error_weight,
                                     const std::size_t column,
                                     const std::size_t index) const {
    // A zero weight is skipped: a z table's terms are its values at a trial's
    // positions, one-hot weighted (SEP 29 G1). The sum's value is unchanged; at most
    // the sign of an exactly zero sum could differ.
    Complex value{};
    if (hosted) {
      // The host's source carries its own factors (solprop SEP 49 section 4): the
      // collar's product alone, as the reference's `stage_output` with unit factors.
      value = window(index) * host_spectrum(column, index);
    } else {
      Complex prefactor{};
      for (std::size_t term = 0; term < n_prefactor; ++term) {
        const double weight = prefactor_weights(stage * n_prefactor + term);
        if (weight != 0.0)
          prefactor += weight * prefactor_terms(term, column, index);
      }
      value =
          prefactor * window(index) *
          (general ? general_source(column, index) : projected(column, crop(index)));
    }
    if (in_stage_absorber && active(index) != 0U)
      value += absorber(column, crop(index));
    value *= stage_inverse(column, index);
    if (flow) {
      flow_drive(column, index) = value;
      return;
    }
    stages(stage, column, index) = value;
    if (stage == 0) {
      solution(column, index) = solution_weight * value;
      error(column, index) = error_weight * value;
    } else {
      solution(column, index) += solution_weight * value;
      error(column, index) += error_weight * value;
    }
  }

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  // Whether a host work item takes its finish and combination as the block forms
  // below: a stage that projects a source into the stage rows, off a coupled flow and
  // the general route, whose stores the per-bin forms keep.
  [[nodiscard]] bool blocks_vectorise() const noexcept {
    return !flow && !general && !zero_response && !hosted;
  }

  // `finish` over the `count` bins of a block from `first`, each bin the same
  // arithmetic in the same order, the block's parts held as separate real and
  // imaginary lanes so that the bins vectorise; only the crop's gathers stay per bin.
  [[maybe_unused]] void finish_block(const std::size_t stage,
                                     const double solution_weight,
                                     const double error_weight,
                                     const std::size_t column, const std::size_t first,
                                     const std::size_t count) const {
    double real[stage_block];
    double imag[stage_block];
    double source_real[stage_block];
    double source_imag[stage_block];
    for (std::size_t bin = 0; bin < count; ++bin) {
      real[bin] = 0.0;
      imag[bin] = 0.0;
    }
    for (std::size_t term = 0; term < n_prefactor; ++term) {
      const double weight = prefactor_weights(stage * n_prefactor + term);
      if (weight == 0.0)
        continue;
      const Complex* values = &prefactor_terms(term, column, first);
      for (std::size_t bin = 0; bin < count; ++bin) {
        real[bin] += weight * values[bin].real();
        imag[bin] += weight * values[bin].imag();
      }
    }
    const std::size_t* bins = &crop(first);
    const auto gather = [&](const Complex* row) {
      if (identity_crop)
        for (std::size_t bin = 0; bin < count; ++bin) {
          source_real[bin] = row[first + bin].real();
          source_imag[bin] = row[first + bin].imag();
        }
      else
        for (std::size_t bin = 0; bin < count; ++bin) {
          source_real[bin] = row[bins[bin]].real();
          source_imag[bin] = row[bins[bin]].imag();
        }
    };
    gather(&projected(column, 0));
    const double* weights = &window(first);
    for (std::size_t bin = 0; bin < count; ++bin) {
      const double scaled_real = real[bin] * weights[bin];
      const double scaled_imag = imag[bin] * weights[bin];
      real[bin] = scaled_real * source_real[bin] - scaled_imag * source_imag[bin];
      imag[bin] = scaled_real * source_imag[bin] + scaled_imag * source_real[bin];
    }
    if (in_stage_absorber) {
      gather(&absorber(column, 0));
      const unsigned char* band = &active(first);
      for (std::size_t bin = 0; bin < count; ++bin) {
        const bool inside = band[bin] != 0U;
        real[bin] = inside ? real[bin] + source_real[bin] : real[bin];
        imag[bin] = inside ? imag[bin] + source_imag[bin] : imag[bin];
      }
    }
    const Complex* inverse = &stage_inverse(column, first);
    Complex* row = &stages(stage, column, first);
    Complex* total = &solution(column, first);
    Complex* estimate = &error(column, first);
    for (std::size_t bin = 0; bin < count; ++bin) {
      const double value_real =
          real[bin] * inverse[bin].real() - imag[bin] * inverse[bin].imag();
      const double value_imag =
          real[bin] * inverse[bin].imag() + imag[bin] * inverse[bin].real();
      row[bin] = Complex(value_real, value_imag);
      if (stage == 0) {
        total[bin] =
            Complex(solution_weight * value_real, solution_weight * value_imag);
        estimate[bin] = Complex(error_weight * value_real, error_weight * value_imag);
      } else {
        total[bin] = Complex(total[bin].real() + solution_weight * value_real,
                             total[bin].imag() + solution_weight * value_imag);
        estimate[bin] = Complex(estimate[bin].real() + error_weight * value_real,
                                estimate[bin].imag() + error_weight * value_imag);
      }
    }
  }

  // `combine` over the `count` bins of a block from `first`, from the block's factors,
  // in the lanes of `finish_block`; with `store_whole`, each propagator is also the
  // whole step's.
  template <EntryPhase Mode>
  [[maybe_unused]] void
  combine_block(const std::size_t stage, const double dz, const std::size_t column,
                const std::size_t first, const std::size_t count,
                const EntryBlock& factors, const bool store_whole) const {
    double real[stage_block];
    double imag[stage_block];
    double sum_real[stage_block];
    double sum_imag[stage_block];
    Complex* inverse = &stage_inverse(column, first);
    for (std::size_t bin = 0; bin < count; ++bin) {
      if constexpr (Mode == EntryPhase::repeated) {
        real[bin] = inverse[bin].real();
        imag[bin] = -inverse[bin].imag();
      } else if constexpr (Mode == EntryPhase::lossless) {
        real[bin] = factors.cosine[bin];
        imag[bin] = factors.sine[bin];
        inverse[bin] = Complex(factors.cosine[bin], -factors.sine[bin]);
      } else {
        real[bin] = factors.magnitude[bin] * factors.cosine[bin];
        imag[bin] = factors.magnitude[bin] * factors.sine[bin];
        inverse[bin] = Complex(factors.inverse[bin] * factors.cosine[bin],
                               -(factors.inverse[bin] * factors.sine[bin]));
      }
    }
    if (store_whole) {
      Complex* propagator = &whole(column, first);
      for (std::size_t bin = 0; bin < count; ++bin)
        propagator[bin] = Complex(real[bin], imag[bin]);
    }
    for (std::size_t bin = 0; bin < count; ++bin) {
      sum_real[bin] = 0.0;
      sum_imag[bin] = 0.0;
    }
    for (std::size_t previous = 0; previous < stage; ++previous) {
      const double weight = tableau(stage * n_stages + previous);
      const Complex* earlier = &stages(previous, column, first);
      for (std::size_t bin = 0; bin < count; ++bin) {
        sum_real[bin] += weight * earlier[bin].real();
        sum_imag[bin] += weight * earlier[bin].imag();
      }
    }
    const Complex* state = &current(column, first);
    for (std::size_t bin = 0; bin < count; ++bin) {
      const double input_real = state[bin].real() + dz * sum_real[bin];
      const double input_imag = state[bin].imag() + dz * sum_imag[bin];
      const double value_real = real[bin] * input_real - imag[bin] * input_imag;
      const double value_imag = real[bin] * input_imag + imag[bin] * input_real;
      real[bin] = value_real;
      imag[bin] = value_imag;
    }
    if (add_noise) {
      const Complex* draw = &noise(column, first);
      for (std::size_t bin = 0; bin < count; ++bin) {
        const double noise_real = draw[bin].real() * factors.noise_cosine[bin] -
                                  draw[bin].imag() * factors.noise_sine[bin];
        const double noise_imag = draw[bin].real() * factors.noise_sine[bin] +
                                  draw[bin].imag() * factors.noise_cosine[bin];
        real[bin] += noise_real;
        imag[bin] += noise_imag;
      }
    }
    Complex* padded = &fine(column, 0);
    const std::size_t* bins = &crop(first);
    if (identity_crop && !analytic) {
      for (std::size_t bin = 0; bin < count; ++bin)
        padded[first + bin] =
            Complex(inverse_n_to * real[bin], inverse_n_to * imag[bin]);
      return;
    }
    for (std::size_t bin = 0; bin < count; ++bin) {
      const double analytic_scale =
          analytic && first + bin > 0 && 2 * bins[bin] < n_to ? 2.0 : 1.0;
      const double scale = analytic_scale * inverse_n_to;
      padded[bins[bin]] = Complex(scale * real[bin], scale * imag[bin]);
    }
  }
#endif
};

} // namespace solprop_native::general
