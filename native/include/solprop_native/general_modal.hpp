// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/core.hpp"
#include "solprop_native/dispersive.hpp"
#include "solprop_native/raman_fft.hpp"

#include <KokkosFFT.hpp>

#include <cstddef>
#include <map>
#include <memory>
#include <string>

namespace solprop_native {

/** Stored pair entries of a real symmetric three-component history: xx yy zz xy xz yz.
 */
inline constexpr std::size_t general_pair_histories = 6;

/**
 * The vector length of the general source's teams: the adjacent output bins one
 * projection team accumulates, and the lanes a device gives a node's samples.
 *
 * The host runs a team's vector range as one vectorised loop and ignores the
 * length; for the projection it is the block of bins (measured on the general
 * route's source on native CPU before the first public release).
 */
inline constexpr std::size_t general_vector_lanes = 8;

/**
 * The block of adjacent output bins one projection team accumulates.
 *
 * The widest power of two from `general_vector_lanes` to 64 that still gives every
 * one of `concurrency` threads a team: wider blocks cost less per row walked, and
 * the blocks do not change any bin's sum, so the choice moves no bit (measured on
 * the general route's source on native CPU before the first public release).
 */
constexpr std::size_t general_projection_lanes(const std::size_t modes,
                                               const std::size_t n_omega,
                                               const std::size_t concurrency) {
  std::size_t lanes = general_vector_lanes;
  while (lanes < 64 && modes * ((n_omega + 2 * lanes - 1) / (2 * lanes)) >= concurrency)
    lanes *= 2;
  return lanes;
}

/**
 * Refuse a general modal problem whose extents or data this source cannot execute.
 *
 * Called before any allocation. A problem that does not declare the route passes.
 */
void require_general_modal(const Problem& problem);

/**
 * The analytic-signal doubling of one coarse bin placed at fine bin `fine`.
 *
 * The zero and Nyquist bins of the fine one-sided spectrum are kept and every other
 * bin is doubled (solprop `fourier.analytic_signal`), so the complex inverse transform
 * of the doubled spectrum is the analytic signal of the real field.
 */
KOKKOS_INLINE_FUNCTION double general_doubling(const std::size_t fine,
                                               const std::size_t n_to) {
  return fine == 0 || 2 * fine == n_to ? 1.0 : 2.0;
}

/**
 * The nonlinear source of solprop's general modal route on one state.
 *
 * The execution twin of `solprop.backends.numpy_ref.general_modal.general_source`:
 * per node chunk, synthesise the three Cartesian components on the coarse grid from
 * the frequency-dependent map S, place them in the fine spectrum, one complex
 * inverse transform to the analytic signal, the tensor cubic (on the real field or
 * on the analytic signal, as declared) with the optional delayed tensor response on
 * the doubled axis, one real forward transform of the real polarisation, and the
 * contraction with Q. Every reduction is a fixed-order serial loop inside one
 * output element, so the result does not depend on the thread count.
 *
 * All Views and FFT plans are allocated at construction; `evaluate` allocates
 * nothing.
 */
class GeneralModalSource {
public:
  using ComplexView2 = Binary64::ComplexView2;
  using RealView2 = Binary64::RealView2;
  using IndexView = Kokkos::View<std::size_t*, Layout, MemorySpace>;

  /**
   * Allocate and upload every View; `plan` creates the transforms afterwards.
   *
   * `delayed_transforms` runs the doubled-axis pair transforms through the engine's
   * Raman FFT provider, whose per-row execution keeps them thread-count invariant on
   * the macOS vDSP build, as it keeps the existing Raman convolution.
   */
  explicit GeneralModalSource(const Problem& problem);
  ~GeneralModalSource();
  GeneralModalSource(const GeneralModalSource&) = delete;
  GeneralModalSource& operator=(const GeneralModalSource&) = delete;
  GeneralModalSource(GeneralModalSource&&) = delete;
  GeneralModalSource& operator=(GeneralModalSource&&) = delete;

  /** The coarse lab-frame channel spectra, times the pad scale and the doubling. */
  [[nodiscard]] const ComplexView2& scaled() const noexcept { return scaled_; }
  /** The synthesis map `evaluate` reads: a moving guide's at the last combined stage.
   */
  [[nodiscard]] const ComplexView& synthesis() const noexcept { return synthesis_; }
  /** The projection map `evaluate` reads, likewise. */
  [[nodiscard]] const ComplexView& projection() const noexcept { return projection_; }
  /** The modal source `-i omega Q . P` on the coarse grid after `evaluate`. */
  [[nodiscard]] const ComplexView2& source() const noexcept { return source_; }
  /**
   * Whether the source can be nonzero: a node's instantaneous tensor is, or SEP 6's
   * dispersive response acts in a gas region.
   */
  [[nodiscard]] bool responds() const noexcept { return responds_; }
  /** Whether a regional tensor or delayed response forms the node polarisation. */
  [[nodiscard]] bool regional() const noexcept { return regional_; }
  /** SEP 6's dispersive response in the gas regions, or null. */
  [[nodiscard]] GeneralDispersive* dispersive() const noexcept {
    return dispersive_.get();
  }
  /** Whether the maps move along z: more than one Chebyshev term (schema 14). */
  [[nodiscard]] bool moving() const noexcept { return map_terms_ > 1; }
  /**
   * Load a trial's map weights, `(n_stages, map terms)` row-major, host-side.
   *
   * Each stage's weights must be zero outside one section; that section is found
   * here, on the host, so the combination reads only its `n_deg` terms. Empty for
   * maps that do not move. Allocates nothing.
   */
  void set_weights(const std::vector<double>& weights);
  /**
   * Evaluate a moving guide's synthesis and projection at one stage.
   *
   * The weighted sum of the stage's section's Chebyshev terms, term by term from
   * zero, written into the Views `evaluate` reads (solprop
   * `GeneralMaps.at`); no effect on maps that do not move.
   */
  void combine(std::size_t stage);
  /** Create the persistent FFT plans over the chunk buffers. */
  void plan();
  /** The provider of the delayed pair transforms, or "none" without them. */
  [[nodiscard]] std::string delayed_fft_provider() const;
  /** Evaluate the source of the state in `scaled()` into `source()`. */
  void evaluate();
  /** Persistent allocations, by role, in bytes. */
  [[nodiscard]] std::map<std::string, std::size_t> memory_ledger() const;
  [[nodiscard]] CpuSizing execution_sizing() const {
    auto result = kernel_pool_.sizing();
    result.merge(fft_pool_.sizing());
    result.merge(raman_pool_.sizing());
    result.merge(raman_tail_pool_.sizing());
    return result;
  }
  /** The same rows, all zero, for a problem that does not take the route. */
  [[nodiscard]] static std::map<std::string, std::size_t> empty_ledger();
  /** Release plans before Views; repeated calls have no effect. */
  void close() noexcept;

  // NVCC extended lambdas require their enclosing member functions to be public.
  void combine_maps(std::size_t stage);
  void synthesise(std::size_t first, std::size_t count);
  void respond(std::size_t first, std::size_t count);
  template <std::size_t Components>
  void respond_components(std::size_t first, std::size_t count);
  void delayed(std::size_t first, std::size_t count);
  void project(std::size_t first, std::size_t count);
  template <std::size_t Lanes> void project_lanes(std::size_t first, std::size_t count);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  void project_device(std::size_t first, std::size_t count);
#endif
  void finish();

private:
  using RealPlan = KokkosFFT::Plan<ExecutionSpace, RealView2, ComplexView2>;
  using DelayedForwardPlan = RamanFftPlan<ExecutionSpace, RealView2, ComplexView2>;
  using DelayedInversePlan = RamanFftPlan<ExecutionSpace, ComplexView2, RealView2>;
  using ComplexPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView2, ComplexView2>;

  std::size_t n_modes_{}, n_omega_{}, n_to_{}, components_{}, nodes_{}, chunk_{};
  std::size_t terms_{};
  /** Chebyshev terms of the maps along z (sections x n_deg), and per section. */
  std::size_t map_terms_{1}, map_n_deg_{1}, n_stages_{};
  /** The fine bin of the first coarse bin; the crop is contiguous from it. */
  std::size_t crop_offset_{};
  /** The projection's block of bins, `general_projection_lanes` at construction. */
  std::size_t projection_lanes_{};
  bool real_form_{}, responds_{}, regional_{}, weights_ready_{};
  double epsilon0_{};
  CpuExecutionPool kernel_pool_{CpuPool::kernels}, fft_pool_{CpuPool::fft};
  CpuExecutionPool raman_pool_{CpuPool::raman_fft},
      raman_tail_pool_{CpuPool::raman_tail_fft};
  ExecutionSpace kernels_, transforms_, delayed_transforms_;
  ExecutionSpace response_, tail_response_, dense_, tail_dense_, maps_;
  const ExecutionSpace& response_for(std::size_t count) const {
    return count == chunk_ ? response_ : tail_response_;
  }
  const ExecutionSpace& dense_for(std::size_t count) const {
    return count == chunk_ ? dense_ : tail_dense_;
  }
  ComplexView synthesis_, projection_, frequency_factor_, delayed_spectrum_;
  /** A moving guide's coefficients; `synthesis_`, `projection_` are then one stage's.
   */
  ComplexView synthesis_terms_, projection_terms_;
  /** A moving guide's trial weights, a persistent host copy, and each stage's section.
   */
  RealView2 map_weights_;
  RealView2::host_mirror_type host_map_weights_;
  std::vector<std::size_t> stage_sections_;
  RealView tensor_, delayed_tensor_;
  IndexView node_response_;
  ComplexView2 scaled_, source_;
  ComplexView2 node_spectrum_, node_field_, node_source_;
  RealView2 node_polarisation_;
  // Aliases of the active prefix: tail plans must not transform unused chunk rows.
  ComplexView2 tail_spectrum_, tail_field_, tail_source_;
  RealView2 tail_polarisation_;
  RealView2 node_drive_, node_histories_, node_delayed_;
  ComplexView2 node_drive_spectrum_, node_product_;
  RealView2 tail_drive_, tail_histories_;
  ComplexView2 tail_drive_spectrum_, tail_product_;
  std::unique_ptr<ComplexPlan> inverse_plan_;
  std::unique_ptr<RealPlan> forward_plan_;
  std::unique_ptr<ComplexPlan> tail_inverse_plan_;
  std::unique_ptr<RealPlan> tail_forward_plan_;
  std::unique_ptr<DelayedForwardPlan> drive_plan_;
  std::unique_ptr<DelayedInversePlan> history_plan_;
  std::unique_ptr<DelayedForwardPlan> tail_drive_plan_;
  std::unique_ptr<DelayedInversePlan> tail_history_plan_;
  std::unique_ptr<GeneralDispersive> dispersive_;
  bool closed_{};
};

} // namespace solprop_native
