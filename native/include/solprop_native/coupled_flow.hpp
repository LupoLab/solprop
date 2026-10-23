// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/core.hpp"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace solprop_native {

/**
 * Refuse a coupled-flow problem whose extents or data this action cannot execute.
 *
 * Called before any allocation. A problem that does not declare the flow passes.
 */
void require_coupled_flow(const Problem& problem);

/**
 * The per-frequency channel mixing of solprop's `linear.coupled_flow`.
 *
 * The execution twin of `NumpyRefBackend._channels` and `_residual_drive` in solprop:
 * on the bins a section's generator is non-zero at, the interaction picture of a
 * coupled block carries every stage input forward by `U(c_i h)` and its source back
 * by `U(c_i h)^-1`, the accepted step forward by `U(h)`, and a frozen-residual flow
 * adds `R_s(z_i) u` to each stage's lab-frame source before it is carried back. The
 * actions themselves are formed by the host adapter from the section's Schur factors
 * (as solprop's reference forms them) and uploaded into persistent Views only when the
 * step's section or length changes; the residual's terms and frozen blocks are
 * uploaded once, and its per-stage combination is formed here.
 *
 * Every product is a fixed-order serial loop over the channel axis inside one output
 * element, so the result does not depend on the thread count. All Views are
 * allocated at construction; no member allocates afterwards.
 */
class CoupledFlowAction {
public:
  using ComplexView2 = Kokkos::View<Complex**, Layout, MemorySpace>;
  using ComplexView3 = Kokkos::View<Complex***, Layout, MemorySpace>;
  using ComplexView4 = Kokkos::View<Complex****, Layout, MemorySpace>;
  using RealView2 = Kokkos::View<double**, Layout, MemorySpace>;
  using IndexView = Kokkos::View<std::size_t*, Layout, MemorySpace>;
  using IndexView2 = Kokkos::View<std::size_t**, Layout, MemorySpace>;

  CoupledFlowAction(const Problem& problem, const ExecutionSpace& kernels);
  ~CoupledFlowAction();
  CoupledFlowAction(const CoupledFlowAction&) = delete;
  CoupledFlowAction& operator=(const CoupledFlowAction&) = delete;
  CoupledFlowAction(CoupledFlowAction&&) = delete;
  CoupledFlowAction& operator=(CoupledFlowAction&&) = delete;

  /**
   * Load the forward and inverse actions of every stage and of the whole step.
   *
   * Each is row-major `(n_stages + 1, n_active, n_modes, n_modes)`; slot `n_stages`
   * is the whole step. `identity` marks a slot whose action is the identity (a zero
   * stage fraction), which is then not applied at all.
   */
  void set_actions(const std::vector<Complex>& forward,
                   const std::vector<Complex>& inverse,
                   const std::vector<unsigned char>& identity);
  /**
   * Form every stage's and the whole step's actions of one section natively.
   *
   * `fractions` holds each stage's node and, last, the whole step's one; the
   * lengths are `fraction * dz`. A bin whose eigenbasis is well conditioned takes
   * `W exp(+-l Lambda) Y`, the others solprop's reference scaling, sixteen-term
   * series and squaring of `l T` and the back-transform `Z F Z^H`; a zero fraction is
   * the identity and a repeated one is copied. Parallel over the active bins, each
   * bin's work sequential and in fixed order, so the result does not depend on the
   * thread count; nothing is allocated. Returns the whole step's measured
   * `norm(U)_2 norm(U^-1)_2` per active bin, for the host's certificate check.
   */
  const std::vector<double>& form_actions(std::size_t section, double dz,
                                          const std::vector<double>& fractions);
  /**
   * Copy the held forward (or inverse) actions to the host, row-major
   * `(n_stages + 1, n_active, n_modes, n_modes)`: an observation for tests.
   */
  [[nodiscard]] std::vector<Complex> actions_to_host(bool inverse) const;
  /** Whether the actions can be formed natively (the eigenbasis was loaded). */
  [[nodiscard]] bool forms_actions() const noexcept { return forms_; }
  /** Load one trial's frozen section and its stages' residual weights. */
  void set_trial(std::size_t section, const std::vector<double>& weights);
  /** Whether actions have been loaded. */
  [[nodiscard]] bool ready() const noexcept { return ready_; }
  /** Whether the flow carries an explicit residual. */
  [[nodiscard]] bool residual() const noexcept { return terms_ != 0; }
  /** Whether a stage's action is the identity. */
  [[nodiscard]] bool identity(std::size_t stage) const;

  /** The carried stage input `U (u + h sum a_ij k_j)`, before the propagator. */
  [[nodiscard]] const ComplexView2& input() const noexcept { return input_; }
  /** The lab-frame stage input, stored for the residual. */
  [[nodiscard]] const ComplexView2& lab() const noexcept { return lab_; }
  /** The stage's source before it is carried back. */
  [[nodiscard]] const ComplexView2& drive() const noexcept { return drive_; }

  /** Form `input()` for one stage from the state and the earlier stages. */
  void carry_input(std::size_t stage, double dz, const ComplexView2& current,
                   const ComplexView3& stages, const RealView& tableau);
  /** Add `exp(-dPhi) R_s(z_i) lab` to `drive()` for one stage, `dPhi` its phase. */
  void add_residual(std::size_t stage, const ComplexView2& stage_phase);
  /** Carry `drive()` back into the stage and accumulate both weighted sums. */
  void carry_back(std::size_t stage, double solution_weight, double error_weight,
                  const ComplexView3& stages, const ComplexView2& solution,
                  const ComplexView2& error);
  /** Carry the state, the solution and the error sum forward by the whole step. */
  void carry_step(const ComplexView2& current, const ComplexView2& solution,
                  const ComplexView2& error);
  /** The whole step's carried state, solution and error sum, after `carry_step`. */
  [[nodiscard]] const ComplexView2& carried_state() const noexcept { return state_; }
  [[nodiscard]] const ComplexView2& carried_solution() const noexcept {
    return solution_;
  }
  [[nodiscard]] const ComplexView2& carried_error() const noexcept { return error_; }

  // The resident data the exponential Adams formulation's coupled passes read (solprop
  // SEP 16 phase 6): each bin's one-based index in the block, zero off it; every
  // section's eigenbasis `(W, Y, lambda)`; and a frozen-residual flow's terms and
  // frozen blocks, zero off the scored bins.
  [[nodiscard]] const IndexView& block_slots() const noexcept { return slot_; }
  [[nodiscard]] const ComplexView4& left() const noexcept { return left_; }
  [[nodiscard]] const ComplexView4& right() const noexcept { return right_; }
  [[nodiscard]] const ComplexView3& eigenvalues() const noexcept {
    return eigenvalues_;
  }
  [[nodiscard]] const ComplexView4& residual_terms() const noexcept {
    return terms_view_;
  }
  [[nodiscard]] const ComplexView4& frozen() const noexcept { return frozen_; }
  [[nodiscard]] std::size_t n_terms() const noexcept { return terms_; }
  [[nodiscard]] std::size_t n_active() const noexcept { return active_; }
  [[nodiscard]] std::size_t n_sections() const noexcept { return sections_; }
  /** Whether every section's every bin is on the eigenbasis route. */
  [[nodiscard]] bool full_route() const noexcept { return forms_ && fallback_ == 0; }

  /** Persistent allocations, by role, in bytes. */
  [[nodiscard]] std::map<std::string, std::size_t> memory_ledger() const;
  /** The same rows, all zero, for a problem that carries no coupled flow. */
  [[nodiscard]] static std::map<std::string, std::size_t> empty_ledger();
  /** Release every View; repeated calls have no effect. */
  void close() noexcept;

private:
  std::size_t n_modes_{}, n_omega_{}, n_stages_{}, active_{}, sections_{}, terms_{};
  std::size_t section_{}, fallback_{};
  bool ready_{}, closed_{}, forms_{};
  std::vector<unsigned char> identity_;
  // The per-stage counts of `active_terms_`, kept on the host for the launch.
  std::vector<std::size_t> listed_counts_;
  ExecutionSpace kernels_;
  IndexView slot_;
  ComplexView4 forward_, inverse_, frozen_, terms_view_;
  RealView2 weights_;
  IndexView2 active_terms_;
  IndexView active_counts_;
  IndexView2::host_mirror_type host_active_terms_;
  IndexView::host_mirror_type host_active_counts_;
  ComplexView2 input_, lab_, drive_, state_, solution_, error_;
  // The native formation's data and scratch.
  ComplexView4 left_, right_, fallback_work_;
  ComplexView3 eigenvalues_, fallback_vectors_, fallback_triangular_, exponents_, gram_;
  IndexView2 route_;
  RealView fractions_, measured_;
  RealView::host_mirror_type host_fractions_, host_measured_;
  std::vector<double> measured_host_;
};

} // namespace solprop_native
