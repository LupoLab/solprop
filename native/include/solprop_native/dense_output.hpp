// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "solprop_native/adams_march.hpp"
#include "solprop_native/core.hpp"
#include "solprop_native/dense.hpp"
#include "solprop_native/stage_tables.hpp"

/*
 * The native session's dense output over the resident state (solprop SEP 47 section
 * 5.3): the session-side half of `session::Dense`, whose field arithmetic is the
 * general state's.
 *
 * The weights an observation reads are formed here, by the operations of solprop's
 * Python in their order, so that they are the transitional driver's to the bit: the
 * Adams march's interpolant weights over `lagrange::basis_rows` (the native helpers
 * the driver's `integrated_weights` and `lagrange_coefficients` call) and an
 * observation's fraction of its step (`dense::`). This translation unit (dense.cpp)
 * is compiled strict: no contraction, no fast-math.
 *
 * Host code only.
 */
namespace solprop_native::dense {

/** An observation's fraction of its step, `(position - z) / dz`. */
[[nodiscard]] double fraction(double position, double z, double dz);
/** The interaction formulation's interpolant weights over `count` nodes `(1,
    theta...)` at the fraction `theta` of a step `dz` (`_ResidentAdams.observe`): the
    corrector's, `dz` times the integrated weights from 0 to `theta`, in the binding's
    slots. */
[[nodiscard]] AdamsWeights fused_weights(const double* nodes, std::size_t count,
                                         double theta, double dz);
/** The exponential formulation's: the corrector's Lagrange coefficients on the
    `count` nodes and, when `compensated`, the integrated weights from 0 to `xi` with
    the nodes they are taken on. */
[[nodiscard]] ExponentialWeights exponential_weights(const double* nodes,
                                                     std::size_t count, double xi,
                                                     bool compensated);
/** `a b + c` as this translation unit rounds it; never inlined, as
    `march::strict_probe`. */
[[nodiscard, gnu::noinline]] double strict_probe(double a, double b, double c);

} // namespace solprop_native::dense

namespace solprop_native::engines {

/** The Adams march's dense output (solprop's `_ResidentAdams.march_stepped`, `observe`,
    `hold` and `restore`): the interaction formulation's corrector interpolant, or the
    exponential formulation's corrector with its compensation and carrier frame, over
    the last accepted step the engine kept. The cubic Hermite interpolant is a
    comparison method's, which a session refuses. */
class AdamsDense final : public session::Dense {
public:
  AdamsDense(GeneralState& state, const stage::Tables& tables, stage::Anchor& anchor,
             bool fused, bool compensated);
  void stepped(const march::Result& result) override;
  [[nodiscard]] bool crosses(double position, double target, int ulps) const override;
  void observe(double position) override;
  void restore() override;

private:
  GeneralState& state_;
  const stage::Tables& tables_;
  stage::Anchor& anchor_;
  bool fused_;
  bool compensated_;
  /** Whether a step is held, its anchor and step, m, and its nodes `(1, theta...)`,
      `order + 1` of them, with the step's carrier rotations (empty for none). */
  bool held_ = false;
  double z_ = 0.0;
  double dz_ = 0.0;
  std::size_t count_ = 0;
  std::array<double, march::max_history + 1> nodes_{};
  std::vector<double> carrier_;
};

/** DOPRI5's continuous extension in the interaction picture (solprop's
    `StepDriver._dense_record` and `_observe_inside`): the accepted step's extension is
    contracted from the state before it and the step's stages before the step is
    finished (`hold`, which the Runge-Kutta engine's `extend` calls), evaluated at the
    plane's fraction and carried there by the linear flow. */
class RungeKuttaDense final : public session::Dense {
public:
  /** `weights` are the tableau's dense-output weights, one per stage
      (`solprop.stepper.tableau.continuous_extension`). */
  RungeKuttaDense(GeneralState& state, const stage::Tables& tables,
                  stage::Anchor& anchor, const std::vector<double>& weights);
  RungeKuttaDense(const RungeKuttaDense&) = delete;
  RungeKuttaDense& operator=(const RungeKuttaDense&) = delete;
  RungeKuttaDense(RungeKuttaDense&&) = delete;
  RungeKuttaDense& operator=(RungeKuttaDense&&) = delete;
  /** Release the extension's storage, with the session that closes. */
  ~RungeKuttaDense() override;
  /** Contract the accepted, unfinished step from `z` over `dz` into its extension. */
  void hold(double z, double dz);
  void stepped(const march::Result& result) override;
  [[nodiscard]] bool crosses(double position, double target, int ulps) const override;
  void observe(double position) override;
  void restore() override;

private:
  GeneralState& state_;
  const stage::Tables& tables_;
  stage::Anchor& anchor_;
  bool held_ = false;
  double z_ = 0.0;
  double dz_ = 0.0;
};

} // namespace solprop_native::engines
