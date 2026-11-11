// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "solprop_native/adams_march.hpp"
#include "solprop_native/core.hpp"
#include "solprop_native/rk_march.hpp"
#include "solprop_native/session.hpp"
#include "solprop_native/stage_tables.hpp"

/*
 * The engines the native marches hand their field arithmetic to (solprop SEP 39 and
 * SEP 45): the resident engines over a native state, and the scripted engines of the
 * decision-trace tests, which score a script of norms in place of the field. The
 * marches (`march::run`, `rk::run`) own every decision; an engine only forms a trial,
 * adopts it and reads the committed state's boundary energies.
 *
 * Python-free: the scripted engines log what the march asked into C++ records, which
 * the binding converts for solprop's tests.
 *
 * Host code only.
 */
namespace solprop_native::engines {

/** A failure injected at one of the resident engine's stages, after that stage has run
    `after` more times: solprop's lifecycle tests fail the native march with it. With
    `signal`, that signal is raised in the native call instead and the stage goes on,
    as a signal arriving while native code holds no interpreter lock does. */
struct MarchFault {
  march::Stage stage = march::Stage::none;
  std::size_t after = 0;
  march::Category category = march::Category::backend;
  int signal = 0;
};

/** A scalar carrier frame's shift and rotations (solprop SEP 35), packed by solprop's
    adapter as (Im sigma, Re f_0, Im f_0, ..., Re f_{n-1}, Im f_{n-1}) over the `rows`
    history rows a formula reads; empty for none. */
void carrier_frame(ExponentialWeights& weights, const std::vector<double>& values,
                   std::size_t rows);

/** The resident engine: the general state's startup, exponential trial and accept,
    with the stage tables of phase 1 (solprop's `_ResidentAdams`). */
class ResidentEngine final : public march::Engine {
public:
  ResidentEngine(GeneralState& state, const stage::Tables& tables,
                 stage::Anchor& anchor, bool fused, bool compensated, bool interpolate,
                 std::optional<MarchFault>& fault);

  void start(double z) override;
  void trial(double z, double dz, const march::Stencil& stencil,
             double* norms) override;
  void accept(double dz, std::size_t free, const march::Stencil& stencil,
              const std::vector<march::Entry>& kept) override;
  double chord_phase(double z1, double z2) override;
  std::vector<double> step_carrier(std::size_t order) override;
  std::vector<double> reading() override;

private:
  void inject(march::Stage stage);

  GeneralState& state_;
  const stage::Tables& tables_;
  stage::Anchor& anchor_;
  bool fused_;
  bool compensated_;
  bool interpolate_;
  std::optional<MarchFault>& fault_;
  /** The last trial's carrier rotations, `_trial_carrier`. */
  std::vector<double> carrier_;
};

/** The resident Runge-Kutta engine (solprop SEP 45 phase 3): a staged trial, its norm
    by the native weak scale, and the accept's recycled stage and finish. */
class ResidentRungeKutta final : public rk::Engine {
public:
  using Trial = std::function<std::vector<double>(double, double)>;
  ResidentRungeKutta(GeneralState& state, Trial trial, double rtol, double atol,
                     const std::vector<double>& weights);
  double trial(double z, double dz) override;
  /** solprop SEP 45 phase 4: a weighted controller's norm, `_score_trial`'s weighted
      path with its sums in order. */
  static double weighted_norm(GeneralState& state, const std::vector<double>& partials,
                              double rtol, double atol,
                              const std::vector<double>& weights);
  void accept(double dz) override;
  std::vector<double> reading() override;
  /** Hold the accepted trial as a dense extension (a native session's dense output,
      `engines::RungeKuttaDense::hold`); refused without one. */
  void extend(double z, double dz) override;
  using Extension = std::function<void(double, double)>;
  void set_extension(Extension extension) { extension_ = std::move(extension); }

private:
  GeneralState& state_;
  Trial trial_;
  Extension extension_;
  double rtol_;
  double atol_;
  const std::vector<double>& weights_;
};

/** The resident side of a native session (solprop SEP 47): the general state's field
    and statistics, which the session reads where the state lives. */
class GeneralResident final : public session::Resident {
public:
  GeneralResident(GeneralState& state, std::size_t elements, std::size_t n_omega);
  [[nodiscard]] std::size_t elements() const override { return elements_; }
  [[nodiscard]] std::size_t n_omega() const override { return n_omega_; }
  const double* host_field() override;
  void field_to_host(double* out) override;
  void count_to_host(std::size_t bytes) override;
  std::vector<double> statistics(const session::StatisticsRecipe& recipe,
                                 double z) override;
  std::vector<double> boundary_reading() override;
  void set_collar_quiet(bool quiet) override;
  double quadrature_peak() override;
  transverse::Reading edge_reading(bool faces, bool whole, bool totals, bool strip,
                                   bool current) override;

private:
  GeneralState& state_;
  std::size_t elements_;
  std::size_t n_omega_;
};

/** One entry of a scripted Adams march's log: a startup, or a trial and the row its
    accept freed. */
struct ScriptedEvent {
  bool start = false;
  double z = 0.0;
  double dz = 0.0;
  std::vector<double> nodes;
  std::vector<std::size_t> slots;
  std::size_t order = 0;
  std::vector<std::size_t> orders;
  std::optional<std::size_t> free;
};

/** A script of norms in place of the field arithmetic, logging what the march asked:
    the decision-trace tests run solprop's corpus through it. */
class ScriptedEngine final : public march::Engine {
public:
  ScriptedEngine(std::vector<double> norms, double carrier_rate,
                 std::vector<ScriptedEvent>& log);

  void start(double z) override;
  void trial(double z, double dz, const march::Stencil& stencil,
             double* norms) override;
  void accept(double dz, std::size_t free, const march::Stencil& stencil,
              const std::vector<march::Entry>& kept) override;
  double chord_phase(double z1, double z2) override;
  [[nodiscard]] std::size_t used() const { return next_; }

private:
  std::vector<double> norms_;
  double carrier_rate_;
  std::vector<ScriptedEvent>& log_;
  std::size_t next_ = 0;
};

/** One trial a scripted Runge-Kutta march made: its start and its step. */
struct ScriptedTrial {
  double z = 0.0;
  double dz = 0.0;
};

/** The scripted Runge-Kutta engine (SEP 45 phase 3): each trial returns its script's
    norm, or with `h0` the script's noise times the model norm `(dz / h0) ** power`, as
    solprop's harness forms it. */
class ScriptedRungeKutta final : public rk::Engine {
public:
  ScriptedRungeKutta(const std::vector<double>& norms, std::size_t& next, double h0,
                     double power, std::vector<ScriptedTrial>& log);
  double trial(double z, double dz) override;
  void accept(double dz) override;
  std::vector<double> reading() override;

private:
  const std::vector<double>& norms_;
  std::size_t& next_;
  double h0_;
  double power_;
  std::vector<ScriptedTrial>& log_;
};

} // namespace solprop_native::engines
