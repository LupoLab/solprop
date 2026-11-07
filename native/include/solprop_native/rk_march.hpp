// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

#include "solprop_native/adams_march.hpp"

/*
 * The Runge-Kutta march (solprop SEP 45): a trial's stage positions and its scaled
 * error norm (phase 2), and the adaptive march itself, its controller and its loop
 * (phase 3), each performing the operations of the solprop function it is named after
 * in the same order. This translation unit is compiled without contraction and without
 * fast-math on every build; `pow` is the C library's, as Python's float power is.
 */
namespace solprop_native::rk {

/** A trial's stage positions `z + c_i dz`, each product and sum rounded on its own as
    `[z + float(node) * dz for node in nodes]` is, then the step's end `z + dz`. */
[[nodiscard]] std::vector<double> positions(double z, double dz,
                                            const std::vector<double>& nodes);

/**
 * A trial's weak scaled error norm from its partials, `(3, slots)`: the sums of
 * `|y|^2`, `|y_new|^2` and `|e|^2` over the norm's bins, slot by slot. Each row is
 * summed in order, then `sqrt(e) / (rtol max(sqrt(y), sqrt(y_new)) + atol)`, Python's
 * `max` and the scale's operations in `weak_norm_scale`'s order (solprop SEP 45 section
 * 4.4: only the order of the sums differs from the host's NumPy reduction).
 */
[[nodiscard]] double score(const double* partials, std::size_t slots, double rtol,
                           double atol);

/** The strictness test's witness, as this translation unit rounds `a b + c`. */
[[nodiscard]] double strict_probe(double a, double b, double c);

/**
 * Every number the Runge-Kutta march's controller reads (solprop's `RungeKuttaPolicy`
 * with the controller's fields, SEP 45 section 4.2). The landings -- breakpoints,
 * kinks, planes, stretch, cap, landing tolerance and the quiet collar's interval --
 * are the Adams march's own record's fields, read by the same `march::landing`.
 */
struct Policy {
  march::Policy landings;
  double safety = 0.8;
  double beta1 = 0.0;
  double beta2 = 0.0;
  double min_factor = 0.2;
  double max_factor = 1.1;
  std::size_t lag = 1;
  bool reject_free = false;
  /** `estimator_order + 1`, the power of the equivalent norm. */
  double power = 5.0;
  /** `1.0 / (estimator_order + 1)`, as solprop forms it. */
  double exponent = 0.2;
  double start_growth = 100.0;
  bool auto_init = true;
  double dz_min = 0.0;
  double dz_max = 0.0;
};

/** The march's state between calls (SEP 45 section 4.2): a continuation, not a resume.
 */
struct State {
  double z = 0.0;
  double proposal = 0.0;
  /** The accepted steps' equivalent norms, oldest first, at most `lag + 2`. */
  std::deque<double> norms;
  watch::State watch;
};

/** The field arithmetic a Runge-Kutta march hands its trials to. */
class Engine {
public:
  Engine() = default;
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) = delete;
  Engine& operator=(Engine&&) = delete;
  virtual ~Engine() = default;
  /** Form a trial from `z` over `dz`; return its scaled error norm. */
  virtual double trial(double z, double dz) = 0;
  /** Adopt the trial: promote its recycled stage, then finish the state. */
  virtual void accept(double dz) = 0;
  /** The committed state's boundary reading, for a check the march takes itself. */
  virtual std::vector<double> reading() = 0;
};

/** Where the floor stopped a march (SEP 45 section 4.3). */
enum class FloorSite : int { none = 0, rejection = 1, start_up = 2, proposal = 3 };

/** What a Runge-Kutta call returns besides the Adams march's own result fields. */
struct Result {
  march::Result march;
  FloorSite floor_site = FloorSite::none;
};

// The controller, solprop.stepper.controller's functions and limit_step.
[[nodiscard]] double equivalent(const Policy& policy, double norm, double dz_taken,
                                double dz_proposed);
[[nodiscard]] bool accepts(const Policy& policy, double norm);
[[nodiscard]] double propose(const Policy& policy, const std::deque<double>& norms);
[[nodiscard]] double rejection_factor(const Policy& policy, double norm);
[[nodiscard]] double start_factor(const Policy& policy, double norm);
/** `limit_step`'s proposal, `min(dz factor, dz_max)`; the caller tests the floor. */
[[nodiscard]] double limited(const Policy& policy, double dz, double factor);

/** Advance `state` by the algorithm of SEP 45 section 4.3 to an event (adaptive). */
Result run(const Policy& policy, State& state, Engine& engine,
           const march::Bounds& bounds);

} // namespace solprop_native::rk
