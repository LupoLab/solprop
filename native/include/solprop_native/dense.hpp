// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/adams_march.hpp"

/*
 * Dense output in the native execution session (solprop SEP 47 section 5.3): a save
 * plane interpolated inside the accepted step that crossed it (`interpolate_planes`,
 * SEP 14 phase 3 and ADR-036), formed where the state lives. The session keeps the
 * committed propagation state and the dense observation's position apart: it asks
 * for an observation once the march has crossed the plane, exports and measures the
 * interpolated field, and restores the committed state before the march goes on. The
 * observation neither rewinds the controller nor advances the history.
 *
 * The implementations (`engines::AdamsDense`, `engines::RungeKuttaDense`,
 * dense_output.hpp) are solprop's own observations transcribed: the Adams march's
 * interpolant over the last accepted step (`_ResidentAdams.observe`), and DOPRI5's
 * continuous extension of Hairer, Norsett and Wanner II.6 carried to the plane by the
 * linear flow (`StepDriver._observe_inside`).
 *
 * Python-free and Kokkos-free: host code only.
 */
namespace solprop_native::session {

/** A march's dense output: what a session interpolates a crossed plane from. */
class Dense {
public:
  Dense() = default;
  Dense(const Dense&) = delete;
  Dense& operator=(const Dense&) = delete;
  Dense(Dense&&) = delete;
  Dense& operator=(Dense&&) = delete;
  virtual ~Dense() = default;
  /** Note a march call's return: an Adams call that accepted a step leaves the last
      one's anchor, step, order, nodes and carrier rotations (`march_stepped`). */
  virtual void stepped(const march::Result& result) = 0;
  /** Whether a march that returned at the committed `position` stepped across
      `target`, which is then observed inside its last step: past it for the Adams
      march (`backend._z > target`), past its landing tolerance of `ulps` for the
      Runge-Kutta march (`not reached(target, z)`). */
  [[nodiscard]] virtual bool crosses(double position, double target,
                                     int ulps) const = 0;
  /** Hold the committed state aside and make the state the last accepted step's
      interpolated field at `position`, inside that step. */
  virtual void observe(double position) = 0;
  /** Make the held committed state current again. */
  virtual void restore() = 0;
};

} // namespace solprop_native::session
