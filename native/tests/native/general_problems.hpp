// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The small problems the general-state and CPU-execution tests build: a one-node,
// one-stage problem of each field family on four time samples (`base_problem`), and
// the same with the Kerr response on and one-element transform matrices
// (`general_problem`).

#include "execution.hpp"
#include "solprop_native/problem/problem.hpp"
#include "solprop_native/runtime/execution.hpp"

#include <cstddef>

namespace solprop_native::test {

inline Problem base_problem(const FieldFamily family) {
  Problem problem;
  problem.cpu_execution = test::table();
  problem.dims.n_t = 4;
  problem.dims.n_to = family == FieldFamily::envelope ? 4 : 8;
  problem.dims.n_omega = family == FieldFamily::envelope ? 4 : 3;
  problem.stepper.n_stages = 1;
  problem.linear.n_phi_terms = 1;
  problem.linear.n_prefactor_terms = 1;
  problem.dims.family = family;
  problem.grid.crop_index = {0, 1, 2};
  if (family == FieldFamily::envelope)
    problem.grid.crop_index.push_back(3);
  problem.grid.active_mask.assign(problem.dims.n_omega, 1U);
  problem.grid.norm_mask.assign(problem.dims.n_omega, 1U);
  problem.grid.omega_window.assign(problem.dims.n_omega, 1.0);
  for (std::size_t index = 0; index < problem.dims.n_omega; ++index)
    problem.grid.omega.push_back(static_cast<double>(index + 1));
  problem.grid.dt = 0.25;
  problem.grid.time_window.assign(problem.dims.n_to, 1.0);
  problem.grid.time_window_derivative.assign(problem.dims.n_to, 0.0);
  problem.grid.carrier.assign(problem.dims.n_to, Complex(1.0, 0.0));
  problem.linear.phi_terms.assign(problem.dims.n_omega, Complex{});
  problem.linear.prefactor_terms.assign(problem.dims.n_omega, Complex(1.0, 0.0));
  problem.stepper.tableau_matrix = {0.0};
  problem.stepper.tableau_nodes = {0.0};
  problem.stepper.solution_weights = {1.0};
  problem.stepper.error_weights = {0.25};
  problem.transform.synthesis = 2.0;
  problem.transform.projection = 0.5;
  problem.kerr.coefficient = 0.125;
  problem.kerr.self_factor = 0.5;
  problem.kerr.conjugate_factor = 0.25;
  problem.kerr.harmonic_factor = 0.0;
  problem.kerr.take_real = family != FieldFamily::envelope;
  return problem;
}

inline Problem general_problem(const FieldFamily family) {
  auto problem = base_problem(family);
  problem.kerr.nonlinear = true;
  problem.transform.synthesis_matrix = {problem.transform.synthesis};
  problem.transform.projection_matrix = {problem.transform.projection};
  return problem;
}

} // namespace solprop_native::test
