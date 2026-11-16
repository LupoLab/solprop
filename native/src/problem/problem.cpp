// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// `Problem::validate` (problem.hpp): the presence of the CPU execution record and the
// shapes the field tables cannot check within one group -- the fixed terms', the Raman
// response's, the plasma rates' and the envelope rules' arrays against their counts --
// and `compact_problem`.

#include "solprop_native/problem/problem.hpp"

#include "solprop_native/runtime/cpu_execution.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace solprop_native {

void Problem::validate() const {
  const auto require = [](const bool holds, const char* message) {
    if (!holds)
      throw std::invalid_argument(message);
  };
  require(cpu_execution != CpuExecution{},
          "a native problem needs its CPU execution record (cpu_execution, solprop "
          "SEP 46)");
  // A bound array's recorded shape is the layout its counts give; an array set in C++
  // records none.
  const auto shaped = [](const Shapes& shapes, const char* field,
                         const std::vector<std::size_t>& expected) {
    const auto found = shapes.find(field);
    return found == shapes.end() || found->second == expected;
  };
  for (const auto& [field, count, rows] :
       {std::tuple{"phi_terms", linear.n_phi_terms,
                   "phi_terms must hold n_phi_terms rows along its first axis"},
        std::tuple{"prefactor_terms", linear.n_prefactor_terms,
                   "prefactor_terms must hold n_prefactor_terms rows along its first "
                   "axis"}})
    if (const auto found = linear.shapes.find(field); found != linear.shapes.end()) {
      const auto& shape = found->second;
      require(shape.size() == 2 || shape.size() == 3,
              "fixed terms must be rank-two or rank-three arrays");
      require(shape[0] == count, rows);
    }
  if (raman.enabled) {
    require(raman.n_terms != 0 &&
                dims.n_to <=
                    std::numeric_limits<std::size_t>::max() / (2 * raman.n_terms) &&
                raman.undamped.size() == raman.n_terms * 2 * dims.n_to &&
                shaped(raman.shapes, "undamped", {raman.n_terms, 2 * dims.n_to}),
            "raman_undamped must have shape (n_terms, 2 * n_to)");
    require(raman.tensor_weights.size() == 2 * raman.n_terms &&
                shaped(raman.shapes, "tensor_weights", {raman.n_terms, 2}),
            "raman_tensor_weights must have shape (n_terms, 2)");
  } else {
    require(raman.n_terms == 0,
            "raman.n_terms counts the terms of an enabled response");
  }
  require(plasma.rate_coefficients.size() == 4 * plasma.rate_intervals &&
              shaped(plasma.shapes, "rate_coefficients", {4, plasma.rate_intervals}),
          "rate_coefficients must have shape (4, n_intervals)");
  if (plasma.envelope) {
    constexpr std::size_t rule = std::size_t{4} * 512;
    const std::vector<std::size_t> layout{plasma.n_channels, 4, 512};
    require(plasma.envelope_rule_nodes.size() == plasma.n_channels * rule &&
                plasma.envelope_rule_weights.size() == plasma.n_channels * rule &&
                shaped(plasma.shapes, "envelope_rule_nodes", layout) &&
                shaped(plasma.shapes, "envelope_rule_weights", layout),
            "envelope rules require (n_channels, 4, 512)");
    require(plasma.envelope_drude_factor.size() == dims.n_omega &&
                plasma.envelope_loss_factor.size() == dims.n_omega,
            "envelope current factors require (n_omega,)");
  }
}

bool compact_problem(const Problem& problem) noexcept {
  return problem.n_columns() == 1 && problem.dims.n_nodes == 1 &&
         problem.dims.n_pol_transform == 1 && !problem.general.enabled &&
         !problem.flow.enabled && !problem.separable_geometry() &&
         problem.dims.n_to < problem.cpu_execution.compact_max_fine_samples;
}

} // namespace solprop_native
