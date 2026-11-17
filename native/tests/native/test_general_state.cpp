// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Tests of the general state (`general/general_state.hpp`): each field family's whole
// step, Raman, plasma and held-current responses against straightforward host
// computations, its stashes, statistics and memory ledger, transform providers, hybrid
// planar stages, absorbers and low-storage passes, and its refusals before allocation.

#include "execution.hpp"
#include "general_problems.hpp"
#include "solprop_native/general/general_state.hpp"
#include "solprop_native/problem/problem.hpp"
#include "solprop_native/runtime/cpu_execution.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/transforms/gemm.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace native = solprop_native;
using native::Complex;
using native::test::base_problem;
using native::test::general_problem;

constexpr double kEpsilon = std::numeric_limits<double>::epsilon();

Complex cis(const double angle) { return {std::cos(angle), std::sin(angle)}; }

std::vector<Complex> dft(const std::vector<Complex>& values) {
  std::vector<Complex> result(values.size());
  for (std::size_t frequency = 0; frequency < values.size(); ++frequency) {
    for (std::size_t sample = 0; sample < values.size(); ++sample) {
      const double angle = -2.0 * std::numbers::pi_v<double> *
                           static_cast<double>(frequency * sample) /
                           static_cast<double>(values.size());
      result[frequency] += values[sample] * cis(angle);
    }
  }
  return result;
}

std::vector<Complex> envelope_time(const std::vector<Complex>& spectrum,
                                   const std::size_t n_time) {
  std::vector<Complex> result(n_time);
  for (std::size_t sample = 0; sample < n_time; ++sample) {
    for (std::size_t frequency = 0; frequency < spectrum.size(); ++frequency) {
      const double angle = 2.0 * std::numbers::pi_v<double> *
                           static_cast<double>(frequency * sample) /
                           static_cast<double>(n_time);
      result[sample] += spectrum[frequency] * cis(angle);
    }
    result[sample] /= static_cast<double>(n_time);
  }
  return result;
}

std::vector<double> real_time(const std::vector<Complex>& spectrum,
                              const std::size_t n_time) {
  std::vector<Complex> full(n_time);
  for (std::size_t frequency = 0; frequency < spectrum.size(); ++frequency) {
    full[frequency] = spectrum[frequency];
    if (frequency != 0 && frequency != n_time / 2) {
      full[n_time - frequency] = Kokkos::conj(spectrum[frequency]);
    }
  }
  const auto transformed = envelope_time(full, n_time);
  std::vector<double> result(n_time);
  for (std::size_t sample = 0; sample < n_time; ++sample) {
    result[sample] = transformed[sample].real();
  }
  return result;
}

native::TrialScalars trial(const native::Problem& problem, const double taper = 1.0) {
  return {
      .phi_delta = std::vector<double>(
          (problem.stepper.n_stages + 1) * problem.linear.n_phi_terms, 0.0),
      .phi_at = std::vector<double>(
          problem.stepper.n_stages * problem.linear.n_phi_terms, 0.0),
      .prefactor = std::vector<double>(
          problem.stepper.n_stages * problem.linear.n_prefactor_terms, 1.0),
      .density = std::vector<double>(problem.stepper.n_stages, 1.0),
      .taper = std::vector<double>(problem.stepper.n_stages, taper),
      .preionisation = {},
      .raman_rates =
          std::vector<double>(problem.stepper.n_stages * problem.raman.n_terms, 0.0),
      .kerr_density = {},
      .plasma_densities = {},
      .plasma_preionisation = {},
      .raman_densities = {},
      .flow_section = 0,
      .flow_weights = {},
      .transform_weights = {},
      .dispersive_densities = {},
  };
}

native::Problem multimode_problem(const std::size_t node_chunk = 2) {
  auto problem = base_problem(native::FieldFamily::field_real);
  problem.dims.n_modes = 2;
  problem.dims.n_pol_state = 1;
  problem.dims.n_pol_transform = 1;
  problem.dims.n_nodes = 2;
  problem.dims.node_chunk = node_chunk;
  problem.kerr.nonlinear = true;
  problem.transform.synthesis_matrix = {1.0, 0.5, -0.25, 1.5};
  problem.transform.projection_matrix = {0.75, -0.5, 0.25, 1.25};
  problem.transform.projection_coarse_matrix = {0.5, -0.25, 0.125, 1.0};
  problem.linear.phi_terms.assign(problem.dims.n_modes * problem.dims.n_omega,
                                  Complex{});
  problem.linear.prefactor_terms.assign(problem.dims.n_modes * problem.dims.n_omega,
                                        Complex(1.0, 0.0));
  return problem;
}

native::Problem plasma_problem() {
  auto problem = general_problem(native::FieldFamily::field_real);
  problem.kerr.nonlinear = false;
  problem.plasma.enabled = true;
  problem.plasma.stash = true;
  problem.grid.dt_fine = 0.125;
  problem.plasma.charge_mass_ratio = 2.0;
  problem.plasma.ionisation_potential = 3.0;
  problem.plasma.log_field_min = std::log(0.01);
  problem.plasma.d_log_field = std::log(1000.0);
  problem.plasma.field_min = 0.01;
  problem.plasma.field_max = 10.0;
  problem.plasma.rate_max = 0.5;
  problem.plasma.rate_intervals = 1;
  problem.plasma.rate_coefficients = {0.0, 0.0, 0.0, std::log(0.5)};
  return problem;
}

native::Problem raman_problem(const native::FieldFamily family) {
  auto problem = general_problem(family);
  problem.kerr.nonlinear = false;
  problem.raman.enabled = true;
  problem.raman.thg = family == native::FieldFamily::field_real;
  problem.raman.n_terms = 2;
  problem.raman.tensor_weights = {1.0, 0.0, 1.0, 0.0};
  problem.raman.fraction = 0.4;
  problem.raman.epsilon0 = 2.0;
  problem.grid.dt_fine = 0.125;
  problem.raman.undamped.assign(problem.raman.n_terms * 2 * problem.dims.n_to, 0.0);
  problem.raman.undamped[1] = 1.0;
  problem.raman.undamped[2] = 0.5;
  problem.raman.undamped[3] = -0.25;
  const auto second = 2 * problem.dims.n_to;
  problem.raman.undamped[second + 1] = 0.2;
  problem.raman.undamped[second + 2] = 0.1;
  problem.raman.undamped[second + 3] = 0.05;
  return problem;
}

std::vector<Complex> raman_reference(const native::Problem& problem,
                                     const std::vector<Complex>& spectrum,
                                     const std::vector<double>& rates) {
  std::vector<Complex> modal;
  if (problem.dims.family == native::FieldFamily::field_real) {
    for (const double value : real_time(spectrum, problem.dims.n_to))
      modal.emplace_back(value, 0.0);
  } else if (problem.dims.family == native::FieldFamily::field_analytic) {
    auto analytic = std::vector<Complex>(problem.dims.n_to);
    analytic[0] = spectrum[0];
    for (std::size_t index = 1; index < spectrum.size(); ++index)
      analytic[index] = 2.0 * spectrum[index];
    modal = envelope_time(analytic, problem.dims.n_to);
  } else {
    modal = envelope_time(spectrum, problem.dims.n_to);
  }

  const auto doubled = 2 * problem.dims.n_to;
  std::vector<double> kernel(doubled);
  for (std::size_t time = 0; time < doubled; ++time) {
    double total = 0.0;
    const double delay = static_cast<double>(time) * problem.grid.dt_fine;
    for (std::size_t term = 0; term < problem.raman.n_terms; ++term)
      total += problem.raman.undamped[term * doubled + time] *
               std::exp(-rates[term] * delay);
    kernel[time] = problem.raman.fraction * total;
  }

  std::vector<Complex> projected(problem.dims.n_to);
  for (std::size_t node = 0; node < problem.dims.n_nodes; ++node) {
    const double synthesis = problem.transform.synthesis_matrix[node];
    const double projection = problem.transform.projection_matrix[node];
    std::vector<Complex> field(problem.dims.n_to);
    std::vector<double> drive(problem.dims.n_to);
    for (std::size_t time = 0; time < problem.dims.n_to; ++time) {
      field[time] = synthesis * modal[time];
      drive[time] = problem.raman.thg ? field[time].real() * field[time].real()
                                      : 0.5 * (field[time].real() * field[time].real() +
                                               field[time].imag() * field[time].imag());
    }
    for (std::size_t time = 0; time < problem.dims.n_to; ++time) {
      double convolved = 0.0;
      for (std::size_t previous = 0; previous <= time; ++previous)
        convolved += drive[previous] * kernel[time - previous];
      convolved *= problem.grid.dt_fine;
      const Complex multiplier = problem.dims.family == native::FieldFamily::envelope
                                     ? field[time]
                                     : Complex(field[time].real(), 0.0);
      projected[time] += projection * problem.raman.epsilon0 * multiplier * convolved *
                         problem.grid.time_window[time];
    }
  }
  const auto transformed = dft(projected);
  return {transformed.begin(),
          transformed.begin() + static_cast<std::ptrdiff_t>(problem.dims.n_omega)};
}

std::vector<Complex> multimode_reference(const native::Problem& problem,
                                         const std::vector<Complex>& spectrum,
                                         const bool coarse = false) {
  std::vector<std::vector<double>> modal(problem.dims.n_modes);
  for (std::size_t mode = 0; mode < problem.dims.n_modes; ++mode) {
    const auto first =
        spectrum.begin() + static_cast<std::ptrdiff_t>(mode * problem.dims.n_omega);
    const std::vector<Complex> one(
        first, first + static_cast<std::ptrdiff_t>(problem.dims.n_omega));
    modal[mode] = real_time(one, problem.dims.n_to);
  }
  std::vector<std::vector<double>> polarisation(problem.dims.n_nodes,
                                                std::vector<double>(problem.dims.n_to));
  for (std::size_t node = 0; node < problem.dims.n_nodes; ++node) {
    for (std::size_t time = 0; time < problem.dims.n_to; ++time) {
      double value = 0.0;
      for (std::size_t mode = 0; mode < problem.dims.n_modes; ++mode)
        value +=
            problem.transform.synthesis_matrix[node * problem.dims.n_modes + mode] *
            modal[mode][time];
      polarisation[node][time] = problem.kerr.coefficient * value * value * value;
    }
  }
  std::vector<Complex> result(problem.dims.n_modes * problem.dims.n_omega);
  const auto& projection = coarse ? problem.transform.projection_coarse_matrix
                                  : problem.transform.projection_matrix;
  for (std::size_t mode = 0; mode < problem.dims.n_modes; ++mode) {
    std::vector<Complex> projected(problem.dims.n_to);
    for (std::size_t time = 0; time < problem.dims.n_to; ++time) {
      for (std::size_t node = 0; node < problem.dims.n_nodes; ++node)
        projected[time] +=
            projection[mode * problem.dims.n_nodes + node] * polarisation[node][time];
    }
    const auto transformed = dft(projected);
    for (std::size_t index = 0; index < problem.dims.n_omega; ++index)
      result[mode * problem.dims.n_omega + index] = transformed[index];
  }
  return result;
}

std::vector<Complex> nonlinear_reference(const native::Problem& problem,
                                         const std::vector<Complex>& spectrum,
                                         const double taper) {
  std::vector<Complex> time;
  if (problem.dims.family == native::FieldFamily::envelope) {
    time = envelope_time(spectrum, problem.dims.n_to);
  } else if (problem.dims.family == native::FieldFamily::field_real) {
    const auto real = real_time(spectrum, problem.dims.n_to);
    for (const auto value : real)
      time.emplace_back(value, 0.0);
  } else {
    std::vector<Complex> analytic(problem.dims.n_to);
    analytic[0] = spectrum[0];
    for (std::size_t index = 1; index < spectrum.size(); ++index) {
      analytic[index] = 2.0 * spectrum[index];
    }
    time = envelope_time(analytic, problem.dims.n_to);
  }
  std::vector<Complex> projected(problem.dims.n_to);
  for (std::size_t index = 0; index < problem.dims.n_to; ++index) {
    const Complex node = problem.transform.synthesis * taper * time[index];
    Complex polarisation;
    if (problem.dims.family == native::FieldFamily::field_real) {
      polarisation = problem.kerr.coefficient * node.real() * node.real() *
                     Complex(node.real(), 0.0);
    } else {
      const double modulus = node.real() * node.real() + node.imag() * node.imag();
      polarisation = problem.kerr.coefficient *
                     (problem.kerr.self_factor + problem.kerr.conjugate_factor) *
                     modulus * node;
    }
    polarisation *= problem.transform.projection / taper;
    projected[index] =
        problem.kerr.take_real ? Complex(polarisation.real(), 0.0) : polarisation;
  }
  const auto full = dft(projected);
  return {full.begin(),
          full.begin() + static_cast<std::ptrdiff_t>(problem.dims.n_omega)};
}

void expect_complex_near(const Complex actual, const Complex expected,
                         const double tolerance) {
  EXPECT_NEAR(actual.real(), expected.real(), tolerance);
  EXPECT_NEAR(actual.imag(), expected.imag(), tolerance);
}

TEST(GeneralState, MasksAndAppliesCombinedWholeStepPropagator) {
  auto problem = general_problem(native::FieldFamily::envelope);
  problem.linear.n_phi_terms = 2;
  problem.linear.phi_terms.clear();
  for (std::size_t term = 0; term < 2; ++term) {
    for (std::size_t index = 0; index < problem.dims.n_omega; ++index) {
      problem.linear.phi_terms.emplace_back(0.0, static_cast<double>(term + 1) *
                                                     static_cast<double>(index + 1));
    }
  }
  problem.kerr.nonlinear = false;
  problem.grid.active_mask[2] = 0U;
  auto scalars = trial(problem);
  scalars.phi_delta = {0.1, 0.2, 0.3, 0.4};
  const std::vector<Complex> initial(problem.dims.n_omega, {1.0, -0.5});
  native::GeneralState state(problem);
  state.set_initial(initial, {});
  const auto partials = state.trial_step(0.25, scalars);
  state.finish(0.25);
  const auto result = state.field_to_host();
  for (std::size_t index = 0; index < result.size(); ++index) {
    const Complex masked = index == 2 ? Complex{} : initial[index];
    const double angle = 0.3 * static_cast<double>(index + 1) +
                         0.4 * 2.0 * static_cast<double>(index + 1);
    expect_complex_near(result[index], cis(angle) * masked, 64.0 * kEpsilon);
  }
  // Exact: a linear problem has no stage derivative, so its embedded error sums
  // exact zeros.
  EXPECT_EQ(partials[2], 0.0);
}

TEST(GeneralState, KerrFamiliesMatchStraightforwardHostTransforms) {
  for (const auto family :
       {native::FieldFamily::field_real, native::FieldFamily::field_analytic,
        native::FieldFamily::envelope}) {
    SCOPED_TRACE(static_cast<int>(family));
    auto problem = general_problem(family);
    problem.kerr.nonlinear = true;
    std::vector<Complex> initial(problem.dims.n_omega);
    initial[0] = Complex(1.0, 0.0);
    initial[1] = Complex(0.25, -0.125);
    initial[2] = Complex(-0.1, 0.05);
    if (problem.dims.n_omega == 4)
      initial[3] = Complex(0.05, 0.025);
    constexpr double taper = 1.5;
    constexpr double dz = 0.02;
    const auto derivative = nonlinear_reference(problem, initial, taper);
    native::GeneralState state(problem);
    state.set_initial(initial, {});
    const auto partials = state.trial_step(dz, trial(problem, taper));
    state.finish(dz);
    const auto result = state.field_to_host();
    double magnitude = 1.0;
    for (std::size_t index = 0; index < result.size(); ++index) {
      const Complex expected = initial[index] + dz * derivative[index];
      magnitude = std::max(magnitude, Kokkos::abs(expected));
      const double tolerance =
          512.0 * kEpsilon * static_cast<double>(problem.dims.n_to) * magnitude;
      expect_complex_near(result[index], expected, tolerance);
    }
    EXPECT_GT(partials[2], 0.0);
  }
}

TEST(GeneralState, MultimodeTransformMatchesStraightforwardHostImplementation) {
  const auto problem = multimode_problem();
  const std::vector<Complex> initial{{1.0, 0.0}, {0.25, -0.125}, {-0.1, 0.05},
                                     {0.5, 0.0}, {-0.2, 0.1},    {0.05, -0.025}};
  const auto derivative = multimode_reference(problem, initial);
  constexpr double dz = 0.01;
  native::GeneralState state(problem);
  state.set_initial(initial, {});
  EXPECT_THROW(static_cast<void>(state.stage_to_host(0)), std::logic_error);
  const auto partials = state.trial_step(dz, trial(problem));
  const auto stage = state.stage_to_host(0);
  EXPECT_THROW(static_cast<void>(state.stage_to_host(problem.stepper.n_stages)),
               std::out_of_range);
  ASSERT_EQ(stage.size(), derivative.size());
  for (std::size_t index = 0; index < stage.size(); ++index) {
    const double tolerance =
        256.0 * kEpsilon * std::max(1.0, Kokkos::abs(derivative[index]));
    expect_complex_near(stage[index], derivative[index], tolerance);
  }
  state.finish(dz);
  const auto result = state.field_to_host();
  ASSERT_EQ(partials.size(), 3 * problem.dims.n_modes);
  ASSERT_EQ(result.size(), initial.size());
  for (std::size_t index = 0; index < result.size(); ++index) {
    const Complex expected = initial[index] + dz * derivative[index];
    const double tolerance = 256.0 * kEpsilon * std::max(1.0, Kokkos::abs(expected));
    expect_complex_near(result[index], expected, tolerance);
  }
}

TEST(GeneralState, NodeChunkingPreservesTheMultimodeResult) {
  auto whole_problem = multimode_problem(2);
  auto chunked_problem = multimode_problem(1);
  const std::vector<Complex> initial{{1.0, 0.0}, {0.25, -0.125}, {-0.1, 0.05},
                                     {0.5, 0.0}, {-0.2, 0.1},    {0.05, -0.025}};
  native::GeneralState whole(whole_problem);
  native::GeneralState chunked(chunked_problem);
  whole.set_initial(initial, {});
  chunked.set_initial(initial, {});
  const auto whole_partials = whole.trial_step(0.01, trial(whole_problem));
  const auto chunked_partials = chunked.trial_step(0.01, trial(chunked_problem));
  EXPECT_EQ(whole_partials.size(), chunked_partials.size());
  whole.finish(0.01);
  chunked.finish(0.01);
  const auto expected = whole.field_to_host();
  const auto actual = chunked.field_to_host();
  for (std::size_t index = 0; index < actual.size(); ++index)
    expect_complex_near(actual[index], expected[index], 256.0 * kEpsilon);
}

TEST(GeneralState, QuadratureErrorUsesFineAndCoarseProjectionOfTheSameResponse) {
  auto problem = multimode_problem();
  problem.stepper.first_same_as_last = false;
  const std::vector<Complex> initial{{1.0, 0.0}, {0.25, -0.125}, {-0.1, 0.05},
                                     {0.5, 0.0}, {-0.2, 0.1},    {0.05, -0.025}};
  const auto fine = multimode_reference(problem, initial);
  const auto coarse = multimode_reference(problem, initial, true);
  double scale = 0.0;
  double difference = 0.0;
  for (std::size_t index = 0; index < fine.size(); ++index) {
    scale += std::norm(std::complex<double>(fine[index].real(), fine[index].imag()));
    const auto gap = coarse[index] - fine[index];
    difference += std::norm(std::complex<double>(gap.real(), gap.imag()));
  }
  const double expected = std::sqrt(difference / scale);
  native::GeneralState state(problem);
  state.set_initial(initial, {});
  static_cast<void>(state.trial_step(0.01, trial(problem)));
  state.promote_fsal();
  EXPECT_NEAR(state.quadrature_error(), expected,
              512.0 * static_cast<double>(problem.dims.n_to) * kEpsilon);
}

TEST(GeneralState, WeightedResidualDividesEachBinBeforeReducing) {
  auto problem = multimode_problem();
  problem.stepper.weighted_controller = true;
  const std::vector<Complex> initial{{1.0, 0.0}, {0.25, -0.125}, {-0.1, 0.05},
                                     {0.5, 0.0}, {-0.2, 0.1},    {0.05, -0.025}};
  native::GeneralState state(problem);
  state.set_initial(initial, {});
  const auto ordinary = state.trial_step(0.01, trial(problem));
  const auto weighted =
      state.weighted_error(std::vector<double>(problem.dims.n_omega, 2.0));
  ASSERT_EQ(weighted.size(), problem.n_columns());
  for (std::size_t column = 0; column < problem.n_columns(); ++column)
    EXPECT_NEAR(weighted[column], ordinary[2 * problem.n_columns() + column] / 4.0,
                32.0 * kEpsilon);
}

TEST(GeneralState, PackedStatisticsMatchStraightforwardHostReductions) {
  auto problem = multimode_problem();
  problem.dims.family = native::FieldFamily::envelope;
  problem.dims.n_t = 4;
  problem.dims.n_to = 4;
  problem.dims.n_omega = 4;
  problem.grid.crop_index = {0, 1, 2, 3};
  problem.grid.active_mask.assign(4, 1U);
  problem.grid.norm_mask.assign(4, 1U);
  problem.grid.omega_window.assign(4, 1.0);
  problem.grid.omega = {1.0, 2.0, 3.0, 4.0};
  problem.grid.time_window.assign(4, 1.0);
  problem.grid.carrier.assign(4, Complex(1.0, 0.0));
  problem.linear.phi_terms.assign(8, Complex{});
  problem.linear.prefactor_terms.assign(8, Complex(1.0, 0.0));
  const std::vector<Complex> initial{{1.0, 0.0},  {0.5, 0.25}, {0.0, 0.0}, {0.0, 0.0},
                                     {0.25, 0.0}, {0.0, 0.5},  {0.0, 0.0}, {0.0, 0.0}};
  constexpr auto flags =
      static_cast<std::uint64_t>(native::statistics_omega0) |
      static_cast<std::uint64_t>(native::statistics_energy) |
      static_cast<std::uint64_t>(native::statistics_peak_power) |
      static_cast<std::uint64_t>(native::statistics_peak_power_total) |
      static_cast<std::uint64_t>(native::statistics_peak_intensity);
  native::GeneralState state(problem);
  state.set_initial(initial, {});
  const auto values = state.statistics(flags, {}, {1.0, 0.5}, 0.0, 1.0, 0.0, 2.0);
  ASSERT_EQ(values.size(), 8);
  for (std::size_t mode = 0; mode < 2; ++mode) {
    double power = 0.0;
    double weighted = 0.0;
    for (std::size_t index = 0; index < 4; ++index) {
      const double one = std::norm(std::complex<double>(
          initial[mode * 4 + index].real(), initial[mode * 4 + index].imag()));
      power += one;
      weighted += problem.grid.omega[index] * one;
    }
    EXPECT_NEAR(values[mode], weighted / power, 32.0 * kEpsilon);
    EXPECT_NEAR(values[2 + mode], problem.grid.dt / 4.0 * power, 32.0 * kEpsilon);
  }
  std::array<std::vector<Complex>, 2> time{
      envelope_time(std::vector<Complex>(initial.begin(), initial.begin() + 4), 4),
      envelope_time(std::vector<Complex>(initial.begin() + 4, initial.end()), 4)};
  std::array<double, 2> peaks{};
  double total_peak = 0.0;
  double intensity_peak = 0.0;
  for (std::size_t sample = 0; sample < 4; ++sample) {
    double total = 0.0;
    for (std::size_t mode = 0; mode < 2; ++mode) {
      const double one = std::norm(
          std::complex<double>(time[mode][sample].real(), time[mode][sample].imag()));
      peaks[mode] = std::max(peaks[mode], one);
      total += one;
    }
    total_peak = std::max(total_peak, total);
    const auto on_axis = time[0][sample] + 0.5 * time[1][sample];
    intensity_peak =
        std::max(intensity_peak,
                 std::norm(std::complex<double>(on_axis.real(), on_axis.imag())));
  }
  EXPECT_NEAR(values[4], peaks[0], 64.0 * kEpsilon);
  EXPECT_NEAR(values[5], peaks[1], 64.0 * kEpsilon);
  EXPECT_NEAR(values[6], total_peak, 64.0 * kEpsilon);
  EXPECT_NEAR(values[7], 2.0 * intensity_peak, 64.0 * kEpsilon);
}

TEST(GeneralState, EveryFieldFamilyMatchesTheStraightforwardHostResponse) {
  for (const auto family :
       {native::FieldFamily::field_real, native::FieldFamily::field_analytic,
        native::FieldFamily::envelope}) {
    SCOPED_TRACE(static_cast<int>(family));
    auto problem = general_problem(family);
    std::vector<Complex> initial(problem.dims.n_omega);
    initial[0] = Complex(1.0, 0.0);
    initial[1] = Complex(0.25, -0.125);
    initial[2] = Complex(-0.1, 0.05);
    if (problem.dims.n_omega == 4)
      initial[3] = Complex(0.05, 0.025);
    constexpr double taper = 1.5;
    constexpr double dz = 0.02;
    const auto derivative = nonlinear_reference(problem, initial, taper);
    native::GeneralState state(problem);
    state.set_initial(initial, {});
    static_cast<void>(state.trial_step(dz, trial(problem, taper)));
    state.finish(dz);
    const auto result = state.field_to_host();
    for (std::size_t index = 0; index < result.size(); ++index) {
      const Complex expected = initial[index] + dz * derivative[index];
      const double tolerance = 512.0 * kEpsilon *
                               static_cast<double>(problem.dims.n_to) *
                               std::max(1.0, Kokkos::abs(expected));
      expect_complex_near(result[index], expected, tolerance);
    }
  }
}

TEST(GeneralState, RamanMatchesDirectCausalConvolutionForEveryDriveForm) {
  const std::vector<double> rates{0.125, 0.25};
  for (const auto family :
       {native::FieldFamily::field_real, native::FieldFamily::field_analytic,
        native::FieldFamily::envelope}) {
    SCOPED_TRACE(static_cast<int>(family));
    auto problem = raman_problem(family);
    std::vector<Complex> initial(problem.dims.n_omega);
    initial[0] = Complex(1.0, 0.0);
    initial[1] = Complex(0.25, -0.125);
    initial[2] = Complex(-0.1, 0.05);
    if (problem.dims.n_omega == 4)
      initial[3] = Complex(0.05, 0.025);
    const auto derivative = raman_reference(problem, initial, rates);
    auto scalars = trial(problem);
    scalars.raman_rates = rates;
    constexpr double dz = 0.02;
    native::GeneralState state(problem);
    state.set_initial(initial, {});
    static_cast<void>(state.trial_step(dz, scalars));
    state.finish(dz);
    const auto result = state.field_to_host();
    for (std::size_t index = 0; index < result.size(); ++index) {
      const Complex expected = initial[index] + dz * derivative[index];
      const double tolerance = 512.0 * kEpsilon *
                               std::log2(2.0 * static_cast<double>(problem.dims.n_to)) *
                               std::max(1.0, Kokkos::abs(expected));
      expect_complex_near(result[index], expected, tolerance);
    }
  }
}

TEST(GeneralState, RamanMatchesDirectConvolutionForAShortFinalNodeChunk) {
  auto problem = raman_problem(native::FieldFamily::field_real);
  problem.dims.n_nodes = 3;
  problem.dims.node_chunk = 2;
  problem.transform.synthesis_matrix = {1.0, 0.75, -0.5};
  problem.transform.projection_matrix = {0.5, 0.25, -0.75};
  const std::vector<double> rates{0.125, 0.25};
  const std::vector<Complex> initial{{1.0, 0.0}, {0.25, -0.125}, {-0.1, 0.05}};
  const auto derivative = raman_reference(problem, initial, rates);
  auto scalars = trial(problem);
  scalars.raman_rates = rates;
  constexpr double dz = 0.02;
  native::GeneralState state(problem);
  state.set_initial(initial, {});
  static_cast<void>(state.trial_step(dz, scalars));
  state.finish(dz);
  const auto result = state.field_to_host();
  for (std::size_t index = 0; index < result.size(); ++index) {
    const Complex expected = initial[index] + dz * derivative[index];
    const double tolerance = 512.0 * kEpsilon *
                             std::log2(2.0 * static_cast<double>(problem.dims.n_to)) *
                             std::max(1.0, Kokkos::abs(expected));
    expect_complex_near(result[index], expected, tolerance);
  }
}

TEST(GeneralState, RamanKernelMemoizesTheExactOrderedRateTuple) {
  native::GeneralState state(raman_problem(native::FieldFamily::field_real));
  EXPECT_EQ(state.raman_kernel_transforms(), 0U);
  state.prepare_raman({0.125, 0.25});
  EXPECT_EQ(state.raman_kernel_transforms(), 1U);
  state.prepare_raman({0.125, 0.25});
  EXPECT_EQ(state.raman_kernel_transforms(), 1U);
  state.prepare_raman({0.25, 0.125});
  EXPECT_EQ(state.raman_kernel_transforms(), 2U);
  EXPECT_GT(state.raman_planning_seconds(), 0.0);
}

// Independent physical-channel contraction: no effective trace/traceless kernels.
std::vector<Complex> vector_raman_reference(const native::Problem& problem,
                                            const std::vector<Complex>& spectrum,
                                            const std::vector<double>& rates,
                                            const bool isotropic_only) {
  std::array<std::vector<Complex>, 2> modal;
  for (std::size_t pol = 0; pol < 2; ++pol) {
    const auto offset = static_cast<std::ptrdiff_t>(pol * problem.dims.n_omega);
    std::vector<Complex> input(spectrum.begin() + offset,
                               spectrum.begin() + offset +
                                   static_cast<std::ptrdiff_t>(problem.dims.n_omega));
    if (problem.dims.family == native::FieldFamily::field_real) {
      for (const double value : real_time(input, problem.dims.n_to))
        modal[pol].emplace_back(value, 0.0);
    } else {
      if (problem.dims.family == native::FieldFamily::field_analytic) {
        for (std::size_t j = 1; j < input.size(); ++j)
          input[j] *= 2.0;
      }
      modal[pol] = envelope_time(input, problem.dims.n_to);
    }
  }
  std::array<std::vector<Complex>, 2> projected{
      std::vector<Complex>(problem.dims.n_to), std::vector<Complex>(problem.dims.n_to)};
  for (std::size_t node = 0; node < problem.dims.n_nodes; ++node) {
    const double synthesis = problem.transform.synthesis_matrix[node];
    for (std::size_t time = 0; time < problem.dims.n_to; ++time) {
      std::array<Complex, 2> field{synthesis * modal[0][time],
                                   synthesis * modal[1][time]};
      if (problem.dims.family != native::FieldFamily::envelope)
        for (auto& value : field)
          value = Complex(value.real(), 0.0);
      for (std::size_t previous = 0; previous <= time; ++previous) {
        const Complex x = synthesis * modal[0][previous];
        const Complex y = synthesis * modal[1][previous];
        const double half = problem.raman.thg ? 1.0 : 0.5;
        const double xx = half * (x.real() * x.real() + x.imag() * x.imag());
        const double yy = half * (y.real() * y.real() + y.imag() * y.imag());
        const double xy = half * (x.real() * y.real() + x.imag() * y.imag());
        const std::array<std::array<double, 2>, 2> q{{{xx, xy}, {xy, yy}}};
        const auto lag = time - previous;
        const double delay = static_cast<double>(lag) * problem.grid.dt_fine;
        double hv = problem.raman.undamped[2 * problem.dims.n_to + lag] *
                    std::exp(-rates[1] * delay);
        double hr = problem.raman.undamped[lag] * std::exp(-rates[0] * delay);
        if (isotropic_only) {
          hv += hr;
          hr = 0.0;
        }
        for (std::size_t pol = 0; pol < 2; ++pol) {
          Complex value = hv * (xx + yy) * field[pol];
          for (std::size_t other = 0; other < 2; ++other)
            value += 0.5 * hr * (3.0 * q[pol][other] - (pol == other ? xx + yy : 0.0)) *
                     field[other];
          projected[pol][time] += problem.transform.projection_matrix[node] *
                                  problem.raman.epsilon0 * problem.raman.fraction *
                                  problem.grid.dt_fine *
                                  problem.grid.time_window[time] * value;
        }
      }
    }
  }
  std::vector<Complex> derivative;
  for (const auto& component : projected) {
    const auto transformed = dft(component);
    derivative.insert(derivative.end(), transformed.begin(),
                      transformed.begin() +
                          static_cast<std::ptrdiff_t>(problem.dims.n_omega));
  }
  return derivative;
}

TEST(GeneralState, VectorRamanMatchesPhysicalCausalSumsAndRepeatsWithTailPlans) {
  for (const auto family :
       {native::FieldFamily::field_real, native::FieldFamily::field_analytic,
        native::FieldFamily::envelope}) {
    for (const std::size_t count : {8U, 12U, 16U}) {
      for (const bool isotropic_only : {false, true}) {
        auto problem = raman_problem(family);
        const auto old_count = problem.dims.n_to;
        const auto old_rows = problem.raman.undamped;
        problem.dims.n_to = count;
        problem.grid.time_window.assign(count, 1.0);
        problem.grid.carrier.assign(count, Complex(1.0, 0.0));
        problem.raman.undamped.assign(problem.raman.n_terms * 2 * count, 0.0);
        for (std::size_t term = 0; term < problem.raman.n_terms; ++term)
          for (std::size_t sample = 0; sample < old_count; ++sample)
            problem.raman.undamped[term * 2 * count + sample] =
                old_rows[term * 2 * old_count + sample];
        problem.dims.n_pol_state = 2;
        problem.raman.tensor_capable = true;
        problem.raman.tensor_weights = isotropic_only
                                           ? std::vector<double>{1.0, 0.0, 1.0, 0.0}
                                           : std::vector<double>{0.25, 0.75, 1.0, 0.0};
        problem.dims.n_nodes = 3;
        problem.dims.node_chunk = 2;
        problem.transform.synthesis_matrix = {1.0, 0.75, -0.5};
        problem.transform.projection_matrix = {0.5, 0.25, -0.75};
        problem.linear.phi_terms.assign(2 * problem.dims.n_omega, Complex{});
        problem.linear.prefactor_terms.assign(2 * problem.dims.n_omega,
                                              Complex(1.0, 0.0));
        const std::vector<double> rates{0.125, 0.25};
        std::vector<Complex> initial(2 * problem.dims.n_omega);
        for (std::size_t j = 0; j < initial.size(); ++j)
          initial[j] = Complex(std::cos(static_cast<double>(j)),
                               0.25 * std::sin(0.7 * static_cast<double>(j)));
        const auto expected =
            vector_raman_reference(problem, initial, rates, isotropic_only);
        // Three FFT pairs plus length-n causal sums, tensor products, three node
        // reductions and the output transform. The fixed absolute floor also covers
        // cancellation in an individual spectral sample; real and imaginary input
        // components <= 1.
        const double tolerance =
            (3.0 * 512.0 * std::log2(2.0 * static_cast<double>(problem.dims.n_to)) +
             64.0 * static_cast<double>(problem.dims.n_to * problem.dims.n_nodes)) *
            kEpsilon;
        ASSERT_GT(tolerance, 0.0);
        native::GeneralState state(problem);
        auto scalars = trial(problem);
        scalars.raman_rates = rates;
        state.set_initial(initial, {});
        static_cast<void>(state.trial_step(0.01, scalars));
        const auto actual = state.stage_to_host(0);
        for (std::size_t j = 0; j < actual.size(); ++j)
          expect_complex_near(actual[j], expected[j], tolerance);
        // A repeated trial is the same discrete computation: held to the same
        // rounding budget, not to its bits (solprop SEP 25).
        state.set_initial(initial, {});
        static_cast<void>(state.trial_step(0.01, scalars));
        const auto repeated = state.stage_to_host(0);
        ASSERT_EQ(repeated.size(), actual.size());
        for (std::size_t j = 0; j < actual.size(); ++j)
          expect_complex_near(repeated[j], actual[j], tolerance);
        const auto ledger = state.memory_ledger();
        EXPECT_EQ(ledger.at("node_raman_drive"),
                  problem.dims.node_chunk * problem.raman_drives() * 2 *
                      problem.dims.n_to * sizeof(double));
        EXPECT_EQ(ledger.at("raman_tensor_weights"),
                  problem.raman.n_terms * 2 * sizeof(double));
      }
    }
  }
}

TEST(GeneralState, RamanTensorWeightsAreValidatedBeforeAllocation) {
  for (const auto& weights :
       {std::vector<double>{1.0}, std::vector<double>{1.0, 0.5, 1.0, 0.0},
        std::vector<double>{1.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0}}) {
    auto problem = raman_problem(native::FieldFamily::envelope);
    problem.raman.tensor_weights = weights;
    EXPECT_THROW(static_cast<void>(native::GeneralState(std::move(problem))),
                 std::invalid_argument);
  }
}

TEST(GeneralState, RamanValidationPrecedesFieldAllocation) {
  auto vector = raman_problem(native::FieldFamily::field_real);
  vector.dims.n_pol_state = 2;
  vector.linear.phi_terms.assign(2 * vector.dims.n_omega, Complex{});
  vector.linear.prefactor_terms.assign(2 * vector.dims.n_omega, Complex(1.0, 0.0));
  EXPECT_THROW(static_cast<void>(native::GeneralState(std::move(vector))),
               std::invalid_argument);

  auto wrong = raman_problem(native::FieldFamily::field_real);
  wrong.raman.undamped.pop_back();
  EXPECT_THROW(static_cast<void>(native::GeneralState(std::move(wrong))),
               std::invalid_argument);

  auto valid = raman_problem(native::FieldFamily::field_real);
  native::GeneralState state(valid);
  state.set_initial(std::vector<Complex>(valid.dims.n_omega), {});
  auto rates = trial(valid);
  rates.raman_rates = {-1.0, 0.0};
  EXPECT_THROW(static_cast<void>(state.trial_step(0.01, rates)), std::invalid_argument);
}

TEST(GeneralState, TwoStatePolarisationsUseTheSharedVectorModulus) {
  auto problem = general_problem(native::FieldFamily::field_real);
  problem.dims.n_pol_state = 2;
  problem.linear.phi_terms.assign(2 * problem.dims.n_omega, Complex{});
  problem.linear.prefactor_terms.assign(2 * problem.dims.n_omega, Complex(1.0, 0.0));
  const std::vector<Complex> initial{{1.0, 0.0}, {0.25, -0.125}, {-0.1, 0.05},
                                     {0.5, 0.0}, {-0.2, 0.1},    {0.05, -0.025}};
  std::array<std::vector<double>, 2> time{
      real_time(std::vector<Complex>(initial.begin(), initial.begin() + 3),
                problem.dims.n_to),
      real_time(std::vector<Complex>(initial.begin() + 3, initial.end()),
                problem.dims.n_to)};
  std::array<std::vector<Complex>, 2> derivative;
  for (std::size_t pol = 0; pol < 2; ++pol) {
    std::vector<Complex> response(problem.dims.n_to);
    for (std::size_t sample = 0; sample < problem.dims.n_to; ++sample) {
      const double first = problem.transform.synthesis * time[0][sample];
      const double second = problem.transform.synthesis * time[1][sample];
      const double modulus = first * first + second * second;
      response[sample] =
          Complex(problem.transform.projection * problem.kerr.coefficient * modulus *
                      (pol == 0 ? first : second),
                  0.0);
    }
    const auto transformed = dft(response);
    derivative[pol] = {transformed.begin(), transformed.begin() + 3};
  }
  constexpr double dz = 0.01;
  native::GeneralState state(problem);
  state.set_initial(initial, {});
  static_cast<void>(state.trial_step(dz, trial(problem)));
  state.finish(dz);
  const auto result = state.field_to_host();
  for (std::size_t pol = 0; pol < 2; ++pol) {
    for (std::size_t index = 0; index < problem.dims.n_omega; ++index) {
      const auto flat = pol * problem.dims.n_omega + index;
      const Complex expected = initial[flat] + dz * derivative[pol][index];
      expect_complex_near(result[flat], expected,
                          1024.0 * static_cast<double>(problem.dims.n_to) * kEpsilon);
    }
  }
}

TEST(GeneralState, HeldCurrentMatchesTheDiscreteSinusoidalDrift) {
  auto problem = plasma_problem();
  problem.plasma.stash = false;
  const double theta =
      2.0 * std::numbers::pi_v<double> / static_cast<double>(problem.dims.n_to);
  problem.grid.omega = {0.0, theta / problem.grid.dt_fine,
                        2.0 * theta / problem.grid.dt_fine};
  problem.grid.active_mask[0] = 0;
  problem.plasma.field_min = 1e6;
  problem.plasma.field_max = 1e7;
  problem.plasma.log_field_min = std::log(problem.plasma.field_min);
  problem.plasma.d_log_field = std::numbers::ln10;
  constexpr double population = 0.25;
  // The unit mode at bin one produces E_n = 2*synthesis/N*cos(n theta).
  // The exact finite trapezoid sum of cos is dt*cot(theta/2)*sin(n theta)/2.
  const double amplitude =
      2.0 * problem.transform.synthesis / static_cast<double>(problem.dims.n_to);
  const double current_scale = population * problem.plasma.charge_mass_ratio *
                               amplitude * problem.grid.dt_fine /
                               (2.0 * std::tan(theta / 2.0));
  std::vector<Complex> product(problem.dims.n_to);
  std::vector<Complex> edge(problem.dims.n_to);
  double operand = 0.0;
  for (std::size_t time = 0; time < problem.dims.n_to; ++time) {
    const double angle = theta * static_cast<double>(time);
    problem.grid.time_window[time] = 0.5 * (1.0 - std::cos(angle));
    problem.grid.time_window_derivative[time] =
        0.5 * theta / problem.grid.dt_fine * std::sin(angle);
    const double current = current_scale * std::sin(angle);
    product[time] =
        problem.transform.projection * problem.grid.time_window[time] * current;
    edge[time] = problem.transform.projection *
                 problem.grid.time_window_derivative[time] * current;
    operand +=
        Kokkos::abs(product[time]) / problem.grid.omega[1] +
        Kokkos::abs(edge[time]) / (problem.grid.omega[1] * problem.grid.omega[1]);
  }
  // Independent trigonometric sums, N trapezoids and three transforms. The bound
  // uses both absolute operands before their cancellation, fixed before execution.
  const double tolerance =
      512.0 * static_cast<double>(problem.dims.n_to) * kEpsilon * operand;
  ASSERT_GT(tolerance, 0.0);
  const auto x = dft(product);
  const auto z = dft(edge);
  native::GeneralState state(problem);
  auto scalars = trial(problem);
  scalars.preionisation = {population};
  state.set_initial(std::vector<Complex>{{}, {1.0, 0.0}, {}}, {});
  static_cast<void>(state.trial_step(0.0, scalars));
  const auto actual = state.stage_to_host(0);
  EXPECT_EQ(actual[0], Complex{});
  for (std::size_t bin = 1; bin < problem.dims.n_omega; ++bin) {
    const double omega = problem.grid.omega[bin];
    const Complex expected =
        Complex(0.0, -1.0 / omega) * x[bin] + z[bin] / (omega * omega);
    expect_complex_near(actual[bin], expected, tolerance);
  }
  state.set_initial(std::vector<Complex>(problem.dims.n_omega), {});
  static_cast<void>(state.trial_step(0.0, scalars));
  for (const auto value : state.stage_to_host(0))
    EXPECT_EQ(value, Complex{});
}

TEST(GeneralState, HeldCurrentRefusesMissingDerivativeAndNonpositiveActiveFrequency) {
  auto problem = plasma_problem();
  problem.grid.time_window_derivative.clear();
  EXPECT_THROW(native::GeneralState state(problem), std::invalid_argument);
  problem = plasma_problem();
  problem.grid.time_window_derivative[0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(native::GeneralState state(problem), std::invalid_argument);
  problem = plasma_problem();
  problem.grid.omega[0] = 0.0;
  EXPECT_THROW(native::GeneralState state(problem), std::invalid_argument);
  problem.grid.active_mask[0] = 0;
  EXPECT_NO_THROW(native::GeneralState state(problem));
}

TEST(GeneralState, RejectedPlasmaTrialCannotReplaceTheAcceptedStash) {
  const auto problem = plasma_problem();
  const std::vector<Complex> initial{{2.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  auto accepted = trial(problem);
  accepted.preionisation = {0.1};
  auto rejected = trial(problem);
  rejected.preionisation = {0.7};
  constexpr double density = 4.0;
  const double exposure = problem.plasma.rate_max * problem.grid.dt_fine *
                          static_cast<double>(problem.dims.n_to - 1);
  const auto final_fraction = [exposure](const double preionisation) {
    return 1.0 - (1.0 - preionisation) * std::exp(-exposure);
  };
  native::GeneralState state(problem);
  state.set_initial(initial, {});
  static_cast<void>(state.trial_step(0.0, accepted));
  state.finish(0.0);
  static_cast<void>(state.trial_step(0.0, rejected));
  constexpr auto flags =
      static_cast<std::uint64_t>(native::statistics_electron_density) |
      static_cast<std::uint64_t>(native::statistics_peak_ionisation_rate);
  const auto retained = state.statistics(flags, {}, {}, density, 1.0, 0.7, 1.0);
  ASSERT_EQ(retained.size(), 2);
  EXPECT_NEAR(retained[0], density * final_fraction(0.1), 64.0 * kEpsilon);
  EXPECT_NEAR(retained[1], problem.plasma.rate_max, 32.0 * kEpsilon);
  state.clear_fsal();
  const auto rescanned = state.statistics(flags, {}, {}, density, 1.0, 0.7, 1.0);
  EXPECT_NEAR(rescanned[0], density * final_fraction(0.7), 64.0 * kEpsilon);
  state.set_initial(initial, {});
  const auto reset = state.statistics(flags, {}, {}, density, 1.0, 0.2, 1.0);
  EXPECT_NEAR(reset[0], density * final_fraction(0.2), 64.0 * kEpsilon);
}

TEST(GeneralState, IndependentChannelStashesSurviveRejectedTrials) {
  auto problem = plasma_problem();
  problem.plasma.n_channels = 2;
  const std::array<double, 2> rates{0.125, 0.25};
  for (std::size_t channel = 0; channel < 2; ++channel) {
    const std::array<double, 9> parameters{problem.plasma.log_field_min,
                                           problem.plasma.d_log_field,
                                           problem.plasma.field_min,
                                           problem.plasma.field_max,
                                           rates[channel],
                                           1.0,
                                           4.0 * static_cast<double>(channel),
                                           problem.plasma.charge_mass_ratio,
                                           problem.plasma.ionisation_potential};
    problem.plasma.channel_parameters.insert(problem.plasma.channel_parameters.end(),
                                             parameters.begin(), parameters.end());
    const std::array<double, 4> coefficients{0.0, 0.0, 0.0, std::log(rates[channel])};
    problem.plasma.channel_coefficients.insert(
        problem.plasma.channel_coefficients.end(), coefficients.begin(),
        coefficients.end());
  }
  auto accepted = trial(problem);
  accepted.preionisation = {0.0};
  accepted.plasma_densities = {2.0, 3.0};
  accepted.plasma_preionisation = {0.1, 0.3};
  auto rejected = accepted;
  rejected.plasma_preionisation = {0.7, 0.8};
  // Constant rates integrate exactly by trapezoids; fewer than 64 scalar
  // operations enter each fraction, including exp, before statistics are read.
  constexpr double tolerance = 64.0 * kEpsilon;
  const auto expected = [&](const double initial, const std::size_t channel) {
    return 1.0 - (1.0 - initial) * std::exp(-rates[channel] * problem.grid.dt_fine *
                                            static_cast<double>(problem.dims.n_to - 1));
  };
  native::GeneralState state(problem);
  state.set_initial(std::vector<Complex>{{2.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}, {});
  static_cast<void>(state.trial_step(0.0, accepted));
  state.finish(0.0);
  state.promote_fsal();
  static_cast<void>(state.trial_step(0.0, rejected));
  const auto retained = state.statistics(native::statistics_populations, {}, {}, 0.0,
                                         1.0, 0.0, 1.0, rejected.plasma_preionisation);
  ASSERT_EQ(retained.size(), 4U);
  for (std::size_t channel = 0; channel < 2; ++channel) {
    EXPECT_NEAR(retained[2 * channel],
                expected(accepted.plasma_preionisation[channel], channel), tolerance);
    EXPECT_NEAR(retained[2 * channel + 1], rates[channel], tolerance);
  }
  state.clear_fsal();
  const auto fresh = state.statistics(native::statistics_populations, {}, {}, 0.0, 1.0,
                                      0.0, 1.0, rejected.plasma_preionisation);
  for (std::size_t channel = 0; channel < 2; ++channel) {
    EXPECT_NEAR(fresh[2 * channel],
                expected(rejected.plasma_preionisation[channel], channel), tolerance);
  }
}

TEST(GeneralState, MemoryLedgerMatchesConditionalAllocationFormulae) {
  auto problem = multimode_problem();
  problem.stepper.weighted_controller = true;
  problem.plasma.enabled = true;
  problem.plasma.stash = true;
  problem.grid.dt_fine = 0.125;
  problem.plasma.charge_mass_ratio = 2.0;
  problem.plasma.ionisation_potential = 3.0;
  problem.plasma.log_field_min = std::log(0.01);
  problem.plasma.d_log_field = std::log(1000.0);
  problem.plasma.field_min = 0.01;
  problem.plasma.field_max = 10.0;
  problem.plasma.rate_max = 0.5;
  problem.plasma.rate_intervals = 1;
  problem.plasma.rate_coefficients = {0.0, 0.0, 0.0, std::log(0.5)};
  native::GeneralState state(problem);
  const auto ledger = state.memory_ledger();
  const std::size_t real = sizeof(double);
  const std::size_t complex = sizeof(Complex);
  EXPECT_EQ(ledger.at("state"), problem.n_columns() * problem.dims.n_omega * complex);
  EXPECT_EQ(ledger.at("stages"), problem.stepper.n_stages * problem.n_columns() *
                                     problem.dims.n_omega * complex);
  EXPECT_EQ(ledger.at("transform_matrices"),
            (problem.transform.synthesis_matrix.size() +
             problem.transform.projection_matrix.size() +
             problem.transform.projection_coarse_matrix.size()) *
                real);
  EXPECT_EQ(ledger.at("stacked_projection"),
            2 * problem.n_columns() * problem.dims.n_to * real);
  EXPECT_EQ(ledger.at("scaled_transform_matrices"),
            5 * problem.dims.n_modes * problem.dims.n_nodes * real);
  EXPECT_EQ(ledger.at("plasma_stash"),
            2 * problem.dims.n_nodes * (1 + 2 * problem.plasma.n_channels) * real);
  // The device column's rows; on the host the band statistics' three rows per team
  // slot, at most one slot per thread (a trapezoid problem keeps no band rows).
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const auto chunk_rows = problem.dims.node_chunk * problem.effective_node_pol();
  // With the trapezoid's binary64 tile totals, a row's tiles of 1,024 samples.
  const auto tiles = (problem.dims.n_to + 1023) / 1024;
  EXPECT_EQ(ledger.at("plasma_device_scan"),
            (3 * problem.dims.node_chunk * problem.dims.n_to +
             3 * chunk_rows * problem.dims.n_to + chunk_rows * tiles) *
                real);
#else
  // One slot per thread of the response operation's own width (problem-size policy).
  const auto response_elements =
      problem.dims.node_chunk * problem.effective_node_pol() * problem.dims.n_to;
  const int width = native::cpu_size_policy_enabled
                        ? native::operation_threads(
                              native::thread_policy(), native::test::table(),
                              native::CpuPool::kernels, native::CpuOperation::response,
                              response_elements, problem.dims.n_to)
                        : native::ExecutionSpace().concurrency();
  const auto slots =
      std::min(problem.dims.node_chunk, static_cast<std::size_t>(std::max(1, width)));
  EXPECT_EQ(ledger.at("plasma_device_scan"), 3 * slots * problem.dims.n_to * real);
#endif
  EXPECT_EQ(ledger.at("weighted_control"),
            (problem.dims.n_omega + problem.n_columns()) * real);
  EXPECT_EQ(ledger.at("error_partials"), 3 * problem.n_columns() * real);
  EXPECT_EQ(ledger.at("statistics_spectrum"),
            problem.n_columns() * problem.dims.n_t * complex);
}

TEST(GeneralState, RamanMemoryLedgerMatchesTheConditionalViewFormulae) {
  const auto problem = raman_problem(native::FieldFamily::field_real);
  native::GeneralState state(problem);
  const auto ledger = state.memory_ledger();
  const auto real = sizeof(double);
  const auto complex = sizeof(Complex);
  EXPECT_EQ(ledger.at("node_raman_drive"),
            problem.dims.node_chunk * 2 * problem.dims.n_to * real);
  EXPECT_EQ(ledger.at("node_raman_spectrum"),
            problem.dims.node_chunk * (problem.dims.n_to + 1) * complex);
  EXPECT_EQ(ledger.at("node_raman_convolved"),
            problem.dims.node_chunk * 2 * problem.dims.n_to * real);
  EXPECT_EQ(ledger.at("raman_undamped"),
            problem.raman.n_terms * 2 * problem.dims.n_to * real);
  EXPECT_EQ(ledger.at("raman_kernel_time"), 2 * problem.dims.n_to * real);
  EXPECT_EQ(ledger.at("raman_kernel_spectrum"), (problem.dims.n_to + 1) * complex);
  // Two slots of the packed stage scalars, the trial's step and its sample.
  EXPECT_EQ(ledger.at("stage_raman_rates"),
            2 * problem.stepper.n_stages * problem.raman.n_terms * real);

  native::GeneralState ordinary(multimode_problem());
  const auto absent = ordinary.memory_ledger();
  for (const auto* name :
       {"node_raman_drive", "node_raman_spectrum", "node_raman_convolved",
        "raman_undamped", "raman_kernel_time", "raman_kernel_spectrum",
        "stage_raman_rates"})
    EXPECT_EQ(absent.at(name), 0U);
}

TEST(GeneralState, ProviderPolicyAndLifecycleAreIntrospectable) {
  native::GeneralState state(multimode_problem());
  // A host build's hybrid policy sends every eligible transform to the vendor GEMM; a
  // device build keeps its shape guard, which this two-node problem is below.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr bool hybrid_vendor = false;
#else
  constexpr bool hybrid_vendor = true;
#endif
  const auto expected = hybrid_vendor
                            ? std::string(native::selected_blas_provider_name())
                            : std::string("custom");
  EXPECT_EQ(state.synthesis_provider(), expected);
  EXPECT_EQ(state.projection_provider(), expected);
  EXPECT_NE(state.gemm_shape_policy().find("every whole transform on the host"),
            std::string::npos);
  state.close();
  state.close();
  EXPECT_TRUE(state.closed());
  EXPECT_THROW(state.set_initial(std::vector<Complex>(6), {}), std::logic_error);
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
std::vector<std::string> synthesis_regions;

struct SynthesisCapture {
  Kokkos::Tools::Experimental::EventSet previous =
      Kokkos::Tools::Experimental::get_callbacks();

  SynthesisCapture() {
    synthesis_regions.clear();
    Kokkos::Tools::Experimental::set_push_region_callback(
        [](const char* name) { synthesis_regions.emplace_back(name); });
  }
  ~SynthesisCapture() { Kokkos::Tools::Experimental::set_callbacks(previous); }
  SynthesisCapture(const SynthesisCapture&) = delete;
  SynthesisCapture& operator=(const SynthesisCapture&) = delete;
  SynthesisCapture(SynthesisCapture&&) = delete;
  SynthesisCapture& operator=(SynthesisCapture&&) = delete;
};

// Five nodes and modes in chunks of three, the tail two. The square map takes the
// retained-spectrum synthesis; with an in-stage absorber, which reads the modal time
// field, it keeps the interleaved real GEMM over the time samples. Both must reach
// BLAS for the full chunk and the tail rather than the scalar custom loop, and agree
// with the whole transform.
void check_radial_analytic_chunks(const bool absorber, const char* expected_region) {
  auto problem = general_problem(native::FieldFamily::field_analytic);
  problem.dims.n_modes = 5;
  problem.dims.n_nodes = 5;
  problem.dims.node_chunk = 3;
  problem.spatial.spectral_weights.assign(problem.dims.n_modes, 1.0);
  problem.spatial.on_axis.assign(problem.dims.n_modes, 1.0);
  problem.linear.phi_terms.assign(problem.dims.n_modes * problem.dims.n_omega,
                                  Complex{});
  problem.linear.prefactor_terms.assign(problem.dims.n_modes * problem.dims.n_omega,
                                        Complex(1.0, 0.0));
  problem.absorber.in_stage = absorber;
  if (absorber) {
    problem.absorber.fine.resize(problem.dims.n_to);
    for (std::size_t time = 0; time < problem.dims.n_to; ++time)
      problem.absorber.fine[time] = 0.01 * static_cast<double>(time % 3);
  }
  problem.transform.synthesis_matrix.resize(problem.dims.n_nodes *
                                            problem.dims.n_modes);
  problem.transform.projection_matrix.resize(problem.dims.n_modes *
                                             problem.dims.n_nodes);
  for (std::size_t row = 0; row < problem.dims.n_nodes; ++row) {
    for (std::size_t column = 0; column < problem.dims.n_modes; ++column) {
      const auto index = row * problem.dims.n_modes + column;
      problem.transform.synthesis_matrix[index] =
          row == column ? 1.0 : 0.125 * static_cast<double>(row + 1);
      problem.transform.projection_matrix[index] =
          row == column ? 0.75 : -0.0625 * static_cast<double>(column + 1);
    }
  }
  std::vector<Complex> initial(problem.dims.n_modes * problem.dims.n_omega);
  for (std::size_t index = 0; index < initial.size(); ++index)
    initial[index] = Complex(0.01 * static_cast<double>(index + 1), -0.02);
  auto whole_problem = problem;
  whole_problem.dims.node_chunk = whole_problem.dims.n_nodes;
  native::GeneralState whole(whole_problem);
  whole.set_initial(initial, {});
  static_cast<void>(whole.trial_step(0.01, trial(whole_problem)));
  whole.finish(0.01);
  const auto expected = whole.field_to_host();
  native::GeneralState chunked(problem);
  const auto ledger = chunked.memory_ledger();
  chunked.set_initial(initial, {});
  {
    const SynthesisCapture capture;
    static_cast<void>(chunked.trial_step(0.01, trial(problem)));
  }
  const auto selected =
      std::count(synthesis_regions.begin(), synthesis_regions.end(), expected_region);
  const auto custom = std::count(synthesis_regions.begin(), synthesis_regions.end(),
                                 "solprop_native.transform.synthesis.custom");
  EXPECT_EQ(selected, 2);
  EXPECT_EQ(custom, 0);
  // Only the absorber-free map keeps the retained spectra.
  EXPECT_EQ(ledger.at("retained_synthesis") > 0, !absorber);
  chunked.finish(0.01);
  const auto actual = chunked.field_to_host();
  // Both computations have one inverse/forward FFT and two length-five products;
  // carry their rounding through the cubic and the stage's scaled increment.
  const double tolerance =
      512.0 * kEpsilon * static_cast<double>(problem.dims.n_to) +
      4.0 * (static_cast<double>(problem.dims.n_modes) * kEpsilon /
             (1.0 - static_cast<double>(problem.dims.n_modes) * kEpsilon));
  ASSERT_GT(tolerance, 0.0);
  for (std::size_t index = 0; index < actual.size(); ++index)
    expect_complex_near(actual[index], expected[index],
                        tolerance * std::max(1.0, Kokkos::abs(expected[index])));
  EXPECT_EQ(chunked.memory_ledger(), ledger);
}

TEST(GeneralState, RadialAnalyticChunksUseInterleavedGemmIncludingTheTail) {
  check_radial_analytic_chunks(
      true, "solprop_native.transform.synthesis.kokkosblas_interleaved");
}

TEST(GeneralState, SquareRadialAnalyticChunksSynthesiseRetainedSpectra) {
  check_radial_analytic_chunks(false, "solprop_native.transform.synthesis.retained");
}
#endif

TEST(GeneralState, ChunkedBlasBuffersHaveExactPersistentByteCost) {
  auto problem = multimode_problem(3);
  problem.dims.n_to = 1024;
  problem.dims.n_nodes = 8;
  problem.grid.time_window.assign(problem.dims.n_to, 1.0);
  problem.grid.carrier.assign(problem.dims.n_to, Complex(1.0, 0.0));
  problem.transform.synthesis_matrix.assign(problem.dims.n_nodes * problem.dims.n_modes,
                                            0.5);
  problem.transform.projection_matrix.assign(
      problem.dims.n_modes * problem.dims.n_nodes, 0.25);
  problem.transform.projection_coarse_matrix.assign(
      problem.dims.n_modes * problem.dims.n_nodes, 0.125);

  native::GeneralState state(problem);
  const auto ledger = state.memory_ledger();
  EXPECT_EQ(ledger.at("chunk_transform_buffers"),
            (3U + 2U) * problem.dims.n_modes * 3U * sizeof(double));

  auto whole_problem = problem;
  whole_problem.dims.node_chunk = whole_problem.dims.n_nodes;
  const std::vector<Complex> initial{{1.0, 0.0}, {0.25, -0.125}, {-0.1, 0.05},
                                     {0.5, 0.0}, {-0.2, 0.1},    {0.05, -0.025}};
  native::GeneralState whole(whole_problem);
  whole.set_initial(initial, {});
  static_cast<void>(whole.trial_step(0.01, trial(whole_problem)));
  whole.finish(0.01);
  state.set_initial(initial, {});
  static_cast<void>(state.trial_step(0.01, trial(problem)));
  state.finish(0.01);
  const auto reference = whole.field_to_host();
  const auto chunked = state.field_to_host();
  const auto dot_length = static_cast<double>(problem.dims.n_nodes);
  const double dot_gamma = dot_length * kEpsilon / (1.0 - dot_length * kEpsilon);
  const double tolerance =
      512.0 * kEpsilon * std::ceil(std::log2(static_cast<double>(problem.dims.n_to))) +
      4.0 * dot_gamma;
  for (std::size_t index = 0; index < chunked.size(); ++index)
    expect_complex_near(chunked[index], reference[index],
                        tolerance * std::max(1.0, Kokkos::abs(reference[index])));
}

#ifdef SOLPROP_NATIVE_TPL_BLAS_OPENBLAS
TEST(GeneralState, OpenBlasEnvelopePolicyAllocatesAndExecutesComplexTransforms) {
  auto problem = multimode_problem(8);
  problem.dims.family = native::FieldFamily::envelope;
  problem.dims.n_t = 1024;
  problem.dims.n_to = 1024;
  problem.dims.n_omega = 1024;
  problem.dims.n_nodes = 8;
  problem.grid.crop_index.clear();
  for (std::size_t index = 0; index < problem.dims.n_omega; ++index)
    problem.grid.crop_index.push_back(index);
  problem.grid.active_mask.assign(problem.dims.n_omega, 1U);
  problem.grid.norm_mask.assign(problem.dims.n_omega, 1U);
  problem.grid.omega_window.assign(problem.dims.n_omega, 1.0);
  problem.grid.omega.assign(problem.dims.n_omega, 1.0);
  problem.grid.time_window.assign(problem.dims.n_to, 1.0);
  problem.grid.carrier.assign(problem.dims.n_to, Complex(1.0, 0.0));
  problem.linear.phi_terms.assign(problem.dims.n_modes * problem.dims.n_omega,
                                  Complex{});
  problem.linear.prefactor_terms.assign(problem.dims.n_modes * problem.dims.n_omega,
                                        Complex(1.0, 0.0));
  problem.transform.synthesis_matrix.assign(problem.dims.n_nodes * problem.dims.n_modes,
                                            0.5);
  problem.transform.projection_matrix.assign(
      problem.dims.n_modes * problem.dims.n_nodes, 0.25);
  problem.transform.projection_coarse_matrix.assign(
      problem.dims.n_modes * problem.dims.n_nodes, 0.125);

  native::GeneralState state(problem);
  // Scalar envelopes use a real GEMM over the complex samples' interleaved view.
  EXPECT_EQ(state.synthesis_provider(), "kokkosblas-tpl-interleaved-real");
  EXPECT_EQ(state.projection_provider(), "kokkosblas-tpl-interleaved-real");
  const auto ledger = state.memory_ledger();
  const auto real_matrix_values = 3 * problem.dims.n_modes * problem.dims.n_nodes;
  const auto complex_matrix_values = 5 * problem.dims.n_modes * problem.dims.n_nodes;
  EXPECT_EQ(ledger.at("scaled_transform_matrices"),
            real_matrix_values * sizeof(double) +
                complex_matrix_values * sizeof(Complex));
  EXPECT_EQ(ledger.at("stacked_projection"),
            2 * problem.dims.n_modes * problem.dims.n_to * sizeof(Complex));

  state.set_initial(std::vector<Complex>(problem.dims.n_modes * problem.dims.n_omega),
                    {});
  const auto partials = state.trial_step(0.01, trial(problem));
  // Exact: a zero field has zero norms and a zero error.
  EXPECT_EQ(partials, std::vector<double>(3 * problem.dims.n_modes, 0.0));
}
#endif

TEST(GeneralState, RejectsOverflowingExtentsBeforeAllocation) {
  auto column_overflow = general_problem(native::FieldFamily::field_real);
  column_overflow.dims.n_modes = std::numeric_limits<std::size_t>::max();
  column_overflow.dims.n_pol_state = 2;
  EXPECT_THROW(static_cast<void>(native::GeneralState(std::move(column_overflow))),
               std::length_error);

  auto field_overflow = general_problem(native::FieldFamily::field_real);
  field_overflow.dims.n_modes =
      std::numeric_limits<std::size_t>::max() / field_overflow.dims.n_omega + 1;
  EXPECT_THROW(static_cast<void>(native::GeneralState(std::move(field_overflow))),
               std::length_error);
}

// solprop's hybrid planar kind (kernel schema 15): two slab channels on a three-node x
// rule, times a four-row y axis, field_real Kerr. Columns (p, k_y) are p + 2 iy, nodes
// (x, y) are ix + 3 iy, and each column's operator subtracts its channel's wall term.
native::Problem planar_problem(const std::size_t rows_per_chunk) {
  auto problem = base_problem(native::FieldFamily::field_real);
  problem.spatial.n_p = 2;
  problem.spatial.n_x = 3;
  problem.spatial.n_y = 4;
  problem.dims.n_modes = problem.spatial.n_p * problem.spatial.n_y;
  problem.dims.n_nodes = problem.spatial.n_x * problem.spatial.n_y;
  problem.dims.node_chunk = rows_per_chunk * problem.spatial.n_x;
  problem.kerr.nonlinear = true;
  problem.linear.phi_terms.clear();
  problem.linear.prefactor_terms.clear();
  problem.transform.synthesis_matrix = {1.0, 0.5, -0.25, 1.5, 0.75, -0.5};
  problem.transform.projection_matrix = {0.5, 0.25, -0.125, 0.375, 0.625, -0.25};
  problem.spatial.k_squared = {{40.0, 0.5}, {60.0, 1.0}, {90.0, 1.5}};
  problem.spatial.k_squared_low = {{1e-15, 0.0}, {-2e-15, 0.0}, {0.0, 1e-16}};
  problem.spatial.numerator = {{0.0, -1.0}, {0.0, -2.0}, {0.0, -3.0}};
  problem.spatial.clamped = {{0.0, -0.25}, {0.0, -0.5}, {0.0, -0.75}};
  problem.spatial.frame = {5.0, 6.0, 7.0};
  problem.spatial.spectral_rate = {0.0, 0.25, 0.5};
  problem.spatial.user_rate = {0.0, 0.0, 0.0};
  // k_y^2 per column, the channel fastest; the last row is evanescent in the first
  // bin, so both branches of the operator are exercised.
  const std::vector<double> ky_squared{0.0, 4.0, 45.0, 4.0};
  for (std::size_t column = 0; column < problem.dims.n_modes; ++column) {
    problem.spatial.kperp_sq.push_back(ky_squared[column / problem.spatial.n_p]);
    problem.spatial.kperp_sq_low.push_back(column == 3 ? 1e-16 : 0.0);
  }
  problem.spatial.absorber_rate.assign(problem.dims.n_modes, 0.0);
  problem.spatial.kperp_rate.assign(problem.dims.n_modes, 0.0);
  problem.spatial.evanescent_rate = 200.0;
  problem.spatial.kappa_sq = {{1.0, 0.25}, {2.0, 0.5},  {3.0, 0.75},
                              {9.0, 1.0},  {12.0, 1.5}, {15.0, 2.0}};
  return problem;
}

struct PlanarElement {
  Complex linear, prefactor;
};

PlanarElement planar_element(const native::Problem& problem, const std::size_t column,
                             const std::size_t omega) {
  const Complex square =
      (problem.spatial.k_squared[omega] - problem.spatial.kperp_sq[column]) +
      (problem.spatial.k_squared_low[omega] - problem.spatial.kperp_sq_low[column]) -
      problem.spatial
          .kappa_sq[(column % problem.spatial.n_p) * problem.dims.n_omega + omega];
  const std::complex<double> root =
      std::sqrt(std::complex<double>(square.real(), square.imag()));
  const bool propagating = square.real() > 0.0;
  const double attenuation =
      propagating ? root.imag()
                  : std::min(std::abs(root.imag()), problem.spatial.evanescent_rate);
  return {.linear =
              Complex(-attenuation, -(root.real() - problem.spatial.frame[omega])) -
              0.5 * problem.spatial.spectral_rate[omega],
          .prefactor = propagating ? problem.spatial.numerator[omega] / root.real()
                                   : problem.spatial.clamped[omega]};
}

// The stage straight from the definitions: the inverse DFT in y, each row's real time
// signal, the slab synthesis, the cube, the slab projection, the DFT in time and in y,
// and each column's prefactor.
std::vector<Complex> planar_reference(const native::Problem& problem,
                                      const std::vector<Complex>& columns) {
  const auto n_p = problem.spatial.n_p;
  const auto n_x = problem.spatial.n_x;
  const auto n_y = problem.spatial.n_y;
  const auto n = problem.dims.n_omega;
  std::vector<std::vector<double>> rows(n_p * n_y);
  for (std::size_t iy = 0; iy < n_y; ++iy)
    for (std::size_t p = 0; p < n_p; ++p) {
      std::vector<Complex> spectrum(n);
      for (std::size_t omega = 0; omega < n; ++omega)
        for (std::size_t ky = 0; ky < n_y; ++ky)
          spectrum[omega] +=
              columns[(p + n_p * ky) * n + omega] *
              cis(2.0 * std::numbers::pi_v<double> * static_cast<double>(ky * iy) /
                  static_cast<double>(n_y)) /
              static_cast<double>(n_y);
      rows[p + n_p * iy] = real_time(spectrum, problem.dims.n_to);
    }
  std::vector<Complex> projected(n_p * n_y * n);
  for (std::size_t iy = 0; iy < n_y; ++iy) {
    std::vector<std::vector<double>> polarisation(
        n_x, std::vector<double>(problem.dims.n_to));
    for (std::size_t ix = 0; ix < n_x; ++ix)
      for (std::size_t time = 0; time < problem.dims.n_to; ++time) {
        double field = 0.0;
        for (std::size_t p = 0; p < n_p; ++p)
          field += problem.transform.synthesis_matrix[ix * n_p + p] *
                   rows[p + n_p * iy][time];
        polarisation[ix][time] = problem.kerr.coefficient * field * field * field;
      }
    for (std::size_t p = 0; p < n_p; ++p) {
      std::vector<Complex> signal(problem.dims.n_to);
      for (std::size_t time = 0; time < problem.dims.n_to; ++time)
        for (std::size_t ix = 0; ix < n_x; ++ix)
          signal[time] += problem.transform.projection_matrix[p * n_x + ix] *
                          polarisation[ix][time];
      const auto spectrum = dft(signal);
      for (std::size_t omega = 0; omega < n; ++omega)
        projected[(p + n_p * iy) * n + omega] = spectrum[omega];
    }
  }
  std::vector<Complex> result(n_p * n_y * n);
  for (std::size_t ky = 0; ky < n_y; ++ky)
    for (std::size_t p = 0; p < n_p; ++p)
      for (std::size_t omega = 0; omega < n; ++omega) {
        Complex sum{};
        for (std::size_t iy = 0; iy < n_y; ++iy)
          sum += projected[(p + n_p * iy) * n + omega] *
                 cis(-2.0 * std::numbers::pi_v<double> * static_cast<double>(ky * iy) /
                     static_cast<double>(n_y));
        const auto column = p + n_p * ky;
        result[column * n + omega] = planar_element(problem, column, omega).prefactor *
                                     problem.grid.omega_window[omega] * sum;
      }
  return result;
}

TEST(GeneralState, HybridPlanarStageMatchesStraightforwardHostImplementation) {
  const std::vector<Complex> initial{
      {1.0, 0.0},     {0.25, -0.125}, {-0.1, 0.05},  {0.5, 0.0},  {-0.2, 0.1},
      {0.05, -0.025}, {0.3, 0.1},     {0.1, -0.2},   {0.05, 0.0}, {-0.4, 0.0},
      {0.2, 0.3},     {-0.1, 0.1},    {0.6, 0.0},    {0.0, 0.25}, {-0.05, 0.1},
      {0.2, 0.0},     {0.1, 0.1},     {0.1, -0.1},   {-0.3, 0.0}, {0.15, 0.05},
      {0.0, -0.1},    {0.25, 0.0},    {-0.05, 0.05}, {0.02, 0.01}};
  // One y row per chunk (a partial chunk nowhere), then three (a partial last one).
  for (const std::size_t rows_per_chunk : {std::size_t{1}, std::size_t{3}}) {
    const auto problem = planar_problem(rows_per_chunk);
    ASSERT_EQ(initial.size(), problem.dims.n_modes * problem.dims.n_omega);
    const auto derivative = planar_reference(problem, initial);
    constexpr double dz = 0.01;
    auto scalars = trial(problem);
    scalars.phi_delta = {0.0, dz};
    native::GeneralState state(problem);
    state.set_initial(initial, {});
    static_cast<void>(state.trial_step(dz, scalars));
    const auto stage = state.stage_to_host(0);
    ASSERT_EQ(stage.size(), derivative.size());
    for (std::size_t index = 0; index < stage.size(); ++index)
      expect_complex_near(stage[index], derivative[index],
                          256.0 * kEpsilon *
                              std::max(1.0, Kokkos::abs(derivative[index])));
    // The accepted step is each column's exact linear flow over dz of the stage-
    // weighted update (one stage, unit weight), which carries the wall terms.
    state.finish(dz);
    const auto result = state.field_to_host();
    for (std::size_t column = 0; column < problem.dims.n_modes; ++column)
      for (std::size_t omega = 0; omega < problem.dims.n_omega; ++omega) {
        const auto index = column * problem.dims.n_omega + omega;
        const Complex exponent = dz * planar_element(problem, column, omega).linear;
        const Complex expected = std::exp(exponent.real()) * cis(exponent.imag()) *
                                 (initial[index] + dz * derivative[index]);
        expect_complex_near(result[index], expected,
                            256.0 * kEpsilon * std::max(1.0, Kokkos::abs(expected)));
      }
  }
}

TEST(GeneralState, HybridPlanarProblemsAreValidatedBeforeAllocation) {
  auto partial_row = planar_problem(1);
  partial_row.dims.node_chunk = 4;
  EXPECT_THROW(native::GeneralState state(partial_row), std::invalid_argument);
  auto short_kappa = planar_problem(1);
  short_kappa.spatial.kappa_sq.pop_back();
  EXPECT_THROW(native::GeneralState state(short_kappa), std::invalid_argument);
  auto coarse = planar_problem(1);
  coarse.transform.projection_coarse_matrix = coarse.transform.projection_matrix;
  EXPECT_THROW(native::GeneralState state(coarse), std::invalid_argument);
  auto vector = planar_problem(1);
  vector.dims.n_pol_state = 2;
  EXPECT_THROW(native::GeneralState state(vector), std::invalid_argument);
  auto columns = planar_problem(1);
  columns.spatial.n_p = 3;
  EXPECT_THROW(native::GeneralState state(columns), std::invalid_argument);
  auto modal = general_problem(native::FieldFamily::field_real);
  modal.spatial.kappa_sq = {{1.0, 0.0}};
  EXPECT_THROW(native::GeneralState state(modal), std::invalid_argument);
}

TEST(GeneralState, RepeatedTrialsOverwriteFullCropAndStageZeroAccumulators) {
  auto problem = general_problem(native::FieldFamily::envelope);
  problem.kerr.nonlinear = true;
  const std::vector<Complex> initial = {
      {1.0, 0.0}, {0.25, -0.125}, {-0.1, 0.05}, {0.05, 0.025}};
  native::GeneralState state(problem);
  state.set_initial(initial, {});
  const auto first = state.trial_step(0.02, trial(problem, 1.5));
  const auto second = state.trial_step(0.02, trial(problem, 1.5));
  // The same trial twice: the partials, squared norms of the field, the solution and
  // the embedded error, agree to rounding (solprop SEP 25). The stage carries the
  // Kerr families' `512 eps n_to` relative to the field; a squared norm doubles it.
  // Accumulators left over from the first trial would add a whole partial.
  ASSERT_EQ(second.size(), first.size());
  double largest = 0.0;
  for (const double value : first)
    largest = std::max(largest, std::abs(value));
  ASSERT_GT(largest, 0.0);
  const double tolerance =
      2.0 * 512.0 * kEpsilon * static_cast<double>(problem.dims.n_to) * largest;
  for (std::size_t index = 0; index < first.size(); ++index)
    EXPECT_NEAR(second[index], first[index], tolerance) << index;
  // Exact: a trial never writes the accepted state.
  EXPECT_EQ(state.field_to_host(), initial);
}

TEST(GeneralState, RejectsDuplicateCropBins) {
  auto problem = general_problem(native::FieldFamily::envelope);
  problem.grid.crop_index[1] = problem.grid.crop_index[0];
  EXPECT_THROW(native::GeneralState state(problem), std::invalid_argument);
}

TEST(GeneralState, SplitStepAbsorberDampsOnlyAcceptedState) {
  auto problem = general_problem(native::FieldFamily::envelope);
  problem.kerr.nonlinear = false;
  problem.absorber.split_step = true;
  problem.absorber.coarse.assign(problem.dims.n_t, 2.0);
  const std::vector<Complex> initial(problem.dims.n_omega, {1.0, -0.25});
  constexpr double dz = 0.1;
  native::GeneralState state(problem);
  state.set_initial(initial, {});
  static_cast<void>(state.trial_step(dz, trial(problem)));
  const auto before = state.field_to_host();
  // Exact: the absorber damps only an accepted state, and a trial writes none.
  EXPECT_EQ(before, initial);
  state.finish(dz);
  const auto after = state.field_to_host();
  const double scale = std::exp(-0.5 * 2.0 * dz);
  for (std::size_t index = 0; index < after.size(); ++index) {
    expect_complex_near(after[index], scale * initial[index], 256.0 * kEpsilon);
  }
  state.promote_fsal();
  EXPECT_FALSE(state.fsal_ready());
}

TEST(GeneralState, InStageAbsorberPreservesFirstSameAsLastEligibility) {
  auto problem = general_problem(native::FieldFamily::envelope);
  problem.kerr.nonlinear = false;
  problem.absorber.in_stage = true;
  problem.stepper.first_same_as_last = true;
  problem.absorber.fine.assign(problem.dims.n_to, 2.0);
  const std::vector<Complex> initial(problem.dims.n_omega, {1.0, 0.0});
  constexpr double dz = 0.1;
  native::GeneralState state(problem);
  state.set_initial(initial, {});
  static_cast<void>(state.trial_step(dz, trial(problem)));
  state.finish(dz);
  state.promote_fsal();
  EXPECT_TRUE(state.fsal_ready());
  const auto after = state.field_to_host();
  for (const auto value : after) {
    expect_complex_near(value, {0.9, 0.0}, 256.0 * kEpsilon);
  }
  state.clear_fsal();
  EXPECT_FALSE(state.fsal_ready());
}

// solprop SEP 24 phase 3: the low-storage RK4IP march on a constant multimode operator,
// against the midpoint interaction picture formed on the host from the straightforward
// source (`multimode_reference`) and the exact half-step flows.
native::Problem low_storage_problem(const std::size_t flows) {
  auto problem = multimode_problem();
  problem.stepper.n_stages = 3;
  problem.stepper.tableau_matrix.assign(9, 0.0);
  problem.stepper.tableau_nodes = {0.0, 0.5, 1.0};
  problem.stepper.solution_weights = {1.0, 0.0, 0.0};
  problem.stepper.error_weights = {0.25, 0.0, -0.25};
  problem.transform.projection_coarse_matrix.clear();
  problem.stepper.low_storage = true;
  problem.stepper.low_storage_flows = flows;
  for (std::size_t mode = 0; mode < problem.dims.n_modes; ++mode)
    for (std::size_t index = 0; index < problem.dims.n_omega; ++index)
      problem.linear.phi_terms[mode * problem.dims.n_omega + index] =
          Complex(-0.3 * static_cast<double>(index), 2.0 + static_cast<double>(mode));
  return problem;
}

native::TrialScalars low_storage_scalars(const native::Problem& problem,
                                         const double dz) {
  auto scalars = trial(problem);
  scalars.phi_delta[0] = 0.5 * dz;
  scalars.phi_delta[1] = 0.5 * dz;
  scalars.phi_delta[2] = 0.5 * dz;
  return scalars;
}

std::vector<Complex> low_storage_reference(const native::Problem& problem,
                                           const std::vector<Complex>& state,
                                           const double dz) {
  const auto size = state.size();
  std::vector<Complex> flow(size);
  for (std::size_t element = 0; element < size; ++element) {
    const Complex exponent = 0.5 * dz * problem.linear.phi_terms[element];
    flow[element] = std::exp(exponent.real()) *
                    Complex(std::cos(exponent.imag()), std::sin(exponent.imag()));
  }
  const auto combine = [&](const std::vector<Complex>& base, const double weight,
                           const std::vector<Complex>& slope) {
    std::vector<Complex> result(size);
    for (std::size_t element = 0; element < size; ++element)
      result[element] = base[element] + weight * slope[element];
    return result;
  };
  std::vector<Complex> picture(size);
  std::vector<Complex> g1(size);
  const auto f1 = multimode_reference(problem, state);
  for (std::size_t element = 0; element < size; ++element) {
    picture[element] = flow[element] * state[element];
    g1[element] = flow[element] * f1[element];
  }
  const auto f2 = multimode_reference(problem, combine(picture, 0.5 * dz, g1));
  const auto f3 = multimode_reference(problem, combine(picture, 0.5 * dz, f2));
  auto fourth = combine(picture, dz, f3);
  for (std::size_t element = 0; element < size; ++element)
    fourth[element] *= flow[element];
  const auto f4 = multimode_reference(problem, fourth);
  std::vector<Complex> result(size);
  for (std::size_t element = 0; element < size; ++element)
    result[element] =
        flow[element] * (picture[element] + dz / 6.0 * g1[element] +
                         dz / 3.0 * f2[element] + dz / 3.0 * f3[element]) +
        dz / 6.0 * f4[element];
  return result;
}

TEST(GeneralState, LowStoragePassesMatchTheHostInteractionPictureStep) {
  const std::vector<Complex> initial{{1.0, 0.0}, {0.25, -0.125}, {-0.1, 0.05},
                                     {0.5, 0.0}, {-0.2, 0.1},    {0.05, -0.025}};
  constexpr double dz = 0.05;
  for (const std::size_t flows : {std::size_t{0}, std::size_t{1}}) {
    const auto problem = low_storage_problem(flows);
    native::GeneralState state(problem);
    state.set_initial(initial, {});
    auto expected = initial;
    for (int step = 0; step < 3; ++step) {
      // The table is built at the first step and reused at the others.
      const double squared = state.low_storage_step(
          dz, low_storage_scalars(problem, dz), flows == 1 && step == 0);
      state.finish(dz);
      expected = low_storage_reference(problem, expected, dz);
      const auto result = state.field_to_host();
      ASSERT_EQ(result.size(), expected.size());
      double norm = 0.0;
      for (std::size_t index = 0; index < result.size(); ++index) {
        // Four evaluations a step of the straightforward transforms' rounding,
        // carried over three steps: the multimode test's budget, 16 times.
        const double tolerance =
            4096.0 * kEpsilon * std::max(1.0, Kokkos::abs(expected[index]));
        expect_complex_near(result[index], expected[index], tolerance);
        norm += Kokkos::abs(result[index]) * Kokkos::abs(result[index]);
      }
      EXPECT_NEAR(squared, norm, 64.0 * kEpsilon * norm);
    }
    const auto ledger = state.memory_ledger();
    const auto field = initial.size() * sizeof(Complex);
    EXPECT_EQ(ledger.at("register_accumulator"), field);
    EXPECT_EQ(ledger.at("register_stage"), field);
    EXPECT_EQ(ledger.at("propagators"), flows * field);
    for (const char* row :
         {"advanced", "stages", "accumulator_solution", "accumulator_error",
          "weighted_residual", "recycled_stage", "stage_inverse"})
      EXPECT_EQ(ledger.at(row), 0U) << row;
  }
}

TEST(GeneralState, LowStorageStepsAreRefusedOutsideTheirState) {
  auto runge_kutta = low_storage_problem(1);
  runge_kutta.stepper.low_storage = false;
  runge_kutta.stepper.low_storage_flows = 0;
  native::GeneralState state(runge_kutta);
  state.set_initial(std::vector<Complex>(6), {});
  EXPECT_THROW(static_cast<void>(state.low_storage_step(
                   0.1, low_storage_scalars(runge_kutta, 0.1), false)),
               std::logic_error);
  auto three = low_storage_problem(3);
  EXPECT_THROW(native::GeneralState too_many(three), std::invalid_argument);
  auto elided = low_storage_problem(0);
  elided.stepper.low_storage_stage_register = false;
  EXPECT_THROW(native::GeneralState modal(elided), std::invalid_argument);
  native::GeneralState march(low_storage_problem(0));
  march.set_initial(std::vector<Complex>(6), {});
  EXPECT_THROW(static_cast<void>(march.low_storage_step(
                   0.1, low_storage_scalars(runge_kutta, 0.1), true)),
               std::logic_error);
  EXPECT_THROW(static_cast<void>(march.low_storage_step(
                   -0.1, low_storage_scalars(runge_kutta, 0.1), false)),
               std::invalid_argument);
}

// The message of the invalid_argument a native state's construction throws, or empty.
std::string construction_refusal(const native::Problem& problem) {
  try {
    native::GeneralState state(problem);
  } catch (const std::invalid_argument& error) {
    return error.what();
  }
  return {};
}

// A Cartesian low-storage problem: the planar test's operator on three by four nodes
// with no slab channels.
native::Problem low_storage_free_problem() {
  auto problem = planar_problem(1);
  problem.spatial.n_p = 0;
  problem.dims.n_modes = problem.spatial.n_x * problem.spatial.n_y;
  problem.dims.n_nodes = problem.dims.n_modes;
  problem.transform.synthesis_matrix.clear();
  problem.transform.projection_matrix.clear();
  problem.spatial.kappa_sq.clear();
  // Construction only: the unit transverse square is k_x = 1, k_y = 0.
  problem.spatial.k_x.assign(problem.spatial.n_x, 1.0);
  problem.spatial.k_y.assign(problem.spatial.n_y, 0.0);
  problem.spatial.kperp_sq.assign(problem.dims.n_modes, 1.0);
  problem.spatial.kperp_sq_low.assign(problem.dims.n_modes, 0.0);
  problem.spatial.absorber_rate.assign(problem.dims.n_modes, 0.0);
  problem.spatial.kperp_rate.assign(problem.dims.n_modes, 0.0);
  problem.stepper.low_storage = true;
  return problem;
}

TEST(GeneralState, LowStorageRefusesRegistersItsFoldsCannotRead) {
  // The exchange fold of an in-stage absorber re-forms its input from the stage
  // register, so it is never elided there.
  auto elided = low_storage_free_problem();
  elided.stepper.low_storage_stage_register = false;
  EXPECT_EQ(construction_refusal(elided), "");
  elided.absorber.in_stage = true;
  elided.absorber.fine.assign(elided.dims.n_to, 1.0);
  EXPECT_NE(construction_refusal(elided).find("beside an in-stage absorber"),
            std::string::npos);
  // A separable fold applies one flow volume, a constant operator's table.
  auto table = low_storage_free_problem();
  table.stepper.low_storage_flows = 1;
  EXPECT_EQ(construction_refusal(table), "");
  auto pair = low_storage_free_problem();
  pair.stepper.low_storage_flows = 2;
  EXPECT_NE(construction_refusal(pair).find("at most one flow volume"),
            std::string::npos);
  auto planar = planar_problem(1);
  planar.stepper.low_storage = true;
  planar.stepper.low_storage_flows = 2;
  EXPECT_NE(construction_refusal(planar).find("at most one flow volume"),
            std::string::npos);
}

TEST(GeneralState, LowStorageRefusesATableNeverBuilt) {
  const auto problem = low_storage_problem(1);
  const std::vector<Complex> initial{{1.0, 0.0}, {0.25, -0.125}, {-0.1, 0.05},
                                     {0.5, 0.0}, {-0.2, 0.1},    {0.05, -0.025}};
  constexpr double dz = 0.05;
  const auto scalars = low_storage_scalars(problem, dz);
  native::GeneralState state(problem);
  state.set_initial(initial, {});
  EXPECT_THROW(static_cast<void>(state.low_storage_step(dz, scalars, false)),
               std::logic_error);
  // The refusal wrote nothing: the state is the loaded one.
  EXPECT_EQ(state.field_to_host(), initial);
  static_cast<void>(state.low_storage_step(dz, scalars, true));
  state.finish(dz);
  static_cast<void>(state.low_storage_step(dz, scalars, false));
  state.finish(dz);
  // Loading a field forgets the table, as solprop's engine forgets its key.
  state.set_initial(initial, {});
  EXPECT_THROW(static_cast<void>(state.low_storage_step(dz, scalars, false)),
               std::logic_error);
  static_cast<void>(state.low_storage_step(dz, scalars, true));
  state.close();
  EXPECT_THROW(static_cast<void>(state.low_storage_step(dz, scalars, false)),
               std::logic_error);
}

TEST(GeneralState, CloseIsIdempotentAndRejectsFurtherUse) {
  native::GeneralState state(general_problem(native::FieldFamily::envelope));
  state.close();
  state.close();
  EXPECT_TRUE(state.closed());
  EXPECT_THROW(state.set_initial(std::vector<Complex>(4), {}), std::logic_error);
}

} // namespace
