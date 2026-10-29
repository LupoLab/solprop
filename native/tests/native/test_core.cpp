// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/core.hpp"
#include "solprop_native/gemm.hpp"
#include "solprop_native/plasma.hpp"

#include "bounds.hpp"

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
namespace test = solprop_native::test;

/** An empty team body: what a team-size query is asked about. A functor rather than a
 * lambda, which CUDA refuses inside a test's (private) body. */
struct EmptyTeamBody {
  KOKKOS_INLINE_FUNCTION void operator()(
      const Kokkos::TeamPolicy<native::ExecutionSpace>::member_type& /*team*/) const {}
};
using native::Complex;

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

native::Problem base_problem(const native::FieldFamily family) {
  native::Problem problem;
  problem.n_t = family == native::FieldFamily::envelope ? 4 : 4;
  problem.n_to = family == native::FieldFamily::envelope ? 4 : 8;
  problem.n_omega = family == native::FieldFamily::envelope ? 4 : 3;
  problem.n_stages = 1;
  problem.n_phi_terms = 1;
  problem.n_prefactor_terms = 1;
  problem.family = family;
  problem.crop_index = {0, 1, 2};
  if (family == native::FieldFamily::envelope)
    problem.crop_index.push_back(3);
  problem.active_mask.assign(problem.n_omega, 1U);
  problem.norm_mask.assign(problem.n_omega, 1U);
  problem.omega_window.assign(problem.n_omega, 1.0);
  for (std::size_t index = 0; index < problem.n_omega; ++index)
    problem.omega.push_back(static_cast<double>(index + 1));
  problem.dt = 0.25;
  problem.time_window.assign(problem.n_to, 1.0);
  problem.time_window_derivative.assign(problem.n_to, 0.0);
  problem.carrier.assign(problem.n_to, Complex(1.0, 0.0));
  problem.phi_terms.assign(problem.n_omega, Complex{});
  problem.prefactor_terms.assign(problem.n_omega, Complex(1.0, 0.0));
  problem.tableau_matrix = {0.0};
  problem.tableau_nodes = {0.0};
  problem.solution_weights = {1.0};
  problem.error_weights = {0.25};
  problem.synthesis = 2.0;
  problem.projection = 0.5;
  problem.kerr_coefficient = 0.125;
  problem.self_factor = 0.5;
  problem.conjugate_factor = 0.25;
  problem.harmonic_factor = 0.0;
  problem.cubic_real = family == native::FieldFamily::field_real;
  problem.take_real = family != native::FieldFamily::envelope;
  return problem;
}

native::TrialScalars trial(const native::Problem& problem, const double taper = 1.0) {
  return {
      std::vector<double>((problem.n_stages + 1) * problem.n_phi_terms, 0.0),
      std::vector<double>(problem.n_stages * problem.n_phi_terms, 0.0),
      std::vector<double>(problem.n_stages * problem.n_prefactor_terms, 1.0),
      std::vector<double>(problem.n_stages, 1.0),
      std::vector<double>(problem.n_stages, taper),
      {},
      std::vector<double>(problem.n_stages * problem.n_raman_terms, 0.0),
      {},
      {},
      {},
      {},
      0,
      {},
      {},
      {},
  };
}

native::Problem multimode_problem(const std::size_t node_chunk = 2) {
  auto problem = base_problem(native::FieldFamily::field_real);
  problem.n_modes = 2;
  problem.n_pol_state = 1;
  problem.n_pol_transform = 1;
  problem.n_nodes = 2;
  problem.node_chunk = node_chunk;
  problem.nonlinear = true;
  problem.synthesis_matrix = {1.0, 0.5, -0.25, 1.5};
  problem.projection_matrix = {0.75, -0.5, 0.25, 1.25};
  problem.projection_coarse_matrix = {0.5, -0.25, 0.125, 1.0};
  problem.phi_terms.assign(problem.n_modes * problem.n_omega, Complex{});
  problem.prefactor_terms.assign(problem.n_modes * problem.n_omega, Complex(1.0, 0.0));
  return problem;
}

native::Problem general_problem(const native::FieldFamily family) {
  auto problem = base_problem(family);
  problem.nonlinear = true;
  problem.synthesis_matrix = {problem.synthesis};
  problem.projection_matrix = {problem.projection};
  return problem;
}

native::Problem plasma_problem() {
  auto problem = general_problem(native::FieldFamily::field_real);
  problem.nonlinear = false;
  problem.plasma = true;
  problem.stash_plasma = true;
  problem.dt_fine = 0.125;
  problem.charge_mass_ratio = 2.0;
  problem.ionisation_potential = 3.0;
  problem.log_field_min = std::log(0.01);
  problem.d_log_field = std::log(1000.0);
  problem.field_min = 0.01;
  problem.field_max = 10.0;
  problem.rate_max = 0.5;
  problem.rate_intervals = 1;
  problem.rate_coefficients = {0.0, 0.0, 0.0, std::log(0.5)};
  return problem;
}

native::Problem raman_problem(const native::FieldFamily family) {
  auto problem = general_problem(family);
  problem.nonlinear = false;
  problem.raman = true;
  problem.raman_thg = family == native::FieldFamily::field_real;
  problem.n_raman_terms = 2;
  problem.raman_tensor_weights = {1.0, 0.0, 1.0, 0.0};
  problem.raman_fraction = 0.4;
  problem.raman_epsilon0 = 2.0;
  problem.dt_fine = 0.125;
  problem.raman_undamped.assign(problem.n_raman_terms * 2 * problem.n_to, 0.0);
  problem.raman_undamped[1] = 1.0;
  problem.raman_undamped[2] = 0.5;
  problem.raman_undamped[3] = -0.25;
  const auto second = 2 * problem.n_to;
  problem.raman_undamped[second + 1] = 0.2;
  problem.raman_undamped[second + 2] = 0.1;
  problem.raman_undamped[second + 3] = 0.05;
  return problem;
}

std::vector<Complex> raman_reference(const native::Problem& problem,
                                     const std::vector<Complex>& spectrum,
                                     const std::vector<double>& rates) {
  std::vector<Complex> modal;
  if (problem.family == native::FieldFamily::field_real) {
    for (const double value : real_time(spectrum, problem.n_to))
      modal.emplace_back(value, 0.0);
  } else if (problem.family == native::FieldFamily::field_analytic) {
    auto analytic = std::vector<Complex>(problem.n_to);
    analytic[0] = spectrum[0];
    for (std::size_t index = 1; index < spectrum.size(); ++index)
      analytic[index] = 2.0 * spectrum[index];
    modal = envelope_time(analytic, problem.n_to);
  } else {
    modal = envelope_time(spectrum, problem.n_to);
  }

  const auto doubled = 2 * problem.n_to;
  std::vector<double> kernel(doubled);
  for (std::size_t time = 0; time < doubled; ++time) {
    double total = 0.0;
    const double delay = static_cast<double>(time) * problem.dt_fine;
    for (std::size_t term = 0; term < problem.n_raman_terms; ++term)
      total += problem.raman_undamped[term * doubled + time] *
               std::exp(-rates[term] * delay);
    kernel[time] = problem.raman_fraction * total;
  }

  std::vector<Complex> projected(problem.n_to);
  for (std::size_t node = 0; node < problem.n_nodes; ++node) {
    const double synthesis = problem.synthesis_matrix[node];
    const double projection = problem.projection_matrix[node];
    std::vector<Complex> field(problem.n_to);
    std::vector<double> drive(problem.n_to);
    for (std::size_t time = 0; time < problem.n_to; ++time) {
      field[time] = synthesis * modal[time];
      drive[time] = problem.raman_thg ? field[time].real() * field[time].real()
                                      : 0.5 * (field[time].real() * field[time].real() +
                                               field[time].imag() * field[time].imag());
    }
    for (std::size_t time = 0; time < problem.n_to; ++time) {
      double convolved = 0.0;
      for (std::size_t previous = 0; previous <= time; ++previous)
        convolved += drive[previous] * kernel[time - previous];
      convolved *= problem.dt_fine;
      const Complex multiplier = problem.family == native::FieldFamily::envelope
                                     ? field[time]
                                     : Complex(field[time].real(), 0.0);
      projected[time] += projection * problem.raman_epsilon0 * multiplier * convolved *
                         problem.time_window[time];
    }
  }
  const auto transformed = dft(projected);
  return {transformed.begin(),
          transformed.begin() + static_cast<std::ptrdiff_t>(problem.n_omega)};
}

std::vector<Complex> multimode_reference(const native::Problem& problem,
                                         const std::vector<Complex>& spectrum,
                                         const bool coarse = false) {
  std::vector<std::vector<double>> modal(problem.n_modes);
  for (std::size_t mode = 0; mode < problem.n_modes; ++mode) {
    const auto first =
        spectrum.begin() + static_cast<std::ptrdiff_t>(mode * problem.n_omega);
    const std::vector<Complex> one(
        first, first + static_cast<std::ptrdiff_t>(problem.n_omega));
    modal[mode] = real_time(one, problem.n_to);
  }
  std::vector<std::vector<double>> polarisation(problem.n_nodes,
                                                std::vector<double>(problem.n_to));
  for (std::size_t node = 0; node < problem.n_nodes; ++node) {
    for (std::size_t time = 0; time < problem.n_to; ++time) {
      double value = 0.0;
      for (std::size_t mode = 0; mode < problem.n_modes; ++mode)
        value +=
            problem.synthesis_matrix[node * problem.n_modes + mode] * modal[mode][time];
      polarisation[node][time] = problem.kerr_coefficient * value * value * value;
    }
  }
  std::vector<Complex> result(problem.n_modes * problem.n_omega);
  const auto& projection =
      coarse ? problem.projection_coarse_matrix : problem.projection_matrix;
  for (std::size_t mode = 0; mode < problem.n_modes; ++mode) {
    std::vector<Complex> projected(problem.n_to);
    for (std::size_t time = 0; time < problem.n_to; ++time) {
      for (std::size_t node = 0; node < problem.n_nodes; ++node)
        projected[time] +=
            projection[mode * problem.n_nodes + node] * polarisation[node][time];
    }
    const auto transformed = dft(projected);
    for (std::size_t index = 0; index < problem.n_omega; ++index)
      result[mode * problem.n_omega + index] = transformed[index];
  }
  return result;
}

std::vector<Complex> nonlinear_reference(const native::Problem& problem,
                                         const std::vector<Complex>& spectrum,
                                         const double taper) {
  std::vector<Complex> time;
  if (problem.family == native::FieldFamily::envelope) {
    time = envelope_time(spectrum, problem.n_to);
  } else if (problem.family == native::FieldFamily::field_real) {
    const auto real = real_time(spectrum, problem.n_to);
    for (const auto value : real)
      time.emplace_back(value, 0.0);
  } else {
    std::vector<Complex> analytic(problem.n_to);
    analytic[0] = spectrum[0];
    for (std::size_t index = 1; index < spectrum.size(); ++index) {
      analytic[index] = 2.0 * spectrum[index];
    }
    time = envelope_time(analytic, problem.n_to);
  }
  std::vector<Complex> projected(problem.n_to);
  for (std::size_t index = 0; index < problem.n_to; ++index) {
    const Complex node = problem.synthesis * taper * time[index];
    Complex polarisation;
    if (problem.family == native::FieldFamily::field_real) {
      polarisation = problem.kerr_coefficient * node.real() * node.real() *
                     Complex(node.real(), 0.0);
    } else {
      const double modulus = node.real() * node.real() + node.imag() * node.imag();
      polarisation = problem.kerr_coefficient *
                     (problem.self_factor + problem.conjugate_factor) * modulus * node;
    }
    polarisation *= problem.projection / taper;
    projected[index] =
        problem.take_real ? Complex(polarisation.real(), 0.0) : polarisation;
  }
  const auto full = dft(projected);
  return {full.begin(), full.begin() + static_cast<std::ptrdiff_t>(problem.n_omega)};
}

void expect_complex_near(const Complex actual, const Complex expected,
                         const double tolerance) {
  EXPECT_NEAR(actual.real(), expected.real(), tolerance);
  EXPECT_NEAR(actual.imag(), expected.imag(), tolerance);
}

TEST(Fft, ImpulseProducesConstantSpectrum) {
  constexpr std::size_t size = 32;
  native::ComplexView input("native_test.fft.input", size);
  native::ComplexView output("native_test.fft.output", size);
  Kokkos::deep_copy(input, native::Complex{});
  const auto host_input = Kokkos::create_mirror_view(input);
  host_input(0) = native::Complex(2.0, -3.0);
  Kokkos::deep_copy(input, host_input);

  const native::ExecutionSpace execution_space;
  native::fft(execution_space, input, output);
  execution_space.fence("native FFT test boundary");
  const auto host_output =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);

  const double tolerance = 64.0 * kEpsilon * std::log2(size) * std::hypot(2.0, 3.0);
  for (std::size_t index = 0; index < size; ++index) {
    expect_complex_near(host_output(index), {2.0, -3.0}, tolerance);
  }
}

TEST(Fft, RejectsInvalidExtents) {
  const native::ExecutionSpace execution_space;
  const native::ComplexView empty("native_test.fft.empty", 0);
  EXPECT_THROW(native::fft(execution_space, empty, empty), std::invalid_argument);
  const native::ComplexView input("native_test.fft.input", 4);
  const native::ComplexView output("native_test.fft.output", 3);
  EXPECT_THROW(native::fft(execution_space, input, output), std::invalid_argument);
}

TEST(MultiplyPhase, AppliesQuarterTurnAndPreservesMagnitude) {
  constexpr std::size_t size = 4;
  native::ComplexView values("native_test.phase.values", size);
  native::RealView phase("native_test.phase.angles", size);
  native::ComplexView output("native_test.phase.output", size);
  Kokkos::deep_copy(values, native::Complex(3.0, 4.0));
  Kokkos::deep_copy(phase, std::numbers::pi_v<double> / 2.0);
  const native::ExecutionSpace execution_space;
  native::multiply_phase(execution_space, values, phase, output);
  execution_space.fence("native phase test boundary");
  const auto host_output =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  const double tolerance = 32.0 * kEpsilon;
  for (std::size_t index = 0; index < size; ++index) {
    expect_complex_near(host_output(index), {-4.0, 3.0}, tolerance);
    EXPECT_NEAR(Kokkos::abs(host_output(index)), 5.0, tolerance);
  }
}

TEST(State, MasksAndAppliesCombinedWholeStepPropagator) {
  auto problem = base_problem(native::FieldFamily::envelope);
  problem.n_phi_terms = 2;
  problem.phi_terms.clear();
  for (std::size_t term = 0; term < 2; ++term) {
    for (std::size_t index = 0; index < problem.n_omega; ++index) {
      problem.phi_terms.emplace_back(0.0, static_cast<double>(term + 1) *
                                              static_cast<double>(index + 1));
    }
  }
  problem.nonlinear = false;
  problem.active_mask[2] = 0U;
  auto scalars = trial(problem);
  scalars.phi_delta = {0.1, 0.2, 0.3, 0.4};
  const std::vector<Complex> initial(problem.n_omega, {1.0, -0.5});
  native::State state(problem);
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

TEST(State, KerrFamiliesMatchStraightforwardHostTransforms) {
  for (const auto family :
       {native::FieldFamily::field_real, native::FieldFamily::field_analytic,
        native::FieldFamily::envelope}) {
    SCOPED_TRACE(static_cast<int>(family));
    auto problem = base_problem(family);
    problem.nonlinear = true;
    std::vector<Complex> initial(problem.n_omega);
    initial[0] = Complex(1.0, 0.0);
    initial[1] = Complex(0.25, -0.125);
    initial[2] = Complex(-0.1, 0.05);
    if (problem.n_omega == 4)
      initial[3] = Complex(0.05, 0.025);
    constexpr double taper = 1.5;
    constexpr double dz = 0.02;
    const auto derivative = nonlinear_reference(problem, initial, taper);
    native::State state(problem);
    state.set_initial(initial, {});
    const auto partials = state.trial_step(dz, trial(problem, taper));
    state.finish(dz);
    const auto result = state.field_to_host();
    double magnitude = 1.0;
    for (std::size_t index = 0; index < result.size(); ++index) {
      const Complex expected = initial[index] + dz * derivative[index];
      magnitude = std::max(magnitude, Kokkos::abs(expected));
      const double tolerance =
          512.0 * kEpsilon * static_cast<double>(problem.n_to) * magnitude;
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
  EXPECT_THROW(static_cast<void>(state.stage_to_host(problem.n_stages)),
               std::out_of_range);
  ASSERT_EQ(stage.size(), derivative.size());
  for (std::size_t index = 0; index < stage.size(); ++index) {
    const double tolerance =
        256.0 * kEpsilon * std::max(1.0, Kokkos::abs(derivative[index]));
    expect_complex_near(stage[index], derivative[index], tolerance);
  }
  state.finish(dz);
  const auto result = state.field_to_host();
  ASSERT_EQ(partials.size(), 3 * problem.n_modes);
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
  problem.first_same_as_last = false;
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
              512.0 * static_cast<double>(problem.n_to) * kEpsilon);
}

TEST(GeneralState, WeightedResidualDividesEachBinBeforeReducing) {
  auto problem = multimode_problem();
  problem.weighted_controller = true;
  const std::vector<Complex> initial{{1.0, 0.0}, {0.25, -0.125}, {-0.1, 0.05},
                                     {0.5, 0.0}, {-0.2, 0.1},    {0.05, -0.025}};
  native::GeneralState state(problem);
  state.set_initial(initial, {});
  const auto ordinary = state.trial_step(0.01, trial(problem));
  const auto weighted = state.weighted_error(std::vector<double>(problem.n_omega, 2.0));
  ASSERT_EQ(weighted.size(), problem.n_columns());
  for (std::size_t column = 0; column < problem.n_columns(); ++column)
    EXPECT_NEAR(weighted[column], ordinary[2 * problem.n_columns() + column] / 4.0,
                32.0 * kEpsilon);
}

TEST(GeneralState, PackedStatisticsMatchStraightforwardHostReductions) {
  auto problem = multimode_problem();
  problem.family = native::FieldFamily::envelope;
  problem.n_t = 4;
  problem.n_to = 4;
  problem.n_omega = 4;
  problem.crop_index = {0, 1, 2, 3};
  problem.active_mask.assign(4, 1U);
  problem.norm_mask.assign(4, 1U);
  problem.omega_window.assign(4, 1.0);
  problem.omega = {1.0, 2.0, 3.0, 4.0};
  problem.time_window.assign(4, 1.0);
  problem.carrier.assign(4, Complex(1.0, 0.0));
  problem.phi_terms.assign(8, Complex{});
  problem.prefactor_terms.assign(8, Complex(1.0, 0.0));
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
      weighted += problem.omega[index] * one;
    }
    EXPECT_NEAR(values[mode], weighted / power, 32.0 * kEpsilon);
    EXPECT_NEAR(values[2 + mode], problem.dt / 4.0 * power, 32.0 * kEpsilon);
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
    std::vector<Complex> initial(problem.n_omega);
    initial[0] = Complex(1.0, 0.0);
    initial[1] = Complex(0.25, -0.125);
    initial[2] = Complex(-0.1, 0.05);
    if (problem.n_omega == 4)
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
      const double tolerance = 512.0 * kEpsilon * static_cast<double>(problem.n_to) *
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
    std::vector<Complex> initial(problem.n_omega);
    initial[0] = Complex(1.0, 0.0);
    initial[1] = Complex(0.25, -0.125);
    initial[2] = Complex(-0.1, 0.05);
    if (problem.n_omega == 4)
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
                               std::log2(2.0 * static_cast<double>(problem.n_to)) *
                               std::max(1.0, Kokkos::abs(expected));
      expect_complex_near(result[index], expected, tolerance);
    }
  }
}

TEST(GeneralState, RamanMatchesDirectConvolutionForAShortFinalNodeChunk) {
  auto problem = raman_problem(native::FieldFamily::field_real);
  problem.n_nodes = 3;
  problem.node_chunk = 2;
  problem.synthesis_matrix = {1.0, 0.75, -0.5};
  problem.projection_matrix = {0.5, 0.25, -0.75};
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
                             std::log2(2.0 * static_cast<double>(problem.n_to)) *
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
    const auto offset = static_cast<std::ptrdiff_t>(pol * problem.n_omega);
    std::vector<Complex> input(spectrum.begin() + offset,
                               spectrum.begin() + offset +
                                   static_cast<std::ptrdiff_t>(problem.n_omega));
    if (problem.family == native::FieldFamily::field_real) {
      for (const double value : real_time(input, problem.n_to))
        modal[pol].emplace_back(value, 0.0);
    } else {
      if (problem.family == native::FieldFamily::field_analytic) {
        for (std::size_t j = 1; j < input.size(); ++j)
          input[j] *= 2.0;
      }
      modal[pol] = envelope_time(input, problem.n_to);
    }
  }
  std::array<std::vector<Complex>, 2> projected{std::vector<Complex>(problem.n_to),
                                                std::vector<Complex>(problem.n_to)};
  for (std::size_t node = 0; node < problem.n_nodes; ++node) {
    const double synthesis = problem.synthesis_matrix[node];
    for (std::size_t time = 0; time < problem.n_to; ++time) {
      std::array<Complex, 2> field{synthesis * modal[0][time],
                                   synthesis * modal[1][time]};
      if (problem.family != native::FieldFamily::envelope)
        for (auto& value : field)
          value = Complex(value.real(), 0.0);
      for (std::size_t previous = 0; previous <= time; ++previous) {
        const Complex x = synthesis * modal[0][previous];
        const Complex y = synthesis * modal[1][previous];
        const double half = problem.raman_thg ? 1.0 : 0.5;
        const double xx = half * (x.real() * x.real() + x.imag() * x.imag());
        const double yy = half * (y.real() * y.real() + y.imag() * y.imag());
        const double xy = half * (x.real() * y.real() + x.imag() * y.imag());
        const std::array<std::array<double, 2>, 2> q{{{xx, xy}, {xy, yy}}};
        const auto lag = time - previous;
        const double delay = static_cast<double>(lag) * problem.dt_fine;
        double hv = problem.raman_undamped[2 * problem.n_to + lag] *
                    std::exp(-rates[1] * delay);
        double hr = problem.raman_undamped[lag] * std::exp(-rates[0] * delay);
        if (isotropic_only) {
          hv += hr;
          hr = 0.0;
        }
        for (std::size_t pol = 0; pol < 2; ++pol) {
          Complex value = hv * (xx + yy) * field[pol];
          for (std::size_t other = 0; other < 2; ++other)
            value += 0.5 * hr * (3.0 * q[pol][other] - (pol == other ? xx + yy : 0.0)) *
                     field[other];
          projected[pol][time] += problem.projection_matrix[node] *
                                  problem.raman_epsilon0 * problem.raman_fraction *
                                  problem.dt_fine * problem.time_window[time] * value;
        }
      }
    }
  }
  std::vector<Complex> derivative;
  for (const auto& component : projected) {
    const auto transformed = dft(component);
    derivative.insert(derivative.end(), transformed.begin(),
                      transformed.begin() +
                          static_cast<std::ptrdiff_t>(problem.n_omega));
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
        const auto old_count = problem.n_to;
        const auto old_rows = problem.raman_undamped;
        problem.n_to = count;
        problem.time_window.assign(count, 1.0);
        problem.carrier.assign(count, Complex(1.0, 0.0));
        problem.raman_undamped.assign(problem.n_raman_terms * 2 * count, 0.0);
        for (std::size_t term = 0; term < problem.n_raman_terms; ++term)
          for (std::size_t sample = 0; sample < old_count; ++sample)
            problem.raman_undamped[term * 2 * count + sample] =
                old_rows[term * 2 * old_count + sample];
        problem.n_pol_state = 2;
        problem.raman_tensor_capable = true;
        problem.raman_tensor_weights = isotropic_only
                                           ? std::vector<double>{1.0, 0.0, 1.0, 0.0}
                                           : std::vector<double>{0.25, 0.75, 1.0, 0.0};
        problem.n_nodes = 3;
        problem.node_chunk = 2;
        problem.synthesis_matrix = {1.0, 0.75, -0.5};
        problem.projection_matrix = {0.5, 0.25, -0.75};
        problem.phi_terms.assign(2 * problem.n_omega, Complex{});
        problem.prefactor_terms.assign(2 * problem.n_omega, Complex(1.0, 0.0));
        const std::vector<double> rates{0.125, 0.25};
        std::vector<Complex> initial(2 * problem.n_omega);
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
            (3.0 * 512.0 * std::log2(2.0 * static_cast<double>(problem.n_to)) +
             64.0 * static_cast<double>(problem.n_to * problem.n_nodes)) *
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
        EXPECT_EQ(ledger.at("node_raman_drive"), problem.node_chunk *
                                                     problem.raman_drives() * 2 *
                                                     problem.n_to * sizeof(double));
        EXPECT_EQ(ledger.at("raman_tensor_weights"),
                  problem.n_raman_terms * 2 * sizeof(double));
      }
    }
  }
}

TEST(GeneralState, RamanTensorWeightsAreValidatedBeforeAllocation) {
  for (const auto& weights :
       {std::vector<double>{1.0}, std::vector<double>{1.0, 0.5, 1.0, 0.0},
        std::vector<double>{1.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0}}) {
    auto problem = raman_problem(native::FieldFamily::envelope);
    problem.raman_tensor_weights = weights;
    EXPECT_THROW(static_cast<void>(native::GeneralState(std::move(problem))),
                 std::invalid_argument);
  }
}

TEST(GeneralState, RamanValidationPrecedesFieldAllocation) {
  auto vector = raman_problem(native::FieldFamily::field_real);
  vector.n_pol_state = 2;
  vector.phi_terms.assign(2 * vector.n_omega, Complex{});
  vector.prefactor_terms.assign(2 * vector.n_omega, Complex(1.0, 0.0));
  EXPECT_THROW(static_cast<void>(native::GeneralState(std::move(vector))),
               std::invalid_argument);

  auto wrong = raman_problem(native::FieldFamily::field_real);
  wrong.raman_undamped.pop_back();
  EXPECT_THROW(static_cast<void>(native::GeneralState(std::move(wrong))),
               std::invalid_argument);

  auto valid = raman_problem(native::FieldFamily::field_real);
  native::GeneralState state(valid);
  state.set_initial(std::vector<Complex>(valid.n_omega), {});
  auto rates = trial(valid);
  rates.raman_rates = {-1.0, 0.0};
  EXPECT_THROW(static_cast<void>(state.trial_step(0.01, rates)), std::invalid_argument);
}

TEST(GeneralState, TwoStatePolarisationsUseTheSharedVectorModulus) {
  auto problem = general_problem(native::FieldFamily::field_real);
  problem.n_pol_state = 2;
  problem.phi_terms.assign(2 * problem.n_omega, Complex{});
  problem.prefactor_terms.assign(2 * problem.n_omega, Complex(1.0, 0.0));
  const std::vector<Complex> initial{{1.0, 0.0}, {0.25, -0.125}, {-0.1, 0.05},
                                     {0.5, 0.0}, {-0.2, 0.1},    {0.05, -0.025}};
  std::array<std::vector<double>, 2> time{
      real_time(std::vector<Complex>(initial.begin(), initial.begin() + 3),
                problem.n_to),
      real_time(std::vector<Complex>(initial.begin() + 3, initial.end()),
                problem.n_to)};
  std::array<std::vector<Complex>, 2> derivative;
  for (std::size_t pol = 0; pol < 2; ++pol) {
    std::vector<Complex> response(problem.n_to);
    for (std::size_t sample = 0; sample < problem.n_to; ++sample) {
      const double first = problem.synthesis * time[0][sample];
      const double second = problem.synthesis * time[1][sample];
      const double modulus = first * first + second * second;
      response[sample] = Complex(problem.projection * problem.kerr_coefficient *
                                     modulus * (pol == 0 ? first : second),
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
    for (std::size_t index = 0; index < problem.n_omega; ++index) {
      const auto flat = pol * problem.n_omega + index;
      const Complex expected = initial[flat] + dz * derivative[pol][index];
      expect_complex_near(result[flat], expected,
                          1024.0 * static_cast<double>(problem.n_to) * kEpsilon);
    }
  }
}

TEST(GeneralState, HeldCurrentMatchesTheDiscreteSinusoidalDrift) {
  auto problem = plasma_problem();
  problem.stash_plasma = false;
  const double theta =
      2.0 * std::numbers::pi_v<double> / static_cast<double>(problem.n_to);
  problem.omega = {0.0, theta / problem.dt_fine, 2.0 * theta / problem.dt_fine};
  problem.active_mask[0] = 0;
  problem.field_min = 1e6;
  problem.field_max = 1e7;
  problem.log_field_min = std::log(problem.field_min);
  problem.d_log_field = std::log(10.0);
  constexpr double population = 0.25;
  // The unit mode at bin one produces E_n = 2*synthesis/N*cos(n theta).
  // The exact finite trapezoid sum of cos is dt*cot(theta/2)*sin(n theta)/2.
  const double amplitude = 2.0 * problem.synthesis / static_cast<double>(problem.n_to);
  const double current_scale = population * problem.charge_mass_ratio * amplitude *
                               problem.dt_fine / (2.0 * std::tan(theta / 2.0));
  std::vector<Complex> product(problem.n_to), edge(problem.n_to);
  double operand = 0.0;
  for (std::size_t time = 0; time < problem.n_to; ++time) {
    const double angle = theta * static_cast<double>(time);
    problem.time_window[time] = 0.5 * (1.0 - std::cos(angle));
    problem.time_window_derivative[time] =
        0.5 * theta / problem.dt_fine * std::sin(angle);
    const double current = current_scale * std::sin(angle);
    product[time] = problem.projection * problem.time_window[time] * current;
    edge[time] = problem.projection * problem.time_window_derivative[time] * current;
    operand += Kokkos::abs(product[time]) / problem.omega[1] +
               Kokkos::abs(edge[time]) / (problem.omega[1] * problem.omega[1]);
  }
  // Independent trigonometric sums, N trapezoids and three transforms. The bound
  // uses both absolute operands before their cancellation, fixed before execution.
  const double tolerance =
      512.0 * static_cast<double>(problem.n_to) * kEpsilon * operand;
  ASSERT_GT(tolerance, 0.0);
  const auto x = dft(product), z = dft(edge);
  native::GeneralState state(problem);
  auto scalars = trial(problem);
  scalars.preionisation = {population};
  state.set_initial(std::vector<Complex>{{}, {1.0, 0.0}, {}}, {});
  static_cast<void>(state.trial_step(0.0, scalars));
  const auto actual = state.stage_to_host(0);
  EXPECT_EQ(actual[0], Complex{});
  for (std::size_t bin = 1; bin < problem.n_omega; ++bin) {
    const double omega = problem.omega[bin];
    const Complex expected =
        Complex(0.0, -1.0 / omega) * x[bin] + z[bin] / (omega * omega);
    expect_complex_near(actual[bin], expected, tolerance);
  }
  state.set_initial(std::vector<Complex>(problem.n_omega), {});
  static_cast<void>(state.trial_step(0.0, scalars));
  for (const auto value : state.stage_to_host(0))
    EXPECT_EQ(value, Complex{});
}

TEST(GeneralState, HeldCurrentRefusesMissingDerivativeAndNonpositiveActiveFrequency) {
  auto problem = plasma_problem();
  problem.time_window_derivative.clear();
  EXPECT_THROW(native::GeneralState state(problem), std::invalid_argument);
  problem = plasma_problem();
  problem.time_window_derivative[0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(native::GeneralState state(problem), std::invalid_argument);
  problem = plasma_problem();
  problem.omega[0] = 0.0;
  EXPECT_THROW(native::GeneralState state(problem), std::invalid_argument);
  problem.active_mask[0] = 0;
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
  const double exposure =
      problem.rate_max * problem.dt_fine * static_cast<double>(problem.n_to - 1);
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
  EXPECT_NEAR(retained[1], problem.rate_max, 32.0 * kEpsilon);
  state.clear_fsal();
  const auto rescanned = state.statistics(flags, {}, {}, density, 1.0, 0.7, 1.0);
  EXPECT_NEAR(rescanned[0], density * final_fraction(0.7), 64.0 * kEpsilon);
  state.set_initial(initial, {});
  const auto reset = state.statistics(flags, {}, {}, density, 1.0, 0.2, 1.0);
  EXPECT_NEAR(reset[0], density * final_fraction(0.2), 64.0 * kEpsilon);
}

TEST(GeneralState, IndependentChannelStashesSurviveRejectedTrials) {
  auto problem = plasma_problem();
  problem.n_channels = 2;
  const std::array<double, 2> rates{0.125, 0.25};
  for (std::size_t channel = 0; channel < 2; ++channel) {
    const std::array<double, 9> parameters{problem.log_field_min,
                                           problem.d_log_field,
                                           problem.field_min,
                                           problem.field_max,
                                           rates[channel],
                                           1.0,
                                           4.0 * static_cast<double>(channel),
                                           problem.charge_mass_ratio,
                                           problem.ionisation_potential};
    problem.channel_parameters.insert(problem.channel_parameters.end(),
                                      parameters.begin(), parameters.end());
    const std::array<double, 4> coefficients{0.0, 0.0, 0.0, std::log(rates[channel])};
    problem.channel_coefficients.insert(problem.channel_coefficients.end(),
                                        coefficients.begin(), coefficients.end());
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
    return 1.0 - (1.0 - initial) * std::exp(-rates[channel] * problem.dt_fine *
                                            static_cast<double>(problem.n_to - 1));
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
  problem.weighted_controller = true;
  problem.plasma = true;
  problem.stash_plasma = true;
  problem.dt_fine = 0.125;
  problem.charge_mass_ratio = 2.0;
  problem.ionisation_potential = 3.0;
  problem.log_field_min = std::log(0.01);
  problem.d_log_field = std::log(1000.0);
  problem.field_min = 0.01;
  problem.field_max = 10.0;
  problem.rate_max = 0.5;
  problem.rate_intervals = 1;
  problem.rate_coefficients = {0.0, 0.0, 0.0, std::log(0.5)};
  native::GeneralState state(problem);
  const auto ledger = state.memory_ledger();
  const std::size_t real = sizeof(double);
  const std::size_t complex = sizeof(Complex);
  EXPECT_EQ(ledger.at("state"), problem.n_columns() * problem.n_omega * complex);
  EXPECT_EQ(ledger.at("stages"),
            problem.n_stages * problem.n_columns() * problem.n_omega * complex);
  EXPECT_EQ(ledger.at("transform_matrices"),
            (problem.synthesis_matrix.size() + problem.projection_matrix.size() +
             problem.projection_coarse_matrix.size()) *
                real);
  EXPECT_EQ(ledger.at("stacked_projection"),
            2 * problem.n_columns() * problem.n_to * real);
  EXPECT_EQ(ledger.at("scaled_transform_matrices"),
            5 * problem.n_modes * problem.n_nodes * real);
  EXPECT_EQ(ledger.at("plasma_stash"),
            2 * problem.n_nodes * (1 + 2 * problem.n_channels) * real);
  // The device column's rows; on the host the band statistics' three rows per team
  // slot, at most one slot per thread (a trapezoid problem keeps no band rows).
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const auto chunk_rows = problem.node_chunk * problem.effective_node_pol();
  // With the trapezoid's binary64 tile totals, a row's tiles of 1,024 samples.
  const auto tiles = (problem.n_to + 1023) / 1024;
  EXPECT_EQ(ledger.at("plasma_device_scan"),
            (3 * problem.node_chunk * problem.n_to + 3 * chunk_rows * problem.n_to +
             chunk_rows * tiles) *
                real);
#else
  // One slot per thread of the response operation's own width (problem-size policy).
  const auto response_elements =
      problem.node_chunk * problem.effective_node_pol() * problem.n_to;
  const int width =
      native::cpu_size_policy_enabled
          ? native::operation_threads(native::thread_policy(), native::CpuPool::kernels,
                                      native::CpuOperation::response, response_elements,
                                      problem.n_to)
          : native::ExecutionSpace().concurrency();
  const auto slots =
      std::min(problem.node_chunk, static_cast<std::size_t>(std::max(1, width)));
  EXPECT_EQ(ledger.at("plasma_device_scan"), 3 * slots * problem.n_to * real);
#endif
  EXPECT_EQ(ledger.at("weighted_control"),
            (problem.n_omega + problem.n_columns()) * real);
  EXPECT_EQ(ledger.at("error_partials"), 3 * problem.n_columns() * real);
  EXPECT_EQ(ledger.at("statistics_spectrum"),
            problem.n_columns() * problem.n_t * complex);
}

TEST(GeneralState, RamanMemoryLedgerMatchesTheConditionalViewFormulae) {
  const auto problem = raman_problem(native::FieldFamily::field_real);
  native::GeneralState state(problem);
  const auto ledger = state.memory_ledger();
  const auto real = sizeof(double);
  const auto complex = sizeof(Complex);
  EXPECT_EQ(ledger.at("node_raman_drive"),
            problem.node_chunk * 2 * problem.n_to * real);
  EXPECT_EQ(ledger.at("node_raman_spectrum"),
            problem.node_chunk * (problem.n_to + 1) * complex);
  EXPECT_EQ(ledger.at("node_raman_convolved"),
            problem.node_chunk * 2 * problem.n_to * real);
  EXPECT_EQ(ledger.at("raman_undamped"),
            problem.n_raman_terms * 2 * problem.n_to * real);
  EXPECT_EQ(ledger.at("raman_kernel_time"), 2 * problem.n_to * real);
  EXPECT_EQ(ledger.at("raman_kernel_spectrum"), (problem.n_to + 1) * complex);
  // Two slots of the packed stage scalars, the trial's step and its sample.
  EXPECT_EQ(ledger.at("stage_raman_rates"),
            2 * problem.n_stages * problem.n_raman_terms * real);

  native::GeneralState ordinary(multimode_problem());
  const auto absent = ordinary.memory_ledger();
  for (const auto* name :
       {"node_raman_drive", "node_raman_spectrum", "node_raman_convolved",
        "raman_undamped", "raman_kernel_time", "raman_kernel_spectrum",
        "stage_raman_rates"})
    EXPECT_EQ(absent.at(name), 0U);
}

// The host kernel computes KokkosBlas's real GEMM, and both are held to the GEMM
// rounding bound (solprop SEP 25), not to each other's bits: each element of either is
// within `gamma_{k+1} (|beta c| + sum_k |a_ik b_kj|)` of the exact `beta c + a b`
// whatever order or contraction it sums in, so the two differ by at most twice that.
// Rows outside the written subview are a structural contract and stay exact.
// Shapes cover an inner length below, at and past one 16-deep block and several blocks,
// partial row and column tiles, both beta values the engine passes, and row subviews.
TEST(Gemm, NativeOrderKernelIsKokkosBlasWithinTheGemmBound) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // A device build never calls the kernel (`native_order_gemm_enabled` is false).
  GTEST_SKIP() << "the native-order kernel is the host build's";
#endif
  using RealView2 = Kokkos::View<double**, native::Layout, native::MemorySpace>;
  const native::ExecutionSpace execution;
  const std::array<std::array<std::size_t, 3>, 9> shapes{{{33, 3, 2048},
                                                          {3, 33, 2048},
                                                          {6, 33, 257},
                                                          {4, 528, 130},
                                                          {17, 16, 64},
                                                          {5, 17, 63},
                                                          {41, 48, 100},
                                                          {1, 1, 1},
                                                          {9, 2, 5}}};
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  const auto uniform = [&state] {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    return static_cast<double>(state >> 11U) * 0x1.0p-53 - 0.5;
  };
  for (const auto& [rows, inner, columns] : shapes) {
    for (const double beta : {0.0, 1.0}) {
      RealView2 left("left", rows + 2, inner);
      RealView2 right("right", inner, columns);
      RealView2 expected("expected", rows + 2, columns);
      RealView2 actual("actual", rows + 2, columns);
      const auto host_left = Kokkos::create_mirror_view(left);
      const auto host_right = Kokkos::create_mirror_view(right);
      const auto host_start = Kokkos::create_mirror_view(expected);
      for (std::size_t row = 0; row < rows + 2; ++row)
        for (std::size_t index = 0; index < inner; ++index)
          host_left(row, index) = uniform();
      for (std::size_t index = 0; index < inner; ++index)
        for (std::size_t column = 0; column < columns; ++column)
          host_right(index, column) = uniform();
      // On a host build the mirror aliases `expected`, which the GEMM overwrites, so
      // the starting values are kept apart for the bound.
      std::vector<double> start((rows + 2) * columns);
      for (std::size_t row = 0; row < rows + 2; ++row)
        for (std::size_t column = 0; column < columns; ++column) {
          host_start(row, column) = uniform();
          start[row * columns + column] = host_start(row, column);
        }
      Kokkos::deep_copy(left, host_left);
      Kokkos::deep_copy(right, host_right);
      Kokkos::deep_copy(expected, host_start);
      Kokkos::deep_copy(actual, host_start);
      const auto window = std::make_pair(std::size_t{1}, rows + 1);
      const auto a = Kokkos::subview(left, window, Kokkos::ALL());
      native::kokkosblas_gemm(execution, a, right,
                              Kokkos::subview(expected, window, Kokkos::ALL()), beta);
      native::native_order_gemm(execution, a, right,
                                Kokkos::subview(actual, window, Kokkos::ALL()), beta);
      execution.fence("native order GEMM test");
      const auto host_expected =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, expected);
      const auto host_actual =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, actual);
      const double gemm_gamma = test::higham_gamma(inner + 1);
      for (std::size_t row = 0; row < rows + 2; ++row)
        for (std::size_t column = 0; column < columns; ++column) {
          if (row == 0 || row == rows + 1) {
            // Outside the subview: never written, so exactly the start.
            ASSERT_EQ(host_actual(row, column), start[row * columns + column])
                << rows << 'x' << inner << 'x' << columns << " at " << row;
            ASSERT_EQ(host_expected(row, column), start[row * columns + column])
                << rows << 'x' << inner << 'x' << columns << " at " << row;
            continue;
          }
          double magnitude = beta * std::abs(start[row * columns + column]);
          for (std::size_t index = 0; index < inner; ++index)
            magnitude += std::abs(host_left(row, index) * host_right(index, column));
          ASSERT_LE(std::abs(host_actual(row, column) - host_expected(row, column)),
                    2.0 * gemm_gamma * magnitude)
              << rows << 'x' << inner << 'x' << columns << " beta " << beta << " at "
              << row << ',' << column;
        }
    }
  }
  RealView2 wrong("wrong", 2, 3);
  EXPECT_THROW(native::native_order_gemm(execution, wrong, wrong, wrong, 0.0),
               std::invalid_argument);
}

// The one-mode kernels sum each element over the nodes in order from zero, which a
// GEMM may do in any order: both lie within gamma_{k+1} sum |a| |b| of the exact
// product, so they differ by at most twice that. A synthesis element is one product.
TEST(Gemm, OneModeKernelsAreKokkosBlasWithinTheGemmBound) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  GTEST_SKIP() << "the one-mode kernels are the host vendor-BLAS build's";
#endif
  using RealView2 = Kokkos::View<double**, native::Layout, native::MemorySpace>;
  std::uint64_t state = 0x2545f4914f6cdd1dULL;
  const auto uniform = [&state] {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    return static_cast<double>(state >> 11U) * 0x1.0p-53 - 0.5;
  };
  const auto fill = [&uniform](const RealView2& view) {
    const auto host = Kokkos::create_mirror_view(view);
    for (std::size_t row = 0; row < view.extent(0); ++row)
      for (std::size_t column = 0; column < view.extent(1); ++column)
        host(row, column) = uniform();
    Kokkos::deep_copy(view, host);
  };
  const auto host = [](const RealView2& view) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  };
  for (const int threads : {1, 3, native::ExecutionSpace().concurrency()})
    for (const auto& [rows, inner, columns] : std::array<std::array<std::size_t, 3>, 5>{
             {{1, 33, 6400}, {2, 33, 6400}, {2, 33, 257}, {1, 2, 7}, {2, 5, 1}}}) {
      SCOPED_TRACE(std::to_string(threads) + " threads, " + std::to_string(rows) + 'x' +
                   std::to_string(inner) + 'x' + std::to_string(columns));
      const auto execution = native::execution_instance(threads);
      RealView2 weights("weights", rows, inner);
      RealView2 nodes("nodes", inner, columns);
      RealView2 expected("expected", rows, columns);
      RealView2 fine("fine", 1, columns);
      RealView2 coarse("coarse", 1, columns);
      fill(weights);
      fill(nodes);
      native::kokkosblas_gemm(execution, weights, nodes, expected, 0.0);
      native::one_mode_projection(execution, weights, nodes, fine, coarse);
      execution.fence("one-mode projection test");
      const auto a = host(weights);
      const auto b = host(nodes);
      const auto reference = host(expected);
      const auto first = host(fine);
      const auto second = host(coarse);
      const double gamma = test::higham_gamma(inner + 1);
      for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column) {
          double magnitude = 0.0;
          for (std::size_t node = 0; node < inner; ++node)
            magnitude += std::abs(a(row, node) * b(node, column));
          const double actual = row == 0 ? first(0, column) : second(0, column);
          ASSERT_LE(std::abs(actual - reference(row, column)), 2.0 * gamma * magnitude)
              << row << ',' << column;
        }
      // The synthesis: `inner` node rows from one mode's row.
      RealView2 column_weights("column_weights", inner, 1);
      RealView2 mode("mode", 1, columns);
      RealView2 synthesised("synthesised", inner, columns);
      RealView2 direct("direct", inner, columns);
      fill(column_weights);
      fill(mode);
      native::kokkosblas_gemm(execution, column_weights, mode, direct, 0.0);
      native::one_mode_synthesis(execution, column_weights, mode, synthesised);
      execution.fence("one-mode synthesis test");
      const auto w = host(column_weights);
      const auto m = host(mode);
      const auto out = host(synthesised);
      const auto blas = host(direct);
      for (std::size_t node = 0; node < inner; ++node)
        for (std::size_t column = 0; column < columns; ++column) {
          ASSERT_EQ(out(node, column), w(node, 0) * m(0, column));
          ASSERT_LE(std::abs(out(node, column) - blas(node, column)),
                    2.0 * test::higham_gamma(2) * std::abs(out(node, column)));
        }
    }
  const native::ExecutionSpace execution;
  RealView2 three("three", 3, 4);
  RealView2 nodes("nodes", 4, 8);
  RealView2 row("row", 1, 8);
  EXPECT_THROW(native::one_mode_projection(execution, three, nodes, row, row),
               std::invalid_argument);
  EXPECT_THROW(native::one_mode_projection(execution, nodes, three, row, row),
               std::invalid_argument);
  EXPECT_THROW(native::one_mode_synthesis(execution, three, nodes, row),
               std::invalid_argument);
}

TEST(Gemm, EngineSelectsTheNativeOrderKernelWithoutAVendorBlas) {
  EXPECT_EQ(native::native_order_gemm_enabled,
            native::selected_blas_provider_name() == "kokkosblas-native-order");
}

TEST(GeneralState, ProviderPolicyAndLifecycleAreIntrospectable) {
  native::GeneralState state(multimode_problem());
  const auto compiled = native::compiled_gemm_provider();
  // A host build's hybrid policy sends every eligible transform to the vendor GEMM; a
  // device build keeps its shape guard, which this two-node problem is below.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr bool hybrid_vendor = false;
#else
  constexpr bool hybrid_vendor = true;
#endif
  const auto expected =
      compiled == native::GemmProvider::custom ||
              (compiled == native::GemmProvider::hybrid && !hybrid_vendor)
          ? std::string("custom")
          : std::string(native::selected_blas_provider_name());
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
};

// Five nodes and modes in chunks of three, the tail two. The square map takes the
// retained-spectrum synthesis; with an in-stage absorber, which reads the modal time
// field, it keeps the interleaved real GEMM over the time samples. Both must reach
// BLAS for the full chunk and the tail rather than the scalar custom loop, and agree
// with the whole transform.
void check_radial_analytic_chunks(const bool absorber, const char* expected_region) {
  auto problem = general_problem(native::FieldFamily::field_analytic);
  problem.n_modes = 5;
  problem.n_nodes = 5;
  problem.node_chunk = 3;
  problem.spatial_spectral_weights.assign(problem.n_modes, 1.0);
  problem.spatial_on_axis.assign(problem.n_modes, 1.0);
  problem.phi_terms.assign(problem.n_modes * problem.n_omega, Complex{});
  problem.prefactor_terms.assign(problem.n_modes * problem.n_omega, Complex(1.0, 0.0));
  problem.in_stage_absorber = absorber;
  if (absorber) {
    problem.absorber_fine.resize(problem.n_to);
    for (std::size_t time = 0; time < problem.n_to; ++time)
      problem.absorber_fine[time] = 0.01 * static_cast<double>(time % 3);
  }
  problem.synthesis_matrix.resize(problem.n_nodes * problem.n_modes);
  problem.projection_matrix.resize(problem.n_modes * problem.n_nodes);
  for (std::size_t row = 0; row < problem.n_nodes; ++row) {
    for (std::size_t column = 0; column < problem.n_modes; ++column) {
      const auto index = row * problem.n_modes + column;
      problem.synthesis_matrix[index] =
          row == column ? 1.0 : 0.125 * static_cast<double>(row + 1);
      problem.projection_matrix[index] =
          row == column ? 0.75 : -0.0625 * static_cast<double>(column + 1);
    }
  }
  std::vector<Complex> initial(problem.n_modes * problem.n_omega);
  for (std::size_t index = 0; index < initial.size(); ++index)
    initial[index] = Complex(0.01 * static_cast<double>(index + 1), -0.02);
  auto whole_problem = problem;
  whole_problem.node_chunk = whole_problem.n_nodes;
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
  const bool uses_blas =
      native::compiled_gemm_provider() != native::GemmProvider::custom;
  EXPECT_EQ(selected, uses_blas ? 2 : 0);
  EXPECT_EQ(custom, uses_blas ? 0 : 2);
  // Only the absorber-free map keeps the retained spectra.
  EXPECT_EQ(ledger.at("retained_synthesis") > 0, uses_blas && !absorber);
  chunked.finish(0.01);
  const auto actual = chunked.field_to_host();
  // Both computations have one inverse/forward FFT and two length-five products;
  // carry their rounding through the cubic and the stage's scaled increment.
  const double tolerance =
      512.0 * kEpsilon * static_cast<double>(problem.n_to) +
      4.0 * (static_cast<double>(problem.n_modes) * kEpsilon /
             (1.0 - static_cast<double>(problem.n_modes) * kEpsilon));
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
  problem.n_to = 1024;
  problem.n_nodes = 8;
  problem.time_window.assign(problem.n_to, 1.0);
  problem.carrier.assign(problem.n_to, Complex(1.0, 0.0));
  problem.synthesis_matrix.assign(problem.n_nodes * problem.n_modes, 0.5);
  problem.projection_matrix.assign(problem.n_modes * problem.n_nodes, 0.25);
  problem.projection_coarse_matrix.assign(problem.n_modes * problem.n_nodes, 0.125);

  native::GeneralState state(problem);
  const auto ledger = state.memory_ledger();
  const auto provider = native::compiled_gemm_provider();
  const auto expected = provider == native::GemmProvider::custom
                            ? 0U
                            : (3U + 2U) * problem.n_modes * 3U * sizeof(double);
  EXPECT_EQ(ledger.at("chunk_transform_buffers"), expected);

  if (provider != native::GemmProvider::custom) {
    auto whole_problem = problem;
    whole_problem.node_chunk = whole_problem.n_nodes;
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
    const auto dot_length = static_cast<double>(problem.n_nodes);
    const double dot_gamma = dot_length * kEpsilon / (1.0 - dot_length * kEpsilon);
    const double tolerance =
        512.0 * kEpsilon * std::ceil(std::log2(static_cast<double>(problem.n_to))) +
        4.0 * dot_gamma;
    for (std::size_t index = 0; index < chunked.size(); ++index)
      expect_complex_near(chunked[index], reference[index],
                          tolerance * std::max(1.0, Kokkos::abs(reference[index])));
  }
}

#if defined(SOLPROP_NATIVE_TPL_BLAS_OPENBLAS)
TEST(GeneralState, OpenBlasEnvelopePolicyAllocatesAndExecutesComplexTransforms) {
  auto problem = multimode_problem(8);
  problem.family = native::FieldFamily::envelope;
  problem.n_t = 1024;
  problem.n_to = 1024;
  problem.n_omega = 1024;
  problem.n_nodes = 8;
  problem.crop_index.clear();
  for (std::size_t index = 0; index < problem.n_omega; ++index)
    problem.crop_index.push_back(index);
  problem.active_mask.assign(problem.n_omega, 1U);
  problem.norm_mask.assign(problem.n_omega, 1U);
  problem.omega_window.assign(problem.n_omega, 1.0);
  problem.omega.assign(problem.n_omega, 1.0);
  problem.time_window.assign(problem.n_to, 1.0);
  problem.carrier.assign(problem.n_to, Complex(1.0, 0.0));
  problem.phi_terms.assign(problem.n_modes * problem.n_omega, Complex{});
  problem.prefactor_terms.assign(problem.n_modes * problem.n_omega, Complex(1.0, 0.0));
  problem.synthesis_matrix.assign(problem.n_nodes * problem.n_modes, 0.5);
  problem.projection_matrix.assign(problem.n_modes * problem.n_nodes, 0.25);
  problem.projection_coarse_matrix.assign(problem.n_modes * problem.n_nodes, 0.125);

  native::GeneralState state(problem);
  // Scalar envelopes use a real GEMM over the complex samples' interleaved view.
  EXPECT_EQ(state.synthesis_provider(), "kokkosblas-tpl-interleaved-real");
  EXPECT_EQ(state.projection_provider(), "kokkosblas-tpl-interleaved-real");
  const auto ledger = state.memory_ledger();
  const auto real_matrix_values = 3 * problem.n_modes * problem.n_nodes;
  const auto complex_matrix_values = 5 * problem.n_modes * problem.n_nodes;
  EXPECT_EQ(ledger.at("scaled_transform_matrices"),
            real_matrix_values * sizeof(double) +
                complex_matrix_values * sizeof(Complex));
  EXPECT_EQ(ledger.at("stacked_projection"),
            2 * problem.n_modes * problem.n_to * sizeof(Complex));

  state.set_initial(std::vector<Complex>(problem.n_modes * problem.n_omega), {});
  const auto partials = state.trial_step(0.01, trial(problem));
  // Exact: a zero field has zero norms and a zero error.
  EXPECT_EQ(partials, std::vector<double>(3 * problem.n_modes, 0.0));
}
#endif

TEST(GeneralState, RejectsOverflowingExtentsBeforeAllocation) {
  auto column_overflow = general_problem(native::FieldFamily::field_real);
  column_overflow.n_modes = std::numeric_limits<std::size_t>::max();
  column_overflow.n_pol_state = 2;
  EXPECT_THROW(static_cast<void>(native::GeneralState(std::move(column_overflow))),
               std::length_error);

  auto field_overflow = general_problem(native::FieldFamily::field_real);
  field_overflow.n_modes =
      std::numeric_limits<std::size_t>::max() / field_overflow.n_omega + 1;
  EXPECT_THROW(static_cast<void>(native::GeneralState(std::move(field_overflow))),
               std::length_error);
}

// solprop's hybrid planar kind (kernel schema 15): two slab channels on a three-node x
// rule, times a four-row y axis, field_real Kerr. Columns (p, k_y) are p + 2 iy, nodes
// (x, y) are ix + 3 iy, and each column's operator subtracts its channel's wall term.
native::Problem planar_problem(const std::size_t rows_per_chunk) {
  auto problem = base_problem(native::FieldFamily::field_real);
  problem.n_p = 2;
  problem.n_x = 3;
  problem.n_y = 4;
  problem.n_modes = problem.n_p * problem.n_y;
  problem.n_nodes = problem.n_x * problem.n_y;
  problem.node_chunk = rows_per_chunk * problem.n_x;
  problem.nonlinear = true;
  problem.phi_terms.clear();
  problem.prefactor_terms.clear();
  problem.synthesis_matrix = {1.0, 0.5, -0.25, 1.5, 0.75, -0.5};
  problem.projection_matrix = {0.5, 0.25, -0.125, 0.375, 0.625, -0.25};
  problem.spatial_k_squared = {{40.0, 0.5}, {60.0, 1.0}, {90.0, 1.5}};
  problem.spatial_k_squared_low = {{1e-15, 0.0}, {-2e-15, 0.0}, {0.0, 1e-16}};
  problem.spatial_numerator = {{0.0, -1.0}, {0.0, -2.0}, {0.0, -3.0}};
  problem.spatial_clamped = {{0.0, -0.25}, {0.0, -0.5}, {0.0, -0.75}};
  problem.spatial_frame = {5.0, 6.0, 7.0};
  problem.spatial_spectral_rate = {0.0, 0.25, 0.5};
  // k_y^2 per column, the channel fastest; the last row is evanescent in the first
  // bin, so both branches of the operator are exercised.
  const std::vector<double> ky_squared{0.0, 4.0, 45.0, 4.0};
  for (std::size_t column = 0; column < problem.n_modes; ++column) {
    problem.spatial_kperp_sq.push_back(ky_squared[column / problem.n_p]);
    problem.spatial_kperp_sq_low.push_back(column == 3 ? 1e-16 : 0.0);
  }
  problem.spatial_absorber_rate.assign(problem.n_modes, 0.0);
  problem.spatial_kperp_rate.assign(problem.n_modes, 0.0);
  problem.spatial_evanescent_rate = 200.0;
  problem.spatial_kappa_sq = {{1.0, 0.25}, {2.0, 0.5},  {3.0, 0.75},
                              {9.0, 1.0},  {12.0, 1.5}, {15.0, 2.0}};
  return problem;
}

struct PlanarElement {
  Complex linear, prefactor;
};

PlanarElement planar_element(const native::Problem& problem, const std::size_t column,
                             const std::size_t omega) {
  const Complex square =
      (problem.spatial_k_squared[omega] - problem.spatial_kperp_sq[column]) +
      (problem.spatial_k_squared_low[omega] - problem.spatial_kperp_sq_low[column]) -
      problem.spatial_kappa_sq[(column % problem.n_p) * problem.n_omega + omega];
  const std::complex<double> root =
      std::sqrt(std::complex<double>(square.real(), square.imag()));
  const bool propagating = square.real() > 0.0;
  const double attenuation =
      propagating ? root.imag()
                  : std::min(std::abs(root.imag()), problem.spatial_evanescent_rate);
  return {Complex(-attenuation, -(root.real() - problem.spatial_frame[omega])) -
              0.5 * problem.spatial_spectral_rate[omega],
          propagating ? problem.spatial_numerator[omega] / root.real()
                      : problem.spatial_clamped[omega]};
}

// The stage straight from the definitions: the inverse DFT in y, each row's real time
// signal, the slab synthesis, the cube, the slab projection, the DFT in time and in y,
// and each column's prefactor.
std::vector<Complex> planar_reference(const native::Problem& problem,
                                      const std::vector<Complex>& columns) {
  const auto n_p = problem.n_p, n_x = problem.n_x, n_y = problem.n_y;
  const auto n = problem.n_omega;
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
      rows[p + n_p * iy] = real_time(spectrum, problem.n_to);
    }
  std::vector<Complex> projected(n_p * n_y * n);
  for (std::size_t iy = 0; iy < n_y; ++iy) {
    std::vector<std::vector<double>> polarisation(n_x,
                                                  std::vector<double>(problem.n_to));
    for (std::size_t ix = 0; ix < n_x; ++ix)
      for (std::size_t time = 0; time < problem.n_to; ++time) {
        double field = 0.0;
        for (std::size_t p = 0; p < n_p; ++p)
          field += problem.synthesis_matrix[ix * n_p + p] * rows[p + n_p * iy][time];
        polarisation[ix][time] = problem.kerr_coefficient * field * field * field;
      }
    for (std::size_t p = 0; p < n_p; ++p) {
      std::vector<Complex> signal(problem.n_to);
      for (std::size_t time = 0; time < problem.n_to; ++time)
        for (std::size_t ix = 0; ix < n_x; ++ix)
          signal[time] +=
              problem.projection_matrix[p * n_x + ix] * polarisation[ix][time];
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
                                     problem.omega_window[omega] * sum;
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
    ASSERT_EQ(initial.size(), problem.n_modes * problem.n_omega);
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
    for (std::size_t column = 0; column < problem.n_modes; ++column)
      for (std::size_t omega = 0; omega < problem.n_omega; ++omega) {
        const auto index = column * problem.n_omega + omega;
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
  partial_row.node_chunk = 4;
  EXPECT_THROW(native::GeneralState state(partial_row), std::invalid_argument);
  auto short_kappa = planar_problem(1);
  short_kappa.spatial_kappa_sq.pop_back();
  EXPECT_THROW(native::GeneralState state(short_kappa), std::invalid_argument);
  auto coarse = planar_problem(1);
  coarse.projection_coarse_matrix = coarse.projection_matrix;
  EXPECT_THROW(native::GeneralState state(coarse), std::invalid_argument);
  auto vector = planar_problem(1);
  vector.n_pol_state = 2;
  EXPECT_THROW(native::GeneralState state(vector), std::invalid_argument);
  auto columns = planar_problem(1);
  columns.n_p = 3;
  EXPECT_THROW(native::GeneralState state(columns), std::invalid_argument);
  auto modal = general_problem(native::FieldFamily::field_real);
  modal.spatial_kappa_sq = {{1.0, 0.0}};
  EXPECT_THROW(native::GeneralState state(modal), std::invalid_argument);
}

TEST(GeneralState, PortablePlasmaTeamScanMatchesTheSerialTrapezoid) {
  using View = Kokkos::View<double**, native::Layout, native::MemorySpace>;
  const native::ExecutionSpace execution_space;
  View rates("solprop_native.test.plasma_rates", 2, 8);
  View prefixes("solprop_native.test.plasma_prefixes", 2, 8);
  auto host_rates = Kokkos::create_mirror_view(rates);
  for (std::size_t row = 0; row < 2; ++row)
    for (std::size_t sample = 0; sample < 8; ++sample)
      host_rates(row, sample) = static_cast<double>((row + 1) * (sample + 2)) / 16.0;
  Kokkos::deep_copy(execution_space, rates, host_rates);
  native::portable_team_trapezoid_scan(execution_space, rates, prefixes, 0.125);
  execution_space.fence("solprop_native plasma team-scan test");
  const auto host_prefixes =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, prefixes);
  for (std::size_t row = 0; row < 2; ++row) {
    double expected = 0.0;
    // Exact: the empty prefix.
    EXPECT_EQ(host_prefixes(row, 0), expected);
    for (std::size_t sample = 1; sample < 8; ++sample) {
      expected += 0.125 * (host_rates(row, sample - 1) + host_rates(row, sample)) / 2.0;
      EXPECT_NEAR(host_prefixes(row, sample), expected, 8.0 * kEpsilon);
    }
  }
}

// solprop SEP 18's band filter written out independently: the row extended by point
// reflection, then the centre and the weighted pairs in the reference's order, every
// product and sum its own statement.
std::vector<double> reference_band_filter(const std::vector<double>& row,
                                          const std::vector<double>& taps) {
  const auto samples = static_cast<std::ptrdiff_t>(row.size());
  const auto half = static_cast<std::ptrdiff_t>(taps.size()) - 1;
  const std::ptrdiff_t last = samples - 1;
  const auto extended = [&](const std::ptrdiff_t index) {
    if (index < 0) {
      const double doubled = 2.0 * row.front();
      return doubled - row[static_cast<std::size_t>(-index)];
    }
    if (index > last) {
      const double doubled = 2.0 * row.back();
      return doubled - row[static_cast<std::size_t>(2 * last - index)];
    }
    return row[static_cast<std::size_t>(index)];
  };
  std::vector<double> filtered(row.size());
  for (std::ptrdiff_t index = 0; index < samples; ++index) {
    double total = taps[0] * row[static_cast<std::size_t>(index)];
    for (std::ptrdiff_t k = 1; k <= half; ++k) {
      const double pair = extended(index - k) + extended(index + k);
      const double term = taps[static_cast<std::size_t>(k)] * pair;
      total = total + term;
    }
    filtered[static_cast<std::size_t>(index)] = total;
  }
  return filtered;
}

// The rounding bound of each filtered sample, `gamma_{2K+3}` times the sum of its
// terms' magnitudes (a reflected sample `2 y_0 - y_m` bounded by `2 |y_0| + |y_m|`),
// whatever order or contraction a compiler gives the products and sums.
std::vector<double> filter_tolerance(const std::vector<double>& row,
                                     const std::vector<double>& taps) {
  const auto samples = static_cast<std::ptrdiff_t>(row.size());
  const auto half = static_cast<std::ptrdiff_t>(taps.size()) - 1;
  const std::ptrdiff_t last = samples - 1;
  const auto size = [&](const std::ptrdiff_t index) {
    if (index < 0)
      return 2.0 * std::abs(row.front()) +
             std::abs(row[static_cast<std::size_t>(-index)]);
    if (index > last)
      return 2.0 * std::abs(row.back()) +
             std::abs(row[static_cast<std::size_t>(2 * last - index)]);
    return std::abs(row[static_cast<std::size_t>(index)]);
  };
  const double operations = static_cast<double>(2 * half + 3);
  const double gamma = operations * kEpsilon / (1.0 - operations * kEpsilon);
  std::vector<double> tolerance(row.size());
  for (std::ptrdiff_t index = 0; index < samples; ++index) {
    double total = std::abs(taps[0]) * size(index);
    for (std::ptrdiff_t k = 1; k <= half; ++k)
      total += std::abs(taps[static_cast<std::size_t>(k)]) *
               (size(index - k) + size(index + k));
    tolerance[static_cast<std::size_t>(index)] = gamma * total;
  }
  return tolerance;
}

// One band scan of `rows` rows of `samples` samples, staged or direct, on the host.
template <bool Staged>
std::pair<std::vector<double>, std::vector<double>>
team_band_scan(const std::vector<double>& values, const std::size_t rows,
               const std::size_t samples, const std::vector<double>& taps,
               const double spacing, const int team_size) {
  using View = Kokkos::View<double**, native::Layout, native::MemorySpace>;
  const native::ExecutionSpace execution_space;
  View input("solprop_native.test.band_input", rows, samples);
  View filtered("solprop_native.test.band_filtered", rows, samples);
  View prefixes("solprop_native.test.band_prefixes", rows, samples);
  Kokkos::View<double*, native::Layout, native::MemorySpace> tap_view(
      "solprop_native.test.band_taps", taps.size());
  auto host_input = Kokkos::create_mirror_view(input);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t sample = 0; sample < samples; ++sample)
      host_input(row, sample) = values[row * samples + sample];
  auto host_taps = Kokkos::create_mirror_view(tap_view);
  for (std::size_t k = 0; k < taps.size(); ++k)
    host_taps(k) = taps[k];
  Kokkos::deep_copy(execution_space, input, host_input);
  Kokkos::deep_copy(execution_space, tap_view, host_taps);
  native::portable_team_band_scan<Staged>(execution_space, input, filtered, prefixes,
                                          tap_view, spacing, team_size);
  execution_space.fence("solprop_native band team-scan test");
  const auto host_filtered =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, filtered);
  const auto host_prefixes =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, prefixes);
  std::vector<double> filtered_values(rows * samples), prefix_values(rows * samples);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t sample = 0; sample < samples; ++sample) {
      filtered_values[row * samples + sample] = host_filtered(row, sample);
      prefix_values[row * samples + sample] = host_prefixes(row, sample);
    }
  return {filtered_values, prefix_values};
}

TEST(GeneralState, BandTeamScanMatchesTheSerialFilterAndTrapezoid) {
  // A K = 8 rule of unit sum with alternating taps, as the certified designs have.
  std::vector<double> taps(native::plasma_max_taps);
  double wings = 0.0;
  for (std::size_t k = 1; k < taps.size(); ++k) {
    taps[k] = std::pow(-0.3, static_cast<double>(k)) / static_cast<double>(k + 1);
    wings += taps[k];
  }
  taps[0] = 1.0 - 2.0 * wings;
  constexpr double spacing = 0.125;
  // The build's team size and narrower teams: a device's 64 lanes, or on the host
  // (whose own choice is one thread) teams of two and four threads where it has them.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const std::vector<int> team_sizes = {0, 64};
#else
  using TeamPolicy = Kokkos::TeamPolicy<native::ExecutionSpace>;
  const TeamPolicy probe(native::ExecutionSpace{}, 1, 1);
  const int widest = probe.team_size_max(EmptyTeamBody{}, Kokkos::ParallelForTag());
  std::vector<int> team_sizes = {0};
  for (const int size : {2, 4})
    if (size <= widest)
      team_sizes.push_back(size);
#endif
  // A short row whose reflections reach across it, one under a tile, and rows across
  // several staging tiles with a remainder.
  for (const std::size_t samples :
       {std::size_t{9}, std::size_t{40}, std::size_t{1000}, std::size_t{2600}}) {
    for (const std::size_t half : {std::size_t{0}, std::size_t{3}, std::size_t{8}}) {
      if (samples <= half)
        continue;
      // K = 0 is the trapezoid's one identity tap.
      const std::vector<double> local =
          half == 0
              ? std::vector<double>{1.0}
              : std::vector<double>(
                    taps.begin(), taps.begin() + static_cast<std::ptrdiff_t>(half) + 1);
      constexpr std::size_t rows = 3;
      // A rate-like row: exactly zero below a threshold, a smooth pulse above it, and
      // a pre-ionised row that is live at both ends.
      std::vector<double> values(rows * samples);
      for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t sample = 0; sample < samples; ++sample) {
          const double x =
              (static_cast<double>(sample) + 0.5) / static_cast<double>(samples) - 0.5;
          const double pulse =
              std::exp(-40.0 * x * x) * (1.0 + 0.25 * std::cos(90.0 * x));
          values[row * samples + sample] =
              row == 0 ? (pulse > 0.2 ? pulse : 0.0)
                       : (row == 1 ? pulse : 0.5 + pulse * static_cast<double>(row));
        }
      for (const int team_size : team_sizes) {
        const auto direct =
            team_band_scan<false>(values, rows, samples, local, spacing, team_size);
        const auto staged =
            team_band_scan<true>(values, rows, samples, local, spacing, team_size);
        for (std::size_t row = 0; row < rows; ++row) {
          const std::vector<double> input(
              values.begin() + static_cast<std::ptrdiff_t>(row * samples),
              values.begin() + static_cast<std::ptrdiff_t>((row + 1) * samples));
          const auto expected = reference_band_filter(input, local);
          // Each filtered sample is held to the rounding bound of its K + 1 products
          // and 2K + 1 sums, whatever order or contraction a compiler gives them
          // (solprop SEP 25); the staged path, which only reads its halo from
          // elsewhere, to twice that against the direct one.
          const auto tolerance = filter_tolerance(input, local);
          for (std::size_t sample = 0; sample < samples; ++sample) {
            const auto flat = row * samples + sample;
            ASSERT_NEAR(direct.first[flat], expected[sample], tolerance[sample])
                << samples << " " << half << " " << team_size << " " << row << " "
                << sample;
            ASSERT_NEAR(staged.first[flat], direct.first[flat], 2.0 * tolerance[sample])
                << samples << " " << half << " " << team_size << " " << row << " "
                << sample;
          }
          // The scan's tree associates differently from the serial sum. Both stay
          // within gamma_n of the sum of the increments' magnitudes, so they differ
          // by at most twice that (Higham, the recursive and pairwise summation
          // bounds), and the filtered samples' own bound carries into the prefix.
          double serial = 0.0;
          double magnitude = 0.0;
          double carried = 0.0;
          const double gamma = static_cast<double>(samples) * kEpsilon /
                               (1.0 - static_cast<double>(samples) * kEpsilon);
          for (std::size_t sample = 0; sample < samples; ++sample) {
            if (sample != 0) {
              const double increment =
                  spacing * (expected[sample] + expected[sample - 1]) / 2.0;
              serial += increment;
              magnitude += std::abs(increment);
              carried += spacing * (tolerance[sample] + tolerance[sample - 1]) / 2.0;
            }
            const auto flat = row * samples + sample;
            const double bound = 2.0 * gamma * magnitude + carried;
            EXPECT_NEAR(direct.second[flat], serial, bound)
                << samples << " " << half << " " << team_size << " " << row << " "
                << sample;
            // Staged against direct: each within `bound` of the serial sum.
            EXPECT_NEAR(staged.second[flat], direct.second[flat], 2.0 * bound)
                << samples << " " << half << " " << team_size << " " << row << " "
                << sample;
          }
        }
      }
    }
  }
}

// One tiled band scan (`portable_tiled_band_scan`) of `rows` rows of `samples`
// samples in tiles of `tile`, on teams of `team_size` threads.
std::vector<double> tiled_band_scan(const std::vector<double>& values,
                                    const std::size_t rows, const std::size_t samples,
                                    const std::vector<double>& taps,
                                    const double spacing, const std::size_t tile,
                                    const int team_size) {
  using View = Kokkos::View<double**, native::Layout, native::MemorySpace>;
  const native::ExecutionSpace execution_space;
  const auto tiles = native::band_tile_count(samples, tile);
  View input("solprop_native.test.tiled_input", rows, samples);
  View locals("solprop_native.test.tiled_locals", rows, samples);
  View totals("solprop_native.test.tiled_totals", rows, tiles);
  View prefixes("solprop_native.test.tiled_prefixes", rows, samples);
  Kokkos::View<double*, native::Layout, native::MemorySpace> tap_view(
      "solprop_native.test.tiled_taps", taps.size());
  auto host_input = Kokkos::create_mirror_view(input);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t sample = 0; sample < samples; ++sample)
      host_input(row, sample) = values[row * samples + sample];
  auto host_taps = Kokkos::create_mirror_view(tap_view);
  for (std::size_t k = 0; k < taps.size(); ++k)
    host_taps(k) = taps[k];
  Kokkos::deep_copy(execution_space, input, host_input);
  Kokkos::deep_copy(execution_space, tap_view, host_taps);
  // A stale local row would show as NaN.
  Kokkos::deep_copy(execution_space, locals, std::numeric_limits<double>::quiet_NaN());
  native::portable_tiled_band_scan(execution_space, input, locals, totals, prefixes,
                                   tap_view, spacing, tile, team_size);
  execution_space.fence("solprop_native tiled band-scan test");
  const auto host_prefixes =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, prefixes);
  std::vector<double> result(rows * samples);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t sample = 0; sample < samples; ++sample)
      result[row * samples + sample] = host_prefixes(row, sample);
  return result;
}

// The tiled scan's team sizes: the build's and narrower ones, as for the team scan.
std::vector<int> tiled_team_sizes() {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  return {0, 64};
#else
  using TeamPolicy = Kokkos::TeamPolicy<native::ExecutionSpace>;
  const TeamPolicy probe(native::ExecutionSpace{}, 1, 1);
  const int widest = probe.team_size_max(EmptyTeamBody{}, Kokkos::ParallelForTag());
  std::vector<int> sizes = {0};
  for (const int size : {2, 4})
    if (size <= widest)
      sizes.push_back(size);
  return sizes;
#endif
}

TEST(GeneralState, TiledTrapezoidScanIsExactOnIntegers) {
  // The device trapezoid plasma's two-level scan: integer samples and a spacing of
  // two make every increment and partial sum an exact integer, so any association
  // gives the serial result exactly, over tiles and team sizes.
  using View = Kokkos::View<double**, native::Layout, native::MemorySpace>;
  const native::ExecutionSpace execution_space;
  constexpr double spacing = 2.0;
  for (const std::size_t samples :
       {std::size_t{1}, std::size_t{9}, std::size_t{2600}}) {
    constexpr std::size_t rows = 3;
    View input("solprop_native.test.trapezoid_input", rows, samples);
    View prefixes("solprop_native.test.trapezoid_prefixes", rows, samples);
    auto host_input = Kokkos::create_mirror_view(input);
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t sample = 0; sample < samples; ++sample)
        host_input(row, sample) =
            static_cast<double>((sample * 7 + row * 13) % 23) - 11.0;
    Kokkos::deep_copy(execution_space, input, host_input);
    for (const std::size_t tile :
         {std::size_t{1}, std::size_t{5}, std::size_t{64}, std::size_t{1024}}) {
      View totals("solprop_native.test.trapezoid_totals", rows,
                  (samples + tile - 1) / tile);
      for (const int team_size : tiled_team_sizes()) {
        Kokkos::deep_copy(execution_space, prefixes,
                          std::numeric_limits<double>::quiet_NaN());
        native::portable_tiled_trapezoid_scan(execution_space, input, totals, prefixes,
                                              spacing, tile, team_size);
        execution_space.fence("solprop_native tiled trapezoid test");
        const auto host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, prefixes);
        for (std::size_t row = 0; row < rows; ++row) {
          double serial = 0.0;
          for (std::size_t sample = 0; sample < samples; ++sample) {
            if (sample != 0)
              serial += spacing *
                        (host_input(row, sample) + host_input(row, sample - 1)) / 2.0;
            ASSERT_EQ(host(row, sample), serial)
                << samples << " " << tile << " " << team_size << " " << row << " "
                << sample;
          }
        }
      }
    }
  }
}

TEST(GeneralState, BandTileSizesFollowTheRowLength) {
  EXPECT_EQ(native::band_tile_samples(64), native::band_tile_min);
  EXPECT_EQ(native::band_tile_samples(8192), std::size_t{256});
  EXPECT_EQ(native::band_tile_samples(16384), std::size_t{512});
  EXPECT_EQ(native::band_tile_samples(32768), native::band_tile_max);
  EXPECT_EQ(native::band_tile_samples(65536), native::band_tile_max);
  EXPECT_EQ(native::band_tile_samples(65536, 512), std::size_t{512});
  EXPECT_EQ(native::band_tile_count(65536, 1024), std::size_t{64});
  EXPECT_EQ(native::band_tile_count(1000, 256), std::size_t{4});
}

TEST(GeneralState, TiledBandScanIsExactOnIntegers) {
  // Integer taps, integer samples and a spacing of two: every reflection, filtered
  // sample, increment `2 (y_n + y_{n-1}) / 2` and partial sum is an integer far below
  // 2^53, so any association and contraction gives the serial result exactly.
  const std::vector<double> taps = {6.0, -2.0, 1.0, -1.0};
  constexpr double spacing = 2.0;
  for (const std::size_t samples :
       {std::size_t{9}, std::size_t{40}, std::size_t{1000}, std::size_t{2600}}) {
    for (const std::size_t half : {std::size_t{0}, std::size_t{3}}) {
      const std::vector<double> local = half == 0 ? std::vector<double>{1.0} : taps;
      constexpr std::size_t rows = 3;
      std::vector<double> values(rows * samples);
      for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t sample = 0; sample < samples; ++sample)
          values[row * samples + sample] =
              static_cast<double>((sample * 7 + row * 13) % 23) - 11.0;
      for (const std::size_t tile :
           {std::size_t{1}, std::size_t{5}, std::size_t{64}, std::size_t{256}}) {
        for (const int team_size : tiled_team_sizes()) {
          const auto scanned =
              tiled_band_scan(values, rows, samples, local, spacing, tile, team_size);
          for (std::size_t row = 0; row < rows; ++row) {
            const std::vector<double> input(
                values.begin() + static_cast<std::ptrdiff_t>(row * samples),
                values.begin() + static_cast<std::ptrdiff_t>((row + 1) * samples));
            const auto filtered = reference_band_filter(input, local);
            double serial = 0.0;
            for (std::size_t sample = 0; sample < samples; ++sample) {
              if (sample != 0)
                serial += spacing * (filtered[sample] + filtered[sample - 1]) / 2.0;
              ASSERT_EQ(scanned[row * samples + sample], serial)
                  << samples << " " << half << " " << tile << " " << team_size << " "
                  << row << " " << sample;
            }
          }
        }
      }
    }
  }
}

TEST(GeneralState, TiledBandScanMatchesTheSerialFilterAndTrapezoid) {
  std::vector<double> taps(native::plasma_max_taps);
  double wings = 0.0;
  for (std::size_t k = 1; k < taps.size(); ++k) {
    taps[k] = std::pow(-0.3, static_cast<double>(k)) / static_cast<double>(k + 1);
    wings += taps[k];
  }
  taps[0] = 1.0 - 2.0 * wings;
  constexpr double spacing = 0.125;
  for (const std::size_t samples :
       {std::size_t{9}, std::size_t{40}, std::size_t{1000}, std::size_t{2600}}) {
    for (const std::size_t half : {std::size_t{0}, std::size_t{3}, std::size_t{8}}) {
      if (samples <= half)
        continue;
      const std::vector<double> local =
          half == 0
              ? std::vector<double>{1.0}
              : std::vector<double>(
                    taps.begin(), taps.begin() + static_cast<std::ptrdiff_t>(half) + 1);
      constexpr std::size_t rows = 3;
      std::vector<double> values(rows * samples);
      for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t sample = 0; sample < samples; ++sample) {
          const double x =
              (static_cast<double>(sample) + 0.5) / static_cast<double>(samples) - 0.5;
          const double pulse =
              std::exp(-40.0 * x * x) * (1.0 + 0.25 * std::cos(90.0 * x));
          values[row * samples + sample] =
              row == 0 ? (pulse > 0.2 ? pulse : 0.0)
                       : (row == 1 ? pulse : 0.5 + pulse * static_cast<double>(row));
        }
      for (const std::size_t tile :
           {std::size_t{7}, std::size_t{256}, native::band_tile_max}) {
        for (const int team_size : tiled_team_sizes()) {
          const auto scanned =
              tiled_band_scan(values, rows, samples, local, spacing, tile, team_size);
          for (std::size_t row = 0; row < rows; ++row) {
            const std::vector<double> input(
                values.begin() + static_cast<std::ptrdiff_t>(row * samples),
                values.begin() + static_cast<std::ptrdiff_t>((row + 1) * samples));
            const auto expected = reference_band_filter(input, local);
            const auto tolerance = filter_tolerance(input, local);
            // A prefix's tree (the tile's scan, the earlier totals' reduction and the
            // addition joining them) sums its increments in n - 1 additions, so it and
            // the serial sum are each within gamma_n of the increments' magnitudes and
            // differ by at most twice that, plus the filtered samples' own bound.
            double serial = 0.0;
            double magnitude = 0.0;
            double carried = 0.0;
            const double gamma = static_cast<double>(samples) * kEpsilon /
                                 (1.0 - static_cast<double>(samples) * kEpsilon);
            for (std::size_t sample = 0; sample < samples; ++sample) {
              if (sample != 0) {
                const double increment =
                    spacing * (expected[sample] + expected[sample - 1]) / 2.0;
                serial += increment;
                magnitude += std::abs(increment);
                carried += spacing * (tolerance[sample] + tolerance[sample - 1]) / 2.0;
              }
              ASSERT_NEAR(scanned[row * samples + sample], serial,
                          2.0 * gamma * magnitude + carried)
                  << samples << " " << half << " " << tile << " " << team_size << " "
                  << row << " " << sample;
            }
          }
        }
      }
    }
  }
}

// One row through the host column's two forms on a team of `team_size` threads: the
// whole-row band filter and scan, and the filter and scan limited to the row's live
// window (`band_support`, `team_trapezoid_window`), whose filtered row starts as NaN so
// a read outside the window would show.
struct WindowedScan {
  std::vector<double> filtered;
  std::vector<double> whole;
  std::vector<double> windowed;
  native::BandWindow window;
  double total;
};

WindowedScan windowed_band_scan(const std::vector<double>& row,
                                const std::vector<double>& taps, const double spacing,
                                const int team_size) {
  using View = Kokkos::View<double**, native::Layout, native::MemorySpace>;
  const native::ExecutionSpace execution_space;
  const auto samples = row.size();
  const int count = static_cast<int>(samples);
  const int half = static_cast<int>(taps.size()) - 1;
  int live_first = count;
  int live_last = -1;
  for (int i = 0; i < count; ++i)
    if (row[static_cast<std::size_t>(i)] != 0.0) {
      live_first = std::min(live_first, i);
      live_last = i;
    }
  const auto window = native::band_support(live_first, live_last, count, half);
  View input("solprop_native.test.window_input", 1, samples);
  View filtered("solprop_native.test.window_filtered", 1, samples);
  View partial("solprop_native.test.window_partial", 1, samples);
  View whole("solprop_native.test.window_whole", 1, samples);
  View windowed("solprop_native.test.window_windowed", 1, samples);
  Kokkos::View<double*, native::Layout, native::MemorySpace> totals(
      "solprop_native.test.window_total", 1);
  auto host_input = Kokkos::create_mirror_view(input);
  for (std::size_t i = 0; i < samples; ++i)
    host_input(0, i) = row[i];
  Kokkos::deep_copy(execution_space, input, host_input);
  Kokkos::deep_copy(execution_space, partial, std::numeric_limits<double>::quiet_NaN());
  Kokkos::View<double*, native::Layout, native::MemorySpace> tap_view(
      "solprop_native.test.window_taps", taps.size());
  auto host_taps = Kokkos::create_mirror_view(tap_view);
  for (std::size_t k = 0; k < taps.size(); ++k)
    host_taps(k) = taps[k];
  Kokkos::deep_copy(execution_space, tap_view, host_taps);
  using TeamPolicy = Kokkos::TeamPolicy<native::ExecutionSpace>;
  Kokkos::parallel_for(
      "solprop_native.test.windowed_band_scan",
      TeamPolicy(execution_space, 1, team_size),
      KOKKOS_LAMBDA(const TeamPolicy::member_type& team) {
        double local[native::plasma_max_taps] = {};
        for (int k = 0; k <= half; ++k)
          local[k] = tap_view(static_cast<std::size_t>(k));
        const auto tile = native::band_tile<false>(team);
        native::team_band_filter<false>(team, tile, input, 0, filtered, 0, count, local,
                                        half);
        native::team_trapezoid_prefix(team, filtered, 0, count, spacing,
                                      [&](const int sample, const double prefix) {
                                        whole(0, static_cast<std::size_t>(sample)) =
                                            prefix;
                                      });
        if (window.first <= window.last)
          native::team_band_filter<false>(team, tile, input, 0, partial, 0, count,
                                          local, half, window.first, window.last + 1);
        const double total = native::team_trapezoid_window(
            team, partial, 0, window, count, spacing,
            [&](const int sample, const double prefix) {
              windowed(0, static_cast<std::size_t>(sample)) = prefix;
            });
        Kokkos::single(Kokkos::PerTeam(team), [&] { totals(0) = total; });
      });
  execution_space.fence("solprop_native windowed band-scan test");
  const auto host_filtered =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, filtered);
  const auto host_whole =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, whole);
  const auto host_windowed =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, windowed);
  const auto host_totals =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, totals);
  WindowedScan result{std::vector<double>(samples), std::vector<double>(samples),
                      std::vector<double>(samples), window, host_totals(0)};
  for (std::size_t i = 0; i < samples; ++i) {
    result.filtered[i] = host_filtered(0, i);
    result.whole[i] = host_whole(0, i);
    result.windowed[i] = host_windowed(0, i);
  }
  return result;
}

TEST(GeneralState, BandWindowSkipsOnlyExactZeros) {
  std::vector<double> taps(native::plasma_max_taps);
  double wings = 0.0;
  for (std::size_t k = 1; k < taps.size(); ++k) {
    taps[k] = std::pow(-0.3, static_cast<double>(k)) / static_cast<double>(k + 1);
    wings += taps[k];
  }
  taps[0] = 1.0 - 2.0 * wings;
  constexpr double spacing = 0.125;
  constexpr std::size_t samples = 64;
  const std::size_t half = taps.size() - 1;
  using TeamPolicy = Kokkos::TeamPolicy<native::ExecutionSpace>;
  const TeamPolicy probe(native::ExecutionSpace{}, 1, 1);
  const int widest = probe.team_size_max(EmptyTeamBody{}, Kokkos::ParallelForTag());
  // A zero row, and spikes at the first sample, within K of the start, within K of the
  // end and in the interior: the window reaches an end or stops K samples short of it.
  std::vector<std::vector<double>> rows(5, std::vector<double>(samples, 0.0));
  const std::array<std::size_t, 4> spikes = {0, half - 1, samples - half, samples / 2};
  for (std::size_t i = 0; i < spikes.size(); ++i)
    rows[i + 1][spikes[i]] = 1.0 + 0.25 * static_cast<double>(i);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    for (const int team_size : {1, 2, 4}) {
      if (team_size > widest)
        continue;
      const auto scan = windowed_band_scan(rows[r], taps, spacing, team_size);
      const int first = scan.window.first;
      const int last = scan.window.last;
      for (std::size_t i = 0; i < samples; ++i) {
        const auto index = static_cast<int>(i);
        // Exact, structural: outside the window every sample the filter reads is zero,
        // so the whole-row filter gives exactly zero there; the window's prefix is
        // exactly zero before the window and exactly its total after it.
        if (index < first || index > last) {
          ASSERT_EQ(scan.filtered[i], 0.0) << r << " " << team_size << " " << i;
        }
        if (index < first) {
          ASSERT_EQ(scan.windowed[i], 0.0) << r << " " << team_size << " " << i;
        }
        if (index > last) {
          ASSERT_EQ(scan.windowed[i], scan.total) << r << " " << team_size << " " << i;
        }
        // Exact for one thread: Kokkos's host team scan then adds the increments in
        // sample order in both forms, each filtered sample is the same function of
        // the same inputs, and the increments the window skips are exact zeros that
        // the whole-row form adds to an exact zero or to its total. With more threads
        // the chunks start at different samples and the forms associate differently,
        // so they agree to the scan's rounding bound alone (tests/test_band_stress.py
        // holds the column to the reference there).
        if (team_size == 1) {
          ASSERT_EQ(scan.windowed[i], scan.whole[i]) << r << " " << i;
        }
      }
      if (r == 0) {
        EXPECT_GT(first, last);
        EXPECT_EQ(scan.total, 0.0);
      } else {
        EXPECT_NE(scan.total, 0.0);
      }
    }
  }
}

TEST(PlasmaBandScan, FilteringPreservesShortRowsBlockTailsAndDynamicRange) {
  const std::vector<double> taps{1.1, -0.1, 0.0625, -0.025, 0.0125};
  const int half = static_cast<int>(taps.size()) - 1;
  using TeamPolicy = Kokkos::TeamPolicy<native::ExecutionSpace>;
  const TeamPolicy probe(native::ExecutionSpace{}, 1, 1);
  const int widest = probe.team_size_max(EmptyTeamBody{}, Kokkos::ParallelForTag());
  for (const int samples : {5, 17, 255, 257, 519}) {
    SCOPED_TRACE(samples);
    std::vector<double> row(static_cast<std::size_t>(samples));
    for (int sample = 0; sample < samples; ++sample)
      row[static_cast<std::size_t>(sample)] = sample % 3 == 0   ? 1e100
                                              : sample % 3 == 1 ? -1e100
                                                                : 1e-100;
    for (const int team_size : {1, 2, 4}) {
      if (team_size > widest)
        continue;
      const auto scan = windowed_band_scan(row, taps, 0.125, team_size);
      const auto reflected = [&](const int index) -> long double {
        if (index < 0)
          return 2.0L * row.front() - row[static_cast<std::size_t>(-index)];
        if (index >= samples)
          return 2.0L * row.back() -
                 row[static_cast<std::size_t>(2 * (samples - 1) - index)];
        return row[static_cast<std::size_t>(index)];
      };
      for (int sample = 0; sample < samples; ++sample) {
        long double expected = taps[0] * reflected(sample);
        long double magnitude = std::abs(expected);
        for (int tap = 1; tap <= half; ++tap) {
          const auto before = reflected(sample - tap);
          const auto after = reflected(sample + tap);
          expected += taps[static_cast<std::size_t>(tap)] * (before + after);
          magnitude += std::abs(taps[static_cast<std::size_t>(tap)]) *
                       (std::abs(before) + std::abs(after));
        }
        // Centre + ordered tap pairs + reflections, in both computations. This
        // also covers platforms whose long double has no extra mantissa bits.
        const double bound = (8.0 * half + 16.0) * kEpsilon *
                             std::max(1.0, static_cast<double>(magnitude));
        ASSERT_GT(bound, 0.0);
        const auto actual = scan.filtered[static_cast<std::size_t>(sample)];
        ASSERT_TRUE(std::isfinite(actual));
        EXPECT_NEAR(actual, static_cast<double>(expected), bound);
      }
    }
  }
}

TEST(State, RepeatedTrialsOverwriteFullCropAndStageZeroAccumulators) {
  auto problem = base_problem(native::FieldFamily::envelope);
  problem.nonlinear = true;
  const std::vector<Complex> initial = {
      {1.0, 0.0}, {0.25, -0.125}, {-0.1, 0.05}, {0.05, 0.025}};
  native::State state(problem);
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
      2.0 * 512.0 * kEpsilon * static_cast<double>(problem.n_to) * largest;
  for (std::size_t index = 0; index < first.size(); ++index)
    EXPECT_NEAR(second[index], first[index], tolerance) << index;
  // Exact: a trial never writes the accepted state.
  EXPECT_EQ(state.field_to_host(), initial);
}

TEST(State, RejectsDuplicateCropBins) {
  auto problem = base_problem(native::FieldFamily::envelope);
  problem.crop_index[1] = problem.crop_index[0];
  EXPECT_THROW(native::State state(problem), std::invalid_argument);
}

TEST(State, SplitStepAbsorberDampsOnlyAcceptedState) {
  auto problem = base_problem(native::FieldFamily::envelope);
  problem.nonlinear = false;
  problem.split_step_absorber = true;
  problem.absorber_coarse.assign(problem.n_t, 2.0);
  const std::vector<Complex> initial(problem.n_omega, {1.0, -0.25});
  constexpr double dz = 0.1;
  native::State state(problem);
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

TEST(State, InStageAbsorberPreservesFirstSameAsLastEligibility) {
  auto problem = base_problem(native::FieldFamily::envelope);
  problem.nonlinear = false;
  problem.in_stage_absorber = true;
  problem.first_same_as_last = true;
  problem.absorber_fine.assign(problem.n_to, 2.0);
  const std::vector<Complex> initial(problem.n_omega, {1.0, 0.0});
  constexpr double dz = 0.1;
  native::State state(problem);
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
  problem.n_stages = 3;
  problem.tableau_matrix.assign(9, 0.0);
  problem.tableau_nodes = {0.0, 0.5, 1.0};
  problem.solution_weights = {1.0, 0.0, 0.0};
  problem.error_weights = {0.25, 0.0, -0.25};
  problem.projection_coarse_matrix.clear();
  problem.low_storage = true;
  problem.low_storage_flows = flows;
  for (std::size_t mode = 0; mode < problem.n_modes; ++mode)
    for (std::size_t index = 0; index < problem.n_omega; ++index)
      problem.phi_terms[mode * problem.n_omega + index] =
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
    const Complex exponent = 0.5 * dz * problem.phi_terms[element];
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
  std::vector<Complex> picture(size), g1(size);
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
          "weighted_residual", "recycled_stage", "stage_inverse", "held_plane"})
      EXPECT_EQ(ledger.at(row), 0U) << row;
    // The field-free plane held in the memory space, one volume, copied back once.
    EXPECT_THROW(static_cast<void>(state.held_to_host()), std::logic_error);
    state.hold_plane();
    EXPECT_EQ(state.memory_ledger().at("held_plane"), field);
    const auto held = state.held_to_host();
    const auto current = state.field_to_host();
    for (std::size_t index = 0; index < held.size(); ++index)
      EXPECT_EQ(held[index], current[index]);
  }
}

TEST(GeneralState, LowStorageStepsAreRefusedOutsideTheirState) {
  auto runge_kutta = low_storage_problem(1);
  runge_kutta.low_storage = false;
  runge_kutta.low_storage_flows = 0;
  native::GeneralState state(runge_kutta);
  state.set_initial(std::vector<Complex>(6), {});
  EXPECT_THROW(static_cast<void>(state.low_storage_step(
                   0.1, low_storage_scalars(runge_kutta, 0.1), false)),
               std::logic_error);
  auto three = low_storage_problem(3);
  EXPECT_THROW(native::GeneralState too_many(three), std::invalid_argument);
  auto elided = low_storage_problem(0);
  elided.low_storage_stage_register = false;
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
  problem.n_p = 0;
  problem.n_modes = problem.n_x * problem.n_y;
  problem.n_nodes = problem.n_modes;
  problem.synthesis_matrix.clear();
  problem.projection_matrix.clear();
  problem.spatial_kappa_sq.clear();
  // Construction only: the unit transverse square is k_x = 1, k_y = 0.
  problem.spatial_k_x.assign(problem.n_x, 1.0);
  problem.spatial_k_y.assign(problem.n_y, 0.0);
  problem.spatial_kperp_sq.assign(problem.n_modes, 1.0);
  problem.spatial_kperp_sq_low.assign(problem.n_modes, 0.0);
  problem.spatial_absorber_rate.assign(problem.n_modes, 0.0);
  problem.spatial_kperp_rate.assign(problem.n_modes, 0.0);
  problem.low_storage = true;
  return problem;
}

TEST(GeneralState, LowStorageRefusesRegistersItsFoldsCannotRead) {
  // The exchange fold of an in-stage absorber re-forms its input from the stage
  // register, so it is never elided there.
  auto elided = low_storage_free_problem();
  elided.low_storage_stage_register = false;
  EXPECT_EQ(construction_refusal(elided), "");
  elided.in_stage_absorber = true;
  elided.absorber_fine.assign(elided.n_to, 1.0);
  EXPECT_NE(construction_refusal(elided).find("beside an in-stage absorber"),
            std::string::npos);
  // A separable fold applies one flow volume, a constant operator's table.
  auto table = low_storage_free_problem();
  table.low_storage_flows = 1;
  EXPECT_EQ(construction_refusal(table), "");
  auto pair = low_storage_free_problem();
  pair.low_storage_flows = 2;
  EXPECT_NE(construction_refusal(pair).find("at most one flow volume"),
            std::string::npos);
  auto planar = planar_problem(1);
  planar.low_storage = true;
  planar.low_storage_flows = 2;
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

TEST(State, CloseIsIdempotentAndRejectsFurtherUse) {
  native::State state(base_problem(native::FieldFamily::envelope));
  state.close();
  state.close();
  EXPECT_TRUE(state.closed());
  EXPECT_THROW(state.set_initial(std::vector<Complex>(4), {}), std::logic_error);
}

TEST(CpuThreadPolicy, AutomaticBoundariesAreMonotoneAndCapped) {
  using native::CpuOperation;
  using native::CpuPool;
  const std::array operations{
      CpuOperation::stage, CpuOperation::response, CpuOperation::dense,
      CpuOperation::fft,   CpuOperation::raman,    CpuOperation::copy,
      CpuOperation::flow,  CpuOperation::modal,    CpuOperation::adams};
  const std::array thresholds{
      native::stage_thread_min_elements, native::response_thread_min_elements,
      native::dense_thread_min_elements, native::default_threaded_fft_min_samples,
      native::raman_thread_min_elements, native::copy_thread_min_elements,
      native::flow_thread_min_elements,  native::modal_thread_min_elements,
      native::adams_thread_min_elements};
  const std::array full_thresholds{
      native::stage_full_thread_elements, native::response_full_thread_elements,
      native::dense_full_thread_elements, native::fft_full_thread_elements,
      native::raman_full_thread_elements, native::copy_thread_min_elements,
      native::flow_thread_min_elements,   native::modal_thread_min_elements,
      native::adams_full_thread_elements};
  for (const int cap : {1, 2, 3, 8, 72}) {
    native::ThreadPolicy policy{cap, cap, cap, cap,
                                native::default_threaded_fft_min_samples};
    for (std::size_t index = 0; index < operations.size(); ++index) {
      const auto width = [&](const std::size_t elements) {
        return native::operation_threads(policy, CpuPool::kernels, operations[index],
                                         elements);
      };
      EXPECT_EQ(width(0), native::cpu_size_policy_enabled ? 1 : cap);
      EXPECT_EQ(width(thresholds[index] - 1),
                native::cpu_size_policy_enabled ? 1 : cap);
      EXPECT_EQ(width(thresholds[index]),
                native::cpu_size_policy_enabled
                    ? (thresholds[index] < full_thresholds[index]
                           ? std::min(cap, native::cpu_medium_threads)
                           : cap)
                    : cap);
      EXPECT_EQ(width(full_thresholds[index]), cap);
      int previous = 1;
      for (std::size_t elements = 1; elements <= 4 * full_thresholds[index];
           elements *= 2) {
        const auto actual = width(elements);
        EXPECT_GE(actual, previous);
        EXPECT_LE(actual, cap);
        previous = actual;
      }
    }
  }
}

TEST(CpuThreadPolicy, ShortCopiesDoNotInheritTheStageWidth) {
  native::ThreadPolicy policy{18, 18, 18, 18, native::default_threaded_fft_min_samples};
  EXPECT_EQ(native::operation_threads(policy, native::CpuPool::kernels,
                                      native::CpuOperation::copy, 8192),
            native::cpu_size_policy_enabled ? 1 : 18);
  EXPECT_EQ(native::operation_threads(policy, native::CpuPool::kernels,
                                      native::CpuOperation::stage, 8196),
            18);
  policy.explicit_pools = 1;
  EXPECT_EQ(native::operation_threads(policy, native::CpuPool::kernels,
                                      native::CpuOperation::copy, 8192),
            18);
}

TEST(CpuThreadPolicy, SmoothGridSizesKeepThePowerOfTwoWidths) {
  // The four-mode RDW on n_t 4000 (a 2-3-5-smooth size) against 4096: on Linux its
  // stage of 8004 elements and coarse FFT batch of 32000 samples take the widths of
  // 8196 and 32768, rather than falling to one thread just under a power-of-two
  // boundary. macOS keeps the earlier boundaries, where the serial ones are faster.
#if defined(__APPLE__)
  GTEST_SKIP() << "macOS keeps the 8192/32768 boundaries";
#endif
  native::ThreadPolicy policy{18, 18, 18, 18, native::default_threaded_fft_min_samples};
  const auto width = [&](const native::CpuOperation operation, const std::size_t n) {
    return native::operation_threads(policy, native::CpuPool::kernels, operation, n);
  };
  using native::CpuOperation;
  EXPECT_EQ(width(CpuOperation::stage, 8004), width(CpuOperation::stage, 8196));
  EXPECT_EQ(width(CpuOperation::stage, 8004), 18);
  EXPECT_EQ(width(CpuOperation::fft, 32000), width(CpuOperation::fft, 32768));
  EXPECT_GT(width(CpuOperation::fft, 32000), 1);
}

TEST(CpuThreadPolicy, CoupledMatrixWindowRetainsFullWidth) {
  native::ThreadPolicy policy{72, 72, 72, 72, 32768, -1};
  // Active bins * modes^3 in the measured three-mode wall, independent of runtime.
  EXPECT_EQ(native::operation_threads(policy, native::CpuPool::kernels,
                                      native::CpuOperation::flow, 1433 * 3 * 3 * 3),
            72);
}

TEST(CpuThreadPolicy, ResponseShapeThresholdsAreMonotoneAndCapped) {
  using native::CpuOperation;
  using native::CpuPool;
  for (const int cap : {1, 3, 18, 72}) {
    native::ThreadPolicy policy{cap, cap, cap, cap,
                                native::default_threaded_fft_min_samples};
    const auto width = [&](const std::size_t rows, const std::size_t samples) {
      return native::operation_threads(policy, CpuPool::kernels, CpuOperation::response,
                                       rows * samples, samples);
    };
    EXPECT_EQ(width(33, 4096),
              native::cpu_size_policy_enabled ? std::min(cap, 4) : cap);
    const auto columns = native::response_full_thread_columns;
    EXPECT_EQ(width(columns - 1, 1024),
              native::cpu_size_policy_enabled ? std::min(cap, 4) : cap);
    EXPECT_EQ(width(columns, 1024), cap);
    EXPECT_EQ(width(columns, 1), native::cpu_size_policy_enabled ? 1 : cap);
    EXPECT_EQ(width(44, 2048), cap);
    EXPECT_EQ(width(68, 2048), cap);
    EXPECT_EQ(width(1, native::response_full_thread_samples - 1),
              native::cpu_size_policy_enabled ? std::min(cap, 4) : cap);
    EXPECT_EQ(width(1, native::response_full_thread_samples), cap);
    EXPECT_EQ(width(0, native::response_full_thread_samples),
              native::cpu_size_policy_enabled ? 1 : cap);
    for (std::size_t rows = 1; rows <= 64; rows *= 2) {
      int previous = 1;
      for (std::size_t samples = 1024; samples <= 131072; samples *= 2) {
        const auto actual = width(rows, samples);
        EXPECT_GE(actual, previous);
        EXPECT_LE(actual, cap);
        previous = actual;
      }
    }
    policy.explicit_pools = 1;
    EXPECT_EQ(width(1, 1), cap);
  }
}

TEST(CpuThreadPolicy, ExplicitPoolsBypassSizingIndependently) {
  using native::CpuOperation;
  using native::CpuPool;
  native::ThreadPolicy policy{7, 6, 5, 4, native::default_threaded_fft_min_samples};
  const std::array pools{CpuPool::kernels, CpuPool::fft, CpuPool::raman_fft,
                         CpuPool::raman_tail_fft};
  for (unsigned requested = 0; requested < pools.size(); ++requested) {
    policy.explicit_pools = 1U << requested;
    for (unsigned other = 0; other < pools.size(); ++other) {
      const auto actual =
          native::operation_threads(policy, pools[other], CpuOperation::fft, 1);
      EXPECT_EQ(actual, requested == other || !native::cpu_size_policy_enabled
                            ? 7 - static_cast<int>(other)
                            : 1);
    }
  }
  policy.explicit_pools = 0;
  policy.threaded_fft_min_samples = 17;
  EXPECT_EQ(native::operation_threads(policy, CpuPool::fft, CpuOperation::fft, 16),
            native::cpu_size_policy_enabled ? 1 : 6);
  EXPECT_EQ(native::operation_threads(policy, CpuPool::fft, CpuOperation::fft, 17),
            native::cpu_size_policy_enabled ? std::min(6, native::cpu_medium_threads)
                                            : 6);
}

TEST(CpuThreadPolicy, GeneralModalTransformsKeepTheirMeasuredWidth) {
  native::ThreadPolicy policy{18, 18, 18, 18, native::default_threaded_fft_min_samples};
  for (const std::size_t elements : {98496U, 209304U}) {
    EXPECT_EQ(native::operation_threads(policy, native::CpuPool::kernels,
                                        native::CpuOperation::modal, elements),
              18);
    EXPECT_EQ(native::operation_threads(policy, native::CpuPool::kernels,
                                        native::CpuOperation::dense, elements),
              native::cpu_size_policy_enabled ? 1 : 18);
  }
}

TEST(CpuThreadPolicy, MetadataNamesEveryOperationClass) {
  native::CpuExecutionPool pool(native::CpuPool::kernels);
  const std::array operations{
      native::CpuOperation::stage, native::CpuOperation::response,
      native::CpuOperation::dense, native::CpuOperation::fft,
      native::CpuOperation::raman, native::CpuOperation::copy,
      native::CpuOperation::flow,  native::CpuOperation::modal,
      native::CpuOperation::adams};
  const std::array names{"stage", "response", "dense", "fft",  "raman",
                         "copy",  "flow",     "modal", "adams"};
  for (std::size_t i = 0; i < operations.size(); ++i) {
    pool.get(operations[i], 1);
    EXPECT_EQ(
        pool.sizing().at(std::string("kernels.") + names[i] + ".1").at("elements"), 1U);
  }
}

TEST(CpuThreadPolicy, AKernelPoolAtFullWidthKeepsItsTeam) {
  // GNU OpenMP ends a pool's surplus threads when a narrower region follows a wider
  // one, so a kernel pool that holds a full-width instance hands it to medium-width
  // operations too; without one, they keep the medium width.
  const int cap = native::thread_policy().kernels;
  if (!native::cpu_size_policy_enabled || cap <= native::cpu_medium_threads)
    GTEST_SKIP() << "needs per-operation sizing and more than the medium width";
  const auto response = std::size_t{33} * 4096;
  native::CpuExecutionPool alone(native::CpuPool::kernels);
  EXPECT_EQ(alone.get(native::CpuOperation::response, response, 4096).concurrency(),
            native::cpu_medium_threads);
  EXPECT_EQ(alone.get(native::CpuOperation::stage, 1).concurrency(), 1);
  native::CpuExecutionPool full(native::CpuPool::kernels);
  EXPECT_EQ(full.get(native::CpuOperation::stage, 1U << 20).concurrency(), cap);
  EXPECT_EQ(full.get(native::CpuOperation::response, response, 4096).concurrency(),
            cap);
  EXPECT_EQ(full.get(native::CpuOperation::adams, 16384).concurrency(), cap);
  EXPECT_EQ(full.get(native::CpuOperation::stage, 1).concurrency(), 1);
  const auto key = "kernels.response." + std::to_string(response) + ".samples.4096";
  EXPECT_EQ(full.sizing().at(key).at("threads"), static_cast<std::size_t>(cap));
  // The FFT pools keep their medium widths: FFTW's threaded loops run on the
  // runtime's whole team and only distribute fewer iterations.
  native::CpuExecutionPool transforms(native::CpuPool::fft);
  EXPECT_EQ(transforms.get(native::CpuOperation::fft, 1U << 20).concurrency(), cap);
  EXPECT_EQ(transforms.get(native::CpuOperation::fft, 32768).concurrency(),
            native::cpu_medium_threads);
}

TEST(CpuThreadPolicy, ACompactPoolKeepsAFewThreads) {
  // A compact problem's kernels share a team of at most cpu_compact_threads, medium
  // widths promoted to it; its transforms take that team only where FFTW's loops
  // follow a scope, and run serially otherwise.
  const auto policy = native::thread_policy();
  if (!native::cpu_size_policy_enabled || policy.explicit_pools != 0)
    GTEST_SKIP() << "needs automatic per-operation sizing";
  const int team = std::min(policy.kernels, native::cpu_compact_threads);
  native::CpuExecutionPool kernels(native::CpuPool::kernels);
  kernels.compact();
  EXPECT_EQ(kernels.widest(), 0);
  EXPECT_EQ(kernels.get(native::CpuOperation::stage, 1U << 20).concurrency(), team);
  EXPECT_EQ(kernels.get(native::CpuOperation::response, std::size_t{33} * 4096, 4096)
                .concurrency(),
            team);
  EXPECT_EQ(kernels.get(native::CpuOperation::stage, 1).concurrency(), 1);
  EXPECT_EQ(kernels.widest(), team);
  native::CpuExecutionPool transforms(native::CpuPool::fft);
  transforms.compact();
  EXPECT_EQ(transforms.get(native::CpuOperation::fft, 1U << 20).concurrency(),
            native::fftw_team_loops() ? team : 1);
  native::CpuExecutionPool full(native::CpuPool::kernels);
  EXPECT_EQ(full.get(native::CpuOperation::stage, 1U << 20).concurrency(),
            policy.kernels);
}

TEST(CpuThreadPolicy, CompactProblemsHaveOneColumnOnOneNode) {
  const auto base = general_problem(native::FieldFamily::field_real);
  EXPECT_TRUE(native::compact_problem(base));
  auto nodes = base;
  nodes.n_nodes = 2;
  EXPECT_FALSE(native::compact_problem(nodes));
  auto modes = base;
  modes.n_modes = 2;
  EXPECT_FALSE(native::compact_problem(modes));
  auto pairs = base;
  pairs.n_pol_transform = 2;
  EXPECT_FALSE(native::compact_problem(pairs));
  auto long_column = base;
  long_column.n_to = native::compact_max_fine_samples;
  EXPECT_FALSE(native::compact_problem(long_column));
  auto modal = base;
  modal.general_modal = true;
  EXPECT_FALSE(native::compact_problem(modal));
  auto flow = base;
  flow.coupled_flow = true;
  EXPECT_FALSE(native::compact_problem(flow));
  auto free_space = base;
  free_space.n_x = 2;
  EXPECT_FALSE(native::compact_problem(free_space));
}

TEST(CpuThreadPolicy, MetadataDistinguishesEqualWorkWithDifferentResponseShapes) {
  native::CpuExecutionPool pool(native::CpuPool::kernels);
  const auto samples = native::response_full_thread_samples;
  pool.get(native::CpuOperation::response, samples, 4096);
  pool.get(native::CpuOperation::response, samples, samples);
  const auto sizing = pool.sizing();
  for (const auto fine : {std::size_t{4096}, samples}) {
    const auto key = "kernels.response." + std::to_string(samples) + ".samples." +
                     std::to_string(fine);
    EXPECT_EQ(sizing.at(key).at("elements"), samples);
    EXPECT_EQ(sizing.at(key).at("fine_samples"), fine);
    EXPECT_EQ(sizing.at(key).at("columns"), samples / fine);
  }
}

TEST(CpuThreadPolicy, MatrixPackingDoesNotInheritTheDenseWorkWidth) {
  auto problem = general_problem(native::FieldFamily::field_real);
  problem.n_to = 32768;
  problem.n_nodes = 35;
  problem.node_chunk = 33;
  problem.synthesis_matrix.assign(problem.n_nodes, 1.0);
  problem.projection_matrix.assign(problem.n_nodes, 1.0 / 35.0);
  problem.projection_coarse_matrix = problem.projection_matrix;
  problem.time_window.assign(problem.n_to, 1.0);
  problem.time_window_derivative.assign(problem.n_to, 0.0);
  problem.carrier.assign(problem.n_to, Complex(1.0, 0.0));
  native::GeneralState state(problem);
  const auto sizing = state.execution_sizing();
  // Full and tail matrices contain 33 and 2 entries, or twice those counts
  // for stacked fine/coarse projections; the output has 32768 samples.
  for (const std::size_t elements : {2U, 4U, 33U, 66U, 32768U}) {
    const auto& choice = sizing.at("kernels.copy." + std::to_string(elements));
    EXPECT_EQ(choice.at("elements"), elements);
    EXPECT_EQ(choice.at("threads"),
              native::cpu_size_policy_enabled
                  ? 1U
                  : static_cast<unsigned>(native::ExecutionSpace().concurrency()));
  }
  EXPECT_EQ(sizing.at("kernels.dense.1081344").at("threads"),
            static_cast<std::size_t>(native::ExecutionSpace().concurrency()));
}

TEST(CpuThreadPolicy, WorkspacesKeepIndependentChoicesAndReleaseTheirPools) {
  auto problem = base_problem(native::FieldFamily::envelope);
  native::State small(problem);
  problem.n_to = native::fft_full_thread_elements;
  problem.time_window.assign(problem.n_to, 1.0);
  problem.carrier.assign(problem.n_to, Complex(1.0, 0.0));
  native::State large(problem);
  native::State another_small(base_problem(native::FieldFamily::envelope));
  const auto small_sizing = small.execution_sizing();
  EXPECT_EQ(small_sizing, another_small.execution_sizing());
  const auto key = "fft.fft." + std::to_string(problem.n_to);
  EXPECT_EQ(large.execution_sizing().at(key).at("elements"), problem.n_to);
  EXPECT_EQ(large.execution_sizing().at(key).at("threads"),
            static_cast<std::size_t>(native::ExecutionSpace().concurrency()));
  large.close();
  EXPECT_EQ(small.execution_sizing(), small_sizing);
  small.close();
  another_small.close();
}

} // namespace
