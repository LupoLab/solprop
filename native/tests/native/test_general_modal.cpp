// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/core.hpp"
#include "solprop_native/general_modal.hpp"

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace native = solprop_native;
using native::Complex;

constexpr double kEpsilon = std::numeric_limits<double>::epsilon();
constexpr std::size_t kComponents = 3;
constexpr std::size_t kModes = 2;
constexpr std::size_t kNodes = 5;
constexpr std::size_t kTerms = 2;
constexpr std::size_t kResponses = 2;

// Every source element accumulates at most M + C^4 + C + Q C + 2 n_to (the two
// transforms, summed directly by the reference) complex products: 2 + 81 + 3 + 15 +
// 16 = 117 for the instantaneous term, and the delayed term adds C^3 + 2 (2 n_to) =
// 59 for each of its two channels. Four real operations per complex product and a
// rounding each gives the 1024 below, relative to the largest element; the delayed
// case doubles it. Fixed before the first run. The real cubic's symmetric
// contraction rounds less than the dense count C^4 it is held to: at most 17
// operations a sample (five forming a coefficient, two a monomial, ten in the sum).
constexpr double kInstantaneousBound = 1024 * kEpsilon;
constexpr double kDelayedBound = 2048 * kEpsilon;

Complex cis(const double angle) { return {std::cos(angle), std::sin(angle)}; }

// A deterministic, non-degenerate fill: no symmetry the route could exploit.
double fill(const std::size_t index, const double scale) {
  return scale * std::sin(0.7 * static_cast<double>(index) + 0.3);
}

std::size_t fourth(const std::size_t value) { return value * value * value * value; }

// The default grid has three coarse bins at the head of an eight-sample fine grid;
// `n_omega`, `offset` and `n_to` place a wider crop anywhere inside a finer one.
native::Problem modal_problem(const bool real_form, const std::size_t chunk,
                              const bool delayed, const std::size_t n_omega = 3,
                              const std::size_t offset = 0,
                              const std::size_t n_to = 8) {
  native::Problem problem;
  problem.n_t = n_to / 2;
  problem.n_to = n_to;
  problem.n_omega = n_omega;
  problem.n_stages = 1;
  problem.n_phi_terms = 1;
  problem.n_prefactor_terms = 1;
  problem.n_modes = kModes;
  problem.family = native::FieldFamily::field_real;
  problem.crop_index.resize(n_omega);
  std::iota(problem.crop_index.begin(), problem.crop_index.end(), offset);
  problem.active_mask.assign(problem.n_omega, 1U);
  problem.norm_mask.assign(problem.n_omega, 1U);
  problem.omega_window.assign(problem.n_omega, 1.0);
  for (std::size_t w = 0; w < n_omega; ++w)
    problem.omega.push_back(static_cast<double>(w + 1));
  problem.dt = 0.25;
  problem.time_window.assign(problem.n_to, 1.0);
  problem.carrier.assign(problem.n_to, Complex(1.0, 0.0));
  problem.phi_terms.assign(kModes * problem.n_omega, Complex{});
  problem.prefactor_terms.assign(kModes * problem.n_omega, Complex(1.0, 0.0));
  problem.tableau_matrix = {0.0};
  problem.tableau_nodes = {0.0};
  problem.solution_weights = {1.0};
  problem.error_weights = {0.25};
  problem.synthesis_matrix = {0.0, 0.0};
  problem.projection_matrix = {0.0, 0.0};
  problem.phase_b_engine = true;
  problem.general_modal = true;
  problem.general_real_form = real_form;
  problem.general_components = kComponents;
  problem.general_nodes = kNodes;
  problem.general_node_chunk = chunk;
  problem.general_pad_scale = 2.0;
  problem.general_epsilon0 = 1.0;
  const auto maps = kNodes * kComponents * kModes * problem.n_omega;
  for (std::size_t index = 0; index < maps; ++index) {
    problem.general_synthesis.emplace_back(fill(index, 1.0), fill(index + 97, 0.5));
    problem.general_projection.emplace_back(fill(index + 13, 0.8),
                                            fill(index + 41, 0.6));
  }
  for (std::size_t index = 0; index < kNodes * fourth(kComponents); ++index)
    problem.general_tensor.push_back(fill(index + 7, 1.0));
  for (std::size_t w = 0; w < n_omega; ++w)
    problem.general_frequency_factor.emplace_back(0.0, -static_cast<double>(w + 1));
  if (delayed) {
    problem.general_responses = kResponses;
    problem.general_terms = kTerms;
    for (std::size_t index = 0; index < kResponses * kTerms * fourth(kComponents);
         ++index)
      problem.general_delayed_tensor.push_back(fill(index + 3, 0.5));
    for (std::size_t index = 0; index < kResponses * kTerms * (problem.n_to + 1);
         ++index)
      problem.general_delayed_spectrum.emplace_back(fill(index + 5, 1.0),
                                                    fill(index + 11, 0.4));
    problem.general_node_response = {1, 2, 0, 1, 2};
  }
  return problem;
}

std::vector<Complex> scaled_state(const native::Problem& problem) {
  std::vector<Complex> state;
  for (std::size_t index = 0; index < kModes * problem.n_omega; ++index)
    state.emplace_back(fill(index + 17, 1.0), fill(index + 23, 1.0));
  return state;
}

// solprop's `general_source`, written as sums over samples and bins.
std::vector<Complex> reference_source(const native::Problem& problem,
                                      const std::vector<Complex>& scaled) {
  const auto n_to = problem.n_to;
  const auto n_omega = problem.n_omega;
  const auto components = kComponents;
  const auto two_pi = 2.0 * std::numbers::pi_v<double>;
  std::vector<Complex> source(kModes * n_omega);
  for (std::size_t node = 0; node < kNodes; ++node) {
    std::vector<std::vector<Complex>> field(components, std::vector<Complex>(n_to));
    for (std::size_t c = 0; c < components; ++c)
      for (std::size_t w = 0; w < n_omega; ++w) {
        Complex bin{};
        for (std::size_t m = 0; m < kModes; ++m)
          bin +=
              problem
                  .general_synthesis[((node * components + c) * kModes + m) * n_omega +
                                     w] *
              scaled[m * n_omega + w];
        const auto fine = problem.crop_index[w];
        for (std::size_t t = 0; t < n_to; ++t)
          field[c][t] +=
              bin *
              cis(two_pi * static_cast<double>(fine * t) / static_cast<double>(n_to)) /
              static_cast<double>(n_to);
      }
    if (problem.general_real_form)
      for (auto& component : field)
        for (auto& value : component)
          value = Complex(value.real(), 0.0);
    std::vector<std::vector<double>> polarisation(components,
                                                  std::vector<double>(n_to));
    const auto base = node * fourth(components);
    for (std::size_t t = 0; t < n_to; ++t)
      for (std::size_t i = 0; i < components; ++i) {
        Complex value{};
        for (std::size_t j = 0; j < components; ++j)
          for (std::size_t k = 0; k < components; ++k)
            for (std::size_t l = 0; l < components; ++l)
              value += problem.general_tensor[base +
                                              ((i * components + j) * components + k) *
                                                  components +
                                              l] *
                       field[k][t] * Kokkos::conj(field[l][t]) * field[j][t];
        polarisation[i][t] = problem.general_epsilon0 * value.real();
      }
    if (problem.general_terms != 0 && problem.general_node_response[node] != 0) {
      const auto owner = problem.general_node_response[node] - 1;
      const auto doubled = 2 * n_to;
      for (std::size_t term = 0; term < problem.general_terms; ++term) {
        std::vector<std::vector<std::vector<double>>> history(
            components,
            std::vector<std::vector<double>>(components, std::vector<double>(n_to)));
        for (std::size_t k = 0; k < components; ++k)
          for (std::size_t l = 0; l < components; ++l) {
            std::vector<double> drive(doubled);
            for (std::size_t t = 0; t < n_to; ++t) {
              const double pair = (field[k][t] * Kokkos::conj(field[l][t])).real();
              drive[t] = problem.general_real_form ? pair : 0.5 * pair;
            }
            std::vector<Complex> product(n_to + 1);
            for (std::size_t f = 0; f <= n_to; ++f) {
              Complex bin{};
              for (std::size_t t = 0; t < doubled; ++t)
                bin += drive[t] * cis(-two_pi * static_cast<double>(f * t) /
                                      static_cast<double>(doubled));
              product[f] =
                  problem.general_delayed_spectrum
                      [(owner * problem.general_terms + term) * (n_to + 1) + f] *
                  bin;
            }
            for (std::size_t t = 0; t < n_to; ++t) {
              double value =
                  product[0].real() + product[n_to].real() * (t % 2 == 0 ? 1.0 : -1.0);
              for (std::size_t f = 1; f < n_to; ++f)
                value += 2.0 * (product[f] * cis(two_pi * static_cast<double>(f * t) /
                                                 static_cast<double>(doubled)))
                                   .real();
              history[k][l][t] = value / static_cast<double>(doubled);
            }
          }
        const auto tensor = (owner * problem.general_terms + term) * fourth(components);
        for (std::size_t t = 0; t < n_to; ++t)
          for (std::size_t i = 0; i < components; ++i) {
            double value = 0.0;
            for (std::size_t j = 0; j < components; ++j)
              for (std::size_t k = 0; k < components; ++k)
                for (std::size_t l = 0; l < components; ++l)
                  value +=
                      problem.general_delayed_tensor
                          [tensor +
                           ((i * components + j) * components + k) * components + l] *
                      history[k][l][t] * field[j][t].real();
            polarisation[i][t] += problem.general_epsilon0 * value;
          }
      }
    }
    for (std::size_t c = 0; c < components; ++c)
      for (std::size_t w = 0; w < n_omega; ++w) {
        const auto fine = problem.crop_index[w];
        Complex bin{};
        for (std::size_t t = 0; t < n_to; ++t)
          bin += polarisation[c][t] * cis(-two_pi * static_cast<double>(fine * t) /
                                          static_cast<double>(n_to));
        for (std::size_t m = 0; m < kModes; ++m)
          source[m * n_omega + w] +=
              problem
                  .general_projection[((m * kNodes + node) * components + c) * n_omega +
                                      w] *
              bin;
      }
  }
  for (std::size_t m = 0; m < kModes; ++m)
    for (std::size_t w = 0; w < n_omega; ++w)
      source[m * n_omega + w] *= problem.general_frequency_factor[w];
  return source;
}

std::vector<Complex> evaluated(const native::Problem& problem,
                               const std::vector<Complex>& scaled) {
  native::GeneralModalSource source(problem);
  source.plan();
  auto input = Kokkos::create_mirror_view(source.scaled());
  for (std::size_t m = 0; m < kModes; ++m)
    for (std::size_t w = 0; w < problem.n_omega; ++w)
      input(m, w) = scaled[m * problem.n_omega + w];
  Kokkos::deep_copy(source.scaled(), input);
  source.evaluate();
  const auto output =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, source.source());
  std::vector<Complex> result;
  for (std::size_t m = 0; m < kModes; ++m)
    for (std::size_t w = 0; w < problem.n_omega; ++w)
      result.push_back(output(m, w));
  return result;
}

void expect_close(const std::vector<Complex>& actual,
                  const std::vector<Complex>& expected, const double relative) {
  ASSERT_EQ(actual.size(), expected.size());
  double scale = 0.0;
  for (const auto& value : expected)
    scale = std::max(scale, Kokkos::abs(value));
  ASSERT_GT(scale, 0.0);
  for (std::size_t index = 0; index < actual.size(); ++index)
    EXPECT_LE(Kokkos::abs(actual[index] - expected[index]), relative * scale)
        << "element " << index;
}

// A repeated trial's partials -- squared norms of the field, the solution and the
// embedded error per mode -- against the first trial's (solprop SEP 25). The stage
// carries the delayed source's rounding relative to its largest element and a squared
// norm doubles a relative error; the bound is taken relative to the largest partial,
// so a whole partial left over from the first trial still fails it.
void expect_partials_close(const std::vector<double>& actual,
                           const std::vector<double>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  double scale = 0.0;
  for (const double value : expected)
    scale = std::max(scale, std::abs(value));
  ASSERT_GT(scale, 0.0);
  for (std::size_t index = 0; index < actual.size(); ++index)
    EXPECT_LE(std::abs(actual[index] - expected[index]), 2.0 * kDelayedBound * scale)
        << "partial " << index;
}

TEST(GeneralModal, BothCubicsMatchTheStraightforwardHostSource) {
  for (const bool real_form : {true, false}) {
    const auto problem = modal_problem(real_form, 2, false);
    const auto scaled = scaled_state(problem);
    expect_close(evaluated(problem, scaled), reference_source(problem, scaled),
                 kInstantaneousBound);
  }
}

// Five coarse bins at fine bins 0 to 4 of an eight-sample grid, the last the Nyquist
// bin: the real form's complex-to-real transform halves the doubled bins and drops
// the imaginary parts of the zero and Nyquist bins, as the analytic signal's real part
// does.
TEST(GeneralModal, ACropThroughTheNyquistBinMatchesTheStraightforwardHostSource) {
  for (const bool real_form : {true, false}) {
    const auto problem = modal_problem(real_form, 2, false, 5, 0, 8);
    const auto scaled = scaled_state(problem);
    expect_close(evaluated(problem, scaled), reference_source(problem, scaled),
                 kInstantaneousBound);
  }
}

// Eleven coarse bins at fine bins 3 to 13 of a 32-sample grid: the projection's
// teams of `general_vector_lanes` bins include a partial one, and the crop starts
// away from bin zero. The products per element grow to 2 + 81 + 3 + 15 + 64 = 165,
// still inside the 1024 eps above.
TEST(GeneralModal, AWideOffsetCropMatchesTheStraightforwardHostSource) {
  static_assert(native::general_vector_lanes < 11);
  for (const bool real_form : {true, false}) {
    const auto problem = modal_problem(real_form, 2, false, 11, 3, 32);
    const auto scaled = scaled_state(problem);
    expect_close(evaluated(problem, scaled), reference_source(problem, scaled),
                 kInstantaneousBound);
  }
}

// The projection's block widens while every thread still has a team, and the widest
// block is 64 bins: the graded-index guide's 2 modes x 65 bins take 64 at one and
// four threads and 8 at eighteen (a measured optimum on an 18-core socket).
TEST(GeneralModal, TheProjectionBlockWidensWhileEveryThreadHasATeam) {
  EXPECT_EQ(native::general_projection_lanes(2, 65, 1), 64U);
  EXPECT_EQ(native::general_projection_lanes(2, 65, 4), 64U);
  EXPECT_EQ(native::general_projection_lanes(2, 65, 10), 16U);
  EXPECT_EQ(native::general_projection_lanes(2, 65, 18), native::general_vector_lanes);
  EXPECT_EQ(native::general_projection_lanes(1, 3, 1000), native::general_vector_lanes);
  EXPECT_EQ(native::general_projection_lanes(40, 4097, 1), 64U);
}

TEST(GeneralModal, DelayedTensorResponseMatchesTheDirectConvolution) {
  for (const bool real_form : {true, false}) {
    const auto problem = modal_problem(real_form, 2, true);
    const auto scaled = scaled_state(problem);
    expect_close(evaluated(problem, scaled), reference_source(problem, scaled),
                 kDelayedBound);
  }
}

TEST(GeneralModal, ChunkingChangesOnlyTheGroupingOfTheNodeSum) {
  const auto whole = modal_problem(false, kNodes, true);
  const auto scaled = scaled_state(whole);
  const auto expected = evaluated(whole, scaled);
  for (const std::size_t chunk : {std::size_t{1}, std::size_t{2}, std::size_t{3}})
    expect_close(evaluated(modal_problem(false, chunk, true), scaled), expected,
                 kDelayedBound);
}

// The same evaluation twice is the same discrete computation: held to the delayed
// source's rounding bound, not to its bits (solprop SEP 25).
TEST(GeneralModal, RepeatedEvaluationAgreesToRounding) {
  for (const bool delayed : {false, true}) {
    const auto problem = modal_problem(true, 2, delayed);
    const auto scaled = scaled_state(problem);
    const auto expected = reference_source(problem, scaled);
    native::GeneralModalSource source(problem);
    const auto ledger = source.memory_ledger();
    source.plan();
    EXPECT_EQ(source.memory_ledger(), ledger); // Tail views alias existing storage.
    auto input = Kokkos::create_mirror(source.scaled());
    for (int repeat = 0; repeat < 3; ++repeat) {
      for (std::size_t m = 0; m < kModes; ++m)
        for (std::size_t w = 0; w < problem.n_omega; ++w)
          input(m, w) = scaled[m * problem.n_omega + w];
      Kokkos::deep_copy(source.scaled(), input);
      source.evaluate();
      const auto output =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, source.source());
      std::vector<Complex> result(output.data(), output.data() + output.size());
      expect_close(result, expected, delayed ? kDelayedBound : kInstantaneousBound);
      Kokkos::deep_copy(source.scaled(), Complex{});
      source.evaluate();
      const auto zero =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, source.source());
      for (std::size_t index = 0; index < zero.size(); ++index)
        EXPECT_EQ(zero.data()[index], Complex{});
      EXPECT_EQ(source.memory_ledger(), ledger);
    }
    source.close();
    source.close();
  }
}

TEST(GeneralModal, MalformedDataIsRefusedBeforeAllocation) {
  auto short_map = modal_problem(true, 2, false);
  short_map.general_synthesis.pop_back();
  EXPECT_THROW(native::require_general_modal(short_map), std::invalid_argument);
  auto chunk = modal_problem(true, 2, false);
  chunk.general_node_chunk = kNodes + 1;
  EXPECT_THROW(native::require_general_modal(chunk), std::invalid_argument);
  auto owner = modal_problem(true, 2, true);
  owner.general_node_response[0] = kResponses + 1;
  EXPECT_THROW(native::require_general_modal(owner), std::invalid_argument);
  auto two = modal_problem(true, 2, true);
  two.general_components = 2;
  EXPECT_THROW(native::require_general_modal(two), std::invalid_argument);
  auto legacy = modal_problem(true, 2, false);
  legacy.nonlinear = true;
  EXPECT_THROW(native::require_general_modal(legacy), std::invalid_argument);
  auto envelope = modal_problem(true, 2, false);
  envelope.family = native::FieldFamily::envelope;
  EXPECT_THROW(native::require_general_modal(envelope), std::invalid_argument);
  auto scattered = modal_problem(true, 2, false);
  scattered.crop_index = {0, 2, 1};
  EXPECT_THROW(native::require_general_modal(scattered), std::invalid_argument);
  auto stray = modal_problem(true, 2, false);
  stray.general_modal = false;
  EXPECT_THROW(native::require_general_modal(stray), std::invalid_argument);
  auto infinite = modal_problem(true, 2, false);
  infinite.general_tensor[3] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(native::GeneralState state(infinite), std::invalid_argument);
}

TEST(GeneralModal, FullAndTailTransformsUseTheDedicatedPolicy) {
  // Three full nodes and two tail nodes, with three components and two modes.
  // These cross the modal threshold while staying below the generic GEMM threshold.
  auto problem = modal_problem(true, 3, false, 4096, 0, 16384);
  native::GeneralModalSource source(problem);
  const auto sizing = source.execution_sizing();
  const auto full = 3 * kComponents * kModes * problem.n_omega;
  const auto tail = 2 * kComponents * kModes * problem.n_omega;
  const auto capacity =
      static_cast<std::size_t>(native::ExecutionSpace().concurrency());
  EXPECT_EQ(sizing.at("kernels.modal." + std::to_string(full)).at("threads"), capacity);
  EXPECT_EQ(sizing.at("kernels.modal." + std::to_string(tail)).at("threads"),
            native::cpu_size_policy_enabled ? 1U : capacity);
}

TEST(GeneralModal, LedgerRowsAreTheAllocatedViews) {
  const auto problem = modal_problem(true, 2, true);
  native::GeneralModalSource source(problem);
  const auto ledger = source.memory_ledger();
  const auto empty = native::GeneralModalSource::empty_ledger();
  ASSERT_EQ(ledger.size(), empty.size());
  for (const auto& [name, bytes] : empty) {
    EXPECT_EQ(bytes, 0U);
    EXPECT_TRUE(ledger.contains(name)) << name;
  }
  const auto complex = sizeof(Complex);
  const auto real = sizeof(double);
  const auto rows = 2 * kComponents;
  const auto pairs = 2 * native::general_pair_histories;
  const auto n_to = problem.n_to;
  EXPECT_EQ(ledger.at("general_synthesis"), problem.general_synthesis.size() * complex);
  EXPECT_EQ(ledger.at("general_projection"),
            problem.general_projection.size() * complex);
  EXPECT_EQ(ledger.at("general_tensor"), problem.general_tensor.size() * real);
  EXPECT_EQ(ledger.at("general_node_spectrum"), rows * n_to * complex);
  EXPECT_EQ(ledger.at("general_node_field"), rows * n_to * complex);
  EXPECT_EQ(ledger.at("general_node_polarisation"), rows * n_to * real);
  EXPECT_EQ(ledger.at("general_node_source"), rows * (n_to / 2 + 1) * complex);
  EXPECT_EQ(ledger.at("general_node_drive"), pairs * 2 * n_to * real);
  EXPECT_EQ(ledger.at("general_node_histories"), pairs * 2 * n_to * real);
  EXPECT_EQ(ledger.at("general_node_drive_spectrum"), pairs * (n_to + 1) * complex);
  EXPECT_EQ(ledger.at("general_node_product"), pairs * (n_to + 1) * complex);
  EXPECT_EQ(ledger.at("general_node_delayed"), rows * n_to * real);
  EXPECT_EQ(ledger.at("general_delayed_spectrum"),
            problem.general_delayed_spectrum.size() * complex);
  EXPECT_EQ(ledger.at("general_node_response"), kNodes * sizeof(std::size_t));
}

TEST(GeneralModal, TheStateCarriesTheSourceAndItsLedger) {
  auto problem = modal_problem(true, 2, true);
  native::GeneralState state(problem);
  const auto ledger = state.memory_ledger();
  EXPECT_EQ(ledger.at("general_synthesis"),
            problem.general_synthesis.size() * sizeof(Complex));
  // The host maps are released once uploaded; the retained problem is small.
  const auto host_maps =
      (problem.general_synthesis.size() + problem.general_projection.size() +
       problem.general_delayed_spectrum.size()) *
          sizeof(Complex) +
      (problem.general_tensor.size() + problem.general_delayed_tensor.size()) *
          sizeof(double);
  EXPECT_LT(ledger.at("host_problem_arrays"), host_maps);
  std::vector<Complex> field(kModes * problem.n_omega);
  for (std::size_t index = 0; index < field.size(); ++index)
    field[index] = Complex(fill(index, 1.0), fill(index + 1, 1.0));
  state.set_initial(field, {});
  const native::TrialScalars scalars{{0.0, 0.0}, {0.0}, {1.0}, {1.0}, {1.0}, {}, {}, {},
                                     {},         {},    {},    0,     {},    {}, {}};
  const auto first = state.trial_step(0.01, scalars);
  const auto stage = state.stage_to_host(0);
  const double magnitude = std::accumulate(
      stage.begin(), stage.end(), 0.0,
      [](const double sum, const Complex value) { return sum + Kokkos::abs(value); });
  EXPECT_GT(magnitude, 0.0);
  EXPECT_EQ(first.size(), 3 * kModes);
  const auto second = state.trial_step(0.01, scalars);
  expect_partials_close(second, first);
}

// solprop SEP 29 G2: the resident Adams history's rows cross to the host and back
// losslessly, and a malformed row is refused before anything is copied.
TEST(GeneralModal, TheAdamsRowsRoundTripExactly) {
  auto problem = modal_problem(true, 2, false);
  native::GeneralState state(problem);
  std::vector<Complex> field(kModes * problem.n_omega, Complex(1.0, 0.5));
  state.set_initial(field, {});
  constexpr std::size_t kSlots = 4;
  state.adams_allocate(kSlots, true, false);
  EXPECT_EQ(state.adams_row_names(),
            (std::vector<std::string>{"store", "states", "rates"}));
  const std::array<std::size_t, 3> extents{kSlots, kModes, problem.n_omega};
  for (const std::string name : {"store", "states", "rates"}) {
    EXPECT_EQ(state.adams_row_extents(name), extents);
    std::vector<Complex> rows(kSlots * kModes * problem.n_omega);
    for (std::size_t index = 0; index < rows.size(); ++index)
      rows[index] = Complex(fill(index, 1.0), fill(index + name.size(), 1e-300));
    state.adams_rows_from_host(name, rows);
    // Exact: a copy, element by element, with no arithmetic.
    const auto back = state.adams_rows_to_host(name);
    ASSERT_EQ(back.size(), rows.size());
    for (std::size_t index = 0; index < rows.size(); ++index) {
      EXPECT_EQ(back[index].real(), rows[index].real());
      EXPECT_EQ(back[index].imag(), rows[index].imag());
    }
  }
  const auto store = state.adams_rows_to_host("store");
  std::vector<Complex> short_rows(store.size() - 1);
  EXPECT_THROW(state.adams_rows_from_host("store", short_rows), std::invalid_argument);
  EXPECT_EQ(state.adams_rows_to_host("store"), store);
  std::vector<std::complex<double>> direct(store.size());
  state.adams_rows_to_host("store", direct.data(), direct.size());
  for (std::size_t index = 0; index < store.size(); ++index) {
    EXPECT_EQ(direct[index].real(), store[index].real());
    EXPECT_EQ(direct[index].imag(), store[index].imag());
  }
  EXPECT_THROW(state.adams_rows_to_host("store", direct.data(), direct.size() - 1),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(state.adams_row_extents("slopes")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(state.adams_rows_to_host("applied")),
               std::logic_error);
}

TEST(GeneralModal, TheAdamsRowsNeedTheHistory) {
  auto problem = modal_problem(true, 2, false);
  native::GeneralState state(problem);
  std::vector<Complex> field(kModes * problem.n_omega, Complex(1.0, 0.5));
  state.set_initial(field, {});
  EXPECT_THROW(static_cast<void>(state.adams_row_names()), std::logic_error);
  state.adams_allocate(3, false, false);
  EXPECT_EQ(state.adams_row_names(), std::vector<std::string>{"store"});
  EXPECT_THROW(static_cast<void>(state.adams_row_extents("states")), std::logic_error);
}

TEST(GeneralModal, ALinearGuideLeavesTheStageZero) {
  auto problem = modal_problem(true, 2, false);
  std::fill(problem.general_tensor.begin(), problem.general_tensor.end(), 0.0);
  native::GeneralState state(problem);
  std::vector<Complex> field(kModes * problem.n_omega, Complex(1.0, 0.5));
  state.set_initial(field, {});
  const native::TrialScalars scalars{{0.0, 0.0}, {0.0}, {1.0}, {1.0}, {1.0}, {}, {}, {},
                                     {},         {},    {},    0,     {},    {}, {}};
  static_cast<void>(state.trial_step(0.01, scalars));
  // Exact: a zero tensor contracts to exact zeros.
  for (const auto value : state.stage_to_host(0)) {
    EXPECT_EQ(value.real(), 0.0);
    EXPECT_EQ(value.imag(), 0.0);
  }
}

// ---------------------------------------------------------------- moving maps
// A moving guide's synthesis and projection are
// Chebyshev terms along z, `sections x n_deg` of them, combined per stage over the
// trial's transform_weights (kernel schema 14), which are zero outside the section the
// stage lies in.

constexpr std::size_t kSections = 2;
constexpr std::size_t kDegree = 3;
constexpr std::size_t kMapTerms = kSections * kDegree;

// Each element of a combined map is a sum of n_deg products of a weight and a term,
// each part rounded at most twice per term (a product and a sum; once with a fused
// multiply-add): 2 n_deg eps of sum |w| |c|, fixed before the first run.
constexpr double kCombinationBound = 2 * kDegree * kEpsilon;

// The constant problem's maps become term zero of each section's expansion, and the
// other terms fall off as a Chebyshev series' do.
native::Problem moving_problem(const bool real_form, const std::size_t stages = 2) {
  auto problem = modal_problem(real_form, 2, false);
  problem.n_stages = stages;
  problem.general_map_intervals = kSections;
  problem.general_map_n_deg = kDegree;
  const auto synthesis = problem.general_synthesis;
  const auto projection = problem.general_projection;
  const auto maps = synthesis.size();
  problem.general_synthesis.clear();
  problem.general_projection.clear();
  for (std::size_t term = 0; term < kMapTerms; ++term) {
    const double scale = 1.0 / static_cast<double>(1 + term % kDegree);
    for (std::size_t index = 0; index < maps; ++index) {
      const auto shift = 31 * term;
      problem.general_synthesis.push_back(
          term == 0
              ? synthesis[index]
              : Complex(fill(index + shift, scale), fill(index + shift + 5, scale)));
      problem.general_projection.push_back(
          term == 0 ? projection[index]
                    : Complex(fill(index + shift + 9, scale),
                              fill(index + shift + 2, scale)));
    }
  }
  return problem;
}

// Stage zero in section zero, stage one in section one; Chebyshev values at an
// interior point, every one non-zero.
std::vector<double> moving_weights() {
  std::vector<double> weights(2 * kMapTerms, 0.0);
  const double first[kDegree] = {1.0, -0.4, -0.68};
  const double second[kDegree] = {1.0, 0.7, -0.02};
  for (std::size_t term = 0; term < kDegree; ++term) {
    weights[term] = first[term];
    weights[kMapTerms + kDegree + term] = second[term];
  }
  return weights;
}

std::vector<Complex> host_map(const std::vector<Complex>& terms,
                              const std::vector<double>& weights,
                              const std::size_t stage, const bool magnitude) {
  const auto one = terms.size() / kMapTerms;
  std::vector<Complex> result(one);
  for (std::size_t term = 0; term < kMapTerms; ++term) {
    const double weight = weights[stage * kMapTerms + term];
    for (std::size_t index = 0; index < one; ++index) {
      const auto value = terms[term * one + index];
      result[index] += magnitude ? Complex(std::abs(weight) * Kokkos::abs(value), 0.0)
                                 : weight * value;
    }
  }
  return result;
}

std::vector<Complex> to_host(const native::ComplexView& view) {
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  return {host.data(), host.data() + host.size()};
}

void load_scaled(native::GeneralModalSource& source, const native::Problem& problem,
                 const std::vector<Complex>& scaled) {
  auto input = Kokkos::create_mirror_view(source.scaled());
  for (std::size_t m = 0; m < kModes; ++m)
    for (std::size_t w = 0; w < problem.n_omega; ++w)
      input(m, w) = scaled[m * problem.n_omega + w];
  Kokkos::deep_copy(source.scaled(), input);
}

TEST(GeneralModal, MovingMapsAreEachStageSWeightedSumOfItsSectionSTerms) {
  const auto problem = moving_problem(true);
  const auto weights = moving_weights();
  native::GeneralModalSource source(problem);
  ASSERT_TRUE(source.moving());
  source.set_weights(weights);
  for (std::size_t stage = 0; stage < 2; ++stage) {
    source.combine(stage);
    for (const bool synthesis : {true, false}) {
      const auto& terms =
          synthesis ? problem.general_synthesis : problem.general_projection;
      const auto actual = to_host(synthesis ? source.synthesis() : source.projection());
      const auto expected = host_map(terms, weights, stage, false);
      const auto magnitude = host_map(terms, weights, stage, true);
      ASSERT_EQ(actual.size(), expected.size());
      for (std::size_t index = 0; index < actual.size(); ++index)
        EXPECT_LE(Kokkos::abs(actual[index] - expected[index]),
                  kCombinationBound * magnitude[index].real())
            << "stage " << stage << " element " << index;
    }
  }
}

// A stage whose weights select one term (one, then exact zeros) evaluates that term's
// maps exactly: every other product adds an exact zero, under any order or
// contraction, a structural contract. The source is then the constant guide's whose
// maps are that term, evaluated again: equal to the source's rounding bound
// (solprop SEP 25).
TEST(GeneralModal, OneSelectedTermIsTheConstantGuideOfThatTerm) {
  for (const bool real_form : {true, false}) {
    const auto problem = moving_problem(real_form, 1);
    auto constant = modal_problem(real_form, 2, false);
    const auto scaled = scaled_state(constant);
    std::vector<double> weights(kMapTerms, 0.0);
    weights[0] = 1.0;
    native::GeneralModalSource source(problem);
    source.plan();
    source.set_weights(weights);
    source.combine(0);
    for (const bool synthesis : {true, false}) {
      const auto& terms =
          synthesis ? problem.general_synthesis : problem.general_projection;
      const auto maps = to_host(synthesis ? source.synthesis() : source.projection());
      const auto selected = host_map(terms, weights, 0, false);
      ASSERT_EQ(maps.size(), selected.size());
      for (std::size_t index = 0; index < maps.size(); ++index)
        EXPECT_EQ(maps[index], selected[index]) << synthesis << " " << index;
    }
    load_scaled(source, problem, scaled);
    source.evaluate();
    const auto host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, source.source());
    const std::vector<Complex> actual(host.data(), host.data() + host.size());
    expect_close(actual, evaluated(constant, scaled), kInstantaneousBound);
  }
}

// The source of the combined maps against the straightforward host source of the
// host-combined maps: the source's bound, plus the combination's carried through the
// synthesis (the cubic, three times) and the projection (once).
TEST(GeneralModal, AMovingStageSSourceIsTheSourceOfItsCombinedMaps) {
  const auto problem = moving_problem(false);
  const auto weights = moving_weights();
  for (std::size_t stage = 0; stage < 2; ++stage) {
    auto combined = modal_problem(false, 2, false);
    combined.general_synthesis =
        host_map(problem.general_synthesis, weights, stage, false);
    combined.general_projection =
        host_map(problem.general_projection, weights, stage, false);
    const auto scaled = scaled_state(combined);
    native::GeneralModalSource source(problem);
    source.plan();
    source.set_weights(weights);
    source.combine(stage);
    load_scaled(source, problem, scaled);
    source.evaluate();
    const auto host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, source.source());
    std::vector<Complex> actual(host.data(), host.data() + host.size());
    // The terms fall off as 1, 1/2, 1/3 against weights of at most one, so sum |w||c|
    // is at most twice a combined map's own scale: 4 x (3 + 1) x the combination.
    expect_close(actual, reference_source(combined, scaled),
                 kInstantaneousBound + 16 * kCombinationBound);
  }
}

TEST(GeneralModal, MovingMapsRefuseMalformedWeightsAndAMissingTrial) {
  const auto problem = moving_problem(true);
  native::GeneralModalSource source(problem);
  EXPECT_THROW(source.combine(0), std::logic_error);
  EXPECT_THROW(source.set_weights({}), std::invalid_argument);
  auto spanning = moving_weights();
  spanning[kDegree] = 0.5; // stage zero reaches into section one
  EXPECT_THROW(source.set_weights(spanning), std::invalid_argument);
  auto infinite = moving_weights();
  infinite[1] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(source.set_weights(infinite), std::invalid_argument);
  source.set_weights(moving_weights());
  EXPECT_THROW(source.combine(2), std::invalid_argument);
  // A refused load leaves no weights behind it.
  EXPECT_THROW(source.set_weights(spanning), std::invalid_argument);
  EXPECT_THROW(source.combine(0), std::logic_error);
  const auto constant = modal_problem(true, 2, false);
  native::GeneralModalSource fixed(constant);
  EXPECT_FALSE(fixed.moving());
  EXPECT_THROW(fixed.set_weights(moving_weights()), std::invalid_argument);
  fixed.set_weights({});
  fixed.combine(0);
  auto short_terms = moving_problem(true);
  short_terms.general_projection.resize(short_terms.general_projection.size() - 1);
  EXPECT_THROW(native::require_general_modal(short_terms), std::invalid_argument);
  auto no_sections = moving_problem(true);
  no_sections.general_map_intervals = 0;
  EXPECT_THROW(native::require_general_modal(no_sections), std::invalid_argument);
  auto no_stages = moving_problem(true);
  no_stages.n_stages = 0;
  EXPECT_THROW(native::require_general_modal(no_stages), std::invalid_argument);
}

TEST(GeneralModal, MovingLedgerRowsAreTheTermsTheStageMapsAndTheWeights) {
  const auto problem = moving_problem(true);
  native::GeneralModalSource source(problem);
  const auto ledger = source.memory_ledger();
  const auto complex = sizeof(Complex);
  const auto one = problem.general_synthesis.size() / kMapTerms;
  EXPECT_EQ(ledger.at("general_synthesis"), kMapTerms * one * complex);
  EXPECT_EQ(ledger.at("general_projection"), kMapTerms * one * complex);
  EXPECT_EQ(ledger.at("general_stage_maps"), 2 * one * complex);
  EXPECT_EQ(ledger.at("general_map_weights"), 2 * kMapTerms * sizeof(double));
  const auto constant = modal_problem(true, 2, false);
  native::GeneralModalSource fixed(constant);
  EXPECT_EQ(fixed.memory_ledger().at("general_stage_maps"), 0U);
  EXPECT_EQ(fixed.memory_ledger().at("general_map_weights"), 0U);
}

// The state loads a trial's weights and combines each stage before its source; a
// trial without them is refused, and so are weights on a problem with no moving maps.
TEST(GeneralModal, TheStateCombinesAMovingGuideSMapsPerTrial) {
  const auto problem = moving_problem(true, 1);
  native::GeneralState state(problem);
  EXPECT_EQ(state.memory_ledger().at("general_stage_maps"),
            2 * problem.general_synthesis.size() / kMapTerms * sizeof(Complex));
  std::vector<Complex> field(kModes * problem.n_omega);
  for (std::size_t index = 0; index < field.size(); ++index)
    field[index] = Complex(fill(index, 1.0), fill(index + 1, 1.0));
  state.set_initial(field, {});
  native::TrialScalars scalars{{0.0, 0.0}, {0.0}, {1.0}, {1.0}, {1.0}, {}, {}, {},
                               {},         {},    {},    0,     {},    {}, {}};
  EXPECT_THROW(static_cast<void>(state.trial_step(0.01, scalars)),
               std::invalid_argument);
  scalars.transform_weights.assign(kMapTerms, 0.0);
  scalars.transform_weights[kDegree] = 1.0;
  const auto second = state.trial_step(0.01, scalars);
  const auto stage_second = state.stage_to_host(0);
  scalars.transform_weights.assign(kMapTerms, 0.0);
  scalars.transform_weights[0] = 1.0;
  const auto first = state.trial_step(0.01, scalars);
  const auto stage_first = state.stage_to_host(0);
  ASSERT_EQ(stage_first.size(), stage_second.size());
  double moved = 0.0;
  for (std::size_t index = 0; index < stage_first.size(); ++index)
    moved = std::max(moved, Kokkos::abs(stage_first[index] - stage_second[index]));
  EXPECT_GT(moved, 0.0);
  const auto again = state.trial_step(0.01, scalars);
  expect_partials_close(again, first);
  auto constant = modal_problem(true, 2, false);
  native::GeneralState fixed(constant);
  fixed.set_initial(field, {});
  EXPECT_THROW(static_cast<void>(fixed.trial_step(0.01, scalars)),
               std::invalid_argument);
}

} // namespace
