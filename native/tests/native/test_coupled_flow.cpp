// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/core.hpp"
#include "solprop_native/coupled_flow.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace native = solprop_native;
using native::Complex;

constexpr double kEpsilon = std::numeric_limits<double>::epsilon();
constexpr std::size_t kModes = 3;
constexpr std::size_t kOmega = 4;
constexpr std::size_t kStages = 3;
constexpr std::size_t kSections = 2;
constexpr std::size_t kTerms = 4; // two sections of two Chebyshev degrees
constexpr double kStep = 0.01;

// Each element of the trial is a chain of at most three stages of three-by-three
// products and a residual of four terms, about 60 complex products; four real
// operations and a rounding each, relative to the largest element. Fixed before the
// first run.
constexpr double kBound = 256 * kEpsilon;

double fill(const std::size_t index, const double scale) {
  return scale * std::sin(0.7 * static_cast<double>(index) + 0.3);
}

// Bins 1 and 3 carry the block; 0 and 2 have none, and the action is the identity.
const std::vector<std::size_t> kSlot{0, 1, 0, 2};
constexpr std::size_t kActive = 2;

std::size_t block(const std::size_t count) { return count * kActive * kModes * kModes; }

native::Problem flow_problem(const bool residual) {
  native::Problem problem;
  problem.n_t = 4;
  problem.n_to = 8;
  problem.n_omega = kOmega;
  problem.n_stages = kStages;
  problem.n_phi_terms = 1;
  problem.n_prefactor_terms = 1;
  problem.n_modes = kModes;
  problem.family = native::FieldFamily::field_real;
  problem.crop_index = {0, 1, 2, 3};
  problem.active_mask.assign(kOmega, 1U);
  problem.norm_mask.assign(kOmega, 1U);
  problem.omega_window.assign(kOmega, 1.0);
  problem.omega = {1.0, 2.0, 3.0, 4.0};
  problem.dt = 0.25;
  problem.time_window.assign(problem.n_to, 1.0);
  problem.carrier.assign(problem.n_to, Complex(1.0, 0.0));
  // A mode-independent carrier, as the flow requires: exp(dPhi) is a scalar per bin.
  for (std::size_t mode = 0; mode < kModes; ++mode)
    for (std::size_t index = 0; index < kOmega; ++index)
      problem.phi_terms.emplace_back(-0.3, 5.0 * static_cast<double>(index));
  problem.prefactor_terms.assign(kModes * kOmega, Complex(1.0, 0.0));
  // Heun's pair with a third stage at c = 1: a zero first fraction and a repeated
  // whole one, the two cases the action table has.
  problem.tableau_matrix = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0};
  problem.tableau_nodes = {0.0, 1.0, 1.0};
  problem.solution_weights = {0.5, 0.5, 0.0};
  problem.error_weights = {0.5, 0.5, -1.0};
  problem.synthesis_matrix.assign(kModes, 0.0);
  problem.projection_matrix.assign(kModes, 0.0);
  problem.phase_b_engine = true;
  problem.coupled_flow = true;
  problem.flow_active = kActive;
  problem.flow_sections = kSections;
  problem.flow_slot = kSlot;
  if (residual) {
    problem.flow_terms = kTerms;
    for (std::size_t index = 0; index < block(kSections); ++index)
      problem.flow_frozen.emplace_back(fill(index + 3, 2.0), fill(index + 5, 1.5));
    for (std::size_t index = 0; index < block(kTerms); ++index)
      problem.flow_residual_terms.emplace_back(fill(index + 7, 2.5),
                                               fill(index + 11, 1.0));
  }
  return problem;
}

struct Actions {
  std::vector<Complex> forward, inverse;
  std::vector<unsigned char> identity;
};

// Slot 0 is the zero fraction's identity; the others a dense, non-normal block.
Actions actions() {
  Actions result;
  result.identity = {1U, 0U, 0U, 0U};
  for (std::size_t index = 0; index < block(kStages + 1); ++index) {
    result.forward.emplace_back(fill(index + 19, 0.6), fill(index + 23, 0.4));
    result.inverse.emplace_back(fill(index + 29, 0.5), fill(index + 31, 0.3));
  }
  return result;
}

native::TrialScalars scalars(const bool residual, const std::size_t section) {
  native::TrialScalars result{
      std::vector<double>((kStages + 1), kStep),
      std::vector<double>(kStages, kStep),
      std::vector<double>(kStages, 1.0),
      std::vector<double>(kStages, 1.0),
      std::vector<double>(kStages, 1.0),
      {},
      {},
      {},
      {},
      {},
      {},
      section,
      {},
      {},
      {},
  };
  if (residual)
    for (std::size_t index = 0; index < kStages * kTerms; ++index)
      result.flow_weights.push_back(fill(index + 37, 1.0));
  return result;
}

std::vector<Complex> initial() {
  std::vector<Complex> field;
  for (std::size_t index = 0; index < kModes * kOmega; ++index)
    field.emplace_back(fill(index + 41, 1.0), fill(index + 43, 1.0));
  return field;
}

Complex cexp(const Complex value) {
  const double scale = std::exp(value.real());
  return {scale * std::cos(value.imag()), scale * std::sin(value.imag())};
}

// solprop `NumpyRefBackend._trial_step` on a linear coupled flow, with its
// `_channels` and `_residual_drive`, written directly: every stage input carried
// forward, the source (the residual alone, since no response is declared) carried
// back, the accepted step carried by the whole action.
std::vector<Complex> reference_step(const native::Problem& problem, const Actions& flow,
                                    const native::TrialScalars& trial) {
  const auto at = [](const std::vector<Complex>& values, const std::size_t slot,
                     const std::size_t bin, const std::size_t row,
                     const std::size_t column) {
    return values[((slot * kActive + bin) * kModes + row) * kModes + column];
  };
  const auto current = initial();
  std::vector<std::vector<Complex>> stages;
  std::vector<Complex> solution(current.size());
  const auto phase = [&](const std::size_t stage, const std::size_t mode,
                         const std::size_t index) {
    return trial.phi_delta[stage] * problem.phi_terms[mode * kOmega + index];
  };
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    std::vector<Complex> sum(current.size());
    for (std::size_t previous = 0; previous < stage; ++previous)
      for (std::size_t element = 0; element < sum.size(); ++element)
        sum[element] += problem.tableau_matrix[stage * kStages + previous] *
                        stages[previous][element];
    std::vector<Complex> lab(current.size());
    std::vector<Complex> drive(current.size());
    for (std::size_t index = 0; index < kOmega; ++index) {
      const auto slot = kSlot[index];
      for (std::size_t row = 0; row < kModes; ++row) {
        Complex state = current[row * kOmega + index];
        Complex carried = sum[row * kOmega + index];
        if (slot != 0 && flow.identity[stage] == 0U) {
          state = Complex{};
          carried = Complex{};
          for (std::size_t column = 0; column < kModes; ++column) {
            const auto action = at(flow.forward, stage, slot - 1, row, column);
            state += action * current[column * kOmega + index];
            carried += action * sum[column * kOmega + index];
          }
        }
        lab[row * kOmega + index] =
            cexp(phase(stage, row, index)) * (state + kStep * carried);
      }
      if (slot == 0 || problem.flow_terms == 0)
        continue;
      for (std::size_t row = 0; row < kModes; ++row) {
        Complex residual{};
        for (std::size_t column = 0; column < kModes; ++column) {
          Complex combined{};
          for (std::size_t term = 0; term < kTerms; ++term)
            combined +=
                trial.flow_weights[stage * kTerms + term] *
                problem.flow_residual_terms
                    [((term * kActive + slot - 1) * kModes + row) * kModes + column];
          combined -=
              problem.flow_frozen[((trial.flow_section * kActive + slot - 1) * kModes +
                                   row) *
                                      kModes +
                                  column];
          residual += combined * lab[column * kOmega + index];
        }
        drive[row * kOmega + index] = cexp(-phase(stage, row, index)) * residual;
      }
    }
    std::vector<Complex> stage_value = drive;
    if (flow.identity[stage] == 0U)
      for (std::size_t index = 0; index < kOmega; ++index) {
        const auto slot = kSlot[index];
        if (slot == 0)
          continue;
        for (std::size_t row = 0; row < kModes; ++row) {
          Complex value{};
          for (std::size_t column = 0; column < kModes; ++column)
            value += at(flow.inverse, stage, slot - 1, row, column) *
                     drive[column * kOmega + index];
          stage_value[row * kOmega + index] = value;
        }
      }
    for (std::size_t element = 0; element < solution.size(); ++element)
      solution[element] += problem.solution_weights[stage] * stage_value[element];
    stages.push_back(stage_value);
  }
  std::vector<Complex> advanced(current.size());
  for (std::size_t index = 0; index < kOmega; ++index) {
    const auto slot = kSlot[index];
    for (std::size_t row = 0; row < kModes; ++row) {
      Complex state = current[row * kOmega + index];
      Complex carried = solution[row * kOmega + index];
      if (slot != 0) {
        state = Complex{};
        carried = Complex{};
        for (std::size_t column = 0; column < kModes; ++column) {
          const auto action = at(flow.forward, kStages, slot - 1, row, column);
          state += action * current[column * kOmega + index];
          carried += action * solution[column * kOmega + index];
        }
      }
      advanced[row * kOmega + index] =
          cexp(phase(kStages, row, index)) * (state + kStep * carried);
    }
  }
  return advanced;
}

std::vector<Complex> native_step(const native::Problem& problem, const Actions& flow,
                                 const native::TrialScalars& trial) {
  native::GeneralState state(problem);
  state.set_initial(initial(), {});
  state.set_flow_actions(flow.forward, flow.inverse, flow.identity);
  static_cast<void>(state.trial_step(kStep, trial));
  state.finish(kStep);
  return state.field_to_host();
}

double largest(const std::vector<Complex>& values) {
  double result = 0.0;
  for (const auto& value : values)
    result = std::max(result, Kokkos::abs(value));
  return result;
}

double departure(const std::vector<Complex>& left, const std::vector<Complex>& right) {
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index)
    result = std::max(result, Kokkos::abs(left[index] - right[index]));
  return result;
}

} // namespace

TEST(CoupledFlow, ConstantBlockCarriesTheStepByItsWholeAction) {
  const auto problem = flow_problem(false);
  const auto flow = actions();
  const auto trial = scalars(false, 0);
  const auto expected = reference_step(problem, flow, trial);
  const auto actual = native_step(problem, flow, trial);
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_LE(departure(actual, expected), kBound * largest(expected));
  // Off the block the action is the identity: only the carrier moves those bins.
  const auto start = initial();
  for (std::size_t row = 0; row < kModes; ++row) {
    const auto element = row * kOmega;
    EXPECT_NE(actual[element + 1],
              start[element + 1] * cexp(kStep * problem.phi_terms[1]));
    EXPECT_LE(Kokkos::abs(actual[element] -
                          start[element] * cexp(kStep * problem.phi_terms[0])),
              kBound * largest(expected));
  }
}

TEST(CoupledFlow, ResidualStagesAreTheReferencesInEitherSection) {
  for (const std::size_t section : {std::size_t{0}, std::size_t{1}}) {
    const auto problem = flow_problem(true);
    const auto flow = actions();
    const auto trial = scalars(true, section);
    const auto expected = reference_step(problem, flow, trial);
    const auto actual = native_step(problem, flow, trial);
    EXPECT_LE(departure(actual, expected), kBound * largest(expected)) << section;
    // The residual moved the answer: the same step without it differs.
    const auto without = reference_step(flow_problem(false), flow, scalars(false, 0));
    EXPECT_GT(departure(expected, without), 1e-3 * largest(expected));
  }
}

// The same trial twice is the same discrete computation: held to the step's rounding
// bound, not to its bits (solprop SEP 25).
TEST(CoupledFlow, RepeatedTrialsAgreeToRounding) {
  const auto problem = flow_problem(true);
  const auto flow = actions();
  const auto trial = scalars(true, 1);
  const auto first = native_step(problem, flow, trial);
  const auto second = native_step(problem, flow, trial);
  ASSERT_EQ(second.size(), first.size());
  EXPECT_LE(departure(second, first), kBound * largest(first));
}

TEST(CoupledFlow, NoStageIsReusedAcrossSteps) {
  auto problem = flow_problem(false);
  problem.first_same_as_last = true;
  const auto flow = actions();
  native::GeneralState state(problem);
  state.set_initial(initial(), {});
  state.set_flow_actions(flow.forward, flow.inverse, flow.identity);
  static_cast<void>(state.trial_step(kStep, scalars(false, 0)));
  state.finish(kStep);
  state.promote_fsal();
  EXPECT_FALSE(state.fsal_ready());
}

TEST(CoupledFlow, MalformedDataAreRefusedBeforeAllocation) {
  auto slot = flow_problem(false);
  slot.flow_slot = {0, 2, 0, 1};
  EXPECT_THROW(native::GeneralState{slot}, std::invalid_argument);
  auto count = flow_problem(false);
  count.flow_active = 3;
  EXPECT_THROW(native::GeneralState{count}, std::invalid_argument);
  auto pol = flow_problem(false);
  pol.n_pol_state = 2;
  EXPECT_THROW(native::GeneralState{pol}, std::invalid_argument);
  auto noise = flow_problem(false);
  noise.modified_noise = true;
  EXPECT_THROW(native::GeneralState{noise}, std::invalid_argument);
  auto terms = flow_problem(true);
  terms.flow_residual_terms.pop_back();
  EXPECT_THROW(native::GeneralState{terms}, std::invalid_argument);
  auto stray = flow_problem(false);
  stray.coupled_flow = false;
  EXPECT_THROW(native::GeneralState{stray}, std::invalid_argument);
  auto nonfinite = flow_problem(true);
  nonfinite.flow_frozen[0] = Complex(std::numeric_limits<double>::quiet_NaN(), 0.0);
  EXPECT_THROW(native::GeneralState{nonfinite}, std::invalid_argument);
}

// A coherent wall beside a mode set resolving E_x and E_y. The flow
// contracts the channel axis of the scalar state and never reads the synthesis, so a
// transform of two components is admitted and the linear step is the scalar
// transform's, to the step's rounding bound (solprop SEP 25).
TEST(CoupledFlow, ATwoComponentSynthesisLeavesTheFlowAsItIs) {
  for (const bool residual : {false, true}) {
    const auto scalar = flow_problem(residual);
    auto vector = scalar;
    vector.n_pol_transform = 2;
    vector.synthesis_matrix.assign(2 * kModes, 0.0);
    vector.projection_matrix.assign(2 * kModes, 0.0);
    const auto flow = actions();
    const auto trial = scalars(residual, 1);
    const auto expected = native_step(scalar, flow, trial);
    const auto actual = native_step(vector, flow, trial);
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_LE(departure(actual, expected), kBound * largest(expected)) << residual;
  }
}

TEST(CoupledFlow, ActionsAndTrialsAreChecked) {
  const auto problem = flow_problem(true);
  native::GeneralState state(problem);
  state.set_initial(initial(), {});
  EXPECT_THROW(static_cast<void>(state.trial_step(kStep, scalars(true, 0))),
               std::logic_error);
  auto flow = actions();
  flow.identity.back() = 1U;
  EXPECT_THROW(state.set_flow_actions(flow.forward, flow.inverse, flow.identity),
               std::invalid_argument);
  flow = actions();
  flow.forward.pop_back();
  EXPECT_THROW(state.set_flow_actions(flow.forward, flow.inverse, flow.identity),
               std::invalid_argument);
  flow = actions();
  state.set_flow_actions(flow.forward, flow.inverse, flow.identity);
  EXPECT_THROW(static_cast<void>(state.trial_step(kStep, scalars(true, kSections))),
               std::invalid_argument);
  auto short_weights = scalars(true, 0);
  short_weights.flow_weights.pop_back();
  EXPECT_THROW(static_cast<void>(state.trial_step(kStep, short_weights)),
               std::invalid_argument);
}

TEST(CoupledFlow, LedgerCountsTheFlowViewsAndIsZeroElsewhere) {
  const auto problem = flow_problem(true);
  native::GeneralState state(problem);
  const auto ledger = state.memory_ledger();
  const auto complex_bytes = sizeof(Complex);
  EXPECT_EQ(ledger.at("flow_forward"), block(kStages + 1) * complex_bytes);
  EXPECT_EQ(ledger.at("flow_inverse"), block(kStages + 1) * complex_bytes);
  EXPECT_EQ(ledger.at("flow_frozen"), block(kSections) * complex_bytes);
  EXPECT_EQ(ledger.at("flow_residual_terms"), block(kTerms) * complex_bytes);
  EXPECT_EQ(ledger.at("flow_weights"), kStages * kTerms * sizeof(double));
  EXPECT_EQ(ledger.at("flow_active_terms"),
            (kStages * kTerms + kStages) * sizeof(std::size_t));
  EXPECT_EQ(ledger.at("stage_phase"), kModes * kOmega * complex_bytes);
  EXPECT_EQ(ledger.at("flow_slot"), kOmega * sizeof(std::size_t));
  for (const char* name : {"flow_input", "flow_lab", "flow_drive", "flow_carried_state",
                           "flow_carried_solution", "flow_carried_error"})
    EXPECT_EQ(ledger.at(name), kModes * kOmega * complex_bytes) << name;
  auto plain = flow_problem(false);
  plain.coupled_flow = false;
  plain.flow_active = 0;
  plain.flow_sections = 0;
  plain.flow_slot.clear();
  native::GeneralState other(plain);
  for (const auto& [name, bytes] : native::CoupledFlowAction::empty_ledger())
    EXPECT_EQ(other.memory_ledger().at(name), 0U) << name;
}

// The residual sums a stage's terms of non-zero
// weight only. A zero weight adds an exact zero to a sum that starts at +0 and so is
// never -0, so a flow carrying extra terms whose weights are all zero is exactly the
// flow without them, whatever those terms hold: an explicit no-op, a structural
// contract that stays exact under any order or contraction (solprop SEP 25).
TEST(CoupledFlow, ZeroWeightTermsAreSkippedWithoutMovingABit) {
  const auto problem = flow_problem(true);
  const auto flow = actions();
  const auto trial = scalars(true, 1);
  auto padded = problem;
  padded.flow_terms = 2 * kTerms;
  for (std::size_t index = 0; index < block(kTerms); ++index)
    padded.flow_residual_terms.emplace_back(fill(index + 53, 9.0),
                                            fill(index + 59, -7.0));
  auto padded_trial = trial;
  padded_trial.flow_weights.clear();
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    for (std::size_t term = 0; term < kTerms; ++term)
      padded_trial.flow_weights.push_back(trial.flow_weights[stage * kTerms + term]);
    for (std::size_t term = 0; term < kTerms; ++term)
      padded_trial.flow_weights.push_back(0.0);
  }
  EXPECT_EQ(native_step(padded, flow, padded_trial), native_step(problem, flow, trial));
}

TEST(CoupledFlow, ABatchRefusesAFlowMember) {
  auto state = std::make_shared<native::GeneralState>(flow_problem(false));
  state->set_initial(initial(), {});
  EXPECT_THROW(native::GeneralBatch({state}), std::invalid_argument);
}

// The engine forms the actions itself, from each section's
// eigenbasis `W exp(+-l Lambda) Y` or, on the bins routed away from it, by solprop's
// reference scaling, series and squaring of `l T` with the back-transform `Z F Z^H`.
namespace {

using Matrix = std::vector<Complex>; // row-major kModes x kModes

Matrix multiply(const Matrix& left, const Matrix& right) {
  Matrix result(kModes * kModes);
  for (std::size_t row = 0; row < kModes; ++row)
    for (std::size_t column = 0; column < kModes; ++column) {
      Complex total{};
      for (std::size_t inner = 0; inner < kModes; ++inner)
        total += left[row * kModes + inner] * right[inner * kModes + column];
      result[row * kModes + column] = total;
    }
  return result;
}

Matrix adjoint(const Matrix& value) {
  Matrix result(kModes * kModes);
  for (std::size_t row = 0; row < kModes; ++row)
    for (std::size_t column = 0; column < kModes; ++column)
      result[row * kModes + column] = Kokkos::conj(value[column * kModes + row]);
  return result;
}

// A unitary Z: two plane rotations with phases, different per block.
Matrix unitary(const std::size_t seed) {
  const double first = 0.3 + 0.2 * static_cast<double>(seed);
  const double second = 0.7 - 0.1 * static_cast<double>(seed);
  Matrix rotate(kModes * kModes);
  Matrix turn(kModes * kModes);
  for (std::size_t index = 0; index < kModes; ++index) {
    rotate[index * kModes + index] = Complex(1.0, 0.0);
    turn[index * kModes + index] = Complex(1.0, 0.0);
  }
  const Complex phase(std::cos(first), std::sin(first));
  rotate[0] = std::cos(first);
  rotate[1] = -std::sin(first) * Kokkos::conj(phase);
  rotate[kModes] = std::sin(first) * phase;
  rotate[kModes + 1] = std::cos(first);
  turn[kModes + 1] = std::cos(second);
  turn[kModes + 2] = std::sin(second);
  turn[2 * kModes + 1] = -std::sin(second);
  turn[2 * kModes + 2] = std::cos(second);
  return multiply(rotate, turn);
}

// Section 0's blocks are diagonal (normal); section 1's carry an upper coupling.
Matrix triangular(const std::size_t section, const std::size_t bin) {
  Matrix result(kModes * kModes);
  for (std::size_t mode = 0; mode < kModes; ++mode)
    result[mode * kModes + mode] =
        Complex(-2.0 - static_cast<double>(mode) - static_cast<double>(bin),
                30.0 * static_cast<double>(mode + 1) + 7.0 * static_cast<double>(bin));
  if (section == 1) {
    result[1] = Complex(3.0, -1.0);
    result[2] = Complex(-1.5, 2.0);
    result[kModes + 2] = Complex(2.5, 0.5);
  }
  return result;
}

// The eigenbasis solprop's `schur_eigenbasis` forms: V by back substitution, its unit
// triangular inverse, `W = Z V` and `Y = V^-1 Z^H`.
std::pair<Matrix, Matrix> eigenbasis(const Matrix& vectors, const Matrix& upper) {
  Matrix right_vectors(kModes * kModes);
  Matrix inverse(kModes * kModes);
  for (std::size_t index = 0; index < kModes; ++index) {
    right_vectors[index * kModes + index] = Complex(1.0, 0.0);
    inverse[index * kModes + index] = Complex(1.0, 0.0);
  }
  for (std::size_t column = 0; column < kModes; ++column)
    for (std::size_t row = column; row-- > 0;) {
      Complex total{};
      for (std::size_t inner = row + 1; inner <= column; ++inner)
        total += upper[row * kModes + inner] * right_vectors[inner * kModes + column];
      right_vectors[row * kModes + column] =
          total / (upper[column * kModes + column] - upper[row * kModes + row]);
    }
  for (std::size_t column = 0; column < kModes; ++column)
    for (std::size_t row = column; row-- > 0;) {
      Complex total{};
      for (std::size_t inner = row + 1; inner <= column; ++inner)
        total += right_vectors[row * kModes + inner] * inverse[inner * kModes + column];
      inverse[row * kModes + column] = -total;
    }
  return {multiply(vectors, right_vectors), multiply(inverse, adjoint(vectors))};
}

// The flow problem with its eigenbasis loaded; `fallback` routes every bin to
// scaling and squaring instead.
native::Problem forming_problem(const bool fallback) {
  auto problem = flow_problem(false);
  std::size_t route = 0;
  for (std::size_t section = 0; section < kSections; ++section)
    for (std::size_t bin = 0; bin < kActive; ++bin) {
      const auto vectors = unitary(section * kActive + bin);
      const auto upper = triangular(section, bin);
      const auto [left, right] = eigenbasis(vectors, upper);
      for (std::size_t mode = 0; mode < kModes; ++mode)
        problem.flow_eigenvalues.push_back(upper[mode * kModes + mode]);
      if (fallback) {
        problem.flow_route.push_back(++route);
        problem.flow_left.insert(problem.flow_left.end(), kModes * kModes, Complex{});
        problem.flow_right.insert(problem.flow_right.end(), kModes * kModes, Complex{});
        problem.flow_fallback_vectors.insert(problem.flow_fallback_vectors.end(),
                                             vectors.begin(), vectors.end());
        problem.flow_fallback_triangular.insert(problem.flow_fallback_triangular.end(),
                                                upper.begin(), upper.end());
      } else {
        problem.flow_route.push_back(0);
        problem.flow_left.insert(problem.flow_left.end(), left.begin(), left.end());
        problem.flow_right.insert(problem.flow_right.end(), right.begin(), right.end());
      }
    }
  return problem;
}

// Distinct lengths, a zero one and a repeated whole one.
const std::vector<double> kFractions{0.0, 0.5, 1.0, 1.0};
constexpr double kLength = 0.02; // |l lambda| up to 1.9: the squarings are exercised

struct Formed {
  std::vector<Complex> forward, inverse;
  std::vector<double> measured;
};

Formed form(const native::Problem& problem, const std::size_t section) {
  native::GeneralState state(problem);
  Formed result;
  result.measured = state.form_flow_actions(section, kLength, kFractions);
  result.forward = state.flow_actions_to_host(false);
  result.inverse = state.flow_actions_to_host(true);
  return result;
}

Matrix block_at(const std::vector<Complex>& values, const std::size_t slot,
                const std::size_t bin) {
  const auto start = ((slot * kActive) + bin) * kModes * kModes;
  return {values.begin() + static_cast<std::ptrdiff_t>(start),
          values.begin() + static_cast<std::ptrdiff_t>(start + kModes * kModes)};
}

double largest_entry(const Matrix& value) {
  double result = 0.0;
  for (const auto& entry : value)
    result = std::max(result, Kokkos::abs(entry));
  return result;
}

// Both routes against each other and each against the identity `U(l) U(-l) = I`:
// solprop's formation bounds at `||l T|| <= 2`, two squarings and kappa <= 3 are
// below 4096 eps of the action here. Fixed before the first run.
constexpr double kFormationBound = 4096 * kEpsilon;

} // namespace

TEST(CoupledFlowFormation, BothRoutesFormTheSameActions) {
  for (const std::size_t section : {std::size_t{0}, std::size_t{1}}) {
    const auto eigen = form(forming_problem(false), section);
    const auto squared = form(forming_problem(true), section);
    for (std::size_t slot = 1; slot < kFractions.size(); ++slot)
      for (std::size_t bin = 0; bin < kActive; ++bin) {
        const auto forward = block_at(eigen.forward, slot, bin);
        const auto inverse = block_at(eigen.inverse, slot, bin);
        const auto scale = largest_entry(forward) * largest_entry(inverse);
        EXPECT_LE(departure(forward, block_at(squared.forward, slot, bin)),
                  kFormationBound * largest_entry(forward))
            << section << slot << bin;
        EXPECT_LE(departure(inverse, block_at(squared.inverse, slot, bin)),
                  kFormationBound * largest_entry(inverse))
            << section << slot << bin;
        auto identity = multiply(forward, inverse);
        for (std::size_t mode = 0; mode < kModes; ++mode)
          identity[mode * kModes + mode] -= Complex(1.0, 0.0);
        EXPECT_LE(largest_entry(identity), kFormationBound * scale) << section << slot;
      }
    // A repeated fraction is the earlier slot's matrix, copied.
    EXPECT_EQ(block_at(eigen.forward, 3, 1), block_at(eigen.forward, 2, 1));
    EXPECT_EQ(block_at(squared.inverse, 3, 0), block_at(squared.inverse, 2, 0));
    for (std::size_t bin = 0; bin < kActive; ++bin)
      EXPECT_LE(std::abs(eigen.measured[bin] / squared.measured[bin] - 1.0),
                kFormationBound);
  }
}

TEST(CoupledFlowFormation, ANormalBlockMeasuresItsExactConditioning) {
  const auto eigen = form(forming_problem(false), 0);
  for (std::size_t bin = 0; bin < kActive; ++bin) {
    // Section 0 is normal: norm(U) = max |e^{l lambda}| and norm(U^-1) its inverse
    // counterpart, so the measured product is exp(l (max - min) Re lambda).
    const auto upper = triangular(0, bin);
    double high = -1e300;
    double low = 1e300;
    for (std::size_t mode = 0; mode < kModes; ++mode) {
      high = std::max(high, upper[mode * kModes + mode].real());
      low = std::min(low, upper[mode * kModes + mode].real());
    }
    EXPECT_NEAR(eigen.measured[bin] / std::exp(kLength * (high - low)), 1.0,
                64 * kModes * kEpsilon)
        << bin;
  }
}

TEST(CoupledFlowFormation, FormedActionsCarryATrialAsLoadedOnesDo) {
  const auto problem = forming_problem(false);
  const auto formed = form(problem, 1);
  Actions flow;
  flow.forward = formed.forward;
  flow.inverse = formed.inverse;
  flow.identity = {1U, 0U, 0U, 0U};
  const auto trial = scalars(false, 1);
  native::GeneralState state(problem);
  state.set_initial(initial(), {});
  static_cast<void>(state.form_flow_actions(1, kLength, kFractions));
  static_cast<void>(state.trial_step(kStep, trial));
  state.finish(kStep);
  // The same trial on the same actions, formed here and loaded there: rounding only.
  const auto loaded = native_step(problem, flow, trial);
  const auto formed_step = state.field_to_host();
  ASSERT_EQ(formed_step.size(), loaded.size());
  EXPECT_LE(departure(formed_step, loaded), kBound * largest(loaded));
}

TEST(CoupledFlowFormation, FormationAndItsDataAreChecked) {
  native::GeneralState plain(flow_problem(false));
  EXPECT_THROW(static_cast<void>(plain.form_flow_actions(0, kLength, kFractions)),
               std::logic_error);
  EXPECT_THROW(static_cast<void>(plain.flow_actions_to_host(false)), std::logic_error);
  native::GeneralState state(forming_problem(false));
  EXPECT_THROW(
      static_cast<void>(state.form_flow_actions(kSections, kLength, kFractions)),
      std::invalid_argument);
  EXPECT_THROW(static_cast<void>(state.form_flow_actions(0, kLength, {0.0, 1.0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(state.form_flow_actions(0, -kLength, kFractions)),
               std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(state.form_flow_actions(0, kLength, {0.0, 0.5, 2.0, 1.0})),
      std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(state.form_flow_actions(0, kLength, {0.0, 0.5, 1.0, 0.0})),
      std::invalid_argument);
  auto numbering = forming_problem(true);
  numbering.flow_route[0] = 2;
  EXPECT_THROW(native::GeneralState{numbering}, std::invalid_argument);
  auto fallback = forming_problem(true);
  fallback.flow_fallback_vectors.pop_back();
  EXPECT_THROW(native::GeneralState{fallback}, std::invalid_argument);
  auto partial = forming_problem(false);
  partial.flow_right.pop_back();
  EXPECT_THROW(native::GeneralState{partial}, std::invalid_argument);
  auto nonfinite = forming_problem(false);
  nonfinite.flow_eigenvalues[0] = Complex(std::numeric_limits<double>::infinity(), 0.0);
  EXPECT_THROW(native::GeneralState{nonfinite}, std::invalid_argument);
  auto stray = flow_problem(false);
  stray.coupled_flow = false;
  stray.flow_active = 0;
  stray.flow_sections = 0;
  stray.flow_slot.clear();
  stray.flow_route = {0};
  EXPECT_THROW(native::GeneralState{stray}, std::invalid_argument);
}

TEST(CoupledFlowFormation, LedgerCountsTheFormationViews) {
  const auto complex_bytes = sizeof(Complex);
  for (const bool fallback : {false, true}) {
    native::GeneralState state(forming_problem(fallback));
    const auto ledger = state.memory_ledger();
    EXPECT_EQ(ledger.at("flow_eigenbasis"),
              (2 * block(kSections) + kSections * kActive * kModes) * complex_bytes +
                  kSections * kActive * sizeof(std::size_t));
    const auto fallbacks = fallback ? kSections * kActive : 0;
    EXPECT_EQ(ledger.at("flow_fallback"),
              5 * fallbacks * kModes * kModes * complex_bytes);
    EXPECT_EQ(ledger.at("flow_formation"),
              (2 * kActive * kModes + kActive * kModes * kModes) * complex_bytes +
                  (kStages + 1 + kActive) * sizeof(double));
  }
  native::GeneralState plain(flow_problem(false));
  EXPECT_EQ(plain.memory_ledger().at("flow_eigenbasis"), 0U);
  EXPECT_EQ(plain.memory_ledger().at("flow_formation"), 0U);
}
