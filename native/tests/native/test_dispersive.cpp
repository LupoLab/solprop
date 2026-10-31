// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/core.hpp"
#include "solprop_native/dispersive.hpp"

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace {

namespace native = solprop_native;
using native::Complex;

constexpr double kEpsilon = std::numeric_limits<double>::epsilon();

double distance(const Complex left, const Complex right) {
  return Kokkos::abs(left - right);
}

// The general isotropic contraction of three two-component fields, written out as the
// tensor sum `T_ijkl x_j y_k z_l` with `T = (d_ij d_kl + d_ik d_jl + d_il d_jk) / 3`.
void tensor_sum(const Complex (&x)[2], const Complex (&y)[2], const Complex (&z)[2],
                Complex (&out)[2]) {
  for (int i = 0; i < 2; ++i) {
    Complex value{};
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k)
        for (int l = 0; l < 2; ++l) {
          const double weight = ((i == j && k == l) ? 1.0 : 0.0) +
                                ((i == k && j == l) ? 1.0 : 0.0) +
                                ((i == l && j == k) ? 1.0 : 0.0);
          value += (weight / 3.0) * x[j] * y[k] * z[l];
        }
    out[i] = value;
  }
}

TEST(DispersiveBodies, TheContractionIsTheIsotropicTensorSum) {
  const Complex x[2] = {{0.3, -1.1}, {0.7, 0.2}};
  const Complex y[2] = {{-0.4, 0.5}, {1.3, -0.6}};
  const Complex z[2] = {{0.9, 0.1}, {-0.2, 0.8}};
  Complex expected[2];
  tensor_sum(x, y, z, expected);
  Complex out0, out1;
  native::dispersive_isotropic(x[0], x[1], y[0], y[1], z[0], z[1], out0, out1);
  EXPECT_LT(distance(out0, expected[0]), 16 * kEpsilon);
  EXPECT_LT(distance(out1, expected[1]), 16 * kEpsilon);
  // One component: the contraction is the plain product.
  native::dispersive_isotropic(x[0], Complex{}, y[0], Complex{}, z[0], Complex{}, out0,
                               out1);
  EXPECT_LT(distance(out0, x[0] * y[0] * z[0]), 8 * kEpsilon);
  EXPECT_EQ(out1, Complex{});
}

TEST(DispersiveBodies, TheScalarClosedFormsAreTheGeneralGroups) {
  // The closed forms of `(u, u, u)`, `(u, u, f)` and `(u, f, f)` of one component
  // against the general groups of a field whose second component is zero.
  const Complex u{0.6, -0.8};
  const Complex f{-1.2, 0.35};
  const Complex carrier{std::cos(0.7), std::sin(0.7)};
  const Complex zero{};
  Complex general, unused, harmonic;
  native::dispersive_conserving(u, zero, u, zero, u, zero, general, unused);
  EXPECT_LT(distance(native::dispersive_conserving_self(u), general), 16 * kEpsilon);
  native::dispersive_conserving(u, zero, u, zero, f, zero, general, unused);
  EXPECT_LT(distance(native::dispersive_conserving_pair(u, f), general), 16 * kEpsilon);
  native::dispersive_conserving(u, zero, f, zero, f, zero, general, unused);
  EXPECT_LT(distance(native::dispersive_conserving_cross(u, f), general),
            16 * kEpsilon);
  native::dispersive_harmonic(u, zero, u, zero, u, zero, carrier, harmonic, unused);
  EXPECT_LT(distance(native::dispersive_harmonic_self(u, carrier), harmonic),
            16 * kEpsilon);
  native::dispersive_harmonic(u, zero, u, zero, f, zero, carrier, harmonic, unused);
  EXPECT_LT(distance(native::dispersive_harmonic_pair(u, f, carrier), harmonic),
            16 * kEpsilon);
  native::dispersive_harmonic(u, zero, f, zero, f, zero, carrier, harmonic, unused);
  EXPECT_LT(distance(native::dispersive_harmonic_cross(u, f, carrier), harmonic),
            16 * kEpsilon);
  // The real products are the reference's groupings. Exact: the operands are
  // dyadic, so every product is representable and no grouping rounds.
  EXPECT_EQ(native::dispersive_real_self(1.5), (1.5 * 1.5) * 1.5);
  EXPECT_EQ(native::dispersive_real_pair(1.5, -0.25), (1.5 * 1.5) * -0.25);
  EXPECT_EQ(native::dispersive_real_cross(1.5, -0.25), (1.5 * -0.25) * -0.25);
}

TEST(DispersiveBodies, TheConservingGroupIsTheQuartetsWithTwoPositiveLegs) {
  // `|u|^2 u` contracted isotropically: `(u . u*) u` twice and `(u . u) u*` once, the
  // number-conserving group of `(u, u, u)` (SEP 6 Phase 0).
  const Complex u0{0.4, 0.9}, u1{-0.7, 0.2};
  Complex out0, out1;
  native::dispersive_conserving(u0, u1, u0, u1, u0, u1, out0, out1);
  const Complex square = u0 * u0 + u1 * u1;
  const double modulus =
      Kokkos::abs(u0) * Kokkos::abs(u0) + Kokkos::abs(u1) * Kokkos::abs(u1);
  EXPECT_LT(distance(out0, 2.0 * modulus * u0 + square * Kokkos::conj(u0)),
            32 * kEpsilon);
  EXPECT_LT(distance(out1, 2.0 * modulus * u1 + square * Kokkos::conj(u1)),
            32 * kEpsilon);
}

// The general isotropic contraction of three three-component fields as the tensor sum.
void tensor_sum3(const Complex (&x)[3], const Complex (&y)[3], const Complex (&z)[3],
                 Complex (&out)[3]) {
  for (int i = 0; i < 3; ++i) {
    Complex value{};
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l) {
          const double weight = ((i == j && k == l) ? 1.0 : 0.0) +
                                ((i == k && j == l) ? 1.0 : 0.0) +
                                ((i == l && j == k) ? 1.0 : 0.0);
          value += (weight / 3.0) * x[j] * y[k] * z[l];
        }
    out[i] = value;
  }
}

TEST(DispersiveBodies, TheThreeComponentContractionIsTheTensorSum) {
  // The general modal route's three Cartesian components (solprop
  // `dispersive_isotropic3`); two components are the two-component body, to the
  // rounding of its sixteen products (solprop SEP 25).
  const Complex x[3] = {{0.3, -1.1}, {0.7, 0.2}, {-0.5, 0.4}};
  const Complex y[3] = {{-0.4, 0.5}, {1.3, -0.6}, {0.2, 0.9}};
  const Complex z[3] = {{0.9, 0.1}, {-0.2, 0.8}, {0.6, -0.3}};
  Complex expected[3];
  tensor_sum3(x, y, z, expected);
  Complex out[3];
  native::dispersive_isotropic_n(x, y, z, out);
  for (int c = 0; c < 3; ++c)
    EXPECT_LT(distance(out[c], expected[c]), 32 * kEpsilon) << c;
  const Complex x2[2] = {x[0], x[1]}, y2[2] = {y[0], y[1]}, z2[2] = {z[0], z[1]};
  Complex two[2];
  native::dispersive_isotropic_n(x2, y2, z2, two);
  Complex out0, out1;
  native::dispersive_isotropic(x[0], x[1], y[0], y[1], z[0], z[1], out0, out1);
  // Each body is within 16 eps of the tensor sum (the test above), so 32 eps of the
  // other.
  EXPECT_LT(distance(two[0], out0), 32 * kEpsilon);
  EXPECT_LT(distance(two[1], out1), 32 * kEpsilon);
  // The conserving group: the conjugated contractions summed, as the two-component
  // one; each within 32 eps of its quartets (the test above), so 64 eps of the other.
  native::dispersive_conserving_n(x2, y2, z2, two);
  native::dispersive_conserving(x[0], x[1], y[0], y[1], z[0], z[1], out0, out1);
  EXPECT_LT(distance(two[0], out0), 64 * kEpsilon);
  EXPECT_LT(distance(two[1], out1), 64 * kEpsilon);
  // The real form, as solprop's real arrays: (x . x) x for one component is x^3.
  // Exact: dyadic operands, every product and sum representable.
  const double r[1] = {1.5};
  double cube[1];
  native::dispersive_isotropic_n(r, r, r, cube);
  EXPECT_EQ(cube[0], ((1.5 * 1.5) * 1.5 + (1.5 * 1.5) * 1.5 + (1.5 * 1.5) * 1.5) / 3.0);
}

TEST(DispersiveMixture, TheRowsAreTheReferenceArithmetic) {
  // solprop `mixture_filter`: the species summed in member order from zero, the
  // quotient by the members' total, the fourth root as two square roots. Held to the
  // rounding of those operations (solprop SEP 25): two products and a sum, a
  // quotient, two square roots (each halving the relative error it is given) and the
  // scale, under 8 eps relative; 16 eps fixed beforehand. A zero scale is a hard mask
  // and gives an exact zero.
  constexpr std::size_t channels = 3;
  constexpr std::size_t bins = 5;
  const std::vector<double> quartic_values = {1.0, 1.25, 1.5, 2.0, 3.25, //
                                              1.0, 1.1,  1.3, 1.7, 2.9,  //
                                              1.0, 1.05, 1.2, 1.6, 2.2};
  const std::vector<double> factor_values = {2.5e-3, 0.0, 7.5e-3, 1.0};
  const std::vector<std::size_t> member_values = {2, 0};
  const std::vector<double> scale_in = {0.0, 1.0, 1.0, 1.0, 0.0};
  const std::vector<double> scale_out = {0.0, 0.125, 0.125, 0.125, 0.0};
  Kokkos::View<double**, native::Layout, native::MemorySpace> quartics("q", channels,
                                                                       bins);
  Kokkos::View<std::size_t*, native::Layout, native::MemorySpace> members("m", 2);
  Kokkos::View<double*, native::Layout, native::MemorySpace> factors("f", 4);
  Kokkos::View<double*, native::Layout, native::MemorySpace> input_scale("i", bins);
  Kokkos::View<double*, native::Layout, native::MemorySpace> output_scale("o", bins);
  Kokkos::View<double*, native::Layout, native::MemorySpace> inputs("in", bins);
  Kokkos::View<double*, native::Layout, native::MemorySpace> outputs("out", bins);
  const auto load = [](const auto& view, const auto& values) {
    auto host = Kokkos::create_mirror_view(view);
    for (std::size_t index = 0; index < values.size(); ++index)
      host.data()[index] = values[index];
    Kokkos::deep_copy(view, host);
  };
  load(quartics, quartic_values);
  load(members, member_values);
  load(factors, factor_values);
  load(input_scale, scale_in);
  load(output_scale, scale_out);
  const native::ExecutionSpace space;
  double total = 0.0;
  for (const auto member : member_values)
    total += factor_values[member];
  native::dispersive_mixture_rows(space, quartics, members, factors, total, input_scale,
                                  output_scale, inputs, outputs);
  const auto in_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, inputs);
  const auto out_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, outputs);
  for (std::size_t k = 0; k < bins; ++k) {
    double quartic = 0.0;
    for (const auto member : member_values)
      quartic += factor_values[member] * quartic_values[member * bins + k];
    const double curve = std::sqrt(std::sqrt(quartic / total));
    const double in_expected = curve * scale_in[k];
    const double out_expected = curve * scale_out[k];
    EXPECT_NEAR(in_host(k), in_expected, 16 * kEpsilon * std::abs(in_expected)) << k;
    EXPECT_NEAR(out_host(k), out_expected, 16 * kEpsilon * std::abs(out_expected)) << k;
  }
  // The static bin gives one, as `g_mix(0) = 1` for any composition. Exact: the
  // quotient of a sum by the same sum is one, and so are its roots.
  EXPECT_EQ(std::sqrt(std::sqrt((2.5e-3 + 7.5e-3) / total)), 1.0);
  // A vacuum: the filter is one, the rows the scales. Exact: an explicit no-op.
  native::dispersive_mixture_rows(space, quartics, members, factors, 0.0, input_scale,
                                  output_scale, inputs, outputs);
  Kokkos::deep_copy(in_host, inputs);
  Kokkos::deep_copy(out_host, outputs);
  for (std::size_t k = 0; k < bins; ++k) {
    EXPECT_EQ(in_host(k), scale_in[k]);
    EXPECT_EQ(out_host(k), scale_out[k]);
  }
}

native::Problem dispersive_problem() {
  native::Problem problem;
  problem.n_t = 4;
  problem.n_to = 8;
  problem.n_omega = 3;
  problem.n_stages = 1;
  problem.family = native::FieldFamily::field_real;
  problem.crop_index = {0, 1, 2};
  problem.active_mask = {1, 1, 1};
  problem.norm_mask = {1, 1, 1};
  problem.omega_window = {1.0, 1.0, 1.0};
  problem.time_window.assign(problem.n_to, 1.0);
  problem.carrier.assign(problem.n_to, Complex{1.0, 0.0});
  problem.phi_terms.assign(problem.n_omega, Complex{});
  problem.prefactor_terms.assign(problem.n_omega, Complex{1.0, 0.0});
  problem.n_phi_terms = 1;
  problem.n_prefactor_terms = 1;
  problem.tableau_matrix = {0.0};
  problem.tableau_nodes = {0.0};
  problem.solution_weights = {1.0};
  problem.error_weights = {0.0};
  problem.synthesis_matrix = {1.0};
  problem.projection_matrix = {1.0};
  problem.synthesis = 1.0;
  problem.projection = 1.0;
  problem.phase_b_engine = true;
  const std::size_t bins = problem.n_to / 2 + 1;
  problem.dispersive = true;
  problem.dispersive_symmetric = true;
  problem.dispersive_form = 0;
  problem.dispersive_inputs.assign(bins, 1.0);
  problem.dispersive_term_multipliers.assign(bins, 1.0);
  problem.dispersive_term_rows = {0};
  problem.dispersive_term_channels = {0};
  problem.dispersive_coefficients = {2.0};
  problem.dispersive_pair_constants = {0.0, 0.0};
  problem.dispersive_group_offsets = {0, 1};
  problem.dispersive_group_members = {0};
  problem.dispersive_input_scale.assign(bins, 1.0);
  problem.dispersive_output_scale.assign(bins, 1.0);
  return problem;
}

TEST(DispersiveResponse, AUnitFilterIsTheConstantCubicOfOneNode) {
  // A lone symmetric response with `g = 1` is `c rho w E^3`, the constant cubic of a
  // field grid's full sector, on the stage's own transforms.
  auto problem = dispersive_problem();
  native::GeneralState state(problem);
  const auto path = state.dispersive_path();
  EXPECT_TRUE(path.at("alone"));
  EXPECT_TRUE(path.at("in_place"));
  EXPECT_TRUE(path.at("borrowed"));
  const std::vector<Complex> field = {{0.0, 0.0}, {0.5, -0.25}, {0.125, 0.75}};
  state.set_initial(field, {});
  native::TrialScalars scalars;
  scalars.phi_delta = {0.0, 0.0};
  scalars.phi_at = {0.0};
  scalars.prefactor = {1.0};
  scalars.density = {1.0};
  scalars.taper = {1.0};
  scalars.dispersive_densities = {3.0};
  static_cast<void>(state.trial_step(0.0, scalars));
  const auto projected = state.projected_to_host(false);
  // The field in time, cubed, transformed back.
  const std::size_t n_to = problem.n_to;
  std::vector<double> time(n_to, 0.0);
  for (std::size_t sample = 0; sample < n_to; ++sample) {
    double value = field[0].real() / static_cast<double>(n_to);
    for (std::size_t bin = 1; bin < field.size(); ++bin) {
      const double angle = 2.0 * std::numbers::pi * static_cast<double>(bin * sample) /
                           static_cast<double>(n_to);
      value +=
          2.0 / static_cast<double>(n_to) *
          (field[bin].real() * std::cos(angle) - field[bin].imag() * std::sin(angle));
    }
    time[sample] = value;
  }
  for (std::size_t bin = 0; bin < n_to / 2 + 1; ++bin) {
    Complex expected{};
    for (std::size_t sample = 0; sample < n_to; ++sample) {
      const double angle = -2.0 * std::numbers::pi * static_cast<double>(bin * sample) /
                           static_cast<double>(n_to);
      expected += (6.0 * time[sample] * time[sample] * time[sample]) *
                  Complex(std::cos(angle), std::sin(angle));
    }
    EXPECT_LT(distance(projected[bin], expected), 64 * kEpsilon) << bin;
  }
  const auto ledger = state.memory_ledger();
  // In place and borrowed: no field-sized buffer of its own.
  EXPECT_EQ(ledger.at("dispersive_filtered_time"), 0U);
  EXPECT_EQ(ledger.at("dispersive_products"), 0U);
  EXPECT_EQ(ledger.at("dispersive_product_spectra"), 0U);
  EXPECT_EQ(ledger.at("dispersive_node_fields"), 0U);
  EXPECT_GT(ledger.at("dispersive_multipliers"), 0U);
  state.close();
  EXPECT_EQ(state.memory_ledger().at("dispersive_multipliers"), 0U);
}

TEST(DispersiveResponse, ATrialNeedsEachStageSDensities) {
  auto problem = dispersive_problem();
  native::GeneralState state(problem);
  state.set_initial(std::vector<Complex>{{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}}, {});
  native::TrialScalars scalars;
  scalars.phi_delta = {0.0, 0.0};
  scalars.phi_at = {0.0};
  scalars.prefactor = {1.0};
  scalars.density = {1.0};
  scalars.taper = {1.0};
  EXPECT_THROW(static_cast<void>(state.trial_step(0.0, scalars)),
               std::invalid_argument);
  scalars.dispersive_densities = {-1.0};
  EXPECT_THROW(static_cast<void>(state.trial_step(0.0, scalars)),
               std::invalid_argument);
}

TEST(DispersiveResponse, MalformedLayoutsAreRefusedBeforeAllocation) {
  {
    auto problem = dispersive_problem();
    problem.dispersive_form = 1; // the conserving sector on the real family
    EXPECT_THROW(native::GeneralState{problem}, std::invalid_argument);
  }
  {
    auto problem = dispersive_problem();
    problem.dispersive_input_scale.pop_back();
    EXPECT_THROW(native::GeneralState{problem}, std::invalid_argument);
  }
  {
    auto problem = dispersive_problem();
    problem.dispersive_term_rows = {1}; // a product the layout does not have
    EXPECT_THROW(native::GeneralState{problem}, std::invalid_argument);
  }
  {
    auto problem = dispersive_problem();
    problem.dispersive_group_members = {1};
    EXPECT_THROW(native::GeneralState{problem}, std::invalid_argument);
  }
  {
    auto problem = dispersive_problem();
    problem.dispersive = false; // data without the response
    EXPECT_THROW(native::GeneralState{problem}, std::invalid_argument);
  }
}

TEST(DispersiveResponse, AProblemWithoutItNamesTheRowsAtZero) {
  auto problem = dispersive_problem();
  problem.dispersive = false;
  problem.dispersive_inputs.clear();
  problem.dispersive_term_multipliers.clear();
  problem.dispersive_term_rows.clear();
  problem.dispersive_term_channels.clear();
  problem.dispersive_coefficients.clear();
  problem.dispersive_pair_constants.clear();
  problem.dispersive_group_offsets.clear();
  problem.dispersive_group_members.clear();
  problem.dispersive_input_scale.clear();
  problem.dispersive_output_scale.clear();
  problem.nonlinear = true;
  native::GeneralState state(problem);
  const auto ledger = state.memory_ledger();
  for (const auto& [name, bytes] : native::DispersiveResponse::empty_ledger())
    EXPECT_EQ(ledger.at(name), bytes) << name;
  EXPECT_TRUE(state.dispersive_path().empty());
}

native::Problem general_problem(const bool symmetric) {
  // A layout of the general modal route: three channels, helium in region 3, argon and
  // krypton in region 5, as solprop's leaf groups them.
  native::Problem problem;
  problem.n_to = 8;
  problem.n_stages = 1;
  problem.family = native::FieldFamily::field_real;
  problem.general_modal = true;
  problem.general_nodes = 5;
  problem.general_node_chunk = 2;
  problem.general_components = 3;
  const std::size_t bins = problem.n_to / 2 + 1;
  problem.dispersive = true;
  problem.dispersive_symmetric = symmetric;
  problem.dispersive_form = 1;
  problem.dispersive_coefficients = {1.0, 2.0, 3.0};
  problem.dispersive_pair_constants.assign(6, 0.5);
  problem.dispersive_input_scale.assign(bins, 1.0);
  problem.dispersive_output_scale.assign(bins, 0.125);
  problem.dispersive_node_region = {5, 5, 3, 0, 3};
  if (symmetric) {
    problem.dispersive_group_offsets = {0, 1, 3};
    problem.dispersive_group_members = {0, 1, 2};
    problem.dispersive_group_keys = {3, 5};
    problem.dispersive_inputs.assign(2 * bins, 1.0);
    problem.dispersive_term_rows = {0, 1};
    problem.dispersive_term_channels = {0, 4};
    problem.dispersive_term_multipliers.assign(2 * bins, 0.125);
    problem.dispersive_quartics.assign(3 * bins, 1.0);
  } else {
    problem.dispersive_group_offsets = {0, 1, 2, 3};
    problem.dispersive_group_members = {0, 1, 2};
    problem.dispersive_group_keys = {3, 5, 5};
    problem.dispersive_inputs.assign(6 * bins, 1.0);
    problem.dispersive_term_rows = {0, 2, 3, 1, 4, 5, 1, 6, 7};
    problem.dispersive_term_channels = {0, 0, 3, 1, 1, 3, 2, 2, 3};
    problem.dispersive_term_multipliers.assign(9 * bins, 0.125);
  }
  return problem;
}

TEST(DispersiveWeights, ThePolynomialFormsOneCubePerRegion) {
  const auto problem = general_problem(false);
  const native::DispersiveWeights weights(problem, native::ExecutionSpace());
  EXPECT_EQ(weights.groups(), 3U);
  EXPECT_EQ(weights.cubes(), 2U);
  EXPECT_EQ(weights.fields(), 6U);
  EXPECT_EQ(weights.products(), 8U);
  EXPECT_EQ(weights.cube(0), 0U);
  EXPECT_EQ(weights.cube(1), 1U);
  EXPECT_EQ(weights.cube(2), 1U);
  EXPECT_EQ(weights.cube_key(0), 3U);
  EXPECT_EQ(weights.cube_key(1), 5U);
  const auto symmetric = general_problem(true);
  const native::DispersiveWeights mixture(symmetric, native::ExecutionSpace());
  EXPECT_EQ(mixture.products(), 2U);
  EXPECT_EQ(mixture.cubes(), 0U);
  EXPECT_EQ(mixture.key(1), 5U);
}

TEST(DispersiveWeights, RegionsBelongToTheGeneralRouteAlone) {
  {
    auto problem = general_problem(false);
    problem.dispersive_node_region.pop_back();
    EXPECT_THROW(native::require_dispersive(problem), std::invalid_argument);
  }
  {
    auto problem = general_problem(false);
    problem.dispersive_group_keys.pop_back();
    EXPECT_THROW(native::require_dispersive(problem), std::invalid_argument);
  }
  {
    auto problem = general_problem(false);
    problem.dispersive_form = 2; // an envelope's harmonic groups
    EXPECT_THROW(native::require_dispersive(problem), std::invalid_argument);
  }
  {
    auto problem = general_problem(false);
    problem.dispersive_term_rows.back() = 8; // one product past the layout
    EXPECT_THROW(native::require_dispersive(problem), std::invalid_argument);
  }
  {
    // Off the general route every channel acts at every node.
    auto problem = dispersive_problem();
    problem.dispersive_group_keys = {0};
    EXPECT_THROW(native::GeneralState{problem}, std::invalid_argument);
  }
  EXPECT_NO_THROW(native::require_dispersive(general_problem(true)));
}

} // namespace
