// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The workspace (solprop SEP 52 section 5.3): the lease between it and the session or
// probe that uses it, the coupled flow's guard, the run records' native forms, and the
// timing probe, each moved out of the binding unit and tested here.

#include "execution.hpp"
#include "solprop_native/workspace.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <csignal>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

namespace native = solprop_native;
using native::Complex;

// A one-stage envelope of four bins with a constant operator and Kerr response.
native::Problem envelope_problem() {
  native::Problem problem;
  problem.cpu_execution = native::test::table();
  problem.dims.n_t = 4;
  problem.dims.n_to = 4;
  problem.dims.n_omega = 4;
  problem.dims.family = native::FieldFamily::envelope;
  problem.stepper.n_stages = 1;
  problem.linear.n_phi_terms = 1;
  problem.linear.n_prefactor_terms = 1;
  problem.grid.crop_index = {0, 1, 2, 3};
  problem.grid.active_mask.assign(4, 1U);
  problem.grid.norm_mask.assign(4, 1U);
  problem.grid.omega_window.assign(4, 1.0);
  problem.grid.omega = {1.0, 2.0, 3.0, 4.0};
  problem.grid.dt = 0.25;
  problem.grid.time_window.assign(4, 1.0);
  problem.grid.time_window_derivative.assign(4, 0.0);
  problem.grid.carrier.assign(4, Complex(1.0, 0.0));
  problem.linear.phi_terms.assign(4, Complex(0.0, -1.0));
  problem.linear.prefactor_terms.assign(4, Complex(1.0, 0.0));
  problem.stepper.tableau_matrix = {0.0};
  problem.stepper.tableau_nodes = {0.0};
  problem.stepper.solution_weights = {1.0};
  problem.stepper.error_weights = {0.25};
  problem.transform.synthesis = 2.0;
  problem.transform.projection = 0.5;
  problem.transform.synthesis_matrix = {2.0};
  problem.transform.projection_matrix = {0.5};
  problem.kerr.nonlinear = true;
  problem.kerr.coefficient = 0.125;
  problem.kerr.self_factor = 0.5;
  problem.kerr.conjugate_factor = 0.25;
  return problem;
}

native::stage::Table constant_table(const std::vector<double>& values) {
  native::stage::Table table;
  table.edges = {-1.0, 2.0};
  table.lower = -1.0;
  table.upper = 2.0;
  table.coefficients = values;
  table.n_terms = 1;
  table.n_values = values.size();
  return table;
}

native::Setup runge_kutta_setup() {
  native::Setup setup;
  native::stage::Tables tables;
  tables.n_stages = 1;
  tables.phi = native::stage::Terms::constant;
  tables.prefactor = native::stage::Terms::constant;
  tables.prefactor_low = -1.0;
  tables.prefactor_high = 2.0;
  tables.density = constant_table({1.0});
  setup.stage_tables = tables;
  setup.runge_kutta = native::RungeKuttaConfig{
      .nodes = {0.0}, .rtol = 1e-6, .atol = 1e-12, .weights = {}, .dense_weights = {}};
  return setup;
}

native::Problem with_setup(native::Problem problem, native::Setup setup) {
  problem.setup = std::make_shared<native::Setup>(std::move(setup));
  return problem;
}

std::vector<Complex> launch() {
  return {Complex(1.0, 0.5), Complex(0.25, -0.5), Complex(-0.75, 0.0),
          Complex(0.0, 0.125)};
}

} // namespace

TEST(Workspace, TheLeaseIsExclusiveAndReleasedOnlyByItsHolder) {
  native::Workspace workspace(envelope_problem());
  int first = 0;
  int second = 0;
  workspace.lease(&first);
  EXPECT_TRUE(workspace.leased());
  EXPECT_THROW(workspace.lease(&second), std::logic_error);
  EXPECT_THROW(static_cast<void>(workspace.state()), std::logic_error);
  // The fixed facts stay readable under a lease.
  EXPECT_NO_THROW(static_cast<void>(workspace.built()));
  EXPECT_NO_THROW(static_cast<void>(workspace.leased_state(&first)));
  EXPECT_THROW(static_cast<void>(workspace.leased_state(&second)), std::logic_error);
  workspace.release(&second);
  EXPECT_TRUE(workspace.leased());
  workspace.release(&first);
  EXPECT_FALSE(workspace.leased());
  EXPECT_NO_THROW(static_cast<void>(workspace.state()));
}

TEST(Workspace, AClosedWorkspaceRefusesEverything) {
  native::Workspace workspace(envelope_problem());
  workspace.close();
  EXPECT_TRUE(workspace.closed());
  EXPECT_THROW(static_cast<void>(workspace.built()), std::logic_error);
  EXPECT_THROW(workspace.load(launch(), {}), std::logic_error);
  workspace.close(); // idempotent
}

TEST(Workspace, TheSetupIsAppliedWhenTheWorkspaceIsBuilt) {
  native::Workspace workspace(with_setup(envelope_problem(), runge_kutta_setup()));
  EXPECT_TRUE(workspace.stage_tables().has_value());
  EXPECT_EQ(workspace.kind(), native::MethodKind::runge_kutta);
  auto setup = runge_kutta_setup();
  setup.low_storage = native::rk::low_storage::Config{};
  native::Workspace replay(with_setup(envelope_problem(), setup));
  EXPECT_EQ(replay.kind(), native::MethodKind::low_storage);
}

TEST(Workspace, StageTablesOfAnotherTableauAreRefused) {
  auto setup = runge_kutta_setup();
  ASSERT_TRUE(setup.stage_tables.has_value());
  setup.stage_tables.value().n_stages = 2;
  EXPECT_THROW(native::Workspace(with_setup(envelope_problem(), setup)),
               std::invalid_argument);
}

TEST(Workspace, TheFlowGuardRefusesTooManySquaringsBeforeForming) {
  native::Workspace workspace(envelope_problem());
  workspace.configure_flow_guard({.row_sum = {1e6},
                                  .condition_bound = {1.0},
                                  .taylor_scale = 0.5,
                                  .max_squarings = 4,
                                  .allowance = 2.0});
  // A node of zero forms nothing; the step's end needs ceil(log2(1e6 / 0.5)) = 21.
  try {
    workspace.guard_flow(0, 1.0, {0.0});
    FAIL() << "the guard admitted too many squarings";
  } catch (const native::BackendRefusal& error) {
    EXPECT_NE(std::string(error.what()).find("21 squarings, above the 4"),
              std::string::npos);
  }
  EXPECT_THROW(workspace.guard_flow(1, 1.0, {0.0}), std::out_of_range);
}

TEST(Workspace, AFlowGuardsCertificateIsOneRowASection) {
  native::Workspace workspace(envelope_problem());
  EXPECT_THROW(workspace.configure_flow_guard({.row_sum = {1.0, 2.0},
                                               .condition_bound = {1.0, 2.0, 3.0},
                                               .taylor_scale = 0.5,
                                               .max_squarings = 4,
                                               .allowance = 2.0}),
               std::invalid_argument);
  EXPECT_THROW(workspace.guard_flow(0, 1.0, {0.0}), std::logic_error);
}

TEST(Workspace, AScheduleRecordsEdgesAccumulateInStepOrder) {
  // 0.1 + 0.2 rounds above 0.3: the edges are the running sum, as NumPy's cumsum.
  const auto schedule = native::to_native(
      native::ScheduleRecord{.steps = {0.1, 0.2, 0.3}, .z_start = 1.0});
  ASSERT_EQ(schedule.edges.size(), 4U);
  double sum = 0.0;
  EXPECT_EQ(schedule.edges[0], 1.0);
  for (std::size_t index = 0; index < 3; ++index) {
    sum += schedule.steps[index];
    EXPECT_EQ(schedule.edges[index + 1], 1.0 + sum);
  }
}

TEST(Workspace, ARungeKuttaPolicyFlattensItsController) {
  native::RungeKuttaPolicyRecord record;
  record.controller.safety = 0.8;
  record.controller.beta1 = 0.12;
  record.controller.beta2 = -0.04;
  record.controller.lag = 1;
  record.controller.estimator_order = 4;
  record.dz_min = 1e-6;
  record.dz_max = 1e-2;
  record.breakpoints = {0.0, 1.0};
  record.quiet_interval = 0.5;
  const auto policy = native::to_native(record);
  EXPECT_EQ(policy.power, 5.0);
  EXPECT_EQ(policy.exponent, 1.0 / 5.0);
  EXPECT_EQ(policy.safety, 0.8);
  EXPECT_EQ(policy.lag, 1U);
  EXPECT_TRUE(policy.landings.has_quiet_interval);
  EXPECT_EQ(policy.landings.quiet_interval, 0.5);
  record.breakpoints = {0.0};
  EXPECT_THROW(static_cast<void>(native::to_native(record)), std::invalid_argument);
}

TEST(Workspace, AResumptionReadsItsNumbersByName) {
  native::ResumptionRecord record;
  record.record = true;
  record.numbers = {{"launch_energy", 2.0}, {"steps", 7.0}};
  record.planes = std::vector<std::vector<double>>(5, std::vector<double>{0.5});
  const auto resumption = native::to_native(record);
  EXPECT_EQ(resumption.launch_energy, 2.0);
  EXPECT_EQ(resumption.steps, 7.0);
  EXPECT_FALSE(resumption.collar_max.has_value());
  EXPECT_TRUE(resumption.has_planes);
  record.numbers["nonsense"] = 1.0;
  EXPECT_THROW(static_cast<void>(native::to_native(record)), std::invalid_argument);
  native::EdgeResumptionRecord edge;
  edge.numbers = {{"nonsense", 1.0}};
  // A record that is none carries nothing else, as solprop's resumption writes it.
  EXPECT_FALSE(native::to_native(edge).record);
}

TEST(Workspace, AFaultIsNamedByItsStageAndCategory) {
  const auto source = native::fault_from("trial_source", 2, "value");
  EXPECT_TRUE(source.in_source);
  EXPECT_EQ(source.stage, native::march::Stage::trial);
  EXPECT_EQ(source.category, native::march::Category::value);
  EXPECT_EQ(source.after, 2U);
  EXPECT_EQ(native::fault_from("accept", 0, "sigterm").signal, SIGTERM);
  EXPECT_THROW(static_cast<void>(native::fault_from("finish", 0, "value")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(native::fault_from("trial", 0, "other")),
               std::invalid_argument);
}

TEST(Workspace, AProbesTrialsRepeatAndItsFieldOwnsAKeptStep) {
  native::Workspace workspace(with_setup(envelope_problem(), runge_kutta_setup()));
  auto probe = workspace.probe(launch(), {}, 0.0);
  // The probe holds the workspace: no load, session or other probe meanwhile.
  EXPECT_TRUE(workspace.leased());
  EXPECT_THROW(workspace.load(launch(), {}), std::logic_error);
  EXPECT_THROW(static_cast<void>(workspace.probe(launch(), {}, 0.0)), std::logic_error);
  EXPECT_THROW(static_cast<void>(probe->field()), native::BackendRefusal);
  const double first = probe->trial(0.0, 0.1);
  ASSERT_TRUE(std::isfinite(first));
  // The same trial from the same loaded state: the same operations on the same build,
  // held to a few roundings of a four-bin norm rather than to its bits (SEP 25).
  const double bound = 16.0 * std::numeric_limits<double>::epsilon() * std::abs(first);
  EXPECT_NEAR(probe->trial(0.0, 0.1), first, bound);
  const auto field = probe->field();
  EXPECT_EQ(field.size(), 4U);
  // The loaded state was loaded again after the read-back.
  EXPECT_NEAR(probe->trial(0.0, 0.1), first, bound);
  EXPECT_EQ(probe->trials(), 3U);
  probe->close();
  EXPECT_FALSE(workspace.leased());
  EXPECT_FALSE(workspace.closed());
}

TEST(Workspace, ALowStorageProbeNeedsItsReplaysSchedule) {
  auto setup = runge_kutta_setup();
  setup.low_storage = native::rk::low_storage::Config{};
  native::Workspace workspace(with_setup(envelope_problem(), setup));
  EXPECT_THROW(static_cast<void>(workspace.probe(launch(), {}, 0.0)), std::logic_error);
  EXPECT_FALSE(workspace.leased());
}
