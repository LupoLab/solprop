// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The low-storage replay in the native session (solprop SEP 24, SEP 47 section 6.3).
// solprop's session tests hold the whole replay to the driver's over the same native
// passes, bitwise; these check what they cannot isolate: the table keys of solprop's
// `table_steps` and `schedule_key`, the stage scalars a step reads, and, on a toy field
// stepped in place, the replay's records and the poisoned terminal of a step left in
// flight, by a failure or a divergence.

#include "solprop_native/low_storage.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "solprop_native/session.hpp"

namespace {
namespace low_storage = solprop_native::rk::low_storage;
namespace march = solprop_native::march;
namespace rk = solprop_native::rk;
namespace session = solprop_native::session;
namespace stage = solprop_native::stage;
namespace statistics = solprop_native::statistics;

volatile double factor_a = 1.0 + 0x1p-30;
volatile double factor_b = 1.0 - 0x1p-30;
volatile double addend = -1.0;

constexpr int landing_ulps = 8;

// A uniform cut of [start, stop] into `count` steps, its steps the edges' differences.
void cut(std::vector<double>& steps, std::vector<double>& edges, const double start,
         const double stop, const std::size_t count) {
  if (edges.empty())
    edges.push_back(start);
  for (std::size_t index = 1; index <= count; ++index) {
    const double fraction = static_cast<double>(index) / static_cast<double>(count);
    const double edge = index == count ? stop : start + (stop - start) * fraction;
    steps.push_back(edge - edges.back());
    edges.push_back(edge);
  }
}

// A one-interval table of constant values over [0, 1].
stage::Table constant_table(const std::vector<double>& values) {
  stage::Table table;
  table.edges = {0.0, 1.0};
  table.lower = 0.0;
  table.upper = 1.0;
  table.coefficients = values;
  table.n_terms = 1;
  table.n_values = values.size();
  return table;
}

// The tables of a constant operator and prefactor with a constant density, four stages.
stage::Tables constant_tables() {
  stage::Tables tables;
  tables.n_stages = 4;
  tables.phi = stage::Terms::constant;
  tables.prefactor = stage::Terms::constant;
  tables.prefactor_low = 0.0;
  tables.prefactor_high = 1.0;
  tables.density = constant_table({2.5});
  return tables;
}

} // namespace

TEST(LowStorage, TheKeysAndPositionsRoundEveryOperation) {
  // The exact product is 1 - 2^-60, which rounds to 1: separately rounded the sum is 0,
  // fused -2^-60.
  EXPECT_EQ(low_storage::strict_probe(factor_a, factor_b, addend), 0.0);
}

TEST(LowStorage, ARunIsKeyedOnItsMeanStep) {
  // `table_steps`'s doctest: a uniform schedule cut at a plane keys each segment on its
  // own mean, one key over each run.
  std::vector<double> steps;
  std::vector<double> edges;
  cut(steps, edges, 0.0, 0.25, 3);
  cut(steps, edges, 0.25, 1.0, 8);
  const auto keys = low_storage::table_steps(steps, edges, landing_ulps);
  ASSERT_EQ(keys.size(), steps.size());
  for (std::size_t index = 0; index < 3; ++index)
    EXPECT_EQ(keys[index], (edges[3] - edges[0]) / 3.0);
  for (std::size_t index = 3; index < keys.size(); ++index)
    EXPECT_EQ(keys[index], (edges[11] - edges[3]) / 8.0);
  EXPECT_NEAR(keys.front(), 0.25 / 3.0, 1e-15);
  EXPECT_EQ(keys.back(), 0.09375);
}

TEST(LowStorage, StepsThatDifferInTheirLastBitsFarFromTheOriginKeepOneKey) {
  // Steps of a uniform cut far from the origin differ by ulps of the position, many
  // ulps of the step: one run, one key.
  std::vector<double> steps;
  std::vector<double> edges;
  cut(steps, edges, 1000.0, 1000.0 + 1e-3, 7);
  bool differ = false;
  for (const double step : steps)
    differ = differ || step != steps.front();
  ASSERT_TRUE(differ);
  const auto keys = low_storage::table_steps(steps, edges, landing_ulps);
  for (const double key : keys)
    EXPECT_EQ(key, (edges.back() - edges.front()) / 7.0);
}

TEST(LowStorage, ADiscoveredScheduleKeysEachStepOnItself) {
  const std::vector<double> steps = {0.1, 0.2, 0.15, 0.05};
  const std::vector<double> edges = {0.0, 0.1, 0.30000000000000004, 0.45, 0.5};
  const auto keys = low_storage::table_steps(steps, edges, landing_ulps);
  EXPECT_EQ(keys, steps);
  // Found by the step's midpoint, clamped to the steps.
  EXPECT_EQ(low_storage::schedule_key(keys, edges, 0.1, 0.2), 0.2);
  EXPECT_EQ(low_storage::schedule_key(keys, edges, -1.0, 0.1), 0.1);
  EXPECT_EQ(low_storage::schedule_key(keys, edges, 0.5, 0.1), 0.05);
  EXPECT_THROW((void)low_storage::table_steps(steps, {0.0, 1.0}, landing_ulps),
               std::invalid_argument);
}

TEST(LowStorage, EachPositionIsItsOwnAnchorAndTheEndPadsTheStages) {
  const auto tables = constant_tables();
  stage::Anchor anchor;
  const double z = 0.125;
  const double dz = 0.0625;
  const double midpoint = z + 0.5 * dz;
  const double end = z + dz;
  const low_storage::Config recompute{0, false, 0.0};
  const auto scalars =
      low_storage::scalars(tables, anchor, recompute, z, dz, false, std::nan(""));
  EXPECT_EQ(scalars.phi_at, (std::vector<double>{z, midpoint, end, end}));
  EXPECT_EQ(scalars.prefactor, (std::vector<double>{1.0, 1.0, 1.0, 1.0}));
  EXPECT_EQ(scalars.density, (std::vector<double>{2.5, 2.5, 2.5, 2.5}));
  EXPECT_EQ(scalars.taper, (std::vector<double>{1.0, 1.0, 1.0, 1.0}));
  EXPECT_EQ(scalars.kerr_density, (std::vector<double>{0.0, 0.0, 0.0, 0.0}));
  // A flow formed for the step reads its interval: the two half steps' increments,
  // each the difference of the positions, the other rows zero.
  EXPECT_EQ(scalars.phi_delta,
            (std::vector<double>{midpoint - z, end - midpoint, 0.0, 0.0, 0.0}));
}

TEST(LowStorage, AConstantOperatorsTableIsBuiltFromItsOrigin) {
  const auto tables = constant_tables();
  stage::Anchor anchor;
  const double z = 0.375;
  const double dz = 0.1;
  const double key = 0.09999999999999998;
  const low_storage::Config table{1, false, 0.25};
  const auto built = low_storage::scalars(tables, anchor, table, z, dz, true, key);
  // fl(origin + key / 2) - origin, not key / 2: solprop's engines build it so.
  const double half = 0.25 + 0.5 * key;
  EXPECT_EQ(built.phi_delta, (std::vector<double>{0.0, 0.0, half - 0.25, 0.0, 0.0}));
  const auto reused = low_storage::scalars(tables, anchor, table, z, dz, false, key);
  EXPECT_EQ(reused.phi_delta, (std::vector<double>(5, 0.0)));
  // A separable operator's increments are the distances themselves.
  const low_storage::Config separable{1, true, 0.25};
  const auto spatial =
      low_storage::scalars(tables, anchor, separable, z, dz, true, key);
  EXPECT_EQ(spatial.phi_delta,
            (std::vector<double>{0.5 * dz, 0.5 * dz, 0.5 * key, 0.0, 0.0}));
  EXPECT_THROW((void)low_storage::scalars(tables, anchor,
                                          low_storage::Config{0, false, 0.0}, z, dz,
                                          true, key),
               std::logic_error);
}

TEST(LowStorage, AnExpansionsIncrementsAreItsWeightsDifferences) {
  auto tables = constant_tables();
  tables.phi = stage::Terms::expansion;
  // Two values, each linear in z over [0, 1]: c0 T0 + c1 T1 with x = 2 z - 1.
  tables.phi_weights.edges = {0.0, 1.0};
  tables.phi_weights.lower = 0.0;
  tables.phi_weights.upper = 1.0;
  tables.phi_weights.coefficients = {0.5, -1.0, 0.25, 3.0};
  tables.phi_weights.n_terms = 2;
  tables.phi_weights.n_values = 2;
  stage::Anchor anchor;
  const double z = 0.3;
  const double dz = 0.2;
  const double midpoint = z + 0.5 * dz;
  const auto scalars = low_storage::scalars(
      tables, anchor, low_storage::Config{2, false, 0.0}, z, dz, false, std::nan(""));
  stage::Anchor fresh;
  const auto first = stage::row(tables, fresh, z, midpoint).phi_delta;
  const auto second = stage::row(tables, fresh, midpoint, z + dz).phi_delta;
  ASSERT_EQ(scalars.phi_delta.size(), 5U * 2U);
  EXPECT_EQ(scalars.phi_delta[0], first[0]);
  EXPECT_EQ(scalars.phi_delta[1], first[1]);
  EXPECT_EQ(scalars.phi_delta[2], second[0]);
  EXPECT_EQ(scalars.phi_delta[3], second[1]);
  for (std::size_t index = 4; index < scalars.phi_delta.size(); ++index)
    EXPECT_EQ(scalars.phi_delta[index], 0.0);
  // Each row's value of `Phi` is the expansion's weights at its own position.
  std::vector<double> at_end(2);
  tables.phi_weights.at(z + dz, at_end.data());
  EXPECT_EQ(scalars.phi_at[6], at_end[0]);
  EXPECT_EQ(scalars.phi_at[7], at_end[1]);
}

namespace {

// A field of four bins in one column; its statistics are the position and the energy.
class Toy final : public session::Resident {
public:
  [[nodiscard]] std::size_t elements() const override { return field_.size() / 2; }
  [[nodiscard]] std::size_t n_omega() const override { return 4; }
  const double* host_field() override { return field_.data(); }
  void field_to_host(double* out) override {
    std::memcpy(out, field_.data(), field_.size() * sizeof(double));
  }
  void count_to_host(std::size_t /*bytes*/) override {}
  std::vector<double> statistics(const session::StatisticsRecipe& /*recipe*/,
                                 const double z) override {
    double energy = 0.0;
    for (const double value : field_)
      energy += value * value;
    return {z, energy};
  }
  std::vector<double> field_{1.0, 0.5, -0.25, 2.0, 3.0, 0.0, -1.0, 0.25};
};

// A step accepted in place, as the low-storage engine takes one: the trial doubles the
// field itself and is in flight until its accept. The trial `fail_at` throws after
// doubling; the trial `diverge_at` scores a non-finite norm.
class InPlace final : public rk::Engine {
public:
  InPlace(Toy& toy, const int fail_at, const int diverge_at)
      : toy_(toy), fail_at_(fail_at), diverge_at_(diverge_at) {}
  double trial(double /*z*/, double /*dz*/) override {
    in_flight_ = true;
    ++trials;
    for (double& value : toy_.field_)
      value *= 2.0;
    if (trials == fail_at_)
      throw std::runtime_error("a scripted failure partway through a step");
    return trials == diverge_at_ ? std::numeric_limits<double>::infinity() : 0.0;
  }
  void accept(double /*dz*/) override { in_flight_ = false; }
  std::vector<double> reading() override { return {}; }
  [[nodiscard]] bool in_flight() const noexcept override { return in_flight_; }
  int trials = 0;

private:
  Toy& toy_;
  int fail_at_;
  int diverge_at_;
  bool in_flight_ = false;
};

struct Replay {
  std::unique_ptr<session::Session> session;
  Toy* toy = nullptr;
};

// Twelve steps of 2.5 mm over three planes, a row every second step.
Replay replay(const int fail_at = 0, const int diverge_at = 0) {
  auto toy = std::make_unique<Toy>();
  Replay built;
  built.toy = toy.get();
  session::RungeKuttaMethod method;
  method.policy.landings.landing_ulps = landing_ulps;
  cut(method.schedule.steps, method.schedule.edges, 0.0, 0.03, 12);
  method.engine = std::make_unique<InPlace>(*toy, fail_at, diverge_at);
  session::Plan plan;
  plan.planes = {0.01, 0.02, 0.03};
  auto statistics = std::make_shared<statistics::Plan>();
  statistics->outputs = {statistics::Stat::energy};
  plan.plane_statistics.plan = std::move(statistics);
  plan.row_statistics = plan.plane_statistics;
  plan.period = 2;
  plan.replay = true;
  plan.replay_steps = 12;
  plan.replay_length = 0.03;
  built.session = std::make_unique<session::Session>(std::move(toy), std::move(method),
                                                     std::move(plan), 0.0);
  return built;
}

// Advance and acknowledge to the terminal batch, returning it unacknowledged.
session::Batch to_terminal(session::Session& run, std::vector<double>& attempts) {
  while (true) {
    auto batch = run.advance(256, 10.0);
    attempts.insert(attempts.end(), batch.attempts.begin(), batch.attempts.end());
    if (batch.event.kind == session::EventKind::terminal)
      return batch;
    run.acknowledge(batch.sequence);
  }
}

} // namespace

TEST(LowStorage, TheReplayRecordsEveryStepAcceptedWithNoEstimate) {
  auto built = replay();
  auto& run = *built.session;
  std::vector<double> attempts;
  const auto terminal = to_terminal(run, attempts);
  EXPECT_EQ(terminal.event.outcome, session::Outcome::completed);
  ASSERT_EQ(attempts.size(), 12U * session::attempt_columns);
  for (std::size_t row = 0; row < 12; ++row) {
    const double* values = attempts.data() + row * session::attempt_columns;
    EXPECT_EQ(values[2], 0.0) << row;
    EXPECT_EQ(values[3], 1.0) << row;
  }
  EXPECT_EQ(attempts[11 * session::attempt_columns], 0.03);
  EXPECT_EQ(run.status().rows_taken, 6U);
  EXPECT_EQ(built.toy->field_[0], 4096.0);
}

TEST(LowStorage, AFailurePartwayThroughAStepPoisonsTheSession) {
  auto built = replay(6);
  auto& run = *built.session;
  std::vector<double> attempts;
  const auto terminal = to_terminal(run, attempts);
  EXPECT_EQ(terminal.event.outcome, session::Outcome::failed);
  EXPECT_EQ(terminal.event.failure.reason, march::Reason::error);
  EXPECT_EQ(terminal.event.failure.stage, march::Stage::trial);
  // No checkpoint: the state is neither end of the step, and is not exported.
  EXPECT_FALSE(terminal.event.ticket);
  std::vector<double> out(2 * run.elements());
  EXPECT_THROW(run.export_field(terminal.sequence, terminal.event.generation, false,
                                false, nullptr, out.data()),
               session::LifecycleError);
  // The five committed steps stand in the records, and the terminal is where they end.
  EXPECT_EQ(attempts.size(), 5U * session::attempt_columns);
  EXPECT_EQ(run.status().attempts, 5U);
  EXPECT_EQ(terminal.event.z, attempts[4 * session::attempt_columns]);
  EXPECT_EQ(terminal.event.failure.cursor, 5U);
  run.acknowledge(terminal.sequence);
  EXPECT_THROW((void)run.advance(256, 10.0), session::LifecycleError);
}

TEST(LowStorage, ADivergedStepPoisonsTheSession) {
  auto built = replay(0, 4);
  auto& run = *built.session;
  std::vector<double> attempts;
  const auto terminal = to_terminal(run, attempts);
  EXPECT_EQ(terminal.event.outcome, session::Outcome::failed);
  EXPECT_EQ(terminal.event.failure.reason, march::Reason::diverged);
  EXPECT_FALSE(terminal.event.ticket);
  // The fourth step's, uncommitted: the terminal is where the third ended.
  std::vector<double> steps;
  std::vector<double> edges;
  cut(steps, edges, 0.0, 0.03, 12);
  EXPECT_EQ(terminal.event.failure.replay_dz, steps[3]);
  EXPECT_EQ(terminal.event.failure.replay_end, edges[4]);
  EXPECT_EQ(terminal.event.z, edges[3]);
  EXPECT_EQ(run.status().attempts, 3U);
}
