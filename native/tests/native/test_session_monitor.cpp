// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The native session's own boundary monitor (solprop SEP 47 sections 5.3 and 6.4).
// solprop's tests hold the monitor's decisions and record to its Python monitor on
// recorded readings, and a monitored session to the driver's end to end; these check,
// on a toy field whose readings the test scripts, the order the session gives the
// monitor's actions: the launch judged after the launch plane, a warning a barrier at
// its committed endpoint before that step's statistics row, the rest of the point's
// work done once after it, a stop a terminal failure there, a stale step checked at
// its plane, and the final check before completion.

#include "solprop_native/session.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {
namespace session = solprop_native::session;
namespace rk = solprop_native::rk;
namespace watch = solprop_native::watch;

// A toy field whose boundary readings are scripted by the accepted step: the guard
// share of step `k` is `guard[k]` (zero past the script), and the collar's removal
// rate is `rate` throughout.
class Scripted final : public session::Resident {
public:
  [[nodiscard]] std::size_t elements() const override { return 2; }
  [[nodiscard]] std::size_t n_omega() const override { return 2; }
  const double* host_field() override { return field_.data(); }
  void field_to_host(double* out) override {
    std::memcpy(out, field_.data(), field_.size() * sizeof(double));
  }
  void count_to_host(std::size_t /*bytes*/) override {}
  std::vector<double> statistics(const session::StatisticsRecipe& /*recipe*/,
                                 const double z) override {
    return {z, static_cast<double>(accepted)};
  }
  std::vector<double> boundary_reading() override {
    ++readings;
    std::vector<double> reading(watch::reading_head, 0.0);
    reading[0] = 1.0;
    reading[1] = accepted < guard.size() ? guard[accepted] : 0.0;
    reading[4] = rate;
    return reading;
  }
  std::vector<double> field_{1.0, 0.0, 0.5, 0.0};
  std::vector<double> guard;
  double rate = 0.0;
  std::size_t accepted = 0;
  int readings = 0;
};

class Engine final : public rk::Engine {
public:
  explicit Engine(Scripted& field) : field_(field) {}
  double trial(double /*z*/, double /*dz*/) override { return 0.25; }
  void accept(double /*dz*/) override { ++field_.accepted; }
  std::vector<double> reading() override { return field_.boundary_reading(); }

private:
  Scripted& field_;
};

session::MonitorPlan monitor(const std::uint64_t cadence = 1) {
  session::MonitorPlan plan;
  auto& policy = plan.config.policy;
  policy.activity = 1e-8;
  policy.warn = 1e-3;
  policy.stop = 1e-2;
  policy.quiet_share = 1e-12;
  policy.quiet_checks = 2;
  policy.cadence = cadence;
  policy.interval = 1.0;
  plan.config.floor_share = 1e-12;
  plan.config.floor_margin = 3.0;
  plan.config.budget = 1e-3;
  plan.config.rate = {0.0, 1.0};
  return plan;
}

struct Built {
  std::unique_ptr<session::Session> session;
  Scripted* field = nullptr;
};

Built build(std::vector<double> guard, const double rate = 0.0,
            const std::uint64_t cadence = 1) {
  auto field = std::make_unique<Scripted>();
  field->guard = std::move(guard);
  field->rate = rate;
  Built built;
  built.field = field.get();
  session::RungeKuttaMethod method;
  method.policy.safety = 0.9;
  method.policy.beta1 = 0.0;
  method.policy.beta2 = 0.0;
  method.policy.min_factor = 0.2;
  method.policy.max_factor = 1.0;
  method.policy.lag = 1;
  method.policy.auto_init = false;
  method.policy.dz_min = 1e-9;
  method.policy.dz_max = 1e-3;
  method.policy.landings.dz_min = 1e-9;
  method.policy.landings.dz_max = 1e-3;
  method.policy.landings.stretch = 1.0;
  method.policy.landings.z_end = 0.01;
  method.policy.landings.save_planes = {0.005, 0.01};
  method.policy.landings.breakpoints = {0.0, 0.01};
  method.policy.landings.landing_ulps = 8;
  method.state.proposal = 1e-3;
  method.engine = std::make_unique<Engine>(*field);
  session::Plan plan;
  plan.planes = {0.005, 0.01};
  plan.row_statistics.kind = session::StatisticsRecipe::Kind::temporal;
  plan.row_statistics.flags = 1;
  plan.period = 1;
  plan.monitor = monitor(cadence);
  built.session = std::make_unique<session::Session>(
      std::move(field), std::move(method), std::move(plan), 0.0);
  return built;
}

// Advance once, requiring a batch of `kind`; acknowledge it unless told not to.
session::Batch next(session::Session& run, const session::EventKind kind,
                    const bool acknowledge = true) {
  auto batch = run.advance(4096, 10.0);
  EXPECT_EQ(batch.event.kind, kind);
  if (acknowledge)
    run.acknowledge(batch.sequence);
  return batch;
}

// The rows' accepted-step values a batch carries.
std::vector<double> rows_of(const session::Batch& batch) {
  std::vector<double> steps;
  for (std::size_t row = 0; row < batch.row_z.size(); ++row)
    steps.push_back(batch.row_values[2 * row + 1]);
  return steps;
}
} // namespace

TEST(SessionMonitor, TheLaunchIsJudgedAfterTheLaunchPlane) {
  auto built = build({4e-6}); // the launch's guard amplitude is 2e-3: a warning
  auto& run = *built.session;
  next(run, session::EventKind::observation);
  EXPECT_EQ(built.field->readings, 0) << "the monitor read the launch before its plane";
  const auto warned = next(run, session::EventKind::warning);
  EXPECT_EQ(warned.event.notice.kind, session::Notice::Kind::guard);
  EXPECT_TRUE(warned.event.notice.launch);
  EXPECT_DOUBLE_EQ(warned.event.notice.value, 2e-3);
  EXPECT_TRUE(warned.event.ticket);
  EXPECT_TRUE(warned.attempts.empty());
  const auto record = run.monitor_record();
  ASSERT_TRUE(record.has_value());
  EXPECT_EQ(record->planes[0], std::vector<double>{0.0});
}

TEST(SessionMonitor, AWarningIsABarrierBeforeItsStepsRow) {
  // The third accepted step's check warns: the barrier carries the rows of the first
  // two, and the third's row follows it, once, after the acknowledgment.
  auto built = build({0.0, 0.0, 0.0, 4e-6});
  auto& run = *built.session;
  next(run, session::EventKind::observation);
  const auto warned = next(run, session::EventKind::warning);
  EXPECT_FALSE(warned.event.notice.launch);
  EXPECT_EQ(rows_of(warned), (std::vector<double>{1.0, 2.0}));
  EXPECT_EQ(warned.attempts.size() / session::attempt_columns, 3U);
  EXPECT_DOUBLE_EQ(warned.event.notice.z, warned.event.z);
  const auto plane = next(run, session::EventKind::observation);
  EXPECT_EQ(rows_of(plane), (std::vector<double>{3.0, 4.0, 5.0}));
  next(run, session::EventKind::observation);
  next(run, session::EventKind::terminal);
  const auto record = run.monitor_record();
  ASSERT_TRUE(record.has_value());
  EXPECT_EQ(record->steps, 10U);
  EXPECT_DOUBLE_EQ(record->guard_max[0], 4e-6);
}

TEST(SessionMonitor, AStopFailsAtItsEndpointWithoutItsRow) {
  auto built = build({0.0, 0.0, 4e-4}); // an amplitude of 2e-2: past the stop
  auto& run = *built.session;
  next(run, session::EventKind::observation);
  const auto stopped = next(run, session::EventKind::terminal, false);
  EXPECT_EQ(stopped.event.outcome, session::Outcome::failed);
  EXPECT_EQ(stopped.event.failure.notice.kind, session::Notice::Kind::guard);
  EXPECT_DOUBLE_EQ(stopped.event.failure.notice.value, 2e-2);
  EXPECT_TRUE(stopped.event.ticket);
  EXPECT_EQ(rows_of(stopped), std::vector<double>{1.0});
  EXPECT_EQ(stopped.attempts.size() / session::attempt_columns, 2U);
}

TEST(SessionMonitor, APlaneChecksTheStepItsCadenceSkipped) {
  // Checked every eighth step only: the plane after five steps checks the fifth, and
  // its warning is a barrier before the plane is observed.
  auto built = build({0.0, 0.0, 0.0, 0.0, 0.0, 4e-6}, 0.0, 8);
  auto& run = *built.session;
  next(run, session::EventKind::observation);
  const auto warned = next(run, session::EventKind::warning);
  EXPECT_DOUBLE_EQ(warned.event.z, 0.005);
  EXPECT_EQ(rows_of(warned), (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}));
  const auto plane = next(run, session::EventKind::observation);
  EXPECT_EQ(plane.event.observation, session::Observation::plane);
  EXPECT_TRUE(plane.attempts.empty());
  const auto record = run.monitor_record();
  ASSERT_TRUE(record.has_value());
  EXPECT_EQ(record->steps + record->skipped, 5U);
  EXPECT_EQ(record->planes[0], (std::vector<double>{0.0, 0.005}));
  EXPECT_DOUBLE_EQ(record->planes[2][1], 4e-6);
}

TEST(SessionMonitor, TheFinalCheckWarnsBeforeCompletion) {
  // The collar removes a tenth of the launch energy per metre: 1e-3 over the run,
  // whose square root is above the budget.
  auto built = build({}, 0.1);
  auto& run = *built.session;
  next(run, session::EventKind::observation);
  next(run, session::EventKind::observation);
  next(run, session::EventKind::observation);
  const auto warned = next(run, session::EventKind::warning);
  EXPECT_EQ(warned.event.notice.kind, session::Notice::Kind::removal);
  EXPECT_NEAR(warned.event.notice.value, 1e-3, 1e-15);
  const auto done = next(run, session::EventKind::terminal);
  EXPECT_EQ(done.event.outcome, session::Outcome::completed);
}
