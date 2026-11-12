// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The native session's transverse edge (solprop SEP 40 section 5.3, SEP 47 sections
// 5.3 and 6.4). solprop's tests hold the edge's decisions and record to its Python edge
// on recorded readings, and an edged session to the driver's end to end; these check,
// on a toy field whose edge readings the test scripts, the order the session gives the
// edge's actions: launched after the launch plane, read where its spacing falls due
// with every step held to that spacing, each of a launch's warnings its own barrier, a
// read's warning a barrier before its step's statistics row, a stop a terminal failure,
// a plane read before its observation, and a collar's current taken at every step
// with no more batches than the planes'.

#include "solprop_native/session.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace {
namespace session = solprop_native::session;
namespace rk = solprop_native::rk;
namespace transverse = solprop_native::transverse;
namespace watch = solprop_native::watch;
using Kind = transverse::Notice::Kind;

// A toy field: no light at the time window's guards, and at the transverse edge's
// first face `from_share` of the launch from the position `from_z` on; the launch
// clips `clip`, and a collar's current is `current` per metre.
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
    std::vector<double> reading(watch::reading_head, 0.0);
    reading[0] = 1.0;
    return reading;
  }
  transverse::Reading edge_reading(const bool faces, const bool whole,
                                   const bool totals, const bool strip,
                                   const bool want_current) override {
    transverse::Reading reading;
    if (faces) {
      ++reads;
      read_at.push_back(position);
      const double share = position >= from_z ? from_share : 0.0;
      reading.faces = std::array<double, 4>{share, 0.0, 0.0, 0.0};
      reading.columns = {1.0, 0.0};
      const std::array<double, 4> sums{1.0, 0.0, 0.0, 0.0};
      if (whole) {
        reading.clip = clip;
        reading.totals = sums;
      } else if (totals) {
        reading.totals = sums;
      }
      if (strip)
        reading.strip = std::vector<double>{0.0, 0.0};
    }
    if (want_current) {
      ++currents;
      reading.current = current;
    }
    return reading;
  }
  std::vector<double> field_{1.0, 0.0, 0.5, 0.0};
  std::array<double, 4> clip{};
  double from_z = 1.0;
  double from_share = 0.0;
  double current = 0.0;
  double position = 0.0;
  std::size_t accepted = 0;
  int reads = 0;
  int currents = 0;
  std::vector<double> read_at;
};

class Engine final : public rk::Engine {
public:
  explicit Engine(Scripted& field) : field_(field) {}
  double trial(double /*z*/, double /*dz*/) override { return 0.25; }
  void accept(const double dz) override {
    ++field_.accepted;
    field_.position += dz;
    steps.push_back(dz);
  }
  std::vector<double> reading() override { return field_.boundary_reading(); }
  std::vector<double> steps;

private:
  Scripted& field_;
};

constexpr double length = 0.01;
constexpr double spacing = length / 16.0;

session::MonitorPlan monitor(const bool collar) {
  session::MonitorPlan plan;
  auto& policy = plan.config.policy;
  policy.activity = 1e-8;
  policy.warn = 1e-3;
  policy.stop = 1e-2;
  policy.quiet_share = 1e-12;
  policy.quiet_checks = 2;
  policy.cadence = 1;
  policy.interval = 1.0;
  plan.config.floor_share = 1e-12;
  plan.config.floor_margin = 3.0;
  plan.config.budget = 1e-3;
  plan.config.rate = {0.0, 1.0};
  // Two faces, the spectrum at no transverse wavenumber: the spacing is the
  // backstop, L / 16.
  transverse::Config edge;
  edge.faces = {"x-", "x+"};
  edge.warn = 1e-6;
  edge.stop = 1e-4;
  edge.length = length;
  edge.guard = 1e-4;
  edge.k_min = 1e7;
  edge.k_floor = 1e-3;
  edge.columns = {1.0, 1.0};
  edge.kappa = {0.0, 1e5};
  edge.kappa_order = {0, 1};
  if (collar) {
    edge.design = true;
    edge.ramp_rate = {0.0, 0.0};
    edge.collar = true;
    edge.collar_rate = true;
  }
  plan.edge = edge;
  return plan;
}

struct Built {
  std::unique_ptr<session::Session> session;
  Scripted* field = nullptr;
  Engine* engine = nullptr;
};

Built build(const bool collar = false) {
  auto field = std::make_unique<Scripted>();
  Built built;
  built.field = field.get();
  session::RungeKuttaMethod method;
  // A trial norm of a quarter: the proposal holds at its largest, 1e-3.
  method.policy.safety = 0.9;
  method.policy.beta1 = 0.2;
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
  method.policy.landings.z_end = length;
  method.policy.landings.save_planes = {0.005, length};
  method.policy.landings.breakpoints = {0.0, length};
  method.policy.landings.landing_ulps = 8;
  method.state.proposal = 1e-3;
  auto engine = std::make_unique<Engine>(*field);
  built.engine = engine.get();
  method.engine = std::move(engine);
  session::Plan plan;
  plan.planes = {0.005, length};
  auto statistics = std::make_shared<solprop_native::statistics::Plan>();
  statistics->outputs = {solprop_native::statistics::Stat::energy};
  plan.row_statistics.plan = std::move(statistics);
  plan.period = 1;
  plan.monitor = monitor(collar);
  built.session = std::make_unique<session::Session>(
      std::move(field), std::move(method), std::move(plan), 0.0);
  return built;
}

session::Batch next(session::Session& run, const session::EventKind kind,
                    const bool acknowledge = true) {
  auto batch = run.advance(4096, 10.0);
  EXPECT_EQ(batch.event.kind, kind) << batch.event.failure.message;
  if (acknowledge)
    run.acknowledge(batch.sequence);
  return batch;
}

transverse::Record record_of(const session::Session& run) {
  const auto record = run.edge_record();
  EXPECT_TRUE(record.has_value());
  return record ? record->first : transverse::Record{};
}
} // namespace

TEST(SessionTransverse, TheEdgeIsReadOnItsSpacingWithoutAnyMoreBatches) {
  auto built = build();
  auto& run = *built.session;
  next(run, session::EventKind::observation);
  EXPECT_EQ(built.field->reads, 0) << "the edge read the launch before its plane";
  next(run, session::EventKind::observation);
  next(run, session::EventKind::observation);
  next(run, session::EventKind::terminal);
  EXPECT_EQ(run.status().batches, 4U);
  // Every step is held to the read spacing, and the reads are at most a spacing
  // apart, at the launch and before a step would carry past the next read.
  for (const double step : built.engine->steps)
    EXPECT_LE(step, spacing);
  const auto& at = built.field->read_at;
  ASSERT_GE(at.size(), 16U);
  EXPECT_EQ(at.front(), 0.0);
  for (std::size_t index = 1; index < at.size(); ++index)
    EXPECT_LE(at[index] - at[index - 1], spacing * (1.0 + 1e-12));
  const auto record = record_of(run);
  EXPECT_EQ(record.reads, static_cast<std::int64_t>(at.size()) - 1);
  EXPECT_EQ(record.status, transverse::Status::clear);
  EXPECT_LE(record.largest_gap, spacing * (1.0 + 1e-12));
}

TEST(SessionTransverse, ALaunchsWarningsAreConsecutiveBarriers) {
  auto built = build();
  built.field->clip = {2e-6, 3e-6, 0.0, 0.0};
  auto& run = *built.session;
  next(run, session::EventKind::observation);
  const auto first = next(run, session::EventKind::warning);
  EXPECT_EQ(first.event.notice.kind, session::Notice::Kind::transverse);
  EXPECT_EQ(first.event.notice.edge.kind, Kind::clip);
  EXPECT_EQ(first.event.notice.edge.face, 0U);
  EXPECT_TRUE(first.event.ticket);
  const auto second = next(run, session::EventKind::warning);
  EXPECT_EQ(second.event.notice.edge.kind, Kind::clip);
  EXPECT_EQ(second.event.notice.edge.face, 1U);
  EXPECT_TRUE(second.attempts.empty());
  // The launch then finishes, once: its baselines and its first read.
  const auto plane = next(run, session::EventKind::observation);
  EXPECT_EQ(plane.event.observation, session::Observation::plane);
  EXPECT_EQ(built.field->read_at.front(), 0.0);
  EXPECT_EQ(built.field->read_at[1] > 0.0, true);
  EXPECT_LE(built.field->read_at[1], spacing);
}

TEST(SessionTransverse, AReadsWarningIsABarrierBeforeItsStepsRow) {
  auto built = build();
  built.field->from_z = 0.002;
  built.field->from_share = 3e-5;
  auto& run = *built.session;
  next(run, session::EventKind::observation);
  const auto warned = next(run, session::EventKind::warning);
  EXPECT_EQ(warned.event.notice.kind, session::Notice::Kind::transverse);
  EXPECT_EQ(warned.event.notice.edge.kind, Kind::face);
  EXPECT_FALSE(warned.event.notice.edge.stop);
  EXPECT_DOUBLE_EQ(warned.event.notice.edge.z, warned.event.z);
  EXPECT_GE(warned.event.z, 0.002);
  const std::size_t attempts = warned.attempts.size() / session::attempt_columns;
  // The barrier carries the rows of the steps before the warned one.
  EXPECT_EQ(warned.row_z.size(), attempts - 1);
  const auto plane = next(run, session::EventKind::observation);
  ASSERT_FALSE(plane.row_step.empty());
  EXPECT_EQ(plane.row_step[0], -1);
  EXPECT_DOUBLE_EQ(plane.row_z[0], warned.event.z);
  next(run, session::EventKind::observation);
  next(run, session::EventKind::terminal);
  const auto record = record_of(run);
  EXPECT_EQ(record.status, transverse::Status::warned);
  EXPECT_DOUBLE_EQ(record.warned_at, warned.event.z);
}

TEST(SessionTransverse, AStopIsATerminalFailureAtItsRead) {
  auto built = build();
  built.field->from_z = 0.002;
  built.field->from_share = 2e-4;
  auto& run = *built.session;
  next(run, session::EventKind::observation);
  const auto stopped = next(run, session::EventKind::terminal, false);
  EXPECT_EQ(stopped.event.outcome, session::Outcome::failed);
  const auto& notice = stopped.event.failure.notice;
  EXPECT_EQ(notice.kind, session::Notice::Kind::transverse);
  EXPECT_EQ(notice.edge.kind, Kind::face);
  EXPECT_TRUE(notice.edge.stop);
  EXPECT_TRUE(stopped.event.ticket);
  EXPECT_GE(stopped.event.z, 0.002);
  EXPECT_LT(stopped.event.z, 0.002 + spacing + 1e-3);
  EXPECT_EQ(record_of(run).status, transverse::Status::stopped);
}

TEST(SessionTransverse, APlaneIsReadBeforeItsObservation) {
  // Light at the edge from just short of the first plane: the plane's read warns, a
  // barrier at the plane, before the plane is observed.
  auto built = build();
  built.field->from_z = 0.005 * (1.0 - 1e-12);
  built.field->from_share = 3e-5;
  auto& run = *built.session;
  next(run, session::EventKind::observation);
  const auto warned = next(run, session::EventKind::warning);
  EXPECT_DOUBLE_EQ(warned.event.z, 0.005);
  const auto plane = next(run, session::EventKind::observation);
  EXPECT_EQ(plane.event.observation, session::Observation::plane);
  EXPECT_DOUBLE_EQ(plane.event.z, 0.005);
}

TEST(SessionTransverse, ACollarsCurrentIsTakenAtEveryStepWithoutAReturn) {
  auto built = build(true);
  built.field->current = 1e-5;
  auto& run = *built.session;
  next(run, session::EventKind::observation);
  next(run, session::EventKind::observation);
  next(run, session::EventKind::observation);
  next(run, session::EventKind::terminal);
  EXPECT_EQ(run.status().batches, 4U);
  EXPECT_EQ(static_cast<std::size_t>(built.field->currents), built.field->accepted);
  const auto record = record_of(run);
  // A constant current of 1e-5 of the launch per metre from the first step on.
  const double first = built.engine->steps.front();
  EXPECT_NEAR(record.current, 1e-5 * (length - first), 1e-17);
  EXPECT_EQ(record.status, transverse::Status::clear);
}
