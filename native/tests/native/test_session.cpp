// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The native execution session's unit (solprop SEP 47). solprop's lifecycle
// conformance suite holds the session to the protocol through its Python adapter, and
// its control traces to the driver's over the same marches; these check, on a toy field
// whose trials double it, what those cannot isolate: the event order and its ranges,
// the cadence of the resident rows, yields that change nothing, the stop and snapshot
// controls, the export's recipe to the bit, and the strict unit's own rounding.

#include "solprop_native/session.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {
namespace session = solprop_native::session;
namespace march = solprop_native::march;
namespace rk = solprop_native::rk;

volatile double factor_a = 1.0 + 0x1p-30;
volatile double factor_b = 1.0 - 0x1p-30;
volatile double addend = -1.0;

[[gnu::noinline]] double contracting(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}

// A field of four bins in one column; its statistics are the position and the energy.
class Toy final : public session::Resident {
public:
  Toy() : field_{1.0, 0.5, -0.0, -2.0, 3.0, 0.0, -1.0, 0.25} {}
  [[nodiscard]] std::size_t elements() const override { return field_.size() / 2; }
  [[nodiscard]] std::size_t n_omega() const override { return 4; }
  const double* host_field() override { return field_.data(); }
  void field_to_host(double* out) override {
    std::memcpy(out, field_.data(), field_.size() * sizeof(double));
  }
  void count_to_host(const std::size_t bytes) override { crossed += bytes; }
  std::vector<double> statistics(const session::StatisticsRecipe& /*recipe*/,
                                 const double z) override {
    ++evaluations;
    double energy = 0.0;
    for (const double value : field_)
      energy += value * value;
    return {z, energy};
  }
  void double_it() {
    for (double& value : field_)
      value *= 2.0;
  }
  std::vector<double> field_;
  std::size_t crossed = 0;
  int evaluations = 0;
};

// Trials that score 0.25, or 5.0 at the trials `rejects` names; an accept doubles the
// field, so after `k` accepted steps it is the launch times `2^k` exactly.
class Doubling final : public rk::Engine {
public:
  Doubling(Toy& toy, std::vector<int> rejects, const int fail_at = 0)
      : toy_(toy), rejects_(std::move(rejects)), fail_at_(fail_at) {}
  double trial(double /*z*/, double /*dz*/) override {
    ++trials;
    if (trials == fail_at_)
      throw std::invalid_argument("a scripted trial failure");
    for (const int reject : rejects_)
      if (reject == trials)
        return 5.0;
    return 0.25;
  }
  void accept(double /*dz*/) override { toy_.double_it(); }
  std::vector<double> reading() override { return {}; }
  int trials = 0;

private:
  Toy& toy_;
  std::vector<int> rejects_;
  int fail_at_;
};

rk::Policy controller() {
  rk::Policy policy;
  policy.safety = 0.9;
  policy.beta1 = 0.7 / 5.0;
  policy.beta2 = 0.4 / 5.0;
  policy.min_factor = 0.2;
  policy.max_factor = 1.5;
  policy.lag = 1;
  policy.power = 5.0;
  policy.exponent = 1.0 / 5.0;
  policy.start_growth = 100.0;
  policy.auto_init = false;
  policy.dz_min = 1e-9;
  policy.dz_max = 4e-3;
  policy.landings.dz_min = 1e-9;
  policy.landings.dz_max = 4e-3;
  policy.landings.stretch = 1.0;
  policy.landings.z_end = 0.03;
  policy.landings.save_planes = {0.01, 0.02, 0.03};
  policy.landings.breakpoints = {0.0, 0.03};
  policy.landings.landing_ulps = 8;
  return policy;
}

struct Built {
  std::unique_ptr<session::Session> session;
  Toy* toy = nullptr;
  Doubling* engine = nullptr;
};

Built build(std::vector<int> rejects = {}, const int fail_at = 0,
            const std::uint64_t step_planes = 3, const std::uint64_t period = 2) {
  auto toy = std::make_unique<Toy>();
  Built built;
  built.toy = toy.get();
  session::RungeKuttaMethod method;
  method.policy = controller();
  method.state.proposal = 1e-3;
  auto engine = std::make_unique<Doubling>(*toy, std::move(rejects), fail_at);
  built.engine = engine.get();
  method.engine = std::move(engine);
  session::Plan plan;
  plan.planes = {0.01, 0.02, 0.03};
  plan.step_planes = step_planes;
  plan.plane_statistics.kind = session::StatisticsRecipe::Kind::temporal;
  plan.plane_statistics.flags = 2;
  plan.row_statistics = plan.plane_statistics;
  plan.period = period;
  built.session = std::make_unique<session::Session>(std::move(toy), std::move(method),
                                                     std::move(plan), 0.0);
  return built;
}

// What one observation reported, without wall times, and its field.
struct Seen {
  session::EventKind kind;
  session::Observation observation;
  double z;
  std::int64_t plane;
  std::vector<double> attempts;
  std::vector<double> row_z;
  std::vector<std::uint64_t> row_step;
  std::vector<double> rows;
  std::vector<double> stats;
  std::vector<double> field;
  session::Outcome outcome;
  bool operator==(const Seen&) const = default;
};

std::vector<double> controls(const session::Batch& batch, const session::Event& event) {
  std::vector<double> out;
  for (std::size_t row = event.attempts_begin; row < event.attempts_end; ++row)
    for (std::size_t column = 0; column + 1 < session::attempt_columns; ++column)
      out.push_back(batch.attempts[row * session::attempt_columns + column]);
  return out;
}

// Drive a session to its end, exporting every field-bearing event's identity field.
std::vector<Seen> drive(session::Session& run, const std::size_t work_limit,
                        const double deadline, std::size_t* batches = nullptr) {
  std::vector<Seen> seen;
  while (run.status().state != session::State::finished) {
    const auto batch = run.advance(work_limit, deadline);
    if (batches != nullptr)
      ++*batches;
    for (const auto& event : batch.events) {
      if (event.kind == session::EventKind::progress ||
          event.kind == session::EventKind::snapshot)
        continue;
      Seen entry{event.kind,
                 event.observation,
                 event.z,
                 event.plane_index,
                 controls(batch, event),
                 {},
                 {},
                 {},
                 event.has_stats ? batch.plane_stats : std::vector<double>{},
                 {},
                 event.outcome};
      if (event.has_rows) {
        entry.row_z = batch.row_z;
        entry.row_step = batch.row_step;
        entry.rows = batch.row_values;
      }
      if (event.ticket) {
        entry.field.resize(2 * run.elements());
        run.export_field(batch.sequence, event.generation, false, false, nullptr,
                         entry.field.data());
      }
      seen.push_back(std::move(entry));
    }
    run.acknowledge(batch.sequence);
  }
  return seen;
}
} // namespace

TEST(Session, TheSchedulerRoundsEveryOperation) {
  EXPECT_EQ(session::strict_probe(factor_a, factor_b, addend), 0.0);
#if defined(__OPTIMIZE__) && defined(__FP_FAST_FMA) && !defined(__CUDACC__) &&         \
    !defined(__HIPCC__)
  ASSERT_EQ(contracting(factor_a, factor_b, addend), -0x1p-60)
      << "the control did not fuse, so the witness proves nothing";
#endif
}

TEST(Session, TheToleranceComparisonsAreTheDriversOwn) {
  // `numpy.isclose(z, target, rtol=1e-12, atol=0)`.
  EXPECT_TRUE(session::plane_reached(1.0, 1.0));
  EXPECT_TRUE(session::plane_reached(1.0 + 5e-13, 1.0));
  EXPECT_FALSE(session::plane_reached(1.0 + 2e-12, 1.0));
  EXPECT_TRUE(session::plane_reached(0.0, 0.0));
  // `max(LANDING_ULPS, n_steps) * spacing(max(length, |position|))`.
  EXPECT_EQ(session::replay_tolerance(3, 1.0, 0.5, 8), 8 * 0x1p-52);
  EXPECT_EQ(session::replay_tolerance(160, 1.0, 2.0, 8), 160 * 0x1p-51);
  EXPECT_EQ(session::steps_to_next(3, 0), 3U);
  EXPECT_EQ(session::steps_to_next(3, 4), 2U);
}

TEST(Session, TheRecipeIsApplyRecipesToTheBit) {
  // NumPy's `state * sqrt(w)` then `(-1)^k conj(.)`, each a complex product with its
  // zero terms: signed zeros and infinities come out as NumPy makes them.
  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<double> state = {-0.0, -1.0, 2.0, -0.0, -0.0, 0.0, inf, 1.0};
  const std::vector<double> weight = {4.0, 1.0, 0.25, 9.0};
  std::vector<double> out(state.size());
  session::apply_recipe(state.data(), 4, 4, weight.data(), true, true, out.data());
  // Element 0: (-0 * 2 - (-1) * 0, -0 * 0 + (-1) * 2) = (+0, -2); conj (+0, 2); k = 0:
  // (0 - 0 * 2, 2 + 0 * 0) = (+0, +2).
  EXPECT_EQ(out[0], 0.0);
  EXPECT_FALSE(std::signbit(out[0]));
  EXPECT_EQ(out[1], 2.0);
  // Element 1, k = 1: (2, -0) times 1 is (2, +0 ... ) then conj and the sign.
  EXPECT_EQ(out[2], -2.0);
  // Element 3: an infinite real part times zero is NaN in the imaginary part.
  EXPECT_TRUE(std::isnan(out[7]) || std::isnan(out[6]));
  // Without scaling the state is copied; without optics no sign or conjugation.
  session::apply_recipe(state.data(), 4, 4, nullptr, false, false, out.data());
  EXPECT_EQ(std::memcmp(out.data(), state.data(), state.size() * sizeof(double)), 0);
}

TEST(Session, EventsFollowTheCommittedOrder) {
  auto built = build({2, 7});
  auto& run = *built.session;
  // The launch plane comes first, with its statistics and no attempts.
  const auto launch = run.advance(256, 10.0);
  ASSERT_EQ(launch.events.size(), 1U);
  EXPECT_EQ(launch.events[0].kind, session::EventKind::observation);
  EXPECT_EQ(launch.events[0].observation, session::Observation::launch);
  EXPECT_EQ(launch.events[0].plane_index, -1);
  EXPECT_EQ(launch.events[0].attempts_end, 0U);
  EXPECT_TRUE(launch.events[0].ticket);
  EXPECT_EQ(launch.plane_stats.at(0), 0.0);
  // No barrier is crossed before its batch is acknowledged; a stale receipt is refused.
  EXPECT_THROW((void)run.advance(256, 10.0), session::LifecycleError);
  EXPECT_THROW(run.acknowledge(launch.sequence + 1), session::LifecycleError);
  run.acknowledge(launch.sequence);
  EXPECT_THROW(run.acknowledge(launch.sequence), session::LifecycleError);
  const auto seen = drive(run, 256, 10.0);
  ASSERT_FALSE(seen.empty());
  EXPECT_EQ(seen.back().kind, session::EventKind::terminal);
  EXPECT_EQ(seen.back().outcome, session::Outcome::completed);
  EXPECT_TRUE(seen.back().field.empty());
  std::vector<std::int64_t> planes;
  std::uint64_t accepted = 0;
  std::uint64_t attempts = 0;
  for (const auto& entry : seen) {
    if (entry.kind != session::EventKind::observation)
      continue;
    const std::size_t n = entry.attempts.size() / 4;
    attempts += n;
    std::uint64_t before = accepted;
    for (std::size_t row = 0; row < n; ++row)
      accepted += entry.attempts[row * 4 + 3] != 0.0 ? 1U : 0U;
    // The field is the launch times 2^accepted, exactly.
    EXPECT_EQ(entry.field.at(0), std::ldexp(1.0, static_cast<int>(accepted)));
    if (entry.observation == session::Observation::plane)
      planes.push_back(entry.plane);
    if (entry.observation == session::Observation::step) {
      EXPECT_EQ(accepted % 3, 0U);
    }
    // A row every second accepted step, at an accepted attempt, its position that
    // attempt's end.
    for (std::size_t row = 0; row < entry.row_z.size(); ++row) {
      const auto step = entry.row_step[row];
      ASSERT_LT(step, n);
      EXPECT_EQ(entry.attempts[step * 4 + 3], 1.0);
      EXPECT_EQ(entry.row_z[row], entry.attempts[step * 4]);
      std::uint64_t counted = before;
      for (std::size_t index = 0; index <= step; ++index)
        counted += entry.attempts[index * 4 + 3] != 0.0 ? 1U : 0U;
      EXPECT_EQ(counted % 2, 0U);
      EXPECT_EQ(entry.rows[2 * row], entry.row_z[row]);
    }
  }
  EXPECT_EQ(planes, (std::vector<std::int64_t>{0, 1, 2}));
  const auto status = run.status();
  EXPECT_EQ(status.attempts, attempts);
  EXPECT_EQ(status.rejected, 2U);
  EXPECT_EQ(status.z, 0.03);
  EXPECT_THROW((void)run.advance(256, 10.0), session::LifecycleError);
}

TEST(Session, YieldsChangeNothing) {
  auto baseline = build({2, 7, 12});
  const auto expected = drive(*baseline.session, 1000000, 1e6);
  for (const std::size_t limit : {std::size_t{1}, std::size_t{7}}) {
    auto built = build({2, 7, 12});
    std::size_t batches = 0;
    EXPECT_EQ(drive(*built.session, limit, 1e6, &batches), expected) << limit;
    if (limit == 1) {
      EXPECT_EQ(batches, built.session->status().attempts + 2);
    }
  }
  auto deadline = build({2, 7, 12});
  EXPECT_EQ(drive(*deadline.session, 256, 1e-9), expected);
  // Rows were taken natively: one evaluation a row and one an observation, no more.
  std::uint64_t observations = 0;
  for (const auto& entry : expected)
    observations += entry.kind == session::EventKind::observation ? 1U : 0U;
  EXPECT_GT(baseline.session->status().rows_taken, 0U);
  EXPECT_EQ(baseline.session->status().rows_taken + observations,
            static_cast<std::uint64_t>(baseline.toy->evaluations));
}

TEST(Session, AStopBeforeTheFirstStepEndsAtTheEntrance) {
  auto built = build();
  auto& run = *built.session;
  run.acknowledge(run.advance(256, 10.0).sequence);
  run.request_stop();
  const auto batch = run.advance(256, 10.0);
  ASSERT_EQ(batch.events.size(), 1U);
  const auto& terminal = batch.events[0];
  EXPECT_EQ(terminal.kind, session::EventKind::terminal);
  EXPECT_EQ(terminal.outcome, session::Outcome::stopped);
  EXPECT_EQ(terminal.z, 0.0);
  EXPECT_TRUE(terminal.ticket);
  EXPECT_EQ(terminal.attempts_end, 0U);
  EXPECT_EQ(built.engine->trials, 0);
  run.acknowledge(batch.sequence);
  EXPECT_EQ(run.status().state, session::State::finished);
  EXPECT_EQ(run.status().outcome, session::Outcome::stopped);
}

TEST(Session, ASnapshotWaitsForAnAcceptedEndpointAndConsumesNothing) {
  auto built = build({1}, 0, 0, 1);
  auto& run = *built.session;
  run.acknowledge(run.advance(256, 10.0).sequence);
  EXPECT_FALSE(run.request_snapshot().coalesced);
  EXPECT_TRUE(run.request_snapshot().coalesced);
  const auto batch = run.advance(256, 10.0);
  const auto& snapshot = batch.events.back();
  ASSERT_EQ(snapshot.kind, session::EventKind::snapshot);
  // The first trial was rejected; the snapshot is the first accepted endpoint's.
  ASSERT_EQ(snapshot.attempts_end, 2U);
  EXPECT_EQ(batch.attempts[session::attempt_columns + 3], 1.0);
  EXPECT_EQ(snapshot.z, batch.attempts[session::attempt_columns]);
  EXPECT_EQ(batch.row_z.size(), 1U);
  std::vector<double> field(2 * run.elements());
  run.export_field(batch.sequence, snapshot.generation, false, false, nullptr,
                   field.data());
  EXPECT_EQ(field[0], 2.0);
  const std::vector<double> held(batch.attempts.begin(),
                                 batch.attempts.begin() + 2 * session::attempt_columns);
  run.acknowledge(batch.sequence);
  // The ticket expired with its batch.
  EXPECT_THROW(run.export_field(batch.sequence, snapshot.generation, false, false,
                                nullptr, field.data()),
               session::LifecycleError);
  auto plane = run.advance(256, 10.0);
  while (plane.events.back().kind != session::EventKind::observation) {
    run.acknowledge(plane.sequence);
    plane = run.advance(256, 10.0);
  }
  EXPECT_EQ(plane.events.back().plane_index, 0);
  for (std::size_t index = 0; index < held.size(); ++index) {
    if (index % session::attempt_columns != 4) {
      EXPECT_EQ(plane.attempts[index], held[index]);
    }
  }
  EXPECT_EQ(plane.row_z.front(), batch.row_z.front());
}

TEST(Session, AFailedTrialEndsFailedAtTheCommittedState) {
  auto built = build({}, 5);
  auto& run = *built.session;
  const auto seen = drive(run, 256, 10.0);
  const auto& terminal = seen.back();
  EXPECT_EQ(terminal.kind, session::EventKind::terminal);
  EXPECT_EQ(terminal.outcome, session::Outcome::failed);
  // Four accepted steps stand; the field is the launch times 2^4.
  ASSERT_FALSE(terminal.field.empty());
  EXPECT_EQ(terminal.field[0], 16.0);
  EXPECT_EQ(run.status().attempts, 4U);
}

TEST(Session, AnExportIsValidatedAndCounted) {
  auto built = build();
  auto& run = *built.session;
  const auto launch = run.advance(256, 10.0);
  std::vector<double> out(2 * run.elements());
  const std::vector<double> bad = {1.0, 0.0, 1.0, 1.0};
  EXPECT_THROW(run.export_field(launch.sequence, launch.events[0].generation, true,
                                false, bad.data(), out.data()),
               std::domain_error);
  const std::vector<double> ones = {1.0, 1.0, 1.0, 1.0};
  run.export_field(launch.sequence, launch.events[0].generation, true, true,
                   ones.data(), out.data());
  std::vector<double> expected(out.size());
  session::apply_recipe(built.toy->field_.data(), 4, 4, nullptr, false, true,
                        expected.data());
  EXPECT_EQ(std::memcmp(out.data(), expected.data(), out.size() * sizeof(double)), 0);
  EXPECT_EQ(run.status().exports, 1U);
  EXPECT_EQ(built.toy->crossed, out.size() * sizeof(double));
  run.close();
  run.close();
  EXPECT_THROW((void)run.advance(256, 10.0), session::LifecycleError);
  EXPECT_THROW(run.acknowledge(launch.sequence), session::LifecycleError);
  EXPECT_THROW((void)run.request_snapshot(), session::LifecycleError);
}

namespace {
// An Adams engine whose every trial is accepted, doubling the toy field.
class Steady final : public march::Engine {
public:
  explicit Steady(Toy& toy) : toy_(toy) {}
  void start(double /*z*/) override {}
  void trial(double /*z*/, double /*dz*/, const march::Stencil& stencil,
             double* norms) override {
    for (std::size_t index = 0; index < stencil.n_orders; ++index)
      norms[index] = 0.25;
  }
  void accept(double /*dz*/, std::size_t /*free*/, const march::Stencil& /*stencil*/,
              const std::vector<march::Entry>& /*kept*/) override {
    toy_.double_it();
  }
  double chord_phase(double /*z1*/, double /*z2*/) override { return 0.0; }

private:
  Toy& toy_;
};
} // namespace

TEST(Session, AnAdamsSessionLandsOnEveryPlane) {
  auto toy = std::make_unique<Toy>();
  session::AdamsMethod method;
  method.policy.order = 5;
  method.policy.variable_order = true;
  method.policy.safety = 0.9;
  method.policy.max_growth = 2.0;
  method.policy.min_factor = 0.2;
  method.policy.rejection_growth = 0.9;
  method.policy.hysteresis = 1.2;
  method.policy.restart_after = 3;
  method.policy.collar_stability = 1.0;
  method.policy.real_intervals = {2.0, 1.0, 0.5, 0.3, 0.2, 0.1};
  method.policy.carrier_cap = 1.0;
  method.policy.equal_approach = true;
  method.policy.dz_min = 1e-9;
  method.policy.dz_max = 4e-3;
  method.policy.stretch = 1.0;
  method.policy.z_end = 0.03;
  method.policy.save_planes = {0.01, 0.02, 0.03};
  method.policy.breakpoints = {0.0, 0.03};
  method.policy.landing_ulps = 8;
  method.state.proposal = 1e-3;
  method.engine = std::make_unique<Steady>(*toy);
  session::Plan plan;
  plan.planes = {0.01, 0.02, 0.03};
  session::Session run(std::move(toy), std::move(method), std::move(plan), 0.0);
  const auto seen = drive(run, 3, 10.0);
  std::vector<double> planes;
  for (const auto& entry : seen)
    if (entry.observation == session::Observation::plane &&
        entry.kind == session::EventKind::observation)
      planes.push_back(entry.z);
  EXPECT_EQ(planes, (std::vector<double>{0.01, 0.02, 0.03}));
  EXPECT_EQ(seen.back().outcome, session::Outcome::completed);
}
