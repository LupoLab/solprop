// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The transverse edge's unit (solprop SEP 40 section 5.3, SEP 47 section 6.4). Its
// decisions and record are held to solprop's `TransverseEdge` on drawn readings by
// solprop's tests (tests/backends/test_transverse_owner.py); these check what those
// cannot see: that the unit rounds every product and sum separately, that a
// configuration is refused by the field at fault, the tie order of the worst face, a
// point's several notices served one at a time, and the spacing's ceiling.

#include "solprop_native/transverse.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace transverse = solprop_native::transverse;
using Kind = transverse::Notice::Kind;

volatile double factor_a = 1.0 + 0x1p-30;
volatile double factor_b = 1.0 - 0x1p-30;
volatile double addend = -1.0;

// This file is compiled with the build's contraction (solprop #135's control).
[[gnu::noinline]] double contracting(const double a, const double b, const double c) {
  const double product = a * b;
  return product + c;
}

// Four columns at wavenumbers 0, 1e6, 2e6 and 3e6 rad/m, weighted one each; a safety
// ramp on the top one when `ramp`.
transverse::Config config(std::vector<std::string> faces = {"x-", "x+", "y-", "y+"},
                          const bool ramp = false) {
  transverse::Config config;
  config.faces = std::move(faces);
  config.warn = 1e-6;
  config.stop = 1e-4;
  config.length = 0.016;
  config.guard = 1e-5;
  config.k_min = 7.85e6;
  config.k_floor = 1e-3;
  config.columns = {1.0, 1.0, 1.0, 1.0};
  config.kappa = {0.0, 1e6, 2e6, 3e6};
  config.kappa_order = {0, 1, 2, 3};
  if (ramp) {
    config.design = true;
    config.ramp_rate = {0.0, 0.0, 0.0, 1e4};
  }
  return config;
}

// A launch of unit energy with nothing at the faces, all of it in the first column.
transverse::Reading launch_reading() {
  transverse::Reading reading;
  reading.faces = std::array<double, 4>{};
  reading.clip = std::array<double, 4>{};
  reading.totals = std::array<double, 4>{1.0, 0.0, 0.0, 0.0};
  reading.columns = {1.0, 0.0, 0.0, 0.0};
  return reading;
}

transverse::Reading read_reading(const std::array<double, 4> faces,
                                 std::vector<double> columns = {1.0, 0.0, 0.0, 0.0}) {
  transverse::Reading reading;
  reading.faces = faces;
  reading.columns = std::move(columns);
  return reading;
}
} // namespace

TEST(TransverseEdge, TheJudgementRoundsEveryOperation) {
  EXPECT_EQ(transverse::strict_probe(factor_a, factor_b, addend), 0.0);
}

TEST(TransverseEdge, TheJudgementStaysStrictWhereTheBuildFuses) {
#if defined(__OPTIMIZE__) && defined(__FP_FAST_FMA) && !defined(__CUDACC__) &&         \
    !defined(__HIPCC__)
  ASSERT_EQ(contracting(factor_a, factor_b, addend), -0x1p-60)
      << "the control did not fuse, so the witness proves nothing";
  EXPECT_EQ(transverse::strict_probe(factor_a, factor_b, addend), 0.0);
#else
  const double value = contracting(factor_a, factor_b, addend);
  EXPECT_TRUE(value == 0.0 || value == -0x1p-60);
  GTEST_SKIP() << "an unoptimised build, or a target without fused multiply-add, "
                  "cannot show a contraction";
#endif
}

TEST(TransverseEdge, AConfigurationIsRefusedByTheFieldAtFault) {
  const auto refused = [](const transverse::Config& bad, const std::string& field) {
    try {
      transverse::Edge edge(bad);
      ADD_FAILURE() << "accepted a configuration with a bad " << field;
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string(error.what()).find(field), std::string::npos)
          << error.what();
    }
  };
  refused(config({"x-", "x+", "y-"}), "faces");
  auto order = config();
  order.kappa_order = {0, 1, 1, 3};
  refused(order, "permutation");
  auto ramp = config({"r"}, true);
  ramp.ramp_rate.pop_back();
  refused(ramp, "ramp rate");
  auto collar = config();
  collar.collar = true;
  refused(collar, "collar");
}

TEST(TransverseEdge, TiedFacesStopOnTheGreatestName) {
  // `max` of `(share, name)` tuples: "+" sorts before "-", so "y-" is the greatest.
  transverse::Edge edge(config());
  EXPECT_EQ(edge.launch(launch_reading(), 0.0).kind, Kind::none);
  const auto notice = edge.read(read_reading({2e-4, 2e-4, 2e-4, 2e-4}), 1e-4);
  EXPECT_EQ(notice.kind, Kind::face);
  EXPECT_TRUE(notice.stop);
  EXPECT_EQ(notice.face, 2U);
  EXPECT_EQ(edge.status(), transverse::Status::stopped);
  // A greater share wins over a greater name.
  transverse::Edge other(config());
  (void)other.launch(launch_reading(), 0.0);
  EXPECT_EQ(other.read(read_reading({3e-4, 2e-4, 2e-4, 2e-4}), 1e-4).face, 0U);
}

TEST(TransverseEdge, ALaunchServesEachOfItsWarningsInTurn) {
  // Two clipped faces and a schedule's step longer than the spacing: three notices,
  // the rest of the launch done once the last is served.
  auto settings = config({"x-", "x+"});
  settings.longest_step = 0.01;
  transverse::Edge edge(settings);
  auto reading = launch_reading();
  reading.clip = std::array<double, 4>{2e-6, 3e-6, 0.0, 0.0};
  auto notice = edge.launch(reading, 0.0);
  EXPECT_EQ(notice.kind, Kind::clip);
  EXPECT_EQ(notice.face, 0U);
  EXPECT_DOUBLE_EQ(notice.value, 2e-6);
  EXPECT_TRUE(edge.pending());
  EXPECT_FALSE(std::isfinite(edge.next_read())) << "the spacing was set too early";
  notice = edge.proceed();
  EXPECT_EQ(notice.kind, Kind::clip);
  EXPECT_EQ(notice.face, 1U);
  notice = edge.proceed();
  EXPECT_EQ(notice.kind, Kind::coarse);
  EXPECT_DOUBLE_EQ(notice.value, 0.01);
  EXPECT_DOUBLE_EQ(notice.spacing, 0.001); // L / 16, the spectrum having no reach
  EXPECT_EQ(edge.status(), transverse::Status::coarse);
  EXPECT_EQ(edge.proceed().kind, Kind::none);
  EXPECT_FALSE(edge.pending());
  EXPECT_DOUBLE_EQ(edge.next_read(), 0.001);
  EXPECT_EQ(edge.proceed().kind, Kind::none);
}

TEST(TransverseEdge, AReadWarnsOnAFaceThenJudgesTheRamp) {
  transverse::Edge edge(config({"r"}, true));
  EXPECT_EQ(edge.launch(launch_reading(), 0.0, nullptr).kind, Kind::none);
  // The top column fills at a read the ramp has had no time to act on.
  EXPECT_EQ(edge.read(read_reading({0.0, 0, 0, 0}, {1.0, 0, 0, 0.5}), 1e-9).kind,
            Kind::none);
  const auto warned = edge.read(read_reading({3e-5, 0, 0, 0}, {1.0, 0, 0, 0.5}), 1e-3);
  EXPECT_EQ(warned.kind, Kind::face);
  EXPECT_FALSE(warned.stop);
  EXPECT_DOUBLE_EQ(warned.z, 1e-3);
  EXPECT_TRUE(edge.pending());
  const auto stopped = edge.proceed();
  EXPECT_EQ(stopped.kind, Kind::ramp_removal);
  EXPECT_TRUE(stopped.stop);
  EXPECT_NEAR(stopped.value, -0.5 * std::expm1(-1e4 * (1e-3 - 1e-9)), 1e-15);
  EXPECT_FALSE(edge.pending());
  EXPECT_EQ(edge.status(), transverse::Status::stopped);
}

TEST(TransverseEdge, TheSpacingIsTheGuardOverTheReachUnderTheBackstop) {
  // Content at 3e6 rad/m above the warning: the reach is kappa / k_z, the spacing
  // 4 g / v_q; nothing above it leaves the backstop, which doubles while it holds.
  transverse::Edge edge(config());
  auto reading = launch_reading();
  reading.columns = {1.0, 0.0, 0.0, 1e-3};
  (void)edge.launch(reading, 0.0);
  const double k_z = std::sqrt(7.85e6 * 7.85e6 - 3e6 * 3e6);
  EXPECT_DOUBLE_EQ(edge.ceiling(), 4.0 * 1e-5 / (3e6 / k_z));
  EXPECT_TRUE(edge.due(0.0, edge.ceiling()));
  EXPECT_FALSE(edge.due(0.0, 0.5 * edge.ceiling()));
  transverse::Edge calm(config());
  (void)calm.launch(launch_reading(), 0.0);
  EXPECT_DOUBLE_EQ(calm.ceiling(), 0.001);
  // No reach at all: the backstop stays at its start.
  (void)calm.read(read_reading({0.0, 0, 0, 0}), 0.001);
  EXPECT_DOUBLE_EQ(calm.ceiling(), 0.001);
  EXPECT_FALSE(calm.wants_read(0.001));
  EXPECT_TRUE(calm.wants_read(std::nextafter(0.001, 1.0)));
}

TEST(TransverseEdge, AResumeTakesTheLaunchUnjudgedAndKeepsTheRecord) {
  transverse::Resumption record;
  record.record = true;
  record.launch_energy = 2.0;
  record.edge_baseline = std::vector<double>{1e-7, 2e-7};
  record.edge_max = std::vector<double>{3e-7, 4e-7, 5e-7}; // does not fit: kept as is
  record.warned_at = 0.004;
  record.reads = 7;
  record.status = "warned";
  transverse::Edge edge(config({"x-", "x+"}));
  edge.resume(record);
  ASSERT_TRUE(edge.resumed());
  auto reading = read_reading({1.0, 1.0, 0, 0});
  EXPECT_EQ(edge.launch(reading, 0.005).kind, Kind::none);
  const auto kept = edge.record();
  EXPECT_DOUBLE_EQ(kept.launch_energy, 2.0);
  EXPECT_EQ(kept.edge_baseline, (std::vector<double>{1e-7, 2e-7}));
  EXPECT_EQ(kept.edge_max, (std::vector<double>{0.0, 0.0}));
  EXPECT_EQ(kept.reads, 7);
  EXPECT_EQ(kept.status, transverse::Status::warned);
  EXPECT_TRUE(kept.resumed);
  // It warned before: a face past the warning does not warn again.
  EXPECT_EQ(edge.read(read_reading({2.0 * 3e-5, 0, 0, 0}), 0.006).kind, Kind::none);
  // A record without a fitting baseline starts the edge again.
  record.edge_baseline = std::vector<double>{1e-7};
  transverse::Edge fresh(config({"x-", "x+"}));
  fresh.resume(record);
  EXPECT_FALSE(fresh.resumed());
}

TEST(TransverseEdge, ACollarsCurrentIsIntegratedByTheTrapezoidRule) {
  auto settings = config({"x-", "x+"});
  settings.design = true;
  settings.ramp_rate = {0.0, 0.0, 0.0, 0.0};
  settings.collar = true;
  settings.collar_rate = true;
  transverse::Edge edge(settings);
  auto reading = launch_reading();
  reading.strip = std::vector<double>{0.0, 0.0, 0.0, 0.0};
  reading.totals = std::array<double, 4>{2.0, 0.0, 1e-4, 0.0};
  (void)edge.launch(reading, 0.0);
  ASSERT_TRUE(edge.every_step());
  edge.step(0.0, 0.0);
  edge.step(2e-3, 1e-3);
  edge.step(2e-3, 3e-3);
  const auto kept = edge.record();
  // Shares of 0, 1e-3 and 1e-3 of the launch: 0.5e-6 + 2e-6.
  EXPECT_DOUBLE_EQ(kept.current,
                   0.5 * 1e-3 * (0.0 + 1e-3) + 0.5 * 2e-3 * (1e-3 + 1e-3));
  EXPECT_DOUBLE_EQ(kept.current_baseline, 5e-5);
}
