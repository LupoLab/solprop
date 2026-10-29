// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The native Adams march's unit (solprop SEP 39 phase 2). Its decisions are held to
// solprop's on prescribed norms by solprop's decision-trace tests; these check what
// those cannot see: that the unit rounds every product and sum separately, that a
// policy is refused by the field at fault, and that a run on a toy engine keeps its
// ledger and lands on its target.

#include "solprop_native/adams_march.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {
namespace march = solprop_native::march;

volatile double factor_a = 1.0 + 0x1p-30;
volatile double factor_b = 1.0 - 0x1p-30;
volatile double addend = -1.0;

march::Policy policy() {
  march::Policy policy;
  policy.order = 5;
  policy.variable_order = true;
  policy.safety = 0.9;
  policy.max_growth = 2.0;
  policy.min_factor = 0.2;
  policy.rejection_growth = 0.9;
  policy.hysteresis = 1.2;
  policy.restart_after = 3;
  policy.collar_stability = 1.0;
  policy.real_intervals = {2.0, 1.0, 0.5, 0.3, 0.2, 0.1};
  policy.carrier_cap = 1.0;
  policy.equal_approach = true;
  policy.dz_min = 1e-9;
  policy.dz_max = 1e-2;
  policy.stretch = 1.0;
  policy.z_end = 0.1;
  policy.breakpoints = {0.0, 0.1};
  policy.landing_ulps = 8;
  return policy;
}

// A field whose every trial scores the same norms: every step is accepted.
class Steady final : public march::Engine {
public:
  void start(double /*z*/) override { ++starts; }
  void trial(double /*z*/, double /*dz*/, const march::Stencil& stencil,
             double* norms) override {
    for (std::size_t index = 0; index < stencil.n_orders; ++index)
      norms[index] = 0.25;
  }
  void accept(double /*dz*/, std::size_t /*free*/,
              const march::Stencil& /*stencil*/) override {
    ++accepts;
  }
  double chord_phase(double /*z1*/, double /*z2*/) override { return 0.0; }
  int starts = 0;
  int accepts = 0;
};
} // namespace

TEST(AdamsMarch, TheDecisionsRoundEveryOperation) {
  // The exact product 1 - 2^-60 rounds to 1: separately rounded, the sum is 0.
  EXPECT_EQ(march::strict_probe(factor_a, factor_b, addend), 0.0);
}

TEST(AdamsMarch, APolicyIsRefusedByTheFieldAtFault) {
  auto bad = policy();
  bad.safety = 1.5;
  try {
    march::validate(bad);
    FAIL() << "a safety above one was accepted";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("safety"), std::string::npos);
  }
  EXPECT_NO_THROW(march::validate(policy()));
}

TEST(AdamsMarch, ARunKeepsItsLedgerAndLandsOnItsTarget) {
  const auto rules = policy();
  march::State state;
  state.proposal = 1e-4;
  Steady engine;
  march::Bounds bounds;
  bounds.target = 0.05;
  bounds.max_attempts = 1000;
  bounds.max_seconds = 10.0;
  const auto result = march::run(rules, state, engine, bounds);
  ASSERT_EQ(result.reason, march::Reason::reached);
  EXPECT_TRUE(march::reached(state.z, bounds.target, rules.landing_ulps));
  EXPECT_EQ(engine.starts, 1);
  EXPECT_EQ(static_cast<std::uint64_t>(engine.accepts), state.ledger.accepted_steps);
  EXPECT_EQ(state.ledger.rejected_steps, 0U);
  EXPECT_EQ(state.ledger.total(), 1U + state.ledger.accepted_steps);
  EXPECT_EQ(result.rows.size(), march::row_columns * state.ledger.accepted_steps);
  EXPECT_EQ(result.orders.size(), state.ledger.accepted_steps);
  // The history starts at order one and never exceeds the policy's.
  EXPECT_EQ(result.orders.front(), 1U);
  for (const auto order : result.orders)
    EXPECT_LE(order, rules.order);
}
