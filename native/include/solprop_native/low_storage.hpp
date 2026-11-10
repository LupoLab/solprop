// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "solprop_native/core.hpp"
#include "solprop_native/engines.hpp"
#include "solprop_native/rk_march.hpp"
#include "solprop_native/stage_tables.hpp"

/*
 * The low-storage RK4IP replay in the native session (solprop SEP 24, SEP 47 section
 * 6.3): the engine that takes one in-place step of the march from its stage scalars,
 * which it forms natively from the uploaded stage tables, and the table keys of the
 * schedule it replays. The replay loop itself is the Runge-Kutta march's
 * (`rk::run` with a schedule), whose every step this engine takes: the schedule's steps
 * as given, never rejected, each committed and recorded in order.
 *
 * Each function performs the operations of the solprop function it is named after in
 * the same order: `table_steps` and `schedule_key` of `solprop.backends.low_storage`,
 * and `scalars`, the arrays the native backend's `_ResidentLowStorage.step` packs from
 * its bundles before its native call. The translation unit is compiled without
 * contraction on every build, as the marches' are.
 *
 * A step is accepted in place: from its first pass to its finish the state is neither
 * end of the step, and a failure there -- a native exception, or a non-finite norm --
 * leaves no consistent state behind (`ResidentLowStorage::in_flight`), which the
 * session reports as a poisoned terminal with no checkpoint (SEP 47 section 5.5).
 *
 * Host code only.
 */
namespace solprop_native::rk::low_storage {

/** The flows a step holds and where its operator starts, as the native backend's
    adapter resolved them (`_ResidentLowStorage`). */
struct Config {
  /** The native problem's `low_storage_flows`: 0 recomputes every flow per element,
      1 holds a constant operator's table, 2 a varying operator's `P` and `Q`. */
  std::size_t flows = 0;
  /** A separable geometry's operator, whose increments are the distances themselves. */
  bool separable = false;
  /** The operator's origin, `linear.z_span[0]`, m, from which a table is built. */
  double origin = 0.0;
};

/**
 * `solprop.backends.low_storage.table_steps`: the key each of a schedule's steps builds
 * its table on, the mean step of its run -- the longest stretch of steps within
 * `landing_ulps` ulps of the run's first, the ulps of the run's farthest position or of
 * the step -- or its own step for a run of one.
 */
[[nodiscard]] std::vector<double> table_steps(const std::vector<double>& steps,
                                              const std::vector<double>& edges,
                                              int landing_ulps);

/** `schedule_key`: the key of the schedule step from `z`, found by its midpoint. */
[[nodiscard]] double schedule_key(const std::vector<double>& keys,
                                  const std::vector<double>& edges, double z,
                                  double dz);

/**
 * A step's scalars from `z` over `dz`, as `_ResidentLowStorage.step` packs them: each
 * of the start, the midpoint and the end read as its own anchor, the end's row repeated
 * to the tableau's stages; the increment's rows the flows' -- a separable operator's
 * half steps, or a varying operator's `Phi(z_m) - Phi(z)` and `Phi(z + h) - Phi(z_m)`
 * where its flows are formed for the step -- and, when `rebuild`, the table's interval
 * from the operator's origin to half its `key`. A z table's three positions are loaded
 * together first, as the host loads them.
 */
[[nodiscard]] TrialScalars scalars(const stage::Tables& tables, stage::Anchor& anchor,
                                   const Config& config, double z, double dz,
                                   bool rebuild, double key);

/** The strictness test's witness, as this translation unit rounds `a b + c`; never
    inlined, as `march::strict_probe`. */
[[nodiscard, gnu::noinline]] double strict_probe(double a, double b, double c);

} // namespace solprop_native::rk::low_storage

namespace solprop_native::engines {

/**
 * The resident low-storage engine (solprop SEP 24 section 6.7, SEP 47 section 6.3): a
 * trial is one whole in-place step of the four register passes, its scalars formed
 * natively, scoring 0.0 ("not estimated") or, when the field diverged, the accepted
 * state's non-finite squared norm; the accept is the backend's finish and the
 * accepted-step hook. A constant operator's table is rebuilt only where the schedule's
 * key changes, as `_ResidentLowStorage` keeps it.
 */
class ResidentLowStorage final : public rk::Engine {
public:
  ResidentLowStorage(GeneralState& state, const stage::Tables& tables,
                     stage::Anchor& anchor, rk::low_storage::Config config,
                     const march::Schedule& schedule, int landing_ulps,
                     std::optional<MarchFault>& fault);
  double trial(double z, double dz) override;
  void accept(double dz) override;
  std::vector<double> reading() override;
  [[nodiscard]] bool in_flight() const noexcept override { return in_flight_; }

private:
  void inject(march::Stage stage);

  GeneralState& state_;
  const stage::Tables& tables_;
  stage::Anchor& anchor_;
  rk::low_storage::Config config_;
  std::vector<double> keys_;
  std::vector<double> edges_;
  std::optional<MarchFault>& fault_;
  /** The key the held table was built for; NaN before the first step. */
  double table_step_;
  bool in_flight_ = false;
};

} // namespace solprop_native::engines
