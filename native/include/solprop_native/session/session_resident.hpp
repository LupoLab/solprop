// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// What a native session (`session.hpp`) reads from and writes through the state it
// leases, behind one interface the resident engines implement (`engines.hpp`), so that
// they need not see the session. Host code only.
//
// Types: `StatisticsRecipe` (a prepared `statistics::Plan`) and `Resident`, the
// interface for the leased state's field transfers, statistics, boundary and edge
// readings and collar flags.

#include "solprop_native/session/statistics_plan.hpp"
#include "solprop_native/session/transverse.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace solprop_native::session {

/** A statistics evaluation the resident state makes, with every argument fixed at
    preparation (`statistics::Plan`, solprop SEP 47 section 6.4); none without a
    plan. */
struct StatisticsRecipe {
  std::shared_ptr<const statistics::Plan> plan;
  [[nodiscard]] bool empty() const noexcept { return !plan || plan->empty(); }
};

/** What a session reads from and writes through the state it leases: the field and
    the statistics, where they live. The resident engines' implementation is
    `engines::GeneralResident`. */
class Resident {
public:
  Resident() = default;
  Resident(const Resident&) = delete;
  Resident& operator=(const Resident&) = delete;
  Resident(Resident&&) = delete;
  Resident& operator=(Resident&&) = delete;
  virtual ~Resident() = default;
  /** The field's complex element count, `columns * n_omega`. */
  [[nodiscard]] virtual std::size_t elements() const = 0;
  /** The field's last axis, the frequency bins, along which `(-1)^k` alternates. */
  [[nodiscard]] virtual std::size_t n_omega() const = 0;
  /** The committed field as host memory, interleaved real and imaginary parts, or
      null where it is not host-accessible. */
  virtual const double* host_field() = 0;
  /** Copy the committed field into `out`, `2 elements()` doubles. */
  virtual void field_to_host(double* out) = 0;
  /** Count bytes a host reader of `host_field` took out of the state. */
  virtual void count_to_host(std::size_t bytes) = 0;
  /** Evaluate `recipe` on the committed state at `z`. */
  virtual std::vector<double> statistics(const StatisticsRecipe& recipe, double z) = 0;
  /** The committed state's boundary reading (`monitor_reading`, `watch::reading_head`
      doubles and the occupancy buckets), for the monitor's launch and plane checks. */
  virtual std::vector<double> boundary_reading() {
    throw std::logic_error("this state takes no boundary readings");
  }
  /** Leave the in-stage temporal collar out of the stages, or put it back, dropping
      what was evaluated under the other setting (`HostSession._set_collar_quiet`). */
  virtual void set_collar_quiet(bool /*quiet*/) {
    throw std::logic_error("this state cannot rest its collar");
  }
  /** The largest accepted step's transverse quadrature score since the last read,
      clearing it (`quadrature_peak`); NaN where none was scored. */
  virtual double quadrature_peak() { return std::numeric_limits<double>::quiet_NaN(); }
  /** The committed state's transverse edge reading (`state.edge_reading`, SEP 44): the
      faces' sums and the columns when `faces`, with the clip and totals of every node
      when `whole`, the totals when `totals`, a collar's strip when `strip` and its
      current when `current`. */
  virtual transverse::Reading edge_reading(bool /*faces*/, bool /*whole*/,
                                           bool /*totals*/, bool /*strip*/,
                                           bool /*current*/) {
    throw std::logic_error("this state takes no transverse edge readings");
  }
};

} // namespace solprop_native::session
