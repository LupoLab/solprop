// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The native session's transverse edge actions (solprop SEP 40 section 5.3, SEP 47
// sections 5.3 and 6.4): `StepDriver.attach_monitor`'s launch of the edge,
// `_read_transverse` at a committed endpoint and at a plane, and the barrier each
// warning makes. The edge itself is `transverse::Edge`; the readings are the resident
// state's (`Resident::edge_reading`), so nothing here returns to Python.

#include "solprop_native/session.hpp"

#include <optional>
#include <utility>

// As session.cpp: the scheduler's comparisons round as solprop's Python does.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace solprop_native::session {

double Session::next_step() const {
  // `StepDriver._next_step`: a replay's scheduled step, none past its end, else the
  // controller's proposal.
  return std::visit(
      [this](const auto& method) {
        if (!plan_.replay)
          return method.state.proposal;
        const auto& steps = method.schedule.steps;
        const auto index = static_cast<std::size_t>(method.state.cursor);
        return index < steps.size() ? steps[index] : 0.0;
      },
      method_);
}

transverse::Reading Session::edge_reading(const bool faces, const bool whole,
                                          const bool current) {
  // `NativeBackend._edge_reading`: a collar's totals and, with the faces, its strip.
  const bool collar = edge_->every_step();
  return resident_->edge_reading(faces, whole, collar, faces && collar, current);
}

bool Session::edge_judged(Batch& batch, const transverse::Notice& notice,
                          const Continuation continued) {
  // A stop ends the run at this committed endpoint; a warning is a barrier, whose
  // continuation serves the rest of its point.
  if (notice.kind == transverse::Notice::Kind::none)
    return false;
  Notice reported;
  reported.kind = Notice::Kind::transverse;
  reported.z = here();
  reported.edge = notice;
  if (notice.stop) {
    monitor_stop(batch, reported);
    return true;
  }
  continuation_ = continued;
  warning(batch, reported);
  return true;
}

bool Session::edge_proceed(Batch& batch, const Continuation continued) {
  return edge_judged(batch, edge_->proceed(), continued);
}

bool Session::edge_launch(Batch& batch) {
  // A resumed launch reads no share, but continues a collar's current.
  const bool resumed = edge_->resumed();
  const transverse::Reading reading =
      edge_reading(true, !resumed, edge_->every_step() && resumed);
  return edge_judged(batch, edge_->launch(reading, position()),
                     Continuation::edge_launch);
}

bool Session::edge_endpoint(Batch& batch) {
  // `_read_transverse(plane=False)`: read before the next step would carry past the
  // read, and a collar's current at every committed step, from one reading.
  const double z = position();
  const bool read = edge_->due(z, next_step()) && edge_->wants_read(z);
  const bool accumulate = edge_->every_step();
  if (!read && !accumulate)
    return false;
  const transverse::Reading reading = edge_reading(read, false, accumulate);
  if (accumulate)
    edge_->step(reading.current.value_or(0.0), z);
  if (!read)
    return false;
  return edge_judged(batch, edge_->read(reading, z), Continuation::edge_endpoint);
}

bool Session::edge_plane(Batch& batch) {
  // `_read_transverse(plane=True)`: a plane the step that reached it already read has
  // nothing to add, and no current is taken; an interpolated plane is read at its own
  // position.
  const double z = here();
  if (!edge_->wants_read(z))
    return false;
  return edge_judged(batch, edge_->read(edge_reading(true, false, false), z),
                     Continuation::edge_plane);
}

std::optional<std::pair<transverse::Record, transverse::Finish>>
Session::edge_record() const {
  const std::scoped_lock lock(mutex_);
  if (!edge_)
    return std::nullopt;
  return std::make_pair(edge_->record(), edge_->finish());
}

} // namespace solprop_native::session
