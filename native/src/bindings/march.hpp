// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/march/adams_march.hpp"
#include "solprop_native/march/rk_march.hpp"
#include "solprop_native/session/watch.hpp"

#include <nanobind/nanobind.h>

/*
 * The native marches' records from and to solprop's dictionaries (march.cpp): the
 * Adams march's policy, state, bounds and result (solprop SEP 39 phase 2), the
 * Runge-Kutta march's state and result, and the boundary watch's policy, state,
 * configuration, record and judgement (solprop SEP 45 phase 1, SEP 47 section 6.4).
 */
namespace solprop_native::bindings {

namespace nb = nanobind;
namespace native = solprop_native;

native::march::Policy march_policy_from(nb::dict values);
native::watch::Policy watch_policy_from(nb::dict values);
native::watch::State watch_state_from(nb::dict values);
nb::dict watch_state_to(const native::watch::State& state);
native::watch::Config watch_config_from(nb::handle values);
nb::dict watch_record_to(const native::watch::Record& record);
nb::dict watch_judgement_to(const native::watch::Judgement& judgement);
native::rk::State rk_state_from(nb::dict values);
nb::dict rk_result_to(const native::rk::Result& out, const native::rk::State& state);
native::march::State march_state_from(nb::dict values);
nb::dict march_state_to(const native::march::State& state);
native::march::Bounds march_bounds_from(nb::dict values);
nb::dict march_result_to(const native::march::Result& result,
                         const native::march::State& state);

} // namespace solprop_native::bindings
