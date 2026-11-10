// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/transverse.hpp"

#include <nanobind/nanobind.h>

namespace solprop_native::bindings {

/*
 * The transverse edge's bindings (solprop SEP 47 section 6.4): its configuration, the
 * record a resume continues, its record and its notices, from and to solprop's
 * dictionaries (`solprop_native._transverse`), and `TransverseEdge`, the edge itself,
 * for the recorded-readings harness.
 */

/** The edge's configuration from `EdgeConfig.native()`; throws `std::invalid_argument`
    for a missing field or a wrong type. */
transverse::Config transverse_config_from(nanobind::dict values);
/** The record a resume continues, from `_transverse.resumption(...)`. */
transverse::Resumption transverse_resumption_from(nanobind::dict values);
/** The input noise's readings a launch carries (`noise` of `TransverseEdge.launch`,
    `_transverse.noise_readings`). */
transverse::Noise transverse_noise_from(nanobind::dict values);
/** The edge's record and finish, as `_transverse.record_of` reads them. */
nanobind::dict transverse_record_to(const transverse::Record& record,
                                    const transverse::Finish& finish);
/** A notice, as `_transverse.EdgeNotice.of` reads it. */
nanobind::dict transverse_notice_to(const transverse::Notice& notice);
/** Add `TransverseEdge` to the extension module. */
void bind_transverse(nanobind::module_& module_);

} // namespace solprop_native::bindings
