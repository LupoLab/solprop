// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/session/transverse.hpp"

#include <nanobind/nanobind.h>

namespace solprop_native::bindings {

/*
 * The transverse edge's bindings (solprop SEP 47 section 6.4): its record and its
 * notices, to solprop's dictionaries (`solprop_native._transverse`), and
 * `TransverseEdge`, the edge itself, for the recorded-readings harness, which reads
 * its configuration, the record a resume continues and a launch's noise readings
 * (transverse.cpp).
 */

/** The edge's record and finish, as `_transverse.record_of` reads them. */
nanobind::dict transverse_record_to(const transverse::Record& record,
                                    const transverse::Finish& finish);
/** A notice, as `_transverse.EdgeNotice.of` reads it. */
nanobind::dict transverse_notice_to(const transverse::Notice& notice);
/** Add `TransverseEdge` to the given module: the private `_testing` module. */
void bind_transverse(nanobind::module_& module_);

} // namespace solprop_native::bindings
