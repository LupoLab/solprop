// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <nanobind/nanobind.h>

namespace solprop_native::bindings {

/**
 * Add the host helpers solprop's per-stage and per-step evaluations take from this
 * backend to the extension module: `clenshaw_points`, `basis_rows` and
 * `trial_tables` (helpers.cpp), and the vector-maths layer's batch functions for its
 * tests.
 */
void bind_helpers(nanobind::module_& module_);

} // namespace solprop_native::bindings
