// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The entry point of helpers.cpp: the host helpers solprop's shared evaluations take
// from this backend, and the vector-maths batch functions for their tests.

#include <nanobind/nanobind.h>

namespace solprop_native::bindings {

/**
 * Add the host helpers solprop's per-stage and per-step evaluations take from this
 * backend (helpers.cpp): `clenshaw_points` to the extension module, and `basis_rows`,
 * `trial_tables` and the vector-maths layer's batch functions, for their tests, to the
 * private `_testing` module.
 */
void bind_helpers(nanobind::module_& module_, nanobind::module_& testing);

} // namespace solprop_native::bindings
