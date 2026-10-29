// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <nanobind/nanobind.h>

namespace solprop_cpunative::bindings {

/**
 * Add the host helpers solprop's per-stage and per-step evaluations take from
 * this backend to the extension module: `clenshaw_points` and `basis_rows`
 * (helpers.cpp).
 */
void bind_helpers(nanobind::module_& module_);

} // namespace solprop_cpunative::bindings
