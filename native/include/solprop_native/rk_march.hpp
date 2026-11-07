// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <vector>

/*
 * The Runge-Kutta march's arithmetic that must round as solprop's Python does (solprop
 * SEP 45). Phase 2: a trial's stage positions and its scaled error norm, the weak
 * norm's reduction of a trial's partials and its scale. This translation unit is
 * compiled without contraction and without fast-math on every build.
 */
namespace solprop_native::rk {

/** A trial's stage positions `z + c_i dz`, each product and sum rounded on its own as
    `[z + float(node) * dz for node in nodes]` is, then the step's end `z + dz`. */
[[nodiscard]] std::vector<double> positions(double z, double dz,
                                            const std::vector<double>& nodes);

/**
 * A trial's weak scaled error norm from its partials, `(3, slots)`: the sums of
 * `|y|^2`, `|y_new|^2` and `|e|^2` over the norm's bins, slot by slot. Each row is
 * summed in order, then `sqrt(e) / (rtol max(sqrt(y), sqrt(y_new)) + atol)`, Python's
 * `max` and the scale's operations in `weak_norm_scale`'s order (solprop SEP 45 section
 * 4.4: only the order of the sums differs from the host's NumPy reduction).
 */
[[nodiscard]] double score(const double* partials, std::size_t slots, double rtol,
                           double atol);

/** The strictness test's witness, as this translation unit rounds `a b + c`. */
[[nodiscard]] double strict_probe(double a, double b, double c);

} // namespace solprop_native::rk
