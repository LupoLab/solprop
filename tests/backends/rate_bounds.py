# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The rounding bound of one evaluation of the ionisation rate table.

Held apart from the tests whose budgets it enters, such as the planar ladder's peak
rate (``test_planar_hybrid.py``).
"""

import numpy as np

from bounds import EPS, gamma


def rate_error_bound(real, table, rate):
    """Bound one evaluation of exp(cubic(log |E|)) from its actual operands.

    With normal field components, gamma(8) bounds the norm's relative error.
    Assuming one-epsilon log/exp, delta(log E) <= gamma(8) + eps |log E|.
    Forming x = log E - (log_min + index*d) adds gamma(3) times the sum
    of those absolute operands. The cubic contributes its derivative bound
    times delta(x), plus gamma(6) times its absolute Horner polynomial.
    Exponentiation turns delta(p) into expm1(delta(p)) relative error.

    Near an interval boundary, either adjacent stored polynomial can be
    selected. Their evaluated difference and both evaluation bounds cover
    any rounding discontinuity without assuming exact spline continuity.
    These stress inputs stay away from the table's discontinuous lower clip.
    No compiled result or observed discrepancy enters this calculation.
    """
    magnitude = np.sqrt(np.sum(real * real, axis=1))
    out = np.zeros_like(rate)
    uncertainty = gamma(8) * magnitude
    assert np.all(abs(magnitude - table.field_min) > uncertainty)
    assert np.all(abs(magnitude - table.field_max) > uncertainty)
    inside = (magnitude >= table.field_min) & (magnitude <= table.field_max)
    logarithm = np.log(magnitude[inside])
    log_error = gamma(8) + EPS * abs(logarithm)
    coordinate = (logarithm - table.log_field_min) / table.d_log_field
    coordinate_error = (
        log_error + gamma(2) * (abs(logarithm) + abs(table.log_field_min))
    ) / table.d_log_field
    last = table.coefficients.shape[1] - 1
    index = np.clip(coordinate.astype(np.intp), 0, last)

    def polynomial(indices):
        centre = table.log_field_min + indices * table.d_log_field
        local = logarithm - centre
        local_error = log_error + gamma(3) * (
            abs(logarithm) + abs(table.log_field_min) + abs(indices * table.d_log_field)
        )
        c = table.coefficients[:, indices]
        radius = abs(local) + local_error
        condition = (
            (abs(c[0]) * radius + abs(c[1])) * radius + abs(c[2])
        ) * radius + abs(c[3])
        derivative = (3 * abs(c[0]) * radius + 2 * abs(c[1])) * radius + abs(c[2])
        value = ((c[0] * local + c[1]) * local + c[2]) * local + c[3]
        return value, local_error * derivative + gamma(6) * condition

    value, base_error = polynomial(index)
    error = base_error.copy()
    for neighbour in (coordinate - coordinate_error, coordinate + coordinate_error):
        alternate = np.clip(np.floor(neighbour).astype(np.intp), 0, last)
        other_value, other_error = polynomial(alternate)
        error = np.maximum(
            error,
            np.where(
                alternate == index,
                base_error,
                abs(other_value - value) + other_error + base_error,
            ),
        )
    # Account for rounding of the reference exponential used to express the
    # absolute scale, and round the bound outward after its own arithmetic.
    out[inside] = rate[inside] * (np.expm1(error) + EPS) / (1 - EPS)
    return np.nextafter(out * (1 + gamma(32)), np.inf)
