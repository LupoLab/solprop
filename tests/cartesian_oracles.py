# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent decimal arithmetic on the unsquared Cartesian inputs."""

from decimal import Decimal, localcontext

import numpy as np
from jaxtyping import Complex

from solprop.constants import C


def longitudinal_square(
    space, grid, index=1.0
) -> Complex[np.ndarray, "n_modes n_omega"]:
    """Evaluate the exact-input wave-number difference to 100 decimal digits."""
    index = np.broadcast_to(np.asarray(index, complex), grid.omega.shape)
    result = np.empty((space.n_columns, grid.n_omega), dtype=np.complex128)
    with localcontext() as context:
        context.prec = 100
        light_speed = Decimal.from_float(C)
        x_square = [Decimal.from_float(float(one)) ** 2 for one in space.k_x]
        y_square = [Decimal.from_float(float(one)) ** 2 for one in space.k_y]
        transverse = [x + y for y in y_square for x in x_square]
        for column, radial_square in enumerate(transverse):
            for frequency, omega in enumerate(grid.omega):
                scale = Decimal.from_float(float(omega)) / light_speed
                real = scale * Decimal.from_float(float(index[frequency].real))
                imag = scale * Decimal.from_float(float(index[frequency].imag))
                result[column, frequency] = complex(
                    float(real * real - imag * imag - radial_square),
                    float(2 * real * imag),
                )
    return result
