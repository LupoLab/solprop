# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent axial-admittance oracles for the medium flux measure (SEP 34).

The plane measure weighs each transverse column and frequency bin by the real
dimensionless admittance ``c Re(k_z) / omega`` with
``k_z = sqrt((omega n / c)**2 - k_perp**2)`` on the forward branch. Only active,
positive-frequency propagating bins are forward power ports; every other bin has
none. These helpers derive that weight from the index directly, without the
package's flux code, so a test can state its expectation independently.
"""

import numpy as np

from solprop.constants import C


def admittance(grid, transverse, index=1.0):
    """Return the real admittance ``c Re(k_z)/omega`` per column and bin.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Temporal grid; its active mask and angular frequencies.
    transverse : numpy.ndarray
        Squared transverse wave numbers ``k_perp**2`` in m^-2, one per column.
    index : complex or numpy.ndarray, optional
        Refractive index, scalar or one per bin. Default vacuum.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_columns, n_omega)``; in vacuum the obliquity
        ``sqrt(1 - (c k_perp/omega)**2)``, at most one.
    """
    omega = grid.omega
    live = grid.active_mask & (omega > 0)
    index = np.broadcast_to(np.asarray(index, dtype=complex), omega.shape)[live]
    transverse = np.asarray(transverse, dtype=float)
    square = (omega[live] * index / C) ** 2 - transverse[:, None]
    weight = np.zeros((transverse.size, grid.n_omega))
    weight[:, live] = np.where(
        np.real(square) > 0, C * np.real(np.sqrt(square)) / omega[live], 0.0
    )
    return weight
