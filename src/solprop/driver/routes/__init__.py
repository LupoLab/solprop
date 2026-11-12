# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The keyword routes that build a :class:`solprop.driver.simulation.Simulation`.

One module per family of constructor: :mod:`~solprop.driver.routes.gnlse`,
:mod:`~solprop.driver.routes.hollow` (capillaries and supplied hollow guides),
:mod:`~solprop.driver.routes.general` (general modal guides) and
:mod:`~solprop.driver.routes.spatial` (free space and planar grids), with the material
selection they share in :mod:`~solprop.driver.routes.materials` and the naming of the
keyword a refusal concerns in :mod:`~solprop.driver.routes.keywords`.
"""
