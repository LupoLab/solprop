# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Couple simulations through a coherent field at a reference plane (SEP 31).

A physics-layer package: NumPy, SciPy and other physics modules only. A
:class:`FieldPlane` is one physical plane of a field in its native representation,
with its references, geometry, medium, metric and provenance; :func:`regrid` moves
it onto another temporal lattice or reference; an :class:`InputPort` describes what
a receiving simulation accepts at its entrance, and :func:`same_basis` launches a
plane into a port of the same basis. Extraction from a saved run is
:func:`solprop.processing.field_plane`; installing a launch is
:meth:`solprop.driver.Simulation.with_input`. The manual page is
``docs/guide/workflows/coupling.md``.
"""

from solprop.coupling.plane import (
    PLANE_KINDS,
    Calibration,
    Exterior,
    FieldPlane,
    NodalField,
    PlaneBasis,
    PlaneGeometry,
    PlaneKind,
    PlaneMetric,
    PlaneProvenance,
    PlaneReference,
    basis_identity,
    lattice_identity,
    modal_basis_identity,
    reference_at,
    space_record,
)
from solprop.coupling.port import (
    ENTRANCE_TOLERANCE,
    METRIC_TOLERANCE,
    CouplingLaunch,
    InputPort,
    same_basis,
)
from solprop.coupling.remap import DEFAULT_TOLERANCE, RegridReport, regrid

# The transport core (solprop.coupling.transport and solprop.coupling.elements:
# transport, FreeSpace, ThinLens, Aperture, TransmissionMask, PlanarInterface,
# PolarisationTransform, Mirror, SpectralTransfer, Outcouple) is exported here when
# it lands, beside the names below.

__all__ = [
    "DEFAULT_TOLERANCE",
    "ENTRANCE_TOLERANCE",
    "METRIC_TOLERANCE",
    "PLANE_KINDS",
    "Calibration",
    "CouplingLaunch",
    "Exterior",
    "FieldPlane",
    "InputPort",
    "NodalField",
    "PlaneBasis",
    "PlaneGeometry",
    "PlaneKind",
    "PlaneMetric",
    "PlaneProvenance",
    "PlaneReference",
    "RegridReport",
    "basis_identity",
    "lattice_identity",
    "modal_basis_identity",
    "reference_at",
    "regrid",
    "same_basis",
    "space_record",
]
