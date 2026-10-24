# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Couple simulations through a coherent field at a reference plane (SEP 31).

A physics-layer package: NumPy, SciPy and other physics modules only. A
:class:`~solprop.coupling.plane.FieldPlane` is one physical plane of a field in its
native representation, with its references, geometry, medium, metric and provenance;
:func:`~solprop.coupling.remap.regrid` moves it onto another temporal lattice or
reference; an :class:`~solprop.coupling.port.InputPort` describes what a receiving
simulation accepts at its entrance, and :func:`~solprop.coupling.port.same_basis`
launches a plane into a port of the same basis. Extraction from a saved run is
:func:`solprop.processing.planes.field_plane`; installing a launch is
:meth:`solprop.driver.simulation.Simulation.with_input`. The manual page is
``docs/guide/workflows/coupling.md``.
"""

from solprop.coupling.elements import (
    Alignment,
    Aperture,
    FreeSpace,
    Mirror,
    PlanarInterface,
    PolarisationTransform,
    SpectralCalibration,
    SpectralTransfer,
    ThinLens,
    TransmissionMask,
)
from solprop.coupling.plane import (
    PLANE_KINDS,
    Calibration,
    Exterior,
    FieldPlane,
    MetricKind,
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
from solprop.coupling.transport import TransportReport, transport

__all__ = [
    "DEFAULT_TOLERANCE",
    "ENTRANCE_TOLERANCE",
    "METRIC_TOLERANCE",
    "PLANE_KINDS",
    "Alignment",
    "Aperture",
    "Calibration",
    "CouplingLaunch",
    "Exterior",
    "FieldPlane",
    "FreeSpace",
    "InputPort",
    "MetricKind",
    "Mirror",
    "NodalField",
    "PlanarInterface",
    "PlaneBasis",
    "PlaneGeometry",
    "PlaneKind",
    "PlaneMetric",
    "PlaneProvenance",
    "PlaneReference",
    "PolarisationTransform",
    "RegridReport",
    "SpectralCalibration",
    "SpectralTransfer",
    "ThinLens",
    "TransmissionMask",
    "TransportReport",
    "basis_identity",
    "lattice_identity",
    "modal_basis_identity",
    "reference_at",
    "regrid",
    "same_basis",
    "space_record",
    "transport",
]
