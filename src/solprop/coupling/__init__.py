# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Couple simulations through a coherent field at a reference plane (SEP 31).

A physics-layer package: NumPy, SciPy and other physics modules only. A
:class:`~solprop.coupling.plane.FieldPlane` is one physical plane of a field in its
native representation, with its references, geometry, medium, metric and provenance;
:func:`~solprop.coupling.remap.regrid` moves it onto another temporal lattice or
reference; an :class:`~solprop.coupling.port.InputPort` describes what a receiving
simulation accepts at its entrance, and :func:`~solprop.coupling.couple.couple`
launches a plane into it under a declared
:class:`~solprop.coupling.models.CouplingModel`;
:class:`~solprop.coupling.outcouple.Outcouple` synthesises a guide's exit field.
Extraction from a saved run is :func:`solprop.processing.planes.field_plane`;
installing a launch is :meth:`solprop.driver.simulation.Simulation.with_input`.
The manual page is ``docs/guide/workflows/coupling.md``.
"""

from solprop.coupling.couple import (
    AdmissionItem,
    CouplingReport,
    CouplingResult,
    couple,
)
from solprop.coupling.envelopes import (
    CAPILLARY_ENVELOPE,
    STEP_INDEX_ENVELOPE,
    PairEnvelope,
)
from solprop.coupling.models import (
    BasisChange,
    CouplingModel,
    ScalarModeBasis,
)
from solprop.coupling.outcouple import Outcouple, OutcoupleResult
from solprop.coupling.plane import (
    PLANE_KINDS,
    AnalyticRecord,
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
)
from solprop.coupling.remap import DEFAULT_TOLERANCE, RegridReport, regrid

# The transport core (solprop.coupling.transport and solprop.coupling.elements:
# transport, FreeSpace, ThinLens, Aperture, TransmissionMask, PlanarInterface,
# PolarisationTransform, Mirror, SpectralTransfer) is exported here when it lands,
# beside the names below. Outcouple lives in solprop.coupling.outcouple.

__all__ = [
    "CAPILLARY_ENVELOPE",
    "DEFAULT_TOLERANCE",
    "ENTRANCE_TOLERANCE",
    "METRIC_TOLERANCE",
    "PLANE_KINDS",
    "STEP_INDEX_ENVELOPE",
    "AdmissionItem",
    "AnalyticRecord",
    "BasisChange",
    "Calibration",
    "CouplingLaunch",
    "CouplingModel",
    "CouplingReport",
    "CouplingResult",
    "Exterior",
    "FieldPlane",
    "InputPort",
    "MetricKind",
    "NodalField",
    "Outcouple",
    "OutcoupleResult",
    "PairEnvelope",
    "PlaneBasis",
    "PlaneGeometry",
    "PlaneKind",
    "PlaneMetric",
    "PlaneProvenance",
    "PlaneReference",
    "RegridReport",
    "ScalarModeBasis",
    "basis_identity",
    "couple",
    "lattice_identity",
    "modal_basis_identity",
    "reference_at",
    "regrid",
    "space_record",
]
