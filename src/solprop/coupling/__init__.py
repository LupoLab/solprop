# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Couple simulations through a coherent field at a reference plane (SEP 31).

A physics-layer package: NumPy, SciPy and other physics modules only. A
:class:`~solprop.coupling.plane.FieldPlane` is one physical plane of a field in its
native representation, with its references, geometry, medium, metric and provenance;
:func:`~solprop.coupling.remap.regrid` moves it onto another temporal lattice or
reference; an :class:`~solprop.coupling.port.InputPort` describes what a receiving
simulation accepts at its entrance, and :func:`~solprop.coupling.coupler.couple`
launches a plane into it under a declared
:class:`~solprop.coupling.models.CouplingModel`;
:class:`~solprop.coupling.outcouple.Outcouple` synthesises a guide's exit field,
and :func:`~solprop.coupling.transport` carries a spatial plane through
an ordered sequence of linear elements.
Extraction from a saved run is :func:`solprop.processing.planes.field_plane`;
installing a launch is :meth:`solprop.driver.simulation.Simulation.with_input`.
The manual page is ``docs/guide/workflows/coupling.md``.
"""

from solprop.coupling.coupler import (
    AdmissionItem,
    CouplingReport,
    CouplingResult,
    couple,
)
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
from solprop.coupling.envelopes import (
    CAPILLARY_ENVELOPE,
    STEP_INDEX_BEAM_ENVELOPE,
    STEP_INDEX_ENVELOPE,
    VECTOR_STEP_INDEX_ENVELOPE,
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
from solprop.coupling.transport import TransportReport, transport

__all__ = [
    "CAPILLARY_ENVELOPE",
    "DEFAULT_TOLERANCE",
    "ENTRANCE_TOLERANCE",
    "METRIC_TOLERANCE",
    "PLANE_KINDS",
    "STEP_INDEX_BEAM_ENVELOPE",
    "STEP_INDEX_ENVELOPE",
    "VECTOR_STEP_INDEX_ENVELOPE",
    "AdmissionItem",
    "Alignment",
    "AnalyticRecord",
    "Aperture",
    "BasisChange",
    "Calibration",
    "CouplingLaunch",
    "CouplingModel",
    "CouplingReport",
    "CouplingResult",
    "Exterior",
    "FieldPlane",
    "FreeSpace",
    "InputPort",
    "MetricKind",
    "Mirror",
    "NodalField",
    "Outcouple",
    "OutcoupleResult",
    "PairEnvelope",
    "PlanarInterface",
    "PlaneBasis",
    "PlaneGeometry",
    "PlaneKind",
    "PlaneMetric",
    "PlaneProvenance",
    "PlaneReference",
    "PolarisationTransform",
    "RegridReport",
    "ScalarModeBasis",
    "SpectralCalibration",
    "SpectralTransfer",
    "ThinLens",
    "TransmissionMask",
    "TransportReport",
    "basis_identity",
    "couple",
    "lattice_identity",
    "modal_basis_identity",
    "reference_at",
    "regrid",
    "space_record",
    "transport",
]
