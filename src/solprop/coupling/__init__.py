# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Couple simulations through a coherent field at a reference plane (SEP 31).

- :mod:`~solprop.coupling.plane`: :class:`~solprop.coupling.plane.FieldPlane`, one
  physical plane of a field in its native representation, with its references, geometry,
  medium, metric and provenance.
- :mod:`~solprop.coupling.remap`: :func:`~solprop.coupling.remap.regrid` moves a plane
  onto another temporal lattice or reference.
- :mod:`~solprop.coupling.port`: :class:`~solprop.coupling.port.InputPort`, what a
  receiving simulation accepts at its entrance, and the
  :class:`~solprop.coupling.port.CouplingLaunch` bound to it.
- :mod:`~solprop.coupling.coupler` and :mod:`~solprop.coupling.models`:
  :func:`~solprop.coupling.coupler.couple` launches a plane into a port under a declared
  :class:`~solprop.coupling.models.CouplingModel`, admitted against the recorded pair
  envelopes of :mod:`~solprop.coupling.envelopes`.
- :mod:`~solprop.coupling.outcouple`: :class:`~solprop.coupling.outcouple.Outcouple`
  synthesises a guide's exit field on a spatial grid.
- :mod:`~solprop.coupling.elements` and the ``transport`` module (whose function shadows
  it): :func:`solprop.coupling.transport` carries a spatial plane through an
  ordered sequence of linear elements, with its
  :class:`~solprop.coupling.transport.TransportReport`.

A physics-layer package: it imports no translation, driver or backend module. A
spatial plane's power metric is a :class:`solprop.spatial.SpatialFlux` (SEP 34), whose
resolution and admittance call translation code through lazy imports in
:mod:`solprop.spatial`, so a coupling that measures a plane runs it at runtime.
Extraction from a saved run is :func:`solprop.processing.planes.field_plane`;
installing a launch is :meth:`solprop.driver.simulation.Simulation.with_input`. The
manual page is ``docs/guide/workflows/coupling.md``.
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
