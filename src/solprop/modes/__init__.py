# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Waveguide modes: the protocol, the mode families, and the machinery behind them.

This namespace re-exports the names a run is built from; the rest is imported from
its submodule.

- Protocol and derived quantities: :mod:`~solprop.modes.protocol`
  (:class:`~solprop.modes.protocol.Mode`), :mod:`~solprop.modes.base` (``beta1``,
  ``beta2``, ``zdw``, ``aeff``, ...).
- Mode families: :mod:`~solprop.modes.capillary` (Marcatili, sharing
  :mod:`~solprop.modes.fill`), :mod:`~solprop.modes.antiresonant`,
  :mod:`~solprop.modes.rectangular`, :mod:`~solprop.modes.slab` (planar),
  :mod:`~solprop.modes.stepindex`, :mod:`~solprop.modes.tabulated`.
- Mode sets and quadrature: :mod:`~solprop.modes.collection`,
  :mod:`~solprop.modes.quadrature`; the capillary's coherent wall,
  :mod:`~solprop.modes.wall`.
- General modal guides: :mod:`~solprop.modes.vector` (the provider records),
  :mod:`~solprop.modes.material` (region laws and responses), the providers
  :mod:`~solprop.modes.stepindex_vector` and :mod:`~solprop.modes.graded_index`, their
  sampling in :mod:`~solprop.modes.tables` and, along a moving geometry,
  :mod:`~solprop.modes.path`, the moving-basis :mod:`~solprop.modes.connection`, and the
  guide itself, :mod:`~solprop.modes.guide`.
"""

from solprop.modes.antiresonant import (
    TubeLattice,
    VincettiMode,
    core_radius_from_gap,
    tube_gap,
    tube_radius_from_gap,
)
from solprop.modes.base import (
    aeff,
    alpha,
    beta,
    beta1,
    beta2,
    beta3,
    db_per_m,
    loss_length,
    transmission,
    zdw,
)
from solprop.modes.capillary import (
    EVANESCENT_INDEX_FLOOR,
    MarcatiliMode,
    MarcatiliModel,
    ModeKind,
)
from solprop.modes.collection import (
    ModeCollection,
    orthonormality,
    overlap,
)
from solprop.modes.graded_index import (
    GradedIndexProvider,
    PowerLawPermittivity,
    RadialPermittivity,
)
from solprop.modes.guide import ModalGuide, ModalSampling, ProviderChannelMode
from solprop.modes.material import (
    GasKerr,
    IndexMaterial,
    MaterialKerr,
    MaterialRegion,
    PermittivityLaw,
)
from solprop.modes.protocol import (
    ChannelLabel,
    CrossSection,
    Mode,
)
from solprop.modes.quadrature import (
    TransverseQuadrature,
    transverse_quadrature,
)
from solprop.modes.rectangular import RectMode
from solprop.modes.slab import (
    PlanarGrid,
    SlabMode,
    SlabPolarisation,
    gaussian_coupling,
    gaussian_projection,
    gaussian_to_per_unit_width,
    per_unit_width_to_gaussian,
)
from solprop.modes.stepindex import StepIndexMode
from solprop.modes.stepindex_vector import StepIndexProvider
from solprop.modes.tabulated import (
    IndexTable,
    TabulatedHollowMode,
    TabulatedMode,
    sample_index_table,
)
from solprop.modes.vector import (
    ModalProvider,
    ModalSample,
)
from solprop.modes.wall import CoherentWall, vector_wall_operator, wall_operator

__all__ = [
    "EVANESCENT_INDEX_FLOOR",
    "ChannelLabel",
    "CoherentWall",
    "CrossSection",
    "GasKerr",
    "GradedIndexProvider",
    "IndexMaterial",
    "IndexTable",
    "MarcatiliMode",
    "MarcatiliModel",
    "MaterialKerr",
    "MaterialRegion",
    "ModalGuide",
    "ModalProvider",
    "ModalSample",
    "ModalSampling",
    "Mode",
    "ModeCollection",
    "ModeKind",
    "PermittivityLaw",
    "PlanarGrid",
    "PowerLawPermittivity",
    "ProviderChannelMode",
    "RadialPermittivity",
    "RectMode",
    "SlabMode",
    "SlabPolarisation",
    "StepIndexMode",
    "StepIndexProvider",
    "TabulatedHollowMode",
    "TabulatedMode",
    "TransverseQuadrature",
    "TubeLattice",
    "VincettiMode",
    "aeff",
    "alpha",
    "beta",
    "beta1",
    "beta2",
    "beta3",
    "core_radius_from_gap",
    "db_per_m",
    "gaussian_coupling",
    "gaussian_projection",
    "gaussian_to_per_unit_width",
    "loss_length",
    "orthonormality",
    "overlap",
    "per_unit_width_to_gaussian",
    "sample_index_table",
    "transmission",
    "transverse_quadrature",
    "tube_gap",
    "tube_radius_from_gap",
    "vector_wall_operator",
    "wall_operator",
    "zdw",
]
