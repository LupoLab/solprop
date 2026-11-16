# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Waveguide modes.

The mode protocol and the modes a run is built from -- capillary, antiresonant,
rectangular, planar, step-index, graded-index and tabulated -- with mode collections,
the general modal guide, and the derived quantities of any mode
(:mod:`solprop.modes.base`: ``beta2``, ``zdw``, ``aeff``, ``db_per_m``, ...). The
numerical machinery behind them (wall operators, connections, quadrature rules) is
imported from its own submodule.
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
    CoherentWall,
    MarcatiliMode,
    MarcatiliModel,
    ModeKind,
    vector_wall_operator,
    wall_operator,
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
from solprop.modes.guide import (
    ModalGuide,
    ModalSampling,
    ProviderChannelMode,
    StepIndexProvider,
)
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
