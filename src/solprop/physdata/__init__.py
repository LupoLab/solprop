# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Material data.

Species, mixtures and solids from optmat (SEP 3), their equations of state and
transport properties, the Kerr coefficient at the carrier, and the types a caller's
own data take (physics layer).
"""

from typing import TYPE_CHECKING

from solprop.physdata.equation_of_state import density, ideal_gas_density, pressure
from solprop.physdata.gases import Gas, as_gas, is_molecule
from solprop.physdata.glasses import (
    CompositeGlass,
    GlassIndex,
    OptmatGlass,
    SellmeierGlass,
    SilicaBundle,
    TabulatedGlass,
    barium_fluoride,
    bk7,
    calcium_fluoride,
    fused_silica,
    glass,
    glass_names,
    magnesium_fluoride,
    sapphire,
)
from solprop.physdata.metals import aluminium, silver
from solprop.physdata.nonlinear_data import (
    DispersiveKerrModel,
    OptmatKerr,
    chi3,
    chi3_from_n2,
    gamma3,
    n2,
    n2_from_chi3,
)
from solprop.physdata.raman_types import (
    IntermediateBroadeningRaman,
    LinAgrawalRaman,
    Linewidth,
    MolecularRaman,
    RotationalRaman,
    VibrationalRaman,
    linewidth_from_dephasing_time,
)
from solprop.physdata.selection import (
    MaterialManifest,
    MaterialRecord,
    MaterialSelection,
)
from solprop.physdata.sellmeier import (
    GasSusceptibility,
    InversePoleIndex,
    OptmatSusceptibility,
    PoleSumSusceptibility,
    ReferenceState,
    RefractivitySusceptibility,
    WavenumberPoleIndex,
    chi1,
    dchi1_domega,
    drefractive_index_domega,
    gamma,
    refractive_index,
    refractive_index_minus_one,
)
from solprop.physdata.species import (
    EquationOfState,
    IdealGas,
    IonisationData,
    OptmatGas,
    Species,
    TabulatedSusceptibility,
    ionisation_potential,
    species,
)
from solprop.physdata.transport import (
    heat_capacity_ratio,
    mean_free_path,
    molar_mass,
    sound_speed,
    viscosity,
)

if TYPE_CHECKING:
    from solprop.physdata.mixtures import (
        Component,
        Mixture,
    )

__all__ = [
    "Component",
    "CompositeGlass",
    "DispersiveKerrModel",
    "EquationOfState",
    "Gas",
    "GasSusceptibility",
    "GlassIndex",
    "IdealGas",
    "IntermediateBroadeningRaman",
    "InversePoleIndex",
    "IonisationData",
    "LinAgrawalRaman",
    "Linewidth",
    "MaterialManifest",
    "MaterialRecord",
    "MaterialSelection",
    "Mixture",
    "MolecularRaman",
    "OptmatGas",
    "OptmatGlass",
    "OptmatKerr",
    "OptmatSusceptibility",
    "PoleSumSusceptibility",
    "ReferenceState",
    "RefractivitySusceptibility",
    "RotationalRaman",
    "SellmeierGlass",
    "SilicaBundle",
    "Species",
    "TabulatedGlass",
    "TabulatedSusceptibility",
    "VibrationalRaman",
    "WavenumberPoleIndex",
    "aluminium",
    "as_gas",
    "barium_fluoride",
    "bk7",
    "calcium_fluoride",
    "chi1",
    "chi3",
    "chi3_from_n2",
    "dchi1_domega",
    "density",
    "drefractive_index_domega",
    "fused_silica",
    "gamma",
    "gamma3",
    "glass",
    "glass_names",
    "heat_capacity_ratio",
    "ideal_gas_density",
    "ionisation_potential",
    "is_molecule",
    "linewidth_from_dephasing_time",
    "magnesium_fluoride",
    "mean_free_path",
    "molar_mass",
    "n2",
    "n2_from_chi3",
    "pressure",
    "refractive_index",
    "refractive_index_minus_one",
    "sapphire",
    "silver",
    "sound_speed",
    "species",
    "viscosity",
]


def __getattr__(name: str) -> object:
    """Load compositions after the scalar-profile module can finish importing.

    Parameters
    ----------
    name : str
        Requested public composition symbol.

    Returns
    -------
    object
        The exported composition class.

    Raises
    ------
    AttributeError
        If the symbol is not exported here.
    """
    if name in ("Component", "Mixture"):
        from solprop.physdata import (  # noqa: PLC0415 - profiles import physdata
            mixtures,
        )

        return getattr(mixtures, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
