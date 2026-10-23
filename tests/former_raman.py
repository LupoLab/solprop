# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""solprop's former Raman constants, as a caller's supplied data (SEP 3, Phase 4b).

Until SEP 3 Phase 4b solprop carried its own molecular Raman constants and its own
fused-silica line sets (``physdata/raman_data.py``). Every built-in material is now
optmat's; these are the same numbers, transcribed from solprop's former registry at
dc4ca0d and checked equal to it exactly (the same literals), held here as **test
inputs**: the tests that use them exercise the response *numerics* (the ladder, the
oscillator, the convolution, the tensor contraction), whose expected values were
recorded on these inputs. They are not a material database and nothing in the package
reads them; the citations are the former registry's, kept so that each number's origin
stays stated.

The supplied molecular route takes them through a caller's :class:`Species`
(:func:`supplied_species`), never by a gas's name.
"""

import math

from solprop.constants import AMAGAT, ATOMIC_MASS, BAR, K_B, STD_ATMOSPHERE, C
from solprop.nonlinear.raman import MolecularOptions, RamanResponse, response_for
from solprop.physdata import (
    Gas,
    IntermediateBroadeningRaman,
    LinAgrawalRaman,
    Linewidth,
    MolecularRaman,
    PoleSumSusceptibility,
    ReferenceState,
    RotationalRaman,
    Species,
    VibrationalRaman,
    linewidth_from_dephasing_time,
)

__all__ = [
    "FORMER_CH4",
    "FORMER_D2",
    "FORMER_H2",
    "FORMER_N2",
    "FORMER_O2",
    "FORMER_SF6",
    "FORMER_SILICA_LINES",
    "FORMER_SILICA_TENSOR",
    "former_response",
    "supplied_species",
]


def _wavenumber_to_omega(per_cm: float) -> float:
    """Convert a wavenumber in cm^-1 to rad/s, in the former module's arithmetic."""
    return 2.0 * math.pi * C * per_cm * 100.0


_NITROGEN_MASS = 14.0067 * ATOMIC_MASS
_HYDROGEN_MASS = 1.00784 * ATOMIC_MASS
_DEUTERIUM_MASS = 2.014 * ATOMIC_MASS
_FLUORINE_MASS = 18.998403 * ATOMIC_MASS

FORMER_O2 = MolecularRaman(
    gas=Gas.O2,
    rotational=RotationalRaman(
        b=143.7674521,
        d=4.839482e-4,
        degeneracy_odd=1,
        degeneracy_even=0,
        polarisability_anisotropy=1.02e-30,
        linewidth=Linewidth(
            density=87.6e-3 * 100.0 * C * AMAGAT * K_B * 295.0 / STD_ATMOSPHERE
        ),
        citation=(
            "solprop's former registry at dc4ca0d: Hajigeorgiou, J. Chem. Phys. 138, "
            "014309 (2013); Wahlstrand et al., Phys. Rev. A 85, 043820 (2012); "
            "Holzer et al., Appl. Opt. 60, C76 (2021), N=7, 295 K"
        ),
    ),
    vibrational=VibrationalRaman(
        frequency=_wavenumber_to_omega(1556.3906),
        polarisability_derivative=1.46e-20,
        reduced_mass=7.99745730978 * ATOMIC_MASS,
        linewidth=Linewidth(
            density=2.0 * 45.3e-3 * 100.0 * C * AMAGAT * K_B * 295.0 / STD_ATMOSPHERE
        ),
        citation=(
            "solprop's former registry at dc4ca0d: Hajigeorgiou (2013); Murphy et "
            "al., Appl. Spectrosc. 23, 211 (1969); Millot et al., J. Chem. Phys. 96, "
            "961 (1992), Q(7), 295 K"
        ),
    ),
)
"""Oxygen: spin-free rotor (odd N only) and one vibrational band."""

FORMER_N2 = MolecularRaman(
    gas=Gas.N2,
    rotational=RotationalRaman(
        b=199.0,
        d=5.74e-4,
        degeneracy_odd=1,
        degeneracy_even=2,
        polarisability_anisotropy=6.7e-31,
        linewidth=Linewidth(density=3.3e9),
        citation=(
            "solprop's former registry at dc4ca0d: J. Raman Spectrosc. 2, 133 "
            "(1974); Wahlstrand et al. (2012); Phys. Rev. A 34, 1944 (1986)"
        ),
    ),
    vibrational=VibrationalRaman(
        frequency=_wavenumber_to_omega(2330.0),
        polarisability_derivative=1.75e-20,
        reduced_mass=_NITROGEN_MASS / 2.0,
        linewidth=linewidth_from_dephasing_time(6e-12),
        citation=(
            "solprop's former registry at dc4ca0d: J. Raman Spectrosc. 2, 133 "
            "(1974); Murphy et al. (1969); an inherited 6 ps band dephasing time"
        ),
    ),
)
"""Nitrogen: rotational ladder and one vibrational band."""

FORMER_H2 = MolecularRaman(
    gas=Gas.H2,
    rotational=RotationalRaman(
        b=5890.0,
        d=5.0,
        degeneracy_odd=3,
        degeneracy_even=1,
        polarisability_anisotropy=3e-31,
        linewidth=Linewidth(inverse_density=6.15e6, density=114e6),
        citation=(
            "solprop's former registry at dc4ca0d: Phys. Rev. A 92, 063828 (2015); "
            "Phys. Rev. A 34, 1944 (1986), S(1), 295 K"
        ),
    ),
    vibrational=VibrationalRaman(
        frequency=2.0 * math.pi * 124.5669e12,
        polarisability_derivative=1.3e-20,
        reduced_mass=_HYDROGEN_MASS / 2.0,
        linewidth=Linewidth(inverse_density=309e6, density=52.2e6),
        citation=(
            "solprop's former registry at dc4ca0d: Phys. Rev. A 92, 063828 (2015); "
            "Phys. Rev. A 33, 3113 (1986), Q(1), 298 K"
        ),
    ),
)
"""Hydrogen: rotational ladder (with Dicke narrowing) and one vibrational band."""

FORMER_D2 = MolecularRaman(
    gas=Gas.D2,
    rotational=RotationalRaman(
        b=2930.0,
        d=2.1,
        degeneracy_odd=1,
        degeneracy_even=2,
        polarisability_anisotropy=3e-31,
        linewidth=Linewidth(density=4.05e-3 * 100.0 * C * AMAGAT * K_B * 293.0 / BAR),
        citation=(
            "solprop's former registry at dc4ca0d: Phys. Rev. A 92, 063828 (2015); "
            "J. Raman Spectrosc. 22, 607 (1991), S(0), 293 K"
        ),
    ),
    vibrational=VibrationalRaman(
        frequency=_wavenumber_to_omega(2987.0),
        polarisability_derivative=1.4e-20,
        reduced_mass=_DEUTERIUM_MASS / 2.0,
        linewidth=Linewidth(inverse_density=101e6, density=120e6),
        citation=(
            "solprop's former registry at dc4ca0d: Phys. Rev. A 92, 063828 (2015); "
            "IEEE J. Quantum Electron. 24, 2076 (1988), Q(2)"
        ),
    ),
)
"""Deuterium: rotational ladder and one vibrational band."""

FORMER_CH4 = MolecularRaman(
    gas=Gas.CH4,
    rotational=None,
    vibrational=VibrationalRaman(
        frequency=_wavenumber_to_omega(2914.0),
        polarisability_derivative=1.04e-20,
        reduced_mass=_HYDROGEN_MASS / 4.0,
        linewidth=Linewidth(constant=8220e6, density=384e6),
        citation=(
            "solprop's former registry at dc4ca0d: Appl. Spectrosc. 23, 211 (1969), "
            "Eq. (5); IEEE J. Quantum Electron. 24, 2076 (1988)"
        ),
    ),
)
"""Methane: a spherical top, one vibrational band and no rotational spectrum."""

FORMER_SF6 = MolecularRaman(
    gas=Gas.SF6,
    rotational=None,
    vibrational=VibrationalRaman(
        frequency=_wavenumber_to_omega(775.0),
        polarisability_derivative=1.23e-20,
        reduced_mass=_FLUORINE_MASS / 6.0,
        linewidth=linewidth_from_dephasing_time(6.6e-12),
        citation=(
            "solprop's former registry at dc4ca0d: Appl. Spectrosc. 23, 211 (1969); "
            "Opt. Commun. 64, 393 (1987)"
        ),
    ),
)
"""Sulphur hexafluoride: a spherical top, one vibrational band."""

_SILICA_CENTRES_PER_CM = (
    56.25,
    100.0,
    231.25,
    362.50,
    463.00,
    497.00,
    611.50,
    691.67,
    793.67,
    835.50,
    930.0,
    1080.00,
    1215.00,
)
_SILICA_AMPLITUDES = (
    1.0,
    11.40,
    36.67,
    67.67,
    74.00,
    4.50,
    6.80,
    4.60,
    4.20,
    4.50,
    2.70,
    3.10,
    3.00,
)
_SILICA_GAUSSIAN_PER_CM = (
    52.10,
    110.42,
    175.00,
    162.50,
    135.33,
    24.50,
    41.50,
    155.0,
    59.50,
    64.30,
    150.00,
    91.00,
    160.00,
)
_SILICA_LORENTZIAN_PER_CM = (
    17.37,
    38.81,
    58.33,
    54.17,
    45.11,
    8.17,
    13.83,
    51.67,
    19.83,
    21.43,
    50.00,
    30.33,
    53.33,
)

FORMER_SILICA_LINES = IntermediateBroadeningRaman(
    frequencies=tuple(_wavenumber_to_omega(v) for v in _SILICA_CENTRES_PER_CM),
    amplitudes=_SILICA_AMPLITUDES,
    # The former registry halved both tabulated widths (read as full widths).
    gaussian_widths=tuple(
        0.5 * _wavenumber_to_omega(v) for v in _SILICA_GAUSSIAN_PER_CM
    ),
    lorentzian_widths=tuple(
        0.5 * _wavenumber_to_omega(v) for v in _SILICA_LORENTZIAN_PER_CM
    ),
    citation=(
        "solprop's former registry at dc4ca0d: D. Hollenbeck and C. D. Cantrell, "
        "J. Opt. Soc. Am. B 19, 2886 (2002), table 1, as Luna carries it"
    ),
)
"""The thirteen-line intermediate-broadening fused-silica set Luna also uses."""

FORMER_SILICA_TENSOR = LinAgrawalRaman(
    tau1=12.2e-15,
    tau2=32.0e-15,
    tau_b=96.0e-15,
    isotropic_weight=0.75,
    boson_weight=0.21,
    anisotropic_weight=0.04,
    fraction=0.245,
    citation=(
        "solprop's former registry at dc4ca0d: Q. Lin and G. P. Agrawal, Opt. Lett. "
        "31, 3086 (2006), doi:10.1364/OL.31.003086"
    ),
)
"""The Lin--Agrawal 2006 fused-silica parameters, as a caller would supply them."""

# A placeholder linear law: the species exists to carry its Raman constants to the
# supplied molecular route, and no test here reads its index.
_PLACEHOLDER_LAW_STATE = ReferenceState(pressure_bar=1.0, temperature=273.15)


def supplied_species(raman: MolecularRaman) -> Species:
    """Return a caller's species carrying `raman`, never resolved as a built-in gas.

    Parameters
    ----------
    raman : MolecularRaman
        The molecular constants.

    Returns
    -------
    Species
        Named after the gas, with a placeholder supplied linear law and no optmat
        model, so :func:`response_for` takes the supplied molecular route.
    """
    gas = Gas(raman.gas) if isinstance(raman.gas, str) else raman.gas
    law = PoleSumSusceptibility(
        gas=gas,
        source="supplied",
        reference_state=_PLACEHOLDER_LAW_STATE,
        wavelength_range=(200e-9, 5e-6),
        citation="placeholder: these tests read only the species' Raman constants",
        b=(5.8e-4,),
        c=(7.4e-3,),
    )
    return Species(
        gas.value,
        law,
        raman=raman,
        citation="supplied in the tests from solprop's former registry at dc4ca0d",
    )


def former_response(
    raman: MolecularRaman,
    options: MolecularOptions | None = None,
    *,
    thg: bool = True,
) -> RamanResponse:
    """Return the supplied molecular response built from `raman`.

    Parameters
    ----------
    raman : MolecularRaman
        The molecular constants.
    options : MolecularOptions or None, optional
        Which parts to build, and at what temperature.
    thg : bool, optional
        Whether a field grid keeps the third harmonic. Default ``True``.

    Returns
    -------
    RamanResponse
        One term per part of the spectrum: the rotational ladder (``rank2``), then
        the vibrational band (``isotropic``).
    """
    return response_for(supplied_species(raman), options, thg=thg)
