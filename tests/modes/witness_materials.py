# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The material inputs of the mode witnesses, as explicit test-local data.

solprop has no gas-index registry of its own since SEP 3's Phase 4b: a named gas is
optmat's. The stored Luna runs (``tests/data/luna_*``), the symbolic derivations
(``tests/data/symbolic_*``) and the FEM records were made with the published fits below,
so a test that compares against them supplies the same fit explicitly through the
``susceptibility=`` of the mode, as a comparison with another code is meant to (see
:class:`~solprop.modes.MarcatiliMode`). The equation of state is the gas's own,
optmat's, both for the fill density and for the reference density that turns each fit
into a per-particle susceptibility.

The cladding side is the same: :func:`malitson_silica` is the real part of solprop's
former default silica, which the silica-clad witnesses measured their documented
differences against.

Every coefficient here is solprop's former registry at dc4ca0d
(``physdata/sellmeier.py``, ``_MODELS``; ``physdata/glasses.py``), which was the input
of these witnesses.
"""

from solprop.physdata import Gas
from solprop.physdata.glasses import CompositeGlass, SellmeierGlass, fused_silica
from solprop.physdata.sellmeier import PoleSumSusceptibility, ReferenceState

__all__ = [
    "absorbing_silica",
    "borzsonyi",
    "ermolov_helium",
    "kingston_argon",
    "malitson_silica",
]

ONE_BAR_0C = ReferenceState(pressure_bar=1.0, temperature=273.15)
"""Börzsönyi's state, 1000 mbar and 0 degrees Celsius; also the Kingston argon fit's."""

ONE_ATM_0C = ReferenceState(pressure_bar=1.01325, temperature=273.15)
"""1 atm and 0 degrees Celsius, the state of the Ermolov helium expansion."""

BORZSONYI_CITATION = (
    "A. Börzsönyi et al., Appl. Opt. 47, 4856 (2008), doi:10.1364/AO.47.004856; "
    "measured 0.4-1.0 um at 1 bar and 0 degrees Celsius"
)

_BORZSONYI: dict[Gas, tuple[float, float, float, float]] = {
    # (B1, C1 in um^2, B2, C2 in um^2), Table 1 of the paper.
    Gas.He: (4977.77e-8, 28.54e-6, 1856.94e-8, 7.76e-3),
    Gas.Ne: (9154.48e-8, 656.97e-6, 4018.63e-8, 5.728e-3),
    Gas.Ar: (20332.29e-8, 206.12e-6, 34458.31e-8, 8.066e-3),
    Gas.Kr: (26102.88e-8, 2.01e-6, 56946.82e-8, 10.043e-3),
    Gas.Xe: (103701.61e-8, 12750e-6, 31228.61e-8, 0.561e-3),
    Gas.N2: (39209.95e-8, 1146.24e-6, 18806.48e-8, 13.476e-3),
}


def borzsonyi(gas: Gas | str) -> PoleSumSusceptibility:
    """Return Börzsönyi's two-pole fit for `gas`, the index of the Luna witnesses.

    Parameters
    ----------
    gas : Gas or str
        One of He, Ne, Ar, Kr, Xe or N2.

    Returns
    -------
    PoleSumSusceptibility
        The fit at 1 bar and 0 degrees Celsius, valid over 0.4-1.0 um.
    """
    species = Gas(gas)
    b1, c1, b2, c2 = _BORZSONYI[species]
    return PoleSumSusceptibility(
        gas=species,
        source="borzsonyi",
        reference_state=ONE_BAR_0C,
        wavelength_range=(400e-9, 1000e-9),
        citation=BORZSONYI_CITATION,
        b=(b1, b2),
        c=(c1, c2),
    )


def ermolov_helium() -> PoleSumSusceptibility:
    """Return the Ermolov three-pole helium expansion, solprop's former helium default.

    Returns
    -------
    PoleSumSusceptibility
        The expansion normalised by the 1 atm, 0 degrees Celsius density.
    """
    return PoleSumSusceptibility(
        gas=Gas.He,
        source="ermolov",
        reference_state=ONE_ATM_0C,
        wavelength_range=(90e-9, 1000e-9),
        citation=(
            "A. Ermolov et al., Phys. Rev. A 92, 033821 (2015), "
            "doi:10.1103/PhysRevA.92.033821"
        ),
        b=(2.16463842e-05, 2.10561127e-07, 4.75092720e-05),
        c=(-6.80769781e-04, 5.13251289e-03, 3.18621354e-03),
    )


def kingston_argon() -> PoleSumSusceptibility:
    """Return the Kingston three-pole argon fit, solprop's former argon default.

    Returns
    -------
    PoleSumSusceptibility
        The fit normalised by the 1 bar, 0 degrees Celsius density, as Luna uses it.
    """
    return PoleSumSusceptibility(
        gas=Gas.Ar,
        source="kingston",
        reference_state=ONE_BAR_0C,
        wavelength_range=(110e-9, 1000e-9),
        citation=(
            "three-pole fit to A. E. Kingston, J. Opt. Soc. Am. 54, 1145 (1964), "
            "doi:10.1364/JOSA.54.001145, joined to Börzsönyi et al. (2008); "
            "coefficients from Luna (commit 56cc3ab)"
        ),
        b=(0.00032323117217767093, 0.00011557814904827939, 0.00010909808164540697),
        c=(0.0045416501944977915, 0.011120847461156543, 0.0006827046691889898),
    )


def malitson_silica() -> SellmeierGlass:
    """Return Malitson's fused-silica Sellmeier law, relative to air as published.

    It is the real part of solprop's former default silica (``"malitson+franta"``,
    whose Franta extinction is negligible over every band these witnesses sample), and
    it differs from optmat's absolute ``fit/SiO2`` by about 2.7e-4 in the index -- which
    a capillary barely notices and an antiresonant wall, sitting near resonance, does.

    Returns
    -------
    SellmeierGlass
        The three-pole expansion stated for 0.21-3.71 um at 20 degrees Celsius.
    """
    return SellmeierGlass(
        name="SiO2",
        source="malitson",
        wavelength_range=(0.21e-6, 3.71e-6),
        citation=(
            "I. H. Malitson, J. Opt. Soc. Am. 55, 1205-1208 (1965), "
            "doi:10.1364/JOSA.55.001205"
        ),
        poles=((0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161)),
    )


def absorbing_silica() -> CompositeGlass:
    """Return a complex silica outside optmat: Malitson's real part, optmat's ``k``.

    The stand-in for solprop's former complex default silica (Malitson's real part with
    Franta's extinction), for the refusals of the modal routes that need a lossless
    index: an optmat index is held transparent there (SEP 3, Q3), so the absorbing
    material has to be the caller's.

    Returns
    -------
    CompositeGlass
        ``Re n`` from :func:`malitson_silica`, ``Im n`` from optmat's ``fit/SiO2``.
    """
    real_part = malitson_silica()
    return CompositeGlass(
        name="SiO2",
        source="malitson+optmat",
        wavelength_range=real_part.wavelength_range,
        citation=f"{real_part.citation}; extinction from optmat's fit/SiO2",
        real_part=real_part,
        absorption=fused_silica(),
    )
