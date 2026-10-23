# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Published gas index laws, supplied as a caller's own data, for the physdata tests.

solprop no longer carries gas index laws of its own (SEP 3, Phase 4b): every built-in
gas law is optmat's. These are the published fits solprop's former registry carried
(at dc4ca0d), written out as the supplied types a caller uses --
:class:`~solprop.physdata.PoleSumSusceptibility`,
:class:`~solprop.physdata.InversePoleIndex` and
:class:`~solprop.physdata.WavenumberPoleIndex` -- so the tests can exercise those
types and keep Luna's stored evaluations of the same expressions as witnesses
(tests/physdata/test_luna_reference.py). Each coefficient is the source's own, with
the reference state the source reports; their reference densities come from optmat's
equation of state.

The laws are plain named constants: nothing here looks a law up by name.
"""

from solprop.constants import BAR, STD_ATMOSPHERE
from solprop.physdata import (
    Gas,
    GasSusceptibility,
    InversePoleIndex,
    PoleSumSusceptibility,
    ReferenceState,
    WavenumberPoleIndex,
)

ATM_IN_BAR = STD_ATMOSPHERE / BAR

BAR_0C = ReferenceState(pressure_bar=1.0, temperature=273.15)
"""1 bar, 0 degrees Celsius: Börzsönyi et al. 2008."""

ATM_0C = ReferenceState(pressure_bar=ATM_IN_BAR, temperature=273.15)
"""1 atm, 0 degrees Celsius: Peck and Hung, Bideau-Mehu, Ermolov, the krypton fit."""

ATM_15C = ReferenceState(pressure_bar=ATM_IN_BAR, temperature=288.15)
"""1 atm, 15 degrees Celsius: He et al. 2021, section 3.3."""

ATM_20C = ReferenceState(pressure_bar=ATM_IN_BAR, temperature=293.15)
"""1 atm, 20 degrees Celsius: Zhang et al. 2008 as corrected by Kren 2011."""

BORZSONYI_CITATION = (
    "A. Börzsönyi et al., Appl. Opt. 47, 4856 (2008), doi:10.1364/AO.47.004856; "
    "measured 0.4-1.0 um at 1 bar and 0 degrees Celsius"
)
KINGSTON_CITATION = (
    "three-pole fit to the vacuum-ultraviolet data of A. E. Kingston, "
    "J. Opt. Soc. Am. 54, 1145 (1964), doi:10.1364/JOSA.54.001145, joined to the "
    "visible and near-infrared data of Börzsönyi et al. (2008); coefficients from "
    "Luna (commit 56cc3ab)"
)
PECK_CITATION = (
    "E. R. Peck and S. Hung, J. Opt. Soc. Am. 67, 1550 (1977), "
    "doi:10.1364/JOSA.67.001550; 0.168-1.6945 um at 1 atm and 0 degrees Celsius"
)
QUANFU_HE_CITATION = (
    "Q. He et al., Atmos. Chem. Phys. 21, 14927 (2021), "
    "doi:10.5194/acp-21-14927-2021; at 1 atm and 15 degrees Celsius"
)


def borzsonyi(
    gas: Gas, b1: float, c1: float, b2: float, c2: float
) -> PoleSumSusceptibility:
    """Return a Börzsönyi two-pole law, valid over 0.4-1.0 um.

    Parameters
    ----------
    gas : Gas
        The species.
    b1, c1, b2, c2 : float
        The published pole strengths (dimensionless) and positions (um^2).

    Returns
    -------
    PoleSumSusceptibility
        The law, at Börzsönyi's reference state.
    """
    return PoleSumSusceptibility(
        gas=gas,
        source="borzsonyi",
        reference_state=BAR_0C,
        wavelength_range=(400e-9, 1000e-9),
        citation=BORZSONYI_CITATION,
        b=(b1, b2),
        c=(c1, c2),
    )


AIR_BORZSONYI = borzsonyi(Gas.Air, 14926.44e-8, 19.36e-6, 41807.57e-8, 7.434e-3)
HE_BORZSONYI = borzsonyi(Gas.He, 4977.77e-8, 28.54e-6, 1856.94e-8, 7.76e-3)
NE_BORZSONYI = borzsonyi(Gas.Ne, 9154.48e-8, 656.97e-6, 4018.63e-8, 5.728e-3)
AR_BORZSONYI = borzsonyi(Gas.Ar, 20332.29e-8, 206.12e-6, 34458.31e-8, 8.066e-3)
KR_BORZSONYI = borzsonyi(Gas.Kr, 26102.88e-8, 2.01e-6, 56946.82e-8, 10.043e-3)
XE_BORZSONYI = borzsonyi(Gas.Xe, 103701.61e-8, 12750e-6, 31228.61e-8, 0.561e-3)
N2_BORZSONYI = borzsonyi(Gas.N2, 39209.95e-8, 1146.24e-6, 18806.48e-8, 13.476e-3)

HE_ERMOLOV = PoleSumSusceptibility(
    gas=Gas.He,
    source="ermolov",
    # Normalised by the 1 atm density: by the 1 bar density the expansion sits 1.2 to
    # 1.3 % above Mansfield and Peck (1969), by the 1 atm density within 0.13 %.
    reference_state=ATM_0C,
    wavelength_range=(90e-9, 1000e-9),
    citation=(
        "A. Ermolov et al., Phys. Rev. A 92, 033821 (2015), "
        "doi:10.1103/PhysRevA.92.033821; a three-pole expansion fitted to "
        "vacuum-ultraviolet data, bounded at 90 nm and 1000 nm"
    ),
    b=(2.16463842e-05, 2.10561127e-07, 4.75092720e-05),
    c=(-6.80769781e-04, 5.13251289e-03, 3.18621354e-03),
)

AR_KINGSTON = PoleSumSusceptibility(
    gas=Gas.Ar,
    source="kingston",
    reference_state=BAR_0C,
    wavelength_range=(110e-9, 1000e-9),
    citation=KINGSTON_CITATION,
    b=(0.00032323117217767093, 0.00011557814904827939, 0.00010909808164540697),
    c=(0.0045416501944977915, 0.011120847461156543, 0.0006827046691889898),
)

H2_PECK = InversePoleIndex(
    gas=Gas.H2,
    source="peck",
    reference_state=ATM_0C,
    wavelength_range=(168.0e-9, 1694.5e-9),
    citation=PECK_CITATION,
    offset=0.0,
    b=(0.0148956, 0.0049037),
    c=(180.7, 92.0),
)

# Deuterium takes the hydrogen coefficients (Orr, Trans. Faraday Soc. 32, 1556, 1936),
# normalised by deuterium's own density at the reference state.
D2_PECK = InversePoleIndex(
    gas=Gas.D2,
    source="peck",
    reference_state=ATM_0C,
    wavelength_range=(168.0e-9, 1694.5e-9),
    citation=f"the hydrogen coefficients of {PECK_CITATION}",
    offset=0.0,
    b=(0.0148956, 0.0049037),
    c=(180.7, 92.0),
)

O2_ZHANG = InversePoleIndex(
    gas=Gas.O2,
    source="zhang",
    reference_state=ATM_20C,
    wavelength_range=(400e-9, 1800e-9),
    citation=(
        "J. Zhang, Z. H. Lu and L. J. Wang, Appl. Opt. 47, 3143 (2008), "
        "doi:10.1364/AO.47.003143, corrected by P. Kren, Appl. Opt. 50, 6484 (2011), "
        "doi:10.1364/AO.50.006484; 0.4-1.8 um at 1 atm and 20 degrees Celsius"
    ),
    offset=1.181494e-4,
    b=(9.708931e-3,),
    c=(75.4,),
)

QUANFU_HE_COEFFICIENTS = {
    Gas.CH4: (3603.09, 4.40362e14, 1.1741e10, (307e-9, 725e-9)),
    Gas.N2O: (22095.0, 1.66291e14, 6.75226e9, (307e-9, 725e-9)),
    Gas.SF6: (18997.7, 8.27663e14, 1.56833e10, (264e-9, 725e-9)),
}
"""He et al. 2021: ``(n - 1) 1e8 = a + b / (c - nu^2)``, nu in cm^-1, and the range."""


def quanfu_he(gas: Gas) -> WavenumberPoleIndex:
    """Return He et al.'s refractivity law of one of their three gases.

    Parameters
    ----------
    gas : Gas
        CH4, N2O or SF6.

    Returns
    -------
    WavenumberPoleIndex
        The law at 1 atm and 15 degrees Celsius.
    """
    a, b, c, wavelength_range = QUANFU_HE_COEFFICIENTS[gas]
    return WavenumberPoleIndex(
        gas=gas,
        source="quanfu-he",
        reference_state=ATM_15C,
        wavelength_range=wavelength_range,
        citation=QUANFU_HE_CITATION,
        a=a,
        b=b,
        c=c,
    )


ALL_LAWS: tuple[GasSusceptibility, ...] = (
    AIR_BORZSONYI,
    HE_BORZSONYI,
    HE_ERMOLOV,
    NE_BORZSONYI,
    AR_BORZSONYI,
    AR_KINGSTON,
    KR_BORZSONYI,
    XE_BORZSONYI,
    N2_BORZSONYI,
    H2_PECK,
    D2_PECK,
    O2_ZHANG,
    quanfu_he(Gas.CH4),
    quanfu_he(Gas.N2O),
    quanfu_he(Gas.SF6),
)
"""Every law above, one of each type and reference state among them."""
