# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later


"""The fill and the mode a design evaluates.

A fill as the run interfaces accept it (:data:`Fill`), resolved to an optmat mixture;
its Kerr coefficient; the default cladding; and the zero-dispersion point of a mode
nearest a wavelength. Every design search evaluates its probes inside ``_probing``.
"""

import math
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import numpy as np

from solprop.constants import ROOM_TEMPERATURE, C
from solprop.exceptions import (
    DiluteScalingWarning,
    MaterialError,
    MaterialRangeWarning,
    NoZeroDispersionError,
    ToolsError,
)
from solprop.modes import base
from solprop.modes.protocol import Mode
from solprop.physdata import (
    Gas,
    Mixture,
    is_air,
    n2_from_chi3,
    species,
)
from solprop.physdata.glasses import GlassIndex, fused_silica
from solprop.physdata.nonlinear_data import carriers_agree
from solprop.physdata.selection import MaterialSelection
from solprop.profiles import ZProfile
from solprop.tools.pulse import _omega

__all__ = ["Fill"]

Fill = Gas | str | Mapping[str, float] | Mixture
"""A fill as the run interfaces accept it: a species, mole fractions or a mixture."""


_ZDW_RANGE = (100e-9, 3e-6)
"""Window of the zero-dispersion search in m: :func:`modes.base.zdw`'s."""


@contextmanager
def _probing() -> Iterator[None]:
    """Silence material-range and dilute-scaling warnings while a search samples.

    A root search evaluates the index across a window far wider than its answer --
    in frequency, or in pressure up to a hundred bar, where an optmat fill's linear
    density scaling departs from its finite-density law -- and the ends are probes,
    not results. The answer itself is re-evaluated outside this context, so a root
    that lies outside a model's range, or at a density beyond the dilute law,
    still warns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        warnings.simplefilter("ignore", DiluteScalingWarning)
        yield


def _resolve_fill(
    gas: Fill,
    pressure: float | None,
    temperature: float | None,
    materials: MaterialSelection | None = None,
) -> Mixture:
    """Return the mixture a fill declaration means, as the run interfaces read it.

    - ``pressure``: total pressure in bar; must be ``None`` beside a `Mixture`, which
      carries its own, and defaults to one bar otherwise.
    - ``temperature``: fill temperature in K; ``None`` is room temperature, or the
      mixture's own beside a `Mixture`, which refuses any other.
    - ``materials``: resolve a name, a fraction mapping's names and ``"air"`` from
      optmat under this selection, as the drivers' ``capillary_fill`` does; ``None`` is
      ``MaterialSelection()``. A mixture is taken as it is.
    """
    if isinstance(gas, Mixture):
        if pressure is not None:
            raise ToolsError("a Mixture carries its own pressure; omit pressure")
        if isinstance(gas.temperature, ZProfile):
            raise ToolsError(
                "the design calculators take one temperature in K, and this mixture's "
                "varies along z; give a Mixture at one temperature"
            )
        if temperature is not None and float(temperature) != gas.temperature:
            # As the run interfaces: the mixture's densities were set at its own
            # temperature, so another one cannot be honoured.
            raise ToolsError(
                f"the Mixture carries its own temperature ({gas.temperature:g} K), "
                f"and temperature={temperature!r} was given beside it: omit "
                "temperature=, or build the Mixture at the temperature you mean"
            )
        return gas
    temperature = ROOM_TEMPERATURE if temperature is None else temperature
    total = 1.0 if pressure is None else float(pressure)
    if isinstance(gas, Mapping):
        fractions = {
            species(name, selection=materials): fraction
            for name, fraction in gas.items()
        }
        return Mixture.fill(fractions, total, temperature=temperature)
    if is_air(gas):
        return Mixture.air(
            total, temperature=temperature, argon=False, selection=materials
        )
    return Mixture.fill(
        {species(gas, selection=materials): 1.0}, total, temperature=temperature
    )


def _check_carrier(materials: MaterialSelection | None, wavelength: float) -> None:
    """Refuse a selection whose carrier is not the tool's pump wavelength.

    A run under a selection refuses a carrier other than its grid's reference
    wavelength, so a design tool refuses one other than the wavelength it evaluates
    the Kerr coefficient at: the two then evaluate it at the same carrier.

    - ``wavelength``: the pump wavelength in m.
    """
    carrier = None if materials is None else materials.carrier_wavelength
    if carrier is not None and not carriers_agree(float(carrier), wavelength):
        msg = (
            f"the selection's carrier_wavelength is {carrier:g} m and the design's "
            f"wavelength {wavelength:g} m: one carrier sets the Kerr coefficient of "
            "every optmat species, as in a run. Leave the selection's "
            "carrier_wavelength unset, or design at it"
        )
        raise ToolsError(msg)


def _default_cladding(
    cladding: GlassIndex | complex | None,
) -> GlassIndex | complex:
    """Return the cladding a design's capillary takes, as the capillary driver does.

    - ``cladding``: the cladding given; ``None`` is the package default.
    """
    return fused_silica() if cladding is None else cladding


def _fill_chi3(mixture: Mixture, wavelength: float, z: float = 0.0) -> float:
    """Return a fill's density-weighted third-order susceptibility.

    - ``wavelength``: vacuum wavelength in m, the carrier at which an optmat Kerr model
      is taken.
    - ``z``: position in m at which the densities are evaluated.
    - Returns: ``chi3`` in m^2/V^2.
    """
    try:
        return float(mixture.gamma3_path(wavelength)(np.array([float(z)]))[0])
    except MaterialError as error:
        raise ToolsError(f"the fill has no nonlinear index: {error}") from error


def _fill_n2(
    mixture: Mixture, wavelength: float, z: float = 0.0
) -> tuple[float, float]:
    """Return a fill's nonlinear index and its linear index at a wavelength.

    - ``wavelength``: vacuum wavelength in m, the carrier at which an optmat Kerr model
      is taken.
    - ``z``: position in m at which the densities are evaluated.
    - Returns: ``n2`` in m^2/W, from the density-weighted ``chi3`` and the exact
      ``n0^2`` of the fill, and that linear index ``n0``.
    """
    chi3 = _fill_chi3(mixture, wavelength, z)
    here = np.array([float(z)])
    susceptibility = mixture.susceptibility(np.array([_omega(wavelength)]), here)[0, 0]
    index = float(np.real(np.sqrt(1.0 + susceptibility)))
    return n2_from_chi3(chi3, index), index


def _zdw_near(mode: Mode, omega: float, z: float) -> float:
    """Return the zero-dispersion wavelength nearest a pump frequency, or NaN.

    :func:`solprop.modes.base.zdw` over :data:`_ZDW_RANGE` with the pump as its
    reference: the nearest zero that no resonance separates from the pump.

    - ``omega``: pump angular frequency in rad/s.
    - ``z``: position along the waveguide, m.
    - Returns: wavelength in m, or NaN when the search finds no zero.
    """
    try:
        return base.zdw(
            mode, z, wavelength_limits=_ZDW_RANGE, near=2.0 * math.pi * C / omega
        )
    except NoZeroDispersionError:
        return math.nan
