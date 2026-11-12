# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later


"""The fill and the mode a design evaluates.

A fill as the run interfaces accept it, resolved to an optmat mixture; its Kerr
coefficient; the default cladding; and the zero-dispersion point of a mode nearest a
wavelength. Every design search evaluates its probes inside ``_probing``.
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
    ToolsError,
)
from solprop.modes import base
from solprop.modes.protocol import Mode
from solprop.physdata import (
    Gas,
    Mixture,
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


_ZDW_SAMPLES = 512
"""Samples, evenly spaced in frequency, on which the dispersion's sign is read."""


_ZDW_REFINE_ROUNDS = 4
"""Rounds of array samples that refine a sign change of the dispersion. Four take a
bracket of 3.6e13 rad/s to 3.4e7 rad/s, where interpolating linearly reaches the noise
of the differenced ``beta2``."""


_ZDW_REFINE_SAMPLES = 33
"""Samples of each refinement round: 32 intervals, so each round narrows 32-fold."""


_POLE_FACTOR = 2.0
"""How far the group delay at a refined sign change of ``beta2`` may sit from its value
at the bracket's ends, in units of the change a bounded ``beta2`` could produce across
the bracket, before the sign change is read as a pole rather than a zero. At a zero
``beta1`` is extremal and moves by at most half that unit; at a gas resonance it
diverges. The differenced ``beta2`` itself cannot tell them apart: ``beta1`` is even
about a pole, so its central difference vanishes there too."""


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
    if gas is Gas.Air or (isinstance(gas, str) and gas.strip().lower() == "air"):
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


def _refine_zero(mode: Mode, low: float, high: float, z: float) -> float:
    """Return a sign change of the dispersion inside a bracket, by array samples.

    A mode's index models evaluate an array at about the cost of one point, so each
    round narrows the bracket 32-fold for the cost of one Brent step, where a scalar
    search costs a glass model evaluation per iterate. The rounds stop short of a pole
    of the differenced ``beta2``, which a scalar search converges onto until the model
    returns NaN.

    - ``low``, ``high``: the bracket, rad/s, whose ends have ``beta2`` of opposite
      signs.
    - ``z``: position along the waveguide, m.
    - Returns: the first sign change upwards in frequency, rad/s, linearly interpolated
      in the last bracket; or NaN when samples that are not finite separate the signs.
    """
    first = last = math.nan
    for _ in range(_ZDW_REFINE_ROUNDS):
        grid = np.linspace(low, high, _ZDW_REFINE_SAMPLES)
        sampled = base.beta2(mode, grid, z)
        crossing = np.flatnonzero(np.sign(sampled[:-1]) * np.sign(sampled[1:]) <= 0.0)
        if crossing.size == 0:
            return math.nan
        i = int(crossing[0])
        low, high = float(grid[i]), float(grid[i + 1])
        # The first change has a nonzero sample below it, so the two cannot be equal.
        first, last = float(sampled[i]), float(sampled[i + 1])
    return low - first * (high - low) / (last - first)


def _zdw_near(mode: Mode, omega: float, z: float) -> float:
    """Return the zero-dispersion wavelength nearest a pump frequency.

    :func:`solprop.modes.base.zdw` brackets one sign change across its whole window,
    which a gas resonance in the vacuum ultraviolet defeats: krypton flips the sign of
    the dispersion twice near 120 nm, the ends then agree, and the zero at 769 nm of a
    14 um capillary at 15 bar is refused. Here the sign is sampled, the changes are
    refined outwards from the pump, the changes through a pole are dropped, and the zero
    nearest the pump is the one reported; a change further than the best so far cannot
    beat it, which spares the dozens of changes an antiresonant fibre's resonances
    give. Two limits of a sampled search remain: a zero and a pole inside one sample
    interval (3.6e13 rad/s) restore the sign and are both missed, and a sample that is
    not finite cannot bracket anything.

    - ``omega``: pump angular frequency in rad/s.
    - ``z``: position along the waveguide, m.
    - Returns: wavelength in m, or NaN when the dispersion has no zero in 100 nm to 3
      um.
    """
    grid = np.linspace(
        2.0 * math.pi * C / _ZDW_RANGE[1],
        2.0 * math.pi * C / _ZDW_RANGE[0],
        _ZDW_SAMPLES,
    )
    nearest = math.nan
    # The refinement converges onto a pole as readily as onto a zero, and a model may
    # divide by zero exactly there; the group-delay test below tells the two apart.
    with _probing(), np.errstate(divide="ignore", invalid="ignore"):
        sampled = base.beta2(mode, grid, z)
        changes = np.flatnonzero(np.sign(sampled[:-1]) * np.sign(sampled[1:]) < 0.0)
        # How near each bracket can bring a zero: 0 for one holding the pump.
        reach = np.maximum.reduce(
            [grid[changes] - omega, omega - grid[changes + 1], np.zeros(changes.size)]
        )
        for i in changes[np.argsort(reach, kind="stable")]:
            low, high = float(grid[i]), float(grid[i + 1])
            if not math.isnan(nearest) and max(low - omega, omega - high) >= abs(
                nearest - omega
            ):
                break
            root = _refine_zero(mode, low, high, z)
            if math.isnan(root):
                continue
            delay = base.beta1(mode, np.array([low, root, high]), z)
            bounded = max(abs(sampled[i]), abs(sampled[i + 1])) * (high - low)
            closer = math.isnan(nearest) or abs(root - omega) < abs(nearest - omega)
            if closer and abs(delay[1] - delay[0]) <= _POLE_FACTOR * bounded:
                nearest = root
    if math.isnan(nearest):
        return math.nan
    # Outside the probing context, so a zero beyond the index model's range warns.
    base.beta(mode, np.array([nearest]), z)
    return 2.0 * math.pi * C / nearest
