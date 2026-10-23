# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Raman response models and the reference evaluation of the Raman polarisation.

A Raman response is a medium's *delayed* answer to the field: the molecular vibration
or rotation the pulse drives keeps ringing after the pulse has gone, and the
polarisation at time ``t`` depends on the whole history of the drive. It is therefore a
convolution, where the Kerr response of :mod:`solprop.nonlinear.responses` is pointwise,
and that difference is the whole of why this module exists separately.

The models here are a damped oscillator, the
vibrational mode of a molecular gas, the non-rigid-rotor rotational ladder with its
Boltzmann populations, the intermediate-broadening line set of an amorphous solid, and
a shape sampled from an optmat silica bundle. Each is a frozen dataclass returning
``h(t)`` for ``t >= 0``, and each declares how its coherence dephases with density.
The molecular and line-set models take a caller's own constants
(:mod:`solprop.physdata.raman_types`); every built-in gas's response is optmat's
(:mod:`solprop.nonlinear.optmat_raman`, SEP 3) and fused silica's is its bundle.

Beside them is :class:`SilicaTensorResponse`: the delayed
**tensor** response of fused silica declared by a material region of a general modal
guide, whose two channels are a share of the region's own ``chi3`` rather than a
per-density response, and whose kernels are therefore in **1/s** and not in the units
below.

**Units.** Every model of a gas or a scalar route returns ``h(t)`` in
**m^5 V^-2 s^-1**, the units that make

.. math::

    P(t) = \varepsilon_0\, \rho\, E(t) \int h(t - t')\, E^2(t')\, dt'

hold with ``P`` in C/m^2, the number density ``rho`` in m^-3 and the field ``E`` in V/m.
So ``integral h dt`` carries the units of
:attr:`solprop.nonlinear.responses.KerrResponse.chi3` per unit density, m^5/V^2, and the
Raman and Kerr responses of the same medium are directly comparable. The ``eps0``
stands outside the integral, so ``h`` here is linear in ``eps0``; a convention whose
outer expression has no ``eps0`` hides it inside every ``K`` as ``(4 pi eps0)^2`` and
has an ``h`` quadratic in ``eps0``, larger by that factor. The scalar vibrational factor
also includes the independent amplitude correction described in :func:`vibrational`.

The manual page is ``docs/guide/media/raman.md``; the kernel's view of the response is
``docs/dev/kernel-contract.md`` section 5, and the sampling onto the doubled grid is
:mod:`solprop.translate.raman`.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol, cast, runtime_checkable

import numpy as np
from jaxtyping import Complex, Float, Int, Num
from scipy import special

from solprop import fourier
from solprop._display import Displayed, engineering, line
from solprop.constants import EPS0, HBAR, K_B, ROOM_TEMPERATURE, C
from solprop.exceptions import GridError, MaterialError
from solprop.nonlinear.responses import ResponseTraits
from solprop.physdata.gases import Gas, as_gas
from solprop.physdata.glasses import SilicaBundle
from solprop.physdata.raman_types import (
    Dephasing,
    IntermediateBroadeningRaman,
    LinAgrawalRaman,
    Linewidth,
    MolecularOptions,
    MolecularRaman,
    RotationalRaman,
    VibrationalRaman,
    linewidth_from_dephasing_time,
)
from solprop.physdata.selection import MaterialRecord, MaterialSelection
from solprop.physdata.species import Species, built_in_population

__all__ = [
    "MIN_ROTATIONAL_LEVELS",
    "SILICA_NAMES",
    "SILICA_TENSOR_AMPLITUDE",
    "SILICA_TENSOR_INTEGRALS",
    "DampedOscillator",
    "Dephasing",
    "IntermediateBroadening",
    "MolecularOptions",
    "RamanModel",
    "RamanResponse",
    "RamanTensorTerm",
    "RotationalLines",
    "SampledShape",
    "SilicaTensorResponse",
    "bundle_response",
    "normalised_damped_oscillator",
    "raman_drive",
    "response_for",
    "rotational_nonrigid",
    "undamped_dephasing",
    "vibrational",
]

logger = logging.getLogger(__name__)

MIN_ROTATIONAL_LEVELS: Final = 3
"""Levels a rotational ladder needs before one ``J -> J + 2`` pair exists."""


def _tensor_bundle() -> SilicaBundle:
    """Return optmat's recommended tensor bundle of fused silica, the default.

    Returns
    -------
    SilicaBundle
        ``SilicaBundle.resolve(tensor=True)``: Lin and Agrawal's shape (SEP 3, D11).
    """
    return SilicaBundle.resolve(tensor=True)


def undamped_dephasing() -> Dephasing:
    """Return the dephasing of a coherence that does not decay with density.

    The right choice for a model that carries its own damping, such as the
    intermediate-broadening line set of a glass, whose Lorentzian widths are the
    dephasing and do not depend on how much glass there is.

    Returns
    -------
    Dephasing
        A dephasing whose factor is one at every density and every delay.

    Examples
    --------
    >>> from solprop.nonlinear.raman import undamped_dephasing
    >>> undamped_dephasing().decay_rate(1e25)
    0.0
    """
    return Dephasing(Linewidth())


@runtime_checkable
class RamanModel(Protocol):
    """One term of a Raman response: an undamped shape and a density-dependent decay.

    The split is what lets the translation layer sample the expensive part once per
    grid and recompute only the cheap exponential when the density changes along a
    pressure gradient. A model is responsible for the part of ``h(t)`` that does not
    move with density; :attr:`dephasing` supplies the rest.
    """

    @property
    def dephasing(self) -> Dephasing:
        """How this term's coherence decays with the local number density."""
        ...

    def undamped(self, t: Float[np.ndarray, " n_to"]) -> Float[np.ndarray, " n_to"]:
        """Return ``h(t)`` without the density-dependent decay, in m^5 V^-2 s^-1.

        Zero before ``t = 0``, so that the response is causal. At ``t = 0`` it is zero
        for a response that rises continuously from zero; a response that jumps there
        takes half its right-hand limit, the trapezoid endpoint of the causal
        convolution.

        Parameters
        ----------
        t : numpy.ndarray
            Delays since the drive, s, shape ``(n_to,)``.

        Returns
        -------
        numpy.ndarray
            The undamped response, m^5 V^-2 s^-1, shape ``(n_to,)``.
        """
        ...


@dataclass(frozen=True)
class DampedOscillator:
    """A single Raman line as a damped harmonic oscillator.

    ``h(t) = amplitude * sin(frequency * t) * exp(-t / tau2)`` for ``t > 0`` and zero
    otherwise: the impulse response of a driven oscillator in the weak-damping limit,
    and the model behind every molecular vibration and every rotational line.

    Attributes
    ----------
    amplitude : float
        Scale factor ``K``, in m^5 V^-2 s^-1, the units of ``h`` itself. It may
        be negative: a rotational line's weight changes sign with the population
        difference it comes from.
    frequency : float
        Angular frequency of the transition, rad/s, positive.
    dephasing : Dephasing
        How the coherence decays with the local number density.

    Raises
    ------
    MaterialError
        If the frequency is not positive and finite.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.raman import DampedOscillator, undamped_dephasing
    >>> line = DampedOscillator(2.0, np.pi / 2.0, undamped_dephasing())
    >>> line.undamped(np.array([-1.0, 0.0, 1.0])).tolist()
    [0.0, 0.0, 2.0]
    """

    amplitude: float
    frequency: float
    dephasing: Dephasing

    def __post_init__(self) -> None:
        """Reject a line that cannot oscillate.

        Raises
        ------
        MaterialError
            If the frequency is not positive and finite.
        """
        if not (self.frequency > 0.0 and math.isfinite(self.frequency)):
            raise MaterialError(
                f"a Raman line needs a positive, finite angular frequency, got "
                f"{self.frequency} rad/s"
            )

    def undamped(self, t: Float[np.ndarray, " n_to"]) -> Float[np.ndarray, " n_to"]:
        """Return the undamped response on a time axis, in m^5 V^-2 s^-1.

        Parameters
        ----------
        t : numpy.ndarray
            Delays, s, shape ``(n_to,)``.

        Returns
        -------
        numpy.ndarray
            ``amplitude * sin(frequency * t)`` where ``t > 0`` and zero elsewhere,
            shape ``(n_to,)``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.raman import DampedOscillator, undamped_dephasing
        >>> line = DampedOscillator(1.0, 1.0, undamped_dephasing())
        >>> float(line.undamped(np.array([np.pi / 2.0]))[0])
        1.0
        """
        times = np.asarray(t, dtype=float)
        return np.where(
            times > 0.0, self.amplitude * np.sin(self.frequency * times), 0.0
        )


def normalised_damped_oscillator(
    amplitude: float, frequency: float, dephasing_time: float
) -> DampedOscillator:
    """Build a damped oscillator whose response integrates to a given value.

    The integral of ``K sin(Omega t) exp(-t / tau2)`` over the positive axis is
    ``K Omega / (Omega^2 + 1/tau2^2)``, so scaling ``K`` by the reciprocal of that
    factor makes the integral exactly
    ``amplitude``. Useful wherever the Raman strength is known as a fraction of the
    total nonlinearity rather than from molecular constants, which is how a fibre GNLSE
    states it.

    A module-level function rather than a classmethod, for the reason given in
    :func:`solprop.physdata.raman_types.linewidth_from_dephasing_time`.

    Parameters
    ----------
    amplitude : float
        The value ``integral h dt`` is to take, m^5/V^2 per unit density.
    frequency : float
        Angular frequency of the transition, rad/s, positive.
    dephasing_time : float
        Density-independent coherence time ``tau2`` in s, positive.

    Returns
    -------
    DampedOscillator
        The line, with a density-independent dephasing.

    Raises
    ------
    MaterialError
        If the frequency or the dephasing time is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import quad
    >>> from solprop.nonlinear.raman import normalised_damped_oscillator
    >>> line = normalised_damped_oscillator(1.0, 2 * np.pi * 100e12, 100e-15)
    >>> integral = quad(
    ...     lambda t: float(line.undamped(np.array([t]))[0])
    ...     * float(line.dephasing.damping(np.array([t]), 1.0)[0]),
    ...     0.0,
    ...     2e-12,
    ...     limit=400,
    ... )[0]
    >>> bool(abs(integral - 1.0) < 1e-8)
    True
    """
    if not dephasing_time > 0.0:
        raise MaterialError(
            f"a normalised Raman line needs a positive dephasing time, got "
            f"{dephasing_time} s"
        )
    if not (frequency > 0.0 and math.isfinite(frequency)):
        raise MaterialError(
            f"a Raman line needs a positive, finite angular frequency, got "
            f"{frequency} rad/s"
        )
    scaled = amplitude * (frequency**2 + 1.0 / dephasing_time**2) / frequency
    return DampedOscillator(
        amplitude=scaled,
        frequency=frequency,
        dephasing=Dephasing(linewidth_from_dephasing_time(dephasing_time)),
    )


def vibrational(parameters: VibrationalRaman) -> DampedOscillator:
    r"""Build the Raman response of a molecular vibration.

    A vibrational mode is one damped oscillator whose strength comes from the
    volume polarisability derivative along a displacement coordinate,

    .. math::

        K = \frac{(4\pi\varepsilon_0\, d\alpha/dQ)^2}
            {2\mu\Omega_v\varepsilon_0},

    the ``(4 pi eps0)`` converting the tabulated *volume* polarisability derivative
    to an SI one and the final ``eps0`` being the one this package's ``h`` carries
    outside the integral (see the module docstring).
    The induced-dipole energy ``-alpha_SI(Q) E_real^2 / 2`` gives force
    ``alpha_SI'(Q) E_real^2 / 2``. Its retarded displacement response gives the
    denominator two above. The cycle average is already in :func:`raman_drive`;
    applying another half here would count it twice, which is why the denominator is
    two and not four. For a diatomic bond coordinate use its reduced mass; do not insert
    a mass-weighted normal-coordinate derivative while retaining that same mass.

    Parameters
    ----------
    parameters : VibrationalRaman
        The mode's constants, the caller's own (a built-in gas's response is optmat's,
        :func:`solprop.nonlinear.optmat_raman.gas_raman`).

    Returns
    -------
    DampedOscillator
        The vibrational line, with the mode's own density-dependent dephasing.

    References
    ----------
    S.-F. Gao, Y.-Y. Wang, F. Belli, C. Brahms, P. Wang and J. C. Travers, "From
    Raman frequency combs to supercontinuum generation in nitrogen-filled hollow-core
    anti-resonant fiber", Laser Photonics Rev. 16, 2100426 (2022),
    doi:10.1002/lpor.202100426, and Wahlstrand et al., Phys. Rev. A 92, 063828 (2015),
    doi:10.1103/PhysRevA.92.063828, are the
    inherited model lineage. The prefactor here is independently derived from
    the induced-dipole Hamiltonian under this module's field convention, rather
    than copied from that lineage. N2 bond-coordinate data are from Murphy,
    Holzer and Bernstein, Appl. Spectrosc. 23, 211 (1969), Table I and Eq. (4),
    doi:10.1366/000370269774380824. The prefactor's common amplitude is verified by
    an independent quantum-oscillator impulse calculation; each material's tabulated
    derivative is not independently recalibrated, and the transfer of N2's derivative
    from 435.8 nm to the near infrared and the omission of the anisotropic vibrational
    response remain assumptions.

    Examples
    --------
    >>> from solprop.constants import ATOMIC_MASS
    >>> from solprop.nonlinear.raman import vibrational
    >>> from solprop.physdata import Linewidth, VibrationalRaman
    >>> mode = VibrationalRaman(
    ...     frequency=7.826769e14,
    ...     polarisability_derivative=1.3e-20,
    ...     reduced_mass=1.00784 * ATOMIC_MASS / 2.0,
    ...     linewidth=Linewidth(inverse_density=309e6, density=52.2e6),
    ...     citation="a caller's H2 constants, Phys. Rev. A 92, 063828 (2015)",
    ... )
    >>> line = vibrational(mode)
    >>> line.frequency == mode.frequency
    True
    >>> f"{line.amplitude:.6e}"
    '1.803981e-37'
    """
    polarisability_si = 4.0 * math.pi * EPS0 * parameters.polarisability_derivative
    amplitude = polarisability_si**2 / (
        2.0 * parameters.reduced_mass * parameters.frequency * EPS0
    )
    return DampedOscillator(
        amplitude=amplitude,
        frequency=parameters.frequency,
        dephasing=Dephasing(parameters.linewidth),
    )


@dataclass(frozen=True)
class RotationalLines:
    """The rotational Raman spectrum of a linear molecule, line by line.

    A non-rigid rotor's levels are ``E_J = 2 pi hbar c (B J(J+1) - D (J(J+1))^2)`` and
    the Raman-active transitions are the ``S`` branch, ``J -> J + 2``. Each pair
    contributes a damped oscillator whose weight is set by the population difference
    between its two levels, so a line can have either sign, and the whole ladder shares
    one dephasing: the ``J`` dependence of the linewidth is dropped.

    Build one with :func:`rotational_nonrigid`; the arrays are stored so that a test or
    a manual page can look at the ladder itself rather than only at its sum.

    Attributes
    ----------
    quantum_numbers : numpy.ndarray
        The rotational quantum numbers ``J`` retained, shape ``(n_levels,)``, integer.
    energies : numpy.ndarray
        Level energies in J, shape ``(n_levels,)``.
    populations : numpy.ndarray
        Fractional level populations, shape ``(n_levels,)``, summing to one.
    frequencies : numpy.ndarray
        Transition angular frequencies of the ``J -> J + 2`` pairs, rad/s, shape
        ``(n_lines,)`` with ``n_lines = n_levels - 2``.
    amplitudes : numpy.ndarray
        Line scale factors in m^5 V^-2 s^-1, shape ``(n_lines,)``. Signed.
    dephasing : Dephasing
        The dephasing shared by every line.
    temperature : float
        The temperature the populations were computed at, K.

    Raises
    ------
    MaterialError
        If the array lengths do not agree, or there are fewer than three levels.

    Examples
    --------
    >>> from solprop.nonlinear.raman import rotational_nonrigid
    >>> from solprop.physdata import Linewidth, RotationalRaman
    >>> nitrogen = RotationalRaman(
    ...     b=199.0,
    ...     d=5.74e-4,
    ...     degeneracy_odd=1,
    ...     degeneracy_even=2,
    ...     polarisability_anisotropy=6.7e-31,
    ...     linewidth=Linewidth(density=3.3e9),
    ...     citation="a caller's N2 constants",
    ... )
    >>> lines = rotational_nonrigid(nitrogen)
    >>> lines.frequencies.size == lines.quantum_numbers.size - 2
    True
    >>> f"{float(lines.populations.sum()):.15f}"
    '1.000000000000000'
    """

    # Five independent axis names for five arrays whose lengths are related, so that
    # __post_init__ is what enforces the relation. Sharing a name would let the test
    # session's runtime checker enforce it instead, and the check would then be dead
    # code in production, where no hook runs (AGENTS.md, "Types and arrays").
    quantum_numbers: Int[np.ndarray, " n_levels"]
    energies: Float[np.ndarray, " n_energies"]
    populations: Float[np.ndarray, " n_populations"]
    frequencies: Float[np.ndarray, " n_lines"]
    amplitudes: Float[np.ndarray, " n_amplitudes"]
    dephasing: Dephasing
    temperature: float

    def __post_init__(self) -> None:
        """Check the ladder's shapes and freeze its arrays.

        The axes are independent names in the annotations, so this validation is what
        enforces the relation between them where no runtime type checker is installed.

        Raises
        ------
        MaterialError
            If the array lengths do not agree, or there are fewer than three levels.
        """
        n_levels = self.quantum_numbers.size
        if n_levels < MIN_ROTATIONAL_LEVELS:
            raise MaterialError(
                f"a rotational Raman ladder needs at least {MIN_ROTATIONAL_LEVELS} "
                f"levels to form one J, J+2 pair, got {n_levels}"
            )
        if not self.energies.size == self.populations.size == n_levels:
            raise MaterialError(
                f"the level arrays must all have {n_levels} entries, got "
                f"{self.energies.size} energies and {self.populations.size} populations"
            )
        if not self.frequencies.size == self.amplitudes.size == n_levels - 2:
            raise MaterialError(
                f"a ladder of {n_levels} levels has {n_levels - 2} lines, got "
                f"{self.frequencies.size} frequencies and {self.amplitudes.size} "
                f"amplitudes"
            )
        for name in (
            "quantum_numbers",
            "energies",
            "populations",
            "frequencies",
            "amplitudes",
        ):
            array = np.array(getattr(self, name), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def n_lines(self) -> int:
        """Return the number of ``J -> J + 2`` transitions.

        Returns
        -------
        int
            Two fewer than the number of levels.
        """
        return int(self.frequencies.size)

    def undamped(self, t: Float[np.ndarray, " n_to"]) -> Float[np.ndarray, " n_to"]:
        """Return the undamped response of the whole ladder, in m^5 V^-2 s^-1.

        Parameters
        ----------
        t : numpy.ndarray
            Delays, s, shape ``(n_to,)``.

        Returns
        -------
        numpy.ndarray
            ``sum_i amplitude_i sin(frequency_i t)`` where ``t > 0`` and zero
            elsewhere, shape ``(n_to,)``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.raman import rotational_nonrigid
        >>> from solprop.physdata import Linewidth, RotationalRaman
        >>> hydrogen = RotationalRaman(
        ...     b=5890.0,
        ...     d=5.0,
        ...     degeneracy_odd=3,
        ...     degeneracy_even=1,
        ...     polarisability_anisotropy=3e-31,
        ...     linewidth=Linewidth(inverse_density=6.15e6, density=114e6),
        ...     citation="a caller's H2 constants",
        ... )
        >>> lines = rotational_nonrigid(hydrogen)
        >>> bool(np.all(lines.undamped(np.array([-1e-12, 0.0])) == 0.0))
        True
        """
        times = np.asarray(t, dtype=float)
        total = np.zeros_like(times)
        # A loop over the tens of lines rather than an outer product: the fine grid can
        # be 2^17 samples and the (n_lines, n_to) intermediate would be tens of
        # megabytes for no gain.
        for amplitude, frequency in zip(self.amplitudes, self.frequencies, strict=True):
            total += amplitude * np.sin(frequency * times)
        return np.where(times > 0.0, total, 0.0)


def rotational_nonrigid(
    parameters: RotationalRaman,
    temperature: float = ROOM_TEMPERATURE,
    *,
    min_j: int = 0,
    max_j: int = 50,
) -> RotationalLines:
    r"""Build the rotational Raman spectrum of a linear molecule.

    The ladder is truncated wherever the non-rigid-rotor energy stops increasing with
    ``J``: beyond that point the expansion ``B J(J+1) - D (J(J+1))^2`` has left its
    range of validity and the "levels" it predicts are an artefact. For H2, whose
    centrifugal distortion is large, that happens at ``J = 24``.

    Populations are Boltzmann with the nuclear-spin statistical weights, normalised
    over the retained levels. The line weight of the ``J -> J + 2`` pair is

    .. math::

        K \frac{(J+1)(J+2)}{2J+3}
        \left(\frac{p_{J+2}}{2J+5} - \frac{p_J}{2J+1}\right),
        \qquad
        K = -\frac{(4\pi\varepsilon_0)^2}{\varepsilon_0}
            \frac{2}{15}\frac{\Delta\alpha^2}{\hbar},

    with ``p_J`` the fractional populations. ``K`` is negative and so is the bracket,
    because the population per magnetic sublevel falls with ``J``, so the lines come
    out positive: the same sign as a vibration, and the sign that amplifies the Stokes
    side.

    Parameters
    ----------
    parameters : RotationalRaman
        The molecule's rotational constants, the caller's own.
    temperature : float, optional
        Gas temperature in K, positive. Default
        :data:`solprop.constants.ROOM_TEMPERATURE`.
    min_j : int, optional
        Lowest rotational quantum number to include, non-negative. Default 0.
    max_j : int, optional
        Highest rotational quantum number to consider before truncation. Default 50.

    Returns
    -------
    RotationalLines
        The retained levels, their populations and the resulting lines.

    Raises
    ------
    MaterialError
        If the temperature is not positive, the ``J`` range is inverted or shorter
        than three levels, or the truncation leaves too few levels to form a pair.

    References
    ----------
    Laser Photonics Rev. 16, 2100426 (2022), doi:10.1002/lpor.202100426, the SI form
    of the model of Wahlstrand et al., Phys. Rev. A 92, 063828 (2015),
    doi:10.1103/PhysRevA.92.063828.

    Examples
    --------
    >>> from solprop.nonlinear.raman import rotational_nonrigid
    >>> from solprop.physdata import Linewidth, RotationalRaman
    >>> hydrogen = RotationalRaman(
    ...     b=5890.0,
    ...     d=5.0,
    ...     degeneracy_odd=3,
    ...     degeneracy_even=1,
    ...     polarisability_anisotropy=3e-31,
    ...     linewidth=Linewidth(inverse_density=6.15e6, density=114e6),
    ...     citation="a caller's H2 constants",
    ... )
    >>> lines = rotational_nonrigid(hydrogen, 293.15)
    >>> int(lines.quantum_numbers[-1])
    24
    >>> lines.n_lines
    23
    """
    if not (temperature > 0.0 and math.isfinite(temperature)):
        raise MaterialError(
            f"a rotational Raman ladder needs a positive, finite temperature, got "
            f"{temperature} K"
        )
    if min_j < 0 or max_j < min_j:
        raise MaterialError(
            f"the rotational quantum numbers must satisfy 0 <= min_j <= max_j, got "
            f"min_j={min_j}, max_j={max_j}"
        )
    quantum_numbers = np.arange(min_j, max_j + 1, dtype=np.int64)
    rotation = quantum_numbers * (quantum_numbers + 1)
    energies = (
        2.0
        * math.pi
        * HBAR
        * C
        * (parameters.b * rotation - parameters.d * rotation**2)
    )
    if quantum_numbers.size < MIN_ROTATIONAL_LEVELS:
        raise MaterialError(
            f"a rotational Raman ladder needs at least {MIN_ROTATIONAL_LEVELS} levels "
            f"to form one J, J+2 pair; min_j={min_j}, max_j={max_j} gives "
            f"{quantum_numbers.size}"
        )
    decreasing = np.flatnonzero(np.diff(energies) < 0.0)
    if decreasing.size:
        # Keep every level up to and including the last one before the energy turns
        # over. The index is an array position, not a J value, so this holds for any
        # min_j.
        kept = int(decreasing[0]) + 1
        quantum_numbers = quantum_numbers[:kept]
        energies = energies[:kept]
        if quantum_numbers.size < MIN_ROTATIONAL_LEVELS:
            raise MaterialError(
                f"the non-rigid rotor energy of this molecule stops increasing after "
                f"{quantum_numbers.size} levels, too few to form a J, J+2 pair; B and "
                f"D are {parameters.b} and {parameters.d} m^-1"
            )
    degeneracy = np.where(
        quantum_numbers % 2 == 1,
        parameters.degeneracy_odd,
        parameters.degeneracy_even,
    )
    populations = (
        degeneracy * (2 * quantum_numbers + 1) * np.exp(-energies / (K_B * temperature))
    )
    populations = populations / populations.sum()
    lower = quantum_numbers[:-2]
    # The S branch: hbar cancels between the level energies and the division, so the
    # line positions carry no Planck constant at all, only c and the molecular data.
    frequencies = (energies[2:] - energies[:-2]) / HBAR
    scale = (
        -((4.0 * math.pi * EPS0) ** 2)
        * (2.0 / 15.0)
        * parameters.polarisability_anisotropy**2
        / (HBAR * EPS0)
    )
    amplitudes = (
        scale
        * (lower + 1)
        * (lower + 2)
        / (2 * lower + 3)
        * (populations[2:] / (2 * lower + 5) - populations[:-2] / (2 * lower + 1))
    )
    return RotationalLines(
        quantum_numbers=quantum_numbers,
        energies=energies,
        populations=populations,
        frequencies=frequencies,
        amplitudes=amplitudes,
        dephasing=Dephasing(parameters.linewidth),
        temperature=temperature,
    )


@dataclass(frozen=True)
class IntermediateBroadening:
    r"""The Raman response of an amorphous solid, as intermediately broadened lines.

    Each line of the set contributes a damped sinusoid with both a homogeneous
    (Lorentzian) and an inhomogeneous (Gaussian) width,

    .. math::

        h(t) \propto \sum_i A_i \mathrm{e}^{-\gamma_i t}
                     \mathrm{e}^{-\Gamma_i^2 t^2/4}\sin(\omega_i t),

    the Hollenbeck-Cantrell model of fused silica. The widths *are* the dephasing, so
    the response does not depend on density and :attr:`dephasing` is the trivial one;
    a glass is a fixed medium and its ``density`` path is one.

    The amplitudes of the line set are relative, so the model is normalised: its
    integral is exactly :attr:`amplitude`, which is therefore the Raman contribution to
    the medium's third-order susceptibility in m^5/V^2 per unit of the density argument.

    **The normalisation is analytic.** A numerical quadrature of the oscillatory sum
    would need to reach out to about 1 ns over an integrand supported in its first
    0.1 % and crossing zero some thousands of times, at the mercy of the quadrature's
    tolerance. Here

    .. math::

        \int_0^\infty \mathrm{e}^{-\gamma t}\mathrm{e}^{-\Gamma^2t^2/4}\sin(\omega t)
        \,dt = \mathrm{Im}\left[\frac{\sqrt{\pi}}{\Gamma}\,
        w\!\left(\frac{\omega + \mathrm{i}\gamma}{\Gamma}\right)\right],

    with ``w`` the Faddeeva function :data:`scipy.special.wofz`, which is the stable
    spelling of ``exp(z^2) erfc(z)`` in the half plane these arguments occupy.

    Attributes
    ----------
    lines : IntermediateBroadeningRaman
        The line set, the caller's own; fused silica's built-in response is its
        optmat bundle (:func:`bundle_response`).
    amplitude : float, optional
        The value ``integral h dt`` is to take, m^5/V^2 per unit density. Default 1.0,
        which makes the model a pure shape.

    Raises
    ------
    MaterialError
        If the line set integrates to zero, so that it cannot be normalised.

    References
    ----------
    D. Hollenbeck and C. D. Cantrell, "Multiple-vibrational-mode model for fiber-optic
    Raman gain spectrum and response function", J. Opt. Soc. Am. B 19, 2886 (2002),
    doi:10.1364/JOSAB.19.002886.

    Examples
    --------
    >>> from solprop.nonlinear.raman import IntermediateBroadening
    >>> from solprop.physdata import IntermediateBroadeningRaman
    >>> glass = IntermediateBroadening(
    ...     IntermediateBroadeningRaman(
    ...         frequencies=(8.2e13,),
    ...         amplitudes=(1.0,),
    ...         gaussian_widths=(9.6e12,),
    ...         lorentzian_widths=(3.0e12,),
    ...         citation="one illustrative line",
    ...     )
    ... )
    >>> f"{glass.scale:.9e}"
    '8.154299442e+13'
    """

    lines: IntermediateBroadeningRaman
    amplitude: float = 1.0

    @property
    def dephasing(self) -> Dephasing:
        """Return the density-dependent dephasing, which for a glass is none.

        Returns
        -------
        Dephasing
            A dephasing whose factor is one everywhere; the Lorentzian widths of
            :attr:`lines` already carry the decay.
        """
        return undamped_dephasing()

    @property
    def raw_integral(self) -> float:
        """Return the integral of the unnormalised line sum over the positive axis.

        Returns
        -------
        float
            ``sum_i A_i Im[(sqrt(pi)/Gamma_i) w((omega_i + i gamma_i)/Gamma_i)]``, in
            the units of the tabulated amplitudes times seconds.

        Examples
        --------
        >>> from solprop.nonlinear.raman import IntermediateBroadening
        >>> from solprop.physdata import IntermediateBroadeningRaman
        >>> glass = IntermediateBroadening(
        ...     IntermediateBroadeningRaman(
        ...         frequencies=(8.2e13,),
        ...         amplitudes=(1.0,),
        ...         gaussian_widths=(9.6e12,),
        ...         lorentzian_widths=(3.0e12,),
        ...         citation="one illustrative line",
        ...     )
        ... )
        >>> f"{glass.raw_integral:.9e}"
        '1.226346919e-14'
        """
        lines = self.lines
        amplitudes = np.asarray(lines.amplitudes, dtype=float)
        frequencies = np.asarray(lines.frequencies, dtype=float)
        gaussian = np.asarray(lines.gaussian_widths, dtype=float)
        lorentzian = np.asarray(lines.lorentzian_widths, dtype=float)
        per_line = np.imag(
            (np.sqrt(np.pi) / gaussian)
            * special.wofz((frequencies + 1j * lorentzian) / gaussian)
        )
        return float(np.dot(amplitudes, per_line))

    @property
    def scale(self) -> float:
        """Return the factor the raw line sum is multiplied by.

        Returns
        -------
        float
            :attr:`amplitude` divided by :attr:`raw_integral`, so that the response
            integrates to :attr:`amplitude`.

        Raises
        ------
        MaterialError
            If the line set integrates to zero.
        """
        integral = self.raw_integral
        if integral == 0.0:
            raise MaterialError(
                "this Raman line set integrates to zero and cannot be normalised"
            )
        return self.amplitude / integral

    def undamped(self, t: Float[np.ndarray, " n_to"]) -> Float[np.ndarray, " n_to"]:
        """Return the normalised response on a time axis, in m^5 V^-2 s^-1.

        Named ``undamped`` for the :class:`RamanModel` protocol, but the Lorentzian and
        Gaussian widths of the line set are already in it: what the name excludes is
        the *density-dependent* decay, and a glass has none.

        Parameters
        ----------
        t : numpy.ndarray
            Delays, s, shape ``(n_to,)``.

        Returns
        -------
        numpy.ndarray
            The response where ``t > 0`` and zero elsewhere, shape ``(n_to,)``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.raman import IntermediateBroadening
        >>> from solprop.physdata import IntermediateBroadeningRaman
        >>> glass = IntermediateBroadening(
        ...     IntermediateBroadeningRaman(
        ...         frequencies=(8.2e13,),
        ...         amplitudes=(1.0,),
        ...         gaussian_widths=(9.6e12,),
        ...         lorentzian_widths=(3.0e12,),
        ...         citation="one illustrative line",
        ...     )
        ... )
        >>> bool(np.all(glass.undamped(np.array([-1e-15, 0.0])) == 0.0))
        True
        """
        times = np.asarray(t, dtype=float)
        total = np.zeros_like(times)
        for amplitude, frequency, gaussian, lorentzian in zip(
            self.lines.amplitudes,
            self.lines.frequencies,
            self.lines.gaussian_widths,
            self.lines.lorentzian_widths,
            strict=True,
        ):
            total += (
                amplitude
                * np.exp(-lorentzian * times)
                * np.exp(-((gaussian * times) ** 2) / 4.0)
                * np.sin(frequency * times)
            )
        return np.where(times > 0.0, self.scale * total, 0.0)


def _check_evaluate(
    values: np.ndarray, kind: str, dt: float, max_components: int = 2
) -> None:
    """Validate the arguments of :meth:`RamanResponse.evaluate`.

    ``kind`` is a plain string rather than :data:`solprop.fourier.GridKind` for the
    reason given in :func:`solprop.nonlinear.responses._check_evaluate`.

    Parameters
    ----------
    values : numpy.ndarray
        The samples passed as ``field_t``.
    kind : str
        Layout name to check.
    dt : float
        Sample spacing of the fine grid, s.
    max_components : int, optional
        How many Cartesian components the caller carries: two for the legacy
        routes, whose transverse state is at most two-component, and three for the
        general modal guide's tensor responses.

    Raises
    ------
    GridError
        If the layout is unknown, the array shape or dtype does not match it, the
        sample spacing is not positive, or samples are not finite numbers.
    """
    if kind not in ("field", "envelope"):
        msg = f"kind must be 'field' or 'envelope', got {kind!r}"
        raise GridError(msg)
    if values.ndim != 2:
        msg = f"field_t must have shape (n_pol, n_to), got {values.shape}"
        raise GridError(msg)
    if not 1 <= values.shape[0] <= max_components:
        msg = (
            f"Raman needs between one and {max_components} physical components, "
            f"got {values.shape[0]}"
        )
        raise GridError(msg)
    if values.shape[1] < 2:
        msg = f"a Raman convolution needs at least two samples, got {values.shape[1]}"
        raise GridError(msg)
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise GridError("Raman field samples must be finite numbers")
    if kind == "field" and np.iscomplexobj(values):
        msg = "a field grid carries the real field; got a complex array"
        raise GridError(msg)
    if kind == "envelope" and not np.iscomplexobj(values):
        msg = "an envelope grid carries a complex envelope; got a real array"
        raise GridError(msg)
    if not (dt > 0.0 and math.isfinite(dt)):
        msg = f"dt must be a positive, finite sample spacing in s, got {dt!r}"
        raise GridError(msg)


def raman_drive(
    field_t: Num[np.ndarray, "n_pol n_to"], *, kind: fourier.GridKind, thg: bool
) -> Float[np.ndarray, " n_to"]:
    """Return the real quantity the Raman response is driven by.

    The Raman coherence is driven by the *intensity* of the field, and which real
    quadratic form that is depends on the grid, exactly as the Kerr cubic does
    (``docs/guide/media/nonlinear.md``):

    ==========  =======  ===========================================================
    grid        ``thg``  drive
    ==========  =======  ===========================================================
    field       True     ``E^2``, which carries the ``2 omega`` beat and so radiates
                         the third harmonic through the later multiplication by ``E``
    field       False    ``|E_a|^2 / 2``, the cycle-averaged intensity, with the beat
                         removed
    envelope    --       ``|A|^2 / 2``, the same quantity written on the envelope
    ==========  =======  ===========================================================

    The factor of one half is the cycle average of ``cos^2``: ``E^2`` and
    ``|E_a|^2 / 2`` agree in their slowly varying part, so the two field-grid forms
    drive the coherence identically and differ only in the harmonic.

    Parameters
    ----------
    field_t : numpy.ndarray
        The field on the fine time grid, shape ``(1, n_to)``: real ``E(t)`` in V/m on a
        field grid, complex ``A(t)`` in V/m on an envelope grid.
    kind : {'field', 'envelope'}
        The spectral layout the samples belong to.
    thg : bool
        Whether the third harmonic is kept. Ignored on an envelope grid, which cannot
        carry it in a quadratic drive.

    Returns
    -------
    numpy.ndarray
        The drive in V^2/m^2, real, shape ``(n_to,)``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.raman import raman_drive
    >>> raman_drive(np.array([[3.0, -2.0]]), kind="field", thg=True).tolist()
    [9.0, 4.0]
    >>> raman_drive(np.array([[2.0 + 0j]]), kind="envelope", thg=True).tolist()
    [2.0]
    """
    values = np.asarray(field_t)
    column = values[0]
    if kind == "envelope":
        return _half_modulus_squared(column)
    if thg:
        return np.ascontiguousarray(np.real(column) ** 2)
    return _half_modulus_squared(fourier.analytic_signal(np.real(column)))


def _half_modulus_squared(
    values: Complex[np.ndarray, " n_to"],
) -> Float[np.ndarray, " n_to"]:
    """Return ``|z|^2 / 2`` without the square root that ``abs`` would take.

    ``abs(z) ** 2`` rounds twice and can miss the exact answer for an integer-valued
    complex number; the sum of squares is both closer and cheaper.

    Parameters
    ----------
    values : numpy.ndarray
        Complex samples, shape ``(n_to,)``.

    Returns
    -------
    numpy.ndarray
        Half the squared modulus, real, shape ``(n_to,)``.
    """
    return 0.5 * (np.real(values) ** 2 + np.imag(values) ** 2)


@dataclass(frozen=True)
class RamanTensorTerm:
    """Declare how one scalar linear-polarisation kernel contracts a vector field.

    A model's class does not determine its tensor character. In particular, a
    custom oscillator needs an explicit physical assignment before vector use.

    Attributes
    ----------
    model : RamanModel
        Scalar kernel in m^5 V^-2 s^-1, with its own dephasing law.
    contraction : {'scalar_only', 'isotropic', 'rank2'}
        No established vector contraction, the isotropic trace channel, or the
        three-dimensional traceless rank-two channel, respectively. Each channel
        reduces to ``model`` for a fixed real linear polarisation.

    Raises
    ------
    MaterialError
        If the model does not implement the Raman protocol or the label is unknown.
    """

    model: RamanModel
    contraction: Literal["scalar_only", "isotropic", "rank2"]

    def __post_init__(self) -> None:
        """Reject an incomplete model or an unknown tensor assignment."""
        if not isinstance(self.model, RamanModel):
            raise MaterialError("a Raman tensor term needs a RamanModel")
        if self.contraction not in ("scalar_only", "isotropic", "rank2"):
            raise MaterialError(f"unknown Raman contraction {self.contraction!r}")


def _check_density(density: float) -> None:
    """Reject a non-finite or negative molecular number density.

    Parameters
    ----------
    density : float
        Number density in m^-3.
    """
    if not math.isfinite(density) or density < 0.0:
        raise MaterialError(
            f"a Raman response needs a finite, non-negative number density, got "
            f"{density} m^-3"
        )


def _check_delays(t: Float[np.ndarray, " n_to"]) -> Float[np.ndarray, " n_to"]:
    """Validate finite real delays before evaluating molecular models.

    Parameters
    ----------
    t : numpy.ndarray
        Delays in s, shape ``(n_to,)``.

    Returns
    -------
    numpy.ndarray
        Finite real delays as float64.
    """
    times = np.asarray(t)
    if times.ndim != 1 or not np.issubdtype(times.dtype, np.number):
        raise GridError("Raman delays must be a one-dimensional numeric array")
    if np.iscomplexobj(times) or not np.all(np.isfinite(times)):
        raise GridError("Raman delays must be finite and real")
    return np.asarray(times, dtype=float)


def _causal_convolution(
    kernel: Float[np.ndarray, " n_to"],
    drive: Float[np.ndarray, "... n_to"],
    dt: float,
) -> Float[np.ndarray, "... n_to"]:
    """Evaluate a causal convolution with doubled padding along the last axis.

    Parameters
    ----------
    kernel : numpy.ndarray
        Causal kernel in m^5 V^-2 s^-1, shape ``(n_to,)``.
    drive : numpy.ndarray
        Real quadratic drive in V^2/m^2, with ``n_to`` samples on its last axis.
    dt : float
        Positive temporal spacing in s.

    Returns
    -------
    numpy.ndarray
        Causal integral with the drive's shape, in m^3.
    """
    n_to = kernel.size
    spectrum = np.fft.rfft(kernel, n=2 * n_to) * dt
    return np.fft.irfft(
        spectrum * np.fft.rfft(drive, n=2 * n_to, axis=-1),
        n=2 * n_to,
        axis=-1,
    )[..., :n_to]


def _quadratic_tensor(
    values: Num[np.ndarray, "n_pol n_to"], *, kind: fourier.GridKind, thg: bool
) -> Float[np.ndarray, "n_pol n_pol n_to"]:
    """Form the real molecular drive, retaining delayed off-diagonal coherence.

    Parameters
    ----------
    values : numpy.ndarray
        Local field in V/m, shape ``(n_pol, n_to)``.
    kind : {'field', 'envelope'}
        Temporal representation.
    thg : bool
        Keep the instantaneous real-field drive; ignored for envelopes.

    Returns
    -------
    numpy.ndarray
        Real symmetric quadratic tensor in V^2/m^2, shape ``(n_pol, n_pol, n_to)``.
    """
    if kind == "field" and thg:
        return np.real(values[:, None, :] * values[None, :, :])
    analytic = (
        values
        if kind == "envelope"
        else np.stack([fourier.analytic_signal(np.real(row)) for row in values])
    )
    return 0.5 * np.real(analytic[:, None, :] * np.conj(analytic[None, :, :]))


@dataclass(frozen=True, init=False)
class RamanResponse:
    """The delayed third-order polarisation of a Raman-active medium.

    The scalar reduction is ``P(t) = eps0 rho E(t) integral h(t - t') E^2(t') dt'``,
    with ``h`` the sum of the
    :class:`RamanModel` terms and the drive chosen per grid by :func:`raman_drive`.
    The convolution is what makes this response different in kind from the Kerr one:
    it has memory, so a backend evaluates it with a transform pair rather than sample
    by sample, and :mod:`solprop.translate.raman` prepares the kernel for that.

    Build one with :func:`response_for` rather than by hand, unless the models are
    yours. A tensor declaration couples the local Cartesian components through
    the isotropic trace and three-dimensional traceless molecular responses.
    Envelope drives use ``Re(A A†)/2``; beat-removed fields use their analytic
    signals for that drive and the real field for the final multiplication.

    Parameters
    ----------
    models : tuple of RamanModel or None, optional
        Legacy scalar-only kernels, in summation order. Supply exactly one nonempty
        route: ``models`` or keyword-only ``tensor_terms``.
    fraction : float or None, optional
        Overall Raman share in ``(0, 1]``, or ``None`` for absolute kernels.
    thg : bool, optional
        Retain the real-field harmonic drive. Default True; ignored for envelopes.
    species : str or None, optional
        Label selecting the amplitude density in a mixture.
    tensor_terms : tuple of RamanTensorTerm or None, optional
        Explicit labelled kernels, preserving each term's own dephasing.

    Attributes
    ----------
    tensor_terms : tuple of RamanTensorTerm
        The canonical immutable declaration. Supply this keyword or legacy
        ``models``, exactly one nonempty route. Legacy models become scalar-only
        terms. Each term retains its own dephasing and explicit tensor assignment.
    models : tuple of RamanModel
        Derived compatibility view of the declared models, in their original order.
    fraction : float or None, optional
        The Raman fraction ``f_R``, in ``(0, 1]``, multiplying the whole response, or
        ``None`` when the models are already absolute -- which every molecular model
        here is. It exists for a medium whose Raman strength is stated as a share of
        its total nonlinearity, as a fibre GNLSE does. Default ``None``.
    thg : bool, optional
        Whether a field grid keeps the third harmonic of the drive. Ignored on an
        envelope grid. Default ``True``.
    species : str or None, optional
        Component label whose density sets this response's amplitude.

    Raises
    ------
    MaterialError
        If there are no models, or the fraction is outside ``(0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.raman import response_for
    >>> from solprop.physdata import Gas, MaterialSelection
    >>> selection = MaterialSelection(carrier_wavelength=800e-9)
    >>> response = response_for(Gas.N2, selection=selection)
    >>> response.traits.domain
    'convolution'
    >>> len(response.models)
    72
    """

    tensor_terms: tuple[RamanTensorTerm, ...]
    fraction: float | None = None
    thg: bool = True
    species: str | None = None

    def __init__(
        self,
        models: tuple[RamanModel, ...] | None = None,
        fraction: float | None = None,
        thg: bool = True,
        species: str | None = None,
        *,
        tensor_terms: tuple[RamanTensorTerm, ...] | None = None,
    ) -> None:
        """Normalise exactly one constructor route into labelled immutable terms."""
        if (models is None) == (tensor_terms is None):
            raise MaterialError("supply exactly one of models or tensor_terms")
        terms = (
            tuple(RamanTensorTerm(model, "scalar_only") for model in models)
            if models is not None
            else tuple(tensor_terms or ())
        )
        if not terms:
            raise MaterialError("a Raman response needs at least one model")
        if not all(isinstance(term, RamanTensorTerm) for term in terms):
            raise MaterialError(
                "tensor_terms must contain RamanTensorTerm declarations"
            )
        if fraction is not None and not 0.0 < fraction <= 1.0:
            raise MaterialError(
                f"a Raman fraction is a share of the nonlinearity and must lie in "
                f"(0, 1], got {fraction}"
            )
        object.__setattr__(self, "tensor_terms", terms)
        object.__setattr__(self, "fraction", fraction)
        object.__setattr__(self, "thg", thg)
        object.__setattr__(self, "species", species)

    @property
    def models(self) -> tuple[RamanModel, ...]:
        """Return the scalar models in declaration order.

        Returns
        -------
        tuple of RamanModel
            A derived compatibility view; replace ``tensor_terms`` to update models.
        """
        return tuple(term.model for term in self.tensor_terms)

    @property
    def tensor_capable(self) -> bool:
        """Return whether every term has an established vector contraction.

        Returns
        -------
        bool
            False if any term is declared ``scalar_only``.
        """
        return all(term.contraction != "scalar_only" for term in self.tensor_terms)

    @property
    def scale(self) -> float:
        """Return the overall multiplier the response carries.

        Returns
        -------
        float
            :attr:`fraction`, or 1.0 when it is ``None``.
        """
        return 1.0 if self.fraction is None else self.fraction

    @property
    def traits(self) -> ResponseTraits:
        """Return the declared traits of the Raman response.

        Returns
        -------
        ResponseTraits
            A convolution on both grids consuming the density path. Tensor-capable
            declarations couple Cartesian components; other declarations are scalar.

        Examples
        --------
        >>> from solprop.nonlinear.raman import response_for
        >>> response_for("SiO2").traits.polarisation
        'scalar'
        """
        return ResponseTraits(
            domain="convolution",
            grids=("field", "envelope"),
            polarisation="tensor" if self.tensor_capable else "scalar",
            z_profiles=(
                "density" if self.species is None else f"density:{self.species}",
            ),
        )

    def impulse(
        self, t: Float[np.ndarray, " n_to"], density: float
    ) -> Float[np.ndarray, " n_to"]:
        """Return the response function ``h(t)`` at one density, in m^5 V^-2 s^-1.

        Each model is damped by its **own** dephasing before the sum, which is what a
        gas with both a rotational and a vibrational coherence requires: at one bar
        those two decay 3.5 times (H2) to 17 times (N2) apart.

        Parameters
        ----------
        t : numpy.ndarray
            Delays, s, shape ``(n_to,)``. Zero and negative delays give zero.
        density : float
            Number density in m^-3, non-negative.

        Returns
        -------
        numpy.ndarray
            ``h(t)``, m^5 V^-2 s^-1, shape ``(n_to,)``.

        Raises
        ------
        MaterialError
            If the density is negative.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.raman import response_for
        >>> from solprop.physdata import Gas, MaterialSelection
        >>> selection = MaterialSelection(carrier_wavelength=800e-9)
        >>> response = response_for(Gas.N2, selection=selection)
        >>> bool(np.all(response.impulse(np.array([-1e-15, 0.0]), 1e25) == 0.0))
        True
        """
        _check_density(density)
        times = _check_delays(t)
        total = np.zeros_like(times)
        for model in self.models:
            total += model.undamped(times) * model.dephasing.damping(times, density)
        return self.scale * total

    def tensor_impulse(
        self, t: Float[np.ndarray, " n_to"], density: float
    ) -> Float[np.ndarray, "2 n_to"]:
        """Return the isotropic and rank-two kernels in that order.

        These are the scalar linear-polarisation kernels ``h_v`` and ``h_r``,
        before multiplication by ``eps0 * density``. Damping and the declared
        fraction are included. Their sum equals :meth:`impulse` up to rounding.

        Parameters
        ----------
        t : numpy.ndarray
            Finite real delays in s, shape ``(n_to,)``; nonpositive delays give zero.
        density : float
            Finite non-negative number density in m^-3, setting the damping.

        Returns
        -------
        numpy.ndarray
            Shape ``(2, n_to)``, in m^5 V^-2 s^-1. Row zero is isotropic; row one
            is rank two. A channel with no declared terms is zero.

        Raises
        ------
        MaterialError
            If any term is scalar-only or the density is invalid.
        GridError
            If delays are not a finite real one-dimensional array.
        """
        _check_density(density)
        times = _check_delays(t)
        if not self.tensor_capable:
            raise MaterialError("scalar_only Raman terms have no tensor impulse")
        channels = np.zeros((2, times.size))
        for term in self.tensor_terms:
            row = 0 if term.contraction == "isotropic" else 1
            model = term.model
            channels[row] += model.undamped(times) * model.dephasing.damping(
                times, density
            )
        return self.scale * channels

    def evaluate(
        self,
        field_t: Num[np.ndarray, "n_pol n_to"],
        density: float,
        *,
        kind: fourier.GridKind,
        dt: float,
    ) -> Num[np.ndarray, "n_pol n_to"]:
        """Return the Raman polarisation on the fine time grid.

        The reference implementation, and therefore the oracle for every backend. It
        allocates, transforms twice on a grid of twice the length, and rebuilds the
        kernel on every call; :mod:`solprop.translate.raman` does none of those things.

        The convolution runs on a **doubled, zero-padded** grid. A circular convolution
        on the fine grid alone would wrap the tail of the response back onto the head
        of the drive, and for a gas whose rotational coherence outlives the window --
        nanoseconds against tens of picoseconds -- that wrapped tail is not small. The
        doubled grid makes the transform product the linear convolution exactly.
        The ``dt`` scaling is the sample spacing the discrete sum needs to become the
        integral; the inverse transform's ``1/N`` supplies the rest.

        Parameters
        ----------
        field_t : numpy.ndarray
            Finite local Cartesian field, shape ``(n_pol, n_to)``, with one or two
            components: real ``E(t)`` in V/m on a field grid, complex ``A(t)`` in
            V/m on an envelope grid. Two components require tensor-capable terms.
        density : float
            Number density in m^-3, non-negative. It both scales the polarisation and
            sets the dephasing.
        kind : {'field', 'envelope'}
            The spectral layout the samples belong to.
        dt : float
            Sample spacing of the fine time grid, s, positive.

        Returns
        -------
        numpy.ndarray
            The polarisation in C/m^2, shape ``(n_pol, n_to)``, real on a field grid and
            complex on an envelope grid.

        Raises
        ------
        GridError
            If the layout is unknown, the shape or dtype does not match it, the sample
            spacing is not positive, samples are non-finite, or a scalar-only term
            is evaluated on a vector field.
        MaterialError
            If the density is negative or non-finite.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.raman import response_for
        >>> from solprop.physdata import Gas, MaterialSelection
        >>> selection = MaterialSelection(carrier_wavelength=800e-9)
        >>> t = (np.arange(64) - 32) * 1e-15
        >>> field = (1e9 * np.exp(-((t / 5e-15) ** 2)))[None, :]
        >>> response = response_for(Gas.N2, selection=selection)
        >>> out = response.evaluate(field, 1e25, kind="field", dt=1e-15)
        >>> out.shape, out.dtype
        ((1, 64), dtype('float64'))
        """
        values = np.asarray(field_t)
        _check_evaluate(values, kind, dt)
        _check_density(density)
        if values.shape[0] == 2:
            if not self.tensor_capable:
                raise GridError("vector Raman requires tensor terms; found scalar_only")
            return self._evaluate_tensor(values, density, kind=kind, dt=dt)
        n_to = values.shape[1]
        kernel = np.zeros(2 * n_to)
        kernel[:n_to] = self.impulse(np.arange(n_to) * dt, density)
        drive = np.zeros(2 * n_to)
        drive[:n_to] = raman_drive(values, kind=kind, thg=self.thg)
        # The same order of operations as solprop.translate.raman.kernel_spectrum, so
        # that the two agree to the last bit and the parity test can say so.
        spectrum = np.fft.rfft(kernel) * dt
        convolved = np.fft.irfft(spectrum * np.fft.rfft(drive), n=2 * n_to)[:n_to]
        return (EPS0 * density) * values * convolved[None, :]

    def _evaluate_tensor(
        self,
        values: Num[np.ndarray, "2 n_to"],
        density: float,
        *,
        kind: fourier.GridKind,
        dt: float,
    ) -> Num[np.ndarray, "2 n_to"]:
        """Contract the molecular tensor directly, independently of backend layout.

        Parameters
        ----------
        values : numpy.ndarray
            Local Cartesian field in V/m, shape ``(2, n_to)``.
        density : float
            Non-negative number density in m^-3.
        kind : {'field', 'envelope'}
            Temporal representation.
        dt : float
            Positive temporal spacing in s.

        Returns
        -------
        numpy.ndarray
            Cartesian polarisation in C/m^2, shape ``(2, n_to)``.
        """
        quadratic = _quadratic_tensor(values, kind=kind, thg=self.thg)
        trace = quadratic[0, 0] + quadratic[1, 1]
        isotropic, rank2 = self.tensor_impulse(np.arange(values.shape[1]) * dt, density)
        identity = np.eye(2)[:, :, None]
        # The molecular orientation average is three-dimensional even though only
        # transverse field components are present. Its traceless factor stays 3.
        tensor = _causal_convolution(isotropic, trace, dt) * identity
        tensor += 0.5 * _causal_convolution(
            rank2, 3.0 * quadratic - trace * identity, dt
        )
        return (EPS0 * density) * np.einsum("ijt,jt->it", tensor, values)


SILICA_TENSOR_INTEGRALS: Final = (5.0 / 6.0, 1.0 / 6.0)
r"""What the two normalised **shapes** of :class:`SilicaTensorResponse` integrate to.

These are the shape integrals, not the kernel integrals: the kernels carry the
amplitude :data:`SILICA_TENSOR_AMPLITUDE` times ``f_R`` on top of them, so they
integrate to ``(3/2) f_R`` times the pair below. The pair itself sums to **one**,
which is what makes it a shape.

Lin and Agrawal write the delayed polarisation as an isotropic term in
:math:`a(\tau)` and a dyadic term in :math:`b(\tau)`, with
:math:`a = w_a h_a` and :math:`b = w_b h_b + w_{ab} h_a`
(Opt. Lett. **31**, 3086 (2006), Eq. (1); the vector theory is J. Opt. Soc. Am.
B **20**, 1616 (2003)). The record the seam carries instead splits the same
polarisation into a **trace** channel and the three-dimensional **traceless
rank-two** channel ``T(Q) = (3Q - tr(Q) I)/2``, which is the contraction the
existing vector Raman route already uses. Matching

``a tr(Q) E + b Q E  =  r_trace tr(Q) E + r_rank T(Q) E``

gives ``r_trace = a + b/3`` and ``r_rank = 2 b/3``. Because ``h_a`` and ``h_b`` are
each normalised, the two integrate to

``r_trace = w_a + (w_b + w_ab)/3``  and  ``r_rank = 2 (w_b + w_ab)/3`` ,

whose sum is ``w_a + w_b + w_ab = 1``. The values stored here, **5/6** and **1/6**,
are what those expressions give for Lin and Agrawal's (2006) weights --
``w_a = 0.75`` and ``w_b + w_ab = 0.25`` -- and for no others: a different ``w_a``
moves both numbers. They are not the paper's own ``int a = 3/4`` and
``int b = 1/4``, which belong to a different pair of channels.
"""

SILICA_TENSOR_AMPLITUDE: Final = 1.5
r"""The field-domain amplitude factor of :class:`SilicaTensorResponse`'s kernels.

``f_R`` and ``n2`` are **envelope** (Agrawal) quantities: ``f_R`` is the share of
the *effective* nonlinearity on the fundamental that the delayed term carries. The
kernels here act in the **field** domain, where the delayed polarisation
``eps0 chi3 E int h(tau) E^2(t - tau) dtau`` is driven by the cycle average
``|A|^2/2`` and so contributes only ``(1/2) (int h) |A|^2 A`` on the fundamental,
against the instantaneous cubic's ``(3/4) chi3 |A|^2 A``. A field-domain kernel of
integral ``X`` therefore acts like a Kerr term of ``(2/3) X``, and carrying the
share ``f_R`` of one ``chi3`` needs

``int h = (3/2) f_R chi3`` .

This is exactly :func:`solprop.nonlinear.gnlse.raman_susceptibility`'s
``2 n2 eps0 c = (3/2) chi3``, the identity
``(2/3) f_R X + kerr_susceptibility(n2, f_R) == kerr_susceptibility(n2)`` it is
built to satisfy: the ``(3/2) chi3_total`` amplitude mapping, with the fraction
applied exactly once. Without it the declared ``n2``
comes out ``f_R/3`` -- 8.2 % for silica -- low.
"""


@dataclass(frozen=True)
class SilicaTensorResponse(Displayed):
    r"""The delayed tensor Raman response of fused silica (Lin--Agrawal).

    A **material region's** declared delayed response, beside
    :class:`solprop.modes.material.MaterialKerr`: the two share one
    :math:`\chi^{(3)}`, of which this one takes the fraction :math:`f_R` its
    parameters declare and the instantaneous Kerr term keeps :math:`1 - f_R`. It is
    a separately named model and changes nothing about the existing scalar silica
    response of :func:`response_for`, which is a different fit
    (Hollenbeck--Cantrell) reached by a different route (a gas-filled or
    mode-averaged run's response list).

    **The polarisation.** For the real three-component field
    :math:`\mathbf E(t)`, Lin and Agrawal's delayed terms are

    .. math::

        P_i^{\rm R}(t) = \varepsilon_0\chi^{(3)}\Big[
          E_i(t)\!\int_0^\infty\! a(\tau)\,
            \mathbf E(t-\tau)\!\cdot\!\mathbf E(t-\tau)\,\mathrm d\tau
          + \sum_j E_j(t)\!\int_0^\infty\! b(\tau)\,
            E_i(t-\tau)E_j(t-\tau)\,\mathrm d\tau\Big],

    and this class carries them as the two channels
    :data:`SILICA_TENSOR_INTEGRALS` names, over the **six** real symmetric pair
    histories :math:`Q_{kl} = E_kE_l` of three components:

    ``P_i = eps0 chi3 sum_term sum_jkl D[term]_ijkl E_j (h[term] * Q)_kl`` ,

    with ``D[trace]_ijkl = d_ij d_kl`` and
    ``D[rank]_ijkl = (3 (d_ik d_jl + d_il d_jk) - 2 d_ij d_kl)/4``, the second being
    ``T(Q)`` written as a contraction and symmetrised in its last two indices
    because a pair history is symmetric. **The scalar limit is exact**: for one
    fixed linear polarisation the two channels add to
    ``E(t) int (3/2) f_R h_R(tau) E^2(t - tau) dtau`` with ``h_R = a + b``
    integrating to one -- the standard fibre model this package's scalar routes
    carry, at the field-domain amplitude :data:`SILICA_TENSOR_AMPLITUDE` explains,
    which is what makes the declared ``n2`` come out right on the fundamental.

    On an **envelope** grid, and on a field grid with the analytic-signal cubic, the
    drive is the beat-free ``Re(E_a,k conj(E_a,l))/2`` instead, exactly as
    :func:`raman_drive` chooses it for the scalar response.

    **From optmat.** With `parameters` an optmat tensor bundle
    (:class:`~solprop.physdata.glasses.SilicaBundle`, ``SilicaBundle.resolve(
    tensor=True)``), Hellwarth's :math:`a` and :math:`b` are optmat's own
    ``h_r(t, "a")`` and ``h_r(t, "b")``, normalised so that :math:`\int(a+b)=1`, and
    the fraction is the bundle's (0.245 for Lin and Agrawal's shape): the same two
    channels, :math:`a + b/3` and :math:`2b/3`, and the same endpoint rule, with the
    right-hand limits at zero delay taken from optmat. A copolarised-only bundle
    (Hollenbeck and Cantrell's, the scalar default) has no :math:`a` and :math:`b`
    and is refused by name.

    Attributes
    ----------
    parameters : solprop.physdata.raman_types.LinAgrawalRaman or SilicaBundle
        An optmat silica bundle with a tensor shape, by default optmat's
        recommended tensor bundle (``SilicaBundle.resolve(tensor=True)``, Lin and
        Agrawal's shape, SEP 3, D11), or a caller's own time constants, weights and
        Raman fraction.

    Raises
    ------
    MaterialError
        If an optmat bundle has no tensor parts.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.raman import SilicaTensorResponse
    >>> response = SilicaTensorResponse()
    >>> response.fraction
    0.245
    >>> t = np.arange(20000) * 1e-16
    >>> integrals = response.kernels(t).sum(axis=1) * 1e-16 / response.fraction
    >>> [float(round(one, 4)) for one in integrals]
    [1.25, 0.25]
    """

    parameters: LinAgrawalRaman | SilicaBundle = field(default_factory=_tensor_bundle)

    def __post_init__(self) -> None:
        """Refuse an optmat bundle whose shape has no tensor parts, by name.

        Raises
        ------
        MaterialError
            If the bundle is copolarised only.
        """
        bundle = self.parameters
        if isinstance(bundle, SilicaBundle) and not bundle.has_tensor:
            msg = (
                f"{bundle.name} ({bundle.shape}) is copolarised only: it has no "
                "isotropic and anisotropic parts for a tensor response. Use the "
                "tensor bundle, SilicaBundle.resolve(tensor=True)"
            )
            raise MaterialError(msg)

    def __repr__(self) -> str:
        """Return a one-line description: the fraction and the three time constants.

        Returns
        -------
        str
            For example ``SilicaTensorResponse(f_R=0.245, 12.2 fs/32 fs/96 fs)``,
            or the optmat bundle's name in place of the time constants.
        """
        parameters = self.parameters
        if isinstance(parameters, SilicaBundle):
            return line(
                "SilicaTensorResponse",
                [("", f"f_R={self.fraction:.4g}"), ("", parameters.name)],
            )
        times = "/".join(
            engineering(one, "s")
            for one in (parameters.tau1, parameters.tau2, parameters.tau_b)
        )
        return line(
            "SilicaTensorResponse",
            [("", f"f_R={self.fraction:.4g}"), ("", times)],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        parameters = self.parameters
        if isinstance(parameters, SilicaBundle):
            return "SilicaTensorResponse: the delayed tensor Raman of fused silica", [
                ("fraction", f"{self.fraction:.4g} of chi3"),
                ("instantaneous share", f"{self.instantaneous_share:.4g} of chi3"),
                ("bundle", f"{parameters.name} ({parameters.shape})"),
                ("digest", parameters.digest),
                (
                    "kernel amplitude",
                    f"(3/2) f_R = {SILICA_TENSOR_AMPLITUDE * self.fraction:.4g} of "
                    "chi3",
                ),
            ]
        weights = (
            f"{parameters.isotropic_weight:.4g} h_a, "
            f"{parameters.boson_weight:.4g} h_b, "
            f"{parameters.anisotropic_weight:.4g} h_a (anisotropic)"
        )
        # Computed from the declared weights rather than printed as 5/6 and 1/6,
        # which they are only while w_a is 0.75 (SILICA_TENSOR_INTEGRALS).
        shape_trace, shape_rank = self._channels(
            *self._hellwarth_parts(np.ones(1), np.ones(1))
        )[:, 0]
        return "SilicaTensorResponse: the delayed tensor Raman of fused silica", [
            ("fraction", f"{self.fraction:.4g} of chi3"),
            ("instantaneous share", f"{self.instantaneous_share:.4g} of chi3"),
            (
                "tau1, tau2",
                f"{engineering(parameters.tau1, 's')}, "
                f"{engineering(parameters.tau2, 's')}",
            ),
            ("tau_b", engineering(parameters.tau_b, "s")),
            ("weights", weights),
            (
                "shape integrals",
                f"{float(shape_trace):.4g} trace, {float(shape_rank):.4g} rank two",
            ),
            (
                "kernel amplitude",
                f"(3/2) f_R = {SILICA_TENSOR_AMPLITUDE * self.fraction:.4g} of chi3",
            ),
            ("source", parameters.citation),
        ]

    @property
    def fraction(self) -> float:
        """Return the Raman fraction ``f_R`` the parameters declare.

        Returns
        -------
        float
            A share of the medium's third-order susceptibility, in ``(0, 1]``: an
            optmat bundle's own ``f_R``.
        """
        parameters = self.parameters
        if isinstance(parameters, SilicaBundle):
            return parameters.raman_fraction
        return float(parameters.fraction)

    @property
    def instantaneous_share(self) -> float:
        """Return ``1 - f_R``, the share the instantaneous Kerr term keeps.

        Returns
        -------
        float
            A dimensionless share in ``[0, 1)``.
        """
        return 1.0 - self.fraction

    @property
    def traits(self) -> ResponseTraits:
        """Return the declared traits of this response.

        Returns
        -------
        ResponseTraits
            A convolution on a field grid, coupling Cartesian components, with no
            density path at all: a glass has none, and the magnitude comes from
            the region's own ``chi3``.

        Examples
        --------
        >>> from solprop.nonlinear.raman import SilicaTensorResponse
        >>> SilicaTensorResponse().traits.polarisation
        'tensor'
        """
        return ResponseTraits(
            domain="convolution",
            grids=("field",),
            polarisation="tensor",
            z_profiles=(),
        )

    def shapes(
        self, t: Float[np.ndarray, " n_to"]
    ) -> Float[np.ndarray, "2 n_to_shapes"]:
        r"""Return the two normalised response functions ``h_a`` and ``h_b``.

        Each integrates to one over the positive axis; both are zero at and below
        zero delay, which is what makes the later convolution causal.

        Parameters
        ----------
        t : numpy.ndarray
            Delays in s, shape ``(n_to,)``, finite and real.

        Returns
        -------
        numpy.ndarray
            Shape ``(2, n_to)`` in 1/s: row zero ``h_a``, row one ``h_b``.

        Raises
        ------
        GridError
            If the delays are not a finite real one-dimensional array.
        MaterialError
            For an optmat bundle, which gives Hellwarth's ``a`` and ``b`` rather
            than ``h_a`` and ``h_b`` (:meth:`hellwarth`).

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.raman import SilicaTensorResponse
        >>> from solprop.physdata import LinAgrawalRaman
        >>> declared = LinAgrawalRaman(
        ...     tau1=12.2e-15,
        ...     tau2=32.0e-15,
        ...     tau_b=96.0e-15,
        ...     isotropic_weight=0.75,
        ...     boson_weight=0.21,
        ...     anisotropic_weight=0.04,
        ...     fraction=0.245,
        ...     citation="Lin and Agrawal, Opt. Lett. 31, 3086 (2006)",
        ... )
        >>> response = SilicaTensorResponse(declared)
        >>> shapes = response.shapes(np.array([-1e-15, 0.0, 1e-14]))
        >>> [float(round(one, 4)) for one in shapes[:, 0]]
        [0.0, 0.0]
        """
        times = _check_delays(t)
        parameters = self.parameters
        if isinstance(parameters, SilicaBundle):
            msg = (
                f"{parameters.name} gives Hellwarth's a and b, not Lin and Agrawal's "
                "h_a and h_b; use SilicaTensorResponse.hellwarth"
            )
            raise MaterialError(msg)
        tau1, tau2, tau_b = parameters.tau1, parameters.tau2, parameters.tau_b
        positive = times > 0.0
        delay = np.where(positive, times, 0.0)
        vibration = (
            (tau1**2 + tau2**2)
            / (tau1 * tau2**2)
            * np.exp(-delay / tau2)
            * np.sin(delay / tau1)
        )
        boson = (2.0 * tau_b - delay) / tau_b**2 * np.exp(-delay / tau_b)
        return np.stack(
            [np.where(positive, vibration, 0.0), np.where(positive, boson, 0.0)]
        )

    def _hellwarth_parts(
        self,
        shape_a: Float[np.ndarray, " n_channel_samples"],
        shape_b: Float[np.ndarray, " n_channel_samples"],
    ) -> tuple[
        Float[np.ndarray, " n_channel_samples"], Float[np.ndarray, " n_channel_samples"]
    ]:
        """Weight Lin and Agrawal's two shapes into Hellwarth's ``a`` and ``b``.

        Parameters
        ----------
        shape_a : numpy.ndarray
            ``h_a``, in 1/s, or a one-element array of its one-sided limit.
        shape_b : numpy.ndarray
            ``h_b``, on the same samples.

        Returns
        -------
        tuple of numpy.ndarray
            ``a = w_a h_a`` and ``b = w_b h_b + w_ab h_a``, in the inputs' units.
        """
        parameters = cast("LinAgrawalRaman", self.parameters)
        isotropic = parameters.isotropic_weight * shape_a
        anisotropic = (
            parameters.boson_weight * shape_b + parameters.anisotropic_weight * shape_a
        )
        return isotropic, anisotropic

    def hellwarth(
        self, t: Float[np.ndarray, " n_to"]
    ) -> Float[np.ndarray, "2 n_to_hellwarth"]:
        r"""Return Hellwarth's isotropic ``a`` and anisotropic ``b``, in 1/s.

        ``a + b`` is the copolarised response :math:`h_R` and integrates to one;
        both are zero at and below zero delay. From the declared parameters,
        ``a = w_a h_a`` and ``b = w_b h_b + w_ab h_a``; from an optmat bundle,
        optmat's ``h_r(t, "a")`` and ``h_r(t, "b")``.

        Parameters
        ----------
        t : numpy.ndarray
            Delays in s, shape ``(n_to,)``, finite and real.

        Returns
        -------
        numpy.ndarray
            Shape ``(2, n_to)`` in 1/s: row zero ``a``, row one ``b``.

        Raises
        ------
        GridError
            If the delays are not a finite real one-dimensional array.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.raman import SilicaTensorResponse
        >>> parts = SilicaTensorResponse().hellwarth(np.arange(20000) * 1e-16)
        >>> float(round(parts.sum() * 1e-16, 3))
        1.0
        """
        parameters = self.parameters
        if isinstance(parameters, SilicaBundle):
            times = _check_delays(t)
            return np.stack([parameters.h_r(times, "a"), parameters.h_r(times, "b")])
        return np.stack(self._hellwarth_parts(*self.shapes(t)))

    def _limits(self) -> Float[np.ndarray, "2 1"]:
        """Return the right-hand limits of ``a`` and ``b`` at zero delay.

        Returns
        -------
        numpy.ndarray
            Shape ``(2, 1)`` in 1/s.
        """
        parameters = self.parameters
        if isinstance(parameters, SilicaBundle):
            return np.stack(
                [
                    _right_limit(lambda t: parameters.h_r(t, "a")),
                    _right_limit(lambda t: parameters.h_r(t, "b")),
                ]
            )
        # h_a(0+) = 0 because it opens with sin(t/tau1); h_b(0+) = 2/tau_b.
        return np.stack(
            self._hellwarth_parts(np.zeros(1), np.full(1, 2.0 / parameters.tau_b))
        )

    @staticmethod
    def _channels(
        hellwarth_a: Float[np.ndarray, " n_channel_samples"],
        hellwarth_b: Float[np.ndarray, " n_channel_samples"],
    ) -> Float[np.ndarray, "2 n_channel_samples"]:
        """Combine Hellwarth's ``a`` and ``b`` into the trace and rank-two channels.

        Parameters
        ----------
        hellwarth_a : numpy.ndarray
            ``a``, in 1/s, or a one-element array of its one-sided limit.
        hellwarth_b : numpy.ndarray
            ``b``, on the same samples.

        Returns
        -------
        numpy.ndarray
            ``a + b/3`` and ``2 b/3`` stacked along a leading axis, in the inputs'
            units.
        """
        return np.stack([hellwarth_a + hellwarth_b / 3.0, 2.0 * hellwarth_b / 3.0])

    def kernels(
        self, t: Float[np.ndarray, " n_to"]
    ) -> Float[np.ndarray, "2 n_to_kernels"]:
        r"""Return the trace and rank-two seam kernels, in 1/s.

        ``(3/2) f_R (a + b/3)`` and ``(3/2) f_R (2 b/3)`` of the class docstring,
        which integrate to :data:`SILICA_TENSOR_AMPLITUDE` times ``f_R`` times
        :data:`SILICA_TENSOR_INTEGRALS`. Both the amplitude factor and the Raman
        fraction are included **here and nowhere else**, so that a compiled record
        multiplies by ``chi3`` alone and the fraction is applied exactly once; this
        is also the one place the sampled kernel is built, so the endpoint rule
        below is seen by every caller.

        **The endpoint quadrature is the trapezoid**, and the sample at zero delay
        carries **half** the kernel's right-hand limit there. ``h_b(0^+) =
        2/\tau_b`` is not zero, so a causal kernel sampled on ``arange(n) dt`` and
        integrated by the rectangle-rule transform product misses ``dt h(0^+)/2``
        and converges only to first order: sampling the jump at its midpoint --
        the mean of the two one-sided limits, one of which is zero -- turns that
        sum into the trapezoid rule and restores second order. The half-weighted
        sample multiplies ``Q(t)`` itself, so it costs no causality. The late lobe
        of ``h_b``, where ``t > 2\tau_b``, is genuinely negative and is left
        unclipped: it is part of the response, not a gain.

        Parameters
        ----------
        t : numpy.ndarray
            Delays in s, shape ``(n_to,)``, finite and real. The endpoint rule
            assumes a uniform axis starting at zero, which is what both callers
            pass.

        Returns
        -------
        numpy.ndarray
            Shape ``(2, n_to)`` in 1/s: row zero the trace channel, row one the
            traceless rank-two channel.

        Raises
        ------
        GridError
            If the delays are not a finite real one-dimensional array.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.raman import SilicaTensorResponse
        >>> kernels = SilicaTensorResponse().kernels(np.arange(8) * 1e-15)
        >>> kernels.shape
        (2, 8)
        """
        channels = self._channels(*self.hellwarth(t))
        at_zero = np.asarray(t, dtype=float) == 0.0
        if bool(at_zero.any()):
            channels[:, at_zero] = 0.5 * self._channels(*self._limits())
        return (SILICA_TENSOR_AMPLITUDE * self.fraction) * channels

    def tensors(self) -> Float[np.ndarray, "2 3 3 3 3"]:
        r"""Return the two dimensionless rank-four channel tensors.

        ``D[trace]_ijkl = d_ij d_kl`` and
        ``D[rank]_ijkl = (3 (d_ik d_jl + d_il d_jk) - 2 d_ij d_kl)/4``, contracted
        as ``P_i = ... D_ijkl E_j H_kl`` with ``H`` the convolved pair history.
        Both are symmetric in their **last two** indices, because a pair history
        is, and that symmetry is what makes the six stored histories enough.

        Returns
        -------
        numpy.ndarray
            Shape ``(2, 3, 3, 3, 3)``, dimensionless.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.raman import SilicaTensorResponse
        >>> tensors = SilicaTensorResponse().tensors()
        >>> [float(one[0, 0, 0, 0]) for one in tensors]
        [1.0, 1.0]
        """
        delta = np.eye(3)
        trace = np.einsum("ij,kl->ijkl", delta, delta)
        rank = (
            3.0
            * (
                np.einsum("ik,jl->ijkl", delta, delta)
                + np.einsum("il,jk->ijkl", delta, delta)
            )
            - 2.0 * trace
        ) / 4.0
        return np.stack([trace, rank])

    def evaluate(
        self,
        field_t: Num[np.ndarray, "n_pol n_to"],
        chi3: float,
        *,
        kind: fourier.GridKind,
        dt: float,
        thg: bool = True,
    ) -> Num[np.ndarray, "n_pol n_to"]:
        r"""Return the delayed polarisation on the fine time grid, in C/m^2.

        The reference implementation, and therefore the oracle a compiled route is
        measured against. Like :meth:`RamanResponse.evaluate` it convolves on a
        **doubled, zero-padded** grid, so the transform product is the linear
        convolution exactly and the response's tail cannot wrap onto the head of
        the drive.

        Parameters
        ----------
        field_t : numpy.ndarray
            The local Cartesian field on the fine time grid, shape
            ``(n_pol, n_to)`` with one, two or three components: real ``E(t)`` in
            V/m on a field grid, complex ``A(t)`` in V/m on an envelope grid.
        chi3 : float
            The region's absolute third-order susceptibility in m^2/V^2,
            non-negative. The Raman fraction is already in the kernels.
        kind : {'field', 'envelope'}
            The temporal representation the samples belong to.
        dt : float
            Sample spacing of the fine time grid, s, positive.
        thg : bool, optional
            On a field grid, whether the drive keeps the ``2 omega`` beat --
            ``E_kE_l``, which radiates the third harmonic through the later
            multiplication by ``E`` -- or is the beat-free
            ``Re(E_a,k conj(E_a,l))/2``. Ignored on an envelope grid, which cannot
            carry the beat.

        Returns
        -------
        numpy.ndarray
            The polarisation in C/m^2, shape ``(n_pol, n_to)``, real on a field
            grid and complex on an envelope grid.

        Raises
        ------
        GridError
            If the layout is unknown, the shape or dtype does not match it, the
            spacing is not positive, or samples are non-finite.
        MaterialError
            If the susceptibility is negative or not finite.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.raman import SilicaTensorResponse
        >>> t = (np.arange(64) - 32) * 1e-15
        >>> field = np.stack([1e9 * np.exp(-((t / 5e-15) ** 2)), np.zeros(64),
        ...                   np.zeros(64)])
        >>> out = SilicaTensorResponse().evaluate(
        ...     field, 2.2e-22, kind="field", dt=1e-15
        ... )
        >>> out.shape, out.dtype
        ((3, 64), dtype('float64'))
        """
        values = np.asarray(field_t)
        _check_evaluate(values, kind, dt, max_components=3)
        if not math.isfinite(chi3) or chi3 < 0.0:
            raise MaterialError(
                f"a silica tensor response needs a finite, non-negative chi3, got "
                f"{chi3} m^2/V^2"
            )
        drive = _quadratic_tensor(values, kind=kind, thg=thg)
        kernels = self.kernels(np.arange(values.shape[1]) * dt)
        components = int(values.shape[0])
        tensors = self.tensors()[:, :components, :components, :components, :components]
        polarisation = np.zeros(values.shape, dtype=values.dtype)
        for kernel, tensor in zip(kernels, tensors, strict=True):
            history = _causal_convolution(kernel, drive, dt)
            polarisation = polarisation + np.einsum(
                "ijkl,klt,jt->it", tensor, history, values
            )
        return (EPS0 * chi3) * polarisation


def _optmat_response(
    gas: str,
    options: MolecularOptions | None,
    selection: MaterialSelection,
    fraction: float | None,
    thg: bool,
    density: float | None,
) -> RamanResponse:
    """Build a gas's Raman response from optmat for :func:`response_for`.

    Parameters
    ----------
    gas : str
        The species' name.
    options : MolecularOptions or None
        Term flags and temperature.
    selection : MaterialSelection
        The selection.
    fraction : float or None
        Refused: the response is absolute.
    thg : bool
        Whether a field grid keeps the third harmonic.
    density : float or None
        The fill's constant density in m^-3.

    Returns
    -------
    RamanResponse
        The response from optmat, on solprop's terms.

    Raises
    ------
    MaterialError
        With a fraction, or as optmat's adapter refuses.
    """
    from solprop.nonlinear.optmat_raman import gas_raman  # noqa: PLC0415 - a cycle

    if fraction is not None:
        msg = (
            f"the optmat Raman response of {gas} is absolute; a fraction would scale it"
        )
        raise MaterialError(msg)
    return gas_raman(gas, options, selection=selection, thg=thg, density=density)


def _right_limit(
    function: Callable[[Float[np.ndarray, " n_limit"]], Float[np.ndarray, " n_limit"]],
) -> Float[np.ndarray, " n_limit"]:
    """Return a causal response's limit from the right at zero delay.

    optmat's responses are zero at ``t = 0`` itself. The limit is extrapolated from
    the two smallest normal delays, ``2 h(tiny) - h(2 tiny)``: a term that opens
    linearly, ``sin(t / tau)``, doubles exactly between them (a power-of-two scaling
    rounds alike), so it cancels to an exact zero, and a jump, the boson peak's
    ``2 / tau_b``, is the same at both and survives.

    Parameters
    ----------
    function : callable
        The response of a delay array in s, in 1/s.

    Returns
    -------
    numpy.ndarray
        The limit, shape ``(1,)``.
    """
    tiny = np.finfo(float).tiny
    return 2.0 * function(np.full(1, tiny)) - function(np.full(1, 2.0 * tiny))


@dataclass(frozen=True)
class SampledShape:
    r"""A density-independent Raman term sampled from optmat's silica bundle.

    The undamped row is the bundle's normalised shape times `amplitude`, and the
    dephasing is none: a glass's widths are already in its shape, and it has no
    density to scale (SEP 3, section 4.5). The channels:

    - ``"parallel"``: the copolarised :math:`h_R = a + b`, unit integral, for a
      scalar run (a ``scalar_only`` term);
    - ``"isotropic"``: :math:`h_0 = a + b/3`, the trace channel;
    - ``"rank2"``: :math:`h_2 = 2b/3`, the traceless rank-two channel,

    with Hellwarth's :math:`a` and :math:`b` from ``h_r(t, "a")`` and
    ``h_r(t, "b")``. With the seam's weights, ``(1, 0)`` for the isotropic term and
    ``(1/4, 3/4)`` for the rank-two one, a copolarised probe then sees
    :math:`a + b` and a cross-polarised one :math:`a`.

    **At zero delay** the sample is half the shape's right-hand limit, the trapezoid
    rule's endpoint weight, and zero below it. That is exactly zero for a shape that
    opens with a sine (Hollenbeck and Cantrell's, and :math:`a`); for the boson peak
    in :math:`b`, whose limit is :math:`2 f_b/\tau_b`, it keeps the convolution
    second order where a zero sample would leave it first order, at no cost to
    causality, as :meth:`SilicaTensorResponse.kernels` explains.

    Attributes
    ----------
    bundle : SilicaBundle
        The optmat silica bundle.
    channel : {'parallel', 'isotropic', 'rank2'}, optional
        The part sampled. Default ``"parallel"``.
    amplitude : float, optional
        The value ``integral h dt`` takes for the ``"parallel"`` channel, m^5/V^2
        per unit density. Default 1.0, a pure shape.

    Raises
    ------
    MaterialError
        If the channel is unknown, the amplitude is not finite, or a tensor
        channel is asked of a copolarised-only bundle, naming it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.raman import SampledShape
    >>> from solprop.physdata.glasses import SilicaBundle
    >>> shape = SampledShape(SilicaBundle.resolve())
    >>> t = np.arange(40000) * 5e-17
    >>> f"{shape.undamped(t).sum() * 5e-17:.4f}"
    '1.0000'
    """

    bundle: SilicaBundle
    channel: Literal["parallel", "isotropic", "rank2"] = "parallel"
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        """Validate the channel and the amplitude.

        Raises
        ------
        MaterialError
            If the channel is unknown or needs a tensor the bundle lacks, or the
            amplitude is not finite.
        """
        if self.channel not in ("parallel", "isotropic", "rank2"):
            msg = (
                f"unknown silica Raman channel {self.channel!r}; one of 'parallel', "
                "'isotropic' and 'rank2'"
            )
            raise MaterialError(msg)
        if not math.isfinite(self.amplitude):
            msg = f"a Raman amplitude must be finite, got {self.amplitude!r}"
            raise MaterialError(msg)
        if self.channel != "parallel" and not self.bundle.has_tensor:
            msg = (
                f"{self.bundle.name} ({self.bundle.shape}) is copolarised only: it "
                f"has no {self.channel} channel. A vector run needs the tensor "
                "bundle, SilicaBundle.resolve(tensor=True)"
            )
            raise MaterialError(msg)

    @property
    def name(self) -> str:
        """Return the term's name: the bundle's and the channel.

        Returns
        -------
        str
            For example ``"silica_bundle/default:parallel"``.
        """
        return f"{self.bundle.name}:{self.channel}"

    @property
    def material_records(self) -> tuple[MaterialRecord, ...]:
        """Return the records of the bundle the term samples.

        Returns
        -------
        tuple of MaterialRecord
            The bundle's, its Kerr fit's and its shape's.
        """
        return self.bundle.material_records

    @property
    def dephasing(self) -> Dephasing:
        """Return the density-dependent dephasing, which for a glass is none.

        Returns
        -------
        Dephasing
            A dephasing whose factor is one everywhere.
        """
        return undamped_dephasing()

    def shape(self, t: Float[np.ndarray, " n_to"]) -> Float[np.ndarray, " n_to"]:
        """Return the channel's normalised shape, in 1/s, zero at and below zero delay.

        Parameters
        ----------
        t : numpy.ndarray
            Delays in s, shape ``(n_to,)``.

        Returns
        -------
        numpy.ndarray
            ``a + b``, ``a + b/3`` or ``2 b/3``, shape ``(n_to,)``.
        """
        times = np.asarray(t, dtype=float)
        if self.channel == "parallel":
            return self.bundle.h_r(times, "parallel")
        part_b = self.bundle.h_r(times, "b")
        if self.channel == "rank2":
            return 2.0 * part_b / 3.0
        return self.bundle.h_r(times, "a") + part_b / 3.0

    def undamped(self, t: Float[np.ndarray, " n_to"]) -> Float[np.ndarray, " n_to"]:
        """Return the sampled term, in m^5 V^-2 s^-1.

        Parameters
        ----------
        t : numpy.ndarray
            Delays in s, shape ``(n_to,)``.

        Returns
        -------
        numpy.ndarray
            ``amplitude`` times the shape where ``t > 0``, half its right-hand limit
            at ``t = 0`` and zero below, shape ``(n_to,)``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.raman import SampledShape
        >>> from solprop.physdata.glasses import SilicaBundle
        >>> shape = SampledShape(SilicaBundle.resolve())
        >>> shape.undamped(np.array([-1e-15, 0.0])).tolist()
        [0.0, 0.0]
        """
        times = np.asarray(t, dtype=float)
        values = self.shape(times)
        at_zero = times == 0.0
        if bool(at_zero.any()):
            limit = float(_right_limit(self.shape)[0])
            values = np.where(at_zero, 0.5 * limit, values)
        return self.amplitude * np.where(times >= 0.0, values, 0.0)


def bundle_response(
    bundle: SilicaBundle,
    *,
    amplitude: float = 1.0,
    fraction: float | None = None,
    thg: bool = True,
    tensor: bool = False,
) -> RamanResponse:
    """Build fused silica's Raman response from an optmat bundle.

    The scalar response is one ``scalar_only`` :class:`SampledShape` of the bundle's
    copolarised shape. The tensor response is the bundle's isotropic channel
    ``a + b/3`` and rank-two channel ``2 b/3`` as two :class:`RamanTensorTerm`
    declarations, which the seam weights ``(1, 0)`` and ``(1/4, 3/4)``.

    Parameters
    ----------
    bundle : SilicaBundle
        The optmat silica bundle; a tensor response needs one with tensor parts.
    amplitude : float, optional
        The value ``integral h dt`` of the copolarised response, m^5/V^2 per unit
        density, before `fraction`. Default 1.0.
    fraction : float or None, optional
        The Raman fraction in ``(0, 1]`` the response carries, or ``None``.
    thg : bool, optional
        Whether a field grid keeps the third harmonic of the drive. Default ``True``.
    tensor : bool, optional
        Build the tensor declaration, for a vector run. Default ``False``.

    Returns
    -------
    RamanResponse
        One scalar-only term, or an isotropic and a rank-two term.

    Raises
    ------
    MaterialError
        If a tensor response is asked of a copolarised-only bundle, naming it, or
        the fraction is outside ``(0, 1]``.

    Examples
    --------
    >>> from solprop.nonlinear.raman import bundle_response
    >>> from solprop.physdata.glasses import SilicaBundle
    >>> bundle_response(SilicaBundle.resolve()).tensor_capable
    False
    >>> response = bundle_response(SilicaBundle.resolve(tensor=True), tensor=True)
    >>> [term.contraction for term in response.tensor_terms]
    ['isotropic', 'rank2']
    """
    if not tensor:
        return RamanResponse(
            models=(SampledShape(bundle, "parallel", amplitude),),
            fraction=fraction,
            thg=thg,
        )
    if not bundle.has_tensor:
        msg = (
            f"a vector run needs a tensor Raman response, and {bundle.name} "
            f"({bundle.shape}) is copolarised only: it has no isotropic and "
            "anisotropic parts. Use the tensor bundle, "
            "SilicaBundle.resolve(tensor=True)"
        )
        raise MaterialError(msg)
    return RamanResponse(
        tensor_terms=(
            RamanTensorTerm(SampledShape(bundle, "isotropic", amplitude), "isotropic"),
            RamanTensorTerm(SampledShape(bundle, "rank2", amplitude), "rank2"),
        ),
        fraction=fraction,
        thg=thg,
    )


SILICA_NAMES: Final = ("sio2", "silica", "fused_silica", "fused silica")
"""Spellings :func:`response_for` accepts for fused silica."""


def _molecular_models(
    parameters: MolecularRaman, options: MolecularOptions
) -> tuple[RamanTensorTerm, ...]:
    """Build the requested terms of a molecular gas's Raman response.

    Parameters
    ----------
    parameters : MolecularRaman
        The species' available molecular data.
    options : MolecularOptions
        Which parts to build, and the thermodynamic state; a term flag of ``None``
        takes the term when the species has it, ``True`` requires it.

    Returns
    -------
    tuple of RamanTensorTerm
        The rotational ladder first, then the vibrational mode; either may be absent.

    Raises
    ------
    MaterialError
        If a required term has no model, both parts are switched off, or neither
        exists for this gas.
    """
    for selected, data, name in (
        (options.rotation, parameters.rotational, "rotational"),
        (options.vibration, parameters.vibrational, "vibrational"),
    ):
        if selected is True and data is None:
            raise MaterialError(
                f"{parameters.gas} has no Raman response for the selected {name} "
                f"term: {parameters.missing or 'no model is available'}"
            )
    temperature = (
        ROOM_TEMPERATURE if options.temperature is None else options.temperature
    )
    models: list[RamanTensorTerm] = []
    if options.rotation is not False and parameters.rotational is not None:
        models.append(
            RamanTensorTerm(
                rotational_nonrigid(
                    parameters.rotational,
                    temperature,
                    min_j=options.min_j,
                    max_j=options.max_j,
                ),
                "rank2",
            )
        )
    if options.vibration is not False and parameters.vibrational is not None:
        # The retained band is the mean vibrational response. This does not assert
        # that a diatomic molecule's full vibrational polarizability is isotropic.
        models.append(RamanTensorTerm(vibrational(parameters.vibrational), "isotropic"))
    if not models:
        raise MaterialError(
            f"{parameters.gas} has no Raman response with rotation="
            f"{options.rotation} and vibration={options.vibration}: it has "
            f"{'a' if parameters.rotational else 'no'} rotational spectrum and "
            f"{'a' if parameters.vibrational else 'no'} vibrational mode"
        )
    return tuple(models)


def response_for(  # noqa: PLR0913 - the material, its options and the response's switches
    material: Gas | str | Species,
    options: MolecularOptions | None = None,
    *,
    amplitude: float = 1.0,
    fraction: float | None = None,
    thg: bool = True,
    selection: MaterialSelection | None = None,
    density: float | None = None,
) -> RamanResponse:
    """Build the Raman response of a named material.

    A built-in gas's response is optmat's (SEP 3), absolute, so its ``fraction`` must
    be left ``None``; a caller's species with its own molecular constants
    (:class:`~solprop.physdata.raman_types.MolecularRaman`) gives an absolute response
    built from them; fused silica gives optmat's bundle shape, normalised, whose
    magnitude the caller supplies, either as ``amplitude`` or, downstream, as
    ``fraction`` times a susceptibility.

    No material selects a normalised damped oscillator; build one with
    :func:`normalised_damped_oscillator` if you want it.

    Parameters
    ----------
    material : Gas, str or Species
        A built-in gas, a caller's species with Raman constants, or one of
        :data:`SILICA_NAMES` for fused silica.
    options : MolecularOptions or None, optional
        Which parts of a molecular spectrum to build, and at what temperature. Ignored
        for silica. ``None`` and the default options include every available term at
        :data:`solprop.constants.ROOM_TEMPERATURE`; a term set to ``True`` is refused
        by species name when it has no model.
    amplitude : float, optional
        For silica only: the value ``integral h dt`` is to take, m^5/V^2 per unit
        density. Default 1.0, a pure shape.
    fraction : float or None, optional
        The Raman fraction ``f_R`` in ``(0, 1]``, or ``None``. Refused for a molecular
        gas, whose models are absolute and would be scaled twice.
    thg : bool, optional
        Whether a field grid keeps the third harmonic of the drive. Default ``True``.
    selection : MaterialSelection or None, optional
        ``None`` is ``MaterialSelection()``. A built-in gas's response is optmat's
        under it, dispersed to its ``carrier_wavelength``, which must then be set
        (:func:`solprop.nonlinear.optmat_raman.gas_raman`); fused silica's is the
        bundle its ``raman`` names. Refused beside a caller's species.
    density : float or None, optional
        For a built-in gas, the constant number density in m^-3 of the fill: checked
        against the response's validated range, and the density at which H2's
        shifted lines are built.

    Returns
    -------
    RamanResponse
        The response: for silica one term; for a built-in gas an
        :class:`~solprop.nonlinear.optmat_raman.OptmatRamanResponse` with one term per
        damping group and tensor family; for a caller's species one term per part
        of its spectrum.

    Raises
    ------
    MaterialError
        If the material is unknown, has no Raman data, has no part left after the
        options, is a gas given a ``fraction``, or is a built-in gas without a
        carrier.

    Examples
    --------
    >>> from solprop.nonlinear.raman import response_for
    >>> from solprop.physdata import Gas, MaterialSelection
    >>> selection = MaterialSelection(carrier_wavelength=800e-9)
    >>> type(response_for(Gas.N2, selection=selection)).__name__
    'OptmatRamanResponse'
    >>> len(response_for("SiO2", fraction=0.18).models)
    1
    """
    if isinstance(material, str) and material.strip().lower() in SILICA_NAMES:
        chosen = "recommended" if selection is None else selection.raman
        return bundle_response(
            SilicaBundle.resolve(chosen),
            amplitude=amplitude,
            fraction=fraction,
            thg=thg,
        )
    if isinstance(material, Species) and not built_in_population(material):
        return _supplied_response(material, options, fraction, thg, selection, density)
    gas = material.name if isinstance(material, Species) else as_gas(material).value
    return _optmat_response(
        gas,
        options,
        MaterialSelection() if selection is None else selection,
        fraction,
        thg,
        density,
    )


def _supplied_response(
    material: Species,
    options: MolecularOptions | None,
    fraction: float | None,
    thg: bool,
    selection: MaterialSelection | None,
    density: float | None,
) -> RamanResponse:
    """Build a caller's species' Raman response from its own molecular constants.

    Parameters
    ----------
    material : Species
        A supplied species, never matched to a built-in gas by its name.
    options : MolecularOptions or None
        Term flags and temperature.
    fraction : float or None
        Refused: the response is absolute.
    thg : bool
        Whether a field grid keeps the third harmonic.
    selection : MaterialSelection or None
        Refused: a selection names optmat's models, not a caller's.
    density : float or None
        Refused: it belongs to an optmat response.

    Returns
    -------
    RamanResponse
        One term per part of the spectrum.

    Raises
    ------
    MaterialError
        If the species has no Raman constants, or a fraction, selection or density
        is given.
    """
    if selection is not None or density is not None:
        msg = (
            f"{material.name} is a caller's species: its Raman response is built from "
            "its own constants, which a selection or a density does not change"
        )
        raise MaterialError(msg)
    if material.raman is None:
        raise MaterialError(f"{material.name} has no Raman response data")
    if fraction is not None:
        raise MaterialError(
            f"the Raman response of {material.name} is built from its own molecular "
            "constants and is already absolute; a fraction would scale it twice. "
            "Pass fraction only for a normalised model such as fused silica"
        )
    return RamanResponse(
        tensor_terms=_molecular_models(material.raman, options or MolecularOptions()),
        fraction=None,
        thg=thg,
    )
