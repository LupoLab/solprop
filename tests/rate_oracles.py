# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test-only ionisation-rate oracles: the ADK and PPT forms with explicit parameters.

solprop keeps no ionisation-rate formula of its own (SEP 3, D1): every runtime rate is
optmat's (:class:`solprop.nonlinear.optmat_rates.OptmatRate`) or the caller's. Tests of
the numerics around a rate -- tabulation, the plasma kernels, backend parity, the
envelope quadrature -- still want a fixed, closed-form rate that does not move when
optmat's curated models do. :class:`AdkOracle` is that rate: the quasistatic
tunnelling form of Ammosov, Delone and Krainov (Sov. Phys. JETP 64, 1191 (1986)), with
``n* = Z / sqrt(2 Ip)``, ``l* = n* - 1`` and the angular factor summed over ``m``, the
arithmetic solprop's own ``ADK`` used before SEP 3, Phase 4b. It is a caller's rate as
far as the package is concerned: it carries its species name and its energy per event,
and nothing in ``src/`` reads it. :class:`PptOracle` is the same for the PPT rate in the
form of Ilkov, Decker and Chin, with the Stark shift, the ion's dipole factor and Tong
and Lin's barrier-suppression factor solprop's own ``PPT`` applied by default before
Phase 4b; it is the rate of the M2 witness (``tests/driver/test_prop_capillary.py``).

:data:`HISTORICAL_ATOMS` holds the atomic inputs (ionisation potential in hartree,
orbital and effective charge) that solprop's former registry gave the noble gases, the
inputs of the historical test witnesses recorded with them. They are test inputs, not a
material database: the package never consults them.
"""

import math
from dataclasses import KW_ONLY, dataclass
from typing import Final

import numpy as np
from jaxtyping import Float
from numpy.typing import ArrayLike
from scipy import special

from solprop.constants import AU_EFIELD, AU_ENERGY, AU_POLARISABILITY, AU_TIME, C
from solprop.nonlinear.ionisation import MAX_RATE

__all__ = [
    "HISTORICAL_ATOMS",
    "HISTORICAL_POLARISABILITIES",
    "HISTORICAL_TONG_LIN_ALPHA",
    "AdkOracle",
    "PptOracle",
    "adk_oracle",
    "ppt_oracle",
]

HISTORICAL_ATOMS: Final[dict[str, tuple[float, int, float]]] = {
    "He": (0.9036, 0, 1.0),
    "Ne": (0.7925, 1, 1.0),
    "Ar": (0.5792, 1, 1.0),
    "Kr": (0.5142, 1, 1.0),
    "Xe": (0.4458, 1, 1.0),
}
"""``(Ip in hartree, orbital l, effective charge Z)`` of the noble gases, as solprop's
former registry gave them (before SEP 3, Phase 4b): the historical witnesses' inputs."""

HISTORICAL_POLARISABILITIES: Final[dict[str, tuple[float, float]]] = {
    "He": (1.3207, 0.2811),
    "Ne": (2.376, 1.2417),
    "Ar": (10.762, 6.807),
}
"""Static dipole polarisabilities ``(neutral, cation)`` in atomic units, as solprop's
former registry gave them: K. Wang et al., Eur. Phys. J. D 75, 46 (2021),
doi:10.1140/epjd/s10053-021-00046-2. The PPT Stark shift and dipole factor's inputs."""

HISTORICAL_TONG_LIN_ALPHA: Final[dict[str, float]] = {
    "He": 7.0,
    "Ne": 9.0,
    "Ar": 9.0,
    "Kr": 9.0,
    "Xe": 9.0,
}
"""The ``alpha`` of Tong and Lin's barrier-suppression factor, J. Phys. B 38, 2593
(2005), table 2, as solprop's former ``PPT`` applied it by default."""

_LOG_AU_TIME: Final = math.log(AU_TIME)
_LOG_TINY: Final = math.log(float(np.finfo(np.float64).tiny))
_LOG_PI: Final = math.log(math.pi)
_LOG_2: Final = math.log(2.0)
_LOG_4_SQRT2_OVER_PI: Final = math.log(4.0 * math.sqrt(2.0) / math.pi)
_GAMMA_SERIES_LIMIT: Final = 1e-2
_MAX_FIELD_AU: Final = 1e12
_THRESHOLD_ITERATIONS: Final = 8


def _angular_factor(orbital: int, m_abs: int) -> float:
    """Return ``(2l+1)(l+|m|)! / (2^|m| |m|! (l-|m|)!)``."""
    return (
        (2 * orbital + 1)
        * math.factorial(orbital + m_abs)
        / (2**m_abs * math.factorial(m_abs) * math.factorial(orbital - m_abs))
    )


@dataclass(frozen=True)
class AdkOracle:
    """The ADK tunnelling rate of explicitly supplied atomic parameters (test only).

    Attributes
    ----------
    species_identity : str
        The species the rate is for.
    ionisation_potential : float
        Ionisation potential in J; also the energy per event of the plasma loss.
    orbital : int
        Orbital angular momentum of the shell.
    charge : float
        Charge seen by the departing electron.
    occupancy : int
        Equivalent electrons multiplying the rate.
    cycle_averaged : bool
        Return the cycle average, ``sqrt(3E / (pi F_0))`` times the instantaneous rate.
    hydrogenic : bool
        Use ``l* = 0``, ``Z = 1`` and no angular factor.
    maximum : float
        Ceiling on the rate in s^-1.
    """

    species_identity: str
    ionisation_potential: float
    orbital: int = 1
    charge: float = 1.0
    _: KW_ONLY
    occupancy: int = 2
    cycle_averaged: bool = False
    hydrogenic: bool = False
    maximum: float = MAX_RATE

    @property
    def plasma_energy(self) -> float:
        """Return the energy per event the plasma loss pays, in J: the potential."""
        return self.ionisation_potential

    @property
    def convention(self) -> str:
        """Return the optical-cycle convention."""
        return "cycle-averaged" if self.cycle_averaged else "instantaneous-isotropic"

    @property
    def citation(self) -> str:
        """Return the reference of the form."""
        return (
            "Ammosov, Delone and Krainov, Sov. Phys. JETP 64, 1191 (1986); test oracle "
            f"of {self.species_identity}"
        )

    @property
    def parameters(self) -> tuple[tuple[str, float | int | bool | str], ...]:
        """Return every setting that changes the rate, name-sorted and hashable."""
        return tuple(
            sorted(
                {
                    "model": "adk-oracle",
                    "species": self.species_identity,
                    "potential": self.ionisation_potential,
                    "orbital": self.orbital,
                    "charge": self.charge,
                    "occupancy": self.occupancy,
                    "cycle_averaged": self.cycle_averaged,
                    "hydrogenic": self.hydrogenic,
                    "maximum": self.maximum,
                }.items()
            )
        )

    def _constants(
        self,
    ) -> tuple[float, float, float, float, tuple[tuple[int, float], ...]]:
        """Return ``(Ip, n*, log|C|^2, F0, ((|m|, f_lm), ...))`` in atomic units."""
        charge = 1.0 if self.hydrogenic else self.charge
        ip_au = self.ionisation_potential / AU_ENERGY
        n_star = charge / math.sqrt(2.0 * ip_au)
        l_star = 0.0 if self.hydrogenic else n_star - 1.0
        orders = (
            (0,) if self.hydrogenic else tuple(range(-self.orbital, self.orbital + 1))
        )
        angular = tuple(
            (abs(m), 1.0 if self.hydrogenic else _angular_factor(self.orbital, abs(m)))
            for m in orders
        )
        log_coulomb = (
            2.0 * n_star * math.log(2.0)
            - math.log(n_star)
            - float(special.gammaln(n_star + l_star + 1.0))
            - float(special.gammaln(n_star - l_star))
        )
        return ip_au, n_star, log_coulomb, (2.0 * ip_au) ** 1.5, angular

    def _log_prefactor(self) -> float:
        """Return ``log(N |C|^2 Ip)`` in atomic units."""
        ip_au, _, log_coulomb, _, _ = self._constants()
        return math.log(self.occupancy) + log_coulomb + math.log(ip_au)

    @property
    def threshold(self) -> float:
        """Return the field, V/m, below which the rate underflows and is zero."""
        _, n_star, _, f0, angular = self._constants()
        zero = 1.0 if self.hydrogenic else _angular_factor(self.orbital, 0)
        del angular
        exponent = 2.0 * n_star - 1.0
        k = self._log_prefactor() + math.log(zero) - _LOG_TINY - _LOG_AU_TIME
        u = k
        for _ in range(_THRESHOLD_ITERATIONS):
            u = k + exponent * math.log(3.0 * u)
        return 2.0 * f0 / (3.0 * u) * AU_EFIELD

    def _log_rate_au(
        self, field_au: Float[np.ndarray, " n_field"]
    ) -> Float[np.ndarray, " n_field"]:
        """Return ``log W`` in atomic units."""
        _, n_star, _, f0, angular = self._constants()
        ratio = 2.0 * f0 / field_au
        powers = sum(
            factor * ratio ** (2.0 * n_star - m_abs - 1.0) for m_abs, factor in angular
        )
        log_rate = self._log_prefactor() + np.log(powers) - 2.0 * f0 / (3.0 * field_au)
        if self.cycle_averaged:
            log_rate = log_rate + 0.5 * (
                math.log(3.0) + np.log(field_au) - _LOG_PI - math.log(f0)
            )
        return log_rate

    def __call__(
        self, field_amplitude: ArrayLike | float
    ) -> Float[np.ndarray, " *shape"]:
        """Return the rate in s^-1: zero below the threshold, clipped at the maximum."""
        field = np.asarray(np.abs(np.asarray(field_amplitude, dtype=np.float64)))
        flat = np.atleast_1d(field).ravel()
        rate = np.zeros_like(flat)
        active = flat >= self.threshold
        if bool(np.any(active)):
            field_au = np.minimum(flat[active] / AU_EFIELD, _MAX_FIELD_AU)
            rate[active] = np.exp(self._log_rate_au(field_au) - _LOG_AU_TIME)
        np.minimum(rate, self.maximum, out=rate)
        rate[~np.isfinite(rate)] = self.maximum
        return rate.reshape(field.shape)


def adk_oracle(gas: str, **options: int | bool | float) -> AdkOracle:
    """Return the oracle of a noble gas on its historical inputs.

    Parameters
    ----------
    gas : str
        One of :data:`HISTORICAL_ATOMS`.
    **options : int, bool or float
        ``occupancy``, ``cycle_averaged``, ``hydrogenic`` or ``maximum``.

    Returns
    -------
    AdkOracle
        The rate, numerically solprop's former ``ADK(gas, **options)``.
    """
    ip_hartree, orbital, charge = HISTORICAL_ATOMS[gas]
    return AdkOracle(gas, ip_hartree * AU_ENERGY, orbital, charge, **options)  # pyright: ignore[reportArgumentType] - checked at construction


def _alpha_of_gamma(gamma: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
    """Return ``2(asinh gamma - gamma / sqrt(1 + gamma^2))``, a series when small."""
    series = (
        2.0 / 3.0 * gamma**3 * (1.0 - 0.9 * gamma**2 + 0.8035714285714286 * gamma**4)
    )
    direct = 2.0 * (np.arcsinh(gamma) - gamma / np.sqrt(1.0 + gamma**2))
    return np.where(gamma < _GAMMA_SERIES_LIMIT, series, direct)


def _bracket_of_gamma(gamma: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
    """Return ``asinh gamma - gamma sqrt(1 + gamma^2) / (1 + 2 gamma^2)``, likewise."""
    series = (
        4.0 / 3.0 * gamma**3 * (1.0 - 2.1 * gamma**2 + 4.232142857142857 * gamma**4)
    )
    direct = np.arcsinh(gamma) - gamma * np.sqrt(1.0 + gamma**2) / (
        1.0 + 2.0 * gamma**2
    )
    return np.where(gamma < _GAMMA_SERIES_LIMIT, series, direct)


@dataclass(frozen=True)
class PptOracle:
    r"""The PPT rate of explicitly supplied atomic parameters (test only).

    Perelomov, Popov and Terent'ev, Sov. Phys. JETP 23, 924 (1966), in the form of
    Ilkov, Decker and Chin, J. Phys. B 25, 4005 (1992) (Berge et al., Rep. Prog. Phys.
    70, 1633 (2007), eq. (A14)), with the channel sum replaced by its integral over the
    channel phase: the arithmetic of solprop's own ``PPT`` with ``sum_integral=True``
    before SEP 3, Phase 4b. The ionisation potential is Stark-shifted by
    ``(alpha_0 - alpha_ion) E^2 / 2``, the rate multiplied by the ion's dipole factor
    ``exp(-2 alpha_ion E)`` and by Tong and Lin's barrier-suppression factor
    ``exp(-alpha_TL (Z^2 / Ip) E / kappa^3)`` (clamped at one, on the field-free
    potential), each when its parameter is nonzero.

    Attributes
    ----------
    species_identity : str
        The species the rate is for.
    wavelength : float
        Driving wavelength, m.
    ionisation_potential : float
        Field-free ionisation potential in J; also the energy per event.
    orbital : int
        Orbital angular momentum of the shell.
    charge : float
        Charge seen by the departing electron.
    polarisabilities : tuple of float
        ``(neutral, cation)`` static polarisabilities in atomic units; zeros switch
        the Stark shift and the dipole factor off.
    tong_lin_alpha : float
        Tong and Lin's ``alpha``; zero switches the barrier suppression off.
    occupancy : int
        Equivalent electrons multiplying the rate.
    m_sum : bool
        Sum over the magnetic sublevels rather than ``m = 0`` alone.
    cycle_averaged : bool
        Return the cycle average rather than the instantaneous rate.
    maximum : float
        Ceiling on the rate in s^-1.
    """

    species_identity: str
    wavelength: float
    ionisation_potential: float
    orbital: int = 1
    charge: float = 1.0
    _: KW_ONLY
    polarisabilities: tuple[float, float] = (0.0, 0.0)
    tong_lin_alpha: float = 0.0
    occupancy: int = 2
    m_sum: bool = True
    cycle_averaged: bool = False
    maximum: float = MAX_RATE

    @property
    def plasma_energy(self) -> float:
        """Return the energy per event the plasma loss pays, in J: the potential."""
        return self.ionisation_potential

    @property
    def convention(self) -> str:
        """Return the optical-cycle convention."""
        return "cycle-averaged" if self.cycle_averaged else "instantaneous-isotropic"

    @property
    def citation(self) -> str:
        """Return the reference of the form."""
        return (
            "Perelomov, Popov and Terent'ev, Sov. Phys. JETP 23, 924 (1966) in the "
            "form of Ilkov, Decker and Chin, J. Phys. B 25, 4005 (1992); test oracle "
            f"of {self.species_identity}"
        )

    @property
    def parameters(self) -> tuple[tuple[str, float | int | bool | str], ...]:
        """Return every setting that changes the rate, name-sorted and hashable."""
        return tuple(
            sorted(
                {
                    "model": "ppt-oracle",
                    "species": self.species_identity,
                    "wavelength": self.wavelength,
                    "potential": self.ionisation_potential,
                    "orbital": self.orbital,
                    "charge": self.charge,
                    "neutral_polarisability": self.polarisabilities[0],
                    "ion_polarisability": self.polarisabilities[1],
                    "tong_lin_alpha": self.tong_lin_alpha,
                    "occupancy": self.occupancy,
                    "m_sum": self.m_sum,
                    "cycle_averaged": self.cycle_averaged,
                    "maximum": self.maximum,
                }.items()
            )
        )

    def _angular(self) -> tuple[tuple[int, float], ...]:
        """Return the ``(|m|, f_lm)`` pairs summed over."""
        orders = range(-self.orbital, self.orbital + 1) if self.m_sum else (0,)
        return tuple((abs(m), _angular_factor(self.orbital, abs(m))) for m in orders)

    @property
    def threshold(self) -> float:
        """Return the field, V/m, below which the rate underflows and is zero.

        From the tunnelling limit of the ``|m| = 0`` term, where the Stark shift is a
        part in 1e8 of the potential and the corrections are one, as before.
        """
        ip_au = self.ionisation_potential / AU_ENERGY
        n_star = self.charge / math.sqrt(2.0 * ip_au)
        l_star = n_star - 1.0
        log_coulomb = (
            2.0 * n_star * _LOG_2
            - math.log(n_star)
            - float(special.gammaln(n_star + l_star + 1.0))
            - float(special.gammaln(n_star - l_star))
        )
        prefactor = (
            math.log(self.occupancy)
            + log_coulomb
            + math.log(ip_au)
            + math.log(_angular_factor(self.orbital, 0))
        )
        k = prefactor - _LOG_TINY - _LOG_AU_TIME
        u = k
        for _ in range(_THRESHOLD_ITERATIONS):
            u = k + (2.0 * n_star - 1.0) * math.log(3.0 * u)
        return 2.0 * (2.0 * ip_au) ** 1.5 / (3.0 * u) * AU_EFIELD

    def _log_rate_au(
        self, field_au: Float[np.ndarray, " n_field"]
    ) -> Float[np.ndarray, " n_field"]:
        """Return ``log W`` in atomic units."""
        # Through SI and back, as the former rate read its registry's values.
        neutral_si, ion_si = (one * AU_POLARISABILITY for one in self.polarisabilities)
        difference = (neutral_si - ion_si) / AU_POLARISABILITY
        ion = ion_si / AU_POLARISABILITY
        free_ip = self.ionisation_potential / AU_ENERGY
        omega = 2.0 * math.pi * C / self.wavelength * AU_TIME
        ip_au = free_ip + 0.5 * difference * field_au**2
        n_star = self.charge / np.sqrt(2.0 * ip_au)
        log_f0 = 1.5 * np.log(2.0 * ip_au)
        # l* = n* - 1, so Gamma(n* + l* + 1) = Gamma(2 n*) and Gamma(n* - l*) = 1.
        log_coulomb = (
            2.0 * n_star * _LOG_2 - np.log(n_star) - special.gammaln(2.0 * n_star)
        )
        gamma_k = omega * np.sqrt(2.0 * ip_au) / field_au
        g2 = gamma_k**2
        alpha = _alpha_of_gamma(gamma_k)
        beta = 2.0 * gamma_k / np.sqrt(1.0 + g2)
        v = (ip_au + field_au**2 / (4.0 * omega**2)) / omega
        log_common = (
            _LOG_4_SQRT2_OVER_PI
            + log_coulomb
            - 2.0 * v * _bracket_of_gamma(gamma_k)
            + np.log(ip_au)
            + np.log(g2)
            - np.log1p(g2)
        )
        if not self.cycle_averaged:
            log_common = log_common + 0.5 * (
                _LOG_PI + log_f0 - math.log(3.0) - np.log(field_au)
            )
        log_scale = _LOG_2 + log_f0 - np.log(field_au) - 0.5 * np.log1p(g2)
        terms = [
            log_common
            + (2.0 * n_star - m_abs - 1.5) * log_scale
            + math.log(factor)
            - float(special.gammaln(m_abs + 1.0))
            # The channel sum's integral over the channel phase, in closed form.
            + (
                0.5 * _LOG_PI
                + float(special.gammaln(m_abs + 1.0))
                + m_abs * np.log(beta)
                - _LOG_2
                - (m_abs + 1.0) * np.log(alpha + beta)
                + 0.5 * (np.log(beta) - np.log(alpha))
            )
            for m_abs, factor in self._angular()
        ]
        log_rate = special.logsumexp(np.stack(terms), axis=0) + math.log(self.occupancy)
        if self.tong_lin_alpha:
            kappa = math.sqrt(2.0 * free_ip)
            log_rate = log_rate + np.minimum(
                -self.tong_lin_alpha * self.charge**2 / free_ip * field_au / kappa**3,
                0.0,
            )
        if ion != 0.0:
            log_rate = log_rate - 2.0 * ion * field_au
        return log_rate

    def __call__(
        self, field_amplitude: ArrayLike | float
    ) -> Float[np.ndarray, " *shape"]:
        """Return the rate in s^-1: zero below the threshold, clipped at the maximum."""
        field = np.asarray(np.abs(np.asarray(field_amplitude, dtype=np.float64)))
        flat = np.atleast_1d(field).ravel()
        rate = np.zeros_like(flat)
        active = flat >= self.threshold
        if bool(np.any(active)):
            field_au = np.minimum(flat[active] / AU_EFIELD, _MAX_FIELD_AU)
            rate[active] = np.exp(self._log_rate_au(field_au) - _LOG_AU_TIME)
        np.minimum(rate, self.maximum, out=rate)
        rate[~np.isfinite(rate)] = self.maximum
        return rate.reshape(field.shape)


def ppt_oracle(gas: str, wavelength: float, **options: int | bool | float) -> PptOracle:
    """Return the PPT oracle of a noble gas on its historical inputs.

    With no options it is numerically solprop's former default ``PPT(gas,
    wavelength)``: the Stark shift and the dipole factor on where the polarisabilities
    were tabulated, Tong and Lin's barrier suppression on, every sublevel, two
    electrons.

    Parameters
    ----------
    gas : str
        One of :data:`HISTORICAL_ATOMS`.
    wavelength : float
        Driving wavelength, m.
    **options : int, bool or float
        ``tong_lin_alpha`` (0 for none), ``occupancy``, ``m_sum``, ``cycle_averaged``
        or ``maximum``.

    Returns
    -------
    PptOracle
        The rate.
    """
    ip_hartree, orbital, charge = HISTORICAL_ATOMS[gas]
    alpha = float(options.pop("tong_lin_alpha", HISTORICAL_TONG_LIN_ALPHA[gas]))
    polarisabilities = HISTORICAL_POLARISABILITIES.get(gas, (0.0, 0.0))
    return PptOracle(
        gas,
        wavelength,
        ip_hartree * AU_ENERGY,
        orbital,
        charge,
        polarisabilities=polarisabilities,
        tong_lin_alpha=alpha,
        **options,  # pyright: ignore[reportArgumentType] - checked at construction
    )
