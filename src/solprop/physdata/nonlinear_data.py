# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Third-order nonlinear data for gases: the Kerr coefficient at the carrier.

The nonlinearity is held as a **per-particle** quantity, exactly as the linear
susceptibility is: :func:`gamma3` returns the third-order hyperpolarisability
:math:`\gamma^{(3)}` in m\ :sup:`5`/V\ :sup:`2`, and

.. math:: \chi^{(3)} = \gamma^{(3)}\,\rho .

So the Kerr response of a gas is strictly linear in number density, and a pressure
gradient scales it without re-deriving anything.

The nonlinear index follows from the susceptibility by

.. math:: n_2 = \frac{3\chi^{(3)}}{4\varepsilon_0 c\,n_0^2},

with :math:`n_0` the linear index at the carrier. :func:`n2_from_chi3` and
:func:`chi3_from_n2` are that relation and its inverse, in the same convention as
``docs/dev/conventions.md`` section 4: an intensity :math:`I` shifts the index by
:math:`n_2 I`.

The coefficient is optmat's (SEP 3, section 4.3): :class:`OptmatKerr` adapts optmat's
fitted electronic hyperpolarizability, which is dispersive, and evaluates its Kerr
process at the run's **carrier** wavelength -- one number per species and run, the
``carrier-kerr`` approximation, never a universal constant. The Taylor convention of
the fit, :math:`p^{(3)} = \gamma_T E^3/6`, gives
:math:`\gamma^{(3)} = \gamma_T/(6\varepsilon_0)`, which optmat's public
``gamma3("kerr", wavelength)`` returns; optmat's own gas ``n2``, taken with
:math:`n_0 = 1`, is never imported, so that :func:`n2_from_chi3` applies the exact
:math:`n_0^2`. Air has no hyperpolarizability of its own: its populations each carry
theirs.

SEP 6's dispersive electronic policies take the whole fit instead:
:class:`DispersiveKerrModel` is optmat's model of one gas in SI, with the per-leg bound
and the band edges solprop derives from optmat's support record, and
:class:`ElectronicRegions` classifies a run's band against them (SEP 6, section 4.10).
"""

import functools
import logging
import math
import warnings
from dataclasses import dataclass
from typing import Final

import numpy as np
from jaxtyping import Float

from solprop.constants import EPS0, ROOM_TEMPERATURE, C
from solprop.exceptions import MaterialError
from solprop.physdata import _optmat
from solprop.physdata.gases import Gas, as_gas
from solprop.physdata.selection import MaterialRecord, MaterialSelection

__all__ = [
    "CarrierKerr",
    "DispersiveKerrModel",
    "ElectronicRegions",
    "OptmatKerr",
    "carriers_agree",
    "chi3",
    "chi3_from_n2",
    "gamma3",
    "n2",
    "n2_from_chi3",
    "selected_carrier",
]

logger = logging.getLogger(__name__)


def gamma3(gas: Gas | str, *, wavelength: float) -> float:
    """Return a gas's per-particle Kerr coefficient at a carrier wavelength.

    optmat's electronic hyperpolarizability evaluated at the carrier
    (:class:`OptmatKerr`); the coefficient is dispersive, so the carrier is required.

    Parameters
    ----------
    gas : Gas or str
        A built-in gas other than air.
    wavelength : float
        Vacuum carrier wavelength in m, positive and finite.

    Returns
    -------
    float
        ``gamma3`` in m^5/V^2, such that ``chi3 = gamma3 * rho``.

    Raises
    ------
    MaterialError
        If the gas is unknown or is air, or the wavelength is not positive and finite.

    Warns
    -----
    MaterialRangeWarning
        Outside the model's fitted wavelengths.

    Examples
    --------
    >>> from solprop.physdata import Gas, gamma3
    >>> f"{gamma3(Gas.Ar, wavelength=800e-9):.6e}"
    '1.476195e-51'
    """
    return _kerr(as_gas(gas)).gamma3(wavelength)


@functools.cache
def _kerr(gas: Gas) -> "OptmatKerr":  # noqa: UP037 - the class is defined below
    """Return the Kerr adapter of a gas, once per gas.

    Parameters
    ----------
    gas : Gas
        The species.

    Returns
    -------
    OptmatKerr
        Its adapter.

    Raises
    ------
    MaterialError
        If optmat has no hyperpolarizability model of the gas.
    """
    if gas is Gas.Air:
        msg = (
            "air has no Kerr coefficient of its own: its populations (N2, O2, Ar) each "
            "carry theirs; build the composition with Mixture.air"
        )
        raise MaterialError(msg)
    return OptmatKerr(gas)


def n2_from_chi3(chi3_value: float, index: float) -> float:
    """Convert a third-order susceptibility to a nonlinear index.

    ``n2 = 3 chi3 / (4 eps0 c n0^2)``.

    Parameters
    ----------
    chi3_value : float
        Third-order susceptibility in m^2/V^2.
    index : float
        Linear refractive index at the wavelength of interest, > 0.

    Returns
    -------
    float
        Nonlinear index in m^2/W, in the convention where an intensity ``I`` shifts the
        refractive index by ``n2 * I``.

    Examples
    --------
    >>> from solprop.physdata import n2_from_chi3
    >>> f"{n2_from_chi3(3e-26, 1.0003):.6e}"
    '8.471348e-24'
    """
    return 3.0 * chi3_value / (4.0 * EPS0 * C * index**2)


def chi3_from_n2(n2_value: float, index: float) -> float:
    """Convert a nonlinear index to a third-order susceptibility.

    The inverse of :func:`n2_from_chi3`.

    Parameters
    ----------
    n2_value : float
        Nonlinear index in m^2/W.
    index : float
        Linear refractive index at the wavelength of interest, > 0.

    Returns
    -------
    float
        Third-order susceptibility in m^2/V^2.

    Examples
    --------
    >>> from solprop.physdata import chi3_from_n2, n2_from_chi3
    >>> f"{chi3_from_n2(n2_from_chi3(3e-26, 1.0003), 1.0003):.12e}"
    '3.000000000000e-26'
    """
    return 4.0 * n2_value * EPS0 * C * index**2 / 3.0


def chi3(
    gas: Gas | str,
    *,
    pressure: float,
    temperature: float = ROOM_TEMPERATURE,
    selection: MaterialSelection | None = None,
    wavelength: float | None = None,
) -> float:
    """Return the third-order susceptibility of a gas at a carrier.

    ``chi3 = gamma3 * rho``: strictly linear in number density, with optmat's Kerr
    coefficient at the carrier (:class:`OptmatKerr`) and optmat's equation of state.

    Parameters
    ----------
    gas : Gas or str
        The species.
    pressure : float
        Absolute pressure in **bar** (the documented non-SI user-interface unit), >= 0.
    temperature : float, optional
        Temperature in K, > 0. Defaults to 293.15 K.
    selection : MaterialSelection or None, optional
        ``None`` is ``MaterialSelection()``; its ``carrier_wavelength`` may name the
        carrier.
    wavelength : float or None, optional
        The carrier wavelength in m; it must agree with the selection's
        ``carrier_wavelength`` when both are given, and one of the two is required.

    Returns
    -------
    float
        Third-order susceptibility in m^2/V^2.

    Raises
    ------
    MaterialError
        If the gas is unknown or has no Kerr coefficient (air), or there is no
        carrier.

    Examples
    --------
    >>> from solprop.physdata import Gas, chi3
    >>> f"{chi3(Gas.Ar, pressure=1.0, wavelength=800e-9):.6e}"
    '3.649772e-26'
    """
    # species() builds on this module's adapter, so it is imported where it is needed
    # rather than at module scope.
    from solprop.physdata.species import species  # noqa: PLC0415 - a module cycle

    selection = MaterialSelection() if selection is None else selection
    material = species(gas, selection=selection)
    carrier = selected_carrier(selection, wavelength)
    coefficient = material.kerr_coefficient(carrier)
    if coefficient is None:
        msg = (
            f"no Kerr coefficient for {material.name}: its populations each carry "
            "their own; build the composition with Mixture.air"
        )
        raise MaterialError(msg)
    return coefficient * material.equation_of_state.density(pressure, temperature)


def n2(
    gas: Gas | str,
    *,
    pressure: float,
    temperature: float = ROOM_TEMPERATURE,
    wavelength: float | None = None,
    selection: MaterialSelection | None = None,
) -> float:
    """Return the nonlinear index of a gas at a carrier.

    The exact ``n2 = 3 chi3 / (4 eps0 c n0^2)`` with optmat's Kerr coefficient, its
    density and its linear index ``n0``, all at the carrier; optmat's own gas ``n2``,
    which takes ``n0 = 1``, is not used.

    Parameters
    ----------
    gas : Gas or str
        The species.
    pressure : float
        Absolute pressure in **bar** (the documented non-SI user-interface unit), >= 0.
    temperature : float, optional
        Temperature in K, > 0. Defaults to 293.15 K.
    wavelength : float or None, optional
        The carrier wavelength in m at which the dispersive Kerr coefficient and
        ``n0`` are evaluated: it, or the selection's ``carrier_wavelength``, is
        required, and the two must agree when both are given.
    selection : MaterialSelection or None, optional
        ``None`` is ``MaterialSelection()``.

    Returns
    -------
    float
        Nonlinear index in m^2/W.

    Raises
    ------
    MaterialError
        If the gas is unknown or has no Kerr coefficient, or the carriers disagree
        or are both missing.

    Warns
    -----
    MaterialRangeWarning
        If the carrier lies outside the qualified range of the gas's models.

    Examples
    --------
    Helium at 1 bar and 800 nm.

    >>> from solprop.physdata import Gas, n2
    >>> f"{n2(Gas.He, pressure=1.0, wavelength=800e-9):.3e}"
    '3.635e-25'
    """
    from solprop.physdata.species import species  # noqa: PLC0415 - as in chi3

    selection = MaterialSelection() if selection is None else selection
    # The coefficient is dispersive: the carrier is the caller's or the selection's,
    # never a silent default.
    wavelength = selected_carrier(selection, wavelength)
    omega = 2.0 * math.pi * C / wavelength
    material = species(gas, selection=selection)
    rho = material.equation_of_state.density(pressure, temperature)
    linear = complex(rho * material.susceptibility.susceptibility(omega))
    index = float(np.real(np.sqrt(1.0 + linear)))
    return n2_from_chi3(
        chi3(
            gas,
            pressure=pressure,
            temperature=temperature,
            selection=selection,
            wavelength=wavelength,
        ),
        index,
    )


_CARRIER_TOLERANCE: Final[float] = 1e-12
"""Relative difference below which two carrier wavelengths are the same one."""


def carriers_agree(first: float, second: float) -> bool:
    """Return whether two carrier wavelengths are the same, to rounding.

    Parameters
    ----------
    first : float
        A vacuum wavelength in m.
    second : float
        Another, in m.

    Returns
    -------
    bool
        True when they differ by at most 1e-12 relative: a wavelength computed back
        from an angular frequency is the same carrier.

    Examples
    --------
    >>> from solprop.physdata.nonlinear_data import carriers_agree
    >>> carriers_agree(800e-9, 800e-9 * (1 + 1e-15)), carriers_agree(800e-9, 1030e-9)
    (True, False)
    """
    return math.isclose(first, second, rel_tol=_CARRIER_TOLERANCE, abs_tol=0.0)


def _checked_carrier(wavelength: float, material: str) -> float:
    """Validate a carrier wavelength.

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength in m.
    material : str
        The gas, for the message.

    Returns
    -------
    float
        The wavelength as a float.

    Raises
    ------
    MaterialError
        If it is not positive and finite.
    """
    value = float(wavelength)
    if not (math.isfinite(value) and value > 0.0):
        msg = (
            f"{material}: the Kerr coefficient is evaluated at a positive, finite "
            f"carrier wavelength in m, got {wavelength!r}"
        )
        raise MaterialError(msg)
    return value


@dataclass(frozen=True)
class CarrierKerr:
    """Record one gas's Kerr coefficient at the carrier, and what it leaves out.

    The ``carrier-kerr`` approximation (SEP 3, section 4.3) freezes the dispersive
    electronic response at one wavelength and uses it for every process the cubic
    term drives, the third harmonic included. optmat's distinct third-harmonic
    coefficient measures what that costs: the record keeps both, and their ratio
    where optmat evaluates the third harmonic within its fitted data.

    Attributes
    ----------
    material : str
        The gas.
    model : str
        optmat's model name, ``"hyperpolarizability/<gas>"``.
    digest : str
        Its digest.
    carrier_wavelength : float
        The vacuum carrier wavelength in m.
    gamma3 : float
        The Kerr coefficient there, m^5/V^2 per particle: what the run uses.
    gamma3_thg : float
        optmat's third-harmonic coefficient at the same fundamental, m^5/V^2.
    thg_ratio : float or None
        ``gamma3_thg / gamma3``; ``None`` where optmat's third-harmonic evaluation
        leaves its fitted data, so that the ratio would only compare extrapolations.
    thg_reason : str
        Empty for a reported ratio; otherwise optmat's range warning.
    """

    material: str
    model: str
    digest: str
    carrier_wavelength: float
    gamma3: float
    gamma3_thg: float
    thg_ratio: float | None
    thg_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return the record as plain values, for a run's material manifest.

        Returns
        -------
        dict
            Every field.
        """
        return {
            "material": self.material,
            "model": self.model,
            "digest": self.digest,
            "carrier_wavelength": self.carrier_wavelength,
            "gamma3": self.gamma3,
            "gamma3_thg": self.gamma3_thg,
            "thg_ratio": self.thg_ratio,
            "thg_reason": self.thg_reason,
        }


@dataclass(frozen=True)
class OptmatKerr:
    r"""Take a gas's electronic Kerr coefficient from optmat, at the carrier.

    optmat fits each gas's second hyperpolarizability :math:`\gamma_T(\omega)` in the
    Taylor convention, :math:`p^{(3)} = \gamma_T E^3/6`, so that solprop's
    per-particle coefficient in :math:`P^{(3)} = \varepsilon_0\gamma^{(3)}N E^3` is

    .. math:: \gamma^{(3)}(\lambda_c) = \frac{\gamma_{T,\rm Kerr}(\lambda_c)}
              {6\varepsilon_0},

    evaluated at the carrier :math:`\lambda_c` (optmat's public
    ``gamma3("kerr", wavelength)``). The adapter holds the model, not a number: the
    carrier is the run's, so one species serves runs at any carrier. No Raman
    amplitude is multiplied by it, and no local-field factor is added
    (SEP 3, section 4.3).

    Attributes
    ----------
    gas : Gas or str
        The species; optmat has a model for every built-in gas but air, whose
        populations each carry their own.

    Raises
    ------
    MaterialError
        If optmat has no hyperpolarizability model of the gas.

    Examples
    --------
    >>> from solprop.physdata.nonlinear_data import OptmatKerr
    >>> argon = OptmatKerr("Ar")
    >>> f"{argon.gamma3(800e-9):.6e}"
    '1.476195e-51'
    """

    gas: Gas | str

    def __post_init__(self) -> None:
        """Coerce the species and resolve its model, so that a gas without one fails.

        Raises
        ------
        MaterialError
            If optmat has no hyperpolarizability model of the gas.
        """
        object.__setattr__(self, "gas", as_gas(self.gas))
        _ = self.model

    @property
    def key(self) -> Gas:
        """Return the species as a registry member.

        Returns
        -------
        Gas
            The gas.
        """
        return as_gas(self.gas)

    @functools.cached_property
    def model(self) -> _optmat.Untyped:
        """Return optmat's hyperpolarizability model of the gas, resolved once.

        Returns
        -------
        optmat.gas.nonlinear.HyperpolarizabilityModel
            ``hyperpolarizability_models()[gas]``.

        Raises
        ------
        MaterialError
            If optmat has none for the gas.
        """
        from optmat.gas.nonlinear import (  # noqa: PLC0415 - optmat is imported lazily
            hyperpolarizability_models,
        )

        models = hyperpolarizability_models()
        name = self.key.value
        # optmat keys its models by its own Gas, a StrEnum, which a name finds.
        if name not in models:
            msg = (
                f"optmat has no hyperpolarizability model of {name}; it has one for "
                f"{', '.join(sorted(str(gas) for gas in models))}"
            )
            raise MaterialError(msg)
        return models[name]  # pyright: ignore[reportCallIssue, reportArgumentType] - a StrEnum key

    @property
    def material_records(self) -> tuple[MaterialRecord, ...]:
        """Return the record of the hyperpolarizability model.

        Returns
        -------
        tuple of MaterialRecord
            One record.
        """
        return (_optmat.record(self.model),)

    def gamma3(self, wavelength: float) -> float:
        """Return the per-particle Kerr coefficient at a carrier wavelength.

        Parameters
        ----------
        wavelength : float
            Vacuum carrier wavelength in m, positive and finite.

        Returns
        -------
        float
            ``gamma3`` in m^5/V^2, such that ``chi3 = gamma3 * rho``.

        Raises
        ------
        MaterialError
            If the wavelength is not positive and finite.

        Warns
        -----
        MaterialRangeWarning
            Once per evaluation context, outside the model's fitted wavelengths
            (400 nm to 20 um for the built-in gases).
        """
        carrier = _checked_carrier(wavelength, self.key.value)
        model = self.model
        _optmat.check_status(model, self.key.value)
        with _optmat.translated(
            model,
            "Kerr coefficient",
            material=self.key.value,
            wavelength=np.array([carrier]),
        ):
            return float(model.gamma3("kerr", carrier))

    def dispersive_model(self) -> "DispersiveKerrModel":  # noqa: UP037 - defined below
        """Return the whole fit in SI, for a dispersive electronic policy (SEP 6).

        Returns
        -------
        DispersiveKerrModel
            The gas's optmat model in SI, with its leg bound and band edges.

        Raises
        ------
        MaterialError
            If optmat's model of the gas carries no dispersion parameters or support
            record.

        Warns
        -----
        MaterialStatusWarning
            Once per evaluation context, for a provisional or untested model or
            support record.

        Examples
        --------
        >>> from solprop.physdata.nonlinear_data import OptmatKerr
        >>> f"{OptmatKerr('He').dispersive_model().gamma3_static:.6e}"
        '5.064575e-53'
        """
        model = self.model
        _optmat.check_status(model, self.key.value)
        support = getattr(model, "support", None)
        if support is not None:
            _optmat.check_status(support, self.key.value)
        return DispersiveKerrModel.from_optmat(model)

    def at_carrier(self, wavelength: float) -> CarrierKerr:
        """Return the carrier record: the coefficient used and the THG discrepancy.

        Parameters
        ----------
        wavelength : float
            Vacuum carrier wavelength in m, positive and finite.

        Returns
        -------
        CarrierKerr
            The Kerr coefficient at the carrier, optmat's third-harmonic one at the
            same fundamental, and their ratio where the latter is within optmat's
            fitted data.

        Raises
        ------
        MaterialError
            If the wavelength is not positive and finite.

        Examples
        --------
        >>> from solprop.physdata.nonlinear_data import OptmatKerr
        >>> record = OptmatKerr("Ar").at_carrier(800e-9)
        >>> f"{record.thg_ratio:.3f}"
        '1.177'
        """
        from optmat.exceptions import ModelRangeWarning  # noqa: PLC0415 - lazily

        kerr = self.gamma3(wavelength)
        model = self.model
        entry = _optmat.record(model)
        # The third harmonic is evaluated for the record only, never propagated, so
        # its range warning is the record's reason rather than a run warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            thg = float(model.gamma3("thg", float(wavelength)))
        outside = [
            str(item.message)
            for item in caught
            if issubclass(item.category, ModelRangeWarning)
        ]
        return CarrierKerr(
            material=self.key.value,
            model=entry.name,
            digest=entry.digest,
            carrier_wavelength=float(wavelength),
            gamma3=kerr,
            gamma3_thg=thg,
            thg_ratio=None if outside else thg / kerr,
            thg_reason="; ".join(outside),
        )


_TAYLOR_FACTOR: Final[float] = 6.0
"""The Taylor convention's ``p = gamma_T E^3 / 6`` (SEP 6, section 4.2)."""

_CM2_TO_M2: Final[float] = 1e-14
"""optmat's ``A`` in units of 1e-10 cm^2, to m^2 (SEP 6, section 4.5)."""

_CM4_TO_M4: Final[float] = 1e-28
"""optmat's ``B`` and ``B'`` in units of 1e-20 cm^4, to m^4 (SEP 6, section 4.5)."""


@dataclass(frozen=True)
class ElectronicRegions:
    r"""Classify a run's band against one gas's hyperpolarizability support.

    SEP 6, section 4.10, as qualified by its Phase 0 report. Every leg of a dispersive
    evaluator is held at the gas's leg bound :math:`\nu_c`, and the run is never
    refused for the frequencies its field occupies; what is recorded is which part of
    the band falls in which region:

    - **supported**: every leg within :math:`\nu_c` and inside the band edge, where
      the bound changes no quartet within the fitted sums;
    - **bounded**: beyond :math:`\nu_c` and inside the band edge, where the legs are
      held at the bound -- a bounded estimate, which can underestimate a response
      that rises below resonance but is not a certified bound on it;
    - **best effort**: beyond the band edge, or beyond :math:`\nu_c` for a gas whose
      edge is undefined (no curated two-photon limit), where no off-resonant model is
      valid and the response is held at its value at the edge of the data.

    Each region is one interval of vacuum wavelength, since the classification is
    monotonic in frequency.

    Attributes
    ----------
    material : str
        The gas.
    polarisation : {'linear', 'any'}
        Which band edge applied: one linear polarisation, or any (a vector run).
    band : tuple of float
        The classified band, shortest and longest vacuum wavelength in m.
    leg_bound : float
        :math:`\nu_c`, m^-1.
    band_edge : float or None
        The band edge that applied, m^-1; ``None`` where optmat's record leaves it
        undefined.
    supported : tuple of float or None
        Shortest and longest wavelength in m of the supported part, or ``None``.
    bounded : tuple of float or None
        The same of the bounded part.
    best_effort : tuple of float or None
        The same of the best-effort part.
    """

    material: str
    polarisation: str
    band: tuple[float, float]
    leg_bound: float
    band_edge: float | None
    supported: tuple[float, float] | None
    bounded: tuple[float, float] | None
    best_effort: tuple[float, float] | None

    @property
    def beyond_support(self) -> bool:
        """Return whether any part of the band is bounded or best effort.

        Returns
        -------
        bool
            True when a run's legs reach beyond the gas's leg bound or band edge.
        """
        return self.bounded is not None or self.best_effort is not None

    def to_dict(self) -> dict[str, object]:
        """Return the classification as plain values, for a run's manifest.

        Returns
        -------
        dict
            Every field; intervals as ``[shortest, longest]`` wavelengths in m.
        """
        return {
            "material": self.material,
            "polarisation": self.polarisation,
            "band": list(self.band),
            "leg_bound": self.leg_bound,
            "band_edge": self.band_edge,
            "supported": None if self.supported is None else list(self.supported),
            "bounded": None if self.bounded is None else list(self.bounded),
            "best_effort": None if self.best_effort is None else list(self.best_effort),
        }

    def describe(self) -> str:
        """Return a one-line account of the regions, for a run's log.

        Returns
        -------
        str
            The gas, its band and each region's wavelengths in nm.
        """

        def span(interval: tuple[float, float] | None) -> str | None:
            """Return an interval of wavelengths in nm, or ``None``.

            Parameters
            ----------
            interval : tuple of float or None
                Shortest and longest wavelength, m.

            Returns
            -------
            str or None
                The interval in nm, or ``None`` for no interval.
            """
            if interval is None:
                return None
            return f"{interval[0] * 1e9:.1f}-{interval[1] * 1e9:.1f} nm"

        parts = [
            f"{name} {text}"
            for name, text in (
                ("supported", span(self.supported)),
                ("bounded", span(self.bounded)),
                ("best effort", span(self.best_effort)),
            )
            if text is not None
        ]
        edge = (
            "undefined" if self.band_edge is None else f"{1e9 / self.band_edge:.1f} nm"
        )
        return (
            f"{self.material}: band {span(self.band)} is {', '.join(parts)} (legs held "
            f"at {1e9 / self.leg_bound:.1f} nm, band edge {edge}, "
            f"{self.polarisation} polarisation)"
        )


@dataclass(frozen=True)
class DispersiveKerrModel:
    r"""Hold one gas's dispersive hyperpolarizability in SI, with its leg bound.

    optmat's fit of the Taylor-convention second hyperpolarizability over a quartet of
    signed vacuum wavenumbers :math:`\nu_j = \omega_j/(2\pi c)`, output first and
    counted negative, so that the four sum to zero (SEP 6, section 4.5):

    .. math:: \gamma^{(3)} = \gamma^{(3)}_0\,[1 + a S_2 + b S_2^2 + b' S_4],
              \qquad S_2 = \sum_j \nu_j^2,\ S_4 = \sum_j \nu_j^4,

    with :math:`\gamma^{(3)}_0 = \gamma_{0,\rm au}\,\mathrm{au}/(6\varepsilon_0)`
    solprop's per-particle coefficient in m^5/V^2 (section 4.2), and optmat's ``A`` in
    1e-10 cm^2 and ``B``, ``B'`` in 1e-20 cm^4 taken to m^2 and m^4. Every leg enters
    through its bounded weight
    :math:`\tilde d(\nu) = \min(\nu^2, \nu_c^2)`, the rule of section 4.10, which
    changes no quartet within the fitted sums.

    The leg bound and the band edges are derived from optmat's support record of the
    model (optmat's OEP 5): :math:`\nu_c = \sqrt{3 W_{2,\max}/4}`
    (``support.leg_max_in_w2``) and :math:`\min(E_1, E_2/2)` for one or for any
    polarisation (``support.band_edge``). solprop hard-codes none of them.

    Attributes
    ----------
    material : str
        The gas.
    gamma3_static : float
        :math:`\gamma^{(3)}_0`, the static per-particle coefficient, m^5/V^2.
    a : float
        The quadratic coefficient, m^2.
    b : float
        The coefficient of :math:`S_2^2`, m^4.
    b_prime : float
        The coefficient of :math:`S_4`, m^4.
    leg_bound : float
        :math:`\nu_c`, m^-1, positive.
    band_edge_linear : float or None
        The band edge for one linear polarisation, m^-1; ``None`` where optmat's
        record leaves it undefined (CO2, CH4, SF6, N2O and O3 at optmat 0.4).
    band_edge_any : float or None
        The band edge for any polarisation, m^-1, or ``None``.
    material_records : tuple of MaterialRecord
        The records of optmat's hyperpolarizability model and of its support record,
        in that order.

    Raises
    ------
    MaterialError
        If a coefficient is not finite, the static coefficient or the leg bound is not
        positive, or the Kerr diagonal is not positive up to the leg bound, where its
        fourth root would not be real.

    Examples
    --------
    >>> from solprop.physdata.nonlinear_data import OptmatKerr
    >>> argon = OptmatKerr("Ar").dispersive_model()
    >>> round(1e9 / argon.leg_bound, 1), round(1e9 / argon.band_edge(vector=False), 1)
    (215.9, 189.4)
    """

    material: str
    gamma3_static: float
    a: float
    b: float
    b_prime: float
    leg_bound: float
    band_edge_linear: float | None
    band_edge_any: float | None
    material_records: tuple[MaterialRecord, ...]

    def __post_init__(self) -> None:
        """Check the coefficients and that the symmetric factor is real.

        Raises
        ------
        MaterialError
            Naming the offending coefficient.
        """
        values = (self.gamma3_static, self.a, self.b, self.b_prime, self.leg_bound)
        if not all(math.isfinite(value) for value in values):
            msg = f"{self.material}: the dispersive Kerr model has a non-finite value"
            raise MaterialError(msg)
        if self.gamma3_static <= 0.0 or self.leg_bound <= 0.0:
            msg = (
                f"{self.material}: the static hyperpolarizability and the leg bound "
                "must be positive"
            )
            raise MaterialError(msg)
        diagonal = self.kerr_ratio(np.linspace(0.0, self.leg_bound**2, 257))
        if not np.all(diagonal > 0.0):
            msg = (
                f"{self.material}: the Kerr diagonal of the fit is not positive up to "
                "the leg bound, so the symmetric factorisation has no real fourth root "
                "(SEP 6, section 4.4)"
            )
            raise MaterialError(msg)

    @classmethod
    def from_optmat(cls, model: object) -> "DispersiveKerrModel":  # noqa: UP037 - the hook resolves class names eagerly
        """Build the SI model of an optmat hyperpolarizability model.

        Parameters
        ----------
        model : optmat.gas.nonlinear.HyperpolarizabilityModel
            Any gas's model, one solprop builds as a species or not.

        Returns
        -------
        DispersiveKerrModel
            The model in SI with its leg bound and band edges.

        Raises
        ------
        MaterialError
            If the model carries no four dispersion parameters or no support record,
            or its support record describes another model.
        """
        from optmat.gas.nonlinear import (  # noqa: PLC0415 - optmat is imported lazily
            AU_HYPERPOLARIZABILITY,
        )

        handle = _optmat.untyped(model)
        identity = _optmat.record(model)
        parameters: tuple[float, ...] = tuple(
            float(value) for value in getattr(handle, "parameters", ())
        )
        support = getattr(handle, "support", None)
        if len(parameters) != 4 or support is None:
            msg = (
                f"{identity.material}: optmat's model {identity.name} carries no "
                "dispersion parameters and support record, which a dispersive "
                "electronic policy needs (SEP 6); it is refused rather than given a "
                "borrowed curve"
            )
            raise MaterialError(msg)
        if support.model_digest != identity.digest:
            msg = (
                f"{identity.material}: optmat's support record describes the model "
                f"{support.model_digest[:12]}, not {identity.name} "
                f"({identity.digest[:12]}); a modified model has no support record of "
                "its own"
            )
            raise MaterialError(msg)
        gamma0, a, b, b_prime = parameters
        return cls(
            material=identity.material,
            gamma3_static=gamma0 * AU_HYPERPOLARIZABILITY / (_TAYLOR_FACTOR * EPS0),
            a=a * _CM2_TO_M2,
            b=b * _CM4_TO_M4,
            b_prime=b_prime * _CM4_TO_M4,
            leg_bound=float(support.leg_max_in_w2),
            band_edge_linear=_edge(support.band_edge("linear")),
            band_edge_any=_edge(support.band_edge("any")),
            material_records=(identity, _optmat.record(support)),
        )

    @property
    def model_digest(self) -> str:
        """Return the digest of optmat's hyperpolarizability model.

        Returns
        -------
        str
            Sixty-four hexadecimal digits.
        """
        return self.material_records[0].digest

    @property
    def support_digest(self) -> str:
        """Return the digest of optmat's support record of the model.

        Returns
        -------
        str
            Sixty-four hexadecimal digits.
        """
        return self.material_records[1].digest

    def band_edge(self, *, vector: bool) -> float | None:
        """Return the band edge a run takes, m^-1.

        Parameters
        ----------
        vector : bool
            Whether the run carries two polarisation components, which takes the
            any-polarisation edge.

        Returns
        -------
        float or None
            The edge, or ``None`` where it is undefined.
        """
        return self.band_edge_any if vector else self.band_edge_linear

    def leg_weight(
        self, omega: Float[np.ndarray, "*bins"]
    ) -> Float[np.ndarray, "*bins"]:
        r"""Return the bounded leg weight at angular frequencies, m^-2.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s, of either sign, any shape.

        Returns
        -------
        numpy.ndarray
            :math:`\min((\omega/2\pi c)^2, \nu_c^2)`.
        """
        nu = np.asarray(omega, dtype=float) / (2.0 * math.pi * C)
        return np.minimum(nu**2, self.leg_bound**2)

    def kerr_ratio(
        self, weight: Float[np.ndarray, "*bins"]
    ) -> Float[np.ndarray, "*bins"]:
        r"""Return the Kerr diagonal over its static value at a leg weight.

        Parameters
        ----------
        weight : numpy.ndarray
            Leg weights :math:`\tilde d = \nu^2`, m^-2, any shape.

        Returns
        -------
        numpy.ndarray
            :math:`1 + 4 a \tilde d + (16 b + 4 b')\tilde d^2`, the diagonal
            :math:`S_2 = 4\tilde d`, :math:`S_4 = 4\tilde d^2`.
        """
        d = np.asarray(weight, dtype=float)
        return 1.0 + 4.0 * self.a * d + (16.0 * self.b + 4.0 * self.b_prime) * d**2

    def gamma3(
        self, legs: Float[np.ndarray, "*quartets 4"]
    ) -> Float[np.ndarray, "*quartets"]:
        r"""Return the bounded per-particle coefficient of quartets, m^5/V^2.

        Parameters
        ----------
        legs : numpy.ndarray
            Signed vacuum wavenumbers in m^-1, the last axis the quartet's four
            legs; only their squares enter.

        Returns
        -------
        numpy.ndarray
            :math:`\gamma^{(3)}_0[1 + a S_2 + b S_2^2 + b' S_4]` with every leg's
            square bounded at :math:`\nu_c^2`.
        """
        d = np.minimum(np.asarray(legs, dtype=float) ** 2, self.leg_bound**2)
        s2 = d.sum(-1)
        s4 = (d**2).sum(-1)
        return self.gamma3_static * (
            1.0 + self.a * s2 + self.b * s2**2 + self.b_prime * s4
        )

    def regions(
        self, omega_low: float, omega_high: float, *, vector: bool
    ) -> ElectronicRegions:
        """Classify a band of angular frequencies (SEP 6, section 4.10).

        Parameters
        ----------
        omega_low : float
            The lowest angular frequency of the band in rad/s, positive.
        omega_high : float
            The highest, in rad/s, above `omega_low`.
        vector : bool
            Whether the run carries two polarisation components, which takes the
            any-polarisation band edge.

        Returns
        -------
        ElectronicRegions
            The supported, bounded and best-effort parts of the band.

        Raises
        ------
        MaterialError
            If the band is not a positive, finite, increasing pair.

        Examples
        --------
        >>> import math
        >>> from solprop.constants import C
        >>> from solprop.physdata.nonlinear_data import OptmatKerr
        >>> helium = OptmatKerr("He").dispersive_model()
        >>> omega = [2 * math.pi * C / wavelength for wavelength in (4e-6, 90e-9)]
        >>> regions = helium.regions(*omega, vector=False)
        >>> [round(x * 1e9, 1) for x in regions.bounded]
        [120.3, 151.9]
        >>> regions.best_effort is not None
        True
        """
        if not (0.0 < omega_low < omega_high and math.isfinite(omega_high)):
            msg = (
                f"a band is a positive, finite, increasing pair of angular "
                f"frequencies, got ({omega_low!r}, {omega_high!r})"
            )
            raise MaterialError(msg)
        low = omega_low / (2.0 * math.pi * C)
        high = omega_high / (2.0 * math.pi * C)
        edge = self.band_edge(vector=vector)
        cap = self.leg_bound
        supported_top = cap if edge is None else min(cap, edge)
        effort_from = cap if edge is None else edge
        supported = (low, min(high, supported_top)) if low <= supported_top else None
        bounded = None
        if edge is not None and edge > cap and high > cap and low < edge:
            bounded = (max(low, cap), min(high, edge))
        best_effort = (max(low, effort_from), high) if high > effort_from else None
        return ElectronicRegions(
            material=self.material,
            polarisation="any" if vector else "linear",
            band=_wavelengths((low, high)),
            leg_bound=cap,
            band_edge=edge,
            supported=None if supported is None else _wavelengths(supported),
            bounded=None if bounded is None else _wavelengths(bounded),
            best_effort=None if best_effort is None else _wavelengths(best_effort),
        )


def _edge(value: float | None) -> float | None:
    """Return a band edge as a float, keeping an undefined one undefined.

    Parameters
    ----------
    value : float or None
        The edge optmat's record derives, m^-1.

    Returns
    -------
    float or None
        The edge as a Python float, or ``None``.
    """
    return None if value is None else float(value)


def _wavelengths(interval: tuple[float, float]) -> tuple[float, float]:
    """Return an interval of wavenumbers as its shortest and longest wavelengths.

    Parameters
    ----------
    interval : tuple of float
        Lower and upper vacuum wavenumber, m^-1, positive.

    Returns
    -------
    tuple of float
        ``(1 / upper, 1 / lower)``, m.
    """
    return (1.0 / interval[1], 1.0 / interval[0])


def selected_carrier(selection: MaterialSelection, wavelength: float | None) -> float:
    """Return the carrier a selection's Kerr coefficient is evaluated at.

    Parameters
    ----------
    selection : MaterialSelection
        The selection; its ``carrier_wavelength`` may be ``None``.
    wavelength : float or None
        The caller's wavelength in m, or ``None``.

    Returns
    -------
    float
        The one carrier in m both name.

    Raises
    ------
    MaterialError
        If neither names one, or the two disagree.

    Examples
    --------
    >>> from solprop.physdata import MaterialSelection
    >>> from solprop.physdata.nonlinear_data import selected_carrier
    >>> selected_carrier(MaterialSelection(carrier_wavelength=1030e-9), None)
    1.03e-06
    """
    chosen = selection.carrier_wavelength
    if wavelength is not None and chosen is not None:
        if not carriers_agree(float(wavelength), chosen):
            msg = (
                f"the selection's carrier_wavelength {chosen:g} m and the wavelength "
                f"{wavelength:g} m differ: one carrier sets the Kerr coefficient, so "
                "give the same wavelength, or leave the selection's unset"
            )
            raise MaterialError(msg)
        return float(wavelength)
    if wavelength is None and chosen is None:
        msg = (
            "optmat's Kerr coefficient is evaluated at a carrier wavelength: give "
            "wavelength=, or set the selection's carrier_wavelength"
        )
        raise MaterialError(msg)
    if wavelength is not None:
        return float(wavelength)
    assert chosen is not None
    return float(chosen)
