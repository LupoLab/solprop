# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Describe material laws and region ownership independently of longitudinal paths."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from jaxtyping import Complex
from numpy.typing import ArrayLike

from solprop._display import Displayed, engineering, line
from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.nonlinear.raman import SilicaTensorResponse
from solprop.physdata.glasses import GlassIndex, lossless_index
from solprop.physdata.mixtures import Mixture
from solprop.physdata.nonlinear_data import chi3_from_n2, n2_from_chi3
from solprop.profiles import ZProfile, as_profile

__all__ = [
    "MEASURED_DIGITS",
    "GasKerr",
    "IndexMaterial",
    "JsonValue",
    "MaterialKerr",
    "MaterialRegion",
    "MaterialResponse",
    "PermittivityLaw",
    "quantised_measurement",
]

# This module owns the package's structural JSON value domain, because it is the
# leaf both the modes package and `solprop.output.schema` can import: the schema
# imports `solprop.modes.protocol`, so defining the alias there instead would
# close an import cycle through the output layer.
type JsonValue = (
    bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
)
"""Anything :func:`solprop.output.schema.serialise` may return, and therefore
anything JSON can carry. Re-exported by that module, which owns the file format."""

MEASURED_DIGITS = 6
"""Significant digits a **measured** float keeps when it enters an identity.

A declared number -- a radius, a ``chi3``, a tolerance -- is the one the caller
wrote and enters an identity exactly. A *measured* one is the output of a dense
solve and a reduction, so its last bits depend on the BLAS that did them: the
same guide re-solved on another machine, or after a library update, gives a
Gram defect and a profile error that differ in the last few digits and nothing
else. Six digits is far more than any of them is accurate to (they are
tolerances of order 1e-5 gated against budgets of order 1e-4) and far coarser
than the spread a re-solve can produce, so quantising to it makes a resume key
and a reconstruction gate compare equal for a run that did not change.
"""


def quantised_measurement(value: float) -> float:
    """Return a measured float rounded to :data:`MEASURED_DIGITS` significant digits.

    For the identities a modal run records (its resume key and reconstruction gate):
    what is *measured* is quantised before it is recorded, so that a re-translation
    of the same guide does not refuse a resume it should accept. What is *declared*
    is never passed through here.

    Parameters
    ----------
    value : float
        The measured quantity, of any magnitude; ``nan`` and the infinities pass
        through unchanged.

    Returns
    -------
    float
        The same number to six significant digits.

    Examples
    --------
    >>> from solprop.modes.material import quantised_measurement
    >>> quantised_measurement(3.768575004590464e-05)
    3.76858e-05
    >>> quantised_measurement(0.0)
    0.0
    >>> quantised_measurement(float("inf"))
    inf
    """
    number = float(value)
    if not np.isfinite(number):
        return number
    return float(f"{number:.{MEASURED_DIGITS - 1}e}")


@runtime_checkable
class PermittivityLaw(Protocol):
    """Describe an optical permittivity law independently of longitudinal paths.

    Named for the law rather than for a material, because
    :class:`solprop.optics.Material` is a length of a real medium in the flat
    ``solprop.sol`` namespace and the two must not be confused.
    """

    def permittivity(
        self, omega: ArrayLike
    ) -> Complex[np.ndarray, " n_omega_permittivity"]:
        """Evaluate relative permittivity in the optics convention.

        Parameters
        ----------
        omega : array_like
            Positive angular frequencies in rad/s, shape (n_omega_permittivity,).

        Returns
        -------
        numpy.ndarray
            Dimensionless complex permittivity with the same shape.
        """
        ...

    @property
    def omega_domain(self) -> tuple[float, float]:
        """Return the material validity interval in rad/s."""
        ...

    @property
    def identity(self) -> Mapping[str, JsonValue]:
        """Return the material state and source identity."""
        ...


@dataclass(frozen=True)
class IndexMaterial:
    """Adapt an existing glass law or a positive constant refractive index.

    The law of a general modal guide's material region, which both providers this
    package ships solve without loss: an optmat index
    (:class:`~solprop.physdata.glasses.OptmatGlass`) is therefore held as its
    transparent approximation (:func:`~solprop.physdata.glasses.lossless_index`),
    the counterpart a :class:`~solprop.modes.stepindex.StepIndexMode` of the same
    adapter holds, with the dropped ``k`` recorded for a run's manifest (SEP 3, Q3).
    Any other glass is taken as it is.

    Attributes
    ----------
    index : GlassIndex or float
        Optical refractive-index law, or positive dimensionless constant.
    """

    index: GlassIndex | float

    def __post_init__(self) -> None:
        """Validate the constant material index, and take an optmat index lossless.

        Raises
        ------
        ModeError
            If a constant index is not positive and finite.
        """
        if not isinstance(self.index, GlassIndex) and (
            not np.isfinite(self.index) or self.index <= 0
        ):
            raise ModeError("material index must be positive and finite")
        object.__setattr__(self, "index", lossless_index(self.index))

    def permittivity(
        self, omega: ArrayLike
    ) -> Complex[np.ndarray, " n_omega_permittivity"]:
        """Evaluate the optical relative permittivity.

        Parameters
        ----------
        omega : array_like
            Positive frequencies in rad/s, shape (n_omega_permittivity,).

        Returns
        -------
        numpy.ndarray
            Complex relative permittivity, dimensionless, with the input shape.

        Raises
        ------
        ModeError
            If frequencies are malformed or outside the material domain.
        """
        values = np.asarray(omega, dtype=float)
        if (
            values.ndim != 1
            or not values.size
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ModeError("material frequencies must be a finite positive vector")
        if np.any(values < self.omega_domain[0]) or np.any(
            values > self.omega_domain[1]
        ):
            raise ModeError("frequencies are outside the material domain")
        index = (
            self.index.index(values)
            if isinstance(self.index, GlassIndex)
            else np.full(values.shape, self.index, complex)
        )
        return np.asarray(index**2, dtype=complex)

    @property
    def omega_domain(self) -> tuple[float, float]:
        """Return the validity interval in rad/s.

        Returns
        -------
        tuple of float
            Lower and upper angular frequency in rad/s.
        """
        if isinstance(self.index, GlassIndex):
            low, high = self.index.wavelength_range
            return 2 * np.pi * C / high, 2 * np.pi * C / low
        return 0.0, float("inf")

    @property
    def identity(self) -> Mapping[str, JsonValue]:
        """Return the material model and its validity interval.

        Returns
        -------
        collections.abc.Mapping
            Serializable physical model and construction metadata.
        """
        return {
            "index": self.index
            if not isinstance(self.index, GlassIndex)
            else repr(self.index),
            "omega_domain": [v if np.isfinite(v) else None for v in self.omega_domain],
        }


def _radius_text(value: float | ZProfile | None) -> str:
    """Return a printable radius without evaluating a declared path.

    Parameters
    ----------
    value : float or ZProfile or None
        Radius in m, a declared path, or None for the unbounded exterior.

    Returns
    -------
    str
        An engineering radius, the path's one-line description, or ``unbounded``.
    """
    if value is None:
        return "unbounded"
    if isinstance(value, ZProfile):
        return repr(value)
    return engineering(float(value), "m")


@dataclass(frozen=True)
class MaterialKerr(Displayed):
    """Declare a material's instantaneous isotropic Kerr response.

    The nonlinear half of a :class:`MaterialRegion`: the region owns its response
    and the general modal route reads it from there
    (:func:`solprop.translate.responses.regional_responses`), rather than
    taking a response list beside the guide. It is the absolute susceptibility of
    a condensed material, not a per-particle hyperpolarisability: a general guide
    carries no density path and the route's density is the dimensionless one.

    Give **one** of `chi3` and `n2`. ``chi3`` is used as it stands. ``n2`` is
    converted with ``chi3 = 4 eps0 c n_0^2 n_2 / 3``
    (:func:`solprop.physdata.nonlinear_data.chi3_from_n2`), where ``n_0`` is the
    region's **own** linear index at the wavelength the ``n2`` was measured at, and
    the result is then constant across the band, which is what the compiled
    per-region response carries.

    **An ``n2`` declares its wavelength.** ``at=`` is that wavelength in metres --
    the one the published number was measured at, 1550 nm for a telecom fibre
    figure and 800 nm for most gas measurements -- and it is where the linear index
    of the conversion is read. ``at=None``, the default, reads it at the frequency
    the caller asks for, which in a run is the run's own reference frequency. The
    two differ: a silica ``n2`` declared at 800 nm and at 2 um give ``chi3`` values
    2.13 % apart, the square of the index ratio, and nothing downstream can tell
    them apart afterwards because the leaf carries one number. A region whose index
    varies appreciably across the run's active band is therefore **refused** an
    ``n2`` with no ``at`` by
    :func:`solprop.translate.general.general_modal_route`, rather than converted at
    a frequency nobody declared.

    **The region's own law governs the conversion**, and not the index the
    channels were solved with: the conversion asks
    :attr:`MaterialRegion.material` for its permittivity and nothing else. The
    two must therefore be the same material, which
    :class:`solprop.modes.guide.StepIndexProvider` checks at construction against
    :data:`solprop.modes.guide.MATERIAL_AGREEMENT_RTOL` -- a core region declared
    at 2.0 beside channels solved at 1.45 would otherwise convert a declared
    ``n2`` to a ``chi3`` 90 % too large.

    Attributes
    ----------
    chi3 : float or None
        Absolute third-order susceptibility in m²/V², finite and non-negative;
        ``None`` when `n2` is given.
    n2 : float or None
        Nonlinear index in m²/W in the convention where an intensity ``I`` shifts
        the refractive index by ``n2 I``, finite and non-negative; ``None`` when
        `chi3` is given.
    at : float or None
        Wavelength in m the `n2` was **measured at**, positive and finite, which
        is where the linear index of the conversion is read. ``None`` reads it at
        the frequency asked for. Refused beside `chi3`, which is absolute and
        declares no measurement wavelength.

    Raises
    ------
    ModeError
        If neither or both coefficients are given, one is negative or not finite,
        `at` is not positive and finite, or `at` is given beside `chi3`.

    Examples
    --------
    >>> from solprop.modes.material import IndexMaterial, MaterialKerr
    >>> from solprop.physdata import fused_silica
    >>> silica = IndexMaterial(fused_silica())
    >>> f"{MaterialKerr(chi3=2.2e-22).susceptibility(silica, 2.3e15):.4e}"
    '2.2000e-22'
    >>> f"{MaterialKerr(n2=2.6e-20).susceptibility(silica, 2.3e15):.4e}"
    '1.9438e-22'
    >>> f"{MaterialKerr(n2=2.6e-20, at=1550e-9).susceptibility(silica, 2.3e15):.4e}"
    '1.9198e-22'
    """

    chi3: float | None = None
    n2: float | None = None
    at: float | None = None

    def __post_init__(self) -> None:
        """Require exactly one non-negative, finite coefficient, and check `at`.

        Raises
        ------
        ModeError
            If neither or both are given, the one given is unusable, or the
            declared measurement wavelength is unusable or beside a ``chi3``.
        """
        given = [name for name in ("chi3", "n2") if getattr(self, name) is not None]
        if len(given) != 1:
            raise ModeError(
                "a material Kerr response is declared by chi3 or by n2, exactly "
                f"one of them, got {given or 'neither'}"
            )
        value = float(getattr(self, given[0]))
        if not np.isfinite(value) or value < 0.0:
            raise ModeError(
                f"{given[0]} must be non-negative and finite, got {value!r}"
            )
        if self.at is None:
            return
        if self.chi3 is not None:
            raise ModeError(
                "at= is the wavelength an n2 was measured at, and this response "
                "declares chi3, which is an absolute susceptibility and is used "
                "as it stands: drop at=, or declare n2 instead"
            )
        if not np.isfinite(self.at) or self.at <= 0.0:
            raise ModeError(
                f"at= is a wavelength in metres, positive and finite, got {self.at!r}"
            )

    def __repr__(self) -> str:
        """Return a one-line description: the coefficient that was declared.

        Returns
        -------
        str
            For example ``MaterialKerr(chi3=2.2e-22 m^2/V^2)``.
        """
        declared = (
            ("chi3", self.chi3, "m^2/V^2")
            if self.chi3 is not None
            else ("n2", self.n2, "m^2/W")
        )
        rows = [("", f"{declared[0]}={declared[1]:.4g} {declared[2]}")]
        if self.at is not None:
            rows.append(("", f"at {engineering(float(self.at), 'm')}"))
        return line("MaterialKerr", rows)

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "MaterialKerr: an instantaneous isotropic Kerr response", [
            ("declared as", "chi3" if self.chi3 is not None else "n2"),
            (
                "chi3",
                (
                    "from n2, at the frequency asked for"
                    if self.at is None
                    else f"from n2, at {engineering(float(self.at), 'm')}"
                )
                if self.chi3 is None
                else f"{self.chi3:.4g} m^2/V^2",
            ),
            (
                "n2",
                "from chi3, at the frequency asked for"
                if self.n2 is None
                else f"{self.n2:.4g} m^2/W",
            ),
            (
                "measured at",
                "the frequency asked for"
                if self.at is None
                else engineering(float(self.at), "m"),
            ),
        ]

    def susceptibility(self, material: PermittivityLaw, omega: float) -> float:
        """Return the absolute third-order susceptibility at one frequency.

        Parameters
        ----------
        material : PermittivityLaw
            The region's own law, which supplies the linear index the ``n2``
            conversion needs. Unused when ``chi3`` was declared.
        omega : float
            Positive angular frequency in rad/s inside the material's domain, at
            which the conversion is made when the response declares no `at`.

        Returns
        -------
        float
            ``chi3`` in m²/V², non-negative.

        Raises
        ------
        ModeError
            When ``n2`` was declared and the frequency is not positive and finite,
            the declared `at` lies outside the material's own domain, or the
            material's index there is not a positive real number. A declared
            ``chi3`` is returned without touching `material` or `omega` and raises
            nothing.
        """
        if self.chi3 is not None:
            return float(self.chi3)
        frequency = self._conversion(omega)
        try:
            index = _linear_index(material, frequency)
        except ModeError as outside:
            if self.at is None:
                raise
            # A declared measurement wavelength is a caller's number and can sit
            # outside the region's own index model; the generic refusal names the
            # frequency and not the declaration it came from.
            msg = (
                f"this Kerr response declares its n2 measured at "
                f"{self.at:.6e} m, which is {frequency:.6e} rad/s, and the "
                f"region's material cannot supply a linear index there: {outside}"
            )
            raise ModeError(msg) from outside
        return chi3_from_n2(float(self.n2 or 0.0), index)

    def _conversion(self, omega: float) -> float:
        """Return the frequency the ``n2`` conversion reads the index at, rad/s.

        Parameters
        ----------
        omega : float
            The frequency asked for, used when no measurement wavelength is
            declared.

        Returns
        -------
        float
            ``2 pi c / at`` when `at` is declared, and `omega` otherwise.
        """
        return float(omega) if self.at is None else 2.0 * np.pi * C / float(self.at)

    def nonlinear_index(self, material: PermittivityLaw, omega: float) -> float:
        """Return the nonlinear index at one frequency, whichever was declared.

        Parameters
        ----------
        material : PermittivityLaw
            The region's own law, which supplies the linear index.
        omega : float
            Positive angular frequency in rad/s inside the material's domain, at
            which a declared ``chi3`` is converted back.

        Returns
        -------
        float
            ``n2`` in m²/W, non-negative. A declared ``n2`` is returned as it
            stands, at the wavelength it declares.

        Raises
        ------
        ModeError
            If the frequency is not positive and finite, or the material's index
            there is not a positive real number.
        """
        if self.n2 is not None:
            return float(self.n2)
        return n2_from_chi3(float(self.chi3 or 0.0), _linear_index(material, omega))


def _linear_index(material: PermittivityLaw, omega: float) -> float:
    """Return a material's real linear refractive index at one frequency.

    Parameters
    ----------
    material : PermittivityLaw
        The law to evaluate.
    omega : float
        Positive angular frequency in rad/s inside the material's domain.

    Returns
    -------
    float
        ``n_0 = sqrt(Re eps_r)``, dimensionless and positive.

    Raises
    ------
    ModeError
        If the frequency or the resulting index is unusable.
    """
    if not np.isfinite(omega) or omega <= 0.0:
        raise ModeError(f"omega must be positive and finite, got {omega!r}")
    permittivity = float(
        np.real(material.permittivity(np.array([float(omega)], dtype=float))[0])
    )
    if not np.isfinite(permittivity) or permittivity <= 0.0:
        raise ModeError(
            f"{type(material).__name__} has relative permittivity "
            f"{permittivity!r} at {omega!r} rad/s, which is not a linear index a "
            "nonlinear coefficient can be converted against"
        )
    return float(np.sqrt(permittivity))


@dataclass(frozen=True)
class GasKerr(Displayed):
    """Declare a region's gas fill as the source of its electronic cubic response.

    Where :class:`MaterialKerr` is one absolute ``chi3`` the caller states, this is
    the fill's own: each species' optmat hyperpolarizability model at its number
    density, resolved by the run's material selection
    (:func:`solprop.nonlinear.mixtures.electronic_response`). Under a carrier
    electronic policy the region's response is the constant isotropic cubic of
    ``chi3 = sum_s N_s gamma3_s(lambda_0)``, the Kerr coefficient at the run's
    carrier, as every other route evaluates a gas; under one of SEP 6's dispersive
    policies it is each species' dispersive cubic, the scalar kernel times the
    isotropic Kleinman tensor
    (``docs/dev/seps/archive/sep-0006-dispersive-nonlinearity.md``).

    The fill is the region's nonlinear medium only: the region's linear law is its
    own ``material``, as for every other region. A general modal guide has constant
    geometry, so every density of the fill must be constant along z.

    Attributes
    ----------
    fill : solprop.physdata.mixtures.Mixture
        The gas: its species and their constant number densities, m^-3.

    Raises
    ------
    ModeError
        If a density of the fill varies along z.

    Examples
    --------
    >>> from solprop.modes.material import GasKerr
    >>> from solprop.physdata import Mixture
    >>> GasKerr(Mixture.fill({"Ar": 1.0}, 2.0)).fill.labels
    ('Ar',)
    """

    fill: Mixture

    def __post_init__(self) -> None:
        """Require constant densities.

        Raises
        ------
        ModeError
            Naming the population whose density varies along z.
        """
        for component in self.fill.components:
            if not component.density.is_constant:
                msg = (
                    "a region's gas fill propagates on a general modal guide of "
                    f"constant geometry, and the density of {component.label!r} "
                    "varies along z: give it a constant pressure"
                )
                raise ModeError(msg)

    def __repr__(self) -> str:
        """Return a one-line description: the fill's populations.

        Returns
        -------
        str
            For example ``GasKerr(Ar)``.
        """
        return line("GasKerr", [("", ", ".join(self.fill.labels))])

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "GasKerr: the electronic response of a region's gas fill", [
            ("fill", repr(self.fill)),
            ("law", "the run's electronic policy"),
        ]

    def densities(self) -> tuple[float, ...]:
        """Return each population's number density, in mixture order.

        Returns
        -------
        tuple of float
            Constant densities in m^-3.
        """
        return tuple(float(value) for value in self.fill.density(np.zeros(1))[:, 0])


type MaterialResponse = MaterialKerr | GasKerr | SilicaTensorResponse
"""What a material region may declare: the instantaneous cubic, the delayed tensor.

At most one of each kind, and one electronic response: :class:`MaterialKerr`, which
carries the region's absolute ``chi3``, or :class:`GasKerr`, its gas fill's.
:class:`~solprop.nonlinear.raman.SilicaTensorResponse` takes the share ``f_R`` of a
:class:`MaterialKerr`'s ``chi3`` and leaves ``1 - f_R`` to the instantaneous term, so a
region that declares the delayed response without one has no susceptibility to share
and is refused by name.
"""


@dataclass(frozen=True)
class MaterialRegion(Displayed):
    """Assign one material and response list to an annular region.

    Attributes
    ----------
    name : str
        Non-empty region name.
    inner_radius : float or ZProfile
        Non-negative inner radius in m, possibly following a declared path.
    outer_radius : float or ZProfile or None
        Outer radius in m; None denotes the unbounded exterior.
    material : PermittivityLaw
        Relative permittivity and immutable material identity.
    responses : tuple of MaterialKerr, GasKerr or SilicaTensorResponse
        The region's own declared physical responses, at most one of each kind and
        one electronic response, :class:`MaterialKerr` or :class:`GasKerr`.
        Empty declares a **linear** region: its nodes have no polarisation, which
        is how a linear cladding is said to be linear. A
        :class:`~solprop.nonlinear.raman.SilicaTensorResponse` beside the
        :class:`MaterialKerr` makes the region's response the standard fibre pair,
        ``1 - f_R`` instantaneous and ``f_R`` delayed, of one ``chi3``.
    """

    name: str
    inner_radius: float | ZProfile
    outer_radius: float | ZProfile | None
    material: PermittivityLaw
    responses: tuple[MaterialResponse, ...] = ()

    def __repr__(self) -> str:
        """Return a one-line description: the name, the radii and the law.

        Returns
        -------
        str
            For example ``MaterialRegion(core, 0 m to 250 nm, IndexMaterial)``.
        """
        return line(
            "MaterialRegion",
            [
                ("", self.name),
                (
                    "",
                    f"{_radius_text(self.inner_radius)} to "
                    f"{_radius_text(self.outer_radius)}",
                ),
                ("", type(self.material).__name__),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return f"MaterialRegion: {self.name}", [
            ("inner", _radius_text(self.inner_radius)),
            ("outer", _radius_text(self.outer_radius)),
            ("material", type(self.material).__name__),
            (
                "responses",
                ", ".join(repr(one) for one in self.responses) or "none (linear)",
            ),
        ]

    @property
    def raman(self) -> SilicaTensorResponse | None:
        """Return the region's declared delayed response, or ``None``.

        Returns
        -------
        solprop.nonlinear.raman.SilicaTensorResponse or None
            The one declared delayed response, or ``None``.
        """
        declared = [
            one for one in self.responses if isinstance(one, SilicaTensorResponse)
        ]
        return declared[0] if declared else None

    @property
    def kerr(self) -> MaterialKerr | None:
        """Return the region's declared Kerr response, or ``None`` if it is linear.

        Returns
        -------
        MaterialKerr or None
            The one declared instantaneous response, or ``None``.
        """
        declared = [one for one in self.responses if isinstance(one, MaterialKerr)]
        return declared[0] if declared else None

    @property
    def gas(self) -> GasKerr | None:
        """Return the region's gas fill response, or ``None``.

        Returns
        -------
        GasKerr or None
            The one declared gas fill, or ``None``.
        """
        declared = [one for one in self.responses if isinstance(one, GasKerr)]
        return declared[0] if declared else None

    def __post_init__(self) -> None:
        """Validate region names, entrance radii and the declared responses.

        Raises
        ------
        ModeError
            If the region has an empty name, invalid entrance radii, a response
            this phase does not carry, or two responses of one kind.
        """
        admitted = (MaterialKerr, GasKerr, SilicaTensorResponse)
        unknown = [one for one in self.responses if not isinstance(one, admitted)]
        if unknown:
            raise ModeError(
                f"material region {self.name!r} declares "
                f"{[type(one).__name__ for one in unknown]}, and a region carries "
                "the instantaneous MaterialKerr or GasKerr response and the delayed "
                "SilicaTensorResponse, at most one of each"
            )
        if self.kerr is not None and self.gas is not None:
            raise ModeError(
                f"material region {self.name!r} declares a MaterialKerr and a "
                "GasKerr: a region has one electronic response, a declared chi3 or "
                "its gas fill's"
            )
        for kind in admitted:
            if sum(isinstance(one, kind) for one in self.responses) > 1:
                raise ModeError(
                    f"material region {self.name!r} declares more than one "
                    f"{kind.__name__}; one material has one"
                )
        if self.raman is not None and self.kerr is None:
            raise ModeError(
                f"material region {self.name!r} declares a SilicaTensorResponse "
                "and no MaterialKerr: the delayed tensor model is the share f_R "
                "of the region's own chi3 and the instantaneous term keeps "
                "1 - f_R, so there is no susceptibility for it to be a share of. "
                "Declare MaterialKerr(chi3=...) or MaterialKerr(n2=..., at=...) "
                "beside it"
            )
        lower = float(as_profile(self.inner_radius)(np.array([0.0]))[0])
        upper = (
            float("inf")
            if self.outer_radius is None
            else float(as_profile(self.outer_radius)(np.array([0.0]))[0])
        )
        if (
            not self.name
            or not np.isfinite(lower)
            or lower < 0
            or upper <= lower
            or np.isnan(upper)
        ):
            raise ModeError(
                "material region needs a name and increasing non-negative radii"
            )
