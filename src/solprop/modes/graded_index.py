# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Solve radial TE/TM Maxwell sectors and a separately labelled LP approximation.

The symmetric variational problem uses continuous linear elements on a declared
radial domain matched to an exact decaying exterior. Hybrid HE/EH sectors are
refused: TE/TM evidence cannot
qualify hybrid GRIN. The exterior material is explicit and the truncation radius
is refined independently of the radial mesh.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Literal, Protocol, cast, runtime_checkable

import numpy as np
from jaxtyping import Complex, Float
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.special import kve, roots_legendre

from solprop._display import Displayed, line
from solprop._process import host_blas
from solprop.constants import EPS0, C
from solprop.exceptions import ModeError
from solprop.modes.guide import radial_node_rule
from solprop.modes.material import (
    JsonValue,
    MaterialRegion,
    MaterialResponse,
    PermittivityLaw,
    _radius_text,
)
from solprop.modes.protocol import ChannelLabel
from solprop.modes.vector import ModalNodeRule, ModalSample, flux_product
from solprop.profiles import ZProfile, as_profile, profile_identity

__all__ = [
    "MAX_EXTERIOR_TAIL_FRACTION",
    "GradedIndexProvider",
    "PowerLawPermittivity",
    "RadialPermittivity",
]

MAX_EXTERIOR_TAIL_FRACTION = 0.0001
"""Largest share of a channel's power admitted beyond the matching radius.

The near-cutoff guard for the radial provider, the counterpart of
:data:`solprop.modes.stepindex.PROFILE_MIN_W`. It is written as a power fraction
rather than as the dimensionless decay ``gamma * extent * radius`` because that
decay mixes the physics with the caller's numerical choice of `extent`: the same
mode has ``gamma * extent * radius`` of 26.5, 35.3 and 44.1 at ``extent`` 1.5, 2
and 2.5. The fraction is the quantity that says how much of the mode the exact
exterior continuation has to carry, and the provider evaluates it exactly, from
the analytic modified-Bessel tail integral over the same full-domain norm the
fields are normalised by.

Measured at 0.0001: over every reference case of the radial-provider qualification --
both sectors, both radial laws, 24 to 96 elements, matching radii 1.5/2/2.5 and
contrasts 0.001/0.005/0.0098 -- the largest fraction is 6.57e-6, a factor 15
below the floor, and the weakest reference channel is the second LP channel at
contrast 0.001. A five-channel TE case at 48 elements puts 5.97e-8 of its last
channel outside the matching radius at a 25.00 um core and 3.24e-4 at 22.98 um,
so the floor refuses that channel with a factor 3.2 of margin.

The value is a judgement, not a derived bound: it is kept at 1e-4 pending
verification against an observable -- an overlap or a propagation constant of a
near-cutoff channel converged in the matching radius -- which none of those reference
cases supplies. Until then a refusal near this floor is a
prompt to widen ``extent``, not a physical statement about the channel.
"""


@runtime_checkable
class RadialPermittivity(Protocol):
    """Evaluate a radial material law independently of a longitudinal path."""

    def permittivity(
        self, omega: float, radial_fraction: Float[np.ndarray, " n_points"]
    ) -> Complex[np.ndarray, " n_points"]:
        """Evaluate the core's relative permittivity.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s.
        radial_fraction : numpy.ndarray
            Dimensionless radius divided by the declared core radius, in [0, 1].

        Returns
        -------
        numpy.ndarray
            Relative permittivity, shape (n_points,), in the optics convention.
        """
        ...

    @property
    def material(self) -> PermittivityLaw:
        """Return the material owning core responses."""
        ...

    @property
    def identity(self) -> Mapping[str, JsonValue]:
        """Return the law's physical and approximation identity."""
        ...

    @property
    def omega_domain(self) -> tuple[float, float]:
        """Return the material frequency interval in rad/s."""
        ...


@dataclass(frozen=True)
class PowerLawPermittivity:
    """Describe a finite radial power-law permittivity profile.

    Attributes
    ----------
    centre : PermittivityLaw
        Material law at the axis; its relative permittivity multiplies the shape.
    contrast : float
        Dimensionless contrast Delta in (0, 0.5); eps=eps_axis*(1-2 Delta rho**p).
    exponent : float
        Shape exponent p>=2; p=2 is parabolic and p=4 is quartic.
    """

    centre: PermittivityLaw
    contrast: float
    exponent: float = 2.0

    def __post_init__(self) -> None:
        """Validate the finite radial law.

        Raises
        ------
        ModeError
            If contrast or exponent is outside the declared range.
        """
        if (
            not 0 < self.contrast < 0.5
            or not np.isfinite(self.exponent)
            or self.exponent < 2
        ):
            raise ModeError("radial contrast must lie in (0, 0.5), exponent >= 2")

    def permittivity(
        self, omega: float, radial_fraction: Float[np.ndarray, " n_points"]
    ) -> Complex[np.ndarray, " n_points"]:
        """Evaluate the radial relative permittivity.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s within the centre material's domain.
        radial_fraction : numpy.ndarray
            Dimensionless core positions in [0, 1], shape (n_points,).

        Returns
        -------
        numpy.ndarray
            Complex relative permittivity, shape (n_points,).

        Raises
        ------
        ModeError
            If a core position is non-finite or outside [0, 1].
        """
        if (
            not np.all(np.isfinite(radial_fraction))
            or np.any(radial_fraction < 0)
            or np.any(radial_fraction > 1)
        ):
            raise ModeError("radial_fraction must lie in [0, 1]")
        return np.asarray(
            self.centre.permittivity(np.array([omega], dtype=float))[0]
            * (1 - 2 * self.contrast * radial_fraction**self.exponent),
            dtype=complex,
        )

    @property
    def omega_domain(self) -> tuple[float, float]:
        """Return the centre material's frequency interval in rad/s.

        Returns
        -------
        tuple of float
            Lower and upper angular frequency in rad/s.
        """
        return self.centre.omega_domain

    @property
    def material(self) -> PermittivityLaw:
        """Return the material owning core responses.

        Returns
        -------
        PermittivityLaw
            Material law owning the core response declarations.
        """
        return self.centre

    @property
    def identity(self) -> Mapping[str, JsonValue]:
        """Return the declared centre, contrast and radial exponent.

        Returns
        -------
        collections.abc.Mapping
            Serializable physical model and construction metadata.
        """
        return {
            "centre": dict(self.centre.identity),
            "contrast": self.contrast,
            "exponent": self.exponent,
        }


@dataclass(frozen=True, eq=False)
class _RadialSolution:
    """Retain the radial mesh, eigenfunctions and physical residuals."""

    nodes: Float[np.ndarray, " n_radial_nodes"]
    profiles: Float[np.ndarray, "n_radial_profiles n_modes_profiles"]
    beta: Float[np.ndarray, " n_modes_beta"]
    residual: Float[np.ndarray, " n_modes_residual"]


@dataclass(frozen=True, eq=False, kw_only=True)
class GradedIndexProvider(Displayed):
    """Solve bound radial vector sectors on an explicitly refined finite domain.

    A channel too close to cutoff for the declared finite domain is refused by
    name: more than :data:`MAX_EXTERIOR_TAIL_FRACTION` of its power beyond the
    matching radius means the exact exterior continuation, not the discretised
    core, carries the mode. A returned
    :class:`~solprop.modes.vector.ModalSample` is therefore always fully valid --
    ``valid`` is all ``True`` -- because an unsupported channel is refused rather
    than masked.

    Attributes
    ----------
    permittivity : RadialPermittivity
        Core law without a z argument, evaluated at the local radial fraction.
    exterior : PermittivityLaw
        Homogeneous exterior material law, independently declared.
    radius : float or ZProfile
        Positive core radius in m; all z dependence follows this named path.
    family : {'TE', 'TM', 'LP', 'HE', 'EH'}
        TE0/TM0 Maxwell sector or explicitly weak-guidance scalar LP approximation.
        Only ``TE``, ``TM`` and ``LP`` are supported. ``HE`` and ``EH`` stay in the
        type so that construction reaches the refusal and can be tested there;
        passing either raises rather than silently solving a hybrid sector.
    count : int
        Number of consecutive radial channels, at least one.
    elements : int
        At least eight radial elements per core radius; must be refined.
    extent : float
        Dimensionless matching radius, greater than one; exact Bessel exterior.
    azimuthal_order : int
        Zero for TE/TM; non-negative scalar angular order for LP.
    core_responses : tuple
        The **core** region's own declared physical responses, at most one of each
        kind, exactly as :class:`solprop.modes.material.MaterialRegion` takes them.
        Empty, the default, declares a linear guide: its nodes carry no
        polarisation, which is what a linear self-imaging reference wants. The
        exterior is always linear, because the radial sectors this provider solves
        are bound and their exterior is the exact Bessel continuation -- what
        little field is there is the tail :data:`MAX_EXTERIOR_TAIL_FRACTION`
        already bounds, and a response declared on it would be evaluated at nodes
        the rule barely resolves.
    """

    permittivity: RadialPermittivity
    exterior: PermittivityLaw
    radius: float | ZProfile
    family: Literal["TE", "TM", "LP", "HE", "EH"]
    count: int
    elements: int
    extent: float
    azimuthal_order: int = 0
    core_responses: tuple[MaterialResponse, ...] = ()

    def __repr__(self) -> str:
        """Return a one-line description: sector, channels and discretisation.

        Returns
        -------
        str
            For example ``GradedIndexProvider(TE0, 2 channels, 32 elements,
            extent=2)``.
        """
        return line(
            "GradedIndexProvider",
            [
                ("", f"{self.family}{self.azimuthal_order}"),
                ("", f"{self.count} channel{'' if self.count == 1 else 's'}"),
                ("", f"{self.elements} elements"),
                ("extent", f"{self.extent:g}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "GradedIndexProvider: radial P1 elements, exact Bessel exterior", [
            ("sector", f"{self.family}{self.azimuthal_order}"),
            (
                "approximation",
                "weak-guidance LP" if self.family == "LP" else "vector sector",
            ),
            ("channels", self.count),
            ("radius", _radius_text(self.radius)),
            ("elements", self.elements),
            ("extent", f"{self.extent:g} core radii"),
            ("exterior", type(self.exterior).__name__),
        ]

    def __post_init__(self) -> None:
        """Validate the supported sector and explicit discretisation.

        Raises
        ------
        ModeError
            If a hybrid sector or invalid discretisation is requested.
        """
        if self.family not in ("TE", "TM", "LP"):
            raise ModeError(
                "hybrid GRIN HE/EH sectors are unqualified; "
                "select TE0, TM0 or labelled LP"
            )
        if self.azimuthal_order < 0 or (
            self.family != "LP" and self.azimuthal_order != 0
        ):
            raise ModeError("TE/TM require azimuthal order zero")
        if (
            self.elements < 8
            or self.count < 1
            or self.count >= self.elements
            or not np.isfinite(self.extent)
            or self.extent <= 1
        ):
            raise ModeError(
                "require elements >= 8, 1 <= count < elements and extent > 1"
            )
        self._radius(0.0)

    def _radius(self, z: float) -> float:
        """Evaluate and validate the declared radius path.

        Parameters
        ----------
        z : float
            Position in m within the declared path.

        Returns
        -------
        float
            Positive core radius in m.
        """
        path = as_profile(self.radius)
        value = float(path(np.array([z], dtype=float))[0])
        if (
            not np.isfinite(z)
            or not path.z_span[0] <= z <= path.z_span[1]
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ModeError(
                "radius path must be positive and finite inside its z domain"
            )
        return value

    @property
    def parameter_paths(self) -> Mapping[str, ZProfile]:
        """Return the declared core-radius path in m.

        Returns
        -------
        collections.abc.Mapping
            The core_radius path in m, named as every other
            :class:`~solprop.modes.protocol.ParameterisedMode` names it.
        """
        return {"core_radius": as_profile(self.radius)}

    def at_parameters(self, values: Mapping[str, float]) -> "GradedIndexProvider":  # noqa: UP037 - runtime hook evaluates class annotations eagerly
        """Freeze the declared radius path at explicit parameter values.

        Parameters
        ----------
        values : collections.abc.Mapping
            Any subset of core_radius, in m and strictly positive. A name left
            out keeps its path; a name this provider does not declare is an error.

        Returns
        -------
        GradedIndexProvider
            The same physical law and discretisation with a constant radius.

        Raises
        ------
        ModeError
            If a name is unknown or the radius is invalid.
        """
        unknown = set(values) - {"core_radius"}
        if unknown:
            raise ModeError(f"unknown radial provider parameters: {unknown}")
        return replace(self, radius=values.get("core_radius", self.radius))

    @property
    def regions(self) -> tuple[MaterialRegion, ...]:
        """Return radial-core ownership and the homogeneous exterior.

        Returns
        -------
        tuple of MaterialRegion
            Ordered core and unbounded exterior ownership.
        """
        # The radial law owns the core material values, not a fictitious bulk index.
        return (
            MaterialRegion(
                "core",
                0.0,
                self.radius,
                self.permittivity.material,
                tuple(self.core_responses),
            ),
            MaterialRegion("exterior", self.radius, None, self.exterior),
        )

    @property
    def omega_domain(self) -> tuple[float, float]:
        """Return the common material frequency interval in rad/s.

        Returns
        -------
        tuple of float
            Lower and upper angular frequency in rad/s.
        """
        return max(
            self.permittivity.omega_domain[0], self.exterior.omega_domain[0]
        ), min(self.permittivity.omega_domain[1], self.exterior.omega_domain[1])

    @property
    def z_domain(self) -> tuple[float, float]:
        """Return the radius-path interval in m.

        Returns
        -------
        tuple of float
            Lower and upper position in m.
        """
        return as_profile(self.radius).z_span

    @property
    def z_breakpoints(self) -> tuple[float, ...]:
        """Return physical radius-path breakpoints in m.

        Returns
        -------
        tuple of float
            Positions of physical path kinks in m.
        """
        return tuple(float(v) for v in as_profile(self.radius).breakpoints)

    @property
    def channel_labels(self) -> tuple[ChannelLabel, ...]:
        """Return one label per requested radial channel, in solve order.

        Returns
        -------
        tuple of ChannelLabel
            The sector's family and azimuthal order with radial order ``1..count``,
            in the order :meth:`solve` returns the channels, which is decreasing
            propagation constant. A radial sector has no parity partner and no
            handedness, so both are ``None``.

        Examples
        --------
        >>> from solprop.modes import GradedIndexProvider, IndexMaterial
        >>> from solprop.modes import PowerLawPermittivity
        >>> provider = GradedIndexProvider(
        ...     permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
        ...     exterior=IndexMaterial(1.44),
        ...     radius=25e-6,
        ...     family="TE",
        ...     count=2,
        ...     elements=8,
        ...     extent=2.0,
        ... )
        >>> [label.display_label for label in provider.channel_labels]
        ['TE0,1', 'TE0,2']
        """
        return tuple(
            ChannelLabel(
                self.family,
                self.azimuthal_order,
                radial,
                None,
                None,
                f"{self.family}{self.azimuthal_order},{radial}",
            )
            for radial in range(1, self.count + 1)
        )

    @property
    def identity(self) -> Mapping[str, JsonValue]:
        """Return physical law, sector and explicit discretisation identity.

        Returns
        -------
        collections.abc.Mapping
            Serializable physical model and construction metadata.
        """
        return {
            "spectral_convention": "internal",
            "model": "radial-P1-Bessel-exterior",
            "family": self.family,
            "channels": [dict(label.as_mapping()) for label in self.channel_labels],
            "approximation": "weak-guidance-LP"
            if self.family == "LP"
            else "vector-sector",
            "law": dict(self.permittivity.identity),
            # A structural path identity, not a repr: numpy summarises an array over
            # 1000 elements, so two different long tapers would share an identity.
            "radius_path": cast("JsonValue", profile_identity(self.radius)),
            "exterior": dict(self.exterior.identity),
            # Which responses the core declares, by name. Their **numbers** reach a
            # run's resume key through the regions of
            # `solprop.translate.general.guide_identity`, which reads the same
            # tuple; what belongs here is that this provider is the one that
            # declared them at all.
            "core_responses": [type(one).__name__ for one in self.core_responses],
            "elements": self.elements,
            "extent": self.extent,
            "count": self.count,
            "azimuthal_order": self.azimuthal_order,
        }

    def _eps(
        self, omega: float, radial_fraction: Float[np.ndarray, " n_points"]
    ) -> Float[np.ndarray, " n_points"]:
        """Evaluate the lossless radial permittivity on both materials.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s.
        radial_fraction : numpy.ndarray
            Dimensionless radii divided by the core radius, shape (n_points,);
            anything above one takes the exterior value.

        Returns
        -------
        numpy.ndarray
            Relative permittivity, shape (n_points,).

        Raises
        ------
        ModeError
            If a permittivity is lossy, non-finite or not positive.
        """
        core = self.permittivity.permittivity(omega, np.minimum(radial_fraction, 1.0))
        exterior = self.exterior.permittivity(np.array([omega], dtype=float))[0]
        values = np.where(radial_fraction <= 1, core, exterior)
        if (
            np.any(np.imag(values) != 0)
            or not np.all(np.isfinite(values))
            or np.any(np.real(values) <= 0)
        ):
            raise ModeError("radial provider requires positive lossless permittivity")
        return np.asarray(np.real(values))

    def _weak_guidance_contrast(self, omega: float) -> float:
        """Return the axis-to-exterior relative permittivity contrast.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s.

        Returns
        -------
        float
            Dimensionless ``eps_axis / eps_exterior - 1``, the quantity the LP
            weak-guidance approximation is declared valid over.
        """
        axis = float(self._eps(omega, np.zeros(1))[0])
        exterior = float(self._eps(omega, np.full(1, 2.0))[0])
        return axis / exterior - 1

    @host_blas()
    def _solution(self, omega: float, z: float) -> _RadialSolution:
        """Solve the variational eigenproblem with an exact exterior.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s.
        z : float
            Position in m within the declared path.

        Returns
        -------
        _RadialSolution
            Dimensionless radial mesh, profiles, beta and interface defects.

        Raises
        ------
        ModeError
            If the frequency, the LP contrast or a requested channel's binding
            is outside the provider's declared validity.
        """
        radius = self._radius(z)
        if (
            not np.isfinite(omega)
            or not self.omega_domain[0] < omega <= self.omega_domain[1]
        ):
            raise ModeError("frequency is outside radial provider support")
        # Once per solve, on the axis, where a decreasing radial law attains its
        # maximum. _eps is evaluated per element inside every eigenvalue iteration
        # -- of order 1e4 times per solve -- so testing the same axis-versus-exterior
        # contrast there repeated the check thousands of times and made its value
        # depend on which panel happened to be passed.
        if self.family == "LP" and self._weak_guidance_contrast(omega) > 0.02:
            raise ModeError(
                "LP weak-guidance range requires permittivity contrast <= 0.02"
            )
        nodes = np.concatenate(
            (
                np.linspace(0, 1, self.elements + 1),
                np.linspace(
                    1, self.extent, int(np.ceil(self.elements * (self.extent - 1))) + 1
                )[1:],
            )
        )
        wave_number = omega * radius / C
        mass, operator = self._matrices(omega, nodes, wave_number)
        start = 0 if self.family == "LP" and self.azimuthal_order == 0 else 1
        a, b = operator[start:, start:], mass[start:, start:]
        exterior = float(
            np.real(self.exterior.permittivity(np.array([omega], dtype=float))[0])
        )
        lower = exterior * wave_number**2
        upper = float(np.max(self._eps(omega, nodes))) * wave_number**2
        if upper <= lower:
            raise ModeError("radial core has no guided interval above its exterior")
        order = self.azimuthal_order if self.family == "LP" else 1
        profiles = np.zeros((len(nodes), self.count))
        values, residual = np.zeros(self.count), np.zeros(self.count)
        for channel in range(self.count):
            eigenindex = len(a) - channel - 1

            def boundary_matrix(value: float) -> Float[np.ndarray, "n_matrix n_matrix"]:
                decay = np.sqrt(value - lower)
                argument = decay * self.extent
                logarithmic = (
                    -decay
                    * (kve(abs(order - 1), argument) + kve(order + 1, argument))
                    / (2 * kve(order, argument))
                )
                if self.family != "LP":
                    logarithmic += 1 / self.extent
                correction = np.zeros_like(a)
                correction[-1, -1] = (
                    self.extent
                    * logarithmic
                    / (exterior if self.family == "TM" else 1.0)
                )
                return a + correction

            def equation(value: float, eigenindex: int = eigenindex) -> float:
                eigenvalue = eigh(
                    boundary_matrix(value),
                    b,
                    subset_by_index=(eigenindex, eigenindex),
                    eigvals_only=True,
                )[0]
                return float(eigenvalue - value)

            margin = 64 * np.finfo(float).eps * max(abs(lower), abs(upper))
            if equation(lower + margin) <= 0:
                raise ModeError("requested radial channel is below cutoff")
            value = brentq(
                equation,
                lower + margin,
                upper,
                xtol=margin,
                rtol=8 * np.finfo(float).eps,
            )
            matrix = boundary_matrix(value)
            _, eigenvector = eigh(matrix, b, subset_by_index=(eigenindex, eigenindex))
            vector = eigenvector[:, 0]
            values[channel] = value
            profiles[start:, channel] = vector
            residual[channel] = np.linalg.norm(
                matrix @ vector - value * (b @ vector)
            ) / (np.linalg.norm(matrix) * np.linalg.norm(vector))
        # P1 fields have a first-order derivative trace error. Report it instead
        # of mistaking the algebraic eigenpair residual for Maxwell continuity.
        residual = np.maximum(
            residual, self._trace_defect(nodes, profiles, omega, exterior)
        )
        # Fix each scalar eigenfunction by its largest component, independently of
        # the eigensolver's arbitrary sign and of the requested evaluation nodes.
        signs = np.sign(
            profiles[np.argmax(abs(profiles), axis=0), np.arange(self.count)]
        )
        solution = _RadialSolution(
            nodes, profiles * signs, np.sqrt(values) / radius, residual
        )
        self._require_bound_channels(solution, omega, radius)
        return solution

    def _trace_defect(
        self,
        nodes: Float[np.ndarray, " n_radial_nodes"],
        profiles: Float[np.ndarray, "n_radial_profiles n_modes_profiles"],
        omega: float,
        exterior: float,
    ) -> Float[np.ndarray, " n_modes"]:
        """Return the one-sided derivative-trace defect at the material interface.

        Parameters
        ----------
        nodes : numpy.ndarray
            Dimensionless radial mesh, shape (n_radial_nodes,).
        profiles : numpy.ndarray
            Nodal eigenfunctions, shape (n_radial_profiles, n_modes_profiles).
        omega : float
            Positive angular frequency in rad/s.
        exterior : float
            Exterior relative permittivity, dimensionless.

        Returns
        -------
        numpy.ndarray
            Scaled defect of the continuous quantity of the sector, shape
            (n_modes,): the radial gradient for TE/TM and its inverse-permittivity
            weighting for TM, the plain slope for LP.
        """
        interface = int(np.searchsorted(nodes, 1.0))
        left = (profiles[interface] - profiles[interface - 1]) / (
            nodes[interface] - nodes[interface - 1]
        )
        right = (profiles[interface + 1] - profiles[interface]) / (
            nodes[interface + 1] - nodes[interface]
        )
        if self.family != "LP":
            left, right = left + profiles[interface], right + profiles[interface]
        if self.family == "TM":
            left = left / float(self._eps(omega, np.array([1.0]))[0])
            right = right / exterior
        scale = np.maximum(np.max(abs(profiles), axis=0), np.finfo(float).tiny)
        return np.asarray(abs(left - right) / scale)

    def _require_bound_channels(
        self, solution: _RadialSolution, omega: float, radius: float
    ) -> None:
        """Refuse a channel the declared finite domain cannot carry.

        Parameters
        ----------
        solution : _RadialSolution
            Solved radial eigenfunctions and propagation constants.
        omega : float
            Positive angular frequency in rad/s.
        radius : float
            Core radius in m.

        Raises
        ------
        ModeError
            If a channel keeps more than :data:`MAX_EXTERIOR_TAIL_FRACTION` of its
            power beyond the matching radius.
        """
        # Above cutoff is not the same as bound on this domain: a channel whose
        # exterior decay is slow puts its power in the analytic continuation, not
        # in the discretised core, and neither the eigenvalue nor the profile is
        # then a converged statement about the guide.
        outside = self._tail_fraction(solution, omega, radius, self.extent * radius)
        if np.any(outside > MAX_EXTERIOR_TAIL_FRACTION):
            channel = int(np.argmax(outside))
            raise ModeError(
                f"{self.family}{self.azimuthal_order}{channel + 1} carries "
                f"{outside[channel]:.4g} of its power beyond the matching radius, "
                f"above the floor {MAX_EXTERIOR_TAIL_FRACTION}: the channel is too "
                "close to cutoff for this finite domain. Increase extent, raise the "
                "core radius or contrast, or request fewer channels."
            )

    def _matrices(
        self,
        omega: float,
        nodes: Float[np.ndarray, " n_radial_nodes"],
        wave_number: float,
    ) -> tuple[
        Float[np.ndarray, "n_radial_nodes n_radial_nodes"],
        Float[np.ndarray, "n_radial_nodes n_radial_nodes"],
    ]:
        """Assemble the radial weak mass and Helmholtz operators.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s.
        nodes : numpy.ndarray
            Explicit dimensionless radial mesh, shape (n_radial_nodes,),
            increasing from the axis to the matching radius.
        wave_number : float
            Vacuum wave number multiplied by core radius, dimensionless.

        Returns
        -------
        tuple of numpy.ndarray
            Dimensionless mass and operator matrices on the radial nodes.
        """
        mass = np.zeros((nodes.size, nodes.size))
        operator = np.zeros_like(mass)
        roots, weights = roots_legendre(8)
        shape = np.array([(1 - roots) / 2, (1 + roots) / 2])
        for panel, (left, right) in enumerate(pairwise(nodes)):
            x = left + (right - left) * (roots + 1) / 2
            measure = weights * (right - left) / 2
            eps = self._eps(omega, x)
            derivative = np.repeat(
                np.array([[-1.0], [1.0]]) / (right - left), x.size, axis=1
            )
            inverse = 1 / eps if self.family == "TM" else np.ones_like(x)
            local_mass = (shape * (measure * x * inverse)) @ shape.T
            if self.family == "LP":
                local_operator = (
                    -(derivative * (measure * x)) @ derivative.T
                    + (
                        shape
                        * (
                            measure
                            * (x * eps * wave_number**2 - self.azimuthal_order**2 / x)
                        )
                    )
                    @ shape.T
                )
            else:
                gradient = derivative + shape / x
                local_operator = (
                    -(gradient * (measure * x * inverse)) @ gradient.T
                    + (shape * (measure * x * inverse * eps * wave_number**2)) @ shape.T
                )
            mass[panel : panel + 2, panel : panel + 2] += local_mass
            operator[panel : panel + 2, panel : panel + 2] += local_operator
        return mass, operator

    def node_rule(
        self,
        omega: float,
        z: float = 0.0,
        *,
        n_r: int | None = None,
        n_theta: int | None = None,
    ) -> ModalNodeRule:
        """Build nodes split at the material interface and every finite element.

        The frequency is not a discretisation choice here -- the mesh is the declared
        element count over the declared extent -- but it is what the discarded power
        beyond the matching radius depends on, so the rule certifies that tail at this
        frequency, as :meth:`solprop.modes.guide.StepIndexProvider.node_rule` does.
        One signature for every provider.

        **That certificate costs a solve.** Building the mesh is 0.22 ms on the
        reference parabolic sector (32 elements, two channels, on an Apple-silicon
        laptop) and the whole call is 4.0 ms, because the tail bound needs the
        channels' fields. It is construction-time work, once per guide rather than
        once per step, but a caller that builds a rule per frequency knot is paying
        for a solve per knot and should pass one rule instead.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s in the bound band; the rule's tail
            bound is certified there.
        z : float, optional
            Position in m in the radius path.
        n_r : int or None, optional
            Gauss nodes per element, at least two; ``None`` uses the four a P1
            element mesh of the declared element count needs.
        n_theta : int or None, optional
            Angular nodes, at least two; ``None`` takes two for an ``LP`` sector of
            azimuthal order zero, whose fields do not depend on the angle, so that
            two equally spaced nodes integrate every product exactly; nine for a
            ``TE``/``TM`` sector -- its Cartesian components carry one angular
            harmonic, whose quartic products need five -- and
            ``4 * (order + 1) + 1`` where a higher LP angular order needs more.

        Returns
        -------
        ModalNodeRule
            Core/exterior ownership on the declared finite domain, with the exactly
            integrated discarded-power fraction of each channel in its identity.

        Raises
        ------
        ModeError
            If the frequency, the position or a requested channel is unusable.
        """
        n_r = 4 if n_r is None else n_r
        radius = self._radius(z)
        tails = self.tail_bound(omega, z, outer_radius=self.extent * radius)
        if n_theta is None:
            n_theta = (
                # An LP sector of azimuthal order zero is one scalar profile in a
                # fixed polarisation: nothing it integrates depends on the angle, so
                # the smallest rule the angular grid admits is exact.
                2
                if self.family == "LP" and self.azimuthal_order == 0
                else max(9, 4 * (self.azimuthal_order + 1) + 1)
            )
        interfaces = np.concatenate(
            (
                np.linspace(0, 1, self.elements + 1)[1:],
                np.linspace(
                    1, self.extent, int(np.ceil(self.elements * (self.extent - 1))) + 1
                )[1:-1],
            )
        )
        rule = radial_node_rule(
            tuple(interfaces * radius), self.extent * radius, n_r=n_r, n_theta=n_theta
        )
        region = (np.hypot(rule.xy[:, 0], rule.xy[:, 1]) > radius).astype(np.int32)
        return ModalNodeRule(
            rule.xy,
            rule.weights,
            region,
            {
                **rule.identity,
                "tail": "exact Bessel continuation beyond the matching radius",
                "tail_power_bound": [float(value) for value in tails],
                # Beside the bound: a tail bound only holds at the frequency it
                # was taken, and a consumer should not have to infer which that is.
                "tail_power_bound_omega": float(omega),
                "omega": float(omega),
                "z": float(z),
            },
        )

    def solve(self, omega: float, z: float, *, nodes: ModalNodeRule) -> ModalSample:
        """Evaluate a variational vector sector or labelled LP approximation.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s in the bound band.
        z : float
            Position in m in the declared radius path.
        nodes : ModalNodeRule
            Cartesian nodes within the declared finite domain.

        Returns
        -------
        ModalSample
            Poynting-normalised E/H; boundary_residual includes the one-sided Maxwell
            interface trace defect. Spatial discretisation and exterior truncation
            need refinement. Its ``valid`` mask is all ``True``: a channel too near
            cutoff is refused by name, not masked.

        Raises
        ------
        ModeError
            If nodes, ownership or requested bound channels are invalid.
        """
        solution = self._solution(omega, z)
        radius = self._radius(z)
        radial = np.hypot(nodes.xy[:, 0], nodes.xy[:, 1]) / radius
        if not np.array_equal(nodes.region_index, (radial > 1).astype(np.int32)):
            raise ModeError("radial nodes have wrong material ownership")
        theta = np.arctan2(nodes.xy[:, 1], nodes.xy[:, 0])
        panel = np.clip(
            np.searchsorted(solution.nodes, radial, side="right") - 1,
            0,
            len(solution.nodes) - 2,
        )
        widths = np.diff(solution.nodes)[panel]
        fraction = (radial - solution.nodes[panel]) / widths
        profile = (
            solution.profiles[panel] * (1 - fraction[:, None])
            + solution.profiles[panel + 1] * fraction[:, None]
        )
        slope = (solution.profiles[panel + 1] - solution.profiles[panel]) / widths[
            :, None
        ]
        outer = radial > self.extent
        decay = np.sqrt(
            (solution.beta * radius) ** 2
            - float(
                np.real(self.exterior.permittivity(np.array([omega], dtype=float))[0])
            )
            * (omega * radius / C) ** 2
        )
        order = self.azimuthal_order if self.family == "LP" else 1
        argument = radial[outer, None] * decay
        edge = self.extent * decay
        factor = np.exp(edge - argument) / kve(order, edge)
        profile[outer] = solution.profiles[-1] * kve(order, argument) * factor
        slope[outer] = (
            -solution.profiles[-1]
            * decay
            * (kve(abs(order - 1), argument) + kve(order + 1, argument))
            * factor
            / 2
        )
        gradient = (
            slope
            + np.divide(
                profile,
                radial[:, None],
                # Regular TE0/TM0 profiles vanish linearly: f/r -> f'.
                out=slope.copy(),
                where=radial[:, None] != 0,
            )
        ) / radius
        eps = self._eps(omega, radial)
        e, h = (
            np.zeros((len(radial), 3, self.count), complex),
            np.zeros((len(radial), 3, self.count), complex),
        )
        if self.family == "TE":
            e[:, 0], e[:, 1] = (
                -np.sin(theta)[:, None] * profile,
                np.cos(theta)[:, None] * profile,
            )
            h[:, 0] = (
                -EPS0 * C**2 / omega * np.cos(theta)[:, None] * profile * solution.beta
            )
            h[:, 1] = (
                -EPS0 * C**2 / omega * np.sin(theta)[:, None] * profile * solution.beta
            )
            h[:, 2] = 1j * EPS0 * C**2 / omega * gradient
        elif self.family == "TM":
            h[:, 0], h[:, 1] = (
                -np.sin(theta)[:, None] * profile,
                np.cos(theta)[:, None] * profile,
            )
            e[:, 0] = (
                np.cos(theta)[:, None]
                * profile
                * solution.beta
                / (omega * EPS0 * eps[:, None])
            )
            e[:, 1] = (
                np.sin(theta)[:, None]
                * profile
                * solution.beta
                / (omega * EPS0 * eps[:, None])
            )
            e[:, 2] = -1j * gradient / (omega * EPS0 * eps[:, None])
        else:
            e[:, 0] = profile * np.cos(self.azimuthal_order * theta)[:, None]
            h[:, 1] = e[:, 0] * solution.beta * (EPS0 * C**2 / omega)
        # Integrate the continuous P1 profile on its own element rule, not the
        # caller's quadrature. The field product is polynomial for TE and LP;
        # TM's 1/eps weight is integrated by the same eight-point element rule.
        norms = self._norms(solution, omega, radius)
        e, h = e / np.sqrt(norms), h / np.sqrt(norms)
        metric = flux_product(e, h, nodes.weights)
        return ModalSample(
            omega,
            z,
            e,
            h,
            -1j * np.diag(solution.beta),
            metric,
            solution.residual,
            np.ones(self.count, dtype=bool),
            e.conj(),
            -h.conj(),
            -4 * metric,
            "radial-largest-component-positive",
            json.dumps(dict(self.identity), sort_keys=True, allow_nan=False),
        )

    def _norms(
        self, solution: _RadialSolution, omega: float, radius: float
    ) -> Float[np.ndarray, " n_modes"]:
        """Integrate full-domain power independently of requested field nodes.

        Parameters
        ----------
        solution : _RadialSolution
            Solved radial eigenfunctions and propagation constants.
        omega : float
            Positive angular frequency in rad/s.
        radius : float
            Core radius in m.

        Returns
        -------
        numpy.ndarray
            Positive power norms in W, shape (n_modes,).
        """
        roots, weights = roots_legendre(8)
        norms = np.zeros(self.count)
        angular = (
            2 * np.pi if self.family != "LP" or self.azimuthal_order == 0 else np.pi
        )
        for panel, (left, right) in enumerate(pairwise(solution.nodes)):
            x = left + (right - left) * (roots + 1) / 2
            f = (1 - roots[:, None]) / 2 * solution.profiles[panel] + (
                1 + roots[:, None]
            ) / 2 * solution.profiles[panel + 1]
            factor = (
                1 / (omega * EPS0 * self._eps(omega, x))
                if self.family == "TM"
                else np.full_like(x, EPS0 * C**2 / omega)
            )
            norms += (
                angular
                * radius**2
                / 2
                * solution.beta
                * np.sum(
                    (weights * (right - left) / 2 * x * factor)[:, None] * f**2, axis=0
                )
            )
        exterior = float(
            np.real(self.exterior.permittivity(np.array([omega], dtype=float))[0])
        )
        factor = (
            1 / (omega * EPS0 * exterior)
            if self.family == "TM"
            else EPS0 * C**2 / omega
        )
        return (
            norms
            + angular
            * radius** 2
            / 2
            * solution.beta
            * factor
            * self._tail_integral(solution, omega, radius, self.extent)
        )

    def _tail_integral(
        self, solution: _RadialSolution, omega: float, radius: float, lower: float
    ) -> Float[np.ndarray, " n_modes"]:
        """Integrate the squared decaying radial profile exactly.

        Parameters
        ----------
        solution : _RadialSolution
            Solved radial eigenfunctions and propagation constants.
        omega : float
            Positive angular frequency in rad/s.
        radius : float
            Core radius in m.
        lower : float
            Lower integration radius divided by core radius.

        Returns
        -------
        numpy.ndarray
            Dimensionless radial integrals, shape (n_modes,).
        """
        exterior = float(
            np.real(self.exterior.permittivity(np.array([omega], dtype=float))[0])
        )
        decay = np.sqrt(
            (solution.beta * radius) ** 2 - exterior * (omega * radius / C) ** 2
        )
        order = self.azimuthal_order if self.family == "LP" else 1
        edge, argument = self.extent * decay, lower * decay
        amplitude = (
            solution.profiles[-1]
            * np.exp(edge - argument)
            * kve(order, argument)
            / kve(order, edge)
        )
        # Integrating the modified-Bessel equation gives the exact quadratic
        # tail, with the increasing exponential scaled out of every K ratio.
        return (
            amplitude**2
            * lower**2
            / 2
            * (
                kve(abs(order - 1), argument)
                * kve(order + 1, argument)
                / kve(order, argument) ** 2
                - 1
            )
        )

    def tail_bound(
        self, omega: float, z: float = 0.0, *, outer_radius: float
    ) -> Float[np.ndarray, " n_modes"]:
        """Return exact omitted exterior power fractions for the variational modes.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s within the bound band.
        z : float, optional
            Position in m in the declared radius path.
        outer_radius : float
            Finite radius in m, at or beyond the Bessel matching radius.

        Returns
        -------
        numpy.ndarray
            Per-channel discarded power fractions, shape (n_modes,). These certify
            exterior quadrature truncation, not finite-element discretisation.

        Raises
        ------
        ModeError
            If the requested tail begins inside the finite-element domain.
        """
        radius = self._radius(z)
        if not np.isfinite(outer_radius) or outer_radius < self.extent * radius:
            raise ModeError("tail must start at or beyond the Bessel matching radius")
        return self._tail_fraction(
            self._solution(omega, z), omega, radius, outer_radius
        )

    def _tail_fraction(
        self,
        solution: _RadialSolution,
        omega: float,
        radius: float,
        outer_radius: float,
    ) -> Float[np.ndarray, " n_modes"]:
        """Return the exact power fraction of each channel beyond a radius.

        Parameters
        ----------
        solution : _RadialSolution
            Solved radial eigenfunctions and propagation constants.
        omega : float
            Positive angular frequency in rad/s.
        radius : float
            Core radius in m.
        outer_radius : float
            Radius in m at or beyond the Bessel matching radius.

        Returns
        -------
        numpy.ndarray
            Per-channel discarded power fractions, shape (n_modes,).
        """
        exterior = float(
            np.real(self.exterior.permittivity(np.array([omega], dtype=float))[0])
        )
        factor = (
            1 / (omega * EPS0 * exterior)
            if self.family == "TM"
            else EPS0 * C**2 / omega
        )
        angular = (
            2 * np.pi if self.family != "LP" or self.azimuthal_order == 0 else np.pi
        )
        tail = (
            angular
            * radius**2
            / 2
            * solution.beta
            * factor
            * self._tail_integral(solution, omega, radius, outer_radius / radius)
        )
        return np.asarray(tail / self._norms(solution, omega, radius))
