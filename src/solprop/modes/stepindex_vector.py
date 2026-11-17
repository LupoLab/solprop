# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Recover a step-index fibre's bound Maxwell fields and offer them as a modal provider.

The vector E/H fields of a :class:`~solprop.modes.stepindex.StepIndexMode` follow
analytically from its existing eigenvalue roots: Bessel ``J`` in the core and scaled
``K`` in the cladding, normalised by an independent full-domain power integral
(:meth:`~solprop.modes.stepindex.StepIndexMode.electromagnetic_field` reads them
from here).
:class:`StepIndexProvider` packages a set of those modes, with the material regions
they are solved in (agreeing to :data:`MATERIAL_AGREEMENT_RTOL`), as a
:class:`~solprop.modes.vector.ModalProvider` the general modal route samples
(:class:`~solprop.modes.guide.ModalGuide`).
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import cast

import numpy as np
from jaxtyping import Complex, Float
from scipy.integrate import quad
from scipy.special import jv, jvp, kve

from solprop._display import Displayed, engineering, line
from solprop.constants import EPS0, C
from solprop.exceptions import ModeError
from solprop.modes.collection import ModeCollection
from solprop.modes.fill import _alpha_values
from solprop.modes.material import JsonValue, MaterialRegion
from solprop.modes.stepindex import (
    PROFILE_MIN_W,
    StepIndexMode,
    _k_decay,
    _log_slope,
    real_index_and_slope,
)
from solprop.modes.vector import (
    ModalNodeRule,
    ModalSample,
    flux_product,
    radial_node_rule,
)
from solprop.physdata.glasses import GlassIndex
from solprop.profiles import as_profile, profile_identity

__all__ = ["MATERIAL_AGREEMENT_RTOL", "StepIndexProvider"]

MATERIAL_AGREEMENT_RTOL = 1e-6
"""Largest relative disagreement a step-index provider admits between a region's
declared permittivity law and the index its channels were solved with.

The modes own the eigenproblem's indices (``StepIndexMode.core`` and
``cladding``) and the regions own the material laws; the two describe the same
material and nothing else made them agree. The regions' laws are what a declared
``n2`` is converted against (:meth:`solprop.modes.material.MaterialKerr.susceptibility`)
and what a guide's identity records, so a core region declared at 2.0 beside
channels solved at 1.45 would propagate the 1.45 fields with a ``chi3`` 90 % too
large and nothing downstream would say so.

The budget is on the **permittivity**, and 1e-6 is chosen to admit a law evaluated
by two paths -- a glass law asked for its index and the same law asked for its
permittivity -- while refusing any disagreement a reader would call physical: the
smallest index step this package's own weakly guiding fixtures use is 0.001 in the
index, which is 1.4e-3 in the permittivity, three orders above this.
"""


def _radial(
    order: int, argument: Float[np.ndarray, " n_points"], edge: float, inside: bool
) -> tuple[
    Float[np.ndarray, " n_points"],
    Float[np.ndarray, " n_points"],
    Float[np.ndarray, " n_points"],
]:
    """Evaluate the scaled radial function, slope and axis limit.

    - ``edge``: dimensionless Bessel argument at the interface.
    - ``inside``: the regular core solution (``J``) rather than the cladding one.
    - Returns: dimensionless radial function, argument derivative and divided value.
    """
    if inside:
        value = jv(order, argument) / jv(order, edge)
        slope = jvp(order, argument) / jv(order, edge)
        # J_n(x)/x has a finite nonzero axis limit only for n=1;
        # the angular derivative multiplying it vanishes for n=0.
        divided = np.divide(
            value, argument, out=np.zeros_like(value), where=argument != 0
        )
        divided = np.where(
            argument == 0, 0.5 / jv(order, edge) if order == 1 else 0.0, divided
        )
    elif math.isfinite(kve(order + 1, edge)):
        factor = np.exp(edge - argument) / kve(order, edge)
        value = kve(order, argument) * factor
        slope = -(kve(abs(order - 1), argument) + kve(order + 1, argument)) * factor / 2
        divided = value / argument
    else:
        # At large order and small decay kve overflows; the same ratios come
        # from the quotient recurrences, finite wherever the field is.
        value = _k_decay(order, edge, argument / edge)
        slope = _log_slope(order, argument) * value
        divided = value / argument
    return np.asarray(value), np.asarray(slope), np.asarray(divided)


@dataclass(frozen=True)
class _FieldParameters:
    """Hold one lossless vector solution at a fixed frequency and position.

    Attributes
    ----------
    n : int
        Non-negative azimuthal order.
    parity : str
        Angular parity, ``even`` or ``odd``.
    radius : float
        Core radius in m.
    k, beta : float
        Vacuum and modal wavenumbers in rad/m.
    u, w : float
        Dimensionless core and cladding transverse parameters.
    core, clad : float
        Real core and cladding indices.
    electric_amplitude, magnetic_amplitude : float
        Longitudinal amplitudes in the existing unnormalised Maxwell solution.
    """

    n: int
    parity: str
    radius: float
    k: float
    beta: float
    u: float
    w: np.float64
    core: float
    clad: float
    electric_amplitude: float
    magnetic_amplitude: float | np.float64


def _field_parameters(mode: StepIndexMode, omega: float, z: float) -> _FieldParameters:
    """Solve the position-independent parameters of one transverse field.

    - ``omega``: positive angular frequency in rad/s.
    - ``z``: axial position in m within the declared path.
    """
    if not np.isfinite(omega) or omega <= 0 or not np.isfinite(z):
        raise ModeError("omega must be positive and z finite")
    frequency = np.array([omega], dtype=float)
    if np.any(_alpha_values(mode.alpha, frequency)[0] != 0):
        raise ModeError(
            "leaky or attenuating E/H modes are unqualified: bound lossless fields only"
        )
    if any(
        isinstance(material, GlassIndex)
        and np.any(np.imag(material.index(frequency)) != 0)
        for material in (mode._core, mode._cladding)
    ):
        raise ModeError("absorbing material E/H modes are unqualified")
    index, transverse = mode._eigenpairs(frequency, z)
    radius = mode._radius(z)
    core = float(real_index_and_slope(mode._core, frequency)[0][0])
    clad = float(real_index_and_slope(mode._cladding, frequency)[0][0])
    k, beta, u = omega / C, float(index[0]) * omega / C, float(transverse[0])
    w = np.float64(np.sqrt((k * radius) ** 2 * (core - clad) * (core + clad) - u**2))
    if w < PROFILE_MIN_W:
        raise ModeError(
            "full-vector profile is too close to cutoff: "
            "cladding decay below PROFILE_MIN_W"
        )
    a = jvp(mode.n, u) / (u * jv(mode.n, u))
    b = float(_log_slope(mode.n, w)) / w
    electric_amplitude = 0.0 if mode.kind == "TE" else 1.0
    magnetic_amplitude = 1.0 if mode.kind == "TE" else 0.0
    if mode.n:
        magnetic_amplitude = np.float64(
            -beta / k * mode.n * (1 / u**2 + 1 / w**2) / (a + b)
        )
    return _FieldParameters(
        mode.n,
        mode.parity,
        radius,
        k,
        beta,
        u,
        w,
        core,
        clad,
        electric_amplitude,
        magnetic_amplitude,
    )


def _fields(
    mode: StepIndexMode,
    points: Float[np.ndarray, "n_points 2"],
    omega: float,
    z: float,
    *,
    exterior: bool | None = None,
) -> tuple[Complex[np.ndarray, "n_points 3"], Complex[np.ndarray, "n_points 3"]]:
    """Recover full Maxwell fields from analytic longitudinal amplitudes.

    - ``omega``: positive angular frequency in rad/s.
    - ``z``: position in m within the declared path.
    - ``exterior``: force one side for an exact interface trace; None follows each
      point.
    - Returns: unnormalised E and H, each shape (n_points, 3).
    """
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ModeError("points must be finite Cartesian (x, y) pairs")
    return _prepared_fields(
        _field_parameters(mode, omega, z), points, exterior=exterior
    )


def _prepared_fields(
    parameters: _FieldParameters,
    points: Float[np.ndarray, "n_points 2"],
    *,
    exterior: bool | None = None,
) -> tuple[Complex[np.ndarray, "n_points 3"], Complex[np.ndarray, "n_points 3"]]:
    """Evaluate one solved mode at Cartesian positions without solving it again.

    - ``points``: Cartesian positions in m, shape (n_points, 2).
    - ``exterior``: force one side for an exact interface trace; None follows each
      point.
    - Returns: unnormalised E and H, each shape (n_points, 3).
    """
    radius, k, beta = parameters.radius, parameters.k, parameters.beta
    u, w, core, clad = parameters.u, parameters.w, parameters.core, parameters.clad
    electric_amplitude = parameters.electric_amplitude
    magnetic_amplitude = parameters.magnetic_amplitude
    r, theta = (
        np.hypot(points[:, 0], points[:, 1]),
        np.arctan2(points[:, 1], points[:, 0]),
    )
    if parameters.n == 0:
        angular_e, angular_h = np.ones_like(r), np.ones_like(r)
        derivative_e, derivative_h = np.zeros_like(r), np.zeros_like(r)
    elif parameters.parity == "even":
        angular_e, angular_h = (
            np.cos(parameters.n * theta),
            np.sin(parameters.n * theta),
        )
        derivative_e, derivative_h = -parameters.n * angular_h, parameters.n * angular_e
    else:
        angular_e, angular_h = (
            np.sin(parameters.n * theta),
            -np.cos(parameters.n * theta),
        )
        derivative_e, derivative_h = -parameters.n * angular_h, parameters.n * angular_e
    electric, magnetic = np.zeros((len(r), 3), complex), np.zeros((len(r), 3), complex)
    for inside in (True, False):
        mask = (
            (r <= radius) if exterior is None else np.full(r.shape, not exterior)
        ) == inside
        rr, tt = r[mask], theta[mask]
        scale = (u if inside else w) / radius
        q2 = scale**2 if inside else -(scale**2)
        eps = core**2 if inside else clad**2
        value, slope, divided = _radial(
            parameters.n, scale * rr, u if inside else w, inside
        )
        er = (
            beta * electric_amplitude * scale * slope * angular_e[mask]
            + k * magnetic_amplitude * scale * divided * derivative_h[mask]
        ) / q2
        et = (
            beta * electric_amplitude * scale * divided * derivative_e[mask]
            - k * magnetic_amplitude * scale * slope * angular_h[mask]
        ) / q2
        hr = (
            beta * magnetic_amplitude * scale * slope * angular_h[mask]
            - k * eps * electric_amplitude * scale * divided * derivative_e[mask]
        ) / q2
        ht = (
            beta * magnetic_amplitude * scale * divided * derivative_h[mask]
            + k * eps * electric_amplitude * scale * slope * angular_e[mask]
        ) / q2
        electric[mask] = np.column_stack(
            (
                er * np.cos(tt) - et * np.sin(tt),
                er * np.sin(tt) + et * np.cos(tt),
                1j * electric_amplitude * value * angular_e[mask],
            )
        )
        magnetic[mask] = (
            EPS0
            * C
            * np.column_stack(
                (
                    hr * np.cos(tt) - ht * np.sin(tt),
                    hr * np.sin(tt) + ht * np.cos(tt),
                    1j * magnetic_amplitude * value * angular_h[mask],
                )
            )
        )
    return electric, magnetic


_TAIL_EPSABS_FRACTION = 1e-15
"""Absolute quadrature floor for the exterior power piece, as a fraction of the core.

A bound mode's exterior power falls exponentially -- 1e-176 of the total for a
25 um fibre integrated from two core radii -- and asking for 2e-11 relative
accuracy on such a number is asking the adaptive rule to subdivide forever.
1e-15 of the core piece is three orders below the 1e-12 discarded-power
certificate the provider's node rule asserts, so the floor can never be what
makes that certificate pass, and it binds only where the tail is already
negligible: measured, it changes no returned power and leaves the largest
relative error of the total at 3e-12.
"""

_POWER_CACHE_SIZE = 4096
"""Distinct (mode, omega, z, start) power integrals kept.

The key holds the mode, so an unbounded cache pins every mode ever normalised
for the life of the process, along with its material objects. A dense modal
table is the heaviest caller: 17 frequency knots times 5 positions times a
handful of channels, with one full-domain and one tail integral each, is a few
hundred entries, so 4096 keeps every repeat within one table and several tables
besides.
"""


@lru_cache(maxsize=_POWER_CACHE_SIZE)
def _power(
    mode: StepIndexMode, omega: float, z: float, start: float = 0.0
) -> tuple[float, float]:
    """Integrate signed full-domain or exterior-tail Poynting power.

    - ``omega``: positive angular frequency in rad/s.
    - ``z``: position in m within the declared path.
    - ``start``: lower integration radius divided by the core radius; ``start >= 1``
      integrates an exterior tail alone.
    - Returns: power in W and the adaptive absolute integration error in W.
    """
    # The quadrature varies radius alone. Resolve the eigenpair and material
    # values once for this integral, without caching a mutable profile's inputs.
    parameters = _field_parameters(mode, omega, z)
    radius = parameters.radius
    angles = np.arange(2 * mode.n + 3) * 2 * np.pi / (2 * mode.n + 3)

    def integrand(x: float) -> float:
        """Evaluate the angularly integrated radial flux at ``x = r / a``.

        The caller multiplies by the squared radius, the Jacobian of ``x``.
        """
        points = radius * x * np.column_stack((np.cos(angles), np.sin(angles)))
        e, h = _prepared_fields(parameters, points)
        return float(
            np.mean(np.real(e[:, 0] * h[:, 1].conj() - e[:, 1] * h[:, 0].conj()))
            * np.pi
            * x
        )

    # Dimensionless radius avoids an absolute SI-area tolerance hiding missing flux,
    # but the integrand still carries the field normalisation, so a fixed epsabs is
    # not scale free: 1e-12 was 4.8e-11 of the 250 nm strand's power and 1.3e-14 of
    # the 25 um fibre's, the first looser than the epsrel beside it. The core piece
    # therefore asks for relative accuracy alone; only the exponentially small
    # exterior piece gets an absolute floor, and that floor is a fixed fraction of
    # the core piece, so it means the same thing at every scale. The floor sits
    # _TAIL_EPSABS_FRACTION below the power, three orders under the 1e-12 tail
    # certificate of the provider's node rule, which reads the error returned here.
    # A tail integrated on its own (start >= 1) is that certificate's own integral
    # and gets no floor: it must be resolved relative to its own tiny value.
    if start < 1:
        core, core_error = quad(
            integrand, start, 1.0, epsabs=0.0, epsrel=2e-11, limit=150
        )
        tail, tail_error = quad(
            integrand,
            1.0,
            np.inf,
            epsabs=_TAIL_EPSABS_FRACTION * abs(core),
            epsrel=2e-11,
            limit=150,
        )
        return radius**2 * (core + tail), radius**2 * (core_error + tail_error)
    value, error = quad(integrand, start, np.inf, epsabs=0.0, epsrel=2e-11, limit=150)
    return radius**2 * value, radius**2 * error


@dataclass(frozen=True, eq=False)
class StepIndexProvider(Displayed):
    """Evaluate exact bound step-index E/H using independent full-domain power norms.

    A returned :class:`~solprop.modes.vector.ModalSample` is always fully valid:
    an unsupported channel is refused by name rather than masked out, so
    ``valid`` is all ``True`` in every sample this provider returns.

    Attributes
    ----------
    modes : tuple of StepIndexMode
        Unique physical channels of one core/cladding geometry, in state order.
    regions : tuple of MaterialRegion
        Core and unbounded exterior, with material laws matching the modes.
    """

    modes: tuple[StepIndexMode, ...]
    regions: tuple[MaterialRegion, ...]

    def __repr__(self) -> str:
        """Return a one-line description: the channels and the region count.

        Returns
        -------
        str
            For example ``StepIndexProvider(HE11, 2 regions)``.
        """
        return line(
            "StepIndexProvider",
            [
                ("", ", ".join(f"{m.kind}{m.n}{m.m}" for m in self.modes)),
                ("", f"{len(self.regions)} regions"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        low, high = self.z_domain
        return "StepIndexProvider: exact bound E/H, unit full-domain power", [
            ("channels", ", ".join(f"{m.kind}{m.n}{m.m}" for m in self.modes)),
            ("regions", ", ".join(region.name for region in self.regions)),
            ("z domain", f"{engineering(low, 'm')} to {engineering(high, 'm')}"),
        ]

    def __post_init__(self) -> None:
        """Validate the common guide and the two material regions.

        Raises
        ------
        ModeError
            If the channels or material ownership are inconsistent.
        """
        ModeCollection(self.modes)
        if any(not self.modes[0].same_guide(mode) for mode in self.modes):
            raise ModeError("step-index channels must share one physical guide")
        if len(self.regions) != 2 or self.regions[-1].outer_radius is not None:
            raise ModeError("step-index provider requires core and unbounded cladding")
        self._check_material_agreement()

    def _check_material_agreement(self) -> None:
        """Require each region's law to be the index its channels were solved with."""
        mode = self.modes[0]
        omega = float(2 * np.pi * C / mode.profile_wavelength)
        # The resolved laws, not the declared ones: `core=None` means fused silica
        # and the mode has already substituted it.
        solved = (mode._core, mode._cladding)
        for region, law in zip(self.regions, solved, strict=True):
            low, high = region.material.omega_domain
            if not low <= omega <= high:
                raise ModeError(
                    f"region {region.name!r} declares a law valid from {low:.6e} to "
                    f"{high:.6e} rad/s, which does not cover the channels' profile "
                    f"wavelength {mode.profile_wavelength:.6e} m ({omega:.6e} "
                    "rad/s): the region's law is what a declared n2 is converted "
                    "against and what the guide's identity records"
                )
            index = float(real_index_and_slope(law, np.array([omega]))[0][0])
            declared = float(
                np.real(region.material.permittivity(np.array([omega]))[0])
            )
            if abs(declared - index**2) > MATERIAL_AGREEMENT_RTOL * index**2:
                raise ModeError(
                    f"region {region.name!r} declares relative permittivity "
                    f"{declared!r} at the channels' profile wavelength "
                    f"{mode.profile_wavelength:.6e} m, and the channels were solved "
                    f"with index {index!r} there, which is {index**2!r}: a relative "
                    f"{abs(declared - index**2) / index**2:.3g}, above "
                    f"{MATERIAL_AGREEMENT_RTOL:.3g}. The eigenproblem uses the "
                    "modes' indices and the region's law is what a declared n2 is "
                    "converted against, so the two must be the same material"
                )

    @property
    def z_breakpoints(self) -> tuple[float, ...]:  # numpydoc ignore=RT01
        """Radius-path breakpoints (physical path kinks), m."""
        return tuple(float(x) for x in self.modes[0].core_radius.breakpoints)

    @property
    def z_domain(self) -> tuple[float, float]:  # numpydoc ignore=RT01
        """Radius-path interval ``(lower, upper)``, m."""
        return self.modes[0].core_radius.z_span

    @property
    def omega_domain(self) -> tuple[float, float]:  # numpydoc ignore=RT01
        """Material frequency support ``(lower, upper)``, rad/s.

        :meth:`solve` checks each mode's cutoff.
        """
        return max(r.material.omega_domain[0] for r in self.regions), min(
            r.material.omega_domain[1] for r in self.regions
        )

    @property
    def identity(self) -> Mapping[str, JsonValue]:  # numpydoc ignore=RT01
        """Internal convention, bound-model and channel identities.

        Serializable physical model and construction metadata.
        """
        return {
            "spectral_convention": "internal",
            "model": "bound-step-index",
            "channels": [
                dict(label.as_mapping())
                for label in ModeCollection(self.modes).channel_labels
            ],
            "materials": [dict(r.material.identity) for r in self.regions],
            # A structural path identity, not a repr: numpy summarises an array over
            # 1000 elements, so two different long tapers would share an identity.
            "radius_path": cast(
                "JsonValue", profile_identity(self.modes[0].core_radius)
            ),
        }

    def node_rule(
        self,
        omega: float,
        z: float = 0.0,
        *,
        n_r: int | None = None,
        n_theta: int | None = None,
    ) -> ModalNodeRule:
        """Build a common rule and certify the discarded cladding power.

        Parameters
        ----------
        omega : float
            Positive frequency in rad/s in the guided band.
        z : float, optional
            Position in m within the radius path.
        n_r : int or None, optional
            Gauss nodes per radial panel, at least two; ``None`` uses the 64 this
            provider's exponential cladding tail needs.
        n_theta : int or None, optional
            Angular node count; None uses 4*(maximum n+1)+1 for quartic products.

        Returns
        -------
        ModalNodeRule
            Material-split rule with independently integrated tail bounds.

        Raises
        ------
        ModeError
            If the bound field or tail cannot be certified.
        """
        omega, z = float(omega), float(z)
        n_r = 64 if n_r is None else n_r
        radius = self.modes[0]._radius(z)
        decay = []
        for mode in self.modes:
            neff = float(mode.neff(np.array([omega], dtype=float), z)[0].real)
            clad = float(
                real_index_and_slope(mode._cladding, np.array([omega], dtype=float))[0][
                    0
                ]
            )
            decay.append(omega / C * np.sqrt((neff - clad) * (neff + clad)))
        outer = radius + 24 / min(decay)
        rule = radial_node_rule(
            (radius,),
            outer,
            n_r=n_r,
            n_theta=4 * (max(m.n for m in self.modes) + 1) + 1
            if n_theta is None
            else n_theta,
        )
        tails = []
        for mode in self.modes:
            total, error = _power(mode, omega, z)
            tail, tail_error = _power(mode, omega, z, outer / radius)
            if total <= error or (abs(tail) + tail_error) / (total - error) > 1e-12:
                raise ModeError(
                    "cladding tail could not be certified below 1e-12 of full power"
                )
            tails.append((abs(tail) + tail_error) / (total - error))
        return ModalNodeRule(
            rule.xy,
            rule.weights,
            rule.region_index,
            {
                **rule.identity,
                "tail_power_bound": tails,
                # Beside the bound, so that a consumer reads the certificate
                # without having to know that "omega" is the frequency it holds
                # at. A tail bound is only a bound at the frequency it was taken.
                "tail_power_bound_omega": omega,
                "omega": omega,
                "z": z,
            },
        )

    def solve(self, omega: float, z: float, *, nodes: ModalNodeRule) -> ModalSample:
        """Solve bound channels on explicitly owned physical nodes.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s within material and guided support.
        z : float
            Position in m in the radius path.
        nodes : ModalNodeRule
            Common Cartesian nodes with correct core/cladding ownership.

        Returns
        -------
        ModalSample
            Unit full-domain power E/H and their sampled physical Gram matrix. Its
            ``valid`` mask is all ``True``: an unsupported channel is refused here,
            not masked.

        Raises
        ------
        ModeError
            If domains, material ownership, norm or a boundary residual is invalid.
        """
        if (
            not self.omega_domain[0] < omega <= self.omega_domain[1]
            or not self.z_domain[0] <= z <= self.z_domain[1]
        ):
            raise ModeError("sample is outside provider frequency or z domain")
        omega, z = float(omega), float(z)
        radius = self.modes[0]._radius(z)
        boundaries = [
            float(as_profile(value)(np.array([z], dtype=float))[0])
            for value in (
                self.regions[0].inner_radius,
                self.regions[0].outer_radius,
                self.regions[1].inner_radius,
            )
            if value is not None
        ]
        if boundaries != [0.0, radius, radius]:
            raise ModeError("material region radii disagree with the core interface")
        expected = (np.hypot(nodes.xy[:, 0], nodes.xy[:, 1]) > radius).astype(np.int32)
        if not np.array_equal(nodes.region_index, expected):
            raise ModeError("node region ownership disagrees with the core interface")
        for region, material in zip(
            self.regions, (self.modes[0]._core, self.modes[0]._cladding), strict=True
        ):
            eps = region.material.permittivity(np.array([omega], dtype=float))
            target = (
                real_index_and_slope(material, np.array([omega], dtype=float))[0] ** 2
            )
            if np.any(np.imag(eps) != 0) or not np.allclose(
                np.real(eps), target, rtol=16 * np.finfo(float).eps, atol=0
            ):
                raise ModeError(
                    "provider material differs from its lossless step-index law"
                )
        electric, magnetic, beta, residual = [], [], [], []

        angles = np.arange(17) * 2 * np.pi / 17
        boundary = radius * np.column_stack((np.cos(angles), np.sin(angles)))
        # Compare exact one-sided traces, without a finite radial offset.
        radial = boundary / radius
        tangent = np.column_stack((-radial[:, 1], radial[:, 0]))
        for mode in self.modes:
            e, h = _fields(mode, nodes.xy, omega, z)
            norm, error = _power(mode, omega, z)
            if norm <= error:
                raise ModeError("full-domain power norm is non-positive or unresolved")
            ei, hi = _fields(mode, boundary, omega, z, exterior=False)
            eo, ho = _fields(mode, boundary, omega, z, exterior=True)
            eps_core = (
                float(
                    real_index_and_slope(mode._core, np.array([omega], dtype=float))[0][
                        0
                    ]
                )
                ** 2
            )
            eps_clad = (
                float(
                    real_index_and_slope(
                        mode._cladding, np.array([omega], dtype=float)
                    )[0][0]
                )
                ** 2
            )
            e_defect = np.column_stack(
                (
                    np.sum((ei[:, :2] - eo[:, :2]) * tangent, axis=1),
                    ei[:, 2] - eo[:, 2],
                    np.sum(
                        (eps_core * ei[:, :2] - eps_clad * eo[:, :2]) * radial, axis=1
                    )
                    / eps_core,
                )
            )
            defect = max(
                float(np.max(abs(e_defect))) / float(np.max(abs(ei))),
                float(np.max(abs(hi - ho))) / float(np.max(abs(hi))),
            )
            if defect > 1e-9:
                raise ModeError("step-index Maxwell boundary residual exceeds 1e-9")
            electric.append(e / np.sqrt(norm))
            magnetic.append(h / np.sqrt(norm))
            beta.append(
                float(mode.neff(np.array([omega], dtype=float), z)[0].real) * omega / C
            )
            residual.append(defect)
        e, h = np.stack(electric, axis=-1), np.stack(magnetic, axis=-1)
        metric = flux_product(e, h, nodes.weights)
        return ModalSample(
            omega,
            z,
            e,
            h,
            -1j * np.diag(beta),
            metric,
            np.array(residual),
            np.ones(len(beta), dtype=bool),
            e.conj(),
            -h.conj(),
            -4 * metric,
            "analytic-real-transverse",
            json.dumps(dict(self.identity), sort_keys=True, allow_nan=False),
        )
