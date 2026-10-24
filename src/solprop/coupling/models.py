# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The coupling models of SEP 31 section 4.3 and the scalar mode profiles they use.

:class:`CouplingModel` selects one of three models and carries its approximation
budget:

``same_basis``
    Exact coordinate recovery in a checked common reference: a copy for an identical
    basis, a declared invertible :class:`BasisChange`, or the flux-pairing recovery
    :math:`\mathbf a = F^{-1}\mathbf B` of a general modal plane into a port of the
    same guide. Not an interface calculation, so it takes no budget.
``scalar_paraxial``
    The weighted least-squares fit :math:`G = S^\dagger M S`,
    :math:`\mathbf a = G^{-1}S^\dagger M\mathbf f` of a scalar field onto scalar
    target profiles in the common paraxial measure :math:`M_0 = Y_0\,dA/2`,
    :math:`Y_0 = n_{\rm ref}\varepsilon_0c`, followed by the conversion to the
    target's native true-power coefficients :math:`b = \sqrt{\operatorname{Re}
    n_{\rm eff}}\,a`.
``vector_paraxial``
    The flux-pairing recovery :math:`\mathbf a = F^{-1}\mathbf B` onto a general
    modal provider's E/H at its nodes, with the incident H or the forward closure
    :math:`\mathbf H_t = Y_0\hat{\mathbf z}\times\mathbf E_t`.

:class:`ScalarModeBasis` is the scalar transverse profile set the scalar model uses:
the capillary :math:`HE_{1m}` family, :math:`J_0(u_m r/a)` inside the core and zero
outside (SEP 31 section 4.4), and the weak-guidance :math:`LP_{0m}` family of a
step-index fibre, :math:`J_0(Ur/a)` inside and :math:`J_0(U)K_0(Wr/a)/K_0(W)`
outside (Snyder and Love 1983, section 14-3; Gloge 1971). Each profile is normalised
to unit area integral of its square, and its order-zero Hankel transform is
evaluated in closed form, so an overlap can be taken in a plane's own spectral
quadrature without sampling a profile's kink.

The manual page is ``docs/guide/workflows/coupling.md``.
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
from jaxtyping import Complex, Float
from scipy.optimize import brentq
from scipy.special import j0, j1, jn_zeros, k0, k0e, k1, k1e

from solprop.constants import C
from solprop.exceptions import CouplingError
from solprop.grid import TimeGrid
from solprop.modes.capillary import MarcatiliMode
from solprop.modes.protocol import Mode
from solprop.modes.stepindex import StepIndexMode, _material

__all__ = [
    "MODEL_KINDS",
    "BasisChange",
    "CouplingModel",
    "ModelKind",
    "ProfileFamily",
    "ScalarModeBasis",
    "cone_angle",
    "radial_rule",
]

ModelKind = Literal["same_basis", "scalar_paraxial", "vector_paraxial"]
"""A coupling model of SEP 31 section 4.3."""

MODEL_KINDS: tuple[ModelKind, ...] = (
    "same_basis",
    "scalar_paraxial",
    "vector_paraxial",
)
"""Every :data:`ModelKind`, in a fixed order."""

ProfileFamily = Literal["capillary", "step_index"]
"""The scalar profile family of a :class:`ScalarModeBasis`."""

_PANEL_NODES = 24
"""Gauss--Legendre nodes per panel of the profile integrals.

A panel spans at most half a period of the Bessel functions involved, so 24 nodes
integrate the smooth piece on it to rounding; the panels break at every core radius,
where a profile's derivative jumps.
"""

_EXTERIOR_DECAY = math.log(1e17)
"""Decay, in units of the profile's own e-folding, of the last exterior node.

An LP profile's cladding tail falls as ``exp(-W r/a)``; at this many e-foldings its
square is below 1e-34 of the core value, beneath rounding of any overlap.
"""


def cone_angle(model_error: float) -> float:
    r"""Return the admitted cone half-angle :math:`\theta_*` of a model budget.

    SEP 31 sections 4.3 and 4.6: :math:`1 - \cos\theta_* = \epsilon_m/8`.

    Parameters
    ----------
    model_error : float
        The budget :math:`\epsilon_m`, in (0, 1).

    Returns
    -------
    float
        :math:`\theta_*` in rad.

    Examples
    --------
    >>> f"{cone_angle(0.08):.6f}"
    '0.141539'
    """
    return math.acos(1.0 - model_error / 8.0)


@dataclass(frozen=True, eq=False)
class BasisChange:
    """A declared invertible change of modal basis or gauge for ``same_basis``.

    The target coefficients are ``matrix @ b`` per bin and polarisation. The change
    must preserve energy between the plane's metric and the port's,
    ``T^H Q_B T = Q_A``, to :data:`solprop.coupling.port.METRIC_TOLERANCE`
    relative; it relabels the same physical field and is refused otherwise.

    Attributes
    ----------
    matrix : numpy.ndarray
        Optics-convention map from the plane's columns to the port's, shape
        ``(n_modes_target, n_modes_source)`` or with a trailing ``n_omega`` axis for
        a frequency-dependent gauge, dimensionless.
    source : str
        The plane's basis identity the change was declared for.
    target : str
        The port's basis identity.

    Raises
    ------
    CouplingError
        If the matrix is not finite and two- or three-dimensional.
    """

    matrix: Complex[np.ndarray, "n_modes_target n_modes_source *n_omega"]
    source: str
    target: str

    def __post_init__(self) -> None:
        """Own a read-only copy of the matrix."""
        matrix = np.array(self.matrix, dtype=np.complex128, copy=True)
        if matrix.ndim not in (2, 3) or not np.all(np.isfinite(matrix)):
            msg = "a basis change is a finite (n_target, n_source[, n_omega]) matrix"
            raise CouplingError(msg)
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    def at(self, n_omega: int) -> Complex[np.ndarray, "n_omega n_modes_t n_modes_s"]:
        """Return the matrix at every bin, frequency axis first.

        Parameters
        ----------
        n_omega : int
            The number of bins.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_omega, n_modes_target, n_modes_source)``.

        Raises
        ------
        CouplingError
            If a frequency-dependent matrix has another number of bins.
        """
        if self.matrix.ndim == 2:
            return np.broadcast_to(self.matrix, (n_omega, *self.matrix.shape))
        if self.matrix.shape[2] != n_omega:
            msg = f"the basis change has {self.matrix.shape[2]} bins, not {n_omega}"
            raise CouplingError(msg)
        return np.moveaxis(self.matrix, 2, 0)


@dataclass(frozen=True)
class CouplingModel:
    r"""Select a coupling model of SEP 31 section 4.3 and its approximation budget.

    Attributes
    ----------
    kind : {'same_basis', 'scalar_paraxial', 'vector_paraxial'}
        The model.
    model_error : float or None, optional
        The dimensionless energy budget :math:`\epsilon_m` in (0, 1) that the
        approximation's defects, angular tails, closure, passivity excess and
        numerical errors together must fit; required by the two approximate models,
        refused by ``same_basis``, which approximates nothing. It is separate from
        numerical accuracy and is not a coupling-efficiency floor.
    basis_change : BasisChange or None, optional
        A declared invertible change of basis or gauge, ``same_basis`` only.

    Raises
    ------
    CouplingError
        If a budget is missing, outside (0, 1) or given to ``same_basis``, or a
        basis change is given to an approximate model.

    Examples
    --------
    >>> CouplingModel("scalar_paraxial", model_error=0.05)
    CouplingModel(kind='scalar_paraxial', model_error=0.05, basis_change=None)
    """

    kind: ModelKind
    model_error: float | None = None
    basis_change: BasisChange | None = None

    def __post_init__(self) -> None:
        """Check the budget against the kind."""
        # The kind's membership is the Literal's: static typing and the test
        # session's runtime checks hold it, so no unreachable check is repeated here.
        if self.kind == "same_basis":
            if self.model_error is not None:
                msg = "same_basis approximates nothing, so it takes no model_error"
                raise CouplingError(msg)
            return
        if self.basis_change is not None:
            msg = "a basis change belongs to same_basis, not to an approximate model"
            raise CouplingError(msg)
        if self.model_error is None:
            msg = (
                f"{self.kind} is an approximation: give its energy budget "
                "model_error=, a number in (0, 1)"
            )
            raise CouplingError(msg)
        if not (math.isfinite(self.model_error) and 0.0 < self.model_error < 1.0):
            msg = f"model_error lies in (0, 1), got {self.model_error!r}"
            raise CouplingError(msg)

    @property
    def cone(self) -> float:
        """Return the admitted cone half-angle of the budget.

        Returns
        -------
        float
            :func:`cone_angle` of `model_error`, rad.

        Raises
        ------
        CouplingError
            For ``same_basis``, which has no budget.
        """
        if self.model_error is None:
            msg = "same_basis has no angular budget"
            raise CouplingError(msg)
        return cone_angle(self.model_error)


def radial_rule(
    breaks: Sequence[float], outer: float, *, panel: float
) -> tuple[Float[np.ndarray, " n_r"], Float[np.ndarray, " n_r"]]:
    """Return a composite Gauss--Legendre rule for ``int f(r) 2 pi r dr``.

    Panels break at each of `breaks` (a profile's derivative jumps there) and are at
    most `panel` long, so each holds a smooth piece of the integrand.

    Parameters
    ----------
    breaks : sequence of float
        Interior radii where the integrand is not smooth, m.
    outer : float
        The rule's outer radius, m.
    panel : float
        Largest panel length, m.

    Returns
    -------
    tuple of numpy.ndarray
        Nodes in m and area weights in m^2 (the Jacobian ``2 pi r`` included), each
        shape ``(n_r,)``.

    Examples
    --------
    >>> import numpy as np
    >>> r, w = radial_rule([1.0], 2.0, panel=0.5)
    >>> f"{np.sum(w):.12f}"  # the area of the disc of radius 2
    '12.566370614359'
    """
    edges = sorted({0.0, *(float(b) for b in breaks if 0.0 < b < outer), float(outer)})
    nodes, weights = np.polynomial.legendre.leggauss(_PANEL_NODES)
    radii: list[np.ndarray] = []
    area: list[np.ndarray] = []
    for low, high in itertools.pairwise(edges):
        count = max(1, math.ceil((high - low) / panel))
        bounds = np.linspace(low, high, count + 1)
        for left, right in itertools.pairwise(bounds):
            half = 0.5 * (right - left)
            r = left + half * (nodes + 1.0)
            radii.append(r)
            area.append(half * weights * 2.0 * np.pi * r)
    return np.concatenate(radii), np.concatenate(area)


def _hankel_core(
    u: float, q: Float[np.ndarray, " n_k"], radius: float
) -> Float[np.ndarray, " n_k"]:
    r"""Return :math:`\int_0^a J_0(ur/a)J_0(\kappa r)r\,dr` in closed form.

    The Lommel integral
    :math:`a^2[uJ_1(u)J_0(q) - qJ_0(u)J_1(q)]/(u^2 - q^2)`, :math:`q = \kappa a`;
    within 1e-4 of :math:`q = u`, where the quotient cancels, the integral is
    evaluated by quadrature instead.

    Parameters
    ----------
    u : float
        The profile's dimensionless transverse wavenumber.
    q : numpy.ndarray
        :math:`\kappa a`, dimensionless, shape ``(n_k,)``.
    radius : float
        The core radius a, m.

    Returns
    -------
    numpy.ndarray
        The integral in m^2, shape ``(n_k,)``.
    """
    q = np.asarray(q, dtype=np.float64)
    near = np.abs(q - u) < 1e-4 * max(u, 1.0)
    safe = np.where(near, u + 1.0, q)
    value = (u * j1(u) * j0(safe) - safe * j0(u) * j1(safe)) / (u * u - safe * safe)
    if np.any(near):
        # The quotient cancels where q meets u; there the integrand is smooth and
        # a 64-node Gauss--Legendre rule on [0, 1] integrates it to rounding.
        nodes, weights = np.polynomial.legendre.leggauss(64)
        x = 0.5 * (nodes + 1.0)
        direct = 0.5 * np.sum(
            (weights * x * j0(u * x))[None, :] * j0(q[near][:, None] * x[None, :]),
            axis=1,
        )
        value = value.copy()
        value[near] = direct
    return radius * radius * value


def _hankel_cladding(
    w: float, q: Float[np.ndarray, " n_k"], radius: float
) -> Float[np.ndarray, " n_k"]:
    r"""Return :math:`\int_a^\infty K_0(wr/a)J_0(\kappa r)r\,dr` in closed form.

    From the antiderivative
    :math:`x[\beta K_0(\alpha x)J_1(\beta x) - \alpha K_1(\alpha x)J_0(\beta x)]
    /(\alpha^2 + \beta^2)` of :math:`xK_0(\alpha x)J_0(\beta x)`, which vanishes at
    infinity: :math:`a^2[wK_1(w)J_0(q) - qK_0(w)J_1(q)]/(w^2 + q^2)`.

    Parameters
    ----------
    w : float
        The profile's dimensionless cladding decay.
    q : numpy.ndarray
        :math:`\kappa a`, dimensionless, shape ``(n_k,)``.
    radius : float
        The core radius a, m.

    Returns
    -------
    numpy.ndarray
        The integral in m^2, shape ``(n_k,)``.
    """
    q = np.asarray(q, dtype=np.float64)
    return radius * radius * (w * k1(w) * j0(q) - q * k0(w) * j1(q)) / (w * w + q * q)


def _lp_parameter(v: float, m: int) -> tuple[float, float]:
    r"""Solve the weak-guidance :math:`LP_{0m}` eigenvalue equation.

    :math:`UJ_1(U)/J_0(U) = WK_1(W)/K_0(W)` with :math:`U^2 + W^2 = V^2`
    (Gloge 1971, eq. 15 with l = 0), written without poles as
    :math:`UJ_1(U)K_0(W) - WK_1(W)J_0(U) = 0` and bracketed between consecutive
    zeros of :math:`J_1`, where the :math:`m`-th root lies.

    Parameters
    ----------
    v : float
        The normalised frequency V.
    m : int
        Radial order, at least 1.

    Returns
    -------
    tuple of float
        U and W.

    Raises
    ------
    CouplingError
        If the mode is below its cutoff at this V.
    """
    lower = 0.0 if m == 1 else float(jn_zeros(1, m - 1)[-1])
    if v <= lower * (1.0 + 1e-9):
        msg = f"LP0{m} is cut off at V = {v:.6g}"
        raise CouplingError(msg)
    upper = min(float(jn_zeros(0, m)[-1]), v)

    def equation(u: float) -> float:
        """Return the pole-free characteristic function at U.

        Parameters
        ----------
        u : float
            The core parameter U.

        Returns
        -------
        float
            The function's value, zero at an eigenvalue.
        """
        w = math.sqrt(max(v * v - u * u, 0.0))
        # Exponentially scaled K keeps W large finite; the common factor exp(-W)
        # is positive and does not move the root.
        return float(u * j1(u) * k0e(w) - w * k1e(w) * j0(u))

    tiny = 1e-12 * max(v, 1.0)
    u = brentq(equation, lower + tiny, upper - tiny, xtol=1e-15, rtol=1e-15)
    return float(u), math.sqrt(v * v - u * u)


def _waveguide_of(mode: Mode) -> Mode:
    """Return the waveguide mode a mode-averaged fibre was built from, or the mode.

    Parameters
    ----------
    mode : solprop.modes.protocol.Mode
        A run's mode.

    Returns
    -------
    solprop.modes.protocol.Mode
        The capillary or step-index mode its dispersion forwards to, when it has
        one; otherwise `mode` itself.
    """
    inner = getattr(getattr(mode, "dispersion", None), "mode", None)
    return inner if isinstance(inner, MarcatiliMode | StepIndexMode) else mode


@dataclass(frozen=True, eq=False)
class ScalarModeBasis:
    r"""Scalar transverse profiles of an ordinary mode set, bound to a temporal grid.

    Each channel is a profile :math:`v_m(r)` normalised to
    :math:`\int|v_m|^2dA = 1`, times a real unit Jones vector. The guide's own
    vacuum-normalised field per internal amplitude is
    :math:`\sqrt{2/(\varepsilon_0c)}\,v_m`, so a true-power coefficient
    :math:`b_m = \sqrt{\operatorname{Re}n_{{\rm eff},m}}a_m` carries the field
    :math:`\sqrt{2/(\varepsilon_0c\operatorname{Re}n_{{\rm eff},m})}\,b_m v_m`
    (``docs/dev/conventions.md`` section 4). The profiles are those of the run's
    own basis: a capillary's do not depend on frequency, and a step-index mode's are
    frozen at its declared profile wavelength exactly as the run's are.

    Build one with :meth:`from_modes`.

    Attributes
    ----------
    family : {'capillary', 'step_index'}
        The profile family.
    labels : tuple of str
        Channel labels, in column order.
    radius : float
        Core radius at the plane, m.
    u : numpy.ndarray
        Dimensionless core wavenumber of each profile, shape ``(n_modes,)``:
        the J0 zero of a capillary HE1m mode, U of an LP0m mode.
    w : numpy.ndarray
        Dimensionless cladding decay W of each LP0m profile, shape ``(n_modes,)``;
        zero for a capillary, whose profile vanishes outside the core.
    jones : numpy.ndarray
        Real unit Jones vector ``(x, y)`` of each channel, shape ``(n_modes, 2)``.
    effective_index : numpy.ndarray
        :math:`\operatorname{Re}n_{\rm eff}` of each channel at each bin, shape
        ``(n_modes, n_omega)``; zero on bins outside the active positive band.
    core_index : numpy.ndarray or None, optional
        Real core index at each bin, shape ``(n_omega,)``: a capillary's gas, a
        step-index fibre's core; zero off the active positive band.
    cladding_index : numpy.ndarray or None, optional
        Real cladding index at each bin of a step-index basis, shape
        ``(n_omega,)``.
    wall_term : numpy.ndarray or None, optional
        :math:`|\nu_n|` of a capillary's HE modes at each bin, shape
        ``(n_omega,)``: the Marcatili wall term
        :math:`\nu_n = (\varepsilon + 1)/(2\sqrt{\varepsilon - 1})` of the wall
        permittivity :math:`\varepsilon`; zero off the active positive band. It
        sets the leaky field's departure from the scalar profile (:meth:`leaky`).
    identity : str, optional
        A short description of the source modes, for reports.

    Raises
    ------
    CouplingError
        If the arrays disagree in shape.
    """

    family: ProfileFamily
    labels: tuple[str, ...]
    radius: float
    u: Float[np.ndarray, " n_modes"]
    w: Float[np.ndarray, " n_modes"]
    jones: Float[np.ndarray, "n_modes 2"]
    effective_index: Float[np.ndarray, "n_modes n_omega"]
    core_index: Float[np.ndarray, " n_omega"] | None = None
    cladding_index: Float[np.ndarray, " n_omega"] | None = None
    wall_term: Float[np.ndarray, " n_omega"] | None = None
    identity: str = ""
    _norms: Float[np.ndarray, " n_modes"] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Own read-only arrays and precompute each profile's normalisation."""
        n_modes = len(self.labels)
        for name, shape in (
            ("u", (n_modes,)),
            ("w", (n_modes,)),
            ("jones", (n_modes, 2)),
        ):
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if array.shape != shape or not np.all(np.isfinite(array)):
                msg = f"a scalar basis's {name} has shape {shape}, got {array.shape}"
                raise CouplingError(msg)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        index = np.array(self.effective_index, dtype=np.float64, copy=True)
        # The test session's shape checker refuses a wrong shape before this guard,
        # which is the only one outside it.
        if index.ndim != 2 or index.shape[0] != n_modes:  # pragma: no cover - see above
            msg = "a scalar basis's effective index has shape (n_modes, n_omega)"
            raise CouplingError(msg)
        index.setflags(write=False)
        object.__setattr__(self, "effective_index", index)
        for name in ("core_index", "cladding_index", "wall_term"):
            value = getattr(self, name)
            if value is not None:
                array = np.array(value, dtype=np.float64, copy=True)
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "_norms", self._square_integrals())

    def __repr__(self) -> str:
        """Return a one-line description without arrays.

        Returns
        -------
        str
            The family, channels and radius.
        """
        return (
            f"ScalarModeBasis({self.family}, {', '.join(self.labels)}, "
            f"a={self.radius:.6g} m)"
        )

    def leaky(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Float[np.ndarray, " n_omega"] | None:
        r"""Return a capillary's leaky-field parameter at each bin.

        Marcatili's leaky HE1m core field is :math:`J_0(u_m(1 - i\delta)r/a)` with
        :math:`\delta = \nu_n c/(\omega a)`, where the scalar profile has
        :math:`\delta = 0` (Marcatili and Schmeltzer 1964,
        doi:10.1002/j.1538-7305.1964.tb04108.x). Its energy-weighted mean over a
        coupling's occupied bins is the range parameter of
        :data:`~solprop.coupling.envelopes.CAPILLARY_ENVELOPE`.

        Parameters
        ----------
        omega : numpy.ndarray
            Absolute angular frequency of each bin, rad/s, shape ``(n_omega,)``.

        Returns
        -------
        numpy.ndarray or None
            :math:`|\nu_n|c/(\omega a)`, dimensionless, shape ``(n_omega,)``, zero
            off the active positive band; None for a basis without a wall term (a
            step-index one).
        """
        if self.wall_term is None:
            return None
        live = (self.wall_term > 0.0) & (omega > 0.0)
        values = np.zeros(omega.shape)
        values[live] = self.wall_term[live] * C / (omega[live] * self.radius)
        return values

    @property
    def n_modes(self) -> int:
        """Return the number of channels.

        Returns
        -------
        int
            ``len(labels)``.
        """
        return len(self.labels)

    def _square_integrals(self) -> Float[np.ndarray, " n_modes"]:
        r"""Return :math:`\int|\hat v_m|^2dA` of each unnormalised profile, m^2.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_modes,)``: :math:`\pi a^2J_1(u)^2` for a capillary profile,
            :math:`\pi a^2[J_0^2 + J_1^2](U) + \pi a^2(J_0(U)/K_0(W))^2
            [K_1^2 - K_0^2](W)` for an LP profile.
        """
        area = math.pi * self.radius**2
        core = area * (j0(self.u) ** 2 + j1(self.u) ** 2)
        if self.family == "capillary":
            return core
        ratio = j0(self.u) / k0(self.w)
        return core + area * ratio**2 * (k1(self.w) ** 2 - k0(self.w) ** 2)

    def outer(self) -> float:
        """Return the radius beyond which every profile is negligible, m.

        Returns
        -------
        float
            The core radius for a capillary; for an LP set, where the slowest
            cladding tail has decayed by 39 e-foldings, ``log(1e17)``.
        """
        if self.family == "capillary":
            return self.radius
        return self.radius * (1.0 + _EXTERIOR_DECAY / float(np.min(self.w)))

    def values(self, r: Float[np.ndarray, " n_r"]) -> Float[np.ndarray, "n_modes n_r"]:
        """Return each normalised profile at radii.

        Parameters
        ----------
        r : numpy.ndarray
            Non-negative radii, m, shape ``(n_r,)``.

        Returns
        -------
        numpy.ndarray
            :math:`v_m(r)` in 1/m, shape ``(n_modes, n_r)``.
        """
        x = np.asarray(r, dtype=np.float64)[None, :] / self.radius
        inside = x <= 1.0
        core = j0(self.u[:, None] * x)
        if self.family == "capillary":
            profile = np.where(inside, core, 0.0)
        else:
            ratio = (j0(self.u) / k0(self.w))[:, None]
            tail = ratio * k0(self.w[:, None] * np.maximum(x, 1.0))
            profile = np.where(inside, core, tail)
        return profile / np.sqrt(self._norms)[:, None]

    def hankel(
        self, kappa: Float[np.ndarray, " n_k"]
    ) -> Float[np.ndarray, "n_modes n_k"]:
        r"""Return each normalised profile's order-zero Hankel transform.

        :math:`\tilde v_m(\kappa) = \int_0^\infty v_m(r)J_0(\kappa r)r\,dr`, the
        convention of :mod:`solprop.hankel`, in closed form; the two-dimensional
        Fourier transform of the axisymmetric profile is :math:`2\pi\tilde v_m`.

        Parameters
        ----------
        kappa : numpy.ndarray
            Transverse wavenumbers, rad/m, non-negative, shape ``(n_k,)``.

        Returns
        -------
        numpy.ndarray
            In m, shape ``(n_modes, n_k)``.
        """
        q = np.asarray(kappa, dtype=np.float64) * self.radius
        rows = []
        for one in range(self.n_modes):
            value = _hankel_core(float(self.u[one]), q, self.radius)
            if self.family == "step_index":
                ratio = float(j0(self.u[one]) / k0(self.w[one]))
                value = value + ratio * _hankel_cladding(
                    float(self.w[one]), q, self.radius
                )
            rows.append(value / math.sqrt(float(self._norms[one])))
        return np.array(rows)

    def gram(self) -> Float[np.ndarray, "n_modes n_modes"]:
        """Return the exact whole-plane Gram of the normalised profiles.

        Distinct profiles of one family are orthogonal (different zeros of the
        core's eigenvalue equation); the same profile carried by two channels, the
        two polarisations of one mode, overlaps itself fully, and their Jones
        vectors make the channels orthogonal.

        Returns
        -------
        numpy.ndarray
            One where two profiles are the same and zero otherwise, shape
            ``(n_modes, n_modes)``.
        """
        same = (self.u[:, None] == self.u[None, :]) & (
            self.w[:, None] == self.w[None, :]
        )
        return same.astype(np.float64)

    def cone_gram(self, kappa_star: float) -> Float[np.ndarray, "n_modes n_modes"]:
        r"""Return the Gram of the profiles' angular content inside a cone.

        :math:`\int_{|\boldsymbol\kappa|<\kappa_*}\tilde V_m\tilde V_n
        d^2\kappa/(2\pi)^2
        = \int_0^{\kappa_*}\tilde v_m\tilde v_n\,2\pi\kappa\,d\kappa`,
        by composite Gauss--Legendre panels of at most half a period of
        :math:`J_0(\kappa a)`. The tail outside the cone is the whole-plane
        :meth:`gram` minus this.

        Parameters
        ----------
        kappa_star : float
            The cone's transverse wavenumber, rad/m.

        Returns
        -------
        numpy.ndarray
            Real symmetric, dimensionless, shape ``(n_modes, n_modes)``.
        """
        if kappa_star <= 0.0:
            return np.zeros((self.n_modes, self.n_modes))
        # The transform of a compactly supported or exponentially decaying profile
        # is smooth in kappa, so panels of half a period of J0(kappa a) suffice.
        nodes, weights = radial_rule([], kappa_star * self.radius, panel=0.5 * math.pi)
        kappa = nodes / self.radius
        transforms = self.hankel(kappa)
        # radial_rule's weights are 2 pi q dq; dividing by a^2 gives 2 pi k dk.
        return (transforms * (weights / self.radius**2)[None, :]) @ transforms.T

    @classmethod
    def from_modes(
        cls, modes: Sequence[Mode], grid: TimeGrid, z: float = 0.0
    ) -> "ScalarModeBasis":  # noqa: UP037 - beartype evaluates the class body
        """Return the scalar profiles of an ordinary mode set at a plane.

        Admitted are a capillary's :math:`HE_{1m}` modes (Marcatili; the scalar
        J0 profile is exactly their transverse field) and a step-index fibre's
        :math:`HE_{1m}` modes taken in the weak-guidance limit as :math:`LP_{0m}`.
        Every other family has angular structure the scalar model cannot hold and is
        refused by name.

        Parameters
        ----------
        modes : sequence of solprop.modes.protocol.Mode
            The run's modes, in column order.
        grid : solprop.grid.TimeGrid
            The run's temporal grid, on whose active positive bins the effective
            indices are evaluated.
        z : float, optional
            The plane's position along the guide, m.

        Returns
        -------
        ScalarModeBasis
            The profiles.

        Raises
        ------
        CouplingError
            For an empty set, mixed or unsupported families, or modes of different
            radii.
        """
        if not modes:
            msg = "a scalar basis needs at least one mode"
            raise CouplingError(msg)
        live = grid.active_mask & (grid.omega > 0)
        omega = grid.omega[live]
        index = np.zeros((len(modes), grid.n_omega))
        for row, mode in enumerate(modes):
            index[row, live] = np.real(np.asarray(mode.neff(omega, z)))
        labels = tuple(mode.channel_label.display_label for mode in modes)
        # A mode-averaged GNLSE fibre wraps the waveguide it was built from (its
        # dispersion forwards to that mode); the profile is the waveguide's, the
        # effective index the run's own.
        modes = [_waveguide_of(mode) for mode in modes]
        if all(isinstance(mode, MarcatiliMode) for mode in modes):
            return cls._capillary(
                cast("Sequence[MarcatiliMode]", modes), labels, index, grid, z
            )
        if all(isinstance(mode, StepIndexMode) for mode in modes):
            return cls._step_index(
                cast("Sequence[StepIndexMode]", modes), labels, index, grid, z
            )
        msg = (
            "the scalar model has profiles for capillary HE1m modes and weakly "
            f"guiding step-index HE1m modes; got {[type(m).__name__ for m in modes]}"
        )
        raise CouplingError(msg)

    @classmethod
    def _capillary(
        cls,
        modes: Sequence[MarcatiliMode],
        labels: tuple[str, ...],
        index: Float[np.ndarray, "n_modes n_omega"],
        grid: TimeGrid,
        z: float,
    ) -> "ScalarModeBasis":  # noqa: UP037 - beartype evaluates the class body
        """Return the J0 profiles of capillary HE1m modes.

        Parameters
        ----------
        modes : sequence of MarcatiliMode
            The modes.
        labels : tuple of str
            Their labels.
        index : numpy.ndarray
            Their real effective indices, shape ``(n_modes, n_omega)``.
        grid : solprop.grid.TimeGrid
            The temporal grid, for the core gas's index.
        z : float
            The plane's position, m.

        Returns
        -------
        ScalarModeBasis
            The profiles.

        Raises
        ------
        CouplingError
            For a family other than HE1m, or modes of different radii.
        """
        radii = {float(mode.dimlimits(z)[2][0]) for mode in modes}
        if len(radii) != 1:
            msg = f"the capillary modes have different radii {sorted(radii)}"
            raise CouplingError(msg)
        for mode, label in zip(modes, labels, strict=True):
            if mode.kind != "HE" or mode.n != 1:
                msg = (
                    f"{label} has angular structure: the scalar model holds the "
                    "capillary's HE1m modes only"
                )
                raise CouplingError(msg)
        jones = np.array([[math.cos(mode.phi), math.sin(mode.phi)] for mode in modes])
        live = grid.active_mask & (grid.omega > 0)
        core = np.zeros(grid.n_omega)
        core[live] = 1.0 + np.real(
            np.asarray(modes[0].core_index_minus_one(grid.omega[live], z))
        )
        wall = np.zeros(grid.n_omega)
        wall[live] = np.abs(modes[0].sampled_terms(grid.omega[live]).cladding_term)
        radius = radii.pop()
        return cls(
            "capillary",
            labels,
            radius,
            np.array([mode.u_nm for mode in modes]),
            np.zeros(len(modes)),
            jones,
            index,
            core,
            wall_term=wall,
            identity=f"capillary HE1m x{len(modes)}, a={radius:.6g} m",
        )

    @classmethod
    def _step_index(
        cls,
        modes: Sequence[StepIndexMode],
        labels: tuple[str, ...],
        index: Float[np.ndarray, "n_modes n_omega"],
        grid: TimeGrid,
        z: float,
    ) -> "ScalarModeBasis":  # noqa: UP037 - beartype evaluates the class body
        """Return the weak-guidance LP0m profiles of step-index HE1m modes.

        Parameters
        ----------
        modes : sequence of StepIndexMode
            The modes, of one fibre.
        labels : tuple of str
            Their labels.
        index : numpy.ndarray
            Their real effective indices, shape ``(n_modes, n_omega)``.
        grid : solprop.grid.TimeGrid
            The temporal grid.
        z : float
            The plane's position, m.

        Returns
        -------
        ScalarModeBasis
            The profiles, frozen at the modes' profile wavelength.

        Raises
        ------
        CouplingError
            For a family other than HE1m, or modes of different fibres.
        """
        first = modes[0]
        for mode, label in zip(modes, labels, strict=True):
            if mode.kind != "HE" or mode.n != 1:
                msg = (
                    f"{label} has angular structure: the weak-guidance scalar model "
                    "holds a step-index fibre's HE1m (LP0m) modes only"
                )
                raise CouplingError(msg)
            if (
                mode.profile_wavelength != first.profile_wavelength
                or mode._core != first._core
                or mode._cladding != first._cladding
                or mode.dimlimits(z)[2] != first.dimlimits(z)[2]
            ):
                msg = "step-index modes of one scalar basis share one fibre and profile"
                raise CouplingError(msg)
        radius = float(first._radius(z))
        omega_p = 2.0 * math.pi * 299792458.0 / first.profile_wavelength
        core_p = float(_material(first._core, np.array([omega_p]))[0][0])
        cladding_p = float(_material(first._cladding, np.array([omega_p]))[0][0])
        v = omega_p / 299792458.0 * radius * math.sqrt(core_p**2 - cladding_p**2)
        parameters = [_lp_parameter(v, mode.m) for mode in modes]
        live = grid.active_mask & (grid.omega > 0)
        core = np.zeros(grid.n_omega)
        cladding = np.zeros(grid.n_omega)
        core[live] = _material(first._core, grid.omega[live])[0]
        cladding[live] = _material(first._cladding, grid.omega[live])[0]
        jones = np.array(
            [[1.0, 0.0] if mode.parity == "even" else [0.0, 1.0] for mode in modes]
        )
        return cls(
            "step_index",
            labels,
            radius,
            np.array([one[0] for one in parameters]),
            np.array([one[1] for one in parameters]),
            jones,
            index,
            core,
            cladding,
            identity=(
                f"step-index LP0m x{len(modes)}, a={radius:.6g} m, "
                f"n={core_p:.6g}/{cladding_p:.6g} at {first.profile_wavelength:.6g} m"
            ),
        )
