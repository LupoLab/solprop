# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Derive the connection of a basis whose material interfaces move along z.

A guide whose transverse profile moves is written in a **moving basis**, and a
moving basis costs a term. With the field expanded on the local modes at every
position, :math:`E(r, z) = \sum_n c_n(z)\,\psi_n(r; z)`, the coefficients obey

.. math::
   \frac{dc}{dz} = \bigl(\text{the local generator}\bigr) c - \Gamma c ,
   \qquad \Gamma = W^{-1} S , \qquad
   S_{mn} = \langle \psi_m \mid \partial_z \psi_n \rangle ,

where :math:`\langle\cdot\mid\cdot\rangle` is **this package's own flux metric**
-- the Hermitian Poynting pairing of :func:`solprop.modes.vector.flux_product` --
and :math:`W` is that pairing's Gram matrix, the same
:attr:`~solprop.modes.vector.ModalSample.flux_metric` the source projection map
``Q`` is built from. Using one pairing for the dual, the projection and the
connection is what makes the three consistent; the bilinear reciprocity pairing
never enters, exactly as for the source path.

**The flux self-terms are kept**, and this module exists because they must be.
Differentiating the Gram matrix gives the one identity the whole certificate rests
on:

.. math::
   W' = \frac{dW}{dz} = \langle \partial_z\psi_m \mid \psi_n\rangle
        + \langle \psi_m \mid \partial_z\psi_n\rangle = S^{H} + S ,
   \qquad\text{equivalently}\qquad
   W\Gamma + \Gamma^{H} W = W' .

So :math:`\Gamma` is anti-Hermitian **with respect to the flux metric**, up to that
metric's own drift, and the consequence is exact: with
:math:`dc/dz = -\Gamma c` the flux norm obeys
:math:`d(c^{H} W c)/dz = c^{H}(W' - W\Gamma - \Gamma^{H}W)c = 0`. The connection
alone moves no power. Zeroing the diagonal -- which the Bessel formula for the
capillary does, correctly, for *its* basis -- breaks that identity. For diagonal
:math:`W`, the resulting norm defect is
:math:`\sum_n |c_n|^2 W'_{nn}`, which is why
:func:`solprop.modes.capillary.wall_connection`'s formula is **not** transplanted
here.

The diagonal splits into the two pieces the identity names:

* for diagonal :math:`W`, its real part is :math:`W'_{nn}/(2W_{nn})`, reducing
  to :math:`\tfrac12 W'_{nn}` where :math:`W_{nn}=1`. This flux self-term
  describes the rate of change of the mode's power normalisation. For a dense
  metric the full matrix identity must be used;
* its imaginary part is the **gauge phase rate**, a geometric (Berry) connection.
  It vanishes only in a gauge that parallel-transports the phase, which is what
  :func:`solprop.translate.modal_tables.align_modal_sample` does to first order by
  making the overlap with the previous knot real and positive.

Neither is zero in general for a moving interface, and a declared z-varying change
of coordinates :math:`c = T(z) d` adds :math:`T^{H}T'` to the connection (and
its negative to the generator) in the
declared basis, whose diagonal is pure imaginary and is
exactly the second piece.

**How it is computed.** :math:`\partial_z\psi_n` is a three-point Lagrange
derivative of the provider's own solves, tracked into one gauge, on **one shared
node set whose panel edges land on every stencil position's interfaces**
(:func:`connection_nodes`). That last point is not a detail: a rule built at one
position and reused at another leaves its panel edge off the moving interface, and
the quadrature of a Poynting integrand with a jump at the interface is then wrong
by a term first order in the displacement -- which is first order in the step and
therefore survives the difference undiminished. Measured on a step-index taper it
puts a spurious ``0.98`` 1/m on the diagonal and reads ``|W'| = 2.8`` 1/m where the
true value is below ``1e-8``. Splitting the panels at
every stencil radius restores piecewise-smooth quadrature. Finite quadrature,
mode-solve and derivative errors remain and require independent refinement.

The step is chosen so that the difference's truncation, which falls as ``h^2``, and
its round-off, which grows as ``1/h``, are balanced (:data:`STEP_FRACTION`). The
measured :attr:`ModalConnection.antisymmetry` is the certificate, and it is the
residual of the identity above with ``W'`` differenced independently of the pairing,
in 1/m.

**What this is not.** The connection describes the basis the provider retains. What
lies outside the retained set does not vanish at a material interface while every
retained basis function's expansion of it converges only in :math:`L^2`, so a
propagated coefficient departs from the direct projection of the same frozen field
at **second order in the total contraction** -- as the capillary's wall connection
does in the stated comparisons. This observed scaling is not a general error
bound for arbitrary taper shapes or contractions.
"""

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex, Float

from solprop._display import Displayed, engineering, line
from solprop.exceptions import GeneralModalBasisWarning, ModeError
from solprop.modes.guide import ChannelBasis, radial_node_rule
from solprop.modes.material import JsonValue
from solprop.modes.vector import ModalNodeRule, ModalProvider, ModalSample
from solprop.profiles import as_profile

__all__ = [
    "CERTIFICATE_WARNING_RTOL",
    "MAX_STENCIL_SPAN",
    "NEGLIGIBLE_CONVERSION",
    "STEP_FRACTION",
    "ModalConnection",
    "band_connection",
    "basis_connection",
    "connection_nodes",
    "modal_connection",
]

STEP_FRACTION: float = (16.0 * float(np.finfo(float).eps)) ** (1.0 / 3.0)
"""Difference step as a fraction of the z scale, dimensionless: 1.53e-5.

The three-point derivative's truncation falls as ``(h/L)^2`` and its round-off grows
as ``delta L/h``, where ``L`` is the scale the sampled fields vary over and ``delta``
is the relative noise of a solve; the two are equal at ``h/L = (delta/2)^(1/3)``.
``delta = 16 eps`` -- the rounding of a short reduction, not of one operation -- is
what a provider that solves an eigenproblem and integrates a quadrature actually
carries, and the measurement agrees. On a step-index taper of ``L = 1e-3`` m and a
connection whose own scale is 3.05e2 1/m, the certificate
:attr:`ModalConnection.antisymmetry` reads 5.3e-10, 2.8e-10, **7.4e-11**, 5.6e-10,
2.4e-9 and 9.5e-9 relative at ``h/L`` of 6.1e-6, 1.0e-5, **1.53e-5**, 3.0e-5, 6.0e-5
and 1.2e-4, and the connection itself agrees to ten digits across the whole scan. A
caller whose provider is noisier declares its own step and reads the certificate.
"""

MAX_STENCIL_SPAN: float = 0.25
"""Largest share of the z scale a difference stencil may span, dimensionless.

The derivative is local, so a stencil that reaches a quarter of the way across the
scale it was sized from is not measuring a local slope. A caller that asks for a
step this large is refused rather than given a secant.

It is also what makes the three positions **fit**: at a quarter of the scale a
stencil clamped against either end of the interval still lies inside it, so
:func:`modal_connection` needs no second guard for that. A value above a half would,
which is why this one is not a tuning knob.
"""

CERTIFICATE_WARNING_RTOL: float = 1e-4
"""Relative antisymmetry above which a measured connection warns, dimensionless.

``antisymmetry / max|Gamma|``. The default step meets 7.4e-11 and 5.7e-9 on the two
qualified providers and the tests admit 1e-8 there; a stencil shifted onto one side
of a declared kink reads 1e-5, round-off limited and harmless; a caller declaring a
step of a fifth of the scale reads 2.6e-2 with the connection itself 1.2e-2 wrong
while still inside :data:`MAX_STENCIL_SPAN`. The warning sits at 1e-4, where the
connection is itself wrong at a part in ten thousand, so the certificate is what says
a declared step was too coarse.
:func:`modal_connection` warns by name above it and does not refuse: the certificate
is the caller's to read, and a noisy provider may legitimately sit above it.
"""

NEGLIGIBLE_CONVERSION: float = 1e-6
"""Below this much conversion over the z domain a connection is rounding, dimensionless.

``max|Gamma| * (z_high - z_low)``: a degenerate pair's connection is zero by symmetry
and reads ~1e-8 1/m of solve noise, so its relative certificate is meaningless and
:func:`modal_connection` does not warn on it.
"""


@dataclass(frozen=True, eq=False)
class ModalConnection(Displayed):
    """Carry one position's moving-basis connection with the certificate it met.

    Attributes
    ----------
    matrix : numpy.ndarray
        The connection ``Gamma = W^-1 S`` in 1/m, complex, shape
        ``(n_modes_connection_out, n_modes_connection_in)``. The coefficients obey
        ``dc/dz = ... - Gamma c``.
    metric : numpy.ndarray
        The flux Gram ``W`` at the same position, dimensionless and Hermitian,
        shape ``(n_modes_metric_out, n_modes_metric_in)``.
    metric_rate : numpy.ndarray
        Its derivative ``W'`` in 1/m, same shape, differenced from the stencil's
        own Gram matrices. The derivation makes it ``S + S^H``, twice the
        Hermitian part of the pairing, and :attr:`antisymmetry` is what that
        costs to meet.
    antisymmetry : float
        The certificate, in 1/m: ``norm(W Gamma + Gamma^H W - W', 2)`` with ``W'``
        differenced independently of the pairing, which the derivation makes
        exactly zero. What is measured is the stencil's own truncation and
        round-off -- and a quadrature whose panel edge has left a moving
        interface, which is what it exists to catch.
    step : float
        The difference step actually used, m, positive.
    z : float
        The position the connection was taken at, m.
    omega : float
        The frequency it was taken at, rad/s, positive.

    Raises
    ------
    solprop.exceptions.ModeError
        If the matrices are not square and equally wide, or a scalar is out of
        range.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.connection import ModalConnection
    >>> link = ModalConnection(
    ...     np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex),
    ...     np.eye(2, dtype=complex), np.zeros((2, 2), dtype=complex),
    ...     0.0, 1e-9, 0.5, 2.3e15,
    ... )
    >>> link.n_modes, float(np.max(abs(link.self_terms)))
    (2, 0.0)
    """

    matrix: Complex[np.ndarray, "n_modes_connection_out n_modes_connection_in"]
    metric: Complex[np.ndarray, "n_modes_metric_out n_modes_metric_in"]
    metric_rate: Complex[np.ndarray, "n_modes_rate_out n_modes_rate_in"]
    antisymmetry: float
    step: float
    z: float
    omega: float

    @property
    def n_modes(self) -> int:
        """Return how many channels the connection couples.

        Returns
        -------
        int
            The channel count.
        """
        return int(self.matrix.shape[0])

    @property
    def self_terms(self) -> Complex[np.ndarray, " n_modes_self"]:
        """Return the diagonal: the flux self-term and the gauge phase rate.

        The real part is ``W'_nn/2``, the rate the channel's own power
        normalisation moves; the imaginary part is the gauge's phase rate. Neither
        is zero in general for a moving material interface.

        Returns
        -------
        numpy.ndarray
            Complex, shape ``(n_modes,)``, in 1/m.
        """
        return np.diag(self.matrix)

    def __repr__(self) -> str:
        """Return a one-line description, never a matrix.

        Returns
        -------
        str
            For example ``ModalConnection(2 channels, 3.05e+02 1/m, 500 um)``.
        """
        return line(
            "ModalConnection",
            [
                ("", f"{self.n_modes} channel{'' if self.n_modes == 1 else 's'}"),
                ("", f"{float(np.max(abs(self.matrix))):.2e} 1/m"),
                ("", engineering(float(self.z), "m")),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "ModalConnection: the cost of a basis that moves along z", [
            ("channels", self.n_modes),
            ("z", engineering(float(self.z), "m")),
            ("omega", f"{float(self.omega):.6e} rad/s"),
            ("largest", f"{float(np.max(abs(self.matrix))):.4e} 1/m"),
            ("self terms", f"{float(np.max(abs(self.self_terms))):.4e} 1/m largest"),
            ("metric departure", f"{float(np.max(abs(self.metric_rate))):.4e} 1/m"),
            ("antisymmetry", f"{self.antisymmetry:.4e} 1/m"),
            ("step", engineering(float(self.step), "m")),
        ]

    def __post_init__(self) -> None:
        """Copy, validate and freeze the three matrices.

        Raises
        ------
        solprop.exceptions.ModeError
            Naming what is out of range.
        """
        for name in ("matrix", "metric", "metric_rate"):
            values = np.array(
                getattr(self, name), dtype=np.complex128, order="C", copy=True
            )
            if (
                values.ndim != 2
                or values.shape[0] != values.shape[1]
                or not values.shape[0]
                or values.shape != np.shape(self.matrix)
            ):
                raise ModeError(
                    f"a modal connection's {name} must be a non-empty square matrix "
                    f"as wide as its connection, got {values.shape}"
                )
            if not np.all(np.isfinite(values)):
                raise ModeError(f"a modal connection's {name} must be finite")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        for name in ("antisymmetry", "step", "omega"):
            value = float(getattr(self, name))
            floor = 0.0 if name == "antisymmetry" else None
            if not math.isfinite(value) or (
                value < 0.0 if floor is not None else value <= 0.0
            ):
                raise ModeError(
                    f"a modal connection's {name} is a finite non-negative number, "
                    f"got {getattr(self, name)!r}"
                )
            object.__setattr__(self, name, value)
        if not math.isfinite(float(self.z)):
            raise ModeError(f"a modal connection's z must be finite, got {self.z!r}")
        object.__setattr__(self, "z", float(self.z))


def basis_connection(
    transform: Complex[np.ndarray, "n_modes_t_out n_modes_t_in"],
    derivative: Complex[np.ndarray, "n_modes_d_out n_modes_d_in"],
) -> Complex[np.ndarray, "n_modes_b_out n_modes_b_in"]:
    r"""Return the connection a declared z-varying change of coordinates costs.

    In general, for :math:`c = T(z) d` the declared
    coordinates carry :math:`-T^{-1}T'` beside the transformed generator, so the
    connection in the declared basis gains :math:`T^{-1}T'`. For a **unitary**
    ``T`` -- the only kind :class:`~solprop.modes.guide.ChannelBasis` declares --
    that is :math:`T^{H}T'`, which needs no inverse and is exactly anti-Hermitian
    because differentiating :math:`T^{H}T = I` gives
    :math:`T^{H}T' + (T^{H}T')^{H} = 0`. Its diagonal is therefore pure imaginary
    and is the gauge phase rate of this module's opening derivation: a *relabelling*
    that produces no physical conversion, which is the pure change-of-basis test.

    Parameters
    ----------
    transform : numpy.ndarray
        The unitary ``T(z)``, shape ``(n_modes, n_modes)``, dimensionless.
    derivative : numpy.ndarray
        ``T'(z)`` in 1/m, the same shape.

    Returns
    -------
    numpy.ndarray
        ``T^H T'`` in 1/m, anti-Hermitian, shape ``(n_modes, n_modes)``.

    Raises
    ------
    solprop.exceptions.ModeError
        If the two are not square matrices of the same width.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.connection import basis_connection
    >>> angle, rate = 0.3, 2.0
    >>> turn = np.array([[np.cos(angle), -np.sin(angle)],
    ...                  [np.sin(angle), np.cos(angle)]], dtype=complex)
    >>> slope = rate * np.array([[-np.sin(angle), -np.cos(angle)],
    ...                          [np.cos(angle), -np.sin(angle)]], dtype=complex)
    >>> link = basis_connection(turn, slope)
    >>> bool(np.allclose(link, -link.conj().T)), round(float(np.real(link[0, 1])), 12)
    (True, -2.0)
    """
    left = np.asarray(transform, dtype=np.complex128)
    right = np.asarray(derivative, dtype=np.complex128)
    if (
        left.ndim != 2
        or left.shape[0] != left.shape[1]
        or not left.shape[0]
        or right.shape != left.shape
    ):
        raise ModeError(
            "a declared basis connection takes a square transform and its "
            f"derivative of the same width, got {left.shape} and {right.shape}"
        )
    return left.conj().T @ right


def connection_nodes(
    provider: ModalProvider,
    omega: float,
    positions: Sequence[float] | Float[np.ndarray, " n_z"],
    *,
    nodes: ModalNodeRule | None = None,
    n_r: int | None = None,
    n_theta: int | None = None,
) -> tuple[ModalNodeRule, ...]:
    """Build a shared node set split at every stencil position's panel edges.

    The rules returned share their coordinates and their area weights **exactly** --
    that is what makes a difference of the sampled fields meaningful -- and differ
    only in ``region_index``, which each position computes from its own material
    radii. The radial panels are split at the union of **every position's own declared
    panel edges**, so each solve's integrand is smooth on every panel and quadrature can
    converge on its smooth pieces, which a single frozen rule cannot ensure (see this
    module's own docstring for what a frozen rule costs).

    Taking the union of the declared edges rather than of the material radii alone
    is what makes this work for a discretised provider: a graded-index sector is a
    P1 finite-element field whose slope breaks at every element edge, and its whole
    mesh scales with the core radius, so a rule split at the interface alone
    integrates a kinked function with a Gauss rule and reads a flux metric 0.39 from
    the identity (measured).

    Parameters
    ----------
    provider : solprop.modes.vector.ModalProvider
        The guide, whose ``regions`` declare the radii that move and whose own
        ``node_rule`` declares the panel edges at each position.
    omega : float
        Positive angular frequency in rad/s, at which each position's rule is asked
        for.
    positions : sequence of float or numpy.ndarray
        The stencil positions in m, inside the provider's z domain, shape
        ``(n_z,)``.
    nodes : solprop.modes.vector.ModalNodeRule or None, optional
        A further rule whose panel edges and resolution join the union, so that a
        caller can measure the connection on a superset of the quadrature its run
        uses. ``None`` takes the provider's own rules alone.
    n_r, n_theta : int or None, optional
        The resolution each position's own rule is asked for, forwarded to the
        provider's ``node_rule``; ``None`` takes the provider's own default, which is
        what :func:`modal_connection` measures at. A moving guide's path passes the
        resolution its own sampling rule declares, so that the connection and the
        maps it is propagated beside are resolved alike.

    Returns
    -------
    tuple of ModalNodeRule
        One rule per position, in the same order.

    Raises
    ------
    solprop.exceptions.ModeError
        If a rule does not declare the panel edges, the radial resolution and the
        truncation radius this needs, or none of its edges lies inside the domain.
    """
    at = tuple(float(one) for one in positions)
    declared = [
        provider.node_rule(float(omega), one, n_r=n_r, n_theta=n_theta) for one in at
    ]
    if nodes is not None:
        declared.append(nodes)
    identities = [_panel_identity(provider, rule.identity) for rule in declared]
    outer = max(one[0] for one in identities)
    edges = sorted(
        {
            radius
            for one in identities
            for radius in one[1]
            if 0.0 < float(radius) < outer
        }
    )
    if not edges:
        raise ModeError(
            f"a moving-basis connection splits its panels at the union of every "
            f"stencil position's declared edges and {type(provider).__name__} "
            f"declares none inside the {outer!r} m truncation radius at {at}"
        )
    rule = radial_node_rule(
        tuple(edges),
        outer,
        n_r=max(one[2] for one in identities),
        n_theta=max(one[3] for one in identities),
    )
    distance = np.hypot(rule.xy[:, 0], rule.xy[:, 1])
    return tuple(
        ModalNodeRule(
            rule.xy,
            rule.weights,
            np.searchsorted(
                np.asarray(_interfaces(provider, one)), distance, side="left"
            ).astype(np.int32),
            dict(declared[index].identity),
        )
        for index, one in enumerate(at)
    )


def _panel_identity(
    provider: ModalProvider, identity: Mapping[str, JsonValue]
) -> tuple[float, tuple[float, ...], int, int]:
    """Return the truncation radius, panel edges and resolution a rule declares.

    Parameters
    ----------
    provider : solprop.modes.vector.ModalProvider
        The guide, named in the refusal.
    identity : collections.abc.Mapping
        One rule's own identity.

    Returns
    -------
    tuple
        The outer radius in m, its panel edges in m, and the two node counts.

    Raises
    ------
    solprop.exceptions.ModeError
        If the rule does not declare all four.
    """
    scalars = ("outer_radius", "n_r", "n_theta")
    missing = [
        name
        for name in (*scalars, "interfaces")
        if not isinstance(identity.get(name), float | int | list)
        or isinstance(identity.get(name), bool)
    ]
    if missing:
        raise ModeError(
            f"a moving-basis connection refines the node rule onto every stencil "
            f"position's panel edges and needs the rule's own {missing} to do it; "
            f"{type(provider).__name__}'s rule declares {sorted(identity)}. A "
            "provider whose rule is not a radial panel rule must supply one that is"
        )
    outer, radial, angular = (float(identity[name]) for name in scalars)  # type: ignore[arg-type]
    edges = identity["interfaces"]
    assert isinstance(edges, list)
    return outer, tuple(float(one) for one in edges), int(radial), int(angular)  # type: ignore[arg-type]


def _interfaces(provider: ModalProvider, z: float) -> tuple[float, ...]:
    """Return a provider's declared material radii at one position, increasing.

    Parameters
    ----------
    provider : solprop.modes.vector.ModalProvider
        The guide.
    z : float
        Position in m.

    Returns
    -------
    tuple of float
        The distinct positive finite radii in m, increasing.
    """
    values: set[float] = set()
    for region in provider.regions:
        for declared in (region.inner_radius, region.outer_radius):
            if declared is None:
                continue
            radius = float(as_profile(declared)(np.array([float(z)], dtype=float))[0])
            if math.isfinite(radius) and radius > 0.0:
                values.add(radius)
    return tuple(sorted(values))


def modal_connection(  # noqa: PLR0913 - a provider, a point, and four declarations
    provider: ModalProvider,
    omega: float,
    z: float,
    *,
    nodes: ModalNodeRule | None = None,
    step: float | None = None,
    degenerate_blocks: Sequence[Sequence[int]] = (),
    basis: ChannelBasis | None = None,
) -> ModalConnection:
    r"""Measure the moving-basis connection of a provider at one point.

    The derivation is this module's own docstring: ``Gamma = W^-1 S`` with
    ``S_mn = <psi_m | d_z psi_n>`` on the package's flux metric, with the flux
    self-terms kept. The three samples of the difference stencil are put in **one
    gauge** by :func:`solprop.translate.modal_tables.align_modal_sample` against the
    centre, so the phase the derivative measures is the parallel-transported one and
    not the solver's arbitrary one, and a declared `basis` is applied after it in the
    same fixed order the tables compose.

    The stencil is centred where it fits and shifted where it does not, at the ends
    of the provider's z domain and against a declared kink; the derivative weights
    are the exact Lagrange ones of the three positions actually used, so the order is
    the same either way.

    Parameters
    ----------
    provider : solprop.modes.vector.ModalProvider
        The guide, solved at three positions.
    omega : float
        Positive angular frequency in rad/s inside the provider's band.
    z : float
        Position in m inside the provider's z domain.
    nodes : solprop.modes.vector.ModalNodeRule or None, optional
        The rule to refine onto the moving interfaces (:func:`connection_nodes`);
        ``None`` asks the provider for one at `z`.
    step : float or None, optional
        The difference step in m; ``None`` takes :data:`STEP_FRACTION` of the z
        scale, which is the provider's domain narrowed to the interval between the
        kinks `z` lies between.
    degenerate_blocks : sequence of sequence of int, optional
        Channel blocks the tracking may rotate inside, as
        :func:`solprop.translate.modal_tables.sample_modal_table` takes them.
    basis : solprop.modes.guide.ChannelBasis or None, optional
        A declared unitary change of the channel coordinates, applied after the
        tracking gauge. It is a **constant** declaration: a transform that varies
        with z adds :func:`basis_connection` on top, which the caller composes.

    Returns
    -------
    ModalConnection
        The connection, the metric, its measured rate and the certificate.

    Raises
    ------
    solprop.exceptions.ModeError
        If `step` is not positive and finite, if the stencil spans more than
        :data:`MAX_STENCIL_SPAN` of the z scale or does not fit inside the z domain
        at all, or on anything the provider, the tracking or the declared basis
        refuses.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.connection import modal_connection
    >>> from solprop.modes.graded_index import (
    ...     GradedIndexProvider, PowerLawPermittivity,
    ... )
    >>> from solprop.modes.material import IndexMaterial
    >>> from solprop.profiles import Linear
    >>> guide = GradedIndexProvider(
    ...     permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
    ...     exterior=IndexMaterial(1.45 * np.sqrt(1 - 2 * 0.005)),
    ...     radius=Linear(25e-6, 26e-6, 1e-3), family="LP", count=2,
    ...     elements=16, extent=2.0,
    ... )
    >>> link = modal_connection(guide, 2 * np.pi * 3e8 / 1.03e-6, 5e-4)
    >>> link.n_modes, bool(link.antisymmetry < 1e-6 * np.max(abs(link.matrix)))
    (2, True)
    """
    at, span = _stencil(provider, float(z), step)
    rules = connection_nodes(provider, float(omega), at, nodes=nodes)
    return _measured_connection(
        provider,
        float(omega),
        float(z),
        (at, span, rules),
        degenerate_blocks=degenerate_blocks,
        basis=basis,
    )


def band_connection(  # noqa: PLR0913 - a provider, a band, a point, and five declarations
    provider: ModalProvider,
    omega: Sequence[float] | Float[np.ndarray, " n_omega"],
    z: float,
    *,
    nodes: ModalNodeRule | None = None,
    step: float | None = None,
    degenerate_blocks: Sequence[Sequence[int]] = (),
    basis: ChannelBasis | None = None,
    n_r: int | None = None,
    n_theta: int | None = None,
) -> tuple[ModalConnection, ...]:
    r"""Measure the moving-basis connection at one position over a band of frequencies.

    :func:`modal_connection` at every frequency of `omega`, with one difference that
    is a cost and not a physics choice: the stencil and its union node rule
    (:func:`connection_nodes`) are built **once**, at the **lowest** frequency of the
    band, where a bound field reaches furthest and the rule's certified tail is a
    bound over the whole band -- the same reason
    :meth:`solprop.modes.guide.ModalGuide.from_provider` asks for its rule there. A
    rule with a certified tail costs a provider solve of its own, so building three of
    them per frequency would cost more than the connection itself. Each frequency is
    then measured and certified exactly as :func:`modal_connection` measures it: the
    three solves tracked into the centre's gauge, the declared `basis` applied after
    the tracking, ``Gamma = W^-1 S`` on the flux pairing and the antisymmetry
    certificate against ``W'`` differenced independently.

    This is what a moving guide's path needs: the connection at
    every sampled position and every frequency knot of the guide, interpolated in
    frequency onto the run's bins as the fields are.

    Parameters
    ----------
    provider : solprop.modes.vector.ModalProvider
        The guide, solved at three positions per frequency.
    omega : sequence of float or numpy.ndarray
        Positive angular frequencies in rad/s inside the provider's band, shape
        ``(n_omega,)``, at least one.
    z : float
        Position in m inside the provider's z domain.
    nodes : solprop.modes.vector.ModalNodeRule or None, optional
        A further rule joining the union, as :func:`connection_nodes` takes it.
    step : float or None, optional
        The difference step in m, as :func:`modal_connection` takes it.
    degenerate_blocks : sequence of sequence of int, optional
        Channel blocks the tracking may rotate inside.
    basis : solprop.modes.guide.ChannelBasis or None, optional
        A declared constant unitary change of the channel coordinates.
    n_r, n_theta : int or None, optional
        The resolution each stencil position's rule is asked for
        (:func:`connection_nodes`).

    Returns
    -------
    tuple of ModalConnection
        One per frequency, in the order given.

    Raises
    ------
    solprop.exceptions.ModeError
        If `omega` is empty, or on anything :func:`modal_connection` refuses.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.connection import band_connection
    >>> from solprop.modes.graded_index import (
    ...     GradedIndexProvider, PowerLawPermittivity,
    ... )
    >>> from solprop.modes.material import IndexMaterial
    >>> from solprop.profiles import Linear
    >>> guide = GradedIndexProvider(
    ...     permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
    ...     exterior=IndexMaterial(1.45 * np.sqrt(1 - 2 * 0.005)),
    ...     radius=Linear(25e-6, 26e-6, 1e-3), family="LP", count=2,
    ...     elements=16, extent=2.0,
    ... )
    >>> links = band_connection(guide, [1.80e15, 1.84e15], 5e-4)
    >>> len(links), links[0].n_modes
    (2, 2)
    """
    frequencies = np.asarray(omega, dtype=float).reshape(-1)
    if not frequencies.size:
        raise ModeError("a band connection needs at least one frequency")
    at, span = _stencil(provider, float(z), step)
    rules = connection_nodes(
        provider,
        float(np.min(frequencies)),
        at,
        nodes=nodes,
        n_r=n_r,
        n_theta=n_theta,
    )
    return tuple(
        _measured_connection(
            provider,
            float(one),
            float(z),
            (at, span, rules),
            degenerate_blocks=degenerate_blocks,
            basis=basis,
        )
        for one in frequencies
    )


def _measured_connection(
    provider: ModalProvider,
    omega: float,
    z: float,
    stencil: tuple[tuple[float, float, float], float, tuple[ModalNodeRule, ...]],
    *,
    degenerate_blocks: Sequence[Sequence[int]],
    basis: ChannelBasis | None,
) -> ModalConnection:
    """Measure and certify the connection at one frequency on a built stencil.

    Parameters
    ----------
    provider : solprop.modes.vector.ModalProvider
        The guide.
    omega : float
        Positive angular frequency in rad/s.
    z : float
        Position in m.
    stencil : tuple
        The three positions in m, the step in m and one rule per position, from
        :func:`_stencil` and :func:`connection_nodes`.
    degenerate_blocks : sequence of sequence of int
        Channel blocks the tracking may rotate inside.
    basis : solprop.modes.guide.ChannelBasis or None
        A declared constant unitary change of the channel coordinates.

    Returns
    -------
    ModalConnection
        The connection, the metric, its measured rate and the certificate.
    """
    # Imported here: solprop.translate imports solprop.modes, so a module-level
    # import of the tracking would close a cycle through the translate package body.
    from solprop.translate.modal_tables import (  # noqa: PLC0415
        align_modal_sample,
        apply_channel_basis,
    )

    at, span, rules = stencil
    middle = int(np.argmin(np.abs(np.asarray(at) - float(z))))
    solved = [
        provider.solve(float(omega), one, nodes=rule)
        for one, rule in zip(at, rules, strict=True)
    ]
    centre = solved[middle]
    samples = [
        one
        if index == middle
        else align_modal_sample(
            centre, one, rules[index], degenerate_blocks=degenerate_blocks
        )
        for index, one in enumerate(solved)
    ]
    if basis is not None:
        samples = [apply_channel_basis(one, basis) for one in samples]
    weights = rules[0].weights
    slope = _lagrange_slope(at, float(z))
    derivative = [
        np.tensordot(
            np.asarray(slope),
            np.stack([getattr(one, name) for one in samples]).astype(np.complex128),
            axes=1,
        )
        for name in ("electric", "magnetic")
    ]
    reference = samples[middle]
    pairing = _flux_pairing(derivative[0], derivative[1], reference, weights)
    metric = np.asarray(reference.flux_metric, dtype=np.complex128)
    # The metric's rate is differenced from the samples' own Gram matrices, not
    # read off `S + S^H`. Deriving it from the pairing would make the certificate
    # below an identity of the arithmetic -- `W Gamma = S` holds to the solve's
    # round-off whatever the fields are -- and it has to be a statement about the
    # sampling: that the derivative of the measured Gram really is the pairing
    # symmetrised, which is false the moment a panel edge leaves a moving
    # interface or the stencil is too coarse.
    rate = np.tensordot(
        np.asarray(slope),
        np.stack([one.flux_metric for one in samples]).astype(np.complex128),
        axes=1,
    )
    matrix = np.linalg.solve(metric, pairing)
    residual = metric @ matrix + matrix.conj().T @ metric - rate
    antisymmetry = float(np.linalg.norm(residual, ord=2))
    scale = float(np.max(abs(matrix)))
    low, high = (float(one) for one in provider.z_domain)
    # A connection that converts less than `NEGLIGIBLE_CONVERSION` over the whole z
    # domain is rounding -- a degenerate pair's is zero by symmetry -- and its
    # relative certificate says nothing; only a connection that matters is warned on.
    matters = scale * (high - low) > NEGLIGIBLE_CONVERSION
    if matters and antisymmetry > CERTIFICATE_WARNING_RTOL * scale:
        warnings.warn(
            f"the moving-basis connection at z = {float(z):.6g} m, omega = "
            f"{float(omega):.6g} rad/s meets its antisymmetry certificate only to "
            f"{antisymmetry / scale:.3e} of its own scale, above the "
            f"{CERTIFICATE_WARNING_RTOL:.1e} the qualified providers meet at the "
            f"default step: the {span:.6g} m stencil is too coarse for this provider, "
            "or its solves are noisier than the derivation assumes",
            GeneralModalBasisWarning,
            stacklevel=3,
        )
    return ModalConnection(
        matrix,
        metric,
        rate,
        antisymmetry,
        span,
        float(z),
        float(omega),
    )


def _flux_pairing(
    electric: Complex[np.ndarray, "n_nodes_p 3 n_modes_p"],
    magnetic: Complex[np.ndarray, "n_nodes_m 3 n_modes_m"],
    sample: ModalSample,
    weights: Float[np.ndarray, " n_nodes_w"],
) -> Complex[np.ndarray, "n_modes_s_out n_modes_s_in"]:
    r"""Return ``S_mn = <psi_m | d_z psi_n>`` on the Hermitian Poynting pairing.

    The same pairing :func:`solprop.modes.vector.flux_product` integrates, with the
    right-hand slot carrying the derivative instead of the field:
    ``S = [X(dE, H) + X(E, dH)^H]/4`` with
    ``X(A, B)_mn = int (A_n x conj(B_m)).zhat dA``. Taking the Gram with the same
    two einsums is what makes ``S + S^H`` the derivative of the Gram itself.

    Parameters
    ----------
    electric, magnetic : numpy.ndarray
        The derivatives ``dE/dz`` and ``dH/dz`` in V/(m sqrt(W) m) and
        A/(m sqrt(W) m), shape ``(n_nodes, 3, n_modes)``.
    sample : solprop.modes.vector.ModalSample
        The centre sample, whose fields fill the left-hand slot.
    weights : numpy.ndarray
        The shared area weights in m², shape ``(n_nodes,)``.

    Returns
    -------
    numpy.ndarray
        ``S`` in 1/m, shape ``(n_modes, n_modes)``.
    """
    field, magnetics = sample.electric, sample.magnetic
    left = np.einsum(
        "qm,qn,q->mn", magnetics[:, 1].conj(), electric[:, 0], weights
    ) - np.einsum("qm,qn,q->mn", magnetics[:, 0].conj(), electric[:, 1], weights)
    right = np.einsum(
        "qm,qn,q->mn", magnetic[:, 1].conj(), field[:, 0], weights
    ) - np.einsum("qm,qn,q->mn", magnetic[:, 0].conj(), field[:, 1], weights)
    return (left + right.conj().T) / 4


def _stencil(
    provider: ModalProvider, z: float, step: float | None
) -> tuple[tuple[float, float, float], float]:
    """Return three positions the provider can be solved at, and the step used.

    Parameters
    ----------
    provider : solprop.modes.vector.ModalProvider
        The guide, whose z domain and breakpoints bound the stencil.
    z : float
        The position in m.
    step : float or None
        The declared step in m, or ``None`` for :data:`STEP_FRACTION` of the scale.

    Returns
    -------
    tuple
        The three increasing positions in m and the step in m.

    Raises
    ------
    solprop.exceptions.ModeError
        If `step` is out of range or no stencil of it fits inside the z domain.
    """
    low, high = (float(one) for one in provider.z_domain)
    # The scale is the interval `z` lies in, not the whole domain: a derivative
    # across a declared kink is not the derivative of either side, and the run's
    # sections land on every kink, so a stencil never needs to cross one.
    edges = sorted({low, high, *(float(one) for one in provider.z_breakpoints)})
    index = min(
        max(int(np.searchsorted(np.asarray(edges), z, side="right")) - 1, 0),
        len(edges) - 2,
    )
    left, right = edges[index], edges[index + 1]
    scale = right - left
    width = STEP_FRACTION * scale if step is None else float(step)
    if not math.isfinite(width) or width <= 0.0:
        raise ModeError(
            f"a moving-basis connection takes a positive finite step, got {step!r}"
        )
    if width > MAX_STENCIL_SPAN * scale:
        raise ModeError(
            f"a moving-basis connection's step of {width:.6g} m is more than "
            f"{MAX_STENCIL_SPAN} of the {scale:.6g} m between the declared "
            f"positions [{left:.6g}, {right:.6g}] m that {z:.6g} m lies in: a "
            "derivative taken over a quarter of the scale it varies on is a secant"
        )
    # The stencil fits by construction, and there is no second guard for it:
    # `width <= MAX_STENCIL_SPAN * scale` with that constant at a quarter, so
    # `right - 2 width >= left + scale/2 > left` and the clamp below never lands
    # outside the interval. A larger `MAX_STENCIL_SPAN` than a half would break
    # that, which is why the refusal above is the one that has to be met.
    start = min(max(z - width, left), right - 2.0 * width)
    return (start, start + width, start + 2.0 * width), width


def _lagrange_slope(positions: Sequence[float], z: float) -> tuple[float, float, float]:
    """Return the three-point Lagrange derivative weights at one position.

    Exact for a quadratic through `positions`, so a shifted stencil carries the same
    order as a centred one and a centred one reduces to ``(-1, 0, 1)/(2h)``.

    Parameters
    ----------
    positions : sequence of float
        Three distinct increasing positions in m.
    z : float
        Where the derivative is taken, m.

    Returns
    -------
    tuple of float
        The weights in 1/m, summing to zero.
    """
    first, second, third = (float(one) for one in positions)
    return (
        (2.0 * z - second - third) / ((first - second) * (first - third)),
        (2.0 * z - first - third) / ((second - first) * (second - third)),
        (2.0 * z - first - second) / ((third - first) * (third - second)),
    )
