# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Ordered, validated sets of modes, and the overlaps between them.

:class:`ModeCollection` is the set of modes a multimode run propagates through one
transverse rule and one pair of transform matrices: ordered, because the kernel's mode
axis is, and validated once to share a cross-section. It reports whether the azimuthal
coordinate has to be resolved and to what order, and expands a spatial set into its
polarisation partners (:func:`polarisation_pairs`). :func:`transverse_measure` names
the measure a set is integrated over.

:func:`overlap` integrates adaptively (:func:`scipy.integrate.dblquad`, or
:func:`scipy.integrate.quad` over x alone, per metre of y, for a slab set), and
:func:`orthonormality` integrates the whole matrix in one pass, adaptively in the first
coordinate with :func:`scipy.integrate.quad_vec` and on a rule refined to convergence
in the second -- both deliberately *not* on the fixed rule whose orthonormality they
check. They use the package normalisation (``docs/dev/conventions.md`` section 4),
:math:`\langle m_1|m_2\rangle = \tfrac12\sqrt{\varepsilon_0/\mu_0}\int e_1^*\cdot
e_2\,\mathrm{d}A / \sqrt{N_1N_2}`, so a mode overlaps itself by one. The manual page is
``docs/guide/waveguides/mode-sets.md``.
"""

import logging
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from jaxtyping import Float
from scipy.integrate import dblquad, quad, quad_vec

from solprop._display import Displayed
from solprop.constants import EPS0, C
from solprop.exceptions import ModeError
from solprop.modes.antiresonant import TubeLattice, VincettiMode
from solprop.modes.capillary import (
    MarcatiliMode,
    MarcatiliModel,
    ModeKind,
)
from solprop.modes.fill import LossFunction
from solprop.modes.protocol import (
    ChannelLabel,
    CrossSection,
    Mode,
    TransverseMeasure,
)
from solprop.modes.slab import SlabMode, SlabPolarisation
from solprop.modes.stepindex import StepIndexMode
from solprop.physdata.gases import Gas
from solprop.physdata.glasses import GlassIndex
from solprop.physdata.mixtures import Mixture
from solprop.physdata.sellmeier import GasSusceptibility
from solprop.profiles import ZProfile

__all__ = [
    "ModeCollection",
    "orthonormality",
    "overlap",
    "polarisation_pairs",
    "transverse_measure",
]

logger = logging.getLogger(__name__)

_GRAM_TOLERANCE = 1e-12
"""Absolute tolerance on each normalised overlap of :func:`orthonormality`: of the
adaptive integral in the first coordinate, and of the change on doubling the rule in
the second. Far below the driver's acceptance of a set (1e-6)."""

_FIRST_NODES = 16
"""Fewest nodes of the second coordinate's rule before the first doubling; a polar set
starts above its products' highest harmonic (:func:`_first_nodes`)."""

_MAX_NODES = 1024
"""Most nodes of the second coordinate's rule, or four times the first rule if that is
more: a set that needs more is refused."""


def _normalisation(mode: Mode, z: float) -> float:
    """Return a mode's closed-form normalisation.

    - ``z``: position along the waveguide, m.
    - Returns: ``N`` in W.
    """
    closed_form: Callable[[float], float] | None = getattr(mode, "normalisation", None)
    if not callable(closed_form):
        raise ModeError(
            f"{type(mode).__name__} provides no normalisation(z), so its overlaps "
            "cannot be normalised"
        )
    return float(closed_form(z))


def overlap(first: Mode, second: Mode, z: float = 0.0) -> float:
    """Return the normalised overlap integral of two modes.

    Parameters
    ----------
    first, second : Mode
        The modes, which must share a cross-section.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    float
        ``1`` for a mode with itself and ``0`` for an orthogonal pair. Integrated
        adaptively, not on the transform's fixed rule, so that the rule can be
        checked against it.

    Raises
    ------
    ModeError
        If the modes do not share a cross-section, or either has no closed-form
        normalisation.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.collection import overlap
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{overlap(mode, mode):.9f}"
    '1.000000000'
    """
    limits = first.dimlimits(z)
    if second.dimlimits(z) != limits:
        raise ModeError(
            f"an overlap needs one cross-section; {limits} and "
            f"{second.dimlimits(z)} differ"
        )
    kind, lower, upper = limits
    # The scale goes inside the integrands: the integrals are then of order one, so
    # the adaptive rules' default absolute tolerance is small against them. Outside,
    # an unnormalised integral (2e-13 for a 75 um capillary's HE1,24) lies below
    # that tolerance and the rule stops unconverged, without a warning.
    scale = (
        0.5 * EPS0 * C / math.sqrt(_normalisation(first, z) * _normalisation(second, z))
    )

    def _integrand(inner: float, outer: float) -> float:
        """Return the scaled product of the two profiles times the area element.

        - ``inner``: the first cross-section coordinate, ``r`` or ``x`` in m. dblquad
          passes the *inner* variable first, which is why the order looks reversed.
        - ``outer``: the second coordinate, ``theta`` in rad or ``y`` in m.
        """
        point = np.array([[inner, outer]])
        product = scale * float(np.sum(first.field(point, z) * second.field(point, z)))
        return product * inner if kind == "polar" else product

    if kind == "slab":
        # Uniform in y: the overlap is per metre of y, over x alone, as N is.
        return _overlap_along_x(first, second, (lower[0], upper[0]), z, scale)
    # The outer variable is the second coordinate and the inner the first, with the
    # polar Jacobian r folded into the integrand above.
    value, _ = dblquad(_integrand, lower[1], upper[1], lower[0], upper[0])
    return value


def _overlap_along_x(
    first: Mode, second: Mode, limits: tuple[float, float], z: float, scale: float
) -> float:
    """Return the scaled integral over x of two slab profiles' product, per metre of y.

    - ``limits``: the gap's edges in m.
    - ``z``: position along the waveguide, m.
    - ``scale``: the overlap's normalisation, multiplied into the integrand.
    - Returns: ``scale * int e_1 . e_2 dx``, integrated adaptively.
    """

    def along_x(x: float) -> float:
        """Return the product of the two slab profiles at one x.

        - ``x``: transverse coordinate in m.
        """
        point = np.array([[x, 0.0]])
        return scale * float(np.sum(first.field(point, z) * second.field(point, z)))

    value, _ = quad(along_x, limits[0], limits[1], limit=200)
    return value


def orthonormality(
    modes: Sequence[Mode], z: float = 0.0
) -> Float[np.ndarray, "n_modes n_modes"]:
    """Return the matrix of normalised overlaps of a mode set.

    Parameters
    ----------
    modes : sequence of Mode
        The modes, non-empty.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_modes, n_modes)``, symmetric, the identity for an orthonormal set;
        each entry the :func:`overlap` of its pair to about ``1e-12``.

    Raises
    ------
    ModeError
        If `modes` is empty, the modes do not share a cross-section or one has no
        closed-form normalisation, or the integration does not converge in either
        coordinate.

    Notes
    -----
    Every entry is integrated in one pass: adaptively in the first coordinate (``r`` or
    ``x``) with :func:`scipy.integrate.quad_vec`, whose error control covers the whole
    matrix, and in the second on an offset periodic trapezoid (polar) or a
    Gauss--Legendre rule (Cartesian), doubled until no entry changes by more than
    ``1e-12``: from 16 nodes, or for a polar set from above twice its highest azimuthal
    order, so that no product's harmonic aliases alike in two rules. Each mode's field
    is evaluated once per node, not once per pair, so a set of 48 modes takes seconds
    rather than the minutes of :func:`overlap` pair by pair. A slab set integrates over
    x alone, per metre of y.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.collection import orthonormality
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, m=m) for m in (1, 2)]
    >>> bool(np.max(np.abs(orthonormality(modes) - np.eye(2))) < 1e-9)
    True
    """
    if not modes:
        raise ModeError("orthonormality needs at least one mode")
    limits = modes[0].dimlimits(z)
    for mode in modes[1:]:
        if mode.dimlimits(z) != limits:
            raise ModeError(
                f"an overlap needs one cross-section; {limits} and "
                f"{mode.dimlimits(z)} differ"
            )
    kind, lower, upper = limits
    # Each mode's share of the overlap's normalisation, inside the integrand: the
    # integrals are then of order one, as the absolute tolerance assumes.
    scale = np.sqrt(0.5 * EPS0 * C / np.array([_normalisation(m, z) for m in modes]))
    first = (lower[0], upper[0])
    if kind == "slab":
        # Uniform in y: one node at y = 0 of unit weight, per metre of y as N is.
        return _gram(modes, scale, first, (np.zeros(1), np.ones(1)), z=z, polar=False)
    nodes = _first_nodes(modes) if kind == "polar" else _FIRST_NODES
    # Every set is allowed at least two doublings, so that convergence is shown on
    # three rules even where a polar set's first rule reaches the cap.
    most = max(_MAX_NODES, 4 * nodes)
    previous = None
    while nodes <= most:
        rule = _second_rule(kind, (lower[1], upper[1]), nodes)
        matrix = _gram(modes, scale, first, rule, z=z, polar=kind == "polar")
        if (
            previous is not None
            and np.max(np.abs(matrix - previous)) <= _GRAM_TOLERANCE
        ):
            return matrix
        previous = matrix
        nodes *= 2
    raise ModeError(
        f"the overlaps of this {kind} mode set did not converge in the second "
        f"coordinate with {most} nodes"
    )


def _first_nodes(modes: Sequence[Mode]) -> int:
    """Return the polar rule's first node count, above every product's harmonics.

    A pair's product reaches azimuthal degree ``h_a + h_b``, and the offset periodic
    trapezoid of N nodes integrates degree k exactly unless N divides k. Started at
    or below the degree, two consecutive rules can alias a harmonic alike -- 16 and 32
    nodes both read ``cos(64 theta)`` as +1 at every node -- and agree on a wrong
    value. The first rule is therefore the smallest power of two above twice the
    set's highest order, and at least :data:`_FIRST_NODES`.
    """
    degree = 2 * max(mode.azimuthal_order for mode in modes)
    nodes = _FIRST_NODES
    while nodes <= degree:
        nodes *= 2
    return nodes


def _second_rule(
    kind: CrossSection, limits: tuple[float, float], nodes: int
) -> tuple[Float[np.ndarray, " n_nodes"], Float[np.ndarray, " n_nodes"]]:
    """Return the nodes and weights of the rule in the second coordinate.

    Polar: the offset periodic trapezoid over the full circle, exact for trigonometric
    polynomials below degree `nodes`; its offset keeps nodes off the axes, where
    products such as ``E_x E_y`` vanish (:mod:`solprop.modes.quadrature`). Cartesian:
    Gauss--Legendre over the y limits.
    """
    low, high = limits
    if kind == "polar":
        step = (high - low) / nodes
        return low + step * (np.arange(nodes) + 0.5), np.full(nodes, step)
    x, weights = np.polynomial.legendre.leggauss(nodes)
    half = 0.5 * (high - low)
    return low + half * (x + 1.0), half * weights


def _gram(
    modes: Sequence[Mode],
    scale: Float[np.ndarray, " n_modes"],
    limits: tuple[float, float],
    rule: tuple[Float[np.ndarray, " n_nodes"], Float[np.ndarray, " n_nodes"]],
    *,
    z: float,
    polar: bool,
) -> Float[np.ndarray, "n_modes n_modes"]:
    """Return the normalised overlap matrix on one rule in the second coordinate.

    - ``scale``: each mode's ``sqrt(eps0 c / (2 N))``: two scaled fields' product
      integrates over the area to the dimensionless overlap.
    - ``limits``: the first coordinate's range, m.
    - ``polar``: whether the first coordinate is a radius; its Jacobian r is folded in.
    """
    second, weights = rule

    def _line(first: float) -> Float[np.ndarray, "n_modes n_modes"]:
        """Return the matrix integrand at one value of the first coordinate."""
        points = np.column_stack((np.full(second.size, first), second))
        fields = np.stack([mode.field(points, z) for mode in modes])
        fields = fields * scale[:, None, None]
        matrix = np.einsum("mpc,p,npc->mn", fields, weights, fields)
        return matrix * first if polar else matrix

    matrix, _, info = quad_vec(
        _line,
        limits[0],
        limits[1],
        epsabs=_GRAM_TOLERANCE,
        epsrel=0.0,
        norm="max",
        full_output=True,
    )
    # quad_vec returns its best estimate without a warning when it stops short of the
    # tolerance; an unconverged matrix must not pass as an orthonormality check.
    if info.status != 0:
        raise ModeError(
            "the overlaps of this mode set did not converge in the first coordinate: "
            f"{info.message}"
        )
    matrix = np.asarray(matrix)
    # Symmetric by construction up to the summation order; made exactly so.
    return 0.5 * (matrix + matrix.T)


def transverse_measure(modes: Sequence[Mode]) -> TransverseMeasure:
    """Return what a mode set's transverse integrals measure, refusing a mixture.

    Parameters
    ----------
    modes : sequence of Mode
        The modes, non-empty. Each reports the optional ``transverse_measure``
        trait by name; a mode without it takes the measure of the mode it was
        built from -- a :class:`solprop.nonlinear.gnlse.SpecifiedMode` made
        from a slab mode keeps its source's per-unit-width normalisation and
        area -- and measures an area when there is none.

    Returns
    -------
    str
        ``"area"`` or ``"per_unit_width"``
        (:data:`solprop.modes.protocol.TransverseMeasure`).

    Raises
    ------
    ModeError
        If the set is empty, or mixes a per-unit-width mode with an area mode, whose
        powers would be in W/m and W in one state.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode, SlabMode
    >>> from solprop.modes.collection import transverse_measure
    >>> transverse_measure([MarcatiliMode(125e-6)])
    'area'
    >>> transverse_measure([SlabMode(50e-6)])
    'per_unit_width'
    """
    if not modes:
        raise ModeError("a transverse measure needs at least one mode")
    measures: list[TransverseMeasure] = [_measure_of(mode) for mode in modes]
    for index, measure in enumerate(measures[1:], start=1):
        if measure != measures[0]:
            raise ModeError(
                f"mode {index} ({type(modes[index]).__name__}) measures "
                f"{measure!r} but mode 0 ({type(modes[0]).__name__}) measures "
                f"{measures[0]!r}: a mode uniform in y carries a power per metre "
                "of y and an area mode a power, and one state cannot hold both"
            )
    return measures[0]


def _measure_of(mode: Mode) -> TransverseMeasure:
    """Return one mode's transverse measure, following the modes it was built from.

    The chain is the one :func:`solprop.translate.modal._radius_kinks` walks: a
    specified mode keeps its source in its dispersion's ``mode``, and its area --
    ``aeff()`` of that source unless one was declared -- is in the source's units.

    - Returns: the first declared ``transverse_measure`` along the chain, else
      ``"area"``.
    """
    one: object = mode
    while one is not None:
        declared = getattr(one, "transverse_measure", None)
        if declared is not None:
            return cast("TransverseMeasure", declared)
        one = getattr(getattr(one, "dispersion", None), "mode", None)
    return "area"


def polarisation_pairs(modes: Sequence[Mode]) -> tuple[Mode, ...]:
    """Expand a mode set so that each degenerate mode carries both polarisations.

    A vector run propagates the two orthogonal polarisation states of a hybrid mode as
    two entries on the mode axis. The order is fixed and is the obvious one -- each
    mode immediately followed by its partner -- because the mode axis of the kernel
    state is positional and a reordering would be silent.

    A mode with no degenerate partner, such as ``TE_0m`` or ``TM_0m``, is carried once.

    Parameters
    ----------
    modes : sequence of Mode
        The spatial modes.

    Returns
    -------
    tuple of Mode
        The expanded set.

    Raises
    ------
    ModeError
        If a mode cannot say whether it has a partner, which every ported mode can.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.collection import polarisation_pairs
    >>> pairs = polarisation_pairs([MarcatiliMode(125e-6, "He", 1.0)])
    >>> len(pairs), f"{pairs[1].phi:.4f}"
    (2, '1.5708')
    """
    expanded: list[Mode] = []
    missing = object()
    for mode in modes:
        partner: Mode | None = getattr(mode, "orthogonal_polarisation", missing)  # type: ignore[assignment] - checked immediately below
        if partner is missing:
            raise ModeError(
                f"{type(mode).__name__} does not report an orthogonal polarisation, "
                "so a vector mode set cannot be built from it"
            )
        expanded.append(mode)
        if partner is not None:
            expanded.append(partner)
    return tuple(expanded)


@dataclass(frozen=True, eq=False)
class ModeCollection(Displayed, Sequence[Mode]):
    """An ordered set of modes sharing one cross-section.

    Validated once, at construction: the mode axis of the kernel state is positional,
    and one transform matrix serves the whole set, so a set whose members disagree
    about the geometry is not a set.

    Attributes
    ----------
    modes : sequence of Mode
        The modes, in the order the mode axis will carry them. Any sequence is
        accepted and is stored as a tuple, so a collection cannot be changed after it
        has been validated.
    """

    modes: Sequence[Mode]

    def __post_init__(self) -> None:
        """Check that the set is non-empty and shares a cross-section.

        Raises
        ------
        ModeError
            If the set is empty, mixes a per-unit-width (slab) mode with an area
            mode, or the cross-sections disagree at ``z = 0``.
        """
        object.__setattr__(self, "modes", tuple(self.modes))
        if not self.modes:
            raise ModeError("a mode collection needs at least one mode")
        transverse_measure(self.modes)
        self._validate_channels()
        limits = self.modes[0].dimlimits(0.0)
        for index, mode in enumerate(self.modes[1:], start=1):
            if mode.dimlimits(0.0) != limits and not self._same_stepindex_guide(mode):
                raise ModeError(
                    f"every mode in a collection must share a cross-section; mode "
                    f"{index} has {mode.dimlimits(0.0)} against {limits}"
                )
        logger.debug(
            "mode collection of %d modes, %s, azimuthal order %d, full rule %s",
            len(self.modes),
            limits[0],
            self.azimuthal_order,
            self.needs_full_rule,
        )

    def _same_stepindex_guide(self, mode: Mode) -> bool:
        """Check whether a channel shares the first physical guide."""
        first = self.modes[0]
        return (
            isinstance(first, StepIndexMode)
            and isinstance(mode, StepIndexMode)
            and (first.radius is mode.radius or first.radius == mode.radius)
            and first._core == mode._core
            and first._cladding == mode._cladding
        )

    def _validate_channels(self) -> None:
        """Reject two members that name the same physical channel.

        Uniqueness is on the members' own :class:`ChannelLabel`, for every family:
        the label is what the mode axis carries and what the run record persists,
        so two members sharing one is two rows for one channel.
        """
        seen: set[ChannelLabel] = set()
        for mode in self.modes:
            label = mode.channel_label
            if label in seen:
                raise ModeError(
                    f"duplicate physical channel {label.display_label}: two members "
                    "name the same channel, and an axially symmetric TE/TM channel "
                    "has no parity partner to tell two of them apart"
                )
            seen.add(label)

    @property
    def channel_labels(self) -> tuple[ChannelLabel, ...]:  # numpydoc ignore=RT01
        """Each member's physical label, in the order of the mode axis.

        The members' own :attr:`solprop.modes.protocol.Mode.channel_label`, in
        order. A label a family does not have is ``None``, never a string
        standing for one: an axially symmetric TE0m or TM0m channel has no
        parity partner, so its parity is ``None``, exactly as its handedness is.
        The run record persists these labels.

        Examples
        --------
        >>> from solprop.modes import MarcatiliMode
        >>> from solprop.modes.collection import ModeCollection
        >>> modes = ModeCollection([MarcatiliMode(125e-6, "He", 1.0)])
        >>> [label.display_label for label in modes.channel_labels]
        ['HE1,1:even']
        """
        return tuple(mode.channel_label for mode in self.modes)

    @classmethod
    def with_polarisations(cls, modes: Sequence[Mode]) -> "ModeCollection":
        """Build a collection carrying both polarisations of every degenerate mode.

        Parameters
        ----------
        modes : sequence of Mode
            The spatial modes.

        Returns
        -------
        ModeCollection
            The expanded set, in the fixed order of :func:`polarisation_pairs`.

        Examples
        --------
        >>> from solprop.modes import MarcatiliMode
        >>> from solprop.modes.collection import ModeCollection
        >>> modes = [MarcatiliMode(125e-6, "He", 1.0, m=m) for m in (1, 2)]
        >>> len(ModeCollection.with_polarisations(modes))
        4
        """
        return cls(polarisation_pairs(modes))

    @classmethod
    def he1m(
        cls,
        radius: float | ZProfile,
        gas: Gas | str | None = None,
        pressure: float | ZProfile = 0.0,
        *,
        count: int = 1,
        model: MarcatiliModel = "full",
        alpha: float | LossFunction | None = None,
        loss_scale: float = 1.0,
        cladding: GlassIndex | complex | None = None,
        temperature: float | ZProfile | None = None,
        susceptibility: GasSusceptibility | None = None,
        mixture: Mixture | None = None,
    ) -> "ModeCollection":
        """Build the first `count` HE_1m modes of one gas-filled capillary.

        The linearly polarised hybrid modes of a hollow capillary, in the order of
        their radial index. They share a cross-section by construction, their
        azimuthal order is zero, so no azimuthal quadrature is needed, and they are
        the set a capillary run uses unless told otherwise.

        Every keyword after `count` is
        :class:`~solprop.modes.capillary.MarcatiliMode`'s own and is given to each
        mode unchanged, so a set can carry a test-only cladding or gas susceptibility
        the way a single mode can.

        Parameters
        ----------
        radius : float or ZProfile
            Core radius, m, constant or along z.
        gas : Gas, str or None, optional
            Filling gas; None for vacuum or an explicit mixture.
        pressure : float or ZProfile, optional
            Fill pressure, bar, constant or along z; default zero.
        count : int, optional
            How many modes, at least one; the modes are HE_11 to HE_1count.
        model : {'full', 'reduced'}, optional
            Which Marcatili model the modes use.
        alpha : float, callable or None, optional
            Absolute power attenuation in 1/m replacing the model's loss.
        loss_scale : float, optional
            Factor on the model's own loss; 0 is lossless.
        cladding : GlassIndex, complex or None, optional
            The cladding index; ``None`` is the package default silica.
        temperature : float, ZProfile or None, optional
            Gas temperature, K, or a path of it along z; ``None`` is room
            temperature, or the mixture's own.
        susceptibility : GasSusceptibility or None, optional
            A susceptibility to use in place of the built-in gas's.
        mixture : Mixture or None, optional
            Ordered species and density paths; owns the gas and susceptibility.

        Returns
        -------
        ModeCollection
            The validated set, in radial order.

        Raises
        ------
        ModeError
            If `count` is below one or the modes are invalid.

        Examples
        --------
        >>> from solprop.modes import ModeCollection
        >>> modes = ModeCollection.he1m(125e-6, "He", 0.8, count=4)
        >>> len(modes), modes[3].m, modes.azimuthal_order
        (4, 4, 0)
        """
        if count < 1:
            raise ModeError(
                f"a capillary mode set needs at least one mode, got {count}"
            )
        return cls(
            [
                MarcatiliMode(
                    radius,
                    gas,
                    pressure,
                    kind="HE",
                    n=1,
                    m=m,
                    model=model,
                    alpha=alpha,
                    loss_scale=loss_scale,
                    cladding=cladding,
                    temperature=temperature,
                    susceptibility=susceptibility,
                    mixture=mixture,
                )
                for m in range(1, count + 1)
            ]
        )

    @classmethod
    def slab(
        cls,
        half_gap: float | ZProfile,
        gas: Gas | str | None = None,
        pressure: float | ZProfile = 0.0,
        *,
        orders: Sequence[int] = (1,),
        polarisation: SlabPolarisation = "te",
        alpha: float | LossFunction | None = None,
        loss_scale: float = 1.0,
        cladding: GlassIndex | complex | None = None,
        temperature: float | ZProfile | None = None,
        susceptibility: GasSusceptibility | None = None,
        mixture: Mixture | None = None,
    ) -> "ModeCollection":
        """Build the given orders of one hollow planar guide, uniform in y.

        A centred, symmetric launch couples only to odd orders, so the set a
        compressor run wants is usually ``(1, 3, 5)``; an offset or tilted launch
        needs the even ones too. Every keyword after `orders` is
        :class:`~solprop.modes.slab.SlabMode`'s own and is given to each mode
        unchanged. For TE and TM rows in one set, build the two polarisations and
        pass them to the constructor, or use :meth:`with_polarisations`: each is a
        distinct mode with its own row.

        Parameters
        ----------
        half_gap : float or ZProfile
            Half the plate separation in m, constant or along z.
        gas : Gas, str or None, optional
            Filling gas; None for vacuum or an explicit mixture.
        pressure : float or ZProfile, optional
            Gas pressure in bar, constant or along z; default zero.
        orders : sequence of int, optional
            Mode orders ``p >= 1``, non-empty and without repeats, in state order;
            default the lowest order alone.
        polarisation : {'te', 'tm'}, optional
            ``"te"`` (E along y, the default) or ``"tm"`` (E along x), for every
            order.
        alpha : float, callable or None, optional
            Absolute power attenuation in 1/m replacing the model's loss.
        loss_scale : float, optional
            Factor on the model's own loss; 0 is lossless.
        cladding : GlassIndex, complex or None, optional
            Plate index; None selects the default silica.
        temperature : float, ZProfile or None, optional
            Gas temperature in K, or a path of it along z; ``None`` is room
            temperature, or the mixture's own.
        susceptibility : GasSusceptibility or None, optional
            Explicit gas susceptibility.
        mixture : Mixture or None, optional
            Ordered species and density paths; owns the gas and susceptibility.

        Returns
        -------
        ModeCollection
            The validated set, in the order given.

        Raises
        ------
        ModeError
            If `orders` is empty, repeats an order, or describes an invalid mode.

        Examples
        --------
        >>> from solprop.modes import ModeCollection
        >>> modes = ModeCollection.slab(63.5e-6, "Ar", 1.0, orders=(1, 3, 5))
        >>> [label.display_label for label in modes.channel_labels]
        ['TE1', 'TE3', 'TE5']
        >>> modes.transverse_measure, modes.needs_full_rule
        ('per_unit_width', False)
        """
        if not orders:
            raise ModeError("a slab mode set needs at least one order")
        return cls(
            tuple(
                SlabMode(
                    half_gap,
                    gas,
                    pressure,
                    order=order,
                    polarisation=polarisation,
                    alpha=alpha,
                    loss_scale=loss_scale,
                    cladding=cladding,
                    temperature=temperature,
                    susceptibility=susceptibility,
                    mixture=mixture,
                )
                for order in orders
            )
        )

    @classmethod
    def vincetti(
        cls,
        radius: float | ZProfile,
        gas: Gas | str | None = None,
        pressure: float | ZProfile = 0.0,
        *,
        count: int = 1,
        lattice: TubeLattice,
        alpha: float | LossFunction | None = None,
        loss_scale: float = 1.0,
        cladding: GlassIndex | complex | None = None,
        temperature: float | ZProfile | None = None,
        susceptibility: GasSusceptibility | None = None,
        mixture: Mixture | None = None,
    ) -> "ModeCollection":
        """Build the first `count` HE_1m modes of a tube-lattice fibre.

        The field and transverse quadrature remain Marcatili's. Corrections and
        loss are fundamental-mode fits; higher radial orders log that limitation.

        Parameters
        ----------
        radius : float or ZProfile
            Positive core radius in m, constant or along z.
        gas : Gas, str or None, optional
            Filling gas; None for vacuum or an explicit mixture.
        pressure : float or ZProfile, optional
            Gas pressure in bar, constant or along z; default zero.
        count : int, optional
            Number of modes, at least one, in radial order.
        lattice : TubeLattice
            Tube geometry shared by all modes.
        alpha : float, callable or None, optional
            Absolute power attenuation in 1/m replacing the model's loss.
        loss_scale : float, optional
            Factor on the model's own loss; 0 is lossless.
        cladding : GlassIndex, complex or None, optional
            Tube index; None selects the default silica.
        temperature : float, ZProfile or None, optional
            Gas temperature in K, or a path of it along z; ``None`` is room
            temperature, or the mixture's own.
        susceptibility : GasSusceptibility or None, optional
            Explicit gas susceptibility.
        mixture : Mixture or None, optional
            Ordered species and density paths; owns the gas and susceptibility.

        Returns
        -------
        ModeCollection
            Validated modes sharing the tube geometry and cross-section.

        Raises
        ------
        ModeError
            If the mode count or geometry is invalid.
        """
        if count < 1:
            raise ModeError(f"a Vincetti mode set needs at least one mode, got {count}")
        return cls(
            tuple(
                VincettiMode(
                    radius,
                    gas,
                    pressure,
                    m=index + 1,
                    lattice=lattice,
                    alpha=alpha,
                    loss_scale=loss_scale,
                    cladding=cladding,
                    temperature=temperature,
                    susceptibility=susceptibility,
                    mixture=mixture,
                )
                for index in range(count)
            )
        )

    @classmethod
    def marcatili(
        cls,
        radius: float | ZProfile,
        gas: Gas | str | None = None,
        pressure: float | ZProfile = 0.0,
        *,
        kinds: Sequence[ModeKind],
        n: Sequence[int],
        m: Sequence[int],
        model: MarcatiliModel = "full",
        alpha: float | LossFunction | None = None,
        loss_scale: float = 1.0,
        cladding: GlassIndex | complex | None = None,
        temperature: float | ZProfile | None = None,
        susceptibility: GasSusceptibility | None = None,
        mixture: Mixture | None = None,
    ) -> "ModeCollection":
        """Build an ordered mixture of hybrid and transverse capillary modes.

        Parameters
        ----------
        radius : float or ZProfile
            Core radius, m, constant or along z.
        gas : Gas, str or None, optional
            Filling gas; None for vacuum or an explicit mixture.
        pressure : float or ZProfile, optional
            Fill pressure, bar, constant or along z; default zero.
        kinds : sequence of {'HE', 'TE', 'TM'}
            Family of each mode, non-empty.
        n : sequence of int
            Azimuthal indices, one per family; nonzero for HE and zero for TE/TM.
        m : sequence of int
            Positive radial indices, one per family.
        model : {'full', 'reduced'}, optional
            Marcatili approximation used by every mode.
        alpha : float, callable or None, optional
            Absolute power attenuation in 1/m replacing the model's loss.
        loss_scale : float, optional
            Factor on the model's own loss; 0 is lossless.
        cladding : GlassIndex, complex or None, optional
            Cladding index; ``None`` selects the package default silica.
        temperature : float, ZProfile or None, optional
            Gas temperature, K, positive, or a path of it along z; ``None`` is
            room temperature, or the mixture's own.
        susceptibility : GasSusceptibility or None, optional
            Explicit susceptibility in place of the selected fit.
        mixture : Mixture or None, optional
            Ordered species and density paths; owns the gas and susceptibility.

        Returns
        -------
        ModeCollection
            Modes in the input order, all with ``phi = 0``. Use
            :meth:`with_polarisations` to add degenerate partners.

        Raises
        ------
        ModeError
            If the index sequences are empty, have different lengths, or describe
            an invalid mode.

        Examples
        --------
        >>> from solprop.modes import ModeCollection
        >>> modes = ModeCollection.marcatili(
        ...     125e-6, "He", 1.0, kinds=["HE", "HE", "TM"], n=[1, 2, 0], m=[1, 1, 1]
        ... )
        >>> len(modes), modes.azimuthal_order
        (3, 1)
        """
        if not kinds or len(kinds) != len(n) or len(kinds) != len(m):
            raise ModeError(
                "kinds, n and m must be non-empty sequences of equal length"
            )
        return cls(
            tuple(
                MarcatiliMode(
                    radius,
                    gas,
                    pressure,
                    kind=kind,
                    n=azimuthal,
                    m=radial,
                    model=model,
                    alpha=alpha,
                    loss_scale=loss_scale,
                    cladding=cladding,
                    temperature=temperature,
                    susceptibility=susceptibility,
                    mixture=mixture,
                )
                for kind, azimuthal, radial in zip(kinds, n, m, strict=True)
            )
        )

    def __repr__(self) -> str:
        """Return a one-line description: how many modes, and the first of them.

        Returns
        -------
        str
            For example ``ModeCollection(4 modes: MarcatiliMode(HE11, ...), ...)``.
        """
        count = len(self.modes)
        rest = ", ..." if count > 1 else ""
        return (
            f"ModeCollection({count} mode{'s' * (count != 1)}: {self.modes[0]!r}{rest})"
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        count = len(self.modes)
        return f"ModeCollection: {count} mode{'s' * (count != 1)}", [
            (str(index), repr(mode)) for index, mode in enumerate(self.modes)
        ]

    def __len__(self) -> int:
        """Return the number of modes.

        Returns
        -------
        int
            The size of the mode axis this set implies.
        """
        return len(self.modes)

    def __getitem__(self, index: int) -> Mode:  # type: ignore[override] - positional only; see below
        """Return one mode.

        Narrower than :meth:`collections.abc.Sequence.__getitem__`, which also accepts a
        slice, because the runtime type checker of the test session enforces the
        annotation and a slice would then return a sequence where a mode is declared.
        The mode axis is addressed by position everywhere in the package, so nothing
        slices a collection.

        Parameters
        ----------
        index : int
            Position on the mode axis.

        Returns
        -------
        Mode
            The mode.
        """
        return self.modes[index]

    def __iter__(self) -> Iterator[Mode]:
        """Iterate over the modes in order.

        Returns
        -------
        collections.abc.Iterator
            The modes.
        """
        return iter(self.modes)

    def dimlimits(
        self, z: float = 0.0
    ) -> tuple[CrossSection, tuple[float, ...], tuple[float, ...]]:
        """Return the cross-section the whole set shares.

        Parameters
        ----------
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        tuple
            As :meth:`solprop.modes.protocol.Mode.dimlimits`.

        Raises
        ------
        ModeError
            If the modes disagree at this position, which a taper of one mode but not
            another would cause.

        Examples
        --------
        >>> from solprop.modes import MarcatiliMode
        >>> from solprop.modes.collection import ModeCollection
        >>> ModeCollection([MarcatiliMode(125e-6, "He", 1.0)]).dimlimits()[0]
        'polar'
        """
        limits = self.modes[0].dimlimits(z)
        for index, mode in enumerate(self.modes[1:], start=1):
            if mode.dimlimits(z) != limits and self._same_stepindex_guide(mode):
                other = mode.dimlimits(z)
                limits = (
                    limits[0],
                    limits[1],
                    (max(limits[2][0], other[2][0]), limits[2][1]),
                )
            elif mode.dimlimits(z) != limits:
                raise ModeError(
                    f"the modes disagree about the cross-section at z = {z}: mode "
                    f"{index} has {mode.dimlimits(z)} against {limits}"
                )
        return limits

    @property
    def azimuthal_order(self) -> int:  # numpydoc ignore=RT01
        """Highest azimuthal order in the set.

        The bound that fixes how many azimuthal nodes a quartic product needs,
        ``n_theta >= 4 h + 1``.
        """
        return max(mode.azimuthal_order for mode in self.modes)

    @property
    def needs_full_rule(self) -> bool:  # numpydoc ignore=RT01
        """Whether the azimuthal coordinate has to be resolved.

        ``False`` for a set whose products are azimuthally symmetric -- any set of
        ``HE_1m`` modes -- which then needs the radial rule alone, and for a slab
        set, which is uniform in y; ``True`` otherwise.
        """
        kind = self.dimlimits()[0]
        if kind == "slab":
            return False
        return self.azimuthal_order > 0 or kind == "cartesian"

    @property
    def transverse_measure(self) -> TransverseMeasure:  # numpydoc ignore=RT01
        """What the set's transverse integrals measure.

        ``"per_unit_width"`` for a slab set, whose powers are W/m and energies
        J/m, and ``"area"`` otherwise (:func:`transverse_measure`).
        """
        return transverse_measure(self.modes)

    @property
    def z_constant(self) -> bool:  # numpydoc ignore=RT01
        """Whether nothing in the set varies along z.

        ``True`` only when every mode says so.
        """
        return all(mode.z_constant for mode in self.modes)

    @property
    def scale_invariant(self) -> bool:  # numpydoc ignore=RT01
        """Whether every mode rescales with the core radius.

        ``True`` when the transform may rescale its matrices along a taper rather
        than rebuilding them.
        """
        return all(mode.scale_invariant for mode in self.modes)

    def orthonormality(self, z: float = 0.0) -> Float[np.ndarray, "n_modes n_modes"]:
        """Return the matrix of normalised overlaps of the set.

        Parameters
        ----------
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_modes, n_modes)``; the identity for an orthonormal set.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.modes import MarcatiliMode
        >>> from solprop.modes.collection import ModeCollection
        >>> modes = ModeCollection(
        ...     [MarcatiliMode(125e-6, "He", 1.0, m=m) for m in (1, 2)]
        ... )
        >>> bool(np.max(np.abs(modes.orthonormality() - np.eye(2))) < 1e-9)
        True
        """
        return orthonormality(self.modes, z)
