# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Ordered, validated sets of modes, and the overlaps between them.

A multimode simulation propagates a *set* of modes through one transverse quadrature
rule and one pair of transform matrices, which only works if they share a
cross-section. :class:`ModeCollection` is that set: ordered, because the mode axis of
the kernel state is, and validated once at construction rather than at every use.

It also answers the two questions the translation layer asks of a mode set --
whether the azimuthal coordinate has to be resolved, and to what order -- and offers
the overlaps that say whether the set is orthonormal at all.

:func:`overlap` integrates adaptively with :func:`scipy.integrate.dblquad`,
deliberately *not* on the fixed rule. The fixed rule's orthonormality is the thing
being checked in ``tests/modes/test_quadrature.py``; if the check used the same rule it
would be true by construction and would say nothing.

Overlaps use the package normalisation (``docs/dev/conventions.md`` section 4),

.. math:: \langle m_1 | m_2 \rangle
   = \tfrac12 \sqrt{\varepsilon_0/\mu_0} \int e_1^* \cdot e_2 \, \mathrm{d}A
   \Big/ \sqrt{N_1 N_2},

so a mode overlaps itself by one. For a slab set (:mod:`solprop.modes.slab`) the
integral is over x alone, per metre of y, with :func:`scipy.integrate.quad`.
"""

import logging
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from jaxtyping import Float
from scipy.integrate import dblquad, quad

from solprop._display import Displayed
from solprop.constants import EPS0, ROOM_TEMPERATURE, C
from solprop.exceptions import ModeError
from solprop.modes.antiresonant import TubeLattice, VincettiMode
from solprop.modes.capillary import (
    LossSpec,
    MarcatiliMode,
    MarcatiliModel,
    ModeKind,
)
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


def _normalisation(mode: Mode, z: float) -> float:
    """Return a mode's closed-form normalisation.

    Parameters
    ----------
    mode : Mode
        The mode.
    z : float
        Position along the waveguide, m.

    Returns
    -------
    float
        ``N`` in W.

    Raises
    ------
    ModeError
        If the mode has no closed-form normalisation.
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
    scale = (
        0.5 * EPS0 * C / math.sqrt(_normalisation(first, z) * _normalisation(second, z))
    )

    def integrand(inner: float, outer: float) -> float:
        """Return the product of the two profiles times the area element.

        Parameters
        ----------
        inner : float
            The first cross-section coordinate, ``r`` or ``x`` in m. dblquad passes
            the *inner* variable first, which is why the order looks reversed.
        outer : float
            The second coordinate, ``theta`` in rad or ``y`` in m.

        Returns
        -------
        float
            The integrand of the overlap.
        """
        point = np.array([[inner, outer]])
        product = float(np.sum(first.field(point, z) * second.field(point, z)))
        return product * inner if kind == "polar" else product

    if kind == "slab":
        # Uniform in y: the overlap is per metre of y, over x alone, as N is.
        return scale * _overlap_along_x(first, second, (lower[0], upper[0]), z)
    # The outer variable is the second coordinate and the inner the first, with the
    # polar Jacobian r folded into the integrand above.
    value, _ = dblquad(integrand, lower[1], upper[1], lower[0], upper[0])
    return scale * value


def _overlap_along_x(
    first: Mode, second: Mode, limits: tuple[float, float], z: float
) -> float:
    """Return the integral over x of two slab profiles' product, per metre of y.

    Parameters
    ----------
    first, second : Mode
        Slab modes sharing one gap.
    limits : tuple of float
        The gap's edges in m.
    z : float
        Position along the waveguide, m.

    Returns
    -------
    float
        ``int e_1 . e_2 dx`` in m, integrated adaptively.
    """

    def along_x(x: float) -> float:
        """Return the product of the two slab profiles at one x.

        Parameters
        ----------
        x : float
            Transverse coordinate in m.

        Returns
        -------
        float
            The integrand of the per-unit-width overlap.
        """
        point = np.array([[x, 0.0]])
        return float(np.sum(first.field(point, z) * second.field(point, z)))

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
        Shape ``(n_modes, n_modes)``, symmetric, the identity for an orthonormal set.

    Raises
    ------
    ModeError
        If `modes` is empty, or the modes do not share a cross-section.

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
    count = len(modes)
    matrix = np.eye(count)
    for row in range(count):
        matrix[row, row] = overlap(modes[row], modes[row], z)
        for column in range(row + 1, count):
            value = overlap(modes[row], modes[column], z)
            matrix[row, column] = value
            matrix[column, row] = value
    return matrix


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
    >>> from solprop.modes import MarcatiliMode, SlabMode, transverse_measure
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

    Parameters
    ----------
    mode : Mode
        The mode.

    Returns
    -------
    str
        The first declared ``transverse_measure`` along the chain, else ``"area"``.
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
        """Check whether a channel shares the first physical guide.

        Parameters
        ----------
        mode : StepIndexMode
            Existing bound lossless mode.

        Returns
        -------
        bool
            Whether radius paths and both material laws agree.
        """
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
    def channel_labels(self) -> tuple[ChannelLabel, ...]:
        """Return each member's physical label, in the order of the mode axis.

        Returns
        -------
        tuple of ChannelLabel
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
    def with_polarisations(cls, modes: Sequence[Mode]) -> "ModeCollection":  # noqa: UP037 - bound after the class body
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
    def he1m(  # noqa: PLR0913 - MarcatiliMode's own keywords, carried through
        cls,
        radius: float | ZProfile,
        gas: Gas | str,
        pressure: float | ZProfile,
        count: int,
        *,
        model: MarcatiliModel = "full",
        loss: LossSpec = True,
        cladding: GlassIndex | complex | None = None,
        temperature: float | ZProfile = ROOM_TEMPERATURE,
        susceptibility: GasSusceptibility | None = None,
    ) -> "ModeCollection":  # noqa: UP037 - bound after the class body
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
        gas : Gas or str
            Filling gas.
        pressure : float or ZProfile
            Fill pressure, bar, constant or along z.
        count : int
            How many modes, at least one; the modes are HE_11 to HE_1count.
        model : {'full', 'reduced'}, optional
            Which Marcatili model the modes use.
        loss : bool, float or callable, optional
            Whether to include propagation loss, or the loss to use.
        cladding : GlassIndex, complex or None, optional
            The cladding index; ``None`` is the package default silica.
        temperature : float or ZProfile, optional
            Gas temperature, K, or a path of it along z.
        susceptibility : GasSusceptibility or None, optional
            A susceptibility to use in place of the built-in gas's.

        Returns
        -------
        ModeCollection
            The validated set, in radial order.

        Raises
        ------
        ModeError
            If `count` is below one.

        Examples
        --------
        >>> from solprop.modes import ModeCollection
        >>> modes = ModeCollection.he1m(125e-6, "He", 0.8, 4)
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
                    loss=loss,
                    cladding=cladding,
                    temperature=temperature,
                    susceptibility=susceptibility,
                )
                for m in range(1, count + 1)
            ]
        )

    @classmethod
    def slab(  # noqa: PLR0913 - SlabMode's own keywords, carried through
        cls,
        half_gap: float | ZProfile,
        orders: Sequence[int],
        gas: Gas | str | None = None,
        pressure: float | ZProfile = 0.0,
        *,
        polarisation: SlabPolarisation = "te",
        loss: LossSpec = True,
        cladding: GlassIndex | complex | None = None,
        temperature: float = ROOM_TEMPERATURE,
        susceptibility: GasSusceptibility | None = None,
        mixture: Mixture | None = None,
    ) -> "ModeCollection":  # noqa: UP037 - bound after the class body
        """Build the given orders of one hollow planar guide, uniform in y.

        A centred, symmetric launch couples only to odd orders, so the set a
        compressor run wants is usually ``(1, 3, 5)``; an offset or tilted launch
        needs the even ones too. Every keyword is
        :class:`~solprop.modes.slab.SlabMode`'s own and is given to each mode
        unchanged. For TE and TM rows in one set, build the two polarisations and
        pass them to the constructor, or use :meth:`with_polarisations`: each is a
        distinct mode with its own row.

        Parameters
        ----------
        half_gap : float or ZProfile
            Half the plate separation in m, constant or along z.
        orders : sequence of int
            Mode orders ``p >= 1``, non-empty and without repeats, in state order.
        gas : Gas, str or None, optional
            Filling gas; None for vacuum or an explicit mixture.
        pressure : float or ZProfile, optional
            Gas pressure in bar, constant or along z; default zero.
        polarisation : {'te', 'tm'}, optional
            ``"te"`` (E along y, the default) or ``"tm"`` (E along x), for every
            order.
        loss : bool, float or callable, optional
            Inherited mode loss policy.
        cladding : GlassIndex, complex or None, optional
            Plate index; None selects the default silica.
        temperature : float, optional
            Gas temperature in K.
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
        >>> modes = ModeCollection.slab(63.5e-6, (1, 3, 5), "Ar", 1.0)
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
                    loss=loss,
                    cladding=cladding,
                    temperature=temperature,
                    susceptibility=susceptibility,
                    mixture=mixture,
                )
                for order in orders
            )
        )

    @classmethod
    def vincetti(  # noqa: PLR0913 - the mode's material keywords, carried through
        cls,
        radius: float | ZProfile,
        gas: Gas | str | None = None,
        pressure: float | ZProfile = 0.0,
        count: int = 1,
        *,
        lattice: TubeLattice,
        loss: LossSpec = True,
        cladding: GlassIndex | complex | None = None,
        temperature: float | ZProfile = ROOM_TEMPERATURE,
        susceptibility: GasSusceptibility | None = None,
        mixture: Mixture | None = None,
    ) -> "ModeCollection":  # noqa: UP037 - bound after the class body
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
        loss : bool, float or callable, optional
            Inherited mode loss policy.
        cladding : GlassIndex, complex or None, optional
            Tube index; None selects the default silica.
        temperature : float or ZProfile, optional
            Gas temperature in K, or a path of it along z.
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
                    loss=loss,
                    cladding=cladding,
                    temperature=temperature,
                    susceptibility=susceptibility,
                    mixture=mixture,
                )
                for index in range(count)
            )
        )

    @classmethod
    def marcatili(  # noqa: PLR0913 - mode indices and MarcatiliMode's material keywords
        cls,
        radius: float | ZProfile,
        gas: Gas | str,
        pressure: float | ZProfile,
        kinds: Sequence[ModeKind],
        n: Sequence[int],
        m: Sequence[int],
        *,
        model: MarcatiliModel = "full",
        loss: LossSpec = True,
        cladding: GlassIndex | complex | None = None,
        temperature: float | ZProfile = ROOM_TEMPERATURE,
        susceptibility: GasSusceptibility | None = None,
    ) -> "ModeCollection":  # noqa: UP037 - bound after the class body
        """Build an ordered mixture of hybrid and transverse capillary modes.

        Parameters
        ----------
        radius : float or ZProfile
            Core radius, m, constant or along z.
        gas : Gas or str
            Filling gas.
        pressure : float or ZProfile
            Fill pressure, bar, constant or along z.
        kinds : sequence of {'HE', 'TE', 'TM'}
            Family of each mode, non-empty.
        n : sequence of int
            Azimuthal indices, one per family; nonzero for HE and zero for TE/TM.
        m : sequence of int
            Positive radial indices, one per family.
        model : {'full', 'reduced'}, optional
            Marcatili approximation used by every mode.
        loss : bool, float or callable, optional
            Propagation-loss specification passed to every mode.
        cladding : GlassIndex, complex or None, optional
            Cladding index; ``None`` selects the package default silica.
        temperature : float or ZProfile, optional
            Gas temperature, K, positive, or a path of it along z.
        susceptibility : GasSusceptibility or None, optional
            Explicit susceptibility in place of the selected fit.

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
        ...     125e-6, "He", 1.0, ["HE", "HE", "TM"], [1, 2, 0], [1, 1, 1]
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
                    loss=loss,
                    cladding=cladding,
                    temperature=temperature,
                    susceptibility=susceptibility,
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

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
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
    def azimuthal_order(self) -> int:
        """Return the highest azimuthal order in the set.

        Returns
        -------
        int
            The bound that fixes how many azimuthal nodes a quartic product needs,
            ``n_theta >= 4 h + 1``.
        """
        return max(mode.azimuthal_order for mode in self.modes)

    @property
    def needs_full_rule(self) -> bool:
        """Return whether the azimuthal coordinate has to be resolved.

        Returns
        -------
        bool
            ``False`` for a set whose products are azimuthally symmetric -- any set of
            ``HE_1m`` modes -- which then needs the radial rule alone, and for a slab
            set, which is uniform in y; ``True`` otherwise.
        """
        kind = self.dimlimits()[0]
        if kind == "slab":
            return False
        return self.azimuthal_order > 0 or kind == "cartesian"

    @property
    def transverse_measure(self) -> TransverseMeasure:
        """Return what the set's transverse integrals measure.

        Returns
        -------
        str
            ``"per_unit_width"`` for a slab set, whose powers are W/m and energies
            J/m, and ``"area"`` otherwise (:func:`transverse_measure`).
        """
        return transverse_measure(self.modes)

    @property
    def z_constant(self) -> bool:
        """Return whether nothing in the set varies along z.

        Returns
        -------
        bool
            ``True`` only when every mode says so.
        """
        return all(mode.z_constant for mode in self.modes)

    @property
    def scale_invariant(self) -> bool:
        """Return whether every mode rescales with the core radius.

        Returns
        -------
        bool
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
