# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The request a run records statistics by: :class:`StatsSpec` and its apertures.

- :class:`StatsSpec`: the names, the between-planes subset, the energy windows and the
  far-field regions and apertures, checked against the catalogue on construction.
- :class:`ApertureSet`, :data:`ApertureWeighting`, :func:`check_aperture_choices` and
  the limits :data:`MAX_APERTURES` and :data:`MAX_REGIONS` (SEP 23).
- :func:`far_field_tables`: the far-field statistics' tables of a spec on Cartesian
  columns, which every backend and the saved-plane statistics share.
"""

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from jaxtyping import Complex, Float

from solprop.exceptions import ApertureEdgeWarning, OutputError, ProcessingError
from solprop.spatial import (
    ApertureOutline,
    ApertureProfile,
    ApertureShape,
    ApertureTables,
    aperture_outline,
    aperture_scale,
    check_aperture_profile,
    check_aperture_shape,
)
from solprop.stats.catalogue import CATALOGUE

__all__ = [
    "MAX_APERTURES",
    "MAX_REGIONS",
    "ApertureSet",
    "ApertureWeighting",
    "StatsSpec",
    "check_aperture_choices",
    "far_field_tables",
]


ApertureWeighting = Literal["amplitude", "quadrature"]
"""How an aperture's profile weights the collected spectrum (SEP 23 section 4.5)."""

MAX_APERTURES = 16
"""The most apertures one :class:`ApertureSet` carries."""

MAX_REGIONS = 16
"""The most rows :attr:`StatsSpec.spectral_regions` carries."""


def _aperture_array(
    values: object, shape: tuple[int, ...] | None, name: str
) -> np.ndarray:
    """Own a finite float64 copy of one aperture table, read-only.

    - ``shape``: the shape it must have; ``None`` asks for any flat vector.
    - Returns: a read-only, C-contiguous float64 copy.
    """
    stated = "a flat vector" if shape is None else f"shape {shape}"
    try:
        array = np.array(values, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError) as error:
        msg = f"ApertureSet.{name} must be numeric, {stated}"
        raise ProcessingError(msg) from error
    fits = array.ndim == 1 if shape is None else array.shape == shape
    if not fits or not np.all(np.isfinite(array)):
        msg = f"ApertureSet.{name} must be finite, {stated}, got {array.shape}"
        raise ProcessingError(msg)
    array.setflags(write=False)
    return array


def _known(check: Callable[[object], None], value: object) -> None:
    """Refuse an unknown aperture profile or shape as a malformed request.

    - ``check``: :func:`~solprop.spatial.check_aperture_profile` or
      :func:`~solprop.spatial.check_aperture_shape`, whose refusal is the spatial
      tables' :class:`~solprop.exceptions.OutputError`.
    """
    try:
        check(value)
    except OutputError as error:
        raise ProcessingError(str(error)) from error


def check_aperture_choices(profile: object, weighting: object) -> None:
    """Refuse, by name, an aperture profile or weighting that does not exist.

    The check :class:`ApertureSet` makes of its two choices. It takes any object,
    because a caller outside the test session's runtime type checker can pass
    anything.

    Parameters
    ----------
    profile : object
        The edge profile asked for.
    weighting : object
        The weighting asked for.

    Raises
    ------
    ProcessingError
        If either is not one of its choices.

    Examples
    --------
    >>> from solprop.stats import check_aperture_choices
    >>> check_aperture_choices("tanh", "quadrature") is None
    True
    """
    _known(check_aperture_profile, profile)
    if weighting not in ("amplitude", "quadrature"):
        msg = (
            f"unknown aperture weighting {weighting!r}; choose 'amplitude' or "
            "'quadrature'"
        )
        raise ProcessingError(msg)


def _off_axis(shape: str) -> int:
    """Return the index of the coordinate a half-plane does not depend on."""
    return 1 if shape.startswith("half_plane_x") else 0


@dataclass(frozen=True, eq=False)
class ApertureSet:
    """Apertures in a far-field or Fourier plane behind a free-space run.

    The tables of ``aperture_spectrum`` and ``aperture_on_axis_spectrum`` (SEP 23):
    each aperture collects the light whose paraxial image ``x = z c kappa / omega``
    falls inside it, a chromatic weight on the stored transverse spectrum. Every
    position is physical: a beam travelling towards positive x lands at positive x.
    Circles by default; `shapes` asks for an ellipse, a rectangle or a half-plane on
    any row. Equality is by identity; the arrays are owned read-only copies.

    Attributes
    ----------
    centres : numpy.ndarray
        Aperture centres ``(x, y)`` in the mask plane, in m, shape
        ``(n_apertures, 2)``. A half-plane has an edge rather than a centre: its row
        holds the edge's position on the axis it names and zero on the other.
    diameters : numpy.ndarray
        Diameters in m, shape ``(n_apertures,)``: finite and positive on a circle's
        row, zero on any other. On a line a circle is a slit of this full width
        along x.
    distances : numpy.ndarray
        Distance from the exit face to the mask plane, or a lens's focal length for
        a Fourier plane, in m, finite and positive, shape ``(n_apertures,)``.
    edges : numpy.ndarray or None
        Edge widths ``w`` in m for ``tanh`` and ``rcos``, orders ``n >= 2`` for
        ``supergauss``, zero for a ``hard`` row beside others, shape
        ``(n_apertures,)``; ``None`` when every row is ``hard``, and only then. An
        ellipse's edge width is in its circular frame: ``w a/sqrt(a b)`` along x and
        ``w b/sqrt(a b)`` along y, in m.
    profile : {'hard', 'tanh', 'rcos', 'supergauss'}
        The edge profile the set shares (:func:`solprop.spatial.aperture_profile`).
        The default is ``hard``.
    weighting : {'amplitude', 'quadrature'}
        ``amplitude`` (the default) models a mask of amplitude transmission ``W``
        and weights the collected spectrum by ``W**2``; ``quadrature`` uses a smooth
        profile once, at the corrected diameter, as a quadrature weight for a hard
        hole. Quadrature is refused with ``supergauss``.
    labels : tuple of str
        One unique non-empty label per aperture, recorded with the run's
        configuration; ``None`` (the default) labels them ``"0"``, ``"1"``, ...
    shapes : tuple of str or None
        Each row's outline, one of :data:`solprop.spatial.APERTURE_SHAPES`: ``circle``,
        ``ellipse``, ``rectangle``, or a half-plane named for the side it keeps,
        ``half_plane_x_ge`` (``x >= X``), ``half_plane_x_lt`` (``x < X``),
        ``half_plane_y_ge`` and ``half_plane_y_lt``. ``None`` (the default) is a
        circle on every row, and all circles are stored as ``None``. A set of unset
        shapes, profiles and extents records and hashes as it did before they existed.
    profiles : tuple of str or None
        Each row's edge profile, for a set that mixes them; ``profile`` is then left
        at its default, and rows that all agree are stored as that ``profile``.
    extents : numpy.ndarray or None
        Full widths ``(D_x, D_y)`` in m, shape ``(n_apertures, 2)``: finite and
        positive on the row of an ellipse or a rectangle, zero on any other; ``None``
        when no row needs them, and only then.

    Raises
    ------
    ProcessingError
        If there are no apertures or more than
        :data:`~solprop.stats.spec.MAX_APERTURES`; an array is
        not finite or not of its stated shape; a diameter, an extent or a distance
        breaks its rule above; a half-plane's centre is off the axis it names; an edge
        is missing, given for a ``hard`` set or row, or out of its range (with ``D``
        a circle's diameter, ``sqrt(D_x D_y)`` of an ellipse and the smaller of a
        rectangle's widths: ``0 < w <= D/4`` for ``rcos`` and for ``tanh`` under
        amplitude weighting, ``0 < w <= D/8`` for ``tanh`` under quadrature weighting,
        ``D/10`` on a rectangle; ``w > 0`` on a half-plane; ``n >= 2`` for
        ``supergauss``); quadrature weighting or a half-plane is asked of
        ``supergauss``; ``profile`` is given beside `profiles`; an aperture reaches the
        paraxial disc's edge, ``|X| + D/2 (+ w for rcos) >= z`` (a half-plane's edge
        must lie inside it); or the labels are not unique non-empty strings, one per
        aperture.

    Warns
    -----
    solprop.exceptions.ApertureEdgeWarning
        Once per set, for ``tanh`` under amplitude weighting with ``r/w < 4``
        (``w > D/8``): the profile slopes at its centre, so a bright beam there is
        collected with a lattice error near ``1e-5``. The set is built as asked;
        ``rcos`` is the recommended smooth profile.

    Examples
    --------
    >>> from solprop.stats import ApertureSet
    >>> holes = ApertureSet(
    ...     centres=[[4e-3, 4e-3], [-4e-3, -4e-3]],
    ...     diameters=[1.5e-3, 1.5e-3],
    ...     distances=[0.25, 0.25],
    ...     edges=[40e-6, 40e-6],
    ...     profile="tanh",
    ...     weighting="quadrature",
    ...     labels=("probe", "signal"),
    ... )
    >>> holes.n_apertures, holes.labels
    (2, ('probe', 'signal'))

    A knife edge at a fixed position in the plane, beside an apodised hole:

    >>> mixed = ApertureSet(
    ...     centres=[[2e-3, 0.0], [-4e-3, 0.0]],
    ...     diameters=[0.0, 1.5e-3],
    ...     distances=[0.25, 0.25],
    ...     edges=[0.0, 40e-6],
    ...     shapes=("half_plane_x_ge", "circle"),
    ...     profiles=("hard", "rcos"),
    ... )
    >>> mixed.row_shapes
    ('half_plane_x_ge', 'circle')
    """

    centres: Sequence[Sequence[float]] | np.ndarray
    diameters: Sequence[float] | np.ndarray
    distances: Sequence[float] | np.ndarray
    edges: Sequence[float] | np.ndarray | None = None
    profile: ApertureProfile = "hard"
    weighting: ApertureWeighting = "amplitude"
    labels: tuple[str, ...] | None = None
    shapes: tuple[ApertureShape, ...] | None = None
    profiles: tuple[ApertureProfile, ...] | None = None
    extents: Sequence[Sequence[float]] | np.ndarray | None = None

    def __post_init__(self) -> None:
        """Own the tables and check them against the list of SEP 23 section 5.

        Raises
        ------
        ProcessingError
            Naming the field and the rule it breaks.
        """
        diameters = _aperture_array(self.diameters, None, "diameters")
        count = diameters.size
        if not 1 <= count <= MAX_APERTURES:
            msg = f"an ApertureSet holds 1 to {MAX_APERTURES} apertures, got {count}"
            raise ProcessingError(msg)
        object.__setattr__(self, "diameters", diameters)
        object.__setattr__(
            self, "centres", _aperture_array(self.centres, (count, 2), "centres")
        )
        object.__setattr__(
            self, "distances", _aperture_array(self.distances, (count,), "distances")
        )
        check_aperture_choices(self.profile, self.weighting)
        self._adopt_rows(count)
        self._check_outlines(count)
        if np.any(np.asarray(self.distances) <= 0):
            msg = "every aperture diameter and distance must be positive"
            raise ProcessingError(msg)
        self._check_edges(count)
        self._check_paraxial()
        labels = (
            tuple(str(index) for index in range(count))
            if self.labels is None
            else tuple(self.labels)
        )
        if (
            len(labels) != count
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != count
        ):
            msg = f"ApertureSet.labels must be {count} unique non-empty strings"
            raise ProcessingError(msg)
        object.__setattr__(self, "labels", labels)

    def _adopt_rows(self, count: int) -> None:
        """Check the per-row choices and store rows that say nothing new as unset.

        All circles are stored as no `shapes`, and rows of one profile as that
        `profile`, so that one request has one record.
        """
        if self.shapes is not None:
            shapes = tuple(self.shapes)
            if len(shapes) != count:
                msg = f"ApertureSet.shapes must hold {count}, got {len(shapes)}"
                raise ProcessingError(msg)
            for shape in shapes:
                _known(check_aperture_shape, shape)
            object.__setattr__(
                self, "shapes", None if set(shapes) == {"circle"} else shapes
            )
        if self.profiles is None:
            return
        profiles = tuple(self.profiles)
        if len(profiles) != count:
            msg = (
                f"ApertureSet.profiles must hold {count} profiles, got {len(profiles)}"
            )
            raise ProcessingError(msg)
        for profile in profiles:
            _known(check_aperture_profile, profile)
        if self.profile != "hard":
            msg = (
                "give the set's edge profile or a profile per row, not both; pass "
                "profile or profiles"
            )
            raise ProcessingError(msg)
        one = set(profiles)
        object.__setattr__(self, "profile", profiles[0] if len(one) == 1 else "hard")
        object.__setattr__(self, "profiles", None if len(one) == 1 else profiles)

    def _check_outlines(self, count: int) -> None:
        """Own the extents and check each row's measures against its shape."""
        shapes = self.row_shapes
        circle = np.array([shape == "circle" for shape in shapes])
        sized = np.array([shape in ("ellipse", "rectangle") for shape in shapes])
        diameters = np.asarray(self.diameters)
        if np.any(diameters[circle] <= 0):
            msg = "every aperture diameter and distance must be positive"
            raise ProcessingError(msg)
        if np.any(diameters[~circle] != 0):
            msg = "a diameter is a circle's; give zero on every row of another shape"
            raise ProcessingError(msg)
        if self.extents is None:
            if np.any(sized):
                msg = "an ellipse or a rectangle needs extents, (D_x, D_y) per aperture"
                raise ProcessingError(msg)
        else:
            extents = _aperture_array(self.extents, (count, 2), "extents")
            object.__setattr__(self, "extents", extents)
            if not np.any(sized):
                msg = (
                    "extents are an ellipse's or a rectangle's, and would be silently "
                    "unused here; pass extents=None"
                )
                raise ProcessingError(msg)
            if np.any(extents[sized] <= 0) or np.any(extents[~sized] != 0):
                msg = (
                    "ApertureSet.extents must be positive on the row of an ellipse or "
                    "a rectangle and zero on any other"
                )
                raise ProcessingError(msg)
        centres = np.asarray(self.centres)
        for row, shape in enumerate(shapes):
            if shape.startswith("half_plane") and centres[row, _off_axis(shape)] != 0:
                msg = (
                    f"aperture {row}, a half-plane, has an edge on one axis only; its "
                    "centre on the other must be zero"
                )
                raise ProcessingError(msg)

    def _edge_measures(self) -> Float[np.ndarray, " n_apertures"]:
        """Return the width each row's edge width is measured against.

        - Returns: a circle's diameter, ``sqrt(D_x D_y)`` of an ellipse (the diameter of
          its circular frame), the smaller of a rectangle's widths, and zero on a
          half-plane, which has none, in m, shape ``(n_apertures,)``.
        """
        measures = np.array(np.asarray(self.diameters))
        if self.extents is not None:
            extents = np.asarray(self.extents)
            for row, shape in enumerate(self.row_shapes):
                if shape == "ellipse":
                    measures[row] = math.sqrt(extents[row, 0] * extents[row, 1])
                elif shape == "rectangle":
                    measures[row] = np.min(extents[row])
        return measures

    def _check_edges(self, count: int) -> None:
        """Own the edge table and check it against the profiles and the weighting."""
        kinds = self.row_profiles
        smooth = [kind for kind in kinds if kind != "hard"]
        if not smooth:
            if self.edges is not None:
                msg = (
                    "a hard aperture has no edge, and an edge given beside it would "
                    "be silently unused; pass edges=None"
                )
                raise ProcessingError(msg)
            return
        if self.edges is None:
            msg = f"a {smooth[0]} aperture needs edges, one per aperture"
            raise ProcessingError(msg)
        edges = _aperture_array(self.edges, (count,), "edges")
        object.__setattr__(self, "edges", edges)
        rows = np.array(kinds)
        if np.any(edges[rows == "hard"] != 0):
            msg = (
                "a hard aperture has no edge, and an edge given beside it would be "
                "silently unused; give zero on its row"
            )
            raise ProcessingError(msg)
        for kind in dict.fromkeys(smooth):
            if kind == "supergauss":
                self._check_orders(edges, rows == kind)
            else:
                self._check_widths(kind, edges, rows == kind)

    def _check_orders(
        self, edges: Float[np.ndarray, " n_apertures"], rows: np.ndarray
    ) -> None:
        """Check the rows with a supergauss profile."""
        if self.weighting == "quadrature":
            msg = (
                "quadrature weighting needs a symmetric edge, and a supergauss "
                "profile has none; use amplitude weighting"
            )
            raise ProcessingError(msg)
        if any(
            shape.startswith("half_plane")
            for shape, row in zip(self.row_shapes, rows, strict=True)
            if row
        ):
            msg = (
                "a supergauss profile falls from a centre, and a half-plane has none; "
                "use a hard, tanh or rcos edge"
            )
            raise ProcessingError(msg)
        if np.any(edges[rows] < 2):
            msg = "every supergauss order must be at least 2"
            raise ProcessingError(msg)

    def _check_widths(
        self, kind: str, edges: Float[np.ndarray, " n_apertures"], rows: np.ndarray
    ) -> None:
        """Check the rows with one symmetric edge profile.

        - ``kind``: the profile, ``tanh`` or ``rcos``.
        """
        shapes = self.row_shapes
        half_plane = np.array([shape.startswith("half_plane") for shape in shapes])
        rectangle = np.array([shape == "rectangle" for shape in shapes])
        tail = kind == "tanh" and self.weighting == "quadrature"
        measures = self._edge_measures()
        for chosen, fraction in (
            (rows & ~half_plane & ~rectangle, 8 if tail else 4),
            # Two tails, each (w/2a) exp(-2a/w) of a width: D/10 keeps their sum below
            # the 2e-5 that D/8 keeps a circle's (SEP 23 section 4.5).
            (rows & rectangle, 10 if tail else 4),
        ):
            if np.any(edges[chosen] <= 0) or np.any(
                edges[chosen] > measures[chosen] / fraction
            ):
                msg = (
                    f"a {kind} aperture under {self.weighting} weighting needs "
                    f"0 < w <= D/{fraction} for every aperture"
                )
                raise ProcessingError(msg)
        if np.any(edges[rows & half_plane] <= 0):
            msg = f"a {kind} half-plane needs a positive edge width"
            raise ProcessingError(msg)
        if kind == "tanh" and self.weighting == "amplitude":
            # tanh is not flat at its centre: W'(0) = -sech^2(r/w) / 2w, so for
            # r/w < 4 a bright beam there converges only at third order in the mask
            # plane's spacing (about 1e-5 of the collected energy at r/w = 2). At
            # r/w >= 4 the slope is 0.02 of that and the error near 2e-7.
            wide = np.flatnonzero(rows & ~half_plane & (measures / 2 < 4 * edges))
            if wide.size:
                warnings.warn(
                    f"tanh apertures {wide.tolist()} under amplitude weighting have "
                    "r/w < 4: the profile slopes at its centre, so light there is "
                    "collected with a lattice error near 1e-5; rcos is flat inside "
                    "r - w and is preferred (or w <= D/8)",
                    ApertureEdgeWarning,
                    stacklevel=5,
                )

    def _check_paraxial(self) -> None:
        """Check that every aperture lies inside the paraxial disc ``|x| < z``.

        A half-plane reaches the disc's edge by its nature, and the columns there
        carry zero weight (SEP 23 section 4.2); its edge must lie inside the disc, or
        it would select all of the disc or nothing.
        """
        centres = np.asarray(self.centres)
        reach = np.array(
            [
                self._reach(row, shape, centres[row])
                for row, shape in enumerate(self.row_shapes)
            ]
        )
        outside = np.flatnonzero(reach >= np.asarray(self.distances))
        if outside.size:
            msg = (
                f"apertures {outside.tolist()} reach the paraxial disc's edge, "
                "|X| + D/2 (+ w for rcos) >= z; a component there cannot leave the "
                "exit face (SEP 23 section 4.2)"
            )
            raise ProcessingError(msg)

    def _reach(self, row: int, shape: str, centre: Float[np.ndarray, " 2"]) -> float:
        """Return how far from the origin a row's outline reaches in the mask plane.

        - ``centre``: its centre, m.
        - Returns: the reach in m. A raised cosine's reaches its edge width further out;
          the unbounded ``tanh`` and ``supergauss`` tails are zeroed at the disc's edge.
        """
        width = (
            float(np.asarray(self.edges)[row])
            if self.row_profiles[row] == "rcos"
            else 0.0
        )
        if shape.startswith("half_plane"):
            return float(abs(centre[1 - _off_axis(shape)])) + width
        if shape == "circle":
            return (
                float(np.hypot(centre[0], centre[1]))
                + float(np.asarray(self.diameters)[row]) / 2
                + width
            )
        half_x, half_y = 0.5 * np.asarray(self.extents)[row]
        if shape == "rectangle":
            return float(
                np.hypot(
                    abs(centre[0]) + half_x + width, abs(centre[1]) + half_y + width
                )
            )
        # The ellipse's semi-axes grow by w/sqrt(a b) of themselves in its frame.
        grown = max(half_x, half_y) * (1.0 + width / math.sqrt(half_x * half_y))
        return float(np.hypot(centre[0], centre[1])) + float(grown)

    @property
    def row_shapes(self) -> tuple[ApertureShape, ...]:  # numpydoc ignore=RT01
        """Every row's outline.

        One of :data:`solprop.spatial.APERTURE_SHAPES` per aperture.
        """
        if self.shapes is None:
            return ("circle",) * int(np.asarray(self.diameters).size)
        return self.shapes

    @property
    def row_profiles(self) -> tuple[ApertureProfile, ...]:  # numpydoc ignore=RT01
        """Every row's edge profile.

        One of :data:`solprop.spatial.APERTURE_PROFILES` per aperture.
        """
        if self.profiles is None:
            return (self.profile,) * int(np.asarray(self.diameters).size)
        return self.profiles

    @property
    def n_apertures(self) -> int:  # numpydoc ignore=RT01
        """How many apertures the set holds."""
        return int(np.asarray(self.diameters).size)

    def same_as(self, other: "ApertureSet | None") -> bool:
        """Return whether another set describes the same apertures, entry for entry.

        Parameters
        ----------
        other : ApertureSet or None
            The set to compare.

        Returns
        -------
        bool
            True when every table, each row's shape and profile, the weighting and
            the labels agree exactly: a structural comparison of two requests.
        """
        if other is None:
            return False
        tables = ("centres", "diameters", "distances")
        return (
            all(
                np.array_equal(np.asarray(getattr(self, name)), getattr(other, name))
                for name in tables
            )
            and all(
                (mine is None) == (theirs is None)
                and (
                    mine is None or np.array_equal(np.asarray(mine), np.asarray(theirs))
                )
                for mine, theirs in (
                    (self.edges, other.edges),
                    (self.extents, other.extents),
                )
            )
            and (self.row_shapes, self.row_profiles, self.weighting, self.labels)
            == (other.row_shapes, other.row_profiles, other.weighting, other.labels)
        )


def _no_windows() -> Float[np.ndarray, "n_windows 2"]:
    """Return the empty window list a spec defaults to."""
    return np.empty((0, 2))


def _no_regions() -> Float[np.ndarray, "n_regions 4"]:
    """Return the empty region table a spec defaults to."""
    return np.empty((0, 4))


_APERTURE_NAMES = frozenset(
    name for name, entry in CATALOGUE.items() if "apertures" in entry.requires
)
"""The statistics that read :attr:`StatsSpec.apertures`."""


@dataclass(frozen=True, eq=False)
class StatsSpec:
    """Which statistics a run asks for, and the windows and period they need.

    This is the object of the kernel contract §8: the translation layer carries it into
    a backend, the driver hands it to :func:`~solprop.stats.collect`, and both read the
    same names. Equality is by identity (``eq=False``) because ``energy_windows`` is an
    array.

    Attributes
    ----------
    names : tuple
        Statistics to evaluate at every saved plane, in the order they are evaluated.
        Every name must be in :data:`~solprop.stats.catalogue.CATALOGUE` and must be
        implemented.
    energy_windows : numpy.ndarray
        Spectral windows, shape ``(n_windows, 2)``, rad/s, each row ``(lo, hi)`` with
        ``lo < hi``. Infinite edges are allowed, so that a pair of windows can tile the
        whole axis. Required by ``energy_window`` and ``peak_power_window``.
    periodic : tuple
        The subset of ``names`` a backend also evaluates between planes, for step-level
        diagnostics. Must be a subset of ``names``, and every entry must carry
        :attr:`~solprop.stats.StatDefinition.between_planes`: a statistic whose cost is
        only bearable once per plane is refused here rather than made the run's per-step
        cost.
    period : int
        How many accepted steps apart the periodic subset is evaluated, at least 1.
    apertures : ApertureSet or None
        Far-field apertures, required by ``aperture_spectrum`` and
        ``aperture_on_axis_spectrum`` (SEP 23). ``None`` by default, and then left
        out of the run's configuration record, so every earlier record keeps its
        hash.
    spectral_regions : numpy.ndarray
        Boxes of physical transverse wavevector in rad/m, shape ``(n_regions, 4)``, rows
        ``(kx_lo, kx_hi, ky_lo, ky_hi)``, each a half-open box ``[lo, hi)`` on both
        axes, at most :data:`~solprop.stats.spec.MAX_REGIONS` rows, infinite edges
        allowed. Required by ``region_spectrum``. Empty by default, and then left out of
        the configuration record.

    Raises
    ------
    ProcessingError
        If a name is unknown, reserved or repeated; if a window or a region is
        malformed; if ``periodic`` is not a subset of ``names`` or names a statistic
        no backend evaluates between planes; if ``period`` is below 1; or if a name
        needs a far-field table the spec does not carry.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.stats import StatsSpec
    >>> spec = StatsSpec(
    ...     names=("energy", "energy_window"),
    ...     energy_windows=np.array([[2.0e15, 3.0e15]]),
    ... )
    >>> spec.n_windows
    1
    """

    names: tuple[str, ...] = ()
    energy_windows: Float[np.ndarray, "n_windows 2"] = field(
        default_factory=_no_windows
    )
    periodic: tuple[str, ...] = ()
    period: int = 1
    apertures: ApertureSet | None = None
    spectral_regions: Float[np.ndarray, "n_regions 4"] = field(
        default_factory=_no_regions
    )

    def __post_init__(self) -> None:
        """Check the names, the tables and the period, and freeze the tables.

        Raises
        ------
        ProcessingError
            If a name is unknown, reserved or repeated; if a window or a region is
            malformed; if ``periodic`` is not a subset of ``names`` or names a
            statistic no backend evaluates between planes; if ``period`` is below 1;
            or if a name needs a far-field table the spec does not carry.
        """
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "periodic", tuple(self.periodic))
        _check_names(self.names)
        extra = sorted(set(self.periodic) - set(self.names))
        if extra:
            msg = f"the periodic subset asks for statistics the run does not: {extra}"
            raise ProcessingError(msg)
        _check_between_planes(self.periodic)
        if self.period < 1:
            msg = f"the statistics period must be at least 1, got {self.period}"
            raise ProcessingError(msg)
        windows = np.array(self.energy_windows, dtype=np.float64, copy=True)
        _check_windows(windows)
        windows.setflags(write=False)
        object.__setattr__(self, "energy_windows", windows)
        regions = np.array(self.spectral_regions, dtype=np.float64, copy=True)
        _check_regions(regions)
        regions.setflags(write=False)
        object.__setattr__(self, "spectral_regions", regions)
        missing = (
            sorted(set(self.names) & _APERTURE_NAMES) if self.apertures is None else []
        )
        if "region_spectrum" in self.names and self.n_regions == 0:
            missing.append("region_spectrum")
        if missing:
            msg = (
                f"the statistics {missing} need a far-field table this spec does not "
                "carry: apertures=ApertureSet(...) for the aperture spectra, "
                "spectral_regions for region_spectrum"
            )
            raise ProcessingError(msg)

    @property
    def n_apertures(self) -> int:  # numpydoc ignore=RT01
        """How many far-field apertures the spec carries.

        The apertures of :attr:`apertures`, or zero without them.
        """
        return 0 if self.apertures is None else self.apertures.n_apertures

    @property
    def n_regions(self) -> int:  # numpydoc ignore=RT01
        """How many regions of transverse wavevector the spec carries.

        The rows of :attr:`spectral_regions`.
        """
        return int(self.spectral_regions.shape[0])

    @property
    def n_windows(self) -> int:  # numpydoc ignore=RT01
        """How many spectral windows the spec carries."""
        return int(self.energy_windows.shape[0])


def _check_names(names: Sequence[str]) -> None:
    """Check that every requested statistic exists and appears once."""
    unknown = sorted(set(names) - set(CATALOGUE))
    if unknown:
        msg = f"unknown statistics {unknown}; the catalogue holds {sorted(CATALOGUE)}"
        raise ProcessingError(msg)
    repeated = sorted({name for name in names if list(names).count(name) > 1})
    if repeated:
        msg = f"each statistic may be asked for once; these repeat: {repeated}"
        raise ProcessingError(msg)


def _check_between_planes(names: Sequence[str]) -> None:
    """Check that every periodic statistic is one a backend evaluates between planes.

    The refusal is by name and at construction because evaluating them anyway can cost
    a fifth of a step: ``zdw`` is a root search per mode and does not belong on every
    accepted step.
    """
    refused = sorted(name for name in names if not CATALOGUE[name].between_planes)
    if refused:
        available = sorted(
            name for name, entry in CATALOGUE.items() if entry.between_planes
        )
        msg = (
            f"the statistics {refused} cannot be evaluated between save planes, only "
            f"at them; the periodic subset may name {available}"
        )
        raise ProcessingError(msg)


def _check_windows(windows: Float[np.ndarray, "n_windows 2"]) -> None:
    """Check that every spectral window is a properly ordered pair.

    - ``windows``: window edges, shape ``(n_windows, 2)``, rad/s.
    """
    if np.isnan(windows).any():
        msg = "energy_windows must not hold a NaN edge"
        raise ProcessingError(msg)
    bad = np.flatnonzero(windows[:, 0] >= windows[:, 1])
    if bad.size:
        msg = (
            f"every energy window needs lo < hi; rows {bad.tolist()} do not: "
            f"{windows[bad].tolist()}"
        )
        raise ProcessingError(msg)


def _check_regions(regions: Float[np.ndarray, "n_regions 4"]) -> None:
    """Check that every region of transverse wavevector is a properly ordered box.

    - ``regions``: region edges, shape ``(n_regions, 4)``, rad/m.
    """
    if regions.shape[0] > MAX_REGIONS:
        msg = (
            f"spectral_regions holds at most {MAX_REGIONS} rows, got {regions.shape[0]}"
        )
        raise ProcessingError(msg)
    if np.isnan(regions).any():
        msg = "spectral_regions must not hold a NaN edge"
        raise ProcessingError(msg)
    bad = np.flatnonzero(
        (regions[:, 0] >= regions[:, 1]) | (regions[:, 2] >= regions[:, 3])
    )
    if bad.size:
        msg = (
            f"every spectral region needs lo < hi on both axes; rows {bad.tolist()} "
            f"do not: {regions[bad].tolist()}"
        )
        raise ProcessingError(msg)


def far_field_tables(
    spec: StatsSpec,
    omega: Float[np.ndarray, " n_omega"],
    *,
    k_x: Float[np.ndarray, " n_kx"],
    k_y: Float[np.ndarray, " n_ky"],
    weights: Float[np.ndarray, " n_weights"],
    on_axis: Complex[np.ndarray, " n_axis_weights"],
) -> ApertureTables:
    """Form the far-field statistics' tables of a spec on Cartesian columns (SEP 23 §6).

    The per-aperture constants -- each row's outline of
    :func:`solprop.spatial.aperture_outline`, whose half-width under quadrature
    weighting is the corrected ``D_q/2`` of a circle or an ellipse (unchanged on a
    line, whose edges are straight, and on the straight edges of a rectangle or a
    half-plane) -- and the scale table of :func:`solprop.spatial.aperture_scale`, with
    the columns' axes, weights and on-axis vector and the spec's regions. On a line
    every column has ``kappa_y = 0`` and the light is uniform in y, so a rectangle has
    no y edges to meet: it is a slit of its x width, the row a circle of that diameter
    forms. Every backend reaches this through
    :func:`solprop.translate.spatial.aperture_tables` at construction and
    :func:`solprop.stats.collect` calls it for a saved plane, so all of them use the
    same scale to the bit.

    Parameters
    ----------
    spec : solprop.stats.StatsSpec
        The request and its far-field tables.
    omega : numpy.ndarray
        Physical frequency axis in rad/s, shape ``(n_omega,)``.
    k_x, k_y : numpy.ndarray
        The stored transverse wavevector axes in rad/m, in FFT order; ``k_y`` has one
        entry, zero, on a line uniform in y.
    weights : numpy.ndarray
        Parseval weights of the stored columns, shape ``(n_x*n_y,)``.
    on_axis : numpy.ndarray
        On-axis reconstruction weights of the columns, complex, shape ``(n_x*n_y,)``.

    Returns
    -------
    solprop.spatial.ApertureTables
        The tables.

    Raises
    ------
    ProcessingError
        On a line, for an aperture whose centre is off ``y = 0``, a half-plane bounded
        in y or an ellipse, which have no meaning uniform in y, or a region whose
        ``kappa_y`` interval excludes zero, which would select nothing.
    """
    apertures = spec.apertures
    regions = spec.spectral_regions
    line = k_y.size == 1
    if line:
        if apertures is not None and np.any(np.asarray(apertures.centres)[:, 1] != 0):
            msg = (
                "on a line uniform in y an aperture is a slit along x, so its y centre "
                "must be zero"
            )
            raise ProcessingError(msg)
        if apertures is not None:
            _refuse_off_line(apertures)
        excluded = np.flatnonzero(~((regions[:, 2] <= 0) & (regions[:, 3] > 0)))
        if excluded.size:
            msg = (
                "on a line every column has kappa_y = 0, so regions "
                f"{excluded.tolist()}, whose kappa_y interval excludes zero, would "
                "select nothing"
            )
            raise ProcessingError(msg)
    if apertures is None:
        empty = np.empty(0)
        return ApertureTables(
            k_x=k_x,
            k_y=k_y,
            weights=weights,
            on_axis=on_axis,
            regions=regions,
            centres=np.empty((0, 2)),
            half_widths=empty,
            edges=empty,
            distances=empty,
            scale=np.empty((0, omega.size)),
        )
    outlines = _outlines(apertures, line=line)
    distances = np.asarray(apertures.distances)
    return ApertureTables(
        k_x=k_x,
        k_y=k_y,
        weights=weights,
        on_axis=on_axis,
        regions=regions,
        centres=np.asarray(apertures.centres),
        half_widths=np.array([one.half_width for one in outlines]),
        edges=np.array([one.edge for one in outlines]),
        distances=distances,
        scale=aperture_scale(distances, omega),
        profiles=tuple(one.kind for one in outlines),
        exponent=1 if apertures.weighting == "quadrature" else 2,
        shapes=tuple(one.shape for one in outlines),
        half_heights=np.array([one.half_height for one in outlines]),
        stretches=np.array([one.stretch for one in outlines]),
    )


def _refuse_off_line(apertures: ApertureSet) -> None:
    """Refuse the outlines a line uniform in y gives no meaning."""
    for row, shape in enumerate(apertures.row_shapes):
        if shape == "ellipse" or shape.startswith("half_plane_y"):
            msg = (
                f"on a line every column has kappa_y = 0 and the light is uniform in "
                f"y, so aperture {row}, a {shape}, has no meaning there; a line takes "
                "a circle or a rectangle, each a slit along x, or a half-plane bounded "
                "in x"
            )
            raise ProcessingError(msg)


def _outlines(apertures: ApertureSet, *, line: bool) -> list[ApertureOutline]:
    """Return each row's outline on a plane or a line."""
    count = apertures.n_apertures
    diameters = np.asarray(apertures.diameters)
    edges = np.zeros(count) if apertures.edges is None else np.asarray(apertures.edges)
    extents = (
        np.zeros((count, 2))
        if apertures.extents is None
        else np.asarray(apertures.extents)
    )
    corrected = apertures.weighting == "quadrature" and not line
    outlines = []
    for row in range(count):
        shape = apertures.row_shapes[row]
        diameter, extent = float(diameters[row]), extents[row]
        if line and shape == "rectangle":
            # sqrt(x*x) is |x| in correctly rounded arithmetic, so the slit decides a
            # hard edge as a circle of this diameter does (SEP 23 section 4.10).
            shape, diameter = "circle", float(extent[0])
        outlines.append(
            aperture_outline(
                shape,
                diameter,
                (float(extent[0]), float(extent[1])),
                float(edges[row]),
                apertures.row_profiles[row],
                corrected=corrected,
            )
        )
    return outlines
