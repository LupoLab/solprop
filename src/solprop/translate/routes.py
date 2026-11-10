# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The closed union of route descriptions a propagation problem carries (SEP 47 6.2).

A :class:`~solprop.translate.problem.PropagationProblem` shares its grid, operator,
response and source components between every route; what differs between routes is
carried by exactly one of three variants, each holding the fields its executor needs and
nothing else:

- :class:`ModalRoute`: modal amplitudes through the static synthesis and projection
  of the transform -- the mode-averaged effective-area route and the resolved
  quadrature, a wall's coupled flow among them. The TDSE response is its
  reference-only extension (:class:`TDSESource`).
- :class:`GeneralRoute`: a general modal guide, whose transform carries the full E/H
  reference (``transform.modal_tables``) and whose static matrices are declared
  placeholders.
- :class:`SpatialRoute`: free space on a transverse grid of one of four kinds --
  radial, Cartesian, planar (the hybrid slab transform) or line (a Cartesian grid one
  sample wide in y) -- with the plane measure every spatial statistic is weighted by
  and the resolved transverse boundary.

Each variant checks its own fields on construction; what spans groups is checked by
:meth:`~solprop.translate.problem.PropagationProblem.validate`, which visits the variant
explicitly. A backend admits a problem the same way: it visits ``problem.route`` with a
``match`` statement and refuses, by name and before it allocates, a variant or a kind it
does not execute. Extending a route changes this typed contract and its tests in both
distributions; there is no string-keyed bag of optional capabilities
(``docs/dev/kernel-contract.md``, the extension procedure).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Float

from solprop.exceptions import TranslationError
from solprop.spatial_absorber import SpatialAbsorberDesign
from solprop.translate.modal import Transform
from solprop.translate.spatial import FluxPlan, SpatialGeometry
from solprop.translate.tdse import TDSEDescriptor

__all__ = [
    "SPATIAL_KINDS",
    "GeneralRoute",
    "ModalRoute",
    "Route",
    "SpatialKind",
    "SpatialRoute",
    "TDSESource",
    "flux_of",
    "route_name",
    "spatial_kind",
    "tdse_descriptor",
    "tdse_of",
    "transverse_of",
]

type SpatialKind = Literal["radial", "cartesian", "planar", "line"]
"""The four kinds of transverse grid a spatial route runs on."""

SPATIAL_KINDS: tuple[SpatialKind, ...] = ("radial", "cartesian", "planar", "line")
"""Every spatial kind, in the order the manual lists them."""


def spatial_kind(geometry: SpatialGeometry) -> SpatialKind:
    """Return the spatial kind a transverse geometry is.

    Parameters
    ----------
    geometry : solprop.translate.spatial.SpatialGeometry
        The transform's spatial leaf.

    Returns
    -------
    str
        ``"radial"`` for a Hankel geometry, ``"planar"`` for the hybrid slab transform,
        ``"line"`` for a Cartesian grid one sample wide in y (:class:`solprop.grid.
        LineGrid`) and ``"cartesian"`` for any other Cartesian grid.
    """
    if geometry.kind == "radial":
        return "radial"
    if geometry.kind == "slab":
        return "planar"
    return "line" if geometry.shape[1] == 1 else "cartesian"


@dataclass(frozen=True, eq=False)
class TDSESource:
    """The TDSE response: the modal route's reference-only extension (SEP 26).

    Only the NumPy reference executes it; every other backend refuses it by name
    (``docs/dev/kernel-contract.md`` section 6). The atomic binding itself is an
    execution resource, owned by the driver and handed to the backend explicitly; this
    record is its immutable scientific identity.

    Attributes
    ----------
    descriptor : solprop.translate.tdse.TDSEDescriptor
        The atomic specification, the resolved response lattice and the scientific
        identity.
    current_numerator : numpy.ndarray
        The independent current source numerator ``-crop_scale/(2 epsilon_0 c)``,
        float64, shape ``(n_modes, n_pol, n_omega)``, read-only and C-contiguous, zero
        outside the retained band and nowhere positive. A stage divides it by its real
        effective index without dividing by frequency.

    Raises
    ------
    TranslationError
        If the numerator is not a frozen, finite, non-positive float64 row.
    """

    descriptor: TDSEDescriptor
    current_numerator: Float[np.ndarray, "n_modes n_pol n_omega"]

    def __post_init__(self) -> None:
        """Check the record on construction (:meth:`validate`).

        Raises
        ------
        TranslationError
            As :meth:`validate` does.
        """
        self.validate()

    def validate(self) -> None:
        """Check the numerator is a frozen source row.

        Raises
        ------
        TranslationError
            If it is not.
        """
        values = self.current_numerator
        if (
            values.dtype != np.float64
            or values.flags.writeable
            or not values.flags.c_contiguous
            or not np.all(np.isfinite(values))
            or np.any(values > 0.0)
        ):
            raise TranslationError(
                "current numerator must be a frozen finite source row"
            )


@dataclass(frozen=True, eq=False)
class ModalRoute:
    """Modal amplitudes through the transform's static synthesis and projection.

    The mode-averaged effective-area route and the resolved quadrature, with or without
    a coupled flow (a capillary wall's). Its transform carries neither a spatial
    geometry nor general modal tables.

    Attributes
    ----------
    transform : solprop.translate.modal.Transform
        Synthesis and projection, of kind ``"mode_averaged"`` or ``"modal"``.
    tdse : TDSESource or None, optional
        The TDSE response, the reference's alone; ``None`` without one.

    Raises
    ------
    TranslationError
        If the transform is not a modal one.
    """

    transform: Transform
    tdse: TDSESource | None = None

    def __post_init__(self) -> None:
        """Check the record on construction (:meth:`validate`).

        Raises
        ------
        TranslationError
            As :meth:`validate` does.
        """
        self.validate()

    def validate(self) -> None:
        """Refuse a transform that belongs to another route.

        Raises
        ------
        TranslationError
            Naming the transform's kind.
        """
        if self.transform.kind not in ("mode_averaged", "modal") or (
            self.transform.modal_tables is not None
        ):
            kind = self.transform.kind
            tables = " with general modal tables" if self.transform.modal_tables else ""
            msg = f"a modal route carries a modal transform, not a {kind!r} one{tables}"
            raise TranslationError(msg)
        if self.tdse is not None:
            self.tdse.validate()


@dataclass(frozen=True, eq=False)
class GeneralRoute:
    """A general modal guide, its transform carrying the full E/H reference.

    The route of :class:`solprop.translate.general.GeneralModalRoute`, the record a run
    is assembled from. Its transform's static matrices are declared placeholders
    (``docs/dev/kernel-contract.md`` section 4).

    Attributes
    ----------
    transform : solprop.translate.modal.Transform
        A modal transform with ``modal_tables``.

    Raises
    ------
    TranslationError
        If the transform carries no general modal tables.
    """

    transform: Transform

    def __post_init__(self) -> None:
        """Check the record on construction (:meth:`validate`).

        Raises
        ------
        TranslationError
            As :meth:`validate` does.
        """
        self.validate()

    def validate(self) -> None:
        """Refuse a transform without the general modal tables.

        Raises
        ------
        TranslationError
            If ``transform.modal_tables`` is absent.
        """
        if self.transform.modal_tables is None:
            raise TranslationError(
                "a general modal route carries transform.modal_tables, and this "
                "transform has none"
            )


@dataclass(frozen=True, eq=False)
class SpatialRoute:
    """Free space on a transverse grid of one of four kinds.

    Attributes
    ----------
    kind : {'radial', 'cartesian', 'planar', 'line'}
        The transverse grid: Hankel, two-dimensional Cartesian, the hybrid planar slab
        transform, or a Cartesian grid one sample wide in y. It equals
        :func:`spatial_kind` of the transform's geometry.
    transform : solprop.translate.modal.Transform
        A transform with a spatial ``geometry``.
    flux : solprop.translate.spatial.FluxPlan
        The plane measure every spatial statistic is weighted by (SEP 34).
    transverse : solprop.spatial_absorber.SpatialAbsorberDesign or None, optional
        The resolved transverse boundary (SEP 40); ``None`` for none.

    Raises
    ------
    TranslationError
        If the transform has no geometry or the kind is not the geometry's.
    """

    kind: SpatialKind
    transform: Transform
    flux: FluxPlan
    transverse: SpatialAbsorberDesign | None = None

    def __post_init__(self) -> None:
        """Check the record on construction (:meth:`validate`).

        Raises
        ------
        TranslationError
            As :meth:`validate` does.
        """
        self.validate()

    def validate(self) -> None:
        """Check the kind against the transform's geometry.

        Raises
        ------
        TranslationError
            Naming the disagreement.
        """
        geometry = self.transform.geometry
        if geometry is None:
            raise TranslationError(
                "a spatial route carries a transform with a spatial geometry"
            )
        if not isinstance(
            self.flux, FluxPlan
        ):  # pragma: no cover - refused by the annotation where it is checked
            raise TranslationError(
                "a spatial route requires its plane measure (flux), a FluxPlan"
            )
        actual = spatial_kind(geometry)
        if self.kind != actual:
            msg = f"a {self.kind!r} spatial route carries a {actual!r} geometry"
            raise TranslationError(msg)


type Route = ModalRoute | GeneralRoute | SpatialRoute
"""The closed union of route descriptions."""


def route_name(route: object) -> str:
    """Return the name a refusal gives a route.

    Parameters
    ----------
    route : object
        The route: a variant of ``Route``, or anything else a refusal names.

    Returns
    -------
    str
        ``"modal"``, ``"general modal"`` or ``"<kind> spatial"``, with ``" with TDSE"``
        for the modal route's extension; any other object's type name.
    """
    match route:
        case ModalRoute(tdse=None):
            return "modal"
        case ModalRoute():
            return "modal with TDSE"
        case GeneralRoute():
            return "general modal"
        case SpatialRoute(kind=kind):
            return f"{kind} spatial"
        case _:
            # A variant this release does not know, named by its type for a refusal.
            return type(route).__name__


def tdse_of(route: Route) -> TDSESource | None:
    """Return the modal route's TDSE extension, or ``None`` on every other route.

    Parameters
    ----------
    route : Route
        The route.

    Returns
    -------
    TDSESource or None
        The extension, when the route is a modal one carrying it.
    """
    match route:
        case ModalRoute(tdse=source):
            return source
        case _:
            return None


def flux_of(route: Route) -> FluxPlan | None:
    """Return a spatial route's plane measure, or ``None`` on a modal route.

    Parameters
    ----------
    route : Route
        The route.

    Returns
    -------
    solprop.translate.spatial.FluxPlan or None
        The plane measure, which every spatial route carries.
    """
    match route:
        case SpatialRoute(flux=flux):
            return flux
        case _:
            return None


def transverse_of(route: Route) -> SpatialAbsorberDesign | None:
    """Return a spatial route's transverse boundary, or ``None``.

    Parameters
    ----------
    route : Route
        The route.

    Returns
    -------
    solprop.spatial_absorber.SpatialAbsorberDesign or None
        The resolved design, ``None`` on a modal route or without a boundary.
    """
    match route:
        case SpatialRoute(transverse=design):
            return design
        case _:
            return None


def tdse_descriptor(route: Route) -> TDSEDescriptor | None:
    """Return the TDSE extension's scientific identity, or ``None`` without one.

    Parameters
    ----------
    route : Route
        The route.

    Returns
    -------
    solprop.translate.tdse.TDSEDescriptor or None
        The descriptor of the modal route's TDSE extension.
    """
    source = tdse_of(route)
    return None if source is None else source.descriptor
