# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The geometries solprop's routes build, as cards that name them (design 6.1).

A card is presentation: a title, the solprop route it builds and the values that make
a recipe that card's (the mode-averaged capillary is the capillary route with
``mode_average = true``; a line beam is the free route with a ``LineGrid`` space). A
recipe on a route no problem class takes is named by its card (:func:`card_of`).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

__all__ = ["CARDS", "IMPLIED", "RouteCard", "card_of", "type_of"]


@dataclass(frozen=True)
class RouteCard:
    """One geometry a route builds.

    Attributes
    ----------
    key : str
        Its identity.
    title : str
        What it is called.
    route : str
        The solprop route it builds.
    preset : dict
        The ``[build]`` values that make a recipe this card's; a value equal to
        solprop's default (:data:`IMPLIED`) may also be left out of a recipe.
    example : str or None
        solprop's example of it (``Recipe.example``), or of the example
        :attr:`rewrite` changes into it.
    marker : tuple of str
        Keywords whose presence makes a recipe the card's, beside its preset (an
        antiresonant fibre is a capillary with a ``lattice``).
    types : dict
        Keywords whose registered type makes a recipe the card's, by keyword, each
        with the type names that do (a line beam's ``space`` is a ``LineGrid``); for
        a list, its first entry's type.
    rewrite : dict
        Changes that make :attr:`example`'s ``[build]`` the card's, where solprop has
        no example of the card's own: a value replaces the keyword's, ``None``
        removes it.

    Examples
    --------
    >>> RouteCard("gnlse", "Fibre (GNLSE)", "gnlse").preset
    {}
    """

    key: str
    title: str
    route: str
    preset: dict[str, object] = field(default_factory=dict)
    example: str | None = None
    marker: tuple[str, ...] = ()
    types: dict[str, tuple[str, ...]] = field(default_factory=dict)
    rewrite: dict[str, object] = field(default_factory=dict)


IMPLIED: Final = {"mode_average": False}
"""solprop's defaults of the keywords the cards' presets name; a test holds them to
solprop's metadata."""

CARDS: Final = (
    RouteCard("gnlse", "Fibre (GNLSE)", "gnlse", example="soliton"),
    RouteCard(
        "step-index", "Step-index fibre", "gnlse", types={"fibre": ("StepIndexMode",)}
    ),
    RouteCard(
        "grin",
        "Graded-index fibre",
        "gnlse",
        types={"fibre": ("GradedIndexProvider",)},
    ),
    RouteCard(
        "capillary-averaged",
        "Capillary, mode-averaged",
        "capillary",
        {"mode_average": True},
    ),
    RouteCard(
        "capillary-modal", "Capillary, multimode", "capillary", {"mode_average": False}
    ),
    RouteCard(
        "antiresonant",
        "Antiresonant fibre",
        "capillary",
        example="antiresonant_fibre",
        marker=("lattice",),
    ),
    RouteCard(
        "cartesian",
        "Free space, Cartesian",
        "free",
        example="cartesian_beam",
        types={"space": ("FreeGrid",)},
    ),
    RouteCard(
        "line",
        "Free space, line",
        "free",
        example="cartesian_beam",
        types={"space": ("LineGrid",)},
        rewrite={"space": {"type": "LineGrid", "radius_x": "6 mm", "n_x": 64}},
    ),
    RouteCard("radial", "Free space, radial", "radial", example="radial_kerr"),
    RouteCard(
        "rectangular",
        "Rectangular waveguide",
        "waveguide",
        example="hollow_rectangular",
        types={"modes": ("RectMode",)},
    ),
    RouteCard(
        "slab",
        "Slab waveguide",
        "waveguide",
        example="hollow_rectangular",
        types={"modes": ("SlabMode", "ModeCollection.slab")},
        rewrite={
            "modes": [{"type": "SlabMode", "half_gap": "50 um"}],
            "energy": "0.5 J",
            "n_theta": None,
            "seed": None,
        },
    ),
    RouteCard("planar", "Planar waveguide", "planar", example="planar_hybrid"),
)
"""The cards."""


def card_of(route: str, build: dict[str, object]) -> RouteCard | None:
    """Return the card a recipe is, if any.

    Parameters
    ----------
    route : str
        The recipe's route.
    build : dict
        Its ``[build]`` table.

    Returns
    -------
    RouteCard or None
        The card of that route whose preset values, marker keywords and keyword
        types the recipe holds, the cards with markers or types tried first.

    Examples
    --------
    >>> card_of("capillary", {"mode_average": True}).key
    'capillary-averaged'
    >>> card_of("capillary", {"modes": 2}).key
    'capillary-modal'
    >>> card_of("capillary", {"lattice": {"type": "TubeLattice"}}).key
    'antiresonant'
    >>> card_of("free", {"space": {"type": "LineGrid"}}).key
    'line'
    >>> card_of("waveguide", {"modes": [{"type": "SlabMode"}]}).key
    'slab'
    >>> card_of("waveguide", {"modes": {"type": "ModeCollection.slab"}}).key
    'slab'
    >>> card_of("radial", {}).key
    'radial'
    >>> card_of("nope", {}) is None
    True
    """
    for entry in sorted(CARDS, key=lambda c: -len(c.marker) - len(c.types)):
        if (
            entry.route == route
            and all(build.get(k, IMPLIED.get(k)) == v for k, v in entry.preset.items())
            and all(k in build for k in entry.marker)
            and all(type_of(build.get(k)) in v for k, v in entry.types.items())
        ):
            return entry
    return None


def type_of(value: object) -> str | None:
    """Return the registered type a keyword's value names.

    Parameters
    ----------
    value : object
        The value as a recipe gives it: a table with ``type``, or a list of them.

    Returns
    -------
    str or None
        The table's ``type``, or the first entry's; ``None`` for another value.

    Examples
    --------
    >>> type_of({"type": "FreeGrid"}), type_of([{"type": "RectMode"}]), type_of(3)
    ('FreeGrid', 'RectMode', None)
    """
    if isinstance(value, list | tuple) and value:
        value = value[0]
    if isinstance(value, Mapping):
        found = value.get("type")
        return None if found is None else str(found)
    return None
