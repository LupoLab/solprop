# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The routes the geometry page offers, as cards (design 6.1).

A card is presentation: a title, a sentence, the solprop route it builds and the
values that make it that card (the mode-averaged capillary is the capillary route with
``mode_average = true``; a line beam is the free route with a ``LineGrid`` space). A new
setup from a card starts from solprop's example of it, or holds the card's values and
nothing else; what else the route requires, solprop's check says. A card the window
does not offer yet is hidden, its code kept.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

__all__ = [
    "CARDS",
    "EXPLORER_SETUP",
    "GUIDED",
    "IMPLIED",
    "RouteCard",
    "card",
    "card_of",
    "group_examples",
    "type_of",
]


@dataclass(frozen=True)
class RouteCard:
    """One choice on the geometry page.

    Attributes
    ----------
    key : str
        Its identity.
    title : str
        What it is called.
    summary : str
        One sentence.
    route : str
        The solprop route it builds.
    preset : dict
        The ``[build]`` values that make a recipe this card's; a value equal to
        solprop's default (:data:`IMPLIED`) may also be left out of a recipe.
    extra : dict
        Values a new setup from the card starts with beyond its preset, which do not
        make a recipe the card's.
    reason : str or None
        Why it cannot be chosen yet; ``None`` when it can.
    example : str or None
        The solprop example a new setup from the card starts from
        (``Recipe.example``), in place of a blank recipe of the preset.
    marker : tuple of str
        Keywords whose presence makes a recipe the card's, beside its preset (an
        antiresonant fibre is a capillary with a ``lattice``).
    types : dict
        Keywords whose registered type makes a recipe the card's, by keyword, each
        with the type names that do (a line beam's ``space`` is a ``LineGrid``); for
        a list, its first entry's type.
    rewrite : dict
        Changes made to :attr:`example`'s ``[build]`` for a new setup, where solprop
        has no example of the card's own: a value replaces the keyword's, ``None``
        removes it.
    group : str
        The heading the gallery shows it under.
    hidden : bool
        Whether the gallery and the examples leave it out.

    Examples
    --------
    >>> card("capillary-averaged").enabled
    True
    """

    key: str
    title: str
    summary: str
    route: str
    preset: dict[str, object] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict)
    reason: str | None = None
    example: str | None = None
    marker: tuple[str, ...] = ()
    types: dict[str, tuple[str, ...]] = field(default_factory=dict)
    rewrite: dict[str, object] = field(default_factory=dict)
    group: str = "Guided"
    hidden: bool = False

    @property
    def enabled(self) -> bool:
        """Return whether the card can be chosen.

        Returns
        -------
        bool
            True when it has no reason against it.
        """
        return self.reason is None

    def blank(self, title: str | None = None) -> str:
        """Return a new recipe for this card, holding its preset values only.

        Parameters
        ----------
        title : str or None, optional
            The recipe's title.

        Returns
        -------
        str
            The recipe's TOML text.
        """
        lines = ["[recipe]", "format = 1", f'route = "{self.route}"']
        if title is not None:
            escaped = title.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'title = "{escaped}"')
        lines.append("")
        lines.append("[build]")
        for name, value in {**self.preset, **self.extra}.items():
            if name in IMPLIED and IMPLIED[name] == value:
                continue  # solprop's default: left to solprop
            lines.append(f"{name} = {_toml(value)}")
        return "\n".join(lines) + "\n"


def _toml(value: object) -> str:
    """Write a preset value as TOML.

    Parameters
    ----------
    value : object
        A boolean, an integer or a string.

    Returns
    -------
    str
        Its TOML.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


IMPLIED: Final = {"mode_average": False}
"""solprop's defaults of the keywords the cards' presets name; a test holds them to
solprop's metadata."""

CARDS: Final = (
    RouteCard(
        "capillary-averaged",
        "Capillary, mode-averaged",
        "A gas-filled hollow capillary, one mode with its nonlinearity averaged over "
        "the profile: the quickest model of a capillary.",
        "capillary",
        {"mode_average": True},
        group="Hollow capillaries and fibres",
    ),
    RouteCard(
        "capillary-modal",
        "Capillary, multimode",
        "The capillary's modes coupled through the nonlinearity: the first HE1m "
        "modes by number, or any HE, TE and TM modes listed.",
        "capillary",
        {"mode_average": False},
        {"modes": 4},
        group="Hollow capillaries and fibres",
    ),
    RouteCard(
        "antiresonant",
        "Antiresonant fibre",
        "A hollow core bounded by a ring of thin-walled tubes, gas-filled: the "
        "capillary's modes with the lattice's loss and dispersion.",
        "capillary",
        example="antiresonant_fibre",
        marker=("lattice",),
        group="Hollow capillaries and fibres",
    ),
    RouteCard(
        "gnlse",
        "Fibre (GNLSE)",
        "A single-mode fibre given by its dispersion and nonlinear coefficient.",
        "gnlse",
        example="soliton",
        group="Hollow capillaries and fibres",
    ),
    RouteCard(
        "rectangular",
        "Rectangular waveguide",
        "A gas-filled hollow channel of rectangular section, in its modes.",
        "waveguide",
        example="hollow_rectangular",
        types={"modes": ("RectMode",)},
        group="Hollow waveguides",
    ),
    RouteCard(
        "slab",
        "Slab waveguide",
        "A gas-filled gap between two plates, in its slab modes: uniform along the "
        "plates, so energies and powers are per unit width.",
        "waveguide",
        example="hollow_rectangular",
        types={"modes": ("SlabMode", "ModeCollection.slab")},
        rewrite={
            "modes": [{"type": "SlabMode", "half_gap": "50 um"}],
            "energy": "0.5 J",
            "n_theta": None,
            "seed": None,
        },
        group="Hollow waveguides",
    ),
    RouteCard(
        "cartesian",
        "Free space, Cartesian",
        "A beam in a bulk medium or gas on a transverse x-y grid.",
        "free",
        example="cartesian_beam",
        types={"space": ("FreeGrid",)},
        group="Free space",
    ),
    RouteCard(
        "line",
        "Free space, line",
        "A beam uniform in y on a line of x nodes: energies and powers are per unit "
        "width.",
        "free",
        example="cartesian_beam",
        types={"space": ("LineGrid",)},
        rewrite={"space": {"type": "LineGrid", "radius_x": "6 mm", "n_x": 64}},
        group="Free space",
    ),
    RouteCard(
        "radial",
        "Free space, radial",
        "A radially symmetric beam in a bulk medium or gas, on a Hankel grid.",
        "radial",
        example="radial_kerr",
        group="Free space",
    ),
    RouteCard(
        "planar",
        "Planar waveguide",
        "A slab waveguide, its modes across and free space along.",
        "planar",
        reason="Planar guides are not offered for now.",
        example="planar_hybrid",
        hidden=True,
    ),
)
"""The cards, in the order the page shows them."""


def card(key: str) -> RouteCard:
    """Return a card by its key.

    Parameters
    ----------
    key : str
        The card's key.

    Returns
    -------
    RouteCard
        The card.

    Raises
    ------
    KeyError
        If no card has that key.
    """
    for entry in CARDS:
        if entry.key == key:
            return entry
    raise KeyError(key)


def card_of(
    route: str, build: dict[str, object], *, enabled_only: bool = True
) -> RouteCard | None:
    """Return the enabled card a recipe is, if any.

    Parameters
    ----------
    route : str
        The recipe's route.
    build : dict
        Its ``[build]`` table.
    enabled_only : bool, optional
        Whether only an enabled card counts (the default); ``False`` names the
        class of an example on a route the window does not offer yet.

    Returns
    -------
    RouteCard or None
        The enabled card of that route whose preset values, marker keywords and
        keyword types the recipe holds, the cards with markers or types tried first.

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
            (entry.enabled or not enabled_only)
            and entry.route == route
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


GUIDED: Final = frozenset({"capillary", "waveguide", "planar", "gnlse"})
"""The routes with a mode set for the explorer to show (``Recipe.modes``)."""

EXPLORER_SETUP: Final = """\
[recipe]
format = 1
route = "capillary"
title = "Waveguide explorer"

[build]
radius = "125 um"
length = "1 m"
gas = "He"
pressure = "1 bar"
modes = 4
wavelength = "800 nm"
fwhm = "10 fs"
energy = "100 uJ"
wavelength_limits = ["150 nm", "4 um"]
trange = "400 fs"
"""
"""The setup the standalone waveguide explorer opens with: a helium-filled capillary
and its first four modes, a starting point to change and sweep."""


def group_examples(
    examples: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    """Sort solprop's examples into the geometry classes the cards name.

    Parameters
    ----------
    examples : Sequence of Mapping
        Each example's ``name``, ``title``, ``route`` and ``build`` table.

    Returns
    -------
    dict
        By card key, in the cards' order, the examples it recognises as its own
        (:func:`card_of`, disabled cards included); ``"other"`` for the rest. A
        hidden card's examples are left out.

    Examples
    --------
    >>> grouped = group_examples([
    ...     {"name": "a", "route": "capillary", "build": {"mode_average": True}},
    ...     {"name": "b", "route": "gnlse", "build": {}},
    ...     {"name": "c", "route": "nope", "build": {}},
    ...     {"name": "d", "route": "planar", "build": {}},
    ... ])
    >>> {key: [e["name"] for e in found] for key, found in grouped.items()}
    {'capillary-averaged': ['a'], 'gnlse': ['b'], 'other': ['c']}
    """
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for example in examples:
        given = example.get("build")
        build: dict[str, object] = dict(given) if isinstance(given, Mapping) else {}
        entry = card_of(str(example["route"]), build, enabled_only=False)
        if entry is not None and entry.hidden:
            continue
        grouped.setdefault("other" if entry is None else entry.key, []).append(example)
    order = [card.key for card in CARDS] + ["other"]
    return {key: grouped[key] for key in order if key in grouped}
