# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The routes the geometry page offers, as cards (design 6.1).

A card is presentation: a title, a sentence, the solprop route it builds and the
values that make it that card (the mode-averaged capillary is the capillary route with
``mode_average = true``). A new setup from a card holds those values and nothing else;
what else the route requires, solprop's check says. A card whose milestone has not
come yet is shown disabled, with the reason.
"""

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
    ),
    RouteCard(
        "capillary-modal",
        "Capillary, multimode",
        "The capillary's HE1m modes, coupled through the nonlinearity.",
        "capillary",
        {"mode_average": False},
        {"modes": 4},
    ),
    RouteCard(
        "antiresonant",
        "Antiresonant fibre",
        "A hollow core bounded by a ring of thin-walled tubes, gas-filled: the "
        "capillary's modes with the lattice's loss and dispersion.",
        "capillary",
        example="antiresonant_fibre",
        marker=("lattice",),
    ),
    RouteCard(
        "waveguide",
        "Hollow waveguide, supplied modes",
        "A gas-filled guide given by its modes: a rectangular channel, a planar guide "
        "per unit width (slab modes), or capillary modes of your choosing.",
        "waveguide",
        example="hollow_rectangular",
    ),
    RouteCard(
        "gnlse",
        "Fibre (GNLSE)",
        "A single-mode fibre given by its dispersion and nonlinear coefficient.",
        "gnlse",
        example="soliton",
    ),
    RouteCard(
        "free",
        "Free space, Cartesian",
        "A beam in a bulk medium or gas on a transverse x-y grid (or a line in x).",
        "free",
        example="cartesian_beam",
    ),
    RouteCard(
        "radial",
        "Free space, radial",
        "A radially symmetric beam in a bulk medium or gas, on a Hankel grid.",
        "radial",
        reason="Needs a radial example recipe in solprop to start from (#405).",
    ),
    RouteCard(
        "planar",
        "Planar waveguide",
        "A slab waveguide, its modes across and free space along.",
        "planar",
        example="planar_hybrid",
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


def card_of(route: str, build: dict[str, object]) -> RouteCard | None:
    """Return the enabled card a recipe is, if any.

    Parameters
    ----------
    route : str
        The recipe's route.
    build : dict
        Its ``[build]`` table.

    Returns
    -------
    RouteCard or None
        The enabled card of that route whose preset values and marker keywords the
        recipe holds, the cards with markers tried first.

    Examples
    --------
    >>> card_of("capillary", {"mode_average": True}).key
    'capillary-averaged'
    >>> card_of("capillary", {"modes": 2}).key
    'capillary-modal'
    >>> card_of("capillary", {"lattice": {"type": "TubeLattice"}}).key
    'antiresonant'
    >>> card_of("radial", {}) is None
    True
    """
    for entry in sorted(CARDS, key=lambda c: -len(c.marker)):
        if (
            entry.enabled
            and entry.route == route
            and all(build.get(k, IMPLIED.get(k)) == v for k, v in entry.preset.items())
            and all(k in build for k in entry.marker)
        ):
            return entry
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
