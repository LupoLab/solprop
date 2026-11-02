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

__all__ = ["CARDS", "RouteCard", "card", "card_of"]


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
        The ``[build]`` values that make a recipe this card's.
    reason : str or None
        Why it cannot be chosen yet; ``None`` when it can.

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
    reason: str | None = None

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
        for name, value in self.preset.items():
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


_LATER = "Comes with milestone {} of the front end."

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
        reason=_LATER.format("M2"),
    ),
    RouteCard(
        "gnlse",
        "Fibre (GNLSE)",
        "A single-mode fibre given by its dispersion and nonlinear coefficient.",
        "gnlse",
        reason=_LATER.format("M3"),
    ),
    RouteCard(
        "free",
        "Free space",
        "A beam in a bulk medium or gas, radially symmetric or in full.",
        "free",
        reason=_LATER.format("M5"),
    ),
    RouteCard(
        "planar",
        "Planar waveguide",
        "A slab waveguide, its modes across and free space along.",
        "planar",
        reason=_LATER.format("M3"),
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
        The first enabled card of that route whose preset values the recipe holds.

    Examples
    --------
    >>> card_of("capillary", {"mode_average": True}).key
    'capillary-averaged'
    >>> card_of("capillary", {"modes": 2}) is None
    True
    """
    for entry in CARDS:
        if (
            entry.enabled
            and entry.route == route
            and all(build.get(k) == v for k, v in entry.preset.items())
        ):
            return entry
    return None
