# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Where solprop's manual explains a keyword (design 12.4).

solprop's metadata gives each keyword an anchor: a page under ``docs/`` with an
optional section. The front end opens it in the online manual of the installed
version's series; the bundled manual comes with the installers (milestone M8).
"""

from typing import Final

__all__ = ["MANUAL", "guide_url"]

MANUAL: Final = "https://solprop.readthedocs.io/en/latest/"
"""The online manual."""


def guide_url(anchor: str, manual: str = MANUAL) -> str:
    """Return the manual's address for a metadata anchor.

    Parameters
    ----------
    anchor : str
        A page under ``docs/``, as ``guide/setup/recipes.md#pins`` or without its
        ``.md``.
    manual : str, optional
        The manual's root address, ending in ``/``.

    Returns
    -------
    str
        The page's HTML address, with its section.

    Examples
    --------
    >>> guide_url("guide/setup/recipes.md#pins")
    'https://solprop.readthedocs.io/en/latest/guide/setup/recipes.html#pins'
    >>> guide_url("")
    'https://solprop.readthedocs.io/en/latest/'
    """
    page, _, section = anchor.partition("#")
    if not page:
        return manual
    page = page.removeprefix("docs/").removesuffix(".md").removesuffix(".html")
    address = f"{manual}{page}.html"
    return f"{address}#{section}" if section else address
