# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Recipes: a simulation written as a TOML document, built into a ``Simulation``.

A recipe holds the keywords of one route's constructor, with quantities written in
their units (``"125 um"``, ``"1.2 bar"``), and the run's execution settings. It is
loaded, checked against the parameter metadata, built, run, exported as a Python script
and saved. The format is SEP 41's; the manual page is ``docs/guide/setup/recipes.md``.

>>> from solprop.recipe import Recipe
>>> recipe = Recipe.example("rdw")
>>> recipe.route
'capillary'
"""

import importlib
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from solprop.recipe.core import Recipe, RecipeIssue, RecipeScan, resume_file
    from solprop.recipe.errors import RecipeError
    from solprop.recipe.metadata import Form, Parameter, Resolution, parameters

__all__ = [
    "Form",
    "Parameter",
    "Recipe",
    "RecipeError",
    "RecipeIssue",
    "RecipeScan",
    "Resolution",
    "parameters",
    "resume_file",
]

_HOMES: Final = {
    "Form": "metadata",
    "Parameter": "metadata",
    "Recipe": "core",
    "RecipeError": "errors",
    "RecipeIssue": "core",
    "RecipeScan": "core",
    "Resolution": "metadata",
    "parameters": "metadata",
    "resume_file": "core",
}
"""The submodule each name of ``__all__`` is defined in, imported on first access."""

_SUBMODULES: Final = (
    "core",
    "document",
    "errors",
    "export",
    "metadata",
    "paths",
    "pins",
    "preview",
    "provenance",
    "quantity",
    "registry",
)
"""The submodules reachable as attributes; a test keeps it equal to what is on disk."""


def __getattr__(name: str) -> object:
    """Import a name or submodule the first time it is asked for (:pep:`562`).

    The package's names are imported on first use so that its light modules --
    ``quantity``, ``paths``, ``document`` and ``errors`` -- import without the driver,
    the backends or h5py: a front end parses and writes quantities with them in a
    process that never runs a simulation.

    Parameters
    ----------
    name : str
        The attribute looked up on the package.

    Returns
    -------
    object
        The name from the submodule that defines it, or the submodule itself.

    Raises
    ------
    AttributeError
        If `name` is neither a public name nor a submodule of the package.
    """
    if name in _HOMES:
        module = importlib.import_module(f"{__name__}.{_HOMES[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """List the package's public surface, whether or not it has been loaded yet.

    Returns
    -------
    list of str
        The sorted names ``dir`` and tab completion show: ``__all__``, the submodules
        and the dunders.
    """
    names = {*globals(), *__all__, *_SUBMODULES}
    return sorted(
        name
        for name in names
        if name.startswith("__") or name in __all__ or name in _SUBMODULES
    )
