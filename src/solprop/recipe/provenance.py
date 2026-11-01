# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The recipe a run records: the authored text, the resolved one, and the data files.

A run built from a recipe writes a ``recipe`` group in its output, once, at its first
start (SEP 41, S2). The **authored** recipe is the text as loaded, or its canonical
TOML once edited. The **resolved** recipe is what rebuilds the run: the authored keys,
the noise seed that was drawn, each keyword the metadata marks resolvable at the value
the build used, and ``[execution]`` from the settings at run start. A keyword is
resolvable only where its value written back gives the same record; the rest stay
with the build, so a later solprop's default cannot move a resolvable one, and the
resume gate notices the others.
"""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from solprop.nonlinear.raman import RamanResponse
from solprop.output.schema import RecipeFile, RecipeRecord
from solprop.recipe.document import (
    FORMAT,
    Document,
    decode_data_file,
    dump_text,
    encode_integer,
    read_data_file,
)
from solprop.recipe.metadata import REQUIRED, parameters

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation
    from solprop.recipe.core import Recipe
    from solprop.stats import StatsSpec

__all__ = ["EMBED_LIMIT", "record", "resolved_document"]

EMBED_LIMIT: Final = 16 * 2**20
"""Bytes of data files a run's recipe embeds; beyond, each is kept by path and hash."""

_PLAIN: Final = (bool, int, float, str)
"""The scalar values a default written back may be."""


def _stats_table(stats: "StatsSpec") -> dict[str, object] | None:  # noqa: UP037
    """Return a statistics selection as its registered table, if it has one.

    Parameters
    ----------
    stats : solprop.stats.StatsSpec
        The selection.

    Returns
    -------
    dict or None
        The table, or ``None`` for a selection with apertures, which have no table.
    """
    if stats.apertures is not None:
        return None
    table: dict[str, object] = {"type": "StatsSpec", "names": list(stats.names)}
    windows = np.asarray(stats.energy_windows)
    if windows.size:
        table["energy_windows"] = windows.tolist()
    if stats.periodic:
        table["periodic"] = list(stats.periodic)
        table["period"] = int(stats.period)
    regions = np.asarray(stats.spectral_regions)
    if regions.size:
        table["spectral_regions"] = regions.tolist()
    return table


def _used(
    recipe: "Recipe",  # noqa: UP037 - checked at run time
    simulation: "Simulation",  # noqa: UP037 - checked at run time
) -> dict[str, object]:
    """Return the value the build used for each resolvable keyword it can name.

    Parameters
    ----------
    recipe : Recipe
        The recipe.
    simulation : Simulation
        What it built, with its settings at run start.

    Returns
    -------
    dict
        Values by keyword, as a recipe writes them.
    """
    settings = simulation.settings
    values: dict[str, object] = {}
    controller = settings.controller
    if controller is not None:
        values["rtol"] = controller.rtol
        values["atol"] = controller.atol
    values["dz_max"] = settings.dz_max
    integrator = settings.integrator
    if integrator is not None and integrator == type(integrator)():
        values["integrator"] = {"type": type(integrator).__name__}
    if settings.stats is not None and (table := _stats_table(settings.stats)):
        values["stats"] = table
    values["step_planes"] = int(settings.step_planes)
    planes = np.asarray(settings.save_planes, dtype=float)
    values["save_planes"] = planes.tolist()
    capillary = simulation.configuration.get("capillary")
    if isinstance(capillary, Mapping):
        temperature = capillary.get("temperature")
        if isinstance(temperature, float):
            values["temperature"] = temperature
        values["raman"] = any(
            isinstance(response, RamanResponse) for response in simulation.responses
        )
    build = recipe.to_dict()["build"]
    for name, entry in parameters(recipe.route).items():
        if not entry.resolvable or name in values:
            continue
        if entry.resolver is not None:
            resolved = entry.resolver(build).value
            if isinstance(resolved, _PLAIN):
                values[name] = resolved
        elif entry.default is not REQUIRED and isinstance(entry.default, _PLAIN):
            values[name] = entry.default
    return values


def resolved_document(
    recipe: "Recipe",  # noqa: UP037 - checked at run time
    simulation: "Simulation",  # noqa: UP037 - checked at run time
) -> Document:
    """Return the recipe that rebuilds a run, as its tables.

    Parameters
    ----------
    recipe : Recipe
        The recipe the run was built from.
    simulation : Simulation
        What it built, translated, with its settings at run start.

    Returns
    -------
    dict
        The authored tables, with the drawn seed, each resolvable keyword the recipe
        left out at the value used, and ``[execution]`` from the settings.
    """
    document = recipe.to_dict()
    document["recipe"]["format"] = FORMAT
    build = document["build"]
    table = parameters(recipe.route)
    for name, value in _used(recipe, simulation).items():
        if table[name].resolvable and name not in build:
            build[name] = value
    if "save_planes" in build:
        # The planes n_save stood for are now written out.
        build.pop("n_save", None)
    noise = simulation.noise
    if noise is not None and noise.seed is not None and "seed" not in build:
        build["seed"] = encode_integer(int(noise.seed))
    execution = dict(document.get("execution", {}))
    settings = simulation.settings
    execution["fields"] = settings.fields
    execution["field_planes"] = list(settings.field_planes)
    if settings.resume_history is not None:
        execution["resume_history"] = settings.resume_history
    document["execution"] = execution
    return document


def _references(value: object) -> Iterator[Mapping[str, object]]:
    """Yield every data-file reference inside a recipe value.

    Parameters
    ----------
    value : object
        A value as the recipe writes it.

    Yields
    ------
    collections.abc.Mapping
        Each table holding ``file`` and ``sha256``.
    """
    if isinstance(value, Mapping):
        if "file" in value and "sha256" in value:
            yield value
        for item in value.values():
            yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _files(
    recipe: "Recipe",  # noqa: UP037 - checked at run time
    base: Path | None,
    *,
    allow_outside: bool,
) -> dict[str, RecipeFile]:
    """Return the data files a recipe refers to, embedded up to :data:`EMBED_LIMIT`.

    Parameters
    ----------
    recipe : Recipe
        The recipe.
    base : pathlib.Path or None
        Its directory.
    allow_outside : bool
        Whether a file may lie outside it.

    Returns
    -------
    dict
        Each file by a generated name, with its contents while they fit.
    """
    files: dict[str, RecipeFile] = {}
    total = 0
    for index, reference in enumerate(_references(recipe.to_dict()["build"])):
        data_file = decode_data_file(
            {"file": reference["file"], "sha256": reference["sha256"]}
        )
        data: bytes | None = None
        if base is not None:
            contents = read_data_file(data_file, base, allow_outside=allow_outside)
            if total + len(contents) <= EMBED_LIMIT:
                data = contents
                total += len(contents)
        name = f"file-{index}{Path(data_file.file).suffix}"
        files[name] = RecipeFile(data_file.file, data_file.sha256, data)
    return files


def record(
    recipe: "Recipe",  # noqa: UP037 - checked at run time
    simulation: "Simulation",  # noqa: UP037 - checked at run time
    *,
    base: Path | None = None,
    allow_outside: bool = False,
) -> RecipeRecord:
    """Return the record a run built from a recipe keeps in its output.

    Parameters
    ----------
    recipe : Recipe
        The recipe.
    simulation : Simulation
        What it built, translated, with its settings at run start.
    base : pathlib.Path or None, optional
        The recipe's directory, from which its data files are read.
    allow_outside : bool, optional
        Whether a data file may lie outside it.

    Returns
    -------
    solprop.output.schema.RecipeRecord
        The authored and resolved texts, and the data files.
    """
    # A pinned recipe builds through its resolved copy, which remembers the authored
    # text with its pins (Recipe.resolve).
    authored = recipe._authored or recipe
    return RecipeRecord(
        FORMAT,
        authored.to_text(),
        dump_text(resolved_document(recipe, simulation)),
        _files(recipe, base, allow_outside=allow_outside),
    )
