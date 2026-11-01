# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A recipe written out as the plain Python script a person would write (SEP 41, S1).

The script is the way out of a recipe for whatever a recipe cannot hold: a custom
pulse, a callable, an unregistered type. It makes the same run: the same simulation,
hash included, and the same execution settings, which the hash does not cover.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from solprop.recipe.document import NONE
from solprop.recipe.errors import RecipeError
from solprop.recipe.registry import (
    REGISTRY,
    BuildContext,
    UnitSpec,
    decode,
    decode_keyword,
    unit_of,
)

if TYPE_CHECKING:
    from solprop.recipe.core import Recipe

__all__ = ["script"]

_INDENT: Final = "    "


class _Writer:
    """Render recipe values as Python expressions, and collect what they import."""

    def __init__(self, recipe: "Recipe") -> None:  # noqa: UP037 - run time
        """Start with no imports."""
        self.recipe = recipe
        self.imports: dict[str, set[str]] = {}
        # The name the script binds the route's grid to, when the recipe gives one.
        self.grid: str | None = None

    def need(self, module: str, name: str) -> None:
        """Record an import.

        Parameters
        ----------
        module : str
            The module.
        name : str
            The name it provides.
        """
        self.imports.setdefault(module, set()).add(name)

    def expression(  # noqa: PLR0911 - one return per kind of value
        self, value: object, unit: UnitSpec, path: str
    ) -> str:
        """Return a value as a Python expression.

        Parameters
        ----------
        value : object
            The value as the recipe writes it.
        unit : str, callable or None
            Its unit.
        path : str
            Its parameter path.

        Returns
        -------
        str
            The expression, numbers in SI.

        Raises
        ------
        RecipeError
            For a value built on the route's grid, which the script cannot name
            before the route has built it.
        """
        if isinstance(value, Mapping):
            if "type" in value:
                return self.factory(value, path)
            if set(value) == {"file", "sha256"}:
                self.need("numpy", "")
                return f"numpy.load({value['file']!r}, allow_pickle=False)"
            if set(value) == {"re", "im"}:
                return repr(complex(float(value["re"]), float(value["im"])))  # type: ignore[arg-type]
            items = ", ".join(
                f"{key!r}: {self.expression(item, None, f'{path}.{key}')}"
                for key, item in value.items()
            )
            return "{" + items + "}"
        if isinstance(value, list):
            if any(isinstance(item, Mapping) for item in value):
                items = ", ".join(
                    self.expression(item, unit, f"{path}[{index}]")
                    for index, item in enumerate(value)
                )
                return f"[{items}]"
            return repr(decode(value, unit, BuildContext(), path))
        if value == NONE:
            return "None"
        return repr(decode(value, unit, BuildContext(), path))

    def factory(self, table: Mapping[str, object], path: str) -> str:
        """Return a registered type's call.

        Parameters
        ----------
        table : collections.abc.Mapping
            The table with its ``type``.
        path : str
            Its parameter path.

        Returns
        -------
        str
            The call, with the name imported.

        Raises
        ------
        RecipeError
            For a factory built on the route's grid.
        """
        factory = REGISTRY[str(table["type"])]
        if factory.context and self.grid is None:
            msg = (
                f"{factory.name} is built on the route's grid, which a script can "
                "name only when the recipe gives the grid; give grid, or write that "
                "keyword by hand"
            )
            raise RecipeError(msg, path=path)
        root = factory.name.split(".")[0]
        self.need(factory.target.__module__, root)
        arguments = [] if not factory.context else [str(self.grid)]
        if factory.data_file is not None:
            arguments.append(f"{table['file']!r}")
        for key, item in table.items():
            if key in ("type", "file", "sha256"):
                continue
            rendered = self.expression(item, factory.units.get(key), f"{path}.{key}")
            arguments.append(f"{key}={rendered}")
        return f"{factory.name}({', '.join(arguments)})"


def _comment(value: object, unit: UnitSpec) -> str:
    """Return the comment that states a quantity as the recipe wrote it.

    Parameters
    ----------
    value : object
        The value as written.
    unit : str, callable or None
        Its unit; a value without one gets no comment.

    Returns
    -------
    str
        ``"  # 125 um"`` for a quantity string, or a list holding one; empty
        otherwise.
    """
    if unit is None:
        return ""
    if isinstance(value, str) and value != NONE:
        return f"  # {value}"
    if isinstance(value, list) and any(isinstance(item, str) for item in value):
        return "  # " + ", ".join(str(item) for item in value)
    return ""


def script(
    recipe: "Recipe",  # noqa: UP037 - checked at run time
    output: str | None,
) -> str:
    """Write a recipe as a plain Python script.

    Parameters
    ----------
    recipe : Recipe
        The recipe.
    output : str or None
        The file the run writes, or ``None`` for memory.

    Returns
    -------
    str
        The script.
    """
    from solprop.recipe.metadata import parameters  # noqa: PLC0415 - cycle

    writer = _Writer(recipe)
    table = parameters(recipe.route)
    document = recipe.to_dict()
    lines = []
    prelude = []
    grid = document["build"].get("grid")
    if isinstance(grid, Mapping):
        # A value built on the grid needs it by name: bind the grid first.
        prelude.append(f"grid = {writer.expression(grid, None, 'grid')}")
        writer.grid = "grid"
    for name, value in document["build"].items():
        entry = table[name]
        if writer.grid is not None and name == "grid":
            rendered = writer.grid
        elif entry.none and value == NONE:
            rendered = "None"
        elif isinstance(value, Mapping | list):
            rendered = writer.expression(value, unit_of(entry, value), name)
        else:
            rendered = repr(decode_keyword(entry, value, BuildContext()))
        comment = _comment(value, unit_of(entry, value))
        lines.append(f"{_INDENT}{name}={rendered},{comment}")
    execution = dict(document.get("execution", {}))
    budget = execution.pop("memory_budget", None)
    target = repr(output)
    title = recipe.title or f"a {recipe.route} run"
    header = [f'"""{title}, written from its recipe by solprop."""', ""]
    if execution or budget is not None:
        writer.need("solprop.driver", "Simulation")
        settings = ", ".join(
            f"{key}={tuple(item) if isinstance(item, list) else item!r}"
            for key, item in execution.items()
        )
        body = [
            f"simulation = Simulation.from_{recipe.route}(",
            *lines,
            ")",
        ]
        if settings:
            writer.need("dataclasses", "")
            body.append(
                "simulation.settings = dataclasses.replace("
                f"simulation.settings, {settings})"
            )
        run = f"output = simulation.run({target}"
        if budget is not None:
            run += f", memory_budget={None if budget == NONE else budget!r}"
        body.append(run + ")")
    else:
        writer.need("solprop.driver", f"prop_{recipe.route}")
        body = [f"output = prop_{recipe.route}(", *lines]
        if output is not None:
            body.append(f"{_INDENT}output={target},")
        body.append(")")
    return "\n".join([*header, *_imports(writer), "", *prelude, *body]) + "\n"


def _imports(writer: _Writer) -> list[str]:
    """Return the script's import lines: plain imports first, then from-imports.

    Parameters
    ----------
    writer : _Writer
        The writer that collected the names.

    Returns
    -------
    list of str
        The lines, each group sorted by module.
    """
    plain = [
        f"import {module}"
        for module in sorted(writer.imports)
        if not any(writer.imports[module])
    ]
    named = [
        f"from {module} import {', '.join(sorted(n for n in names if n))}"
        for module, names in sorted(writer.imports.items())
        if any(names)
    ]
    return [*plain, *named]
