# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A recipe written out as the plain Python script a person would write (SEP 41, S1).

The script is the way out of a recipe for whatever a recipe cannot hold: a custom
pulse, a callable, an unregistered type. It makes the same run: the same simulation,
hash included, and the same execution settings, which the hash does not cover.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from solprop.recipe.document import NONE
from solprop.recipe.errors import RecipeError
from solprop.recipe.metadata import surface
from solprop.recipe.registry import (
    REGISTRY,
    BuildContext,
    UnitSpec,
    coerce,
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

    def __init__(self, recipe: "Recipe") -> None:
        """Start with no imports."""
        self.recipe = recipe
        self.imports: dict[str, set[str]] = {}
        # The name the script binds the route's grid to, when the recipe gives one.
        self.grid: str | None = None

    def need(self, module: str, name: str) -> None:
        """Record an import."""
        self.imports.setdefault(module, set()).add(name)

    def expression(  # noqa: PLR0911 - one return per kind of value
        self, value: object, unit: UnitSpec, path: str
    ) -> str:
        """Return a value as a Python expression."""
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
        # "none" says None for a keyword only; inside a table it is read as it stands.
        return repr(decode(value, unit, BuildContext(), path))

    def factory(self, table: Mapping[str, object], path: str) -> str:
        """Return a registered type's call.

        - ``table``: the table with its ``type``.
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
            if isinstance(item, list) and isinstance(
                coerce(factory.target, key, []), tuple
            ):
                # A parameter typed as a tuple is written as one, as by hand.
                inner = rendered[1:-1]
                rendered = f"({inner},)" if len(item) == 1 else f"({inner})"
            arguments.append(f"{key}={rendered}")
        return f"{factory.name}({', '.join(arguments)})"


def _comment(value: object, unit: UnitSpec) -> str:
    """Return the comment that states a quantity as the recipe wrote it.

    - Returns: ``" # 125 um"`` for a quantity string, or a list holding one; empty
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
    recipe: "Recipe",
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
    document["build"] = _array_pins(recipe, document["build"])
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
    pinned = _pins(recipe, writer)
    if pinned:
        # The literals are bound once, the pins solved on them, and the route called
        # with both (SEP 41: the solver call before the route's call).
        prelude.extend(["keywords = dict(", *lines, ")", *pinned.lines])
        lines = [f"{_INDENT}**keywords,", *pinned.keywords]
    execution = dict(document.get("execution", {}))
    target = repr(output)
    title = recipe.title or f"a {recipe.route} run"
    # The route's entry point: the radial route's is the free-space one.
    entry = surface(recipe.route).__name__.removeprefix("from_")
    header = [f'"""{title}, written from its recipe by solprop."""', ""]
    if execution:
        writer.need("solprop.driver", "Simulation")
        settings = ", ".join(
            f"{key}={tuple(item) if isinstance(item, list) else item!r}"
            for key, item in execution.items()
        )
        body = [
            f"simulation = Simulation.from_{entry}(",
            *lines,
            ")",
        ]
        writer.need("dataclasses", "")
        body.append(
            "simulation.settings = dataclasses.replace("
            f"simulation.settings, {settings})"
        )
        body.append(f"output = simulation.run({target})")
    else:
        writer.need("solprop.driver", f"prop_{entry}")
        body = [f"output = prop_{entry}(", *lines]
        if output is not None:
            body.append(f"{_INDENT}output={target},")
        body.append(")")
    return "\n".join([*header, *_imports(writer), "", *prelude, *body]) + "\n"


@dataclass(frozen=True)
class _Pinned:
    """The script lines that solve a recipe's pins.

    Attributes
    ----------
    lines : list of str
        The solver calls, each binding a name.
    keywords : list of str
        The route call's keyword lines that pass the solved values.
    """

    lines: list[str]
    keywords: list[str]

    def __bool__(self) -> bool:
        """Return whether there are pins."""
        return bool(self.lines)


def _pins(
    recipe: "Recipe",
    writer: _Writer,
) -> _Pinned:
    """Write a recipe's pins as solver calls on the bound keywords.

    The pin on the guide or fill becomes a :func:`~solprop.recipe.pins.solve_pin`
    call, a route-level pulse pin a :func:`~solprop.recipe.pins.solve_pulse` call on
    the keywords with the guide's value. A pin on one of ``pulses`` is written at the
    value it solves to now, with the pin in a comment: the pulse object cannot be
    built without its energy.
    """
    from solprop.recipe.pins import read_pins  # noqa: PLC0415 - cycle

    pins = read_pins(recipe)
    lines: list[str] = []
    keywords: list[str] = []
    solved: list[str] = []
    for pin in pins:
        name = pin.path
        if pin.stage == "guide":
            writer.need("solprop.recipe.pins", "solve_pin")
            kind = "ZDW" if pin.target == "zdw" else "RDW"
            writer.need("solprop.tools", kind)
            if pin.target == "zdw":
                target = f"ZDW({pin.keys['wavelength']!r})"
            else:
                extra = "".join(
                    f", {key}={pin.keys[key]!r}"
                    for key in ("peak_power", "fwhm", "shape", "nonlinear_phase")
                    if key in pin.keys
                )
                target = f"RDW(keywords['wavelength'], {pin.keys['rdw']!r}{extra})"
            lines.append(
                f"{name} = solve_pin({recipe.route!r}, keywords, {name!r}, {target}, "
                f"{pin.search!r}).pick({pin.root!r})"
            )
        elif "[" not in name:
            writer.need("solprop.recipe.pins", "solve_pulse")
            given = "".join(f", {item!r}: {item}" for item in solved)
            on = f"{{**keywords{given}}}" if solved else "keywords"
            lines.append(
                f"{name} = solve_pulse({recipe.route!r}, {on}, {name!r}, "
                f"{pin.keys['order']!r}, {pin.search!r}).pick({pin.root!r})"
            )
        else:
            # Written into the pulse's table as its literal (_array_pins).
            lines.append(
                f"# {name}: soliton order {pin.keys['order']!r}, solved when exported"
            )
            continue
        solved.append(name)
        keywords.append(f"{_INDENT}{name}={name},")
    return _Pinned(lines, keywords)


def _array_pins(
    recipe: "Recipe",
    build: dict[str, object],
) -> dict[str, object]:
    """Return ``[build]`` with each pin on one of ``pulses`` set to its solved value.

    - ``build``: its ``[build]`` table.
    - Returns: the table, unchanged when no such pin exists.
    """
    from solprop.recipe.paths import parse_path, set_path  # noqa: PLC0415 - cycle

    paths = [path for path in recipe.to_dict().get("pins", {}) if "[" in path]
    if not paths:
        return build
    resolved = recipe.resolve()
    for path in paths:
        build = set_path(build, parse_path(path), resolved.get(path))
    return build


def _imports(writer: _Writer) -> list[str]:
    """Return the script's import lines: plain imports first, then from-imports."""
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
