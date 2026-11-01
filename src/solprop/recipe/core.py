# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The :class:`Recipe`: a versioned TOML description of a simulation (SEP 41, S1).

A recipe is read, checked against the parameter metadata, built into a
:class:`~solprop.driver.simulation.Simulation`, run, written back and exported as a
plain Python script. It is one-way: objects are built from it and never turned back
into tables, so a saved recipe always comes from a recipe or from plain values
(:meth:`Recipe.from_keywords`).
"""

import copy
import dataclasses
import importlib.resources
import math
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from solprop.exceptions import SolpropError, SolpropWarning
from solprop.recipe.document import (
    FORMAT,
    NONE,
    Document,
    dump_text,
    parse_text,
    unknown_key,
    validate_document,
)
from solprop.recipe.errors import RecipeError
from solprop.recipe.metadata import REQUIRED, Parameter, parameters, surface
from solprop.recipe.paths import format_path, get_path, parse_path, set_path
from solprop.recipe.quantity import Unit, format_quantity, parse_quantity
from solprop.recipe.registry import REGISTRY, BuildContext, coerce, decode_keyword

if TYPE_CHECKING:
    from solprop._interrupts import StopFlag
    from solprop.driver.simulation import Simulation
    from solprop.grid import TimeGrid
    from solprop.log import ProgressChoice, ProgressReporter
    from solprop.output.schema import Output, RecipeRecord

__all__ = ["Recipe", "RecipeIssue", "resume_file"]

ROUTES: Final = ("gnlse", "capillary")
"""The routes a recipe of format 1 describes in this release (SEP 41, phase 1)."""

_EXAMPLES: Final = "examples"
"""The package directory that holds the example recipes."""


@dataclass(frozen=True)
class RecipeIssue:
    """One problem :meth:`Recipe.check` found, placed at the parameter it concerns.

    Attributes
    ----------
    message : str
        What is wrong, in the words of the refusal or warning.
    parameter : str or tuple of str or None
        The path of the value at fault, or the paths of a conflict; ``None`` when the
        issue concerns no single value.
    severity : {'error', 'warning'}
        Whether the recipe cannot be built, or builds with a warning.
    rule : str or None
        The identifier of the combination rule a refusal enforces, where it is one.

    Examples
    --------
    >>> from solprop.recipe import RecipeIssue
    >>> RecipeIssue("must be positive", "pressure", "error").severity
    'error'
    """

    message: str
    parameter: str | tuple[str, ...] | None
    severity: Literal["error", "warning"]
    rule: str | None = None


class Recipe:
    """A simulation written as TOML: read, checked, built, run, saved and exported.

    Construct one with :meth:`load`, :meth:`loads`, :meth:`from_dict`,
    :meth:`from_keywords` or :meth:`example`, not directly.

    Parameters
    ----------
    document : collections.abc.Mapping
        The recipe's tables, as :func:`tomllib.loads` reads them.
    text : str or None, optional
        The text it was read from, kept so that an unedited recipe saves verbatim.
    base : pathlib.Path or None, optional
        The directory its data files resolve against.
    allow_outside : bool, optional
        Whether a data file may lie outside `base`.

    Attributes
    ----------
    route : str
        The route the recipe builds.

    Examples
    --------
    >>> from solprop.recipe import Recipe
    >>> recipe = Recipe.loads('''
    ... [recipe]
    ... format = 1
    ... route = "gnlse"
    ...
    ... [build]
    ... gamma = 0.11
    ... length = "1 cm"
    ... betas = [0, 0, "-0.01183 ps^2/m"]
    ... wavelength = "835 nm"
    ... wavelength_limits = ["500 nm", "2 um"]
    ... trange = "1 ps"
    ... width = "100 fs"
    ... peak_power = "9.8 W"
    ... shape = "sech"
    ... noise = "none"
    ... ''')
    >>> recipe.route, recipe.get("length")
    ('gnlse', '1 cm')
    >>> recipe.set("length", "2 cm")
    >>> recipe.get("length")
    '2 cm'
    """

    def __init__(
        self,
        document: Mapping[str, object],
        *,
        text: str | None = None,
        base: Path | None = None,
        allow_outside: bool = False,
    ) -> None:
        """Adopt a checked document."""
        self._document: Document = validate_document(document)
        self._text = text
        self._base = base
        self._allow_outside = allow_outside
        # Data files a run's output embedded, by the name the recipe refers to them by,
        # and the generated name each is filed under (from_output).
        self._embedded: dict[str, bytes] = {}
        self._filed: dict[str, str] = {}
        self.route = str(self._document["recipe"]["route"])
        if self.route not in ROUTES:
            raise unknown_key(
                self.route, ROUTES, where="the routes a recipe describes", path="route"
            )
        self._check_keys()

    # ------------------------------------------------------------------ construction

    @classmethod
    def loads(
        cls, text: str, *, base: Path | None = None, allow_outside: bool = False
    ) -> "Recipe":  # noqa: UP037 - the class, checked at run time
        """Read a recipe from TOML text.

        Parameters
        ----------
        text : str
            The recipe.
        base : pathlib.Path or None, optional
            The directory its data files resolve against; ``None`` allows none.
        allow_outside : bool, optional
            Whether a data file may lie outside `base`.

        Returns
        -------
        Recipe
            The recipe, which saves as `text` until it is edited.
        """
        return cls(parse_text(text), text=text, base=base, allow_outside=allow_outside)

    @classmethod
    def load(cls, path: str | Path, *, allow_outside: bool = False) -> "Recipe":  # noqa: UP037 - the class, checked at run time
        """Read a recipe file.

        Parameters
        ----------
        path : str or pathlib.Path
            The ``.toml`` file.
        allow_outside : bool, optional
            Whether a data file it refers to may lie outside its directory.

        Returns
        -------
        Recipe
            The recipe, whose data files resolve against the file's directory.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        return cls.loads(text, base=path.resolve().parent, allow_outside=allow_outside)

    @classmethod
    def from_dict(
        cls, document: Mapping[str, object], *, base: Path | None = None
    ) -> "Recipe":  # noqa: UP037 - the class, checked at run time
        """Build a recipe from its tables.

        Parameters
        ----------
        document : collections.abc.Mapping
            The tables, as TOML would read them.
        base : pathlib.Path or None, optional
            The directory its data files resolve against.

        Returns
        -------
        Recipe
            The recipe.
        """
        return cls(copy.deepcopy(dict(document)), base=base)

    @classmethod
    def from_keywords(
        cls, route: str, *, title: str | None = None, **keywords: object
    ) -> "Recipe":  # noqa: UP037 - the class, checked at run time
        """Write a route's keyword call as a recipe, from plain values only.

        Parameters
        ----------
        route : {'gnlse', 'capillary'}
            The route.
        title : str or None, optional
            The recipe's title.
        **keywords : object
            The route's build keywords, as numbers in their SI unit (pressure in bar),
            strings, booleans, and lists and mappings of these.

        Returns
        -------
        Recipe
            The recipe.

        Raises
        ------
        RecipeError
            If a value is an object or a callable, which a recipe holds only as a table
            with a ``type`` key, or through the Python export.

        Examples
        --------
        >>> from solprop.recipe import Recipe
        >>> recipe = Recipe.from_keywords(
        ...     "capillary", radius=125e-6, length=1.0, gas="He", pressure=1.2,
        ...     wavelength=800e-9, fwhm=7.5e-15, energy=275e-6,
        ... )
        >>> recipe.get("radius")
        0.000125
        """
        build = {name: _plain(value, name) for name, value in keywords.items()}
        header: dict[str, object] = {"format": FORMAT, "route": route}
        if title is not None:
            header["title"] = title
        return cls({"recipe": header, "build": build})

    @classmethod
    def examples(cls) -> tuple[str, ...]:
        """Return the names of the example recipes solprop ships.

        Returns
        -------
        tuple of str
            The names :meth:`example` takes, sorted.

        Examples
        --------
        >>> from solprop.recipe import Recipe
        >>> "rdw" in Recipe.examples()
        True
        """
        root = importlib.resources.files("solprop.recipe").joinpath(_EXAMPLES)
        return tuple(
            sorted(
                entry.name.removesuffix(".toml")
                for entry in root.iterdir()
                if entry.name.endswith(".toml")
            )
        )

    @classmethod
    def example(cls, name: str) -> "Recipe":  # noqa: UP037 - checked at run time
        """Load one of the example recipes solprop ships, each a gallery example's.

        Parameters
        ----------
        name : str
            The example's name, one of :meth:`examples`.

        Returns
        -------
        Recipe
            The recipe.

        Raises
        ------
        RecipeError
            If no example has that name.
        """
        if name not in cls.examples():
            raise unknown_key(name, cls.examples(), where="the example recipes")
        root = importlib.resources.files("solprop.recipe").joinpath(_EXAMPLES)
        with importlib.resources.as_file(root.joinpath(f"{name}.toml")) as path:
            return cls.load(path)

    @classmethod
    def from_output(cls, path: str | Path, *, resolved: bool = False) -> "Recipe":  # noqa: UP037 - the class, checked at run time
        """Read back the recipe a run's output records (SEP 41, S2).

        Parameters
        ----------
        path : str or pathlib.Path
            The HDF5 output.
        resolved : bool, optional
            Read the resolved recipe, which rebuilds the run as it ran, rather than
            the authored one, for editing.

        Returns
        -------
        Recipe
            The recipe, holding the data files the output embedded.

        Raises
        ------
        RecipeError
            If the file records no recipe, being a file from a script, or its recipe
            is of a newer format.
        """
        from solprop.output.hdf5 import open_file  # noqa: PLC0415 - layer above

        with open_file(path) as handler:
            record = handler.recipe
        if record is None:
            msg = (
                f"{path} records no recipe: it was written from a script, which is "
                "what resumes or rebuilds it"
            )
            raise RecipeError(msg)
        text = record.resolved if resolved else record.authored
        recipe = cls.loads(text)
        for name, entry in record.files.items():
            recipe._filed[entry.path] = name
            if entry.data is not None:
                recipe._embedded[entry.path] = entry.data
        return recipe

    # ------------------------------------------------------------------ the values

    @property
    def title(self) -> str | None:
        """The recipe's title, or ``None``."""
        title = self._document["recipe"].get("title")
        return None if title is None else str(title)

    def to_dict(self) -> Document:
        """Return a copy of the recipe's tables.

        Returns
        -------
        dict
            The five tables that are present, as TOML would read them.
        """
        return copy.deepcopy(self._document)

    def __eq__(self, other: object) -> bool:
        """Compare two recipes by their values, not their text or spelling.

        Parameters
        ----------
        other : object
            The other recipe.

        Returns
        -------
        bool
            Whether the tables are equal.
        """
        if not isinstance(other, Recipe):
            return NotImplemented
        return self._canonical_document() == other._canonical_document()

    def _canonical_document(self) -> Document:
        """Return the tables with each quantity spelled in its display unit.

        Returns
        -------
        dict
            The document a canonical save writes, so that two spellings of one value,
            ``"1 cm"`` and ``0.01``, compare equal.
        """
        document = self.to_dict()
        table = self._table()
        document["build"] = {
            name: _canonical(table[name], value)
            for name, value in document["build"].items()
        }
        return document

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        """Return the route and the title.

        Returns
        -------
        str
            For example ``Recipe(capillary, 'RDW in a helium capillary')``.
        """
        title = "" if self.title is None else f", {self.title!r}"
        return f"Recipe({self.route}{title})"

    def get(self, path: str) -> object:
        """Return a value of the ``[build]`` table as the recipe writes it.

        Parameters
        ----------
        path : str
            A parameter path, as ``"pressure"`` or ``"pulses[1].energy"``.

        Returns
        -------
        object
            The value as written: a number, a quantity string, a list or a table.
        """
        return get_path(self._document["build"], path)

    def set(self, path: str, value: object) -> None:
        """Set a value of the ``[build]`` table, checked against the metadata.

        Parameters
        ----------
        path : str
            A parameter path; the keyword it starts with must be the route's.
        value : object
            A number in the keyword's unit, a quantity string, a boolean, a string,
            or a list or table of these.

        Raises
        ------
        RecipeError
            If the keyword is not the route's, or the value does not decode.
        """
        steps = parse_path(path)
        build = set_path(self._document["build"], steps, _plain(value, path))
        keyword = format_path(steps[:1])
        context = self._context()
        self._decode(keyword, build[keyword], _deferred(context))
        self._document["build"] = build
        self._text = None

    # ------------------------------------------------------------------ checking

    def _table(self) -> Mapping[str, Parameter]:
        """Return the route's parameter metadata.

        Returns
        -------
        collections.abc.Mapping
            :class:`~solprop.recipe.metadata.Parameter` by keyword.
        """
        return parameters(self.route)

    def _check_keys(self) -> None:
        """Refuse a build or execution key the route does not have.

        Raises
        ------
        RecipeError
            Naming the key and the nearest valid one.
        """
        table = self._table()
        for key in self._document["build"]:
            if key not in table:
                raise unknown_key(
                    key,
                    list(table),
                    where=f"the {self.route} route's keywords",
                    path=key,
                )

    def _decode(self, name: str, value: object, context: BuildContext) -> object:
        """Return one build keyword's value as the constructor takes it.

        Parameters
        ----------
        name : str
            The keyword.
        value : object
            Its value as written.
        context : BuildContext
            The data files' directory and the route's grid.

        Returns
        -------
        object
            The decoded value.
        """
        table = self._table()
        if name not in table:
            raise unknown_key(
                name, list(table), where=f"the {self.route} route's keywords", path=name
            )
        decoded = decode_keyword(table[name], value, context)
        return coerce(surface(self.route), name, decoded)

    def check(self) -> list[RecipeIssue]:
        """Find what stops the recipe building, or makes it warn (SEP 41, S10).

        Every value is decoded against the metadata first, each problem reported at
        its path. If none is refused, the simulation is built and translated, which is
        where the refusals that depend on several values are found, stopping before a
        backend is constructed; the first refusal is reported with its parameter, and
        every warning of solprop's with its own. It costs a translation: call it on a
        pause in typing, not on every keystroke.

        Returns
        -------
        list of RecipeIssue
            The issues, errors first; empty when the recipe builds cleanly.
        """
        issues: list[RecipeIssue] = []
        context = self._context()
        for name, value in self._document["build"].items():
            try:
                self._decode(name, value, _deferred(context))
            except RecipeError as error:
                issues.append(_issue(error))
        missing = [
            name
            for name, entry in self._table().items()
            if entry.default is REQUIRED and name not in self._document["build"]
        ]
        issues.extend(
            RecipeIssue(f"{name} is required", name, "error") for name in missing
        )
        if issues:
            return issues
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SolpropWarning)
            try:
                self.build().translate()
            except SolpropError as error:
                issues.append(_issue(error))
        issues.extend(
            RecipeIssue(
                str(record.message),
                getattr(record.message, "parameter", None),
                "warning",
            )
            for record in caught
            if isinstance(record.message, SolpropWarning)
        )
        return issues

    # ------------------------------------------------------------------ building

    def keywords(
        self,
        *,
        grid: "TimeGrid | None" = None,  # noqa: UP037 - checked at run time
    ) -> dict[str, object]:
        """Return the route's build keywords as the constructor takes them.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid or None, optional
            The route's grid, for a value built on it; ``None`` leaves such values
            out.

        Returns
        -------
        dict
            Keyword arguments for the route's ``Simulation.from_*``.
        """
        context = self._context(grid)
        out: dict[str, object] = {}
        for name, value in self._document["build"].items():
            if grid is None and _needs_grid(value):
                continue
            out[name] = self._decode(name, value, context)
        return out

    def build(self) -> "Simulation":  # noqa: UP037 - checked at run time
        """Build the simulation the recipe describes, without running it.

        Returns
        -------
        solprop.driver.simulation.Simulation
            The simulation, with ``[execution]``'s settings applied, carrying this
            recipe, which its first run records in its output.
        """
        builder = surface(self.route)
        simulation = builder(**self.keywords())
        if any(_needs_grid(value) for value in self._document["build"].values()):
            simulation = builder(**self.keywords(grid=simulation._base_grid))
        execution = self._execution()
        changes = {
            name: value for name, value in execution.items() if name != "memory_budget"
        }
        if changes:
            simulation.settings = dataclasses.replace(simulation.settings, **changes)
        # The simulation carries its recipe, which its first run records (S2).
        simulation._recipe = self
        return simulation

    def record(
        self,
        simulation: "Simulation",  # noqa: UP037 - checked at run time
    ) -> "RecipeRecord":  # noqa: UP037 - checked at run time
        """Return the record a run built from this recipe keeps in its output.

        Called by :meth:`~solprop.driver.simulation.Simulation.run` at the run's
        start, with the settings it starts with.

        Parameters
        ----------
        simulation : solprop.driver.simulation.Simulation
            The simulation this recipe built.

        Returns
        -------
        solprop.output.schema.RecipeRecord
            The authored and resolved texts, and the data files
            (:mod:`solprop.recipe.provenance`).
        """
        from solprop.recipe.provenance import record  # noqa: PLC0415 - cycle

        return record(
            self, simulation, base=self._base, allow_outside=self._allow_outside
        )

    def _execution(self) -> dict[str, object]:
        """Return the ``[execution]`` table's values as the settings take them.

        Returns
        -------
        dict
            The values, ``"none"`` read as ``None`` for ``memory_budget``.
        """
        table = parameters("execution")
        out: dict[str, object] = {}
        for name, value in self._document.get("execution", {}).items():
            out[name] = (
                None
                if table[name].none and value == NONE
                else tuple(value)
                if isinstance(value, list)
                else value
            )
        return out

    def run(
        self,
        output: "str | Path | Output | None" = None,  # noqa: UP037 - run time
        **run_keywords: object,
    ) -> "Output":  # noqa: UP037 - checked at run time
        """Build the simulation and run it, with ``[execution]`` applied.

        Parameters
        ----------
        output : str, pathlib.Path, Output or None, optional
            Where to write, as :meth:`~solprop.driver.simulation.Simulation.run`
            takes it; ``None`` keeps the run in memory.
        **run_keywords : object
            Further keywords of :meth:`~solprop.driver.simulation.Simulation.run`,
            such as ``progress`` or ``resume``. ``memory_budget`` comes from the
            recipe's ``[execution]`` table unless given here.

        Returns
        -------
        solprop.output.schema.Output
            The run's output.
        """
        execution = self._execution()
        if "memory_budget" in execution and "memory_budget" not in run_keywords:
            run_keywords["memory_budget"] = execution["memory_budget"]
        return self.build().run(output, **run_keywords)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ writing

    def to_text(self) -> str:
        """Return the recipe's TOML text.

        Returns
        -------
        str
            The text it was read from while it is unedited, comments included;
            otherwise canonical TOML, each quantity in its keyword's display unit.
        """
        if self._text is not None:
            return self._text
        return dump_text(self._canonical_document())

    def _context(self, grid: "TimeGrid | None" = None) -> BuildContext:  # noqa: UP037
        """Return the context the recipe's values are decoded in.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid or None, optional
            The route's grid, for a value built on it.

        Returns
        -------
        BuildContext
            The data files' directory, the embedded files and the grid.
        """
        return BuildContext(
            base=self._base,
            allow_outside=self._allow_outside,
            grid=grid,
            embedded=self._embedded,
        )

    def save(self, path: str | Path) -> None:
        """Write the recipe to a file.

        A recipe read back from a run's output (:meth:`from_output`) writes the data
        files the output embedded beside it, under the generated names they were
        filed by, and refers to them by those names; a file the output only referred
        to stays a reference, which :meth:`check` reports when it is missing.

        Parameters
        ----------
        path : str or pathlib.Path
            The ``.toml`` file to write.
        """
        path = Path(path)
        if not self._embedded:
            path.write_text(self.to_text(), encoding="utf-8")
            return
        for original, data in self._embedded.items():
            (path.parent / self._filed[original]).write_bytes(data)
        document = self._canonical_document()
        document["build"] = cast(
            "dict[str, object]", _renamed(document["build"], self._filed)
        )
        path.write_text(dump_text(document), encoding="utf-8")

    def to_python(self, output: str | None = None) -> str:
        """Return a plain Python script that makes the same run.

        The route's ``prop_*`` call when every key is on its surface, otherwise
        ``Simulation.from_*``, the settings and ``run``: keyword arguments in SI, with
        a comment per keyword stating the recipe's value where it was a quantity.

        Parameters
        ----------
        output : str or None, optional
            The file the script's run writes; ``None`` keeps it in memory.

        Returns
        -------
        str
            The script.
        """
        from solprop.recipe.export import script  # noqa: PLC0415 - only here

        return script(self, output)

    def _values(self) -> Iterator[tuple[str, object, object]]:
        """Yield each build keyword, its value as written and as decoded.

        Yields
        ------
        tuple
            The keyword, the value as written and the decoded value.
        """
        context = self._context()
        for name, value in self._document["build"].items():
            yield name, value, self._decode(name, value, _deferred(context))


def _deferred(context: BuildContext) -> BuildContext:
    """Return a context for checking values built on the grid before it exists.

    Parameters
    ----------
    context : BuildContext
        The recipe's context.

    Returns
    -------
    BuildContext
        The same, with a stand-in grid, so a factory on the grid can be checked.
    """
    from solprop.grid import EnvelopeGrid  # noqa: PLC0415 - only for the stand-in

    grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 200e-15)
    return dataclasses.replace(context, grid=grid)


def _needs_grid(value: object) -> bool:
    """Return whether a value holds a factory built on the route's grid.

    Parameters
    ----------
    value : object
        A value as written.

    Returns
    -------
    bool
        True if a table in it names a factory whose context is the grid.
    """
    if isinstance(value, Mapping):
        factory = REGISTRY.get(str(value.get("type")))
        if factory is not None and "grid" in factory.context:
            return True
        return any(_needs_grid(item) for item in value.values())
    if isinstance(value, list):
        return any(_needs_grid(item) for item in value)
    return False


def _issue(error: SolpropError) -> RecipeIssue:
    """Return the issue a refusal makes.

    Parameters
    ----------
    error : solprop.exceptions.SolpropError
        The refusal.

    Returns
    -------
    RecipeIssue
        Its message, parameter and rule, as an error.
    """
    message = error.message if isinstance(error, RecipeError) else str(error)
    return RecipeIssue(message, error.parameter, "error", error.rule)


def _plain(value: object, path: str) -> object:
    """Return a value as a recipe holds it, refusing an object or a callable.

    Parameters
    ----------
    value : object
        A Python value.
    path : str
        Its path, for the refusal.

    Returns
    -------
    object
        The value, with tuples as lists and complex numbers as ``{re, im}``.

    Raises
    ------
    RecipeError
        If the value is an object or a callable.
    """
    if value is None:
        return NONE
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, complex):
        return {"re": value.real, "im": value.imag}
    if isinstance(value, Sequence):
        return [_plain(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        return {str(key): _plain(item, f"{path}.{key}") for key, item in value.items()}
    kind = "a callable" if callable(value) else f"a {type(value).__name__}"
    msg = (
        f"{kind} cannot be written into a recipe: give it as a table with a type key "
        "(see the recipe guide), or write the run in Python"
    )
    raise RecipeError(msg, path=path)


def _canonical(entry: Parameter, value: object) -> object:
    """Spell a value in its keyword's display unit, for a canonical recipe.

    Parameters
    ----------
    entry : Parameter
        The keyword's metadata.
    value : object
        The value as written.

    Returns
    -------
    object
        A quantity string in the display unit for a number of a keyword that has
        one, element by element for a list; anything else as it stands.
    """
    unit = next((form.unit for form in entry.forms if form.unit is not None), None)
    if entry.display is None or unit is None:
        return value
    if isinstance(value, list):
        return [_canonical(entry, item) for item in value]
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return value
    if isinstance(value, str) and value == NONE:
        return value
    number = parse_quantity(value, Unit(unit), path=entry.name)
    if not math.isfinite(number):  # pragma: no cover - refused by parse_quantity
        return value
    return format_quantity(number, Unit(unit), entry.display)


def _renamed(value: object, names: Mapping[str, str]) -> object:
    """Return a value with its data-file references renamed.

    Parameters
    ----------
    value : object
        A value as the recipe writes it.
    names : collections.abc.Mapping
        The new name of each file, by the name the recipe refers to it by.

    Returns
    -------
    object
        A copy, each ``file`` that `names` holds replaced.
    """
    if isinstance(value, Mapping):
        out = {key: _renamed(item, names) for key, item in value.items()}
        if "sha256" in out and isinstance(out.get("file"), str):
            out["file"] = names.get(str(out["file"]), out["file"])
        return out
    if isinstance(value, list):
        return [_renamed(item, names) for item in value]
    return value


def resume_file(
    path: str | Path,
    mode: bool | Literal["interpolated", "force"] = True,
    *,
    memory_budget: int | str | None = None,
    node_budget: int | None = None,
    progress: "ProgressReporter | ProgressChoice | None" = None,  # noqa: UP037
    stop: "StopFlag | None" = None,  # noqa: UP037 - checked at run time
) -> "Output":  # noqa: UP037 - checked at run time
    """Continue a run from its output file alone, rebuilding it from its recipe.

    The simulation is rebuilt from the resolved recipe the file records (SEP 41, S2)
    and run into the same file with ``resume=mode``, so the resume gate decides by its
    comparison of the stored and rebuilt records, and records any difference as it
    does for any resume (SEP 42). A file written from a script records no recipe; its
    script resumes it.

    Resuming a file runs the setup it holds. A file received from someone else does
    not choose its own limits: the recipe's ``memory_budget`` and ``node_budget`` are
    ignored, and the caller's apply, refusing by default rather than warning.

    Parameters
    ----------
    path : str or pathlib.Path
        The output file.
    mode : bool or {'interpolated', 'force'}, optional
        As :meth:`~solprop.driver.simulation.Simulation.run`'s ``resume`` takes it.
    memory_budget : int or str or None, optional
        The run's memory budget; ``None`` takes the machine's default, four fifths of
        its memory, as a refusal.
    node_budget : int or None, optional
        The node-buffer budget of a capillary run; ``None`` takes solprop's default.
    progress : object, optional
        Where progress goes, as :meth:`~solprop.driver.simulation.Simulation.run`
        takes it.
    stop : object, optional
        The run's stop flag, any object with an ``is_set()`` method.

    Returns
    -------
    solprop.output.schema.Output
        The handler the run continued in.

    Raises
    ------
    RecipeError
        If the file records no recipe, or one of a newer format.
    """
    from solprop.driver.memory import resolve_memory_budget  # noqa: PLC0415 - layer D

    recipe = Recipe.from_output(path, resolved=True)
    document = recipe.to_dict()
    document["build"].pop("node_budget", None)
    if node_budget is not None:
        document["build"]["node_budget"] = node_budget
    document.get("execution", {}).pop("memory_budget", None)
    rebuilt = Recipe(document)
    rebuilt._embedded = recipe._embedded
    rebuilt._filed = recipe._filed
    if memory_budget is None:
        default = resolve_memory_budget(None)
        memory_budget = default.limit if default.limit is not None else "none"
    return rebuilt.build().run(
        Path(path),
        resume=mode,
        memory_budget=memory_budget,
        progress=progress,
        stop=stop,
    )
