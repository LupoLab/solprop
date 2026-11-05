# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a problem class declares, and the readings of it that hold for every class."""

import math
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field

from solprop.recipe.errors import RecipeError
from solprop.recipe.paths import Attribute, get_path, parse_path, set_path

from solprop_gui.model.quantities import split
from solprop_gui.model.recipe import MISSING
from solprop_gui.registry.units import si

__all__ = [
    "AnalysisPage",
    "Derived",
    "Field",
    "ProblemClass",
    "Section",
    "Target",
    "derived_text",
    "derived_value",
    "even_planes",
    "field_changes",
    "field_meta",
    "field_value",
    "linked_paths",
    "negated_value",
    "other_keywords",
    "page_refusal",
    "pin_entry",
    "sheet_paths",
    "target_text",
]


@dataclass(frozen=True)
class Field:
    """One row of a sheet.

    Attributes
    ----------
    path : str
        Its recipe path in the ``[build]`` table.
    label : str or None
        What the row is called; ``None`` takes the front end's label of the keyword.
    choices : str or None
        The kernel's request that lists what the row offers, as ``"gases"`` does, in
        place of free text.
    also : tuple of str
        Paths that hold the same value and move with this row, where the object routes
        write it several times; ``[*]`` stands for every element of a list.
    negated : bool
        Whether the row shows the value with its sign changed, as a distance to a focus
        is the launch's distance from it.
    planes_over : str or None
        Makes this a count of evenly spaced saved planes, over the length at this path.
    moves_planes : str or None
        The saved planes at this path, which this length respaces.
    member : str or None
        Makes this a switch: whether a list at the path holds an entry of this type.
    goes_with : tuple of str
        Paths the switch's entry needs set, taken from the class's example.
    """

    path: str
    label: str | None = None
    choices: str | None = None
    also: tuple[str, ...] = ()
    negated: bool = False
    planes_over: str | None = None
    moves_planes: str | None = None
    member: str | None = None
    goes_with: tuple[str, ...] = ()


@dataclass(frozen=True)
class Section:
    """A titled group of a sheet's rows.

    Attributes
    ----------
    title : str
        Its title.
    fields : tuple of Field
        Its rows, in order.
    folded : bool
        Whether it starts folded, as a simulation window does.
    """

    title: str
    fields: tuple[Field, ...]
    folded: bool = False


@dataclass(frozen=True)
class Target:
    """What typing into a derived number means: solprop's pin for it (design 7.5).

    Attributes
    ----------
    kind : str
        The pin's ``target``.
    key : str
        The pin's key that holds what was typed.
    unit : str or None
        The unit a typed number is in; ``None`` for a pure number.
    moves : tuple of str
        The keywords that can move to meet it, the first by default.
    """

    kind: str
    key: str
    unit: str | None
    moves: tuple[str, ...]


@dataclass(frozen=True)
class Derived:
    """A number a sheet shows, which solprop works out from the setup.

    Attributes
    ----------
    key : str
        Its name in the kernel's reply.
    label : str
        What it is called.
    unit : str
        Its unit, unprefixed where the number takes an SI prefix.
    factor : float
        What the reply's SI value is multiplied by to read in the unit.
    prefixed : bool
        Whether the number takes an SI prefix.
    target : Target or None
        What typing into it does; ``None`` when it can only be read.
    """

    key: str
    label: str
    unit: str = ""
    factor: float = 1.0
    prefixed: bool = True
    target: Target | None = None


@dataclass(frozen=True)
class AnalysisPage:
    """A page of the Analyse rail.

    Attributes
    ----------
    key : str
        Its identity.
    title : str
        Its name on the rail.
    needs : tuple of str
        What the file must hold for the page to be offered, each of ``"fields"``,
        ``"statistics"``, ``"modes"`` and ``"ensemble"``.
    operation : str or None
        The operation of ``solprop.capabilities.output`` whose refusal disables the
        page, with solprop's reason.
    view : str or None
        Which view draws it, where that is not the page's own key.
    """

    key: str
    title: str
    needs: tuple[str, ...] = ()
    operation: str | None = None
    view: str | None = None


@dataclass(frozen=True)
class ProblemClass:
    """One problem class.

    Attributes
    ----------
    key : str
        Its identity.
    title : str
        Its name.
    summary : str
        One sentence for its tile on Home.
    route : str
        The solprop route it builds.
    example : str
        The solprop example a new sheet starts from (``Recipe.example``), so that the
        front end holds no defaults of its own.
    marker : tuple of str
        Keywords whose presence in ``[build]`` makes a recipe of the route this class's
        rather than another's: an antiresonant fibre is a capillary with a ``lattice``.
    sections : tuple of Section
        Its sheet.
    derived : tuple of Derived
        The numbers its sheet shows.
    preview : str
        What its sheet's plot shows.
    pages : tuple of AnalysisPage
        Its Analyse rail, in order; the first opens on a new file.
    statistics : tuple of str
        The statistics its pages need recorded.
    numbers : str
        The kernel's request for its derived numbers, asked at each edit where it is
        quick enough; ``"summarise"`` follows solprop's check instead.
    curve : str or None
        The kernel's request for its preview's curve, if it has one yet.
    lens : tuple of str or None
        The waist, launch distance and wavelength paths a lens sets and reads, where the
        sheet offers one.
    """

    key: str
    title: str
    summary: str
    route: str
    example: str
    sections: tuple[Section, ...]
    pages: tuple[AnalysisPage, ...]
    preview: str
    derived: tuple[Derived, ...] = ()
    marker: tuple[str, ...] = field(default=())
    statistics: tuple[str, ...] = ()
    numbers: str = "summarise"
    curve: str | None = None
    lens: tuple[str, str, str] | None = None


def derived_value(summary: Mapping[str, object] | None, key: str) -> float | None:
    """Find a derived number in the kernel's summary.

    Parameters
    ----------
    summary : Mapping or None
        The kernel's ``summarise`` reply; ``None`` when it failed.
    key : str
        The number's key, in the summary or in its ``design`` table.

    Returns
    -------
    float or None
        The number, or ``None`` where solprop gave none.

    Examples
    --------
    >>> derived_value({"peak_power": 2.0, "design": {"zdw": 3.0}}, "zdw")
    3.0
    >>> derived_value({"peak_power": 2.0}, "peak_power")
    2.0
    >>> derived_value({"design": None}, "zdw"), derived_value(None, "zdw")
    (None, None)
    """
    if summary is None:
        return None
    design = summary.get("design")
    tables = (summary, design) if isinstance(design, Mapping) else (summary,)
    for table in tables:
        value = table.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def derived_text(number: Derived, value: float | None) -> str:
    """Write a derived number as its sheet shows it.

    Parameters
    ----------
    number : Derived
        What it is.
    value : float or None
        Its SI value; ``None`` when solprop gave none.

    Returns
    -------
    str
        The text: ``"—"`` for none, ``"none"`` for a number that does not exist, as
        a zero-dispersion wavelength may not.

    Examples
    --------
    >>> derived_text(Derived("zdw", "ZDW", "m"), 4.23e-7)
    '423 nm'
    >>> derived_text(Derived("i", "I", "W/cm²", 1e-4, prefixed=False), 3.7e18)
    '3.7e+14 W/cm²'
    >>> derived_text(Derived("n", "N", prefixed=False), 2.5)
    '2.5'
    >>> derived_text(Derived("n", "N"), None), derived_text(Derived("n", "N"), math.nan)
    ('—', 'none')
    """
    if value is None:
        return "—"
    if not math.isfinite(value):
        return "none"
    scaled = value * number.factor
    if number.prefixed:
        return si(scaled, number.unit, digits=3)
    return f"{scaled:.3g} {number.unit}".strip()


def pin_entry(
    target: Target, value: str, search: Sequence[str] | None = None
) -> dict[str, object]:
    """Write what was typed into a derived number as a ``[pins]`` entry.

    Parameters
    ----------
    target : Target
        What it means.
    value : str
        The text typed, a number in the target's unit.
    search : Sequence of str or None, optional
        Where a pin on the guide or fill searches.

    Returns
    -------
    dict
        The entry.

    Raises
    ------
    ValueError
        If the text is not a number.

    Examples
    --------
    >>> order = Target("soliton_order", "order", None, ("energy",))
    >>> pin_entry(order, "3")
    {'target': 'soliton_order', 'order': 3.0}
    >>> zdw = Target("zdw", "wavelength", "nm", ("pressure",))
    >>> pin_entry(zdw, "450", ["0.01 bar", "20 bar"])["wavelength"]
    '450 nm'
    """
    number = float(value)
    entry: dict[str, object] = {
        "target": target.kind,
        target.key: number if target.unit is None else f"{value.strip()} {target.unit}",
    }
    if search is not None:
        entry["search"] = list(search)
    return entry


def target_text(target: Target, entry: Mapping[str, object]) -> str:
    """Write a pin's target as a sheet says it.

    Parameters
    ----------
    target : Target
        What it means.
    entry : Mapping
        The ``[pins]`` entry.

    Returns
    -------
    str
        What it holds.

    Examples
    --------
    >>> target_text(Target("soliton_order", "order", None, ()), {"order": 3.0})
    '3'
    >>> target_text(Target("zdw", "wavelength", "nm", ()), {"wavelength": "450 nm"})
    '450 nm'
    """
    held = entry.get(target.key)
    return f"{held:g}" if isinstance(held, int | float) else str(held)


_NEEDS = {
    "fields": "This run kept no fields, so this page has nothing to read.",
    "statistics": "This run recorded no statistics.",
    "modes": "This run has one mode.",
    "ensemble": "This run is not a seeded ensemble.",
}
"""Why a page is not offered, by what the file lacks."""


def page_refusal(page: AnalysisPage, capabilities: Mapping[str, object]) -> str | None:
    """Say why a file cannot show a page.

    Parameters
    ----------
    page : AnalysisPage
        The page.
    capabilities : Mapping
        The kernel's ``describe_output`` reply on the file: its ``z``, ``field_planes``,
        ``labels``, ``refused`` and ``ensemble``.

    Returns
    -------
    str or None
        The reason, solprop's own where it refuses; ``None`` when the page can be shown.

    Examples
    --------
    >>> page = AnalysisPage("compress", "Compress", ("fields",), "compression")
    >>> page_refusal(page, {"field_planes": [0], "refused": {}}) is None
    True
    >>> page_refusal(page, {"field_planes": [0], "refused": {"compression": "No."}})
    'No.'
    >>> page_refusal(page, {})
    'This run kept no fields, so this page has nothing to read.'
    """
    held = {
        "fields": bool(capabilities.get("field_planes")),
        # Every saved plane has its statistics row, a field or not.
        "statistics": bool(capabilities.get("z")),
        "modes": len(capabilities.get("labels") or ()) > 1,  # pyright: ignore[reportArgumentType]
        "ensemble": bool(capabilities.get("ensemble")),
    }
    for need in page.needs:
        if not held[need]:
            return _NEEDS[need]
    refused = capabilities.get("refused")
    if page.operation is not None and isinstance(refused, Mapping):
        reason = refused.get(page.operation)
        return None if reason is None else str(reason)
    return None


def sheet_paths(problem: ProblemClass) -> Iterator[str]:
    """Yield every recipe path a class's sheet shows.

    Parameters
    ----------
    problem : ProblemClass
        The class.

    Yields
    ------
    str
        Each path, in the sheet's order.

    Examples
    --------
    >>> rows = (Field("radius"), Field("length"))
    >>> problem = ProblemClass(
    ...     "c", "C", "", "capillary", "rdw", (Section("S", rows),), (), "β₂"
    ... )
    >>> list(sheet_paths(problem))
    ['radius', 'length']
    """
    for section in problem.sections:
        for row in section.fields:
            yield row.path


def field_meta(
    path: str,
    description: Mapping[str, object],
    build: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Return the metadata of a path, following the registered types the recipe names.

    ``initial.temporal.fwhm`` is the ``fwhm`` of the type ``initial.temporal`` names in
    this recipe, so a sheet's row is described by what the recipe holds.

    Parameters
    ----------
    path : str
        The recipe path, of keywords and attributes.
    description : Mapping
        The kernel's ``describe`` reply on the route.
    build : Mapping
        The recipe's ``[build]`` table.

    Returns
    -------
    Mapping or None
        The keyword's metadata, or ``None`` when the route has no such path.

    Examples
    --------
    >>> described = {
    ...     "parameters": [{"name": "initial"}],
    ...     "types": {"Beam": [{"name": "waist"}]},
    ... }
    >>> field_meta("initial.waist", described, {"initial": {"type": "Beam"}})
    {'name': 'waist'}
    >>> field_meta("initial", described, {}), field_meta("x", described, {})
    ({'name': 'initial'}, None)
    >>> field_meta("initial.waist", described, {})
    >>> field_meta("initial[0]", described, {})
    """
    try:
        steps = parse_path(path)
    except RecipeError:
        return None
    entries: Sequence[Mapping[str, object]] = description["parameters"]  # pyright: ignore[reportAssignmentType]
    catalog: Mapping[str, Sequence[Mapping[str, object]]] = description.get("types", {})  # pyright: ignore[reportAssignmentType]
    holder: object = build
    meta: Mapping[str, object] | None = None
    for step in steps:
        if not isinstance(step, Attribute) or not isinstance(holder, Mapping):
            return None
        meta = next((e for e in entries if e.get("name") == step.name), None)
        if meta is None:
            return None
        holder = holder.get(step.name)
        kind = holder.get("type") if isinstance(holder, Mapping) else None
        entries = catalog.get(str(kind), ())
    return meta


def _held(build: Mapping[str, object], path: str) -> object:
    """Return what a path of ``[build]`` holds.

    Parameters
    ----------
    build : Mapping
        The table.
    path : str
        The path.

    Returns
    -------
    object
        The value, or :data:`MISSING`.
    """
    try:
        return get_path(build, path)
    except RecipeError:
        return MISSING


def linked_paths(build: Mapping[str, object], pattern: str) -> list[str]:
    """Return the paths a pattern names in a recipe, those that exist.

    Parameters
    ----------
    build : Mapping
        The recipe's ``[build]`` table.
    pattern : str
        A path, in which ``[*]`` stands for every element of a list.

    Returns
    -------
    list of str
        The paths.

    Examples
    --------
    >>> build = {"responses": [{"gas": "Ar"}, {"thg": False}], "medium": {"gas": "Ar"}}
    >>> linked_paths(build, "responses[*].gas"), linked_paths(build, "medium.gas")
    (['responses[0].gas'], ['medium.gas'])
    >>> linked_paths(build, "nothing[*].gas"), linked_paths(build, "medium.none")
    ([], [])
    """
    if "[*]" not in pattern:
        return [pattern] if _held(build, pattern) is not MISSING else []
    head, tail = pattern.split("[*]", 1)
    held = _held(build, head)
    count = len(held) if isinstance(held, Sequence) and not isinstance(held, str) else 0
    found = (f"{head}[{index}]{tail}" for index in range(count))
    return [path for path in found if _held(build, path) is not MISSING]


def negated_value(value: object) -> object:
    """Return a value with its sign changed.

    Parameters
    ----------
    value : object
        A number, a quantity string, or :data:`MISSING`.

    Returns
    -------
    object
        The value negated, in its own unit.

    Examples
    --------
    >>> negated_value("10 cm"), negated_value("-10 cm"), negated_value(2.5)
    ('-10 cm', '10 cm', -2.5)
    >>> negated_value("0 mm")
    '0 mm'
    >>> negated_value(MISSING) is MISSING
    True
    """
    if isinstance(value, bool) or value is MISSING:
        return value
    if isinstance(value, int | float):
        return -value
    number, unit = split(str(value))
    if float(number) == 0:
        return value
    flipped = number[1:] if number.startswith("-") else f"-{number}"
    return f"{flipped} {unit}".strip()


def even_planes(count: int, length: object) -> list[object]:
    """Return evenly spaced saved planes along a length, ending at it.

    Parameters
    ----------
    count : int
        How many, the entrance not counted.
    length : object
        The length, a quantity string or a number.

    Returns
    -------
    list
        The positions, in the length's own unit.

    Examples
    --------
    >>> even_planes(2, "25 cm")
    ['12.5 cm', '25 cm']
    >>> even_planes(2, 1.0)
    [0.5, 1.0]
    """
    if isinstance(length, int | float):
        return [float(length) * index / count for index in range(1, count + 1)]
    number, unit = split(str(length))
    total = float(number)
    return [f"{total * index / count:.8g} {unit}" for index in range(1, count + 1)]


def field_value(row: Field, build: Mapping[str, object]) -> object:
    """Return what a sheet's row shows of a recipe.

    Parameters
    ----------
    row : Field
        The row.
    build : Mapping
        The recipe's ``[build]`` table.

    Returns
    -------
    object
        The value as the row shows it; a switch's is whether its entry is there.

    Examples
    --------
    >>> build = {"planes": ["1 m", "2 m"], "launch": "-5 cm", "list": [{"type": "A"}]}
    >>> field_value(Field("planes", planes_over="length"), build)
    2
    >>> field_value(Field("launch", negated=True), build)
    '5 cm'
    >>> field_value(Field("list", member="A"), build)
    True
    >>> field_value(Field("list", member="B"), build)
    False
    """
    held = _held(build, row.path)
    if row.member is not None:
        entries = held if isinstance(held, Sequence) else ()
        return any(
            isinstance(entry, Mapping) and entry.get("type") == row.member
            for entry in entries
        )
    if row.planes_over is not None:
        return len(held) if isinstance(held, Sequence) else MISSING
    return negated_value(held) if row.negated else held


def _switched(
    problem: ProblemClass,
    row: Field,
    on: bool,
    build: Mapping[str, object],
    example: Mapping[str, object],
) -> list[tuple[str, object]]:
    """Return the changes that switch one entry of a list on or off.

    An entry comes back from the class's example, which orders the list, with what the
    sheet's linked rows hold now.

    Parameters
    ----------
    problem : ProblemClass
        The class.
    row : Field
        The switch.
    on : bool
        Whether its entry is wanted.
    build : Mapping
        The recipe's ``[build]`` table.
    example : Mapping
        The ``[build]`` table of the class's example.

    Returns
    -------
    list of tuple
        Each path and its value.
    """
    held = _held(build, row.path)
    current = [dict(e) for e in (held if isinstance(held, Sequence) else ())]
    wanted = _held(example, row.path)
    source = [dict(e) for e in (wanted if isinstance(wanted, Sequence) else ())]
    kept = [e for e in current if e.get("type") != row.member]
    if on:
        kept += deepcopy([e for e in source if e.get("type") == row.member])
    order = [e.get("type") for e in source]
    kept.sort(
        key=lambda e: order.index(e.get("type")) if e.get("type") in order else 99
    )
    changed = {**build, row.path: kept}
    for section in problem.sections:
        for field in section.fields:
            value = _held(build, field.path)
            for pattern in field.also if value is not MISSING else ():
                if not pattern.startswith(f"{row.path}["):
                    continue
                for path in linked_paths(changed, pattern):
                    set_path(changed, path, value)
    changes: list[tuple[str, object]] = [(row.path, kept)]
    for path in row.goes_with:
        changes.append((path, _held(example, path) if on else MISSING))
    return changes


def field_changes(
    problem: ProblemClass,
    row: Field,
    value: object,
    build: Mapping[str, object],
    example: Mapping[str, object] | None = None,
) -> list[tuple[str, object]]:
    """Return what a sheet's row changes in a recipe.

    The object routes write some values several times, and a run's length as its last
    saved plane (#533), so one row can move several paths.

    Parameters
    ----------
    problem : ProblemClass
        The class.
    row : Field
        The row.
    value : object
        What it now shows.
    build : Mapping
        The recipe's ``[build]`` table.
    example : Mapping or None, optional
        The ``[build]`` table of the class's example, where a switch's entry comes from.

    Returns
    -------
    list of tuple
        Each path and its value, or :data:`MISSING`.

    Examples
    --------
    >>> problem = ProblemClass("c", "C", "", "radial", "e", (), (), "")
    >>> build = {"gas": "Ar", "list": [{"gas": "Ar"}], "planes": ["1 m", "2 m"]}
    >>> field_changes(problem, Field("gas", also=("list[*].gas",)), "He", build)
    [('gas', 'He'), ('list[0].gas', 'He')]
    >>> field_changes(problem, Field("length", moves_planes="planes"), "4 m", build)
    [('length', '4 m'), ('planes', ['2 m', '4 m'])]
    >>> build["length"] = "6 m"
    >>> field_changes(problem, Field("planes", planes_over="length"), 3, build)
    [('planes', ['2 m', '4 m', '6 m'])]
    >>> field_changes(problem, Field("x", negated=True), "5 cm", build)
    [('x', '-5 cm')]
    """
    if row.member is not None:
        return _switched(problem, row, bool(value), build, example or {})
    if row.planes_over is not None:
        length = _held(build, row.planes_over)
        return [(row.path, even_planes(int(value), length))]  # pyright: ignore[reportArgumentType]
    written = negated_value(value) if row.negated else value
    changes: list[tuple[str, object]] = [(row.path, written)]
    for pattern in row.also:
        changes += [(path, written) for path in linked_paths(build, pattern)]
    if row.moves_planes is not None and value is not MISSING:
        held = _held(build, row.moves_planes)
        count = len(held) if isinstance(held, Sequence) else 0
        if count:
            changes.append((row.moves_planes, even_planes(count, value)))
    return changes


def other_keywords(
    problem: ProblemClass, description: Mapping[str, object]
) -> list[Mapping[str, object]]:
    """Return the route's keywords a class's sheet does not show, for its Advanced.

    A sheet row on a path inside a keyword (``initial.waist``) leaves the keyword itself
    among these, since the rest of it is reachable only there.

    Parameters
    ----------
    problem : ProblemClass
        The class.
    description : Mapping
        The kernel's ``describe`` reply on its route.

    Returns
    -------
    list of Mapping
        The keywords' metadata, in the route's order.

    Examples
    --------
    >>> rows = (Field("radius"),)
    >>> problem = ProblemClass(
    ...     "c", "C", "", "capillary", "rdw", (Section("S", rows),), (), "β₂"
    ... )
    >>> described = {"parameters": [{"name": "radius"}, {"name": "rtol"}]}
    >>> other_keywords(problem, described)
    [{'name': 'rtol'}]
    """
    # A switch shows one entry of a list: the rest of the list is reachable only there.
    shown = {
        row.path
        for section in problem.sections
        for row in section.fields
        if row.member is None
    }
    entries: Sequence[Mapping[str, object]] = description["parameters"]  # pyright: ignore[reportAssignmentType]
    return [entry for entry in entries if entry.get("name") not in shown]
