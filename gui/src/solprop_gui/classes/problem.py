# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a problem class declares, and the readings of it that hold for every class."""

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from solprop.recipe.errors import RecipeError
from solprop.recipe.paths import Attribute, parse_path

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
    "field_meta",
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
    """

    path: str
    label: str | None = None
    choices: str | None = None


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
    """

    key: str
    title: str
    needs: tuple[str, ...] = ()
    operation: str | None = None


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
    shown = set(sheet_paths(problem))
    entries: Sequence[Mapping[str, object]] = description["parameters"]  # pyright: ignore[reportAssignmentType]
    return [entry for entry in entries if entry.get("name") not in shown]
