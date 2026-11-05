# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a problem class declares, and the readings of it that hold for every class."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from solprop.recipe.errors import RecipeError
from solprop.recipe.paths import Attribute, parse_path

__all__ = [
    "AnalysisPage",
    "Derived",
    "Field",
    "ProblemClass",
    "Section",
    "field_meta",
    "other_keywords",
    "page_refusal",
    "sheet_paths",
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
    """

    path: str
    label: str | None = None


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
    """

    key: str
    label: str
    unit: str = ""
    factor: float = 1.0
    prefixed: bool = True


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
        The kernel's ``describe_output`` reply on the file: its ``field_planes``,
        ``statistics_planes``, ``labels``, ``refused`` and ``ensemble``.

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
        "statistics": bool(capabilities.get("statistics_planes")),
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
