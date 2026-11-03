# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A compressor: a chain of linear optical elements after the run, some searched.

The Post-process step builds a :class:`Stage` list: a Taylor spectral phase, a
material (a glass, or a gas such as an air path), chirped mirrors (optionally with
wedges), a prism pair or a grating pair, each of :mod:`solprop.optics`. A stage is
either **fixed**, its values as given, or **auto**: solprop's compressor of that kind
(``processing.compress_taylor`` and the rest) searches its free values for the highest
peak power. The kernel applies the stages in order and searches one auto stage at a
time, each on the pulse the stages before it leave (:mod:`.service.postprocess`).

Values are held in SI (s, m, bar, lines per metre); :data:`FIELDS` says how each is
shown. The starting values of a new stage are the user guide's examples
(``docs/guide/workflows/compression.md``), to be changed.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

__all__ = [
    "FIELDS",
    "KINDS",
    "Field",
    "Stage",
    "found_values",
    "new_stage",
    "payload",
]

KINDS: Final = {
    "taylor": "Spectral phase (GDD, TOD, FOD)",
    "material": "Material (glass or gas)",
    "mirrors": "Chirped mirrors",
    "prisms": "Prism pair",
    "gratings": "Grating pair",
}
"""The kinds of stage, with their names."""


@dataclass(frozen=True)
class Field:
    """One value of a stage, as the editor shows it.

    Attributes
    ----------
    name : str
        Its key in the stage's values.
    label : str
        Its label.
    unit : str
        Its display unit; empty for a choice, a count or a switch.
    scale : float
        The display unit's size in SI.
    form : {'number', 'count', 'choice', 'switch'}
        How it is entered: a number in the unit, a whole number, a name from the
        kernel's catalogue, or a switch.
    when : {'fixed', 'auto', 'both'}
        When it is shown: for a fixed stage, an auto one, or both (an auto stage's
        starting point).
    choices : str
        For a choice, the catalogue it is drawn from (``"glasses"``, ``"media"``,
        ``"mirrors"``), or none.
    optional : bool
        Whether it may be left blank (a wedge, a search range not wanted).
    """

    name: str
    label: str
    unit: str = ""
    scale: float = 1.0
    form: str = "number"
    when: str = "both"
    choices: str = ""
    optional: bool = False

    def shown(self, auto: bool) -> bool:
        """Return whether the field is shown for a stage that is auto or fixed.

        Parameters
        ----------
        auto : bool
            Whether the stage is searched.

        Returns
        -------
        bool
            True when it applies.

        Examples
        --------
        >>> Field("order", "Highest order", when="auto").shown(False)
        False
        """
        return self.when == "both" or (self.when == "auto") == auto


_FS = 1e-15
_MM = 1e-3

FIELDS: Final[dict[str, tuple[Field, ...]]] = {
    "taylor": (
        Field("gdd", "GDD", "fs²", _FS**2, when="fixed"),
        Field("tod", "TOD", "fs³", _FS**3, when="fixed"),
        Field("fod", "FOD", "fs⁴", _FS**4, when="fixed"),
        Field("order", "Highest order", form="count", when="auto"),
    ),
    "material": (
        Field("medium", "Material", form="choice", choices="media"),
        Field("pressure", "Pressure (a gas)", "bar", optional=True),
        Field("thickness", "Thickness", "mm", _MM, when="fixed"),
        Field("thickness_min", "Thinnest", "mm", _MM, when="auto"),
        Field("thickness_max", "Thickest", "mm", _MM, when="auto"),
    ),
    "mirrors": (
        Field("mirror", "Mirror", form="choice", choices="mirrors"),
        Field("reflections", "Reflections", form="count", when="fixed"),
        Field("max_reflections", "Most reflections", form="count", when="auto"),
        Field(
            "wedge",
            "Wedges of",
            form="choice",
            when="auto",
            choices="glasses",
            optional=True,
        ),
        Field("wedge_min", "Wedge, thinnest", "mm", _MM, when="auto", optional=True),
        Field("wedge_max", "Wedge, thickest", "mm", _MM, when="auto", optional=True),
    ),
    "prisms": (
        Field("medium", "Glass", form="choice", choices="glasses"),
        Field("separation", "Separation R", "cm", 1e-2),
        Field("insertion", "Insertion H", "mm", _MM),
        Field("first_face", "First-face distance", "mm", _MM),
        Field("double_pass", "Double pass", form="switch"),
        Field("band_min", "Working band, from", "nm", 1e-9, optional=True),
        Field("band_max", "Working band, to", "nm", 1e-9, optional=True),
        Field("separation_min", "R, from", "cm", 1e-2, when="auto", optional=True),
        Field("separation_max", "R, to", "cm", 1e-2, when="auto", optional=True),
        Field("insertion_min", "H, from", "mm", _MM, when="auto", optional=True),
        Field("insertion_max", "H, to", "mm", _MM, when="auto", optional=True),
    ),
    "gratings": (
        Field("line_density", "Lines", "per mm", 1e3),
        Field("separation", "Separation", "mm", _MM),
        Field(
            "incidence",
            "Incidence (blank: Littrow)",
            "°",
            0.017453292519943295,
            optional=True,
        ),
        Field("double_pass", "Double pass", form="switch"),
        Field("band_min", "Working band, from", "nm", 1e-9, optional=True),
        Field("band_max", "Working band, to", "nm", 1e-9, optional=True),
        Field("separation_min", "Separation, from", "mm", _MM, when="auto"),
        Field("separation_max", "Separation, to", "mm", _MM, when="auto"),
    ),
}
"""Each kind's values, in the order the editor shows them."""

_STARTS: Final[dict[str, dict[str, object]]] = {
    "taylor": {"gdd": 0.0, "tod": 0.0, "fod": 0.0, "order": 3},
    "material": {
        "medium": "fused silica",
        "pressure": None,
        "thickness": 1e-3,
        "thickness_min": 0.0,
        "thickness_max": 5e-3,
    },
    "mirrors": {
        "mirror": "PC70",
        "reflections": 4,
        "max_reflections": 10,
        "wedge": None,
        "wedge_min": 0.0,
        "wedge_max": 4e-3,
    },
    "prisms": {
        "medium": "fused silica",
        "separation": 0.6,
        "insertion": 0.015,
        "first_face": 0.003,
        "double_pass": True,
        "separation_min": None,
        "separation_max": None,
        "insertion_min": 0.01,
        "insertion_max": 0.02,
        "band_min": 650e-9,
        "band_max": 1050e-9,
    },
    "gratings": {
        "line_density": 1.2e6,
        "separation": 0.05,
        "incidence": None,
        "double_pass": True,
        "separation_min": 0.01,
        "separation_max": 0.1,
        "band_min": 650e-9,
        "band_max": 1000e-9,
    },
}


@dataclass(frozen=True)
class Stage:
    """One element of the compressor.

    Attributes
    ----------
    kind : str
        One of :data:`KINDS`.
    auto : bool
        Whether solprop searches its free values.
    values : dict
        Its values, in SI, by :data:`FIELDS` name.
    """

    kind: str
    auto: bool = False
    values: dict[str, object] = field(default_factory=dict)

    def with_value(self, name: str, value: object) -> Stage:
        """Return the stage with one value changed.

        Parameters
        ----------
        name : str
            The value's name.
        value : object
            The new value, in SI; ``None`` for blank.

        Returns
        -------
        Stage
            The changed stage.

        Examples
        --------
        >>> new_stage("taylor").with_value("gdd", -1e-28).values["gdd"]
        -1e-28
        """
        return replace(self, values={**self.values, name: value})

    def describe(self) -> str:
        """Return a one-line description, for the stage list.

        Returns
        -------
        str
            Its kind, whether it is searched, and its main value.

        Examples
        --------
        >>> new_stage("mirrors", auto=True).describe()
        'Chirped mirrors, auto: PC70, up to 10 reflections'
        >>> new_stage("material").describe()
        'Material (glass or gas): 1 mm of fused silica'
        """
        values = self.values
        text = {
            "taylor": (
                f"to order {values.get('order')}"
                if self.auto
                else f"GDD {_shown(values.get('gdd'), 1e-30)} fs²"
            ),
            "material": (
                f"{values.get('medium')}"
                if self.auto
                else f"{_shown(values.get('thickness'), 1e3)} mm of "
                f"{values.get('medium')}"
            ),
            "mirrors": (
                f"{values.get('mirror')}, up to {values.get('max_reflections')} "
                "reflections"
                if self.auto
                else f"{values.get('mirror')}, {values.get('reflections')} reflections"
            ),
            "prisms": (
                f"{values.get('medium')}, R {_shown(values.get('separation'), 100)} cm"
            ),
            "gratings": f"{_shown(values.get('line_density'), 1e-3)} lines/mm, "
            f"{_shown(values.get('separation'), 1e3)} mm",
        }[self.kind]
        return f"{KINDS[self.kind]}{', auto' if self.auto else ''}: {text}"


def _shown(value: object, scale: float) -> str:
    """Return a value in a display unit, compactly.

    Parameters
    ----------
    value : object
        The value in SI, or ``None``.
    scale : float
        What one SI unit is in the display unit.

    Returns
    -------
    str
        The number, or a dash for none.

    Examples
    --------
    >>> _shown(0.0015, 1e3), _shown(None, 1.0)
    ('1.5', '-')
    """
    if not isinstance(value, int | float):
        return "-"
    return f"{value * scale:.4g}"


def new_stage(kind: str, *, auto: bool = False) -> Stage:
    """Return a new stage of a kind, with the guide's starting values.

    Parameters
    ----------
    kind : str
        One of :data:`KINDS`.
    auto : bool, optional
        Whether it is searched.

    Returns
    -------
    Stage
        The stage.

    Raises
    ------
    KeyError
        For an unknown kind.

    Examples
    --------
    >>> new_stage("gratings").values["line_density"]
    1200000.0
    """
    return Stage(kind, auto, dict(_STARTS[kind]))


def payload(stages: Sequence[Stage]) -> list[dict[str, object]]:
    """Return the stages as the kernel takes them.

    Parameters
    ----------
    stages : Sequence of Stage
        The compressor, in order.

    Returns
    -------
    list of dict
        Each stage's ``kind``, ``auto`` and ``values``, plain data.

    Examples
    --------
    >>> payload([new_stage("taylor", auto=True)])[0]["auto"]
    True
    """
    return [
        {"kind": stage.kind, "auto": stage.auto, "values": dict(stage.values)}
        for stage in stages
    ]


def found_values(stage: Stage, found: Mapping[str, object]) -> list[Stage]:
    """Return an auto stage made fixed at the values solprop found for it.

    Parameters
    ----------
    stage : Stage
        The stage that was searched.
    found : Mapping
        The kernel's report of it: ``values``, the found values by name.

    Returns
    -------
    list of Stage
        The stage, fixed, with the found values in place: mirrors searched with
        wedges become the mirrors and the wedges' material. A stage that was not
        searched is returned as it is.

    Examples
    --------
    >>> stage = new_stage("material", auto=True)
    >>> (fixed,) = found_values(stage, {"values": {"thickness": 2e-3}})
    >>> fixed.auto, fixed.values["thickness"]
    (False, 0.002)
    >>> mirrors = new_stage("mirrors", auto=True).with_value("wedge", "caf2")
    >>> [s.kind for s in found_values(mirrors, {"values": {"reflections": 3,
    ...     "thickness": 1e-3}})]
    ['mirrors', 'material']
    """
    if not stage.auto:
        return [stage]
    values: Mapping[str, object] = found.get("values") or {}  # pyright: ignore[reportAssignmentType]
    fixed = [Stage(stage.kind, False, {**stage.values, **values})]
    wedge = stage.values.get("wedge")
    if stage.kind == "mirrors" and wedge is not None:
        material = new_stage("material")
        thickness = values.get("thickness", material.values["thickness"])
        fixed.append(
            material.with_value("medium", wedge).with_value("thickness", thickness)
        )
    return fixed
