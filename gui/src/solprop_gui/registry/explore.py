# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What the waveguide explorer plots, and in which units (design 7.4).

Every number comes from solprop's mode preview (``Recipe.modes``): the effective index,
``beta1`` to ``beta3``, the loss, the effective area and the nonlinear coefficient. The
explorer only shows them in a chosen unit (a group index is ``c beta1``, ``D`` is
``-2 pi c beta2 / lambda^2``, a loss in dB/m is ``10 log10(e)`` times the power
loss), and against a chosen axis: wavelength, frequency or photon energy.
"""

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = [
    "AXES",
    "DESIGN_GUIDE",
    "DESIGN_PULSE",
    "PIN_KEYS",
    "PIN_SEARCH",
    "PIN_TITLES",
    "QUANTITIES",
    "SWEEPS",
    "Axis",
    "Quantity",
    "axis_values",
    "design_keywords",
    "design_lines",
    "quantity_values",
    "sweep_values",
    "table_rows",
    "value_at",
]

_C: Final = 299_792_458.0
_H: Final = 6.626_070_15e-34
_E: Final = 1.602_176_634e-19


@dataclass(frozen=True)
class Quantity:
    """One quantity the explorer plots.

    Attributes
    ----------
    key : str
        Its identity.
    title : str
        Its name, with its unit.
    source : str
        The mode preview's field it is made from.
    scale : float
        What the field is multiplied by for the unit shown.
    """

    key: str
    title: str
    source: str
    scale: float


QUANTITIES: Final = (
    Quantity("neff", "n_eff - 1", "neff_minus_one", 1.0),
    Quantity("group_index", "Group index", "beta1", _C),
    Quantity("beta2", "β₂ (fs²/m)", "beta2", 1e30),
    Quantity("dispersion", "D (ps/(nm km))", "beta2", 1e6),
    Quantity("beta3", "β₃ (fs³/m)", "beta3", 1e45),
    Quantity("loss", "Loss (dB/m)", "alpha", 10.0 / math.log(10.0)),
    Quantity("aeff", "A_eff (µm²)", "aeff", 1e12),
    Quantity("gamma", "γ (1/(W km))", "gamma", 1e3),  # noqa: RUF001 - gamma
    Quantity("mismatch", "Phase mismatch Δβ (1/cm)", "phase_mismatch", 1e-2),
)
"""The quantities, in the order the explorer offers them."""


@dataclass(frozen=True)
class Axis:
    """One horizontal axis the explorer plots against.

    Attributes
    ----------
    key : str
        Its identity.
    title : str
        Its name, with its unit.
    """

    key: str
    title: str


AXES: Final = (
    Axis("wavelength", "Wavelength (nm)"),
    Axis("frequency", "Frequency (THz)"),
    Axis("energy", "Photon energy (eV)"),
)
"""The axes, in the order the explorer offers them."""

SWEEPS: Final = {
    "capillary": ("pressure", "radius", "gas", "temperature", "modes"),
}
"""The keywords a route's explorer can sweep, by route."""

PIN_TITLES: Final = {
    "zdw": "from a zero-dispersion wavelength",
    "rdw": "from a dispersive-wave wavelength",
    "soliton_order": "from a soliton order",
}
"""How a pin's target reads beside the pinned keyword."""

PIN_KEYS: Final = {
    "wavelength": ("Zero-dispersion wavelength", "m", "nm"),
    "rdw": ("Dispersive-wave wavelength", "m", "nm"),
    "order": ("Soliton order", None, None),
}
"""Each target key's label, its unit and the unit it is shown in; a test holds them
to solprop's targets."""

DESIGN_GUIDE: Final = ("pressure", "radius")
"""The guide keywords the design tool can solve for, in the order offered."""

DESIGN_PULSE: Final = ("energy", "peak_power")
"""The pulse sizes the design tool can solve for: the one the setup gives is used."""


PIN_SEARCH: Final = {
    "pressure": ["0.01 bar", "20 bar"],
    "radius": ["10 um", "500 um"],
}
"""Where a new pin on the guide or fill searches, until the user says otherwise."""


def axis_values(omega: np.ndarray, axis: str) -> np.ndarray:
    """Return angular frequencies on an axis.

    Parameters
    ----------
    omega : numpy.ndarray
        Angular frequencies, rad/s.
    axis : str
        An :data:`AXES` key.

    Returns
    -------
    numpy.ndarray
        Wavelengths in nm, frequencies in THz or photon energies in eV.

    Raises
    ------
    KeyError
        If the axis is not one of :data:`AXES`.

    Examples
    --------
    >>> import numpy as np
    >>> omega = np.array([2 * np.pi * 299_792_458.0 / 800e-9])
    >>> round(float(axis_values(omega, "wavelength")[0]), 6)
    800.0
    >>> round(float(axis_values(omega, "energy")[0]), 4)
    1.5498
    """
    frequency = np.asarray(omega, dtype=float) / (2.0 * np.pi)
    if axis == "wavelength":
        return _C / frequency * 1e9
    if axis == "frequency":
        return frequency * 1e-12
    if axis == "energy":
        return _H * frequency / _E
    raise KeyError(axis)


def quantity_values(
    mode: Mapping[str, object], omega: np.ndarray, quantity: Quantity
) -> np.ndarray | None:
    """Return one mode's quantity over the band, in the unit shown.

    Parameters
    ----------
    mode : Mapping
        The mode, as the kernel's ``modes`` request gives it.
    omega : numpy.ndarray
        The band's angular frequencies, rad/s.
    quantity : Quantity
        The quantity.

    Returns
    -------
    numpy.ndarray or None
        The values at each frequency; ``None`` where the mode has none (no ``gamma``
        without a fill's Kerr data).

    Examples
    --------
    >>> import numpy as np
    >>> omega = np.array([2 * np.pi * 299_792_458.0 / 1e-6])
    >>> mode = {"beta2": np.array([-1e-27]), "aeff": 2e-9, "gamma": None}
    >>> by_key = {q.key: q for q in QUANTITIES}
    >>> round(float(quantity_values(mode, omega, by_key["dispersion"])[0]), 6)
    1.883652
    >>> round(float(quantity_values(mode, omega, by_key["aeff"])[0]), 9)
    2000.0
    >>> quantity_values(mode, omega, by_key["gamma"]) is None
    True
    """
    value = mode.get(quantity.source)
    if value is None:
        return None
    data = np.broadcast_to(np.asarray(value, dtype=float), np.shape(omega))
    if quantity.key == "dispersion":
        wavelength = 2.0 * np.pi * _C / np.asarray(omega, dtype=float)
        data = -2.0 * np.pi * _C / wavelength**2 * data
    return np.asarray(data * quantity.scale)


def value_at(x: np.ndarray, y: np.ndarray, at: float) -> float | None:
    """Return a curve's value at one abscissa, by linear interpolation.

    Parameters
    ----------
    x : numpy.ndarray
        The abscissae, in either order.
    y : numpy.ndarray
        The values.
    at : float
        Where.

    Returns
    -------
    float or None
        The value; ``None`` outside the curve.

    Examples
    --------
    >>> import numpy as np
    >>> value_at(np.array([3.0, 2.0, 1.0]), np.array([30.0, 20.0, 10.0]), 1.5)
    15.0
    >>> value_at(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 5.0) is None
    True
    """
    order = np.argsort(x)
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    if len(xs) == 0 or not xs[0] <= at <= xs[-1]:
        return None
    return float(np.interp(at, xs, ys))


def sweep_values(text: str) -> list[str]:
    """Read a sweep's values: a list, with a unit written once at its end if wanted.

    Parameters
    ----------
    text : str
        The values, separated by commas or spaces, as ``"0.5, 1, 2 bar"`` or
        ``"He, Ne, Ar"``.

    Returns
    -------
    list of str
        Each value as a recipe writes it, a trailing unit given to every bare number.

    Examples
    --------
    >>> sweep_values("0.5, 1, 2 bar")
    ['0.5 bar', '1 bar', '2 bar']
    >>> sweep_values("100 um, 0.2 mm")
    ['100 um', '0.2 mm']
    >>> sweep_values("He Ne Ar")
    ['He', 'Ne', 'Ar']
    >>> sweep_values("")
    []
    """
    values: list[str] = []
    for word in text.replace(",", " ").split():
        if not _number(word) and values and _number(values[-1]):
            values[-1] = f"{values[-1]} {word}"  # a number's unit
        else:
            values.append(word)
    unit = values[-1].split(" ", 1)[1] if values and " " in values[-1] else None
    if unit is None:
        return values
    return [f"{value} {unit}" if _number(value) else value for value in values]


def _number(word: str) -> bool:
    """Return whether a word is a number.

    Parameters
    ----------
    word : str
        The word.

    Returns
    -------
    bool
        Whether ``float`` reads it.
    """
    try:
        float(word)
    except ValueError:
        return False
    return True


def table_rows(
    sets: Sequence[Mapping[str, object]], pump: float
) -> list[dict[str, object]]:
    """Return the readout table at the pump: one row per mode of the first set.

    Parameters
    ----------
    sets : Sequence of Mapping
        The mode sets, as the kernel's ``modes`` request gives them.
    pump : float
        The pump wavelength, m.

    Returns
    -------
    list of dict
        Each mode's ``label`` and, by quantity key, its value at the pump in the
        unit shown (``None`` where it has none, or the pump is outside the band).
    """
    first = next((entry for entry in sets if "modes" in entry), None)
    if first is None:
        return []
    omega = np.asarray(first["omega"], dtype=float)
    wavelength = axis_values(omega, "wavelength")
    rows: list[dict[str, object]] = []
    for mode in first["modes"]:  # pyright: ignore[reportGeneralTypeIssues]
        row: dict[str, object] = {"label": mode["label"]}
        for quantity in QUANTITIES:
            values = quantity_values(mode, omega, quantity)
            row[quantity.key] = (
                None if values is None else value_at(wavelength, values, pump * 1e9)
            )
        rows.append(row)
    return rows


def design_keywords(
    keywords: Sequence[Mapping[str, object]], given: Collection[str]
) -> tuple[list[Mapping[str, object]], Mapping[str, object] | None]:
    """Return the keywords the design tool solves for, as the kernel describes them.

    Parameters
    ----------
    keywords : Sequence of Mapping
        The route's keywords (``describe``), each with its ``pins``.
    given : Collection of str
        The keywords the setup gives, as value or pin.

    Returns
    -------
    tuple
        The pinnable guide keywords of :data:`DESIGN_GUIDE`, in its order, and the
        pulse size: ``peak_power`` where the setup gives it, else ``energy``;
        ``None`` where the route pins neither.

    Examples
    --------
    >>> keywords = [{"name": name, "pins": [1]} for name in
    ...             ("radius", "energy", "peak_power")] + [{"name": "gas", "pins": []}]
    >>> guide, pulse = design_keywords(keywords, {"peak_power"})
    >>> [entry["name"] for entry in guide], pulse["name"]
    (['radius'], 'peak_power')
    >>> design_keywords(keywords[3:], ())
    ([], None)
    """
    pinnable = {str(entry["name"]): entry for entry in keywords if entry.get("pins")}
    guide = [pinnable[name] for name in DESIGN_GUIDE if name in pinnable]
    pulse = "peak_power" if "peak_power" in given else "energy"
    return guide, pinnable.get(pulse)


_DESIGN_NUMBERS: Final = (
    ("soliton_order", "Soliton order N", 1.0, ""),
    ("beta2", "β₂ at the pump", 1e30, "fs²/m"),
    ("gamma", "γ", 1e3, "1/(W km)"),  # noqa: RUF001 - gamma
    ("peak_intensity", "Peak intensity", 1e-4, "W/cm²"),
    ("zdw", "ZDW", 1e9, "nm"),
    ("rdw", "Phase-matched RDW", 1e9, "nm"),
    ("fission_length", "Fission length", 1e2, "cm"),
    ("critical_power", "Critical power", 1e-9, "GW"),
)
"""The summary's design numbers the tool shows: key, title, scale and unit."""


def design_lines(numbers: Mapping[str, float]) -> list[str]:
    """Return the design numbers at a solved design, one line each.

    Parameters
    ----------
    numbers : Mapping
        The kernel's ``design`` numbers, SI.

    Returns
    -------
    list of str
        ``"<title>: <value> <unit>"`` for each number given.

    Examples
    --------
    >>> design_lines({"soliton_order": 3.0, "beta2": -7.23e-30})
    ['Soliton order N: 3', 'β₂ at the pump: -7.23 fs²/m']
    """
    return [
        f"{title}: {numbers[key] * scale:.4g} {unit}".rstrip()
        for key, title, scale, unit in _DESIGN_NUMBERS
        if key in numbers
    ]
