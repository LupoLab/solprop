# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Post-process step's kernel: a compressor chain applied to a saved plane.

The stages (the window's ``solprop_gui.model.compressor``) become elements of
:mod:`solprop.optics` and are applied in order to the plane's stored spectrum. A fixed
stage is the element as given; an auto stage is searched by solprop's compressor of
its kind (``compress_taylor``, ``compress_material``, ``compress_mirrors``,
``compress_prisms``, ``compress_gratings``) on the spectrum the stages before it leave
(``optics.stored_factor`` of their ``Chain``), one stage at a time. The pulse before,
after the whole chain and at the transform limit comes from ``get_power``; each
number here is solprop's.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from solprop import capabilities
from solprop.constants import C
from solprop.exceptions import SolpropError
from solprop.optics import (
    Chain,
    ChirpedMirror,
    GratingPair,
    Material,
    PrismGeometry,
    PrismPair,
    TaylorPhase,
    TransferFunction,
    brewster_angle,
    brewster_apex,
    chirped_mirrors,
    littrow_angle,
    stored_factor,
)
from solprop.output import Output, open_file
from solprop.physdata.gases import Gas
from solprop.physdata.glasses import glass_names
from solprop.processing import (
    compress_gratings,
    compress_material,
    compress_mirrors,
    compress_prisms,
    compress_taylor,
    get_power,
    grid_of,
)
from solprop.processing.compression import Compression
from solprop.processing.time import fwhm
from solprop.service.reduce import TIME_OVERSAMPLING
from solprop.service.results import band_of

__all__ = ["POSTPROCESS_HANDLERS", "catalogue", "compress_chain"]

Payload = Mapping[str, object]


def catalogue(_payload: Payload) -> dict[str, list[str]]:
    """Return the names a compressor's stages choose from.

    Parameters
    ----------
    _payload : Mapping
        Unused.

    Returns
    -------
    dict
        ``glasses`` (``glass_names``), ``gases`` (solprop's gases), ``media`` (both,
        glasses first) and ``mirrors`` (``chirped_mirrors``).
    """
    glasses = list(glass_names())
    gases = [gas.value for gas in Gas]
    return {
        "glasses": glasses,
        "gases": gases,
        "media": glasses + gases,
        "mirrors": list(chirped_mirrors()),
    }


def _number(values: Mapping[str, object], name: str) -> float:
    """Return a stage's value as a number."""
    value = values.get(name)
    if value is None:
        msg = f"the stage needs its {name.replace('_', ' ')}"
        raise SolpropError(msg)
    return float(value)  # pyright: ignore[reportArgumentType]


def _range(values: Mapping[str, object], name: str) -> tuple[float, float] | None:
    """Return a stage's search range, or none where it is left blank.

    - Returns: the range, ascending; ``None`` when both ends are blank.
    """
    low, high = values.get(f"{name}_min"), values.get(f"{name}_max")
    if low is None and high is None:
        return None
    if low is None or high is None:
        msg = f"the {name.replace('_', ' ')} range needs both its ends"
        raise SolpropError(msg)
    low, high = float(low), float(high)  # pyright: ignore[reportArgumentType]
    return min(low, high), max(low, high)


def _pressure(values: Mapping[str, object]) -> float | None:
    """Return a material's pressure, which only a gas takes.

    - Returns: the pressure in bar for a gas (one bar when left blank), else ``None``.
    """
    medium = str(values.get("medium"))
    if medium not in {gas.value for gas in Gas}:
        return None
    pressure = values.get("pressure")
    return 1.0 if pressure is None else float(pressure)  # pyright: ignore[reportArgumentType]


def _prisms(values: Mapping[str, object], wavelength: float) -> PrismPair:
    """Return a stage's prism pair: Brewster entry and apex at the reference.

    The pair works within its band (a wavelength range): outside it, rays of a
    broad spectrum miss the faces.

    - ``wavelength``: the reference wavelength, m.
    """
    glass = str(values.get("medium"))
    geometry = PrismGeometry(
        separation=_number(values, "separation"),
        insertion=_number(values, "insertion"),
        first_face_distance=_number(values, "first_face"),
        apex_angle=brewster_apex(glass, wavelength),
        incidence_angle=brewster_angle(glass, wavelength),
    )
    return PrismPair(
        glass,
        geometry,
        wavelength=wavelength,
        double_pass=bool(values.get("double_pass", True)),
        band=_range(values, "band"),
    )


def _gratings(values: Mapping[str, object], wavelength: float) -> GratingPair:
    """Return a stage's grating pair: first order, Littrow where no angle is given.

    - ``wavelength``: the reference wavelength, m.
    """
    density = _number(values, "line_density")
    incidence = values.get("incidence")
    angle = (
        littrow_angle(density, wavelength) if incidence is None else float(incidence)  # pyright: ignore[reportArgumentType]
    )
    return GratingPair(
        line_density=density,
        separation=_number(values, "separation"),
        incidence_angle=angle,
        wavelength=wavelength,
        double_pass=bool(values.get("double_pass", True)),
        band=_range(values, "band"),
    )


def element(stage: Mapping[str, object], wavelength: float) -> TransferFunction:
    """Return a fixed stage's optical element.

    Parameters
    ----------
    stage : Mapping
        ``kind`` and ``values``.
    wavelength : float
        The reference wavelength, m.

    Returns
    -------
    TransferFunction
        The element of :mod:`solprop.optics`.

    Raises
    ------
    solprop.exceptions.SolpropError
        For a value missing, or a kind not known.
    """
    kind = str(stage["kind"])
    values: Mapping[str, object] = stage["values"]  # pyright: ignore[reportAssignmentType]
    if kind == "taylor":
        coefficients = [0.0, 0.0] + [
            float(values.get(name) or 0.0)  # pyright: ignore[reportArgumentType]
            for name in ("gdd", "tod", "fod")
        ]
        return TaylorPhase(coefficients, wavelength)
    if kind == "material":
        return Material(
            str(values.get("medium")),
            _number(values, "thickness"),
            wavelength=wavelength,
            pressure=_pressure(values),
        )
    if kind == "mirrors":
        mirror = ChirpedMirror.lookup(str(values.get("mirror")))
        return mirror.bounces(int(_number(values, "reflections")))
    if kind == "prisms":
        return _prisms(values, wavelength)
    if kind == "gratings":
        return _gratings(values, wavelength)
    msg = f"no compressor stage of kind {kind!r}"
    raise SolpropError(msg)


def search(
    stage: Mapping[str, object], wavelength: float, **common: Any
) -> tuple[Compression, dict[str, object]]:
    """Search an auto stage with solprop's compressor of its kind.

    Parameters
    ----------
    stage : Mapping
        ``kind`` and ``values``.
    wavelength : float
        The reference wavelength, m.
    **common : Any
        The compressor's input and selection: ``grid``, ``spectrum``, ``modes`` and
        ``bandpass``.

    Returns
    -------
    tuple
        The :class:`~solprop.processing.compression.Compression` solprop found,
        and the values it found, by the stage's names.

    Raises
    ------
    solprop.exceptions.SolpropError
        For a value missing, or a kind not known.
    """
    kind = str(stage["kind"])
    values: Mapping[str, object] = stage["values"]  # pyright: ignore[reportAssignmentType]
    if kind == "taylor":
        found = compress_taylor(
            order=int(_number(values, "order")), wavelength=wavelength, **common
        )
        gdd, tod, fod = [*found.coefficients[2:], 0.0, 0.0, 0.0][:3]
        return found, {"gdd": gdd, "tod": tod, "fod": fod}
    if kind == "material":
        thickness = _range(values, "thickness")
        if thickness is None:
            msg = "an auto material needs its thickness range"
            raise SolpropError(msg)
        found = compress_material(
            medium=str(values.get("medium")),
            thickness_range=thickness,
            wavelength=wavelength,
            pressure=_pressure(values),
            **common,
        )
        return found, {"thickness": found.thickness}
    if kind == "mirrors":
        wedge = values.get("wedge")
        found = compress_mirrors(
            mirror=str(values.get("mirror")),
            max_reflections=int(_number(values, "max_reflections")),
            medium=None if wedge is None else str(wedge),
            thickness_range=None if wedge is None else _range(values, "wedge"),
            wavelength=wavelength,
            **common,
        )
        result: dict[str, object] = {"reflections": found.reflections}
        if not math.isnan(found.thickness):
            result["thickness"] = found.thickness
        return found, result
    if kind == "prisms":
        found = compress_prisms(
            pair=_prisms(values, wavelength),
            insertion_range=_range(values, "insertion"),
            separation_range=_range(values, "separation"),
            **common,
        )
        geometry = found.transfer.geometry  # pyright: ignore[reportAttributeAccessIssue]
        return found, {
            "separation": geometry.separation,
            "insertion": geometry.insertion,
        }
    if kind == "gratings":
        separation = _range(values, "separation")
        if separation is None:
            msg = "an auto grating pair needs its separation range"
            raise SolpropError(msg)
        found = compress_gratings(
            pair=_gratings(values, wavelength), separation_range=separation, **common
        )
        return found, {"separation": found.transfer.separation}  # pyright: ignore[reportAttributeAccessIssue]
    msg = f"no compressor stage of kind {kind!r}"
    raise SolpropError(msg)


def _pulse(t: np.ndarray, power: np.ndarray) -> dict[str, float]:
    """Return a pulse's width and peak, as solprop measures them.

    - ``t``: the time axis, s.
    - ``power``: the power, W.
    - Returns: ``fwhm`` (s) and ``peak`` (W).
    """
    return {"fwhm": fwhm(t, power), "peak": float(np.max(power))}


def compress_chain(payload: Payload) -> dict[str, object]:
    """Apply a compressor to a saved plane, searching its auto stages in turn.

    Parameters
    ----------
    payload : Mapping
        ``path``, ``index`` (the plane), ``stages`` (each ``kind``, ``auto`` and
        ``values``, as the window's ``solprop_gui.model.compressor.payload`` gives
        them), and optionally ``modes`` (``"sum"`` or a mode's index) and ``bandpass``
        (a wavelength band, m).

    Returns
    -------
    dict
        ``t`` (s); ``before``, ``after`` and ``limit``, the power (W) before the
        compressor, after it and at the transform limit; ``pulses``, their
        ``fwhm`` and ``peak`` by the same names; ``stages``, per stage its
        ``kind``, ``auto`` and, for a searched one, the ``values`` found, solprop's
        ``text``, ``quality``, ``success`` and ``message``; and ``wavelength``,
        the reference the elements are written about (the grid's).

    Raises
    ------
    solprop.exceptions.SolpropError
        For a free-space run, which solprop's compressors refuse, or a stage that
        cannot be built.
    """
    path = str(payload["path"])
    if capabilities.output(path).basis == "spatial":
        msg = (
            "solprop's compressors work on modal amplitudes; a free-space run's "
            "columns are field samples across the beam"
        )
        raise SolpropError(msg)
    index = int(payload["index"])  # pyright: ignore[reportArgumentType]
    chosen = payload.get("modes", "sum")
    selection = None if chosen == "sum" else int(chosen)  # pyright: ignore[reportArgumentType]
    band = band_of(payload)
    stages: Sequence[Mapping[str, object]] = payload.get("stages") or []  # pyright: ignore[reportAssignmentType]
    reports: list[dict[str, object]] = []
    elements: list[TransferFunction] = []
    with open_file(path) as handler:
        grid = grid_of(handler)
        wavelength = 2 * np.pi * C / grid.omega0
        stored = np.asarray(handler.fields("fft")[index])  # pyright: ignore[reportAttributeAccessIssue]
        for stage in stages:
            report: dict[str, object] = {
                "kind": stage["kind"],
                "auto": bool(stage.get("auto")),
            }
            if stage.get("auto"):
                current = (
                    stored * stored_factor(grid, Chain(elements))
                    if elements
                    else stored
                )
                found, values = search(
                    stage,
                    wavelength,
                    grid=grid,
                    spectrum=current,
                    modes=selection,
                    bandpass=band,
                )
                elements.append(found.transfer)
                report.update(
                    values=values,
                    text=repr(found),
                    quality=found.quality,
                    success=found.success,
                    message=found.message,
                )
            else:
                elements.append(element(stage, wavelength))
            reports.append(report)
        found_power = _powers(handler, index, selection, band, elements)
    return {**found_power, "stages": reports, "wavelength": wavelength}


def _powers(
    handler: Output,
    index: int,
    selection: int | None,
    band: tuple[float, float] | None,
    elements: Sequence[TransferFunction],
) -> dict[str, object]:
    """Return the pulse before, after the elements and at the transform limit.

    - ``selection``: one mode, or ``None`` for their sum.
    - Returns: ``t``, ``before``, ``after``, ``limit`` and ``pulses``.
    """
    modes = "sum" if selection is None else selection
    common = {
        "plane": index,
        "modes": modes,
        "bandpass": band,
        "oversampling": TIME_OVERSAMPLING,
    }
    t, before = get_power(handler, **common)  # pyright: ignore[reportArgumentType]
    _, after = get_power(
        handler,
        transfer=Chain(list(elements)) if elements else None,
        **common,  # pyright: ignore[reportArgumentType]
    )
    _, limit = get_power(handler, transform_limit=True, **common)  # pyright: ignore[reportArgumentType]
    traces = {"before": before, "after": after, "limit": limit}
    return {
        "t": np.asarray(t),
        **{name: np.asarray(trace) for name, trace in traces.items()},
        "pulses": {
            name: _pulse(np.asarray(t), trace) for name, trace in traces.items()
        },
    }


POSTPROCESS_HANDLERS: dict[str, Callable[[Payload], object]] = {
    "optics_catalogue": catalogue,
    "compress_chain": compress_chain,
}
"""The Post-process step's handlers, by request kind."""
