# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hold spatial accuracy checks to their derived bounds; record each series' margin."""

import math

from bounds import record_backend
from convergence_records import Column, RecordError, Study, record, writing
from convergence_studies import BACKEND_EQUIVALENCE

SPATIAL_ACCURACY = Study(
    name="spatial_accuracy",
    title="spatial propagation: the largest fraction of its derived bound per series",
    case={
        "geometries": "radial QDHT and Cartesian Fourier transforms",
        "oracles": "Bessel contour, Gaussian diffraction, energy balance, "
        "constant flow and matrix exponential",
        "units": "Each series states the units of its measurement; the fraction "
        "is dimensionless.",
    },
    axis={"name": "case", "unit": "1", "label": "the series' largest case"},
    columns=(
        Column(
            "fraction",
            "1",
            "The largest measured error over its independently derived bound among "
            "the series' cases; each writer asserts every case below one. Last-bit "
            "errors vary across machines, so it is recorded for the figure and not "
            "compared.",
        ),
    ),
)

_LARGEST: dict[str, float] = {}
"""Each series' largest fraction so far in a write session."""


def record_metric(request, series, axis, measured, bound, backend="numpy_ref"):
    """Assert a measured error within its derived bound; record its series' margin.

    The bound is the one the test derives for this case before measuring, and it is
    the whole criterion. The record keeps one row per series, the largest fraction of
    its bound any case reached, which ``docs/dev/accuracy.md`` section 10 draws; an
    ordinary run checks only that its series has a row.
    """
    measured, bound = float(measured), float(bound)
    if not (math.isfinite(bound) and bound > 0 and 0 <= measured <= bound):
        raise RecordError(
            f"{series} {axis}: error {measured!r} against the derived bound {bound!r}"
        )
    series = f"{record_backend(backend)}: {series}"
    fraction = measured / bound
    if writing():
        fraction = max(_LARGEST.get(series, 0.0), fraction)
        _LARGEST[series] = fraction
    record(
        SPATIAL_ACCURACY,
        request=request,
        backend=record_backend(backend),
        rows=[{"series": series, "axis": "largest", "fraction": fraction}],
    )


def record_equivalence(request, case, backend, rung, measured, bound):
    """Record an asserted spatial ladder in the common backend comparison study."""
    geometry, family = case
    record(
        BACKEND_EQUIVALENCE,
        request=request,
        backend=record_backend(backend),
        rows=[
            {
                "series": f"{geometry} ({record_backend(backend)}, {family})",
                "axis": str(rung),
                "deviation": float(measured),
                "bound": float(bound),
                "order": 20 if geometry == "radial" else 21,
            }
        ],
    )
