# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Record spatial accuracy gates without treating platform roundoff as physics."""

import math

from bounds import record_backend
from convergence_records import Column, RecordError, Study, load, path_of, record
from convergence_studies import BACKEND_EQUIVALENCE

SPATIAL_ACCURACY = Study(
    name="spatial_accuracy",
    title="spatial propagation: independently derived bound beside measurement",
    case={
        "geometries": "radial QDHT and Cartesian Fourier transforms",
        "oracles": "Bessel contour, Gaussian diffraction, energy balance, "
        "constant flow and matrix exponential",
        "units": "Each series states its physical units; "
        "relative errors are dimensionless.",
    },
    axis={"name": "case", "unit": "1", "label": "independent accuracy check"},
    columns=(
        Column(
            "deviation",
            "series units",
            "Every writer asserts its derived bound. Last-bit errors vary across "
            "machines; stored deviations are evidence rather than golden truth.",
        ),
        Column(
            "bound",
            "series units",
            "The original predeclared acceptance limit is fixed for this case. "
            "Every measurement must meet both this limit and its freshly derived "
            "local bound; rewriting records cannot increase the accepted error.",
            rtol=0,
        ),
        Column(
            "derived_bound",
            "series units",
            "The bound derived before this measurement, including local special "
            "functions and quadrature estimates. Record it at full precision; "
            "the test and writer enforce it independently of the fixed limit.",
        ),
    ),
)


def record_metric(request, series, axis, measured, bound, backend="numpy_ref"):
    """Enforce both predeclared limits and stamp their measured error."""
    series = f"{record_backend(backend)}: {series}"
    axis = str(axis)
    measured, derived = float(measured), float(bound)
    fixed = derived
    if path_of(SPATIAL_ACCURACY.name).exists():
        for previous in load(SPATIAL_ACCURACY.name)["rows"]:
            if (previous["series"], str(previous["axis"])) == (series, axis):
                fixed = float(previous["bound"])
                break
    if not (
        math.isfinite(derived)
        and math.isfinite(fixed)
        and derived > 0
        and fixed > 0
        and 0 <= measured <= min(fixed, derived)
    ):
        raise RecordError(
            f"{series} {axis}: error {measured!r}, fixed limit {fixed!r}, "
            f"locally derived bound {derived!r}"
        )
    record(
        SPATIAL_ACCURACY,
        request=request,
        backend=record_backend(backend),
        rows=[
            {
                "series": series,
                "axis": axis,
                "deviation": measured,
                "bound": fixed,
                "derived_bound": derived,
            }
        ],
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
