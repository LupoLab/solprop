# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The convergence studies that more than one test file measures part of.

A study whose rungs are all in one file keeps its :class:`~convergence_records.Study`
beside them, so that the bound and the measured number sit together. The ones here are
split -- the backend-equivalence ladder is six functions in ``tests/backends`` plus one
in ``tests/driver``, the M2 studies are split across tiers -- and a shared definition is
the only way their rows can land in one record and one figure. Each rung still states
its own bound and its own measurement where it is measured.
"""

from convergence_records import Column, Study

# `tests/backends/test_numba_cpu.py` derives every bound from a summation length and
# then measures it. What the record holds is that pair, per rung, and
# what the figure draws is the distance between them. The deviations are two or three
# ulps: their last digits are not a platform-independent quantity at all -- another
# BLAS sums a dot product in another order -- so the tolerance is a factor of two,
# which still catches any change worth the name, because a real one crosses the bound
# the test itself asserts. A rung whose deviation is exactly zero is not here: a log
# axis cannot draw a zero.
BACKEND_EQUIVALENCE = Study(
    name="backend_equivalence",
    title=(
        "compiled backends against the numpy_ref oracle, bound and measurement per rung"
    ),
    case={
        "backends": (
            "numba_cpu and installed plugins against numpy_ref, the reference oracle"
        ),
        "metric": "peak-normalised deviation of the final spectrum",
        "why_not_bitwise": "compiled arithmetic and FFT plans may associate operations "
        "differently from NumPy",
    },
    axis={"name": "rung", "unit": "1", "label": "ladder rung"},
    columns=(
        Column(
            name="deviation",
            unit="1 (peak-normalised)",
            why="a factor of two: these are two or three ulps, whose last digits "
            "depend on the order a BLAS sums in, while a real change crosses the "
            "bound the rung's own test asserts",
            rtol=1.0,
            atol=1e-17,
        ),
        Column(
            name="bound",
            unit="1 (peak-normalised)",
            why="the rung's derived bound, an input rather than a measurement, so it "
            "is compared exactly and a quietly loosened bound is a failure",
            rtol=0.0,
        ),
        Column(
            name="order",
            unit="1",
            why="where the rung sits along the ladder, so the figure can order rungs "
            "that have no numeric axis; compared exactly",
            rtol=0.0,
        ),
    ),
)
