# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Records of host computations: their structure exact, their numbers to a bound.

A record (a run's hashed metadata, an entry point's configuration) used to be pinned
by one SHA-256 of its JSON. That hash also covered the last bits of the numbers in it,
which another platform's libm or FFT moves (SEP 25, class D). The record is split
instead: its structure -- keys, strings, integers, flags and where every number sits --
is identity metadata and is hashed exactly; its numbers are stored as an array and
compared to :data:`RECORD_RTOL`, each with its array's FFT floor.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from bounds import fft_bound

RECORD_RTOL = 1e-11
"""Agreement asked of each number in a run record with its stored value, relative.

The record's numbers are host computations: the configuration's own literals, and a
few derived values (material densities and weights, an ionisation threshold). The
largest departure a change of libm, BLAS or compiler can make is the threshold field's,
found by bisection to a relative bracket of 1e-12 (``optmat_rates``); a rounding that
flips one bisection decision moves it within that bracket. The rest are a few hundred
correctly rounded operations, below 1e-13. Fixed from that bracket with a factor of
five to spare, and far below any deliberate change to a configured number.
"""


def record_skeleton(value: object) -> tuple[object, list[float], list[float]]:
    """Split a record into its structure, its floating-point numbers and their floors.

    Every float is replaced by a placeholder in a copy and returned in walk order: the
    structure (keys, strings, integers, flags, the position of every number) is
    identity metadata and is hashed exactly; the numbers are compared to
    :data:`RECORD_RTOL`. A number inside a stored array also gets an absolute floor,
    returned beside it: :func:`bounds.fft_bound` of the array's length times the
    array's peak. A record carries computed spectra (the launch's), whose empty wings
    hold FFT rounding of order 1e-16 of the peak that another platform's transform
    moves; relative to the wing's own tiny values that is anything at all.
    """
    numbers: list[float] = []
    floors: list[float] = []

    def flat(values: list) -> list[float]:
        found: list[float] = []
        for child in values:
            if isinstance(child, float):
                found.append(abs(child))
            elif isinstance(child, list):
                found.extend(flat(child))
        return found

    def peak_floor(values: list) -> float:
        # Over the whole (possibly nested) array: a stored array is a nested list.
        floats = flat(values)
        return fft_bound(len(floats)) * max(floats) if len(floats) > 1 else 0.0

    def walk(item: object, floor: float | None = None) -> object:
        if isinstance(item, float):
            numbers.append(item)
            floors.append(0.0 if floor is None else floor)
            return "<float>"
        if isinstance(item, dict):
            # A complex array is stored as its real and imaginary parts; its rounding
            # scales with the whole field, so both parts take the joint peak's floor.
            parts = [
                item[key] for key in ("__ndarray__", "__imaginary__") if key in item
            ]
            joint = None
            if len(parts) == 2 and all(isinstance(part, list) for part in parts):
                joint = max(peak_floor(part) for part in parts)
            return {key: walk(child, joint) for key, child in item.items()}
        if isinstance(item, list):
            own = peak_floor(item)
            inner = own if floor is None else max(floor, own)
            return [walk(child, inner) for child in item]
        return item

    return walk(value), numbers, floors


def skeleton_hash(skeleton: object) -> str:
    """Return the SHA-256 of a record's structure, as canonical JSON."""
    text = json.dumps(skeleton, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_record_numbers(numbers, floors, stored: Path) -> None:
    """Compare a record's numbers with the array stored at `stored`.

    Non-finite numbers (an unset limit) are identity and match exactly; every finite
    number is within :data:`RECORD_RTOL` of its stored value plus its floor.
    """
    with np.load(stored) as data:
        expected = data["numbers"]
    numbers = np.asarray(numbers, dtype=float)
    floors = np.asarray(floors, dtype=float)
    assert expected.shape == numbers.shape
    finite = np.isfinite(expected)
    assert np.array_equal(finite, np.isfinite(numbers))
    assert np.array_equal(expected[~finite], numbers[~finite], equal_nan=True)
    limit = RECORD_RTOL * np.abs(expected[finite]) + floors[finite]
    outside = np.abs(numbers[finite] - expected[finite]) > limit
    assert not outside.any(), f"{int(outside.sum())} numbers outside their bound"


def without_sep21_absorber(
    record: dict, *, bare: bool = False, spatial: bool = False
) -> dict:
    """Return a hashed record as it was before output schema 27 (SEP 21).

    Schema 26 records ``Absorber.spatial``; the projection drops it. A run that declared
    no absorber recorded ``None`` before and now records the bare boundaries that stand
    for it (``tests/legacy_geometry.py``); with `bare` the projection restores ``None``.
    A spatial run that named no collar scheme now records ``"split_step"``, its route's
    default; with `spatial` the projection restores the former ``"nonlinear"``.

    Parameters
    ----------
    record : dict
        A record from ``RunMetadata.hashed``.
    bare : bool, optional
        Whether the run's absorber stands for the former ``None``.
    spatial : bool, optional
        Whether the run is a spatial one that named no collar scheme.

    Returns
    -------
    dict
        The projected record; `record` is not modified.
    """
    configuration = dict(record["configuration"])
    collar = configuration.get("absorber")
    if bare:
        configuration["absorber"] = None
    elif isinstance(collar, dict):
        configuration["absorber"] = {
            key: value for key, value in collar.items() if key != "spatial"
        }
    if spatial and configuration.get("absorber_scheme") == "split_step":
        configuration["absorber_scheme"] = "nonlinear"
    return {**record, "configuration": configuration}
