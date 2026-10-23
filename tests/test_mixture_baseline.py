# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hold the translated M2 problem to its stored record.

A compatibility witness, not a physical oracle: every array the reduced M2 case
translates to is compared with the value it had when it was recorded, so that a change
to the translation is seen even where no propagation test reaches it. The record was a
SHA-256 per array until SEP 25 (``data/mixture_single_component_baseline.json``, in the
git history with its provenance); it is now the arrays themselves, stored by
``ndarrays_regression`` beside this test and recaptured from the state those digests
certified. Each kind of leaf is compared by what its computation promises:

- integer and boolean arrays, and every shape, exactly (structure);
- arrays sampled at fixed points (axes, windows, the transverse transform, the rate
  table, the launch, the tableau) to :data:`HOST_BOUND` of their peak: the same host
  computation, differing between platforms by libm and BLAS rounding alone;
- the adaptive representations (piecewise Chebyshev series and the low-rank parameter
  expansions) by their values at fixed positions along the fibre, to the accuracy their
  builders promise: an adaptive builder may choose other intervals and degrees after a
  rounding change, so its coefficients and breakpoints are not compared at all.
"""

import dataclasses
import math
import warnings

import numpy as np

from driver.test_prop_capillary import reduced_m2_simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.linear import Frame
from solprop.translate.chebyshev import DEFAULT_TOL, PiecewiseChebyshev
from solprop.translate.expansion import ParameterExpansion
from solprop.translate.ionisation import RateTable
from solprop.translate.linop import MAX_TOL, _dispersion_phi

HOST_BOUND = 1e-12
"""Departure allowed a fixed-sample array, as a fraction of its peak.

Each value is a host computation of at most about a thousand correctly rounded or
one-ulp library operations (a rate and its logarithm, a Bessel quadrature weight, a
Gaussian and its normalising sum, a spline solve on a uniform table), so ``gamma(1000)``
of about 2.2e-13 of its peak, with a factor of four for the table's conditioning.
"""

CHEBYSHEV_BOUND = 20 * DEFAULT_TOL
"""Departure allowed the values of a piecewise Chebyshev series, of their peak.

Each build holds the function to ``DEFAULT_TOL`` of its value scale, or to the sampling
noise floor of its own node positions (ADR-040), which exceeds that only where the value
is far below its peak; two builds differ by twice it, with a factor of ten to spare.
"""

PHI_BOUND = 2 * MAX_TOL
"""Departure allowed the dispersion exponent's values, of their peak.

The operator's expansion is built to its requested tolerance or, when the interval
budget runs out, relaxed to at most ``MAX_TOL``; two builds differ by twice that.
"""

POSITIONS = 33
"""Positions along the fibre, ends included, at which a scalar series is sampled."""

SPECTRAL_POSITIONS = 5
"""Positions at which a representation with a frequency axis is sampled: fewer, since
each holds a whole spectrum, and the record stays small."""

HISTORICAL_FIELDS = {
    # Added after the record (kernel schema 23): the norm's mask.
    "norm_mask",
    "node_weights",
    "species_densities",
    "kerr_density",
    "plasma_densities",
    "plasma_preionisation",
    "raman_densities",
    # Kernel schema 25: the plasma column's running-integral rule.
    "plasma_integral",
    # SEP 27: the analytic collar derivative has its own independent tests.
    "t_window_fine_derivative",
    # SEP 23: the statistics' table of transverse-wavevector regions.
    "spectral_regions",
}
"""Fields the problem gained after the record was made, outside its projection."""

REPRESENTATION_FIELDS = {"phi_breakpoints", "phi_degrees"}
"""The operator's own adaptive intervals and degrees: compared through its values."""


def sampled(series: PiecewiseChebyshev | ParameterExpansion) -> np.ndarray:
    """Return a representation's values at fixed positions, its ends included."""
    low, high = series.z_span
    count = POSITIONS if math.prod(series.value_shape) == 1 else SPECTRAL_POSITIONS
    return np.stack([np.asarray(series(z)) for z in np.linspace(low, high, count)])


def rate_midpoints(table):
    """Return a rate table's spline values at every interval's midpoint, ``log W``."""
    t = 0.5 * table.d_log_field
    powers = t ** np.arange(table.coefficients.shape[0] - 1, -1, -1)
    return powers @ table.coefficients


def rate_leaves(table, prefix, seen):
    """Yield a rate table's leaves, its spline by value.

    A spline's coefficients amplify its samples' last bits by its knot spacing cubed:
    another platform's log and exp move the cubic terms by 1e-6 relative, its values
    by rounding. The table is compared by its values at every interval's midpoint, and
    its other fields as they are.
    """
    for name, child in historical_fields(table):
        if name == "coefficients":
            yield f"{prefix}.coefficients(mid)", rate_midpoints(table), "host"
        else:
            yield from leaves(child, f"{prefix}.{name}", seen)


def leaves(value, prefix="", seen=None):
    """Walk translated data in its historical projection, yielding comparable leaves.

    Yields ``(path, array, kind)``: ``kind`` is ``"exact"`` for an integer or boolean
    array, ``"host"`` for a fixed-sample float array, ``"chebyshev"`` and ``"phi"``
    for a representation's sampled values.
    """
    seen = set() if seen is None else seen
    if isinstance(value, np.ndarray):
        yield prefix, value, "exact" if value.dtype.kind in "biu" else "host"
    elif isinstance(value, PiecewiseChebyshev):
        yield f"{prefix}(z)", sampled(value), "chebyshev"
    elif isinstance(value, RateTable):
        if id(value) not in seen:
            seen.add(id(value))
            yield from rate_leaves(value, prefix, seen)
    elif isinstance(value, ParameterExpansion):
        kind = "phi" if prefix.endswith("phi_expansion") else "chebyshev"
        yield f"{prefix}(z)", sampled(value), kind
    elif id(value) not in seen:
        seen.add(id(value))
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for name, child in historical_fields(value):
                yield from leaves(child, f"{prefix}.{name}", seen)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                yield from leaves(item, f"{prefix}[{index}]", seen)
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from leaves(item, f"{prefix}.{key}", seen)


def historical_fields(value):
    """Yield a dataclass's ``(name, value)`` pairs at their schema-5 record paths."""
    for field in dataclasses.fields(value):
        name = field.name
        if name in HISTORICAL_FIELDS or name in REPRESENTATION_FIELDS:
            continue
        if name == "density" and type(value).__name__ in {
            "KerrCoefficients",
            "PlasmaChannel",
            "RamanCoefficients",
        }:
            continue
        child = getattr(value, name)
        if name in ("plasma_channels", "raman_terms", "ionisation_table"):
            # Schema 6 wraps the existing leaves in channel tuples. Compare their
            # contents at their schema-5 paths.
            assert len(child) <= 1
            child = child[0] if child else None
            name = {
                "plasma_channels": "plasma_coefficients",
                "raman_terms": "raman_coefficients",
            }.get(name, name)
        yield name, child


def test_m2_problem_keeps_every_recorded_array(ndarrays_regression):
    simulation = reduced_m2_simulation(backend="numba_cpu")
    with warnings.catch_warnings(record=True) as diagnostics:
        warnings.simplefilter("always", MaterialRangeWarning)
        problem = simulation.translate()
    # This witness intentionally uses Luna's broad helium band. Preserve and assert
    # that existing diagnostic instead of changing its source model or spectral span.
    assert diagnostics
    assert all(issubclass(item.category, MaterialRangeWarning) for item in diagnostics)
    # Hold the record in its original convention. The power weight adds log-weight
    # terms and rescales the launch; their independent closed-form checks live in
    # test_power_weight.py, not in a regenerated historical record.
    assert simulation.settings.absorber is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        dispersion = _dispersion_phi(
            simulation.modes,
            simulation.grid,
            Frame.comoving(simulation.grid),
            z_max=simulation.grid.length,
            absorber_rate=simulation.settings.absorber.spectral_rate,
        )
    original = dataclasses.replace(
        problem,
        linear=dispersion,
        initial=dataclasses.replace(problem.initial, field0=simulation._field0[None]),
    )
    bounds = {"host": HOST_BOUND, "chebyshev": CHEBYSHEV_BOUND, "phi": PHI_BOUND}
    arrays, tolerances = {}, {}
    for path, array, kind in leaves(original):
        if array.size == 0:
            # Nothing to compare but the shape.
            arrays[f"{path}.shape"] = np.asarray(array.shape)
            tolerances[f"{path}.shape"] = {"rtol": 0.0, "atol": 0.0}
            continue
        arrays[path] = np.ascontiguousarray(array)
        if kind == "exact":
            tolerances[path] = {"rtol": 0.0, "atol": 0.0}
        else:
            peak = float(np.max(np.abs(array)))
            tolerances[path] = {"rtol": 0.0, "atol": bounds[kind] * peak}
    ndarrays_regression.check(arrays, tolerances=tolerances)
