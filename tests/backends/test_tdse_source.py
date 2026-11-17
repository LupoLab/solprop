# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""State the shared TDSE source's terms independently of the module (SEP 49, 7.1).

Both backends call :mod:`solprop.backends.tdse_source`, so their agreement cannot see an
error inside it. Each test here writes its expectation from the kernel contract or the
documented bound with NumPy alone: the nodal synthesis and its taper, the projection's
division by the taper, the edge array's scaling, the zero source off the active band
and the diagnostics' rounding floors. The signs of ``S = J - j_1`` and of the cubic
replacement, the density and the gate are stated in ``test_tdse_sources.py``, and the
edge term and crop in ``test_tdse_edge_free.py`` and ``test_tdse_projection.py``.
"""

import dataclasses
import math
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from edge_source_cases import EPS, EdgeCase, make_case
from solprop.backends.tdse_source import (
    FFT_ERROR_CONSTANT,
    atomic_edge_source,
    atomic_sources,
    node_fields,
    project_atomic_sources,
)
from solprop.exceptions import BackendError
from solprop.nonlinear.responses import KerrResponse
from solprop.translate.linop import power_weight_at
from solprop.translate.routes import tdse_of
from solprop.translate.tdse import TDSEResult, lattice_times, waveform_identity


def _result(case: EdgeCase, fields: np.ndarray, *, cubic: bool) -> TDSEResult:
    """Return analytic histories whose remainder and cubic shadow are both nonzero."""
    times = lattice_times(case.grid.n_to, case.grid.dt_fine)
    envelope = np.exp(-((times / 8e-15) ** 2))
    remainder = 1e-27 * envelope * np.sin(2.4e15 * times)
    first = 1e-30 * fields
    third = 1e-40 * fields**3 if cubic else None
    total = first + remainder + (0.0 if third is None else third)
    return TDSEResult(
        total,
        first,
        input_identity=waveform_identity(fields),
        configuration_identity=case.descriptor.identity,
        valid=np.ones(fields.shape, dtype=bool),
        shadow_orders=(1, 3) if cubic else (1,),
        third_order_current=third,
    )


def _fields(case: EdgeCase) -> np.ndarray:
    """Return a 2 fs-scale pulse of 1e9 V/m on every node."""
    times = lattice_times(case.grid.n_to, case.grid.dt_fine)
    pulse = 1e9 * np.exp(-((times / 6e-15) ** 2)) * np.cos(2.35e15 * times)
    rows = case.problem.transform.n_nodes
    return np.ascontiguousarray(np.broadcast_to(pulse, (rows, case.grid.n_to)))


def _evaluate(case: EdgeCase, bundle=None) -> Any:
    """Return the projected source and its diagnostics, with the inputs used."""
    bundle = case.bundle if bundle is None else bundle
    fields = _fields(case)
    result = _result(case, fields, cubic=not case.descriptor.gated)
    polarisation, current = atomic_sources(fields, result, case.problem, bundle)
    edge = atomic_edge_source(result, case.problem, bundle)
    source, diagnostics = project_atomic_sources(
        polarisation, current, result, case.problem, bundle, case.prefactor, edge=edge
    )
    return source, diagnostics, result, (polarisation, current, edge)


def _cubic_case() -> EdgeCase:
    """Return a cubic replacement whose compiled Kerr term gives a nonzero P."""
    return make_case(kerr_model="optmat_cubic", responses=(KerrResponse(chi3=1e-25),))


@pytest.fixture(scope="module", params=["tdse_native", "optmat_cubic"])
def case(request):
    return make_case() if request.param == "tdse_native" else _cubic_case()


def test_node_fields_are_the_taper_times_the_synthesis_of_each_bin():
    """One coarse bin ``a`` at fine bin ``c`` is ``(2/N) Re(a exp(2 pi i c n / N))``."""
    case = make_case()
    grids, transform = case.problem.grids, case.problem.transform
    bundle = dataclasses.replace(case.bundle, taper_scale=1.7)
    bin_index, amplitude = 17, 3.0 - 2.0j
    lab = np.zeros((1, 1, grids.n_omega), dtype=np.complex128)
    lab[0, 0, bin_index] = amplitude
    samples = grids.n_to
    phase = 2 * np.pi * grids.crop_index[bin_index] * np.arange(samples) / samples
    modal = 2 / samples * np.real(amplitude * np.exp(1j * phase))
    expected = 1.7 * transform.synthesis[:, :1] * modal
    fields = node_fields(case.problem, lab, bundle)
    assert fields.shape == (transform.n_nodes, samples)
    assert fields.flags.c_contiguous
    # Normwise inverse-FFT bound (Higham, Theorem 24.2) plus the two products and the
    # oracle's own cosine: (c_F log2 N + 4) eps of the norm.
    budget = (FFT_ERROR_CONSTANT * math.ceil(math.log2(samples)) + 4) * EPS
    error = np.linalg.norm(fields - expected)
    assert budget > 0
    assert error <= budget * np.linalg.norm(expected)


def test_node_fields_refuse_two_polarisations():
    case = make_case()
    lab = np.zeros((1, 2, case.problem.grids.n_omega), dtype=np.complex128)
    with pytest.raises(BackendError, match="one polarisation"):
        node_fields(case.problem, lab, case.bundle)


def test_the_projection_divides_by_the_taper_scale(case):
    """The projection is ``W / tau`` (kernel contract section 5).

    Halving through ``tau = 2`` is exact in binary64 -- a power-of-two scaling commutes
    with every rounding of the products, sums and transform -- so the identity holds
    bit for bit.
    """
    source, *_ = _evaluate(case)
    doubled = dataclasses.replace(case.bundle, taper_scale=2.0)
    halved, *_ = _evaluate(case, doubled)
    assert np.any(source != 0)
    np.testing.assert_array_equal(2 * halved, source)


def test_the_edge_array_is_density_scale_and_gate_derivative_of_the_remainder():
    """``E = rho s g'(t) (J - j_1)``, the density and scale entering once."""
    case = make_case()
    bundle = dataclasses.replace(case.bundle, density=3.0)
    fields = _fields(case)
    result = _result(case, fields, cubic=False)
    edge = atomic_edge_source(result, case.problem, bundle)
    assert edge is not None
    descriptor = case.descriptor
    remainder = result.total_current - result.first_order_current
    expected = (
        descriptor.source_gate_derivative()
        * remainder
        * (descriptor.remainder_scale * 3.0)
    )
    # Three products, associated differently in the oracle: gamma_3 of each value.
    np.testing.assert_allclose(edge[:, 0, :], expected, rtol=4 * EPS, atol=0)
    assert np.any(expected != 0)


def test_the_source_is_exactly_zero_off_the_active_band(case):
    """Both spectral factors vanish off the active band by construction, so must F."""
    source, *_ = _evaluate(case)
    active = case.problem.grids.active_mask
    assert np.all(np.isfinite(source))
    assert np.all(source[..., ~active] == 0)
    assert np.any(source[..., active] != 0)


def _absolute_projection(values: np.ndarray, projection: np.ndarray, taper: float):
    """Return ``sum_n |W_mn| values_n / |tau|`` per mode and sample."""
    return np.einsum("mn,nt->mt", np.abs(projection), values[:, 0, :]) / abs(taper)


def _floor_terms(case: EdgeCase, magnitudes: np.ndarray, factor, projection):
    """Return the documented operand bound ``|factor w| sum_t |W| x / |tau|``."""
    grids = case.problem.grids
    spectral = _absolute_projection(magnitudes, projection, case.bundle.taper_scale)
    return np.abs(factor * grids.omega_window) * spectral.sum(axis=-1)[:, None, None]


def _accumulation(case: EdgeCase) -> float:
    """``(8 + 2 n_nodes + 5 ceil(log2 N) [+1 for a gate]) eps`` of the floor."""
    operations = 8 + 2 * case.problem.transform.n_nodes
    operations += 5 * math.ceil(math.log2(case.problem.grids.n_to))
    return (operations + (1 if case.descriptor.gated else 0)) * EPS


def test_the_complete_floor_is_the_documented_operand_bound():
    """The ungated (cubic) source's floors, from the absolute raw operands.

    Each floor is ``|| |c w| sum_t |W| |rho s| (|J| + |j_1| + |j_3|) / |tau| ||``
    times ``a / (1 - a)``, for the current and likewise ``|p w|`` and ``|P|`` for the
    polarisation, over the fine and the embedded maps.
    """
    case = _cubic_case()
    _, diagnostics, result, (polarisation, _, _) = _evaluate(case)
    transform, bundle = case.problem.transform, case.bundle
    source = tdse_of(case.problem.route)
    assert source is not None
    factor = source.current_numerator / power_weight_at(case.problem, bundle.z1)
    operands = (
        np.abs(result.total_current)
        + np.abs(result.first_order_current)
        + np.abs(result.third_order_current)
    )
    scale = abs(bundle.density * case.descriptor.remainder_scale)
    current = scale * operands[:, None, :]
    accumulation = _accumulation(case)
    ratio = accumulation / (1 - accumulation)
    floors = {"current": 0.0, "polarisation": 0.0}
    maps = [transform.projection]
    if not np.array_equal(transform.projection, transform.projection_coarse):
        # The embedded map adds its own floors only when it is a distinct rule.
        maps.append(transform.projection_coarse)
    for projection in maps:
        floors["current"] += ratio * float(
            np.linalg.norm(_floor_terms(case, current, factor, projection))
        )
        floors["polarisation"] += ratio * float(
            np.linalg.norm(
                _floor_terms(case, np.abs(polarisation), case.prefactor, projection)
            )
        )
    # The bound is itself evaluated in floating point, with sums of a few thousand
    # terms reassociated here: 1e-12 relative covers that and nothing else.
    assert diagnostics["remainder_source"]["roundoff_floor"] == pytest.approx(
        floors["current"], rel=1e-12, abs=0
    )
    assert diagnostics["complete_source"]["roundoff_floor"] == pytest.approx(
        floors["current"] + floors["polarisation"], rel=1e-12, abs=0
    )
    assert floors["polarisation"] > 0


def test_the_edge_floors_are_the_documented_section_4_5_bounds():
    """A gated source's band floors (SEP 26 section 4.5), written from the docstring.

    Per bin, with ``c_k = |factor_k w_k|``: the truncating part ``c_k sum |W| |rho s|
    |g| (|J| + |j_1|) a / (1 - a)``; the edge transform ``c_k sqrt(N) (c_F eps log2 N
    + (n_nodes + 6) u) ||y||_2 / omega_lo`` with ``y`` the absolute projection of
    ``|rho s| |g'| (|J| + |j_1|)``, plus ``c_k sqrt(N) ||y_d||_2 / omega_lo`` for the
    derivative's own error; and the division ``2 eps c_k (|X_k| + |E_k| / omega_k)``.
    """
    case = make_case()
    _, diagnostics, result, (_, current, edge) = _evaluate(case)
    assert edge is not None
    problem, bundle, descriptor = case.problem, case.bundle, case.descriptor
    grids, projection = problem.grids, problem.transform.projection
    source = tdse_of(problem.route)
    assert source is not None
    factor = source.current_numerator / power_weight_at(problem, bundle.z1)
    collar = np.abs(factor * grids.omega_window)
    raw = np.abs(result.total_current) + np.abs(result.first_order_current)
    scale = abs(bundle.density * descriptor.remainder_scale)
    gate = grids.t_window_fine if descriptor.source_gate() is None else None
    assert gate is not None
    accumulation = _accumulation(case)
    truncating = _floor_terms(
        case, scale * (raw * gate)[:, None, :], factor, projection
    )
    truncating = truncating * accumulation / (1 - accumulation)
    samples, nodes = grids.n_to, problem.transform.n_nodes
    lowest = float(np.min(grids.omega[grids.active_mask]))
    transform = FFT_ERROR_CONSTANT * EPS * math.ceil(math.log2(samples)) + (
        nodes + 6
    ) * (EPS / 2)
    y = _absolute_projection(
        scale * (raw * np.abs(descriptor.source_gate_derivative()))[:, None, :],
        projection,
        bundle.taper_scale,
    )
    y_d = _absolute_projection(
        scale * (raw * descriptor.source_gate_derivative_error())[:, None, :],
        projection,
        bundle.taper_scale,
    )
    norms = transform * np.linalg.norm(y, axis=-1) + np.linalg.norm(y_d, axis=-1)
    edge_bins = collar * math.sqrt(samples) * norms[:, None, None] / lowest

    def spectrum(values):
        projected = np.einsum("mn,nt->mt", projection, values[:, 0, :])
        return np.fft.rfft(projected / bundle.taper_scale)[:, None, grids.crop_index]

    inverse = np.zeros(grids.n_omega)
    inverse[grids.active_mask] = 1 / grids.omega[grids.active_mask]
    division = (
        2
        * EPS
        * collar
        * (np.abs(spectrum(current)) + np.abs(spectrum(edge)) * inverse)
    )
    band = diagnostics["edge"]["bands"]["active"]
    active = grids.active_mask
    # As above: the bounds are evaluated in floating point and reassociated here.
    assert band["truncating_floor"] == pytest.approx(
        np.linalg.norm(truncating[..., active]), rel=1e-12, abs=0
    )
    assert band["edge_floor"] == pytest.approx(
        np.linalg.norm(edge_bins[..., active]) + np.linalg.norm(division[..., active]),
        rel=1e-12,
        abs=0,
    )
    assert np.linalg.norm(division[..., active]) > 0


def test_the_shared_source_imports_neither_backend():
    """Native calls this module and never imports the reference (SEP 49, 6.1)."""
    probe = (
        "import sys, solprop.backends.tdse_source; "
        "loaded = [name for name in sys.modules if 'numpy_ref' in name "
        "or name.startswith('solprop_native')]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", probe], check=True, timeout=120)
