# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``numba_cpu``'s FFTW tensor Raman chain against the causal sums it transforms."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import hilbert

from backends.test_numba_cpu_kernels import run_column
from solprop.backends.numba_cpu import fft, kernels
from solprop.constants import EPS0
from vector_raman_cases import vector_problem
from vector_raman_tensor_oracle import cartesian_direct, convolution_bound


@pytest.mark.parametrize("count", [18, 30])
@pytest.mark.parametrize("material", ["H2", "SF6"])
@pytest.mark.parametrize("family", ["field_real", "field_analytic", "envelope"])
def test_fftw_tensor_chain_matches_causal_sums_at_non_power_of_two_lengths(
    count, material, family
):
    problem, _, _ = vector_problem(family, material)
    spec = kernels.column_spec(problem.responses, problem.grid_kind, 2, parallel=False)
    rng = np.random.default_rng(4048)
    values = rng.standard_normal((2, 2, count))
    if family == "envelope":
        values = values + 1j * rng.standard_normal(values.shape)
    analytic = (
        hilbert(np.real(values), axis=-1) if family == "field_analytic" else values
    )
    nodes = np.ascontiguousarray(analytic.reshape(4, count))
    plans = fft.build_plans(
        family, 4, count, count, flag="estimate", chunk_columns=2 * spec.raman_drives
    )
    window = np.linspace(0.7, 1.0, count)
    run_column(spec, problem.responses, nodes, window, drive=plans.drive)
    dt = 0.07
    delay = np.arange(count) * dt
    hv = 0.7 * np.sin(2.3 * delay) * np.exp(-0.17 * delay)
    hr = 1.1 * np.sin(0.9 * delay) if material == "H2" else np.zeros(count)
    effective = [hv + hr / 4]
    if spec.raman_drives == 3:
        effective.append(3 * hr / 4)
    spectrum = np.fft.rfft(np.array(effective), n=2 * count, axis=-1) * dt
    plans.raman_to_frequency.execute()
    kernels.scale_tensor_spectrum(plans.drive_spectrum, spectrum, 1.0 / (2 * count))
    plans.raman_to_time.execute()
    result = np.zeros((4, count), complex if family == "envelope" else float)
    kernels.make_raman_finish(spec)(
        result, nodes, plans.convolved[:, :count], window, EPS0
    )
    for node in range(2):
        expected = cartesian_direct(
            analytic[node], hv, hr, dt, envelope=family != "field_real"
        )
        if family == "field_analytic":
            # The analytic drive produces a real response matrix. Its real final
            # contraction is the required multiplication by the real carrier.
            expected = expected.real
        bound = EPS0 * convolution_bound(
            analytic[node], hv, hr, dt, envelope=family != "field_real", fft=True
        )
        assert bound > 0
        assert_allclose(
            result.reshape(2, 2, count)[node],
            EPS0 * expected * window,
            rtol=0,
            atol=bound,
        )
