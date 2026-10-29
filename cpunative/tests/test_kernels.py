# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Check transforms and response streams against independent array calculations."""

import numpy as np
import pytest
from scipy.signal import fftconvolve
from solprop.translate.stage import terms
from test_parity import problem as problem
from test_workspace import small as small

from solprop_cpunative import _core
from solprop_cpunative.backend import CPUNativeBackend, _payload, _stage


@pytest.fixture(autouse=True)
def serial(monkeypatch):
    monkeypatch.setenv("SOLPROP_THREADS", "1")
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "estimate")


@pytest.mark.parametrize(("nt", "nf"), [(12, 24), (14, 30), (32, 64)])
@pytest.mark.parametrize("threads", [1, 4])
def test_fft_scaling_padding_and_endpoint_bins(small, nt, nf, threads):
    _, p = small
    data = _payload(p, terms(p))
    modes, nw = p.shape[1], nt // 2 + 1
    data.update(
        nt=nt,
        nf=nf,
        nw=nw,
        threads=threads,
        split=False,
        phi=np.zeros((1, modes, 1, nw), complex),
        prefactor=np.ones((1, modes, 1, nw), complex),
        crop=np.arange(nw, dtype=np.uintp),
        active=np.ones(nw, dtype=np.uint8),
        norm=np.ones(nw, dtype=np.uint8),
        window=np.ones(nf),
        window_derivative=np.zeros(nf),
        omega=np.arange(nw, dtype=float),
        omega_window=np.ones(nw),
        absorber=np.zeros(nf),
        absorber_coarse=np.zeros(nt),
    )
    state = _core.State(data)
    rng = np.random.default_rng(935)
    field = rng.normal(size=(modes, nw)) + 1j * rng.normal(size=(modes, nw))
    field[:, 0] = field[:, 0].real
    state.load(field)
    padded = np.zeros((modes, nf // 2 + 1), complex)
    padded[:, :nw] = field
    expected = np.fft.irfft(padded, n=nf)
    actual = np.asarray(state._temporal()).reshape(modes, nf)
    # FFT forward-error budget based on depth, independent of observed differences.
    bound = 64 * np.finfo(float).eps * np.log2(nf)
    np.testing.assert_allclose(
        actual, expected, rtol=bound, atol=bound * np.max(np.abs(expected))
    )
    np.testing.assert_allclose(
        np.asarray(state._roundtrip()).reshape(modes, nw),
        field,
        rtol=bound,
        atol=bound * np.max(np.abs(field)),
    )


def reflected_integral(values, taps, dt):
    reach = len(taps) - 1
    padded = np.pad(values, reach, mode="reflect", reflect_type="odd")
    filtered = taps[0] * values
    for k in range(1, reach + 1):
        filtered += taps[k] * (
            padded[reach - k : reach - k + len(values)]
            + padded[reach + k : reach + k + len(values)]
        )
    return np.concatenate(([0.0], np.cumsum(dt * (filtered[1:] + filtered[:-1]) / 2)))


@pytest.mark.parametrize("boundary", [False, True])
def test_kerr_raman_rate_and_held_current_streams(problem, boundary):
    sim, p = problem
    b = CPUNativeBackend(p, grid=sim.grid, modes=sim.modes)
    b.set_initial(p.initial.field0, None)
    st = _stage(b._stage_scalars(0.0, 0.0))
    data = _payload(p, terms(p))
    n = p.grids.n_to
    # One physical node, with an edge spike to exercise point reflection and
    # negative filtered exposure; the entire causal Raman support is retained.
    t = np.arange(n)
    e = 8e10 * np.exp(-(((t - 0.45 * n) / (0.09 * n)) ** 2)) * np.cos(0.71 * t)
    if boundary:
        e[:3] = [8e10, 0.0, -4e10]
    got = np.asarray(b._cpu._response(e, st)).reshape(5, n)
    rate_table = p.responses.plasma_channels[0].rate
    magnitude = np.abs(e)
    rate = np.zeros(n)
    inside = (magnitude >= rate_table.field_min) & (magnitude < rate_table.field_max)
    logarithm = np.log(magnitude[inside])
    coeff = rate_table.coefficients
    index = np.clip(
        np.floor(
            (logarithm - rate_table.log_field_min) / rate_table.d_log_field
        ).astype(int),
        0,
        coeff.shape[1] - 1,
    )
    x = logarithm - (rate_table.log_field_min + index * rate_table.d_log_field)
    rate[inside] = np.exp(
        ((coeff[0, index] * x + coeff[1, index]) * x + coeff[2, index]) * x
        + coeff[3, index]
    )
    rate[magnitude >= rate_table.field_max] = rate_table.rate_max
    exposure = reflected_integral(rate, data["taps"], data["dt"])
    fraction = 1 - (1 - st["preionisation"]) * np.exp(-np.maximum(exposure, 0))
    free = reflected_integral(fraction * data["charge"] * e, data["taps"], data["dt"])
    loss = np.zeros(n)
    live = (rate > 0) & (e != 0)
    loss[live] = data["potential"] * rate[live] * (1 - fraction[live]) / e[live]
    current = st["plasma_density"] * (free + loss)
    polarisation = data["kerr"] * st["kerr_density"] * e**3
    if p.responses.raman_terms:
        undamped = p.responses.raman_terms[0].undamped
        kernel = data["raman_fraction"] * np.sum(
            undamped
            * np.exp(
                -np.asarray(st["raman_rates"])[:, None]
                * np.arange(2 * n)[None, :]
                * data["dt"]
            ),
            axis=0,
        )
        polarisation += (
            data["epsilon0"]
            * st["raman_density"]
            * e
            * fftconvolve(e**2, kernel)[:n]
            * data["dt"]
        )
    expected = [
        polarisation * data["window"],
        current * data["window"],
        current * data["window_derivative"],
        rate,
        fraction,
    ]
    # Sequential running integrals have O(n*eps) accumulated rounding. Fractions
    # subtract from one, so their absolute scale is one even below ionisation.
    bound = 64 * np.finfo(float).eps * n
    for index, (actual, reference) in enumerate(zip(got, expected, strict=True)):
        scale = max(
            np.max(np.abs(reference)), 1.0 if index == 4 else np.finfo(float).tiny
        )
        if index in (1, 2):
            # An oscillatory current can cancel before it reaches the taper.
            # Bound the integral by its absolute work, including its FIR gain,
            # rather than by the tiny surviving tail multiplied by g or g'.
            filter_norm = abs(data["taps"][0]) + 2 * np.sum(np.abs(data["taps"][1:]))
            absolute_work = data["dt"] * filter_norm * np.sum(
                np.abs(fraction * data["charge"] * e)
            ) + np.max(np.abs(loss))
            window = data["window"] if index == 1 else data["window_derivative"]
            scale = max(
                scale, st["plasma_density"] * absolute_work * np.max(np.abs(window))
            )
        np.testing.assert_allclose(
            actual, reference, rtol=bound, atol=bound * scale, err_msg=f"stream {index}"
        )
    b.close()
