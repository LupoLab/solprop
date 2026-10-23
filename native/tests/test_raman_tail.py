# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify actual-width Raman transforms on real, analytic and envelope grids."""

from __future__ import annotations

import dataclasses
import math
from typing import cast

import numpy as np
import pytest
from conftest import ComplexArray
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.raman import response_for
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import Gas
from solprop.physdata.selection import MaterialSelection
from solprop.pulses import GaussianPulse
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble

from solprop_native import NativeBackend, build_info


def _bound(samples: int) -> float:
    """Bound eight fixed steps against another run of them, relative to the peak.

    Eight steps, seven stage transforms and eight doubled-grid Raman pairs per step;
    the existing ``512 eps ceil(log2 N)`` FFT budget on both devices.
    """
    eps = np.finfo(np.float64).eps
    return (
        8
        * 512
        * eps
        * (7 * math.ceil(math.log2(samples)) + 8 * math.ceil(math.log2(2 * samples)))
    )


@pytest.mark.gpu
@pytest.mark.parametrize("family", ["real", "analytic", "envelope"])
@pytest.mark.parametrize("chunk", [1, 2, 4])
def test_raman_partial_chunk_matches_reference_and_repeats(
    family: str, chunk: int
) -> None:
    grid_class = EnvelopeGrid if family == "envelope" else FieldGrid
    grid = grid_class.from_limits(1e-6, 800e-9, (200e-9, 4e-6), 80e-15)
    thg = family == "real"
    mode = MarcatiliMode(125e-6, "H2", 2.0, cladding=1.45, loss=False)
    quadrature = transverse_quadrature([mode], n_r=7, radial_rule="gauss_kronrod")
    # optmat disperses H2's vibrational strengths to the carrier and builds its
    # density-shifted lines at the fill's constant density.
    raman = response_for(
        Gas.H2,
        thg=thg,
        selection=MaterialSelection(carrier_wavelength=800e-9),
        density=float(mode.density(np.zeros(1))[0]),
    )
    # optmat's H2 index holds from 112 nm to the static limit, past this grid's
    # collar, and 2 bar is above H2's Dicke cutoff: SEP 3 qualifies no warning here.
    problem = assemble(
        grid,
        [mode],
        [KerrResponse(chi3=1e-51, thg=thg), raman],
        GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=20e-6).build(grid),
        quadrature=quadrature,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_init=grid.length / 8,
        dz_max=grid.length / 8,
    )
    problem = dataclasses.replace(
        problem,
        transform=dataclasses.replace(problem.transform, node_chunk=chunk),
        stepper=dataclasses.replace(
            problem.stepper,
            schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / 8),
        ),
    )
    if chunk > 1:
        assert problem.transform.n_nodes % chunk != 0
    fields: list[ComplexArray] = []
    for backend_class in (NumpyRefBackend, NativeBackend):
        backend = backend_class(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(grid.length)
            field = backend.observe().require_field().copy()
            assert np.isfinite(field).all()
            fields.append(field)
            if isinstance(backend, NativeBackend):
                info = build_info()
                metadata = backend.metadata()
                sizing = cast(dict[str, dict[str, int]], metadata["execution_sizing"])
                full = info.raman_fft_threads
                tail = info.raman_tail_fft_threads
                kernel = full
                if info.cpu_size_policy_enabled:
                    full = sizing[f"raman_fft.raman.{chunk * 2 * grid.n_to}"]["threads"]
                    kernel = sizing[f"raman_fft.raman.{2 * grid.n_to}"]["threads"]
                    if 7 % chunk:
                        tail = sizing[
                            f"raman_tail_fft.raman.{(7 % chunk) * 2 * grid.n_to}"
                        ]["threads"]
                accelerated = info.raman_fft_option == "ACCELERATE" and (
                    full > 1 or 2 * grid.n_to > 16384
                )
                tail_accelerated = info.raman_fft_option == "ACCELERATE" and (
                    tail > 1 or 2 * grid.n_to > 16384
                )
                kernel_accelerated = info.raman_fft_option == "ACCELERATE" and (
                    kernel > 1 or 2 * grid.n_to > 16384
                )
                assert metadata["raman_fft_provider"] == (
                    "Accelerate_vDSP" if accelerated else "KokkosFFT"
                )
                assert metadata["raman_fft_tasks"] == {
                    "full": min(chunk, full) if accelerated else full,
                    "tail": (min(7 % chunk, tail) if tail_accelerated else tail)
                    if 7 % chunk
                    else 0,
                    "kernel": min(1, kernel) if kernel_accelerated else kernel,
                }
                ledger = backend.memory_ledger()
                assert ledger["node_raman_drive"] == chunk * 2 * grid.n_to * 8
                # A repeat of the same prescribed march: rounding only, within the
                # equivalence bound below (solprop SEP 25).
                backend.set_initial(problem.initial.field0, None)
                backend.propagate_to(grid.length)
                np.testing.assert_allclose(
                    backend.observe().require_field(),
                    field,
                    rtol=0.0,
                    atol=_bound(grid.n_to) * np.max(np.abs(field)),
                )
        finally:
            backend.close()
        if isinstance(backend, NativeBackend):
            assert backend._native_state.closed  # pyright: ignore[reportPrivateUsage] - native lifetime
    scale = np.max(np.abs(fields[0]))
    np.testing.assert_allclose(
        fields[1], fields[0], rtol=0.0, atol=_bound(grid.n_to) * scale
    )
