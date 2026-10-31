# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Schema-six channel payloads, validation, populations and Raman cache ownership."""

from __future__ import annotations

import numpy as np
import pytest
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear import responses_for
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.physdata import Mixture
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.translate.problem import KernelProblem, assemble

from solprop_native import NativeBackend, _core
from solprop_native.backend import (
    _native_problem,  # pyright: ignore[reportPrivateUsage] - corrupt native payloads
)


@pytest.fixture
def mixture_case() -> tuple[KernelProblem, FieldGrid]:
    """Declare nitrogen and oxygen channels with optmat's molecular rates."""
    mixture = Mixture.air(1.0)
    grid = FieldGrid.from_limits(1e-4, 800e-9, (400e-9, 2e-6), 400e-15, ffac=4.0)
    mode = MarcatiliMode(30e-6, mixture=mixture, cladding=1.45, loss=False)
    responses = responses_for(
        mixture,
        thg=False,
        plasma={"N2": OptmatRate("N2", 800e-9), "O2": OptmatRate("O2", 800e-9)},
        wavelength=800e-9,
    )
    # optmat's N2, O2 and Ar indices hold from below 200 nm to the static limit,
    # past this grid's collar: SEP 3 qualifies no range warning here.
    problem = assemble(
        grid,
        mode,
        responses,
        GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-8).build(grid),
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
        ionisation_field_range=(1e8, 1e11),
    )
    return problem, grid


def test_species_statistics_preserve_an_inert_component_and_close(
    mixture_case: tuple[KernelProblem, FieldGrid],
) -> None:
    problem, grid = mixture_case
    backend = NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        values = backend.stats(
            StatsSpec(names=("species_ionisation", "species_density"))
        )
        assert values["species_ionisation"].shape == (1, 3, 1)
        # Exact: the inert component has no rate, so nothing ever ionises it.
        np.testing.assert_array_equal(values["species_ionisation"][:, 2], 0.0)
        assert np.all(values["species_density"] > 0.0)
        ledger = backend.memory_ledger()
        assert ledger["channel_scratch"] == 2 * problem.transform.node_chunk * 4 * 8
        assert ledger["plasma_statistics_scratch"] == problem.transform.n_nodes * 5 * 8
        assert backend.metadata()["plasma_channels"] == 2
    finally:
        backend.close()
    assert backend._native_state.closed  # pyright: ignore[reportPrivateUsage] - lifecycle
    backend.close()


@pytest.mark.parametrize(
    "damage",
    [
        "zero_channels",
        "parameter_extent",
        "nonfinite",
        "offset",
        "interval",
        "coefficients",
    ],
)
def test_native_channel_payload_refusals(
    mixture_case: tuple[KernelProblem, FieldGrid], damage: str
) -> None:
    problem, grid = mixture_case
    backend = NativeBackend(problem, grid=grid)
    try:
        payload = _native_problem(
            problem,
            backend._terms,  # pyright: ignore[reportPrivateUsage] - fixed payload
        )
        parameters = np.array(payload["channel_parameters"], dtype=float, copy=True)
        if damage == "zero_channels":
            payload["n_channels"] = 0
        elif damage == "parameter_extent":
            parameters = parameters[:, :-1].copy()
        elif damage == "nonfinite":
            parameters[1, 7] = np.nan
        elif damage == "offset":
            parameters[1, 6] += 1
        elif damage == "interval":
            parameters[1, 5] = -1
        else:
            payload["channel_coefficients"] = np.empty(0)
        payload["channel_parameters"] = parameters
        with pytest.raises(ValueError, match=r"channel|Channel|plasma"):
            _core.State(payload)
    finally:
        backend.close()


def test_native_statistics_refuse_channel_preionisation_shape(
    mixture_case: tuple[KernelProblem, FieldGrid],
) -> None:
    problem, grid = mixture_case
    backend = NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        with pytest.raises(ValueError, match="channel preionisation"):
            backend._native_state.statistics(  # pyright: ignore[reportPrivateUsage] - native refusal
                1 << 11, np.empty(0), np.empty(0), 0.0, 1.0, 0.0, 1.0, np.zeros(1)
            )
        with pytest.raises(ValueError, match="channel preionisation"):
            backend._native_state.statistics(  # pyright: ignore[reportPrivateUsage] - native refusal
                1 << 11,
                np.empty(0),
                np.empty(0),
                0.0,
                1.0,
                0.0,
                1.0,
                np.array([0.0, 1.01]),
            )
    finally:
        backend.close()
