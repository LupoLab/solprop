# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify schema-nine separate envelope currents and population ownership."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from legacy_geometry import LEGACY_GRID
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import EnvelopeGrid, FreeGrid, RadialGrid
from solprop.modes import MarcatiliMode, ModeCollection, transverse_quadrature
from solprop.nonlinear import EnvelopePlasmaResponse, responses_for
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.physdata import Mixture
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import KernelProblem, assemble
from solprop.translate.stage import terms

from solprop_native import NativeBackend, _core, build_info
from solprop_native.backend import (
    _native_problem,  # pyright: ignore[reportPrivateUsage] - validate the native boundary
)

# The FFT collar extends beyond the neutral-gas index fit. These are current
# assembly witnesses; the active band and extrapolation policy remain unchanged.
pytestmark = pytest.mark.filterwarnings(
    f"ignore::{MaterialRangeWarning.__module__}.MaterialRangeWarning"
)
ROUTES = ("modal", "rows", "radial", "free")
"""The routes the thread-count probe marches, two components each."""
NAMES = (
    "electron_density",
    "peak_ionisation_rate",
    "species_ionisation",
    "species_electron_density",
    "species_peak_ionisation_rate",
    "species_density",
    "species_ionisation_energy",
)


def native_bound(problem: KernelProblem, grid: EnvelopeGrid) -> float:
    """Bound two Kokkos runs of one case's eight prescribed steps against each other.

    The reference comparison's rounding term without its 2e-10 for the reference's
    independent angular rule: the same eight seven-stage steps of node projections
    and temporal and spatial transforms, at the same operation count, on both sides.
    A population or rate statistic is held to the same relative bound, as the
    reference comparison holds it.
    """
    return 128 * grid.n_to * problem.transform.n_nodes * 56 * float(np.finfo(float).eps)


def test_envelope_fields_and_all_populations_agree_across_cpu_threads(
    tmp_path: Path,
) -> None:
    """One and four threads, fresh processes, to the matched-run bound.

    Every route replays its eight prescribed steps on both sides, so the thread count
    moves results by rounding only (solprop SEP 25): :func:`native_bound` for the
    fields and statistics, and the schedule's eight steps exactly.
    """
    if build_info().device_backend != "none":
        pytest.skip("CPU thread-count invariance requires a host execution space")
    records: list[dict[str, object]] = []
    runs: list[dict[str, np.ndarray]] = []
    for threads in (1, 4):
        out = tmp_path / f"envelope-{threads}.npz"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("envelope_thread_probe.py")),
                str(out),
            ],
            text=True,
            capture_output=True,
            check=False,
            env={
                **os.environ,
                "SOLPROP_NATIVE_THREADS": str(threads),
                "SOLPROP_OVERSUBSCRIBE": "1",
                "SOLPROP_THREADS": "",
                "SOLPROP_NATIVE_KERNEL_THREADS": str(threads),
                "SOLPROP_NATIVE_FFT_THREADS": str(threads),
                "OMP_DYNAMIC": "FALSE",
                "OMP_PROC_BIND": "false",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            },
        )
        assert result.returncode == 0, result.stderr
        records.append(
            cast(dict[str, object], json.loads(result.stdout.splitlines()[-1]))
        )
        with np.load(out) as stored:
            runs.append({name: stored[name] for name in stored.files})
    assert [record["threads"] for record in records] == [1, 4]
    assert records[0]["steps"] == records[1]["steps"] == [8] * 4
    one, four = runs
    assert one.keys() == four.keys()
    for route in ROUTES:
        problem, grid = envelope_case(route, 2)
        bound = native_bound(problem, grid)
        expected = one[f"{route}_field"]
        np.testing.assert_allclose(
            four[f"{route}_field"],
            expected,
            rtol=bound,
            atol=bound * np.max(np.abs(expected)),
            err_msg=route,
        )
        for name in one:
            if name.startswith(f"{route}_") and name != f"{route}_field":
                np.testing.assert_allclose(
                    four[name], one[name], rtol=bound, atol=1e-30, err_msg=name
                )


def test_envelope_current_streams_use_large_radial_projection() -> None:
    grid = EnvelopeGrid.from_limits(
        8e-6, 800e-9, (600e-9, 1.2e-6), 300e-15, **LEGACY_GRID
    )
    space = RadialGrid(1e-3, 256)
    initial = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-3), 1e-4
    ).build(grid, space, medium=SpatialMedium())
    problem = assemble(
        grid,
        None,
        [
            EnvelopePlasmaResponse(
                "Ar", OptmatRate("Ar", 800e-9, model="ppt_ilkov"), preionisation=0.1
            )
        ],
        initial,
        spatial_grid=space,
        medium=SpatialMedium(),
        density=1e24,
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=[grid.length],
        dz_init=1e-6,
        dz_max=1e-6,
        ionisation_field_range=(1e7, 1e11),
    )
    assert problem.transform.node_chunk == space.n_r
    reference, actual = (
        NumpyRefBackend(problem, grid=grid),
        NativeBackend(problem, grid=grid),
    )
    try:
        for owner in (reference, actual):
            owner.set_initial(problem.initial.field0, None)
        expected = reference._nonlinear(  # pyright: ignore[reportPrivateUsage]
            problem.initial.field0,
            reference._stage_scalars(0.0, 0.0),  # pyright: ignore[reportPrivateUsage]
            np.ones(problem.shape, complex),
        )[0]
        actual._trial_step(0.0, 0.0)  # pyright: ignore[reportPrivateUsage]
        observed = np.asarray(
            actual._native_state.stage_to_host(0)  # pyright: ignore[reportPrivateUsage]
        ).reshape(problem.shape[1:])
        info = build_info()
        if info.device_backend == "none" and info.gemm_provider in {
            "HYBRID",
            "KOKKOSBLAS_TPL",
        }:
            assert str(actual.metadata()["projection_provider"]).endswith(
                "-packed-real"
            )
        # Independent angular arithmetic contributes 2e-10; projection contributes
        # two length-n_r reductions and temporal FFTs to each current stream.
        bound = 2e-10 + 128 * space.n_r * grid.n_to * np.finfo(float).eps
        scale = np.max(np.abs(expected))
        assert scale > 0
        np.testing.assert_allclose(observed, expected, rtol=bound, atol=bound * scale)
    finally:
        actual.close()
        reference.close()


def envelope_case(
    route: str, n_pol: int, *, mixture: bool = True
) -> tuple[KernelProblem, EnvelopeGrid]:
    fill = Mixture.fill({"Ar": 0.5, "He": 0.5} if mixture else {"Ar": 1.0}, 0.5)
    fill = replace(
        fill,
        components=tuple(
            replace(one, preionisation=0.125 * index)
            for index, one in enumerate(fill.components)
        ),
    )
    grid = EnvelopeGrid.from_limits(
        8e-6, 800e-9, (500e-9, 1.2e-6), 48e-15, **LEGACY_GRID
    )
    pulse = GaussianPulse(wavelength=800e-9, fwhm=12e-15, energy=1e-6)
    responses = responses_for(
        fill,
        envelope=True,
        kerr=False,
        raman=False,
        plasma={
            one.label: OptmatRate(one.species.name, 800e-9, model="ppt_ilkov")
            for one in fill.components
        },
    )
    spec = StatsSpec(names=NAMES, periodic=NAMES[:4], period=1)
    schedule = FixedSchedule.uniform(0, grid.length, grid.length / 8)
    controller = PIController(rtol=1e-6, atol=1e-12)
    jones = (1,) if n_pol == 1 else (1, 0.35j)
    if route in ("free", "radial"):
        space = (
            FreeGrid(3e-4, 4, 4e-4, 4, collar_fraction=0)
            if route == "free"
            else RadialGrid(6e-4, 8, collar_fraction=0)
        )
        initial = SpatialPulse(pulse, 80e-6, jones=jones).build(
            grid, space, medium=SpatialMedium(fill)
        )
        problem = assemble(
            grid,
            None,
            responses,
            initial,
            spatial_grid=space,
            medium=SpatialMedium(fill),
            controller=controller,
            save_planes=[grid.length],
            dz_init=grid.length / 8,
            dz_max=grid.length / 8,
            schedule=schedule,
            stats_spec=spec,
            ionisation_field_range=(1e7, 1e11),
        )
    else:
        mode = MarcatiliMode(30e-6, mixture=fill, cladding=1.45, loss=False)
        initial = pulse.build(grid) * np.asarray(jones)[None, :, None]
        modes = (
            ModeCollection.with_polarisations([mode])
            if route == "rows"
            else ModeCollection([mode])
        )
        if route == "rows":
            initial = initial.transpose(1, 0, 2).copy()
        problem = assemble(
            grid,
            modes,
            responses,
            initial,
            quadrature=transverse_quadrature(modes, n_r=2, radial_rule="gauss_kronrod"),
            controller=controller,
            save_planes=[grid.length],
            dz_init=grid.length / 8,
            dz_max=grid.length / 8,
            schedule=schedule,
            stats_spec=spec,
            ionisation_field_range=(1e7, 1e11),
        )
    return replace(
        problem,
        transform=replace(problem.transform, node_chunk=2 if route == "modal" else 3),
    ), grid


@pytest.mark.parametrize(
    ("route", "n_pol"),
    [
        ("modal", 1),
        ("modal", 2),
        ("rows", 2),
        ("radial", 1),
        ("radial", 2),
        ("free", 1),
        ("free", 2),
    ],
)
def test_envelope_currents_steps_statistics_and_repeatability(
    route: str, n_pol: int
) -> None:
    problem, grid = envelope_case(route, n_pol)
    reference, actual = (
        NumpyRefBackend(problem, grid=grid),
        NativeBackend(problem, grid=grid),
    )
    repeat = NativeBackend(problem, grid=grid)
    try:
        for owner in (reference, actual, repeat):
            owner.set_initial(problem.initial.field0, None)
        for target in (0.0, grid.length / 8, grid.length):
            if target:
                for owner in (reference, actual, repeat):
                    owner.propagate_to(target)
            expected, got = (
                reference.observe().require_field(),
                actual.observe().require_field(),
            )
            # The shared-vs-SVD cycle oracle allows 2e-10; eight seven-stage
            # steps also sum node projections and temporal/spatial transforms.
            bound = (
                2e-10
                + 128 * grid.n_to * problem.transform.n_nodes * 56 * np.finfo(float).eps
            )
            np.testing.assert_allclose(
                got, expected, rtol=bound, atol=bound * np.max(np.abs(expected))
            )
            # A repeat is the same discrete computation: rounding only.
            repeat_bound = native_bound(problem, grid)
            np.testing.assert_allclose(
                repeat.observe().require_field(),
                got,
                rtol=repeat_bound,
                atol=repeat_bound * np.max(np.abs(got)),
            )
            wanted, observed = (
                reference.stats(problem.stats_spec),
                actual.stats(problem.stats_spec),
            )
            for name in NAMES:
                np.testing.assert_allclose(
                    observed[name], wanted[name], rtol=bound, atol=1e-30
                )
        ledger = actual.memory_ledger()
        assert ledger["envelope_drude_stream"] > 0
        assert ledger["envelope_loss_stream"] == ledger["envelope_drude_stream"]
        assert ledger["envelope_status"] > 0
    finally:
        actual.close()
        repeat.close()
        reference.close()


@pytest.mark.parametrize(
    "corruption",
    ["field", "shape", "nan", "weight", "inactive", "sign", "thg", "stash"],
)
def test_envelope_native_payload_rejects_inconsistent_inputs(corruption: str) -> None:
    problem, _ = envelope_case("modal", 1, mixture=False)
    payload = _native_problem(problem, terms(problem))
    if corruption in ("field", "thg", "stash"):
        key, value = {
            "field": ("family", "field_real"),
            "thg": ("harmonic_factor", 1.0),
            "stash": ("stash_plasma", True),
        }[corruption]
        payload[key] = value
    elif corruption in ("shape", "nan", "weight"):
        key = (
            "envelope_rule_nodes" if corruption != "weight" else "envelope_rule_weights"
        )
        array = cast(npt.NDArray[np.float64], payload[key]).copy()
        if corruption == "shape":
            payload[key] = array.ravel()
        else:
            array.flat[0] = np.nan if corruption == "nan" else 0.0
            payload[key] = array
    else:
        array = cast(npt.NDArray[np.float64], payload["envelope_drude_factor"]).copy()
        mask = problem.grids.active_mask & (problem.grids.omega > 0)
        index = int(np.flatnonzero(mask if corruption == "sign" else ~mask)[0])
        array[index] = 1.0
        payload["envelope_drude_factor"] = array
    with pytest.raises(ValueError, match="envelope"):
        _core.State(payload)


@pytest.mark.parametrize(
    ("geometry", "n_pol"), [("modal", 1), ("radial", 2), ("free", 2)]
)
def test_envelope_angular_failure_is_a_solprop_convergence_error(
    monkeypatch: pytest.MonkeyPatch,
    geometry: str,
    n_pol: int,
) -> None:
    from solprop.exceptions import ConvergenceError

    import solprop_native.backend as adapter

    problem, grid = envelope_case(geometry, n_pol, mixture=False)
    payload = _native_problem(problem, terms(problem))
    nodes = cast(npt.NDArray[np.float64], payload["envelope_rule_nodes"]).copy()
    # Sorted interior nodes and normalized positive weights still satisfy native
    # shape/domain checks. Alternating incompatible rules must fail refinement.
    for level in (1, 3):
        order = 64 << level
        nodes[:, level, :order] = 0.5 * (nodes[:, level, :order] + 1)
    payload["envelope_rule_nodes"] = nodes

    def corrupt_payload(*args: object) -> dict[str, object]:
        return payload

    monkeypatch.setattr(adapter, "_native_problem", corrupt_payload)
    owner = NativeBackend(problem, grid=grid)
    try:
        # Spatial pulses need enough local field to enter the angular rule; a
        # below-threshold sample correctly bypasses refinement even with bad nodes.
        owner.set_initial(
            np.asarray(10.0 * problem.initial.field0, dtype=np.complex128), None
        )
        with pytest.raises(ConvergenceError, match="angular refinement"):
            owner.propagate_to(grid.length / 8)
        assert owner.observe().z == 0.0
        # Reusing the same allocation must reset a failed device reduction to
        # its success identity when every new sample lies below threshold.
        owner.set_initial(np.zeros_like(problem.initial.field0), None)
        owner.propagate_to(grid.length / 8)
        assert owner.observe().z == grid.length / 8
    finally:
        owner.close()


def test_envelope_native_batch_keeps_noisy_members_and_statistics_independent() -> None:
    """Noisy members of a batch agree with the same members run alone.

    Each pair shares its stored noise draw and replays eight prescribed steps: the
    same discrete computation on both sides over a short window, held to
    :func:`native_bound` (solprop SEP 25), with the schedule's row positions exact.
    """
    from contextlib import ExitStack

    from solprop.noise import modified_noise

    from solprop_native import NativeBatchBackend

    problem, grid = envelope_case("modal", 2)
    bound = native_bound(problem, grid)
    with ExitStack() as cleanup:
        members: list[NativeBackend] = []
        singletons: list[NativeBackend] = []
        for index in range(2):
            draw = modified_noise(
                grid, np.random.default_rng(390 + index), n_modes=1, n_pol=2
            )
            initial = replace(
                problem.initial,
                field0=problem.initial.field0 * (1 + 0.125 * index),
                noise0=draw.spectrum[None],
                noise_model=draw.model,
            )
            one = replace(
                problem, initial=initial, flags=replace(problem.flags, has_noise=True)
            )
            for destination in (members, singletons):
                owner = NativeBackend(one, grid=grid)
                cleanup.callback(owner.close)
                owner.set_initial(initial.field0, draw)
                destination.append(owner)
        batch = NativeBatchBackend(members)
        cleanup.callback(batch.close)
        for target in (grid.length / 8, grid.length):
            batch.propagate_to(target)
            got = batch.stats(problem.stats_spec)
            for index, singleton in enumerate(singletons):
                singleton.propagate_to(target)
                field = singleton.observe().require_field()
                np.testing.assert_allclose(
                    members[index].observe().require_field(),
                    field,
                    rtol=bound,
                    atol=bound * np.max(np.abs(field)),
                )
                expected = singleton.stats(problem.stats_spec)
                for name in NAMES:
                    np.testing.assert_allclose(
                        got[name][index], expected[name][0], rtol=bound, atol=1e-30
                    )
                left, right = (
                    members[index].observe().step_stats,
                    singleton.observe().step_stats,
                )
                assert left is not None
                assert right is not None
                # Exact: the prescribed schedule's positions.
                np.testing.assert_array_equal(left.z, right.z)
                for name in problem.stats_spec.periodic:
                    np.testing.assert_allclose(
                        left.values[name], right.values[name], rtol=bound, atol=1e-30
                    )
