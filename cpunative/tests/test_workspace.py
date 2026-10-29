# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise admission, ownership, trials and ordinary output through the seam."""

import copy
import dataclasses
import os
import subprocess
import sys

import numpy as np
import pytest
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.exceptions import BackendError
from solprop.output import open_file
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule
from solprop.translate.stage import terms

from solprop_cpunative import _core
from solprop_cpunative.backend import (
    CPUNativeBackend,
    _payload,
    _stage,
    memory_estimate,
)


@pytest.fixture(autouse=True)
def pools(monkeypatch):
    monkeypatch.setenv("SOLPROP_THREADS", "1")
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "estimate")


def simulation(**changes):
    kwargs = dict(
        lambda0=800e-9,
        wavelength_limits=(400e-9, 2e-6),
        trange=100e-15,
        fwhm=20e-15,
        energy=1e-7,
        modes=2,
        thg=True,
        plasma=False,
        raman=False,
        noise=None,
        n_save=3,
        backend="cpunative",
        stats=StatsSpec(("energy",)),
    )
    kwargs.update(changes)
    return Simulation.from_capillary(125e-6, 2e-5, "He", 1.0, **kwargs)


@pytest.fixture
def small():
    sim = simulation()
    return sim, sim.translate()


@pytest.mark.parametrize("chunk", [1, 2, 7])
@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("provider", ["hand", "blas"])
def test_source_chunks_and_threads(small, chunk, threads, provider, monkeypatch):
    sim, original = small
    problem = dataclasses.replace(
        original,
        transform=dataclasses.replace(original.transform, node_chunk=chunk),
        responses=dataclasses.replace(
            original.responses, temporal_absorber=None, temporal_absorber_fine=None
        ),
    )
    monkeypatch.setenv("SOLPROP_THREADS", str(threads))
    monkeypatch.setenv("SOLPROP_CPUNATIVE_CONTRACTION", provider)
    fields = []
    for cls in (NumpyRefBackend, CPUNativeBackend):
        b = cls(problem, grid=sim.grid, modes=sim.modes)
        b.set_initial(problem.initial.field0, None)
        if cls is CPUNativeBackend:
            fields.append(
                np.asarray(b._cpu.evaluate(_stage(b._stage_scalars(0.0, 0.0)))).reshape(
                    problem.shape
                )
            )
            assert sum(b.memory_ledger().values()) == memory_estimate(problem)
            assert sum(b.memory_ledger().values()) == b._cpu.memory_bytes
            assert b.memory_ledger()["cpunative.state"] == problem.initial.field0.nbytes
        else:
            fields.append(b._adams_lab_source(b._state, 0.0))
        b.close()
    # Independent FFT and short dot-product rounding budget, fixed before execution.
    scale = np.linalg.norm(fields[0])
    assert np.linalg.norm(fields[1] - fields[0]) < 256 * np.finfo(float).eps * scale


@pytest.mark.parametrize("method", ["adams", "rk"])
@pytest.mark.parametrize("scheme", ["split_step", "nonlinear"])
def test_absorber_reset_and_retained_fields(small, method, scheme):
    sim, original = small
    dz = 2e-5 / 8
    p = dataclasses.replace(
        original,
        responses=dataclasses.replace(
            original.responses,
            absorber_scheme=scheme,
            temporal_absorber=original.responses.temporal_absorber * 1e-3,
            temporal_absorber_fine=original.responses.temporal_absorber_fine * 1e-3,
        ),
        stepper=dataclasses.replace(
            original.stepper,
            integrator=None if method == "rk" else original.stepper.integrator,
            schedule=FixedSchedule.uniform(0.0, 2e-5, dz, save_planes=[1e-5, 2e-5]),
        ),
    )
    p.responses.temporal_absorber.flags.writeable = False
    p.responses.temporal_absorber_fine.flags.writeable = False
    # Shift the packet into the collar so the absorber is actually exercised.
    field = p.initial.field0 * np.exp(1j * p.grids.omega * 38e-15)
    observed = []
    for cls in (NumpyRefBackend, CPUNativeBackend):
        b = cls(p, grid=sim.grid, modes=sim.modes)
        b.set_initial(field, None)
        b.propagate_to(1e-5)
        held = b.observe().require_field()
        saved = held.copy()
        b.propagate_to(2e-5)
        end = b.observe().require_field()
        np.testing.assert_array_equal(
            held, saved
        )  # Host observations own their storage.
        assert np.count_nonzero(end[..., ~p.grids.active_mask]) == 0
        observed.append(end.copy())
        b.set_initial(field, None)
        b.propagate_to(1e-5)
        np.testing.assert_allclose(
            b.observe().require_field(),
            saved,
            rtol=1e-12,
            atol=1e-12 * np.max(np.abs(saved)),
        )
        b.close()
        b.close()
        if cls is CPUNativeBackend:
            assert not b.memory_ledger()
    np.testing.assert_allclose(
        observed[1], observed[0], rtol=1e-11, atol=1e-11 * np.max(np.abs(observed[0]))
    )


def test_dopri_rejection_and_fsal(small):
    sim, original = small
    p = dataclasses.replace(
        original, stepper=dataclasses.replace(original.stepper, integrator=None)
    )
    b = CPUNativeBackend(p, grid=sim.grid, modes=sim.modes)
    b.set_initial(p.initial.field0, None)
    initial = b.observe().require_field()
    b._trial_step(0.0, 1e-5)  # Discarded trial must not commit state or FSAL.
    np.testing.assert_array_equal(b.observe().require_field(), initial)
    assert b._cpu.evaluations == 7
    _, trial = b._trial_step(0.0, 1e-6)
    assert b._cpu.evaluations == 14
    b._finish(trial, 1e-6)
    _, trial = b._trial_step(1e-6, 1e-6)
    assert b._cpu.evaluations == 20
    b._finish(trial, 1e-6)
    b.set_initial(p.initial.field0, None)
    assert b._cpu.evaluations == 0
    b.close()


def test_snapshot_resume(small):
    sim, p = small
    a, b = [CPUNativeBackend(p, grid=sim.grid, modes=sim.modes) for _ in range(2)]
    a.set_initial(p.initial.field0, None)
    a.propagate_to(1e-5)
    snapshot = a.march_snapshot()
    assert snapshot is not None
    b.set_initial(a.observe().require_field(), None, z=1e-5)
    assert b.restore_march(snapshot) is None
    for backend in (a, b):
        backend.propagate_to(2e-5)
    field = a.observe().require_field()
    np.testing.assert_allclose(
        b.observe().require_field(),
        field,
        rtol=1e-11,
        atol=1e-11 * np.max(np.abs(field)),
    )
    a.close()
    b.close()


def test_adams_rejection_preserves_resident_history(small, monkeypatch):
    sim, p = small
    rejected, direct = [
        CPUNativeBackend(p, grid=sim.grid, modes=sim.modes) for _ in range(2)
    ]
    for backend in (rejected, direct):
        backend.set_initial(p.initial.field0, None)
        backend._adams._start()
    engine = rejected._resident
    before = engine.snapshot_rows()
    initial = rejected.observe().require_field()
    trial = engine.exponential_trial

    def forced(*args):
        return np.full_like(trial(*args), 1e9)

    monkeypatch.setattr(engine, "exponential_trial", forced)
    accepted, norm = rejected._adams._attempt_exponential(1e-6)
    assert not accepted
    assert norm == 1e9
    np.testing.assert_array_equal(rejected.observe().require_field(), initial)
    for name, row in before.items():
        np.testing.assert_array_equal(engine.snapshot_rows()[name], row)
    monkeypatch.setattr(engine, "exponential_trial", trial)
    for backend in (rejected, direct):
        assert backend._adams._attempt_exponential(1e-6, fixed=True)[0]
    expected = direct.observe().require_field()
    # Identical accepted histories and arithmetic; allow ordinary FFT rounding.
    np.testing.assert_allclose(
        rejected.observe().require_field(),
        expected,
        rtol=1e-12,
        atol=1e-12 * np.max(np.abs(expected)),
    )
    assert rejected._cpu.evaluations == direct._cpu.evaluations + 1
    assert rejected.metadata()["response_calls"]["rejected_steps"] == 1
    rejected.close()
    direct.close()


@pytest.mark.parametrize("method", ["adams", "rk"])
def test_exact_zero_field(small, method):
    _, p = small
    if method == "rk":
        p = dataclasses.replace(
            p, stepper=dataclasses.replace(p.stepper, integrator=None)
        )
    backend = CPUNativeBackend(p)
    backend.set_initial(np.zeros(p.shape, dtype=np.complex128), None)
    backend.propagate_to(2e-5)
    assert np.count_nonzero(backend.observe().require_field()) == 0
    backend.close()


def test_invalid_initial_field_does_not_partly_replace_state(small):
    _, p = small
    backend = CPUNativeBackend(p)
    backend.set_initial(p.initial.field0, None)
    original = backend.observe().require_field()
    invalid = 2 * p.initial.field0
    invalid.flat[-1] = np.nan
    with pytest.raises(BackendError, match="nonfinite initial field"):
        backend.set_initial(invalid, None)
    np.testing.assert_array_equal(backend.observe().require_field(), original)
    backend.close()


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("single", "single precision"),
        ("envelope", "envelope"),
        ("dense", "dense-output"),
        ("batch", "batch"),
        ("schema", "schema"),
        ("polarisation", "vector polarisation"),
        ("maps", "complex modal maps"),
        ("taper", "varying radius"),
        ("spatial", "spatial routes"),
        ("general", "general modal guides"),
        ("coupled", "coupled operators"),
        ("tdse", "TDSE"),
        ("dispersive", "dispersive Kerr"),
        ("formulation", "alternative Adams"),
        ("carrier", "carrier frames"),
    ],
)
def test_refusals_before_allocation(small, case, match, monkeypatch):
    _, p = small
    if case == "single":
        p = simulation(precision="single", backend="numpy_ref").translate()
    elif case == "envelope":
        p = dataclasses.replace(p, grid_kind="envelope")
    elif case == "dense":
        p = dataclasses.replace(
            p, stepper=dataclasses.replace(p.stepper, interpolate_planes=True)
        )
    elif case == "batch":
        p = dataclasses.replace(p, shape=(2, *p.shape[1:]))
    elif case == "polarisation":
        p = dataclasses.replace(p, shape=(1, p.shape[1], 2, p.shape[3]))
    elif case in ("maps", "taper", "spatial", "general"):
        changes = {
            "maps": {"synthesis": p.transform.synthesis.astype(np.complex128)},
            "taper": {"taper_scale": object()},
            "spatial": {"geometry": object()},
            "general": {"modal_tables": object()},
        }
        # Admission must name the unsupported capability before table validation.
        transform = copy.copy(p.transform)
        for name, value in changes[case].items():
            object.__setattr__(transform, name, value)
        p = dataclasses.replace(p, transform=transform)
    elif case == "coupled":
        linear = copy.copy(p.linear)
        object.__setattr__(linear, "coupled_flow", object())
        p = dataclasses.replace(p, linear=linear)
    elif case == "tdse":
        p = copy.copy(p)
        object.__setattr__(p, "tdse", object())
    elif case == "dispersive":
        responses = copy.copy(p.responses)
        object.__setattr__(responses, "dispersive_kerr", object())
        p = dataclasses.replace(p, responses=responses)
    elif case in ("formulation", "carrier"):
        changes = (
            {"formulation": "interaction"}
            if case == "formulation"
            else {"carrier_frame": "scalar"}
        )
        method = dataclasses.replace(p.stepper.integrator, **changes)
        p = dataclasses.replace(
            p, stepper=dataclasses.replace(p.stepper, integrator=method)
        )
    else:
        p = dataclasses.replace(p, schema_version=-1)
    monkeypatch.setattr(
        _core, "State", lambda _: pytest.fail("allocated unsupported workspace")
    )
    with pytest.raises(BackendError, match=match):
        CPUNativeBackend(p)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("modes", -1),
        ("nf", 1),
        ("crop", np.array([999999], dtype=np.uintp)),
        ("window_derivative", np.zeros(1)),
        ("tableau", np.zeros(1)),
    ],
)
def test_extension_validates_dimensions(small, key, value):
    _, p = small
    data = _payload(p, terms(p))
    data[key] = value
    with pytest.raises(ValueError, match="workspace"):
        _core.State(data)


def test_driver_memory_and_hdf5(tmp_path):
    sim = simulation(mode_average=True, modes=1)
    memory = sim.run(progress="none")
    assert memory.n_saved == 3
    path = tmp_path / "cpunative.h5"
    sim.run(path, progress="none")
    with open_file(path) as stored:
        # Same method on repeat: allow ordinary floating-point variation.
        np.testing.assert_allclose(
            stored["field"],
            memory["field"],
            rtol=1e-12,
            atol=1e-12 * np.max(np.abs(memory["field"])),
        )


def test_field_free_planes_keep_statistics():
    sim = simulation(mode_average=True, modes=1)
    sim.settings = dataclasses.replace(sim.settings, fields="none")
    result = sim.run(progress="none")
    assert result.n_saved == 3
    assert result["field_planes"].size == 0
    assert np.isfinite(result["stats/energy"]).all()


@pytest.mark.parametrize(
    "modules",
    [
        ("solprop_cpunative", "solprop_native", "numpy", "scipy", "pyfftw"),
        ("solprop_native", "solprop_cpunative", "numpy", "scipy", "pyfftw"),
        ("pyfftw", "scipy", "numpy", "solprop_cpunative", "solprop_native"),
    ],
)
def test_import_order_and_lifecycle(modules):
    code = f"""
import sys
import importlib
import importlib.util
for module in {modules!r}:
    if importlib.util.find_spec(module) is None:
        continue
    try:
        importlib.import_module(module)
    except RuntimeError as error:
        if not (module == 'solprop_native' and sys.platform == 'darwin'
                and 'before pyfftw' in str(error)):
            raise
from solprop_cpunative.backend import CPUNativeBackend
from benchmarks.builders import capillary
sim = capillary('rdw')
problem = sim.translate()
for _ in range(3):
    backend = CPUNativeBackend(problem)
    backend.set_initial(problem.initial.field0, None)
    backend.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_runtime_queries_preserve_bound_thread_budget():
    code = """
from solprop._process import thread_budget
expected = thread_budget().threads
import solprop_cpunative
from threadpoolctl import threadpool_info
solprop_cpunative.build_info()
threadpool_info()
assert thread_budget().threads == expected
"""
    environment = os.environ.copy() | {
        "SOLPROP_THREADS": "2",
        "SOLPROP_CPUNATIVE_THREADS": "2",
        "OMP_NUM_THREADS": "2",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
