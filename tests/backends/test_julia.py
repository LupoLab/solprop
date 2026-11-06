# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Julia backend experiment against the NumPy oracle, on short benchmark windows.

Optional: skipped unless ``solprop[julia]`` is installed and ``julia/bootstrap.py`` has
prepared the runtime. ``SOLPROP_JULIA_DEVICE=cuda`` runs the same tests on an NVIDIA GPU
(``pytest -m gpu``). Every kernel body is checked to be one a GPU could take
(``SOLPROP_JULIA_CHECK``), so that a CPU run catches what would fail on a device.
"""

import importlib.util
import os

import numpy as np
import pytest

from benchmarks.builders import build
from benchmarks.registry import CASES
from solprop.backends import select
from solprop.exceptions import BackendError

os.environ.setdefault("SOLPROP_JULIA_CHECK", "1")
DEVICE = os.environ.get("SOLPROP_JULIA_DEVICE", "cpu")
pytestmark = [pytest.mark.medium] + ([pytest.mark.gpu] if DEVICE != "cpu" else [])


@pytest.fixture(scope="module")
def runtime():
    """Start the runtime once per worker process, or skip without one."""
    if importlib.util.find_spec("juliacall") is None:
        pytest.skip("solprop[julia] is not installed")
    from solprop.backends.julia._runtime import runtime as start  # noqa: PLC0415

    try:
        return start()
    except BackendError as exc:
        pytest.skip(f"the Julia runtime is not prepared: {exc}")


# (case, stepper, fraction): each a few steps of a focus route (docs/guide/performance/
# julia.md). The fractions keep the NumPy oracle to seconds.
WINDOWS = [
    ("rdw", "rk", 0.002),
    ("rdw", "default", 0.002),
    ("argon_blueshift", "default", 0.002),
    ("h2_raman_6ps", "default", 0.0005),
    ("radial_filament_256", "default", 0.002),
    ("line", "default", 0.005),
    ("kerr_beam_192", "default", 1.0),
]


def _run(name, configuration):
    """Propagate the configured window on one backend; return the field and steps."""
    simulation = build(configuration | {"backend": name})
    problem = simulation.translate()
    engine = select(
        name,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    engine.set_initial(problem.initial.field0, None, z=problem.initial.z)
    engine.propagate_to(float(configuration["stop"]))
    observation = engine.observe(field=True)
    steps = (observation.n_steps, observation.n_rejected)
    engine.close()
    return observation.field, steps


# The Cartesian beam's fused silica is evaluated past Malitson's range, by design of the
# benchmark case; the warning says so and is not this test's subject.
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize(("case", "stepper", "fraction"), WINDOWS)
def test_julia_matches_the_reference(runtime, case, stepper, fraction):
    """Same steps, and the same field to rounding, as numpy_ref on the same contract."""
    del runtime
    configuration = CASES[case].configuration(
        stepper, 2, "julia", fraction=fraction, noise="none"
    )
    field, steps = _run("julia", configuration)
    reference, reference_steps = _run("numpy_ref", configuration)
    assert steps == reference_steps
    # Matched discrete methods on one step sequence: the fields differ by the rounding
    # of their transforms and sums, which the native/julia A/B measures at <= 1e-13
    # over these windows; 1e-10 leaves three orders for longer accumulation and is
    # still far below the controller's 1e-6 relative tolerance.
    scale = float(np.linalg.norm(reference))
    assert scale > 0
    assert float(np.linalg.norm(field - reference)) <= 1e-10 * scale


def test_julia_refuses_single_precision(runtime):
    """Single precision is not admitted until its error budgets are qualified."""
    del runtime
    configuration = CASES["rdw"].configuration("rk", 1, "julia", fraction=0.001)
    simulation = build(configuration | {"precision": "single", "backend": "numpy_ref"})
    problem = simulation.translate()
    with pytest.raises(BackendError, match="single precision"):
        select("julia", problem, grid=simulation.grid, modes=simulation.modes)


def test_julia_reports_its_engine(runtime):
    """Metadata names the runtime and the sources; the ledger counts owned arrays."""
    del runtime
    configuration = CASES["rdw"].configuration("rk", 1, "julia", fraction=0.001)
    simulation = build(configuration | {"noise": "none"})
    problem = simulation.translate()
    engine = select("julia", problem, grid=simulation.grid, modes=simulation.modes)
    metadata = engine.metadata()
    assert metadata["experimental"] is True
    assert len(metadata["julia_source_digest"]) == 64
    assert metadata["device"].startswith(DEVICE)
    ledger = engine.memory_ledger()
    assert ledger["work.y"] == 16 * problem.shape[1] * problem.shape[3]
    engine.close()
