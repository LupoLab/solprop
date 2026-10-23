# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Each backend evaluates its stage tables and Adams bases with its own helpers.

SEP 29 section 6.6 (R16): the NumPy reference never runs the native helpers or the
Numba dispatchers behind the per-stage and per-step evaluations, and a native run never
reaches those dispatchers, even with both backends run in one process. Each check
translates a fresh problem, so that no table's kept rows from an earlier run stand in
for a call, then replaces the forbidden implementations -- including translation's own
one-point pass, which no backend's run needs -- by ones that raise, counts the allowed
one, and runs a short propagation. The pressure gradient makes the linear
operator and the response prefactor parameter expansions, so the stage rows read
``Phi`` and the prefactor besides the density. Every march runs: both Adams
formulations (the exponential one on a varying operator also integrates its
compensation), Runge-Kutta, whose trial reads its stages together, and the
low-storage march; and a native run on the z table, whose values the native backend
evaluates itself.
"""

import dataclasses

import numpy as np
import pytest

from capability_routes import LENGTH, STEP, capillary_gradient
from driver.test_specified_paths import GRID, PULSE, linear_walkoff
from driver.test_specified_paths import LENGTH as MOVING_LENGTH
from solprop import _helpers
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.grid import EnvelopeGrid
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, RungeKutta
from solprop.stepper import multistep as stepper_multistep
from solprop.stepper.multistep import Multistep
from solprop.translate import chebyshev

pytestmark = pytest.mark.medium

native = pytest.importorskip("solprop_native")
NATIVE = pytest.importorskip("solprop_native._helpers").NATIVE

INTEGRATORS = {
    "adams-interaction": Multistep(formulation="interaction"),
    "adams-exponential": Multistep(formulation="exponential"),
    "runge-kutta": RungeKutta(),
    "low-storage": LowStorageRungeKutta(),
}
"""Each march: the Adams ones read the basis, every one reads the stage tables."""


class ForbiddenHelperError(AssertionError):
    """A backend reached an implementation it must not use."""


def forbid(monkeypatch, record):
    """Make both of a record's operations raise."""

    def refuse(*_):
        raise ForbiddenHelperError(record.name)

    monkeypatch.setitem(record.__dict__, "points", refuse)
    monkeypatch.setitem(record.__dict__, "basis", refuse)


def forbid_numba(monkeypatch):
    """Make the Numba record and the factories of its dispatchers raise."""
    forbid(monkeypatch, _helpers.NUMBA)

    def refuse():
        raise ForbiddenHelperError("a Numba dispatcher")

    monkeypatch.setattr(chebyshev, "_clenshaw_batch", refuse)
    # Translation's one-point pass too: the problem is translated before this.
    monkeypatch.setattr(chebyshev, "_clenshaw", refuse)
    monkeypatch.setattr(stepper_multistep, "_basis", refuse)


def count(monkeypatch, record):
    """Count a record's calls; return the tally."""
    tally = {"points": 0, "basis": 0}
    for name in tally:
        function = getattr(record, name)

        def counted(*arguments, _name=name, _function=function):
            tally[_name] += 1
            return _function(*arguments)

        monkeypatch.setitem(record.__dict__, name, counted)
    return tally


def translated(integrator):
    """Return a fresh gradient-capillary simulation and its problem."""
    simulation = capillary_gradient(None)
    method = INTEGRATORS[integrator]
    changes: dict[str, object] = {"integrator": method}
    if isinstance(method, LowStorageRungeKutta):
        # It replays a prescribed schedule, the same two steps, which the temporal
        # collar's exact integration keeps stable.
        changes["schedule"] = FixedSchedule.uniform(
            0.0, LENGTH, STEP, save_planes=[LENGTH]
        )
        changes["absorber_scheme"] = "split_step"
    simulation.settings = dataclasses.replace(simulation.settings, **changes)
    return simulation, simulation.translate()


def propagate(backend_class, prepared):
    """Construct the backend on a translated problem and run its two short steps."""
    simulation, problem = prepared
    backend = backend_class(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        for plane in problem.stepper.save_planes:
            backend.propagate_to(float(plane))
        observed = backend.observe()
        assert observed.field is not None
    finally:
        backend.close()


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_the_reference_runs_only_the_plain_forms(monkeypatch, integrator):
    # Run native first in the same process, so its record is loaded and live.
    propagate(native.NativeBackend, translated(integrator))
    prepared = translated(integrator)
    forbid(monkeypatch, NATIVE)
    forbid_numba(monkeypatch)
    tally = count(monkeypatch, _helpers.PLAIN)
    propagate(NumpyRefBackend, prepared)
    assert tally["points"] > 0, tally
    assert tally["basis"] > 0 or not integrator.startswith("adams"), tally


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_native_runs_never_reach_the_numba_dispatchers(monkeypatch, integrator):
    # Run the reference first in the same process, then forbid everything but native.
    propagate(NumpyRefBackend, translated(integrator))
    prepared = translated(integrator)
    forbid_numba(monkeypatch)
    forbid(monkeypatch, _helpers.PLAIN)
    tally = count(monkeypatch, NATIVE)
    propagate(native.NativeBackend, prepared)
    assert tally["points"] > 0, tally
    assert tally["basis"] > 0 or not integrator.startswith("adams"), tally


def z_table(integrator):
    """Return a fresh moving-dispersion fibre on the z table and its problem.

    ``Phi`` takes the z table (its declared parameters are not the whole of its z
    dependence), which the native CPU build evaluates in its own kernel (SEP 29
    section 6.6): the stage rows weigh the table's slots, so no Chebyshev pass of
    solprop's runs per stage.
    """
    grid = EnvelopeGrid.from_limits(MOVING_LENGTH, 835e-9, (600e-9, 1300e-9), 2e-12)
    simulation = Simulation.from_gnlse(
        0.1,
        MOVING_LENGTH,
        fibre=linear_walkoff(grid),
        shock=False,
        n_save=2,
        noise=None,
        schedule=FixedSchedule(np.full(4, MOVING_LENGTH / 4)),
        integrator=INTEGRATORS[integrator],
        **GRID,
        **PULSE,
    )
    problem = simulation.translate()
    assert problem.linear.phi_representation == "table"
    return simulation, problem


@pytest.mark.parametrize(
    "integrator",
    ["adams-exponential", "adams-interaction", "runge-kutta", "low-storage"],
)
def test_a_native_z_table_run_never_reaches_the_numba_dispatchers(
    monkeypatch, integrator
):
    # The exponential Adams march on a varying operator reads its compensation's rate
    # from the native derivative table too.
    prepared = z_table(integrator)
    forbid_numba(monkeypatch)
    forbid(monkeypatch, _helpers.PLAIN)
    tally = count(monkeypatch, NATIVE)
    propagate(native.NativeBackend, prepared)
    assert tally["basis"] > 0 or not integrator.startswith("adams"), tally


@pytest.mark.parametrize(
    "integrator",
    ["adams-exponential", "adams-interaction", "runge-kutta", "low-storage"],
)
def test_a_reference_z_table_run_reads_the_table_with_the_plain_forms(
    monkeypatch, integrator
):
    # The reference evaluates the z table itself, per stage (`delta_phi`, and the
    # compensation's rate): with its own helpers, never the shared compiled pass.
    prepared = z_table(integrator)
    forbid(monkeypatch, NATIVE)
    forbid_numba(monkeypatch)
    tally = count(monkeypatch, _helpers.PLAIN)
    propagate(NumpyRefBackend, prepared)
    assert tally["points"] > 0, tally
