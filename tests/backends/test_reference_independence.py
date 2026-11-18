# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy reference runs on its own host machinery.

SEP 47 section 6.6: the reference session is built independently of the native
backend. The reference's marches are its own
(:mod:`solprop.backends.numpy_ref.marches`), and the run layer serves a reference
backend with its own session (:class:`~solprop.backends.numpy_ref.session.HostSession`).
"""

from contextlib import closing

import pytest

from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.marches.adams import HostAdamsMarch
from solprop.backends.numpy_ref.marches.low_storage import HostLowStorageMarch
from solprop.backends.numpy_ref.session import HostSession
from solprop.backends.records import adams_policy
from solprop.backends.session import Start
from solprop.backends.sessions import start_session
from solprop.exceptions import BackendError
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, Multistep


def test_the_reference_runs_its_own_marches(soliton_problem, narrow_grid):
    length = 1e-3
    schedule = FixedSchedule.uniform(0.0, length, length / 4, save_planes=[length])
    marches = {
        "adams": (Multistep(), HostAdamsMarch),
        "low_storage": (LowStorageRungeKutta(), HostLowStorageMarch),
    }
    for integrator, march in marches.values():
        problem, run_request = soliton_problem(
            narrow_grid,
            1,
            save_planes=[length],
            schedule=schedule,
            dz_max=length / 4,
            dz_init=length / 4,
            integrator=integrator,
        )
        with closing(
            NumpyRefBackend(problem, run_request, grid=narrow_grid)
        ) as backend:
            assert type(backend.march) is march
            assert type(backend.method(None).march) is march


def test_the_run_layer_serves_the_reference_with_the_host_session(
    soliton_problem, narrow_grid
):
    problem, run_request = soliton_problem(narrow_grid, 1, save_planes=[1e-3])
    request = run_request
    reference = NumpyRefBackend(problem, run_request, grid=narrow_grid)
    with start_session(reference, problem, request, Start.entrance(problem)) as session:
        assert type(session) is HostSession


def test_an_injected_adams_engine_is_gated_and_checked_as_the_bridge_s(
    soliton_problem, narrow_grid
):
    # The reference's march takes an injected engine and names what it lacks.
    problem, run_request = soliton_problem(narrow_grid, 1, save_planes=[1e-3])
    with (
        closing(NumpyRefBackend(problem, run_request, grid=narrow_grid)) as backend,
        pytest.raises(BackendError, match="Adams engine lacks accept, begin_attempt"),
    ):
        HostAdamsMarch(
            object(),
            Multistep(),
            None,
            policy=adams_policy(
                backend._problem, run_request.stepper, Multistep(), carrier=False
            ),
            field=backend.workspace.committed,
        )
