# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy reference runs on its own host machinery.

SEP 47 section 6.6: the reference session is built independently of the native
backend. The reference's marches and its group are the host copies, and the run layer
serves a reference backend with the host session and a reference group with the host
group session.
"""

from contextlib import closing

import pytest

from solprop.backends.host.adams import HostAdamsMarch
from solprop.backends.host.driver import HostDriver
from solprop.backends.host.group import HostGroup
from solprop.backends.host.group_session import HostGroupSession
from solprop.backends.host.low_storage import HostLowStorageMarch
from solprop.backends.host.session import HostSession
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.batch import NumpyBatchBackend
from solprop.backends.session import Start
from solprop.backends.sessions import start_group, start_session
from solprop.exceptions import BackendError
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, Multistep


@pytest.mark.parametrize("backend", [NumpyRefBackend, NumpyBatchBackend])
def test_a_reference_backend_inherits_the_host_driver(backend):
    assert issubclass(backend, HostDriver)


def test_the_reference_marches_and_groups_with_the_host_copies(
    soliton_problem, narrow_grid
):
    length = 1e-3
    schedule = FixedSchedule.uniform(0.0, length, length / 4, save_planes=[length])
    marches = {
        "_adams": (Multistep(), HostAdamsMarch),
        "_low_storage": (LowStorageRungeKutta(), HostLowStorageMarch),
    }
    for attribute, (integrator, march) in marches.items():
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
            assert type(getattr(backend, attribute)) is march
    problem, run_request = soliton_problem(
        narrow_grid,
        1,
        save_planes=[length],
        schedule=schedule,
        dz_max=length / 4,
        dz_init=length / 4,
    )
    member = NumpyRefBackend(problem, run_request, grid=narrow_grid)
    member.set_initial(problem.initial.field0, None)
    with closing(NumpyBatchBackend([member])) as group:
        assert isinstance(group, HostGroup)


def test_the_run_layer_serves_the_reference_with_the_host_session(
    soliton_problem, narrow_grid
):
    problem, run_request = soliton_problem(narrow_grid, 1, save_planes=[1e-3])
    request = run_request
    reference = NumpyRefBackend(problem, run_request, grid=narrow_grid)
    with start_session(reference, problem, request, Start.entrance(problem)) as session:
        assert type(session) is HostSession


def test_the_run_layer_serves_a_reference_group_with_the_host_group_session(
    soliton_problem, narrow_grid
):
    length = 1e-3
    schedule = FixedSchedule.uniform(0.0, length, length / 4, save_planes=[length])
    problem, run_request = soliton_problem(
        narrow_grid,
        1,
        save_planes=[length],
        schedule=schedule,
        dz_max=length / 4,
        dz_init=length / 4,
    )
    members = [NumpyRefBackend(problem, run_request, grid=narrow_grid)]
    with start_group(
        members, [problem], [run_request], [Start.entrance(problem)]
    ) as group:
        assert type(group) is HostGroupSession
        assert isinstance(getattr(group, "_runner"), NumpyBatchBackend)  # noqa: B009


def test_an_injected_adams_engine_is_gated_and_checked_as_the_bridge_s(
    soliton_problem, narrow_grid
):
    # The reference's march takes an injected engine and names what it lacks.
    problem, run_request = soliton_problem(narrow_grid, 1, save_planes=[1e-3])
    with closing(NumpyRefBackend(problem, run_request, grid=narrow_grid)) as backend:
        with pytest.raises(BackendError, match="exponential path lacks"):
            HostAdamsMarch(backend, Multistep(), engine=object())
        with pytest.raises(BackendError, match="fused path lacks"):
            HostAdamsMarch(
                backend, Multistep(formulation="interaction"), engine=object()
            )
