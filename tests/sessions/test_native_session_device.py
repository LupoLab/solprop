# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session's representative device path against the driver, on a device.

SEP 47 phase 3 exercises one device path through the native execution session early
enough to find contract defects (section 11); the complete device matrix is phase 4.
On a CUDA or HIP build the session admits Adams and the embedded Runge-Kutta tableaux
on the modal route, with planes, periodic rows, step planes and host export
(:func:`solprop_native.session.device_refusal`); the rest is refused by name and
served by the transitional driver session, on the device, as before.

Both sessions run the same strict marches over the same device kernels and make their
decisions on the host from the same device reductions, so the control trace, the
rows, the plane statistics and the exported fields agree bitwise, as on a CPU build's
one thread. The session crosses the field to the host only for an export.
"""

import numpy as np
import pytest

from sessions.executors import (
    CONSUMER,
    STORED,
    Started,
    assert_same_run,
    drive,
    entrance,
    fibre,
    window,
)
from sessions.test_native_session import (
    CASES,
    _case,
    _driver_session,
    _make_native,
)
from solprop.backends.driver_session import DriverSession
from solprop.backends.registry import select
from solprop.backends.sessions import start_session
from solprop.translate.run_request import describe_run

native_session = pytest.importorskip("solprop_native.session")

pytestmark = pytest.mark.gpu

DEVICE_CASES = ("rk", "rk-rejections", "adams", "adams-period-3", "rk-replay")
"""The representative path: adaptive and replayed, with rejections, periodic rows on
two periods and step planes."""


@pytest.fixture(scope="module", autouse=True)
def _device() -> None:
    """Skip the module without a device build of the native backend."""
    native = pytest.importorskip("solprop_native")
    if native.build_info().device_backend == "none":
        pytest.skip("the installed solprop-native build has no accelerator backend")


@pytest.mark.parametrize("recipe", [CONSUMER, STORED], ids=["consumer", "stored"])
@pytest.mark.parametrize("case", DEVICE_CASES)
def test_the_device_session_gives_the_drivers_trace_and_fields(case, recipe):
    """SEP 47 sections 7.1 and 11: the device session's run is the driver's, bitwise.

    Every attempt's end, step, norm and acceptance, every observation, row and plane
    statistic, and every exported field, against the driver session on the same
    device; the session's field reaches the host only as its exports.
    """
    problem = _case(**CASES[case])
    request = describe_run(problem)
    baseline = drive(_driver_session(problem, request), export=recipe)
    session = _make_native(problem, request)
    run = drive(Started(session), export=recipe)
    status = session.native.status()
    backend = session._backend  # pyright: ignore[reportPrivateUsage]
    assert backend is not None
    to_host, _ = backend._native_state.crossed_bytes()  # pyright: ignore[reportPrivateUsage]
    session.close()
    assert_same_run(run, baseline)
    assert run.status.attempts == baseline.status.attempts
    assert run.status.rejected == baseline.status.rejected
    if case == "rk-rejections":
        assert run.status.rejected > 0
    # One crossing per export, of the field alone: no row, statistic or decision read
    # the field on the host.
    nbytes = np.prod(problem.shape) * 16
    assert status.exports == len(run.fields)
    assert to_host == status.export_bytes == status.exports * nbytes


def test_the_production_door_serves_the_device_path_natively():
    """SEP 47 phase 3: on a device build ``start_session`` serves an admitted run in
    the native session and a refused one in the driver session, on the device."""
    problem = _case()
    backend = select("native", problem, grid=window(), modes=[fibre().mode])
    session = start_session(backend, problem, describe_run(problem), entrance(problem))
    try:
        assert isinstance(session, native_session.NativeSession)
    finally:
        session.close()
    dense = _case(interpolate=True)
    request = describe_run(dense)
    assert "on a device build" in (native_session.refusal(dense, request) or "")
    backend = select("native", dense, grid=window(), modes=[fibre().mode])
    session = start_session(backend, dense, request, entrance(dense))
    try:
        assert isinstance(session, DriverSession)
    finally:
        session.close()
