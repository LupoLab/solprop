# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The host-only seams refuse the native backend by name (SEP 47 section 6.8).

Native runs only in its own session: neither the reference's session nor the host
stepping seam takes it.
"""

import pytest

from sessions.executors import (  # pyright: ignore[reportPrivateUsage]
    _soliton_context,
    entrance,
    soliton,
)
from solprop.backends.host.session import HostSession
from solprop.backends.protocol import stepping
from solprop.backends.registry import available, select
from solprop.exceptions import BackendError

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)


def test_the_reference_session_refuses_a_native_backend():
    problem, request = soliton()
    backend = select("native", problem, request, **_soliton_context())
    try:
        with pytest.raises(BackendError, match="serves a host driver"):
            HostSession(backend, problem, request, entrance(problem))
    finally:
        backend.close()


def test_the_stepping_seam_refuses_a_native_backend():
    problem, request = soliton()
    backend = select("native", problem, request, **_soliton_context())
    try:
        with pytest.raises(BackendError, match="runs in its own session"):
            stepping(backend)
    finally:
        backend.close()
