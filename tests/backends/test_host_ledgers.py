# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A reference driver with no atomic service refuses atomic call ledgers.

:meth:`solprop.backends.host.driver.HostDriver.restore_call_ledgers` restores the
logical Adams counts every host driver keeps. Only the reference backend's TDSE binding
keeps atomic counts; any other host driver -- a lockstep group, or a stand-in -- is
refused them by name rather than silently dropping a budget a resume must honour.
"""

import pytest

from scripted_driver import HostScriptedDriver
from sessions.executors import soliton, window
from solprop.exceptions import BackendError


def test_a_host_driver_without_an_atomic_service_refuses_atomic_ledgers():
    driver = HostScriptedDriver(soliton(), grid=window())
    try:
        with pytest.raises(BackendError, match="does not restore atomic"):
            driver.restore_call_ledgers({}, {"attempted": 1})
        # Nothing atomic, and a march with no logical ledger: nothing to install.
        driver.restore_call_ledgers({}, {})
        driver.restore_call_ledgers({}, None)
    finally:
        driver.close()
