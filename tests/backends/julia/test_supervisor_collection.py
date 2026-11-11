# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Collect the POSIX supervisor tests without evaluating platform-only signals."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_COLLECT = """\
import signal
import sys
from pathlib import Path

import pytest

path = Path(sys.argv[1])
sys.path.insert(0, str(path.parents[2]))
for name in sys.argv[2:]:
    if hasattr(signal, name):
        delattr(signal, name)

status = pytest.main([
    '--collect-only', '-q', '-o', 'addopts=',
    '--confcutdir', str(path.parent), str(path),
])
assert status == 0, status
assert 'juliacall' not in sys.modules
"""


@pytest.mark.parametrize("missing", [("SIGKILL",), ("SIGKILL", "SIGTERM", "SIGSEGV")])
def test_supervisor_module_collects_when_posix_signal_constants_are_unavailable(
    missing,
):
    target = Path(__file__).with_name("test_runtime_supervisor.py")
    result = subprocess.run(
        [sys.executable, "-c", _COLLECT, str(target), *missing],
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # No fixture or process supervision runs during collection; all original
    # deadline/readiness cases remain present for the actual POSIX execution lane.
    assert "38 tests collected" in result.stdout
