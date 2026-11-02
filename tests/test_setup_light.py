# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The lightweight contributor setup (SEP 43 section 6.7)."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "tools" / "setup_light.py"
spec = importlib.util.spec_from_file_location("setup_light", SCRIPT)
assert spec is not None and spec.loader is not None
setup_light = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_light)


@pytest.mark.parametrize("status", [0, 3])
def test_it_skips_native_and_explains_only_after_success(monkeypatch, capsys, status):
    calls = []

    def run(command, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, status)

    monkeypatch.setattr(setup_light.subprocess, "run", run)
    assert setup_light.main() == status
    assert calls == [
        ("uv", "sync", "--locked", "--no-install-package", "solprop-native")
    ]
    assert ("UV_NO_SYNC=1" in capsys.readouterr().out) == (status == 0)
