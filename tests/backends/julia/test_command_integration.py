# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise Julia management through the shared solprop command dispatcher."""

import sys

from solprop.backends.julia import command
from solprop.cli import main
from solprop.exceptions import BackendError


def test_status_routes_without_creating_a_depot_or_importing_julia(
    tmp_path, monkeypatch, capsys
):
    depot = tmp_path / "depot"
    monkeypatch.setenv("SOLPROP_JULIA_DEPOT", str(depot))
    assert main(["julia", "status"]) == 0
    assert "No published Julia setup" in capsys.readouterr().out
    assert not depot.exists()
    assert "juliacall" not in sys.modules


def test_setup_routes_explicit_options(monkeypatch, setup_record, capsys):
    received = []

    def setup(**kwargs):
        received.append(kwargs)
        return setup_record

    monkeypatch.setattr(command, "setup", setup)
    monkeypatch.setattr(command, "describe", lambda: "prepared stack\n")
    assert main(["julia", "setup", "--device", "metal", "--cpu-target", "generic"]) == 0
    assert received == [
        {
            "devices": ["metal"],
            "julia": None,
            "cuda_runtime": None,
            "cuda_local": False,
            "cpu_target": "generic",
        }
    ]
    assert setup_record.setup_id in capsys.readouterr().out


def test_management_refusal_uses_shared_error_exit(monkeypatch, capsys):
    def refused(**_):
        raise BackendError("the julia bridge needs its matching extra")

    monkeypatch.setattr(command, "setup", refused)
    assert main(["julia", "setup"]) == 2
    assert "matching extra" in capsys.readouterr().err
