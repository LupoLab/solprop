# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise foreign storage aliases through the actual managed PythonCall bridge."""

import os
from pathlib import Path

import pytest

from solprop.backends.julia._bridge import invoke
from solprop.backends.julia._runtime import JuliaRuntime, get_runtime

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _run(runtime: JuliaRuntime, source: Path) -> None:
    # Test helpers have their own Julia module. Wrappers stay inside this guarded
    # helper, including an assertion failure's sanitized traceback.
    module = runtime.namespace.seval("module TransferAliasQualification end")
    module.include(str(source))


def test_real_pythoncall_preserves_foreign_aliases_and_rejects_noninjective_outputs():
    runtime = get_runtime(setup_id=os.environ["SOLPROP_TEST_JULIA_SETUP_ID"])
    source = (
        Path(__file__).resolve().parents[3] / "julia/test/transfer_aliases_bridge.jl"
    )
    invoke(lambda: _run(runtime, source), "Julia transfer alias qualification failed")
