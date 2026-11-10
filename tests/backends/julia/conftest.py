# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Share inert setup records for isolated runtime boundary tests."""

import pytest

from solprop.backends.julia._identity import SetupRecord


@pytest.fixture
def setup_record(tmp_path):
    return SetupRecord(
        setup_id="a" * 64,
        engine_digest="b" * 64,
        bundle_digest="c" * 64,
        julia_executable=tmp_path / "bin/julia",
        julia_version="1.13.3",
        julia_binary_digest="d" * 64,
        bridge_version="0.9.37",
        depot=tmp_path / "depot",
        engine=tmp_path / "engine",
        projects=(("cpu", tmp_path / "cpu"), ("cuda", tmp_path / "cuda")),
        manifests=(),
        inventories=(),
        platform="linux-x86_64",
        cpu_target="generic",
        cpu_name="generic",
        cuda_runtime="12.8",
        cuda_local=False,
        elapsed_seconds=1.0,
    )
