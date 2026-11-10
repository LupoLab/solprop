# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native marches' strict files stay strict in the installed build (#135).

ADR-060, SEP 45 and SEP 47 compile the Adams march's decisions, the boundary watch's
predictions, the Runge-Kutta march's positions and norm, the low-storage replay's
positions and table keys, the native session's scheduling, export and statistics
finishing and the transverse edge's judgement without contraction, so that they round
as solprop's Python does. Each file holds a probe, ``p = a * b; return p + c``, which a
fused multiply-add changes. The witness runs in the build that is installed, with its
own flags, link-time optimisation and visibility, beside a control from a contracting
file: where the control fuses, a probe that fused too has lost its file's rule.
"""

from __future__ import annotations

import pytest

from solprop.backends import available

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)
native = pytest.importorskip("solprop_native")

FACTOR_A = 1.0 + 2.0**-30
FACTOR_B = 1.0 - 2.0**-30
ADDEND = -1.0
"""The exact product is 1 - 2**-60, which rounds to 1: separately rounded, the sum is 0,
and fused it is -2**-60."""

FUSING_BASELINES = frozenset({"x86-64-v3", "apple-m1", "neoverse-n1", "armv8.2-a"})
"""The distribution baselines whose targets have fused multiply-add (CMake's
``SOLPROP_NATIVE_CPU_BASELINE``); x86-64-v2, the older processors' variant, has none."""


def test_the_strict_files_round_every_operation_where_the_build_fuses():
    control, *probes = native._core.strict_witness(  # pyright: ignore[reportPrivateUsage]
        FACTOR_A, FACTOR_B, ADDEND
    )
    assert probes == [0.0] * 7, "a strict file's product and sum were fused"
    if control == -(2.0**-60):
        return
    assert control == 0.0, control
    info = native.build_info()
    # A distribution wheel for a baseline with fused multiply-add must fuse the control,
    # or the witness proves nothing where it matters most.
    assert not (
        info.build_profile == "distribution"
        and info.cpu_baseline in FUSING_BASELINES
        and info.device_backend == "none"
    ), f"the control did not fuse on a {info.cpu_baseline} distribution build"
    pytest.skip(
        "this build does not fuse a product and a sum, so the witness proves nothing"
    )
