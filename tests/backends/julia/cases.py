# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Share short binary64 numerical witnesses for the Julia session composition."""

from dataclasses import replace

import numpy as np

from sessions.executors import soliton
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.problem import Translation
from solprop.translate.run_request import ReplaySpec


def cpu_case(*, steps: int | None = 8, rtol=1e-8, dz_max=0.02):
    """Return a two-centimetre soliton without statistics."""
    original = soliton(planes=(0.01, 0.02))
    request = replace(
        original.request,
        error_control=replace(
            original.request.error_control,
            controller=PIController(rtol=rtol, atol=1e-16),
            dz_max=dz_max,
            dz_init=min(1e-3, dz_max),
        ),
        replay=None
        if steps is None
        else ReplaySpec(FixedSchedule(np.full(steps, 0.02 / steps))),
        observation=replace(
            original.request.observation, stats=StatsSpec(names=(), periodic=())
        ),
    )
    return Translation(original.problem, request)
