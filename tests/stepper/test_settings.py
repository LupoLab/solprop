# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The stepper section a run request carries (``solprop.stepper.settings``).

Its construction checks: the planes it adopts, and the controller settings the Adams
march would ignore; the low-storage march's refusals are tested with its backend
(``tests/backends/test_low_storage.py``).
"""

from typing import cast

import numpy as np
import pytest

from solprop.exceptions import TranslationError
from solprop.stepper import DOPRI5, Multistep, PIController
from solprop.stepper.settings import StepperSettings


def test_the_stepper_group_refuses_planes_that_do_not_increase():
    with pytest.raises(TranslationError, match="non-empty and increasing"):
        StepperSettings(
            DOPRI5,
            PIController(rtol=1e-8, atol=1e-14),
            1e-9,
            0.1,
            1e-3,
            1.1,
            None,
            [1.0, 0.5],
            "tabulate",
        )


def test_the_stepper_group_leaves_the_caller_s_planes_writeable():
    planes = np.array([0.5, 1.0])
    settings = StepperSettings(
        DOPRI5,
        PIController(rtol=1e-8, atol=1e-14),
        1e-9,
        0.1,
        1e-3,
        1.1,
        None,
        planes,
        "tabulate",
    )
    held = cast("np.ndarray", settings.save_planes)
    assert planes.flags.writeable
    assert not held.flags.writeable
    # Not a view either: writing the caller's array later leaves the settings alone.
    planes[0] = 0.25
    assert held[0] == 0.5


def _stepper(controller, integrator):
    """Return stepper settings with one controller and one integrator."""
    return StepperSettings(
        DOPRI5, controller, 1e-9, 0.1, 1e-3, 1.1, None, [0.5, 1.0], "tabulate",
        integrator=integrator,
    )  # fmt: skip


def test_the_adams_march_refuses_the_controller_settings_it_ignores():
    # It reads the tolerances, the weights and the region, and chooses its steps by its
    # own safety, min_factor and max_growth: anything else would be silently unused.
    plain = PIController(rtol=1e-8, atol=1e-14, weights=None, region="active")
    _stepper(plain, Multistep())
    tuned = PIController(rtol=1e-8, atol=1e-14, safety=0.5, reject_free=True)
    with pytest.raises(TranslationError, match="safety, reject_free would be ignored"):
        _stepper(tuned, Multistep())
    # The Runge-Kutta march reads them all.
    _stepper(tuned, None)


def test_the_last_plane_is_where_the_run_stops():
    assert _stepper(PIController(rtol=1e-8, atol=1e-14), None).z_end == 1.0
