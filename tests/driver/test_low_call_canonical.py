# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 13's full-length qualification: the selected Adams method on the canonical run.

The 4.2 m helium gas-flow capillary of SEP 13 section 2 runs in seconds on
``numpy_ref``, so it is qualified at full length (an exception, for this case, to the
short-run rule; SEP 13 section 7). The selected method, ``Multistep()`` in the
interaction formulation with the per-step collar this pulse never reaches, is scored
at all 50 planes against IP-DOPRI5 at ``rtol=1e-6``, whose own error against
``rtol=1e-8`` is 1.3e-6 (SEP 13 Part II).
"""

import numpy as np
import pytest

from solprop.driver import prop_capillary
from solprop.gasflow import steady_fill
from solprop.processing import fields
from solprop.stepper import RungeKutta
from solprop.stepper.multistep import Multistep

LENGTH = 4.2
RADIUS = 256e-6
WINDOW = 30e-15


def canonical(rtol, **extra):
    """Run SEP 13's canonical case at full length."""
    fill = steady_fill("He", [0.0, LENGTH / 2, LENGTH], [0.0, 0.2, 0.0], radius=RADIUS)
    return prop_capillary(
        RADIUS,
        LENGTH,
        "He",
        fill,
        wavelength=800e-9,
        fwhm=6e-15,
        energy=1e-3,
        modes=1,
        trange=60e-15,
        wavelength_limits=(90e-9, 1.6e-6),
        rtol=rtol,
        n_save=50,
        noise=None,
        backend="numpy_ref",
        **extra,
    )


def attempts(output):
    """Return the step attempts of a run, accepted and rejected."""
    return sum(plane.step_history.n_steps for plane in output.planes())


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_the_selected_method_meets_its_accuracy_at_a_fifth_of_the_calls():
    reference = canonical(1e-6, integrator=RungeKutta())
    adams = canonical(
        1e-3,
        integrator=Multistep(formulation="interaction"),
        absorber_scheme="split_step",
    )
    time, truth = fields.get_field(reference)
    _, field = fields.get_field(adams)
    window = np.abs(time) <= WINDOW
    error = np.linalg.norm((field - truth)[..., window], axis=-1) / np.linalg.norm(
        truth[..., window], axis=-1
    )
    # 1.3e-3 measured at the worst plane against rtol=1e-8 (SEP 13 Part II); the
    # reference adds its own 1.3e-6. Bounded at a little over twice the measurement.
    assert error.max() < 3e-3
    # One call per attempt, plus a startup at the entrance and one after the 2.1 m
    # port's restart: 292 measured, against IP-DOPRI5's seven per attempt: 742 at the
    # same rtol, 2.5 times as many (1344 and 4.6 times under the step limits before SEP
    # 14). The bounds leave a fifth to a third for platform drift.
    adams_calls = attempts(adams) + 2
    assert adams_calls < 400
    reference_at_same_rtol = canonical(1e-3, integrator=RungeKutta())
    assert 7 * attempts(reference_at_same_rtol) > 2.0 * adams_calls
