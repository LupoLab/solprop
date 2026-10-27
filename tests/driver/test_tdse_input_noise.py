# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Input shot noise beside the TDSE response (SEP 37, section 6 and D5).

The input draw is part of the launch, so the only thing it changes is the field the
atom is driven by; the modified draw is kept apart from that field and stays refused.
The atom here is the manufactured zero-current binding of the shared Adams tests,
which records every drive it is handed: what is under test is the drive and the
record, not an atomic calculation. The weak-field statement is the amplitude one --
the draw alone drives the atom far below the intensity at which the TDSE's own
weak-field consistency check certifies its linear response -- because a real atomic
engine is an optional provider these tests do not need.
"""

import dataclasses

import numpy as np
import pytest
from tests.driver.test_sep12_sep14_integration import _Binding

from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.constants import EPS0, C
from solprop.driver import Simulation
from solprop.exceptions import DriverError, TranslationError
from solprop.nonlinear.tdse import (
    TDSEBasis,
    TDSECalibration,
    TDSEExecution,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.stats import StatsSpec
from solprop.stepper.multistep import Multistep

pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.TDSECostWarning")

_STEP = 2.0**-20
"""One atomic step, m; binary-exact, as in the shared Adams fixtures."""

_SEED = 37
"""Seed of the draw."""

WEAK_FIELD_CHECK_INTENSITY = 1e15
"""Peak intensity of the TDSE's weak-field consistency pulse, W/m^2 (1e11 W/cm^2).

The fixed pulse ``docs/guide/media/tdse.md`` names, at which the first-order current
is compared with optmat's polarisability (``|kappa_1 - 1|`` 0.035 for haCC(1)).
"""


def _atomic(noise: str | None) -> Simulation:
    """Return a one-mode helium capillary whose electronic response is a TDSE."""
    parent = Simulation.from_capillary(
        256e-6,
        4 * _STEP,
        "He",
        0.1,
        lambda0=800e-9,
        wavelength_limits=(400e-9, 1.6e-6),
        trange=60e-15,
        fwhm=6e-15,
        energy=1e-12,
        modes=1,
        n_r=4,
        n_save=2,
        plasma=False,
        noise=noise,  # pyright: ignore[reportArgumentType]
        seed=None if noise is None else _SEED,
        backend="numpy_ref",
        integrator=Multistep(formulation="interaction"),
        dz_init=_STEP,
        dz_max=_STEP,
    )
    basis = TDSEBasis(
        "manufactured", (0.0, 1e-9), (8,), 5, 1e-16, source="input-noise fixture"
    )
    calibration = TDSECalibration(
        "sep37-input-noise",
        "research",
        basis.identity,
        (400e-9, 2e-6),
        source="manufactured zero current; no atomic calculation",
        limitations=("no physical qualification",),
    )
    response = TDSEGasResponse(
        "helium", basis, 1e-19, "tdse_native", 1.0, calibration, TDSESampling()
    )
    return Simulation(
        parent.grid,
        parent.modes,
        [response],
        parent._field0,
        settings=dataclasses.replace(
            parent.settings,
            tdse_execution=TDSEExecution(max_atomic_calls=20),
            absorber=None,
            stats=StatsSpec(names=("energy",)),
        ),
        noise=parent.noise,
        profiles=parent.profiles,
        frame=parent.frame,
    )


def _first_drive(simulation: Simulation, *, pulse: bool = True) -> np.ndarray:
    """Return the real field the atom is driven by at its first evaluation."""
    problem = simulation.translate()
    assert problem.tdse is not None
    binding = _Binding(problem.tdse, 20)
    backend = NumpyRefBackend(problem, grid=simulation.grid, tdse_binding=binding)
    try:
        launch = problem.initial.field0 if pulse else np.zeros(problem.shape, complex)
        backend.set_initial(launch, None, z=0.0)
        backend.propagate_to(_STEP)
    finally:
        backend.close()
        binding.close()
    assert binding.purposes[0] == "startup"
    return binding.inputs[0]


@pytest.mark.medium
def test_the_atom_is_driven_by_the_pulse_plus_the_draw_at_the_entrance():
    """The drive is linear in the launch, so its difference is the draw's own drive.

    The rounding budget: the drive is a real transform and a radial synthesis of the
    launch, a few hundred operations per sample at the pulse's own magnitude, so the
    three drives agree to 1e-12 of the noisy one's peak (measured 2.3e-16). The draw
    is 2.5e-3 of the pulse's amplitude here, so a drive that ignored it would miss by
    more than 1e9 times the gate.
    """
    noisy = _atomic("input")
    clean = _first_drive(_atomic(None))
    with_draw = _first_drive(noisy)
    draw = _first_drive(noisy, pulse=False)
    scale = float(np.max(abs(with_draw)))
    assert float(np.max(abs(draw))) > 1e-6 * scale
    np.testing.assert_allclose(with_draw - clean, draw, rtol=0.0, atol=1e-12 * scale)

    # The realisation is stored: the run's record carries the model, the seed and
    # the spectrum the launch was given.
    record = noisy.metadata().noise
    assert record is not None and noisy.noise is not None
    assert (record.model, record.seed) == ("input", _SEED)
    np.testing.assert_array_equal(record.spectrum, noisy.noise.spectrum)

    # The weak-field regime: the draw alone drives the atom at an intensity many
    # orders below the TDSE's weak-field check, where its first-order response is
    # certified against optmat's. The native source subtracts that first-order
    # current, and what remains grows as the intensity relative to the linear current,
    # so at the draw's intensity (measured 3.5e4 W/m^2, 3.5e-11 of the check's) the
    # source the draw alone drives is negligible. The gate leaves room for any grid.
    peak = EPS0 * C * float(np.max(draw**2))
    assert peak < 1e-9 * WEAK_FIELD_CHECK_INTENSITY


def test_the_draw_lies_inside_the_band_the_response_declares():
    """The draw occupies the active band and nothing else, as the pulse does.

    That band is the one the gated source is certified on (its steady-current
    certificate is taken over exactly these bins at translation), so the draw adds no
    frequency the response was not declared for.
    """
    simulation = _atomic("input")
    assert simulation.noise is not None
    problem = simulation.translate()
    problem.validate()
    assert problem.tdse is not None
    occupied = np.any(simulation.noise.spectrum != 0.0, axis=(0, 1))
    assert np.array_equal(occupied, problem.grids.active_mask)
    assert not np.any(occupied & (problem.grids.omega <= 0.0))


def test_the_modified_draw_is_refused_beside_the_tdse():
    """The modified draw is added at every stage, apart from the drive the atom sees."""
    with pytest.raises(DriverError, match="TDSE refuses noise='modified'"):
        _atomic("modified")
    parent = _atomic(None)
    problem = parent.translate()
    assert problem.tdse is not None
    with pytest.raises(TranslationError, match="scalar resolved modal Adams"):
        dataclasses.replace(
            problem,
            initial=dataclasses.replace(
                problem.initial,
                noise0=np.zeros_like(problem.initial.field0),
                noise_model="modified",
            ),
        ).validate()
