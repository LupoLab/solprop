# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Small deterministic Cartesian Raman problems shared by qualification lanes.

The molecular responses are the supplied route's, built from solprop's former Raman
constants (``former_raman``, from its registry at dc4ca0d) and named here by gas:
"H2" has a rotational ladder (``rank2``) and a vibration, "SF6" a vibration only.
They are test inputs for the tensor numerics, not the built-in gases, whose
responses are optmat's (SEP 3).
"""

import math
from dataclasses import replace

import numpy as np

from former_raman import (
    FORMER_CH4,
    FORMER_D2,
    FORMER_H2,
    FORMER_N2,
    FORMER_O2,
    FORMER_SF6,
    former_response,
)
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.raman import RamanResponse
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import MolecularRaman
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble

LENGTH = 0.0002

CASES = {
    str(getattr(one.gas, "value", one.gas)): one
    for one in (FORMER_H2, FORMER_D2, FORMER_N2, FORMER_O2, FORMER_CH4, FORMER_SF6)
}
"""The case names these problems take, and the former constants each one means."""


def raman_response(
    material: str | MolecularRaman = "H2", *, thg: bool = True
) -> RamanResponse:
    """Return the Raman response of a case: a supplied molecular response.

    Parameters
    ----------
    material : str or MolecularRaman, optional
        A case name of :data:`CASES`, or the constants themselves.
    thg : bool, optional
        Whether a field grid keeps the third harmonic. Default ``True``.

    Returns
    -------
    RamanResponse
        The rotational ladder (``rank2``) when the constants have one, then the
        vibration (``isotropic``).
    """
    constants = CASES[material] if isinstance(material, str) else material
    return former_response(constants, thg=thg)


def vector_problem(
    family="envelope", material="H2", route="axis", steps=8, chunk: int | None = 5
):
    """Build a short constant-density run with a delayed oblique weak probe."""
    grid_class = EnvelopeGrid if family == "envelope" else FieldGrid
    grid = grid_class.from_limits(LENGTH, 800e-9, (300e-9, 2e-6), 100e-15)
    mode = MarcatiliMode(125e-6, cladding=1.45, loss=False)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=15e-15, energy=2e-7).build(grid)
    # A frequency-domain delay uses the pulse's internal FFT sign. Both helicities
    # and an oblique delayed probe make the launch time-dependent in polarisation.
    delayed = pulse * np.exp(-1j * grid.omega * 20e-15)
    polarised = np.concatenate(
        [pulse / np.sqrt(2) + 0.02 * delayed, 1j * pulse / np.sqrt(2) - 0.03 * delayed],
        axis=1,
    )
    if route == "rows":
        modes = [mode, mode.rotated(math.pi / 2)]
        initial = polarised.transpose(1, 0, 2).copy()
    else:
        modes = [mode]
        initial = np.sqrt(2) * polarised[:, :1] if route == "scalar" else polarised
    thg = family == "field_real"
    responses = [KerrResponse(chi3=1e-50, thg=thg), raman_response(material, thg=thg)]
    problem = assemble(
        grid,
        modes,
        responses,
        initial,
        density=1e25,
        quadrature=transverse_quadrature(modes, n_r=16),
        controller=PIController(rtol=1e-9, atol=1e-15),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH / steps,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / steps),
        stats_spec=StatsSpec(names=("energy", "peak_power_total", "peak_intensity")),
    )
    if chunk is not None:
        problem = replace(
            problem, transform=replace(problem.transform, node_chunk=chunk)
        )
    return problem, grid, modes


def result(backend_class, problem, grid, modes):
    """Return the first stage, final state, statistics and buffer ledger."""
    backend = backend_class(problem, grid=grid, modes=modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        if backend.name == "numpy_ref":
            backend._hoist(0.0)
            bundle = backend._stage_scalars(0.0, 0.0)
            rhs = backend._nonlinear(
                problem.initial.field0, bundle, np.ones(problem.shape, complex)
            )
        else:
            backend._trial_step(0.0, LENGTH / 8)
            if backend.name == "native":
                rhs = backend._native_state.stage_to_host(0).copy()
            else:
                rhs = backend._stages[0].copy()
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(LENGTH)
        return (
            rhs.reshape(2, -1),
            backend.observe().require_field().copy().reshape(2, -1),
            backend.stats(problem.stats_spec),
            backend.memory_ledger(),
        )
    finally:
        backend.close()
