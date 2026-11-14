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

import dataclasses
import math

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
from sessions.runs import load_native, native_trial, saved
from sessions.runs import run as run_session
from solprop.backends.session import ExecutionOptions
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, RadialGrid
from solprop.modes import MarcatiliMode, ModeKind, transverse_quadrature
from solprop.nonlinear.raman import RamanResponse
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import MolecularRaman
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.assembly import assemble

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


CHUNK = ExecutionOptions(node_chunk=5)
"""Five-node chunks, which leave a partial chunk on :func:`vector_problem`'s rule."""


def vector_problem(family="envelope", material="H2", route="axis", steps=8):
    """Return :func:`vector_translation`'s problem, grid and modes."""
    problem, _, grid, modes = vector_translation(family, material, route, steps)
    return problem, grid, modes


def vector_translation(family="envelope", material="H2", route="axis", steps=8):
    """Build a short constant-density run with a delayed oblique weak probe.

    :func:`result` runs it in :data:`CHUNK`'s partial chunks. Returns the problem, its
    run's request (SEP 47 section 5.2), the grid and the modes.
    """
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
    problem, run_request = assemble(
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
    return problem, run_request, grid, modes


def result(backend_class, problem, run_request, grid, modes, options=CHUNK):
    """Return the first stage, final state, statistics and buffer ledger.

    The run goes through the backend's session (SEP 47 phase 5); the first stage is the
    reference's own nonlinear pass, or the native state's first stage of a trial from
    the launch, loaded into the state itself.
    """
    backend = backend_class(
        problem, run_request, grid=grid, modes=modes, options=options
    )
    try:
        if backend.name == "numpy_ref":
            backend.set_initial(problem.initial.field0, None)
            backend._hoist(0.0)
            bundle = backend._stage_scalars(0.0, 0.0)
            rhs = backend._nonlinear(
                problem.initial.field0[None],
                bundle,
                np.ones((1, *problem.shape), complex),
            )
        else:
            assert backend.name == "native"
            assert problem.initial.noise0 is None
            backend._staged_rows()
            state = load_native(backend, problem.initial.field0, clear_fsal=False)
            native_trial(backend, 0.0, LENGTH / 8)
            rhs = np.array(state.stage_to_host(0))
        run = run_session(backend, problem, run_request, owns=False)
        (final, state), *_ = reversed(saved(run))
        assert final.z == LENGTH
        return (
            rhs.reshape(2, -1),
            state.reshape(2, -1),
            dict(final.stats),
            backend.memory_ledger(),
        )
    finally:
        backend.close()


def direction_translation(kind: ModeKind, n: int):
    """Return a two-mode run whose Raman acts along a position-dependent direction.

    A ``kind`` mode of azimuthal order `n` beside HE11, each rotated (0.37 and 0.19
    rad), synthesised on 4 x 8 transverse nodes from :func:`vector_translation`'s
    elliptically polarised launch, hydrogen's Raman alone, two fixed steps. Returns
    the problem, its run's request and the grid.
    """
    original, original_request, grid, _ = vector_translation(steps=2)
    modes = [
        MarcatiliMode(125e-6, kind=kind, n=n, phi=0.37, cladding=1.45, loss=False),
        MarcatiliMode(125e-6, phi=0.19, cladding=1.45, loss=False),
    ]
    initial = original.initial.field0.transpose(1, 0, 2).copy()
    problem, run_request = assemble(
        grid,
        modes,
        [former_response(FORMER_H2, thg=False)],
        initial,
        density=1e25,
        quadrature=transverse_quadrature(modes, n_r=4, n_theta=8),
        controller=original_request.stepper.controller,
        save_planes=[grid.length],
        dz_max=grid.length / 2,
        dz_init=grid.length / 2,
        schedule=original_request.stepper.schedule,
    )
    assert problem.transform.n_pol * problem.shape[1] == 2
    return problem, run_request, grid


def raman_populations(varying_density: bool, geometry: str = "modal"):
    """Return the two envelope plasma populations, each with its own Raman kernel.

    ``host_cases_envelope_plasma``'s two synthetic populations with hydrogen's and
    nitrogen's former Raman responses labelled to them, on `geometry`; with
    `varying_density` each population's density falls linearly to a fifth over the
    8 um. Returns the problem, its run's request and the grid.
    """
    # Imported here: the module imports the native wheel where it is installed.
    from backends.host_cases_envelope_plasma import (  # noqa: PLC0415
        build_envelope_mixture,
        geometry_case,
    )

    mixture, plasma = build_envelope_mixture()
    if varying_density:
        mixture = dataclasses.replace(
            mixture,
            components=tuple(
                dataclasses.replace(
                    component,
                    density=Linear(
                        float(component.density(np.array([0.0]))[0]),
                        float(component.density(np.array([0.0]))[0]) * 0.2,
                        8e-6,
                    ),
                )
                for component in mixture.components
            ),
        )
    # Synthetic population labels with independently audited molecular kernels
    # isolate density routing from any proposed new molecular ionisation model.
    responses = plasma + tuple(
        dataclasses.replace(
            former_response(material, thg=False), species=component.label
        )
        for material, component in zip(
            (FORMER_H2, FORMER_N2), mixture.components, strict=True
        )
    )
    problem, run_request, grid = geometry_case((mixture, responses), geometry, 2)
    assert len(problem.responses.raman_terms) == 2
    return problem, run_request, grid


SPATIAL_LENGTH = 1e-5
"""The spatial tails' run, m: two fixed steps of 5 um."""


def spatial_tails(geometry: str, family: str):
    """Return hydrogen's vector Raman on a radial or Cartesian beam, two fixed steps.

    An elliptically polarised 80 um Gaussian beam (Jones ``(1, 0.35j)``) of a 20 fs,
    100 nJ pulse in free space, on a radial grid of 8 nodes or a 4 x 4 Cartesian grid,
    no collar; hydrogen's former Raman response beside a zero Kerr term at 1e25 m^-3,
    the third harmonic kept on a real field. The actual-width tails of the
    convolution are exercised. Returns the problem, its run's request and the grid.
    """
    constructor = EnvelopeGrid if family == "envelope" else FieldGrid
    grid = constructor.from_limits(SPATIAL_LENGTH, 800e-9, (500e-9, 1.2e-6), 100e-15)
    space = (
        RadialGrid(6e-4, 8, collar_fraction=0.0)
        if geometry == "radial"
        else FreeGrid(3e-4, 4, 4e-4, 4, collar_fraction=0.0)
    )
    initial = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-7),
        80e-6,
        jones=(1, 0.35j),
    ).build(grid, space, medium=SpatialMedium())
    problem, run_request = assemble(
        grid,
        None,
        [
            KerrResponse(0.0, thg=family == "field_real"),
            former_response(FORMER_H2, thg=family == "field_real"),
        ],
        initial,
        density=1e25,
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[SPATIAL_LENGTH],
        dz_max=SPATIAL_LENGTH / 2,
        dz_init=SPATIAL_LENGTH / 2,
        schedule=FixedSchedule.uniform(0.0, SPATIAL_LENGTH, SPATIAL_LENGTH / 2),
    )
    return problem, run_request, grid
