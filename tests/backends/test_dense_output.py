# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 14 phase 3: save planes observed inside steps through DOPRI5's extension.

``interpolate_planes=True`` steps across the save planes and evaluates each through the
continuous extension of Hairer, Norsett and Wanner II.6 in the interaction picture,
then transports it to the plane. The run's end and every hard kink are still landed
on. The admission matrix refuses what the interpolant does not contain: the per-step
temporal collar, a coupled flow and the ``modified`` noise.
"""

import dataclasses
import math

import numpy as np
import pytest

from solprop.backends import available
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver.simulation import Simulation, _interpolated
from solprop.exceptions import DriverError, TranslationError
from solprop.log import NullReporter
from solprop.noise import modified_noise
from solprop.output import MemoryOutput
from solprop.output.hdf5 import resume
from solprop.output.schema import (
    INTERPOLATED_RESUMES_ANNOTATION,
    PlaneStore,
    StepLimits,
    StepperRecord,
    _hashed_stepper,
)
from solprop.stats import PLASMA_INTEGRATION_STATISTICS
from solprop.stepper import DOPRI5, PIController, RungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.stepper.tableau import (
    TSIT5,
    continuous_extension,
    extension_coefficients,
    extension_value,
)
from solprop.translate.coupled import constant_schur_flow

COMPILED: list[type] = [NumbaCpuBackend]
if "native" in available():
    from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]

    COMPILED.append(NativeBackend)

# The soliton fixture's period (tests/backends/conftest.py).
SOLITON_PERIOD = math.pi / 2 * (100e-15) ** 2 / 11.83e-27
PLANES = list(np.linspace(SOLITON_PERIOD / 16, SOLITON_PERIOD / 2, 8))


def linear_step(rate, start, dz):
    """Return the DOPRI5 stage derivatives of one step of ``y' = rate y``."""
    stages: list[np.ndarray] = []
    for row in DOPRI5.matrix:
        increment = sum(
            (float(a) * k for a, k in zip(row, stages, strict=False)),
            np.zeros_like(start),
        )
        stages.append(rate * (start + dz * increment))
    return stages


def test_the_extension_is_fourth_order_and_meets_the_step_ends():
    rate, start = -1.0 + 3.0j, np.array([1.0 + 0.5j])
    errors = []
    for dz in (0.1, 0.05, 0.025):
        stages = linear_step(rate, start, dz)
        coefficients = extension_coefficients(DOPRI5, start, stages, dz)
        # The ends are the step's own: theta = 0 the start, theta = 1 the solution.
        # At theta = 0 every term but the constant is a product with zero, so the
        # start comes back exactly whatever the evaluation order.
        np.testing.assert_array_equal(extension_value(coefficients, 0.0), start)
        solution = start + dz * sum(
            float(b) * k for b, k in zip(DOPRI5.solution_weights, stages, strict=True)
        )
        np.testing.assert_allclose(
            extension_value(coefficients, 1.0), solution, rtol=1e-15
        )
        exact = start * np.exp(rate * 0.4 * dz)
        errors.append(abs(extension_value(coefficients, 0.4) - exact)[0])
    # A local error of order h^5: halving the step divides it by 32 (measured 34.8
    # and 33.2, approaching it).
    orders = np.log2([errors[0] / errors[1], errors[1] / errors[2]])
    np.testing.assert_allclose(orders, 5.0, atol=0.15)


def test_only_dopri5_has_an_extension_here():
    assert continuous_extension(DOPRI5) is not None
    assert continuous_extension(TSIT5) is None
    with pytest.raises(ValueError, match="no continuous extension"):
        extension_coefficients(TSIT5, np.zeros(1), [np.zeros(1)] * 7, 0.1)


def run(backend_class, problem, grid):
    """Propagate through every plane; return the fields and the attempts."""
    backend = backend_class(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    fields, attempts = [], 0
    for plane in problem.stepper.save_planes:
        backend.propagate_to(float(plane))
        observed = backend.observe()
        fields.append(np.array(observed.field))
        attempts += observed.step_history.n_steps
    backend.close()
    return np.array(fields), attempts


def test_dense_planes_match_landed_ones_at_fewer_steps(narrow_grid, soliton_problem):
    common = {
        "save_planes": PLANES,
        "controller": PIController(rtol=1e-6, atol=1e-18),
        "dz_max": SOLITON_PERIOD,
    }
    landed, landed_attempts = run(
        NumpyRefBackend, soliton_problem(narrow_grid, 2, **common), narrow_grid
    )
    dense, dense_attempts = run(
        NumpyRefBackend,
        soliton_problem(narrow_grid, 2, interpolate_planes=True, **common),
        narrow_grid,
    )
    assert dense_attempts < landed_attempts
    # The end is landed on in both, so it is the same march's to the tolerance; the
    # interior planes differ by the interpolant's error and the two marches' own.
    scale = np.max(np.abs(landed), axis=-1)
    error = np.max(np.abs(dense - landed), axis=-1) / scale
    # Measured at most 2.3e-7 of the peak at rtol 1e-6, over 125 attempts against 130.
    assert np.max(error) < 1e-6


@pytest.mark.parametrize("backend_class", COMPILED, ids=lambda one: one.name)
def test_a_compiled_backend_interpolates_as_the_reference(
    narrow_grid, soliton_problem, backend_class
):
    problem = soliton_problem(
        narrow_grid,
        2,
        save_planes=PLANES,
        controller=PIController(rtol=1e-6, atol=1e-18),
        dz_max=SOLITON_PERIOD,
        interpolate_planes=True,
    )
    reference, reference_attempts = run(NumpyRefBackend, problem, narrow_grid)
    fields, attempts = run(backend_class, problem, narrow_grid)
    assert attempts == reference_attempts
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(fields, reference, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("backend", [one.name for one in COMPILED])
def test_statistics_at_interpolated_planes_match_the_reference(backend):
    # A plasma capillary: the compiled backends' statistics at an interpolated plane
    # are the interpolated field's, not the march state's they last scanned.
    def build(name):
        return Simulation.from_capillary(
            150e-6,
            0.02,
            "He",
            1.0,
            lambda0=800e-9,
            fwhm=6e-15,
            energy=1e-4,
            trange=60e-15,
            wavelength_limits=(90e-9, 1.6e-6),
            n_save=6,
            interpolate_planes=True,
            noise=None,
            backend=name,
        ).run(progress="none")

    reference, compiled = build("numpy_ref"), build(backend)
    assert isinstance(reference, MemoryOutput)
    assert np.max(np.asarray(reference["stats/electron_density"])) > 0.0
    for name in reference.planes()[1].stats:
        if name == "quadrature_error":
            # The compiled backends score the steps of one call, and a plane inside a
            # step already taken has none (nan); the reference scores the field.
            continue
        expected = np.asarray(reference[f"stats/{name}"])
        scale = max(float(np.max(np.abs(expected))), 1e-300)
        if name in PLASMA_INTEGRATION_STATISTICS:
            # SEP 18's integration diagnostics are themselves relative errors (a work
            # defect of 2e-7 of the work freed): their scale is the quantity they are
            # relative to, one, not their own small value, which a forked step moves
            # by a fifth (Linux, both compiled backends).
            scale = 1.0
        # Round-off can flip a step decision between backends and platforms, and the
        # interpolated planes then differ at the interpolant's error: 4.3e-7 of the
        # electron density on Linux numba_cpu and on a CUDA build, 1e-9 on macOS. A
        # statistic read off the march's state instead differs by percent.
        np.testing.assert_allclose(
            np.asarray(compiled[f"stats/{name}"]), expected, rtol=0, atol=1e-5 * scale
        )


def test_the_stepper_settings_refuse_what_cannot_interpolate(
    narrow_grid, soliton_problem
):
    problem = soliton_problem(narrow_grid, 1, save_planes=[0.1])
    stepper = problem.stepper
    with pytest.raises(TranslationError, match="no continuous extension"):
        dataclasses.replace(stepper, tableau=TSIT5, interpolate_planes=True)
    # An Adams integrator interpolates with its own polynomial, whatever the tableau.
    adams = dataclasses.replace(
        stepper,
        tableau=TSIT5,
        integrator=Multistep(formulation="interaction"),
        interpolate_planes=True,
    )
    assert adams.interpolate_planes
    schedule = FixedSchedule(np.full(2, 0.05))
    with pytest.raises(TranslationError, match="a schedule lands"):
        dataclasses.replace(stepper, schedule=schedule, interpolate_planes=True)


@pytest.mark.parametrize(
    "dense",
    [
        {"interpolate_planes": True},
        {
            "integrator": Multistep(formulation="interaction"),
            "interpolate_planes": True,
        },
    ],
)
def test_the_admission_matrix_refuses_noise_outside_the_interpolant(
    narrow_grid, soliton_problem, dense
):
    with pytest.raises(TranslationError, match="'modified' noise model"):
        soliton_problem(
            narrow_grid,
            1,
            save_planes=[0.1],
            noise=modified_noise(narrow_grid, np.random.default_rng(3)),
            **dense,
        )


def test_the_admission_matrix_refuses_a_coupled_flow(narrow_grid, soliton_problem):
    from solprop.translate.problem import _admit_dense  # noqa: PLC0415 - private

    problem = soliton_problem(
        narrow_grid, 1, save_planes=[0.1], interpolate_planes=True
    )
    flow = constant_schur_flow(
        np.sort(problem.grids.omega),
        problem.linear.phi_breakpoints[[0, -1]],
        np.zeros((1, 1, problem.grids.omega.size), dtype=complex),
        dz_max=problem.stepper.dz_max,
        condition_limit=1e6,
        gauge_id="test",
        certificate_id="constant-schur",
    )
    coupled = dataclasses.replace(
        problem, linear=dataclasses.replace(problem.linear, coupled_flow=flow)
    )
    with pytest.raises(TranslationError, match="a coupled linear flow"):
        _admit_dense(coupled)


def capillary(**extra):
    """Return a short helium capillary run with the given settings."""
    return Simulation.from_capillary(
        150e-6,
        0.02,
        "He",
        1.0,
        lambda0=800e-9,
        fwhm=10e-15,
        energy=1e-6,
        trange=100e-15,
        wavelength_limits=(200e-9, 2e-6),
        n_save=4,
        **{"noise": None, **extra},
    )


@pytest.mark.parametrize(
    "dense",
    [
        {"integrator": RungeKutta(), "interpolate_planes": True},
        {
            "integrator": Multistep(formulation="interaction"),
            "interpolate_planes": True,
        },
    ],
    ids=["runge-kutta", "adams"],
)
def test_the_collar_scheme_refuses_interpolated_planes(dense):
    simulation = capillary(absorber_scheme="split_step", **dense)
    with pytest.raises(TranslationError, match="per-step temporal collar"):
        simulation.translate()


class StopError(RuntimeError):
    """Raised by the reporter below to stop a run like a crash."""


class Stop(NullReporter):
    """A progress reporter that stops the run after its second saved plane."""

    def __init__(self):
        self.seen = 0

    def plane(self, z, summary):
        self.seen += 1
        if self.seen >= 2:
            raise StopError("stopped")


def test_a_dense_run_is_recorded_in_the_resume_key():
    simulation = capillary(interpolate_planes=True, backend="numpy_ref")
    assert simulation.settings.interpolate_planes
    metadata = simulation.metadata()
    assert metadata.stepper.limits.interpolate_planes
    landed = capillary(backend="numpy_ref").metadata()
    assert metadata.configuration_hash() != landed.configuration_hash()


@pytest.mark.parametrize(
    "dense",
    [
        {"integrator": RungeKutta(), "interpolate_planes": True},
        {
            "integrator": Multistep(formulation="interaction"),
            "interpolate_planes": True,
        },
    ],
    ids=["runge-kutta", "adams"],
)
def test_an_interpolated_plane_is_resumed_only_on_request(tmp_path, dense):
    path = tmp_path / "dense.h5"
    with pytest.raises(StopError):
        capillary(backend="numpy_ref", **dense).run(path, progress=Stop())
    state = resume(path)
    assert _interpolated(state)
    with pytest.raises(DriverError, match="resume='interpolated'"):
        capillary(backend="numpy_ref", **dense).run(path, resume=True)
    handler = capillary(backend="numpy_ref", **dense).run(
        path, resume="interpolated", progress="none"
    )
    assert isinstance(handler, PlaneStore)
    assert handler.annotations[INTERPOLATED_RESUMES_ANNOTATION] == [state.plane.z]
    # The run is complete and ends landed, so it resumes without the request.
    finished = resume(path)
    assert finished.n_saved == 4
    assert not _interpolated(finished)
    capillary(backend="numpy_ref", **dense).run(path, resume=True, progress="none")


def test_interpolation_enters_the_resume_key_at_schema_eleven():
    control = PIController(rtol=1e-6, atol=0.0)
    old = StepperRecord.from_settings(DOPRI5, control, limits=StepLimits())
    new = StepperRecord.from_settings(
        DOPRI5, control, limits=StepLimits(interpolate_planes=True)
    )
    assert _hashed_stepper(old, 10) == _hashed_stepper(new, 10)
    assert _hashed_stepper(old, 11) != _hashed_stepper(new, 11)
