# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise power-row refusals and the unit-index statistics boundary."""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import (
    BackendError,
    GasFlowWarning,
    ModeError,
    OutputError,
    TranslationError,
)
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.linear import Frame
from solprop.modes import MarcatiliMode
from solprop.modes.protocol import ParameterisedMode
from solprop.profiles import Linear, PressureGradient
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec, _poynting_statistics, analytic_field
from solprop.stepper import PIController
from solprop.translate import expansion, linop
from solprop.translate.problem import assemble


@dataclass
class WeightedProblem:
    """Supply the linear row through its statically declared protocol."""

    linear: linop.LinearOperator


@pytest.fixture
def grid():
    return EnvelopeGrid.from_limits(0.01, 800e-9, (600e-9, 1.2e-6), 100e-15)


@pytest.fixture
def launch(grid):
    return GaussianPulse(wavelength=800e-9, fwhm=12e-15, energy=1e-9).build(grid)


@pytest.fixture
def cumulative():
    paths = {"density": Linear(1.0, 3.0, 1.0)}
    result = expansion.build_expansion(
        lambda values: np.array([1j * values["density"]]),
        paths,
        0.0,
        1.0,
        integrate=True,
    )
    assert result is not None
    return result, paths


@pytest.mark.parametrize("degrees", [np.array([-1]), np.array([1, 2])])
def test_endpoint_axes_are_validated(cumulative, degrees):
    integral, _ = cumulative
    with pytest.raises(TranslationError, match="endpoint_degrees"):
        replace(integral, endpoint_degrees=degrees)


def test_unresolved_endpoint_is_reported(cumulative):
    integral, paths = cumulative
    assert (
        expansion.append_endpoint_expansion(
            integral,
            lambda values: np.array([np.sin(1000 * values["density"])]),
            paths,
            tol=1e-14,
        )
        is None
    )


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_phi_refuses_reuse_with_a_different_index_convention(grid):
    mode = MarcatiliMode(
        125e-6, "Ar", PressureGradient.two_point(grid.length, 20.0, 1.0), cladding=1.45
    )
    physical = linop.build_phi(mode, grid, Frame.comoving(grid), z_max=grid.length)
    with pytest.raises(TranslationError, match="index override differs"):
        linop.build_phi(
            mode,
            grid,
            Frame.comoving(grid),
            z_max=grid.length,
            reuse=physical.phi_expansion,
            index=1.0,
        )


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_weight_expansion_falls_back_but_unresolved_phi_is_refused(grid, monkeypatch):
    mode = MarcatiliMode(
        125e-6, "Ar", PressureGradient.two_point(grid.length, 20.0, 1.0), cladding=1.45
    )
    linear = linop._dispersion_phi(mode, grid, Frame.comoving(grid), z_max=grid.length)
    monkeypatch.setattr(linop, "build_expansion", lambda *args, **kwargs: None)
    weighted = linop.with_power_weight(linear, (mode,), grid, None)
    assert weighted.power_weight_representation == "table"
    monkeypatch.setattr(linop, "parameter_paths", lambda *args: None)
    assert (
        linop.with_power_weight(linear, (mode,), grid, None).power_weight_representation
        == "table"
    )
    monkeypatch.undo()
    expected = np.real(mode.neff(grid.omega[grid.active_mask], 0.003))
    np.testing.assert_allclose(
        linop.power_weight_at(WeightedProblem(linear=weighted), 0.003)[
            0, 0, grid.active_mask
        ],
        expected,
        rtol=1e-14,
        atol=0,
    )  # 1e-14, fixed for all weight representations, including fallback.
    monkeypatch.setattr(
        linop, "append_endpoint_expansion", lambda *args, **kwargs: None
    )
    with pytest.raises(TranslationError, match="logarithmic power weight"):
        linop._power_phi(linear, (mode,), grid, None)


def test_weight_sampler_refuses_an_invalid_physical_index(grid, monkeypatch):
    mode = MarcatiliMode(125e-6, "Ar", 1.0, cladding=1.45)
    monkeypatch.setattr(
        MarcatiliMode, "neff", lambda self, omega, z: np.zeros_like(omega)
    )
    # One helper, one exception type: the sampler defers to solprop.linear, so an
    # invalid weight raises ModeError whichever of the two is built first.
    with pytest.raises(ModeError, match=r"positive and finite.*prefactor_index"):
        linop._weight_sampler((mode,), grid, 1, None)(0.0)


def test_spatial_contract_refuses_a_modal_weight(grid):
    space = RadialGrid(6e-4, 8)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 60e-6
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
    )
    problem = Simulation.from_radial(
        grid, space, SpatialMedium(), [], pulse, settings=settings
    ).translate()
    simulation = Simulation.from_radial(
        grid, space, SpatialMedium(), [], pulse, settings=settings
    )
    simulation.translate()
    updated = simulation.with_input(simulation.initial)
    np.testing.assert_array_equal(
        updated.translate().initial.field0, problem.initial.field0
    )
    with pytest.raises(TranslationError, match="absent for spatial geometry"):
        linop.power_weight_at(problem, 0.0)
    with pytest.raises(TranslationError, match="absent for spatial geometry"):
        linop.power_weight_factors(problem, np.array([0.0]))
    with pytest.raises(TranslationError, match="absent for spatial geometry"):
        replace(
            problem,
            linear=replace(problem.linear, power_weight_representation="constant"),
        ).validate()


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_retranslating_the_same_modal_input_is_exactly_identical():
    # `with_input` divides the replacement launch by the weight row, so `assemble`
    # must divide by that same row: a second, direct evaluation of the mode's index
    # moves the launch by 2.4e-15 on this gradient, where replacing the input with
    # itself must be an exact no-op (a structural contract). The spatial test above
    # asserts the same identity where there is no modal weight at all.
    # A strong gradient over a millimetre, kept at its declared ends: the flow is far
    # from laminar, which the fill says.
    with pytest.warns(GasFlowWarning, match="Reynolds"):
        simulation = Simulation.from_capillary(
            125e-6,
            1e-3,
            "Ar",
            (20.0, 1.0),
            vacuum_end="floor",
            lambda0=800e-9,
            wavelength_limits=(400e-9, 1200e-9),
            trange=400e-15,
            fwhm=10e-15,
            energy=1e-9,
            n_r=8,
            n_save=2,
            thg=False,
            loss=False,
            cladding=1.45,
        )
    problem = simulation.translate()
    assert problem.linear.power_weight_representation != "constant"
    updated = simulation.with_input(simulation.initial)
    np.testing.assert_array_equal(
        updated.translate().initial.field0, problem.initial.field0
    )


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_taper_with_a_floored_vacuum_end_translates():
    # The logarithmic weight's endpoint block is resolved over the whole (core
    # radius, density) box, with a tolerance measured on that box: at the
    # floor-density corner each element is far smaller and its relative round-off far
    # larger than anywhere along the path, so a tolerance taken from the path alone
    # never resolves the radius axis and the translation is refused. A uniform bore
    # has no radius axis, and a higher end pressure no such corner.
    length = 0.5
    simulation = Simulation.from_capillary(
        Linear(150e-6, 100e-6, length),
        length,
        "Ar",
        (2.0, 0.0),
        vacuum_end="floor",
        lambda0=800e-9,
        wavelength_limits=(200e-9, 2e-6),
        trange=100e-15,
        fwhm=10e-15,
        energy=1e-6,
        n_r=8,
        n_save=2,
        rtol=1e-6,
        plasma=False,
    )
    problem = simulation.translate()
    linear = problem.linear
    phi = linear.phi_expansion
    assert phi is not None and phi.endpoint_degrees is not None
    assert phi.parameters == ("core_radius", "density")
    n_pol = linear.value_shape[2]
    (mode,) = simulation.modes
    assert isinstance(mode, ParameterisedMode)
    sample = linop._log_weight_sampler((mode,), simulation.grid, n_pol, None)

    def value_at(values):
        return linop._log_weight_sampler(
            [mode.at_parameters(values)], simulation.grid, n_pol, None
        )(0.0)

    # The block is resolved to `_power_phi`'s tolerance relative to each element's
    # scale, so the largest element bounds its absolute error.
    floor = expansion.parameter_roundoff_floor(value_at, phi.parameters, phi.ranges)
    entrance = sample(0.0)
    bound = 8 * floor * float(np.abs(entrance).max())
    # The endpoint block alone is -ln(Re n_eff) / 2 relative to the entrance.
    n_endpoint = int(np.prod(phi.endpoint_degrees + 1))
    exit_side = length - np.logspace(-9, -3, 7)
    for z in np.concatenate((np.linspace(0.0, length, 11), exit_side)):
        expected = sample(float(z)) - entrance
        actual = np.tensordot(
            phi.weights(float(z))[-n_endpoint:], phi.terms[-n_endpoint:], axes=1
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=bound)


@pytest.mark.parametrize("backend_class", [NumpyRefBackend, NumbaCpuBackend])
@pytest.mark.parametrize("with_modes", [False, True])
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_unit_index_intensity_retains_the_vacuum_limit(
    grid, launch, backend_class, with_modes
):
    mode = MarcatiliMode(125e-6, "Ar", 1.0, cladding=1.45, loss=False)
    problem = assemble(
        grid,
        mode,
        [],
        launch,
        prefactor_index=1.0,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
    )
    backend = backend_class(problem, grid=grid, modes=(mode,) if with_modes else None)
    spec = StatsSpec(names=("peak_intensity",))
    try:
        backend.set_initial(problem.initial.field0, None)
        if not with_modes:
            with pytest.raises(BackendError, match="mode set"):
                backend.stats(spec)
        else:
            # Unit weight is the exact monochromatic vacuum limit at every bin.
            axis = backend._on_axis_synthesis((mode,))
            expected = _poynting_statistics(
                spec, grid, launch, np.ones(launch.shape), axis
            )
            np.testing.assert_allclose(
                backend.stats(spec)["peak_intensity"][0],
                expected["peak_intensity"],
                rtol=64 * np.finfo(float).eps,
                atol=0,
            )
    finally:
        backend.close()


def test_poynting_windows_and_missing_dependencies(grid, launch):
    weight = np.linspace(1.0, 1.1, grid.n_omega)[None, None]
    with pytest.raises(OutputError, match="at least one energy window"):
        _poynting_statistics(StatsSpec(names=("energy_window",)), grid, launch, weight)
    with pytest.raises(OutputError, match="on-axis synthesis"):
        _poynting_statistics(StatsSpec(names=("peak_intensity",)), grid, launch, weight)
    windows = np.array([[0.0, np.inf]])
    spec = StatsSpec(names=("peak_power_window",), energy_windows=windows)
    values = _poynting_statistics(spec, grid, launch, weight)
    expected = np.real(
        analytic_field(grid, launch) * np.conj(analytic_field(grid, launch * weight))
    )
    np.testing.assert_allclose(
        values["peak_power_window"][0],
        np.max(expected, axis=-1).ravel(),
        rtol=32 * np.finfo(float).eps,
        atol=0,
    )  # Identical full-band Hermitian product.


@pytest.mark.medium
def test_power_weight_fallback_with_production_imports(tmp_path):
    # This must stay, and it cannot run in-process. The type hook rewrites
    # nested-function annotations, so the two fall-through arcs into the z-table
    # fallback's local sampler -- "paths is None" and "build_expansion returned None"
    # -- land on the rewritten `def` rather than on the source line, and the
    # monkeypatched test above reaches the same code without recording them. Deleting
    # this test loses exactly those two branch arcs from `with_power_weight`. It is
    # `medium` rather than fast because the child process costs 1.3 s, over the
    # fast tier's one-second budget (docs/dev/testing.md section 2); the default and
    # coverage lanes both run the medium tier, so nothing is lost.
    # Merge only coverage actually collected by the child.
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from coverage import Coverage, CoverageData  # noqa: PLC0415

    coverage = Coverage.current()
    data_path = tmp_path / ".coverage.production"
    script = tmp_path / "production_weight.py"
    script.write_text(
        """import importlib.util
import sys
import warnings
from unittest.mock import patch

if importlib.util.find_spec("solprop_native") is not None:
    importlib.import_module("solprop_native")
import numpy as np
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import EnvelopeGrid
from solprop.linear import Frame
from solprop.modes import MarcatiliMode
from solprop.modes.protocol import ParameterisedMode
from solprop.profiles import PressureGradient
from solprop.translate import linop

if sys.argv[2] == "coverage":
    from coverage import Coverage
    measured = Coverage(data_file=sys.argv[1], branch=True,
                        source=["solprop.translate.linop"])
    measured.start()

with warnings.catch_warnings():
    warnings.simplefilter("ignore", MaterialRangeWarning)
    grid = EnvelopeGrid.from_limits(0.01, 800e-9, (600e-9, 1.2e-6), 100e-15)
    mode = MarcatiliMode(125e-6, "Ar",
        PressureGradient.two_point(grid.length, 20.0, 1.0), cladding=1.45)
    linear = linop._dispersion_phi(mode, grid, Frame.comoving(grid),
                                   z_max=grid.length)
    assert linear.phi_representation == "expansion"
    for name in ("build_expansion", "parameter_paths"):
        with patch.object(linop, name, return_value=None):
            weighted = linop.with_power_weight(linear, (mode,), grid, None)
        assert weighted.power_weight_representation == "table"
        for z in np.random.default_rng(42).uniform(0.0, grid.length, 20):
            actual = weighted.power_weight(float(z))[0, 0, grid.active_mask]
            expected = np.real(mode.neff(grid.omega[grid.active_mask], float(z)))
            # The table's build tolerance, the default 1e-12.
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
if sys.argv[2] == "coverage":
    measured.stop()
    measured.save()
"""
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(data_path),
            "coverage" if coverage is not None else "plain",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if coverage is not None:
        child = CoverageData(str(data_path))
        child.read()
        coverage.get_data().update(child)
