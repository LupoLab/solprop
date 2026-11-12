# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The closed route union of the propagation problem (SEP 47 section 6.2)."""

import dataclasses

import pytest

from modal_provider_cases import GENERAL_Z_END, general_grid, general_guide
from solprop.backends.numpy_ref.backend import NumpyRefBackend
from solprop.boundaries import SpatialAbsorber
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BackendError, TranslationError
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes import PlanarGrid, SlabMode
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController
from solprop.translate.problem import _route
from solprop.translate.routes import (
    SPATIAL_KINDS,
    GeneralRoute,
    ModalRoute,
    SpatialKind,
    SpatialRoute,
    flux_of,
    route_name,
    spatial_kind,
    tdse_descriptor,
    tdse_of,
    transverse_of,
)

LENGTH = 1e-3


def gnlse():
    """A mode-averaged fibre: the modal route."""
    return Simulation.from_gnlse(
        0.11,
        LENGTH,
        [0.0, 0.0, -1.183e-26],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        shape="sech",
        noise=None,
    )


def spatial(kind):
    """A short free-space run on a grid of each spatial kind."""
    settings = SimulationSettings(
        controller=PIController(rtol=1e-7, atol=0.0),
        save_planes=[LENGTH],
        dz_max=LENGTH / 8,
        dz_init=LENGTH * 1e-3,
        backend="numpy_ref",
        spatial_absorber=SpatialAbsorber(real_space=False),
    )
    grid = EnvelopeGrid.from_limits(LENGTH, 800e-9, (700e-9, 900e-9), 200e-15)
    temporal = GaussianPulse(wavelength=800e-9, fwhm=50e-15, energy=1e-9)
    if kind == "planar":
        modes = [SlabMode(63.5e-6, "Ar", 1.0, order=p) for p in (1, 3)]
        return Simulation.from_planar(
            EnvelopeGrid.from_limits(LENGTH, 800e-9, (650e-9, 1000e-9), 600e-15),
            PlanarGrid(modes, 1e-4, 16),
            [],
            PlanarPulse(
                GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
                20e-6,
                waist_x=0.735 * 63.5e-6,
            ),
            settings=settings,
        )
    pulse = SpatialPulse(temporal, 40e-6)
    if kind == "radial":
        return Simulation.from_free(
            grid,
            RadialGrid(300e-6, 16),
            SpatialMedium(1.0),
            [],
            pulse,
            settings=settings,
            noise=None,
        )
    space = LineGrid(300e-6, 16) if kind == "line" else FreeGrid(300e-6, 16, 300e-6, 16)
    return Simulation.from_free(
        grid, space, SpatialMedium(1.0), [], pulse, settings=settings, noise=None
    )


def general():
    """A general modal guide: the general route."""
    grid = general_grid()
    return Simulation.from_waveguide(
        (),
        GENERAL_Z_END,
        provider=general_guide(grid=grid),
        grid=grid,
        noise=None,
        wavelength=835e-9,
        fwhm=20e-15,
        energy=1e-12,
        shape="gauss",
        save_planes=[GENERAL_Z_END / 10],
    )


@pytest.fixture(scope="module")
def modal_problem():
    return gnlse().translate()


@pytest.fixture(scope="module", params=SPATIAL_KINDS)
def spatial_problem(request):
    return request.param, spatial(request.param).translate()


def test_a_fibre_translates_to_the_modal_route(modal_problem):
    route = modal_problem.route
    assert isinstance(route, ModalRoute)
    assert route.tdse is None and tdse_of(route) is None
    assert tdse_descriptor(route) is None
    assert flux_of(route) is None and transverse_of(route) is None
    assert modal_problem.transform is route.transform
    assert route_name(route) == "modal"


def test_each_spatial_kind_translates_to_its_spatial_route(spatial_problem):
    kind, problem = spatial_problem
    route = problem.route
    assert isinstance(route, SpatialRoute)
    geometry = route.transform.geometry
    assert geometry is not None
    assert route.kind == kind == spatial_kind(geometry)
    assert flux_of(route) is route.flux
    assert transverse_of(route) is route.transverse
    assert tdse_of(route) is None
    assert route_name(route) == f"{kind} spatial"


def test_a_general_guide_translates_to_the_general_route():
    route = general().translate().route
    assert isinstance(route, GeneralRoute)
    assert route.transform.modal_tables is not None
    assert flux_of(route) is None and tdse_of(route) is None
    assert route_name(route) == "general modal"


def test_a_modal_route_refuses_another_route_s_transform(spatial_problem):
    _, problem = spatial_problem
    with pytest.raises(TranslationError, match="a modal route carries a modal"):
        ModalRoute(transform=problem.transform)


def test_a_modal_route_refuses_general_tables():
    transform = general().translate().transform
    with pytest.raises(TranslationError, match="with general modal tables"):
        ModalRoute(transform=transform)


def test_a_general_route_requires_its_tables(modal_problem):
    with pytest.raises(TranslationError, match=r"carries transform\.modal_tables"):
        GeneralRoute(transform=modal_problem.transform)


def test_a_spatial_route_requires_a_geometry_a_measure_and_its_own_kind(
    modal_problem, spatial_problem
):
    _, problem = spatial_problem
    route = problem.route
    assert isinstance(route, SpatialRoute)
    with pytest.raises(TranslationError, match="spatial geometry"):
        SpatialRoute(
            kind=route.kind, transform=modal_problem.transform, flux=route.flux
        )
    # Refused by the annotation under the test session's type checks, and by the
    # route itself without them.
    with pytest.raises((TranslationError, TypeError), match="flux"):
        SpatialRoute(kind=route.kind, transform=route.transform, flux=None)  # pyright: ignore[reportArgumentType]
    other: SpatialKind = next(one for one in SPATIAL_KINDS if one != route.kind)
    with pytest.raises(TranslationError, match=f"a {other!r} spatial route"):
        SpatialRoute(kind=other, transform=route.transform, flux=route.flux)


@dataclasses.dataclass(frozen=True)
class _Unknown:
    """A route variant no backend knows, as a later SEP might add one."""

    transform: object


def test_validation_and_each_backend_refuse_an_unknown_route_by_name(modal_problem):
    spoiled = dataclasses.replace(modal_problem)
    object.__setattr__(spoiled, "route", _Unknown(modal_problem.transform))
    run_request = gnlse()._run_request()
    with pytest.raises(BackendError, match="numpy_ref does not execute the"):
        NumpyRefBackend(spoiled, run_request)
    native = pytest.importorskip("solprop_native")
    with pytest.raises(BackendError, match="solprop-native does not execute the"):
        native.NativeBackend(spoiled, run_request)


def test_assembly_gives_each_leaf_to_the_route_that_carries_it(
    modal_problem, spatial_problem
):
    _, problem = spatial_problem
    flux = problem.route.flux
    with pytest.raises(TranslationError, match="requires its plane measure"):
        _route(problem.transform, tdse=None, flux=None, transverse=None)
    with pytest.raises(TranslationError, match="requires its plane measure"):
        _route(modal_problem.transform, tdse=None, flux=flux, transverse=None)
    design = problem.route.transverse
    assert design is not None
    with pytest.raises(TranslationError, match="belongs to a spatial route"):
        _route(modal_problem.transform, tdse=None, flux=None, transverse=design)
    rebuilt = _route(problem.transform, tdse=None, flux=flux, transverse=design)
    assert isinstance(rebuilt, SpatialRoute)
    assert rebuilt.transverse is design
