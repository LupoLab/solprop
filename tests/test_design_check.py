# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The design check of a requested collar (SEP 40 section 4.4, Part II's witnesses).

The scaled suite case is the full case with its lengths scaled paraxially by 1/8, so
its returned energies are the full case's: the direct check reads 8.91e-7, 6.88e-9,
1.24e-10 and 2.74e-12 at f = 1, 1.5, 2 and 2.6 on one frequency, and first passes the
4e-8 budget near f = 1.3.
"""

import math
from typing import cast

import numpy as np
import pytest

from solprop.boundaries import SpatialAbsorber
from solprop.driver import Simulation
from solprop.exceptions import BoundaryWarning, DriverError, TranslationError
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid, RadialGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.translate.collar_check import (
    BUDGET,
    REFINED_BUDGET,
    _periodic_return,
    _radial_return,
)
from test_transverse_resolution import W0, suite


@pytest.mark.parametrize(
    ("fraction", "alpha0", "expected"),
    [(1.0, 698.5, 8.91e-7), (1.5, 436.0, 6.88e-9), (2.0, 318.4, 1.24e-10)],
)
def test_the_radial_check_reproduces_the_phase_0_estimates(
    fraction: float, alpha0: float, expected: float
) -> None:
    # Part II: the full suite case at 800 nm, the strengths designed on the requested
    # band; within 10 % (the phase-0 reference's quadrature floor is 5e-13).
    spacing, radius, waist = 4.6875e-6, 300e-6, 100e-6
    space = RadialGrid(
        radius, round(radius * (1 + fraction) / spacing), collar_fraction=fraction
    )
    rate = alpha0 * (1 - np.asarray(space.window))
    hankel = space.hankel
    forward = (
        hankel.forward_scale * (hankel.j1[:, None] / hankel.j1[None, :]) * hankel.matrix
    )
    launch = (forward @ np.exp(-((np.asarray(space.r) / waist) ** 2))).astype(complex)
    planes = np.linspace(0, 0.08, 9)[1:]
    difference, energy = _radial_return(
        space, rate, launch, 2 * math.pi / 800e-9, planes, 8 * radius
    )
    assert float(np.max(difference)) / energy == pytest.approx(expected, rel=0.1)


def test_the_linear_witnesses_read_their_truth() -> None:
    # Part II: a linear Gaussian, R 150 um, spacing 3 um, 2 cm, f 0.9, alpha0 6880,
    # waists 10 and 15 um: the direct check over the truth 5.25 and 6.70 (x 1e-4).
    radius, fraction, alpha0 = 150e-6, 0.9, 6880.0
    space = RadialGrid(
        radius, round(radius * (1 + fraction) / 3e-6), collar_fraction=fraction
    )
    rate = alpha0 * (1 - np.asarray(space.window))
    hankel = space.hankel
    forward = (
        hankel.forward_scale * (hankel.j1[:, None] / hankel.j1[None, :]) * hankel.matrix
    )
    for waist, truth in ((10e-6, 5.25e-4), (15e-6, 6.70e-4)):
        launch = (forward @ np.exp(-((np.asarray(space.r) / waist) ** 2))).astype(
            complex
        )
        difference, energy = _radial_return(
            space,
            rate,
            launch,
            2 * math.pi / 800e-9,
            np.linspace(0, 0.02, 65)[1:],
            8 * radius * (1 + fraction),
        )
        assert float(np.max(difference)) / energy == pytest.approx(truth, rel=0.05)


def test_the_profile_s_floor_refuses_before_the_check() -> None:
    # On the suite beam the launch-derived band has kappa_p w = 12.7 at f = 1, below
    # Planck's floor of 22.3 (section 4.3), whatever the check would say.
    with pytest.raises(TranslationError, match="floor") as caught:
        suite(1.0, SpatialAbsorber()).translate()
    assert "collar_fraction of at least 1.75" in str(caught.value)


def test_a_design_that_returns_too_much_is_refused_naming_a_fraction() -> None:
    # A declared band clears the floor at f = 1; the stronger collar returns 2.6e-6.
    with pytest.raises(TranslationError, match="design check returns") as caught:
        suite(1.0, SpatialAbsorber(kappa=2.2e5)).translate()
    message = str(caught.value)
    assert "collar_fraction=1.5" in message
    # On a slow beam a bare wall at the same size returns less, and the message says
    # so (section 4.4).
    assert "bare wall" in message


def test_a_wide_design_passes_and_records_its_check() -> None:
    design = suite(2.6, SpatialAbsorber()).translate().transverse
    assert design is not None
    check = design.check
    assert check["status"] == "passed"
    returned = check["returned"]
    assert isinstance(returned, float)
    assert returned < 1e-9
    assert cast("int", check["planes"]) >= 64
    assert len(check["nodes"]) >= 1  # pyright: ignore[reportArgumentType]


def test_a_design_near_the_budget_is_refined() -> None:
    # f = 1.5 with kappa = 2.2e5 returns about 2.4e-8, within a factor of two of the
    # budget: the estimate is refined once and decides at 3e-8.
    design = suite(1.5, SpatialAbsorber(kappa=2.2e5)).translate().transverse
    assert design is not None
    assert design.check["refined"] is True
    assert design.check["status"] == "passed"
    returned = design.check["returned"]
    assert isinstance(returned, float)
    assert 0.5 * BUDGET <= returned <= REFINED_BUDGET


def test_a_zero_launch_with_a_declared_band_leaves_the_check_pending() -> None:
    design = suite(2.6, SpatialAbsorber(kappa=1.2e5), energy=0.0).translate().transverse
    assert design is not None
    assert design.check["status"] == "pending"


def test_no_check_is_run_without_a_collar() -> None:
    design = suite(1.0, None).translate().transverse
    assert design is not None
    assert design.check == {}


def _pulse(waist: float = W0, energy: float = 1e-9) -> SpatialPulse:
    """Return the scaled suite's launch, or another waist."""
    return SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=energy), waist
    )


def test_a_rescaled_launch_reuses_the_verdict() -> None:
    # The check is linear and normalised by the launch's energy (section 5.2).
    parent = suite(2.6, SpatialAbsorber())
    design = parent.translate().transverse
    child = parent.with_input(_pulse(energy=3e-9))
    assert child.translate().transverse is design


def test_a_new_launch_gets_its_own_check() -> None:
    # A narrower beam spreads beyond the band the collar was derived from: it warns,
    # and its own check decides, on the shared grid.
    parent = suite(2.6, SpatialAbsorber())
    with pytest.warns(BoundaryWarning, match="spreads beyond the band"):
        try:
            child = parent.with_input(_pulse(waist=0.6 * W0))
        except DriverError as error:
            assert "translate a new simulation" in str(error)
            return
    design = child.translate().transverse
    assert design is not None
    assert design.check["status"] == "passed"
    assert design.check is not parent.translate().transverse.check  # pyright: ignore[reportOptionalMemberAccess]


def test_a_pending_check_runs_on_the_launch_it_is_given() -> None:
    # A coupled child is translated before its launch exists (section 5.2): it cannot
    # start pending, and with_input checks the launch it is given.
    child = suite(2.6, SpatialAbsorber(kappa=1.2e5), energy=0.0)
    with pytest.raises(DriverError, match="has not been checked"):
        child.run(progress="none")
    launched = child.with_input(_pulse())
    design = launched.translate().transverse
    assert design is not None
    assert design.check["status"] == "passed"


def _periodic(kind: str) -> Simulation:
    """Return a small periodic run with room for a collar, a 20 um beam on 10 um."""
    from capability_routes import (  # noqa: PLC0415
        SPATIAL_GRID,
        _argon,
        _spatial_settings,
    )

    medium, responses = _argon()
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 20e-6
    )
    if kind == "planar":
        from solprop.modes import PlanarGrid, SlabMode  # noqa: PLC0415
        from solprop.pulses import PlanarPulse  # noqa: PLC0415

        modes = [SlabMode(63.5e-6, "Ar", 1.0, order=p) for p in (1, 3)]
        return Simulation.from_planar(
            EnvelopeGrid.from_limits(2e-4, 800e-9, (650e-9, 1000e-9), 600e-15),
            PlanarGrid(modes, 1e-4, 40, collar_fraction=1.0),
            responses,
            PlanarPulse(
                GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
                20e-6,
                waist_x=0.735 * 63.5e-6,
            ),
            settings=_spatial_settings(),
            spatial_absorber=SpatialAbsorber(kappa=2.5e5),
        )
    space = (
        LineGrid(1e-4, 40, collar_fraction=1.0)
        if kind == "line"
        else FreeGrid(1e-4, 40, 1e-4, 40, collar_fraction=1.0)
    )
    return Simulation.from_free(
        SPATIAL_GRID,
        space,
        medium,
        responses,
        pulse,
        settings=_spatial_settings(),
        noise=None,
        spatial_absorber=SpatialAbsorber(kappa=2.5e5),
    )


@pytest.mark.parametrize("kind", ["cartesian", "line", "planar"])
def test_the_periodic_check_runs_on_every_periodic_route(kind: str) -> None:
    # The Strang march (section 4.4): a design either passes with its check recorded,
    # or is refused by the check by name; nothing else.
    try:
        design = _periodic(kind).translate().transverse
    except TranslationError as error:
        assert "design check" in str(error)
        return
    assert design is not None
    assert design.check["status"] in ("passed", "pending")
    assert cast("int", design.check["planes"]) >= 1


def test_the_periodic_march_absorbs_what_a_wrap_returns() -> None:
    # The linear witness on a Cartesian box (R 150 um, spacing 3 um, f 0.9, alpha0
    # 6880 1/m, 2 cm, waist 10 um): the collar returns 3.6e-4 of the launch where the
    # bare wrap returns far more; 64 Strang steps converge to 0.3 % (a phase-1
    # measurement, recorded here as a regression).
    radius, fraction = 150e-6, 0.9
    count = round(2 * radius * (1 + fraction) / 3e-6)
    space = FreeGrid(radius, count, radius, count, collar_fraction=fraction)
    x, y = np.meshgrid(space.x, space.y)
    launch = np.fft.fftn(np.exp(-(x**2 + y**2) / 10e-6**2)).reshape(-1)
    planes = np.linspace(0, 0.02, 65)[1:]
    half = 4 * radius * (1 + fraction)
    returns = []
    for alpha0 in (6880.0, 0.0):
        rate = alpha0 * (1 - np.asarray(space.window))
        difference, energy = _periodic_return(
            (np.asarray(space.x), np.asarray(space.y)),
            (radius, radius),
            rate,
            launch,
            2 * math.pi / 800e-9,
            planes,
            0.02 / 64,
            (half, half),
        )
        returns.append(float(np.max(difference)) / energy)
    collar, wrap = returns
    assert collar == pytest.approx(3.62e-4, rel=0.05)
    assert collar < 0.1 * wrap
