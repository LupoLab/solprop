# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Extract advanced guide workloads without executing their source galleries.

This module is imported only after a worker configures scientific libraries.
Material choices and model approximations are those of the registry's sources;
no new physical fit or validity claim is introduced by these benchmarks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex
from numpy.typing import ArrayLike

from benchmarks.records import SuiteError
from solprop.boundaries import absorber
from solprop.constants import C
from solprop.coupling import CouplingModel
from solprop.driver import Simulation
from solprop.grid import FieldGrid
from solprop.modes import (
    CoherentWall,
    GradedIndexProvider,
    IndexMaterial,
    MarcatiliMode,
    MaterialKerr,
    MaterialRegion,
    ModalGuide,
    PowerLawPermittivity,
    StepIndexMode,
)
from solprop.modes.material import JsonValue
from solprop.physdata import fused_silica
from solprop.profiles import PiecewiseLinear
from solprop.pulses import GaussianPulse, SpatialPulse

__all__ = ["ScaledPermittivity", "advanced", "coupled", "grin", "moving", "step_index"]


@dataclass(frozen=True)
class ScaledPermittivity:
    """Retain the nonlinear GRIN example's scaled cladding law.

    Attributes
    ----------
    law : IndexMaterial
        Original absolute-index law, here Malitson (1965),
        doi:10.1364/JOSA.55.001205, supplied through optmat.
    factor : float
        Dimensionless multiplier of relative permittivity.
    """

    law: IndexMaterial
    factor: float

    def permittivity(
        self, omega: ArrayLike
    ) -> Complex[np.ndarray, " n_omega_permittivity"]:
        """Evaluate dimensionless permittivity at angular frequencies in rad/s."""
        return self.factor * self.law.permittivity(omega)

    @property
    def omega_domain(self) -> tuple[float, float]:
        """Return the original law's validity interval in rad/s."""
        return self.law.omega_domain

    @property
    def identity(self) -> Mapping[str, JsonValue]:
        """Identify the original law and its declared multiplier."""
        return {"law": dict(self.law.identity), "factor": self.factor}


def coupled(name: str) -> Simulation:
    """Preserve the wall tutorials' ten-step guides, grids and profile corners."""
    if name not in ("scalar_wall", "vector_wall", "taper", "taper_wall"):
        raise SuiteError(f"coupled guide builder not implemented for {name}")
    length = 2e-4
    grid = FieldGrid.from_limits(length, 800e-9, (400e-9, 1.6e-6), 300e-15)
    radius = (
        PiecewiseLinear.from_stages(
            (0.0, 6e-5, 1.4e-4, length), [50e-6, 50e-6, 40e-6, 40e-6]
        )
        if name in ("taper", "taper_wall")
        else 50e-6
    )
    modes = (
        [
            MarcatiliMode(50e-6, "He", 1.0, n=order, m=radial)
            for order in (1, -1)
            for radial in (1, 2)
        ]
        if name == "vector_wall"
        else 3
    )
    return Simulation.from_capillary(
        radius,
        length,
        "He",
        1.0,
        lambda0=800e-9,
        grid=grid,
        absorber=absorber(grid, boundary_length=1e3),
        fwhm=30e-15,
        energy=1e-9,
        modes=modes,
        wall=None
        if name == "taper"
        else CoherentWall(impedance="te_tm" if name == "vector_wall" else "mean"),
        taper_coupling=True,
        dz_max=length / 10,
        dz_init=length / 10,
        plasma=False,
        raman=False,
        backend="numpy_ref",
        seed=20261027,
    )


def step_index() -> Simulation:
    """Extract SEP 25's nonlinear HE11 parity pair and its declared test medium."""
    grid = FieldGrid.from_limits(1e-3, 835e-9, (700e-9, 1e-6), 400e-15)
    modes = tuple(
        StepIndexMode(
            1.5e-6,
            core=1.45,
            cladding=1.0,
            profile_wavelength=835e-9,
            kind="HE",
            n=1,
            m=1,
            parity=parity,
        )
        for parity in ("even", "odd")
    )
    core, cladding = IndexMaterial(1.45), IndexMaterial(1.0)
    # The source fixture declares this silica-like susceptibility to exercise
    # arithmetic; it is not presented as a measured material coefficient.
    regions = (
        MaterialRegion(
            "core", 0.0, modes[0].core_radius, core, (MaterialKerr(chi3=2.2e-22),)
        ),
        MaterialRegion("cladding", modes[0].core_radius, None, cladding),
    )
    provider = ModalGuide.from_step_index(
        modes, core=core, cladding=cladding, regions=regions
    ).field_provider
    active = grid.omega[grid.active_mask]
    guide = ModalGuide.from_provider(
        provider,
        np.linspace(0.99 * float(active[0]), 1.01 * float(active[-1]), 33),
        nodes=provider.node_rule(0.99 * float(active[0]), n_r=32),
        degenerate_blocks=((0, 1),),
    )
    return Simulation.from_waveguide(
        (),
        1e-3,
        provider=guide,
        grid=grid,
        lambda0=835e-9,
        fwhm=50e-15,
        energy=1e-9,
        backend="numpy_ref",
        seed=20261027,
    )


def grin(*, nonlinear: bool, length: float = 0.1) -> Simulation:
    """Preserve the linear lesson or the four-channel dispersive GRIN example.

    The nonlinear source describes the GPI model and its chosen Kerr coefficient
    alongside Krupa et al., doi:10.1103/PhysRevLett.116.183901. This extraction
    retains its four-channel truncation, not a mode-count convergence claim.
    `length` shortens the nonlinear guide, which is uniform along z.
    """
    radius, contrast = (50e-6, 0.0095) if nonlinear else (25e-6, 0.005)
    wavelength = 1.064e-6 if nonlinear else 1.03e-6
    material = (
        IndexMaterial(fused_silica(model="malitson_1965/sellmeier"))
        if nonlinear
        else IndexMaterial(1.45)
    )
    exterior = (
        ScaledPermittivity(material, 1 - 2 * contrast)
        if nonlinear
        else IndexMaterial(1.45 * np.sqrt(1 - 2 * contrast))
    )
    provider = GradedIndexProvider(
        permittivity=PowerLawPermittivity(material, contrast),
        exterior=exterior,
        radius=radius,
        family="LP",
        count=4 if nonlinear else 2,
        elements=64,
        extent=1.6 if nonlinear else 2.0,
        core_responses=(MaterialKerr(n2=2.7e-20, at=wavelength),) if nonlinear else (),
    )
    grid = FieldGrid.from_limits(
        length if nonlinear else 1e-3,
        wavelength,
        (450e-9, 1.8e-6) if nonlinear else (1e-6, 1.06e-6),
        1.2e-12 if nonlinear else 100e-15,
    )
    active = grid.omega[grid.active_mask]
    low, high = 0.99 * float(active[0]), 1.01 * float(active[-1])
    nodes = provider.node_rule(low, n_r=2, **({"n_theta": 2} if nonlinear else {}))
    guide = ModalGuide.from_provider(
        provider,
        np.geomspace(low, high, 97) if nonlinear else np.linspace(low, high, 13),
        nodes=nodes,
    )
    centre = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    if nonlinear:
        beta = (
            np.array(
                [
                    mode.neff(grid.omega[centre : centre + 1])[0].real
                    for mode in guide.mode_collection.modes
                ]
            )
            * float(grid.omega[centre])
            / C
        )
        period = 2 * np.pi / float(np.mean(-np.diff(beta)))
        # The incident beam's energy is over its whole plane; the channels launch
        # the share they capture, coupled under the vector paraxial model.
        beam = SpatialPulse(
            GaussianPulse(
                wavelength=wavelength,
                fwhm=300e-15,
                energy=5e5 * 300e-15 * np.sqrt(np.pi / (4 * np.log(2))),
            ),
            12e-6,
        )
        return Simulation.from_waveguide(
            (),
            length,
            provider=guide,
            grid=grid,
            incident=beam,
            coupling=CouplingModel("vector_paraxial", model_error=0.2),
            lambda0=wavelength,
            noise="input",
            seed=41,
            dz_max=period / 10,
            backend="numpy_ref",
        )
    q = float(grid.omega[centre]) / C * radius * 1.45
    beta = np.sqrt(q * q - 2 * q * np.sqrt(2 * contrast) * np.array([1, 3])) / radius
    length = float(2 * np.pi / (beta[0] - beta[1]))
    simulation = Simulation.from_waveguide(
        (),
        length,
        provider=guide,
        grid=grid,
        lambda0=wavelength,
        fwhm=40e-15,
        energy=1e-15,
        dz_max=length / 10,
        dz_init=length / 10,
        backend="numpy_ref",
        seed=20261027,
    )
    launch = np.zeros((2, 1, grid.n_omega), dtype=complex)
    launch[:, 0, centre] = 1.0
    # with_input shares an already translated problem. Keep this public
    # constructor untranslated so the worker can first select its integrator,
    # controller and observation planes, while reusing the resolved guide.
    return Simulation(
        grid,
        simulation.modes,
        simulation.responses,
        launch,
        settings=simulation.settings,
        configuration=simulation.configuration,
        general=simulation.general,
        guide=guide,
        materials=simulation.materials,
    )


def moving() -> Simulation:
    """Keep the moving-solid example's nonlinear taper, waist and two channels."""
    length, wavelength, contrast = 15e-3, 1.03e-6, 0.0095
    path = PiecewiseLinear.from_stages((0.0, 5e-3, length), [25e-6, 15e-6, 15e-6])
    grid = FieldGrid.from_limits(length, wavelength, (0.93e-6, 1.15e-6), 2e-12)
    provider = GradedIndexProvider(
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), contrast),
        exterior=IndexMaterial(1.45 * np.sqrt(1 - 2 * contrast)),
        radius=path,
        family="LP",
        count=2,
        elements=48,
        extent=2.0,
        core_responses=(MaterialKerr(n2=2.7e-20, at=wavelength),),
    )
    active = grid.omega[grid.active_mask]
    frequencies = np.geomspace(0.99 * float(active[0]), 1.01 * float(active[-1]), 25)
    guide = ModalGuide.from_provider(
        provider,
        frequencies,
        nodes=provider.node_rule(float(frequencies[0]), 0.0, n_r=2, n_theta=2),
    )
    return Simulation.from_waveguide(
        (),
        length,
        provider=guide,
        grid=grid,
        kerr_form="analytic",
        lambda0=wavelength,
        fwhm=200e-15,
        peak_power=4e5,
        dz_max=1.2e-4,
        backend="numpy_ref",
        seed=20261027,
    )


def advanced(name: str, length: float) -> Simulation:
    """Select a declared guide before applying the suite's measurement settings.

    `length` is the registered guide length; only the nonlinear GRIN guide takes it.
    """
    if name in ("scalar_wall", "vector_wall", "taper", "taper_wall"):
        return coupled(name)
    if name == "step_index_pair":
        return step_index()
    if name in ("grin_linear", "grin_nonlinear"):
        return grin(nonlinear=name == "grin_nonlinear", length=length)
    if name == "moving_solid_guide":
        return moving()
    raise SuiteError(f"advanced guide builder not implemented for {name}")
