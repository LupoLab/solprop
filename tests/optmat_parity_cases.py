# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Translated optmat problems for SEP 3's backend qualification (Phase 2).

Every case is a run under :class:`~solprop.physdata.selection.MaterialSelection`,
built through the driver that a user would call -- ``Simulation.from_capillary`` and
``Simulation.from_gnlse`` -- and translated to the
:class:`~solprop.translate.problem.KernelProblem` the backends receive. Each takes
:data:`STEPS` fixed steps over a short guide: the qualification compares the
backends' arithmetic on the arrays optmat's models produce, not the physics of a
long run (``AGENTS.md``, qualification scope).

The cases cover the handoffs of Phases 1b, 1c and 1d (SEP 3, Part II):

- ``ar_field``, ``ar_envelope``: argon with optmat's rate, field and envelope;
- ``ar_hold``: argon launched above the rate model's maximum field, so the table
  holds its last value inside the run;
- ``n2_field``, ``n2_envelope``: nitrogen on a gradient with plasma and the 72-row
  Raman leaf, field and envelope;
- ``air_floor``: dry air on a pressure gradient from 1.5 bar to the vacuum floor, the
  126-row composition leaf whose amplitude is the summed density of three labels;
- ``air_vacuum``: the same leaf with every population falling linearly to exact
  vacuum, where N2's and O2's decay rates are zero. Independent density paths take
  the z table, which every CPU backend runs (``docs/dev/backends.md`` section 9);
- ``h2_constant``: hydrogen at constant density, its shifted lines exact;
- ``h2_reference``: hydrogen falling from 2 bar to exact vacuum under
  ``raman_shift="reference-density"``, with the finite vacuum decay rate of its Dicke
  cap;
- ``he_ar_kerr``: a helium/argon Kerr mixture with each population's carrier Kerr
  coefficient;
- ``silver_wall``: argon in a capillary walled by optmat's silver table;
- ``n2_vector``: nitrogen falling to exact vacuum on two polarisations of HE11, the
  72-row leaf's isotropic and rank-two rows contracted as a tensor, and its decay
  rates zero on the last plane; ``n2_vector_envelope`` the same on an envelope grid;
- ``gnlse_bundle``: the GNLSE with fused silica's bundle (a sampled row with a
  non-zero sample at zero delay);
- ``bundle_vector``: the silica tensor bundle on two polarisations, whose rank-two
  row also starts at a non-zero sample.

Every capillary under the selection is walled by optmat's recommended silica, its
default cladding there.
"""

import dataclasses
import functools
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.gnlse import kerr_susceptibility
from solprop.nonlinear.raman import bundle_response
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import Mixture, silver
from solprop.physdata.glasses import SilicaBundle
from solprop.physdata.selection import MaterialSelection
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import KernelProblem, assemble

SELECTION = MaterialSelection()
"""The curated selection every case runs under."""

SHIFTED = MaterialSelection(raman_shift="reference-density")
"""The selection that admits hydrogen on a gradient (SEP 3, Phase 1c)."""

LENGTH = 4e-4
"""Length of every capillary case, m: the Phase 1 runs' 0.4 mm."""

STEPS = 10
"""Fixed steps every case takes over its length."""

WAVELENGTH = 800e-9
"""Carrier wavelength of the gas cases, m."""

QUIET = (MaterialRangeWarning,)
"""Warnings the cases raise on purpose: bands and densities outside a model's range.

The gas fits warn beyond their qualified bands on these windows, and N2 and H2 at
these fills outside the Raman responses' validated densities; the adapters' own tests
assert those warnings.
"""

CASES = (
    "ar_field",
    "ar_envelope",
    "ar_hold",
    "n2_field",
    "n2_envelope",
    "air_floor",
    "air_vacuum",
    "h2_constant",
    "h2_reference",
    "he_ar_kerr",
    "silver_wall",
    "n2_vector",
    "n2_vector_envelope",
    "gnlse_bundle",
    "bundle_vector",
)
"""Every case, in the order the report lists them."""


@dataclass(frozen=True)
class Case:
    """One translated case and what its run needs besides the problem.

    Attributes
    ----------
    name : str
        The case's name in :data:`CASES`.
    simulation : Simulation or None
        The driver's run, or ``None`` for a case assembled directly.
    problem : KernelProblem
        The translated problem, on its fixed schedule.
    grid : FieldGrid or EnvelopeGrid
        The time grid.
    modes : tuple or None
        The mode set a backend is constructed with.
    """

    name: str
    simulation: Simulation | None
    problem: KernelProblem
    grid: Any
    modes: tuple[Any, ...] | None

    @property
    def n_rows(self) -> int:
        """Return the Raman rows the problem carries, summed over its leaves."""
        return sum(term.n_terms for term in self.problem.responses.raman_terms)


def _capillary(
    gas: Any, *, envelope: bool = False, length: float = LENGTH, **changes: Any
) -> Simulation:
    """Return a ten-step capillary under the selection, numpy_ref by default."""
    kind = EnvelopeGrid if envelope else FieldGrid
    window = changes.pop("trange", 200e-15)
    grid = kind.from_limits(length, WAVELENGTH, (300e-9, 3e-6), window)
    options: dict[str, Any] = {
        "pressure": 1.0,
        "lambda0": WAVELENGTH,
        "grid": grid,
        "absorber": absorber(grid, boundary_length=1.0),
        "fwhm": 30e-15,
        "energy": 100e-6,
        "plasma": False,
        "raman": False,
        "envelope": envelope,
        "n_r": 8,
        "schedule": FixedSchedule(np.full(STEPS, length / STEPS)),
        "n_save": 3,
        "backend": "numpy_ref",
        "materials": SELECTION,
    } | changes
    pressure = options.pop("pressure")
    return Simulation.from_capillary(50e-6, length, gas, pressure, **options)


def _gradient(fractions: dict[str, float], pressures, selection=SELECTION) -> Mixture:
    """Return a fill along the case's length under a selection."""
    return Mixture.fill(fractions, pressures, length=LENGTH, selection=selection)


def _to_vacuum(fill: Mixture) -> Mixture:
    """Return a constant fill's populations falling linearly to exact vacuum at L.

    Density paths, not a pressure path: an endpoint pair keeps a numerical floor at
    its outer end, and a linear pressure profile's density table rounded to a
    negative value there, which the translation refused until the table's vacuum end
    was made an exact zero (SEP 3, Q23). On these paths the decay rates take
    their vacuum values on the run's last plane: zero for N2's and O2's groups,
    H2's Dicke cap for its own.
    """
    return Mixture.of(
        *(
            dataclasses.replace(
                component,
                density=Linear(float(component.density(np.zeros(1))[0]), 0.0, LENGTH),
            )
            for component in fill.components
        ),
        temperature=fill.temperature,
    )


def _vector_launch(grid, energy: float) -> np.ndarray:
    """Return an elliptical launch on two polarisations, ``(n_modes, 1, n_omega)``.

    The construction of ``vector_raman_cases.vector_problem``: both helicities and a
    delayed oblique probe, so the polarisation varies in time and the rank-two rows
    act on more than one direction.
    """
    pulse = GaussianPulse(
        wavelength=grid.reference_wavelength, fwhm=30e-15, energy=energy
    )
    built = pulse.build(grid)
    delayed = built * np.exp(-1j * grid.omega * 20e-15)
    return np.concatenate(
        [
            built / math.sqrt(2) + 0.02 * delayed,
            1j * built / math.sqrt(2) - 0.03 * delayed,
        ],
        axis=0,
    )


def _from_simulation(name: str, simulation: Simulation) -> Case:
    """Translate a driver's run."""
    return Case(
        name, simulation, simulation.translate(), simulation.grid, simulation.modes
    )


def _build(name: str, changes: dict[str, Any]) -> Case:  # noqa: PLR0911 - one branch per case
    """Build one case, its capillary keywords overridden by `changes`."""

    def run(gas: Any, **options: Any) -> Case:
        return _from_simulation(name, _capillary(gas, **(options | changes)))

    if name in {"ar_field", "ar_envelope"}:
        return run("Ar", plasma=True, envelope=name == "ar_envelope")
    if name == "ar_hold":
        # 1 mJ in 30 fs puts the launch's peak field at 1.06e11 V/m, above argon's
        # maximum field of 8.6e10 V/m and inside the table's thirtyfold range. Over
        # the other cases' 40 um steps the field ionises past what a fixed step
        # holds, so this one takes its ten steps over 40 um.
        return run("Ar", plasma=True, energy=1e-3, length=LENGTH / 10)
    if name in {"n2_field", "n2_envelope"}:
        return run(
            _gradient({"N2": 1.0}, (1.5, 0.3)),
            pressure=None,
            plasma=True,
            raman=True,
            envelope=name == "n2_envelope",
        )
    if name == "air_floor":
        # A fixed wall index, as Phase 2 qualified it: optmat's silica was then
        # evaluated afresh at every sample of the frame tables, 17 s of a 21 s
        # translation. Since Q22 it is evaluated once per frequency array and costs
        # nothing here (SEP 3, Phase 2 follow-up); the other cases keep it.
        fill = Mixture.air((1.5, 0.0), length=LENGTH, selection=SELECTION)
        return run(fill, pressure=None, raman=True, cladding=1.45)
    if name == "air_vacuum":
        fill = _to_vacuum(Mixture.air(1.5, selection=SELECTION))
        return run(fill, pressure=None, raman=True)
    if name == "h2_constant":
        return run("H2", pressure=2.0, raman=True)
    if name == "h2_reference":
        fill = _to_vacuum(Mixture.fill({"H2": 1.0}, 2.0, selection=SHIFTED))
        return run(fill, pressure=None, raman=True, materials=SHIFTED)
    if name == "he_ar_kerr":
        return run({"He": 0.5, "Ar": 0.5}, pressure=2.0)
    if name == "silver_wall":
        return run("Ar", cladding=silver(model="recommended"))
    if name in {"n2_vector", "n2_vector_envelope"}:
        fill = _to_vacuum(Mixture.fill({"N2": 1.0}, 1.5, selection=SELECTION))
        mode = MarcatiliMode(50e-6, mixture=fill, cladding=1.45)
        modes = [mode, mode.rotated(math.pi / 2)]
        options = {
            "pressure": None,
            "raman": True,
            "modes": modes,
            "envelope": name == "n2_vector_envelope",
        } | changes
        simulation = _capillary(fill, **options)
        launch = _vector_launch(simulation.grid, 100e-6)
        return _from_simulation(name, simulation.with_input(launch))
    if name == "gnlse_bundle":
        simulation = Simulation.from_gnlse(
            0.11,
            1e-3,
            [0.0, 0.0, -1.183e-26],
            lambda0=835e-9,
            wavelength_limits=(500e-9, 2e-6),
            trange=1e-12,
            width=50e-15,
            peak_power=1e3,
            pulseshape="sech",
            shock=False,
            schedule=FixedSchedule(np.full(STEPS, 1e-3 / STEPS)),
            n_save=3,
            backend="numpy_ref",
            raman=True,
            materials=SELECTION,
            noise=None,
        )
        return _from_simulation(name, simulation)
    if name == "bundle_vector":
        return _bundle_vector()
    msg = f"no case {name!r}"
    raise KeyError(msg)


def _bundle_vector() -> Case:
    """Assemble the silica tensor bundle on two polarisations of a lossless mode.

    Assembled directly, as ``vector_raman_cases`` does: no driver takes the tensor
    bundle on a vector route today (SEP 3, Q8). The amplitudes are fused silica's:
    the bundle's Raman fraction of the susceptibility its ``n2`` gives at the
    carrier, so the delayed and instantaneous parts are in their physical ratio.
    """
    bundle = SilicaBundle.resolve(tensor=True)
    grid = FieldGrid.from_limits(LENGTH, 1030e-9, (500e-9, 3e-6), 400e-15)
    n2 = bundle.n2(1030e-9)
    chi3 = kerr_susceptibility(n2, bundle.raman_fraction)
    raman = bundle_response(
        bundle,
        amplitude=kerr_susceptibility(n2),
        fraction=bundle.raman_fraction,
        thg=True,
        tensor=True,
    )
    mode = MarcatiliMode(10e-6, cladding=1.45, loss=False)
    modes = [mode, mode.rotated(math.pi / 2)]
    launch = _vector_launch(grid, 2e-9)

    problem = assemble(
        grid,
        modes,
        [KerrResponse(chi3=chi3, thg=True), raman],
        launch,
        density=1.0,
        quadrature=transverse_quadrature(modes, n_r=8),
        controller=PIController(rtol=1e-9, atol=1e-15),
        save_planes=[LENGTH / 2, LENGTH],
        dz_max=LENGTH / STEPS,
        dz_init=LENGTH / STEPS,
        schedule=FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        stats_spec=StatsSpec(names=("energy", "peak_power_total", "peak_intensity")),
    )
    return Case("bundle_vector", None, problem, grid, tuple(modes))


@functools.cache
def build_case(name: str) -> Case:
    """Return one case, built once per process with its expected warnings quiet."""
    return variant(name)


def variant(name: str, **changes: Any) -> Case:
    """Return a capillary case with some of its keywords changed, built afresh.

    ``variant("ar_field", plasma=False)`` is the same run without its plasma, which
    the tests use to show that a rung can see the term it qualifies.
    """
    with warnings.catch_warnings():
        for category in QUIET:
            warnings.simplefilter("ignore", category)
        return _build(name, changes)
