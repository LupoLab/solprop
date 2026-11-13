# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The spatial geometries' populations, flux, synthetic leaves and collar: their rows.

**What is compared.** Native against ``numpy_ref`` on six groups of cases, each from the
module that held it before:

- two envelope plasma populations (``host_cases_envelope_plasma.py``) on the modal,
  radial and Cartesian geometries, one and two polarisations, in partial node chunks:
  ``steps`` compares the field after the case's two fixed Runge-Kutta steps, and
  ``launch`` and ``steps`` every population statistic, at the launch and at the end;
  and the modal case along a radius taper whose densities fall to zero (from
  ``test_envelope_plasma_geometries.py``);
- the two-channel mixture (``mixture_cases.py``) on the radial and Cartesian routes,
  with and without the Kerr term's harmonic, one and two polarisations: the field after
  eight fixed steps and every mixture statistic, at the launch and at the end (from
  ``test_spatial_mixture_channels.py``);
- the two channels' final fractions and peak rates at a radial launch of 3.6e10 V/m
  (``native_mixture_launch_probe.py``), on the real and the analytic column: ``launch``
  (from ``test_native_mixture_fractions.py``, which held native to the passes on the
  reference's node fields; the reference's launch statistics are those passes, to the
  bit, measured);
- the axial-flux statistics of one chirped state at one plane, on four lossy routes and
  a radial argon gradient read past the entrance: ``launch``, a row per group of
  statistics that share a rounding budget (from ``test_flux_parity.py``);
- the retired kernels' synthetic separable leaves (loss and gain, an exactly zero
  ``Re q``, evanescent columns, an inactive bin, a wall with gain and a negative real
  part, low words of 1e-17, a safety ramp) on small Cartesian and planar problems: one
  step on the low-storage march and on the Runge-Kutta march (from
  ``test_native_synthetic_operator.py``);
- a requested collar on every route, linear, a converging beam reaching it, on a
  matched fixed schedule of either march: the field at the run's end (from
  ``test_native_edge_collar.py``, whose monitor record stays there).

**The bounds are the old tests', fixed before native runs.** The populations take the
geometry ladder's ``128 eps`` per sample per stage, ``128 n_to 7 2 eps``; the mixture's
channels ``(64 n_to 2 + 128 n_nodes) 7 8 eps``; the fractions ``64 n_to eps``; the flux
statistics their own budget per statistic (``test_flux_parity._budget``); the
synthetic step ``low_storage_cases.parity_bound`` over the step's evaluations, grown by
the leaf's gain and the Kerr term's rate (``test_native_synthetic_operator``'s
``step_limit``); the collar run an ad hoc ``1e-10`` of the field's norm.

**Where the old form could not be kept exactly.** The populations' fields and
statistics were held to ``numpy.testing.assert_allclose(rtol=b, atol=b max|ref|)``,
which admits ``b (max|ref| + |ref|)`` at an element: the rows hold the stricter
peak-normalised ``b max|ref|`` alone. The mixture's ``species_ionisation`` was held to
``rtol=b, atol=b``; the row holds it absolutely to ``b``, the stricter. Every other
comparison is the old one: normwise, peak-normalised, elementwise relative or absolute.

**The guard.** Each row's term is switched off on the translated problem: the envelope
plasma (``optmat_rows.without_plasma``), the field channels' ionisation
(:func:`without_ionisation`; zeroing their densities moves no launch statistic),
the medium's loss or the plane measure's gradient or walls (:func:`lossless`,
:func:`at_the_entrance`, :func:`without_walls`), the synthetic leaf's safety ramp
(:func:`without_the_ramp`) or the collar (:func:`without_the_collar`). Each guarded
rung's reference must then move by more than ``parity_ladder.DETECTION`` (a thousand)
of that rung's bounds. The flux widths are moved by none of these, and are declared
``vacuous`` with what was measured.

**How to read a row.** Every row is a :func:`geometry_row` call, for example::

    geometry_row(
        name="mixture channels radial harmonic two polarisations",
        route="radial",
        stepper="rk",
        response="kerr, plasma",
        features="two channels, third harmonic, two polarisations",
        why="Two plasma channels on the radial route are the reference's ...",
        build=partial(mixture_channels, "radial", True, 2),
        switch_off=without_ionisation,
        rungs={"launch": channel_budget, "steps": channel_budget},
        every_statistic=held_as_the_mixture,
    )

``name`` makes the test id (``geometry mixture channels radial harmonic two
polarisations``); ``route``, ``stepper``, ``response`` and ``features`` are the axes the
row covers. ``build`` makes the problem: here the two-channel mixture on eight radial
nodes, its harmonic on, two polarisations, eight fixed steps. ``rungs`` names what is
compared, each with its bound: the statistics at the launch and the field after the
steps, normwise. Every statistic is then held at the launch and at the end by
:func:`held_as_the_mixture`. ``switch_off`` removes the row's term for the guard.
``why`` is the claim a failure quotes.
"""

import dataclasses
import functools
import math
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from backends import edge_collar_cases as collar
from backends import flux_cases
from backends import synthetic_leaf_cases as leaves
from backends.host_cases_envelope_plasma import (
    build_envelope_mixture,
    geometry_case,
    geometry_options,
)
from backends.host_cases_mixture_threads import NAMES as MIXTURE_STATISTICS
from backends.native_mixture_launch_probe import launch_case
from backends.optmat_rows import without_plasma
from mixture_cases import mixture_translation
from parity_ladder import Built, Detection, Row, Statistic
from solprop.driver import Simulation
from solprop.grid import FreeGrid, RadialGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear.responses import KerrResponse
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, Multistep, RungeKutta
from solprop.translate.assembly import assemble
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import flux_of
from solprop.translate.stage import scalars

EPS = float(np.finfo(float).eps)
"""The machine epsilon."""

POLARISATIONS = {1: "one polarisation", 2: "two polarisations"}
"""How a row's id names its polarisations."""

# --- the envelope populations ---------------------------------------------------------

POPULATION_STATISTICS = (
    "electron_density",
    "peak_ionisation_rate",
    "species_ionisation",
    "species_electron_density",
    "species_peak_ionisation_rate",
    "species_density",
    "species_ionisation_energy",
)
"""Every population statistic the envelope cases read."""

GEOMETRIES = ("modal", "radial", "free")
"""The host geometries the envelope populations run on."""


@functools.cache
def envelope_mixture() -> tuple[Any, tuple[Any, ...]]:
    """Return the two-component mixture and its two envelope plasma responses, once."""
    return build_envelope_mixture()


def population_case(geometry: str, n_pol: int) -> Built:
    """Return the two populations on `geometry` over two fixed steps, every statistic.

    The statistics are observed at the launch, the end and every step's end
    (``period=1``), the four that have a periodic row among them; the spatial
    geometries run in three-node chunks that leave a partial one.
    """
    problem, run_request, grid = geometry_case(envelope_mixture(), geometry, n_pol)
    spec = StatsSpec(
        names=POPULATION_STATISTICS, periodic=POPULATION_STATISTICS[:4], period=1
    )
    return Built(
        problem,
        run_request.for_run(stats=spec),
        grid,
        options=geometry_options(geometry),
    )


def population_taper() -> Built:
    """Return the modal case along a radius taper whose populations' densities fall.

    The capillary narrows from 30 to 27 um and every component's density falls
    linearly to zero over the case, so the stage reads the taper's scale and each
    population's density from their profiles.
    """
    mixture, responses = envelope_mixture()
    original, original_request, grid = geometry_case(envelope_mixture(), "modal", 2)
    mixture = dataclasses.replace(
        mixture,
        components=tuple(
            dataclasses.replace(
                component,
                density=Linear(
                    float(component.density(np.array([0.0]))[0]), 0.0, grid.length
                ),
            )
            for component in mixture.components
        ),
    )
    mode = MarcatiliMode(
        Linear(30e-6, 27e-6, grid.length),
        mixture=mixture,
        cladding=1.45,
        loss=False,
    )
    problem, run_request = assemble(
        grid,
        mode,
        responses,
        original.initial.field0,
        controller=original_request.stepper.controller,
        save_planes=[grid.length],
        dz_max=grid.length / 2,
        dz_init=grid.length / 2,
        schedule=original_request.stepper.schedule,
    )
    return Built(problem, run_request, grid)


def the_taper_reaches_each_population(built: Built) -> bool:
    """Return whether the taper's scale moves and every density falls from live to none.

    The old test's own check of the case, on the stage scalars at either end.
    """
    start = scalars(built.problem, 0.0, 0.0)
    end = scalars(built.problem, built.grid.length, built.grid.length)
    return (
        start.taper_scale != end.taper_scale
        and all(density > 0 for density in start.plasma_densities)
        and bool(np.all(np.asarray(end.plasma_densities) == 0))
    )


def population_budget(built: Built) -> float:
    """Bound the populations' runs: ``128 eps`` per sample per stage, ``128 n_to 7 2``.

    The geometry ladder's budget: the polynomial rate's integrals converge to rounding,
    and the transforms, the two channels' scans and the projections contribute less
    than ``128 eps`` per sample per Runge-Kutta stage, over two steps of seven.
    """
    return 128 * built.grid.n_to * 7 * 2 * EPS


THE_LAUNCH_HOLDS_NO_DENSITY = (
    "Zeroing every population's density on the translated problem moves the "
    "reference's statistics at the launch by exactly 0 bounds (measured): the launch "
    "is the launch field's, which no response has touched. Kept as the old test's "
    "launch-plane comparison; the steps rung carries the guard (1.6e8 bounds)."
)
"""Why an envelope population row keeps its ``launch`` rung unguarded."""


def held_as_the_field(built: Built, name: str, bound: float) -> Statistic:
    """Hold a population statistic peak-normalised to its rung's bound.

    The old test's ``assert_allclose(rtol=b, atol=b max|ref|)`` without its relative
    part, which the row does not keep: the stricter form.
    """
    return Statistic(bound)


# --- the mixture's channels on the spatial routes -------------------------------------


def mixture_channels(geometry: str, thg: bool, n_pol: int) -> Built:
    """Return the two-channel mixture on a spatial route over eight fixed steps.

    Eight radial nodes, or a 4 x 4 Cartesian plane, a 10 fs, 1 nJ beam of 80 um waist;
    the mixture's Kerr term with or without its harmonic; every mixture statistic
    observed.
    """
    mixture, grid, _, responses, (_, original_request) = mixture_translation()
    space = (
        RadialGrid(6e-4, 8, collar_fraction=0.0)
        if geometry == "radial"
        else FreeGrid(3e-4, 4, 4e-4, 4, collar_fraction=0.0)
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-9),
        80e-6,
        jones=(1,) if n_pol == 1 else (1, 0.5j),
    )
    responses = tuple(
        dataclasses.replace(response, thg=thg)
        if isinstance(response, KerrResponse)
        else response
        for response in responses
    )
    problem, run_request = assemble(
        grid,
        None,
        responses,
        pulse.build(grid, space, medium=SpatialMedium(mixture)),
        spatial_grid=space,
        medium=SpatialMedium(mixture),
        controller=original_request.stepper.controller,
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length / 8,
        schedule=FixedSchedule.uniform(0, grid.length, grid.length / 8),
    )
    request = run_request.for_run(stats=StatsSpec(names=MIXTURE_STATISTICS))
    return Built(problem, request, grid)


def channel_budget(built: Built) -> float:
    """Bound the channels' run: ``(64 n_to 2 + 128 n_nodes) 7 8 eps``.

    Two exposure scans and two accumulated current integrals cost at most ``64 eps``
    per temporal sample, the spatial synthesis and projection ``128 eps`` per node;
    summed over the eight seven-stage steps.
    """
    nodes = built.problem.transform.n_nodes
    return (64 * built.grid.n_to * 2 + 128 * nodes) * 7 * 8 * EPS


def held_as_the_mixture(built: Built, name: str, bound: float) -> Statistic:
    """Hold a mixture statistic as the old test did, its fractions absolutely.

    Relative at every element to the rung's bound; the fractions, which the old test
    gave an absolute floor of the same size beside the relative one, absolutely to it
    alone: the stricter.
    """
    if name == "species_ionisation":
        return Statistic(bound, absolute=True)
    return Statistic(bound, elementwise=True)


# --- the channels' fractions at a launch ----------------------------------------------

FAMILIES = {False: "real column", True: "analytic column"}
"""The plasma column each fractions row reads: the real field's, or the analytic
signal's (a Kerr response without its harmonic)."""


def fractions_at_the_launch(kerr: bool, n_pol: int) -> Built:
    """Return the two channels at a radial launch of 3.6e10 V/m, their two statistics.

    ``native_mixture_launch_probe.launch_case``: eight radial nodes, the launch scaled
    so that its largest node field is 3.6e10 V/m, where both channels ionise well past
    their pre-ionisation (0.03 and 0.10) on axis.
    """
    grid, problem, run_request = launch_case(kerr, n_pol)
    return Built(problem, run_request, grid)


def fraction_budget(built: Built) -> float:
    """Bound the two channels' fractions and peak rates: ``64 n_to eps``.

    Each of the two exposure scans accumulates ``n_to`` terms and the rate
    interpolation adds its log and exp error: absolute for the fractions, relative for
    the peak rates, which have no cumulative sum.
    """
    return 64 * built.grid.n_to * EPS


def held_as_the_passes(built: Built, name: str, bound: float) -> Statistic:
    """Hold the fractions absolutely and the peak rates relative at every element."""
    if name == "species_ionisation":
        return Statistic(bound, absolute=True)
    return Statistic(bound, elementwise=True)


def _frozen(values: np.ndarray) -> np.ndarray:
    """Return `values` read-only, as every array crossing the seam is."""
    values.setflags(write=False)
    return values


def _silent(table: Any) -> Any:
    """Return a rate table scaled by ``exp(-1000)``: its rate is zero everywhere."""
    coefficients = np.array(table.coefficients)
    coefficients[-1] -= 1000.0
    return dataclasses.replace(
        table,
        log_rate=_frozen(np.array(table.log_rate) - 1000.0),
        coefficients=_frozen(coefficients),
        rate_max=table.rate_max * math.exp(-1000.0),
    )


def without_ionisation(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem with every field channel's rate scaled by ``exp(-1000)``.

    No population ionises: each keeps its pre-ionisation, and its density, so its
    pre-ionised plasma, is kept.
    """
    responses = problem.responses
    silenced = {id(table): _silent(table) for table in responses.ionisation_table}
    channels = tuple(
        dataclasses.replace(channel, rate=silenced[id(channel.rate)])
        for channel in responses.plasma_channels
    )
    return dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            responses,
            ionisation_table=tuple(silenced.values()),
            plasma_channels=channels,
        ),
    )


# --- the axial flux -------------------------------------------------------------------

FLUX_GROUPS = {
    "energy and spectra": (
        "energy",
        "beam_spectrum",
        "region_spectrum",
        "aperture_spectrum",
    ),
    "coherent spectra": (
        "absorbed_power",
        "aperture_on_axis_spectrum",
        "on_axis_spectrum",
    ),
    "local": ("on_axis_intensity", "peak_intensity"),
    "widths": ("beam_radius", "fwhm_r"),
}
"""The flux statistics by the rounding budget they share (``test_flux_parity._budget``):
a sum of squared moduli; the same with each amplitude's coherent sum over the columns;
the local values synthesised at the nodes; and the widths, which interpolate the
fluence."""


def flux_case(case: str, group: str) -> Built:
    """Return one flux case's chirped state at its plane, observing one group.

    ``test_flux_parity``'s state: the case's launch with a chirp-like phase on every
    column (seed 34), so that the frequencies and the columns interfere at the nodes,
    loaded at the case's plane (the radial gradient's past its entrance). The request
    observes the group's statistics that the case reads; each is the old test's to the
    bit (measured).
    """
    build, z = flux_cases.CASES[case]
    grid, problem, run_request = build()
    rng = np.random.default_rng(34)
    field = problem.initial.field0
    field = field * np.exp(1j * rng.uniform(-0.3, 0.3, field.shape[-1]))
    spec = run_request.observation.stats
    assert spec is not None
    names = tuple(name for name in FLUX_GROUPS[group] if name in spec.names)
    launched = dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=field, z=z)
    )
    return Built(
        launched,
        run_request.for_run(stats=dataclasses.replace(spec, names=names)),
        grid,
    )


def flux_budget(group: str) -> Callable[[Built], float]:
    """Return the bound of one group: its statistics' shared rounding budget."""

    def bound(built: Built) -> float:
        return flux_cases.budget(FLUX_GROUPS[group][0], built.grid, built.problem)

    bound.__name__ = f"flux_budget({group!r})"
    return bound


def _with_flux(problem: PropagationProblem, flux: Any) -> PropagationProblem:
    """Return the problem with its plane measure replaced by `flux`."""
    return dataclasses.replace(
        problem, route=dataclasses.replace(problem.route, flux=flux)
    )


def _flux(problem: PropagationProblem) -> Any:
    """Return a spatial problem's plane measure."""
    flux = flux_of(problem.route)
    assert flux is not None
    return flux


def lossless(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem whose plane measure reads a lossless medium: ``Re k^2``."""
    flux = _flux(problem)
    base = dataclasses.replace(
        flux.base, k_squared=np.real(flux.base.k_squared).astype(complex)
    )
    return _with_flux(problem, dataclasses.replace(flux, base=base))


def at_the_entrance(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem whose plane measure reads the medium at its entrance."""
    flux = _flux(problem)
    return _with_flux(problem, dataclasses.replace(flux, material=None, admission=None))


def without_walls(problem: PropagationProblem) -> PropagationProblem:
    """Return the planar problem whose plane measure has no wall terms."""
    flux = _flux(problem)
    base = dataclasses.replace(flux.base, kappa_sq=np.zeros_like(flux.base.kappa_sq))
    return _with_flux(problem, dataclasses.replace(flux, base=base))


FLUX_CASES: dict[str, tuple[str, str, Callable[[Any], Any]]] = {
    "radial-lossy": ("radial", "lossy glass", lossless),
    "radial-gradient": (
        "radial",
        "argon gradient read past the entrance",
        at_the_entrance,
    ),
    "cartesian-lossy": ("cartesian", "lossy glass, apertures, regions", lossless),
    "line-lossy": ("line", "lossy glass, apertures, regions", lossless),
    "planar": ("planar", "slab walls", without_walls),
}
"""Each flux case's route, features and switch-off."""

FLUX_TERMS = {
    "radial-lossy": "the glass's loss",
    "radial-gradient": "the argon gradient's table",
    "cartesian-lossy": "the glass's loss",
    "line-lossy": "the glass's loss",
    "planar": "the slab's wall terms",
}
"""What each flux case's switch-off removes."""

FLUX_WIDTHS_MOVED = {
    "radial-lossy": "0.19 and 0.21",
    "radial-gradient": "81 and 90",
    "cartesian-lossy": "0.15 and 0.18",
    "line-lossy": "0.19 and 0.31",
    "planar": "0.046 and 0.0093",
}
"""How many width budgets each case's switch-off moves the reference's beam radius and
FWHM by (measured)."""

# --- the synthetic separable leaves ---------------------------------------------------

SYNTHETIC_STAGES = {
    "planar stage material absorbed": (True, True),
    "planar stage material": (True, False),
    "planar stage absorbed": (False, True),
    "planar stage bare": (False, False),
}
"""The Runge-Kutta stage cases: the Kerr term and the in-stage collar, each or not."""

SYNTHETIC_MARCHES = (
    *(
        ("cartesian march", mode, n_pol)
        for n_pol in (1, 2)
        for mode in ("tabulate", "recompute")
    ),
    *(("planar march", mode, 1) for mode in ("tabulate", "recompute")),
)
"""The low-storage cases: the leaf, its flows' mode and its polarisations."""

SYNTHETIC_CASES = (
    *SYNTHETIC_MARCHES,
    *((case, "auto", 1) for case in SYNTHETIC_STAGES),
)
"""Every synthetic case, as :func:`synthetic` takes it."""


def synthetic(case: str, mode: str = "auto", n_pol: int = 1) -> Built:
    """Return one of the retired kernels' synthetic leaves on a small problem.

    ``"cartesian march"`` and ``"planar march"`` take the low-storage march; a
    ``"planar stage ..."`` case the Runge-Kutta march, with the Kerr term and the
    in-stage temporal collar each present or not (``test_native_synthetic_operator``).
    One order-one step of an order-one random state on the leaf's active bins.
    """
    if case == "cartesian march":
        problem, run_request, grid = leaves.free_problem(
            responses=[leaves.KERR], n_pol=n_pol, mode=mode
        )
        leaf = leaves.free_leaf(problem.linear.separable)
    elif case == "planar march":
        problem, run_request, grid = leaves.slab_problem(
            responses=[leaves.KERR], mode=mode
        )
        leaf = leaves.slab_leaf(problem.linear.separable)
    else:
        material, absorbed = SYNTHETIC_STAGES[case]
        problem, run_request, grid = leaves.slab_problem(
            responses=[leaves.KERR] if material else [],
            collar=absorbed,
            integrator=None,  # pyright: ignore[reportArgumentType]
        )
        leaf = leaves.slab_leaf(problem.linear.separable)
    return Built(leaves.on_leaf(problem, leaf), run_request, grid)


def synthetic_step(built: Built) -> float:
    """Bound the one step: ``test_native_synthetic_operator.step_limit`` on the leaf."""
    leaf = built.problem.linear.separable
    return leaves.step_limit(built.problem, built.run_request, built.grid, leaf)


def the_leaf_is_pathological(built: Built) -> bool:
    """Return whether every case the original named occurs on the row's leaf.

    ``test_native_synthetic_operator.assert_the_leaf_is_pathological``; on a stage case
    the two wall channels also differ, since the same ``k_y`` is another rate.
    """
    leaf = built.problem.linear.separable
    assert leaf is not None
    leaves.assert_the_leaf_is_pathological(leaf)
    if leaf.kappa_sq is not None and built.run_request.stepper.integrator is None:
        operator = leaves.separable_values(leaf)[0][0]
        assert not np.allclose(operator[0], operator[1])
    return True


def without_the_ramp(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem on its leaf without the safety ramp's rate."""
    leaf = problem.linear.separable
    assert leaf is not None and leaf.kperp_rate is not None
    bare = dataclasses.replace(leaf, kperp_rate=np.zeros_like(leaf.kperp_rate))
    return dataclasses.replace(
        problem,
        linear=dataclasses.replace(problem.linear, separable=bare),
        responses=dataclasses.replace(problem.responses, separable=bare),
        route=dataclasses.replace(
            problem.route, flux=dataclasses.replace(_flux(problem), base=bare)
        ),
    )


SYNTHETIC_QUIET = (
    "ignore::solprop.exceptions.BoundaryWarning",
    "ignore::solprop.exceptions.MaterialRangeWarning",
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning",
)
"""The synthetic module's warning filters, which its rows keep."""

# --- a requested collar on every route ------------------------------------------------

COLLAR_ROUTES = ("radial", "cartesian", "line", "planar")
"""Every route a requested collar runs on."""

MARCHES = {"rk": RungeKutta, "adams": Multistep}
"""The two marches a collar run takes, by the name in a row's id."""


def collar_run(route: str, march: str) -> Built:
    """Return a linear run with a requested collar, a converging beam reaching it.

    ``test_native_edge_collar``'s case under its ``INWARD`` launch, on a matched fixed
    schedule of eight steps (128 on the planar route) saved at half and the whole of
    its 2 mm; the run's own problem, request, options and modes.
    """
    steps = 16 * collar.STEPS if route == "planar" else collar.STEPS
    simulation: Simulation = collar.collar_case(
        route,
        **collar.INWARD,
        backend="numpy_ref",
        integrator=MARCHES[march](),
        schedule=FixedSchedule.uniform(
            0.0,
            collar.LENGTH,
            collar.LENGTH / steps,
            save_planes=[collar.LENGTH / 2, collar.LENGTH],
        ),
        controller=None,
        dz_max=collar.LENGTH / steps,
    )
    return Built(
        simulation.translate(),
        simulation._run_request(),
        simulation.grid,
        options=simulation.settings.execution_options(),
        modes=simulation.modes,
        simulation=simulation,
    )


PLANAR_COLLAR = {
    "steps": Detection(
        500.0,
        "The planar launch cannot converge on the y collar, which takes 7.6e-8 of the "
        "field over the run: removing it moves the reference by 762 of the ad hoc "
        "1e-10 bounds (measured), against 7,900 to 10,000 on the other routes.",
    )
}
"""The planar collar row's lowered guard."""


def collar_bound(built: Built) -> float:
    """Bound the collar run's field: the old test's ``1e-10`` of its norm."""
    return 1e-10


def without_the_collar(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem without its real-space collar: the geometry's rate zeroed."""
    geometry = problem.transform.geometry
    assert geometry is not None
    rate = np.zeros_like(geometry.absorber_rate)
    rate.setflags(write=False)
    transform = dataclasses.replace(
        problem.transform,
        geometry=dataclasses.replace(geometry, absorber_rate=rate),
    )
    return dataclasses.replace(
        problem, route=dataclasses.replace(problem.route, transform=transform)
    )


# --- the rows -------------------------------------------------------------------------


def geometry_row(
    *,
    name: str,
    route: str,
    stepper: str = "none",
    response: str,
    features: str,
    build: Callable[[], Built],
    **fields: Any,
) -> Row:
    """Return a row of the geometry family, medium when it marches.

    Its id is ``geometry {name}``; `fields` are the rest of :class:`parity_ladder.Row`.
    """
    return Row(
        id=f"geometry {name}",
        axes={
            "route": route,
            "stepper": stepper,
            "response": response,
            "features": features,
        },
        build=build,
        medium="steps" in fields["rungs"],
        **fields,
    )


def population_rows() -> list[Row]:
    """Return the envelope populations' rows: three geometries, two polarisations."""
    return [
        geometry_row(
            name=f"envelope populations {geometry} {POLARISATIONS[n_pol]}",
            route=geometry,
            stepper="rk",
            response="envelope plasma",
            features=f"two populations, partial chunks, {POLARISATIONS[n_pol]}, "
            "every population statistic",
            why=f"Two envelope plasma populations on the {geometry} geometry are the "
            "reference's after two fixed steps, in partial node chunks, and so is "
            "every population statistic at the launch and at the end.",
            bound_note="The geometry ladder's 128 n_to 7 2 eps, held "
            "peak-normalised: the old assert_allclose(rtol=b, atol=b max|ref|) without "
            "its relative part, the stricter.",
            build=partial(population_case, geometry, n_pol),
            switch_off=without_plasma,
            rungs={"launch": population_budget, "steps": population_budget},
            unguarded={"launch": THE_LAUNCH_HOLDS_NO_DENSITY},
            every_statistic=held_as_the_field,
            metric="peak",
        )
        for geometry in GEOMETRIES
        for n_pol in (1, 2)
    ]


def mixture_rows() -> list[Row]:
    """Return the mixture's channel rows: two routes, with and without the harmonic."""
    return [
        geometry_row(
            name=f"mixture channels {geometry} {'harmonic' if thg else 'no harmonic'} "
            f"{POLARISATIONS[n_pol]}",
            route=geometry if geometry == "radial" else "cartesian",
            stepper="rk",
            response="kerr, plasma",
            features=f"two channels, {'third harmonic' if thg else 'no harmonic'}, "
            f"{POLARISATIONS[n_pol]}, every mixture statistic",
            why=f"Two plasma channels on the {geometry} route are the reference's "
            "after eight fixed steps, and so is every population statistic at the "
            "launch and at the end.",
            bound_note="(64 n_to 2 + 128 n_nodes) 7 8 eps, normwise for the field and "
            "relative at every element for the statistics; the fractions absolutely, "
            "where the old test also allowed the bound relative to them.",
            build=partial(mixture_channels, geometry, thg, n_pol),
            switch_off=without_ionisation,
            rungs={"launch": channel_budget, "steps": channel_budget},
            every_statistic=held_as_the_mixture,
        )
        for geometry in ("radial", "free")
        for thg in (False, True)
        for n_pol in (1, 2)
    ]


def fraction_rows() -> list[Row]:
    """Return the channels' fractions at a launch: two columns, two polarisations."""
    return [
        geometry_row(
            name=f"mixture fractions {FAMILIES[kerr]} {POLARISATIONS[n_pol]}",
            route="radial",
            response="kerr, plasma" if kerr else "plasma",
            features=f"two channels, {FAMILIES[kerr]}, {POLARISATIONS[n_pol]}, "
            "launch statistics",
            why=f"At a launch of 3.6e10 V/m on the {FAMILIES[kerr]} native reads both "
            "channels' final fractions and peak ionisation rates as the reference's "
            "passes on its own node fields.",
            bound_note="64 n_to eps: the fractions absolutely, the peak rates "
            "relative at every element, as the old test held them against "
            "passes.plasma_channel_fractions, which the reference's launch statistics "
            "are to the bit (measured). The launch rung also compares both "
            "peak-normalised at the same bound (new).",
            build=partial(fractions_at_the_launch, kerr, n_pol),
            switch_off=without_ionisation,
            rungs={"launch": fraction_budget},
            every_statistic=held_as_the_passes,
            metric="peak",
        )
        for kerr in (False, True)
        for n_pol in (1, 2)
    ]


def flux_rows() -> list[Row]:
    """Return the flux statistics' rows: five cases, a row per group of budgets."""
    rows = []
    for case, (route, features, switch_off) in FLUX_CASES.items():
        for group in FLUX_GROUPS:
            guard: dict[str, Any] = {}
            if group == "widths":
                guard["vacuous"] = (
                    f"Switching off {FLUX_TERMS[case]} moves the reference's beam "
                    f"radius and FWHM by {FLUX_WIDTHS_MOVED[case]} of their budget "
                    "(measured): nothing of the problem moves a fluence width beyond "
                    "it. Kept as the old test held them, a matched computation of one "
                    "state."
                )
            rows.append(
                geometry_row(
                    name=f"flux {case} {group}",
                    route=route,
                    response="none",
                    features=f"{features}, flux statistics ({group}), chirped state",
                    why=f"Native measures the {group} of one chirped state on the "
                    f"{route} route ({features}) at its plane as the reference does.",
                    bound_note="The old test's budget for these statistics, each "
                    "normwise; the row reads the worst of the group, which shares it.",
                    build=partial(flux_case, case, group),
                    switch_off=switch_off,
                    rungs={"launch": flux_budget(group)},
                    quiet=FLUX_QUIET,
                    **guard,
                )
            )
    return rows


FLUX_QUIET = (
    "ignore::solprop.exceptions.MaterialRangeWarning",
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning",
)
"""The flux module's warning filters, which its rows keep."""


def synthetic_rows() -> list[Row]:
    """Return the synthetic leaves' rows: two marches, and the planar stage cases."""
    rows = [
        geometry_row(
            name=f"synthetic {case} {mode}"
            + (f" {POLARISATIONS[n_pol]}" if case.startswith("cartesian") else ""),
            route=case.split()[0],
            stepper="low-storage",
            response="kerr",
            features=f"synthetic leaf, {mode} flows, inactive bin, gain, ramp",
            why="On the retired kernels' synthetic leaf -- loss and gain, an exactly "
            "zero Re q, evanescent columns, an inactive bin, a safety ramp"
            + (", a wall with gain and low words of 1e-17" if "planar" in case else "")
            + " -- one low-storage step is the reference's.",
            bound_note="low_storage_cases.parity_bound over the step's evaluations, "
            "grown by the leaf's gain and the Kerr rate; weaker than the original's "
            "128 eps per kernel, since a whole step is read.",
            build=partial(synthetic, case, mode, n_pol),
            switch_off=without_the_ramp,
            expect={"": the_leaf_is_pathological},
            rungs={"steps": synthetic_step},
            quiet=SYNTHETIC_QUIET,
        )
        for case, mode, n_pol in SYNTHETIC_MARCHES
    ]
    rows += [
        geometry_row(
            name=f"synthetic {case}",
            route="planar",
            stepper="rk",
            response="kerr" if SYNTHETIC_STAGES[case][0] else "none",
            features="synthetic leaf, wall, inactive bin, gain, ramp"
            + (", in-stage collar" if SYNTHETIC_STAGES[case][1] else ""),
            why="On the retired kernels' synthetic planar leaf, with the Kerr term "
            f"{'present' if SYNTHETIC_STAGES[case][0] else 'absent'} and the in-stage "
            f"collar {'present' if SYNTHETIC_STAGES[case][1] else 'absent'}, one "
            "Runge-Kutta step of seven stages is the reference's.",
            bound_note="low_storage_cases.parity_bound over the step's seven "
            "evaluations, grown by the leaf's gain and the Kerr rate.",
            build=partial(synthetic, case),
            switch_off=without_the_ramp,
            expect={"": the_leaf_is_pathological},
            rungs={"steps": synthetic_step},
            quiet=SYNTHETIC_QUIET,
        )
        for case in SYNTHETIC_STAGES
    ]
    return rows


def collar_rows() -> list[Row]:
    """Return the requested collar's rows: four routes, two marches."""
    return [
        geometry_row(
            name=f"edge collar {route} {march}",
            route=route,
            stepper=march,
            response="none",
            features="requested collar, converging beam, matched schedule",
            why=f"A linear run on the {route} route with a requested collar, a "
            "converging beam reaching it, is the reference's at its end on a matched "
            f"{march} schedule.",
            bound_note="An ad hoc 1e-10 of the field's norm, kept (#807); the old "
            "test's driver comparison of the last plane, which this rung's session "
            "reads to the bit (measured).",
            build=partial(collar_run, route, march),
            switch_off=without_the_collar,
            rungs={"steps": collar_bound},
            detection=PLANAR_COLLAR if route == "planar" else {},
            quiet=("ignore::solprop.exceptions.BoundaryWarning",),
        )
        for route in COLLAR_ROUTES
        for march in MARCHES
    ]


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every row of the geometry family."""
    return (
        *population_rows(),
        geometry_row(
            name="envelope populations taper and density",
            route="capillary (taper)",
            stepper="rk",
            response="envelope plasma",
            features="two populations, radius taper, densities falling to zero",
            why="Along a radius taper whose populations' densities fall to zero, two "
            "envelope plasma populations are the reference's after two fixed steps.",
            bound_note="The geometry ladder's 128 n_to 7 2 eps: the two stages of "
            "profile-table interpolation add to the same 128 eps per sample source "
            "and Fourier budget. Held peak-normalised (the old assert_allclose's "
            "relative part dropped, the stricter).",
            build=population_taper,
            switch_off=without_plasma,
            expect={"": the_taper_reaches_each_population},
            rungs={"steps": population_budget},
            metric="peak",
        ),
        *mixture_rows(),
        *fraction_rows(),
        *flux_rows(),
        *synthetic_rows(),
        *collar_rows(),
    )
