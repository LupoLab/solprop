# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthetic saved runs, so that the examples in this package stay short.

Documentation and testing helpers: the pulse is a Gaussian built by
:mod:`solprop.pulses` and carried along ``z`` by the closed-form phase of pure
group-velocity dispersion, which is the one propagation whose answer is known exactly
and needs no solver. Everything in :mod:`solprop.processing` and :mod:`solprop.plotting`
can therefore show a runnable example without importing a backend. Only
:func:`example_output` is public, as :func:`solprop.processing.example_output`, so that
the guide's snippets run as written; the others serve the package's own examples.
"""

import dataclasses

import numpy as np
from jaxtyping import Complex

from solprop.grid import EnvelopeGrid, FieldGrid, TimeGrid
from solprop.modes.guide import ModalGuide
from solprop.modes.protocol import ChannelLabel
from solprop.output.memory import MemoryOutput
from solprop.output.schema import (
    GridRecord,
    ModeRecord,
    RunMetadata,
    SavedPlane,
    StepperRecord,
    StepStats,
)
from solprop.pulses import GaussianPulse
from solprop.stats import (
    GENERAL_MODAL_ACCESSORS,
    StatsSpec,
    collect,
    default_spec,
)
from solprop.translate.modal_tables import ModalTransformTables

__all__ = [
    "example_general_output",
    "example_general_run",
    "example_grid",
    "example_output",
    "gvd_phase",
]

_WAVELENGTH = 800e-9
"""Central wavelength of the example pulse, m."""

_FWHM = 30e-15
"""Intensity full width at half maximum of the example pulse, s."""

_ENERGY = 1e-6
"""Energy of the example pulse, J."""

_LENGTH = 0.1
"""Propagation length of the example run, m."""

_TRANGE = 400e-15
"""Time window of the example grids, s."""

_ROWS_PER_PLANE = 3
"""Between-planes statistics rows the example run records per plane."""

_LIMITS = (400e-9, 2e-6)
"""Wavelength limits of the example grids, m."""


def example_grid(kind: str = "envelope") -> TimeGrid:
    """Return the small grid the synthetic runs are built on.

    Parameters
    ----------
    kind : {'envelope', 'field'}
        Which spectral layout to build.

    Returns
    -------
    solprop.grid.TimeGrid
        A grid over 400 nm to 2 um with a 400 fs window, small enough for a doctest.

    Examples
    --------
    >>> from solprop.processing._examples import example_grid
    >>> example_grid("field").kind
    'field'
    """
    if kind == "field":
        return FieldGrid.from_limits(_LENGTH, _WAVELENGTH, _LIMITS, _TRANGE)
    return EnvelopeGrid.from_limits(_LENGTH, _WAVELENGTH, _LIMITS, _TRANGE)


def gvd_phase(
    grid: TimeGrid, beta2: float, z: float
) -> Complex[np.ndarray, " n_omega"]:
    """Return the propagator of pure group-velocity dispersion, in the FFT convention.

    In the co-moving frame the linear operator keeps only the quadratic term of the
    dispersion relation, so ``D = -i beta2 (omega - omega0)^2 / 2`` and the stored
    spectrum is multiplied by ``exp(D z)`` (``conventions.md`` section 2). A Gaussian of
    duration ``tau0`` then has the closed form
    ``tau(z) = tau0 sqrt(1 + (beta2 z / tau0^2)^2)``.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    beta2 : float
        Group-velocity dispersion, s^2/m.
    z : float
        Propagation distance, m.

    Returns
    -------
    numpy.ndarray
        The phasor, shape ``(n_omega,)``, multiplying an internal-convention
        spectrum.

    Examples
    --------
    >>> from solprop.processing._examples import example_grid, gvd_phase
    >>> grid = example_grid()
    >>> gvd_phase(grid, 1e-27, 0.0).tolist() == [1] * grid.n_omega
    True
    """
    return np.exp(-0.5j * beta2 * (grid.omega - grid.omega0) ** 2 * z)


def example_output(
    kind: str = "envelope",
    *,
    n_modes: int = 1,
    n_pol: int = 1,
    n_z: int = 5,
    with_stats: bool = True,
    with_step_stats: bool = False,
) -> MemoryOutput:
    """Return a synthetic run: a Gaussian broadened by pure dispersion.

    The group-velocity dispersion is chosen so that the pulse doubles its duration over
    the length, which makes every plot and every metric change visibly from plane to
    plane. Modes and polarisation components carry the same pulse at decreasing
    amplitude, so a modal sum differs from any single mode.

    Parameters
    ----------
    kind : {'envelope', 'field'}
        Which spectral layout to build the run on.
    n_modes : int
        Number of propagated modes.
    n_pol : int
        Number of polarisation components, 1 or 2.
    n_z : int
        Number of saved planes, spread evenly from 0 to the grid's length.
    with_stats : bool
        Whether to record the default statistics at each plane.
    with_step_stats : bool
        Whether to record the same statistics *between* planes too, as a run with a
        periodic subset does, so that a reader of the channel has something to read.

    Returns
    -------
    solprop.output.memory.MemoryOutput
        An opened handler holding ``n_z`` planes.

    Examples
    --------
    >>> from solprop.processing._examples import example_output
    >>> output = example_output(n_modes=2)
    >>> output.n_saved, output["field"].shape[1:3]
    (5, (2, 1))
    """
    grid = example_grid(kind)
    # tau0 is the 1/e half-width of the field; the closed form of the broadening is
    # written in terms of it rather than of the intensity FWHM.
    tau0 = _FWHM / (2.0 * np.sqrt(np.log(2.0)))
    beta2 = np.sqrt(3.0) * tau0**2 / _LENGTH
    simulation_kind = "multimode" if n_modes > 1 else "mode_averaged"
    metadata = RunMetadata(
        grid=GridRecord.from_grid(grid),
        modes=tuple(
            ModeRecord(
                kind="solprop.modes.capillary.MarcatiliMode",
                effective_area=0.5e-8 / (index + 1),
                configuration={"index": index},
            )
            for index in range(n_modes)
        ),
        simulation_kind=simulation_kind,
        configuration={"beta2": beta2},
        stepper=StepperRecord(tableau="DOPRI5", controller={"rtol": 1e-8, "atol": 0.0}),
    )
    seed = _seed_field(grid, n_modes, n_pol)
    spec = default_spec(simulation_kind)
    output = MemoryOutput()
    output.open(metadata)
    planes = np.linspace(0.0, _LENGTH, n_z)
    for index, z in enumerate(planes):
        field = seed * gvd_phase(grid, beta2, float(z))
        plane = SavedPlane(z=float(z), field=field)
        if with_stats:
            plane = SavedPlane(
                z=float(z), field=field, stats=collect(spec, grid, plane)
            )
        if with_step_stats and index:
            plane = dataclasses.replace(
                plane,
                step_stats=_example_rows(
                    grid, spec, seed, beta2, float(planes[index - 1]), float(z)
                ),
            )
        output.save(plane)
    return output


def example_general_output(n_z: int = 3) -> MemoryOutput:
    """Return a synthetic **general modal** run, for the general modal accessors.

    The same closed-form dispersion as :func:`example_output`, with the record of a
    general modal guide bolted on: one labelled channel, a table, metric and guide
    identity, an active band and the accessor list a three-component basis admits.
    Nothing here solves a provider -- building a real guide costs half a second --
    so the reconstruction accessors, which need the reference arrays themselves,
    are exercised by the tests and not by a doctest.

    Its ``tables`` and ``guide`` blocks are therefore deliberate **subsets** of
    what :func:`solprop.translate.general.modal_identity` writes: enough for the
    label, printing and energy examples, and not an identity a reconstruction
    could be checked against. A reconstruction offered this record is refused as
    incomplete by :mod:`solprop.processing.modal`, which is where that branch is
    exercised.

    Parameters
    ----------
    n_z : int
        Number of saved planes, spread evenly from 0 to the grid's length.

    Returns
    -------
    solprop.output.memory.MemoryOutput
        An opened handler holding ``n_z`` planes of a general modal run.

    Examples
    --------
    >>> from solprop.processing._examples import example_general_output
    >>> example_general_output().metadata.modal["basis"]
    'general_modal'
    """
    grid = example_grid("field")
    tau0 = _FWHM / (2.0 * np.sqrt(np.log(2.0)))
    beta2 = np.sqrt(3.0) * tau0**2 / _LENGTH
    label = ChannelLabel("HE", 1, 1, "even", None, "HE1,1:even")
    metadata = RunMetadata(
        grid=GridRecord.from_grid(grid),
        modes=(
            ModeRecord(
                kind="solprop.modes.guide.ProviderChannelMode",
                effective_area=1.2e-11,
                configuration={"channel": 0},
                label=dict(label.as_mapping()),
            ),
        ),
        simulation_kind="multimode",
        configuration={"beta2": beta2},
        stepper=StepperRecord(tableau="DOPRI5", controller={"rtol": 1e-8, "atol": 0.0}),
        modal={
            "basis": "general_modal",
            "n_components": 3,
            "n_channels": 1,
            "channels": [dict(label.as_mapping())],
            "tables": {
                "model_id": "example-guide",
                "gauge_id": "analytic-real-transverse",
                "policy_id": "dense-knots-cubic-frequency",
                "n_modes": 1,
            },
            "metric": {
                "kind": "identity",
                "gauge_id": "analytic-real-transverse",
                "certificate_id": "sampled-gram",
                "defect": 1e-8,
                "budget": 1e-6,
            },
            "responses": None,
            "guide": {
                "provider": {"model": "bound-step-index"},
                "sampling": {"n_knots": 13, "n_nodes": 432, "profile_error": 4.0e-5},
                "regions": [{"name": "core"}, {"name": "cladding"}],
            },
            "frame": "shared co-moving",
            "power_weight": "constant",
            "active_band": [
                float(grid.omega[grid.active_mask][0]),
                float(grid.omega[grid.active_mask][-1]),
            ],
            "noise_reference_basis": None,
            "accessors": sorted(GENERAL_MODAL_ACCESSORS),
        },
    )
    seed = _seed_field(grid, 1, 1)
    spec = StatsSpec(names=("omega0", "physical_energy", "analytic_flux"))
    output = MemoryOutput()
    output.open(metadata)
    for z in np.linspace(0.0, _LENGTH, n_z):
        field = seed * gvd_phase(grid, beta2, float(z))
        plane = SavedPlane(z=float(z), field=field)
        output.save(
            SavedPlane(z=float(z), field=field, stats=collect(spec, grid, plane))
        )
    return output


def example_general_run() -> tuple[MemoryOutput, ModalTransformTables, ModalGuide]:
    """Return a **real** general modal run, with the tables and guide it was run on.

    The one example here that solves and propagates, for the accessors that need
    the reference arrays and not only the record: one ``LP_01`` channel of a
    parabolic graded-index fibre (25 um core, contrast 0.0095, a Kerr core) sampled
    at nine knots on a coarse 16-element rule, and two 5 um steps on
    ``numpy_ref`` with two saved planes. About a second, most of it the run's
    first compilation of its plans.

    Returns
    -------
    tuple
        The opened in-memory output, the
        :class:`~solprop.translate.modal_tables.ModalTransformTables` rebuilt from
        the guide, and the :class:`~solprop.modes.guide.ModalGuide` itself.

    Examples
    --------
    >>> from solprop.processing._examples import example_general_run
    >>> run, tables, guide = example_general_run()
    >>> run.metadata.modal["basis"]
    'general_modal'
    """
    # Imported here: this module is otherwise backend-free, and only this helper
    # solves a provider and runs a propagation.
    from solprop.driver import Simulation  # noqa: PLC0415
    from solprop.modes import (  # noqa: PLC0415
        GradedIndexProvider,
        IndexMaterial,
        MaterialKerr,
        PowerLawPermittivity,
    )
    from solprop.translate.general import general_modal_route  # noqa: PLC0415

    wavelength, length, contrast = 1.03e-6, 1e-5, 0.0095
    grid = FieldGrid.from_limits(length, wavelength, (1.0e-6, 1.06e-6), 200e-15)
    provider = GradedIndexProvider(
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), contrast),
        exterior=IndexMaterial(1.45 * np.sqrt(1.0 - 2.0 * contrast)),
        radius=25e-6,
        family="LP",
        count=1,
        elements=16,
        extent=2.0,
        core_responses=(MaterialKerr(n2=2.7e-20, at=wavelength),),
    )
    active = grid.omega[grid.active_mask]
    knots = np.linspace(0.99 * float(active[0]), 1.01 * float(active[-1]), 9)
    guide = ModalGuide.from_provider(
        provider,
        knots,
        nodes=provider.node_rule(float(knots[0]), 0.0, n_r=2, n_theta=2),
    )
    simulation = Simulation.from_waveguide(
        (),
        length,
        provider=guide,
        grid=grid,
        wavelength=wavelength,
        fwhm=30e-15,
        energy=1e-12,
        dz_max=length / 2,
        dz_init=length / 2,
        n_save=2,
        backend="numpy_ref",
    )
    output = simulation.run(progress="none")
    assert isinstance(output, MemoryOutput)
    tables = general_modal_route(
        guide, grid, z_max=length, omega_reference=float(grid.omega0)
    ).tables
    return output, tables, guide


def _example_rows(
    grid: TimeGrid,
    spec: StatsSpec,
    seed: Complex[np.ndarray, "n_modes n_pol n_omega"],
    beta2: float,
    start: float,
    end: float,
) -> StepStats:
    """Return the between-planes rows of one plane, on a few steps inside it.

    The same statistics the plane records, evaluated at intermediate positions, which
    is what a real run's channel holds -- there the backend evaluates them inside its
    own step loop, here the closed-form propagation stands in for one.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the run is on.
    spec : solprop.stats.StatsSpec
        The statistics to evaluate.
    seed : numpy.ndarray
        The input plane.
    beta2 : float
        Group-velocity dispersion, s^2/m.
    start : float
        Position of the previous plane, m.
    end : float
        Position of this plane, m.

    Returns
    -------
    solprop.output.schema.StepStats
        Three rows, the last of them at the plane itself.
    """
    positions = np.linspace(start, end, _ROWS_PER_PLANE + 1)[1:]
    rows = [
        collect(
            spec,
            grid,
            SavedPlane(z=float(z), field=seed * gvd_phase(grid, beta2, float(z))),
        )
        for z in positions
    ]
    return StepStats(
        z=positions,
        step=np.arange(_ROWS_PER_PLANE),
        values={name: np.stack([row[name] for row in rows]) for name in spec.names},
    )


def _seed_field(
    grid: TimeGrid, n_modes: int, n_pol: int
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return the input plane: one Gaussian per mode and polarisation component.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid to build on.
    n_modes : int
        Number of propagated modes.
    n_pol : int
        Number of polarisation components.

    Returns
    -------
    numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, internal convention.
    """
    pulse = GaussianPulse(wavelength=_WAVELENGTH, fwhm=_FWHM, energy=_ENERGY)
    one = pulse.build(grid)
    # Each mode and polarisation carries a weaker copy, so that no reduction over either
    # axis can be mistaken for a single component.
    weights = np.array(
        [[1.0 / ((m + 1) * (p + 1)) for p in range(n_pol)] for m in range(n_modes)]
    )
    return one[0, 0] * weights[:, :, np.newaxis]
