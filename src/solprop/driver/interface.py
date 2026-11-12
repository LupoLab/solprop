# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The keyword interface: one call from GNLSE coefficients to a written run.

:func:`prop_gnlse` builds the grid, the dispersion-specified fibre, the input pulse,
the noise realisation and the numerical settings from keywords, hands them to
:meth:`solprop.driver.simulation.Simulation.from_gnlse`, runs it and returns the output
handler it wrote. :func:`prop_capillary` and :func:`prop_waveguide` do the same for a
gas-filled capillary and for any other modal waveguide.

The step controls ``rtol``, ``atol``, ``dz_init`` and ``dz_max`` are keywords of every
interface. ``raman`` defaults to false, and the whole of ``gamma`` then acts as Kerr;
the Raman share is taken from ``gamma`` only when the Raman response is on. What a run
was asked for is recorded as structured metadata.

The manual page is ``docs/guide/setup/running.md``.
"""

import inspect
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
from jaxtyping import Float

from solprop.boundaries import Absorber, SpatialAbsorber
from solprop.coupling.models import CouplingModel
from solprop.driver.defaults import DEFAULT_ATOL, WAVEGUIDE_ATOL
from solprop.driver.inputs import ModalLaunch
from solprop.driver.resume import _ResumeChoice
from solprop.driver.settings import NoiseSetting, SimulationSettings
from solprop.driver.simulation import Simulation
from solprop.gasflow import VacuumEnd
from solprop.grid import FreeGrid, RadialGrid, TimeGrid
from solprop.linear import Frame
from solprop.log import DEFAULT_PROGRESS_PERIOD, ProgressChoice, ProgressReporter
from solprop.modes.antiresonant import TubeLattice
from solprop.modes.capillary import (
    CoherentWall,
    GlassIndex,
    LossSpec,
    MarcatiliModel,
)
from solprop.modes.guide import ModalGuide
from solprop.modes.protocol import Mode
from solprop.modes.quadrature import RadialRule
from solprop.modes.slab import PlanarGrid
from solprop.modes.tabulated import IndexTable
from solprop.monitor import DEFAULT_BOUNDARY_CHECK, BoundaryCheck
from solprop.nonlinear.ionisation import IntegrationPolicy, IonisationRate
from solprop.nonlinear.raman import MolecularOptions
from solprop.nonlinear.tdse import TDSEExecution, TDSEGasResponse
from solprop.output.schema import Output
from solprop.output.snapshots import SnapshotController
from solprop.output.tdse_archive import TDSEArchive
from solprop.physdata import Gas, Mixture
from solprop.physdata.mixtures import Pressure
from solprop.physdata.selection import MaterialSelection
from solprop.physdata.sellmeier import GasSusceptibility
from solprop.profiles import ZProfile
from solprop.pulses import PlanarPulse, Pulse, SpatialPulse, SpatialPulseSum
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.masks import AbsorberScheme
from solprop.translate.responses import KerrForm

__all__ = [
    "prop_capillary",
    "prop_free",
    "prop_gnlse",
    "prop_planar",
    "prop_waveguide",
]

logger = logging.getLogger(__name__)


_RUN_CONTROLS: Final = (
    "output",
    "snapshots",
    "resume",
    "progress",
    "progress_period",
    "tdse_archive",
)
"""The keywords of a ``prop_*`` function that are :meth:`Simulation.run`'s."""

_RUN_CONTROLS_DOC: Final = """\
output : solprop.output.schema.Output or pathlib.Path or str or None, optional
    Where the run goes: a handler, a path to stream an HDF5 file to, or ``None`` for
    a :class:`solprop.output.memory.MemoryOutput`.
snapshots : solprop.output.SnapshotController or None, optional
    Immutable local checkpoints at committed accepted steps; requires HDF5,
    all fields and non-interpolated observations.
resume : bool or {'interpolated'}, optional
    Continue the file at `output` when its record and this run's have one
    configuration hash; otherwise it is refused, naming what changed. Only a
    path can be resumed. ``"interpolated"`` also continues from a plane
    interpolated inside a step
    (:meth:`solprop.driver.simulation.Simulation.run`).
progress : ProgressReporter, {'auto', 'bar', 'log', 'none'} or None, optional
    Where progress goes; ``None`` takes the process default: a bar on a terminal
    or in a notebook, log lines otherwise (:func:`solprop.driver.run.run_problem`).
progress_period : float, optional
    Seconds between the default reporter's periodic lines, s; ``0`` leaves only the
    plane and closing lines. Ignored when `progress` is given.
tdse_archive : solprop.output.TDSEArchive or None, optional
    Archive every raw provider evaluation of `tdse` in a separate HDF5 file
    (:meth:`~solprop.driver.simulation.Simulation.run`); refused without `tdse`.
"""
"""The run controls' parameter entries, in :data:`_RUN_CONTROLS` order."""

_SECTION = re.compile(r"^(\w[\w ]*)\n-{3,}\n", re.MULTILINE)
"""A numpydoc section header: its title, then its underline."""


def _parameters(doc: str) -> str:
    """Return the entries of a cleaned docstring's ``Parameters`` section."""
    sections = list(_SECTION.finditer(doc))
    for index, section in enumerate(sections):
        if section.group(1) == "Parameters":
            end = sections[index + 1].start() if index + 1 < len(sections) else None
            return doc[section.end() : end].strip("\n") + "\n"
    msg = "the constructor's docstring has no Parameters section"
    raise AssertionError(msg)


def _run_controls(names: Sequence[str]) -> str:
    """Return the entries of the run controls among `names`, in their order."""
    entries = re.split(r"\n(?=\S)", _RUN_CONTROLS_DOC.strip("\n"))
    by_name = {entry.split(" :", 1)[0]: entry for entry in entries}
    return "".join(by_name[name] + "\n" for name in names if name in by_name)


def _propagates[**P](
    build: Callable[..., Simulation],
) -> Callable[[Callable[P, Output]], Callable[P, Output]]:
    """Give a ``prop_*`` function the parameter documentation of its constructor.

    Each keyword is documented once, on ``Simulation.from_*``; the ``prop_*``
    function's own docstring has no ``Parameters`` section, and one is composed here,
    before its ``Returns``: the constructor's entries, then its run controls'.
    """

    def compose(run: Callable[P, Output]) -> Callable[P, Output]:
        """Write the composed docstring onto `run`."""
        own = inspect.cleandoc(run.__doc__ or "")
        names = list(inspect.signature(run).parameters)
        block = (
            "Parameters\n----------\n"
            + _parameters(inspect.cleandoc(build.__doc__ or ""))
            + _run_controls(names)
            + "\n"
        )
        at = own.index("Returns\n-------\n")
        run.__doc__ = own[:at] + block + own[at:]
        return run

    return compose


def _propagate(build: Callable[..., Simulation], scope: Mapping[str, object]) -> Output:
    """Build a simulation from a ``prop_*`` call's keywords and run it.

    - ``scope``: the call's ``locals()``: the run controls go to :meth:`Simulation.run`
      and every other keyword to `build`.
    """
    keywords = dict(scope)
    run = {name: keywords.pop(name) for name in _RUN_CONTROLS if name in keywords}
    simulation = build(**keywords)
    logger.debug(
        "%s: %d modes, %d save planes, %r backend",
        build.__name__,
        len(simulation.modes),
        len(simulation.settings.save_planes),
        simulation.settings.backend,
    )
    # Each control is the type Simulation.run declares: prop_* declares it the same.
    return simulation.run(**cast("Any", run))


@_propagates(Simulation.from_free)
def prop_free(  # numpydoc ignore=PR01
    grid: TimeGrid,
    space: RadialGrid | FreeGrid,
    medium: SpatialMedium,
    responses: Sequence[object],
    initial: SpatialPulse | SpatialPulseSum | np.ndarray,
    *,
    settings: SimulationSettings,
    frame: Frame | None = None,
    noise: NoiseSetting = "default",
    seed: int | None = None,
    materials: MaterialSelection | None = None,
    absorber: Absorber | Literal[False] | None = None,
    absorber_scheme: AbsorberScheme | None = None,
    boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK,
    spatial_absorber: SpatialAbsorber | Literal[False] | None = None,
    output: Output | Path | str | None = None,
    snapshots: SnapshotController | None = None,
    resume: _ResumeChoice = False,
    progress: ProgressReporter | ProgressChoice | None = None,
    progress_period: float = DEFAULT_PROGRESS_PERIOD,
) -> Output:
    """Propagate a beam in free space and return its saved output.

    On a :class:`solprop.grid.RadialGrid` the beam is cylindrically symmetric; on a
    :class:`solprop.grid.FreeGrid` it is resolved on Cartesian axes; a
    :class:`solprop.grid.LineGrid` propagates a beam uniform in y, whose output's
    energies and powers are then per metre of y (J/m, W/m), and processing and
    plotting label them so.

    Its keywords are :meth:`~solprop.driver.simulation.Simulation.from_free`'s,
    then the run controls of :meth:`~solprop.driver.simulation.Simulation.run`.

    Returns
    -------
    solprop.output.schema.Output
        Saved fields, metadata, statistics and step history.
    """
    return _propagate(Simulation.from_free, locals())


@_propagates(Simulation.from_planar)
def prop_planar(  # numpydoc ignore=PR01
    grid: TimeGrid,
    space: PlanarGrid,
    responses: Sequence[object],
    initial: PlanarPulse | Pulse | np.ndarray,
    *,
    settings: SimulationSettings,
    frame: Frame | None = None,
    materials: MaterialSelection | None = None,
    absorber: Absorber | Literal[False] | None = None,
    absorber_scheme: AbsorberScheme | None = None,
    boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK,
    spatial_absorber: SpatialAbsorber | Literal[False] | None = None,
    output: Output | Path | str | None = None,
    snapshots: SnapshotController | None = None,
    resume: _ResumeChoice = False,
    progress: ProgressReporter | ProgressChoice | None = None,
    progress_period: float = DEFAULT_PROGRESS_PERIOD,
) -> Output:
    """Propagate a beam in a hollow planar guide, resolved in y, and return its output.

    The hybrid planar transform: the slab modes of `space` in x and a periodic, collared
    y axis (``docs/guide/waveguides/planar.md``, "Resolving y"). Diffraction uses the
    represented angular-spectrum operator. The run integrates over y, so its energies
    are in J and its powers in W. Constant gap and constant fill only. ``numpy_ref`` and
    native CPU execute the geometry.

    Its keywords are :meth:`~solprop.driver.simulation.Simulation.from_planar`'s,
    then the run controls of :meth:`~solprop.driver.simulation.Simulation.run`.

    Returns
    -------
    solprop.output.schema.Output
        Saved fields, metadata, statistics and step history.
    """
    return _propagate(Simulation.from_planar, locals())


@_propagates(Simulation.from_gnlse)
def prop_gnlse(  # numpydoc ignore=PR01
    gamma: float | None,
    length: float,
    betas: Sequence[float] | np.ndarray | None = None,
    *,
    wavelength: float,
    wavelength_limits: tuple[float, float] | None = None,
    trange: float | None = None,
    grid: TimeGrid | None = None,
    beta: tuple[Float[np.ndarray, " n_nodes"], Float[np.ndarray, " n_beta_nodes"]]
    | Callable[[Float[np.ndarray, " n_omega"]], Float[np.ndarray, " n_omega"]]
    | None = None,
    fibre: Mode | IndexTable | None = None,
    n2: float | str | None = None,
    fwhm: float | None = None,
    width: float | None = None,
    shape: str = "gauss",
    energy: float | None = None,
    peak_power: float | None = None,
    phases: Sequence[float] | np.ndarray = (),
    pulses: Sequence[Pulse] | None = None,
    loss_db_per_m: float | None = None,
    shock: bool = True,
    raman: bool = False,
    raman_fraction: float | None = None,
    noise: NoiseSetting = "default",
    seed: int | None = None,
    rtol: float | None = None,
    atol: float = DEFAULT_ATOL,
    dz_max: float | None = None,
    dz_init: float | None = None,
    schedule: FixedSchedule | None = None,
    integrator: Multistep | RungeKutta | LowStorageRungeKutta | None = None,
    interpolate_planes: bool = False,
    step_planes: int | None = None,
    save_planes: Sequence[float] | np.ndarray | None = None,
    n_save: int | None = None,
    stats: StatsSpec | None = None,
    polarisation_taper: bool = False,
    absorber: Absorber | Literal[False] | None = None,
    absorber_scheme: AbsorberScheme | None = None,
    boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK,
    backend: str = "auto",
    materials: MaterialSelection | None = None,
    output: Output | Path | str | None = None,
    snapshots: SnapshotController | None = None,
    resume: _ResumeChoice = False,
    progress: ProgressReporter | ProgressChoice | None = None,
    progress_period: float = DEFAULT_PROGRESS_PERIOD,
) -> Output:
    """Propagate a pulse through a GNLSE fibre and return the output it was written to.

    The fibre is its dispersion and its nonlinear coefficient: ``beta(omega)`` as Taylor
    coefficients about `wavelength` or as data, ``gamma`` in 1/(W m), and a power loss
    in dB/m. The mapping onto solprop's physics is
    :func:`solprop.nonlinear.gnlse.problem_from` -- ``n2 = gamma A_eff / k_0``,
    ``chi3 = (4/3) n2 eps_0 c``, an effective index of one in the normalisation
    prefactor and the shock term as that prefactor's ``omega`` dependence
    (``docs/guide/waveguides/gnlse.md``). The run is an ordinary mode-averaged envelope
    propagation; there is no separate GNLSE solver.

    Units are SI throughout (``docs/dev/conventions.md`` section 1) and the spectral
    phase is in the optics convention, as is every spectrum the output stores
    (``docs/dev/conventions.md`` section 2).

    Its keywords are :meth:`~solprop.driver.simulation.Simulation.from_gnlse`'s,
    then the run controls of :meth:`~solprop.driver.simulation.Simulation.run`.

    Returns
    -------
    solprop.output.schema.Output
        The handler the run was written to, closed. A
        :class:`solprop.output.memory.MemoryOutput` stays readable; read an HDF5 file
        back with :func:`solprop.output.hdf5.open_file`.

    Raises
    ------
    DriverError
        If the fibre, the grid, the input, the noise model or the save planes are
        specified in a way this interface does not accept; the message names what is
        wrong.
    OutputError
        If a resume is asked for against a file whose configuration hash differs; the
        message names what changed.

    Examples
    --------
    A third-order soliton over a hundredth of its period, saved to memory:

    >>> import numpy as np
    >>> from solprop.driver import prop_gnlse
    >>> beta2, gamma, tau0 = -11.83e-27, 0.11, 100e-15
    >>> peak = 9 * abs(beta2) / (gamma * tau0**2)
    >>> handler = prop_gnlse(
    ...     gamma, 1.57e-3, [0.0, 0.0, beta2], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=2e-12,
    ...     width=tau0, peak_power=peak, shape="sech", shock=False,
    ...     n_save=3, rtol=1e-8, noise=None,
    ... )
    >>> handler.n_saved, handler["z"].tolist()
    (3, [0.0, 0.000785, 0.00157])

    The fibre is lossless and the window holds the soliton's tails clear of the
    collar, so without shot noise (part of which the collar would absorb) the energy
    statistic is the same at every plane:

    >>> energy = handler["stats/energy"][:, 0, 0]
    >>> bool(abs(energy[-1] / energy[0] - 1.0) < 1e-8)
    True
    """
    return _propagate(Simulation.from_gnlse, locals())


@_propagates(Simulation.from_capillary)
def prop_capillary(  # numpydoc ignore=PR01
    radius: float | ZProfile,
    length: float,
    gas: Gas | str | Mapping[str, float] | Mixture,
    pressure: Pressure | None = None,
    *,
    pressure_floor: float | None = None,
    vacuum_end: VacuumEnd | None = None,
    wavelength: float,
    wavelength_limits: tuple[float, float] | None = None,
    trange: float | None = None,
    grid: TimeGrid | None = None,
    envelope: bool = False,
    fwhm: float | None = None,
    width: float | None = None,
    shape: str = "gauss",
    energy: float | None = None,
    peak_power: float | None = None,
    phases: Sequence[float] | np.ndarray = (),
    pulses: Sequence[Pulse] | None = None,
    launch: ModalLaunch | None = None,
    modes: int | Sequence[Mode] = 1,
    mode_average: bool = False,
    lattice: TubeLattice | None = None,
    wall: CoherentWall | None = None,
    taper_coupling: bool = True,
    model: MarcatiliModel = "full",
    loss: LossSpec = True,
    cladding: GlassIndex | complex | None = None,
    susceptibility: GasSusceptibility | None = None,
    temperature: float | ZProfile | None = None,
    kerr: bool = True,
    thg: bool | None = None,
    plasma: bool | IonisationRate | Mapping[str, bool | IonisationRate] | None = None,
    preionisation: float | ZProfile | Mapping[str, float | ZProfile] | None = None,
    plasma_integration: IntegrationPolicy | None = None,
    raman: bool | Sequence[str] | Mapping[str, bool | MolecularOptions] | None = None,
    n_r: int | None = None,
    n_theta: int | None = None,
    radial_rule: RadialRule = "gauss_kronrod",
    ionisation_field_range: tuple[float, float] | None = None,
    absorber: Absorber | Literal[False] | None = None,
    absorber_scheme: AbsorberScheme | None = None,
    boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK,
    noise: NoiseSetting = "default",
    seed: int | None = None,
    rtol: float | None = None,
    atol: float = WAVEGUIDE_ATOL,
    dz_max: float | None = None,
    dz_init: float | None = None,
    schedule: FixedSchedule | None = None,
    integrator: Multistep | RungeKutta | LowStorageRungeKutta | None = None,
    interpolate_planes: bool = False,
    step_planes: int | None = None,
    save_planes: Sequence[float] | np.ndarray | None = None,
    n_save: int | None = None,
    stats: StatsSpec | None = None,
    polarisation_taper: bool = False,
    node_budget: int | None = None,
    backend: str = "auto",
    polarisation: str | float | None = None,
    materials: MaterialSelection | None = None,
    tdse: TDSEGasResponse | None = None,
    tdse_execution: TDSEExecution | None = None,
    output: Output | Path | str | None = None,
    snapshots: SnapshotController | None = None,
    resume: _ResumeChoice = False,
    progress: ProgressReporter | ProgressChoice | None = None,
    progress_period: float = DEFAULT_PROGRESS_PERIOD,
    tdse_archive: TDSEArchive | None = None,
) -> Output:
    """Propagate a pulse through a gas-filled hollow capillary.

    The mode set, the fixed transverse quadrature, the pressure and density paths, the
    Kerr and plasma responses, the input pulse and the numerics are all built from
    keywords, and the run goes into memory or into a file. It is
    :meth:`solprop.driver.simulation.Simulation.from_capillary` followed by
    :meth:`solprop.driver.simulation.Simulation.run`; ``docs/guide/setup/running.md``
    has the discussion.

    What differs from :func:`prop_gnlse` is the physics, not the machinery: a **field**
    grid rather than an envelope one, a set of modes on a shared transverse quadrature
    rather than one mode-averaged effective area (unless `mode_average` asks for that
    area), a gas whose **density varies along z** and carries both the dispersion and
    the nonlinearity, and a plasma response whose ionisation rate is tabulated once
    over the fields the run will reach.

    The step control is reached through `rtol`, `atol`, `dz_max` and `dz_init`, and
    the default step caps are relative to the capillary length rather than absolute.
    The path is scalar: `polarisation` and the vector route are refused by name.

    Its keywords are :meth:`~solprop.driver.simulation.Simulation.from_capillary`'s,
    then the run controls of :meth:`~solprop.driver.simulation.Simulation.run`.

    Returns
    -------
    solprop.output.schema.Output
        The handler the run wrote, closed. A
        :class:`~solprop.output.memory.MemoryOutput` stays readable; read a file back
        with :func:`solprop.output.hdf5.open_file`.

    Raises
    ------
    DriverError
        If a keyword is missing, inconsistent, or names something this path does not
        carry.
    OutputError
        If a resume is asked for against a file whose configuration hash differs
        (:meth:`solprop.driver.simulation.Simulation.run`).

    Examples
    --------
    A short helium capillary with a pressure gradient, two modes and no plasma, run
    into memory. The window a capillary needs is wider than the band any gas index is
    fitted over, so the run reports the extrapolation once per model and the example
    silences it:

    >>> import warnings
    >>> from solprop.driver import prop_capillary
    >>> from solprop.exceptions import MaterialRangeWarning
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", MaterialRangeWarning)
    ...     output = prop_capillary(
    ...         125e-6, 0.02, "He", (2.0, 1.0), wavelength=800e-9,
    ...         wavelength_limits=(300e-9, 3e-6), trange=100e-15,
    ...         fwhm=10e-15, energy=5e-6, modes=2, plasma=False,
    ...         n_r=16, n_save=2, rtol=1e-8,
    ...     )
    >>> output["field"].shape[1:]
    (2, 1, 193)

    The pressure the run saw is recorded beside the field, and it falls along the
    capillary:

    >>> pressure = output["stats/pressure"]
    >>> bool(pressure[-1] < pressure[0])
    True
    """
    return _propagate(Simulation.from_capillary, locals())


@_propagates(Simulation.from_waveguide)
def prop_waveguide(  # numpydoc ignore=PR01
    modes: Sequence[Mode],
    length: float,
    gas: Gas | str | Mapping[str, float] | Mixture | None = None,
    pressure: Pressure | None = None,
    *,
    provider: ModalGuide | None = None,
    channel: int | str | None = None,
    incident: SpatialPulse | None = None,
    coupling: CouplingModel | None = None,
    kerr_form: KerrForm = "real",
    observable_rtol: float | None = None,
    observable_atol: float | None = None,
    pressure_floor: float | None = None,
    vacuum_end: VacuumEnd | None = None,
    wavelength: float,
    wavelength_limits: tuple[float, float] | None = None,
    trange: float | None = None,
    grid: TimeGrid | None = None,
    envelope: bool = False,
    fwhm: float | None = None,
    width: float | None = None,
    shape: str = "gauss",
    energy: float | None = None,
    peak_power: float | None = None,
    phases: Sequence[float] | np.ndarray = (),
    pulses: Sequence[Pulse] | None = None,
    launch: ModalLaunch | None = None,
    susceptibility: GasSusceptibility | None = None,
    temperature: float | ZProfile | None = None,
    kerr: bool = True,
    thg: bool | None = None,
    plasma: bool | IonisationRate | Mapping[str, bool | IonisationRate] | None = None,
    preionisation: float | ZProfile | Mapping[str, float | ZProfile] | None = None,
    plasma_integration: IntegrationPolicy | None = None,
    raman: bool | Sequence[str] | Mapping[str, bool | MolecularOptions] | None = None,
    mode_average: bool = False,
    n_r: int | None = None,
    n_theta: int | None = None,
    radial_rule: RadialRule = "gauss_kronrod",
    ionisation_field_range: tuple[float, float] | None = None,
    absorber: Absorber | Literal[False] | None = None,
    absorber_scheme: AbsorberScheme | None = None,
    boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK,
    noise: NoiseSetting = "default",
    seed: int | None = None,
    rtol: float | None = None,
    atol: float = WAVEGUIDE_ATOL,
    dz_max: float | None = None,
    dz_init: float | None = None,
    schedule: FixedSchedule | None = None,
    integrator: Multistep | RungeKutta | LowStorageRungeKutta | None = None,
    interpolate_planes: bool = False,
    step_planes: int | None = None,
    save_planes: Sequence[float] | np.ndarray | None = None,
    n_save: int | None = None,
    stats: StatsSpec | None = None,
    polarisation_taper: bool = False,
    node_budget: int | None = None,
    backend: str = "auto",
    polarisation: str | float | None = None,
    taper_coupling: bool = True,
    materials: MaterialSelection | None = None,
    output: Output | Path | str | None = None,
    snapshots: SnapshotController | None = None,
    resume: _ResumeChoice = False,
    progress: ProgressReporter | ProgressChoice | None = None,
    progress_period: float = DEFAULT_PROGRESS_PERIOD,
) -> Output:
    """Propagate a pulse through a gas-filled waveguide mode set, or through a guide.

    Build with :meth:`~solprop.driver.simulation.Simulation.from_waveguide` and run
    to the requested output. Without `provider` this is the gas-filled route:
    rectangular polarisation partners carry distinct operator rows, a hollow planar
    set (:class:`~solprop.modes.slab.SlabMode`) runs **per unit width** -- `energy`
    in J/m, `peak_power` in W/m, and the recorded powers and energies likewise --
    and step-index fibres use ``prop_gnlse(fibre=...)`` instead. With `provider` it
    is the **general modal route** (``docs/guide/waveguides/general-modal.md``), on a
    field grid, on every backend (Kokkos on CPUs and GPUs), with HDF5 output.

    Its keywords are :meth:`~solprop.driver.simulation.Simulation.from_waveguide`'s,
    then the run controls of :meth:`~solprop.driver.simulation.Simulation.run`.

    Returns
    -------
    solprop.output.schema.Output
        Saved fields, metadata, statistics and step history.
    """
    return _propagate(Simulation.from_waveguide, locals())
