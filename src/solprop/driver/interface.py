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

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float

from solprop.boundaries import Absorber
from solprop.coupling.models import CouplingModel
from solprop.driver.simulation import (
    DEFAULT_ATOL,
    WAVEGUIDE_ATOL,
    NoiseSetting,
    Simulation,
    SimulationSettings,
)
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
from solprop.pulses import PlanarPulse, Pulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.masks import AbsorberScheme
from solprop.translate.modal import DEFAULT_NODE_BUDGET
from solprop.translate.precision import PrecisionName
from solprop.translate.responses import KerrForm

__all__ = [
    "prop_capillary",
    "prop_free",
    "prop_gnlse",
    "prop_planar",
    "prop_radial",
    "prop_waveguide",
]

logger = logging.getLogger(__name__)


def prop_radial(  # noqa: PLR0913 - the simulation interface and output controls
    grid: TimeGrid,
    space: RadialGrid,
    medium: SpatialMedium,
    responses: Sequence[object],
    initial: SpatialPulse | np.ndarray,
    *,
    settings: SimulationSettings,
    frame: Frame | None = None,
    noise: NoiseSetting = "default",
    seed: int | None = None,
    output: Output | Path | str | None = None,
    snapshots: SnapshotController | None = None,
    resume: bool | Literal["interpolated"] = False,
    progress: ProgressReporter | ProgressChoice | None = None,
    progress_period: float = DEFAULT_PROGRESS_PERIOD,
    materials: MaterialSelection | None = None,
) -> Output:
    """Propagate a radial beam and return its saved output.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Field or envelope temporal grid, in SI units.
    space : solprop.grid.RadialGrid
        Radial nodes and outer radius in m.
    medium : solprop.spatial.SpatialMedium
        Bulk material, optionally with a varying gas density along z.
    responses : collections.abc.Sequence
        Existing nonlinear response objects; an empty sequence gives linear flow.
    initial : solprop.pulses.SpatialPulse or numpy.ndarray
        Beam specification or internal Hankel spectrum in V m, shape
        ``(n_r, n_pol, n_omega)``.
    settings : solprop.driver.simulation.SimulationSettings
        Step tolerances, save planes, statistics and backend.
    frame : solprop.linear.Frame or None, optional
        Moving frame; None uses the responses' THG choice.
    noise : {'default', None, 'input', 'modified'}, optional
        One photon per spatial column, polarisation and active frequency bin.
        ``"default"``, the default, resolves to ``"input"`` wherever the run can
        carry noise, and records the resolution (SEP 37).
    seed : int or None, optional
        Noise seed, required to have a noise model when given.
    output : solprop.output.schema.Output or pathlib.Path or str or None, optional
        Handler, HDF5 path, or None for an in-memory result.
    snapshots : solprop.output.SnapshotController or None, optional
        Immutable local checkpoints at committed accepted steps; requires HDF5,
        ``step_planes=1``, all fields and non-interpolated observations.
    resume : bool or {'interpolated'}, optional
        Continue an existing matching HDF5 file.
        ``"interpolated"`` also continues a dense run from a plane interpolated
        inside a step (:meth:`solprop.driver.simulation.Simulation.run`).
    progress : ProgressReporter, {'auto', 'bar', 'log', 'none'} or None, optional
        Where progress goes; ``None`` takes the process default: a bar on a terminal
        or in a notebook, log lines otherwise (:func:`solprop.driver.run.run_problem`).
    progress_period : float, optional
        Seconds between the line reporter's periodic lines, nonnegative; a bar has
        its own redraw period.
    materials : solprop.physdata.selection.MaterialSelection or None, optional
        Take the run's built-in materials from optmat under this selection: the medium's
        built-in gas resolves through it (see
        :meth:`~solprop.driver.simulation.Simulation.from_radial`). Supplied objects are
        taken as they are, and the run records its material manifest (output schema 8).
        ``None`` (the default) is ``MaterialSelection()``, optmat's recommended models.

    Returns
    -------
    solprop.output.schema.Output
        Saved fields, metadata, statistics and step history.
    """
    simulation = Simulation.from_radial(
        grid,
        space,
        medium,
        responses,
        initial,
        settings=settings,
        frame=frame,
        noise=noise,
        seed=seed,
        materials=materials,
    )
    return simulation.run(
        output,
        snapshots=snapshots,
        resume=resume,
        progress=progress,
        progress_period=progress_period,
    )


def prop_free(  # noqa: PLR0913 - the simulation interface and output controls
    grid: TimeGrid,
    space: FreeGrid,
    medium: SpatialMedium,
    responses: Sequence[object],
    initial: SpatialPulse | np.ndarray,
    *,
    settings: SimulationSettings,
    frame: Frame | None = None,
    noise: NoiseSetting = "default",
    seed: int | None = None,
    output: Output | Path | str | None = None,
    snapshots: SnapshotController | None = None,
    resume: bool | Literal["interpolated"] = False,
    progress: ProgressReporter | ProgressChoice | None = None,
    progress_period: float = DEFAULT_PROGRESS_PERIOD,
    materials: MaterialSelection | None = None,
) -> Output:
    """Propagate a Cartesian beam and return its saved output.

    A :class:`solprop.grid.LineGrid` propagates a beam uniform in y: the output's
    energies and powers are then per metre of y (J/m, W/m), and processing and
    plotting label them so.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Field or envelope temporal grid, in SI units.
    space : solprop.grid.FreeGrid
        Cartesian axes and half-widths in m, or a :class:`solprop.grid.LineGrid`.
    medium : solprop.spatial.SpatialMedium
        Constant bulk material; varying profiles are refused.
    responses : collections.abc.Sequence
        Existing nonlinear response objects; an empty sequence gives linear flow.
    initial : solprop.pulses.SpatialPulse or numpy.ndarray
        Beam specification or internal Cartesian DFT spectrum in V/m, shape
        ``(n_x*n_y, n_pol, n_omega)``.
    settings : solprop.driver.simulation.SimulationSettings
        Step tolerances, save planes, statistics and backend.
    frame : solprop.linear.Frame or None, optional
        Moving frame; None uses the responses' THG choice.
    noise : {'default', None, 'input', 'modified'}, optional
        One photon per spatial column, polarisation and active frequency bin.
        ``"default"``, the default, resolves to ``"input"`` wherever the run can
        carry noise, and records the resolution (SEP 37).
    seed : int or None, optional
        Noise seed, required to have a noise model when given.
    output : solprop.output.schema.Output or pathlib.Path or str or None, optional
        Handler, HDF5 path, or None for an in-memory result.
    snapshots : solprop.output.SnapshotController or None, optional
        Immutable local checkpoints at committed accepted steps; requires HDF5,
        ``step_planes=1``, all fields and non-interpolated observations.
    resume : bool or {'interpolated'}, optional
        Continue an existing matching HDF5 file.
        ``"interpolated"`` also continues a dense run from a plane interpolated
        inside a step (:meth:`solprop.driver.simulation.Simulation.run`).
    progress : ProgressReporter, {'auto', 'bar', 'log', 'none'} or None, optional
        Where progress goes; ``None`` takes the process default: a bar on a terminal
        or in a notebook, log lines otherwise (:func:`solprop.driver.run.run_problem`).
    progress_period : float, optional
        Seconds between the line reporter's periodic lines, nonnegative; a bar has
        its own redraw period.
    materials : solprop.physdata.selection.MaterialSelection or None, optional
        Take the run's built-in materials from optmat under this selection: the medium's
        built-in gas resolves through it (see
        :meth:`~solprop.driver.simulation.Simulation.from_free`). Supplied objects are
        taken as they are, and the run records its material manifest (output schema 8).
        ``None`` (the default) is ``MaterialSelection()``, optmat's recommended models.

    Returns
    -------
    solprop.output.schema.Output
        Saved fields, metadata, statistics and step history.
    """
    simulation = Simulation.from_free(
        grid,
        space,
        medium,
        responses,
        initial,
        settings=settings,
        frame=frame,
        noise=noise,
        seed=seed,
        materials=materials,
    )
    return simulation.run(
        output,
        snapshots=snapshots,
        resume=resume,
        progress=progress,
        progress_period=progress_period,
    )


def prop_planar(  # noqa: PLR0913 - the simulation interface and output controls
    grid: TimeGrid,
    space: PlanarGrid,
    responses: Sequence[object],
    initial: PlanarPulse | Pulse | np.ndarray,
    *,
    settings: SimulationSettings,
    frame: Frame | None = None,
    output: Output | Path | str | None = None,
    snapshots: SnapshotController | None = None,
    resume: bool | Literal["interpolated"] = False,
    progress: ProgressReporter | ProgressChoice | None = None,
    progress_period: float = DEFAULT_PROGRESS_PERIOD,
    materials: MaterialSelection | None = None,
) -> Output:
    """Propagate a beam in a hollow planar guide, resolved in y, and return its output.

    The hybrid planar transform: the slab modes of `space` in x and a periodic, collared
    y axis (``docs/guide/waveguides/planar.md``, "Resolving y"). Diffraction uses the
    represented angular-spectrum operator. The run integrates over y, so its energies
    are in J and its powers in W. Constant gap and constant fill only. ``numpy_ref``,
    ``numba_cpu`` and native CPU execute the geometry.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Field or envelope temporal grid, in SI units.
    space : solprop.modes.slab.PlanarGrid
        The slab modes, their fill and the y axis.
    responses : collections.abc.Sequence
        Existing nonlinear response objects; an empty sequence gives linear flow.
    initial : solprop.pulses.PlanarPulse or numpy.ndarray
        The astigmatic launch, or an internal ``(p, k_y)`` spectrum in sqrt(W/m),
        shape ``(n_p*n_y, 1, n_omega)``.
    settings : solprop.driver.simulation.SimulationSettings
        Step tolerances, save planes, statistics and backend.
    frame : solprop.linear.Frame or None, optional
        Moving frame; None uses the responses' THG choice and the first mode.
    output : solprop.output.schema.Output or pathlib.Path or str or None, optional
        Handler, HDF5 path, or None for an in-memory result.
    snapshots : solprop.output.SnapshotController or None, optional
        Immutable local checkpoints at committed accepted steps; requires HDF5,
        ``step_planes=1``, all fields and non-interpolated observations.
    resume : bool or {'interpolated'}, optional
        Continue an existing matching HDF5 file.
        ``"interpolated"`` also continues a dense run from a plane interpolated
        inside a step (:meth:`solprop.driver.simulation.Simulation.run`).
    progress : ProgressReporter, {'auto', 'bar', 'log', 'none'} or None, optional
        Where progress goes; ``None`` takes the process default.
    progress_period : float, optional
        Seconds between the line reporter's periodic lines, nonnegative.
    materials : solprop.physdata.selection.MaterialSelection or None, optional
        Take the run's built-in materials from optmat under this selection: a
        built-in gas fill takes the selection's single-species susceptibility and a slab
        without a declared cladding optmat's recommended fused silica (see
        :meth:`~solprop.driver.simulation.Simulation.from_planar`). Supplied objects are
        taken as they are, and the run records its material manifest (output schema 8).
        ``None`` (the default) is ``MaterialSelection()``, optmat's recommended models.

    Returns
    -------
    solprop.output.schema.Output
        Saved fields, metadata, statistics and step history.
    """
    simulation = Simulation.from_planar(
        grid,
        space,
        responses,
        initial,
        settings=settings,
        frame=frame,
        materials=materials,
    )
    return simulation.run(
        output,
        snapshots=snapshots,
        resume=resume,
        progress=progress,
        progress_period=progress_period,
    )


def prop_gnlse(  # noqa: PLR0913 - the interface's keyword surface
    gamma: float | None,
    length: float,
    betas: Sequence[float] | np.ndarray | None = None,
    *,
    lambda0: float,
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
    pulseshape: str = "gauss",
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
    step_planes: int = 0,
    save_planes: Sequence[float] | np.ndarray | None = None,
    n_save: int | None = None,
    stats: StatsSpec | None = None,
    polarisation_taper: bool = False,
    output: Output | Path | str | None = None,
    snapshots: SnapshotController | None = None,
    backend: str = "auto",
    precision: PrecisionName = "double",
    resume: bool | Literal["interpolated"] = False,
    progress: ProgressReporter | ProgressChoice | None = None,
    progress_period: float = DEFAULT_PROGRESS_PERIOD,
    materials: MaterialSelection | None = None,
) -> Output:
    """Propagate a pulse through a GNLSE fibre and return the output it was written to.

    The fibre is its dispersion and its nonlinear coefficient: ``beta(omega)`` as Taylor
    coefficients about `lambda0` or as data, ``gamma`` in 1/(W m), and a power loss in
    dB/m. The mapping onto solprop's physics is
    :func:`solprop.nonlinear.gnlse.problem_from` -- ``n2 = gamma A_eff / k_0``,
    ``chi3 = (4/3) n2 eps_0 c``, an effective index of one in the normalisation
    prefactor and the shock term as that prefactor's ``omega`` dependence
    (``docs/guide/waveguides/gnlse.md``). The run is an ordinary mode-averaged envelope
    propagation; there is no separate GNLSE solver.

    Units are SI throughout (``docs/dev/conventions.md`` section 1) and the spectral
    phase is in the optics convention, as is every spectrum the output stores
    (``docs/dev/conventions.md`` section 2).

    Parameters
    ----------
    gamma : float or None
        Nonlinear coefficient in 1/(W m), positive for a self-focusing medium.
    length : float
        Fibre length, m, positive.
    betas : array_like, optional
        Taylor coefficients of ``beta`` about `lambda0`, ``beta_0, beta_1, ...``, in
        s^n/m. ``beta_0`` and ``beta_1`` only fix the frame and change no physics. Give
        exactly one of this, `beta` or `fibre`.
    lambda0 : float
        Reference wavelength, m, positive: the ``gamma`` mapping's reference, the centre
        of the Taylor expansion, the grid's reference and the pulse's centre wavelength.
    wavelength_limits : tuple of float, optional
        Shortest and longest wavelength of the frequency window, m, increasing and
        positive. Required unless `grid` is given.
    trange : float, optional
        Extent of the time window, s. Required unless `grid` is given.
    grid : solprop.grid.TimeGrid, optional
        An envelope grid to propagate on instead of building one from `length`,
        `lambda0`, `wavelength_limits` and `trange`. This is the way to choose the
        sample spacing, the oversampling factor or the transform size. A field grid is
        refused: the GNLSE mapping is the envelope Kerr response.
    beta : tuple or callable, optional
        The propagation constant as data instead of a series: a pair of arrays
        ``(omega, beta)`` in rad/s and 1/m with strictly increasing frequencies, or a
        callable of ``omega`` which is sampled on the grid's active band. Either is
        interpolated by a cubic spline and differentiated exactly in the spline basis
        (:class:`solprop.nonlinear.gnlse.TabulatedDispersion`); evaluation outside the
        data raises rather than extrapolating.
    fibre : solprop.modes.protocol.Mode or solprop.modes.tabulated.IndexTable, optional
        Whole dispersion and loss instead of `betas` or `beta`. A mode goes
        through :meth:`solprop.nonlinear.gnlse.SpecifiedMode.from_mode`, so one
        whose geometry moves along z -- a tapered mode, a moving guide's provider
        channel, a ``SpecifiedMode`` with dispersion and area paths -- hands over
        its dispersion and its area along z; only the fixed-step scaffold requires
        a z-constant fibre. Its fill, if present, is ignored as a nonlinear
        material because the GNLSE uses gamma or n2. The mode supplies only
        dispersion, loss and area; use prop_capillary for its material responses.
        A table's area is evaluated at `lambda0`, and so is a provider channel's,
        by a fresh solve there (along the path of a moving guide); without an area
        column a table requires `gamma` and uses the 1 m^2 convention.
    n2 : float or str, optional
        Nonlinear index in m^2/W, requiring `fibre` and ``gamma=None``.
        Exactly one of `gamma` and `n2` is required. The mapping is
        ``gamma = n2 omega0 / (c A_eff)``, with unit prefactor index. Under
        `materials`, ``"recommended"`` takes optmat's silica bundle's total ``n2``
        (``kerr/SiO2``) at `lambda0`.
    fwhm : float, optional
        Intensity full width at half maximum of the input pulse, s. Give this or, for a
        ``sech`` pulse, `width`.
    width : float, optional
        Natural width of a ``sech**2`` input, s, which is
        ``fwhm / (2 ln(1 + sqrt(2)))``. Only for ``pulseshape="sech"``.
    pulseshape : {'gauss', 'sech'}, optional
        Shape of the input pulse.
    energy : float, optional
        Pulse energy, J. Give this or `peak_power`.
    peak_power : float, optional
        Peak power, W, realised after the spectral phase is applied.
    phases : array_like, optional
        Optics-convention spectral phase as Taylor coefficients about `lambda0`:
        ``[CEP (rad), group delay (s), GDD (s**2), TOD (s**3), ...]``.
    pulses : sequence of solprop.pulses.Pulse or None, optional
        Several pulses launched together, for two-colour and multi-colour
        inputs: each a :mod:`solprop.pulses` object with its own wavelength,
        duration, energy or peak power, phase and delay; the input is the sum of
        their fields (:class:`~solprop.pulses.PulseSum`). Replaces the
        single-pulse keywords `fwhm`, `width`, `pulseshape`, `energy`,
        `peak_power` and `phases`, which must then be left unset; `lambda0` still
        sets the grid's reference wavelength.
    loss_db_per_m : float or None, optional
        Omitted means zero loss on the coefficient routes. Refused beside
        `fibre`, which carries its own loss. Power loss in dB/m; negative values
        describe gain. Converted by
        :func:`solprop.nonlinear.gnlse.attenuation_from_db`.
    shock : bool, optional
        Keep the ``omega`` dependence of the normalisation prefactor, which is the
        optical shock term. Default ``True``.
    raman : bool, optional
        Whether a fused-silica Raman response takes `raman_fraction` of ``gamma``, the
        rest staying with the Kerr term
        (:func:`solprop.nonlinear.gnlse.raman_susceptibility`). Default ``False``, which
        gives the whole of ``gamma`` to the Kerr term.
    raman_fraction : float or None, optional
        Share of ``gamma`` held back from the Kerr term, between 0 and 1. Omitted, it
        is the silica bundle's own ``f_R`` with `raman` true -- 0.2188 for optmat's
        default (SEP 3, D4) -- and 0 with it false. Given, it is held back whatever
        `raman` is, so a pair of runs with the same fraction that switches only
        `raman` isolates the delayed response.
    noise : {'default', None, 'input', 'modified'}, optional
        Which shot-noise model to carry: ``'input'`` adds one photon per bin to the
        input field once, ``'modified'`` keeps the realisation apart and injects it at
        every stage, advanced by the unitary part of the linear operator
        (``docs/guide/media/noise.md``).
        ``"default"``, the default, resolves to ``"input"`` wherever the run can
        carry noise, and records the resolution (SEP 37).
    seed : int, optional
        Entropy of the generator the realisation is drawn from. Recorded in the output,
        so a run is reproducible; ``None`` draws from the operating system and records
        what it drew.
    rtol : float or None, optional
        ``None`` takes the default below, or the single-precision floor.
        Relative tolerance of the step controller, dimensionless and positive. The map
        from this local tolerance to the global error at the output is problem dependent
        and misses by one to three orders either way, so calibrate rather than assume
        (``docs/dev/stepper.md`` section 3).
    atol : float, optional
        Absolute tolerance, in the square root of watts, non-negative.
    dz_max : float, optional
        Cap on the controller's step proposal, m; ``None`` uses half the length.
    dz_init : float, optional
        First step, m; ``None`` uses a ten-thousandth of the length.
    schedule : solprop.stepper.schedule.FixedSchedule, optional
        A fixed schedule to replay instead of stepping adaptively, which makes a run
        bit-identical on re-running (``docs/dev/stepper.md`` section 8). Its edges
        must contain every save plane, which
        :meth:`solprop.stepper.schedule.FixedSchedule.uniform` guarantees.
    integrator : Multistep, RungeKutta or LowStorageRungeKutta, optional
        The method that steps the run (``docs/guide/numerics/choosing-a-stepper.md``).
        ``None``, the default, takes ``RungeKutta()``, the interaction-picture DOPRI5
        march, which SEP 37 measured fastest at matched accuracy on this interface
        (with noise all the more). ``Multistep()`` is the low-call exponential Adams
        method, one response evaluation per step, for an expensive response.
        ``LowStorageRungeKutta()`` replays `schedule` in three registers (SEP 24).
    interpolate_planes : bool, optional
        Observe the save planes inside accepted steps instead of landing on them
        (SEP 14): through the Adams march's own polynomial, or DOPRI5's continuous
        extension with ``RungeKutta()``. Default ``False``.
    step_planes : int, optional
        Also save the end of every `step_planes`-th accepted step, the
        march's own positions, beside the requested planes (SEP 14);
        ``0``, the default, saves the requested planes only.
    save_planes : array_like, optional
        Positions along the fibre to save at, m, increasing and positive; the plane at
        ``z = 0`` is saved as well. Give this or `n_save`.
    n_save : int, optional
        Number of planes to store, counting the one at ``z = 0``, spread evenly over the
        fibre. Default 201.
    stats : solprop.stats.StatsSpec, optional
        Which statistics to record at every plane; ``None`` records
        :func:`solprop.stats.default_spec` of a mode-averaged run. A ``periodic``
        subset is also recorded between the planes.
    polarisation_taper : bool, optional
        Whether the spectral collar tapers the nonlinear polarisation as well as
        absorbing the field through the linear operator. Default ``False``: the taper
        costs a systematic error of the size of the field's content in the collar, and
        the rate absorber acts either way (``docs/guide/numerics/boundaries.md``).
    output : solprop.output.schema.Output or pathlib.Path or str or None, optional
        Where the run goes: a handler, a path to stream an HDF5 file to, or ``None`` for
        a :class:`solprop.output.memory.MemoryOutput`.
    snapshots : solprop.output.SnapshotController or None, optional
        Immutable local checkpoints at committed accepted steps; requires HDF5,
        ``step_planes=1``, all fields and non-interpolated observations.
    backend : str, optional
        Registry name, or ``"auto"`` (the default) to prefer ``native`` if installed
        and otherwise use Numba CPU. Import or construction errors do not fall back.
        :func:`solprop.backends.registry.available` lists what can be used.
    precision : {'double', 'single'}, optional
        The arithmetic the run is held in (SEP 33,
        ``docs/guide/performance/single-precision.md``). ``"double"``, the default, is
        binary64 throughout. ``"single"`` holds the state and the source evaluation in
        binary32: its default `rtol` is the floor
        :data:`~solprop.translate.precision.SINGLE_RTOL_FLOOR`, below which an explicit
        value is refused; its default integrator is ``RungeKutta()``, and the Adams
        method, noise models and every backend but ``numpy_ref`` are refused.
    resume : bool or {'interpolated'}, optional
        Continue the file at `output` if it holds this same run, instead of replacing
        it. Only a path can be resumed.
        ``"interpolated"`` also continues a dense run from a plane interpolated
        inside a step (:meth:`solprop.driver.simulation.Simulation.run`).
    progress : ProgressReporter, {'auto', 'bar', 'log', 'none'} or None, optional
        Where progress goes; ``None`` takes the process default: a bar on a terminal
        or in a notebook, log lines otherwise (:func:`solprop.driver.run.run_problem`).
    progress_period : float, optional
        Seconds between the default reporter's periodic lines, s; ``0`` leaves only the
        plane and closing lines. Ignored when `progress` is given.
    materials : solprop.physdata.selection.MaterialSelection or None, optional
        Take fused silica's nonlinear response from optmat's silica bundle under this
        selection: with `raman` on, an omitted fraction is the bundle's ``f_R`` and the
        response its sampled shape, and ``n2="recommended"`` is its total ``n2``. The
        fibre's dispersion and a given ``gamma`` or ``n2`` are the caller's. The run
        records its material manifest (output schema 8), with the held-back fraction
        and its origin. ``None`` (the default) is ``MaterialSelection()``, optmat's
        recommended models.

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
        If a file is there and was written for a different run; the message names the
        fields that changed.

    Examples
    --------
    A third-order soliton over a hundredth of its period, saved to memory:

    >>> import numpy as np
    >>> from solprop.driver import prop_gnlse
    >>> beta2, gamma, tau0 = -11.83e-27, 0.11, 100e-15
    >>> peak = 9 * abs(beta2) / (gamma * tau0**2)
    >>> handler = prop_gnlse(
    ...     gamma, 1.57e-3, [0.0, 0.0, beta2], lambda0=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=tau0, peak_power=peak, pulseshape="sech", shock=False,
    ...     n_save=3, rtol=1e-8,
    ... )
    >>> handler.n_saved, handler["z"].tolist()
    (3, [0.0, 0.000785, 0.00157])

    The fibre is lossless, so the energy statistic is the same at every plane:

    >>> energy = handler["stats/energy"][:, 0, 0]
    >>> bool(abs(energy[-1] / energy[0] - 1.0) < 1e-9)
    True
    """
    simulation = Simulation.from_gnlse(
        gamma,
        length,
        betas,
        lambda0=lambda0,
        wavelength_limits=wavelength_limits,
        trange=trange,
        grid=grid,
        beta=beta,
        fibre=fibre,
        n2=n2,
        fwhm=fwhm,
        width=width,
        pulseshape=pulseshape,
        energy=energy,
        peak_power=peak_power,
        phases=phases,
        pulses=pulses,
        loss_db_per_m=loss_db_per_m,
        shock=shock,
        raman=raman,
        raman_fraction=raman_fraction,
        noise=noise,
        seed=seed,
        rtol=rtol,
        atol=atol,
        dz_max=dz_max,
        dz_init=dz_init,
        schedule=schedule,
        integrator=integrator,
        interpolate_planes=interpolate_planes,
        step_planes=step_planes,
        save_planes=save_planes,
        n_save=n_save,
        stats=stats,
        polarisation_taper=polarisation_taper,
        backend=backend,
        precision=precision,
        materials=materials,
    )
    logger.debug(
        "prop_gnlse: %.6g m of fibre, gamma = %.6g 1/(W m), %d save planes, %r backend",
        length,
        gamma,
        len(simulation.settings.save_planes),
        simulation.settings.backend,
    )
    return simulation.run(
        output,
        snapshots=snapshots,
        progress=progress,
        progress_period=progress_period,
        resume=resume,
    )


def prop_capillary(  # noqa: PLR0913 - the interface's keyword surface
    radius: float | ZProfile,
    length: float,
    gas: Gas | str | Mapping[str, float] | Mixture,
    pressure: Pressure | None = None,
    *,
    pressure_floor: float | None = None,
    vacuum_end: VacuumEnd | None = None,
    lambda0: float,
    wavelength_limits: tuple[float, float] | None = None,
    trange: float | None = None,
    grid: TimeGrid | None = None,
    envelope: bool = False,
    fwhm: float | None = None,
    width: float | None = None,
    pulseshape: str = "gauss",
    energy: float | None = None,
    peak_power: float | None = None,
    phases: Sequence[float] | np.ndarray = (),
    pulses: Sequence[Pulse] | None = None,
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
    absorber: Absorber | None = None,
    absorber_scheme: AbsorberScheme | None = None,
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
    node_budget: int = DEFAULT_NODE_BUDGET,
    backend: str = "auto",
    precision: PrecisionName = "double",
    polarisation: str | float | None = None,
    output: Output | Path | str | None = None,
    snapshots: SnapshotController | None = None,
    resume: bool | Literal["interpolated"] = False,
    progress: ProgressReporter | ProgressChoice | None = None,
    progress_period: float = DEFAULT_PROGRESS_PERIOD,
    materials: MaterialSelection | None = None,
    tdse: TDSEGasResponse | None = None,
    tdse_execution: TDSEExecution | None = None,
    tdse_archive: TDSEArchive | None = None,
) -> Output:
    """Propagate a pulse through a gas-filled hollow capillary.

    The mode set, the fixed transverse quadrature, the pressure and density paths, the
    Kerr and plasma responses, the input pulse and the numerics are all built from
    keywords, and the run goes into memory or into a file. It is
    :meth:`solprop.driver.simulation.Simulation.from_capillary` followed by
    :meth:`solprop.driver.simulation.Simulation.run`, and every keyword up to `output`
    is that constructor's; see it for their meaning and
    ``docs/guide/setup/running.md`` for the discussion.

    What differs from :func:`prop_gnlse` is the physics, not the machinery: a **field**
    grid rather than an envelope one, a set of modes on a shared transverse quadrature
    rather than one mode-averaged effective area (unless `mode_average` asks for that
    area), a gas whose **density varies along z** and carries both the dispersion and
    the nonlinearity, and a plasma response whose ionisation rate is tabulated once
    over the fields the run will reach.

    The step control is reached through `rtol`, `atol`, `dz_max` and `dz_init`, and
    the default step caps are relative to the capillary length rather than absolute.
    The path is scalar: `polarisation` and the vector route are refused by name.

    Parameters
    ----------
    radius : float or solprop.profiles.ZProfile
        Core radius, m, constant or along z.
    length : float
        Capillary length, m.
    gas : Gas, str, Mapping or Mixture
        Built-in gas, mole fractions keyed by species name, or ordered populations
        carrying their own density paths. A mapping uses total pressure.
    pressure : float, sequence, ZProfile or None, optional
        None uses one bar for a gas or fractions. Omit pressure beside a Mixture;
        every explicit pressure is refused because it already carries its paths.
        Fill pressure in **bar**: a number for a constant fill, a pair
        ``(p_in, p_out)`` for the steady-flow gradient of
        :meth:`solprop.profiles.PressureGradient.two_point`, or ``(z, p)`` with the
        positions and the pressures there for
        :meth:`solprop.profiles.PressureGradient.from_stages`.
    pressure_floor : float or None, optional
        Lowest pressure the fill is allowed to reach, bar; ``None`` takes
        :data:`solprop.profiles.DEFAULT_PRESSURE_FLOOR_FRACTION` of the highest
        stage pressure. With ``vacuum_end="floor"``, pass ``0.0`` for a true vacuum
        end, which warns, because ``dp/dz`` diverges there.
    vacuum_end : {'choked', 'floor'} or None, optional
        How a pressure gradient's outer end below its choked pressure is treated.
        ``None`` (the default) means ``"choked"`` for a circular hollow guide: the fill
        follows the bore and an end exhausting into vacuum takes the choked pressure
        of the flow (:func:`solprop.gasflow.steady_fill`). ``"floor"`` concerns the
        ends only: it keeps the declared end pressures, for a comparison with a
        model that imposes them, while a taper's fill still follows the bore. See
        :meth:`~solprop.driver.simulation.Simulation.from_capillary`.
    lambda0 : float
        Reference wavelength, m, and the pulse's centre wavelength.
    wavelength_limits : tuple of float or None, optional
        Shortest and longest wavelength of the frequency window, m. Required unless
        `grid` is given.
    trange : float or None, optional
        Requested width of the time window, s. Required unless `grid` is given.
    grid : solprop.grid.TimeGrid or None, optional
        A grid of one's own, instead of the three keywords above; the only way to
        reach ``dt``, ``ffac`` or the transform size.
    envelope : bool, optional
        Propagate an envelope. Plasma is off by default; explicit ``plasma=True``
        selects cycle averaging of instantaneous rates. Plasma forbids THG.
    fwhm : float or None, optional
        Pulse duration at half maximum, s.
    width : float or None, optional
        Natural width of a sech pulse, s; give this or `fwhm`.
    pulseshape : {'gauss', 'sech'}, optional
        Transform-limited shape of the input.
    energy : float or None, optional
        Pulse energy, J. Give this or `peak_power`.
    peak_power : float or None, optional
        Peak power, W, realised after the spectral phases.
    phases : sequence of float, optional
        Taylor spectral phases about `lambda0`, optics convention: CEP in rad,
        group delay in s, GDD in s^2, and so on.
    pulses : sequence of solprop.pulses.Pulse or None, optional
        Several pulses launched together, for two-colour and multi-colour
        inputs: each a :mod:`solprop.pulses` object with its own wavelength,
        duration, energy or peak power, phase and delay; the input is the sum of
        their fields (:class:`~solprop.pulses.PulseSum`). Replaces the
        single-pulse keywords `fwhm`, `width`, `pulseshape`, `energy`,
        `peak_power` and `phases`, which must then be left unset; `lambda0` still
        sets the grid's reference wavelength.
    modes : int or sequence of Mode, optional
        How many HE_1m modes to propagate, or the mode set itself. A set is checked for
        orthonormality. Supplied sets may include TabulatedHollowMode or TabulatedMode
        with their own tables. A supplied set carries its own geometry: `radius` must
        agree with every mode's core radius along the run, or the call is refused by
        name.
    mode_average : bool, optional
        Propagate the one mode **mode-averaged**, as a uniform disc of its effective
        area, instead of resolving it on the transverse rule, as Luna.jl's default
        does (``docs/guide/waveguides/mode-averaging.md``). The plasma then sees the
        area-mean field, below the on-axis one (``HE_11``'s axis intensity is 1.768
        times ``P/A_eff``), and ``peak_intensity`` stays on the axis. ``False``, the
        default, resolves the mode. Refused beside more than one mode and beside
        `n_r`, `n_theta` or a `radial_rule`.
    lattice : TubeLattice or None, optional
        Tube geometry for an integer mode count, selecting Vincetti modes.
        Refused with supplied modes, which carry their own geometry, and with
        ``model="reduced"``. None selects ordinary capillary modes.
    wall : solprop.modes.CoherentWall or None, optional
        Declare the scalar **coherent** wall block: the outgoing Robin boundary
        condition condensed onto the retained basis, which couples
        every ``HE_1m`` channel to every other with a definite phase instead of
        giving each its own Marcatili attenuation. It needs the contiguous set
        ``HE_11`` to ``HE_1N`` -- or, with ``impedance="te_tm"``, one azimuthal
        order -- on a constant or a varying guide, a scalar run and no noise model;
        all three backends execute it (``method="cf4"`` on ``numpy_ref`` alone).
        ``None``, the default, carries no wall: the modes keep their own Marcatili
        loss, and a multimode taper its taper coupling (below).
    taper_coupling : bool, optional
        On a multimode capillary whose radius varies along z, carry the **taper
        coupling** -- the connection ``<psi_m|d_z psi_n>`` that moves power between the
        ``HE_1m`` channels as the bore changes -- lossless or not. ``True``, the
        default; ``False`` is the explicit opt-out to the **adiabatic** route, on which
        each channel follows its local mode and exchanges no power. Refused beside
        `wall`, which always carries the connection; no effect on one channel or a
        constant bore. See :meth:`~solprop.driver.simulation.Simulation.from_capillary`.
    model : {'full', 'reduced'}, optional
        Which Marcatili model the modes use; refused beside a
        supplied set unless it is the default or the modes' own.
    loss : bool, float or callable, optional
        Propagation loss of the modes; refused beside a supplied
        set unless it is the default or the modes' own.
    cladding : GlassIndex, complex or None, optional
        Cladding index; ``None`` is the package's cited silica. Beside a
        supplied set, refused unless it is the default or the modes' own.
    susceptibility : GasSusceptibility or None, optional
        A gas susceptibility replacing the one `gas` selects, for a comparison
        against another code's refractive data. Refused for a fraction mapping
        or an existing mixture, which own their species models.
    temperature : float, ZProfile or None, optional
        Gas temperature, K, or a path of it along z in m. Sets the density the
        pressure path maps to, so it changes both the dispersion and the
        nonlinearity. A path also weights a steady-flow gradient by the gas's
        ``eta(T) T`` (docs/guide/waveguides/profiles.md), makes a shared mixture's
        populations independent parameters, and holds the rotational Raman
        populations at one reference temperature with a
        :class:`~solprop.exceptions.RamanTemperatureWarning`
        (:func:`solprop.nonlinear.mixtures.raman_temperature`). A constant path
        is the number, bit for bit. ``None`` (the default) is 293.15 K. A Mixture
        owns its temperature: beside one, a temperature must be omitted or be the
        mixture's own (the same number or path object), and any other is refused by
        name.
    kerr : bool, optional
        Include the Kerr response.
    thg : bool or None, optional
        Include third-harmonic generation in the Kerr term; ``None`` takes
        ``not envelope``.
    plasma : bool, IonisationRate, Mapping or None, optional
        Include photoionisation. ``None`` takes ``not envelope``. ``True`` selects
        optmat's recommended rate model of each built-in gas (SEP 3, D1;
        :class:`~solprop.nonlinear.optmat_rates.OptmatRate`, or the model the
        selection's ``ionisation`` names), for a single gas and for every component of
        a mixture; a custom species needs its own rate. A rate object
        is used as given; a mapping selects a rate or switch per component label, and
        omitted labels are off.
    preionisation : float, ZProfile, Mapping or None, optional
        Initial ionised fraction in [0, 1], globally or per label: a number, or a
        path along z in m giving the fraction ionised before the pulse arrives at
        each position. None and omitted mapping labels retain the component's
        existing declaration.
    plasma_integration : {'band', 'trapezoid'} or None, optional
        Form the field-grid population and free-current integrals (SEP 18):
        ``'band'`` filters each integrand with taps designed for the grid, exact to
        a certified 1e-5 across the active band; ``'trapezoid'`` is the plain
        cumulative trapezoid, short by up to 11 % per integral at the top of the
        default active band, which the frozen witness comparisons use. ``None`` takes
        :class:`~solprop.nonlinear.responses.PlasmaResponse`'s default. Refused with
        envelope plasma, whose currents are converted exactly, and when the plasma
        is off.
    raman : bool, sequence of str, Mapping or None, optional
        ``None`` (the default) turns the Raman response on for every species whose
        optmat response is curated -- N2, O2, H2, D2, CH4 and SF6 -- leaves an atom
        without one, and warns (:class:`solprop.exceptions.SolpropWarning`) for a
        molecule whose response is provisional (N2O, unless the selection's ``raman``
        is ``"provisional"``) rather than leaving it out silently; with ``kerr=False``
        the default is off. ``False`` disables it; ``True`` selects the
        available models of a mixture and is refused by name for a single gas without
        data; explicit labels or :class:`~solprop.physdata.raman_types.MolecularOptions`
        select components and terms, and a named unavailable model fails explicitly.
    n_r : int or None, optional
        Radial nodes of the fixed rule. ``None`` takes
        :func:`solprop.driver.simulation.default_radial_order` of the mode count.
    n_theta : int or None, optional
        Azimuthal nodes. Refused for a set of purely radial modes, which need none.
    radial_rule : str, optional
        The radial rule family (SEP 19): ``"gauss_kronrod"`` (the default), whose
        embedded Gauss subset is the quadrature alarm's error estimate;
        ``"gauss"``, with no alarm; or ``"field_scale"`` and
        ``"field_scale_radau"``, Gaussian in the HE_11 field scale for expensive
        responses, which need exactly one Marcatili HE_11 mode and have no alarm
        (``docs/guide/waveguides/radial-quadrature.md``).
    ionisation_field_range : tuple of float or None, optional
        Field amplitudes in V/m the rate table must cover. ``None`` derives them
        from the input's peak field over the axis and the transverse nodes; see
        :func:`solprop.driver.simulation.default_field_range`.
    absorber : solprop.boundaries.Absorber or None, optional
        The collar rates; ``None`` builds the default collar of the grid.
    absorber_scheme : {'nonlinear', 'split_step'} or None, optional
        How the temporal collar is applied (ADR-025): inside every stage, the
        default, or once per accepted step. The low-call Adams methods of
        SEP 13 step much further with ``'split_step'``, whose first-order
        splitting error stays inside the collar.
    noise : {'default', None, 'input', 'modified'}, optional
        Which shot-noise model to carry.
        ``"default"``, the default, resolves to ``"input"`` wherever the run can
        carry noise, and records the resolution (SEP 37).
    seed : int or None, optional
        Entropy of the noise generator.
    rtol : float or None, optional
        ``None`` takes the default below, or the single-precision floor.
        Relative step tolerance; the default is
        :data:`~solprop.driver.simulation.CAPILLARY_RTOL`, the capillary
        default. Calibrate it for a production run (``docs/dev/stepper.md``
        section 3).
    atol : float, optional
        Absolute step tolerance, in the square root of watts.
    dz_max : float or None, optional
        Cap on the step, m; ``None`` takes half the capillary.
    dz_init : float or None, optional
        First step, m; ``None`` takes ``length * 1e-4``.
    schedule : solprop.stepper.schedule.FixedSchedule or None, optional
        A fixed schedule to replay instead of stepping adaptively.
    integrator : Multistep, RungeKutta or LowStorageRungeKutta, optional
        The method that steps the run (``docs/guide/numerics/choosing-a-stepper.md``).
        ``None``,
        the default, takes ``Multistep()``: the low-call exponential Adams method, one
        response evaluation per step. ``RungeKutta()`` is the interaction-picture
        DOPRI5 march, for a cheap response or a state too large for the Adams history.
        ``LowStorageRungeKutta()`` replays `schedule` in three registers (SEP 24).
    interpolate_planes : bool, optional
        Observe the save planes inside accepted steps instead of landing on them
        (SEP 14): through the Adams march's own polynomial, or DOPRI5's continuous
        extension with ``RungeKutta()``. Default ``False``.
    step_planes : int or None, optional
        Also save the end of every `step_planes`-th accepted step, the
        march's own positions, beside the requested planes (SEP 14); ``0`` saves the
        requested planes only. ``None``, the default, is ``0``, or ``1`` with a
        `tdse` response, which saves every accepted step (SEP 20).
    save_planes : sequence of float or None, optional
        Positions to save at, m. Give this or `n_save`.
    n_save : int or None, optional
        How many planes to save, counting ``z = 0``.
    stats : solprop.stats.StatsSpec or None, optional
        Which statistics to record; ``None`` takes
        :func:`solprop.stats.default_spec` for a multimode run with the profiles,
        plus every statistic only a backend can compute.
    node_budget : int, optional
        Bytes of node-chunk buffers the response pass may hold.
    backend : str, optional
        Registry name, or ``"auto"`` (the default) to prefer ``native`` if installed
        and otherwise use Numba CPU. Import or construction errors do not fall back.
    precision : {'double', 'single'}, optional
        The arithmetic the run is held in (SEP 33,
        ``docs/guide/performance/single-precision.md``). ``"double"``, the default, is
        binary64 throughout. ``"single"`` holds the state and the source evaluation in
        binary32: its default `rtol` is the floor
        :data:`~solprop.translate.precision.SINGLE_RTOL_FLOOR`, below which an explicit
        value is refused; its default integrator is ``RungeKutta()``, and the Adams
        method, noise models and every backend but ``numpy_ref`` are refused.
    polarisation : str, float or None, optional
        Refused; this path is scalar.
    output : solprop.output.schema.Output, pathlib.Path, str or None, optional
        Where to store the run: a handler, a path to stream HDF5 to, or ``None`` for a
        :class:`~solprop.output.memory.MemoryOutput`.
    snapshots : solprop.output.SnapshotController or None, optional
        Immutable local checkpoints at committed accepted steps; requires HDF5,
        ``step_planes=1``, all fields and non-interpolated observations.
    resume : bool or {'interpolated'}, optional
        Continue an interrupted run in the file at `output`, after checking that its
        configuration is this one's.
        ``"interpolated"`` also continues a dense run from a plane interpolated
        inside a step (:meth:`solprop.driver.simulation.Simulation.run`).
    progress : solprop.log.ProgressReporter or None, optional
        Where progress goes; ``None`` logs a periodic line, one line per saved plane
        and one at the end.
    progress_period : float, optional
        Seconds between the default reporter's periodic lines, s; ``0`` leaves only the
        plane and closing lines. Ignored when `progress` is given.
    materials : solprop.physdata.selection.MaterialSelection or None, optional
        Take the run's built-in materials from optmat under this selection: the gas,
        ``"air"`` and the default cladding resolve through it (see
        :meth:`~solprop.driver.simulation.Simulation.from_capillary`). Supplied objects
        are taken as they are, and the run records its material manifest (output schema
        8). ``None`` (the default) is ``MaterialSelection()``, optmat's recommended
        models.
    tdse : solprop.nonlinear.tdse.TDSEGasResponse or None, optional
        A helium TDSE response (SAE or haCC) as the complete electronic response,
        with ``numpy_ref``, final-only requested output and every accepted step
        saved by default (:meth:`~solprop.driver.simulation.Simulation.from_capillary`,
        ``docs/guide/media/tdse.md``).
    tdse_execution : solprop.nonlinear.tdse.TDSEExecution or None, optional
        Its atomic runtime resources; ``None`` beside `tdse` is the default record.
    tdse_archive : solprop.output.TDSEArchive or None, optional
        Archive every raw provider evaluation of `tdse` in a separate HDF5 file
        (:meth:`~solprop.driver.simulation.Simulation.run`); refused without `tdse`.

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
        If a resume is asked for against a file written by a different configuration.

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
    ...         125e-6, 0.02, "He", (2.0, 1.0), lambda0=800e-9,
    ...         wavelength_limits=(300e-9, 3e-6), trange=100e-15,
    ...         fwhm=10e-15, energy=5e-6, modes=2, plasma=False,
    ...         n_r=16, n_save=2, rtol=1e-8,
    ...     )
    >>> output["field"].shape[1:]
    (2, 1, 257)

    The pressure the run saw is recorded beside the field, and it falls along the
    capillary:

    >>> pressure = output["stats/pressure"]
    >>> bool(pressure[-1] < pressure[0])
    True
    """
    simulation = Simulation.from_capillary(
        radius,
        length,
        gas,
        pressure,
        pressure_floor=pressure_floor,
        vacuum_end=vacuum_end,
        lambda0=lambda0,
        wavelength_limits=wavelength_limits,
        trange=trange,
        grid=grid,
        envelope=envelope,
        fwhm=fwhm,
        width=width,
        pulseshape=pulseshape,
        energy=energy,
        peak_power=peak_power,
        phases=phases,
        pulses=pulses,
        modes=modes,
        mode_average=mode_average,
        lattice=lattice,
        wall=wall,
        taper_coupling=taper_coupling,
        model=model,
        loss=loss,
        cladding=cladding,
        susceptibility=susceptibility,
        temperature=temperature,
        kerr=kerr,
        thg=thg,
        plasma=plasma,
        preionisation=preionisation,
        plasma_integration=plasma_integration,
        raman=raman,
        n_r=n_r,
        n_theta=n_theta,
        radial_rule=radial_rule,
        ionisation_field_range=ionisation_field_range,
        absorber=absorber,
        absorber_scheme=absorber_scheme,
        noise=noise,
        seed=seed,
        rtol=rtol,
        atol=atol,
        dz_max=dz_max,
        dz_init=dz_init,
        schedule=schedule,
        integrator=integrator,
        interpolate_planes=interpolate_planes,
        step_planes=step_planes,
        save_planes=save_planes,
        n_save=n_save,
        stats=stats,
        node_budget=node_budget,
        backend=backend,
        precision=precision,
        polarisation=polarisation,
        materials=materials,
        tdse=tdse,
        tdse_execution=tdse_execution,
    )
    logger.debug(
        "prop_capillary: %.6g m of %s capillary, %d modes, %d quadrature nodes, "
        "%d save planes, %r backend",
        length,
        gas,
        len(simulation.modes),
        0
        if simulation.settings.quadrature is None
        else simulation.settings.quadrature.n_nodes,
        len(simulation.settings.save_planes),
        simulation.settings.backend,
    )
    return simulation.run(
        output,
        snapshots=snapshots,
        progress=progress,
        progress_period=progress_period,
        resume=resume,
        tdse_archive=tdse_archive,
    )


def prop_waveguide(  # noqa: PLR0913 - the interface's keyword surface
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
    lambda0: float,
    wavelength_limits: tuple[float, float] | None = None,
    trange: float | None = None,
    grid: TimeGrid | None = None,
    envelope: bool = False,
    fwhm: float | None = None,
    width: float | None = None,
    pulseshape: str = "gauss",
    energy: float | None = None,
    peak_power: float | None = None,
    phases: Sequence[float] | np.ndarray = (),
    pulses: Sequence[Pulse] | None = None,
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
    absorber: Absorber | None = None,
    noise: NoiseSetting = "default",
    seed: int | None = None,
    rtol: float | None = None,
    atol: float = WAVEGUIDE_ATOL,
    dz_max: float | None = None,
    dz_init: float | None = None,
    schedule: FixedSchedule | None = None,
    integrator: Multistep | RungeKutta | LowStorageRungeKutta | None = None,
    save_planes: Sequence[float] | np.ndarray | None = None,
    n_save: int | None = None,
    stats: StatsSpec | None = None,
    node_budget: int = DEFAULT_NODE_BUDGET,
    backend: str = "auto",
    precision: PrecisionName = "double",
    polarisation: str | float | None = None,
    taper_coupling: bool = True,
    output: Output | Path | str | None = None,
    snapshots: SnapshotController | None = None,
    resume: bool | Literal["interpolated"] = False,
    progress: ProgressReporter | ProgressChoice | None = None,
    progress_period: float = DEFAULT_PROGRESS_PERIOD,
    materials: MaterialSelection | None = None,
) -> Output:
    """Propagate a pulse through a gas-filled waveguide mode set, or through a guide.

    Build with :meth:`~solprop.driver.simulation.Simulation.from_waveguide` and run
    to the requested output. Without `provider` this is the gas-filled route:
    rectangular polarisation partners carry distinct operator rows, a hollow planar
    set (:class:`~solprop.modes.slab.SlabMode`) runs **per unit width** -- `energy`
    in J/m, `peak_power` in W/m, and the recorded powers and energies likewise --
    and step-index fibres use ``prop_gnlse(fibre=...)`` instead. With `provider` it
    is the **general modal route** (``docs/guide/waveguides/general-modal.md``), on a
    field grid, on every backend (Kokkos on CPUs and GPUs), with HDF5 output from output
    schema 7.

    Parameters
    ----------
    modes : sequence of solprop.modes.protocol.Mode
        Gas-filled modes sharing a transverse geometry, in propagation row order.
        Each is rebound to this call's fill: its own gas, pressure and temperature
        (a temperature path included) are replaced, its geometry kept. Empty, or
        the guide's own channels, beside `provider`.
    length : float
        Waveguide length, m.
    gas : Gas, str, Mapping, Mixture or None
        Built-in gas, mole fractions keyed by species name, or ordered populations
        carrying their own density paths. A mapping uses total pressure. Required
        without `provider` and refused beside one.
    pressure : float, sequence, ZProfile or None, optional
        None uses one bar for a gas or fractions. Omit pressure beside a Mixture;
        every explicit pressure is refused because it already carries its paths.
        Fill pressure in **bar**: a number for a constant fill, a pair
        ``(p_in, p_out)`` for the steady-flow gradient of
        :meth:`solprop.profiles.PressureGradient.two_point`, or ``(z, p)`` with the
        positions and the pressures there for
        :meth:`solprop.profiles.PressureGradient.from_stages`.
    provider : solprop.modes.guide.ModalGuide or None, optional
        A guide built by :meth:`solprop.modes.guide.ModalGuide.from_provider`,
        which solves its own three-component E and H and whose material regions own
        their responses. Its knots must span the run's whole active band.
    channel : int or str or None, optional
        Which channel the pulse enters, by index or display label; ``None`` takes
        the first. Requires `provider`.
    incident : solprop.pulses.SpatialPulse or None, optional
        An incident beam carrying its own pulse, whose energy is met over the whole
        beam at the entrance; it is coupled into the channels under `coupling`,
        they launch the share they capture and the rest is lost, with the coupling
        report recorded in the run's configuration (see
        :meth:`~solprop.driver.simulation.Simulation.from_waveguide`). Requires
        `provider` and `coupling`, and is refused beside `channel` and the pulse
        keywords.
    coupling : solprop.coupling.models.CouplingModel or None, optional
        The model and budget an `incident` beam is coupled under, a
        ``vector_paraxial`` model. Requires `incident`.
    kerr_form : {'real', 'analytic'}, optional
        Which instantaneous cubic a general modal run's regions respond with: the
        real-field cubic of the field route, the default, or the analytic-signal
        form, which carries the fundamental band alone (see
        :meth:`~solprop.driver.simulation.Simulation.from_waveguide`). Requires
        `provider`.
    observable_rtol, observable_atol : float or None, optional
        Accepted only as ``None``, which means the solver's own `rtol` and `atol`
        (see :meth:`~solprop.driver.simulation.Simulation.from_waveguide`).
        Require `provider`.
    pressure_floor : float or None, optional
        Lowest pressure the fill is allowed to reach, bar; ``None`` takes
        :data:`solprop.profiles.DEFAULT_PRESSURE_FLOOR_FRACTION` of the highest
        stage pressure. With ``vacuum_end="floor"``, pass ``0.0`` for a true vacuum
        end, which warns, because ``dp/dz`` diverges there.
    vacuum_end : {'choked', 'floor'} or None, optional
        How a pressure gradient's outer end below its choked pressure is treated.
        ``None`` (the default) means ``"choked"`` for a circular hollow guide: the fill
        follows the bore and an end exhausting into vacuum takes the choked pressure
        of the flow (:func:`solprop.gasflow.steady_fill`). ``"floor"`` concerns the
        ends only: it keeps the declared end pressures, for a comparison with a
        model that imposes them, while a taper's fill still follows the bore. See
        :meth:`~solprop.driver.simulation.Simulation.from_capillary`.
    lambda0 : float
        Reference wavelength, m, and the pulse's centre wavelength.
    wavelength_limits : tuple of float or None, optional
        Shortest and longest wavelength of the frequency window, m. Required unless
        `grid` is given.
    trange : float or None, optional
        Requested width of the time window, s. Required unless `grid` is given.
    grid : solprop.grid.TimeGrid or None, optional
        A grid of one's own, instead of the three keywords above; the only way to
        reach ``dt``, ``ffac`` or the transform size.
    envelope : bool, optional
        Propagate an envelope. Plasma is off by default; explicit ``plasma=True``
        selects cycle averaging of instantaneous rates. Plasma forbids THG.
    fwhm : float or None, optional
        Pulse duration at half maximum, s.
    width : float or None, optional
        Natural width of a sech pulse, s; give this or `fwhm`.
    pulseshape : {'gauss', 'sech'}, optional
        Transform-limited shape of the input.
    energy : float or None, optional
        Pulse energy, J. Give this or `peak_power`.
    peak_power : float or None, optional
        Peak power, W, realised after the spectral phases.
    phases : sequence of float, optional
        Taylor spectral phases about `lambda0`, optics convention: CEP in rad,
        group delay in s, GDD in s^2, and so on.
    pulses : sequence of solprop.pulses.Pulse or None, optional
        Several pulses launched together, for two-colour and multi-colour
        inputs: each a :mod:`solprop.pulses` object with its own wavelength,
        duration, energy or peak power, phase and delay; the input is the sum of
        their fields (:class:`~solprop.pulses.PulseSum`). Replaces the
        single-pulse keywords `fwhm`, `width`, `pulseshape`, `energy`,
        `peak_power` and `phases`, which must then be left unset; `lambda0` still
        sets the grid's reference wavelength.
    susceptibility : GasSusceptibility or None, optional
        A gas susceptibility replacing the one `gas` selects, for a comparison
        against another code's refractive data. Refused for a fraction mapping
        or an existing mixture, which own their species models.
    temperature : float, ZProfile or None, optional
        Gas temperature, K, or a path of it along z in m. Sets the density the
        pressure path maps to, so it changes both the dispersion and the
        nonlinearity. A path also weights a steady-flow gradient by the gas's
        ``eta(T) T`` (docs/guide/waveguides/profiles.md), makes a shared mixture's
        populations independent parameters, and holds the rotational Raman
        populations at one reference temperature with a
        :class:`~solprop.exceptions.RamanTemperatureWarning`
        (:func:`solprop.nonlinear.mixtures.raman_temperature`). A constant path
        is the number, bit for bit. ``None`` (the default) is 293.15 K. A Mixture
        owns its temperature: beside one, a temperature must be omitted or be the
        mixture's own (the same number or path object), and any other is refused by
        name.
    kerr : bool, optional
        Include the Kerr response.
    thg : bool or None, optional
        Include third-harmonic generation in the Kerr term; ``None`` takes
        ``not envelope``.
    plasma : bool, IonisationRate, Mapping or None, optional
        Include photoionisation. ``None`` takes ``not envelope``. ``True`` selects
        optmat's recommended rate model of each built-in gas (SEP 3, D1;
        :class:`~solprop.nonlinear.optmat_rates.OptmatRate`, or the model the
        selection's ``ionisation`` names), for a single gas and for every component of
        a mixture; a custom species needs its own rate. A rate object
        is used as given; a mapping selects a rate or switch per component label, and
        omitted labels are off.
    preionisation : float, ZProfile, Mapping or None, optional
        Initial ionised fraction in [0, 1], globally or per label: a number, or a
        path along z in m giving the fraction ionised before the pulse arrives at
        each position. None and omitted mapping labels retain the component's
        existing declaration.
    plasma_integration : {'band', 'trapezoid'} or None, optional
        Form the field-grid population and free-current integrals (SEP 18):
        ``'band'`` filters each integrand with taps designed for the grid, exact to
        a certified 1e-5 across the active band; ``'trapezoid'`` is the plain
        cumulative trapezoid, short by up to 11 % per integral at the top of the
        default active band, which the frozen witness comparisons use. ``None`` takes
        :class:`~solprop.nonlinear.responses.PlasmaResponse`'s default. Refused with
        envelope plasma, whose currents are converted exactly, and when the plasma
        is off.
    raman : bool, sequence of str, Mapping or None, optional
        ``None`` (the default) turns the Raman response on for every species whose
        optmat response is curated -- N2, O2, H2, D2, CH4 and SF6 -- leaves an atom
        without one, and warns (:class:`solprop.exceptions.SolpropWarning`) for a
        molecule whose response is provisional (N2O, unless the selection's ``raman``
        is ``"provisional"``) rather than leaving it out silently; with ``kerr=False``
        the default is off. ``False`` disables it; ``True`` selects the
        available models of a mixture and is refused by name for a single gas without
        data; explicit labels or :class:`~solprop.physdata.raman_types.MolecularOptions`
        select components and terms, and a named unavailable model fails explicitly.
    mode_average : bool, optional
        Propagate the one mode **mode-averaged**, as a uniform disc of its effective
        area, instead of resolving it on the transverse rule, as Luna.jl's default
        does (``docs/guide/waveguides/mode-averaging.md``). The plasma then sees the
        area-mean field, below the on-axis one (``HE_11``'s axis intensity is 1.768
        times ``P/A_eff``), and ``peak_intensity`` stays on the axis. ``False``, the
        default, resolves the mode. Refused beside more than one mode and beside
        `n_r`, `n_theta` or a `radial_rule`.
        Refused beside `provider`.
    n_r : int or None, optional
        Radial nodes of the fixed rule, or the x-axis nodes of a Cartesian product
        or slab rule. ``None`` takes
        :func:`solprop.driver.simulation.default_radial_order` of the mode count,
        or :func:`~solprop.modes.quadrature.cartesian_nodes` of the set's highest
        transverse order where that is larger.
    n_theta : int or None, optional
        Azimuthal nodes, or the y-axis nodes of a Cartesian product rule, whose
        default is :func:`~solprop.modes.quadrature.cartesian_nodes`. Refused for a
        set of purely radial modes, which need none, and for a slab set, which is
        uniform in y.
    radial_rule : str, optional
        The radial rule family (SEP 19): ``"gauss_kronrod"`` (the default), whose
        embedded Gauss subset is the quadrature alarm's error estimate;
        ``"gauss"``, with no alarm; or ``"field_scale"`` and
        ``"field_scale_radau"``, Gaussian in the HE_11 field scale for expensive
        responses, which need exactly one Marcatili HE_11 mode and have no alarm
        (``docs/guide/waveguides/radial-quadrature.md``).
    ionisation_field_range : tuple of float or None, optional
        Field amplitudes in V/m the rate table must cover. ``None`` derives them
        from the input's peak field over the axis and the transverse nodes; see
        :func:`solprop.driver.simulation.default_field_range`.
    absorber : solprop.boundaries.Absorber or None, optional
        The collar rates; ``None`` builds the default collar of the grid.
    noise : {'default', None, 'input', 'modified'}, optional
        Which shot-noise model to carry.
        ``"default"``, the default, resolves to ``"input"`` wherever the run can
        carry noise, and records the resolution (SEP 37).
    seed : int or None, optional
        Entropy of the noise generator.
    rtol : float or None, optional
        ``None`` takes the default below, or the single-precision floor.
        Relative step tolerance; the default is
        :data:`~solprop.driver.simulation.WAVEGUIDE_RTOL`, the modal waveguide
        default. Calibrate it for a production run (``docs/dev/stepper.md``
        section 3).
    atol : float, optional
        Absolute step tolerance, in the square root of watts.
    dz_max : float or None, optional
        Cap on the step, m; ``None`` takes half the waveguide.
    dz_init : float or None, optional
        First step, m; ``None`` takes ``length * 1e-4``.
    schedule : solprop.stepper.schedule.FixedSchedule or None, optional
        A fixed schedule to replay instead of stepping adaptively.
    integrator : Multistep, RungeKutta or LowStorageRungeKutta, optional
        The method that steps the run (``docs/guide/numerics/choosing-a-stepper.md``).
        ``None``, the default, takes ``RungeKutta()``, the interaction-picture DOPRI5
        march, which SEP 37 measured fastest at matched accuracy among the methods
        every route of this interface admits. ``Multistep()`` is the low-call
        exponential Adams method, and ``LowStorageRungeKutta()`` replays `schedule` in
        three registers.
    save_planes : sequence of float or None, optional
        Positions to save at, m. Give this or `n_save`.
    n_save : int or None, optional
        How many planes to save, counting ``z = 0``.
    stats : solprop.stats.StatsSpec or None, optional
        Which statistics to record; ``None`` takes
        :func:`solprop.stats.default_spec` for a multimode run with the profiles,
        plus every statistic only a backend can compute.
    node_budget : int, optional
        Bytes of node-chunk buffers the response pass may hold.
    backend : str, optional
        Registry name, or ``"auto"`` (the default) to prefer ``native`` if installed
        and otherwise use Numba CPU. Import or construction errors do not fall back.
    precision : {'double', 'single'}, optional
        The arithmetic the run is held in (SEP 33,
        ``docs/guide/performance/single-precision.md``). ``"double"``, the default, is
        binary64 throughout. ``"single"`` holds the state and the source evaluation in
        binary32: its default `rtol` is the floor
        :data:`~solprop.translate.precision.SINGLE_RTOL_FLOOR`, below which an explicit
        value is refused; its default integrator is ``RungeKutta()``, and the Adams
        method, noise models and every backend but ``numpy_ref`` are refused.
    polarisation : str, float or None, optional
        Refused; this path is scalar.
    taper_coupling : bool, optional
        Carry the taper coupling of a supplied capillary set on a varying radius;
        ``False`` is the opt-out to the adiabatic route, refused beside `provider`
        (see :meth:`~solprop.driver.simulation.Simulation.from_waveguide`).
    output : solprop.output.schema.Output, pathlib.Path, str or None, optional
        Where to store the run: a handler, a path to stream HDF5 to, or ``None`` for a
        :class:`~solprop.output.memory.MemoryOutput`.
    snapshots : solprop.output.SnapshotController or None, optional
        Immutable local checkpoints at committed accepted steps; requires HDF5,
        ``step_planes=1``, all fields and non-interpolated observations.
    resume : bool or {'interpolated'}, optional
        Continue an interrupted run in the file at `output`, after checking that its
        configuration is this one's.
        ``"interpolated"`` also continues a dense run from a plane interpolated
        inside a step (:meth:`solprop.driver.simulation.Simulation.run`).
    progress : solprop.log.ProgressReporter or None, optional
        Where progress goes; ``None`` logs a periodic line, one line per saved plane
        and one at the end.
    progress_period : float, optional
        Seconds between the default reporter's periodic lines, s; ``0`` leaves only the
        plane and closing lines. Ignored when `progress` is given.
    materials : solprop.physdata.selection.MaterialSelection or None, optional
        Take the run's built-in materials from optmat under this selection: the gas and
        ``"air"`` resolve through it (see
        :meth:`~solprop.driver.simulation.Simulation.from_waveguide`). Supplied objects
        are taken as they are, and the run records its material manifest (output schema
        8). ``None`` (the default) is ``MaterialSelection()``, optmat's recommended
        models.

    Returns
    -------
    solprop.output.schema.Output
        Saved fields, metadata, statistics and step history.
    """
    simulation = Simulation.from_waveguide(
        modes,
        length,
        gas,
        pressure,
        provider=provider,
        channel=channel,
        incident=incident,
        coupling=coupling,
        kerr_form=kerr_form,
        observable_rtol=observable_rtol,
        observable_atol=observable_atol,
        pressure_floor=pressure_floor,
        vacuum_end=vacuum_end,
        lambda0=lambda0,
        wavelength_limits=wavelength_limits,
        trange=trange,
        grid=grid,
        envelope=envelope,
        fwhm=fwhm,
        width=width,
        pulseshape=pulseshape,
        energy=energy,
        peak_power=peak_power,
        phases=phases,
        pulses=pulses,
        susceptibility=susceptibility,
        temperature=temperature,
        kerr=kerr,
        thg=thg,
        plasma=plasma,
        preionisation=preionisation,
        plasma_integration=plasma_integration,
        raman=raman,
        mode_average=mode_average,
        n_r=n_r,
        n_theta=n_theta,
        radial_rule=radial_rule,
        ionisation_field_range=ionisation_field_range,
        absorber=absorber,
        noise=noise,
        seed=seed,
        rtol=rtol,
        atol=atol,
        dz_max=dz_max,
        dz_init=dz_init,
        schedule=schedule,
        integrator=integrator,
        save_planes=save_planes,
        n_save=n_save,
        stats=stats,
        node_budget=node_budget,
        backend=backend,
        precision=precision,
        polarisation=polarisation,
        taper_coupling=taper_coupling,
        materials=materials,
    )
    logger.debug(
        "prop_waveguide: %.6g m of %s waveguide, %d modes, %d quadrature nodes, "
        "%d save planes, %r backend",
        length,
        gas,
        len(simulation.modes),
        0
        if simulation.settings.quadrature is None
        else simulation.settings.quadrature.n_nodes,
        len(simulation.settings.save_planes),
        simulation.settings.backend,
    )
    return simulation.run(
        output,
        snapshots=snapshots,
        progress=progress,
        progress_period=progress_period,
        resume=resume,
    )
