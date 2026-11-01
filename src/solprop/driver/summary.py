# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The block a run shows before its first step: what is about to be computed, and how.

Every route reaches :func:`solprop.driver.run.run_problem`, which builds this summary
from the translated :class:`~solprop.translate.problem.KernelProblem`, the run's
:class:`~solprop.output.schema.RunMetadata` (whose ``configuration`` holds the physics
objects the problem was built from: the responses, the medium, the profiles), the
backend's own execution record and the output. It hands the text to the progress
reporter, so it follows the ``progress`` choice exactly as the progress lines do, and
records it on a plane store as the :data:`RUN_SUMMARY_ANNOTATION`.

The rows are descriptions for a person, not a record: everything in them is also in the
run's metadata, which is what a program should read. Nothing here evaluates physics.

The manual page is ``docs/guide/performance/progress.md``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from solprop._display import Displayed, block, engineering, tail
from solprop._host import machine
from solprop._process import thread_budget
from solprop.constants import C
from solprop.log import format_duration
from solprop.monitor import BoundaryCheck
from solprop.output.schema import Output, RunMetadata
from solprop.stepper.multistep import Multistep
from solprop.translate.problem import KernelProblem
from solprop.transverse_monitor import READS

__all__ = [
    "RUN_SUMMARY_ANNOTATION",
    "RunSummary",
    "memory_size",
    "resources",
    "run_summary",
]

RUN_SUMMARY_ANNOTATION = "run_summary"
"""Name of the annotation holding the start-of-run summaries of a plane store.

A list with one mapping per start, ``{"title": str, "rows": [[key, value], ...]}``: a
run that is resumed adds the summary of its continuation to the first one's.
"""


@dataclass(frozen=True)
class RunSummary(Displayed):
    """What a run is about to compute, as one aligned block.

    Attributes
    ----------
    title : str
        The first line: where the run starts and ends.
    rows : tuple of tuple of str
        ``(key, value)`` pairs, one per aspect of the run, in a fixed order.

    Examples
    --------
    >>> from solprop.driver.summary import RunSummary
    >>> print(RunSummary("Starting propagation", (("noise", "off"),)))
    Starting propagation
      noise: off
    """

    title: str
    rows: tuple[tuple[str, str], ...]

    def __repr__(self) -> str:
        """Return a one-line description.

        Returns
        -------
        str
            The title and the number of rows.
        """
        return f"RunSummary({self.title!r}, {len(self.rows)} rows)"

    def _describe(self) -> tuple[str, Sequence[tuple[str, object]]]:
        """Return the title and the rows of the block.

        Returns
        -------
        tuple
            The title, and the rows as they are.
        """
        return self.title, self.rows

    def text(self) -> str:
        """Return the block, as the reporter shows it.

        Returns
        -------
        str
            The title and one line per row, without a trailing newline.
        """
        return block(self.title, self.rows)

    def record(self) -> dict[str, object]:
        """Return the summary as plain JSON values, for a store's annotation.

        Returns
        -------
        dict
            ``{"title": ..., "rows": [[key, value], ...]}``.
        """
        return {"title": self.title, "rows": [list(row) for row in self.rows]}


def run_summary(  # noqa: PLR0913 - the run's parts, each a distinct object
    problem: KernelProblem,
    metadata: RunMetadata,
    *,
    backend: str,
    execution: Mapping[str, object],
    output: Output,
    start_plane: int = 0,
    z_start: float = 0.0,
    boundary_check: BoundaryCheck | None = None,
) -> RunSummary:
    """Describe a run before its first step.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    metadata : solprop.output.schema.RunMetadata
        The run's record; its ``configuration`` supplies what the problem holds only as
        compiled tables (the medium and the response models).
    backend : str
        The backend's name.
    execution : collections.abc.Mapping
        The backend's ``metadata()``, which says how it sized its thread pools and
        which propagator it chose.
    output : solprop.output.schema.Output
        Where the planes go.
    start_plane : int, optional
        Index of the first save plane beyond the loaded state; above zero the run is a
        resume.
    z_start : float, optional
        Where the run starts, m.
    boundary_check : solprop.monitor.BoundaryCheck or None, optional
        The thresholds the window's edges are watched with (SEP 21); ``None`` when they
        are not watched.

    Returns
    -------
    RunSummary
        The title and the rows.

    Examples
    --------
    >>> from solprop.driver.simulation import metadata_for
    >>> from solprop.driver.summary import run_summary
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.output import MemoryOutput
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> grid = EnvelopeGrid.from_limits(0.01, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0,
    ...                      gamma=0.11, shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8).build(grid)
    >>> problem = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-6, atol=1e-16),
    ...                    save_planes=[0.005, 0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> summary = run_summary(problem, metadata_for(problem, grid), backend="numpy_ref",
    ...                       execution={}, output=MemoryOutput())
    >>> dict(summary.rows)["shot noise"]
    'off'
    """
    planes = np.asarray(problem.stepper.save_planes, dtype=np.float64)
    z_end = float(planes[-1])
    title = (
        f"Starting propagation from z = {engineering(z_start, 'm')} "
        f"to {engineering(z_end, 'm')}"
    )
    configuration = metadata.configuration
    rows = (
        ("solprop", _code(metadata)),
        ("machine", _machine(machine())),
        ("backend", backend),
        ("threads", _threads(backend, execution)),
        ("precision", _precision(problem)),
        ("grid", _grid(metadata)),
        ("representation", _representation(problem, metadata)),
        ("medium", _medium(configuration)),
        *(
            (("pressure", pressure),)
            if (pressure := _pressure(configuration)) is not None
            else ()
        ),
        ("responses", _responses(problem, configuration)),
        ("shot noise", _noise(metadata)),
        ("integrator", _integrator(problem)),
        ("propagator", str(execution.get("propagator_mode", "not reported"))),
        ("tolerances", _tolerances(problem)),
        ("absorber", _absorber(problem, boundary_check)),
        *(
            [("transverse edge", _transverse(problem, boundary_check, backend))]
            if problem.transform.geometry is not None
            else []
        ),
        ("save planes", _planes(problem, planes)),
        ("output", _output(output)),
        ("start", _start(start_plane, planes.size, z_start)),
    )
    return RunSummary(title, rows)


def _code(metadata: RunMetadata) -> str:
    """Describe the code that runs: the installed version or, in a checkout, the commit.

    The installed version of a development checkout is the one it had when it was
    installed, not the commit it now runs, so a checkout leads with its commit and
    whether it was modified; the branch and the installed version follow, where a long
    branch name may be cut.

    Parameters
    ----------
    metadata : solprop.output.schema.RunMetadata
        The run's record, whose ``versions`` and ``git`` were read when it was built.

    Returns
    -------
    str
        For example ``"0.1.0"`` or ``"commit 5e8f6884a1b2, modified (main, installed
        as 0.1.dev3+g1a2b3c4)"``.
    """
    installed = str(metadata.versions.get("solprop", "unknown"))
    git = metadata.git
    if git is None:
        return installed
    modified = ", modified" if git.dirty else ""
    return (
        f"commit {git.commit[:12]}{modified} ({git.branch}, installed as {installed})"
    )


def _machine(record: Mapping[str, object]) -> str:
    """Describe the machine: architecture, CPU, CPU counts, memory and system.

    Parameters
    ----------
    record : collections.abc.Mapping
        The machine, as the execution record's ``host`` holds it.

    Returns
    -------
    str
        For example ``"arm64, Apple M4 Max, 16 CPUs (16 usable), 64 GiB, Darwin
        25.0.0"``; a part that could not be read is left out.
    """
    logical, usable = record.get("logical_cpus"), record.get("usable_cpus")
    memory = record.get("memory_bytes")
    parts = [
        record.get("machine") or None,
        record.get("cpu_model"),
        None if logical is None else f"{logical} CPUs ({usable} usable)",
        memory_size(memory) if isinstance(memory, int) else None,
        f"{record.get('system', '')} {record.get('release', '')}".strip() or None,
    ]
    return ", ".join(str(part) for part in parts if part)


def memory_size(size: float) -> str:
    """Write a memory size in binary units, as a person reads it.

    Parameters
    ----------
    size : float
        Bytes.

    Returns
    -------
    str
        Three significant figures and the largest binary unit below the size.

    Examples
    --------
    >>> from solprop.driver.summary import memory_size
    >>> memory_size(137438953472), memory_size(30277632)
    ('128 GiB', '28.9 MiB')
    """
    value, unit = float(size), "B"
    for larger in ("KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0:
            break
        value, unit = value / 1024.0, larger
    return f"{value:.3g} {unit}"


def resources(spent: Mapping[str, float | int | None]) -> str:
    """Describe what a run spent, as the line shown after its closing one.

    Parameters
    ----------
    spent : collections.abc.Mapping
        The run's ``wall_time_s``, ``cpu_time_s`` and ``peak_rss_bytes``, as the
        execution record holds them.

    Returns
    -------
    str
        The run's wall and CPU time, and the process's peak resident memory where the
        system reports it.

    Examples
    --------
    >>> from solprop.driver.summary import resources
    >>> resources({"wall_time_s": 192.4, "cpu_time_s": 2100.0,
    ...            "peak_rss_bytes": 2254857830})
    'resources: 0:03:12 wall, 0:35:00 CPU, process peak memory 2.1 GiB'
    >>> resources({"wall_time_s": 0.132, "cpu_time_s": None, "peak_rss_bytes": None})
    'resources: 0.132 s wall'
    """
    parts = [f"{_duration(float(spent['wall_time_s'] or 0.0))} wall"]
    cpu, peak = spent.get("cpu_time_s"), spent.get("peak_rss_bytes")
    if cpu is not None:
        parts.append(f"{_duration(float(cpu))} CPU")
    if peak is not None:
        parts.append(f"process peak memory {memory_size(peak)}")
    return "resources: " + ", ".join(parts)


def _duration(seconds: float) -> str:
    """Write a duration in seconds below a minute, else as ``H:MM:SS``.

    Parameters
    ----------
    seconds : float
        The duration, s.

    Returns
    -------
    str
        For example ``"0.132 s"`` or ``"0:03:12"``.
    """
    return f"{seconds:.3g} s" if seconds < 60.0 else format_duration(seconds)


def _threads(backend: str, execution: Mapping[str, object]) -> str:
    """Describe the thread pools the backend took.

    Parameters
    ----------
    backend : str
        The backend's name.
    execution : collections.abc.Mapping
        The backend's ``metadata()``.

    Returns
    -------
    str
        The pools and the process budget that bounds them.
    """
    budget = execution.get("thread_budget")
    source = execution.get("thread_budget_source")
    if budget is None:
        process = thread_budget()
        budget, source = process.threads, process.source
    limit = f"budget {budget} ({source})"
    if "numba_threads" in execution:
        kernel, fftw = execution["numba_threads"], execution.get("fftw_threads")
        return f"{kernel} kernel, {fftw} FFTW; {limit}"
    if backend == "numpy_ref":
        # The reference steps in NumPy on one thread, whatever the budget allows.
        return f"1 (NumPy reference); {limit}"
    return limit


def _precision(problem: KernelProblem) -> str:
    """Describe the floating-point precision of the run.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.

    Returns
    -------
    str
        ``"double"`` or ``"single"``.
    """
    # Translation admits no mixed precision, so the state's name is the run's.
    return problem.precision.state


def _grid(metadata: RunMetadata) -> str:
    """Describe the time grid: kind, points, window and wavelength band.

    Parameters
    ----------
    metadata : solprop.output.schema.RunMetadata
        The run's record.

    Returns
    -------
    str
        For example ``"field, 4096 points, window 683.1 fs, 150 nm to 4 um"``.
    """
    grid = metadata.grid
    window = grid.n_t * grid.dt
    shortest = 2.0 * np.pi * C / grid.omega_max
    longest = 2.0 * np.pi * C / grid.omega_min if grid.omega_min > 0.0 else None
    band = (
        f"{engineering(shortest, 'm')} to {engineering(longest, 'm')}"
        if longest is not None
        else f"above {engineering(shortest, 'm')}"
    )
    return (
        f"{grid.kind}, {grid.n_t} points, window {engineering(window, 's')}, "
        f"band {band}"
    )


def _representation(problem: KernelProblem, metadata: RunMetadata) -> str:
    """Describe the transverse representation and the polarisation.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    metadata : solprop.output.schema.RunMetadata
        The run's record.

    Returns
    -------
    str
        For example ``"mode-averaged (1 mode), scalar"``.
    """
    n_modes, n_pol = problem.shape[1], problem.shape[2]
    polarisation = "scalar" if n_pol == 1 else "vector (x and y)"
    kind = metadata.simulation_kind
    # A spatial route carries its geometry; every modal one has none.
    geometry = problem.transform.geometry
    if "gnlse" in metadata.configuration:
        shape = "GNLSE, one mode"
    elif kind == "mode_averaged":
        shape = "mode-averaged, one mode"
    elif metadata.modal is not None:
        shape = f"general guide, {n_modes} channels"
    elif geometry is None:
        shape = f"{n_modes} mode" + ("" if n_modes == 1 else "s")
        quadrature = metadata.quadrature
        if quadrature is not None:
            shape += (
                f", {quadrature.get('kind')} modal integral on "
                f"{quadrature.get('n_nodes')} nodes"
            )
    elif kind == "radial":
        shape = f"radial, {geometry.shape[0]} points"
    elif kind == "free":
        n_x, n_y = geometry.shape[0], geometry.shape[1]
        shape = (
            f"Cartesian line, {n_x} points" if n_y == 1 else f"Cartesian {n_x}x{n_y}"
        )
    else:
        shape = f"planar, {geometry.shape[0]}x{geometry.shape[1]} nodes"
    return f"{shape}, {polarisation}"


def _medium(configuration: Mapping[str, object]) -> str:
    """Describe the medium and its fill.

    Parameters
    ----------
    configuration : collections.abc.Mapping
        The run's recorded physics objects.

    Returns
    -------
    str
        The mixture or the bulk medium with any pressure profile, the GNLSE's
        nonlinear coefficient, or ``"not recorded"``.
    """
    gnlse = configuration.get("gnlse")
    if gnlse is not None:
        gamma = getattr(gnlse, "gamma", None)
        return "fibre" if gamma is None else f"fibre, gamma {gamma:.4g} /W/m"
    medium = configuration.get("medium")
    if medium is None:
        return "not recorded"
    text = _mixture(medium) if isinstance(medium, Mapping) else repr(medium)
    for guide in ("capillary", "waveguide"):
        record = configuration.get(guide)
        if isinstance(record, Mapping) and isinstance(record.get("radius"), float):
            text += f", core radius {engineering(record['radius'], 'm')}"
    return text


def _pressure(configuration: Mapping[str, object]) -> str | None:
    """Describe the fill pressure, constant or along z.

    Parameters
    ----------
    configuration : collections.abc.Mapping
        The run's recorded physics objects.

    Returns
    -------
    str or None
        The constant pressure in bar or the profile's own description; ``None`` for
        a run with no pressure profile.
    """
    profiles = configuration.get("profiles")
    pressure = profiles.get("pressure") if isinstance(profiles, Mapping) else None
    if pressure is None:
        return None
    if type(pressure).__name__ == "Constant":
        return f"{getattr(pressure, 'value', 0.0):.4g} bar"
    return repr(pressure)


def _mixture(record: Mapping[str, object]) -> str:
    """Describe a gas mixture from the record a run keeps of it.

    Parameters
    ----------
    record : collections.abc.Mapping
        The mixture's record: its ``labels``, mole ``fractions`` and ``temperature_K``.

    Returns
    -------
    str
        For example ``"He, 293.15 K"`` or ``"Ar 0.8 + He 0.2, 293.15 K"``.
    """
    labels = record.get("labels")
    fractions = record.get("fractions")
    if not isinstance(labels, Sequence) or isinstance(labels, str):
        return str(record.get("__class__", "recorded"))
    names = [str(label) for label in labels]
    if isinstance(fractions, Sequence) and len(fractions) == len(names) > 1:
        names = [
            f"{name} {float(share):.3g}"  # pyright: ignore[reportArgumentType] - recorded floats
            for name, share in zip(names, fractions, strict=True)
        ]
    temperature = record.get("temperature_K")
    kelvin = f", {temperature:.5g} K" if isinstance(temperature, float) else ""
    return " + ".join(names) + kelvin


def _responses(problem: KernelProblem, configuration: Mapping[str, object]) -> str:
    """Describe the nonlinear responses, one line each, with Raman always named.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem, which says what was compiled when the configuration
        does not hold the response objects.
    configuration : collections.abc.Mapping
        The run's recorded physics objects.

    Returns
    -------
    str
        One line per response; ``"Raman off"`` when none is Raman.
    """
    recorded = configuration.get("responses")
    responses = list(recorded) if isinstance(recorded, Sequence) else []
    lines = [_response(response) for response in responses] or _compiled(
        kerr=problem.responses.kerr_coefficients is not None,
        raman_terms=problem.flags.n_raman if problem.flags.has_raman else None,
        plasma=problem.flags.has_plasma,
        dispersive=problem.responses.dispersive_kerr is not None,
        tdse=problem.tdse is not None,
    )
    if not any(line.startswith("Raman") for line in lines):
        lines.append("Raman off")
    return "\n".join(lines)


def _compiled(
    *, kerr: bool, raman_terms: int | None, plasma: bool, dispersive: bool, tdse: bool
) -> list[str]:
    """Name the compiled responses of a run whose record holds no response objects.

    A problem run directly, with a record built without its configuration, still says
    what it computes, from the compiled terms alone.

    Parameters
    ----------
    kerr : bool
        Whether a Kerr term was compiled.
    raman_terms : int or None
        The Raman terms, or ``None`` for none.
    plasma : bool
        Whether a plasma term was compiled.
    dispersive : bool
        Whether a dispersive Kerr term was compiled.
    tdse : bool
        Whether the run carries a TDSE response.

    Returns
    -------
    list of str
        One name per compiled response.
    """
    named = (
        (kerr, "Kerr"),
        (raman_terms is not None, f"Raman ({raman_terms} terms)"),
        (plasma, "plasma"),
        (dispersive, "dispersive Kerr"),
        (tdse, "TDSE"),
    )
    return [name for present, name in named if present]


def _response(response: object) -> str:
    """Describe one response object.

    Parameters
    ----------
    response : object
        A response as the simulation carries it.

    Returns
    -------
    str
        A short phrase naming the model and its species.
    """
    name = type(response).__name__
    species = getattr(response, "species", None)
    who = f" ({species})" if species else ""
    if name == "KerrResponse":
        harmonic = ", third harmonic" if getattr(response, "thg", False) else ""
        return f"Kerr{who}{harmonic}"
    if name == "DispersiveKerrResponse":
        model = type(getattr(response, "model", None)).__name__
        return f"dispersive Kerr{who}, {model}"
    if name in {"RamanResponse", "OptmatRamanResponse", "SilicaTensorResponse"}:
        labels = getattr(response, "labels", ())
        who = f" ({', '.join(labels)})" if labels else who
        fraction = getattr(response, "fraction", None)
        held = (
            f", fraction {fraction:.4g} held back from Kerr"
            if fraction is not None
            else ""
        )
        return f"Raman{who}{held}"
    if name in {"PlasmaResponse", "EnvelopePlasmaResponse"}:
        kind = "plasma" if name == "PlasmaResponse" else "envelope plasma"
        gas = getattr(response, "gas", None)
        label = getattr(gas, "name", gas)
        return f"{kind} ({label}), rate {_rate(getattr(response, 'rate', None))}"
    if name == "TDSEGasResponse":
        target = getattr(response, "target", "")
        model = getattr(response, "model", "")
        return f"TDSE ({target}, {model})"
    return name


def _rate(rate: object) -> str:
    """Describe an ionisation rate.

    Parameters
    ----------
    rate : object
        The rate object of a plasma response.

    Returns
    -------
    str
        ``"optmat <model>"`` for an optmat rate, else the rate's class.
    """
    if type(rate).__name__ == "OptmatRate":
        return f"optmat {getattr(rate, 'model', '')}"
    return type(rate).__name__


def _noise(metadata: RunMetadata) -> str:
    """Describe the shot noise: its model and seed, or that there is none, and why.

    A model the run took by default says so, and a default that resolved to no noise
    gives its reason (SEP 37), so that a noiseless run is never silently noiseless.

    Parameters
    ----------
    metadata : solprop.output.schema.RunMetadata
        The run's record.

    Returns
    -------
    str
        For example ``"input model (default), seed 3"``, ``"off"``, or
        ``"off (default: single precision carries no noise: ...)"``.
    """
    noise = metadata.noise
    resolution = metadata.noise_resolution
    by_default = resolution is not None and resolution.requested == "default"
    if noise is None:
        if by_default and resolution is not None and resolution.reason is not None:
            return f"off (default: {resolution.reason})"
        return "off"
    seed = "not recoverable" if noise.seed is None else str(noise.seed)
    marker = " (default)" if by_default else ""
    return f"{noise.model} model{marker}, seed {seed}"


def _integrator(problem: KernelProblem) -> str:
    """Describe the march and its formulation.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.

    Returns
    -------
    str
        For example ``"Multistep, variable order up to 5, exponential"``.
    """
    stepper = problem.stepper
    integrator = stepper.integrator
    if integrator is None:
        return f"RungeKutta ({stepper.tableau.name})"
    if isinstance(integrator, Multistep):
        orders = (
            f"variable order up to {integrator.order}"
            if integrator.variable_order
            else f"order {integrator.order}"
        )
        return f"Multistep, {orders}, {integrator.formulation} formulation"
    return f"LowStorageRungeKutta ({integrator.scheme})"


def _tolerances(problem: KernelProblem) -> str:
    """Describe the step control: tolerances and step bound, or a replayed schedule.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.

    Returns
    -------
    str
        For example ``"rtol 1e-06, atol 1e-10, dz_max 50 mm"``.
    """
    stepper = problem.stepper
    controller = stepper.controller
    text = (
        f"rtol {controller.rtol:.3g}, atol {controller.atol:.3g}, "
        f"dz_max {engineering(stepper.dz_max, 'm')}"
    )
    if stepper.schedule is not None:
        text += f"; replaying a fixed schedule of {stepper.schedule.steps.size} steps"
    return text


def _absorber(problem: KernelProblem, check: BoundaryCheck | None) -> str:
    """Describe the temporal absorber and the watch on the window's edges.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    check : solprop.monitor.BoundaryCheck or None
        The edge thresholds, or ``None`` when the edges are not watched.

    Returns
    -------
    str
        ``"none"`` or the scheme it is applied by, then the edge watch.
    """
    text = (
        f"temporal, {problem.responses.absorber_scheme} scheme"
        if problem.flags.has_temporal_absorber
        else "none"
    )
    if check is None:
        return f"{text}; edges not watched"
    return (
        f"{text}; edges watched (guards {check.guard_fraction:g} of the window, "
        f"budget {check.budget:.0e})"
    )


def _transverse(
    problem: KernelProblem, check: BoundaryCheck | None, backend: str
) -> str:
    """Describe a free-space run's transverse boundary and what its edge can report.

    SEP 40 section 5.3: ``unwatched``, ``coarse`` and ``off`` are said before the
    first step, so that a run that cannot reach a clean verdict says so.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    check : solprop.monitor.BoundaryCheck or None
        The edge thresholds, or ``None`` when nothing is watched.
    backend : str
        The backend's name.

    Returns
    -------
    str
        The boundary, then what the monitor will report.
    """
    design = problem.transverse
    if design is None:
        boundary = "wall or wrap, no safety ramp"
    elif design.has_collar:
        boundary = (
            f"requested collar, alpha0 {design.alpha0:.3g} 1/m, design check "
            f"{design.check.get('status', 'not run')}; safety ramp"
        )
    else:
        boundary = "wall or wrap with the safety ramp"
    if check is None:
        return f"{boundary}; off (boundary_check=None)"
    if backend != "numpy_ref":
        return f"{boundary}; unwatched on {backend} (numpy_ref reads it)"
    schedule = problem.stepper.schedule
    if (
        schedule is not None
        and float(np.max(schedule.steps))
        > (float(np.max(problem.stepper.save_planes)) - float(problem.initial.z))
        / READS
    ):
        return f"{boundary}; coarse: the fixed schedule is sparser than its reads"
    return f"{boundary}; watched (budget {check.edge_budget:.0e})"


def _planes(problem: KernelProblem, planes: np.ndarray) -> str:
    """Describe the save planes.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    planes : numpy.ndarray
        The save planes, m.

    Returns
    -------
    str
        The count and the extent, and any per-step saving.
    """
    text = (
        f"{planes.size} after the entrance, {engineering(float(planes[0]), 'm')} to "
        f"{engineering(float(planes[-1]), 'm')}"
    )
    if problem.stepper.step_planes:
        text += f", and every {problem.stepper.step_planes} accepted steps"
    if problem.stepper.interpolate_planes:
        text += ", interpolated"
    return text


def _output(output: Output) -> str:
    """Describe where the planes go.

    Parameters
    ----------
    output : solprop.output.schema.Output
        The handler.

    Returns
    -------
    str
        The handler's class, and its file when it has one.
    """
    path = getattr(output, "path", None)
    name = type(output).__name__
    return name if path is None else f"{name} {tail(str(path), 52)}"


def _start(start_plane: int, n_planes: int, z_start: float) -> str:
    """Describe whether the run is new or resumed.

    Parameters
    ----------
    start_plane : int
        Index of the first save plane beyond the loaded state.
    n_planes : int
        The save planes of the run.
    z_start : float
        Where the run starts, m.

    Returns
    -------
    str
        ``"new run"`` or where the resume continues from.
    """
    if start_plane == 0:
        return "new run"
    return (
        f"resuming at z = {engineering(z_start, 'm')}, "
        f"{n_planes - start_plane} of {n_planes} planes to go"
    )
