# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Selecting a backend by name, and saying why one is not usable.

``docs/dev/backends.md`` section 1: selection is by name, availability is detected with
:func:`importlib.util.find_spec` and **never cached**. The
errors are two-tier, because the two failures need different answers from the reader: a
name this package has never heard of is a :class:`ValueError` listing what is usable,
and a known name whose dependencies are missing is an :class:`ImportError` saying what
to install.

Each backend is one row of :data:`BACKENDS`, a :class:`BackendEntry` naming by import
path everything selection reads of it -- its constructor, its lockstep runner and its
policy hooks -- so nothing here branches on a backend's name to construct one, and an
optional backend in another distribution registers with one row
(``docs/dev/backends.md`` section 1).

The native backend is a second distribution, ``solprop-native``, built from ``native/``
in this repository and a required dependency; probing availability does not import it.
Automatic selection is that backend, with no fallback (SEP 43). It refuses when the
package is absent or this processor is below the installed build's CPU baseline, using
the package's own check (``solprop_native._cpu``), loaded by path and read from the
same code that refuses the import. Import, compatibility and construction errors remain
visible, and they never trigger different arithmetic.
"""

import importlib
import importlib.util
import logging
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final, cast

from solprop.backends.protocol import Backend, BatchBackend
from solprop.backends.session import ExecutionOptions
from solprop.exceptions import BackendError
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.profiles import ZProfile
from solprop.spatial import FAR_FIELD_STATISTICS
from solprop.translate.plasma_source import _BENCHMARK_GATED
from solprop.translate.precision import SINGLE_PRECISION_BACKENDS
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import tdse_descriptor
from solprop.translate.run_request import RunRequest
from solprop.translate.tdse import TDSEBinding

__all__ = [
    "AUTO",
    "BACKENDS",
    "FAR_FIELD_BACKENDS",
    "REMOVED_BACKENDS",
    "REQUIRE_DEVICE_SETTING",
    "BackendEntry",
    "available",
    "resolve_name",
    "select",
    "select_batch",
    "statistics_refusal",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendEntry:
    """What selection needs of a backend, each object named by its import path.

    A path is ``"module:attribute"``, the attribute dotted for one inside a class. It is
    imported only when selection needs it, after ``requires`` was found, so listing a
    backend imports nothing.

    Attributes
    ----------
    requires : tuple of str
        Importable module names that must be present. Empty for a backend that needs
        nothing beyond the package's own dependencies.
    extras : tuple of str
        The optional-dependency groups that supply ``requires``, for the error message.
    factory : str
        The backend's constructor, called as ``factory(problem, request, *, grid,
        modes, profiles, options)``, and with ``tdse_binding`` too for a TDSE run, which
        only the reference is given (:func:`select`).
    batch : str or None, optional
        The reference group's lockstep runner over loaded members, ``batch(owners)``
        (:func:`select_batch`); ``None`` for a backend whose prepared group runs its
        members itself, as the native group session does (SEP 47 section 6.7).
    install : str or None, optional
        What to install when ``requires`` is missing; ``None`` names ``extras``.
    distribution : str or None, optional
        The distribution that ships the backend, named by the messages of the policies
        below; ``None`` names the package or the backend instead.
    first_use : str or None, optional
        Called with no arguments before each construction: the native backend's
        first-use calibration (SEP 46); ``None`` for nothing.
    device : str or None, optional
        The build's description, ``device()``, whose ``device_backend`` (``"none"``
        on a CPU build) and ``device_count`` say where it runs
        (:data:`REQUIRE_DEVICE_SETTING`); the module holding it carries the version.
        ``None`` for a backend that never runs on a device.
    far_field : str or None, optional
        The far-field statistics (SEP 23) the installed backend declares it forms;
        another is refused by name before construction (:func:`statistics_refusal`).
        ``None`` for a backend that forms every one, or declares nothing to check.
    """

    requires: tuple[str, ...]
    extras: tuple[str, ...]
    factory: str
    batch: str | None = None
    install: str | None = None
    distribution: str | None = None
    first_use: str | None = None
    device: str | None = None
    far_field: str | None = None


BACKENDS: Mapping[str, BackendEntry] = {
    "numpy_ref": BackendEntry(
        requires=(),
        extras=(),
        factory="solprop.backends.numpy_ref:construct",
        batch="solprop.backends.numpy_ref.batch:lockstep_runner",
    ),
    "native": BackendEntry(
        requires=("solprop_native",),
        extras=(),
        # The optional package has its own strict type gate; it is deliberately absent
        # from this repository's static-analysis source tree.
        factory="solprop_native:NativeBackend",
        install=(
            "solprop, or build solprop-native from native/ in the solprop repository"
        ),
        distribution="solprop-native",
        first_use="solprop.calibration.first_use:ensure",
        device="solprop_native:build_info",
        far_field="solprop_native:NativeBackend.far_field_statistics",
    ),
}
"""The backends this package knows, by name: one row each."""

AUTO: Final = "native"
"""What ``"auto"`` selects, with no fallback (SEP 43)."""

REMOVED_BACKENDS: Mapping[str, str] = {
    "numba_cpu": (
        "the numba_cpu backend was removed in SEP 43: use backend='native', the "
        "solprop-native package, or backend='numpy_ref', the reference, which is "
        "unthreaded and far slower. Output and snapshots that record numba_cpu still "
        "load"
    ),
    "mlx": (
        "the experimental mlx backend was retired in SEP 47: use backend='native', "
        "the solprop-native package, or, for precision='single', backend='numpy_ref', "
        "the reference. Output that records mlx still opens"
    ),
}
"""What to say about a backend that no longer exists, by the name it had."""

FAR_FIELD_BACKENDS: tuple[str, ...] = ("numpy_ref", "native")
"""The backends that form the far-field statistics of SEP 23.

``numpy_ref`` always does; the native backend declares what it forms by its class's
``far_field_statistics`` attribute (its row's ``far_field``). A statistic it does not
declare is refused by name on this side of the seam (:func:`statistics_refusal`), before
it is built.
"""


def _load(path: str) -> object:
    """Import the object a registry path names.

    Parameters
    ----------
    path : str
        ``"module:attribute"``, the attribute dotted for one inside a class.

    Returns
    -------
    object
        The object.
    """
    module, _, attribute = path.partition(":")
    found: object = importlib.import_module(module)
    for part in attribute.split("."):
        found = getattr(found, part)
    return found


def statistics_refusal(name: str, names: Sequence[str]) -> str | None:
    """Return why a backend cannot form some of the statistics named, or ``None``.

    Asking about a backend whose row declares its far-field statistics (the native
    backend's) imports the installed plugin, as selecting it would, to read them.

    Parameters
    ----------
    name : str
        The resolved backend name.
    names : collections.abc.Sequence of str
        The statistics asked for.

    Returns
    -------
    str or None
        The refusal, naming the backends that do form them, or ``None`` when the
        backend forms every one (or is not installed, which selecting it reports).

    Examples
    --------
    >>> from solprop.backends.registry import statistics_refusal
    >>> statistics_refusal("numpy_ref", ["energy", "beam_spectrum"]) is None
    True
    """
    asked = [one for one in names if one in FAR_FIELD_STATISTICS]
    entry = BACKENDS.get(name)
    # numpy_ref, whose row declares nothing, forms every far-field statistic, in
    # binary64; single precision refuses them in validation (SEP 33).
    if not asked or entry is None or entry.far_field is None:
        return None
    formed = _plugin_far_field(name)
    if formed is None:
        return None
    missing = [one for one in asked if one not in formed]
    if not missing:
        return None
    return (
        f"the installed {name!r} backend does not form the far-field statistics "
        f"{missing}; install a {entry.distribution or name} build that does, or run "
        "them on numpy_ref"
    )


def _plugin_far_field(name: str) -> frozenset[str] | None:
    """Return the far-field statistics an installed plugin backend declares.

    Parameters
    ----------
    name : str
        A registry name.

    Returns
    -------
    frozenset of str or None
        What the plugin declares (its row's ``far_field``: the native backend's
        ``NativeBackend.far_field_statistics``), empty for a build that predates the
        attribute, or ``None`` when the name is not a known, installed plugin that
        declares them.
    """
    entry = BACKENDS.get(name)
    if entry is None or entry.far_field is None or not _satisfied(entry):
        return None
    try:
        declared = _load(entry.far_field)
    except AttributeError:
        # A build from before the far-field statistics declares none.
        return frozenset()
    return frozenset(cast(Collection[str], declared))


def available(precision: str = "double") -> tuple[str, ...]:
    """Return the names of the backends that can run a precision here and now.

    Detection is by :func:`importlib.util.find_spec`, so nothing is imported and nothing
    is cached: a backend whose optional dependency is installed while the process runs
    becomes available without a restart, and initializing a native runtime is never a
    side effect of asking.

    Parameters
    ----------
    precision : {'double', 'single'}, optional
        The arithmetic asked for (SEP 33). ``"double"``, the default, lists every
        backend; ``"single"`` lists those that admit single precision
        (:data:`solprop.translate.precision.SINGLE_PRECISION_BACKENDS`).

    Returns
    -------
    tuple of str
        The usable names, in the table's order.

    Examples
    --------
    >>> from solprop.backends.registry import available
    >>> "numpy_ref" in available()
    True
    """
    names = SINGLE_PRECISION_BACKENDS if precision == "single" else tuple(BACKENDS)
    return tuple(name for name in names if _satisfied(BACKENDS[name]))


def resolve_name(name: str = "auto") -> str:
    """Resolve automatic selection to an installed backend's registry name.

    Parameters
    ----------
    name : str, optional
        ``"auto"`` is ``"native"``, and has no fallback: it refuses if
        ``solprop_native`` can't be found or this processor doesn't meet the installed
        build's CPU baseline. Other names pass through unchanged, but for a removed
        backend
        (:data:`REMOVED_BACKENDS`).

    Returns
    -------
    str
        Concrete selection to store in simulation settings. This uncached probe does
        not import a backend or initialize a native runtime. :func:`select` validates
        the name and constructs the backend; failures there never cause a fallback.

    Raises
    ------
    BackendError
        If ``name`` is a removed backend, or ``"auto"`` has nothing to select; the
        message says what to do.

    Examples
    --------
    >>> resolve_name("numpy_ref")
    'numpy_ref'
    """
    if name in REMOVED_BACKENDS:
        raise BackendError(REMOVED_BACKENDS[name])
    if name != "auto":
        return name
    if not _satisfied(BACKENDS[AUTO]):
        raise BackendError(_NATIVE_MISSING)
    refusal = _native_baseline_shortfall()
    if refusal is None:
        return AUTO
    raise BackendError(str(refusal)) from refusal


_NATIVE_BUILD_GUIDE: Final = (
    "https://solprop.readthedocs.io/en/stable/guide/performance/building-native.html"
)
"""The manual's page on building solprop-native from source."""

REQUIRE_DEVICE_SETTING: Final = "SOLPROP_REQUIRE_DEVICE"
"""Where a run must execute: ``cuda``, ``hip`` or ``any``; unset for no requirement."""

_REQUIRED_DEVICES: Final = {"cuda": {"cuda"}, "hip": {"hip"}, "any": {"cuda", "hip"}}
"""The device backends each value of the setting accepts."""

_NATIVE_MISSING = (
    'backend="auto" needs the native backend, and solprop-native is not installed. '
    "Install it with pip install solprop, or with uv sync in a checkout "
    "(https://solprop.readthedocs.io/en/stable/introduction/installation.html). "
    "A contributor's lightweight setup has none, so name the backend there, "
    'backend="numpy_ref"'
)
"""Why ``"auto"`` refuses with no native package (SEP 43 section 5)."""


def _device_build(name: str) -> bool:
    """Return whether a backend steps on an accelerator in this installation.

    Parameters
    ----------
    name : str
        A backend's name, or ``"auto"``.

    Returns
    -------
    bool
        True for a backend built for a device: ``native`` built for CUDA or HIP.
    """
    entry = BACKENDS.get(resolve_name(name))
    if entry is None or entry.device is None:
        return False
    return _devices(entry)[0] != "none"


def _devices(entry: BackendEntry) -> tuple[str, int]:
    """Return where a backend's installed build runs, from its row's ``device``.

    Parameters
    ----------
    entry : BackendEntry
        A row with a ``device``.

    Returns
    -------
    tuple of (str, int)
        The build's device backend, ``"none"`` on a CPU build, and the devices it sees.
    """
    info = cast("Callable[[], object]", _load(cast(str, entry.device)))()
    backend = cast(str, getattr(info, "device_backend"))  # noqa: B009 - any build's
    return backend, cast(int, getattr(info, "device_count"))  # noqa: B009


def _check_required_device(name: str) -> None:
    """Refuse a run that wouldn't meet :data:`REQUIRE_DEVICE_SETTING` (SEP 43).

    This reads the actual build, never a version label, and runs before any state is
    allocated: a job set up for a GPU should stop, not run for days on a CPU wheel that
    replaced its build.

    Parameters
    ----------
    name : str
        The resolved backend name.

    Raises
    ------
    BackendError
        If the value isn't an accepted one, or the backend or build doesn't meet it.
    """
    value = os.environ.get(REQUIRE_DEVICE_SETTING, "").strip()
    if not value:
        return
    accepted = _REQUIRED_DEVICES.get(value)
    if accepted is None:
        names = ", ".join(sorted(_REQUIRED_DEVICES))
        msg = f"{REQUIRE_DEVICE_SETTING}={value!r}: use one of {names}, or unset it"
        raise BackendError(msg)
    unset = f"unset {REQUIRE_DEVICE_SETTING} to run it anyway"
    entry = BACKENDS.get(name)
    if entry is None or entry.device is None:
        msg = (
            f"{REQUIRE_DEVICE_SETTING}={value} requires a native device build, and "
            f"this run is on {name}; {unset}"
        )
        raise BackendError(msg)
    if not _satisfied(entry):
        return  # select() goes on to say that the package is missing
    package = entry.device.partition(":")[0]
    shipped = entry.distribution or package
    try:
        module = importlib.import_module(package)
    except Exception as error:
        # A device build with no visible device stops in Kokkos as the package loads.
        msg = (
            f"{REQUIRE_DEVICE_SETTING}={value}: {shipped} did not load, so "
            f"this run wouldn't execute on a device: {error}"
        )
        raise BackendError(msg) from error

    device_backend, device_count = _devices(entry)
    if device_backend not in accepted:
        version = getattr(module, "__version__", "unknown")
        msg = (
            f"{REQUIRE_DEVICE_SETTING}={value} requires a device build, and the "
            f"installed {shipped} has device backend "
            f"{device_backend!r}, version {version}. A new release replaces a "
            f"source build with the CPU wheel, so rebuild it ({_NATIVE_BUILD_GUIDE}), "
            f"or {unset}"
        )
        raise BackendError(msg)
    if device_count < 1:
        msg = (
            f"{REQUIRE_DEVICE_SETTING}={value}: the {device_backend} build sees "
            f"no device; check the job's allocation, or {unset}"
        )
        raise BackendError(msg)


def _native_baseline_shortfall() -> ImportError | None:
    """Return why the installed native build can't run on this processor, if it can't.

    The package's ``_cpu.py`` is loaded by path, without importing the package (whose
    ``__init__`` refuses a processor below the baseline and loads the extension), and
    asked the question the package asks at import.

    Returns
    -------
    ImportError or None
        The package's own refusal, or ``None`` when the processor meets the baseline or
        the build records none to check.
    """
    spec = importlib.util.find_spec("solprop_native")
    locations = [] if spec is None else (spec.submodule_search_locations or [])
    for location in locations:
        source = Path(location) / "_cpu.py"
        if source.is_file():
            break
    else:
        return None
    return _baseline_probe(source).baseline_shortfall(source.parent)


def _baseline_probe(source: Path) -> ModuleType:
    """Load the native package's processor check from its file, as its own module.

    Parameters
    ----------
    source : pathlib.Path
        ``solprop_native/_cpu.py`` of the installed package.

    Returns
    -------
    types.ModuleType
        The module, under a private name so that it never stands in for the package's.
    """
    spec = importlib.util.spec_from_file_location("_solprop_native_cpu_probe", source)
    if spec is None or spec.loader is None:  # pragma: no cover - a .py file loads
        raise ImportError(f"cannot load the native processor check {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select(  # noqa: PLR0913 - the problem, its host context and its execution
    name: str,
    problem: PropagationProblem,
    request: RunRequest,
    *,
    grid: TimeGrid | None = None,
    modes: Sequence[Mode] | None = None,
    profiles: Mapping[str, ZProfile] | None = None,
    tdse_binding: TDSEBinding | None = None,
    options: ExecutionOptions | None = None,
) -> Backend:
    """Construct the named backend for a validated problem and its run's request.

    The backend receives the request where it is built (SEP 47 section 5.2): it
    checks the pair (:func:`solprop.translate.problem.validate_run`) and refuses an
    unsupported combination by name before it allocates, and it lowers the method,
    the error control, the itinerary, the planes and the statistics once. A session
    is then started on it with that request, adapted to one run
    (:meth:`~solprop.translate.run_request.RunRequest.for_run`).

    Parameters
    ----------
    name : str
        The backend's registry name, for example ``"numpy_ref"``, or ``"auto"``
        for the native backend, with no fallback (:func:`resolve_name`).
    problem : solprop.translate.problem.PropagationProblem
        The problem to step.
    request : solprop.translate.run_request.RunRequest
        The run's policies. The backend validates it with the problem again.
    grid : solprop.grid.TimeGrid, optional
        The grid the state lives on. Statistics need it and the kernel contract does not
        carry it, so it is passed alongside rather than inside the problem.
    modes : collections.abc.Sequence, optional
        The propagated modes, in state order, for the statistics that need them.
    profiles : collections.abc.Mapping, optional
        The z-profiles keyed by statistic name, for the profile statistics.
    tdse_binding : solprop.translate.tdse.TDSEBinding or None, optional
        Driver-owned host response binding for a TDSE problem on NumPy.
    options : solprop.backends.session.ExecutionOptions or None, optional
        The run's execution options, among them an explicit node budget, from which
        the backend plans its node chunk (SEP 47 section 6.5); ``None`` for the
        automatic choice.

    Returns
    -------
    Backend
        The constructed backend.

    Raises
    ------
    ValueError
        If ``name`` is not a backend this package knows; the message lists what is
        usable.
    BackendError
        If ``name`` is a backend that has been removed, or one that does not form a
        statistic the problem asks for; the message says what to use instead.
    ImportError
        If ``name`` is known but its dependencies are missing; the message says what
        to install.

    Examples
    --------
    >>> from solprop.backends.registry import BACKENDS
    >>> sorted(BACKENDS)
    ['native', 'numpy_ref']
    """
    name = resolve_name(name)
    first_use = BACKENDS[name].first_use if name in BACKENDS else None
    if first_use is not None:
        # SEP 46: a first native workspace in an allocation with no profile row runs
        # the quick calibration first, before the extension's runtime starts if it
        # has not yet.
        cast("Callable[[], None]", _load(first_use))()
    _check_required_device(name)
    if _BENCHMARK_GATED.get() and name != "numpy_ref":
        raise BackendError("the gated plasma benchmark requires numpy_ref")
    if (
        tdse_descriptor(problem.route) is not None or tdse_binding is not None
    ) and name != "numpy_ref":
        raise BackendError(f"{name} does not execute TDSE responses; use numpy_ref")
    refusal = statistics_refusal(name, request.observation.stats.names)
    if refusal is not None:
        raise BackendError(refusal)
    entry = BACKENDS.get(name)
    if entry is None:
        usable = ", ".join(available())
        msg = f"{name!r} is not a known backend; usable backends are: {usable}"
        raise ValueError(msg)
    if not _satisfied(entry):
        missing = ", ".join(
            module
            for module in entry.requires
            if importlib.util.find_spec(module) is None
        )
        install = (
            entry.install
            if entry.install is not None
            else " or ".join(f"solprop[{extra}]" for extra in entry.extras)
        )
        msg = (
            f"the {name!r} backend needs {missing}, which is not installed; install "
            f"{install}"
        )
        raise ImportError(msg)
    backend = _construct(
        name,
        problem,
        request,
        grid=grid,
        modes=modes,
        profiles=profiles,
        tdse_binding=tdse_binding,
        options=options,
    )
    logger.info("Using %s backend", backend.name)
    return backend


def select_batch(owners: Sequence[Backend]) -> BatchBackend:
    """Select batch arithmetic without substituting a different backend.

    This is the reference group's lockstep runner, read from the registry row's
    ``batch``. A backend whose row names none, as ``native``'s, runs its groups through
    its own prepared group instead (:func:`solprop.backends.sessions.start_group`;
    SEP 47 section 6.7), and is refused here by name.

    Parameters
    ----------
    owners : sequence of solprop.backends.protocol.Backend
        Compatible, loaded member owners.

    Returns
    -------
    BatchBackend
        Common fixed march using the selected backend's arithmetic.

    Raises
    ------
    BackendError
        If the members are not of one backend, or their backend has no lockstep runner
        (its groups start through ``start_group``).
    """
    if not owners:
        raise BackendError("Batch requires at least one loaded member")
    if any(owner.name != owners[0].name for owner in owners):
        raise BackendError("Batch refuses mixed backends")
    name = owners[0].name
    entry = BACKENDS.get(name)
    if entry is None:
        raise BackendError(f"Batch has no runner for backend {name!r}")
    if entry.batch is None:
        msg = (
            f"Batch has no runner for backend {name!r}: select_batch is the "
            "reference's lockstep runner, and a backend that registers none runs its "
            "groups through its own prepared group, which "
            "solprop.backends.sessions.start_group starts, as native's group session "
            "does (SEP 47 section 6.7)"
        )
        raise BackendError(msg)
    runner = cast("Callable[[Sequence[Backend]], BatchBackend]", _load(entry.batch))
    return runner(owners)


def _satisfied(entry: BackendEntry) -> bool:
    """Return whether every module an entry requires can be found.

    Parameters
    ----------
    entry : BackendEntry
        The table row.

    Returns
    -------
    bool
        True when nothing is missing.
    """
    return all(
        importlib.util.find_spec(module) is not None for module in entry.requires
    )


def _construct(  # noqa: PLR0913 - as select
    name: str,
    problem: PropagationProblem,
    request: RunRequest,
    *,
    grid: TimeGrid | None,
    modes: Sequence[Mode] | None,
    profiles: Mapping[str, ZProfile] | None,
    tdse_binding: TDSEBinding | None = None,
    options: ExecutionOptions | None = None,
) -> Backend:
    """Import and build the named backend, whose requirements are already met.

    Parameters
    ----------
    name : str
        The backend's registry name.
    problem : solprop.translate.problem.PropagationProblem
        The problem to step.
    request : solprop.translate.run_request.RunRequest
        The run's policies.
    grid : solprop.grid.TimeGrid or None
        The grid the state lives on.
    modes : collections.abc.Sequence or None
        The propagated modes.
    profiles : collections.abc.Mapping or None
        The z-profiles keyed by statistic name.
    tdse_binding : TDSEBinding or None, optional
        Driver-owned atomic runtime, passed only to the NumPy backend.
    options : solprop.backends.session.ExecutionOptions or None, optional
        The run's execution options.

    Returns
    -------
    Backend
        The constructed backend.

    Raises
    ------
    ImportError
        If the name is known and satisfied but no module implements it yet.
    """
    entry = BACKENDS.get(name)
    if entry is None:
        raise AssertionError(f"unhandled backend registry entry {name!r}")
    # Imported here, not at module scope: selecting a backend must not be a reason to
    # import every backend or initialize a native runtime.
    factory = cast("Callable[..., Backend]", _load(entry.factory))
    # Only the reference is given a TDSE binding; select refuses one elsewhere.
    host = {} if tdse_binding is None else {"tdse_binding": tdse_binding}
    return factory(
        problem,
        request,
        grid=grid,
        modes=modes,
        profiles=profiles,
        options=options,
        **host,
    )
