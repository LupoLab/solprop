# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Selecting one of the two backends by name, and saying why one is not usable.

``docs/dev/backends.md`` section 1: there are exactly two backends (ADR-077),
``native`` and ``numpy_ref``. Selection is by name; availability is detected with
:func:`importlib.util.find_spec` and **never cached**. A name that is neither, a
retired backend's included, is a :class:`~solprop.exceptions.BackendError` naming the
two; ``native`` without its package is an :class:`ImportError` saying what to install.

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
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest
from solprop.translate.tdse import TDSEBinding

__all__ = [
    "AUTO",
    "BACKEND_NAMES",
    "REQUIRE_DEVICE_SETTING",
    "available",
    "resolve_name",
    "select",
    "select_batch",
    "statistics_refusal",
]

logger = logging.getLogger(__name__)

BACKEND_NAMES: Final = ("numpy_ref", "native")
"""The two backends (ADR-077): the numerical reference and the native backend."""

AUTO: Final = "native"
"""What ``"auto"`` selects, with no fallback (SEP 43)."""

_NATIVE_PACKAGE: Final = "solprop_native"
"""The native backend's import package, in the ``solprop-native`` distribution."""

_UNKNOWN: Final = (
    "the backends are backend='native', the solprop-native package, and "
    "backend='numpy_ref', the reference, which is unthreaded and far slower. Output "
    "and snapshots that record another backend still load"
)
"""What a name that is neither backend is told to use instead."""

_NATIVE_MISSING: Final = (
    'backend="auto" needs the native backend, and solprop-native is not installed. '
    "Install it with pip install solprop, or with uv sync in a checkout "
    "(https://solprop.readthedocs.io/en/stable/introduction/installation.html). "
    "A contributor's lightweight setup has none, so name the backend there, "
    'backend="numpy_ref"'
)
"""Why ``"auto"`` refuses with no native package (SEP 43 section 5)."""

_NATIVE_INSTALL: Final = (
    "the 'native' backend needs solprop_native, which is not installed; install "
    "solprop, or build solprop-native from native/ in the solprop repository"
)
"""What ``native`` says when its package is missing."""

_NATIVE_BUILD_GUIDE: Final = (
    "https://solprop.readthedocs.io/en/stable/guide/performance/building-native.html"
)
"""The manual's page on building solprop-native from source."""

REQUIRE_DEVICE_SETTING: Final = "SOLPROP_REQUIRE_DEVICE"
"""Where a run must execute: ``cuda``, ``hip`` or ``any``; unset for no requirement."""

_REQUIRED_DEVICES: Final = {"cuda": {"cuda"}, "hip": {"hip"}, "any": {"cuda", "hip"}}
"""The device backends each value of the setting accepts."""


def _native_installed() -> bool:
    """Return whether the native package can be found, without importing it."""
    return importlib.util.find_spec(_NATIVE_PACKAGE) is not None


def statistics_refusal(name: str, names: Sequence[str]) -> str | None:
    """Return why a backend cannot form some of the statistics named, or ``None``.

    Asking about the native backend imports the installed package, as selecting it
    would, to read the far-field statistics its class declares it forms.

    Parameters
    ----------
    name : str
        The resolved backend name.
    names : collections.abc.Sequence of str
        The statistics asked for.

    Returns
    -------
    str or None
        The refusal, naming the backend that forms them, or ``None`` when the backend
        forms every one (or is not installed, which selecting it reports).

    Examples
    --------
    >>> from solprop.backends.registry import statistics_refusal
    >>> statistics_refusal("numpy_ref", ["energy", "beam_spectrum"]) is None
    True
    """
    asked = [one for one in names if one in FAR_FIELD_STATISTICS]
    # numpy_ref forms every far-field statistic.
    if not asked or name != "native" or not _native_installed():
        return None
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeBackend,
    )

    formed = cast(Collection[str], NativeBackend.far_field_statistics)
    missing = [one for one in asked if one not in formed]
    if not missing:
        return None
    return (
        f"the installed native backend does not form the far-field statistics "
        f"{missing}; install a solprop-native build that does, or run them on "
        "numpy_ref"
    )


def available() -> tuple[str, ...]:
    """Return the names of the backends that can run here and now.

    Detection is by :func:`importlib.util.find_spec`, so nothing is imported and nothing
    is cached: a native package installed while the process runs becomes available
    without a restart, and initializing a native runtime is never a side effect of
    asking.

    Returns
    -------
    tuple of str
        The usable names, in the order of :data:`BACKEND_NAMES`.

    Examples
    --------
    >>> from solprop.backends.registry import available
    >>> "numpy_ref" in available()
    True
    """
    return tuple(
        name for name in BACKEND_NAMES if name != "native" or _native_installed()
    )


def resolve_name(name: str = "auto") -> str:
    """Resolve automatic selection to an installed backend's name.

    Parameters
    ----------
    name : str, optional
        ``"auto"`` is ``"native"``, and has no fallback: it refuses if
        ``solprop_native`` can't be found or this processor doesn't meet the installed
        build's CPU baseline. ``"native"`` and ``"numpy_ref"`` pass through unchanged.

    Returns
    -------
    str
        Concrete selection to store in simulation settings. This uncached probe does
        not import a backend or initialize a native runtime. :func:`select` constructs
        the backend; failures there never cause a fallback.

    Raises
    ------
    BackendError
        If ``name`` is neither backend (a retired backend's name included), or
        ``"auto"`` has nothing to select; the message says what to do.

    Examples
    --------
    >>> resolve_name("numpy_ref")
    'numpy_ref'
    """
    if name in BACKEND_NAMES:
        return name
    if name != "auto":
        raise BackendError(f"{name!r} is not a backend: {_UNKNOWN}")
    if not _native_installed():
        raise BackendError(_NATIVE_MISSING)
    refusal = _native_baseline_shortfall()
    if refusal is None:
        return AUTO
    raise BackendError(str(refusal)) from refusal


def _device_build(name: str) -> bool:
    """Return whether a backend steps on an accelerator in this installation."""
    if resolve_name(name) != "native" or not _native_installed():
        return False
    return _devices()[0] != "none"


def _devices() -> tuple[str, int]:
    """Return where the installed native build runs: its device backend and count.

    The device backend is ``"none"`` on a CPU build.
    """
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        build_info,
    )

    info = cast(object, build_info())
    backend = cast(str, getattr(info, "device_backend"))  # noqa: B009 - any build's
    return backend, cast(int, getattr(info, "device_count"))  # noqa: B009


def _check_required_device(name: str) -> None:
    """Refuse a run that wouldn't meet :data:`REQUIRE_DEVICE_SETTING` (SEP 43).

    This reads the actual build, never a version label, and runs before any state is
    allocated: a job set up for a GPU should stop, not run for days on a CPU wheel that
    replaced its build.
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
    if name != "native":
        msg = (
            f"{REQUIRE_DEVICE_SETTING}={value} requires a native device build, and "
            f"this run is on {name}; {unset}"
        )
        raise BackendError(msg)
    if not _native_installed():
        return  # select() goes on to say that the package is missing
    try:
        module = importlib.import_module(_NATIVE_PACKAGE)
    except Exception as error:
        # A device build with no visible device stops in Kokkos as the package loads.
        msg = (
            f"{REQUIRE_DEVICE_SETTING}={value}: solprop-native did not load, so "
            f"this run wouldn't execute on a device: {error}"
        )
        raise BackendError(msg) from error

    device_backend, device_count = _devices()
    if device_backend not in accepted:
        version = getattr(module, "__version__", "unknown")
        msg = (
            f"{REQUIRE_DEVICE_SETTING}={value} requires a device build, and the "
            f"installed solprop-native has device backend "
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

    - Returns: the package's own refusal, or ``None`` when the processor meets the
      baseline or the build records none to check.
    """
    spec = importlib.util.find_spec(_NATIVE_PACKAGE)
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

    Under a private name, so that it never stands in for the package's own module.
    """
    spec = importlib.util.spec_from_file_location("_solprop_native_cpu_probe", source)
    if spec is None or spec.loader is None:  # pragma: no cover - a .py file loads
        raise ImportError(f"cannot load the native processor check {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select(
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
        ``"native"``, ``"numpy_ref"``, or ``"auto"`` for the native backend, with no
        fallback (:func:`resolve_name`).
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
        Driver-owned host response binding for a TDSE problem, on either backend
        (SEP 49).
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
    BackendError
        If ``name`` is neither backend, or the backend does not run what the problem
        asks for; the message says what to use instead.
    ImportError
        If ``name`` is ``"native"`` and its package is missing; the message says what
        to install.

    Examples
    --------
    >>> from solprop.backends.registry import BACKEND_NAMES
    >>> BACKEND_NAMES
    ('numpy_ref', 'native')
    """
    name = resolve_name(name)
    if name == "native":
        # SEP 46: a first native workspace in an allocation with no profile row runs
        # the quick calibration first, before the extension's runtime starts if it
        # has not yet.
        from solprop.calibration.first_use import ensure  # noqa: PLC0415 - layer above

        ensure()
    _check_required_device(name)
    refusal = statistics_refusal(name, request.observation.stats.names)
    if refusal is not None:
        raise BackendError(refusal)
    if name == "native" and not _native_installed():
        raise ImportError(_NATIVE_INSTALL)
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

    This is the reference group's lockstep runner. The native backend runs its groups
    through its own prepared group instead
    (:func:`solprop.backends.sessions.start_group`; SEP 47 section 6.7), and is
    refused here by name.

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
    if name != "numpy_ref":
        msg = (
            f"Batch has no runner for backend {name!r}: select_batch is the "
            "reference's lockstep runner, and native runs its groups through its own "
            "prepared group, which solprop.backends.sessions.start_group starts "
            "(SEP 47 section 6.7)"
        )
        raise BackendError(msg)
    from solprop.backends.numpy_ref.batch import lockstep_runner  # noqa: PLC0415

    return lockstep_runner(owners)


def _construct(
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
    """Import and build the named backend, whose package is installed.

    Imported here, not at module scope: selecting a backend must not be a reason to
    import the other or initialize a native runtime. A TDSE binding goes to either
    backend, which refuses a problem without its binding by name.
    """
    factory: Callable[..., Backend]
    if name == "native":
        from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            NativeBackend,
        )

        factory = cast("Callable[..., Backend]", NativeBackend)
    else:
        from solprop.backends.numpy_ref import construct  # noqa: PLC0415

        factory = construct
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
