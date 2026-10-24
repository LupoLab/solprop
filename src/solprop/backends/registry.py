# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Selecting a backend by name, and saying why one is not usable.

``docs/dev/backends.md`` section 1: selection is by name, availability is detected with
:func:`importlib.util.find_spec` and **never cached**. The
errors are two-tier, because the two failures need different answers from the reader: a
name this package has never heard of is a :class:`ValueError` listing what is usable,
and a known name whose dependencies are missing is an :class:`ImportError` naming the
extra to install.

The native backend is a second distribution, ``solprop-native``, built from ``native/``
in this repository and installed with the ``native`` extra; probing availability does
not import it. Automatic selection prefers it when installed and otherwise chooses Numba
CPU.
Import, compatibility and construction errors remain visible; they never trigger a
fallback to different arithmetic, with one exception. A distribution build of
solprop-native is compiled for a CPU baseline, and on a processor below it automatic
selection chooses Numba CPU with a :class:`~solprop.exceptions.SolpropWarning`, once per
process. It finds that out without importing the package: it loads the package's own
pure-Python processor check (``solprop_native/_cpu.py``) by path and runs it, so the
check that refuses the import and the one that chooses the fallback are one code. An
explicit ``backend="native"`` still raises the package's refusal.
"""

import importlib.util
import logging
import warnings
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

from solprop.backends.protocol import Backend, BatchBackend
from solprop.exceptions import BackendError, SolpropWarning
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.profiles import ZProfile
from solprop.spatial import FAR_FIELD_STATISTICS
from solprop.translate.plasma_source import _BENCHMARK_GATED
from solprop.translate.precision import SINGLE_PRECISION_BACKENDS
from solprop.translate.problem import KernelProblem
from solprop.translate.tdse import TDSEBinding

__all__ = [
    "BACKENDS",
    "FAR_FIELD_BACKENDS",
    "BackendEntry",
    "available",
    "available_device_memory",
    "resolve_name",
    "select",
    "select_batch",
    "statistics_refusal",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendEntry:
    """What a backend needs before it can be constructed.

    Attributes
    ----------
    requires : tuple of str
        Importable module names that must be present. Empty for a backend that needs
        nothing beyond the package's own dependencies.
    extras : tuple of str
        The optional-dependency groups that supply ``requires``, for the error message.
    """

    requires: tuple[str, ...]
    extras: tuple[str, ...]


BACKENDS: Mapping[str, BackendEntry] = {
    "numpy_ref": BackendEntry(requires=(), extras=()),
    # numba and pyfftw are core dependencies, so this entry is always satisfied; it
    # names them so that a wheel built without them says which one is missing rather
    # than raising an ImportError from inside the backend.
    "numba_cpu": BackendEntry(requires=("numba", "pyfftw"), extras=()),
    "native": BackendEntry(requires=("solprop_native",), extras=()),
    # Apple-silicon GPUs through MLX, single precision only (SEP 33).
    "mlx": BackendEntry(requires=("mlx",), extras=("mlx",)),
}
"""The backends this package knows, by name."""

SINGLE_ONLY: frozenset[str] = frozenset({"mlx"})
"""The backends that compute in binary32 only (SEP 33): Apple GPUs have no binary64."""

FAR_FIELD_BACKENDS: tuple[str, ...] = ("numpy_ref", "numba_cpu", "native")
"""The backends that form the far-field statistics of SEP 23.

``numpy_ref`` and ``numba_cpu`` always do; the native backend does from SEP 23's third
phase, which a build declares by its backend's ``far_field_statistics`` attribute. An
older build is refused them by name on this side of the seam
(:func:`statistics_refusal`), before it is built.
"""


def statistics_refusal(name: str, names: Sequence[str]) -> str | None:
    """Return why a backend cannot form some of the statistics named, or ``None``.

    Asking about the native backend for a far-field statistic imports the installed
    plugin, as selecting it would, to read which far-field statistics it declares.

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
    >>> statistics_refusal("numba_cpu", ["energy", "beam_spectrum"]) is None
    True
    """
    asked = [one for one in names if one in FAR_FIELD_STATISTICS]
    # Single precision refuses the far-field statistics in validation (SEP 33).
    if not asked or name in ("numpy_ref", "numba_cpu", "mlx"):
        return None
    formed = _plugin_far_field(name)
    if formed is None:
        return None
    missing = [one for one in asked if one not in formed]
    if not missing:
        return None
    return (
        f"the installed {name!r} backend does not form the far-field statistics "
        f"{missing}; install a solprop-native build that does, or run them on "
        "numpy_ref or numba_cpu"
    )


def _plugin_far_field(name: str) -> frozenset[str] | None:
    """Return the far-field statistics an installed plugin backend declares.

    Parameters
    ----------
    name : str
        A registry name other than the in-tree backends.

    Returns
    -------
    frozenset of str or None
        What the plugin's backend class declares in ``far_field_statistics`` (empty for
        a build that predates the attribute), or ``None`` when the name is not a known,
        installed plugin.
    """
    entry = BACKENDS.get(name)
    if name != "native" or entry is None or not _satisfied(entry):
        return None
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeBackend,
    )

    declared = getattr(NativeBackend, "far_field_statistics", frozenset())
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
        The arithmetic asked for (SEP 33). ``"double"``, the default, lists the binary64
        backends, which leaves out ``mlx``; ``"single"`` lists those that admit single
        precision (:data:`solprop.translate.precision.SINGLE_PRECISION_BACKENDS`).

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
    if precision == "single":
        names = SINGLE_PRECISION_BACKENDS
    else:
        names = tuple(name for name in BACKENDS if name not in SINGLE_ONLY)
    return tuple(name for name in names if _satisfied(BACKENDS[name]))


def metal_available() -> bool:
    """Return whether the ``mlx`` backend can run on a Metal GPU here (SEP 33, Q8).

    Imports MLX only when it is installed: asking on another platform imports nothing.

    Returns
    -------
    bool
        True when MLX is installed and reports a Metal device.
    """
    if not _satisfied(BACKENDS["mlx"]):
        return False
    import mlx.core as mx  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    import solprop.backends.mlx_gpu  # noqa: F401, PLC0415 - sets MLX_ENABLE_TF32 first

    return bool(mx.metal.is_available())


def resolve_name(name: str = "auto") -> str:
    """Resolve automatic selection to an installed backend's registry name.

    Parameters
    ----------
    name : str, optional
        ``"auto"`` prefers ``"native"`` when ``solprop_native`` can be found and this
        processor meets the installed build's CPU baseline, otherwise
        ``"numba_cpu"``. Explicit names pass through unchanged.

    Returns
    -------
    str
        Concrete selection to store in simulation settings. This uncached probe does
        not import a backend or initialize a native runtime. :func:`select` validates
        the name and constructs the backend; failures there never cause a fallback.

    Warns
    -----
    SolpropWarning
        Once per process, when ``"auto"`` passes over an installed native build
        because this processor lacks features of its CPU baseline.

    Examples
    --------
    >>> resolve_name("numpy_ref")
    'numpy_ref'
    """
    if name != "auto":
        return name
    if not _satisfied(BACKENDS["native"]):
        return "numba_cpu"
    shortfall = _native_baseline_shortfall()
    if shortfall is None:
        return "native"
    _warn_baseline_fallback(*shortfall)
    return "numba_cpu"


_NATIVE_BUILD_GUIDE = (
    "https://solprop.readthedocs.io/en/latest/guide/performance/building-native.html"
)
"""The manual's page on building solprop-native from source."""

_ISSUES = "https://github.com/LupoLab/solprop/issues"
"""Where users report problems, and processors below the native wheels' baseline."""

_warned_baseline_fallback = False


def _native_baseline_shortfall() -> tuple[str, tuple[str, ...]] | None:
    """Return the installed native build's baseline and what this processor lacks.

    The package's ``_cpu.py`` is loaded by path, without importing the package (whose
    ``__init__`` refuses a processor below the baseline and loads the extension), and
    asked the question the package asks at import.

    Returns
    -------
    tuple of (str, tuple of str) or None
        The baseline and the missing features, or ``None`` when the processor meets
        the baseline, cannot be checked, or the build records no baseline.
    """
    spec = importlib.util.find_spec("solprop_native")
    locations = [] if spec is None else (spec.submodule_search_locations or [])
    for location in locations:
        source = Path(location) / "_cpu.py"
        if source.is_file():
            break
    else:
        return None
    error = _baseline_probe(source).baseline_shortfall(source.parent)
    if error is None:
        return None
    return error.baseline, tuple(error.missing_features)


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


def _warn_baseline_fallback(baseline: str, missing: tuple[str, ...]) -> None:
    """Warn, once per process, that automatic selection passed over solprop-native.

    Parameters
    ----------
    baseline : str
        The installed build's CPU baseline.
    missing : tuple of str
        The baseline's features this processor lacks.
    """
    global _warned_baseline_fallback  # noqa: PLW0603 - once per process, by design
    if _warned_baseline_fallback:
        return
    _warned_baseline_fallback = True
    logger.info(
        "backend='auto' chooses numba_cpu: this processor is below solprop-native's "
        "%s baseline",
        baseline,
    )
    msg = (
        f"this processor lacks {', '.join(missing)}, required by the installed "
        f"solprop-native build (CPU baseline {baseline}), so solprop is using the "
        "numba_cpu backend instead. To use the native backend on this machine, build "
        f"solprop-native from source for it ({_NATIVE_BUILD_GUIDE}). Please also open "
        f"an issue at {_ISSUES} so that the maintainers know such processors are in use"
    )
    warnings.warn(msg, SolpropWarning, stacklevel=3)


def available_device_memory(name: str) -> int | None:
    """Return a backend's free accelerator bytes before it is constructed.

    The driver admits a run's persistent memory against them before the backend
    allocates (:func:`solprop.driver.memory.admit_memory`). Only the native backend
    reports them, and only a device build; asking imports the plugin, as selecting it
    would.

    Parameters
    ----------
    name : str
        A concrete registry name.

    Returns
    -------
    int or None
        Free bytes on the selected accelerator, or ``None`` for a host backend, a host
        build or a plugin that does not report them.

    Examples
    --------
    >>> from solprop.backends.registry import available_device_memory
    >>> available_device_memory("numpy_ref") is None
    True
    """
    if name != "native" or not _satisfied(BACKENDS["native"]):
        return None
    import solprop_native  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    probe = getattr(solprop_native, "available_device_memory", None)
    return None if probe is None else probe()


def select(
    name: str,
    problem: KernelProblem,
    *,
    grid: TimeGrid | None = None,
    modes: Sequence[Mode] | None = None,
    profiles: Mapping[str, ZProfile] | None = None,
    tdse_binding: TDSEBinding | None = None,
) -> Backend:
    """Construct the named backend for a validated kernel problem.

    Parameters
    ----------
    name : str
        The backend's registry name, for example ``"numpy_ref"``, or ``"auto"``
        to prefer the installed native backend and otherwise use Numba CPU.
    problem : solprop.translate.problem.KernelProblem
        The problem to step. The backend validates it again.
    grid : solprop.grid.TimeGrid, optional
        The grid the state lives on. Statistics need it and the kernel contract does not
        carry it, so it is passed alongside rather than inside the problem.
    modes : collections.abc.Sequence, optional
        The propagated modes, in state order, for the statistics that need them.
    profiles : collections.abc.Mapping, optional
        The z-profiles keyed by statistic name, for the profile statistics.
    tdse_binding : solprop.translate.tdse.TDSEBinding or None, optional
        Driver-owned host response binding for a TDSE problem on NumPy.

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
        If ``name`` is known but its dependencies are missing; the message names the
        extra to install.

    Examples
    --------
    >>> from solprop.backends.registry import BACKENDS
    >>> sorted(BACKENDS)
    ['mlx', 'native', 'numba_cpu', 'numpy_ref']
    """
    name = resolve_name(name)
    if _BENCHMARK_GATED.get() and name != "numpy_ref":
        raise BackendError("the gated plasma benchmark requires numpy_ref")
    if (problem.tdse is not None or tdse_binding is not None) and name != "numpy_ref":
        raise BackendError(f"{name} does not execute TDSE responses; use numpy_ref")
    refusal = statistics_refusal(name, problem.stats_spec.names)
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
            "solprop[native], or build solprop-native from native/ in the solprop "
            "repository"
            if name == "native"
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
        grid=grid,
        modes=modes,
        profiles=profiles,
        tdse_binding=tdse_binding,
    )
    logger.info("Using %s backend", backend.name)
    return backend


def select_batch(owners: Sequence[Backend]) -> BatchBackend:
    """Select batch arithmetic without substituting a different backend.

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
        If the backend does not expose the shared step-driver contract.
    """
    if not owners:
        raise BackendError("Batch requires at least one loaded member")
    if any(owner.name != owners[0].name for owner in owners):
        raise BackendError("Batch refuses mixed backends")
    name = owners[0].name
    if name == "numpy_ref":
        from solprop.backends.numpy_ref.backend import (  # noqa: PLC0415 - lazy backend
            NumpyRefBackend,
        )
        from solprop.backends.numpy_ref.batch import (  # noqa: PLC0415 - lazy backend
            NumpyBatchBackend,
        )

        return NumpyBatchBackend(cast(Sequence[NumpyRefBackend], owners))
    if name == "mlx":
        from solprop.backends.mlx_gpu.backend import (  # noqa: PLC0415 - lazy backend
            MlxBackend,
        )
        from solprop.backends.mlx_gpu.batch import (  # noqa: PLC0415 - lazy backend
            MlxBatchRunner,
        )

        return MlxBatchRunner(cast(Sequence[MlxBackend], owners))
    if name == "numba_cpu":
        from solprop.backends.numba_cpu.backend import (  # noqa: PLC0415 - lazy backend
            NumbaCpuBackend,
        )
        from solprop.backends.numba_cpu.batch import (  # noqa: PLC0415 - lazy backend
            NumbaCpuBatchBackend,
        )

        return NumbaCpuBatchBackend(cast(Sequence[NumbaCpuBackend], owners))
    if name == "native":
        # The optional native package has its own strict type gate.
        from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            NativeBackend,
        )
        from solprop_native.batch import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            NativeBatchBackend,
        )

        return cast(
            BatchBackend, NativeBatchBackend(cast(Sequence[NativeBackend], owners))
        )
    raise BackendError(f"Batch has no runner for backend {name!r}")


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


def _construct(
    name: str,
    problem: KernelProblem,
    *,
    grid: TimeGrid | None,
    modes: Sequence[Mode] | None,
    profiles: Mapping[str, ZProfile] | None,
    tdse_binding: TDSEBinding | None = None,
) -> Backend:
    """Import and build the named backend, whose requirements are already met.

    Parameters
    ----------
    name : str
        The backend's registry name.
    problem : solprop.translate.problem.KernelProblem
        The problem to step.
    grid : solprop.grid.TimeGrid or None
        The grid the state lives on.
    modes : collections.abc.Sequence or None
        The propagated modes.
    profiles : collections.abc.Mapping or None
        The z-profiles keyed by statistic name.
    tdse_binding : TDSEBinding or None, optional
        Driver-owned atomic runtime, passed only to the NumPy backend.

    Returns
    -------
    Backend
        The constructed backend.

    Raises
    ------
    ImportError
        If the name is known and satisfied but no module implements it yet.
    """
    if name == "numpy_ref":
        # Imported here, not at module scope: selecting a backend must not be a
        # reason to import every backend or initialize a native runtime.
        from solprop.backends.numpy_ref import NumpyRefBackend  # noqa: PLC0415

        if problem.precision.single:
            from solprop.backends.numpy_ref.single import (  # noqa: PLC0415
                NumpySingleBackend,
            )

            return NumpySingleBackend(
                problem,
                grid=grid,
                modes=modes,
                profiles=profiles,
                tdse_binding=tdse_binding,
            )
        return NumpyRefBackend(
            problem,
            grid=grid,
            modes=modes,
            profiles=profiles,
            tdse_binding=tdse_binding,
        )
    if name == "numba_cpu":
        from solprop.backends.numba_cpu import NumbaCpuBackend  # noqa: PLC0415

        return NumbaCpuBackend(problem, grid=grid, modes=modes, profiles=profiles)
    if name == "native":
        from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            NativeBackend,
        )

        # The optional package has its own strict type gate; it is deliberately absent
        # from this repository's static-analysis source tree.
        return cast(
            Backend,
            NativeBackend(problem, grid=grid, modes=modes, profiles=profiles),
        )
    if name == "mlx":
        from solprop.backends.mlx_gpu.backend import MlxBackend  # noqa: PLC0415

        return MlxBackend(problem, grid=grid, modes=modes, profiles=profiles)
    raise AssertionError(f"unhandled backend registry entry {name!r}")
