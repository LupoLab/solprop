# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The pyFFTW import, made so that its OpenMP threads sleep when they are idle.

pyFFTW's Linux wheels run FFTW's threads through their own copy of GNU OpenMP
(``libgomp``). That runtime reads ``OMP_WAIT_POLICY`` once, when it is loaded. With the
variable unset, a thread that finishes a parallel region spins for GNU's default count
before it sleeps. The kernels' Numba pool runs next on the same cores, so the spinning
FFTW threads hold cores that the kernels' threads want. The kernels' threads then hold
the cores that FFTW's team needs for its next transform, and each transform waits for
the scheduler.

**Measured** on the four-mode RDW capillary (the suite's ``rdw``), on 8 cores of a
two-socket x86-64 server with the Numba pool at 8 on its TBB layer: with FFTW's pool at
8 threads a planned transform took 3.5-5.9 ms in place of the 0.1 ms it takes at one
thread, and a step 129 ms in place of 25 ms. At 2 and 4 FFTW threads the step took
128 ms. With ``OMP_WAIT_POLICY=passive`` it took 25.4 ms at 8 FFTW threads. On the full
CtC benchmark window, whose transforms get four FFTW threads by the backend's own size
rule, the passive policy took the window from 6.92 to 4.85 s. An Apple-silicon laptop,
whose pyFFTW wheel uses LLVM's OpenMP, showed no such slowdown, and this module leaves
other platforms alone.

So on Linux the import sets ``OMP_WAIT_POLICY=passive`` for pyFFTW's runtime and
restores the environment straight away. An OpenMP runtime that loads later reads the
user's environment, not this setting: the native backend's runtime, or Numba's OpenMP
layer. It is not done when the user has set ``OMP_WAIT_POLICY`` or ``GOMP_SPINCOUNT``,
since then that choice is theirs. It is not done either when ``pyfftw`` was imported
before this module, because its runtime has then already read the environment.
:data:`WAIT_POLICY` says which of these happened, and the backend's metadata records
it as ``fftw_wait_policy``.
"""

import contextlib
import os
import sys
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path

from solprop.exceptions import SolpropWarning

__all__ = [
    "BINDING",
    "BINDING_VARIABLES",
    "VARIABLES",
    "WAIT_POLICY",
    "binding_policy",
    "numba_launch",
    "pyfftw",
    "wait_policy",
]

VARIABLES = ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT")
"""The variables by which a user chooses how GNU OpenMP's idle threads wait."""


def wait_policy(platform: str, environ: Mapping[str, str], *, loaded: bool) -> str:
    """Return how pyFFTW's OpenMP threads will wait, and so whether to set it.

    Parameters
    ----------
    platform : str
        ``sys.platform``.
    environ : Mapping
        The process environment.
    loaded : bool
        Whether ``pyfftw`` is already imported, in which case its runtime has read the
        environment already.

    Returns
    -------
    str
        ``"passive"`` when this import should set it, which is on Linux only and only
        when nothing else decides it. Otherwise it describes what decides it:
        ``"inherited"`` when pyFFTW was imported first, ``"environment: NAME=value"``
        when the user set one of :data:`VARIABLES`, or ``"runtime default"`` on other
        platforms.

    Examples
    --------
    >>> from solprop.backends.numba_cpu.openmp import wait_policy
    >>> wait_policy("linux", {}, loaded=False)
    'passive'
    >>> wait_policy("linux", {"OMP_WAIT_POLICY": "active"}, loaded=False)
    'environment: OMP_WAIT_POLICY=active'
    """
    if loaded:
        return "inherited"
    chosen = [f"{name}={environ[name]}" for name in VARIABLES if name in environ]
    if chosen:
        return "environment: " + ", ".join(chosen)
    return "passive" if platform == "linux" else "runtime default"


BINDING_VARIABLES = ("OMP_PROC_BIND", "OMP_PLACES")
"""The variables that make GNU OpenMP bind its threads (``OMP_PLACES`` alone does)."""


def binding_policy(platform: str, environ: Mapping[str, str], *, loaded: bool) -> str:
    """Return whether a runtime about to load would bind its threads.

    On Linux, GNU OpenMP binds its threads, the main thread among them, when
    ``OMP_PROC_BIND`` is set to anything but ``false`` or when ``OMP_PLACES`` is set.
    For ``numba_cpu``'s runtimes that is ruinous: the comb-to-continuum window took
    45-47 s with binding against 17 s without, on 64 Zen 5 cores, whatever Numba's
    threading layer (SEP 28), because the bound runtimes stack their threads on the
    same few cores. Binding remains the user's choice for the native backend's runtime.

    Parameters
    ----------
    platform : str
        ``sys.platform``.
    environ : Mapping
        The process environment.
    loaded : bool
        Whether the runtime has already read the environment.

    Returns
    -------
    str
        ``"unbound"`` when the load should set ``OMP_PROC_BIND=false``, which is on
        Linux only and only when binding is asked for; ``"inherited: ..."`` when the
        runtime is loaded already under binding; ``"none"`` when nothing asks for it.

    Examples
    --------
    >>> from solprop.backends.numba_cpu.openmp import binding_policy
    >>> binding_policy("linux", {"OMP_PROC_BIND": "close"}, loaded=False)
    'unbound'
    >>> binding_policy("linux", {"OMP_PROC_BIND": "false"}, loaded=False)
    'none'
    """
    asked = [
        f"{name}={environ[name]}"
        for name in BINDING_VARIABLES
        if name in environ and environ[name].strip().lower() not in {"", "false"}
    ]
    if not asked or platform != "linux":
        return "none"
    if loaded:
        return "inherited: " + ", ".join(asked)
    return "unbound"


@contextlib.contextmanager
def _environment(changes: Mapping[str, str]) -> Iterator[None]:
    """Set variables for the duration of the block, then restore their old values.

    Parameters
    ----------
    changes : Mapping
        Environment variable names and temporary string values.

    Yields
    ------
    None
        Control while the temporary environment applies.
    """
    previous = {name: os.environ.get(name) for name in changes}
    os.environ.update(changes)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _changes(wait: str, binding: str) -> dict[str, str]:
    """Return the environment overrides for a runtime's wait and binding policies.

    Parameters
    ----------
    wait : str
        Result of :func:`wait_policy`.
    binding : str
        Result of :func:`binding_policy`.

    Returns
    -------
    dict
        Environment overrides needed before loading the runtime.
    """
    changes = {"OMP_WAIT_POLICY": "passive"} if wait == "passive" else {}
    if binding == "unbound":
        changes["OMP_PROC_BIND"] = "false"
    return changes


WAIT_POLICY = wait_policy(sys.platform, os.environ, loaded="pyfftw" in sys.modules)
"""How pyFFTW's OpenMP threads wait in this process (:func:`wait_policy`)."""

BINDING = binding_policy(sys.platform, os.environ, loaded="pyfftw" in sys.modules)
"""Whether pyFFTW's OpenMP threads are bound here (:func:`binding_policy`)."""

with _environment(_changes(WAIT_POLICY, BINDING)):
    # The runtime reads the variables when the extension's libraries load, so the
    # import happens inside the block.
    import pyfftw

if BINDING.startswith("inherited"):  # pragma: no cover - Linux only
    warnings.warn(
        f"pyfftw was imported before solprop under OpenMP binding ({BINDING[11:]}); "
        "its FFTW threads stay bound, which made numba_cpu runs 3 to 9 times slower. "
        "Import solprop first, or unset OMP_PROC_BIND and OMP_PLACES.",
        SolpropWarning,
        stacklevel=2,
    )


def _gnu_openmp_loaded(maps: Path = Path("/proc/self/maps")) -> bool:
    """Return whether the shared GNU OpenMP runtime (``libgomp.so.1``) is mapped.

    A copy bundled in a wheel carries a hashed name (``libgomp-<hash>.so``) and is a
    different runtime; only the shared one is Numba's OpenMP layer's.

    Parameters
    ----------
    maps : Path, optional
        Linux process memory-map file to inspect.

    Returns
    -------
    bool
        Whether the shared runtime appears in the readable process maps.
    """
    try:
        with maps.open(encoding="utf-8") as handle:
            return any(
                Path(line.split()[-1]).name.startswith("libgomp.so")
                for line in handle
                if line.strip()
            )
    except OSError:
        return False


@contextlib.contextmanager
def numba_launch(platform: str = sys.platform) -> Iterator[str]:
    """Start Numba's threads inside this block so that they wait passively and unbound.

    Numba's OpenMP threading layer loads the shared GNU OpenMP runtime when its
    threads first start. With every core in the pool, its workers spin between
    parallel loops on the cores FFTW's threads need: the comb-to-continuum window took
    17 s on 64 Zen 5 cores against 5.4 s with ``OMP_WAIT_POLICY=passive`` (SEP 28). On
    Linux, when nothing has decided otherwise, the runtime is loaded passive and
    unbound; when it is already loaded (by the native backend's build, say) under
    settings this cannot change, Numba's own ``workqueue`` layer, which uses no
    OpenMP, is chosen instead (5.5 s on the same window).

    Parameters
    ----------
    platform : str, optional
        ``sys.platform``.

    Yields
    ------
    str
        What applies, for the run's metadata: ``"passive"``, ``"workqueue"``,
        ``"started"`` (Numba's threads were running already), ``"environment"`` (the
        user chose a layer or a wait policy) or ``"runtime default"``.
    """
    from numba.core import config  # noqa: PLC0415
    from numba.np.ufunc import parallel  # noqa: PLC0415

    if getattr(parallel, "_is_initialized", False):
        yield "started"
        return
    if platform != "linux":
        yield "runtime default"
        return
    if getattr(config, "THREADING_LAYER", "default") != "default" or any(
        name in os.environ for name in VARIABLES
    ):
        with _environment(
            _changes("", binding_policy(platform, os.environ, loaded=False))
        ):
            yield "environment"
        return
    if _gnu_openmp_loaded():
        config.THREADING_LAYER = "workqueue"  # pyright: ignore[reportAttributeAccessIssue]
        yield "workqueue"
        return
    with _environment(
        _changes("passive", binding_policy(platform, os.environ, loaded=False))
    ):
        yield "passive"
