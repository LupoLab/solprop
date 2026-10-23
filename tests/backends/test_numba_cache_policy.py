# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which ``numba_cpu`` kernels numba may write to its disk cache.

Two rules, both in ``AGENTS.md``. A parallel closure is never disk-cached: numba names
a parfor's gufunc from an object's address and a per-process counter, never from the
closure's constants, so two variants compiled in different processes can carry one
symbol, and loaded into one process the second calls the first's body. A probe on
``combine_and_pad`` found eight such pairs in 800 processes, every one running the
other variant's code. And a closure that captures a dispatcher is not cached either:
numba keys a closure on its pickled cells, a dispatcher pickles differently in every
process, so the entry is never read back and every process appends another.

A third rule sits under both: every kernel is compiled through
:func:`solprop._numba_cache.njit`, whose cache files cannot be cross-wired by two
processes saving at once, as numba's own index can.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest
from numba.core.dispatcher import Dispatcher

import solprop
from solprop._numba_cache import _KeyedFunctionCache
from solprop.backends.numba_cpu import (
    dispersive,
    dispersive_general,
    general_modal,
    kernels,
    multistep,
    spatial,
)
from solprop.backends.numba_cpu.kernels import HOST, ColumnSpec

SOURCE = Path(solprop.__file__).parent


def _disk_cached(dispatcher) -> bool:
    """Return whether numba writes this dispatcher's compilations to its disk cache."""
    return type(dispatcher._cache).__name__ != "NullCache"


def _keyed(dispatcher) -> bool:
    """Return whether the dispatcher's disk cache is the keyed one."""
    return isinstance(dispatcher._cache, _KeyedFunctionCache)


def _factories(parallel: bool) -> Iterator[tuple[str, Dispatcher, bool]]:
    """Yield every closure factory's kernels, each with whether it may be cached.

    Nothing here compiles: the dispatchers are lazy, and the envelope source column,
    the one factory with an eager signature, is covered by the source check below.
    """
    spec = ColumnSpec("field_real", True, False, False, False, 1, parallel, False)
    for contiguous in (False, True):
        for noise in (False, True):
            yield (
                "combine_and_pad",
                kernels.make_combine_and_pad(
                    contiguous=contiguous, noise=noise, parallel=parallel
                ),
                not parallel,
            )
    yield "response_column", kernels.make_response_column(spec), False
    yield "stats_column", kernels.make_stats_column(spec), False
    yield "finish_stage", kernels.make_finish_stage(contiguous=True), True
    for analytic in (False, True):
        made = general_modal.make_general_kernels(
            analytic=analytic, delayed=True, parallel=parallel
        )
        for field in made._fields:
            yield f"general.{field}", getattr(made, field), not parallel
    made = spatial.make_kernels(parallel=parallel, parallel_transfers=parallel)
    for field in made._fields:
        yield f"spatial.{field}", getattr(made, field), not parallel
    made = multistep.make_passes(parallel=parallel)
    for field in made._fields:
        yield f"multistep.{field}", getattr(made, field), not parallel
    # The dispersive response's product kernels capture the bodies: never cached.
    for form in ("real", "conserving", "harmonic"):
        yield (
            f"dispersive.self.{form}",
            dispersive.make_dispersive_self(form, 1, parallel=parallel),
            False,
        )
        yield (
            f"dispersive.pairs.{form}",
            dispersive.make_dispersive_pairs(form, 2, parallel=parallel),
            False,
        )
    for form in ("real", "conserving"):
        made = dispersive_general.make_general_dispersive(form, parallel=parallel)
        for field in made._fields:
            yield f"dispersive.general.{form}.{field}", getattr(made, field), False


@pytest.mark.parametrize("parallel", [False, True])
def test_a_parallel_production_closure_is_never_disk_cached(parallel):
    """The serial variants keep their cache; no parallel one, and no column, has one."""
    made = list(_factories(parallel))
    assert [_disk_cached(one) for _, one, _ in made] == [cached for *_, cached in made]


def _captures_a_dispatcher(value: object) -> bool:
    if isinstance(value, Dispatcher):
        return True
    if isinstance(value, tuple):
        return any(_captures_a_dispatcher(item) for item in value)
    return False


def _bodies() -> list[Dispatcher]:
    bodies = [one for one in vars(HOST).values() if isinstance(one, Dispatcher)]
    assert len(bodies) > 20
    return bodies


def test_no_disk_cached_kernel_captures_a_dispatcher():
    """A cached closure's key must be the same in every process, so no dispatcher."""
    candidates = [
        one for parallel in (False, True) for _, one, _ in _factories(parallel)
    ] + _bodies()
    cached = [one for one in candidates if _disk_cached(one)]
    assert cached, "the check would be vacuous"
    offenders = [
        one.py_func.__qualname__
        for one in cached
        if any(
            _captures_a_dispatcher(cell.cell_contents)
            for cell in one.py_func.__closure__ or ()
        )
    ]
    assert offenders == []
    # The bodies are inlined into the kernels, which cache the result.
    assert not any(_disk_cached(one) for one in _bodies())


def _njit_calls(tree: ast.AST) -> Iterator[ast.Call]:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "njit"
        ):
            yield node


def test_every_parallel_njit_in_the_package_passes_cache_not_parallel():
    """The source check: ``parallel=`` comes with ``cache=False`` or its negation.

    It catches a new factory, which the enumeration above would not know about.
    """
    checked, offenders = 0, []
    for path in sorted(SOURCE.rglob("*.py")):
        for call in _njit_calls(ast.parse(path.read_text(), filename=str(path))):
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            parallel = keywords.get("parallel")
            if parallel is None or (
                isinstance(parallel, ast.Constant) and parallel.value is False
            ):
                continue
            checked += 1
            cache = keywords.get("cache")
            safe = cache is None or (
                isinstance(cache, ast.Constant) and cache.value is False
            )
            negated = (
                isinstance(cache, ast.UnaryOp)
                and isinstance(cache.op, ast.Not)
                and ast.dump(cache.operand) == ast.dump(parallel)
            )
            if not (safe or negated):
                offenders.append(f"{path.relative_to(SOURCE)}:{call.lineno}")
    # combine_and_pad, the two columns, the envelope source column, the general
    # route's compiler and the five spatial kernels.
    assert checked >= 10
    assert offenders == []


def _numba_compilers(tree: ast.AST) -> Iterator[ast.ImportFrom | ast.Attribute]:
    """Yield every import of numba's ``njit`` or ``jit`` and every ``numba.njit``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "numba":
            if any(alias.name in {"njit", "jit"} for alias in node.names):
                yield node
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in {"njit", "jit"}
            and isinstance(node.value, ast.Name)
            and node.value.id == "numba"
        ):
            yield node


def test_every_kernel_compiles_through_the_keyed_njit():
    """Only ``solprop._numba_cache`` calls numba's compiler: every cache is keyed.

    The tests and benchmarks are held to it too, because the test session is where
    many processes compile into one cache at once.
    """
    root = SOURCE.parents[1]
    allowed = {SOURCE / "_numba_cache.py", root / "tests" / "test_numba_cache.py"}
    paths = [
        path
        for tree in ("src", "tests", "benchmarks")
        for path in sorted((root / tree).rglob("*.py"))
        if path not in allowed
    ]
    assert len(paths) > 200
    offenders = [
        f"{path.relative_to(root)}:{node.lineno}"
        for path in paths
        for node in _numba_compilers(ast.parse(path.read_text(), filename=str(path)))
    ]
    assert offenders == []
    cached = [
        one
        for parallel in (False, True)
        for _, one, _ in _factories(parallel)
        if _disk_cached(one)
    ] + [
        one
        for module in (kernels, spatial, general_modal)
        for one in vars(module).values()
        if isinstance(one, Dispatcher) and _disk_cached(one)
    ]
    assert len(cached) > 20
    assert all(_keyed(one) for one in cached)
