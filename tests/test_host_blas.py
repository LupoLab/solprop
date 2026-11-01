# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Host construction borrows BLAS pools and restores the previous owner."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import threadpoolctl

import solprop.translate.problem as translation
from solprop import _process
from solprop.constants import C
from solprop.coupling import CouplingModel, FieldPlane, couple, coupler
from solprop.driver import Simulation
from solprop.exceptions import ModeError
from solprop.modes import graded_index
from solprop.modes.graded_index import GradedIndexProvider, PowerLawPermittivity
from solprop.modes.guide import ModalGuide
from solprop.modes.material import IndexMaterial


class Library:
    """A controllable BLAS, independent of the machine running the test."""

    def __init__(self, filepath, threads=32):
        self.filepath = filepath
        self.num_threads = threads
        self.user_api = "blas"


class Limit:
    """Restore the pool state captured when threadpoolctl applies its limit."""

    def __init__(self, libraries, threads):
        self.before = [(library, library.num_threads) for library in libraries]
        for library in libraries:
            library.num_threads = threads

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.restore_original_limits()

    def restore_original_limits(self):
        for library, threads in self.before:
            library.num_threads = threads


class Controller:
    """Select libraries and apply the same eager limits as threadpoolctl."""

    def __init__(self, libraries):
        self.lib_controllers = libraries

    def select(self, **filters):
        def selected(library):
            return all(
                getattr(library, name)
                in (value if isinstance(value, list) else [value])
                for name, value in filters.items()
            )

        return Controller([one for one in self.lib_controllers if selected(one)])

    def limit(self, *, limits):
        return Limit(self.lib_controllers, limits)


@pytest.fixture
def pools(monkeypatch):
    """Two independent host BLAS libraries, with a four-thread process budget."""
    libraries = [Library("/numpy/libblas"), Library("/scipy/libblas")]
    walks = []

    def walk():
        walks.append(True)
        return Controller(libraries)

    monkeypatch.setattr(threadpoolctl, "ThreadpoolController", walk)
    monkeypatch.setattr(
        _process, "thread_budget", lambda: _process.ThreadBudget(4, "affinity")
    )
    monkeypatch.delenv("SOLPROP_BLAS_THREADS", raising=False)
    return libraries, walks


def widths(pools):
    return [library.num_threads for library in pools[0]]


def test_construction_overrides_vendor_defaults_and_restores_them(monkeypatch, pools):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "32")
    with _process.host_blas():
        assert widths(pools) == [1, 1]
        with _process.host_blas():
            assert widths(pools) == [1, 1]
        assert widths(pools) == [1, 1]
    assert widths(pools) == [32, 32]
    assert len(pools[1]) == 1


def test_overlapping_construction_restores_only_after_the_last_scope(pools):
    first_entered, second_entered, first_left = Event(), Event(), Event()

    def first():
        with _process.host_blas():
            first_entered.set()
            assert second_entered.wait(timeout=5)
            assert widths(pools) == [1, 1]
        first_left.set()

    def second():
        assert first_entered.wait(timeout=5)
        with _process.host_blas():
            second_entered.set()
            assert first_left.wait(timeout=5)
            assert widths(pools) == [1, 1]

    with ThreadPoolExecutor(max_workers=2) as executor:
        a, b = executor.submit(first), executor.submit(second)
        a.result(timeout=10)
        b.result(timeout=10)
    assert widths(pools) == [32, 32]
    assert len(pools[1]) == 1


@pytest.mark.parametrize("requested,expected", [("2", 2), ("8", 4), (" ", 1)])
def test_explicit_construction_threads_follow_the_process_budget(
    monkeypatch, pools, requested, expected
):
    monkeypatch.setenv("SOLPROP_BLAS_THREADS", requested)
    with _process.host_blas():
        assert widths(pools) == [expected, expected]
    assert widths(pools) == [32, 32]


def test_explicit_oversubscription_uses_the_existing_budget_policy(monkeypatch, pools):
    monkeypatch.setenv("SOLPROP_BLAS_THREADS", "8")
    monkeypatch.setattr(
        _process,
        "thread_budget",
        lambda: _process.ThreadBudget(4, "affinity", oversubscribe=True, requested=6),
    )
    with _process.host_blas():
        assert widths(pools) == [6, 6]


@pytest.mark.parametrize("requested", ["bad", "0", "-2"])
def test_bad_configuration_changes_no_pool(monkeypatch, pools, requested):
    monkeypatch.setenv("SOLPROP_BLAS_THREADS", requested)
    with pytest.raises(ValueError, match="SOLPROP_BLAS_THREADS"), _process.host_blas():
        pytest.fail("invalid configuration entered construction")
    assert widths(pools) == [32, 32]


def test_failed_construction_restores_a_live_backends_hold(pools):
    # A live backend has already taken these pools; construction borrows them,
    # restores the backend's width, and leaves its eventual release intact.
    with Controller(pools[0]).limit(limits=2):
        with pytest.raises(ModeError, match="failed solve"), _process.host_blas():
            assert widths(pools) == [1, 1]
            raise ModeError("failed solve")
        assert widths(pools) == [2, 2]
    assert widths(pools) == [32, 32]
    with _process.host_blas():
        assert widths(pools) == [1, 1]


def test_native_own_blas_keeps_its_fitted_width(monkeypatch, pools):
    native_blas = Library("/native/libblas", 4)
    pools[0].append(native_blas)
    monkeypatch.setitem(
        _process.sys.modules,
        "solprop_native",
        SimpleNamespace(
            build_info=lambda: SimpleNamespace(blas_library=native_blas.filepath)
        ),
    )
    with _process.host_blas():
        assert widths(pools) == [1, 1, 4]
    assert widths(pools) == [32, 32, 4]
    # A native BLAS alone must not turn an empty selection into "all libraries".
    pools[0][:] = [native_blas]
    with _process.host_blas():
        assert native_blas.num_threads == 4


def test_no_controllable_blas_is_an_ordinary_scope(pools):
    pools[0].clear()
    with _process.host_blas(), _process.host_blas():
        assert widths(pools) == []
    assert len(pools[1]) == 1


def test_provider_construction_and_standalone_solve_hold_the_pool(monkeypatch, pools):
    original = graded_index.eigh
    observed = []

    def held_eigh(*args, **kwargs):
        observed.append(widths(pools))
        return original(*args, **kwargs)

    monkeypatch.setattr(graded_index, "eigh", held_eigh)
    provider = GradedIndexProvider(
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
        exterior=IndexMaterial(1.45 * np.sqrt(0.99)),
        radius=25e-6,
        family="TE",
        count=1,
        elements=16,
        extent=2.0,
    )
    omega = 2 * np.pi * C / 1.03e-6
    nodes = provider.node_rule(omega)
    assert observed and all(value == [1, 1] for value in observed)
    assert widths(pools) == [32, 32]
    observed.clear()
    guide = ModalGuide.from_provider(
        provider, np.linspace(0.98 * omega, 1.02 * omega, 9), nodes=nodes
    )
    assert observed and all(value == [1, 1] for value in observed)
    assert len(guide.mode_collection.modes) == 1
    assert widths(pools) == [32, 32]
    assert len(pools[1]) == 2


def test_translation_and_coupling_hold_and_restore_the_pool(monkeypatch, pools):
    simulation = Simulation.from_gnlse(
        0.11,
        0.001,
        [0.0, 0.0, -1.183e-26],
        wavelength=835e-9,
        wavelength_limits=(700e-9, 950e-9),
        trange=200e-15,
        fwhm=30e-15,
        peak_power=1.0,
        backend="numpy_ref",
        noise=None,
        n_save=2,
    )
    original_validate = translation.KernelProblem.validate
    observed = []

    def held_validate(problem):
        observed.append(widths(pools))
        return original_validate(problem)

    monkeypatch.setattr(translation.KernelProblem, "validate", held_validate)
    simulation.translate()
    assert observed and all(value == [1, 1] for value in observed)
    assert widths(pools) == [32, 32]
    port = simulation.input_port()
    plane = FieldPlane.from_samples(
        np.zeros(port.shape, dtype=complex),
        port.grid,
        port.geometry,
        exterior="zero",
        basis=port.basis,
        metric=port.metric,
    )
    original_common = coupler._check_common
    observed.clear()

    def held_common(plane, port):
        observed.append(widths(pools))
        return original_common(plane, port)

    monkeypatch.setattr(coupler, "_check_common", held_common)
    result = couple(plane, port, model=CouplingModel("same_basis"))
    assert observed == [[1, 1]]
    assert not np.any(result.launch.samples)
    assert widths(pools) == [32, 32]
