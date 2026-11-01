# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Host construction borrows BLAS pools and restores the previous owner."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import threadpoolctl

import solprop.translate.problem as translation
from modal_provider_cases import GENERAL_Z_END, general_grid, general_guide
from solprop import _process
from solprop.backends import registry
from solprop.backends.numba_cpu import fft
from solprop.constants import C
from solprop.coupling import CouplingModel, FieldPlane, PlaneGeometry, couple, coupler
from solprop.driver import Simulation
from solprop.exceptions import ModeError
from solprop.grid import RadialGrid
from solprop.hankel import QDHT
from solprop.modes import graded_index
from solprop.modes.graded_index import GradedIndexProvider, PowerLawPermittivity
from solprop.modes.guide import ModalGuide
from solprop.modes.material import IndexMaterial
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.translate import general


class Library:
    """A controllable BLAS, independent of the machine running the test."""

    def __init__(self, filepath, threads=32):
        self.filepath = filepath
        self.num_threads = threads
        self.user_api = "blas"
        self.internal_api = "openblas"

    def set_num_threads(self, threads):
        self.num_threads = threads


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


def test_construction_does_not_import_an_absent_native_backend(monkeypatch, pools):
    monkeypatch.delitem(_process.sys.modules, "solprop_native", raising=False)
    with _process.host_blas():
        assert widths(pools) == [1, 1]
        assert "solprop_native" not in _process.sys.modules
    assert widths(pools) == [32, 32]


def test_a_backend_test_double_need_not_supply_native_build_info(monkeypatch, pools):
    monkeypatch.setitem(
        _process.sys.modules, "solprop_native", SimpleNamespace(NativeBackend=object)
    )
    with _process.host_blas():
        assert widths(pools) == [1, 1]


@pytest.mark.parametrize("name", ["native", "numba_cpu"])
@pytest.mark.parametrize("first", ["construction", "backend"])
def test_crossing_backend_and_construction_lifetimes_keep_the_remaining_owner(
    monkeypatch, pools, name, first
):
    controller = Controller(pools[0])
    if name == "native":
        native = pytest.importorskip("solprop_native._host_blas")
        monkeypatch.setattr(native, "_holders", 0)
        monkeypatch.setattr(native, "_limiter", None)

        def native_start():
            monkeypatch.setenv("SOLPROP_BLAS_THREADS", "2")
            native.hold(
                "/native/blas", 4, controller, acquire=_process.acquire_host_blas
            )
            return native.release

        start = native_start

    else:
        # Isolate BLAS ownership from Numba's separate kernel pool.
        monkeypatch.setattr(fft, "set_num_threads", lambda _threads: None)
        monkeypatch.setattr(fft, "get_num_threads", lambda: 1)

        def numba_start():
            settings = replace(fft.choose_pools(1000, 1000, 1000, threads=1), blas=2)
            _, lease = fft.apply_pools(settings, controller=controller)
            return lambda: fft.release_pools(lease)

        start = numba_start

    construction = _process.host_blas()

    def release():
        pass

    try:
        if first == "construction":
            construction.__enter__()
            release = start()
            assert widths(pools) == [1, 1]
            construction.__exit__(None, None, None)
            assert widths(pools) == [2, 2]
            release()
        else:
            release = start()
            monkeypatch.delenv("SOLPROP_BLAS_THREADS", raising=False)
            assert widths(pools) == [2, 2]
            construction.__enter__()
            assert widths(pools) == [1, 1]
            release()
            assert widths(pools) == [1, 1]
            construction.__exit__(None, None, None)
        assert widths(pools) == [32, 32]
    finally:
        release()
        construction.__exit__(None, None, None)


def test_a_failed_lease_acquisition_restores_every_touched_library(monkeypatch, pools):
    failing = pools[0][1]
    original = failing.set_num_threads

    def fail_once(threads):
        monkeypatch.setattr(failing, "set_num_threads", original)
        raise RuntimeError("library setter failed")

    monkeypatch.setattr(failing, "set_num_threads", fail_once)
    with pytest.raises(RuntimeError, match="library setter failed"):
        _process.acquire_host_blas(pools[0], 1)
    assert widths(pools) == [32, 32]
    assert not _process._BLAS_POOLS


def test_releasing_one_owner_twice_keeps_another_owners_narrower_limit(pools):
    wide = _process.acquire_host_blas(pools[0], 4)
    narrow = _process.acquire_host_blas(pools[0], 2)
    assert widths(pools) == [2, 2]
    wide.restore_original_limits()
    wide.restore_original_limits()
    assert widths(pools) == [2, 2]
    narrow.restore_original_limits()
    assert widths(pools) == [32, 32]


def test_a_failed_release_keeps_ownership_for_a_retry(monkeypatch, pools):
    lease = _process.acquire_host_blas(pools[0], 1)
    failing = pools[0][0]
    original = failing.set_num_threads

    def fail_once(threads):
        monkeypatch.setattr(failing, "set_num_threads", original)
        raise RuntimeError("library restore failed")

    monkeypatch.setattr(failing, "set_num_threads", fail_once)
    with pytest.raises(RuntimeError, match="library restore failed"):
        lease.restore_original_limits()
    lease.restore_original_limits()
    assert widths(pools) == [32, 32]
    assert not _process._BLAS_POOLS


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


@pytest.fixture
def simulation():
    """A small host-only construction that any supported CPU backend can own."""
    return Simulation.from_gnlse(
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


def test_translation_and_coupling_hold_and_restore_the_pool(
    monkeypatch, pools, simulation
):
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


def test_general_route_tables_are_built_inside_the_translation_hold(monkeypatch, pools):
    # Simulation.from_waveguide builds these leaves before assemble(), so the
    # route needs its own coarse hold beyond the provider's individual solves.
    grid = general_grid()
    guide = general_guide(grid=grid, chi3=None)
    original = general.modal_transform_tables
    observed = []

    def held_tables(*args, **kwargs):
        observed.append(widths(pools))
        return original(*args, **kwargs)

    monkeypatch.setattr(general, "modal_transform_tables", held_tables)
    route = general.general_modal_route(
        guide, grid, z_max=GENERAL_Z_END, omega_reference=grid.omega0
    )
    assert route.n_modes == 1
    assert observed == [[1, 1]]
    assert widths(pools) == [32, 32]


@pytest.mark.parametrize("measure", ["index", "medium"])
def test_incident_plane_hankel_transform_holds_the_pool(
    monkeypatch, pools, simulation, measure
):
    # An incident launch builds its plane before couple(), including this dense
    # radial transform. Holding only couple() misses the construction product.
    original = QDHT._apply
    observed = []

    def held_transform(*args, **kwargs):
        observed.append(widths(pools))
        return original(*args, **kwargs)

    monkeypatch.setattr(QDHT, "_apply", held_transform)
    grid = simulation.grid
    pulse = SpatialPulse(
        GaussianPulse(wavelength=835e-9, fwhm=30e-15, energy=1e-6), 30e-6
    )
    plane = FieldPlane.from_pulse(
        pulse,
        grid,
        PlaneGeometry(0.0),
        space=RadialGrid(150e-6, 48),
        index=np.ones(grid.n_omega) if measure == "index" else None,
        medium=SpatialMedium() if measure == "medium" else None,
    )
    assert observed and all(value == [1, 1] for value in observed)
    assert plane.samples.shape == (48, 1, grid.n_omega)
    assert np.all(np.isfinite(plane.samples))
    assert widths(pools) == [32, 32]


@pytest.mark.parametrize("name", ["numpy_ref", "numba_cpu", "native"])
def test_construction_beside_a_live_backend_restores_its_pools(
    monkeypatch, simulation, name
):
    if name not in registry.available():
        pytest.skip(f"the {name} backend is not installed")
    problem = simulation.translate()
    backend = registry.select(
        name, problem, grid=simulation.grid, modes=simulation.modes
    )
    try:
        controller = threadpoolctl.ThreadpoolController().select(user_api="blas")
        before = {one.filepath: one.num_threads for one in controller.lib_controllers}
        native = _process.sys.modules.get("solprop_native")
        own = "" if native is None else native.build_info().blas_library
        monkeypatch.setenv("SOLPROP_BLAS_THREADS", "2")
        monkeypatch.setattr(
            _process, "thread_budget", lambda: _process.ThreadBudget(2, "test")
        )
        with _process.host_blas():
            during = {
                one.filepath: one.num_threads for one in controller.lib_controllers
            }
            assert during == {
                path: width
                if path == own
                else 2
                if name == "numpy_ref"
                else min(width, 2)
                for path, width in before.items()
            }
            # Another construction nests into the hold while the backend lives.
            simulation.with_input(simulation.initial).translate()
        assert {
            one.filepath: one.num_threads for one in controller.lib_controllers
        } == before
    finally:
        backend.close()
