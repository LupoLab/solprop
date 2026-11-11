# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A plane observer: shown each saved plane, lazily, without changing the run (S3)."""

import dataclasses
import threading

import numpy as np
import pytest

from legacy_geometry import LEGACY_GRID, bare_absorber
from solprop.backends.host.driver import HostDriver
from solprop.backends.registry import available
from solprop.exceptions import DriverError, SolpropWarning
from solprop.grid import EnvelopeGrid
from solprop.log import NullReporter, PlaneObserver
from solprop.output.hdf5 import open_file
from solprop.processing import get_spectral_density, get_spectrum
from solprop.stepper import FixedSchedule, RungeKutta

pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

LENGTH = 0.01
N_SAVE = 5
BACKENDS = [
    "numpy_ref",
    pytest.param(
        "native",
        marks=pytest.mark.skipif(
            "native" not in available(), reason="native is not installed"
        ),
    ),
]


class Watcher(NullReporter):
    """Keep a copy of every plane's spectrum and field, as a front end would."""

    def __init__(self, *, flag=None, stop_after=None):
        self.z = []
        self.spectra = []
        self.densities = []
        self.fields = []
        self.views = []
        self.flag = flag
        self.stop_after = stop_after

    def observe_plane(self, z, plane):
        self.z.append(z)
        self.spectra.append(np.array(plane.spectrum()[1]))
        self.densities.append(np.array(plane.spectral_density()[1]))
        self.fields.append(np.array(plane.field()[1]))
        self.views.append(plane)
        if self.flag is not None and len(self.z) == self.stop_after:
            self.flag.set()


class Failing(NullReporter):
    """An observer that fails at its second plane."""

    def __init__(self):
        self.calls = 0
        self.planes = 0

    def observe_plane(self, z, plane):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("drawing failed")

    def plane(self, z, summary):
        self.planes += 1


def _scheduled(build_simulation, backend, **keywords):
    planes = np.linspace(0.0, LENGTH, N_SAVE)[1:]
    grid = EnvelopeGrid.from_limits(
        LENGTH, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID
    )
    return build_simulation(
        length=LENGTH,
        grid=grid,
        absorber=bare_absorber(grid),
        n_save=N_SAVE,
        schedule=FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / 40, save_planes=list(planes)
        ),
        integrator=RungeKutta(),
        backend=backend,
        **keywords,
    )


def test_a_quiet_reporter_is_no_observer():
    assert not isinstance(NullReporter(), PlaneObserver)
    assert isinstance(Watcher(), PlaneObserver)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_observer_s_copy_is_the_file_s_plane_and_changes_nothing(
    build_simulation, tmp_path, backend
):
    watcher = Watcher()
    _scheduled(build_simulation, backend).run(tmp_path / "seen.h5", progress=watcher)
    _scheduled(build_simulation, backend).run(tmp_path / "quiet.h5", progress="none")
    assert watcher.z[0] == 0.0  # the entrance is observed
    assert len(watcher.z) == N_SAVE
    with (
        open_file(tmp_path / "seen.h5") as seen,
        open_file(tmp_path / "quiet.h5") as quiet,
    ):
        for index, spectrum in enumerate(watcher.spectra):
            # The same processing function on the same plane (class P).
            assert np.array_equal(spectrum, get_spectrum(seen, plane=index)[1])
            density = get_spectral_density(seen, plane=index)[1]
            assert np.array_equal(watcher.densities[index], density)
        # Observing changes nothing the run computes (class R, bitwise here).
        assert np.array_equal(np.asarray(seen["field"]), np.asarray(quiet["field"]))


def test_a_view_is_valid_only_during_its_call(build_simulation, tmp_path):
    watcher = Watcher()
    _scheduled(build_simulation, "numpy_ref").run(tmp_path / "x.h5", progress=watcher)
    view = watcher.views[-1]
    with pytest.raises(DriverError, match="valid only during observe_plane"):
        view.spectrum()
    assert view.basis == "modal"
    assert view.quantities["energy"].kind == "physical"
    assert set(view.stats) >= {"energy"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_field_free_planes_are_fetched_on_request_and_counted(
    build_simulation, tmp_path, backend
):
    simulation = _scheduled(build_simulation, backend)
    simulation.settings = dataclasses.replace(simulation.settings, fields="none")
    watcher = Watcher()
    path = tmp_path / "statistics.h5"
    simulation.run(path, progress=watcher)
    reference = Watcher()
    _scheduled(build_simulation, backend).run(tmp_path / "full.h5", progress=reference)
    for fetched, kept in zip(watcher.spectra, reference.spectra, strict=True):
        assert np.array_equal(fetched, kept)
    with open_file(path) as handler:
        outcome = handler.outcome
    assert outcome is not None
    resources = outcome.execution["resources"]
    assert isinstance(resources, dict)
    assert resources["observer_transfers"] == N_SAVE
    assert resources["observer_time_s"] > 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_stop_after_a_fetched_plane_checkpoints_the_field_fetched_once(
    build_simulation, tmp_path, backend, monkeypatch
):
    """The observer and the failure checkpoint share one fetch (SEP 47 section 5.4)."""
    fetched = []

    # The reference's driver observes; the native backend's session exports.
    def counted(self, *, field=True, observe=HostDriver.observe):
        observation = observe(self, field=field)
        if field:
            fetched.append(observation.z)
        return observation

    monkeypatch.setattr(HostDriver, "observe", counted)
    if backend == "native":
        # The native session fetches from its resident state: a fetch is a native
        # export, which the session counts.
        session = pytest.importorskip("solprop_native.session").NativeSession
        export = session.export

        def exported(self, ticket, recipe, out=None):
            before = self.native.status().exports
            field = export(self, ticket, recipe, out)
            if self.native.status().exports > before:
                fetched.append(ticket.z)
            return field

        monkeypatch.setattr(session, "export", exported)
    simulation = _scheduled(build_simulation, backend)
    simulation.settings = dataclasses.replace(simulation.settings, fields="none")
    flag = threading.Event()
    path = tmp_path / "stopped.h5"
    watcher = Watcher(flag=flag, stop_after=2)
    simulation.run(path, progress=watcher, stop=flag)
    # The entrance and the plane the run stopped at, each fetched once.
    assert fetched == watcher.z
    with open_file(path) as handler:
        checkpoint = handler.terminal_checkpoint
        assert checkpoint is not None and checkpoint.plane.field is not None
        assert checkpoint.plane.z == watcher.z[-1]


def test_a_failing_observer_is_detached_and_the_run_completes(
    build_simulation, tmp_path
):
    failing = Failing()
    with pytest.warns(SolpropWarning, match="shown no more planes"):
        _scheduled(build_simulation, "numpy_ref").run(
            tmp_path / "failing.h5", progress=failing
        )
    assert failing.calls == 2
    assert failing.planes == N_SAVE - 1
    with open_file(tmp_path / "failing.h5") as handler:
        assert handler.outcome is not None
        assert handler.outcome.status == "complete"


def test_a_stop_from_an_observer_stops_the_run_and_a_resume_shows_the_rest(
    build_simulation, tmp_path
):
    flag = threading.Event()
    path = tmp_path / "stopped.h5"
    first = Watcher(flag=flag, stop_after=2)
    _scheduled(build_simulation, "numpy_ref").run(path, progress=first, stop=flag)
    assert len(first.z) == 2
    second = Watcher()
    _scheduled(build_simulation, "numpy_ref").run(path, resume=True, progress=second)
    # A resume does not show again the planes it continues after.
    assert second.z == pytest.approx(list(np.linspace(0.0, LENGTH, N_SAVE)[2:]))


class SpatialWatcher(NullReporter):
    """Keep a spatial run's stored planes and energy per bin, as a front end would."""

    def __init__(self):
        self.stored = []
        self.energy = []

    def observe_plane(self, z, plane):
        self.stored.append(plane.stored())
        self.energy.append(np.array(plane.field_plane().spectral_energy()))


def test_a_spatial_plane_is_read_while_it_runs_as_the_file_holds_it(tmp_path):
    from solprop.processing import field_plane  # noqa: PLC0415 - only here
    from solprop.recipe import Recipe  # noqa: PLC0415 - only here

    watcher = SpatialWatcher()
    path = tmp_path / "beam.h5"
    Recipe.example("cartesian_beam").run(path, progress=watcher)
    with open_file(path) as saved:
        assert len(watcher.stored) == saved.n_saved
        for index, stored in enumerate(watcher.stored):
            assert np.array_equal(stored, np.asarray(saved.fields("fft")[index]))
            energy = field_plane(saved, index).spectral_energy()
            assert np.array_equal(watcher.energy[index], energy)
