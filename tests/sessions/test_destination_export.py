# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Destination export, end to end through the service loop (SEP 47 section 5.4).

A kept plane's field is exported once, straight into its stored form: into
``MemoryOutput``'s own block row, so it is written once where it is kept, and into a
new array for ``HDF5Output``. A plane observer and a terminal checkpoint at the same
plane share that one export; a field-free plane exports nothing; nothing a consumer
retains is written again. The native session fills a destination by its own fused
pass on every route it runs: modal with unit and true-power weights, radial,
Cartesian, line and planar. Equalities are exact: lossless storage, and the native
pass being the reference recipe operation for operation, are structural contracts.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from sessions.executors import (
    CONSUMER,
    IDENTITY,
    PLANES,
    RECIPES,
    STORED,
    entrance,
    fibre,
    soliton,
    window,
)
from solprop.backends.driver_session import DriverSession
from solprop.backends.events import ExportTicket, ObservationReady
from solprop.backends.export import ExportRecipe, apply_recipe
from solprop.backends.registry import available, select
from solprop.driver.run import run_problem
from solprop.driver.simulation import metadata_for
from solprop.log import NullReporter
from solprop.output import CallbackOutput, HDF5Output, MemoryOutput, open_file
from solprop.output.schema import Output, SavedPlane, StoredField, stored_spectrum
from solprop.stats import StatsSpec
from solprop.translate.linop import power_weight_at

pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

BACKENDS = [
    "numpy_ref",
    pytest.param(
        "native",
        marks=pytest.mark.skipif(
            "native" not in available(), reason="native is not installed"
        ),
    ),
]

FAILING = 2
"""The plane whose reporter call fails: the second save plane."""


class _Spy:
    """Record every export a session makes: its position, recipe and destination."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, ExportRecipe, np.ndarray | None, np.ndarray]] = []

    def wrap(self, export: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        """Return `export`, recording each call."""

        def recorded(
            session: Any,
            ticket: ExportTicket,
            recipe: ExportRecipe,
            out: np.ndarray | None = None,
        ) -> np.ndarray:
            result = export(session, ticket, recipe, out)
            self.calls.append((ticket.z, recipe, out, result))
            return result

        return recorded


def _spied(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    """Spy on the exports of every session a run may start.

    The reference's own session derives from the driver session; a native run is
    served by the native session where it admits the run.
    """
    spy = _Spy()
    monkeypatch.setattr(DriverSession, "export", spy.wrap(DriverSession.export))
    if "native" in available():
        from solprop_native.session import NativeSession  # noqa: PLC0415

        monkeypatch.setattr(NativeSession, "export", spy.wrap(NativeSession.export))
    return spy


class _Exported(MemoryOutput):
    """Keep each exported field by reference, with a copy as it arrived."""

    def __init__(self) -> None:
        super().__init__()
        self.exported: list[tuple[np.ndarray, bytes]] = []

    def save_exported(self, plane: SavedPlane, stored: StoredField) -> None:
        self.exported.append((stored.values, stored.values.tobytes()))
        super().save_exported(plane, stored)


class _Watcher(NullReporter):
    """A plane observer that keeps each plane's field and may fail at one plane."""

    def __init__(self, *, fail_at: int | None = None) -> None:
        self.spectra: list[np.ndarray] = []
        self._fail_at = fail_at
        self._planes = 0

    def observe_plane(self, z, plane) -> None:
        self.spectra.append(np.array(plane.stored()))

    def plane(self, z, summary) -> None:
        self._planes += 1
        if self._planes == self._fail_at:
            msg = f"the reporter failed at z = {z}"
            raise RuntimeError(msg)


def _run(
    backend: str, output: Output, progress: object = "none", **keywords: Any
) -> None:
    """Run the lifecycle suite's soliton on a backend into `output`."""
    problem, run_request = soliton()
    stepping = select(
        backend, problem, run_request, grid=window(), modes=[fibre().mode]
    )
    try:
        run_problem(
            problem,
            stepping,
            output,
            metadata=metadata_for(problem, run_request, window(), modes=[fibre().mode]),
            progress=progress,  # pyright: ignore[reportArgumentType]
            start=entrance(problem),
            **keywords,
        )
    finally:
        stepping.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_kept_field_is_written_once_into_its_row(backend, monkeypatch):
    """One export per kept plane, into the store's own row, which stays as written."""
    spy = _spied(monkeypatch)
    output = _Exported()
    _run(backend, output)
    assert output.n_saved == len(PLANES) + 1
    assert [z for z, _, _, _ in spy.calls] == [0.0, *PLANES]
    stored = np.asarray(output["field"])
    for (_, recipe, out, result), (values, as_written), row in zip(
        spy.calls, output.exported, stored, strict=True
    ):
        assert recipe == STORED
        # The destination is the block's row: no array of its own, no copy after.
        assert out is not None and result is out and values is out
        assert np.shares_memory(out, stored)
        assert out.tobytes() == as_written == row.tobytes()


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_file_is_handed_one_new_array_per_kept_plane(backend, monkeypatch, tmp_path):
    spy = _spied(monkeypatch)
    path = tmp_path / "run.h5"
    output = HDF5Output(path)
    _run(backend, output)
    assert [(z, recipe, out) for z, recipe, out, _ in spy.calls] == [
        (z, STORED, None) for z in (0.0, *PLANES)
    ]
    with open_file(path) as reader:
        stored = np.asarray(reader["field"])
    for (_, _, _, result), row in zip(spy.calls, stored, strict=True):
        assert result.tobytes() == row.tobytes()


@pytest.mark.parametrize("output", ["memory", "file", "callback"])
def test_a_field_free_plane_exports_nothing(output, monkeypatch, tmp_path):
    spy = _spied(monkeypatch)
    seen: list[SavedPlane] = []
    handler: Output = (
        MemoryOutput()
        if output == "memory"
        else HDF5Output(tmp_path / "run.h5")
        if output == "file"
        else CallbackOutput(seen.append)
    )
    _run("numpy_ref", handler, fields="none")
    assert handler.n_saved == len(PLANES) + 1
    assert spy.calls == []
    assert all(plane.field is None for plane in seen)


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_observer_and_a_checkpoint_share_the_plane_s_one_export(
    backend, monkeypatch
):
    """The observer, the store and the failure checkpoint at one plane: one export.

    The reporter fails after the plane's transaction, so the checkpoint is the stored
    plane. The checkpoint holds the stored field in the internal convention, as the
    store's own reader gives it, and the observer saw what the store holds.
    """
    spy = _spied(monkeypatch)
    output = MemoryOutput()
    watcher = _Watcher(fail_at=FAILING)
    with pytest.raises(RuntimeError, match="reporter failed"):
        _run(backend, output, progress=watcher)
    assert output.n_saved == FAILING + 1
    positions = [z for z, _, _, _ in spy.calls]
    assert positions == [0.0, *PLANES[:FAILING]]  # one export each
    checkpoint = output.terminal_checkpoint
    assert checkpoint is not None and checkpoint.plane.field is not None
    assert checkpoint.plane.z == PLANES[FAILING - 1]
    internal = output.plane_field(FAILING, "fft")
    assert checkpoint.plane.field.tobytes() == internal.tobytes()
    for index, spectrum in enumerate(watcher.spectra):
        assert spectrum.tobytes() == output.plane_field(index, "fft").tobytes()


def test_a_callback_is_given_a_fresh_consumer_field_it_may_keep(monkeypatch):
    spy = _spied(monkeypatch)
    kept: list[tuple[np.ndarray, bytes]] = []

    def receive(plane: SavedPlane) -> None:
        assert plane.field is not None and not plane.field.flags.writeable
        kept.append((plane.field, plane.field.tobytes()))

    _run("numpy_ref", CallbackOutput(receive))
    assert [recipe for _, recipe, _, _ in spy.calls] == [CONSUMER] * len(kept)
    assert len(kept) == len(PLANES) + 1
    for field, as_received in kept:
        assert field.tobytes() == as_received


class _TwoSteps(MemoryOutput):
    """A store given every plane through its own save: the two-step path."""

    def save(self, plane: SavedPlane) -> None:
        super().save(plane)


@pytest.mark.parametrize(
    ("precision", "noise"),
    [("double", False), ("double", True), ("single", False)],
    ids=["double", "noise", "single"],
)
def test_a_run_stores_the_bits_of_the_two_step_path(precision, noise):
    """The reference's whole run, exported into the rows or converted by the store.

    A Cartesian Kerr run, with and without a noise model (which single precision
    refuses), in double and single precision (whose widened state carries its
    power-of-two scale): the stored fields and noise phases are the bits the
    two-step path stores.
    """
    from driver.test_field_free_planes import cartesian  # noqa: PLC0415

    stores = {"exported": MemoryOutput(), "two steps": _TwoSteps()}
    assert stores["exported"].takes_exports
    assert not stores["two steps"].takes_exports
    for store in stores.values():
        simulation = cartesian("numpy_ref", noise=noise)
        simulation.settings = dataclasses.replace(
            simulation.settings, precision=precision
        )
        simulation.run(output=store, progress="none")
    names = ["field", "z", *(["noise_phase"] if noise else [])]
    for name in names:
        exported = np.asarray(stores["exported"][name])
        assert exported.tobytes() == np.asarray(stores["two steps"][name]).tobytes()


# ------------------------------------------------------- the native session's pass


@pytest.fixture
def native_session() -> Any:
    """Return the native session's module, skipping where it is not installed."""
    return pytest.importorskip("solprop_native.session")


def _soliton() -> tuple[Any, Any, dict[str, Any]]:
    return *soliton(), {"grid": window(), "modes": [fibre().mode]}


def _from_simulation(simulation: Any) -> tuple[Any, Any, dict[str, Any]]:
    context = {
        "grid": simulation.grid,
        "modes": simulation.modes,
        "profiles": simulation.profiles,
    }
    return simulation.translate(), simulation._run_request(), context


def _capillary() -> tuple[Any, Any, dict[str, Any]]:
    from driver.test_runner import capillary  # noqa: PLC0415

    return _from_simulation(capillary("native"))


def _radial() -> tuple[Any, Any, dict[str, Any]]:
    from backends.test_transverse_backends import radial_collar  # noqa: PLC0415
    from solprop.stepper import RungeKutta  # noqa: PLC0415

    return _from_simulation(radial_collar("native", RungeKutta()))


def _cartesian() -> tuple[Any, Any, dict[str, Any]]:
    from driver.test_field_free_planes import cartesian  # noqa: PLC0415

    return _from_simulation(cartesian("native"))


def _line() -> tuple[Any, Any, dict[str, Any]]:
    from test_transverse_reads import REREVIEW, packet_run  # noqa: PLC0415

    return _from_simulation(packet_run(REREVIEW, backend="native", check=None))


def _planar() -> tuple[Any, Any, dict[str, Any]]:
    from backends.test_planar_native import planar_case  # noqa: PLC0415

    grid, _, problem, run_request = planar_case("envelope_kerr")
    return problem, run_request, {"grid": grid}


ROUTES: dict[str, Callable[[], tuple[Any, Any, dict[str, Any]]]] = {
    "modal, unit weight": _soliton,
    "modal, true-power weight": _capillary,
    "radial": _radial,
    "cartesian": _cartesian,
    "line": _line,
    "planar": _planar,
}


@pytest.mark.parametrize("route", ROUTES)
def test_the_native_pass_fills_a_destination_on_every_route(route, native_session):
    """Each recipe into a destination is ``apply_recipe`` of the state, to the bit,
    at the launch and at the first plane after it; the stored form is the two-step
    conversion of the consumer field.
    """
    problem, run_request, context = ROUTES[route]()
    session = native_session.NativeSession(
        problem, run_request.for_run(stats=StatsSpec()), entrance(problem), **context
    )
    spatial = problem.transform.geometry is not None
    kind = context["grid"].kind
    observed = 0
    try:
        while observed < 2:
            batch = session.advance()
            for event in batch.events:
                if not isinstance(event, ObservationReady):
                    continue
                observed += 1
                field = np.array(session.export(event.ticket, IDENTITY))
                weight = None if spatial else power_weight_at(problem, event.z)
                if route.startswith("modal"):
                    assert bool(np.all(weight == 1.0)) is (
                        route == "modal, unit weight"
                    )
                for recipe in RECIPES:
                    reference = (
                        dataclasses.replace(recipe, power=False) if spatial else recipe
                    )
                    expected = apply_recipe(
                        field, reference, weight=weight if reference.power else None
                    )
                    out = np.empty_like(field)
                    assert session.export(event.ticket, recipe, out=out) is out
                    assert out.tobytes() == expected.tobytes(), (route, recipe)
                two_steps = stored_spectrum(
                    session.export(event.ticket, CONSUMER), kind
                )
                out = np.empty_like(field)
                session.export(event.ticket, STORED, out=out)
                assert out.tobytes() == two_steps.tobytes()
            session.acknowledge(batch.receipt)
    finally:
        session.close()


def _served(
    native_session: Any,
    output: Output,
    problem: Any,
    run_request: Any,
    context: dict[str, Any],
    spy: _Spy,
) -> Any:
    """Serve a native session into `output`; return the session's status."""
    from solprop.driver.run import field_retention  # noqa: PLC0415
    from solprop.driver.service import ServiceContext, serve  # noqa: PLC0415
    from solprop.log import resolve_reporter  # noqa: PLC0415

    request = run_request.for_run(stats=StatsSpec())
    session = native_session.NativeSession(
        problem, request, entrance(problem), **context
    )
    session.export = spy.wrap(type(session).export).__get__(session)
    try:
        serve(
            session,
            output,
            ServiceContext(
                problem=problem,
                request=request,
                metadata=metadata_for(problem, request, context["grid"], modes=()),
                reporter=resolve_reporter("none"),
                retention=field_retention(request, "all", ()),
                resume_history=None,
                snapshots=None,
                tdse_binding=None,
                tdse_archive=None,
                recipe=None,
                start_plane=0,
            ),
        )
        return session.native.status()
    finally:
        session.close()


@pytest.mark.parametrize("route", ["radial", "cartesian"])
def test_the_native_session_writes_a_kept_field_once_into_its_row(
    route, native_session
):
    problem, run_request, context = ROUTES[route]()
    spy = _Spy()
    output = _Exported()
    status = _served(native_session, output, problem, run_request, context, spy)
    stored = np.asarray(output["field"])
    assert status.exports == len(spy.calls) == output.n_saved == stored.shape[0]
    for (_, recipe, out, _), (values, as_written), row in zip(
        spy.calls, output.exported, stored, strict=True
    ):
        assert recipe == STORED and out is not None and values is out
        assert np.shares_memory(out, stored)
        assert out.tobytes() == as_written == row.tobytes()


def test_the_native_session_hands_a_file_one_new_array_per_kept_plane(
    tmp_path, native_session
):
    problem, run_request, context = ROUTES["cartesian"]()
    spy = _Spy()
    path = tmp_path / "run.h5"
    status = _served(
        native_session, HDF5Output(path), problem, run_request, context, spy
    )
    assert status.exports == len(spy.calls)
    with open_file(path) as reader:
        stored = np.asarray(reader["field"])
    assert stored.shape[0] == len(spy.calls)
    for (_, recipe, out, result), row in zip(spy.calls, stored, strict=True):
        assert recipe == STORED and out is None
        assert not result.flags.writeable  # the session's own, kept for its batch
        assert result.tobytes() == row.tobytes()
