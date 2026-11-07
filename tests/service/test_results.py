# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel's results requests on a small run: a plane, statistics, a spectrogram."""

import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.figure import Figure

from service.test_design import MULTIMODE
from solprop.exceptions import SolpropError
from solprop.output import open_file
from solprop.processing import get_power
from solprop.recipe import Recipe
from solprop.service import handlers, results
from solprop.service.reduce import TIME_OVERSAMPLING

STATS = '\nstats = { type = "StatsSpec", names = ["energy", "peak_power", "fwhm_t"] }\n'


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    """Ignore the material-range warnings of the wide bands."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return a two-mode run's results file, with statistics, made once."""
    path = tmp_path_factory.mktemp("results") / "run.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Recipe.loads(MULTIMODE.replace("n_save = 3", "n_save = 3" + STATS)).run(path)
    return str(path)


def ask(kind: str, payload: Mapping[str, object]) -> Any:
    return handlers.HANDLERS[kind](payload)


def test_a_plane_s_spectrum_power_and_phase(run: str):
    summed = ask("plane", {"path": run, "index": 2})
    assert summed["labels"] == ["HE1,1:even", "HE1,2:even"]
    assert summed["spectrum"]["x"].shape == summed["spectrum"]["y"].shape
    assert summed["power"]["y"].max() > 0
    assert "omega" in summed["phase"]  # the first mode's, for a sum
    one = ask("plane", {"path": run, "index": 2, "modes": 1, "axis": "f"})
    assert one["spectrum"]["y"].sum() < summed["spectrum"]["y"].sum() * 1.0001
    assert one["axis"] == "f" and one["spectrum"]["x"].max() > 1e14  # Hz
    # A bandpass narrower than the spectrum leaves less power.
    band = ask("plane", {"path": run, "index": 2, "bandpass": [7.9e-7, 8.1e-7]})
    assert band["power"]["y"].max() < 0.9 * summed["power"]["y"].max()


def test_a_plane_whose_phase_solprop_refuses_says_why(run: str, monkeypatch):
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise SolpropError("no phase here")

    monkeypatch.setattr(results, "get_phase", refuse)
    assert ask("plane", {"path": run, "index": 0})["phase"] == {
        "error": "no phase here"
    }


def test_the_statistics_are_labelled_by_mode(run: str, monkeypatch):
    found = ask("statistics", {"path": run})
    by_name = {entry["name"]: entry for entry in found["stats"]}
    assert by_name["energy"]["units"] == "J"
    labels = [column["label"] for column in by_name["energy"]["columns"]]
    assert labels[0].startswith("HE1,1:even")
    assert [c["label"] for c in by_name["fwhm_t"]["columns"]] == ["fwhm_t"]
    assert found["steps"]["z"].size > 0 and found["z"].size == 3

    def refuse(_handler: object) -> None:
        raise SolpropError("no modes")

    monkeypatch.setattr(results, "channel_labels", refuse)
    unlabelled = ask("statistics", {"path": run})
    energy = next(e for e in unlabelled["stats"] if e["name"] == "energy")
    assert energy["columns"][0]["label"].startswith("energy [")


def test_the_spectrogram_and_figures(run: str, tmp_path: Path, monkeypatch):
    found = ask("spectrogram", {"path": run, "index": 2})
    assert found["values"].shape == (found["x"].size, found["t"].size)
    assert found["mode"] == 0 and found["axis"] == "wavelength"  # a sum: the first
    assert 1e-7 < found["x"][0] < found["x"][-1] < 1e-5  # the recipe's window, even
    assert np.allclose(np.diff(found["x"]), found["x"][1] - found["x"][0])
    narrow = {
        "path": run,
        "index": 2,
        "modes": 1,
        "axis": "f",
        "time_range": [-5e-14, 5e-14],
        "gate_fwhm": 8e-15,
        "positions": 32,
        "bandpass": [7e-7, 9e-7],
    }
    gated = ask("spectrogram", narrow)
    assert gated["mode"] == 1 and gated["gate_fwhm"] == 8e-15
    # On frequency the map starts at the transform's lowest, not the window's.
    assert gated["x"][0] < 1e14 < gated["x"][-1]
    assert np.all(np.abs(gated["t"]) <= 5e-14)
    for kind in ("prop_2d", "spec_1d", "time_1d", "spectrogram", "plot_stats"):
        file = tmp_path / f"{kind}.png"
        written = ask("result_figure", {"path": run, "file": str(file), "plot": kind})
        assert all(Path(path).stat().st_size > 1000 for path in written)
    with pytest.raises(ValueError, match="no plot"):
        ask(
            "result_figure", {"path": run, "file": str(tmp_path / "x.png"), "plot": "x"}
        )
    monkeypatch.setattr(results.plotting, "plot_stats", lambda _h: [Figure(), Figure()])
    many = ask(
        "result_figure",
        {"path": run, "file": str(tmp_path / "stats.pdf"), "plot": "plot_stats"},
    )
    assert [Path(path).name for path in many] == ["stats-1.pdf", "stats-2.pdf"]


def test_a_file_says_its_planes_positions_and_modes(run: str, monkeypatch):
    described = ask("describe_output", {"path": run})
    assert described["labels"] == ["HE1,1:even", "HE1,2:even"]
    assert np.allclose(described["z"], [0.0, 0.025, 0.05])

    def refuse(_handler: object) -> None:
        raise SolpropError("no modes")

    monkeypatch.setattr(handlers, "channel_labels", refuse)
    assert ask("describe_output", {"path": run})["labels"] == []


@pytest.fixture(scope="module")
def beam(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return a Cartesian beam's results file (solprop's example), made once."""
    path = tmp_path_factory.mktemp("beam") / "beam.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Recipe.example("cartesian_beam").run(path)
    return str(path)


def test_a_spatial_plane_has_its_fluence_map(beam: str, monkeypatch):
    found = ask("plane", {"path": beam, "index": 1})
    fluence = found["fluence"]
    assert fluence["values"].shape == (40, 32) and fluence["unit"] == "J/m^2"
    assert set(fluence["axes"]) == {"x", "y"}
    assert ask("describe_output", {"path": beam})["refused"]["compression"]
    with pytest.raises(SolpropError, match="field samples"):
        ask("spectrogram", {"path": beam, "index": 1})
    # The physical readings: the axial flux, and the energy per bin, which sums to
    # the beam's recorded energy.
    spectrum = found["spectrum"]
    assert spectrum["unit"] == "J per bin" and np.all(np.diff(spectrum["x"]) > 0)
    energy = ask("statistics", {"path": beam})["stats"][0]["columns"][0]["y"][1]
    assert spectrum["y"].sum() == pytest.approx(energy, rel=1e-6)
    assert found["power"]["y"].max() > 0 and "error" in found["phase"]
    rows = ask("overview", {"path": beam})["rows"]
    assert len(rows) == 2
    # On a frequency axis, and the flux within a band.
    narrow = {"path": beam, "index": 1, "axis": "f", "bandpass": [7.9e-7, 8.1e-7]}
    found_f = ask("plane", narrow)
    assert found_f["spectrum"]["x"].max() > 1e14
    assert found_f["power"]["y"].max() < 0.9 * found["power"]["y"].max()
    overview = ask("overview", {"path": beam, "bandpass": narrow["bandpass"]})
    assert overview["axes"]["f"][-1] > 1e14 and len(overview["rows"]) == 2
    assert set(overview["rows"][0]["density"]) == {"wavelength", "f"}
    # A radial grid and a line give a profile; another grid, or a refusal, says so.
    from solprop.grid import LineGrid, RadialGrid  # noqa: PLC0415

    for space, axis in ((RadialGrid(1e-3, 8), "r"), (LineGrid(1e-3, 8), "x")):
        monkeypatch.setattr(results, "_space", lambda _path, space=space: space)
        monkeypatch.setattr(results, "node_fluence", lambda _h: np.ones((2, 8)))
        profile = ask("plane", {"path": beam, "index": 1})["fluence"]
        assert list(profile["axes"]) == [axis] and profile["values"].shape == (8,)
    monkeypatch.setattr(results, "_space", lambda *_a: object())
    assert "error" in ask("plane", {"path": beam, "index": 1})["fluence"]

    def refuse(_handler: object) -> None:
        raise SolpropError("no reconstruction")

    monkeypatch.setattr(results, "node_fluence", refuse)
    no_map = ask("plane", {"path": beam, "index": 1})["fluence"]
    assert no_map == {"error": "no reconstruction"}


@pytest.fixture(scope="module")
def scanned(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return a three-point pressure scan's results file, with statistics."""
    from test_design import SCAN  # noqa: PLC0415

    from solprop.driver.scan import run_scan  # noqa: PLC0415 - only here

    text = SCAN.replace('"1 bar"]', '"1 bar", "1.5 bar"]').replace(
        "n_save = 2", "n_save = 2" + STATS
    )
    path = tmp_path_factory.mktemp("scan") / "scan.h5"
    scan = Recipe.loads(text).scan()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_scan(scan.scan, scan.build, output=path, progress="none", **scan.keywords)  # pyright: ignore[reportArgumentType]
    return str(path)


def test_a_scan_s_results_over_its_variables(scanned: str, monkeypatch):
    found = ask("scan_overview", {"path": scanned})
    assert found["variables"][0]["name"] == "pressure"
    assert list(found["variables"][0]["values"]) == [0.5, 1.0, 1.5]
    assert found["shape"] == [3] and found["status"] == ["done"] * 3
    energy = found["results"]["stats/energy"]
    assert energy["units"] == "J" and energy["values"].shape == (3,)
    assert np.all(np.isfinite(energy["values"]))

    class Unrecorded:
        def __init__(self, _path: str) -> None:
            self.recipe = None

    monkeypatch.setattr(results, "ScanOutput", Unrecorded)
    with pytest.raises(SolpropError, match="records no recipe"):
        ask("scan_overview", {"path": scanned})


def test_a_scan_point_opens_as_its_own_setup(scanned: str) -> None:
    found = ask("scan_point", {"path": scanned, "index": 2})
    assert "[scan]" not in found["text"]
    single = Recipe.loads(found["text"])
    assert single.get("pressure") == "1.5 bar"
    assert float(found["values"]["pressure"]) == pytest.approx(1.5)


def test_a_scan_with_no_point_done_has_no_results(scanned: str, monkeypatch):
    real = results.ScanOutput(scanned)

    class NoneDone:
        def __init__(self, _path: str) -> None:
            self.recipe = real.recipe
            self.shape = real.shape
            self.status = np.array(["failed"] * 3)

    monkeypatch.setattr(results, "ScanOutput", NoneDone)
    found = ask("scan_overview", {"path": scanned})
    assert found["results"] == {} and found["status"] == ["failed"] * 3


def test_a_planar_plane_says_why_it_has_no_spectrum(tmp_path: Path):
    path = tmp_path / "planar.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Recipe.example("planar_hybrid").run(path)
    found = ask("plane", {"path": str(path), "index": 1})
    assert "planar" in found["spectrum"]["error"] and found["power"]["y"].max() > 0
    banded = ask("plane", {"path": str(path), "index": 1, "bandpass": [7e-7, 9e-7]})
    assert banded["power"]["y"].max() <= found["power"]["y"].max() * 1.0001
    with pytest.raises(SolpropError, match="planar"):
        ask("overview", {"path": str(path)})


def test_a_plane_gives_its_energy_and_a_band_its_share(finished_run: Path) -> None:
    whole: dict[str, Any] = results.plane({"path": str(finished_run), "index": 0})
    assert whole["energy"] > 0
    assert whole["fwhm"] > 0
    assert "band_energy" not in whole
    low, high = whole["window"]
    assert low < 800e-9 < high
    band: dict[str, Any] = results.plane(
        {"path": str(finished_run), "index": 0, "bandpass": [600e-9, 1000e-9]}
    )
    assert 0 < band["band_energy"] <= band["energy"]
    assert np.asarray(band["full_power"]).size == np.asarray(band["power"]["y"]).size
    # The pulse is drawn four times finer than the run's grid, over the same window.
    with open_file(str(finished_run)) as handler:
        t, _power = get_power(handler, plane=0, modes="sum")
    drawn = np.asarray(whole["power"]["t"])
    assert drawn.size == TIME_OVERSAMPLING * np.asarray(t).size
    assert drawn[0] == pytest.approx(t[0]) and drawn[-1] >= t[-1]


def test_run_health_says_how_the_run_went(finished_run: Path) -> None:
    health: dict[str, Any] = results.run_health({"path": str(finished_run)})
    assert health["status"] == "complete"
    assert health["facts"]["backend"] == "numpy_ref"
    assert np.asarray(health["steps"]["z"]).size > 0
    assert health["log"] is None or isinstance(health["log"], str)


def test_run_health_reads_what_the_record_holds(
    tmp_path: Path, finished_run: Path
) -> None:
    assert results._health_facts({}) == {
        "backend": None,
        "precision": None,
        "threads": None,
        "wall_time": None,
        "peak_memory": None,
        "plasma_source": None,
    }
    facts = results._health_facts(
        {"resources": {"wall_time_s": 2.0}, "plasma_source": {"certificate": 1e-16}}
    )
    assert facts["wall_time"] == 2.0
    assert facts["plasma_source"] == [1e-16, None]
    assert results._boundary_of({}) is None
    assert results._boundary_of({"boundary_monitor": {}}) is None
    boundary = results._boundary_of(
        {
            "boundary_monitor": {
                "budget": 1e-3,
                "planes": {"z": [0.0], "removed_fraction": [0.0]},
            }
        }
    )
    assert boundary is not None
    assert boundary["budget"] == 1e-3
    copy = tmp_path / "run.h5"
    copy.write_bytes(finished_run.read_bytes())
    (tmp_path / results.RUN_LOG).write_text("hello", encoding="utf-8")
    assert results.run_health({"path": str(copy)})["log"] == "hello"


def test_a_column_is_named_by_what_varies(finished_run: Path) -> None:
    reply: dict[str, Any] = results.statistics({"path": str(finished_run)})
    energy = next(entry for entry in reply["stats"] if entry["name"] == "energy")
    assert all("[" not in column["label"] for column in energy["columns"])


def test_a_file_with_no_mode_axis_has_a_plane(tmp_path: Path) -> None:
    path = tmp_path / "soliton.h5"
    text = Recipe.example("soliton").to_text()
    Recipe.loads(text.replace("n_save = 201", "n_save = 3")).run(path)
    reply = results.plane({"path": str(path), "index": 2})
    low, high = reply["window"]  # pyright: ignore[reportGeneralTypeIssues]
    assert 500e-9 < low < 835e-9 < high < 2000e-9


def test_a_basis_with_no_plain_energy_gives_its_physical_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from solprop.exceptions import OutputError  # noqa: PLC0415 - only here

    path = tmp_path / "soliton.h5"
    text = Recipe.example("soliton").to_text()
    Recipe.loads(text.replace("n_save = 201", "n_save = 2")).run(path)

    def refused(*_args: object, **_kwargs: object) -> float:
        msg = "energy is unavailable for this basis"
        raise OutputError(msg)

    monkeypatch.setattr(results, "energy", refused)
    monkeypatch.setattr(results, "physical_energy", lambda *_a, **_k: 1.5e-9)
    reply = results.plane({"path": str(path), "index": 1})
    assert reply["energy"] == 1.5e-9
