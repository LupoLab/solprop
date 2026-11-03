# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel's results requests on a small run: a plane, statistics, compression."""

import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.figure import Figure
from solprop.exceptions import SolpropError
from solprop.recipe import Recipe

from solprop_gui.service import handlers, results
from test_design import MULTIMODE

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


def test_the_spectrogram_compression_and_figures(run: str, tmp_path: Path, monkeypatch):
    image = ask("spectrogram", {"path": run, "z": 0.05})["png"]
    assert image.startswith(b"\x89PNG")
    found = ask("compression", {"path": run, "index": 2})
    assert found["fwhm_limit"] <= found["fwhm_after"] <= found["fwhm_before"] * 1.01
    assert found["after"].shape == found["t"].shape
    single = ask("compression", {"path": run, "index": 2, "modes": 0, "order": 3})
    assert len(single["coefficients"]) >= 3
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
