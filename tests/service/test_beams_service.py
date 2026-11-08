# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a radial beam's pages ask the kernel for."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from solprop.recipe import Recipe
from solprop.service import beams


@pytest.fixture(scope="module")
def beam_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("beam") / "run.h5"
    Recipe.example("radial_kerr").run(path)
    return path


def test_the_overview_has_the_fluence_and_the_on_axis_spectrum(beam_file: Path) -> None:
    reply: dict[str, Any] = beams.beam_overview({"path": str(beam_file)})
    assert reply["fluence"].shape == (reply["z"].size, reply["r"].size)
    assert reply["on_axis"].shape == (reply["z"].size, reply["x"].size)
    assert reply["axis"] == "wavelength"
    assert np.all(np.diff(reply["x"]) > 0)
    on_f: dict[str, Any] = beams.beam_overview({"path": str(beam_file), "axis": "f"})
    assert on_f["axis"] == "f"


def test_a_plane_across_the_beam(beam_file: Path) -> None:
    reply: dict[str, Any] = beams.beam_plane({"path": str(beam_file), "index": 0})
    assert reply["intensity"].shape == (reply["r"].size, reply["t"].size)
    assert reply["spectrum"].shape == (reply["r"].size, reply["x"].size)
    assert reply["fluence"].size == reply["r"].size
    assert reply["fluence"][0] > 0


def test_a_plane_on_the_axis(beam_file: Path) -> None:
    reply: dict[str, Any] = beams.beam_on_axis(
        {"path": str(beam_file), "index": 0, "positions": 32}
    )
    assert reply["t"].size == reply["power"].size
    assert reply["x"].size == reply["spectrum"].size
    drawn = reply["spectrogram"]
    assert drawn["values"].shape == (drawn["x"].size, drawn["t"].size)
    assert np.isfinite(drawn["values"]).all()
    assert drawn["values"].max() == pytest.approx(1.0)
    assert drawn["gate_fwhm"] > 0
    gated: dict[str, Any] = beams.beam_on_axis(
        {"path": str(beam_file), "index": 0, "positions": 32, "gate_fwhm": 2e-14}
    )
    assert gated["spectrogram"]["gate_fwhm"] == pytest.approx(2e-14)


def test_a_far_field(beam_file: Path) -> None:
    reply: dict[str, Any] = beams.beam_far_field({"path": str(beam_file), "index": 0})
    assert reply["values"].shape == (reply["theta"].size, reply["x"].size)
    assert reply["values"].max() > 0


def test_a_cartesian_and_a_line_beam_s_fluence(tmp_path: Path) -> None:
    from solprop.recipe import Recipe  # noqa: PLC0415 - only here
    from solprop.service.beams import beam_fluence, beam_line  # noqa: PLC0415

    cartesian = tmp_path / "cartesian.h5"
    Recipe.example("cartesian_beam").run(cartesian)
    found: Any = beam_fluence({"path": str(cartesian), "index": -1})
    assert np.shape(found["fluence"]) == (found["y"].size, found["x"].size)
    dx = float(np.diff(found["x"][:2])[0])
    dy = float(np.diff(found["y"][:2])[0])
    assert float(found["fluence"].sum()) * dx * dy == pytest.approx(1.5e-9, rel=0.02)
    line = tmp_path / "line.h5"
    Recipe.example("line_beam").run(line)
    along: Any = beam_line({"path": str(line)})
    assert along["z"].tolist() == pytest.approx([0.0, 0.05, 0.1])
    assert np.shape(along["fluence"]) == (3, along["x"].size)
