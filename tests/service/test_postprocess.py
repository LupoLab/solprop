# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Post-process step's compressor: its stages, and the kernel's chain on a run."""

import math
import warnings
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pytest

from solprop.exceptions import SolpropError
from solprop.service import handlers

# What the window's compressor model sends for each stage at its defaults
# (solprop_gui.model.compressor.payload); a test sets only what it changes.
DEFAULTS: dict[str, dict[str, object]] = {
    "taylor": {"gdd": 0.0, "tod": 0.0, "fod": 0.0, "order": 3},
    "material": {
        "medium": "fused silica",
        "pressure": None,
        "thickness": 0.001,
        "thickness_min": 0.0,
        "thickness_max": 0.005,
    },
    "mirrors": {
        "mirror": "PC70",
        "reflections": 4,
        "max_reflections": 10,
        "wedge": None,
        "wedge_min": 0.0,
        "wedge_max": 0.004,
    },
    "prisms": {
        "medium": "fused silica",
        "separation": 0.6,
        "insertion": 0.015,
        "first_face": 0.003,
        "double_pass": True,
        "separation_min": None,
        "separation_max": None,
        "insertion_min": 0.01,
        "insertion_max": 0.02,
        "band_min": 6.5e-07,
        "band_max": 1.05e-06,
    },
    "gratings": {
        "line_density": 1200000.0,
        "separation": 0.05,
        "incidence": None,
        "double_pass": True,
        "separation_min": 0.01,
        "separation_max": 0.1,
        "band_min": 6.5e-07,
        "band_max": 1e-06,
    },
}


def stage(kind: str, *, auto: bool = False, **values: object) -> dict[str, object]:
    """Return one stage as the window sends it."""
    return {"kind": kind, "auto": auto, "values": {**DEFAULTS[kind], **values}}


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    """Ignore the material-range warnings of the wide bands."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def ask(kind: str, body: Mapping[str, object]) -> Any:
    return handlers.HANDLERS[kind](body)


def chain(path: str, *stages: Any, **extra: object) -> Any:
    body = {"path": path, "index": -1, "stages": list(stages), **extra}
    return ask("compress_chain", body)


def test_the_catalogue_names_glasses_gases_and_mirrors():
    found = ask("optics_catalogue", {})
    assert "fused silica" in found["glasses"] and "Air" in found["gases"]
    assert found["media"] == found["glasses"] + found["gases"]
    assert "PC70" in found["mirrors"]


def test_fixed_elements_are_applied_in_order(run: str):
    stages = [
        stage("taylor", gdd=-10e-30),
        stage("material", medium="Air", thickness=0.5),  # no pressure: one bar
        stage("material", pressure=2.0),  # a glass: the pressure is ignored
        stage("mirrors"),
        stage("prisms"),
        stage("gratings", incidence=0.4),
        stage("gratings", separation=0.0),  # Littrow, D = 0
    ]
    found = chain(run, *stages, modes=0, bandpass=[6e-7, 1e-6])
    assert found["after"].shape == found["t"].shape == found["limit"].shape
    assert [entry["auto"] for entry in found["stages"]] == [False] * len(stages)
    pulses = found["pulses"]
    assert pulses["limit"]["peak"] >= pulses["before"]["peak"] * 0.999
    assert math.isclose(found["wavelength"], 8e-7, rel_tol=1e-3)
    empty = chain(run)  # no elements: after is before
    assert np.allclose(empty["after"], empty["before"])


def test_auto_elements_are_searched_one_at_a_time(run: str):
    taylor = stage("taylor", auto=True, order=2)
    found = chain(run, stage("material", thickness=2e-3), taylor)
    (fixed, searched) = found["stages"]
    assert not fixed["auto"] and searched["auto"] and searched["values"]["fod"] == 0.0
    assert "Compression(taylor" in searched["text"] and searched["quality"] <= 1.0
    pulses = found["pulses"]
    assert pulses["after"]["peak"] >= 0.9 * pulses["before"]["peak"]
    wedged = stage(
        "mirrors",
        auto=True,
        max_reflections=2,
        wedge="fused silica",
        wedge_max=2e-3,
    )
    plain = stage("mirrors", auto=True, max_reflections=1)
    found = chain(run, stage("material", auto=True), wedged, plain)
    reports = found["stages"]
    assert 0.0 <= reports[0]["values"]["thickness"] <= 5e-3
    assert set(reports[1]["values"]) == {"reflections", "thickness"}
    assert set(reports[2]["values"]) == {"reflections"}  # no wedge: no thickness


def test_pairs_are_searched_within_their_ranges(run: str):
    found = chain(
        run,
        stage("prisms", auto=True),
        stage("gratings", auto=True, separation_max=0.02),
    )
    prism, grating = found["stages"]
    assert 0.01 <= prism["values"]["insertion"] <= 0.02
    assert prism["values"]["separation"] == pytest.approx(0.6)  # held
    assert 0.01 <= grating["values"]["separation"] <= 0.02


def test_a_compressor_says_what_it_cannot_do(run: str, beam: str):
    with pytest.raises(SolpropError, match="field samples"):
        chain(beam, stage("taylor"))
    with pytest.raises(SolpropError, match="its thickness"):
        chain(run, stage("material", thickness=None))
    with pytest.raises(SolpropError, match="needs both its ends"):
        chain(run, stage("prisms", auto=True, insertion_max=None))
    with pytest.raises(SolpropError, match="thickness range"):
        chain(run, stage("material", auto=True, thickness_min=None, thickness_max=None))
    with pytest.raises(SolpropError, match="separation range"):
        chain(
            run,
            stage("gratings", auto=True, separation_min=None, separation_max=None),
        )
    lens = {"kind": "lens", "auto": False, "values": {}}
    body = {"path": run, "index": -1, "stages": [lens]}
    with pytest.raises(SolpropError, match="no compressor stage"):
        ask("compress_chain", body)
    with pytest.raises(SolpropError, match="no compressor stage"):
        ask("compress_chain", {**body, "stages": [{**lens, "auto": True}]})
