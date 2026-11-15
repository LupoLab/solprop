# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hold what a hand-built fill of the dispersion-only ``Air`` entry does on a capillary.

The built-in ``Air`` entry carries optmat's index of dry air as a whole and no Kerr,
ionisation or Raman data (``docs/guide/media/mixtures.md``). A run that asks for one
of those channels is refused with a pointer to ``Mixture.air``; a run whose user
switched the responses off runs linear, logged once at INFO on the envelope route,
where plasma is off by default.
"""

import logging
from typing import Any

import pytest

from solprop.driver import Simulation
from solprop.exceptions import MaterialError
from solprop.physdata.mixtures import Mixture
from solprop.physdata.selection import MaterialSelection

COMMON: dict[str, Any] = {
    "wavelength": 800e-9,
    "wavelength_limits": (400e-9, 2e-6),
    "trange": 400e-15,
    "fwhm": 30e-15,
    "energy": 5e-6,
    "n_save": 2,
    "noise": None,
    "backend": "numpy_ref",
}
"""The issue's reproduction settings: a short capillary, translated, never run."""

POINTER = "Mixture.air (solprop.physdata.mixtures.Mixture.air)"
LINEAR = "the fill has no nonlinear response"


def _translate(**keywords: Any) -> Simulation:
    simulation = Simulation.from_capillary(
        100e-6, 0.05, Mixture.fill({"Air": 1.0}, 1.0), **COMMON, **keywords
    )
    simulation.translate()
    return simulation


@pytest.mark.parametrize(
    ("keywords", "refusal"),
    [
        (
            {},
            "Air (Air): Air has no Kerr data: the built-in Air entry is dispersion",
        ),
        ({"envelope": True}, "Air (Air): Air has no Kerr data"),
        ({"kerr": True, "plasma": False}, "Air (Air): Air has no Kerr data"),
        ({"kerr": False}, "Air (Air) has no ionisation data: the built-in Air entry"),
        # The two refusals beside the issue's six cases: a dispersive electronic
        # policy, and Raman selected by the entry's label.
        (
            {"materials": MaterialSelection(electronic="gas-polynomial")},
            "Air carries none",
        ),
        ({"raman": ("Air",)}, "Air (Air) has no molecular Raman response"),
    ],
)
def test_an_air_entry_fill_asking_for_a_channel_is_pointed_to_mixture_air(
    keywords, refusal
):
    with pytest.raises(MaterialError) as raised:
        _translate(**keywords)
    message = str(raised.value)
    assert refusal in message
    assert POINTER in message
    assert "custom species" not in message


@pytest.mark.parametrize(
    ("keywords", "logged"),
    [
        ({"envelope": True, "kerr": False}, True),
        ({"kerr": False, "plasma": False}, False),
        ({"envelope": True, "kerr": False, "plasma": False}, False),
    ],
)
def test_an_air_entry_fill_with_its_responses_switched_off_runs_linear(
    keywords, logged, caplog
):
    with caplog.at_level(logging.INFO, logger="solprop"):
        simulation = _translate(**keywords)
    assert tuple(simulation.responses) == ()
    notes = [record for record in caplog.records if LINEAR in record.getMessage()]
    assert len(notes) == int(logged)
    if logged:
        assert notes[0].levelno == logging.INFO
        assert notes[0].name.startswith("solprop.")
        assert POINTER in notes[0].getMessage()
