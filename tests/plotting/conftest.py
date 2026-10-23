# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Figures are drawn on the Agg backend and closed after every test.

The backend is fixed before matplotlib is imported anywhere, so that the suite never
depends on what the machine running it has available, and the environment variable is
set as well as the backend so that a subprocess sees the same choice. Every test closes
its figures: matplotlib warns after twenty are left open, and warnings are errors here.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from solprop.output import open_file
from solprop.processing._examples import example_output


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(params=["envelope", "field"])
def kind(request):
    return request.param


@pytest.fixture
def single(kind):
    return example_output(kind)


@pytest.fixture
def multimode(kind):
    return example_output(kind, n_modes=3)


@pytest.fixture
def on_file(single, tmp_path):
    single.to_file(tmp_path / "run.h5")
    with open_file(tmp_path / "run.h5") as handler:
        yield handler
