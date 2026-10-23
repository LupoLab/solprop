# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The accuracy page's figures render, and they render from the records.

The plot directives that draw `docs/dev/accuracy.md` only run in the last step of
`poe check`, so a drift between what a study writes and what the figure reads would
surface as a Sphinx warning minutes after the tests were green. This file renders every
figure in the default lane instead, by loading the docs-side helper by path.
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
import pytest

import convergence_records as cr

# The figures are rendered headless here, whatever the host would otherwise choose.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

HELPER = Path(__file__).resolve().parents[1] / "docs" / "dev" / "accuracy_figures.py"
spec = importlib.util.spec_from_file_location("accuracy_figures", HELPER)
assert spec is not None and spec.loader is not None
figures = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = figures
spec.loader.exec_module(figures)

DRAWN = [name for name in figures.__all__ if name != "load"]


def test_the_helper_finds_the_records_from_its_own_location():
    """The docs build runs with the page's directory as the working directory."""
    assert figures.RECORDS == cr.RECORDS


@pytest.mark.parametrize("name", DRAWN)
def test_every_figure_draws(name):
    getattr(figures, name)()
    figure = plt.gcf()
    assert figure.get_axes()
    for axes in figure.get_axes():
        assert axes.get_xlabel()
        assert axes.lines or axes.collections
    plt.close("all")


def test_every_figure_names_a_study_that_exists():
    """A figure drawing a record nothing writes is a figure of nothing."""
    drawn = set()
    for name in DRAWN:
        getattr(figures, name)()
        plt.close("all")
        drawn.add(name)
    # Every figure function is named after the record it reads, which is what makes
    # the caption on the page and the file on disk checkable against each other.
    assert drawn <= set(cr.studies())
