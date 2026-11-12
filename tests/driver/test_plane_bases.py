# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run records its basis at every planned plane, evaluating a constant guide once.

A solved mode (a step-index fibre's) costs a mode solve per evaluation, so recording a
basis per plane made a run with many saved planes slow to start: 201 planes of a
step-index fibre took about 95 s before the first step. A guide whose every mode is
``z_constant`` has the same basis at every plane, and it is evaluated once.
"""

import pytest

from solprop.driver import metadata, simulation
from solprop.recipe import Recipe


def _counted(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    planes: list[float] = []
    real = simulation.modal_basis_identity

    def counting(modes, grid, z):
        planes.append(z)
        return real(modes, grid, z)

    # Both modules that evaluate a basis: the run's record and its modal identity.
    for module in (metadata, simulation):
        monkeypatch.setattr(module, "modal_basis_identity", counting)
    return planes


def test_a_constant_guide_s_basis_is_evaluated_once_for_every_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluated = _counted(monkeypatch)
    recipe = Recipe.example("soliton")
    recipe.set("n_save", 41)
    recipe.build().metadata()
    assert len(evaluated) == 1  # not once for each of the 41 planes


def test_a_guide_that_varies_is_evaluated_at_each_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluated = _counted(monkeypatch)
    recipe = Recipe.example("rdw")  # a pressure gradient: the modes vary along z
    recipe.set("n_save", 6)
    recipe.build().metadata()
    assert len(evaluated) == len(set(evaluated)) == 6  # each saved plane, z = 0 first
