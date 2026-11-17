# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run computes no basis identity: a plane's basis is derived when it is read.

A solved mode (a step-index fibre's) or a guide that varies along z costs a mode
evaluation per plane, so recording a basis at every planned plane made a run with many
saved planes slow to start: 201 planes of a step-index fibre took about 95 s before the
first step, and a pressure gradient's 201 planes about 0.3 s. Only coupling reads a
basis, so it is derived then (:meth:`solprop.coupling.PlaneBasis.of_run`), and a run
that is never coupled pays nothing for it.
"""

import pytest

from solprop.coupling import plane
from solprop.recipe import Recipe


@pytest.mark.parametrize("example", ["soliton", "rdw"])
def test_a_run_s_record_computes_and_holds_no_basis_identity(
    example: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluated: list[float] = []
    real = plane.modal_basis_identity

    def counting(modes, grid, z):
        evaluated.append(z)
        return real(modes, grid, z)

    monkeypatch.setattr(plane, "modal_basis_identity", counting)
    recipe = Recipe.example(example)  # "rdw" has a pressure gradient: modes vary in z
    recipe.set("n_save", 41)
    metadata = recipe.build().metadata()
    assert evaluated == []
    assert metadata.reference is not None and "basis" not in metadata.reference
