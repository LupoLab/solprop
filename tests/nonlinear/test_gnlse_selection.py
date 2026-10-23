# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The GNLSE's Raman fraction and silica response under a material selection.

SEP 3, D4: the silica bundle's f_R replaces the inline 0.18, which is gone (Phase 4b).
With Raman on an omitted fraction is the bundle's; with Raman off it is 0; a given
fraction is held back whatever ``raman`` is. ``n2="recommended"`` is the bundle's total
n2 at the carrier. An omitted selection is ``MaterialSelection()``.
"""

import math

import pytest

from solprop.constants import C
from solprop.exceptions import MaterialError, ModeError
from solprop.nonlinear.gnlse import (
    kerr_susceptibility,
    problem_from,
    raman_susceptibility,
)
from solprop.nonlinear.raman import SampledShape
from solprop.output.schema import serialise
from solprop.physdata.glasses import SilicaBundle
from solprop.physdata.selection import MaterialSelection

SELECTION = MaterialSelection()
BETAS = [0.0, 5.0e-9, -2.0e-26]
OMEGA = 1.8e15
DEFAULT = SilicaBundle.resolve()


def build(**options):
    return problem_from(BETAS, omega_ref=OMEGA, gamma=0.1, **options)


def test_an_omitted_fraction_with_raman_on_is_the_bundles():
    problem = build(raman=True, selection=SELECTION)
    assert problem.raman_fraction == DEFAULT.raman_fraction
    assert problem.fraction_origin == "bundle"
    assert problem.bundle is not None and problem.bundle.digest == DEFAULT.digest
    raman = problem.raman
    assert raman is not None and raman.fraction == DEFAULT.raman_fraction
    (model,) = raman.models
    assert isinstance(model, SampledShape) and model.channel == "parallel"
    assert model.amplitude == raman_susceptibility(problem.n2)
    assert problem.chi3 == kerr_susceptibility(problem.n2, DEFAULT.raman_fraction)
    # The two together carry the whole of gamma (the long-pulse identity).
    assert (2 / 3) * DEFAULT.raman_fraction * model.amplitude + problem.chi3 == (
        pytest.approx(kerr_susceptibility(problem.n2), rel=1e-15)
    )


def test_an_omitted_fraction_with_raman_off_is_zero():
    problem = build(selection=SELECTION)
    assert problem.raman_fraction == 0.0 and problem.fraction_origin == "raman-off"
    assert problem.raman is None and problem.bundle is None
    assert problem.chi3 == kerr_susceptibility(problem.n2)


@pytest.mark.parametrize("raman", [True, False])
def test_a_given_fraction_is_held_back_whatever_raman_is(raman):
    problem = build(raman=raman, raman_fraction=0.18, selection=SELECTION)
    assert problem.raman_fraction == 0.18 and problem.fraction_origin == "user"
    assert problem.chi3 == kerr_susceptibility(problem.n2, 0.18)
    assert (problem.raman is not None) is raman
    if raman:
        assert problem.raman is not None and problem.raman.fraction == 0.18
        assert isinstance(problem.raman.models[0], SampledShape)
    with pytest.raises(ModeError, match="raman=True with raman_fraction=0"):
        build(raman=True, raman_fraction=0.0, selection=SELECTION)


def test_an_omitted_selection_is_the_default_and_stays_out_of_the_record():
    # selection=None means MaterialSelection(): the bundle's fraction and response.
    on, off, held = build(raman=True), build(), build(raman_fraction=0.18)
    assert (on.raman_fraction, on.fraction_origin) == (DEFAULT.raman_fraction, "bundle")
    assert on.raman is not None
    assert isinstance(on.raman.models[0], SampledShape)
    assert on.bundle is not None and on.bundle.digest == DEFAULT.digest
    assert (off.raman_fraction, off.fraction_origin) == (0.0, "raman-off")
    assert (held.raman_fraction, held.fraction_origin, held.raman) == (
        0.18,
        "user",
        None,
    )
    # The provenance stays out of the configuration record: the bundle reaches the
    # run's record through its material manifest, not the configuration.
    record = serialise(on)
    assert isinstance(record, dict)
    assert not {"bundle", "fraction_origin", "n2_origin"} & set(record)


def test_n2_recommended_is_the_bundles_total_at_the_carrier():
    area = 80e-12
    problem = problem_from(
        BETAS, omega_ref=OMEGA, n2="recommended", aeff=area, selection=SELECTION
    )
    carrier = 2 * math.pi * C / OMEGA
    assert problem.n2 == DEFAULT.n2(carrier)
    assert problem.n2_origin == "bundle"
    assert problem.gamma == pytest.approx(problem.n2 * OMEGA / (C * area), rel=1e-15)
    # Raman off: the bundle supplied n2 only, and none of it is held back.
    assert problem.raman_fraction == 0.0 and problem.bundle is not None


def test_n2_takes_one_name():
    with pytest.raises(ModeError, match="n2='kerr/SiO2'"):
        problem_from(BETAS, omega_ref=OMEGA, n2="kerr/SiO2", selection=SELECTION)


def test_the_selection_names_the_bundle():
    tensor = build(raman=True, selection=MaterialSelection(raman="tensor"))
    # The scalar GNLSE samples the tensor bundle's copolarised a + b with its f_R.
    assert tensor.raman_fraction == 0.245
    assert tensor.bundle is not None and tensor.bundle.shape == "lin_2006"
    with pytest.raises(MaterialError, match="no silica bundle 'N2'"):
        build(raman=True, selection=MaterialSelection(raman="N2"))
