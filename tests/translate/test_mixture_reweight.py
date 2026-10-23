# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify range-checked reuse independently of the driver and optical functions."""

import logging

import numpy as np
import pytest

from solprop.profiles import Constant, FromCallable, Linear
from solprop.translate.expansion import build_expansion, reweight_expansion


@pytest.fixture
def quadratic():
    result = build_expansion(
        lambda values: np.array([values["density"] ** 2]),
        {"density": Linear(1.0, 3.0, 1.0)},
        0.0,
        1.0,
    )
    assert result is not None
    return result


@pytest.mark.parametrize("integrate", [False, True])
def test_reweighted_polynomial_shares_terms_and_matches_analytic(quadratic, integrate):
    # A quadratic needs three basis terms. 512 eps per term covers basis mapping,
    # interpolation and integration, followed by the three-term accumulation.
    bound = 512 * quadratic.n_terms * np.finfo(float).eps
    assert bound > 0
    result = reweight_expansion(
        quadratic, {"density": Linear(1.2, 1.8, 1.0)}, integrate=integrate
    )
    assert result is not None
    assert result.terms is quadratic.terms
    assert result.weights is not quadratic.weights
    positions = np.linspace(0, 1, 41)
    expected = (
        1.44 * positions + 0.72 * positions**2 + 0.12 * positions**3
        if integrate
        else (1.2 + 0.6 * positions) ** 2
    )
    actual = np.array([result(float(z))[0] for z in positions])
    np.testing.assert_allclose(actual, expected, rtol=bound, atol=bound)


def test_covered_constant_path_keeps_the_original_basis(quadratic):
    result = reweight_expansion(quadratic, {"density": Constant(2.0)})
    assert result is not None
    assert result.terms is quadratic.terms
    # The same three-term arithmetic allowance as the varying-path test.
    bound = 512 * quadratic.n_terms * np.finfo(float).eps
    np.testing.assert_allclose(result(0.5), [4.0], rtol=bound, atol=0.0)


@pytest.mark.parametrize(
    "paths, reason",
    [
        ({"density": Linear(0.9, 2.0, 1.0)}, "range"),
        ({"density": Linear(2.0, 3.1, 1.0)}, "range"),
        ({"pressure": Linear(1.0, 2.0, 1.0)}, "parameter names"),
        (
            {"density": Linear(1.2, 1.8, 1.0), "radius": Linear(1.0, 2.0, 1.0)},
            "parameter names",
        ),
    ],
)
def test_uncovered_or_changed_parameters_log_a_rebuild(
    quadratic, paths, reason, caplog
):
    with caplog.at_level(logging.INFO, logger="solprop.translate.expansion"):
        assert reweight_expansion(quadratic, paths) is None
    assert reason in caplog.text


def test_unresolved_replacement_weights_request_a_rebuild(quadratic, caplog):
    # The function's polynomial box is covered, but 10,000 path oscillations
    # cannot fit the fixed scalar interval/degree budget. Keep that budget intact.
    path = FromCallable(
        lambda z: 2.0 + 0.1 * np.sin(20000 * np.pi * z), span=(0.0, 1.0)
    )
    with caplog.at_level(logging.INFO, logger="solprop"):
        assert reweight_expansion(quadratic, {"density": path}) is None
    assert "scalar weights exhausted" in caplog.text
