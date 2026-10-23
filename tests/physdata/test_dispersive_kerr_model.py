# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""optmat's dispersive hyperpolarizability in SI, its leg bound and regions (SEP 6).

The SI model of :class:`solprop.physdata.nonlinear_data.DispersiveKerrModel` is checked
against optmat's own ``gamma3`` for every gas optmat fits, by the conversion check of
SEP 6's Phase 0 (``tests/sep6_quartets.py``, ``si_adapter``), and the region
classification of section 4.10 against the band edges and leg bounds optmat's support
record gives.
"""

import dataclasses
import math
import warnings

import numpy as np
import pytest
from optmat.exceptions import ModelRangeWarning
from optmat.gas.nonlinear import hyperpolarizability_models

from sep6_quartets import GASES
from solprop.constants import C
from solprop.exceptions import MaterialError
from solprop.nonlinear.mixtures import electronic_response
from solprop.nonlinear.responses import DispersiveKerrResponse, KerrResponse
from solprop.physdata import MaterialSelection, Species, species
from solprop.physdata.nonlinear_data import DispersiveKerrModel, OptmatKerr
from solprop.physdata.selection import DISPERSIVE_POLICIES, ELECTRONIC_POLICIES

# Phase 0 measured 3.3e-16 for the same comparison: one evaluation each of the
# polynomial in double precision; 1e-14 leaves a factor of 30 for another platform.
SI_RTOL = 1e-14


def omega(wavelength: float) -> float:
    return 2 * math.pi * C / wavelength


@pytest.mark.parametrize("gas", GASES)
def test_si_model_equals_optmat_gamma3_for_every_fitted_gas(gas):
    """Section 4.2's Taylor factor and section 4.5's unit conversions, all fifteen."""
    model = hyperpolarizability_models()[gas]
    dispersive = DispersiveKerrModel.from_optmat(model)
    wavelengths = np.array([max(model.wavelength_range[0], 400e-9), 800e-9, 2000e-9])
    nu = 1.0 / wavelengths
    for process, legs in (
        ("kerr", np.stack([nu, nu, nu, nu], -1)),
        ("thg", np.stack([3 * nu, nu, nu, nu], -1)),
    ):
        squares = legs**2
        s2, s4 = squares.sum(-1), (squares**2).sum(-1)
        ours = dispersive.gamma3_static * (
            1 + dispersive.a * s2 + dispersive.b * s2**2 + dispersive.b_prime * s4
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ModelRangeWarning)
            theirs = model.gamma3(process, wavelengths)
        np.testing.assert_allclose(ours, theirs, rtol=SI_RTOL, atol=0)
    support = model.support
    assert support is not None
    assert dispersive.leg_bound == support.leg_max_in_w2
    assert dispersive.band_edge_linear == support.band_edge("linear")
    assert dispersive.band_edge_any == support.band_edge("any")
    assert dispersive.model_digest == model.identity.digest
    assert dispersive.support_digest == support.identity.digest


def test_the_leg_bound_leaves_quartets_within_the_fitted_sums_exact():
    """Section 4.10: no quartet with ``S2 <= S2_max`` has a leg beyond ``nu_c``."""
    model = OptmatKerr("Ar").dispersive_model()
    s2_max = model.leg_bound**2 / 0.75
    rng = np.random.default_rng(3)
    inputs = rng.uniform(-model.leg_bound, model.leg_bound, size=(100_000, 3))
    legs = np.column_stack([-inputs.sum(1), inputs])
    inside = (legs**2).sum(1) <= s2_max
    assert inside.sum() > 1000
    squares = legs[inside] ** 2
    s2, s4 = squares.sum(-1), (squares**2).sum(-1)
    unbounded = model.gamma3_static * (
        1 + model.a * s2 + model.b * s2**2 + model.b_prime * s4
    )
    # The same polynomial evaluated by optmat: a few tens of roundings on a value
    # near gamma3_static, held to 8 eps.
    np.testing.assert_allclose(
        model.gamma3(legs[inside]), unbounded, rtol=8 * np.finfo(float).eps, atol=0
    )
    # A leg beyond the bound is held there.
    held = np.array([[2.0, 1.0, 0.5, 0.5]]) * model.leg_bound
    at_bound = np.array([[1.0, 1.0, 0.5, 0.5]]) * model.leg_bound
    assert model.gamma3(held) == model.gamma3(at_bound)


def test_leg_weight_is_even_and_held_at_the_bound():
    model = OptmatKerr("He").dispersive_model()
    frequencies = np.array([-omega(100e-9), omega(100e-9), omega(800e-9), 0.0])
    weight = model.leg_weight(frequencies)
    assert weight[0] == weight[1] == model.leg_bound**2
    assert weight[2] == pytest.approx((1 / 800e-9) ** 2, rel=1e-15)
    assert weight[3] == 0.0


def test_helium_band_to_90_nm_reaches_all_three_regions():
    """Supported to the leg bound, bounded to the band edge, best effort beyond."""
    model = OptmatKerr("He").dispersive_model()
    regions = model.regions(omega(4e-6), omega(90e-9), vector=False)
    edge = model.band_edge_linear
    assert edge is not None
    assert regions.supported == pytest.approx((1 / model.leg_bound, 4e-6), rel=1e-15)
    assert regions.bounded is not None
    assert regions.bounded == pytest.approx((1 / edge, 1 / model.leg_bound), rel=1e-15)
    assert regions.best_effort == pytest.approx((90e-9, 1 / edge), rel=1e-15)
    assert round(regions.bounded[0] * 1e9, 1) == 120.3
    assert round(regions.bounded[1] * 1e9, 1) == 151.9
    assert regions.beyond_support
    assert "best effort" in regions.describe()


def test_argon_duv_band_is_supported_and_bounded_only():
    """The DUV FWM case: the generated band stays inside argon's 189.4 nm edge."""
    model = OptmatKerr("Ar").dispersive_model()
    # A grid to 210 nm has an active band to 1.1 omega_max, 190.9 nm.
    regions = model.regions(omega(4e-6) / 2, 1.1 * omega(210e-9), vector=False)
    assert regions.supported is not None
    assert regions.bounded is not None
    assert regions.best_effort is None
    assert round(regions.bounded[1] * 1e9, 1) == 215.9


def test_vector_runs_take_the_any_polarisation_edge():
    """Argon's any-polarisation edge is 192.1 nm, above its 189.4 nm linear one."""
    model = OptmatKerr("Ar").dispersive_model()
    band = (omega(4e-6), omega(190e-9))
    assert model.regions(*band, vector=False).best_effort is None
    vector = model.regions(*band, vector=True)
    assert vector.polarisation == "any"
    assert vector.best_effort is not None
    assert round(vector.best_effort[1] * 1e9, 1) == 192.1


@pytest.mark.parametrize("gas", ["N2", "SF6"])
def test_beyond_the_edge_or_without_one_is_best_effort(gas):
    """N2's edge lies inside its data; SF6 has none: beyond its data is best effort."""
    model = OptmatKerr(gas).dispersive_model()
    regions = model.regions(omega(4e-6), omega(180e-9), vector=False)
    assert regions.bounded is None
    assert regions.best_effort is not None
    edge = model.band_edge_linear
    threshold = model.leg_bound if edge is None else edge
    assert regions.best_effort[1] == pytest.approx(1 / threshold, rel=1e-15)


def test_a_band_must_be_positive_and_increasing():
    model = OptmatKerr("Ar").dispersive_model()
    with pytest.raises(MaterialError, match="positive, finite, increasing"):
        model.regions(omega(800e-9), omega(1e-6), vector=False)


def test_a_non_positive_kerr_diagonal_is_refused():
    """The symmetric factor is a real fourth root only of a positive curve."""
    argon = OptmatKerr("Ar").dispersive_model()
    with pytest.raises(MaterialError, match="not positive up to the leg bound"):
        DispersiveKerrModel(
            "X", 1e-50, -1e-12, 0.0, 0.0, 1e7, None, None, argon.material_records
        )


def test_the_dispersive_policies_name_their_evaluators():
    assert set(DISPERSIVE_POLICIES) <= set(ELECTRONIC_POLICIES)
    assert DISPERSIVE_POLICIES == {
        "symmetric-kerr": "symmetric",
        "gas-polynomial": "polynomial",
    }
    with pytest.raises(MaterialError, match="electronic must be one of"):
        MaterialSelection(electronic="output-only")


@pytest.mark.parametrize("policy", ["recommended", "carrier-kerr"])
def test_carrier_policies_keep_the_constant_response(policy):
    response = electronic_response(
        species("Ar"),
        selection=MaterialSelection(electronic=policy),
        wavelength=800e-9,
        thg=True,
    )
    assert isinstance(response, KerrResponse)
    assert response.chi3 == species("Ar").kerr_coefficient(800e-9)


@pytest.mark.parametrize(("policy", "evaluator"), sorted(DISPERSIVE_POLICIES.items()))
def test_dispersive_policies_build_the_whole_model(policy, evaluator):
    response = electronic_response(
        species("Kr"),
        selection=MaterialSelection(electronic=policy),
        wavelength=None,
        thg=False,
        species="krypton",
    )
    assert isinstance(response, DispersiveKerrResponse)
    assert response.evaluator == evaluator
    assert response.sector == "conserving"
    assert response.species == "krypton"
    assert response.model == OptmatKerr("Kr").dispersive_model()


def test_a_species_without_optmat_model_is_refused_by_name():
    """A supplied species has no hyperpolarizability record: no borrowed curve."""
    argon = species("Ar")
    supplied = Species(
        "custom",
        argon.susceptibility,
        argon.equation_of_state,
        1e-51,
        argon.ionisation,
    )
    with pytest.raises(MaterialError, match="custom carries none"):
        electronic_response(
            supplied,
            selection=MaterialSelection(electronic="gas-polynomial"),
            wavelength=800e-9,
            thg=True,
        )


@dataclasses.dataclass(frozen=True)
class _DuckModel:
    """An object with an optmat identity, a fit's parameters and a support record."""

    identity: object
    parameters: tuple[float, ...]
    support: object


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("a", math.nan, "non-finite value"),
        ("leg_bound", math.inf, "non-finite value"),
        ("gamma3_static", 0.0, "must be positive"),
        ("leg_bound", -1e7, "must be positive"),
    ],
)
def test_malformed_coefficients_are_refused(field, value, match):
    argon = OptmatKerr("Ar").dispersive_model()
    with pytest.raises(MaterialError, match=match):
        dataclasses.replace(argon, **{field: value})


def test_a_model_without_a_support_record_is_refused_by_name(monkeypatch):
    """A caller's own parameters have no record of where they are supported."""
    argon = OptmatKerr("Ar").model
    duck = _DuckModel(argon.identity, tuple(argon.parameters), None)
    monkeypatch.setattr(OptmatKerr, "model", property(lambda self: duck))
    with pytest.raises(MaterialError, match="no dispersion parameters and support"):
        OptmatKerr("Ar").dispersive_model()


def test_a_support_record_of_another_model_is_refused():
    """The record must describe the very model whose legs it bounds."""
    argon = OptmatKerr("Ar").model
    helium = OptmatKerr("He").model
    duck = _DuckModel(argon.identity, tuple(argon.parameters), helium.support)
    with pytest.raises(MaterialError, match="support record describes the model"):
        DispersiveKerrModel.from_optmat(duck)
