# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import math
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import pytest

from former_raman import FORMER_H2, FORMER_N2, former_response
from rate_oracles import adk_oracle
from solprop.boundaries import absorber
from solprop.constants import CHARGE_MASS_RATIO, EPS0, C
from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import MarcatiliMode
from solprop.modes.quadrature import taper_scale
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.nonlinear.normalisation import mode_average_prefactor, uppe_prefactor
from solprop.nonlinear.raman import response_for
from solprop.nonlinear.responses import (
    KerrResponse,
    PlasmaResponse,
    ResponseTraits,
)
from solprop.physdata import Gas, ionisation_potential
from solprop.profiles import Constant, Linear, PressureGradient
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.ionisation import build_rate_table
from solprop.translate.linop import profile_table
from solprop.translate.raman import undamped_kernels
from solprop.translate.responses import (
    THG_OVERSAMPLING,
    CompiledResponses,
    PlasmaCoefficients,
    RamanCoefficients,
    ResponseStage,
    _log_prefactor,
    compile_responses,
    kerr_coefficients,
    prefactor_at,
    raman_coefficients,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)


@dataclass(frozen=True)
class Spectral:
    """A response with a domain the translation layer cannot compile yet.

    The spectral-domain responses are evaluated after the forward transform
    and have a field in the contract and no compiler. Named for the domain rather than
    for a material, because the convolution domain beside it *is* compiled now.
    """

    @property
    def traits(self):
        return ResponseTraits("spectral", ("field",), "scalar", ("density",))


@dataclass(frozen=True)
class Unlabelled:
    """A response that does not even declare traits."""


@pytest.fixture(scope="module")
def field_grid():
    return FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 400e-15)


@pytest.fixture(scope="module")
def envelope_grid():
    return EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 400e-15)


def test_an_absent_raman_declaration_has_no_compiled_kernel(envelope_grid):
    assert (
        raman_coefficients(
            None, None, envelope_grid, Constant(1.0), n_pol=1, span=1.0, tol=1e-12
        )
        is None
    )


@pytest.fixture(scope="module")
def thg_grid():
    return EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 400e-15, thg=True)


@pytest.fixture(scope="module")
def mode():
    return MarcatiliMode(125e-6, "He", 1.0, cladding=1.45)


@pytest.fixture(scope="module")
def gradient():
    return MarcatiliMode(
        125e-6, "He", PressureGradient.two_point(1.0, 5.0, 0.5), cladding=1.45
    )


def compile_for(grid, mode, responses=(), **overrides):
    request = {"density": Constant(1.0), "z_max": 1.0} | overrides
    return compile_responses(responses, grid, mode, **request)


# -------------------------------------------------------------- the Kerr coefficients


def test_a_field_grid_with_the_harmonic_uses_the_real_cubic(field_grid):
    compiled = kerr_coefficients(KerrResponse(chi3=3.0, thg=True), field_grid)
    assert compiled.cubic_form == "real"
    assert compiled.coefficient == EPS0 * 3.0
    assert (compiled.self_factor, compiled.conjugate_factor) == (0.0, 0.0)
    assert compiled.harmonic_factor == 0.0
    assert not compiled.analytic_signal
    assert compiled.take_real


def test_a_field_grid_without_the_harmonic_uses_the_analytic_signal(field_grid):
    compiled = kerr_coefficients(KerrResponse(chi3=3.0, thg=False), field_grid)
    assert compiled.cubic_form == "analytic"
    # The weights of `_fundamental_cubic`, read off the reference implementation:
    # (1/2) sum|a|^2 a + (1/4) sum a^2 conj(a), which is (3/4)|a|^2 a for one component.
    assert (compiled.self_factor, compiled.conjugate_factor) == (0.5, 0.25)
    assert compiled.analytic_signal
    assert compiled.take_real


def test_an_envelope_grid_needs_no_analytic_signal(envelope_grid):
    compiled = kerr_coefficients(KerrResponse(chi3=3.0, thg=False), envelope_grid)
    assert compiled.cubic_form == "analytic"
    assert not compiled.analytic_signal
    assert not compiled.take_real
    assert compiled.harmonic_factor == 0.0


def test_an_envelope_grid_that_resolves_the_harmonic_carries_it(thg_grid):
    compiled = kerr_coefficients(KerrResponse(chi3=3.0, thg=True), thg_grid)
    assert compiled.harmonic_factor == 0.25
    assert thg_grid.ffac == THG_OVERSAMPLING


def test_a_harmonic_that_would_alias_is_refused(envelope_grid):
    """An envelope grid built without `thg` oversamples by two; six are needed."""
    assert envelope_grid.ffac < THG_OVERSAMPLING
    with pytest.raises(TranslationError, match="Build it with thg=True"):
        kerr_coefficients(KerrResponse(chi3=3.0, thg=True), envelope_grid)


# ------------------------------------------------------------------ the prefactor


def test_the_prefactor_is_the_mode_averaged_one_times_the_crop_scale(
    envelope_grid, mode
):
    compiled = compile_for(envelope_grid, mode, [KerrResponse(chi3=1e-52, thg=False)])
    expected = mode_average_prefactor(mode, envelope_grid) * envelope_grid.crop_scale
    assert compiled.prefactor_constant is not None
    assert np.array_equal(compiled.prefactor_constant[0, 0], expected)
    assert compiled.prefactor_constant.shape == (1, 1, envelope_grid.n_omega)
    assert compiled.prefactor_value_shape == (1, 1, envelope_grid.n_omega)


def test_the_multimode_prefactor_leaves_the_area_to_the_transform(envelope_grid, mode):
    compiled = compile_for(envelope_grid, mode, mode_averaged=False)
    expected = uppe_prefactor(mode, envelope_grid) * envelope_grid.crop_scale
    assert compiled.prefactor_constant is not None
    assert np.array_equal(compiled.prefactor_constant[0, 0], expected)


def prefactor_kinks(compiled):
    """Return the hard kinks of whichever representation carries the prefactor."""
    if compiled.prefactor_expansion is not None:
        return compiled.prefactor_expansion.weights.hard_kinks
    return cast("PiecewiseChebyshev", compiled.prefactor).hard_kinks


def test_the_prefactor_is_zero_outside_the_active_band(envelope_grid, mode, gradient):
    """Raw, unnormalised polarisation is not left out of band."""
    constant = compile_for(envelope_grid, mode).prefactor_constant
    assert constant is not None
    outside = constant[0, 0][~envelope_grid.active_mask]
    assert outside.size > 0
    assert np.all(outside == 0.0)
    # And at every position of the table, not only at the entrance: the band mask does
    # not move with z, so a bin that is out of band is out of band for the whole run.
    compiled = compile_for(envelope_grid, gradient, density=gradient.density)
    for z in (0.0, 0.4, 1.0):
        here = prefactor_at(compiled, z)
        assert np.all(here[0, 0][~envelope_grid.active_mask] == 0.0)


def test_the_prefactor_is_stored_for_every_mode_and_polarisation(envelope_grid):
    modes = [MarcatiliMode(125e-6, "He", 1.0, m=m, cladding=1.45) for m in (1, 2)]
    compiled = compile_for(envelope_grid, modes, n_pol=2, mode_averaged=False)
    constant = compiled.prefactor_constant
    assert constant is not None
    assert constant.shape == (2, 2, envelope_grid.n_omega)
    assert compiled.prefactor_value_shape == (2, 2, envelope_grid.n_omega)
    assert np.array_equal(constant[:, 0], constant[:, 1])
    assert not np.array_equal(constant[0, 0], constant[1, 0])


def test_the_shock_term_is_the_frequency_dependence_of_the_prefactor(
    envelope_grid, mode
):
    with_shock = compile_for(envelope_grid, mode, shock=True).prefactor_constant
    without = compile_for(envelope_grid, mode, shock=False).prefactor_constant
    assert with_shock is not None
    assert without is not None
    active = envelope_grid.active_mask
    # Both are exactly zero out of band, so the ratio is taken in band only.
    ratio = with_shock[0, 0][active] / without[0, 0][active]
    np.testing.assert_allclose(
        ratio, envelope_grid.omega[active] / envelope_grid.omega0
    )


# --------------------------------------------------------- the prefactor along z


def test_the_prefactor_table_reproduces_the_physics_along_a_gradient(
    envelope_grid, gradient
):
    """The freeze this replaces was dropping 2.5e-4 relative over the target fill."""
    compiled = compile_for(envelope_grid, gradient, density=gradient.density)
    assert compiled.prefactor_span == (0.0, 1.05)
    assert compiled.prefactor_constant is None
    rng = np.random.default_rng(20260905)
    worst = 0.0
    swing = 0.0
    entrance = mode_average_prefactor(gradient, envelope_grid, 0.0)
    for z in rng.uniform(0.0, 1.0, 20):
        # The bore is constant along a fill, so the area factor the table holds at the
        # reference position is the area factor at z, and the two forms coincide.
        expected = (
            mode_average_prefactor(gradient, envelope_grid, float(z))
            * envelope_grid.crop_scale
        )
        scale = float(np.max(np.abs(expected)))
        deviation = np.abs(prefactor_at(compiled, float(z))[0, 0] - expected)
        worst = max(worst, float(np.max(deviation)) / scale)
        swing = max(
            swing,
            float(np.max(np.abs(expected / envelope_grid.crop_scale - entrance)))
            / float(np.max(np.abs(entrance))),
        )
    # Measured 5.5e-15 on the RDW capillary and 3e-15 here; the table is built to 1e-12
    # relative to its value scale, which is the bound asserted.
    assert worst < 1e-12
    # And the profile it carries is worth carrying: the freeze dropped this much.
    assert swing > 1e-5


def test_the_prefactor_table_is_exact_at_the_entrance(envelope_grid, gradient):
    compiled = compile_for(envelope_grid, gradient, density=gradient.density)
    expected = (
        mode_average_prefactor(gradient, envelope_grid, 0.0) * envelope_grid.crop_scale
    )
    # The representation is built from the same expression and the entrance is a node
    # of it, so the value there is the sampled one; a Clenshaw sum and a linear
    # combination still cost round-off, so this is asserted to that rather than
    # bitwise. Measured 0.0 with the table and 2.2e-16 with the expansion.
    np.testing.assert_allclose(
        prefactor_at(compiled, 0.0)[0, 0], expected, rtol=1e-14, atol=0.0
    )


def test_a_taper_leaves_the_effective_area_to_the_transform(envelope_grid):
    """The identity that stops the area being counted one and a half times.

    Section 4's ``taper_scale`` already carries ``A_eff(z)`` and divides the projection
    by it, so the table holds the area at the reference position and only
    ``Re n_eff(omega, z)`` varies. What the two supply together is the full
    mode-averaged prefactor at every position.
    """
    mode = MarcatiliMode(Linear(125e-6, 75e-6, 0.8), "He", 2.0, cladding=1.45)
    # The taper's corner, as `assemble` passes it from the operator's hard kinks.
    compiled = compile_for(envelope_grid, mode, z_max=1.0, mandatory_edges=[0.8])
    assert prefactor_kinks(compiled).tolist() == [0.8]
    for z in (0.1, 0.5, 0.8, 1.0):
        expected = (
            mode_average_prefactor(mode, envelope_grid, z)
            * taper_scale(mode, z)
            * envelope_grid.crop_scale
        )
        # Measured 7.2e-16: the identity is exact and only the representation's own
        # interpolation and the rescaling's round-off separate the two.
        np.testing.assert_allclose(
            prefactor_at(compiled, z)[0, 0], expected, rtol=1e-12, atol=0.0
        )


def test_a_z_constant_mode_carries_the_array_and_a_degenerate_table(
    envelope_grid, mode
):
    compiled = compile_for(envelope_grid, mode)
    assert compiled.prefactor_constant is not None
    # One interval of degree zero: a constant costs the table nothing.
    table = cast("PiecewiseChebyshev", compiled.prefactor)
    assert table.degrees.tolist() == [0]
    np.testing.assert_allclose(
        table(0.5)[0, 0],
        compiled.prefactor_constant[0, 0],
        rtol=1e-14,
        atol=0.0,
    )


def test_the_table_splits_on_the_edges_it_is_given(envelope_grid, gradient):
    """`assemble` passes the operator's hard kinks so the two tables agree on them."""
    compiled = compile_for(
        envelope_grid, gradient, density=gradient.density, mandatory_edges=[0.25, 0.5]
    )
    assert prefactor_kinks(compiled).tolist() == [0.25, 0.5]


def test_the_table_reports_what_it_cost_and_what_it_carries(
    envelope_grid, gradient, caplog
):
    with caplog.at_level(logging.DEBUG, logger="solprop.translate.responses"):
        compile_for(envelope_grid, gradient, density=gradient.density)
    assert "carrying a swing of" in caplog.text


def test_a_vanishing_prefactor_reports_no_swing(envelope_grid, mode, caplog):
    """A prefactor that vanishes has no scale to measure a swing against."""
    compiled = compile_for(envelope_grid, mode)
    empty = cast("PiecewiseChebyshev", compiled.prefactor)
    zeros = replace(
        compiled,
        prefactor=PiecewiseChebyshev(
            np.zeros_like(empty.coefficients),
            empty.breakpoints,
            empty.degrees,
            empty.hard_kinks,
        ),
        prefactor_constant=None,
        prefactor_representation="table",
    )
    with caplog.at_level(logging.DEBUG, logger="solprop.translate.responses"):
        _log_prefactor(zeros, 1.0)
    assert "swing of 0.000e+00" in caplog.text


# ------------------------------------------------------------------------- ionisation


def test_a_range_without_a_plasma_response_is_refused(envelope_grid, mode):
    with pytest.raises(TranslationError, match="without a plasma response"):
        compile_for(envelope_grid, mode, ionisation_field_range=(1e10, 1e12))


# ----------------------------------------------------------------- what cannot compile


def test_a_response_of_another_domain_is_refused_by_name(envelope_grid, mode):
    with pytest.raises(
        TranslationError,
        match=r"Spectral is not compilable by the translation layer yet \(domain",
    ):
        kerr = KerrResponse(chi3=1e-52, thg=False)
        compile_for(envelope_grid, mode, [kerr, Spectral()])


def test_a_response_without_traits_is_refused_too(envelope_grid, mode):
    with pytest.raises(TranslationError, match="Unlabelled is not compilable"):
        compile_for(envelope_grid, mode, [Unlabelled()])


def test_two_kerr_responses_are_refused(envelope_grid, mode):
    with pytest.raises(TranslationError, match="two KerrResponse responses"):
        compile_for(
            envelope_grid,
            mode,
            [KerrResponse(chi3=1e-52, thg=False), KerrResponse(chi3=2e-52, thg=False)],
        )


def test_the_fields_of_the_responses_not_declared_stay_empty(envelope_grid, mode):
    compiled = compile_for(envelope_grid, mode, [KerrResponse(chi3=1e-52, thg=False)])
    assert compiled.plasma_coefficients is None
    assert compiled.raman_coefficients is None
    assert compiled.raman_coefficients is None
    assert compiled.raman_coefficients is None
    assert compiled.spectral_responses == ()
    assert compiled.ionisation_table == ()


def test_no_responses_at_all_leaves_only_the_accumulation(envelope_grid, mode):
    compiled = compile_for(envelope_grid, mode)
    assert compiled.kerr_coefficients is None
    assert compiled.response_order == (ResponseStage.ACCUMULATE,)


# ------------------------------------------------------------------------- plasma

FIELD_RANGE = (1e9, 1e11)  # V/m: below argon's threshold to well past its barrier


def test_a_plasma_response_compiles_into_the_contract(field_grid, mode):
    response = PlasmaResponse(Gas.Ar, adk_oracle(Gas.Ar), preionisation=0.2)
    compiled = compile_for(
        field_grid,
        mode,
        [KerrResponse(chi3=1e-52), response],
        ionisation_field_range=FIELD_RANGE,
    )
    plasma = compiled.plasma_coefficients
    assert plasma is not None
    assert plasma.charge_mass_ratio == CHARGE_MASS_RATIO
    # The rate's own energy per event (SEP 3, D8): the oracle's, the former registry's
    # argon potential, which the response binds in place of optmat's.
    assert plasma.ionisation_potential == adk_oracle("Ar").ionisation_potential
    assert float(plasma.preionisation(0.5)) == pytest.approx(0.2, rel=1e-14)
    # One rate reaches the kernel, not two: the group and the contract's own field are
    # the same object, which `validate` checks by identity.
    assert len(compiled.ionisation_table) == 1
    assert compiled.ionisation_table[0] is plasma.rate


def test_the_plasma_stages_come_before_the_kerr_term(field_grid, mode):
    compiled = compile_for(
        field_grid,
        mode,
        [KerrResponse(chi3=1e-52), PlasmaResponse(Gas.Ar, adk_oracle(Gas.Ar))],
        ionisation_field_range=FIELD_RANGE,
    )
    assert compiled.response_order == (
        ResponseStage.INTENSITY,
        ResponseStage.IONISATION_RATE,
        ResponseStage.PLASMA_SCAN,
        ResponseStage.PLASMA_CURRENT,
        ResponseStage.KERR,
        ResponseStage.ACCUMULATE,
    )


def test_a_plasma_alone_has_no_kerr_stage(field_grid, mode):
    compiled = compile_for(
        field_grid,
        mode,
        [PlasmaResponse(Gas.Ar, adk_oracle(Gas.Ar))],
        ionisation_field_range=FIELD_RANGE,
    )
    assert compiled.kerr_coefficients is None
    assert ResponseStage.KERR not in compiled.response_order


def test_a_rate_table_handed_to_the_response_is_taken_as_it_stands(field_grid, mode):
    table = build_rate_table(adk_oracle(Gas.Ar), FIELD_RANGE)
    compiled = compile_for(field_grid, mode, [PlasmaResponse(Gas.Ar, table)])
    plasma = compiled.plasma_coefficients
    assert plasma is not None
    assert np.array_equal(plasma.rate.log_rate, table.log_rate)
    assert plasma.rate.key == table.key
    # The builder hands out its memoised object, whose arrays are writeable; everything
    # crossing the seam is frozen, so what the contract carries is a frozen copy.
    assert not plasma.rate.log_rate.flags.writeable
    assert not plasma.rate.coefficients.flags.writeable


def test_a_pre_ionisation_path_is_tabulated_along_z(field_grid, gradient):
    response = PlasmaResponse(Gas.Ar, adk_oracle(Gas.Ar), preionisation=0.4)
    compiled = compile_for(
        field_grid,
        gradient,
        [response],
        ionisation_field_range=FIELD_RANGE,
        z_max=0.8,
    )
    plasma = compiled.plasma_coefficients
    assert plasma is not None
    assert isinstance(plasma.preionisation, PiecewiseChebyshev)
    # A plain number is a constant path, so it reads the same at every position and
    # over the same span as the other tables (validate checks that span).
    lo, hi = plasma.preionisation.z_span
    assert (lo, hi) == (0.0, pytest.approx(0.8 * 1.05))
    assert float(plasma.preionisation(hi)) == pytest.approx(0.4, rel=1e-14)


def test_an_envelope_grid_refuses_the_plasma_by_name(envelope_grid, mode):
    with pytest.raises(
        TranslationError, match="PlasmaResponse is defined on the field"
    ):
        compile_for(
            envelope_grid,
            mode,
            [PlasmaResponse(Gas.Ar, adk_oracle(Gas.Ar))],
            ionisation_field_range=FIELD_RANGE,
        )


def test_two_plasma_responses_are_refused(field_grid, mode):
    with pytest.raises(
        TranslationError, match="two PlasmaResponse responses on the same species"
    ):
        compile_for(
            field_grid,
            mode,
            [
                PlasmaResponse(Gas.Ar, adk_oracle(Gas.Ar)),
                PlasmaResponse(Gas.Ne, adk_oracle(Gas.Ne)),
            ],
            ionisation_field_range=FIELD_RANGE,
        )


def test_a_plasma_without_a_field_range_is_refused(field_grid, mode):
    with pytest.raises(TranslationError, match="needs ionisation_field_range"):
        compile_for(field_grid, mode, [PlasmaResponse(Gas.Ar, adk_oracle(Gas.Ar))])


def test_a_field_range_beside_a_ready_table_is_refused(field_grid, mode):
    table = build_rate_table(adk_oracle(Gas.Ar), FIELD_RANGE)
    with pytest.raises(TranslationError, match="states its own span"):
        compile_for(
            field_grid,
            mode,
            [PlasmaResponse(Gas.Ar, table)],
            ionisation_field_range=FIELD_RANGE,
        )


def test_a_bare_callable_rate_cannot_be_tabulated(field_grid, mode):
    def rate(field_amplitude):
        return np.where(np.abs(np.asarray(field_amplitude, float)) > 1e10, 1e12, 0.0)

    with pytest.raises(
        TranslationError, match="neither a threshold nor its parameters"
    ):
        compile_for(
            field_grid,
            mode,
            [PlasmaResponse(Gas.Ar, rate)],
            ionisation_field_range=FIELD_RANGE,
        )


@pytest.mark.parametrize("name", ["charge_mass_ratio", "ionisation_potential"])
@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_a_plasma_constant_that_cannot_give_a_current_is_refused(name, bad):
    table = build_rate_table(adk_oracle(Gas.Ar), FIELD_RANGE)
    good = {
        "charge_mass_ratio": CHARGE_MASS_RATIO,
        "ionisation_potential": ionisation_potential(Gas.Ar),
        "preionisation": PiecewiseChebyshev(
            np.zeros((1, 1)), np.array([0.0, 1.0]), np.array([0]), np.array([])
        ),
        "rate": table,
    }
    with pytest.raises(TranslationError, match=f"{name} must be positive"):
        PlasmaCoefficients(**(good | {name: bad}))


# ---------------------------------------------------------------------- the absorber


def test_the_temporal_collar_crosses_as_a_rate_on_both_axes(envelope_grid, mode):
    collar = absorber(envelope_grid)
    compiled = compile_for(envelope_grid, mode, absorber=collar)
    assert compiled.temporal_absorber is not None
    assert compiled.temporal_absorber_fine is not None
    assert np.array_equal(compiled.temporal_absorber, collar.temporal_rate)
    assert np.array_equal(compiled.temporal_absorber_fine, collar.temporal_rate_fine)
    assert compiled.absorber_scheme == "nonlinear"


def test_the_scheme_is_recorded(envelope_grid, mode):
    compiled = compile_for(
        envelope_grid,
        mode,
        absorber=absorber(envelope_grid),
        absorber_scheme="nonlinear",
    )
    assert compiled.absorber_scheme == "nonlinear"


def test_no_absorber_means_no_rates(envelope_grid, mode):
    compiled = compile_for(envelope_grid, mode)
    assert compiled.temporal_absorber is None
    assert compiled.temporal_absorber_fine is None


def test_one_rate_without_the_other_is_refused(envelope_grid, mode):
    compiled = compile_for(envelope_grid, mode, absorber=absorber(envelope_grid))
    with pytest.raises(TranslationError, match="must both be present"):
        CompiledResponses(
            response_order=compiled.response_order,
            kerr_coefficients=None,
            ionisation_table=(),
            plasma_channels=(),
            raman_terms=(),
            density=compiled.density,
            spectral_responses=(),
            prefactor_representation=compiled.prefactor_representation,
            prefactor=compiled.prefactor,
            prefactor_expansion=compiled.prefactor_expansion,
            prefactor_constant=compiled.prefactor_constant,
            temporal_absorber=compiled.temporal_absorber,
            temporal_absorber_fine=None,
        )


def test_a_prefactor_representation_without_its_member_is_refused(envelope_grid, mode):
    """The kind and the member that carries it are one statement, checked both ways."""
    compiled = compile_for(envelope_grid, mode)
    with pytest.raises(TranslationError, match="prefactor_expansion must be present"):
        CompiledResponses(
            response_order=compiled.response_order,
            kerr_coefficients=None,
            ionisation_table=(),
            plasma_channels=(),
            raman_terms=(),
            density=compiled.density,
            spectral_responses=(),
            prefactor_representation="expansion",
            prefactor=compiled.prefactor,
            prefactor_expansion=None,
            prefactor_constant=None,
            temporal_absorber=None,
            temporal_absorber_fine=None,
        )


# ------------------------------------------------------------------------- the density


def test_the_density_becomes_a_table_over_the_run_and_its_margin(
    envelope_grid, gradient
):
    compiled = compile_for(envelope_grid, gradient, density=gradient.density)
    assert compiled.density.z_span == (0.0, 1.05)
    assert compiled.density.hard_kinks.tolist() == [1.0]
    for z in (0.0, 0.5, 1.0):
        expected = float(gradient.density(np.array([z]))[0])
        assert float(compiled.density(z)) == pytest.approx(expected, rel=1e-12)


def test_a_scalar_density_becomes_a_constant_table(envelope_grid, mode):
    compiled = compile_for(envelope_grid, mode, density=2.5)
    assert float(compiled.density(0.4)) == 2.5


# ------------------------------------------------------------------------------ errors


@pytest.mark.parametrize("z_max", [0.0, -1.0, math.inf])
def test_a_bad_run_length_is_refused(envelope_grid, mode, z_max):
    with pytest.raises(TranslationError, match="z_max must be positive"):
        compile_for(envelope_grid, mode, z_max=z_max)


def test_an_empty_mode_set_is_refused(envelope_grid):
    with pytest.raises(TranslationError, match="at least one mode"):
        compile_for(envelope_grid, [])


def test_a_constant_prefactor_of_the_wrong_shape_is_refused(envelope_grid, mode):
    compiled = compile_for(envelope_grid, mode)
    with pytest.raises(TranslationError, match="prefactor_constant has shape"):
        CompiledResponses(
            response_order=compiled.response_order,
            kerr_coefficients=None,
            ionisation_table=(),
            plasma_channels=(),
            raman_terms=(),
            density=compiled.density,
            spectral_responses=(),
            prefactor_representation=compiled.prefactor_representation,
            prefactor=compiled.prefactor,
            prefactor_expansion=compiled.prefactor_expansion,
            prefactor_constant=np.zeros((1, 1, 7), dtype=complex),
            temporal_absorber=None,
            temporal_absorber_fine=None,
        )


def test_a_path_two_modes_share_is_resolved_once(envelope_grid):
    """Two modes of one tapered fibre carry the same bore object, not copies."""
    bore = Linear(125e-6, 75e-6, 0.8)
    modes = [MarcatiliMode(bore, "He", 2.0, m=m, cladding=1.45) for m in (1, 2)]
    assert modes[0].core_radius is modes[1].core_radius
    compiled = compile_for(
        envelope_grid, modes, z_max=1.0, n_pol=1, mode_averaged=False
    )
    assert compiled.prefactor_value_shape == (2, 1, envelope_grid.n_omega)


def test_an_exhausted_interval_budget_warns(envelope_grid, gradient, caplog):
    with caplog.at_level(logging.WARNING, logger="solprop.translate.responses"):
        compiled = compile_for(
            envelope_grid,
            gradient,
            density=gradient.density,
            n_max=4,
            max_intervals=1,
        )
    assert "interval budget was exhausted" in caplog.text
    assert cast("PiecewiseChebyshev", compiled.prefactor).budget_exceeded


# ------------------------------------------------------------- the evaluation path


def test_prefactor_at_takes_the_array_on_the_fast_path(envelope_grid, mode):
    compiled = compile_for(envelope_grid, mode)
    assert prefactor_at(compiled, 0.5) is compiled.prefactor_constant
    with pytest.raises(TranslationError, match="outside the table"):
        prefactor_at(compiled, 2.0)


def test_prefactor_at_evaluates_the_expansion_on_a_gradient(envelope_grid, gradient):
    compiled = compile_for(envelope_grid, gradient, density=gradient.density)
    assert compiled.prefactor_constant is None
    assert compiled.prefactor_representation == "expansion"
    expansion = compiled.prefactor_expansion
    assert expansion is not None
    for z in (0.0, 0.3, 1.0):
        assert np.array_equal(prefactor_at(compiled, z), expansion(z))


def test_prefactor_at_evaluates_the_z_table_when_that_is_what_was_built(
    envelope_grid, gradient
):
    compiled = compile_for(
        envelope_grid, gradient, density=gradient.density, representation="table"
    )
    assert compiled.prefactor_representation == "table"
    table = cast("PiecewiseChebyshev", compiled.prefactor)
    for z in (0.0, 0.3, 1.0):
        assert np.array_equal(prefactor_at(compiled, z), table(z))


# ---------------------------------------------------------------------- the Raman rows


def test_a_raman_response_fills_the_three_rows(envelope_grid, mode):
    """The rows of section 5, from the kernel sampling and the density path."""
    raman = response_for("SiO2", amplitude=1e-22, fraction=0.18)
    compiled = compile_for(
        envelope_grid, mode, [KerrResponse(chi3=1e-52, thg=False), raman]
    )
    expected = undamped_kernels(raman, envelope_grid)
    assert compiled.raman_coefficients is not None
    assert compiled.raman_coefficients.undamped.shape == (1, 2 * envelope_grid.n_to)
    assert np.array_equal(compiled.raman_coefficients.undamped, expected)
    assert not compiled.raman_coefficients.undamped.flags.writeable
    assert compiled.raman_coefficients.fraction == 0.18
    assert len(compiled.raman_coefficients.decay_rates) == 1


def test_a_molecular_gas_carries_one_row_per_independently_damped_term(
    field_grid, mode
):
    """A gas with a rotation and a vibration has two dephasing times, so the
    kernel cannot be one damped array."""
    compiled = compile_for(
        field_grid,
        mode,
        [KerrResponse(chi3=1e-52, thg=True), former_response(FORMER_H2)],
        density=Constant(2.5e25),
    )
    assert compiled.raman_coefficients is not None
    assert compiled.raman_coefficients is not None
    assert compiled.raman_coefficients.undamped.shape[0] == 2
    assert len(compiled.raman_coefficients.decay_rates) == 2
    # An absolute molecular response takes the whole of its own strength, so the row the
    # contract promises is 1 and not the ``None`` the response's own ``fraction`` is.
    leaf = compiled.raman_coefficients
    assert leaf is not None
    assert leaf.fraction == 1.0


def test_the_raman_stage_follows_the_kerr_term(field_grid, mode):
    compiled = compile_for(
        field_grid,
        mode,
        [KerrResponse(chi3=1e-52, thg=True), former_response(FORMER_N2)],
        density=Constant(2.5e25),
    )
    assert compiled.response_order == (
        ResponseStage.KERR,
        ResponseStage.RAMAN_DRIVE,
        ResponseStage.ACCUMULATE,
    )


def test_a_raman_response_alone_needs_no_kerr_on_an_envelope_grid(envelope_grid, mode):
    """An envelope grid's drive is ``|A|^2/2`` whatever the harmonic flag says, so
    nothing has to be read off a Kerr term there."""
    compiled = compile_for(
        envelope_grid, mode, [response_for("SiO2", amplitude=1e-22, fraction=0.5)]
    )
    assert compiled.response_order == (
        ResponseStage.RAMAN_DRIVE,
        ResponseStage.ACCUMULATE,
    )
    assert compiled.kerr_coefficients is None


@pytest.mark.parametrize("z", [0.0, 0.25, 0.9])
def test_the_decay_tables_follow_the_density_along_a_gradient(gradient, field_grid, z):
    """The rate is composed from the term's own dephasing and the density path, and
    tabulated along z like everything else that varies (``kernel-contract.md`` 5)."""
    raman = former_response(FORMER_N2)
    density = gradient.density
    compiled = compile_for(
        field_grid,
        gradient,
        [KerrResponse(chi3=1e-52, thg=True), raman],
        density=density,
    )
    here = float(density(np.array([z]))[0])
    leaf = compiled.raman_coefficients
    assert leaf is not None
    for model, table in zip(raman.models, leaf.decay_rates, strict=True):
        exact = model.dephasing.decay_rate(here)
        # The Chebyshev table of the rate is built to the same tolerance as every other
        # path along z; measured 8.2e-15 relative at worst over these three positions,
        # against a rate that spans a factor of five over the fill.
        assert float(table(z)) == pytest.approx(exact, rel=1e-12)
    assert float(leaf.decay_rates[0](0.0)) != float(leaf.decay_rates[0](0.9))


def test_two_raman_responses_are_refused(envelope_grid, mode):
    with pytest.raises(TranslationError, match="two RamanResponse responses"):
        compile_for(
            envelope_grid,
            mode,
            [
                response_for("SiO2", amplitude=1e-22),
                response_for("SiO2", amplitude=2e-22),
            ],
        )


def test_a_vector_problem_refuses_the_raman_response(envelope_grid, mode):
    """The response declares itself scalar and there is no tensor kernel."""
    with pytest.raises(TranslationError, match="tensor"):
        compile_for(
            envelope_grid,
            mode,
            [
                KerrResponse(chi3=1e-52, thg=False),
                response_for("SiO2", amplitude=1e-22),
            ],
            n_pol=2,
        )


def test_a_field_grid_raman_without_a_kerr_response_is_refused(field_grid, mode):
    """The drive form is a property of the problem and reaches a backend through the
    Kerr row, so on a field grid there has to be one."""
    with pytest.raises(TranslationError, match="needs a Kerr response beside it"):
        compile_for(
            field_grid, mode, [former_response(FORMER_N2)], density=Constant(2.5e25)
        )


@pytest.mark.parametrize("thg", [True, False])
def test_a_field_grid_refuses_two_responses_that_disagree_about_the_harmonic(
    field_grid, mode, thg
):
    """A drive that keeps the 2 omega beat radiates a third harmonic through the later
    multiplication by the field, and a grid built without one cannot hold it."""
    with pytest.raises(TranslationError, match="the two drives must agree"):
        compile_for(
            field_grid,
            mode,
            [
                KerrResponse(chi3=1e-52, thg=thg),
                former_response(FORMER_N2, thg=not thg),
            ],
            density=Constant(2.5e25),
        )


@pytest.mark.parametrize("thg", [True, False])
def test_two_responses_that_agree_about_the_harmonic_compile(field_grid, mode, thg):
    compiled = compile_for(
        field_grid,
        mode,
        [KerrResponse(chi3=1e-52, thg=thg), former_response(FORMER_N2, thg=thg)],
        density=Constant(2.5e25),
    )
    # The backend reads the drive form off this flag: analytic_signal is set exactly
    # when the field grid's drive is |E_a|^2/2 rather than E^2.
    assert compiled.kerr_coefficients is not None
    assert compiled.kerr_coefficients.analytic_signal is (not thg)


def test_an_envelope_grid_does_not_mind_a_disagreement(envelope_grid, mode):
    """``thg`` is ignored by an envelope grid's quadratic drive, so there is nothing to
    refuse: the harmonic a Kerr term radiates there is a separate cubic term."""
    compiled = compile_for(
        envelope_grid,
        mode,
        [
            KerrResponse(chi3=1e-52, thg=False),
            response_for("SiO2", amplitude=1e-22, thg=True),
        ],
    )
    leaf = compiled.raman_coefficients
    assert leaf is not None
    assert leaf.fraction == 1.0


def test_an_expansion_that_cannot_be_built_is_refused_when_it_is_demanded(
    envelope_grid, gradient
):
    """A GNLSE mode beside a capillary one: there is no shared parameter space."""
    other = SpecifiedMode.from_taylor([0.0, 0.0, 2e-26], omega_ref=envelope_grid.omega0)
    with pytest.raises(TranslationError, match="could not be built as one"):
        compile_for(
            envelope_grid,
            [gradient, other],
            density=gradient.density,
            mode_averaged=False,
            representation="expansion",
        )


def test_a_mode_set_without_a_shared_parameter_space_takes_the_prefactor_table(
    envelope_grid, gradient
):
    # A non-zero beta_0, so that the prefactor's 1 / Re n_eff has a value everywhere.
    other = SpecifiedMode.from_taylor(
        [1.45 * envelope_grid.omega0 / C, 1.47 / C, 2e-26],
        omega_ref=envelope_grid.omega0,
    )
    compiled = compile_for(
        envelope_grid,
        [gradient, other],
        density=gradient.density,
        mode_averaged=False,
    )
    assert compiled.prefactor_representation == "table"


def test_the_raman_leaf_refuses_a_kernel_with_no_terms():
    """A kernel with no rows damps nothing; the rank is the annotation's business."""
    with pytest.raises(TranslationError, match="at least one term"):
        RamanCoefficients(
            np.zeros((0, 8)),
            (),
            0.5,
            thg=False,
            tensor_weights=np.zeros((0, 2)),
            tensor_capable=False,
        )


def test_the_raman_leaf_accepts_the_shape_the_compiler_gives_it():
    rates = (profile_table(Constant(1e12), 0.0, 1.0),)
    leaf = RamanCoefficients(
        np.zeros((1, 8)),
        rates,
        0.5,
        thg=True,
        tensor_weights=np.array([[1.0, 0.0]]),
        tensor_capable=False,
    )
    assert leaf.n_terms == 1
    assert leaf.thg
    assert not leaf.undamped.flags.writeable


def test_a_prefactor_table_without_its_kind_is_refused(envelope_grid, mode):
    compiled = compile_for(envelope_grid, mode)
    with pytest.raises(TranslationError, match="prefactor must be present"):
        CompiledResponses(
            response_order=compiled.response_order,
            kerr_coefficients=None,
            ionisation_table=(),
            plasma_channels=(),
            raman_terms=(),
            density=compiled.density,
            spectral_responses=(),
            prefactor_representation="table",
            prefactor=None,
            prefactor_expansion=None,
            prefactor_constant=None,
            temporal_absorber=None,
            temporal_absorber_fine=None,
        )


def test_the_group_refuses_a_constant_beside_a_kind_that_is_not_one(
    envelope_grid, mode
):
    """The three prefactor members and the kind are one statement, checked both ways.

    A constant present with a ``"table"`` or ``"expansion"`` kind is the mirror of the
    kind saying ``"constant"`` with nothing to show for it, and the group names either.
    """
    compiled = compile_for(envelope_grid, mode)
    assert compiled.prefactor_representation == "constant"
    with pytest.raises(TranslationError, match="prefactor_constant must be present"):
        replace(compiled, prefactor_representation="table")
