# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A caller's own glass data, and the former silica source names (SEP 3, Phase 4b).

Every built-in solid is optmat's (tests/physdata/test_optmat_solids.py). What is left
of solprop's glasses is the supplied types -- :class:`SellmeierGlass`,
:class:`TabulatedGlass`, :class:`CompositeGlass` -- which these tests build from
test-local data, and the mapping of a former ``source=`` name to the optmat model of
the same published formula, or its refusal.
"""

import math
import re
import warnings

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from solprop.constants import C
from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.physdata import (
    CompositeGlass,
    GlassIndex,
    OptmatGlass,
    SellmeierGlass,
    TabulatedGlass,
    fused_silica,
    glass,
)


def omega_of(wavelength):
    return 2.0 * math.pi * C / wavelength


# Malitson (1965), the expansion as printed: J. Opt. Soc. Am. 55, 1205,
# doi:10.1364/JOSA.55.001205, stated for 0.21-3.71 um at 20 C. Supplied here as a
# caller's own data, the coefficients of the paper (optmat carries the same formula as
# malitson_1965/sellmeier, made absolute).
MALITSON = SellmeierGlass(
    name="SiO2",
    source="malitson (test data)",
    wavelength_range=(0.21e-6, 3.71e-6),
    citation=(
        "I. H. Malitson, J. Opt. Soc. Am. 55, 1205-1208 (1965), "
        "doi:10.1364/JOSA.55.001205"
    ),
    poles=((0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161)),
)

# Malitson (1965) Table I and the values every optics handbook quotes from it. The
# expansion reproduces its own tabulated data to about 1e-6, which is the accuracy
# Malitson claims for the interspecimen comparison, so 2e-6 is the right gate here.
MALITSON_PUBLISHED = {
    0.58756e-6: 1.458464,  # sodium d line, the quoted n_d of fused silica
    0.63280e-6: 1.457018,  # helium-neon
    0.48610e-6: 1.463128,  # hydrogen F line
    1.55000e-6: 1.444024,  # telecom band
    2.00000e-6: 1.438085,
}
MALITSON_TOLERANCE = 2e-6

EXTINCTION_FLOOR = 1e-30
"""The floor of the test table's log-k spline: below 1e-15 /m of absorption."""


def tabulated_glass():
    """Return a caller's table: Malitson's real part, absorbing at both ends.

    The extinction is exactly zero between 0.3 and 3 um, as a measured table of a
    transparent window stores it, and rises steeply at an ultraviolet and an infrared
    edge; 400 rows from 0.1 to 10 um, sampled evenly in log wavelength as a measured
    table roughly is.
    """
    wavelength_um = np.geomspace(0.1, 10.0, 400)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        real = np.real(MALITSON.index(omega_of(wavelength_um * 1e-6)))
    real = np.where(real > 0.5, real, 0.5)
    extinction = np.where(
        wavelength_um < 0.3,
        0.3 * np.exp(-(wavelength_um - 0.1) / 0.02),
        np.where(wavelength_um > 3.0, 1e-2 * (wavelength_um - 3.0) ** 2, 0.0),
    )
    log_wavelength = np.log(wavelength_um)
    return TabulatedGlass(
        name="SiO2",
        source="table (test data)",
        wavelength_range=(0.1e-6, 10e-6),
        citation="a synthetic table built by this test",
        provenance={"source": "tests/physdata/test_glasses.py"},
        _real=CubicSpline(log_wavelength, real),
        _log_extinction=CubicSpline(
            log_wavelength, np.log(np.maximum(extinction, EXTINCTION_FLOOR))
        ),
        _log_wavelength_limits=(float(log_wavelength[0]), float(log_wavelength[-1])),
    )


TABLE = tabulated_glass()


def composite_glass():
    """Return Malitson's real part with the test table's absorption."""
    return CompositeGlass(
        name="SiO2",
        source="malitson+table (test data)",
        wavelength_range=(
            max(MALITSON.wavelength_range[0], TABLE.wavelength_range[0]),
            min(MALITSON.wavelength_range[1], TABLE.wavelength_range[1]),
        ),
        citation=f"{MALITSON.citation}; {TABLE.citation}",
        real_part=MALITSON,
        absorption=TABLE,
    )


SUPPLIED = {
    "sellmeier": lambda: MALITSON,
    "table": lambda: TABLE,
    "composite": composite_glass,
}


# ------------------------------------------------------ named silica models


SILICA_NAMED = ["malitson_1965/sellmeier", "franta_2016/udm_lithosil_q2"]


@pytest.mark.parametrize("model", SILICA_NAMED)
def test_a_named_silica_model_is_that_optmat_model(model):
    index = fused_silica(model=model)
    assert isinstance(index, OptmatGlass)
    assert index.source == model
    assert glass("silica", model=model) == index


def test_the_malitson_model_is_absolute_where_the_paper_is_air_relative():
    """optmat's Malitson model is the paper's formula, relative to vacuum.

    Malitson measured in air, so the paper's index is the air-relative one; optmat
    makes it absolute, ``n = n_air n_rel``. At the d line their ratio is air's index:
    a refractivity of 2.76e-4 (measured), inside the 2.72e-4 (20 C) to 2.77e-4 (15 C)
    of dry air there, so 1e-5 separates the reference gas from any other error.
    """
    omega = omega_of(0.58756e-6)
    malitson = fused_silica(model="malitson_1965/sellmeier")
    absolute = float(np.real(malitson.index(omega)))
    relative = float(np.real(MALITSON.index(omega)))
    assert absolute / relative - 1.0 == pytest.approx(2.75e-4, abs=1e-5)


# ------------------------------------------------------ a supplied Sellmeier glass


@pytest.mark.parametrize(("wavelength", "published"), MALITSON_PUBLISHED.items())
def test_a_supplied_sellmeier_reproduces_the_published_values(wavelength, published):
    value = complex(MALITSON.index(omega_of(wavelength)))
    assert value.imag == 0.0
    assert abs(value.real - published) < MALITSON_TOLERANCE


def test_a_supplied_sellmeier_is_real_and_dispersive_in_band():
    omega = omega_of(np.linspace(0.25e-6, 3.0e-6, 64))
    index = MALITSON.index(omega)
    increasing = MALITSON.index(np.sort(omega))
    assert np.all(np.imag(index) == 0.0)
    assert np.all(np.real(index) > 1.4)
    # Normal dispersion: the index falls with wavelength, i.e. rises with omega.
    assert np.all(np.diff(np.real(increasing)) > 0.0)


def test_a_supplied_sellmeier_goes_evanescent_below_its_shortest_pole():
    """Just below the 68.4 nm pole n^2 is negative; the complex root stays defined.

    Luna takes the same complex square root. The point is that far outside the
    validity range the model returns a finite evanescent index rather than a NaN that
    would poison a whole linear operator, and that the sign of the root follows the
    package convention: a positive imaginary part is decay, never gain.
    """
    with pytest.warns(MaterialRangeWarning):
        value = complex(MALITSON.index(omega_of(60e-9)))
    assert math.isfinite(value.real)
    assert math.isfinite(value.imag)
    assert value.real == 0.0
    assert value.imag > 0.0


# --------------------------------------------------------- a supplied table


def test_a_supplied_table_never_returns_gain():
    """A spline through k could overshoot negative; one through log k cannot.

    Negative Im n is gain, and would turn capillary loss into amplification without
    any error. The guard is structural -- k is exp of a spline -- so this sweeps the
    whole table densely, across the steep edges into the exact zeros, rather than
    spot-checking.
    """
    shortest, longest = TABLE.wavelength_range
    omega = omega_of(np.geomspace(shortest * 1.001, longest * 0.999, 20001))
    index = TABLE.index(omega)
    assert np.all(np.imag(index) >= 0.0)
    assert np.all(np.real(index) > 0.0)
    assert np.all(np.isfinite(index))


def test_a_supplied_table_is_transparent_where_its_data_are_zero():
    """The floor of the log-k spline is 1e-30, an absorption below 1e-15 per metre.

    A table that stores exact zeros in its window must not invent loss there. 0.4 to
    1.1 um is well inside the zeros, away from the spline's ringing at the edges.
    """
    omega = omega_of(np.linspace(0.4e-6, 1.1e-6, 128))
    assert np.all(np.imag(TABLE.index(omega)) < 1e-20)
    # 2 omega k / c at 1 um with k at the floor is about 4e-15 per metre.
    assert 2.0 * omega_of(1e-6) * EXTINCTION_FLOOR / C < 1e-13


def test_a_supplied_table_absorbs_where_its_data_do():
    ultraviolet = complex(TABLE.index(omega_of(0.12e-6)))
    infrared = complex(TABLE.index(omega_of(6.0e-6)))
    assert ultraviolet.imag > 0.1
    assert infrared.imag > 1e-2


def test_a_supplied_table_is_held_constant_beyond_its_ends():
    """Constant, not extrapolated: a cubic over nine decades is meaningless."""
    shortest, longest = TABLE.wavelength_range
    edge = complex(TABLE.index(omega_of(longest)))
    short_edge = complex(TABLE.index(omega_of(shortest)))
    with pytest.warns(MaterialRangeWarning, match="SiO2 'table"):
        beyond = complex(TABLE.index(omega_of(longest * 10.0)))
    with pytest.warns(MaterialRangeWarning):
        slope = complex(TABLE.dindex_domega(omega_of(longest * 10.0)))
    with pytest.warns(MaterialRangeWarning):
        short_beyond = complex(TABLE.index(omega_of(shortest / 10.0)))
    # The held value is the edge value, to the rounding of exp(log k).
    assert beyond == pytest.approx(edge, rel=1e-12)
    assert short_beyond == pytest.approx(short_edge, rel=1e-12)
    assert slope == 0.0


def test_zero_frequency_is_a_tables_long_wavelength_limit():
    longest = TABLE.wavelength_range[1]
    with pytest.warns(MaterialRangeWarning):
        static = complex(TABLE.index(0.0))
    # As above, the held value to the rounding of exp(log k).
    assert static == pytest.approx(complex(TABLE.index(omega_of(longest))), rel=1e-12)


# ------------------------------------------------------- a supplied composite


def test_a_composite_takes_each_part_from_its_own_source():
    composite = composite_glass()
    omega = omega_of(np.linspace(0.3e-6, 3.0e-6, 64))
    # Each part is its source's own evaluation, selected rather than recomputed, so
    # the composite holds it exactly (a structural contract).
    np.testing.assert_array_equal(
        np.real(composite.index(omega)), np.real(MALITSON.index(omega))
    )
    np.testing.assert_array_equal(
        np.imag(composite.index(omega)), np.imag(TABLE.index(omega))
    )
    np.testing.assert_array_equal(
        np.real(composite.dindex_domega(omega)),
        np.real(MALITSON.dindex_domega(omega)),
    )
    np.testing.assert_array_equal(
        np.imag(composite.dindex_domega(omega)), np.imag(TABLE.dindex_domega(omega))
    )


# ------------------------------------------------- derivatives and ranges, all


@pytest.mark.parametrize("kind", SUPPLIED)
def test_analytic_derivative_matches_finite_differences(kind):
    """Three step sizes, with the floor of the comparison stated.

    A plain central difference is not the right instrument here. Its truncation error
    is (h^2/6) times the third derivative, which is large wherever tabulated data turn
    over. Richardson extrapolation of two steps cancels that h^2 term. 5e-8 is the
    bound the former built-in models were held to (9e-9 at worst then). The failure
    this guards against -- a missing or inverted chain-rule factor, since d/d(omega)
    is -(1/omega) d/d(log lambda) for the tables and 2 omega / k^2 d/du for the pole
    sum -- is never subtle enough to hide under that.
    """
    model = SUPPLIED[kind]()
    shortest, longest = model.wavelength_range
    omega = omega_of(np.geomspace(shortest * 1.05, longest * 0.95, 60))
    analytic = model.dindex_domega(omega)
    scale = np.max(np.abs(analytic))

    def central(step):
        return (model.index(omega + step) - model.index(omega - step)) / (2.0 * step)

    for relative_step in (1e-4, 1e-5, 1e-6):
        step = relative_step * omega
        richardson = (4.0 * central(step / 2.0) - central(step)) / 3.0
        assert np.max(np.abs(richardson - analytic)) / scale < 5e-8


@pytest.mark.parametrize("kind", SUPPLIED)
def test_permittivity_is_the_square_of_the_index(kind):
    model = SUPPLIED[kind]()
    omega = omega_of(np.linspace(0.3e-6, 3.0e-6, 32))
    np.testing.assert_allclose(
        model.permittivity(omega), model.index(omega) ** 2, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        model.dpermittivity_domega(omega),
        2.0 * model.index(omega) * model.dindex_domega(omega),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("kind", SUPPLIED)
def test_out_of_range_warns_and_names_the_source(kind):
    model = SUPPLIED[kind]()
    assert isinstance(model, GlassIndex)
    shortest = model.wavelength_range[0]
    with pytest.warns(MaterialRangeWarning, match=re.escape(f"SiO2 '{model.source}'")):
        model.index(omega_of(shortest / 2.0))


@pytest.mark.parametrize("kind", SUPPLIED)
def test_in_range_does_not_warn(kind):
    model = SUPPLIED[kind]()
    shortest, longest = model.wavelength_range
    omega = omega_of(np.geomspace(shortest * 1.01, longest * 0.99, 16))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.index(omega)
        model.dindex_domega(omega)


@pytest.mark.parametrize("kind", SUPPLIED)
def test_scalar_input_gives_a_zero_dimensional_result(kind):
    model = SUPPLIED[kind]()
    value = model.index(omega_of(1e-6))
    assert value.shape == ()
    assert model.index([omega_of(1e-6)]).shape == (1,)


@pytest.mark.parametrize("kind", SUPPLIED)
def test_bad_frequencies_are_rejected(kind):
    model = SUPPLIED[kind]()
    with pytest.raises(MaterialError, match="must not be negative"):
        model.index(-1.0)
    with pytest.raises(MaterialError, match="must be finite"):
        model.index(np.inf)
    with pytest.raises(MaterialError, match="must be real"):
        model.index(1j)
    with pytest.raises(MaterialError, match="must be real"):
        model.index("red")
