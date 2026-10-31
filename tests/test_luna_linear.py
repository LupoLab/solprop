# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Comparison against the linear operator, prefactors and Kerr responses of Luna.

The stored file is a regression witness, not truth: where solprop deliberately differs
from Luna the difference is asserted exactly, not merely tolerated. It was generated
once, by ``tests/data/generate_luna_linear.py``, and is frozen: never regenerated, every
input the comparison depends on stated in the test (``docs/dev/testing.md``); the file
records the Luna commit it came from.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import interpolate

from solprop.constants import EPS0, C
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.linear import Frame, LossPolicy, dispersion_operator, frame_quantities
from solprop.nonlinear.gnlse import (
    SpecifiedMode,
    attenuation_from_db,
    kerr_susceptibility,
    problem_from,
)
from solprop.nonlinear.normalisation import mode_average_prefactor
from solprop.nonlinear.responses import KerrResponse, thg_carrier

REFERENCE = json.loads((Path(__file__).parent / "data/luna_linear.json").read_text())
LUNA = REFERENCE["luna"]
PARAMETERS = REFERENCE["parameters"]
SIGNALS = REFERENCE["signals"]

# Luna's own inputs to solprop, rather than quantities solprop reproduces: the frequency
# axes, the band masks, and the beta, alpha and A_eff of the Marcatili mode, which are
# fed to a dispersion-specified mode so that the comparison is of the linear operator
# and not of the capillary model.
INPUTS = {
    "OMEGA_REAL",
    "OMEGA_ENV",
    "SIDX_REAL",
    "SIDX_ENV",
    "GN_OMEGA",
    "GN_SIDX",
    "BETA_REAL",
    "BETA_ENV",
    "ALPHA_REAL",
    "ALPHA_ENV",
    "CAP_AEFF",
    "BETA0_REAL",
    "BETA0_ENV",
    "GN_ALPHA_SCALAR",
    "CONSTANTS",
}

# Quantities solprop reproduces, each named by the test that checks it.
COMPARED = {
    "LINOP_REAL_RE",
    "LINOP_REAL_IM",
    "LINOP_ENV_RE",
    "LINOP_ENV_IM",
    "GN_LINOP_RE",
    "GN_LINOP_IM",
    "GN_BETA",
    "GN_ALPHA",
    "GN_BETA1",
    "NORM_REAL_RE",
    "NORM_REAL_IM",
    "NORM_ENV_RE",
    "NORM_ENV_IM",
    "GN_NORM_RE",
    "GN_NORM_IM",
    "GN_NORM_NOSHOCK_RE",
    "GN_NORM_NOSHOCK_IM",
    "MAP_1",
    "MAP_2",
    "MAP_3",
    "KERR_FIELD",
    "KERR_FIELD_NOTHG",
    "KERR_FIELD_VEC_X",
    "KERR_FIELD_VEC_Y",
    "KERR_ENV_RE",
    "KERR_ENV_IM",
    "KERR_ENV_VEC_X_RE",
    "KERR_ENV_VEC_X_IM",
    "KERR_ENV_VEC_Y_RE",
    "KERR_ENV_VEC_Y_IM",
    "KERR_ENV_THG_RE",
    "KERR_ENV_THG_IM",
}

# Quantities solprop deliberately does not reproduce, with the reason.
DIVERGENT = {
    "BETA1_REAL": "Luna differences beta(omega); solprop's is analytic (ADR-006)",
    "BETA1_ENV": "Luna differences beta(omega); solprop's is analytic (ADR-006)",
}


def stored(key):
    return np.asarray(LUNA[key], dtype=float)


def complex_stored(key):
    return stored(f"{key}_RE") + 1j * stored(f"{key}_IM")


def mask(key):
    return stored(key).astype(bool)


LUNA_EPS0, LUNA_C, _ = stored("CONSTANTS")


# Luna's constants come from PhysicalConstants.CODATA2014 and solprop's from
# scipy.constants, which is CODATA 2022. The vacuum permittivity moved by 1.33e-10
# relative between the two, and that is the whole of the difference in every quantity
# below that carries it: the Kerr polarisation is linear in eps0 and the mode-averaged
# prefactor goes as its inverse square root. Rescaling by the ratio makes the physics
# comparable to round-off instead of hiding a known constant behind a loose tolerance.
def rescaled(values, power):
    """Luna's values as they would be with solprop's eps0, given their power of it."""
    return values * (LUNA_EPS0 / EPS0) ** (-power)


# ------------------------------------------------------------------- provenance


def test_provenance_is_recorded():
    provenance = REFERENCE["provenance"]
    assert len(provenance["luna_commit"]) == 40
    assert provenance["luna_branch"] == "modal-fixed"
    assert provenance["julia_version"].startswith("julia version")


def test_every_stored_quantity_is_classified():
    # A new witness must be triaged as an input, a comparison or a documented
    # divergence, never silently ignored.
    assert set(LUNA) == INPUTS | COMPARED | set(DIVERGENT)


# ------------------------------------------------------------------------ grids


# The witness is frozen (``docs/dev/testing.md``): the grid rules the stored axes were
# reproduced with are stated here rather than taken from solprop's defaults, so that a
# change of a default cannot move what is compared. They are today's defaults.
FIELD_GRID_RULES = {
    "ffac": 6.0,
    "fft_size": "pow2",
    "collar_fraction": 0.05,
    "collar_placement": "inside",
}
ENVELOPE_GRID_RULES = {
    "thg": False,
    "ffac": 2.0,
    "fft_size": "pow2",
    "collar_fraction": 0.05,
    "collar_placement": "inside",
}


@pytest.fixture
def field_grid():
    capillary = PARAMETERS["capillary"]
    return FieldGrid.from_limits(
        capillary["length"],
        capillary["wavelength"],
        tuple(capillary["wavelength_limits"]),
        capillary["trange"],
        **FIELD_GRID_RULES,
    )


@pytest.fixture
def envelope_grid():
    capillary = PARAMETERS["capillary"]
    return EnvelopeGrid.from_limits(
        capillary["length"],
        capillary["wavelength"],
        tuple(capillary["wavelength_limits"]),
        capillary["trange"],
        **ENVELOPE_GRID_RULES,
    )


@pytest.fixture
def gnlse_grid():
    gnlse = PARAMETERS["gnlse"]
    return EnvelopeGrid.from_limits(
        gnlse["length"],
        gnlse["wavelength"],
        tuple(gnlse["wavelength_limits"]),
        gnlse["trange"],
        **ENVELOPE_GRID_RULES,
    )


@pytest.mark.parametrize(
    ("fixture", "key"),
    [
        ("field_grid", "REAL"),
        ("envelope_grid", "ENV"),
        ("gnlse_grid", "GN"),
    ],
)
def test_the_grids_reproduce_lunas_axes(fixture, key, request):
    # Not a claim about the operator, but the comparison below rests on it: the
    # operators are only comparable bin by bin if the axes are the same. The envelope
    # grids agree only because the windows were chosen so (see the generator); the
    # grid's symmetric span rule otherwise makes solprop's grid wider than Luna's.
    grid = request.getfixturevalue(fixture)
    prefix = "GN_OMEGA" if key == "GN" else f"OMEGA_{key}"
    band = "GN_SIDX" if key == "GN" else f"SIDX_{key}"
    omega = stored(prefix)
    assert grid.n_omega == omega.size
    # An envelope axis passes close to zero, so the comparison is absolute, at a few
    # ulp of the axis's own scale: the two build it by different arithmetic.
    assert np.max(np.abs(grid.omega - omega)) < 8.0 * np.finfo(float).eps * np.max(
        np.abs(omega)
    )
    assert np.array_equal(grid.active_mask, mask(band))


# ------------------------------------------------------ the capillary's operator


def tabulated_mode(grid, key):
    """A dispersion-specified mode carrying Luna's own beta and alpha for the mode."""
    active = grid.active_mask
    omega = stored(f"OMEGA_{key}")[active]
    order = np.argsort(omega)
    beta = interpolate.CubicSpline(omega[order], stored(f"BETA_{key}")[active][order])
    alpha = interpolate.CubicSpline(omega[order], stored(f"ALPHA_{key}")[active][order])
    return SpecifiedMode.from_table(
        omega[order],
        stored(f"BETA_{key}")[active][order],
        loss=alpha,
        aeff=stored("CAP_AEFF")[0],
    ), beta


def round_off_bound(beta):
    """Absolute bound on the round-off of the operator's cancellation, in 1/m."""
    return 8.0 * np.finfo(float).eps * float(np.max(np.abs(beta)))


@pytest.mark.parametrize(
    ("fixture", "key", "subtract_beta0"),
    [("field_grid", "REAL", False), ("envelope_grid", "ENV", True)],
)
def test_the_operator_matches_luna_for_a_constant_capillary(
    fixture, key, subtract_beta0, request
):
    grid = request.getfixturevalue(fixture)
    mode, _ = tabulated_mode(grid, key)
    frame = Frame(
        grid.omega0,
        subtract_beta0=subtract_beta0,
        beta0=stored(f"BETA0_{key}")[0],
        beta1=stored(f"BETA1_{key}")[0],
    )
    # Luna's own attenuation limit, so that the two clamp the same bins; solprop's
    # default limit is derived from the fibre length instead and is compared separately
    # below.
    operator = dispersion_operator(mode, grid, frame, loss=LossPolicy(max_alpha=3000.0))
    reference = complex_stored(f"LINOP_{key}")
    worst = float(np.max(np.abs(operator - reference)))
    assert 0.0 < worst < round_off_bound(stored(f"BETA_{key}")[grid.active_mask])


def test_the_two_loss_policies_agree_on_this_witness(field_grid):
    grid = field_grid
    mode, _ = tabulated_mode(grid, "REAL")
    frame = Frame(
        grid.omega0, subtract_beta0=False, beta0=0.0, beta1=stored("BETA1_REAL")[0]
    )
    luna_like = dispersion_operator(mode, grid, frame, loss=LossPolicy(3000.0))
    # Deliberately solprop's default policy: what is measured is that it and Luna's
    # fixed limit clamp nothing on this witness.
    default = dispersion_operator(mode, grid, frame)
    # This capillary loses at most 8 1/m in band, far below either limit -- Luna's
    # 3000 or solprop's 2 * 708 / 0.15 = 9440 for this fibre -- so the two policies
    # cannot differ here and the comparison above is not about the clamp. The clamp
    # itself is exercised in `tests/test_linear.py`.
    assert np.max(stored("ALPHA_REAL")[grid.active_mask]) < 3000.0
    # A clamp that bites nowhere passes every value through untouched: exact.
    assert np.array_equal(luna_like, default)


@pytest.mark.parametrize("key", ["REAL", "ENV"])
def test_the_analytic_group_delay_differs_from_lunas_differenced_one(key, request):
    grid = request.getfixturevalue("field_grid" if key == "REAL" else "envelope_grid")
    mode, _ = tabulated_mode(grid, key)
    _, beta1 = frame_quantities(mode, Frame(grid.omega0, subtract_beta0=False))
    # DIVERGENT: Luna recomputes beta_1 every stage from a seven-point central
    # difference of beta over +-1.8 % of omega_0 (Maths.jl:16-33), while solprop
    # differentiates the mode analytically (ADR-006, docs/dev/decisions.md). Here
    # solprop's analytic derivative is a cubic spline through Luna's own beta, so the
    # residual is the spline's error plus Luna's differencing noise; measured 5.7e-12 on
    # the field grid and 1.0e-12 on the envelope grid, either of which is well above the
    # 1e-16 the two would agree to if both were exact.
    residual = abs(beta1 / stored(f"BETA1_{key}")[0] - 1.0)
    assert 1e-14 < residual < 1e-10


# ---------------------------------------------------------- the GNLSE fibre's mode


@pytest.fixture
def simple_mode():
    gnlse = PARAMETERS["gnlse"]
    return SpecifiedMode.from_taylor(
        gnlse["betas"],
        2.0 * np.pi * C / gnlse["wavelength"],
        loss=attenuation_from_db(gnlse["loss_db_per_m"]),
        aeff=gnlse["aeff"],
    )


def test_the_dispersion_specified_mode_reproduces_lunas_simple_mode(
    simple_mode, gnlse_grid
):
    omega = gnlse_grid.omega
    # Both evaluate the same Taylor polynomial; Luna uses Polynomials.jl and solprop
    # numpy.polynomial, so only the order of the Horner sum differs.
    assert np.allclose(simple_mode.beta(omega), stored("GN_BETA"), rtol=1e-12, atol=0.0)
    assert np.allclose(
        simple_mode.alpha(omega), stored("GN_ALPHA"), rtol=1e-15, atol=0.0
    )
    assert simple_mode.alpha(omega)[0] == pytest.approx(
        stored("GN_ALPHA_SCALAR")[0], rel=1e-15
    )


def test_the_analytic_group_delay_matches_luna_for_a_taylor_mode(
    simple_mode, gnlse_grid
):
    # Luna takes beta_1 of a SimpleMode analytically too, from the differentiated
    # polynomial, so here the two agree to round-off rather than to 1e-6.
    _, beta1 = frame_quantities(
        simple_mode, Frame(gnlse_grid.omega0, subtract_beta0=False)
    )
    assert beta1 == pytest.approx(stored("GN_BETA1")[0], rel=1e-14)


def test_the_operator_matches_luna_for_the_gnlse_fibre(simple_mode, gnlse_grid):
    # Luna's own attenuation limit, as for the capillary; 0.2 dB/m clamps nothing.
    operator = dispersion_operator(
        simple_mode,
        gnlse_grid,
        Frame.comoving(gnlse_grid),
        loss=LossPolicy(max_alpha=3000.0),
    )
    reference = complex_stored("GN_LINOP")
    worst = float(np.max(np.abs(operator - reference)))
    assert 0.0 < worst < round_off_bound(simple_mode.beta(gnlse_grid.omega))


# ----------------------------------------------------------------- the prefactors


@pytest.mark.parametrize(
    ("fixture", "key"), [("field_grid", "REAL"), ("envelope_grid", "ENV")]
)
def test_the_mode_average_prefactor_matches_luna(fixture, key, request):
    grid = request.getfixturevalue(fixture)
    mode, _ = tabulated_mode(grid, key)
    active = grid.active_mask
    # Luna's self-steepening and the mode's own index, stated rather than defaulted.
    prefactor = mode_average_prefactor(mode, grid, shock=True, index=None)[active]
    # The prefactor goes as 1/sqrt(eps0). Luna leaves the out-of-band bins at one and
    # solprop sets them to zero, so the comparison is on the band both agree about.
    reference = rescaled(complex_stored(f"NORM_{key}")[active], -0.5)
    assert np.allclose(prefactor, reference, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize(("suffix", "shock"), [("", True), ("_NOSHOCK", False)])
def test_the_gnlse_prefactor_matches_luna(suffix, shock, simple_mode, gnlse_grid):
    active = gnlse_grid.active_mask
    prefactor = mode_average_prefactor(simple_mode, gnlse_grid, shock=shock, index=1.0)[
        active
    ]
    reference = rescaled(complex_stored(f"GN_NORM{suffix}")[active], -0.5)
    assert np.allclose(prefactor, reference, rtol=1e-13, atol=0.0)


# ------------------------------------------------------------ the gamma mapping


@pytest.mark.parametrize("index", [1, 2, 3])
def test_the_coefficient_mapping_matches_luna_when_raman_is_on(index):
    gamma, aeff, fraction, n2, chi3 = stored(f"MAP_{index}")
    omega_ref = 2.0 * np.pi * C / PARAMETERS["gnlse"]["wavelength"]
    problem = problem_from(
        PARAMETERS["gnlse"]["betas"], omega_ref=omega_ref, gamma=gamma, aeff=aeff
    )
    # Luna forms k0 as 2 pi / wavelength and solprop as omega_0 / c, which is the same
    # number reached by a different route.
    assert problem.n2 == pytest.approx(n2, rel=1e-15)
    # The split the Raman response applies is kerr_susceptibility's, compared here
    # against Luna's chi3.
    assert kerr_susceptibility(problem.n2, fraction) == pytest.approx(
        rescaled(chi3, 1.0), rel=1e-15
    )


@pytest.mark.parametrize("index", [1, 2, 3])
def test_the_whole_of_gamma_acts_as_kerr_when_raman_is_off(index):
    gamma, aeff, fraction, _, chi3 = stored(f"MAP_{index}")
    omega_ref = 2.0 * np.pi * C / PARAMETERS["gnlse"]["wavelength"]
    problem = problem_from(
        PARAMETERS["gnlse"]["betas"],
        omega_ref=omega_ref,
        gamma=gamma,
        aeff=aeff,
        raman=False,
    )
    # `raman_fraction` is left to solprop's rule deliberately: the rule is what this
    # measures against Luna.
    # DIVERGENT by design: Luna's prop_gnlse applies (1 - f_R) whatever `raman` is set
    # to, so it propagates that fraction less nonlinearity than asked for; solprop's
    # default fraction with Raman off is zero.
    assert problem.chi3 == pytest.approx(
        rescaled(chi3, 1.0) / (1.0 - fraction), rel=1e-15
    )


@pytest.mark.parametrize("index", [1, 2, 3])
def test_a_given_fraction_with_raman_off_matches_luna(index):
    gamma, aeff, fraction, _, chi3 = stored(f"MAP_{index}")
    omega_ref = 2.0 * np.pi * C / PARAMETERS["gnlse"]["wavelength"]
    problem = problem_from(
        PARAMETERS["gnlse"]["betas"],
        omega_ref=omega_ref,
        gamma=gamma,
        aeff=aeff,
        raman=False,
        raman_fraction=fraction,
    )
    # A fraction given explicitly is held back with Raman off too: Luna's split.
    assert problem.chi3 == pytest.approx(rescaled(chi3, 1.0), rel=1e-15)
    assert problem.raman is None


# --------------------------------------------------------------- the Kerr responses


def signal(name):
    return np.asarray(SIGNALS[name], dtype=float)


def envelope(component):
    return signal(f"envelope_{component}_re") + 1j * signal(f"envelope_{component}_im")


# The responses are products of three field samples with one prefactor, formed in a
# different order on each side; Higham's n*eps for n around eight is 2e-15. The
# no-third-harmonic field form adds a forward and an inverse transform of length 24,
# whose round-off is a further few times log2(24)*eps.
KERR_RTOL = 1e-14


@pytest.fixture
def kerr():
    return KerrResponse(chi3=PARAMETERS["kerr"]["gamma3"], thg=True)


def density():
    return PARAMETERS["kerr"]["density"]


def test_the_field_kerr_response_matches_luna(kerr):
    out = kerr.evaluate(signal("field_x")[None, :], density(), kind="field")
    assert np.allclose(
        out[0],
        rescaled(stored("KERR_FIELD"), 1.0),
        rtol=KERR_RTOL,
        atol=0.0,
    )


def test_the_field_kerr_response_without_the_harmonic_matches_luna():
    # The stored signal has no Nyquist component, so Luna's analytic signal (which
    # drops that bin) and solprop's (which keeps it, as scipy does) coincide and this
    # compares the response rather than that documented difference.
    response = KerrResponse(chi3=PARAMETERS["kerr"]["gamma3"], thg=False)
    out = response.evaluate(signal("field_x")[None, :], density(), kind="field")
    assert np.allclose(
        out[0],
        rescaled(stored("KERR_FIELD_NOTHG"), 1.0),
        rtol=KERR_RTOL,
        atol=0.0,
    )


def test_the_vector_field_kerr_response_matches_luna(kerr):
    columns = np.stack([signal("field_x"), signal("field_y")])
    out = kerr.evaluate(columns, density(), kind="field")
    assert np.allclose(
        out[0],
        rescaled(stored("KERR_FIELD_VEC_X"), 1.0),
        rtol=KERR_RTOL,
        atol=0.0,
    )
    assert np.allclose(
        out[1],
        rescaled(stored("KERR_FIELD_VEC_Y"), 1.0),
        rtol=KERR_RTOL,
        atol=0.0,
    )


def test_the_envelope_kerr_response_matches_luna():
    response = KerrResponse(chi3=PARAMETERS["kerr"]["gamma3"], thg=False)
    out = response.evaluate(envelope("x")[None, :], density(), kind="envelope")
    assert np.allclose(
        out[0],
        rescaled(complex_stored("KERR_ENV"), 1.0),
        rtol=KERR_RTOL,
        atol=0.0,
    )


def test_the_vector_envelope_kerr_response_matches_luna():
    response = KerrResponse(chi3=PARAMETERS["kerr"]["gamma3"], thg=False)
    columns = np.stack([envelope("x"), envelope("y")])
    out = response.evaluate(columns, density(), kind="envelope")
    assert np.allclose(
        out[0],
        rescaled(complex_stored("KERR_ENV_VEC_X"), 1.0),
        rtol=KERR_RTOL,
        atol=0.0,
    )
    assert np.allclose(
        out[1],
        rescaled(complex_stored("KERR_ENV_VEC_Y"), 1.0),
        rtol=KERR_RTOL,
        atol=0.0,
    )


def test_the_envelope_kerr_response_with_the_harmonic_matches_luna(kerr):
    carrier = thg_carrier(PARAMETERS["kerr"]["omega0"], signal("t"))
    out = kerr.evaluate(
        envelope("x")[None, :], density(), kind="envelope", carrier=carrier
    )
    # The exp(+2i omega0 t) sign was re-derived in solprop's convention rather than
    # copied; agreeing with Luna here confirms the two conventions coincide for this
    # term.
    assert np.allclose(
        out[0],
        rescaled(complex_stored("KERR_ENV_THG"), 1.0),
        rtol=KERR_RTOL,
        atol=0.0,
    )


def test_the_susceptibility_conversion_matches_lunas_glass_formula():
    # PhysData.chi3 for a glass is 4/3 n2 eps0 c n0^2; solprop's chi3_from_n2 is its
    # inverse. The GNLSE mapping uses it with n0 = 1 (ADR-026).
    _, _, fraction, n2, chi3 = stored("MAP_2")
    assert 4.0 / 3.0 * n2 * EPS0 * C == pytest.approx(
        rescaled(chi3, 1.0) / (1.0 - fraction), rel=1e-15
    )
