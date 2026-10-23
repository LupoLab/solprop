# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's dispersive electronic response against Phase 0's independent triplet sum.

:meth:`solprop.nonlinear.responses.DispersiveKerrResponse.evaluate` is the reference
every backend is held to; here it is held to
``tests/dispersive_kerr_oracle.direct_sum``, the explicit sum over signed input
triplets with the kernel evaluated leg by leg, on dimensioned grids: the oracle's
integer bin ``k`` is the angular frequency ``k * DOMEGA``, so that bin 12 is 200 nm and
argon's leg bound (215.9 nm) falls inside the field's band and the bound is exercised.

Tolerances. Every comparison is between two evaluations of the same finite sum in
double precision: at most ``24**3`` triplet products, or a few FFTs of 128 points.
Phase 0's FFT forms met the triplet sum to 1.3e-14; this implementation measured
2.4e-15. ``RTOL = 1e-12`` is Phase 0's accepted acceptance level (decision Q1).
"""

import math

import numpy as np
import pytest

from dispersive_kerr_oracle import analytic, direct_sum
from solprop.constants import EPS0, C
from solprop.exceptions import GridError, MaterialError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.nonlinear.responses import (
    DispersiveKerrResponse,
    KerrResponse,
    thg_carrier,
)
from solprop.physdata.nonlinear_data import DispersiveKerrModel, OptmatKerr

RTOL = 1e-12
N = 128
BAND = 12
T = 2 * np.pi * np.arange(N) / N
DOMEGA = 2 * math.pi * C / 200e-9 / BAND
ARGON = OptmatKerr("Ar").dispersive_model()
FLAT = DispersiveKerrModel(
    "Ar", ARGON.gamma3_static, 0.0, 0.0, 0.0, ARGON.leg_bound, None, None,
    ARGON.material_records,
)  # fmt: skip


def multitone(seed: int, n_pol: int = 1, band: int = BAND) -> np.ndarray:
    """Return a real field with random amplitudes on bins 1..band, no DC."""
    rng = np.random.default_rng(seed)
    spectrum = np.zeros((n_pol, N), complex)
    amplitude = rng.normal(size=(n_pol, band)) + 1j * rng.normal(size=(n_pol, band))
    spectrum[:, 1 : band + 1] = amplitude
    spectrum[:, -band:] = amplitude[:, ::-1].conj()
    return np.real(np.fft.ifft(spectrum, axis=-1)) * N / band


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def kernel_of(response: DispersiveKerrResponse):
    """Return the oracle's kernel of integer quartets: the model leg by leg."""
    model = response.model

    def kernel(legs: np.ndarray) -> np.ndarray:
        nu = legs * DOMEGA / (2 * math.pi * C)
        if response.evaluator == "polynomial":
            return model.gamma3(nu) / model.gamma3_static
        weight = np.minimum(nu**2, model.leg_bound**2)
        return np.prod(model.kerr_ratio(weight) ** 0.25, axis=-1)

    return kernel


def field_frequencies() -> np.ndarray:
    return DOMEGA * np.arange(N // 2 + 1)


def envelope_frequencies(carrier: int) -> np.ndarray:
    return DOMEGA * (np.fft.fftfreq(N, 1.0 / N) + carrier)


CASES = [
    (evaluator, thg, n_pol)
    for evaluator in ("polynomial", "symmetric")
    for thg in (True, False)
    for n_pol in (1, 2)
]
FIELD_CASES = [
    *CASES,
    *((e, thg, 3) for e in ("polynomial", "symmetric") for thg in (True, False)),
]


@pytest.mark.parametrize(("evaluator", "thg", "n_pol"), FIELD_CASES)
def test_field_grid_matches_the_triplet_sum(evaluator, thg, n_pol):
    """Both evaluators, both sectors, scalar and isotropic vector, the bound active.

    Three components are the general modal route's full vector field.
    """
    response = DispersiveKerrResponse(ARGON, evaluator, thg=thg)
    field = multitone(n_pol, n_pol)
    sector = "full" if thg else "conserving"
    expected = direct_sum(
        field if n_pol > 1 else field[0], kernel_of(response), sector=sector
    )
    got = response.evaluate(field, 2.5e25, kind="field", omega=field_frequencies())
    got = got / (2.5e25 * EPS0 * ARGON.gamma3_static)
    assert relative(got if n_pol > 1 else got[0], expected) < RTOL


@pytest.mark.parametrize(("evaluator", "thg", "n_pol"), CASES)
def test_envelope_frame_is_independent_of_the_carrier(evaluator, thg, n_pol):
    """Multipliers at absolute frequency: the envelope equals the field frame (s. 5)."""
    response = DispersiveKerrResponse(ARGON, evaluator, thg=thg)
    field = multitone(10 + n_pol, n_pol)
    sector = "full" if thg else "conserving"
    reference = analytic(
        direct_sum(field if n_pol > 1 else field[0], kernel_of(response), sector=sector)
    )
    for carrier in (5, 8):
        envelope = analytic(field) * np.exp(-1j * carrier * T)
        got = response.evaluate(
            envelope,
            1.0,
            kind="envelope",
            omega=envelope_frequencies(carrier),
            carrier=np.exp(2j * carrier * T) if thg else None,
        )
        got = got * np.exp(1j * carrier * T) / (EPS0 * ARGON.gamma3_static)
        assert relative(got if n_pol > 1 else got[0], reference) < RTOL


@pytest.mark.parametrize("evaluator", ["polynomial", "symmetric"])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_without_dispersion_the_full_cubic_is_the_kerr_response(evaluator, n_pol):
    field = multitone(20 + n_pol, n_pol)
    flat = DispersiveKerrResponse(FLAT, evaluator)
    constant = KerrResponse(ARGON.gamma3_static)
    got = flat.evaluate(field, 3.0, kind="field", omega=field_frequencies())
    expected = constant.evaluate(field, 3.0, kind="field")
    # Two FFT round trips of 128 points against a pointwise product.
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-14 * np.abs(expected).max()
    )


@pytest.mark.parametrize("evaluator", ["polynomial", "symmetric"])
def test_without_dispersion_the_conserving_sector_is_thg_false_below_an_octave(
    evaluator,
):
    """Below an octave ``|u|^2 u`` has no negative-frequency content (decision Q4)."""
    narrow = np.cos(9 * T) + 0.7 * np.sin(11 * T + 0.3) + 0.4 * np.cos(12 * T)
    flat = DispersiveKerrResponse(FLAT, evaluator, thg=False)
    constant = KerrResponse(ARGON.gamma3_static, thg=False)
    got = flat.evaluate(narrow[None, :], 1.0, kind="field", omega=field_frequencies())
    expected = constant.evaluate(narrow[None, :], 1.0, kind="field")
    assert relative(got, expected) < RTOL
    envelope = analytic(narrow)[None, :] * np.exp(-1j * 10 * T)
    for thg in (False, True):
        carrier = np.exp(2j * 10 * T) if thg else None
        got = DispersiveKerrResponse(FLAT, evaluator, thg=thg).evaluate(
            envelope, 1.0, kind="envelope", omega=envelope_frequencies(10),
            carrier=carrier,
        )  # fmt: skip
        expected = KerrResponse(ARGON.gamma3_static, thg=thg).evaluate(
            envelope, 1.0, kind="envelope", carrier=carrier
        )
        assert relative(got, expected) < RTOL
    broad = multitone(30)
    got = flat.evaluate(broad, 1.0, kind="field", omega=field_frequencies())
    expected = constant.evaluate(broad, 1.0, kind="field")
    assert relative(got, expected) > 1e-3


@pytest.mark.parametrize("evaluator", ["polynomial", "symmetric"])
@pytest.mark.parametrize("thg", [True, False])
def test_the_response_does_no_net_work(evaluator, thg):
    """Section 4.9's lossless identity, with the bound and in each sector."""
    field = multitone(40)[0]
    response = DispersiveKerrResponse(ARGON, evaluator, thg=thg)
    polarisation = response.evaluate(
        field[None, :], 1.0, kind="field", omega=field_frequencies()
    )[0]
    k = np.fft.fftfreq(N, 1.0 / N)
    rate = np.real(np.fft.ifft(1j * k * np.fft.fft(field)))
    work = abs(np.sum(rate * polarisation)) / (
        np.linalg.norm(rate) * np.linalg.norm(polarisation)
    )
    assert work < RTOL


def test_the_traits_and_sector_follow_the_declaration():
    response = DispersiveKerrResponse(ARGON, "symmetric", thg=False, species="argon")
    assert response.sector == "conserving"
    assert response.traits.grids == ("field", "envelope")
    assert response.traits.polarisation == "vector_isotropic"
    assert response.traits.z_profiles == ("density:argon",)
    assert response.coefficient == EPS0 * ARGON.gamma3_static
    with pytest.raises(MaterialError, match="evaluator"):
        DispersiveKerrResponse(ARGON, "output-only")  # pyright: ignore[reportArgumentType]


def test_weights_are_read_only_and_exclude_dc_and_nyquist_on_a_field_grid():
    weights = DispersiveKerrResponse(ARGON, "polynomial").weights(
        field_frequencies(), kind="field"
    )
    assert not weights.positive[0] and not weights.positive[-1]
    assert np.all(weights.positive[1:-1])
    assert weights.output is not None and weights.output.shape == (2, N // 2 + 1)
    assert weights.products == (
        3 * (ARGON.b + ARGON.b_prime),
        6 * ARGON.b,
    )
    for array in (weights.positive, weights.filter, weights.output):
        assert not array.flags.writeable
    symmetric = DispersiveKerrResponse(ARGON, "symmetric").weights(
        envelope_frequencies(5), kind="envelope"
    )
    assert symmetric.output is None
    assert symmetric.filter[0] == pytest.approx(
        float(ARGON.kerr_ratio(ARGON.leg_weight(np.array([5 * DOMEGA])))[0]) ** 0.25,
        rel=1e-15,
    )


def test_evaluate_checks_its_arguments():
    response = DispersiveKerrResponse(ARGON, "polynomial")
    field = multitone(50)
    with pytest.raises(GridError, match="omega has shape"):
        response.evaluate(field, 1.0, kind="field", omega=np.arange(N, dtype=float))
    with pytest.raises(GridError, match="density must be non-negative"):
        response.evaluate(field, -1.0, kind="field", omega=field_frequencies())
    envelope = analytic(field) * np.exp(-1j * 5 * T)
    with pytest.raises(GridError, match="needs the carrier"):
        response.evaluate(envelope, 1.0, kind="envelope", omega=envelope_frequencies(5))
    carrier = thg_carrier(1.0, T)
    with pytest.raises(GridError, match="only used on an envelope grid"):
        DispersiveKerrResponse(ARGON, "polynomial", thg=False).evaluate(
            envelope,
            1.0,
            kind="envelope",
            omega=envelope_frequencies(5),
            carrier=carrier,
        )
    with pytest.raises(GridError, match="window has shape"):
        response.evaluate(
            field, 1.0, kind="field", omega=field_frequencies(), window=np.ones(N - 1)
        )


def collar_case(kind: str, thg: bool, centre: float):
    """Return a grid's collar and the two orders of it on an 8 fs pulse at `centre`.

    The pulse's position is a fraction of the fine window's half width. The orders are
    the decided ``G[w (G E)^3]`` (``evaluate(window=w)``) and the former
    ``w G[(G E)^3]``, for both evaluators.
    """
    if kind == "field":
        grid = FieldGrid.from_limits(1.0, 800e-9, (150e-9, 4e-6), 150e-15)
    else:
        grid = EnvelopeGrid.from_limits(1.0, 800e-9, (150e-9, 4e-6), 150e-15, thg=True)
    t, window = grid.t_fine, grid.t_window_fine
    envelope = np.exp(-(((t - centre * t.max()) / 8e-15) ** 2))
    field = envelope * np.cos(grid.omega0 * t) if kind == "field" else envelope + 0j
    carrier = thg_carrier(grid.omega0, t) if kind == "envelope" and thg else None
    pairs = []
    for evaluator in ("symmetric", "polynomial"):
        response = DispersiveKerrResponse(ARGON, evaluator, thg=thg)
        arguments = {"kind": kind, "omega": grid.omega_fine, "carrier": carrier}
        bare = response.evaluate(field[None], 1.0, **arguments)
        unit = response.evaluate(field[None], 1.0, window=np.ones(t.size), **arguments)
        # Structural: multiplying by a unit collar is an exact no-op.
        assert np.array_equal(unit, bare)
        decided = response.evaluate(field[None], 1.0, window=window, **arguments)
        pairs.append((decided, window * bare))
    return window, pairs


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("thg", [True, False])
def test_the_collar_order_matters_only_where_the_collar_varies(kind, thg):
    """``G[w (G E)^3]`` against ``w G[(G E)^3]`` (John Travers, 2026-10-08).

    A unit collar gives the operator without one, exactly: multiplying by one is an
    exact no-op (:func:`collar_case`).
    A pulse in the collar's flat middle gives the two orders to the rounding of the
    transforms: the collar multiplies nothing but samples the field leaves at zero
    (measured 5e-15 to 3.3e-12 of the largest sample). A pulse reaching the collar's
    edge, at nine tenths of the window, separates them by the commutator of the
    collar with the output multipliers and with ``Pi_+``, measured 3.2e-3 to 5.9e-3.
    In the full sector it sits where the collar varies: the part inside the flat
    middle is 0.4 % to 1.1 % of it, the reach of the multipliers' kernels. The
    conserving sector's ``Pi_+`` is a Hilbert transform, whose kernel decays only as
    ``1/t``, so on a field grid a third of it reaches the middle.
    """
    window, pairs = collar_case(kind, thg, 0.0)
    for decided, former in pairs:
        scale = float(np.abs(former).max())
        assert scale > 0
        assert float(np.abs(decided - former).max()) < 1e-10 * scale
    window, pairs = collar_case(kind, thg, 0.9)
    flat = window == 1.0
    assert flat.any() and not flat.all()
    for decided, former in pairs:
        difference = np.abs(decided - former)[0]
        scale = float(np.abs(former).max())
        assert float(difference.max()) > 1e-4 * scale
        if thg or kind == "envelope":
            assert difference[flat].max() < 0.05 * difference[~flat].max()


@pytest.mark.parametrize("wavelength", [800e-9, 400e-9])
def test_an_envelope_grid_built_for_harmonics_holds_the_one_into_three_group(
    wavelength,
):
    """Phase 0's open question 4, measured on a super-octave field.

    A field of three bands, 250 nm, 800 nm and 2.5 um, has one-into-three quartets
    with a positive output. On ``EnvelopeGrid.from_limits(..., thg=True)`` the full
    cubic's three groups are evaluated on the grid's own fine axis and on a four times
    finer one; the two agree across the active band, so the grid holds the group
    rather than aliasing it. Measured 1.2e-14 and 7.5e-14; the bound is 1e-12, a few
    FFTs of 4096 to 16384 points.
    """
    grid = EnvelopeGrid.from_limits(1.0, wavelength, (150e-9, 4e-6), 300e-15, thg=True)
    n, factor = grid.n_to, 4
    half = n // 2
    omega = grid.omega_fine
    spectrum = sum(
        np.exp(-(((omega - 2 * math.pi * C / centre) / width) ** 2))
        for centre, width in ((800e-9, 4e14), (250e-9, 4e14), (2500e-9, 1e14))
    )
    active = (omega > 0.5 * grid.omega_min) & (omega < grid.omega_max_window)
    spectrum = spectrum * active
    # The pulse at the window's centre: the stored phase's origin is the first sample.
    envelope = np.roll(np.fft.ifft(spectrum), half)
    finer = np.zeros(factor * n, complex)
    finer[:half], finer[-half:] = spectrum[:half], spectrum[half:]
    fine_envelope = np.roll(np.fft.ifft(finer) * factor, factor * half)
    fine_t = grid.t_fine[0] + np.arange(factor * n) * grid.dt_fine / factor
    fine_omega = grid.omega0 + 2 * np.pi * np.fft.fftfreq(
        factor * n, fine_t[1] - fine_t[0]
    )
    response = DispersiveKerrResponse(ARGON, "polynomial")
    got = response.evaluate(
        envelope[None], 1.0, kind="envelope", omega=omega,
        carrier=thg_carrier(grid.omega0, grid.t_fine),
    )[0]  # fmt: skip
    reference = response.evaluate(
        fine_envelope[None], 1.0, kind="envelope", omega=fine_omega,
        carrier=thg_carrier(grid.omega0, fine_t),
    )[0]  # fmt: skip
    folded = np.fft.fft(reference) / factor
    expected = np.concatenate([folded[:half], folded[-half:]])
    assert relative(np.fft.fft(got)[active], expected[active]) < RTOL
