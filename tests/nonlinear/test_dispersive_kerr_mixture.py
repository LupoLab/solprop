# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's symmetric evaluator of a gas mixture: one filter from the mixture's curve.

The symmetric evaluator of a mixture takes one filter built from the mixture's own Kerr
diagonal, ``g_mix = [sum_s f_s g_s^4 / sum_s f_s]^(1/4)`` with ``f_s = eps0 gamma3_0,s
rho_s`` (SEP 6 section 6, adopted on 2026-10-08), and ``P = (sum_s f_s) G_mix[w (G_mix
E)^3]``. Here:

- its FFT form against Phase 0's triplet sum of its own kernel, leg by leg;
- its normalisation: the mixture's Kerr diagonal, a unit filter at zero frequency, the
  reference that cancels, a filter between the species', the vacuum;
- one species reduced exactly to its own operator, and a gas split into two
  populations to its own filter;
- the operator against :func:`exact_symmetric_sum`, the exact per-species sum it
  approximates, on a helium/argon and an air mixture, within the mixing error Phase 2b
  measured by section 4.7's procedure.

Tolerances. The triplet-sum comparison is Phase 0's ``RTOL = 1e-12`` (decision Q1).
The identities are a few roundings of one bin (stated where used).
"""

import dataclasses
import math
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from dispersive_kerr_oracle import direct_sum
from solprop.backends.numpy_ref import passes
from solprop.constants import EPS0, C
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import FieldGrid
from solprop.nonlinear.responses import (
    DispersiveKerrResponse,
    DispersiveWeights,
    dispersive_cubic,
    mixture_filter,
)
from solprop.physdata import MaterialSelection
from solprop.physdata.nonlinear_data import DispersiveKerrModel, OptmatKerr
from solprop.profiles import Constant
from solprop.translate.linop import profile_table
from solprop.translate.responses import DispersiveKerrTerms, dispersive_terms
from solprop.translate.stage import scalars

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

EPS = np.finfo(float).eps
RTOL = 1e-12
N = 128
BAND = 12
# Bin 12 is 200 nm: argon's leg bound (215.9 nm) is active, helium's (151.9 nm) not.
DOMEGA = 2 * math.pi * C / 200e-9 / BAND
HELIUM = OptmatKerr("He").dispersive_model()
ARGON = OptmatKerr("Ar").dispersive_model()

# Phase 2b's mixing error, the mixture curve against the exact per-species symmetric
# sum by section 4.7's procedure (SEP 6's mixture-curve study): the largest
# relative departure of a quartet's coefficient over 208 109 quartets with every leg
# and the output on 400-2000 nm.
MIXING_ERROR = {"He/Ar": 0.0157e-2, "air": 0.0446e-2}
MIXTURES: dict[str, dict[str, float]] = {
    "He/Ar": {"He": 0.5, "Ar": 0.5},
    "air": {"N2": 0.7808, "O2": 0.2095, "Ar": 0.0097},
}


def field_frequencies() -> np.ndarray:
    return DOMEGA * np.arange(N // 2 + 1)


def multitone(seed: int, n_pol: int = 1) -> np.ndarray:
    """Return a real field with random amplitudes on bins 1..BAND, no DC."""
    rng = np.random.default_rng(seed)
    spectrum = np.zeros((n_pol, N), complex)
    amplitude = rng.normal(size=(n_pol, BAND)) + 1j * rng.normal(size=(n_pol, BAND))
    spectrum[:, 1 : BAND + 1] = amplitude
    spectrum[:, -BAND:] = amplitude[:, ::-1].conj()
    return np.real(np.fft.ifft(spectrum, axis=-1)) * N / BAND


def own_filter(model: DispersiveKerrModel, omega: np.ndarray) -> np.ndarray:
    """Return a species' own symmetric filter at the absolute frequencies."""
    return (
        DispersiveKerrResponse(model, "symmetric").weights(omega, kind="field").filter
    )


def exact_symmetric_sum(
    field_t: np.ndarray,
    leaf: DispersiveKerrTerms,
    densities: Sequence[float],
    *,
    kind: str,
    carrier: np.ndarray | None = None,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """Return the exact per-species symmetric sum the mixture filter approximates.

    ``P = sum_s eps0 gamma3_0,s rho_s G_s[w (G_s E)^3]``, each species under its own
    filter and leg bound: Phase 1's operator of a mixture, each term the reference
    :func:`~solprop.nonlinear.responses.dispersive_cubic` held to the triplet sum.
    """
    total = np.zeros_like(field_t)
    for channel, (coefficient, density) in enumerate(
        zip(leaf.coefficients, densities, strict=True)
    ):
        total = total + (coefficient * density) * dispersive_cubic(
            field_t,
            leaf.weights(channel),
            kind=kind,
            sector=leaf.sector,
            carrier=carrier,
            window=window,
        )
    return total


# --- the form and its normalisation --------------------------------------------------


@pytest.mark.parametrize("thg", [True, False])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_the_mixture_filter_meets_the_triplet_sum_of_its_kernel(thg, n_pol):
    """``F G_mix[(G_mix E)^3]`` is the cubic of the kernel ``F prod_j g_mix(nu_j)``.

    The oracle evaluates ``g_mix`` leg by leg from the two models, each species' leg
    bounded inside its own term, so this holds the FFT form and its normalisation to
    the stated relation.
    """
    densities = np.array([3e25, 1e25])
    models = (HELIUM, ARGON)
    factors = EPS0 * np.array([model.gamma3_static for model in models]) * densities
    omega = field_frequencies()
    total, curve = mixture_filter(
        np.stack([own_filter(model, omega) for model in models]), factors
    )
    positive = DispersiveKerrResponse(ARGON, "symmetric").weights(omega, kind="field")
    sector = "full" if thg else "conserving"

    def kernel(legs: np.ndarray) -> np.ndarray:
        nu = legs * DOMEGA / (2 * math.pi * C)
        diagonal = sum(
            factor * model.kerr_ratio(np.minimum(nu**2, model.leg_bound**2))
            for factor, model in zip(factors, models, strict=True)
        )
        return total * np.prod((diagonal / total) ** 0.25, axis=-1)

    field = multitone(3, n_pol)
    expected = direct_sum(field if n_pol > 1 else field[0], kernel, sector=sector)
    got = total * dispersive_cubic(
        field,
        DispersiveWeights("symmetric", positive.positive, curve),
        kind="field",
        sector=sector,
    )
    got = got if n_pol > 1 else got[0]
    assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < RTOL


def test_the_diagonal_is_the_mixture_kerr_curve():
    """``F g_mix^4 = sum_s f_s K_s``, the mixture's Kerr diagonal; ``g_mix(0) = 1``.

    Four roundings of the quartic, its sum and two square roots: a few ulp.
    """
    omega = DOMEGA * np.arange(0, 40)
    rows = np.stack([own_filter(model, omega) for model in (HELIUM, ARGON)])
    factors = np.array([0.3, 0.7])
    total, curve = mixture_filter(rows, factors)
    assert total == 1.0
    np.testing.assert_allclose(
        total * curve**4, factors @ rows**4, rtol=8 * EPS, atol=0.0
    )
    assert curve[0] == 1.0
    # Between the species' own filters, which rise with frequency.
    assert np.all(curve >= rows.min(axis=0) * (1 - 4 * EPS))
    assert np.all(curve <= rows.max(axis=0) * (1 + 4 * EPS))
    assert np.all(np.diff(curve) >= 0.0)


def test_the_reference_cancels():
    """Any positive reference ``gamma_*`` gives the same polarisation (section 4.4).

    With ``F' = c F`` and ``g' = g (F / F')^(1/4)`` the kernel ``F' prod g'`` is
    ``F prod g``; the cubic scales as the fourth power of the filter, so the two agree
    to a few roundings of every bin's product.
    """
    omega = field_frequencies()
    rows = np.stack([own_filter(model, omega) for model in (HELIUM, ARGON)])
    total, curve = mixture_filter(rows, np.array([2.0, 1.0]))
    positive = DispersiveKerrResponse(ARGON, "symmetric").weights(omega, kind="field")
    field = multitone(5)
    reference = total * dispersive_cubic(
        field,
        DispersiveWeights("symmetric", positive.positive, curve),
        kind="field",
        sector="full",
    )
    other = 7.5 * total
    shifted = other * dispersive_cubic(
        field,
        DispersiveWeights(
            "symmetric", positive.positive, curve * (total / other) ** 0.25
        ),
        kind="field",
        sector="full",
    )
    scale = float(np.abs(reference).max())
    np.testing.assert_allclose(shifted, reference, rtol=0, atol=64 * EPS * scale)


def test_one_species_is_its_own_operator_and_a_vacuum_is_zero():
    rows = own_filter(ARGON, field_frequencies())[None, :]
    total, curve = mixture_filter(rows, np.array([4.0]))
    assert total == 4.0
    # A fourth power, a binary scaling and a fourth root: a couple of ulp.
    np.testing.assert_allclose(curve, rows[0], rtol=4 * EPS, atol=0.0)
    two = np.vstack([rows, rows])
    total, curve = mixture_filter(two, np.zeros(2))
    # Structural: a vacuum has no total and the unit filter, exactly.
    assert total == 0.0
    np.testing.assert_array_equal(curve, 1.0)


def test_a_gas_split_into_two_populations_keeps_its_own_filter():
    """Helium as two populations: the composition cannot move the filter.

    ``(f1 g^4 + f2 g^4) / (f1 + f2)`` rooted: three roundings, a couple of ulp.
    """
    row = own_filter(HELIUM, field_frequencies())
    total, curve = mixture_filter(np.vstack([row, row]), np.array([0.25, 0.5]))
    assert total == 0.75
    np.testing.assert_allclose(curve, row, rtol=4 * EPS, atol=0.0)


def test_the_leaf_evaluates_one_species_as_before_and_a_mixture_once():
    """``evaluations``: one species is its own channel, its own row; a mixture one term.

    A region of its own is a group of its own.
    """
    grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 200e-15)
    table = profile_table(Constant(1e25), 0.0, 1.0)
    responses = [
        DispersiveKerrResponse(HELIUM, "symmetric", species="He"),
        DispersiveKerrResponse(ARGON, "symmetric", species="Ar"),
    ]
    leaf = dispersive_terms(responses, ["He", "Ar"], [table, table], grid)
    densities = (1e25, 2e25)
    ((channel, factor, weights),) = leaf.evaluations(densities)
    assert channel == 0
    expected_total, expected = mixture_filter(
        leaf.filters,
        np.array(
            [c * rho for c, rho in zip(leaf.coefficients, densities, strict=True)]
        ),
    )
    # The same mixture filter by two calls: within its few roundings.
    assert factor == pytest.approx(expected_total, rel=4 * EPS)
    np.testing.assert_allclose(weights.filter, expected, rtol=4 * EPS, atol=0.0)
    single = dispersive_terms(responses[1:], ["Ar"], [table], grid)
    ((_, factor, weights),) = single.evaluations([2e25])
    assert factor == single.coefficients[0] * 2e25
    # Structural: a single species carries its own stored row.
    np.testing.assert_array_equal(weights.filter, single.filters[0])
    regional = dataclasses.replace(leaf, regions=np.array([3, 4], dtype=np.int32))
    assert regional.groups() == ((0,), (1,))
    assert len(regional.evaluations(densities)) == 2
    polynomial = dispersive_terms(
        [dataclasses.replace(one, evaluator="polynomial") for one in responses],
        ["He", "Ar"],
        [table, table],
        grid,
    )
    assert [term[0] for term in polynomial.evaluations(densities)] == [0, 1]


# --- against the exact per-species sum ------------------------------------------------


def capillary(composition: Mapping[str, float]) -> Simulation:
    """Return a short symmetric capillary run of a mixture, translated on numpy_ref."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return Simulation.from_capillary(
            75e-6,
            2e-3,
            dict(composition),
            1.0,
            wavelength=800e-9,
            wavelength_limits=(150e-9, 4e-6),
            trange=150e-15,
            fwhm=10e-15,
            energy=30e-6,
            materials=MaterialSelection(electronic="symmetric-kerr"),
            plasma=False,
            raman=False,
            backend="numpy_ref",
            n_save=2,
            noise=None,
        )


@pytest.mark.medium
@pytest.mark.parametrize("name", sorted(MIXTURES))
def test_the_operator_is_within_the_measured_mixing_error_of_the_exact_sum(name):
    """The mixture filter against the exact per-species sum, bin by bin.

    A three-colour field (500, 800 and 1600 nm) confined to 400-2000 nm, the
    number-conserving sector. At an output bin on 400-2000 nm every quartet that
    contributes has all four legs on 400-2000 nm, inside section 4.7's survey domain
    (and within every species' fitted sums). Each contribution is a product of field
    amplitudes times a positive coefficient, so the departure at that bin is at most
    ``delta``, the largest relative departure of a coefficient, times the exact sum of
    the field's spectral magnitudes. The field's spectrum is real and positive (flat
    phase), so every contribution is positive and that sum is the exact sum itself:
    ``|P_mix - P_exact| <= delta |P_exact|``, the tightest case of the bound. ``delta``
    is Phase 2b's measured mixing error (a sampled maximum, the figure the curve was
    adopted on); the measured margin is 1.7 (He/Ar) and 1.6 (air). FFT rounding is
    allowed for at 64 eps of the largest bin.
    """
    simulation = capillary(MIXTURES[name])
    problem = simulation.translate()
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.evaluator == "symmetric"
    assert leaf.n_channels == len(MIXTURES[name]) and len(leaf.groups()) == 1
    leaf = dataclasses.replace(leaf, sector="conserving")
    densities = scalars(problem, 0.0, 0.0).dispersive_kerr_densities
    omega = simulation.grid.omega_fine
    n_to = simulation.grid.n_to
    wavelength = 2 * math.pi * C / np.where(omega > 0, omega, np.inf)
    inside = (wavelength >= 400e-9) & (wavelength <= 2000e-9)
    spectrum = sum(
        np.exp(-(((omega - 2 * math.pi * C / centre) / width) ** 2))
        for centre, width in ((500e-9, 0.08e15), (800e-9, 0.1e15), (1600e-9, 0.05e15))
    ) * np.where(inside, 1.0, 0.0)
    series = np.fft.irfft(spectrum, n=n_to)
    field = (5e9 / np.abs(series).max() * series)[None, None, :]

    got = passes.dispersive_kerr_polarisation(field, leaf, densities, kind="field")
    exact = exact_symmetric_sum(field, leaf, densities, kind="field")
    departure = np.abs(np.fft.rfft(got[0, 0] - exact[0, 0]))
    bound = np.abs(np.fft.rfft(exact[0, 0]))
    allowed = MIXING_ERROR[name] * bound + 64 * EPS * bound.max()
    assert np.all(departure[inside] <= allowed[inside])
    # An approximation, not the sum: the departure is well above rounding.
    assert departure[inside].max() > 1e3 * EPS * bound.max()
