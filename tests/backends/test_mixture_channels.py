# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent depletion and shared-current accounting for labelled populations."""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import trapezoid

from mixture_cases import mixture_case
from solprop.backends.numpy_ref import passes
from solprop.constants import EPS0
from solprop.exceptions import BackendError, TranslationError
from solprop.profiles import Constant
from solprop.translate.linop import profile_table
from solprop.translate.stage import scalars

EPS = np.finfo(float).eps


@pytest.fixture
def case():
    return mixture_case()


def test_compiled_channels_share_labels_rates_and_linear_density(case):
    mixture, _, mode, _, problem = case
    assert problem.schema_version == 38
    assert problem.flags.n_channels == 2
    assert problem.flags.n_raman == 0
    assert problem.responses.labels == mixture.labels
    assert problem.responses.n_species == 2
    assert problem.responses.plasma_coefficients is None
    assert tuple(c.label for c in problem.responses.plasma_channels) == mixture.labels
    for z in (0.0, 0.00037, 0.001):
        bundle = scalars(problem, 0.0, z)
        densities = mixture.density(np.array([z]))[:, 0]
        assert_array_equal(bundle.plasma_densities, densities)
        assert bundle.density == float(mode.density(np.array([z]))[0])
        for channel, table in zip(
            problem.responses.plasma_channels,
            problem.responses.ionisation_table,
            strict=True,
        ):
            assert channel.rate is table
        expected = sum(
            c.species.gamma3 * rho
            for c, rho in zip(mixture.components, densities, strict=True)
        )
        assert bundle.kerr_density == expected
        assert problem.responses.kerr_coefficients.coefficient == EPS0


def test_channel_depletion_is_independent_under_a_shared_field(case):
    problem = case[-1]
    bundle = scalars(problem, 0.0, 0.0)
    t = np.linspace(-30e-15, 30e-15, 1025)
    field = (3e10 * np.exp(-((t / 10e-15) ** 2)) * np.cos(2.4e15 * t))[None, None, :]
    dt = float(t[1] - t[0])
    rates, fractions = passes.plasma_channel_fractions(
        field, problem.responses.plasma_channels, bundle.plasma_preionisation, dt
    )
    for index, channel in enumerate(problem.responses.plasma_channels):
        single, _ = passes.plasma_scan(
            field,
            rates[index],
            channel,
            preionisation=bundle.plasma_preionisation[index],
            dt=dt,
        )
        # One channel's scan alone and among several: the plasma ladder's running-sum
        # budget, 4 n eps, holds whatever order the scans take.
        assert_allclose(fractions[index], single, rtol=4 * t.size * EPS, atol=0)
    assert np.all(np.diff(fractions, axis=-1) >= 0)
    assert np.all(fractions <= 1.0)


def test_shared_current_integrals_equal_the_sum_of_independent_responses(case):
    problem = case[-1]
    bundle = scalars(problem, 0.0, 0.0)
    t = np.linspace(-30e-15, 30e-15, 1025)
    field = (3e10 * np.exp(-((t / 10e-15) ** 2)) * np.cos(2.4e15 * t))[None, None, :]
    dt = float(t[1] - t[0])
    # Reordering the weighted trapezoid sums costs at most 16 operations per time
    # sample plus 64 per channel. Scale by the absolute component envelope so
    # cancellation between the channels cannot weaken the bound.
    rounding = (16 * t.size + 64 * problem.flags.n_channels) * EPS
    separate = []
    for channel, density, initial in zip(
        problem.responses.plasma_channels,
        bundle.plasma_densities,
        bundle.plasma_preionisation,
        strict=True,
    ):
        rate = passes.ionisation_rate(field, channel.rate)
        fraction, current = passes.plasma_scan(
            field, rate, channel, preionisation=initial, dt=dt
        )
        separate.append(
            passes.plasma_current(
                field, rate, fraction, current, channel, density=density, dt=dt
            )
        )
    bound = rounding * sum(float(np.max(np.abs(value))) for value in separate)
    assert bound > 0.0
    actual = passes.plasma_channels_polarisation(
        field,
        problem.responses.plasma_channels,
        bundle.plasma_densities,
        bundle.plasma_preionisation,
        dt,
    )
    assert_allclose(actual, sum(separate), rtol=0, atol=bound)


def test_constant_rate_ionisation_work_counts_only_new_electrons():
    problem = mixture_case(exponent=0)[-1]
    channels = problem.responses.plasma_channels
    bundle = scalars(problem, 0.0, 0.0)
    duration, amplitude = 40e-15, 3e10
    errors = []
    for intervals in (128, 256, 512):
        dt = duration / intervals
        t = np.arange(intervals + 1) * dt
        field = np.full((1, 1, t.size), amplitude)
        rates_at_field = np.array(
            [float(channel.rate(amplitude)) for channel in channels]
        )
        # For constant W the trapezoid of W(1-f0)exp(-Wt), divided by the exact
        # new fraction, is (W*dt/2)coth(W*dt/2). Its excess is <= (W*dt)^2/12.
        # Allow cumulative-exposure and exp rounding separately before evaluating.
        relative_bound = (rates_at_field * dt) ** 2 / 12 + (16 * t.size + 64) * EPS
        rates, fractions = passes.plasma_channel_fractions(
            field, channels, bundle.plasma_preionisation, dt
        )
        newly_freed = fractions[:, 0, -1] - np.asarray(bundle.plasma_preionisation)
        inventory = fractions[:, 0, -1]
        work_per_particle = np.array(
            [
                channel.ionisation_potential
                * trapezoid(rate[0] * (1 - fraction[0]), dx=dt)
                for channel, rate, fraction in zip(
                    channels, rates, fractions, strict=True
                )
            ]
        )
        expected = (
            np.array([channel.ionisation_potential for channel in channels])
            * newly_freed
        )
        assert np.all(inventory > newly_freed)
        for actual, target, bound in zip(
            work_per_particle, expected, relative_bound, strict=True
        ):
            assert_allclose(actual, target, rtol=float(bound), atol=0)
        errors.append(float(np.max(np.abs(work_per_particle / expected - 1))))
    # Here W*T <= 0.8. The next term relative to the leading trapezoid defect is
    # <= (0.8/128)^2/60, and the rounding allowance is <0.5% of the finest defect.
    assert_allclose(
        np.log2(np.asarray(errors[:-1]) / errors[1:]), 2.0, rtol=0, atol=0.02
    )


@pytest.mark.parametrize(
    "damage,match",
    [
        ("label", "label"),
        ("rate", "rate table"),
        ("density", "density"),
        ("preionisation", "preionisation"),
        ("species", "matching lengths"),
    ],
)
def test_channel_contract_refuses_inconsistent_population_data(case, damage, match):
    problem = case[-1]
    responses = problem.responses
    channels = responses.plasma_channels
    if damage == "label":
        responses = dataclasses.replace(
            responses,
            plasma_channels=(
                dataclasses.replace(channels[0], label="missing"),
                channels[1],
            ),
        )
    elif damage == "rate":
        responses = dataclasses.replace(
            responses, ionisation_table=tuple(reversed(responses.ionisation_table))
        )
    elif damage in ("density", "preionisation"):
        bad = profile_table(
            Constant(-1.0 if damage == "density" else 1.1),
            0.0,
            problem.linear.z_span[1],
        )
        channels = (dataclasses.replace(channels[0], **{damage: bad}), channels[1])
        responses = dataclasses.replace(responses, plasma_channels=channels)
    else:
        responses = dataclasses.replace(responses, species_names=("one",))
    with pytest.raises(TranslationError, match=match):
        dataclasses.replace(problem, responses=responses).validate()


def test_multi_channel_pass_requires_all_channel_scalars(case):
    channels = case[-1].responses.plasma_channels
    field = np.zeros((1, 1, 4))
    with pytest.raises(BackendError, match="initial fraction"):
        passes.plasma_channel_fractions(field, channels, (), 1e-16)
    with pytest.raises(BackendError, match="density"):
        passes.plasma_channels_polarisation(field, channels, (), (0.0, 0.0), 1e-16)
