# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Audit Yudin--Ivanov phase dependence and the proposed loss-current closure."""

import math

import numpy as np

from envelope_oracle_angular import angular_rule
from envelope_oracle_ppt import elliptic_action
from rate_oracles import HISTORICAL_ATOMS, adk_oracle
from solprop.constants import AU_EFIELD, AU_TIME, C
from solprop.nonlinear.optmat_rates import OptmatRate


def yudin_exponent(gamma, theta):
    """Return Phi from Eq. (12); theta is the phase of a linear driving field."""
    sine = np.abs(np.sin(theta))
    a = 1 + gamma**2 - sine**2
    b = np.hypot(a, 2 * gamma * sine)
    plus = np.sqrt((b + a) / 2)
    # The product of the two roots is gamma*sin(theta), avoiding b-a cancellation.
    minus = gamma * sine / plus
    c = np.hypot(plus + gamma, minus + sine)
    return (
        (gamma**2 + sine**2 + 0.5) * np.log(c) - 1.5 * minus * sine - 0.5 * plus * gamma
    )


def yudin_action_integral(gamma, theta):
    """Evaluate Eq. (5)'s complex action analytically at its zero-velocity saddle."""
    saddle = np.arcsin(np.sin(theta) - 1j * gamma)
    integral = (
        0.5 * (gamma**2 + np.sin(theta) ** 2 + 0.5) * (theta - saddle)
        + np.sin(theta) * (np.cos(theta) - np.cos(saddle))
        - (np.sin(2 * theta) - np.sin(2 * saddle)) / 8
    )
    return 2 * integral.imag


def yudin_experiments():
    rows = []
    omega = 2 * np.pi * C / 800e-9 * AU_TIME
    for species in ("He", "Ar"):
        # The potential, hartree, of solprop's former registry (the oracle's input).
        potential = HISTORICAL_ATOMS[species][0]
        for field in (0.04, 0.06, 0.08):
            gamma = omega * math.sqrt(2 * potential) / field
            scale = field**2 / omega**3
            peak = float(yudin_exponent(gamma, 0))
            zero = float(yudin_exponent(gamma, np.pi / 2))
            means = []
            for order in (32, 64, 128):
                nodes, weights = angular_rule(order)
                phase = (nodes + 1) * np.pi / 4
                means.append(
                    float(
                        np.dot(
                            weights,
                            np.exp(-scale * (yudin_exponent(gamma, phase) - peak)),
                        )
                        / 2
                    )
                )
            existing = {}
            existing_parameters = {}
            # The ADK test oracle is solprop's former ADK; optmat's named PPT form
            # replaces its former PPT (both deleted, SEP 3 Phase 4b).
            for name, rate in (
                ("ADK", adk_oracle(species)),
                ("PPT", OptmatRate(species, 800e-9, model="ppt_ilkov")),
            ):
                existing[name] = float(
                    np.dot(weights, rate(field * AU_EFIELD * np.cos(phase)))
                    / (2 * float(rate(field * AU_EFIELD)))
                )
                existing_parameters[name] = rate.parameters
            offsets = np.array([1e-2, 1e-3, 1e-4])
            near_node = np.pi / 2 - offsets
            relative_rates = np.exp(-scale * (yudin_exponent(gamma, near_node) - peak))
            circular_phi = (gamma**2 + 2) * 2 * np.arcsinh(
                gamma / 2
            ) - 2 * gamma * math.sqrt(1 + gamma**2 / 4)
            circular_ppt = elliptic_action(gamma, 1)[1]
            rows.append(
                {
                    "species": species,
                    "field_au": field,
                    "wavelength_m": 800e-9,
                    "gamma": gamma,
                    "zero_crossing_rate_over_peak": math.exp(-scale * (zero - peak)),
                    "normalised_cycle_means_32_64_128": means,
                    "normalised_existing_cycle_means": existing,
                    "existing_model_parameters": existing_parameters,
                    "angular_order_status": (
                        "inconclusive: spectral quadrature has "
                        "no asserted algebraic order"
                    ),
                    "phase_offsets_from_zero": offsets.tolist(),
                    "naive_loss_magnitude_in_peak_rate_over_amplitude_units": (
                        relative_rates / np.cos(near_node)
                    ).tolist(),
                    "circular_yudin_exponent": scale * circular_phi,
                    "circular_ppt_exponent": 2 * potential / omega * circular_ppt,
                    "scope": (
                        "normalised phase shapes and exponents only; "
                        "Eq. (18) prefactor not implemented"
                    ),
                }
            )
    return rows
