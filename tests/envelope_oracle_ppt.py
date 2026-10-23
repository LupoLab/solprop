# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bounded elliptic PPT study with a separate circular Coulomb-factor audit.

No Coulomb correction is guessed. Circular rates are Barth--Smirnova (2013),
Eqs. (76)--(78), with C_kappa,l=1. Arbitrary ellipticity uses Popov et al. (1968),
Eqs. (37), (41), (47)--(49). Popov (2004), Eq. (2.13), supplies the labelled
circular correction. Atomic asymptotic coefficients remain unspecified.
None of these diagnostics is a production rate.
"""

import math

import numpy as np
from scipy import optimize, special

from envelope_oracle_angular import axis_moments
from rate_oracles import HISTORICAL_ATOMS, adk_oracle
from solprop.constants import AU_EFIELD, AU_TIME, C
from solprop.nonlinear.optmat_rates import OptmatRate


def elliptic_action(gamma, ellipticity):
    """Return tau and f for probability proportional to exp(-2 Ip f/omega)."""

    def equation(tau):
        return (
            np.sinh(tau) ** 2
            - ellipticity**2 * (np.cosh(tau) - np.sinh(tau) / tau) ** 2
            - gamma**2
        )

    tau = optimize.brentq(equation, 1e-7, 20, xtol=1e-14)
    action = (1 + (1 + ellipticity**2) / (2 * gamma**2)) * tau - (
        1 - ellipticity**2 + 2 * ellipticity**2 * np.tanh(tau) / tau
    ) * np.sinh(2 * tau) / (4 * gamma**2)
    return float(tau), float(action)


def circular_coulomb_audit(potential, field, omega):
    """Return Popov (2004) Eq. (2.13)'s factor and Eq. (2.12)'s scale for Z=1."""
    kappa = math.sqrt(2 * potential)
    effective_principal = 1 / kappa
    reduced_field = field / kappa**3
    gamma = omega * kappa / field
    correction = (reduced_field * math.sqrt(1 + gamma**2)) ** (-2 * effective_principal)
    validity_ratio = gamma * math.sqrt(effective_principal * reduced_field)
    return correction, validity_ratio


def circular_short_range(potential, field, omega, order=96, terms=512):
    """Return per-electron s, p0, p+, p- rates in atomic units, C_kappa,l=1."""
    gamma = omega * math.sqrt(2 * potential) / field
    vector_potential = field / omega
    threshold = (potential + vector_potential**2 / 2) / omega
    photons = np.arange(math.floor(threshold) + 1, math.floor(threshold) + 1 + terms)
    momentum = np.sqrt(2 * (photons * omega - potential) - vector_potential**2)
    nodes, weights = special.roots_legendre(order)
    longitudinal = momentum[:, None] * (nodes + 1) / 2
    chi = (
        photons[:, None]
        * omega
        / (vector_potential * np.sqrt(momentum[:, None] ** 2 - longitudinal**2))
    )
    root = np.sqrt(1 - 1 / chi**2)
    common = np.exp(-2 * photons[:, None] * (np.arccosh(chi) - root)) / (
        root * photons[:, None]
    )
    measure = momentum[:, None] * weights / 2
    kappa = math.sqrt(2 * potential)
    s_rate = kappa / (2 * np.pi) * np.sum(common * measure, axis=1)
    p_zero = (
        3 / (2 * np.pi * kappa) * np.sum(longitudinal**2 * common * measure, axis=1)
    )
    rates = [s_rate, p_zero]
    for sign in (1, -1):
        angular = (
            2 * chi**2 * (1 - sign * root)
            - (1 + gamma**2) * photons[:, None] / threshold
        ) ** 2 / chi**2
        rates.append(
            3
            * kappa
            / (16 * np.pi * gamma**2)
            * np.sum(angular * common * measure, axis=1)
        )
    values = np.array(rates)
    return np.sum(values, axis=1), float(
        np.max(np.sum(values[:, -32:], axis=1) / np.sum(values, axis=1))
    )


def elliptic_short_range(
    potential, field, omega, ellipticity, orbital, accuracy=(96, 512)
):
    """Evaluate Popov et al. (1968), (47)--(49), per unpolarised electron, C=1."""
    order, terms = accuracy
    gamma = omega * math.sqrt(2 * potential) / field
    tau, action = elliptic_action(gamma, ellipticity)
    hyperbolic = np.tanh(tau)
    delta = (1 - ellipticity**2) * tau + ellipticity**2 * hyperbolic
    psi = np.arctanh(ellipticity * (1 / hyperbolic - 1 / tau))
    threshold = potential / omega * (1 + (1 + ellipticity**2) / (2 * gamma**2))
    excess = (
        np.arange(math.floor(threshold) + 1, math.floor(threshold) + 1 + terms)
        - threshold
    )
    central_excess = (
        potential / omega * ellipticity**2 / gamma**2 * (np.sinh(tau) / tau) ** 2
    )
    a_coefficient = (
        excess * (1 + (1 - ellipticity**2) * (tau - hyperbolic) / delta) * tau
    )
    b_coefficient = (
        2
        * excess
        * (1 + ellipticity**2 * (1 - hyperbolic / tau) ** 2 / (delta * hyperbolic))
        * tau
    )
    c_coefficient = excess * tau / delta * hyperbolic
    centre = np.sqrt(central_excess / excess)
    nodes, weights = special.roots_legendre(order)
    transverse = 1 - nodes**2
    argument = c_coefficient[:, None] * transverse
    exponent = (
        -a_coefficient[:, None] * transverse
        - b_coefficient[:, None] * (nodes - centre[:, None]) ** 2
    )
    integral = (np.exp(exponent + argument) * special.i0e(argument)) @ weights
    channels = np.sqrt(excess * hyperbolic / 2) * integral
    prefactor = (
        omega
        / np.pi
        * math.sqrt(omega * hyperbolic / (2 * potential))
        * tau
        / delta
        * special.eval_legendre(orbital, np.cosh(2 * psi))
        / np.cosh(psi) ** 2
        * math.exp(-2 * potential / omega * action)
    )
    return float(prefactor * np.sum(channels)), float(
        np.sum(channels[-32:]) / np.sum(channels)
    )


def elliptic_convergence(potential, field, omega, ellipticity, orbital):
    """Record independent angular and photon refinements at a comparison point."""
    levels = [
        elliptic_short_range(potential, field, omega, ellipticity, orbital, (n, 512))
        for n in (32, 64, 128)
    ]
    extended, _ = elliptic_short_range(
        potential, field, omega, ellipticity, orbital, (128, 1024)
    )
    differences = [abs(value[0] / levels[-1][0] - 1) for value in levels[:-1]]
    return (
        levels[-1][0],
        levels[-1][1],
        {
            "relative_differences_32_64_128": differences,
            "512_to_1024_terms_relative_difference": abs(levels[-1][0] / extended - 1),
            "angular_order_status": (
                "inconclusive: spectral quadrature has no asserted algebraic order"
            ),
        },
    )


def ppt_experiments():
    omega = 2 * np.pi * C / 800e-9 * AU_TIME
    rows = []
    circular = []
    for species in ("He", "Ar"):
        # Potential (hartree) and orbital of solprop's former registry, the oracle's
        # inputs (the registry was deleted in SEP 3, Phase 4b).
        potential, orbital, _ = HISTORICAL_ATOMS[species]
        for field in (0.04, 0.06, 0.08):
            for policy in ("fixed_major_axis", "fixed_intensity"):
                for ellipticity in (0.0, 0.35, 0.7, 1.0):
                    major = (
                        field
                        if policy == "fixed_major_axis"
                        else field / math.sqrt(1 + ellipticity**2)
                    )
                    gamma = omega * math.sqrt(2 * potential) / major
                    tau, action = elliptic_action(gamma, ellipticity)
                    _, linear_action = elliptic_action(gamma, 0)
                    # This orbital factor is a short-range unpolarised-shell diagnostic.
                    psi = np.arctanh(ellipticity * (1 / np.tanh(tau) - 1 / tau))
                    shell_factor = float(
                        special.eval_legendre(orbital, np.cosh(2 * psi))
                    )
                    averages = {}
                    # The ADK test oracle is solprop's former ADK; optmat's named
                    # PPT form replaces its former PPT (both deleted, SEP 3 Phase 4b).
                    for name, rate in (
                        ("ADK", adk_oracle(species)),
                        ("PPT", OptmatRate(species, 800e-9, model="ppt_ilkov")),
                    ):
                        moments = axis_moments(
                            major * AU_EFIELD, ellipticity, rate, 256
                        )
                        averages[name] = float(moments[0] + ellipticity**2 * moments[1])
                    candidate, tail, convergence = elliptic_convergence(
                        potential, major, omega, ellipticity, orbital
                    )
                    candidate *= 2 * (2 * orbital + 1) / AU_TIME
                    candidate_model = "Popov-1968-equation-47-short-range"
                    if ellipticity == 1:
                        orbitals, tail = circular_short_range(potential, major, omega)
                        candidate = (
                            2
                            * (orbitals[0] if orbital == 0 else np.sum(orbitals[1:]))
                            / AU_TIME
                        )
                        candidate_model = "Barth-Smirnova-2013-circular-short-range"
                    rows.append(
                        {
                            "species": species,
                            "policy": policy,
                            "reference_field_au": field,
                            "major_au": major,
                            "ellipticity": ellipticity,
                            "gamma": gamma,
                            "tau": tau,
                            "exponent": -2 * potential / omega * action,
                            "elliptic_over_linear_exponential_at_same_major": math.exp(
                                -2 * potential / omega * (action - linear_action)
                            ),
                            "unpolarised_orbital_factor": shell_factor,
                            "short_range_candidate": candidate_model,
                            "short_range_occupied_shell_C_squared_one_s_inverse": float(
                                candidate
                            ),
                            "candidate_tail_relative": tail,
                            "popov1968_quadrature_and_photon_refinement": convergence,
                            "existing_cycle_averages_s_inverse": averages,
                        }
                    )
            levels = [
                circular_short_range(potential, field, omega, order)
                for order in (32, 64, 128)
            ]
            values = levels[-1][0]
            coulomb_factor, coulomb_validity_ratio = circular_coulomb_audit(
                potential, field, omega
            )
            extended, _ = circular_short_range(potential, field, omega, 128, 1024)
            elliptic_levels = [
                elliptic_short_range(potential, field, omega, 1, orbital, (order, 512))
                for order in (32, 64, 128)
            ]
            circular.append(
                {
                    "species_binding_energy": species,
                    "field_au": field,
                    "s_p0_pplus_pminus_per_electron_C_squared_one_s_inverse": (
                        values / AU_TIME
                    ).tolist(),
                    "occupied_shell_C_squared_one_s_inverse": float(
                        2
                        * (values[0] if orbital == 0 else np.sum(values[1:]))
                        / AU_TIME
                    ),
                    "counter_over_corotating": float(values[3] / values[2]),
                    "popov2004_circular_coulomb_factor": coulomb_factor,
                    "gamma_over_popov2004_perturbation_scale": coulomb_validity_ratio,
                    "coulomb_corrected_C_squared_one_shell_s_inverse": float(
                        2
                        * (values[0] if orbital == 0 else np.sum(values[1:]))
                        / AU_TIME
                        * coulomb_factor
                    ),
                    "relative_differences_32_64_128": [
                        float(np.max(np.abs(levels[i][0] - values) / values))
                        for i in (0, 1)
                    ],
                    "last_32_of_512_terms_relative": levels[-1][1],
                    "512_to_1024_terms_relative_difference": float(
                        np.max(np.abs(extended - values) / extended)
                    ),
                    "angular_order_status": "inconclusive: floor limited",
                    "elliptic_formula_at_circular_C_squared_one_s_inverse": (
                        2 * (2 * orbital + 1) * elliptic_levels[-1][0] / AU_TIME
                    ),
                    "elliptic_formula_quadrature_relative_differences": [
                        abs(value[0] / elliptic_levels[-1][0] - 1)
                        for value in elliptic_levels[:-1]
                    ],
                    "elliptic_formula_tail_relative": elliptic_levels[-1][1],
                    "elliptic_formula_circular_status": (
                        "invalid circular limit: nonsummable photon-channel tail"
                    ),
                }
            )
    limits = []
    for gamma in (0.02, 0.01, 0.005):
        for ellipse in (0, 0.35, 1):
            _, action = elliptic_action(gamma, ellipse)
            limits.append(
                {
                    "gamma": gamma,
                    "ellipticity": ellipse,
                    "g_minus_one": 3 * action / (2 * gamma) - 1,
                    "leading_prediction": -(1 - ellipse**2 / 3) * gamma**2 / 10,
                }
            )
    return {
        "elliptic_action_and_existing_rates": rows,
        "circular_short_range": circular,
        "adiabatic_limits": limits,
        "scope": (
            "Short-range C=1 elliptic and circular rates; "
            "no Coulomb-corrected atomic elliptic rate or accepted loss closure."
        ),
    }
