# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validate the envelope plasma numerical contract before backend execution."""

from contextlib import nullcontext
from dataclasses import dataclass, fields, replace

import numpy as np
import pytest

from rate_oracles import adk_oracle
from solprop.exceptions import (
    ConvergenceError,
    SolpropWarning,
    TranslationError,
)
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse
from solprop.profiles import Constant
from solprop.translate.envelope_plasma import (
    AngularRule,
    CycleIntegralData,
    CycleValidation,
    EnvelopePlasmaAssembly,
    EnvelopePlasmaChannel,
    EnvelopeSourceFactors,
    build_cycle_integrals,
    cycle_moments,
    source_factors,
)
from solprop.translate.ionisation import RateTable
from solprop.translate.responses import (
    CompiledResponses,
    PlasmaChannel,
    _compile_envelope_channels,
    _MaterialProfiles,
    _ResponseBuild,
    compile_responses,
)


@pytest.fixture(scope="module")
def grid():
    return EnvelopeGrid.from_limits(0.01, 800e-9, (500e-9, 1.2e-6), 80e-15)


@pytest.fixture(scope="module")
def cycle():
    return build_cycle_integrals(
        EnvelopePlasmaResponse("Ar", adk_oracle("Ar")), (1e9, 1e11)
    )


def test_envelope_plasma_contract_fields_are_frozen():
    expected = {
        RateTable: (
            "log_rate",
            "coefficients",
            "log_field_min",
            "d_log_field",
            "field_min",
            "field_max",
            "rate_max",
            "error",
            "key",
            "convention",
            "species_identity",
            "ionisation_potential",
        ),
        CompiledResponses: (
            "response_order",
            "kerr_coefficients",
            "ionisation_table",
            "plasma_channels",
            "raman_terms",
            "density",
            "spectral_responses",
            "prefactor_representation",
            "prefactor",
            "prefactor_expansion",
            "prefactor_constant",
            "temporal_absorber",
            "temporal_absorber_fine",
            "absorber_scheme",
            "separable",
            "labels",
            "species_names",
            "species_densities",
            "envelope_plasma",
            # Schema 12 appends the general-modal leaf; nothing above it moved.
            "regional_responses",
            # Schema 24 appends SEP 6's dispersive electronic leaf.
            "dispersive_kerr",
            # Schema 25 appends SEP 18's running-integral rule of the plasma column.
            "plasma_integral",
        ),
        AngularRule: ("nodes", "weights"),
        CycleValidation: ("probes", "residuals", "cutoff_rate", "format_version"),
        CycleIntegralData: (
            "rate",
            "rules",
            "validation",
            "identity",
            "moment_rtol",
            "negligible_rate",
        ),
        EnvelopePlasmaChannel: (
            "label",
            "species_identity",
            "ionisation_potential",
            "density",
            "preionisation",
            "cycle",
            "charge_mass_ratio",
            "electrons_per_event",
        ),
        EnvelopeSourceFactors: ("active", "drude", "loss", "carrier_policy", "omega0"),
        EnvelopePlasmaAssembly: ("channels", "factors"),
    }
    for cls, names in expected.items():
        assert tuple(field.name for field in fields(cls)) == names


def test_cycle_data_are_validated_owned_readonly_and_cached(cycle):
    assert cycle is build_cycle_integrals(
        EnvelopePlasmaResponse("Ar", adk_oracle("Ar")), (1e9, 1e11)
    )
    assert cycle.rate.field_min < 1e9
    assert cycle.rate.convention == "instantaneous-isotropic"
    assert tuple(len(rule.nodes) for rule in cycle.rules) == (64, 128, 256, 512)
    assert 0 <= np.max(cycle.validation.residuals) <= 2e-6
    for array in (
        cycle.rate.log_rate,
        cycle.rate.coefficients,
        cycle.validation.probes,
        cycle.validation.residuals,
        *(rule.nodes for rule in cycle.rules),
        *(rule.weights for rule in cycle.rules),
    ):
        assert not array.flags.writeable
        assert array.flags.c_contiguous


@pytest.mark.parametrize("ratio", [0.0, 1e-8, 0.35, 1 - 1e-12, 1.0])
def test_cycle_tables_match_direct_moments(cycle, ratio):
    amplitude = 3.713e10
    actual = cycle_moments(amplitude, ratio, cycle)
    response = EnvelopePlasmaResponse("Ar", cycle.rate)
    source = response.sources(
        np.array([amplitude, 1j * amplitude * ratio])[:, None], 1.0, dt=1e-16
    )
    # Production angular error has its own accepted 2e-6 budget. The adaptive
    # physical oracle is tighter by four orders of magnitude.
    np.testing.assert_allclose(
        actual[0] + ratio**2 * actual[1],
        source.ionisation_rate[0],
        rtol=2e-6,
        atol=1e-30,
    )
    current = (
        2
        * response.potential
        / amplitude
        * np.array([actual[0], 1j * ratio * actual[1]])
    )
    np.testing.assert_allclose(
        current,
        source.loss_current[:, 0],
        rtol=2e-6,
        atol=1e-30 * response.potential / amplitude,
    )


def test_absolute_frequency_source_factors(grid):
    actual = source_factors(grid, shock=True)
    active = grid.active_mask & (grid.omega > 0)
    np.testing.assert_array_equal(actual.drude[~active], 0)
    np.testing.assert_array_equal(actual.loss[~active], 0)
    # Reciprocal squared versus division by the square differs at most two ulps.
    np.testing.assert_allclose(
        actual.drude[active], -1 / grid.omega[active] ** 2, rtol=5e-16, atol=0
    )
    np.testing.assert_allclose(
        actual.loss[active], -1j / grid.omega[active], rtol=2e-16, atol=0
    )
    with pytest.warns(SolpropWarning, match="shock=False") as recorded:
        frozen = source_factors(grid, shock=False)
    assert len(recorded) == 1
    assert frozen.carrier_policy == "frozen-carrier"
    np.testing.assert_array_equal(frozen.active, actual.active)
    np.testing.assert_allclose(
        frozen.drude[active], -1 / grid.omega0**2, rtol=5e-16, atol=0
    )
    np.testing.assert_allclose(
        frozen.loss[active], -1j / grid.omega0, rtol=2e-16, atol=0
    )


def test_compilation_keeps_currents_separate_from_ordinary_polarisation(grid):
    response = EnvelopePlasmaResponse("Ar", adk_oracle("Ar"), preionisation=0.2)
    mode = MarcatiliMode(125e-6, "Ar", 1.0, loss=False)
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        compiled = compile_responses(
            [KerrResponse(1e-52, thg=False), response],
            grid,
            mode,
            density=2.5e25,
            z_max=0.01,
            ionisation_field_range=(1e9, 1e11),
        )
    assembly = compiled.envelope_plasma
    assert assembly is not None
    assert compiled.plasma_channels == ()
    assert compiled.spectral_responses == (assembly,)
    assert compiled.population_channels == assembly.channels
    assert compiled.ionisation_table[0] is assembly.channels[0].rate
    assert assembly.channels[0].preionisation(0) == pytest.approx(0.2, rel=0, abs=1e-16)
    assert tuple(map(str, compiled.response_order)) == (
        "envelope_cycle",
        "envelope_population",
        "envelope_sources",
        "kerr",
        "accumulate",
    )


def test_rate_model_identity_changes_cycle_cache():
    first = build_cycle_integrals(
        EnvelopePlasmaResponse("Ar", OptmatRate("Ar", 800e-9, model="ppt_ilkov")),
        (1e9, 1e11),
    )
    second = build_cycle_integrals(
        EnvelopePlasmaResponse("Ar", OptmatRate("Ar", 900e-9, model="ppt_ilkov")),
        (1e9, 1e11),
    )
    assert first.identity != second.identity


def test_invalid_cycle_data_and_domains_fail(cycle):
    with pytest.raises(TranslationError, match="unvalidated"):
        replace(cycle, moment_rtol=1e-3)
    with pytest.raises(TranslationError, match="unvalidated"):
        replace(cycle, rules=cycle.rules[:1])
    with pytest.raises(TranslationError, match="angular rule"):
        AngularRule(np.array([-0.5, 0.5]), np.array([1.0, -1.0]))
    with pytest.raises(TranslationError, match="ellipse"):
        cycle_moments(1.0, 1.1, cycle)
    np.testing.assert_array_equal(cycle_moments(0.0, 0.0, cycle), 0)
    with pytest.raises(TranslationError, match="ionisation_field_range"):
        build_cycle_integrals(EnvelopePlasmaResponse("Ar", adk_oracle("Ar")), None)
    with pytest.raises(TranslationError, match="own field range"):
        build_cycle_integrals(EnvelopePlasmaResponse("Ar", cycle.rate), (1e9, 1e11))


def test_thg_combined_with_envelope_plasma_is_refused(grid):
    with pytest.raises(TranslationError, match="THG"):
        compile_responses(
            [
                KerrResponse(1e-52, thg=True),
                EnvelopePlasmaResponse("Ar", adk_oracle("Ar")),
            ],
            grid,
            MarcatiliMode(125e-6, "Ar", 1.0, loss=False),
            density=2.5e25,
            z_max=0.01,
            ionisation_field_range=(1e9, 1e11),
        )


def test_cycle_refinement_refuses_unresolved_clipped_peak(cycle):
    # This extreme linearly polarised field is outside the table domain. Its
    # narrow unclipped interval makes the inverse-square loss moment unresolved
    # by 512 points. An explicit failure is required instead of a biased current.
    with pytest.raises(ConvergenceError, match="512"):
        cycle_moments(1e16, 0.0, cycle)


def test_malformed_validation_evidence_and_factors_are_refused(cycle, grid):
    with pytest.raises(TranslationError, match="validation evidence"):
        replace(cycle.validation, residuals=np.zeros((1, 3)))
    factors = source_factors(grid, shock=True)
    with pytest.raises(TranslationError, match="source factors"):
        replace(factors, drude=-factors.drude)
    with pytest.raises(TranslationError, match="unique labelled"):
        EnvelopePlasmaAssembly((), factors)


def test_rate_table_retains_only_valid_convention_and_species_metadata(cycle):
    with pytest.raises(TranslationError, match="potential"):
        replace(cycle.rate, ionisation_potential=-1.0)
    with pytest.raises(TranslationError, match="identity"):
        replace(cycle.rate, species_identity=" ")


def test_analytic_rate_needs_a_valid_tabulation_domain():
    for domain in ((1e11, 1e9), (0.0, 1e11), (1e9, np.inf)):
        with pytest.raises(TranslationError, match="invalid envelope"):
            build_cycle_integrals(
                EnvelopePlasmaResponse("Ar", adk_oracle("Ar")), domain
            )


def test_compiled_envelope_channel_and_spectral_consistency(grid, cycle):
    mode = MarcatiliMode(125e-6, "Ar", 1.0, loss=False)
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        compiled = compile_responses(
            [EnvelopePlasmaResponse("Ar", cycle.rate)],
            grid,
            mode,
            density=2.5e25,
            z_max=0.01,
        )
    assembly = compiled.envelope_plasma
    assert assembly is not None
    channel = assembly.channels[0]
    with pytest.raises(TranslationError, match="channel constants"):
        replace(channel, electrons_per_event=2)
    with pytest.raises(TranslationError, match="spectral_responses"):
        replace(compiled, spectral_responses=())
    field_channel = PlasmaChannel(
        label=channel.label,
        rate=channel.rate,
        ionisation_potential=channel.ionisation_potential,
        charge_mass_ratio=channel.charge_mass_ratio,
        density=channel.density,
        preionisation=channel.preionisation,
    )
    with pytest.raises(TranslationError, match="cannot coexist"):
        replace(compiled, plasma_channels=(field_channel,))


def test_mismatched_envelope_material_is_refused(grid, cycle):
    with pytest.raises(TranslationError, match="inconsistent material"):
        compile_responses(
            [EnvelopePlasmaResponse("Ar", cycle.rate)],
            grid,
            MarcatiliMode(125e-6, "He", 1.0, loss=False),
            density=2.5e25,
            z_max=0.01,
        )


@dataclass(frozen=True)
class UntabulatedRate:
    convention: str = "instantaneous-isotropic"

    def __call__(self, field_amplitude):
        return np.asarray(field_amplitude) ** 2


def test_compilation_requires_rate_tabulation_metadata():
    with pytest.raises(TranslationError, match="declare threshold"):
        build_cycle_integrals(
            EnvelopePlasmaResponse("Ar", UntabulatedRate()), (1e9, 1e11)
        )


def test_independent_validation_rejects_biased_current(monkeypatch):
    original = EnvelopePlasmaResponse.sources

    def biased(self, field_t, density, *, dt):
        actual = original(self, field_t, density, dt=dt)
        return replace(actual, loss_current=actual.loss_current * 1.001)

    monkeypatch.setattr(EnvelopePlasmaResponse, "sources", biased)
    # A distinct domain ensures that this requests new independent evidence.
    with pytest.raises(ConvergenceError, match="independent"):
        build_cycle_integrals(
            EnvelopePlasmaResponse("Ar", adk_oracle("Ar")), (1e9, 4.123e10)
        )


def test_corrupted_rate_convention_is_rejected(cycle):
    table = replace(cycle.rate)
    object.__setattr__(table, "convention", "invalid")
    with pytest.raises(TranslationError, match="optical-cycle convention"):
        table.__post_init__()


def test_cycle_validator_handles_a_malformed_probe_rank(cycle):
    evidence = replace(cycle.validation)
    object.__setattr__(evidence, "probes", np.ones(2))
    with pytest.raises(TranslationError, match="validation evidence"):
        evidence.__post_init__()


def test_channel_compiler_rejects_a_field_grid_even_if_declaration_check_is_bypassed(
    cycle,
):
    field = FieldGrid.from_limits(0.01, 800e-9, (500e-9, 1.2e-6), 80e-15)
    density = Constant(2.5e25)
    material = _MaterialProfiles(("Ar",), ("Ar",), (density,), density)
    build = _ResponseBuild(field, 0.01, 1e-8, None)
    with pytest.raises(TranslationError, match="requires an EnvelopeGrid"):
        _compile_envelope_channels(
            (EnvelopePlasmaResponse("Ar", cycle.rate),), material, {}, build, True
        )
