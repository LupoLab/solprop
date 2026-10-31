# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""optmat's gas Raman responses on solprop's existing Raman terms (SEP 3, Phase 1c).

Every reference here is optmat 0.1.1's own evaluation of the same response
(``GasRamanResponse.response``, ``fwhm``, ``groups``), or an independent statement of
the physics (Hellwarth's tensor, a causal convolution summed directly): what is
tested is that the adapter reproduces the selected optmat response on solprop's terms,
not optmat's physics.
"""

import dataclasses
import logging
import math
import warnings
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from optmat.gas.models.registry import default_model
from optmat.gas.raman_response import raman_response
from optmat.solid.bundle import silica_bundle

from former_raman import FORMER_N2, former_response, supplied_species
from solprop.constants import AMAGAT, EPS0, ROOM_TEMPERATURE
from solprop.exceptions import (
    MaterialError,
    MaterialRangeWarning,
    MaterialStatusWarning,
    RamanTemperatureWarning,
)
from solprop.grid import FieldGrid
from solprop.nonlinear.mixtures import raman_temperature, responses_for
from solprop.nonlinear.optmat_raman import (
    COMPOSITION_TOLERANCE,
    GroupDephasing,
    OptmatRamanResponse,
    RamanGroupLines,
    RamanShift,
    _Path,
    _reemit,
    _reference_density,
    gas_raman,
    mixture_raman,
)
from solprop.nonlinear.raman import RamanResponse, RamanTensorTerm, response_for
from solprop.output.schema import serialise
from solprop.physdata import Component, Gas, Mixture, Species, gamma, species
from solprop.physdata import _optmat as bridge
from solprop.physdata.raman_types import Dephasing, Linewidth, MolecularOptions
from solprop.physdata.selection import MaterialManifest, MaterialSelection
from solprop.profiles import Constant, Linear
from solprop.translate.raman import (
    MAX_DECAY_EXPONENT,
    decay_rate_profiles,
    undamped_kernels,
)

SELECTION = MaterialSelection(carrier_wavelength=800e-9)
SHIFTED = MaterialSelection(carrier_wavelength=800e-9, raman_shift="reference-density")
EPS = np.finfo(float).eps
DELAYS = np.linspace(-0.2e-12, 20e-12, 2001)
UNSHIFTED = ("N2", "O2", "D2", "CH4", "SF6")


def composition(*args: Any, **kwargs: Any) -> Any:
    """:func:`mixture_raman`, loosely typed for reading: each test checks the result."""
    return mixture_raman(*args, **kwargs)


def pure(*args: Any, **kwargs: Any) -> Any:
    """:func:`gas_raman`, loosely typed for reading."""
    return gas_raman(*args, **kwargs)


def optmat_species(name):
    """An optmat-resolved species, as a selection's fill takes it."""
    return species(name, selection=SELECTION)


def constant_fill(fractions, amagat=1.0, temperature=ROOM_TEMPERATURE):
    """Constant populations at the given total density in amagat."""
    return Mixture.of(
        *(
            Component(optmat_species(name), Constant(fraction * amagat * AMAGAT))
            for name, fraction in fractions.items()
        ),
        temperature=temperature,
    )


def summation_bound(response, lines):
    """Rounding bound of two summation orders over a response's lines.

    Each partial sum is bounded by sum |A_k|, and the two orders (optmat's per line,
    solprop's per group then per row) each round at most once per line, so the
    difference is below 4 eps sum |A_k| per line summed.
    """
    amplitude = np.abs(response.isotropic) + np.abs(response.anisotropic)
    return 4.0 * lines * EPS * float(np.sum(amplitude))


# ------------------------------------------------------------------ the group mapping


@pytest.mark.parametrize("gas", UNSHIFTED)
@pytest.mark.parametrize("amagat", [1.0, 1.8])
def test_the_summed_terms_are_optmat_response_at_the_density(gas, amagat):
    response = pure(gas, selection=SELECTION)
    source = response.source
    density = amagat * AMAGAT
    expected = source.response(DELAYS, density)
    got = response.impulse(DELAYS, density)
    assert_allclose(got, expected, rtol=0, atol=summation_bound(source, len(source)))
    # The mapping is exact at any density: the rows do not depend on it at all.
    assert not any(group.shifted for group in source.groups())


@pytest.mark.parametrize("gas", ["N2", "O2", "D2"])
def test_the_isotropic_and_rank_two_rows_are_optmat_shares(gas):
    response = pure(gas, selection=SELECTION)
    source = response.source
    isotropic, rank2 = response.tensor_impulse(DELAYS, AMAGAT)
    bound = summation_bound(source, len(source))
    assert_allclose(isotropic, source.response(DELAYS, AMAGAT, "isotropic"), atol=bound)
    assert_allclose(rank2, source.response(DELAYS, AMAGAT, "anisotropic"), atol=bound)


def test_terms_are_one_per_group_and_tensor_family_with_empty_rows_dropped():
    response = pure("N2", selection=SELECTION)
    groups = response.source.groups()
    expected = sum(
        int(np.any(group.isotropic)) + int(np.any(group.anisotropic))
        for group in groups
    )
    assert response.assembly.groups == len(groups) == 43
    assert response.assembly.terms == len(response.tensor_terms) == expected == 72
    for term in response.tensor_terms:
        row = term.model
        assert isinstance(row, RamanGroupLines)
        assert np.any(row.amplitudes)
        assert term.contraction == (
            "isotropic" if row.share == "isotropic" else "rank2"
        )
    # Rotational S lines are anisotropic only.
    rotational = [
        t for t in response.tensor_terms if "/rotation/" in t.model.line_ids[0]
    ]
    assert rotational
    assert all(t.contraction == "rank2" for t in rotational)


def test_dry_air_has_eighty_two_groups_before_the_tensor_split():
    air = Mixture.air(1.0, selection=SELECTION)
    response = composition(air, True, selection=SELECTION)
    assert response.assembly.groups == 82
    assert response.assembly.terms == 126
    assert response.labels == ("N2", "O2", "Ar")
    assert response.species == "N2"
    assert {name for name, _ in response.assembly.composition} == {"N2", "O2", "Ar"}


def test_the_response_is_causal():
    response = pure("O2", selection=SELECTION)
    before = np.array([-1e-12, -1e-15, 0.0])
    assert np.all(response.impulse(before, AMAGAT) == 0.0)
    for term in response.tensor_terms:
        assert np.all(term.model.undamped(before) == 0.0)
    grid = FieldGrid.from_limits(0.1, 800e-9, (200e-9, 4e-6), 2e-12)
    rows = undamped_kernels(response, grid)
    assert np.all(rows[:, 0] == 0.0)
    assert np.all(rows[:, grid.n_to :] == 0.0)


def test_the_decay_rate_is_pi_times_optmat_full_width():
    for gas in ("N2", "O2", "H2", "D2", "CH4"):
        response = pure(gas, selection=SELECTION, density=AMAGAT)
        source = response.source
        widths = source.fwhm(np.array([AMAGAT, 1.5 * AMAGAT]))
        for group in source.groups():
            dephasing = GroupDephasing.of(group, source)
            for column, density in enumerate((AMAGAT, 1.5 * AMAGAT)):
                expected = math.pi * widths[list(group.lines), column]
                # Gamma = pi FWHM: the group's SI coefficients against the lines'
                # widths in Hz, two roundings of each of three terms apart.
                assert_allclose(dephasing.decay_rate(density), expected, rtol=1e-14)


def test_the_vacuum_limit_is_finite_with_the_dicke_cap_and_zero_without_one():
    hydrogen = pure("H2", selection=SELECTION, density=AMAGAT).source
    for group in hydrogen.groups():
        first = group.lines[0]
        dephasing = GroupDephasing.of(group, hydrogen)
        cap = hydrogen.width_constant[first] + hydrogen.doppler_width[first]
        assert dephasing.decay_rate(0.0) == pytest.approx(math.pi * cap, rel=1e-14)
        assert dephasing.decay_rate(0.0) > 0.0
        assert dephasing.doppler_width == hydrogen.doppler_width[first]
    nitrogen = pure("N2", selection=SELECTION).source
    for group in nitrogen.groups():
        assert GroupDephasing.of(group, nitrogen).decay_rate(0.0) == 0.0


def test_a_dephasing_refuses_a_negative_or_infinite_density():
    group = pure("N2", selection=SELECTION).source.groups()[0]
    dephasing = GroupDephasing.of(group, pure("N2", selection=SELECTION).source)
    for bad in (-1.0, math.inf, math.nan):
        with pytest.raises(MaterialError, match="non-negative number density"):
            dephasing.decay_rate(bad)


# ------------------------------------------------------------------ the tensor form


def hellwarth(a, b, field, density, dt):
    """Return Hellwarth's polarisation by a direct causal sum, field grid with THG.

    ``P_i(t) = eps0 N [E_i(t) int a(tau) (E.E)(t - tau) + sum_j E_j(t) int b(tau)
    E_i(t - tau) E_j(t - tau)]``, each integral the discrete causal sum ``dt sum_m
    k[m] D[n - m]``.
    """

    def causal(kernel, drive):
        return dt * np.convolve(kernel, drive)[: drive.size]

    square = field[0] ** 2 + field[1] ** 2
    out = np.zeros_like(field)
    for i in range(2):
        out[i] = field[i] * causal(a, square)
        for j in range(2):
            out[i] += field[j] * causal(b, field[i] * field[j])
    return EPS0 * density * out


@pytest.mark.parametrize("gas", ["N2", "O2"])
def test_the_vector_polarisation_is_hellwarths_with_optmat_a_and_b(gas):
    response = pure(gas, selection=SELECTION)
    source = response.source
    dt = 2e-15
    n = 1024
    t = np.arange(n) * dt
    density = AMAGAT
    envelope = np.exp(-(((t - 0.3e-12) / 40e-15) ** 2))
    later = np.exp(-(((t - 1.2e-12) / 40e-15) ** 2))
    carrier = 2.35e15
    field = 1e9 * np.stack(
        [
            envelope * np.cos(carrier * t) + 0.3 * later * np.sin(carrier * t),
            0.6 * later * np.cos(carrier * t + 0.4),
        ]
    )
    got = response.evaluate(field, density, kind="field", dt=dt)
    a = source.response(t, density, "a")
    b = source.response(t, density, "b")
    expected = hellwarth(a, b, field, density, dt)
    # The FFT product against a direct sum: rounding of the transforms, a few eps
    # times log2(n) of the largest term.
    assert_allclose(got, expected, rtol=0, atol=1e-11 * float(np.max(np.abs(expected))))


def test_a_cross_polarised_probe_sees_minus_half_of_a_rotational_response():
    # Pure S lines: a = -aniso/2 and a + b = aniso, so after the pump has gone a
    # probe across it sees -1/2 of what a probe along it sees.
    response = pure("N2", MolecularOptions(vibration=False), selection=SELECTION)
    dt = 2e-15
    n = 2048
    t = np.arange(n) * dt
    pump = 1e9 * np.exp(-(((t - 0.2e-12) / 30e-15) ** 2)) * np.cos(2.35e15 * t)
    probe = np.exp(-(((t - 2.0e-12) / 300e-15) ** 2)) * np.cos(2.35e15 * t)
    along = response.evaluate(
        np.stack([pump + probe, 0 * t]), AMAGAT, kind="field", dt=dt
    )
    across = response.evaluate(np.stack([pump, probe]), AMAGAT, kind="field", dt=dt)
    inside = (t > 1.8e-12) & (t < 2.2e-12)
    scale = float(np.max(np.abs(along[0][inside])))
    window = inside & (np.abs(along[0]) > 1e-3 * scale)
    ratio = across[1][window] / along[0][window]
    # The probe's own drive is 1e-12 of the pump's, and the transforms round at
    # 1e-16 of the largest value: kept where the response is above 1e-3 of it, the
    # ratio is -1/2 to 1e-9.
    assert_allclose(ratio, -0.5, rtol=1e-9)


# ------------------------------------------------------------------ mixtures


def test_a_mixture_is_its_composition_response_driven_by_the_total_density():
    air = Mixture.air(1.0, selection=SELECTION)
    response = composition(air, True, selection=SELECTION)
    densities = air.density(np.array([0.0]))[:, 0]
    total = float(np.sum(densities))
    fractions = dict(response.assembly.composition)
    for component, density in zip(air.components, densities, strict=True):
        assert fractions[component.species.name] == pytest.approx(
            density / total, rel=1e-15
        )
    direct = raman_response(fractions, ROOM_TEMPERATURE, wavelength=800e-9)
    expected = direct.response(DELAYS, total)
    bound = summation_bound(direct, len(direct))
    assert_allclose(response.impulse(DELAYS, total), expected, rtol=0, atol=bound)
    assert response.material_records[0].digest == direct.identity.digest


def test_duplicate_labels_of_one_species_merge_into_one_constituent():
    whole = constant_fill({"N2": 1.0})
    split = Mixture.of(
        Component(optmat_species("N2"), Constant(0.4 * AMAGAT), label="a"),
        Component(optmat_species("N2"), Constant(0.6 * AMAGAT), label="b"),
    )
    one = composition(whole, True, selection=SELECTION)
    two = composition(split, True, selection=SELECTION)
    assert two.assembly.composition == (("N2", 1.0),)
    assert two.labels == ("a", "b")
    assert two.species == "a"
    assert_array_equal(two.impulse(DELAYS, AMAGAT), one.impulse(DELAYS, AMAGAT))


def test_the_partners_optmat_curates_are_accepted():
    for fractions in (
        {"N2": 0.78, "O2": 0.21, "Ar": 0.01},
        {"N2": 0.5, "O2": 0.5},
        {"SF6": 0.2, "He": 0.8},
        {"O2": 0.5, "Ar": 0.5},
    ):
        response = composition(constant_fill(fractions), True, selection=SELECTION)
        assert isinstance(response, OptmatRamanResponse)
        assert len(response.labels) == len(fractions)


@pytest.mark.parametrize(
    ("fractions", "pair"),
    [
        ({"H2": 0.5, "He": 0.5}, "H2 by He"),
        ({"N2": 0.5, "He": 0.5}, "N2 by He"),
        ({"CH4": 0.5, "Ar": 0.5}, "CH4 by Ar"),
        ({"H2": 0.5, "N2": 0.5}, "H2 by N2"),
    ],
)
def test_a_partner_optmat_has_not_curated_is_refused_naming_the_pair(fractions, pair):
    with pytest.raises(MaterialError, match=f"broadening of .*{pair}.*D16"):
        composition(constant_fill(fractions), True, selection=SELECTION)


def test_a_mixture_of_atoms_has_no_response_and_is_not_refused(caplog):
    mixture = constant_fill({"He": 0.5, "Ar": 0.5})
    with caplog.at_level(logging.INFO, logger="solprop"):
        assert composition(mixture, True, selection=SELECTION) is None
    assert "no selected population has a molecular Raman response" in caplog.text
    with pytest.raises(MaterialError, match=r"Ar \(Ar\) has no molecular Raman.*atom"):
        composition(mixture, ["Ar"], selection=SELECTION, explicit=True)


def test_a_vacuum_fill_has_no_response(caplog):
    mixture = Mixture.of(Component(optmat_species("N2"), Constant(0.0)))
    with caplog.at_level(logging.INFO, logger="solprop"):
        assert composition(mixture, True, selection=SELECTION) is None
    assert "vacuum along the run" in caplog.text


def test_non_proportional_populations_are_refused():
    length = 1.0
    mixture = Mixture.of(
        Component(optmat_species("N2"), Constant(0.8 * AMAGAT)),
        Component(optmat_species("O2"), Linear(0.2 * AMAGAT, 0.02 * AMAGAT, length)),
    )
    with pytest.raises(MaterialError, match="not proportional"):
        composition(mixture, True, selection=SELECTION, length=length)


def test_a_gradient_fill_keeps_its_composition_within_the_tolerance():
    air = Mixture.air((10.0, 0.0), length=1.0, selection=SELECTION)
    with pytest.warns(MaterialRangeWarning, match="above 2.09 amagat"):
        response = composition(air, True, selection=SELECTION, length=1.0)
    # The partial-pressure rule moves argon's fraction by 1.6e-3 of itself between
    # 10 bar and vacuum: a measured spread, recorded, inside the tolerance.
    spread = response.assembly.composition_spread
    assert 1e-3 < spread < COMPOSITION_TOLERANCE


def test_a_population_optmat_does_not_resolve_is_refused_beside_a_molecule():
    mixture = Mixture.of(
        Component(optmat_species("N2"), Constant(AMAGAT)),
        Component(Species("argon-like", gamma("Ar")), Constant(0.1 * AMAGAT)),
    )
    with pytest.raises(
        MaterialError, match=r"argon-like \(argon-like\) is not a gas optmat resolves"
    ):
        composition(mixture, True, selection=SELECTION)


def test_dry_air_dispersion_entry_carries_no_raman_response():
    mixture = Mixture.of(
        Component(species(Gas.Air, selection=SELECTION), Constant(AMAGAT))
    )
    assert composition(mixture, True, selection=SELECTION) is None
    with pytest.raises(MaterialError, match="not a gas optmat curates"):
        composition(mixture, ["Air"], selection=SELECTION, explicit=True)


def test_labels_select_whose_lines_are_kept_and_partners_stay():
    air = Mixture.air(1.0, selection=SELECTION)
    nitrogen = composition(air, ["N2"], selection=SELECTION, explicit=True)
    species_of = {
        i.split("/")[0] for t in nitrogen.tensor_terms for i in t.model.line_ids
    }
    assert species_of == {"N2"}
    # Oxygen and argon still broaden nitrogen's lines.
    assert {name for name, _ in nitrogen.assembly.composition} == {"N2", "O2", "Ar"}
    assert nitrogen.labels == ("N2", "O2", "Ar")


def test_term_flags_select_bands_by_species():
    rotation = pure("N2", MolecularOptions(vibration=False), selection=SELECTION)
    vibration = pure("N2", MolecularOptions(rotation=False), selection=SELECTION)
    assert all("/rotation/" in t.model.line_ids[0] for t in rotation.tensor_terms)
    assert all("/rotation/" not in t.model.line_ids[0] for t in vibration.tensor_terms)
    assert rotation.assembly.groups + vibration.assembly.groups == 43
    with pytest.raises(MaterialError, match="CH4 has no rotational lines"):
        pure("CH4", MolecularOptions(rotation=True), selection=SELECTION)
    with pytest.raises(MaterialError, match="no Raman lines are left"):
        pure(
            "N2", MolecularOptions(rotation=False, vibration=False), selection=SELECTION
        )
    with pytest.raises(MaterialError, match="takes no truncation"):
        pure("N2", MolecularOptions(max_j=20), selection=SELECTION)


def test_per_label_options_and_their_conflicts():
    air = Mixture.air(1.0, selection=SELECTION)
    response = composition(
        air,
        {"N2": MolecularOptions(vibration=False), "O2": True},
        selection=SELECTION,
        explicit=True,
    )
    lines = [t.model.line_ids[0] for t in response.tensor_terms]
    assert not any(i.startswith("N2/v") for i in lines)
    assert any(i.startswith("O2/v") for i in lines)
    split = Mixture.of(
        Component(optmat_species("N2"), Constant(0.5 * AMAGAT), label="a"),
        Component(optmat_species("N2"), Constant(0.5 * AMAGAT), label="b"),
    )
    with pytest.raises(MaterialError, match="different Raman terms"):
        composition(
            split,
            {"a": MolecularOptions(vibration=False), "b": True},
            selection=SELECTION,
        )
    with pytest.raises(MaterialError, match="name temperatures"):
        composition(
            air,
            {
                "N2": MolecularOptions(temperature=290.0),
                "O2": MolecularOptions(temperature=300.0),
            },
            selection=SELECTION,
        )


def test_an_explicit_temperature_sets_the_response():
    response = pure("O2", MolecularOptions(temperature=400.0), selection=SELECTION)
    assert response.assembly.temperature == 400.0
    assert response.source.temperature == 400.0
    mixture = constant_fill({"N2": 1.0})
    response = composition(
        mixture, {"N2": MolecularOptions(temperature=296.0)}, selection=SELECTION
    )
    assert response.assembly.temperature == 296.0


def test_a_temperature_path_takes_the_summed_density_reference_and_warns():
    mixture = Mixture.fill(
        {"N2": 0.8, "O2": 0.2},
        1.0,
        temperature=Linear(294.0, 298.0, 1.0),
        selection=SELECTION,
    )
    with pytest.warns(RamanTemperatureWarning, match="linewidth temperature laws"):
        response = composition(mixture, True, selection=SELECTION, length=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RamanTemperatureWarning)
        expected = raman_temperature(mixture, ("N2", "O2"), length=1.0, widths=True)
    assert response.assembly.temperature == expected
    assert 294.0 < expected < 298.0


def test_a_supplied_dephasing_becomes_optmat_width_law():
    law = Dephasing(Linewidth(constant=1e9, density=5e8))
    mixture = Mixture.of(
        Component(optmat_species("N2"), Constant(AMAGAT), dephasing=law)
    )
    response = composition(mixture, True, selection=SELECTION)
    record = response.material_records[0]
    assert record.status == "user"
    assert record.modified
    for group in response.source.groups():
        rate = GroupDephasing.of(group, response.source).decay_rate(2 * AMAGAT)
        assert rate == pytest.approx(math.pi * (1e9 + 5e8 * 2.0), rel=1e-14)
    dicke = Mixture.of(
        Component(
            optmat_species("N2"),
            Constant(AMAGAT),
            dephasing=Dephasing(Linewidth(inverse_density=1e7)),
        )
    )
    with pytest.raises(MaterialError, match="Dicke term"):
        composition(dicke, True, selection=SELECTION)
    split = Mixture.of(
        Component(
            optmat_species("N2"), Constant(0.5 * AMAGAT), label="a", dephasing=law
        ),
        Component(
            optmat_species("N2"),
            Constant(0.5 * AMAGAT),
            label="b",
            dephasing=Dephasing(Linewidth(constant=2e9)),
        ),
    )
    with pytest.raises(MaterialError, match="different Raman linewidths"):
        composition(split, True, selection=SELECTION)


# ------------------------------------------------------------------ H2's shift


def test_hydrogen_at_constant_density_is_exact():
    density = 2.0 * AMAGAT
    response = pure("H2", selection=SELECTION, density=density)
    source = response.source
    assert response.assembly.shift == RamanShift("constant-density", density, 0.0)
    expected = source.response(DELAYS, density)
    bound = summation_bound(source, len(source))
    assert_allclose(response.impulse(DELAYS, density), expected, rtol=0, atol=bound)
    filled = composition(constant_fill({"H2": 1.0}, 2.0), True, selection=SELECTION)
    assert filled.assembly.shift.policy == "constant-density"
    assert filled.assembly.shift.density == pytest.approx(density, rel=1e-15)


def test_hydrogen_without_a_density_is_refused():
    with pytest.raises(MaterialError, match="pass density= for a constant fill"):
        pure("H2", selection=SELECTION)


def hydrogen_gradient(start=3.0, end=1.0, length=1.0):
    """A hydrogen fill whose density falls linearly, in amagat."""
    return Mixture.of(
        Component(optmat_species("H2"), Linear(start * AMAGAT, end * AMAGAT, length))
    )


def test_hydrogen_on_a_gradient_is_refused_without_the_named_approximation():
    with pytest.raises(MaterialError, match="raman_shift='reference-density'"):
        composition(hydrogen_gradient(), True, selection=SELECTION, length=1.0)


def test_the_reference_density_approximation_records_a_bound_that_holds():
    start, end, length = 3.0, 1.0, 1.0
    mixture = hydrogen_gradient(start, end, length)
    response = composition(mixture, True, selection=SHIFTED, length=length)
    shift = response.assembly.shift
    low, high = start * AMAGAT, end * AMAGAT
    # The density-weighted mean of a linear path: int N^2 / int N.
    reference = (low**2 + low * high + high**2) / 3.0 / ((low + high) / 2.0)
    assert shift.policy == "reference-density"
    assert shift.density == pytest.approx(reference, rel=1e-9)
    source = response.source
    sigma = max(abs(g.shift) for g in source.groups() if g.shifted)
    deviation = max(abs(low - reference), abs(high - reference))
    assert shift.frequency_error == pytest.approx(sigma * deviation, rel=1e-9)
    # Against direct evaluation along the path: each line's phase is wrong by at most
    # frequency_error * t, and |sin x - sin y| <= |x - y|, so the response is wrong by
    # at most sum |A_k| exp(-Gamma_k t) frequency_error t.
    t = np.linspace(0.0, 5e-12, 2001)
    for amagat in np.linspace(start, end, 7):
        density = amagat * AMAGAT
        direct = source.response(t, density)
        approximate = response.impulse(t, density)
        rates = source.decay_rates(np.array([density]))[:, 0]
        envelope = np.abs(source.amplitude)[:, None] * np.exp(-rates[:, None] * t)
        bound = np.sum(envelope, axis=0) * shift.frequency_error * t
        assert np.all(np.abs(direct - approximate) <= bound + 1e-12 * np.max(envelope))
    assert shift.phase_bound(5e-12) == pytest.approx(shift.frequency_error * 5e-12)
    assert shift.to_dict()["policy"] == "reference-density"


def test_an_unbounded_run_has_no_reference_density():
    path = Linear(2 * AMAGAT, AMAGAT, 1.0)
    z = np.array([0.0, 1.0])
    unbounded = _Path(z, path(z)[None, :], path, (0.0, math.inf))
    assert _reference_density(unbounded) is None
    bounded = _Path(z, path(z)[None, :], path, (0.0, 1.0))
    assert _reference_density(bounded) == pytest.approx(
        (4 + 2 + 1) / 3 / 1.5 * AMAGAT, rel=1e-9
    )


# ------------------------------------------------------------------ range and vacuum


def test_the_density_range_is_checked_once_on_the_non_zero_densities():
    with (
        bridge.material_context() as log,
        pytest.warns(MaterialRangeWarning, match="above 2.09 amagat") as caught,
    ):
        response = composition(
            constant_fill({"N2": 1.0}, 3.0), True, selection=SELECTION
        )
        composition(constant_fill({"N2": 1.0}, 3.0), True, selection=SELECTION)
    assert len([w for w in caught if "above" in str(w.message)]) == 1
    assert response.assembly.densities == (3.0 * AMAGAT, 3.0 * AMAGAT)
    digest = response.material_records[0].digest
    material, name, parts = log.ranges[digest]
    assert material == "N2"
    assert name == response.material_records[0].name
    assert any("3 to 3 amagat" in part for part in parts)


def test_hydrogen_at_one_bar_is_below_its_dicke_cutoff():
    with pytest.warns(MaterialRangeWarning, match=r"below 0.94 amagat"):
        composition(
            Mixture.fill({"H2": 1.0}, 1.0, selection=SELECTION),
            True,
            selection=SELECTION,
        )


def test_the_temperature_range_is_checked_and_recorded():
    with (
        bridge.material_context() as log,
        pytest.warns(MaterialRangeWarning, match="temperature range") as caught,
    ):
        response = pure("N2", MolecularOptions(temperature=280.0), selection=SELECTION)
        pure("N2", MolecularOptions(temperature=280.0), selection=SELECTION)
    assert len(caught) == 1
    _, _, parts = log.ranges[response.material_records[0].digest]
    assert any(part.startswith("280 K") for part in parts)


def test_a_gradient_to_vacuum_keeps_finite_decay_paths_and_declares_no_zero():
    length = 1.0
    grid = FieldGrid.from_limits(length, 800e-9, (200e-9, 4e-6), 1e-12)
    nitrogen = Mixture.of(Component(optmat_species("N2"), Linear(AMAGAT, 0.0, length)))
    with pytest.warns(MaterialRangeWarning, match="below 0.054 amagat"):
        response = composition(nitrogen, True, selection=SELECTION, length=length)
    low, high = response.assembly.densities
    # The path reaches vacuum at z = L: every density down to zero is declared.
    assert low == 0.0
    assert high == pytest.approx(AMAGAT)
    density = nitrogen.components[0].density
    for path in decay_rate_profiles(response, density, grid):
        assert path(np.array([length]))[0] == 0.0
    hydrogen = Mixture.of(
        Component(optmat_species("H2"), Linear(2 * AMAGAT, 0.0, length))
    )
    with pytest.warns(MaterialRangeWarning, match="below 0.94 amagat"):
        response = composition(hydrogen, True, selection=SHIFTED, length=length)
    cap = MAX_DECAY_EXPONENT / grid.dt_fine
    paths = decay_rate_profiles(response, hydrogen.components[0].density, grid)
    for term, path in zip(response.tensor_terms, paths, strict=True):
        at_vacuum = float(path(np.array([length]))[0])
        assert at_vacuum == term.model.dephasing.decay_rate(0.0)
        assert 0.0 < at_vacuum < cap


# ------------------------------------------------------------------ selections


def test_n2o_is_refused_unless_provisional_and_then_warns_once():
    with pytest.raises(MaterialError, match=r"provisional.*raman='provisional'"):
        pure("N2O", selection=SELECTION)
    provisional = MaterialSelection(carrier_wavelength=800e-9, raman="provisional")
    with (
        bridge.material_context(),
        pytest.warns(MaterialStatusWarning, match="provisional") as caught,
    ):
        response = pure("N2O", selection=provisional)
        pure("N2O", selection=provisional)
    assert len(caught) == 1
    assert response.material_records[0].status == "provisional"
    # A validated gas under the provisional selection is the validated response.
    assert pure("N2", selection=provisional).material_records[0].status == ("validated")


def test_a_supplied_optmat_response_is_used_as_it_is():
    curated = raman_response("N2", ROOM_TEMPERATURE, wavelength=800e-9)
    narrowed = curated.with_parameters(width_scale=0.8)
    selection = MaterialSelection(carrier_wavelength=800e-9, raman=narrowed)
    response = pure("N2", selection=selection)
    assert response.source is narrowed
    record = response.material_records[0]
    assert record.status == "user"
    assert record.base == curated.identity.digest
    for wrong in (
        raman_response("N2", 300.0, wavelength=800e-9),
        raman_response("N2", ROOM_TEMPERATURE, wavelength=1030e-9),
        raman_response("N2", ROOM_TEMPERATURE),
        raman_response("O2", ROOM_TEMPERATURE, wavelength=800e-9),
    ):
        with pytest.raises(MaterialError, match="the run needs"):
            pure(
                "N2",
                selection=MaterialSelection(carrier_wavelength=800e-9, raman=wrong),
            )
    with pytest.raises(MaterialError, match="not a gas Raman response"):
        pure(
            "N2",
            selection=MaterialSelection(
                carrier_wavelength=800e-9, raman=default_model("N2")
            ),
        )


@pytest.mark.parametrize("name", ["N2", "tensor", "default", "lin_2006"])
def test_a_raman_name_other_than_the_gas_policies_is_refused(name):
    # A silica bundle's name means a fibre's bundle, never a gas response.
    selection = MaterialSelection(carrier_wavelength=8e-7, raman=name)
    with pytest.raises(MaterialError, match=r"silica bundle name .* GNLSE only"):
        pure("N2", selection=selection)


def test_a_silica_bundle_object_is_refused_on_a_gas_run():
    bundle = MaterialSelection(carrier_wavelength=8e-7, raman=silica_bundle("default"))
    with pytest.raises(MaterialError, match=r"silica_bundle model.*GNLSE only"):
        pure("N2", selection=bundle)


def test_the_carrier_is_the_selection_s_or_the_grid_reference():
    explicit = pure("N2", selection=SELECTION, wavelength=1030e-9)
    assert explicit.assembly.wavelength == 800e-9
    implicit = pure("N2", selection=MaterialSelection(), wavelength=1030e-9)
    assert implicit.assembly.wavelength == 1030e-9
    assert implicit.source.wavelength == 1030e-9
    with pytest.raises(MaterialError, match="needs the carrier"):
        pure("N2", selection=MaterialSelection())


def test_a_pure_gas_refuses_what_is_not_a_molecule_or_a_bad_density():
    with pytest.raises(MaterialError, match="it is an atom"):
        pure("Ar", selection=SELECTION)
    with pytest.raises(MaterialError, match="not a gas optmat curates"):
        pure("Air", selection=SELECTION)
    with pytest.raises(MaterialError, match="positive and finite"):
        pure("N2", selection=SELECTION, density=-1.0)


def test_response_for_takes_the_optmat_path_under_a_selection():
    response = response_for(Gas.N2, selection=SELECTION, density=AMAGAT)
    assert isinstance(response, OptmatRamanResponse)
    assert response.assembly is not None
    assert response.assembly.densities == (AMAGAT, AMAGAT)
    assert response.labels == ()
    assert response.traits.polarisation == "tensor"
    # No selection is MaterialSelection(), whose response needs the carrier (since
    # SEP 3's Phase 4b there is no registry route a bare density could select).
    with pytest.raises(MaterialError, match="needs the carrier"):
        response_for(Gas.N2, density=AMAGAT)
    with pytest.raises(MaterialError, match="is absolute; a fraction"):
        response_for(Gas.N2, selection=SELECTION, fraction=0.5)
    # Silica keeps its own path: its bundle is another phase's.
    assert not isinstance(
        response_for("SiO2", selection=SELECTION), OptmatRamanResponse
    )


def test_responses_for_builds_one_composition_response_in_label_order():
    air = Mixture.air(1.0, selection=SELECTION)
    responses = responses_for(air, raman=True, selection=SELECTION, wavelength=800e-9)
    raman = [r for r in responses if isinstance(r, RamanResponse)]
    assert len(raman) == 1
    assert isinstance(raman[0], OptmatRamanResponse)
    kinds = [type(r).__name__ for r in responses]
    assert kinds.index("OptmatRamanResponse") == 1
    single = responses_for(
        Mixture.fill({"N2": 1.0}, 1.0, selection=SELECTION),
        raman=True,
        selection=SELECTION,
        wavelength=800e-9,
    )
    (one,) = [r for r in single if isinstance(r, OptmatRamanResponse)]
    assert one.species is None
    assert one.labels == ("N2",)
    off = responses_for(air, raman=False, selection=SELECTION, wavelength=800e-9)
    assert not any(isinstance(r, RamanResponse) for r in off)
    atoms = responses_for(
        constant_fill({"Ar": 1.0}), raman=True, selection=SELECTION, wavelength=800e-9
    )
    assert not any(isinstance(r, RamanResponse) for r in atoms)


def test_a_built_in_fill_is_optmats_and_a_supplied_species_keeps_its_own():
    """Since SEP 3's Phase 4b a built-in fill's Raman response is always optmat's,
    the former registry family is refused by name, and a caller's species is
    answered from its own constants, per component."""
    air = Mixture.air(1.0, selection=SELECTION)
    built_in = responses_for(air, raman=True, wavelength=800e-9, selection=SELECTION)
    assert sum(isinstance(r, OptmatRamanResponse) for r in built_in) == 1
    registry = dataclasses.replace(SELECTION, raman="registry")
    with pytest.raises(MaterialError, match="'registry' names no gas Raman response"):
        responses_for(air, raman=True, wavelength=800e-9, selection=registry)
    with pytest.raises(MaterialError, match="'registry' names no gas Raman response"):
        response_for(Gas.N2, selection=registry)
    # solprop's former N2 constants (dc4ca0d), as a caller's species.
    nitrogen = supplied_species(FORMER_N2)
    own = responses_for(Mixture.fill({nitrogen: 1.0}, 1.0), raman=True, kerr=False)
    assert not any(isinstance(r, OptmatRamanResponse) for r in own)
    assert sum(isinstance(r, RamanResponse) for r in own) == 1
    with pytest.raises(MaterialError, match="caller's species"):
        response_for(nitrogen, selection=SELECTION)


# ------------------------------------------------------------------ objects


def test_an_optmat_response_round_trips_through_replace_and_serialise():
    response = pure("N2", selection=SELECTION)
    labelled = dataclasses.replace(response, species="N2", labels=("N2",))
    assert labelled.source is response.source
    assert labelled.assembly == response.assembly
    record: Any = serialise(response)
    assert "source" not in record
    assert record["assembly"]["groups"] == 43
    term = record["tensor_terms"][0]["model"]
    assert "group" not in term["dephasing"]


def test_an_optmat_response_refuses_legacy_models_a_fraction_or_foreign_terms():
    response = pure("N2", selection=SELECTION)
    with pytest.raises(MaterialError, match="neither legacy models nor a fraction"):
        OptmatRamanResponse(fraction=0.5, tensor_terms=response.tensor_terms)
    with pytest.raises(MaterialError, match="neither legacy models nor a fraction"):
        OptmatRamanResponse(models=(response.tensor_terms[0].model,))
    # A caller's molecular terms (solprop's former N2 constants, dc4ca0d).
    foreign = former_response(FORMER_N2).tensor_terms
    with pytest.raises(MaterialError, match="optmat group rows"):
        OptmatRamanResponse(tensor_terms=foreign)
    bare = OptmatRamanResponse(tensor_terms=response.tensor_terms)
    assert bare.material_records == ()


def test_a_group_row_checks_its_lines():
    dephasing = response_row().dephasing
    with pytest.raises(MaterialError, match="one frequency, amplitude and identifier"):
        RamanGroupLines(
            np.array([1e13]), np.array([1.0, 2.0]), dephasing, ("x",), "isotropic"
        )
    with pytest.raises(MaterialError, match="one frequency, amplitude and identifier"):
        RamanGroupLines(np.array([]), np.array([]), dephasing, (), "isotropic")
    with pytest.raises(MaterialError, match="positive and finite"):
        RamanGroupLines(
            np.array([-1.0]), np.array([1.0]), dephasing, ("x",), "isotropic"
        )
    with pytest.raises(MaterialError, match="amplitudes must be finite"):
        RamanGroupLines(
            np.array([1e13]), np.array([math.nan]), dephasing, ("x",), "isotropic"
        )
    row = RamanGroupLines(
        np.array([1e13]), np.array([2.0]), dephasing, ("x",), "isotropic"
    )
    assert not row.frequencies.flags.writeable
    assert RamanTensorTerm(row, "isotropic").model is row


def response_row() -> Any:
    """The first row of nitrogen's response."""
    return pure("N2", selection=SELECTION).tensor_terms[0].model


def test_the_manifest_round_trips_the_raman_record():
    manifest = MaterialManifest(
        "0.1.1",
        SELECTION.to_dict(),
        policies={},
        raman=({"labels": ["N2"], "groups": 43},),
    )
    stored: Any = manifest.to_dict()
    assert stored["raman"] == [{"labels": ["N2"], "groups": 43}]
    assert "raman" not in cast("dict[str, object]", manifest.hashed())
    assert MaterialManifest.from_dict(stored) == manifest


def test_the_record_of_a_state_outside_its_range_reaches_every_open_log():
    response = pure("N2", selection=SELECTION)
    entry = response.material_records[0]
    bridge.record_outside(response.source, "N2", ("nowhere",))
    with bridge.material_context() as outer, bridge.material_context(bridge.open_log()):
        bridge.record_outside(response.source, "N2", ("3 bar",))
    assert outer.ranges[entry.digest][2] == {"3 bar"}


def test_a_molecule_without_curated_raman_lines_is_refused_with_optmat_reason():
    with pytest.raises(MaterialError, match="optmat refuses the Raman response"):
        pure("CO", selection=SELECTION)


def test_other_warnings_of_optmat_pass_through_unchanged():
    response = pure("N2", selection=SELECTION).source
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("not a range warning", UserWarning, stacklevel=1)
    with pytest.warns(UserWarning, match="not a range warning"):
        _reemit(caught, response, kind="density", where="N2", part="1 amagat")


def test_the_assembly_is_logged_at_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="solprop.nonlinear.optmat_raman"):
        pure("N2", selection=SELECTION)
    assert "43 damping groups as 72 terms" in caplog.text


@pytest.mark.parametrize(
    ("fill", "limit"),
    [
        (
            lambda: Mixture.fill(
                {"H2": 1.0}, (10.0, 0.0), length=1.0, floor=0.0, selection=SHIFTED
            ),
            "below 0.94 amagat",
        ),
        (
            lambda: Mixture.air(
                (10.0, 0.0), length=1.0, floor=0.0, selection=SELECTION
            ),
            "below 0.054 amagat",
        ),
    ],
)
def test_a_gradient_to_vacuum_declares_its_tail_down_to_zero(fill, limit):
    # The last positive probe is above 1 amagat; the path between it and the vacuum
    # end crosses the Dicke regime (H2) or the lower limit (N2), and must warn.
    with (
        bridge.material_context() as log,
        pytest.warns(MaterialRangeWarning, match=limit),
    ):
        response = composition(fill(), True, selection=SHIFTED, length=1.0)
    low, high = response.assembly.densities
    assert low == 0.0
    assert high > 9.0 * AMAGAT
    _, _, parts = log.ranges[response.material_records[0].digest]
    assert any(part.startswith("vacuum to ") for part in parts)
