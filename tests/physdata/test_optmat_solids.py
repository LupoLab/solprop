# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Solid indices from optmat: every solid and ray, derivatives, ranges, refusals."""

import math
import re
import warnings
from typing import Any, cast

import numpy as np
import pytest
from optmat.solid.models.formulas import FormulaModel
from optmat.solid.models.registry import default_model, model
from optmat.solid.models.tabulated import TabulatedModel

from solprop.constants import C
from solprop.driver.material_output import material_manifest
from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.modes.material import IndexMaterial
from solprop.modes.stepindex import StepIndexMode
from solprop.physdata import (
    CompositeGlass,
    OptmatGlass,
    SellmeierGlass,
    _optmat,
    aluminium,
    barium_fluoride,
    bk7,
    calcium_fluoride,
    fused_silica,
    glass,
    magnesium_fluoride,
    sapphire,
    silver,
)
from solprop.physdata.glasses import TRANSPARENT_ABSORPTION_BOUND, lossless_index
from solprop.physdata.selection import MaterialLog

# Every built-in solid, by its factory, with optmat's material id and ray.
SOLIDS = {
    "SiO2": (lambda: fused_silica(model="recommended"), "SiO2", None),
    "CaF2": (lambda: calcium_fluoride(model="recommended"), "CaF2", None),
    "MgF2-o": (lambda: magnesium_fluoride(axis="o", model="recommended"), "MgF2", "o"),
    "MgF2-e": (lambda: magnesium_fluoride(axis="e", model="recommended"), "MgF2", "e"),
    "BaF2": (lambda: barium_fluoride(model="recommended"), "BaF2", None),
    "Al2O3-o": (lambda: sapphire(axis="o", model="recommended"), "Al2O3", "o"),
    "Al2O3-e": (lambda: sapphire(axis="e", model="recommended"), "Al2O3", "e"),
    "N-BK7": (lambda: bk7(model="recommended"), "N-BK7", None),
    "Ag": (lambda: silver(model="recommended"), "Ag", None),
    "Al": (lambda: aluminium(model="recommended"), "Al", None),
}
WAVELENGTHS = np.array([110e-9, 120e-9, 300e-9, 800e-9, 1000e-9, 5e-6])


def optmat_glass(index):
    """Return a factory's index, checked to be an optmat adapter."""
    assert isinstance(index, OptmatGlass)
    return index


def omega_of(wavelength):
    return 2.0 * math.pi * C / np.asarray(wavelength, dtype=float)


def inside(reference):
    """Return the sample wavelengths inside a model's qualified intervals."""
    low, high = reference.wavelength_range
    return WAVELENGTHS[(low <= WAVELENGTHS) & (high >= WAVELENGTHS)]


@pytest.mark.parametrize("key", list(SOLIDS))
def test_index_and_derivative_are_optmats_at_the_sample_wavelengths(key):
    factory, material, axis = SOLIDS[key]
    adapted = factory()
    reference = default_model(material, axis=axis)
    assert isinstance(adapted, OptmatGlass)
    assert adapted.source == reference.name
    assert adapted.axis == (axis or "")
    samples = inside(reference)
    assert samples.size
    omega = omega_of(samples)
    # The omega-to-wavelength round trip costs two roundings; optmat's own index is
    # complex and is not conjugated (k >= 0 is loss in both).
    np.testing.assert_allclose(
        adapted.index(omega), reference.refractive_index(samples), rtol=1e-15
    )
    np.testing.assert_allclose(
        adapted.dindex_domega(omega), reference.dindex_domega(samples), rtol=1e-12
    )
    assert np.all(np.imag(adapted.index(omega)) >= 0)
    np.testing.assert_allclose(
        adapted.permittivity(omega), reference.permittivity(samples), rtol=1e-14
    )
    assert adapted.material_records[0].digest == reference.identity.digest


@pytest.mark.parametrize("key", ["SiO2", "CaF2", "Al2O3-e", "Ag", "Al"])
def test_the_derivative_agrees_with_independent_finite_differences(key):
    # The fourth-order central difference at three relative steps; every one must
    # agree, real and imaginary parts alike (complex silica and the metal tables).
    adapted = SOLIDS[key][0]()
    omega = omega_of(np.array([600e-9, 900e-9]))
    analytic = adapted.dindex_domega(omega)
    for relative in (1e-3, 1e-4, 3e-5):
        step = relative * omega
        wide = (adapted.index(omega + step) - adapted.index(omega - step)) / (2 * step)
        narrow = (
            adapted.index(omega + step / 2) - adapted.index(omega - step / 2)
        ) / step
        estimate = (4 * narrow - wide) / 3
        # Truncation falls as (relative)^4 and roundoff grows as 1e-16 n / (relative
        # omega |dn/domega|): measured, the balance floors near 1e-9 to 1e-8 of
        # |dn/domega| over these steps (a metal table's PCHIP derivative included),
        # so 5e-8 bounds every step and catches a derivative wrong in any digit
        # a difference can resolve.
        assert np.all(np.abs(estimate - analytic) <= 5e-8 * np.abs(analytic))


def test_silica_absorbs_and_its_extinction_is_kept():
    silica = fused_silica(model="recommended")
    index = silica.index(omega_of(np.array([110e-9, 120e-9, 800e-9, 10e-6])))
    assert np.all(np.imag(index) > 0)
    # Its vacuum-ultraviolet edge and mid-infrared band are strong, as they must be
    # for a capillary wall's loss at the edges of a supercontinuum.
    assert np.imag(index)[0] > 0.1 and np.imag(index)[-1] > 1e-3


@pytest.mark.parametrize("key", ["CaF2", "MgF2-o", "BaF2", "Al2O3-o", "N-BK7"])
def test_a_band_on_the_range_edges_is_quiet_and_beyond_them_warns(key):
    factory, material, axis = SOLIDS[key]
    adapted = factory()
    low, high = default_model(material, axis=axis).wavelength_range
    # A hair inside each end: the omega round trip moves a wavelength by ulps.
    adapted.index(omega_of(np.array([low * (1 + 1e-12), high * (1 - 1e-12)])))
    with pytest.warns(MaterialRangeWarning, match="below its shortest"):
        adapted.index(omega_of(np.array([low * 0.9])))
    with pytest.warns(MaterialRangeWarning, match="beyond its longest"):
        fresh = factory()
        fresh.index(omega_of(np.array([high * 1.1])))


@pytest.mark.parametrize("key", ["Ag", "Al"])
def test_a_metal_table_holds_its_end_values_outside_with_a_warning(key):
    factory, material, _ = SOLIDS[key]
    adapted = factory()
    low, high = default_model(material).wavelength_range
    edge = adapted.index(omega_of(np.array([high])))
    with pytest.warns(MaterialRangeWarning, match="holds its end values"):
        beyond = adapted.index(omega_of(np.array([high * 3])))
    np.testing.assert_allclose(beyond, edge, rtol=1e-12)
    with pytest.warns(MaterialRangeWarning, match="holds its end values"):
        below = adapted.index(omega_of(np.array([low / 3])))
    np.testing.assert_allclose(
        below, adapted.index(omega_of(np.array([low * (1 + 1e-12)]))), rtol=1e-9
    )


@pytest.mark.parametrize("factory", [magnesium_fluoride, sapphire])
def test_a_uniaxial_crystal_names_its_ray_under_optmat(factory):
    with pytest.raises(MaterialError, match=r"uniaxial.*axis='o'"):
        factory(model="recommended")
    with pytest.raises(MaterialError, match=re.escape("axis is 'o' or 'e'")):
        factory(axis="x", model="recommended")
    # A named model names its ray; asking for the other is refused.
    ordinary = factory(axis="o", model="recommended")
    assert factory(model=ordinary.source).axis == "o"
    with pytest.raises(MaterialError, match=r"describes .* o ray"):
        factory(axis="e", model=ordinary.source)


def test_glass_routes_names_axes_and_models():
    assert glass("silica", model="recommended").source == "fit/SiO2"
    assert glass("BK7", model="recommended").source == "schott_2023/sellmeier"
    assert optmat_glass(glass("MgF2", model="recommended", axis="e")).axis == "e"
    assert glass("sapphire", axis="o").source == sapphire(axis="o").source
    assert glass("CaF2").source == "daimon_2002/sellmeier_20C"
    with pytest.raises(MaterialError, match="isotropic and takes no axis"):
        glass("CaF2", axis="o")
    with pytest.raises(MaterialError, match="isotropic"):
        glass("BaF2", model="recommended", axis="e")


def test_a_named_or_supplied_model_is_never_matched_to_another_material():
    assert fused_silica(model="malitson_1965/sellmeier").source == (
        "malitson_1965/sellmeier"
    )
    with pytest.raises(MaterialError, match="describes CaF2, not SiO2"):
        fused_silica(model="daimon_2002/sellmeier_20C")
    with pytest.raises(MaterialError, match="no solid model 'fit/Zz'"):
        bk7(model="fit/Zz")
    supplied = model("table/Ag")
    assert optmat_glass(silver(model=supplied)).model is supplied
    with pytest.raises(MaterialError, match="describes Ag, not Al"):
        aluminium(model=supplied)


def test_quick_experiments_keep_their_identity_and_status():
    curated: Any = default_model("CaF2")
    terms = dict(curated.terms)
    terms["B"] = tuple(b * 1.001 for b in terms["B"])
    modified = optmat_glass(
        calcium_fluoride(model=curated.with_parameters(terms=terms))
    )
    (record,) = modified.material_records
    assert record.status == "user" and record.base == curated.identity.digest
    assert record.modified[0][0] == "terms"
    omega = omega_of(np.array([800e-9]))
    assert np.real(modified.index(omega)) > np.real(
        calcium_fluoride(model="recommended").index(omega)
    )
    # A caller's own Sellmeier formula or table, adapted as it is.
    formula = FormulaModel.sellmeier(
        "SiO2",
        {"B": (0.6961663, 0.4079426, 0.8974794), "C": (0.004679, 0.013512, 97.934)},
        (210e-9, 3.71e-6),
        citation="Malitson (1965)",
    )
    mine = OptmatGlass.from_model(formula)
    assert mine.material_records[0].status == "user"
    assert float(np.real(mine.index(omega_of(1e-6)))) == pytest.approx(
        1.45042, abs=2e-5
    )
    table = TabulatedModel.from_arrays(
        "my metal",
        np.array([500e-9, 600e-9, 700e-9]),
        np.array([0.9, 0.25, 0.16]),
        np.array([1.9, 3.0, 3.9]),
    )
    assert OptmatGlass.from_model(table).name == "my metal"
    with pytest.raises(MaterialError, match="describes my metal, not Ag"):
        silver(model=table)
    # solprop's own supplied types keep working, and are never matched by name.
    supplied = SellmeierGlass(
        name="SiO2",
        source="mine",
        wavelength_range=(0.2e-6, 3e-6),
        citation="",
        poles=((0.6961663, 0.0684043),),
    )
    assert IndexMaterial(supplied).index is supplied


def test_the_representation_carries_the_digest():
    silica = optmat_glass(fused_silica(model="recommended"))
    digest = silica.material_records[0].digest
    assert re.search(f"digest={digest}", repr(silica))
    # A modal guide's identity records the material by its representation.
    assert digest in str(IndexMaterial(silica).identity["index"])


def test_a_lossless_route_takes_an_absorbing_optmat_index_as_transparent():
    # SEP 3, Q3: never refused for absorption; the route holds the transparent
    # counterpart, one per adapter, which every lossless route of it shares.
    silica = optmat_glass(fused_silica(model="recommended"))
    mode = StepIndexMode(1.25e-6, core=silica, profile_wavelength=835e-9)
    counterpart = lossless_index(silica)
    assert mode.core is counterpart and mode._core is counterpart
    assert counterpart.transparent and not silica.transparent
    assert counterpart.absorption_bound == TRANSPARENT_ABSORPTION_BOUND
    assert IndexMaterial(silica).index is counterpart
    other = StepIndexMode(
        1.25e-6, core=silica, kind="TE", n=0, profile_wavelength=835e-9
    )
    assert other.core is counterpart
    neff = mode.neff(omega_of(np.array([700e-9, 900e-9])))
    assert np.all(np.imag(neff) == 0)
    assert 0 < counterpart.largest_dropped_extinction < 1e-100
    # The real index is optmat's own, selected rather than recomputed: exact.
    np.testing.assert_array_equal(
        counterpart.index(omega_of(np.array([800e-9]))),
        np.real(silica.index(omega_of(np.array([800e-9])))),
    )
    # A cladding is taken the same way, and the counterpart of a counterpart is
    # itself.
    clad = StepIndexMode(1.25e-6, core=1.6, cladding=silica, profile_wavelength=835e-9)
    assert clad.cladding is counterpart and lossless_index(counterpart) is counterpart


def test_lossless_index_returns_anything_else_as_it_is():
    # A caller's own complex index: Malitson's real part (J. Opt. Soc. Am. 55, 1205)
    # with a constant absorption, test data.
    legacy = CompositeGlass(
        name="SiO2",
        source="mine",
        wavelength_range=(0.21e-6, 3.71e-6),
        citation="test data",
        real_part=SellmeierGlass(
            name="SiO2",
            source="malitson",
            wavelength_range=(0.21e-6, 3.71e-6),
            citation="",
            poles=(
                (0.6961663, 0.0684043),
                (0.4079426, 0.1162414),
                (0.8974794, 9.896161),
            ),
        ),
        absorption=SellmeierGlass(
            name="SiO2",
            source="lossy",
            wavelength_range=(0.21e-6, 3.71e-6),
            citation="",
            poles=((-1.000001, 0.0),),  # n^2 = -1e-6: k = 1e-3
        ),
    )
    assert lossless_index(legacy) is legacy
    assert lossless_index(1.45) == 1.45
    explicit = optmat_glass(fused_silica(model="recommended")).as_transparent(1e4)
    assert lossless_index(explicit) is explicit
    # solprop's own complex index keeps the step-index route's documented real part.
    mode = StepIndexMode(1.25e-6, core=legacy, profile_wavelength=835e-9)
    assert mode.core is legacy and mode._core is legacy


def test_the_dropped_absorption_warns_once_per_run():
    counterpart = lossless_index(optmat_glass(fused_silica(model="recommended")))
    edge = omega_of(np.array([120e-9]))
    for _ in range(2):
        # Each context is one run: one warning however often the band is evaluated,
        # naming the material, the absorption it drops and where.
        with _optmat.material_context(), pytest.warns(MaterialRangeWarning) as seen:
            counterpart.index(edge)
            counterpart.index(omega_of(np.array([125e-9, 120e-9])))
        assert len(seen) == 1
        text = str(seen[0].message)
        assert "SiO2 (fit/SiO2) is taken as transparent" in text
        assert "at 120 nm" in text and "beyond the transparent" in text
    # Outside every run the adapter has said it already.
    counterpart.index(edge)
    assert counterpart.transparency["bound_exceeded"]


def test_the_transparent_approximation_is_explicit_and_records_what_it_drops():
    transparent = optmat_glass(fused_silica(model="recommended")).as_transparent()
    assert transparent.transparent
    assert transparent.largest_dropped_extinction == 0.0
    mode = StepIndexMode(1.25e-6, core=transparent, profile_wavelength=835e-9)
    omega = omega_of(np.array([700e-9, 900e-9]))
    assert np.all(np.imag(mode.neff(omega)) == 0)
    dropped = transparent.largest_dropped_extinction
    assert 0 < dropped < 1e-100
    # The real parts are optmat's own, selected rather than recomputed, so exact;
    # the derivative is d(Re n)/domega.
    full = fused_silica(model="recommended")
    np.testing.assert_array_equal(transparent.index(omega), np.real(full.index(omega)))
    np.testing.assert_array_equal(
        transparent.dindex_domega(omega), np.real(full.dindex_domega(omega))
    )
    assert "transparent=True, absorption_bound=100" in repr(transparent)
    record = transparent.transparency
    assert record["absorption_bound"] == TRANSPARENT_ABSORPTION_BOUND
    assert not record["bound_exceeded"]


def test_a_runs_dropped_loss_is_its_own_not_the_adapters():
    # Q28: the transparent counterpart is shared by every lossless route of its
    # source, so what it drops over its life spans runs; each run's log records
    # only what that run dropped, and its manifest takes it from there.
    counterpart = lossless_index(optmat_glass(fused_silica(model="recommended")))
    edge, window = MaterialLog(), MaterialLog()
    with _optmat.material_context(edge), pytest.warns(MaterialRangeWarning):
        counterpart.index(omega_of(np.array([800e-9, 120e-9])))
    with _optmat.material_context(window), warnings.catch_warnings():
        warnings.simplefilter("error", MaterialRangeWarning)
        counterpart.index(omega_of(np.geomspace(300e-9, 2e-6, 50)))
    first = counterpart.run_transparency(edge)
    assert first["bound_exceeded"]
    assert first["wavelength"] == pytest.approx(120e-9, rel=1e-12)
    second = counterpart.run_transparency(window)
    assert not second["bound_exceeded"]
    assert 0.0 < cast(float, second["largest_dropped_absorption"]) < 60.0
    # The adapter's own record still spans both, and a run without a log takes it.
    assert counterpart.transparency["bound_exceeded"]
    assert counterpart.run_transparency(None) == counterpart.transparency
    band = omega_of(np.geomspace(300e-9, 2e-6, 50))
    (solid,) = material_manifest(
        (counterpart,), selection=None, omega=band, log=window
    ).solids
    assert solid["transparency"] == dict(second)


def test_the_transparent_window_passes_the_bound_and_the_edges_do_not():
    # fit/SiO2's absorption coefficient stays below 60 /m from 200 nm to 2.5 um and
    # reaches 1e8 /m at 120 nm and 2800 /m at 2.73 um (the bound's measurement).
    window = optmat_glass(fused_silica(model="recommended")).as_transparent()
    window.index(omega_of(np.geomspace(200e-9, 2.5e-6, 400)))
    quiet = window.transparency
    assert 0 < window.largest_dropped_absorption < 60.0
    assert not quiet["bound_exceeded"]
    for edge in (120e-9, 2.73e-6):
        transparent = optmat_glass(fused_silica(model="recommended")).as_transparent()
        with pytest.warns(MaterialRangeWarning, match="beyond the transparent") as w:
            transparent.index(omega_of(np.array([800e-9, edge])))
            transparent.index(omega_of(np.array([edge])))
        assert len(w) == 1  # once per adapter
        record = transparent.transparency
        assert record["bound_exceeded"]
        assert record["wavelength"] == pytest.approx(edge, rel=1e-12)
        assert transparent.largest_dropped_absorption > 1e3
    # The bound is declared: a looser one passes the 2.73 um band.
    loose = optmat_glass(fused_silica(model="recommended")).as_transparent(1e4)
    loose.index(omega_of(np.array([2.73e-6])))
    assert not loose.transparency["bound_exceeded"]
    # An empty band drops nothing.
    assert loose.index(np.array([])).size == 0
    with pytest.raises(MaterialError, match="absorption bound must be positive"):
        loose.as_transparent(0.0)


def test_a_lossless_optmat_index_passes_the_lossless_provider():
    mode = StepIndexMode(
        1.25e-6,
        core=calcium_fluoride(model="recommended"),
        cladding=1.0,
        profile_wavelength=835e-9,
    )
    assert np.all(np.imag(mode.neff(omega_of(np.array([800e-9])))) == 0)
