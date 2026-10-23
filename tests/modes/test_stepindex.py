# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check step-index eigenvalues, derivatives, fields and validation."""

import inspect
import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.optimize import brentq
from scipy.special import jn_zeros, jv, kv
from witness_materials import malitson_silica

from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.modes import StepIndexMode, stepindex
from solprop.modes.base import beta2, zdw
from solprop.modes.stepindex import (
    PROFILE_MIN_W,
    _characteristic,
    _cutoff,
    _cutoffs,
    _Kind,
    _partials,
    _roots,
)
from solprop.physdata import fused_silica
from solprop.physdata.glasses import GlassIndex
from solprop.profiles import Linear


@dataclass(frozen=True, eq=False, kw_only=True)
class LinearIndex(GlassIndex):
    value: float
    slope: float
    reference: float

    def _index_and_derivative(self, omega):
        return np.asarray(
            self.value + self.slope * (omega - self.reference), dtype=complex
        ), np.full_like(omega, self.slope, dtype=complex)


def _frequency(one, omega):
    """Return the dimensionless V and squared index ratio of a mode's fibre."""
    core = float(np.real(one.core if isinstance(one.core, float) else 1.45))
    cladding = float(one.cladding) if isinstance(one.cladding, float) else 1.0
    scale = one.radius * float(omega[0]) / C
    return scale * math.sqrt(core**2 - cladding**2), (cladding / core) ** 2


def mode(**keywords):
    parameters = dict(keywords)
    parameters.setdefault("core", 1.45)
    parameters.setdefault("cladding", 1.0)
    parameters.setdefault("profile_wavelength", 1030e-9)
    return StepIndexMode(1e-6, **parameters)


def test_stepindex_issue320_guided_modes_and_refusals():
    modes = []
    for kind, n in (("HE", 1), ("TE", 0), ("TM", 0), ("HE", 2)):
        one = StepIndexMode(
            2e-6,
            core=1.47,
            cladding=1.45,
            kind=cast(_Kind, kind),
            n=n,
            profile_wavelength=1e-6,
        )
        modes.append(one)
        with pytest.raises(ModeError, match=r"below cutoff.*V="):
            replace(one, m=2)
    with pytest.raises(ModeError, match="below cutoff"):
        replace(modes[0], kind="EH")
    omega = np.asarray([2 * math.pi * C / 1e-6])
    assert modes[1].neff(omega)[0] != modes[2].neff(omega)[0]


@pytest.mark.parametrize(
    "factor,message",
    [
        # Just above the exact cutoff the HE12 root lies within float64
        # rounding of V and the scan cannot find it; the mode is not below cutoff.
        (
            1.001,
            r"HE12 is above cutoff at V=3\.8355.*\(cutoff V=3\.8317.*\), but its "
            r"radial root m=2 is not resolvable in float64 this close to cutoff",
        ),
        (0.999, r"HE12 is below cutoff at V=3\.8278.*: no radial root m=2; cutoff V="),
    ],
)
def test_stepindex_missing_root_names_its_cause(factor, message):
    wavelength = 1e-6
    v = factor * float(jn_zeros(1, 1)[0])
    radius = v * wavelength / (2 * math.pi * math.sqrt(1.47**2 - 1.45**2))
    with pytest.raises(ModeError, match=message):
        StepIndexMode(
            radius,
            core=1.47,
            cladding=1.45,
            kind="HE",
            n=1,
            m=2,
            profile_wavelength=wavelength,
        )


@pytest.mark.parametrize(
    "kind,n", [("HE", 1), ("HE", 2), ("HE", 3), ("EH", 1), ("TE", 0), ("TM", 0)]
)
def test_stepindex_profile_boundary_and_partner(kind, n):
    one = mode(kind=kind, n=n)
    theta = 0.37
    points = np.asarray([[1e-6, theta], [np.nextafter(1e-6, np.inf), theta]])
    cartesian = one.field(points)
    radial = cartesian[:, 0] * np.cos(theta) + cartesian[:, 1] * np.sin(theta)
    tangent = -cartesian[:, 0] * np.sin(theta) + cartesian[:, 1] * np.cos(theta)
    # Adjacent floats bracket the interface: error is Bessel/root rounding only.
    assert_allclose(radial[0] * 1.45**2, radial[1], rtol=2e-12, atol=2e-14)
    assert_allclose(tangent[0], tangent[1], rtol=2e-12, atol=2e-14)
    if not n:
        # For n = 0 the two continuity lines above are identities (E_r vanishes
        # for TE, E_theta for TM), so the eigenvalue itself is checked here from
        # jv and kv: TE is w K0 J1 + u J0 K1 = 0 and TM carries the index ratio.
        omega0 = np.asarray([2 * math.pi * C / 1030e-9])
        u = float(_roots(*_frequency(one, omega0), 0, cast("_Kind", kind))[0])
        v = _frequency(one, omega0)[0]
        w = math.sqrt((v - u) * (v + u))
        terms = (w * kv(0, w) * jv(1, u), u * jv(0, u) * kv(1, w))
        scale = abs(terms[0]) + abs(terms[1])
        weights = (1.0, 1.0) if kind == "TE" else (1.45**2, 1.0)
        # Rounding of two Bessel products at V = 6.405: measured 0.74 eps (TE)
        # and 9.7 eps (TM), the latter cancelling two larger terms.
        eps = np.finfo(float).eps
        assert abs(weights[0] * terms[0] + weights[1] * terms[1]) < 32 * eps * scale
        # The other family's equation is far from satisfied there: 0.55 and 0.36.
        other = (1.0, 1.0) if kind == "TM" else (1.45**2, 1.0)
        assert abs(other[0] * terms[0] + other[1] * terms[1]) > 0.3 * scale
    assert one.aeff() > 0
    assert one.normalisation() > 0
    assert one.principal_component() is None
    if n:
        partner = one.orthogonal_polarisation
        assert partner is not None
        assert partner.parity == "odd"
        assert_allclose(partner.aeff(), one.aeff(), rtol=0, atol=0)
    else:
        assert one.orthogonal_polarisation is None


def test_stepindex_frozen_profile_refuses_a_vanishing_cladding_decay():
    # The HE12 root of the strand reproducer sits 1.9e-9 below V, so its
    # cladding decay is 1.2e-4 and the 1e-8 field tail would reach 1e5 core
    # radii. The eigenvalue is still reported; only the frozen profile refuses.
    with pytest.raises(ModeError, match=r"cladding decay w=0.000120081.*1030 nm"):
        StepIndexMode(
            605.088e-9,
            core=1.45,
            cladding=1.0,
            kind="HE",
            n=1,
            m=2,
            profile_wavelength=1030e-9,
        )
    assert PROFILE_MIN_W == 0.1
    # The same mode a little wider is comfortably above the floor.
    assert (
        StepIndexMode(
            8e-7, core=1.45, cladding=1.0, n=1, m=2, profile_wavelength=1030e-9
        )._entrance_profile.w
        > 2.0
    )


def test_stepindex_neff_takes_the_better_conditioned_branch():
    # Near cutoff w is the small parameter and n_eff - n_cl is what the guidance
    # guard and the cladding profile need, so the w branch is used there. Both
    # branches are the same eigenvalue; they differ in the last bit or two.
    radius, core, cladding = 605.088e-9, 1.45, 1.0
    one = StepIndexMode(
        radius, core=core, cladding=cladding, kind="EH", n=1, profile_wavelength=1030e-9
    )
    omega = np.asarray([2 * math.pi * C / 1030e-9])
    scale = radius * float(omega[0]) / C
    v = scale * math.sqrt(core**2 - cladding**2)
    u = _roots(v, (cladding / core) ** 2, 1, "EH")[0]
    w = math.sqrt((v - u) * (v + u))
    assert w < u
    assert_allclose(
        np.real(one.neff(omega))[0],
        math.sqrt(cladding**2 + (w / scale) ** 2),
        rtol=0,
        atol=0,
    )
    # Well above cutoff the u branch is the conditioned one and is used.
    wide = StepIndexMode(2e-6, core=1.45, cladding=1.0, profile_wavelength=1030e-9)
    scale = 2e-6 * float(omega[0]) / C
    v = scale * math.sqrt(1.45**2 - 1)
    u = _roots(v, (1 / 1.45) ** 2, 1, "HE")[0]
    assert math.sqrt((v - u) * (v + u)) > u
    assert_allclose(
        np.real(wide.neff(omega))[0],
        math.sqrt(1.45**2 - (u / scale) ** 2),
        rtol=0,
        atol=0,
    )


def test_stepindex_exact_cutoffs_and_weak_guidance_limit():
    # The two values the manual quotes, to the ten significant figures brentq's
    # four-ulp xtol supports; four eps allows the bracket's own rounding.
    for ratio, expected in (
        ((1.45 / 1.47) ** 2, 2.416293235981197),
        ((1 / 1.45) ** 2, 2.760804855996169),
    ):
        assert_allclose(
            _cutoff("HE", 2, 1, ratio), expected, rtol=4 * np.finfo(float).eps, atol=0
        )
    for n in (2, 3):
        for ratio in ((1.45 / 1.47) ** 2, (1 / 1.45) ** 2):
            cutoff = _cutoff("HE", n, 1, ratio)
            assert not _roots(cutoff - 1e-4, ratio, n, "HE")
            assert len(_roots(cutoff + 1e-4, ratio, n, "HE")) == 1
        # Exact equation tends to J_(n-2)=0 with an O(Delta) displacement.
        delta = 1e-8
        assert_allclose(
            _cutoff("HE", n, 1, 1 - 2 * delta),
            jn_zeros(n - 2, 1)[0],
            rtol=delta,
            atol=0,
        )
    assert mode().cutoff_parameter() == 0
    for kind in ("TE", "TM"):
        assert_allclose(
            mode(kind=kind, n=0).cutoff_parameter(), jn_zeros(0, 1)[0], rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "keywords,match",
    [
        ({"kind": "TE"}, "require"),
        ({"kind": "BAD"}, "kind"),
        ({"n": 0}, "require"),
        ({"m": 0}, "radial"),
        ({"parity": "bad"}, "parity"),
        ({"loss": -1.0}, "loss"),
        ({"core": 0.9}, "core index"),
        ({"core": float("nan")}, "indices"),
        ({"profile_wavelength": 0.0}, "profile_wavelength"),
    ],
)
def test_stepindex_invalid_constructor(keywords, match):
    if keywords.get("kind") == "BAD" or keywords.get("parity") == "bad":
        one = mode()
        for name, value in keywords.items():
            object.__setattr__(one, name, value)
        with pytest.raises(ModeError, match=match):
            one.__post_init__()
    else:
        with pytest.raises(ModeError, match=match):
            mode(**keywords)


@pytest.mark.parametrize("omega", [[0.0], [-1.0], [np.inf], [], [[1.0]], [1j]])
def test_stepindex_invalid_frequency(omega):
    with pytest.raises(ModeError, match="omega"):
        mode().neff(omega)


def test_stepindex_taper_freezing_and_na_constructor():
    taper = StepIndexMode(
        Linear(1e-6, 1.2e-6, 0.1), core=1.45, cladding=1.0, profile_wavelength=1030e-9
    )
    omega = np.asarray([2 * math.pi * C / 1030e-9])
    frozen = taper.at_parameters({"core_radius": 1.2e-6})
    assert not taper.z_constant and not taper.scale_invariant
    assert frozen.z_constant and frozen.radius_is_constant
    assert_allclose(frozen.neff(omega), taper.neff(omega, 0.1), rtol=0, atol=0)
    assert_allclose(frozen.aeff(), taper.aeff(0.1), rtol=0, atol=0)
    na = StepIndexMode.from_numerical_aperture(
        4e-6, 0.12, cladding=1.45, profile_wavelength=1030e-9
    )
    direct = StepIndexMode(
        4e-6, core=1.45 + 0.12**2 / 2.9, cladding=1.45, profile_wavelength=1030e-9
    )
    assert_allclose(na.neff(omega), direct.neff(omega), rtol=0, atol=0)


def test_stepindex_mpmath_witness():
    rows = json.loads(
        (Path(__file__).parents[1] / "data/symbolic_stepindex.json").read_text()
    )["cases"]
    index_errors, derivative_errors = [], []
    for row in rows:
        omega = np.asarray([2 * math.pi * C / row["wavelength"]])
        core = row["core"]
        if row["material_slope"]:
            core = LinearIndex(
                name="linear",
                source="test",
                citation="analytic linear index",
                wavelength_range=(200e-9, 2e-6),
                value=core,
                slope=row["material_slope"],
                reference=float(omega[0]),
            )
        one = StepIndexMode(
            row["radius"],
            core=core,
            cladding=row["cladding"],
            kind=row["kind"],
            n=row["n"],
            m=row["m"],
            profile_wavelength=row["wavelength"],
        )
        index = one.neff(omega)[0]
        derivative = one.dneff_domega(omega)[0]
        index_errors.append(abs(index / row["neff"] - 1))
        derivative_errors.append(abs(derivative / row["derivative"] - 1))
        # Independent 40-digit witness targets; float64 Bessel rounding and the
        # implicit derivative's conditioning were measured when they were set.
        assert_allclose(index, row["neff"], rtol=1e-13, atol=0)
        assert_allclose(derivative, row["derivative"], rtol=1e-12, atol=0)
    assert len(rows) > 50


def test_stepindex_corrected_luna_witness():
    rows = json.loads(
        (Path(__file__).parents[1] / "data/luna_stepindex.json").read_text()
    )["cases"]
    stale: dict[tuple[float, int], dict[str, float]] = {}
    for row in rows:
        one = StepIndexMode(
            row["radius"],
            core=row["core"],
            cladding=row["cladding"],
            kind=row["kind"],
            n=row["n"],
            m=row["m"],
            profile_wavelength=row["wavelength"],
            # Luna's lossless, even-parity mode, stated (``docs/dev/testing.md``).
            parity="even",
            loss=0.0,
        )
        omega = np.asarray([2 * math.pi * C / row["wavelength"]])
        # Independent double Bessel libraries and independent bracketing: rounding.
        assert_allclose(one.neff(omega)[0], row["corrected"], rtol=2e-14, atol=0)
        if row["fibre"] == "issue320" and row["kind"] in ("TE", "TM"):
            assert abs(row["corrected"] - row["stale"]) > 1e-5
            stale.setdefault((row["wavelength"], row["m"]), {})[row["kind"]] = row[
                "stale"
            ]
        if "fields" in row:
            profile = replace(one, profile_wavelength=1030e-9)
            samples = np.asarray(row["fields"])
            points = np.column_stack(
                (samples[:, 0] * row["radius"], np.full(len(samples), 0.37))
            )
            # Field amplitudes include Bessel denominators and the eigenpair;
            # cancellation in small components requires an absolute rounding term.
            assert_allclose(
                profile.field(points), samples[:, 1:], rtol=2e-12, atol=2e-13
            )
    # The defect itself, not only the disagreement: Luna's stale TE and TM
    # values are the SAME number at every m = 1 row, because find_zeros returned
    # the J_0 pole for both (1.4574901533331723 at 1 um).
    first = {wavelength: pair for (wavelength, m), pair in stale.items() if m == 1}
    assert len(first) == 3
    for pair in first.values():
        assert pair["TE"] == pair["TM"]
    assert first[1e-6]["TE"] == 1.4574901533331723


def test_stepindex_remaining_geometry_contracts():
    one = mode()
    assert one.azimuthal_order == 1
    assert one.radial_breaks() == (1e-6,)
    assert set(one.parameter_paths) == {"core_radius"}
    assert one.at_parameters({}).radius == one.radius
    partner = one.orthogonal_polarisation
    assert partner is not None
    original = partner.orthogonal_polarisation
    assert original is not None
    assert original.parity == "even"
    points = np.asarray([[0.1e-6, 0.37], [1.1e-6, 0.37]])
    assert np.all(np.isfinite(partner.field(points)))
    with pytest.raises(ModeError, match="unknown mode parameters"):
        one.at_parameters({"density": 1.0})
    with pytest.raises(ModeError, match="core radius"):
        one.at_parameters({"core_radius": 0.0})
    # Beyond optmat's silica, 24.8 nm to 125 um (the registry's ended at 3.71 um).
    with pytest.raises(ModeError, match="profile_wavelength"):
        StepIndexMode(1e-6, profile_wavelength=200e-6)
    with pytest.raises(TypeError):
        inspect.signature(StepIndexMode).bind(1e-6)
    with pytest.raises(ModeError, match="numerical_aperture"):
        StepIndexMode.from_numerical_aperture(1e-6, 0, profile_wavelength=800e-9)
    assert (
        StepIndexMode.from_numerical_aperture(
            4e-6, 0.12, profile_wavelength=800e-9
        ).aeff()
        > 0
    )
    with pytest.raises(ModeError, match="points"):
        one.field(np.asarray([[-1e-6, 0.0]]))
    # Deliberately ill-shaped inputs are rejected by the test hook first. Invoke
    # the boundary body to qualify the same explicit production validation.
    with pytest.raises(ModeError, match="points"):
        inspect.unwrap(StepIndexMode.field)(one, np.asarray([0.0]))


class LossCurve:
    def __call__(self, omega):
        return (omega / 1e15) ** 2

    def derivative(self, omega):
        return 2 * omega / 1e30


def test_stepindex_absolute_loss_and_derivative(caplog):
    omega = np.asarray([2e15, 2.5e15])
    lossless = mode()
    for loss in (2.0, LossCurve(), lambda omega: (omega / 1e15) ** 2):
        one = mode(loss=loss)
        attenuation = (
            np.full_like(omega, loss) if isinstance(loss, (int, float)) else loss(omega)
        )
        slope = (
            np.zeros_like(omega) if isinstance(loss, (int, float)) else 2 * omega / 1e30
        )
        assert_allclose(
            np.real(one.neff_minus_one(omega)),
            np.real(lossless.neff(omega)) - 1,
            rtol=0,
            atol=0,
        )
        assert_allclose(
            np.imag(one.neff(omega)), C * attenuation / (2 * omega), rtol=4e-16, atol=0
        )
        # Richardson rounding is amplified by the 1e-5 relative step, while
        # this quadratic has no truncation error: a 1e-10 envelope suffices.
        assert_allclose(
            np.imag(one.dneff_domega(omega)),
            C * (slope * omega - attenuation) / (2 * omega**2),
            rtol=1e-10,
            atol=0,
        )
    assert "Richardson" in caplog.text


@pytest.mark.parametrize("output", [-1.0, np.nan, np.inf, [1.0, 2.0]])
def test_stepindex_invalid_callable_loss(output):
    one = mode(loss=lambda omega: np.asarray(output))
    with pytest.raises(ModeError, match="loss callable"):
        one.neff(np.asarray([2e15]))


def test_stepindex_invalid_derivative_and_solver_results(monkeypatch):
    omega = np.asarray([2e15])
    curve = LossCurve()
    monkeypatch.setattr(curve, "derivative", lambda omega: np.asarray([np.nan]))
    with pytest.raises(ModeError, match="loss derivative"):
        mode(loss=curve).dneff_domega(omega)
    one = mode()
    with monkeypatch.context() as patch:
        # Failure injection qualifies the public response to a singular implicit
        # derivative, which must never escape as NaN or Inf.
        patch.setattr(stepindex, "_partials", lambda *args: (0.0, 0.0, 0.0))
        with pytest.raises(ModeError, match="singular"):
            one.dneff_domega(omega)
    with monkeypatch.context() as patch:
        patch.setattr(
            stepindex,
            "_material",
            lambda material, omega: (
                np.full_like(omega, 1.45 if material == 1.45 else 1.0),
                np.full_like(omega, np.nan),
            ),
        )
        with pytest.raises(ModeError, match="derivative must be finite"):
            one.dneff_domega(omega)
    with monkeypatch.context() as patch:
        patch.setattr(stepindex, "_roots", lambda *args: [0.0])
        with pytest.raises(ModeError, match="too close to cutoff"):
            one.neff(omega)


@settings(max_examples=12, deadline=None)
@given(
    st.floats(min_value=1.2, max_value=1.8), st.floats(min_value=6.0, max_value=10.0)
)
def test_stepindex_guided_index_bounds_and_radial_order(core, v):
    radius = v * 1030e-9 / (2 * math.pi * math.sqrt(core**2 - 1))
    first = StepIndexMode(radius, core=core, cladding=1.0, profile_wavelength=1030e-9)
    second = replace(first, m=2)
    omega = np.asarray([2 * math.pi * C / 1030e-9])
    assert 1 < np.real(second.neff(omega)[0]) < np.real(first.neff(omega)[0]) < core


@pytest.mark.parametrize(
    "kind,n,v,scaled",
    [
        ("HE", 1, 6.0, True),
        ("EH", 1, 6.0, True),
        ("TE", 0, 6.0, False),
        ("TM", 0, 6.0, False),
        ("HE", 2, 8.0, True),
    ],
)
def test_stepindex_partials_differentiate_the_scanned_equation(kind, n, v, scaled):
    # _partials and _characteristic differ by u**2 w**2 on the hybrid branch and
    # not at all on the TE/TM one; both quotients are the same at a root, and
    # this is what ties the two expressions together (measured worst 1.6e-9).
    ratio = (1.0 / 1.45) ** 2

    def equation(u, w, index=ratio):
        one = math.sqrt(u * u + w * w)
        factor = (u**2 * w**2) if scaled else 1.0
        return float(factor * _characteristic(u, one, index, n, cast("_Kind", kind)))

    for u in _roots(v, ratio, n, cast("_Kind", kind)):
        w = math.sqrt((v - u) * (v + u))
        du, dw, dratio = _partials(u, w, ratio, n, cast("_Kind", kind))
        step = 1e-6
        assert_allclose(
            du,
            (equation(u + step, w) - equation(u - step, w)) / (2 * step),
            rtol=1e-7,
            atol=0,
        )
        assert_allclose(
            dw,
            (equation(u, w + step) - equation(u, w - step)) / (2 * step),
            rtol=1e-7,
            atol=0,
        )
        change = ratio * step
        expected = (equation(u, w, ratio + change) - equation(u, w, ratio - change)) / (
            2 * change
        )
        if kind == "TE":
            assert dratio == expected == 0.0
        else:
            assert_allclose(dratio, expected, rtol=1e-7, atol=0)


def test_stepindex_material_derivative_between_witness_points():
    one = StepIndexMode(1.25e-6, core=fused_silica(), profile_wavelength=835e-9)
    omega = 2 * math.pi * C / np.linspace(450e-9, 1350e-9, 13)
    step = omega * 1e-3
    wide = (one.neff(omega + step) - one.neff(omega - step)) / (2 * step)
    narrow = (one.neff(omega + step / 2) - one.neff(omega - step / 2)) / step
    # Fourth-order truncation plus subtraction of indices near 1.45.
    assert_allclose((4 * narrow - wide) / 3, one.dneff_domega(omega), rtol=1e-9, atol=0)
    assert np.all(np.isfinite(beta2(one, omega)))
    assert 400e-9 < zdw(one, wavelength_range=(400e-9, 1400e-9)) < 1400e-9
    # The cladding's own slope, which every witness fibre leaves at a constant:
    # a numerical-aperture core over a silica cladding (measured 3.36e-11) and a
    # constant core over one (1.06e-10). Both dcladding terms are live here. The
    # silica is Malitson's Sellmeier law, supplied explicitly (the real part of
    # solprop's former registry silica at dc4ca0d, deleted in SEP 3's Phase 4b): the
    # check needs an index smooth to rounding, and optmat's fit/SiO2 carries about
    # 1e-14 of evaluation noise, which holds a Richardson difference near 3e-9.
    band = 2 * math.pi * C / np.linspace(500e-9, 1400e-9, 13)
    silica = malitson_silica()
    for fibre, bound in (
        (
            StepIndexMode.from_numerical_aperture(
                4e-6, 0.12, cladding=silica, profile_wavelength=1030e-9
            ),
            1e-10,
        ),
        (
            StepIndexMode(4e-6, core=1.60, cladding=silica, profile_wavelength=1030e-9),
            4e-10,
        ),
    ):
        step = band * 1e-3
        wide = (fibre.neff(band + step) - fibre.neff(band - step)) / (2 * step)
        narrow = (fibre.neff(band + step / 2) - fibre.neff(band - step / 2)) / step
        assert_allclose(
            (4 * narrow - wide) / 3, fibre.dneff_domega(band), rtol=bound, atol=0
        )


def test_stepindex_weak_guidance_and_characteristic_identity():
    delta, core, v, wavelength = 1e-4, 1.45, 3.04, 1e-6
    cladding = core * math.sqrt(1 - 2 * delta)
    radius = v * wavelength / (2 * math.pi * math.sqrt(core**2 - cladding**2))
    modes = [
        StepIndexMode(
            radius,
            core=core,
            cladding=cladding,
            profile_wavelength=wavelength,
            kind=cast(_Kind, kind),
            n=n,
        )
        for kind, n in (("HE", 1), ("HE", 2), ("TE", 0), ("TM", 0))
    ]

    def lp_equation(u):
        w = math.sqrt(v**2 - u**2)
        return u * jv(1, u) / jv(0, u) - w * kv(1, w) / kv(0, w)

    u = brentq(lp_equation, 1e-6, jn_zeros(0, 1)[0] - 1e-8)
    scalar = math.sqrt(core**2 - (u * wavelength / (2 * math.pi * radius)) ** 2)
    omega = np.asarray([2 * math.pi * C / wavelength])
    values = np.asarray([np.real(one.neff(omega)[0]) for one in modes])
    # LP is the leading weak-guidance equation: its index error and the triplet
    # splitting are O(Delta) of the core/cladding separation, hence O(Delta²).
    assert abs(values[0] - scalar) < 2 * core * delta**2
    assert np.ptp(values[1:]) < 2 * core * delta**2
    profile = modes[0]._entrance_profile
    lhs = 1 / profile.u**2 + profile.ratio / profile.w**2
    rhs = (values[0] / core) ** 2 * (1 / profile.u**2 + 1 / profile.w**2)
    # Algebraic identity evaluated with a handful of rounded operations.
    assert_allclose(lhs, rhs, rtol=8 * np.finfo(float).eps, atol=0)


@pytest.mark.parametrize("v", [3.0, 6.0, 9.0])
@pytest.mark.parametrize("ratio", [(1.45 / 1.47) ** 2, (1 / 1.45) ** 2])
def test_stepindex_root_count_against_the_exact_cutoffs(v, ratio):
    # The exact cutoff relation, not its weak-guidance limit: one root per
    # cutoff below V, and each root between its own cutoff and the next.
    for n in (1, 2, 3):
        for kind in ("HE", "EH"):
            cutoffs = [
                _cutoff(cast("_Kind", kind), n, m, ratio)
                for m in range(1, 6)
                if _cutoff(cast("_Kind", kind), n, m, ratio) < v
            ]
            roots = _roots(v, ratio, n, cast("_Kind", kind))
            assert len(roots) == len(cutoffs)
            for order, root in enumerate(roots):
                assert cutoffs[order] < root <= v
                if order + 1 < len(cutoffs):
                    assert root < cutoffs[order + 1]
    for kind in ("TE", "TM"):
        expected = int(np.count_nonzero(jn_zeros(0, 5) < v))
        assert len(_roots(v, ratio, 0, cast("_Kind", kind))) == expected


def test_stepindex_shared_cutoff_pair_is_found_at_the_strand_reproducer():
    # J_(-1) = -J_1 puts EH11 and HE12 at the same cutoff j_11, both roots of
    # the same n = 1 product, and a uniform scan sees their two sign changes
    # cancel. Here V = j_11 + 0.04400 and both roots are in the last interval.
    radius, core = 605.088e-9, 1.45
    one = StepIndexMode(
        radius, core=core, cladding=1.0, kind="EH", n=1, profile_wavelength=1030e-9
    )
    omega = np.asarray([2 * math.pi * C / 1030e-9])
    scale = radius * float(omega[0]) / C
    v = scale * math.sqrt(core**2 - 1)
    assert 0.043 < v - float(jn_zeros(1, 1)[0]) < 0.045
    ratio = (1.0 / core) ** 2
    eh = _roots(v, ratio, 1, "EH")
    he = _roots(v, ratio, 1, "HE")
    # Solidly guided: w = 0.392, so EH11 is not a cutoff artefact.
    assert_allclose(eh, [3.855790808854866], rtol=4e-15, atol=0)
    assert_allclose(he[1], 3.875702942516672, rtol=4e-15, atol=0)
    assert len(he) == 2 and he[0] < float(jn_zeros(1, 1)[0]) < eh[0] < he[1] < v
    assert_allclose(np.real(one.neff(omega)), [1.0056338752319165], rtol=4e-15, atol=0)


@pytest.mark.parametrize("core,cladding", [(1.45, 1.0), (1.47, 1.45), (1.6, 1.45)])
def test_stepindex_counts_every_family_just_above_a_shared_cutoff(core, cladding):
    ratio = (cladding / core) ** 2
    for zero in jn_zeros(1, 2):
        for offset in np.linspace(0.02, 0.15, 12):
            v = float(zero + offset)
            for n in (1, 2, 3):
                for kind in ("HE", "EH"):
                    cutoffs = _cutoffs(v, ratio, n, cast("_Kind", kind))
                    roots = _roots(v, ratio, n, cast("_Kind", kind))
                    for order, root in enumerate(roots):
                        assert cutoffs[order] < root <= v
                    if offset > 0.03:
                        assert len(roots) == len(cutoffs)
                    else:
                        # Within about 0.03 of its own cutoff the newborn root
                        # sits at V - u below float64 resolution (measured: the
                        # last offset that cannot be separated is 0.027 for
                        # 1.45/1.0 at j_11). It is reported short, never
                        # misaligned, and the caller refuses that m.
                        assert len(cutoffs) - 1 <= len(roots) <= len(cutoffs)
    for zero in jn_zeros(0, 2):
        for offset in np.linspace(0.02, 0.15, 12):
            v = float(zero + offset)
            for kind in ("TE", "TM"):
                roots = _roots(v, ratio, 0, cast("_Kind", kind))
                assert len(roots) == len(_cutoffs(v, ratio, 0, cast("_Kind", kind)))


def test_stepindex_resolves_two_roots_in_last_cutoff_interval():
    v = float(jn_zeros(1, 1)[0]) + 0.04
    ratio = (1.45 / 1.47) ** 2
    he = _roots(v, ratio, 1, "HE")
    eh = _roots(v, ratio, 1, "EH")
    assert len(he) == 2 and len(eh) == 1
    assert float(jn_zeros(1, 1)[0]) < eh[0] < he[1] < v
    # Independent dense bracketing identifies the EH root well inside the short
    # interval and the second HE root within 4e-7 of V. These inequalities check
    # topology; the existing high-precision witnesses set numerical tolerances.
    assert 0.019 < v - eh[0] < 0.022
    assert 2e-7 < v - he[1] < 4e-7


def test_stepindex_refinement_exhausts_its_levels_and_reports_short(
    monkeypatch, caplog
):
    # A refinement that can never recover the missing root walks every level and
    # reports the count short rather than misaligned: the helper is wrapped to drop
    # the family's highest root on every call, which is the unresolvable case of a
    # near-cutoff root made deterministic (the real one needs V within 0.027 of a
    # cutoff).
    real = stepindex._refined_roots

    def short(nodes, values, v, ratio, n, kind):
        found = real(nodes, values, v, ratio, n, kind)
        found[kind] = found[kind][:-1]
        return found

    monkeypatch.setattr(stepindex, "_refined_roots", short)
    ratio = (1.45 / 1.47) ** 2
    with caplog.at_level(logging.DEBUG, logger="solprop.modes.stepindex"):
        roots = _roots(6.0, ratio, 1, "HE")
    assert len(roots) == len(_cutoffs(6.0, ratio, 1, "HE")) - 1
    assert "found 1 of 2 roots" in caplog.text


def test_stepindex_misaligned_roots_are_refused(monkeypatch):
    # More roots than exact cutoffs cannot be given radial orders: the cutoff
    # helper is corrupted at the boundary so the alignment check is what refuses.
    monkeypatch.setattr(stepindex, "_cutoffs", lambda v, ratio, n, kind: [0.0])
    with pytest.raises(ModeError, match="do not align"):
        _roots(6.0, (1.45 / 1.47) ** 2, 1, "HE")
