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

import mpmath
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.optimize import brentq
from scipy.special import jn_zeros, jv, kv, kve
from witness_materials import malitson_silica

from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.modes import StepIndexMode, stepindex
from solprop.modes._stepindex_vector import _radial
from solprop.modes.base import beta2, zdw
from solprop.modes.stepindex import (
    _MAX_V,
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

# The large-V regressions: the core index of a numerical aperture of 0.22 over
# 1.45, as in the reproduction of the scan's failure, and an air-clad strand.
_NA_RATIO = (1.45 / (1.45 + 0.22**2 / 2.9)) ** 2
_AIR_RATIO = (1.0 / 1.45) ** 2


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
    assert 400e-9 < zdw(one, wavelength_limits=(400e-9, 1400e-9)) < 1400e-9
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


def test_stepindex_unrecoverable_root_reports_short(monkeypatch, caplog):
    # A root that neither the product scan nor the family's own factor can
    # bracket is reported short rather than misaligned: the scan helper is
    # wrapped to drop the family's highest root and the factor made sign-definite,
    # which is the unresolvable case of a near-cutoff root made deterministic (the
    # real one needs V within 0.027 of a cutoff).
    real = stepindex._refined_roots

    def short(nodes, values, v, ratio, n, kind):
        found = real(nodes, values, v, ratio, n, kind)
        found[kind] = found[kind][:-1]
        return found

    monkeypatch.setattr(stepindex, "_refined_roots", short)
    monkeypatch.setattr(stepindex, "_branch", lambda u, v, ratio, n, kind: u + 1.0)
    ratio = (1.45 / 1.47) ** 2
    with caplog.at_level(logging.DEBUG, logger="solprop.modes.stepindex"):
        roots = _roots(6.0, ratio, 1, "HE")
    assert len(roots) == len(_cutoffs(6.0, ratio, 1, "HE")) - 1
    assert "found 1 of 2 roots" in caplog.text


def test_stepindex_lost_root_is_recovered_by_its_own_factor(monkeypatch):
    # The factor brackets the root the product scan dropped, and the recovered
    # value is the product's own root to rounding: the two equations are one.
    ratio = (1.45 / 1.47) ** 2
    exact = _roots(6.0, ratio, 1, "HE")
    real = stepindex._refined_roots

    def short(nodes, values, v, ratio, n, kind):
        found = real(nodes, values, v, ratio, n, kind)
        found[kind] = found[kind][1:]
        return found

    monkeypatch.setattr(stepindex, "_refined_roots", short)
    # 4 ulp is the bracketing tolerance of both root solves.
    assert_allclose(
        _roots(6.0, ratio, 1, "HE"), exact, rtol=4 * np.finfo(float).eps, atol=0
    )


@pytest.mark.parametrize(
    ("branch", "scan", "message"),
    [
        # One cutoff interval holds one guided root, so a factor that changes sign
        # three times in an empty one is refused, and the message says why, instead
        # of guessing which candidate is the root.
        (
            lambda u, v, ratio, n, kind: (u - 0.5) * (u - 1.0) * (u - 1.5),
            (),
            "changes sign 3 times",
        ),
        # With the scan capped below V, a missing root cannot be the one within
        # rounding of V, so it is refused with its own cause rather than reported
        # short for the caller to call a near-cutoff mode.
        (
            lambda u, v, ratio, n, kind: u + 1.0,
            (1,),
            "a root away from V could not be resolved",
        ),
    ],
)
def test_stepindex_an_unrecoverable_root_is_refused(monkeypatch, branch, scan, message):
    real = stepindex._refined_roots

    def short(nodes, values, v, ratio, n, kind):
        found = real(nodes, values, v, ratio, n, kind)
        found[kind] = found[kind][1:]
        return found

    monkeypatch.setattr(stepindex, "_refined_roots", short)
    monkeypatch.setattr(stepindex, "_branch", branch)
    with pytest.raises(ModeError, match=message):
        _roots(6.0, (1.45 / 1.47) ** 2, 1, "HE", *scan)


@pytest.mark.parametrize(
    "v,ratio,n,kind",
    [
        (6.0, (1.45 / 1.47) ** 2, 1, "HE"),
        (67.29, _NA_RATIO, 1, "EH"),
        (403.77, _AIR_RATIO, 2, "HE"),
        (1663.17, _NA_RATIO, 0, "TM"),
        (1000.0, _NA_RATIO, 60, "EH"),
    ],
)
def test_stepindex_capped_scan_returns_the_full_scan_bit_for_bit(v, ratio, n, kind):
    # The capped scan takes the same nodes as the full one over its range, so
    # every bracket, and every root, is the same float (a structural contract).
    full = _roots(v, ratio, n, cast("_Kind", kind))
    for orders in (1, 2, 5, len(full) // 2, len(full), len(full) + 3):
        assert _roots(v, ratio, n, cast("_Kind", kind), orders) == full[:orders]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("n", [40, 60, 150, 300])
def test_stepindex_k_quotients_stay_finite_where_kve_overflows(n):
    # kve(n, w) overflows for large n at small w; the quotient recurrence must
    # agree with mpmath there and with the direct scaled form where both exist.
    # Each recurrence step rounds once, so n steps bound the error at 4 n epsilons.
    bound = 4 * n * np.finfo(float).eps
    with mpmath.workdps(40):
        for w in (6e-4, 0.3, 5.0, 40.0):
            lower, centre = mpmath.besselk(n - 1, w), mpmath.besselk(n, w)
            above = mpmath.besselk(n + 1, w)
            assert_allclose(
                stepindex._k_ratio(n, w), float(lower / (w * centre)), rtol=bound
            )
            assert_allclose(
                stepindex._log_slope(n, w),
                float(-(lower + above) / (2 * centre)),
                rtol=bound,
            )
            radius = np.asarray([1.0, 1.01, 1.5, 3.0])
            decay = stepindex._k_decay(n, w, radius)
            expected = [float(mpmath.besselk(n, w * r) / centre) for r in radius]
            assert np.all(np.isfinite(decay))
            assert_allclose(decay, expected, rtol=bound, atol=1e-300)


def test_stepindex_recovery_ignores_a_candidate_in_an_occupied_interval(monkeypatch):
    # The scan intervals searched include the one straddling the empty interval's
    # lower cutoff; a sign change of the factor there whose root falls on the
    # occupied side is not a candidate, and the lost root is still recovered.
    v, ratio, n = 8.0, _NA_RATIO, 2
    exact = _roots(v, ratio, n, "HE")
    cutoffs = _cutoffs(v, ratio, n, "HE")
    nodes = stepindex._scan_nodes(v, n, cutoffs[0], v)
    below = float(nodes[np.searchsorted(nodes, cutoffs[1], side="right") - 1])
    decoy = (below + cutoffs[1]) / 2
    assert below < decoy < cutoffs[1]
    real_scan, real_branch = stepindex._refined_roots, stepindex._branch

    def short(nodes, values, v, ratio, n, kind):
        found = real_scan(nodes, values, v, ratio, n, kind)
        found[kind] = found[kind][:1]
        return found

    monkeypatch.setattr(stepindex, "_refined_roots", short)
    monkeypatch.setattr(
        stepindex,
        "_branch",
        lambda u, v, ratio, n, kind: (u - decoy) * real_branch(u, v, ratio, n, kind),
    )
    # The recovered root is the factor's, within the 4 ulp of both root solves.
    assert_allclose(
        _roots(v, ratio, n, "HE"), exact, rtol=4 * np.finfo(float).eps, atol=0
    )


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("order,edge", [(299, 13.6), (301, 13.6), (300, 2.5)])
def test_stepindex_vector_exterior_stays_finite_where_kve_overflows(order, edge):
    # The vector provider's exterior radial functions fall back to the quotient
    # recurrences where kve(order + 1, edge) overflows; against mpmath, within
    # 4 order epsilons (one rounding per recurrence step).
    assert not math.isfinite(float(kve(order + 1, edge)))
    argument = np.asarray([edge, 1.05 * edge, 2.0 * edge])
    value, slope, divided = _radial(order, argument, edge, inside=False)
    with mpmath.workdps(40):
        reference = mpmath.besselk(order, edge)
        expected = [float(mpmath.besselk(order, x) / reference) for x in argument]
        derivative = [
            float(
                -(mpmath.besselk(order - 1, x) + mpmath.besselk(order + 1, x))
                / (2 * reference)
            )
            for x in argument
        ]
    bound = 4 * order * np.finfo(float).eps
    assert_allclose(value, expected, rtol=bound, atol=1e-300)
    assert_allclose(slope, derivative, rtol=bound, atol=1e-300)
    assert_allclose(divided, value / argument, rtol=0, atol=0)


def test_stepindex_band_above_the_limit_is_refused_before_any_solve(monkeypatch):
    # One check for the whole band: no frequency is solved when the band's
    # largest V is above the limit.
    calls = []
    real = stepindex._roots

    def counted(*arguments):
        calls.append(arguments)
        return real(*arguments)

    one = StepIndexMode(1e-6, core=1.45, cladding=1.0, profile_wavelength=1030e-9)
    monkeypatch.setattr(stepindex, "_roots", counted)
    scale = 1e-6 * math.sqrt(1.45**2 - 1) / C
    omega = np.asarray([10.0, 0.5 * _MAX_V, 1.01 * _MAX_V]) / scale
    with pytest.raises(ModeError, match="the largest V"):
        one.neff(omega)
    assert not calls


def test_stepindex_misaligned_roots_are_refused(monkeypatch):
    # More roots than exact cutoffs cannot be given radial orders: the cutoff
    # helper is corrupted at the boundary so the alignment check is what refuses.
    monkeypatch.setattr(
        stepindex, "_cutoffs", lambda v, ratio, n, kind, count=None: [0.0]
    )
    with pytest.raises(ModeError, match="do not align"):
        _roots(6.0, (1.45 / 1.47) ** 2, 1, "HE")


def _assert_complete(v, ratio, n, kind):
    """Assert that the scan finds one aligned root per exact cutoff below V."""
    cutoffs = _cutoffs(v, ratio, n, cast("_Kind", kind))
    roots = _roots(v, ratio, n, cast("_Kind", kind))
    if cutoffs and v - cutoffs[-1] < 0.03:
        # A root born within 0.03 of V may sit within float64 rounding of V
        # (test_stepindex_counts_every_family_just_above_a_shared_cutoff).
        assert len(roots) >= len(cutoffs) - 1
    else:
        assert len(roots) == len(cutoffs)


@pytest.mark.parametrize(
    "radius,wavelength",
    [
        (25e-6, 1030e-9),
        (50e-6, 1030e-9),
        (100e-6, 1030e-9),
        (300e-6, 1030e-9),
        (300e-6, 250e-9),
        (300e-6, 256e-9),
    ],
)
def test_stepindex_large_core_fibres_build(radius, wavelength):
    # The reproduction's fibres, V from 34 to 1663: the scan lost close HE/EH
    # pairs from V of about 26 and kv underflowed from a cladding argument of
    # about 698. Warnings are errors in this suite, so an invalid division fails.
    one = StepIndexMode.from_numerical_aperture(
        radius, 0.22, profile_wavelength=wavelength
    )
    omega = np.asarray([2 * math.pi * C / wavelength])
    index = np.real(one.neff(omega)[0])
    cladding = float(np.real(fused_silica().index(omega)[0]))
    assert cladding < index < cladding + 0.22**2 / (2 * cladding)
    assert np.isfinite(one.aeff()) and one.aeff() > 0


def test_stepindex_scan_is_complete_where_it_was_intermittent():
    # The reproduction refused 303 of 3700 values of V from 3 to 40, the lowest
    # at 26.01, where n = 1 pairs first come closer than the scan spacing.
    for v in np.arange(26.0, 40.0, 0.1):
        for kind in ("HE", "EH"):
            _assert_complete(float(v), _NA_RATIO, 1, kind)


@pytest.mark.medium
@pytest.mark.parametrize("ratio", [_NA_RATIO, _AIR_RATIO])
def test_stepindex_scan_is_complete_over_v(ratio):
    for v in np.arange(26.0, 700.0, 37.3):
        for n in (1, 2, 3):
            for kind in ("HE", "EH"):
                _assert_complete(float(v), ratio, n, kind)


@pytest.mark.medium
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("ratio", [_NA_RATIO, _AIR_RATIO])
def test_stepindex_scan_is_complete_at_high_azimuthal_order(ratio):
    # J_n underflows at small u for large n, and kve(n, w) overflows at small w:
    # neither may manufacture a root or a NaN. V = 200 and 1000 here, 3000 and
    # 5000 in the slow lane.
    for v in (200.0, 1000.0):
        for n in (40, 60, 150, 300):
            for kind in ("HE", "EH"):
                _assert_complete(v, ratio, n, kind)


@pytest.mark.slow
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("ratio", [_NA_RATIO, _AIR_RATIO])
def test_stepindex_scan_is_complete_up_to_the_largest_v(ratio):
    # The whole qualified range, ending at the limit itself: three times the
    # reproduction's spacing of 1.37 below the old kv limit, then a geometric
    # ladder, for n = 1 to 3, and high azimuthal orders at V = 3000 and 5000
    # (about 35 s for each contrast).
    for v in (3000.0, 5000.0):
        for n in (40, 60, 150, 300):
            for kind in ("HE", "EH"):
                _assert_complete(v, ratio, n, kind)
    for v in np.concatenate(
        (np.arange(5.0, 700.0, 4.11), np.geomspace(700.0, _MAX_V, 10))
    ):
        for n in (1, 2, 3):
            for kind in ("HE", "EH"):
                _assert_complete(float(v), ratio, n, kind)


def test_stepindex_refuses_above_the_largest_v():
    with pytest.raises(ModeError, match="exceeds 20000, the largest V"):
        _roots(_MAX_V * (1 + 1e-12), _NA_RATIO, 1, "HE")
    radius = 1.01 * _MAX_V * 1030e-9 / (2 * math.pi * math.sqrt(1.45**2 - 1))
    with pytest.raises(ModeError, match="the largest V"):
        StepIndexMode(radius, core=1.45, cladding=1.0, profile_wavelength=1030e-9)


def _textbook(u, v, ratio, n, kind):
    """Evaluate the unreduced eigenvalue equation in mpmath, poles and all."""
    w = mpmath.sqrt(v * v - u * u)
    a = mpmath.besselj(n, u, derivative=1) / (u * mpmath.besselj(n, u))
    b = -(mpmath.besselk(abs(n - 1), w) + mpmath.besselk(n + 1, w)) / (
        2 * w * mpmath.besselk(n, w)
    )
    if kind == "TE":
        return a + b
    if kind == "TM":
        return a + ratio * b
    coupling = n**2 * (1 / u**2 + 1 / w**2) * (1 / u**2 + ratio / w**2)
    return (a + b) * (a + ratio * b) - coupling, a, b, coupling


def _textbook_root_equation(v, ratio, n, kind):
    """Return the unreduced equation as a function of u alone, for findroot."""

    def equation(u):
        value = _textbook(u, v, ratio, n, kind)
        return value[0] if isinstance(value, tuple) else value

    return equation


def _textbook_cutoff(kind, n, m, ratio):
    """Return an exact cutoff from mpmath's own Bessel zeros."""
    if kind in ("TE", "TM"):
        return mpmath.besseljzero(0, m)
    if kind == "EH":
        return mpmath.besseljzero(n, m)
    if n == 1:
        return mpmath.mpf(0) if m == 1 else mpmath.besseljzero(1, m - 1)
    contrast = (1 - ratio) / (1 + ratio)
    return mpmath.findroot(
        lambda x: mpmath.besselj(n - 2, x) + contrast * mpmath.besselj(n, x),
        (mpmath.besseljzero(n - 2, m), mpmath.besseljzero(n - 1, m)),
        solver="anderson",
    )


_LOW_ORDERS = (
    ("HE", 1),
    ("EH", 1),
    ("HE", 2),
    ("EH", 2),
    ("HE", 3),
    ("TE", 0),
    ("TM", 0),
)
_NA_CORE = 1.45 + 0.22**2 / 2.9


@pytest.mark.medium
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "v,core,cladding,families",
    [
        (4990.0, _NA_CORE, 1.45, _LOW_ORDERS),
        (1000.0, _NA_CORE, 1.45, (("HE", 40), ("EH", 40), ("HE", 150), ("EH", 150))),
    ],
)
def test_stepindex_large_v_against_a_textbook_mpmath_solve(v, core, cladding, families):
    _assert_matches_textbook(v, core, cladding, families)


@pytest.mark.slow
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "v,core,cladding,families",
    [
        (19990.0, _NA_CORE, 1.45, (("HE", 1), ("EH", 1), ("TE", 0))),
        (4990.0, 1.45, 1.0, (("HE", 300), ("EH", 300))),
    ],
)
def test_stepindex_largest_v_and_order_against_a_textbook_mpmath_solve(
    v, core, cladding, families
):
    # The limit's own V and the highest azimuthal order, about 2.5 s each.
    _assert_matches_textbook(v, core, cladding, families)


def _assert_matches_textbook(v, core, cladding, families):
    """Assert the scan's roots against an mpmath solve of the unreduced equation."""
    # An independent reference: the unreduced equation (J_n'/(u J_n) + K_n'/(w K_n))
    # (J_n'/(u J_n) + ratio K_n'/(w K_n)) = n**2 (1/u**2 + 1/w**2)(1/u**2 + ratio/w**2)
    # with its poles, mpmath's Bessel functions at 40 digits, its own root solver,
    # and the radial order and family assigned from mpmath's own cutoffs and the
    # quadratic's two branches. The modes checked are the first order, two more
    # and, for the hybrids, the root with the closest partner of the other family,
    # which the product scan alone loses at low n.
    ratio_f = (cladding / core) ** 2
    with mpmath.workdps(40):
        ratio = mpmath.mpf(cladding) ** 2 / mpmath.mpf(core) ** 2
        exact_v = mpmath.mpf(v)
        scale = exact_v / mpmath.sqrt(mpmath.mpf(core) ** 2 - mpmath.mpf(cladding) ** 2)
        for kind, n in families:
            # At high n only the lowest orders are checked, through the scan
            # capped at the requested order: mpmath's zero of order n and index m
            # costs about 0.15 s at n = 300 and m in the hundreds.
            capped = 3 if n > 200 else None
            roots = np.asarray(_roots(v, ratio_f, n, cast("_Kind", kind), capped))
            # Roots with u <= 0.8 V keep w >= 0.6 V, where mpmath's besselk takes
            # its asymptotic branch; at w of a few tens its series takes about
            # 0.1 s a call (measured 0.12 s at w = 39), too slow for this tier.
            reach = int(np.count_nonzero(roots <= 0.8 * v))
            orders = {1, reach} if capped else {1, reach // 2, reach}
            if kind in ("HE", "EH") and n <= 3:
                partner = np.asarray(
                    _roots(v, ratio_f, n, "EH" if kind == "HE" else "HE")
                )
                gaps = np.abs(roots[:reach, None] - partner[None, :]).min(axis=1)
                orders.add(int(np.argmin(gaps)) + 1)
                # The closest pair in range is one the product scan loses.
                assert gaps.min() < 0.1
            for m in sorted(orders):
                found = float(roots[m - 1])

                equation = _textbook_root_equation(exact_v, ratio, n, kind)
                # The float64 root is within a few ulp, far inside 1e-11 and far
                # from its partner, which is at least 4e-4 away at these V.
                half = mpmath.mpf(found) * mpmath.mpf("1e-11")
                bracket = (mpmath.mpf(found) - half, mpmath.mpf(found) + half)
                assert equation(bracket[0]) * equation(bracket[1]) < 0
                reference = mpmath.findroot(equation, bracket, solver="anderson")
                if kind in ("HE", "EH"):
                    _, a, b, coupling = _textbook(reference, exact_v, ratio, n, kind)
                    radical = mpmath.sqrt(((1 - ratio) * b / 2) ** 2 + coupling)
                    branches = {
                        "HE": -(1 + ratio) * b / 2 - radical,
                        "EH": -(1 + ratio) * b / 2 + radical,
                    }
                    assert min(branches, key=lambda one: abs(a - branches[one])) == kind
                assert _textbook_cutoff(kind, n, m, ratio) < reference
                assert reference <= min(
                    _textbook_cutoff(kind, n, m + 1, ratio), exact_v
                )
                index = math.sqrt(
                    core**2 - (found * math.sqrt(core**2 - cladding**2) / v) ** 2
                )
                expected = mpmath.sqrt(mpmath.mpf(core) ** 2 - (reference / scale) ** 2)
                # The scalar solves stop within 4 ulp of the root (xtol and
                # rtol), so 8 machine epsilons bound u and the index derived
                # from it; measured worst 3.0 and 1.0 epsilons.
                assert_allclose(
                    found, float(reference), rtol=8 * np.finfo(float).eps, atol=0
                )
                assert_allclose(
                    index, float(expected), rtol=8 * np.finfo(float).eps, atol=0
                )
