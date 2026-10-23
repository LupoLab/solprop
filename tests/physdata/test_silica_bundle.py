# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""optmat's silica nonlinear bundle: total n2, f_R and h_R (SEP 3, section 4.5)."""

import math
from typing import Any

import numpy as np
import pytest
from optmat.solid.bundle import silica_bundle
from optmat.solid.raman_response import silica_raman_response

from solprop.constants import EPS0, C
from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.nonlinear.gnlse import kerr_susceptibility, raman_susceptibility
from solprop.physdata import _optmat
from solprop.physdata.glasses import SilicaBundle, fused_silica
from solprop.physdata.selection import RAMAN_POLICIES

# A causal shape sampled on [0, 4 ps] at 0.05 fs: Hollenbeck's slowest line decays
# within 2 ps, Lin and Agrawal's boson peak within 1 ps.
STEP = 5e-17
DELAYS = np.arange(80001) * STEP


def trapezoid(values):
    return float(np.trapezoid(values, dx=STEP))


def from_the_right(bundle, component="parallel"):
    """Sample a causal shape with its right-hand limit at zero delay.

    optmat's responses are zero at t = 0 itself, and Lin and Agrawal's boson peak
    jumps to 2 f_b / tau_b there; the trapezoid takes the limit as the endpoint.
    """
    values = bundle.h_r(DELAYS, component)
    values[0] = bundle.h_r(np.array([np.finfo(float).tiny]), component)[0]
    return values


def test_the_recommended_bundles_are_optmat_default_and_tensor():
    default, tensor = SilicaBundle.resolve(), SilicaBundle.resolve(tensor=True)
    assert default.name == "silica_bundle/default"
    assert default.shape == "hollenbeck_2002"
    assert default.paired and not default.has_tensor
    assert default.raman_fraction == silica_bundle("default").raman_fraction
    assert default.raman_fraction == pytest.approx(0.2188, abs=5e-5)
    assert tensor.name == "silica_bundle/tensor" and tensor.shape == "lin_2006"
    assert tensor.has_tensor and not tensor.paired
    assert tensor.raman_fraction == 0.245
    assert SilicaBundle.resolve("blow_1989").shape == "blow_1989"
    assert SilicaBundle.resolve(silica_bundle("tensor")).digest == tensor.digest
    assert "f_R=0.2188" in repr(default)


def test_a_bundle_records_itself_its_kerr_fit_and_its_shape():
    names = [record.name for record in SilicaBundle.resolve().material_records]
    assert names == [
        "silica_bundle/default",
        "kerr/SiO2",
        "silica_raman/hollenbeck_2002",
    ]


def test_unknown_bundles_and_other_models_are_refused():
    with pytest.raises(MaterialError, match="no silica bundle 'hollenbeck'"):
        SilicaBundle.resolve("hollenbeck")
    silica: Any = fused_silica(model="recommended")
    with pytest.raises(MaterialError, match="not a silica bundle"):
        SilicaBundle.resolve(silica.model)
    bundle: Any = silica_bundle("default")
    with pytest.raises(MaterialError, match="is not silica_bundle/default"):
        SilicaBundle(bundle, digest="0" * 64)


def test_n2_is_the_kerr_fits_total_and_warns_outside_it_once():
    bundle = SilicaBundle.resolve()
    assert bundle.n2(1030e-9) == float(silica_bundle("default").n2(1030e-9))
    with pytest.raises(MaterialError, match="positive, finite wavelength"):
        bundle.n2(-1.0)
    with _optmat.material_context(), pytest.warns(MaterialRangeWarning) as caught:
        bundle.n2(300e-9)
        bundle.n2(310e-9)
    assert len([w for w in caught if issubclass(w.category, MaterialRangeWarning)]) == 1


def test_each_normalised_response_integrates_to_one():
    # The trapezoid on 0.05 fs against a unit integral. Its error is
    # dt^2 h'(0+) / 12: Lin and Agrawal's oscillator opens with a slope of
    # 0.79 (tau1^2 + tau2^2) / (tau1^2 tau2^2) = 6e27 s^-2, which gives 1.3e-6.
    for name in ("default", "tensor", "blow_1989"):
        bundle = SilicaBundle.resolve(name)
        assert trapezoid(from_the_right(bundle)) == pytest.approx(1.0, abs=3e-6), name
    tensor = SilicaBundle.resolve(tensor=True)
    parts = from_the_right(tensor, "a"), from_the_right(tensor, "b")
    np.testing.assert_allclose(
        parts[0] + parts[1], from_the_right(tensor), rtol=1e-15, atol=1e-3
    )
    # Lin and Agrawal's weights: int a = f_a = 0.75, int b = f_b + f_c = 0.25.
    assert trapezoid(parts[0]) == pytest.approx(0.75, abs=3e-6)
    assert trapezoid(parts[1]) == pytest.approx(0.25, abs=3e-6)


def test_the_hollenbeck_normalisation_agrees_with_an_independent_faddeeva_integral():
    response = silica_raman_response("hollenbeck_2002")
    p = response.parameters
    # Independently of optmat's normaliser: the unnormalised line sum integrated
    # analytically per line and numerically, which must agree.
    lines = (
        p["A"]
        * np.exp(-p["gamma"] * DELAYS[:, None])
        * np.exp(-(p["Gamma"] ** 2) * DELAYS[:, None] ** 2 / 4)
        * np.sin(p["omega"] * DELAYS[:, None])
    ).sum(axis=1)
    numerical = trapezoid(lines)
    from scipy.special import wofz  # noqa: PLC0415 - only this test needs it

    analytic = float(
        np.sum(
            p["A"]
            * math.sqrt(math.pi)
            / p["Gamma"]
            * np.imag(wofz((p["omega"] + 1j * p["gamma"]) / p["Gamma"]))
        )
    )
    assert numerical == pytest.approx(analytic, rel=1e-6)
    np.testing.assert_allclose(
        SilicaBundle.resolve().h_r(DELAYS), lines / analytic, rtol=1e-12, atol=1e-3
    )


def test_a_copolarised_only_shape_has_no_tensor_parts_by_name():
    with pytest.raises(
        MaterialError, match=r"silica_bundle/default \(hollenbeck_2002\)"
    ):
        SilicaBundle.resolve().h_r(DELAYS, "a")


@pytest.mark.filterwarnings("ignore::optmat.exceptions.ModelRangeWarning")
@pytest.mark.parametrize("name", ["default", "tensor"])
@pytest.mark.parametrize("wavelength", [532e-9, 1030e-9, 1550e-9])
def test_optmat_coefficients_carry_n0_squared_and_solprop_takes_n0_as_one(
    name, wavelength
):
    bundle = SilicaBundle.resolve(name)
    response = silica_bundle(name).response
    n2 = bundle.n2(wavelength)
    fraction = bundle.raman_fraction
    index = float(
        np.real(fused_silica(model="recommended").index(2 * math.pi * C / wavelength))
    )
    # solprop's own: the Raman integral K = f_R raman_susceptibility(n2) =
    # 2 eps0 c f_R n2 and the electronic chi_e = kerr_susceptibility(n2, f_R) =
    # (4/3) eps0 c (1 - f_R) n2, with n0 = 1 (conventions, section 4).
    raman = fraction * raman_susceptibility(n2)
    electronic = kerr_susceptibility(n2, fraction)
    assert raman == pytest.approx(2 * EPS0 * C * fraction * n2, rel=1e-15)
    # optmat's include its silica n0^2; the ratio is exactly that, which is why
    # neither is imported. The index is the same recommended model optmat reads.
    assert response.raman_coefficient(wavelength) / raman == pytest.approx(
        index**2, rel=1e-14
    )
    assert response.electronic_susceptibility(wavelength) / electronic == (
        pytest.approx(index**2, rel=1e-14)
    )
    assert index**2 == pytest.approx(2.1, abs=0.05)


def test_the_long_pulse_split_holds_the_total_and_moves_the_share():
    wavelength = 1030e-9
    default, tensor = SilicaBundle.resolve(), SilicaBundle.resolve(tensor=True)
    n2 = default.n2(wavelength)
    assert tensor.n2(wavelength) == n2
    for bundle in (default, tensor):
        fraction = bundle.raman_fraction
        # A field-domain Raman integral X acts like a Kerr term of (2/3) X, so for
        # a long pulse the two parts add to the whole of n2 for either bundle.
        total = (2 / 3) * fraction * raman_susceptibility(n2) + kerr_susceptibility(
            n2, fraction
        )
        assert total == pytest.approx(kerr_susceptibility(n2), rel=1e-15)
    # SEP 3, D11: the vector split's electronic part is 3.3 % smaller and its Raman
    # part 12 % larger than the scalar one's.
    electronic = (1 - tensor.raman_fraction) / (1 - default.raman_fraction)
    assert electronic - 1 == pytest.approx(-0.033, abs=1e-3)
    assert tensor.raman_fraction / default.raman_fraction - 1 == pytest.approx(
        0.12, abs=5e-3
    )


def test_the_gas_policy_provisional_is_refused_for_silica():
    # One Raman policy tuple for gases and silica: "recommended" is the curated
    # default of both, and "provisional", a gas's, is refused by name here.
    assert "provisional" in RAMAN_POLICIES
    with pytest.raises(MaterialError, match="'provisional' admits a gas's"):
        SilicaBundle.resolve("provisional")
