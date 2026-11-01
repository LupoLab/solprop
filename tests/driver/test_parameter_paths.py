# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The parameter a keyword route's refusal or value warning names (SEP 41, S10).

Each case is one refused ``Simulation.from_gnlse`` or ``Simulation.from_capillary``
call: the exception keeps its class and message, and its ``parameter`` is the path of
the keyword whose value is refused, or the tuple of keywords in conflict. Some refusals
are raised inside an object the route builds from a keyword (a pulse, a grid, a fill),
which names its own argument; the route renames the ones it spells otherwise. Nothing
is run but the boundary monitor's: every refusal comes while the simulation is built.
"""

import warnings
from collections.abc import Callable
from typing import Any, cast

import pytest

from driver.test_tdse_entry import _capillary, _sae
from solprop.driver import Simulation
from solprop.exceptions import (
    BoundaryError,
    BoundaryWarning,
    DriverError,
    GridError,
    MaterialError,
    MaterialRangeWarning,
    MaterialStatusWarning,
    ModeError,
    ParameterPath,
    PulseError,
    SolpropError,
    SolpropWarning,
    StepperError,
)
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import CoherentWall
from solprop.modes.capillary import MarcatiliMode
from solprop.nonlinear.tdse import TDSEExecution
from solprop.physdata import Mixture
from solprop.physdata.selection import MaterialSelection
from solprop.pulses import GaussianPulse
from test_monitor import TRANGE, walker

BETAS: tuple[Any, ...] = (0.11, 0.01, [0.0, 0.0, -1.183e-26])
FILL: tuple[Any, ...] = (100e-6, 0.01, "He", 1.0)
GNLSE = {
    "wavelength": 835e-9,
    "wavelength_limits": (500e-9, 2e-6),
    "trange": 1e-12,
    "width": 100e-15,
    "peak_power": 9.8,
    "shape": "sech",
    "n_save": 3,
    "noise": None,
    "backend": "numpy_ref",
}
CAPILLARY = {
    "wavelength": 800e-9,
    "fwhm": 30e-15,
    "energy": 1e-6,
    "wavelength_limits": (400e-9, 2e-6),
    "trange": 200e-15,
    "n_save": 2,
    "noise": None,
    "backend": "numpy_ref",
}


def _gnlse(*arguments: Any, **changes: Any) -> Simulation:
    return Simulation.from_gnlse(*(arguments or BETAS), **{**GNLSE, **changes})


def _hollow(*arguments: Any, **changes: Any) -> Simulation:
    return Simulation.from_capillary(*(arguments or FILL), **{**CAPILLARY, **changes})


def _parameters(records: pytest.WarningsRecorder) -> list[ParameterPath | None]:
    return [cast("SolpropWarning", record.message).parameter for record in records]


REFUSALS = [
    # The gnlse route's own refusals.
    pytest.param(
        lambda: _gnlse(trange=None),
        DriverError,
        "trange",
        "give wavelength_limits and trange",
        id="gnlse-no-window",
    ),
    pytest.param(
        lambda: _gnlse(trange=None, wavelength_limits=None),
        DriverError,
        ("wavelength_limits", "trange"),
        "give wavelength_limits and trange",
        id="gnlse-no-grid",
    ),
    pytest.param(
        lambda: _gnlse(
            wavelength_limits=None,
            trange=None,
            grid=FieldGrid.from_limits(0.01, 835e-9, (500e-9, 2e-6), 1e-12),
        ),
        DriverError,
        "grid",
        "propagates an envelope grid",
        id="gnlse-field-grid",
    ),
    pytest.param(
        lambda: _gnlse(beta=lambda omega: 0.0 * omega),
        DriverError,
        ("betas", "beta", "fibre"),
        "exactly one of betas, beta",
        id="gnlse-dispersion-twice",
    ),
    pytest.param(
        lambda: _gnlse(n2=2e-20),
        DriverError,
        ("gamma", "n2"),
        "exactly one of gamma and n2",
        id="gnlse-gamma-and-n2",
    ),
    pytest.param(
        lambda: _gnlse(
            0.11, 0.01, None, fibre=MarcatiliMode(100e-6), loss_db_per_m=1.0
        ),
        DriverError,
        ("fibre", "loss_db_per_m"),
        "the fibre carries its loss",
        id="gnlse-fibre-loss",
    ),
    pytest.param(
        lambda: _gnlse(raman_fraction=2.0),
        ModeError,
        "raman_fraction",
        "must lie in",
        id="gnlse-raman-fraction",
    ),
    pytest.param(
        lambda: _gnlse(raman=True, raman_fraction=0.0),
        ModeError,
        ("raman", "raman_fraction"),
        "carries none",
        id="gnlse-raman-none",
    ),
    pytest.param(
        lambda: _gnlse(shape="square"),
        DriverError,
        "shape",
        "shape must be",
        id="gnlse-shape",
    ),
    pytest.param(
        lambda: _gnlse(shape="gauss"),
        DriverError,
        ("width", "shape"),
        "intensity fwhm",
        id="gnlse-gaussian-width",
    ),
    pytest.param(
        lambda: _gnlse(
            pulses=[GaussianPulse(wavelength=835e-9, fwhm=50e-15, energy=1e-12)],
            shape="gauss",
            width=None,
        ),
        DriverError,
        ("pulses", "peak_power"),
        "describes every input pulse",
        id="gnlse-pulses-beside",
    ),
    pytest.param(
        lambda: _gnlse(n_save=1), DriverError, "n_save", "at least 2", id="gnlse-n-save"
    ),
    pytest.param(
        lambda: _gnlse(save_planes=[0.005, 0.01]),
        DriverError,
        ("save_planes", "n_save"),
        "not both",
        id="gnlse-planes-twice",
    ),
    pytest.param(
        lambda: _gnlse(save_planes=[0.01, 0.005], n_save=None),
        DriverError,
        "save_planes",
        "positive and increasing",
        id="gnlse-planes-order",
    ),
    pytest.param(
        lambda: _gnlse(absorber=False, absorber_scheme="split_step"),
        DriverError,
        ("absorber", "absorber_scheme"),
        "switches the temporal collar off",
        id="gnlse-absorber",
    ),
    pytest.param(
        lambda: _gnlse(seed=3),
        DriverError,
        ("seed", "noise"),
        "requires a noise model",
        id="gnlse-seed",
    ),
    # Raised by an object the gnlse route builds from one keyword's value.
    pytest.param(
        lambda: _gnlse(fwhm=50e-15),
        PulseError,
        ("fwhm", "width"),
        "only one of fwhm or width",
        id="gnlse-pulse-duration",
    ),
    pytest.param(
        lambda: _gnlse(energy=1e-9),
        PulseError,
        ("energy", "peak_power"),
        "only one of energy or peak_power",
        id="gnlse-pulse-scale",
    ),
    pytest.param(
        lambda: _gnlse(phases=[float("nan")]),
        PulseError,
        "phases",
        "spectral phase coefficient",
        id="gnlse-pulse-phase",
    ),
    pytest.param(
        lambda: _gnlse(pulses=[], shape="gauss", width=None, peak_power=None),
        PulseError,
        "pulses",
        "at least one pulse",
        id="gnlse-pulse-sum",
    ),
    pytest.param(
        lambda: _gnlse(wavelength_limits=(2e-6, 500e-9)),
        GridError,
        "wavelength_limits",
        "strictly increasing pair",
        id="gnlse-grid-limits",
    ),
    pytest.param(
        lambda: _gnlse(wavelength=-835e-9),
        GridError,
        "wavelength",
        "reference_wavelength must be positive",
        id="gnlse-grid-wavelength",
    ),
    pytest.param(
        lambda: _gnlse(rtol=-1.0),
        StepperError,
        "rtol",
        "rtol must be positive",
        id="gnlse-tolerance",
    ),
    # The capillary route's own refusals.
    pytest.param(
        lambda: _hollow(100e-6, 0.01, "N2", 1.0, raman=True, kerr=False),
        DriverError,
        ("raman", "kerr"),
        "needs kerr=True",
        id="capillary-raman-kerr",
    ),
    pytest.param(
        lambda: _hollow(polarisation="x"),
        DriverError,
        "polarisation",
        "this path is scalar",
        id="capillary-polarisation",
    ),
    pytest.param(
        lambda: _hollow(modes=0),
        DriverError,
        "modes",
        "at least one mode",
        id="capillary-modes",
    ),
    pytest.param(
        lambda: _hollow(modes=2, mode_average=True),
        DriverError,
        ("mode_average", "modes"),
        "needs exactly one mode",
        id="capillary-averaged-modes",
    ),
    pytest.param(
        lambda: _hollow(mode_average=True, n_r=8),
        DriverError,
        ("mode_average", "n_r"),
        "given beside mode_average",
        id="capillary-averaged-rule",
    ),
    pytest.param(
        lambda: _hollow(n_theta=4),
        DriverError,
        "n_theta",
        "azimuthal order zero",
        id="capillary-azimuthal",
    ),
    pytest.param(
        lambda: _hollow(envelope=True, plasma=True, thg=True),
        DriverError,
        ("envelope", "plasma", "thg"),
        "cannot be combined with THG",
        id="capillary-envelope-thg",
    ),
    pytest.param(
        lambda: _hollow(plasma=False, plasma_integration="band"),
        DriverError,
        ("plasma_integration", "plasma"),
        "the plasma is off",
        id="capillary-integration-off",
    ),
    pytest.param(
        lambda: _hollow(kerr=False, plasma=False),
        DriverError,
        ("kerr", "plasma"),
        "no nonlinearity",
        id="capillary-no-nonlinearity",
    ),
    pytest.param(
        lambda: _hollow(
            grid=EnvelopeGrid.from_limits(0.01, 800e-9, (400e-9, 2e-6), 200e-15),
            trange=None,
        ),
        DriverError,
        ("grid", "wavelength_limits"),
        "not both",
        id="capillary-grid-twice",
    ),
    pytest.param(
        lambda: _hollow(wavelength_limits=None),
        DriverError,
        "wavelength_limits",
        "needs wavelength_limits and trange",
        id="capillary-no-limits",
    ),
    pytest.param(
        lambda: _hollow(100e-6, 0.01, "He", (1.0, 0.5, 0.2)),
        DriverError,
        "pressure",
        "pressure must be a number",
        id="capillary-pressure",
    ),
    pytest.param(
        lambda: _hollow(
            100e-6, 0.01, Mixture.fill({"He": 1.0}, 1.0), 1.0, pressure_floor=1e-3
        ),
        DriverError,
        ("gas", "pressure", "pressure_floor"),
        "carries its pressure paths",
        id="capillary-mixture-pressure",
    ),
    pytest.param(
        lambda: _hollow(preionisation={"Xe": 0.1}),
        DriverError,
        'preionisation["Xe"]',
        "unknown preionisation",
        id="capillary-preionisation-label",
    ),
    pytest.param(
        lambda: _hollow(wall=CoherentWall(), taper_coupling=False),
        DriverError,
        ("wall", "taper_coupling"),
        "declines the taper",
        id="capillary-wall-taper",
    ),
    pytest.param(
        lambda: _hollow(wall=CoherentWall(), loss=False),
        DriverError,
        ("wall", "loss"),
        "is the loss model",
        id="capillary-wall-loss",
    ),
    pytest.param(
        lambda: _hollow(modes=[MarcatiliMode(50e-6)]),
        DriverError,
        ("radius", "modes[0]"),
        "radius disagrees",
        id="capillary-mode-radius",
    ),
    pytest.param(
        lambda: _hollow(modes=[MarcatiliMode(100e-6)], model="reduced"),
        DriverError,
        ("model", "modes[0]"),
        "disagrees with supplied mode 0",
        id="capillary-mode-model",
    ),
    pytest.param(
        lambda: _hollow(materials=MaterialSelection(carrier_wavelength=1030e-9)),
        DriverError,
        ("materials.carrier_wavelength", "wavelength"),
        "carrier_wavelength is",
        id="capillary-carrier",
    ),
    pytest.param(
        lambda: _hollow(tdse_execution=TDSEExecution()),
        DriverError,
        ("tdse_execution", "tdse"),
        "requires a TDSEGasResponse",
        id="capillary-tdse-execution",
    ),
    pytest.param(
        lambda: _capillary(tdse=_sae(), plasma=True),
        DriverError,
        ("tdse", "plasma"),
        "owns the electronic response",
        id="capillary-tdse-plasma",
    ),
    # Raised by an object the capillary route builds from one keyword's value.
    pytest.param(
        lambda: _hollow(100e-6, 0.01, "Xx", 1.0),
        MaterialError,
        "gas",
        "unknown gas",
        id="capillary-gas",
    ),
    pytest.param(
        lambda: _hollow(temperature=-5.0),
        MaterialError,
        "temperature",
        "must be finite and positive",
        id="capillary-temperature",
    ),
    pytest.param(
        lambda: _hollow(raman=True),
        MaterialError,
        "raman",
        "no molecular Raman response",
        id="capillary-raman-atom",
    ),
    pytest.param(
        lambda: _hollow(fwhm=-1.0),
        PulseError,
        "fwhm",
        "finite positive number",
        id="capillary-pulse-duration",
    ),
    pytest.param(
        lambda: _hollow(trange=-1e-15),
        GridError,
        "trange",
        "trange must be a positive",
        id="capillary-grid-trange",
    ),
]


@pytest.mark.parametrize(("build", "kind", "parameter", "match"), REFUSALS)
def test_a_refusal_names_the_keyword_it_refuses(
    build: Callable[[], object],
    kind: type[SolpropError],
    parameter: ParameterPath,
    match: str,
) -> None:
    with warnings.catch_warnings():
        # The capillary's helium index is asked for beyond its fit; that warning says
        # nothing about the refusal under test.
        warnings.simplefilter("ignore", MaterialRangeWarning)
        with pytest.raises(kind, match=match) as caught:
            build()
    assert type(caught.value) is kind
    assert caught.value.parameter == parameter
    # Which rule each refusal enforces, if any, is tests/test_capability_queries.py's.


def test_an_error_raised_outside_an_interface_names_no_parameter() -> None:
    assert DriverError("refused").parameter is None
    with pytest.raises(PulseError, match="super-Gaussian order") as caught:
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6, m=0)
    assert caught.value.parameter is None


def test_a_material_status_warning_names_the_gas() -> None:
    # SF6 has no calculated or measured ionisation rate, so optmat's is untested.
    with pytest.warns(MaterialStatusWarning, match="SF6") as records:
        _hollow(100e-6, 0.01, "SF6", 1.0)
    assert _parameters(records) == ["gas"]


def test_an_error_filter_still_raises_the_named_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", MaterialStatusWarning)
        with pytest.raises(MaterialStatusWarning, match="SF6") as caught:
            _hollow(100e-6, 0.01, "SF6", 1.0)
    assert caught.value.parameter == "gas"


def test_an_ignore_filter_still_hides_the_named_warning() -> None:
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("ignore", MaterialStatusWarning)
        _hollow(100e-6, 0.01, "SF6", 1.0)
    assert not [one for one in shown if issubclass(one.category, MaterialStatusWarning)]


def test_a_raman_range_warning_names_the_raman_keyword() -> None:
    # optmat's hydrogen response is validated at one density, not at one bar.
    with pytest.warns(MaterialRangeWarning, match="Raman response") as records:
        _hollow(100e-6, 0.01, "H2", 1.0, raman=True)
    assert _parameters(records) == ["raman"]


def test_the_provisional_raman_warning_names_the_raman_keyword() -> None:
    with pytest.warns(SolpropWarning, match="provisionally") as records:
        _hollow(100e-6, 0.01, "N2O", 1.0)
    assert _parameters(records) == ["raman"]


def test_the_tdse_noise_warning_names_the_noise_keyword() -> None:
    with pytest.warns(SolpropWarning, match="beside the TDSE response") as records:
        _capillary(tdse=_sae(), noise="input")
    assert _parameters(records) == ["noise"]


def test_the_boundary_warning_names_the_boundary_check() -> None:
    # One window's walk-off brings the pulse to the guard inside the fibre (as in
    # tests/test_monitor.py).
    with (
        pytest.warns(BoundaryWarning, match="guard occupancy") as records,
        pytest.raises(BoundaryError, match="guards"),
    ):
        walker(1.2 * TRANGE).run(progress="none")
    assert _parameters(records) == ["boundary_check"]
