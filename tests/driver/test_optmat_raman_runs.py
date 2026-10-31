# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""optmat Raman responses through the drivers, the translation and the manifest.

The runs take ten fixed steps of a 0.4 mm capillary on ``numpy_ref``, as SEP 3's
baseline witnesses do: what is tested is what the run carries, translates and
records, not its physics.
"""

import math
from typing import Any, cast

import numpy as np
import pytest

from solprop.boundaries import absorber
from solprop.constants import ROOM_TEMPERATURE
from solprop.driver import Simulation
from solprop.driver.material_output import _policies, _raman_entry, _window
from solprop.exceptions import MaterialError, TranslationError
from solprop.grid import FieldGrid
from solprop.nonlinear.optmat_raman import OptmatRamanResponse, gas_raman
from solprop.nonlinear.raman import RamanResponse
from solprop.physdata import Mixture
from solprop.physdata.raman_types import MolecularOptions
from solprop.physdata.selection import MaterialSelection
from solprop.profiles import Constant
from solprop.stepper import FixedSchedule
from solprop.translate.chebyshev import DEFAULT_TOL
from solprop.translate.linop import profile_table
from solprop.translate.raman import MAX_DECAY_EXPONENT
from solprop.translate.responses import (
    _MaterialProfiles,
    _raman_density,
    _ResponseBuild,
)

# The gas fits warn beyond their qualified band on these windows, and N2 or H2 at the
# fills here outside the Raman responses' validated densities; the unit tests assert
# those warnings themselves.
pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

SELECTION = MaterialSelection()
SHIFTED = MaterialSelection(raman_shift="reference-density")
LENGTH = 4e-4
STEPS = 10


def capillary(
    gas: Any, materials: MaterialSelection | None = SELECTION, **changes: Any
) -> Any:
    """A ten-step capillary with Raman on numpy_ref under a selection."""
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (300e-9, 3e-6), 400e-15)
    options = {
        "pressure": 1.0,
        "wavelength": 800e-9,
        "grid": grid,
        "absorber": absorber(grid, boundary_length=1.0),
        "fwhm": 30e-15,
        "energy": 1e-6,
        "plasma": False,
        "raman": True,
        "n_r": 8,
        "schedule": FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        "n_save": 2,
        "backend": "numpy_ref",
        "materials": materials,
    } | changes
    return Simulation.from_capillary(50e-6, LENGTH, gas, **{"noise": None, **options})


def the_raman(simulation: Any) -> Any:
    """The run's one Raman response."""
    (response,) = [r for r in simulation.responses if isinstance(r, RamanResponse)]
    return response


@pytest.mark.medium
def test_ten_steps_of_a_nitrogen_capillary_under_a_selection():
    simulation = capillary("N2")
    response = the_raman(simulation)
    assert isinstance(response, OptmatRamanResponse)
    assert response.assembly is not None
    assert response.assembly.terms == 72
    result = simulation.run()
    assert np.all(np.isfinite(result.fields("fft")))
    problem = simulation.translate()
    (leaf,) = problem.responses.raman_terms
    assert leaf.n_terms == 72
    assert leaf.tensor_capable
    manifest = simulation.metadata().materials
    assert "raman_response/N2@293.15K,800nm" in [r.name for r in manifest.records]
    assert manifest.policies["raman_reference_temperature"] == ROOM_TEMPERATURE
    assert manifest.policies["raman_terms"] == [
        {"material": "N2", "labels": ["N2"], "groups": 43, "terms": 72, "shift": None}
    ]
    (entry,) = manifest.raman
    assert entry["groups"] == 43
    assert entry["shift_phase_bound_rad"] is None
    assert entry["window_s"] == pytest.approx(400e-15, rel=0.1)


@pytest.mark.medium
def test_ten_steps_of_an_air_capillary_under_a_selection():
    simulation = capillary({"N2": 0.7808, "O2": 0.2095, "Ar": 0.0097})
    response = the_raman(simulation)
    assert response.labels == ("N2", "O2", "Ar")
    assert response.assembly.groups == 82
    result = simulation.run()
    assert np.all(np.isfinite(result.fields("fft")))
    (leaf,) = simulation.translate().responses.raman_terms
    assert leaf.label == "N2"
    assert leaf.n_terms == 126


def test_a_noble_gas_capillary_with_raman_is_refused_by_name():
    with pytest.raises(MaterialError, match="it is an atom"):
        capillary("Ar")


def test_a_hydrogen_fill_at_constant_pressure_holds_its_shift_exactly():
    response = the_raman(capillary("H2", pressure=2.0))
    assert response.assembly.shift.policy == "constant-density"


def test_the_composition_is_driven_and_damped_by_its_summed_density():
    # A dry-air gradient: the leaf's amplitude table and every decay table are the
    # composition's summed density and optmat's group laws evaluated on it.
    simulation = capillary(
        Mixture.air((1.5, 0.3), length=LENGTH, selection=SELECTION), pressure=None
    )
    response = the_raman(simulation)
    problem = simulation.translate()
    leaf: Any = problem.responses.raman_terms[0]
    mixture = simulation.modes[0].mixture
    z = np.linspace(0.0, LENGTH, 41)
    total = np.sum(mixture.density(z), axis=0)
    density = np.array([float(leaf.density(one)) for one in z])
    # The tables' target relative accuracy (DEFAULT_TOL), with room for rounding.
    np.testing.assert_allclose(density, total, rtol=100 * DEFAULT_TOL)
    cap = MAX_DECAY_EXPONENT / simulation.grid.dt_fine
    for term, table in zip(response.tensor_terms, leaf.decay_rates, strict=True):
        direct = np.array([min(term.model.dephasing.decay_rate(n), cap) for n in total])
        tabulated = np.array([float(table(one)) for one in z])
        np.testing.assert_allclose(
            tabulated,
            direct,
            rtol=100 * DEFAULT_TOL,
            atol=1e-12 * float(np.max(direct)),
        )


def test_a_composition_naming_a_label_the_material_lacks_is_refused():
    response = gas_raman("N2", selection=MaterialSelection(carrier_wavelength=8e-7))
    composite = type(response)(
        tensor_terms=response.tensor_terms, labels=("N2", "O2"), species="N2"
    )
    path = Constant(1e25)
    material = _MaterialProfiles(("N2",), ("N2",), (path,), path)
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (300e-9, 3e-6), 400e-15)
    build = _ResponseBuild(grid, LENGTH, DEFAULT_TOL, None)
    densities = {"N2": profile_table(path, 0.0, LENGTH)}
    with pytest.raises(TranslationError, match=r"names labels \['O2'\]"):
        _raman_density(composite, "N2", material, densities, build)
    single = type(response)(tensor_terms=response.tensor_terms, labels=("N2",))
    summed, table = _raman_density(single, "N2", material, densities, build)
    assert summed is path
    assert table is densities["N2"]


def test_hydrogen_on_a_gradient_records_its_phase_bound_over_the_window():
    gradient = Mixture.fill({"H2": 1.0}, (3.0, 1.5), length=LENGTH, selection=SHIFTED)
    with pytest.raises(MaterialError, match="raman_shift='reference-density'"):
        capillary(gradient, pressure=None)
    simulation = capillary(gradient, pressure=None, materials=SHIFTED)
    response = the_raman(simulation)
    shift = response.assembly.shift
    assert shift.policy == "reference-density"
    manifest = simulation.material_manifest()
    (entry,) = manifest.raman
    window = _window(simulation.grid.omega)
    assert entry["window_s"] == window
    assert entry["shift_phase_bound_rad"] == pytest.approx(
        shift.frequency_error * window
    )
    assert manifest.policies["raman_terms"][0]["shift"] == "reference-density"
    assert manifest.policies["raman_shift"] == "reference-density"


def test_an_updated_fill_keeps_the_optmat_response():
    simulation = capillary({"N2": 0.8, "O2": 0.2})
    mixture = simulation.modes[0].mixture
    doubled = {
        c.label: Constant(2.0 * float(c.density(np.array([0.0]))[0]))
        for c in mixture.components
    }
    updated = simulation.with_mixture(mixture.with_densities(doubled))
    assert isinstance(the_raman(updated), OptmatRamanResponse)


def test_the_window_and_the_records_of_a_bare_response():
    assert _window(np.array([1.0])) is None
    assert _window(np.array([3.0, 1.0, 2.0])) == pytest.approx(2.0 * math.pi)
    response = gas_raman("N2", selection=MaterialSelection(carrier_wavelength=8e-7))
    bare = type(response)(tensor_terms=response.tensor_terms, labels=("N2",))
    assert _raman_entry(bare, np.array([1.0, 2.0])) == {"labels": ["N2"]}
    carrier = MaterialSelection(carrier_wavelength=8e-7)
    warm = gas_raman("O2", selection=carrier)
    hot = gas_raman("O2", MolecularOptions(temperature=400.0), selection=carrier)
    policies = _policies(
        None, dilute=False, carrier=None, gnlse=None, raman=[warm, hot, bare]
    )
    assert policies["raman_reference_temperature"] == [ROOM_TEMPERATURE, 400.0]
    assert len(cast("list[object]", policies["raman_terms"])) == 2
    assert (
        _policies(None, dilute=False, carrier=None, gnlse=None)[
            "raman_reference_temperature"
        ]
        is None
    )
