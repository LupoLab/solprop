# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's dispersive response on inputs that make nothing, on native (SEP 43, 38, 39).

**A member at exactly zero density, and a vacuum.** The Numba stage built the mixture
filter from densities ``(3e25, 0, 1e24)`` and a vacuum's ``(0, 0, 0)``, where the
filter is one and the output multipliers exactly zero. Native's rows are the C++ test's
(``DispersiveMixture.TheRowsAreTheReferenceArithmetic``, a zero factor and a zero
total); here they are reached by a real run whose partial pressures fall to zero along
the guide, and compared with ``numpy_ref``.

**A chunk without a gas node.** The node chunk is cut at the first node of the gas
region, so the first chunk holds solid-core nodes alone: no product is formed or landed,
and the evaluation still agrees with the reference.

**Weaker than the original.** The mixture filter's rows and the rule that the same
densities leave them unrebuilt aren't read directly; the source they make is. A native
stage's input row after a vacuum isn't visible, so the exact unit filter stays on the
C++ test.
"""

import warnings

import numpy as np
import pytest

from backends.dispersive_cases import LENGTH, POLICIES, STEPS
from backends.dispersive_route_cases import (
    ARGON_FILL,
    disagreement,
    evaluation_bound,
    general_problem,
    strand,
)
from backends.dispersive_route_cases import run as march
from backends.dispersive_route_cases import source as launch_source
from bounds import assert_within, step_evaluations
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref import backend as numpy_ref_backend
from solprop.driver import Simulation
from solprop.exceptions import GeneralModalBasisWarning, MaterialRangeWarning
from solprop.physdata import MaterialSelection, Mixture
from solprop.profiles import Linear
from solprop.stepper import FixedSchedule
from solprop.translate.stage import scalars

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]

solprop_native = pytest.importorskip("solprop_native")
NativeBackend = solprop_native.NativeBackend
native_backend = pytest.importorskip("solprop_native.backend")


def falling_mixture(evaluator, helium_end, argon_end):
    """Return helium and argon whose partial pressures fall along the guide, stepped."""
    fill = Mixture.partial_pressures(
        {
            "He": Linear(2.0, helium_end, LENGTH),
            "Ar": Linear(0.5, argon_end, LENGTH),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return Simulation.from_capillary(
            75e-6,
            LENGTH,
            fill,
            None,
            wavelength=800e-9,
            wavelength_limits=(150e-9, 4e-6),
            trange=150e-15,
            fwhm=10e-15,
            energy=30e-6,
            modes=3,
            materials=MaterialSelection(electronic=POLICIES[evaluator]),
            plasma=False,
            raman=False,
            backend="numpy_ref",
            n_save=2,
            absorber_scheme="split_step",
            schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / STEPS),
            noise=None,
        )


def source_at(backend_class, problem, grid, z):
    """Return one evaluation of the lab-frame source on the launch at position `z`."""
    backend = backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        return np.array(backend._adams_lab_source(problem.initial.field0, z))
    finally:
        backend.close()


def from_capillary(evaluator, **ends):
    simulation = falling_mixture(evaluator, **ends)
    return simulation.translate(), simulation.grid


@pytest.mark.parametrize("evaluator", sorted(POLICIES))
def test_a_member_at_exactly_zero_density_is_the_reference_s(evaluator):
    """Helium's partial pressure reaches an exact zero; argon carries on alone there."""
    problem, grid = from_capillary(evaluator, helium_end=0.0, argon_end=2.0)
    densities = scalars(problem, 0.0, LENGTH).dispersive_kerr_densities
    assert densities[0] == 0.0
    assert densities[1] > 0.0
    ours = source_at(NativeBackend, problem, grid, LENGTH)
    theirs = source_at(NumpyRefBackend, problem, grid, LENGTH)
    assert disagreement(ours, theirs) < evaluation_bound(problem)
    # The whole march down to it, at the steps' count of one evaluation.
    limit = STEPS * step_evaluations(problem) * evaluation_bound(problem)
    ending = march(NativeBackend, problem, grid)[0]
    reference = march(NumpyRefBackend, problem, grid)[0]
    assert_within(ending, reference, limit, what="the march to a zero member")


@pytest.mark.parametrize("evaluator", sorted(POLICIES))
def test_a_vacuum_has_exactly_no_source(evaluator):
    """Every member at an exact zero: multipliers exactly zero, and no output (class P).

    A vacuum's filter is a unit one by contract, the C++ test's explicit no-op; its
    source, the multiplier's product, must be an exact zero on both, not a rounding.
    """
    problem, grid = from_capillary(evaluator, helium_end=0.0, argon_end=0.0)
    densities = scalars(problem, 0.0, LENGTH).dispersive_kerr_densities
    assert np.all(np.asarray(densities) == 0.0)
    ours = source_at(NativeBackend, problem, grid, LENGTH)
    theirs = source_at(NumpyRefBackend, problem, grid, LENGTH)
    np.testing.assert_array_equal(theirs, 0.0)
    np.testing.assert_array_equal(ours, 0.0)
    # Vacuous unless the same evaluation makes one where there is gas.
    assert np.any(source_at(NativeBackend, problem, grid, 0.0))


@pytest.mark.filterwarnings(
    f"ignore::{GeneralModalBasisWarning.__module__}.{GeneralModalBasisWarning.__name__}"
)
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
def test_a_general_chunk_without_a_gas_node_adds_nothing(monkeypatch, evaluator):
    """A chunk of the solid core's nodes alone, on native as on the reference.

    The original's construction, the chunk cut at the first gas node; native reads it
    from the same function through its own import, so both are patched. With no
    regional tensor in the core, the chunk's polarisation is zero and the evaluation
    still agrees.
    """
    problem, grid = general_problem(strand(ARGON_FILL), evaluator, "real")
    tables = problem.transform.modal_tables
    assert tables is not None
    region = np.asarray(tables.region_index)
    first_gas = int(np.flatnonzero(region == 1)[0])
    assert first_gas >= 1
    assert np.all(region[:first_gas] != 1)
    for module in (numpy_ref_backend, native_backend):
        monkeypatch.setattr(module, "general_node_chunk", lambda *_, **__: first_gas)
    ours = launch_source(NativeBackend, problem, grid)
    theirs = launch_source(NumpyRefBackend, problem, grid)
    assert np.any(theirs)
    assert disagreement(ours, theirs) < evaluation_bound(problem)
