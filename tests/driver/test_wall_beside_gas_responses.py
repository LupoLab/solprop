# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A coherent wall beside the fill's own plasma and Raman responses.

``prop_capillary`` turns the photoionisation and the molecular Raman response on by
default (``plasma=None`` and ``raman=None`` take ``not envelope`` and "every species
with data"), while the other coherent-wall tests pass ``plasma=False,
raman=False``, so the combination the entry point gives a user by default would
otherwise run on no backend in the suite. Here it runs, through the
entry point with those two arguments **omitted**, on nitrogen -- a molecule with both
optmat's Raman response and its molecular rate -- for the scalar wall, the TE/TM wall
and a taper:

* the compiled backends' energy is ``numpy_ref``'s within a ten-step bound written
  first (``wall_gas_cases.parity_bound``), which extends the flow bound of
  ``tests/flow_parity.py`` by the Raman and plasma sources' own rounding, and their
  records carry the same flow;
* the record names the three responses that ran.

The last plane's field within that bound is a row of the case catalogue
(``tests/backends/flow_rows.py``, ``wall beside N2 ...``), one per case and response:
switching the plasma, or the Raman response, off must move the reference by more than
``parity_ladder.DETECTION`` of the bound, so that the parity is about a run in which
both act.

The launch, 30 uJ in 30 fs through a 50 um bore of nitrogen at one bar, was chosen
before any parity measurement from the (then default) MO-ADK rate alone: its peak
field, about 1.8e10 V/m on axis, ionises about 1e-6 of the gas over the pulse, a
plasma phase of order 1e-5 rad over the fifth of a millimetre -- far above rounding
and far below the strong-field regime in which a launch's rounding is amplified by
orders of magnitude. The Kerr phase is a few milliradians.

**Ten fixed steps over a fifth of a millimetre**; nothing here propagates further.
"""

from typing import cast

import numpy as np
import pytest

from flow_parity import (
    LENGTH,
    QUIET,
)
from solprop.backends.registry import available
from solprop.driver import prop_capillary
from solprop.nonlinear.optmat_raman import OptmatRamanResponse
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.output.memory import MemoryOutput
from wall_gas_cases import (
    CASES,
    COMPILED,
    GAS,
    arguments,
    parity_bound,
)

pytestmark = [
    pytest.mark.filterwarnings(f"ignore::{category.__module__}.{category.__name__}")
    for category in QUIET
] + [
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
]
"""The qualification cases' deliberate warnings (``flow_parity.QUIET``)."""


def entry_point(name, **overrides) -> MemoryOutput:
    """Run one case through ``prop_capillary`` and return the handler."""
    keywords = arguments(name, **overrides)
    radius = keywords.pop("radius")
    return cast(
        "MemoryOutput",
        prop_capillary(radius, LENGTH, GAS, 1.0, **{"noise": None, **keywords}),
    )


@pytest.fixture(scope="module")
def reference():
    """Each case's ``numpy_ref`` run through the entry point, once for the module."""
    runs = {}

    def get(name):
        if name not in runs:
            runs[name] = entry_point(name)
        return runs[name]

    return get


def last_field(output: MemoryOutput) -> np.ndarray:
    """Return the last plane's field."""
    field = output.planes()[-1].field
    assert field is not None
    return field


def backend_parameters():
    """Return the (case, backend) pairs, all but the scalar wall's in the slow tier."""
    pairs = []
    for name in CASES:
        for backend in COMPILED:
            marks = [pytest.mark.medium if name == "scalar" else pytest.mark.slow]
            if backend not in available():
                marks.append(pytest.mark.skip(reason=f"{backend} is not installed"))
            pairs.append(
                pytest.param(name, backend, marks=marks, id=f"{name}-{backend}")
            )
    return pairs


@pytest.mark.parametrize(("name", "backend"), backend_parameters())
def test_the_default_responses_beside_a_wall_keep_the_energy_and_the_flow_record(
    reference, name, backend, record_property
):
    """Ten steps through ``prop_capillary`` with the default plasma and Raman.

    The energy statistic of the compiled backend's last plane is the reference's
    within :func:`wall_gas_cases.parity_bound`, derived and stated before the
    measurement, and the two records carry the same flow entry. The last plane's field
    within the same bound, guarded by the plasma and by the Raman response, is a row
    of the case catalogue (``tests/backends/flow_rows.py``, ``wall beside N2 ...``).
    """
    expected = reference(name)
    got = entry_point(name, backend=backend)
    bound = parity_bound(name, backend, last_field(expected))
    record_property("bound", bound)
    assert bound > 0.0
    energies = [
        float(np.sum(output.planes()[-1].stats["energy"])) for output in (expected, got)
    ]
    assert energies[1] == pytest.approx(energies[0], rel=bound, abs=0.0)
    records = [output.metadata for output in (expected, got)]
    assert all(one is not None and one.modal is not None for one in records)
    assert records[0].modal["flow"] == records[1].modal["flow"]  # type: ignore[union-attr]


def test_the_record_names_the_responses_that_ran(reference):
    """The record lists Kerr, the plasma and the Raman response, by type.

    ``prop_capillary`` was given neither ``plasma`` nor ``raman``; what it turned on
    is what the record says, the plasma with the rate it chose for a molecule.
    """
    metadata = reference("scalar").metadata
    assert metadata is not None
    responses = metadata.configuration["responses"]
    kinds = [type(one) for one in responses]  # type: ignore[union-attr]
    # optmat's Raman response and rate since SEP 3's Phase 4a (solprop's own
    # RamanResponse and isotropic MO-ADK before, deleted at Phase 4b).
    assert kinds == [KerrResponse, PlasmaResponse, OptmatRamanResponse]
    plasma = responses[1]  # type: ignore[index]
    assert type(plasma.rate).__name__ == "OptmatRate"  # type: ignore[union-attr]
    assert metadata.modal is not None and "flow" in metadata.modal
