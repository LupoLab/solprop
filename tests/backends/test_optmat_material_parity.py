# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""optmat's Kerr, wall and silica materials on the compiled backends (SEP 3, Phase 2).

A helium/argon mixture with each population's carrier Kerr coefficient (Phase 1b); an
argon capillary walled by optmat's silver table; the GNLSE with fused silica's bundle,
whose sampled row has no decay path; and the tensor bundle on two polarisations, whose
isotropic and rank-two rows both start at a non-zero sample. Every capillary case under
the selection is walled by optmat's recommended silica, its default cladding there.
The bounds are ``tests/optmat_parity.py``'s, derived before any measurement.

Medium tier: ten fixed steps per case, under a second each.

Measured on the development machine against bounds of order 1e-10 to 1e-8, the worst
deviation of any field rung was 7.6e-16 (``gnlse_bundle``'s complete run on
``numba_cpu``) and of any statistic 2.7e-14 (``quadrature_error``).
"""

import numpy as np
import pytest

from optmat_parity import check_ladder
from optmat_parity_cases import build_case
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata.glasses import OptmatGlass

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

CASES = ("he_ar_kerr", "silver_wall", "gnlse_bundle", "bundle_vector")


@pytest.mark.medium
@pytest.mark.parametrize("name", CASES)
def test_every_rung_agrees_with_the_reference(name, equivalence_backend_class):
    check_ladder(name, equivalence_backend_class)


def test_the_cases_carry_optmats_materials():
    # What the parity runs carry: optmat's walls, one Kerr term per population at
    # the carrier, and the bundle's undamped rows starting at a non-zero sample.
    silver, kerr = build_case("silver_wall"), build_case("he_ar_kerr")
    assert silver.modes is not None and kerr.modes is not None
    assert kerr.simulation is not None
    assert {mode.cladding.source for mode in silver.modes} == {"table/Ag"}
    assert all(isinstance(mode.cladding, OptmatGlass) for mode in kerr.modes)
    assert {mode.cladding.source for mode in kerr.modes} == {"fit/SiO2"}
    terms = {
        response.species: response.chi3
        for response in kerr.simulation.responses
        if isinstance(response, KerrResponse)
    }
    assert set(terms) == {"He", "Ar"} and terms["He"] != terms["Ar"]
    assert kerr.problem.responses.labels == ("He", "Ar")
    for name, tensor in (("gnlse_bundle", False), ("bundle_vector", True)):
        (leaf,) = build_case(name).problem.responses.raman_terms
        assert leaf.decay_rates == () or all(
            float(np.max(np.abs(table.coefficients))) == 0.0
            for table in leaf.decay_rates
        )
        if tensor:
            assert np.all(leaf.undamped[:, 0] != 0.0)
