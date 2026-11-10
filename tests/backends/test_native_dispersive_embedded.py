# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The dispersive response's embedded rule on ``native`` (SEP 43).

From the retired ``test_numba_dispersive_kerr.py``. Native's plasma test compares the
whole stage source, which holds one rule; the retired one compared both rules' projected
spectra, and it held that a second finish of an evaluation lands nothing.
"""

import numpy as np
import pytest

pytest.importorskip("solprop_native")


from backends.dispersive_cases import POLICIES, capillary, radial
from backends.test_native_dispersive_kerr import (
    NativeBackend,
    assert_close,
    disagreement,
    embedded,
    evaluation_bound,
    native_evaluation,
    reference_evaluation,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("envelope", [False, True])
def test_plasma_beside_the_response_reads_the_bare_field_on_both_rules(
    envelope, evaluator
):
    """Both rules' projected spectra beside plasma, against the reference.

    The bound is the retired one's: the response's own evaluation budget plus what the
    carrier policy's run already disagrees by on this backend, since the plasma column's
    scan and the prefactor's band edges disagree with the reference's. A plasma current
    the response filtered would be wrong by the multiplier's departure from one, about
    1e-3, far above either. An envelope's currents are assembled apart from these
    spectra on both backends, so there the Kerr share alone is compared.
    """
    kind = {"plasma": True, "thg": False, "envelope": envelope, "energy": 60e-6}
    simulation = capillary(evaluator, **kind)
    problem = simulation.translate()
    assert (problem.responses.envelope_plasma is not None) == envelope
    assert embedded(problem)
    baseline = capillary("carrier-kerr", **kind)
    carrier = baseline.translate()
    ours, theirs = (
        native_evaluation(carrier, baseline.grid),
        reference_evaluation(carrier, baseline.grid),
    )
    assert ours.coarse is not None and theirs.coarse is not None
    floor = max(
        disagreement(ours.projected, theirs.projected),
        disagreement(ours.coarse, theirs.coarse),
    )
    assert_close(
        native_evaluation(problem, simulation.grid),
        reference_evaluation(problem, simulation.grid),
        evaluation_bound(problem) + floor,
    )


@pytest.mark.parametrize("evaluator", ["polynomial", "symmetric"])
def test_the_embedded_rule_is_finished_once_and_only_where_it_exists(evaluator):
    """A second finish of the same evaluation, or one without the rule, does nothing.

    The embedded products are transformed once per evaluation, when the rule is read; a
    repeated read must not land them twice. The radial route has no embedded rule, so
    reading its fine rule leaves it untouched. Explicit no-ops, so exact.
    """
    simulation = capillary(evaluator)
    problem = simulation.translate()
    assert embedded(problem)
    backend = NativeBackend(problem, grid=simulation.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._adams_lab_source(problem.initial.field0[None], 0.0)
        native = backend._native_state
        fine = np.array(native.projected_to_host(False))
        landed = np.array(native.projected_to_host(True))
        assert np.any(landed)
        np.testing.assert_array_equal(native.projected_to_host(True), landed)
        np.testing.assert_array_equal(native.projected_to_host(False), fine)
    finally:
        backend.close()
    radial_problem, grid = radial(evaluator, envelope=False, thg=True)
    assert not embedded(radial_problem)
    backend = NativeBackend(radial_problem, grid=grid)
    try:
        backend.set_initial(radial_problem.initial.field0, None)
        backend._adams_lab_source(radial_problem.initial.field0[None], 0.0)
        native = backend._native_state
        before = np.array(native.projected_to_host(False))
        assert np.any(before)
        np.testing.assert_array_equal(native.projected_to_host(False), before)
    finally:
        backend.close()
