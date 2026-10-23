# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Derive the reciprocal source functional separately from coefficient recovery."""

import numpy as np
from numpy.testing import assert_allclose

from modal_provider_cases import OMEGA, make_provider


def test_reciprocal_source_oracle():
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    nodes = provider.node_rule(OMEGA)
    sample = provider.solve(OMEGA, 0.0, nodes=nodes)
    assert sample.backward_electric is not None
    assert sample.backward_magnetic is not None
    assert sample.reciprocal_gram is not None
    e, h = sample.electric, sample.magnetic
    eb, hb = sample.backward_electric, sample.backward_magnetic
    # Bound fixed before evaluating: four N-node cross-product reductions.
    bound = 64 * np.finfo(float).eps * len(nodes.xy)
    gram = np.einsum("qm,qn,q->mn", hb[:, 1], e[:, 0], nodes.weights) - np.einsum(
        "qm,qn,q->mn", hb[:, 0], e[:, 1], nodes.weights
    )
    gram -= np.einsum("qm,qn,q->mn", eb[:, 0], h[:, 1], nodes.weights) - np.einsum(
        "qm,qn,q->mn", eb[:, 1], h[:, 0], nodes.weights
    )
    assert_allclose(gram, -4 * sample.flux_metric, rtol=0, atol=bound)
    rng = np.random.default_rng(4102)
    polarisation = rng.normal(size=e.shape[:2]) + 1j * rng.normal(size=e.shape[:2])
    reciprocal = np.linalg.solve(
        gram, 1j * OMEGA * np.einsum("qcm,qc,q->m", eb, polarisation, nodes.weights)
    )
    direct = (
        -1j
        * OMEGA
        / 4
        * np.linalg.solve(
            sample.flux_metric,
            np.einsum("qcm,qc,q->m", e.conj(), polarisation, nodes.weights),
        )
    )
    assert_allclose(reciprocal, direct, rtol=bound, atol=0)
    amplitude = np.array([1 + 2j, 0.4 - 0.7j])
    recovered = np.linalg.solve(gram, np.einsum("mn,n->m", gram, amplitude))
    assert_allclose(recovered, amplitude, rtol=bound, atol=0)
