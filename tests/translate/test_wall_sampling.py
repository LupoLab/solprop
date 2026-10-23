# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coherent wall assembly samples each channel's frequency-only terms once.

The terms a Marcatili index is built from do not depend on ``z``, so inside the wall
assembly's scope (:func:`solprop.modes.capillary._shared_sampling`) a mode samples its
gas and cladding models once per frequency vector instead of at every position the
generator, the connection, the normalisation term and the passivity allowance read. The
factorisation is pure translation: the shared terms are the results of the same function
of the same inputs, so the assembled arrays are exactly what sampling at every position
gives (a structural contract: reuse, not a second computation).
"""

import contextlib
import dataclasses

import numpy as np
import pytest

from driver.test_coherent_capillary import varying
from solprop.exceptions import GrazingIncidenceWarning, MaterialRangeWarning
from solprop.modes.capillary import CoherentWall, MarcatiliMode, _shared_sampling
from solprop.translate import problem as problem_module

pytestmark = [
    pytest.mark.filterwarnings(
        f"ignore::{MaterialRangeWarning.__module__}.MaterialRangeWarning"
    ),
    pytest.mark.filterwarnings(
        f"ignore::{GrazingIncidenceWarning.__module__}.GrazingIncidenceWarning"
    ),
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
]

OMEGA = np.linspace(1.5e15, 3.5e15, 7)
"""A few active frequencies of the qualification band, rad/s."""


def _arrays(value: object, path: str = "") -> dict[str, np.ndarray]:
    """Return every array reachable through a dataclass tree, by its field path."""
    if isinstance(value, np.ndarray):
        return {path: value}
    found: dict[str, np.ndarray] = {}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            found |= _arrays(getattr(value, field.name), f"{path}.{field.name}")
    elif isinstance(value, tuple | list):
        for index, item in enumerate(value):
            found |= _arrays(item, f"{path}[{index}]")
    return found


def test_a_scope_returns_the_terms_it_already_sampled_read_only():
    """Inside the scope a repeated read is the same object; outside, a fresh one."""
    mode = MarcatiliMode(50e-6, "He", 1.0)
    outside = mode.sampled_terms(OMEGA)
    assert mode.sampled_terms(OMEGA) is not outside
    assert outside.cladding_term.flags.writeable
    with _shared_sampling():
        first = mode.sampled_terms(OMEGA)
        assert mode.sampled_terms(OMEGA.copy()) is first
        assert mode.sampled_terms(OMEGA[:3]) is not first
        other = MarcatiliMode(50e-6, "He", 1.0, m=2)
        assert other.sampled_terms(OMEGA) is not first
        for array in (first.omega, first.core_susceptibility, first.cladding_term):
            assert not array.flags.writeable
        for name in ("omega", "core_susceptibility", "cladding_term"):
            assert np.array_equal(getattr(first, name), getattr(outside, name))
    assert mode.sampled_terms(OMEGA) is not first


def test_the_scope_keeps_its_own_copy_of_the_frequencies():
    """A caller that later writes into its frequency vector changes nothing held."""
    mode = MarcatiliMode(50e-6, "He", 1.0)
    omega = OMEGA.copy()
    with _shared_sampling():
        held = mode.sampled_terms(omega)
        omega[0] *= 1.5
        assert held.omega[0] == OMEGA[0]
        assert mode.sampled_terms(omega) is not held


def test_the_scope_ends_when_its_block_raises():
    """A translation that fails leaves no shared terms behind."""
    mode = MarcatiliMode(50e-6, "He", 1.0)
    with pytest.raises(RuntimeError, match="stop"), _shared_sampling():
        mode.sampled_terms(OMEGA)
        raise RuntimeError("stop")
    assert mode.sampled_terms(OMEGA).cladding_term.flags.writeable


@pytest.mark.medium
def test_a_varying_wall_translates_exactly_as_it_did_and_samples_far_less(
    monkeypatch,
):
    """The taper's flow, carrier and weights are exactly the unshared assembly's.

    The taper carries the connection, the per-channel normalisation and the passivity
    allowance, so every consumer of the sampled terms is exercised; the count of fresh
    samplings drops from several per position to a handful per assembly.
    """
    calls = {"count": 0}
    original = MarcatiliMode._sample_terms

    def counted(self, values):
        calls["count"] += 1
        return original(self, values)

    monkeypatch.setattr(MarcatiliMode, "_sample_terms", counted)
    shared = varying(wall=CoherentWall()).translate()
    shared_count = calls["count"]
    calls["count"] = 0
    monkeypatch.setattr(problem_module, "_shared_sampling", contextlib.nullcontext)
    unshared = varying(wall=CoherentWall()).translate()
    unshared_count = calls["count"]
    before = _arrays(unshared.linear)
    after = _arrays(shared.linear)
    assert before.keys() == after.keys()
    assert shared.linear.coupled_flow is not None
    for path, array in before.items():
        assert array.dtype == after[path].dtype, path
        assert array.tobytes() == after[path].tobytes(), path
    # The assembly reads each of three channels at about a hundred positions, several
    # times at each; shared, the wall's own reads collapse to a few per channel.
    assert unshared_count - shared_count > 1000
