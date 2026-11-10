# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run's identity did not move when its policy left the problem (SEP 47 section 5.2).

The stepper settings, the save planes and the statistics are the run request's now, not
fields of ``PropagationProblem``, and every record that hashes them reads them from the
request. Whatever counted as the same run for a resume must still count: the
configuration hashes of the examples are the literals recorded before the change, and
files written before it resume (``tests/data/run_request/generate.py``).
"""

import platform
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from solprop.output.hdf5 import open_file
from solprop.recipe import Recipe, resume_file

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.SolpropWarning"),
]

DATA = Path(__file__).parents[1] / "data" / "run_request"

HASHES = {
    "soliton": "d37e2ac23bd6b21ca5308b1f530344d018e0af14948ad21894eb067fe292be36",
    "rdw": "b2ead742fa91c665a95f3375d3ada7cba53893dca8dd6409ad5b4294b03b42cd",
    "radial_kerr": "50705b394c29cdead46043f1781713c6ea9f705eaeed3ab364f2a0fc90e24d12",
    "cartesian_beam": (
        "249df1612749a8f34c2179d8b3c6bc28dbeb57c7630f615fd920c48d9c0e694f"
    ),
    "line_beam": "35e48f6495a175fced8bd6cb6fef5fb11530ec6515f0300b5e663e3ac6783eaa",
    "time_window": "771a79213dacfa1ca5ce36964a592a8529e292aa6c5fa38049fb8c2e9a86dc3b",
    "single_precision": (
        "eea7f04d0cda2cb3edc5a7bdf541217a1cc1aec64e41c8f108fe420ceeb3f01c"
    ),
    "self_phase_modulation": (
        "cb9b13a6976bc06c26fe309253be2e265d90caae881d4211e88e903f1cb7b25c"
    ),
    "coherent_wall": (
        "5be08c7d9b4af229dcffdbdfc65cb6b28065d2df19453f8134592416646acd2a"
    ),
    "planar_hybrid": (
        "317ca8c72d7d11b62abb350e3284782f8017a34b0755834ec8f28e374a8cdbb8"
    ),
    "fibre_supercontinuum": (
        "d6b60f4e07a346bc1db9f16b5300dfb5e42717a2652c76200fe486032ee343e7"
    ),
}
"""Each example's configuration hash at commit 50d47d16, before the change.

Recorded on Linux x86-64. A record that hashes tables evaluated through the platform's
libm (a material's cross sections, say) may differ in its last bits elsewhere, which is
not what these literals test, so they are compared on that platform alone; the stopped
files below are read everywhere. An example that draws an unseeded noise realisation
(``envelope_plasma``) hashes a fresh seed at every build and is not among them.
"""

RECORDED_PLATFORM = sys.platform == "linux" and platform.machine() == "x86_64"


@pytest.mark.skipif(
    not RECORDED_PLATFORM, reason="the literals were recorded on Linux x86-64"
)
@pytest.mark.parametrize("name", sorted(HASHES))
def test_an_example_hashes_as_it_did_when_the_problem_carried_its_policy(name):
    simulation = Recipe.example(name).build()
    assert simulation.metadata().configuration_hash() == HASHES[name]


def _stored_hash(path: Path) -> str:
    with h5py.File(path) as file:
        return str(file.attrs["configuration_hash"])


@pytest.mark.parametrize(
    ("name", "adams"),
    [("soliton_stopped.h5", False), ("soliton_adams_stopped.h5", True)],
)
def test_a_file_stopped_before_the_change_resumes_from_itself(tmp_path, name, adams):
    path = tmp_path / name
    shutil.copy(DATA / name, path)
    stored = _stored_hash(path)
    with open_file(path) as handler:
        before = handler.n_saved
        kept = np.asarray(handler["field"][:before])
    # A resume refuses a file whose recorded hash is not the rebuilt run's own.
    resume_file(path, progress="none")
    assert _stored_hash(path) == stored
    with open_file(path) as handler:
        assert handler.outcome is not None
        assert handler.outcome.status == "complete"
        assert handler.metadata.configuration_hash() == stored
        # The run's five planes, the entrance among them, the stopped ones as written.
        assert handler.n_saved == 5
        assert before < handler.n_saved
        np.testing.assert_array_equal(np.asarray(handler["field"][:before]), kept)
        restart = handler.outcome.execution["restart"]
    if adams:
        # The history the stopped run kept is the one the resumed march continues.
        assert restart["restoration"]["state"] == "restored"
