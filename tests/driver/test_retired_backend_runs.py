# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Runs that ``numba_cpu`` wrote, after its removal (SEP 43 section 9).

The files are real, written by versions that had the backend
(``tests/data/retired_backends/generate.py``). The backend is an execution resource,
outside a run's identity (SEP 42), so a stopped run continues on another backend by
changing the name. The continuation is an adaptive pair, held to an envelope against a
tighter run (SEP 25), never to the original's steps.
"""

import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from march_bounds import TIGHTER, assert_in_envelope
from solprop.driver import prop_gnlse
from solprop.exceptions import BackendError, OutputError

DATA = Path(__file__).parents[1] / "data" / "retired_backends"
STOPPED = DATA / "numba_cpu_stopped.h5"
PUBLISHED = DATA / "numba_cpu_0.1.0.h5"
RTOL = 1e-7  # the default, which generate.py used


def run(backend, output=None, *, rtol=RTOL, resume=False):
    """Run the problem generate.py wrote."""
    return prop_gnlse(
        0.11,
        2e-4,
        [0.0, 0.0, -11.83e-27],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        fwhm=50e-15,
        peak_power=1e3,
        shape="sech",
        backend=backend,
        noise=None,
        output=output,
        n_save=5,
        rtol=rtol,
        resume=resume,
        progress="none",
    )


def recorded_backend(path):
    with h5py.File(path) as file:
        return file.attrs["backend"]


def test_the_stopped_file_is_a_numba_run_of_the_current_schema():
    assert recorded_backend(STOPPED) == "numba_cpu"
    with h5py.File(STOPPED) as file:
        assert file.attrs["schema_version"] == 2
        assert "terminal_checkpoint" in file


def backend_available(name):
    if name == "native":
        pytest.importorskip("solprop_native")


@pytest.mark.parametrize("backend", ["native", "numpy_ref"])
def test_a_stopped_numba_run_continues_on_another_backend(tmp_path, backend):
    backend_available(backend)
    path = tmp_path / "continued.h5"
    shutil.copyfile(STOPPED, path)
    with h5py.File(path) as handle:
        before: Any = handle
        kept = np.array(before["field"][:3])
    run(backend, path, resume=True)
    with h5py.File(path) as handle:
        after: Any = handle
        assert after["z"][-1] == 2e-4
        field = np.array(after["field"][:])
        # What the original run saved stays as it was written.
        np.testing.assert_array_equal(field[:3], kept)
    # Over every plane, not the last alone: this soliton's error at the last plane is
    # about 1e-11 by coincidence, against about 1e-8 at the third, for an uninterrupted
    # run on every backend, so the last plane alone would set no real envelope.
    whole = np.asarray(run(backend)["field"])
    reference = np.asarray(run(backend, rtol=RTOL / TIGHTER)["field"])
    assert_in_envelope([field], reference, whole, what=backend)


def test_naming_the_removed_backend_says_what_to_use(tmp_path):
    with pytest.raises(BackendError, match="numba_cpu backend was removed"):
        run("numba_cpu")
    path = tmp_path / "continued.h5"
    shutil.copyfile(STOPPED, path)
    with pytest.raises(BackendError, match="numba_cpu backend was removed"):
        run("numba_cpu", path, resume=True)


@pytest.mark.parametrize("backend", ["native", "numpy_ref"])
def test_a_file_from_the_published_release_is_refused_by_its_schema(tmp_path, backend):
    """0.1.0 wrote schema 1, which this release doesn't read, whatever the backend."""
    backend_available(backend)
    assert recorded_backend(PUBLISHED) == "numba_cpu"
    path = tmp_path / "published.h5"
    shutil.copyfile(PUBLISHED, path)
    with pytest.raises(OutputError, match="read it with the solprop version"):
        run(backend, path, resume=True)
