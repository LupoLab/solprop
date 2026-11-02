# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""An installation that really has no native package: CI's ``no-native`` lane.

Other tests hide the package from the probe. These need it truly absent, so that any
import of it on the way through a run fails.
"""

import importlib.util
import sys

import pytest

from solprop.driver import prop_gnlse
from solprop.exceptions import BackendError

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("solprop_native") is not None,
        reason="solprop-native is installed",
    ),
    pytest.mark.real_auto,
]


def run(backend):
    return prop_gnlse(
        0.11,
        1e-4,
        [0.0, 0.0, -11.83e-27],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        fwhm=50e-15,
        peak_power=1e3,
        shape="sech",
        backend=backend,
        progress="none",
    )


def test_auto_refuses_and_says_how_to_install():
    with pytest.raises(BackendError, match="pip install solprop"):
        run("auto")


def test_the_reference_by_name_finishes_without_loading_native():
    assert run("numpy_ref")["z"][-1] == 1e-4
    assert "solprop_native" not in sys.modules
