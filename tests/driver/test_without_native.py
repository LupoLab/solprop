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

from solprop.backends import registry
from solprop.driver import prop_gnlse
from solprop.exceptions import SolpropWarning

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("solprop_native") is not None,
    reason="solprop-native is installed",
)


def test_auto_warns_and_names_the_install_command(monkeypatch):
    monkeypatch.setattr(registry, "_warned_native_fallback", False)
    with pytest.warns(
        SolpropWarning, match=r'pip install "solprop\[native\]"'
    ) as caught:
        assert registry.resolve_name() == "numpy_ref"
    message = str(caught[0].message)
    assert "6 to 40 times" in message
    assert 'pass backend="numpy_ref"' in message


@pytest.mark.filterwarnings("ignore:solprop-native is not installed")
@pytest.mark.parametrize("backend", ["auto", "numpy_ref"])
def test_a_short_run_finishes_without_loading_native(backend):
    output = prop_gnlse(
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
    assert output["z"][-1] == 1e-4
    assert "solprop_native" not in sys.modules
