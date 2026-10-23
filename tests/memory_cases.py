# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Problems of the general modal route and the coupled flow at two sizes each.

The memory tests build each backend from these problems and compare its
``memory_ledger()`` with :func:`solprop.backends.memory.estimate_memory`.
**Construction only**: nothing is propagated.

Cases (every one a test fixture's problem, at its own size and at a larger one):

``general``   the parity pair of ``tests/general_parity.py`` (two HE11 parities, a
              1.5 um strand, the real cubic); larger: twice the window and a finer
              radial rule.
``raman``     the one-channel strand with the delayed silica tensor Raman term;
              larger: twice the window.
``wall``      the scalar coherent wall of ``tests/flow_parity.py``, three channels on
              513 bins; larger: six channels on twice the window.
``te_tm``     the TE/TM wall, four channels; larger: twice the window.
``taper``     the varying wall on the 50 to 40 um taper, three channels; larger: twice
              the window.
``split``     the rotated split pair on the general route with a Kerr core; larger:
              twice the window.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("solprop_native") is not None:
    importlib.import_module("solprop_native")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
general_parity = importlib.import_module("general_parity")
flow_parity = importlib.import_module("flow_parity")
cases = importlib.import_module("modal_provider_cases")

from solprop.backends.registry import available  # noqa: E402
from solprop.boundaries import absorber  # noqa: E402
from solprop.driver import Simulation  # noqa: E402
from solprop.grid import FieldGrid  # noqa: E402
from solprop.modes.guide import ModalGuide  # noqa: E402
from solprop.modes.material import MaterialKerr  # noqa: E402
from solprop.nonlinear.raman import SilicaTensorResponse  # noqa: E402
from solprop.stepper import FixedSchedule  # noqa: E402
from solprop.translate.general import general_modal_route  # noqa: E402

BACKENDS = tuple(
    name for name in ("numpy_ref", "numba_cpu", "native") if name in available()
)
GROUPS = ("general_", "coupled_flow_", "flow_")
GENERAL_GROUPS = (
    "general_maps",
    "general_reference",
    "general_chunk",
    "general_delayed",
)


def general_problem(*, window, n_r, raman=False):
    """Return a general modal problem and its grid (the pair, or the Raman strand)."""
    grid = FieldGrid.from_limits(1e-3, 835e-9, (815e-9, 860e-9), window)
    if raman:
        guide = cases.general_guide(grid=grid, n_r=n_r, raman=SilicaTensorResponse())
    else:
        guide = cases.general_guide(
            descriptors=general_parity.PAIR, blocks=((0, 1),), grid=grid, n_r=n_r
        )
    route = general_modal_route(
        guide, grid, z_max=1e-3, omega_reference=float(grid.omega0), kerr_form="real"
    )
    state = general_parity.launch(grid, len(guide.mode_collection))
    return general_parity.problem_for(grid, guide, route, state), grid


def split_simulation(window):
    """Return ``tests/flow_parity.py``'s rotated split pair on a given window."""
    grid = FieldGrid.from_limits(1e-3, 835e-9, (815e-9, 860e-9), window)
    provider = flow_parity.make_provider(
        radius=6e-6,
        descriptors=flow_parity.SPLIT_DESCRIPTORS,
        indices=(1.45, 1.40),
        kerr=MaterialKerr(chi3=cases.GENERAL_CHI3),
    )
    active = grid.omega[grid.active_mask]
    lo, hi = float(active[0]), float(active[-1])
    guide = ModalGuide.from_provider(
        provider,
        np.linspace(0.99 * lo, 1.01 * hi, 17),
        nodes=provider.node_rule(0.99 * lo, n_r=24),
        degenerate_blocks=((0, 1),),
        basis=flow_parity.split_basis(),
    )
    steps = flow_parity.STEPS
    return Simulation.from_waveguide(
        (),
        1e-3,
        provider=guide,
        grid=grid,
        channel=0,
        absorber=absorber(grid, boundary_length=1.0e3),
        lambda0=835e-9,
        fwhm=20e-15,
        energy=2e-9,
        pulseshape="gauss",
        backend="numpy_ref",
        save_planes=[1e-3],
        dz_max=1e-3 / steps,
        dz_init=1e-3 / steps,
        schedule=FixedSchedule(np.full(steps, 1e-3 / steps)),
    )


def flow_problem(name, *, window, **overrides):
    """Return a coupled-flow parity case's problem and grid on a given window."""
    if name == "split":
        simulation = split_simulation(window)
    else:
        grid = FieldGrid.from_limits(
            flow_parity.LENGTH, 800e-9, (400e-9, 1.6e-6), window
        )
        simulation = flow_parity.build(
            name,
            grid=grid,
            absorber=absorber(grid, boundary_length=1.0e3),
            **overrides,
        )
    return simulation.translate(), simulation.grid


SIZES = {
    "general": (
        lambda: general_problem(window=60e-15, n_r=16),
        lambda: general_problem(window=120e-15, n_r=24),
    ),
    "raman": (
        lambda: general_problem(window=60e-15, n_r=16, raman=True),
        lambda: general_problem(window=120e-15, n_r=16, raman=True),
    ),
    "wall": (
        lambda: flow_problem("scalar", window=300e-15),
        lambda: flow_problem("scalar", window=600e-15, modes=6),
    ),
    "te_tm": (
        lambda: flow_problem("te_tm", window=300e-15),
        lambda: flow_problem("te_tm", window=600e-15),
    ),
    "taper": (
        lambda: flow_problem("taper", window=300e-15),
        lambda: flow_problem("taper", window=600e-15),
    ),
    "split": (
        lambda: flow_problem("split", window=60e-15),
        lambda: flow_problem("split", window=120e-15),
    ),
}
