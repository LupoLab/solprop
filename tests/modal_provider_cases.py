# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared physical fixtures for independent modal-provider tests."""

from types import SimpleNamespace

import numpy as np

from solprop.constants import C
from solprop.grid import FieldGrid
from solprop.modes import (
    GradedIndexProvider,
    PowerLawPermittivity,
    StepIndexMode,
)
from solprop.modes.guide import (
    ChannelBasis,
    ModalGuide,
    StepIndexProvider,
    circular_pair,
    provider_channels,
)
from solprop.modes.material import IndexMaterial, MaterialKerr, MaterialRegion
from solprop.modes.protocol import ChannelLabel
from solprop.nonlinear.raman import SilicaTensorResponse
from solprop.profiles import ZProfile
from solprop.translate.general import general_modal_route
from solprop.translate.modal_tables import sample_modal_table

WAVELENGTH = 835e-9
OMEGA = 2 * np.pi * C / WAVELENGTH


def make_provider(
    radius: float | ZProfile = 1.5e-6,
    descriptors=(("HE", 1, 1, "even"),),
    kerr=None,
    indices=(1.45, 1.0),
    raman=None,
):
    """Return a step-index provider, optionally with a responding core."""
    core_index, cladding_index = indices
    modes = tuple(
        StepIndexMode(
            radius,
            core=core_index,
            cladding=cladding_index,
            profile_wavelength=WAVELENGTH,
            kind=kind,
            n=n,
            m=m,
            parity=parity,
        )
        for kind, n, m, parity in descriptors
    )
    core, cladding = IndexMaterial(core_index), IndexMaterial(cladding_index)
    regions = (
        MaterialRegion(
            "core",
            0.0,
            modes[0].core_radius,
            core,
            () if kerr is None else ((kerr,) if raman is None else (kerr, raman)),
        ),
        MaterialRegion("cladding", modes[0].core_radius, None, cladding),
    )
    guide = ModalGuide.from_step_index(
        modes, core=core, cladding=cladding, regions=regions
    )
    assert isinstance(guide.field_provider, StepIndexProvider)
    return guide.field_provider


GENERAL_CHI3 = 2.2e-22
"""Absolute third-order susceptibility of a general guide's core, m²/V².

Silica-like, and declared by the tests rather than looked up: what the general modal
route's tests measure is its arithmetic and not the material.
"""

GENERAL_Z_END = 1e-3
"""Length a general modal test problem declares, m: a millimetre, the cap."""


def general_grid(z_end=GENERAL_Z_END):
    """Return the field grid every general modal test propagates on."""
    return FieldGrid.from_limits(z_end, 835e-9, (815e-9, 860e-9), 60e-15)


def general_guide(  # noqa: PLR0913, PLR0917 - one knob per fixture dimension
    descriptors=(("HE", 1, 1, "even"),),
    blocks=(),
    radius=1.5e-6,
    grid=None,
    chi3: float | None = GENERAL_CHI3,
    n_r=16,
    knots=9,
    band=None,
    indices=(1.45, 1.0),
    kerr: MaterialKerr | None = None,
    raman: SilicaTensorResponse | None = None,
    circular=None,
):
    """Build a step-index strand's guide, sampled over a grid's whole active band.

    The guide's knots must cover the grid's **active** band and not its declared
    one: ``active_mask`` starts at half the lowest declared frequency, and the
    cumulative operator samples every channel's propagation constant across it, so
    a narrower table is refused by the operator before the route is ever reached.
    ``band`` supplies that interval directly, in rad/s, for a comparison that has
    to cover two grids' active bands at once. ``kerr`` declares the core's response
    outright, for a case that needs an ``n2`` or a measurement wavelength rather
    than the absolute ``chi3`` the shorthand builds. ``raman`` adds the delayed
    silica tensor response to the core beside it. ``circular`` is a pair of channel
    indices whose circular partners the guide declares as its basis, which must be
    one of ``blocks``.
    """
    grid = general_grid() if grid is None else grid
    lo, hi = (
        band
        if band is not None
        else (
            float(grid.omega[grid.active_mask][0]),
            float(grid.omega[grid.active_mask][-1]),
        )
    )
    provider = make_provider(
        radius=radius,
        descriptors=descriptors,
        kerr=kerr
        if kerr is not None
        else (None if chi3 is None else MaterialKerr(chi3=chi3)),
        raman=raman,
        indices=indices,
    )
    nodes = provider.node_rule(0.99 * lo, n_r=n_r)
    return ModalGuide.from_provider(
        provider,
        np.linspace(0.99 * lo, 1.01 * hi, knots),
        nodes=nodes,
        degenerate_blocks=blocks,
        basis=None
        if circular is None
        else circular_pair(provider_channels(provider), circular),
    )


GRIN_WAVELENGTH = 1.03e-6
"""Reference wavelength of the graded-index system, m."""

GRIN_RADIUS = 25e-6
"""Core radius of the graded-index system, m."""

GRIN_CONTRAST = 0.005
"""Index contrast Delta of the graded-index system, dimensionless."""

GRIN_AXIS_INDEX = 1.45
"""Index on the axis of the graded-index system, dimensionless."""


def grin_grid(z_end=GENERAL_Z_END):
    """Return the field grid the graded-index system propagates on.

    A narrow declared band around 1.03 um: the active band is twice it, and the
    guide's two channels must stay bound at the long-wavelength end, where the
    finite domain's tail guard is taken. The 100 fs window is the shortest whose
    spectral collar resolves its own falling edge on this band, and every bin of
    it costs a column of the stored E and H reference over thousands of nodes.
    """
    return FieldGrid.from_limits(z_end, GRIN_WAVELENGTH, (1.0e-6, 1.06e-6), 100e-15)


def grin_provider(*, elements=64, count=2, kerr=None):
    """Return the reference parabolic LP sector, optionally with a Kerr core.

    ``LP`` and not ``TE`` because the two sectors are different **mode groups**,
    not two answers to one question. The closed form
    ``beta_g = sqrt(q^2 - 2 q sqrt(2 Delta) g)/a`` takes ``g = 2m + l + 1``, so
    the two ``l = 0`` channels here are groups 1 and 3, while a ``TE_0m`` channel
    satisfies the scalar radial equation at azimuthal harmonic **one** and is
    groups 2 and 4. Against its own groups the TE sector agrees to the same
    discretisation order as the LP one -- 3.06e-7 and 1.53e-6 at 64 elements,
    1.36e-7 and 6.80e-7 at 96 -- and the genuine vector correction, TM against TE,
    is 5.6e-10 and 1.4e-9 at 64. The 4.5e-4 once recorded here is the l = 0 to
    l = 1 group separation and comes from comparing TE against ``g = (1, 3)``
    (``tests/modes/test_graded_index.py``).
    """
    return GradedIndexProvider(
        permittivity=PowerLawPermittivity(
            IndexMaterial(GRIN_AXIS_INDEX), GRIN_CONTRAST
        ),
        exterior=IndexMaterial(GRIN_AXIS_INDEX * np.sqrt(1.0 - 2.0 * GRIN_CONTRAST)),
        radius=GRIN_RADIUS,
        family="LP",
        count=count,
        elements=elements,
        extent=2.0,
        core_responses=() if kerr is None else (kerr,),
    )


def grin_guide(grid=None, *, elements=64, knots=5, n_r=2, kerr=None):
    """Build the graded-index guide over a grid's whole active band."""
    grid = grin_grid() if grid is None else grid
    active = grid.omega[grid.active_mask]
    lo, hi = float(active[0]), float(active[-1])
    provider = grin_provider(elements=elements, kerr=kerr)
    return ModalGuide.from_provider(
        provider,
        np.linspace(0.99 * lo, 1.01 * hi, knots),
        nodes=provider.node_rule(0.99 * lo, n_r=n_r),
    )


def parabolic_beta(omega, groups=(1, 3)):
    """Return the ideal parabolic weak-guidance propagation constants, 1/m.

    ``beta_g = sqrt(q^2 - 2 q sqrt(2 Delta) g)/a`` with ``q = omega n_axis a / c``
    and the group number ``g = 2m + l + 1``, which for the two ``l = 0`` radial
    channels of this sector is 1 and 3. The closed form of an **ideal** parabolic
    profile: the guide truncates its at the core radius, which is the departure
    the tests measure rather than assume.
    """
    q = float(omega) / C * GRIN_RADIUS * GRIN_AXIS_INDEX
    return (
        np.sqrt(q * q - 2.0 * q * np.sqrt(2.0 * GRIN_CONTRAST) * np.asarray(groups))
        / GRIN_RADIUS
    )


def general_case(  # noqa: PLR0913, PLR0917 - one knob per fixture dimension
    descriptors=(("HE", 1, 1, "even"),),
    blocks=(),
    radius=1.5e-6,
    grid=None,
    raman=None,
    kerr_form=None,
    knots=9,
    circular=None,
):
    """Build a step-index strand, its guide, its grid and its assembled route.

    The shared fixture of the general modal route. The route comes from
    :func:`solprop.translate.general.general_modal_route`, which is what
    ``Simulation.from_waveguide(provider=...)`` uses, so the backend tests and the
    driver tests propagate the same leaves.

    ``grid`` replaces the shared 116 fs window, for a case that needs a longer one
    -- a delayed response has memory a short window truncates -- or a wider band,
    for one that needs the third harmonic inside it; a wider band needs more
    ``knots`` to hold the table's own accuracy gate. ``raman`` adds the core's
    delayed response, ``kerr_form`` overrides the route's default cubic, and
    ``circular`` declares the circular partners of a degenerate pair as the basis.
    """
    grid = general_grid() if grid is None else grid
    guide = general_guide(
        descriptors=descriptors,
        blocks=blocks,
        radius=radius,
        grid=grid,
        raman=raman,
        knots=knots,
        circular=circular,
    )
    sampling = guide.sampling
    assert sampling is not None
    table = sample_modal_table(
        guide.field_provider,
        sampling.omega,
        np.array([sampling.z]),
        sampling.nodes,
        degenerate_blocks=sampling.degenerate_blocks,
        basis=sampling.basis,
    )
    route = general_modal_route(
        guide,
        grid,
        z_max=GENERAL_Z_END,
        omega_reference=grid.omega0,
        **({} if kerr_form is None else {"kerr_form": kerr_form}),
    )
    return SimpleNamespace(
        grid=grid,
        provider=guide.field_provider,
        nodes=sampling.nodes,
        table=table,
        guide=guide,
        route=route,
    )


SPLIT_DESCRIPTORS = (("TE", 0, 1, "even"), ("TM", 0, 1, "even"))
"""The two retained members of the weak-guidance ``LP_11`` group.

``TE_01`` and ``TM_01`` are degenerate under weak guidance -- they are two of the
four constituents of the scalar ``LP_11`` mode, the other two being the ``HE_21``
parity pair -- and are **split** by the vector correction, by 120 1/m at 835 nm on the
6 um core the runs propagate and 1.1e-5 of their own propagation constants. That is far
above the ``64 eps max|beta|`` floor an exactly degenerate block is judged at, so a
declared unitary of this pair is the split-block case and not the exactly degenerate
one.
"""

SPLIT_LABELS = (
    ChannelLabel("LP", 1, 1, "even", None, "LP1,1:even"),
    ChannelLabel("LP", 1, 1, "odd", None, "LP1,1:odd"),
)
"""Labels of the equal-weight rotation of :data:`SPLIT_DESCRIPTORS`.

The rotation is a legitimate unitary of the retained pair; it is **not** the whole
scalar ``LP_11`` mode, which also needs the two ``HE_21`` parities. The labels say
which group the rotated channels belong to, as every other declared basis's do.
"""


def split_basis():
    """Return the equal-weight rotation of the retained ``LP_11`` pair."""
    root = 1.0 / np.sqrt(2.0)
    return ChannelBasis(
        np.array([[root, -root], [root, root]], dtype=complex),
        ((0, 1),),
        SPLIT_LABELS,
        "lp11(0,1)",
    )


def split_guide(grid=None, *, rotated=True, radius=6e-6, knots=17, n_r=24):
    """Return the split-block guide, in the rotated basis or the provider's own.

    Seventeen knots and a 6 um core: the pair's **splitting** is a small difference
    of two large propagation constants, so reproducing it between the knots costs
    more of them than reproducing either, and the narrower core's channels are close
    enough to cutoff at the long-wavelength end of this grid's active band that the
    guide's own accuracy gate fails there.
    """
    grid = general_grid() if grid is None else grid
    provider = make_provider(
        radius=radius, descriptors=SPLIT_DESCRIPTORS, indices=(1.45, 1.40)
    )
    active = grid.omega[grid.active_mask]
    lo, hi = float(active[0]), float(active[-1])
    return ModalGuide.from_provider(
        provider,
        np.linspace(0.99 * lo, 1.01 * hi, knots),
        nodes=provider.node_rule(0.99 * lo, n_r=n_r),
        degenerate_blocks=((0, 1),),
        basis=split_basis() if rotated else None,
    )
