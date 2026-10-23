# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The compiled general modal route against the reference: cases and bounds.

Shared by ``tests/backends/test_numba_general_modal.py`` and the general modal
benchmark, so that the bound a test asserts and the bound the benchmark records are one
function.

**The bound, written before any measurement.** One stage's source is a chain of
roundings whose count is known: the synthesis sums ``M`` channel products, the
inverse and forward transforms of length ``n_to`` each carry Higham's
Cooley-Tukey bound ``log2(n_to) eta`` with ``eta ~ 7 u`` for accurate twiddles, the
instantaneous contraction is about thirty operations per sample, the projection
sums ``3 Q`` products over the ``Q`` nodes, and the frequency factor is one more
product. A delayed response adds the doubled axis's transform pair and a
contraction of the same size. Each backend is within that count of the exact
arithmetic, so the two are within **twice** it of each other, relative to the
projection's own scale: the bound is multiplied by the measured conditioning
``kappa = || sum |Q||P| || / || sum Q P ||`` of the projected sum, which is where
cancellation can amplify a relative rounding. Transform bounds are normwise and are
used here as though they were per element, which is the usual engineering reading
of them.

**A moving guide** adds the per-stage evaluation of its maps,
``S(z) = sum_t w_t S_t`` and ``Q(z)`` likewise over the stage's ``transform_weights``.
A weight of zero adds an exact zero on either backend (every coefficient is finite),
so each element is a sum of the ``n_deg`` terms of the stage's section, each product of
a real weight and a complex coefficient rounded once: each backend is within
``gamma_(n_deg) ~ n_deg u`` of the exact sum relative to ``sum_t |w_t| |c_t|``, and
the two within twice it -- in whatever order each adds them (the compiled kernel
sequentially from the first term, the reference by a BLAS product). Normwise, relative
to the map, that is ``2 n_deg u kappa_map`` with the measured
``kappa_map = || sum_t |w_t| |c_t| || / || sum_t w_t c_t ||`` at the stage's weights
(:func:`map_conditioning`). The field is linear in ``S`` and the cubic polarisation
carries it three times; the projection is linear in ``Q``; so the source's relative
departure gains ``2 u n_deg (3 kappa_S + kappa_Q)``, which :func:`source_bound` adds
to its count as ``moving`` before the projection's ``kappa`` multiplies it.

Ten steps compound it. A stage perturbation enters the step through the solution
weights, ``sum |b_i| = 1.645`` for Dormand-Prince, and a step's nonlinear
increment is ``phi_step ||a||``; over the run the perturbations add up to
``1.645 Phi eps_src`` and are carried by at most ``exp(Phi)``, where
``Phi = L ||N(a)|| / ||a||`` is measured at the launch. Both backends also round
the linear half of every step -- the stage combination, the propagator from one
``exp`` and one ``sincos`` against numpy's complex ``exp``, the acceptance -- which
is taken as 16 u each and so 32 u per step between them.
"""

import math

import numpy as np

from modal_provider_cases import (
    GENERAL_CHI3,
    SPLIT_DESCRIPTORS,
    general_grid,
    general_guide,
    grin_grid,
    grin_guide,
    make_provider,
)
from solprop import fourier
from solprop.backends.numpy_ref.backend import _general_maps
from solprop.backends.numpy_ref.general_modal import (
    GeneralMaps,
    delayed_polarisation,
    general_source,
    symmetric_cubic,
    symmetric_kerr_coefficients,
)
from solprop.constants import EPS0
from solprop.grid import FieldGrid
from solprop.modes.guide import ModalGuide
from solprop.modes.material import MaterialKerr
from solprop.nonlinear.raman import SilicaTensorResponse
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.general import general_modal_route
from solprop.translate.problem import assemble
from solprop.translate.responses import KerrForm

LENGTH = 1e-3
"""Length every parity case propagates, m: a millimetre."""

STEPS = 10
"""Fixed steps every parity case takes."""

PHI = 0.5
"""The nonlinear scale ``L ||N(a)|| / ||a||`` every launch is set to."""

UNIT = 2.0**-53
"""Unit roundoff of float64."""

FFT_CONSTANT = 7.0
"""``eta / u`` of Higham's Cooley-Tukey bound with accurately computed twiddles."""

CONTRACTION = 30
"""Roundings of one sample's instantaneous (or delayed) tensor contraction.

An upper count for every contraction the route runs: the real cubic's symmetric
form is within 17 (five forming a coefficient, two a monomial, ten in the sum;
``symmetric_cubic``), the analytic form's dense one within 14 and the delayed one
within 28 (a triple product and 26 additions)."""

STEP_CONSTANT = 32.0
"""Both backends' rounding of the linear half of one step, in units of ``u``."""

RK_WEIGHTS = 1.645
"""``sum_i |b_i|`` over the Dormand-Prince solution weights."""

PAIR = (("HE", 1, 1, "even"), ("HE", 1, 1, "odd"))
"""The two parities of HE11, the degenerate parity pair."""

CASES = ("pair", "pair_analytic", "circular", "raman", "split", "grin")
"""Every parity case, in the order the report lists them."""


def split_guide_with_kerr(grid):
    """Return the split TE01/TM01 guide in its own coordinates, with a Kerr core.

    The split block (``modal_provider_cases``) without its declared rotation, so without
    a coupled flow: the part of that route ``numba_cpu`` executes.
    """
    provider = make_provider(
        radius=6e-6,
        descriptors=SPLIT_DESCRIPTORS,
        indices=(1.45, 1.40),
        kerr=MaterialKerr(chi3=GENERAL_CHI3),
    )
    active = grid.omega[grid.active_mask]
    lo, hi = float(active[0]), float(active[-1])
    return ModalGuide.from_provider(
        provider,
        np.linspace(0.99 * lo, 1.01 * hi, 17),
        nodes=provider.node_rule(0.99 * lo, n_r=24),
        degenerate_blocks=((0, 1),),
    )


def build_case(name) -> tuple[FieldGrid, ModalGuide, KerrForm]:
    """Return the grid, the guide and the declared cubic of one parity case.

    The delayed case runs on the shared 116 fs window: its physics wants a longer
    one (``raman_strand`` in ``tests/backends/test_general_modal.py``), and the
    parity of two backends' arithmetic does not.
    """
    if name in {"pair", "pair_analytic", "circular"}:
        grid = general_grid()
        guide = general_guide(
            descriptors=PAIR,
            blocks=((0, 1),),
            grid=grid,
            circular=(0, 1) if name == "circular" else None,
        )
        return grid, guide, "analytic" if name == "pair_analytic" else "real"
    if name == "raman":
        grid = general_grid()
        return grid, general_guide(grid=grid, raman=SilicaTensorResponse()), "real"
    if name == "grin":
        grid = grin_grid()
        guide = grin_guide(
            grid, elements=64, knots=9, kerr=MaterialKerr(chi3=GENERAL_CHI3)
        )
        return grid, guide, "real"
    grid = general_grid()
    return grid, split_guide_with_kerr(grid), "real"


def launch(grid, n_modes, seed=41):
    """Return a band-limited random-phase launch of unit scale in every channel."""
    rng = np.random.default_rng(seed)
    width = 0.02 * grid.omega0
    envelope = np.exp(-(((grid.omega - grid.omega0) / width) ** 2)) * grid.active_mask
    state = np.zeros((n_modes, 1, grid.n_omega), dtype=complex)
    for channel in range(n_modes):
        state[channel, 0] = envelope * np.exp(
            1j * rng.uniform(0.0, 2 * np.pi, grid.n_omega)
        )
    return state


def problem_for(grid, guide, route, state, **overrides):
    """Assemble one case on the ten-step schedule."""
    arguments = {
        "controller": PIController(rtol=1e-8, atol=1e-14),
        "save_planes": [LENGTH],
        "schedule": FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        "dz_max": LENGTH / STEPS,
        "dz_init": LENGTH / STEPS,
        "general": route,
    } | overrides
    return assemble(grid, list(guide.mode_collection), [], state, **arguments)


def reference_source(maps, state, grid):
    """Return the reference backend's source for a state, shape (n_modes, n_omega)."""
    return general_source(
        state, maps, grid.crop_index, n_to=grid.n_to, pad_scale=grid.pad_scale
    )[:, 0]


def nonlinear_scale(maps, state, grid):
    """Return ``Phi = L ||N(a)|| / ||a||`` at a state, dimensionless."""
    source = reference_source(maps, state, grid)
    return float(
        LENGTH * grid.crop_scale * np.linalg.norm(source) / np.linalg.norm(state)
    )


def conditioning(maps: GeneralMaps, state, grid):
    """Return ``kappa``, the projected sum's conditioning, from the reference pass.

    The pass of ``general_source`` written out over every node at once, keeping the
    projected polarisation spectrum ``P`` so that ``|| sum |Q| |P| || / || sum Q P ||``
    can be formed.
    """
    crop, n_to = grid.crop_index, grid.n_to
    doubling = np.full(fourier.n_omega_for(n_to, "field"), 2.0)
    doubling[0] = doubling[-1] = 1.0
    scaled = state[:, 0, :] * (grid.pad_scale * doubling[crop])
    spectrum = np.einsum("qcmw,mw->qcw", maps.synthesis, scaled)
    buffer = np.zeros((*spectrum.shape[:2], n_to), dtype=complex)
    buffer[..., crop] = spectrum
    analytic = np.fft.ifft(buffer, axis=-1)
    field = np.real(analytic) if maps.form == "real" else analytic
    pair = np.einsum("qkt,qlt->qklt", field, field.conj())
    if maps.form == "real":
        polarisation = EPS0 * symmetric_cubic(
            symmetric_kerr_coefficients(maps.tensor), field
        )
    else:
        weighted = np.einsum("qijkl,qklt->qijt", maps.tensor, pair)
        polarisation = EPS0 * np.einsum("qijt,qjt->qit", weighted, field)
    if maps.n_terms:
        assert maps.node_response is not None
        drive = np.real(pair) if maps.form == "real" else 0.5 * np.real(pair)
        polarisation = polarisation + delayed_polarisation(
            field, drive, maps, maps.node_response
        )
    projected = fourier.forward(np.real(polarisation), kind="field")[..., crop]
    exact = np.einsum("mqcw,qcw->mw", maps.projection, projected)
    majorant = np.einsum("mqcw,qcw->mw", np.abs(maps.projection), np.abs(projected))
    return float(np.linalg.norm(majorant) / np.linalg.norm(exact))


def map_conditioning(terms, weights):
    """Return ``kappa_map`` of one moving map at one stage's weights.

    ``|| sum_t |w_t| |c_t| || / || sum_t w_t c_t ||`` over every element, with
    ``terms`` the map's coefficients (the leading axis the weights') and ``weights``
    the stage bundle's ``transform_weights``.
    """
    flat = terms.reshape(terms.shape[0], -1)
    exact = np.tensordot(weights, flat, axes=1)
    majorant = np.tensordot(abs(weights), abs(flat), axes=1)
    return float(np.linalg.norm(majorant) / np.linalg.norm(exact))


def moving_count(maps: GeneralMaps, stage_weights):
    """Return ``n_deg (3 kappa_S + kappa_Q)``, the largest over the stages given.

    Parameters
    ----------
    maps : GeneralMaps
        The problem's maps, moving.
    stage_weights : iterable of numpy.ndarray
        The ``transform_weights`` of every stage the run evaluates.

    Returns
    -------
    float
        What the per-stage map combination adds to :func:`source_bound`'s count, with
        ``n_deg`` the stage's nonzero weights.
    """
    assert maps.synthesis_terms is not None and maps.projection_terms is not None
    return max(
        np.count_nonzero(weights)
        * (
            3.0 * map_conditioning(maps.synthesis_terms, weights)
            + map_conditioning(maps.projection_terms, weights)
        )
        for weights in stage_weights
    )


def source_bound(maps: GeneralMaps, n_to, kappa, moving=0.0):
    """Return the relative bound on two backends' sources for one stage.

    Parameters
    ----------
    maps : GeneralMaps
        The problem's maps, which fix the channel and node counts.
    n_to : int
        Fine time samples.
    kappa : float
        The projected sum's conditioning, from :func:`conditioning`.
    moving : float, optional
        A moving guide's per-stage map combination, :func:`moving_count`; zero for a
        constant guide.

    Returns
    -------
    float
        ``2 kappa u (M + 2 eta log2(n_to) + 30 + 3 Q + 2 + moving)``, plus the
        doubled axis's pair and contraction when a delayed response is declared.
    """
    count = (
        maps.n_modes
        + 2 * FFT_CONSTANT * math.log2(n_to)
        + CONTRACTION
        + 3 * maps.n_nodes
        + 2
        + moving
    )
    if maps.n_terms:
        count += 2 * FFT_CONSTANT * math.log2(2 * n_to) + CONTRACTION
    return 2 * kappa * UNIT * count


def ten_step_bound(phi, source):
    """Return the relative bound on two backends' states after :data:`STEPS` steps.

    Parameters
    ----------
    phi : float
        The launch's nonlinear scale, from :func:`nonlinear_scale`.
    source : float
        The per-stage bound, from :func:`source_bound`.

    Returns
    -------
    float
        ``exp(Phi) (1.645 Phi eps_src + 10 * 32 u)``.
    """
    return math.exp(phi) * (RK_WEIGHTS * phi * source + STEPS * STEP_CONSTANT * UNIT)


def scaled_case(name):
    """Return a case's grid, guide, route and a launch scaled to :data:`PHI`."""
    grid, guide, form = build_case(name)
    route = general_modal_route(
        guide, grid, z_max=LENGTH, omega_reference=float(grid.omega0), kerr_form=form
    )
    state = launch(grid, len(guide.mode_collection))
    maps = _general_maps(problem_for(grid, guide, route, state))
    assert maps is not None
    state = state * math.sqrt(PHI / nonlinear_scale(maps, state, grid))
    return grid, guide, route, state
