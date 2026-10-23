# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The general modal guide's nonlinear source, on the reference backend.

This is a **second execution path beside** :mod:`solprop.backends.numpy_ref.passes`, not
an extension of it: the legacy pipeline synthesises with one real static matrix in the
time domain, and a general guide's synthesis is a complex, frequency-dependent map of
three Cartesian components, so it happens in the frequency domain instead.

The pipeline of one stage, for a chunk of quadrature nodes:

1. **Scale.** Multiply the coarse channel spectra by the pad scale ``n_to / n_t``,
   which keeps the two time-domain signals equal at coincident samples
   (``conventions.md`` section 3), and by the analytic-signal doubling. Both are
   applied here and not folded into ``S``, because ``S`` is the declared physical
   profile in V/(m sqrt(W)) that the guide's provider declares.
2. **Synthesise.** ``E(q, c, w) = sum_m S(q, c, m, w) a(m, w)`` at the chunk's nodes,
   on the **coarse** grid the state lives on, placed into the fine spectrum at
   ``grids.crop_index``; then one complex inverse transform per node and component of
   that **doubled** one-sided spectrum, which is the analytic signal ``E_a`` of
   ``conventions.md`` section 4 exactly: ``E(t) = Re E_a(t)`` and ``|E_a|^2`` is the
   instantaneous power. A complex profile can only act on an analytic signal.
   Synthesising before the padding rather than after it is the same arithmetic --
   every fine bin outside ``crop_index`` multiplies a zero amplitude -- over a map
   ``n_omega_fine / n_omega`` times smaller.
3. **Respond.** ``P_i = eps0 sum_jkl A(q)_ijkl E_j E_k conj(E_l)`` per node, with
   ``A(q)`` the sum of ``chi3 T`` over the responses of that node's material region
   (:func:`solprop.translate.responses.isotropic_kerr_tensor`). A node whose region
   names no response has no polarisation. Which field ``E`` is here is the record's
   own declaration, read back from its ``model_id``: the **real** field ``Re E_a``
   for the real cubic ``P = eps0 chi3 (E.E) E``, which is the legacy field route's
   model and radiates the third harmonic, or the analytic signal itself for
   ``P = eps0 chi3 (2 |E|^2 E + (E.E) E*) / 4``, which carries the fundamental band
   alone. The tensor carries the normalisation that belongs to each. The analytic
   form is contracted densely, over the pair history ``E_k conj(E_l)`` and then ``j``.
   The real cubic is symmetric in ``(j, k, l)``, so it is evaluated as **the symmetric
   contraction** (:func:`symmetric_cubic`): ten monomials ``(E_j E_k) E_l`` with
   ``j <= k <= l`` per sample, against coefficients that sum each node's tensor over
   each monomial's orderings (:func:`symmetric_kerr_coefficients`), formed once per
   node and evaluation. That is exact for any tensor, and it is the reduction order
   every backend runs.
4. **Project.** The physical polarisation is ``Re P_a``, so its one-sided spectrum --
   the same real forward transform every other route takes -- is what drives the
   one-sided state. Contract it with ``Q`` on the coarse grid and multiply by the
   record's ``source_frequency_factor``, ``-i omega``.

The crop scale ``n_t / n_to`` stays in ``responses.prefactor``, where it is on every
route, and the spectral collar
and the interaction picture are applied by the shared
:func:`solprop.backends.numpy_ref.passes.stage_output`.

**The delayed response.** A material region may declare the silica tensor Raman model
beside its Kerr response (:class:`solprop.nonlinear.raman.SilicaTensorResponse`), and
step 3 then adds its polarisation: **six** real symmetric pair histories
``Q_kl = E_k E_l`` per node, convolved on a doubled axis with that response's trace
and rank-two kernels, and contracted with ``chi3 D[term]`` and the local field. The
instantaneous tensor already carries the ``1 - f_R`` that leaves, so the two shares
add to the region's one ``chi3``.

**What this route does not do.** The
analytic-signal form radiates no third harmonic, which is why it is not the route's
default: it remains selectable, and is the tested oracle
for the fundamental band that the real cubic is measured against.

**Memory in the real form.** The chunk buffers of step 3 are float64 rather than
complex128 when the field is real, and the symmetric contraction holds six pair
products, a monomial and two component-wide sums per node and sample in place of the
pair history and the contracted tensor (the pair history too when a delayed response
needs its drive): at most 22 doubles against the 36 that :func:`general_memory`'s
``general_node_pair`` and ``general_node_weighted`` rows count for three components,
so those rows are an upper bound in that form. The bound is the one a budget needs and
the ledger is not split by form. The symmetric coefficients are formed per chunk from
the tensor, ``n_components * n_monomials`` doubles a node, a transient like the
others the ledger leaves out.
"""

import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from jaxtyping import Complex, Float, Int32, Num

from solprop import fourier
from solprop.constants import EPS0
from solprop.exceptions import BackendError
from solprop.nonlinear.responses import dispersive_cubic
from solprop.translate.modal_tables import ModalTransformTables
from solprop.translate.responses import (
    DELAYED_MODEL_ID,
    KERR_MODEL_FORMS,
    MODEL_ID_SEPARATOR,
    DispersiveKerrTerms,
    KerrForm,
    RegionalResponseTables,
)

__all__ = [
    "BUFFERS_PER_GENERAL_NODE",
    "DELAYED_TERMS",
    "PAIR_BUFFERS_PER_GENERAL_NODE",
    "PAIR_HISTORIES",
    "GeneralMaps",
    "delayed_polarisation",
    "dispersive_polarisation",
    "expand_pair_history",
    "general_dispersive_buffers",
    "general_memory",
    "general_node_chunk",
    "general_source",
    "node_delayed_maps",
    "node_kerr_form",
    "node_kerr_tensor",
    "refuse_regional_responses",
    "symmetric_cubic",
    "symmetric_kerr_coefficients",
    "symmetric_monomials",
]

logger = logging.getLogger(__name__)

DELAYED_TERMS = 2
"""Delayed channels the silica tensor Raman model carries: trace and rank two."""

PAIR_HISTORIES = 6
"""Independent entries of the real symmetric pair history of three components.

``Q_kl = E_k E_l`` is symmetric, so ``xx, yy, zz, xy, xz, yz`` carry all of it and
the convolution -- the expensive part of a delayed response -- runs six times per
node and not nine.
"""

_UPPER = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
"""The six stored pair entries, in order."""

BUFFERS_PER_GENERAL_NODE = 4
"""Chunk buffers the general source holds per node, each ``n_components`` wide.

The synthesised fine spectrum, the analytic time field, the polarisation in time and
its one-sided spectrum.
"""

PAIR_BUFFERS_PER_GENERAL_NODE = 2
"""Chunk buffers the general source holds per node, each ``n_components`` **squared**.

The tensor contraction is written in two steps rather than as one four-operand
``einsum``, so that every intermediate it allocates is a buffer this module names:
the pair history ``E_k conj(E_l)`` and the tensor contracted against it. A single
``einsum`` with ``optimize=True`` allocates the same intermediates and names none of
them, which a memory ledger cannot report.

The static maps ``S`` and ``Q``, the per-node tensor, the dense E/H reference, its
reciprocal partners and the coefficient dual are whole-problem allocations, counted
row by row by :func:`general_memory`; they do not shrink with the chunk, which is why
the counts here and :data:`solprop.translate.modal.BUFFERS_PER_NODE` are different and
neither reuses the other.
"""


@dataclass(frozen=True, eq=False)
class GeneralMaps:
    """The arrays one general modal problem's source pass reads, at one position.

    Derived once at construction from the problem's leaves. The route propagates
    constant geometry, so the maps have one interval of one Chebyshev degree along z
    and there is nothing to evaluate per stage.

    Attributes
    ----------
    synthesis : numpy.ndarray
        ``S`` on the coarse grid, complex128, shape
        ``(n_nodes, n_components, n_modes, n_omega)``, in V/(m sqrt(W)).
    projection : numpy.ndarray
        ``Q`` on the coarse grid, complex128, shape
        ``(n_modes, n_nodes, n_components, n_omega)``, in V m/sqrt(W).
    tensor : numpy.ndarray
        The summed ``chi3 T`` of each node's material region, float64, shape
        ``(n_nodes, n_components, n_components, n_components, n_components)``, in
        m²/V². Zero at a node whose region does not respond.
    frequency_factor : numpy.ndarray
        ``-i omega``, complex128, shape ``(n_omega,)``, in rad/s.
    node_chunk : int
        Nodes the pass handles at a time, at least one.
    node_response : numpy.ndarray
        Which response owns each node, int32, shape ``(n_nodes,)``, ``-1`` where a
        node's region declares none. It is what lets the delayed convolution run
        once per response over that response's own nodes, with its own kernel.
    delayed_tensor : numpy.ndarray
        ``chi3_r D[term]`` of each response and channel, float64, shape
        ``(n_responses, n_terms, n_components, n_components, n_components,
        n_components)``, in m²/V². Empty when no region declares a delayed
        response.
    delayed_spectrum : numpy.ndarray
        The real transform of each kernel on the doubled fine axis, times the
        temporal weight, complex128, shape ``(n_responses, n_terms, n_to + 1)``, in
        s/s. Empty with `delayed_tensor`.
    form : {'analytic', 'real'}
        Which field the tensor is contracted against: the analytic signal, whose
        polarisation carries the fundamental band alone, or the real
        three-component field, whose cubic carries the third harmonic. Read from
        the record's ``model_id`` by :func:`node_kerr_form`; `tensor` already
        carries the normalisation that goes with it.
    dispersive : solprop.translate.responses.DispersiveKerrTerms or None
        The gas regions' dispersive channels (SEP 6), each acting at the nodes of
        its own region, or ``None``.
    node_region : numpy.ndarray or None
        Each node's material region, int32, shape ``(n_nodes,)``: where each
        dispersive channel acts. Present with `dispersive`.
    """

    synthesis: Complex[np.ndarray, "n_nodes_s n_components_s n_modes_s n_omega_s"]
    projection: Complex[np.ndarray, "n_modes_q n_nodes_q n_components_q n_omega_q"]
    tensor: Float[
        np.ndarray,
        "n_nodes_t n_components_t_i n_components_t_j n_components_t_k n_components_t_l",
    ]
    frequency_factor: Complex[np.ndarray, " n_omega_factor"]
    node_chunk: int
    form: KerrForm = "analytic"
    node_response: Int32[np.ndarray, " n_nodes_response"] | None = None
    delayed_tensor: (
        Float[
            np.ndarray,
            "n_responses_delayed n_terms_delayed n_components_delayed_i"
            " n_components_delayed_j n_components_delayed_k n_components_delayed_l",
        ]
        | None
    ) = None
    delayed_spectrum: (
        Complex[np.ndarray, "n_responses_spectrum n_terms_spectrum n_rfft_spectrum"]
        | None
    ) = None
    synthesis_terms: (
        Complex[
            np.ndarray,
            "n_map_terms_s n_nodes_ms n_components_ms n_modes_ms n_omega_ms",
        ]
        | None
    ) = None
    projection_terms: (
        Complex[
            np.ndarray,
            "n_map_terms_q n_modes_mq n_nodes_mq n_components_mq n_omega_mq",
        ]
        | None
    ) = None
    dispersive: DispersiveKerrTerms | None = None
    node_region: Int32[np.ndarray, " n_nodes_region"] | None = None

    @property
    def moving(self) -> bool:
        """Return whether the maps move along z.

        Returns
        -------
        bool
            True when the synthesis and projection are Chebyshev terms a stage
            combines over its ``transform_weights``.
        """
        return self.synthesis_terms is not None

    def at(self, weights: Float[np.ndarray, " n_map_weights"] | None) -> "GeneralMaps":  # noqa: UP037 - the hook resolves the class while its body runs
        """Return the maps a stage evaluates at, from its ``transform_weights``.

        A constant guide's maps are returned as they are, whatever `weights` says,
        which keeps its arithmetic bit for bit what it was. A moving guide's are the
        weighted sums of the stored coefficients, exactly as ``dPhi`` is the weighted
        sum of its terms (``docs/dev/kernel-contract.md`` section 1).

        Parameters
        ----------
        weights : numpy.ndarray or None
            The stage bundle's ``transform_weights``, float64, one per term; ``None``
            for maps that do not move.

        Returns
        -------
        GeneralMaps
            The maps at the stage's position.

        Raises
        ------
        BackendError
            If moving maps are asked for with no weights, or constant ones with some.
        """
        if self.synthesis_terms is None or self.projection_terms is None:
            if weights is not None:
                msg = (
                    "a stage carries transform_weights for maps that do not move; the "
                    "bundle and the tables come from different problems"
                )
                raise BackendError(msg)
            return self
        if weights is None:
            msg = (
                "a moving guide's maps are evaluated from the stage bundle's "
                "transform_weights, and this stage carries none"
            )
            raise BackendError(msg)
        return replace(
            self,
            synthesis=np.tensordot(weights, self.synthesis_terms, axes=1),
            projection=np.tensordot(weights, self.projection_terms, axes=1),
            synthesis_terms=None,
            projection_terms=None,
        )

    @property
    def n_terms(self) -> int:
        """Return how many delayed channels the maps carry.

        Returns
        -------
        int
            Zero when no region declares a delayed response.
        """
        return 0 if self.delayed_tensor is None else int(self.delayed_tensor.shape[1])

    @property
    def n_nodes(self) -> int:
        """Return how many quadrature nodes the guide's rule carries.

        Returns
        -------
        int
            The node axis of ``synthesis``.
        """
        return int(self.synthesis.shape[0])

    @property
    def n_modes(self) -> int:
        """Return how many channels the guide propagates.

        Returns
        -------
        int
            The channel axis of ``synthesis``.
        """
        return int(self.synthesis.shape[2])


def refuse_regional_responses(
    tables: ModalTransformTables,
    regional: RegionalResponseTables | None,
    *,
    backend: str = "numpy_ref",
) -> None:
    """Refuse a per-region response this route does not execute, before it allocates.

    Called from the backend's admission list, where it runs before the driver's
    buffers exist, and again from :func:`node_kerr_tensor`, which is public and must
    not build a tensor from a record it cannot read. Neither call allocates.

    A component count that disagrees with the reference is the dangerous one: a
    one-component tensor **broadcasts** into the three-component contraction and
    propagates a meaningless polarisation, and a two-component one dies inside
    ``einsum`` with a shape message that names no leaf.

    Parameters
    ----------
    tables : solprop.translate.modal_tables.ModalTransformTables
        The guide's reference, which fixes the component count.
    regional : solprop.translate.responses.RegionalResponseTables or None
        The compiled per-region responses, or ``None`` for a linear guide.
    backend : str, optional
        The registry name each refusal names: ``numba_cpu`` executes the same
        responses and runs this same list.

    Raises
    ------
    BackendError
        Naming the leaf and what about it is not executed.
    """
    if regional is None:
        return
    components = int(tables.electric.shape[2])
    if regional.n_components != components:
        msg = (
            f"responses.regional_responses contracts over {regional.n_components} "
            f"Cartesian component(s) and transform.modal_tables resolves "
            f"{components}; {backend} executes the general modal route over the "
            "reference's own components, and a narrower tensor would broadcast into "
            "a polarisation that means nothing"
        )
        raise BackendError(msg)
    if regional.chi3.shape[0] != 1 or regional.chi3.shape[1] != 1:
        msg = (
            f"{backend} executes a general modal guide of constant geometry, and "
            f"responses.regional_responses.chi3 carries {regional.chi3.shape[0]} "
            f"interval(s) of degree {regional.chi3.shape[1] - 1} along z; a varying "
            "susceptibility is not supported on this route"
        )
        raise BackendError(msg)
    tags = regional.model_id.split(MODEL_ID_SEPARATOR)
    unknown = [one for one in tags[1:] if one != DELAYED_MODEL_ID]
    if tags[0] not in KERR_MODEL_FORMS or unknown:
        msg = (
            f"responses.regional_responses declares model_id "
            f"{regional.model_id!r} and {backend} executes the instantaneous "
            f"isotropic cubic in its two declared forms, "
            f"{sorted(KERR_MODEL_FORMS)}, optionally beside "
            f"{DELAYED_MODEL_ID!r}; a model this package does not know must "
            "not be evaluated as one it does"
        )
        raise BackendError(msg)
    if (DELAYED_MODEL_ID in tags) != bool(regional.n_terms):
        msg = (
            f"responses.regional_responses declares model_id "
            f"{regional.model_id!r} and {regional.n_terms} delayed term(s): a "
            "record's model identity and its term axis must agree, or the run "
            "would execute a model its own file does not name"
        )
        raise BackendError(msg)
    if regional.n_terms and regional.n_terms != DELAYED_TERMS:
        msg = (
            f"{backend} executes the {DELAYED_TERMS} delayed channels of the "
            f"silica tensor Raman model -- the trace and the traceless rank-two "
            f"one -- and this record carries {regional.n_terms}"
        )
        raise BackendError(msg)
    weight = regional.temporal_weight
    if regional.n_terms and not np.all(weight == weight[0]):
        msg = (
            "responses.regional_responses.temporal_weight is not uniform, and a "
            "delayed response is evaluated as a transform product on the doubled "
            "fine axis, which is a convolution only against a uniform quadrature "
            "weight; a non-uniform temporal grid needs a different evaluation"
        )
        raise BackendError(msg)


def node_kerr_tensor(
    tables: ModalTransformTables, regional: RegionalResponseTables | None
) -> Float[np.ndarray, "n_nodes 3 3 3 3"]:
    """Sum each node's material responses into one susceptibility tensor.

    A response acts in one material region and a node belongs to exactly one, so the
    tensor a node sees is ``sum_r chi3_r T_r`` over the responses of its region. Two
    responses in one region add, which is what superposed instantaneous
    susceptibilities do; a region no response names contributes nothing, which is how
    a linear cladding is declared.

    ``chi3`` is stored as Chebyshev coefficients along z. This route propagates
    constant geometry, so one interval of degree zero is required and the single
    coefficient is the value.

    Parameters
    ----------
    tables : solprop.translate.modal_tables.ModalTransformTables
        The guide's reference, whose ``region_index`` owns each node's region.
    regional : solprop.translate.responses.RegionalResponseTables or None
        The compiled per-region responses, or ``None`` for a linear guide.

    Returns
    -------
    numpy.ndarray
        The per-node tensor, float64, shape ``(n_nodes, 3, 3, 3, 3)``, in m²/V².

    Raises
    ------
    BackendError
        If the responses do not resolve the reference's components, the
        susceptibility varies along z, or a delayed term is declared.
    """
    refuse_regional_responses(tables, regional)
    nodes = int(tables.region_index.size)
    components = int(tables.electric.shape[2])
    summed = np.zeros((nodes, *(components,) * 4))
    if regional is None:
        return summed
    for index, region in enumerate(regional.response_region):
        rows = tables.region_index == region
        chi3 = float(regional.chi3[0, 0, index])
        summed[rows] += chi3 * regional.instantaneous_tensor[index]
    return summed


def node_kerr_form(regional: RegionalResponseTables | None) -> KerrForm:
    """Return which cubic a compiled per-region record declares.

    The form is a physical-model choice and travels in ``model_id``, which is a
    representation tag inside the resume identity already; it is **not** a leaf
    field, because adding one would move a contract row and bump the kernel schema
    for a string the record can carry.

    Parameters
    ----------
    regional : solprop.translate.responses.RegionalResponseTables or None
        The compiled per-region responses, or ``None`` for a linear guide, whose
        zero tensor makes the form immaterial.

    Returns
    -------
    {'analytic', 'real'}
        The declared form; ``"analytic"`` for a linear guide.

    Raises
    ------
    BackendError
        If the record declares a model identity this backend does not execute.
    """
    if regional is None:
        return "analytic"
    form = KERR_MODEL_FORMS.get(regional.model_id.split(MODEL_ID_SEPARATOR)[0])
    if form is None:
        msg = (
            f"responses.regional_responses declares model_id "
            f"{regional.model_id!r} and numpy_ref executes the instantaneous "
            f"isotropic cubic in its two declared forms, "
            f"{sorted(KERR_MODEL_FORMS)}"
        )
        raise BackendError(msg)
    return cast("KerrForm", form)


def symmetric_monomials(n_components: int) -> tuple[tuple[int, int, int], ...]:
    """Return the real cubic's monomials ``(j, k, l)``, ``j <= k <= l``, in order.

    The order is lexicographic, and it is the order :func:`symmetric_cubic` sums them
    in on every backend: ten for three components, four for two, one for one.

    Parameters
    ----------
    n_components : int
        Cartesian components of the local field, one to three.

    Returns
    -------
    tuple
        The component triples.

    Examples
    --------
    >>> from solprop.backends.numpy_ref.general_modal import symmetric_monomials
    >>> symmetric_monomials(2)
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1))
    """
    return tuple(itertools.combinations_with_replacement(range(n_components), 3))


def symmetric_kerr_coefficients(
    tensor: Float[
        np.ndarray,
        "n_nodes n_components_i n_components_j n_components_k n_components_l",
    ],
) -> Float[np.ndarray, "n_nodes n_components n_monomials"]:
    """Sum a per-node tensor over the orderings of each monomial of the real cubic.

    On a real field ``sum_jkl A_ijkl E_j E_k E_l = sum_m C_im (E_j E_k) E_l`` over
    the monomials of :func:`symmetric_monomials`, with ``C_im`` the sum of
    ``A_ijkl`` over every ordered triple that sorts to monomial ``m``: one ordering
    for ``x x x``, three for ``x x y``, six for ``x y z``. That is an identity for
    any tensor, symmetric or not. The orderings are added in lexicographic order of
    ``(j, k, l)``, each sum sequential from zero, which is the order every backend
    forms them in; a coefficient is within ``gamma_5`` of the exact sum relative to
    the sum of its terms' magnitudes.

    Parameters
    ----------
    tensor : numpy.ndarray
        The per-node ``chi3 T``, float64, shape
        ``(n_nodes, n_components, n_components, n_components, n_components)``, in
        m²/V².

    Returns
    -------
    numpy.ndarray
        The coefficients, float64, shape ``(n_nodes, n_components, n_monomials)``,
        in m²/V².

    Examples
    --------
    >>> from solprop.backends.numpy_ref.general_modal import (
    ...     symmetric_kerr_coefficients,
    ... )
    >>> from solprop.translate.responses import isotropic_kerr_tensor
    >>> coefficients = symmetric_kerr_coefficients(isotropic_kerr_tensor("real")[None])
    >>> coefficients[0, 0, :4]  # x: xxx, xxy, xxz, xyy
    array([1., 0., 0., 1.])
    """
    components = int(tensor.shape[1])
    monomials = symmetric_monomials(components)
    index = {monomial: position for position, monomial in enumerate(monomials)}
    coefficients = np.zeros((tensor.shape[0], components, len(monomials)))
    for j, k, last in itertools.product(range(components), repeat=3):
        low, middle, high = sorted((j, k, last))
        coefficients[:, :, index[low, middle, high]] += tensor[:, :, j, k, last]
    return coefficients


def symmetric_cubic(
    coefficients: Float[np.ndarray, "n_nodes n_components n_monomials"],
    field: Float[np.ndarray, "n_nodes n_components n_to"],
) -> Float[np.ndarray, "n_nodes n_components n_to"]:
    """Return the real cubic ``sum_jkl A_ijkl E_j E_k E_l`` by its symmetric form.

    The reduction order of the real cubic on every backend: each sample forms the
    pair products ``E_j E_k`` (``j <= k``), each monomial as ``(E_j E_k) E_l``, and
    each component as ``sum_m C_im monomial_m`` over the monomials in their order,
    sequential from zero. Against the dense ``sum_j (sum_kl A_ijkl E_k E_l) E_j``
    that is 3 + 10 products a component in place of 81 + 3.

    **Rounding.** With ``M_i = sum_jkl |A_ijkl| |E_j| |E_k| |E_l|``: a coefficient
    is within ``gamma_5`` of its exact sum, a monomial within ``gamma_2``, and the
    ten-term sum of products within ``gamma_10``, so the component is within
    ``gamma_17 M_i`` of exact arithmetic; the dense order is within ``gamma_14``
    (the pair, nine products, the field and three terms), so the two orders differ
    by at most ``gamma_31 M_i``.

    Parameters
    ----------
    coefficients : numpy.ndarray
        :func:`symmetric_kerr_coefficients` of the nodes' tensors, in m²/V².
    field : numpy.ndarray
        The real field at the nodes, float64, shape
        ``(n_nodes, n_components, n_to)``, in V/m.

    Returns
    -------
    numpy.ndarray
        The cubic before ``eps0``, float64, shape ``(n_nodes, n_components, n_to)``,
        in V/m.
    """
    components = int(field.shape[1])
    total = np.zeros(field.shape)
    # A diverging run overflows here before the driver's norm check names it; the
    # dense `einsum` this replaced reported no floating-point state, so neither does
    # this, and the non-finite value reaches that check as it did.
    with np.errstate(over="ignore", invalid="ignore"):
        pairs = {
            (j, k): field[:, j] * field[:, k]
            for j, k in itertools.combinations_with_replacement(range(components), 2)
        }
        for position, (j, k, last) in enumerate(symmetric_monomials(components)):
            monomial = pairs[j, k] * field[:, last]
            total += coefficients[:, :, position, None] * monomial[:, None, :]
    return total


def node_delayed_maps(
    tables: ModalTransformTables, regional: RegionalResponseTables | None
) -> tuple[
    Int32[np.ndarray, " n_nodes"],
    Float[np.ndarray, "n_responses n_terms 3 3 3 3"] | None,
    Complex[np.ndarray, "n_responses n_terms n_rfft"] | None,
]:
    """Derive the delayed response's per-node ownership, tensors and kernel spectra.

    A delayed response has **memory**, so unlike the instantaneous tensor it cannot
    be summed into one per-node array: two regions may declare different kernels and
    a convolution is not linear in the kernel's owner. What is derived instead is
    the response each node belongs to, each response's ``chi3_r D[term]``, and each
    kernel's transform on the doubled fine axis, already scaled by the temporal
    quadrature weight so that the transform product is the causal integral.

    Parameters
    ----------
    tables : solprop.translate.modal_tables.ModalTransformTables
        The guide's reference, whose ``region_index`` owns each node's region.
    regional : solprop.translate.responses.RegionalResponseTables or None
        The compiled per-region responses, or ``None`` for a linear guide.

    Returns
    -------
    tuple
        The per-node response index (``-1`` where a node's region declares none),
        the scaled delayed tensors and the kernel spectra, the last two ``None``
        when no region declares a delayed response.

    Raises
    ------
    BackendError
        Naming the leaf and what about it is not executed.
    """
    refuse_regional_responses(tables, regional)
    nodes = int(tables.region_index.size)
    owner = np.full(nodes, -1, dtype=np.int32)
    if regional is None:
        return owner, None, None
    for index, region in enumerate(regional.response_region):
        owner[tables.region_index == region] = index
    if not regional.n_terms:
        return owner, None, None
    n_to = int(regional.temporal_weight.size)
    scaled = (
        regional.delayed_tensor * regional.chi3[0, 0][:, None, None, None, None, None]
    )
    spectra = np.fft.rfft(regional.delayed_kernel[0, 0], n=2 * n_to, axis=-1) * float(
        regional.temporal_weight[0]
    )
    return owner, scaled, spectra


def expand_pair_history(
    stored: Float[np.ndarray, "n_nodes 6 n_to"],
) -> Float[np.ndarray, "n_nodes 3 3 n_to"]:
    """Return the full symmetric pair history from its six independent entries.

    Parameters
    ----------
    stored : numpy.ndarray
        The six entries ``xx, yy, zz, xy, xz, yz``, shape
        ``(n_nodes, 6, n_to)``, in V²/m².

    Returns
    -------
    numpy.ndarray
        The symmetric ``(n_nodes, 3, 3, n_to)`` history, in V²/m².
    """
    full = np.empty((stored.shape[0], 3, 3, stored.shape[2]), dtype=stored.dtype)
    for entry, (row, column) in enumerate(_UPPER):
        full[:, row, column] = stored[:, entry]
        full[:, column, row] = stored[:, entry]
    return full


def delayed_polarisation(
    field: Num[np.ndarray, "n_nodes 3 n_to"],
    pair: Float[np.ndarray, "n_nodes 3 3 n_to"],
    maps: GeneralMaps,
    owner: Int32[np.ndarray, " n_nodes"],
) -> Float[np.ndarray, "n_nodes 3 n_to"]:
    """Return one chunk's delayed polarisation, in C/m^2, from its pair history.

    Six real symmetric histories per node, convolved with each response's own
    kernel on the **doubled** fine axis -- so the tail of a hundred-femtosecond
    response cannot wrap onto the head of the drive -- and contracted with that
    response's ``chi3 D[term]`` and the local field.

    The loop is over **responses** rather than nodes because a convolution is not
    linear in its kernel's owner: two material regions may declare different delayed
    models, and each one's nodes are convolved with its own kernel.

    Parameters
    ----------
    field : numpy.ndarray
        The local field at the chunk's nodes, shape ``(n_nodes, 3, n_to)``, V/m:
        the real field for the real cubic and the analytic signal otherwise.
    pair : numpy.ndarray
        The **real** drive ``Q_kl``, shape ``(n_nodes, 3, 3, n_to)``, V²/m².
    maps : GeneralMaps
        The problem's maps, carrying the delayed tensors and kernel spectra.
    owner : numpy.ndarray
        Which response owns each node of the chunk, ``-1`` for none.

    Returns
    -------
    numpy.ndarray
        The delayed polarisation, shape ``(n_nodes, 3, n_to)``, C/m².
    """
    tensors, spectra = maps.delayed_tensor, maps.delayed_spectrum
    assert tensors is not None and spectra is not None
    n_to = int(pair.shape[-1])
    total = np.zeros(field.shape, dtype=np.float64)
    for response in range(int(tensors.shape[0])):
        rows = owner == response
        if not rows.any() or not np.any(spectra[response]):
            continue
        stored = np.stack([pair[rows, row, column] for row, column in _UPPER], axis=1)
        drive = np.fft.rfft(stored, n=2 * n_to, axis=-1)
        for term in range(int(tensors.shape[1])):
            history = expand_pair_history(
                np.fft.irfft(spectra[response, term] * drive, n=2 * n_to, axis=-1)[
                    ..., :n_to
                ]
            )
            total[rows] += np.einsum(
                "ijkl,qklt,qjt->qit",
                tensors[response, term],
                history,
                np.real(field[rows]),
            )
    return EPS0 * total


def dispersive_polarisation(
    field: Float[np.ndarray, "n_nodes n_components n_to"],
    terms: DispersiveKerrTerms,
    node_region: Int32[np.ndarray, " n_nodes"],
    densities: Sequence[float],
) -> Float[np.ndarray, "n_nodes n_components n_to"]:
    """Return one chunk's dispersive polarisation, in C/m^2, region by region.

    Each channel of SEP 6's dispersive cubic -- one species of one region's gas fill
    -- acts at the nodes of its own region, on the **real** three-component field:
    ``P = coefficient rho cubic(E)`` for the polynomial, and for the symmetric
    evaluator one cubic per region, under its gas fill's mixture filter
    (:meth:`~solprop.translate.responses.DispersiveKerrTerms.evaluations`), by
    :func:`solprop.nonlinear.responses.dispersive_cubic`, the arithmetic of the
    reference :meth:`~solprop.nonlinear.responses.DispersiveKerrResponse.evaluate`,
    whose isotropic Kleinman contraction is the real cubic's tensor (SEP 6, decision
    Q3). The sector is the leaf's: the full cubic beside ``kerr_form="real"``, the
    number-conserving one beside ``"analytic"``, and both are real polarisations,
    which the projection takes as it takes ``Re P_a``.

    Parameters
    ----------
    field : numpy.ndarray
        The real field at the chunk's nodes, ``Re E_a``, shape
        ``(n_nodes, n_components, n_to)``, in V/m.
    terms : solprop.translate.responses.DispersiveKerrTerms
        The compiled channels, carrying their node regions.
    node_region : numpy.ndarray
        Each node's material region.
    densities : sequence of float
        Each channel's number density, m^-3.

    Returns
    -------
    numpy.ndarray
        The polarisation, shape ``(n_nodes, n_components, n_to)``, C/m^2; zero at a
        node whose region has no channel.
    """
    assert terms.regions is not None
    total = np.zeros(field.shape)
    for channel, factor, weights in terms.evaluations(densities):
        nodes = node_region == terms.regions[channel]
        if not nodes.any():
            continue
        total[nodes] += factor * dispersive_cubic(
            field[nodes], weights, kind="field", sector=terms.sector
        )
    return total


def general_dispersive_buffers(leaf: DispersiveKerrTerms | None) -> int:
    """Return the buffers per node and component SEP 6's response adds to a chunk.

    The compiled general workspace holds, for each node of a chunk and each of its
    components, every filtered field's spectrum and time series and every product and
    its spectrum: two complex-equivalent buffers per field and two per product. The
    symmetric evaluator has one field and one product per region with a gas fill (a
    mixture's under its mixture filter); the polynomial two fields and two products per
    group of equal leg bound, and one cube per region
    (:meth:`~solprop.translate.responses.DispersiveKerrTerms.groups`).

    Parameters
    ----------
    leaf : solprop.translate.responses.DispersiveKerrTerms or None
        The route's dispersive channels, or ``None``.

    Returns
    -------
    int
        Complex-equivalent buffers per node and component; zero without the leaf.
    """
    if leaf is None:
        return 0
    groups = len(leaf.groups())
    if leaf.evaluator == "symmetric":
        fields = products = groups
    else:
        regions = 1 if leaf.regions is None else len(set(leaf.regions.tolist()))
        fields, products = 2 * groups, regions + 2 * groups
    return 2 * (fields + products)


def general_node_chunk(
    n_components: int,
    n_to: int,
    n_omega_fine: int,
    budget: int,
    *,
    dispersive: int = 0,
) -> int:
    """Return the nodes the general source may hold at once within a byte budget.

    The chunk bounds the space-domain work alone:
    :data:`BUFFERS_PER_GENERAL_NODE` complex buffers per node and component,
    :data:`PAIR_BUFFERS_PER_GENERAL_NODE` per node and component squared, and SEP 6's
    dispersive response's own (:func:`general_dispersive_buffers`), over the longer of
    the fine time axis and the fine spectrum. The maps, the reference and the dual are
    whole-problem allocations that no chunk shrinks (:func:`general_memory`).

    Parameters
    ----------
    n_components : int
        Cartesian components of the local field, positive.
    n_to : int
        Fine time samples, positive.
    n_omega_fine : int
        Fine spectral bins, positive.
    budget : int
        Bytes the chunk buffers may occupy in total, positive.
    dispersive : int, optional
        The dispersive response's complex buffers per node and component; zero, the
        default, without one.

    Returns
    -------
    int
        The chunk size, at least one.

    Raises
    ------
    BackendError
        If the budget is not positive.

    Examples
    --------
    >>> from solprop.backends.numpy_ref.general_modal import general_node_chunk
    >>> general_node_chunk(3, 1024, 513, 64 * 2**20)
    136
    """
    if budget <= 0:
        msg = f"the general node budget must be positive, got {budget!r}"
        raise BackendError(msg)
    samples = max(n_to, n_omega_fine)
    columns = (BUFFERS_PER_GENERAL_NODE + dispersive) * n_components + (
        PAIR_BUFFERS_PER_GENERAL_NODE * n_components**2
    )
    per_node = columns * samples * np.dtype(np.complex128).itemsize
    return int(max(1, budget // per_node))


def general_memory(
    maps: GeneralMaps, tables: ModalTransformTables, n_to: int
) -> dict[str, int]:
    """Return the bytes the general source holds, one row per named buffer.

    Three groups. The **execution maps** ``S`` and ``Q`` and the per-node tensor are
    derived once. The **reference** the record carries -- the dense E and H, their
    reciprocal partners and the coefficient dual -- is resident for the whole run
    because the problem owns it, and is counted here although this pass reads none of
    it: it is on the far side of the seam, it is the largest group on a guide with
    many channels, and a budget that did not see it would be wrong by that much. The
    **chunk buffers** are the pass's own, and shrink with ``node_chunk``.

    Per-stage transients are **not** counted: the doubling weights, the inverse
    transform's output beside the zero-filled buffer it replaces, the conjugated
    field, the contiguous copy ``rfft`` makes of a strided real view and its fine
    one-sided output before the crop. Measured with ``tracemalloc`` on a 288-node
    strand, they add **6.2 %** to the rows below at their
    peak (2.52 MiB over 40.9 MiB); each is freed inside the pass.

    Parameters
    ----------
    maps : GeneralMaps
        The derived maps of one problem.
    tables : solprop.translate.modal_tables.ModalTransformTables
        The record the maps came from, whose stored reference is resident.
    n_to : int
        Fine time samples of the run's grid, positive.

    Returns
    -------
    dict
        Byte counts keyed by buffer name, each counted once: ``general_synthesis``,
        ``general_projection`` -- every coefficient of a moving guide's maps --
        ``general_stage_maps``, the two maps a moving guide's stage evaluates (zero
        otherwise), and ``general_tensor``; the resident reference
        ``general_electric``, ``general_magnetic``, ``general_backward_electric``,
        ``general_backward_magnetic``, ``general_reciprocal_gram`` and
        ``general_coordinate_dual``, zero where the record leaves one empty; and the
        chunk buffers ``general_scaled_state``, ``general_node_spectrum``,
        ``general_node_field``, ``general_node_pair``, ``general_node_weighted``,
        ``general_node_polarisation`` and ``general_node_source``.
    """
    item = np.dtype(np.complex128).itemsize
    real = np.dtype(np.float64).itemsize
    chunk = maps.node_chunk
    components = int(maps.synthesis.shape[1])
    n_omega = int(maps.synthesis.shape[3])
    optional = ("backward_electric", "backward_magnetic", "reciprocal_gram")
    resident = {
        f"general_{name}": int(getattr(tables, name).nbytes)
        if getattr(tables, name) is not None
        else 0
        for name in ("electric", "magnetic", *optional, "coordinate_dual")
    }
    delayed = {
        # The delayed pass holds, per node of a chunk: the real drive, the six
        # stored pair histories, their doubled transform and one convolved history
        # per channel. The kernel spectra and the scaled tensors are whole-problem
        # allocations and are counted with the maps.
        "general_node_drive": chunk * components**2 * n_to * real,
        "general_node_histories": chunk * PAIR_HISTORIES * n_to * real,
        "general_node_drive_spectrum": chunk * PAIR_HISTORIES * (n_to + 1) * item,
        "general_node_delayed": chunk * components * n_to * real,
        "general_delayed_tensor": 0
        if maps.delayed_tensor is None
        else int(maps.delayed_tensor.nbytes),
        "general_delayed_spectrum": 0
        if maps.delayed_spectrum is None
        else int(maps.delayed_spectrum.nbytes),
    }
    if not maps.n_terms:
        delayed = dict.fromkeys(delayed, 0)
    # A moving guide holds every coefficient of both maps for the whole run and
    # evaluates one of each per stage: the coefficients are its maps rows, and the
    # two evaluated maps a stage holds are a row of their own.
    stored = (
        (maps.synthesis, maps.projection)
        if maps.synthesis_terms is None or maps.projection_terms is None
        else (maps.synthesis_terms, maps.projection_terms)
    )
    return {
        "general_synthesis": int(stored[0].nbytes),
        "general_projection": int(stored[1].nbytes),
        "general_stage_maps": 0
        if not maps.moving
        else int(maps.synthesis.nbytes + maps.projection.nbytes),
        "general_tensor": int(maps.tensor.nbytes),
        **resident,
        "general_scaled_state": maps.n_modes * n_omega * item,
        "general_node_spectrum": chunk * components * n_omega * item,
        "general_node_field": chunk * components * n_to * item,
        "general_node_pair": chunk * components**2 * n_to * item,
        "general_node_weighted": chunk * components**2 * n_to * item,
        "general_node_polarisation": chunk * components * n_to * item,
        "general_node_source": chunk * components * n_omega * item,
        **delayed,
    }


def general_source(  # noqa: PLR0913 - a state, its maps and the stage's scalars
    state: Complex[np.ndarray, "n_modes n_pol n_omega"],
    maps: GeneralMaps,
    crop_index: np.ndarray,
    *,
    n_to: int,
    pad_scale: float,
    transform_weights: Float[np.ndarray, " n_map_weights"] | None = None,
    dispersive_densities: Sequence[float] = (),
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    r"""Evaluate one stage's modal source through the general guide's maps.

    The four steps of the module docstring, in the cubic ``maps.form`` declares,
    with the gas regions' dispersive channels added at step 3
    (:func:`dispersive_polarisation`).
    The result is the projected
    polarisation times ``-i omega``: ``Q`` is in V m/sqrt(W) and ``P`` in C/m^2, so
    the product carries ``C V m^-1 W^(-1/2) s^-1``, and ``C V`` is a joule -- the
    unit is **sqrt(W)/m**, the amplitude rate ``da/dz`` itself. The crop scale
    ``n_t / n_to`` that ``responses.prefactor`` carries is a dimensionless grid
    convention and changes the number but not the unit.

    Parameters
    ----------
    state : numpy.ndarray
        The lab-frame channel spectra, complex128, shape
        ``(n_modes, n_pol, n_omega)``, in the square root of watts, with ``n_pol``
        one.
    maps : GeneralMaps
        The guide's synthesis, projection, per-node tensor and frequency factor.
    crop_index : numpy.ndarray
        Where each coarse bin sits in the fine spectrum, ``numpy.intp``, shape
        ``(n_omega,)``.
    n_to : int
        Fine time samples, positive and even.
    pad_scale : float
        ``n_to / n_t``, the oversampling scale of ``conventions.md`` section 3.
    transform_weights : numpy.ndarray or None, optional
        The stage bundle's weights of a moving guide's maps (:meth:`GeneralMaps.at`);
        ``None`` for maps that do not move.
    dispersive_densities : sequence of float, optional
        Each dispersive channel's number density at the stage, m^-3, in the
        channels' order: the stage bundle's ``dispersive_kerr_densities``.

    Returns
    -------
    numpy.ndarray
        The modal source on the coarse grid, complex128, shape
        ``(n_modes, n_pol, n_omega)``, in sqrt(W)/m before the crop scale.

    Raises
    ------
    BackendError
        If the densities do not match the dispersive channels.
    """
    maps = maps.at(transform_weights)
    channels = 0 if maps.dispersive is None else maps.dispersive.n_channels
    if len(dispersive_densities) != channels:
        msg = (
            f"the general route carries {channels} dispersive channel(s) and the "
            f"stage {len(dispersive_densities)} densities"
        )
        raise BackendError(msg)
    n_fine = fourier.n_omega_for(n_to, "field")
    # The analytic signal keeps the zero and Nyquist bins of the **fine** spectrum and
    # doubles the rest (`fourier.analytic_signal`), so forming it here costs one
    # weighting of a spectrum the pass builds anyway instead of a Hilbert transform per
    # node. Which coarse bins those are is `crop_index`'s to say: a field grid's coarse
    # Nyquist is an interior bin of the fine spectrum and is doubled like any other.
    doubling = np.full(n_fine, 2.0)
    doubling[0] = 1.0
    doubling[-1] = 1.0
    scaled = state[:, 0, :] * (pad_scale * doubling[crop_index])
    source = np.zeros((maps.n_modes, len(crop_index)), dtype=np.complex128)
    for first in range(0, maps.n_nodes, maps.node_chunk):
        rows = slice(first, first + maps.node_chunk)
        spectrum = np.einsum("qcmw,mw->qcw", maps.synthesis[rows], scaled)
        buffer = np.zeros((*spectrum.shape[:2], n_to), dtype=np.complex128)
        buffer[..., crop_index] = spectrum
        analytic = np.fft.ifft(buffer, axis=-1)
        # The real cubic is the same contraction on the real field, which is the
        # real part of the analytic signal the synthesis already built; the tensor
        # carries the normalisation that goes with each (`isotropic_kerr_tensor`),
        # and with a real field the conjugated last index is the field itself.
        field = np.real(analytic) if maps.form == "real" else analytic
        # The pair history `E_k conj(E_l)` is the analytic form's first step and the
        # delayed response's drive; the real cubic needs it only for the latter.
        pair = (
            np.einsum("qkt,qlt->qklt", field, field.conj())
            if maps.form == "analytic" or maps.n_terms
            else None
        )
        if maps.form == "real":
            coefficients = symmetric_kerr_coefficients(maps.tensor[rows])
            polarisation = EPS0 * symmetric_cubic(coefficients, field)
        else:
            assert pair is not None
            # Two named steps rather than one four-operand `einsum`: the pair
            # history and the tensor contracted against it are exactly the
            # intermediates an optimised `einsum` would allocate, and here the
            # memory ledger can name them (`general_memory`).
            weighted = np.einsum("qijkl,qklt->qijt", maps.tensor[rows], pair)
            polarisation = EPS0 * np.einsum("qijt,qjt->qit", weighted, field)
        if maps.n_terms:
            assert maps.node_response is not None and pair is not None
            # The Raman drive is the real symmetric pair history: `E_k E_l` for the
            # real cubic, which carries the 2 omega beat and so radiates the
            # harmonic through the later multiplication by E, and the beat-free
            # `Re(E_a,k conj(E_a,l))/2` for the analytic one -- exactly
            # `raman_drive`'s two field-grid choices, per component pair.
            drive = np.real(pair) if maps.form == "real" else 0.5 * np.real(pair)
            polarisation = polarisation + delayed_polarisation(
                field, drive, maps, maps.node_response[rows]
            )
        if maps.dispersive is not None:
            assert maps.node_region is not None
            polarisation = polarisation + dispersive_polarisation(
                np.real(analytic),
                maps.dispersive,
                maps.node_region[rows],
                dispersive_densities,
            )
        # The state is the one-sided spectrum of a real field, so what drives it is
        # the one-sided spectrum of the real polarisation. In the analytic form that
        # is `Re P_a`, which is half the analytic polarisation away from the zero
        # and Nyquist bins; in the real form the polarisation is already the real
        # one and `real` is a no-op that keeps one code path.
        projected = fourier.forward(np.real(polarisation), kind="field")[
            ..., crop_index
        ]
        source += np.einsum("mqcw,qcw->mw", maps.projection[:, rows], projected)
    return (source * maps.frequency_factor)[:, None, :]
