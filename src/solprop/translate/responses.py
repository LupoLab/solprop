# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Response definitions compiled into one fused time-domain pass.

A response says what polarisation a medium produces from the field at a point
(:mod:`solprop.nonlinear.responses`); a backend needs coefficients, a table and an
ordering it can execute without knowing any physics. This module performs that
compilation for the responses that exist: the instantaneous Kerr term, the delayed
Raman term, the ionisation rate as a spline table, the UPPE prefactor and the temporal
absorber. Section 5 of ``docs/dev/kernel-contract.md`` is the class
:class:`CompiledResponses`, field for field.

What is not compiled yet is refused rather than half-built. The spectral-domain
responses have a field in the contract and no compiler here, so it stays
empty and its flag stays false; anything else raises a
:class:`solprop.exceptions.TranslationError` naming the response, so nothing reaches a
backend that would evaluate a term it does not have. A response that declares it is
undefined on the problem's grid is refused with that reason: the plasma model is written
for a resolved carrier and has no envelope form.

**The Raman fraction is not this module's business.** A response that declares
``fraction`` is one whose *author* already reduced the Kerr coefficient beside it --
:func:`solprop.nonlinear.gnlse.kerr_susceptibility` does exactly that for the GNLSE --
and a molecular gas's Raman response is absolute, with the full electronic ``chi3``
beside it. Applying ``(1 - f_R)`` here would either double the reduction or invent one,
and would reduce the Kerr term even with Raman switched off.

**The Raman drive form is a property of the problem.** Whether a field grid's drive
keeps the ``2 omega`` beat is not a property of each response: a drive that keeps it
radiates a third harmonic through the later multiplication by the field, and a grid
built without one does not oversample enough to hold it. So the compiler checks that
the declared Kerr and Raman responses agree about ``thg`` on a field grid, and refuses
a field-grid Raman response with no Kerr response beside it. That is why the answer is
a field of :class:`RamanCoefficients` rather than of either response.

The prefactor carries the **crop** scale of the oversampling round trip; the pad scale
is in the synthesis, for the reason given in :mod:`solprop.translate.modal`. It is zero
outside the active band rather than one: raw, unnormalised polarisation left in the
out-of-band bins would be seen by every Runge-Kutta stage and every error estimate.

It also follows z. ``Re n_eff(omega, z)`` moves with the gas and the bore, so on a
gradient fill or a taper the prefactor is not one array but a path. Rather than being
evaluated afresh at every stage, it is resolved once, the way ``Phi`` is
and in whichever of ``Phi``'s two representations the mode set allows: an expansion in
the scalar paths the modes declare, with the plain products of the basis polynomials as
its weights because the prefactor is a value at z and not a cumulative one, or an
adaptive Chebyshev table of z over the same span and split on the same declared kinks.
Either is one pass per stage through :func:`prefactor_at`. A mode set that does not vary
along z carries the array instead, in ``prefactor_constant``, and a backend takes that
fast path.
"""

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Final, Literal, cast

import numpy as np
from jaxtyping import Bool, Complex, Float, Int32

from solprop._display import Displayed, line
from solprop._helpers import Helpers
from solprop.boundaries import Absorber
from solprop.constants import CHARGE_MASS_RATIO, EPS0
from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, TimeGrid
from solprop.modes.protocol import Mode, ParameterisedMode
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.ionisation import IonisationRate, RunningIntegral
from solprop.nonlinear.normalisation import sqrtw_to_vm, uppe_prefactor
from solprop.nonlinear.optmat_raman import OptmatRamanResponse
from solprop.nonlinear.raman import RamanResponse, SilicaTensorResponse
from solprop.nonlinear.responses import (
    DispersiveKerrResponse,
    DispersiveWeights,
    KerrResponse,
    PlasmaResponse,
    mixture_filter,
)
from solprop.physdata.mixtures import Mixture
from solprop.physdata.nonlinear_data import ElectronicRegions
from solprop.physdata.species import Species
from solprop.profiles import Scaled, Sum, ZProfile, as_profile
from solprop.translate.chebyshev import (
    DEFAULT_MAX_DEGREE,
    DEFAULT_MAX_INTERVALS,
    DEFAULT_TOL,
    PiecewiseChebyshev,
    build_from_paths,
)
from solprop.translate.envelope_plasma import (
    EnvelopePlasmaAssembly,
    EnvelopePlasmaChannel,
    build_cycle_integrals,
    source_factors,
)
from solprop.translate.expansion import (
    ParameterExpansion,
    build_expansion,
    reweight_expansion,
)
from solprop.translate.ionisation import RateTable, build_rate_table
from solprop.translate.linop import (
    DEFAULT_MARGIN,
    PhiRepresentation,
    RepresentationChoice,
    declared_paths,
    parameter_paths,
    parameters_own_z,
    profile_table,
)
from solprop.translate.masks import (
    DEFAULT_ABSORBER_SCHEME,
    AbsorberScheme,
    temporal_rates,
)
from solprop.translate.plasma_integration import design_running_integral
from solprop.translate.raman import build_raman_kernels, decay_rate_profiles
from solprop.translate.spatial import SeparableOperator, separable_values

__all__ = [
    "DELAYED_MODEL_ID",
    "KERR_FORM_MODEL_IDS",
    "KERR_MODEL_FORMS",
    "MODEL_ID_SEPARATOR",
    "THG_OVERSAMPLING",
    "CompiledResponses",
    "DispersiveKerrTerms",
    "KerrCoefficients",
    "KerrForm",
    "PlasmaChannel",
    "PlasmaCoefficients",
    "RamanCoefficients",
    "RegionalResponseTables",
    "ResponseStage",
    "compile_responses",
    "dispersive_regions",
    "dispersive_terms",
    "instantaneous_model_id",
    "isotropic_kerr_tensor",
    "kerr_coefficients",
    "plasma_coefficients",
    "prefactor_at",
    "raman_coefficients",
    "regional_responses",
    "response_kinks",
    "unlanded_kinks",
]

logger = logging.getLogger(__name__)

THG_OVERSAMPLING = 6.0
"""Least oversampling a third-harmonic term needs, so that ``3 omega_max`` is resolved.

``conventions.md`` section 3: a cubic response reaching ``3 omega_max`` needs ``ffac =
6``, and ``4`` suffices without the harmonic. A grid built with ``thg=False`` has
``ffac = 2`` on an envelope grid, so a harmonic term on it would alias silently.
"""

type KerrForm = Literal["analytic", "real"]
"""Which cubic a general modal guide's instantaneous response is.

``"analytic"`` is ``P = eps0 chi3 (2 |E|^2 E + (E.E) E*) / 4`` on the analytic signal,
which carries the fundamental band alone; ``"real"`` is ``P = eps0 chi3 (E.E) E`` on the
real three-component field, which is the field route's own model and carries the third
harmonic with it. The two agree on the fundamental.
"""

KERR_FORM_MODEL_IDS: Final[Mapping[str, str]] = {
    "analytic": "isotropic-kerr",
    "real": "isotropic-kerr-real",
}
"""The model identity each instantaneous form is recorded under.

The form is a **physical model** choice and so lives in ``model_id``, which is already
a representation tag inside the resume identity. It is not a new
leaf field: adding one would move a contract row and bump the kernel schema, and there
is nothing else the form needs to carry.
"""

KERR_MODEL_FORMS: Final[Mapping[str, str]] = {
    value: key for key, value in KERR_FORM_MODEL_IDS.items()
}
"""The inverse of :data:`KERR_FORM_MODEL_IDS`, which a backend reads a form back from.

A model identity outside this mapping is refused by name rather than executed as one of
the two, because a caller who declares a model this package does not know must not have
it silently evaluated as another.
"""

_PROFILE_PROBES = 9


class ResponseStage(StrEnum):
    """One step of the fused time-domain traversal, in execution order.

    The translation layer compiles the declared set of responses into a single pass per
    column rather than into one pass per response. The names are the passes of
    ``docs/dev/architecture.md`` section 8; a backend specialises on the sequence at
    compile time.

    Attributes
    ----------
    INTENSITY : str
        Form ``|E|^2`` (or ``|A|^2``) once, for everything that needs it.
    IONISATION_RATE : str
        Look the ionisation rate up in its spline table.
    PLASMA_SCAN : str
        Integrate the electron density along t; sequential, three running sums.
    PLASMA_CURRENT : str
        The plasma current and loss terms.
    KERR : str
        The instantaneous third-order polarisation.
    DISPERSIVE_KERR : str
        SEP 6's frequency-dependent electronic polarisation: filtered fields, their
        products and the output multipliers on the fine grid.
    RAMAN_DRIVE : str
        Write the driving term of the Raman convolution.
    ACCUMULATE : str
        Sum the contributions into ``P(t)`` and apply the temporal window.
    ENVELOPE_CYCLE : str
        Form the averaged rate and both directional moments.
    ENVELOPE_POPULATION : str
        Scan cycle-averaged exposure and retain surviving neutrals.
    ENVELOPE_SOURCES : str
        Form separate Drude drive and loss-current streams before projection.

    Examples
    --------
    >>> from solprop.translate.responses import ResponseStage
    >>> ResponseStage.KERR, str(ResponseStage.ACCUMULATE)
    (<ResponseStage.KERR: 'kerr'>, 'accumulate')
    """

    INTENSITY = "intensity"
    IONISATION_RATE = "ionisation_rate"
    PLASMA_SCAN = "plasma_scan"
    PLASMA_CURRENT = "plasma_current"
    KERR = "kerr"
    DISPERSIVE_KERR = "dispersive_kerr"
    RAMAN_DRIVE = "raman_drive"
    ACCUMULATE = "accumulate"
    ENVELOPE_CYCLE = "envelope_cycle"
    ENVELOPE_POPULATION = "envelope_population"
    ENVELOPE_SOURCES = "envelope_sources"


@dataclass(frozen=True)
class KerrCoefficients:
    """The instantaneous third-order polarisation, reduced to numbers.

    The reference implementation is
    :meth:`solprop.nonlinear.responses.KerrResponse.evaluate`, and these coefficients
    are read off it rather than re-derived, so the compiled pass and the oracle cannot
    drift apart. Two cubic forms exist, because a field grid resolves the carrier and an
    envelope grid does not:

    ``"real"``
        ``P_p = coefficient rho (sum_q E_q^2) E_p`` — the full cubic of a real field,
        which already carries the third harmonic and the counter-rotating terms.
    ``"analytic"``
        ``P_p = coefficient rho [self (sum_q |a_q|^2) a_p + conjugate (sum_q a_q^2)
        conj(a_p)]``, plus ``coefficient rho harmonic exp(2 i omega_0 t) (sum_q a_q^2)
        a_p`` when the harmonic is wanted on an envelope grid. For one component the
        first collapses to ``(3/4)|a|^2 a``.

    Attributes
    ----------
    coefficient : float
        ``eps0 chi3`` per unit of the density path, in m^2/V^2 when the density is
        dimensionless and m^5/V^2 when it is a number density in m^-3. The density is
        the contract's ``density`` table and multiplies this at every position.
    cubic_form : {'real', 'analytic'}
        Which of the two forms above the pass evaluates.
    self_factor : float
        Weight of the ``(sum |a_q|^2) a_p`` term; ``0.5`` in the analytic form.
    conjugate_factor : float
        Weight of the ``(sum a_q^2) conj(a_p)`` term; ``0.25`` in the analytic form.
    harmonic_factor : float
        Weight of the third-harmonic term; ``0.25`` on an envelope grid that radiates
        it, zero otherwise.
    analytic_signal : bool
        Whether the pass must form the analytic signal of the real field first. True on
        a field grid in the analytic form, false on an envelope grid, where the state
        already is one.
    take_real : bool
        Whether the polarisation is the real part of the result. True on a field grid.
    density : PiecewiseChebyshev or None
        The scale multiplying coefficient at each z. A composite uses sum(rho*gamma3)
        with coefficient=eps0; a single response retains its original coefficient and
        density operation order. None is used only by the standalone leaf builder.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.responses import KerrResponse
    >>> from solprop.translate.responses import kerr_coefficients
    >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (700e-9, 2e-6), 2e-12)
    >>> coefficients = kerr_coefficients(KerrResponse(chi3=2e-28, thg=False), grid)
    >>> coefficients.cubic_form, coefficients.self_factor
    ('analytic', 0.5)
    """

    coefficient: float
    cubic_form: str
    self_factor: float
    conjugate_factor: float
    harmonic_factor: float
    analytic_signal: bool
    take_real: bool
    density: PiecewiseChebyshev | None = None


def kerr_coefficients(response: KerrResponse, grid: TimeGrid) -> KerrCoefficients:
    """Reduce a Kerr response on a grid to the numbers a fused pass needs.

    Parameters
    ----------
    response : solprop.nonlinear.responses.KerrResponse
        The response; its ``chi3`` is per unit of the density path.
    grid : solprop.grid.TimeGrid
        The grid, which decides the cubic form and whether a harmonic term is resolved.

    Returns
    -------
    KerrCoefficients
        The compiled coefficients.

    Raises
    ------
    TranslationError
        If a third-harmonic term is asked for on an envelope grid that does not
        oversample enough to resolve it, where it would alias into the fundamental
        without any sign that it had.

    Examples
    --------
    >>> from solprop.constants import EPS0
    >>> from solprop.grid import FieldGrid
    >>> from solprop.nonlinear.responses import KerrResponse
    >>> from solprop.translate.responses import kerr_coefficients
    >>> grid = FieldGrid.from_limits(1.0, 800e-9, (200e-9, 4e-6), 200e-15)
    >>> compiled = kerr_coefficients(KerrResponse(chi3=1.0), grid)
    >>> compiled.cubic_form, compiled.coefficient == EPS0, compiled.take_real
    ('real', True, True)
    """
    envelope = grid.kind == "envelope"
    if envelope and response.thg and grid.ffac < THG_OVERSAMPLING:
        msg = (
            f"a third-harmonic Kerr term needs an envelope grid oversampled by at "
            f"least {THG_OVERSAMPLING:g} so that 3 omega_max is resolved "
            f"(conventions.md section 3); this grid has ffac = {grid.ffac:g}. Build it "
            "with thg=True."
        )
        raise TranslationError(msg)
    real_cubic = grid.kind == "field" and response.thg
    return KerrCoefficients(
        coefficient=EPS0 * response.chi3,
        cubic_form="real" if real_cubic else "analytic",
        self_factor=0.0 if real_cubic else 0.5,
        conjugate_factor=0.0 if real_cubic else 0.25,
        harmonic_factor=0.25 if envelope and response.thg else 0.0,
        analytic_signal=grid.kind == "field" and not response.thg,
        take_real=grid.kind == "field",
    )


@dataclass(frozen=True, eq=False)
class DispersiveKerrTerms:
    """SEP 6's frequency-dependent electronic response, reduced to arrays.

    The reference is
    :meth:`solprop.nonlinear.responses.DispersiveKerrResponse.evaluate`, whose
    arithmetic :func:`solprop.nonlinear.responses.dispersive_cubic` is, and whose
    multipliers these arrays are, sampled on the problem's own fine spectral layout
    at absolute frequency. Each labelled species is its own channel with its own
    multipliers and density. How the channels combine is the evaluator's
    (:meth:`groups`):

    - the **polynomial** is the exact per-species sum,
      ``P = sum_s coefficient_s rho_s cubic_s(E)``, which a backend may regroup
      exactly (species of equal leg bound share their derivative fields);
    - the **symmetric** evaluator takes, for the channels that act at the same nodes
      -- every channel, or one general modal region's -- one filter from their
      mixture's Kerr diagonal, rebuilt from the channels' densities at every stage:
      ``P = F G_mix[w (G_mix E)^3]`` with ``F = sum_s coefficient_s rho_s`` and
      ``g_mix = [sum_s coefficient_s rho_s g_s^4 / F]^(1/4)``
      (:func:`solprop.nonlinear.responses.mixture_filter`; SEP 6 section 6, adopted
      for mixtures on 2026-10-08). One species is its own operator exactly.

    Attributes
    ----------
    evaluator : {'symmetric', 'polynomial'}
        Which evaluator every channel takes.
    sector : {'full', 'conserving'}
        Which quartets are kept: the full cubic (``thg=True``) or the
        number-conserving sector (``thg=False``).
    carrier : bool
        Whether the pass needs ``grids.carrier_2w0``: true exactly for the full
        sector on an envelope grid, whose harmonic groups sit ``+-2 omega0`` from the
        envelope's carrier.
    labels : tuple of str
        The channel's population label, one per channel, in species order.
    coefficients : tuple of float
        ``eps0 gamma3_0`` per channel, the static coefficient per unit density.
    positive : numpy.ndarray
        ``Pi_+`` on the fine spectral layout, bool, shape ``(n_omega_fine,)``.
    filters : numpy.ndarray
        Each channel's input multiplier, float64, shape
        ``(n_channels, n_omega_fine)``: the symmetric factor ``g``, or the
        polynomial's bounded leg weight in m^-2.
    outputs : numpy.ndarray or None
        The polynomial's two output multipliers per channel, float64, shape
        ``(n_channels, 2, n_omega_fine)``; ``None`` for the symmetric evaluator.
    products : numpy.ndarray or None
        The polynomial's two constant product weights per channel, float64, shape
        ``(n_channels, 2)``, in m^4; ``None`` for the symmetric evaluator.
    densities : tuple of PiecewiseChebyshev
        Each channel's number density along z, m^-3.
    regions : numpy.ndarray or None
        On the general modal route, the node region each channel acts in, int32,
        shape ``(n_channels,)``, non-negative: a region's gas fill contributes one
        channel per species, and a node responds to the channels of its own region.
        ``None`` on every other route, where a channel is a population of the run's
        material and acts at every node.

    Raises
    ------
    TranslationError
        If the vocabulary, the channel counts or the array shapes disagree.
    """

    evaluator: str
    sector: str
    carrier: bool
    labels: tuple[str, ...]
    coefficients: tuple[float, ...]
    positive: Bool[np.ndarray, " n_omega_fine"]
    filters: Float[np.ndarray, "n_channels n_omega_fine_f"]
    outputs: Float[np.ndarray, "n_channels_o 2 n_omega_fine_o"] | None
    products: Float[np.ndarray, "n_channels_p 2"] | None
    densities: tuple[PiecewiseChebyshev, ...]
    regions: Int32[np.ndarray, " n_channels_r"] | None = None

    def __post_init__(self) -> None:
        """Check the vocabulary and that every array agrees with the channels.

        Raises
        ------
        TranslationError
            Naming the field that disagrees.
        """
        if self.evaluator not in ("symmetric", "polynomial"):
            msg = f"dispersive_kerr.evaluator {self.evaluator!r} is not an evaluator"
            raise TranslationError(msg)
        if self.sector not in ("full", "conserving"):
            msg = f"dispersive_kerr.sector {self.sector!r} is not a sector"
            raise TranslationError(msg)
        n_channels = len(self.labels)
        if not n_channels or len(set(self.labels)) != n_channels:
            raise TranslationError("dispersive_kerr needs unique channel labels")
        if len(self.coefficients) != n_channels or len(self.densities) != n_channels:
            raise TranslationError(
                "dispersive_kerr needs one coefficient and one density per channel"
            )
        n_omega = self.positive.shape[0]
        if self.filters.shape != (n_channels, n_omega):
            msg = (
                f"dispersive_kerr.filters has shape {self.filters.shape}, expected "
                f"{(n_channels, n_omega)}"
            )
            raise TranslationError(msg)
        polynomial = self.evaluator == "polynomial"
        carried = (self.outputs is not None, self.products is not None)
        if carried != (polynomial, polynomial):
            raise TranslationError(
                "dispersive_kerr carries outputs and products exactly for the "
                "polynomial evaluator"
            )
        if self.outputs is not None and self.outputs.shape != (n_channels, 2, n_omega):
            raise TranslationError("dispersive_kerr.outputs has the wrong shape")
        if self.products is not None and self.products.shape != (n_channels, 2):
            raise TranslationError("dispersive_kerr.products has the wrong shape")
        if self.regions is not None and (
            self.regions.shape != (n_channels,)
            or self.regions.dtype != np.int32
            or np.any(self.regions < 0)
        ):
            raise TranslationError(
                "dispersive_kerr.regions holds one non-negative int32 node region per "
                "channel"
            )

    @property
    def n_channels(self) -> int:
        """Return the number of species channels.

        Returns
        -------
        int
            One per labelled population with a dispersive response.
        """
        return len(self.labels)

    def groups(self, keys: Sequence[int] | None = None) -> tuple[tuple[int, ...], ...]:
        """Return the channels that combine, each group in channel order.

        For the **symmetric** evaluator a group is the channels that act at the same
        nodes, whose mixture filter they share
        (:func:`solprop.nonlinear.responses.mixture_filter`). For the **polynomial** a
        group is the channels whose leg-weight rows are equal bit for bit and that act
        at the same nodes: they share their derivative fields, and summing their
        coefficient tables is an exact rewriting of the per-species sum.

        Parameters
        ----------
        keys : sequence of int or None, optional
            A key per channel that must also agree within a group; ``None`` takes the
            leaf's node regions, or a single key where it has none.

        Returns
        -------
        tuple of tuple of int
            The groups, in the order of their first channel.
        """
        chosen: Sequence[int] = (
            keys
            if keys is not None
            else (0,) * self.n_channels
            if self.regions is None
            else self.regions.tolist()
        )
        found: dict[tuple[bytes, int], list[int]] = {}
        for channel in range(self.n_channels):
            row = (
                b""
                if self.evaluator == "symmetric"
                else np.ascontiguousarray(self.filters[channel]).tobytes()
            )
            found.setdefault((row, int(chosen[channel])), []).append(channel)
        return tuple(tuple(members) for members in found.values())

    def evaluations(
        self, densities: Sequence[float]
    ) -> tuple[tuple[int, float, DispersiveWeights], ...]:
        """Return the cubics the reference evaluates at a stage, and their factors.

        The polarisation is ``sum factor cubic(E; weights)`` over the returned terms,
        each acting at the nodes of its channel's region: one per channel for the
        polynomial, the exact per-species sum; one per group for the symmetric
        evaluator, its mixture filter built from the group's densities here
        (:func:`solprop.nonlinear.responses.mixture_filter`), one species' its own.

        Parameters
        ----------
        densities : sequence of float
            Each channel's number density at the stage, m^-3, in channel order.

        Returns
        -------
        tuple of tuple
            ``(channel, factor, weights)``: the first channel of the term, whose
            region it acts in; ``eps0 gamma3_0 rho`` summed over its channels, C/V^3;
            and its multipliers.
        """
        factors = [
            coefficient * density
            for coefficient, density in zip(self.coefficients, densities, strict=True)
        ]
        if self.evaluator == "polynomial":
            return tuple(
                (channel, factors[channel], self.weights(channel))
                for channel in range(self.n_channels)
            )
        found = []
        for group in self.groups():
            total, curve = mixture_filter(
                self.filters[list(group)], np.array([factors[one] for one in group])
            )
            found.append(
                (group[0], total, DispersiveWeights("symmetric", self.positive, curve))
            )
        return tuple(found)

    def weights(self, channel: int) -> DispersiveWeights:
        """Return one channel's multipliers in the reference's own record.

        Parameters
        ----------
        channel : int
            The channel's index.

        Returns
        -------
        solprop.nonlinear.responses.DispersiveWeights
            Views of this leaf's rows.
        """
        if self.outputs is None or self.products is None:
            return DispersiveWeights("symmetric", self.positive, self.filters[channel])
        first, second = (float(value) for value in self.products[channel])
        return DispersiveWeights(
            "polynomial",
            self.positive,
            self.filters[channel],
            self.outputs[channel],
            (first, second),
        )


def dispersive_terms(
    responses: Sequence[DispersiveKerrResponse],
    labels: Sequence[str],
    densities: Sequence[PiecewiseChebyshev],
    grid: TimeGrid,
    *,
    regions: Sequence[int] | np.ndarray | None = None,
) -> DispersiveKerrTerms:
    """Sample the dispersive responses' multipliers on a grid's fine layout.

    Parameters
    ----------
    responses : sequence of DispersiveKerrResponse
        The declared responses, one per channel, agreeing on evaluator and ``thg``.
    labels : sequence of str
        Each response's population label.
    densities : sequence of PiecewiseChebyshev
        Each population's number density table, m^-3.
    grid : solprop.grid.TimeGrid
        The grid, whose ``omega_fine`` gives the absolute frequency of every bin.
    regions : sequence of int or numpy.ndarray or None, optional
        On the general modal route, the node region of each channel; ``None``
        elsewhere.

    Returns
    -------
    DispersiveKerrTerms
        The compiled leaf.

    Raises
    ------
    TranslationError
        If the responses disagree on evaluator or ``thg``, or the full cubic is
        asked for on an envelope grid that does not oversample enough to hold it.
    """
    first = responses[0]
    if len({one.evaluator for one in responses}) != 1:
        raise TranslationError(
            "dispersive Kerr responses must agree on their evaluator; one run takes "
            "one electronic law"
        )
    envelope = grid.kind == "envelope"
    if envelope and first.thg and grid.ffac < THG_OVERSAMPLING:
        msg = (
            "a dispersive Kerr response's full cubic needs an envelope grid "
            f"oversampled by at least {THG_OVERSAMPLING:g}, so that its harmonic "
            f"groups are resolved (conventions.md section 3); this grid has ffac = "
            f"{grid.ffac:g}. Build it with thg=True, or take the number-conserving "
            "sector with thg=False."
        )
        raise TranslationError(msg)
    sampled = [one.weights(grid.omega_fine, kind=grid.kind) for one in responses]
    polynomial = first.evaluator == "polynomial"
    return DispersiveKerrTerms(
        evaluator=first.evaluator,
        sector=first.sector,
        carrier=envelope and first.thg,
        labels=tuple(labels),
        coefficients=tuple(float(one.coefficient) for one in responses),
        positive=_readonly(sampled[0].positive),
        filters=_readonly(np.stack([one.filter for one in sampled])),
        outputs=_readonly(np.stack([cast("np.ndarray", one.output) for one in sampled]))
        if polynomial
        else None,
        products=_readonly(np.array([one.products for one in sampled], dtype=float))
        if polynomial
        else None,
        densities=tuple(densities),
        regions=None
        if regions is None
        else _readonly(np.asarray(regions, dtype=np.int32)),
    )


@dataclass(frozen=True, eq=False)
class PlasmaCoefficients:
    """The plasma response, reduced to two constants, a path and a table.

    Section 5 of the kernel contract. The reference implementation is
    :meth:`solprop.nonlinear.responses.PlasmaResponse.evaluate`, and these are the only
    numbers it needs beyond the field and the density: the free-electron current is
    ``(e^2/m_e) int f E dt`` and the ionisation loss is ``I_p W (1 - f) E / |E|^2``,
    with ``f = 1 - (1 - f_0) exp(-int W dt)`` -- the pre-ionised fraction enters
    **inside** the exponential, so the ionised fraction never exceeds one.

    ``preionisation`` is a path along z because the response declares it as one, even
    where the value the user gave is a plain number: a backend reads one value per
    plane with a Clenshaw pass and never sees a
    :class:`solprop.profiles.ZProfile`. Equality is by identity (``eq=False``), as it
    is for every group of the contract that holds arrays.

    Attributes
    ----------
    charge_mass_ratio : float
        ``e^2 / m_e`` in C^2/kg, positive: the constant of the free-electron current.
        Carried rather than assumed so that a backend contains no physical constant.
    ionisation_potential : float
        Ionisation potential of the species in J, positive. It is what fixes the size
        of the loss current, whose work against the field is exactly one potential per
        electron freed.
    preionisation : solprop.translate.chebyshev.PiecewiseChebyshev
        The fraction already ionised before the pulse, as a scalar table along z, with
        values in ``[0, 1]``.
    rate : solprop.translate.ionisation.RateTable
        The ionisation rate in s^-1 against field amplitude in V/m, the same object
        :attr:`CompiledResponses.ionisation_table` carries. It is exactly zero below
        its ``field_min``, which is what makes the loss current's division by the field
        finite.

    Raises
    ------
    TranslationError
        If either constant is not positive and finite.

    Examples
    --------
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> from solprop.nonlinear.responses import PlasmaResponse
    >>> from solprop.translate.responses import plasma_coefficients
    >>> response = PlasmaResponse("Ar", OptmatRate("Ar", 800e-9), preionisation=0.25)
    >>> compiled = plasma_coefficients(response, (1e9, 1e11), z_max=1.0)
    >>> float(compiled.preionisation(0.5)), compiled.rate.field_min > 0.0
    (0.25, True)
    """

    charge_mass_ratio: float
    ionisation_potential: float
    preionisation: PiecewiseChebyshev
    rate: RateTable

    def __post_init__(self) -> None:
        """Reject a constant that cannot give a current.

        Raises
        ------
        TranslationError
            Naming the offending field.
        """
        for name in ("charge_mass_ratio", "ionisation_potential"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                msg = f"{name} must be positive and finite, got {value!r}"
                raise TranslationError(msg)


def plasma_coefficients(
    response: PlasmaResponse,
    field_range: tuple[float, float] | None,
    *,
    z_max: float,
    margin: float = DEFAULT_MARGIN,
    tol: float = DEFAULT_TOL,
) -> PlasmaCoefficients:
    """Reduce a plasma response to the coefficients a fused pass needs.

    The response's own rate is what the run uses. A
    :class:`solprop.translate.ionisation.RateTable` handed to the response is taken as
    it stands, because it is already the compiled form; anything else is tabulated over
    `field_range`, which the caller must state -- a table cannot choose its own span.

    Parameters
    ----------
    response : solprop.nonlinear.responses.PlasmaResponse
        The declared response.
    field_range : tuple of float or None
        Lower and upper field amplitude in V/m the rate table must cover. Ignored, and
        refused, when the response already carries a table.
    z_max : float
        Length the run covers, m. The pre-ionisation table spans ``[0, z_max (1 +
        margin)]``, the same range as ``Phi``.
    margin : float, optional
        Fraction of `z_max` the table extends beyond the end.
    tol : float, optional
        Target bound on the uniform error of the pre-ionisation table.

    Returns
    -------
    PlasmaCoefficients
        The compiled coefficients.

    Raises
    ------
    TranslationError
        If a rate that is not already a table is given without a field range, or a
        field range is given beside a table that does not need one.

    Examples
    --------
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> from solprop.nonlinear.responses import PlasmaResponse
    >>> from solprop.translate.responses import plasma_coefficients
    >>> compiled = plasma_coefficients(
    ...     PlasmaResponse("Ar", OptmatRate("Ar", 800e-9)), (1e9, 1e11), z_max=1.0
    ... )
    >>> f"{compiled.ionisation_potential:.4e}"
    '2.5250e-18'
    """
    rate = response.rate
    if isinstance(rate, RateTable):
        if field_range is not None:
            msg = (
                "ionisation_field_range was given beside a plasma response that "
                "already carries a RateTable; the table states its own span"
            )
            raise TranslationError(msg)
        # A table the caller built is the builder's memoised object, whose arrays are
        # writeable; everything crossing the seam is frozen, so freeze a copy.
        table = _frozen_table(rate)
    elif not isinstance(rate, IonisationRate):
        # A bare callable satisfies the response's own EvaluableRate, but a table needs
        # a threshold to clip against and named parameters to key its cache on.
        msg = (
            f"the plasma response carries a {type(rate).__name__} rate, which declares "
            "neither a threshold nor its parameters; tabulate it yourself with "
            "solprop.translate.ionisation.build_rate_table and give the response the "
            "RateTable, or use an optmat rate "
            "(solprop.nonlinear.optmat_rates.OptmatRate)"
        )
        raise TranslationError(msg)
    elif field_range is None:
        msg = (
            "a plasma response whose rate is not already a table needs "
            "ionisation_field_range, the field amplitudes in V/m the run will reach; a "
            "table cannot choose its own span"
        )
        raise TranslationError(msg)
    else:
        table = _tabulate_rate(rate, field_range)
    span = z_max * (1.0 + margin)
    return PlasmaCoefficients(
        charge_mass_ratio=CHARGE_MASS_RATIO,
        ionisation_potential=response.potential,
        preionisation=profile_table(
            _preionisation_profile(response.preionisation), 0.0, span, tol=tol
        ),
        rate=table,
    )


def _preionisation_profile(value: float | ZProfile) -> ZProfile:
    """Return the path a pre-ionisation is tabulated from.

    A number, and a path whose ``is_constant`` is true, both become the same
    :class:`Constant`, so a constant path compiles to the number's table to the last
    bit whatever breakpoints it carries.

    Parameters
    ----------
    value : float or ZProfile
        The initial ionised fraction, dimensionless, or a path of it along z in m.

    Returns
    -------
    ZProfile
        The path to tabulate.
    """
    if isinstance(value, ZProfile) and value.is_constant:
        start = value.z_span[0]
        value = float(np.asarray(value(np.array([start])))[0])
    return as_profile(value, (0.0, 1.0))


def response_kinks(
    responses: Sequence[object], density: ZProfile | float | None = None
) -> tuple[tuple[float, str], ...]:
    """Return where the response paths kink, each with the name of its path.

    A step lands only on the hard kinks of ``Phi``, which the mode set declares. A
    response can carry a path the modes know nothing about: a plasma response's
    pre-ionisation along z, or a density path given to
    :func:`solprop.translate.problem.assemble` explicitly. Its breakpoints are where
    the right-hand side is not smooth, so they must become hard kinks as well, or a
    step straddles them and the integrator loses its order there.

    Parameters
    ----------
    responses : sequence of object
        The declared responses; the plasma responses' pre-ionisation paths are read.
    density : ZProfile, float or None, optional
        An explicit density path in m^-3, when one replaces the modes' own.

    Returns
    -------
    tuple of tuple
        ``(z, name)`` pairs, z in m, in declaration order; a number, or a path whose
        ``is_constant`` is true, declares none.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> from solprop.nonlinear.responses import PlasmaResponse
    >>> from solprop.profiles import PiecewiseLinear
    >>> from solprop.translate.responses import response_kinks
    >>> path = PiecewiseLinear(np.array([0.0, 0.4, 1.0]), np.array([0.0, 0.1, 0.1]))
    >>> rate = OptmatRate("Ar", 800e-9)
    >>> response_kinks([PlasmaResponse("Ar", rate, preionisation=path)])
    ((0.0, 'preionisation'), (0.4, 'preionisation'), (1.0, 'preionisation'))
    """
    found: list[tuple[float, str]] = []
    for response in responses:
        if not isinstance(response, PlasmaResponse | EnvelopePlasmaResponse):
            continue
        path = response.preionisation
        # A path that does not vary has no kink worth landing on, whatever breakpoints
        # it happens to carry; it compiles to the number's own table.
        if isinstance(path, ZProfile) and not path.is_constant:
            name = (
                "preionisation"
                if response.species is None
                else f"preionisation:{response.species}"
            )
            found.extend((float(z), name) for z in np.ravel(path.breakpoints))
    if isinstance(density, ZProfile) and not density.is_constant:
        found.extend((float(z), "density") for z in np.ravel(density.breakpoints))
    return tuple(found)


def unlanded_kinks(
    kinks: Sequence[tuple[float, str]],
    hard_kinks: Float[np.ndarray, " n_kinks"],
    z_max: float,
    *,
    slack: float,
) -> tuple[tuple[float, str], ...]:
    """Return the response kinks inside the run that no step is made to land on.

    Parameters
    ----------
    kinks : sequence of tuple
        ``(z, name)`` pairs from :func:`response_kinks`, z in m.
    hard_kinks : numpy.ndarray
        The linear operator's hard kinks in m, shape ``(n_kinks,)``.
    z_max : float
        End of the run in m; a kink at or beyond it, or at or before the entrance,
        needs no landing.
    slack : float
        Distance in m within which a hard kink counts as the same position.

    Returns
    -------
    tuple of tuple
        The ``(z, name)`` pairs strictly inside ``(0, z_max)`` with no hard kink
        within `slack`; empty when every one is landed on.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.responses import unlanded_kinks
    >>> kinks = ((0.4, "preionisation"), (0.7, "preionisation"))
    >>> unlanded_kinks(kinks, np.array([0.4]), 1.0, slack=1e-12)
    ((0.7, 'preionisation'),)
    """
    return tuple(
        (z, name)
        for z, name in kinks
        if 0.0 < z < z_max
        and (hard_kinks.size == 0 or float(np.min(np.abs(hard_kinks - z))) > slack)
    )


@dataclass(frozen=True, eq=False)
class PlasmaChannel(PlasmaCoefficients):
    """Bind an independent exposure scan to a labelled population.

    Attributes
    ----------
    charge_mass_ratio : float
        Free-current coefficient in C²/kg, positive.
    ionisation_potential : float
        Energy paid per event in J, positive.
    preionisation : solprop.translate.chebyshev.PiecewiseChebyshev
        Initial ionised fraction along z, dimensionless in [0, 1].
    rate : solprop.translate.ionisation.RateTable
        Rate in s^-1 against field amplitude in V/m.
    label : str
        Unique component label in the compiled species order.
    density : solprop.translate.chebyshev.PiecewiseChebyshev
        Population number density in m^-3 along z.
    electrons_per_event : float
        Electrons released per ionisation event; exactly 1 in schema 6.
    """

    label: str
    density: PiecewiseChebyshev
    electrons_per_event: float = 1.0

    def __post_init__(self) -> None:
        """Validate the channel's constants and supported event type."""
        super().__post_init__()
        if not self.label:
            raise TranslationError("plasma channel label must be nonempty")
        if self.electrons_per_event != 1.0:
            raise TranslationError(
                f"plasma channel {self.label}: electrons_per_event must be 1.0"
            )


@dataclass(frozen=True, eq=False)
class RamanCoefficients:
    """The delayed response of a Raman-active medium, as a fused pass needs it.

    A leaf of section 5 of the kernel contract, like :class:`PlasmaCoefficients`, so
    its fields are not rows of that document. It replaces the three separate rows of
    contract version 2 and carries the drive form, which they had nowhere to put:
    which real quadratic form drives the coherence on a *field* grid -- ``E^2``, or the
    cycle-averaged ``|E_a|^2/2`` -- is a property of the problem, and the translation
    layer checks that the declared Kerr and Raman responses agree about it.

    Attributes
    ----------
    undamped : numpy.ndarray
        Each independently damped term's undamped response, float64, shape
        ``(n_terms, n_to_doubled)``: causal from ``t = 0``, tapered by the grid's
        temporal collar and zero-padded to the doubled grid
        (:func:`solprop.translate.raman.undamped_kernels`). ``n_terms`` is 1 for a glass
        or a vibration-only gas and 2 for a gas with a rotation and a vibration. The
        rows are ``h`` itself and carry no ``eps0``; the response writes it outside the
        convolution integral (``docs/guide/media/raman-numerics.md``).
    decay_rates : tuple of solprop.translate.chebyshev.PiecewiseChebyshev
        The decay rate ``1 / tau_2`` of each term along z as a scalar table, one per row
        of `undamped`, composed from the term's dephasing and the density path
        (:func:`solprop.translate.raman.decay_rate_profiles`). Finite everywhere.
    fraction : float
        ``f_R`` multiplying the summed kernel, in ``(0, 1]``; 1 for an absolute
        molecular response.
    thg : bool
        Whether a field grid's drive keeps the ``2 omega`` beat and so radiates a third
        harmonic through the later multiplication by the field. Ignored on an envelope
        grid, whose drive is ``|A|^2/2`` whatever this says, because a quadratic drive
        cannot carry the beat there.
    tensor_weights : numpy.ndarray
        Read-only float64 weights, shape ``(n_terms, 2)``, multiplying the trace
        and traceless effective kernels. Isotropic rows are ``(1, 0)`` and
        rank-two rows ``(1/4, 3/4)``. Scalar-only rows use ``(1, 0)`` but do not
        establish vector capability. Every row keeps its own decay rate.
    tensor_capable : bool
        Whether every declared term has an established vector contraction.
    label : str
        Stable component label; the standalone leaf builder uses "medium".
    density : PiecewiseChebyshev or None
        Component amplitude density in m^-3, distinct from the total density used
        for linewidths. None is used only by the standalone leaf builder.

    Raises
    ------
    TranslationError
        If the kernel is not two-dimensional, the number of decay-rate tables does not
        match its rows, or the fraction is outside ``(0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.profiles import Constant
    >>> from solprop.translate.linop import profile_table
    >>> from solprop.translate.responses import RamanCoefficients
    >>> rates = (profile_table(Constant(1e12), 0.0, 1.0),)
    >>> leaf = RamanCoefficients(
    ...     np.zeros((1, 8)), rates, 0.18, thg=False,
    ...     tensor_weights=np.array([[1.0, 0.0]]), tensor_capable=False,
    ... )
    >>> leaf.n_terms, leaf.thg
    (1, False)
    """

    undamped: Float[np.ndarray, "n_terms n_to_doubled"]
    decay_rates: tuple[PiecewiseChebyshev, ...]
    fraction: float
    thg: bool
    tensor_weights: Float[np.ndarray, "n_tensor_terms 2"]
    tensor_capable: bool
    label: str = "medium"
    density: PiecewiseChebyshev | None = None

    def __post_init__(self) -> None:
        """Freeze the kernel and reject a set of rows that cannot be evaluated.

        Raises
        ------
        TranslationError
            Naming the offending field.
        """
        rows = _readonly(np.asarray(self.undamped, dtype=np.float64))
        object.__setattr__(self, "undamped", rows)
        object.__setattr__(self, "decay_rates", tuple(self.decay_rates))
        if rows.ndim != 2 or rows.shape[0] < 1:
            msg = (
                "undamped must be (n_terms, n_to_doubled) with at least one term, got "
                f"shape {rows.shape}"
            )
            raise TranslationError(msg)
        if len(self.decay_rates) != rows.shape[0]:
            msg = (
                f"{len(self.decay_rates)} decay-rate table(s) for {rows.shape[0]} "
                "kernel row(s); the kernel damps row k by exp(-gamma_k(z) t) and needs "
                "one rate per row"
            )
            raise TranslationError(msg)
        if not np.all(np.isfinite(rows)):
            raise TranslationError("undamped Raman kernels must be finite")
        weights = _readonly(np.asarray(self.tensor_weights, dtype=np.float64))
        if weights.shape != (rows.shape[0], 2):
            raise TranslationError("tensor_weights must have shape (n_terms, 2)")
        if not np.all(
            np.all(weights == (1.0, 0.0), axis=1)
            | np.all(weights == (0.25, 0.75), axis=1)
        ):
            raise TranslationError("tensor_weights must be isotropic or rank-two rows")
        object.__setattr__(self, "tensor_weights", weights)
        fraction = float(self.fraction)
        if not (math.isfinite(fraction) and 0.0 < fraction <= 1.0):
            msg = f"fraction must lie in (0, 1], got {self.fraction!r}"
            raise TranslationError(msg)

    @property
    def n_terms(self) -> int:
        """Return the number of independently damped terms.

        Returns
        -------
        int
            The number of kernel rows, which is also the number of decay-rate tables.
        """
        return int(self.undamped.shape[0])


@dataclass(frozen=True, eq=False)
class RegionalResponseTables(Displayed):
    """Say which material responds at each node, and with what tensor and kernel.

    The optional leaf of contract section 5 that a general modal guide carries: the
    numeric arrays of a general response oracle, assigned to the node regions of
    :class:`solprop.translate.modal_tables.ModalTransformTables`.
    The existing gas and glass response owners remain the source of every law; this
    record is the compiled form of those laws and not a second physics.

    **The delayed term is the silica tensor Raman model**: ``n_terms`` is zero for a
    region that declares the instantaneous cubic alone
    and two -- the trace and traceless rank-two channels -- for one that declares a
    :class:`solprop.nonlinear.raman.SilicaTensorResponse` beside it. Region indices
    are node region indices and are never species or plasma channel indices.

    Attributes
    ----------
    response_region : numpy.ndarray
        Non-negative int32 region index per response, shape ``(n_responses,)``. Each
        entry selects the material region of ``ModalTransformTables.region_index``
        the response acts in.
    chi3 : numpy.ndarray
        Third-order susceptibility in m^2/V^2, shape
        ``(n_intervals, n_deg, n_responses)``, as Chebyshev coefficients along z so
        that a declared density path enters exactly once.
    instantaneous_tensor : numpy.ndarray
        Dimensionless rank-four instantaneous tensor, shape
        ``(n_responses, n_components, n_components, n_components, n_components)``
        with one, two or three Cartesian components.
    delayed_tensor : numpy.ndarray
        The same for each delayed term, shape ``(n_responses, n_terms,
        n_components, n_components, n_components, n_components)``, contracted as
        ``P_i = eps0 chi3 D_ijkl E_j H_kl`` with ``H`` the convolved pair history,
        so it is symmetric in its **last two** indices; the term axis is empty for a
        purely instantaneous response.
    delayed_kernel : numpy.ndarray
        Causal response kernels in 1/s on the doubled fine time axis, shape
        ``(n_intervals, n_deg, n_responses, n_terms, 2 * n_to)``: the causal half
        carries the kernel and the second half is the zero padding that makes a
        transform product the linear convolution. The Raman fraction is already in
        them, so a record multiplies by ``chi3`` alone.
    kernel_valid_samples : numpy.ndarray
        Int32 count of kernel samples inside each term's support, shape
        ``(n_responses, n_terms)``: the length of the causal half, beyond which the
        stored kernel is the zero padding and not the response.
    temporal_weight : numpy.ndarray
        Non-negative quadrature weight in s per fine time sample, shape ``(n_to,)``.
    model_id, policy_id : str
        Non-empty physical-model and numeric-policy identities. Scalar identities,
        never arrays, enter resume identity.

    Raises
    ------
    TranslationError
        Naming the field that failed.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.responses import RegionalResponseTables
    >>> tables = RegionalResponseTables(
    ...     np.zeros(1, dtype=np.int32), np.full((1, 1, 1), 2.2e-52),
    ...     np.ones((1, 3, 3, 3, 3)), np.zeros((1, 0, 3, 3, 3, 3)),
    ...     np.zeros((1, 1, 1, 0, 16)), np.zeros((1, 0), dtype=np.int32),
    ...     np.full(8, 1e-15), "isotropic-kerr", "dense-nodes",
    ... )
    >>> tables.n_responses, tables.n_components, tables.n_terms
    (1, 3, 0)
    """

    response_region: Int32[np.ndarray, " n_responses_region"]
    chi3: Float[np.ndarray, "n_intervals_chi3 n_deg_chi3 n_responses_chi3"]
    instantaneous_tensor: Float[
        np.ndarray,
        "n_responses_instant n_components_instant_i n_components_instant_j"
        " n_components_instant_k n_components_instant_l",
    ]
    delayed_tensor: Float[
        np.ndarray,
        "n_responses_delayed n_terms_delayed n_components_delayed_i"
        " n_components_delayed_j n_components_delayed_k n_components_delayed_l",
    ]
    delayed_kernel: Float[
        np.ndarray,
        "n_intervals_kernel n_deg_kernel n_responses_kernel n_terms_kernel"
        " n_to_doubled_kernel",
    ]
    kernel_valid_samples: Int32[np.ndarray, "n_responses_support n_terms_support"]
    temporal_weight: Float[np.ndarray, " n_to_temporal_weight"]
    model_id: str
    policy_id: str

    @property
    def n_responses(self) -> int:
        """Return how many responses the record assigns.

        Returns
        -------
        int
            The length of ``response_region``.
        """
        return int(self.response_region.size)

    @property
    def n_components(self) -> int:
        """Return how many Cartesian components each tensor contracts over.

        Returns
        -------
        int
            At least one and at most
            :data:`solprop.translate.problem.MAX_N_COMPONENTS`.
        """
        return int(self.instantaneous_tensor.shape[1])

    @property
    def n_terms(self) -> int:
        """Return how many delayed terms each response carries.

        Returns
        -------
        int
            Zero for a purely instantaneous response, two for the silica tensor
            Raman model's trace and rank-two channels.
        """
        return int(self.delayed_tensor.shape[1])

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``RegionalResponseTables(1 response, 3 components, 0
            delayed terms)``.
        """
        plural = "" if self.n_responses == 1 else "s"
        return line(
            "RegionalResponseTables",
            [
                ("", f"{self.n_responses} response{plural}"),
                ("", f"{self.n_components} components"),
                ("", f"{self.n_terms} delayed terms"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        title = "RegionalResponseTables: the compiled response of each material region"
        return title, [
            ("responses", self.n_responses),
            ("regions", sorted({int(one) for one in self.response_region})),
            ("components", self.n_components),
            ("delayed terms", self.n_terms or "none (Kerr only)"),
            ("intervals", int(self.chi3.shape[0])),
            ("fine samples", int(self.temporal_weight.size)),
            ("model", self.model_id),
            ("policy", self.policy_id),
        ]

    def __post_init__(self) -> None:
        """Copy, validate and freeze every leaf array.

        Raises
        ------
        TranslationError
            If an extent is wrong or a value is out of range.
        """
        if not (self.model_id and self.policy_id):
            msg = "regional_responses: model_id and policy_id must be non-empty"
            raise TranslationError(msg)
        regions = np.array(self.response_region, dtype=np.int32, order="C", copy=True)
        if regions.ndim != 1 or not regions.size or np.any(regions < 0):
            msg = (
                "regional_responses.response_region must be a non-empty vector of "
                f"non-negative region indices, got {regions}"
            )
            raise TranslationError(msg)
        regions.setflags(write=False)
        object.__setattr__(self, "response_region", regions)
        self._validate_tensors()
        self._validate_tables()

    def _validate_tensors(self) -> None:
        """Check the two tensors against the response and component counts.

        Raises
        ------
        TranslationError
            Naming the field that failed.
        """
        # The contract's component ceiling lives beside the state's polarisation
        # ceiling, in translate.problem, which imports this module; the import is
        # deferred rather than the number copied, so there is one of it.
        from solprop.translate.problem import (  # noqa: PLC0415 - import cycle
            MAX_N_COMPONENTS,
        )

        responses = self.n_responses
        instant = np.array(
            self.instantaneous_tensor, dtype=np.float64, order="C", copy=True
        )
        admitted = tuple(range(1, MAX_N_COMPONENTS + 1))
        if (
            instant.ndim != 5
            or instant.shape[0] != responses
            or len(set(instant.shape[1:])) != 1
            or instant.shape[1] not in admitted
        ):
            msg = (
                "regional_responses.instantaneous_tensor must have shape "
                f"(n_responses, c, c, c, c) with {responses} response(s) and c in "
                f"{admitted}, got {instant.shape}"
            )
            raise TranslationError(msg)
        if not np.all(np.isfinite(instant)):
            msg = "regional_responses.instantaneous_tensor must be finite"
            raise TranslationError(msg)
        instant.setflags(write=False)
        object.__setattr__(self, "instantaneous_tensor", instant)
        components = int(instant.shape[1])
        delayed = np.array(self.delayed_tensor, dtype=np.float64, order="C", copy=True)
        if (
            delayed.ndim != 6
            or delayed.shape[0] != responses
            or delayed.shape[2:] != (components,) * 4
        ):
            msg = (
                "regional_responses.delayed_tensor must have shape (n_responses, "
                f"n_terms, c, c, c, c) with {responses} response(s) and c = "
                f"{components}, got {delayed.shape}"
            )
            raise TranslationError(msg)
        delayed.setflags(write=False)
        object.__setattr__(self, "delayed_tensor", delayed)

    def _validate_tables(self) -> None:
        """Check the susceptibility, the kernels and the temporal weight.

        Raises
        ------
        TranslationError
            Naming the field that failed.
        """
        responses = self.n_responses
        terms = int(self.delayed_tensor.shape[1])
        chi3 = np.array(self.chi3, dtype=np.float64, order="C", copy=True)
        if chi3.ndim != 3 or not chi3.shape[0] or chi3.shape[2] != responses:
            msg = (
                "regional_responses.chi3 must have shape (n_intervals, n_deg, "
                f"n_responses) with at least one interval and {responses} "
                f"response(s), got {chi3.shape}"
            )
            raise TranslationError(msg)
        if not chi3.shape[1]:
            msg = (
                "regional_responses.chi3 has an empty degree axis; a coefficient "
                "block along z carries at least one degree, and a constant "
                "susceptibility is degree zero with one coefficient"
            )
            raise TranslationError(msg)
        if not np.all(np.isfinite(chi3)):
            msg = "regional_responses.chi3 must be finite"
            raise TranslationError(msg)
        chi3.setflags(write=False)
        object.__setattr__(self, "chi3", chi3)
        weight = np.array(self.temporal_weight, dtype=np.float64, order="C", copy=True)
        if weight.ndim != 1 or not weight.size or np.any(weight < 0):
            msg = (
                "regional_responses.temporal_weight must be a non-empty vector of "
                "non-negative seconds"
            )
            raise TranslationError(msg)
        weight.setflags(write=False)
        object.__setattr__(self, "temporal_weight", weight)
        kernel = np.array(self.delayed_kernel, dtype=np.float64, order="C", copy=True)
        expected = (chi3.shape[0], chi3.shape[1], responses, terms, 2 * weight.size)
        if kernel.shape != expected:
            msg = (
                f"regional_responses.delayed_kernel must have shape {expected}, got "
                f"{kernel.shape}"
            )
            raise TranslationError(msg)
        kernel.setflags(write=False)
        object.__setattr__(self, "delayed_kernel", kernel)
        support = np.array(
            self.kernel_valid_samples, dtype=np.int32, order="C", copy=True
        )
        if support.shape != (responses, terms) or np.any(support < 0):
            msg = (
                "regional_responses.kernel_valid_samples must have shape "
                f"{(responses, terms)} with non-negative counts, got {support.shape}"
            )
            raise TranslationError(msg)
        support.setflags(write=False)
        object.__setattr__(self, "kernel_valid_samples", support)


def isotropic_kerr_tensor(form: KerrForm = "analytic") -> Float[np.ndarray, "3 3 3 3"]:
    r"""Return the dimensionless rank-four tensor of an isotropic Kerr medium.

    The instantaneous third-order polarisation of an isotropic medium in the
    analytic-signal convention is

    ``P_i = eps0 chi3 (2 |E|^2 E_i + (E.E) conj(E_i)) / 4``

    (``docs/guide/media/nonlinear.md``), and writing it as the general
    contraction ``P_i = eps0 chi3 T_ijkl E_j E_k conj(E_l)`` that
    :class:`RegionalResponseTables` carries gives Kleinman-symmetric

    ``T_ijkl = (d_ij d_kl + d_ik d_jl + d_il d_jk) / 4``,

    whose three terms contribute ``E_i (E.E*)``, ``E_i (E.E*)`` and ``conj(E_i)
    (E.E)`` respectively. **The conjugated field is the last index**, which is the
    index convention of the record and of the reference pass that reads it.

    On the **real** three-component field the same contraction is the real cubic
    ``P = eps0 T:EEE`` the legacy field route evaluates, whose scalar limit is
    ``P = eps0 chi3 E^3``. Each of the three delta terms then contributes
    ``E_i (E.E)``, so the normalisation is a third and not a quarter:

    ``T_ijkl = (d_ij d_kl + d_ik d_jl + d_il d_jk) / 3``.

    The two are one model in two representations. For a narrowband real field
    ``E = Re(A e^(-i w t))`` the real cubic's fundamental component has analytic
    signal ``3 eps0 chi3 |A|^2 A / 4``, which is exactly what the analytic form
    returns; the real cubic carries the third harmonic beside it and the analytic
    form does not.

    Parameters
    ----------
    form : {'analytic', 'real'}, optional
        Which field the tensor is contracted against: the analytic signal, the
        default, or the real three-component field.

    Returns
    -------
    numpy.ndarray
        The tensor, shape ``(3, 3, 3, 3)``, dimensionless.

    Raises
    ------
    TranslationError
        If `form` is neither.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.responses import isotropic_kerr_tensor
    >>> field = np.array([0.3 + 0.2j, -0.1 + 0.4j, 0.05 - 0.15j])
    >>> direct = (2 * np.vdot(field, field).real * field
    ...           + (field @ field) * field.conj()) / 4
    >>> contracted = np.einsum("ijkl,j,k,l->i", isotropic_kerr_tensor(),
    ...                        field, field, field.conj())
    >>> bool(np.allclose(contracted, direct))
    True
    >>> real = np.array([0.3, -0.1, 0.05])
    >>> cubic = np.einsum("ijkl,j,k,l->i", isotropic_kerr_tensor("real"),
    ...                   real, real, real)
    >>> bool(np.allclose(cubic, (real @ real) * real))
    True
    """
    if form not in KERR_FORM_MODEL_IDS:
        msg = (
            "an isotropic Kerr tensor is contracted against the analytic signal or "
            f"the real field, got form={form!r}"
        )
        raise TranslationError(msg)
    delta = np.eye(3)
    return (
        np.einsum("ij,kl->ijkl", delta, delta)
        + np.einsum("ik,jl->ijkl", delta, delta)
        + np.einsum("il,jk->ijkl", delta, delta)
    ) / (4.0 if form == "analytic" else 3.0)


DELAYED_MODEL_ID: Final[str] = "silica-tensor-raman"
"""The model tag a compiled record carries when a region declares a delayed term."""

MODEL_ID_SEPARATOR: Final[str] = "+"
"""What joins the model tags of one compiled record's ``model_id``.

The identity is a ``+``-joined list whose **first** tag is the instantaneous form
(:data:`KERR_FORM_MODEL_IDS`) and whose remaining tags name the delayed models, in
compilation order. A backend reads the form back from the first tag and refuses an
identity it does not know rather than executing it as one it does.
"""


def instantaneous_model_id(form: KerrForm, *, delayed: bool) -> str:
    """Return the ``model_id`` a compiled regional record carries.

    Parameters
    ----------
    form : {'analytic', 'real'}
        The instantaneous cubic's form.
    delayed : bool
        Whether the record carries the silica tensor Raman term.

    Returns
    -------
    str
        For example ``isotropic-kerr-real+silica-tensor-raman``.

    Raises
    ------
    TranslationError
        If the form is not one this package compiles, named rather than surfacing
        as a bare lookup failure -- the same refusal
        :func:`isotropic_kerr_tensor` makes.

    Examples
    --------
    >>> from solprop.translate.responses import instantaneous_model_id
    >>> instantaneous_model_id("real", delayed=True)
    'isotropic-kerr-real+silica-tensor-raman'
    """
    if form not in KERR_FORM_MODEL_IDS:
        msg = (
            "a compiled regional record is tagged with the analytic-signal cubic "
            f"or the real one, got form={form!r}"
        )
        raise TranslationError(msg)
    tags = [KERR_FORM_MODEL_IDS[form]]
    if delayed:
        tags.append(DELAYED_MODEL_ID)
    return MODEL_ID_SEPARATOR.join(tags)


def regional_responses(  # noqa: PLR0913 - a medium, a region, a grid and two policies
    chi3: Sequence[float] | np.ndarray,
    regions: Sequence[int] | np.ndarray,
    *,
    raman: Sequence[SilicaTensorResponse | None] | None = None,
    n_to: int,
    dt_fine: float,
    form: KerrForm = "analytic",
    policy_id: str = "dense-nodes",
) -> RegionalResponseTables:
    """Compile one material region's declared responses into the seam's leaf.

    An absolute ``chi3`` per responding region, the isotropic instantaneous tensor
    of :func:`isotropic_kerr_tensor` in the declared form, and -- for a region that
    declares one -- the two delayed channels of
    :class:`solprop.nonlinear.raman.SilicaTensorResponse`. A node whose region names
    no response has no polarisation, which is how a linear cladding is declared.

    **One susceptibility, shared.** Where a delayed response is declared, the
    instantaneous tensor is scaled by its ``1 - f_R`` and the delayed kernels carry
    ``(3/2) f_R`` (:data:`solprop.nonlinear.raman.SILICA_TENSOR_AMPLITUDE`), so the
    record multiplies by ``chi3`` once and the two shares add to **one** ``chi3`` on
    the fundamental band: a field-domain delayed kernel of integral ``X`` acts there
    like a Kerr term of ``(2/3) X``, because its drive is the cycle average, so
    ``(2/3)(3/2) f_R + (1 - f_R) = 1``.
    Every response of one record carries the **same number of delayed terms**,
    because the leaf's term axis is shared; a region with no delayed response gets
    zero kernels on that axis, which contribute nothing.

    ``chi3`` is stored as one Chebyshev coefficient on one interval, because a
    declared material is constant along a constant guide; a z-varying
    susceptibility needs the density path of the legacy compiler and is not
    compiled here.

    Parameters
    ----------
    chi3 : sequence of float
        Absolute third-order susceptibility of each response, m²/V², finite.
    regions : sequence of int
        The node region each response acts in, non-negative, one per ``chi3``.
    raman : sequence or None, optional
        The delayed response each region declares, or ``None`` in that slot for a
        region with none; ``None`` for the whole sequence means no region declares
        one. One entry per ``chi3``.
    n_to : int
        Fine time samples of the run's grid, positive.
    dt_fine : float
        Fine sample spacing in s, positive; the temporal quadrature weight.
    form : {'analytic', 'real'}, optional
        Which cubic the instantaneous tensor is: the analytic-signal one, the
        default, which carries the fundamental band alone, or the real cubic of the
        field route, which carries the third harmonic. It fixes both the tensor and
        the first tag of the ``model_id`` a backend reads the form back from
        (:data:`KERR_FORM_MODEL_IDS`).
    policy_id : str, optional
        Numeric-policy identity recorded on the leaf.

    Returns
    -------
    RegionalResponseTables
        The validated leaf of contract section 5.

    Raises
    ------
    TranslationError
        If the sequences disagree in length, or the record does not validate.

    Examples
    --------
    >>> from solprop.nonlinear.raman import SilicaTensorResponse
    >>> from solprop.translate.responses import regional_responses
    >>> tables = regional_responses([2.2e-22], [0], n_to=16, dt_fine=1e-16)
    >>> tables.n_responses, tables.n_components, tables.n_terms
    (1, 3, 0)
    >>> tables.model_id
    'isotropic-kerr'
    >>> delayed = regional_responses(
    ...     [2.2e-22], [0], raman=[SilicaTensorResponse()],
    ...     n_to=16, dt_fine=1e-16, form="real",
    ... )
    >>> delayed.n_terms, delayed.model_id
    (2, 'isotropic-kerr-real+silica-tensor-raman')
    """
    values = np.asarray(chi3, dtype=np.float64).reshape(-1)
    index = np.asarray(regions, dtype=np.int32).reshape(-1)
    if values.size != index.size or not values.size:
        msg = (
            f"regional responses need one chi3 per region, got {values.size} "
            f"and {index.size}"
        )
        raise TranslationError(msg)
    delayed = [None] * values.size if raman is None else list(raman)
    if len(delayed) != values.size:
        msg = (
            f"regional responses need one delayed declaration per region, got "
            f"{len(delayed)} for {values.size} response(s)"
        )
        raise TranslationError(msg)
    tensor = isotropic_kerr_tensor(form)
    shares = np.array(
        [1.0 if one is None else one.instantaneous_share for one in delayed]
    )
    instantaneous = shares[:, None, None, None, None] * tensor
    terms = _delayed_terms(delayed, n_to=n_to, dt_fine=dt_fine)
    return RegionalResponseTables(
        index,
        values.reshape(1, 1, values.size),
        instantaneous,
        terms.tensors,
        terms.kernels,
        terms.support,
        np.full(n_to, float(dt_fine)),
        instantaneous_model_id(form, delayed=terms.n_terms > 0),
        policy_id,
    )


@dataclass(frozen=True)
class _DelayedTerms:
    """The three delayed arrays of one compiled regional record."""

    tensors: Float[np.ndarray, "n_responses n_terms 3 3 3 3"]
    kernels: Float[np.ndarray, "n_intervals n_deg n_responses n_terms n_to_doubled"]
    support: Int32[np.ndarray, "n_responses n_terms"]

    @property
    def n_terms(self) -> int:
        """Return how many delayed channels each response carries.

        Returns
        -------
        int
            Zero, or the two channels of the silica tensor model.
        """
        return int(self.tensors.shape[1])


def _delayed_terms(
    declared: Sequence[SilicaTensorResponse | None], *, n_to: int, dt_fine: float
) -> _DelayedTerms:
    """Sample each region's delayed response onto the doubled fine time axis.

    The kernels are stored causal-half-then-zeros so that a transform product is the
    **linear** convolution and the response's tail cannot wrap onto the head of the
    drive, exactly as :mod:`solprop.translate.raman` stores the scalar one.

    Parameters
    ----------
    declared : sequence
        One :class:`solprop.nonlinear.raman.SilicaTensorResponse` or ``None`` per
        response, in the record's own order.
    n_to : int
        Fine time samples of the run's grid, positive.
    dt_fine : float
        Fine sample spacing in s, positive.

    Returns
    -------
    _DelayedTerms
        The tensors, the kernels and the support counts, with an empty term axis
        when no region declares a delayed response.
    """
    responses = len(declared)
    present = [one for one in declared if one is not None]
    if not present:
        return _DelayedTerms(
            np.zeros((responses, 0, 3, 3, 3, 3)),
            np.zeros((1, 1, responses, 0, 2 * n_to)),
            np.zeros((responses, 0), dtype=np.int32),
        )
    n_terms = int(present[0].tensors().shape[0])
    tensors = np.zeros((responses, n_terms, 3, 3, 3, 3))
    kernels = np.zeros((1, 1, responses, n_terms, 2 * n_to))
    support = np.zeros((responses, n_terms), dtype=np.int32)
    delays = np.arange(n_to) * float(dt_fine)
    for row, one in enumerate(declared):
        if one is None:
            continue
        tensors[row] = one.tensors()
        kernels[0, 0, row, :, :n_to] = one.kernels(delays)
        support[row] = n_to
    logger.debug(
        "regional responses: %d delayed term(s) on %d region(s), %d fine samples",
        n_terms,
        len(present),
        n_to,
    )
    return _DelayedTerms(tensors, kernels, support)


@dataclass(frozen=True, eq=False)
class CompiledResponses:
    """The nonlinear term of the propagation equation, ready for a kernel.

    Section 5 of the kernel contract. Every array is a read-only, C-contiguous copy of
    the stated dtype.

    Attributes
    ----------
    response_order : tuple of ResponseStage
        The fused-pass sequence, in execution order.
    kerr_coefficients : KerrCoefficients or None
        The compiled Kerr term, or ``None`` when no Kerr response was declared.
    ionisation_table : tuple of RateTable
        Channel rate tables, identity-matched to plasma_channels.
    plasma_channels : tuple of PlasmaChannel
        Independent labelled populations with rates, potentials and density paths.
    raman_terms : tuple of RamanCoefficients
        Raman-active components with their amplitude densities and total-density
        linewidth tables. Each component may carry rotation, vibration or both.
    labels : tuple of str
        Unique component labels, including non-ionising and zero-density components.
    species_names : tuple of str
        Chemical names in label order; a species may occur under several labels.
    species_densities : tuple of PiecewiseChebyshev
        Number densities in m^-3, in label order.
    density : solprop.translate.chebyshev.PiecewiseChebyshev
        Number density along z, in m^-3 for a gas or dimensionless 1 for a medium whose
        ``chi3`` is already absolute, as a table.
    spectral_responses : tuple of EnvelopePlasmaAssembly
        Typed post-transform assembly; empty or the ``envelope_plasma`` record.
    envelope_plasma : EnvelopePlasmaAssembly or None
        Independent envelope populations, cycle data and spectral multipliers.
    prefactor_representation : {'constant', 'expansion', 'table'}
        Which representation carries the prefactor, following section 3's choice for
        ``Phi`` and made the same way. The first two are the device form; the z table is
        the oracle's general representation
        (:attr:`solprop.translate.problem.KernelProblem.device_ready`).
    prefactor : solprop.translate.chebyshev.PiecewiseChebyshev or None
        The UPPE or envelope prefactor times the crop scale, as a table along z with
        value shape ``(n_modes, n_pol, n_omega)``, complex128, in m/(V s) per unit
        polarisation. Purely imaginary and exactly zero outside the active band at every
        position. The table spans the same z range as ``Phi``. Present when
        `prefactor_representation` is ``"table"`` and also for ``"constant"``, whose one
        interval of degree 0 is exact; absent for ``"expansion"``.
    prefactor_expansion : solprop.translate.expansion.ParameterExpansion or None
        The same quantity as an expansion in the mode set's scalar paths, with the plain
        products of the basis polynomials as its weights rather than their integrals,
        since the prefactor is not cumulative. Present exactly when
        `prefactor_representation` is ``"expansion"``.
    prefactor_constant : numpy.ndarray or None
        The prefactor itself, complex128, shape ``(n_modes, n_pol, n_omega)``, present
        exactly when `prefactor_representation` is ``"constant"`` -- every mode reports
        ``z_constant`` -- so that a backend takes the array rather than a Clenshaw pass.
        The mirror of :attr:`solprop.translate.linop.LinearOperator.phi_constant`.
    temporal_absorber : numpy.ndarray or None
        The temporal collar as a rate in 1/m on the coarse time axis, shape
        ``(n_t,)``, or ``None``.
    temporal_absorber_fine : numpy.ndarray or None
        The same rate on the fine time axis, shape ``(n_to,)``, which the
        ``"nonlinear"`` absorber scheme needs, or ``None``.
    absorber_scheme : {'split_step', 'nonlinear'}
        Which of the two exact applications of the temporal rate to use
        (:data:`solprop.translate.masks.AbsorberScheme`).
    separable : solprop.translate.spatial.SeparableOperator or None
        Shared constant spatial operator and prefactor coefficients, present exactly
        for representation ``"separable"``; identical to ``linear.separable``.
    regional_responses : RegionalResponseTables or None
        Kernel schema 12: the node-region response assignment of a general modal
        guide, preserving the existing gas law and table identities. ``None`` on
        every route that goes through the compiled fused pass above.
    dispersive_kerr : DispersiveKerrTerms or None
        Kernel schema 24: SEP 6's frequency-dependent electronic response, one
        channel per population, or ``None``.
    plasma_integral : RunningIntegral
        Kernel schema 25: the running-integral rule of the field-grid plasma column
        (SEP 18), one per problem because the channels share the two current
        integrals. The plain trapezoid without field plasma channels.

    Raises
    ------
    TranslationError
        If the constant prefactor does not have the table's value shape, an absorber
        rate is present without the other, or the three Raman fields are not all present
        or all absent.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.translate.responses import compile_responses
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> problem = problem_from([0.0, 0.0, -1e-26], omega_ref=grid.omega0, gamma=0.1,
    ...                        shock=False)
    >>> compiled = compile_responses([problem.kerr], grid, problem.mode, density=1.0,
    ...                              z_max=1.0, shock=False, prefactor_index=1.0)
    >>> compiled.response_order
    (<ResponseStage.KERR: 'kerr'>, <ResponseStage.ACCUMULATE: 'accumulate'>)
    """

    response_order: tuple[ResponseStage, ...]
    kerr_coefficients: KerrCoefficients | None
    ionisation_table: tuple[RateTable, ...]
    plasma_channels: tuple[PlasmaChannel, ...]
    raman_terms: tuple[RamanCoefficients, ...]
    density: PiecewiseChebyshev
    spectral_responses: tuple[EnvelopePlasmaAssembly, ...]
    prefactor_representation: PhiRepresentation
    prefactor: PiecewiseChebyshev | None
    prefactor_expansion: ParameterExpansion | None
    prefactor_constant: Complex[np.ndarray, "n_modes n_pol n_omega"] | None
    temporal_absorber: Float[np.ndarray, " n_t"] | None
    temporal_absorber_fine: Float[np.ndarray, " n_to"] | None
    absorber_scheme: AbsorberScheme = DEFAULT_ABSORBER_SCHEME
    separable: SeparableOperator | None = None
    labels: tuple[str, ...] = ("medium",)
    species_names: tuple[str, ...] = ("medium",)
    species_densities: tuple[PiecewiseChebyshev, ...] = ()
    envelope_plasma: EnvelopePlasmaAssembly | None = None
    regional_responses: RegionalResponseTables | None = None
    dispersive_kerr: DispersiveKerrTerms | None = None
    plasma_integral: RunningIntegral = field(default_factory=RunningIntegral.trapezoid)

    def __post_init__(self) -> None:
        """Adopt a read-only constant prefactor and check the pairs.

        Whether the constant is present at all is checked against the flags by
        :meth:`solprop.translate.problem.KernelProblem.validate`, which is where the two
        meet; here only its shape against the table's can be seen.

        Raises
        ------
        TranslationError
            Naming the offending field.
        """
        self._check_prefactor_representation()
        expected = () if self.envelope_plasma is None else (self.envelope_plasma,)
        if self.spectral_responses != expected:
            raise TranslationError(
                "spectral_responses must name the envelope plasma assembly"
            )
        if self.envelope_plasma is not None and self.plasma_channels:
            raise TranslationError("field and envelope plasma channels cannot coexist")
        if self.plasma_integral.policy != "trapezoid" and not self.plasma_channels:
            raise TranslationError(
                "a band plasma_integral needs field plasma channels to integrate"
            )
        if not self.species_densities:
            object.__setattr__(self, "species_densities", (self.density,))
        if self.prefactor_constant is not None:
            constant = np.ascontiguousarray(
                self.prefactor_constant, dtype=np.complex128
            )
            constant.setflags(write=False)
            object.__setattr__(self, "prefactor_constant", constant)
            if constant.shape != self.prefactor_value_shape:
                msg = (
                    f"prefactor_constant has shape {constant.shape}, expected "
                    f"{self.prefactor_value_shape} to match the prefactor"
                )
                raise TranslationError(msg)
        if (self.temporal_absorber is None) != (self.temporal_absorber_fine is None):
            msg = (
                "temporal_absorber and temporal_absorber_fine must both be present or "
                "both absent; the two absorber schemes need one axis each"
            )
            raise TranslationError(msg)

    @property
    def harmonic_carrier(self) -> bool:
        """Return whether the pass needs ``grids.carrier_2w0``.

        Returns
        -------
        bool
            True when an envelope grid's constant Kerr term radiates the harmonic, or
            its dispersive term keeps the full cubic.
        """
        kerr = self.kerr_coefficients
        dispersive = self.dispersive_kerr
        return (kerr is not None and kerr.harmonic_factor != 0.0) or (
            dispersive is not None and dispersive.carrier
        )

    @property
    def electronic_thg(self) -> bool | None:
        """Return the field grid's harmonic form, read off the electronic term.

        Returns
        -------
        bool or None
            Whether the electronic cubic keeps the third harmonic on a field grid --
            the constant Kerr term's real cubic, or the dispersive term's full
            sector -- or ``None`` with no electronic term.
        """
        if self.kerr_coefficients is not None:
            return not self.kerr_coefficients.analytic_signal
        if self.dispersive_kerr is not None:
            return self.dispersive_kerr.sector == "full"
        return None

    @property
    def n_species(self) -> int:
        """Return the number of labelled populations.

        Returns
        -------
        int
            Length of the species axis, including inactive populations.
        """
        return len(self.labels)

    @property
    def population_channels(self) -> tuple[PlasmaChannel | EnvelopePlasmaChannel, ...]:
        """Return labelled population channels in material order.

        Returns
        -------
        tuple
            Field channels or envelope channels. Their stage density and initial
            population paths share the same scalar packing order.
        """
        return (
            self.plasma_channels
            if self.envelope_plasma is None
            else self.envelope_plasma.channels
        )

    @property
    def plasma_coefficients(self) -> PlasmaChannel | None:
        """Return the one-channel leaf for the original specialised execution path.

        Returns
        -------
        PlasmaChannel or None
            The sole channel, or ``None`` when there are zero or multiple channels.
        """
        return self.plasma_channels[0] if len(self.plasma_channels) == 1 else None

    @property
    def raman_coefficients(self) -> RamanCoefficients | None:
        """Return the one-component Raman leaf for its original arithmetic.

        Returns
        -------
        RamanCoefficients or None
            The sole component, or ``None`` when there are zero or multiple terms.
        """
        return self.raman_terms[0] if len(self.raman_terms) == 1 else None

    def _check_prefactor_representation(self) -> None:
        """Check that the prefactor's stored representation is the one the kind names.

        A kind that is neither is caught by these two checks together and named by
        their message, so it needs no membership test of its own -- which could not be
        reached anyway from a constructor whose annotation is a literal.

        Raises
        ------
        TranslationError
            If the wrong member is present for the kind named.
        """
        if (self.separable is not None) != (
            self.prefactor_representation == "separable"
        ):
            msg = "separable must be present exactly for a separable prefactor"
            raise TranslationError(msg)
        if (self.prefactor_expansion is not None) != (
            self.prefactor_representation == "expansion"
        ):
            msg = (
                "prefactor_expansion must be present exactly when "
                "prefactor_representation is 'expansion', got "
                f"{self.prefactor_representation!r}"
            )
            raise TranslationError(msg)
        constant = self.prefactor_constant is not None
        if constant != (self.prefactor_representation == "constant"):
            msg = (
                "prefactor_constant must be present exactly when "
                "prefactor_representation is 'constant', got "
                f"{self.prefactor_representation!r}"
            )
            raise TranslationError(msg)
        # A constant carries its exact one interval as well as the array, so that
        # `prefactor_span` and the table checks read the same field for every kind;
        # only an expansion replaces the table outright.
        wants_table = self.prefactor_representation not in ("expansion", "separable")
        if (self.prefactor is not None) != wants_table:
            msg = (
                "prefactor must be present for a 'table' and for a 'constant' and "
                "absent for expansion/separable, got "
                f"{self.prefactor_representation!r}"
            )
            raise TranslationError(msg)

    @property
    def prefactor_value_shape(self) -> tuple[int, ...]:
        """Return the shape of the prefactor at one position.

        Returns
        -------
        tuple of int
            ``(n_modes, n_pol, n_omega)``, whichever representation carries it.
        """
        if self.separable is not None:
            return self.separable.value_shape[1:]
        if self.prefactor_expansion is not None:
            return self.prefactor_expansion.value_shape
        # `_check_prefactor_representation` has already made the narrowing true; a cast
        # rather than an assertion, which `python -O` strips.
        return cast("PiecewiseChebyshev", self.prefactor).value_shape

    @property
    def prefactor_span(self) -> tuple[float, float]:
        """Return the range of z the prefactor covers.

        Returns
        -------
        tuple of float
            ``(first breakpoint, last breakpoint)`` in m, the same as ``Phi``'s.
        """
        if self.separable is not None:
            return self.separable.z_span
        if self.prefactor_expansion is not None:
            return self.prefactor_expansion.z_span
        return cast("PiecewiseChebyshev", self.prefactor).z_span


def _prefactor_kind(constant: bool, expanded: bool) -> PhiRepresentation:
    """Return the representation kind the three prefactor members spell out.

    Parameters
    ----------
    constant : bool
        Whether every mode reports ``z_constant``, so the array itself is carried.
    expanded : bool
        Whether an expansion in the scalar parameters was built.

    Returns
    -------
    str
        ``"constant"``, ``"expansion"`` or ``"table"``.
    """
    if constant:
        return "constant"
    return "expansion" if expanded else "table"


def prefactor_at(
    responses: CompiledResponses, z: float, *, helpers: Helpers | None = None
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return the prefactor at one position, taking the fast path when there is one.

    The one way a backend evaluates the prefactor, as
    :func:`solprop.translate.linop.delta_phi` is the one way it evaluates the linear
    flow: the array of ``prefactor_constant`` when the mode set does not vary, and one
    Clenshaw pass over the table otherwise. Going through this function rather than
    reading the table's arrays keeps the representation the translation layer's
    business, which is what lets it change without touching a backend.

    Parameters
    ----------
    responses : CompiledResponses
        Section 5 of the kernel contract.
    z : float
        Position along the waveguide, m, inside the table's span.
    helpers : solprop._helpers.Helpers or None, optional
        The evaluating backend's Chebyshev helpers, for a z table read during a run;
        ``None``, the default, takes the shared translation-time evaluation.

    Returns
    -------
    numpy.ndarray
        The prefactor times the crop scale, complex128, shape
        ``(n_modes, n_pol, n_omega)``; the stored array itself on the fast path.

    Raises
    ------
    TranslationError
        If `z` lies outside the table.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.translate.responses import compile_responses, prefactor_at
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> problem = problem_from([0.0, 0.0, -1e-26], omega_ref=grid.omega0, gamma=0.1,
    ...                        shock=False)
    >>> compiled = compile_responses([problem.kerr], grid, problem.mode, density=1.0,
    ...                              z_max=1.0, shock=False, prefactor_index=1.0)
    >>> prefactor_at(compiled, 0.5) is compiled.prefactor_constant
    True
    """
    if responses.separable is not None:
        low, high = responses.prefactor_span
        if not low <= z <= high:
            msg = f"z = {z!r} m lies outside the separable span [{low}, {high}] m"
            raise TranslationError(msg)
        return separable_values(responses.separable)[1]
    if responses.prefactor_constant is not None:
        low, high = responses.prefactor_span
        if not low <= z <= high:
            msg = f"z = {z!r} m lies outside the table, which covers [{low}, {high}] m"
            raise TranslationError(msg)
        return responses.prefactor_constant
    if responses.prefactor_expansion is not None:
        return responses.prefactor_expansion(z)
    return cast("PiecewiseChebyshev", responses.prefactor)(z, helpers=helpers)


def _prefactor_at(
    modes: Sequence[Mode],
    grid: TimeGrid,
    n_pol: int,
    mode_averaged: bool,
    shock: bool,
    index: float | None,
) -> Callable[[float], Complex[np.ndarray, "n_modes n_pol n_omega"]]:
    """Return the prefactor of every mode at one position, with the crop scale folded.

    The **effective area is held at the reference position** even in the mode-averaged
    case, because the transform's ``taper_scale`` already carries ``A_eff(z)`` and
    divides the projection by it (``kernel-contract.md`` section 4); carrying it here as
    well would count the area one and a half times on a taper. What varies here is
    ``Re n_eff(omega, z)``, which nothing else carries. The identity the two mechanisms
    satisfy together is ``prefactor(z) = mode_average_prefactor(z) * taper_scale(z) *
    crop_scale``, and it is a test.

    Written in this order the value at ``z = 0`` is bitwise
    ``mode_average_prefactor(mode, grid, 0.0) * crop_scale``, which is what
    ``prefactor_constant`` and every backend's fast path expect.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set, in order.
    grid : solprop.grid.TimeGrid
        The grid.
    n_pol : int
        Polarisation components; the prefactor is the same for both components of one
        mode and is stored for each.
    mode_averaged : bool
        Whether the effective-area conversion is folded in (the one-node case).
    shock : bool
        Keep the ``omega`` dependence of the numerator.
    index : float or None
        Replace the effective index by this constant; the GNLSE passes 1.0.

    Returns
    -------
    callable
        ``z -> p`` of shape ``(n_modes, n_pol, n_omega)``.
    """
    areas = [sqrtw_to_vm(one.aeff(0.0)) if mode_averaged else 1.0 for one in modes]

    def prefactor_at(z: float) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the prefactor at one position, in the canonical layout.

        Parameters
        ----------
        z : float
            Position along the waveguide, m.

        Returns
        -------
        numpy.ndarray
            The prefactor, shape ``(n_modes, n_pol, n_omega)``.
        """
        columns = [
            uppe_prefactor(one, grid, z, shock=shock, index=index)
            / area
            * grid.crop_scale
            for one, area in zip(modes, areas, strict=True)
        ]
        return np.repeat(np.stack(columns)[:, np.newaxis, :], n_pol, axis=1)

    return prefactor_at


def _prefactor_expansion(  # noqa: PLR0913 - the same knobs the table build takes
    modes: Sequence[Mode],
    grid: TimeGrid,
    n_pol: int,
    mode_averaged: bool,
    shock: bool,
    index: float | None,
    *,
    span: float,
    edges: Sequence[float] | np.ndarray,
    tol: float,
    n_max: int,
    max_intervals: int,
) -> ParameterExpansion | None:
    """Expand the prefactor in the mode set's scalar paths, as ``Phi`` is expanded.

    The prefactor's z dependence is ``Re n_eff(omega, z)`` alone -- the effective area
    is held at the reference position and carried by the transform's ``taper_scale`` --
    so it is a function of the same parameters as ``D`` and takes the same
    representation, with plain products of the basis polynomials as its weights rather
    than their integrals: the prefactor is a value at ``z``, not a cumulative one.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set; its declared paths are the dimensions.
    grid : solprop.grid.TimeGrid
        The grid.
    n_pol : int
        Polarisation components; the prefactor is stored for each.
    mode_averaged : bool
        Whether the effective-area conversion is folded in, at the reference position.
    shock : bool
        Keep the ``omega`` dependence of the numerator.
    index : float or None
        Replace the effective index by this constant; the GNLSE passes 1.0.
    span : float
        End of the tabulated range, m.
    edges : array_like
        Positions that must be interval edges of the scalar tables.
    tol : float
        Target bound on the uniform error relative to the value scale.
    n_max : int
        Highest Chebyshev degree tried on a parameter.
    max_intervals : int
        Interval budget for the scalar tables.

    Returns
    -------
    solprop.translate.expansion.ParameterExpansion or None
        The expansion, or ``None`` when the mode set does not allow one.
    """
    varying = parameter_paths(modes, 0.0, span)
    if varying is None:
        return None
    areas = [sqrtw_to_vm(one.aeff(0.0)) if mode_averaged else 1.0 for one in modes]
    frozen = [one for one in modes if isinstance(one, ParameterisedMode)]

    def prefactor_of(
        values: Mapping[str, float],
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the prefactor at one point of the parameter space.

        Parameters
        ----------
        values : collections.abc.Mapping
            One value per expanded path, by name.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_modes, n_pol, n_omega)``.
        """
        columns = [
            uppe_prefactor(
                one.at_parameters(values), grid, 0.0, shock=shock, index=index
            )
            / area
            * grid.crop_scale
            for one, area in zip(frozen, areas, strict=True)
        ]
        return np.repeat(np.stack(columns)[:, np.newaxis, :], n_pol, axis=1)

    return build_expansion(
        prefactor_of,
        varying,
        0.0,
        span,
        tol=tol,
        kinks=edges,
        n_max=n_max,
        max_intervals=max_intervals,
    )


def _prefactor_table(  # noqa: PLR0913 - the build's knobs, as `build_phi` has them
    prefactor_at: Callable[[float], Complex[np.ndarray, "n_modes n_pol n_omega"]],
    modes: Sequence[Mode],
    *,
    span: float,
    edges: Sequence[float] | np.ndarray,
    tol: float,
    n_max: int,
    max_intervals: int,
) -> PiecewiseChebyshev:
    """Tabulate the prefactor along z, the way the cumulative operator is tabulated.

    The same machinery as :func:`solprop.translate.linop.build_phi`, minus the parts it
    needs and this does not. The mode set's scalar paths are resolved first, because
    their structure is what the prefactor's structure is and splitting on their edges
    lets each interval hold a lower degree: measured on the target gradient capillary,
    9 intervals of degree 32 and 9.3 MB with the pre-split against 7 of degree 63 and
    14 MB without, at the same accuracy.

    No round-off floor is measured and no tolerance is relaxed, unlike ``build_phi``.
    Both exist there to recover from the eight digits the co-moving frame cancels out of
    ``beta``, and the prefactor has no such cancellation:
    ``-i omega sqrt(eps0 c A_eff / 2) / (2 eps0 c Re n_eff)`` is evaluated to a relative
    round-off of 8e-16 against the operator's 1e-11, and the build converges at the
    tolerance it is asked for.

    Parameters
    ----------
    prefactor_at : callable
        ``z -> p`` in the canonical layout, from :func:`_prefactor_at`.
    modes : sequence of Mode
        The mode set; its ``core_radius`` and ``density`` paths carry the structure.
    span : float
        End of the tabulated range, m: the same ``z_max (1 + margin)`` as ``Phi``.
    edges : array_like
        Positions that must be interval edges. :func:`compile_responses` is given the
        cumulative operator's hard kinks, so the two tables split on the same edges.
    tol : float
        Target bound on the uniform error relative to the value scale.
    n_max : int
        Highest Chebyshev degree tried on an interval before it is bisected.
    max_intervals : int
        Interval budget; exceeding it warns and leaves the table unresolved.

    Returns
    -------
    solprop.translate.chebyshev.PiecewiseChebyshev
        The prefactor over ``[0, span]``, with read-only arrays.
    """

    def sample(
        z: Float[np.ndarray, " n_z"], *_: object
    ) -> Complex[np.ndarray, "n_z n_modes n_pol n_omega"]:
        """Return the prefactor at every position of a Chebyshev node set.

        Parameters
        ----------
        z : numpy.ndarray
            Positions along the waveguide, m, shape ``(n_z,)``.
        *_ : object
            The path values the builder resolved; the prefactor is evaluated from ``z``
            directly, so they are not used.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_z, n_modes, n_pol, n_omega)``.
        """
        return np.stack([prefactor_at(float(one)) for one in z])

    table = build_from_paths(
        [path.__call__ for path in declared_paths(modes)],
        sample,
        0.0,
        span,
        tol=tol,
        mandatory_breakpoints=sorted(
            {float(edge) for edge in edges if 0.0 < float(edge) < span}
        ),
        n_max=n_max,
        max_intervals=max_intervals,
    )
    if table.budget_exceeded:
        logger.warning(
            "prefactor: the interval budget was exhausted over [0, %.6g] m at %.1e; "
            "the table carries unresolved intervals and its accuracy is not stated",
            span,
            tol,
        )
    return PiecewiseChebyshev(
        _readonly(table.coefficients),
        _readonly(table.breakpoints),
        _readonly(table.degrees),
        _readonly(table.hard_kinks),
        table.budget_exceeded,
    )


def _log_prefactor(compiled: "CompiledResponses", z_max: float) -> None:  # noqa: UP037
    """Report what the prefactor cost and how much of z it carries.

    The swing is measured from the representation rather than from the physics, so the
    report is free: it is the largest departure from the entrance value over the run,
    which is exactly what freezing the prefactor at ``z = 0`` used to discard.

    Parameters
    ----------
    compiled : CompiledResponses
        Section 5, already assembled.
    z_max : float
        Length the run covers, m.
    """
    if compiled.separable is not None:
        logger.debug(
            "prefactor: constant separable coefficients, %d spatial columns, "
            "%d frequencies, no coefficient volume",
            compiled.separable.kperp_sq.size,
            compiled.separable.k_squared.size,
        )
        return
    reference = prefactor_at(compiled, 0.0)
    scale = float(np.max(np.abs(reference)))
    swing = 0.0
    if scale > 0.0:
        for z in np.linspace(0.0, z_max, _PROFILE_PROBES)[1:]:
            here = prefactor_at(compiled, float(z))
            swing = max(swing, float(np.max(np.abs(here - reference))) / scale)
    expansion = compiled.prefactor_expansion
    if expansion is None:
        table = cast("PiecewiseChebyshev", compiled.prefactor)
        shape = f"{table.degrees.size} intervals of degree up to {table.degrees.max()}"
        nbytes = table.coefficients.nbytes
    else:
        shape = (
            f"an expansion in {expansion.parameters} to degrees "
            f"{expansion.degrees.tolist()}, {expansion.n_terms} terms"
        )
        nbytes = expansion.nbytes
    logger.debug(
        "prefactor: %s, %.3f MB, carrying a swing of %.3e relative over %.6g m",
        shape,
        nbytes / 2**20,
        swing,
        z_max,
    )


@dataclass(frozen=True)
class _PrefactorRepresentation:
    """Group the mutually exclusive forms of an already translated prefactor."""

    kind: PhiRepresentation
    table: PiecewiseChebyshev | None = None
    expansion: ParameterExpansion | None = None
    constant: Complex[np.ndarray, "n_modes n_pol n_omega"] | None = None
    separable: SeparableOperator | None = None


def _translated_prefactor(
    supplied: PiecewiseChebyshev | ParameterExpansion | SeparableOperator | None,
) -> _PrefactorRepresentation | None:
    """Classify a caller's own normalisation without rebuilding it as a modal one.

    Parameters
    ----------
    supplied : object or None
        A translated table, parameter expansion or separable coefficient record.
        None requests the ordinary modal normalisation.

    Returns
    -------
    _PrefactorRepresentation or None
        The supplied representation, or None for a modal problem.
    """
    if supplied is None:
        return None
    if isinstance(supplied, SeparableOperator):
        return _PrefactorRepresentation("separable", separable=supplied)
    if isinstance(supplied, ParameterExpansion):
        return _PrefactorRepresentation("expansion", expansion=supplied)
    constant = supplied.degrees.size == 1 and int(supplied.degrees[0]) == 0
    return _PrefactorRepresentation(
        "constant" if constant else "table",
        table=supplied,
        constant=np.asarray(supplied(supplied.z_span[0]), dtype=np.complex128)
        if constant
        else None,
    )


@dataclass(frozen=True)
class _MaterialProfiles:
    """Retain labelled density paths while compiling one material."""

    labels: tuple[str, ...]
    names: tuple[str, ...]
    paths: tuple[ZProfile, ...]
    total: ZProfile


@dataclass(frozen=True)
class _ResponseBuild:
    """Keep the shared grid and scalar-table controls of one response build."""

    grid: TimeGrid
    span: float
    tol: float
    field_range: tuple[float, float] | None
    raman_undamped: Mapping[str, Float[np.ndarray, "... n_to_doubled"]] | None = None


def _material_profiles(
    modes: tuple[Mode, ...], mixture: Mixture | None, profile: ZProfile
) -> _MaterialProfiles:
    """Resolve component profiles while preserving the original single path.

    Parameters
    ----------
    modes : Sequence of Mode or None
        Mode set carrying the material declaration.
    mixture : Mixture or None
        Ordered component populations; None retains the legacy material path.
    profile : ZProfile
        Total density path in m^-3, or the legacy nonlinear scale.

    Returns
    -------
    _MaterialProfiles
        Species metadata with each component density and the total path.
    """
    if mixture is None and modes:
        declared = getattr(modes[0], "mixture", None)
        if isinstance(declared, Mixture):
            mixture = declared
    if mixture is not None:
        return _MaterialProfiles(
            mixture.labels,
            tuple(c.species.name for c in mixture.components),
            tuple(c.density for c in mixture.components),
            profile,
        )
    gas = getattr(modes[0], "gas", None) if modes else None
    name = str(gas) if gas is not None else "medium"
    return _MaterialProfiles((name,), (name,), (profile,), profile)


def _response_label(
    response: KerrResponse
    | DispersiveKerrResponse
    | PlasmaResponse
    | EnvelopePlasmaResponse
    | RamanResponse,
    material: _MaterialProfiles,
) -> str:
    """Require an unambiguous population for every declared response.

    Parameters
    ----------
    response : object
        Declared response whose optional species field binds a component label.
    material : _MaterialProfiles
        Ordered labels, species names and density paths.

    Returns
    -------
    str
        Validated label in the material declaration.
    """
    label = response.species
    if label is None:
        if len(material.labels) == 1:
            return material.labels[0]
        if isinstance(response, PlasmaResponse | EnvelopePlasmaResponse):
            name = (
                response.gas.name
                if isinstance(response.gas, Species)
                else str(response.gas)
            )
            matches = tuple(
                label
                for label, name_in_material in zip(
                    material.labels, material.names, strict=True
                )
                if name_in_material == name
            )
            if len(matches) == 1:
                return matches[0]
        raise TranslationError(
            f"{type(response).__name__} needs a component label for {material.labels}"
        )
    if label not in material.labels:
        raise TranslationError(
            f"unknown response component label {label!r}; expected {material.labels}"
        )
    return label


def _compile_kerr_components(
    responses: tuple[KerrResponse, ...],
    material: _MaterialProfiles,
    densities: Mapping[str, PiecewiseChebyshev],
    build: _ResponseBuild,
) -> KerrCoefficients | None:
    """Fold species amplitudes once, retaining the one-component multiplication.

    Parameters
    ----------
    responses : tuple
        Declared responses of one kind in user order.
    material : _MaterialProfiles
        Ordered labels, species names and density paths.
    densities : Mapping of str to PiecewiseChebyshev
        Density tables in m^-3, keyed by component label.
    build : _ResponseBuild
        Grid, span in m, interpolation tolerance and ionisation field range.

    Returns
    -------
    KerrCoefficients or None
        One cubic form with its own composite density table.
    """
    if not responses:
        return None
    # `compile_responses` has already refused electronic responses that disagree on
    # thg, so every component here has one harmonic sector.
    labels = tuple(_response_label(response, material) for response in responses)
    if len(set(labels)) != len(labels):
        raise TranslationError("two KerrResponse declarations on the same species")
    if len(responses) == 1:
        return replace(
            kerr_coefficients(responses[0], build.grid), density=densities[labels[0]]
        )
    paths = dict(zip(material.labels, material.paths, strict=True))
    susceptibility = Sum(
        tuple(
            Scaled(paths[label], response.chi3)
            for response, label in zip(responses, labels, strict=True)
        )
    )
    return replace(
        kerr_coefficients(KerrResponse(1.0, thg=responses[0].thg), build.grid),
        density=profile_table(susceptibility, 0.0, build.span, tol=build.tol),
    )


def _compile_dispersive_components(
    responses: tuple[DispersiveKerrResponse, ...],
    material: _MaterialProfiles,
    densities: Mapping[str, PiecewiseChebyshev],
    build: _ResponseBuild,
    *,
    regional: DispersiveKerrTerms | None = None,
) -> DispersiveKerrTerms | None:
    """Compile the dispersive responses, one channel per population.

    Each population keeps its own multipliers and density, from which a backend forms
    the evaluator's combination (:meth:`DispersiveKerrTerms.groups`): the symmetric
    mixture filter, rebuilt per stage, or the polynomial's exact per-species sum. A
    general modal guide's gas regions arrive compiled, as `regional`, and are taken as
    they are.

    Parameters
    ----------
    responses : tuple of DispersiveKerrResponse
        Declared responses in user order.
    material : _MaterialProfiles
        Ordered labels, species names and density paths.
    densities : Mapping of str to PiecewiseChebyshev
        Density tables in m^-3, keyed by component label.
    build : _ResponseBuild
        Grid, span in m, interpolation tolerance and ionisation field range.
    regional : DispersiveKerrTerms or None, optional
        A general modal guide's compiled gas channels, or ``None``.

    Returns
    -------
    DispersiveKerrTerms or None
        The compiled leaf, channels in species order, or ``None`` with none declared.

    Raises
    ------
    TranslationError
        If a dispersive response is declared beside a guide's own channels.
    """
    if regional is not None:
        if responses:
            raise TranslationError(
                "a general modal guide's dispersive channels are its regions' own, "
                "and a dispersive response was declared beside them"
            )
        return regional
    if not responses:
        return None
    labels = [_response_label(response, material) for response in responses]
    order = sorted(range(len(labels)), key=lambda i: material.labels.index(labels[i]))
    return dispersive_terms(
        [responses[i] for i in order],
        [labels[i] for i in order],
        [densities[labels[i]] for i in order],
        build.grid,
    )


def dispersive_regions(
    responses: Sequence[object], grid: TimeGrid, *, vector: bool
) -> tuple[ElectronicRegions, ...]:
    """Classify a run's active band for each dispersive response it declares.

    SEP 6, section 4.10: the active band, ``omega_min / 2`` to ``1.1 omega_max``,
    holds every output the grid can carry, and each part of it is supported, bounded
    or best effort for each gas. A run is never refused for the frequencies its field
    occupies; one whose band leaves a gas's support is logged once, at INFO, naming
    the gas and the band.

    Parameters
    ----------
    responses : sequence of object
        The run's declared responses; the dispersive ones are classified.
    grid : solprop.grid.TimeGrid
        The run's grid.
    vector : bool
        Whether the field at a node has two polarisation components, which takes the
        any-polarisation band edge.

    Returns
    -------
    tuple of solprop.physdata.nonlinear_data.ElectronicRegions
        One per dispersive response, in declaration order.
    """
    found = []
    for response in responses:
        if not isinstance(response, DispersiveKerrResponse):
            continue
        regions = response.model.regions(
            0.5 * grid.omega_min, grid.omega_max_window, vector=vector
        )
        if regions.beyond_support:
            logger.info(
                "dispersive electronic response, %s; the run is not refused for "
                "the frequencies its field occupies (SEP 6, section 4.10)",
                regions.describe(),
            )
        found.append(regions)
    return tuple(found)


def _compile_plasma_channels(
    responses: tuple[PlasmaResponse, ...],
    material: _MaterialProfiles,
    densities: Mapping[str, PiecewiseChebyshev],
    build: _ResponseBuild,
) -> tuple[PlasmaChannel, ...]:
    """Bind each exposure scan to its own density and response constants.

    Parameters
    ----------
    responses : tuple
        Declared responses of one kind in user order.
    material : _MaterialProfiles
        Ordered labels, species names and density paths.
    densities : Mapping of str to PiecewiseChebyshev
        Density tables in m^-3, keyed by component label.
    build : _ResponseBuild
        Grid, span in m, interpolation tolerance and ionisation field range.

    Returns
    -------
    tuple of PlasmaChannel
        Independent channels in mixture order.
    """
    channels = []
    for response in responses:
        label = _response_label(response, material)
        leaf = plasma_coefficients(
            response, build.field_range, z_max=build.span, margin=0.0, tol=build.tol
        )
        channels.append(
            PlasmaChannel(
                leaf.charge_mass_ratio,
                leaf.ionisation_potential,
                leaf.preionisation,
                leaf.rate,
                label,
                densities[label],
            )
        )
    if len({channel.label for channel in channels}) != len(channels):
        raise TranslationError("two PlasmaResponse responses on the same species")
    # Channel order is the material order, independent of response declaration order.
    return tuple(
        sorted(channels, key=lambda channel: material.labels.index(channel.label))
    )


def _plasma_integral(
    responses: tuple[PlasmaResponse, ...], grid: TimeGrid
) -> RunningIntegral:
    """Resolve the one running-integral rule of a problem's field plasma (SEP 18).

    The channels share the two current integrals, so they cannot honour different
    rules: every plasma response of one problem must ask for the same one.

    Parameters
    ----------
    responses : tuple of PlasmaResponse
        The field plasma responses, in user order.
    grid : solprop.grid.TimeGrid
        The field grid the band rule is designed for.

    Returns
    -------
    RunningIntegral
        The designed rule, or the trapezoid without plasma.

    Raises
    ------
    TranslationError
        If the responses ask for different rules, or no band design serves the grid.
    """
    if not responses:
        return RunningIntegral.trapezoid()
    policies = {response.integration for response in responses}
    if len(policies) > 1:
        named = ", ".join(
            f"{response.species or response.gas}: {response.integration!r}"
            for response in responses
        )
        msg = (
            "every plasma response of one problem must use the same integration, "
            f"because their currents are integrated together; got {named}"
        )
        raise TranslationError(msg)
    return design_running_integral(grid, responses[0].integration)


def _compile_envelope_channels(
    responses: tuple[EnvelopePlasmaResponse, ...],
    material: _MaterialProfiles,
    densities: Mapping[str, PiecewiseChebyshev],
    build: _ResponseBuild,
    shock: bool,
) -> EnvelopePlasmaAssembly | None:
    """Compile labelled envelope populations and their separate spectral assembly.

    Parameters
    ----------
    responses : tuple of EnvelopePlasmaResponse
        Declared envelope responses.
    material : _MaterialProfiles
        Component identities and paths.
    densities : Mapping
        Compiled component number densities in m^-3.
    build : _ResponseBuild
        Grid, z span, profile tolerance and intended amplitude domain.
    shock : bool
        Frequency-dependent or frozen-carrier policy.

    Returns
    -------
    EnvelopePlasmaAssembly or None
        Numeric channels in material order, or None when not declared.
    """
    if not responses:
        return None
    if not isinstance(build.grid, EnvelopeGrid):
        raise TranslationError("envelope plasma requires an EnvelopeGrid")
    channels = []
    for response in responses:
        label = _response_label(response, material)
        name = (
            response.gas.name
            if isinstance(response.gas, Species)
            else str(response.gas)
        )
        if material.names[material.labels.index(label)] not in (name, "medium"):
            raise TranslationError(
                f"envelope plasma {label!r} has inconsistent material species"
            )
        channels.append(
            EnvelopePlasmaChannel(
                label,
                name,
                response.potential,
                densities[label],
                profile_table(
                    _preionisation_profile(response.preionisation),
                    0.0,
                    build.span,
                    tol=build.tol,
                ),
                build_cycle_integrals(response, build.field_range),
            )
        )
    ordered = tuple(
        sorted(channels, key=lambda channel: material.labels.index(channel.label))
    )
    return EnvelopePlasmaAssembly(ordered, source_factors(build.grid, shock=shock))


def _raman_density(
    response: RamanResponse,
    label: str,
    material: _MaterialProfiles,
    densities: Mapping[str, PiecewiseChebyshev],
    build: _ResponseBuild,
) -> tuple[ZProfile, PiecewiseChebyshev]:
    """Return the density that damps a Raman leaf and the table that drives it.

    A component's own response is driven by its label's density and damped at the
    total density (the total-density rule). An optmat composition's response
    (:class:`solprop.nonlinear.optmat_raman.OptmatRamanResponse`) carries its
    constituents' mole fractions in its amplitudes, so the **summed** density of the
    labels that form it both drives and damps it: still one decay-rate path per term
    and one amplitude table per leaf, as the kernel contract has them.

    Parameters
    ----------
    response : RamanResponse
        The declared response.
    label : str
        The leaf's label.
    material : _MaterialProfiles
        Ordered labels and their density paths.
    densities : Mapping of str to PiecewiseChebyshev
        Density tables in m^-3, keyed by label.
    build : _ResponseBuild
        Span and tolerance of the tables.

    Returns
    -------
    tuple
        The density path in m^-3 the decay rates are composed with, and the
        amplitude density table in m^-3.

    Raises
    ------
    TranslationError
        If a composition names a label the material does not have.
    """
    labels = response.labels if isinstance(response, OptmatRamanResponse) else ()
    if len(labels) < 2:
        return material.total, densities[label]
    missing = [one for one in labels if one not in material.labels]
    if missing:
        msg = (
            f"the Raman composition names labels {missing} the material does not "
            f"have; expected a subset of {material.labels}"
        )
        raise TranslationError(msg)
    paths = tuple(material.paths[material.labels.index(one)] for one in labels)
    summed = Sum(paths)
    return summed, profile_table(summed, 0.0, build.span, tol=build.tol)


def _compile_raman_components(
    responses: tuple[RamanResponse, ...],
    kerr: KerrResponse | DispersiveKerrResponse | None,
    material: _MaterialProfiles,
    densities: Mapping[str, PiecewiseChebyshev],
    build: _ResponseBuild,
    n_pol: int,
) -> tuple[RamanCoefficients, ...]:
    """Compile component amplitudes and linewidths under the total-density rule.

    Parameters
    ----------
    responses : tuple
        Declared responses of one kind in user order.
    kerr : KerrResponse or DispersiveKerrResponse or None
        Electronic response specifying the field-grid harmonic form.
    material : _MaterialProfiles
        Ordered labels, species names and density paths.
    densities : Mapping of str to PiecewiseChebyshev
        Density tables in m^-3, keyed by component label.
    build : _ResponseBuild
        Grid, span in m, interpolation tolerance and ionisation field range.
    n_pol : int
        Number of effective polarisation components, one or two.

    Returns
    -------
    tuple of RamanCoefficients
        Component kernels, amplitude densities and total-density decay tables.
    """
    compiled = []
    for response in responses:
        label = _response_label(response, material)
        path, amplitude = _raman_density(response, label, material, densities, build)
        leaf = raman_coefficients(
            response,
            kerr,
            build.grid,
            path,
            n_pol=n_pol,
            span=build.span,
            tol=build.tol,
            undamped=None
            if build.raman_undamped is None
            else build.raman_undamped.get(label),
        )
        assert leaf is not None
        compiled.append(replace(leaf, label=label, density=amplitude))
    if len({term.label for term in compiled}) != len(compiled):
        raise TranslationError("two RamanResponse declarations on the same species")
    if len({term.thg for term in compiled}) > 1:
        raise TranslationError("Raman components must agree on thg")
    return tuple(sorted(compiled, key=lambda term: material.labels.index(term.label)))


def compile_responses(  # noqa: PLR0913 - the contract's section 5 has this many inputs
    responses: Sequence[object],
    grid: TimeGrid,
    mode: Mode | Sequence[Mode] | None,
    *,
    density: ZProfile | float,
    z_max: float,
    n_pol: int = 1,
    mode_averaged: bool = True,
    shock: bool = True,
    prefactor_index: float | None = None,
    ionisation_field_range: tuple[float, float] | None = None,
    absorber: Absorber | None = None,
    absorber_scheme: AbsorberScheme = DEFAULT_ABSORBER_SCHEME,
    mandatory_edges: Sequence[float] | np.ndarray = (),
    margin: float = DEFAULT_MARGIN,
    tol: float = DEFAULT_TOL,
    n_max: int = DEFAULT_MAX_DEGREE,
    max_intervals: int = DEFAULT_MAX_INTERVALS,
    representation: RepresentationChoice = "auto",
    translated_prefactor: PiecewiseChebyshev
    | ParameterExpansion
    | SeparableOperator
    | None = None,
    mixture: Mixture | None = None,
    regional_responses: RegionalResponseTables | None = None,
    regional_dispersive: DispersiveKerrTerms | None = None,
    reuse_prefactor: ParameterExpansion | None = None,
    reuse_raman: Mapping[str, Float[np.ndarray, "... n_to_doubled"]] | None = None,
) -> CompiledResponses:
    """Compile a set of responses into the nonlinear term a kernel executes.

    The instantaneous Kerr response, the delayed Raman response and the plasma response
    are compiled. A response of any other kind raises rather than being silently
    dropped, because a problem that reaches a backend missing a term it declared would
    run, converge and be wrong.

    A plasma response carries its own ionisation rate, so it supplies the run's
    `ionisation_table`; without one there is no rate table.

    A Raman response fills the three rows of section 5 from
    :func:`solprop.translate.raman.build_raman_kernels` and
    :func:`solprop.translate.raman.decay_rate_profiles`, and adds ``raman_drive`` to
    `response_order` after ``kerr``. It is refused on a two-component problem, which
    would need a tensor kernel no backend can consume, and on a
    field grid it is refused without a Kerr response beside it, or beside one that
    disagrees about ``thg``: the drive form is a property of the problem and reaches a
    backend through the Kerr row.

    Parameters
    ----------
    responses : sequence
        The declared responses: at most one
        :class:`solprop.nonlinear.responses.KerrResponse`, at most one
        :class:`solprop.nonlinear.raman.RamanResponse` and at most one
        :class:`solprop.nonlinear.responses.PlasmaResponse`, each defined on this grid.
    grid : solprop.grid.TimeGrid
        The grid; decides the cubic form and supplies the crop scale.
    mode : Mode or sequence of Mode
        The mode, or the ordered set of a multimode problem, whose effective index and
        effective area enter the prefactor.
    density : solprop.profiles.ZProfile or float
        Number density along z in m^-3, or 1.0 for a medium whose ``chi3`` is absolute.
    z_max : float
        Length the run covers, m, positive and finite.
    n_pol : int, optional
        Polarisation components, 1 or 2.
    mode_averaged : bool, optional
        Whether to use the mode-averaged prefactor, which folds in the effective-area
        conversion. Default ``True``.
    shock : bool, optional
        Keep the ``omega`` dependence of the prefactor's numerator, which *is* the
        optical shock term. Default ``True``.
    prefactor_index : float or None, optional
        Replace the effective index in the prefactor by this constant. The GNLSE passes
        1.0.
    ionisation_field_range : tuple of float or None, optional
        Lower and upper field amplitude in V/m a plasma response's rate table must
        cover; refused without a plasma response. Refused when the rate handed over
        is already a
        :class:`solprop.translate.ionisation.RateTable`, which states its own span.
    absorber : solprop.boundaries.Absorber or None, optional
        The collar rates; only its temporal part is used here, the spectral part being
        already inside the linear operator.
    absorber_scheme : {'split_step', 'nonlinear'}, optional
        Which application of the temporal rate the backend will use.
    mandatory_edges : array_like, optional
        Positions in z that must be interval edges of the prefactor table.
        :func:`solprop.translate.problem.assemble` passes the cumulative operator's
        ``hard_kinks``, which is the union of the mode set's declared breakpoints, so
        that the two tables split on the same edges; a standalone caller should do the
        same, because a table left to locate a kink by bisection spends intervals on it
        and is less accurate there.
    margin : float, optional
        Fraction of `z_max` the density and prefactor tables extend beyond the end.
    tol : float, optional
        Target bound on the uniform error of the tables.
    n_max : int, optional
        Highest Chebyshev degree tried on an interval of the prefactor table before it
        is bisected.
    max_intervals : int, optional
        Interval budget of the prefactor's tables; exceeding it warns.
    representation : {'auto', 'expansion', 'table'}, optional
        Which representation to build for the prefactor, as
        :func:`solprop.translate.linop.build_phi` takes for ``Phi``, and chosen the same
        way. A z-constant mode set takes the constant fast path whatever this says.
    translated_prefactor : object, optional
        An already translated normalisation the caller owns, including the temporal
        crop scaling: spatial geometry's, or the general modal route's constant crop
        scale, whose ``-i omega`` lives in its own record instead. A
        PiecewiseChebyshev, ParameterExpansion or SeparableOperator record. Requires
        ``mode=None`` and ``mode_averaged=False``. The material responses still use
        the same compiler and validation as modal propagation.

    mixture : Mixture or None, optional
        Explicit species and component paths for a spatial material. Modal assembly
        also discovers the mixture carried by its modes. None keeps the legacy path.
    regional_responses : RegionalResponseTables or None, optional
        The compiled per-region response of a general modal guide, carried through
        to contract section 5's optional leaf unchanged. It is not compiled here:
        the general route's responses belong to the guide's material regions and
        are built by :func:`regional_responses`.
    regional_dispersive : DispersiveKerrTerms or None, optional
        The compiled dispersive channels of a general modal guide's gas regions
        (SEP 6), carried through to ``dispersive_kerr`` unchanged, beside
        `regional_responses`; they are not compiled here either.
    reuse_prefactor : ParameterExpansion or None, optional
        Coefficients of the unchanged prefactor function on the same grid and fixed
        parameters; range checks decide whether its terms may be shared.
    reuse_raman : Mapping or None, optional
        Labelled undamped rows for unchanged Raman models and time grid. Densities
        and decay-rate tables are rebuilt.

    Returns
    -------
    CompiledResponses
        Section 5 of the kernel contract.

    Raises
    ------
    TranslationError
        If a response cannot be compiled, more than one response of a kind is given, a
        Raman response arrives on a problem this layer cannot compile it for, a field
        range arrives without a plasma response, or an argument is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.translate.responses import compile_responses
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> problem = problem_from([0.0, 0.0, -1e-26], omega_ref=grid.omega0, gamma=0.1,
    ...                        shock=False)
    >>> compiled = compile_responses([problem.kerr], grid, problem.mode, density=1.0,
    ...                              z_max=1.0, shock=False, prefactor_index=1.0)
    >>> compiled.prefactor.value_shape, compiled.prefactor_constant.shape
    ((1, 1, 4096), (1, 1, 4096))
    >>> bool(np.all(compiled.prefactor_constant[0, 0][~grid.active_mask] == 0.0))
    True
    """
    if not (math.isfinite(z_max) and z_max > 0.0):
        msg = f"z_max must be positive and finite, got {z_max!r}"
        raise TranslationError(msg)
    if (mode is None) != (translated_prefactor is not None):
        msg = "give modal modes or a translated_prefactor with mode=None"
        raise TranslationError(msg)
    if translated_prefactor is not None and mode_averaged:
        msg = "translated_prefactor requires mode_averaged=False"
        raise TranslationError(msg)
    modes = (
        () if mode is None else tuple(mode) if isinstance(mode, Sequence) else (mode,)
    )
    if not modes and translated_prefactor is None:
        msg = "a problem needs at least one mode"
        raise TranslationError(msg)
    kerr, plasma, raman, envelope, dispersive = _declared(responses, grid)
    electronic = (*kerr, *dispersive)
    if len({response.thg for response in electronic}) > 1:
        raise TranslationError(
            "the electronic responses must agree on thg: one problem has one "
            "harmonic sector"
        )
    profile = as_profile(density, (0.0, math.inf))
    span = z_max * (1.0 + margin)
    material = _material_profiles(modes, mixture, profile)
    total_density = profile_table(profile, 0.0, span, tol=tol)
    density_tables = tuple(
        total_density if path is profile else profile_table(path, 0.0, span, tol=tol)
        for path in material.paths
    )
    densities = dict(zip(material.labels, density_tables, strict=True))
    controls = _ResponseBuild(grid, span, tol, ionisation_field_range, reuse_raman)
    _check_population_declarations(envelope, kerr)
    compiled_plasma = _compile_plasma_channels(plasma, material, densities, controls)
    compiled_envelope = _compile_envelope_channels(
        envelope, material, densities, controls, shock
    )
    tables = _population_tables(
        compiled_plasma, compiled_envelope, ionisation_field_range
    )
    compiled_kerr = _compile_kerr_components(kerr, material, densities, controls)
    compiled_dispersive = _compile_dispersive_components(
        dispersive, material, densities, controls, regional=regional_dispersive
    )
    compiled_raman = _compile_raman_components(
        raman,
        electronic[0] if electronic else None,
        material,
        densities,
        controls,
        n_pol,
    )
    coarse, fine = temporal_rates(absorber, scheme=absorber_scheme)
    supplied = _translated_prefactor(translated_prefactor)
    prefactor_at = (
        None
        if supplied is not None
        else _prefactor_at(modes, grid, n_pol, mode_averaged, shock, prefactor_index)
    )
    constant = supplied is None and all(one.z_constant for one in modes)
    expansion = None
    if (
        supplied is None
        and representation != "table"
        and reuse_prefactor is not None
        and modes
        and isinstance(modes[0], ParameterisedMode)
        and parameters_own_z(modes)
    ):
        expansion = reweight_expansion(
            reuse_prefactor, modes[0].parameter_paths, tol=tol, kinks=mandatory_edges
        )
    if (
        supplied is None
        and representation != "table"
        and not constant
        and expansion is None
    ):
        expansion = _prefactor_expansion(
            modes,
            grid,
            n_pol,
            mode_averaged,
            shock,
            prefactor_index,
            span=span,
            edges=mandatory_edges,
            tol=tol,
            n_max=n_max,
            max_intervals=max_intervals,
        )
        if expansion is None and representation == "expansion":
            msg = (
                "an expansion in the scalar parameters was asked for and the prefactor "
                f"could not be built as one over [0, {span:.6g}] m; the debug log says "
                "why, and the z table is the general representation"
            )
            raise TranslationError(msg)
    constant = constant and expansion is None
    prefactor = (
        None
        if supplied is not None or expansion is not None
        else _prefactor_table(
            cast(
                "Callable[[float], Complex[np.ndarray, 'n_modes n_pol n_omega']]",
                prefactor_at,
            ),
            modes,
            span=span,
            # A constant prefactor is one interval of degree zero, whatever else kinks
            # along z: the fast path's own contract. A response's kinks
            # split Phi, not a table that does not vary.
            edges=() if constant else mandatory_edges,
            tol=tol,
            n_max=n_max,
            max_intervals=max_intervals,
        )
    )
    compiled = CompiledResponses(
        response_order=_response_order(
            bool(kerr),
            bool(tables) and not envelope,
            bool(compiled_plasma),
            bool(raman),
            has_dispersive=compiled_dispersive is not None,
        )
        if not envelope
        else (
            ResponseStage.ENVELOPE_CYCLE,
            ResponseStage.ENVELOPE_POPULATION,
            ResponseStage.ENVELOPE_SOURCES,
            *_response_order(
                bool(kerr),
                False,
                False,
                bool(raman),
                has_dispersive=compiled_dispersive is not None,
            ),
        ),
        kerr_coefficients=compiled_kerr,
        ionisation_table=tables,
        plasma_channels=compiled_plasma,
        plasma_integral=_plasma_integral(plasma, grid),
        raman_terms=compiled_raman,
        density=total_density,
        labels=material.labels,
        species_names=material.names,
        species_densities=density_tables,
        spectral_responses=() if compiled_envelope is None else (compiled_envelope,),
        envelope_plasma=compiled_envelope,
        prefactor_representation=_prefactor_kind(constant, expansion is not None)
        if supplied is None
        else supplied.kind,
        prefactor=prefactor if supplied is None else supplied.table,
        prefactor_expansion=expansion if supplied is None else supplied.expansion,
        prefactor_constant=(
            supplied.constant
            if supplied is not None
            else prefactor_at(0.0)
            if constant and prefactor_at is not None
            else None
        ),
        temporal_absorber=coarse,
        temporal_absorber_fine=fine,
        absorber_scheme=absorber_scheme,
        separable=None if supplied is None else supplied.separable,
        regional_responses=regional_responses,
        dispersive_kerr=compiled_dispersive,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "responses: order %s, %d modes x %d polarisations, ionisation table %s, "
            "Raman %s, temporal absorber %s",
            [stage.value for stage in compiled.response_order],
            compiled.prefactor_value_shape[0],
            n_pol,
            "absent"
            if not tables
            else f"{sum(table.n_knots for table in tables)} knots",
            "absent"
            if not compiled_raman
            else f"{sum(term.n_terms for term in compiled_raman)} term(s)",
            "absent" if coarse is None else absorber_scheme,
        )
        _log_prefactor(compiled, z_max)
    return compiled


def _check_population_declarations(
    envelope: tuple[EnvelopePlasmaResponse, ...],
    kerr: tuple[KerrResponse, ...],
) -> None:
    """Reject contradictory source declarations before expensive compilation.

    Parameters
    ----------
    envelope, kerr : tuple
        Declared responses of each kind.
    """
    if envelope and any(response.thg for response in kerr):
        raise TranslationError("envelope plasma cannot be combined with envelope THG")


def _population_tables(
    plasma: tuple[PlasmaChannel, ...],
    envelope: EnvelopePlasmaAssembly | None,
    field_range: tuple[float, float] | None,
) -> tuple[RateTable, ...]:
    """Retain the exact rate table owned by each active population channel.

    Parameters
    ----------
    plasma : tuple of PlasmaChannel
        Compiled field populations.
    envelope : EnvelopePlasmaAssembly or None
        Compiled envelope populations and source assembly.
    field_range : tuple of float or None
        Intended table amplitude domain in V/m.

    Returns
    -------
    tuple of RateTable
        Tables in channel order; empty without a plasma population.

    Raises
    ------
    TranslationError
        If a field range is given without a plasma population to use it.
    """
    if envelope is not None:
        return tuple(channel.rate for channel in envelope.channels)
    if plasma:
        return tuple(channel.rate for channel in plasma)
    if field_range is not None:
        msg = (
            "ionisation_field_range was given without a plasma response; there is "
            "no rate to tabulate over it"
        )
        raise TranslationError(msg)
    return ()


def _declared(
    responses: Sequence[object], grid: TimeGrid
) -> tuple[
    tuple[KerrResponse, ...],
    tuple[PlasmaResponse, ...],
    tuple[RamanResponse, ...],
    tuple[EnvelopePlasmaResponse, ...],
    tuple[DispersiveKerrResponse, ...],
]:
    """Sort a declared set into the responses the translation layer compiles.

    A response that cannot be compiled is refused by name rather than dropped, and one
    that declares it is undefined on this grid is refused with that reason -- the
    plasma model is written for a resolved carrier and has no envelope form.

    Parameters
    ----------
    responses : sequence
        The declared responses.
    grid : solprop.grid.TimeGrid
        The grid, whose ``kind`` every response must support.

    Returns
    -------
    tuple
        The Kerr, plasma, Raman, envelope-plasma and dispersive Kerr responses, each
        empty when not declared.

    Raises
    ------
    TranslationError
        Naming a response that cannot be compiled, one declared twice, two electronic
        responses on one population, or one that is not defined on this grid.
    """
    found: dict[tuple[type, str | None], object] = {}
    for response in responses:
        if not isinstance(
            response,
            KerrResponse
            | PlasmaResponse
            | EnvelopePlasmaResponse
            | RamanResponse
            | DispersiveKerrResponse,
        ):
            traits = getattr(response, "traits", None)
            detail = (
                ""
                if traits is None
                else f" (domain {traits.domain!r}, {traits.polarisation!r})"
            )
            msg = (
                f"{type(response).__name__} is not compilable by the translation layer "
                f"yet{detail}; the instantaneous and the dispersive Kerr, the Raman "
                "and the plasma responses are"
            )
            raise TranslationError(msg)
        # A population has one electronic response: the constant and the dispersive
        # cubic of one gas share the key, so the pair is refused as a duplicate.
        kind = (
            KerrResponse
            if isinstance(response, DispersiveKerrResponse)
            else type(response)
        )
        label = response.species
        if label is None and isinstance(
            response, PlasmaResponse | EnvelopePlasmaResponse
        ):
            label = (
                response.gas.name
                if isinstance(response.gas, Species)
                else str(response.gas)
            )
        key = (kind, label)
        if key in found:
            population = "" if label is None else f" on the same species {label!r}"
            if type(found[key]) is not type(response):
                msg = (
                    f"two electronic responses{population} were declared "
                    f"({type(found[key]).__name__} and {type(response).__name__}); a "
                    "population has one electronic third-order response, constant or "
                    "dispersive"
                )
                raise TranslationError(msg)
            msg = (
                f"two {type(response).__name__} responses{population} were declared; "
                "a fused pass evaluates "
                "each term once, so combine them into one response instead"
            )
            raise TranslationError(msg)
        if grid.kind not in response.traits.grids:
            msg = (
                f"{type(response).__name__} is defined on the "
                f"{', '.join(response.traits.grids)} grid and this problem is on the "
                f"{grid.kind} grid; the response has to be re-derived for it rather "
                "than adapted"
            )
            raise TranslationError(msg)
        found[key] = response
    return (
        tuple(value for value in found.values() if isinstance(value, KerrResponse)),
        tuple(value for value in found.values() if isinstance(value, PlasmaResponse)),
        tuple(value for value in found.values() if isinstance(value, RamanResponse)),
        tuple(
            value
            for value in found.values()
            if isinstance(value, EnvelopePlasmaResponse)
        ),
        tuple(
            value
            for value in found.values()
            if isinstance(value, DispersiveKerrResponse)
        ),
    )


def _tabulate_rate(rate: IonisationRate, field_range: tuple[float, float]) -> RateTable:
    """Build a frozen spline table of an ionisation rate over a field range.

    Parameters
    ----------
    rate : solprop.nonlinear.ionisation.IonisationRate
        The rate to tabulate.
    field_range : tuple of float
        Lower and upper field amplitude in V/m the table must cover.

    Returns
    -------
    solprop.translate.ionisation.RateTable
        The converged table, with read-only arrays.
    """
    return _frozen_table(build_rate_table(rate, field_range))


def _frozen_table(table: RateTable) -> RateTable:
    """Return the same table with read-only arrays.

    The builder memoises its tables and hands out the originals, so a table that
    crosses the seam is a frozen copy: the contract promises read-only arrays and
    ``validate`` checks it.

    Parameters
    ----------
    table : solprop.translate.ionisation.RateTable
        The table to freeze.

    Returns
    -------
    solprop.translate.ionisation.RateTable
        The same values, not writeable.
    """
    if (
        not table.log_rate.flags.writeable
        and not table.coefficients.flags.writeable
        and table.log_rate.flags.c_contiguous
        and table.coefficients.flags.c_contiguous
    ):
        return table
    return replace(
        table,
        log_rate=_readonly(table.log_rate),
        coefficients=_readonly(table.coefficients),
    )


def _readonly(values: np.ndarray) -> np.ndarray:
    """Return a C-contiguous, read-only copy of an array.

    Parameters
    ----------
    values : numpy.ndarray
        The array to freeze.

    Returns
    -------
    numpy.ndarray
        The same values, C-contiguous and not writeable.
    """
    frozen = np.ascontiguousarray(values)
    frozen.setflags(write=False)
    return frozen


def _response_order(
    has_kerr: bool,
    has_ionisation: bool,
    has_plasma: bool,
    has_raman: bool,
    *,
    has_dispersive: bool = False,
) -> tuple[ResponseStage, ...]:
    """Return the fused-pass sequence the declared content implies.

    Parameters
    ----------
    has_kerr : bool
        Whether a Kerr response was declared.
    has_ionisation : bool
        Whether an ionisation table was built.
    has_plasma : bool
        Whether a plasma response was compiled.
    has_raman : bool
        Whether a Raman response was compiled.
    has_dispersive : bool, optional
        Whether a dispersive Kerr response was compiled; its stage follows the
        constant one's.

    Returns
    -------
    tuple of ResponseStage
        The stages in execution order, always ending in ``ACCUMULATE``.
    """
    stages: list[ResponseStage] = []
    if has_ionisation:
        stages += [ResponseStage.INTENSITY, ResponseStage.IONISATION_RATE]
    if has_plasma:
        stages += [ResponseStage.PLASMA_SCAN, ResponseStage.PLASMA_CURRENT]
    if has_kerr:
        stages.append(ResponseStage.KERR)
    if has_dispersive:
        stages.append(ResponseStage.DISPERSIVE_KERR)
    if has_raman:
        stages.append(ResponseStage.RAMAN_DRIVE)
    stages.append(ResponseStage.ACCUMULATE)
    return tuple(stages)


def raman_coefficients(  # noqa: PLR0913 - the leaf needs the grid, the path and the build
    raman: RamanResponse | None,
    kerr: KerrResponse | DispersiveKerrResponse | None,
    grid: TimeGrid,
    density: ZProfile,
    *,
    n_pol: int,
    span: float,
    tol: float,
    undamped: Float[np.ndarray, "... n_to_doubled"] | None = None,
) -> RamanCoefficients | None:
    """Compile a Raman response into the leaf of section 5, or ``None``.

    The expensive half of the kernel is sampled once per grid and the cheap half becomes
    one scalar table along z per term, so that what crosses the seam is arrays and paths
    and no physics object (``docs/guide/media/raman-numerics.md``, "Sampling the
    kernel").

    The fraction is :attr:`solprop.nonlinear.raman.RamanResponse.scale` and not
    ``fraction`` itself, so that an absolute molecular response carries the 1 the
    contract's row promises rather than a ``None`` the other two rows would then have to
    match.

    Parameters
    ----------
    raman : solprop.nonlinear.raman.RamanResponse or None
        The declared response.
    kerr : KerrResponse or DispersiveKerrResponse or None
        The electronic response declared beside it, constant or dispersive, which
        fixes the field grid's drive form.
    grid : solprop.grid.TimeGrid
        The grid supplying the fine time axis and the temporal collar.
    density : solprop.profiles.ZProfile
        The number density along z, which the decay rates are composed with.
    n_pol : int
        Polarisation components of the state.
    span : float
        End of the tabulated range, m: the same ``z_max (1 + margin)`` as ``Phi``.
    tol : float
        Target bound on the uniform error of the decay-rate tables.
    undamped : numpy.ndarray or None, optional
        Undamped kernel rows to share for an identical model and fine time grid.

    Returns
    -------
    RamanCoefficients or None
        The compiled coefficients, or ``None`` when no Raman response was declared.

    Raises
    ------
    TranslationError
        If two physical components request a scalar-only declaration, or a field
        grid has no compatible Kerr response to establish the drive form.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.raman import response_for
    >>> from solprop.profiles import Constant
    >>> from solprop.translate.responses import raman_coefficients
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> response = response_for("SiO2", amplitude=1e-22, fraction=0.18, thg=False)
    >>> leaf = raman_coefficients(
    ...     response, None, grid, Constant(1.0), n_pol=1, span=1.0, tol=1e-12
    ... )
    >>> leaf.n_terms, leaf.thg
    (1, False)
    """
    if raman is None:
        return None
    if n_pol > 1 and not raman.tensor_capable:
        named = [
            str(getattr(term.model, "name", type(term.model).__name__))
            for term in raman.tensor_terms
            if term.contraction == "scalar_only"
        ]
        msg = (
            f"a Raman response was declared on a problem with {n_pol} polarisation "
            f"components but contains scalar_only terms ({', '.join(named)}). "
            "Declare explicit tensor terms with an established physical contraction "
            "for every model; a copolarised-only silica shape, such as optmat's "
            "default bundle, has none, and a vector run takes the tensor bundle"
        )
        raise TranslationError(msg)
    if grid.kind == "field":
        if kerr is None:
            msg = (
                "a field-grid Raman response needs a Kerr response beside it: whether "
                "the drive keeps the 2 omega beat is a property of the problem and "
                "reaches a backend through the Kerr term's own THG flag, and a medium "
                "with a Raman response has an electronic one too"
            )
            raise TranslationError(msg)
        if kerr.thg != raman.thg:
            msg = (
                f"the Kerr response has thg={kerr.thg} and the Raman response "
                f"thg={raman.thg}; on a field grid the two drives must agree, because "
                "a drive that keeps the 2 omega beat radiates a third harmonic through "
                "the later multiplication by the field, and a grid built without one "
                "does not oversample enough to hold it"
            )
            raise TranslationError(msg)
    samples = (
        build_raman_kernels(raman, grid).undamped if undamped is None else undamped
    )
    tables = tuple(
        profile_table(path, 0.0, span, tol=tol)
        for path in decay_rate_profiles(raman, density, grid)
    )
    weights = np.array(
        [
            (0.25, 0.75) if term.contraction == "rank2" else (1.0, 0.0)
            for term in raman.tensor_terms
        ],
        dtype=np.float64,
    )
    return RamanCoefficients(
        _readonly(samples),
        tables,
        raman.scale,
        thg=raman.thg,
        tensor_weights=weights,
        tensor_capable=raman.tensor_capable,
    )
