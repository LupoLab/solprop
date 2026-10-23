# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The general modal route: the seam leaves of one guide, and their certificates.

A general modal guide reaches a backend as three of the four optional leaves kernel
schema 12 added -- the E/H reference and its execution
maps (``transform.modal_tables``), the basis metric (``linear.modal_metric``) and the
per-region response (``responses.regional_responses``) -- plus the ordinary diagonal
linear operator every modal route has. :class:`GeneralModalRoute` is what
:func:`solprop.translate.problem.assemble` is handed: the leaves that were built
together, with the certificate that binds them.

The fourth leaf, ``linear.coupled_flow``, is filled for exactly one case: a
**declared unitary basis of a split block**,
where ``U.H diag(beta) U`` is not diagonal and the channels genuinely beat. The
route carries the transformed generator relative to one reference channel
(:func:`coupled_block`) and :func:`solprop.translate.problem.assemble` turns it into
the leaf with its certificate, because the largest step the flow's conditioning is
declared over is the assembly's to know. A guide in the provider's own coordinates,
and a declared basis of an exactly degenerate block, leave it ``None`` and propagate
through the diagonal flow alone.
"""

import dataclasses
import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from jaxtyping import Bool, Complex, Float

from solprop import fourier
from solprop._display import Displayed, line
from solprop.constants import C
from solprop.exceptions import (
    GeneralModalBasisWarning,
    MaterialError,
    TranslationError,
)
from solprop.grid import TimeGrid
from solprop.modes.guide import ModalGuide
from solprop.modes.material import (
    GasKerr,
    JsonValue,
    MaterialRegion,
    quantised_measurement,
)
from solprop.modes.protocol import ChannelLabel
from solprop.nonlinear.mixtures import electronic_response
from solprop.nonlinear.raman import SilicaTensorResponse
from solprop.nonlinear.responses import DispersiveKerrResponse, KerrResponse
from solprop.physdata.glasses import SilicaBundle
from solprop.physdata.raman_types import LinAgrawalRaman
from solprop.physdata.selection import MaterialSelection
from solprop.profiles import Constant
from solprop.stats import GENERAL_MODAL_ACCESSORS, WITHHELD_WITH_A_FLOW
from solprop.translate.coupled import (
    CoupledFlowTables,
    ModalMetricTables,
    dense_metric,
    orthonormal_metric,
)
from solprop.translate.linop import DEFAULT_MARGIN, profile_table
from solprop.translate.modal_tables import (
    DEFAULT_GRAM_ATOL,
    DenseModalTable,
    ModalTransformTables,
    modal_transform_tables,
    sample_modal_table,
)
from solprop.translate.responses import (
    DispersiveKerrTerms,
    KerrForm,
    RegionalResponseTables,
    dispersive_terms,
    regional_responses,
)

__all__ = [
    "MAX_DENSE_METRIC_DEPARTURE",
    "N2_CONVERSION_RTOL",
    "GeneralModalRoute",
    "check_noise_basis",
    "coupled_block",
    "general_modal_route",
    "guide_identity",
    "interpolated_gram",
    "metric_departure",
    "modal_identity",
    "occupancy_weight",
]

logger = logging.getLogger(__name__)

MAX_DENSE_METRIC_DEPARTURE: Final[float] = 1e-3
"""Largest ``norm(W(omega) - I, 2)`` a dense metric may show and still be executed.

Dimensionless. The general route propagates through the **diagonal** cumulative flow
and measures the step controller's norm on the coordinate amplitudes, both of which
are the basis's own quantities only where that basis is orthonormal in true power. A
declared unitary change of basis of a certified reference stays there -- the spectral
norm is unitarily invariant, so ``norm(U.H W U - I, 2) = norm(W - I, 2)`` exactly --
and a nonunitary transform does not.

**The number.** The quantity is measured on the run's own coarse grid, **between**
the table's knots, where an interpolated basis is further from orthonormal than at
the knots :data:`solprop.translate.modal_tables.DEFAULT_GRAM_ATOL` gates: the weakly
guiding fixture of ``tests/driver/test_general_modal.py``, whose knot defect is
6.97e-8, reaches 1.136e-4 there at 13 knots -- three orders apart. So this budget
cannot be the knot budget.

**What sets the margin is the departure over the *occupied* support**, and it is
smaller than the headline. The interpolated defect lives at the **long-wavelength
end** of the sampled band, far outside where a run puts any light: on that fixture
the worst bins are 1.729 um at 1.136e-4 and 1.685 um at 9.90e-5, while inside the
790--890 nm launch window the departure is 2.32e-7, so the latent error in a
launched 835 nm 40 fs pulse's ``physical_energy`` is 3.70e-8 weighted by its own
spectrum. That occupied quantity is what
:attr:`GeneralModalRoute.occupied_departure` measures and what `gram_atol` gates,
on the identity route as much as on this one; this
constant stays the **whole-band** budget, because what it exists to catch -- a
nonunitary transform -- is nonunitary in every bin and not only where the light
is. The headline number is also sampling-dependent rather than a property of
the provider: at that fixture builder's default nine knots the same block measures
4.03e-4, only 2.5 times under this budget, and the 1.5 um strand fixture measures
2.46e-5 at nine. So 1e-3 is **not** comfortably an order above the loosest
departure a provider here produces; it is a factor of a few above a poorly sampled
one and three orders below the order-one departure a nonunitary transform produces,
which is the departure it exists to catch. It is the loosest of the three gates a
declared basis passes -- the declaration's own 1e-12 unitarity budget
(:data:`solprop.modes.guide.BASIS_UNITARY_ATOL`), which the per-knot invariance
check of :func:`solprop.translate.modal_tables.apply_channel_basis` takes as its own
floor, is the tight one -- and it is the one a **backend** can apply, since it sees
the stored block and not the declaration.
"""

N2_CONVERSION_RTOL: Final[float] = 0.01
"""Spread of a converted ``chi3`` this route will accept at an undeclared frequency.

Relative and dimensionless: the largest ``max/min - 1`` of the executed ``chi3``
across the run's active band that this route will convert at a frequency nobody
declared.

A material region's compiled response carries **one** susceptibility, so a declared
``n2`` becomes a ``chi3`` at one frequency (``chi3 = 4 eps0 c n_0^2 n2 / 3``) and the
choice of frequency multiplies it by the square of an index ratio.

**One per cent is a policy number, not a derived one.** It is chosen to sit
comfortably below the case that motivated it -- fused silica
spans **2.13 %** of its own ``chi3`` between 800 nm and 2 um -- with enough margin
that a material like it is caught whatever band a run declares, while a noble gas
below a few bar, which spans less than 0.1 % across any band, is never asked. The
asymmetry that makes a low threshold cheap is that saying where costs the caller one
keyword, ``MaterialKerr(n2=..., at=<m>)``, while an undeclared conversion is
unrecoverable: the leaf carries one number and nothing downstream can say which
frequency it came from, so the conversion is refused rather than guessed.
"""


@dataclass(frozen=True, eq=False)
class GeneralModalRoute(Displayed):
    """Own the leaves of one general modal guide and the certificate that binds them.

    The record a run is assembled from. It is not itself a kernel-contract field:
    :func:`solprop.translate.problem.assemble` takes it apart into the three leaves
    of the groups they belong to, and the metric it derives from the declared Gram
    defect rather than storing a second copy of the same certificate.

    **The metric is the identity unless a basis is declared.** A guide built in the
    provider's own coordinates is a Poynting-normalised orthogonal basis whose
    sampled Gram passed the construction gate of
    :class:`solprop.translate.modal_tables.DenseModalTable`, so ``W = I`` to within
    `gram_defect` and the route declares the identity. A guide that declares a
    unitary change of basis (:class:`solprop.modes.guide.ChannelBasis`) carries the
    **dense** metric `metric_tables` instead: ``W' = U.H W U`` is the identity in
    exact arithmetic and is not the identity in the measured Gram the run actually
    has, so it is stored and declared rather than decorated. Either way the
    statistics power weight is the constant one: the Poynting amplitude **is** the
    true-power amplitude and the legacy ``sqrt(Re n_eff)`` boundary is not crossed.
    The two leaves are different objects and both exist: ``linear.power_weight`` is
    the statistics boundary and ``linear.modal_metric`` is the basis Gram.

    Attributes
    ----------
    tables : solprop.translate.modal_tables.ModalTransformTables
        The E/H reference and the two execution maps, tabulated on the run's own
        canonical grids and zero outside the guide's support.
    responses : solprop.translate.responses.RegionalResponseTables or None
        The compiled response of each material region, or ``None`` for a guide
        whose materials are all linear -- a propagation with no source at all,
        which is the linear reference case.
    gram_defect : float
        The largest measured ``norm(W - I, 2)`` of the basis the tables were
        sampled from, dimensionless and non-negative; the table's own
        ``gram_defect``. It is the error row of the identity metric.
    certificate_id : str
        Non-empty identity of the certificate `gram_defect` came from.
    occupied_departure : float
        The **interpolated** ``norm(W - I, 2)`` over the bins this run occupies,
        dimensionless and non-negative. `gram_defect` is measured at the table's
        knots; this one is measured at the run's own coarse bins, **between** them,
        where a sampled basis is further from orthonormal -- three orders further
        on the weakly guiding fixture (6.97e-8 against 1.14e-4). Declaring
        ``W = I`` costs a run that much in
        :func:`solprop.stats.physical_energy` to first order, so it is gated
        against `gram_atol` like the knot defect. Where the light is, not where
        the knots are sparsest: see :func:`occupancy_weight`.
        Zero on a record assembled by hand, which measures nothing and claims
        nothing.
    active_departure : float
        The same interpolated ``norm(W - I, 2)``, maximised over **every** bin
        ``grid.active_mask`` admits and weighted by nothing, dimensionless and
        non-negative. `occupied_departure` discounts a bin by how little of the
        launched spectrum sits in it, which certifies the launch; this one
        certifies the run, because a broadening run can reach any active bin.
        It is **warned** on rather than gated
        (:class:`solprop.exceptions.GeneralModalBasisWarning`), because a refusal
        would drop guides whose light never goes there and only the caller knows
        whether it will. Zero on a record assembled by hand.
    gram_atol : float
        Largest defect this route will call the identity, dimensionless and
        positive; :data:`solprop.translate.modal_tables.DEFAULT_GRAM_ATOL` by
        default, which is the budget
        :class:`~solprop.translate.modal_tables.DenseModalTable` gates its own
        sampling with. A defect above it is refused here rather than propagated: the
        route **is** the statement ``W = I``, and on a continuous wave the resulting
        error in ``da/dz`` is ``|1/W_00 - 1|``, which is the defect itself to first
        order. A certificate that is merely recorded and never gated is an unbounded
        silent error.
    guide_identity : collections.abc.Mapping or None
        What the guide this route was built from *is*, as :func:`guide_identity`
        writes it: the provider's own identity, the sampling the reference is
        rebuilt from, and each material region with its declared Kerr response. It
        is carried here because the route is the last object that sees the guide --
        :func:`solprop.translate.problem.assemble` takes the leaves apart and the
        guide never reaches a
        :class:`~solprop.translate.problem.KernelProblem` -- and output schema 7
        persists it inside :attr:`solprop.output.schema.RunMetadata.modal`.
        ``None`` for a record built by hand, which :func:`modal_identity` then
        refuses by name rather than writing a file that cannot say what produced it.
    coupled_generator : numpy.ndarray or None
        The internal-convention generator of a declared **split** block, relative to
        `reference_channel`, in 1/m, complex, shape
        ``(n_modes_coupled_out, n_modes_coupled_in, n_omega_coupled)``, exactly zero
        outside the run's active band (:func:`coupled_block`). ``None`` -- the
        ordinary case -- whenever the generator is diagonal, and then the assembled
        problem carries no ``linear.coupled_flow`` and propagates through the
        diagonal flow alone. It is **not** the leaf: the leaf's conditioning
        certificate is declared over the largest step the run may take, which this
        record does not know.
    reference_channel : int
        Which channel the cumulative ``Phi`` beside a coupled block is built from,
        and therefore what `coupled_generator` is measured relative to. Zero, the
        first channel, as the coherent wall's is; a coupled block needs a ``Phi``
        that is a multiple of the identity in the channel axis, so one channel's
        dispersion is carried there and every channel's departure from it in the
        block.
    moving_generator, moving_connection : numpy.ndarray or None
        A **moving** guide's flow: the Chebyshev coefficients
        along z, on the tables' own sections, of the internal-convention generator
        less the reference channel's **entrance** diagonal, and of the connection
        ``Gamma = W^-1 S`` in the same amplitude, both in 1/m, complex, shape
        ``(n_intervals, n_deg, n_modes, n_modes, n_omega)``, zero outside the active
        band. Supplied together or not at all; ``None`` for a snapshot, a guide
        that does not move.
    moving_error : float
        The relative Chebyshev tail of those expansions, or the path's held-out
        error, whichever is larger, dimensionless; zero with no moving flow.
    connection_certificate : float
        The worst relative antisymmetry certificate of the measured connection,
        ``norm(W Gamma + Gamma^H W - W', 2) / max|Gamma|``, dimensionless; zero with
        no moving flow.
    dispersive : solprop.translate.responses.DispersiveKerrTerms or None
        The dispersive channels of the guide's gas regions under one of SEP 6's
        dispersive electronic policies (:class:`solprop.modes.material.GasKerr`),
        each with its node region, or ``None``. Under a carrier policy a gas region
        is a constant cubic in `responses` instead.
    electronic : tuple of solprop.nonlinear.responses.DispersiveKerrResponse
        The responses `dispersive` was sampled from, one per channel, each labelled
        ``"<region>:<population>"``: what the run's material manifest records.
        Empty without dispersive channels.
    metric_tables : solprop.translate.coupled.ModalMetricTables or None
        The **dense** metric of a guide that declares a unitary change of basis,
        built from the reference's own flux Gram on the run's coarse grid
        (:func:`solprop.translate.coupled.dense_metric`), or ``None`` for a guide in
        the provider's own coordinates, whose metric is the identity. Where it is
        present its ``gauge_id`` is the tables' composed one and its grids are the
        tables' own, both checked here: a metric written on another basis or another
        grid than the reference it describes is not a metric of this run.

    Raises
    ------
    TranslationError
        If the defect is negative, not finite or above `gram_atol`, the budget is
        not positive and finite, the certificate is empty, the tables carry no
        execution map, or a supplied `metric_tables` is not a dense metric of these
        tables' own basis and grids.
    """

    tables: ModalTransformTables
    responses: RegionalResponseTables | None
    gram_defect: float
    certificate_id: str = "sampled-gram"
    occupied_departure: float = 0.0
    active_departure: float = 0.0
    gram_atol: float = DEFAULT_GRAM_ATOL
    guide_identity: Mapping[str, JsonValue] | None = None
    metric_tables: ModalMetricTables | None = None
    coupled_generator: (
        Complex[
            np.ndarray,
            "n_modes_coupled_out n_modes_coupled_in n_omega_coupled",
        ]
        | None
    ) = None
    reference_channel: int = 0
    moving_generator: (
        Complex[
            np.ndarray,
            "n_intervals_moving n_deg_moving n_modes_moving_out n_modes_moving_in"
            " n_omega_moving",
        ]
        | None
    ) = None
    moving_connection: (
        Complex[
            np.ndarray,
            "n_intervals_link n_deg_link n_modes_link_out n_modes_link_in n_omega_link",
        ]
        | None
    ) = None
    moving_error: float = 0.0
    connection_certificate: float = 0.0
    dispersive: DispersiveKerrTerms | None = None
    electronic: tuple[DispersiveKerrResponse, ...] = ()

    def __post_init__(self) -> None:
        """Check the certificate and the presence of the execution maps.

        Raises
        ------
        TranslationError
            Naming what is missing or out of range.
        """
        if not np.isfinite(self.gram_defect) or self.gram_defect < 0.0:
            msg = (
                "a general modal route's gram_defect is the measured "
                f"norm(W - I, 2) of its basis, got {self.gram_defect!r}"
            )
            raise TranslationError(msg)
        if not np.isfinite(self.gram_atol) or self.gram_atol <= 0.0:
            msg = (
                "a general modal route's gram_atol is the largest defect it will "
                f"call the identity, positive and finite, got {self.gram_atol!r}"
            )
            raise TranslationError(msg)
        if self.gram_defect > self.gram_atol:
            msg = (
                f"a general modal route's basis is {self.gram_defect:.4g} from "
                f"orthonormal, above its gram_atol of {self.gram_atol:.4g}: this "
                "route declares linear.modal_metric = identity, and a continuous "
                "wave's da/dz is then wrong by |1/W_00 - 1|, which no later check "
                "sees. Widen or refine the node rule"
            )
            raise TranslationError(msg)
        for name, measured in (
            ("occupied_departure", self.occupied_departure),
            ("active_departure", self.active_departure),
        ):
            if not np.isfinite(measured) or measured < 0.0:
                msg = (
                    f"a general modal route's {name} is a measured interpolated "
                    f"norm(W - I, 2) over a set of its bins, got {measured!r}"
                )
                raise TranslationError(msg)
        if self.occupied_departure > self.gram_atol:
            msg = (
                f"a general modal route's basis is {self.occupied_departure:.4g} "
                f"from orthonormal **between its knots**, over the bins this run "
                f"occupies, above its gram_atol of {self.gram_atol:.4g}. The knot "
                f"defect is {self.gram_defect:.4g} and passes; the interpolated "
                "one is the number a run actually propagates against, and "
                "declaring linear.modal_metric = identity costs its "
                "physical_energy about that much. Add knots to the guide's "
                "sampling -- the departure is an interpolation error and falls "
                "with the knot spacing -- or declare a larger gram_atol"
            )
            raise TranslationError(msg)
        if not self.certificate_id:
            msg = "a general modal route's certificate_id must be non-empty"
            raise TranslationError(msg)
        if self.tables.synthesis is None or self.tables.projection is None:
            msg = (
                "a general modal route executes through the synthesis and "
                "projection maps of its tables, and this record carries neither: a "
                "reference sampled over a path of positions has no execution map "
                "of its own: build a moving guide through general_modal_route, "
                "which expands its maps along z"
            )
            raise TranslationError(msg)
        self._check_coupled()
        self._check_moving()
        self._check_metric()

    def _check_moving(self) -> None:
        """Copy, validate and freeze a moving guide's flow expansions.

        Raises
        ------
        TranslationError
            If only one of the two is given, a split block is given beside them, a
            shape disagrees with the tables' sections, degree, channels or grid, or
            a value is not finite, or a measurement is negative.
        """
        generator, link = self.moving_generator, self.moving_connection
        for name, value in (
            ("moving_error", self.moving_error),
            ("connection_certificate", self.connection_certificate),
        ):
            if not np.isfinite(value) or value < 0.0:
                msg = (
                    f"a general modal route's {name} is a measured non-negative "
                    f"number, got {value!r}"
                )
                raise TranslationError(msg)
        if generator is None and link is None:
            return
        if generator is None or link is None:
            msg = (
                "a moving guide's flow is its generator and its connection, supplied "
                "together: one without the other is a guide whose modes move with "
                "no connection, or a connection with no generator to carry it"
            )
            raise TranslationError(msg)
        if self.coupled_generator is not None:
            msg = (
                "a general modal route carries a constant split block or a moving "
                "guide's flow, not both: the moving flow already carries every "
                "channel's departure from the reference, a split block's included"
            )
            raise TranslationError(msg)
        synthesis = self.tables.synthesis
        assert synthesis is not None
        shape = (
            int(synthesis.shape[0]),
            int(synthesis.shape[1]),
            self.n_modes,
            self.n_modes,
            int(self.tables.omega.size),
        )
        for name, value in (
            ("moving_generator", generator),
            ("moving_connection", link),
        ):
            array = np.array(value, dtype=np.complex128, order="C", copy=True)
            if array.shape != shape or not np.all(np.isfinite(array)):
                msg = (
                    f"a general modal route's {name} is a finite expansion on the "
                    f"tables' own sections, degree, channels and grid, shape {shape}, "
                    f"got {array.shape}"
                )
                raise TranslationError(msg)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def moving(self) -> bool:
        """Return whether the route is a moving guide's.

        Returns
        -------
        bool
            True when the route carries a moving flow, whose maps then move too.
        """
        return self.moving_generator is not None

    def _check_coupled(self) -> None:
        """Copy, validate and freeze a declared split block's generator.

        Raises
        ------
        TranslationError
            If the block is not square on this route's channels and grid, is not
            finite, is diagonal after all, or the reference channel is not one of
            them.
        """
        block = self.coupled_generator
        if block is None:
            if self.reference_channel:
                msg = (
                    f"a general modal route with no coupled block names reference "
                    f"channel {self.reference_channel}: without a block every "
                    "channel carries its own dispersion in the diagonal flow and no "
                    "channel is the reference"
                )
                raise TranslationError(msg)
            return
        modes, bins = self.n_modes, int(self.tables.omega.size)
        values = np.array(block, dtype=np.complex128, order="C", copy=True)
        if values.shape != (modes, modes, bins) or not np.all(np.isfinite(values)):
            msg = (
                f"a general modal route's coupled_generator must be a finite "
                f"({modes}, {modes}, {bins}) block on its own channels and coarse "
                f"grid, got {values.shape}"
            )
            raise TranslationError(msg)
        if not 0 <= self.reference_channel < modes:
            msg = (
                f"a general modal route's reference_channel is one of its "
                f"{modes} channel(s), got {self.reference_channel}"
            )
            raise TranslationError(msg)
        coupling = float(
            np.max(abs(values - values * np.eye(modes)[:, :, None]), initial=0.0)
        )
        if coupling == 0.0:
            msg = (
                "a general modal route's coupled_generator is diagonal at every "
                "bin: a block that couples nothing is the ordinary route, which "
                "leaves it None and propagates each channel's own dispersion "
                "through the cumulative diagonal flow"
            )
            raise TranslationError(msg)
        values.setflags(write=False)
        object.__setattr__(self, "coupled_generator", values)
        object.__setattr__(self, "reference_channel", int(self.reference_channel))

    def _check_metric(self) -> None:
        """Bind a supplied dense metric to the reference it is meant to describe.

        Raises
        ------
        TranslationError
            If the metric is not dense, is on another basis, or is tabulated on
            grids other than the tables' own.
        """
        metric = self.metric_tables
        if metric is None:
            return
        if metric.kind != "dense":
            msg = (
                f"a general modal route's metric_tables is the dense metric of a "
                f"declared change of basis, and this one is of kind {metric.kind!r}: "
                "a route in the provider's own coordinates leaves it None and the "
                "identity metric is built from the measured Gram defect instead"
            )
            raise TranslationError(msg)
        if metric.gauge_id != self.tables.gauge_id:
            msg = (
                f"a general modal route's metric describes the basis its reference "
                f"is in: the tables' gauge is {self.tables.gauge_id!r} and the "
                f"metric's is {metric.gauge_id!r}"
            )
            raise TranslationError(msg)
        for name, mine, theirs in (
            ("omega", self.tables.omega, metric.omega),
            ("z_edges", self.tables.z_edges, metric.z_edges),
        ):
            if mine.shape != theirs.shape or not np.array_equal(mine, theirs):
                msg = (
                    f"a general modal route's metric is tabulated on its tables' "
                    f"own {name}, and these disagree in shape or value"
                )
                raise TranslationError(msg)
        if int(metric.n_modes) != self.n_modes:
            msg = (
                f"a general modal route's metric describes {metric.n_modes} "
                f"channel(s) and its tables carry {self.n_modes}"
            )
            raise TranslationError(msg)
        assert metric.dense is not None
        departure = max(
            metric_departure(metric.dense[interval, degree])
            for interval in range(metric.dense.shape[0])
            for degree in range(metric.dense.shape[1])
        )
        if departure > MAX_DENSE_METRIC_DEPARTURE:
            msg = (
                f"a general modal route's declared basis is {departure:.4g} from "
                f"orthonormal over the run's own grid, above the "
                f"{MAX_DENSE_METRIC_DEPARTURE:.4g} this route will execute: it "
                "propagates through the diagonal cumulative flow and measures the "
                "controller's norm on the coordinate amplitudes, both of which are "
                "the basis's own only where that basis is orthonormal in true "
                "power. A nonunitary basis would need linear.coupled_flow and a "
                "measured transport, which this route does not build"
            )
            raise TranslationError(msg)

    @property
    def n_modes(self) -> int:
        """Return how many channels the guide propagates.

        Returns
        -------
        int
            The channel axis of the stored reference.
        """
        return int(self.tables.electric.shape[3])

    @property
    def metric(self) -> ModalMetricTables:
        """Return the basis metric this route declares.

        Returns
        -------
        solprop.translate.coupled.ModalMetricTables
            The declared dense metric where a change of basis was declared, and
            otherwise the identity metric on the tables' own grids, carrying the
            measured Gram defect as its error row.
        """
        if self.metric_tables is not None:
            return self.metric_tables
        return orthonormal_metric(
            self.tables.omega,
            self.tables.z_edges,
            defect=self.gram_defect,
            gauge_id=self.tables.gauge_id,
            certificate_id=self.certificate_id,
        )

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``GeneralModalRoute(2 channels, 60 nodes, 1 response,
            gram 3e-15)``.
        """
        responses = 0 if self.responses is None else self.responses.n_responses
        return line(
            "GeneralModalRoute",
            [
                ("", f"{self.n_modes} channel{'' if self.n_modes == 1 else 's'}"),
                ("", f"{int(self.tables.electric.shape[1])} nodes"),
                ("", f"{responses} response{'' if responses == 1 else 's'}"),
                ("", f"gram {self.gram_defect:.1e}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        responses = self.responses
        return "GeneralModalRoute: the seam leaves of one general modal guide", [
            ("channels", self.n_modes),
            ("nodes", int(self.tables.electric.shape[1])),
            (
                "frequencies",
                f"{len(self.tables.omega)} coarse, {len(self.tables.omega_fine)} fine",
            ),
            (
                "metric",
                f"{self.metric.kind}, defect {self.gram_defect:.3e} of "
                f"{self.gram_atol:.3e}",
            ),
            ("occupied departure", f"{self.occupied_departure:.3e}"),
            ("active-band departure", f"{self.active_departure:.3e}"),
            (
                "responses",
                "none (a linear guide)"
                if responses is None
                else f"{responses.n_responses} region(s), {responses.model_id}",
            ),
            ("gauge", self.tables.gauge_id),
            ("certificate", self.certificate_id),
        ]


@dataclass(frozen=True)
class _Regional:
    """The compiled responses of a guide's material regions.

    Attributes
    ----------
    responses : RegionalResponseTables or None
        The constant isotropic cubics, or ``None`` with none.
    dispersive : DispersiveKerrTerms or None
        The gas regions' dispersive channels, or ``None`` with none.
    electronic : tuple of DispersiveKerrResponse
        The responses the channels were sampled from, in channel order.
    """

    responses: RegionalResponseTables | None
    dispersive: DispersiveKerrTerms | None
    electronic: tuple[DispersiveKerrResponse, ...]


def general_modal_route(  # noqa: PLR0913 - a guide, a grid, a launch and four policies
    guide: ModalGuide,
    grid: TimeGrid,
    *,
    z_max: float,
    omega_reference: float,
    kerr_form: KerrForm = "real",
    launch: Complex[np.ndarray, "n_modes n_pol n_omega"] | None = None,
    margin: float = DEFAULT_MARGIN,
    gram_atol: float = DEFAULT_GRAM_ATOL,
    selection: MaterialSelection | None = None,
) -> GeneralModalRoute:
    """Build one guide's seam leaves on the grids a run will execute on.

    The bridge between a :class:`solprop.modes.guide.ModalGuide` -- physics, owned
    by the modes package -- and the three schema-12 leaves a backend executes. The
    guide's own declared sampling (:class:`solprop.modes.guide.ModalSampling`) is
    resolved into a dense reference table, the table is evaluated onto the run's
    **canonical** coarse and fine grids, and each material region's declared
    response is compiled into the per-region leaf. Nothing here invents a
    resolution: the knots, the node rule and the degenerate blocks are the guide's.

    **The guide must span the run's active band.** ``Grid.active_mask`` starts at
    half the lowest declared frequency, so a run declared over 815--860 nm has an
    active band of about 756--1650 nm -- a factor of 2.19 in frequency -- and the
    cumulative operator samples every channel's propagation constant across all of
    it. Outside the guide's knots the stored reference is an explicit zero, so a
    run reaching there would be propagated against zeros; it is refused here
    instead, with the measured numbers in the message.

    **The basis is certified where the run occupies it.** The table's construction
    gate measures the Gram at the **knots**; this route also measures it at the
    run's own coarse bins, **between** them, over the bins the run occupies, and
    refuses a basis that is further from orthonormal there than `gram_atol`. That
    interpolated departure is three orders above the knot defect on a
    sparsely sampled guide and is the number a run actually propagates against; it
    is an interpolation error and falls with the knot spacing. Which bins count is
    :func:`occupancy_weight`'s: `launch` supplies the weighting where it is known.

    Parameters
    ----------
    guide : solprop.modes.guide.ModalGuide
        A guide built by :meth:`solprop.modes.guide.ModalGuide.from_provider`, so
        that it carries the sampling its reference is rebuilt from.
    grid : solprop.grid.TimeGrid
        The run's field grid, which supplies the canonical coarse and fine
        frequency axes, the active band and the fine temporal quadrature weight.
    z_max : float
        Length the run covers, m, positive and finite; the tables are tabulated
        over ``[0, z_max (1 + margin)]`` as every other route's are.
    omega_reference : float
        Positive angular frequency in rad/s at which a region's declared ``n2`` is
        converted to an absolute ``chi3`` -- the run's own reference frequency.
        A region that declares ``chi3`` ignores it.
    kerr_form : {'real', 'analytic'}, optional
        Which instantaneous cubic the regions' Kerr responses are. ``"real"``, the
        default, is the real-field cubic ``P = eps0 chi3 (E.E) E`` the legacy field
        route evaluates, third harmonic included; ``"analytic"`` is the
        analytic-signal form, which carries the fundamental band alone and is the
        tested oracle for it.
    launch : numpy.ndarray or None, optional
        The launched spectrum, shape ``(n_modes, n_pol, n_omega)``, on this grid's
        own frequency axis. It weights the occupied-support basis gate by where
        the light actually is; ``None``, the default, falls back to the run's
        declared band (:func:`occupancy_weight`). It is used for that certificate
        and for nothing else -- no leaf carries it.
    margin : float, optional
        Fraction of `z_max` the tables extend beyond the end, matching the linear
        operator's.
    gram_atol : float, optional
        Largest departure from orthonormal this route calls the identity, passed
        both to the table's construction gate and to the record.
    selection : solprop.physdata.selection.MaterialSelection or None, optional
        The run's material selection, whose electronic policy a region's gas fill
        (:class:`~solprop.modes.material.GasKerr`) takes: the Kerr coefficient at
        the carrier ``2 pi c / omega_reference``, or SEP 6's dispersive cubic, with
        ``kerr_form="real"`` its full cubic and ``"analytic"`` its
        number-conserving sector. ``None`` is ``MaterialSelection()``. A region's
        declared :class:`~solprop.modes.material.MaterialKerr` is constant under
        every policy.

    Returns
    -------
    GeneralModalRoute
        The tables, the compiled per-region responses (``None`` for a guide whose
        regions are all linear) and the measured Gram certificate.

    Raises
    ------
    TranslationError
        If the guide declares no sampling, the run is not on a field grid, the
        guide's band does not cover the run's active band, the interpolated basis
        is further from orthonormal than `gram_atol` over the bins the run
        occupies, the real cubic is asked
        for on a grid that cannot hold its third harmonic, a region declares an
        ``n2`` with no measurement wavelength where its own index moves the
        converted ``chi3`` by more than :data:`N2_CONVERSION_RTOL` across the
        active band, a region declares one **outside** its own index model, or the
        assembled record does not validate.
    """
    sampling = guide.sampling
    if sampling is None:
        msg = (
            "this guide declares no sampling, so the dense E/H reference a general "
            "modal run propagates against cannot be rebuilt from it: build it with "
            "ModalGuide.from_provider(provider, omega, nodes=...), which owns the "
            "knots, the node rule and the degenerate blocks, rather than with "
            "from_step_index, which takes constructed modes and declares no "
            "transverse resolution"
        )
        raise TranslationError(msg)
    if grid.kind != "field":
        msg = (
            f"a general modal guide propagates on a field grid and this run is on "
            f"an {grid.kind} grid: the guide's leaves are tabulated on the "
            "canonical increasing coarse axis, and an envelope grid's axis is in "
            "FFT order about omega_0"
        )
        raise TranslationError(msg)
    _check_active_band(sampling.omega_span, grid)
    if sampling.path is not None:
        return _moving_route(
            guide,
            grid,
            z_max=float(z_max),
            launch=launch,
            gram_atol=gram_atol,
            regional=_regional_responses(
                guide,
                grid,
                omega_reference,
                kerr_form,
                selection=selection,
                span=float(z_max) * (1.0 + float(margin)),
            ),
        )
    table = sample_modal_table(
        guide.field_provider,
        sampling.omega,
        np.array([sampling.z]),
        sampling.nodes,
        degenerate_blocks=sampling.degenerate_blocks,
        basis=sampling.basis,
        gram_atol=gram_atol,
    )
    tables = modal_transform_tables(
        table,
        grid.omega,
        omega_fine=grid.omega_fine,
        z=[0.0, float(z_max) * (1.0 + float(margin))],
        profile_error=sampling.profile_error,
    )
    # One sampling of the reference's own flux Gram at the run's coarse bins, used
    # twice: for the occupied-support certificate every route now carries, and for
    # the dense metric a declared basis declares. Measuring it once is what keeps
    # the two from drifting apart, and it costs 2.2 ms of a 28.2 ms route build --
    # 8 % -- on the 1.5 um strand fixture, and nothing per step.
    sampled = interpolated_gram(table, tables)
    occupied, weight = occupancy_weight(grid, launch)
    departure = metric_departure(sampled, occupied=occupied, weight=weight)
    # The same quantity over every bin the run *can* ever occupy, discounting
    # nothing. The gated one above is weighted by the launched spectrum, so on its
    # own it certifies the launch rather than the run: a broadening run fills bins
    # that weighting discounted to nothing, and on the weakly guiding fixture the
    # worst interpolated bin (1.1355e-4 at 1728.5 nm) is inside `grid.active_mask`.
    # It warns rather than refusing, because a refusal would drop guides this
    # package ships fixtures for and only the caller knows whether the run will
    # broaden that far.
    active = metric_departure(sampled, occupied=grid.active_mask)
    logger.debug(
        "general modal basis: knot Gram defect %.3e, interpolated departure over "
        "the %s %.3e, over the whole active band %.3e, budget %.3e",
        table.gram_defect,
        "launched spectrum" if weight is not None else "declared band",
        departure,
        active,
        gram_atol,
    )
    regional = _regional_responses(
        guide,
        grid,
        omega_reference,
        kerr_form,
        selection=selection,
        span=float(z_max) * (1.0 + float(margin)),
    )
    route = GeneralModalRoute(
        tables,
        regional.responses,
        table.gram_defect,
        occupied_departure=departure,
        active_departure=active,
        gram_atol=gram_atol,
        guide_identity=guide_identity(guide),
        metric_tables=None
        if sampling.basis is None
        else _sampled_metric(table, tables, sampled),
        coupled_generator=coupled_block(table, grid),
        dispersive=regional.dispersive,
        electronic=regional.electronic,
    )
    # After the route is assembled and not before: a guide the gate refuses is
    # refused, and a refusal that arrived with a warning about the same basis
    # would be two messages about one thing.
    _warn_over_the_active_band(active, gram_atol, grid, sampled)
    return route


def _moving_route(
    guide: ModalGuide,
    grid: TimeGrid,
    *,
    z_max: float,
    launch: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
    gram_atol: float,
    regional: _Regional,
) -> GeneralModalRoute:
    """Build the leaves of a guide whose geometry moves along z.

    The guide's path is solved again at exactly the positions its accuracy check
    qualified (:class:`solprop.modes.guide.ModalPath`), on the provider's own rule and
    gauge at each, and three things are expanded along z on its sections:

    * the **maps** ``S`` and ``Q``
      (:func:`solprop.translate.modal_path.path_transform_tables`);
    * the **generator** less the reference channel's entrance diagonal, which is what
      ``Phi`` carries (:func:`solprop.translate.problem._general_assembly`): every
      channel's own dispersion and all of the motion is the flow's;
    * the **connection** ``Gamma = W^-1 S`` of
      :func:`solprop.modes.connection.band_connection`, measured at every position in
      the same gauge and interpolated onto the run's bins by the fields' own cubic in
      frequency. It is stored **as measured**, without conjugation: the provider's
      fields are internal-convention, the synthesis stores them unconjugated, so the
      kernel's amplitude is the coefficient of the internal field on them and
      ``da/dz = ... - Gamma a`` holds in it as derived. And it needs no power
      conversion: the Poynting amplitude **is** the true-power one on this route
      (``W = I``), so the power-weight similarity ``W^(-1/2) C W^(1/2)`` a capillary's
      connection takes is the identity here and ``Re Gamma_nn = W'_nn / 2`` is the
      whole of each channel's own normalisation, which a capillary carries in its
      flow's diagonal instead.

    Parameters
    ----------
    guide : solprop.modes.guide.ModalGuide
        A moving guide, whose sampling carries a path.
    grid : solprop.grid.TimeGrid
        The run's field grid.
    z_max : float
        Where the run ends, m.
    launch : numpy.ndarray or None
        The launched spectrum, for the occupied-support gate.
    gram_atol : float
        The basis budget.
    regional : _Regional
        The regions' compiled responses (:func:`_regional_responses`).

    Returns
    -------
    GeneralModalRoute
        The route, carrying the moving maps and flow.
    """
    # Imported here: modal_path imports this module's siblings only, but keeping the
    # moving machinery out of the snapshot route's import path keeps that route's
    # import graph what it was.
    from solprop.translate.coupled import _coefficient_tail  # noqa: PLC0415
    from solprop.translate.modal_path import (  # noqa: PLC0415
        in_frequency,
        path_connections,
        path_transform_tables,
        sample_modal_path,
        section_coefficients,
    )

    sampling = guide.sampling
    assert sampling is not None and sampling.path is not None
    path = sampling.path
    provider = guide.field_provider
    sampled = sample_modal_path(
        provider,
        sampling.omega,
        sampling.nodes,
        path.z_edges,
        rtol=path.rtol,
        degree=path.degree,
        measured=(path.z_error, path.gram_departure),
        degenerate_blocks=sampling.degenerate_blocks,
        basis=sampling.basis,
        gram_atol=gram_atol,
    )
    tables = path_transform_tables(
        sampled,
        grid.omega,
        omega_fine=grid.omega_fine,
        z_end=z_max,
        profile_error=max(sampling.profile_error, path.z_error),
    )
    kept = int(tables.z_edges.size - 1)
    width = path.degree + 1
    used = sampled.tables[: kept * width]
    grams = [interpolated_gram(one, tables) for one in used]
    occupied, weight = occupancy_weight(grid, launch)
    departure = max(
        metric_departure(one, occupied=occupied, weight=weight) for one in grams
    )
    actives = [metric_departure(one, occupied=grid.active_mask) for one in grams]
    worst = int(np.argmax(actives))
    identity = sampling.nodes.identity
    links = path_connections(
        sampled,
        provider,
        sections=kept,
        degenerate_blocks=sampling.degenerate_blocks,
        basis=sampling.basis,
        n_r=int(cast("int", identity["n_r"])),
        n_theta=int(cast("int", identity["n_theta"])),
    )
    modes = int(tables.electric.shape[3])
    active = grid.active_mask
    entrance = used[0]
    reference = np.array(
        [
            entrance.evaluate(
                float(grid.omega[column]), float(entrance.z[0])
            ).generator[0, 0]
            if active[column]
            else 0.0
            for column in range(grid.omega.size)
        ],
        dtype=np.complex128,
    )
    generator = np.zeros((kept, width, modes, modes, grid.omega.size), np.complex128)
    connection = np.zeros_like(generator)
    certificate = 0.0
    for section in range(kept):
        for position, table in enumerate(sampled.section_tables(section)):
            at = float(table.z[0])
            for column in np.flatnonzero(active):
                block = table.evaluate(float(grid.omega[column]), at).generator
                generator[section, position, :, :, column] = block - reference[
                    column
                ] * np.eye(modes)
            measured = links[section * width + position]
            connection[section, position] = in_frequency(
                table.omega,
                np.stack([one.matrix for one in measured]),
                grid.omega,
                active,
            )
            for one in measured:
                # Relative to the connection's own scale; a connection that is zero
                # to the last bit has nothing to be relative to and reads zero.
                scale = float(np.max(abs(one.matrix)))
                certificate = max(
                    certificate,
                    float(
                        np.divide(
                            one.antisymmetry, scale, out=np.zeros(()), where=scale > 0.0
                        )
                    ),
                )
    generator = section_coefficients(generator)
    connection = section_coefficients(connection)
    error = max(
        float(path.z_error),
        *(
            _coefficient_tail(expansion[section])
            for expansion in (generator, connection)
            for section in range(kept)
        ),
    )
    logger.debug(
        "moving general modal guide: %d section(s) at degree %d, %d channels, "
        "connection up to %.3e 1/m, certificate %.3e, expansion error %.3e",
        kept,
        path.degree,
        modes,
        float(np.max(abs(connection))),
        certificate,
        error,
    )
    route = GeneralModalRoute(
        tables,
        regional.responses,
        max(one.gram_defect for one in used),
        occupied_departure=departure,
        active_departure=actives[worst],
        gram_atol=gram_atol,
        guide_identity=guide_identity(guide),
        moving_generator=generator,
        moving_connection=connection,
        moving_error=error,
        connection_certificate=certificate,
        dispersive=regional.dispersive,
        electronic=regional.electronic,
    )
    _warn_over_the_active_band(actives[worst], gram_atol, grid, grams[worst])
    return route


def coupled_block(
    table: DenseModalTable, grid: TimeGrid, *, reference: int = 0
) -> Complex[np.ndarray, "n_modes_block_out n_modes_block_in n_omega_block"] | None:
    """Return a table's coupled generator relative to one channel, or ``None``.

    A guide whose channels do not exchange power has a **diagonal** generator at
    every bin, and then each channel's dispersion goes into the cumulative diagonal
    flow and this returns ``None``: the ordinary route, untouched. A declared
    unitary basis of a **split** block does not, because ``U.H diag(beta) U`` is
    diagonal only where the block's propagation constants are equal
    (:func:`solprop.translate.modal_tables.apply_channel_basis`), and then what the
    run propagates is a matrix flow.

    What is returned is the whole generator **less** ``reference``'s own diagonal
    entry, because that is what the cumulative ``Phi`` beside a coupled block
    carries: the block's exponential and the diagonal one multiply out to the same
    equation only where ``Phi`` is a multiple of the identity in the channel axis,
    so one channel's dispersion is carried there and every channel's **departure**
    from it here, diagonal included. The
    co-moving frame cancels between the two for the same reason.

    It is exactly zero outside ``grid.active_mask``, as every leaf of the contract
    is: the state is masked to zero there and the table is not defined at every one
    of those bins.

    Parameters
    ----------
    table : solprop.translate.modal_tables.DenseModalTable
        The tracked reference, in whatever basis was declared on it.
    grid : solprop.grid.TimeGrid
        The run's field grid, whose coarse bins the block is tabulated on.
    reference : int, optional
        Which channel's dispersion the diagonal flow carries; the first by default,
        as the coherent wall's is.

    Returns
    -------
    numpy.ndarray or None
        The block in 1/m, complex, shape ``(n_modes, n_modes, n_omega)``, or
        ``None`` when the generator is diagonal to the rounding of its own
        propagation constants at every bin.

    Raises
    ------
    TranslationError
        If `reference` is not one of the table's channels.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import FieldGrid
    >>> from solprop.translate.general import coupled_block
    >>> from solprop.translate.modal_tables import sample_modal_table
    >>> from solprop.modes.guide import ModalGuide
    >>> from solprop.modes.stepindex import StepIndexMode
    >>> from solprop.modes.material import IndexMaterial, MaterialRegion
    >>> modes = [
    ...     StepIndexMode(3e-6, core=1.45, cladding=1.4, profile_wavelength=835e-9,
    ...                   kind=kind, n=0, m=1, parity="even")
    ...     for kind in ("TE", "TM")
    ... ]
    >>> core, cladding = IndexMaterial(1.45), IndexMaterial(1.4)
    >>> guide = ModalGuide.from_step_index(
    ...     modes, core=core, cladding=cladding,
    ...     regions=(
    ...         MaterialRegion("core", 0.0, 3e-6, core),
    ...         MaterialRegion("cladding", 3e-6, None, cladding),
    ...     ),
    ... )
    >>> grid = FieldGrid.from_limits(1e-3, 835e-9, (800e-9, 870e-9), 60e-15)
    >>> active = grid.omega[grid.active_mask]
    >>> omega = np.linspace(float(active[0]), float(active[-1]), 4)
    >>> provider = guide.field_provider
    >>> table = sample_modal_table(
    ...     provider, omega, np.array([0.0]),
    ...     provider.node_rule(float(active[0]), n_r=8), gram_atol=1.0,
    ... )
    >>> coupled_block(table, grid) is None
    True
    """
    at = float(table.z[0])
    modes = int(table.samples[0][0].electric.shape[2])
    if not 0 <= int(reference) < modes:
        msg = (
            f"a coupled block is measured relative to one of the table's {modes} "
            f"channel(s), got {reference}"
        )
        raise TranslationError(msg)
    block = np.zeros((modes, modes, int(grid.omega.size)), dtype=np.complex128)
    identity = np.eye(modes, dtype=np.complex128)
    coupling = 0.0
    floor = 0.0
    for column in np.flatnonzero(grid.active_mask):
        generator = table.evaluate(float(grid.omega[column]), at).generator
        diagonal = np.diag(generator)
        coupling = max(coupling, float(np.max(abs(generator - np.diag(diagonal)))))
        floor = max(
            floor,
            64.0 * float(np.finfo(float).eps) * float(np.max(abs(diagonal))),
        )
        block[:, :, column] = generator - generator[reference, reference] * identity
    if coupling <= floor:
        return None
    logger.debug(
        "general modal route: a coupled block of %d channels, off-diagonal %.3e 1/m "
        "against a %.3e 1/m rounding floor, relative to channel %d",
        modes,
        coupling,
        floor,
        int(reference),
    )
    return block


def _warn_over_the_active_band(
    active: float,
    gram_atol: float,
    grid: TimeGrid,
    sampled: Complex[np.ndarray, "n_modes n_modes n_omega"],
) -> None:
    """Warn where the interpolated basis leaves its budget inside the active band.

    Parameters
    ----------
    active : float
        The interpolated departure maximised over ``grid.active_mask``.
    gram_atol : float
        The budget the occupied departure is gated against.
    grid : solprop.grid.TimeGrid
        The run's grid, which names the worst bin's wavelength.
    sampled : numpy.ndarray
        The sampled flux Gram, shape ``(n_modes, n_modes, n_omega)``, from which
        the worst bin is located.

    Warns
    -----
    solprop.exceptions.GeneralModalBasisWarning
        Naming the worst bin, its wavelength and the fix.
    """
    if active <= gram_atol:
        return
    modes = sampled.shape[0]
    per_bin = np.linalg.norm(
        np.moveaxis(sampled, (0, 1), (-2, -1)) - np.eye(modes), ord=2, axis=(-2, -1)
    )
    worst = int(np.argmax(np.where(grid.active_mask, per_bin, -1.0)))
    msg = (
        f"this general modal guide's interpolated basis is {active:.4g} from "
        f"orthonormal at {2 * np.pi * C / float(grid.omega[worst]) * 1e9:.1f} nm, "
        f"above the gram_atol of {gram_atol:.4g}, and that bin is inside the run's "
        "active band. The gate this route applies is weighted by the launched "
        "spectrum, which discounts that bin to nothing, so it certifies the launch "
        "and not the run: a run that broadens into it pays about this much of its "
        "physical_energy, because the route declares linear.modal_metric = "
        "identity there. The departure is an interpolation error and falls with "
        "the knot spacing, so the fix is more knots in the guide's sampling"
    )
    warnings.warn(msg, GeneralModalBasisWarning, stacklevel=3)


def _sampled_metric(
    table: DenseModalTable,
    tables: ModalTransformTables,
    sampled: Complex[np.ndarray, "n_modes n_modes n_omega"],
) -> ModalMetricTables:
    """Build the dense metric of a declared basis from the table's own flux Gram.

    Parameters
    ----------
    table : solprop.translate.modal_tables.DenseModalTable
        The tracked reference, already in the declared basis.
    tables : solprop.translate.modal_tables.ModalTransformTables
        The flat record, whose coarse grid and positions the metric shares.
    sampled : numpy.ndarray
        The reference's flux Gram at those coarse bins, as
        :func:`interpolated_gram` builds it.

    Returns
    -------
    solprop.translate.coupled.ModalMetricTables
        The dense metric of the declared basis.
    """
    coarse = tables.omega
    logger.debug(
        "dense modal metric: %d channels on %d coarse bins, largest departure from "
        "the identity %.3e (the table's knot Gram defect is %.3e)",
        int(tables.electric.shape[3]),
        len(coarse),
        metric_departure(sampled),
        table.gram_defect,
    )
    # The error row is the table's **knot** Gram defect, the same certificate and
    # the same number an identity-metric route declares, so that the two routes are
    # judged against one another and not against two different quantities. The
    # stored block's own whole-band departure, measured just above, is the larger
    # of the two -- it is evaluated at the run's coarse bins, between the knots,
    # where the interpolated basis is further from orthonormal -- and it is gated
    # separately, against `MAX_DENSE_METRIC_DEPARTURE`, by the route and again by
    # the backend, because a nonunitary transform is nonunitary in every bin.
    # The interpolation error itself is not a property of the declared basis at
    # all: the identity route has it too, and both routes measure it over the
    # bins they occupy and gate it against `gram_atol`
    # (`GeneralModalRoute.occupied_departure`).
    return dense_metric(
        coarse,
        tables.z_edges,
        sampled,
        defect=table.gram_defect,
        gauge_id=tables.gauge_id,
        certificate_id="sampled-gram",
    )


def metric_departure(
    sampled: Complex[np.ndarray, "n_modes_out n_modes_in n_omega"],
    *,
    occupied: Bool[np.ndarray, " n_omega_occupied"] | None = None,
    weight: Float[np.ndarray, " n_omega_weight"] | None = None,
) -> float:
    """Return how far a sampled metric is from the identity, over chosen bins.

    The one measure of how far a basis is from orthonormal in true power, shared by
    the route that builds a dense metric and by the backend that admits one, so that
    the declared bound and the executed check are the same quantity.

    By default it is ``max_omega norm(W(omega) - I, 2)`` over **every** bin, which
    is the right question for a declared change of basis: a nonunitary transform is
    nonunitary wherever it is applied. It is the wrong question for the
    **interpolation** error a sampled table has at the run's own bins, because that
    error lives where the table's knots are sparsest and a run need put no light
    there at all. `occupied` narrows the bins looked at and `weight` replaces the
    maximum by a weighted mean, so that a route can ask instead what the departure
    costs **where its light is** -- which, to first order, is exactly the relative
    error an identity metric puts into
    :func:`solprop.stats.physical_energy`.

    Parameters
    ----------
    sampled : numpy.ndarray
        The Hermitian metric per frequency, shape ``(n_modes, n_modes, n_omega)``.
    occupied : numpy.ndarray or None, optional
        Boolean mask over the frequency axis selecting the bins to look at;
        ``None``, the default, looks at every bin.
    weight : numpy.ndarray or None, optional
        Non-negative weights over the frequency axis, of any scale. Where given,
        the result is ``sum_k w_k d_k / sum_k w_k`` over the occupied bins instead
        of ``max_k d_k``; ``None``, the default, takes the maximum.

    Returns
    -------
    float
        Non-negative and dimensionless.

    Raises
    ------
    TranslationError
        If `occupied` or `weight` does not cover the frequency axis, a weight is
        negative or not finite, `occupied` selects no bin, or the selected weights
        sum to zero, so that there is no occupied support to measure over.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.general import metric_departure
    >>> block = np.eye(2, dtype=complex)[:, :, None] * np.ones(3)
    >>> float(metric_departure(block))
    0.0
    >>> block[0, 0, 2] = 1.5
    >>> float(metric_departure(block))
    0.5
    >>> float(metric_departure(block, occupied=np.array([True, True, False])))
    0.0
    """
    block = np.asarray(sampled)
    modes = block.shape[0]
    moved = np.moveaxis(block, (0, 1), (-2, -1)) - np.eye(modes)
    per_bin = np.linalg.norm(moved, ord=2, axis=(-2, -1))
    for name, given in (("occupied mask", occupied), ("weights", weight)):
        if given is not None and per_bin.shape != np.shape(given):
            msg = (
                f"a metric departure's {name} covers its frequency axis: the "
                f"metric has {per_bin.shape} bin(s) and this one {np.shape(given)}"
            )
            raise TranslationError(msg)
    weights = None if weight is None else np.asarray(weight, dtype=float)
    if weights is not None and (
        np.any(weights < 0.0) or not np.all(np.isfinite(weights))
    ):
        msg = "a metric departure's weights are finite and non-negative"
        raise TranslationError(msg)
    if occupied is not None:
        if not np.any(occupied):
            msg = (
                "a metric departure over an occupied support needs at least one "
                "occupied bin, and this mask selects none: a run that occupies no "
                "bin of its own metric has no basis error to measure"
            )
            raise TranslationError(msg)
        per_bin = per_bin[occupied]
        weights = None if weights is None else weights[occupied]
    if weights is None:
        return float(np.max(per_bin))
    total = float(np.sum(weights))
    if total <= 0.0:
        msg = (
            "a metric departure weighted by an occupancy needs a positive total "
            "weight, and this one is zero: nothing occupies the support it was "
            "asked about"
        )
        raise TranslationError(msg)
    return float(np.sum(weights * per_bin) / total)


def interpolated_gram(
    table: DenseModalTable, tables: ModalTransformTables
) -> Complex[np.ndarray, "n_modes n_modes n_omega"]:
    """Sample the reference's flux Gram at the run's own coarse bins.

    The same interpolated samples the reference arrays were built from, one per
    coarse bin **inside** the guide's band, so this block and the reference are the
    same object seen twice and cannot drift apart. It is what both the declared
    dense metric and the identity route's occupied-support gate are measured on.

    **Outside the band the block is the identity**, not zero and not the last knot
    extrapolated. The stored reference is an explicit zero there
    (:func:`solprop.translate.modal_tables.modal_transform_tables`) and a run whose
    active band leaves the guide's support is refused where the problem is
    assembled, so no state ever lives in those bins; what the record must not do is
    declare a **singular** metric there, because a zero block has a zero smallest
    eigenvalue, which is the record's own way of saying "certifies no positive
    error or noise norm". The identity is the honest statement
    that those bins carry no basis information and nothing lives in them.

    Parameters
    ----------
    table : solprop.translate.modal_tables.DenseModalTable
        The tracked reference, in whatever basis it was built in.
    tables : solprop.translate.modal_tables.ModalTransformTables
        The flat record, whose coarse grid the block is sampled on.

    Returns
    -------
    numpy.ndarray
        ``W(omega)``, complex128, shape ``(n_modes, n_modes, n_omega)``,
        dimensionless.
    """
    coarse = tables.omega
    modes = int(tables.electric.shape[3])
    sampled = np.broadcast_to(
        np.eye(modes, dtype=np.complex128)[:, :, None], (modes, modes, len(coarse))
    ).copy()
    inside = np.flatnonzero((table.omega[0] <= coarse) & (coarse <= table.omega[-1]))
    for column in inside:
        sampled[:, :, column] = table.evaluate(
            float(coarse[column]), float(table.z[0])
        ).flux_metric
    return sampled


def occupancy_weight(
    grid: TimeGrid,
    launch: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
) -> tuple[Bool[np.ndarray, " n_omega"], Float[np.ndarray, " n_omega"] | None]:
    """Return which bins a run occupies, and how heavily, for the basis gate.

    Two statements, in order of strength. The bins a run **can** occupy are
    ``grid.active_mask``: the linear operator is identically zero outside it and
    the hard band mask zeroes the state there once per accepted step, so a bin
    outside it is a bin nothing ever occupies. Where a `launch` is declared the
    weighting is that **launched** spectral energy density, which makes the
    weighted departure the first-order relative error an identity metric costs
    the launch's :func:`solprop.stats.physical_energy`.

    **The weighting is the launch and not the run.** Nothing here knows what the
    run does with it: a broadening run occupies bins the launched spectrum
    discounted to nothing, and pays the basis error there in full. That is why
    :func:`general_modal_route` measures the unweighted maximum over
    ``grid.active_mask`` as well, and warns when it leaves the budget
    (:class:`solprop.exceptions.GeneralModalBasisWarning`,
    :attr:`GeneralModalRoute.active_departure`).

    Where no launch is declared -- a record assembled by hand, or a guide checked
    before a pulse exists -- the fall-back is the run's **declared** band,
    ``grid.inband_mask``, unweighted: the only statement about occupancy a grid
    makes on its own. It is the conservative one of the two, since a maximum over
    a band is never below a mean weighted inside it.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The run's grid, whose active and in-band masks say where light can be.
    launch : numpy.ndarray or None
        The launched spectrum, shape ``(n_modes, n_pol, n_omega)``, internal
        convention, or ``None``.

    Returns
    -------
    tuple
        The occupied mask and the weights, the latter ``None`` where no launch was
        declared, for :func:`metric_departure`.

    Raises
    ------
    TranslationError
        If `launch` does not lie on this grid's frequency axis, or carries no
        energy inside the active band at all.
    """
    if launch is None:
        return grid.inband_mask, None
    spectrum = np.asarray(launch)
    if spectrum.ndim != 3 or spectrum.shape[-1] != grid.n_omega:
        msg = (
            f"a general modal launch lies on the run's own frequency axis of "
            f"{grid.n_omega} bins, shaped (n_modes, n_pol, n_omega); got "
            f"{spectrum.shape}"
        )
        raise TranslationError(msg)
    density = np.sum(
        fourier.spectral_density(spectrum, grid.dt, kind=grid.kind), axis=(0, 1)
    )
    weights = np.where(grid.active_mask, density, 0.0)
    if not np.any(weights > 0.0):
        msg = (
            "a general modal launch carries no energy inside the run's active "
            "band, so there is nothing for the basis gate to weigh: check the "
            "pulse's wavelength against the grid's declared limits"
        )
        raise TranslationError(msg)
    return grid.active_mask, weights


def guide_identity(guide: ModalGuide) -> Mapping[str, JsonValue]:
    """Return what a guide *is*, as a run records it and a resume compares it.

    The provider's own identity -- its model, its ordered channels and its
    materials -- together with the two things the provider protocol does not
    declare and a run depends on: the :class:`solprop.modes.guide.ModalSampling`
    the dense reference is rebuilt from, and each material region with the Kerr
    response it declares. Every entry is a scalar, a string or a list of them, so
    that the mapping survives JSON unchanged and two processes agree on it.

    Parameters
    ----------
    guide : solprop.modes.guide.ModalGuide
        A guide carrying a sampling, as
        :meth:`~solprop.modes.guide.ModalGuide.from_provider` builds it.

    Returns
    -------
    collections.abc.Mapping
        JSON-serialisable identity.

    Raises
    ------
    TranslationError
        If the guide declares no sampling, so that nothing says what resolution
        its reference was built at.
    """
    sampling = guide.sampling
    if sampling is None:
        msg = (
            "this guide declares no sampling, so a run cannot record what "
            "resolution its reference was built at: build it with "
            "ModalGuide.from_provider"
        )
        raise TranslationError(msg)
    return {
        "provider": dict(guide.identity),
        "sampling": dict(sampling.identity),
        "regions": [
            {
                "name": region.name,
                "material": dict(region.material.identity),
                "kerr": None
                if region.kerr is None
                else {
                    "chi3": region.kerr.chi3,
                    "n2": region.kerr.n2,
                    # The measurement wavelength changes the executed chi3 by the
                    # square of the index ratio and nothing downstream can see it
                    # afterwards, so it belongs in the key beside the coefficient
                    # it converts.
                    "at": region.kerr.at,
                },
                "raman": None
                if region.raman is None
                else _raman_identity(region.raman.parameters),
                # Written only for a region with a gas fill, so that every other
                # guide's identity, and its resume key, is what it was.
                **({} if region.gas is None else {"gas": _gas_identity(region.gas)}),
            }
            for region in guide.regions
        ],
    }


def _gas_identity(gas: GasKerr) -> dict[str, JsonValue]:
    """Return a region's gas fill as a run records it.

    Parameters
    ----------
    gas : solprop.modes.material.GasKerr
        The fill.

    Returns
    -------
    dict
        Each population's label, species, density in m^-3 and optmat record
        digests; the electronic law the run takes is recorded with its policy.
    """
    return {
        "populations": [
            {
                "label": component.label,
                "species": component.species.name,
                "density": density,
                "records": [
                    record.digest for record in component.species.material_records
                ],
            }
            for component, density in zip(
                gas.fill.components, gas.densities(), strict=True
            )
        ],
        "temperature": gas.fill.temperature
        if isinstance(gas.fill.temperature, float | int)
        else repr(gas.fill.temperature),
    }


def _dispersive_identity(leaf: DispersiveKerrTerms) -> dict[str, JsonValue]:
    """Return a general modal route's dispersive channels as a run records them.

    Parameters
    ----------
    leaf : solprop.translate.responses.DispersiveKerrTerms
        The compiled channels.

    Returns
    -------
    dict
        The evaluator, the sector, the tensor approximation and each channel's
        label and node region; the models and their support records are the
        material manifest's.
    """
    assert leaf.regions is not None
    return {
        "evaluator": leaf.evaluator,
        "sector": leaf.sector,
        "tensor": "isotropic-kleinman",
        "channels": [
            {"label": label, "region": int(region)}
            for label, region in zip(leaf.labels, leaf.regions, strict=True)
        ],
    }


def _raman_identity(
    parameters: LinAgrawalRaman | SilicaBundle,
) -> dict[str, JsonValue]:
    """Return a region's delayed response parameters as a run records them.

    Parameters
    ----------
    parameters : solprop.physdata.raman_types.LinAgrawalRaman or SilicaBundle
        The declared time constants and fraction, or an optmat tensor bundle.

    Returns
    -------
    dict
        The declared parameters field by field; an optmat bundle by its name, its
        digest and its Raman fraction, the digest identifying its numbers (the
        bundle's own model is recorded in the run's material manifest).
    """
    if isinstance(parameters, SilicaBundle):
        return {
            "bundle": parameters.name,
            "digest": parameters.digest,
            "fraction": parameters.raman_fraction,
        }
    return cast("dict[str, JsonValue]", dataclasses.asdict(parameters))


def modal_identity(
    route: GeneralModalRoute,
    grid: TimeGrid,
    *,
    channels: Sequence[ChannelLabel],
    has_noise: bool,
    flow: CoupledFlowTables | None = None,
) -> Mapping[str, JsonValue]:
    """Return the general modal identity output schema 7 persists.

    One mapping answering, for a file read years later, every question the mode
    axis of a general modal run raises: **which channels**, in which order, with
    which quantum numbers; **which tables**, by the three declared identities and
    the extents of the stored reference; **which basis metric** and how far from
    orthonormal it was measured to be; **which responses**, by model and by the
    declared susceptibility of each region; **which guide**, by provider,
    sampling and materials; the shared co-moving frame; the run's active band;
    the basis the noise was drawn in; and which accessors the basis admits
    (:data:`solprop.stats.GENERAL_MODAL_ACCESSORS`, which
    :mod:`solprop.output.capability` gates with).

    It goes into :attr:`solprop.output.schema.RunMetadata.modal`, which is inside
    the schema-7 resume key, so a run continued against a changed guide, table,
    material or channel order is refused by the field that changed.

    **A declared change of basis is in the key by four independent paths**, so a
    file written in one basis cannot resume in another: ``metric.kind`` moves from
    ``"identity"`` to ``"dense"``; ``metric.gauge_id`` and ``tables.gauge_id`` both
    carry the composed identity ``"<tracking>+<basis>"``; the ``channels`` rows
    carry the handedness and lose the parity; and ``guide.sampling.basis`` carries
    the declaration itself, matrix and all. The last of those is present **only**
    where a basis is declared, so the key of a guide in the provider's own
    coordinates does not carry it; the first three are always compared, which is
    what makes that omission safe in both directions.

    **Measured entries are quantised and declared ones are not.** The metric's
    Gram defect, the sampling's profile error and the tables' profile-error
    budget are the outputs of a dense solve and a reduction, so they enter at
    :data:`~solprop.modes.material.MEASURED_DIGITS` significant digits: a
    re-translation of the same guide on another BLAS moves their last bits and
    must not refuse a resume. A declared ``chi3``, a radius, a tolerance and a
    knot enter exactly, because a caller who changes one means to.

    Parameters
    ----------
    route : GeneralModalRoute
        The route the run was assembled from, carrying its `guide_identity`.
    grid : solprop.grid.TimeGrid
        The run's grid, whose active band the identity records.
    channels : collections.abc.Sequence
        The ordered channel labels of the mode axis, one per row.
    has_noise : bool
        Whether the run carries a noise realisation, which fixes the reference
        basis the noise draw was made in.
    flow : solprop.translate.coupled.CoupledFlowTables or None, optional
        The ``linear.coupled_flow`` the assembled problem carries, when it carries
        one. Its description joins the record under ``"flow"``, as
        :func:`solprop.translate.coupled.wall_identity`'s does, so that a resume
        across a changed basis is refused by the block that moved as well as by the
        gauge. The key is written **only** where a flow exists, so the key of a
        run without one does not carry it, exactly as ``guide.sampling.basis`` is
        written only where a basis is declared.

    Returns
    -------
    collections.abc.Mapping
        JSON-serialisable identity, stable across processes.

    Raises
    ------
    TranslationError
        If the route carries no guide identity, or the labels do not match the
        table's channel count.
    """
    identity = route.guide_identity
    if identity is None:
        msg = (
            "this general modal route carries no guide identity, so a file "
            "written from it could not say which guide produced it: build the "
            "route with general_modal_route(guide, ...), which fills it"
        )
        raise TranslationError(msg)
    if len(channels) != route.n_modes:
        msg = (
            f"a general modal identity needs one channel label per row of the "
            f"mode axis: the tables carry {route.n_modes} channel(s) and "
            f"{len(channels)} label(s) were given"
        )
        raise TranslationError(msg)
    active = grid.omega[grid.active_mask]
    metric = route.metric
    responses = route.responses
    record: dict[str, JsonValue] = {
        "basis": "general_modal",
        "n_components": int(route.tables.electric.shape[2]),
        "n_channels": int(route.n_modes),
        "channels": [dict(label.as_mapping()) for label in channels],
        "tables": dict(route.tables.identity),
        "metric": {
            "kind": metric.kind,
            "gauge_id": metric.gauge_id,
            "certificate_id": metric.certificate_id,
            # Measured, so quantised: the sampled Gram of the same guide re-solved
            # on another BLAS differs in its last bits and in nothing else, and a
            # resume must not refuse a run that did not change. The budget beside
            # it is declared, and enters exactly.
            "defect": quantised_measurement(route.gram_defect),
            # Measured too, and for the same reason quantised: the interpolated
            # departure over the bins this run occupies, which is what declaring
            # the identity costs it, and the same quantity over every bin the
            # active band admits, which is what it would cost a run that
            # broadened into the worst of them.
            "occupied_departure": quantised_measurement(route.occupied_departure),
            "active_departure": quantised_measurement(route.active_departure),
            "budget": float(route.gram_atol),
        },
        "responses": None
        if responses is None
        else {
            "model_id": responses.model_id,
            "policy_id": responses.policy_id,
            "n_responses": int(responses.n_responses),
            "region": [int(one) for one in responses.response_region],
            "chi3": [float(one) for one in responses.chi3[0, 0]],
        },
        # Written only where a gas region takes SEP 6's dispersive cubic, so that
        # every other route's record is what it was.
        **(
            {}
            if route.dispersive is None
            else {"dispersive": _dispersive_identity(route.dispersive)}
        ),
        "guide": dict(identity),
        "frame": "shared co-moving",
        "power_weight": "constant",
        "active_band": [float(active[0]), float(active[-1])],
        # The basis the ADR-008 draw was **made** in, which is the orthonormal
        # true-power reference whatever basis the run is executed in: a declared
        # change of basis carries that one draw through the inverse transform
        # rather than drawing again. Which transform it was
        # carried through is `metric.gauge_id`, two rows up.
        "noise_reference_basis": ("orthonormal true power" if has_noise else None),
        "accessors": [
            str(one)
            for one in sorted(GENERAL_MODAL_ACCESSORS)
            if flow is None or one not in WITHHELD_WITH_A_FLOW
        ],
    }
    if flow is not None:
        record["flow"] = {
            "model": "moving guide" if route.moving else "declared channel basis",
            "section_methods": list(flow.section_methods),
            "n_sections": len(flow.section_methods),
            "n_deg": int(flow.n_deg),
            "connection": flow.connection is not None,
            "gauge_id": flow.gauge_id,
            "certificate_id": flow.certificate_id,
            "reference_channel": int(route.reference_channel),
            # A moving guide's connection certificate, written only where there is
            # one so that a split block's file keeps the key it was stamped with.
            **(
                {
                    "connection_certificate": quantised_measurement(
                        route.connection_certificate
                    )
                }
                if route.moving
                else {}
            ),
            # Measured, and therefore quantised against the last bits of the BLAS
            # that produced them, as every other measured entry of this record is.
            "generator_error": quantised_measurement(
                float(np.max(flow.generator_error))
            ),
            "residual_norm_bound": quantised_measurement(
                float(np.max(flow.residual_norm_bound))
            ),
            "inverse_condition_bound": quantised_measurement(
                float(np.max(flow.inverse_condition_bound))
            ),
            "largest_coupling": quantised_measurement(
                float(np.max(abs(flow.generator)))
                if flow.generator is not None
                else 0.0
            ),
        }
    return record


def _check_active_band(span: tuple[float, float], grid: TimeGrid) -> None:
    """Refuse a run whose active band reaches outside the guide's own knots.

    Parameters
    ----------
    span : tuple of float
        Lowest and highest knot the guide was solved at, rad/s.
    grid : solprop.grid.TimeGrid
        The run's grid, whose active mask is the band that must be covered.

    Raises
    ------
    TranslationError
        Naming both bands, in rad/s and in wavelength, and the factor in frequency
        the active band spans.
    """
    active = grid.omega[grid.active_mask]
    low, high = float(active[0]), float(active[-1])
    if span[0] <= low and span[1] >= high:
        return
    msg = (
        f"this guide is solved over {span[0]:.6e} to {span[1]:.6e} rad/s "
        f"({2 * np.pi * C / span[1] * 1e9:.1f} to "
        f"{2 * np.pi * C / span[0] * 1e9:.1f} nm) and the run's active band is "
        f"{low:.6e} to {high:.6e} rad/s "
        f"({2 * np.pi * C / high * 1e9:.1f} to {2 * np.pi * C / low * 1e9:.1f} nm), "
        f"a factor {high / low:.2f} in frequency. Grid.active_mask starts at half "
        "the lowest declared frequency and runs to the window's highest, so a "
        "general guide is always solved over at least twice the band its run "
        "declares: outside its knots every stored profile is an explicit zero and "
        "the run would be propagated against zeros. Sample the provider over "
        "grid.omega[grid.active_mask][[0, -1]] and build the guide from those "
        "knots"
    )
    raise TranslationError(msg)


def _regional_responses(
    guide: ModalGuide,
    grid: TimeGrid,
    omega_reference: float,
    form: KerrForm,
    *,
    selection: MaterialSelection | None,
    span: float,
) -> _Regional:
    """Compile the guide's regions' own declared responses.

    A region's :class:`~solprop.modes.material.MaterialKerr` is one constant
    ``chi3`` under every electronic policy. A region's gas fill
    (:class:`~solprop.modes.material.GasKerr`) takes the selection's policy: under a
    carrier policy it joins the constant cubics with ``chi3 = sum_s N_s gamma3_s``
    at the carrier ``2 pi c / omega_reference``; under a dispersive one each species
    becomes a channel of SEP 6's dispersive cubic, in the sector `form` names, with
    the region as its node region.

    Parameters
    ----------
    guide : solprop.modes.guide.ModalGuide
        The guide whose regions own the responses.
    grid : solprop.grid.TimeGrid
        The run's grid, which supplies the fine temporal quadrature and the fine
        spectral layout.
    omega_reference : float
        Frequency in rad/s at which a declared ``n2`` becomes an absolute ``chi3``,
        and the carrier of a gas fill under a carrier policy.
    form : {'analytic', 'real'}
        Which instantaneous cubic the compiled tensor is; for a dispersive channel,
        the number-conserving sector or the full cubic.
    selection : solprop.physdata.selection.MaterialSelection or None
        The run's material selection; ``None`` is ``MaterialSelection()``.
    span : float
        Length in m the density tables of the dispersive channels cover.

    Returns
    -------
    _Regional
        The constant cubics per responding region, in the guide's own region order
        (``None`` when no region has one), and the dispersive channels (``None``
        without a gas region under a dispersive policy).

    Raises
    ------
    TranslationError
        If a gas fill's species has no Kerr data, or no hyperpolarizability model
        under a dispersive policy.
    """
    active = grid.omega[grid.active_mask]
    band = (float(active[0]), float(active[-1]))
    chosen = MaterialSelection() if selection is None else selection
    values: list[float] = []
    indices: list[int] = []
    delayed: list[SilicaTensorResponse | None] = []
    channels: list[tuple[int, DispersiveKerrResponse, Constant]] = []
    for index, region in enumerate(guide.regions):
        kerr, gas = region.kerr, region.gas
        if kerr is not None:
            _check_declared_n2(region, band, omega_reference)
            values.append(kerr.susceptibility(region.material, omega_reference))
            indices.append(index)
            delayed.append(region.raman)
        elif gas is not None:
            resolved = _gas_responses(region, gas, chosen, omega_reference, form)
            constant = [
                (response, density)
                for response, density in resolved
                if isinstance(response, KerrResponse)
            ]
            if len(constant) == len(resolved):
                values.append(
                    math.fsum(response.chi3 * density for response, density in constant)
                )
                indices.append(index)
                delayed.append(None)
            else:
                channels.extend(
                    (index, cast("DispersiveKerrResponse", response), Constant(density))
                    for response, density in resolved
                )
    if not values and not channels:
        logger.debug("general modal guide: every region is linear, no source compiled")
        return _Regional(None, None, ())
    logger.debug(
        "general modal guide: chi3 %s m^2/V^2 in region(s) %s, at %.6e rad/s, "
        "%d delayed declaration(s), %d dispersive channel(s)",
        [f"{one:.6e}" for one in values],
        indices,
        omega_reference,
        sum(one is not None for one in delayed),
        len(channels),
    )
    _check_harmonic_room(grid, form)
    constant_tables = (
        regional_responses(
            values,
            indices,
            raman=delayed,
            n_to=grid.n_to,
            dt_fine=grid.dt_fine,
            form=form,
        )
        if values
        else None
    )
    if not channels:
        return _Regional(constant_tables, None, ())
    electronic = tuple(response for _, response, _ in channels)
    _log_gas_regions(electronic, grid)
    return _Regional(
        constant_tables,
        dispersive_terms(
            electronic,
            [cast("str", response.species) for response in electronic],
            [profile_table(path, 0.0, span) for _, _, path in channels],
            grid,
            regions=[index for index, _, _ in channels],
        ),
        electronic,
    )


def _gas_responses(
    region: MaterialRegion,
    gas: GasKerr,
    selection: MaterialSelection,
    omega_reference: float,
    form: KerrForm,
) -> list[tuple[KerrResponse | DispersiveKerrResponse, float]]:
    """Resolve a region's gas fill under the run's electronic policy.

    Parameters
    ----------
    region : solprop.modes.material.MaterialRegion
        The region, for its name.
    gas : solprop.modes.material.GasKerr
        Its fill.
    selection : solprop.physdata.selection.MaterialSelection
        The run's material selection.
    omega_reference : float
        The run's carrier in rad/s.
    form : {'analytic', 'real'}
        The cubic's form: ``"real"`` keeps the full cubic.

    Returns
    -------
    list of tuple
        Per population, its electronic response, labelled
        ``"<region>:<population>"``, and its number density in m^-3.

    Raises
    ------
    TranslationError
        Naming the region and the population whose data the policy cannot take.
    """
    resolved: list[tuple[KerrResponse | DispersiveKerrResponse, float]] = []
    for component, density in zip(gas.fill.components, gas.densities(), strict=True):
        try:
            response = electronic_response(
                component.species,
                selection=selection,
                wavelength=2.0 * np.pi * C / omega_reference,
                thg=form == "real",
                species=f"{region.name}:{component.label}",
            )
        except MaterialError as error:
            msg = (
                f"material region {region.name!r}, population {component.label!r}: "
                f"{error}"
            )
            raise TranslationError(msg) from error
        resolved.append((response, density))
    return resolved


def _log_gas_regions(
    responses: Sequence[DispersiveKerrResponse], grid: TimeGrid
) -> None:
    """Log, once, each gas channel whose band leaves its support (SEP 6, 4.10).

    A general modal guide's node field has three Cartesian components, so the
    any-polarisation band edge applies. The run is never refused for the frequencies
    its field occupies.

    Parameters
    ----------
    responses : sequence of DispersiveKerrResponse
        The gas regions' dispersive channels.
    grid : solprop.grid.TimeGrid
        The run's grid, whose active band is classified.
    """
    for response in responses:
        regions = response.model.regions(
            0.5 * grid.omega_min, grid.omega_max_window, vector=True
        )
        if regions.beyond_support:
            logger.info(
                "dispersive electronic response of %s, %s; the run is not refused "
                "for the frequencies its field occupies (SEP 6, section 4.10)",
                response.species,
                regions.describe(),
            )


def _check_harmonic_room(grid: TimeGrid, form: KerrForm) -> None:
    """Refuse the real cubic on a grid whose fine axis folds it into the band.

    The source is evaluated on the **fine** time axis and cropped back to the coarse
    one, so the question is not whether the third harmonic is representable but
    whether what it folds to lands inside the coarse band. Both axes share one bin
    spacing, so with ``n_t`` coarse samples and ``n_to`` fine ones the coarse band
    reaches ``n_t/2`` bins and the fine Nyquist ``n_to/2``. A cubic of a field
    occupying the coarse band reaches ``3 n_t / 2`` bins; sampled on the fine axis
    it folds about the Nyquist to ``n_to - 3 n_t / 2`` and above, and that is at or
    above the coarse top exactly when

    ``n_to >= 2 n_t`` .

    Which is ``ffac = 6`` on a field grid built by
    :meth:`solprop.grid.FieldGrid.from_limits`, its default and the value
    ``conventions.md`` section 3 names for a cubic response. Both axes are rounded
    up to a convenient transform length, so what a given ``ffac`` buys is measured
    and not proportional: on the test window ``ffac`` of 2, 3 and 4 all give
    ``n_to = n_t`` -- no oversampling at all, and the harmonic folds into the band
    -- while 5, 6 and 8 all give ``n_to = 2 n_t``. The bound is exact at
    equality only for content at the very top of the coarse band, which is inside
    the apodisation collar; the field's own declared window reaches ``n_t / 2.2``
    bins, whose harmonic folds to ``n_to - 3 n_t / 2.2``, clear of the coarse top by
    ``0.136 n_t`` bins on the default grid.

    The analytic-signal form generates no harmonic and is not gated here.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The run's field grid.
    form : {'analytic', 'real'}
        Which instantaneous cubic the compiled tensor is.

    Raises
    ------
    TranslationError
        Naming both sizes and the oversampling that fixes it.
    """
    if form != "real" or grid.n_to >= 2 * grid.n_t:
        return
    msg = (
        f"this run asks for the real-field cubic, whose third harmonic reaches "
        f"three times the coarse band, and its grid carries {grid.n_to} fine "
        f"samples to {grid.n_t} coarse ones: the harmonic folds about the fine "
        f"Nyquist into the band the run keeps, and the source would be silently "
        f"wrong there. A field grid needs n_to >= 2 n_t, which is "
        f"FieldGrid.from_limits(..., ffac=6), its default and what "
        "conventions.md section 3 names for a cubic response. Build the grid with "
        'ffac=6, or ask for the analytic-signal form with kerr_form="analytic", '
        "which carries the fundamental band alone"
    )
    raise TranslationError(msg)


def _check_declared_n2(
    region: MaterialRegion, band: tuple[float, float], omega_reference: float
) -> None:
    """Refuse an ``n2`` with no declared wavelength where the default is a guess.

    A region's compiled response is **one** ``chi3``, so a declared ``n2`` has to
    be converted at one frequency and nothing downstream can tell which. When the
    region's index barely moves across the run's active band the choice cannot be
    wrong by much and asking the caller would be noise; when it moves, the default
    -- the run's own reference frequency -- is a guess the caller never made. So a
    spread above :data:`N2_CONVERSION_RTOL` is refused unless the response says
    ``at=``.

    The spread is measured on the executed ``chi3`` itself, over the active band
    clipped into the material's own validity domain and the reference frequency,
    so it is the number the refusal is about and not a proxy for it.

    A declared ``at`` is checked instead: the conversion reads the region's own
    index **there**, so an ``at`` outside the region's ``omega_domain`` is refused
    here, naming the region and both intervals, rather than surfacing later as the
    material's own generic refusal from inside the leaf compile.

    Parameters
    ----------
    region : solprop.modes.material.MaterialRegion
        A region that declares a Kerr response.
    band : tuple of float
        Lowest and highest active frequency of the run, rad/s.
    omega_reference : float
        The frequency an undeclared conversion would use, rad/s.

    Raises
    ------
    TranslationError
        Naming the region, the measured spread and the two ways to declare it, or
        naming a declared ``at`` outside the region's own index model.
    """
    kerr = region.kerr
    if kerr is None or kerr.n2 is None:
        return
    domain = region.material.omega_domain
    if kerr.at is not None:
        measured = 2.0 * np.pi * C / float(kerr.at)
        if not domain[0] <= measured <= domain[1]:
            msg = (
                f"material region {region.name!r} declares its n2 measured at "
                f"{kerr.at * 1e9:.1f} nm, which is {measured:.6e} rad/s, and its "
                f"own index model is valid only over {domain[0]:.6e} to "
                f"{domain[1]:.6e} rad/s "
                f"({2 * np.pi * C / domain[1] * 1e9:.1f} to "
                f"{2 * np.pi * C / domain[0] * 1e9:.1f} nm): the conversion "
                "chi3 = 4 eps0 c n_0^2 n2 / 3 reads the region's own index at that "
                "wavelength and there is none there. Declare a wavelength inside "
                "the model, or declare chi3 directly"
            )
            raise TranslationError(msg)
        return
    low = max(band[0], domain[0])
    high = min(band[1], region.material.omega_domain[1])
    probes = [omega_reference, *([low, high] if high > low else [])]
    values = [kerr.susceptibility(region.material, one) for one in probes]
    smallest, largest = min(values), max(values)
    if smallest <= 0.0 or largest / smallest - 1.0 <= N2_CONVERSION_RTOL:
        return
    msg = (
        f"material region {region.name!r} declares n2 = {kerr.n2:.6e} m^2/W with "
        f"no measurement wavelength, and its own linear index moves enough across "
        f"this run's active band that the conversion chi3 = 4 eps0 c n_0^2 n2 / 3 "
        f"spans {largest / smallest - 1.0:.4%} of itself "
        f"({smallest:.6e} to {largest:.6e} m^2/V^2 over "
        f"{2 * np.pi * C / high * 1e9:.1f} to {2 * np.pi * C / low * 1e9:.1f} nm) "
        f"-- above the {N2_CONVERSION_RTOL:.2%} this route will convert at an "
        "undeclared frequency. The compiled leaf carries one chi3 and nothing "
        "downstream can say which frequency it came from, so declare the "
        "wavelength the n2 was measured at, MaterialKerr(n2=..., at=<m>), or "
        "declare chi3 directly"
    )
    raise TranslationError(msg)


def check_noise_basis(metric: ModalMetricTables | None, *, has_noise: bool) -> None:
    """Refuse a shot-noise draw in a basis it is not defined in.

    The existing draw is one photon per channel per bin with a random phase, and
    that is a statement about an **orthonormal true-power** basis: only there does
    "one photon in this channel" mean an energy of ``hbar omega``. Two metrics
    satisfy that and are admitted:

    * ``"identity"`` -- the provider's own Poynting-normalised coordinates, where
      the draw is added to the state as it stands;
    * ``"dense"`` -- a **declared unitary change of basis** of such a reference,
      where the same draw is made once in the reference basis, with the same seed
      and the same stream, and carried into the run's coordinates by
      :func:`solprop.noise.transformed_noise`. The total photon number per bin is
      exactly what it was and the per-channel numbers are redistributed, because
      fixed-modulus random phases are not invariant in distribution under unitary
      mixing. That the dense
      metric really is such a transform is gated where it is built -- unitarity and
      the block's exact degeneracy at the table, the departure from the identity at
      the route and again at the backend -- so by the time it reaches here it is
      one.

    A ``"diagonal"`` metric is refused: it is a power-orthogonal basis with unequal
    channel norms, its per-channel energies are not ``hbar omega``, nothing in this
    package builds one and no transport for it has been derived.

    Parameters
    ----------
    metric : solprop.translate.coupled.ModalMetricTables or None
        The basis metric the route declares, or ``None`` for a route with none.
    has_noise : bool
        Whether the run carries a shot-noise realisation.

    Raises
    ------
    TranslationError
        If a realisation is carried and the metric is neither the identity nor the
        dense metric of a declared unitary change of basis.
    """
    if not has_noise or metric is None or metric.kind in ("identity", "dense"):
        return
    msg = (
        f"a shot-noise realisation is carried and this general modal guide's "
        f"linear.modal_metric is of kind {metric.kind!r}: the shot-noise draw is one "
        "photon per channel per bin, which is an energy-normalised true-power "
        "amplitude only in a basis that is orthonormal in true power -- the "
        "identity metric, or a declared unitary change of basis of one, whose "
        "draw is made in the reference basis and carried through the inverse "
        "transform. A basis with unequal channel norms is neither, per-channel "
        "energies are not hbar omega there, and no transport for it has been "
        "derived. Run without noise, or in a certified orthonormal basis"
    )
    raise TranslationError(msg)
