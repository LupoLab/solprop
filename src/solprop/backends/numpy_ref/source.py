# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The stage derivative: the nonlinear source of a lab-frame field, on every route.

:class:`StageSource` is what :meth:`NumpyWorkspace.nonlinear
<solprop.backends.numpy_ref.workspace.NumpyWorkspace.nonlinear>` asks. Its
:meth:`~StageSource.evaluate` runs the passes of ``docs/dev/architecture.md`` section 8
(:mod:`solprop.backends.numpy_ref.passes`) on the route the problem takes: the modal
quadrature in node chunks, with the embedded coarse rule projected beside it and the
held plasma current assembled in frequency; the radial route's plasma, projected once in
frequency; the free and planar geometries, through transverse transforms or the slab
synthesis; the general modal guide (:mod:`~solprop.backends.numpy_ref.general_modal`);
and the bound TDSE response (:mod:`solprop.backends.tdse_source`). It adds ADR-025's
in-stage absorbers and returns the derivative in the frame its caller names.
:meth:`~StageSource.score_quadrature` turns the last accepted stage's two projections
into the quadrature alarm. :class:`SourceTables` gathers the arrays the passes read.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex, Float, Inexact, Num

from solprop.backends import tdse_source
from solprop.backends.node_chunk import (
    NodeChunkPlan,
    general_planned_chunk,
    plan_node_chunk,
)
from solprop.backends.numpy_ref.general_modal import GeneralMaps, general_source
from solprop.backends.numpy_ref.passes.plasma import (
    envelope_sources,
    held_plasma_spectrum,
)
from solprop.backends.numpy_ref.passes.responses import (
    NonlinearSources,
    mixture_raman_spectrum,
    nonlinear_sources,
    raman_spectrum,
    tensor_raman_spectrum,
)
from solprop.backends.numpy_ref.passes.stepping import absorber_term, stage_output
from solprop.backends.numpy_ref.passes.transforms import (
    project,
    synthesise,
    to_frequency,
    to_time,
)
from solprop.backends.spatial import (
    radial_absorber_matrix,
    slab_columns,
    slab_rows,
)
from solprop.exceptions import BackendError
from solprop.fourier import GridKind
from solprop.translate.general import (
    node_delayed_maps,
    node_kerr_form,
    node_kerr_tensor,
)
from solprop.translate.plasma_source import _BENCHMARK_GATED
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import tdse_of
from solprop.translate.spatial import separable_values
from solprop.translate.stage import StageScalars
from solprop.translate.tdse import TDSEBinding, TDSEPurpose

__all__ = ["SourceTables", "StageSource"]

logger = logging.getLogger(__name__)

_State = Complex[np.ndarray, "n_modes n_pol n_omega"]


class StageSource:
    """The reference's evaluation of the stage derivative, and what it keeps.

    Built once by the workspace from the validated problem. It reads no state of the
    run: the lab-frame field and the stage's scalars are its arguments. It keeps the
    problem's tables, the last stage's projections (for the quadrature alarm), the
    damped Raman kernel of the last decay rates, and, with a bound TDSE response, the
    accounting of the last evaluation.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The validated problem.
    chunk_plan : solprop.backends.node_chunk.NodeChunkPlan
        The node chunk the response passes hold at once.
    prefactor : callable
        ``bundle -> prefactor``: the workspace's evaluation of the response prefactor
        from a stage's scalars
        (:meth:`~solprop.backends.numpy_ref.workspace.NumpyWorkspace.prefactor`).
    tdse_binding : solprop.translate.tdse.TDSEBinding or None
        The host response binding a TDSE problem declares; ``None`` otherwise.

    Attributes
    ----------
    collar_quiet : bool
        Whether the modal stage leaves a quiet in-stage temporal collar out (SEP 21):
        set by the session through the workspace, read by every stage.
    """

    def __init__(
        self,
        problem: PropagationProblem,
        *,
        chunk_plan: NodeChunkPlan,
        prefactor: Callable[[StageScalars], _State],
        tdse_binding: TDSEBinding | None,
    ) -> None:
        """Gather the tables and caches the stages read."""
        self._problem = problem
        self._node_chunk = chunk_plan.chunk
        self._prefactor = prefactor
        self._tdse_binding = tdse_binding
        self.collar_quiet = False
        self._gated_plasma = _BENCHMARK_GATED.get()
        self._kind: GridKind = problem.grid_kind
        self._tables = SourceTables.of(problem)
        self._carrier = _harmonic_carrier(problem)
        self._embedded = _embedded_projection(problem)
        self._general = _general_maps(problem, chunk_plan)
        self._general_responds = self._general is not None and (
            bool(np.any(self._general.tensor)) or self._general.dispersive is not None
        )
        # ADR-025's in-stage scheme is a second term on every stage rather than a
        # factor on the accepted state, so a problem that has no response at all still
        # has a derivative when it is active.
        self._spatial_absorber = radial_absorber_matrix(
            problem.transform, problem.grids.pad_scale
        )
        self._in_stage_absorber = bool(self._spatial_absorber.size) or (
            problem.responses.temporal_absorber_fine is not None
            and problem.responses.absorber_scheme == "nonlinear"
        )
        geometry = problem.transform.geometry
        if geometry is not None and geometry.kind in ("free", "slab"):
            self._in_stage_absorber |= bool(np.any(geometry.absorber_rate))
        # The last stage's projections under both rules, its bundle and its held
        # streams, for the quadrature alarm.
        self._last_projected: Num[np.ndarray, "n_modes n_pol n_to"] | None = None
        self._last_coarse: Num[np.ndarray, "n_modes n_pol n_to"] | None = None
        self._last_bundle: StageScalars | None = None
        self._last_plasma: (
            tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None
        ) = None
        self._last_envelope: (
            tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None
        ) = None
        self._raman_rates: tuple[float, ...] | None = None
        self._raman_densities: tuple[float, ...] | None = None
        self._raman_kernel: Complex[np.ndarray, "... n_omega_doubled"] | None = None
        self._raman_transforms = 0
        self._alarm = math.nan
        self._tdse_last_purpose: TDSEPurpose | None = None
        self._tdse_last_z: float | None = None
        self._tdse_last_projection: dict[str, object] | None = None

    @property
    def alarm(self) -> float:  # numpydoc ignore=RT01
        """The quadrature alarm :meth:`score_quadrature` last scored, or NaN."""
        return self._alarm

    def reset_alarm(self) -> None:
        """Forget the alarm, so that the next plane reports its own."""
        self._alarm = math.nan

    @property
    def general(self) -> GeneralMaps | None:  # numpydoc ignore=RT01
        """The general modal route's maps, or ``None`` on another route."""
        return self._general

    @property
    def embedded(self) -> bool:  # numpydoc ignore=RT01
        """Whether the quadrature has an embedded coarse rule to score the alarm on."""
        return self._embedded is not None

    @property
    def spatial_absorber_bytes(self) -> int:  # numpydoc ignore=RT01
        """The radial collar's matrix size, which the memory ledger names."""
        return self._spatial_absorber.nbytes

    @property
    def gated_plasma(self) -> bool:  # numpydoc ignore=RT01
        """Whether the private gated plasma benchmark replaces the held current."""
        return self._gated_plasma

    @property
    def tdse_last_evaluation(self) -> dict[str, object]:  # numpydoc ignore=RT01
        """The bound TDSE response's last evaluation: purpose, position, projection."""
        return {
            "purpose": self._tdse_last_purpose,
            "z": self._tdse_last_z,
            "projection": self._tdse_last_projection,
        }

    def close(self) -> None:
        """Release the caches. Idempotent."""
        self._raman_kernel = None
        self._raman_rates = None
        self._raman_densities = None
        self._last_plasma = None
        self._spatial_absorber = np.empty((0, 0))

    def evaluate(
        self,
        lab: Complex[np.ndarray, "n_modes n_pol n_omega"],
        bundle: StageScalars,
        inverse_propagator: Complex[np.ndarray, "n_modes n_pol n_omega"],
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Evaluate one stage derivative through the transform passes.

        The pipeline of ``architecture.md`` section 8 for any node count: one batched
        inverse transform of every modal column, then, per chunk of
        the node chunk's nodes, the synthesis GEMM, the fused response pass and
        the projection GEMM accumulated over the chunks, then one batched forward
        transform, the crop and the prefactor. Mode-averaged propagation is the
        one-node, one-column case of exactly this code.

        Parameters
        ----------
        lab : numpy.ndarray
            The stage's lab-frame field, in the state's layout.
        bundle : solprop.translate.stage.StageScalars
            The stage's scalars.
        inverse_propagator : numpy.ndarray
            The factor that carries the derivative back to the step's anchor.
        purpose : {'startup', 'predictor', 'corrected', 'diagnostic'}, optional
            The host source's accounting label; ignored when none is bound.

        Returns
        -------
        numpy.ndarray
            The stage derivative in the anchor's frame.
        """
        responses = self._problem.responses
        grids = self._problem.grids
        if self._tdse_binding is not None:
            return self._nonlinear_tdse(lab, bundle, inverse_propagator, purpose)
        if self._problem.linear.separable is not None:
            geometry = self._problem.transform.geometry
            evaluate = (
                self._nonlinear_slab
                if geometry is not None and geometry.kind == "slab"
                else self._nonlinear_free
            )
            return evaluate(lab, bundle, inverse_propagator)
        if self._general is not None:
            return self._nonlinear_general(lab, bundle, inverse_propagator)
        if len(responses.response_order) == 1:
            # Nothing but the accumulation was compiled, so the polarisation is
            # identically zero and the step is the exact linear flow. Running the
            # transforms to multiply by nothing would only add round-off -- unless
            # ADR-025's in-stage absorber is active, which is a term of its own and
            # would otherwise be dropped from a problem that has no response at all.
            if not self._in_stage_absorber:
                return np.zeros_like(lab)
            alone = to_time(lab, grids.crop_index, grids.n_to, kind=self._kind)
            return stage_output(
                np.zeros_like(lab),
                np.zeros_like(lab),
                self._tables.omega_window,
                inverse_propagator,
                self._absorber_term(alone, may_quiet=True),
            )
        modal_t = to_time(lab, grids.crop_index, grids.n_to, kind=self._kind)
        geometry = self._problem.transform.geometry
        if (
            geometry is not None
            and geometry.kind == "radial"
            and bool(responses.plasma_channels)
            and not self._gated_plasma
        ):
            spectrum = self._radial_plasma_spectrum(modal_t, bundle)
        else:
            projected, coarse = self._over_nodes(modal_t, bundle)
            self._last_projected = projected
            self._last_coarse = coarse
            self._last_bundle = bundle
            spectrum = to_frequency(projected, grids.crop_index, kind=self._kind)
            if self._last_plasma is not None:
                gated, edge, _, _ = self._last_plasma
                spectrum += self._held_streams(gated, edge)
            if responses.envelope_plasma is not None:
                assert self._last_envelope is not None
                drive, loss, _, _ = self._last_envelope
                factors = responses.envelope_plasma.factors
                spectrum += factors.drude * to_frequency(
                    drive, grids.crop_index, kind=self._kind
                )
                spectrum += factors.loss * to_frequency(
                    loss, grids.crop_index, kind=self._kind
                )
        return stage_output(
            spectrum,
            self._prefactor(bundle),
            self._tables.omega_window,
            inverse_propagator,
            self._absorber_term(modal_t, may_quiet=True),
        )

    def _nonlinear_tdse(
        self,
        lab: Complex[np.ndarray, "n_modes n_pol n_omega"],
        bundle: StageScalars,
        inverse_propagator: Complex[np.ndarray, "n_modes n_pol n_omega"],
        purpose: TDSEPurpose,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Evaluate the declared atomic binding and combine independent P/J sources.

        Steps 1 to 4 are :mod:`solprop.backends.tdse_source`'s, shared with native;
        the stage output is this backend's own.
        """
        problem = self._problem
        grids = problem.grids
        binding = self._tdse_binding
        assert binding is not None and tdse_of(problem.route) is not None
        modal_t = to_time(lab, grids.crop_index, grids.n_to, kind="field")
        fields = tdse_source.node_fields(problem, lab, bundle)
        result = binding.evaluate(
            fields, z=bundle.z1, purpose=purpose, density=bundle.density
        )
        polarisation, current = tdse_source.atomic_sources(
            fields, result, problem, bundle
        )
        source, projection = tdse_source.project_atomic_sources(
            polarisation,
            current,
            result,
            problem,
            bundle,
            self._prefactor(bundle),
            edge=tdse_source.atomic_edge_source(result, problem, bundle),
        )
        self._tdse_last_purpose, self._tdse_last_z = purpose, bundle.z1
        self._tdse_last_projection = projection
        return stage_output(
            source,
            np.ones_like(source),
            grids.omega_window,
            inverse_propagator,
            self._absorber_term(modal_t),
        )

    def _nonlinear_general(
        self,
        lab: Complex[np.ndarray, "n_modes n_pol n_omega"],
        bundle: StageScalars,
        inverse_propagator: Complex[np.ndarray, "n_modes n_pol n_omega"],
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Evaluate one stage derivative through a general modal guide's maps.

        The route's own execution path, in
        :mod:`solprop.backends.numpy_ref.general_modal`, beside the fused pass and
        not an extension of it. The synthesis is a complex
        frequency-dependent map of three Cartesian components, so it happens in the
        frequency domain and the field a response sees is the analytic signal. What
        is shared with the channel node pipeline is everything after the projection
        -- the prefactor, which is the crop scale alone here, the spectral collar,
        the in-stage absorber and the return to the interaction picture.

        The quadrature alarm is not scored on this route: the general node rule has
        no embedded coarse pair, and a difference that is identically zero read as
        convergence would be wrong, as for any unpaired rule.

        A guide with no responding region is short-circuited, as the channel node
        pipeline short-circuits a problem whose `response_order` is the accumulation
        alone.
        """
        assert self._general is not None
        grids = self._problem.grids
        # A guide whose every material region is linear has an identically zero
        # tensor, so the source is zero for every state and every position: the
        # synthesis, the 3 x n_nodes transforms, the contraction and the projection
        # would spend a tenth of a second per step producing it (measured 108 ms on
        # the 288-node strand). ADR-025's in-stage absorber is a term of its own and
        # is not dropped with it.
        spectrum = (
            general_source(
                lab,
                self._general,
                grids.crop_index,
                n_to=grids.n_to,
                pad_scale=grids.pad_scale,
                transform_weights=bundle.transform_weights,
                dispersive_densities=bundle.dispersive_kerr_densities,
            )
            if self._general_responds
            else np.zeros_like(lab)
        )
        absorber = None
        if self._in_stage_absorber:
            absorber = self._absorber_term(
                to_time(lab, grids.crop_index, grids.n_to, kind=self._kind)
            )
        return stage_output(
            spectrum,
            self._prefactor(bundle),
            grids.omega_window,
            inverse_propagator,
            absorber,
        )

    def _nonlinear_free(
        self,
        lab: np.ndarray,
        bundle: StageScalars,
        inverse_propagator: np.ndarray,
    ) -> np.ndarray:
        """Apply transverse FFTs before bounded temporal response transforms.

        - ``lab``: Lab-frame stage spectrum.
        """
        problem = self._problem
        transform, grids, responses = (
            problem.transform,
            problem.grids,
            problem.responses,
        )
        geometry = transform.geometry
        if geometry is None:
            msg = "Cartesian nonlinear evaluation requires spatial geometry"
            raise BackendError(msg)
        shape = (geometry.shape[1], geometry.shape[0], *lab.shape[1:])
        result = np.zeros_like(lab)
        if len(responses.response_order) == 1 and not self._in_stage_absorber:
            return result
        scratch = np.empty_like(lab)
        if len(responses.response_order) > 1:
            scratch.reshape(shape)[...] = np.fft.ifft2(lab.reshape(shape), axes=(0, 1))
            raman = self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
            for first in range(0, transform.n_nodes, self._node_chunk):
                rows = slice(first, first + self._node_chunk)
                field_t = grids.pad_scale * to_time(
                    scratch[rows], grids.crop_index, grids.n_to, kind=self._kind
                )
                sources = self._column_sources(field_t, bundle, raman)
                polarisation_t = sources.polarisation
                scratch[rows] = to_frequency(
                    polarisation_t, grids.crop_index, kind=self._kind
                )
                if sources.current is not None:
                    scratch[rows] += self._held_current(sources.current)
                if responses.envelope_plasma is not None:
                    drive, loss = self._envelope_sources(field_t, bundle)
                    factors = responses.envelope_plasma.factors
                    scratch[rows] += factors.drude * to_frequency(
                        drive, grids.crop_index, kind=self._kind
                    )
                    scratch[rows] += factors.loss * to_frequency(
                        loss, grids.crop_index, kind=self._kind
                    )
            view = scratch.reshape(shape)
            view[...] = np.fft.fft2(view, axes=(0, 1))
            data = problem.linear.separable
            assert data is not None
            for first in range(0, transform.n_nodes, self._node_chunk):
                rows = slice(first, first + self._node_chunk)
                _, prefactor = separable_values(data, rows)
                result[rows] = prefactor * self._tables.omega_window * scratch[rows]
        if self._in_stage_absorber:
            # The rate is additive to the normalised material term. Reusing the
            # transverse scratch avoids a second retained physical spectral volume.
            scratch.reshape(shape)[...] = np.fft.ifft2(lab.reshape(shape), axes=(0, 1))
            temporal = self._tables.temporal_rate
            timed = temporal is not None and responses.absorber_scheme == "nonlinear"
            for first in range(0, transform.n_nodes, self._node_chunk):
                rows = slice(first, first + self._node_chunk)
                rate = geometry.absorber_rate[rows, None, None]
                if not timed:
                    # A rate of the node alone commutes with the temporal transforms:
                    # a spatial-only pass needs no temporal pair (SEP 40 section 4.7).
                    scratch[rows] = -0.5 * rate * scratch[rows]
                    continue
                field_t = to_time(
                    scratch[rows], grids.crop_index, grids.n_to, kind=self._kind
                )
                scratch[rows] = to_frequency(
                    -0.5 * (rate + temporal) * field_t,
                    grids.crop_index,
                    kind=self._kind,
                )
            view = scratch.reshape(shape)
            view[...] = np.fft.fft2(view, axes=(0, 1))
            result += grids.active_mask * scratch
        return result * inverse_propagator

    def _nonlinear_slab(
        self,
        lab: np.ndarray,
        bundle: StageScalars,
        inverse_propagator: np.ndarray,
    ) -> np.ndarray:
        """Evaluate one hybrid planar stage: slab synthesis in x, an FFT in y.

        The ``(p, k_y)`` columns go to ``(p, y)`` rows by the inverse DFT in y, and
        the rows to time -- a row is a slab amplitude, whose time signal is a real
        field's on a field grid -- so that, as on the modal route, only ``n_p * n_y``
        temporal transforms are taken each way. In time, the slab synthesis (which
        carries the pad scale) gives the field at the ``(x, y)`` nodes, the responses
        are the Cartesian ones node by node, and the slab rule's dual basis projects
        the polarisation back onto the rows; the rows return to frequency and to
        ``(p, k_y)``, and each column takes its own prefactor from the separable leaf
        (kernel schema 15). The y collar's in-stage rate depends on y alone, so it
        acts on the rows directly.

        - ``lab``: Lab-frame stage spectrum.
        """
        problem = self._problem
        transform, grids, responses = (
            problem.transform,
            problem.grids,
            problem.responses,
        )
        geometry = transform.geometry
        assert geometry is not None
        result = np.zeros_like(lab)
        if len(responses.response_order) == 1 and not self._in_stage_absorber:
            return result
        n_x, n_y = geometry.shape
        n_p = transform.synthesis.shape[1]
        rows_f = slab_rows(transform, lab)
        temporal = responses.temporal_absorber_fine
        timed = temporal is not None and responses.absorber_scheme == "nonlinear"
        rows_t = (
            to_time(rows_f, grids.crop_index, grids.n_to, kind=self._kind)
            if len(responses.response_order) > 1 or timed
            else None
        )
        if len(responses.response_order) > 1:
            assert rows_t is not None
            raman = self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
            envelope = responses.envelope_plasma is not None
            plasma = bool(responses.plasma_channels) and not self._gated_plasma
            projected = np.zeros_like(rows_t)
            # The envelope plasma's Drude drive and loss current, projected as the
            # polarisation is; empty unless the response is present.
            drives = [
                np.zeros_like(rows_t) for _ in range(2 if envelope or plasma else 0)
            ]
            # Whole y rows at a time, as many as the node chunk holds.
            block = max(1, self._node_chunk // n_x)
            for first in range(0, n_y, block):
                rows = slice(first * n_p, min(first + block, n_y) * n_p)
                count = (rows.stop - rows.start) // n_p
                field_t = np.matmul(
                    transform.synthesis, rows_t[rows].reshape(count, n_p, -1)
                ).reshape(count * n_x, *rows_t.shape[1:])
                sources = self._column_sources(field_t, bundle, raman)
                current = sources.current
                source_rows = [sources.polarisation]
                if envelope:
                    assert responses.envelope_plasma is not None
                    source_rows.extend(
                        envelope_sources(
                            field_t,
                            responses.envelope_plasma,
                            bundle,
                            grids.dt_fine,
                            grids.t_window_fine,
                        )
                    )
                if current is not None:
                    source_rows.extend(
                        (
                            grids.t_window_fine * current,
                            grids.t_window_fine_derivative * current,
                        )
                    )
                for target, source in zip(
                    (projected, *drives), source_rows, strict=True
                ):
                    target[rows] = np.matmul(
                        transform.projection, source.reshape(count, n_x, -1)
                    ).reshape(count * n_p, *rows_t.shape[1:])
            spectrum = to_frequency(projected, grids.crop_index, kind=self._kind)
            if envelope:
                assert responses.envelope_plasma is not None
                factors = responses.envelope_plasma.factors
                spectrum += factors.drude * to_frequency(
                    drives[0], grids.crop_index, kind=self._kind
                )
                spectrum += factors.loss * to_frequency(
                    drives[1], grids.crop_index, kind=self._kind
                )
            if plasma:
                spectrum += self._held_streams(drives[0], drives[1])
            columns = slab_columns(transform, spectrum)
            data = problem.linear.separable
            assert data is not None
            for first in range(0, transform.n_modes, self._node_chunk):
                rows = slice(first, first + self._node_chunk)
                _, prefactor = separable_values(data, rows)
                result[rows] = prefactor * grids.omega_window * columns[rows]
        if self._in_stage_absorber:
            result += self._slab_absorber(rows_f, rows_t)
        return result * inverse_propagator

    def _over_nodes(
        self,
        modal_t: Inexact[np.ndarray, "n_modes n_pol n_to"],
        bundle: StageScalars,
    ) -> tuple[
        Num[np.ndarray, "n_modes n_pol n_to"],
        Num[np.ndarray, "n_modes_c n_pol_c n_to_c"] | None,
    ]:
        """Run the response pass over the quadrature in chunks and project it back.

        The chunk loop of ``architecture.md`` section 13: the space-domain buffers are
        sized by the node chunk's and not by the node count, so a chunk of one
        and a chunk of every node differ only in round-off. The embedded coarse rule is
        projected in the same loop, from the same node polarisation, because that is
        what makes the quadrature alarm free: the two rules share every node the coarse
        one has.

        - Returns: the modal polarisation on the fine time grid, and the same under the
          embedded coarse rule, or ``None`` when the quadrature has no embedded pair
          and the estimate would be identically zero.
        """
        responses, grids = self._problem.responses, self._problem.grids
        transform = self._problem.transform
        taper = bundle.taper_scale
        # Once per stage rather than once per chunk: the kernel spectrum does not depend
        # on the nodes, and along a gradient it is one real transform of 2 n_to against
        # n_modes n_pol transforms of n_to that the stage costs anyway.
        raman = self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
        n_pol = transform.n_pol
        projected = np.zeros(
            (transform.n_modes, self._problem.shape[1], grids.n_to),
            dtype=modal_t.dtype,
        )
        coarse = None if self._embedded is None else np.zeros_like(projected)
        envelope = self._problem.responses.envelope_plasma
        envelope_projected = (
            None
            if envelope is None
            else (
                np.zeros_like(projected),
                np.zeros_like(projected),
                None if coarse is None else np.zeros_like(coarse),
                None if coarse is None else np.zeros_like(coarse),
            )
        )
        plasma_projected = (
            (
                np.zeros_like(projected),
                np.zeros_like(projected),
                None if coarse is None else np.zeros_like(coarse),
                None if coarse is None else np.zeros_like(coarse),
            )
            if bool(responses.plasma_channels) and not self._gated_plasma
            else None
        )
        for first in range(0, transform.n_nodes, self._node_chunk):
            rows = slice(first * n_pol, (first + self._node_chunk) * n_pol)
            field_t = synthesise(modal_t, self._tables.synthesis[rows], taper, n_pol)
            sources = self._column_sources(field_t, bundle, raman)
            polarisation_t = sources.polarisation
            projected = projected + project(
                polarisation_t, self._tables.projection[:, rows], taper, n_pol
            )
            if coarse is not None and self._embedded is not None:
                coarse = coarse + project(
                    polarisation_t, self._embedded[:, rows], taper, n_pol
                )
            if sources.current is not None and plasma_projected is not None:
                gated, edge, coarse_gated, coarse_edge = plasma_projected
                for target, reduced, window in (
                    (gated, coarse_gated, self._tables.t_window),
                    (edge, coarse_edge, self._tables.t_window_derivative),
                ):
                    current = window * sources.current
                    target[...] += project(
                        current, self._tables.projection[:, rows], taper, n_pol
                    )
                    if reduced is not None and self._embedded is not None:
                        reduced[...] += project(
                            current, self._embedded[:, rows], taper, n_pol
                        )
            if envelope is not None and envelope_projected is not None:
                drive, loss = self._envelope_sources(field_t, bundle)
                full_drive, full_loss, coarse_drive, coarse_loss = envelope_projected
                full_drive += project(
                    drive, self._tables.projection[:, rows], taper, n_pol
                )
                full_loss += project(
                    loss, self._tables.projection[:, rows], taper, n_pol
                )
                if (
                    self._embedded is not None
                    and coarse_drive is not None
                    and coarse_loss is not None
                ):
                    coarse_drive += project(
                        drive, self._embedded[:, rows], taper, n_pol
                    )
                    coarse_loss += project(loss, self._embedded[:, rows], taper, n_pol)
        self._last_plasma = plasma_projected
        self._last_envelope = envelope_projected
        return projected, coarse

    def _radial_plasma_spectrum(
        self,
        modal_t: np.ndarray,
        bundle: StageScalars,
    ) -> np.ndarray:
        """Transform three nodal streams, combine, then project once in frequency."""
        problem = self._problem
        transform, grids = problem.transform, problem.grids
        spectrum = np.zeros(problem.shape, dtype=np.complex128)
        raman = self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
        for first in range(0, transform.n_nodes, self._node_chunk):
            rows = slice(
                first * transform.n_pol,
                (first + self._node_chunk) * transform.n_pol,
            )
            field = synthesise(
                modal_t,
                self._tables.synthesis[rows],
                bundle.taper_scale,
                transform.n_pol,
            )
            sources = self._column_sources(field, bundle, raman)
            assert sources.current is not None
            nodal = to_frequency(sources.polarisation, grids.crop_index, kind="field")
            nodal += self._held_current(sources.current)
            spectrum += project(
                nodal,
                self._tables.projection[:, rows],
                bundle.taper_scale,
                transform.n_pol,
            )
        return spectrum

    def _column_sources(
        self,
        field_t: np.ndarray,
        bundle: StageScalars,
        raman: np.ndarray | None,
    ) -> NonlinearSources:
        """Evaluate one column; the historical integral is benchmark-only.

        - ``field_t``: physical node fields in V/m, with time last.
        - ``raman``: damped Raman convolution spectrum, when required by the response.
        - Returns: the collared local polarisation in C/m² and the ungated current in
          A/m².
        """
        responses, grids = self._problem.responses, self._problem.grids
        sources = nonlinear_sources(
            field_t,
            responses,
            self._tables.t_window,
            kind=self._kind,
            density=bundle.density,
            preionisation=bundle.preionisation,
            dt=grids.dt_fine,
            carrier=self._carrier,
            raman_spectrum=raman,
            scalars=bundle,
        )
        if not self._gated_plasma or sources.current is None:
            return sources
        return NonlinearSources(
            sources.polarisation
            + self._tables.t_window
            * responses.plasma_integral.integrate(sources.current, grids.dt_fine),
            None,
        )

    def _held_streams(self, gated: np.ndarray, edge: np.ndarray) -> np.ndarray:
        """Transform and combine the current products in polarisation units.

        - ``gated``, ``edge``: fine-time ``gJ`` and ``g'J`` rows, optionally projected
          before this call.
        - Returns: cropped raw-DFT polarisation-equivalent source, zero on inactive
          bins.
        """
        grids = self._problem.grids
        return held_plasma_spectrum(
            gated, edge, grids.omega, grids.crop_index, grids.active_mask
        )

    def _held_current(self, current: np.ndarray) -> np.ndarray:
        """Form and transform the two collared products of a nodal current.

        - ``current``: the ungated nodal current in A/m², fine time last.
        - Returns: cropped raw-DFT polarisation-equivalent source at the nodes.
        """
        return self._held_streams(
            self._tables.t_window * current, self._tables.t_window_derivative * current
        )

    def _envelope_sources(
        self, field_t: Complex[np.ndarray, "n_nodes n_pol n_to"], bundle: StageScalars
    ) -> tuple[
        Complex[np.ndarray, "n_nodes n_pol n_to"],
        Complex[np.ndarray, "n_nodes n_pol n_to"],
    ]:
        """Return the envelope-grid plasma's windowed drive and loss at some nodes.

        - ``field_t``: the field at the nodes, V/m.
        """
        envelope = self._problem.responses.envelope_plasma
        assert envelope is not None
        return envelope_sources(
            field_t,
            envelope,
            bundle,
            self._problem.grids.dt_fine,
            self._tables.t_window,
        )

    def _raman_spectrum(
        self, rates: tuple[float, ...], densities: tuple[float, ...] = ()
    ) -> Complex[np.ndarray, "... n_omega_doubled"] | None:
        """Return the damped Raman kernel spectrum for a set of rates, anew.

        Memoised on the **decay rates** rather than on the density, because the rates
        are what the spectrum depends on: a glass's are zero at every density, so its
        kernel is transformed once for the whole run, and a molecular gas on a gradient
        presents new rates at every stage and is transformed once per stage. Only the
        most recent set is kept, which is the right choice here -- a growing cache would
        hold thousands of spectra and hit none.

        - ``rates``: the decay rate of each term at the stage's position, s^-1, off the
          bundle.
        - ``densities``: component amplitude densities in m^-3, in Raman-component
          order. Multiple components include every amplitude in the cache key; one keeps
          density outside the convolution and retains the original arithmetic.
        - Returns: the spectrum of shape ``(n_to + 1,)``, or ``None`` when no Raman
          kernel was compiled.
        """
        components = self._problem.responses.raman_terms
        if not components:
            return None
        amplitudes = densities if len(components) > 1 else ()
        if (
            self._raman_kernel is None
            or self._raman_rates != rates
            or self._raman_densities != amplitudes
        ):
            if self._problem.transform.n_pol * self._problem.shape[1] == 2:
                self._raman_kernel = tensor_raman_spectrum(
                    components,
                    rates,
                    densities if len(components) > 1 else (1.0,),
                    self._problem.grids.dt_fine,
                )
            elif len(components) == 1:
                compiled = components[0]
                self._raman_kernel = raman_spectrum(
                    compiled.undamped,
                    rates,
                    compiled.fraction,
                    self._problem.grids.dt_fine,
                )
            else:
                self._raman_kernel = mixture_raman_spectrum(
                    components, rates, densities, self._problem.grids.dt_fine
                )
            self._raman_rates = rates
            self._raman_densities = amplitudes
            self._raman_transforms += 1
        return self._raman_kernel

    def _absorber_term(
        self,
        modal_t: Inexact[np.ndarray, "n_modes n_pol n_to"],
        *,
        may_quiet: bool = False,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"] | None:
        """Return the in-stage absorber's contribution, or ``None`` in the other scheme.

        On the modal stage (`may_quiet`) a temporal collar alone is left out while the
        boundary monitor finds it quiet (SEP 21): the term is then below
        :data:`solprop.monitor.QUIET`, and its transform is the stage's largest cost
        after the response's.

        - Returns: the absorber's term on the coarse grid, or ``None`` when the
          ``"split_step"`` scheme is active and the rate is applied to the state after
          the step instead.
        """
        if not self._in_stage_absorber:
            return None
        radial = self._spatial_absorber if self._spatial_absorber.size else None
        if may_quiet and self.collar_quiet and radial is None:
            return None
        rate = self._tables.temporal_rate
        in_stage = self._problem.responses.absorber_scheme == "nonlinear"
        grids = self._problem.grids
        return absorber_term(
            modal_t,
            rate if in_stage else None,
            radial,
            grids.crop_index,
            grids.active_mask,
            kind=self._kind,
        )

    def _slab_absorber(
        self,
        rows_f: Complex[np.ndarray, "n_modes n_pol n_omega"],
        rows_t: Inexact[np.ndarray, "n_modes n_pol n_to"] | None,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the planar in-stage collar's term on the ``(p, k_y)`` columns.

        The collar's rate depends on y alone, so it acts on the ``(p, y)`` rows. With
        no temporal rate in-stage it commutes with the temporal transforms, and the
        pass needs no temporal pair (SEP 40 section 4.7).

        - ``rows_t``: the same rows in time; required when a temporal rate applies
          in-stage.
        """
        problem = self._problem
        transform, grids, responses = (
            problem.transform,
            problem.grids,
            problem.responses,
        )
        geometry = transform.geometry
        assert geometry is not None
        n_x, n_y = geometry.shape
        n_p = transform.synthesis.shape[1]
        rate = np.repeat(geometry.absorber_rate.reshape(n_y, n_x)[:, 0], n_p)[
            :, None, None
        ]
        temporal = responses.temporal_absorber_fine
        if temporal is not None and responses.absorber_scheme == "nonlinear":
            assert rows_t is not None
            term = to_frequency(
                -0.5 * (rate + temporal) * rows_t, grids.crop_index, kind=self._kind
            )
        else:
            term = -0.5 * rate * rows_f
        return grids.active_mask * slab_columns(transform, term)

    def score_quadrature(self) -> tuple[float, float]:
        """Score the quadrature from the last stage of an accepted step.

        The relative difference between the projections of the same node polarisation
        with the Kronrod rule and with its embedded Gauss subset, at the cost of one
        extra forward transform per accepted step rather than a whole second evaluation
        of the transform.
        It is a **statistic and a warning, never a controller input**: a
        quadrature too coarse for the integrand is a modelling error the step size
        cannot fix, and feeding it to the controller would only shrink the step until
        the run stopped.

        The value is ``nan`` when the rule has no embedded pair, because the difference
        is then identically zero and reading that as convergence would be wrong; and
        ``nan`` where the polarisation itself vanishes, at a true vacuum end, because a
        relative measure of nothing says nothing. The session reports a value above
        :data:`~solprop.translate.modal.QUADRATURE_ALARM` once per plane, and the
        ``quadrature_error`` statistic reads it.

        Returns
        -------
        float
            The alarm, dimensionless, or NaN.
        float
            Where the last stage was evaluated, m, or NaN without one.
        """
        if (
            self._last_coarse is None
            or self._last_projected is None
            or self._last_bundle is None
        ):
            self._alarm = math.nan
            return self._alarm, math.nan
        grids = self._problem.grids
        weight = self._prefactor(self._last_bundle) * self._tables.omega_window
        fine = weight * to_frequency(
            self._last_projected, grids.crop_index, kind=self._kind
        )
        coarse = weight * to_frequency(
            self._last_coarse, grids.crop_index, kind=self._kind
        )
        if self._last_envelope is not None:
            assembly = self._problem.responses.envelope_plasma
            assert assembly is not None
            drive, loss, coarse_drive, coarse_loss = self._last_envelope
            assert coarse_drive is not None and coarse_loss is not None
            for factor, full, reduced in (
                (assembly.factors.drude, drive, coarse_drive),
                (assembly.factors.loss, loss, coarse_loss),
            ):
                fine += (
                    weight
                    * factor
                    * to_frequency(full, grids.crop_index, kind=self._kind)
                )
                coarse += (
                    weight
                    * factor
                    * to_frequency(reduced, grids.crop_index, kind=self._kind)
                )
        if self._last_plasma is not None:
            gated, edge, coarse_gated, coarse_edge = self._last_plasma
            assert coarse_gated is not None and coarse_edge is not None
            fine += weight * self._held_streams(gated, edge)
            coarse += weight * self._held_streams(coarse_gated, coarse_edge)
        scale = float(np.linalg.norm(fine))
        self._alarm = (
            math.nan if scale == 0.0 else float(np.linalg.norm(coarse - fine) / scale)
        )
        return self._alarm, self._last_bundle.z1


@dataclass(frozen=True)
class SourceTables:
    """The problem's tables a source evaluation multiplies by.

    The arrays the passes read, gathered once from the problem.

    Attributes
    ----------
    synthesis : numpy.ndarray
        ``transform.synthesis``.
    projection : numpy.ndarray
        ``transform.projection``.
    t_window : numpy.ndarray
        ``grids.t_window_fine``.
    t_window_derivative : numpy.ndarray
        ``grids.t_window_fine_derivative``.
    omega_window : numpy.ndarray
        ``grids.omega_window``.
    temporal_rate : numpy.ndarray or None
        ``responses.temporal_absorber_fine``.
    """

    synthesis: Float[np.ndarray, "n_rows n_modes_s"]
    projection: Float[np.ndarray, "n_modes n_columns"]
    t_window: Float[np.ndarray, " n_to"]
    t_window_derivative: Float[np.ndarray, " n_to"]
    omega_window: Float[np.ndarray, " n_omega"]
    temporal_rate: Float[np.ndarray, " n_to"] | None

    @classmethod
    def of(cls, problem: PropagationProblem) -> "SourceTables":
        """Return the problem's own tables.

        Parameters
        ----------
        problem : solprop.translate.problem.PropagationProblem
            The problem.

        Returns
        -------
        SourceTables
            References to its arrays, uncopied.
        """
        grids = problem.grids
        return cls(
            problem.transform.synthesis,
            problem.transform.projection,
            grids.t_window_fine,
            grids.t_window_fine_derivative,
            grids.omega_window,
            problem.responses.temporal_absorber_fine,
        )


def _general_maps(
    problem: PropagationProblem, plan: NodeChunkPlan | None = None
) -> GeneralMaps | None:
    """Derive the general modal route's maps, or return ``None`` for another route.

    The maps are read out of the leaves once, here, because the route propagates
    constant geometry: one interval of one Chebyshev degree along z, so there is
    nothing to evaluate per stage. The chunk is the backend's own memory strategy
    within the plan's budget, or the plan's exact chunk when one was forced.

    - ``plan``: the backend's node-chunk plan; ``None`` plans the automatic one.
    - Returns: the derived maps, or ``None`` when the problem carries no modal tables.
    """
    tables = problem.transform.modal_tables
    if tables is None:
        return None
    assert tables.synthesis is not None and tables.projection is not None
    chunk = general_planned_chunk(
        problem, plan_node_chunk(problem) if plan is None else plan
    )
    regional = problem.responses.regional_responses
    owner, delayed_tensor, delayed_spectrum = node_delayed_maps(tables, regional)
    moving = tables.synthesis.shape[0] * tables.synthesis.shape[1] > 1
    maps = GeneralMaps(
        tables.synthesis[0, 0],
        tables.projection[0, 0],
        node_kerr_tensor(tables, regional),
        tables.source_frequency_factor,
        chunk,
        node_kerr_form(regional),
        owner,
        delayed_tensor,
        delayed_spectrum,
        # A moving guide's coefficients, flattened over (n_intervals, n_deg) in the
        # order the stage bundle's transform_weights are.
        tables.synthesis.reshape(-1, *tables.synthesis.shape[2:]) if moving else None,
        tables.projection.reshape(-1, *tables.projection.shape[2:]) if moving else None,
        problem.responses.dispersive_kerr,
        None if problem.responses.dispersive_kerr is None else tables.region_index,
    )
    logger.debug(
        "numpy_ref general modal route: %d channels on %d nodes, chunk %d, "
        "%d fine bins, gauge %s, %s cubic, %d delayed term(s)",
        maps.n_modes,
        maps.n_nodes,
        maps.node_chunk,
        int(tables.omega_fine.size),
        tables.gauge_id,
        maps.form,
        maps.n_terms,
    )
    return maps


def _harmonic_carrier(
    problem: PropagationProblem,
) -> Complex[np.ndarray, " n_to"] | None:
    """Return ``exp(2 i omega_0 t)`` on the fine grid when the harmonic is radiated.

    The envelope-grid third-harmonic term needs the carrier the coarse grid does not
    resolve; it arrives as ``grids.carrier_2w0``, so that no physics is formed inside a
    backend (``docs/dev/architecture.md`` section 3). The sign is the FFT convention's
    (``docs/guide/media/nonlinear.md``). ``validate`` refuses a harmonic term without
    the carrier, so only a hand-built problem reaches the refusal here.

    - Returns: the carrier of shape ``(n_to,)``, or ``None`` when no harmonic term is
      present.
    """
    if not problem.responses.harmonic_carrier:
        return None
    if problem.grids.carrier_2w0 is None:
        msg = (
            "a third-harmonic Kerr term is compiled and grids.carrier_2w0 is absent; "
            "the envelope's harmonic sits an offset 2 omega_0 from its carrier and the "
            "response pass cannot place it without the factor"
        )
        raise BackendError(msg)
    return problem.grids.carrier_2w0


def _embedded_projection(
    problem: PropagationProblem,
) -> Float[np.ndarray, "n_modes n_columns"] | None:
    """Return the embedded coarse-rule projection, when the quadrature has one.

    ``transform.projection_coarse`` equals ``transform.projection`` when the rule has no
    embedded pair, in which case the difference of the two projections is identically
    zero and reading it as convergence would be wrong
    (:class:`solprop.modes.quadrature.ModeMatrices`). The alarm reports ``nan`` there
    instead, and the second projection is not computed at all.

    - Returns: the coarse projection, or ``None`` when it is the fine one.
    """
    transform = problem.transform
    if np.array_equal(transform.projection, transform.projection_coarse):
        return None
    return transform.projection_coarse
