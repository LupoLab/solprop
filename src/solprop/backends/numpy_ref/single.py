# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The single-precision path of the reference backend: SEP 33's oracle.

:class:`NumpySingleBackend` is :class:`~solprop.backends.numpy_ref.NumpyRefBackend`
with the propagated state, the stage values, every transform and the responses held in
IEEE binary32, under the contract of ``docs/dev/seps/sep-0033-single-precision-mlx.md``
section 4.2. It is the target the ``mlx`` backend is compared against, so it keeps the
reference's one-function-per-pass shape and states every order it sums in.

What is binary32 and what is not:

* **The state** is held as ``a / s``, ``s`` the launch's exact power-of-two scale
  (:func:`solprop.translate.precision.state_scale`). :meth:`observe` and the statistics
  read ``s`` times the state, widened to binary64, which is exact.
* **The tables** a source evaluation multiplies by are narrowed once at construction
  (:meth:`_upload`). Each has its block exponent
  (:func:`solprop.translate.precision.block_exponent`); where narrowing the table
  directly keeps every entry the block-exponent form keeps (the table's largest
  magnitude at least one, or no entry below binary32's smallest normal), the exponent is
  applied at upload, exactly, and otherwise the table is refused unless its pass
  carries the exponent itself, as the Raman kernel's does.
* **The propagator** ``exp(dPhi)`` is formed from the binary64 exponent in binary64 and
  rounded once (section 4.1, item 3); every other z-dependent array a pass multiplies by
  -- the prefactor, a z-table's value, the spatial collar's rate -- is evaluated on the
  host in binary64 and rounded once.
* **The responses** run in binary32 on the scaled field. Each takes its homogeneity
  factor (section 4.3, rule 4) in one binary64 coefficient combined on the host
  (:func:`solprop.translate.precision.combine`): ``s**2`` for the cubic terms; the
  plasma is evaluated at the physical field ``s E``, an exact multiplication, and its
  current divided by ``s``. The field-grid plasma uses the stable population forms of
  section 4.4 and blocked running integrals
  (:func:`solprop.translate.precision.prefix_sum`).
* **Host evaluations of the oracle.** The ionisation rate's table lookup, the
  envelope-grid plasma's cycle moments and populations, and the statistics are
  evaluated in binary64 from the binary32 field and rounded (or, for the statistics,
  returned) once: the oracle defines the reference these comparisons are made against.
* **The controller** reads a norm reduced as a scaled sum of squares
  (:func:`solprop.translate.precision.scaled_square_sum`), against an absolute
  tolerance divided by ``s``.
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence

import numpy as np
from jaxtyping import Complex, Float, Inexact, Num

from solprop.backends.numpy_ref import passes
from solprop.backends.numpy_ref.backend import NumpyRefBackend, SourceTables
from solprop.backends.numpy_ref.envelope_plasma import envelope_sources
from solprop.constants import EPS0
from solprop.exceptions import BackendError
from solprop.fourier import GridKind
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.profiles import ZProfile
from solprop.stats import StatsSpec
from solprop.translate.precision import (
    block_exponent,
    combine,
    launch_peak,
    prefix_sum,
    scaled_square_sum,
    state_scale,
)
from solprop.translate.problem import KernelProblem
from solprop.translate.responses import (
    CompiledResponses,
    KerrCoefficients,
    PlasmaChannel,
    ResponseStage,
)
from solprop.translate.stage import StageScalars
from solprop.translate.tdse import TDSEBinding

__all__ = [
    "NumpySingleBackend",
    "running_integral",
    "stable_population",
]

_TINY = float(np.finfo(np.float32).tiny)

# Stages the single-precision source evaluation reaches elsewhere: the rate and the
# population inside the plasma current, the sum and window at the end, and the
# envelope plasma through the backend's own hook.
_ELSEWHERE = frozenset(
    {
        ResponseStage.INTENSITY,
        ResponseStage.IONISATION_RATE,
        ResponseStage.PLASMA_SCAN,
        ResponseStage.ACCUMULATE,
        ResponseStage.ENVELOPE_CYCLE,
        ResponseStage.ENVELOPE_POPULATION,
        ResponseStage.ENVELOPE_SOURCES,
    }
)
_HUGE = float(np.finfo(np.float32).max)


class NumpySingleBackend(NumpyRefBackend):
    """Propagate a single-precision problem: the reference backend in binary32.

    Constructed by the registry for ``backend="numpy_ref"`` when the problem's
    precision is single. See the module docstring for what is held in which precision.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A validated problem with ``precision.single``.
    grid : solprop.grid.TimeGrid, optional
        The grid the state lives on, needed only by :meth:`stats`.
    modes : collections.abc.Sequence, optional
        The propagated modes, needed only by the ``zdw`` statistic.
    profiles : collections.abc.Mapping, optional
        The z-profiles keyed by statistic name.
    tdse_binding : solprop.translate.tdse.TDSEBinding or None, optional
        Must be ``None``: TDSE responses are refused under single precision.

    Raises
    ------
    BackendError
        If the problem is not single precision, or a table cannot be narrowed without
        losing range that its pass does not carry.
    """

    def __init__(
        self,
        problem: KernelProblem,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        tdse_binding: TDSEBinding | None = None,
    ) -> None:
        """Adopt a single-precision problem and narrow its tables."""
        if not problem.precision.single:
            raise BackendError("the single-precision path needs a single problem")
        super().__init__(
            problem,
            grid=grid,
            modes=modes,
            profiles=profiles,
            tdse_binding=tdse_binding,
        )
        self._real = np.dtype(np.float32)
        self._complex = np.dtype(np.complex64)
        self._scale = state_scale(
            launch_peak(problem.initial.field0, problem.grid_kind)
        )
        self._scale_exponent = math.frexp(self._scale)[1] - 1
        # Block exponents of every narrowed table, by name (SEP 33 section 4.3).
        self.block_exponents: dict[str, int] = {}
        tables = self._tables
        self._tables = SourceTables(
            self._upload("synthesis", tables.synthesis),
            self._upload("projection", tables.projection),
            self._upload("t_window", tables.t_window),
            self._upload("t_window_derivative", tables.t_window_derivative),
            self._upload("omega_window", tables.omega_window),
            None
            if tables.temporal_rate is None
            else self._upload("temporal_rate", tables.temporal_rate),
        )
        if self._embedded is not None:
            self._embedded = self._upload("embedded", self._embedded)
        if self._spatial_absorber.size:
            self._spatial_absorber = self._upload(
                "spatial_absorber", self._spatial_absorber
            )
        if self._carrier is not None:
            self._carrier = self._upload("carrier", self._carrier)
        grids = problem.grids
        self._held_factors: tuple[np.ndarray, np.ndarray] | None = None
        if problem.responses.plasma_channels:
            q = 1j * grids.omega[grids.active_mask]
            self._held_factors = (
                self._upload("held_current", 1.0 / q),
                self._upload("held_edge", -1.0 / q**2),
            )
        self._raman_exponent = 0
        controller = problem.stepper.controller
        self._controller = dataclasses.replace(
            controller, atol=controller.atol / self._scale
        )
        # The largest and smallest nonzero magnitude of each pass's intermediates, kept
        # only when the study asks for them (SEP 33 section 7.2).
        self.intermediate_ranges: dict[str, tuple[float, float]] | None = None

    @property
    def scale(self) -> float:
        """Return the state's exact power-of-two scale ``s``.

        Returns
        -------
        float
            The state is held as ``a / s``.
        """
        return self._scale

    def _upload(self, name: str, values: Inexact[np.ndarray, "*shape"]) -> np.ndarray:
        """Narrow one table to binary32, keeping its range, and record its exponent.

        Parameters
        ----------
        name : str
            The table's name, for the record and a refusal.
        values : numpy.ndarray
            The binary64 table.

        Returns
        -------
        numpy.ndarray
            The read-only binary32 table, its block exponent applied exactly.

        Raises
        ------
        BackendError
            If the table's largest magnitude overflows binary32, or it is below one and
            holds an entry below binary32's smallest normal number, which narrowing
            directly would lose and a block exponent would keep.
        """
        source = np.asarray(values)
        exponent = block_exponent(source)
        self.block_exponents[name] = exponent
        magnitude = np.abs(source)
        largest = float(np.max(magnitude, initial=0.0))
        if largest >= _HUGE:
            msg = f"the {name} table reaches {largest:.3e}, beyond binary32's range"
            raise BackendError(msg)
        if exponent < 0 and bool(np.any((magnitude > 0.0) & (magnitude < _TINY))):
            msg = (
                f"the {name} table holds entries below binary32's smallest normal "
                f"number beside a largest of {largest:.3e}: its pass must carry its "
                "block exponent, and none does (SEP 33 section 4.3)"
            )
            raise BackendError(msg)
        dtype = self._complex if np.iscomplexobj(source) else self._real
        narrowed = np.ascontiguousarray(source, dtype=dtype)
        narrowed.setflags(write=False)
        return narrowed

    def _cast(self, values: Inexact[np.ndarray, "*shape"]) -> np.ndarray:
        """Round a host-evaluated binary64 array once, refusing a lost value.

        Parameters
        ----------
        values : numpy.ndarray
            The binary64 array.

        Returns
        -------
        numpy.ndarray
            The binary32 array.

        Raises
        ------
        BackendError
            If an entry overflows binary32 or a nonzero entry falls below its smallest
            normal number.
        """
        source = np.asarray(values)
        dtype = self._complex if np.iscomplexobj(source) else self._real
        narrowed = source.astype(dtype)
        magnitude = np.abs(narrowed)
        if not bool(np.all(np.isfinite(magnitude))) or bool(
            np.any((magnitude < _TINY) & (np.abs(source) > 0.0))
        ):
            raise BackendError(
                "a host-evaluated array leaves binary32's normal range; the pass "
                "applying it must carry a block exponent (SEP 33 section 4.3)"
            )
        return narrowed

    # The state and what crosses to the host.

    def set_initial(
        self,
        field: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        noise: object,
        *,
        z: float = 0.0,
    ) -> None:
        """Load a state, holding it at the launch's scale in binary32.

        Parameters
        ----------
        field : numpy.ndarray
            The state in the canonical layout, physical units.
        noise : object
            Must be ``None``; noise models are refused under single precision.
        z : float, optional
            The position the state belongs to, m.
        """
        del noise  # refused under single precision by validation
        super().set_initial(field, None, z=z)
        state = self._require_state()
        self._state = (state * (1.0 / self._scale)).astype(self._complex)

    def _observed_field(
        self, state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return the physical state, widened to binary64: ``s`` times it, exactly.

        Parameters
        ----------
        state : numpy.ndarray
            The scaled binary32 state.

        Returns
        -------
        numpy.ndarray
            An owned complex128 array.
        """
        return state.astype(np.complex128) * self._scale

    def stats(self, spec: StatsSpec) -> dict[str, np.ndarray]:
        """Collect statistics from the physical state widened to binary64.

        The oracle's statistics are binary64 functions of the exactly widened binary32
        state, so they carry the state's rounding and none of their own.

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            The statistics to collect.

        Returns
        -------
        dict
            As :meth:`NumpyRefBackend.stats`.
        """
        held = self._require_state()
        self._state = self._observed_field(held)
        try:
            return super().stats(spec)
        finally:
            self._state = held

    # The propagator, the norm and the end of a step.

    def _propagator(
        self, exponent: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return ``exp(dPhi)`` evaluated in binary64 and rounded once.

        Parameters
        ----------
        exponent : numpy.ndarray
            The binary64 exponent or its negative.

        Returns
        -------
        numpy.ndarray
            The complex64 propagator.
        """
        with np.errstate(over="ignore"):
            return np.exp(exponent).astype(self._complex)

    def _rounded(self, values: Inexact[np.ndarray, "*shape"]) -> np.ndarray:
        """Round a host-evaluated coefficient array once (SEP 33 section 4.2).

        Parameters
        ----------
        values : numpy.ndarray
            A binary64 array.

        Returns
        -------
        numpy.ndarray
            The binary32 array.
        """
        return self._cast(values)

    def _prefactor(
        self, bundle: StageScalars
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the stage's prefactor, evaluated in binary64 and rounded once.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        numpy.ndarray
            The complex64 prefactor.
        """
        return self._cast(super()._prefactor(bundle))

    def _error_norm(
        self,
        error: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        advanced: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    ) -> float:
        """Return the scaled error norm from scaled sums of squares.

        The state norms and the residual norm are each a scaled sum of squares in
        binary32 (:func:`solprop.translate.precision.scaled_square_sum`), restored on
        the host; the absolute tolerance is the controller's divided by ``s``.

        Parameters
        ----------
        error : numpy.ndarray
            The embedded difference carried to the end of the step.
        state : numpy.ndarray
            The state at the anchor.
        advanced : numpy.ndarray
            The trial state at the end of the step.

        Returns
        -------
        float
            The norm the controller reads, binary64; infinite if any value is not
            finite, which the controller refuses.
        """
        mask = self._problem.grids.norm_mask
        residual = error[..., mask]
        if not bool(np.all(np.isfinite(residual))) or not bool(
            np.all(np.isfinite(advanced))
        ):
            return math.inf
        scale = self._controller.scale(
            math.sqrt(scaled_square_sum(state[..., mask])),
            math.sqrt(scaled_square_sum(advanced[..., mask])),
        )
        if isinstance(scale, np.ndarray):
            return math.sqrt(
                scaled_square_sum(residual, (1.0 / scale[mask]).astype(self._real) ** 2)
            )
        return math.sqrt(scaled_square_sum(residual)) / scale

    def _finish(
        self,
        state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        dz: float,
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Mask the accepted state and apply a split-step collar, in binary32.

        Parameters
        ----------
        state : numpy.ndarray
            The accepted state.
        dz : float
            The step's length, m.

        Returns
        -------
        numpy.ndarray
            The finished complex64 state; a split-step collar's factor is formed in
            binary64 and the product rounded once.
        """
        return np.asarray(super()._finish(state, dz)).astype(self._complex)

    # The source evaluation.

    def _column_sources(
        self,
        field_t: np.ndarray,
        bundle: StageScalars,
        raman: np.ndarray | None,
    ) -> passes.NonlinearSources:
        """Evaluate the responses at a chunk of nodes, in binary32.

        Parameters
        ----------
        field_t : numpy.ndarray
            The scaled field at the chunk's nodes, ``E / s``.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.
        raman : numpy.ndarray or None
            The narrowed Raman kernel spectrum, whose exponent is
            ``self._raman_exponent``.

        Returns
        -------
        solprop.backends.numpy_ref.passes.NonlinearSources
            The windowed polarisation and the ungated current, both divided by ``s``.
        """
        sources = single_sources(
            field_t,
            self._problem.responses,
            self._tables.t_window,
            kind=self._kind,
            scale_exponent=self._scale_exponent,
            dt=self._problem.grids.dt_fine,
            scalars=bundle,
            carrier=self._carrier,
            raman_spectrum=raman,
            raman_exponent=self._raman_exponent,
        )
        if self.intermediate_ranges is not None:
            self._note_range("field", field_t)
            self._note_range("polarisation", sources.polarisation)
            if sources.current is not None:
                self._note_range("current", sources.current)
        return sources

    def _envelope_sources(
        self, field_t: Complex[np.ndarray, "n_nodes n_pol n_to"], bundle: StageScalars
    ) -> tuple[
        Complex[np.ndarray, "n_nodes n_pol n_to"],
        Complex[np.ndarray, "n_nodes n_pol n_to"],
    ]:
        """Return the envelope plasma's sources, evaluated in binary64 and rounded.

        The cycle moments' per-sample decomposition and adaptive quadrature are
        evaluated on the host in binary64 from the physical field ``s E``, widened
        exactly, and the drive and loss divided by ``s`` and rounded once (SEP 33
        section 6.2).

        Parameters
        ----------
        field_t : numpy.ndarray
            The scaled field at the nodes, complex64.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        tuple of numpy.ndarray
            The drive and the loss, complex64, divided by ``s``.
        """
        envelope = self._problem.responses.envelope_plasma
        assert envelope is not None
        grids = self._problem.grids
        drive, loss = envelope_sources(
            field_t.astype(np.complex128) * self._scale,
            envelope,
            bundle,
            grids.dt_fine,
            grids.t_window_fine,
        )
        inverse = 1.0 / self._scale
        return self._cast(drive * inverse), self._cast(loss * inverse)

    def _raman_spectrum(
        self, rates: tuple[float, ...], densities: tuple[float, ...] = ()
    ) -> Complex[np.ndarray, "... n_omega_doubled"] | None:
        """Return the narrowed Raman kernel spectrum, with its exponent kept.

        The spectrum is formed in binary64 as the reference forms it, then narrowed
        with its block exponent, which :func:`single_sources` folds into the Raman
        coefficient: a molecular kernel's spectrum of order ``1e-50`` keeps every bin.

        Parameters
        ----------
        rates : tuple of float
            The decay rate of each term at the stage, s^-1.
        densities : tuple of float, optional
            Component amplitude densities in m^-3.

        Returns
        -------
        numpy.ndarray or None
            The complex64 spectrum scaled by ``2**-e``, or ``None`` without Raman.
        """
        cached = self._raman_kernel
        spectrum = super()._raman_spectrum(rates, densities)
        if spectrum is None:
            return None
        if spectrum is not cached or spectrum.dtype != self._complex:
            exponent = block_exponent(spectrum)
            narrowed = (spectrum * math.ldexp(1.0, -exponent)).astype(self._complex)
            self._raman_exponent = exponent
            self.block_exponents["raman_kernel"] = exponent
            self._raman_kernel = narrowed
            spectrum = narrowed
        return spectrum

    def _held_streams(self, gated: np.ndarray, edge: np.ndarray) -> np.ndarray:
        """Return the held plasma current's spectrum, in binary32.

        Parameters
        ----------
        gated : numpy.ndarray
            The windowed current, divided by ``s``.
        edge : numpy.ndarray
            The current times the window's derivative, divided by ``s``.

        Returns
        -------
        numpy.ndarray
            The complex64 source on the coarse grid.
        """
        assert self._held_factors is not None
        grids = self._problem.grids
        inverse, inverse_squared = self._held_factors
        current = np.fft.rfft(gated, axis=-1)[..., grids.crop_index]
        derivative = np.fft.rfft(edge, axis=-1)[..., grids.crop_index]
        result = np.zeros_like(current)
        active = grids.active_mask
        result[..., active] = (
            current[..., active] * inverse + derivative[..., active] * inverse_squared
        )
        return result

    def _note_range(self, name: str, values: np.ndarray) -> None:
        """Widen the recorded magnitude range of one intermediate.

        Parameters
        ----------
        name : str
            The intermediate.
        values : numpy.ndarray
            Its values at this evaluation.
        """
        assert self.intermediate_ranges is not None
        magnitude = np.abs(values)
        nonzero = magnitude[magnitude > 0]
        if nonzero.size == 0:
            return
        low, high = float(np.min(nonzero)), float(np.max(nonzero))
        if name in self.intermediate_ranges:
            old_low, old_high = self.intermediate_ranges[name]
            low, high = min(low, old_low), max(high, old_high)
        self.intermediate_ranges[name] = (low, high)


def single_sources(  # noqa: PLR0913 - the fused pass takes what the responses need
    field_t: Num[np.ndarray, "n_nodes_chunk n_pol n_to"],
    responses: CompiledResponses,
    t_window: Float[np.ndarray, " n_to"],
    *,
    kind: GridKind,
    scale_exponent: int,
    dt: float,
    scalars: StageScalars,
    carrier: Complex[np.ndarray, " n_to"] | None = None,
    raman_spectrum: Complex[np.ndarray, "... n_omega_doubled"] | None = None,
    raman_exponent: int = 0,
) -> passes.NonlinearSources:
    r"""Evaluate the responses of :func:`.passes.nonlinear_sources` in binary32.

    The field is the scaled one, ``E / s`` with ``s = 2**k``. The result is in the
    scaled state's units: the polarisation and the current divided by ``s``. Each
    response takes its homogeneity factor (SEP 33 section 4.3, rule 4) in one binary64
    coefficient: the cubic terms ``s**2`` with the density, the Raman term also the
    kernel's block exponent; the plasma is evaluated at the physical field ``s E``
    (exact) and its current divided by ``s`` (exact).

    Parameters
    ----------
    field_t : numpy.ndarray
        The scaled field at the chunk's nodes, binary32.
    responses : solprop.translate.responses.CompiledResponses
        The compiled responses.
    t_window : numpy.ndarray
        The narrowed temporal collar profile.
    kind : {'field', 'envelope'}
        The spectral layout.
    scale_exponent : int
        ``k`` with ``s = 2**k``.
    dt : float
        The fine time grid's spacing, s.
    scalars : solprop.translate.stage.StageScalars
        The stage's host scalars: densities and preionisation.
    carrier : numpy.ndarray or None, optional
        The narrowed harmonic carrier of an envelope grid.
    raman_spectrum : numpy.ndarray or None, optional
        The narrowed Raman kernel spectrum.
    raman_exponent : int, optional
        Its block exponent.

    Returns
    -------
    solprop.backends.numpy_ref.passes.NonlinearSources
        The windowed polarisation and the ungated current, in the scaled units.

    Raises
    ------
    BackendError
        For a response the single-precision path does not run.
    """
    total = np.zeros_like(field_t)
    current = None
    multiple = len(responses.plasma_channels) > 1
    k = scale_exponent
    for stage in responses.response_order:
        if stage is ResponseStage.KERR:
            kerr = responses.kerr_coefficients
            assert kerr is not None
            factor = combine(
                kerr.coefficient, scalars.kerr_density, exponent=2 * k
            ).binary32()
            total = total + kerr_single(field_t, kerr, factor, carrier)
        elif stage is ResponseStage.RAMAN_DRIVE:
            assert raman_spectrum is not None
            density = (
                1.0
                if len(responses.raman_terms) > 1
                else scalars.raman_densities[0]
                if scalars.raman_densities
                else scalars.density
            )
            factor = combine(EPS0, density, exponent=2 * k + raman_exponent)
            total = total + raman_single(
                field_t,
                raman_spectrum,
                factor.binary32(),
                kind=kind,
                thg=passes._raman_thg(responses),
            )
        elif stage is ResponseStage.PLASMA_CURRENT:
            physical = np.ldexp(field_t, k) if field_t.dtype.kind == "f" else None
            if physical is None:
                raise BackendError("the field-grid plasma needs a real field")
            channels = responses.plasma_channels
            if multiple:
                densities = scalars.plasma_densities
                initials = scalars.plasma_preionisation
            else:
                plasma_density = (
                    scalars.plasma_densities[0]
                    if scalars.plasma_densities
                    else scalars.density
                )
                densities = (plasma_density,)
                initials = (scalars.preionisation,)
            current = np.ldexp(
                plasma_current_single(
                    physical,
                    channels,
                    densities,
                    initials,
                    dt,
                    responses.plasma_integral,
                ),
                -k,
            )
        elif stage in _ELSEWHERE:
            continue
        else:
            msg = f"the single-precision path does not run the {stage.name} stage"
            raise BackendError(msg)
    windowed = (total * t_window).astype(field_t.dtype, copy=False)
    return passes.NonlinearSources(windowed, current)


def kerr_single(
    field_t: Num[np.ndarray, "n_nodes_chunk n_pol n_to"],
    kerr: KerrCoefficients,
    factor: float,
    carrier: Complex[np.ndarray, " n_to"] | None,
) -> Num[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    """Evaluate the Kerr polarisation of :func:`.passes.kerr_polarisation` in binary32.

    Parameters
    ----------
    field_t : numpy.ndarray
        The scaled field, binary32.
    kerr : solprop.translate.responses.KerrCoefficients
        The compiled coefficients.
    factor : float
        ``coefficient * density * s**2``, combined on the host and rounded once.
    carrier : numpy.ndarray or None
        The narrowed harmonic carrier, when the envelope grid radiates the harmonic.

    Returns
    -------
    numpy.ndarray
        The polarisation divided by ``s``, binary32.
    """
    values = analytic_single(field_t) if kerr.analytic_signal else field_t
    if kerr.cubic_form == "real":
        polarisation = factor * np.sum(values**2, axis=-2, keepdims=True) * values
    else:
        squares = np.sum(values**2, axis=-2, keepdims=True)
        moduli = np.sum(
            np.real(values) ** 2 + np.imag(values) ** 2, axis=-2, keepdims=True
        )
        polarisation = factor * (
            kerr.self_factor * moduli * values
            + kerr.conjugate_factor * squares * np.conj(values)
        )
        if carrier is not None:
            polarisation = polarisation + (
                (factor * kerr.harmonic_factor) * carrier * squares * values
            )
    return polarisation.real if kerr.take_real else polarisation


def analytic_single(
    values: Num[np.ndarray, "*batch n_t"],
) -> Complex[np.ndarray, "*batch n_t"]:
    """Return the analytic signal of a real binary32 array, in binary32.

    The same construction as :func:`solprop.fourier.analytic_signal` (doubling the
    positive frequencies, keeping zero and Nyquist), with the multiplier held in the
    values' precision.

    Parameters
    ----------
    values : numpy.ndarray
        Real values along the last axis.

    Returns
    -------
    numpy.ndarray
        The complex64 analytic signal.
    """
    real = np.real(values).astype(np.float32, copy=False)
    n = real.shape[-1]
    spectrum = np.fft.fft(real, axis=-1)
    weights = np.zeros(n, dtype=np.float32)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[n // 2] = 1.0
        weights[1 : n // 2] = 2.0
    else:
        weights[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * weights, axis=-1)


def _drive_single(
    row: Num[np.ndarray, "n_pol n_to"], *, kind: GridKind, thg: bool
) -> Float[np.ndarray, " n_to"]:
    """Return the Raman drive of one node in binary32 (:func:`raman_drive`).

    Parameters
    ----------
    row : numpy.ndarray
        One node's scaled field.
    kind : {'field', 'envelope'}
        The spectral layout.
    thg : bool
        Whether the drive keeps the harmonic.

    Returns
    -------
    numpy.ndarray
        The float32 drive.
    """
    column = row[0]
    if kind == "envelope":
        return 0.5 * (np.real(column) ** 2 + np.imag(column) ** 2)
    if thg:
        return np.real(column) ** 2
    signal = analytic_single(np.real(column))
    return 0.5 * (np.real(signal) ** 2 + np.imag(signal) ** 2)


def raman_single(
    field_t: Num[np.ndarray, "n_nodes_chunk n_pol n_to"],
    spectrum: Complex[np.ndarray, "... n_omega_doubled"],
    factor: float,
    *,
    kind: GridKind,
    thg: bool,
) -> Num[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    """Evaluate the delayed Raman polarisation in binary32.

    The convolution of :func:`.passes.raman_polarisation` with the narrowed kernel;
    the coefficient carries ``eps0``, the density, ``s**2`` and the kernel's exponent,
    and is applied once to the product with the field.

    Parameters
    ----------
    field_t : numpy.ndarray
        The scaled field, binary32.
    spectrum : numpy.ndarray
        The narrowed kernel spectrum, ``(n_to + 1,)`` scalar or ``(1 | 2, n_to + 1)``
        tensor.
    factor : float
        The combined coefficient, rounded once.
    kind : {'field', 'envelope'}
        The spectral layout.
    thg : bool
        Whether the drive keeps the harmonic.

    Returns
    -------
    numpy.ndarray
        The polarisation divided by ``s``.

    Raises
    ------
    BackendError
        If the field has neither one nor two components.
    """
    n_nodes, n_pol, n_to = field_t.shape
    if n_pol == 2:
        return _vector_raman_single(field_t, spectrum, factor, kind=kind, thg=thg)
    if n_pol != 1:
        raise BackendError("Raman needs one or two physical components")
    drive = np.zeros((n_nodes, 2 * n_to), dtype=np.float32)
    for node in range(n_nodes):
        drive[node, :n_to] = _drive_single(field_t[node], kind=kind, thg=thg)
    convolved = np.fft.irfft(
        spectrum * np.fft.rfft(drive, axis=-1), n=2 * n_to, axis=-1
    )[..., :n_to]
    return factor * field_t * convolved[:, np.newaxis, :]


def _vector_raman_single(
    field_t: Num[np.ndarray, "n_nodes_chunk 2 n_to"],
    spectrum: Complex[np.ndarray, "n_kernels n_omega_doubled"],
    factor: float,
    *,
    kind: GridKind,
    thg: bool,
) -> Num[np.ndarray, "n_nodes_chunk 2 n_to"]:
    """Evaluate the two-component Raman polarisation in binary32.

    Parameters
    ----------
    field_t : numpy.ndarray
        The scaled two-component field.
    spectrum : numpy.ndarray
        One isotropic or two (isotropic, anisotropic) narrowed kernel spectra.
    factor : float
        The combined coefficient.
    kind : {'field', 'envelope'}
        The spectral layout.
    thg : bool
        Whether the drive keeps the harmonic.

    Returns
    -------
    numpy.ndarray
        The polarisation divided by ``s``.

    Raises
    ------
    BackendError
        If the kernel spectra have the wrong shape.
    """
    n_to = field_t.shape[-1]
    if spectrum.ndim != 2 or spectrum.shape not in ((1, n_to + 1), (2, n_to + 1)):
        raise BackendError("vector Raman needs one or two tensor kernel spectra")
    values = field_t
    if kind == "field" and not thg:
        values = analytic_single(np.real(field_t))
    x, y = values[:, 0], values[:, 1]
    half = 0.5 if kind == "envelope" or not thg else 1.0
    xx = half * (np.real(x) ** 2 + np.imag(x) ** 2)
    yy = half * (np.real(y) ** 2 + np.imag(y) ** 2)
    n_drives = 1 if spectrum.shape[0] == 1 else 3
    drive = np.zeros((field_t.shape[0], n_drives, 2 * n_to), dtype=np.float32)
    drive[:, 0, :n_to] = xx + yy
    if n_drives == 3:
        drive[:, 1, :n_to] = xx - yy
        drive[:, 2, :n_to] = (
            2 * half * (np.real(x) * np.real(y) + np.imag(x) * np.imag(y))
        )
    kernels = spectrum if n_drives == 1 else spectrum[[0, 1, 1]]
    convolved = np.fft.irfft(
        kernels * np.fft.rfft(drive, axis=-1), n=2 * n_to, axis=-1
    )[..., :n_to]
    if n_drives == 1:
        return factor * field_t * convolved
    u, v, w = convolved[:, 0], convolved[:, 1], convolved[:, 2]
    result = np.empty_like(field_t)
    result[:, 0] = (u + v) * field_t[:, 0] + w * field_t[:, 1]
    result[:, 1] = w * field_t[:, 0] + (u - v) * field_t[:, 1]
    return factor * result


def running_integral(
    values: Float[np.ndarray, "... n_to"], dt: float, rule: RunningIntegral
) -> Float[np.ndarray, "... n_to"]:
    """Return a running integral in binary32, in the oracle's blocked order.

    The rule of :meth:`solprop.nonlinear.ionisation.RunningIntegral.integrate`: the
    band rule's filter (its taps rounded once, the pairs added in the same order), then
    the cumulative trapezoid, whose increments are summed by
    :func:`solprop.translate.precision.prefix_sum`. Each prefix is a sum of depth at
    most ``ceil(log2 n)``, where a sequential binary32 cumulative sum of ``n`` terms
    would carry an error growing with ``n``.

    Parameters
    ----------
    values : numpy.ndarray
        The integrand along the last axis, binary32.
    dt : float
        The sample spacing, s.
    rule : solprop.nonlinear.ionisation.RunningIntegral
        The problem's rule.

    Returns
    -------
    numpy.ndarray
        The running integral, zero at the first sample, binary32.
    """
    rule.check(dt)
    filtered = values if rule.half_width == 0 else _band_filter(values, rule)
    increments = (filtered[..., :-1] + filtered[..., 1:]) * np.float32(0.5 * dt)
    result = np.zeros_like(filtered)
    result[..., 1:] = prefix_sum(increments, axis=-1)
    return result


def _band_filter(
    values: Float[np.ndarray, "... n_to"], rule: RunningIntegral
) -> Float[np.ndarray, "... n_to"]:
    """Apply a band rule's filter in binary32, as :meth:`RunningIntegral.filter` does.

    Parameters
    ----------
    values : numpy.ndarray
        The integrand, binary32.
    rule : solprop.nonlinear.ionisation.RunningIntegral
        A band rule.

    Returns
    -------
    numpy.ndarray
        The filtered integrand, binary32.
    """
    half = rule.half_width
    n_to = values.shape[-1]
    reach = np.arange(half, 0, -1)
    head = 2 * values[..., :1] - values[..., reach]
    tail = 2 * values[..., -1:] - values[..., n_to - 1 - reach[::-1]]
    extended = np.concatenate([head, values, tail], axis=-1)
    taps = rule.taps.astype(np.float32)
    filtered = taps[0] * values
    for k in range(1, half + 1):
        filtered = filtered + taps[k] * (
            extended[..., half - k : half - k + n_to]
            + extended[..., half + k : half + k + n_to]
        )
    return filtered


def stable_population(
    exposure: Float[np.ndarray, "*shape"], preionisation: float
) -> tuple[Float[np.ndarray, "*shape"], Float[np.ndarray, "*shape"]]:
    """Return the ionised fraction and the neutral survival in their stable forms.

    SEP 33 section 4.4: ``f = p + (1 - p) (-expm1(-X))`` keeps relative accuracy for a
    small exposure ``X``, and ``sigma = (1 - p) exp(-X)`` for a large one, where ``f``
    rounds to one. The loss current and the ionisation work are formed from ``sigma``,
    never from ``1 - f``.

    Parameters
    ----------
    exposure : numpy.ndarray
        ``X``, the running integral of the rate, non-negative, binary32.
    preionisation : float
        ``p`` in ``[0, 1]``.

    Returns
    -------
    fraction : numpy.ndarray
        ``f``, binary32.
    survival : numpy.ndarray
        ``sigma``, binary32.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.single import stable_population
    >>> f, sigma = stable_population(np.array([1e-8, 20.0], np.float32), 0.0)
    >>> bool(f[0] > 0), bool(sigma[1] > 0)
    (True, True)
    """
    keep = np.float32(1.0 - preionisation)
    survival = keep * np.exp(-exposure)
    fraction = np.float32(preionisation) + keep * -np.expm1(-exposure)
    return fraction, survival


def plasma_current_single(
    field: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    channels: tuple[PlasmaChannel, ...],
    densities: Sequence[float],
    preionisation: Sequence[float],
    dt: float,
    rule: RunningIntegral,
) -> Float[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Return the field-grid plasma's ungated current density in binary32.

    The current of :func:`.passes.nonlinear_sources` (one channel) and of its channel
    sum (several), with the stable populations of :func:`stable_population`: the free
    current integrates ``sum_c rho_c f_c (e^2/m) n_c E`` and the loss current is
    ``sum_c rho_c I_c W_c sigma_c E / |E|^2``, its division masked where the rate or
    the field is zero. The rate's table lookup is evaluated in binary64 from the
    binary32 magnitude and rounded once.

    Parameters
    ----------
    field : numpy.ndarray
        The physical field ``s E``, V/m, binary32.
    channels : tuple of solprop.translate.responses.PlasmaChannel
        The compiled channels.
    densities : sequence of float
        Each channel's density at the stage, m^-3.
    preionisation : sequence of float
        Each channel's pre-ionised fraction at the stage.
    dt : float
        The sample spacing, s.
    rule : solprop.nonlinear.ionisation.RunningIntegral
        The running-integral rule.

    Returns
    -------
    numpy.ndarray
        The current density, A/m^2, binary32.
    """
    magnitude = (
        np.abs(field[..., 0, :])
        if field.shape[-2] == 1
        else np.hypot(field[..., 0, :], field[..., 1, :])
    )
    weight = np.zeros_like(magnitude)
    work = np.zeros_like(magnitude)
    for channel, density, initial in zip(
        channels, densities, preionisation, strict=True
    ):
        rate = channel.rate(magnitude.astype(np.float64)).astype(np.float32)
        exposure = running_integral(rate, dt, rule)
        if rule.policy != "trapezoid":
            exposure = np.maximum(exposure, np.float32(0.0))
        fraction, survival = stable_population(exposure, initial)
        weight += (
            np.float32(
                density * channel.charge_mass_ratio * channel.electrons_per_event
            )
            * fraction
        )
        work += np.float32(density * channel.ionisation_potential) * (rate * survival)
    free = running_integral(weight[..., np.newaxis, :] * field, dt, rule)
    squared = magnitude * magnitude
    guard = np.zeros_like(work)
    np.divide(work, squared, out=guard, where=(work > 0.0) & (squared > 0.0))
    return free + guard[..., np.newaxis, :] * field
