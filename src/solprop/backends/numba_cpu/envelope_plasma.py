# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Own compiled envelope-current workspaces and their forward transforms."""

import time
from functools import cache
from typing import Any, no_type_check

import numpy as np
from numba import prange
from numba.core import types

from solprop._numba_cache import njit
from solprop.backends.bodies import RateArrays
from solprop.backends.numba_cpu.fft import PLAN_FLAGS, _heap_bytes
from solprop.backends.numba_cpu.kernels import HOST
from solprop.backends.numba_cpu.openmp import pyfftw
from solprop.exceptions import ConvergenceError
from solprop.translate.envelope_plasma import EnvelopePlasmaAssembly
from solprop.translate.stage import StageScalars

__all__ = ["EnvelopeWorkspace", "make_source_column"]


@cache
def make_source_column(n_pol: int, *, parallel: bool, statistics: bool = False) -> Any:
    """Compile a deterministic source pass over independent physical nodes.

    Parameters
    ----------
    n_pol : int
        Cartesian components per node, one or two.
    parallel : bool
        Parallelise independent nodes, preserving each temporal scan's order.
    statistics : bool, optional
        Only fill population histories; omit current formation.

    Returns
    -------
    object
        Explicitly typed Numba source kernel; all work buffers are supplied.
    """
    population_at = HOST.envelope_population
    axes_at, moments_at, loss_at = (
        HOST.envelope_axes,
        HOST.envelope_moments,
        HOST.envelope_loss,
    )

    @no_type_check
    def column(
        field: Any,
        workspaces: Any,
        table: RateArrays,
        rules: Any,
        scalars: Any,
        window: Any,
    ) -> None:
        """Accumulate one channel's two sources after a fresh population scan.

        Parameters
        ----------
        field : numpy.ndarray
            Complex physical fields, rows node*components, columns fine time.
        workspaces : tuple
            Aggregate sources (2, n_rows, n_to), five float histories per node
            (Mc, Ms, rate, fraction, survival) and per-node failure status.
        table : RateArrays
            Instantaneous magnitude table.
        rules : tuple
            Packed rule nodes and weights, each shape (4, 512).
        scalars : tuple
            Density, initial fraction, e²/m_e, potential and time spacing, SI.
        window : numpy.ndarray
            Fine material-source collar.
        """
        output, scratch, status = workspaces
        density, initial, charge_mass, potential, dt = scalars
        nodes, weights = rules
        for node in prange(field.shape[0] // n_pol):
            base = node * n_pol
            status[node] = 0
            for sample in range(field.shape[1]):
                first = field[base, sample]
                second = field[base + 1, sample] if n_pol == 2 else 0j
                axes = axes_at(first, second)
                mc, ms, failed = moments_at(axes[0], axes[1], table, nodes, weights)
                if failed:
                    status[node] = sample + 1
                    break
                scratch[0, node, sample] = mc
                scratch[1, node, sample] = ms
                scratch[2, node, sample] = mc + axes[1] * axes[1] * ms
            if status[node] != 0:
                continue
            exposure = 0.0
            for sample in range(field.shape[1]):
                if sample > 0:
                    exposure += dt * (
                        0.5 * scratch[2, node, sample - 1]
                        + 0.5 * scratch[2, node, sample]
                    )
                fraction, survival = population_at(exposure, initial)
                scratch[3, node, sample] = fraction
                scratch[4, node, sample] = survival
                if statistics:
                    continue
                first = field[base, sample]
                second = field[base + 1, sample] if n_pol == 2 else 0j
                axes = axes_at(first, second)
                loss_x, loss_y = loss_at(
                    axes,
                    scratch[0, node, sample],
                    scratch[1, node, sample],
                    potential * density * survival,
                )
                drive = charge_mass * density * fraction
                output[0, base, sample] += window[sample] * (drive * first)
                output[1, base, sample] += window[sample] * loss_x
                if n_pol == 2:
                    output[0, base + 1, sample] += window[sample] * (drive * second)
                    output[1, base + 1, sample] += window[sample] * loss_y

    real_readonly = types.Array(types.float64, 2, "C", readonly=True)
    rate_type = types.NamedTuple((real_readonly, *(types.float64,) * 5), RateArrays)
    signature = types.void(
        types.complex128[:, ::1],
        types.Tuple(
            (types.complex128[:, :, ::1], types.float64[:, :, ::1], types.int64[::1])
        ),
        rate_type,
        types.UniTuple(real_readonly, 2),
        types.UniTuple(types.float64, 5),
        types.Array(types.float64, 1, "C", readonly=True),
    )
    # Never disk-cached: the closure captures dispatchers, so its cache
    # key differs in every process, and a parallel closure may not be cached.
    return njit(signature, cache=False, nogil=True, parallel=parallel)(column)


@no_type_check
def _assemble(destination: Any, spectra: Any, crop: Any, drude: Any, loss: Any) -> None:
    """Assemble current spectra on the active coarse bins of a fine FFT buffer.

    Parameters
    ----------
    destination : numpy.ndarray
        Ordinary polarisation spectrum, updated in place.
    spectra : numpy.ndarray
        Drude-drive and loss-current spectra, with a leading source axis.
    crop : numpy.ndarray
        Fine-bin indices in coarse-axis order.
    drude, loss : numpy.ndarray
        Compiled masked internal-FFT multipliers.
    """
    for row in range(destination.shape[0]):
        for index in range(crop.size):
            fine = crop[index]
            destination[row, fine] += (
                drude[index] * spectra[0, row, fine]
                + loss[index] * spectra[1, row, fine]
            )


@cache
def _assembly_kernel() -> Any:
    """Compile spectral assembly eagerly when the first envelope workspace is built.

    Returns
    -------
    object
        Explicitly typed cached CPU kernel, or its interpreted coverage body.
    """
    signature = types.void(
        types.complex128[:, ::1],
        types.complex128[:, :, ::1],
        types.Array(types.intp, 1, "C", readonly=True),
        types.Array(types.float64, 1, "C", readonly=True),
        types.Array(types.complex128, 1, "C", readonly=True),
    )
    return njit(signature, cache=True, nogil=True)(_assemble)


class EnvelopeWorkspace:
    """Preallocate channel histories, current projection buffers and FFT plans.

    Parameters
    ----------
    assembly : EnvelopePlasmaAssembly
        Validated channels and spectral multipliers.
    shape : tuple of int
        Chunk nodes, node components, temporal columns and fine samples.
    embedded : bool
        Allocate the additional embedded-rule source projections.
    parallel : bool
        Parallelise source evaluation across physical nodes.
    flag : str
        Existing FFTW planning policy.
    threads : int
        Existing modal FFT thread count.
    """

    def __init__(
        self,
        assembly: EnvelopePlasmaAssembly,
        shape: tuple[int, int, int, int],
        *,
        embedded: bool,
        parallel: bool,
        flag: str,
        threads: int,
    ) -> None:
        self.assembly = assembly
        self.n_pol = shape[1]
        chunk, pol, columns, samples = shape
        self.sources = np.zeros((2, chunk * pol, samples), dtype=np.complex128)
        self.histories = np.empty((5, chunk, samples), dtype=np.float64)
        self.status = np.zeros(chunk, dtype=np.int64)
        self.projected = pyfftw.empty_aligned((2, columns, samples), dtype="complex128")
        self.spectrum = pyfftw.empty_aligned(self.projected.shape, dtype="complex128")
        self.projection_scratch = np.empty((columns, samples), dtype=np.complex128)
        self.coarse_projected = (
            pyfftw.empty_aligned(self.projected.shape, dtype="complex128")
            if embedded
            else None
        )
        self.coarse_spectrum = (
            pyfftw.empty_aligned(self.projected.shape, dtype="complex128")
            if embedded
            else None
        )
        heap_before = _heap_bytes()
        started = time.perf_counter()
        self.forward = pyfftw.FFTW(
            self.projected,
            self.spectrum,
            axes=(-1,),
            direction="FFTW_FORWARD",
            flags=(PLAN_FLAGS[flag],),
            threads=threads,
        )
        self.coarse_forward = (
            None
            if not embedded
            else pyfftw.FFTW(
                self.coarse_projected,
                self.coarse_spectrum,
                axes=(-1,),
                direction="FFTW_FORWARD",
                flags=(PLAN_FLAGS[flag],),
                threads=threads,
            )
        )
        self.planning_seconds = time.perf_counter() - started
        self.plan_heap_growth = max(0, _heap_bytes() - heap_before)
        self.rules = []
        for channel in assembly.channels:
            nodes, weights = np.zeros((4, 512)), np.zeros((4, 512))
            for index, rule in enumerate(channel.cycle.rules):
                nodes[index, : len(rule.nodes)] = rule.nodes
                weights[index, : len(rule.weights)] = rule.weights
            nodes.setflags(write=False)
            weights.setflags(write=False)
            self.rules.append((nodes, weights))
        self.rates = tuple(RateArrays.of(channel.rate) for channel in assembly.channels)
        self.column = make_source_column(pol, parallel=parallel)
        self.statistics_column = make_source_column(
            pol, parallel=parallel, statistics=True
        )
        self.assemble_kernel = _assembly_kernel()

    def respond(
        self, field: np.ndarray, scalars: StageScalars, dt: float, window: np.ndarray
    ) -> None:
        """Recompute independent channel histories for one current RHS chunk.

        Parameters
        ----------
        field : numpy.ndarray
            Physical V/m fields, rows node*component, columns fine time.
        scalars : StageScalars
            Current position's channel densities and initial populations.
        dt : float
            Fine time spacing in s.
        window : numpy.ndarray
            Read-only material-source collar.

        Raises
        ------
        ConvergenceError
            If any node fails angular convergence; the driver sees this before
            a source enters a forward transform.
        """
        self.sources[:] = 0
        for index, channel in enumerate(self.assembly.channels):
            values = (
                scalars.plasma_densities[index],
                scalars.plasma_preionisation[index],
                channel.charge_mass_ratio,
                channel.ionisation_potential,
                dt,
            )
            self.column(
                field,
                (self.sources, self.histories, self.status),
                self.rates[index],
                self.rules[index],
                values,
                window,
            )
            if np.any(self.status[: field.shape[0] // self.n_pol]):
                raise ConvergenceError(
                    f"envelope plasma channel {channel.label!r} exceeded "
                    "the angular refinement budget"
                )

    def populations(
        self,
        field: np.ndarray,
        initials: tuple[float, ...],
        dt: float,
        window: np.ndarray,
        output: np.ndarray,
    ) -> None:
        """Scan one statistics chunk into preallocated channel/node partials.

        Parameters
        ----------
        field : numpy.ndarray
            Physical V/m envelopes, shape (n_nodes*n_pol, n_to).
        initials : tuple of float
            Initial fractions in channel order at the requested plane.
        dt : float
            Fine time spacing in s.
        window : numpy.ndarray
            Read-only source collar; unused by this statistics specialization.
        output : numpy.ndarray
            Output rows alternate final fraction and peak cycle-averaged rate
            in s^-1 for each channel, columns are nodes in this chunk.

        Raises
        ------
        ConvergenceError
            If angular refinement fails at any node.
        """
        count = field.shape[0] // self.n_pol
        for index, (channel, initial) in enumerate(
            zip(self.assembly.channels, initials, strict=True)
        ):
            self.statistics_column(
                field,
                (self.sources, self.histories, self.status),
                self.rates[index],
                self.rules[index],
                (0.0, initial, 0.0, 0.0, dt),
                window,
            )
            if np.any(self.status[:count]):
                raise ConvergenceError(
                    f"envelope plasma channel {channel.label!r} exceeded "
                    "the angular refinement budget"
                )
            output[2 * index] = self.histories[3, :count, -1]
            np.max(self.histories[2, :count], axis=-1, out=output[2 * index + 1])

    def project(
        self,
        projection: np.ndarray,
        embedded: np.ndarray | None,
        n_modes: int,
        *,
        first: bool,
        count: int,
    ) -> None:
        """Accumulate this chunk's two currents before temporal transformation.

        Parameters
        ----------
        projection : numpy.ndarray
            Already scaled projection columns for the active nodes.
        embedded : numpy.ndarray or None
            Corresponding embedded-rule projection.
        n_modes : int
            Output mode count.
        first : bool
            Overwrite on the first chunk and accumulate later chunks in order.
        count : int
            Active node-component rows, including a final partial chunk.
        """
        for stream in range(2):
            source = self.sources[stream, :count].reshape(projection.shape[1], -1)
            scratch = self.projection_scratch.reshape(n_modes, -1)
            np.dot(projection, source, out=scratch)
            if first:
                self.projected[stream] = self.projection_scratch
            else:
                np.add(
                    self.projected[stream],
                    self.projection_scratch,
                    out=self.projected[stream],
                )
            if embedded is not None and self.coarse_projected is not None:
                np.dot(embedded, source, out=scratch)
                if first:
                    self.coarse_projected[stream] = self.projection_scratch
                else:
                    np.add(
                        self.coarse_projected[stream],
                        self.projection_scratch,
                        out=self.coarse_projected[stream],
                    )

    def assemble(
        self, destination: np.ndarray, crop: np.ndarray, *, coarse: bool = False
    ) -> None:
        """Transform both current streams and add their masked spectral sources.

        Parameters
        ----------
        destination : numpy.ndarray
            Ordinary polarisation fine spectrum, updated in place.
        crop : numpy.ndarray
            Coarse-to-fine bin mapping.
        coarse : bool
            Transform embedded spatial projections for the quadrature alarm.
        """
        spectrum = self.coarse_spectrum if coarse else self.spectrum
        plan = self.coarse_forward if coarse else self.forward
        assert plan is not None and spectrum is not None
        plan.execute()
        self.assemble_kernel(
            destination,
            spectrum,
            crop,
            self.assembly.factors.drude,
            self.assembly.factors.loss,
        )

    def memory_ledger(self) -> dict[str, int]:
        """List each owned array once, including full-width partial-chunk padding.

        Returns
        -------
        dict
            Named persistent buffer sizes in bytes; no arrays are allocated by
            a final partial chunk or an RHS call.
        """
        result = {
            f"envelope_{name}": getattr(self, name).nbytes
            for name in (
                "sources",
                "histories",
                "status",
                "projected",
                "spectrum",
                "projection_scratch",
            )
        }
        result["envelope_coarse_projected"] = (
            0 if self.coarse_projected is None else self.coarse_projected.nbytes
        )
        result["envelope_coarse_spectrum"] = (
            0 if self.coarse_spectrum is None else self.coarse_spectrum.nbytes
        )
        result["envelope_rules"] = sum(
            nodes.nbytes + weights.nbytes for nodes, weights in self.rules
        )
        result["envelope_fft_plan_heap_growth"] = self.plan_heap_growth
        return result
