# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Own the held field-current streams, transforms and frequency assembly."""

import time
from functools import cache
from typing import Any, no_type_check

import numpy as np
from numba.core import types

from solprop._numba_cache import njit
from solprop.backends.numba_cpu.fft import PLAN_FLAGS, _heap_bytes
from solprop.backends.numba_cpu.kernels import HOST
from solprop.backends.numba_cpu.openmp import pyfftw
from solprop.translate.problem import KernelProblem

__all__ = ["HeldWorkspace"]


@cache
def _assembly_kernel(radial: bool) -> Any:
    """Compile masked assembly, either adding to P or writing cropped nodal P+A_J.

    Parameters
    ----------
    radial : bool
        Write the combined cropped nodal spectrum instead of adding modal current.

    Returns
    -------
    callable
        Compiled kernel, cached only in this process.
    """
    source = HOST.held_plasma_source

    @no_type_check
    def assemble(
        destination: Any, spectra: Any, crop: Any, inverse: Any, active: Any
    ) -> None:
        """Combine the transformed streams at active frequencies.

        Parameters
        ----------
        destination : numpy.ndarray
            Complex output rows, updated in place.
        spectra : numpy.ndarray
            Fine spectra of P, gJ and g'J for radial assembly, or gJ and g'J.
        crop : numpy.ndarray
            Fine-grid indices of the retained frequencies.
        inverse : numpy.ndarray
            Reciprocal angular frequencies in s; zero outside the active band.
        active : numpy.ndarray
            Boolean hard mask over the retained frequencies.
        """
        for row in range(destination.shape[0]):
            for index in range(crop.size):
                fine = crop[index]
                if radial:
                    destination[row, index] = (
                        spectra[0, row, fine]
                        + source(
                            spectra[1, row, fine], spectra[2, row, fine], inverse[index]
                        )
                        if active[index]
                        else 0j
                    )
                elif active[index]:
                    destination[row, fine] += source(
                        spectra[0, row, fine], spectra[1, row, fine], inverse[index]
                    )

    signature = types.void(
        types.complex128[:, ::1],
        types.complex128[:, :, ::1],
        types.Array(types.intp, 1, "C", readonly=True),
        types.Array(types.float64, 1, "C", readonly=True),
        types.Array(types.boolean, 1, "C", readonly=True),
    )
    # Captures a dispatcher; memoise in process, never append disk-cache entries.
    return njit(signature, cache=False, nogil=True)(assemble)


@cache
def _scatter_kernel() -> Any:
    """Compile radial projection accumulation into the existing fine-spectrum rows.

    Returns
    -------
    callable
        Compiled accumulation kernel, cached only in this process.
    """

    @no_type_check
    def scatter(destination: Any, projected: Any, crop: Any) -> None:
        """Accumulate one radial chunk at its retained fine-spectrum indices.

        Parameters
        ----------
        destination : numpy.ndarray
            Complex fine-spectrum rows, updated in place.
        projected : numpy.ndarray
            Combined cropped source after radial projection.
        crop : numpy.ndarray
            Fine-grid indices of the retained frequencies.
        """
        for row in range(projected.shape[0]):
            for index in range(crop.size):
                destination[row, crop[index]] += projected[row, index]

    return njit(
        types.void(
            types.complex128[:, ::1],
            types.complex128[:, ::1],
            types.Array(types.intp, 1, "C", readonly=True),
        ),
        cache=False,
        nogil=True,
    )(scatter)


class HeldWorkspace:
    """Allocate the field-current streams and plan their transforms once.

    Modal and planar routes project both products before their temporal FFTs.
    Radial routes transform three nodal streams, combine and crop them, then
    make one real projection on interleaved spectral components. Cartesian and
    line routes combine on physical nodes before the transverse transform.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        Validated field-grid problem with plasma channels.
    shape : tuple of int
        Chunk node-component rows and temporal transform rows.
    embedded : bool
        Whether the modal quadrature has an embedded projection for its alarm.
    flag : str
        Existing FFTW planning policy.
    threads : int
        Existing temporal FFT thread count.
    """

    def __init__(
        self,
        problem: KernelProblem,
        shape: tuple[int, int],
        *,
        embedded: bool,
        flag: str,
        threads: int,
    ) -> None:
        grids, transform = problem.grids, problem.transform
        geometry = transform.geometry
        self.radial = geometry is not None and geometry.kind == "radial"
        self.crop, self.active = grids.crop_index, grids.active_mask
        self.inverse = np.zeros(grids.n_omega)
        np.divide(1.0, grids.omega, out=self.inverse, where=self.active)
        self.inverse.setflags(write=False)
        chunk, columns = shape
        samples = grids.n_to
        streams = 3 if self.radial else 2
        columns = chunk if self.radial else columns
        self.current = np.empty((chunk, samples))
        self.sources = np.empty((2, chunk, samples))
        self.projected = pyfftw.empty_aligned(
            (streams, columns, samples), dtype="float64"
        )
        self.spectrum = pyfftw.empty_aligned(
            (streams, columns, samples // 2 + 1), dtype="complex128"
        )
        self.projection_scratch = np.empty(
            (0, 0) if self.radial else (columns, samples)
        )
        self.coarse_projected = (
            pyfftw.empty_aligned(self.projected.shape, dtype="float64")
            if embedded
            else None
        )
        self.coarse_spectrum = (
            pyfftw.empty_aligned(self.spectrum.shape, dtype="complex128")
            if embedded
            else None
        )
        self.nodal_spectrum = np.empty(
            (chunk, grids.n_omega) if self.radial else (0, 0), dtype=np.complex128
        )
        self.radial_projection = np.empty(
            (problem.shape[1] * problem.shape[2], grids.n_omega)
            if self.radial
            else (0, 0),
            dtype=np.complex128,
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
        self.assemble_kernel = _assembly_kernel(self.radial)
        self.scatter = _scatter_kernel() if self.radial else None

    def products(self, gate: np.ndarray, slope: np.ndarray, count: int) -> None:
        """Write both complete current products and clear unused chunk rows.

        Parameters
        ----------
        gate, slope : numpy.ndarray
            Fine-grid temporal collar and its analytic derivative, in 1 and s⁻¹.
        count : int
            Populated node-component rows; rows beyond this are padding.
        """
        np.multiply(self.current[:count], gate, out=self.sources[0, :count])
        np.multiply(self.current[:count], slope, out=self.sources[1, :count])
        self.current[count:] = 0
        self.sources[:, count:] = 0

    def project(
        self,
        projection: np.ndarray,
        embedded: np.ndarray | None,
        n_modes: int,
        *,
        first: bool,
        count: int,
    ) -> None:
        """Project each current product onto modal rows before transformation.

        Parameters
        ----------
        projection : numpy.ndarray
            Scaled projection for this chunk.
        embedded : numpy.ndarray or None
            Scaled embedded-rule projection of the same nodes.
        n_modes : int
            Propagated modes, before the separate state-component axis.
        first : bool
            Overwrite the output on the first chunk; accumulate later chunks.
        count : int
            Active node-component rows.
        """
        scratch = self.projection_scratch.reshape(n_modes, -1)
        for stream in range(2):
            values = self.sources[stream, :count].reshape(projection.shape[1], -1)
            np.dot(projection, values, out=scratch)
            if first:
                self.projected[stream] = self.projection_scratch
            else:
                np.add(
                    self.projected[stream],
                    self.projection_scratch,
                    out=self.projected[stream],
                )
            if embedded is not None and self.coarse_projected is not None:
                np.dot(embedded, values, out=scratch)
                if first:
                    self.coarse_projected[stream] = self.projection_scratch
                else:
                    np.add(
                        self.coarse_projected[stream],
                        self.projection_scratch,
                        out=self.coarse_projected[stream],
                    )

    def assemble(self, destination: np.ndarray, *, coarse: bool = False) -> None:
        """Transform projected currents and add their active-bin polarisation.

        Parameters
        ----------
        destination : numpy.ndarray
            Fine non-plasma polarisation spectrum, updated in place.
        coarse : bool
            Use the embedded quadrature's projected current streams.
        """
        spectrum = self.coarse_spectrum if coarse else self.spectrum
        plan = self.coarse_forward if coarse else self.forward
        assert spectrum is not None and plan is not None
        plan.execute()
        self.assemble_kernel(
            destination, spectrum, self.crop, self.inverse, self.active
        )

    def project_radial(
        self,
        polarisation: np.ndarray,
        projection: np.ndarray,
        destination: np.ndarray,
        *,
        first: bool,
        count: int,
    ) -> None:
        """Transform nodal P, gJ and g'J, combine, then project once in frequency.

        Parameters
        ----------
        polarisation : numpy.ndarray
            This chunk's collared non-plasma polarisation.
        projection : numpy.ndarray
            Scaled real radial projection for the current nodes.
        destination : numpy.ndarray
            Full modal fine spectrum, accumulated at the coarse bins.
        first : bool
            Clear the whole destination before the first chunk.
        count : int
            Populated node-component rows.
        """
        self.projected[0, :count] = polarisation[:count]
        self.projected[0, count:] = 0
        self.projected[1:] = self.sources
        self.forward.execute()
        self.assemble_kernel(
            self.nodal_spectrum, self.spectrum, self.crop, self.inverse, self.active
        )
        values = (
            self.nodal_spectrum[:count]
            .reshape(projection.shape[1], -1)
            .view(np.float64)
        )
        target = self.radial_projection.reshape(projection.shape[0], -1).view(
            np.float64
        )
        np.dot(projection, values, out=target)
        if first:
            destination[:] = 0
        assert self.scatter is not None
        self.scatter(destination, self.radial_projection, self.crop)

    def memory_ledger(self) -> dict[str, int]:
        """List every owned buffer and the measured transform-plan heap growth.

        Returns
        -------
        dict
            Persistent bytes, including partial-chunk padding and embedded rows.
        """
        arrays = (
            "current",
            "sources",
            "projected",
            "spectrum",
            "projection_scratch",
            "coarse_projected",
            "coarse_spectrum",
            "nodal_spectrum",
            "radial_projection",
            "inverse",
        )
        ledger = {
            f"held_current_{name}": 0
            if (value := getattr(self, name)) is None
            else value.nbytes
            for name in arrays
        }
        ledger["held_current_fft_plan_heap_growth"] = self.plan_heap_growth
        return ledger
