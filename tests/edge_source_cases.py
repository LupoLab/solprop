# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared manufactured problems and oracles for the SEP 26 edge-free TDSE source.

No atomic provider is involved: a manufactured remainder ``S`` is attached to a real
assembled capillary problem, every node carries the same ``S``, and the source is
mapped back to raw DFT units of ``S`` by the exactly known projection and current
factor. Oracles are written in those raw units, ``x_k = sum_n x_n exp(-2 pi i k n /
N)`` with its phase origin at ``n = 0`` and samples at ``t_n = (n - N/2) dt``, so a
continuous transform enters as ``(-1)^k F(omega_k) / dt`` (SEP 26 section 4.4).
"""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from solprop.backends.numpy_ref import passes
from solprop.backends.numpy_ref.tdse import (
    FFT_ERROR_CONSTANT,
    _current_operands,
    _current_part,
    atomic_edge_source,
    atomic_sources,
    project_atomic_sources,
)
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.tdse import (
    TDSEAtomicWindow,
    TDSEBasis,
    TDSECalibration,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.stepper import PIController
from solprop.stepper.multistep import Multistep
from solprop.translate.linop import power_weight_at
from solprop.translate.problem import KernelProblem, assemble
from solprop.translate.responses import prefactor_at
from solprop.translate.stage import StageScalars, scalars
from solprop.translate.tdse import (
    AtomicCallBudget,
    TDSEDescriptor,
    TDSEResult,
    lattice_times,
    refine_response_grid,
    waveform_identity,
)
from solprop.translate.tdse_source import current_numerator

EPS = float(np.finfo(float).eps)

LIMITS = (400e-9, 1.6e-6)
"""The default fixture band: 0.4-1.6 um on a 60 fs window."""


@dataclass(frozen=True)
class EdgeCase:
    """One assembled problem with a manufactured TDSE descriptor.

    Attributes
    ----------
    problem : KernelProblem
        Validated problem carrying the descriptor and the current numerator.
    grid : FieldGrid
        The refined response grid.
    bundle : StageScalars
        Stage scalars at z = 0.
    prefactor : numpy.ndarray
        Stage polarisation prefactor.
    """

    problem: KernelProblem
    grid: FieldGrid
    bundle: StageScalars
    prefactor: np.ndarray

    @property
    def times(self) -> np.ndarray:
        """Return the fine lattice times in s."""
        return lattice_times(self.grid.n_to, self.grid.dt_fine)

    @property
    def active(self) -> np.ndarray:
        """Return the active coarse bins."""
        return self.problem.grids.active_mask

    @property
    def omega(self) -> np.ndarray:
        """Return the coarse angular frequencies in rad/s."""
        return self.problem.grids.omega

    @property
    def descriptor(self) -> TDSEDescriptor:
        """Return the descriptor."""
        assert self.problem.tdse is not None
        return self.problem.tdse

    def to_source(self, raw: np.ndarray) -> np.ndarray:
        """Map a raw coarse spectrum of ``S`` to the source's modal units.

        Every node carries the same ``S``, so the projection is ``K S`` with ``K =
        sum_n P_n / taper`` times density and scale, and the source is the current
        factor times ``K`` times the raw DFT, on the coarse bins.
        """
        problem, bundle = self.problem, self.bundle
        ones = np.ones((problem.transform.n_nodes, 1, 1))
        weight = passes.project(
            ones, problem.transform.projection, bundle.taper_scale, 1
        )
        scale = bundle.density * self.descriptor.remainder_scale
        factor = problem.current_numerator / power_weight_at(problem, bundle.z1)
        return factor * (scale * weight) * raw

    def raw_dft(self, values: np.ndarray) -> np.ndarray:
        """Return the raw forward DFT cropped to the coarse bins."""
        return np.fft.rfft(values)[self.problem.grids.crop_index]


def make_case(  # noqa: PLR0913 - one keyword per lattice choice of the fixture.
    *,
    window: TDSEAtomicWindow | None = None,
    trange: float = 60e-15,
    limits: tuple[float, float] = LIMITS,
    refinement: int = 8,
    fft_size: str = "pow2",
    n_r: int = 4,
    kerr_model: str = "tdse_native",
) -> EdgeCase:
    """Assemble a capillary problem and attach a manufactured SAE descriptor.

    Parameters
    ----------
    window : TDSEAtomicWindow or None
        Optional atomic window; None gates with the grid collar.
    trange : float
        Requested time window in s.
    limits : tuple of float
        Wavelength limits in m.
    refinement : int
        Response refinement.
    fft_size : {'pow2', 'smooth'}
        Coarse transform sizing.
    n_r : int
        Radial nodes.
    kerr_model : str
        ``tdse_native`` for the gated source, or a cubic replacement, whose source
        is ungated.

    Returns
    -------
    EdgeCase
        The assembled case.
    """
    parent = FieldGrid.from_limits(0.001, 800e-9, limits, trange, fft_size=fft_size)
    grid = refine_response_grid(parent, TDSESampling(refinement))
    mode = MarcatiliMode(125e-6, cladding=1.45, loss=False)
    problem = assemble(
        grid,
        mode,
        (),
        np.zeros((1, 1, grid.n_omega), dtype=np.complex128),
        controller=PIController(rtol=1e-2, atol=1e-14),
        save_planes=[0.001],
        dz_init=1e-5,
        dz_max=1e-4,
        density=1.0,
        quadrature=transverse_quadrature([mode], n_r=n_r),
        components=("x",),
        integrator=Multistep(formulation="interaction"),
    )
    basis = TDSEBasis(
        "manufactured", (0.0, 1e-9), (8,), 3, 1e-16, source="edge-free fixture"
    )
    calibration = TDSECalibration(
        "edge-free",
        "research",
        basis.identity,
        limits,
        source="edge-free fixture",
        limitations=("no atomic qualification",),
    )
    response = TDSEGasResponse(
        "helium",
        basis,
        1e-19,
        kerr_model,  # pyright: ignore[reportArgumentType]
        1.0,
        calibration,
        TDSESampling(refinement),
        atomic_window=window,
    )
    problem = dataclasses.replace(
        problem,
        tdse=TDSEDescriptor.from_response(response, grid),
        current_numerator=current_numerator(grid),
    )
    return EdgeCase(
        problem, grid, scalars(problem, 0.0, 0.0), prefactor_at(problem.responses, 0.0)
    )


def case_from_problem(problem: KernelProblem, grid: FieldGrid, z: float) -> EdgeCase:
    """Wrap a translated TDSE problem at one position for direct evaluation.

    Parameters
    ----------
    problem : KernelProblem
        A problem from ``Simulation.translate``.
    grid : FieldGrid
        Its response grid.
    z : float
        Position in m.

    Returns
    -------
    EdgeCase
        The same problem, with its stage scalars and prefactor at ``z``.
    """
    return EdgeCase(
        problem, grid, scalars(problem, z, z), prefactor_at(problem.responses, z)
    )


@dataclass(frozen=True)
class Evaluation:
    """The source of one manufactured remainder, with its per-bin budget.

    Attributes
    ----------
    source : numpy.ndarray
        Modal source, ``(1, 1, n_omega)``.
    diagnostics : dict
        The projection diagnostic.
    budget : numpy.ndarray or None
        Per-bin section 4.5 rounding budget of the current, in source units, for a
        gated filter.
    """

    source: np.ndarray
    diagnostics: dict
    budget: np.ndarray | None


def evaluate(case: EdgeCase, remainder: np.ndarray, first=None) -> Evaluation:
    """Assemble and project a manufactured remainder on every node.

    Parameters
    ----------
    case : EdgeCase
        The problem.
    remainder : numpy.ndarray
        ``S`` in A/m² per atom, shape ``(n_to,)`` or ``(n_nodes, n_to)``.
    first : numpy.ndarray or None
        Optional first-order current; it cancels in ``J - j_1``.

    Returns
    -------
    Evaluation
        Source, diagnostics and budget.
    """
    problem = case.problem
    rows = problem.transform.n_nodes
    values = np.broadcast_to(remainder, (rows, case.grid.n_to)).astype(float)
    linear = (
        np.zeros_like(values) if first is None else np.broadcast_to(first, values.shape)
    )
    fields = np.zeros_like(values)
    result = TDSEResult(
        values + linear,
        np.array(linear, dtype=float),
        input_identity=waveform_identity(fields),
        configuration_identity=case.descriptor.identity,
        valid=np.ones(values.shape, dtype=bool),
        shadow_orders=(1,),
    )
    polarisation, current = atomic_sources(fields, result, problem, case.bundle)
    edge = atomic_edge_source(result, problem, case.bundle)
    source, diagnostics = project_atomic_sources(
        polarisation, current, result, problem, case.bundle, case.prefactor, edge=edge
    )
    operands = _current_operands(current, edge, result, problem, case.bundle)
    part = _current_part(operands, problem.transform.projection, problem, case.bundle)
    budget = (
        None
        if part.bounds is None
        else part.bounds[0] + part.bounds[1] + part.bounds[2]
    )
    return Evaluation(source, diagnostics, budget)


def spectral_derivative_oracle(case: EdgeCase, remainder: np.ndarray):
    """Return ``DFT[g IDFT(i omega DFT S)] / (i omega)`` and its own budget.

    The oracle never forms ``g'``: it differentiates ``S`` spectrally (Nyquist bin
    zeroed, as for a real trigonometric interpolant) and gates the derivative. Its
    budget (SEP 26 section 4.5) is ``c_F eps log2 N sqrt(N) (omega_N ||S||_2 +
    ||g IDFT(i omega DFT S)||_2) / omega_lo`` in raw units, per bin.

    Returns
    -------
    tuple
        Raw coarse oracle spectrum on active bins (zero elsewhere) and the per-bin
        raw budget, a scalar.
    """
    samples, spacing = case.grid.n_to, case.grid.dt_fine
    omega = 2 * np.pi * np.fft.rfftfreq(samples, spacing)
    derivative_spectrum = 1j * omega * np.fft.rfft(remainder)
    derivative_spectrum[-1] = 0.0
    derivative = np.fft.irfft(derivative_spectrum, n=samples)
    gate = _gate(case)
    gated = gate * derivative
    active = case.active
    oracle = np.zeros(case.omega.size, dtype=complex)
    oracle[active] = case.raw_dft(gated)[active] / (1j * case.omega[active])
    omega_lo = float(np.min(case.omega[active]))
    budget = (
        FFT_ERROR_CONSTANT
        * EPS
        * math.ceil(math.log2(samples))
        * math.sqrt(samples)
        * (np.pi / spacing * np.linalg.norm(remainder) + np.linalg.norm(gated))
        / omega_lo
    )
    return oracle, budget


def _gate(case: EdgeCase) -> np.ndarray:
    """Return the source gate on the lattice: window or collar."""
    gate = case.descriptor.source_gate()
    return case.problem.grids.t_window_fine if gate is None else gate


def gate_values(case: EdgeCase) -> np.ndarray:
    """Return the source gate on the lattice."""
    return _gate(case)


def truncating_source(case: EdgeCase, remainder: np.ndarray) -> np.ndarray:
    """Return the truncating source ``c DFT[g S]`` of the same gate, as an oracle.

    It is the edge-free source without its edge term: the current switched off in
    the trailing taper, which the removed ``time_collar`` filter propagated.
    """
    return case.to_source(case.raw_dft(_gate(case) * remainder))


def steady_residual(case: EdgeCase) -> np.ndarray:
    """Return the raw edge-free source of ``S = 1`` per coarse bin (the alias)."""
    gate = _gate(case)
    derivative = case.descriptor.source_gate_derivative()
    active = case.active
    residual = np.zeros(case.omega.size, dtype=complex)
    residual[active] = case.raw_dft(gate)[active] - case.raw_dft(derivative)[active] / (
        1j * case.omega[active]
    )
    return residual


class ManufacturedBinding:
    """Return a manufactured remainder ``S(t)`` beside a large linear shadow.

    The total current is ``j_1 + S`` with ``j_1 = linear E``, so ``S`` reaches the
    source only through the backend's own ``J - j_1``; ``law(times)`` gives ``S`` on
    the lattice, independent of the field, and two chunks are charged per call. The
    default ``linear = 2^-90`` keeps ``u |j_1|`` (about 1e-37 A m at 1e6 V/m) ten
    orders below a 1e-27 A m remainder, so the cancellation keeps ``S``.
    """

    def __init__(self, descriptor, limit, law=None, *, linear=2.0**-90):
        self.linear = linear
        self.descriptor = descriptor
        self.budget = AtomicCallBudget(limit, 2)
        self.law = law
        self.purposes = []
        self.closed = False

    @property
    def ledger(self):
        return self.budget.snapshot()

    def reserve(self, logical_calls):
        self.budget.reserve(logical_calls, z=0.0)

    def release_reservation(self):
        self.budget.release_reservation()

    def restore_ledger(self, ledger):
        self.budget.restore(ledger)

    def record_trial(self, *, accepted):
        del accepted

    def evaluate(self, fields, *, z, purpose, density=None):
        del density
        self.purposes.append(purpose)
        self.budget.begin_evaluation(purpose=purpose)
        for _ in range(2):
            self.budget.charge(purpose=purpose, z=z)
        first = self.linear * fields
        times = lattice_times(self.descriptor.samples, self.descriptor.spacing)
        remainder = (
            np.zeros_like(fields)
            if self.law is None
            else np.broadcast_to(self.law(times), fields.shape)
        )
        return TDSEResult(
            first + remainder,
            first,
            input_identity=waveform_identity(fields),
            configuration_identity=self.descriptor.identity,
            valid=np.ones(fields.shape, dtype=bool),
            shadow_orders=(1,),
        )

    def metadata(self):
        return {"kind": "manufactured edge-free fixture", "closed": self.closed}

    def memory_ledger(self):
        return {}

    def close(self):
        self.closed = True
