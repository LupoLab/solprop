# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Count the prototype's owned arrays before starting Julia."""

import numpy as np

from solprop.backends.julia._backend import _compensated, _varying
from solprop.stepper import LowStorageRungeKutta, Multistep
from solprop.translate.problem import KernelProblem


def _plasma_bytes(problem: KernelProblem, chunk: int) -> int:
    """Count current arrays, omitting modal projections on physical-space chunks."""
    if not problem.responses.plasma_channels:
        return 0
    nt, nw = problem.grids.n_to, problem.shape[-1]
    free = problem.transform.kind == "free"
    columns = chunk if free else problem.shape[1]
    channel = problem.responses.plasma_channels[0]
    total = channel.rate.coefficients.nbytes + 72
    total += problem.responses.plasma_integral.taps.nbytes + 8 * (nt + nw)
    total += 48 * nt * chunk + 32 * (nt // 2 + 1) * columns
    total += 16 * problem.transform.n_nodes
    return total if free else total + 32 * nt * columns


def owned_bytes(problem: KernelProblem) -> int:
    """Count the Julia configuration, work arrays and host power weight.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        Supported scalar modal or spatial field-grid contract.

    Returns
    -------
    int
        Persistent array bytes, excluding opaque FFT plans and runtime memory.
    """
    nm, nw = problem.shape[1], problem.shape[-1]
    nt, nc = problem.grids.n_to, problem.grids.n_t
    nodes = problem.transform.n_nodes
    chunk = min(problem.transform.node_chunk, nodes)
    stages = problem.stepper.tableau.n_stages
    state = 16 * nw * nm
    if problem.transform.kind == "free":
        # Shared frequency/transverse coefficient vectors, never dense operators.
        config = 107 * nw + 32 * nm + 16 * nt + 8 * nc
        config += 8 * (stages * stages + 3 * stages)
        low_storage = isinstance(problem.stepper.integrator, LowStorageRungeKutta)
        envelope = problem.grid_kind == "envelope"
        work = (8 if low_storage else 10 + stages) * state + 24 * nm
        nf = nt if envelope else nt // 2 + 1
        work += chunk * (
            32 * nf
            + (64 if envelope else 40) * nt
            + (16 if envelope else 8) * nc
            + 16 * nw
        )
        return config + work + _plasma_bytes(problem, chunk)
    config = 2 * state + 24 * nm * nodes + 26 * nw + 16 * nt + 8 * nc
    config += 8 * (stages * stages + 3 * stages)
    geometry = problem.transform.geometry
    if geometry is not None and np.any(geometry.absorber_rate):
        config += 8 * nm * nm
    envelope = problem.grid_kind == "envelope"
    nf = nt if envelope else nt // 2 + 1
    work = (6 + 3 * stages) * state + 48 * nf * nm
    work += (64 if envelope else 32) * nt * nm
    work += (32 if envelope else 24) * nt * chunk
    work += (16 if envelope else 8) * nc * nm + 24 * nm
    plasma = _plasma_bytes(problem, chunk)
    if problem.responses.raman_terms:
        work += 16 * (nt + 1) + (32 if envelope else 24) * nt * chunk
        work += 16 * (nt + 1) * chunk
    path = 0
    if _varying(problem):
        n_phi = (
            1
            if problem.linear.phi_expansion is None
            else len(problem.linear.phi_expansion.terms)
        )
        n_pre = (
            1
            if problem.responses.prefactor_expansion is None
            else len(problem.responses.prefactor_expansion.terms)
        )
        path = (n_phi + n_pre) * state + 8 * stages * (n_phi + n_pre + 3)
    adams = 0
    method = problem.stepper.integrator
    if isinstance(method, Multistep):
        slots, width = method.order + 2, method.order + 1
        compensated = _compensated(problem)
        n_phi = (
            1
            if problem.linear.phi_expansion is None
            else len(problem.linear.phi_expansion.terms)
        )
        adams = state * (slots + width + compensated * (2 * slots + width))
        adams += 16 * width * width + 96 * width + 8 * (n_phi + slots) + 24 + 24 * nm
    return (
        config + work + plasma + path + adams + (8 * nw * nm if geometry is None else 0)
    )
