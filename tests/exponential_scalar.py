# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A scalar fixed-step PEC reference march for SEP 16's long-march witnesses.

One complex bin, ``u' = D(z) u + F(z, u)``, with the cumulative operator ``Phi`` known
in closed form. The march is the predictor-evaluate-correct recurrence of the production
method -- a ``q``-node predictor, one source call at the predicted endpoint, a
``q + 1``-node corrector through it, the sample stored with the *predicted* state --
in either formulation of SEP 16 section 4.2:

- ``"interaction"``: the transported-source (Lawson) history of SEP 13, each sample
  carried to the anchor by ``U(z_n, z_j) = exp(Phi(z_n) - Phi(z_j))``;
- ``"exponential"``: untransported samples integrated against the reference flow
  ``exp(h B_n)``, ``B_n = (Phi(z_n + h) - Phi(z_n)) / h``, with the linear
  compensation ``C_j = (D(z_j) - B_n) u_j`` treated as ``"transported"`` (in the
  reference flow's frame), ``"untransported"`` (interpolated with the source) or
  ``"chord"`` (omitted, the excluded control).

Starting values are exact, so every error is the recurrence's. The weights come from
the production algebra (:func:`solprop.stepper.multistep.integrated_weights` and
:func:`solprop.backends.exponential.exponential_weights`); the recurrence itself is
written here independently of the backends' march.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from solprop.backends.exponential import exponential_weights
from solprop.stepper.multistep import integrated_weights

__all__ = ["ScalarProblem", "march"]


@dataclass(frozen=True)
class ScalarProblem:
    """One bin's equation and its exact solution.

    Attributes
    ----------
    rate : callable
        ``D(z)``, 1/m.
    phase : callable
        ``Phi(z) = int_0^z D``, dimensionless.
    source : callable
        ``F(z, u)``.
    exact : callable
        ``u(z)``.
    """

    rate: Callable[[float], complex]
    phase: Callable[[float], complex]
    source: Callable[[float, complex], complex]
    exact: Callable[[float], complex]


def march(
    problem: ScalarProblem,
    length: float,
    steps: int,
    order: int,
    formulation: str,
    compensation: str = "transported",
) -> float:
    """March ``steps`` equal steps of PEC ``order``/``order + 1``; return the error.

    Returns
    -------
    float
        ``|u_N - u(L)| / |u(L)|``.
    """
    h = length / steps
    # Exact history of `order` points behind and at the anchor, newest last.
    positions = [h * index for index in range(order)]
    states = [problem.exact(z) for z in positions]
    samples = [problem.source(z, u) for z, u in zip(positions, states, strict=True)]
    u = states[-1]
    predictor_nodes = -np.arange(order, dtype=float)
    corrector_nodes = np.concatenate(([1.0], predictor_nodes))
    for step in range(order - 1, steps):
        z = h * step
        history = _History(
            positions[-order:][::-1], states[-order:][::-1], samples[-order:][::-1]
        )
        flow = np.exp(problem.phase(z + h) - problem.phase(z))
        if formulation == "interaction":
            carried = [
                np.exp(problem.phase(z) - problem.phase(zj)) * fj
                for zj, fj in zip(history.positions, history.sources, strict=True)
            ]
            predicted = flow * (
                u + h * np.dot(integrated_weights(predictor_nodes), carried)
            )
            new = problem.source(z + h, predicted)
            through = [np.exp(problem.phase(z) - problem.phase(z + h)) * new, *carried]
            u_next = flow * (
                u + h * np.dot(integrated_weights(corrector_nodes), through)
            )
        else:
            reference = (problem.phase(z + h) - problem.phase(z)) / h
            frame = _Frame(problem, z, h, reference, compensation)
            predicted = flow * u + frame.increment(predictor_nodes, history)
            new = problem.source(z + h, predicted)
            u_next = flow * u + frame.increment(
                corrector_nodes,
                _History(
                    [z + h, *history.positions],
                    [predicted, *history.states],
                    [new, *history.sources],
                ),
            )
        positions.append(z + h)
        states.append(predicted)
        samples.append(new)
        u = u_next
    exact = problem.exact(h * steps)
    return float(abs(u - exact) / abs(exact))


@dataclass(frozen=True)
class _History:
    """Samples newest first: positions, the states they were taken at, sources."""

    positions: list[float]
    states: list[complex]
    sources: list[complex]


@dataclass(frozen=True)
class _Frame:
    """One exponential step's reference and compensation treatment."""

    problem: ScalarProblem
    z: float
    h: float
    reference: complex
    compensation: str

    def increment(self, nodes: np.ndarray, history: _History) -> complex:
        """Return ``h sum_j w_j F_j`` plus the step's compensation."""
        h, reference = self.h, self.reference
        exponential = exponential_weights(nodes, h * reference)
        total = h * np.dot(exponential, history.sources)
        terms = [
            (self.problem.rate(zj) - reference) * sj
            for zj, sj in zip(history.positions, history.states, strict=True)
        ]
        if self.compensation == "untransported":
            total += h * np.dot(exponential, terms)
        elif self.compensation == "transported":
            adams = integrated_weights(nodes)
            total += h * sum(
                weight * np.exp((self.z + h - zj) * reference) * cj
                for weight, zj, cj in zip(adams, history.positions, terms, strict=True)
            )
        return complex(total)
