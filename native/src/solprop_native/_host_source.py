# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The TDSE response as the native Adams march's host source (solprop SEP 49).

The march calls three entry points through the binding layer, which takes the
interpreter lock for each: ``begin(cost)`` before an attempt, ``evaluate`` at each
source evaluation, and ``end(outcome)`` after its decision. They are the reference's
binding calls at the same points of an attempt -- ``reserve``, the evaluation with
solprop's shared source assembly (``solprop.backends.tdse_source``), ``record_trial``
and ``release_reservation`` -- and the source each evaluation returns is ``F`` of
SEP 49 section 4, which the engine finishes as it finishes every source. An exception
raised here is kept, and the march reports only that one was; the session raises the
kept exception unchanged, with every later one attached as a note.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from solprop.backends import tdse_source
from solprop.translate.problem import PropagationProblem
from solprop.translate.responses import prefactor_at
from solprop.translate.stage import StageScalars, StageTerms, combine, scalars
from solprop.translate.tdse import TDSEBinding, TDSEPurpose

__all__ = ["TDSEHostSource"]

# The march's attempt outcomes (`HostOutcome` in host_source.hpp).
_REJECTED, _ACCEPTED, _ABANDONED = 0, 1, 2


class TDSEHostSource:
    """The driver's TDSE binding as the native march's host source.

    - ``binding``: the driver's binding, which the backend never owns.
    - ``terms``: the stage seam's fixed arrays, from which the prefactor is combined as
      the reference combines it.
    """

    def __init__(
        self, binding: TDSEBinding, problem: PropagationProblem, terms: StageTerms
    ) -> None:
        self._binding = binding
        self._problem = problem
        self._terms = terms
        self.error: BaseException | None = None
        # The last evaluation's purpose, position and projection diagnostics, as the
        # reference records them (`tdse_last_evaluation`).
        self.last: dict[str, object] = {
            "purpose": None,
            "z": None,
            "projection": None,
        }

    def begin(self, cost: int) -> None:
        """Reserve every atomic call of the attempt's `cost` evaluations.

        A new attempt starts with no kept exception: one the session did not take was
        the previous attempt's, and was reported with it.
        """
        self.error = None
        self._kept(lambda: self._binding.reserve(int(cost)))

    def evaluate(
        self,
        field: npt.NDArray[np.complex128],
        z: float,
        purpose: TDSEPurpose,
        out: npt.NDArray[np.complex128],
    ) -> None:
        """Evaluate the source of the lab-frame `field` at `z`, m, into `out`.

        Both arrays are the engine's host rows, valid for this call alone; `field`
        is read-only.
        """

        def _run() -> None:
            problem = self._problem
            lab = np.array(field, dtype=np.complex128, copy=True).reshape(1, 1, -1)
            bundle = scalars(problem, float(z), float(z))
            fields = tdse_source.node_fields(problem, lab, bundle)
            result = self._binding.evaluate(
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
            out[:] = source.reshape(-1)
            self.last = {"purpose": purpose, "z": bundle.z1, "projection": projection}

        self._kept(_run)

    def end(self, outcome: int) -> None:
        """Record a decided attempt's outcome, then release what it reserved.

        An abandoned attempt (a failure or a floor before its decision) is not
        recorded, as the reference records none; its reservation is released all the
        same.
        """

        def _run() -> None:
            try:
                if outcome != _ABANDONED:
                    self._binding.record_trial(accepted=outcome == _ACCEPTED)
            finally:
                self._binding.release_reservation()

        self._kept(_run)

    def take(self) -> BaseException | None:
        """Return the kept exception, once."""
        error, self.error = self.error, None
        return error

    def _prefactor(self, bundle: StageScalars) -> Any:
        """Return the stage prefactor as the reference forms it (`HostDriver`)."""
        if bundle.prefactor is None or self._terms.prefactor is None:
            return prefactor_at(self._problem.responses, bundle.z1)
        return combine(bundle.prefactor, self._terms.prefactor)

    def _kept(self, call: Any) -> None:
        """Run an entry point, keeping its exception: the first, later ones as notes."""
        try:
            call()
        except BaseException as error:
            if self.error is None:
                self.error = error
            elif error is not self.error:
                self.error.add_note(f"then, on the host source's cleanup: {error!r}")
            raise
