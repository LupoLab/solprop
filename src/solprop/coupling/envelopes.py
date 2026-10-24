# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Recorded benchmark envelopes of the scalar coupling model's provider pairs.

SEP 31 section 4.3 admits a provider pair to an approximate coupling model only with
an independent benchmark envelope for what the model omits, recorded with its
identity, range and independently bounded numerical error, and never extrapolated.
Each :class:`PairEnvelope` here is one such record. The numbers were measured by the
oracles in ``tests/coupling/test_envelopes.py``, which recompute them and hold the
recorded bounds; the manual page ``docs/guide/workflows/coupling.md`` describes them.

- :data:`CAPILLARY_ENVELOPE`: the scalar J0 overlap the model uses against the overlap
  of the capillary's own vector Marcatili mode fields
  (:meth:`solprop.modes.capillary.MarcatiliMode.field` and its normalisation) on an
  independent two-dimensional quadrature, for centred, displaced and tilted
  Gaussian beams and for a change of capillary radius.
- :data:`STEP_INDEX_ENVELOPE`: the weak-guidance LP01 overlap of two step-index
  fibres of one material pair and different radii against the full-vector flux
  overlap of their exact bound HE11 fields
  (:meth:`solprop.modes.stepindex.StepIndexMode.electromagnetic_field`). It replaces the
  contrast screen within its range.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = ["CAPILLARY_ENVELOPE", "STEP_INDEX_ENVELOPE", "PairEnvelope"]


@dataclass(frozen=True)
class PairEnvelope:
    """A recorded benchmark envelope of one provider pair under the scalar model.

    Attributes
    ----------
    identity : str
        The envelope's name, recorded in a coupling report.
    family : {'capillary', 'step_index'}
        The profile family of both members of the pair.
    ranges : collections.abc.Mapping
        Each range parameter's closed interval ``(low, high)``; a pair outside any
        one is not covered.
    bound : float
        The largest measured difference of captured energy share, scalar model
        against the independent reference, over the range, dimensionless, numerical
        error included: an energy discrepancy relative to the incident energy.
    numerical : float
        The independently estimated numerical error of the measurement, included
        in `bound`.
    reference : str
        What the independent reference is.
    """

    identity: str
    family: str
    ranges: Mapping[str, tuple[float, float]]
    bound: float
    numerical: float
    reference: str

    def __post_init__(self) -> None:
        """Hold the ranges read-only."""
        object.__setattr__(self, "ranges", MappingProxyType(dict(self.ranges)))

    def covers(self, parameters: Mapping[str, float]) -> str | None:
        """Return why a pair lies outside the envelope, or None if it is covered.

        Parameters
        ----------
        parameters : collections.abc.Mapping
            The pair's value of each range parameter.

        Returns
        -------
        str or None
            The first parameter outside its range, named with its value; None when
            every parameter is inside.
        """
        for name, (low, high) in self.ranges.items():
            value = parameters.get(name)
            if value is None or not low <= value <= high:
                return (
                    f"{name} = {value!r} lies outside the envelope "
                    f"{self.identity}'s range [{low:.6g}, {high:.6g}]"
                )
        return None


CAPILLARY_ENVELOPE = PairEnvelope(
    identity="capillary-he1m-scalar-vs-marcatili-2026-10",
    family="capillary",
    ranges={"modes": (1.0, 8.0), "radius_ratio": (0.5, 2.0)},
    bound=1e-12,
    numerical=1e-13,
    reference=(
        "overlap of MarcatiliMode.field (both transverse components, normalised by "
        "MarcatiliMode.normalisation) on an independent polar Gauss-Legendre "
        "quadrature; Gaussian waists 0.3a to 1.0a, displacement to 0.2a and tilt to "
        "1 mrad at 800 nm in a 75 um argon capillary with eight HE1m modes, and "
        "capillary radius ratios 0.5 to 2; largest difference 1.8e-14 of the "
        "incident energy, the reference quadrature's own rounding"
    ),
)
"""The capillary HE1m envelope; ``radius_ratio`` is source over target radius (1 for
an incident beam), ``modes`` the larger HE1m count of the pair."""


STEP_INDEX_ENVELOPE = PairEnvelope(
    identity="step-index-lp01-vs-vector-he11-2026-10",
    family="step_index",
    ranges={
        "contrast": (0.0, 0.01),
        "v_source": (1.5, 2.4),
        "v_target": (1.5, 2.4),
        "radius_ratio": (0.7, 1.43),
        "modes": (1.0, 1.0),
    },
    bound=3.0e-4,
    numerical=1e-12,
    reference=(
        "full-vector flux overlap of StepIndexMode.electromagnetic_field (exact bound "
        "HE11 even, E and H) of the two fibres on an independent polar quadrature; "
        "core 1.45, contrast 0.001 to 0.01, V 1.5 to 2.4 and radius ratios 0.7 to "
        "1.43 at 1064 nm (constant indices: the overlaps depend on V alone); largest "
        "difference 2.92e-4, at contrast 0.01, V 2.4 and ratio 0.7"
    ),
)
"""The weak-guidance step-index LP01 envelope; ``contrast`` is the largest
``|n^2 - n_ref^2|/(2 n_ref^2)`` of either fibre, ``v_*`` each fibre's normalised
frequency at its profile wavelength, ``radius_ratio`` source over target core radius,
``modes`` the larger LP0m count (LP01 only)."""
