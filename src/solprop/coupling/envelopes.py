# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Recorded benchmark envelopes of the approximate coupling models' provider pairs.

SEP 31 section 4.3 admits a provider pair to an approximate coupling model only with
an independent benchmark envelope for what the model omits, recorded with its
identity, range and independently bounded numerical error, and never extrapolated.
Each :class:`PairEnvelope` here is one such record. The numbers were measured by the
oracles in ``tests/coupling/test_envelopes.py``, which recompute them and hold the
recorded bounds; the manual page ``docs/guide/workflows/coupling.md`` describes them.

- :data:`CAPILLARY_ENVELOPE`: the scalar J0 profile the model projects onto against
  Marcatili's leaky HE1m core field :math:`J_0(u_m(1 - i\delta)r/a)`,
  :math:`\delta = |\nu_n|c/(\omega a)`, the physics the scalar model omits, swept
  over centred, displaced and tilted Gaussian beams and changes of capillary radius.
  Its bound scales linearly with the pair's :math:`\delta`.
- :data:`STEP_INDEX_ENVELOPE`: the weak-guidance LP01 overlap of two step-index
  fibres of one material pair and different radii against the full-vector flux
  overlap of their exact bound HE11 fields
  (:meth:`solprop.modes.stepindex.StepIndexMode.electromagnetic_field`). It replaces the
  contrast screen within its range.
- :data:`STEP_INDEX_BEAM_ENVELOPE`: an incident beam's scalar overlap with a weakly
  guiding fibre's LP01 profile against its full-vector flux coupling into the exact
  HE11 pair, the beam's H from its own plane-wave spectrum. The bound is the largest
  discrepancy over every incident field inside the widest admitted cone, so it
  covers any beam shape, alignment and polarisation; it scales linearly with the
  fibre's contrast.
- :data:`VECTOR_STEP_INDEX_ENVELOPE`: the vector model's flux recovery into a
  :class:`~solprop.modes.guide.StepIndexProvider` guide against an independent
  full-vector mode-matching solution of the planar entrance facet, the reflection
  and backward fields the model omits. Its bound scales with the square of the
  guide's contrast.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "CAPILLARY_ENVELOPE",
    "STEP_INDEX_BEAM_ENVELOPE",
    "STEP_INDEX_ENVELOPE",
    "VECTOR_STEP_INDEX_ENVELOPE",
    "PairEnvelope",
]


@dataclass(frozen=True)
class PairEnvelope:
    """A recorded benchmark envelope of one provider pair under an approximate model.

    Attributes
    ----------
    identity : str
        The envelope's name, recorded in a coupling report.
    family : {'capillary', 'step_index', 'step_index_vector'}
        The profile family of the pair's guided members: the target, and a modal
        source (an incident beam has none); for the vector model, the target
        guide's provider.
    ranges : collections.abc.Mapping
        Each range parameter's closed interval ``(low, high)``; a pair outside any
        one is not covered.
    bound : float
        The largest measured difference of captured energy share, model against
        the independent reference, over the range, dimensionless, numerical error
        included: an energy discrepancy relative to the incident energy.
    numerical : float
        The independently estimated numerical error of the measurement, included
        in `bound`.
    reference : str
        What the independent reference is.
    scaled_by : str or None, optional
        A range parameter the discrepancy grows with: the allowance is then `bound`
        times that parameter over its range's upper end (:meth:`allowance`). None
        (the default) holds `bound` over the whole range.
    """

    identity: str
    family: str
    ranges: Mapping[str, tuple[float, float]]
    bound: float
    numerical: float
    reference: str
    scaled_by: str | None = None

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

    def allowance(self, parameters: Mapping[str, float]) -> float:
        """Return the envelope's energy allowance for a covered pair.

        Parameters
        ----------
        parameters : collections.abc.Mapping
            The pair's value of each range parameter.

        Returns
        -------
        float
            Dimensionless: `bound`, scaled linearly by `scaled_by` when it is set.

        Examples
        --------
        >>> from solprop.coupling import CAPILLARY_ENVELOPE as envelope
        >>> high = envelope.ranges["leaky"][1]
        >>> envelope.allowance({"leaky": 0.5 * high}) == 0.5 * envelope.bound
        True
        """
        if self.scaled_by is None:
            return self.bound
        return self.bound * parameters[self.scaled_by] / self.ranges[self.scaled_by][1]


CAPILLARY_ENVELOPE = PairEnvelope(
    identity="capillary-he1m-scalar-vs-leaky-marcatili-2026-10",
    family="capillary",
    ranges={"modes": (1.0, 8.0), "radius_ratio": (0.5, 2.0), "leaky": (0.0, 1e-2)},
    bound=6e-3,
    numerical=1e-10,
    reference=(
        "Marcatili's leaky HE1m core field J0(u_m (1 - i delta) r/a), delta = "
        "|nu_n| c/(omega a), against the scalar J0 profile, both overlapped on an "
        "independent polar Gauss-Legendre rule at delta = 1e-2 (the larger of a "
        "pair's two capillaries): Gaussian beams of waist 0.3a to 1.0a, displaced to "
        "0.2a, tilted to 0.59/(k a) (1 mrad at a = 75 um and 800 nm), 1, 2, 4 and 8 "
        "HE1m modes, and coherent three-mode capillary exits into 8 modes at radius "
        "ratios 0.5 to 2. The discrepancy is the larger of the summed per-mode share "
        "changes and the least-squares captured share change: largest 2.9e-3, at "
        "equal radii (the leaky modes' nonorthogonality); bound twice that. It "
        "falls at least linearly with delta (halving delta halves the beam cases and "
        "quarters the radius cases), so the allowance is scaled by delta/1e-2"
    ),
    scaled_by="leaky",
)
"""The capillary HE1m envelope; ``radius_ratio`` is source over target radius (1 for
an incident beam), ``modes`` the larger HE1m count of the pair, ``leaky`` the larger
:meth:`~solprop.coupling.models.ScalarModeBasis.leaky` of the pair over the occupied
bins (about 5.0e-3 for a 37.5 um silica capillary at 800 nm)."""


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


STEP_INDEX_BEAM_ENVELOPE = PairEnvelope(
    identity="step-index-beam-lp01-vs-vector-he11-2026-10",
    family="step_index",
    ranges={"contrast": (0.0, 0.01), "v_target": (1.5, 2.4), "modes": (1.0, 1.0)},
    bound=1.0e-2,
    numerical=1e-12,
    reference=(
        "the full-vector flux coupling 1/4 int(E x h* + e* x H).z dA of an incident "
        "field into the exact unit-power HE11 even and odd modes "
        "(StepIndexMode.electromagnetic_field), the field's H and E_z from its own "
        "plane-wave spectrum in the core index, against the scalar LP01 overlap the "
        "model takes, both in the common measure M0; evaluated in the transverse "
        "Fourier plane by azimuthal decomposition and Gauss-Legendre Hankel "
        "transforms, inside the widest admitted cone (1 - cos theta = 1/8). The "
        "discrepancy of captured energy over incident energy, maximised over every "
        "field in that cone (the largest eigenvalue modulus of the difference of the "
        "two rank-two capture forms), is 0.467 Delta at V 1.5, falling to 0.284 "
        "Delta at V 2.4, for Delta 5e-4 to 1e-2, independent of the core index and "
        "the wavelength: largest 4.67e-3 at Delta 0.01 and V 1.5. Gaussian beams "
        "(waists 0.5a to 2.6a, displaced to 0.5a and tilted to half the acceptance "
        "angle along x and y, x-polarised) reach 3.3e-3. The bound is twice the "
        "largest per unit contrast, rounded up, so the allowance is scaled by "
        "Delta/1e-2; refining the quadrature changes the maximum by 2e-13"
    ),
    scaled_by="contrast",
)
"""The beam-into-fibre LP01 envelope; ``contrast`` is the fibre's largest
``|n^2 - n_ref^2|/(2 n_ref^2)`` against its core over the occupied bins, ``v_target``
its normalised frequency at the profile wavelength, ``modes`` the number of distinct
LP0m profiles (LP01 only: V below 2.405 guides no other). The bound holds for any
incident field inside the admitted cone; the angular item counts what lies outside
it, and the source-metric item the beam's medium."""

VECTOR_STEP_INDEX_ENVELOPE = PairEnvelope(
    identity="step-index-vector-facet-mode-matching-2026-10",
    family="step_index_vector",
    ranges={
        "contrast": (0.0, 0.05),
        "v": (1.5, 8.0),
        "contrast_squared": (0.0, 2.5e-3),
    },
    bound=1.4e-3,
    numerical=2e-5,
    reference=(
        "independent full-vector mode matching of the planar entrance facet "
        "(incident and reflected fields in the core-index input medium, the "
        "fibre's forward guided and radiation fields, tangential E and H "
        "continuous), solved per azimuthal order in a conducting cylinder of 8 "
        "core radii on its own TE/TM Bessel basis, against the model's flux "
        "pairing with the same incident H: x-polarised Gaussian beams of waist "
        "1a to 2a, displaced to 0.5a and tilted to 0.3 NA, into HE11 at V 1.5 to "
        "8 and HE12, HE21, TE01 and TM01 at V 4; contrast 0.05 (the range's top), "
        "input medium of the core's index. Largest omitted share 6.9e-4 (HE11, "
        "V 1.5, a 2a waist); bound twice that; refining the basis and widening "
        "the cylinder changes it by 1.1e-5. It is second order in the index step "
        "(halving the contrast quarters it, and at weak guidance it is the local "
        "Fresnel estimate to a few per cent), so the allowance is scaled by the "
        "squared contrast over 0.05**2"
    ),
    scaled_by="contrast_squared",
)
"""The vector model's step-index facet envelope; ``contrast`` is the guide's
``|n^2 - n_ref^2|/(2 n_ref^2)``, ``v`` its normalised frequency at the entrance and
``contrast_squared`` the square of the contrast, each weighted by the incident energy
over the occupied bins. It bounds the reflection and backward fields at the facet,
not the incident beam's H closure (the angular item) or the guide's tabulation; it
covers a beam in the core's index only, since a bulk index change is a separate
planar interface (SEP 31, section 4.3)."""
