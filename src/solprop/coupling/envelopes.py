# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Recorded benchmark envelopes of the approximate coupling models' provider pairs.

SEP 31 section 4.3 admits a provider pair to an approximate coupling model only with
an independent benchmark envelope for what the model omits, recorded with its
identity, range and independently bounded numerical error, and never extrapolated.
Each :class:`PairEnvelope` or :class:`FieldEnvelope` is one such record:

- :data:`CAPILLARY_ENVELOPE`: the scalar J0 profile against Marcatili's leaky HE1m
  core field. Its discrepancy depends on the field (mode m departs by about
  :math:`u_m\delta`), so no fixed bound covers every field usefully: it is a
  :class:`FieldEnvelope`, evaluated for the field coupled in closed form
  (``solprop.coupling._leaky``), with its range and numerical error recorded.
- :data:`STEP_INDEX_ENVELOPE`: the weak-guidance LP01 overlap of two step-index fibres
  against the full-vector flux overlap of their exact HE11 fields
  (:meth:`solprop.modes.stepindex.StepIndexMode.electromagnetic_field`); it replaces
  the contrast screen within its range.
- :data:`STEP_INDEX_BEAM_ENVELOPE`: a beam's scalar LP01 overlap against its
  full-vector coupling into the exact HE11 pair, maximised over every field inside
  the widest admitted cone; it scales linearly with the fibre's contrast.
- :data:`VECTOR_STEP_INDEX_ENVELOPE`: the vector model into a
  :class:`~solprop.modes.stepindex_vector.StepIndexProvider` guide against a
  full-vector mode-matching solution of the entrance facet; it scales with the
  square of the contrast.

The numbers were measured by the oracles in ``tests/coupling/test_envelopes.py``, which
recompute them and hold the recorded bounds; ``docs/guide/workflows/coupling.md``,
section "The admission budget and the report", describes each envelope.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "CAPILLARY_ENVELOPE",
    "STEP_INDEX_BEAM_ENVELOPE",
    "STEP_INDEX_ENVELOPE",
    "VECTOR_STEP_INDEX_ENVELOPE",
    "FieldEnvelope",
    "PairEnvelope",
]


def _outside(
    identity: str,
    ranges: Mapping[str, tuple[float, float]],
    parameters: Mapping[str, float],
) -> str | None:
    """Return the first range parameter outside its interval, named, or None."""
    for name, (low, high) in ranges.items():
        value = parameters.get(name)
        if value is None or not low <= value <= high:
            return (
                f"{name} = {value!r} lies outside the envelope "
                f"{identity}'s range [{low:.6g}, {high:.6g}]"
            )
    return None


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
        return _outside(self.identity, self.ranges, parameters)

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
        >>> from solprop.coupling import STEP_INDEX_BEAM_ENVELOPE as envelope
        >>> high = envelope.ranges["contrast"][1]
        >>> envelope.allowance({"contrast": 0.5 * high}) == 0.5 * envelope.bound
        True
        """
        if self.scaled_by is None:
            return self.bound
        return self.bound * parameters[self.scaled_by] / self.ranges[self.scaled_by][1]


@dataclass(frozen=True)
class FieldEnvelope:
    """An approximate model's omitted physics, evaluated for each field coupled.

    Where the discrepancy depends so strongly on the field that no recorded bound
    covers every field usefully, the coupling evaluates it for the field at hand and
    reports it as the ``field`` range parameter; the record holds the range the
    evaluation is qualified over, its numerical error and what it is checked
    against. The same interface as :class:`PairEnvelope`.

    Attributes
    ----------
    identity : str
        The envelope's name, recorded in a coupling report.
    family : {'capillary'}
        The profile family of the pair's guided members.
    ranges : collections.abc.Mapping
        Each range parameter's closed interval ``(low, high)``; a pair outside any
        one is not covered.
    numerical : float
        The evaluation's numerical error, dimensionless, added to the allowance.
    reference : str
        What the evaluation computes and what it is checked against.
    """

    identity: str
    family: str
    ranges: Mapping[str, tuple[float, float]]
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
        return _outside(self.identity, self.ranges, parameters)

    def allowance(self, parameters: Mapping[str, float]) -> float:
        """Return the envelope's energy allowance for a covered field.

        Parameters
        ----------
        parameters : collections.abc.Mapping
            The pair's range parameters and ``field``, the evaluated discrepancy.

        Returns
        -------
        float
            Dimensionless: ``field`` plus `numerical`.

        Examples
        --------
        >>> from solprop.coupling import CAPILLARY_ENVELOPE as envelope
        >>> envelope.allowance({"field": 2e-3}) == 2e-3 + envelope.numerical
        True
        """
        return parameters["field"] + self.numerical


CAPILLARY_ENVELOPE = FieldEnvelope(
    identity="capillary-he1m-scalar-vs-leaky-marcatili-field-2026-11",
    family="capillary",
    ranges={"modes": (1.0, 24.0), "radius_ratio": (0.5, 2.0), "leaky": (0.0, 1e-2)},
    numerical=1e-8,
    reference=(
        "Marcatili's leaky HE1m core field J0(u_m (1 - i delta) r/a), delta = "
        "|nu_n| c/(omega a), against the scalar J0 profile, evaluated for the field "
        "coupled at every bin: the summed change of the power launched into each "
        "channel, |G^-1 o|^2 on unit profiles, relative to the incident power (a "
        "beam's energy on its plane, a capillary exit's modal power). Closed-form "
        "Lommel overlaps; a beam's leaky overlaps from the Taylor series of the leaky "
        "profile in delta, its remainder bounded; a capillary exit's exactly. "
        "Numerical error below 1e-8: Lommel's midpoint branch, 2e-9 relative. "
        "Checked against an independent polar Gauss-Legendre quadrature for centred, "
        "displaced, tilted and structured beams on Cartesian and radial planes into "
        "4 to 24 HE1m, and capillary exits of random complex amplitudes at radius "
        "ratios 0.5 to 2 into up to 24 HE1m"
    ),
)
"""The capillary HE1m envelope, evaluated per field; ``radius_ratio`` is source over
target radius (1 for an incident beam), ``modes`` the highest HE1m radial order of the
pair, ``leaky`` the larger :meth:`~solprop.coupling.models.ScalarModeBasis.leaky` of
the pair over the occupied bins (about 5.0e-3 for a 37.5 um silica capillary at
800 nm), ``field`` the field's evaluated discrepancy weighted by incident energy."""


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
``modes`` the highest LP0m radial order of the pair (LP01 only)."""


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
        "Delta at V 2.4 (conservative there: the cone's edge truncates the widening "
        "HE11 spectrum); at V 1.5 it is linear in Delta from 5e-4 to 1e-2 and "
        "independent of the core index and the wavelength: largest 4.67e-3 at "
        "Delta 0.01 and V 1.5. Gaussian beams "
        "(waists 0.5a to 2.6a, displaced to 0.5a and tilted to half the acceptance "
        "angle along x and y, x-polarised) reach 3.3e-3. The bound is twice the "
        "largest per unit contrast, rounded up, so the allowance is scaled by "
        "Delta/1e-2; refining the quadrature changes the maximum by 2e-13"
    ),
    scaled_by="contrast",
)
"""The beam-into-fibre LP01 envelope; ``contrast`` is the fibre's largest
``|n^2 - n_ref^2|/(2 n_ref^2)`` against its core over the occupied bins, ``v_target``
its normalised frequency at the profile wavelength, ``modes`` the highest LP0m
radial order (LP01 only: V below 2.405 guides no other). The bound holds for any
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
    bound=3.0e-3,
    numerical=2e-5,
    reference=(
        "independent full-vector mode matching of the planar entrance facet "
        "(incident and reflected fields in the core-index input medium, the "
        "fibre's forward guided and radiation fields, tangential E and H "
        "continuous), solved per azimuthal order in a conducting cylinder of 30 "
        "core radii on its own 300 TE and 300 TM Bessel fields, against the "
        "model's flux pairing with the same incident H. The omitted share summed "
        "over an order's guided modes is a Hermitian form of the incident field: "
        "its largest eigenvalue modulus inside the widest admitted cone "
        "(1 - cos theta = 1/8) bounds every field of any shape and polarisation, "
        "and orders do not mix. Swept at contrast 0.05 (the range's top) over V "
        "1.5 to 8 with points 0.01 to 0.3 above every exact cutoff, each order's "
        "guided count held to the cutoffs: largest 1.371e-3 at V 3.905 (order "
        "one, HE12 just above its cutoff), 1.24e-3 just above the TE01/TM01 "
        "cutoff, 7.4e-4 at V 1.5. A guided mode wholly in the cladding approaches "
        "the normal-incidence Fresnel ceiling R(2 - R)/(1 - R) = 1.39e-3 (about "
        "Delta**2/2), which no resolved case exceeds; within 0.04 to 0.07 above "
        "the HE12 and HE13 cutoffs no cylinder up to 60a holds the newborn mode, "
        "so that band is bounded by the ceiling, not resolved. Bound twice the "
        "ceiling, numerical "
        "included; widening the cylinder to 45a and 60a changes the resolved "
        "values by at most 1.2e-5. It is second order in the index step "
        "(halving the contrast quarters it; 0.50 to 0.55 Delta**2 at the worst "
        "point for contrasts 0.0025 to 0.05), so the allowance is scaled by the "
        "squared contrast over 0.05**2"
    ),
    scaled_by="contrast_squared",
)
"""The vector model's step-index facet envelope; ``contrast`` is the guide's
``|n^2 - n_ref^2|/(2 n_ref^2)``, ``v`` its normalised frequency at the entrance and
``contrast_squared`` the square of the contrast, each weighted by the incident energy
over the occupied bins. It bounds the energy launched into all guided modes together,
not each channel's share, for any incident field inside the admitted cone. It covers
the reflection and backward fields at the facet, not the incident beam's H closure
(the angular item) or the guide's tabulation, and a beam in the core's index only,
since a bulk index change is a separate planar interface (SEP 31, section 4.3)."""
