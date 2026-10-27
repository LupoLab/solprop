# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The rounding-margin indicator of a noise-seeded mixed-precision run (SEP 36 §5).

A mixed-precision run evaluates its source in binary32, so every accepted step injects
rounding of relative size about ``c_s u32 log2(n)`` times the step's nonlinear phase
``phi = dz ||k_last|| / ||A||``. The rounding is amplified from where it is injected by
the same gain as the quantum seed, so the ratio of the two at a saved plane says how
far the seed stands above binary32's own noise. The indicator is a **heuristic**: it
omits the gain, which acts on both alike only while the dynamics are linear in the
seed, and the coherent bias of the coefficients' rounding.

Per accepted step the backend supplies ``phi`` in the norm its controller already uses
(the in-band L2 of the state, which on the multimode and radial routes is their energy
norm); rejected trials never reach the state and add nothing, and a first-same-as-last
evaluation counts once, with the step it belongs to. At each saved plane

``sigma_r = c_s u32 log2(n) sqrt(sum(phi^2) n_occ / n)``

is the accumulated rounding amplitude per occupied bin relative to the state's mean
occupied amplitude, and ``sigma_q = sqrt(sum_occ |N|^2 / sum_occ |A|^2)`` is the
seed's, with ``N`` the run's noise realisation and ``n_occ`` the bins whose spectral
energy is above -30 dB of the state's peak. The margin is ``20 log10(sigma_q /
sigma_r)`` in dB.

``c_s`` is calibrated by SEP 36's noise study; the record keeps ``sum(phi^2)`` and
``n_occ`` beside every value so that :func:`rounding_amplitude` recomputes the
indicator for another ``c_s`` without rerunning.
"""

import math
from collections.abc import Mapping

import numpy as np
from jaxtyping import Complex

__all__ = [
    "OCCUPIED_DB",
    "ROUNDING_MARGIN_CONSTANT",
    "RoundingMargin",
    "occupied_bins",
    "rounding_amplitude",
    "step_phase",
]

#: The default ``c_s``, pending the noise study's calibration (SEP 36 §7.3).
ROUNDING_MARGIN_CONSTANT = 1.0

#: A bin is occupied when its spectral energy is above this level of the peak, dB.
OCCUPIED_DB = -30.0

# The unit roundoff of binary32, 2**-24.
_U32 = float(np.finfo(np.float32).eps) / 2.0

_LABEL = (
    "heuristic: accumulated binary32 rounding of the source against the quantum "
    "seed, omitting gain and coherent bias (SEP 36 section 5)"
)


def occupied_bins(
    energy: Complex[np.ndarray, " n_omega"] | np.ndarray,
) -> np.ndarray:
    """Return the bins whose spectral energy is above -30 dB of the peak.

    Parameters
    ----------
    energy : numpy.ndarray
        Spectral energy per bin, nonnegative, shape ``(n_omega,)``.

    Returns
    -------
    numpy.ndarray
        Boolean mask, shape ``(n_omega,)``; all false for a zero spectrum.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.rounding_margin import occupied_bins
    >>> occupied_bins(np.array([1.0, 1e-2, 1e-4])).tolist()
    [True, True, False]
    """
    values = np.asarray(energy, dtype=np.float64)
    peak = float(values.max(initial=0.0))
    if peak <= 0.0:
        return np.zeros(values.shape, dtype=np.bool_)
    return values >= peak * 10.0 ** (OCCUPIED_DB / 10.0)


def rounding_amplitude(
    sum_phi_squared: float,
    n: int,
    n_occupied: int,
    c_s: float = ROUNDING_MARGIN_CONSTANT,
) -> float:
    """Return the heuristic accumulated rounding amplitude per occupied bin.

    ``sigma_r = c_s u32 log2(n) sqrt(sum(phi^2) n_occ / n)`` (SEP 36 §5).

    Parameters
    ----------
    sum_phi_squared : float
        The running sum of the accepted steps' squared nonlinear phases.
    n : int
        The number of spectral bins, positive.
    n_occupied : int
        The number of occupied bins.
    c_s : float, optional
        The calibration constant, 1 by default.

    Returns
    -------
    float
        ``sigma_r``, relative to the state's mean occupied amplitude.

    Examples
    --------
    >>> from solprop.backends.rounding_margin import rounding_amplitude
    >>> f"{rounding_amplitude(0.05**2 * 1e4, 2**14, 2**12):.1e}"
    '2.1e-06'
    """
    return c_s * _U32 * math.log2(n) * math.sqrt(sum_phi_squared * n_occupied / n)


def step_phase(
    state: Complex[np.ndarray, "*leading n_omega"],
    stage: Complex[np.ndarray, "*leading n_omega"],
    dz: float,
    mask: np.ndarray,
) -> float:
    """Return a step's nonlinear phase ``dz ||k_last|| / ||A||``, controller's norm.

    Parameters
    ----------
    state : numpy.ndarray
        The state at the step's anchor, last axis the spectrum.
    stage : numpy.ndarray
        The step's last stage value, same layout.
    dz : float
        The step's length, m.
    mask : numpy.ndarray
        The bins the controller's norm scores, Boolean, shape ``(n_omega,)``.

    Returns
    -------
    float
        The phase, dimensionless; ``nan`` for a zero state norm.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.rounding_margin import step_phase
    >>> state = np.ones((1, 4), dtype=complex)
    >>> step_phase(state, 2.0 * state, 0.1, np.ones(4, dtype=bool))
    0.2
    """
    size = float(np.linalg.norm(state[..., mask]))
    if size == 0.0:
        return math.nan
    return dz * float(np.linalg.norm(stage[..., mask])) / size


def _spectral_energy(values: np.ndarray) -> np.ndarray:
    """Return the energy per bin of a canonical array, summed over the leading axes.

    Parameters
    ----------
    values : numpy.ndarray
        Complex array, last axis the spectrum.

    Returns
    -------
    numpy.ndarray
        Real array, shape ``(n_omega,)``.
    """
    magnitude = np.abs(np.asarray(values)) ** 2
    return magnitude.reshape(-1, magnitude.shape[-1]).sum(axis=0)


class RoundingMargin:
    """Accumulate the indicator over a run and record it at each saved plane.

    Parameters
    ----------
    reason : str or None
        Why the run records no indicator values (no noise model, a march that does not
        supply its last stage), or ``None`` for an active indicator.
    c_s : float, optional
        The calibration constant.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.rounding_margin import RoundingMargin
    >>> margin = RoundingMargin(None)
    >>> margin.accept(1e-2)
    >>> state = np.ones((1, 1, 1, 8), dtype=complex)
    >>> margin.plane(0.1, state, 1e-3 * state)
    >>> record = margin.record()
    >>> record["planes"][0]["n_occupied"], round(record["planes"][0]["sigma_q"], 6)
    (8, 0.001)
    """

    def __init__(
        self, reason: str | None, c_s: float = ROUNDING_MARGIN_CONSTANT
    ) -> None:
        self.reason = reason
        self.c_s = float(c_s)
        self.sum_phi_squared = 0.0
        self.z: float | None = None
        self.unsupported: str | None = None
        self._planes: dict[float, dict[str, object]] = {}

    @property
    def active(self) -> bool:
        """Whether the run accumulates the indicator."""
        return self.reason is None and self.unsupported is None

    def reset(self, z: float) -> None:
        """Start afresh at a loaded state: a new run, or a reused runner.

        Parameters
        ----------
        z : float
            The position of the loaded state, m.
        """
        self.sum_phi_squared = 0.0
        self.z = z
        self.unsupported = None
        self._planes = {}

    def accept(self, phi: float | None, z: float | None = None) -> None:
        """Add one accepted step's nonlinear phase.

        Parameters
        ----------
        phi : float or None
            ``dz ||k_last|| / ||A||``, or ``None`` when the step's march does not
            supply it, which ends the indicator for the run with a stated reason. A
            zero state norm passes ``nan`` and adds nothing.
        z : float or None, optional
            The step's end, m.
        """
        if not self.active:
            return
        if phi is None:
            self.unsupported = (
                "the step's march does not supply its last stage value; only the "
                "Runge-Kutta march records the indicator"
            )
            return
        if math.isfinite(phi):
            self.sum_phi_squared += phi * phi
        if z is not None:
            self.z = z

    def restore(self, record: Mapping[str, object], z: float) -> bool:
        """Continue the running sum from a saved record at a resume position.

        Parameters
        ----------
        record : collections.abc.Mapping
            What :meth:`record` returned for the interrupted run.
        z : float
            The resume position, m.

        Returns
        -------
        bool
            Whether the record held the sum at `z`; otherwise the sum starts at zero
            and the record says so.
        """
        if not self.active:
            return False
        planes = record.get("planes", [])
        candidates: list[tuple[float, float]] = []
        if isinstance(planes, list):
            for entry in planes:
                if isinstance(entry, Mapping):
                    candidates.append(
                        (float(entry["z"]), float(entry["sum_phi_squared"]))  # pyright: ignore[reportArgumentType] - JSON scalars
                    )
        last_z, last_sum = record.get("z"), record.get("sum_phi_squared")
        if isinstance(last_z, float | int) and isinstance(last_sum, float | int):
            candidates.append((float(last_z), float(last_sum)))
        for position, value in candidates:
            if position == z:
                self.sum_phi_squared = value
                self.z = z
                return True
        self.unsupported = (
            "resumed from a record without the running sum at the resume position"
        )
        return False

    def plane(
        self,
        z: float,
        state: Complex[np.ndarray, "*leading n_omega"],
        seed: Complex[np.ndarray, "*leading n_omega"] | None,
    ) -> None:
        """Record the indicator at a saved plane.

        Parameters
        ----------
        z : float
            The plane's position, m.
        state : numpy.ndarray
            The state, canonical layout, last axis the spectrum.
        seed : numpy.ndarray or None
            The run's noise realisation in the same layout.
        """
        if not self.active:
            return
        energy = _spectral_energy(state)
        n = int(energy.shape[-1])
        entry: dict[str, object] = {
            "z": float(z),
            "sum_phi_squared": self.sum_phi_squared,
        }
        occupied = occupied_bins(energy)
        n_occupied = int(occupied.sum())
        if n_occupied == 0 or seed is None:
            entry |= {
                "n_occupied": n_occupied,
                "sigma_r": None,
                "sigma_q": None,
                "margin_db": None,
                "absent": "zero state norm" if n_occupied == 0 else "no noise seed",
            }
        else:
            sigma_r = rounding_amplitude(self.sum_phi_squared, n, n_occupied, self.c_s)
            seeded = float(_spectral_energy(seed)[occupied].sum())
            sigma_q = math.sqrt(seeded / float(energy[occupied].sum()))
            margin = (
                20.0 * math.log10(sigma_q / sigma_r)
                if sigma_r > 0.0 and sigma_q > 0.0
                else None
            )
            entry |= {
                "n_occupied": n_occupied,
                "sigma_r": sigma_r,
                "sigma_q": sigma_q,
                "margin_db": margin,
            }
        self._planes[float(z)] = entry
        self.z = float(z)

    def record(self) -> dict[str, object]:
        """Return the indicator for the run's metadata, JSON scalars and lists.

        Returns
        -------
        dict
            The label, ``c_s``, the running sum at the last accepted position and one
            entry per saved plane; or the label and why the values are absent.
        """
        label: dict[str, object] = {"label": _LABEL, "c_s": self.c_s}
        if self.reason is not None:
            return label | {"absent": self.reason}
        record = label | {
            "z": self.z,
            "sum_phi_squared": self.sum_phi_squared,
            "planes": [self._planes[key] for key in sorted(self._planes)],
        }
        if self.unsupported is not None:
            record["absent"] = self.unsupported
        return record
